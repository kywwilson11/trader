"""Learned sentiment lexicon — offline supervised lexicon induction.

What: terms are weighted by the realized forward returns that followed their
appearance in stock headlines/summaries (Jegadeesh & Wu 2013 "word power";
Ke, Kelly & Xiu 2019 SESTM screen-then-fit: univariate t-stat screen, then a
weighted ridge fit on the surviving binary term indicators). Stock book only:
crypto sentiment is the Fear & Greed index (fng_daily) — there are no crypto
headlines in the articles table, so no crypto lexicon is possible.

DARK BY CONSTRUCTION: this module writes learned_lexicon.json and
lexicon_eval_report.json; NOTHING in the live system consumes either artifact.
It becomes a live feature only via a later owner harvest+retrain decision on
the IC evidence in the report (CLAUDE.md gotcha #2). It does not touch the
hand-built keyword lexicon in sentiment.py — B13 research forbids EXPANDING
that lexicon, and this is consistent with it: a separate offline artifact
whose entire point is the honest out-of-sample IC verdict before anyone
trusts a learned wordlist ("measure before trusting").

REFIT CADENCE — why this is NOT an online perceptron: online per-headline
weight updates chase nonstationary news noise, cannot be purged-walk-forward
evaluated (there is no frozen artifact per period to score out-of-sample),
and make the learned weights unreproducible. Full refits from scratch on a
trailing window at harvest cadence (weekly), evaluated under purged expanding
walk-forward, are the evaluable version of "weights that refit over time".

PIT DISCIPLINE: publication granularity is the UTC day — sentiment_history
stores only a date string, no intraday timestamp — so forward returns enter
at the FIRST TRADING DAY STRICTLY AFTER the publication day (a same-day move
can never leak into a label). Overlapping-horizon labels get inline
1/concurrency average-uniqueness weights (deliberately self-contained — this
module does not import sample_weights). Term screening and lambda selection
happen inside each training fold only; validation rows are never seen by
either.

FAIL-OPEN CONTRACT: pure functions raise ValueError on malformed input; the
CLI (scripts/train_lexicon.py) wraps everything — an instrument failure never
blocks anything, because nothing depends on this module.
"""

import collections
import datetime
import json
import math
import os
import re
import sqlite3
import zlib

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HORIZONS = (1, 3, 5)     # trading days
MIN_DF = 10                      # min document frequency for a term
SCREEN_T = 2.0                   # |t| threshold for the SESTM-style screen
N_FOLDS = 4                      # purged expanding walk-forward folds
LAMBDA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_LAMBDA = 1.0             # fallback when CV cannot run (grid median)
BOOTSTRAP_B = 500                # block-bootstrap resamples for the IC CI
BLOCK_DAYS = 5                   # bootstrap block width (calendar days)
TOP_TERMS = 25                   # top/bottom terms surfaced in the report
# Mirror of novelty.py's values (novelty.py is the semantic source for the
# offline novelty mirror below — do not drift these without drifting it).
NOVELTY_WINDOW_DAYS = 7
SHINGLE_W = 3

# Offline-only stopword list: standard English function words + finance
# boilerplate. NOT model-facing — editing it changes only this research
# instrument, never a live feature.
STOPWORDS = frozenset([
    'a', 'about', 'above', 'after', 'again', 'all', 'also', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'but', 'by', 'can', 'could', 'did', 'do', 'does', 'down', 'for', 'from',
    'had', 'has', 'have', 'he', 'her', 'his', 'how', 'if', 'in', 'into',
    'is', 'it', 'its', 'just', 'may', 'more', 'most', 'no', 'nor', 'not',
    'of', 'on', 'or', 'other', 'our', 'out', 'over', 'own', 'she', 'so',
    'some', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there',
    'these', 'they', 'this', 'to', 'too', 'under', 'up', 'was', 'we', 'were',
    'what', 'when', 'where', 'which', 'while', 'who', 'will', 'with',
    'would', 'you', 'your',
    # finance boilerplate
    'inc', 'corp', 'ltd', 'co', 'says', 'said', 'announces', 'reports',
    'stock', 'stocks', 'shares', 'share', 'market', 'company', 'today',
    'week', 'year', 'quarter', 'reuters', 'zacks', 'motley', 'fool',
    'nyse', 'nasdaq',
])

# Leading letter kills pure numbers; min total length 2.
_TOKEN_RE = re.compile(r"[a-z][a-z0-9'-]{1,}")
# Summaries may carry HTML — same guard sentiment.py uses.
_HTML_TAG = re.compile(r'<[^>]+>')
# Word regex for the novelty shingle mirror (novelty._WORD_RE).
_NOV_WORD_RE = re.compile(r'[a-z0-9]+')

_ARTICLE_COLS = ['symbol', 'date', 'headline', 'summary',
                 'keyword_score', 'llm_score']

_VERDICT_NOTE = (
    "Recent literature expects weak/fast-decaying single-name lexicon "
    "effects; no_incremental_ic is an expected, valid outcome. Any candidate "
    "signal still requires owner harvest+retrain decision (gotcha #2)."
)


def _as_date(d):
    """Coerce str/Timestamp/datetime/date to datetime.date."""
    if isinstance(d, datetime.datetime):
        return d.date()
    if isinstance(d, datetime.date):
        return d
    return datetime.date.fromisoformat(str(d)[:10])


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text, blacklist=frozenset()):
    """Lowercase, strip HTML, keep letter-led tokens, drop stopwords and
    blacklisted words (the lowercased corpus symbol set — kills ticker
    self-mentions), then emit surviving unigrams + adjacent-pair bigrams
    over the SURVIVING sequence joined with '_'. Deterministic and
    order-preserving."""
    if not text:
        return []
    cleaned = _HTML_TAG.sub(' ', str(text).lower())
    toks = _TOKEN_RE.findall(cleaned)
    kept = [w for w in toks if w not in STOPWORDS and w not in blacklist]
    bigrams = [kept[i] + '_' + kept[i + 1] for i in range(len(kept) - 1)]
    return kept + bigrams


# ---------------------------------------------------------------------------
# Corpus loading (read-only)
# ---------------------------------------------------------------------------

def load_articles(db_path, start=None, end=None, symbols=None):
    """Read the articles table from sentiment_cache.db, READ-ONLY (uri
    mode=ro — the trainer can never write the live cache). Missing file /
    missing table / any sqlite error -> empty DataFrame with the right
    columns (fail-open, no raise)."""
    conn = None
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        q = ("SELECT symbol, date, headline, summary, keyword_score, "
             "llm_score FROM articles")
        conds, params = [], []
        if start is not None:
            conds.append("date >= ?")
            params.append(_as_date(start).isoformat())
        if end is not None:
            conds.append("date <= ?")
            params.append(_as_date(end).isoformat())
        if symbols:
            syms = sorted(set(str(s) for s in symbols))
            conds.append("symbol IN (%s)" % ",".join("?" * len(syms)))
            params.extend(syms)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY symbol, date, id"
        rows = conn.execute(q, params).fetchall()
        return pd.DataFrame(rows, columns=_ARTICLE_COLS)
    except (sqlite3.Error, OSError, ValueError):
        return pd.DataFrame(columns=_ARTICLE_COLS)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Offline novelty mirror (novelty.py semantics, no import — novelty.py has
# disk-store side effects and only holds a live trailing window)
# ---------------------------------------------------------------------------

def _shingles(text):
    """Exact mirror of novelty._shingles (crc32 3-word shingles, with the
    <3-words single-shingle fallback)."""
    words = _NOV_WORD_RE.findall(str(text).lower())
    if len(words) < SHINGLE_W:
        return {zlib.crc32(' '.join(words).encode())} if words else set()
    return {zlib.crc32(' '.join(words[i:i + SHINGLE_W]).encode())
            for i in range(len(words) - SHINGLE_W + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def offline_novelty(articles):
    """Per-article novelty in [0,1]: 1 - max Jaccard vs the SAME symbol's
    STRICTLY-EARLIER-DATE articles within NOVELTY_WINDOW_DAYS calendar days.

    Strictly-earlier-date only: same-day articles have no stored intra-day
    order, so comparing within the day would be order-dependent — not PIT.
    First article of a symbol -> 1.0. Empty shingles -> 1.0 (NEUTRAL — a
    deliberate divergence from novelty.headline_novelty's 0.0: that path
    gates trading, this path weights regression samples, and a garbage
    headline must not zero its doc's weight)."""
    n = len(articles)
    out = np.ones(n, dtype=float)
    if n == 0:
        return out
    if 'symbol' not in articles.columns or 'date' not in articles.columns \
            or 'headline' not in articles.columns:
        raise ValueError("offline_novelty needs symbol/date/headline columns")
    syms = articles['symbol'].to_numpy()
    dates = [_as_date(d) for d in articles['date']]
    heads = ['' if h is None else str(h)
             for h in articles['headline'].to_numpy()]
    by_sym = {}
    for i in range(n):
        by_sym.setdefault(syms[i], []).append(i)
    for _, idxs in by_sym.items():
        order = sorted(idxs, key=lambda i: dates[i])  # stable within a day
        history = []   # (date, shingles) with date strictly earlier
        pending = []   # current-day batch, committed on date change
        cur_date = None
        for i in order:
            d = dates[i]
            if cur_date is None or d != cur_date:
                history.extend(pending)
                pending = []
                cur_date = d
                # Prune entries the window check below could never match —
                # keeps this O(n * window) instead of O(n^2) per symbol.
                history = [(dj, shj) for dj, shj in history
                           if (d - dj).days <= NOVELTY_WINDOW_DAYS]
            sh = _shingles(heads[i])
            if not sh:
                out[i] = 1.0
                continue
            max_sim = 0.0
            for dj, shj in history:
                if 0 < (d - dj).days <= NOVELTY_WINDOW_DAYS and shj:
                    sim = _jaccard(sh, shj)
                    if sim > max_sim:
                        max_sim = sim
                        if max_sim > 0.999:
                            break
            out[i] = 1.0 - max_sim
            pending.append((d, sh))
    return out


# ---------------------------------------------------------------------------
# Docs (symbol-day bags)
# ---------------------------------------------------------------------------

def build_docs(articles, novelty=None):
    """Aggregate articles into (symbol, day) docs.

    Columns: symbol, date (datetime.date), tokens (collections.Counter over
    headline + ' . ' + summary, symbol-blacklisted), n_articles, kw_score
    (mean stored keyword_score — the static-lexicon baseline), llm_doc
    (mean non-null llm_score else NaN), llm_n, doc_novelty (mean per-article
    novelty; 1.0 when novelty is None). Indexed 0..n-1."""
    cols = ['symbol', 'date', 'tokens', 'n_articles', 'kw_score',
            'llm_doc', 'llm_n', 'doc_novelty']
    if articles is None or len(articles) == 0:
        return pd.DataFrame(columns=cols)
    art = articles.reset_index(drop=True)
    for c in ('symbol', 'date', 'headline', 'keyword_score'):
        if c not in art.columns:
            raise ValueError(f"build_docs: articles missing column {c!r}")
    if novelty is not None and len(novelty) != len(art):
        raise ValueError("build_docs: novelty length mismatch")
    nov = (np.asarray(novelty, dtype=float) if novelty is not None
           else np.ones(len(art)))
    blacklist = frozenset(str(s).lower() for s in art['symbol'].unique())
    has_summary = 'summary' in art.columns
    has_llm = 'llm_score' in art.columns
    agg = {}
    for i in range(len(art)):
        sym = art.at[i, 'symbol']
        d = _as_date(art.at[i, 'date'])
        head = art.at[i, 'headline']
        summ = art.at[i, 'summary'] if has_summary else ''
        text = f"{'' if head is None else head} . " \
               f"{'' if (summ is None or (isinstance(summ, float) and math.isnan(summ))) else summ}"
        key = (sym, d)
        rec = agg.get(key)
        if rec is None:
            rec = {'tokens': collections.Counter(), 'n': 0,
                   'kw': [], 'llm': [], 'nov': []}
            agg[key] = rec
        rec['tokens'].update(tokenize(text, blacklist))
        rec['n'] += 1
        kw = art.at[i, 'keyword_score']
        rec['kw'].append(float(kw) if pd.notna(kw) else 0.0)
        if has_llm:
            lv = art.at[i, 'llm_score']
            if pd.notna(lv):
                rec['llm'].append(float(lv))
        rec['nov'].append(float(nov[i]))
    rows = []
    for (sym, d) in sorted(agg.keys()):
        rec = agg[(sym, d)]
        rows.append({
            'symbol': sym, 'date': d, 'tokens': rec['tokens'],
            'n_articles': rec['n'],
            'kw_score': float(np.mean(rec['kw'])),
            'llm_doc': float(np.mean(rec['llm'])) if rec['llm'] else np.nan,
            'llm_n': len(rec['llm']),
            'doc_novelty': float(np.mean(rec['nov'])),
        })
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Forward returns (strict PIT: entry strictly after the publication day)
# ---------------------------------------------------------------------------

def attach_forward_returns(docs, prices, horizons):
    """Add per-horizon forward returns in trading-day index space.

    prices: tidy [symbol, date, close] + optional [open]. Convention:
    entry_pos = searchsorted(trading_dates, pub_date, side='right') — the
    FIRST trading day STRICTLY AFTER publication. Entry price = open[entry]
    when 'open' exists else close[entry]; exit price = close[entry + h - 1]
    in open-entry mode, close[entry + h] in close-entry mode (~h days of
    exposure either way). fwd_ret_h = exit/entry - 1. Out-of-range index or
    non-finite price -> NaN (caller drops and counts).

    Adds: entry_idx, entry_date, and per h: fwd_ret_{h}, exit_idx_{h},
    exit_date_{h}."""
    horizons = tuple(int(h) for h in horizons)
    if any(h < 1 for h in horizons):
        raise ValueError("horizons must be >= 1")
    docs = docs.copy().reset_index(drop=True)
    n = len(docs)
    entry_idx = np.full(n, -1, dtype=np.int64)
    entry_date = np.full(n, None, dtype=object)
    fwd = {h: np.full(n, np.nan) for h in horizons}
    exit_idx = {h: np.full(n, -1, dtype=np.int64) for h in horizons}
    exit_date = {h: np.full(n, None, dtype=object) for h in horizons}

    if n > 0 and prices is not None and len(prices) > 0:
        for c in ('symbol', 'date', 'close'):
            if c not in prices.columns:
                raise ValueError(f"prices missing column {c!r}")
        use_open = 'open' in prices.columns
        pgroups = {}
        for sym, grp in prices.groupby('symbol'):
            pdates = [_as_date(d) for d in grp['date']]
            if len(set(pdates)) != len(pdates):
                raise ValueError(f"duplicate price dates for {sym}")
            order = np.argsort(np.array(pdates, dtype='datetime64[D]'))
            D = [pdates[i] for i in order]
            aclose = grp['close'].to_numpy(dtype=float)[order]
            aopen = (grp['open'].to_numpy(dtype=float)[order]
                     if use_open else aclose)
            pgroups[sym] = (np.array(D, dtype='datetime64[D]'), D,
                            aopen, aclose)
        doc_dates = [_as_date(d) for d in docs['date']]
        for sym, pos in docs.groupby('symbol').indices.items():
            if sym not in pgroups:
                continue
            D64, D, aopen, aclose = pgroups[sym]
            pos = np.asarray(pos)
            pub64 = np.array([doc_dates[i] for i in pos],
                             dtype='datetime64[D]')
            ep = np.searchsorted(D64, pub64, side='right')
            ok_e = ep < len(D)
            epc = np.minimum(ep, len(D) - 1)
            entry_price = (aopen if use_open else aclose)[epc]
            entry_idx[pos[ok_e]] = ep[ok_e]
            for j in np.where(ok_e)[0]:
                entry_date[pos[j]] = D[ep[j]]
            for h in horizons:
                xp = ep + (h - 1 if use_open else h)
                ok = ok_e & (xp < len(D))
                xpc = np.minimum(xp, len(D) - 1)
                ex_price = aclose[xpc]
                good = ok & np.isfinite(entry_price) \
                    & np.isfinite(ex_price) & (entry_price > 0)
                f = np.where(good, ex_price / np.where(entry_price > 0,
                                                       entry_price, 1.0) - 1.0,
                             np.nan)
                fwd[h][pos[good]] = f[good]
                exit_idx[h][pos[good]] = xp[good]
                for j in np.where(good)[0]:
                    exit_date[h][pos[j]] = D[xp[j]]

    docs['entry_idx'] = entry_idx
    docs['entry_date'] = entry_date
    for h in horizons:
        docs[f'fwd_ret_{h}'] = fwd[h]
        docs[f'exit_idx_{h}'] = exit_idx[h]
        docs[f'exit_date_{h}'] = exit_date[h]
    return docs


# ---------------------------------------------------------------------------
# Sample weights (inline average uniqueness — deliberately self-contained)
# ---------------------------------------------------------------------------

def uniqueness_weights(docs, horizon):
    """Lopez de Prado average uniqueness, simple inline form: per symbol,
    sample i occupies trading-day interval [entry_idx, exit_idx_h];
    concurrency per day via a diff-array; weight_i = mean over the interval
    of 1/concurrency. NaN fwd_ret -> NaN weight; non-overlapping -> 1.0."""
    h = int(horizon)
    n = len(docs)
    w = np.full(n, np.nan)
    if n == 0:
        return w
    fwd = docs[f'fwd_ret_{h}'].to_numpy(dtype=float)
    ent = docs['entry_idx'].to_numpy()
    ext = docs[f'exit_idx_{h}'].to_numpy()
    for _, pos in docs.groupby('symbol').indices.items():
        pos = np.asarray(pos)
        ok = pos[np.isfinite(fwd[pos])]
        if ok.size == 0:
            continue
        starts = ent[ok].astype(np.int64)
        ends = ext[ok].astype(np.int64)
        if (starts < 0).any() or (ends < starts).any():
            raise ValueError("uniqueness_weights: bad entry/exit indices")
        L = int(ends.max()) + 2
        diff = np.zeros(L)
        np.add.at(diff, starts, 1.0)
        np.add.at(diff, ends + 1, -1.0)
        conc = np.cumsum(diff)
        for i, a, b in zip(ok, starts, ends):
            w[i] = float(np.mean(1.0 / conc[a:b + 1]))
    return w


def combine_weights(uniq, doc_novelty, extra=None):
    """Elementwise product of uniqueness, novelty and an optional extra
    column; clipped at 1e-6 and renormalized to mean 1 over finite
    entries. NaN stays NaN (excluded downstream)."""
    w = np.asarray(uniq, dtype=float) * np.asarray(doc_novelty, dtype=float)
    if extra is not None:
        w = w * np.asarray(extra, dtype=float)
    fin = np.isfinite(w)
    if fin.any():
        w[fin] = np.clip(w[fin], 1e-6, None)
        m = float(w[fin].mean())
        if m > 0:
            w = w / m
    return w


# ---------------------------------------------------------------------------
# Vocabulary + SESTM screen
# ---------------------------------------------------------------------------

def build_vocab(docs, min_df=MIN_DF):
    """Binary-presence document frequency; keep df >= min_df. Returns
    (sorted terms, postings: term -> np.array of doc positions). Postings
    only — no dense matrix before screening (Jetson 8GB discipline)."""
    df_counts = collections.Counter()
    for c in docs['tokens']:
        df_counts.update(set(c))
    terms = sorted(t for t, cnt in df_counts.items() if cnt >= int(min_df))
    termset = set(terms)
    lists = {t: [] for t in terms}
    for i, c in enumerate(docs['tokens']):
        for t in c.keys():
            if t in termset:
                lists[t].append(i)
    postings = {t: np.asarray(v, dtype=np.int64) for t, v in lists.items()}
    return terms, postings


def _weighted_stats(y, w, mask):
    ws = w[mask]
    ys = y[mask]
    sw = float(ws.sum())
    if sw <= 0:
        return 0.0, 0.0, 0.0
    neff = sw * sw / float(np.sum(ws * ws))
    mu = float(np.sum(ws * ys) / sw)
    var = float(np.sum(ws * (ys - mu) ** 2) / sw)
    if neff > 1.0:
        var *= neff / (neff - 1.0)
    return mu, var, neff


def screen_terms(postings, y, w, screen_t=SCREEN_T):
    """SESTM-style screen: weighted Welch t-stat of mean(y | term present)
    vs mean(y | absent) with reliability-weighted moments and effective
    n = (sum w)^2 / sum w^2 per side; require n_eff >= 5 on BOTH sides and
    |t| >= screen_t.

    LEAK CONTRACT (loud, per spec): y/w MUST already be restricted to the
    fold's training rows — postings must index rows of y/w, and screening
    NEVER sees validation rows. Postings referencing rows beyond len(y)
    raise ValueError (leak guard)."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(y) != len(w):
        raise ValueError("screen_terms: y/w length mismatch")
    n = len(y)
    base = np.isfinite(y) & np.isfinite(w) & (w > 0)
    screened, tstats = [], {}
    for t in sorted(postings):
        idx = np.asarray(postings[t])
        if idx.size and int(idx.max()) >= n:
            raise ValueError(
                "screen_terms: postings index beyond y — postings must be "
                "built on the same (training-only) rows as y/w")
        pres = np.zeros(n, dtype=bool)
        pres[idx] = True
        m1 = base & pres
        m0 = base & ~pres
        mu1, v1, n1 = _weighted_stats(y, w, m1)
        mu0, v0, n0 = _weighted_stats(y, w, m0)
        if n1 < 5 or n0 < 5:
            continue
        denom = math.sqrt(max(v1 / n1 + v0 / n0, 0.0))
        diff = mu1 - mu0
        if denom < 1e-12:
            t_val = 0.0 if abs(diff) < 1e-12 else math.copysign(1e9, diff)
        else:
            t_val = diff / denom
        tstats[t] = float(t_val)
        if abs(t_val) >= float(screen_t):
            screened.append(t)
    return screened, tstats


def build_X(docs, terms):
    """Binary-presence float32 matrix (n_docs x K). Built only AFTER
    screening, so K is small."""
    n = len(docs)
    X = np.zeros((n, len(terms)), dtype=np.float32)
    tpos = {t: k for k, t in enumerate(terms)}
    for i, c in enumerate(docs['tokens']):
        for t in c.keys():
            k = tpos.get(t)
            if k is not None:
                X[i, k] = 1.0
    return X


# ---------------------------------------------------------------------------
# Ridge (pure numpy — no sklearn, ever)
# ---------------------------------------------------------------------------

def fit_ridge(X, y, w, lam):
    """Weighted, centered closed-form ridge: solve
    (Xc' W Xc + lam * sum(w) * I) beta = Xc' W yc, intercept
    = ybar - xbar @ beta. np.linalg.solve with lstsq fallback."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if X.ndim != 2 or X.shape[0] != len(y) or len(y) != len(w):
        raise ValueError("fit_ridge: shape mismatch")
    k = X.shape[1]
    sw = float(w.sum())
    if sw <= 0:
        return np.zeros(k), 0.0
    if k == 0:
        return np.zeros(0), float(np.sum(w * y) / sw)
    xbar = (w @ X) / sw
    ybar = float(np.sum(w * y) / sw)
    Xc = X - xbar
    yc = y - ybar
    A = Xc.T @ (Xc * w[:, None]) + float(lam) * sw * np.eye(k)
    b = Xc.T @ (w * yc)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return beta, float(ybar - xbar @ beta)


# ---------------------------------------------------------------------------
# Purged expanding walk-forward
# ---------------------------------------------------------------------------

def purged_folds(docs, horizon, n_folds=N_FOLDS, embargo_days=None):
    """Expanding walk-forward with purge AND embargo. Sorted unique pub
    dates are split into n_folds+1 contiguous blocks; fold k (1..n_folds):
    val = docs with pub_date in block k; train = docs with
    exit_date < val_start AND pub_date <= val_start - embargo_days.
    embargo_days default = the horizon (>= max horizon in play is the
    caller's contract — train_lexicon passes max(horizons)). Only docs with
    finite fwd_ret for this horizon participate. Folds with train < 50 or
    val < 20 rows are skipped. Returns [(train_idx, val_idx)] positional."""
    h = int(horizon)
    if embargo_days is None:
        embargo_days = h
    if len(docs) == 0:
        return []
    fwd = docs[f'fwd_ret_{h}'].to_numpy(dtype=float)
    pub64 = np.array([_as_date(d) for d in docs['date']],
                     dtype='datetime64[D]')
    exit64 = np.array(
        [np.datetime64(e) if e is not None else np.datetime64('NaT')
         for e in docs[f'exit_date_{h}']], dtype='datetime64[D]')
    ok = np.isfinite(fwd) & ~np.isnat(exit64)
    udates = np.unique(pub64[ok])
    if udates.size < (n_folds + 1):
        return []
    blocks = np.array_split(udates, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        blk = blocks[k]
        if blk.size == 0:
            continue
        val_start, val_end = blk[0], blk[-1]
        emb_cut = val_start - np.timedelta64(int(embargo_days), 'D')
        va = np.where(ok & (pub64 >= val_start) & (pub64 <= val_end))[0]
        tr = np.where(ok & (exit64 < val_start) & (pub64 <= emb_cut))[0]
        if len(tr) >= 50 and len(va) >= 20:
            folds.append((tr, va))
    return folds


# ---------------------------------------------------------------------------
# Lambda selection (screen on TRAIN ONLY inside each fold)
# ---------------------------------------------------------------------------

def choose_lambda(docs, horizon, grid=LAMBDA_GRID, weights=None, folds=None,
                  min_df=MIN_DF, screen_t=SCREEN_T, n_folds=N_FOLDS,
                  embargo_days=None):
    """CV lambda: for each lam, over purged_folds, screen on TRAIN ONLY,
    fit on train, weighted MSE on val (val score = intercept + sum of beta
    over screened terms present). Returns (best_lam, cv_table). Falls back
    to DEFAULT_LAMBDA when no fold produces a score."""
    h = int(horizon)
    if weights is None:
        weights = np.ones(len(docs))
    weights = np.asarray(weights, dtype=float)
    if folds is None:
        folds = purged_folds(docs, h, n_folds=n_folds,
                             embargo_days=embargo_days)
    y = docs[f'fwd_ret_{h}'].to_numpy(dtype=float) if len(docs) else \
        np.zeros(0)
    prep = []
    for tr, va in folds:
        sub = docs.iloc[tr].reset_index(drop=True)
        _, postings = build_vocab(sub, min_df=min_df)
        scr, _ = screen_terms(postings, y[tr], weights[tr],
                              screen_t=screen_t)
        if not scr:
            continue
        Xtr = build_X(sub, scr)
        Xva = build_X(docs.iloc[va].reset_index(drop=True), scr)
        prep.append((Xtr, y[tr], weights[tr], Xva, y[va], weights[va]))
    cv_table = []
    best_lam, best_mse = None, np.inf
    for lam in grid:
        mses = []
        for Xtr, ytr, wtr, Xva, yva, wva in prep:
            beta, b0 = fit_ridge(Xtr, ytr, wtr, lam)
            pred = Xva @ beta + b0
            fin = np.isfinite(yva) & np.isfinite(wva) & (wva > 0)
            if fin.sum() == 0:
                continue
            mses.append(float(np.average((pred[fin] - yva[fin]) ** 2,
                                         weights=wva[fin])))
        mean_mse = float(np.mean(mses)) if mses else float('nan')
        cv_table.append({'lambda': float(lam), 'mse': mean_mse,
                         'n_folds': len(mses)})
        if mses and mean_mse < best_mse:
            best_mse, best_lam = mean_mse, float(lam)
    if best_lam is None:
        best_lam = DEFAULT_LAMBDA
    return best_lam, cv_table


# ---------------------------------------------------------------------------
# OOS evaluation
# ---------------------------------------------------------------------------

def _spearman_ic(a, b):
    """(spearman rho, n) on the pairwise-finite subset; NaN when degenerate.
    Lazy scipy import — defensive parity with ic_diagnostic.py style."""
    from scipy.stats import spearmanr
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    nm = int(m.sum())
    if nm < 3 or len(np.unique(a[m])) < 2 or len(np.unique(b[m])) < 2:
        return float('nan'), nm
    r = spearmanr(a[m], b[m]).correlation
    return (float(r) if np.isfinite(r) else float('nan')), nm


def _block_bootstrap_ci(scores, rets, dates, b=BOOTSTRAP_B,
                        block_days=BLOCK_DAYS, seed=0):
    """Percentile 2.5/97.5 CI on the Spearman IC via block bootstrap over
    contiguous block_days-day pub-date blocks (seeded default_rng)."""
    scores = np.asarray(scores, dtype=float)
    rets = np.asarray(rets, dtype=float)
    if len(scores) == 0:
        return None, None
    d0 = min(dates)
    bid = np.array([(d - d0).days // block_days for d in dates])
    ub = np.unique(bid)
    if ub.size < 2:
        return None, None
    idx_by = {u: np.where(bid == u)[0] for u in ub}
    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(int(b)):
        pick = rng.choice(ub, size=ub.size, replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        r, _ = _spearman_ic(scores[idx], rets[idx])
        if np.isfinite(r):
            ics.append(r)
    if len(ics) < 10:
        return None, None
    return (float(np.percentile(ics, 2.5)),
            float(np.percentile(ics, 97.5)))


def _j(x):
    """JSON-safe float: None when not finite."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def walkforward_eval(docs, horizons=DEFAULT_HORIZONS, weights_by_h=None,
                     journal_scores=None, min_df=MIN_DF, screen_t=SCREEN_T,
                     n_folds=N_FOLDS, lambda_grid=LAMBDA_GRID,
                     embargo_days=None, bootstrap_b=BOOTSTRAP_B, seed=0):
    """Purged expanding walk-forward OOS evaluation. Per horizon and fold:
    screen + choose_lambda + fit on TRAIN ONLY (lambda by inner CV inside
    the fold's train slice), score val, collect OOS rows. Then per horizon:
    learned IC vs the static-keyword baseline IC and the LLM baselines
    (b1 = per-article llm_score doc mean; b2 = llm_analysis journal daily
    mean when supplied), block-bootstrap CI, per-fold ICs, and a mechanical
    verdict."""
    horizons = tuple(int(h) for h in horizons)
    emb = int(embargo_days) if embargo_days is not None else max(horizons)
    if weights_by_h is None:
        weights_by_h = {}
    jmap = {}
    if journal_scores is not None and len(journal_scores) > 0:
        for r in journal_scores.itertuples(index=False):
            jmap[(r.symbol, _as_date(r.date))] = float(r.s_mean)
    out_h = {}
    total_folds = 0
    zero_screen_total = 0
    for h in horizons:
        w = weights_by_h.get(h)
        if w is None:
            w = np.ones(len(docs))
        w = np.asarray(w, dtype=float)
        y = docs[f'fwd_ret_{h}'].to_numpy(dtype=float) if len(docs) else \
            np.zeros(0)
        folds = purged_folds(docs, h, n_folds=n_folds, embargo_days=emb)
        oos = {'score': [], 'fwd': [], 'kw': [], 'llm': [], 'js': [],
               'pub': [], 'fold': []}
        per_fold = []
        zero_screen = 0
        for fold_no, (tr, va) in enumerate(folds, start=1):
            total_folds += 1
            sub = docs.iloc[tr].reset_index(drop=True)
            _, postings = build_vocab(sub, min_df=min_df)
            scr, _ = screen_terms(postings, y[tr], w[tr],
                                  screen_t=screen_t)
            if not scr:
                zero_screen += 1
                zero_screen_total += 1
                per_fold.append({'fold': fold_no, 'n': 0, 'ic': None,
                                 'n_terms': 0})
                continue
            # Lambda selection strictly inside this fold's train slice.
            lam, _ = choose_lambda(sub, h, grid=lambda_grid, weights=w[tr],
                                   min_df=min_df, screen_t=screen_t,
                                   n_folds=2, embargo_days=emb)
            Xtr = build_X(sub, scr)
            beta, b0 = fit_ridge(Xtr, y[tr], w[tr], lam)
            vsub = docs.iloc[va].reset_index(drop=True)
            Xva = build_X(vsub, scr)
            scores = Xva @ beta + b0
            f_ic, f_n = _spearman_ic(scores, y[va])
            per_fold.append({'fold': fold_no, 'n': int(f_n),
                             'ic': _j(f_ic), 'n_terms': len(scr)})
            for j in range(len(va)):
                sym = vsub.at[j, 'symbol']
                d = vsub.at[j, 'date']
                oos['score'].append(float(scores[j]))
                oos['fwd'].append(float(y[va][j]))
                oos['kw'].append(float(vsub.at[j, 'kw_score']))
                oos['llm'].append(float(vsub.at[j, 'llm_doc'])
                                  if pd.notna(vsub.at[j, 'llm_doc'])
                                  else float('nan'))
                oos['js'].append(jmap.get((sym, d), float('nan')))
                oos['pub'].append(d)
                oos['fold'].append(fold_no)
        sc = np.asarray(oos['score'], dtype=float)
        fr = np.asarray(oos['fwd'], dtype=float)
        ic_l, n_oos = _spearman_ic(sc, fr)
        ic_k, n_k = _spearman_ic(np.asarray(oos['kw'], dtype=float), fr)
        ic_b1, n_b1 = _spearman_ic(np.asarray(oos['llm'], dtype=float), fr)
        ic_b2, n_b2 = _spearman_ic(np.asarray(oos['js'], dtype=float), fr)
        if n_oos > 0:
            lo, hi = _block_bootstrap_ci(sc, fr, oos['pub'], b=bootstrap_b,
                                         block_days=BLOCK_DAYS, seed=seed)
        else:
            lo, hi = None, None
        out_h[str(h)] = {
            'n_oos': int(n_oos), 'n_folds_run': len(folds),
            'zero_screen_folds': zero_screen,
            'ic_learned': _j(ic_l), 'ic_ci': [_j(lo), _j(hi)],
            'ic_kw': _j(ic_k), 'n_kw': int(n_k),
            'ic_llm': _j(ic_b1), 'n_llm': int(n_b1),
            'ic_journal': _j(ic_b2), 'n_journal': int(n_b2),
            'per_fold': per_fold,
        }
    # Mechanical verdict — never stronger than review-required.
    all_small = all(v['n_oos'] < 200 for v in out_h.values()) if out_h \
        else True
    zero_majority = (total_folds == 0
                     or zero_screen_total > total_folds / 2.0)
    if all_small or zero_majority:
        verdict = 'insufficient_data'
    else:
        any_incremental = False
        for v in out_h.values():
            icl, ick = v['ic_learned'], v['ic_kw']
            lo, hi = v['ic_ci']
            if icl is None or lo is None or hi is None:
                continue
            ci_contains_zero = (lo <= 0.0 <= hi)
            beats_kw = icl > ((ick if ick is not None else 0.0) + 0.01)
            if not ci_contains_zero and beats_kw:
                any_incremental = True
        verdict = ('candidate_signal_review_required' if any_incremental
                   else 'no_incremental_ic')
    return {'horizons': out_h, 'verdict': verdict,
            'verdict_note': _VERDICT_NOTE}


# ---------------------------------------------------------------------------
# LLM journal baseline (b2) — pure stdlib JSONL scan, no repo imports
# ---------------------------------------------------------------------------

def load_llm_journal_scores(journal_dir, days=0):
    """Daily mean of llm_analysis journal `s` per (symbol, ts-date) over the
    last `days` days of journals/YYYY-MM-DD.jsonl. Mirrors llm_eval's row
    semantics without importing it (this module stays repo-import-free).
    s is [0,1] with 0.5 neutral — rank-based IC needs no rescaling. Missing
    dir / days<=0 / any error -> empty frame (fail-open)."""
    cols = ['symbol', 'date', 's_mean', 'n']
    try:
        days = int(days)
        if days <= 0 or not journal_dir or not os.path.isdir(journal_dir):
            return pd.DataFrame(columns=cols)
        acc = {}
        today = datetime.date.today()
        for k in range(days):
            d = today - datetime.timedelta(days=k)
            fp = os.path.join(str(journal_dir), f'{d.isoformat()}.jsonl')
            if not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    lines = f.readlines()
            except OSError:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue          # torn line — skip
                if not isinstance(row, dict) \
                        or row.get('action') != 'llm_analysis':
                    continue
                try:
                    rd = datetime.datetime.fromisoformat(
                        str(row.get('ts'))).date()
                except (TypeError, ValueError):
                    continue
                scores = row.get('scores')
                if not isinstance(scores, dict):
                    continue
                for sym, v in scores.items():
                    s = v.get('s') if isinstance(v, dict) else None
                    if s is None or not isinstance(s, (int, float)) \
                            or not math.isfinite(float(s)):
                        continue      # null s journaled on purpose (c26 D33)
                    acc.setdefault((sym, rd), []).append(float(s))
        rows = [{'symbol': sym, 'date': d,
                 's_mean': float(np.mean(vals)), 'n': len(vals)}
                for (sym, d), vals in sorted(acc.items())]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _empty_meta(horizons, fit_horizon, embargo, seed,
                min_df, screen_t, n_docs=0):
    return {
        'fit_window': None, 'lambda': None, 'screen_t': float(screen_t),
        'min_df': int(min_df), 'n_docs': int(n_docs),
        'n_terms_screened': 0, 'fit_horizon': int(fit_horizon),
        'horizons_evaluated': [int(h) for h in horizons],
        'embargo_days': int(embargo), 'seed': int(seed),
        'generated_at': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'consumers': 'NONE — dark artifact (gotcha #2)',
    }


def train_lexicon(articles, prices, horizons=DEFAULT_HORIZONS,
                  fit_horizon=3, min_df=MIN_DF, screen_t=SCREEN_T,
                  embargo_days=None, n_folds=N_FOLDS,
                  lambda_grid=LAMBDA_GRID, novelty=None, extra_weights=None,
                  journal_scores=None, bootstrap_b=BOOTSTRAP_B, seed=0):
    """Full offline run: docs -> forward returns -> weights -> purged
    walk-forward OOS eval FIRST, then the final artifact (screen +
    choose_lambda + ridge fit on the FULL window, fit_horizon only).
    Degenerate inputs return an empty artifact with
    verdict='insufficient_data' — never raise."""
    horizons = tuple(sorted(set(int(h) for h in horizons)
                            | {int(fit_horizon)}))
    fit_horizon = int(fit_horizon)
    emb = int(embargo_days) if embargo_days is not None else max(horizons)
    base_report = {
        'horizons': {}, 'verdict': 'insufficient_data',
        'verdict_note': _VERDICT_NOTE, 'top_terms': [],
        'dropped_rows': {},
        'sample_sizes': {'n_articles': 0, 'n_docs': 0, 'per_horizon_n': {}},
    }
    if articles is None or len(articles) == 0:
        return ({'terms': {}, 'meta': _empty_meta(horizons, fit_horizon,
                                                  emb, seed, min_df,
                                                  screen_t)},
                base_report)
    docs = build_docs(articles, novelty=novelty)
    if len(docs) == 0:
        return ({'terms': {}, 'meta': _empty_meta(horizons, fit_horizon,
                                                  emb, seed, min_df,
                                                  screen_t)},
                base_report)
    docs = attach_forward_returns(docs, prices, horizons)

    dropped = {}
    per_h_n = {}
    weights_by_h = {}
    nov_col = docs['doc_novelty'].to_numpy(dtype=float)
    extra = (np.asarray(extra_weights, dtype=float)
             if extra_weights is not None else None)
    for h in horizons:
        fwd = docs[f'fwd_ret_{h}'].to_numpy(dtype=float)
        dropped[str(h)] = int(np.sum(~np.isfinite(fwd)))
        per_h_n[str(h)] = int(np.sum(np.isfinite(fwd)))
        uniq = uniqueness_weights(docs, h)
        weights_by_h[h] = combine_weights(uniq, nov_col, extra)

    # EVAL FIRST — the honest verdict is the point of the whole exercise.
    report = walkforward_eval(
        docs, horizons=horizons, weights_by_h=weights_by_h,
        journal_scores=journal_scores, min_df=min_df, screen_t=screen_t,
        n_folds=n_folds, lambda_grid=lambda_grid, embargo_days=emb,
        bootstrap_b=bootstrap_b, seed=seed)
    report['dropped_rows'] = dropped
    report['sample_sizes'] = {
        'n_articles': int(len(articles)), 'n_docs': int(len(docs)),
        'per_horizon_n': per_h_n,
    }

    # FINAL ARTIFACT: full-window fit on fit_horizon only.
    h = fit_horizon
    y = docs[f'fwd_ret_{h}'].to_numpy(dtype=float)
    w = weights_by_h[h]
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    idx = np.where(mask)[0]
    terms_out, lam_used, tstats, top = {}, None, {}, []
    if idx.size >= 50:
        fit_docs = docs.iloc[idx].reset_index(drop=True)
        _, postings = build_vocab(fit_docs, min_df=min_df)
        scr, tstats = screen_terms(postings, y[idx], w[idx],
                                   screen_t=screen_t)
        if scr:
            lam_used, _ = choose_lambda(
                fit_docs, h, grid=lambda_grid, weights=w[idx],
                min_df=min_df, screen_t=screen_t, n_folds=n_folds,
                embargo_days=emb)
            X = build_X(fit_docs, scr)
            beta, _b0 = fit_ridge(X, y[idx], w[idx], lam_used)
            terms_out = {t: float(b) for t, b in zip(scr, beta)}
            ranked = sorted(terms_out.items(), key=lambda kv: -kv[1])
            picks = ranked[:TOP_TERMS] + (
                ranked[-TOP_TERMS:] if len(ranked) > TOP_TERMS else [])
            seen = set()
            for t, b in picks:
                if t in seen:
                    continue
                seen.add(t)
                top.append({'term': t, 'weight': float(b),
                            't_stat': _j(tstats.get(t))})
    report['top_terms'] = top

    fit_dates = [d.isoformat() for d in
                 (min(docs['date']), max(docs['date']))]
    meta = {
        'fit_window': fit_dates,
        'lambda': _j(lam_used),
        'screen_t': float(screen_t), 'min_df': int(min_df),
        'n_docs': int(len(docs)),
        'n_terms_screened': int(len(terms_out)),
        'fit_horizon': fit_horizon,
        'horizons_evaluated': [int(hh) for hh in horizons],
        'embargo_days': emb, 'seed': int(seed),
        'generated_at': datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        'consumers': 'NONE — dark artifact (gotcha #2)',
    }
    return {'terms': terms_out, 'meta': meta}, report


# ---------------------------------------------------------------------------
# Atomic artifact write (same pattern as novelty._save)
# ---------------------------------------------------------------------------

def write_json_atomic(path, obj):
    """Serialize (sorted keys, indent 2) to path+'.tmp' in the same
    directory, then os.replace — a crashed run never leaves a torn
    artifact."""
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        f.write(json.dumps(obj, sort_keys=True, indent=2, default=str))
    os.replace(tmp, str(path))
