"""Lexicon-fix impact instrument (G4-05 apostrophe + B13 phrase mask) — measurement-only.

Why: the owner decision KW_SCORER_V2 (W5 owner packet ITEM 1) asks whether two
correctness defects in sentiment._score_text move the stored Daily_Sentiment
feature enough to bundle the fix into the next re-harvest:
  G4-05  _PUNCT (sentiment.py:182) turns U+2019 into a space, so "isn’t"
         becomes "isn t": the negator test misses it and n_words is inflated.
  B13    phase 1 scores a phrase, then phase 2 scores the same words again
         (no masking), e.g. "rate cut" = +phrase, -"cut" -> 0.000.
This script is the rerunnable version of W5's scratch measurement. It never
edits sentiment.py and never writes the DB.

Variants (all scored in-process over the `articles` table):
  (a) a_current     the live sentiment._score_text, unchanged.
  (b) b_apostrophe  (a) on text with U+2019/U+2018 -> "'" (a wrapper around
                    the live function; identical to W5 lexfix.score(apos=True)).
  (c) c_apos_mask2  (b) + W5 "mask2": phase 2 drops every token overlapping a
                    SCORED phrase span, EXCEPT negator tokens inside the phrase,
                    and also drops the external negator that phase 1 consumed
                    for a negated phrase; n_words is kept (tanh scale
                    unchanged). A local copy of W5 lexfix.score2(apos=True).
The row score mirrors sentiment_history._keyword_score (sentiment_history.py
:106-128: headline 0.6 / summary 0.4, a lone field at full weight). The
ticker-day cell mirrors sentiment_history._aggregate_daily (:135-171): all-LLM
-> LLM mean; some LLM -> 0.7 LLM mean + 0.3 keyword mean; none -> keyword mean.
Cells are (symbol, date) with date >= max(articles.date) - --days.

PRE-REGISTERED RULE (SCOUT_A E3, verbatim):
  If (b) or (c) changes the ticker-day aggregate by >0.05 on >=1% of non-zero
  ticker-days, then bundle the fix (default-OFF flag + byte-compat test) into
  the H-audit re-harvest (already a gotcha-#2 event). IC deltas are reported,
  not gating. Otherwise, log it as a correctness-only owner item.
Operationalised here: a "non-zero ticker-day" is a cell whose variant-(a)
aggregate is != 0; the fraction is (cells among those with |delta| > 0.05) /
(non-zero cells). Verdict line: "BUNDLE" iff (b) or (c) reaches 0.01, else
"correctness-only". The all-cells fraction (W5's denominator) is printed too.
(The PIT IC column of E3 is not computed here: reported-not-gating, and it
needs price data this script does not read.)

Also scores the tests/test_sentiment_headlines.py scoring cases (imported
read-only) through each variant with the runner's own pass rule. The runner's
_validate_text cases are scorer-independent and are run once; its 4
_score_articles cases are NOT run (they can reach the LLM transport) and are
counted as passing, as in the live runner, in the "runner-equivalent" total.

The DB is opened READ-ONLY (sqlite `file:...?mode=ro` URI). No network, no LLM.

    python scripts/lexicon_fix_impact.py
    python scripts/lexicon_fix_impact.py --db other.db --days 365 --json out.json
Exit: 0 ok; 2 DB missing/unreadable or without an `articles` table.
"""
import argparse
import datetime as dt
import importlib.util
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import sentiment as S  # noqa: E402  (lexicon tables + live scorer; read-only)

DEFAULT_DB = REPO / 'sentiment_cache.db'
RUNNER_PATH = REPO / 'tests' / 'test_sentiment_headlines.py'
VARIANTS = ('a_current', 'b_apostrophe', 'c_apos_mask2')
RULE_DELTA = 0.05          # pre-registered |delta aggregate|
RULE_MIN_FRAC = 0.01       # pre-registered share of non-zero ticker-days
N_SCORE_ARTICLES_CASES = 4  # runner cases not run here (see docstring)

_APOS = str.maketrans({'’': "'", '‘': "'"})
_TOK = re.compile(r'\S+')


# --------------------------------------------------------------------------
# Scorer variants
# --------------------------------------------------------------------------

def score_a(text):
    """(a) the live scorer."""
    return S._score_text(text)


def score_b(text):
    """(b) apostrophe normalisation, then the live scorer."""
    return S._score_text(text.translate(_APOS))


def _isneg(w):
    return w in S._NEGATORS or w.endswith("n't")


def score_mask2(text, apos=True, mask=True):
    """Local copy of W5 lexfix.score2 (the 'mask2' rule). mask=False disables
    the masking only, which makes it a copy of the live _score_text (pinned
    by tests). Offsets align because _PUNCT.sub replaces 1 char with 1."""
    if apos:
        text = text.translate(_APOS)
    text_lower = text.lower()
    raw_score = 0.0
    pspans, nspans = [], []
    for table in (S._POS_PHRASE_RES, S._NEG_PHRASE_RES):
        for phrase, pat, weight in table:
            if phrase not in text_lower:
                continue
            m = pat.search(text_lower)
            if m:
                lo = max(0, m.start() - 15)
                prefix = text_lower[lo:m.start()]
                if S._NEG_PREFIX.search(prefix):
                    raw_score -= weight * 0.7
                    for nm in S._NEG_PREFIX.finditer(prefix):
                        nspans.append((lo + nm.start(),
                                       lo + nm.start() + len(nm.group().rstrip())))
                else:
                    raw_score += weight
                pspans.append((m.start(), m.end()))
    clean = S._PUNCT.sub(' ', text_lower)
    toks = list(_TOK.finditer(clean))
    words = [t.group() for t in toks]
    masked = set()
    if mask:
        for i, t in enumerate(toks):
            a, b = t.start(), t.end()
            if any(a < e and s < b for s, e in nspans) or \
               (not _isneg(words[i]) and any(a < e and s < b for s, e in pspans)):
                masked.add(i)
    word_scores, negator_positions = [], []
    for i, word in enumerate(words):
        if i in masked:
            continue
        if _isneg(word):
            negator_positions.append(i)
        elif word in S._POSITIVE:
            word_scores.append((i, 1.0))
        elif word in S._NEGATIVE:
            word_scores.append((i, -1.0))
    used = set()
    for idx, base in word_scores:
        for ni in negator_positions:
            if ni in used:
                continue
            if 0 < abs(idx - ni) <= 3:
                lo, hi = min(idx, ni), max(idx, ni)
                filler = sum(1 for j in range(lo + 1, hi)
                             if words[j] not in S._POSITIVE and words[j] not in S._NEGATIVE
                             and words[j] not in S._NEGATORS)
                if filler <= 2:
                    raw_score -= base * 0.7
                    raw_score -= base * 1.0
                    used.add(ni)
                    break
        else:
            raw_score += base
    n_words = max(len(words), 1)
    return math.tanh(raw_score * (0.4 / math.sqrt(n_words / 10)))


def score_c(text):
    """(c) apostrophe normalisation + mask2."""
    return score_mask2(text, apos=True, mask=True)


SCORERS = {'a_current': score_a, 'b_apostrophe': score_b, 'c_apos_mask2': score_c}


def keyword_score(headline, summary, fn):
    """Mirror of sentiment_history._keyword_score (:106-128) with scorer fn."""
    parts = []
    for t in (headline, summary):
        v = S._validate_text(t)
        if v:
            parts.append(fn(v))
    if not parts:
        return 0.0
    return parts[0] if len(parts) == 1 else parts[0] * 0.6 + parts[1] * 0.4


def aggregate_cell(kw_scores, llm_scores):
    """Mirror of sentiment_history._aggregate_daily (:135-171) score only.
    llm_scores holds one entry per article (None = unscored)."""
    n = len(kw_scores)
    ls = [x for x in llm_scores if x is not None]
    if len(ls) == n:
        return sum(ls) / len(ls)
    if ls:
        return (sum(ls) / len(ls)) * 0.7 + (sum(kw_scores) / n) * 0.3
    return sum(kw_scores) / n


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _pct(sorted_vals, q):
    """Linear-interpolated percentile (numpy's default method)."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    f = math.floor(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def row_stats(base, var):
    n = len(base)
    deltas = [v - b for b, v in zip(base, var)]
    changed = sorted(abs(d) for d in deltas if abs(d) > 1e-12)
    return {
        'n_rows': n,
        'n_changed': len(changed),
        'pct_changed': round(100.0 * len(changed) / n, 2) if n else 0.0,
        'sign_flips': sum(1 for b, v in zip(base, var) if b * v < 0),
        'mean_delta': (sum(deltas) / n) if n else 0.0,
        'p50_abs_changed': _pct(changed, 0.5),
        'p90_abs_changed': _pct(changed, 0.9),
        'max_abs': changed[-1] if changed else 0.0,
        'n_gt_0.05': sum(1 for d in deltas if abs(d) > 0.05),
        'n_gt_0.2': sum(1 for d in deltas if abs(d) > 0.2),
    }


def daily_stats(base_cells, var_cells):
    n = len(base_cells)
    d = [abs(v - b) for b, v in zip(base_cells, var_cells)]
    nz = [i for i, b in enumerate(base_cells) if b != 0]
    nnz = len(nz)
    out = {
        'n_cells': n, 'n_nonzero': nnz,
        'n_gt_0.05': sum(1 for x in d if x > 0.05),
        'n_gt_0.2': sum(1 for x in d if x > 0.2),
        'nonzero_n_gt_0.05': sum(1 for i in nz if d[i] > 0.05),
        'nonzero_n_gt_0.2': sum(1 for i in nz if d[i] > 0.2),
        'sign_flips': sum(1 for b, v in zip(base_cells, var_cells) if b * v < 0),
        'max_abs': max(d) if d else 0.0,
    }
    out['frac_gt_0.05'] = out['n_gt_0.05'] / n if n else 0.0
    out['frac_gt_0.2'] = out['n_gt_0.2'] / n if n else 0.0
    out['nonzero_frac_gt_0.05'] = out['nonzero_n_gt_0.05'] / nnz if nnz else 0.0
    out['nonzero_frac_gt_0.2'] = out['nonzero_n_gt_0.2'] / nnz if nnz else 0.0
    return out


def verdict(daily):
    """Pre-registered E3 rule over daily_stats for b_apostrophe / c_apos_mask2."""
    hit = [k for k in ('b_apostrophe', 'c_apos_mask2')
           if k in daily and daily[k]['nonzero_frac_gt_0.05'] >= RULE_MIN_FRAC]
    return ('BUNDLE' if hit else 'correctness-only'), hit


# --------------------------------------------------------------------------
# Runner cases
# --------------------------------------------------------------------------

def _runner_ok(score, expected):
    if expected == 'pos':
        return score > 0.05
    if expected == 'neg':
        return score < -0.05
    if expected == 'neutral':
        return abs(score) <= 0.15
    return expected == 'mixed'


def runner_scores(path=RUNNER_PATH):
    spec = importlib.util.spec_from_file_location('_tsh_cases', str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = mod.ALL_TESTS
    out = {}
    for name, fn in SCORERS.items():
        res = [_runner_ok(fn(h), e) for h, e, _c in cases]
        out[name] = {'pass': sum(res), 'total': len(res)}
    vp, vf = mod.run_validation_tests()
    out['validate_text'] = {'pass': vp, 'total': vp + vf}
    for name in SCORERS:
        r = out[name]
        r['runner_equiv_pass'] = r['pass'] + vp + N_SCORE_ARTICLES_CASES
        r['runner_equiv_total'] = r['total'] + vp + vf + N_SCORE_ARTICLES_CASES
    return out


# --------------------------------------------------------------------------
# DB pass
# --------------------------------------------------------------------------

def open_ro(db_path):
    p = Path(db_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f'no such database: {p}')
    con = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)
    con.execute('SELECT 1 FROM articles LIMIT 1').fetchall()
    return con


def measure(con, days):
    maxd = con.execute('SELECT MAX(date) FROM articles').fetchone()[0]
    cutoff = None
    if maxd:
        cutoff = (dt.date.fromisoformat(maxd[:10]) - dt.timedelta(days=days)).isoformat()
    scores = {k: [] for k in VARIANTS}
    stored_drift = 0
    cells = {}
    cur = con.execute("SELECT symbol, date, headline, COALESCE(summary, ''), "
                      "keyword_score, llm_score FROM articles")
    for i, (sym, date, head, summ, stored, llm) in enumerate(cur):
        a = keyword_score(head, summ, score_a)
        text = (head or '') + (summ or '')
        b = keyword_score(head, summ, score_b) if ('’' in text or '‘' in text) else a
        c = keyword_score(head, summ, score_c)
        scores['a_current'].append(a)
        scores['b_apostrophe'].append(b)
        scores['c_apos_mask2'].append(c)
        if stored is None or abs(float(stored) - a) > 1e-9:
            stored_drift += 1
        if cutoff is not None and date >= cutoff:
            cells.setdefault((sym, date), []).append((i, llm))
    base = scores['a_current']
    rows = {k: row_stats(base, scores[k]) for k in VARIANTS[1:]}
    keys = sorted(cells)

    def cell_vals(vs):
        return [aggregate_cell([vs[i] for i, _ in cells[k]],
                               [l for _, l in cells[k]]) for k in keys]
    base_cells = cell_vals(base)
    daily = {k: daily_stats(base_cells, cell_vals(scores[k])) for k in VARIANTS[1:]}
    return {'n_articles': len(base), 'max_date': maxd, 'cutoff': cutoff, 'days': days,
            'stored_keyword_score_ne_current': stored_drift,
            'rows': rows, 'daily': daily}


def _print(res):
    print(f"articles: {res['n_articles']:,}   window: date >= {res['cutoff']} "
          f"(--days {res['days']}, max date {res['max_date']})")
    print(f"stored keyword_score != current recompute: "
          f"{res['stored_keyword_score_ne_current']:,} rows")
    print('\nROWS (vs a_current)')
    print(f"{'variant':14s} {'changed':>15s} {'flips':>6s} {'mean d':>10s} "
          f"{'p50|d|':>7s} {'p90|d|':>7s} {'max|d|':>7s} {'>0.05':>6s} {'>0.2':>6s}")
    for k, r in res['rows'].items():
        print(f"{k:14s} {r['n_changed']:>7,} ({r['pct_changed']:>5.2f}%) {r['sign_flips']:>6,} "
              f"{r['mean_delta']:>+10.6f} {r['p50_abs_changed']:>7.4f} "
              f"{r['p90_abs_changed']:>7.4f} {r['max_abs']:>7.4f} "
              f"{r['n_gt_0.05']:>6,} {r['n_gt_0.2']:>6,}")
    print('\nTICKER-DAY CELLS (sentiment_history._aggregate_daily mirror)')
    print(f"{'variant':14s} {'cells':>6s} {'nonzero':>7s} {'nz>0.05':>8s} {'nz>0.2':>7s} "
          f"{'all>0.05':>8s} {'all>0.2':>7s} {'flips':>6s} {'max|d|':>7s}")
    for k, d in res['daily'].items():
        print(f"{k:14s} {d['n_cells']:>6,} {d['n_nonzero']:>7,} "
              f"{100 * d['nonzero_frac_gt_0.05']:>7.2f}% {100 * d['nonzero_frac_gt_0.2']:>6.2f}% "
              f"{100 * d['frac_gt_0.05']:>7.2f}% {100 * d['frac_gt_0.2']:>6.2f}% "
              f"{d['sign_flips']:>6,} {d['max_abs']:>7.4f}")
    if 'runner' in res:
        r = res['runner']
        print(f"\nRUNNER CASES (tests/test_sentiment_headlines.py; _validate_text "
              f"{r['validate_text']['pass']}/{r['validate_text']['total']})")
        for k in VARIANTS:
            x = r[k]
            print(f"{k:14s} scoring {x['pass']}/{x['total']}   runner-equivalent "
                  f"{x['runner_equiv_pass']}/{x['runner_equiv_total']}")
    print(f"\nVERDICT: {res['verdict']}"
          + (f"  (rule met by: {', '.join(res['verdict_by'])})" if res['verdict_by'] else '')
          + f"  [rule: (b) or (c) moves >{RULE_DELTA} on >={RULE_MIN_FRAC:.0%} "
            f"of non-zero ticker-days]")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Lexicon-fix impact (G4-05 + B13), read-only')
    ap.add_argument('--db', default=str(DEFAULT_DB),
                    help='sentiment cache sqlite (opened read-only; default repo root)')
    ap.add_argument('--days', type=int, default=365,
                    help='ticker-day window: date >= max(articles.date) - DAYS (default 365)')
    ap.add_argument('--json', metavar='PATH', default=None, help='also write the result as JSON')
    ap.add_argument('--skip-runner', action='store_true',
                    help='do not score the test_sentiment_headlines.py cases')
    args = ap.parse_args(argv)
    t0 = time.time()
    try:
        con = open_ro(args.db)
    except (OSError, sqlite3.Error) as e:
        print(f'ERROR: cannot read articles from {args.db}: {e}', file=sys.stderr)
        return 2
    try:
        res = measure(con, max(0, int(args.days)))
    finally:
        con.close()
    res['db'] = str(Path(args.db).resolve())
    if not args.skip_runner:
        res['runner'] = runner_scores()
    res['verdict'], res['verdict_by'] = verdict(res['daily'])
    res['wall_s'] = round(time.time() - t0, 1)
    try:
        import resource
        res['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except (ImportError, AttributeError):
        pass
    _print(res)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))
        print(f'JSON: {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
