"""LGB lag-subset A/B (SCOUT-2 design D1, 2026-09-27) — MEASUREMENT-ONLY.

Question: does the LightGBM leg lose anything if it sees only a sparse set of
window lags L (default {0,1,2,3,5,8,12,23}) instead of the full seq_len x F
flatten — and does spending the freed bytes on MORE training rows help?
Design: research/campaign_2026-09_jetson/research_signal.md, section
"2026-09-27 R2 scout brief (SCOUT-2)" -> "D1 — LGB lag-subset A/B".

Arms (features, LGB params, horizon identical; mean leg only):
  A  legacy full flatten at the legacy row cap
     min(LGB_MAX_ROWS, max(20k, LGB_X_BYTE_BUDGET // (seq*F*4))) — constants
     imported from scripts/hypersearch_v2.py (:1322/:1330, cap math :1395-1403)
  B  lag subset L at A's rows            (lag effect)
  C  lag subset L on ALL fold-train rows (row effect, B -> C)

Per fold (hypersearch_v2.get_walk_forward_folds(..., purge_val_labels=True),
the last --folds of the production NUM_FOLDS geometry): a RobustScaler fit on
the fold-train rows (the _ScaledCache.get rule, hypersearch_v2.py:767-795);
the validation window is split at its median time — the FIRST half (rows
whose max-horizon label completes before the split) is the early-stopping
set, the SECOND half (after a seq_len-bar embargo, objective_utils.
embargo_end_time) is the scored set. Scored halves are pooled across folds.

Statistic: IC = Spearman(pred, target) on the pooled scored rows; target =
all_returns_by_fb[fb] (Target_Return_{fb}, the column train_lgb_ensemble
uses, hypersearch_v2.py:1373-1379 / load_data :268-272). Paired
dIC = IC(arm) - IC(A). Weekly-block bootstrap (calendar weeks resampled
jointly across tickers, B draws) of dIC on full-sample ranks; one-sided 90 %
lower bound. Seed check: --extra-seeds extra A fits; SD_seed = sd of A's IC.
Stock also reports per-bar cross-sectional rank IC (objective_utils.cs_rank_ic).

Verdict (pre-registered D1 rule, per book; `verdict()` below):
  KILL     any arm's 90 % lower bound < -0.02, or peak RSS(C) > RSS(A);
           with --pair-with <other book json>: point dIC(C) < 0 on BOTH books,
           or SD_seed > |dIC(C)| on BOTH books.
  ADOPT-C  C: point dIC >= 0 and lower bound >= -0.01 and RSS(C) <= RSS(A)
           and wall(C) <= wall(A); else ADOPT-B on the same test for B.
  HOLD     otherwise ("no harm detected, underpowered" when point >= 0 but
           the bound < -0.01).
  Conditions (ii) holdout blend DSR and (iv) q10 floor coverage are NOT
  measured here — an ADOPT verdict is necessary, not sufficient, for any
  future LGB_LAG_SUBSET flip (challenger -> shadow DM-HLN still decides).

Writes nothing into the repo root: output JSON goes to --json <path> or
<--out dir>/lgb_lag_ab_<book>_<UTC stamp>.json (default dir logs/lgb_lag_ab/,
gitignored via logs/). --dry-run loads NO data (parquet footer only), prints
the plan + per-arm byte arithmetic and writes nothing. Exit code is 0 on every
measurement run; failures are reported, never raised past main().

Jetson: run through the campaign hwlock, CUDA_VISIBLE_DEVICES='', one process,
LightGBM num_threads=1 (--threads). Idle box only for the full run (arm A
peaks at ~X_train 600 MB + X_val + bins; see the dry-run arithmetic).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gc
import json
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np

DEFAULT_LAGS = (0, 1, 2, 3, 5, 8, 12, 23)
WEEK_SECONDS = 7 * 86400
VAL_CAP = 30_000          # stop-set cap, = hypersearch_v2.py:1404-1405
NI_BOUND = -0.01          # non-inferiority margin, D1 rule (i)
KILL_BOUND = -0.02        # D1 kill: any lower bound below this


# ---------------------------------------------------------------------------
# Pure helpers (no torch / lightgbm / data) — unit-tested
# ---------------------------------------------------------------------------

def lag_column_index(seq_len, n_features, lags):
    """Column indices of `lags` inside model_lgb.flatten_sequence's layout.

    flatten_sequence (model_lgb.py:16-40) flattens a (seq_len, F) window
    row-major: position t (t=0 oldest) occupies columns t*F .. t*F+F-1 and
    carries lag = seq_len-1-t. Returned columns are sorted ascending, i.e.
    oldest-first with features inner — the same relative order the full
    flatten has, so X_full[:, idx] is the lag-subset matrix. This is the ONE
    mapping a future LGB_LAG_SUBSET flag must reuse byte-for-byte.
    """
    seq_len = int(seq_len)
    n_features = int(n_features)
    lags = sorted({int(l) for l in lags})
    if not lags:
        raise ValueError("lags must be non-empty")
    if lags[0] < 0 or lags[-1] >= seq_len:
        raise AssertionError(
            f"lags must satisfy 0 <= lag < seq_len={seq_len}: {lags}")
    ts = sorted(seq_len - 1 - l for l in lags)
    return np.concatenate([np.arange(t * n_features, (t + 1) * n_features)
                           for t in ts]).astype(np.int64)


def lag_offsets(seq_len, lags):
    """gather_windows offsets for the lag subset, oldest-first — building
    windows with these offsets and reshaping equals X_full[:, lag_column_index]
    (the full window uses offsets np.arange(-seq_len, 0))."""
    seq_len = int(seq_len)
    ts = sorted(seq_len - 1 - int(l) for l in set(lags))
    return np.arange(-seq_len, 0)[ts]


def legacy_row_cap(seq_len, n_features, max_rows_const, byte_budget):
    """hypersearch_v2.py:1395-1397 cap math (constants passed in)."""
    row_bytes = int(seq_len) * int(n_features) * 4
    return int(min(max_rows_const, max(20_000, byte_budget // max(row_bytes, 1))))


def average_ranks(x):
    """Tie-averaged ranks (float64), 0-based."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    if len(x) == 0:
        return ranks
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(x)]
    avg = (starts + ends - 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float('nan')
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def _block_stats(rx, ry, block_ids):
    """Per-block sufficient statistics for Pearson(rx, ry)."""
    _, inv = np.unique(block_ids, return_inverse=True)
    nb = int(inv.max()) + 1 if len(inv) else 0
    def s(v):
        return np.bincount(inv, weights=v, minlength=nb)
    return np.stack([np.bincount(inv, minlength=nb).astype(np.float64),
                     s(rx), s(ry), s(rx * rx), s(ry * ry), s(rx * ry)])


def _pearson_from_stats(S):
    """S: (6, n_draws) summed stats -> Pearson per draw."""
    n, sx, sy, sxx, syy, sxy = S
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    with np.errstate(invalid='ignore', divide='ignore'):
        return cov / np.sqrt(vx * vy)


def paired_block_bootstrap(pred_arm, pred_base, y, block_ids, n_boot=2000,
                           seed=0, alpha=0.10):
    """Paired weekly-block bootstrap of dIC = Spearman(arm,y) - Spearman(base,y).

    Ranks are computed ONCE on the full pooled sample (the point estimate is
    then the exact Spearman); each draw resamples whole blocks with
    replacement and recomputes the Pearson of those fixed ranks from per-block
    sufficient statistics (standard rank-bootstrap approximation — no
    re-ranking per draw). Returns {'point', 'lo_one_sided', 'ci90', 'sd',
    'n_blocks', 'n'} with lo_one_sided the alpha-quantile (one-sided
    (1-alpha) lower bound).
    """
    pa = np.asarray(pred_arm, np.float64)
    pb = np.asarray(pred_base, np.float64)
    yy = np.asarray(y, np.float64)
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(yy)
    pa, pb, yy = pa[ok], pb[ok], yy[ok]
    blocks = np.asarray(block_ids)[ok]
    ry = average_ranks(yy)
    ra, rb = average_ranks(pa), average_ranks(pb)
    point = spearman(pa, yy) - spearman(pb, yy)
    Sa = _block_stats(ra, ry, blocks)
    Sb = _block_stats(rb, ry, blocks)
    nb = Sa.shape[1]
    out = {'point': float(point), 'n': int(len(yy)), 'n_blocks': int(nb),
           'n_boot': int(n_boot)}
    if nb < 2 or n_boot < 1:
        out.update(lo_one_sided=float('nan'), ci90=[float('nan')] * 2,
                   sd=float('nan'))
        return out
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    chunk = 250
    for i in range(0, n_boot, chunk):
        k = min(chunk, n_boot - i)
        counts = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=k).T  # (nb,k)
        draws[i:i + k] = (_pearson_from_stats(Sa @ counts)
                          - _pearson_from_stats(Sb @ counts))
    draws = draws[np.isfinite(draws)]
    out['lo_one_sided'] = float(np.quantile(draws, alpha)) if draws.size else float('nan')
    out['ci90'] = ([float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))]
                   if draws.size else [float('nan')] * 2)
    out['sd'] = float(draws.std(ddof=1)) if draws.size > 1 else float('nan')
    return out


def _arm_passes(d):
    """D1 rule (i)+(iii) for one arm summary dict."""
    return (d.get('point') is not None and np.isfinite(d.get('point', np.nan))
            and d['point'] >= 0
            and np.isfinite(d.get('lo', np.nan)) and d['lo'] >= NI_BOUND
            and d.get('rss_ok', False) and d.get('wall_ok', False))


def verdict(book, other=None):
    """Pre-registered D1 verdict for one book.

    book / other: {'C': arm, 'B': arm, 'sd_seed': float|None} with
    arm = {'point', 'lo', 'rss_ok', 'wall_ok'} (rss_ok/wall_ok = arm <= A).
    `other` (the other book's dict, optional) enables the both-books kills.
    Returns (token, reason) with token in ADOPT-C / ADOPT-B / HOLD / KILL.
    """
    arms = {k: book.get(k) for k in ('C', 'B') if book.get(k)}
    if 'C' not in arms or not np.isfinite(arms['C'].get('point', np.nan)):
        return 'HOLD', 'incomplete: arm C missing or failed'
    for k, d in arms.items():
        if np.isfinite(d.get('lo', np.nan)) and d['lo'] < KILL_BOUND:
            return 'KILL', f'arm {k} 90% lower bound {d["lo"]:+.4f} < {KILL_BOUND}'
    if arms['C'].get('rss_ok') is False:
        return 'KILL', 'peak RSS(C) > RSS(A)'
    if other and other.get('C'):
        oc = other['C']
        if arms['C']['point'] < 0 and oc.get('point', 0) < 0:
            return 'KILL', 'point dIC(C) < 0 on both books'
        sd1, sd2 = book.get('sd_seed'), other.get('sd_seed')
        if (sd1 is not None and sd2 is not None
                and sd1 > abs(arms['C']['point']) and sd2 > abs(oc.get('point', 0))):
            return 'KILL', 'SD_seed > |dIC(C)| on both books (FR-02 owns the window question)'
    for k in ('C', 'B'):
        if k in arms and _arm_passes(arms[k]):
            return f'ADOPT-{k}', (f'{k}: point {arms[k]["point"]:+.4f} >= 0, '
                                  f'lower bound {arms[k]["lo"]:+.4f} >= {NI_BOUND}, '
                                  f'RSS/wall <= A; (ii) holdout DSR and (iv) q10 '
                                  f'coverage NOT measured here')
    c = arms['C']
    if c['point'] >= 0 and np.isfinite(c.get('lo', np.nan)) and c['lo'] < NI_BOUND:
        return 'HOLD', 'no harm detected, underpowered (point >= 0, bound < -0.01)'
    if c['point'] < 0:
        return 'HOLD', 'point dIC(C) < 0 on this book (KILL needs both books)'
    return 'HOLD', 'C fails the resource condition (wall(C) > wall(A)) or bound unavailable'


def parse_lags(text):
    return tuple(sorted({int(x) for x in str(text).split(',') if x.strip() != ''}))


# ---------------------------------------------------------------------------
# Resource measurement
# ---------------------------------------------------------------------------

def _rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float('nan')


class RssSampler:
    """Background VmRSS sampler: per-arm peak (ru_maxrss is process-monotonic)."""

    def __init__(self, period=0.2):
        self.period = period
        self.peak = _rss_mb()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            r = _rss_mb()
            if r == r and r > self.peak:
                self.peak = r
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        r = _rss_mb()
        if r == r and r > self.peak:
            self.peak = r


# ---------------------------------------------------------------------------
# Data (column-projected; mirrors hypersearch_v2.load_data semantics)
# ---------------------------------------------------------------------------

def _hs():
    import hypersearch_v2 as hs  # pulls torch; imported lazily
    return hs


def _store_path(prefix, data):
    if data:
        return Path(data)
    stem = 'stock_training_data' if str(prefix).startswith('stock') else 'training_data'
    return _ROOT / f'{stem}.parquet'


def feature_columns_from_schema(schema, preset):
    """load_data's feature selection (hypersearch_v2.py:222-236) from a
    pyarrow schema: exclude Ticker/Date/Datetime/NextClose/Target_Return*/
    TB_*/Eff_Spread_Pct, keep float/int32/int64, filter by the preset,
    keep store column order."""
    import pyarrow as pa
    from indicator_config import load_indicator_config, get_preset_features
    preset_name = preset or load_indicator_config()["preset"]
    excl = {'Ticker', 'Date', 'Datetime', 'NextClose', 'Eff_Spread_Pct'}
    cols = []
    for f in schema:
        n = f.name
        if n in excl or n.startswith('Target_Return') or n.startswith('TB_'):
            continue
        if n.startswith('__index_level'):
            continue
        t = f.type
        if (pa.types.is_float32(t) or pa.types.is_float64(t)
                or pa.types.is_int32(t) or pa.types.is_int64(t)):
            cols.append(n)
    pf = get_preset_features(preset_name)
    if pf is not None:
        cols = [c for c in cols if c in pf]
    return cols, preset_name


def load_panel(path, preset, max_rows, fb):
    """Column-projected, row-capped panel with load_data's array contract.

    Two passes over the parquet: (1) Ticker + time only -> the per-ticker
    most-recent max_rows//n_tickers row mask (load_data :240-251); (2) the
    projected feature/target columns in record batches, keeping masked rows
    only — the full store is never materialised (stock store is 705 MB).
    Label time = the bar max(FORWARD_BARS) ahead (load_data :278-282).
    """
    import pyarrow.parquet as pq
    hs = _hs()
    pfile = pq.ParquetFile(str(path))
    schema = pfile.schema_arrow
    feat_cols, preset_name = feature_columns_from_schema(schema, preset)
    names = set(schema.names)
    tcol = 'Datetime' if 'Datetime' in names else (
        schema.pandas_metadata or {}).get('index_columns', [None])[0]
    tgt = f'Target_Return_{fb}' if f'Target_Return_{fb}' in names else 'Target_Return'

    base = pq.read_table(str(path), columns=['Ticker', tcol])
    tick = np.asarray(base.column('Ticker').to_pandas().astype(str))
    tt = base.column(tcol).to_pandas()
    times = (np.asarray(tt.values.astype('datetime64[s]').astype(np.int64)))
    del base, tt
    uniq = list(dict.fromkeys(tick.tolist()))
    n_t = len(uniq)
    keep = np.zeros(len(tick), bool)
    per = (max_rows // n_t) if (max_rows and len(tick) > max_rows and n_t) else None
    for t in uniq:
        idx = np.flatnonzero(tick == t)
        idx = idx[np.argsort(times[idx], kind='mergesort')]
        keep[idx if per is None else idx[-per:]] = True
    rows = np.flatnonzero(keep)

    feats, tgts = [], []
    off = 0
    for batch in pfile.iter_batches(batch_size=65_536, columns=feat_cols + [tgt]):
        n = batch.num_rows
        sel = rows[(rows >= off) & (rows < off + n)] - off
        if sel.size:
            b = batch.take(sel)
            feats.append(np.column_stack([np.asarray(b.column(c).to_numpy(zero_copy_only=False),
                                                     dtype=np.float32) for c in feat_cols]))
            tgts.append(np.asarray(b.column(tgt).to_numpy(zero_copy_only=False), np.float32))
        off += n
    X = np.vstack(feats)
    Y = np.concatenate(tgts)
    del feats, tgts
    tick_k, time_k = tick[rows], times[rows]

    max_fb = max(hs.FORWARD_BARS)
    parts_x, parts_y, parts_t, parts_l = [], [], [], []
    bounds = {}
    o = 0
    for t in uniq:
        m = np.flatnonzero(tick_k == t)
        m = m[np.argsort(time_k[m], kind='mergesort')]
        if m.size == 0:
            continue
        ts = time_k[m]
        li = np.minimum(np.arange(len(m)) + max_fb, len(m) - 1)
        parts_x.append(X[m]); parts_y.append(Y[m])
        parts_t.append(ts); parts_l.append(ts[li])
        bounds[t] = (o, o + len(m))
        o += len(m)
    all_features = np.vstack(parts_x)
    del X, parts_x
    all_features, n_bad, _, _ = hs._sanitize_nonfinite_features(all_features, copy=False)
    gc.collect()
    return {'features': all_features, 'returns': np.concatenate(parts_y),
            'times': np.concatenate(parts_t), 'label_times': np.concatenate(parts_l),
            'tickers': [t for t in uniq if t in bounds], 'bounds': bounds,
            'feature_cols': feat_cols, 'preset': preset_name, 'target_col': tgt,
            'n_nonfinite': int(n_bad)}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def _split_val(val_idx, times, label_times, seq_len):
    """Stop half (labels complete before the median split) / scored half
    (after a seq_len-bar embargo on the pooled grid)."""
    from objective_utils import embargo_end_time
    tv = times[val_idx]
    t_mid = float(np.median(tv))
    stop = val_idx[(tv < t_mid) & (label_times[val_idx] <= t_mid)]
    start = embargo_end_time(tv, t_mid, seq_len)
    score = val_idx[tv >= start]
    return stop, score


def _build(all_scaled, idx, offsets, chunk=8192):
    hs = _hs()
    out = np.empty((len(idx), len(offsets) * all_scaled.shape[1]), np.float32)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        out[i:i + len(j)] = hs.gather_windows(all_scaled, j, offsets).reshape(len(j), -1)
    return out


def _predict(booster, all_scaled, idx, offsets, chunk=8192):
    hs = _hs()
    p = np.empty(len(idx), np.float64)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        p[i:i + len(j)] = booster.predict(
            hs.gather_windows(all_scaled, j, offsets).reshape(len(j), -1))
    return p


def run_arm(all_scaled, returns, tr_idx, stop_idx, score_idx, offsets, seed, threads):
    """Fit one mean-leg booster (model_lgb.train_lgb, production params +
    seed/num_threads) and predict the scored rows. Returns dict."""
    from model_lgb import train_lgb
    res = {'rows': int(len(tr_idx)), 'cols': int(len(offsets) * all_scaled.shape[1])}
    t0 = time.time()
    with RssSampler() as smp:
        X = _build(all_scaled, tr_idx, offsets)
        Xs = _build(all_scaled, stop_idx, offsets)
        res['x_train_mb'] = round(X.nbytes / 1e6, 1)
        bst = train_lgb(X, returns[tr_idx], Xs, returns[stop_idx],
                        params={'n_jobs': int(threads), 'seed': int(seed)})
        del X, Xs
        gc.collect()
        res['pred'] = _predict(bst, all_scaled, score_idx, offsets)
        res['rounds'] = int(bst.current_iteration())
        del bst
        gc.collect()
    res['wall_s'] = round(time.time() - t0, 2)
    res['peak_rss_mb'] = round(smp.peak, 1)
    return res


def run_measurement(args, lags):
    from sklearn.preprocessing import RobustScaler
    hs = _hs()
    path = _store_path(args.prefix, args.data)
    t_load = time.time()
    P = load_panel(path, args.preset, args.max_rows, args.fb)
    seq, F = args.seq_len, P['features'].shape[1]
    print(f"[LOAD] {path.name}: {P['features'].shape} preset={P['preset']} "
          f"({F} feats) target={P['target_col']} in {time.time() - t_load:.1f}s, "
          f"RSS {_rss_mb():.0f} MB")
    returns = P['returns']
    folds = hs.get_walk_forward_folds(P['times'], P['label_times'], P['tickers'],
                                      P['bounds'], seq, n_folds=hs.NUM_FOLDS,
                                      purge_val_labels=True)
    folds = folds[-args.folds:] if args.folds > 0 else folds
    cap_a = legacy_row_cap(seq, F, hs.LGB_MAX_ROWS, hs.LGB_X_BYTE_BUDGET)
    full_off = np.arange(-seq, 0)
    sub_off = lag_offsets(seq, lags)
    arms = [a for a in args.arms if a in ('A', 'B', 'C')]
    seeds_a = [args.seed + 1 + k for k in range(args.extra_seeds)]
    pooled = {a: [] for a in arms}
    pooled.update({f'A_s{s}': [] for s in seeds_a})
    y_all, t_all, pooled_rows = [], [], []
    per_fold = []
    for fi, (tr, va) in enumerate(folds):
        tr = tr[~np.isnan(returns[tr])]
        va = va[~np.isnan(returns[va])]
        stop, score = _split_val(va, P['times'], P['label_times'], seq)
        if len(stop) > VAL_CAP:
            stop = stop[np.argsort(P['times'][stop])][-VAL_CAP:]
        tr_a = tr[np.argsort(P['times'][tr], kind='mergesort')][-cap_a:] if len(tr) > cap_a else tr
        scaler = RobustScaler().fit(P['features'][tr])
        all_scaled = scaler.transform(P['features']).astype(np.float32)
        fd = {'fold': fi, 'train_pool': int(len(tr)), 'rows_A': int(len(tr_a)),
              'stop_rows': int(len(stop)), 'score_rows': int(len(score)), 'arms': {}}
        plan = []
        for a in arms:
            plan.append((a, tr_a if a in ('A', 'B') else tr,
                         full_off if a == 'A' else sub_off, args.seed))
        if 'A' in arms:
            plan += [(f'A_s{s}', tr_a, full_off, s) for s in seeds_a]
        for name, rows, off, sd in plan:
            try:
                r = run_arm(all_scaled, returns, rows, stop, score, off, sd, args.threads)
                pooled[name].append(r.pop('pred'))
                fd['arms'][name] = r
                print(f"[FOLD {fi}] arm {name}: rows={r['rows']} cols={r['cols']} "
                      f"X={r['x_train_mb']} MB rounds={r['rounds']} wall={r['wall_s']}s "
                      f"peakRSS={r['peak_rss_mb']} MB")
            except Exception as e:  # fail-soft per arm
                fd['arms'][name] = {'error': repr(e)}
                pooled[name].append(np.full(len(score), np.nan))
                print(f"[FOLD {fi}] arm {name} FAILED: {e!r}")
        y_all.append(returns[score]); t_all.append(P['times'][score])
        pooled_rows.append(score)
        per_fold.append(fd)
        del all_scaled
        gc.collect()
    if not per_fold:
        return {'status': 'no_folds'}
    y = np.concatenate(y_all)
    tt = np.concatenate(t_all)
    blocks = tt // (WEEK_SECONDS * max(1, int(args.boot_blocks_weeks)))
    preds = {k: np.concatenate(v) for k, v in pooled.items() if v}
    ic = {k: spearman(v[np.isfinite(v) & np.isfinite(y)], y[np.isfinite(v) & np.isfinite(y)])
          for k, v in preds.items()}
    a_ics = [ic[k] for k in ['A'] + [f'A_s{s}' for s in seeds_a] if k in ic and np.isfinite(ic[k])]
    sd_seed = float(np.std(a_ics, ddof=1)) if len(a_ics) >= 2 else None

    def _agg(name, key):
        vals = [f['arms'].get(name, {}).get(key) for f in per_fold]
        vals = [v for v in vals if v is not None]
        return (max(vals) if key == 'peak_rss_mb' else sum(vals)) if vals else None

    res = {'ic': {k: round(v, 5) for k, v in ic.items()}, 'sd_seed': sd_seed,
           'n_scored': int(len(y)), 'arms': {}}
    if P['bounds'] and str(args.prefix).startswith('stock'):
        from objective_utils import cs_rank_ic
        res['cs_rank_ic'] = {k: cs_rank_ic(v, y, tt).get('mean') for k, v in preds.items()}
    book = {'sd_seed': sd_seed}
    for a in ('B', 'C'):
        if a not in preds or 'A' not in preds:
            continue
        bs = paired_block_bootstrap(preds[a], preds['A'], y, blocks,
                                    n_boot=args.B, seed=args.seed)
        rss_ok = (_agg(a, 'peak_rss_mb') is not None and _agg('A', 'peak_rss_mb') is not None
                  and _agg(a, 'peak_rss_mb') <= _agg('A', 'peak_rss_mb'))
        wall_ok = (_agg(a, 'wall_s') is not None and _agg('A', 'wall_s') is not None
                   and _agg(a, 'wall_s') <= _agg('A', 'wall_s'))
        noise = (sd_seed is not None and abs(bs['point']) < 2 * sd_seed)
        res['arms'][a] = {**bs, 'lo': bs['lo_one_sided'], 'rss_ok': bool(rss_ok),
                          'wall_ok': bool(wall_ok), 'within_seed_noise': bool(noise),
                          'wall_s_total': _agg(a, 'wall_s'),
                          'peak_rss_mb': _agg(a, 'peak_rss_mb')}
        book[a] = res['arms'][a]
    if 'A' in preds:
        res['arms']['A'] = {'wall_s_total': _agg('A', 'wall_s'),
                            'peak_rss_mb': _agg('A', 'peak_rss_mb')}
    other = None
    if args.pair_with:
        try:
            with open(args.pair_with) as f:
                o = json.load(f)
            other = {'sd_seed': o['result'].get('sd_seed'), **o['result'].get('arms', {})}
        except Exception as e:
            print(f"[PAIR] could not read {args.pair_with}: {e!r}")
    tok, why = verdict(book, other)
    res.update(verdict=tok, verdict_reason=why, per_fold=per_fold,
               panel={'rows': int(P['features'].shape[0]), 'n_features': F,
                      'feature_cols': P['feature_cols'], 'preset': P['preset'],
                      'target_col': P['target_col'], 'tickers': P['tickers'],
                      'cap_A': cap_a, 'n_nonfinite': P['n_nonfinite']},
               status='ok')
    return res


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run_plan(args, lags):
    """Plan + per-arm byte arithmetic from the parquet FOOTER only."""
    hs = _hs()
    path = _store_path(args.prefix, args.data)
    seq = int(args.seq_len)
    info = {'store': str(path), 'seq_len': seq, 'fb': args.fb, 'lags': list(lags)}
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(path))
        feat_cols, preset_name = feature_columns_from_schema(pf.schema_arrow, args.preset)
        n_rows = int(pf.metadata.num_rows)
    except Exception as e:
        print(f"[DRY] store footer unreadable ({e!r}); using --n-features {args.n_features}")
        feat_cols, preset_name, n_rows = [None] * int(args.n_features), args.preset, None
    F = len(feat_cols)
    cap_a = legacy_row_cap(seq, F, hs.LGB_MAX_ROWS, hs.LGB_X_BYTE_BUDGET)
    n_eff = min(n_rows, args.max_rows) if n_rows else None
    # last production fold trains on labels <= the 0.85 search-region quantile
    # of the (1 - HOLDOUT_FRACTION) search region -> ~0.85*0.88 of the panel.
    pool_c = int(n_eff * (1 - hs.HOLDOUT_FRACTION) * 0.85) if n_eff else None
    L = len(lags)
    rows = {'A': cap_a if pool_c is None else min(cap_a, pool_c),
            'B': cap_a if pool_c is None else min(cap_a, pool_c), 'C': pool_c}
    cols = {'A': seq * F, 'B': L * F, 'C': L * F}
    arms = {}
    for a in args.arms:
        if a not in rows:
            continue
        r = rows[a]
        arms[a] = {'rows': r, 'cols': cols[a],
                   'x_train_mb': round(r * cols[a] * 4 / 1e6, 1) if r else None,
                   'cells_M': round(r * cols[a] / 1e6, 1) if r else None,
                   'x_stop_mb_max': round(VAL_CAP * cols[a] * 4 / 1e6, 1)}
    n_fits = len(args.arms) * args.folds + (args.extra_seeds * args.folds if 'A' in args.arms else 0)
    info.update(preset=preset_name, n_features=F, store_rows=n_rows, rows_after_max_rows=n_eff,
                cap_A=cap_a, lgb_max_rows=hs.LGB_MAX_ROWS,
                lgb_x_byte_budget=hs.LGB_X_BYTE_BUDGET, fold_train_pool_est=pool_c,
                arms=arms, folds=args.folds, n_fits=n_fits, B=args.B,
                boot_block_weeks=args.boot_blocks_weeks)
    print(f"[DRY] store={path.name} rows={n_rows} -> max_rows {args.max_rows} -> {n_eff}; "
          f"preset={preset_name} F={F}; seq_len={seq} fb={args.fb} lags={list(lags)}")
    print(f"[DRY] cap_A = min({hs.LGB_MAX_ROWS}, max(20000, {hs.LGB_X_BYTE_BUDGET} // "
          f"({seq}*{F}*4))) = {cap_a}; fold-train pool (last fold, estimate) ~ {pool_c}")
    for a, d in arms.items():
        print(f"[DRY] arm {a}: rows {d['rows']} x cols {d['cols']} x 4 B = "
              f"{d['x_train_mb']} MB X_train ({d['cells_M']} M cells); stop set <= "
              f"{d['x_stop_mb_max']} MB")
    print(f"[DRY] {n_fits} mean-leg fits ({args.folds} fold(s) x arms {','.join(args.arms)}"
          f" + {args.extra_seeds} extra A seed(s)/fold); bootstrap B={args.B}, "
          f"block={args.boot_blocks_weeks} week(s). Nothing loaded, nothing written.")
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='LGB lag-subset A/B (D1, measurement-only)')
    ap.add_argument('--data', type=str, default=None,
                    help="Parquet store (default: training_data.parquet, or "
                         "stock_training_data.parquet when --prefix stock)")
    ap.add_argument('--prefix', type=str, default='', help="'' = crypto, 'stock' = stocks")
    ap.add_argument('--preset', type=str, default='stationary')
    ap.add_argument('--seq-len', type=int, default=24)
    ap.add_argument('--fb', type=int, default=24, help='forward bars (Target_Return_{fb})')
    ap.add_argument('--lags', type=str, default=','.join(map(str, DEFAULT_LAGS)))
    ap.add_argument('--arms', type=str, default='A,B,C')
    ap.add_argument('--folds', type=int, default=3,
                    help='use the LAST N of the production NUM_FOLDS folds')
    ap.add_argument('--max-rows', type=int, default=None,
                    help='panel row cap (default: 500000 crypto = hypersearch default; '
                         '200000 stock = run_pipeline.py:1243)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--boot-blocks-weeks', type=int, default=1)
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--extra-seeds', type=int, default=2)
    ap.add_argument('--threads', type=int, default=1, help='LightGBM num_threads')
    ap.add_argument('--n-features', type=int, default=30,
                    help='dry-run fallback F when the store footer is unreadable')
    ap.add_argument('--pair-with', type=str, default=None,
                    help="other book's result JSON (enables the both-books kills)")
    ap.add_argument('--json', type=str, default=None, help='output JSON path')
    ap.add_argument('--out', type=str, default=None,
                    help='output dir (default logs/lgb_lag_ab/) when --json is not given')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    a.arms = [x.strip().upper() for x in a.arms.split(',') if x.strip()]
    if a.max_rows is None:
        a.max_rows = 200_000 if str(a.prefix).startswith('stock') else 500_000
    return a


def main(argv=None):
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0)
    try:
        lags = parse_lags(args.lags)
        lag_column_index(args.seq_len, 1, lags)  # asserts 0 <= lag < seq_len
        if args.dry_run:
            dry_run_plan(args, lags)
            return 0
        t0 = time.time()
        res = run_measurement(args, lags)
        book = 'stock' if str(args.prefix).startswith('stock') else 'crypto'
        out = {'tool': 'lgb_lag_ab', 'design': 'research_signal.md SCOUT-2 D1',
               'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'book': book, 'args': {k: v for k, v in vars(args).items()},
               'lags': list(lags), 'wall_s': round(time.time() - t0, 1),
               'result': res}
        line = (f"VERDICT[{book}]: {res.get('verdict', 'HOLD')} — "
                f"{res.get('verdict_reason', res.get('status'))}")
        out['verdict_line'] = line
        for a in ('B', 'C'):
            d = res.get('arms', {}).get(a)
            if d:
                print(f"[RESULT] arm {a}: dIC {d['point']:+.4f} 90%-lower {d['lo']:+.4f} "
                      f"CI90 [{d['ci90'][0]:+.4f},{d['ci90'][1]:+.4f}] blocks={d['n_blocks']} "
                      f"rss_ok={d['rss_ok']} wall_ok={d['wall_ok']} "
                      f"seed-noise={d['within_seed_noise']}")
        print(f"[RESULT] IC {res.get('ic')} SD_seed={res.get('sd_seed')}")
        print(line)
        if args.json:
            dest = Path(args.json)
        else:
            d = Path(args.out) if args.out else _ROOT / 'logs' / 'lgb_lag_ab'
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            dest = d / f'lgb_lag_ab_{book}_{stamp}.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(out, f, indent=1, default=lambda o: (o.tolist() if hasattr(o, 'tolist') else str(o)))
        os.replace(tmp, dest)
        print(f"[OUT] {dest}")
    except AssertionError as e:
        print(f"[ERROR] {e}")
    except Exception as e:  # measurement: never raise past main
        print(f"[ERROR] lgb_lag_ab failed: {e!r}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
