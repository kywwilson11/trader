"""Pure decision/math helpers for the hypersearch objective (2026-08 T1).

hypersearch_v2 imports torch and cannot run on the dev Mac; everything
provable with numpy lives here. simulate_trades_core is the EXACT legacy
non-overlapping hold walk from hypersearch_v2.simulate_trades when its new
arguments are None (pinned behavior-identical by a seeded fuzz test in
tests/test_c26_T1.py); block_ids / long_veto are the OBJECTIVE_V3
extensions (per-ticker boundary reset + q10 long-entry veto mirror).
"""
import hashlib

import numpy as np


def derive_seed(base_seed, *parts):
    """Deterministic 32-bit sub-seed from a base seed + context tokens
    (R2C-04 / L3 TRAINER_SEED plumbing — the FR-02 / FR-13 determinism
    prerequisite).

    Hypersearch derives every RNG consumer's seed from
    (TRAINER_SEED, study_name, trial.number, fold) so two runs at the
    same base seed are reproducible while folds / trials / the sampler
    never share a stream. sha256 of 'base|part|part|...' truncated to
    4 bytes — the [0, 2**32) range every consumer accepts
    (torch.manual_seed, np.random.default_rng, optuna TPESampler).
    """
    key = '|'.join([str(int(base_seed))] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big')


def lagged_regime_series(actual_returns, forward_bars, window=50):
    """TRAINING_REPAIRS_V1 (R2C-04 / L2) regime-label kernel: the trailing
    scaled mean of fb-bar returns COMPLETED by each row.

    The legacy series was a trailing mean over rows t-window+1..t of the
    FORWARD fb-bar returns stamped at those rows — but a return stamped at
    row s covers bars s..s+fb and is only knowable at s+fb, so the legacy
    bull/bear mask at t embedded returns through t+fb (look-ahead), and
    the convolve('full') prefix left rows 0..window-2 with under-scaled
    partial sums.

    Repaired: the value at row t averages the `window` most recent returns
    stamped at rows <= t - fb (all completed by t), scaled by window/fb
    (the same cumulative-return approximation as legacy). Rows without a
    full completed window (t < window-1+fb) are NaN — the caller must
    treat NaN as "no regime" (excluded from every mask), not sideways.
    """
    r = np.asarray(actual_returns, dtype=np.float64)
    finite = np.where(np.isfinite(r), r, 0.0)
    n = len(finite)
    fb = int(max(forward_bars, 1))
    w = int(window)
    out = np.full(n, np.nan)
    if n == 0 or w <= 0:
        return out
    kernel = np.ones(w) / w
    trailing_mean = np.convolve(finite, kernel, mode='full')[:n]
    start = w - 1 + fb
    if start < n:
        out[start:] = trailing_mean[w - 1:n - fb] * (w / fb)
    return out


def embargo_end_time(times, boundary, n_bars):
    """TRAINING_REPAIRS_V1 (R2C-04 / L6): embargo end denominated in BARS.

    Returns the timestamp of the n_bars-th DISTINCT bar timestamp strictly
    after `boundary` on the pooled bar grid `times` (multi-ticker rows
    sharing a timestamp collapse to one bar); validation admits rows with
    t >= the returned value, i.e. the first n_bars-1 bars after the
    boundary are embargoed and the n_bars-th is the first admitted bar.
    On a continuous hourly grid with an on-grid boundary this equals the
    legacy `boundary + n_bars*3600` rule exactly; on the stock RTH grid it
    counts TRADING bars (the legacy calendar-seconds rule shrank a 40-bar
    embargo to ~11 RTH bars).

    n_bars <= 0 or an empty grid returns `boundary` (legacy zero-embargo
    semantics); fewer than n_bars bars after the boundary returns +inf
    (the embargo swallows the region — the caller's minimum-val-size
    check then drops the fold).
    """
    u = np.unique(np.asarray(times))
    if int(n_bars) <= 0 or u.size == 0:
        return float(boundary)
    pos = int(np.searchsorted(u, boundary, side='right'))
    k = pos + int(n_bars) - 1
    if k >= u.size:
        return float('inf')
    return float(u[k])


def ticker_block_ids(global_rows, ticker_boundaries):
    """Block id (int64) per global row index.

    ticker_boundaries: dict {ticker: (start, end)} or iterable of
    (start, end) pairs over the contiguous ticker-concatenated index —
    the same math as evaluate_on_holdout's hoisted block reconstruction.
    """
    if isinstance(ticker_boundaries, dict):
        blocks = sorted(ticker_boundaries.values())
    else:
        blocks = sorted(ticker_boundaries)
    starts = np.asarray([b[0] for b in blocks])
    return np.searchsorted(starts, np.asarray(global_rows, np.int64),
                           side='right') - 1


def simulate_trades_core(predictions, actual_returns, threshold, forward_bars,
                         txn_cost_pct, long_only=False, block_ids=None,
                         long_veto=None):
    """Non-overlapping hold walk. Returns (trade_returns f64, entries i64).

    Exact legacy semantics when block_ids and long_veto are None:
    long entry on p > threshold & finite r -> r - cost, i += forward_bars;
    short entry on (not long_only) & p < -threshold & finite r ->
    -r - cost, i += forward_bars; else i += 1.

    block_ids (len n, OBJECTIVE_V3): a hold never spans a ticker-block
    boundary — after an entry at i the scan resumes at
    min(i + forward_bars, first index of the next block).
    long_veto (bool, len n): True at i blocks the LONG entry only (falls
    through to the short test / i += 1) — the exact mirror of backtest.py's
    q10 tail veto and the live base_loop q10_tail_veto (both gate entries,
    i.e. longs).
    """
    predictions = np.asarray(predictions)
    actual_returns = np.asarray(actual_returns)
    n = len(predictions)
    next_block_start = None
    if block_ids is not None:
        bids = np.asarray(block_ids)
        # First index of the NEXT block for every row: change-points where
        # block id differs from the previous row, mapped per row via
        # searchsorted (O(n log n) max).
        change = np.flatnonzero(np.diff(bids) != 0) + 1
        ext = np.append(change, n)
        next_block_start = ext[np.searchsorted(change, np.arange(n),
                                               side='right')]
    trade_returns = []
    entries = []
    i = 0
    while i < n:
        p = predictions[i]
        r = actual_returns[i]
        vetoed = long_veto is not None and bool(long_veto[i])
        if (not vetoed) and p > threshold and np.isfinite(r):
            trade_returns.append(r - txn_cost_pct)
            entries.append(i)
            nxt = i + forward_bars
            if next_block_start is not None:
                nxt = min(nxt, int(next_block_start[i]))
            i = nxt
        elif (not long_only) and p < -threshold and np.isfinite(r):
            trade_returns.append(-r - txn_cost_pct)
            entries.append(i)
            nxt = i + forward_bars
            if next_block_start is not None:
                nxt = min(nxt, int(next_block_start[i]))
            i = nxt
        else:
            i += 1
    return (np.asarray(trade_returns, dtype=np.float64),
            np.asarray(entries, dtype=np.int64))


def v3_trade_threshold_range(asset_type):
    """OBJECTIVE_V3 trade_threshold Optuna range, floor-anchored to the
    book's DEPLOYMENT edge (computed from the SAME fees functions the
    live/backtest admission gates use — crypto [0.96, 2.0], stock
    [0.18, 0.57] at today's constants; the values float automatically if
    the fee schedule moves).

    Lower bound = 0.8x the admission floor — just below the live admission
    point so TPE sees the gradient across it; upper = 2.5x, clamped to
    adaptive_config HARD_LIMITS' 2.0.
    """
    from fees import required_edge_pct, FLAT_SPREAD_PCT
    key = 'crypto' if asset_type == 'crypto' else 'stock'
    floor = required_edge_pct(key, spread_pct=FLAT_SPREAD_PCT[key])
    lo = round(0.8 * floor, 2)
    hi = round(min(2.5 * floor, 2.0), 2)
    if hi <= lo:
        hi = round(lo + 0.05, 2)
    return [lo, hi]


def refit_epoch_budget(fold_best_epochs, max_epochs=60):
    """Fixed epoch budget for the final refit (B12.1 "collective early
    stopping"): median of the winning trial's per-fold best epochs, clamped
    to [1, max_epochs]. None when no usable record exists."""
    arr = np.asarray(list(fold_best_epochs or []), dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size == 0:
        return None
    return int(min(max(int(np.median(arr)), 1), int(max_epochs)))


def lgb_refit_indices(valid_idx, label_times, times, returns, boundary,
                      max_rows=None):
    """R2C-03 (LGB_REFIT_FULL): purged pre-holdout row set for the LGB
    full refit — final_refit's exact purge (a row is kept only when its
    LABEL window completes on/before the holdout boundary,
    label_times <= boundary), then the NaN-label filter, then the SAME
    most-recent-first row cap train_lgb_ensemble applies to the fold
    path (sort by bar time, keep the newest max_rows). Order matters:
    the cap runs LAST so it counts only usable rows.
    """
    idx = np.asarray(valid_idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    idx = idx[np.asarray(label_times)[idx] <= boundary]
    idx = idx[~np.isnan(np.asarray(returns, dtype=np.float64)[idx])]
    if max_rows is not None and len(idx) > int(max_rows):
        idx = idx[np.argsort(np.asarray(times)[idx])][-int(max_rows):]
    return idx


def holdout_boundary(all_times, fixed_days=None, holdout_fraction=0.12):
    """R2C-05 (FR-01): the search/holdout boundary timestamp — the ONE
    rule every consumer (folds, refit purge, blend M4 guard, OOF pack,
    the holdout certificate) inherits via hypersearch's
    get_holdout_boundary choke point.

    fixed_days None (legacy, byte-identical): the (1 - holdout_fraction)
    quantile of the pooled row timestamps — a PROPORTIONAL holdout whose
    calendar width scales with the data span (a 1Y window arm gets ~44
    days while full history gets ~200), which makes cross-arm DSRs
    incomparable and lets the promotion gate's n_eff >= 10 fail-closed
    floor flunk short windows on gate mechanics, not skill.

    fixed_days set: max(all_times) - fixed_days * 86400 — a fixed
    trailing calendar span, comparable across FR-02 window arms and
    successive weekly retrains. Purge semantics downstream are unchanged
    (label windows crossing the boundary are excluded the same way for
    either rule).
    """
    t = np.asarray(all_times)
    if fixed_days is None:
        return int(np.quantile(t, 1.0 - holdout_fraction))
    return int(t.max() - float(fixed_days) * 86400.0)


def window_cutoff(times, window_days):
    """R2C-05 (FR-02): trailing training-window cutoff for ONE ticker.

    Returns the minimum timestamp KEPT — max(times) - window_days*86400,
    anchored to the ticker's own final bar (rows with t >= cutoff
    survive; equal-timestamp boundary rows are kept). None when
    window_days is None / non-finite / <= 0 or times is empty: no mask,
    the full-history arm.
    """
    if window_days is None:
        return None
    t = np.asarray(times)
    if t.size == 0:
        return None
    wd = float(window_days)
    if not np.isfinite(wd) or wd <= 0:
        return None
    return float(t.max() - wd * 86400.0)


def _average_ranks(x):
    """0-based midranks (ties averaged) — the Spearman rank transform,
    no scipy. mergesort keeps the transform deterministic."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    sx = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    n = x.size
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def cs_rank_ic(preds, y, group_ids, min_group=5, n_splits=4):
    """R2C-07 (FR-05): per-timestamp cross-sectional rank IC.

    Groups rows by group_ids (bar timestamps: np.unique order ==
    chronological), computes Spearman rho per group as the Pearson
    correlation of tie-averaged ranks (no scipy), and pools:

      {'mean', 'se', 'n_groups', 'n_skipped', 'splits'}

    - Only groups with >= min_group rows where BOTH pred and y are finite
      are scored (crypto's ~6-name cross-sections just clear the default
      5; thinner timestamps are counted in n_skipped, never scored).
    - A group whose ranks are degenerate (all preds or all y tied) is
      skipped too — 0/0 is not evidence.
    - se = std(ddof=1)/sqrt(n_groups) over the per-group ICs (None when
      n_groups < 2); mean is None when no group scored.
    - splits: n_splits contiguous chronological chunks of the per-group
      IC sequence (np.array_split), each {'mean', 'n_groups'} — the
      sub-period stability read the FR-05 certificate lines print.
    """
    p = np.asarray(preds, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    g = np.asarray(group_ids)
    finite = np.isfinite(p) & np.isfinite(yy)
    ics = []
    n_skipped = 0
    if g.size:
        order = np.argsort(g, kind='mergesort')
        gs = g[order]
        seg_starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        seg_ends = np.r_[seg_starts[1:], gs.size]
        for s, e in zip(seg_starts, seg_ends):
            rows = order[s:e]
            rows = rows[finite[rows]]
            if rows.size < int(min_group):
                n_skipped += 1
                continue
            rp = _average_ranks(p[rows])
            ry = _average_ranks(yy[rows])
            sp = rp.std()
            sy = ry.std()
            if sp == 0.0 or sy == 0.0:
                n_skipped += 1
                continue
            ics.append(float(np.mean((rp - rp.mean()) * (ry - ry.mean()))
                             / (sp * sy)))
    arr = np.asarray(ics, dtype=np.float64)
    ng = int(arr.size)
    mean = float(arr.mean()) if ng else None
    se = float(arr.std(ddof=1) / np.sqrt(ng)) if ng >= 2 else None
    splits = [{'mean': (float(c.mean()) if c.size else None),
               'n_groups': int(c.size)}
              for c in np.array_split(arr, int(n_splits))]
    return {'mean': mean, 'se': se, 'n_groups': ng,
            'n_skipped': int(n_skipped), 'splits': splits}


def fixed_boost_rounds(best_iteration, current_iteration):
    """Fixed round count for the no-early-stopping LGB refit (the
    collective-early-stopping analog of refit_epoch_budget, R2C-03): the
    fold booster's early-stopped best_iteration when one was recorded
    (> 0), else its total trained iteration count; None when neither is
    usable — the caller must then skip the refit and fall back to the
    fold booster.
    """
    for cand in (best_iteration, current_iteration):
        try:
            v = int(cand) if cand is not None else 0
        except (TypeError, ValueError):
            v = 0
        if v > 0:
            return v
    return None
