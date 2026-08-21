"""Pure decision/math helpers for the hypersearch objective (2026-08 T1).

hypersearch_v2 imports torch and cannot run on the dev Mac; everything
provable with numpy lives here. simulate_trades_core is the EXACT legacy
non-overlapping hold walk from hypersearch_v2.simulate_trades when its new
arguments are None (pinned behavior-identical by a seeded fuzz test in
tests/test_c26_T1.py); block_ids / long_veto are the OBJECTIVE_V3
extensions (per-ticker boundary reset + q10 long-entry veto mirror).
"""
import numpy as np


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
