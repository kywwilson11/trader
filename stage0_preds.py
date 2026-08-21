"""Stage-0 predictions dump + hourly MTM equity (B02, measurement-only).

Pure numpy/pandas/stdlib kernels consumed by backtest.run_backtest to emit
the per-(symbol, bar) predictions frame both Stage-0 consumers were waiting
on, plus an honest mark-to-market equity curve for the replay window.

Row schema (one dict per selected symbol-bar):
    ts                str(times[i]) — same repr as backtest trade times
                      (pd.to_datetime-parseable, lexicographically sortable)
    symbol            ticker name
    pred / signal     the SAME blended prediction value, duplicated on
                      purpose: scripts/rank_gradient_report.py hard-codes
                      column 'signal' while scripts/ic_by_name.py defaults
                      to 'pred' — one dump feeds both with zero flags
    fwd_return        (close[i+h] - close[i]) / close[i] * 100
    horizon_bars      h
    lstm_pred/lgb_pred  blend legs when captured (None when unavailable)
    meta_p, q10       gate inputs when available (None otherwise)
    pred_thresh_ratio pred / trade_threshold (mirrors base_loop._conv_fields)

UNITS: pred/signal and fwd_return are PERCENT, matching harvest
Target_Return_* and fees.round_trip_cost_pct — rank_gradient_report's
--cost-pct subtracts raw, so the units must agree.

NON-OVERLAPPING guarantee: select_row_indices spaces selected bars >= the
horizon per name (and anchors them to every horizon-th union-grid
timestamp so cross-name panel periods align). ic_by_name therefore needs
no --min-t adjustment, and rank_gradient_report should be run with
--fwd-bars 1 — its ci90 widening exists for OVERLAPPING dumps, which this
producer never emits.

Consumers (both parse the JSON dump with zero flags):
    python scripts/ic_by_name.py --in stage0_preds.json --time-key ts
    python scripts/rank_gradient_report.py --preds stage0_preds.json \
        --fwd-bars 1 --cost-pct <fees> --extra-cols meta_p,pred_thresh_ratio
The JSON file is a BARE LIST of row dicts (ic_by_name's stricter contract).
"""
import csv
import json
import os

import numpy as np
import pandas as pd


def index_ns(index):
    """int64 NANOSECONDS since epoch for a datetime-like index/iterable.

    pandas >= 2 indexes can carry non-ns resolutions (pandas 3 defaults to
    'us'), where .asi8 returns the raw unit — normalize to ns so grid,
    per-name bar times, and trade times parsed via Timestamp.value (always
    ns) all live on one clock.
    """
    idx = pd.DatetimeIndex(index)
    try:
        idx = idx.as_unit('ns')
    except (AttributeError, TypeError):
        pass  # pandas < 2: asi8 is already ns
    return np.asarray(idx.asi8, dtype=np.int64)


def global_anchor_ns(union_times_ns, horizon):
    """Every horizon-th timestamp of the sorted union grid.

    The cross-name alignment grid: without it each name's rows start at its
    own first finite pred and portfolio_backtest.panel_from_frame periods
    degenerate to 1 candidate, destroying the rank buckets.
    """
    u = np.unique(np.asarray(union_times_ns, dtype=np.int64))
    return u[::max(1, int(horizon))]


def select_row_indices(times_ns, preds, horizon, anchor_ns=None):
    """Dump-row indices for one name: finite pred, fwd_return computable
    inside the window (i + horizon <= n-1), spaced >= horizon bars apart
    (non-overlap enforced per name even when the name has missing bars
    relative to the union grid), and on the anchor grid when given."""
    times_ns = np.asarray(times_ns, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.float64)
    n = len(preds)
    h = max(1, int(horizon))
    if n == 0 or h >= n:
        return []
    on_anchor = None
    if anchor_ns is not None:
        on_anchor = np.isin(times_ns,
                            np.asarray(anchor_ns, dtype=np.int64))
    out = []
    last = None
    for i in range(n - h):
        if not np.isfinite(preds[i]):
            continue
        if last is not None and i - last < h:
            continue
        if on_anchor is not None and not on_anchor[i]:
            continue
        out.append(int(i))
        last = i
    return out


def _opt(arr, i, digits):
    """round(float(arr[i]), digits) when arr is given and finite, else None."""
    if arr is None:
        return None
    try:
        v = float(arr[i])
    except (TypeError, ValueError, IndexError):
        return None
    return round(v, digits) if np.isfinite(v) else None


def build_rows(times, symbol, preds, closes, horizon, idx, *, lstm=None,
               lgb=None, meta_probs=None, q10=None, threshold=None):
    """Dump rows for one name at the selected indices (see module schema).

    Rows are bar-ordered per name by construction (idx ascending). Rows
    whose fwd_return is not computable (zero/non-finite close) are skipped.
    """
    h = max(1, int(horizon))
    rows = []
    for i in idx:
        c0 = float(closes[i])
        c1 = float(closes[i + h])
        if not (np.isfinite(c0) and np.isfinite(c1)) or c0 == 0.0:
            continue
        fwd = (c1 - c0) / c0 * 100.0
        if not np.isfinite(fwd):
            continue
        p = float(preds[i])
        row = {
            'ts': str(times[i]),
            'symbol': str(symbol),
            'pred': round(p, 6),
            # duplicated on purpose: rank_gradient_report hard-codes
            # 'signal', ic_by_name defaults to 'pred'
            'signal': round(p, 6),
            'fwd_return': round(fwd, 6),
            'horizon_bars': int(h),
            'lstm_pred': _opt(lstm, i, 6),
            'lgb_pred': _opt(lgb, i, 6),
            'meta_p': _opt(meta_probs, i, 4),
            'q10': _opt(q10, i, 4),
            'pred_thresh_ratio': (round(p / float(threshold), 4)
                                  if threshold and float(threshold) > 0
                                  else None),
        }
        rows.append(row)
    return rows


def write_rows(rows, path):
    """Atomic dump: .csv suffix -> csv.DictWriter, else a BARE JSON list
    (ic_by_name's contract). tmp + os.replace; tmp unlinked on failure."""
    path_s = str(path)
    tmp = path_s + '.tmp'
    try:
        if path_s.endswith('.csv'):
            fieldnames = list(rows[0].keys()) if rows else []
            stragglers = sorted({k for r in rows for k in r}
                                - set(fieldnames))
            fieldnames += stragglers
            with open(tmp, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, restval='')
                w.writeheader()
                w.writerows(rows)
        else:
            with open(tmp, 'w') as f:
                json.dump(rows, f)
        os.replace(tmp, path_s)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def max_drawdown_from_equity(equity):
    """Max (running peak - equity), running max seeded with 0.0 — the same
    convention as backtest.aggregate_metrics' trade-ordinal drawdown."""
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(
        np.concatenate([[0.0], equity]))[1:]
    return float(np.max(running_max - equity))


def mtm_equity(trades, price_series, grid_ns):
    """Hourly mark-to-market book equity from replay fills — the honest
    replacement input for the trade-ordinal cumsum drawdown (and for later
    block-bootstrap Sharpe work).

    trades: backtest trade dicts ({'entry_time','exit_time','entry',
    'net_pct','ticker',...}). price_series: {ticker: (times_ns int64 sorted,
    closes float array)}. grid_ns: sorted int64 ns mark timestamps — the
    union of the replay window's BAR timestamps (bar-close marks, not a
    wall-clock grid, so stock overnight gaps do not pad the series).

    Convention: equal-weight ADDITIVE PERCENT (the direct generalization of
    aggregate_metrics' cumsum): equity_pct(t) = sum of net_pct over trades
    closed by t, plus (px(t)-entry)/entry*100 over trades open at t, where
    px(t) is the ticker's last bar close at or before t. Cost is charged at
    exit (the open term is gross unrealized). Fail-soft per trade: unknown
    ticker / no price coverage / unparseable times -> the trade contributes
    only its closed step (counted in n_unmarked_trades).

    Invariant: with full price coverage and all trades closed by the window
    end, equity_pct[-1] == sum(net_pct) (within fp tolerance).
    """
    grid = np.asarray(grid_ns, dtype=np.int64)
    n_marks = int(grid.size)
    ts_list = [str(t) for t in pd.DatetimeIndex(grid.view('datetime64[ns]'))]
    closed = np.zeros(n_marks, dtype=np.float64)
    open_mtm = np.zeros(n_marks, dtype=np.float64)
    n_unmarked = 0
    parsed = []
    for t in (trades or []):
        try:
            e_ns = int(pd.to_datetime(t['entry_time']).value)
            x_ns = int(pd.to_datetime(t['exit_time']).value)
            net = float(t['net_pct'])
        except Exception:
            n_unmarked += 1
            continue
        parsed.append((e_ns, x_ns, net, t))
    if parsed and n_marks:
        # Closed step function: cumsum of net_pct in exit order, evaluated
        # at each grid mark via searchsorted.
        parsed.sort(key=lambda r: r[1])
        exit_sorted = np.array([r[1] for r in parsed], dtype=np.int64)
        cum = np.cumsum([r[2] for r in parsed])
        pos = np.searchsorted(exit_sorted, grid, side='right')
        closed = np.where(pos > 0, cum[np.maximum(pos - 1, 0)], 0.0)
        # Open (gross unrealized) marks per trade.
        for e_ns, x_ns, _net, t in parsed:
            mask_idx = np.flatnonzero((grid >= e_ns) & (grid < x_ns))
            if len(mask_idx) == 0:
                continue
            ser = price_series.get(t.get('ticker')) if price_series else None
            entry_px = float(t.get('entry') or 0.0)
            if ser is None or not np.isfinite(entry_px) or entry_px == 0.0:
                n_unmarked += 1
                continue
            times_ns = np.asarray(ser[0], dtype=np.int64)
            px_arr = np.asarray(ser[1], dtype=np.float64)
            look = np.searchsorted(times_ns, grid[mask_idx],
                                   side='right') - 1
            ok = look >= 0
            if ok.any():
                px = px_arr[look[ok]]
                ok2 = np.isfinite(px)
                tgt = mask_idx[ok][ok2]
                open_mtm[tgt] += (px[ok2] - entry_px) / entry_px * 100.0
            else:
                n_unmarked += 1
    equity = closed + open_mtm
    return {
        'ts': ts_list,
        'equity_pct': [round(float(x), 4) for x in equity],
        'max_drawdown_pct': max_drawdown_from_equity(equity),
        'n_marks': n_marks,
        'n_unmarked_trades': int(n_unmarked),
    }
