"""All chart math for gui.py.

MUST stay free of PySide6 / pyqtgraph / pandas — unit-tested on the dev Mac with
numpy + stdlib only. gui.py owns every Qt/pyqtgraph object; this module hands
back plain numpy arrays, dicts, and small dataclasses that gui.py paints.
"""

import datetime
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 1) Status
OK = 'ok'
PARTIAL = 'partial'
EMPTY = 'empty'
ERROR = 'error'

def format_age(seconds):
    """Coarse single-unit age string: '12s' / '5m' / '3h' / '2d' (floored)."""
    seconds = max(0, seconds)
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def artifact_freshness(items, now=None, default_stale_s=172800.0):
    """[{name, path, exists, age_s, stale}] for the GUI reports-freshness
    strip. items: iterable of (name, path) or (name, path, stale_after_s);
    stale_after_s None => never age-stale (append-only ledgers). Missing
    file => exists False, age_s None, stale True. Never raises."""
    try:
        if now is None:
            now = time.time()
        rows = []
        for item in items:
            try:
                if len(item) >= 3:
                    name, path, stale_after_s = item[0], item[1], item[2]
                else:
                    name, path = item[0], item[1]
                    stale_after_s = default_stale_s
                path = str(path)
                try:
                    mtime = os.path.getmtime(path)
                    exists = True
                except (OSError, TypeError, ValueError):
                    exists = False
                    mtime = None
                age_s = max(0.0, float(now) - float(mtime)) if exists else None
                stale = (not exists) or (stale_after_s is not None
                                         and age_s > stale_after_s)
                rows.append({'name': name, 'path': path, 'exists': exists,
                             'age_s': age_s, 'stale': bool(stale)})
            except Exception:
                rows.append({'name': str(item), 'path': None, 'exists': False,
                             'age_s': None, 'stale': True})
        return rows
    except Exception:
        return []


def format_si(v):
    """SI-abbreviated tick label for a volume axis: '1.2K' / '3.4M' / '5.6B'.
    Magnitudes under 1000 print as a plain rounded integer; at/above 1000
    the largest applicable unit (K/M/B) is used with exactly one decimal,
    then a trailing '.0' is stripped (1000 -> '1K', not '1.0K'). Sign is
    kept as a leading '-' (chosen from the signed value, applied to the
    magnitude-derived text). Never raises: non-numeric/non-finite input
    -> '0'."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '0'
    if not math.isfinite(v):
        return '0'
    sign = '-' if v < 0 else ''
    mag = abs(v)
    if mag < 1000:
        return f'{sign}{int(round(mag))}'
    for suffix, size in (('B', 1e9), ('M', 1e6), ('K', 1e3)):
        if mag >= size:
            text = f'{mag / size:.1f}'
            if text.endswith('.0'):
                text = text[:-2]
            return f'{sign}{text}{suffix}'
    return f'{sign}{int(round(mag))}'  # unreachable: mag >= 1000 always hits the K tier
@dataclass
class ChartStatus:
    status: str = OK
    message: str = ''
    updated_at: "float | None" = None
    note: str = ''
    def title_suffix(self, now: "float | None" = None) -> str:
        if self.status == OK and not self.note:
            return ''
        text = self.message or self.note
        suffix = f' — {text}' if text else ''
        if self.updated_at is not None:
            now = now if now is not None else time.time()
            suffix += f' · updated {format_age(now - self.updated_at)} ago'
        return suffix
# 2) Coercion
ZOOM_DAYS = {'1Y': 365, '3M': 90, '1M': 30, '1W': 7, '1D': 1}

def coerce_xy(ts, *ys):
    """Coerce (ts, *ys) to float arrays: truncate to common length, drop
    non-finite rows jointly, sort by ts if needed. Returns (t, [ys...], note).
    """
    try:
        t = np.asarray(ts, dtype=float)
        ys_arr = [np.asarray(y, dtype=float) for y in ys]
    except (TypeError, ValueError):
        return np.array([], dtype=float), [], 'non-numeric data'
    lengths = [len(t)] + [len(y) for y in ys_arr]
    n = min(lengths) if lengths else 0
    note = ''
    if len(set(lengths)) > 1:
        note = f'length mismatch: truncated to {n}'
    t = t[:n]
    ys_arr = [y[:n] for y in ys_arr]
    if n == 0:
        return t, [y for y in ys_arr], note
    mask = np.isfinite(t)
    for y in ys_arr:
        mask &= np.isfinite(y)
    t = t[mask]
    ys_arr = [y[mask] for y in ys_arr]
    if len(t) > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind='stable')
        t = t[order]
        ys_arr = [y[order] for y in ys_arr]
    return t, ys_arr, note
# 3) Downsampling (LTTB)
def lttb_indices(x, y, n_out):
    """Largest-Triangle-Three-Buckets downsampling. Returns selected indices,
    always including 0 and len-1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= n_out or n_out < 3:
        return np.arange(n)
    idx_out = np.empty(n_out, dtype=np.int64)
    idx_out[0] = 0
    idx_out[-1] = n - 1
    every = (n - 2) / (n_out - 2)
    a = 0
    for i in range(n_out - 2):
        avg_start = min(int(math.floor((i + 1) * every) + 1), n - 1)
        avg_end = min(int(math.floor((i + 2) * every) + 1), n)
        if avg_end <= avg_start:
            avg_end = avg_start + 1
        avg_x = np.mean(x[avg_start:avg_end])
        avg_y = np.mean(y[avg_start:avg_end])
        range_start = min(int(math.floor(i * every) + 1), n - 1)
        range_end = min(int(math.floor((i + 1) * every) + 1), n)
        if range_end <= range_start:
            range_end = range_start + 1
        ax, ay = x[a], y[a]
        bx = x[range_start:range_end]
        by = y[range_start:range_end]
        area = np.abs((ax - avg_x) * (by - ay) - (ax - bx) * (avg_y - ay))
        max_idx = range_start + int(np.argmax(area))
        idx_out[i + 1] = max_idx
        a = max_idx
    return idx_out
def downsample(x, y, max_points=1500):
    idx = lttb_indices(x, y, max_points)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return idx, x[idx], y[idx]
# 4) Equity math
def compute_hwm(equity):
    equity = np.asarray(equity, dtype=float)
    if len(equity) == 0:
        return equity.copy()
    return np.maximum.accumulate(equity)
SECONDS_PER_YEAR = 365.25 * 86400.0

def perf_stats(equity, pnl, t=None):
    """Equity-curve performance tile stats. Every stat is NaN-safe: each key
    is either a finite float or None — never NaN/inf — so a missing or
    degenerate input can't leak garbage into the GUI.

    - total_return: sum of the daily pnl series. 0.0 if pnl is empty.
    - best_day / worst_day: max/min of the finite daily pnl entries. 0.0 if
      none are finite. [unchanged from the original implementation]
    - max_dd_pct: running-peak drawdown, in percent, off `equity`. 0.0 if
      equity is empty. [unchanged from the original implementation]
    - sharpe: mean(returns) / std(returns) * ann, where `returns` is the
      simple period-over-period % change of equity's finite values (ddof=1,
      matching beta_ledger.py's convention). None unless at least 2 such
      returns exist and their std is nonzero.
    - sortino: same numerator and `ann` as sharpe, but the denominator is
      the stdev of only the below-zero returns (downside deviation). None
      if fewer than 2 downside returns exist, or their stdev is zero.
    - volatility: annualized stdev of the same returns used by sharpe. Same
      existence gate as sharpe, but reports 0.0 rather than None when the
      std is exactly zero — a stdev of zero is itself a real, computable
      answer, unlike a ratio with zero in the denominator.
    - win_rate: fraction of nonzero, finite pnl entries that are > 0. None
      if there are no nonzero finite pnl entries.
    - cagr: (last/first equity) ** (1 year / elapsed) - 1, where `elapsed`
      is the wall-clock span between the first and last (equity, t) pair
      that are both finite. None if `t` is None, fewer than 2 such pairs
      exist, elapsed <= 0, or either endpoint equity is <= 0 (a
      non-positive endpoint has no real-valued CAGR).

    Annualization (`ann`, shared by sharpe/sortino/volatility): this is a
    mixed 24/7-crypto + RTH-only-stocks book, so neither a fixed 252
    (trading days) nor 365 (calendar days) constant is right for both.
    When `t` — epoch-second timestamps parallel to `equity`, same
    convention as the rest of this module — is given, `ann` is instead
    derived from the data's OWN median sample spacing:
    sqrt(365.25*86400 / median(diff(t))), which self-adjusts to hourly
    crypto bars, daily stock bars, or anything else. Existing positional
    callers (equity, pnl) keep computing the original 4 keys byte-for-byte
    unchanged, and get the classic sqrt(252) daily convention for the new
    annualized keys.
    """
    equity = np.asarray(equity, dtype=float)
    pnl = np.asarray(pnl, dtype=float)
    finite_pnl = pnl[np.isfinite(pnl)]
    total_return = float(np.nansum(pnl)) if len(pnl) else 0.0
    best_day = float(np.nanmax(finite_pnl)) if len(finite_pnl) else 0.0
    worst_day = float(np.nanmin(finite_pnl)) if len(finite_pnl) else 0.0
    max_dd = 0.0
    if len(equity):
        peak = equity[0]
        for eq in equity:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak else 0.0
            if dd > max_dd:
                max_dd = dd

    # win_rate: independent of equity/t. Finite-first like best/worst_day
    # above — nan != 0 is True in numpy, so an unfiltered nan pnl would
    # silently count as a loss below.
    win_rate = None
    if len(finite_pnl):
        nonzero_pnl = finite_pnl[finite_pnl != 0]
        if len(nonzero_pnl):
            win_rate = float(np.mean(nonzero_pnl > 0))

    # Annualization factor: real median sample spacing when timestamps are
    # available; classic sqrt(252) daily convention otherwise (also what
    # every existing positional (equity, pnl) caller gets).
    ann = math.sqrt(252.0)
    if t is not None:
        t_finite = np.asarray(t, dtype=float)
        t_finite = t_finite[np.isfinite(t_finite)]
        if len(t_finite) >= 2:
            dt = np.diff(t_finite)
            dt = dt[dt > 0]
            if len(dt):
                ann = math.sqrt(SECONDS_PER_YEAR / float(np.median(dt)))

    # sharpe/sortino/volatility: from equity's OWN finite values only — a
    # bad/missing timestamp must not silently discard good equity data (t
    # only feeds `ann` above and `cagr` below).
    sharpe = sortino = volatility = None
    eq_ok = equity[np.isfinite(equity)]
    if len(eq_ok) >= 2:
        prev = eq_ok[:-1]
        nz = prev != 0
        rets = (eq_ok[1:][nz] - prev[nz]) / prev[nz]
        rets = rets[np.isfinite(rets)]
        if len(rets) >= 2:
            mean_r = float(np.mean(rets))
            std_r = float(np.std(rets, ddof=1))
            volatility = std_r * ann
            if std_r > 0:
                sharpe = mean_r / std_r * ann
            downside = rets[rets < 0]
            if len(downside) >= 2:
                dd_dev = float(np.std(downside, ddof=1))
                if dd_dev > 0:
                    sortino = mean_r / dd_dev * ann

    # cagr: needs equity PAIRED with t — elapsed time is meaningless without
    # a matching timestamp for each equity reading used.
    cagr = None
    if t is not None:
        n = min(len(equity), len(t))
        eq_n = equity[:n]
        t_n = np.asarray(t, dtype=float)[:n]
        paired = np.isfinite(eq_n) & np.isfinite(t_n)
        eq_p = eq_n[paired]
        t_p = t_n[paired]
        if len(eq_p) >= 2:
            elapsed = float(t_p[-1] - t_p[0])
            first_eq = float(eq_p[0])
            last_eq = float(eq_p[-1])
            if elapsed > 0 and first_eq > 0 and last_eq > 0:
                try:
                    # A tiny `elapsed` (e.g. two equity snapshots seconds
                    # apart, early in a fresh bot's life) blows the exponent
                    # up enough that Python's ** raises OverflowError rather
                    # than saturating to inf — None is the correct "can't
                    # extrapolate a year from this" answer either way.
                    cagr = (last_eq / first_eq) ** (SECONDS_PER_YEAR / elapsed) - 1.0
                except OverflowError:
                    cagr = None

    def _fin(x):
        """None-through, else finite-float-or-None (guards rare overflow/
        near-zero-denominator blowups from ever reaching the GUI as inf)."""
        return x if (x is not None and math.isfinite(x)) else None

    return {
        'total_return': total_return,
        'best_day': best_day,
        'worst_day': worst_day,
        'max_dd_pct': float(max_dd),
        'sharpe': _fin(sharpe),
        'sortino': _fin(sortino),
        'volatility': _fin(volatility),
        'win_rate': win_rate,
        'cagr': _fin(cagr),
    }


def obs_per_year(t):
    """Observed sampling rate of a series: SECONDS_PER_YEAR / median positive
    spacing of epoch-second timestamps `t`. None if < 2 finite points or no
    positive gaps. Same definition as beta_ledger's period.obs_per_year_grid
    (which computes 365.25 / median gap DAYS on a DatetimeIndex) and the same
    median-spacing convention perf_stats uses for `ann`."""
    tt = np.asarray(t, dtype=float)
    tt = tt[np.isfinite(tt)]
    if len(tt) < 2:
        return None
    d = np.diff(np.sort(tt))
    d = d[d > 0]
    if not len(d):
        return None
    return float(SECONDS_PER_YEAR / float(np.median(d)))


def align_benchmark(t_equity, t_bench, bench_close, equity):
    """Resample a benchmark close series onto the equity view's OWN
    timestamps, rescaled to sit on top of the equity curve at the first
    point where the two overlap — the pure-numpy half of the "are we
    beating buy-and-hold" equity overlay (SPY/BTC vs the bot's equity).

    Signature note: the original ask was `align_benchmark(t_equity,
    t_bench, bench_close, base_value)` with `base_value` a scalar. That
    would force the caller to independently re-derive which `t_equity`
    index is "the first overlapping point" (to look up its equity value)
    using the same overlap rule this function uses internally — two
    implementations of one rule, one bug waiting to happen. Instead this
    takes `equity` (the array parallel to `t_equity`, i.e. EquityView.equity)
    so there is exactly one place that decides the overlap and reads off
    the anchor value.

    Args:
        t_equity: epoch-second timestamps of the equity curve to overlay
            onto (e.g. `EquityView.ts`) — any order/length, need not line
            up with t_bench at all.
        t_bench, bench_close: the benchmark's OWN epoch-second timestamps
            and close prices (e.g. straight off `fetch_chart`'s SPY/BTC
            cache) — arbitrary length/order/resolution.
        equity: array parallel to t_equity (e.g. `EquityView.equity`) —
            used only to read the dollar value at the first overlapping
            timestamp, so the returned series starts exactly on the
            equity line instead of an arbitrary 1.0/100 base.

    Returns:
        A float array of exactly `len(t_equity)`: the benchmark, linearly
        interpolated onto every `t_equity` timestamp and scaled so it
        equals `equity[i0]` at `i0` (the first `t_equity` index inside
        `[min(t_bench), max(t_bench)]`). NaN at every `t_equity` position
        outside that coverage range. Never raises — an all-NaN array
        (never an exception) comes back for empty inputs, no overlap at
        all, or a non-positive/non-finite benchmark price at the anchor
        (a <= 0 benchmark price can't form a meaningful ratio).
    """
    try:
        n = len(t_equity)
    except TypeError:
        return _f64()
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out
    try:
        t_equity_f = np.asarray(t_equity, dtype=float)
        equity_f = np.asarray(equity, dtype=float)
    except (TypeError, ValueError):
        return out
    t_bench_c, bench_ys, _ = coerce_xy(t_bench, bench_close)
    if len(t_bench_c) == 0 or len(equity_f) == 0:
        return out
    t_bench_min, t_bench_max = t_bench_c[0], t_bench_c[-1]
    in_range = (np.isfinite(t_equity_f) & (t_equity_f >= t_bench_min)
                & (t_equity_f <= t_bench_max))
    if not np.any(in_range):
        return out
    i0 = int(np.argmax(in_range))  # first True
    if i0 >= len(equity_f) or not np.isfinite(equity_f[i0]):
        return out
    bench_close_c = bench_ys[0]
    bench_interp = np.interp(t_equity_f, t_bench_c, bench_close_c)
    anchor_bench = bench_interp[i0]
    if not np.isfinite(anchor_bench) or anchor_bench <= 0:
        return out
    scale = float(equity_f[i0]) / float(anchor_bench)
    result = bench_interp * scale
    result[~in_range] = np.nan
    return result
def bar_widths(t, frac=0.8):
    """Per-bar width so irregular gaps (weekends) don't overlap adjacent bars."""
    t = np.asarray(t, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([frac * 86400.0])
    gaps = np.diff(t)
    left = np.empty(n)
    right = np.empty(n)
    left[1:] = gaps
    left[0] = gaps[0]
    right[:-1] = gaps
    right[-1] = gaps[-1]
    return frac * np.minimum(left, right)
def window_start_index(t, cutoff):
    t = np.asarray(t, dtype=float)
    return int(np.searchsorted(t, cutoff, side='left'))
def nearest_index(t_sorted, x):
    t_sorted = np.asarray(t_sorted, dtype=float)
    n = len(t_sorted)
    if n == 0:
        return 0
    i = int(np.searchsorted(t_sorted, x))
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    before = x - t_sorted[i - 1]
    after = t_sorted[i] - x
    return i - 1 if before <= after else i
def fingerprint(*arrays):
    h = hashlib.blake2b(digest_size=8)
    for a in arrays:
        if isinstance(a, np.ndarray):
            h.update(repr(a.shape).encode())
            h.update(str(a.dtype).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(repr(a).encode())
    return h.hexdigest()
# 5) OHLC aggregation
def ohlc_aggregate(t, o, h, l, c, v, factor: int):
    """Group consecutive `factor` bars: t=first, o=first, h=max, l=min,
    c=last, v=sum. Alpaca bar timestamps mark bar START. Final partial
    bucket is included."""
    t = np.asarray(t, dtype=float)
    o = np.asarray(o, dtype=float)
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)
    c = np.asarray(c, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(t)
    factor = max(int(factor), 1)
    if factor <= 1 or n == 0:
        return t, o, h, l, c, v
    n_buckets = n // factor
    n_full = n_buckets * factor
    def _agg(arr, reducer):
        parts = []
        if n_buckets > 0:
            parts.append(reducer(arr[:n_full].reshape(n_buckets, factor)))
        if n_full < n:
            parts.append(reducer(arr[n_full:][None, :]))
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    t2 = _agg(t, lambda a: a[:, 0])
    o2 = _agg(o, lambda a: a[:, 0])
    h2 = _agg(h, lambda a: np.nanmax(a, axis=1))
    l2 = _agg(l, lambda a: np.nanmin(a, axis=1))
    c2 = _agg(c, lambda a: a[:, -1])
    v2 = _agg(v, lambda a: np.nansum(a, axis=1))
    return t2, o2, h2, l2, c2, v2
# 5b) Price overlays (SMA / Wilder ATR) — pure functions, called by
# build_price_view AFTER windowing/aggregation so they line up bar-for-bar
# with whatever is actually displayed.
def trailing_sma(close, n):
    """Trailing simple moving average of `close` over an `n`-bar window.
    NaN for the first n-1 positions (not enough bars yet for a full
    window) — the render layer is responsible for masking NaN, this never
    fabricates a partial-window average."""
    close = np.asarray(close, dtype=float)
    m = len(close)
    out = np.full(m, np.nan, dtype=float)
    n = int(n)
    if n <= 0 or m < n:
        return out
    csum = np.cumsum(close)
    out[n - 1] = csum[n - 1] / n
    if m > n:
        out[n:] = (csum[n:] - csum[:m - n]) / n
    return out
def wilder_atr(high, low, close, length=14):
    """Wilder's Average True Range — TRUE Wilder recurrence, not a plain
    rolling mean of true range:
      tr[0] = high[0] - low[0]
      tr[i] = max(high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]|)
      atr[length-1] = mean(tr[:length])                     (seed)
      atr[i]        = (atr[i-1] * (length-1) + tr[i]) / length,  i >= length
    NaN for the first length-1 positions (not enough bars for the seed)."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    m = len(close)
    out = np.full(m, np.nan, dtype=float)
    if m == 0:
        return out
    tr = np.empty(m, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, m):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    length = int(length)
    if length <= 0 or m < length:
        return out
    seed = float(np.mean(tr[:length]))
    out[length - 1] = seed
    prev = seed
    for i in range(length, m):
        prev = (prev * (length - 1) + tr[i]) / length
        out[i] = prev
    return out
# 6) View builders
def _f64():
    return np.array([], dtype=np.float64)
@dataclass
class EquityView:
    status: ChartStatus
    ts: np.ndarray = field(default_factory=_f64)
    equity: np.ndarray = field(default_factory=_f64)
    hwm: np.ndarray = field(default_factory=_f64)
    pnl_ts: np.ndarray = field(default_factory=_f64)
    pnl: np.ndarray = field(default_factory=_f64)
    pnl_widths: np.ndarray = field(default_factory=_f64)
    stats: "dict | None" = None
    x_range: "tuple | None" = None
    y_range: "tuple | None" = None
    fingerprint: str = ''
def build_equity_view(history: dict, now=None, max_points=1500) -> EquityView:
    now = now if now is not None else time.time()
    equity_raw = history.get('equity') or []
    ts_raw = history.get('timestamp') or []
    pnl_raw = history.get('profit_loss') or []
    if len(equity_raw) == 0 or len(ts_raw) == 0:
        return EquityView(status=ChartStatus(
            status=EMPTY, message='no history from Alpaca', updated_at=now))
    ts_full, ys, eq_note = coerce_xy(ts_raw, equity_raw)
    equity_full = ys[0] if ys else _f64()
    if len(ts_full) == 0:
        return EquityView(status=ChartStatus(
            status=EMPTY, message=eq_note or 'no history from Alpaca', updated_at=now))
    hwm_full = compute_hwm(equity_full)
    status_kind = OK
    note = eq_note
    pnl_ts_full = _f64()
    pnl_full = _f64()
    if len(pnl_raw) > 0:
        pnl_ts_full, pnl_ys, pnl_note = coerce_xy(ts_raw, pnl_raw)
        pnl_full = pnl_ys[0] if pnl_ys else _f64()
        if pnl_note:
            status_kind = PARTIAL
            note = f'P&L truncated to {len(pnl_full)} days'
    stats = perf_stats(equity_full, pnl_full, t=ts_full)
    idx = lttb_indices(ts_full, equity_full, max_points)
    ts_view = ts_full[idx]
    equity_view = equity_full[idx]
    hwm_view = hwm_full[idx]
    pnl_ts_view = pnl_ts_full
    pnl_view = pnl_full
    if len(pnl_ts_view) > 4000:
        stride = math.ceil(len(pnl_ts_view) / 4000)
        pnl_ts_view = pnl_ts_view[::stride]
        pnl_view = pnl_view[::stride]
    pnl_widths = bar_widths(pnl_ts_view)
    combined = np.concatenate([equity_view, hwm_view])
    y_min = float(np.min(combined))
    y_max = float(np.max(combined))
    pad = (y_max - y_min) * 0.05 if y_max > y_min else (y_max * 0.02 if y_max else 0.0)
    y_range = (y_min - pad, y_max + pad)
    x_range = (float(ts_view[0]), float(ts_view[-1]))
    fp = fingerprint(ts_view, equity_view, pnl_view)
    return EquityView(
        status=ChartStatus(status=status_kind, note=note, updated_at=now),
        ts=ts_view, equity=equity_view, hwm=hwm_view,
        pnl_ts=pnl_ts_view, pnl=pnl_view, pnl_widths=pnl_widths,
        stats=stats, x_range=x_range, y_range=y_range, fingerprint=fp,
    )
@dataclass
class PriceView:
    status: ChartStatus
    mode: str = 'none'
    t: np.ndarray = field(default_factory=_f64)
    o: np.ndarray = field(default_factory=_f64)
    h: np.ndarray = field(default_factory=_f64)
    l: np.ndarray = field(default_factory=_f64)
    c: np.ndarray = field(default_factory=_f64)
    w: np.ndarray = field(default_factory=_f64)
    up: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    line_t: np.ndarray = field(default_factory=_f64)
    line_c: np.ndarray = field(default_factory=_f64)
    vol_t: np.ndarray = field(default_factory=_f64)
    vol_v: np.ndarray = field(default_factory=_f64)
    vol_w: np.ndarray = field(default_factory=_f64)
    has_volume: bool = False
    markers: dict = field(default_factory=dict)
    overlays: dict = field(default_factory=dict)
    x_range: "tuple | None" = None
    y_range: "tuple | None" = None
    vol_y_range: "tuple | None" = None
    fingerprint: str = ''
def _empty_markers():
    return {'entry_t': _f64(), 'entry_p': _f64(), 'exit_t': _f64(),
            'exit_p': _f64(), 'exit_pnl': _f64()}
def _filter_markers(markers_raw, cutoff):
    def _get(key):
        try:
            return np.asarray((markers_raw or {}).get(key, []), dtype=float)
        except (TypeError, ValueError):
            return _f64()
    entry_t, entry_p = _get('entry_t'), _get('entry_p')
    exit_t, exit_p, exit_pnl = _get('exit_t'), _get('exit_p'), _get('exit_pnl')
    ne = min(len(entry_t), len(entry_p))
    entry_t, entry_p = entry_t[:ne], entry_p[:ne]
    nx = min(len(exit_t), len(exit_p))
    exit_t, exit_p = exit_t[:nx], exit_p[:nx]
    exit_pnl = exit_pnl[:nx] if len(exit_pnl) >= nx else np.concatenate(
        [exit_pnl, np.full(nx - len(exit_pnl), np.nan)])
    em = entry_t >= cutoff
    xm = exit_t >= cutoff
    return {
        'entry_t': entry_t[em], 'entry_p': entry_p[em],
        'exit_t': exit_t[xm], 'exit_p': exit_p[xm], 'exit_pnl': exit_pnl[xm],
    }
_PRICE_OVERLAY_NAMES = frozenset({'sma20', 'sma50', 'atr_band'})

def _build_price_overlays(overlays, h, l, c, atr_mult):
    """{'sma20': arr, 'sma50': arr, 'atr_band': (upper, lower)} for whichever
    of `overlays` were requested, computed from the FINAL post-window/
    post-aggregation h/l/c (so every overlay lines up bar-for-bar with the
    candles actually drawn). Unknown names are silently ignored — a typo in
    a GUI-side literal must never crash a chart. No VWAP / new oscillators
    (kill-list) — sma20/sma50/atr_band is the complete set."""
    out = {}
    requested = _PRICE_OVERLAY_NAMES.intersection(overlays)
    if 'sma20' in requested:
        out['sma20'] = trailing_sma(c, 20)
    if 'sma50' in requested:
        out['sma50'] = trailing_sma(c, 50)
    if 'atr_band' in requested:
        atr = wilder_atr(h, l, c, 14)
        out['atr_band'] = (c + atr_mult * atr, c - atr_mult * atr)
    return out
def build_price_view(data: dict, zoom: str, now=None, max_candles=300,
                      max_line_points=1500, overlays=(), atr_mult=2.0) -> PriceView:
    now = now if now is not None else time.time()
    error = data.get('error')
    if error:
        return PriceView(status=ChartStatus(status=ERROR, message=str(error),
                                             updated_at=now), mode='none')
    closes_raw = data.get('closes') or []
    ts_raw = data.get('timestamps') or []
    if len(closes_raw) == 0 or len(ts_raw) == 0:
        return PriceView(status=ChartStatus(status=EMPTY, message='no data',
                                             updated_at=now), mode='none')
    opens_raw = data.get('opens')
    highs_raw = data.get('highs')
    lows_raw = data.get('lows')
    vols_raw = data.get('volumes')
    have_ohlc = (opens_raw is not None and highs_raw is not None and lows_raw is not None
                 and len(opens_raw) == len(ts_raw) and len(highs_raw) == len(ts_raw)
                 and len(lows_raw) == len(ts_raw))
    have_vol = vols_raw is not None and len(vols_raw) == len(ts_raw)
    extra = []
    if have_ohlc:
        extra += [opens_raw, highs_raw, lows_raw]
    if have_vol:
        extra += [vols_raw]
    ts_full, ys, note = coerce_xy(ts_raw, closes_raw, *extra)
    if len(ts_full) == 0:
        return PriceView(status=ChartStatus(status=EMPTY, message=note or 'no data',
                                             updated_at=now), mode='none')
    closes_full = ys[0]
    opens_full = highs_full = lows_full = None
    j = 1
    if have_ohlc:
        opens_full, highs_full, lows_full = ys[1], ys[2], ys[3]
        j = 4
    vols_full = ys[j] if have_vol else None
    cutoff = now - ZOOM_DAYS.get(zoom, 30) * 86400
    i0 = window_start_index(ts_full, cutoff)
    if i0 >= len(ts_full):
        return PriceView(status=ChartStatus(status=EMPTY, message='no data in window',
                                             updated_at=data.get('cached_at') or now),
                          mode='none')
    t = ts_full[i0:]
    c = closes_full[i0:]
    o = opens_full[i0:] if have_ohlc else None
    h = highs_full[i0:] if have_ohlc else None
    l = lows_full[i0:] if have_ohlc else None
    v = vols_full[i0:] if have_vol else None
    agg_note = ''
    vol_t = vol_v = vol_w = _f64()
    has_volume = False
    up = np.array([], dtype=bool)
    w = _f64()
    line_t = line_c = _f64()
    if have_ohlc:
        mode = 'candles'
        n = len(t)
        if n > max_candles:
            factor = math.ceil(n / max_candles)
            v_in = v if v is not None else np.zeros(n)
            t, o, h, l, c, v_agg = ohlc_aggregate(t, o, h, l, c, v_in, factor)
            if v is not None:
                v = v_agg
            agg_note = f'aggregated {factor}×'
        w = bar_widths(t)
        up = c >= o
        if v is not None:
            has_volume = True
            vol_t, vol_v, vol_w = t, v, w
        lo_arr, hi_arr = l, h
        x0, x1 = float(t[0]), float(t[-1])
    else:
        mode = 'line'
        _idx, line_t, line_c = downsample(t, c, max_line_points)
        if v is not None:
            has_volume = True
            vol_t = t
            vol_v = v
            if len(vol_t) > 4000:
                stride = math.ceil(len(vol_t) / 4000)
                vol_t = vol_t[::stride]
                vol_v = vol_v[::stride]
            vol_w = bar_widths(vol_t)
        lo_arr, hi_arr = line_c, line_c
        x0, x1 = float(line_t[0]), float(line_t[-1])
    overlays_out = {}
    if overlays and mode == 'candles':
        overlays_out = _build_price_overlays(overlays, h, l, c, atr_mult)
    if len(lo_arr):
        y_min = float(np.min(lo_arr))
        y_max = float(np.max(hi_arr))
        pad = (y_max - y_min) * 0.05 if y_max > y_min else (y_max * 0.02 if y_max else 0.0)
        y_range = (y_min - pad, y_max + pad)
        x_range = (x0, x1)
    else:
        y_range = None
        x_range = None
    vol_y_range = None
    if has_volume and len(vol_v):
        vmax = float(np.max(vol_v))
        vol_y_range = (0.0, vmax * 1.05 if vmax > 0 else 1.0)
    markers = _filter_markers(data.get('markers'), cutoff)
    combined_note = '; '.join(x for x in (note, agg_note) if x)
    fp = fingerprint(t if mode == 'candles' else line_t,
                      c if mode == 'candles' else line_c,
                      vol_v, markers['entry_t'], markers['exit_t'], mode)
    return PriceView(
        status=ChartStatus(status=OK, note=combined_note,
                            updated_at=data.get('cached_at') or now),
        mode=mode, t=t, o=(o if o is not None else _f64()),
        h=(h if h is not None else _f64()), l=(l if l is not None else _f64()),
        c=c, w=w, up=up, line_t=line_t, line_c=line_c,
        vol_t=vol_t, vol_v=vol_v, vol_w=vol_w, has_volume=has_volume,
        markers=markers, overlays=overlays_out, x_range=x_range, y_range=y_range,
        vol_y_range=vol_y_range, fingerprint=fp,
    )
# 7) Palette (computed contrast, not eyeballed)
def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')[:6]
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
def _rgb_to_hex(rgb):
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return f'#{r:02x}{g:02x}{b:02x}'
def _linearize(v):
    v = v / 255.0
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
def rel_luminance(hex_color):
    r, g, b = _hex_to_rgb(hex_color)
    return 0.2126 * _linearize(r) + 0.7152 * _linearize(g) + 0.0722 * _linearize(b)
def contrast_ratio(a, b):
    la = rel_luminance(a) + 0.05
    lb = rel_luminance(b) + 0.05
    return la / lb if la > lb else lb / la
def mix(a, b, t):
    ra, ga, ba = _hex_to_rgb(a)
    rb, gb, bb = _hex_to_rgb(b)
    return _rgb_to_hex((ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t))
def _hue_deg(hex_color):
    r, g, b = (v / 255.0 for v in _hex_to_rgb(hex_color))
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d == 0:
        return 0.0
    if mx == r:
        hh = ((g - b) / d) % 6
    elif mx == g:
        hh = (b - r) / d + 2
    else:
        hh = (r - g) / d + 4
    return hh * 60.0
def _hue_dist(a_deg, b_deg):
    diff = abs(a_deg - b_deg) % 360
    return min(diff, 360 - diff)
def ensure_contrast(color, bg, min_ratio=3.0, step=0.08):
    # Default floor 3.0 = WCAG-AA for graphical objects / large text (was 2.5).
    result = color
    toward = '#ffffff' if rel_luminance(bg) < 0.5 else '#000000'
    for _ in range(12):
        if contrast_ratio(result, bg) >= min_ratio:
            return result
        result = mix(result, toward, step)
    return result
def separate_luminance(up, down, bg, min_gap=0.08, min_ratio=3.0, step=0.08,
                       max_iter=24):
    """Force a brightness gap between the up/down chart colors so direction is
    legible under total color-blindness (CVD), not only by hue. ALWAYS applied
    (not just the rare hue-collapse case): if the relative-luminance gap is
    under `min_gap`, lighten `up` toward white and darken `down` toward black
    one `step` at a time — but only in the direction that KEEPS each color at
    >= `min_ratio` contrast vs `bg`. On a dark bg the gap opens mostly by
    lightening up (its contrast grows); on a light bg mostly by darkening down;
    if neither can move without breaking its contrast floor, stop (best effort).
    """
    for _ in range(max_iter):
        if abs(rel_luminance(up) - rel_luminance(down)) >= min_gap:
            break
        moved = False
        cand_up = mix(up, '#ffffff', step)
        if contrast_ratio(cand_up, bg) >= min_ratio:
            up, moved = cand_up, True
        cand_down = mix(down, '#000000', step)
        if contrast_ratio(cand_down, bg) >= min_ratio:
            down, moved = cand_down, True
        if not moved:
            break
    return up, down
def derive_chart_palette(theme: dict) -> dict:
    """theme: the 13 GUI role hexes ('#rrggbb'). Single source of theme ->
    chart-color truth — no per-theme special cases."""
    bg = theme['bg_dark']
    up = ensure_contrast(theme['green'], bg)
    down = ensure_contrast(theme['red'], bg)
    if _hue_dist(_hue_deg(up), _hue_deg(down)) < 40 and contrast_ratio(up, down) < 1.4:
        down = ensure_contrast(mix(down, '#ff9900', 0.5), bg)
    # Always-on CVD luminance separation (not only the hue-collapse guard above).
    up, down = separate_luminance(up, down, bg)
    equity = ensure_contrast(theme['accent'], bg)
    crosshair = ensure_contrast(theme['yellow'], bg, 2.0)
    return {
        'bg': bg,
        'fg': theme['white'],
        'grid': theme['muted'],
        'up': up,
        'down': down,
        'equity': equity,
        'hwm': theme['muted'],
        'dd_fill': down + '38',
        'vol_up': up + '73',
        'vol_down': down + '73',
        'crosshair': crosshair,
        'marker_entry': up,
        'marker_exit': ensure_contrast(theme['yellow'], bg),
        'title_warn': crosshair,
        'title_err': down,
        'tile_neutral': mix(theme['bg_hover'], theme['muted'], 0.25),
    }
def heatmap_style(chg_pct, palette):
    """(bg_hex, text_hex) for a heatmap tile; sqrt intensity gives small
    movers visible differentiation vs a linear scale."""
    intensity = math.sqrt(min(abs(chg_pct) / 5.0, 1.0))
    if chg_pct == 0:
        bg = palette['tile_neutral']
    elif chg_pct > 0:
        bg = mix(palette['tile_neutral'], palette['up'], intensity)
    else:
        bg = mix(palette['tile_neutral'], palette['down'], intensity)
    text = '#000000' if contrast_ratio('#000000', bg) >= contrast_ratio('#ffffff', bg) else '#ffffff'
    return bg, text
# 8) Trade markers (journals/*.jsonl — mtime-cached, corrupt-line tolerant)
_marker_cache = {}

def load_trade_markers(symbol: str, journal_dir, since_ts: float, now=None) -> dict:
    """Read journals/YYYY-MM-DD.jsonl from date(since_ts) to today (cap 370
    files), extract buy/sell rows for `symbol`. Never raises — file IO only
    happens here, and only gui.py's DataFetcher thread may call this."""
    try:
        now = now if now is not None else time.time()
        jdir = Path(journal_dir)
        if not jdir.exists():
            return _empty_markers()
        start_date = datetime.date.fromtimestamp(since_ts)
        end_date = datetime.date.fromtimestamp(now)
        span = max((end_date - start_date).days, 0)
        span = min(span, 369)
        dates = [end_date - datetime.timedelta(days=i) for i in range(span + 1)]
        files = [jdir / f"{d.isoformat()}.jsonl" for d in dates]
        files = [p for p in files if p.exists()]
        cache_key = (str(jdir), symbol)
        mtimes = tuple(sorted((str(p), p.stat().st_mtime) for p in files))
        cached = _marker_cache.get(cache_key)
        if cached and cached[0] == mtimes:
            return cached[1]
        entry_t, entry_p, exit_t, exit_p, exit_pnl = [], [], [], [], []
        for p in files:
            try:
                with open(p) as f:
                    lines = f.readlines()
            except OSError:
                continue
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('symbol') != symbol:
                    continue
                action = row.get('action')
                if action not in ('buy', 'sell'):
                    continue
                try:
                    ts = datetime.datetime.fromisoformat(row['ts']).timestamp()
                except Exception:
                    continue
                if ts < since_ts:
                    continue
                price_src = row.get('fill_price') or row.get('decision_price')
                if not price_src:
                    continue
                try:
                    price = float(price_src)
                except (TypeError, ValueError):
                    continue
                if action == 'buy':
                    entry_t.append(ts)
                    entry_p.append(price)
                else:
                    exit_t.append(ts)
                    exit_p.append(price)
                    pnl_raw = row.get('pnl_pct')
                    try:
                        pnl = float(pnl_raw) if pnl_raw else float('nan')
                    except (TypeError, ValueError):
                        pnl = float('nan')
                    exit_pnl.append(pnl)
        result = {
            'entry_t': np.array(entry_t, dtype=np.float64),
            'entry_p': np.array(entry_p, dtype=np.float64),
            'exit_t': np.array(exit_t, dtype=np.float64),
            'exit_p': np.array(exit_p, dtype=np.float64),
            'exit_pnl': np.array(exit_pnl, dtype=np.float64),
        }
        _marker_cache[cache_key] = (mtimes, result)
        return result
    except Exception:
        return _empty_markers()


def sizing_stack_summary(journal_dir, since_ts, now=None) -> dict:
    """Aggregate the buy-journal sizing decomposition (base_loop journals
    buy_rec['sizing'] with detail['v2'] + detail['stack'] since c26 S3) so
    the GUI can show the v2 shadow composition beside legacy. Read-only,
    fully defensive; zeroed shape when nothing found."""
    def _median(vals):
        vals = [v for v in vals if isinstance(v, float) and math.isfinite(v)]
        return float(np.median(vals)) if vals else None

    out = {'n_buy_rows': 0, 'n_with_sizing': 0, 'n_with_v2': 0,
           'stack_counts': {}, 'legacy_tilt_median': None,
           'v2_tilt_median': None, 'tilt_divergence_median': None,
           'v2_min_src_counts': {}}
    try:
        now = now if now is not None else time.time()
        jdir = Path(journal_dir)
        if not jdir.exists():
            return out
        start_date = datetime.date.fromtimestamp(since_ts)
        end_date = datetime.date.fromtimestamp(now)
        span = max((end_date - start_date).days, 0)
        span = min(span, 369)
        dates = [end_date - datetime.timedelta(days=i) for i in range(span + 1)]
        files = [jdir / f"{d.isoformat()}.jsonl" for d in dates]
        files = [p for p in files if p.exists()]
        legacy_tilts, v2_tilts, divergences = [], [], []
        for p in files:
            try:
                with open(p) as f:
                    lines = f.readlines()
            except OSError:
                continue
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get('action') != 'buy':
                    continue
                out['n_buy_rows'] += 1
                detail = row.get('sizing')
                if not isinstance(detail, dict):
                    continue
                out['n_with_sizing'] += 1
                legacy_tilt = float('nan')
                try:
                    if detail.get('tilt') is not None:
                        legacy_tilt = float(detail.get('tilt'))
                        legacy_tilts.append(legacy_tilt)
                except (TypeError, ValueError):
                    pass
                stack = detail.get('stack')
                if stack is not None:
                    try:
                        key = str(stack)
                        out['stack_counts'][key] = out['stack_counts'].get(key, 0) + 1
                    except Exception:
                        pass
                v2 = detail.get('v2')
                if isinstance(v2, dict) and v2.get('tilt') is not None:
                    out['n_with_v2'] += 1
                    try:
                        v2_tilt = float(v2.get('tilt'))
                        v2_tilts.append(v2_tilt)
                        if math.isfinite(v2_tilt) and math.isfinite(legacy_tilt):
                            divergences.append(v2_tilt - legacy_tilt)
                    except (TypeError, ValueError):
                        pass
                    src = v2.get('min_src')
                    if src is not None:
                        try:
                            skey = str(src)
                            out['v2_min_src_counts'][skey] = \
                                out['v2_min_src_counts'].get(skey, 0) + 1
                        except Exception:
                            pass
        out['legacy_tilt_median'] = _median(legacy_tilts)
        out['v2_tilt_median'] = _median(v2_tilts)
        out['tilt_divergence_median'] = _median(divergences)
        return out
    except Exception:
        return out


# --- report render-models / formatters (pure; consumed by gui.py) ---

def gate_verdict_class(verdict, insufficient_n) -> str:
    """Map a decision_report gate/signal-exit verdict string (+ its
    insufficient_n flag) to a render class. Consumer-side mapping only —
    the strings come from decision_report._gate_verdict/_signal_exit_verdict."""
    try:
        v = str(verdict) if verdict is not None else ''
        if insufficient_n or v.startswith('insufficient'):
            return 'insufficient'
        if v.startswith('REVIEW'):
            return 'review'
        if v.startswith('OK'):
            return 'ok'
        if v.startswith('NO CHANGE'):
            return 'no_change'
        if v.startswith('CHANGE'):
            return 'change'
        return 'inconclusive'
    except Exception:
        return 'inconclusive'


def gate_panel_model(rep) -> dict:
    """Pure render model for the GUI gate-attribution panel from a parsed
    decision_report.json dict. Never raises; tolerates {}, None, and the
    _write_stale_report stub."""
    out = {'stale': False, 'stale_reason': None, 'representative': None,
           'quality_line': None, 'generated': None, 'gates': [],
           'signal_exit': None}
    try:
        if not isinstance(rep, dict):
            return out
        out['stale'] = bool(rep.get('stale'))
        if out['stale']:
            out['stale_reason'] = (str(rep['stale_reason'])
                                   if rep.get('stale_reason') is not None
                                   else 'no API when generated — '
                                        'counterfactuals not priced')
        gen = rep.get('generated')
        out['generated'] = str(gen) if gen is not None else None
        q = rep.get('quality')
        if isinstance(q, dict):
            rep_flag = q.get('representative')
            out['representative'] = (bool(rep_flag) if rep_flag is not None
                                     else None)
            try:
                priced = int(q.get('priced', 0))
                unpriced = int(q.get('unpriced', 0))
                rate = float(q.get('unpriced_rate', 0.0))
                fetch_failed = int(q.get('fetch_failed', 0))
                out['quality_line'] = (f"priced {priced} · unpriced {unpriced} "
                                       f"({rate:.0%}) · fetch-failed {fetch_failed}")
            except (TypeError, ValueError):
                pass
        gates = rep.get('gates')
        if isinstance(gates, dict):
            for name, g in gates.items():
                if str(name).startswith('_') or not isinstance(g, dict):
                    continue
                insufficient = bool(g.get('insufficient_n'))
                verdict = g.get('verdict')
                out['gates'].append({
                    'name': str(name),
                    'verdict': str(verdict) if verdict is not None else None,
                    'verdict_class': gate_verdict_class(verdict, insufficient),
                    'n': g.get('vetoes_priced'),
                    'raw': g.get('vetoes_raw'),
                    'mean': g.get('counterfactual_mean_net_pct'),
                    'ci90': g.get('ci90'),
                    'hit': g.get('counterfactual_hit_rate'),
                    'saved': g.get('saved_total_pct'),
                    'insufficient_n': insufficient,
                })
        se = rep.get('signal_exit')
        if isinstance(se, dict):
            try:
                priced = int(se.get('priced') or 0)
            except (TypeError, ValueError):
                priced = 0
            if priced > 0:
                verdict = se.get('verdict')
                out['signal_exit'] = {
                    'verdict': str(verdict) if verdict is not None else None,
                    'verdict_class': gate_verdict_class(
                        verdict, bool(se.get('insufficient_n'))),
                    'n': se.get('n_signal_sells'),
                    'priced': priced,
                }
        return out
    except Exception:
        return out


def meta_panel_model(meta, refused) -> dict:
    """Pure render model for the meta-gate panel from parsed
    {p}meta_meta.json + {p}meta_refused.json (either may be None)."""
    out = {'present': False, 'pred_source': None, 'trained_at': None,
           'val_auc': None, 'n_trades': None, 'oof_note': None,
           'refused': False, 'refused_at': None, 'refused_reasons': []}
    try:
        if isinstance(meta, dict):
            out['present'] = True
            ps = meta.get('pred_source')
            out['pred_source'] = str(ps) if ps is not None else None
            ta = meta.get('trained_at')
            out['trained_at'] = str(ta) if ta is not None else None
            out['val_auc'] = meta.get('val_auc')
            out['n_trades'] = meta.get('n_trades')
            oof = meta.get('oof')
            if isinstance(oof, dict):
                fr = oof.get('fallback_reason')
                if fr is not None:
                    out['oof_note'] = str(fr)
        elif meta is not None:
            out['present'] = True
        if refused is not None:
            out['refused'] = True
            if isinstance(refused, dict):
                ra = refused.get('refused_at')
                out['refused_at'] = str(ra) if ra is not None else None
                reasons = refused.get('reasons')
                if isinstance(reasons, (list, tuple)):
                    out['refused_reasons'] = [str(r) for r in reasons[:3]]
                elif reasons is not None:
                    out['refused_reasons'] = [str(reasons)]
        return out
    except Exception:
        return out


def _num(v, fmt='.2f', default='n/a'):
    """nan-safe number formatter for the report summaries."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return format(f, fmt)
    except (TypeError, ValueError):
        return default


def format_llm_eval_summary(rep) -> str:
    """One-paragraph text summary of llm_eval_report.json for the GUI report
    dialog. Never raises; tolerates the no_data stub and {}."""
    try:
        if not isinstance(rep, dict) or not rep:
            return 'LLM EVAL — no data: empty/unreadable report'
        if rep.get('verdict') == 'no_data':
            return f"LLM EVAL — no data: {rep.get('reason', '?')}"
        lines = []
        meta = rep.get('meta') if isinstance(rep.get('meta'), dict) else {}
        days = meta.get('days')
        gen = meta.get('generated_at')
        head = 'LLM EVAL'
        if days is not None:
            head += f" ({days}d)"
        if gen is not None:
            head += f"  generated {gen}"
        lines.append(head)
        lines.append(f"verdict: {rep.get('verdict', '?')}")
        inc = rep.get('incremental') if isinstance(rep.get('incremental'), dict) else {}
        n = inc.get('n', rep.get('n'))
        nwp = rep.get('n_with_pred')
        if n is not None or nwp is not None:
            lines.append(f"n={n if n is not None else '?'}"
                         + (f" (with pred {nwp})" if nwp is not None else ''))
        enc = inc.get('encompassing') if isinstance(inc.get('encompassing'), dict) else {}
        if enc:
            est = enc.get('estimator', '?')
            tag = ' (DK primary)' if est == 'driscoll_kraay' else ''
            lines.append(f"encompassing b2={_num(enc.get('b2_s'), '+.4f')} "
                         f"p={_num(enc.get('p_value'), '.3f')} "
                         f"[{est}{tag}]")
        leg = inc.get('legacy_b2') if isinstance(inc.get('legacy_b2'), dict) else {}
        if leg:
            lines.append(f"legacy rows-HAC (deprecated): "
                         f"b2={_num(leg.get('b2_s'), '+.4f')} "
                         f"p={_num(leg.get('p_value'), '.3f')}")
        sl = rep.get('spend_ledger') if isinstance(rep.get('spend_ledger'), dict) else {}
        if sl:
            lines.append(f"spend: ${_num(sl.get('daily_cost_usd'), '.2f')}/"
                         f"${_num(sl.get('daily_cost_limit_usd'), '.2f')} daily "
                         f"(read_ok={sl.get('cost_read_ok')}); window "
                         f"${_num(sl.get('window_journaled_cost_usd'), '.2f')} "
                         f"over {sl.get('n_entries_with_cost', '?')} entries")
            lines.append(f"benefit: tilt {_num(sl.get('llm_tilt_bps_per_trade'), '+.1f')} "
                         f"bps/trade; veto avoided "
                         f"{_num(sl.get('veto_avoided_ret_pct_sum'), '+.2f')}% fwd ret")
        if rep.get('veto_counterfactual_pct') is not None:
            lines.append(f"veto counterfactual: "
                         f"{_num(rep.get('veto_counterfactual_pct'), '+.2f')}%")
        return '\n'.join(lines)
    except Exception:
        return 'LLM EVAL — summary unavailable'


def format_advisor_summary(rep) -> str:
    """One-paragraph text summary of llm_advisor_report.json. Never raises."""
    try:
        if not isinstance(rep, dict) or not rep:
            return 'LLM ADVISOR — no data: empty/unreadable report'
        if rep.get('verdict') == 'no_data':
            return f"LLM ADVISOR — no data: {rep.get('reason', '?')}"
        lines = []
        meta = rep.get('meta') if isinstance(rep.get('meta'), dict) else {}
        gen = meta.get('generated_at')
        lines.append('LLM ADVISOR' + (f"  generated {gen}" if gen else ''))
        lines.append(f"verdict: {rep.get('verdict', '?')}")
        nt, nc = rep.get('n_total'), rep.get('n_calibratable')
        if nt is not None or nc is not None:
            lines.append(f"n_total={nt}  n_calibratable={nc}")
        ss = rep.get('signal_source')
        if ss is not None:
            lines.append(f"signal={ss}  p_up present "
                         f"{_num(rep.get('p_up_present_frac'), '.0%')}")
        if rep.get('n_dedup_hit') is not None:
            lines.append(f"dedup: {rep.get('n_dedup_hit')} hits "
                         f"({_num(rep.get('dedup_hit_frac'), '.0%')}), "
                         f"{rep.get('n_unique_llm_calls', '?')} unique calls")
        bm = rep.get('by_model')
        if isinstance(bm, dict) and bm:
            lines.append(f"models: {len(bm)}")
        ipo = rep.get('incremental_p_up_only')
        if isinstance(ipo, dict) and ipo.get('verdict') is not None:
            lines.append(f"p_up-only verdict: {ipo.get('verdict')}")
        return '\n'.join(lines)
    except Exception:
        return 'LLM ADVISOR — summary unavailable'


def format_execution_summary(rep) -> str:
    """One-paragraph text summary of execution_report.json. Never raises."""
    try:
        if not isinstance(rep, dict) or not rep:
            return 'EXECUTION — no data: empty/unreadable report'
        lines = []
        head = 'EXECUTION'
        if rep.get('window_days') is not None:
            head += f" ({rep.get('window_days')}d)"
        if rep.get('generated_at') is not None:
            head += f"  generated {rep.get('generated_at')}"
        lines.append(head)
        rows = [(k, v) for k, v in rep.items()
                if isinstance(k, str) and '/' in k and isinstance(v, dict)
                and 'n' in v]
        if not rows:
            lines.append('no fills with slippage data in window')
            return '\n'.join(lines)
        if rep.get('overall_mean_bps') is not None:
            lines.append(f"overall mean {_num(rep.get('overall_mean_bps'), '+.1f')} bps")
        for k, v in rows:
            lines.append(f"{k}  n={v.get('n', '?')}  "
                         f"mean={_num(v.get('mean_bps'), '+.1f')}bps  "
                         f"p90={_num(v.get('p90_bps'), '+.1f')}")
        return '\n'.join(lines)
    except Exception:
        return 'EXECUTION — summary unavailable'


def format_sizing_stack(summary) -> str:
    """One-line text summary of sizing_stack_summary()'s dict. Never raises."""
    try:
        if not isinstance(summary, dict) or not summary.get('n_with_sizing'):
            return ('no sizing decomposition journaled '
                    '(CONVICTION_JOURNAL_ENABLED off or no buys)')
        sc = summary.get('stack_counts') or {}
        stack_txt = ' '.join(f"{k} {v}" for k, v in sorted(sc.items())) or '—'
        src = summary.get('v2_min_src_counts') or {}
        src_txt = ' '.join(f"{v}×{k}" for k, v in sorted(src.items())) or '—'
        return (f"sizing stack (30d buys): n={summary.get('n_buy_rows', 0)} "
                f"({summary.get('n_with_sizing', 0)} with sizing, "
                f"{summary.get('n_with_v2', 0)} with v2)  "
                f"legacy tilt med={_num(summary.get('legacy_tilt_median'), '.2f')}  "
                f"v2 tilt med={_num(summary.get('v2_tilt_median'), '.2f')}  "
                f"divergence med={_num(summary.get('tilt_divergence_median'), '+.2f')}  "
                f"stack: {stack_txt}  min_src: {src_txt}")
    except Exception:
        return 'sizing stack — summary unavailable'
