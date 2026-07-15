"""All chart math for gui.py.

MUST stay free of PySide6 / pyqtgraph / pandas — unit-tested on the dev Mac with
numpy + stdlib only. gui.py owns every Qt/pyqtgraph object; this module hands
back plain numpy arrays, dicts, and small dataclasses that gui.py paints.
"""

import datetime
import hashlib
import json
import math
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
def perf_stats(equity, pnl):
    """Total/best/worst/max-drawdown — replicates the GUI's original
    running-peak loop semantics exactly."""
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
    return {
        'total_return': total_return,
        'best_day': best_day,
        'worst_day': worst_day,
        'max_dd_pct': float(max_dd),
    }
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
    stats = perf_stats(equity_full, pnl_full)
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
def build_price_view(data: dict, zoom: str, now=None, max_candles=300,
                      max_line_points=1500) -> PriceView:
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
        markers=markers, x_range=x_range, y_range=y_range,
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
def ensure_contrast(color, bg, min_ratio=2.5, step=0.08):
    result = color
    toward = '#ffffff' if rel_luminance(bg) < 0.5 else '#000000'
    for _ in range(12):
        if contrast_ratio(result, bg) >= min_ratio:
            return result
        result = mix(result, toward, step)
    return result
def derive_chart_palette(theme: dict) -> dict:
    """theme: the 13 GUI role hexes ('#rrggbb'). Single source of theme ->
    chart-color truth — no per-theme special cases."""
    bg = theme['bg_dark']
    up = ensure_contrast(theme['green'], bg)
    down = ensure_contrast(theme['red'], bg)
    if _hue_dist(_hue_deg(up), _hue_deg(down)) < 40 and contrast_ratio(up, down) < 1.4:
        down = ensure_contrast(mix(down, '#ff9900', 0.5), bg)
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
