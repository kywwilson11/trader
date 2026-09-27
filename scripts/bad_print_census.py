"""Wick-only bad-print census for the training stores (SIG-R2 X6).

MEASUREMENT-ONLY. Reads a parquet store with pyarrow column projection
(never the whole 110-column stock store), writes nothing but the report
files named by --out / --json. Stores are opened read-only.

The test (explicit + parametric, OBJECTIVE given the parameters):
  body     = |Close - Open| / Open < body_frac                (default 0.01)
  low-side = Low  < Open * (1 - low_frac)   AND body          (default 0.15)
  high-side= High > Open * (1 + high_frac)  AND body          (default = low_frac)
i.e. a bar that opens and closes at the same price but prints an extreme
wick. A real flash crash that recovers inside the hour also passes this
test — so the census counts CANDIDATES; a second source (another venue's
hourly Low for the same hour) decides bad print vs real move.

Per ticker it reports: count (low/high/any), the first 20 flagged bars
verbatim, the fraction of rows, the TB_Reason_{fb} class counts on the
flagged rows (column names discovered from the schema; codes from
policy_exits.REASON_NAMES), how many sit in the trailing --holdout-days
(the walk-forward/holdout region), and the max wick depth.

--footprint adds the downstream contamination: for each flagged ticker it
recomputes the High/Low-consuming features on the stored rows twice —
as stored, and with the flagged wicks repaired by the rule below — and
counts the rows whose value moves (the print's exact reach per feature).
It also reports the stored ATR at each print vs the trailing 500-bar
median ATR before it. --tb-flips (needs numba: Jetson) re-runs the
policy_exits label kernel on both versions and counts label flips.

Repair rule used for the counterfactual (mirrors the staged SIG-R2-X6
harvest guard): with medTR_t = median true range over the 24 bars
strictly BEFORE t (point-in-time; falls back to the ticker's whole-series
median TR when fewer than 5 are available),
  low-side : Low  := max(Low,  min(Open, Close) - medTR_t)
  high-side: High := min(High, max(Open, Close) + medTR_t)
No row is dropped (interior drops corrupt TB_Bars offsets — R4).

Usage:
  python scripts/bad_print_census.py [--data training_data.parquet]
      [--low-frac 0.15] [--body-frac 0.01] [--high-frac F]
      [--tickers A,B] [--holdout-days 365] [--asset crypto|stock]
      [--footprint] [--tb-flips] [--out F.txt] [--json F.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
# Mirror of policy_exits.REASON_NAMES (pinned equal by the test) — kept
# local so the census runs without importing the numba kernel module.
REASON_NAMES = {0: 'end_of_data', 1: 'hard_stop', 2: 'take_profit',
                3: 'trailing', 4: 'signal_sell', 5: 'eod_flatten',
                6: 'vertical'}
MED_TR_WINDOW = 24
MED_TR_MIN = 5


# --------------------------------------------------------------- detection
def flag_wick_prints(o, h, l, c, low_frac=0.15, body_frac=0.01,
                     high_frac=None):
    """Return (low_mask, high_mask) boolean arrays for the wick test."""
    o = np.asarray(o, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    hf = low_frac if high_frac is None else high_frac
    with np.errstate(invalid='ignore', divide='ignore'):
        ok = np.isfinite(o) & (o > 0)
        body = ok & (np.abs(c - o) / np.where(ok, o, 1.0) < body_frac)
        low = body & (l < o * (1.0 - low_frac))
        high = body & (h > o * (1.0 + hf))
    return low, high


def true_range(h, l, c):
    h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def repair_frame(g, low_mask, high_mask):
    """Counterfactual repair of ONE ticker's sorted bars (see module doc).
    Returns a copy with Low/High repaired on flagged rows only."""
    out = g.copy()
    tr = pd.Series(true_range(g['High'], g['Low'], g['Close']), index=g.index)
    med = tr.shift(1).rolling(MED_TR_WINDOW, min_periods=MED_TR_MIN).median()
    med = med.fillna(float(np.nanmedian(tr.values))).values
    o = g['Open'].values
    c = g['Close'].values
    lo = g['Low'].values.copy()
    hi = g['High'].values.copy()
    lo[low_mask] = np.maximum(lo[low_mask],
                              np.minimum(o, c)[low_mask] - med[low_mask])
    hi[high_mask] = np.minimum(hi[high_mask],
                               np.maximum(o, c)[high_mask] + med[high_mask])
    out['Low'] = lo
    out['High'] = hi
    return out


# ------------------------------------------------------------------ loading
def load_store(path, want, tickers=None):
    """Projected read. `want` = desired columns; returns (df, cols_read).
    Datetime becomes a UTC index whether stored as index or column."""
    import pyarrow.parquet as pq
    names = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in want if c in names]
    if 'Datetime' in names and 'Datetime' not in cols:
        cols.append('Datetime')
    # strings_to_categorical: the Ticker column as a dictionary keeps the
    # 2.2M-row stock sidecar well under the 600 MB worker budget.
    if tickers:
        # Streamed ticker filter: batch-wise decode + mask keeps the peak at
        # one batch plus the kept rows (a filtered read_table decodes whole
        # 770k-row stock row groups first).
        import pyarrow as pa
        import pyarrow.compute as pc
        want_t = pa.array(list(tickers))
        pf = pq.ParquetFile(path)
        parts = []
        for rb in pf.iter_batches(batch_size=65536, columns=cols):
            rb = rb.filter(pc.is_in(rb.column('Ticker'), value_set=want_t))
            if rb.num_rows:
                parts.append(rb)
        tbl = (pa.Table.from_batches(parts) if parts else
               pa.Table.from_batches([], schema=pa.schema(
                   [pf.schema_arrow.field(c) for c in cols],
                   metadata=pf.schema_arrow.metadata)))
        del parts
    else:
        tbl = pq.read_table(path, columns=cols)
    df = tbl.to_pandas(strings_to_categorical=True, self_destruct=True,
                       split_blocks=True)
    del tbl
    for col in ('Ticker', 'Src'):
        if col in df.columns and hasattr(df[col], 'cat'):
            df[col] = df[col].cat.reorder_categories(
                sorted(df[col].cat.categories))
    if 'Datetime' in df.columns:
        df = df.set_index('Datetime')
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
    return df, names


# ------------------------------------------------------------------- census
def _tb_reason_cols(columns):
    return sorted([c for c in columns if c.startswith('TB_Reason_')],
                  key=lambda s: int(s.rsplit('_', 1)[1]))


def census(df, low_frac=0.15, body_frac=0.01, high_frac=None,
           holdout_days=365, max_dates=20, t_max=None):
    """Pure census over a frame with Ticker + OHLC (+ optional TB_Reason_*).
    `t_max` (default: the frame's max ts) anchors the trailing window —
    pass the STORE max when the frame is a ticker subset.
    Returns a JSON-serialisable dict."""
    reason_cols = _tb_reason_cols(df.columns)
    t_max = df.index.max() if t_max is None else pd.Timestamp(t_max)
    cutoff = t_max - pd.Timedelta(days=holdout_days)
    per = {}
    tot = {'rows': 0, 'low': 0, 'high': 0, 'any': 0, 'in_trailing': 0}
    for t, g in df.groupby('Ticker', sort=True, observed=True):
        g = g.sort_index()
        lo, hi = flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'],
                                  low_frac, body_frac, high_frac)
        anym = lo | hi
        n_any = int(anym.sum())
        rec = {'rows': int(len(g)), 'low': int(lo.sum()),
               'high': int(hi.sum()), 'any': n_any,
               'frac_any': n_any / len(g) if len(g) else 0.0,
               'in_trailing': int((anym & (g.index > cutoff)).sum()),
               'max_low_depth': None, 'max_high_depth': None,
               'tb_reason_counts': {}, 'dates': []}
        if lo.any():
            rec['max_low_depth'] = float(np.max(1.0 - g['Low'].values[lo]
                                                / g['Open'].values[lo]))
        if hi.any():
            rec['max_high_depth'] = float(np.max(g['High'].values[hi]
                                                 / g['Open'].values[hi] - 1.0))
        for rc in reason_cols:
            vals = g[rc].values[anym]
            cnt = {}
            for v in vals:
                key = 'nan' if not np.isfinite(v) else REASON_NAMES.get(
                    int(v), str(int(v)))
                cnt[key] = cnt.get(key, 0) + 1
            rec['tb_reason_counts'][rc] = cnt
        sub = g[anym]
        side = np.where(lo[anym], 'low', 'high')
        for (ts, row), s in zip(sub.head(max_dates).iterrows(), side):
            d = {'ts': ts.isoformat(), 'side': str(s),
                 'Open': float(row['Open']), 'High': float(row['High']),
                 'Low': float(row['Low']), 'Close': float(row['Close'])}
            if 'Volume' in row:
                d['Volume'] = float(row['Volume'])
            d['depth'] = float(1 - row['Low'] / row['Open']) if s == 'low' \
                else float(row['High'] / row['Open'] - 1)
            if 'Src' in row:
                d['Src'] = str(row['Src'])
            rec['dates'].append(d)
        per[str(t)] = rec
        for k in ('rows', 'low', 'high', 'any', 'in_trailing'):
            tot[k] += rec[k]
    return {'params': {'low_frac': low_frac, 'body_frac': body_frac,
                       'high_frac': low_frac if high_frac is None else high_frac,
                       'holdout_days': holdout_days,
                       'trailing_cutoff': cutoff.isoformat(),
                       'store_max_ts': t_max.isoformat()},
            'tb_reason_columns': reason_cols, 'totals': tot,
            'per_ticker': per}


def census_store(path, cols, tickers=None, low_frac=0.15, body_frac=0.01,
                 high_frac=None, holdout_days=365, max_dates=20,
                 chunk_rows=400_000):
    """census() over a parquet store, loading ticker CHUNKS of about
    `chunk_rows` rows so RSS stays flat on the 2.2M-row stock sidecar.
    The trailing window is anchored on the whole-store max timestamp."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    tcol = pq.read_table(path, columns=['Ticker']).column('Ticker')
    counts = {str(r['values']): int(r['counts'])
              for r in pc.value_counts(tcol).to_pylist()}
    del tcol
    names = sorted(counts) if tickers is None else [t for t in tickers
                                                     if t in counts]
    t_max = None
    if 'Datetime' in pf.schema_arrow.names:
        t_max = pd.Timestamp(pc.max(pq.read_table(
            path, columns=['Datetime']).column('Datetime')).as_py())
        t_max = t_max.tz_localize('UTC') if t_max.tzinfo is None \
            else t_max.tz_convert('UTC')
    groups, cur, n = [], [], 0
    for t in names:
        if cur and n + counts[t] > chunk_rows:
            groups.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += counts[t]
    if cur:
        groups.append(cur)
    out = None
    for grp in groups:
        df, _ = load_store(path, cols, grp)
        c = census(df, low_frac, body_frac, high_frac, holdout_days,
                   max_dates, t_max=t_max)
        del df
        if out is None:
            out = c
        else:
            out['per_ticker'].update(c['per_ticker'])
            for k in out['totals']:
                out['totals'][k] += c['totals'][k]
            out['tb_reason_columns'] = sorted(
                set(out['tb_reason_columns']) | set(c['tb_reason_columns']),
                key=lambda s: int(s.rsplit('_', 1)[1]))
    out['per_ticker'] = dict(sorted(out['per_ticker'].items()))
    return out


# ---------------------------------------------------------------- footprint
def _range_features(g):
    """High/Low-consuming features recomputed with the production formulas
    (pure pandas; indicators.py cites in FEATURE_SOURCES)."""
    h, l, c = g['High'], g['Low'], g['Close']
    tr = pd.Series(true_range(h, l, c), index=g.index)
    atr = tr.rolling(14).mean()
    f = {'ATR': atr, 'ATR_Pct': atr / c * 100}
    f['ATR_Percentile'] = atr.rolling(100).apply(
        lambda x: (x[:-1] < x[-1]).sum() / len(x), raw=True)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    rk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    f['STOCHk_14_3_3'] = rk.rolling(3).mean()
    f['STOCHd_14_3_3'] = f['STOCHk_14_3_3'].rolling(3).mean()
    for w in (20, 60):
        hi, lo = h.rolling(w, min_periods=w).max(), l.rolling(w, min_periods=w).min()
        rng = (hi - lo).replace(0, np.nan)
        f[f'Pos_Range_{w}h'] = ((c - lo) / rng).clip(0, 1)
        f[f'MidRange_Gap_{w}h'] = ((0.5 * (hi + lo) - c) / rng).clip(-1, 1)
    # Parkinson daily realized range (volatility.daily_realized_range, the
    # HAR-RV input) — per DAY, mapped back to the bars of that day.
    pk = np.log(h / l) ** 2 / (4.0 * np.log(2.0))
    day = g.index.normalize()
    f['Parkinson_RRV_day'] = pd.Series(pk.groupby(day).transform('sum').values,
                                       index=g.index)
    return f


FEATURE_SOURCES = {
    'ATR': 'indicators.py:509 (compute_atr 357-375; TR 174-183, 14-bar mean)',
    'ATR_Pct': 'indicators.py:647-652,661,672 (stock only)',
    'ATR_Percentile': 'indicators.py:553 (rolling-100 rank of ATR, 433-443)',
    'STOCHk_14_3_3': 'indicators.py:527 (compute_stoch 398-416; 14 min/max, 3 smooth)',
    'STOCHd_14_3_3': 'indicators.py:527-529 (3-mean of STOCHk)',
    'Pos_Range_20h': 'indicators.py:831-837 (_jkx_ranges, stock only)',
    'MidRange_Gap_20h': 'indicators.py:838-839 (stock only)',
    'Pos_Range_60h': 'indicators.py:834-837 (stock only)',
    'MidRange_Gap_60h': 'indicators.py:838-839 (stock only)',
    'Parkinson_RRV_day': 'volatility.py:158-164 daily_realized_range (live HAR input, not a store column)',
}


# Day-aggregated features: a print moves its whole calendar day (rows
# before it included), so 'reach after the print' is not defined; the
# downstream reach is the HAR regressors' 5/22-day means and the 250-day
# fit window + max-clamp (volatility.py:172-216).
DAY_LEVEL = {'Parkinson_RRV_day'}


def _reach(delta, anym, tol):
    """Rows moved (|delta|>tol) and the longest reach AFTER a print
    (bars from the print to the last moved row before the next print)."""
    moved = np.abs(np.nan_to_num(delta, nan=0.0)) > tol
    pos = np.flatnonzero(anym)
    max_after = 0
    for k, p in enumerate(pos):
        end = pos[k + 1] if k + 1 < len(pos) else len(moved)
        seg = np.flatnonzero(moved[p:end])
        if len(seg):
            max_after = max(max_after, int(seg[-1]))
    return int(moved.sum()), max_after


def footprint(df, low_frac=0.15, body_frac=0.01, high_frac=None,
              rel_tol=1e-3):
    """Per-feature contamination counts over the flagged tickers."""
    out = {'rel_tol': rel_tol, 'features': {}, 'atr_at_print': []}
    agg = {}
    for t, g in df.groupby('Ticker', sort=True, observed=True):
        g = g.sort_index()
        lo, hi = flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'],
                                  low_frac, body_frac, high_frac)
        anym = lo | hi
        if not anym.any():
            continue
        a = _range_features(g)
        b = _range_features(repair_frame(g, lo, hi))
        for name in a:
            av, bv = a[name].values, b[name].values
            scale = np.nanmedian(np.abs(av)) or 1.0
            n_moved, reach = _reach((av - bv) / scale, anym, rel_tol)
            if name in DAY_LEVEL:   # same-day rows BEFORE a print move too
                reach = -1
            r = agg.setdefault(name, {'rows_moved': 0, 'max_reach_bars': 0,
                                      'max_ratio_at_print': 0.0,
                                      'in_store': name in g.columns,
                                      'source': FEATURE_SOURCES.get(name, '')})
            r['rows_moved'] += n_moved
            r['max_reach_bars'] = -1 if reach < 0 else max(r['max_reach_bars'], reach)
            with np.errstate(invalid='ignore', divide='ignore'):
                rat = np.abs(av[anym] / bv[anym])
            rat = rat[np.isfinite(rat)]
            if len(rat):
                r['max_ratio_at_print'] = max(r['max_ratio_at_print'],
                                              float(rat.max()))
        if 'ATR' in g.columns:
            atr = g['ATR']
            med = atr.shift(1).rolling(500, min_periods=50).median()
            # store ATR vs our recomputation (sanity: formula parity)
            with np.errstate(invalid='ignore', divide='ignore'):
                par = np.nanmedian(np.abs(atr.values / a['ATR'].values - 1))
            agg.setdefault('_parity', {})[str(t)] = float(par)
            for ts in g.index[anym]:
                i = g.index.get_loc(ts)
                m = med.iloc[i]
                win = atr.iloc[i:i + 30].values
                out['atr_at_print'].append({
                    'ticker': str(t), 'ts': ts.isoformat(),
                    'atr_over_trailing_median': float(atr.iloc[i] / m) if m else None,
                    'max_atr_over_median_next14': float(np.nanmax(win[:14]) / m) if m else None,
                    'bars_atr_gt_1p5x_median': int(np.sum(win > 1.5 * m)) if m else None})
    out['atr_store_vs_recomputed_median_absrel'] = agg.pop('_parity', {})
    out['features'] = agg
    return out


def tb_flips(df, asset_type, low_frac=0.15, body_frac=0.01, high_frac=None,
             fbs=None):
    """Re-run policy_exits.compute_tb_labels on stored rows, as stored vs
    repaired (ATR shifted by the repair's TR delta). Jetson (numba)."""
    from policy_exits import compute_tb_labels
    res = {}
    for t, g in df.groupby('Ticker', sort=True, observed=True):
        g = g.sort_index()
        lo, hi = flag_wick_prints(g['Open'], g['High'], g['Low'], g['Close'],
                                  low_frac, body_frac, high_frac)
        if not (lo | hi).any() or 'ATR' not in g.columns:
            continue
        horizons = fbs or [int(c.rsplit('_', 1)[1]) for c in
                           _tb_reason_cols(g.columns)] or [24]
        r = repair_frame(g, lo, hi)
        d_atr = (pd.Series(true_range(r['High'], r['Low'], r['Close']), index=g.index)
                 - pd.Series(true_range(g['High'], g['Low'], g['Close']), index=g.index)
                 ).rolling(14).mean().fillna(0.0)
        r['ATR'] = g['ATR'] + d_atr
        cols = ['Open', 'High', 'Low', 'Close', 'ATR']
        la = compute_tb_labels(g[cols], horizons, asset_type)
        lb = compute_tb_labels(r[cols], horizons, asset_type)
        rec = {}
        for fb in horizons:
            ra, rb = la[f'TB_Reason_{fb}'], lb[f'TB_Reason_{fb}']
            ok = np.isfinite(ra) & np.isfinite(rb)
            rec[str(fb)] = {
                'labels': int(ok.sum()),
                'reason_flips': int(np.sum(ok & (ra != rb))),
                'hard_stop_before': int(np.sum(ra[ok] == 1)),
                'hard_stop_after': int(np.sum(rb[ok] == 1)),
                'hard_stop_removed': int(np.sum(ok & (ra == 1) & (rb != 1))),
                'ret_changed': int(np.sum(ok & (np.abs(la[f'TB_Ret_{fb}'] - lb[f'TB_Ret_{fb}']) > 1e-9))),
            }
            if 'TB_Reason_%d' % fb in g.columns:
                st = g[f'TB_Reason_{fb}'].values
                okk = np.isfinite(st) & np.isfinite(ra)
                rec[str(fb)]['store_vs_recomputed_agree'] = float(
                    np.mean(st[okk] == ra[okk])) if okk.any() else None
        res[str(t)] = rec
    return res


# ------------------------------------------------------------- venue check
COINBASE_URL = ('https://api.exchange.coinbase.com/products/{p}/candles'
                '?granularity=3600&start={s}&end={e}')
VENUE_WINDOW_H = 299          # Coinbase returns <= 300 candles per call
ABSENT_DEPTH = 0.03           # second-venue wick < 3 % beyond the body


def plan_windows(flags, max_requests=10, must=()):
    """Greedy: cover the flag list with <=max_requests (ticker, start, end)
    windows of VENUE_WINDOW_H hours, most-covering first; `must` = list of
    (ticker, iso_ts) that each get a window before the greedy fill."""
    fl = sorted({(f['ticker'], pd.Timestamp(f['ts'])) for f in flags})
    cands = []
    for t, ts in fl:
        end = ts + pd.Timedelta(hours=VENUE_WINDOW_H)
        cov = [(t2, x) for t2, x in fl if t2 == t and ts <= x <= end]
        cands.append((t, ts, end, cov))
    chosen, covered = [], set()
    for t, iso in must:
        ts = pd.Timestamp(iso)
        for c in cands:
            if c[0] == t and c[1] == ts:
                chosen.append(c)
                covered.update(c[3])
    while len(chosen) < max_requests:
        best = max(cands, key=lambda c: len(set(c[3]) - covered), default=None)
        if best is None or not (set(best[3]) - covered):
            break
        chosen.append(best)
        covered.update(best[3])
    return [(c[0], c[1], c[2]) for c in chosen[:max_requests]]


def venue_verdict(flag, venue_low, venue_high, venue_open=None,
                  venue_close=None):
    """bad_body : the Alpaca Close lies > ABSENT_DEPTH OUTSIDE the second
                  venue's [Low, High] for that hour (the whole bar is
                  off-market — a wick repair anchored on that body would
                  be wrong; returned depth = the Close's distance outside);
    bad_print: the second venue shows (almost) no wick on that side;
    real_move: it shows >= half the Alpaca wick depth; partial otherwise.
    venue_open/venue_close are accepted for the record only."""
    o, c = flag['Open'], flag['Close']
    if venue_low and venue_high:
        out_lo = venue_low / c - 1          # > 0: Close below the venue Low
        out_hi = c / venue_high - 1         # > 0: Close above the venue High
        off = max(out_lo, out_hi)
        if off > ABSENT_DEPTH:
            base = min(o, c) if flag['side'] == 'low' else max(o, c)
            d1 = (1 - flag['Low'] / base if flag['side'] == 'low'
                  else flag['High'] / base - 1)
            return 'bad_body', float(d1), float(off)
    if flag['side'] == 'low':
        base = min(o, c)
        d1 = 1 - flag['Low'] / base
        d2 = 1 - venue_low / base
    else:
        base = max(o, c)
        d1 = flag['High'] / base - 1
        d2 = venue_high / base - 1
    v = ('bad_print' if d2 < ABSENT_DEPTH else
         'real_move' if d2 >= 0.5 * d1 else 'partial')
    return v, float(d1), float(d2)


def rescore_venue_checks(vc):
    """(Re)apply venue_verdict to saved checks (no network); dedupe
    (ticker, ts) duplicates from overlapping windows; retally."""
    seen, checks = set(), []
    for chk in vc['checks']:
        k = (chk['ticker'], chk['ts'])
        if k in seen:
            continue
        seen.add(k)
        if 'venue_low' in chk:
            chk['verdict'], chk['alpaca_depth'], chk['venue_depth'] = \
                venue_verdict(chk, chk['venue_low'], chk['venue_high'],
                              chk.get('venue_open'), chk.get('venue_close'))
        checks.append(chk)
    vc['checks'] = checks
    tally = {}
    for chk in checks:
        tally[chk['verdict']] = tally.get(chk['verdict'], 0) + 1
    vc['tally'] = tally
    vc['rule'] = (f'bad_body if the Alpaca Close is > {ABSENT_DEPTH:.0%} outside the venue [Low, High]; '
                  f'else bad_print if second-venue wick beyond the body < '
                  f'{ABSENT_DEPTH:.0%}; real_move if >= 50% of the Alpaca wick; '
                  'else partial')
    return vc


def coinbase_venue_check(flags, windows, sleep_s=2.0, timeout=20):
    """One public GET per window (no key). Returns a JSON-able dict."""
    import time
    import requests
    out = {'venue': 'coinbase_exchange_1h', 'rule': (
        f'bad_print if second-venue wick beyond the body < {ABSENT_DEPTH:.0%}; '
        'real_move if >= 50% of the Alpaca wick; else partial'),
           'requests': [], 'checks': []}
    for k, (t, s, e) in enumerate(windows):
        if k:
            time.sleep(sleep_s)
        url = COINBASE_URL.format(p=t, s=s.strftime('%Y-%m-%dT%H:%M:%SZ'),
                                  e=e.strftime('%Y-%m-%dT%H:%M:%SZ'))
        rec = {'ticker': t, 'start': s.isoformat(), 'end': e.isoformat(),
               'url': url}
        try:
            r = requests.get(url, timeout=timeout,
                             headers={'User-Agent': 'trader-badprint-census'})
            rec['status'] = r.status_code
            rows = r.json() if r.status_code == 200 else []
        except Exception as ex:          # fail-soft: measurement only
            rec['error'] = str(ex)
            rows = []
        rec['candles'] = len(rows) if isinstance(rows, list) else 0
        out['requests'].append(rec)
        bar = {}
        for row in rows if isinstance(rows, list) else []:
            # [time, low, high, open, close, volume]
            bar[pd.Timestamp(int(row[0]), unit='s', tz='UTC')] = row
        for f in flags:
            ts = pd.Timestamp(f['ts'])
            if f['ticker'] != t or not (s <= ts <= e):
                continue
            row = bar.get(ts)
            chk = {k2: f[k2] for k2 in ('ticker', 'ts', 'side', 'Open', 'High',
                                        'Low', 'Close')}
            chk['Volume'] = f.get('Volume')
            if row is None:
                chk['verdict'] = 'no_venue_bar'
            else:
                chk.update(venue_low=float(row[1]), venue_high=float(row[2]),
                           venue_open=float(row[3]), venue_close=float(row[4]),
                           venue_volume=float(row[5]))
            out['checks'].append(chk)
    return rescore_venue_checks(out)


# --------------------------------------------------------------------- text
def render(c, fp=None, flips=None, label=''):
    p = c['params']
    L = [f"bad-print census {label}".rstrip(),
         f"  test: body |C-O|/O < {p['body_frac']}; low-side L < O*(1-{p['low_frac']}); "
         f"high-side H > O*(1+{p['high_frac']})",
         f"  store max ts {p['store_max_ts']}; trailing-{p['holdout_days']}d cutoff {p['trailing_cutoff']}",
         f"  TB reason columns: {', '.join(c['tb_reason_columns']) or '(none)'}",
         f"  TOTAL rows {c['totals']['rows']}  low {c['totals']['low']}  high {c['totals']['high']}"
         f"  any {c['totals']['any']}  in-trailing {c['totals']['in_trailing']}", '']
    for t, r in c['per_ticker'].items():
        if not r['any']:
            continue
        L.append(f"{t}: rows {r['rows']} low {r['low']} high {r['high']} any {r['any']} "
                 f"({100 * r['frac_any']:.4f}%) trailing {r['in_trailing']} "
                 f"maxLowDepth {r['max_low_depth']} maxHighDepth {r['max_high_depth']}")
        for rc, cnt in r['tb_reason_counts'].items():
            L.append(f"    {rc}: {cnt}")
        for d in r['dates']:
            L.append(f"    {d['ts']} {d['side']:4s} O {d['Open']:.6g} H {d['High']:.6g} "
                     f"L {d['Low']:.6g} C {d['Close']:.6g} V {d.get('Volume', float('nan')):.4g} "
                     f"depth {100 * d['depth']:.1f}% {d.get('Src', '')}")
    if fp:
        L += ['', f"FOOTPRINT (stored rows recomputed as-is vs repaired; moved = |d|/median|x| > {fp['rel_tol']})"]
        for n, r in fp['features'].items():
            L.append(f"  {n:18s} in_store={r['in_store']!s:5s} rows_moved {r['rows_moved']:6d} "
                     f"max_reach {('day' if r['max_reach_bars'] < 0 else r['max_reach_bars'])!s:>4s} bars  max|orig/rep| at print "
                     f"{r['max_ratio_at_print']:.3g}  [{r['source']}]")
        L.append(f"  ATR store-vs-recomputed median |rel|: {fp['atr_store_vs_recomputed_median_absrel']}")
        big = sorted([x for x in fp['atr_at_print'] if x['max_atr_over_median_next14']],
                     key=lambda x: -x['max_atr_over_median_next14'])[:15]
        for x in big:
            L.append(f"    {x['ticker']} {x['ts']} ATR/trailing-median at print "
                     f"{x['atr_over_trailing_median']:.2f}, max next14 {x['max_atr_over_median_next14']:.2f}, "
                     f"bars>1.5x {x['bars_atr_gt_1p5x_median']}/30")
    if flips:
        L += ['', 'TB LABEL FLIPS (policy_exits kernel on stored rows, as-is vs repaired)']
        for t, rec in flips.items():
            for fb, r in rec.items():
                L.append(f"  {t} fb={fb}: {r}")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='training_data.parquet')
    ap.add_argument('--low-frac', type=float, default=0.15)
    ap.add_argument('--body-frac', type=float, default=0.01)
    ap.add_argument('--high-frac', type=float, default=None)
    ap.add_argument('--tickers', default=None, help='comma list (projection filter)')
    ap.add_argument('--holdout-days', type=int, default=365)
    ap.add_argument('--asset', choices=['crypto', 'stock'], default=None,
                    help='for --tb-flips; default: stock iff "stock" in --data')
    ap.add_argument('--footprint', action='store_true')
    ap.add_argument('--tb-flips', action='store_true')
    ap.add_argument('--max-dates', type=int, default=20)
    ap.add_argument('--out', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--venue-check', default=None, metavar='OUT_JSON',
                 help='opt-in NETWORK: Coinbase 1h candles for the flagged '
                      'bars (<= --venue-max public GETs, 2 s apart)')
    ap.add_argument('--venue-max', type=int, default=10)
    ap.add_argument('--venue-must', default='',
                 help='TICKER@ISO,... flags that must get their own window')
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    names = pq.ParquetFile(a.data).schema_arrow.names
    base = ['Ticker'] + OHLCV + (['Src'] if 'Src' in names else [])
    base += [c for c in names if c.startswith('TB_Reason_')]
    if 'Target_Return' in names:
        base.append('Target_Return')
    tickers = a.tickers.split(',') if a.tickers else None
    c = census_store(a.data, base, tickers, a.low_frac, a.body_frac,
                     a.high_frac, a.holdout_days, a.max_dates)
    df = None
    fp = flips = None
    flagged = [t for t, r in c['per_ticker'].items() if r['any']]
    if (a.footprint or a.tb_flips) and flagged:
        feat = [n for n in ('ATR', 'ATR_Pct', 'ATR_Percentile', 'STOCHk_14_3_3',
                            'STOCHd_14_3_3', 'Pos_Range_20h', 'Pos_Range_60h',
                            'MidRange_Gap_20h', 'MidRange_Gap_60h') if n in names]
        del df
        fp_parts, fl_parts = [], {}
        asset = a.asset or ('stock' if 'stock' in Path(a.data).name else 'crypto')
        for t in flagged:     # one ticker at a time keeps RSS flat
            g, _ = load_store(a.data, ['Ticker'] + OHLCV + feat
                              + [x for x in names if x.startswith('TB_Reason_')], [t])
            if a.footprint:
                fp_parts.append(footprint(g, a.low_frac, a.body_frac, a.high_frac))
            if a.tb_flips:
                fl_parts.update(tb_flips(g, asset, a.low_frac, a.body_frac, a.high_frac))
        if a.footprint:
            fp = {'rel_tol': fp_parts[0]['rel_tol'], 'features': {},
                  'atr_at_print': [], 'atr_store_vs_recomputed_median_absrel': {}}
            for part in fp_parts:
                fp['atr_at_print'] += part['atr_at_print']
                fp['atr_store_vs_recomputed_median_absrel'].update(
                    part['atr_store_vs_recomputed_median_absrel'])
                for n, r in part['features'].items():
                    q = fp['features'].setdefault(n, dict(r, rows_moved=0,
                                                          max_reach_bars=0,
                                                          max_ratio_at_print=0.0))
                    q['rows_moved'] += r['rows_moved']
                    q['max_reach_bars'] = (-1 if r['max_reach_bars'] < 0 else
                                           max(q['max_reach_bars'], r['max_reach_bars']))
                    q['max_ratio_at_print'] = max(q['max_ratio_at_print'], r['max_ratio_at_print'])
                    q['in_store'] = q['in_store'] or r['in_store']
        flips = fl_parts if a.tb_flips else None
    if a.venue_check:
        flags = [dict(d, ticker=t) for t, r in c['per_ticker'].items()
                 for d in r['dates']]
        must = [tuple(x.split('@', 1)) for x in a.venue_must.split(',') if x]
        wins = plan_windows(flags, a.venue_max, must)
        vc = coinbase_venue_check(flags, wins)
        vc['params'] = c['params']
        Path(a.venue_check).write_text(json.dumps(vc, indent=1))
        sys.stdout.write(f"venue tally {vc['tally']} over {len(wins)} requests\n")
    txt = render(c, fp, flips, label=str(a.data))
    if a.out:
        Path(a.out).write_text(txt)
    else:
        sys.stdout.write(txt)
    if a.json:
        Path(a.json).write_text(json.dumps({'census': c, 'footprint': fp,
                                            'tb_flips': flips}, indent=1))
    return c


if __name__ == '__main__':
    main()
