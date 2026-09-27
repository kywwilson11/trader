"""Bars-per-year census of a training store — measurement-only.

Why: `BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}` (four copies:
backtest.py, scripts/hypersearch_v2.py, portfolio_backtest.py, volatility.py)
annualizes per-bar Sharpe / vol targets. 1638 = 252 x 6.5 assumes RTH-only
hourly bars, but Alpaca's stock hourly bars span the extended session
(04:00-20:00 ET, market_data.py `_SIP_SESSION_*`). This script measures what a
store ACTUALLY contains so the owner can judge the constant on numbers.

Reads ONLY the timestamp + ticker columns (pyarrow column projection; the
timestamp is normally the pandas index, stored as a 'Datetime' column). Never
writes the store. Exit status is always 0.

Bar convention: each bar is labelled by its OPEN time and lasts one bar
interval (inferred as the modal positive within-ticker timestamp step; 1h for
these stores). A bar is classified against the RTH session [09:30, 16:00) ET by
the overlap of [open, open+interval) with that session:
    rth_full       fully inside RTH
    open_straddle  crosses 09:30 (e.g. the Alpaca 09:00 bar)
    close_straddle crosses 16:00 (e.g. a yfinance :30-labelled 15:30 bar
                   is NOT one — it ends exactly at 16:00)
    pre            ends at/before 09:30
    post           opens at/after 16:00
    weekend        Saturday/Sunday in the session timezone (crypto)
The "RTH-only subset" = every bar with any RTH overlap (rth_full + straddles),
i.e. what an RTH-filtered hourly harvest would keep.

bars/year, two estimators, per ticker then averaged across tickers:
    span:  rows / ((last - first + interval) / 365.25 days)   — calendar rate;
           this is the quantity hypersearch_v2.compute_sharpe needs (rows fed
           per ticker per calendar year).
    day:   per-ticker mean bars/day x days_per_year (252 stock / 365.25
           crypto), averaged across tickers (each ticker weighs equally)
    pool:  mean over ALL ticker-days of bars/day x days_per_year — the
           row-weighted figure a pooled Sharpe (compute_sharpe) needs
Reported for the whole store, its RTH-overlap subset, the trailing 365 days
(recent_365d — where the walk-forward folds/holdout sit) and per calendar year.

    python scripts/bars_per_year_census.py --data stock_training_data.parquet
    python scripts/bars_per_year_census.py --data training_data.parquet --json c.json
"""
import argparse
import json
import math
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
DAYS_PER_YEAR = {'crypto': 365.25, 'stock': 252.0}
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60
_TS_NAMES = ('Datetime', 'datetime', 'Timestamp', 'timestamp', 'Date',
             'date', 'time', 'ts')
_TICKER_NAMES = ('Ticker', 'ticker', 'Symbol', 'symbol')


def _pick_columns(path):
    """(timestamp_column, ticker_column) from the parquet schema only."""
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    names = list(schema.names)
    ts_col = None
    md = schema.pandas_metadata or {}
    for ic in md.get('index_columns', []) or []:
        if isinstance(ic, str) and ic in names:
            ts_col = ic
            break
    if ts_col is None:
        for n in _TS_NAMES:
            if n in names:
                ts_col = n
                break
    if ts_col is None:
        for n in names:
            if str(schema.field(n).type).startswith('timestamp'):
                ts_col = n
                break
    tk_col = next((n for n in _TICKER_NAMES if n in names), None)
    return ts_col, tk_col


def load_index(path):
    """Projection read of (timestamp, ticker) -> DataFrame[ts (UTC), ticker]."""
    import pyarrow.parquet as pq
    ts_col, tk_col = _pick_columns(path)
    if ts_col is None:
        raise ValueError(f"no timestamp column/index found in {path}")
    cols = [ts_col] + ([tk_col] if tk_col else [])
    tbl = pq.read_table(path, columns=cols,
                        read_dictionary=[tk_col] if tk_col else None)
    ts = pd.to_datetime(tbl.column(ts_col).to_pandas(), utc=True)
    if tk_col:
        tk = tbl.column(tk_col).to_pandas().astype(str).values
    else:
        tk = np.full(len(ts), '_single_')
    del tbl
    return pd.DataFrame({'ts': ts.values, 'ticker': pd.Categorical(tk)}).assign(
        ts=lambda d: pd.to_datetime(d['ts'], utc=True))


def _classify(open_min, interval_min, weekday):
    end_min = open_min + interval_min
    overlap = (np.minimum(end_min, RTH_CLOSE_MIN)
               - np.maximum(open_min, RTH_OPEN_MIN)).clip(min=0)
    cls = np.full(len(open_min), 'post', dtype=object)
    cls[end_min <= RTH_OPEN_MIN] = 'pre'
    cls[(overlap > 0) & (open_min < RTH_OPEN_MIN)] = 'open_straddle'
    cls[(overlap > 0) & (open_min >= RTH_OPEN_MIN) & (end_min <= RTH_CLOSE_MIN)] = 'rth_full'
    cls[(overlap > 0) & (open_min >= RTH_OPEN_MIN) & (end_min > RTH_CLOSE_MIN)] = 'close_straddle'
    cls[weekday >= 5] = 'weekend'
    return cls


def _dist(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {'median': None, 'mean': None, 'p10': None, 'p90': None}
    return {'median': float(np.median(x)), 'mean': float(np.mean(x)),
            'p10': float(np.percentile(x, 10)),
            'p90': float(np.percentile(x, 90))}


def _rates(df, interval, days_per_year):
    """Per-ticker span and day estimators (averaged across tickers)."""
    g = df.groupby('ticker', observed=True)
    n = g.size()
    span_days = ((g['ts'].max() - g['ts'].min() + interval)
                 / pd.Timedelta(days=1))
    span_rate = (n / (span_days / 365.25)).astype(float)
    per_day = df.groupby(['ticker', 'day'], observed=True).size()
    bpd_ticker = per_day.groupby(level=0, observed=True).mean()
    return {
        'bars_per_ticker_day': _dist(per_day.values),
        'bars_per_year_span_mean': float(span_rate.mean()),
        'bars_per_year_span_median': float(span_rate.median()),
        'bars_per_year_day_based': float(bpd_ticker.mean() * days_per_year),
        # Row-weighted (pooled over ticker-days): the estimator that matches
        # a POOLED Sharpe such as hypersearch_v2.compute_sharpe, which turns
        # sum(rows) into ticker-years via rows / bars_per_year.
        'bars_per_year_day_pooled': float(per_day.mean() * days_per_year),
        'trading_days_per_ticker_year': float(
            (g['day'].nunique() / (span_days / 365.25)).mean()),
    }


def _by_year(df, days_per_year):
    """Per local calendar year: tickers present, rows/ticker, trading days
    per ticker, mean bars per ticker-day and the day-based bars/year. Shows
    whether coverage (extended-hours depth, holes) drifts over time — the
    walk-forward validation folds sit at the recent end."""
    yr = df['day'].dt.year
    res = {}
    for y, sub in df.groupby(yr):
        per_day = sub.groupby(['ticker', 'day'], observed=True).size()
        nt = int(sub['ticker'].nunique())
        res[int(y)] = {
            'tickers': nt,
            'rows_per_ticker': float(len(sub) / nt),
            'days_per_ticker': float(len(per_day) / nt),
            'bars_per_ticker_day_mean': float(per_day.mean()),
            'bars_per_year_day_based': float(per_day.mean() * days_per_year),
        }
    return res


def census(df, tz='America/New_York', asset='auto', rth_only=False):
    """Pure census over a (ts UTC, ticker) frame. Returns a JSON-able dict.
    rth_only=True drops every non-RTH bar first (after the bar interval and
    the asset class were inferred on the full frame)."""
    out = {'rows': int(len(df)), 'tz': tz}
    if len(df) == 0:
        return out
    df = df.sort_values(['ticker', 'ts'], kind='mergesort').reset_index(drop=True)
    loc = df['ts'].dt.tz_convert(tz)
    df['day'] = loc.dt.tz_localize(None).dt.normalize()
    weekday = loc.dt.weekday.values
    open_min = (loc.dt.hour * 60 + loc.dt.minute).values

    step = df.groupby('ticker', observed=True)['ts'].diff().dropna()
    step = step[step > pd.Timedelta(0)]
    interval = step.mode().iloc[0] if len(step) else pd.Timedelta(hours=1)
    interval_min = int(interval / pd.Timedelta(minutes=1))

    weekend_share = float((weekday >= 5).mean())
    if asset == 'auto':
        asset = 'crypto' if weekend_share > 0.05 else 'stock'
    dpy = DAYS_PER_YEAR[asset]
    legacy = LEGACY_BARS_PER_YEAR[asset]

    cls = _classify(open_min, interval_min, weekday)
    df['cls'] = cls
    if rth_only:
        keep = np.isin(cls, ['open_straddle', 'rth_full', 'close_straddle'])
        df, cls, loc = df[keep], cls[keep], loc[keep]
        out['rows'] = int(len(df))
        if len(df) == 0:
            return out
    cats = ['pre', 'open_straddle', 'rth_full', 'close_straddle', 'post',
            'weekend']
    counts = pd.Series(cls).value_counts()
    out.update({
        'asset': asset,
        'tickers': int(df['ticker'].nunique()),
        'start': str(df['ts'].min()), 'end': str(df['ts'].max()),
        'bar_interval_min': interval_min,
        'weekend_share': weekend_share,
        'minute_of_hour_hist': {int(k): int(v) for k, v in
                                pd.Series(loc.dt.minute.values)
                                .value_counts().sort_index().items()},
        'hour_hist_local': {int(k): int(v) for k, v in
                            pd.Series(loc.dt.hour.values)
                            .value_counts().sort_index().items()},
        'session_counts': {c: int(counts.get(c, 0)) for c in cats},
        'session_share': {c: float(counts.get(c, 0)) / len(df) for c in cats},
        'days_per_year_basis': dpy,
        'legacy_bars_per_year': legacy,
    })
    out['all'] = _rates(df, interval, dpy)
    out['by_year'] = _by_year(df, dpy)
    # The walk-forward folds and the holdout sit at the recent end of the
    # store; extended-hours depth grows over time, so the trailing year is
    # the most relevant single estimate.
    recent = df[df['ts'] > df['ts'].max() - pd.Timedelta(days=365)]
    out['recent_365d'] = _rates(recent, interval, dpy) if len(recent) else None
    rth = df[df['cls'].isin(['open_straddle', 'rth_full', 'close_straddle'])]
    out['rth_only'] = _rates(rth, interval, dpy) if len(rth) else None
    for key in ('all', 'rth_only', 'recent_365d'):
        blk = out[key]
        if not blk:
            continue
        for est in ('span_mean', 'day_based', 'day_pooled'):
            v = blk[f'bars_per_year_{est}']
            blk[f'ratio_{est}_vs_legacy'] = v / legacy
            blk[f'sqrt_ratio_{est}_vs_legacy'] = math.sqrt(v / legacy)
    return out


def _fmt(res):
    L = []
    L.append(f"rows={res['rows']:,}  tickers={res.get('tickers')}  "
             f"asset={res.get('asset')}  tz={res['tz']}")
    if res['rows'] == 0:
        return '\n'.join(L)
    L.append(f"span {res['start']} -> {res['end']}  "
             f"bar_interval={res['bar_interval_min']}min  "
             f"weekend_share={res['weekend_share']:.4f}")
    L.append(f"minute-of-hour (local): {res['minute_of_hour_hist']}")
    L.append(f"hour histogram (local, bar OPEN time): {res['hour_hist_local']}")
    L.append("session       count        share")
    for c, v in res['session_counts'].items():
        L.append(f"  {c:<14}{v:>10,}   {res['session_share'][c]:7.2%}")
    L.append(f"legacy BARS_PER_YEAR = {res['legacy_bars_per_year']}  "
             f"(day basis {res['days_per_year_basis']})")
    L.append(f"{'subset':<9}{'bpd med':>8}{'mean':>7}{'p10':>6}{'p90':>6}"
             f"{'days/yr':>9}{'bpy span':>10}{'bpy day':>9}"
             f"{'span/leg':>9}{'sqrt':>7}{'day/leg':>8}{'sqrt':>7}"
             f"{'bpy pool':>9}{'pool/leg':>9}{'sqrt':>7}")
    for key in ('all', 'rth_only', 'recent_365d'):
        b = res.get(key)
        if not b:
            continue
        d = b['bars_per_ticker_day']
        L.append(f"{key[:9]:<9}{d['median']:>8.1f}{d['mean']:>7.2f}{d['p10']:>6.1f}"
                 f"{d['p90']:>6.1f}{b['trading_days_per_ticker_year']:>9.1f}"
                 f"{b['bars_per_year_span_mean']:>10.0f}"
                 f"{b['bars_per_year_day_based']:>9.0f}"
                 f"{b['ratio_span_mean_vs_legacy']:>9.3f}"
                 f"{b['sqrt_ratio_span_mean_vs_legacy']:>7.3f}"
                 f"{b['ratio_day_based_vs_legacy']:>8.3f}"
                 f"{b['sqrt_ratio_day_based_vs_legacy']:>7.3f}"
                 f"{b['bars_per_year_day_pooled']:>9.0f}"
                 f"{b['ratio_day_pooled_vs_legacy']:>9.3f}"
                 f"{b['sqrt_ratio_day_pooled_vs_legacy']:>7.3f}")
    by = res.get('by_year') or {}
    if by:
        L.append("by year  tickers  rows/tk  days/tk  bpd mean  bpy(day)")
        for y, b in by.items():
            L.append(f"  {y}  {b['tickers']:>7}{b['rows_per_ticker']:>9.0f}"
                     f"{b['days_per_ticker']:>9.1f}"
                     f"{b['bars_per_ticker_day_mean']:>10.2f}"
                     f"{b['bars_per_year_day_based']:>10.0f}")
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='training store (.parquet)')
    ap.add_argument('--tz', default='America/New_York',
                    help='session timezone for day/hour bucketing')
    ap.add_argument('--asset', default='auto', choices=['auto', 'crypto', 'stock'])
    ap.add_argument('--rth-only', action='store_true',
                    help='restrict the whole census to bars overlapping RTH')
    ap.add_argument('--json', default=None, help='write the result dict here')
    args = ap.parse_args(argv)
    try:
        df = load_index(args.data)
        res = census(df, tz=args.tz, asset=args.asset, rth_only=args.rth_only)
        res['data'] = str(args.data)
        res['rth_only_flag'] = bool(args.rth_only)
        res['peak_rss_mb'] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(_fmt(res))
        print(f"peak RSS {res['peak_rss_mb']:.0f} MB")
        if args.json:
            Path(args.json).write_text(json.dumps(res, indent=2, default=str))
    except Exception as e:  # measurement-only: never a non-zero exit
        print(f"[bars_per_year_census] ERROR: {e!r}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
