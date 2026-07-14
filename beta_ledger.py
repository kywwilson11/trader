"""Beta ledger — how much of the strategy's P&L is market exposure?

The 2026-07 review's core finding: this is a long-only book on two highly
internally-correlated factor sleeves (BTC-complex crypto, high-beta US tech),
timed by trend/VIX/drawdown gates — and NOTHING measured its realized beta.
The literature is unambiguous that this measurement comes before any alpha
claim:

  * Asness-Krail-Liew (JPM 2001): "market-neutral" hedge funds showed near-zero
    contemporaneous beta but large LAGGED betas; summing contemporaneous+lagged
    roughly doubled measured exposure and erased most apparent alpha. Hence the
    Dimson-style lagged regression here.
  * Goulding-Harvey-Mazzoleni (JFE 2023): ~2/3 of even genuine trend-strategy
    alpha IS market timing of the factor — so betas are reported conditional on
    the trend state, not just unconditionally.
  * Henriksson-Merton (1981): timing shows up as up/down beta asymmetry, so the
    up-/down-market split is reported too.

Everything statistical here is pure numpy/pandas (testable on the dev Mac);
only the data loaders touch Alpaca/yfinance (Jetson / networked runs).

Usage (Jetson):
    python beta_ledger.py --days 90                # Alpaca equity + yfinance benchmarks
    python beta_ledger.py --equity-csv eq.csv --benchmarks-csv b.csv   # offline
CSV formats: equity = date,equity columns; benchmarks = date + one PRICE
column per benchmark (e.g. date,SPY,BTC).

Measurement-only: reads the account and market data, writes a report. It
gates nothing and touches no trading path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 252


# --- regression core (pure) ---

def ols_hac(X: np.ndarray, y: np.ndarray, hac_lags: int | None = None):
    """OLS with Newey-West (Bartlett) HAC standard errors.

    X must already include the intercept column. Returns (beta, se, resid).
    hac_lags=None -> Newey-West (1994) automatic plug-in floor(4*(n/100)^(2/9)).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k + 1:
        raise ValueError(f"too few observations ({n}) for {k} regressors")
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    if hac_lags is None:
        hac_lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    hac_lags = max(0, min(hac_lags, n - 2))
    xtx_inv = np.linalg.pinv(X.T @ X)
    scores = X * resid[:, None]
    S = scores.T @ scores
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        S += w * (gamma + gamma.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return beta, se, resid


def align_benchmark_returns(equity: pd.Series,
                            bench_prices: pd.DataFrame) -> pd.DataFrame:
    """Benchmark returns over the SAME date-to-date spans as the equity curve.

    The equity series lives on the broker's trading-day grid while a 24/7
    benchmark (BTC) has calendar-day prices. As-of aligning PRICES to the
    equity dates first, then differencing, makes a Friday->Monday equity move
    regress against the Friday->Monday benchmark move — span-consistent, which
    naive daily-return joins are not.
    """
    idx = equity.index
    aligned = {}
    for col in bench_prices.columns:
        px = bench_prices[col].dropna()
        px_asof = px.reindex(px.index.union(idx)).ffill().reindex(idx)
        aligned[col] = px_asof.pct_change()
    return pd.DataFrame(aligned, index=idx)


def lagged_beta_regression(strat_ret: pd.Series,
                           bench_ret: pd.DataFrame,
                           lags: int = 1) -> dict:
    """Dimson/Asness-style regression of strategy returns on benchmark
    returns at lags 0..lags (all benchmarks jointly), HAC errors.

    Returns per-benchmark contemporaneous beta, summed (lag-aggregated) beta
    with its HAC t-stat, plus alpha (per-period and annualized) and R².
    """
    df = pd.concat([strat_ret.rename('_y'), bench_ret], axis=1, sort=False).dropna()
    if len(df) < 20:
        raise ValueError(f"only {len(df)} aligned observations — need >= 20")
    y = df['_y'].values
    names = list(bench_ret.columns)
    cols, labels = [], []
    for name in names:
        for lag in range(lags + 1):
            cols.append(df[name].shift(lag).values)
            labels.append((name, lag))
    X = np.column_stack([np.ones(len(df))] + cols)
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]
    beta, se, resid = ols_hac(X, y)
    n = len(y)
    out = {'n_obs': int(n), 'lags': lags,
           'alpha_daily': float(beta[0]),
           'alpha_annual': float(beta[0] * ANNUALIZATION_DAYS),
           'alpha_t': float(beta[0] / se[0]) if se[0] > 0 else float('nan'),
           'r2': float(1.0 - resid.var() / y.var()) if y.var() > 0 else 0.0,
           'betas': {}}
    for name in names:
        idxs = [i + 1 for i, (nm, _) in enumerate(labels) if nm == name]
        b = beta[idxs]
        # variance of the sum via the same HAC covariance would need the full
        # matrix; conservative sum-of-se bound is reported alongside lag-0's
        # exact t-stat (the headline Asness-Krail-Liew number is the SUM beta)
        contemp_i = idxs[0]
        out['betas'][name] = {
            'contemporaneous': float(beta[contemp_i]),
            'contemporaneous_t': float(beta[contemp_i] / se[contemp_i])
            if se[contemp_i] > 0 else float('nan'),
            'summed': float(b.sum()),
            'per_lag': [float(v) for v in b],
        }
    return out


def up_down_betas(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Henriksson-Merton style up-/down-market betas vs one benchmark.

    A trend/stop-gated long book should show beta_up > beta_down (convex
    timing); symmetric betas mean the gates are not changing the exposure
    profile, just its average level.
    """
    df = pd.concat([strat_ret.rename('y'), bench_ret.rename('x')],
                   axis=1, sort=False).dropna()
    x, y = df['x'].values, df['y'].values
    X = np.column_stack([np.ones(len(x)),
                         np.where(x > 0, x, 0.0),
                         np.where(x <= 0, x, 0.0)])
    beta, se, _ = ols_hac(X, y)
    return {'beta_up': float(beta[1]), 'beta_down': float(beta[2]),
            'beta_up_t': float(beta[1] / se[1]) if se[1] > 0 else float('nan'),
            'beta_down_t': float(beta[2] / se[2]) if se[2] > 0 else float('nan'),
            'n_obs': int(len(x))}


def conditional_betas(strat_ret: pd.Series, bench_ret: pd.Series,
                      state: pd.Series, state_name: str = 'state') -> dict:
    """Contemporaneous beta split by a boolean condition series (e.g. SPY
    above/below its 200d SMA) — the Goulding-Harvey-Mazzoleni question:
    is the 'alpha' just beta held conditionally?"""
    df = pd.concat([strat_ret.rename('y'), bench_ret.rename('x'),
                    state.rename('s')], axis=1, sort=False).dropna()
    out = {}
    for label, mask in ((f'{state_name}_true', df['s'].astype(bool)),
                        (f'{state_name}_false', ~df['s'].astype(bool))):
        sub = df[mask]
        if len(sub) < 15:
            out[label] = {'beta': float('nan'), 'n_obs': int(len(sub))}
            continue
        X = np.column_stack([np.ones(len(sub)), sub['x'].values])
        beta, se, _ = ols_hac(X, sub['y'].values)
        out[label] = {'beta': float(beta[1]),
                      't': float(beta[1] / se[1]) if se[1] > 0 else float('nan'),
                      'n_obs': int(len(sub))}
    return out


def rolling_beta(strat_ret: pd.Series, bench_ret: pd.Series,
                 window: int = 30) -> pd.Series:
    """Rolling contemporaneous beta — the series a hedge overlay would size
    against (short bench notional = rolling_beta * book equity)."""
    df = pd.concat([strat_ret.rename('y'), bench_ret.rename('x')],
                   axis=1, sort=False).dropna()
    cov = df['y'].rolling(window, min_periods=max(10, window // 2)).cov(df['x'])
    var = df['x'].rolling(window, min_periods=max(10, window // 2)).var()
    return (cov / var.replace(0.0, np.nan)).rename('rolling_beta')


def beta_report(equity: pd.Series, bench_prices: pd.DataFrame,
                lags: int = 1, spy_col: str = 'SPY',
                btc_col: str = 'BTC') -> dict:
    """Full ledger: joint lagged regression + per-benchmark diagnostics."""
    equity = equity.dropna()
    strat_ret = equity.pct_change().dropna()
    bench_ret = align_benchmark_returns(equity, bench_prices).loc[strat_ret.index]

    report = {'period': {'start': str(strat_ret.index[0].date()),
                         'end': str(strat_ret.index[-1].date()),
                         'n_days': int(len(strat_ret))},
              'strategy': {
                  'ann_return': float(strat_ret.mean() * ANNUALIZATION_DAYS),
                  'ann_vol': float(strat_ret.std(ddof=1)
                                   * math.sqrt(ANNUALIZATION_DAYS)),
              }}
    sr_vol = report['strategy']['ann_vol']
    report['strategy']['sharpe'] = (report['strategy']['ann_return'] / sr_vol
                                    if sr_vol > 0 else float('nan'))

    report['joint'] = lagged_beta_regression(strat_ret, bench_ret, lags=lags)

    for col in bench_ret.columns:
        diag = {'up_down': up_down_betas(strat_ret, bench_ret[col])}
        px = bench_prices[col].dropna()
        if col == spy_col and len(px) >= 200:
            sma200 = px.rolling(200).mean()
            state = (px > sma200).reindex(strat_ret.index, method='ffill')
            diag['trend_conditional'] = conditional_betas(
                strat_ret, bench_ret[col], state, state_name='above_200d')
        rb = rolling_beta(strat_ret, bench_ret[col]).dropna()
        if len(rb):
            diag['rolling_beta_last'] = float(rb.iloc[-1])
        report[col] = diag
    return report


def format_report(report: dict) -> str:
    lines = []
    p = report['period']
    s = report['strategy']
    j = report['joint']
    lines.append(f"BETA LEDGER  {p['start']} .. {p['end']}  ({p['n_days']} trading days)")
    lines.append(f"  strategy: {s['ann_return']:+.1%}/yr at {s['ann_vol']:.1%} vol"
                 f"  (Sharpe {s['sharpe']:.2f})")
    lines.append(f"  joint regression (lags 0..{j['lags']}, Newey-West):"
                 f"  R^2 = {j['r2']:.2f}")
    lines.append(f"  alpha: {j['alpha_annual']:+.1%}/yr  (t = {j['alpha_t']:+.2f})"
                 f"   <-- the number that must be significant before any"
                 f" 'alpha' claim")
    for name, b in j['betas'].items():
        lines.append(f"  beta[{name}]: contemporaneous {b['contemporaneous']:+.3f}"
                     f" (t {b['contemporaneous_t']:+.2f}),"
                     f" summed(AKL) {b['summed']:+.3f}")
    for name in j['betas']:
        d = report.get(name, {})
        ud = d.get('up_down')
        if ud:
            lines.append(f"  {name} up/down beta: {ud['beta_up']:+.3f} /"
                         f" {ud['beta_down']:+.3f}"
                         f"   (convex timing iff up >> down)")
        tc = d.get('trend_conditional')
        if tc:
            above = tc.get('above_200d_true', {})
            below = tc.get('above_200d_false', {})
            lines.append(f"  {name} beta above/below 200d:"
                         f" {above.get('beta', float('nan')):+.3f} /"
                         f" {below.get('beta', float('nan')):+.3f}")
        if 'rolling_beta_last' in d:
            lines.append(f"  {name} rolling 30d beta now:"
                         f" {d['rolling_beta_last']:+.3f}"
                         f"   (hedge notional = this x book equity)")
    lines.append("  variance share of the factor block = R^2 above; the review's"
                 " structural estimate while invested was ~0.85-0.95.")
    return '\n'.join(lines)


# --- data loaders (network / broker; not exercised by unit tests) ---

def load_equity_alpaca(days: int) -> pd.Series:
    from trading_utils import get_api
    api = get_api()
    period = f"{max(1, math.ceil(days / 30))}M"
    hist = api.get_portfolio_history(period=period, timeframe='1D')
    ts = pd.to_datetime(pd.Series(list(hist.timestamp)), unit='s', utc=True)
    eq = pd.Series([float(v) for v in hist.equity], index=ts.dt.normalize())
    eq = eq[eq > 0]
    return eq.tail(days + 1).rename('equity')


def load_benchmarks_yf(days: int, spy='SPY', btc='BTC-USD') -> pd.DataFrame:
    import yfinance as yf
    # 200d SMA state needs deep SPY history regardless of the report window
    lookback = max(days + 5, 420)
    px = {}
    for label, sym in (('SPY', spy), ('BTC', btc)):
        h = yf.download(sym, period=f"{lookback}d", interval='1d',
                        progress=False, auto_adjust=True)
        if h is None or h.empty:
            continue
        close = h['Close']
        if hasattr(close, 'columns'):  # yfinance MultiIndex quirk
            close = close.iloc[:, 0]
        idx = pd.to_datetime(close.index)
        idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
        px[label] = pd.Series(close.values, index=idx.normalize())
    return pd.DataFrame(px)


def _read_csv_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    idx = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    return df.drop(columns=[date_col]).set_index(idx)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--days', type=int, default=90)
    ap.add_argument('--lags', type=int, default=1,
                    help='benchmark lags in the joint regression (AKL)')
    ap.add_argument('--equity-csv', help='offline equity curve (date,equity)')
    ap.add_argument('--benchmarks-csv',
                    help='offline benchmark PRICES (date,SPY,BTC)')
    ap.add_argument('--json', help='also write the report dict to this path')
    args = ap.parse_args()

    if args.equity_csv:
        eq_df = _read_csv_series(args.equity_csv)
        equity = eq_df[eq_df.columns[0]].astype(float).rename('equity')
    else:
        equity = load_equity_alpaca(args.days)
    equity = equity.tail(args.days + 1)

    if args.benchmarks_csv:
        bench = _read_csv_series(args.benchmarks_csv).astype(float)
    else:
        bench = load_benchmarks_yf(args.days)
    if bench.empty:
        raise SystemExit('no benchmark data available')

    rep = beta_report(equity, bench, lags=args.lags)
    print(format_report(rep))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(rep, f, indent=2, default=str)
        print(f"[beta_ledger] wrote {args.json}")
