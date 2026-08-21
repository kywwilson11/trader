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
import os
import sys

import numpy as np
import pandas as pd

ANNUALIZATION_DAYS = 252

MIN_OBS = 20                   # joint-regression floor, enforced POST lag-shift
MIN_BUCKET_OBS = 15            # per-bucket floor shared by up/down and conditional splits
FLAT_RETURN_EPS = 1e-9         # |daily return| below this => flat/uninvested day
OUTLIER_DAILY_RETURN = 0.15    # |daily return| above this => deposit/reset warning
ROLLING_BETA_WINDOW = 30       # hedge-sizing window reported per benchmark
RESERVED_REPORT_KEYS = {'period', 'strategy', 'joint', 'joint_active', 'joint_clean',
                        'warnings', 'data_quality', 'excluded_benchmarks',
                        'schema_version', '_y'}


# --- regression core (pure) ---

def ols_hac(X: np.ndarray, y: np.ndarray, hac_lags: int | None = None,
           return_extras: bool = False):
    """OLS with Newey-West (Bartlett) HAC standard errors.

    X must already include the intercept column. Returns (beta, se, resid).
    hac_lags=None -> Newey-West (1994) automatic plug-in floor(4*(n/100)^(2/9)).

    return_extras=True appends a fourth element: a dict with the full HAC
    covariance ('cov'), the resolved bandwidth ('hac_lags'), and the lstsq
    design rank ('rank'). The default 3-tuple contract is unchanged.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    if n <= k + 1:
        raise ValueError(f"too few observations ({n}) for {k} regressors")
    beta, _ss, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
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
    if return_extras:
        return beta, se, resid, {'cov': cov, 'hac_lags': int(hac_lags),
                                 'rank': int(rank)}
    return beta, se, resid


def align_benchmark_returns(equity: pd.Series,
                            bench_prices: pd.DataFrame) -> pd.DataFrame:
    """Benchmark returns over the SAME date-to-date spans as the equity curve.

    The equity series lives on the broker's trading-day grid while a 24/7
    benchmark (BTC) has calendar-day prices. As-of aligning PRICES to the
    equity dates first, then differencing, makes a Friday->Monday equity move
    regress against the Friday->Monday benchmark move — span-consistent, which
    naive daily-return joins are not.

    The ffill is bounded by each benchmark's own coverage: it never
    extrapolates past that benchmark's last real observation.
    """
    idx = equity.index
    aligned = {}
    for col in bench_prices.columns:
        px = bench_prices[col].dropna()
        px_asof = px.reindex(px.index.union(idx)).ffill().reindex(idx)
        # pandas pin (c26-T3/B21): px_asof is already ffilled above —
        # fill_method=None == pandas-2 pad semantics, pandas-3-proof
        ret = px_asof.pct_change(fill_method=None)
        if len(px):
            # never extrapolate past the benchmark's last real observation —
            # a trailing ffill fabricates exact-0.0 returns for days the vendor
            # has not published, biasing beta toward 0 and the residual into alpha
            ret[idx > px.index.max()] = np.nan
        aligned[col] = ret
    return pd.DataFrame(aligned, index=idx)


def clean_returns_from_pl(equity: pd.Series, profit_loss: pd.Series) -> pd.Series:
    """Per-period return with transfers removed: pl[t] / equity[t-1] on the
    equity grid. Alpaca's profit_loss excludes deposits/withdrawals, so a
    top-up day shows ~0 here while raw pct_change fabricates a huge 'return'.
    NaN where pl is missing/non-finite or prior equity is not > 0."""
    eq = equity.dropna()
    eq = eq[~eq.index.duplicated(keep='last')].sort_index()
    pl = profit_loss.reindex(eq.index)
    prev = eq.shift(1)
    r = pl / prev.where(prev > 0)
    r = r.mask(~np.isfinite(r.values.astype(float)))
    return r.rename('clean_ret')


def lagged_beta_regression(strat_ret: pd.Series,
                           bench_ret: pd.DataFrame,
                           lags: int = 1) -> dict:
    """Dimson/Asness-style regression of strategy returns on benchmark
    returns at lags 0..lags (all benchmarks jointly), HAC errors.

    Returns per-benchmark contemporaneous beta, summed (lag-aggregated) beta
    with its exact HAC se/t (summed_se/summed_t, computed from the full HAC
    covariance — not a conservative bound), plus alpha (per-period and
    annualized) and R².
    """
    if lags < 0:
        raise ValueError('lags must be >= 0')
    df = pd.concat([strat_ret.rename('_y'), bench_ret], axis=1, sort=False).dropna()
    names = list(bench_ret.columns)
    cols, labels = [], []
    for name in names:
        for lag in range(lags + 1):
            cols.append(df[name].shift(lag).values)
            labels.append((name, lag))
    X = np.column_stack([np.ones(len(df))] + cols)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(df['_y'].values)
    X = X[valid]
    y = df['_y'].values[valid]
    k = X.shape[1]
    n = len(y)
    if n < MIN_OBS:
        raise ValueError(f"only {n} usable observations after {lags} lag(s) "
                         f"for {k} regressors — need >= {MIN_OBS}")
    beta, se, resid, extras = ols_hac(X, y, return_extras=True)
    cov = extras['cov']
    yv = y.var()
    r2 = (float(1.0 - resid.var() / yv)
         if np.isfinite(yv) and yv > 0 else float('nan'))
    used_idx = df.index[valid]
    g = used_idx.to_series().diff().dt.days.dropna()
    out = {'n_obs': int(n), 'lags': lags,
           'alpha_daily': float(beta[0]),
           'alpha_annual': float(beta[0] * ANNUALIZATION_DAYS),
           'alpha_t': float(beta[0] / se[0]) if se[0] > 0 else float('nan'),
           'alpha_se': float(se[0]),
           'r2': r2,
           'hac_lags': int(extras['hac_lags']),
           'rank': int(extras['rank']),
           'rank_deficient': bool(extras['rank'] < k),
           'obs_per_param': float(n / k),
           'underpowered': bool(n / k < 10),
           'first_obs': str(used_idx[0].date()),
           'last_obs': str(used_idx[-1].date()),
           'index_regularity': {
               'median_gap_days': float(g.median()) if len(g) else float('nan'),
               'max_gap_days': float(g.max()) if len(g) else float('nan'),
           },
           'betas': {}}
    # Finite-sample d.o.f. correction (module-improve-v3 owner item):
    # statsmodels' HAC default scales the sandwich by n/(n-k); the base
    # keys keep the uncorrected convention — these are ADDITIVE parallels.
    dof_scale = math.sqrt(n / (n - k)) if n > k else float('nan')
    out['dof_scale'] = float(dof_scale) if np.isfinite(dof_scale) else float('nan')
    out['alpha_se_corrected'] = (float(se[0] * dof_scale)
                                 if np.isfinite(dof_scale) else float('nan'))
    out['alpha_t_corrected'] = (float(beta[0] / (se[0] * dof_scale))
                                if se[0] > 0 and np.isfinite(dof_scale)
                                else float('nan'))
    for name in names:
        idxs = [i + 1 for i, (nm, _) in enumerate(labels) if nm == name]
        b = beta[idxs]
        contemp_i = idxs[0]
        w = np.zeros(k)
        w[idxs] = 1.0
        summed = float(b.sum())
        summed_se = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
        out['betas'][name] = {
            'contemporaneous': float(beta[contemp_i]),
            'contemporaneous_t': float(beta[contemp_i] / se[contemp_i])
            if se[contemp_i] > 0 else float('nan'),
            'summed': summed,
            'summed_se': summed_se,
            'summed_t': float(summed / summed_se) if summed_se > 0 else float('nan'),
            'per_lag': [float(v) for v in b],
        }
    return out


def up_down_betas(strat_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Henriksson-Merton style up-/down-market betas vs one benchmark.

    A trend/stop-gated long book should show beta_up > beta_down (convex
    timing); symmetric betas mean the gates are not changing the exposure
    profile, just its average level. A side with fewer than MIN_BUCKET_OBS
    observations reports nan — an unidentified coefficient, not a measured
    zero (a near-empty side previously reported lstsq's min-norm artifact,
    which reads as a confident zero exposure that was never measured).
    """
    df = pd.concat([strat_ret.rename('y'), bench_ret.rename('x')],
                   axis=1, sort=False).dropna()
    x, y = df['x'].values, df['y'].values
    n_up = int((x > 0).sum())
    n_down = int((x <= 0).sum())
    X = np.column_stack([np.ones(len(x)),
                         np.where(x > 0, x, 0.0),
                         np.where(x <= 0, x, 0.0)])
    beta, se, _ = ols_hac(X, y)
    beta_up = float(beta[1])
    beta_down = float(beta[2])
    beta_up_t = float(beta[1] / se[1]) if se[1] > 0 else float('nan')
    beta_down_t = float(beta[2] / se[2]) if se[2] > 0 else float('nan')
    if n_up < MIN_BUCKET_OBS:
        beta_up = beta_up_t = float('nan')
    if n_down < MIN_BUCKET_OBS:
        beta_down = beta_down_t = float('nan')
    return {'beta_up': beta_up, 'beta_down': beta_down,
            'beta_up_t': beta_up_t, 'beta_down_t': beta_down_t,
            'n_obs': int(len(x)), 'n_up': n_up, 'n_down': n_down}


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
        if len(sub) < MIN_BUCKET_OBS:
            out[label] = {'beta': float('nan'), 't': float('nan'),
                         'n_obs': int(len(sub)), 'insufficient': True}
            continue
        X = np.column_stack([np.ones(len(sub)), sub['x'].values])
        beta, se, _ = ols_hac(X, sub['y'].values)
        out[label] = {'beta': float(beta[1]),
                      't': float(beta[1] / se[1]) if se[1] > 0 else float('nan'),
                      'n_obs': int(len(sub)), 'insufficient': False}
    return out


def rolling_beta(strat_ret: pd.Series, bench_ret: pd.Series,
                 window: int = ROLLING_BETA_WINDOW) -> pd.Series:
    """Rolling contemporaneous beta — the series a hedge overlay would size
    against (short bench notional = rolling_beta * book equity)."""
    df = pd.concat([strat_ret.rename('y'), bench_ret.rename('x')],
                   axis=1, sort=False).dropna()
    cov = df['y'].rolling(window, min_periods=max(10, window // 2)).cov(df['x'])
    var = df['x'].rolling(window, min_periods=max(10, window // 2)).var()
    return (cov / var.replace(0.0, np.nan)).rename('rolling_beta')


def beta_report(equity: pd.Series, bench_prices: pd.DataFrame,
                lags: int = 1, clean_ret: pd.Series | None = None) -> dict:
    """Full ledger: joint lagged regression + per-benchmark diagnostics.

    clean_ret: optional per-period transfer-clean return series on the equity
    grid (see clean_returns_from_pl); when given, additive *_clean keys are
    emitted alongside the unchanged raw keys."""
    bad = RESERVED_REPORT_KEYS & set(map(str, bench_prices.columns))
    if bad:
        raise ValueError(f"benchmark column names collide with report keys: "
                         f"{sorted(bad)}")

    # --- input normalization (no-op on already-clean input) ---
    equity = equity.dropna()
    n_raw = len(equity)
    deduped = equity[~equity.index.duplicated(keep='last')].sort_index()
    n_dup = n_raw - len(deduped)
    equity = deduped
    finite_pos = equity[np.isfinite(equity.values) & (equity.values > 0)]
    n_nonpos = len(equity) - len(finite_pos)
    equity = finite_pos
    bench_prices = bench_prices[~bench_prices.index.duplicated(keep='last')].sort_index()

    strat_ret = equity.pct_change(fill_method=None).dropna()
    n_ret_before = len(strat_ret)
    strat_ret = strat_ret[np.isfinite(strat_ret.values)]
    n_nonfinite_ret = n_ret_before - len(strat_ret)
    if len(strat_ret) < MIN_OBS:
        raise ValueError(f"only {len(strat_ret)} usable equity return days — "
                         f"need >= {MIN_OBS} (equity had {n_raw} points)")

    bench_ret = align_benchmark_returns(equity, bench_prices).loc[strat_ret.index]
    # a zero/negative benchmark PRICE (CSV path) makes pct_change emit ±inf,
    # which dropna() does NOT catch — mask to NaN once here so every consumer
    # (joint, up/down, conditional, rolling) sees one consistent missing value
    n_inf_bench = int(np.isinf(bench_ret.values).sum())
    if n_inf_bench:
        bench_ret = bench_ret.mask(np.isinf(bench_ret))
    excluded = {}
    for col in list(bench_ret.columns):
        n_valid = int(bench_ret[col].notna().sum())
        if n_valid < MIN_OBS:
            excluded[col] = n_valid
            bench_ret = bench_ret.drop(columns=[col])
    if bench_ret.shape[1] == 0:
        raise ValueError(f"no benchmark has >= {MIN_OBS} aligned observations "
                         f"(excluded: {excluded})")

    report = {'period': {'start': str(strat_ret.index[0].date()),
                         'end': str(strat_ret.index[-1].date()),
                         'n_days': int(len(strat_ret))}}
    gaps = strat_ret.index.to_series().diff().dt.days.dropna()
    report['period']['median_spacing_days'] = (float(gaps.median())
                                                if len(gaps) else float('nan'))
    span_days = max(1, (strat_ret.index[-1] - strat_ret.index[0]).days)
    obs_per_year = float(len(strat_ret) * 365.25 / span_days)
    report['period']['obs_per_year'] = obs_per_year
    # Median-spacing rate — same definition as chart_core.obs_per_year
    # (SECONDS_PER_YEAR / median spacing): 365.25/yr for a calendar-daily
    # grid, ~252-adjacent only if the grid truly skips weekends in its
    # MEDIAN gap (a b-daily grid's median gap is 1 day => 365.25 — the
    # span-based obs_per_year above is the trading-day-count view).
    med_gap = float(gaps.median()) if len(gaps) else float('nan')
    report['period']['obs_per_year_grid'] = (
        float(365.25 / med_gap) if np.isfinite(med_gap) and med_gap > 0
        else float('nan'))
    report['period']['annualization_days'] = ANNUALIZATION_DAYS

    abs_ret = strat_ret.abs()
    flat_mask = abs_ret < FLAT_RETURN_EPS
    outlier_mask = abs_ret > OUTLIER_DAILY_RETURN
    report['strategy'] = {
        'ann_return': float(strat_ret.mean() * ANNUALIZATION_DAYS),
        'ann_vol': float(strat_ret.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)),
    }
    sr_vol = report['strategy']['ann_vol']
    report['strategy']['sharpe'] = (report['strategy']['ann_return'] / sr_vol
                                    if sr_vol > 0 else float('nan'))
    report['strategy']['flat_day_share'] = float(flat_mask.mean())
    report['strategy']['n_flat_days'] = int(flat_mask.sum())
    report['strategy']['max_abs_daily_return'] = float(abs_ret.max())
    report['strategy']['n_outlier_days'] = int(outlier_mask.sum())

    warnings: list[str] = []
    report['warnings'] = warnings

    if outlier_mask.any():
        dates = [str(d.date()) for d in strat_ret.index[outlier_mask][:5]]
        warnings.append(
            f"{int(outlier_mask.sum())} day(s) with |daily return| > "
            f"{OUTLIER_DAILY_RETURN:.0%} ({', '.join(dates)}) — possible "
            f"deposit/withdrawal or paper-account reset, not P&L")
    for name, n_valid in excluded.items():
        warnings.append(f"benchmark '{name}' excluded: only {n_valid} aligned "
                        f"observations (need >= {MIN_OBS})")
    if n_inf_bench:
        warnings.append(f"{n_inf_bench} non-finite benchmark return(s) masked "
                        f"to NaN — check for zero/negative benchmark prices")
    if abs(obs_per_year - ANNUALIZATION_DAYS) > 25:
        warnings.append(f"obs_per_year={obs_per_year:.0f} far from "
                        f"ANNUALIZATION_DAYS={ANNUALIZATION_DAYS} — annualized "
                        f"figures may be mis-scaled for this sampling frequency")

    report['joint'] = lagged_beta_regression(strat_ret, bench_ret, lags=lags)
    report['period']['n_obs_used'] = report['joint']['n_obs']
    coverage_pct = round(report['joint']['n_obs'] / max(1, len(strat_ret)), 3)
    report['period']['coverage_pct'] = coverage_pct
    if coverage_pct < 0.9:
        warnings.append(f"joint regression used only {report['joint']['n_obs']}"
                        f" of {len(strat_ret)} days ({coverage_pct:.0%} coverage)")
    if report['joint'].get('underpowered'):
        warnings.append(f"joint regression is underpowered: "
                        f"{report['joint']['obs_per_param']:.1f} observations "
                        f"per parameter")

    # --- ADDITIVE transfer-clean parallel (owner item R4): same bench_ret,
    # same lags; raw keys above are byte-identical with or without this ---
    if clean_ret is not None:
        cr = clean_ret.reindex(strat_ret.index)
        cr = cr[np.isfinite(cr.values.astype(float))]
        if len(cr) >= MIN_OBS:
            report['strategy']['ann_return_clean'] = float(cr.mean() * ANNUALIZATION_DAYS)
            cv = float(cr.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS))
            report['strategy']['ann_vol_clean'] = cv
            report['strategy']['sharpe_clean'] = (
                report['strategy']['ann_return_clean'] / cv if cv > 0 else float('nan'))
            try:
                jc = lagged_beta_regression(cr, bench_ret.loc[cr.index], lags=lags)
                report['joint_clean'] = jc
                report['joint']['alpha_annual_clean'] = jc['alpha_annual']
                report['joint']['alpha_t_clean'] = jc['alpha_t']
                report['joint']['n_obs_clean'] = jc['n_obs']
                report['joint']['contamination_delta'] = float(
                    jc['alpha_annual'] - report['joint']['alpha_annual'])
            except ValueError as e:
                warnings.append(f"clean-return joint regression skipped: {e}")
        else:
            warnings.append(f"clean-return series has only {len(cr)} usable "
                            f"observations (need >= {MIN_OBS}) — *_clean keys omitted")

    # R² over invested days only vs all days (Goulding-Harvey-Mazzoleni: is
    # the unconditional R² diluted by days the book was flat/uninvested?)
    if bool(flat_mask.any()):
        try:
            report['joint_active'] = lagged_beta_regression(
                strat_ret[~flat_mask], bench_ret[~flat_mask], lags=lags)
        except ValueError:
            pass

    for col in bench_ret.columns:
        diag = {'up_down': up_down_betas(strat_ret, bench_ret[col])}

        col_ret = bench_ret[col].dropna()
        if float(col_ret.std(ddof=0)) == 0.0:
            diag['degenerate'] = True
            warnings.append(f"benchmark '{col}' returns have zero variance "
                            f"over the window — beta is NOT measured")

        px = bench_prices[col].dropna()
        if len(px):
            diag['bench_last_date'] = str(px.index.max().date())
            stale_days = int((strat_ret.index > px.index.max()).sum())
            diag['stale_days'] = stale_days
            if stale_days > 0:
                warnings.append(
                    f"benchmark '{col}' has {stale_days} day(s) past its last "
                    f"observation ({diag['bench_last_date']}) — those returns "
                    f"are NaN, not extrapolated")

        if len(px) >= 200:
            sma200 = px.rolling(200).mean()
            # .where() is the warm-up fix: (px > NaN) is False, which would
            # silently classify undefined-SMA days as "below 200d" — .where
            # makes them NaN so conditional_betas' dropna excludes them
            state = (px > sma200).where(sma200.notna()).reindex(
                strat_ret.index, method='ffill')
            diag['trend_conditional'] = conditional_betas(
                strat_ret, bench_ret[col], state, state_name='above_200d')
            diag['trend_conditional']['n_state_undefined'] = int(state.isna().sum())
            # PIT variant (owner-decision rollout): state known at t-1 —
            # SMA-crossing days no longer classified by their own move.
            # Emitted ALONGSIDE trend_conditional for one measurement
            # cycle; the canonical pick is an owner decision.
            state_lag = (px > sma200).where(sma200.notna()).shift(1).reindex(
                strat_ret.index, method='ffill')
            diag['trend_conditional_lagged'] = conditional_betas(
                strat_ret, bench_ret[col], state_lag, state_name='above_200d')
            diag['trend_conditional_lagged']['n_state_undefined'] = int(
                state_lag.isna().sum())

        rb = rolling_beta(strat_ret, bench_ret[col],
                          window=ROLLING_BETA_WINDOW).dropna()
        if len(rb):
            diag['rolling_beta_last'] = float(rb.iloc[-1])
            diag['rolling_beta'] = {
                'window': ROLLING_BETA_WINDOW,
                'last': float(rb.iloc[-1]),
                'asof': str(rb.index[-1].date()),
                'mean': float(rb.mean()),
                'std': float(rb.std(ddof=1)),
                'min': float(rb.min()),
                'max': float(rb.max()),
                'n': int(len(rb)),
            }
        report[col] = diag

    report['data_quality'] = {
        'n_duplicate_days_dropped': int(n_dup),
        'n_nonpositive_equity_dropped': int(n_nonpos),
        'n_nonfinite_returns_dropped': int(n_nonfinite_ret),
        'n_nonfinite_bench_returns_masked': n_inf_bench,
    }
    report['excluded_benchmarks'] = excluded
    report['schema_version'] = 1
    return report


def format_report(report: dict) -> str:
    lines = []
    p = report['period']
    s = report['strategy']
    j = report['joint']
    lines.append(f"BETA LEDGER  {p['start']} .. {p['end']}  "
                 f"({p['n_days']} days, {p.get('n_obs_used', p['n_days'])} in regression)")
    for w in report.get('warnings', []):
        lines.append(f"  WARNING: {w}")
    lines.append(f"  strategy: {s['ann_return']:+.1%}/yr at {s['ann_vol']:.1%} vol"
                 f"  (Sharpe {s['sharpe']:.2f})")
    lines.append(f"  joint regression (lags 0..{j['lags']}, Newey-West "
                 f"L={j.get('hac_lags', '?')}):  R^2 = {j['r2']:.2f}")
    lines.append(f"  alpha: {j['alpha_annual']:+.1%}/yr  (t = {j['alpha_t']:+.2f})"
                 f"   <-- the number that must be significant before any"
                 f" 'alpha' claim")
    if 'alpha_t_corrected' in j:
        lines.append(f"  alpha t (n/(n-k) corrected): {j['alpha_t_corrected']:+.2f}")
    if 'alpha_annual_clean' in j:
        lines.append(f"  alpha CLEAN (transfer-free): "
                     f"{j['alpha_annual_clean']:+.1%}/yr "
                     f"(t {j.get('alpha_t_clean', float('nan')):+.2f})  "
                     f"contamination delta "
                     f"{j.get('contamination_delta', float('nan')):+.1%}/yr")
    for name, b in j['betas'].items():
        summed_t = b.get('summed_t', float('nan'))
        lines.append(f"  beta[{name}]: contemporaneous {b['contemporaneous']:+.3f}"
                     f" (t {b['contemporaneous_t']:+.2f}),"
                     f" summed(AKL) {b['summed']:+.3f} (t {summed_t:+.2f})")
    for name in j['betas']:
        d = report.get(name, {})
        ud = d.get('up_down')
        if ud:
            lines.append(f"  {name} up/down beta: {ud['beta_up']:+.3f} /"
                         f" {ud['beta_down']:+.3f}"
                         f" (n {ud.get('n_up', '?')}/{ud.get('n_down', '?')})"
                         f"   (convex timing iff up >> down)")
        tc = d.get('trend_conditional')
        if tc:
            above = tc.get('above_200d_true', {})
            below = tc.get('above_200d_false', {})
            lines.append(f"  {name} beta above/below 200d:"
                         f" {above.get('beta', float('nan')):+.3f} /"
                         f" {below.get('beta', float('nan')):+.3f}")
        tcl = d.get('trend_conditional_lagged')
        if tcl:
            above = tcl.get('above_200d_true', {})
            below = tcl.get('above_200d_false', {})
            lines.append(f"  {name} beta above/below 200d (PIT lag-1):"
                         f" {above.get('beta', float('nan')):+.3f} /"
                         f" {below.get('beta', float('nan')):+.3f}")
        if 'rolling_beta_last' in d:
            asof_txt = ''
            if 'rolling_beta' in d:
                asof_txt = f" (as of {d['rolling_beta']['asof']})"
            lines.append(f"  {name} rolling 30d beta now:"
                         f" {d['rolling_beta_last']:+.3f}{asof_txt}"
                         f"   (hedge notional = this x book equity)")
    if 'joint_active' in report:
        ja = report['joint_active']
        lines.append(
            f"  variance share of the factor block: R^2 = {j['r2']:.2f} over"
            f" all days, {ja['r2']:.2f} over the {ja['n_obs']} invested days"
            f" (flat {s.get('flat_day_share', 0.0):.0%} of the window); the"
            f" review's structural estimate while invested was ~0.85-0.95.")
    else:
        lines.append(
            f"  variance share of the factor block: R^2 = {j['r2']:.2f}; the"
            f" review's structural estimate while invested was ~0.85-0.95.")
    return '\n'.join(lines)


# --- data loaders (network / broker; not exercised by unit tests) ---

def _period_for_days(days: int) -> str:
    """Alpaca portfolio-history period covering `days` DAILY observations.

    Alpaca months are calendar months (~21 trading days), so ceil(days/30)
    months under-delivers ~30% on a trading-day grid ('90d' fetched ~63).
    Request generously (the caller tails to the exact count) and switch to
    year units past 12 months ('13M' is not a documented period).
    """
    months = max(1, math.ceil(days / 20) + 1)
    if months > 12:
        return f"{math.ceil(months / 12)}A"
    return f"{months}M"


def load_equity_alpaca(days: int, with_pl: bool = False):
    """Daily equity Series from Alpaca portfolio history. with_pl=True returns
    (equity, profit_loss_or_None) — pl is the per-period P&L series (transfer-
    free), None when the API exposes no usable profit_loss."""
    from trading_utils import get_api
    api = get_api()
    hist = api.get_portfolio_history(period=_period_for_days(days), timeframe='1D')
    raw_ts, raw_eq = list(hist.timestamp or []), list(hist.equity or [])
    keep = [i for i in range(min(len(raw_ts), len(raw_eq))) if raw_eq[i] is not None]
    ts = pd.to_datetime(pd.Series([raw_ts[i] for i in keep]), unit='s', utc=True)
    eq = pd.Series([float(raw_eq[i]) for i in keep], index=ts.dt.normalize())
    eq = eq[np.isfinite(eq.values) & (eq.values > 0)]
    eq = eq[~eq.index.duplicated(keep='last')].sort_index()
    if eq.empty:
        raise ValueError('Alpaca portfolio history returned no positive equity points')
    eq = eq.tail(days + 1).rename('equity')
    if not with_pl:
        return eq
    raw_pl = list(getattr(hist, 'profit_loss', None) or [])
    pl = None
    if raw_pl:
        vals = [float(raw_pl[i]) if i < len(raw_pl) and raw_pl[i] is not None
                else np.nan for i in keep]
        pl_full = pd.Series(vals, index=ts.dt.normalize())
        pl_full = pl_full[~pl_full.index.duplicated(keep='last')].sort_index()
        pl_full = pl_full.tail(days + 1).rename('profit_loss')
        # do not re-filter by the equity positivity mask — align by reindex
        pl_full = pl_full.reindex(eq.index)
        if pl_full.notna().any():
            pl = pl_full
    return eq, pl


def _bench_days_for_equity(days: int, equity: pd.Series) -> int:
    """yfinance fetch window: at least `days`, stretched to the actual equity
    span — a whole-file --equity-csv (no explicit --days) can cover far more
    than the 90d default, and fetching only 90d of benchmarks would regress
    years of equity on a sliver of benchmark coverage."""
    if len(equity) > 1:
        return max(days, int((equity.index[-1] - equity.index[0]).days) + 1)
    return days


def load_benchmarks_yf(days: int, spy='SPY', btc='BTC-USD') -> pd.DataFrame:
    import yfinance as yf
    # 200 TRADING days of SMA warm-up is ~290 calendar days, and that must be
    # provisioned ON TOP of the report window — the old flat max(days+5, 420)
    # capped at 420 for every days <= 415, leaving only ~90 defined-SMA days.
    lookback = max(int(days * 1.5) + 300, 420)
    px = {}
    for label, sym in (('SPY', spy), ('BTC', btc)):
        h = yf.download(sym, period=f"{lookback}d", interval='1d',
                        progress=False, auto_adjust=True)
        if h is None or h.empty:
            print(f"[beta_ledger] WARNING: no data for {sym} — skipping {label}",
                 file=sys.stderr)
            continue
        close = h['Close']
        if hasattr(close, 'columns'):  # yfinance MultiIndex quirk
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            print(f"[beta_ledger] WARNING: no data for {sym} — skipping {label}",
                 file=sys.stderr)
            continue
        idx = pd.to_datetime(close.index)
        idx = idx.tz_localize('UTC') if idx.tz is None else idx.tz_convert('UTC')
        px[label] = pd.Series(close.values, index=idx.normalize())
    return pd.DataFrame(px)


def _read_csv_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    try:
        idx = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    except (ValueError, TypeError) as e:
        raise SystemExit(f"[beta_ledger] {path}: column '{date_col}' is not "
                        f"parseable as dates ({e})")
    out = df.drop(columns=[date_col]).set_index(idx).sort_index()
    if out.shape[1] == 0:
        raise SystemExit(f"[beta_ledger] {path}: no value columns besides "
                         f"the date column '{date_col}'")
    return out[~out.index.duplicated(keep='last')]


def _json_safe(obj):
    """Non-finite floats -> None so --json is valid RFC-8259 (bare NaN
    breaks jq/JS); gui renders null as 'n/a', same as NaN today."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--days', type=int, default=None,
                    help='(default 90; a full --equity-csv is analyzed whole '
                         'unless --days is given)')
    ap.add_argument('--lags', type=int, default=1,
                    help='benchmark lags in the joint regression (AKL)')
    ap.add_argument('--equity-csv', help='offline equity curve (date,equity)')
    ap.add_argument('--benchmarks-csv',
                    help='offline benchmark PRICES (date,SPY,BTC)')
    ap.add_argument('--json', help='also write the report dict to this path')
    args = ap.parse_args()

    days = args.days if args.days is not None else 90
    if days < 2:
        ap.error('--days must be >= 2')
    if not 0 <= args.lags <= 10:
        ap.error('--lags must be between 0 and 10')

    pl = None
    if args.equity_csv:
        eq_df = _read_csv_series(args.equity_csv)
        equity = eq_df[eq_df.columns[0]].astype(float).rename('equity')
        if args.days is not None:
            equity = equity.tail(days + 1)
    else:
        equity, pl = load_equity_alpaca(days, with_pl=True)

    if args.benchmarks_csv:
        bench = _read_csv_series(args.benchmarks_csv).astype(float)
    else:
        bench = load_benchmarks_yf(_bench_days_for_equity(days, equity))
    if bench.empty:
        raise SystemExit('no benchmark data available')

    clean = clean_returns_from_pl(equity, pl) if pl is not None else None

    try:
        rep = beta_report(equity, bench, lags=args.lags, clean_ret=clean)
    except ValueError as e:
        raise SystemExit(f'[beta_ledger] {e}')

    if args.json:
        tmp = args.json + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(_json_safe(rep), f, indent=2, allow_nan=False, default=str)
        os.replace(tmp, args.json)
        print(f"[beta_ledger] wrote {args.json}")
    print(format_report(rep))
