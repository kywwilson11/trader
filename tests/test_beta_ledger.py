"""beta_ledger regression-core tests — synthetic data, pure numpy/pandas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beta_ledger import (
    ols_hac, align_benchmark_returns, lagged_beta_regression,
    up_down_betas, conditional_betas, rolling_beta, beta_report,
    format_report,
)

N = 400
DATES = pd.bdate_range('2025-01-02', periods=N, tz='UTC')


def _rng(seed=42):
    # per-test generators: shared module state would make the synthetic
    # draws depend on test execution order
    return np.random.default_rng(seed)


def _bench_returns(seed=42):
    rng = _rng(seed)
    spy = pd.Series(rng.normal(0.0004, 0.010, N), index=DATES, name='SPY')
    btc = pd.Series(rng.normal(0.0010, 0.025, N), index=DATES, name='BTC')
    return spy, btc


def _prices_from_returns(ret: pd.Series, p0=100.0) -> pd.Series:
    return p0 * (1 + ret).cumprod()


def test_ols_hac_recovers_coefficients():
    rng = _rng(1)
    x = rng.normal(0, 1, N)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, N)
    X = np.column_stack([np.ones(N), x])
    beta, se, resid = ols_hac(X, y)
    assert abs(beta[0] - 0.5) < 0.1
    assert abs(beta[1] - 2.0) < 0.1
    assert np.all(se > 0) and np.all(np.isfinite(se))


def test_joint_regression_recovers_planted_betas_and_alpha():
    spy, btc = _bench_returns(seed=2)
    rng = _rng(3)
    alpha = 0.0004  # ~10%/yr planted alpha
    strat = alpha + 0.35 * spy + 0.15 * btc \
        + pd.Series(rng.normal(0, 0.002, N), index=DATES)
    bench = pd.concat([spy, btc], axis=1)
    out = lagged_beta_regression(strat, bench, lags=1)
    assert abs(out['betas']['SPY']['summed'] - 0.35) < 0.08
    assert abs(out['betas']['BTC']['summed'] - 0.15) < 0.05
    assert abs(out['alpha_daily'] - alpha) < 2.5e-4
    assert out['alpha_t'] > 2.0          # planted alpha is detectable
    assert out['r2'] > 0.75              # factor block dominates variance


def test_lagged_betas_catch_stale_exposure():
    # Asness-Krail-Liew: exposure hidden at lag 1 must show in the SUM
    spy, _ = _bench_returns(seed=4)
    rng = _rng(5)
    strat = 0.2 * spy + 0.2 * spy.shift(1) \
        + pd.Series(rng.normal(0, 0.002, N), index=DATES)
    bench = spy.to_frame()
    lag0 = lagged_beta_regression(strat.dropna(), bench, lags=0)
    lag1 = lagged_beta_regression(strat.dropna(), bench, lags=1)
    assert abs(lag0['betas']['SPY']['summed'] - 0.2) < 0.08   # misses half
    assert abs(lag1['betas']['SPY']['summed'] - 0.4) < 0.08   # catches it


def test_up_down_beta_asymmetry_detected():
    spy, _ = _bench_returns(seed=6)
    rng = _rng(7)
    # convex book: full beta in up markets, none in down (perfect timing)
    strat = spy.clip(lower=0.0) \
        + pd.Series(rng.normal(0, 0.001, N), index=DATES)
    ud = up_down_betas(strat, spy)
    assert ud['beta_up'] > 0.8
    assert abs(ud['beta_down']) < 0.15


def test_conditional_beta_split():
    spy, _ = _bench_returns(seed=8)
    rng = _rng(9)
    state = pd.Series(np.arange(N) < N // 2, index=DATES)  # first half "true"
    strat = pd.Series(np.where(state, 0.8, 0.1), index=DATES) * spy \
        + pd.Series(rng.normal(0, 0.001, N), index=DATES)
    out = conditional_betas(strat, spy, state, state_name='regime')
    assert abs(out['regime_true']['beta'] - 0.8) < 0.1
    assert abs(out['regime_false']['beta'] - 0.1) < 0.1


def test_align_benchmark_returns_spans_gaps():
    # Equity on a trading-day grid; benchmark prices on ALL calendar days.
    # The Friday->Monday equity move must regress on the Friday->Monday
    # benchmark move (3 calendar days), not on Monday's 1-day move.
    eq_idx = pd.DatetimeIndex(['2025-01-03', '2025-01-06'], tz='UTC')  # Fri, Mon
    equity = pd.Series([100.0, 102.0], index=eq_idx)
    cal = pd.date_range('2025-01-01', '2025-01-06', freq='D', tz='UTC')
    btc_px = pd.Series([100, 100, 100.0, 110, 121, 133.1], index=cal,
                       name='BTC')
    out = align_benchmark_returns(equity, btc_px.to_frame())
    # Fri(100)->Mon(133.1) = +33.1% over the same span as the equity move
    assert abs(out['BTC'].iloc[-1] - 0.331) < 1e-9


def test_rolling_beta_tracks_regime_change():
    spy, _ = _bench_returns(seed=10)
    rng = _rng(11)
    loading = pd.Series(np.where(np.arange(N) < N // 2, 0.0, 0.6),
                        index=DATES)
    strat = loading * spy + pd.Series(rng.normal(0, 0.001, N), index=DATES)
    rb = rolling_beta(strat, spy, window=30).dropna()
    assert abs(rb.iloc[100] - 0.0) < 0.15
    assert abs(rb.iloc[-1] - 0.6) < 0.15


def test_beta_report_end_to_end_and_format():
    spy, btc = _bench_returns(seed=12)
    rng = _rng(13)
    strat_ret = 0.3 * spy + 0.1 * btc \
        + pd.Series(rng.normal(0.0002, 0.002, N), index=DATES)
    equity = 100_000.0 * (1 + strat_ret).cumprod()
    # 500 days of SPY prices so the 200d trend split has history
    long_dates = pd.bdate_range(end=DATES[-1], periods=500, tz='UTC')
    spy_px = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 500))),
        index=long_dates)
    btc_px = _prices_from_returns(btc).reindex(long_dates).bfill()
    bench_px = pd.DataFrame({'SPY': spy_px, 'BTC': btc_px})

    rep = beta_report(equity, bench_px, lags=1)
    assert rep['joint']['n_obs'] > 300
    assert 'SPY' in rep['joint']['betas'] and 'BTC' in rep['joint']['betas']
    assert 'up_down' in rep['SPY']
    assert 'trend_conditional' in rep['SPY']
    text = format_report(rep)
    assert 'BETA LEDGER' in text and 'alpha' in text
    # betas in a sane range (prices were rebuilt from returns, so only loose)
    assert -0.2 < rep['joint']['betas']['SPY']['summed'] < 0.9


def test_too_few_observations_raises():
    spy, _ = _bench_returns(seed=14)
    short = spy.iloc[:10]
    try:
        lagged_beta_regression(short, short.to_frame(), lags=1)
        assert False, 'expected ValueError'
    except ValueError:
        pass
