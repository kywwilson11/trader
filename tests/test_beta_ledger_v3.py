"""beta_ledger v3 hardening tests — pure numpy/pandas synthetic data.

Covers: ols_hac return_extras (HAC covariance/rank), summed-beta exact
se/t, up/down and conditional-beta bucket floors, the 200d-SMA warm-up
fix, bounded (non-extrapolating) benchmark alignment, beta_report's
input-normalization/thin-benchmark/degenerate-benchmark/warnings/
coverage/joint_active machinery, _period_for_days, _json_safe, the
load_equity_alpaca None-handling contract (via a fake trading_utils),
inf-benchmark-return masking, the equity-span benchmark fetch sizing,
and the date-only-CSV guard.

No importorskip needed: only stdlib + numpy + pandas + pytest are used,
matching beta_ledger.py's own module-level import discipline.
"""
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import beta_ledger
from beta_ledger import (
    ols_hac, align_benchmark_returns, lagged_beta_regression,
    up_down_betas, conditional_betas, rolling_beta, beta_report,
    format_report, _period_for_days, _json_safe, MIN_OBS,
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


# --- T1: ols_hac return_extras ---

def test_ols_hac_extras():
    rng = _rng(1)
    n = 400
    x = rng.normal(0, 1, n)
    y = 0.5 + 2.0 * x + rng.normal(0, 0.5, n)
    X = np.column_stack([np.ones(n), x])
    beta, se, resid, extras = ols_hac(X, y, return_extras=True)
    assert extras['cov'].shape == (2, 2)
    assert np.allclose(np.sqrt(np.diag(extras['cov'])), se)
    expected_lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    assert extras['hac_lags'] == expected_lags
    assert extras['rank'] == 2
    # default contract unchanged: still a plain 3-tuple
    beta3, se3, resid3 = ols_hac(X, y)
    assert np.allclose(beta3, beta)


# --- T2: summed beta exact HAC se/t ---

def test_summed_se_and_t():
    spy, _ = _bench_returns(seed=20)
    rng = _rng(21)
    strat = 0.2 * spy + 0.2 * spy.shift(1) \
        + pd.Series(rng.normal(0, 0.002, N), index=DATES)
    bench = spy.to_frame()
    out = lagged_beta_regression(strat.dropna(), bench, lags=1)
    b = out['betas']['SPY']
    assert b['summed_t'] > 5
    assert 0 < b['summed_se'] < 0.1

    rng2 = _rng(22)
    noise_strat = pd.Series(rng2.normal(0, 0.01, N), index=DATES)
    out_noise = lagged_beta_regression(noise_strat, bench, lags=1)
    assert abs(out_noise['betas']['SPY']['summed_t']) < 3


# --- T3/T4: up/down bucket floors ---

def test_updown_empty_down_bucket():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    rng = _rng(30)
    x = rng.uniform(0.001, 0.02, 60)  # strictly positive -> empty down bucket
    y = 0.5 * x + rng.normal(0, 0.001, 60)
    xs = pd.Series(x, index=dates, name='x')
    ys = pd.Series(y, index=dates, name='y')
    out = up_down_betas(ys, xs)
    assert math.isnan(out['beta_down'])
    assert math.isnan(out['beta_down_t'])
    assert out['n_down'] == 0
    assert out['n_up'] == 60
    assert math.isfinite(out['beta_up'])


def test_updown_counts():
    spy, _ = _bench_returns(seed=31)
    rng = _rng(32)
    strat = 0.4 * spy + pd.Series(rng.normal(0, 0.002, N), index=DATES)
    out = up_down_betas(strat, spy)
    assert out['n_up'] + out['n_down'] == out['n_obs']
    assert math.isfinite(out['beta_up'])
    assert math.isfinite(out['beta_down'])


# --- T5: conditional_betas shared key-set ---

def test_conditional_shape_stable():
    spy, _ = _bench_returns(seed=33)
    rng = _rng(34)
    strat = 0.3 * spy + pd.Series(rng.normal(0, 0.002, N), index=DATES)
    state = pd.Series(np.arange(N) < 5, index=DATES)  # only 5 "true" days
    out = conditional_betas(strat, spy, state, state_name='regime')
    assert set(out['regime_true'].keys()) == set(out['regime_false'].keys())
    assert out['regime_true']['insufficient'] is True
    assert math.isnan(out['regime_true']['t'])


# --- T6: 200d SMA warm-up must not read as "downtrend" ---

def test_sma_warmup_not_misclassified():
    spy_dates = pd.bdate_range('2024-01-02', periods=289, tz='UTC')
    spy_px = pd.Series(100 * (1.001 ** np.arange(289)), index=spy_dates,
                       name='SPY')
    spy_ret_full = spy_px.pct_change()
    eq_dates = spy_dates[-280:]
    rng = _rng(35)
    strat_ret = (0.5 * spy_ret_full.reindex(eq_dates)
                + pd.Series(rng.normal(0, 0.001, 280), index=eq_dates)).dropna()
    equity = 100.0 * (1 + strat_ret).cumprod()
    rep = beta_report(equity, spy_px.to_frame(), lags=1)
    tc = rep['SPY']['trend_conditional']
    assert tc['above_200d_false']['n_obs'] == 0
    assert math.isnan(tc['above_200d_false']['beta'])
    assert tc['n_state_undefined'] >= 180


# --- T7: alignment never extrapolates past benchmark coverage ---

def test_align_no_trailing_extrapolation():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    rng = _rng(36)
    equity = 100.0 * (1 + pd.Series(rng.normal(0.0003, 0.005, 60),
                                    index=dates)).cumprod()
    spy_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, 40))),
                       index=dates[:40], name='SPY')
    out = align_benchmark_returns(equity, spy_px.to_frame())
    assert out['SPY'].tail(20).isna().all()

    rep = beta_report(equity, spy_px.to_frame(), lags=1)
    assert rep['SPY']['stale_days'] == 20


# --- T8: a single non-positive equity point is dropped, not fatal ---

def test_nonfinite_equity_survives():
    dates = pd.bdate_range('2025-01-02', periods=80, tz='UTC')
    rng = _rng(37)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 80), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 80),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    eq.iloc[10] = 0.0
    spy_px = _prices_from_returns(spy_ret)
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert math.isfinite(rep['joint']['alpha_annual'])
    assert math.isfinite(rep['joint']['betas']['SPY']['summed'])
    assert rep['data_quality']['n_nonpositive_equity_dropped'] == 1


# --- T9: a flat book reports r2/sharpe as nan, not a confident 0.0 ---

def test_flat_book_r2_is_nan_not_zero():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    eq = pd.Series(100.0, index=dates)
    rng = _rng(39)
    spy_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 60))),
                       index=dates, name='SPY')
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert math.isnan(rep['joint']['r2'])
    assert math.isnan(rep['strategy']['sharpe'])


# --- T10: sort/dedupe invariance ---

def test_sort_and_dedupe_invariance():
    dates = pd.bdate_range('2025-01-02', periods=80, tz='UTC')
    rng = _rng(40)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 80), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 80),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    bench = _prices_from_returns(spy_ret).to_frame()

    rep_fwd = beta_report(eq, bench, lags=1)
    rep_rev = beta_report(eq.iloc[::-1], bench, lags=1)
    assert abs(rep_fwd['joint']['alpha_annual']
              - rep_rev['joint']['alpha_annual']) < 1e-12
    assert rep_fwd['period']['start'] == rep_rev['period']['start']
    assert rep_fwd['period']['end'] == rep_rev['period']['end']

    dup_idx = eq.index.insert(10, eq.index[10])
    dup_vals = np.insert(eq.values, 10, eq.values[10])
    eq_dup = pd.Series(dup_vals, index=dup_idx)
    rep_dup = beta_report(eq_dup, bench, lags=1)
    assert rep_dup['period']['n_days'] == len(eq_dup) - 2


# --- T11-T14: guard rails ---

def test_reserved_column_raises():
    dates = pd.bdate_range('2025-01-02', periods=40, tz='UTC')
    eq = pd.Series(np.linspace(100, 110, 40), index=dates)
    bad_bench = pd.DataFrame({'joint': np.linspace(100, 105, 40)}, index=dates)
    with pytest.raises(ValueError, match='collide'):
        beta_report(eq, bad_bench, lags=1)


def test_lags_negative_raises():
    spy, _ = _bench_returns(seed=41)
    with pytest.raises(ValueError):
        lagged_beta_regression(spy, spy.to_frame(), lags=-1)


def test_small_equity_raises_valueerror():
    dates1 = pd.bdate_range('2025-01-02', periods=1, tz='UTC')
    eq1 = pd.Series([100.0], index=dates1)
    bench1 = pd.DataFrame({'SPY': [100.0]}, index=dates1)
    with pytest.raises(ValueError, match='equity'):
        beta_report(eq1, bench1, lags=1)

    dates10 = pd.bdate_range('2025-01-02', periods=10, tz='UTC')
    eq10 = pd.Series(np.linspace(100, 105, 10), index=dates10)
    bench10 = pd.DataFrame({'SPY': np.linspace(100, 102, 10)}, index=dates10)
    with pytest.raises(ValueError, match='equity'):
        beta_report(eq10, bench10, lags=1)


def test_min_obs_post_shift():
    dates_min = pd.bdate_range('2025-01-02', periods=MIN_OBS, tz='UTC')
    rng = _rng(42)
    x_min = pd.Series(rng.normal(0, 0.01, MIN_OBS), index=dates_min, name='SPY')
    y_min = 0.5 * x_min + pd.Series(rng.normal(0, 0.001, MIN_OBS), index=dates_min)
    with pytest.raises(ValueError, match='usable'):
        lagged_beta_regression(y_min, x_min.to_frame(), lags=1)

    dates_ok = pd.bdate_range('2025-01-02', periods=MIN_OBS + 5, tz='UTC')
    rng2 = _rng(43)
    x_ok = pd.Series(rng2.normal(0, 0.01, MIN_OBS + 5), index=dates_ok, name='SPY')
    y_ok = 0.5 * x_ok + pd.Series(rng2.normal(0, 0.001, MIN_OBS + 5), index=dates_ok)
    out = lagged_beta_regression(y_ok, x_ok.to_frame(), lags=1)
    assert out['n_obs'] == MIN_OBS + 4


# --- T15: underpowered flag ---

def test_underpowered_flag():
    dates = pd.bdate_range('2025-01-02', periods=26, tz='UTC')
    rng = _rng(44)
    spy = pd.Series(rng.normal(0, 0.01, 26), index=dates, name='SPY')
    btc = pd.Series(rng.normal(0, 0.02, 26), index=dates, name='BTC')
    y = 0.3 * spy + 0.1 * btc + pd.Series(rng.normal(0, 0.002, 26), index=dates)
    bench = pd.concat([spy, btc], axis=1)
    out = lagged_beta_regression(y, bench, lags=3)
    assert out['underpowered'] is True
    assert out['obs_per_param'] < 3


# --- T16/T17: thin vs degenerate benchmark handling ---

def test_excluded_thin_benchmark():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    rng = _rng(45)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 60), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 60),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = _prices_from_returns(spy_ret)
    bench = pd.DataFrame({'SPY': spy_px, 'BTC': np.nan}, index=dates)
    rep = beta_report(eq, bench, lags=1)
    assert 'SPY' in rep
    assert rep['excluded_benchmarks'] == {'BTC': 0}
    assert 'BTC' not in rep['joint']['betas']


def test_degenerate_benchmark_flagged():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    rng = _rng(46)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 60), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 60),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = _prices_from_returns(spy_ret)
    btc_px = pd.Series(50.0, index=dates, name='BTC')  # constant -> zero variance
    bench = pd.DataFrame({'SPY': spy_px, 'BTC': btc_px})
    rep = beta_report(eq, bench, lags=1)
    assert rep['BTC'].get('degenerate') is True
    assert rep['joint']['rank_deficient'] is True
    assert any('BTC' in w for w in rep['warnings'])


# --- T18: outlier-day warning ---

def test_outlier_warning():
    dates = pd.bdate_range('2025-01-02', periods=100, tz='UTC')
    rng = _rng(47)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 100), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 100),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    eq.iloc[50:] = eq.iloc[50:] * 1.5  # one-time +50% level jump (deposit-like)
    spy_px = _prices_from_returns(spy_ret)
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert rep['strategy']['n_outlier_days'] == 1
    assert any(('deposit' in w or 'reset' in w or '15%' in w)
              for w in rep['warnings'])


# --- T19: coverage fields ---

def test_coverage_fields():
    dates = pd.bdate_range('2025-01-02', periods=90, tz='UTC')
    rng = _rng(48)
    strat_ret = pd.Series(rng.normal(0.0003, 0.005, 90), index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 40))),
                       index=dates[-40:], name='SPY')
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert rep['period']['n_obs_used'] == rep['joint']['n_obs']
    assert rep['period']['coverage_pct'] < 0.6


# --- T20: joint_active vs joint, flat-day share ---

def test_joint_active_and_flat_share():
    dates = pd.bdate_range('2025-01-02', periods=300, tz='UTC')
    rng = _rng(49)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 300), index=dates, name='SPY')
    alternating = (np.arange(300) % 2 == 0)
    strat_ret = pd.Series(np.where(alternating, 0.0, 0.9 * spy_ret.values),
                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = _prices_from_returns(spy_ret)
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert abs(rep['strategy']['flat_day_share'] - 0.5) < 0.05
    assert 'joint_active' in rep
    assert rep['joint_active']['r2'] > rep['joint']['r2']


# --- T21: rolling_beta summary dict ---

def test_rolling_beta_summary():
    spy, _ = _bench_returns(seed=50)
    rng = _rng(51)
    loading = pd.Series(np.where(np.arange(N) < N // 2, 0.0, 0.6), index=DATES)
    strat_ret = loading * spy + pd.Series(rng.normal(0, 0.001, N), index=DATES)
    eq = 100.0 * (1 + strat_ret).cumprod()
    long_dates = pd.bdate_range(end=DATES[-1], periods=500, tz='UTC')
    spy_px = _prices_from_returns(spy).reindex(long_dates).bfill()
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    rb = rep['SPY']['rolling_beta']
    assert rb['min'] < 0.2
    assert rb['max'] > 0.45
    assert rb['last'] == rep['SPY']['rolling_beta_last']
    pd.Timestamp(rb['asof'])  # must parse as a date


# --- T22: JSON-safety of a NaN-laden report ---

def test_json_safe_valid():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    eq = pd.Series(100.0, index=dates)
    rng = _rng(52)
    spy_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 60))),
                       index=dates, name='SPY')
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    txt = json.dumps(_json_safe(rep), allow_nan=False, default=str)
    assert 'NaN' not in txt


# --- T23: Alpaca period-string sizing ---

def test_period_for_days():
    assert _period_for_days(30) == '3M'
    assert _period_for_days(90) == '6M'
    assert _period_for_days(200) == '11M'
    assert _period_for_days(365) == '2A'
    for days in (30, 90, 200):
        p = _period_for_days(days)
        assert p.endswith('M')
        assert int(p[:-1]) * 21 >= days


# --- T24: load_equity_alpaca pairwise None-drop (fake trading_utils) ---

def test_load_equity_alpaca_none_handling():
    prior = sys.modules.get('trading_utils')
    fake_hist = types.SimpleNamespace(
        timestamp=[1700000000, 1700086400, 1700172800, 1700259200],
        equity=[None, None, 100.0, '101.0'],
    )
    fake_api = types.SimpleNamespace(
        get_portfolio_history=lambda **kw: fake_hist)
    sys.modules['trading_utils'] = types.SimpleNamespace(get_api=lambda: fake_api)
    try:
        eq = beta_ledger.load_equity_alpaca(90)
    finally:
        if prior is not None:
            sys.modules['trading_utils'] = prior
        else:
            sys.modules.pop('trading_utils', None)
    assert list(eq.values) == [100.0, 101.0]
    assert len(eq) == 2


# --- T25: annualization diagnostics ---

def test_annualization_diagnostics():
    cal_dates = pd.date_range('2025-01-01', periods=100, freq='D', tz='UTC')
    rng = _rng(53)
    strat_ret = pd.Series(rng.normal(0.0003, 0.005, 100), index=cal_dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 100))),
                       index=cal_dates, name='SPY')
    rep = beta_report(eq, spy_px.to_frame(), lags=1)
    assert rep['period']['obs_per_year'] > 330
    assert rep['period']['annualization_days'] == 252

    bdates = pd.bdate_range('2025-01-02', periods=100, tz='UTC')
    rng2 = _rng(54)
    strat_ret_b = pd.Series(rng2.normal(0.0003, 0.005, 100), index=bdates)
    eq_b = 100.0 * (1 + strat_ret_b).cumprod()
    spy_px_b = pd.Series(100 * np.exp(np.cumsum(rng2.normal(0.0003, 0.01, 100))),
                        index=bdates, name='SPY')
    rep_b = beta_report(eq_b, spy_px_b.to_frame(), lags=1)
    assert 240 < rep_b['period']['obs_per_year'] < 280


# --- T27: an inf benchmark return (zero price) is masked, not propagated ---

def test_inf_bench_return_masked():
    dates = pd.bdate_range('2025-01-02', periods=60, tz='UTC')
    rng = _rng(60)
    spy_ret = pd.Series(rng.normal(0.0004, 0.01, 60), index=dates, name='SPY')
    strat_ret = 0.3 * spy_ret + pd.Series(rng.normal(0.0002, 0.002, 60),
                                          index=dates)
    eq = 100.0 * (1 + strat_ret).cumprod()
    spy_px = _prices_from_returns(spy_ret)
    btc_px = _prices_from_returns(
        pd.Series(rng.normal(0.001, 0.02, 60), index=dates, name='BTC'))
    btc_px.iloc[30] = 0.0  # zero price -> next-day pct_change = +inf
    bench = pd.DataFrame({'SPY': spy_px, 'BTC': btc_px})
    rep = beta_report(eq, bench, lags=1)
    assert rep['data_quality']['n_nonfinite_bench_returns_masked'] == 1
    assert math.isfinite(rep['joint']['betas']['BTC']['summed'])
    assert math.isfinite(rep['BTC']['up_down']['beta_up'])
    assert any('non-finite benchmark' in w for w in rep['warnings'])


# --- T28: yf benchmark fetch is sized from the ACTUAL equity span ---

def test_bench_days_for_equity():
    dates = pd.bdate_range('2024-01-02', periods=500, tz='UTC')
    eq = pd.Series(np.linspace(100.0, 120.0, 500), index=dates)
    span = int((dates[-1] - dates[0]).days) + 1
    assert span > 500  # bdate grid spans more calendar days than rows
    assert beta_ledger._bench_days_for_equity(90, eq) == span
    assert beta_ledger._bench_days_for_equity(10_000, eq) == 10_000
    assert beta_ledger._bench_days_for_equity(90, eq.iloc[:1]) == 90


# --- T29: a date-only CSV exits cleanly instead of IndexError ---

def test_read_csv_no_value_columns(tmp_path):
    p = tmp_path / 'dates_only.csv'
    p.write_text('date\n2025-01-02\n2025-01-03\n')
    with pytest.raises(SystemExit, match='no value columns'):
        beta_ledger._read_csv_series(str(p))


# --- T26: trend split now applies to every benchmark column, not just SPY ---

def test_trend_split_for_btc():
    spy, btc = _bench_returns(seed=55)
    rng = _rng(56)
    strat_ret = 0.3 * spy + 0.1 * btc \
        + pd.Series(rng.normal(0.0002, 0.002, N), index=DATES)
    eq = 100_000.0 * (1 + strat_ret).cumprod()
    long_dates = pd.bdate_range(end=DATES[-1], periods=500, tz='UTC')
    spy_px = _prices_from_returns(spy).reindex(long_dates).bfill()
    btc_px = _prices_from_returns(btc).reindex(long_dates).bfill()
    bench_px = pd.DataFrame({'SPY': spy_px, 'BTC': btc_px})
    rep = beta_report(eq, bench_px, lags=1)
    assert 'trend_conditional' in rep['BTC']
