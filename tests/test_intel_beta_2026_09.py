"""INTEL W16 (2026-09) — beta_ledger robust leg: Welch (2022) slope-winsorized
beta per benchmark (equal-weight + 120-obs half-life WLS), the pre-registered
`beta_stable` flag (Scout B E4: |beta_OLS - beta_winsor| < 0.15 at n >= 60),
and `alpha_mintrl_years` = (t*/SR)^2 with t* = 2.

Report-only and ADDITIVE: every pre-existing report key/value and every
pre-existing format_report line must be unchanged. Pure numpy/pandas,
synthetic data — runs on the dev Mac, CI and the Jetson.
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import beta_ledger  # noqa: E402

NEW_BENCH_KEYS = {'beta_ols_univariate', 'beta_winsor', 'beta_winsor_wls',
                  'beta_stable', 'winsor', 'robust_beta_error'}
NEW_STRAT_KEYS = {'alpha_mintrl_years', 'alpha_mintrl_obs',
                  'alpha_mintrl_t_star', 'alpha_mintrl_window_years',
                  'alpha_estimable'}

# sha256 of the PRE-EDIT beta_ledger output on _golden_fixture() (floats
# rounded to 9 significant digits so BLAS/platform last-bit noise cannot
# flip it). Captured 2026-09-27 from the pre-edit module (Jetson); the
# post-edit module produced identical digests.
GOLDEN_REPORT_SHA = '3fbef7125baec37afc0ee0040bb1698f5d6117ae7e1b29b69666d9cd0a493ec7'
GOLDEN_TEXT_SHA = 'f15af2b1b038681de0541a0b6130ece14d201f11b4f3da7742d605c8e5893ba6'


def _req(a, b):
    """Recursive equality with nan == nan."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_req(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_req(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def _strip_new(rep):
    rep = copy.deepcopy(rep)
    for k in NEW_STRAT_KEYS:
        rep.get('strategy', {}).pop(k, None)
    for name in list(rep.get('joint', {}).get('betas', {})):
        d = rep.get(name)
        if isinstance(d, dict):
            for k in NEW_BENCH_KEYS:
                d.pop(k, None)
    return rep


def _golden_fixture(seed=7, n=121):
    """(equity, bench_prices, clean_ret) on a calendar-daily grid."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=n, freq='D', tz='UTC')
    spy_r = rng.normal(0.0003, 0.010, n)
    btc_r = rng.normal(0.0008, 0.030, n)
    strat_r = 0.0002 + 0.4 * spy_r + 0.2 * btc_r + rng.normal(0, 0.004, n)
    strat_r[0] = spy_r[0] = btc_r[0] = 0.0
    eq = pd.Series(100000.0 * np.cumprod(1 + strat_r), index=idx, name='equity')
    bench = pd.DataFrame({'SPY': 400.0 * np.cumprod(1 + spy_r),
                          'BTC': 60000.0 * np.cumprod(1 + btc_r)}, index=idx)
    clean = eq.pct_change(fill_method=None).rename('clean_ret')
    return eq, bench, clean


def _round9(obj):
    if isinstance(obj, dict):
        return {k: _round9(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round9(v) for v in obj]
    if isinstance(obj, float):
        return float(f'{obj:.9g}')
    return obj


def _golden_digests(mod=beta_ledger):
    """(report_sha, text_sha) of the pre-existing keys / lines. Called on the
    PRE-EDIT module once to fill the GOLDEN_* constants."""
    eq, bench, clean = _golden_fixture()
    rep = _strip_new(mod.beta_report(eq, bench, lags=1, clean_ret=clean))
    blob = json.dumps(_round9(mod._json_safe(rep)), sort_keys=True,
                      allow_nan=False, default=str)
    rep_full = mod.beta_report(eq, bench, lags=1, clean_ret=clean)
    old_lines = mod.format_report(rep_full).split('\n')
    n_new = sum(1 for ln in old_lines
                if 'robust beta (Welch' in ln or 'beta stability' in ln
                or 'alpha MinTRL' in ln or 'alpha not estimable' in ln)
    text = '\n'.join(old_lines[:len(old_lines) - n_new])
    return (hashlib.sha256(blob.encode()).hexdigest(),
            hashlib.sha256(text.encode()).hexdigest())


def _clean_and_bad(seed, n=120, k=60, x_k=0.015, bad=0.40, beta=1.0):
    """Clean Gaussian market-model returns, and a copy with ONE bad print."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-02', periods=n, freq='D', tz='UTC')
    x = rng.normal(0.0003, 0.010, n)
    x[k] = x_k
    y = 0.0002 + beta * x + rng.normal(0.0, 0.002, n)
    yb = y.copy()
    yb[k] = bad
    return (pd.Series(y, index=idx), pd.Series(yb, index=idx),
            pd.Series(x, index=idx))


# --------------------------------------------------------------------------
# 1. additive-only: pre-existing keys and lines are untouched
# --------------------------------------------------------------------------

@pytest.mark.parametrize('with_clean', [False, True])
def test_robust_leg_is_purely_additive(monkeypatch, with_clean):
    eq, bench, clean = _golden_fixture()
    kw = {'clean_ret': clean} if with_clean else {}
    rep_on = beta_ledger.beta_report(eq, bench, lags=1, **kw)
    text_on = beta_ledger.format_report(rep_on)
    monkeypatch.setattr(beta_ledger, '_add_robust_leg', lambda *a, **k: None)
    rep_off = beta_ledger.beta_report(eq, bench, lags=1, **kw)
    text_off = beta_ledger.format_report(rep_off)
    # every new key present, and stripping them gives the old report exactly
    for name in ('SPY', 'BTC'):
        assert NEW_BENCH_KEYS - {'robust_beta_error'} <= set(rep_on[name])
        assert not (NEW_BENCH_KEYS & set(rep_off[name]))
    assert NEW_STRAT_KEYS <= set(rep_on['strategy'])
    assert _req(_strip_new(rep_on), rep_off)
    # insertion order of every pre-existing key is unchanged
    assert list(rep_on) == list(rep_off)
    for name in ('SPY', 'BTC'):
        assert [k for k in rep_on[name] if k not in NEW_BENCH_KEYS] \
            == list(rep_off[name])
    assert [k for k in rep_on['strategy'] if k not in NEW_STRAT_KEYS] \
        == list(rep_off['strategy'])
    # the old printout is an exact line prefix of the new one
    on_lines, off_lines = text_on.split('\n'), text_off.split('\n')
    assert on_lines[:len(off_lines)] == off_lines
    assert len(on_lines) == len(off_lines) + 4   # 2 benchmarks + verdict + MinTRL
    json.dumps(beta_ledger._json_safe(rep_on), allow_nan=False, default=str)


def test_golden_preexisting_keys_match_pre_edit_module():
    rep_sha, text_sha = _golden_digests()
    assert rep_sha == GOLDEN_REPORT_SHA
    assert text_sha == GOLDEN_TEXT_SHA


# --------------------------------------------------------------------------
# 2. the Welch rule, weights and slope kernel
# --------------------------------------------------------------------------

def test_welch_winsorize_band_is_minus2_plus4_times_market():
    assert beta_ledger.WELCH_DELTA == 3.0
    x = np.array([0.01, 0.01, 0.01, -0.01, -0.01, 0.0, 0.01])
    y = np.array([0.50, -0.50, 0.02, 0.50, -0.50, 0.03, -0.019])
    out = beta_ledger.welch_winsorize(y, x)
    np.testing.assert_allclose(out, [0.04, -0.02, 0.02, 0.02, -0.04, 0.0,
                                     -0.019], atol=1e-15)
    # delta = 0 collapses every return onto the market (beta 1 by force)
    np.testing.assert_allclose(beta_ledger.welch_winsorize(y, x, 0.0), x)
    with pytest.raises(ValueError):
        beta_ledger.welch_winsorize(y, x, -1.0)


def test_half_life_weights_normalised_and_halving():
    w = beta_ledger.half_life_weights(300, 120.0)
    assert w.shape == (300,)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(w) > 0)                      # newest weighs most
    assert w[-1] / w[-1 - 120] == pytest.approx(2.0, rel=1e-12)
    assert w[-1] / w[-1 - 240] == pytest.approx(4.0, rel=1e-12)
    u = beta_ledger.half_life_weights(5, None)
    np.testing.assert_allclose(u, np.full(5, 0.2))
    assert beta_ledger.half_life_weights(0, 120.0).size == 0
    with pytest.raises(ValueError):
        beta_ledger.half_life_weights(10, 0.0)


def test_wls_slope_matches_polyfit():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 200)
    y = 0.001 + 0.7 * x + rng.normal(0, 0.003, 200)
    assert beta_ledger.wls_slope(y, x) == pytest.approx(
        np.polyfit(x, y, 1)[0], rel=1e-9)
    w = beta_ledger.half_life_weights(200, 50.0)
    # np.polyfit's w multiplies the residual, i.e. it is sqrt(WLS weight)
    assert beta_ledger.wls_slope(y, x, w) == pytest.approx(
        np.polyfit(x, y, 1, w=np.sqrt(w))[0], rel=1e-9)
    assert math.isnan(beta_ledger.wls_slope(y, np.zeros(200)))
    assert math.isnan(beta_ledger.wls_slope(y[:2], x[:2]))


# --------------------------------------------------------------------------
# 3. behaviour on clean data and under one bad print
# --------------------------------------------------------------------------

@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_winsor_equals_ols_on_clean_gaussian(seed):
    y, _, x = _clean_and_bad(seed)
    rb = beta_ledger.robust_betas(y, x)
    assert rb['beta_ols_univariate'] == pytest.approx(
        np.polyfit(x.values, y.values, 1)[0], rel=1e-9)
    assert abs(rb['beta_winsor'] - rb['beta_ols_univariate']) < 0.05
    assert abs(rb['beta_winsor_wls'] - rb['beta_ols_univariate']) < 0.10
    assert rb['beta_stable'] is True
    assert rb['winsor']['n_obs'] == 120
    assert rb['winsor']['band_mult'] == [-2.0, 4.0]


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_single_40pct_print_moves_winsor_far_less_and_flips_stable(seed):
    y, yb, x = _clean_and_bad(seed)
    clean = beta_ledger.robust_betas(y, x)
    bad = beta_ledger.robust_betas(yb, x)
    d_ols = bad['beta_ols_univariate'] - clean['beta_ols_univariate']
    d_w = bad['beta_winsor'] - clean['beta_winsor']
    assert abs(d_ols) > 0.15                 # the print really does bite OLS
    assert abs(d_w) < 0.25 * abs(d_ols)
    assert clean['beta_stable'] is True
    assert bad['beta_stable'] is False
    assert bad['winsor']['abs_diff_ols_winsor'] >= beta_ledger.BETA_STABLE_TOL


def test_beta_stable_requires_min_obs():
    y, _, x = _clean_and_bad(0)
    rb = beta_ledger.robust_betas(y.iloc[:59], x.iloc[:59])
    assert rb['winsor']['abs_diff_ols_winsor'] < 0.15
    assert rb['beta_stable'] is False            # n = 59 < 60
    assert beta_ledger.robust_betas(y.iloc[:60], x.iloc[:60])['beta_stable']


def test_beta_report_end_to_end_bad_print_flips_spy_verdict():
    y, yb, x = _clean_and_bad(11)
    idx = x.index.insert(0, x.index[0] - pd.Timedelta(days=1))
    spy_px = pd.Series(400.0 * np.cumprod(np.r_[1.0, 1 + x.values]), index=idx)
    bench = pd.DataFrame({'SPY': spy_px})
    reps = {}
    for label, r in (('clean', y), ('bad', yb)):
        eq = pd.Series(1e5 * np.cumprod(np.r_[1.0, 1 + r.values]), index=idx)
        reps[label] = beta_ledger.beta_report(eq, bench, lags=1)
    c, b = reps['clean']['SPY'], reps['bad']['SPY']
    assert c['beta_stable'] is True and b['beta_stable'] is False
    d_ols = b['beta_ols_univariate'] - c['beta_ols_univariate']
    assert abs(b['beta_winsor'] - c['beta_winsor']) < 0.25 * abs(d_ols)
    text = beta_ledger.format_report(reps['bad'])
    assert 'SPY robust beta (Welch 2022 slope-winsorized -2/+4x)' in text
    assert 'beta stability (|OLS - winsor| < 0.15 at n >= 60): SPY UNSTABLE' in text
    assert 'beta stability' in beta_ledger.format_report(reps['clean'])
    assert 'SPY STABLE' in beta_ledger.format_report(reps['clean'])


# --------------------------------------------------------------------------
# 4. alpha MinTRL
# --------------------------------------------------------------------------

def test_alpha_mintrl_formula():
    assert beta_ledger.MINTRL_T_STAR == 2.0
    assert beta_ledger.alpha_mintrl_years(0.97) == pytest.approx(
        (2.0 / 0.97) ** 2, rel=1e-12)                 # Scout B F8: ~4.3 y
    assert beta_ledger.alpha_mintrl_years(2.0) == pytest.approx(1.0)
    assert beta_ledger.alpha_mintrl_years(-1.0) == pytest.approx(4.0)
    assert beta_ledger.alpha_mintrl_years(1.0, t_star=3.0) == pytest.approx(9.0)
    assert beta_ledger.alpha_mintrl_years(0.0) == float('inf')
    assert beta_ledger.alpha_mintrl_years(float('nan')) == float('inf')


def test_mintrl_keys_consistent_with_report_sharpe():
    eq, bench, _ = _golden_fixture()
    rep = beta_ledger.beta_report(eq, bench, lags=1)
    s = rep['strategy']
    assert s['alpha_mintrl_years'] == pytest.approx(
        (2.0 / abs(s['sharpe'])) ** 2, rel=1e-12)
    assert s['alpha_mintrl_obs'] == pytest.approx(
        s['alpha_mintrl_years'] * beta_ledger.ANNUALIZATION_DAYS, rel=1e-12)
    assert s['alpha_mintrl_window_years'] == pytest.approx(
        rep['period']['n_days'] / beta_ledger.ANNUALIZATION_DAYS, rel=1e-12)
    assert s['alpha_estimable'] is (s['alpha_mintrl_years']
                                    <= s['alpha_mintrl_window_years'])


def test_mintrl_message_branches():
    eq, bench, _ = _golden_fixture()
    rep = beta_ledger.beta_report(eq, bench, lags=1)
    s = rep['strategy']
    s.update(alpha_mintrl_years=4.25, alpha_mintrl_window_years=90 / 252,
             alpha_estimable=False)
    text = beta_ledger.format_report(rep)
    assert ('alpha not estimable in this horizon (MinTRL 4.2 y > window 0.4 y)'
            in text or
            'alpha not estimable in this horizon (MinTRL 4.3 y > window 0.4 y)'
            in text)
    assert 'alpha MinTRL' not in text
    s.update(alpha_mintrl_years=0.1, alpha_mintrl_window_years=0.5,
             alpha_estimable=True)
    text = beta_ledger.format_report(rep)
    assert 'alpha MinTRL (t*=2): 0.1 y <= window 0.5 y' in text
    assert 'not estimable' not in text
    s.update(alpha_mintrl_years=float('inf'), alpha_estimable=False)
    assert '(MinTRL inf y > window 0.5 y)' in beta_ledger.format_report(rep)
    # zero-Sharpe book: inf survives --json as null
    js = beta_ledger._json_safe(rep)
    assert js['strategy']['alpha_mintrl_years'] is None
    json.dumps(js, allow_nan=False, default=str)
