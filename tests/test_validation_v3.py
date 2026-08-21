"""Panel-review v3 regression tests for validation.py (adjudicated spec).

Pins today's n_eff floor + the new instrumentation echoes; sr_std
sanitization; degenerate-SR fail-closed guard; PBO-family fail-open
coercion; modal fold width; constant-row / single-column-half / split-cap
CSCV guards; Lo-factor max_lag<1 no-op and n_eff_serial alias; docstring
contracts. Runs entirely on the dev Mac (numpy only).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import validation as V


def _series(n=300, seed=1, mu=0.05):
    return np.random.default_rng(seed).normal(mu, 1.0, n)


def test_dsr_result_echoes_inputs():
    d = V.dsr_from_trade_returns(_series(), n_trials=137)
    assert d['n_trials'] == 137
    assert d['n_eff_requested'] is None and d['n_eff_source'] is None
    assert d['n_dropped'] == 0
    assert d['sr_std_null'] == pytest.approx(1.0 / math.sqrt(300))
    # the record is sufficient to recompute the bar
    assert V.expected_max_sharpe(d['n_trials'], d['sr_std_null']) == \
        pytest.approx(d['expected_max_sr'])


def test_dsr_neff_floor_pinned_and_visible():
    r = np.random.default_rng(21).normal(1.0, 1.0, 30)
    a = V.dsr_from_trade_returns(r, n_trials=100, n_eff=4.0)
    b = V.dsr_from_trade_returns(r, n_trials=100, n_eff=10.0)
    assert a['dsr'] == b['dsr'] and a['n_eff'] == 10.0   # today's floor, pinned
    assert a['n_eff_requested'] == 4.0                    # ...but now visible
    assert b['n_eff_requested'] == 10.0


def test_dsr_shape_is_path_invariant():
    r = _series()
    assert set(V.dsr_from_trade_returns(r, 50).keys()) == \
        set(V.dsr_from_trade_returns(r[:5], 50).keys())


def test_dsr_nonpositive_or_nonfinite_sr_std_falls_back():
    r = np.random.RandomState(0).normal(0.03, 1.0, 400)
    base = V.dsr_from_trade_returns(r, n_trials=400)
    for bad in (0.0, -0.5, float('nan'), float('inf')):
        assert V.dsr_from_trade_returns(
            r, n_trials=400, sr_std_across_trials=bad) == base


def test_dsr_degenerate_dispersion_fails_closed():
    r = np.full(50, 0.005) + np.random.default_rng(5).normal(0, 2e-12, 50)
    d = V.dsr_from_trade_returns(r, n_trials=500)
    assert d['dsr'] == 0.0 and d['sr'] == 0.0        # was exactly 1.0
    assert V.dsr_from_trade_returns(_series(seed=6), 500)['dsr'] > 0.0


def test_dsr_n_dropped_counts_nonfinite():
    r = list(_series(100, seed=7))
    d = V.dsr_from_trade_returns(r + [np.nan, np.inf], n_trials=50)
    assert d['n'] == 100 and d['n_dropped'] == 2


def test_dsr_source_echo_and_nan_neff_still_matches_iid():
    r = _series()
    iid = V.dsr_from_trade_returns(r, n_trials=50)
    tagged = V.dsr_from_trade_returns(r, n_trials=50, n_eff=120.0,
                                      n_eff_source='uniqueness')
    assert tagged['n_eff_source'] == 'uniqueness'
    assert iid['n_eff_source'] is None
    # b15 contract: nan n_eff must still equal the IID dict exactly
    assert V.dsr_from_trade_returns(r, n_trials=50, n_eff=float('nan')) == iid


def test_expected_max_sharpe_huge_n_capped():
    assert math.isfinite(V.expected_max_sharpe(10 ** 17, 0.1))  # raised before


def test_dsr_uncoercible_n_trials_uses_echoed_minimum():
    # int(nan) raises — the healthy path must fall back to the clamp floor
    # of 2 (as the degenerate paths already did), never raise mid-gate, and
    # the echoed n_trials must be the value the math actually used.
    r = _series()
    d = V.dsr_from_trade_returns(r, n_trials=float('nan'))
    assert d['n_trials'] == 2
    assert d['expected_max_sr'] == pytest.approx(
        V.expected_max_sharpe(2, d['sr_std_null']))
    assert math.isfinite(d['dsr'])


def test_pbo_oos_blocks_drops_uncoercible_rows():
    good = [list(np.random.default_rng(i).normal(size=8)) for i in range(6)]
    base = V.pbo_from_oos_blocks(good, n_groups=8)
    assert base is not None
    dirty = good + ['n/a', {'a': 1}, [[1.0, 2.0], [3.0]], 3.5]
    assert V.pbo_from_oos_blocks(dirty, n_groups=8) == base  # raised before


def test_build_oos_blocks_fails_open_on_bad_input():
    assert V.build_oos_blocks('n/a', 8) is None              # raised before
    assert V.build_oos_blocks({'a': 1}, 8) is None           # raised before
    assert V.build_oos_blocks(np.ones((5, 20)), 8) is None   # 2-D refused
    b = V.build_oos_blocks(np.arange(80.0), 8)
    assert b.shape == (8,)
    assert b.mean() == pytest.approx(np.arange(80.0).mean())
    # non-divisible: mean of block means intentionally != overall mean
    assert V.build_oos_blocks(np.arange(11.0), 8).mean() != \
        pytest.approx(np.arange(11.0).mean())


def test_pbo_fold_scores_modal_width():
    rows = [list(np.random.default_rng(i).normal(size=3)) for i in range(20)]
    base = V.pbo_from_fold_scores(rows)
    straggler = rows + [list(np.random.default_rng(99).normal(size=2))]
    assert V.pbo_from_fold_scores(straggler) == base   # was 0.5 vs 0.6667
    assert V.pbo_from_fold_scores(rows[:7]) is None    # 8-row minimum


def test_pbo_cscv_constant_row_dropped():
    noise = np.random.RandomState(3).normal(-0.3, 1.0, (10, 160))
    base = V.pbo_cscv(noise, n_groups=8)
    with_zero = np.vstack([np.zeros(160), noise])
    assert V.pbo_cscv(with_zero, n_groups=8) == base   # was 0.043 vs 0.514
    assert base['pbo'] > 0.4
    assert base['n_trials'] == 10


def test_pbo_cscv_single_column_half_returns_none():
    m = np.random.default_rng(4).normal(size=(20, 2))
    assert V.pbo_cscv(m, n_groups=2) is None           # was {'pbo': 1.0}
    m16 = np.random.default_rng(4).normal(size=(20, 16))
    assert V.pbo_cscv(m16, n_groups=2) is not None     # 8 cols per half


def test_pbo_cscv_split_cap():
    m = np.random.default_rng(0).normal(size=(4, 264))
    assert V.pbo_cscv(m, n_groups=22) is None          # C(22,11) > cap
    out = V.pbo_cscv(np.random.default_rng(0).normal(size=(10, 120)),
                     n_groups=6)
    assert out['n_splits'] == math.comb(6, 3)


def test_serial_factor_max_lag_below_one_is_noop():
    e = np.random.RandomState(9).normal(0, 1, 500)
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    out = V.serial_correlation_factor(ar, max_lag=0)
    assert out['factor'] == 1.0 and out['sharpe_scale'] == 1.0
    assert out['max_lag'] == 0
    assert V.serial_correlation_factor(ar, max_lag=-5)['factor'] == 1.0
    assert V.serial_correlation_factor(ar, max_lag=1)['factor'] != 1.0


def test_serial_factor_neff_serial_alias_and_unclamped_scale():
    e = np.random.RandomState(2).normal(0, 1, 600)
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = -0.6 * x[i - 1] + e[i]
    out = V.serial_correlation_factor(x)
    assert out['n_eff_serial'] == out['n_eff'] == out['n']   # clamped to n
    assert out['sharpe_scale'] > 1.0   # documented: raw Lo quantity, unclamped
    short = V.serial_correlation_factor([0.1, 0.2, -0.1])
    assert short['n_eff_serial'] == short['n_eff']


def test_docstring_contracts():
    assert 'EXACTLY ONE' in V.dsr_from_trade_returns.__doc__
    assert 'EXACTLY ONE' in V.deflated_sharpe_ratio.__doc__
    assert 'gotcha #4' in V.dsr_from_trade_returns.__doc__
    assert 'gotcha #4' in V.deflated_sharpe_ratio.__doc__
    assert 'CHRONOLOGICAL ORDER' in V.build_oos_blocks.__doc__
    assert '(matrix, col_indices)' in V.pbo_cscv.__doc__
    assert 'judge it against 0.5' in V.pbo_from_fold_scores.__doc__
    # existing byte-pins must survive every docstring edit
    assert '(1 - k/(q+1))' in V.serial_correlation_factor.__doc__
    assert '(1 - k/n)' not in V.serial_correlation_factor.__doc__
    assert 'DSR >= DSR_MIN' in V.__doc__
