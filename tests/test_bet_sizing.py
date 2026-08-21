"""Wave-9 #5: edge/probability bet-sizing kernels."""
import numpy as np
import pytest

from bet_sizing import afml_bet_size, kelly_edge_odds, concurrency_scale, breakeven_p


def test_afml_bet_size_zero_at_base_rate_and_monotone():
    assert afml_bet_size(0.5, base_rate=0.5) == pytest.approx(0.0, abs=1e-9)
    assert afml_bet_size(0.55, base_rate=0.55) == pytest.approx(0.0, abs=1e-9)
    ps = np.linspace(0.05, 0.95, 50)
    sizes = afml_bet_size(ps, base_rate=0.5)
    assert np.all(np.diff(sizes) > 0)               # strictly increasing in p
    assert sizes.min() >= -1.0 and sizes.max() <= 1.0
    # below the base rate -> negative (a long-only caller would take max(0,.))
    assert afml_bet_size(0.3, base_rate=0.5) < 0 < afml_bet_size(0.7, base_rate=0.5)


def test_afml_base_rate_shifts_neutral_point():
    # With base 0.4, a 0.5 name is now a BUY (above base), unlike base 0.5.
    assert afml_bet_size(0.5, base_rate=0.4) > 0
    assert afml_bet_size(0.5, base_rate=0.6) < 0


def test_afml_step_discretizes():
    s = afml_bet_size(0.73, base_rate=0.5, step=0.05)
    assert abs(s / 0.05 - round(s / 0.05)) < 1e-9


def test_kelly_zero_at_breakeven_for_2to1():
    a = 0.02              # stop move
    b = 0.04              # 2:1 take-profit
    assert breakeven_p(b, a) == pytest.approx(1 / 3)
    assert kelly_edge_odds(1 / 3, b, a) == pytest.approx(0.0, abs=1e-9)
    # above breakeven -> positive and increasing (uncapped so the cap can't mask it;
    # raw Kelly is large leverage for a 2:1 edge, which is exactly why we cap live)
    assert kelly_edge_odds(0.5, b, a, cap=1e9) > 0
    assert kelly_edge_odds(0.6, b, a, cap=1e9) > kelly_edge_odds(0.5, b, a, cap=1e9)
    # below breakeven -> clipped to 0 (long-only, no negative sizing)
    assert kelly_edge_odds(0.2, b, a) == 0.0


def test_kelly_fraction_and_cap():
    a, b = 0.02, 0.04
    # Un-vacuous halving check (cap high enough not to mask the fraction; the
    # old cap=1.0 form clipped both sides to 1.0 and could never fail).
    full = kelly_edge_odds(0.9, b, a, fraction=1.0, cap=1e9)
    half = kelly_edge_odds(0.9, b, a, fraction=0.5, cap=1e9)
    assert full == pytest.approx(0.9 / a - 0.1 / b)          # 42.5, not 1.0
    assert half == pytest.approx(0.5 * full)
    # At cap=1.0 the cap binds BOTH (the flat-top saturation, made explicit).
    assert kelly_edge_odds(0.9, b, a, fraction=1.0, cap=1.0) == 1.0
    assert kelly_edge_odds(0.9, b, a, fraction=0.5, cap=1.0) == 1.0
    # cap binds
    assert kelly_edge_odds(0.99, b, a, fraction=1.0, cap=0.25) == 0.25
    assert kelly_edge_odds(0.7, 0.0, 0.02) == 0.0   # degenerate odds -> 0


def test_concurrency_scale_is_one_over_k():
    assert concurrency_scale(1) == 1.0
    assert concurrency_scale(4) == 0.25
    assert concurrency_scale(0) == 1.0              # guard


def test_vectorized_inputs_supported():
    ps = np.array([0.3, 0.5, 0.7])
    assert afml_bet_size(ps).shape == (3,)
    assert kelly_edge_odds(ps, 0.04, 0.02).shape == (3,)
