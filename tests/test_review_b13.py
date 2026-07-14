"""Review batch b13: bet_sizing.py + blend_fit.py fixes.

Pins the reviewer-approved fixes:
- afml_bet_size clips to [-1, 1] AFTER step rounding (AFML snippet 10.4).
- kelly_edge_odds fails closed on non-finite p and clips p to [0, 1].
- breakeven_p guards degenerate odds (a<=0 or b<=0 -> 1.0, unreachable breakeven).
- kelly fraction halving pinned un-vacuously (cap high enough not to mask it) —
  the in-place repair of tests/test_bet_sizing.py:47 is outside this batch's
  file ownership, so the property is pinned here instead.
- fit_blend_weight sharpe branch: plateau/all-tie selection resolves toward the
  shrink target, not np.argmax's leftmost grid edge.
- fit_blend_weight nnls degenerate branch returns the (clipped) shrink target,
  matching the docstring, and shrink_to is clipped to [0, 1].
- fit_blend_weight raises ValueError on unknown objective (no silent misroute).
"""
import numpy as np
import pytest

from bet_sizing import afml_bet_size, kelly_edge_odds, breakeven_p
from blend_fit import fit_blend_weight


# ---------------------------------------------------------------- bet_sizing

def test_afml_step_discretization_stays_in_unit_interval():
    # Steps that do not divide 1 evenly used to overshoot (0.999/0.15 -> 1.05).
    for step in (0.13, 0.15, 0.3, 0.7):
        assert -1.0 <= afml_bet_size(0.999, step=step) <= 1.0
        assert -1.0 <= afml_bet_size(0.001, step=step) <= 1.0
    # Vectorized path too.
    ps = np.array([0.001, 0.3, 0.5, 0.7, 0.999])
    sizes = afml_bet_size(ps, step=0.15)
    assert sizes.min() >= -1.0 and sizes.max() <= 1.0


def test_afml_step_still_discretizes_for_even_steps():
    # The clip must not disturb the wave-9 plan's step=0.05 (divides 1 evenly).
    s = afml_bet_size(0.73, base_rate=0.5, step=0.05)
    assert abs(s / 0.05 - round(s / 0.05)) < 1e-9
    assert -1.0 <= s <= 1.0


def test_kelly_edge_odds_fails_closed_on_nonfinite_p():
    b, a = 0.04, 0.02
    assert kelly_edge_odds(float('nan'), b, a) == 0.0
    assert kelly_edge_odds(float('inf'), b, a) == 0.0
    assert kelly_edge_odds(float('-inf'), b, a) == 0.0
    # Vectorized: NaN rows size to 0, finite rows unaffected.
    out = kelly_edge_odds(np.array([np.nan, 0.5, np.inf]), b, a, cap=1e9)
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[1] == pytest.approx(0.5 / a - 0.5 / b)


def test_kelly_edge_odds_clips_p_to_unit_interval():
    # Consistent with sibling afml_bet_size: out-of-range p is clipped, so
    # p>1 sizes exactly like p=1 (uncapped to expose the raw fraction).
    b, a = 0.04, 0.02
    assert kelly_edge_odds(1.5, b, a, cap=1e9) == kelly_edge_odds(1.0, b, a, cap=1e9)
    assert kelly_edge_odds(-0.5, b, a) == kelly_edge_odds(0.0, b, a) == 0.0


def test_breakeven_p_degenerate_odds_guard():
    # Mirrors kelly_edge_odds: degenerate odds -> fail-closed. kelly sizes 0,
    # breakeven reports 1.0 (unreachable).
    assert breakeven_p(0.0, 0.0) == 1.0
    assert breakeven_p(0.0, 0.02) == 1.0
    assert breakeven_p(0.04, 0.0) == 1.0
    assert breakeven_p(-0.04, 0.02) == 1.0
    assert breakeven_p(0.04, -0.02) == 1.0
    # Normal 2:1 case unchanged.
    assert breakeven_p(0.04, 0.02) == pytest.approx(1 / 3)
    assert kelly_edge_odds(breakeven_p(0.04, 0.02), 0.04, 0.02) == pytest.approx(0.0, abs=1e-9)


def test_kelly_fraction_halving_pinned_unvacuously():
    # tests/test_bet_sizing.py:47 asserts this at cap=1.0 where both sides clip
    # to 1.0 and the `or half <= full` arm always passes; pin the real property
    # with the cap unmasked.
    a, b = 0.02, 0.04
    full = kelly_edge_odds(0.9, b, a, fraction=1.0, cap=1e9)
    half = kelly_edge_odds(0.9, b, a, fraction=0.5, cap=1e9)
    assert full == pytest.approx(0.9 / a - 0.1 / b)          # 42.5, not 1.0
    assert half == pytest.approx(0.5 * full)


# ----------------------------------------------------------------- blend_fit

def test_sharpe_degenerate_input_returns_shrink_target():
    # Mirrors test_thin_or_degenerate_input_returns_shrink_target for the
    # sharpe branch: identical legs (all 101 grid points tie) and an
    # unreachable threshold (no w admits the 5-trade minimum) both fail-safe
    # to the shrink target, not argmax's w=0.0 -> 0.25.
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    y = rng.normal(size=100)
    assert fit_blend_weight(a, a.copy(), y, objective='sharpe') == pytest.approx(0.5)
    b = rng.normal(size=100)
    assert fit_blend_weight(a, b, y, objective='sharpe', threshold=1e9) == pytest.approx(0.5)
    # Non-default shrink target honored too (0.4 is on the 0.01 grid).
    assert fit_blend_weight(a, a.copy(), y, objective='sharpe',
                            shrink_to=0.4) == pytest.approx(0.4)


def test_sharpe_plateau_tie_breaks_toward_shrink_target():
    # pred = w*a + (1-w)*(a-1) = a - 1 + w; threshold 0 takes rows with
    # a >= 1 - w. 24 always-taken good rows plus one a=0.5 disaster row that
    # enters the take-set only for w >= 0.5 -> the max-sharpe plateau is
    # exactly w in [0.00, 0.49]. Leftmost-argmax returned 0.0; tie-aware
    # selection returns the plateau point nearest shrink_to=0.5, i.e. 0.49.
    a = np.full(25, 5.0)
    a[-1] = 0.5
    y = np.array([1.0, 0.5] * 12 + [-50.0])
    b = a - 1.0
    w = fit_blend_weight(a, b, y, objective='sharpe', threshold=0.0,
                         shrink_lambda=0.0, shrink_to=0.5)
    assert w == pytest.approx(0.49)


def test_nnls_degenerate_returns_shrink_target_exactly():
    # Was (1-lam)*0.5 + lam*shrink_to = 0.45 for shrink_to=0.4; docstring
    # promises shrink_to.
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    y = rng.normal(size=100)
    assert fit_blend_weight(a, a.copy(), y, objective='nnls',
                            shrink_to=0.4) == pytest.approx(0.4)
    assert fit_blend_weight(a, a.copy(), y, objective='nnls',
                            shrink_to=0.5) == pytest.approx(0.5)


def test_shrink_to_is_clipped_to_unit_interval():
    # Bad shrink_to can no longer violate the "w in [0,1]" contract.
    assert fit_blend_weight([1, 2, 3], [3, 2, 1], [0, 1, 0], shrink_to=1.7) == 1.0
    assert fit_blend_weight([1, 2, 3], [3, 2, 1], [0, 1, 0], shrink_to=-0.3) == 0.0
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    y = rng.normal(size=100)
    for obj in ('nnls', 'sharpe'):
        w = fit_blend_weight(a, a.copy(), y, objective=obj, shrink_to=2.0)
        assert 0.0 <= w <= 1.0


def test_unknown_objective_raises():
    rng = np.random.default_rng(6)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    y = rng.normal(size=50)
    with pytest.raises(ValueError, match='nlls'):
        fit_blend_weight(a, b, y, objective='nlls')     # the typo that used to
    with pytest.raises(ValueError):                     # silently run sharpe
        fit_blend_weight(a, b, y, objective='')
