"""Panel campaign (module-improve-v3) Batch A: bet_sizing.py hardening.

Pins the fail-closed input perimeter (cap/fraction/odds/base_rate/step/
n_concurrent), the vectorized per-name odds broadcast, the ndtr bit-identity
with scipy.stats.norm.cdf, and the scalar-Python-float return contracts.
Every input any pre-existing test exercises is byte-identical; only
NaN/inf/None/negative-knob paths changed, to the module's documented
fail-closed values.
"""
import numpy as np
import pytest

from bet_sizing import afml_bet_size, kelly_edge_odds, concurrency_scale, breakeven_p


# ---------------------------------------------------------- kelly_edge_odds

def test_negative_or_nonfinite_cap_fails_closed():
    assert kelly_edge_odds(0.9, 0.04, 0.02, cap=-0.5) == 0.0
    assert kelly_edge_odds(0.9, 0.04, 0.02, cap=float('nan')) == 0.0
    assert kelly_edge_odds(0.9, 0.04, 0.02, cap=float('inf')) == 0.0
    out = kelly_edge_odds(np.array([0.3, 0.9]), 0.04, 0.02, cap=-0.5)
    assert out.shape == (2,) and np.all(out == 0.0)


def test_negative_or_nonfinite_fraction_fails_closed():
    assert kelly_edge_odds(0.9, 0.04, 0.02, fraction=float('nan')) == 0.0
    # A negative fraction must not flip a below-breakeven edge into a max bet.
    assert kelly_edge_odds(0.2, 0.04, 0.02, fraction=-1.0) == 0.0


def test_valid_fraction_and_cap_unchanged():
    a, b = 0.02, 0.04
    assert kelly_edge_odds(0.9, b, a, fraction=0.5, cap=1e9) == pytest.approx(
        0.5 * (0.9 / a - 0.1 / b))
    assert kelly_edge_odds(0.99, b, a, cap=0.25) == 0.25


def test_nonfinite_or_none_odds_fail_closed():
    assert kelly_edge_odds(0.6, float('nan'), 0.02) == 0.0
    assert kelly_edge_odds(0.6, 0.04, float('nan')) == 0.0
    assert kelly_edge_odds(0.6, float('inf'), 0.02) == 0.0
    assert kelly_edge_odds(0.6, None, 0.02) == 0.0


def test_degenerate_odds_preserve_vector_shape():
    out = kelly_edge_odds(np.array([0.3, 0.5, 0.7]), 0.0, 0.02)
    assert out.shape == (3,) and np.all(out == 0.0)
    r = kelly_edge_odds(0.7, 0.0, 0.02)
    assert r == 0.0 and isinstance(r, float)


def test_vectorized_per_name_odds():
    p = np.array([0.5, 0.6])
    b = np.array([0.04, 0.05])
    a = np.array([0.02, 0.02])
    out = kelly_edge_odds(p, b, a, cap=1e9)
    assert out.shape == (2,)
    for i in range(2):
        assert out[i] == pytest.approx(
            kelly_edge_odds(float(p[i]), float(b[i]), float(a[i]), cap=1e9))
    # One degenerate element zeroes only that element.
    out2 = kelly_edge_odds(p, np.array([0.0, 0.05]), a, cap=1e9)
    assert out2[0] == 0.0 and out2[1] > 0


def test_scalar_return_type_contract():
    assert isinstance(kelly_edge_odds(0.7, 0.04, 0.02), float)
    assert isinstance(breakeven_p(0.04, 0.02), float)
    assert isinstance(breakeven_p(0.0, 0.0), float)


# -------------------------------------------------------------- breakeven_p

def test_breakeven_nonfinite_odds_fail_closed():
    assert breakeven_p(float('nan'), 0.02) == 1.0
    assert breakeven_p(0.04, float('nan')) == 1.0
    assert breakeven_p(float('inf'), 0.02) == 1.0
    assert breakeven_p(None, 0.02) == 1.0


def test_breakeven_vectorized_and_argument_order():
    out = breakeven_p(np.array([0.04, 0.0]), np.array([0.02, 0.02]))
    assert out.shape == (2,)
    assert out[0] == pytest.approx(1 / 3) and out[1] == 1.0
    # (b, a) order: the reversed call is the complement — pinned so a future
    # signature change cannot silently reinterpret the order.
    assert breakeven_p(0.04, 0.02) == pytest.approx(1 / 3)
    assert breakeven_p(0.02, 0.04) == pytest.approx(2 / 3)


# ------------------------------------------------------------ afml_bet_size

def test_nonfinite_base_rate_fails_closed():
    for br in (float('nan'), float('inf'), float('-inf'), None):
        assert afml_bet_size(0.7, base_rate=br) == 0.0
        out = afml_bet_size(np.array([0.3, 0.7]), base_rate=br)
        assert out.shape == (2,) and np.all(out == 0.0)


def test_nonfinite_p_sizes_exactly_zero_for_any_base_rate():
    # br=0.0 used to leak +0.0008 through the post-substitution clip.
    assert afml_bet_size(float('nan'), base_rate=0.0) == 0.0
    assert afml_bet_size(float('nan'), base_rate=1.0) == 0.0


def test_out_of_range_base_rate_stays_monotone():
    ps = np.linspace(0.05, 0.95, 50)
    for br in (-0.5, 0.0, 0.4, 0.5, 0.6, 1.0, 1.5, 55.0):
        sizes = afml_bet_size(ps, base_rate=br)
        assert np.all(np.isfinite(sizes))
        assert np.all(np.diff(sizes) >= 0)
        assert sizes[-1] > sizes[0]


def test_in_domain_base_rate_values_unchanged():
    assert afml_bet_size(0.55, base_rate=0.55) == 0.0
    assert afml_bet_size(0.5, base_rate=0.4) > 0 > afml_bet_size(0.5, base_rate=0.6)


def test_step_nonfinite_ignored():
    ref = afml_bet_size(0.7)
    assert afml_bet_size(0.7, step=float('inf')) == ref
    assert afml_bet_size(0.7, step=float('nan')) == ref
    assert afml_bet_size(0.7, step=-0.05) == ref
    s = afml_bet_size(0.73, step=0.05)
    assert abs(s / 0.05 - round(s / 0.05)) < 1e-9


def test_afml_matches_scipy_norm_cdf_exactly():
    # Guards the ndtr swap: scipy.stats.norm._cdf IS scipy.special.ndtr, so the
    # public API must match a norm.cdf reference BIT-exactly, not approximately.
    from scipy.stats import norm
    ps = np.linspace(0.01, 0.99, 199)
    z = (ps - 0.5) / np.sqrt(ps * (1 - ps))
    assert np.array_equal(afml_bet_size(ps), 2.0 * norm.cdf(z) - 1.0)


# -------------------------------------------------------- concurrency_scale

def test_concurrency_scale_fails_closed():
    assert concurrency_scale(float('nan')) == 1.0
    assert concurrency_scale(float('inf')) == 1.0
    assert concurrency_scale(None) == 1.0
    assert concurrency_scale('junk') == 1.0


def test_concurrency_scale_existing_and_truncation_pinned():
    assert concurrency_scale(1) == 1.0
    assert concurrency_scale(4) == 0.25
    assert concurrency_scale(0) == 1.0
    # int() truncation pinned as CURRENT behavior — changing it to 1/float(n)
    # is a sizing-value change and must go through the owner path.
    assert concurrency_scale(2.9) == 0.5
    assert concurrency_scale(3.999) == pytest.approx(1 / 3)
