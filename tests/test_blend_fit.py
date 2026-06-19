"""Wave-9 #2: OOF stacked blend-weight selection (un-hardcode the 0.6/0.4)."""
import numpy as np
import pytest

from blend_fit import fit_blend_weight, _policy_sharpe


def test_recovers_low_weight_when_lgb_leg_is_stronger():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    lgb = y + rng.normal(0, 0.2, 2000)      # strong leg
    lstm = rng.normal(size=2000)            # noise leg
    for obj in ('nnls', 'sharpe'):
        w = fit_blend_weight(lstm, lgb, y, objective=obj, shrink_lambda=0.0)
        assert w < 0.4                       # both objectives down-weight the noise leg
    # default 0.6 is NOT what the data warrants here
    assert abs(fit_blend_weight(lstm, lgb, y, objective='nnls', shrink_lambda=0.0) - 0.6) > 0.2


def test_recovers_high_weight_when_lstm_leg_is_stronger():
    rng = np.random.default_rng(2)
    y = rng.normal(size=2000)
    lstm = y + rng.normal(0, 0.2, 2000)
    lgb = rng.normal(size=2000)
    w = fit_blend_weight(lstm, lgb, y, objective='nnls', shrink_lambda=0.0)
    assert w > 0.6


def test_weight_is_convex_bounded():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    for obj in ('nnls', 'sharpe'):
        w = fit_blend_weight(a, b, y, objective=obj)
        assert 0.0 <= w <= 1.0


def test_shrinkage_pulls_toward_half():
    rng = np.random.default_rng(4)
    y = rng.normal(size=2000)
    lgb = y + rng.normal(0, 0.2, 2000)
    lstm = rng.normal(size=2000)
    raw = fit_blend_weight(lstm, lgb, y, objective='nnls', shrink_lambda=0.0)
    shr = fit_blend_weight(lstm, lgb, y, objective='nnls', shrink_lambda=0.5)
    assert abs(shr - 0.5) < abs(raw - 0.5)   # closer to 0.5 than the raw estimate
    assert shr == pytest.approx(0.5 * raw + 0.25, abs=1e-9)


def test_thin_or_degenerate_input_returns_shrink_target():
    assert fit_blend_weight([1, 2, 3], [3, 2, 1], [0, 1, 0]) == 0.5    # <20 rows
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    # identical legs -> d@d ~ 0 -> NNLS degenerate -> shrink target
    w = fit_blend_weight(a, a.copy(), rng.normal(size=100), objective='nnls', shrink_to=0.5)
    assert w == pytest.approx(0.5, abs=1e-9)


def test_nan_rows_are_dropped():
    rng = np.random.default_rng(6)
    y = rng.normal(size=300)
    lgb = y + rng.normal(0, 0.2, 300)
    lstm = rng.normal(size=300)
    lstm[:50] = np.nan                       # missing OOF rows
    w = fit_blend_weight(lstm, lgb, y, objective='nnls', shrink_lambda=0.0)
    assert 0.0 <= w < 0.4 and np.isfinite(w)


def test_policy_sharpe_helper():
    pred = np.array([0.5, -0.1, 0.3, 0.2, 0.4, -0.2, 0.6])
    y = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0])
    assert _policy_sharpe(pred, y, 0.0) > 0   # the pred>=0 trades net positive
    assert _policy_sharpe(pred, y, 5.0) == 0.0  # nothing clears -> 0
