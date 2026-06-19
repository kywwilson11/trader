"""Wave-9 #7: BTC trend / TSMOM risk-off gate (graded, debounced, fail-open)."""
import numpy as np
import pytest

from crypto_trend import sma_gap, trend_scalar, hysteresis_state, smooth_state


def test_sma_gap_value_and_insufficient():
    closes = np.concatenate([np.full(199, 100.0), [110.0]])
    # SMA over last 200 = (199*100 + 110)/200 = 100.05; gap = (110-100.05)/100.05
    assert sma_gap(closes, window=200) == pytest.approx((110 - 100.05) / 100.05)
    assert sma_gap(np.full(50, 100.0), window=200) is None      # < window
    assert sma_gap(np.full(200, -1.0), window=200) is None       # non-positive SMA


def test_trend_scalar_graded_and_fail_open():
    assert trend_scalar(0.5) == pytest.approx(1.0, abs=1e-6)      # deep above -> full
    assert trend_scalar(-0.5, floor=0.5) == pytest.approx(0.5, abs=1e-6)  # deep below -> floor
    assert 0.5 < trend_scalar(0.0, floor=0.5) < 1.0              # at SMA -> mid de-risk
    assert trend_scalar(None) == 1.0                             # fail-open, never silent de-risk
    assert trend_scalar(np.nan) == 1.0
    grid = np.linspace(-0.3, 0.3, 50)
    vals = [trend_scalar(g, floor=0.5) for g in grid]
    assert vals == sorted(vals)                                  # monotone increasing
    assert all(0.5 <= v <= 1.0 for v in vals)


def test_hysteresis_asymmetric_schmitt():
    assert hysteresis_state(-0.03, 'risk_on', b_lo=-0.02, b_hi=0.01) == 'risk_off'  # de-risk fast
    assert hysteresis_state(0.02, 'risk_off', b_lo=-0.02, b_hi=0.01) == 'risk_on'   # re-arm slow
    # inside the band -> hold the prior state (no whipsaw)
    assert hysteresis_state(0.0, 'risk_off', b_lo=-0.02, b_hi=0.01) == 'risk_off'
    assert hysteresis_state(0.0, 'risk_on', b_lo=-0.02, b_hi=0.01) == 'risk_on'
    assert hysteresis_state(None, 'risk_off') == 'risk_off'      # fail-open


def test_smooth_state_debounces_single_bar_flip():
    # a lone risk_off in a risk_on stream is ignored
    raw = ['risk_on', 'risk_on', 'risk_off', 'risk_on', 'risk_on']
    assert smooth_state(raw, persistence=3) == ['risk_on'] * 5


def test_smooth_state_commits_after_persistence():
    raw = ['risk_on', 'risk_off', 'risk_off', 'risk_off', 'risk_off']
    out = smooth_state(raw, persistence=3)
    assert out[-1] == 'risk_off'                                 # sustained switch commits
    assert out[0] == 'risk_on'


def test_smooth_state_sawtooth_yields_no_flip():
    raw = ['risk_on', 'risk_off'] * 20                           # rapid alternation
    out = smooth_state(raw, persistence=3)
    assert set(out) == {'risk_on'}                              # never reaches persistence
    assert smooth_state([], persistence=3) == []
