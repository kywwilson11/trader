"""Wave-7 flagship: free offline option-pricing + overlay decision harness.

BS priced against Hull textbook values + put-call parity; the defined-risk
vertical's gap-proof payoff caps; the CORRECTED one-spread-per-leg friction;
and the pre-registered NO-GO that the friction gate returns for short-dated
overnight overlays on wide-spread names."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import options_overlay as ov


class TestBlackScholes:
    def test_hull_textbook_call(self):
        # Hull, Options Futures & Other Derivatives: S=42,K=40,r=0.10,
        # sigma=0.20,T=0.5 -> call 4.76, put 0.81.
        assert ov.bs_price(42, 40, 0.5, 0.10, 0.20, True) == pytest.approx(4.76, abs=0.01)
        assert ov.bs_price(42, 40, 0.5, 0.10, 0.20, False) == pytest.approx(0.81, abs=0.01)

    def test_put_call_parity(self):
        S, K, T, r, sig = 100, 95, 0.75, 0.03, 0.25
        c = ov.bs_price(S, K, T, r, sig, True)
        p = ov.bs_price(S, K, T, r, sig, False)
        # c - p == S - K e^{-rT}
        assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-6)

    def test_intrinsic_at_expiry(self):
        assert ov.bs_price(110, 100, 0.0, 0.03, 0.2, True) == pytest.approx(10.0)
        assert ov.bs_price(90, 100, 0.0, 0.03, 0.2, False) == pytest.approx(10.0)
        assert ov.bs_price(90, 100, 0.0, 0.03, 0.2, True) == pytest.approx(0.0)

    def test_greeks_signs_and_ranges(self):
        g = ov.bs_greeks(100, 100, 0.5, 0.03, 0.25, True)
        assert 0 < g['delta'] < 1 and g['gamma'] > 0 and g['vega'] > 0
        assert g['theta'] < 0  # long option bleeds time
        gp = ov.bs_greeks(100, 100, 0.5, 0.03, 0.25, False)
        assert -1 < gp['delta'] < 0

    def test_call_delta_minus_put_delta_is_one(self):
        c = ov.bs_greeks(100, 105, 0.4, 0.03, 0.3, True)['delta']
        p = ov.bs_greeks(100, 105, 0.4, 0.03, 0.3, False)['delta']
        assert c - p == pytest.approx(1.0, abs=1e-9)


class TestVertical:
    def test_debit_positive_and_bounded(self):
        # bull call spread debit is in (0, K2-K1)
        debit = ov.vertical_debit(100, 100, 105, 0.5, 0.03, 0.25, True)
        assert 0 < debit < 5

    def test_max_loss_is_debit_gap_proof(self):
        # below both strikes the call spread expires worthless -> lose exactly
        # the debit, no matter how far the gap (defined risk).
        K1, K2 = 100, 105
        debit = ov.vertical_debit(100, K1, K2, 0.5, 0.03, 0.25, True)
        for crash_S in (80, 50, 1):
            payoff = ov.vertical_payoff_at_expiry(crash_S, K1, K2, True)
            assert payoff == pytest.approx(0.0)  # max loss capped at debit
        # above both strikes -> full width
        assert ov.vertical_payoff_at_expiry(200, K1, K2, True) == pytest.approx(K2 - K1)

    def test_put_spread_payoff(self):
        K1, K2 = 95, 100  # bear put spread: long K2, short K1
        assert ov.vertical_payoff_at_expiry(80, K1, K2, False) == pytest.approx(5.0)
        assert ov.vertical_payoff_at_expiry(120, K1, K2, False) == pytest.approx(0.0)

    def test_requires_ordered_strikes(self):
        with pytest.raises(ValueError):
            ov.vertical_debit(100, 105, 100, 0.5, 0.03, 0.25, True)


class TestFriction:
    def test_one_spread_per_leg(self):
        # two legs at premium 3.0 and 1.5, 6% spread -> 0.06*(4.5) = 0.27
        assert ov.option_round_trip_cost([3.0, 1.5], 0.06) == pytest.approx(0.27)

    def test_friction_amortizes_over_nights(self):
        legs = [2.0, 1.0]
        debit = 1.0
        total1, per1 = ov.friction_fraction_per_night(debit, legs, 0.14, 1)
        total14, per14 = ov.friction_fraction_per_night(debit, legs, 0.14, 14)
        assert total1 == pytest.approx(total14)          # same total
        assert per14 == pytest.approx(per1 / 14)          # amortized
        assert per1 > 0.4                                 # ~46% @ 1 night

    def test_required_edge_gate(self):
        clears, req = ov.required_edge_clears(0.10, 0.20, 2.0)
        assert not clears and req == pytest.approx(0.40)
        clears2, _ = ov.required_edge_clears(0.50, 0.20, 2.0)
        assert clears2


class TestIVBootstrap:
    def test_band_ordering(self):
        lo, mid, hi = ov.iv_from_har(0.40)
        assert lo < mid < hi
        assert mid == pytest.approx(0.40 * 1.25, abs=1e-9)


class TestOverlayDecision:
    def test_pre_registered_no_go_on_spec_tier(self):
        # flat-ish overnight series on a tier-C (14% spread) name: friction
        # dwarfs any edge -> NO_GO, the pre-registered outcome.
        rng = np.random.RandomState(0)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 60))
        # next open ~ small overnight noise around close
        opens = close * (1 + rng.normal(0, 0.005, 60))
        out = ov.overlay_decision(close, opens, 'C', rv_sigma_annual=0.6)
        assert out['verdict'] == 'NO_GO'
        assert out['friction_frac_of_debit'] > out['mean_edge_frac_of_debit']

    def test_insufficient_data(self):
        out = ov.overlay_decision([100, 101], [100, 102], 'A', 0.4)
        assert out['verdict'] == 'INSUFFICIENT_DATA'

    def test_tier_a_friction_lower_than_tier_c(self):
        rng = np.random.RandomState(1)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 80))
        opens = close * (1 + rng.normal(0, 0.004, 80))
        a = ov.overlay_decision(close, opens, 'A', 0.5)
        c = ov.overlay_decision(close, opens, 'C', 0.5)
        assert a['friction_frac_of_debit'] < c['friction_frac_of_debit']

    def test_output_is_loggable_and_flagged(self):
        rng = np.random.RandomState(2)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 40))
        opens = close * (1 + rng.normal(0, 0.005, 40))
        out = ov.overlay_decision(close, opens, 'B', 0.45)
        assert 'PROXY' in out['note'] and 'INSURANCE' in out['note']
        assert set(out) >= {'verdict', 'friction_frac_of_debit',
                            'mean_edge_frac_of_debit', 'required_edge_frac'}
