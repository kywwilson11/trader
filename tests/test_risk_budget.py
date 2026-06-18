"""Wave-6 Tier-2: cross-book account risk cap + two-book equity simulator.

Verifies the account stop-risk nets the two books with a cross-book
correlation (so it sits between max-of and sum-of the books), that the budget
solver brings the account exactly to the cap, that the cap only ever shrinks a
candidate, and that the concurrent-equity simulator captures cross-book
drawdown concentration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import risk_budget as rb
from portfolio import diversified_book_risk


class TestAccountStopRisk:
    def test_between_max_and_sum(self):
        s = [0.005, 0.005, 0.005]   # ~diversified ~0.9-1.5%
        c = [0.005, 0.005]
        r_s = diversified_book_risk(s, 0.3)
        r_c = diversified_book_risk(c, 0.3)
        acct = rb.account_stop_risk(s, c, 0.3, 0.3, 0.5)
        assert max(r_s, r_c) <= acct <= r_s + r_c + 1e-12

    def test_rho_cross_monotone(self):
        s, c = [0.01, 0.01], [0.01, 0.01]
        lo = rb.account_stop_risk(s, c, 0.2, 0.2, -0.5)
        mid = rb.account_stop_risk(s, c, 0.2, 0.2, 0.0)
        hi = rb.account_stop_risk(s, c, 0.2, 0.2, 1.0)
        assert lo < mid < hi

    def test_lockstep_equals_sum_of_book_risks(self):
        s, c = [0.01, 0.01], [0.008]
        r_s = diversified_book_risk(s, 1.0)
        r_c = diversified_book_risk(c, 1.0)
        acct = rb.account_stop_risk(s, c, 1.0, 1.0, 1.0)
        assert acct == pytest.approx(r_s + r_c, abs=1e-9)

    def test_empty_book(self):
        acct = rb.account_stop_risk([], [0.01], 0.3, 0.3, 0.5)
        assert acct == pytest.approx(diversified_book_risk([0.01], 0.3))


class TestAccountRiskBudget:
    def test_budget_brings_account_to_cap(self):
        s, c = [0.01], [0.01]
        cap = 0.03
        budget = rb.account_risk_budget('stock', s, c, 0.4, 0.4, 0.6,
                                        cap=cap, max_risk=0.05)
        # adding exactly `budget` to the stock book hits the cap
        acct = rb.account_stop_risk(s + [budget], c, 0.4, 0.4, 0.6)
        assert acct == pytest.approx(cap, abs=1e-6)

    def test_zero_when_cap_exhausted(self):
        # books already over the cap -> no room
        s, c = [0.025, 0.025], [0.025, 0.025]
        budget = rb.account_risk_budget('crypto', s, c, 0.8, 0.8, 0.9,
                                        cap=0.03)
        assert budget == 0.0

    def test_full_headroom_when_uncapped(self):
        s, c = [], []
        budget = rb.account_risk_budget('stock', s, c, 0.3, 0.3, 0.5,
                                        cap=0.03, max_risk=0.004)
        # tiny lone position is well under cap -> full requested risk allowed
        assert budget == pytest.approx(0.004, abs=1e-9)

    def test_scale_only_shrinks(self):
        s, c = [0.02], [0.02]
        # a candidate that would breach the cap gets scale < 1
        scale, _ = rb.scale_for_account_cap(0.02, 'stock', s, c, 0.6, 0.6, 0.8)
        assert 0.0 <= scale < 1.0
        # a tiny candidate in an empty account is untouched
        scale2, _ = rb.scale_for_account_cap(0.002, 'stock', [], [], 0.3, 0.3,
                                             0.5)
        assert scale2 == pytest.approx(1.0)

    def test_fail_open_on_bad_input(self):
        scale, budget = rb.scale_for_account_cap(np.nan, 'stock', [], [],
                                                 0.3, 0.3, 0.5)
        assert scale == 1.0 and budget == float('inf')


class TestAllocateBookCaps:
    def test_lower_vol_book_gets_more(self):
        cap_s, cap_c = rb.allocate_book_caps(0.10, 0.40)  # stock calmer
        assert cap_s > cap_c
        assert cap_s + cap_c == pytest.approx(rb.ACCOUNT_RISK_CAP, abs=1e-9)

    def test_clamped(self):
        # extreme vol disparity is clamped to [0.25,0.75]
        cap_s, cap_c = rb.allocate_book_caps(0.001, 10.0)
        assert cap_s == pytest.approx(rb.ACCOUNT_RISK_CAP * 0.75, abs=1e-9)

    def test_even_split_when_vol_missing(self):
        cap_s, cap_c = rb.allocate_book_caps(None, 0.3)
        assert cap_s == pytest.approx(cap_c)


class TestTwoBookSimulator:
    def _trades(self, pnls, weight=1.0):
        return [{'exit_period': i, 'net_pct': p, 'weight': weight}
                for i, p in enumerate(pnls)]

    def test_combined_equity_is_sum(self):
        s = self._trades([1.0, -2.0, 1.0])
        c = self._trades([0.5, 0.5, -1.0])
        out = rb.simulate_two_books(s, c, periods=3)
        # total = sum of all net_pct
        assert out['combined_total_pct'] == pytest.approx(1.0 - 2 + 1 + 0.5 + 0.5 - 1)

    def test_correlated_books_deepen_drawdown(self):
        # two books that draw down in the SAME periods vs OFFSET periods
        together_s = self._trades([1, -3, -3, 2])
        together_c = self._trades([1, -3, -3, 2])
        out_together = rb.simulate_two_books(together_s, together_c, periods=4)

        offset_s = self._trades([-3, 2, 1, -3])
        offset_c = self._trades([2, -3, -3, 2])  # losses in different periods
        out_offset = rb.simulate_two_books(offset_s, offset_c, periods=4)

        assert (out_together['combined_max_drawdown_pct']
                > out_offset['combined_max_drawdown_pct'])
        assert out_together['drawdown_concentration'] > 0.9  # near lockstep

    def test_realized_cross_corr_sign(self):
        s = self._trades([1, -1, 1, -1, 1, -1])
        c = self._trades([1, -1, 1, -1, 1, -1])  # identical -> corr ~ +1
        out = rb.simulate_two_books(s, c, periods=6)
        assert out['realized_cross_corr'] == pytest.approx(1.0, abs=1e-6)

    def test_empty(self):
        out = rb.simulate_two_books([], [])
        assert out['n_periods'] == 0 and out['combined_sharpe'] == 0.0

    def test_weight_scales_pnl(self):
        full = rb.simulate_two_books(self._trades([2.0], 1.0), [], periods=1)
        half = rb.simulate_two_books(self._trades([2.0], 0.5), [], periods=1)
        assert half['combined_total_pct'] == pytest.approx(
            full['combined_total_pct'] * 0.5)
