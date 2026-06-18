"""Wave-7: overnight forfeited-drift + GTC gap-through audit (gap_audit).

Pure analysis functions verified on synthetic data: the overnight/intraday
split, heavy-tail recovery (Student-t df + excess kurtosis), forfeited-drift
sign, and the gap-through EXCESS-beyond-stop accounting."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gap_audit as ga


def _daily(opens, closes):
    n = len(opens)
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    return pd.DataFrame({'Open': opens, 'High': np.maximum(opens, closes) * 1.001,
                         'Low': np.minimum(opens, closes) * 0.999,
                         'Close': closes}, index=idx)


class TestSplit:
    def test_overnight_and_intraday_decomposition(self):
        # close: 100 -> 110 -> 99 ; open: -, 105, 121
        df = _daily([100, 105, 121], [100, 110, 99])
        overnight, intraday = ga.overnight_intraday_returns(df)
        # overnight[1] = 105/100-1 = 0.05 ; 121/110-1 = 0.10
        assert overnight == pytest.approx([0.05, 0.10], abs=1e-9)
        # intraday[1] = 110/105-1 ; 99/121-1
        assert intraday == pytest.approx([110 / 105 - 1, 99 / 121 - 1], abs=1e-9)

    def test_too_short(self):
        o, i = ga.overnight_intraday_returns(_daily([100], [100]))
        assert len(o) == 0 and len(i) == 0


class TestGapStats:
    def test_recovers_heavy_tails(self):
        rng = np.random.RandomState(0)
        # synthetic Student-t df=4 overnight gaps -> excess kurtosis > 0,
        # fitted df in the heavy-tail range
        r = rng.standard_t(4, size=4000) * 0.01
        s = ga.gap_stats(r)
        assert s['excess_kurtosis'] > 0.5
        assert 2.5 < s['t_df'] < 8.0  # fit recovers a low df (heavy tail)

    def test_gaussian_is_light_tailed(self):
        rng = np.random.RandomState(1)
        r = rng.normal(0, 0.01, 4000)
        s = ga.gap_stats(r)
        assert abs(s['excess_kurtosis']) < 0.5  # ~0 for normal

    def test_too_few(self):
        assert ga.gap_stats(np.zeros(5))['std'] is None


class TestForfeitedDrift:
    def test_positive_drift_is_forfeited(self):
        r = np.full(252, 0.0004)  # +4bps/night
        fd = ga.forfeited_drift_annual(r, notional=5000)
        assert fd == pytest.approx(0.0004 * 5000 * 252, abs=1e-6)
        assert fd > 0

    def test_negative_overnight_session_is_a_benefit(self):
        r = np.full(252, -0.0003)
        assert ga.forfeited_drift_annual(r, 5000) < 0  # flatten avoids loss


class TestGapThrough:
    def test_only_excess_beyond_stop_counts(self):
        # long, 2% stop. gaps: -1% (within stop -> 0 excess), -5% (3% excess),
        # +4% (favorable -> 0). Over 252 nights repeating the worst case.
        r = np.array([-0.05] * 252)  # every night gaps 5% down
        gt = ga.gap_through_cost_annual(r, stop_dist_frac=0.02, notional=5000)
        # excess = 5% - 2% = 3% per night
        assert gt == pytest.approx(0.03 * 5000 * 252, abs=1e-6)

    def test_within_stop_gaps_cost_nothing_extra(self):
        r = np.array([-0.01] * 252)  # 1% down < 2% stop
        gt = ga.gap_through_cost_annual(r, stop_dist_frac=0.02, notional=5000)
        assert gt == pytest.approx(0.0, abs=1e-9)

    def test_short_side_hurt_by_gap_ups(self):
        r = np.array([0.05] * 252)  # gap UP hurts a short
        gt = ga.gap_through_cost_annual(r, 0.02, 5000, side='short')
        assert gt == pytest.approx(0.03 * 5000 * 252, abs=1e-6)


class TestAuditName:
    def test_composition(self):
        rng = np.random.RandomState(3)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
        opens = closes * (1 + rng.normal(0, 0.005, 300))
        df = _daily(opens, closes)
        a = ga.audit_name(df, notional=5000, stop_dist_frac=0.02)
        assert 'gap_stats' in a and 'gap_through_cost_annual' in a
        assert a['overnight_mean_bps'] is not None
        assert a['gap_through_cost_annual'] >= 0
