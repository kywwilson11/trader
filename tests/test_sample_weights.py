"""Wave-6 Tier-1: average-uniqueness sample weights + effective-n DSR.

These cover the de-biasing dependency root: the uniqueness producer
(sample_weights.py) and the effective-n correction to the Deflated-Sharpe
promotion gate (validation.py). The decisive assertion is that overlapping
labels yield a LOWER DSR than the IID assumption — the gate gets harder, never
easier."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sample_weights as sw
import validation


class TestAverageUniqueness:
    def test_non_overlapping_labels_are_fully_unique(self):
        # Each label resolves in 1 bar and the next starts after it -> no
        # overlap -> uniqueness 1.0 everywhere.
        hold = np.zeros(10, dtype=float)  # span [i, i] = single bar
        u = sw.average_uniqueness(hold)
        assert np.allclose(u, 1.0)
        assert sw.effective_n(u) == pytest.approx(10.0)

    def test_heavy_overlap_shrinks_uniqueness(self):
        # Every row holds 5 bars -> deep overlap in the middle -> u << 1.
        hold = np.full(20, 5.0)
        u = sw.average_uniqueness(hold)
        assert np.all(u <= 1.0)
        assert np.nanmean(u) < 0.5  # heavily overlapped
        # n_eff strictly below the raw count
        assert sw.effective_n(u) < 20.0

    def test_more_overlap_means_smaller_neff(self):
        n = 40
        light = sw.effective_n(sw.average_uniqueness(np.full(n, 2.0)))
        heavy = sw.effective_n(sw.average_uniqueness(np.full(n, 12.0)))
        assert heavy < light < n

    def test_nan_spans_are_skipped(self):
        hold = np.array([2.0, np.nan, 2.0, 2.0, np.nan])
        u = sw.average_uniqueness(hold)
        assert math.isnan(u[1]) and math.isnan(u[4])
        # NaN rows contribute nothing to n_eff
        assert sw.effective_n(u) == pytest.approx(np.nansum(u))

    def test_ticker_boundaries_isolate_concurrency(self):
        # Two independent 1-bar-label blocks: a label in block A must not see
        # block B's concurrency. Boundaries make every label fully unique.
        hold = np.zeros(8, dtype=float)
        bounds = {'A': (0, 4), 'B': (4, 8)}
        u = sw.average_uniqueness(hold, bounds)
        assert np.allclose(u, 1.0)

    def test_boundaries_dict_and_tuples_match(self):
        hold = np.full(12, 3.0)
        u_dict = sw.average_uniqueness(hold, {'A': (0, 6), 'B': (6, 12)})
        u_tup = sw.average_uniqueness(hold, [(0, 6), (6, 12)])
        assert np.allclose(u_dict, u_tup, equal_nan=True)

    def test_empty_input(self):
        assert sw.average_uniqueness(np.array([])).shape == (0,)


class TestEffectiveN:
    def test_mask_selects_subset(self):
        u = np.array([1.0, 0.5, 0.25, np.nan])
        mask = np.array([True, True, False, True])
        # 1.0 + 0.5 (+ nan ignored)
        assert sw.effective_n(u, mask) == pytest.approx(1.5)

    def test_ignores_nan(self):
        assert sw.effective_n(np.array([1.0, np.nan, 0.5])) == pytest.approx(1.5)


class TestUniquenessWeights:
    def test_mean_normalized_to_one(self):
        hold = np.full(30, 4.0)
        w = sw.uniqueness_weights(hold)
        finite = w[w > 0]
        assert finite.mean() == pytest.approx(1.0, abs=1e-9)

    def test_return_magnitude_blend(self):
        hold = np.ones(20)
        rets = np.linspace(-3, 3, 20)
        w = sw.uniqueness_weights(hold, returns=rets)
        # Larger |return| -> larger weight (after uniqueness, which is uniform
        # here since spans are equal).
        assert w[0] > w[10]  # |−3| emphasized over |~0|

    def test_return_cap_applied(self):
        hold = np.zeros(5)  # single-bar spans -> uniqueness uniformly 1.0
        rets = np.array([0.0, 0.0, 0.0, 0.0, 1000.0])
        w = sw.uniqueness_weights(hold, returns=rets, ret_cap=50.0)
        # the 1000-return row is capped, not unbounded
        assert w[4] / w[0] == pytest.approx((50.0) / 1.0, rel=0.01)

    def test_nan_spans_get_zero_weight(self):
        hold = np.array([2.0, np.nan, 2.0])
        w = sw.uniqueness_weights(hold)
        assert w[1] == 0.0


class TestEffectiveNDSR:
    """The flagship: feeding measured n_eff makes the gate HARDER."""

    def _series(self, n=300, seed=0):
        rng = np.random.RandomState(seed)
        # mild positive-mean noisy returns
        return rng.normal(0.05, 1.0, n)

    def test_neff_below_n_lowers_dsr(self):
        r = self._series()
        iid = validation.dsr_from_trade_returns(r, n_trials=200)
        overlapping = validation.dsr_from_trade_returns(
            r, n_trials=200, n_eff=len(r) / 4.0)
        # Same observed Sharpe, fewer independent samples -> wider null,
        # higher expected-max bar -> strictly lower DSR.
        assert overlapping['dsr'] < iid['dsr']
        assert overlapping['n_eff'] == pytest.approx(len(r) / 4.0)
        assert iid['n_eff'] == pytest.approx(float(len(r)))

    def test_neff_defaults_to_iid(self):
        r = self._series()
        a = validation.dsr_from_trade_returns(r, n_trials=200)
        b = validation.dsr_from_trade_returns(r, n_trials=200, n_eff=len(r))
        assert a['dsr'] == pytest.approx(b['dsr'])

    def test_neff_clamped_to_raw_count(self):
        # An n_eff above the row count must not loosen the gate.
        r = self._series()
        a = validation.dsr_from_trade_returns(r, n_trials=200)
        b = validation.dsr_from_trade_returns(
            r, n_trials=200, n_eff=len(r) * 10.0)
        assert b['dsr'] == pytest.approx(a['dsr'])
        assert b['n_eff'] == pytest.approx(float(len(r)))

    def test_deflated_sharpe_ratio_neff_param(self):
        # Direct: lower n_eff -> smaller z -> lower probability.
        hi = validation.deflated_sharpe_ratio(0.15, 0.05, n_obs=400)
        lo = validation.deflated_sharpe_ratio(0.15, 0.05, n_obs=400, n_eff=40)
        assert lo < hi

    def test_end_to_end_overlap_vs_iid_on_spans(self):
        # Build a synthetic block-overlapping label set, derive n_eff from the
        # uniqueness module, and confirm the gate is harder than IID.
        r = self._series(n=240, seed=3)
        hold = np.full(240, 8.0)  # heavy overlap
        u = sw.average_uniqueness(hold)
        n_eff = sw.effective_n(u)
        assert n_eff < 240
        overlapping = validation.dsr_from_trade_returns(
            r, n_trials=150, n_eff=n_eff)
        iid = validation.dsr_from_trade_returns(r, n_trials=150)
        assert overlapping['dsr'] < iid['dsr']
