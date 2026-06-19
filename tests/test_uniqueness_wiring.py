"""Wave-8 #1: fold_train_weights — the missing train-side uniqueness wiring.

The DSR gate already deflates by effective-n; these guard the OTHER half — the
per-row training weight that stops overlapping hourly labels being over-counted.
The headline hazard is ALIGNMENT: uniqueness must be computed over the full panel
(per-ticker concurrency) and then sliced to the fold's panel row indices, never
slice-then-compute. And the default must be PURE uniqueness, never the toxic
uniqueness x |return| blend.
"""
import inspect

import numpy as np
import pytest

from sample_weights import (
    average_uniqueness,
    fold_train_weights,
)


def _mean1(w):
    f = w[np.isfinite(w) & (w > 0)]
    return f.mean()


def test_weights_are_mean_one_over_finite_rows():
    rng = np.random.default_rng(1)
    hold = rng.integers(0, 6, size=200).astype(float)
    w = fold_train_weights(hold, np.arange(200))
    assert _mean1(w) == pytest.approx(1.0)


def test_overlap_is_down_weighted_relative_to_unique_rows():
    # One ticker: first 20 rows heavily overlap (span 8), last 20 are unique (span 0).
    hold = np.concatenate([np.full(20, 8.0), np.full(20, 0.0)])
    w = fold_train_weights(hold, np.arange(40))
    overlap_w = w[:20].mean()
    unique_w = w[20:].mean()
    assert unique_w > overlap_w            # fresh labels carry more weight
    assert _mean1(w) == pytest.approx(1.0)


def test_nan_span_rows_get_zero_weight():
    hold = np.array([0.0, 0.0, np.nan, 0.0, -1.0, 0.0])
    w = fold_train_weights(hold, np.arange(6))
    assert w[2] == 0.0 and w[4] == 0.0
    assert np.all(w[[0, 1, 3, 5]] > 0)


def test_determinism_gives_mean_and_q10_legs_the_same_vector():
    rng = np.random.default_rng(2)
    hold = rng.integers(0, 5, size=120).astype(float)
    idx = np.arange(120)
    a = fold_train_weights(hold, idx)
    b = fold_train_weights(hold, idx)
    # Identical inputs -> identical vector, so the LGB-mean and LGB-q10 legs are
    # provably weighted by the SAME mass (a correctness requirement).
    assert np.array_equal(a, b)


def test_alignment_uses_full_panel_concurrency_not_slice_then_compute():
    # Two tickers; A heavily overlaps, B is unique. Boundaries keep concurrency
    # from leaking across the seam.
    hold = np.concatenate([np.full(10, 3.0), np.full(10, 0.0)])
    bounds = [(0, 10), (10, 20)]
    idx = np.array([2, 3, 4, 5, 12, 13])

    w = fold_train_weights(hold, idx, ticker_boundaries=bounds)

    # Correct: full-panel uniqueness, then sliced to idx, then mean-1.
    u_full = average_uniqueness(hold, bounds)[idx]
    expected = u_full / u_full[np.isfinite(u_full) & (u_full > 0)].mean()
    np.testing.assert_allclose(w, expected, rtol=1e-12)

    # Wrong: slicing the spans first loses the overlap context -> different.
    u_sliced = average_uniqueness(hold[idx])
    wrong = u_sliced / u_sliced[np.isfinite(u_sliced) & (u_sliced > 0)].mean()
    assert not np.allclose(w, wrong)


def test_pure_uniqueness_differs_from_magnitude_blend():
    # The default (returns=None) must NOT equal the |return|-blended variant, so
    # the safe path is provably the one in use.
    rng = np.random.default_rng(3)
    hold = rng.integers(0, 5, size=150).astype(float)
    rets = rng.normal(scale=3.0, size=150)
    idx = np.arange(150)
    pure = fold_train_weights(hold, idx, returns=None)
    blended = fold_train_weights(hold, idx, returns=rets)
    assert not np.allclose(pure, blended)
    assert _mean1(pure) == pytest.approx(1.0)
    assert _mean1(blended) == pytest.approx(1.0)


def test_per_book_normalization_avoids_cross_book_imbalance():
    # Book A heavily overlaps (low u), book B unique (u=1). Per-book weighting
    # gives EACH book mean 1; pooling one normalization imbalances them
    # (the documented A 0.38 vs B 1.62 failure).
    hold = np.concatenate([np.full(30, 9.0), np.full(30, 0.0)])
    bounds = [(0, 30), (30, 60)]
    wa = fold_train_weights(hold, np.arange(0, 30), ticker_boundaries=bounds)
    wb = fold_train_weights(hold, np.arange(30, 60), ticker_boundaries=bounds)
    assert _mean1(wa) == pytest.approx(1.0)
    assert _mean1(wb) == pytest.approx(1.0)

    pooled = fold_train_weights(hold, np.arange(60), ticker_boundaries=bounds)
    # Pooled: the overlapping book is pulled BELOW 1 and the unique book ABOVE 1.
    assert pooled[:30].mean() < 0.9 < 1.1 < pooled[30:].mean()


def test_train_lgb_accepts_sample_weight_param():
    import model_lgb
    sig = inspect.signature(model_lgb.train_lgb)
    assert 'sample_weight' in sig.parameters
    assert 'sample_weight_val' in sig.parameters
    assert sig.parameters['sample_weight'].default is None
    assert sig.parameters['sample_weight_val'].default is None
