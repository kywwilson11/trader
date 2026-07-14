"""Stage-3 improvement batch for sample_weights.py.

Covers:
  - C1: overflow-safe span clamp in _avg_uniqueness_block (inf / huge spans
    no longer crash or corrupt the numba int64 cast; they clip to block end).
  - C2: dedupe of the blend/normalize logic shared by uniqueness_weights and
    fold_train_weights into _blend_normalize, plus a fail-loud returns-length
    alignment guard.
  - C3: single dtype conversion in clustered_effective_n's datetime64 branch.

Golden-value / identity assertions double as behavior-neutrality anchors for
the refactor.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sample_weights as sw


def test_inf_span_no_crash_clips_to_block_end():
    u = sw.average_uniqueness(np.array([np.inf, 1.0, 1.0]))
    assert np.all(np.isfinite(u))
    assert 0.0 < u[0] <= 1.0
    ref = sw.average_uniqueness(np.array([3.0, 1.0, 1.0]))
    assert np.allclose(u, ref)


def test_huge_span_matches_end_clip():
    u = sw.average_uniqueness(np.array([1e300, 1.0, 1.0]))
    ref = sw.average_uniqueness(np.array([3.0, 1.0, 1.0]))
    assert np.allclose(u, ref)


def test_golden_uniqueness_values():
    # conc = [1, 2, 2]; u0 = mean(1, 1/2, 1/2) = 2/3; u1 = mean(1/2, 1/2) = 1/2
    u = sw.average_uniqueness(np.array([2.0, 2.0, np.nan]))
    assert u[0] == pytest.approx(2 / 3)
    assert u[1] == pytest.approx(0.5)
    assert np.isnan(u[2])


def test_uniqueness_weights_matches_fold_identity():
    hb = np.array([3., 2., np.nan, 1., 4., 0., 2.])
    ret = np.array([0.1, -2., 0., 60., np.nan, 1., -0.5])

    w1 = sw.uniqueness_weights(hb)
    w2 = sw.fold_train_weights(hb, np.arange(7))
    assert np.allclose(w1, w2, equal_nan=True)

    w3 = sw.uniqueness_weights(hb, returns=ret)
    w4 = sw.fold_train_weights(hb, np.arange(7), returns=ret)
    assert np.allclose(w3, w4, equal_nan=True)


def test_rows_outside_boundaries_are_nan():
    u = sw.average_uniqueness(np.ones(6), ticker_boundaries=[(0, 2), (4, 6)])
    assert np.isnan(u[2]) and np.isnan(u[3])
    for i in (0, 1, 4, 5):
        assert np.isfinite(u[i])


def test_returns_length_mismatch_raises():
    with pytest.raises(ValueError):
        sw.uniqueness_weights(np.array([1., 1., 1.]), returns=np.array([0.5]))
    with pytest.raises(ValueError):
        sw.fold_train_weights(np.array([1., 1., 1.]), np.arange(3),
                              returns=np.array([0.5, 0.5]))


def test_clustered_effective_n_datetime_nat_golden():
    et = np.array(['2026-01-01T00', '2026-01-01T01', 'NaT', '2026-01-02T00'],
                  dtype='datetime64[h]')
    xt = np.array(['2026-01-01T02', '2026-01-01T03', '2026-01-01T05', '2026-01-02T01'],
                  dtype='datetime64[h]')
    assert sw.clustered_effective_n(et, xt) == 2
