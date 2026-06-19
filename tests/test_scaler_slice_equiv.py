"""Wave-8 #8: slicing the window BEFORE scaler.transform is bit-identical.

predict_now.get_live_prediction used to transform the entire fetched feature
window and then keep only the last `seq_len` rows. Because the live scaler
(sklearn RobustScaler) applies a strictly per-column, row-independent affine map
(x - center) / scale, transforming only the last `seq_len` rows yields the EXACT
same `sequence` that feeds the model — for far fewer rows on the 30s Jetson cycle.

These tests prove the invariant on a dependency-free RobustScaler-equivalent (no
sklearn/torch needed on the dev Mac) and guard the source change against a revert.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _robust_transform(X, center, scale):
    """Exactly what sklearn RobustScaler.transform does: per-column, row-wise."""
    return (X - center) / scale


@pytest.mark.parametrize("n_rows,seq_len,n_feat", [
    (120, 8, 12), (100, 40, 30), (74, 24, 18), (50, 50, 5), (200, 1, 3),
])
def test_slice_then_transform_is_bit_identical(n_rows, seq_len, n_feat):
    rng = np.random.default_rng(seq_len * 97 + n_feat)
    X = rng.normal(size=(n_rows, n_feat)) * rng.uniform(0.5, 5.0, n_feat)
    # RobustScaler params: column median + IQR (any finite, non-zero scale works).
    center = np.median(X, axis=0)
    scale = np.subtract(*np.percentile(X, [75, 25], axis=0))
    scale[scale == 0] = 1.0

    full_then_slice = _robust_transform(X, center, scale)[-seq_len:]
    slice_then_full = _robust_transform(X[-seq_len:], center, scale)

    # Bit-identical, not merely close — row-independence makes this exact.
    assert np.array_equal(full_then_slice, slice_then_full)
    # And the reshape the model consumes is identical too.
    a = full_then_slice.reshape(1, seq_len, -1)
    b = slice_then_full.reshape(1, seq_len, -1)
    assert np.array_equal(a, b)


def test_equivalence_holds_with_negative_centers_and_outliers():
    rng = np.random.default_rng(7)
    X = rng.normal(-3.0, 2.0, size=(90, 8))
    X[0, :] = 1e6  # leading outlier that the slice discards
    center = np.median(X, axis=0)
    scale = np.subtract(*np.percentile(X, [75, 25], axis=0))
    scale[scale == 0] = 1.0
    seq_len = 16
    assert np.array_equal(
        _robust_transform(X, center, scale)[-seq_len:],
        _robust_transform(X[-seq_len:], center, scale),
    )


def test_source_uses_sliced_transform_no_dead_intermediate():
    """Regression guard: the dead full-window intermediate stays gone."""
    src = (REPO / "predict_now.py").read_text()
    assert "scaler_X.transform(current_features[-seq_len:])" in src, \
        "expected the sliced transform in get_live_prediction"
    # The old dead intermediate must not come back in the live path.
    assert "current_features_scaled = scaler_X.transform(current_features)" not in src, \
        "the full-window transform intermediate was reintroduced"
    # Sanity: predict_now still parses.
    ast.parse(src)
