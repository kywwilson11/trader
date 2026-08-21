"""Stage-3(v3) hardening batch for sample_weights.py.

Covers the input-validation guards added on top of the Stage-3(C1-C3) refactor:
  - row_indices / mask contract enforcement (fold_train_weights, effective_n) —
    the #1 documented alignment hazard (silent bool/float reinterpretation).
  - ticker_boundaries disjointness + non-empty requirements in
    average_uniqueness (silent overwrite / all-NaN otherwise).
  - clustered_effective_n dtype/length guards (object dtype, mixed
    datetime/numeric, length mismatch) replacing silent 0/garbage returns.
  - _blend_normalize fail-closed on degenerate (all-NaN / all-zero-weight)
    input and on invalid ret_cap, instead of silently emitting a zero vector.
  - the njit no-op fallback's decorator-argument-form handling (adopted from
    policy_exits.py) and the _avg_uniqueness_block dead-branch removal, which
    must remain EXACTLY bit-identical to the original three-pass kernel.

All of this needs only numpy + pytest; numba's absence on this dev Mac is
covered by the in-module no-op shim, so no importorskip is required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sample_weights as sw


# ---------------------------------------------------------------------------
# fold_train_weights: row_indices contract
# ---------------------------------------------------------------------------

def test_row_indices_bool_mask_rejected():
    with pytest.raises(TypeError):
        sw.fold_train_weights(np.full(10, 3.0), np.zeros(10, dtype=bool))


def test_row_indices_float_rejected():
    with pytest.raises(TypeError):
        sw.fold_train_weights(np.full(8, 2.0), np.array([0.9, 1.9]))


def test_row_indices_negative_rejected():
    with pytest.raises(IndexError):
        sw.fold_train_weights(np.full(5, 1.0), np.array([-1, 0]))


def test_row_indices_out_of_range_rejected():
    with pytest.raises(IndexError):
        sw.fold_train_weights(np.full(5, 1.0), np.array([0, 5]))


def test_row_indices_2d_rejected():
    with pytest.raises(ValueError):
        sw.fold_train_weights(np.full(6, 1.0), np.arange(6).reshape(2, 3))


def test_row_indices_valid_and_empty_paths_unchanged():
    mask = np.zeros(10, bool)
    mask[3:7] = True
    w = sw.fold_train_weights(np.full(10, 3.0), np.flatnonzero(mask))
    assert w.shape == (4,) and np.allclose(w, 1.0)
    # empty-list tolerance pinned
    assert sw.fold_train_weights(np.full(5, 1.0), []).shape == (0,)


# ---------------------------------------------------------------------------
# _blend_normalize: fail-closed on degenerate weight mass
# ---------------------------------------------------------------------------

def test_all_nan_spans_raise_fold():
    with pytest.raises(ValueError):
        sw.fold_train_weights(np.full(20, np.nan), np.arange(20))


def test_all_nan_spans_raise_uniqueness():
    with pytest.raises(ValueError):
        sw.uniqueness_weights(np.full(6, np.nan))


def test_partial_nan_keeps_mean1_and_mass():
    hold = np.array([2.0, 2.0, np.nan, 2.0, np.nan, 2.0])
    w = sw.uniqueness_weights(hold)
    assert w[2] == 0.0 and w[4] == 0.0
    assert w[w > 0].mean() == pytest.approx(1.0)
    assert w.sum() == pytest.approx(4.0)


def test_empty_input_returns_empty_not_raise():
    assert sw.uniqueness_weights(np.array([])).shape == (0,)


# ---------------------------------------------------------------------------
# effective_n: mask contract
# ---------------------------------------------------------------------------

def test_effective_n_int_mask_rejected():
    with pytest.raises(TypeError):
        sw.effective_n(np.ones(3), mask=np.array([0, 1, 2]))


def test_effective_n_mask_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        sw.effective_n(np.ones(5), mask=np.array([True, False, True]))


def test_effective_n_bool_mask_still_works():
    assert sw.effective_n(np.array([1.0, 0.5, 0.25, np.nan]),
                          np.array([True, True, False, True])) == pytest.approx(1.5)


def test_effective_n_sentinel_zero_pinned():
    assert sw.effective_n(np.full(5, np.nan)) == 0.0
    assert sw.effective_n(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# clustered_effective_n: dtype / length guards
# ---------------------------------------------------------------------------

def test_clustered_length_mismatch_raises():
    with pytest.raises(ValueError):
        sw.clustered_effective_n(np.arange(5.0), np.arange(4.0))


def test_clustered_empty_returns_zero():
    assert sw.clustered_effective_n(np.array([]), np.array([])) == 0


def test_clustered_mixed_dtype_raises():
    dt = np.array(['2026-01-01', '2026-01-05'], dtype='datetime64[ns]')
    with pytest.raises(TypeError):
        sw.clustered_effective_n(np.array([0.0, 3.0]), dt)
    with pytest.raises(TypeError):
        sw.clustered_effective_n(dt, np.array([0.0, 3.0]))


def test_clustered_object_dtype_raises_clear():
    import datetime as _dt
    a = np.array([_dt.datetime(2026, 1, 1), _dt.datetime(2026, 1, 3)], dtype=object)
    b = np.array([_dt.datetime(2026, 1, 2), _dt.datetime(2026, 1, 4)], dtype=object)
    with pytest.raises(TypeError, match='object-dtype'):
        sw.clustered_effective_n(a, b)


# ---------------------------------------------------------------------------
# average_uniqueness: ticker_boundaries contract
# ---------------------------------------------------------------------------

def test_boundaries_overlap_raises():
    with pytest.raises(ValueError):
        sw.average_uniqueness(np.full(10, 2.0), {'A': (0, 6), 'B': (4, 10)})
    with pytest.raises(ValueError):
        sw.average_uniqueness(np.ones(12), [(0, 8), (4, 12)])


def test_boundaries_empty_raises():
    with pytest.raises(ValueError):
        sw.average_uniqueness(np.zeros(5), {})
    with pytest.raises(ValueError):
        sw.average_uniqueness(np.zeros(5), [])


def test_boundaries_touching_half_open_do_not_raise():
    # Half-open adjacency (0,3),(3,6) is the CANONICAL valid layout — the
    # overlap guard must be strictly `<`, never `<=`.
    u = sw.average_uniqueness(np.ones(6), {'A': (0, 3), 'B': (3, 6)})
    assert np.isfinite(u).all()


def test_boundaries_container_shapes_normalized():
    # ndarray (k,2) and mixed list/tuple spans must behave exactly like
    # the tuple form (raw sorted() would crash on both).
    hold = np.ones(8)
    ref = sw.average_uniqueness(hold, [(0, 4), (4, 8)])
    got_nd = sw.average_uniqueness(hold, np.array([[0, 4], [4, 8]]))
    got_mix = sw.average_uniqueness(hold, [[0, 4], (4, 8)])
    assert np.array_equal(got_nd, ref, equal_nan=True)
    assert np.array_equal(got_mix, ref, equal_nan=True)
    with pytest.raises(ValueError):  # guard still fires post-normalization
        sw.average_uniqueness(hold, np.array([[0, 5], [3, 8]]))


def test_boundaries_gap_still_nan():
    u = sw.average_uniqueness(np.ones(6), [(0, 2), (4, 6)])
    assert np.isnan(u[2:4]).all()
    assert np.isfinite(u[:2]).all() and np.isfinite(u[4:]).all()


def test_hold_bars_2d_raises():
    with pytest.raises(ValueError):
        sw.average_uniqueness(np.ones((4, 3)))


# ---------------------------------------------------------------------------
# _blend_normalize: ret_cap / returns validation
# ---------------------------------------------------------------------------

def test_ret_cap_invalid_raises_only_when_used():
    with pytest.raises(ValueError):
        sw.uniqueness_weights(np.zeros(4), returns=np.ones(4), ret_cap=0.0)
    with pytest.raises(ValueError):
        sw.uniqueness_weights(np.ones(3), returns=np.zeros(3), ret_cap=-2.0)
    # returns=None -> cap unused, must NOT raise
    w = sw.uniqueness_weights(np.zeros(4), ret_cap=0.0)
    assert w.shape == (4,)


def test_scalar_returns_raises_valueerror():
    with pytest.raises(ValueError):
        sw.uniqueness_weights(np.zeros(3), returns=1.5)


def test_nan_return_zeroes_that_row():
    w = sw.uniqueness_weights(np.zeros(4), returns=np.array([1.0, 1.0, np.nan, 1.0]))
    assert w[2] == 0.0 and (w[[0, 1, 3]] > 0).all()


# ---------------------------------------------------------------------------
# Kernel bit-identity: the Step 3 dead-branch removal must not change a
# single float operation's order or value.
# ---------------------------------------------------------------------------

def _reference_avg_uniqueness_block(hold_bars):
    """Plain-Python replica of the ORIGINAL _avg_uniqueness_block, including
    the removed `if c > 0.0` guard, the `cnt` counter, and the `else 1.0`
    fallback — kept byte-for-byte as it was before the Step 3 kernel edit."""
    n = len(hold_bars)
    out = np.empty(n, dtype=np.float64)
    diff = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        h = hold_bars[i]
        if not (h == h) or h < 0.0:  # NaN or negative -> not a label
            continue
        if h > n:
            h = float(n)
        end = i + int(h)
        if end > n - 1:
            end = n - 1
        diff[i] += 1.0
        diff[end + 1] -= 1.0
    conc = np.empty(n, dtype=np.float64)
    run = 0.0
    for t in range(n):
        run += diff[t]
        conc[t] = run
    for i in range(n):
        h = hold_bars[i]
        if not (h == h) or h < 0.0:
            out[i] = np.nan
            continue
        if h > n:
            h = float(n)
        end = i + int(h)
        if end > n - 1:
            end = n - 1
        s = 0.0
        cnt = 0
        for t in range(i, end + 1):
            c = conc[t]
            if c > 0.0:
                s += 1.0 / c
                cnt += 1
        out[i] = s / cnt if cnt > 0 else 1.0
    return out


def _random_blocks(seed=0, n_blocks=100):
    """100 random (n, hold_bars) blocks, n in [1, 200], values drawn from a
    fixed pathological set (NaN, negative, zero, unit, inf, huge, and
    n-dependent overflow triggers) — deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(n_blocks):
        n = int(rng.integers(1, 201))
        choices = np.array([np.nan, -3.0, 0.0, 1.0, 5.0, 17.0, np.inf, 1e300,
                            float(n), float(n + 5)])
        hb = rng.choice(choices, size=n).astype(np.float64)
        blocks.append(np.ascontiguousarray(hb))
    return blocks


def test_kernel_bit_identical_to_original_semantics():
    for hb in _random_blocks(seed=0, n_blocks=100):
        got = sw._avg_uniqueness_block(np.ascontiguousarray(hb))
        ref = _reference_avg_uniqueness_block(hb)
        assert np.array_equal(got, ref, equal_nan=True)


def test_u_property_unit_interval():
    for hb in _random_blocks(seed=0, n_blocks=100):
        u = sw._avg_uniqueness_block(np.ascontiguousarray(hb))
        finite = u[np.isfinite(u)]
        assert ((finite > 0) & (finite <= 1.0)).all()


# ---------------------------------------------------------------------------
# njit no-op fallback: decorator-argument-form handling (policy_exits.py form)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sw._HAS_NUMBA, reason='numba present — real njit in play')
def test_njit_fallback_signature_form():
    f = sw.njit('float64[:](float64[:])', cache=True)(lambda x: x)
    assert callable(f) and f(3) == 3
    g = sw.njit(lambda x: x + 1)
    assert g(1) == 2
    h = sw.njit(cache=True)(lambda x: x)
    assert h(7) == 7
