"""Wave-9 #3 gate: per-name IC diagnostic that decides universe promotion."""
import numpy as np
import pytest

from ic_diagnostic import rank_ic, ic_by_name, promote_set


def test_rank_ic_basic():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=500)
    fwd = pred + rng.normal(0, 0.5, 500)            # pred predicts fwd
    assert rank_ic(pred, fwd) > 0.5
    assert abs(rank_ic(pred, rng.normal(size=500))) < 0.15   # noise ~ 0
    assert rank_ic([1, 2], [1, 2]) is None          # too few
    assert rank_ic(np.ones(50), rng.normal(size=50)) is None  # degenerate pred


def _rows(name, pred, fwd):
    return [{'symbol': name, 'pred': float(p), 'fwd_return': float(f)}
            for p, f in zip(pred, fwd)]


def test_ic_by_name_separates_edge_from_noise():
    rng = np.random.default_rng(2)
    n = 400
    edge_pred = rng.normal(size=n)
    edge_fwd = 0.6 * edge_pred + rng.normal(0, 0.5, n)        # real edge
    noise_pred = rng.normal(size=n)
    noise_fwd = rng.normal(size=n)                            # no edge
    rows = _rows('EDGE', edge_pred, edge_fwd) + _rows('NOISE', noise_pred, noise_fwd)

    table = ic_by_name(rows, n_subperiods=4)
    assert table['EDGE']['ic'] > 0.2
    assert table['EDGE']['positive_consistency'] >= 0.75      # positive across sub-periods
    assert abs(table['NOISE']['ic']) < 0.15
    assert len(table['EDGE']['subperiod_ics']) == 4


def test_promote_set_keeps_only_real_edge():
    rng = np.random.default_rng(3)
    n = 400
    rows = (_rows('GOOD', (g := rng.normal(size=n)), 0.6 * g + rng.normal(0, 0.5, n))
            + _rows('BAD', rng.normal(size=n), rng.normal(size=n)))
    table = ic_by_name(rows, n_subperiods=4)
    promoted = promote_set(table, min_ic=0.02, min_consistency=0.6)
    assert promoted == ['GOOD']                              # BAD is killed (training-only)


def test_promote_set_empty_when_nothing_predicts():
    rng = np.random.default_rng(4)
    rows = _rows('A', rng.normal(size=300), rng.normal(size=300)) \
        + _rows('B', rng.normal(size=300), rng.normal(size=300))
    assert promote_set(ic_by_name(rows), min_ic=0.05) == []
