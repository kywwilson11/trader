"""Wave-9 #3: live tradable-universe promotion + sector-bucket completeness."""
import numpy as np
import pytest

from panel_ranks import live_tradable_members


def _dvs(**kw):
    return {k: float(v) for k, v in kw.items()}


def test_top_k_slice_by_dollar_volume():
    dvs = _dvs(A=10, B=9, C=8, D=7, E=6)
    out = live_tradable_members(dvs, top_k=3)
    assert out == ['A', 'B', 'C']            # top 3 by dv


def test_returned_in_dv_rank_order():
    dvs = _dvs(A=1, B=5, C=3, D=9)
    out = live_tradable_members(dvs, top_k=4)
    assert out == ['D', 'B', 'C', 'A']


def test_hysteresis_keeps_held_name_in_the_band():
    dvs = _dvs(A=10, B=9, C=8, D=7, E=6, F=5)
    # D is rank 3 (0-based) -> outside k_enter=3 but inside k_hold=5.
    assert 'D' not in live_tradable_members(dvs, top_k=3, k_enter=3, k_hold=5)
    assert 'D' in live_tradable_members(dvs, top_k=3, k_enter=3, k_hold=5, held={'D'})
    # ...but a name PAST k_hold that is NOT held stays out.
    assert 'F' not in live_tradable_members(dvs, top_k=3, k_enter=3, k_hold=5, held={'D'})


def test_held_name_past_k_hold_is_still_included_for_exit():
    dvs = _dvs(A=10, B=9, C=8, D=7, E=6, Z=1)
    out = live_tradable_members(dvs, top_k=2, k_enter=2, k_hold=3, held={'Z'})
    assert 'Z' in out                        # never orphan a live position


def test_deterministic_on_ties():
    dvs = _dvs(B=5, A=5, C=5)                 # equal dv -> symbol order
    assert live_tradable_members(dvs, top_k=2) == ['A', 'B']


def test_fail_open_on_bad_dv_and_empty():
    dvs = {'A': 10.0, 'B': np.nan, 'C': 0.0, 'D': -1.0, 'E': 4.0}
    out = live_tradable_members(dvs, top_k=5)
    assert out == ['A', 'E']                 # non-finite / non-positive dropped
    # empty dv but a held position -> still managed
    assert live_tradable_members({}, top_k=5, held={'X'}) == ['X']


def test_sector_map_covers_previously_unmapped_names():
    from stock_config import SECTOR_BUCKETS, TRAINING_CANDIDATE_POOL, load_stock_universe
    # Every non-ETF training-pool name now has a bucket (was uncapped before).
    for name in TRAINING_CANDIDATE_POOL:
        assert name in SECTOR_BUCKETS, f"{name} still unmapped (uncapped)"
    # A representative sampling from the live universe is mapped too.
    for name in ('JPM', 'XOM', 'JNJ', 'PG', 'CAT', 'META', 'TSLA', 'SPY'):
        assert name in SECTOR_BUCKETS
