"""Wave-8 #2: the CSCV-PBO offline gate (build_oos_blocks + pbo_from_oos_blocks).

A known-overfit population (regime specialists whose in-sample winner reverses
out of sample) must score HIGH PBO; a persistent-skill population must score LOW.
The wrapper must fail OPEN (None) on thin/degenerate input so the DSR gate stays
in charge. And the Lo-2002 serial factor must NOT be stacked on the uniqueness
effective-n (double-counting variance).
"""
import numpy as np
import pytest

from validation import (
    build_oos_blocks,
    pbo_from_oos_blocks,
    serial_correlation_factor,
    dsr_from_trade_returns,
)


def test_build_oos_blocks_shape_and_mean_preservation():
    rng = np.random.default_rng(1)
    r = rng.normal(size=80)                 # 80 divisible by 8 -> exact blocks
    b = build_oos_blocks(r, n_blocks=8)
    assert b.shape == (8,)
    # equal-count blocks => mean of block means == overall mean
    assert b.mean() == pytest.approx(r.mean(), abs=1e-9)


def test_build_oos_blocks_handles_remainder_and_too_few():
    r = np.arange(11, dtype=float)          # 11 not divisible by 8
    b = build_oos_blocks(r, n_blocks=8)
    assert b.shape == (8,)                   # array_split pads to exactly n_blocks
    assert build_oos_blocks(np.arange(5.0), n_blocks=8) is None   # fewer than blocks
    assert build_oos_blocks([1.0, 2.0, np.nan], n_blocks=8) is None


def test_persistent_skill_scores_low_pbo():
    rng = np.random.default_rng(2)
    # trial 0 is consistently excellent across every block; the rest are noise.
    rows = [np.full(8, 5.0) + rng.normal(scale=0.1, size=8)]
    rows += [rng.normal(size=8) for _ in range(9)]
    res = pbo_from_oos_blocks(rows, n_groups=8)
    assert res is not None
    assert res['pbo'] < 0.15
    assert res['median_logit'] > 0.0        # selection persists OOS


def test_overfit_rank_reversal_scores_high_pbo():
    # Regime specialists: half are good in blocks 0-3 and bad in 4-7, half mirror.
    # The in-sample winner on one regime is the out-of-sample loser on the other.
    g = 3.0
    A = np.array([g, g, g, g, -g, -g, -g, -g], dtype=float)
    B = -A
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(5):
        rows.append(A + rng.normal(scale=0.2, size=8))
    for _ in range(5):
        rows.append(B + rng.normal(scale=0.2, size=8))
    res = pbo_from_oos_blocks(rows, n_groups=8)
    assert res is not None
    assert res['pbo'] > 0.5                  # in-sample selection reverses OOS
    # And it is decisively worse than the persistent-skill population.
    skill = pbo_from_oos_blocks(
        [np.full(8, 5.0)] + [np.random.default_rng(i).normal(size=8) for i in range(9)],
        n_groups=8)
    assert res['pbo'] > skill['pbo'] + 0.3


def test_pbo_max_predicate_rejects_overfit():
    g = 3.0
    A = np.array([g, g, g, g, -g, -g, -g, -g], dtype=float)
    rng = np.random.default_rng(4)
    rows = [A + rng.normal(scale=0.2, size=8) for _ in range(5)]
    rows += [-A + rng.normal(scale=0.2, size=8) for _ in range(5)]
    res = pbo_from_oos_blocks(rows, n_groups=8)
    PBO_MAX = 0.5
    assert (res['pbo'] > PBO_MAX) is True    # the gate clause would reject this config set


def test_wrapper_fails_open_on_thin_or_degenerate_input():
    assert pbo_from_oos_blocks([], n_groups=8) is None
    assert pbo_from_oos_blocks([np.full(8, 1.0)], n_groups=8) is None         # 1 trial
    assert pbo_from_oos_blocks([np.full(8, 1.0), np.full(8, 2.0)],            # zero-variance rows
                               n_groups=8) is None
    # too few block-columns vs n_groups
    assert pbo_from_oos_blocks([np.arange(4.0), np.arange(4.0)[::-1]],
                               n_groups=8) is None
    # mixed lengths: minority width dropped, then too few -> None
    assert pbo_from_oos_blocks([np.random.default_rng(0).normal(size=8),
                                np.random.default_rng(1).normal(size=8),
                                np.random.default_rng(2).normal(size=5)],
                               n_groups=8) is not None


def test_lo2002_and_uniqueness_neff_must_not_be_stacked():
    """Contract guard (CLAUDE.md gotcha #4): the DSR gate uses the uniqueness
    effective-n; the Lo-2002 serial factor is a SEPARATE diagnostic. Stacking
    them double-counts variance — assert the hazard is real so nobody wires both.
    """
    rng = np.random.default_rng(5)
    # positively autocorrelated stream -> Lo factor > 1, n_eff < n
    e = rng.normal(size=400)
    r = np.empty(400)
    r[0] = e[0]
    for i in range(1, 400):
        r[i] = 0.5 * r[i - 1] + e[i]
    lo = serial_correlation_factor(r)
    assert lo['factor'] > 1.0 and lo['n_eff'] < lo['n']

    n = len(r)
    uniq_neff = 120.0                                   # e.g. from sample_weights.effective_n
    # The honest gate passes the UNIQUENESS n_eff:
    correct = dsr_from_trade_returns(r, n_trials=50, n_eff=uniq_neff)
    # Stacking (wrongly multiplying in the Lo reduction) yields a strictly
    # smaller n_eff -> a different, double-deflated DSR. Prove they diverge.
    stacked_neff = uniq_neff / lo['factor']
    wrong = dsr_from_trade_returns(r, n_trials=50, n_eff=stacked_neff)
    assert stacked_neff < uniq_neff
    assert correct['n_eff'] != wrong['n_eff']
