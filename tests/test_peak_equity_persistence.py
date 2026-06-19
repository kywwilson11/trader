"""Wave-8 #4: the drawdown ladder must survive restarts.

base_loop can't import on the dev Mac (torch/joblib via predict_now), so the
restart-survival logic lives in pure functions in drawdown.py and is tested here.
The headline scenario: a bot restarted at 15% drawdown from a prior 200k peak must
restore that peak and size at 0.50x — not reset to ~current equity and disable the
ladder. A source guard confirms base_loop actually persists the field.
"""
from pathlib import Path

import pytest

from drawdown import (
    update_peak_equity,
    restore_peak_equity,
    drawdown_fraction,
    drawdown_size_multiplier,
    DRAWDOWN_LADDER,
    PEAK_SEED,
)

REPO = Path(__file__).resolve().parent.parent


def test_update_peak_ratchets_up_only():
    assert update_peak_equity(100_000, 120_000) == 120_000
    assert update_peak_equity(120_000, 110_000) == 120_000     # never drops
    assert update_peak_equity(100_000, 100_000) == 100_000
    assert update_peak_equity(100_000, "bad") == 100_000       # bad input -> unchanged


def test_drawdown_fraction_and_multiplier_rungs():
    assert drawdown_fraction(200_000, 200_000) == 0.0
    assert drawdown_fraction(200_000, 170_000) == pytest.approx(0.15)
    assert drawdown_fraction(0, 50) == 0.0                     # undefined peak -> 0
    assert drawdown_fraction(100, 130) == 0.0                  # equity above peak -> floored
    assert drawdown_size_multiplier(0.05) == 1.0
    assert drawdown_size_multiplier(0.10) == 0.75
    assert drawdown_size_multiplier(0.15) == 0.50
    assert drawdown_size_multiplier(0.20) == 0.25
    assert drawdown_size_multiplier(0.50) == 0.25              # past the last rung


def test_restart_mid_drawdown_restores_peak_and_arms_ladder():
    # Bot was at a 200k peak, restarts with equity 170k (15% underwater).
    saved_peak = 200_000.0
    current = 170_000.0
    peak = restore_peak_equity(saved_peak, current)
    assert peak == 200_000.0                                   # NOT reset to ~current
    dd = drawdown_fraction(peak, current)
    assert dd == pytest.approx(0.15)
    assert drawdown_size_multiplier(dd) == 0.50                # ladder de-risks, not 1.0x


def test_restore_then_ratchet_does_not_clobber_higher_peak():
    peak = restore_peak_equity(200_000.0, 170_000.0)           # -> 200k
    # The very next equity update (still underwater) must NOT lower the peak.
    peak = update_peak_equity(peak, 170_000.0)
    assert peak == 200_000.0
    # A genuine new high does raise it.
    peak = update_peak_equity(peak, 210_000.0)
    assert peak == 210_000.0


def test_cold_start_and_legacy_and_corrupt_state():
    # Cold start: no equity yet -> seed.
    assert restore_peak_equity(None, 0.0) == PEAK_SEED
    # Legacy state file (no peak_equity) with a live equity -> max(seed, current).
    assert restore_peak_equity(None, 150_000.0) == 150_000.0
    # Corrupt / non-finite / non-positive saved values fall back to the seed.
    assert restore_peak_equity(float('nan'), 90_000.0) == PEAK_SEED
    assert restore_peak_equity(-5.0, 90_000.0) == PEAK_SEED
    assert restore_peak_equity("garbage", 120_000.0) == 120_000.0
    # Never below current equity.
    assert restore_peak_equity(50_000.0, 130_000.0) == 130_000.0


def test_ladder_is_monotone_non_increasing_in_drawdown():
    last = 1.0
    for dd in [0.0, 0.05, 0.10, 0.14, 0.15, 0.19, 0.20, 0.40]:
        m = drawdown_size_multiplier(dd)
        assert m <= last
        last = m
    # rungs are exactly the documented 25/50/75 cuts
    assert DRAWDOWN_LADDER == ((0.20, 0.25), (0.15, 0.50), (0.10, 0.75))


def test_base_loop_persists_peak_equity():
    """Regression guard: the save dict carries peak_equity and reconstruct restores it."""
    src = (REPO / "base_loop.py").read_text()
    assert "'peak_equity': self._peak_equity" in src, "save dict must persist peak_equity"
    assert "restore_peak_equity" in src, "reconstruct must restore the peak"
    assert "update_peak_equity" in src and "drawdown_size_multiplier" in src
