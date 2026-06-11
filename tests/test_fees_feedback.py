"""Tests for the realized maker-share fee feedback (LIVE gate only)."""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fees
from fees import (crypto_entry_fee_bps, realized_crypto_maker_share,
                  required_edge_pct, round_trip_cost_pct)


@pytest.fixture
def journals(tmp_path, monkeypatch):
    """Point trade_journal.JOURNAL_DIR at a tmp dir; reset the cache."""
    import trade_journal
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', tmp_path)
    fees._maker_share_cache = None
    yield tmp_path
    fees._maker_share_cache = None


def _write_entries(path_dir, n_maker, n_taker, day_offset=0):
    day = (datetime.now().date() - timedelta(days=day_offset)).isoformat()
    rows = ([{'symbol': 'BTC/USD', 'action': 'buy', 'entry_tactic': 'maker'}]
            * n_maker
            + [{'symbol': 'ETH/USD', 'action': 'buy',
                'entry_tactic': 'taker_fallback'}] * n_taker)
    with open(path_dir / f'{day}.jsonl', 'a') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def test_share_computed_from_journals(journals):
    _write_entries(journals, n_maker=30, n_taker=10)
    assert realized_crypto_maker_share() == pytest.approx(0.75)


def test_thin_sample_returns_none_and_taker(journals):
    _write_entries(journals, n_maker=10, n_taker=5)  # 15 < 30 min
    assert realized_crypto_maker_share() is None
    assert crypto_entry_fee_bps(live=True) == fees.CRYPTO_TAKER_BPS


def test_static_contexts_always_taker(journals):
    _write_entries(journals, n_maker=40, n_taker=0)  # 100% maker realized
    # Training/backtest pricing must NOT move
    assert crypto_entry_fee_bps(live=False) == fees.CRYPTO_TAKER_BPS
    assert round_trip_cost_pct('crypto', 0.1) == pytest.approx(0.60)


def test_live_blend_and_edge_floor(journals):
    _write_entries(journals, n_maker=30, n_taker=10)  # 75% maker
    # entry = 0.75*15 + 0.25*25 = 17.5 bps; RT = 17.5+25 = 42.5bps + 10 spread
    assert crypto_entry_fee_bps(live=True) == pytest.approx(17.5)
    assert round_trip_cost_pct('crypto', 0.1, live=True) == pytest.approx(0.525)
    live_floor = required_edge_pct('crypto', 0.1, live=True)
    static_floor = required_edge_pct('crypto', 0.1)
    assert live_floor < static_floor  # realized fills relax the LIVE gate


def test_blend_bounded_by_fee_schedule(journals):
    _write_entries(journals, n_maker=100, n_taker=0)
    assert crypto_entry_fee_bps(live=True) == pytest.approx(
        fees.CRYPTO_MAKER_BPS)  # floor at pure maker, never below


def test_stale_journals_outside_window_ignored(journals):
    _write_entries(journals, n_maker=50, n_taker=0, day_offset=20)  # too old
    assert realized_crypto_maker_share() is None


def test_stock_and_skip_entries_excluded(journals):
    day = datetime.now().date().isoformat()
    rows = ([{'symbol': 'NVDA', 'action': 'buy', 'entry_tactic': 'marketable'}]
            * 50
            + [{'symbol': 'BTC/USD', 'action': 'skip',
                'entry_tactic': 'maker'}] * 50)
    with open(journals / f'{day}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    assert realized_crypto_maker_share() is None  # no crypto BUYS counted


def test_journal_failure_fails_safe(journals, monkeypatch):
    fees._maker_share_cache = None
    import trade_journal
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR',
                        Path('/nonexistent-dir-xyz'))
    assert realized_crypto_maker_share() is None
    assert crypto_entry_fee_bps(live=True) == fees.CRYPTO_TAKER_BPS
