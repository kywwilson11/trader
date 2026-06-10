"""Tests for the headline novelty filter (shingle-Jaccard staleness)."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import novelty
from novelty import filter_novel, headline_novelty


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(novelty, '_STORE_FILE', tmp_path / 'store.json')
    monkeypatch.setattr(novelty, '_store', None)
    monkeypatch.setattr(novelty, '_dirty', False)
    yield tmp_path


def test_first_sighting_is_fully_novel():
    n = headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    assert n == 1.0


def test_exact_reprint_is_stale():
    h = 'Bitcoin surges past 100k on ETF inflows'
    headline_novelty('BTC/USD', h)
    assert headline_novelty('BTC/USD', h) < 0.01


def test_near_duplicate_detected():
    headline_novelty('BTC/USD',
                     'Bitcoin surges past 100k on record ETF inflows Monday')
    n = headline_novelty('BTC/USD',
                         'Bitcoin surges past 100k on record ETF inflows')
    assert n < novelty.NOVELTY_MIN  # near-identical wire reprint


def test_genuinely_different_story_is_novel():
    headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    n = headline_novelty('BTC/USD',
                         'SEC delays decision on Solana staking products')
    assert n > 0.9


def test_per_symbol_isolation():
    headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    n = headline_novelty('ETH/USD', 'Bitcoin surges past 100k on ETF inflows')
    assert n == 1.0  # ETH store never saw it


def test_window_expiry(monkeypatch):
    headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    # 8 days later the same story counts as fresh again
    real_time = time.time
    monkeypatch.setattr(time, 'time', lambda: real_time() + 8 * 86400)
    n = headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    assert n == 1.0


def test_filter_drops_reprints_keeps_fresh():
    h1 = 'Bitcoin surges past 100k on ETF inflows'
    h2 = 'Miner outflows hit yearly low as hashrate stabilizes'
    headline_novelty('BTC/USD', h1)
    out = filter_novel('BTC/USD', [h1, h2])
    assert out == [h2]


def test_filter_keeps_most_novel_when_all_stale():
    h1 = 'Bitcoin surges past 100k on ETF inflows'
    headline_novelty('BTC/USD', h1)
    out = filter_novel('BTC/USD', [h1])
    assert out == [h1]  # never returns empty when input is non-empty


def test_filter_empty_and_error_fail_open(monkeypatch):
    assert filter_novel('BTC/USD', []) == []
    monkeypatch.setattr(novelty, 'headline_novelty',
                        lambda *a, **k: 1 / 0)
    assert filter_novel('BTC/USD', ['x y z headline']) == ['x y z headline']


def test_store_persists_across_reload(sandbox, monkeypatch):
    headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    monkeypatch.setattr(novelty, '_store', None)  # force reload from disk
    n = headline_novelty('BTC/USD',
                         'Bitcoin surges past 100k on ETF inflows')
    assert n < 0.01
