"""Tests for the EDGAR 8-K item veto + M&A blacklist rules."""

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import edgar_events as ee

TODAY = dt.date(2026, 6, 10)


def test_fresh_8k_bankruptcy_blocks():
    filings = [('8-K', '2026-06-08', '1.03,9.01')]
    blocked, reason = ee._evaluate(filings, TODAY)
    assert blocked and '1.03' in reason


def test_each_veto_item_blocks():
    for code in ('1.03', '2.04', '4.02', '5.02'):
        filings = [('8-K', '2026-06-09', f'{code},9.01')]
        blocked, reason = ee._evaluate(filings, TODAY)
        assert blocked and code in reason, code


def test_stale_8k_does_not_block():
    # Item 5.02 filed 10 days ago — outside the 5-day window
    filings = [('8-K', '2026-05-31', '5.02')]
    assert ee._evaluate(filings, TODAY) == (False, None)


def test_benign_8k_items_pass():
    # 2.02 (results) + 9.01 (exhibits): routine earnings 8-K
    filings = [('8-K', '2026-06-09', '2.02,9.01')]
    assert ee._evaluate(filings, TODAY) == (False, None)


def test_8ka_amendment_also_caught():
    filings = [('8-K/A', '2026-06-09', '4.02')]
    blocked, reason = ee._evaluate(filings, TODAY)
    assert blocked


def test_ma_forms_block_within_90d():
    for form in ('425', 'SC 14D9', 'DEFM14A', 'S-4'):
        filings = [(form, '2026-04-01', '')]  # 70 days ago
        blocked, reason = ee._evaluate(filings, TODAY)
        assert blocked and 'M&A' in reason, form


def test_ancient_ma_form_expires():
    filings = [('425', '2025-12-01', '')]  # >90 days
    assert ee._evaluate(filings, TODAY) == (False, None)


def test_garbage_dates_ignored():
    filings = [('8-K', 'not-a-date', '1.03'), ('425', None, '')]
    assert ee._evaluate(filings, TODAY) == (False, None)


def test_entry_blocked_cached_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, '_CACHE_FILE', tmp_path / 'cache.json')
    monkeypatch.setattr(ee, '_TICKER_MAP_FILE', tmp_path / 'tickers.json')
    calls = []

    def fake_map():
        calls.append('map')
        return {'XYZ': '0000000001'}

    def fake_filings(cik):
        calls.append('filings')
        return [('8-K', dt.date.today().isoformat(), '1.03')]

    monkeypatch.setattr(ee, '_ticker_cik_map', fake_map)
    monkeypatch.setattr(ee, '_recent_filings', fake_filings)
    assert ee.entry_blocked('XYZ')[0] is True
    assert ee.entry_blocked('XYZ')[0] is True  # served from cache
    assert calls.count('filings') == 1


def test_entry_blocked_fails_open(tmp_path, monkeypatch):
    monkeypatch.setattr(ee, '_CACHE_FILE', tmp_path / 'cache.json')

    def boom():
        raise OSError('edgar down')

    monkeypatch.setattr(ee, '_ticker_cik_map', boom)
    assert ee.entry_blocked('NVDA') == (False, None)
