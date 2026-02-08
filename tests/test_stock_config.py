"""Tests for stock_config.py — symbol universe management."""

import json
import pytest

from stock_config import _clean, load_stock_universe, _DEFAULTS


class TestClean:
    def test_deduplicates(self):
        assert _clean(["TSLA", "tsla", "TSLA"]) == ["TSLA"]

    def test_stocks_before_crypto(self):
        result = _clean(["BTC/USD", "AAPL", "ETH/USD", "MSFT"])
        assert result == ["AAPL", "MSFT", "BTC/USD", "ETH/USD"]

    def test_strips_whitespace(self):
        result = _clean(["  AAPL ", " MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_uppercases(self):
        result = _clean(["aapl", "btc/usd"])
        assert result == ["AAPL", "BTC/USD"]

    def test_empty_strings_filtered(self):
        result = _clean(["AAPL", "", "  ", "MSFT"])
        assert result == ["AAPL", "MSFT"]

    def test_alphabetical_within_groups(self):
        result = _clean(["MSFT", "AAPL", "GOOG"])
        assert result == ["AAPL", "GOOG", "MSFT"]


class TestLoadStockUniverse:
    def test_returns_list(self):
        result = load_stock_universe()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_defaults_have_stocks_and_crypto(self):
        stocks = [s for s in _DEFAULTS if "/" not in s]
        crypto = [s for s in _DEFAULTS if "/" in s]
        assert len(stocks) > 0
        assert len(crypto) > 0
