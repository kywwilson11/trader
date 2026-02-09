"""Tests for fundamentals.py — formatting and caching."""

import time
import pytest

from fundamentals import format_fundamentals_for_llm, _cache_get, _cache_set, _cache


class TestFormatFundamentalsForLLM:
    def test_full_data(self):
        fund = {
            "pe_ratio": 25.3,
            "pb_ratio": 8.1,
            "market_cap": 3e12,
            "revenue_growth": 0.15,
            "eps": 6.42,
            "dividend_yield": 0.005,
            "sector": "Technology",
            "beta": 1.23,
            "week52_high": 200.0,
            "week52_low": 120.0,
        }
        result = format_fundamentals_for_llm("AAPL", fund)
        assert "P/E=25.3" in result
        assert "P/B=8.1" in result
        assert "$3.0T" in result
        assert "Technology" in result
        assert "Beta=1.23" in result
        assert "$120.00-$200.00" in result

    def test_billion_market_cap(self):
        fund = {"market_cap": 50e9}
        result = format_fundamentals_for_llm("TSLA", fund)
        assert "$50.0B" in result

    def test_million_market_cap(self):
        fund = {"market_cap": 500e6}
        result = format_fundamentals_for_llm("SMALL", fund)
        assert "$500M" in result

    def test_empty_data(self):
        result = format_fundamentals_for_llm("XYZ", {})
        assert "limited data" in result

    def test_insider_activity(self):
        fund = {"pe_ratio": 10.0}
        insider = {"summary": "3 buys, 1 sells (net +5000 shares)"}
        result = format_fundamentals_for_llm("TSLA", fund, insider=insider)
        assert "Insider Activity" in result
        assert "3 buys" in result

    def test_filing_summary(self):
        fund = {"pe_ratio": 10.0}
        result = format_fundamentals_for_llm(
            "TSLA", fund, filing_summary="Revenue grew 20% YoY"
        )
        assert "SEC Filing" in result
        assert "Revenue grew" in result

    def test_dividend_yield_percentage(self):
        fund = {"dividend_yield": 3.5}
        result = format_fundamentals_for_llm("T", fund)
        assert "3.5%" in result

    def test_dividend_yield_fraction(self):
        fund = {"dividend_yield": 0.035}
        result = format_fundamentals_for_llm("T", fund)
        assert "3.50%" in result

    def test_insider_na_skipped(self):
        fund = {"pe_ratio": 10.0}
        insider = {"summary": "N/A"}
        result = format_fundamentals_for_llm("TSLA", fund, insider=insider)
        assert "Insider" not in result

    def test_none_values_skipped(self):
        fund = {
            "pe_ratio": None,
            "pb_ratio": None,
            "market_cap": None,
            "sector": None,
        }
        result = format_fundamentals_for_llm("XYZ", fund)
        assert "P/E" not in result
        assert "limited data" in result

    def test_only_52_week_with_both(self):
        fund = {"week52_high": 150.0, "week52_low": 100.0}
        result = format_fundamentals_for_llm("X", fund)
        assert "52wk" in result

    def test_only_52_week_high_no_low(self):
        fund = {"week52_high": 150.0}
        result = format_fundamentals_for_llm("X", fund)
        assert "52wk" not in result


class TestCache:
    def setup_method(self):
        """Clear cache before each test."""
        _cache.clear()

    def test_cache_set_and_get(self):
        _cache_set("test_key", {"value": 42})
        result = _cache_get("test_key", ttl=300)
        assert result == {"value": 42}

    def test_cache_miss(self):
        assert _cache_get("nonexistent", ttl=300) is None

    def test_cache_expired(self):
        _cache["expired_key"] = (time.time() - 1000, "old_data")
        result = _cache_get("expired_key", ttl=500)
        assert result is None

    def test_cache_fresh(self):
        _cache["fresh_key"] = (time.time() - 10, "fresh_data")
        result = _cache_get("fresh_key", ttl=300)
        assert result == "fresh_data"

    def test_cache_overwrite(self):
        _cache_set("key", "first")
        _cache_set("key", "second")
        assert _cache_get("key", ttl=300) == "second"

    def test_cache_stores_none(self):
        _cache_set("none_key", None)
        # None is a valid cached value — should still return it
        # Actually _cache_get returns None for missing too, so check cache directly
        assert "none_key" in _cache
