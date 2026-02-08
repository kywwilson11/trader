"""Tests for fundamentals.py — format_fundamentals_for_llm (pure formatting)."""

import pytest

from fundamentals import format_fundamentals_for_llm


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
        # When yfinance returns >1, it's already a percentage
        fund = {"dividend_yield": 3.5}
        result = format_fundamentals_for_llm("T", fund)
        assert "3.5%" in result

    def test_dividend_yield_fraction(self):
        # When yfinance returns <1, it's a fraction
        fund = {"dividend_yield": 0.035}
        result = format_fundamentals_for_llm("T", fund)
        assert "3.50%" in result
