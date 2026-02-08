"""Tests for llm_analyst.py — response parsing and prompt building."""

import json
import pytest

from llm_analyst import _parse_response, _build_prompt


class TestParseResponse:
    def test_valid_json(self):
        response = json.dumps({
            "TSLA": {"m": 1.2, "r": "strong momentum"},
            "AAPL": {"m": 0.5, "r": "overvalued"},
        })
        result = _parse_response(response, ["TSLA", "AAPL"])
        assert result["TSLA"]["m"] == 1.2
        assert result["AAPL"]["m"] == 0.5

    def test_clamps_multiplier_high(self):
        response = json.dumps({"TSLA": {"m": 5.0, "r": "very bullish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["m"] == 1.5  # clamped to max

    def test_clamps_multiplier_low(self):
        response = json.dumps({"TSLA": {"m": -2.0, "r": "bearish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["m"] == 0.0  # clamped to min

    def test_markdown_wrapped_json(self):
        response = '```json\n{"TSLA": {"m": 1.0, "r": "ok"}}\n```'
        result = _parse_response(response, ["TSLA"])
        assert "TSLA" in result

    def test_invalid_json_returns_empty(self):
        result = _parse_response("this is not json at all", ["TSLA"])
        assert result == {}

    def test_missing_symbol_skipped(self):
        response = json.dumps({"AAPL": {"m": 1.0, "r": "ok"}})
        result = _parse_response(response, ["TSLA", "AAPL"])
        assert "TSLA" not in result
        assert "AAPL" in result

    def test_crypto_slash_stripped(self):
        # llm_analyst tries both "BTC/USD" and "BTCUSD"
        response = json.dumps({"BTCUSD": {"m": 1.1, "r": "bullish"}})
        result = _parse_response(response, ["BTC/USD"])
        assert "BTC/USD" in result

    def test_invalid_multiplier_defaults_to_1(self):
        response = json.dumps({"TSLA": {"m": "not_a_number", "r": "test"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["m"] == 1.0


class TestBuildPrompt:
    def test_contains_symbol(self):
        candidates = [{"symbol": "TSLA", "bull_pred": 0.5, "bear_pred": -0.2}]
        prompt = _build_prompt(candidates, "stock", 100000, ["AAPL"], 55)
        assert "TSLA" in prompt
        assert "stock" in prompt
        assert "$100,000" in prompt
        assert "AAPL" in prompt
        assert "55" in prompt

    def test_handles_empty_candidates(self):
        prompt = _build_prompt([], "crypto", 0, None, None)
        assert "Trade Candidates" in prompt

    def test_includes_fundamentals(self):
        candidates = [{
            "symbol": "AAPL",
            "bull_pred": 0.3,
            "fundamentals_text": "P/E=25.0, MktCap=$3.0T",
        }]
        prompt = _build_prompt(candidates, "stock", 0, None, None)
        assert "P/E=25.0" in prompt
