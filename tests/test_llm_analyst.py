"""Tests for llm_analyst.py — response parsing and prompt building."""

import json
import pytest

from llm_analyst import _parse_response, _build_prompt


class TestParseResponse:
    """Test the new {s, bull, bear, r} format and legacy {m, r} compat."""

    def test_new_format_score(self):
        response = json.dumps({
            "TSLA": {"bull": "Strong momentum.", "bear": "No risks.", "s": 0.72, "r": "Positive setup."},
            "AAPL": {"bull": "Earnings beat.", "bear": "High valuation.", "s": 0.45, "r": "Mixed."},
        })
        result = _parse_response(response, ["TSLA", "AAPL"])
        assert result["TSLA"]["s"] == 0.72
        assert result["AAPL"]["s"] == 0.45
        # m should be derived from s (s * 1.5)
        assert abs(result["TSLA"]["m"] - 1.08) < 0.01
        assert abs(result["AAPL"]["m"] - 0.675) < 0.01

    def test_clamps_score_high(self):
        response = json.dumps({"TSLA": {"s": 5.0, "r": "very bullish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["s"] == 1.0  # clamped to max

    def test_clamps_score_low(self):
        response = json.dumps({"TSLA": {"s": -2.0, "r": "bearish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["s"] == 0.0  # clamped to min

    def test_catastrophic_score_detection(self):
        response = json.dumps({"BTC/USD": {"s": 0.08, "bull": "", "bear": "Exchange hacked.", "r": "VETO"}})
        result = _parse_response(response, ["BTC/USD"])
        assert result["BTC/USD"]["s"] < 0.15  # catastrophic threshold

    def test_bull_bear_fields_extracted(self):
        response = json.dumps({
            "ETH/USD": {"bull": "DeFi growing.", "bear": "Regulatory risk.", "s": 0.55, "r": "Neutral."}
        })
        result = _parse_response(response, ["ETH/USD"])
        assert result["ETH/USD"]["bull"] == "DeFi growing."
        assert result["ETH/USD"]["bear"] == "Regulatory risk."

    def test_default_score_when_missing(self):
        response = json.dumps({"TSLA": {"r": "no score field"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["s"] == 0.5  # default neutral

    def test_legacy_format_backward_compat(self):
        """Old format with 'm' field should still work."""
        response = json.dumps({
            "TSLA": {"m": 1.2, "r": "strong momentum"},
            "AAPL": {"m": 0.5, "r": "overvalued"},
        })
        result = _parse_response(response, ["TSLA", "AAPL"])
        assert result["TSLA"]["m"] == 1.2
        assert result["AAPL"]["m"] == 0.5
        # s should be derived from m (m / 1.5)
        assert abs(result["TSLA"]["s"] - 0.8) < 0.01
        assert abs(result["AAPL"]["s"] - 0.3333) < 0.01

    def test_clamps_legacy_multiplier_high(self):
        response = json.dumps({"TSLA": {"m": 5.0, "r": "very bullish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["m"] == 1.5  # clamped to max

    def test_clamps_legacy_multiplier_low(self):
        response = json.dumps({"TSLA": {"m": -2.0, "r": "bearish"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["m"] == 0.0  # clamped to min

    def test_markdown_wrapped_json(self):
        response = '```json\n{"TSLA": {"s": 0.65, "r": "ok"}}\n```'
        result = _parse_response(response, ["TSLA"])
        assert "TSLA" in result

    def test_invalid_json_returns_empty(self):
        result = _parse_response("this is not json at all", ["TSLA"])
        assert result == {}

    def test_missing_symbol_skipped(self):
        response = json.dumps({"AAPL": {"s": 0.6, "r": "ok"}})
        result = _parse_response(response, ["TSLA", "AAPL"])
        assert "TSLA" not in result
        assert "AAPL" in result

    def test_crypto_slash_stripped(self):
        # llm_analyst tries both "BTC/USD" and "BTCUSD"
        response = json.dumps({"BTCUSD": {"s": 0.7, "r": "bullish"}})
        result = _parse_response(response, ["BTC/USD"])
        assert "BTC/USD" in result

    def test_invalid_score_defaults_to_neutral(self):
        response = json.dumps({"TSLA": {"s": "not_a_number", "r": "test"}})
        result = _parse_response(response, ["TSLA"])
        assert result["TSLA"]["s"] == 0.5


class TestBuildPrompt:
    def test_contains_symbol(self):
        candidates = [{"symbol": "TSLA", "pred_return": 0.5}]
        prompt = _build_prompt(candidates, "stock", 100000, ["AAPL"], 55, {})
        assert "TSLA" in prompt
        assert "stock" in prompt
        assert "$100,000" in prompt
        assert "AAPL" in prompt
        assert "55" in prompt

    def test_handles_empty_candidates(self):
        prompt = _build_prompt([], "crypto", 0, None, None, {})
        assert "crypto" in prompt

    def test_includes_fundamentals(self):
        candidates = [{
            "symbol": "AAPL",
            "pred_return": 0.3,
            "fundamentals_text": "P/E=25.0, MktCap=$3.0T",
        }]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {})
        assert "P/E=25.0" in prompt

    def test_no_technical_indicators(self):
        """Prompt should NOT contain technical indicator references."""
        candidates = [{
            "symbol": "TSLA",
            "pred_return": 0.5,
        }]
        prompt = _build_prompt(candidates, "stock", 100000, [], 50, {})
        # These should NOT appear in the prompt (they're technical indicators)
        assert "RSI" not in prompt
        assert "MACD" not in prompt
        assert "Stoch" not in prompt
        assert "Bollinger" not in prompt
        assert "SMA20" not in prompt

    def test_includes_ml_prediction(self):
        """Prompt should include the ML model's prediction."""
        candidates = [{"symbol": "BTC/USD", "pred_return": 0.35}]
        prompt = _build_prompt(candidates, "crypto", 50000, None, 45, {})
        assert "+0.35" in prompt
        assert "bullish" in prompt.lower()

    def test_includes_news_headlines(self):
        candidates = [{
            "symbol": "ETH/USD",
            "pred_return": 0.1,
            "news_headlines": ["Ethereum 2.0 upgrade complete", "DeFi TVL hits new high"],
        }]
        prompt = _build_prompt(candidates, "crypto", 0, None, None, {})
        assert "Ethereum 2.0 upgrade complete" in prompt
        assert "DeFi TVL hits new high" in prompt

    def test_includes_fng_regime(self):
        candidates = [{"symbol": "BTC/USD", "pred_return": 0.1}]
        prompt = _build_prompt(candidates, "crypto", 0, None, 15, {})
        assert "Fear" in prompt
        assert "15" in prompt

    def test_output_format_instructions(self):
        candidates = [{"symbol": "TSLA", "pred_return": 0.2}]
        prompt = _build_prompt(candidates, "stock", 0, None, None, {})
        assert "bull" in prompt
        assert "bear" in prompt
        assert "s (" in prompt or '"s"' in prompt
