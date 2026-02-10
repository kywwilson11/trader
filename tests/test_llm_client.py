"""Tests for llm_client.py — Gemini-only LLM client with quota tracking."""

import json
import pytest
from unittest.mock import patch, MagicMock

from llm_client import (
    call_llm, call_gemini, _call_gemini,
    get_budget, record_call, _model_calls, _maybe_reset_quota,
    GEMINI_MODELS,
)


class TestCallLLM:
    def test_returns_none_when_disabled(self):
        config = {"enabled": False, "models": {}}
        with patch("llm_client.load_llm_config", return_value=config):
            assert call_llm("test prompt") is None

    def test_returns_none_when_no_api_key(self):
        config = {
            "enabled": True,
            "models": {"gemini": {"api_key": "", "model": "gemini-2.5-flash"}},
            "max_llm_latency_sec": 5,
        }
        with patch("llm_client.load_llm_config", return_value=config):
            assert call_llm("test prompt") is None


class TestCallGemini:
    def test_returns_none_when_disabled(self):
        config = {"enabled": False, "models": {}}
        with patch("llm_client.load_llm_config", return_value=config):
            assert call_gemini("test prompt") is None

    def test_returns_none_when_no_api_key(self):
        config = {
            "enabled": True,
            "models": {"gemini": {"api_key": "", "model": "gemini-2.5-flash"}},
            "max_llm_latency_sec": 5,
        }
        with patch("llm_client.load_llm_config", return_value=config):
            assert call_gemini("test prompt") is None

    def test_parses_response(self):
        fake_response = json.dumps({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello world"}]
                }
            }]
        }).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _call_gemini("prompt", "", "fake-key", "model", 100, 10)

        assert result == "Hello world"


class TestQuotaTracking:
    def setup_method(self):
        _model_calls.clear()

    def test_get_budget_returns_full_budget(self):
        remaining, total = get_budget("gemini-2.5-flash")
        assert remaining == total

    def test_record_call_decrements_budget(self):
        remaining_before, total = get_budget("gemini-2.5-flash")
        record_call("gemini-2.5-flash")
        remaining_after, _ = get_budget("gemini-2.5-flash")
        assert remaining_after == remaining_before - 1

    def test_gemini_models_list(self):
        assert "gemini-2.5-pro" in GEMINI_MODELS
        assert "gemini-2.5-flash" in GEMINI_MODELS
        assert "gemini-2.5-flash-lite" in GEMINI_MODELS
