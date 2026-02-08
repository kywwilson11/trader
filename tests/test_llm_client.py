"""Tests for llm_client.py — LLM dispatch and response parsing."""

import json
import pytest
from unittest.mock import patch, MagicMock

from llm_client import call_llm, _dispatch, _call_gemini


class TestCallLLM:
    def test_returns_none_when_disabled(self):
        config = {"enabled": False, "provider": "gemini", "models": {}}
        with patch("llm_client.load_llm_config", return_value=config):
            assert call_llm("test prompt") is None

    def test_returns_none_when_no_api_key(self):
        config = {
            "enabled": True,
            "provider": "gemini",
            "models": {"gemini": {"api_key": "", "model": "gemini-2.5-flash-lite"}},
            "max_llm_latency_sec": 5,
        }
        with patch("llm_client.load_llm_config", return_value=config):
            # No gemini key for fallback either
            assert call_llm("test prompt") is None


class TestDispatch:
    def test_routes_to_gemini(self):
        with patch("llm_client._call_gemini", return_value="ok") as mock:
            result = _dispatch("gemini", "p", "s", "key", "model", 100, 10)
        mock.assert_called_once_with("p", "s", "key", "model", 100, 10)
        assert result == "ok"

    def test_routes_to_claude(self):
        with patch("llm_client._call_claude", return_value="ok") as mock:
            result = _dispatch("claude", "p", "s", "key", "model", 100, 10)
        mock.assert_called_once()
        assert result == "ok"

    def test_routes_to_openai(self):
        with patch("llm_client._call_openai", return_value="ok") as mock:
            result = _dispatch("openai", "p", "s", "key", "model", 100, 10)
        mock.assert_called_once()
        assert result == "ok"

    def test_unknown_provider_returns_none(self):
        assert _dispatch("unknown", "p", "s", "key", "m", 100, 10) is None


class TestCallGemini:
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
