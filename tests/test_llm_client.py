"""Tests for llm_client.py — Gemini-only LLM client with quota tracking."""

import io
import json
import time
import urllib.error
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from llm_client import (
    call_llm, call_gemini, _call_gemini, _parse_retry_after,
    _rate_limit_ok, _call_timestamps, _RATE_LIMIT_RPM,
    get_budget, record_call, _model_calls, _maybe_reset_quota,
    _DAILY_BUDGETS, _quota_reset_date,
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

    def test_budget_floors_at_zero(self):
        """Budget should never go negative."""
        model = "gemini-2.5-pro"
        total = _DAILY_BUDGETS[model]
        for _ in range(total + 10):
            record_call(model)
        remaining, _ = get_budget(model)
        assert remaining == 0

    def test_unknown_model_default_budget(self):
        """Unknown models should get a default budget of 50."""
        remaining, total = get_budget("gemini-unknown-model")
        assert total == 50
        assert remaining == 50


class TestParseRetryAfter:
    """Tests for _parse_retry_after() — 429 error body parsing."""

    def _make_http_error(self, body_text):
        """Create a fake HTTPError with readable body."""
        body = io.BytesIO(body_text.encode())
        return urllib.error.HTTPError(
            url="https://example.com", code=429, msg="Too Many Requests",
            hdrs={}, fp=body,
        )

    def test_extracts_integer_delay(self):
        err = self._make_http_error("Please retry in 30s after rate limit")
        assert _parse_retry_after(err) == 30.0

    def test_extracts_float_delay(self):
        err = self._make_http_error("retry in 2.5s")
        assert _parse_retry_after(err) == 2.5

    def test_daily_quota_exhausted_returns_none(self):
        err = self._make_http_error("limit: 0 remaining requests")
        assert _parse_retry_after(err) is None

    def test_no_retry_info_returns_none(self):
        err = self._make_http_error("rate limited, try later")
        assert _parse_retry_after(err) is None

    def test_read_error_returns_none(self):
        mock_err = MagicMock()
        mock_err.read.side_effect = Exception("read failed")
        assert _parse_retry_after(mock_err) is None


class TestRateLimit:
    """Tests for _rate_limit_ok() — sliding window rate limiter."""

    def setup_method(self):
        _call_timestamps.clear()

    def teardown_method(self):
        _call_timestamps.clear()

    def test_allows_first_call(self):
        assert _rate_limit_ok() is True

    def test_blocks_after_limit(self):
        for _ in range(_RATE_LIMIT_RPM):
            assert _rate_limit_ok() is True
        assert _rate_limit_ok() is False

    def test_allows_after_window_expires(self):
        # Fill up the window with old timestamps
        old_time = time.time() - 61  # 61 seconds ago
        for _ in range(_RATE_LIMIT_RPM):
            _call_timestamps.append(old_time)
        # Should allow since old timestamps expired
        assert _rate_limit_ok() is True


class TestTimezone:
    """Tests for _maybe_reset_quota() timezone handling."""

    def setup_method(self):
        _model_calls.clear()

    def test_uses_zoneinfo_not_hardcoded_offset(self):
        """Verify the timezone import uses ZoneInfo for DST correctness."""
        import llm_client
        import inspect
        source = inspect.getsource(llm_client._maybe_reset_quota)
        assert "ZoneInfo" in source
        assert "timedelta(hours=-8)" not in source


class TestCallGeminiResponse:
    """Tests for _call_gemini response parsing edge cases."""

    def test_empty_candidates_returns_none(self):
        fake_response = json.dumps({"candidates": []}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _call_gemini("prompt", "", "key", "model", 100, 10)
        assert result is None

    def test_missing_content_returns_none(self):
        fake_response = json.dumps({
            "candidates": [{"finishReason": "STOP"}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = _call_gemini("prompt", "", "key", "model", 100, 10)
        assert result is None

    def test_system_message_added_to_contents(self):
        """When system is provided, it should appear in the request body."""
        fake_response = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
        }).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
            _call_gemini("hello", "be helpful", "key", "model", 100, 10)
            # Check the request body includes system turn
            call_args = mock_open.call_args
            req = call_args[0][0]
            body = json.loads(req.data.decode())
            # Should have 3 content turns: system user, model ack, actual user
            assert len(body["contents"]) == 3
            assert body["contents"][0]["parts"][0]["text"] == "be helpful"
