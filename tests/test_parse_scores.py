"""Tests for _parse_scores() and _parse_llm_json() — LLM response parsing edge cases."""

import json
from unittest.mock import patch

import pytest

from sentiment import _parse_llm_json, _llm_score_chunk


class TestParseLlmJson:
    """Tests for _parse_llm_json() — robust JSON parsing from LLM output."""

    def test_valid_json(self):
        result = _parse_llm_json('{"1": 0.5, "2": -0.3}')
        assert result == {"1": 0.5, "2": -0.3}

    def test_markdown_code_fence(self):
        result = _parse_llm_json('```json\n{"1": 0.7}\n```')
        assert result == {"1": 0.7}

    def test_single_quotes(self):
        result = _parse_llm_json("{'1': 0.5, '2': -0.3}")
        assert result == {"1": 0.5, "2": -0.3}

    def test_trailing_text(self):
        result = _parse_llm_json('{"1": 0.5} Here is my explanation...')
        assert result == {"1": 0.5}

    def test_leading_text(self):
        result = _parse_llm_json('Here are the scores: {"1": 0.5}')
        assert result == {"1": 0.5}

    def test_empty_string_returns_none(self):
        assert _parse_llm_json("") is None

    def test_no_braces_returns_none(self):
        assert _parse_llm_json("no json here") is None

    def test_too_short_returns_none(self):
        assert _parse_llm_json("abc") is None

    def test_non_dict_returns_none(self):
        assert _parse_llm_json("[1, 2, 3]") is None

    def test_nested_braces(self):
        # JSON with nested structure — should still find outer dict
        result = _parse_llm_json('{"1": 0.5, "meta": {"model": "test"}}')
        assert result is not None
        assert result["1"] == 0.5


class TestParseScoresViaChunk:
    """Test _parse_scores behavior through _llm_score_chunk (since _parse_scores is private)."""

    @patch("llm_client.call_llm", return_value='{"1": 0.5, "2": -0.3}')
    def test_normal_scores(self, mock_llm):
        articles = [{"headline": "Article one"}, {"headline": "Article two"}]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores == [0.5, -0.3]

    @patch("llm_client.call_llm", return_value='{"1": "not_a_number", "2": -0.3}')
    def test_malformed_value_becomes_none(self, mock_llm):
        """Non-numeric LLM output for a score should produce None (not crash)."""
        articles = [{"headline": "Article one"}, {"headline": "Article two"}]
        scores = _llm_score_chunk(articles, [None, None])
        # One valid + one malformed = 50% match rate, so scores should be returned
        assert scores is not None
        assert scores[0] is None  # malformed → None
        assert scores[1] == -0.3

    @patch("llm_client.call_llm", return_value='{"1": null, "2": 0.3}')
    def test_json_null_value_becomes_none(self, mock_llm):
        """JSON null for a score should produce None (not crash)."""
        articles = [{"headline": "Article one"}, {"headline": "Article two"}]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores is not None
        assert scores[0] is None  # null → None via float(None) → TypeError
        assert scores[1] == 0.3

    @patch("llm_client.call_llm", return_value='{"1": 0.5}')
    def test_below_50_pct_threshold_returns_none(self, mock_llm):
        """If LLM scores less than 50% of articles, chunk fails."""
        articles = [{"headline": f"Article {i}"} for i in range(4)]
        scores = _llm_score_chunk(articles, [None] * 4)
        assert scores is None

    @patch("llm_client.call_llm", return_value='{"1": 0.5, "2": 0.3}')
    def test_at_50_pct_threshold_returns_scores(self, mock_llm):
        """If LLM scores exactly 50% of articles, scores should be returned."""
        articles = [{"headline": f"Article {i}"} for i in range(4)]
        scores = _llm_score_chunk(articles, [None] * 4)
        # 2 of 4 = 50%, should pass
        assert scores is not None
        assert scores[0] == 0.5
        assert scores[1] == 0.3
        assert scores[2] is None
        assert scores[3] is None

    @patch("llm_client.call_llm", return_value='{"1": 5.0, "2": -5.0}')
    def test_scores_clamped_to_range(self, mock_llm):
        """Scores outside [-1, 1] should be clamped."""
        articles = [{"headline": "Bull"}, {"headline": "Bear"}]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores[0] == 1.0
        assert scores[1] == -1.0

    @patch("llm_client.call_llm", return_value='{"1": 0.0}')
    def test_zero_score_is_valid(self, mock_llm):
        """A genuine 0.0 score should be preserved (not treated as missing)."""
        articles = [{"headline": "Neutral article"}]
        scores = _llm_score_chunk(articles, [None])
        assert scores is not None
        assert scores[0] == 0.0

    @patch("llm_client.call_llm", return_value='{"1": 0.5, "2": -0.3}')
    def test_specific_model_uses_call_gemini(self, mock_llm):
        """When model is specified, should use call_gemini instead of call_llm."""
        articles = [{"headline": "Article one"}, {"headline": "Article two"}]
        with patch("llm_client.call_gemini", return_value='{"1": 0.7, "2": -0.1}') as mock_gem:
            scores = _llm_score_chunk(articles, [None, None], model="gemini-2.5-pro")
            mock_gem.assert_called_once()
            mock_llm.assert_not_called()
