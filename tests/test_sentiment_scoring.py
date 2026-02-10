"""Tests for LLM/KW article scoring — score_article_batch, try_llm_upgrade, _llm_score_chunk."""

import json
from unittest.mock import patch, MagicMock

import pytest

from sentiment import score_article_batch, try_llm_upgrade, _llm_score_chunk


class TestScoreArticleBatch:
    """Tests for score_article_batch() — public API for article scoring."""

    def test_empty_returns_empty_kw(self):
        scores, method = score_article_batch([])
        assert scores == []
        assert method == "KW"

    @patch("sentiment._llm_score_batch", return_value=[0.5, -0.3])
    def test_llm_path_returns_llm(self, mock_llm):
        articles = [
            {"headline": "Bitcoin surges higher today"},
            {"headline": "Crypto crash wipes billions"},
        ]
        scores, method = score_article_batch(articles)
        assert method == "LLM"
        assert len(scores) == 2

    @patch("sentiment._llm_score_batch", return_value=None)
    @patch("sentiment._fetch_article_text", return_value=None)
    def test_keyword_fallback_returns_kw(self, mock_fetch, mock_llm):
        articles = [
            {"headline": "Bitcoin rallies strongly today"},
        ]
        scores, method = score_article_batch(articles)
        assert method == "KW"
        assert len(scores) == 1
        assert scores[0] > 0  # "rallies" is positive


class TestTryLlmUpgrade:
    """Tests for try_llm_upgrade() — attempt LLM re-scoring."""

    def test_empty_returns_none(self):
        result = try_llm_upgrade([])
        assert result is None

    @patch("sentiment._llm_score_batch", return_value=[0.4])
    def test_delegates_to_llm_score_batch(self, mock_llm):
        articles = [{"headline": "Test headline here"}]
        result = try_llm_upgrade(articles)
        assert result == [0.4]
        mock_llm.assert_called_once_with(articles)

    @patch("sentiment._llm_score_batch", return_value=None)
    def test_returns_none_on_failure(self, mock_llm):
        articles = [{"headline": "Test headline here"}]
        result = try_llm_upgrade(articles)
        assert result is None


class TestLlmScoreChunk:
    """Tests for _llm_score_chunk() — single-chunk LLM scoring."""

    @patch("llm_client.call_llm", return_value='{"1": 0.5, "2": -0.3}')
    def test_json_parsing(self, mock_llm):
        articles = [
            {"headline": "Good news for stock"},
            {"headline": "Bad news for crypto"},
        ]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores == [0.5, -0.3]

    @patch("llm_client.call_llm", return_value='```json\n{"1": 0.7, "2": -0.1}\n```')
    def test_markdown_fence_stripping(self, mock_llm):
        articles = [
            {"headline": "Headline one here"},
            {"headline": "Headline two here"},
        ]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores == [0.7, -0.1]

    @patch("llm_client.call_llm", return_value=None)
    def test_none_on_llm_failure(self, mock_llm):
        articles = [{"headline": "Test headline"}]
        scores = _llm_score_chunk(articles, [None])
        assert scores is None

    @patch("llm_client.call_llm", return_value='not valid json at all')
    def test_none_on_bad_json(self, mock_llm):
        articles = [{"headline": "Test headline"}]
        scores = _llm_score_chunk(articles, [None])
        assert scores is None

    @patch("llm_client.call_llm", return_value='{"1": 0.5}')
    def test_missing_articles_fails_chunk(self, mock_llm):
        """If LLM only scores 1 of 4 articles (<50%), chunk should fail."""
        articles = [
            {"headline": f"Article {i}"} for i in range(4)
        ]
        scores = _llm_score_chunk(articles, [None] * 4)
        assert scores is None

    @patch("llm_client.call_llm", return_value='{"1": 2.0, "2": -3.0}')
    def test_scores_clamped(self, mock_llm):
        articles = [
            {"headline": "Extreme bullish article"},
            {"headline": "Extreme bearish article"},
        ]
        scores = _llm_score_chunk(articles, [None, None])
        assert scores[0] == 1.0  # clamped from 2.0
        assert scores[1] == -1.0  # clamped from -3.0
