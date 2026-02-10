"""Tests for sentiment.py — keyword scoring, text validation, article dedup, and LLM retry queue."""

import time
from unittest.mock import patch

import pytest

from sentiment import (
    _score_text, _validate_text, _deduplicate_articles,
    _score_articles, _aggregate_scores, _try_llm_retry,
    _llm_retry_queue, _cache, CACHE_TTL,
)


class TestScoreText:
    def test_positive_headline(self):
        assert _score_text("Bitcoin surges to all time high") > 0

    def test_negative_headline(self):
        assert _score_text("Crypto crash wipes out billions") < 0

    def test_neutral_returns_near_zero(self):
        score = _score_text("The weather is nice today")
        assert abs(score) < 0.3

    def test_negation_flips_sentiment(self):
        pos = _score_text("Bitcoin rallies strongly")
        neg = _score_text("Bitcoin not rallying strongly")
        assert pos > neg

    def test_phrase_matching(self):
        assert _score_text("Stock beats expectations with record earnings") > 0
        assert _score_text("Revenue miss disappoints investors") < 0

    def test_output_bounded(self):
        # Even extreme text should be in (-1, 1) due to tanh
        score = _score_text("surge rally moon bull profit gain soar jump")
        assert -1 < score < 1

    def test_empty_string(self):
        score = _score_text("")
        assert abs(score) < 0.01

    def test_death_cross_negative(self):
        assert _score_text("Analysts warn of death cross pattern forming") < 0

    def test_rate_cut_positive(self):
        assert _score_text("Fed cuts interest rates by 50 basis points") > 0

    def test_mixed_sentiment(self):
        # Both positive and negative signals — should not saturate
        score = _score_text("Stock surges on earnings beat but faces regulatory risk")
        assert -1 < score < 1


class TestValidateText:
    def test_valid_headline(self):
        assert _validate_text("Bitcoin rises 5% on ETF approval") is not None

    def test_none_input(self):
        assert _validate_text(None) is None

    def test_empty_string(self):
        assert _validate_text("") is None

    def test_too_short(self):
        assert _validate_text("Hi") is None

    def test_url_rejected(self):
        assert _validate_text("https://example.com/article") is None

    def test_html_stripped(self):
        result = _validate_text("<p>Bitcoin rises on <b>ETF</b> approval news</p>")
        assert result is not None
        assert "<p>" not in result

    def test_non_ascii_rejected(self):
        # More than 50% non-ASCII
        assert _validate_text("\u4e2d\u6587\u6d4b\u8bd5\u4e2d\u6587\u6d4b\u8bd5\u4e2d\u6587") is None

    def test_whitespace_only(self):
        assert _validate_text("          ") is None

    def test_non_string_input(self):
        assert _validate_text(12345) is None

    def test_exactly_10_chars_valid(self):
        result = _validate_text("1234567890")
        assert result is not None

    def test_9_chars_too_short(self):
        assert _validate_text("123456789") is None


class TestDeduplicateArticles:
    def test_removes_duplicates(self):
        articles = [
            {"headline": "Bitcoin surges today"},
            {"headline": "Bitcoin surges today"},
            {"headline": "Ethereum drops sharply"},
        ]
        result = _deduplicate_articles(articles)
        assert len(result) == 2

    def test_case_insensitive(self):
        articles = [
            {"headline": "Bitcoin SURGES Today"},
            {"headline": "bitcoin surges today"},
        ]
        result = _deduplicate_articles(articles)
        assert len(result) == 1

    def test_preserves_order(self):
        articles = [
            {"headline": "First headline here"},
            {"headline": "Second headline here"},
            {"headline": "First headline here"},
        ]
        result = _deduplicate_articles(articles)
        assert result[0]["headline"] == "First headline here"
        assert result[1]["headline"] == "Second headline here"

    def test_empty_list(self):
        assert _deduplicate_articles([]) == []

    def test_none_headline_skipped(self):
        articles = [
            {"headline": None},
            {"headline": "Valid headline text"},
        ]
        result = _deduplicate_articles(articles)
        assert len(result) == 1
        assert result[0]["headline"] == "Valid headline text"

    def test_empty_headline_skipped(self):
        articles = [
            {"headline": ""},
            {"headline": "   "},
            {"headline": "Actual headline here"},
        ]
        result = _deduplicate_articles(articles)
        assert len(result) == 1

    def test_missing_headline_key(self):
        articles = [
            {"summary": "no headline key"},
            {"headline": "Has a headline here"},
        ]
        result = _deduplicate_articles(articles)
        assert len(result) == 1


class TestScoreArticlesReturnType:
    """Test that _score_articles returns (result_dict, used_llm) tuple."""

    @patch("sentiment._llm_score_batch", return_value=None)
    def test_keyword_fallback_sets_used_llm_false(self, mock_llm):
        articles = [{"headline": "Bitcoin surges on bullish momentum"}]
        result, used_llm = _score_articles(articles)
        assert used_llm is False
        assert isinstance(result, dict)
        assert "sentiment_score" in result

    @patch("sentiment._llm_score_batch", return_value=[0.5])
    def test_llm_success_sets_used_llm_true(self, mock_llm):
        articles = [{"headline": "Bitcoin surges on bullish momentum"}]
        result, used_llm = _score_articles(articles)
        assert used_llm is True
        assert isinstance(result, dict)
        assert result["sentiment_score"] == pytest.approx(0.5)

    def test_empty_articles_returns_used_llm_true(self):
        result, used_llm = _score_articles([])
        assert used_llm is True
        assert result["article_count"] == 0


class TestAggregateScores:
    def test_empty_scores(self):
        result = _aggregate_scores([])
        assert result["sentiment_score"] == 0.0
        assert result["article_count"] == 0

    def test_positive_scores(self):
        result = _aggregate_scores([0.5, 0.3, 0.7])
        assert result["sentiment_score"] == pytest.approx(0.5)
        assert result["article_count"] == 3
        assert result["positive_ratio"] == pytest.approx(1.0)
        assert result["negative_ratio"] == pytest.approx(0.0)

    def test_mixed_scores(self):
        result = _aggregate_scores([0.5, -0.5])
        assert result["sentiment_score"] == pytest.approx(0.0)
        assert result["positive_ratio"] == pytest.approx(0.5)
        assert result["negative_ratio"] == pytest.approx(0.5)


class TestTryLlmRetry:
    """Test _try_llm_retry queue draining behavior."""

    def setup_method(self):
        """Clear queue and cache before each test."""
        _llm_retry_queue.clear()
        # Save and restore cache keys we touch
        self._saved_cache_keys = []

    def teardown_method(self):
        _llm_retry_queue.clear()
        for key in self._saved_cache_keys:
            _cache.pop(key, None)

    def test_empty_queue_is_noop(self):
        _try_llm_retry()  # should not raise

    def test_discards_stale_entry(self):
        stale_time = time.time() - CACHE_TTL - 10
        _llm_retry_queue.append(("test_stale", [], stale_time))
        _try_llm_retry()
        assert len(_llm_retry_queue) == 0  # consumed and discarded

    def test_discards_superseded_entry(self):
        queued_at = time.time() - 60
        cache_key = "_test_superseded"
        self._saved_cache_keys.append(cache_key)
        # Simulate a newer cache update after queuing
        _cache[cache_key] = (queued_at + 30, {"sentiment_score": 0.1})
        _llm_retry_queue.append((cache_key, [], queued_at))
        _try_llm_retry()
        assert len(_llm_retry_queue) == 0  # consumed and discarded

    @patch("sentiment._llm_score_batch", return_value=None)
    def test_pushes_back_on_failure(self, mock_llm):
        now = time.time()
        articles = [{"headline": "Test article headline here"}]
        _llm_retry_queue.append(("_test_pushback", articles, now))
        self._saved_cache_keys.append("_test_pushback")
        _try_llm_retry()
        assert len(_llm_retry_queue) == 1
        assert _llm_retry_queue[0][0] == "_test_pushback"

    @patch("sentiment._llm_score_batch", return_value=[0.6])
    def test_updates_cache_on_success(self, mock_llm):
        now = time.time()
        cache_key = "_test_upgrade"
        self._saved_cache_keys.append(cache_key)
        articles = [{"headline": "Bitcoin surges higher today"}]
        _llm_retry_queue.append((cache_key, articles, now))
        _try_llm_retry()
        assert len(_llm_retry_queue) == 0
        assert cache_key in _cache
        _, result = _cache[cache_key]
        assert result["sentiment_score"] == pytest.approx(0.6)

    def test_queue_bounded(self):
        for i in range(60):
            _llm_retry_queue.append((f"key_{i}", [], time.time()))
        assert len(_llm_retry_queue) == 50  # maxlen enforced
