"""Tests for sentiment.py — keyword scoring, text validation, and article dedup."""

import pytest

from sentiment import _score_text, _validate_text, _deduplicate_articles


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
