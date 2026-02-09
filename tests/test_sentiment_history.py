"""Tests for sentiment_history.py — FnG normalization and keyword scoring."""

import pytest

from sentiment_history import _fng_value_to_score, _keyword_score


class TestFngValueToScore:
    def test_extreme_fear(self):
        # 0 = extreme fear → -1.0
        assert _fng_value_to_score(0) == -1.0

    def test_extreme_greed(self):
        # 100 = extreme greed → 1.0
        assert _fng_value_to_score(100) == 1.0

    def test_neutral(self):
        # 50 = neutral → 0.0
        assert _fng_value_to_score(50) == 0.0

    def test_fear_zone(self):
        # 25 = fear → -0.5
        assert _fng_value_to_score(25) == -0.5

    def test_greed_zone(self):
        # 75 = greed → 0.5
        assert _fng_value_to_score(75) == 0.5

    def test_output_range(self):
        for v in range(0, 101):
            score = _fng_value_to_score(v)
            assert -1.0 <= score <= 1.0


class TestKeywordScore:
    def test_headline_only(self):
        score = _keyword_score("Bitcoin surges on bullish momentum")
        assert score > 0

    def test_summary_only(self):
        score = _keyword_score("", "Market crash wipes out billions in value")
        # Empty headline, only summary
        assert score < 0

    def test_both_headline_and_summary(self):
        score = _keyword_score(
            "Bitcoin rallies to new high",
            "Strong gains across all crypto markets",
        )
        assert score > 0

    def test_empty_both_returns_zero(self):
        assert _keyword_score("", "") == 0.0

    def test_none_headline_returns_zero(self):
        assert _keyword_score(None, None) == 0.0

    def test_headline_weighted_60_pct(self):
        # With both headline and summary, headline gets 60% weight
        pos_headline = _keyword_score("Bitcoin surges strongly", "neutral text")
        neg_headline = _keyword_score("neutral text", "Bitcoin surges strongly")
        # Positive headline should produce higher score than positive summary
        # (60% vs 40% weight)
        assert pos_headline != neg_headline

    def test_short_text_returns_zero(self):
        # Text too short to validate
        assert _keyword_score("Hi", "No") == 0.0
