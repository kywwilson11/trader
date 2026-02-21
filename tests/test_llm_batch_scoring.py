"""Tests for _llm_score_batch() — tiered scoring with gap-fill and model tagging."""

import time
from unittest.mock import patch, MagicMock

import pytest

from sentiment import _llm_score_batch, _get_scoring_tiers


class TestLlmScoreBatch:
    """Tests for _llm_score_batch() — tiered Gemini scoring with KW gap-fill."""

    def test_empty_returns_none(self):
        assert _llm_score_batch([]) is None

    @patch("sentiment._fetch_full_texts")
    @patch("sentiment._llm_score_chunk")
    @patch("llm_client.get_budget", return_value=(50, 80))
    def test_all_scored_returns_scores(self, mock_budget, mock_chunk, mock_texts):
        """When all tiers succeed, should return all scores."""
        n = 10
        mock_texts.return_value = [None] * n
        mock_chunk.side_effect = lambda arts, texts, model=None: [0.5] * len(arts)
        articles = [{"headline": f"Article {i} here"} for i in range(n)]
        scores = _llm_score_batch(articles)
        assert scores is not None
        assert len(scores) == n

    @patch("sentiment._fetch_full_texts")
    @patch("sentiment._llm_score_chunk", return_value=None)
    @patch("llm_client.get_budget", return_value=(0, 80))
    def test_all_models_exhausted_returns_none(self, mock_budget, mock_chunk, mock_texts):
        """When all model budgets are 0 and chunks fail, should return None."""
        n = 10
        mock_texts.return_value = [None] * n
        articles = [{"headline": f"Article {i} here"} for i in range(n)]
        scores = _llm_score_batch(articles)
        assert scores is None

    @patch("sentiment._fetch_full_texts")
    @patch("sentiment._llm_score_chunk")
    @patch("llm_client.get_budget", return_value=(50, 80))
    def test_gap_fill_uses_keyword(self, mock_budget, mock_chunk, mock_texts):
        """Articles with None scores should fall back to keyword scoring."""
        # Use 10 articles so tiered chunks are large enough to have gaps
        n = 10
        mock_texts.return_value = [None] * n
        # Return scores with one None gap per chunk
        def chunk_with_gap(chunk_arts, chunk_texts, model=None):
            return [0.5 if i == 0 else None for i in range(len(chunk_arts))]
        mock_chunk.side_effect = chunk_with_gap
        articles = [{"headline": f"Financial headline number {i}"} for i in range(n)]
        scores = _llm_score_batch(articles)
        assert scores is not None
        assert len(scores) == n
        # At least one article should be gap-filled by KW
        kw_articles = [a for a in articles if a.get('_scored_by_model') == 'KW']
        assert len(kw_articles) > 0

    @patch("sentiment._fetch_full_texts")
    @patch("sentiment._llm_score_chunk")
    @patch("llm_client.get_budget", return_value=(50, 80))
    def test_model_tagging(self, mock_budget, mock_chunk, mock_texts):
        """Articles scored by LLM should be tagged with the model name."""
        n = 10
        mock_texts.return_value = [None] * n
        mock_chunk.side_effect = lambda arts, texts, model=None: [0.3] * len(arts)
        articles = [{"headline": f"Financial headline {i}"} for i in range(n)]
        _llm_score_batch(articles)
        # All articles should have a model tag
        for a in articles:
            tag = a.get('_scored_by_model', '')
            gemini_models = [m for m, _ in _get_scoring_tiers()]
            assert tag in gemini_models or tag == 'KW'

    @patch("sentiment._fetch_full_texts")
    @patch("sentiment._llm_score_chunk")
    @patch("llm_client.get_budget", return_value=(50, 80))
    def test_zero_score_preserved_not_gap_filled(self, mock_budget, mock_chunk, mock_texts):
        """A genuine 0.0 LLM score should NOT be treated as a gap."""
        n = 10
        mock_texts.return_value = [None] * n
        mock_chunk.side_effect = lambda arts, texts, model=None: [0.0] * len(arts)
        articles = [{"headline": f"Neutral headline {i}"} for i in range(n)]
        scores = _llm_score_batch(articles)
        assert scores is not None
        # All should be 0.0 (not gap-filled)
        assert all(s == 0.0 for s in scores)
        # None should be tagged KW
        assert all(a.get('_scored_by_model') != 'KW' for a in articles)


class TestScoringTiers:
    """Tests for _get_scoring_tiers() configuration."""

    def test_tiers_sum_to_one(self):
        """The last tier should cover 100% of articles."""
        tiers = _get_scoring_tiers()
        assert tiers[-1][1] == 1.0

    def test_tiers_ascending(self):
        """Cumulative fractions should be monotonically increasing."""
        fracs = [f for _, f in _get_scoring_tiers()]
        for i in range(1, len(fracs)):
            assert fracs[i] > fracs[i - 1]
