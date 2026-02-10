"""Tests for sentiment trade gating, Fear & Greed endpoints, and CNN FnG."""

import time
from unittest.mock import patch, MagicMock

import pytest

from sentiment import (
    get_fear_greed, get_cnn_fear_greed, sentiment_gate,
    _cache, CACHE_TTL,
)


class TestGetFearGreed:
    """Tests for get_fear_greed() — Crypto Fear & Greed Index."""

    def setup_method(self):
        self._saved = _cache.pop('__fng__', None)

    def teardown_method(self):
        _cache.pop('__fng__', None)
        if self._saved is not None:
            _cache['__fng__'] = self._saved

    @patch("sentiment.requests.get")
    def test_parses_value_and_label(self, mock_get):
        mock_get.return_value.json.return_value = {
            'data': [{'value': '42', 'value_classification': 'Fear'}]
        }
        result = get_fear_greed()
        assert result == {'value': 42, 'label': 'Fear'}

    @patch("sentiment.requests.get")
    def test_cache_hit(self, mock_get):
        mock_get.return_value.json.return_value = {
            'data': [{'value': '50', 'value_classification': 'Neutral'}]
        }
        r1 = get_fear_greed()
        r2 = get_fear_greed()
        assert r1 == r2
        assert mock_get.call_count == 1  # only 1 HTTP call

    @patch("sentiment.requests.get", side_effect=Exception("network error"))
    def test_error_returns_none(self, mock_get):
        result = get_fear_greed()
        assert result is None


class TestGetCnnFearGreed:
    """Tests for get_cnn_fear_greed() — CNN Fear & Greed + VIX."""

    def setup_method(self):
        self._saved = _cache.pop('__cnn_fng__', None)

    def teardown_method(self):
        _cache.pop('__cnn_fng__', None)
        if self._saved is not None:
            _cache['__cnn_fng__'] = self._saved

    @patch("sentiment.requests.get")
    def test_parses_score_and_rating(self, mock_get):
        mock_get.return_value.json.return_value = {
            'fear_and_greed': {
                'score': 65.3,
                'rating': 'greed',
                'previous_close': 62.1,
                'previous_1_week': 58.0,
            },
            'market_volatility_vix': {
                'data': [{'y': 15.5, 'rating': 'low_volatility'}],
            },
        }
        result = get_cnn_fear_greed()
        assert result['score'] == 65.3
        assert result['rating'] == 'Greed'
        assert result['vix'] == 15.5
        assert result['vix_rating'] == 'Low Volatility'

    @patch("sentiment.requests.get")
    def test_missing_vix_data(self, mock_get):
        mock_get.return_value.json.return_value = {
            'fear_and_greed': {
                'score': 50.0,
                'rating': 'neutral',
                'previous_close': 48.0,
                'previous_1_week': 45.0,
            },
            'market_volatility_vix': {},
        }
        result = get_cnn_fear_greed()
        assert result['vix'] == 0.0
        assert result['vix_rating'] == ''

    @patch("sentiment.requests.get", side_effect=Exception("timeout"))
    def test_error_returns_none(self, mock_get):
        result = get_cnn_fear_greed()
        assert result is None


class TestSentimentGate:
    """Tests for sentiment_gate() — trade multiplier computation."""

    @patch("sentiment.get_market_sentiment", return_value=None)
    @patch("sentiment.get_news_sentiment", return_value=None)
    @patch("sentiment.get_fear_greed", return_value=None)
    def test_no_data_returns_1x(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult == pytest.approx(1.0)

    @patch("sentiment.get_market_sentiment", return_value=None)
    @patch("sentiment.get_news_sentiment", return_value=None)
    @patch("sentiment.get_fear_greed", return_value={'value': 10, 'label': 'Extreme Fear'})
    def test_extreme_fear_reduces(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult < 1.0
        assert any('extreme_fear' in r for r in reasons)

    @patch("sentiment.get_market_sentiment", return_value=None)
    @patch("sentiment.get_news_sentiment", return_value={
        'sentiment_score': -0.6, 'article_count': 5,
        'positive_ratio': 0.0, 'negative_ratio': 1.0,
    })
    @patch("sentiment.get_fear_greed", return_value={'value': 10, 'label': 'Extreme Fear'})
    def test_catastrophic_news_floor(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult == pytest.approx(0.15, abs=0.01)

    @patch("sentiment.get_market_sentiment", return_value=None)
    @patch("sentiment.get_news_sentiment", return_value={
        'sentiment_score': 0.5, 'article_count': 10,
        'positive_ratio': 0.8, 'negative_ratio': 0.1,
    })
    @patch("sentiment.get_fear_greed", return_value={'value': 55, 'label': 'Greed'})
    def test_bullish_above_1x(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult > 1.0

    @patch("sentiment.get_market_sentiment", return_value=None)
    @patch("sentiment.get_news_sentiment", return_value=None)
    @patch("sentiment.get_fear_greed", return_value=None)
    def test_stock_skips_crypto_fng(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('TSLA', 'stock')
        mock_fng.assert_not_called()
        assert mult == pytest.approx(1.0)

    @patch("sentiment.get_market_sentiment", return_value={
        'sentiment_score': 0.8, 'article_count': 20,
        'positive_ratio': 0.9, 'negative_ratio': 0.05,
    })
    @patch("sentiment.get_news_sentiment", return_value={
        'sentiment_score': 0.6, 'article_count': 10,
        'positive_ratio': 0.9, 'negative_ratio': 0.05,
    })
    @patch("sentiment.get_fear_greed", return_value={'value': 55, 'label': 'Neutral'})
    def test_multiplier_capped_at_1_5(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult <= 1.5

    @patch("sentiment.get_market_sentiment", return_value={
        'sentiment_score': -0.8, 'article_count': 30,
        'positive_ratio': 0.0, 'negative_ratio': 1.0,
    })
    @patch("sentiment.get_news_sentiment", return_value={
        'sentiment_score': -0.7, 'article_count': 10,
        'positive_ratio': 0.0, 'negative_ratio': 1.0,
    })
    @patch("sentiment.get_fear_greed", return_value={'value': 5, 'label': 'Extreme Fear'})
    def test_multiplier_floored_at_0_15(self, mock_fng, mock_news, mock_market):
        mult, reasons = sentiment_gate('BTC/USD', 'crypto')
        assert mult >= 0.15
