"""Tests for data_sources.py — CryptoCompare fetch and fallback chain."""

import json
from unittest import mock

import pandas as pd
import pytest


def _mock_cc_response(bars):
    """Create a mock urllib response for CryptoCompare."""
    data = {
        'Response': 'Success',
        'Data': {
            'Data': bars,
        }
    }

    class MockResp:
        def read(self):
            return json.dumps(data).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return MockResp()


def test_fetch_cryptocompare_hourly():
    """Mock CryptoCompare API and verify DataFrame shape."""
    from data_sources import fetch_cryptocompare_hourly

    # Generate 10 fake hourly bars
    import time as _time
    base_ts = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp())
    bars = [
        {'time': base_ts + i * 3600, 'open': 100, 'high': 101,
         'low': 99, 'close': 100.5, 'volumefrom': 1000}
        for i in range(10)
    ]

    with mock.patch('data_sources.urllib.request.urlopen') as mock_urlopen:
        mock_urlopen.return_value = _mock_cc_response(bars)
        df = fetch_cryptocompare_hourly('BTC-USD', '2024-01-01', '2024-01-01 10:00')

    assert df is not None
    assert len(df) == 10
    assert set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns)
    assert df.index.tz is not None  # should be UTC


def test_fetch_with_fallback_alpaca_only():
    """Alpaca succeeds — yfinance and CC should still be tried but Alpaca data preferred."""
    from data_sources import fetch_with_fallback

    dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
    alpaca_df = pd.DataFrame({
        'Open': [100]*5, 'High': [101]*5, 'Low': [99]*5,
        'Close': [100.5]*5, 'Volume': [1000]*5,
    }, index=dates)

    mock_yf = mock.MagicMock()
    mock_yf.download.return_value = pd.DataFrame()

    with mock.patch('market_data.fetch_historical_bars', return_value=alpaca_df), \
         mock.patch.dict('sys.modules', {'yfinance': mock_yf}), \
         mock.patch('market_data.flatten_yfinance_columns', return_value=pd.DataFrame()), \
         mock.patch('data_sources.fetch_cryptocompare_hourly', return_value=None):
        df = fetch_with_fallback('BTC-USD', '2024-01-01', api=mock.MagicMock(),
                                  asset_type='crypto')

    assert df is not None
    assert len(df) >= 5


def test_fetch_with_fallback_yfinance_fallback():
    """Alpaca fails, yfinance succeeds."""
    from data_sources import fetch_with_fallback

    dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
    yf_df = pd.DataFrame({
        'Open': [200]*5, 'High': [201]*5, 'Low': [199]*5,
        'Close': [200.5]*5, 'Volume': [2000]*5,
    }, index=dates)

    mock_yf = mock.MagicMock()
    mock_yf.download.return_value = yf_df

    with mock.patch('market_data.fetch_historical_bars', side_effect=Exception("API error")), \
         mock.patch.dict('sys.modules', {'yfinance': mock_yf}), \
         mock.patch('market_data.flatten_yfinance_columns', return_value=yf_df), \
         mock.patch('data_sources.fetch_cryptocompare_hourly', return_value=None):
        df = fetch_with_fallback('BTC-USD', '2024-01-01', api=mock.MagicMock(),
                                  asset_type='crypto')

    assert df is not None
    assert len(df) == 5


def test_fetch_with_fallback_all_fail():
    """All sources fail — returns None."""
    from data_sources import fetch_with_fallback

    mock_yf = mock.MagicMock()
    mock_yf.download.return_value = pd.DataFrame()

    with mock.patch('market_data.fetch_historical_bars', side_effect=Exception("fail")), \
         mock.patch.dict('sys.modules', {'yfinance': mock_yf}), \
         mock.patch('market_data.flatten_yfinance_columns', return_value=pd.DataFrame()), \
         mock.patch('data_sources.fetch_cryptocompare_hourly', return_value=None):
        df = fetch_with_fallback('BTC-USD', '2024-01-01', api=mock.MagicMock(),
                                  asset_type='crypto')

    assert df is None


def test_cc_symbol_conversion():
    """CryptoCompare symbol conversion handles various formats."""
    from data_sources import _cc_symbol
    assert _cc_symbol('BTC-USD') == 'BTC'
    assert _cc_symbol('BTC/USD') == 'BTC'
    assert _cc_symbol('ETH') == 'ETH'


def test_fetch_with_fallback_stock_no_cc():
    """Stocks should not try CryptoCompare."""
    from data_sources import fetch_with_fallback

    dates = pd.date_range('2024-01-01', periods=5, freq='h', tz='UTC')
    yf_df = pd.DataFrame({
        'Open': [200]*5, 'High': [201]*5, 'Low': [199]*5,
        'Close': [200.5]*5, 'Volume': [2000]*5,
    }, index=dates)

    mock_yf = mock.MagicMock()
    mock_yf.download.return_value = yf_df

    with mock.patch('market_data.fetch_historical_bars', return_value=None), \
         mock.patch.dict('sys.modules', {'yfinance': mock_yf}), \
         mock.patch('market_data.flatten_yfinance_columns', return_value=yf_df), \
         mock.patch('data_sources.fetch_cryptocompare_hourly') as mock_cc:
        df = fetch_with_fallback('AAPL', '2024-01-01', api=None,
                                  asset_type='stock')

    # CryptoCompare should NOT be called for stocks
    mock_cc.assert_not_called()
    assert df is not None
