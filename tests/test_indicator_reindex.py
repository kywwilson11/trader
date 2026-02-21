"""Tests for indicators.py reindex behavior — btc_close and spy_close alignment."""

import numpy as np
import pandas as pd
import pytest
import warnings

from indicators import compute_features, compute_stock_features


class TestBtcReindex:
    """Test that BTC cross-asset features handle misaligned indices correctly."""

    def test_btc_close_with_gaps(self):
        """BTC close with missing timestamps should be forward-filled."""
        n = 120
        np.random.seed(42)
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": np.random.randn(n).cumsum() + 100,
            "High": np.random.randn(n).cumsum() + 101,
            "Low": np.random.randn(n).cumsum() + 99,
            "Close": np.random.randn(n).cumsum() + 100,
            "Volume": np.abs(np.random.randn(n)) * 1000 + 100,
        }, index=idx)

        # BTC close with every other hour missing
        btc_idx = idx[::2]  # only even hours
        btc = pd.Series(np.random.randn(len(btc_idx)).cumsum() + 50000, index=btc_idx)

        result = compute_features(df, btc_close=btc)
        assert "BTC_Return_1h" in result.columns
        # Forward-fill should mean no NaN after the first BTC observation
        btc_ret = result["BTC_Return_1h"].iloc[2:]  # skip first couple
        assert btc_ret.notna().sum() > len(btc_ret) * 0.8

    def test_no_deprecation_warning(self):
        """Verify reindex does not produce FutureWarning about method='ffill'."""
        n = 60
        np.random.seed(42)
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": np.random.randn(n).cumsum() + 100,
            "High": np.random.randn(n).cumsum() + 101,
            "Low": np.random.randn(n).cumsum() + 99,
            "Close": np.random.randn(n).cumsum() + 100,
            "Volume": np.abs(np.random.randn(n)) * 1000 + 100,
        }, index=idx)
        btc = pd.Series(np.random.randn(n).cumsum() + 50000, index=idx)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # Should NOT raise FutureWarning about deprecated method='ffill'
            compute_features(df, btc_close=btc)


class TestSpyReindex:
    """Test that SPY benchmark features handle misaligned indices correctly."""

    def test_spy_close_with_gaps(self):
        """SPY close with missing timestamps should be forward-filled."""
        n = 120
        np.random.seed(42)
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": np.random.randn(n).cumsum() + 100,
            "High": np.random.randn(n).cumsum() + 101,
            "Low": np.random.randn(n).cumsum() + 99,
            "Close": np.random.randn(n).cumsum() + 100,
            "Volume": np.abs(np.random.randn(n)) * 1000 + 100,
        }, index=idx)

        # SPY only has market hours (not every hour)
        spy_idx = idx[::3]
        spy = pd.Series(np.random.randn(len(spy_idx)).cumsum() + 500, index=spy_idx)

        result = compute_stock_features(df, spy_close=spy)
        assert "RS_vs_SPY" in result.columns
        rs = result["RS_vs_SPY"].dropna()
        assert len(rs) > 0

    def test_no_deprecation_warning_spy(self):
        """Verify SPY reindex does not produce FutureWarning."""
        n = 60
        np.random.seed(42)
        idx = pd.date_range("2025-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "Open": np.random.randn(n).cumsum() + 100,
            "High": np.random.randn(n).cumsum() + 101,
            "Low": np.random.randn(n).cumsum() + 99,
            "Close": np.random.randn(n).cumsum() + 100,
            "Volume": np.abs(np.random.randn(n)) * 1000 + 100,
        }, index=idx)
        spy = pd.Series(np.random.randn(n).cumsum() + 500, index=idx)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            compute_stock_features(df, spy_close=spy)
