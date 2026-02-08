"""Tests for indicators.py — pure numeric functions."""

import numpy as np
import pandas as pd
import pytest

from indicators import (
    compute_atr,
    compute_bbands,
    compute_features,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_stoch,
    compute_stock_features,
)


class TestComputeRSI:
    def test_output_length(self, sample_ohlcv_df):
        rsi = compute_rsi(sample_ohlcv_df["Close"])
        assert len(rsi) == len(sample_ohlcv_df)

    def test_range_0_to_100(self, sample_ohlcv_df):
        rsi = compute_rsi(sample_ohlcv_df["Close"]).dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_constant_series(self):
        series = pd.Series([50.0] * 30)
        rsi = compute_rsi(series).dropna()
        # Constant price → RSI should be ~50 (no gains or losses)
        assert all(abs(v - 50) < 1 for v in rsi)


class TestComputeMACD:
    def test_returns_three_series(self, sample_ohlcv_df):
        macd, hist, signal = compute_macd(sample_ohlcv_df["Close"])
        assert isinstance(macd, pd.Series)
        assert isinstance(hist, pd.Series)
        assert isinstance(signal, pd.Series)

    def test_output_length(self, sample_ohlcv_df):
        macd, hist, signal = compute_macd(sample_ohlcv_df["Close"])
        assert len(macd) == len(sample_ohlcv_df)


class TestComputeATR:
    def test_positive_values(self, sample_ohlcv_df):
        atr = compute_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        ).dropna()
        assert (atr >= 0).all()

    def test_output_length(self, sample_ohlcv_df):
        atr = compute_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert len(atr) == len(sample_ohlcv_df)


class TestComputeBBands:
    def test_returns_five_series(self, sample_ohlcv_df):
        result = compute_bbands(sample_ohlcv_df["Close"])
        assert len(result) == 5
        lower, mid, upper, bw, pct_b = result
        assert isinstance(lower, pd.Series)

    def test_upper_above_lower(self, sample_ohlcv_df):
        lower, mid, upper, bw, pct_b = compute_bbands(sample_ohlcv_df["Close"])
        valid = lower.dropna().index
        assert (upper[valid] >= lower[valid]).all()


class TestComputeStoch:
    def test_returns_two_series(self, sample_ohlcv_df):
        k, d = compute_stoch(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_range_0_to_100(self, sample_ohlcv_df):
        k, d = compute_stoch(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        k_valid = k.dropna()
        assert k_valid.min() >= -0.01  # allow tiny float imprecision
        assert k_valid.max() <= 100.01


class TestComputeOBV:
    def test_output_length(self, sample_ohlcv_df):
        obv = compute_obv(sample_ohlcv_df["Close"], sample_ohlcv_df["Volume"])
        assert len(obv) == len(sample_ohlcv_df)


class TestComputeFeatures:
    def test_adds_columns(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        # Add a DatetimeIndex for temporal features
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        assert "RSI" in result.columns
        assert "MACD_12_26_9" in result.columns
        assert "ATR" in result.columns

    def test_no_nan_rows_at_end(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        # Last row should have all features computed
        assert not result.iloc[-1][["RSI", "ATR", "OBV"]].isna().any()


class TestComputeStockFeatures:
    def test_adds_stock_columns(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_stock_features(df)
        assert "VWAP" in result.columns
        assert "Gap_Pct" in result.columns
