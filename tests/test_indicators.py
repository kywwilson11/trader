"""Tests for indicators.py — pure numeric functions."""

import numpy as np
import pandas as pd
import pytest

from indicators import (
    compute_atr,
    compute_bbands,
    compute_features,
    compute_gap,
    compute_linear_slope,
    compute_macd,
    compute_normalized_atr,
    compute_obv,
    compute_relative_strength,
    compute_roc,
    compute_rolling_percentile,
    compute_rsi,
    compute_stoch,
    compute_stock_features,
    compute_vwap,
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

    def test_strong_uptrend_high_rsi(self):
        # Random walk biased strongly upward
        np.random.seed(42)
        changes = np.random.randn(120) * 0.5 + 0.5  # mean +0.5 per step
        series = pd.Series(100 + np.cumsum(changes))
        rsi = compute_rsi(series).dropna()
        assert len(rsi) > 0
        assert rsi.values[-1] > 70

    def test_strong_downtrend_low_rsi(self):
        np.random.seed(42)
        changes = np.random.randn(120) * 0.5 - 0.5  # mean -0.5 per step
        series = pd.Series(100 + np.cumsum(changes))
        rsi = compute_rsi(series).dropna()
        assert len(rsi) > 0
        assert rsi.values[-1] < 30


class TestComputeMACD:
    def test_returns_three_series(self, sample_ohlcv_df):
        macd, hist, signal = compute_macd(sample_ohlcv_df["Close"])
        assert isinstance(macd, pd.Series)
        assert isinstance(hist, pd.Series)
        assert isinstance(signal, pd.Series)

    def test_output_length(self, sample_ohlcv_df):
        macd, hist, signal = compute_macd(sample_ohlcv_df["Close"])
        assert len(macd) == len(sample_ohlcv_df)

    def test_histogram_is_macd_minus_signal(self, sample_ohlcv_df):
        macd, hist, signal = compute_macd(sample_ohlcv_df["Close"])
        diff = (macd - signal).dropna()
        assert np.allclose(hist.dropna().values, diff.values, atol=1e-10)


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

    def test_zero_range_bars(self):
        n = 30
        flat = pd.Series([100.0] * n)
        atr = compute_atr(flat, flat, flat).dropna()
        assert (atr == 0).all()


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

    def test_mid_is_sma(self, sample_ohlcv_df):
        close = sample_ohlcv_df["Close"]
        _, mid, _, _, _ = compute_bbands(close, length=20)
        sma = close.rolling(20).mean()
        valid = sma.dropna().index
        assert np.allclose(mid[valid].values, sma[valid].values, atol=0.01)


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

    def test_rising_prices_positive_obv(self):
        close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        volume = pd.Series([100.0] * 5)
        obv = compute_obv(close, volume)
        assert obv.iloc[-1] > 0

    def test_falling_prices_negative_obv(self):
        close = pd.Series([14.0, 13.0, 12.0, 11.0, 10.0])
        volume = pd.Series([100.0] * 5)
        obv = compute_obv(close, volume)
        assert obv.iloc[-1] < 0


class TestComputeROC:
    def test_output_length(self, sample_ohlcv_df):
        roc = compute_roc(sample_ohlcv_df["Close"])
        assert len(roc) == len(sample_ohlcv_df)

    def test_known_value(self):
        close = pd.Series([100.0] * 12 + [110.0])
        roc = compute_roc(close, length=12)
        # (110 - 100) / 100 * 100 = 10%
        assert abs(roc.iloc[-1] - 10.0) < 0.01

    def test_negative_return(self):
        close = pd.Series([100.0] * 12 + [90.0])
        roc = compute_roc(close, length=12)
        assert roc.iloc[-1] < 0

    def test_zero_at_flat_price(self):
        close = pd.Series([50.0] * 20)
        roc = compute_roc(close, length=12).dropna()
        assert (roc == 0).all()


class TestComputeVWAP:
    def test_output_length(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        vwap = compute_vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        assert len(vwap) == len(df)

    def test_between_high_and_low(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        vwap = compute_vwap(df["High"], df["Low"], df["Close"], df["Volume"]).dropna()
        # VWAP based on typical price should be between low and high (per day)
        # Check for most rows (cumulative can drift slightly)
        within = ((vwap >= df["Low"].loc[vwap.index] - 1) &
                  (vwap <= df["High"].loc[vwap.index] + 1))
        assert within.mean() > 0.8


class TestComputeGap:
    def test_output_length(self, sample_ohlcv_df):
        gap = compute_gap(sample_ohlcv_df["Open"], sample_ohlcv_df["Close"])
        assert len(gap) == len(sample_ohlcv_df)

    def test_first_bar_nan(self, sample_ohlcv_df):
        gap = compute_gap(sample_ohlcv_df["Open"], sample_ohlcv_df["Close"])
        assert pd.isna(gap.iloc[0])

    def test_known_gap(self):
        open_price = pd.Series([100.0, 110.0])
        close = pd.Series([100.0, 105.0])
        gap = compute_gap(open_price, close)
        # Bar 1: (110 - 100) / 100 * 100 = 10%
        assert abs(gap.iloc[1] - 10.0) < 0.01


class TestComputeRelativeStrength:
    def test_outperformance(self):
        n = 20
        stock = pd.Series(np.linspace(100, 130, n))  # +30%
        bench = pd.Series(np.linspace(100, 110, n))   # +10%
        rs = compute_relative_strength(stock, bench).dropna()
        assert rs.iloc[-1] > 1.0

    def test_underperformance(self):
        n = 20
        stock = pd.Series(np.linspace(100, 105, n))   # +5%
        bench = pd.Series(np.linspace(100, 130, n))    # +30%
        rs = compute_relative_strength(stock, bench).dropna()
        assert rs.iloc[-1] < 1.0


class TestComputeNormalizedATR:
    def test_positive_percentage(self, sample_ohlcv_df):
        natr = compute_normalized_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        ).dropna()
        assert (natr >= 0).all()

    def test_output_length(self, sample_ohlcv_df):
        natr = compute_normalized_atr(
            sample_ohlcv_df["High"],
            sample_ohlcv_df["Low"],
            sample_ohlcv_df["Close"],
        )
        assert len(natr) == len(sample_ohlcv_df)


class TestComputeRollingPercentile:
    def test_output_length(self, sample_ohlcv_df):
        rp = compute_rolling_percentile(sample_ohlcv_df["Close"], window=20)
        assert len(rp) == len(sample_ohlcv_df)

    def test_range_0_to_1(self, sample_ohlcv_df):
        rp = compute_rolling_percentile(sample_ohlcv_df["Close"], window=20).dropna()
        assert rp.min() >= 0
        assert rp.max() <= 1.0

    def test_max_at_highest(self):
        # Rising series — last value always highest in window
        series = pd.Series(np.arange(1.0, 51.0))
        rp = compute_rolling_percentile(series, window=20).dropna()
        # Last value should have high percentile
        assert rp.iloc[-1] > 0.8


class TestComputeLinearSlope:
    def test_positive_slope_rising(self):
        series = pd.Series(np.arange(1.0, 21.0))
        slope = compute_linear_slope(series, window=5).dropna()
        assert (slope > 0).all()

    def test_negative_slope_falling(self):
        series = pd.Series(np.arange(20.0, 0.0, -1.0))
        slope = compute_linear_slope(series, window=5).dropna()
        assert (slope < 0).all()

    def test_zero_slope_flat(self):
        series = pd.Series([5.0] * 20)
        slope = compute_linear_slope(series, window=5).dropna()
        assert np.allclose(slope.values, 0.0, atol=1e-10)

    def test_output_length(self, sample_ohlcv_df):
        slope = compute_linear_slope(sample_ohlcv_df["Close"], window=5)
        assert len(slope) == len(sample_ohlcv_df)


class TestComputeFeatures:
    def test_adds_columns(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        assert "RSI" in result.columns
        assert "MACD_12_26_9" in result.columns
        assert "ATR" in result.columns

    def test_no_nan_rows_at_end(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        assert not result.iloc[-1][["RSI", "ATR", "OBV"]].isna().any()

    def test_phase_c_indicators(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        assert "ATR_Percentile" in result.columns
        assert "RSI_Divergence" in result.columns
        assert "Vol_Price_Confirm" in result.columns
        assert "SMA_100" in result.columns

    def test_btc_cross_asset_features(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        btc = pd.Series(np.random.randn(len(df)).cumsum() + 50000, index=df.index)
        result = compute_features(df, btc_close=btc)
        assert "BTC_Return_1h" in result.columns
        assert "BTC_SMA_Ratio" in result.columns
        assert "BTC_RSI" in result.columns

    def test_temporal_features(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_features(df)
        assert "Hour_sin" in result.columns
        assert "Hour_cos" in result.columns
        assert "Day_sin" in result.columns
        assert "Day_cos" in result.columns
        # sin/cos values should be in [-1, 1]
        assert result["Hour_sin"].min() >= -1.01
        assert result["Hour_sin"].max() <= 1.01


class TestComputeStockFeatures:
    def test_adds_stock_columns(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_stock_features(df)
        assert "VWAP" in result.columns
        assert "Gap_Pct" in result.columns
        assert "ATR_Pct" in result.columns
        assert "RS_vs_SPY" in result.columns

    def test_rs_vs_spy_with_benchmark(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        spy = pd.Series(np.random.randn(len(df)).cumsum() + 500, index=df.index)
        result = compute_stock_features(df, spy_close=spy)
        assert "RS_vs_SPY" in result.columns
        # Should NOT all be 1.0 since we provided real benchmark data
        assert not (result["RS_vs_SPY"].dropna() == 1.0).all()

    def test_rs_defaults_to_one_without_spy(self, sample_ohlcv_df):
        df = sample_ohlcv_df.copy()
        df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
        result = compute_stock_features(df, spy_close=None)
        assert (result["RS_vs_SPY"] == 1.0).all()
