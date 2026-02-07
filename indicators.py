import pandas as pd
import numpy as np


def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


def compute_atr(high, low, close, length=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def compute_bbands(close, length=20, std=2):
    sma = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bandwidth = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)
    return lower, sma, upper, bandwidth, pct_b


def compute_stoch(high, low, close, k=14, d=3, smooth_k=3):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def compute_obv(close, volume):
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()


def compute_roc(close, length=12):
    return ((close - close.shift(length)) / close.shift(length)) * 100


def compute_features(df):
    """Compute all features matching harvest_crypto_data.py exactly."""
    df['RSI'] = compute_rsi(df['Close'], length=14)

    macd_line, macd_hist, macd_signal = compute_macd(df['Close'])
    df['MACD_12_26_9'] = macd_line
    df['MACDh_12_26_9'] = macd_hist
    df['MACDs_12_26_9'] = macd_signal

    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], length=14)

    bb_lower, bb_mid, bb_upper, bb_bw, bb_pct = compute_bbands(df['Close'], length=20, std=2)
    df['BBL_20_2.0'] = bb_lower
    df['BBM_20_2.0'] = bb_mid
    df['BBU_20_2.0'] = bb_upper
    df['BBB_20_2.0'] = bb_bw
    df['BBP_20_2.0'] = bb_pct

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']

    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    df['ROC'] = compute_roc(df['Close'], length=12)

    stoch_k, stoch_d = compute_stoch(df['High'], df['Low'], df['Close'])
    df['STOCHk_14_3_3'] = stoch_k
    df['STOCHd_14_3_3'] = stoch_d

    idx = df.index
    hour = idx.hour
    day = idx.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_cos'] = np.cos(2 * np.pi * day / 7)

    return df


# --- STOCK-SPECIFIC INDICATORS ---

def compute_vwap(high, low, close, volume):
    """Session VWAP — resets each trading day.
    Returns a Series with intraday VWAP values.
    """
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume

    # Group by date for daily reset
    dates = close.index.date
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def compute_gap(open_price, close):
    """Overnight gap: (today's open - yesterday's close) / yesterday's close * 100.
    Returns a Series with gap % for each bar (only meaningful on first bar of day).
    """
    prev_close = close.shift(1)
    gap = (open_price - prev_close) / prev_close * 100
    return gap


def compute_relative_strength(close, benchmark_close):
    """Relative strength vs benchmark (e.g. SPY).
    RS = stock ROC / benchmark ROC over a rolling window.
    Returns ratio > 1 means outperforming, < 1 means underperforming.
    """
    stock_roc = close.pct_change(12)
    bench_roc = benchmark_close.pct_change(12)
    # Avoid division by zero
    rs = stock_roc / bench_roc.replace(0, np.nan)
    return rs


def compute_normalized_atr(high, low, close, length=14):
    """ATR normalized by price: ATR / close * 100.
    Gives volatility as a percentage, comparable across different price levels.
    """
    atr = compute_atr(high, low, close, length)
    return atr / close * 100


def compute_stock_features(df, spy_close=None):
    """Compute all features for stocks. Includes base crypto features + stock-specific ones.

    Args:
        df: DataFrame with OHLCV columns (DatetimeIndex)
        spy_close: Optional Series of SPY close prices aligned to df's index
                   for relative strength calculation
    """
    # Base features (same as crypto)
    df = compute_features(df)

    # VWAP
    df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']

    # Overnight gap
    df['Gap_Pct'] = compute_gap(df['Open'], df['Close'])

    # Normalized ATR (volatility as % of price)
    df['ATR_Pct'] = compute_normalized_atr(df['High'], df['Low'], df['Close'])

    # Relative strength vs SPY
    if spy_close is not None:
        # Align SPY close to df's index
        spy_aligned = spy_close.reindex(df.index, method='ffill')
        df['RS_vs_SPY'] = compute_relative_strength(df['Close'], spy_aligned)
    else:
        df['RS_vs_SPY'] = 1.0  # neutral if no benchmark

    return df
