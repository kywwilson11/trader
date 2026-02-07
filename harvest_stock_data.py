import yfinance as yf
import pandas as pd
import numpy as np
from indicators import (
    compute_rsi, compute_macd, compute_atr, compute_bbands,
    compute_stoch, compute_obv, compute_roc,
    compute_vwap, compute_gap, compute_normalized_atr,
    compute_relative_strength,
)

# ~30 high-beta, liquid stocks (no penny stocks)
STOCK_TICKERS = [
    'AMD', 'PLTR', 'SNAP', 'ROKU', 'SQ', 'HOOD', 'SHOP', 'NET', 'CRWD', 'COIN',
    'MARA', 'MSTR', 'UBER', 'SOFI', 'ABNB', 'DASH', 'RBLX', 'SMCI', 'MRVL', 'ARM',
    'FSLR', 'ENPH', 'OXY', 'MRNA', 'CRSP', 'ARKK', 'TQQQ', 'SOXL', 'TSLA', 'NVDA',
    'META',
]

BENCHMARK = 'SPY'


def fetch_spy_close():
    """Download SPY hourly data for benchmark relative strength."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = yf.download(BENCHMARK, period="1y", interval="1h", prepost=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    return df['Close']


def prepare_stock_data(ticker, spy_close=None):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", prepost=True, progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    # === BASE FEATURES (same as crypto) ===
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

    # Cyclical time encoding
    idx = df.index
    hour = idx.hour
    day = idx.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_cos'] = np.cos(2 * np.pi * day / 7)

    # === STOCK-SPECIFIC FEATURES ===

    # VWAP (resets daily)
    df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']

    # Overnight gap
    df['Gap_Pct'] = compute_gap(df['Open'], df['Close'])

    # Normalized ATR (volatility as % of price)
    df['ATR_Pct'] = compute_normalized_atr(df['High'], df['Low'], df['Close'])

    # Relative strength vs SPY
    if spy_close is not None:
        spy_aligned = spy_close.reindex(df.index, method='ffill')
        df['RS_vs_SPY'] = compute_relative_strength(df['Close'], spy_aligned)
    else:
        df['RS_vs_SPY'] = 1.0

    # === TARGET ===
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    # Drop rows with NaN (from indicator warmup + last row with no NextClose)
    df = df.dropna()

    return df


# Main
spy_close = fetch_spy_close()

all_data = []
for t in STOCK_TICKERS:
    stock_df = prepare_stock_data(t, spy_close)
    if stock_df is not None:
        stock_df['Ticker'] = t
        all_data.append(stock_df)

# Combine and Save — sort chronologically for time-series split in training
final_df = pd.concat(all_data)
final_df = final_df.sort_index()
final_df.to_csv('stock_training_data.csv')
print(f"Done! Saved {len(final_df)} rows of stock training data to stock_training_data.csv")

# Summary
print(f"\nStocks harvested: {len(all_data)}/{len(STOCK_TICKERS)}")
print(f"Feature columns: {len([c for c in final_df.columns if c not in ['Target_Return', 'Ticker', 'NextClose']])}")
print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
