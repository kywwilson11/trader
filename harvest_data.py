import yfinance as yf
import pandas as pd
import numpy as np

# Crypto-only tickers matching crypto_loop.py (BCH not UNI)
CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
    'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'BCH-USD',
]

TICKERS = CRYPTO_TICKERS


# --- INDICATOR FUNCTIONS (matches predict_now.py exactly) ---
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


def prepare_data(ticker):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    # === ORIGINAL FEATURES ===
    df['RSI'] = compute_rsi(df['Close'], length=14)

    macd_line, macd_hist, macd_signal = compute_macd(df['Close'])
    df['MACD_12_26_9'] = macd_line
    df['MACDh_12_26_9'] = macd_hist
    df['MACDs_12_26_9'] = macd_signal

    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], length=14)

    # === NEW FEATURES ===

    # Bollinger Bands (20-period)
    bb_lower, bb_mid, bb_upper, bb_bw, bb_pct = compute_bbands(df['Close'], length=20, std=2)
    df['BBL_20_2.0'] = bb_lower
    df['BBM_20_2.0'] = bb_mid
    df['BBU_20_2.0'] = bb_upper
    df['BBB_20_2.0'] = bb_bw
    df['BBP_20_2.0'] = bb_pct

    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    # Price / SMA ratio
    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']

    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    # Rate of Change (12-period)
    df['ROC'] = compute_roc(df['Close'], length=12)

    # Stochastic Oscillator
    stoch_k, stoch_d = compute_stoch(df['High'], df['Low'], df['Close'])
    df['STOCHk_14_3_3'] = stoch_k
    df['STOCHd_14_3_3'] = stoch_d

    # Cyclical time encoding (hour-of-day, day-of-week)
    idx = df.index
    hour = idx.hour
    day = idx.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_cos'] = np.cos(2 * np.pi * day / 7)

    # === TARGET ===
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    # Drop rows with NaN (from indicator warmup + last row with no NextClose)
    df = df.dropna()

    return df


# Main Loop
all_data = []
for t in TICKERS:
    stock_df = prepare_data(t)
    if stock_df is not None:
        stock_df['Ticker'] = t
        all_data.append(stock_df)

# Combine and Save — sort chronologically for time-series split in training
final_df = pd.concat(all_data)
final_df = final_df.sort_index()
final_df.to_csv('training_data.csv')
print(f"Done! Saved {len(final_df)} rows of training data to training_data.csv")
