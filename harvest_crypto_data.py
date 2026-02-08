import yfinance as yf
import pandas as pd
from indicators import compute_features

# Crypto-only tickers matching crypto_loop.py (BCH not UNI)
CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
    'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'BCH-USD',
]

BENCHMARK = 'BTC-USD'


def fetch_btc_close():
    """Download BTC-USD hourly data for cross-asset features."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = yf.download(BENCHMARK, period="1y", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    return df['Close']


def prepare_data(ticker, btc_close=None):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    df = compute_features(df, btc_close=btc_close)

    # Target
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


# Main
btc_close = fetch_btc_close()

all_data = []
for t in CRYPTO_TICKERS:
    crypto_df = prepare_data(t, btc_close=btc_close)
    if crypto_df is not None:
        crypto_df['Ticker'] = t
        all_data.append(crypto_df)

# Combine and Save — sort chronologically for time-series split in training
final_df = pd.concat(all_data)
final_df = final_df.sort_index()
final_df.to_csv('training_data.csv')
print(f"Done! Saved {len(final_df)} rows of training data to training_data.csv")
print(f"\nCryptos harvested: {len(all_data)}/{len(CRYPTO_TICKERS)}")
print(f"Feature columns: {len([c for c in final_df.columns if c not in ['Target_Return', 'Ticker', 'NextClose']])}")
print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
