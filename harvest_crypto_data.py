import yfinance as yf
import pandas as pd
from indicators import compute_features

# Crypto-only tickers matching crypto_loop.py (BCH not UNI)
CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
    'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'BCH-USD',
]


def prepare_data(ticker):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    df = compute_features(df)

    # Target
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


# Main
all_data = []
for t in CRYPTO_TICKERS:
    crypto_df = prepare_data(t)
    if crypto_df is not None:
        crypto_df['Ticker'] = t
        all_data.append(crypto_df)

# Combine and Save — sort chronologically for time-series split in training
final_df = pd.concat(all_data)
final_df = final_df.sort_index()
final_df.to_csv('training_data.csv')
print(f"Done! Saved {len(final_df)} rows of training data to training_data.csv")
