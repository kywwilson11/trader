"""Harvest crypto training data — downloads hourly OHLCV for 10 cryptos via yfinance.

Computes technical features (indicators.compute_features), labels each bar with
the next-bar return, and saves the combined dataset to training_data.csv for use
by hypersearch_dual.py.
"""

import yfinance as yf
import pandas as pd
from indicators import compute_features
from market_data import flatten_yfinance_columns

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
    df = flatten_yfinance_columns(df)
    if df.empty:
        return None
    return df['Close']


def prepare_data(ticker, btc_close=None):
    """Download hourly bars, compute features, and add next-bar return target."""
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", progress=False)
    df = flatten_yfinance_columns(df)

    if df.empty:
        return None

    df = compute_features(df, btc_close=btc_close)

    # Target: next bar's return as a percentage
    df['NextClose'] = df['Close'].shift(-4)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


def main():
    btc_close = fetch_btc_close()

    all_data = []
    for t in CRYPTO_TICKERS:
        crypto_df = prepare_data(t, btc_close=btc_close)
        if crypto_df is not None:
            crypto_df['Ticker'] = t
            all_data.append(crypto_df)

    # Combine and save — sort chronologically for time-series split in training
    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()

    # Add historical sentiment (Fear & Greed Index for crypto)
    # Use cached data if available (instant); fetch in background for next run
    try:
        from sentiment_history import fetch_crypto_sentiment_history
        import threading
        start_date = str(final_df.index.min().date())
        end_date = str(final_df.index.max().date())
        sentiment = fetch_crypto_sentiment_history(start_date, end_date)
        final_df['Daily_Sentiment'] = pd.Series(
            final_df.index.date.astype(str), index=final_df.index
        ).map(sentiment).fillna(0.0).values
        filled = (final_df['Daily_Sentiment'] != 0).sum()
        print(f"Daily_Sentiment: {filled}/{len(final_df)} bars have sentiment")
    except Exception as e:
        print(f"WARNING: Could not fetch crypto sentiment history: {e}")
        final_df['Daily_Sentiment'] = 0.0

    final_df.to_csv('training_data.csv')
    print(f"Done! Saved {len(final_df)} rows of training data to training_data.csv")
    print(f"\nCryptos harvested: {len(all_data)}/{len(CRYPTO_TICKERS)}")
    print(f"Feature columns: {len([c for c in final_df.columns if c not in ['Target_Return', 'Ticker', 'NextClose']])}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")


if __name__ == '__main__':
    main()
