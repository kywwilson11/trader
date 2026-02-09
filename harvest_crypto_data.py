"""Harvest crypto training data — downloads hourly OHLCV for 10 cryptos via yfinance.

Computes technical features (indicators.compute_features), labels each bar with
the forward return, and saves the combined dataset to training_data.csv for use
by hypersearch_dual.py.

Usage:
    python harvest_crypto_data.py                  # default 4-bar forward return
    python harvest_crypto_data.py --forward-bars 1 # single-bar (legacy)
"""

import argparse
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


FORWARD_BARS = 4  # default: predict 4-hour forward return


def prepare_data(ticker, btc_close=None, forward_bars=FORWARD_BARS):
    """Download hourly bars, compute features, and add forward return target."""
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", progress=False)
    df = flatten_yfinance_columns(df)

    if df.empty:
        return None

    df = compute_features(df, btc_close=btc_close)

    # Target: forward return as a percentage (multi-bar lookahead)
    df['NextClose'] = df['Close'].shift(-forward_bars)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Harvest crypto training data')
    parser.add_argument('--forward-bars', type=int, default=FORWARD_BARS,
                        help=f'Number of bars to look ahead for target return (default: {FORWARD_BARS})')
    return parser.parse_args()


def main():
    args = parse_args()
    forward_bars = args.forward_bars
    print(f"Forward bars: {forward_bars} (predicting {forward_bars}-hour return)")

    btc_close = fetch_btc_close()

    all_data = []
    for t in CRYPTO_TICKERS:
        crypto_df = prepare_data(t, btc_close=btc_close, forward_bars=forward_bars)
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
