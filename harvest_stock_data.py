"""Harvest stock training data — downloads hourly OHLCV for ~45 stocks via yfinance.

Computes stock-specific technical features (indicators.compute_stock_features)
including SPY relative strength, labels each bar with the forward return, and
saves the combined dataset to stock_training_data.csv for use by
hypersearch_dual.py --prefix stock.

Usage:
    python harvest_stock_data.py                  # default 4-bar forward return
    python harvest_stock_data.py --forward-bars 1 # single-bar (legacy)
"""

import argparse
import yfinance as yf
import pandas as pd
from indicators import compute_stock_features
from market_data import flatten_yfinance_columns
from stock_config import load_stock_universe

STOCK_TICKERS = [t for t in load_stock_universe() if '/' not in t]

BENCHMARK = 'SPY'


def fetch_spy_close():
    """Download SPY hourly data for benchmark relative strength."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = yf.download(BENCHMARK, period="1y", interval="1h", prepost=True, progress=False)
    df = flatten_yfinance_columns(df)
    if df.empty:
        return None
    return df['Close']


FORWARD_BARS = 4  # default: predict 4-hour forward return


def prepare_stock_data(ticker, spy_close=None, forward_bars=FORWARD_BARS):
    """Download hourly bars, compute stock features, and add forward return target."""
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", prepost=True, progress=False)
    df = flatten_yfinance_columns(df)

    if df.empty:
        return None

    df = compute_stock_features(df, spy_close=spy_close)

    # Target: forward return as a percentage (multi-bar lookahead)
    df['NextClose'] = df['Close'].shift(-forward_bars)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


def parse_args():
    parser = argparse.ArgumentParser(description='Harvest stock training data')
    parser.add_argument('--forward-bars', type=int, default=FORWARD_BARS,
                        help=f'Number of bars to look ahead for target return (default: {FORWARD_BARS})')
    return parser.parse_args()


def main():
    args = parse_args()
    forward_bars = args.forward_bars
    print(f"Forward bars: {forward_bars} (predicting {forward_bars}-hour return)")

    spy_close = fetch_spy_close()

    all_data = []
    for t in STOCK_TICKERS:
        stock_df = prepare_stock_data(t, spy_close, forward_bars=forward_bars)
        if stock_df is not None:
            stock_df['Ticker'] = t
            all_data.append(stock_df)

    # Combine and save — sort chronologically for time-series split in training
    final_df = pd.concat(all_data)
    final_df = final_df.sort_index()

    # Add historical sentiment — use cached data if available, else 0.0
    # (run_pipeline.py fetches fresh sentiment in background during training)
    try:
        from sentiment_history import fetch_stock_sentiment_history
        start_date = str(final_df.index.min().date())
        end_date = str(final_df.index.max().date())
        sentiment = fetch_stock_sentiment_history(
            STOCK_TICKERS, start_date, end_date, cached_only=True)
        final_df['Daily_Sentiment'] = [
            sentiment.get((ticker, str(date)), 0.0)
            for ticker, date in zip(final_df['Ticker'], final_df.index.date)
        ]
        filled = sum(1 for v in final_df['Daily_Sentiment'] if v != 0.0)
        print(f"Daily_Sentiment: {filled}/{len(final_df)} bars have sentiment")
    except Exception as e:
        print(f"WARNING: Could not load stock sentiment history: {e}")
        final_df['Daily_Sentiment'] = 0.0

    final_df.to_csv('stock_training_data.csv')
    print(f"Done! Saved {len(final_df)} rows of stock training data to stock_training_data.csv")

    # Summary
    print(f"\nStocks harvested: {len(all_data)}/{len(STOCK_TICKERS)}")
    print(f"Feature columns: {len([c for c in final_df.columns if c not in ['Target_Return', 'Ticker', 'NextClose']])}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")


if __name__ == '__main__':
    main()
