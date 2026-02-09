"""Harvest stock training data — downloads hourly OHLCV for ~45 stocks via yfinance.

Computes stock-specific technical features (indicators.compute_stock_features)
including SPY relative strength, labels each bar with the next-bar return, and
saves the combined dataset to stock_training_data.csv for use by
hypersearch_dual.py --prefix stock.
"""

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


def prepare_stock_data(ticker, spy_close=None):
    """Download hourly bars, compute stock features, and add next-bar return target."""
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", prepost=True, progress=False)
    df = flatten_yfinance_columns(df)

    if df.empty:
        return None

    df = compute_stock_features(df, spy_close=spy_close)

    # Target: next bar's return as a percentage
    df['NextClose'] = df['Close'].shift(-4)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


def main():
    spy_close = fetch_spy_close()

    all_data = []
    for t in STOCK_TICKERS:
        stock_df = prepare_stock_data(t, spy_close)
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
