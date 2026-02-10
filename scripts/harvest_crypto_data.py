"""Harvest crypto training data — Alpaca historical + yfinance hourly OHLCV.

Fetches data from two sources:
  - Alpaca: Jan 2021 – present (via get_crypto_bars auto-pagination)
  - yfinance: Most recent 730 days (max for hourly)
Merges, deduplicates, computes technical features, and generates multi-horizon
return targets for use by hypersearch_dual.py.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

from indicators import compute_features
from market_data import flatten_yfinance_columns, fetch_historical_bars

load_dotenv()

# Top 6 cryptos by market cap, matching crypto_loop.py
# yfinance format — converted to Alpaca format (/ instead of -) for Alpaca calls
CRYPTO_TICKERS = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
    'LINK-USD',
]

BENCHMARK = 'BTC-USD'

ALPACA_START = '2021-01-01'

# Multi-horizon forward returns (bars ahead)
FORWARD_BARS = [1, 2, 4, 8, 12, 16, 24, 32]


def _to_alpaca(yf_ticker):
    """Convert yfinance ticker to Alpaca format: BTC-USD -> BTC/USD."""
    return yf_ticker.replace('-', '/')


def _get_alpaca_api():
    """Build Alpaca REST client, or None if credentials missing."""
    try:
        import alpaca_trade_api as tradeapi
        # Increase SDK internal retry backoff (default 3s is too aggressive)
        os.environ.setdefault('APCA_RETRY_WAIT', '5')
        os.environ.setdefault('APCA_RETRY_MAX', '5')
        key = os.getenv('ALPACA_API_KEY')
        secret = os.getenv('ALPACA_API_SECRET')
        url = os.getenv('ALPACA_BASE_URL')
        if not key or not secret:
            return None
        return tradeapi.REST(key, secret, url, api_version='v2')
    except Exception as e:
        print(f"WARNING: Could not create Alpaca API client: {e}")
        return None


def fetch_ticker_data(ticker, api=None):
    """Fetch hourly bars from Alpaca + yfinance, merge and deduplicate.

    Returns a DataFrame with OHLCV columns and DatetimeIndex, or None.
    """
    frames = []

    # 1. Alpaca historical data (2021+)
    if api is not None:
        alpaca_sym = _to_alpaca(ticker)
        alpaca_df = fetch_historical_bars(api, alpaca_sym, ALPACA_START, asset_type='crypto')
        if alpaca_df is not None and not alpaca_df.empty:
            # Ensure tz-aware UTC index for dedup
            if alpaca_df.index.tz is None:
                alpaca_df.index = alpaca_df.index.tz_localize('UTC')
            else:
                alpaca_df.index = alpaca_df.index.tz_convert('UTC')
            frames.append(alpaca_df)
        time.sleep(2)  # pacing between tickers

    # 2. yfinance recent data (up to 730 days)
    print(f"  [YF] Fetching {ticker}...")
    yf_df = yf.download(ticker, period="max", interval="1h", progress=False)
    yf_df = flatten_yfinance_columns(yf_df)
    if yf_df is not None and not yf_df.empty:
        if yf_df.index.tz is None:
            yf_df.index = yf_df.index.tz_localize('UTC')
        else:
            yf_df.index = yf_df.index.tz_convert('UTC')
        frames.append(yf_df)

    if not frames:
        return None

    # Merge: concat, drop duplicate timestamps (keep='last' = prefer yfinance for overlap)
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    print(f"  [MERGED] {ticker}: {len(combined)} bars ({combined.index.min().date()} to {combined.index.max().date()})")
    return combined


def fetch_btc_close(api=None):
    """Fetch BTC-USD hourly close from both sources for cross-asset features."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = fetch_ticker_data(BENCHMARK, api=api)
    if df is None or df.empty:
        return None
    return df['Close']


def prepare_data(ticker, btc_close=None, api=None):
    """Fetch bars, compute features, and add multi-horizon return targets."""
    print(f"Processing {ticker}...")

    df = fetch_ticker_data(ticker, api=api)
    if df is None or df.empty:
        return None

    df = compute_features(df, btc_close=btc_close)

    # Multi-horizon targets: return over N bars as a percentage
    for fb in FORWARD_BARS:
        future_close = df['Close'].shift(-fb)
        df[f'Target_Return_{fb}'] = (future_close - df['Close']) / df['Close'] * 100

    # Backward compat: Target_Return = Target_Return_4
    df['Target_Return'] = df['Target_Return_4']

    df = df.dropna()
    return df


def main():
    api = _get_alpaca_api()
    if api is None:
        print("WARNING: No Alpaca API credentials — using yfinance only (limited to ~730 days)")
    else:
        print("Alpaca API connected — fetching historical data from 2021")

    btc_close = fetch_btc_close(api=api)

    all_data = []
    for t in CRYPTO_TICKERS:
        crypto_df = prepare_data(t, btc_close=btc_close, api=api)
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
    print(f"\nDone! Saved {len(final_df)} rows of training data to training_data.csv")
    print(f"Cryptos harvested: {len(all_data)}/{len(CRYPTO_TICKERS)}")
    target_cols = [c for c in final_df.columns if c.startswith('Target_Return')]
    exclude = set(target_cols) | {'Ticker', 'Date', 'Datetime'}
    feature_count = len([c for c in final_df.columns if c not in exclude])
    print(f"Feature columns: {feature_count}")
    print(f"Target columns: {target_cols}")
    print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")


if __name__ == '__main__':
    main()
