"""Market data fetching and ATR computation.

Provides bar-fetching functions for both Alpaca (crypto + stock) and yfinance,
plus a live ATR helper used by the trading loops for adaptive stop-losses.
"""

import pandas as pd

# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf

from indicators import compute_atr


# --- YFINANCE HELPERS ---

def flatten_yfinance_columns(df):
    """Flatten yfinance MultiIndex columns to single level.

    yfinance >= 0.2.x returns MultiIndex columns like ('Close', 'BTC-USD').
    This collapses them to just ('Close', 'Open', ...).
    No-op if columns are already flat.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# --- ALPACA BAR FETCHING ---

def fetch_bars_alpaca(api, symbol, limit=120):
    """Fetch hourly bars from Alpaca's crypto data API.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca format e.g. 'BTC/USD'
        limit: Max number of bars to fetch

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    try:
        start = datetime.now(timezone.utc) - timedelta(days=6)
        bars = api.get_crypto_bars(symbol, '1Hour', start=start.isoformat(), limit=limit)
        rows = []
        timestamps = []
        for bar in bars:
            rows.append({
                'Open': float(bar.o),
                'High': float(bar.h),
                'Low': float(bar.l),
                'Close': float(bar.c),
                'Volume': float(bar.v),
            })
            timestamps.append(bar.t)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(timestamps)
        df.index.name = 'Datetime'
        return df
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def fetch_bars_yfinance(symbol):
    """Fetch hourly bars from yfinance (standalone/fallback).

    Args:
        symbol: yfinance format e.g. 'BTC-USD'

    Returns:
        DataFrame with OHLCV columns, or None if empty.
    """
    df = yf.download(symbol, period="5d", interval="1h", progress=False)
    if df.empty:
        return None
    return flatten_yfinance_columns(df)


def fetch_stock_bars_alpaca(api, symbol, limit=120):
    """Fetch hourly bars from Alpaca's stock data API.

    Args:
        api: Alpaca REST API object
        symbol: Stock symbol e.g. 'TSLA'
        limit: Max number of bars to fetch

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    try:
        start = datetime.now(timezone.utc) - timedelta(days=6)
        bars = api.get_bars(symbol, '1Hour', start=start.isoformat(), limit=limit)
        rows = []
        timestamps = []
        for bar in bars:
            rows.append({
                'Open': float(bar.o),
                'High': float(bar.h),
                'Low': float(bar.l),
                'Close': float(bar.c),
                'Volume': float(bar.v),
            })
            timestamps.append(bar.t)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(timestamps)
        df.index.name = 'Datetime'
        return df
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def fetch_spy_bars_alpaca(api, limit=120):
    """Fetch SPY hourly bars for relative strength calculation."""
    return fetch_stock_bars_alpaca(api, 'SPY', limit)


# --- ATR ---

def get_live_atr(api, symbol, asset_type='crypto', length=14):
    """Fetch recent bars and compute the latest ATR value.

    Args:
        api: Alpaca API object
        symbol: Alpaca format symbol (e.g. 'BTC/USD' or 'TSLA')
        asset_type: 'crypto' or 'stock'
        length: ATR period (default 14)

    Returns:
        float ATR value, or None on error.
    """
    try:
        if asset_type == 'crypto':
            df = fetch_bars_alpaca(api, symbol, limit=max(60, length * 3))
        else:
            df = fetch_stock_bars_alpaca(api, symbol, limit=max(60, length * 3))

        if df is None or len(df) < length + 1:
            return None

        atr_series = compute_atr(df['High'], df['Low'], df['Close'], length)
        atr_val = atr_series.dropna().iloc[-1]
        return float(atr_val)
    except Exception as e:
        print(f"  [ATR] Error computing ATR for {symbol}: {e}")
        return None
