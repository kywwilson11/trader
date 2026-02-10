"""Market data fetching and ATR computation.

Provides bar-fetching functions for both Alpaca (crypto + stock) and yfinance,
plus a live ATR helper used by the trading loops for adaptive stop-losses.
"""

import time

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


# --- HISTORICAL BAR FETCHING (for training data harvest) ---

def _fetch_chunk(api, symbol, start_iso, end_iso, asset_type, max_retries=4):
    """Fetch one date-range chunk with exponential backoff.

    Returns list of (row_dict, timestamp) tuples, or None on failure.
    """
    for attempt in range(max_retries):
        try:
            if asset_type == 'crypto':
                bars = api.get_crypto_bars(
                    symbol, '1Hour', start=start_iso, end=end_iso)
            else:
                bars = api.get_bars(
                    symbol, '1Hour', start=start_iso, end=end_iso)

            rows = []
            for bar in bars:
                rows.append(({
                    'Open': float(bar.o), 'High': float(bar.h),
                    'Low': float(bar.l), 'Close': float(bar.c),
                    'Volume': float(bar.v),
                }, bar.t))
            return rows

        except Exception as e:
            err_str = str(e).lower()
            # Subscription errors are permanent — no point retrying
            if 'subscription' in err_str or 'not permit' in err_str:
                return None
            is_rate_limit = ('rate' in err_str or '429' in err_str
                             or 'too many' in err_str)
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 2)  # 4, 8, 16, 32s
                print(f"  [HIST] Rate limited on {symbol} chunk, "
                      f"backoff {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
            elif is_rate_limit:
                print(f"  [HIST] Rate limit exhausted for {symbol} chunk "
                      f"{start_iso[:10]}..{end_iso[:10]}")
                return None
            else:
                print(f"  [HIST] Error fetching {symbol}: {e}")
                return None
    return None


def fetch_historical_bars(api, symbol, start_date, asset_type='crypto',
                          chunk_months=6):
    """Fetch historical hourly bars from Alpaca in date-range chunks.

    Breaks the full range into chunks to avoid triggering rate limits
    on the SDK's internal pagination. Adds adaptive pacing between chunks.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca format e.g. 'BTC/USD' or 'TSLA'
        start_date: ISO date string e.g. '2021-01-01'
        asset_type: 'crypto' or 'stock'
        chunk_months: Size of each date chunk in months

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timezone

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    # Build chunk boundaries
    chunks = []
    chunk_start = start_dt
    while chunk_start < now:
        # Advance by chunk_months
        m = chunk_start.month + chunk_months
        y = chunk_start.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        chunk_end = chunk_start.replace(year=y, month=m)
        if chunk_end > now:
            chunk_end = now
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    all_rows = []
    pace = 0.5  # seconds between chunks, adapts on rate limits

    for i, (c_start, c_end) in enumerate(chunks):
        result = _fetch_chunk(
            api, symbol,
            c_start.isoformat(), c_end.isoformat(),
            asset_type,
        )
        if result is None:
            # Skip retry for the last chunk — likely a subscription limit on
            # recent data; yfinance will cover it
            if i < len(chunks) - 1:
                pace = min(pace * 3, 30)
                print(f"  [HIST] Pacing increased to {pace:.0f}s, retrying chunk...")
                time.sleep(pace)
                result = _fetch_chunk(
                    api, symbol,
                    c_start.isoformat(), c_end.isoformat(),
                    asset_type,
                )
        if result:
            all_rows.extend(result)

        # Adaptive pacing: slow down between chunks
        if i < len(chunks) - 1:
            time.sleep(pace)

    if not all_rows:
        print(f"  [HIST] No bars returned for {symbol}")
        return None

    rows_data = [r[0] for r in all_rows]
    timestamps = [r[1] for r in all_rows]
    df = pd.DataFrame(rows_data)
    df.index = pd.DatetimeIndex(timestamps)
    df.index.name = 'Datetime'
    # Dedup in case chunk boundaries overlap
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    print(f"  [HIST] {symbol}: {len(df)} bars from Alpaca "
          f"({df.index.min().date()} to {df.index.max().date()}) "
          f"[{len(chunks)} chunks]")
    return df


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
