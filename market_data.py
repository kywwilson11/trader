"""Market data fetching and ATR computation.

Provides bar-fetching functions for both Alpaca (crypto + stock) and yfinance,
plus a live ATR helper used by the trading loops for adaptive stop-losses.
"""

import calendar
import threading
import time

import pandas as pd

# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf

from indicators import compute_atr


# --- LIVE BAR CACHE ---
# One trading cycle re-uses the same bars for prediction, sizing (GARCH),
# ATR, and post-fill bookkeeping. A short TTL cache collapses those 3-4
# REST calls per symbol per cycle into one.

BAR_CACHE_TTL = 20.0  # seconds — shorter than the 30s loop interval
_bar_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_bar_cache_lock = threading.Lock()


def _bar_cache_get(key):
    with _bar_cache_lock:
        hit = _bar_cache.get(key)
    if hit is not None and (time.monotonic() - hit[0]) < BAR_CACHE_TTL:
        # Copy so callers adding indicator columns don't pollute the cache
        return hit[1].copy()
    return None


def _bar_cache_put(key, df):
    with _bar_cache_lock:
        _bar_cache[key] = (time.monotonic(), df)
        if len(_bar_cache) > 256:  # bound the cache
            cutoff = time.monotonic() - BAR_CACHE_TTL
            for k in [k for k, v in _bar_cache.items() if v[0] < cutoff]:
                del _bar_cache[k]


def _filter_bad_prints(df):
    """Drop bars whose Close is wildly off its local neighborhood.

    Brownlees-Gallo (2006)-style cleaning: flag a bar when its close is
    more than 6 robust sigmas (MAD-scaled) away from the rolling 11-bar
    median, with a 1% absolute floor so quiet series don't over-flag.
    One bad print otherwise inflates ATR for days (wrong stop distances)
    and can fire stops/features off a price that never traded in size.
    Real crashes survive: a -15% hourly move shifts the rolling median
    with it across consecutive bars; an isolated wick does not.
    """
    if df is None or len(df) < 15:
        return df
    close = df['Close']
    med = close.rolling(11, center=True, min_periods=5).median()
    mad = (close - med).abs().rolling(11, center=True, min_periods=5).median()
    sigma = 1.4826 * mad
    floor = med * 0.01
    dev = (close - med).abs()
    bad = dev > (6 * sigma).clip(lower=floor)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"  [BARS] Dropped {n_bad} outlier bar(s) (bad prints)")
        return df[~bad]
    return df


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


# --- CRYPTO VOLUME (supplements Alpaca crypto which reports zero volume) ---

def fetch_crypto_volume(symbols: list[str], limit: int = 24) -> dict[str, float]:
    """Fetch volume ratios from CryptoCompare for crypto symbols.

    Returns dict mapping symbol -> volume_ratio (last completed bar vol / 20-bar avg).
    Uses CryptoCompare public API — no auth needed, no geo-block.
    """
    import urllib.request
    import json as _json

    result = {}
    for symbol in symbols:
        try:
            fsym = symbol.split("/")[0] if "/" in symbol else symbol.replace("USD", "")
            url = (f"https://min-api.cryptocompare.com/data/v2/histohour"
                   f"?fsym={fsym}&tsym=USD&limit={max(limit, 21)}")
            req = urllib.request.Request(url, headers={"User-Agent": "trader/1.0"})
            resp = urllib.request.urlopen(req, timeout=5)
            data = _json.loads(resp.read())
            bars = data.get("Data", {}).get("Data", [])
            if len(bars) < 2:
                continue
            # volumeto = USD-denominated volume (aggregated across exchanges)
            volumes = [b.get("volumeto", 0) for b in bars]
            # Last bar is in-progress; use second-to-last as "current completed"
            current_vol = volumes[-2]
            # 20-bar average excluding the last 2
            avg_window = volumes[:-2][-20:] if len(volumes) > 22 else volumes[:-2]
            avg_vol = sum(avg_window) / len(avg_window) if avg_window else 1.0
            ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
            result[symbol] = round(ratio, 2)
        except Exception as e:
            print(f"  [VOLUME] {symbol}: {e}")
    return result


# --- ALPACA BAR FETCHING ---

def fetch_bars_alpaca(api, symbol, limit=250):
    """Fetch hourly bars from Alpaca's crypto data API.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca format e.g. 'BTC/USD'
        limit: Max number of bars to fetch. Default 250: the 100-bar
            indicator warmups (SMA_100 / Hurst / ATR_Percentile) eat the
            head of the frame, and the old 120-bar fetch left only ~20
            usable rows — one seq_len hyperparameter away from zero
            (the search space allows seq_len up to 40).

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    cache_key = ('crypto', symbol, limit)
    cached = _bar_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        # Fetch a wider window than `limit` and keep the NEWEST bars.
        # Alpaca returns bars ascending from `start`, so passing limit= to the
        # API would truncate to the OLDEST bars and serve ~24h-stale data.
        start = datetime.now(timezone.utc) - timedelta(hours=limit + 24)
        bars = api.get_crypto_bars(symbol, '1Hour', start=start.isoformat())
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
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = _filter_bad_prints(df).tail(limit)
        _bar_cache_put(cache_key, df)
        return df.copy()
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
    # 60d (not 5d): the 100-bar hourly warmups plus the daily-window stock
    # features need real history or every row dies in the feature dropna.
    df = yf.download(symbol, period="60d", interval="1h", progress=False)
    if df.empty:
        return None
    return flatten_yfinance_columns(df)


def fetch_stock_bars_alpaca(api, symbol, limit=320):
    """Fetch hourly bars from Alpaca's stock data API.

    Args:
        api: Alpaca REST API object
        symbol: Stock symbol e.g. 'TSLA'
        limit: Max number of bars to fetch

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    cache_key = ('stock', symbol, limit)
    cached = _bar_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        # 45 days ≈ 220 market-hours bars: the 100-bar hourly warmups
        # (SMA_100 / Hurst / ATR_Percentile) consume the head, and ~30 days
        # left almost no post-warmup rows. The LONG-window daily features
        # (MA_Dist_200d, RM_252_21, ...) can never warm up on a live frame —
        # they are neutral-filled via indicators.fill_warmup_features, the
        # exact fill the harvest applies, so train/serve stays consistent.
        # Do NOT pass limit= to the API: bars come back ascending, so the API
        # would truncate to the OLDEST bars and serve days-stale data.
        start = datetime.now(timezone.utc) - timedelta(days=45)
        # adjustment='all', matching the harvest (_fetch_chunk below): the SDK
        # default is RAW bars, so any split/dividend ex-date inside the live
        # window made every price-derived feature (returns, SMA, RSI, ATR,
        # GARCH inputs) disagree with the adjusted distribution the model
        # trained on — a dividend drifts large caps every quarter; a split
        # makes the symbol's features garbage for weeks (2026-07 review P1).
        bars = api.get_bars(symbol, '1Hour', start=start.isoformat(),
                            adjustment='all')
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
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = _filter_bad_prints(df).tail(limit)
        _bar_cache_put(cache_key, df)
        return df.copy()
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def fetch_spy_bars_alpaca(api, limit=320):
    """Fetch SPY hourly bars for relative strength calculation.

    Default matches fetch_stock_bars_alpaca so the benchmark series covers
    the full stock frame (a shorter SPY series would leave NaN heads in
    RS_vs_SPY after reindex).
    """
    return fetch_stock_bars_alpaca(api, 'SPY', limit)


def drop_forming_bar(df, bar_seconds=3600):
    """Drop the trailing IN-PROGRESS bar from a fetched hourly frame.

    Alpaca labels an hourly bar with its OPEN time and serves it while it is
    still forming (partial volume, minutes-old close). Training windows hold
    CLOSED bars only (window offsets -seq_len..-1 exclude the entry bar) and
    the backtest enters at the signal bar's close — so live inference must see
    closed bars only, or the final sequence row is drawn from a distribution
    the model never trained on. Bars older than one bar-length are closed by
    definition (stock frames after hours / weekends lose nothing).
    """
    if df is None or len(df) == 0:
        return df
    from datetime import datetime, timedelta, timezone
    last = df.index[-1]
    try:
        last_py = last.to_pydatetime()
    except AttributeError:
        last_py = last
    if last_py.tzinfo is None:
        last_py = last_py.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    if now < last_py + timedelta(seconds=bar_seconds):
        return df.iloc[:-1]
    return df


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
                # adjustment='all': Alpaca's default is RAW bars — an
                # unadjusted 10:1 split would inject a fake -90% move into
                # the training labels.
                bars = api.get_bars(
                    symbol, '1Hour', start=start_iso, end=end_iso,
                    adjustment='all')

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
        # Advance by chunk_months (clamp day to last day of target month)
        m = chunk_start.month + chunk_months
        y = chunk_start.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        last_day = calendar.monthrange(y, m)[1]
        chunk_end = chunk_start.replace(year=y, month=m,
                                        day=min(chunk_start.day, last_day))
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
        # Use the default fetch limit so this hits the bar cache populated by
        # the prediction pass instead of issuing a second REST call.
        if asset_type == 'crypto':
            df = fetch_bars_alpaca(api, symbol)
        else:
            df = fetch_stock_bars_alpaca(api, symbol)

        if df is None or len(df) < length + 1:
            return None

        atr_series = compute_atr(df['High'], df['Low'], df['Close'], length)
        atr_val = atr_series.dropna().iloc[-1]
        return float(atr_val)
    except Exception as e:
        print(f"  [ATR] Error computing ATR for {symbol}: {e}")
        return None
