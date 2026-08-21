"""Backup data sources and fallback chain for historical bar fetching.

Provides CryptoCompare as a third historical source (keyless min-api access
now returns HTTP 401 — CoinDesk sunset it ~2026-07, so this leg is inert in
production until revived with a key; see fetch_cryptocompare_hourly) and
a unified fetch_with_fallback() that tries Alpaca -> yfinance -> CryptoCompare.
"""

import os
import time
import json
import urllib.request
import urllib.error

import pandas as pd


def _yf_window_slice_enabled() -> bool:
    """TRADER_YF_WINDOW_SLICE flag, read at CALL time (default OFF)."""
    return os.environ.get('TRADER_YF_WINDOW_SLICE',
                          '0').strip().lower() in ('1', 'true', 'yes')


# --- CryptoCompare ---

_CC_BASE = 'https://min-api.cryptocompare.com/data/v2/histohour'
_CC_MAX_BARS = 2000  # max per request


def _cc_symbol(ticker: str) -> str:
    """Convert ticker to CryptoCompare format: BTC-USD -> BTC, BTC/USD -> BTC."""
    return ticker.replace('-USD', '').replace('/USD', '').upper()


def fetch_cryptocompare_hourly(symbol: str, start_date: str,
                                end_date: str | None = None) -> pd.DataFrame | None:
    """Fetch hourly bars from CryptoCompare.

    NOTE: the keyless min-api endpoint now returns HTTP 401 (CoinDesk sunset
    keyless access ~2026-07); this fetch catches it, logs, and returns None,
    so the leg is currently a no-op. Revive with a key or remove (owner call).

    Args:
        symbol: Ticker in any format (BTC-USD, BTC/USD, BTC)
        start_date: ISO date string e.g. '2021-01-01'
        end_date: ISO date string or None for now

    Returns:
        DataFrame with OHLCV columns and UTC DatetimeIndex, or None.
    """
    fsym = _cc_symbol(symbol)
    tsym = 'USD'

    start_ts = int(pd.Timestamp(start_date, tz='UTC').timestamp())
    end_ts = int(pd.Timestamp(end_date, tz='UTC').timestamp()) if end_date else int(time.time())

    all_bars = []
    cursor_ts = end_ts

    while cursor_ts > start_ts:
        url = f'{_CC_BASE}?fsym={fsym}&tsym={tsym}&limit={_CC_MAX_BARS}&toTs={cursor_ts}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'trader-bot/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())

            if data.get('Response') != 'Success':
                msg = data.get('Message', 'unknown error')
                print(f"[CC] {fsym}: API error: {msg}")
                break

            bars = data.get('Data', {}).get('Data', [])
            if not bars:
                break

            # Filter bars within our range
            filtered = [b for b in bars if b['time'] >= start_ts and b['close'] > 0]
            all_bars.extend(filtered)

            # Move cursor back
            oldest_ts = min(b['time'] for b in bars)
            if oldest_ts >= cursor_ts:
                break  # no progress
            cursor_ts = oldest_ts - 1

            time.sleep(0.3)  # rate limiting courtesy

        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            print(f"[CC] {fsym}: fetch error: {e}")
            break

    if not all_bars:
        return None

    # Deduplicate by timestamp
    seen = set()
    unique = []
    for b in all_bars:
        if b['time'] not in seen:
            seen.add(b['time'])
            unique.append(b)

    df = pd.DataFrame(unique)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.set_index('timestamp')
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volumefrom': 'Volume',
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.sort_index()

    # Remove zero-price rows (CryptoCompare returns zeros for missing data)
    df = df[df['Close'] > 0]

    print(f"[CC] {fsym}: {len(df)} bars ({df.index.min().date()} to {df.index.max().date()})")
    return df


def fetch_with_fallback(ticker: str, start_date: str, api=None,
                        asset_type: str = 'crypto') -> pd.DataFrame | None:
    """Fetch historical bars with fallback chain.

    Crypto: merges Alpaca + yfinance + CryptoCompare, deduplicated with
    Alpaca > yfinance > CC priority on timestamp collisions.
    Stocks: Alpaca only; yfinance is used exclusively as a fallback when
    Alpaca returns nothing (different bar grid — never merged).

    Args:
        ticker: Ticker symbol (yfinance format for crypto: BTC-USD)
        start_date: ISO date string
        api: Alpaca REST client or None
        asset_type: 'crypto' or 'stock'

    Returns:
        DataFrame with OHLCV columns and UTC DatetimeIndex, or None.
    """
    from market_data import flatten_yfinance_columns, fetch_historical_bars
    import yfinance as yf

    frames = []          # in PRIORITY order: Alpaca, yfinance, CryptoCompare
    source_names = []

    def _to_utc(frame):
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize('UTC')
        else:
            frame.index = frame.index.tz_convert('UTC')
        return frame

    # 1. Alpaca (primary — the venue we trade; split/dividend-adjusted)
    alpaca_ok = False
    if api is not None:
        alpaca_sym = ticker.replace('-', '/') if asset_type == 'crypto' else ticker
        try:
            alpaca_df = fetch_historical_bars(api, alpaca_sym, start_date,
                                              asset_type=asset_type)
            if alpaca_df is not None and not alpaca_df.empty:
                a = _to_utc(alpaca_df)
                a['Src'] = 'alpaca'   # D08 provenance stamp
                frames.append(a)
                source_names.append('Alpaca')
                alpaca_ok = True
        except Exception as e:
            print(f"  [ALPACA] {ticker}: {e}")
        time.sleep(2)

    # 2. yfinance (FALLBACK)
    # Stocks: yfinance hourly bars are :30-aligned (9:30, 10:30, ...) while
    # Alpaca's are :00-aligned — merging both interleaves two bar grids with
    # mixed adjustment conventions, so indicator windows and Target_Return_fb
    # meant different wall-clock spans across the dataset. For stocks,
    # yfinance is used ONLY when Alpaca returned nothing.
    # Crypto: yfinance 1h bars are :00-aligned 24/7, so a true merge is safe.
    # KNOWN HAZARD (model-facing):
    #  (a) yf.download(period='max') IGNORES start_date and returns ~730d of
    #      bars; in incremental harvests (fetch_start = latest-48h) the in-call
    #      keep='first' merge protects only the 48h Alpaca covered, so the
    #      store-level append_ticker_data(keep='last') then overwrites Alpaca
    #      rows with yfinance prices/volume for the trailing 730d.
    #      FIX exists behind TRADER_YF_WINDOW_SLICE (default OFF): slice the
    #      yfinance frame to index >= start_date. Activation requires a full
    #      re-harvest + retrain (gotcha #2) — flip alongside TRADER_RAW_SIDECAR.
    #  (b) the stock :30/:00 grid guard below holds only WITHIN one call — a
    #      full-Alpaca-outage harvest still interleaves grids across runs.
    #  Fixing (b) changes training-data composition -> owner + reharvest.
    if asset_type != 'stock' or not alpaca_ok:
        try:
            print(f"  [YF] Fetching {ticker}...")
            yf_df = yf.download(ticker, period='max', interval='1h',
                                progress=False, auto_adjust=True,
                                prepost=False)
            yf_df = flatten_yfinance_columns(yf_df)
            if yf_df is not None and not yf_df.empty:
                y = _to_utc(yf_df)
                if _yf_window_slice_enabled():
                    # D08 flag fix: honor start_date that period='max'
                    # ignores, so an incremental harvest can no longer feed
                    # ~730d of Yahoo composite into the store-level
                    # keep='last' merge. Needs full re-harvest.
                    y = y[y.index >= pd.Timestamp(start_date, tz='UTC')]
                if not y.empty:
                    y['Src'] = 'yfinance'   # D08 provenance stamp
                    frames.append(y)
                    source_names.append('yfinance')
        except Exception as e:
            print(f"  [YF] {ticker}: {e}")

    # 3. CryptoCompare (tertiary — crypto only, fills gaps; NOT preferred
    # over Alpaca: inference serves Alpaca bars, so training on CC prices
    # where both exist would create train/serve skew)
    if asset_type == 'crypto':
        try:
            cc_df = fetch_cryptocompare_hourly(ticker, start_date)
            if cc_df is not None and not cc_df.empty:
                cc = _to_utc(cc_df)
                cc['Src'] = 'cryptocompare'   # D08 provenance stamp
                frames.append(cc)
                source_names.append('CryptoCompare')
        except Exception as e:
            print(f"  [CC] {ticker}: {e}")

    if not frames:
        print(f"  [DATA] {ticker}: all sources failed")
        return None

    # Merge in PRIORITY order: keep='first' so the highest-priority source
    # wins on timestamp collisions (the old keep='last' inverted this).
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()

    sources = ' + '.join(source_names)
    # Surviving-row composition by provenance (D08): which source's bars
    # actually won the keep='first' collisions.
    comp = ''
    if 'Src' in combined.columns:
        counts = combined['Src'].value_counts().to_dict()
        comp = ' src={' + ', '.join(
            f'{k}:{v}' for k, v in counts.items()) + '}'
    print(f"  [MERGED] {ticker}: {len(combined)} bars from {sources} "
          f"({combined.index.min().date()} to {combined.index.max().date()})"
          f"{comp}")
    return combined
