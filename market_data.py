"""Market data fetching and ATR computation.

Provides bar-fetching functions for both Alpaca (crypto + stock) and yfinance,
plus a live ATR helper used by the trading loops for adaptive stop-losses.
"""

import calendar
import os
import threading
import time

import pandas as pd

# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf

from indicators import compute_atr


def closed_bars_v2_enabled() -> bool:
    """TRADER_CLOSED_BARS_V2 flag, read at CALL time (default OFF).

    Central D38 flag — live closed-bar enforcement. Flipped HERE for the
    panel ranks (compute_live_panel_ranks) and get_live_atr; base_loop's
    GARCH sigma fetches and shadow/decision_report call sites flip via
    their closed_only= parameters (separate handoffs). Model-facing: it
    changes live feature/sizing values -> challenger/shadow path only.
    """
    return os.environ.get('TRADER_CLOSED_BARS_V2',
                          '0').strip().lower() in ('1', 'true', 'yes')


def daily_feature_restore_enabled() -> bool:
    """TRADER_DAILY_FEATURE_RESTORE flag, read at CALL time (default OFF).

    D11 — model-facing: restores REAL live values for the 9 daily-window
    features that are all-NaN on a 45-day live frame and therefore served
    as warmup-fill constants (indicators.DAILY_RESTORE_COLUMNS), fed from
    the daily-bars cache below. OFF = warmup fill exactly as today,
    byte-identical feature values.
    RUNBOOK before enabling (Jetson): bit-parity check — over the last
    ~30 sessions of the harvest window, compute DAILY_RESTORE_COLUMNS via
    (a) the harvest path (compute_stock_features on the full hourly store)
    and (b) indicators.build_daily_restore_features fed by this cache, and
    diff at matching timestamps. Parity holds -> semantics-restoring, no
    retrain needed. Any mismatch -> STOP (likely Alpaca daily-bar OHLC vs
    the harvest's resample('1D') aggregates: official 16:00 close vs last
    extended-hours hourly close; UTC-midnight day boundary). Documented
    fallback: switch the cache to a once-daily ~480-day HOURLY refetch
    aggregated exactly like the harvest (owner decision).
    """
    return os.environ.get('TRADER_DAILY_FEATURE_RESTORE',
                          '0').strip().lower() in ('1', 'true', 'yes')


def har_daily_feed_enabled() -> bool:
    """TRADER_HAR_DAILY_FEED flag, read at CALL time (default OFF).

    D30 — model-facing sizing: feeds volatility.get_sigma's HAR-RV path a
    COMPLETE-day RRV series (stocks from the daily-bars cache below,
    crypto from volatility's persisted per-symbol store), switching sizing
    sigma from GARCH to HAR once history suffices. Deliberately SEPARATE
    from strategy_config.HAR_VOL_ENABLED, which is already True while the
    HAR path is structurally dead (live frames never hold >=60 daily obs)
    — fixing the feed without this new gate would silently activate a
    sizing change. OFF = GARCH fallback exactly as today.
    INTERACTION (c26 final review): with strategy_config.DERISK_STACK_V2 ON
    the per-position vol ratio composes at exactly 1.0 (v2 contract (g)), so
    this feed's sigma reaches ONLY the shadow-journaled vol_mult and the
    reader-less Position.garch_sigma field — run the QLIKE
    certification/flip while DERISK_STACK_V2 is OFF, or accept that under v2
    the HAR feed is measurement-only until a book-vol routing decision
    (owner queue).
    RUNBOOK before enabling (Jetson): run the B11 one-shot offline QLIKE
    gate on the harvest parquet (HAR-with-c-scaling vs current EGARCH, per
    symbol) to convert the certification from literature-backed to
    self-measured. Crypto warm-up: BTC seeds from crypto_rv_history.json;
    other crypto symbols accumulate ~50 further days after their 250-bar
    seed (~10 days) before HAR activates per-symbol — expected, visible in
    the [VOL] sigma-source diagnostic.
    """
    return os.environ.get('TRADER_HAR_DAILY_FEED',
                          '0').strip().lower() in ('1', 'true', 'yes')


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

def fetch_bars_alpaca(api, symbol, limit=250, closed_only=False):
    """Fetch hourly bars from Alpaca's crypto data API.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca format e.g. 'BTC/USD'
        limit: Max number of bars to fetch. Default 250: the 100-bar
            indicator warmups (SMA_100 / Hurst / ATR_Percentile) eat the
            head of the frame, and the old 120-bar fetch left only ~20
            usable rows — one seq_len hyperparameter away from zero
            (the search space allows seq_len up to 40).
        closed_only: drop the trailing in-progress bar (D38). The cache
            stores RAW frames so mixed callers coexist on one key.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    cache_key = ('crypto', symbol, limit)
    cached = _bar_cache_get(cache_key)
    if cached is not None:
        return drop_forming_bar(cached) if closed_only else cached
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
        out = df.copy()
        return drop_forming_bar(out) if closed_only else out
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


def fetch_stock_bars_alpaca(api, symbol, limit=320, closed_only=False):
    """Fetch hourly bars from Alpaca's stock data API.

    Args:
        api: Alpaca REST API object
        symbol: Stock symbol e.g. 'TSLA'
        limit: Max number of bars to fetch
        closed_only: drop the trailing in-progress bar (D38). The cache
            stores RAW frames so mixed callers coexist on one key.

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timedelta, timezone
    cache_key = ('stock', symbol, limit)
    cached = _bar_cache_get(cache_key)
    if cached is not None:
        return drop_forming_bar(cached) if closed_only else cached
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
        out = df.copy()
        return drop_forming_bar(out) if closed_only else out
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def fetch_spy_bars_alpaca(api, limit=320, closed_only=False):
    """Fetch SPY hourly bars for relative strength calculation.

    Default matches fetch_stock_bars_alpaca so the benchmark series covers
    the full stock frame (a shorter SPY series would leave NaN heads in
    RS_vs_SPY after reindex).
    """
    return fetch_stock_bars_alpaca(api, 'SPY', limit, closed_only=closed_only)


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


# --- DAILY BARS CACHE (shared: D11 feature restore + D30 HAR feed) ---
# One on-disk JSON store of COMPLETE daily bars per symbol. Any bar dated
# >= today's UTC date is dropped at refresh time (today's daily bar is
# partial while the market trades — this is the day-level drop_forming_bar):
# the daily feature mapping and the HAR regressors both require completed
# days only. Refreshed at most once per 24h per symbol (refresh_daily_bars,
# triggered ONLY from predict_now's stock fetch branch — get_sigma has no
# api by design), consumed read-only by indicators.apply_daily_restore
# (D11, under TRADER_DAILY_FEATURE_RESTORE) and volatility.get_sigma
# (D30, under TRADER_HAR_DAILY_FEED).

_DAILY_CACHE_FILE = os.path.join(os.path.dirname(__file__),
                                 'daily_bars_cache.json')
_DAILY_CACHE_REFRESH_SEC = 86400
_DAILY_CACHE_FETCH_DAYS = 480   # calendar -> ~320 trading days (B11: 450-500)
_DAILY_CACHE_MAX_ROWS = 340
_daily_cache = {'loaded': False, 'symbols': {}}
_daily_cache_lock = threading.Lock()
_daily_inflight: set = set()   # symbols with a fetch in progress (guarded by _daily_cache_lock)


def _daily_cache_load() -> None:
    """Load the persisted daily-bars store into _daily_cache. Corrupt or
    missing file -> fresh start (one warning on corruption). Never raises.
    Format: {'symbols': {SYM: {'fetched_at': epoch,
                               'bars': {ISO-date: [o, h, l, c, v]}}}}."""
    import json
    syms = {}
    try:
        with open(_DAILY_CACHE_FILE, 'r') as f:
            data = json.load(f)
        for sym, entry in (data.get('symbols') or {}).items():
            bars = {}
            for day, vals in (entry.get('bars') or {}).items():
                try:
                    o, h, l, c, v = (float(x) for x in vals)
                except Exception:
                    continue
                bars[str(day)] = [o, h, l, c, v]
            syms[str(sym)] = {'fetched_at': float(entry.get('fetched_at', 0.0)),
                              'bars': bars}
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  [DAILY-CACHE] cache file corrupt ({e}) — starting empty")
        syms = {}
    _daily_cache['symbols'] = syms
    _daily_cache['loaded'] = True


def _daily_cache_save() -> None:
    """Atomic persist (tmp -> os.replace). Never raises."""
    import json
    try:
        tmp = _DAILY_CACHE_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'symbols': _daily_cache['symbols']}, f)
        os.replace(tmp, _DAILY_CACHE_FILE)
    except Exception as e:
        print(f"  [DAILY-CACHE] save failed: {e}")


def refresh_daily_bars(api, symbol) -> bool:
    """Refresh one symbol's daily-bars cache entry (throttled to 1/24h).

    Fetches ~480 calendar days of '1Day' bars with adjustment='all' — the
    SAME adjustment convention as the harvest and the live hourly fetch, or
    every price-derived daily feature would disagree with the distribution
    the model trained on. Keeps ONLY COMPLETE days (drops any bar dated
    >= today's UTC date). Any failure logs one line, keeps the previous
    entry untouched and returns False (fail-open to current behavior).

    The lock is NOT held across the network fetch — readers
    (load_daily_bars / daily_bars_fetched_at, incl. the stock loop's sizing
    path) must never block behind a slow or hung REST call. A concurrent
    refresh of the same symbol is skipped (returns False, fail-open to the
    previous entry).
    """
    from datetime import datetime, timedelta, timezone
    with _daily_cache_lock:
        if not _daily_cache['loaded']:
            _daily_cache_load()
        entry = _daily_cache['symbols'].get(symbol)
        now = time.time()
        if entry is not None and \
                (now - entry.get('fetched_at', 0.0)) < _DAILY_CACHE_REFRESH_SEC:
            return True
        if symbol in _daily_inflight:
            return False        # another worker is fetching; keep previous entry
        _daily_inflight.add(symbol)
    try:
        start = (datetime.now(timezone.utc)
                 - timedelta(days=_DAILY_CACHE_FETCH_DAYS))
        bars = api.get_bars(symbol, '1Day', start=start.isoformat(),
                            adjustment='all')
        today = datetime.now(timezone.utc).date()
        rows = {}
        for bar in bars:
            t = bar.t
            d = t.date() if hasattr(t, 'date') else t
            if d >= today:
                continue   # forming (or bogus future-dated) daily bar
            # dict assignment = dedup keep-last
            rows[d.isoformat()] = [float(bar.o), float(bar.h),
                                   float(bar.l), float(bar.c),
                                   float(bar.v)]
        if not rows:
            raise ValueError('no complete daily bars returned')
        keep = sorted(rows)[-_DAILY_CACHE_MAX_ROWS:]
        with _daily_cache_lock:
            _daily_cache['symbols'][symbol] = {
                'fetched_at': now,
                'bars': {k: rows[k] for k in keep},
            }
            _daily_cache_save()
        return True
    except Exception as e:
        print(f"  [DAILY-CACHE] {symbol}: refresh failed ({e}) — "
              f"keeping previous entry")
        return False
    finally:
        with _daily_cache_lock:
            _daily_inflight.discard(symbol)


def load_daily_bars(symbol):
    """Read-only cache access (no api — callable from volatility.get_sigma).

    Returns a DataFrame of COMPLETE daily bars (Open/High/Low/Close/Volume,
    tz-aware UTC-midnight DatetimeIndex, ascending), or None if absent/empty.
    """
    with _daily_cache_lock:
        if not _daily_cache['loaded']:
            _daily_cache_load()
        entry = _daily_cache['symbols'].get(symbol)
        if not entry or not entry.get('bars'):
            return None
        bars = dict(entry['bars'])
    try:
        days = sorted(bars)
        df = pd.DataFrame(
            [bars[d] for d in days],
            columns=['Open', 'High', 'Low', 'Close', 'Volume'],
            index=pd.DatetimeIndex([pd.Timestamp(d, tz='UTC') for d in days]),
        )
        df.index.name = 'Datetime'
        return df
    except Exception as e:
        print(f"  [DAILY-CACHE] {symbol}: load failed ({e})")
        return None


def daily_bars_fetched_at(symbol):
    """Epoch timestamp of the symbol's last successful refresh, or None —
    the staleness check consumers apply before trusting the cache."""
    with _daily_cache_lock:
        if not _daily_cache['loaded']:
            _daily_cache_load()
        entry = _daily_cache['symbols'].get(symbol)
    if not entry:
        return None
    try:
        return float(entry['fetched_at'])
    except Exception:
        return None


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
                          chunk_months=6, end_date=None):
    """Fetch historical hourly bars from Alpaca in date-range chunks.

    Breaks the full range into chunks to avoid triggering rate limits
    on the SDK's internal pagination. Adds adaptive pacing between chunks.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca format e.g. 'BTC/USD' or 'TSLA'
        start_date: ISO date string e.g. '2021-01-01'
        asset_type: 'crypto' or 'stock'
        chunk_months: Size of each date chunk in months
        end_date: ISO date string, or None for now — bounded windows let
            the sidecar gap repair refetch just an interior hole (D39)

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex, or None on error.
    """
    from datetime import datetime, timezone

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    end_dt = (datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
              if end_date else now)

    # Build chunk boundaries
    chunks = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        # Advance by chunk_months (clamp day to last day of target month)
        m = chunk_start.month + chunk_months
        y = chunk_start.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        last_day = calendar.monthrange(y, m)[1]
        chunk_end = chunk_start.replace(year=y, month=m,
                                        day=min(chunk_start.day, last_day))
        if chunk_end > end_dt:
            chunk_end = end_dt
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
        closed = closed_bars_v2_enabled()
        if asset_type == 'crypto':
            df = fetch_bars_alpaca(api, symbol, closed_only=closed)
        else:
            df = fetch_stock_bars_alpaca(api, symbol, closed_only=closed)

        if df is None or len(df) < length + 1:
            return None

        atr_series = compute_atr(df['High'], df['Low'], df['Close'], length)
        atr_val = atr_series.dropna().iloc[-1]
        return float(atr_val)
    except Exception as e:
        print(f"  [ATR] Error computing ATR for {symbol}: {e}")
        return None
