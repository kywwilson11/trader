import pandas as pd
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── Numba-accelerated core functions ─────────────────────────────────────────
# These operate on raw numpy arrays for maximum speed.
# Each has a pure-numpy fallback if Numba is unavailable.

if _HAS_NUMBA:
    @njit(cache=True)
    def _ewm_alpha(arr, alpha, min_periods):
        """EWM with alpha parameter, matching pandas ewm(alpha=..., min_periods=...)."""
        n = len(arr)
        out = np.empty(n)
        out[:] = np.nan
        # Seed with first non-NaN value after min_periods
        s = np.nan
        count = 0
        for i in range(n):
            v = arr[i]
            if np.isnan(v):
                out[i] = np.nan
                continue
            count += 1
            if np.isnan(s):
                s = v
            else:
                s = alpha * v + (1.0 - alpha) * s
            if count >= min_periods:
                out[i] = s
        return out

    @njit(cache=True)
    def _ewm_span(arr, span):
        """EWM with span parameter (adjust=False), matching pandas ewm(span=..., adjust=False)."""
        n = len(arr)
        out = np.empty(n)
        alpha = 2.0 / (span + 1.0)
        out[0] = arr[0]
        for i in range(1, n):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    @njit(cache=True)
    def _rolling_mean(arr, window):
        n = len(arr)
        out = np.empty(n)
        out[:] = np.nan
        # Track consecutive valid count to know when we have a full window
        s = 0.0
        valid_run = 0
        for i in range(n):
            if np.isnan(arr[i]):
                s = 0.0
                valid_run = 0
                continue
            s += arr[i]
            valid_run += 1
            if valid_run > window:
                s -= arr[i - window]
            if valid_run >= window:
                out[i] = s / window
        return out

    @njit(cache=True)
    def _rolling_std(arr, window):
        n = len(arr)
        out = np.empty(n)
        out[:window - 1] = np.nan
        for i in range(window - 1, n):
            s = 0.0
            s2 = 0.0
            for j in range(i - window + 1, i + 1):
                s += arr[j]
                s2 += arr[j] * arr[j]
            mean = s / window
            var = s2 / window - mean * mean
            # Bessel correction (ddof=1) to match pandas default
            out[i] = np.sqrt(var * window / (window - 1)) if window > 1 else 0.0
        return out

    @njit(cache=True)
    def _rolling_min(arr, window):
        n = len(arr)
        out = np.empty(n)
        out[:window - 1] = np.nan
        for i in range(window - 1, n):
            m = arr[i]
            for j in range(i - window + 1, i):
                if arr[j] < m:
                    m = arr[j]
            out[i] = m
        return out

    @njit(cache=True)
    def _rolling_max(arr, window):
        n = len(arr)
        out = np.empty(n)
        out[:window - 1] = np.nan
        for i in range(window - 1, n):
            m = arr[i]
            for j in range(i - window + 1, i):
                if arr[j] > m:
                    m = arr[j]
            out[i] = m
        return out

    @njit(cache=True)
    def _rsi_core(close, length):
        n = len(close)
        rsi = np.empty(n)
        rsi[0] = np.nan

        gain = np.empty(n)
        loss = np.empty(n)
        gain[0] = 0.0
        loss[0] = 0.0
        for i in range(1, n):
            d = close[i] - close[i - 1]
            gain[i] = d if d > 0 else 0.0
            loss[i] = -d if d < 0 else 0.0

        alpha = 1.0 / length
        avg_gain = _ewm_alpha(gain, alpha, length)
        avg_loss = _ewm_alpha(loss, alpha, length)

        for i in range(n):
            if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]) or avg_loss[i] == 0:
                rsi[i] = np.nan
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    @njit(cache=True)
    def _macd_core(close, fast, slow, signal):
        ema_fast = _ewm_span(close, fast)
        ema_slow = _ewm_span(close, slow)
        n = len(close)
        macd_line = np.empty(n)
        for i in range(n):
            macd_line[i] = ema_fast[i] - ema_slow[i]
        signal_line = _ewm_span(macd_line, signal)
        histogram = np.empty(n)
        for i in range(n):
            histogram[i] = macd_line[i] - signal_line[i]
        return macd_line, histogram, signal_line

    @njit(cache=True)
    def _atr_core(high, low, close, length):
        n = len(high)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, max(hc, lc))
        return _rolling_mean(tr, length)

    @njit(cache=True)
    def _bbands_core(close, length, num_std):
        sma = _rolling_mean(close, length)
        std_dev = _rolling_std(close, length)
        n = len(close)
        upper = np.empty(n)
        lower = np.empty(n)
        bandwidth = np.empty(n)
        pct_b = np.empty(n)
        for i in range(n):
            upper[i] = sma[i] + num_std * std_dev[i]
            lower[i] = sma[i] - num_std * std_dev[i]
            if np.isnan(sma[i]) or sma[i] == 0:
                bandwidth[i] = np.nan
            else:
                bandwidth[i] = (upper[i] - lower[i]) / sma[i]
            diff = upper[i] - lower[i]
            if np.isnan(diff) or diff == 0:
                pct_b[i] = np.nan
            else:
                pct_b[i] = (close[i] - lower[i]) / diff
        return lower, sma, upper, bandwidth, pct_b

    @njit(cache=True)
    def _stoch_core(high, low, close, k, d, smooth_k):
        lowest = _rolling_min(low, k)
        highest = _rolling_max(high, k)
        n = len(close)
        raw_k = np.empty(n)
        for i in range(n):
            diff = highest[i] - lowest[i]
            if np.isnan(diff) or diff == 0:
                raw_k[i] = np.nan
            else:
                raw_k[i] = 100.0 * (close[i] - lowest[i]) / diff
        stoch_k = _rolling_mean(raw_k, smooth_k)
        stoch_d = _rolling_mean(stoch_k, d)
        return stoch_k, stoch_d

    @njit(cache=True)
    def _obv_core(close, volume):
        n = len(close)
        out = np.empty(n)
        out[0] = 0.0
        for i in range(1, n):
            d = close[i] - close[i - 1]
            if d > 0:
                out[i] = out[i - 1] + volume[i]
            elif d < 0:
                out[i] = out[i - 1] - volume[i]
            else:
                out[i] = out[i - 1]
        return out

    @njit(cache=True)
    def _rolling_percentile(arr, window):
        """Rolling percentile rank of the last value within its window (0-1)."""
        n = len(arr)
        out = np.empty(n)
        out[:window - 1] = np.nan
        for i in range(window - 1, n):
            val = arr[i]
            count_below = 0
            valid = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(arr[j]):
                    valid += 1
                    if arr[j] < val:
                        count_below += 1
            if valid > 0:
                out[i] = count_below / valid
            else:
                out[i] = np.nan
        return out

    @njit(cache=True)
    def _linear_slope(arr, window):
        """Rolling linear regression slope over window (simple least-squares)."""
        n = len(arr)
        out = np.empty(n)
        out[:window - 1] = np.nan
        # Precompute x values: 0, 1, ..., window-1
        x_mean = (window - 1.0) / 2.0
        ss_x = 0.0
        for k in range(window):
            ss_x += (k - x_mean) * (k - x_mean)
        for i in range(window - 1, n):
            y_mean = 0.0
            for j in range(window):
                y_mean += arr[i - window + 1 + j]
            y_mean /= window
            ss_xy = 0.0
            for j in range(window):
                ss_xy += (j - x_mean) * (arr[i - window + 1 + j] - y_mean)
            out[i] = ss_xy / ss_x if ss_x != 0 else 0.0
        return out


# ── Public API (unchanged signatures) ────────────────────────────────────────

def compute_rsi(series, length=14):
    if _HAS_NUMBA:
        result = _rsi_core(series.values.astype(np.float64), length)
        return pd.Series(result, index=series.index)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    if _HAS_NUMBA:
        ml, hist, sl = _macd_core(series.values.astype(np.float64), fast, slow, signal)
        idx = series.index
        return pd.Series(ml, index=idx), pd.Series(hist, index=idx), pd.Series(sl, index=idx)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


def compute_atr(high, low, close, length=14):
    if _HAS_NUMBA:
        result = _atr_core(high.values.astype(np.float64),
                           low.values.astype(np.float64),
                           close.values.astype(np.float64), length)
        return pd.Series(result, index=close.index)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def compute_bbands(close, length=20, std=2):
    if _HAS_NUMBA:
        lo, mid, up, bw, pb = _bbands_core(close.values.astype(np.float64), length, std)
        idx = close.index
        return (pd.Series(lo, index=idx), pd.Series(mid, index=idx),
                pd.Series(up, index=idx), pd.Series(bw, index=idx),
                pd.Series(pb, index=idx))
    sma = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bandwidth = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)
    return lower, sma, upper, bandwidth, pct_b


def compute_stoch(high, low, close, k=14, d=3, smooth_k=3):
    if _HAS_NUMBA:
        sk, sd = _stoch_core(high.values.astype(np.float64),
                             low.values.astype(np.float64),
                             close.values.astype(np.float64), k, d, smooth_k)
        idx = close.index
        return pd.Series(sk, index=idx), pd.Series(sd, index=idx)
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def compute_obv(close, volume):
    if _HAS_NUMBA:
        result = _obv_core(close.values.astype(np.float64),
                           volume.values.astype(np.float64))
        return pd.Series(result, index=close.index)
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()


def compute_rolling_percentile(series, window=100):
    if _HAS_NUMBA:
        result = _rolling_percentile(series.values.astype(np.float64), window)
        return pd.Series(result, index=series.index)
    # Pure-pandas fallback
    return series.rolling(window).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=False
    )


def compute_linear_slope(series, window=5):
    if _HAS_NUMBA:
        result = _linear_slope(series.values.astype(np.float64), window)
        return pd.Series(result, index=series.index)
    # Pure-pandas fallback
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    ss_x = ((x - x_mean) ** 2).sum()
    def _slope(arr):
        y_mean = arr.mean()
        return ((x - x_mean) * (arr - y_mean)).sum() / ss_x
    return series.rolling(window).apply(_slope, raw=True)


def compute_roc(close, length=12):
    return ((close - close.shift(length)) / close.shift(length)) * 100


def compute_features(df, btc_close=None):
    """Compute all features for crypto (and base features for stocks).

    Args:
        df: DataFrame with OHLCV columns (DatetimeIndex)
        btc_close: Optional Series of BTC/USD close prices for cross-asset features
    """
    df['RSI'] = compute_rsi(df['Close'], length=14)

    macd_line, macd_hist, macd_signal = compute_macd(df['Close'])
    df['MACD_12_26_9'] = macd_line
    df['MACDh_12_26_9'] = macd_hist
    df['MACDs_12_26_9'] = macd_signal

    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], length=14)

    bb_lower, _bb_mid, bb_upper, bb_bw, bb_pct = compute_bbands(df['Close'], length=20, std=2)
    df['BBL_20_2.0'] = bb_lower
    df['BBU_20_2.0'] = bb_upper
    df['BBB_20_2.0'] = bb_bw
    df['BBP_20_2.0'] = bb_pct

    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']

    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    df['ROC'] = compute_roc(df['Close'], length=12)

    stoch_k, stoch_d = compute_stoch(df['High'], df['Low'], df['Close'])
    df['STOCHk_14_3_3'] = stoch_k
    df['STOCHd_14_3_3'] = stoch_d

    idx = df.index
    hour = idx.hour
    day = idx.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_cos'] = np.cos(2 * np.pi * day / 7)

    # --- Return-based features (higher predictive power for multi-hour moves) ---
    df['Return_4h'] = df['Close'].pct_change(4) * 100
    df['Return_12h'] = df['Close'].pct_change(12) * 100
    df['Volatility_12h'] = df['Close'].pct_change(1).rolling(12).std() * 100

    # --- New indicators (Phase C) ---

    # SMA_100 + ratio (longer-term trend)
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['Price_SMA100_Ratio'] = df['Close'] / df['SMA_100']

    # ATR Percentile: rolling 100-period rank of ATR (volatility regime)
    df['ATR_Percentile'] = compute_rolling_percentile(df['ATR'], window=100)

    # RSI Divergence: RSI slope minus price slope (5-period)
    rsi_slope = compute_linear_slope(df['RSI'], window=5)
    price_slope = compute_linear_slope(df['Close'], window=5)
    # Normalize price slope by price level so it's comparable to RSI slope
    df['RSI_Divergence'] = rsi_slope - (price_slope / df['Close'] * 100)

    # Volume-Price Confirmation: sign agreement between OBV slope and price slope
    obv_slope = compute_linear_slope(df['OBV'], window=5)
    obv_sign = np.sign(obv_slope)
    price_sign = np.sign(price_slope)
    df['Vol_Price_Confirm'] = obv_sign * price_sign  # +1=confirm, -1=diverge

    # BTC cross-asset features (crypto only)
    if btc_close is not None:
        btc_aligned = btc_close.reindex(df.index, method='ffill')
        df['BTC_Return_1h'] = btc_aligned.pct_change(1)
        btc_sma20 = btc_aligned.rolling(window=20).mean()
        df['BTC_SMA_Ratio'] = btc_aligned / btc_sma20
        df['BTC_RSI'] = compute_rsi(btc_aligned, length=14)

    return df


# --- STOCK-SPECIFIC INDICATORS ---

def compute_vwap(high, low, close, volume):
    """Session VWAP — resets each trading day.
    Returns a Series with intraday VWAP values.
    """
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume

    # Group by date for daily reset
    dates = close.index.date
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = volume.groupby(dates).cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap


def compute_gap(open_price, close):
    """Overnight gap: (today's open - yesterday's close) / yesterday's close * 100.
    Returns a Series with gap % for each bar (only meaningful on first bar of day).
    """
    prev_close = close.shift(1)
    gap = (open_price - prev_close) / prev_close * 100
    return gap


def compute_relative_strength(close, benchmark_close):
    """Relative strength vs benchmark (e.g. SPY).
    RS = stock ROC / benchmark ROC over a rolling window.
    Returns ratio > 1 means outperforming, < 1 means underperforming.
    """
    stock_roc = close.pct_change(12)
    bench_roc = benchmark_close.pct_change(12)
    # Avoid division by zero
    rs = stock_roc / bench_roc.replace(0, np.nan)
    return rs


def compute_normalized_atr(high, low, close, length=14):
    """ATR normalized by price: ATR / close * 100.
    Gives volatility as a percentage, comparable across different price levels.
    """
    atr = compute_atr(high, low, close, length)
    return atr / close * 100


def compute_stock_features(df, spy_close=None):
    """Compute all features for stocks. Includes base crypto features + stock-specific ones.

    Args:
        df: DataFrame with OHLCV columns (DatetimeIndex)
        spy_close: Optional Series of SPY close prices aligned to df's index
                   for relative strength calculation
    """
    # Base features (same as crypto)
    df = compute_features(df)

    # VWAP
    df['VWAP'] = compute_vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['Price_VWAP_Ratio'] = df['Close'] / df['VWAP']

    # Overnight gap
    df['Gap_Pct'] = compute_gap(df['Open'], df['Close'])

    # Normalized ATR (volatility as % of price)
    df['ATR_Pct'] = compute_normalized_atr(df['High'], df['Low'], df['Close'])

    # Relative strength vs SPY
    if spy_close is not None:
        # Align SPY close to df's index
        spy_aligned = spy_close.reindex(df.index, method='ffill')
        df['RS_vs_SPY'] = compute_relative_strength(df['Close'], spy_aligned)
    else:
        df['RS_vs_SPY'] = 1.0  # neutral if no benchmark

    return df
