"""GARCH(1,1) volatility forecasting for adaptive stop-loss and position sizing.

Provides forward-looking volatility estimates that adapt to changing market
conditions, unlike ATR which is purely backward-looking. Falls back to ATR
when GARCH fitting fails (short series, convergence issues).

Based on Engle (2003 Nobel) ARCH/GARCH framework.
"""

import time
from log_config import get_logger
import numpy as np

logger = get_logger(__name__)

# Cache fitted models per symbol (refit hourly, not every 30s cycle)
_model_cache: dict[str, tuple[object, float]] = {}
_REFIT_INTERVAL = 3600  # seconds


def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1):
    """Fit a GARCH(p,q) model to a return series.

    Args:
        returns: Array of percentage returns (e.g. daily or hourly)
        p: GARCH lag order (default 1)
        q: ARCH lag order (default 1)

    Returns:
        Fitted arch model result, or None on failure.
    """
    if len(returns) < 100:
        logger.debug("GARCH: insufficient data (%d points, need 100)", len(returns))
        return None

    try:
        from arch import arch_model
        # Scale returns to percentage if they look like decimals
        if np.abs(returns).mean() < 0.01:
            returns = returns * 100

        # Try EGARCH first (captures asymmetry: crashes increase vol more than rallies)
        try:
            am = arch_model(returns, vol='EGARCH', p=p, q=q, mean='Zero',
                            rescale=False)
            result = am.fit(disp='off', show_warning=False)
            return result
        except Exception:
            pass
        # Fallback to standard GARCH
        am = arch_model(returns, vol='Garch', p=p, q=q, mean='Zero',
                        rescale=False)
        result = am.fit(disp='off', show_warning=False)
        return result
    except Exception as e:
        logger.debug("GARCH fit failed: %s", e)
        return None


def forecast_volatility(model_result) -> float | None:
    """Forecast 1-step-ahead conditional volatility (sigma).

    Returns:
        Annualized sigma as a decimal (e.g. 0.25 = 25% vol), or None.
    """
    if model_result is None:
        return None
    try:
        forecasts = model_result.forecast(horizon=1)
        variance = forecasts.variance.values[-1, 0]
        if variance <= 0:
            return None
        # Model was fit on percentage returns, so sigma is in percentage points
        sigma_pct = np.sqrt(variance)
        return sigma_pct / 100.0  # Convert to decimal
    except Exception as e:
        logger.debug("GARCH forecast failed: %s", e)
        return None


def get_garch_stop(entry_price: float, sigma: float, multiplier: float = 2.0,
                   floor_pct: float = 0.03, ceil_pct: float = 0.10) -> float:
    """Compute stop-loss price using GARCH volatility.

    Args:
        entry_price: Entry price
        sigma: GARCH sigma (decimal, e.g. 0.02 = 2%)
        multiplier: Number of sigmas for stop distance
        floor_pct: Minimum stop distance as fraction of price
        ceil_pct: Maximum stop distance as fraction of price

    Returns:
        Stop price (below entry for long positions).
    """
    stop_dist = max(floor_pct, min(ceil_pct, sigma * multiplier))
    return entry_price * (1 - stop_dist)


def get_cached_sigma(symbol: str, returns: np.ndarray) -> float | None:
    """Get GARCH sigma for a symbol, using cache to avoid refitting every cycle.

    Args:
        symbol: Trading symbol
        returns: Recent return series (percentage)

    Returns:
        Sigma (decimal) or None if fitting fails.
    """
    now = time.time()

    if symbol in _model_cache:
        cached_result, cached_time = _model_cache[symbol]
        if now - cached_time < _REFIT_INTERVAL:
            sigma = forecast_volatility(cached_result)
            if sigma is not None:
                return sigma

    # Fit new model
    result = fit_garch(returns)
    if result is not None:
        _model_cache[symbol] = (result, now)
        sigma = forecast_volatility(result)
        if sigma is not None:
            logger.debug("GARCH %s: sigma=%.4f", symbol, sigma)
            return sigma

    return None


BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
BARS_PER_DAY = {'crypto': 24.0, 'stock': 6.5}


# --- HAR-RV on realized range (Corsi 2009; HARQ insanity filter from
# Bollerslev-Patton-Quaedvlieg 2016). Wave-4, red-team conf 5/5:
# realized measures OBSERVE variance from intraday data instead of
# inferring it by MLE from daily closes — a 3-regressor OLS then beats
# GARCH(1,1) on RV forecasting near-universally (Hansen-Lunde's "nothing
# beats GARCH" verdict only holds for close-to-close data). ---

_HAR_MIN_DAYS = 60
_HAR_WINDOW = 250
_har_cache: dict[str, tuple[object, float]] = {}   # symbol -> (day, sigma)


def daily_realized_range(bars) -> 'pd.Series':
    """Daily Parkinson realized variance from intraday High/Low bars:
    RRV_d = sum_i ln(H_i/L_i)^2 / (4 ln 2), in squared-decimal units."""
    import pandas as pd
    hl = np.log(bars['High'] / bars['Low']) ** 2 / (4.0 * np.log(2.0))
    hl = hl.replace([np.inf, -np.inf], np.nan).dropna()
    return hl.groupby(hl.index.normalize()).sum()


def har_forecast_sigma(bars, asset_type: str = 'stock') -> float | None:
    """Next-day HAR-RV sigma forecast as a PER-BAR decimal (the same
    units forecast_volatility returns), or None when history is thin.

    log RRV_{d+1} ~ c + log RRV_d + log mean(RRV,5d) + log mean(RRV,22d)
    fit by rolling OLS over <=250 days; log-normal bias correction;
    forecast clamped to the estimation window's [min, max] (the BPQ
    'insanity filter' — one bad range print must not triple the stop).
    """
    import pandas as pd
    try:
        rrv = daily_realized_range(bars)
        rrv = rrv[rrv > 0].tail(_HAR_WINDOW)
        if len(rrv) < _HAR_MIN_DAYS:
            return None
        m5 = rrv.rolling(5, min_periods=5).mean()
        m22 = rrv.rolling(22, min_periods=22).mean()
        df = pd.DataFrame({'y': np.log(rrv).shift(-1), 'x1': np.log(rrv),
                           'x2': np.log(m5), 'x3': np.log(m22)}).dropna()
        if len(df) < _HAR_MIN_DAYS - 22:
            return None
        X = np.column_stack([np.ones(len(df)), df['x1'], df['x2'], df['x3']])
        beta, *_ = np.linalg.lstsq(X, df['y'].values, rcond=None)
        resid = df['y'].values - X @ beta
        sig2_resid = float(np.var(resid, ddof=4)) if len(df) > 8 else 0.0
        x_now = np.array([1.0, np.log(rrv.iloc[-1]),
                          np.log(m5.iloc[-1]), np.log(m22.iloc[-1])])
        rrv_hat = float(np.exp(x_now @ beta + 0.5 * sig2_resid))
        rrv_hat = min(max(rrv_hat, float(rrv.min())), float(rrv.max()))
        sigma_daily = np.sqrt(rrv_hat)
        return float(sigma_daily / np.sqrt(BARS_PER_DAY.get(asset_type, 6.5)))
    except Exception as e:
        logger.debug("HAR forecast failed: %s", e)
        return None


def get_sigma(symbol: str, returns: np.ndarray, bars=None,
              asset_type: str = 'stock') -> float | None:
    """Per-bar sigma: HAR-RV from intraday ranges when enough bars
    exist, EGARCH/GARCH on returns otherwise. One day of cache per
    symbol (daily RRV only changes at the day roll)."""
    try:
        from strategy_config import HAR_VOL_ENABLED
    except ImportError:
        HAR_VOL_ENABLED = True
    if HAR_VOL_ENABLED and bars is not None and len(bars) > 0:
        day = bars.index[-1].date() if hasattr(bars.index[-1], 'date') else None
        hit = _har_cache.get(symbol)
        if hit is not None and hit[0] == day:
            return hit[1]
        sigma = har_forecast_sigma(bars, asset_type)
        if sigma is not None and sigma > 0:
            _har_cache[symbol] = (day, sigma)
            logger.debug("HAR %s: sigma=%.4f", symbol, sigma)
            return sigma
    return get_cached_sigma(symbol, returns)


def compute_vol_adjusted_size(base_notional: float, sigma: float,
                              asset_type: str = 'crypto') -> float:
    """Volatility-targeted position sizing (Moreira & Muir 2017).

    Adjusts notional so each position targets approximately the same
    dollar volatility. Higher vol assets get smaller positions.

    The target is derived from the ANNUALIZED portfolio vol target in
    strategy_config, converted to per-bar units. The old code compared a
    2% PER-HOURLY-BAR target (≈ 187%/yr crypto!) against the 1-step
    hourly sigma — typically <1% — so the "vol target" silently doubled
    nearly every position via the 2.0x clamp.

    Args:
        base_notional: Base dollar amount per trade
        sigma: GARCH 1-bar-ahead sigma (decimal, hourly)
        asset_type: 'crypto' or 'stock'

    Returns:
        Adjusted notional (clamped to 0.5x - 1.5x base).
    """
    if sigma <= 0:
        return base_notional
    from strategy_config import PORTFOLIO_VOL_TARGET
    annual_target = PORTFOLIO_VOL_TARGET.get(asset_type, 0.25)
    target_per_bar = annual_target / np.sqrt(BARS_PER_YEAR.get(asset_type, 8760))
    ratio = target_per_bar / sigma
    ratio = max(0.5, min(1.5, ratio))
    return base_notional * ratio
