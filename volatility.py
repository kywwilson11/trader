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
