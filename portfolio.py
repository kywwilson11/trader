"""Correlation-aware portfolio management.

Prevents over-concentrated positions by checking pairwise correlation
before adding new positions. Based on Markowitz (1990 Nobel) portfolio theory.
"""

import time
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)

# Cache
_corr_cache: tuple[dict, float] | None = None
_CORR_CACHE_TTL = 3600  # 1 hour

MAX_AVG_CORRELATION = 0.7  # reject if avg pairwise > this


def compute_correlation_matrix(returns_dict: dict[str, np.ndarray],
                               window: int = 30) -> dict[tuple[str, str], float]:
    """Compute pairwise correlation from recent returns.

    Args:
        returns_dict: {symbol: array_of_returns}
        window: Number of recent bars to use

    Returns:
        Dict mapping (sym1, sym2) -> correlation coefficient
    """
    symbols = sorted(returns_dict.keys())
    corr = {}

    for i, s1 in enumerate(symbols):
        r1 = returns_dict[s1][-window:]
        for j, s2 in enumerate(symbols):
            if j <= i:
                continue
            r2 = returns_dict[s2][-window:]
            min_len = min(len(r1), len(r2))
            if min_len < 10:
                corr[(s1, s2)] = 0.0
                continue
            c = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
            if np.isnan(c):
                c = 0.0
            corr[(s1, s2)] = c
            corr[(s2, s1)] = c

    return corr


def get_returns_for_symbols(api, symbols: list[str],
                            asset_type: str = 'crypto') -> dict[str, np.ndarray]:
    """Fetch recent returns for a list of symbols.

    Args:
        api: Alpaca API client
        symbols: List of symbols
        asset_type: 'crypto' or 'stock'

    Returns:
        {symbol: array_of_pct_returns}
    """
    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca

    returns_dict = {}
    for sym in symbols:
        try:
            if asset_type == 'crypto':
                df = fetch_bars_alpaca(api, sym)
            else:
                df = fetch_stock_bars_alpaca(api, sym)
            if df is not None and len(df) > 10:
                rets = df['Close'].pct_change().dropna().values
                returns_dict[sym] = rets
        except Exception as e:
            logger.debug("Correlation: failed to fetch %s: %s", sym, e)

    return returns_dict


def check_portfolio_correlation(current_positions: list[str],
                                candidate: str,
                                corr_matrix: dict[tuple[str, str], float],
                                max_avg_corr: float = MAX_AVG_CORRELATION) -> tuple[bool, float]:
    """Check if adding a candidate would push avg pairwise correlation too high.

    Args:
        current_positions: List of currently held symbols
        candidate: Symbol we want to add
        corr_matrix: Precomputed correlation dict
        max_avg_corr: Maximum allowed average pairwise correlation

    Returns:
        (allowed: bool, avg_corr: float)
    """
    if not current_positions:
        return True, 0.0

    correlations = []
    for sym in current_positions:
        pair = (sym, candidate)
        alt_pair = (candidate, sym)
        c = corr_matrix.get(pair, corr_matrix.get(alt_pair, 0.0))
        correlations.append(abs(c))

    avg_corr = np.mean(correlations) if correlations else 0.0

    if avg_corr > max_avg_corr:
        logger.info("[PORTFOLIO] %s rejected: avg corr %.2f > %.2f with %s",
                    candidate, avg_corr, max_avg_corr,
                    ', '.join(current_positions))
        return False, avg_corr

    return True, avg_corr


def get_correlation_sizing_factor(candidate: str,
                                  current_positions: list[str],
                                  corr_matrix: dict[tuple[str, str], float]) -> float:
    """Reduce sizing proportionally to correlation with existing positions.

    Returns multiplier between 0.5 (highly correlated) and 1.0 (uncorrelated).
    """
    if not current_positions:
        return 1.0

    correlations = []
    for sym in current_positions:
        pair = (sym, candidate)
        alt_pair = (candidate, sym)
        c = corr_matrix.get(pair, corr_matrix.get(alt_pair, 0.0))
        correlations.append(abs(c))

    avg_corr = np.mean(correlations) if correlations else 0.0
    # Linear scale: corr=0 → 1.0x, corr=0.7 → 0.65x, corr=1.0 → 0.5x
    factor = max(0.5, 1.0 - 0.5 * avg_corr)
    return factor
