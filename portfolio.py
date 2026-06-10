"""Correlation-aware portfolio management.

Prevents over-concentrated positions by checking pairwise correlation
before adding new positions. Based on Markowitz (1990 Nobel) portfolio theory.
"""

import time
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)

# Cache (per asset class — the combined bot runner shares this module
# between the crypto and stock threads)
_corr_cache: dict[str, tuple[dict, float]] = {}
_CORR_CACHE_TTL = 3600  # 1 hour

MAX_AVG_CORRELATION = 0.7  # reject if avg pairwise > this


def get_correlation_matrix_cached(api, symbols, asset_type='crypto'):
    """Correlation matrix with a 1h TTL cache.

    Pairwise correlations over 30 bars barely move cycle to cycle; the old
    path serially refetched the whole universe and re-ran ~1,000
    LedoitWolf fits every 10th cycle while this cache sat unused.
    """
    now = time.monotonic()
    hit = _corr_cache.get(asset_type)
    if hit is not None and (now - hit[1]) < _CORR_CACHE_TTL:
        return hit[0]
    returns_dict = get_returns_for_symbols(api, symbols, asset_type)
    corr = compute_correlation_matrix(returns_dict) if returns_dict else {}
    if corr:
        _corr_cache[asset_type] = (corr, now)
    return corr


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
            try:
                from sklearn.covariance import LedoitWolf
                X = np.column_stack([r1[-min_len:], r2[-min_len:]])
                lw = LedoitWolf().fit(X)
                cov = lw.covariance_
                std = np.sqrt(np.diag(cov))
                c = cov[0, 1] / (std[0] * std[1]) if std[0] > 0 and std[1] > 0 else 0.0
            except Exception:
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
    """Variance-correct sizing factor: 1/sqrt(1 + n*rho_bar).

    A candidate entering a book of n positions at average |corr| rho_bar
    contributes marginal variance ~ (1 + n*rho_bar) times its standalone
    variance; scaling size by the inverse square root keeps its RISK
    contribution constant as the book fills up. The old linear heuristic
    (1 - 0.5*corr) ignored n entirely — the 5th BTC-clone got the same
    haircut as the 1st. Floored at 0.4.
    """
    if not current_positions:
        return 1.0

    correlations = []
    for sym in current_positions:
        pair = (sym, candidate)
        alt_pair = (candidate, sym)
        c = corr_matrix.get(pair, corr_matrix.get(alt_pair, 0.0))
        correlations.append(abs(c))

    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    n = len(current_positions)
    return max(0.4, 1.0 / float(np.sqrt(1.0 + n * avg_corr)))


def avg_book_correlation(symbols: list[str],
                         corr_matrix: dict[tuple[str, str], float]) -> float:
    """Average absolute pairwise correlation across a set of symbols."""
    if len(symbols) < 2 or not corr_matrix:
        return 0.0
    vals = []
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1:]:
            c = corr_matrix.get((s1, s2), corr_matrix.get((s2, s1)))
            if c is not None:
                vals.append(abs(c))
    return float(np.mean(vals)) if vals else 0.0


def diversified_book_risk(risks: list[float], avg_corr: float) -> float:
    """Book stop-risk under the equicorrelation model.

    With per-position stop-risks r_i (fractions of equity) and constant
    pairwise correlation rho, portfolio risk is exactly
        sqrt((1-rho) * sum(r_i^2) + rho * (sum r_i)^2)
    — between sqrt-sum-of-squares (independent book) and the plain sum
    (lockstep book). This is the effective-number-of-bets view:
    correlated books get less headroom for the same gross risk.
    """
    r = np.asarray([x for x in risks if x and x > 0], dtype=float)
    if r.size == 0:
        return 0.0
    rho = min(max(float(avg_corr), 0.0), 1.0)
    return float(np.sqrt(max(
        (1.0 - rho) * np.sum(r ** 2) + rho * np.sum(r) ** 2, 0.0)))


def book_risk_budget(existing_risks: list[float], avg_corr: float,
                     cap: float) -> float:
    """Max stop-risk (fraction of equity) a NEW position may add.

    Solves diversified_book_risk(existing + [r_c]) = cap for r_c under
    equicorrelation: with S1 = sum(r_i) and A = current book risk^2,
        r_c = -rho*S1 + sqrt(rho^2*S1^2 + cap^2 - A)
    Returns 0 when the cap is already used up.
    """
    rho = min(max(float(avg_corr), 0.0), 1.0)
    r = np.asarray([x for x in existing_risks if x and x > 0], dtype=float)
    s1 = float(np.sum(r))
    a = (1.0 - rho) * float(np.sum(r ** 2)) + rho * s1 ** 2  # book risk^2
    if cap ** 2 <= a:
        return 0.0
    return float(-rho * s1 + np.sqrt(rho ** 2 * s1 ** 2 + cap ** 2 - a))


# --- Book-level realized-vol scalar (closes the loop that per-asset GARCH
# targeting leaves open: correlation buildup raises BOOK vol even when each
# position individually sits at its own target) ---

_book_vol_cache: dict[str, tuple[float, float]] = {}
_BOOK_VOL_TTL = 3600

EWMA_LAMBDA = 0.94          # RiskMetrics daily decay
_TRADING_DAYS = 252


def ewma_annualized_vol(equity_curve, lam: float = EWMA_LAMBDA) -> float | None:
    """Annualized EWMA volatility of a daily equity curve, or None."""
    eq = np.asarray([e for e in (equity_curve or []) if e], dtype=float)
    if eq.size < 11:
        return None
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size < 10:
        return None
    var = float(np.var(rets))
    for x in rets:
        var = lam * var + (1.0 - lam) * x * x
    return float(np.sqrt(var * _TRADING_DAYS))


def get_book_vol_scalar_cached(api, asset_type: str = 'crypto') -> float:
    """De-risk-only sizing scalar from REALIZED account vol (EWMA λ=0.94).

    Moreira-Muir (2017): volatility-managed sizing adds value mostly by
    CUTTING exposure when realized vol runs hot — so this clamps to
    [0.5, 1.0] and never boosts (a mostly-cash paper account shows tiny
    realized vol; boosting on that signal would be spurious). Neutral 1.0
    on any data failure. Cached 1h per asset class.
    """
    now = time.monotonic()
    hit = _book_vol_cache.get(asset_type)
    if hit is not None and (now - hit[1]) < _BOOK_VOL_TTL:
        return hit[0]
    scalar = 1.0
    try:
        hist = api.get_portfolio_history(period='3M', timeframe='1D')
        realized = ewma_annualized_vol(getattr(hist, 'equity', None))
        if realized is not None and realized > 1e-6:
            from strategy_config import PORTFOLIO_VOL_TARGET
            target = PORTFOLIO_VOL_TARGET.get(asset_type, 0.25)
            scalar = min(max(target / realized, 0.5), 1.0)
            if scalar < 1.0:
                logger.info("[BOOK-VOL] %s: realized %.0f%% > target %.0f%% "
                            "-> %.2fx entries", asset_type, realized * 100,
                            target * 100, scalar)
    except Exception as e:
        logger.debug("[BOOK-VOL] unavailable (%s) — neutral 1.0", e)
    _book_vol_cache[asset_type] = (scalar, now)
    return scalar
