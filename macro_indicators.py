"""Macro indicators for regime-based risk management.

Fetches financial stress, VIX, CAPE, and stablecoin peg data.
Combines into a MacroRegime that trading loops use for position sizing
and stop-loss adjustments.

Sources:
- Financial Stress: FRED STLFSI2 (free, no auth)
- VIX: yfinance (already installed)
- CAPE: estimated from SPY trailing P/E x1.6 (real-time Shiller APIs
  proved unreliable — see fetch_cape)
- Stablecoins: Alpaca crypto quotes
"""

import time
from log_config import get_logger

logger = get_logger(__name__)

# Cache durations (seconds)
_VIX_CACHE_TTL = 3600       # 1 hour
_STRESS_CACHE_TTL = 86400   # 1 day (weekly data anyway)
_CAPE_CACHE_TTL = 86400     # 1 day
_STABLECOIN_TTL = 300        # 5 min

# Cache storage
_cache: dict[str, tuple[object, float]] = {}


def _get_cached(key: str, ttl: float):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < ttl:
            return val
    return None


def _set_cached(key: str, val):
    _cache[key] = (val, time.time())


# --- VIX ---

def fetch_vix() -> float | None:
    """Fetch current VIX level: yfinance primary, FRED VIXCLS fallback.

    yfinance is unofficial scraping and Yahoo's throttling correlates with
    crash-day traffic — exactly when the VIX risk ladders matter most. A
    VIX of None makes every ladder silently pass at 1.0x, so a 1-day-lagged
    official FRED value is far better than blindness. (The sizing layer
    additionally clamps tilt when multiple advisory inputs are missing.)
    """
    cached = _get_cached('vix', _VIX_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX')
        hist = vix.history(period='5d')
        if hist is not None and not hist.empty:
            val = float(hist['Close'].iloc[-1])
            _set_cached('vix', val)
            logger.info("[MACRO] VIX: %.1f", val)
            return val
    except Exception as e:
        logger.debug("[MACRO] VIX fetch error: %s", e)

    # Fallback: FRED VIXCLS (official CBOE close, ~1 day lag, free CSV)
    try:
        import urllib.request
        req = urllib.request.Request(
            'https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS',
            headers={'User-Agent': 'trader/1.0'})
        body = urllib.request.urlopen(req, timeout=10).read().decode()
        for line in reversed(body.strip().splitlines()):
            parts = line.split(',')
            if len(parts) == 2 and parts[1] not in ('.', 'VIXCLS', ''):
                val = float(parts[1])
                _set_cached('vix', val)
                logger.info("[MACRO] VIX (FRED fallback, 1d lag): %.1f", val)
                return val
    except Exception as e:
        logger.debug("[MACRO] FRED VIX fallback error: %s", e)
    logger.warning("[MACRO] VIX unavailable from ALL sources — "
                   "VIX risk ladders blind (pass at 1.0x)")
    return None


# --- Financial Stress Index ---

def fetch_financial_stress() -> float | None:
    """Fetch St. Louis Financial Stress Index (STLFSI2) from FRED.

    Zero = normal, positive = above-average stress.
    Units are standard deviations from the mean.
    """
    cached = _get_cached('stress', _STRESS_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        import requests
        # FRED fredgraph.csv endpoint (free CSV export, no auth)
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI2"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            if len(lines) > 1:
                last_line = lines[-1]
                parts = last_line.split(',')
                if len(parts) == 2 and parts[1] != '.':
                    val = float(parts[1])
                    _set_cached('stress', val)
                    logger.info("[MACRO] Financial Stress (STLFSI2): %.2f", val)
                    return val
    except Exception as e:
        logger.debug("[MACRO] STLFSI2 fetch error: %s", e)
    logger.warning("[MACRO] Financial stress (STLFSI2) unavailable — "
                   "stress rule blind")
    return None


# --- Shiller CAPE ---

def fetch_cape() -> float | None:
    """Fetch Shiller CAPE ratio estimate.

    Uses a simple approximation: SPY P/E * 1.6 adjustment factor
    since real-time Shiller CAPE APIs are unreliable.
    """
    cached = _get_cached('cape', _CAPE_CACHE_TTL)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        spy = yf.Ticker('SPY')
        info = spy.info
        pe = info.get('trailingPE')
        if pe is not None:
            # CAPE is roughly 1.5-1.8x trailing PE historically
            cape_est = pe * 1.6
            _set_cached('cape', cape_est)
            logger.info("[MACRO] CAPE estimate: %.1f (PE=%.1f)", cape_est, pe)
            return cape_est
    except Exception as e:
        logger.debug("[MACRO] CAPE fetch error: %s", e)
    logger.warning("[MACRO] CAPE estimate unavailable (no SPY trailingPE) — "
                   "valuation rule blind")
    return None


# --- Stablecoin Contagion ---

_STABLECOINS = ['USDT/USD', 'USDC/USD']
_STABLECOIN_WARN_DEVIATION = 0.005   # 0.5%
_STABLECOIN_EMERGENCY_DEVIATION = 0.02  # 2%


def check_stablecoin_pegs(api) -> dict:
    """Check stablecoin prices for depeg risk.

    Returns:
        dict with:
            - depegged: bool (any stablecoin > 0.5% from $1)
            - emergency: bool (any > 2% from $1)
            - deviations: {symbol: pct_deviation}
    """
    cached = _get_cached('stablecoins', _STABLECOIN_TTL)
    if cached is not None:
        return cached

    result = {'depegged': False, 'emergency': False, 'deviations': {}}

    for symbol in _STABLECOINS:
        try:
            quotes = api.get_latest_crypto_quotes([symbol])
            q = quotes[symbol]
            mid = (float(q.bp) + float(q.ap)) / 2
            deviation = abs(mid - 1.0)
            result['deviations'][symbol] = deviation

            if deviation > _STABLECOIN_EMERGENCY_DEVIATION:
                result['emergency'] = True
                result['depegged'] = True
                logger.warning("[CONTAGION] %s EMERGENCY depeg: $%.4f (%.2f%% off)",
                               symbol, mid, deviation * 100)
            elif deviation > _STABLECOIN_WARN_DEVIATION:
                result['depegged'] = True
                logger.warning("[CONTAGION] %s depeg warning: $%.4f (%.2f%% off)",
                               symbol, mid, deviation * 100)
        except Exception as e:
            logger.debug("[CONTAGION] Error checking %s: %s", symbol, e)

    if _STABLECOINS and not result['deviations']:
        # Every quote fetch failed: peg status is UNKNOWN, not "fine".
        # Do NOT cache — the next call retries immediately instead of
        # serving a total outage as all-clear for the full TTL.
        logger.warning("[CONTAGION] All stablecoin quote fetches failed — "
                       "peg status UNKNOWN")
        return result

    _set_cached('stablecoins', result)
    return result


# --- SPY 200-day trend filter ---

def get_spy_trend_ok(api) -> bool | None:
    """True when SPY closes above its 200-day SMA (Faber's trend filter).

    Faber (2007): the 200d MA filter cut max drawdown 83.7% -> 42.2% and
    is the best-evidenced simple regime gate. Below trend, the stock loop
    blocks non-safe-haven entries. Cached 1h. Returns None when data is
    unavailable (callers should fail OPEN so a dead data feed doesn't
    silently halt all trading — the VIX gates still protect).
    """
    cached = _get_cached('spy_trend', 3600)
    if cached is not None:
        return cached
    try:
        from datetime import datetime, timedelta, timezone
        start = datetime.now(timezone.utc) - timedelta(days=320)
        bars = api.get_bars('SPY', '1Day', start=start.isoformat(),
                            adjustment='all')
        closes = [float(b.c) for b in bars]
        if len(closes) < 200:
            logger.warning("[MACRO] SPY trend: only %d daily bars (<200) — "
                           "filter fails OPEN", len(closes))
            return None
        sma200 = sum(closes[-200:]) / 200
        ok = closes[-1] > sma200
        _set_cached('spy_trend', ok)
        return ok
    except Exception as e:
        logger.warning("[MACRO] SPY trend fetch failed (filter fails OPEN): %s", e)
        return None


# --- Regime Computation ---

# Historical CAPE mean and std (approximate)
_CAPE_MEAN = 25.0
_CAPE_STD = 8.0


def get_macro_regime(api=None, asset_type='crypto') -> 'MacroRegime':
    """Compute current macro regime with sizing and stop multipliers.

    Regime rules:
        VIX < 15 → normal (1.0x sizing)
        VIX 15-25 → caution (0.8x sizing)
        VIX 25-35 → defensive (0.5x sizing)
        VIX > 35 → halt new stock entries
        STLFSI2 > 1.0 → reduce sizing 50%, tighten stops
        CAPE z-score > 1.5 → reduce stock sizing 30%

    Returns:
        MacroRegime dataclass with sizing_mult and stop_mult.
    """
    from types_mod import MacroRegime

    vix = fetch_vix()
    stress = fetch_financial_stress()
    cape = fetch_cape() if asset_type == 'stock' else None

    sizing_mult = 1.0
    stop_mult = 1.0
    labels = []

    # VIX-based regime
    if vix is not None:
        if vix > 35:
            sizing_mult *= 0.3
            labels.append('crisis')
        elif vix > 25:
            sizing_mult *= 0.5
            labels.append('defensive')
        elif vix > 15:
            sizing_mult *= 0.8
            labels.append('caution')
        else:
            labels.append('normal')
    else:
        # A blind regime otherwise labels itself 'normal' — indistinguishable
        # in the operator logs from a genuinely calm market.
        logger.warning("[MACRO] Regime computed WITHOUT VIX — "
                       "VIX tiers skipped, label may read 'normal' while blind")

    # Financial stress
    if stress is not None and stress > 1.0:
        sizing_mult *= 0.5
        stop_mult *= 0.8  # tighter stops
        labels.append('high_stress')

    # CAPE (stocks only)
    if cape is not None:
        cape_z = (cape - _CAPE_MEAN) / _CAPE_STD
        if cape_z > 1.5:
            sizing_mult *= 0.7
            labels.append('overvalued')

    # Stablecoin check (crypto only)
    stablecoin_alert = False
    if api is not None and asset_type == 'crypto':
        peg_status = check_stablecoin_pegs(api)
        if peg_status['emergency']:
            stablecoin_alert = True
            sizing_mult *= 0.0  # halt all crypto
            labels.append('stablecoin_emergency')
        elif peg_status['depegged']:
            stablecoin_alert = True
            stop_mult *= 0.7  # much tighter stops
            labels.append('stablecoin_warning')

    regime_label = '+'.join(labels) if labels else 'normal'

    # Above VIX 20 (mid-'caution' and up), drop the cached VIX so the next
    # regime update refetches. Regime updates run every 10th loop cycle
    # (~5 min per bot at the 30s LOOP_INTERVAL); combined-bot mode has two
    # loops sharing this module cache, so effective refetch can be ~2-3 min.
    if vix is not None and vix > 20:
        _cache.pop('vix', None)

    return MacroRegime(
        stress_level=stress,
        vix=vix,
        cape=cape,
        regime_label=regime_label,
        sizing_mult=round(sizing_mult, 3),
        stop_mult=round(stop_mult, 3),
        stablecoin_alert=stablecoin_alert,
    )
