"""Macro indicators for regime-based risk management.

Fetches financial stress, VIX, and stablecoin peg data.
Combines into a MacroRegime that trading loops use for position sizing
and stop-loss adjustments.

Sources:
- Financial Stress: FRED STLFSI2 (free, no auth)
- VIX: yfinance primary, FRED VIXCLS CSV fallback
- Stablecoins: Alpaca crypto quotes

The pseudo-CAPE estimator (SPY trailing P/E x1.6) and its 0.7x sizing
haircut were DELETED 2026-08-22 by owner ruling (KILL_LIST pending ask #3:
"fake data driving a real haircut"). Verbatim code + restoration notes:
research/campaign_2026-08/08_removed_code.md.
"""

import math
import time
from log_config import get_logger

logger = get_logger(__name__)

# Cache durations (seconds)
_VIX_CACHE_TTL = 3600       # 1 hour
_STRESS_CACHE_TTL = 86400   # 1 day (weekly data anyway)
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


# --- DERISK_STACK_V2 (c26 S3 / 02_research B06): the ONE VIX tier map ---
# Tier thresholds: enter 25/35 (strict >), exit 22/31 (strict <) — ~12% gap.
_VIX_TIER_ENTER = (25.0, 35.0)
_VIX_TIER_EXIT = (22.0, 31.0)
_VIX_TIER_MULTS = (1.0, 0.5, 0.3)   # normal / defensive / crisis
_vix_tier_state = {'tier': 0}        # VIX is global — one state for both books


def _reset_vix_tier_state():
    _vix_tier_state['tier'] = 0


def vix_tier_mult_v2(vix) -> float:
    """B06 single VIX tier map with asymmetric hysteresis. None -> 1.0
    fail-open, state untouched (base_loop's degraded clamp still guards).
    Stateful preview also runs while DERISK_STACK_V2 is OFF (shadow journal)."""
    if vix is None:
        return 1.0
    t = _vix_tier_state['tier']
    while t < 2 and vix > _VIX_TIER_ENTER[t]:
        t += 1                      # enter immediately, worst tier wins
    while t > 0 and vix < _VIX_TIER_EXIT[t - 1]:
        t -= 1                      # exit only below the hysteresis floor (cascades)
    _vix_tier_state['tier'] = t
    return _VIX_TIER_MULTS[t]


def regime_family_mults_v2(regime, asset_type: str) -> dict:
    """De-risk REGIME-family components for the DERISK_STACK_V2 MIN aggregation.

    stock  -> {'vix': vix_tier_mult_v2(regime.vix), 'stress': 0.5|1.0}
    crypto -> {'stress': 0.5|1.0}   (VIX replaced by BTC-RV state — caller adds it)
    The HMM multiplier and book-vol scalar are composed by the caller.
    (Pseudo-CAPE, formerly excluded here with a one-shot announce, was
    DELETED from the module 2026-08-22 by owner ruling — see module header.)
    Never raises; regime None -> {}.
    """
    if regime is None:
        return {}
    try:
        out = {}
        if asset_type != 'crypto':
            out['vix'] = vix_tier_mult_v2(regime.vix)
        # Same constant as the legacy STLFSI2 rule in get_macro_regime.
        out['stress'] = (0.5 if (regime.stress_level is not None
                                 and regime.stress_level > 1.0) else 1.0)
        return out
    except Exception as e:
        logger.warning("[DERISK-V2] regime_family_mults_v2 failed: %s", e)
        return {}


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
            # A not-yet-populated last row reads Close=NaN. Accept only a
            # finite, positive level: a NaN VIX silently passes every ladder
            # at 1.0x AND evades the `vix is None` blind-warning/degraded
            # clamp, and would be cached for the full TTL. Fall to FRED.
            if math.isfinite(val) and val > 0:
                _set_cached('vix', val)
                logger.info("[MACRO] VIX: %.1f", val)
                return val
            logger.debug("[MACRO] VIX yfinance close non-finite/non-positive "
                         "(%r) — trying FRED", val)
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
                if not (math.isfinite(val) and val > 0):
                    continue            # never cache/serve a NaN/inf/<=0 level
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

def get_macro_regime(api=None, asset_type='crypto') -> 'MacroRegime':
    """Compute current macro regime with sizing and stop multipliers.

    Regime rules:
        VIX < 15 → normal (1.0x sizing)
        VIX 15-25 → caution (0.8x sizing)
        VIX 25-35 → defensive (0.5x sizing)
        VIX > 35 → halt new stock entries
        STLFSI2 > 1.0 → reduce sizing 50%, tighten stops

    (The pseudo-CAPE z>1.5 -> 0.7x stock haircut was deleted 2026-08-22
    by owner ruling — see module header. MacroRegime.cape is always None.)

    Returns:
        MacroRegime dataclass with sizing_mult and stop_mult.
    """
    from types_mod import MacroRegime

    vix = fetch_vix()
    stress = fetch_financial_stress()

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
        # Field kept in types_mod for row/fixture compat; always None since
        # the 2026-08-22 pseudo-CAPE deletion.
        cape=None,
        regime_label=regime_label,
        sizing_mult=round(sizing_mult, 3),
        stop_mult=round(stop_mult, 3),
        stablecoin_alert=stablecoin_alert,
    )
