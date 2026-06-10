"""Perpetual funding-rate positioning signal (live, free, no key).

Evidence (BIS WP 1087 / Schmeling-Schrimpf-Todorov "Crypto Carry",
Management Science 2024): perp funding embeds leveraged positioning, and
EXTREME positive funding (crowded longs paying shorts heavily) precedes
crashes — the 12-48h unwind channel this system trades. Used here as a
bounded de-risk TILT on new crypto entries, never as standalone alpha.

Source: OKX public REST (verified reachable from US IPs; Binance live REST
is geo-blocked 451). Cached 15 minutes; a rolling history persists to disk
so a z-score forms over time — until ~30 samples exist, absolute
annualized-rate thresholds apply instead.
"""

import json
import os
import threading
import time
import urllib.request

from log_config import get_logger

logger = get_logger(__name__)

_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'funding_history.json')
_CACHE_TTL = 900  # 15 min — funding updates on 8h cycles; this is plenty
_MAX_SAMPLES = 270  # ~90 days of 8h fundings

# Alpaca spot symbol -> OKX perp instrument
OKX_INSTRUMENTS = {
    'BTC/USD': 'BTC-USDT-SWAP', 'ETH/USD': 'ETH-USDT-SWAP',
    'XRP/USD': 'XRP-USDT-SWAP', 'SOL/USD': 'SOL-USDT-SWAP',
    'DOGE/USD': 'DOGE-USDT-SWAP', 'LINK/USD': 'LINK-USDT-SWAP',
    'AVAX/USD': 'AVAX-USDT-SWAP', 'DOT/USD': 'DOT-USDT-SWAP',
    'LTC/USD': 'LTC-USDT-SWAP', 'BCH/USD': 'BCH-USDT-SWAP',
}

# Tilt thresholds (8h rate annualized = rate * 3 * 365)
CROWDED_ANNUALIZED = 0.30   # >30%/yr funding -> crowded longs -> 0.6x
EXTREME_ANNUALIZED = 0.75   # >75%/yr -> block-equivalent 0.25x
CROWDED_Z = 2.0
EXTREME_Z = 3.0

_lock = threading.Lock()
_cache: dict[str, tuple[float, float]] = {}   # symbol -> (ts, 8h rate)
_history: dict[str, list[float]] | None = None


def _load_history() -> dict:
    global _history
    if _history is None:
        try:
            with open(_HISTORY_FILE) as f:
                _history = json.load(f)
        except (OSError, json.JSONDecodeError):
            _history = {}
    return _history


def _save_history():
    try:
        tmp = _HISTORY_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(_history, f)
        os.replace(tmp, _HISTORY_FILE)
    except OSError:
        pass


def get_funding_rate(symbol: str) -> float | None:
    """Current 8h funding rate for the mapped OKX perp (cached 15 min)."""
    inst = OKX_INSTRUMENTS.get(symbol)
    if inst is None:
        return None
    now = time.monotonic()
    with _lock:
        hit = _cache.get(symbol)
        if hit and (now - hit[0]) < _CACHE_TTL:
            return hit[1]
    try:
        url = f"https://www.okx.com/api/v5/public/funding-rate?instId={inst}"
        req = urllib.request.Request(url, headers={'User-Agent': 'trader/1.0'})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        rate = float(data['data'][0]['fundingRate'])
    except Exception as e:
        logger.debug('[FUNDING] %s: fetch failed: %s', symbol, e)
        return None
    with _lock:
        _cache[symbol] = (now, rate)
        hist = _load_history().setdefault(symbol, [])
        # One sample per ~8h cycle: only append if meaningfully spaced
        if not hist or abs(hist[-1] - rate) > 1e-9 or len(hist) < 3:
            hist.append(rate)
            if len(hist) > _MAX_SAMPLES:
                del hist[:len(hist) - _MAX_SAMPLES]
            _save_history()
    return rate


def live_funding_features(symbol: str) -> dict | None:
    """Live values for the model's funding FEATURES (training parity).

    Combines the Binance archive (the same series the harvest trained on)
    with the current OKX rate so the live z-score uses the same trailing
    distribution as training — without this, live z would read 0 until
    30 days of local samples accumulated (train/serve skew).
    Returns {'Funding_Rate_Ann', 'Funding_Z', 'Funding_Chg_24h'} or None.
    """
    rate = get_funding_rate(symbol)
    if rate is None:
        return None
    try:
        from funding_archive import get_funding_series
        s = get_funding_series(symbol)
    except Exception:
        s = None

    ann = rate * 3 * 365
    z = 0.0
    chg = 0.0
    samples = None
    if s is not None and len(s) >= 33:
        samples = list(s.values[-90:])
    else:
        hist = _load_history().get(symbol, [])
        if len(hist) >= 33:
            samples = hist[-90:]
    if samples is not None:
        import statistics
        mu = statistics.fmean(samples)
        sd = statistics.pstdev(samples)
        if sd > 1e-12:
            z = (rate - mu) / sd
        chg = (rate - samples[-3]) * 3 * 365
    return {'Funding_Rate_Ann': ann, 'Funding_Z': z, 'Funding_Chg_24h': chg}


def funding_tilt(symbol: str) -> float:
    """Bounded entry-size tilt from funding positioning. 1.0 = neutral.

    Crowded longs (high positive funding) -> shrink LONG entries; the
    tilt NEVER boosts above 1.0 (negative funding is not a buy signal by
    itself — carry alpha has decayed; see arXiv 2510.14435).
    """
    rate = get_funding_rate(symbol)
    if rate is None:
        return 1.0
    annualized = rate * 3 * 365

    hist = _load_history().get(symbol, [])
    z = None
    if len(hist) >= 30:
        import statistics
        mu = statistics.fmean(hist)
        sd = statistics.pstdev(hist)
        if sd > 1e-12:
            z = (rate - mu) / sd

    extreme = (annualized > EXTREME_ANNUALIZED
               or (z is not None and z > EXTREME_Z))
    crowded = (annualized > CROWDED_ANNUALIZED
               or (z is not None and z > CROWDED_Z))
    if extreme:
        logger.info('[FUNDING] %s: EXTREME crowded longs '
                    '(%.1f%%/yr, z=%s) -> 0.25x entries',
                    symbol, annualized * 100,
                    f'{z:.1f}' if z is not None else 'n/a')
        return 0.25
    if crowded:
        logger.info('[FUNDING] %s: crowded longs (%.1f%%/yr) -> 0.6x entries',
                    symbol, annualized * 100)
        return 0.6
    return 1.0
