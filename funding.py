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
_MAX_SAMPLES = 270  # ~90 days at one sample/8h — TRUE only with time
                    # thinning ON; the default value-change guard admits
                    # ~96 samples/day, spanning ~2.8 days (D28)
# D28 (c26-P5): the value-change append guard admits the continuously
# drifting OKX predicted rate nearly every 15-min poll, so the z baseline
# spans ~2.8 days and the 0.6x/0.25x crowding tilts fire on 3-day noise.
# Time-thinned appends (one sample per ~7.5h, oi_archive's pattern) +
# archive-preferred z baseline fix this but CHANGE TILT FREQUENCY ->
# default OFF. Activate on the Jetson with TRADER_FUNDING_Z_TIME_THINNING=1
# (recommended; on activation consider deleting funding_history.json so the
# old dense 3-day history does not linger in the window for ~84 days).
FUNDING_Z_TIME_THINNING = os.getenv(
    'TRADER_FUNDING_Z_TIME_THINNING', '0').strip().lower() in ('1', 'true', 'yes')
_THIN_SEC = 27000.0          # ~7.5h: just under the 8h funding cycle
_TS_KEY = '_last_append_ts'  # reserved history-file key: {symbol: epoch};
                             # safe — real keys look like 'BTC/USD' and the
                             # only readers .get() exact symbols

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
# {symbol: [rates...]}; under FUNDING_Z_TIME_THINNING also the _TS_KEY
# sidecar dict — never iterate .items() assuming list values
_history: dict | None = None
_save_warned = False
_thin_advice_logged = False  # once-per-process activation recommendation
_stale_warned: set[str] = set()   # symbols warned once about a stale archive baseline
_ARCHIVE_STALE_H = 48.0           # warn if archive tail older than ~2 funding cycles


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
    global _save_warned
    try:
        tmp = _HISTORY_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(_history, f)
        os.replace(tmp, _HISTORY_FILE)
    except OSError as e:
        # Full/read-only disk: the z baseline restarts cold on every bot
        # restart if this never persists — say so once, don't spam
        if not _save_warned:
            _save_warned = True
            logger.warning('[FUNDING] history persist failed: %s', e)


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
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        rate = float(data['data'][0]['fundingRate'])
    except Exception as e:
        logger.debug('[FUNDING] %s: fetch failed: %s', symbol, e)
        return None
    global _thin_advice_logged
    with _lock:
        _cache[symbol] = (now, rate)
        hist_all = _load_history()
        hist = hist_all.setdefault(symbol, [])
        if FUNDING_Z_TIME_THINNING:
            # One sample per ~7.5h WALL CLOCK (oi_archive pattern): the
            # baseline then really spans ~_MAX_SAMPLES * 7.5h ≈ 84 days.
            ts_map = hist_all.setdefault(_TS_KEY, {})
            wall = time.time()
            last = ts_map.get(symbol)
            if last is None or (wall - last) >= _THIN_SEC:
                hist.append(rate)
                if len(hist) > _MAX_SAMPLES:
                    del hist[:len(hist) - _MAX_SAMPLES]
                ts_map[symbol] = wall
                _save_history()
        else:
            # One sample per ~8h cycle: only append if meaningfully spaced
            # (D28: the predicted rate drifts every poll, so this appends
            # ~96x/day and the window spans ~2.8 days — flag above fixes it)
            if not hist or abs(hist[-1] - rate) > 1e-9 or len(hist) < 3:
                hist.append(rate)
                if len(hist) > _MAX_SAMPLES:
                    del hist[:len(hist) - _MAX_SAMPLES]
                _save_history()
                if not _thin_advice_logged and len(hist) >= 30:
                    _thin_advice_logged = True
                    logger.warning(
                        '[FUNDING] z-baseline history is value-thinned: at '
                        '15-min polls the %d-sample window spans ~2.8 days, '
                        'not ~90 — crowding tilts fire on short-term noise. '
                        'RECOMMENDED: set TRADER_FUNDING_Z_TIME_THINNING=1 '
                        '(and delete funding_history.json once) to enable '
                        'time-thinned ~7.5h sampling.', _MAX_SAMPLES)
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
    except Exception as e:
        logger.debug('[FUNDING] %s: archive unavailable, using local '
                     'history: %s', symbol, e)
        s = None

    ann = rate * 3 * 365
    z = 0.0
    chg = 0.0
    samples = None
    if s is not None and len(s) >= 33:
        samples = list(s.values[-90:])
        # Observability only (returned values are unchanged): the archive is
        # synced at weekly harvests and holds complete months, so its tail can
        # be up to ~5 weeks stale — a lagged baseline shifts z/chg semantics
        # vs training. Stamp the tail age; warn once per symbol when stale.
        try:
            age_h = (time.time() - s.index[-1].timestamp()) / 3600.0
            if age_h > _ARCHIVE_STALE_H and symbol not in _stale_warned:
                _stale_warned.add(symbol)
                logger.warning('[FUNDING] %s: archive baseline tail is %.0fh '
                               'stale (z/chg computed vs a lagged window)',
                               symbol, age_h)
            else:
                logger.debug('[FUNDING] %s: archive tail age %.1fh',
                             symbol, age_h)
        except Exception:
            pass
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
    samples = None
    if FUNDING_Z_TIME_THINNING:
        # Prefer the harvest-synced archive baseline — the same trailing
        # distribution live_funding_features already uses — falling back
        # to the (time-thinned) local history.
        try:
            from funding_archive import get_funding_series
            s = get_funding_series(symbol)
        except Exception as e:
            logger.debug('[FUNDING] %s: tilt archive unavailable, using '
                         'local history: %s', symbol, e)
            s = None
        if s is not None and len(s) >= 33:
            samples = list(s.values[-90:])
        elif len(hist) >= 30:
            samples = hist[-90:]
    elif len(hist) >= 30:
        samples = hist
    if samples is not None:
        import statistics
        mu = statistics.fmean(samples)
        sd = statistics.pstdev(samples)
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
        # Include z: this branch fires on z > 2.0 alone, where the
        # annualized rate can sit well under the 30%/yr threshold
        logger.info('[FUNDING] %s: crowded longs (%.1f%%/yr, z=%s) '
                    '-> 0.6x entries', symbol, annualized * 100,
                    f'{z:.1f}' if z is not None else 'n/a')
        return 0.6
    return 1.0
