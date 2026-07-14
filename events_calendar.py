"""Earnings-calendar awareness (Finnhub free tier, existing key).

The single largest tail in the stock book: holding a high-beta name
through its earnings print. GTC stop orders do NOT protect against gaps —
they convert to market at the open and fill wherever the gap lands
(20%+ earnings gaps are routine in this universe). Policy:

  - OVERNIGHT SLEEVE: any symbol reporting between today's close and the
    next open (+1 day buffer) is ineligible — FAIL CLOSED: if the
    calendar is unavailable, the sleeve is empty.
  - NEW ENTRIES: blocked within 1 day of a known print — fail OPEN with
    a warning (a Finnhub outage shouldn't halt all stock trading; the
    sleeve is where the gap risk lives).
  - POST-PRINT: first trading day after a report gets a 0.5x size tilt
    (daily ATR/GARCH lag the 3-5x realized-vol expansion by ~1 day).

Cache: one unfiltered 14-day-window call per day -> earnings_calendar.json.
"""

import datetime
import json
import os
import threading
import time

from log_config import get_logger

logger = get_logger(__name__)

_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'earnings_calendar.json')
_REFRESH_SEC = 24 * 3600
_lock = threading.Lock()
_mem: dict | None = None
_last_attempt = 0.0


def _load_cache() -> dict:
    global _mem
    if _mem is None:
        try:
            with open(_CACHE_FILE) as f:
                _mem = json.load(f)
        except (OSError, json.JSONDecodeError):
            _mem = {}
        if not isinstance(_mem, dict):
            # Corrupt-but-parseable file (e.g. a JSON list): reset — the
            # sleeve then fails closed via calendar_available()==False
            # instead of AttributeError-crashing the unguarded
            # _select_overnight_keepers call path. Same guard as novelty._load.
            _mem = {}
    return _mem


def _fetch_calendar() -> dict | None:
    """One unfiltered earnings-calendar call covering -3..+11 days."""
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return None
    try:
        import finnhub
        client = finnhub.Client(api_key=api_key)
        today = datetime.date.today()
        cal = client.earnings_calendar(
            _from=(today - datetime.timedelta(days=3)).isoformat(),
            to=(today + datetime.timedelta(days=11)).isoformat(),
            symbol='', international=False)
        entries = cal.get('earningsCalendar', []) or []
        by_symbol: dict[str, list] = {}
        for e in entries:
            sym = (e.get('symbol') or '').upper()
            date = e.get('date')
            if sym and date:
                by_symbol.setdefault(sym, []).append(
                    {'date': date, 'hour': e.get('hour', '')})
        return {'fetched_at': datetime.datetime.now().isoformat(),
                'by_symbol': by_symbol}
    except Exception as e:
        logger.warning('[EARNINGS] calendar fetch failed: %s', e)
        return None


def refresh_if_stale() -> bool:
    """Refresh the cache if older than 24h. Returns availability."""
    global _mem, _last_attempt
    with _lock:
        cache = _load_cache()
        fetched = cache.get('fetched_at')
        fresh = False
        if fetched:
            try:
                age = (datetime.datetime.now()
                       - datetime.datetime.fromisoformat(fetched)).total_seconds()
                fresh = age < _REFRESH_SEC
            except (TypeError, ValueError):
                pass
        if fresh:
            return True
        # Throttle failed attempts to one per 30 min
        if time.monotonic() - _last_attempt < 1800:
            return bool(cache.get('by_symbol'))
        _last_attempt = time.monotonic()

    new = _fetch_calendar()
    with _lock:
        if new is not None:
            _mem = new
            try:
                # pid-unique tmp: a fixed '.tmp' let a live bot and a manual
                # CLI run interleave one tmp file — mirrors edgar_events.
                tmp = f"{_CACHE_FILE}.{os.getpid()}.tmp"
                with open(tmp, 'w') as f:
                    json.dump(new, f)
                os.replace(tmp, _CACHE_FILE)
            except OSError:
                pass
            logger.info('[EARNINGS] calendar refreshed: %d symbols',
                        len(new['by_symbol']))
            return True
        return bool(_load_cache().get('by_symbol'))


def calendar_available() -> bool:
    refresh_if_stale()
    return bool(_load_cache().get('by_symbol'))


def _dates_for(symbol: str) -> list[dict]:
    return _load_cache().get('by_symbol', {}).get(symbol.upper(), [])


def earnings_within_days(symbol: str, days: int = 1) -> bool:
    """True if the symbol reports within the next `days` calendar days."""
    refresh_if_stale()
    today = datetime.date.today()
    horizon = today + datetime.timedelta(days=days)
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if today <= d <= horizon:
            return True
    return False


def blocks_overnight_hold(symbol: str) -> bool:
    """True if holding overnight crosses a known print (+1 day buffer).

    Covers: reports after today's close (amc/unknown today) and any report
    tomorrow or the day after (bmo prints gap at the next open; the buffer
    absorbs date ambiguity in the feed).
    """
    refresh_if_stale()
    today = datetime.date.today()
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if today <= d <= today + datetime.timedelta(days=2):
            return True
    return False


def reported_recently(symbol: str) -> bool:
    """True on the first trading day(s) after a report (post-print vol)."""
    refresh_if_stale()
    today = datetime.date.today()
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if today - datetime.timedelta(days=1) <= d <= today:
            # Reported yesterday (any time) or this morning (bmo) — the
            # report date itself counts because amc prints affect TOMORROW,
            # which blocks_overnight_hold handles
            if d < today or e.get('hour') == 'bmo':
                return True
    return False
