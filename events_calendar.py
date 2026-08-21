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

# --- Trading-day windows (D07, flag strategy_config.EVENTS_TRADING_DAY_WINDOWS) ---
# Static NYSE full-closure days, current + next year, refreshed annually by
# hand (no API dependency). Failure directions are both safe: a MISSING
# holiday degrades to weekend-only skipping (window still >= calendar mode);
# an EXTRA/stale date only ever WIDENS a block window.
_NYSE_HOLIDAYS = frozenset({
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    # 2027 (Juneteenth obs Fri 6/18; July-4 obs Mon 7/5; Christmas obs Fri 12/24;
    # Jan 1 2028 is a Saturday -> NYSE does NOT close Fri 2027-12-31)
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
})


def _trading_day_mode() -> bool:
    """Read the default-OFF flag; any failure -> calendar mode (current behavior)."""
    try:
        import strategy_config
        return bool(getattr(strategy_config, 'EVENTS_TRADING_DAY_WINDOWS', False))
    except Exception:
        return False


def _is_trading_day(d: datetime.date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in _NYSE_HOLIDAYS


def _add_trading_days(start: datetime.date, n: int) -> datetime.date:
    """Date n trading days after start (weekend/NYSE-holiday aware).

    Always >= start + n calendar days for n >= 0, so trading-day windows
    can only ever block MORE than calendar windows — never fewer.
    """
    d = start
    steps = 0
    while steps < n:
        d += datetime.timedelta(days=1)
        if _is_trading_day(d):
            steps += 1
    return d


def _prev_trading_day(d: datetime.date) -> datetime.date:
    """Nearest trading day strictly before d (always <= d - 1 day)."""
    p = d - datetime.timedelta(days=1)
    while not _is_trading_day(p):
        p -= datetime.timedelta(days=1)
    return p


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


def next_earnings_date(symbol: str) -> str | None:
    """Nearest upcoming earnings date (ISO 'YYYY-MM-DD'), or None.

    Cache-READ ONLY — deliberately does NOT call refresh_if_stale() (that
    path can fetch). This is called from the hot LLM-candidate-building
    path (llm_analyst.build_compact_evidence), where a network call would
    be a hidden hot-loop fetch; the calendar is refreshed elsewhere on its
    own 24h cadence, and this just reads whatever's already on disk.
    """
    today = datetime.date.today()
    best = None
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if d >= today and (best is None or d < best):
            best = d
    return best.isoformat() if best else None


def earnings_within_days(symbol: str, days: int = 1) -> bool:
    """True if the symbol reports within the next `days` days.

    Calendar days by default; TRADING days (weekend/NYSE-holiday aware,
    strictly wider) when strategy_config.EVENTS_TRADING_DAY_WINDOWS is on.
    """
    refresh_if_stale()
    today = datetime.date.today()
    if _trading_day_mode():
        horizon = _add_trading_days(today, days)
    else:
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

    With EVENTS_TRADING_DAY_WINDOWS on, the +2 buffer walks trading days
    (Friday covers through Tuesday).
    """
    refresh_if_stale()
    today = datetime.date.today()
    if _trading_day_mode():
        end = _add_trading_days(today, 2)   # Friday -> Tuesday; wider, never narrower
    else:
        end = today + datetime.timedelta(days=2)
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if today <= d <= end:
            return True
    return False


def reported_recently(symbol: str) -> bool:
    """True on the first trading day(s) after a report (post-print vol)."""
    refresh_if_stale()
    today = datetime.date.today()
    if _trading_day_mode():
        start = _prev_trading_day(today)    # Monday reaches back to Friday
    else:
        start = today - datetime.timedelta(days=1)
    for e in _dates_for(symbol):
        try:
            d = datetime.date.fromisoformat(e['date'])
        except (KeyError, ValueError):
            continue
        if start <= d <= today:
            # Reported yesterday (any time) or this morning (bmo) — the
            # report date itself counts because amc prints affect TOMORROW,
            # which blocks_overnight_hold handles
            if d < today or e.get('hour') == 'bmo':
                return True
    return False
