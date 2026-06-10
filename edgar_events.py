"""SEC EDGAR corporate-event entry veto (8-K items + pending M&A).

Free, keyless, official. Two classes of state invalidate this system's
hourly momentum/mean-reversion signals:

  1. Fresh 8-K disclosures of solvency/credibility events — Item 1.03
     (bankruptcy), 2.04 (debt acceleration), 4.02 (non-reliance on prior
     financials, i.e. restatement), 5.02 (officer/director departure).
     Post-filing drift after these is dominated by information the model
     never sees. Entries blocked for VETO_8K_WINDOW_DAYS after filing.
     (5.02 also catches routine board churn — over-broad, but it only
     pauses NEW entries for a few days.)

  2. Pending M&A on the target — price pins to deal terms and trades on
     deal odds, not technicals (425 / SC 14D9 / DEFM14A / S-4 filings).
     Blocked for MA_WINDOW_DAYS.

EDGAR etiquette: declared User-Agent, <=10 req/s (we do ~top-N sequential
fetches once per day per symbol, disk-cached). FAIL OPEN — an EDGAR
outage must not halt trading; the earnings sleeve handles its own
fail-closed logic separately.
"""

import datetime as dt
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from log_config import get_logger

logger = get_logger(__name__)

_UA = {'User-Agent': 'trader-research kywwilson@gmail.com'}
_CACHE_FILE = BASE_DIR / 'edgar_cache.json'
_TICKER_MAP_FILE = BASE_DIR / 'edgar_tickers.json'
_TICKER_MAP_MAX_AGE_DAYS = 7

VETO_8K_ITEMS = {
    '1.03': 'bankruptcy filing',
    '2.04': 'debt acceleration',
    '4.02': 'restatement (non-reliance)',
    '5.02': 'officer/director departure',
}
VETO_8K_WINDOW_DAYS = 5
MA_FORMS = {'425', 'SC 14D9', 'DEFM14A', 'DEFM14C', 'S-4'}
MA_WINDOW_DAYS = 90


def _get_json(url: str, timeout: int = 5):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _ticker_cik_map() -> dict[str, str]:
    """{TICKER: 10-digit CIK}, disk-cached for a week."""
    try:
        st = _TICKER_MAP_FILE.stat()
        age = dt.datetime.now().timestamp() - st.st_mtime
        if age < _TICKER_MAP_MAX_AGE_DAYS * 86400:
            with open(_TICKER_MAP_FILE) as f:
                return json.load(f)
    except OSError:
        pass
    data = _get_json('https://www.sec.gov/files/company_tickers.json',
                     timeout=10)
    out = {}
    for rec in data.values():
        out[str(rec['ticker']).upper()] = f"{int(rec['cik_str']):010d}"
    tmp = str(_TICKER_MAP_FILE) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, _TICKER_MAP_FILE)
    return out


def _recent_filings(cik: str) -> list[tuple[str, str, str]]:
    """[(form, filing_date, items), ...] from the submissions API."""
    data = _get_json(f'https://data.sec.gov/submissions/CIK{cik}.json')
    recent = (data.get('filings') or {}).get('recent') or {}
    forms = recent.get('form') or []
    dates = recent.get('filingDate') or []
    items = recent.get('items') or [''] * len(forms)
    return list(zip(forms, dates, items))


def _evaluate(filings: list[tuple[str, str, str]],
              today: dt.date) -> tuple[bool, str | None]:
    """Apply the veto rules to a filing list."""
    for form, fdate, items in filings:
        try:
            filed = dt.date.fromisoformat(fdate)
        except (TypeError, ValueError):
            continue
        age = (today - filed).days
        if age < 0 or age > MA_WINDOW_DAYS:
            continue
        form_u = str(form).upper().strip()
        if form_u in MA_FORMS:
            return True, f"pending M&A ({form_u} filed {fdate})"
        if form_u.startswith('8-K') and age <= VETO_8K_WINDOW_DAYS:
            for code, label in VETO_8K_ITEMS.items():
                if code in str(items):
                    return True, f"8-K item {code} {label} (filed {fdate})"
    return False, None


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        tmp = str(_CACHE_FILE) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, _CACHE_FILE)
    except OSError:
        pass


def entry_blocked(symbol: str) -> tuple[bool, str | None]:
    """(blocked, reason) for a stock symbol. One EDGAR fetch/symbol/day,
    disk-cached. FAIL OPEN on any error."""
    today = dt.date.today()
    cache = _load_cache()
    hit = cache.get(symbol)
    if hit and hit.get('date') == today.isoformat():
        return bool(hit.get('blocked')), hit.get('reason')
    blocked, reason = False, None
    try:
        cik = _ticker_cik_map().get(symbol.upper())
        if cik:
            blocked, reason = _evaluate(_recent_filings(cik), today)
            if blocked:
                logger.info("[EDGAR] %s: entry blocked — %s", symbol, reason)
    except Exception as e:
        logger.debug("[EDGAR] %s: check failed (%s) — fail open", symbol, e)
        return False, None  # do NOT cache failures; retry next call
    cache[symbol] = {'date': today.isoformat(), 'blocked': blocked,
                     'reason': reason}
    _save_cache(cache)
    return blocked, reason


if __name__ == '__main__':
    for sym in sys.argv[1:] or ['NVDA']:
        print(sym, entry_blocked(sym))
