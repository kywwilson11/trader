"""Historical sentiment data — fetch, cache, score, and background-enrich.

SQLite-backed cache for historical news sentiment. Provides Daily_Sentiment
as a training feature for LSTM models and live inference.

Architecture:
  1. Harvest: fetch articles via Finnhub, keyword-score instantly -> train immediately
  2. Background worker: LLM re-scores articles in batches -> DB updates in-place
  3. Weekly retrain: picks up improved LLM scores automatically

Data sources:
  - Crypto: Fear & Greed Index (free, no key) -> fng_daily table
  - Stocks: Finnhub company_news -> articles + daily_sentiment tables
"""

import datetime
import logging
import math
import os
import sqlite3
import time
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dev Mac: python-dotenv absent; env comes from the shell
    pass

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sentiment_cache.db')
_db_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT DEFAULT '',
    url TEXT DEFAULT '',
    keyword_score REAL NOT NULL,
    llm_score REAL,
    fetched_at TEXT NOT NULL,
    llm_scored_at TEXT,
    UNIQUE(symbol, date, headline)
);

CREATE TABLE IF NOT EXISTS daily_sentiment (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL NOT NULL,
    article_count INTEGER NOT NULL,
    llm_count INTEGER DEFAULT 0,
    score_type TEXT DEFAULT 'keyword',
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fng_daily (
    date TEXT PRIMARY KEY,
    value INTEGER NOT NULL,
    score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_unscored ON articles(llm_score) WHERE llm_score IS NULL;
CREATE INDEX IF NOT EXISTS idx_symbol_date ON articles(symbol, date);
"""


def _get_db():
    """Get thread-local SQLite connection (WAL mode for concurrent reads)."""
    db = getattr(_db_local, 'conn', None)
    if db is not None:
        return db
    db = sqlite3.connect(_DB_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    # Schema creation needs exclusive lock — retry if another process holds it
    for attempt in range(3):
        try:
            db.executescript(_SCHEMA)
            break
        except sqlite3.OperationalError:
            if attempt < 2:
                time.sleep(2)
            else:
                raise
    _db_local.conn = db
    return db


# ---------------------------------------------------------------------------
# Keyword scoring (reuses sentiment._score_text)
# ---------------------------------------------------------------------------

def _keyword_score(headline, summary=''):
    """Score text using keyword-based sentiment. Returns float in (-1, 1).

    NOT the same combine as sentiment._kw_score_article: when only ONE field
    is scorable this returns that field's score at FULL weight, while
    _kw_score_article always applies fixed 0.6/0.4 (zeroing the missing
    part). Stored keyword_score rows are training features — do NOT unify
    the two without a retrain/promotion decision.
    """
    from sentiment import _score_text, _validate_text
    parts = []
    h = _validate_text(headline)
    if h:
        parts.append(('h', _score_text(h)))
    s = _validate_text(summary)
    if s:
        parts.append(('s', _score_text(s)))
    if not parts:
        return 0.0
    if len(parts) == 1:
        return parts[0][1]
    # headline 60%, summary 40%
    return parts[0][1] * 0.6 + parts[1][1] * 0.4


# ---------------------------------------------------------------------------
# Daily aggregation
# ---------------------------------------------------------------------------

def _aggregate_daily(db, symbol, date_str):
    """Recompute daily_sentiment row from articles for a symbol+date.

    Scoring priority:
      - All articles have LLM scores: use LLM average, type='llm'
      - Some have LLM scores: weighted blend (LLM 0.7, keyword 0.3), type='mixed'
      - None have LLM scores: use keyword average, type='keyword'
    """
    rows = db.execute(
        "SELECT keyword_score, llm_score FROM articles WHERE symbol=? AND date=?",
        (symbol, date_str),
    ).fetchall()

    if not rows:
        db.execute("DELETE FROM daily_sentiment WHERE symbol=? AND date=?",
                    (symbol, date_str))
        return

    article_count = len(rows)
    llm_scores = [r[1] for r in rows if r[1] is not None]
    keyword_scores = [r[0] for r in rows]
    llm_count = len(llm_scores)

    if llm_count == article_count:
        # All articles have LLM scores
        score = sum(llm_scores) / llm_count
        score_type = 'llm'
    elif llm_count > 0:
        # Mixed: blend LLM average with keyword average
        llm_avg = sum(llm_scores) / llm_count
        kw_avg = sum(keyword_scores) / article_count
        score = llm_avg * 0.7 + kw_avg * 0.3
        score_type = 'mixed'
    else:
        # All keyword
        score = sum(keyword_scores) / article_count
        score_type = 'keyword'

    db.execute(
        """INSERT INTO daily_sentiment (symbol, date, score, article_count, llm_count, score_type)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(symbol, date) DO UPDATE SET
             score=excluded.score, article_count=excluded.article_count,
             llm_count=excluded.llm_count, score_type=excluded.score_type""",
        (symbol, date_str, score, article_count, llm_count, score_type),
    )


# ---------------------------------------------------------------------------
# Fear & Greed Index history (crypto)
# ---------------------------------------------------------------------------

def _fng_value_to_score(value):
    """Normalize FnG value (0-100) to (-1, 1). 50 = 0.0."""
    return (value - 50) / 50.0


# fng_daily date-basis marker (H audit 2026-09-26). Rows cached before commit
# 5e23096 were bucketed with the LOCAL timezone (America/Chicago on the
# Jetson), so fng_daily[D] held the value alternative.me PUBLISHED at
# (D+1) 00:00 UTC (verified 1875/1875 against the API) -- up to 24h of
# lookahead for every crypto training bar. The UTC-bucketing fix below only
# governs NEW inserts and cached dates are never rewritten, so a re-harvest
# re-leaked. The one-time migration moves the legacy rows aside (delete
# nothing) and empties fng_daily so the next fetch refills it UTC-dated.
_FNG_BASIS_KEY = 'fng_date_basis'
_FNG_BASIS_UTC = 'utc_publication'
_FNG_LEGACY_TABLE = 'fng_daily_legacy_localtz'


def _fng_is_utc(db):
    """True once fng_daily holds UTC-publication-dated rows (marker set)."""
    row = db.execute("SELECT value FROM state WHERE key=?",
                     (_FNG_BASIS_KEY,)).fetchone()
    return bool(row and row[0] == _FNG_BASIS_UTC)


# A refill must replace at least this share of the legacy row count, or the
# migration is refused (an API returning a truncated series must not swap
# ~2000 leaked-but-real values for a handful of correct ones).
_FNG_REFILL_MIN_RATIO = 0.90
# Crypto harvest coverage floor: fewer nonzero Daily_Sentiment days than this
# share of the requested range logs a WARNING (REVIEW M2).
_FNG_MIN_COVERAGE = 0.90


def _migrate_fng_date_basis(db, refill_rows=None):
    """One-time repair of legacy local-tz-dated fng_daily rows.

    Keyed on state[fng_date_basis] == 'utc_publication': once the marker is
    set this is a no-op, so it runs at most once per DB.

    `refill_rows` is the FULL UTC-publication-dated series, (date, value,
    score) tuples, fetched BEFORE this call (REVIEW M2, 2026-09-26: the old
    order cleared fng_daily first, so a failed refetch left it empty and the
    crypto harvest stamped Daily_Sentiment = 0.0 into a retrain). Under one
    BEGIN IMMEDIATE transaction it
      (a) copies every fng_daily row into fng_daily_legacy_localtz,
      (b) verifies the copy is complete, then clears fng_daily,
      (c) inserts `refill_rows`,
      (d) sets the marker.
    Any failure rolls back and re-raises -- fng_daily is never cleared
    without its rows safely preserved AND replaced, and never left
    half-migrated. Legacy rows with no refill (or a refill covering fewer
    than _FNG_REFILL_MIN_RATIO of them) raise ValueError, nothing touched.
    A table with no legacy rows just gets the refill (if any) + the marker.

    Returns the number of legacy rows moved aside (0 when already migrated
    or the table was empty).
    """
    if _fng_is_utc(db):
        return 0
    refill_rows = list(refill_rows or ())
    if db.in_transaction:   # settle any pending implicit txn on this conn
        db.commit()
    db.execute("BEGIN IMMEDIATE")
    try:
        # Re-check inside the write lock (another process may have migrated
        # between the unlocked read above and BEGIN IMMEDIATE).
        if _fng_is_utc(db):
            db.rollback()
            return 0
        n_legacy = db.execute("SELECT COUNT(*) FROM fng_daily").fetchone()[0]
        n_refill = len({r[0] for r in refill_rows})
        if n_legacy and n_refill < _FNG_REFILL_MIN_RATIO * n_legacy:
            raise ValueError(
                f"fng_daily migration: refill has {n_refill} dates for "
                f"{n_legacy} legacy rows (< {_FNG_REFILL_MIN_RATIO:.0%}) -- "
                "refusing to clear fng_daily without a full UTC-dated refill")
        if n_legacy:
            db.execute(
                f"CREATE TABLE IF NOT EXISTS {_FNG_LEGACY_TABLE} ("
                "date TEXT PRIMARY KEY, value INTEGER NOT NULL, "
                "score REAL NOT NULL)")
            db.execute(
                f"INSERT OR IGNORE INTO {_FNG_LEGACY_TABLE} (date, value, score) "
                "SELECT date, value, score FROM fng_daily")
            missing = db.execute(
                f"SELECT COUNT(*) FROM fng_daily f LEFT JOIN {_FNG_LEGACY_TABLE} l "
                "ON l.date = f.date AND l.value = f.value "
                "WHERE l.date IS NULL").fetchone()[0]
            if missing:
                raise RuntimeError(
                    f"fng_daily migration: {missing} rows not preserved in "
                    f"{_FNG_LEGACY_TABLE} (pre-existing conflicting legacy "
                    "rows?) -- refusing to clear fng_daily")
            db.execute("DELETE FROM fng_daily")
        if refill_rows:
            db.executemany(
                "INSERT OR REPLACE INTO fng_daily (date, value, score) "
                "VALUES (?, ?, ?)", refill_rows)
        db.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                   (_FNG_BASIS_KEY, _FNG_BASIS_UTC))
        db.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
                   (_FNG_BASIS_KEY + '_migrated_at',
                    datetime.datetime.now(datetime.timezone.utc).isoformat()))
        db.commit()
    except Exception:
        db.rollback()
        raise
    if n_legacy:
        print(f"[SENTIMENT_HIST] FnG: migrated {n_legacy} local-tz-dated rows "
              f"to {_FNG_LEGACY_TABLE}; fng_daily refilled with {n_refill} "
              "UTC-dated days in the same transaction")
    return n_legacy


def _warn_fng_coverage(result, start_date, end_date, total_days, basis):
    """WARNING when fewer than _FNG_MIN_COVERAGE of the requested days carry
    a nonzero score -- the crypto harvest maps missing days to 0.0 and only
    prints a count, so a thin cache would otherwise train on zeros quietly."""
    nonzero = sum(1 for v in result.values() if v)
    if total_days > 0 and nonzero < _FNG_MIN_COVERAGE * total_days:
        log.warning("[SENTIMENT_HIST] FnG coverage LOW: %d/%d days in %s..%s "
                    "have a nonzero score (%.0f%% < %.0f%%; %s) -- the crypto "
                    "harvest fills the rest with Daily_Sentiment=0.0",
                    nonzero, total_days, start_date, end_date,
                    100.0 * nonzero / total_days, 100 * _FNG_MIN_COVERAGE,
                    basis)
        return False
    return True


def fetch_crypto_sentiment_history(start_date=None, end_date=None):
    """Fetch historical Crypto Fear & Greed Index and cache in SQLite.

    Args:
        start_date: str 'YYYY-MM-DD' or None (defaults to 365 days ago)
        end_date: str 'YYYY-MM-DD' or None (defaults to today)

    Returns:
        dict[str_date, float_score] — same score for all crypto symbols

    On a DB not yet migrated to UTC dating the full series is fetched FIRST
    and the migration (move-aside + clear + refill + marker) runs only on a
    successful fetch; if the fetch (or the migration) fails, the legacy
    local-tz rows are left in place, the marker is NOT set, and their
    values are returned with a WARNING -- they carry the old 1-day
    look-ahead leak but are not the all-zero feature a cleared table gave.
    Never raises for a fetch failure: the harvest maps a raise to an
    all-zero Daily_Sentiment column (scripts/harvest_crypto_data.py).
    """
    import requests

    db = _get_db()
    migrated = _fng_is_utc(db)

    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    # Check what we already have cached
    cached = db.execute(
        "SELECT date, score FROM fng_daily WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchall()
    cached_dates = {r[0] for r in cached}
    result = {r[0]: r[1] for r in cached}

    # Calculate how many days we need
    d_start = datetime.date.fromisoformat(start_date)
    d_end = datetime.date.fromisoformat(end_date)
    total_days = (d_end - d_start).days + 1
    needed = total_days - len(cached_dates)

    if migrated and needed <= 0:
        print(f"[SENTIMENT_HIST] FnG: {len(result)} days cached, 0 to fetch")
        _warn_fng_coverage(result, start_date, end_date, total_days,
                           'utc_publication')
        return result

    # Fetch from alternative.me (1 API call, free, no key)
    print(f"[SENTIMENT_HIST] FnG: fetching {total_days} days...")
    data = []
    try:
        # limit=0 = the full history. alternative.me counts `limit` back
        # from TODAY, not from end_date, so limit=total_days silently
        # dropped the earliest days of any range ending before today.
        resp = requests.get(
            'https://api.alternative.me/fng/?limit=0&format=json',
            timeout=15,
        )
        data = resp.json().get('data', []) or []
    except Exception as e:
        print(f"[SENTIMENT_HIST] FnG fetch error: {e}")

    fetched = []
    for entry in data:
        try:
            ts = int(entry['timestamp'])
            value = int(entry['value'])
        except (KeyError, TypeError, ValueError):
            continue
        # alternative.me timestamps are midnight UTC of the day the value is
        # PUBLISHED. Bucketing with the local timezone shifted every value
        # one day (US hosts), giving training bars up to 24h of future
        # sentiment. UTC bucketing makes day D's bars see exactly the value
        # known at D 00:00 UTC.
        date_str = datetime.datetime.fromtimestamp(
            ts, tz=datetime.timezone.utc).date().isoformat()
        fetched.append((date_str, value, _fng_value_to_score(value)))

    if not migrated:
        # One-time UTC re-dating, ONLY after a successful fetch (REVIEW M2).
        err = None
        if not fetched:
            err = 'fetch returned no rows'
        else:
            try:
                _migrate_fng_date_basis(db, fetched)
            except Exception as e:      # rolled back: fng_daily untouched
                err = f'migration refused: {e}'
        if err is not None:
            log.warning("[SENTIMENT_HIST] FnG UTC re-dating NOT done (%s): "
                        "legacy local-tz fng_daily kept, marker not set; "
                        "serving %d LEGACY values that still carry the 1-day "
                        "look-ahead leak (not zero) -- rerun the harvest once "
                        "alternative.me is reachable", err, len(result))
            _warn_fng_coverage(result, start_date, end_date, total_days,
                               'LEGACY local-tz')
            return result
        rows = db.execute(
            "SELECT date, score FROM fng_daily WHERE date >= ? AND date <= ?",
            (start_date, end_date)).fetchall()
        result = {r[0]: r[1] for r in rows}
        print(f"[SENTIMENT_HIST] FnG: UTC refill {len(fetched)} days, "
              f"{len(result)} in range")
        _warn_fng_coverage(result, start_date, end_date, total_days,
                           'utc_publication')
        return result

    inserted = 0
    for date_str, value, score in fetched:
        if date_str < start_date or date_str > end_date:
            continue
        if date_str in cached_dates:
            continue
        # The fetched value is known whether or not the cache write succeeds
        # (G4-02): a failed INSERT (e.g. 'database is locked') must not drop
        # it from the returned dict -- the harvest would stamp 0.0.
        result[date_str] = score
        try:
            db.execute(
                "INSERT OR IGNORE INTO fng_daily (date, value, score) VALUES (?, ?, ?)",
                (date_str, value, score),
            )
            inserted += 1
        except sqlite3.Error:
            pass                    # cache miss only; the next run re-inserts

    db.commit()
    print(f"[SENTIMENT_HIST] FnG: {inserted} new days cached, {len(result)} total")
    _warn_fng_coverage(result, start_date, end_date, total_days,
                       'utc_publication')
    return result


# ---------------------------------------------------------------------------
# Stock sentiment history (Finnhub)
# ---------------------------------------------------------------------------

def _get_finnhub():
    """Get Finnhub client. Returns None if unavailable."""
    api_key = os.getenv('FINNHUB_API_KEY')
    if not api_key:
        return None
    try:
        import finnhub
        return finnhub.Client(api_key=api_key)
    except ImportError:
        print("[SENTIMENT_HIST] finnhub-python not installed")
        return None


def fetch_stock_sentiment_history(tickers, start_date=None, end_date=None,
                                   cached_only=False):
    """Fetch historical news for stock tickers via Finnhub and keyword-score.

    Fetches in 30-day windows, rate-limited at 25 calls/min. Caches all
    articles in SQLite. Refresh policy (D27 dark half, c26-P5): a ticker
    with ZERO cached articles in [start_date, end_date] gets a full-range
    fetch; a ticker WITH cached articles gets an INCREMENTAL fetch from its
    newest cached article date (inclusive — same-day re-sends dedupe via
    the UNIQUE(symbol,date,headline) constraint) to end_date; a ticker
    cached through end_date is skipped. Harvest reads use cached_only=True
    and this network path runs only when the owner invokes it, so training
    VALUES change only after the owner runs this fetch AND the next
    harvest+retrain — feature reactivation stays an owner decision.

    Args:
        tickers: List of stock ticker symbols (crypto symbols with '/' are skipped)
        start_date: str 'YYYY-MM-DD' or None (defaults to 365 days ago)
        end_date: str 'YYYY-MM-DD' or None (defaults to today)
        cached_only: If True, only return already-cached data (no network calls)

    Returns:
        dict[(ticker, date_str), float_score]
    """
    if cached_only:
        db = _get_db()
        if start_date is None:
            start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
        if end_date is None:
            end_date = datetime.date.today().isoformat()
        rows = db.execute(
            "SELECT symbol, date, score FROM daily_sentiment WHERE date >= ? AND date <= ?",
            (start_date, end_date),
        ).fetchall()
        return {(sym, dt): score for sym, dt, score in rows}

    client = _get_finnhub()
    if client is None:
        print("[SENTIMENT_HIST] No Finnhub API key — stock sentiment will be 0.0")
        return {}

    db = _get_db()

    if start_date is None:
        start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    # Filter out crypto symbols
    stock_tickers = [t for t in tickers if '/' not in t and '-USD' not in t]

    # Load existing cached results
    result = {}
    cached_rows = db.execute(
        "SELECT symbol, date, score FROM daily_sentiment WHERE date >= ? AND date <= ?",
        (start_date, end_date),
    ).fetchall()
    for sym, dt, score in cached_rows:
        result[(sym, dt)] = score

    d_start = datetime.date.fromisoformat(start_date)
    d_end = datetime.date.fromisoformat(end_date)

    # Determine which symbols need fetching, and from which date:
    #   zero cached articles in range -> full-range fetch from d_start
    #   cached but tail < end_date    -> incremental fetch from the newest
    #                                    cached article date (inclusive)
    #   cached through end_date       -> skip
    tickers_to_fetch = []  # (ticker, fetch_start: datetime.date)
    n_full = n_incr = 0
    for ticker in stock_tickers:
        max_dt = db.execute(
            "SELECT MAX(date) FROM articles WHERE symbol=? AND date >= ? AND date <= ?",
            (ticker, start_date, end_date),
        ).fetchone()[0]
        if max_dt is None:
            tickers_to_fetch.append((ticker, d_start))
            n_full += 1
        elif max_dt < end_date:
            tickers_to_fetch.append((ticker, datetime.date.fromisoformat(max_dt)))
            n_incr += 1

    if not tickers_to_fetch:
        print(f"[SENTIMENT_HIST] All {len(stock_tickers)} tickers current, "
              f"{len(result)} daily scores")
        return result

    print(f"[SENTIMENT_HIST] Fetching news for {len(tickers_to_fetch)} tickers "
          f"({n_full} full, {n_incr} incremental, "
          f"{len(stock_tickers) - len(tickers_to_fetch)} current)...")

    # Rate limiter: 25 calls/min (Finnhub free tier = 30/min, leave headroom)
    call_times = []
    calls_per_min = 25

    total_articles = 0
    now_iso = datetime.datetime.now().isoformat()

    for ti, (ticker, fetch_start) in enumerate(tickers_to_fetch):
      try:
        ticker_articles = 0
        ticker_scores = []

        # Fetch in 30-day windows (reduces API calls ~4x vs 7-day)
        window_start = fetch_start
        while window_start <= d_end:
            window_end = min(window_start + datetime.timedelta(days=29), d_end)

            # Rate limit
            now = time.time()
            call_times = [t for t in call_times if now - t < 60]
            if len(call_times) >= calls_per_min:
                sleep_time = 60 - (now - call_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
            call_times.append(time.time())

            # Retry with exponential backoff on rate limits
            articles = None
            for attempt in range(3):
                try:
                    articles = client.company_news(
                        ticker,
                        _from=window_start.isoformat(),
                        to=window_end.isoformat(),
                    )
                    break
                except Exception as e:
                    if '429' in str(e):
                        if attempt == 2:
                            # G4-08: no retry follows the last attempt, so a
                            # back-off sleep here would buy nothing.
                            print(f"[SENTIMENT_HIST] Rate limited, giving up on "
                                  f"{ticker} {window_start}..{window_end}")
                            break
                        wait = 62 * (2 ** attempt)  # 62s, 124s
                        print(f"[SENTIMENT_HIST] Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"[SENTIMENT_HIST] Finnhub error {ticker} "
                              f"{window_start}..{window_end}: {e}")
                        break

            if articles is None:
                window_start = window_end + datetime.timedelta(days=1)
                continue

            for a in articles:
                # `or ''`: Finnhub sends present-but-None fields (G4-03), which
                # .get(k, '') would return as None.
                headline = (a.get('headline') or '').strip()
                if not headline:
                    continue
                summary = (a.get('summary') or '').strip()
                url = (a.get('url') or '').strip()

                # Determine article date from datetime field (UTC — local
                # bucketing shifted dates and leaked future articles into
                # the prior day's training bars)
                article_ts = a.get('datetime', 0)
                if article_ts:
                    article_date = datetime.datetime.fromtimestamp(
                        article_ts, tz=datetime.timezone.utc).date().isoformat()
                else:
                    article_date = window_start.isoformat()

                score = _keyword_score(headline, summary)

                cur = db.execute(
                    """INSERT OR IGNORE INTO articles
                       (symbol, date, headline, summary, url, keyword_score, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ticker, article_date, headline, summary, url, score, now_iso),
                )
                # OR IGNORE swallows duplicates instead of raising, so count
                # real inserts via rowcount (0 for an ignored row).
                ticker_articles += cur.rowcount

            # G4-01: commit per window so no write transaction spans the next
            # network call, rate-limit sleep or 429 back-off (other writers on
            # this WAL DB would hit the busy timeout -> 'database is locked').
            db.commit()
            window_start = window_end + datetime.timedelta(days=1)

        db.commit()

        # Aggregate daily scores for this ticker
        dates_with_articles = db.execute(
            "SELECT DISTINCT date FROM articles WHERE symbol=? AND date >= ? AND date <= ?",
            (ticker, fetch_start.isoformat(), end_date),
        ).fetchall()

        for (dt,) in dates_with_articles:
            _aggregate_daily(db, ticker, dt)

        db.commit()

        # Read back aggregated scores
        rows = db.execute(
            "SELECT date, score FROM daily_sentiment WHERE symbol=? AND date >= ? AND date <= ?",
            (ticker, start_date, end_date),
        ).fetchall()
        for dt, score in rows:
            result[(ticker, dt)] = score
            ticker_scores.append(score)

        avg_score = sum(ticker_scores) / len(ticker_scores) if ticker_scores else 0.0
        total_articles += ticker_articles
        print(f"[SENTIMENT_HIST] {ticker}: {ticker_articles} articles, "
              f"avg {avg_score:+.2f}  [{ti + 1}/{len(tickers_to_fetch)}]")
      except Exception as e:
        print(f"[SENTIMENT_HIST] Skipping {ticker}: {e}")
        continue

    print(f"[SENTIMENT_HIST] Done: {total_articles} total articles, "
          f"{len(result)} daily scores")
    return result


# ---------------------------------------------------------------------------
# Daily sentiment lookups (for harvest and live inference)
# ---------------------------------------------------------------------------

def get_daily_sentiment(symbol, date_str):
    """Read daily sentiment from DB. Returns float or 0.0 if missing."""
    db = _get_db()
    row = db.execute(
        "SELECT score FROM daily_sentiment WHERE symbol=? AND date=?",
        (symbol, date_str),
    ).fetchone()
    return row[0] if row else 0.0


def stock_sentiment_lookup_dates(index):
    """daily_sentiment date key ('YYYY-MM-DD') for each stock bar timestamp.

    Rule: ((t - 6h).date() - 1 day), t in UTC (a naive index is taken as
    UTC). The legacy article cache is America/Chicago-dated, so the
    aggregate for calendar day X includes articles published up to
    (X+1) ~05:00-06:00 UTC. The plain D-1 rule was PIT-safe for pre-market
    and RTH bars but NOT for the 00:00 UTC extended-hours bar, which could
    see ~6h of later articles; the -6h shift sends every bar before 06:00
    UTC one more day back so the looked-up bucket is strictly in the past.
    Since t - 6h ~= Chicago wall-clock, this equals the live rule
    (get_live_daily_sentiment: local today - 1 on the Chicago box) for
    every bar outside 05:00-06:00 UTC in daylight time, where serve is one
    day newer but still PIT-complete. UTC-dated articles (current insert
    code) are strictly older under this key as well.
    """
    import pandas as pd
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert('UTC')
    shifted = (idx - pd.Timedelta(hours=6)).date
    return [str(d - datetime.timedelta(days=1)) for d in shifted]


_live_fng_warned = False


def get_live_daily_sentiment(symbol, asset_type='crypto'):
    """Get the sentiment value live inference should see RIGHT NOW.

    Crypto: today's FnG (published at 00:00 UTC — fully known intraday).
    Stocks: YESTERDAY's completed daily score — training attributes day
    D-1's aggregate to day-D bars (a day-D aggregate includes articles
    published after the current bar), so serving must match.

    Returns float in [-1, 1], defaults 0.0.
    """
    global _live_fng_warned
    if asset_type == 'crypto':
        try:
            from sentiment import get_fear_greed
            fng = get_fear_greed()
            if fng is not None:
                return _fng_value_to_score(fng['value'])
        except Exception as e:
            # Log once — a persistently broken source was invisible because
            # predict_now's wrapper also defaults to 0.0 without logging.
            if not _live_fng_warned:
                _live_fng_warned = True
                print(f"[SENTIMENT_HIST] live FnG failed (logged once): {e}")
        return 0.0
    else:
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        # Clean symbol for DB lookup (strip any exchange suffix)
        clean = symbol.replace('/', '').replace('-USD', '')
        return get_daily_sentiment(clean, yesterday)


# ---------------------------------------------------------------------------
# Background LLM backfill worker
# ---------------------------------------------------------------------------

def set_live_mode(active):
    """Set coordination flag for trading bots. Worker pauses when active."""
    db = _get_db()
    db.execute(
        "INSERT INTO state (key, value) VALUES ('live_mode', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        ('1' if active else '0',),
    )
    db.commit()


def _is_live_mode():
    """Check if trading bots are actively running."""
    db = _get_db()
    row = db.execute("SELECT value FROM state WHERE key='live_mode'").fetchone()
    return row is not None and row[0] == '1'


def get_backfill_stats():
    """Get backfill progress stats (used by this module's --stats CLI)."""
    db = _get_db()
    total = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    scored = db.execute(
        "SELECT COUNT(*) FROM articles WHERE llm_score IS NOT NULL"
    ).fetchone()[0]
    return {
        'total_articles': total,
        'llm_scored': scored,
        'remaining': total - scored,
        'pct_complete': (scored / total * 100) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Gemini Batch API backfill (50% price, separate quota from live calls)
# ---------------------------------------------------------------------------

_BATCH_CHUNK = 25            # articles per generateContent request
_BATCH_MAX_ARTICLES = 2000   # per batch job
_BATCH_POLL_SEC = 120
_BATCH_MAX_AGE_H = 48        # a job pending this long is dead — clear it
_batch_unavailable_until = 0.0

_BATCH_SCORE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "i": {"type": "INTEGER"},
                    "s": {"type": "NUMBER"},
                },
                "required": ["i", "s"],
            },
        },
    },
    "required": ["scores"],
}

_BATCH_SCORE_PROMPT = """\
Score the market sentiment of each numbered financial news item for the
bracketed symbol, on a -1.0 to +1.0 scale. Anchor examples:
  -0.8: bankruptcy filing, fraud charges, delisting notice
  -0.3: earnings miss, guidance cut, analyst downgrade
   0.0: routine/neutral news, no market implication
  +0.3: earnings beat, contract win, analyst upgrade
  +0.8: transformative acquisition at premium, blockbuster approval
Use precise values (e.g. -0.45, +0.2). Score the impact on the BRACKETED
symbol specifically, not the market. Respond per the JSON schema with one
entry per item, keyed by its number `i`.

Items:
"""


def _batch_state_get(db):
    row = db.execute("SELECT value FROM state WHERE key='pending_batch'").fetchone()
    if not row or not row[0]:
        return None
    try:
        import json as _json
        return _json.loads(row[0])
    except Exception:
        return None


def _batch_state_set(db, payload):
    import json as _json
    if payload is None:
        db.execute("DELETE FROM state WHERE key='pending_batch'")
    else:
        db.execute(
            "INSERT INTO state (key, value) VALUES ('pending_batch', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (_json.dumps(payload),))
    db.commit()


def _gemini_batch_http(method, url_path, body=None, api_key=''):
    import json as _json
    import urllib.request
    url = f"https://generativelanguage.googleapis.com/v1beta/{url_path}"
    data = _json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return _json.loads(resp.read())


def submit_sentiment_batch(db, model=None) -> bool:
    """Submit unscored articles as ONE Gemini Batch job (inline requests).

    Batch pricing is 50% of interactive and draws from a SEPARATE enqueued
    quota — the old synchronous 8-RPM worker competed with the live trading
    loops for the same RPM/RPD all day.
    """
    global _batch_unavailable_until
    from llm_config import load_llm_config
    from llm_client import get_recommended_model

    config = load_llm_config()
    api_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not api_key or not config.get("enabled"):
        return False

    rows = db.execute(
        """SELECT id, symbol, date, headline, summary FROM articles
           WHERE llm_score IS NULL ORDER BY date DESC LIMIT ?""",
        (_BATCH_MAX_ARTICLES,)).fetchall()
    if not rows:
        return False

    model = model or get_recommended_model('backfill')
    requests_payload = []
    id_map = []  # per chunk: [[article_id, symbol, date], ...]
    for start in range(0, len(rows), _BATCH_CHUNK):
        chunk = rows[start:start + _BATCH_CHUNK]
        lines = []
        for i, (aid, symbol, date, headline, summary) in enumerate(chunk):
            text = headline if not summary else f"{headline} — {summary[:200]}"
            lines.append(f"{i}. [{symbol}] {text}")
        prompt = _BATCH_SCORE_PROMPT + "\n".join(lines)
        requests_payload.append({
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json",
                    "responseSchema": _BATCH_SCORE_SCHEMA,
                },
            },
            "metadata": {"key": f"chunk-{start // _BATCH_CHUNK}"},
        })
        id_map.append([[aid, symbol, date] for aid, symbol, date, _h, _s in chunk])

    body = {"batch": {
        "display_name": f"sentiment-backfill-{datetime.date.today().isoformat()}",
        "input_config": {"requests": {"requests": requests_payload}},
    }}
    try:
        resp = _gemini_batch_http('POST', f"models/{model}:batchGenerateContent",
                                  body, api_key)
        name = resp.get('name')
        if not name:
            raise ValueError(f"no batch name in response: {str(resp)[:200]}")
        _batch_state_set(db, {'name': name, 'model': model, 'id_map': id_map,
                              'submitted_at': datetime.datetime.now().isoformat()})
        print(f"[BACKFILL] Batch submitted: {name} "
              f"({len(rows)} articles in {len(requests_payload)} chunks)")
        return True
    except Exception as e:
        print(f"[BACKFILL] Batch submit failed ({e}) — sync fallback for 1h")
        _batch_unavailable_until = time.time() + 3600
        return False


def poll_and_ingest_batch(db) -> str:
    """Poll the pending batch job. Returns 'none'|'pending'|'ingested'|'failed'."""
    import json as _json
    import urllib.error
    from llm_config import load_llm_config

    state = _batch_state_get(db)
    if not state:
        return 'none'
    config = load_llm_config()
    api_key = config.get("models", {}).get("gemini", {}).get("api_key", "")
    if not api_key:
        return 'none'

    # Staleness guard: a job pending for days is gone server-side (or the
    # persisted state is malformed) — clear it so the worker can move on
    # instead of polling forever.
    try:
        age_h = (datetime.datetime.now()
                 - datetime.datetime.fromisoformat(state['submitted_at'])
                 ).total_seconds() / 3600.0
    except (KeyError, TypeError, ValueError):
        age_h = float('inf')
    if age_h > _BATCH_MAX_AGE_H:
        print(f"[BACKFILL] Batch state older than {_BATCH_MAX_AGE_H}h — clearing")
        _batch_state_set(db, None)
        return 'failed'

    try:
        resp = _gemini_batch_http('GET', state['name'], None, api_key)
    except urllib.error.HTTPError as e:
        # 4xx is permanent (job deleted/expired server-side, revoked key
        # scope) — except 408/429 which are transient. Returning 'pending'
        # on these wedged the worker in a poll loop forever.
        if 400 <= e.code < 500 and e.code not in (408, 429):
            print(f"[BACKFILL] Batch poll HTTP {e.code} — clearing dead job")
            _batch_state_set(db, None)
            return 'failed'
        print(f"[BACKFILL] Batch poll failed: {e}")
        return 'pending'
    except Exception as e:
        print(f"[BACKFILL] Batch poll failed: {e}")
        return 'pending'  # transient — try again next pass

    job_state = (resp.get('metadata') or {}).get('state', '')
    if job_state in ('JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
        print(f"[BACKFILL] Batch {job_state} — clearing")
        _batch_state_set(db, None)
        return 'failed'
    if job_state != 'JOB_STATE_SUCCEEDED':
        return 'pending'

    # Ingest: match responses to chunks by the echoed metadata key (robust
    # to reordering/omission); fall back to position, which the API
    # preserves today.
    inlined = ((resp.get('response') or {}).get('inlinedResponses')) or []
    id_map = state.get('id_map', [])
    now_iso = datetime.datetime.now().isoformat()
    scored = 0
    updated_pairs = set()
    for pos, item in enumerate(inlined):
        key = (item.get('metadata') or {}).get('key') or ''
        try:
            chunk_idx = int(key.split('-')[1])
        except (IndexError, ValueError):
            chunk_idx = pos
        if not 0 <= chunk_idx < len(id_map):
            continue
        chunk_ids = id_map[chunk_idx]
        cand = (item.get('response') or {}).get('candidates') or []
        if not cand:
            continue
        parts = (cand[0].get('content') or {}).get('parts') or []
        text = next((p.get('text', '') for p in reversed(parts)
                     if p.get('text', '').strip()), '')
        try:
            payload = _json.loads(text)
            entries = payload.get('scores', [])
        except (ValueError, AttributeError):
            continue
        for entry in entries:
            try:
                i = int(entry['i'])
                s = float(entry['s'])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(s):
                continue  # G4-07: NaN would clamp to +1.0; skip = unscored
            s = max(-1.0, min(1.0, s))
            if 0 <= i < len(chunk_ids):
                aid, symbol, date = chunk_ids[i]
                db.execute(
                    "UPDATE articles SET llm_score=?, llm_scored_at=? WHERE id=?",
                    (s, now_iso, aid))
                updated_pairs.add((symbol, date))
                scored += 1

    for symbol, date_str in updated_pairs:
        _aggregate_daily(db, symbol, date_str)
    db.commit()
    _batch_state_set(db, None)
    print(f"[BACKFILL] Batch ingested: {scored} articles scored, "
          f"{len(updated_pairs)} daily aggregates refreshed")
    return 'ingested'


def run_backfill_worker(max_rpm=8):
    """Background worker: LLM-score unscored articles, newest first.

    Runs WHILE the bots trade (max_rpm=8 deliberately leaves quota headroom
    for live calls) and PAUSES during training windows, when the Jetson's
    memory and the LLM quota both belong to the retrain. The old check was
    inverted: it paused whenever live_mode was set (i.e. roughly always,
    since the crypto bot runs 24/7) and only ran during retrains —
    exactly backwards.

    Args:
        max_rpm: Maximum LLM API calls per minute
    """
    global _batch_unavailable_until
    try:
        from sentiment import _llm_score_batch, _validate_text
    except ImportError:
        print("[BACKFILL] Cannot import sentiment scoring — aborting")
        return

    print("[BACKFILL] Worker started")
    batch_num = 0
    min_interval = 60.0 / max_rpm  # seconds between batches

    while True:
        # live_mode=True -> bots trading -> we run (throttled).
        # live_mode=False -> training window -> we pause.
        if not _is_live_mode():
            time.sleep(60)
            continue

        db = _get_db()

        # --- Batch-first: Gemini Batch API is 50% price on a SEPARATE
        # quota (zero contention with the live loops). Poll any pending
        # job; otherwise submit a new one. Synchronous scoring below is
        # the fallback when batch is unavailable.
        if time.time() >= _batch_unavailable_until:
            status = poll_and_ingest_batch(db)
            if status == 'pending':
                time.sleep(_BATCH_POLL_SEC)
                continue
            if status == 'ingested':
                continue  # check immediately for more work
            if status == 'failed':
                # A server-side failure is often systemic (schema/model
                # rejection) — resubmitting the same articles immediately
                # just churns failing jobs. Cool off on the sync fallback.
                _batch_unavailable_until = time.time() + 3600
            # 'none' — try submitting a fresh job
            elif submit_sentiment_batch(db):
                time.sleep(_BATCH_POLL_SEC)
                continue
            # submit returned False: nothing unscored, or batch unavailable

        # Get next batch of unscored articles (newest first)
        rows = db.execute(
            """SELECT id, symbol, date, headline, summary, url
               FROM articles WHERE llm_score IS NULL
               ORDER BY date DESC LIMIT 50""",
        ).fetchall()

        if not rows:
            print("[BACKFILL] All articles scored. Sleeping 1 hour...")
            time.sleep(3600)
            continue

        batch_num += 1
        remaining = db.execute(
            "SELECT COUNT(*) FROM articles WHERE llm_score IS NULL"
        ).fetchone()[0]

        # Build article dicts for LLM scoring
        articles = []
        for row in rows:
            articles.append({
                'id': row[0],
                'symbol': row[1],
                'date': row[2],
                'headline': row[3],
                'summary': row[4],
                'url': row[5],
            })

        # Score batch via LLM
        start = time.time()
        llm_articles = [{'headline': a['headline'], 'summary': a['summary'],
                         'url': a['url']} for a in articles]
        scores = _llm_score_batch(llm_articles)

        if scores is None:
            print(f"[BACKFILL] Batch {batch_num}: LLM unavailable, sleeping 5 min...")
            time.sleep(300)
            continue

        # Update articles with LLM scores
        now_iso = datetime.datetime.now().isoformat()
        updated_pairs = set()  # (symbol, date) pairs to re-aggregate
        for a, score in zip(articles, scores):
            db.execute(
                "UPDATE articles SET llm_score=?, llm_scored_at=? WHERE id=?",
                (score, now_iso, a['id']),
            )
            updated_pairs.add((a['symbol'], a['date']))

        # Re-aggregate affected daily scores
        for symbol, date_str in updated_pairs:
            _aggregate_daily(db, symbol, date_str)

        db.commit()

        elapsed = time.time() - start
        date_range = articles[-1]['date'] if articles else '?'
        print(f"[BACKFILL] Batch {batch_num}: scored {len(articles)} articles "
              f"({date_range}), {remaining - len(articles)} remaining, "
              f"{elapsed:.1f}s")

        # Rate limit
        sleep_time = max(0, min_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# CLI entry point for standalone backfill
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Sentiment history tools')
    parser.add_argument('--backfill', action='store_true',
                        help='Run LLM backfill worker')
    parser.add_argument('--fetch-stocks', action='store_true',
                        help='Fetch stock sentiment history (Finnhub)')
    parser.add_argument('--stats', action='store_true',
                        help='Show backfill statistics')
    parser.add_argument('--rpm', type=int, default=8,
                        help='Max LLM calls per minute (default: 8)')
    args = parser.parse_args()

    if args.stats:
        stats = get_backfill_stats()
        print(f"Total articles:  {stats['total_articles']}")
        print(f"LLM scored:      {stats['llm_scored']}")
        print(f"Remaining:       {stats['remaining']}")
        print(f"Progress:        {stats['pct_complete']:.1f}%")
    elif args.fetch_stocks:
        from stock_config import load_stock_universe
        tickers = [t for t in load_stock_universe() if '/' not in t]
        fetch_stock_sentiment_history(tickers)
    elif args.backfill:
        run_backfill_worker(max_rpm=args.rpm)
    else:
        parser.print_help()
