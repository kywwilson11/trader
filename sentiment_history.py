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
import math
import os
import sqlite3
import time
import threading

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

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
    """Score text using keyword-based sentiment. Returns float in (-1, 1)."""
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


def fetch_crypto_sentiment_history(start_date=None, end_date=None):
    """Fetch historical Crypto Fear & Greed Index and cache in SQLite.

    Args:
        start_date: str 'YYYY-MM-DD' or None (defaults to 365 days ago)
        end_date: str 'YYYY-MM-DD' or None (defaults to today)

    Returns:
        dict[str_date, float_score] — same score for all crypto symbols
    """
    import requests

    db = _get_db()

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

    if needed <= 0:
        print(f"[SENTIMENT_HIST] FnG: {len(result)} days cached, 0 to fetch")
        return result

    # Fetch from alternative.me (1 API call, free, no key)
    print(f"[SENTIMENT_HIST] FnG: fetching {total_days} days...")
    try:
        resp = requests.get(
            f'https://api.alternative.me/fng/?limit={total_days}&format=json',
            timeout=15,
        )
        data = resp.json().get('data', [])
    except Exception as e:
        print(f"[SENTIMENT_HIST] FnG fetch error: {e}")
        return result

    inserted = 0
    for entry in data:
        ts = int(entry['timestamp'])
        date_str = datetime.date.fromtimestamp(ts).isoformat()
        if date_str < start_date or date_str > end_date:
            continue
        if date_str in cached_dates:
            continue
        value = int(entry['value'])
        score = _fng_value_to_score(value)
        try:
            db.execute(
                "INSERT OR IGNORE INTO fng_daily (date, value, score) VALUES (?, ?, ?)",
                (date_str, value, score),
            )
            result[date_str] = score
            inserted += 1
        except sqlite3.Error:
            pass

    db.commit()
    print(f"[SENTIMENT_HIST] FnG: {inserted} new days cached, {len(result)} total")
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

    Fetches in 7-day windows, rate-limited at 25 calls/min. Caches all articles
    in SQLite — subsequent runs only fetch new/uncached date ranges.

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
    cached_symbols = set()
    for sym, dt, score in cached_rows:
        result[(sym, dt)] = score
        cached_symbols.add(sym)

    # Determine which symbols need fetching
    # A symbol needs fetching if it has no articles in our date range
    tickers_to_fetch = []
    for ticker in stock_tickers:
        count = db.execute(
            "SELECT COUNT(*) FROM articles WHERE symbol=? AND date >= ? AND date <= ?",
            (ticker, start_date, end_date),
        ).fetchone()[0]
        if count == 0:
            tickers_to_fetch.append(ticker)

    if not tickers_to_fetch:
        print(f"[SENTIMENT_HIST] All {len(stock_tickers)} tickers cached, {len(result)} daily scores")
        return result

    print(f"[SENTIMENT_HIST] Fetching news for {len(tickers_to_fetch)} tickers "
          f"({len(stock_tickers) - len(tickers_to_fetch)} cached)...")

    d_start = datetime.date.fromisoformat(start_date)
    d_end = datetime.date.fromisoformat(end_date)

    # Rate limiter: 25 calls/min (Finnhub free tier = 30/min, leave headroom)
    call_times = []
    calls_per_min = 25

    total_articles = 0
    now_iso = datetime.datetime.now().isoformat()

    for ti, ticker in enumerate(tickers_to_fetch):
      try:
        ticker_articles = 0
        ticker_scores = []

        # Fetch in 30-day windows (reduces API calls ~4x vs 7-day)
        window_start = d_start
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
                        wait = 62 * (2 ** attempt)  # 62s, 124s, 248s
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
                headline = a.get('headline', '').strip()
                if not headline:
                    continue
                summary = a.get('summary', '').strip()
                url = a.get('url', '').strip()

                # Determine article date from datetime field
                article_ts = a.get('datetime', 0)
                if article_ts:
                    article_date = datetime.date.fromtimestamp(article_ts).isoformat()
                else:
                    article_date = window_start.isoformat()

                score = _keyword_score(headline, summary)

                try:
                    db.execute(
                        """INSERT OR IGNORE INTO articles
                           (symbol, date, headline, summary, url, keyword_score, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (ticker, article_date, headline, summary, url, score, now_iso),
                    )
                    ticker_articles += 1
                except sqlite3.IntegrityError:
                    pass  # duplicate

            window_start = window_end + datetime.timedelta(days=1)

        db.commit()

        # Aggregate daily scores for this ticker
        dates_with_articles = db.execute(
            "SELECT DISTINCT date FROM articles WHERE symbol=? AND date >= ? AND date <= ?",
            (ticker, start_date, end_date),
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


def get_live_daily_sentiment(symbol, asset_type='crypto'):
    """Get today's sentiment for live inference.

    Crypto: reads today's FnG (via existing sentiment.get_fear_greed())
    Stocks: reads today's daily_sentiment from DB, or returns 0.0

    Returns float in [-1, 1], defaults 0.0.
    """
    if asset_type == 'crypto':
        try:
            from sentiment import get_fear_greed
            fng = get_fear_greed()
            if fng is not None:
                return _fng_value_to_score(fng['value'])
        except Exception:
            pass
        return 0.0
    else:
        today = datetime.date.today().isoformat()
        # Clean symbol for DB lookup (strip any exchange suffix)
        clean = symbol.replace('/', '').replace('-USD', '')
        return get_daily_sentiment(clean, today)


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
    """Get backfill progress stats for GUI display."""
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


def run_backfill_worker(max_rpm=8):
    """Background worker: LLM-score unscored articles, newest first.

    Respects live_mode flag — pauses when trading bots are active.
    Rate-limits to max_rpm requests per minute (default 8, leaving headroom
    for live sentiment calls at 15 RPM).

    Args:
        max_rpm: Maximum LLM API calls per minute
    """
    try:
        from sentiment import _llm_score_batch, _validate_text
    except ImportError:
        print("[BACKFILL] Cannot import sentiment scoring — aborting")
        return

    print("[BACKFILL] Worker started")
    batch_num = 0
    min_interval = 60.0 / max_rpm  # seconds between batches

    while True:
        # Check live mode — sleep longer when bots are trading
        if _is_live_mode():
            time.sleep(60)
            continue

        db = _get_db()

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
