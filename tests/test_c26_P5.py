"""c26-P5: funding.py time-thinning flag (D28) + sentiment_history.py
incremental refresh (D27 dark half).

Runs on the dev Mac: pure stdlib + monkeypatch, no torch/lightgbm/finnhub.
funding_archive is pure-pandas and importable/monkeypatchable here (see
tests/test_grp_deriv.py). sentiment_history._keyword_score is monkeypatched
to a constant so sentiment.py (and its heavier scoring path) is never
imported; the fake Finnhub client means finnhub-python is never imported
either.

FUNDING (D28): pins the flag-OFF value-thinned append/z-baseline behavior
byte-identical to pre-change, and exercises the flag-ON time-thinned append
+ archive-preferred z baseline (mirroring live_funding_features).

SENTIMENT (D27 dark half): pins the new refresh policy — zero cached
articles -> full-range fetch; cached-but-stale -> incremental fetch from
the newest cached article date (inclusive, dedup on overlap); cached
through end_date -> skipped entirely. cached_only=True stays untouched
(no network, ever).
"""

import datetime
import json
import logging
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import funding
import funding_archive
import sentiment_history as sh


# ---------------------------------------------------------------------------
# funding.py fixtures/helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def funding_env(tmp_path, monkeypatch):
    """Isolate funding.py module state and its history file."""
    monkeypatch.setattr(funding, '_HISTORY_FILE', str(tmp_path / 'fh.json'))
    monkeypatch.setattr(funding, '_cache', {})
    monkeypatch.setattr(funding, '_history', None)
    monkeypatch.setattr(funding, '_save_warned', False)
    monkeypatch.setattr(funding, '_thin_advice_logged', False)
    return tmp_path


class _Resp:
    """Fake urlopen response usable as a context manager."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _rate_resp(rate: float) -> _Resp:
    return _Resp(json.dumps({'data': [{'fundingRate': str(rate)}]}).encode())


class _FakeSeries:
    """Minimal stand-in for the funding_archive Series-like return."""

    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)


# ---------------------------------------------------------------------------
# funding.py: D28 flag default + append behavior
# ---------------------------------------------------------------------------

def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv('TRADER_FUNDING_Z_TIME_THINNING', raising=False)
    assert funding.FUNDING_Z_TIME_THINNING is False


def test_default_appends_every_changed_rate(funding_env, monkeypatch):
    """PINS current behavior: flag OFF appends on any value drift."""
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', False)
    r1 = 0.0001
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: _rate_resp(r1))
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(r1)

    funding._cache.clear()
    r2 = r1 + 1e-8
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: _rate_resp(r2))
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(r2)

    hist = funding._load_history()['BTC/USD']
    assert len(hist) == 2


def test_default_identical_rate_not_appended(funding_env, monkeypatch):
    """PINS current behavior: flag OFF, unchanged rate is not re-appended."""
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', False)
    r = 0.0001
    monkeypatch.setattr(funding, '_history', {'BTC/USD': [r, r, r]})
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: _rate_resp(r))
    funding.get_funding_rate('BTC/USD')
    assert len(funding._history['BTC/USD']) == 3


def test_default_advice_warning_once(funding_env, monkeypatch, caplog):
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', False)
    # Below the len>=30 gate: no warning even on an appending drift.
    monkeypatch.setattr(funding, '_history', {'BTC/USD': [0.0001] * 5})
    with caplog.at_level(logging.WARNING, logger='funding'):
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: _rate_resp(0.0009))
        funding.get_funding_rate('BTC/USD')
    early_warns = [r for r in caplog.records
                   if 'TRADER_FUNDING_Z_TIME_THINNING' in r.getMessage()]
    assert len(early_warns) == 0

    # At/above the gate: exactly one warning across two further appends.
    seed = [0.0001 + i * 1e-9 for i in range(30)]
    monkeypatch.setattr(funding, '_history', {'BTC/USD': list(seed)})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='funding'):
        funding._cache.clear()
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: _rate_resp(0.0002))
        funding.get_funding_rate('BTC/USD')
        funding._cache.clear()
        monkeypatch.setattr(urllib.request, 'urlopen',
                            lambda req, timeout=10: _rate_resp(0.0003))
        funding.get_funding_rate('BTC/USD')
    warns = [r for r in caplog.records
             if 'TRADER_FUNDING_Z_TIME_THINNING' in r.getMessage()]
    assert len(warns) == 1
    assert warns[0].levelno == logging.WARNING


# ---------------------------------------------------------------------------
# funding.py: D28 flag-ON time thinning
# ---------------------------------------------------------------------------

def test_thinning_on_spaces_appends_by_time(funding_env, monkeypatch):
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', True)
    clock = {'t': 1_700_000_000.0}
    monkeypatch.setattr(funding.time, 'time', lambda: clock['t'])
    rate = 0.0001
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: _rate_resp(rate))

    funding.get_funding_rate('BTC/USD')          # t0: first append
    funding._cache.clear()

    clock['t'] += 100                             # +100s: too soon
    funding.get_funding_rate('BTC/USD')
    funding._cache.clear()
    assert len(funding._load_history()['BTC/USD']) == 1

    clock['t'] = 1_700_000_000.0 + 27_001          # +27001s from t0: due
    funding.get_funding_rate('BTC/USD')
    hist = funding._load_history()['BTC/USD']
    assert len(hist) == 2

    # The sidecar epoch must be PERSISTED, not just in-memory: read the
    # on-disk JSON directly.
    with open(funding._HISTORY_FILE) as f:
        on_disk = json.load(f)
    assert on_disk[funding._TS_KEY]['BTC/USD'] == pytest.approx(
        1_700_000_000.0 + 27_001)
    assert on_disk['BTC/USD'] == [pytest.approx(rate)] * 2


def test_thinning_file_roundtrip_and_reader_isolation(funding_env, monkeypatch):
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', True)
    clock = {'t': 1_700_000_000.0}
    monkeypatch.setattr(funding.time, 'time', lambda: clock['t'])
    rate = 0.0001
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: _rate_resp(rate))
    funding.get_funding_rate('BTC/USD')

    # Force a fresh disk reload — the sidecar key must not corrupt the
    # per-symbol list a reader pulls back out.
    monkeypatch.setattr(funding, '_history', None)
    raw = funding._load_history()
    assert funding._TS_KEY in raw
    assert isinstance(raw[funding._TS_KEY], dict)
    hist = raw['BTC/USD']
    assert isinstance(hist, list)
    assert all(isinstance(v, float) for v in hist)
    assert hist == [pytest.approx(rate)]

    # Reader isolation: funding_tilt (flag OFF) must ignore the sidecar key
    # and behave normally against the same on-disk file.
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', False)
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: rate)
    assert funding.funding_tilt('BTC/USD') == 1.0


# ---------------------------------------------------------------------------
# funding.py: D28 funding_tilt z-baseline source
# ---------------------------------------------------------------------------

def test_tilt_default_ignores_archive(funding_env, monkeypatch):
    """PINS current behavior: flag OFF, tilt never touches funding_archive."""
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', False)
    mu, sd = 0.0, 1e-6
    hist = [mu - sd, mu + sd] * 15   # 30 samples, pstdev == sd
    rate = mu + 2.5 * sd             # z = 2.5 -> crowded; ann tiny
    monkeypatch.setattr(funding, '_history', {'BTC/USD': hist})
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: rate)

    def _boom(sym):
        raise AssertionError('archive must not be called when flag is OFF')
    monkeypatch.setattr(funding_archive, 'get_funding_series', _boom)

    assert funding.funding_tilt('BTC/USD') == 0.6


def test_tilt_flag_on_prefers_archive(funding_env, monkeypatch):
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', True)
    # Tight local history: taken alone the current rate would be an
    # EXTREME z (>3) -> 0.25x.
    tight_mu, tight_sd = 0.0, 1e-7
    hist = [tight_mu] * 27 + [tight_mu + tight_sd] * 3
    rate = tight_mu + 5 * tight_sd
    monkeypatch.setattr(funding, '_history', {'BTC/USD': hist})
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: rate)

    import statistics
    local_z = (rate - statistics.fmean(hist)) / statistics.pstdev(hist)
    assert local_z > funding.EXTREME_Z

    # Wide-dispersion archive series -> same rate reads as unremarkable.
    wide_vals = [0.0002 * ((-1) ** i) for i in range(40)]
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda s: _FakeSeries(wide_vals))

    assert funding.funding_tilt('BTC/USD') == 1.0


def test_tilt_flag_on_archive_failure_falls_back(funding_env, monkeypatch):
    monkeypatch.setattr(funding, 'FUNDING_Z_TIME_THINNING', True)
    mu, sd = 0.0001, 1e-6
    hist = [mu - sd, mu + sd] * 15    # 30 samples
    rate = mu + 2.5 * sd              # z = 2.5 -> crowded (0.6x)
    monkeypatch.setattr(funding, '_history', {'BTC/USD': hist})
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: rate)

    def _boom(sym):
        raise RuntimeError('archive down')
    monkeypatch.setattr(funding_archive, 'get_funding_series', _boom)

    assert funding.funding_tilt('BTC/USD') == 0.6

    # Fetch-failure path (get_funding_rate -> None) still never blocks/shrinks.
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: None)
    assert funding.funding_tilt('BTC/USD') == 1.0


# ---------------------------------------------------------------------------
# sentiment_history.py fixtures/helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sent_env(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, '_DB_PATH', str(tmp_path / 's.db'))
    monkeypatch.setattr(sh, '_db_local', threading.local())
    monkeypatch.setattr(sh, '_keyword_score', lambda h, s='': 0.5)
    monkeypatch.setattr(sh.time, 'sleep', lambda s: None)
    return sh


class _FakeFinnhubClient:
    """Records (ticker, _from, to) calls; returns canned articles."""

    def __init__(self, default_articles=None):
        self.calls = []
        self._default = default_articles or []

    def company_news(self, ticker, _from, to):
        self.calls.append((ticker, _from, to))
        return list(self._default)


def _article(headline, date_str, summary=''):
    d = datetime.date.fromisoformat(date_str)
    ts = int(datetime.datetime(d.year, d.month, d.day, 12, 0,
                               tzinfo=datetime.timezone.utc).timestamp())
    return {'headline': headline, 'summary': summary, 'url': '', 'datetime': ts}


def _seed_article(db, symbol, date_str, headline,
                  fetched_at='2026-01-01T00:00:00'):
    db.execute(
        """INSERT OR IGNORE INTO articles
           (symbol, date, headline, summary, url, keyword_score, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (symbol, date_str, headline, '', '', 0.1, fetched_at),
    )
    db.commit()
    sh._aggregate_daily(db, symbol, date_str)
    db.commit()


# ---------------------------------------------------------------------------
# sentiment_history.py: D27 dark-half refresh policy
# ---------------------------------------------------------------------------

def test_incremental_fetch_from_max_cached_date(sent_env, monkeypatch):
    db = sh._get_db()
    dates = [f'2026-05-{d:02d}' for d in range(1, 11)]
    for i, d in enumerate(dates):
        _seed_article(db, 'AAPL', d, f'Cached headline {i}')

    fake = _FakeFinnhubClient(default_articles=[_article('New headline',
                                                          '2026-08-10')])
    monkeypatch.setattr(sh, '_get_finnhub', lambda: fake)

    out = sh.fetch_stock_sentiment_history(['AAPL'], '2026-01-01', '2026-08-19')

    assert fake.calls, 'client should have been called for the incremental window'
    assert fake.calls[0][1] == '2026-05-10'   # inclusive from newest cached date

    n_new = db.execute(
        "SELECT COUNT(*) FROM articles WHERE symbol='AAPL' AND date='2026-08-10'"
    ).fetchone()[0]
    assert n_new == 1
    assert ('AAPL', '2026-08-10') in out


def test_incremental_preserves_prior_daily_values(sent_env, monkeypatch):
    db = sh._get_db()
    dates = [f'2026-05-{d:02d}' for d in range(1, 11)]
    for i, d in enumerate(dates):
        _seed_article(db, 'AAPL', d, f'Cached headline {i}')

    before = sorted(db.execute(
        "SELECT date, score, article_count, llm_count, score_type "
        "FROM daily_sentiment WHERE symbol='AAPL' AND date < '2026-05-10'"
    ).fetchall())
    before_overlap = db.execute(
        "SELECT COUNT(*) FROM articles WHERE symbol='AAPL' AND date='2026-05-10'"
    ).fetchone()[0]
    assert before_overlap == 1

    # Overlap-day re-send: identical headline to what's already cached for
    # the newest cached date — must dedupe via UNIQUE(symbol,date,headline).
    fake = _FakeFinnhubClient(default_articles=[_article('Cached headline 9',
                                                          '2026-05-10')])
    monkeypatch.setattr(sh, '_get_finnhub', lambda: fake)

    sh.fetch_stock_sentiment_history(['AAPL'], '2026-01-01', '2026-08-19')

    after = sorted(db.execute(
        "SELECT date, score, article_count, llm_count, score_type "
        "FROM daily_sentiment WHERE symbol='AAPL' AND date < '2026-05-10'"
    ).fetchall())
    assert after == before

    after_overlap = db.execute(
        "SELECT COUNT(*) FROM articles WHERE symbol='AAPL' AND date='2026-05-10'"
    ).fetchone()[0]
    assert after_overlap == 1


def test_cached_only_no_network_and_stable(sent_env, monkeypatch):
    db = sh._get_db()
    dates = ['2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05']
    for i, d in enumerate(dates):
        _seed_article(db, 'AAPL', d, f'AAPL headline {i}')

    def _boom():
        raise AssertionError('_get_finnhub must not be called under cached_only=True')
    monkeypatch.setattr(sh, '_get_finnhub', _boom)

    before = sh.fetch_stock_sentiment_history(
        ['AAPL'], '2026-05-01', '2026-05-05', cached_only=True)
    assert len(before) == 5

    # A real incremental fetch for a different ticker/date range runs elsewhere.
    _seed_article(db, 'MSFT', '2026-01-01', 'MSFT seed headline')
    fake = _FakeFinnhubClient(default_articles=[_article('MSFT new headline',
                                                          '2026-06-05')])
    monkeypatch.setattr(sh, '_get_finnhub', lambda: fake)
    sh.fetch_stock_sentiment_history(['MSFT'], '2026-06-01', '2026-06-10')

    monkeypatch.setattr(sh, '_get_finnhub', _boom)
    after = sh.fetch_stock_sentiment_history(
        ['AAPL'], '2026-05-01', '2026-05-05', cached_only=True)
    assert after == before


def test_up_to_date_ticker_skipped(sent_env, monkeypatch):
    db = sh._get_db()
    _seed_article(db, 'AAPL', '2026-05-10', 'Already current headline')

    fake = _FakeFinnhubClient()
    monkeypatch.setattr(sh, '_get_finnhub', lambda: fake)

    out = sh.fetch_stock_sentiment_history(['AAPL'], '2026-05-01', '2026-05-10')
    assert fake.calls == []
    assert ('AAPL', '2026-05-10') in out


def test_zero_cache_ticker_full_range(sent_env, monkeypatch):
    fake = _FakeFinnhubClient()
    monkeypatch.setattr(sh, '_get_finnhub', lambda: fake)

    sh.fetch_stock_sentiment_history(['NFLX'], '2026-05-01', '2026-05-05')

    assert fake.calls, 'unseen ticker should trigger a fetch'
    assert fake.calls[0][1] == '2026-05-01'   # full-range from start_date
