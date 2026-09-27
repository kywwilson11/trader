"""PIT repair of Daily_Sentiment (H audit, 2026-09-26).

Crypto: legacy fng_daily rows were local-tz dated (fng_daily[D] == value
published at (D+1) 00:00 UTC). A one-time, marker-keyed migration moves them
to fng_daily_legacy_localtz and refills fng_daily UTC-dated; the fetch asks
for the full history (limit=0).
REVIEW M2 (2026-09-26): the refill is fetched FIRST and move-aside + clear +
refill + marker commit in ONE transaction; a failed fetch leaves the legacy
table and returns its (leaky, non-zero) values with a WARNING, never an empty
table (the harvest maps missing days to Daily_Sentiment = 0.0).
Stock: the harvest key is ((t - 6h).date() - 1 day) so Chicago-dated article
buckets are strictly in the past for every bar, incl. 00:00 UTC.

Mac-safe: sqlite3 + pandas on a temp DB, no network (requests.get stubbed).
"""
import datetime as dt
import os
import sqlite3
import threading

import pandas as pd
import pytest

import sentiment_history as sh

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    path = str(tmp_path / 'sc.db')
    monkeypatch.setattr(sh, '_DB_PATH', path)
    monkeypatch.setattr(sh, '_db_local', threading.local())
    return path


def _seed_legacy(path, rows):
    sh._get_db()  # create schema at the patched path
    con = sqlite3.connect(path)
    con.executemany(
        "INSERT INTO fng_daily (date, value, score) VALUES (?, ?, ?)",
        [(d, v, sh._fng_value_to_score(v)) for d, v in rows])
    con.commit()
    con.close()


def _utc_midnight(date_str):
    d = dt.date.fromisoformat(date_str)
    return int(dt.datetime(d.year, d.month, d.day,
                           tzinfo=dt.timezone.utc).timestamp())


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def _stub_api(monkeypatch, pairs):
    """pairs = [(published 'YYYY-MM-DD' at 00:00 UTC, value)], newest first."""
    import requests
    calls = []
    payload = {'data': [{'value': str(v), 'timestamp': str(_utc_midnight(d)),
                         'value_classification': 'x'} for d, v in pairs]}

    def fake_get(url, timeout=None, **kw):
        calls.append(url)
        return _Resp(payload)

    monkeypatch.setattr(requests, 'get', fake_get)
    return calls


# ---------------------------------------------------------------- (a)
def test_migration_preserves_legacy_clears_and_marks_once(db_env):
    legacy = [('2025-01-01', 40), ('2025-01-02', 55), ('2025-01-03', 70)]
    _seed_legacy(db_env, legacy)
    db = sh._get_db()
    refill = [('2025-01-01', 30, sh._fng_value_to_score(30)),
              ('2025-01-02', 40, sh._fng_value_to_score(40)),
              ('2025-01-03', 55, sh._fng_value_to_score(55)),
              ('2025-01-04', 70, sh._fng_value_to_score(70))]

    moved = sh._migrate_fng_date_basis(db, refill)
    assert moved == 3
    got = db.execute(
        "SELECT date, value FROM fng_daily_legacy_localtz ORDER BY date"
    ).fetchall()
    assert got == legacy
    # cleared AND refilled in the same transaction: never observed empty
    assert db.execute("SELECT date, value FROM fng_daily ORDER BY date"
                      ).fetchall() == [(d, v) for d, v, _ in refill]
    assert db.execute("SELECT value FROM state WHERE key='fng_date_basis'"
                      ).fetchone()[0] == 'utc_publication'

    # Second call is a no-op: rows written after the migration survive and
    # the legacy table is untouched.
    db.execute("INSERT OR REPLACE INTO fng_daily (date, value, score) VALUES "
               "('2025-01-05', 41, -0.18)")
    db.commit()
    before = db.execute("SELECT * FROM fng_daily ORDER BY date").fetchall()
    assert sh._migrate_fng_date_basis(db) == 0
    assert sh._migrate_fng_date_basis(db, [('2025-01-05', 99, 0.98)]) == 0
    assert db.execute("SELECT * FROM fng_daily ORDER BY date"
                      ).fetchall() == before
    assert db.execute("SELECT COUNT(*) FROM fng_daily_legacy_localtz"
                      ).fetchone()[0] == 3


def test_migration_on_fresh_db_only_sets_marker(db_env):
    db = sh._get_db()
    assert sh._migrate_fng_date_basis(db) == 0
    assert db.execute("SELECT value FROM state WHERE key='fng_date_basis'"
                      ).fetchone()[0] == 'utc_publication'
    tables = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'fng_daily_legacy_localtz' not in tables


def test_migration_refuses_when_rows_cannot_be_preserved(db_env):
    _seed_legacy(db_env, [('2025-01-01', 40)])
    db = sh._get_db()
    # A pre-existing, conflicting legacy row for the same date: the copy
    # cannot preserve value 40 -> fng_daily must NOT be cleared.
    db.execute("CREATE TABLE fng_daily_legacy_localtz (date TEXT PRIMARY KEY, "
               "value INTEGER NOT NULL, score REAL NOT NULL)")
    db.execute("INSERT INTO fng_daily_legacy_localtz VALUES ('2025-01-01', 99, 0.98)")
    db.commit()
    with pytest.raises(RuntimeError):
        sh._migrate_fng_date_basis(db, [('2025-01-01', 33, -0.34),
                                        ('2025-01-02', 44, -0.12)])
    assert db.execute("SELECT date, value FROM fng_daily").fetchall() == [
        ('2025-01-01', 40)]
    assert db.execute("SELECT value FROM state WHERE key='fng_date_basis'"
                      ).fetchone() is None


# ------------------------------------------------------- (a2) REVIEW M2
def _state(db):
    return db.execute("SELECT value FROM state WHERE key='fng_date_basis'"
                      ).fetchone()


def _tables(db):
    return {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


@pytest.mark.parametrize('refill', [None, [],
                                    [('2025-01-01', 50, 0.0)]])  # 1/3 < 90%
def test_migration_refuses_to_clear_without_full_refill(db_env, refill):
    legacy = [('2025-01-01', 40), ('2025-01-02', 55), ('2025-01-03', 70)]
    _seed_legacy(db_env, legacy)
    db = sh._get_db()
    with pytest.raises(ValueError, match='refusing to clear'):
        sh._migrate_fng_date_basis(db, refill)
    assert db.execute("SELECT date, value FROM fng_daily ORDER BY date"
                      ).fetchall() == legacy
    assert _state(db) is None
    assert 'fng_daily_legacy_localtz' not in _tables(db)
    assert not db.in_transaction


def _stub_fail(monkeypatch, mode):
    import requests
    calls = []

    def fake_get(url, timeout=None, **kw):
        calls.append(url)
        if mode == 'raise':
            raise requests.exceptions.ConnectionError('network down')
        if mode == 'empty':
            return _Resp({'data': []})
        if mode == 'garbage':
            return _Resp({'metadata': {'error': 'rate limited'}})

        class _Bad:
            def json(self):
                raise ValueError('not json')
        return _Bad()

    monkeypatch.setattr(requests, 'get', fake_get)
    return calls


@pytest.mark.parametrize('mode', ['raise', 'empty', 'garbage', 'badjson'])
def test_failed_fetch_keeps_legacy_no_marker_and_warns(db_env, monkeypatch,
                                                       caplog, mode):
    days = pd.date_range('2025-03-01', '2025-03-10').strftime('%Y-%m-%d')
    legacy = [(d, 20 + i) for i, d in enumerate(days)]
    _seed_legacy(db_env, legacy)
    calls = _stub_fail(monkeypatch, mode)

    with caplog.at_level('WARNING', logger='sentiment_history'):
        out = sh.fetch_crypto_sentiment_history('2025-03-01', '2025-03-10')

    assert len(calls) == 1
    # legacy values served -- leaky, but NOT the all-zero feature
    assert out == {d: sh._fng_value_to_score(v) for d, v in legacy}
    assert all(v != 0 for v in out.values())
    db = sh._get_db()
    assert db.execute("SELECT date, value FROM fng_daily ORDER BY date"
                      ).fetchall() == legacy
    assert _state(db) is None
    assert 'fng_daily_legacy_localtz' not in _tables(db)
    msgs = [r.getMessage() for r in caplog.records
            if r.levelname == 'WARNING']
    assert any('NOT done' in m and 'LEGACY' in m and 'leak' in m
               and 'not zero' in m for m in msgs), msgs

    # the next harvest with a reachable API migrates atomically
    pub = [(d, 60) for d in reversed(list(days))]
    _stub_api(monkeypatch, pub)
    out2 = sh.fetch_crypto_sentiment_history('2025-03-01', '2025-03-10')
    assert out2 == {d: sh._fng_value_to_score(60) for d in days}
    assert _state(db)[0] == 'utc_publication'
    assert db.execute("SELECT COUNT(*) FROM fng_daily_legacy_localtz"
                      ).fetchone()[0] == len(legacy)


def test_migration_conflict_in_fetch_path_serves_legacy_not_raise(
        db_env, monkeypatch, caplog):
    # A RuntimeError from the migration must not reach the harvest (which
    # maps any raise to Daily_Sentiment = 0.0): rolled back, legacy served.
    _seed_legacy(db_env, [('2025-01-01', 40), ('2025-01-02', 60)])
    db = sh._get_db()
    db.execute("CREATE TABLE fng_daily_legacy_localtz (date TEXT PRIMARY KEY, "
               "value INTEGER NOT NULL, score REAL NOT NULL)")
    db.execute("INSERT INTO fng_daily_legacy_localtz VALUES ('2025-01-01', 99, 0.98)")
    db.commit()
    _stub_api(monkeypatch, [('2025-01-02', 70), ('2025-01-01', 30)])
    with caplog.at_level('WARNING', logger='sentiment_history'):
        out = sh.fetch_crypto_sentiment_history('2025-01-01', '2025-01-02')
    assert out == {'2025-01-01': sh._fng_value_to_score(40),
                   '2025-01-02': sh._fng_value_to_score(60)}
    assert _state(db) is None
    assert any('migration refused' in r.getMessage() for r in caplog.records)


def test_fresh_db_failed_fetch_sets_no_marker_and_warns_coverage(
        db_env, monkeypatch, caplog):
    _stub_fail(monkeypatch, 'raise')
    with caplog.at_level('WARNING', logger='sentiment_history'):
        out = sh.fetch_crypto_sentiment_history('2025-03-01', '2025-03-10')
    assert out == {}
    assert _state(sh._get_db()) is None
    assert any('coverage LOW' in r.getMessage() and '0/10' in r.getMessage()
               for r in caplog.records)


def test_coverage_warning_threshold_on_migrated_db(db_env, monkeypatch, caplog):
    db = sh._get_db()
    sh._migrate_fng_date_basis(db)                      # fresh: marker only
    days = list(pd.date_range('2025-03-01', '2025-03-20').strftime('%Y-%m-%d'))
    # 18/20 nonzero == 90% -> no warning; cache full -> no fetch
    rows = [(d, 50 if i < 2 else 70) for i, d in enumerate(days)]
    db.executemany("INSERT INTO fng_daily VALUES (?, ?, ?)",
                   [(d, v, sh._fng_value_to_score(v)) for d, v in rows])
    db.commit()
    calls = _stub_fail(monkeypatch, 'raise')
    with caplog.at_level('WARNING', logger='sentiment_history'):
        sh.fetch_crypto_sentiment_history(days[0], days[-1])
    assert calls == []
    assert not [r for r in caplog.records if 'coverage LOW' in r.getMessage()]
    # 17/20 nonzero -> WARNING
    db.execute("UPDATE fng_daily SET value=50, score=0.0 WHERE date=?",
               (days[5],))
    db.commit()
    with caplog.at_level('WARNING', logger='sentiment_history'):
        sh.fetch_crypto_sentiment_history(days[0], days[-1])
    assert any('coverage LOW' in r.getMessage() and '17/20' in r.getMessage()
               for r in caplog.records)


def test_already_migrated_db_is_a_noop(db_env, monkeypatch):
    # The live-DB shape: marker set, fully cached -> no network, no writes.
    db = sh._get_db()
    sh._migrate_fng_date_basis(db)
    days = list(pd.date_range('2025-03-01', '2025-03-05').strftime('%Y-%m-%d'))
    db.executemany("INSERT INTO fng_daily VALUES (?, ?, ?)",
                   [(d, 70, sh._fng_value_to_score(70)) for d in days])
    db.commit()
    snap = (db.execute("SELECT * FROM fng_daily ORDER BY date").fetchall(),
            db.execute("SELECT * FROM state ORDER BY key").fetchall())
    calls = _stub_fail(monkeypatch, 'raise')
    assert sh._migrate_fng_date_basis(db, [('2025-03-01', 1, -0.98)]) == 0
    out = sh.fetch_crypto_sentiment_history(days[0], days[-1])
    assert calls == []
    assert out == {d: sh._fng_value_to_score(70) for d in days}
    assert (db.execute("SELECT * FROM fng_daily ORDER BY date").fetchall(),
            db.execute("SELECT * FROM state ORDER BY key").fetchall()) == snap


_LIVE_DB = os.path.join(_ROOT, 'sentiment_cache.db')


@pytest.mark.skipif(not os.path.exists(_LIVE_DB),
                    reason='no local sentiment_cache.db (dev Mac / CI)')
def test_live_db_copy_idempotent(tmp_path, monkeypatch):
    # Copy the real cache with the sqlite backup API (consistent snapshot,
    # WAL-safe, read-only on the source) and prove the migration is a no-op.
    src = sqlite3.connect(f'file:{_LIVE_DB}?mode=ro', uri=True, timeout=60)
    dst_path = str(tmp_path / 'live_copy.db')
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    src.close()
    dst.close()
    monkeypatch.setattr(sh, '_DB_PATH', dst_path)
    monkeypatch.setattr(sh, '_db_local', threading.local())
    db = sh._get_db()
    if not sh._fng_is_utc(db):
        pytest.skip('local sentiment_cache.db not yet UTC-migrated')
    snap = (db.execute("SELECT * FROM fng_daily ORDER BY date").fetchall(),
            db.execute("SELECT * FROM state WHERE key LIKE 'fng%' ORDER BY key"
                       ).fetchall())
    assert len(snap[0]) > 0
    calls = _stub_fail(monkeypatch, 'raise')
    assert sh._migrate_fng_date_basis(db) == 0
    assert sh._migrate_fng_date_basis(db, [('2020-01-01', 1, -0.98)]) == 0
    lo, hi = snap[0][0][0], snap[0][-1][0]
    out = sh.fetch_crypto_sentiment_history(lo, hi)
    assert out == {d: sc for d, _, sc in snap[0]}
    assert (db.execute("SELECT * FROM fng_daily ORDER BY date").fetchall(),
            db.execute("SELECT * FROM state WHERE key LIKE 'fng%' ORDER BY key"
                       ).fetchall()) == snap
    # full-range cache -> no network; a gap would fetch once and fail soft
    assert len(calls) <= 1


# ---------------------------------------------------------------- (b)
def test_refill_dates_each_value_at_utc_publication_day(db_env, monkeypatch):
    # Legacy (local-tz) cache: fng_daily[D] == value published at D+1.
    pub = [('2025-03-05', 15), ('2025-03-04', 14), ('2025-03-03', 13),
           ('2025-03-02', 12), ('2025-03-01', 11)]           # newest first
    leaked = [('2025-03-01', 12), ('2025-03-02', 13), ('2025-03-03', 14),
              ('2025-03-04', 15)]
    _seed_legacy(db_env, leaked)
    calls = _stub_api(monkeypatch, pub)

    out = sh.fetch_crypto_sentiment_history('2025-03-01', '2025-03-05')

    assert len(calls) == 1 and 'limit=0' in calls[0]
    db = sh._get_db()
    rows = dict(db.execute("SELECT date, value FROM fng_daily").fetchall())
    # fng_daily[D] == the value published at D 00:00 UTC, for every D
    assert rows == {d: v for d, v in pub}
    assert out == {d: sh._fng_value_to_score(v) for d, v in pub}
    # legacy rows kept verbatim
    assert dict(db.execute(
        "SELECT date, value FROM fng_daily_legacy_localtz").fetchall()
    ) == dict(leaked)


def test_fixture_discriminates_legacy_local_tz_bucketing():
    # The legacy bug: date.fromtimestamp(ts) in America/Chicago puts the
    # value published at D 00:00 UTC on D-1. Confirm the fixture timestamps
    # would be mis-dated that way (so test (b) can tell the two rules apart)
    # and that the shipped rule -- the UTC date -- is independent of the
    # host zone.
    from zoneinfo import ZoneInfo
    for d in ('2025-01-15', '2025-07-15'):            # CST and CDT
        ts = _utc_midnight(d)
        legacy = dt.datetime.fromtimestamp(ts, tz=ZoneInfo('America/Chicago')).date()
        assert str(legacy) == str(dt.date.fromisoformat(d) - dt.timedelta(days=1))
        assert dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date().isoformat() == d
    src = open(os.path.join(_ROOT, 'sentiment_history.py')).read()
    body = src.split('def fetch_crypto_sentiment_history')[1].split('\ndef ')[0]
    assert 'tz=datetime.timezone.utc).date().isoformat()' in body
    assert 'date.fromtimestamp(ts)' not in body


# ---------------------------------------------------------------- (c)
@pytest.mark.parametrize('ts, expected', [
    ('2026-02-10 00:00', '2026-02-08'),   # prior-US-evening ext-hours bar
    ('2026-02-10 05:59', '2026-02-08'),
    ('2026-02-10 06:00', '2026-02-09'),
    ('2026-02-10 14:30', '2026-02-09'),   # RTH
    ('2026-02-10 23:00', '2026-02-09'),
])
def test_stock_lookup_key(ts, expected):
    aware = pd.DatetimeIndex([pd.Timestamp(ts, tz='UTC')])
    naive = pd.DatetimeIndex([pd.Timestamp(ts)])
    chicago = aware.tz_convert('America/Chicago')
    assert sh.stock_sentiment_lookup_dates(aware) == [expected]
    assert sh.stock_sentiment_lookup_dates(naive) == [expected]
    assert sh.stock_sentiment_lookup_dates(chicago) == [expected]


def test_stock_key_bucket_is_strictly_past_for_chicago_dated_articles():
    # A Chicago-dated bucket X ends at (X+1) 00:00 Chicago = (X+1) 06:00 UTC
    # (CST) / 05:00 UTC (CDT). It must end at or before every bar that reads it.
    idx = pd.date_range('2026-01-01', '2026-08-31 23:00', freq='h', tz='UTC')
    keys = sh.stock_sentiment_lookup_dates(idx)
    for t, k in zip(idx, keys):
        bucket_end = (pd.Timestamp(k, tz='America/Chicago')
                      + pd.Timedelta(days=1)).tz_convert('UTC')
        assert bucket_end <= t, (t, k)


def test_harvest_uses_shared_key_helper():
    src = open(os.path.join(_ROOT, 'scripts', 'harvest_stock_data.py')).read()
    assert 'stock_sentiment_lookup_dates(final_df.index)' in src
    # the old unshifted D-1 key is gone
    assert 'zip(final_df[\'Ticker\'], final_df.index.date)' not in src


# ---------------------------------------------------------------- (d)
def test_source_pin_full_history_fetch():
    src = open(os.path.join(_ROOT, 'sentiment_history.py')).read()
    assert "fng/?limit=0&format=json" in src
    assert 'limit={total_days}' not in src
    body = src.split('def fetch_crypto_sentiment_history')[1].split('\ndef ')[0]
    # REVIEW M2: the migration runs only AFTER the fetch, with its refill
    get_at = body.index('requests.get(')
    mig_at = body.index('_migrate_fng_date_basis(db, fetched)')
    assert get_at < mig_at
    assert body.count('_migrate_fng_date_basis(') == 1
    assert '_fng_is_utc(db)' in body.split('if start_date is None')[0]
