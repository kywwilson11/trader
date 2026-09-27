"""G4 hunt fixes (2026-09-26): LLM + sentiment layer robustness.

G4-01  sentiment_history.fetch_stock_sentiment_history commits per 30-day
       window -> no SQLite write transaction spans a Finnhub call / sleep.
G4-02  fetch_crypto_sentiment_history keeps a fetched FnG day in the result
       even when its cache INSERT fails.
G4-03  present-but-None Finnhub fields (headline/summary/url) no longer crash
       sentiment.get_news_sentiment's crypto filter or skip a ticker in
       fetch_stock_sentiment_history.
G4-04  novelty._save / llm_config.save_llm_config use a per-writer tmp name
       (trade_memory idiom) -> an interleaved second writer cannot tear the file.
G4-08  no back-off sleep after the final 429 attempt.
G4-09  llm_client._rate_limit_ok's deque prune/check/append is locked.
G4-10  load_llm_config returns deep copies of _DEFAULTS.

Mac-safe: sqlite3 on temp DBs, faked Finnhub / requests / json.dump; no network.
Templates: the hunter's repros (scratchpad hunt/g4/*.py).
"""
import collections
import copy
import datetime as dt
import json
import sqlite3
import sys
import threading
import types

import pytest

import sentiment_history as sh

D0 = dt.date(2026, 1, 1)


def _ts(d, hour=15):
    return int(dt.datetime(d.year, d.month, d.day, hour,
                           tzinfo=dt.timezone.utc).timestamp())


@pytest.fixture
def sh_db(tmp_path, monkeypatch):
    path = str(tmp_path / 'sent.db')
    monkeypatch.setattr(sh, '_DB_PATH', path)
    monkeypatch.setattr(sh, '_db_local', threading.local())
    sh._get_db()
    yield path
    conn = getattr(sh._db_local, 'conn', None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _no_sleep(monkeypatch, record=None):
    def fake_sleep(s):
        if record is not None:
            record.append(s)
    monkeypatch.setattr(sh.time, 'sleep', fake_sleep)


# --------------------------------------------------------------------- G4-01
def test_g4_01_no_write_txn_across_network_call(sh_db, monkeypatch):
    _no_sleep(monkeypatch)
    in_txn = []
    other_writer = []

    class Fake:
        calls = 0

        def company_news(self, t, _from, to):
            Fake.calls += 1
            if Fake.calls == 2:
                in_txn.append(sh._get_db().in_transaction)
                # A concurrent writer (backfill worker / set_live_mode) must
                # get the lock immediately while the fetch is "on the wire".
                other = sqlite3.connect(sh_db, timeout=0.2)
                try:
                    other.execute("INSERT OR REPLACE INTO state (key, value) "
                                  "VALUES ('live_mode', '1')")
                    other.commit()
                    other_writer.append('ok')
                except sqlite3.OperationalError as e:
                    other_writer.append(str(e))
                finally:
                    other.close()
            f = dt.date.fromisoformat(_from)
            return [{'headline': f'{t} beats estimates {f}', 'summary': 'good',
                     'url': '', 'datetime': _ts(f)}]

    monkeypatch.setattr(sh, '_get_finnhub', lambda: Fake())
    end = (D0 + dt.timedelta(days=40)).isoformat()
    res = sh.fetch_stock_sentiment_history(['AAA'], D0.isoformat(), end)

    assert Fake.calls == 2
    assert in_txn == [False]
    assert other_writer == ['ok']
    # Output unchanged: both windows' articles stored and aggregated.
    c = sqlite3.connect(sh_db)
    art_dates = sorted(r[0] for r in c.execute(
        "SELECT date FROM articles WHERE symbol='AAA'"))
    daily = dict(c.execute(
        "SELECT date, score FROM daily_sentiment WHERE symbol='AAA'"))
    c.close()
    assert art_dates == ['2026-01-01', '2026-01-31']
    assert sorted(daily) == art_dates
    assert {k[1]: v for k, v in res.items() if k[0] == 'AAA'} == daily


# --------------------------------------------------------------------- G4-02
class _LockedOnFngInsert:
    def __init__(self, c, bad_day):
        self.c, self.bad_day = c, bad_day

    def execute(self, sql, *a):
        if (sql.lstrip().startswith('INSERT OR IGNORE INTO fng_daily')
                and self.bad_day in a[0]):
            raise sqlite3.OperationalError('database is locked')
        return self.c.execute(sql, *a)

    def __getattr__(self, n):
        return getattr(self.c, n)


def test_g4_02_failed_fng_insert_keeps_fetched_value(sh_db, monkeypatch):
    import requests
    db = sh._get_db()
    db.execute("INSERT INTO state VALUES ('fng_date_basis', 'utc_publication')")
    db.commit()
    days = [dt.date(2026, 3, d) for d in (1, 2, 3)]
    payload = {'data': [{'timestamp': str(_ts(d, 0)), 'value': str(v)}
                        for d, v in zip(days, (20, 75, 40))]}
    monkeypatch.setattr(requests, 'get', lambda *a, **k: types.SimpleNamespace(
        json=lambda: payload))
    monkeypatch.setattr(sh._db_local, 'conn', _LockedOnFngInsert(db, '2026-03-02'))

    res = sh.fetch_crypto_sentiment_history('2026-03-01', '2026-03-03')

    assert res == {'2026-03-01': sh._fng_value_to_score(20),
                   '2026-03-02': sh._fng_value_to_score(75),
                   '2026-03-03': sh._fng_value_to_score(40)}
    cached = {r[0] for r in db.execute("SELECT date FROM fng_daily")}
    assert cached == {'2026-03-01', '2026-03-03'}   # only the cache write missed


# --------------------------------------------------------------------- G4-03
def _stock_fetch(sh_db_path, monkeypatch, summary_for_window2):
    class FakeStock:
        def company_news(self, t, _from, to):
            f = dt.date.fromisoformat(_from)
            if t == 'AAA' and f > D0:
                return [{'headline': 'AAA misses revenue estimates badly',
                         'summary': summary_for_window2, 'url': None,
                         'datetime': _ts(f)}]
            return [{'headline': f'{t} beats estimates {k}', 'summary': 'good',
                     'url': '', 'datetime': _ts(f + dt.timedelta(days=k))}
                    for k in (0, 4, 19)]

    monkeypatch.setattr(sh, '_get_finnhub', lambda: FakeStock())
    end = (D0 + dt.timedelta(days=45)).isoformat()
    res = sh.fetch_stock_sentiment_history(['AAA', 'BBB'], D0.isoformat(), end)
    c = sqlite3.connect(sh_db_path)
    daily = sorted(c.execute(
        "SELECT symbol, date, score FROM daily_sentiment ORDER BY symbol, date"))
    arts = sorted(c.execute(
        "SELECT symbol, date, headline, summary, url FROM articles"))
    c.close()
    return res, daily, arts


def test_g4_03_none_fields_do_not_skip_ticker(tmp_path, monkeypatch):
    _no_sleep(monkeypatch)
    out = {}
    for label, summary in (('none', None), ('empty', '')):
        monkeypatch.setattr(sh, '_DB_PATH', str(tmp_path / f'{label}.db'))
        monkeypatch.setattr(sh, '_db_local', threading.local())
        out[label] = _stock_fetch(sh._DB_PATH, monkeypatch, summary)
        sh._db_local.conn.close()

    res_none, daily_none, arts_none = out['none']
    res_empty, daily_empty, arts_empty = out['empty']
    aaa_dates = [d for s, d, _ in daily_none if s == 'AAA']
    # AAA not skipped: every article day aggregated, incl. the None-summary one
    assert aaa_dates == ['2026-01-01', '2026-01-05', '2026-01-20', '2026-01-31']
    # identical to the same feed with '' in place of None
    assert daily_none == daily_empty
    assert res_none == res_empty
    assert arts_none == arts_empty


def test_g4_03_crypto_filter_none_summary(monkeypatch):
    import sentiment as s
    arts = [
        {'headline': 'BTC jumps as ETF inflows accelerate', 'summary': 'x', 'url': ''},
        {'headline': 'Traders pile into BTC options', 'summary': None, 'url': ''},
        {'headline': 'Why BTC miners are selling', 'summary': None, 'url': ''},
        {'headline': 'Altcoins slide across the board', 'summary': None, 'url': ''},
        {'headline': None, 'summary': 'btc dominance rises', 'url': ''},
    ]

    class FakeCrypto:
        def general_news(self, cat, min_id=0):
            return [dict(a) for a in arts]

    seen = []

    def fake_score(relevant):
        seen.append(list(relevant))
        return {'score': 0.1, 'n': len(relevant)}, True

    monkeypatch.setattr(s, '_get_finnhub', lambda: FakeCrypto())
    monkeypatch.setattr(s, '_score_articles', fake_score)
    monkeypatch.setattr(s, '_try_llm_retry', lambda: None)
    monkeypatch.setattr(s, '_cache', {})
    monkeypatch.setattr(s, '_llm_retry_queue', collections.deque(maxlen=50))

    result = s.get_news_sentiment('BTC/USD', 'crypto')

    assert result == {'score': 0.1, 'n': 4}
    assert [a['headline'] for a in seen[0]] == [
        'BTC jumps as ETF inflows accelerate', 'Traders pile into BTC options',
        'Why BTC miners are selling', None]
    assert 'news_BTC/USD' in s._cache


# --------------------------------------------------------------------- G4-08
def test_g4_08_no_sleep_after_last_429(sh_db, monkeypatch):
    sleeps, calls = [], []
    _no_sleep(monkeypatch, sleeps)

    class Always429:
        def company_news(self, t, _from, to):
            calls.append((_from, to))
            raise Exception('FinnhubAPIException(status_code: 429)')

    monkeypatch.setattr(sh, '_get_finnhub', lambda: Always429())
    res = sh.fetch_stock_sentiment_history(['AAA'], '2026-01-01', '2026-01-10')
    assert len(calls) == 3                 # same requests as before
    assert sleeps == [62, 124]             # no 248 s sleep after the last one
    assert not any(k[0] == 'AAA' for k in res)


# --------------------------------------------------------------------- G4-04
def _json_shim(pause_thread_name, paused, release):
    """json module stand-in whose dump() writes half, then (on the named
    thread only) waits for `release` before writing the rest."""
    real = json

    def dump(obj, f, **kw):
        text = real.dumps(obj, **kw)
        if threading.current_thread().name == pause_thread_name:
            f.write(text[:len(text) // 2])
            f.flush()
            paused.set()
            assert release.wait(10)
            f.write(text[len(text) // 2:])
        else:
            f.write(text)

    shim = types.SimpleNamespace(**{k: getattr(real, k) for k in dir(real)
                                    if not k.startswith('__')})
    shim.dump = dump
    return shim


def test_g4_04_novelty_interleaved_writers_no_torn_store(tmp_path, monkeypatch):
    import novelty
    store = tmp_path / 'novelty_store.json'
    monkeypatch.setattr(novelty, '_STORE_FILE', store)
    paused, release = threading.Event(), threading.Event()
    monkeypatch.setattr(novelty, 'json', _json_shim('writer-A', paused, release))
    now = __import__('time').time()
    a_store = {'BTC/USD': [[now, [i, i + 1, i + 2]] for i in range(40)]}
    b_store = {'AAPL': [[now, [7, 8, 9]]]}
    errs = []

    monkeypatch.setattr(novelty, '_store', a_store)
    monkeypatch.setattr(novelty, '_dirty', True)

    def writer_a():
        try:
            novelty._save()
        except Exception as e:  # pragma: no cover - surfaced below
            errs.append(e)

    ta = threading.Thread(target=writer_a, name='writer-A')
    ta.start()
    assert paused.wait(10)
    # Writer B (the other book's process in split mode) flushes meanwhile.
    novelty._store = b_store
    novelty._dirty = True
    novelty._save()
    release.set()
    ta.join(10)

    assert not errs
    loaded = json.loads(store.read_text())      # not torn
    assert loaded in (a_store, b_store)
    assert list(tmp_path.glob('*.tmp')) == []


def test_g4_04_llm_config_interleaved_writers_no_torn_file(tmp_path, monkeypatch):
    import llm_config as lc
    path = tmp_path / 'llm_config.json'
    monkeypatch.setattr(lc, 'LLM_CONFIG_FILE', path)
    paused, release = threading.Event(), threading.Event()
    monkeypatch.setattr(lc, 'json', _json_shim('writer-A', paused, release))
    cfg_a = {'enabled': True, 'provider': 'gemini',
             'models': {'gemini': {'api_key': 'A' * 200, 'model': 'x'}}}
    cfg_b = {'enabled': False, 'provider': 'claude'}

    ta = threading.Thread(target=lc.save_llm_config, args=(cfg_a,),
                          name='writer-A')
    ta.start()
    assert paused.wait(10)
    lc.save_llm_config(cfg_b)
    release.set()
    ta.join(10)

    loaded = json.loads(path.read_text())       # not torn
    assert loaded in (cfg_a, cfg_b)
    assert list(tmp_path.glob('*.tmp')) == []


def test_g4_04_llm_config_failed_save_leaves_no_tmp(tmp_path, monkeypatch):
    import llm_config as lc
    monkeypatch.setattr(lc, 'LLM_CONFIG_FILE', tmp_path / 'llm_config.json')

    def boom(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(lc.os, 'replace', boom)
    lc.save_llm_config({'enabled': True})       # never raises
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------- G4-10
def test_g4_10_load_llm_config_returns_copies(tmp_path, monkeypatch):
    import llm_config as lc
    before = copy.deepcopy(lc._DEFAULTS)
    path = tmp_path / 'llm_config.json'
    monkeypatch.setattr(lc, 'LLM_CONFIG_FILE', path)
    path.write_text(json.dumps({'enabled': True, 'models': {
        'gemini': {'api_key': 'G', 'model': 'gemini-2.5-flash'}}}))

    cfg = lc.load_llm_config()
    # exactly what gui._on_settings_changed does per provider row:
    for p in list(cfg['models']):
        cfg.setdefault('models', {}).setdefault(p, {})['api_key'] = 'TYPED'
    for k, v in cfg.items():                    # and any other nested mutation
        if isinstance(v, dict):
            v['__mutated__'] = 1
        elif isinstance(v, list):
            v.append('__mutated__')

    assert lc._DEFAULTS == before
    path.write_text(json.dumps({'enabled': True, 'models': {}}))
    cfg2 = lc.load_llm_config()
    for p, row in cfg2['models'].items():
        assert row.get('api_key') != 'TYPED', p
    for k, v in cfg2.items():
        if isinstance(v, dict):
            assert '__mutated__' not in v, k
        elif isinstance(v, list):
            assert '__mutated__' not in v, k


# --------------------------------------------------------------------- G4-09
class _PausingDeque(collections.deque):
    """deque whose [0] read pauses the designated thread mid-prune."""

    def __init__(self, *a, pause_thread=None, paused=None, release=None):
        super().__init__(*a)
        self._pt, self._paused, self._release = pause_thread, paused, release

    def __getitem__(self, i):
        v = super().__getitem__(i)
        if threading.current_thread().name == self._pt and not self._paused.is_set():
            self._paused.set()
            self._release.wait(5)
        return v


def test_g4_09_rate_limit_prune_is_atomic(monkeypatch):
    import llm_client as lc
    paused, release = threading.Event(), threading.Event()
    clock = [1000.0]
    monkeypatch.setattr(lc, 'time', types.SimpleNamespace(
        time=lambda: clock[0]))
    monkeypatch.setattr(lc, '_get_rate_limit_rpm', lambda: 30)
    dq = _PausingDeque([1.0], pause_thread='A', paused=paused, release=release)
    monkeypatch.setattr(lc, '_call_timestamps', dq)
    errs, results = [], {}

    def run(name):
        try:
            results[name] = lc._rate_limit_ok()
        except Exception as e:
            errs.append(f'{name}: {type(e).__name__}: {e}')

    ta = threading.Thread(target=run, args=('A',), name='A')
    ta.start()
    assert paused.wait(5)                  # A is inside the prune check
    tb = threading.Thread(target=run, args=('B',), name='B')
    tb.start()
    tb.join(0.3)
    assert tb.is_alive()                   # B is blocked on the lock
    release.set()
    ta.join(5)
    tb.join(5)
    assert errs == []                      # unlocked code: A -> IndexError
    assert results == {'A': True, 'B': True}
    assert list(dq) == [1000.0, 1000.0]


def test_g4_09_rate_limit_stress_no_indexerror(monkeypatch):
    """Scaled-down hunter stress: 6 threads, tiny GIL switch interval, clock
    with periodic full-window expiry; never raises, never over-admits."""
    import llm_client as lc
    rpm = 30
    clock = [0.0]
    clock_lock = threading.Lock()

    def tick(i):
        with clock_lock:
            clock[0] += (61.0 if i % 40 == 0 else 0.01)

    monkeypatch.setattr(lc, 'time', types.SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(lc, '_get_rate_limit_rpm', lambda: rpm)
    monkeypatch.setattr(lc, '_call_timestamps', collections.deque())
    errors = collections.Counter()
    over = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        def worker():
            for i in range(4000):
                tick(i)
                try:
                    lc._rate_limit_ok()
                except Exception as e:
                    errors[f'{type(e).__name__}: {e}'] += 1
                if len(lc._call_timestamps) > rpm:
                    over.append(len(lc._call_timestamps))

        ts = [threading.Thread(target=worker) for _ in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    finally:
        sys.setswitchinterval(old)
    assert dict(errors) == {}
    assert over == []
