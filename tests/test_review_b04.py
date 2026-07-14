"""2026-07 review batch b04: sentiment / novelty / macro_calendar fixes.

novelty and macro_calendar import cleanly on the dev Mac. sentiment needs
dotenv at module scope, so it is imported here under a stubbed `dotenv`
that is removed from sys.modules afterwards — every other test module
(test_parse_scores, test_sentiment_headlines, ...) keeps collecting exactly
as on the baseline run.

Covers:
  - sentiment: _fetch_full_texts hard 15s deadline (no `with` executor —
    shutdown(wait=False, cancel_futures=True)); _build_score_prompt survives
    None-valued headline/summary; try_llm_upgrade only tags/counts articles
    the model actually scored; get_recent_headlines TTL cache + error
    logging + docstring; _try_llm_retry re-checks cache freshness after the
    slow LLM call; get_cnn_fear_greed returns None when the payload lacks
    'fear_and_greed'; dead code removed (_LLM_CHUNK_SIZE, unused
    _trigger_429_cooldown / call_gemini imports); _article_cache bounded;
    _kw_score_article is the single KW-fallback implementation;
    sentiment_gate docstring matches the real multipliers.
  - novelty: thread-safe read-modify-write (_LOCK); store rewritten only
    when it changed + one flush per filter_novel batch (Jetson flash wear);
    full-store sweep drops rotated-out symbols; _load resets a non-dict
    store; _save failures are logged (rate-limited) and clean up the tmp.
  - macro_calendar: Feb-2026 CPI row is the rescheduled 02-13;
    calendar_exhausted gets macro_standdown's naive-datetime UTC guard;
    plus the module's first behavioral net (windows, boundaries, weekdays).
"""
import collections
import concurrent.futures
import datetime
import importlib
import json
import logging
import os
import re
import sys
import threading
import time
import types
import zoneinfo
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import macro_calendar
import novelty


def _import_sentiment():
    """Import sentiment with a stubbed dotenv, leaving sys.modules as found."""
    if 'sentiment' in sys.modules:  # full-stack CI: reuse the real import
        return sys.modules['sentiment']
    stubbed = False
    if 'dotenv' not in sys.modules:
        try:
            import dotenv  # noqa: F401
        except ImportError:  # dev-Mac path
            fake = types.ModuleType('dotenv')
            fake.load_dotenv = lambda *a, **k: None
            sys.modules['dotenv'] = fake
            stubbed = True
    try:
        mod = importlib.import_module('sentiment')
    finally:
        if stubbed:
            # Leave sys.modules exactly as the baseline run sees it: other
            # test modules must keep collecting/failing identically.
            sys.modules.pop('dotenv', None)
            sys.modules.pop('sentiment', None)
    return mod


sentiment = _import_sentiment()
SENT_SRC = (REPO / 'sentiment.py').read_text()

ET = zoneinfo.ZoneInfo('America/New_York')
UTC = datetime.timezone.utc


def _et(y, mo, d, hh, mi):
    return datetime.datetime(y, mo, d, hh, mi, tzinfo=ET)


# ===========================================================================
# macro_calendar
# ===========================================================================

def test_fomc_window_boundaries():
    blocked, reason = macro_calendar.macro_standdown(_et(2026, 7, 29, 14, 0))
    assert blocked and 'FOMC' in reason
    assert macro_calendar.macro_standdown(_et(2026, 7, 29, 12, 0))[0]   # start inclusive
    assert macro_calendar.macro_standdown(_et(2026, 7, 29, 15, 29))[0]
    assert not macro_calendar.macro_standdown(_et(2026, 7, 29, 11, 59))[0]
    assert not macro_calendar.macro_standdown(_et(2026, 7, 29, 15, 30))[0]  # half-open end


def test_cpi_window_boundaries():
    blocked, reason = macro_calendar.macro_standdown(_et(2026, 7, 14, 8, 30))
    assert blocked and 'CPI' in reason
    assert macro_calendar.macro_standdown(_et(2026, 7, 14, 6, 30))[0]
    assert not macro_calendar.macro_standdown(_et(2026, 7, 14, 6, 29))[0]
    assert not macro_calendar.macro_standdown(_et(2026, 7, 14, 9, 30))[0]   # half-open end


def test_non_event_days_do_not_stand_down():
    for day in (28, 30):   # FOMC statement day is Jul 29
        assert not macro_calendar.macro_standdown(_et(2026, 7, day, 14, 0))[0]
    for day in (13, 15):   # CPI print day is Jul 14
        assert not macro_calendar.macro_standdown(_et(2026, 7, day, 8, 30))[0]


def test_aware_utc_input_equals_et_input():
    # 18:00 UTC == 14:00 EDT on 2026-07-29
    utc_in = datetime.datetime(2026, 7, 29, 18, 0, tzinfo=UTC)
    assert macro_calendar.macro_standdown(utc_in) == \
        macro_calendar.macro_standdown(_et(2026, 7, 29, 14, 0))


def test_feb_cpi_row_is_the_rescheduled_date():
    # Jan-2026-data CPI slipped Feb 11 -> Fri Feb 13 (early-Feb-2026
    # shutdown); the original 02-11 row stood the bot down on a non-event
    # day and left it free to trade into the real print.
    assert (2026, 2, 13) in macro_calendar.CPI_RELEASE_DAYS
    assert (2026, 2, 11) not in macro_calendar.CPI_RELEASE_DAYS


def test_calendar_exhausted_flips_after_last_event():
    assert not macro_calendar.calendar_exhausted(_et(2026, 12, 10, 12, 0))
    assert macro_calendar.calendar_exhausted(_et(2026, 12, 11, 0, 1))


def test_naive_datetimes_treated_as_utc_in_both_functions():
    # macro_standdown already normalized naive -> UTC; calendar_exhausted
    # used to read naive input as SYSTEM LOCAL time (machine-dependent).
    naive = datetime.datetime(2026, 7, 14, 12, 30)          # 08:30 ET as UTC
    aware = naive.replace(tzinfo=UTC)
    assert macro_calendar.macro_standdown(naive) == \
        macro_calendar.macro_standdown(aware)
    assert macro_calendar.macro_standdown(naive)[0]
    naive2 = datetime.datetime(2026, 12, 11, 2, 0)          # Dec 10 21:00 ET
    aware2 = naive2.replace(tzinfo=UTC)
    assert macro_calendar.calendar_exhausted(naive2) == \
        macro_calendar.calendar_exhausted(aware2)
    assert not macro_calendar.calendar_exhausted(naive2)


def test_event_tables_weekday_sanity():
    for y, m, d in macro_calendar.FOMC_STATEMENT_DAYS:
        assert datetime.date(y, m, d).weekday() == 2       # statements: Wednesdays
    for y, m, d in macro_calendar.CPI_RELEASE_DAYS:
        assert datetime.date(y, m, d).weekday() < 5        # prints: weekdays


# ===========================================================================
# novelty
# ===========================================================================

@pytest.fixture()
def nov_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(novelty, '_STORE_FILE', tmp_path / 'store.json')
    monkeypatch.setattr(novelty, '_store', None)
    monkeypatch.setattr(novelty, '_dirty', False)
    monkeypatch.setattr(novelty, '_last_save_warn', 0.0)
    return tmp_path


def _count_replaces(monkeypatch):
    calls = []
    real = os.replace

    def counting(src, dst):
        calls.append(dst)
        return real(src, dst)

    monkeypatch.setattr(novelty.os, 'replace', counting)
    return calls


def test_concurrent_scoring_smoke_and_lock_exists(nov_sandbox):
    assert isinstance(novelty._LOCK, type(threading.Lock()))
    errors = []

    def worker(sym, seed):
        try:
            for i in range(120):
                novelty.headline_novelty(
                    sym, f'{seed} headline number {i} moves markets today')
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=('BTC/USD', 'bitcoin')),
               threading.Thread(target=worker, args=('TSLA', 'tesla'))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    data = json.loads((nov_sandbox / 'store.json').read_text())
    assert set(data) == {'BTC/USD', 'TSLA'}


def test_exact_repeat_does_not_rewrite_store(nov_sandbox, monkeypatch):
    calls = _count_replaces(monkeypatch)
    h = 'Bitcoin surges past 100k on ETF inflows'
    assert novelty.headline_novelty('BTC/USD', h) == 1.0
    assert len(calls) == 1                       # first sighting persists
    assert novelty.headline_novelty('BTC/USD', h) < 0.01
    assert len(calls) == 1                       # steady-state reprint: no I/O


def test_filter_novel_flushes_once_per_batch(nov_sandbox, monkeypatch):
    calls = _count_replaces(monkeypatch)
    batch = [
        'Bitcoin surges past 100k on ETF inflows',
        'SEC delays decision on Solana staking products',
        'Miner outflows hit yearly low as hashrate stabilizes',
        'Fed cuts rates boosting risk assets broadly today',
    ]
    out = novelty.filter_novel('BTC/USD', batch)
    assert out == batch                          # all fresh on first sight
    assert len(calls) == 1                       # ONE flush for the batch
    calls.clear()
    novelty.filter_novel('BTC/USD', batch[:2])   # pure reprints
    assert calls == []                           # nothing changed -> no write


def test_direct_call_still_persists_for_reload(nov_sandbox, monkeypatch):
    # Guards the flush=True default that tests/test_novelty.py relies on.
    novelty.headline_novelty('BTC/USD', 'Bitcoin surges past 100k on ETF inflows')
    monkeypatch.setattr(novelty, '_store', None)  # force reload from disk
    n = novelty.headline_novelty('BTC/USD',
                                 'Bitcoin surges past 100k on ETF inflows')
    assert n < 0.01


def test_sweep_drops_rotated_out_symbols(nov_sandbox):
    old_ts = time.time() - 8 * 86400
    (nov_sandbox / 'store.json').write_text(
        json.dumps({'DEAD/SYM': [[old_ts, [1, 2, 3]]]}))
    n = novelty.headline_novelty('BTC/USD',
                                 'Bitcoin surges past 100k on ETF inflows')
    assert n == 1.0
    assert 'DEAD/SYM' not in novelty._store       # swept at load
    on_disk = json.loads((nov_sandbox / 'store.json').read_text())
    assert 'DEAD/SYM' not in on_disk and 'BTC/USD' in on_disk  # swept at save


def test_load_resets_non_dict_store(nov_sandbox, caplog):
    (nov_sandbox / 'store.json').write_text('[1, 2, 3]')   # valid JSON, wrong shape
    with caplog.at_level(logging.WARNING, logger='novelty'):
        n = novelty.headline_novelty(
            'BTC/USD', 'Solana staking approval expands institutional adoption')
    assert n == 1.0                               # no AttributeError, store rebuilt
    assert isinstance(novelty._store, dict)


def test_save_failure_logged_rate_limited_and_no_raise(nov_sandbox, monkeypatch, caplog):
    def boom(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(novelty.os, 'replace', boom)
    with caplog.at_level(logging.WARNING, logger='novelty'):
        n1 = novelty.headline_novelty('BTC/USD',
                                      'Bitcoin surges past 100k on ETF inflows')
        n2 = novelty.headline_novelty('BTC/USD',
                                      'SEC delays decision on Solana staking products')
    assert n1 == 1.0 and n2 > 0.9                 # scoring unaffected by dead disk
    warns = [r for r in caplog.records if 'save failed' in r.getMessage()]
    assert len(warns) == 1                        # rate-limited (one per hour)
    assert not (nov_sandbox / 'store.json.tmp').exists()  # tmp cleaned up


# ===========================================================================
# sentiment — behavioral (module imported under stubbed dotenv)
# ===========================================================================

@pytest.fixture()
def sent_cache(monkeypatch):
    cache = {}
    monkeypatch.setattr(sentiment, '_cache', cache)
    return cache


def test_build_score_prompt_survives_none_fields():
    # Finnhub sends keys with None VALUES; .get('headline', '') doesn't help.
    articles = [{'headline': None, 'summary': None},
                {'headline': 'BTC surges', 'summary': None},
                {'headline': None, 'summary': 'plunge feared'}]
    prompt, n = sentiment._build_score_prompt(articles, [None, None, None])
    assert n == 3
    assert '2. BTC surges' in prompt


def test_fetch_full_texts_hard_deadline_and_cancel(monkeypatch):
    release = threading.Event()
    started = []

    def slow_fetch(url):
        started.append(url)
        release.wait(10)   # self-healing: workers unblock even on failure
        return None

    def instant_timeout(futs, timeout=None):
        raise concurrent.futures.TimeoutError()

    monkeypatch.setattr(sentiment, '_fetch_article_text', slow_fetch)
    monkeypatch.setattr(concurrent.futures, 'as_completed', instant_timeout)
    articles = [{'url': f'http://x/{i}'} for i in range(25)]
    t0 = time.monotonic()
    out = sentiment._fetch_full_texts(articles)
    elapsed = time.monotonic() - t0
    release.set()
    assert out == [None] * 25          # un-fetched bodies stay None
    assert elapsed < 5                 # old `with` blocked on every fetch
    assert len(started) <= 10          # cancel_futures dropped the queued 15


def test_fetch_full_texts_source_uses_hard_shutdown():
    assert 'with ThreadPoolExecutor' not in SENT_SRC
    assert 'cancel_futures=True' in SENT_SRC


def test_try_llm_upgrade_keeps_gaps_retryable(monkeypatch):
    articles = [{'headline': 'a', '_scored_by_model': 'KW'},
                {'headline': 'b', '_scored_by_model': 'KW'},
                {'headline': 'c', '_scored_by_model': 'gemini-2.5-pro'}]
    fake_llm = types.ModuleType('llm_client')
    fake_llm.get_budget = lambda model: (5, 100)
    monkeypatch.setitem(sys.modules, 'llm_client', fake_llm)
    monkeypatch.setattr(sentiment, '_fetch_full_texts',
                        lambda arts: [None] * len(arts))
    monkeypatch.setattr(sentiment, '_llm_score_chunk',
                        lambda arts, texts, model=None: [0.4, None])
    out = sentiment.try_llm_upgrade(articles)
    assert out == [0.4, None, None]
    assert articles[0]['_scored_by_model'] == 'gemini-2.5-pro'
    # The None gap keeps its old tag, so a later upgrade pass can retry it
    # (and the GUI model column doesn't lie).
    assert articles[1]['_scored_by_model'] == 'KW'
    assert articles[2]['_scored_by_model'] == 'gemini-2.5-pro'


def test_get_recent_headlines_caches_and_survives_none_headline(monkeypatch, sent_cache):
    calls = []

    class FakeClient:
        def general_news(self, cat, min_id=0):
            calls.append(cat)
            return [{'headline': ' BTC rips ', 'summary': 'btc'},
                    {'headline': None, 'summary': 'btc note'},   # None value
                    {'headline': 'ETH story', 'summary': ''}]

    monkeypatch.setattr(sentiment, '_get_finnhub', lambda: FakeClient())
    out1 = sentiment.get_recent_headlines('BTC/USD', 'crypto')
    out2 = sentiment.get_recent_headlines('BTC/USD', 'crypto')
    assert out1 == ['BTC rips'] and out2 == out1
    assert len(calls) == 1             # second call served from _cache
    assert any(k.startswith('headlines_') for k in sent_cache)


def test_get_recent_headlines_logs_swallowed_errors(monkeypatch, sent_cache, capsys):
    class Boom:
        def general_news(self, *a, **k):
            raise RuntimeError('finnhub down')

    monkeypatch.setattr(sentiment, '_get_finnhub', lambda: Boom())
    assert sentiment.get_recent_headlines('BTC/USD') == []
    assert 'finnhub down' in capsys.readouterr().out
    assert 'Cached alongside get_news_sentiment' not in SENT_SRC  # docstring fixed


def test_llm_retry_drops_result_when_cache_refreshed_mid_scoring(monkeypatch, sent_cache):
    q = collections.deque(maxlen=50)
    monkeypatch.setattr(sentiment, '_llm_retry_queue', q)
    queued_at = time.time() - 10
    q.append(('news_BTC/USD', [{'headline': 'x'}], queued_at))
    fresher = {'sentiment_score': 0.9, 'article_count': 3,
               'positive_ratio': 1.0, 'negative_ratio': 0.0}

    def batch_with_concurrent_refresh(articles):
        # combined-bot mode: the other loop thread refreshes the same key
        # while this thread sits in the slow LLM call
        sent_cache['news_BTC/USD'] = (time.time(), fresher)
        return [0.1]

    monkeypatch.setattr(sentiment, '_llm_score_batch', batch_with_concurrent_refresh)
    sentiment._try_llm_retry()
    assert sent_cache['news_BTC/USD'][1] is fresher   # retry result dropped
    assert not q                                       # consumed, not re-queued


def test_cnn_fear_greed_missing_key_returns_none(monkeypatch, sent_cache):
    resp = SimpleNamespace(json=lambda: {'error': 'nope'})
    monkeypatch.setattr(sentiment, 'requests',
                        SimpleNamespace(get=lambda *a, **k: resp))
    assert sentiment.get_cnn_fear_greed() is None      # not a zero-filled dict
    assert '__cnn_fng__' not in sent_cache             # errors are not cached


def test_article_cache_bounded(monkeypatch):
    cache = {}
    monkeypatch.setattr(sentiment, '_article_cache', cache)
    now = time.time()
    stale_ts = now - sentiment._ARTICLE_CACHE_TTL - 60
    for i in range(sentiment._ARTICLE_CACHE_MAX):
        sentiment._article_cache_put(f'http://old/{i}', stale_ts, None)
    assert len(cache) == sentiment._ARTICLE_CACHE_MAX
    sentiment._article_cache_put('http://fresh', now, 'text')
    assert cache == {'http://fresh': (now, 'text')}    # expired all purged


def test_article_cache_writes_go_through_put():
    # def + 4 call sites; the only direct dict write lives inside the helper
    assert SENT_SRC.count('_article_cache_put(') == 5
    assert SENT_SRC.count('_article_cache[url] =') == 1


def test_kw_score_article_headline_summary_weights():
    a = {'headline': 'Bitcoin surges on ETF inflows and strong momentum',
         'summary': 'Institutional buying accelerates today.'}
    h = sentiment._score_text(a['headline'])
    s = sentiment._score_text(a['summary'])
    assert sentiment._kw_score_article(a) == pytest.approx(h * 0.6 + s * 0.4)
    # Neither field scorable -> None (callers decide skip vs 0.0)
    assert sentiment._kw_score_article({'headline': None, 'summary': ''}) is None
    assert sentiment._kw_score_article({'headline': 'short', 'summary': None}) is None


def test_kw_score_article_full_text_weights(monkeypatch):
    body = 'profits surge as growth beats expectations everywhere'
    monkeypatch.setattr(sentiment, '_fetch_article_text', lambda url: body)
    a = {'headline': 'Company rallies strongly today', 'summary': '',
         'url': 'http://x'}
    h = sentiment._score_text(a['headline'])
    f = sentiment._score_text(body)
    got = sentiment._kw_score_article(a, fetch_full=True)
    assert got == pytest.approx(h * 0.25 + 0.0 * 0.25 + f * 0.50)


def test_kw_fallback_single_source_of_truth():
    assert SENT_SRC.count('def _kw_score_article') == 1
    assert SENT_SRC.count('_kw_score_article(') == 4   # def + 3 call sites
    assert SENT_SRC.count('* 0.6 + s * 0.4') == 1      # weights only in helper


def test_sentiment_dead_code_removed():
    assert '_LLM_CHUNK_SIZE' not in SENT_SRC
    assert '_trigger_429_cooldown' not in SENT_SRC
    assert not re.search(r'^\s*from llm_client import .*call_gemini',
                         SENT_SRC, re.M)


def test_sentiment_gate_multipliers_match_rewritten_docs(monkeypatch):
    doc = sentiment.sentiment_gate.__doc__
    for token in ('0.35', '1.35', '[0.15, 1.5]', '80-89'):
        assert token in doc
    assert '0.5  = reduce' not in doc                  # stale table gone

    def gate_with(fng_val, sym_score, mkt_score):
        monkeypatch.setattr(sentiment, 'get_fear_greed',
                            lambda: {'value': fng_val, 'label': 'x'})
        monkeypatch.setattr(
            sentiment, 'get_news_sentiment',
            lambda s, a='crypto': {'sentiment_score': sym_score,
                                   'article_count': 4,
                                   'positive_ratio': 0.5,
                                   'negative_ratio': 0.5})
        monkeypatch.setattr(
            sentiment, 'get_market_sentiment',
            lambda: {'sentiment_score': mkt_score, 'article_count': 5,
                     'positive_ratio': 0.5, 'negative_ratio': 0.5})
        return sentiment.sentiment_gate('BTC/USD', 'crypto')[0]

    assert gate_with(85, 0.45, 0.0) == pytest.approx(0.85 * 1.35)
    assert gate_with(85, -0.9, -0.9) == 0.15           # clamped at the floor
    assert gate_with(50, 0.45, 0.45) == pytest.approx(1.35 * 1.1)
