"""Group tests for the sentiment/events review group (S1-S12 changes).

Covers: sentiment.py S1 phrase-prefilter parity + S6/S7 doc/import fixes,
sentiment_history.py S3/S4/S5 socket-hygiene/doc fixes, events_calendar.py
S8/S9/S10 corrupt-cache hardening, edgar_events.py S11 memoized cache.
"""

import collections
import datetime
import json
import os
import time

import pytest

import sentiment as s
import sentiment_history as sh
import events_calendar as ec
import edgar_events as ee


# ---------------------------------------------------------------------------
# T1 — _score_text goldens (pins S1 parity; captured from CURRENT implementation)
# ---------------------------------------------------------------------------

def test_t1_score_text_goldens():
    assert s._score_text("Bitcoin surges to all time high") == pytest.approx(
        0.8593867635096133, abs=1e-9)
    assert s._score_text("Crypto crash wipes out billions") == pytest.approx(
        -0.8114879110711591, abs=1e-9)
    # pins the KNOWN duplicate double-fire; update deliberately when the
    # deferred weight fix ships
    assert s._score_text("analyst price target slashed for acme") == pytest.approx(
        -0.9810124534848998, abs=1e-9)
    # phrase negation
    assert s._score_text("Fed will not beat expectations this quarter") == pytest.approx(
        -0.8654727419789566, abs=1e-9)
    assert s._score_text("TSLA earnings beat expectations, guidance raised") == pytest.approx(
        0.9975733365264466, abs=1e-9)
    # word negation
    assert s._score_text("no losses reported as growth continues") == pytest.approx(
        0.8841076674344449, abs=1e-9)
    assert s._score_text("Company files for bankruptcy amid fraud investigation") == pytest.approx(
        -0.8925392095450269, abs=1e-9)
    assert s._score_text("quiet session with little movement in markets") == pytest.approx(
        0.0, abs=1e-9)
    # sqrt-scale path
    long_bearish = ("shares of the company fell after the earnings miss and "
                    "guidance cut while analysts warned of slowing growth "
                    "and rising risks " * 20)
    assert s._score_text(long_bearish) == pytest.approx(
        -0.9992590333122554, abs=1e-9)


# ---------------------------------------------------------------------------
# T2 — Phrase-table invariants (the prefilter's correctness preconditions)
# ---------------------------------------------------------------------------

def test_t2_phrase_table_invariants():
    all_triples = s._POS_PHRASE_RES + s._NEG_PHRASE_RES
    for phrase, pat, weight in all_triples:
        assert phrase == phrase.lower()
        probe = f"xx {phrase} yy"
        assert phrase in probe
        assert pat.search(probe)
        # boundary at punctuation
        punct_probe = "(" + phrase + ")!"
        assert pat.search(punct_probe)

    neg_phrases = [p for p, _pat, _w in s._NEG_PHRASE_RES]
    pos_phrases = [p for p, _pat, _w in s._POS_PHRASE_RES]
    neg_dups = {p: c for p, c in collections.Counter(neg_phrases).items() if c > 1}
    # known, deferred to owner — new dupes must not appear
    assert neg_dups == {'price target slashed': 2}
    pos_dups = {p: c for p, c in collections.Counter(pos_phrases).items() if c > 1}
    assert pos_dups == {}

    assert s._POSITIVE & s._NEGATIVE == frozenset()


# ---------------------------------------------------------------------------
# T3 — KW-combine divergence pins (S5 anti-unification fence)
# ---------------------------------------------------------------------------

def test_t3_kw_combine_divergence():
    art = {'headline': 'TSLA earnings beat expectations, guidance raised',
           'summary': 'Analysts raised price targets amid strong growth'}
    assert s._kw_score_article(art) == pytest.approx(0.9555596857338787, abs=1e-9)

    h_only = {'headline': 'TSLA earnings beat expectations, guidance raised',
              'summary': ''}
    assert s._kw_score_article(h_only) == pytest.approx(0.598544001915868, abs=1e-9)
    assert sh._keyword_score(
        'TSLA earnings beat expectations, guidance raised', ''
    ) == pytest.approx(0.9975733365264466, abs=1e-9)


# ---------------------------------------------------------------------------
# T4 — events_calendar corrupt cache fails CLOSED, not crash (pins S8)
# ---------------------------------------------------------------------------

def test_t4_corrupt_cache_fails_closed(tmp_path, monkeypatch):
    cache_file = tmp_path / 'earnings_calendar.json'
    cache_file.write_text('[1, 2, 3]')
    monkeypatch.setattr(ec, '_CACHE_FILE', str(cache_file))
    monkeypatch.setattr(ec, '_mem', None)
    monkeypatch.setattr(ec, '_last_attempt', time.monotonic())  # throttles -> no network

    assert ec.calendar_available() is False
    assert ec.blocks_overnight_hold('AAPL') is False


# ---------------------------------------------------------------------------
# T5 — non-string fetched_at (pins S9)
# ---------------------------------------------------------------------------

def test_t5_non_string_fetched_at(monkeypatch):
    monkeypatch.setattr(ec, '_mem', {
        'fetched_at': 12345,
        'by_symbol': {'AAPL': [{'date': '2026-01-01'}]},
    })
    monkeypatch.setattr(ec, '_last_attempt', time.monotonic())

    assert ec.refresh_if_stale() is True  # no TypeError


# ---------------------------------------------------------------------------
# T6 — events_calendar window boundaries (documents CURRENT calendar-day
# semantics; the deferred trading-day fix must consciously update these)
# ---------------------------------------------------------------------------

def test_t6_window_boundaries(monkeypatch):
    today = datetime.date.today()

    def _mk(offset_days):
        d = today + datetime.timedelta(days=offset_days)
        return d.isoformat()

    # --- blocks_overnight_hold: d=today -> True; +2 -> True; +3 -> False; -1 -> False
    for offset, expected in [(0, True), (2, True), (3, False), (-1, False)]:
        monkeypatch.setattr(ec, 'refresh_if_stale', lambda: True)
        monkeypatch.setattr(ec, '_mem', {
            'by_symbol': {'AAPL': [{'date': _mk(offset)}]}
        })
        assert ec.blocks_overnight_hold('AAPL') is expected, f"offset={offset}"

    # --- earnings_within_days(days=1): d=today -> True; +1 -> True; +2 -> False
    for offset, expected in [(0, True), (1, True), (2, False)]:
        monkeypatch.setattr(ec, 'refresh_if_stale', lambda: True)
        monkeypatch.setattr(ec, '_mem', {
            'by_symbol': {'AAPL': [{'date': _mk(offset)}]}
        })
        assert ec.earnings_within_days('AAPL', days=1) is expected, f"offset={offset}"

    # --- reported_recently: (today-1,'amc') -> True; (today,'bmo') -> True;
    #     (today,'amc') -> False; (today-2,'amc') -> False
    cases = [(-1, 'amc', True), (0, 'bmo', True), (0, 'amc', False), (-2, 'amc', False)]
    for offset, hour, expected in cases:
        monkeypatch.setattr(ec, 'refresh_if_stale', lambda: True)
        monkeypatch.setattr(ec, '_mem', {
            'by_symbol': {'AAPL': [{'date': _mk(offset), 'hour': hour}]}
        })
        assert ec.reported_recently('AAPL') is expected, f"offset={offset} hour={hour}"


# ---------------------------------------------------------------------------
# T7 — edgar memo semantics (pins S11)
# ---------------------------------------------------------------------------

def test_t7_edgar_memo_semantics(tmp_path, monkeypatch):
    cache_file = tmp_path / 'edgar_cache.json'
    monkeypatch.setattr(ee, '_CACHE_FILE', cache_file)
    monkeypatch.setattr(ee, '_cache_memo', None)

    # (a) copy isolation
    cache_file.write_text(json.dumps(
        {'AAPL': {'date': 'x', 'blocked': False, 'reason': None}}))
    monkeypatch.setattr(ee, '_cache_memo', None)
    c1 = ee._load_cache()
    c1['MUT'] = 1
    assert 'MUT' not in ee._load_cache()

    # (b) save -> load roundtrip
    ee._save_cache({'MSFT': {'date': 'y', 'blocked': True, 'reason': 'r'}})
    loaded = ee._load_cache()
    assert loaded['MSFT']['blocked'] is True
    siblings = [p for p in os.listdir(tmp_path) if '.tmp' in p]
    assert siblings == []

    # (c) external replace visible
    different = {'GOOG': {'date': 'z', 'blocked': False, 'reason': None},
                 'EXTRA': {'date': 'w', 'blocked': True, 'reason': 'longer json body'}}
    sibling_tmp = tmp_path / 'edgar_cache.json.external.tmp'
    sibling_tmp.write_text(json.dumps(different))
    os.replace(str(sibling_tmp), str(cache_file))
    os.utime(str(cache_file), None)  # bump mtime
    reloaded = ee._load_cache()
    assert reloaded.get('GOOG', {}).get('blocked') is False
    assert 'MSFT' not in reloaded

    # (d) corrupt non-dict
    cache_file.write_text('[1]')
    monkeypatch.setattr(ee, '_cache_memo', None)
    assert ee._load_cache() == {}


# ---------------------------------------------------------------------------
# T8 — _gemini_batch_http closes the response (pins S3)
# ---------------------------------------------------------------------------

def test_t8_gemini_batch_http_closes_response(monkeypatch):
    class _FakeResp:
        def __init__(self):
            self.exited = False

        def read(self):
            return b'{"ok": 1}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.exited = True
            return False

    resp = _FakeResp()

    def _fake_urlopen(req, timeout=60):
        return resp

    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', _fake_urlopen)

    result = sh._gemini_batch_http('GET', 'models/x', None, 'k')
    assert result == {'ok': 1}
    assert resp.exited is True


# ---------------------------------------------------------------------------
# T9 — doc-truth fences
# ---------------------------------------------------------------------------

def test_t9_doc_truth_fences():
    assert 'only fetch new/uncached' not in sh.fetch_stock_sentiment_history.__doc__
    assert 'ZERO cached articles' in sh.fetch_stock_sentiment_history.__doc__
    assert 'keyword gap-fill' in s._llm_score_batch.__doc__
