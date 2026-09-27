"""G4 hunt fixes, second batch (2026-09-27): llm_analyst / sentiment / volatility.

G4-06  llm_analyst.build_compact_evidence and _build_symbol_profiles coerce raw
       yfinance fundamentals through fundamentals._safe_float, so 'Infinity' /
       'N/A' / '' / None never drop the whole evidence block / profile.
G4-07  a non-finite (NaN/inf/-inf) score is treated as MISSING, never clamped
       to the most bullish value:
         * llm_analyst._parse_response: s -> 0.5 (same as a missing "s"),
           p_up -> None, conviction Infinity -> None (no OverflowError out of
           analyze_trades);
         * sentiment._parse_scores: -> None gap (keyword fallback), not matched;
         * sentiment_history.poll_and_ingest_batch: entry skipped (unscored).
G4-09  sentiment._try_llm_retry survives the empty-deque check-then-popleft race.
IMPL_exec F3 sibling: volatility._har_rrv_save writes a per-writer tmp
       ({path}.{pid}.{thread_ident}.tmp) and removes it on failure.

Mac-safe: numpy/pandas + sqlite3 on temp DBs; yfinance / LLM transports faked;
no network.
"""
import collections
import datetime as dt
import json
import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

import llm_analyst
import sentiment
import sentiment_history as sh
import volatility

BAD_VALUES = [float('nan'), float('inf'), float('-inf'), None, 'abc',
              'Infinity', 'N/A', '']


# ===========================================================================
# G4-06 — build_compact_evidence
# ===========================================================================

_SNAP = {'Close': 101.5, 'Return_4h': 0.5, 'RSI': 55.0}


def _evidence(fund):
    # asset_type='crypto' skips the earnings-calendar disk read.
    return llm_analyst.build_compact_evidence('AAA', dict(_SNAP), fund,
                                              asset_type='crypto')


def test_compact_evidence_numeric_output_unchanged():
    fund = {'pe_ratio': 21.3, 'pb_ratio': 3.25, 'market_cap': 2.5e12,
            'revenue_growth': 0.123, 'beta': 1.15, 'sector': 'Tech',
            'week52_high': 150.0, 'week52_low': 50.0}
    out = _evidence(fund)
    assert out.splitlines()[-1] == (
        "- P/E 21.3 | P/B 3.2 | MktCap $2.5T | RevGrowth +12.3% | "
        "Beta 1.15 | Sector Tech | 52w-pos 0.52")


@pytest.mark.parametrize('bad', BAD_VALUES)
def test_compact_evidence_survives_bad_fundamentals(bad):
    fund = {k: bad for k in ('pe_ratio', 'pb_ratio', 'market_cap',
                             'revenue_growth', 'beta', 'week52_high',
                             'week52_low')}
    fund['sector'] = 'Tech'
    out = _evidence(fund)          # used to raise ValueError / TypeError
    assert out is not None
    assert 'Close $101.50' in out  # technical line kept
    val_line = out.splitlines()[-1]
    assert val_line == '- Sector Tech'  # every bad numeric field omitted


def test_compact_evidence_mixed_good_and_bad():
    fund = {'pe_ratio': 'Infinity', 'pb_ratio': 2.0, 'market_cap': 'N/A',
            'revenue_growth': '', 'beta': None, 'week52_high': 'Infinity',
            'week52_low': 10.0}
    out = _evidence(fund)
    assert out.splitlines()[-1] == '- P/B 2.0'


# ===========================================================================
# G4-06 — _build_symbol_profiles (yfinance faked)
# ===========================================================================

def _fake_yf(monkeypatch, info):
    idx = pd.date_range('2026-01-01', periods=60, freq='D')
    close = pd.Series(np.linspace(100.0, 130.0, 60), index=idx)
    hist = pd.DataFrame({'Close': close, 'High': close * 1.01,
                         'Low': close * 0.99,
                         'Volume': np.full(60, 1e6)}, index=idx)

    class _Tk:
        def __init__(self):
            self.info = info
            self.news = []

        def history(self, period='1y'):
            return hist

    class _Tickers:
        def __init__(self, syms):
            self.tickers = {s: _Tk() for s in syms}

    monkeypatch.setitem(sys.modules, 'yfinance',
                        types.SimpleNamespace(Tickers=_Tickers))


def test_symbol_profile_numeric_fundamentals_formatted(monkeypatch):
    _fake_yf(monkeypatch, {'marketCap': 3.1e9, 'trailingPE': 21.3,
                           'revenueGrowth': 0.05, 'sector': 'Tech'})
    prof = llm_analyst._build_symbol_profiles(['AAA'])
    assert 'AAA' in prof
    assert 'Fundamentals: MktCap: $3.1B | P/E: 21.30 | RevGrowth: +5.0% | ' \
           'Sector: Tech' in prof['AAA']


@pytest.mark.parametrize('bad', BAD_VALUES)
def test_symbol_profile_survives_bad_fundamentals(monkeypatch, bad):
    info = {k: bad for k in ('marketCap', 'trailingPE', 'forwardPE',
                             'priceToBook', 'revenueGrowth', 'earningsGrowth',
                             'beta', 'shortRatio', 'twoHundredDayAverage',
                             'targetMeanPrice')}
    info.update({'numberOfAnalystOpinions': 7, 'sector': 'Tech'})
    _fake_yf(monkeypatch, info)
    prof = llm_analyst._build_symbol_profiles(['AAA'])
    assert 'AAA' in prof                      # profile no longer dropped
    assert 'Price: Current: $130.00' in prof['AAA']
    assert 'Fundamentals: Sector: Tech' in prof['AAA']


# ===========================================================================
# G4-07 — llm_analyst._parse_response
# ===========================================================================

# Raw JSON tokens: json.loads accepts bare NaN / Infinity / -Infinity.
BAD_TOKENS = ['NaN', 'Infinity', '-Infinity', 'null', '"abc"', '"nan"',
              '"inf"']


@pytest.mark.parametrize('tok', BAD_TOKENS)
def test_parse_response_nonfinite_score_is_neutral(tok):
    resp = '{"AAA": {"s": %s, "r": "x"}, "BBB": {"s": 0.8}}' % tok
    out = llm_analyst._parse_response(resp, ['AAA', 'BBB'])
    assert out['AAA']['s'] == 0.5            # NOT 1.0 (was: NaN -> 1.0)
    assert out['AAA']['m'] == 0.75
    assert out['BBB']['s'] == 0.8


def test_parse_response_missing_score_matches_nonfinite():
    miss = llm_analyst._parse_response('{"AAA": {"r": "x"}}', ['AAA'])
    nan = llm_analyst._parse_response('{"AAA": {"s": NaN, "r": "x"}}', ['AAA'])
    assert miss == nan


def test_parse_response_finite_clamps_unchanged():
    out = llm_analyst._parse_response(
        '{"A": {"s": 1.7}, "B": {"s": -0.2}, "C": {"s": "0.3"}}',
        ['A', 'B', 'C'])
    assert [out[k]['s'] for k in 'ABC'] == [1.0, 0.0, 0.3]


@pytest.mark.parametrize('tok', BAD_TOKENS)
def test_parse_response_extended_nonfinite_fields_missing(tok):
    resp = ('{"AAA": {"s": 0.6, "p_up": %s, "conviction": %s, '
            '"abstain": false}}' % (tok, tok))
    out = llm_analyst._parse_response(resp, ['AAA'], extended=True)
    assert out['AAA']['s'] == 0.6
    assert out['AAA']['p_up'] is None
    assert out['AAA']['conviction'] is None


def test_parse_response_extended_finite_unchanged():
    resp = '{"AAA": {"s": 0.6, "p_up": 1.4, "conviction": 9}}'
    out = llm_analyst._parse_response(resp, ['AAA'], extended=True)
    assert out['AAA']['p_up'] == 1.0 and out['AAA']['conviction'] == 5


@pytest.fixture
def _sandboxed_cost_ledger(monkeypatch, tmp_path):
    """Test hygiene (INTEL W7, 2026-09-27): analyze_trades -> llm_analyst
    get_routing_info -> llm_client._maybe_reset_quota -> _cost_file_lock
    opens llm_client._COST_FILE + '.lock' for writing (llm_client.py:286)
    and a date rollover also rewrites _COST_FILE via its '.tmp' sibling
    (_save_shared_cost). Every ledger path derives from the single constant
    llm_client._COST_FILE, so pointing it into tmp_path keeps the repo-root
    llm_cost.json{,.lock,.tmp} untouched. The in-memory ledger globals are
    set through monkeypatch too, so they are restored after the test and no
    tmp-derived state leaks into later tests."""
    import llm_client
    monkeypatch.setattr(llm_client, '_COST_FILE',
                        str(tmp_path / 'llm_cost.json'))
    monkeypatch.setattr(llm_client, '_cost_reset_date', '')
    monkeypatch.setattr(llm_client, '_daily_cost', 0.0)
    return tmp_path


def test_analyze_trades_conviction_infinity_does_not_raise(
        monkeypatch, _sandboxed_cost_ledger):
    cfg = {'enabled': True, 'advisor_v2_enabled': True,
           'analyst_dedup_ttl_sec': 0}
    monkeypatch.setattr(llm_analyst, 'load_llm_config', lambda: cfg)
    monkeypatch.setattr(llm_analyst, 'get_recommended_model',
                        lambda role: 'fake-model')
    monkeypatch.setattr(llm_analyst, '_compute_event_lines',
                        lambda sym, at: ([], []))
    monkeypatch.setattr(
        llm_analyst, 'call_model',
        lambda *a, **k: '{"AAA": {"s": 0.7, "p_up": NaN, '
                        '"conviction": Infinity, "abstain": false, '
                        '"key_risks": [], "event_flags": []}}')
    monkeypatch.setattr(llm_analyst, 'call_llm', lambda *a, **k: None)
    out = llm_analyst.analyze_trades(
        [{'symbol': 'AAA', 'pred_return': 0.01, 'fundamentals_text': '',
          'news_headlines': []}], 'stock', persist=False)
    assert out['AAA']['s'] == 0.7
    assert out['AAA']['conviction'] is None and out['AAA']['p_up'] is None


# ===========================================================================
# G4-07 — sentiment._parse_scores
# ===========================================================================

def test_sentiment_parse_scores_nonfinite_are_gaps():
    # 10 articles: 5 finite, 5 bad -> matched 5 (>= 50%) -> list returned.
    raw = ('{"1": NaN, "2": 0.1, "3": Infinity, "4": -Infinity, "5": null, '
           '"6": "abc", "7": -0.4, "8": 2.5, "9": "0.2", "10": -3}')
    out = sentiment._parse_scores(raw, 10)
    assert out == [None, 0.1, None, None, None, None, -0.4, 1.0, 0.2, -1.0]


def test_sentiment_parse_scores_all_nan_fails_chunk():
    # Non-finite values are NOT counted as matched -> whole chunk fails
    # (keyword fallback), instead of becoming five +1.0 scores.
    raw = '{"1": NaN, "2": NaN, "3": NaN, "4": NaN}'
    assert sentiment._parse_scores(raw, 4) is None


# ===========================================================================
# G4-09 sibling — sentiment._try_llm_retry empty-deque race
# ===========================================================================

class _RacedDeque(collections.deque):
    """Truthy at the emptiness check, empty by the time popleft runs —
    exactly the window another loop thread's popleft opens."""

    def __bool__(self):
        return True

    def __len__(self):
        return 1

    def popleft(self):
        raise IndexError('pop from an empty deque')


def test_try_llm_retry_empty_deque_race_is_noop(monkeypatch):
    monkeypatch.setattr(sentiment, '_llm_retry_queue', _RacedDeque())
    assert sentiment._try_llm_retry() is None     # used to raise IndexError


def test_try_llm_retry_empty_queue_is_noop(monkeypatch):
    monkeypatch.setattr(sentiment, '_llm_retry_queue',
                        collections.deque(maxlen=50))
    assert sentiment._try_llm_retry() is None


def test_try_llm_retry_stale_item_still_drained(monkeypatch):
    q = collections.deque(maxlen=50)
    q.append(('k', [], 0.0))                      # queued at epoch -> stale
    monkeypatch.setattr(sentiment, '_llm_retry_queue', q)
    sentiment._try_llm_retry()
    assert len(q) == 0


# ===========================================================================
# G4-07 — sentiment_history.poll_and_ingest_batch
# ===========================================================================

@pytest.fixture
def sh_db(tmp_path, monkeypatch):
    path = str(tmp_path / 'sent.db')
    monkeypatch.setattr(sh, '_DB_PATH', path)
    monkeypatch.setattr(sh, '_db_local', threading.local())
    db = sh._get_db()
    yield db
    try:
        db.close()
    except Exception:
        pass


def test_batch_ingest_skips_nonfinite_scores(sh_db, monkeypatch):
    db = sh_db
    now = dt.datetime.now().isoformat()
    ids = []
    for n in range(6):
        cur = db.execute(
            "INSERT INTO articles (symbol, date, headline, keyword_score, "
            "fetched_at) VALUES ('AAA', '2026-01-05', ?, 0.0, ?)",
            (f'headline {n}', now))
        ids.append([cur.lastrowid, 'AAA', '2026-01-05'])
    db.commit()
    sh._batch_state_set(db, {'name': 'batches/x', 'submitted_at': now,
                             'id_map': [ids]})
    import llm_config
    monkeypatch.setattr(llm_config, 'load_llm_config',
                        lambda: {'models': {'gemini': {'api_key': 'k'}}})
    payload = ('{"scores": [{"i": 0, "s": NaN}, {"i": 1, "s": Infinity}, '
               '{"i": 2, "s": -Infinity}, {"i": 3, "s": null}, '
               '{"i": 4, "s": "abc"}, {"i": 5, "s": 0.4}, '
               '{"i": Infinity, "s": 0.9}]}')
    resp = {'metadata': {'state': 'JOB_STATE_SUCCEEDED'},
            'response': {'inlinedResponses': [
                {'metadata': {'key': 'chunk-0'},
                 'response': {'candidates': [
                     {'content': {'parts': [{'text': payload}]}}]}}]}}
    monkeypatch.setattr(sh, '_gemini_batch_http', lambda *a, **k: resp)
    assert sh.poll_and_ingest_batch(db) == 'ingested'
    rows = dict(db.execute("SELECT id, llm_score FROM articles").fetchall())
    assert [rows[i[0]] for i in ids] == [None] * 5 + [0.4]


# ===========================================================================
# volatility._har_rrv_save — per-writer tmp
# ===========================================================================

@pytest.fixture
def har_file(tmp_path, monkeypatch):
    path = str(tmp_path / 'har_rrv.json')
    monkeypatch.setattr(volatility, '_HAR_RRV_FILE', path)
    monkeypatch.setitem(volatility._har_rrv_store, 'symbols',
                        {'AAA': {f'2026-01-{d:02d}': 0.01 * d
                                 for d in range(1, 29)}})
    return tmp_path, path


def test_har_rrv_save_interleaved_writers_no_tear(har_file, monkeypatch):
    """Writer A pauses mid-dump; writer B saves completely; A resumes.
    With a shared fixed '.tmp' B truncated A's file and A kept writing
    into the now-live inode -> torn JSON. Per-writer tmps can't collide."""
    tmp_dir, path = har_file
    real_dump = json.dump
    a_half_written = threading.Event()
    b_done = threading.Event()

    def shim(obj, f, *args, **kw):
        name = threading.current_thread().name
        if name == 'writer-A':
            text = json.dumps(obj)
            f.write(text[:len(text) // 2])
            f.flush()
            a_half_written.set()
            assert b_done.wait(10)
            f.write(text[len(text) // 2:])
        elif name == 'writer-B':
            real_dump({'B': 1}, f)          # short, distinct payload
        else:
            real_dump(obj, f, *args, **kw)

    monkeypatch.setattr(json, 'dump', shim)

    def run_b():
        assert a_half_written.wait(10)
        volatility._har_rrv_save()
        b_done.set()

    ta = threading.Thread(target=volatility._har_rrv_save, name='writer-A')
    tb = threading.Thread(target=run_b, name='writer-B')
    ta.start(); tb.start()
    ta.join(15); tb.join(15)
    assert not ta.is_alive() and not tb.is_alive()

    with open(path) as f:
        data = json.load(f)                  # torn file -> JSONDecodeError
    assert data == volatility._har_rrv_store['symbols']  # A replaced last
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []


def test_har_rrv_save_failure_removes_tmp(har_file, monkeypatch):
    tmp_dir, path = har_file

    def boom(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(volatility.os, 'replace', boom)
    volatility._har_rrv_save()               # never raises
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []
    assert not (tmp_dir / 'har_rrv.json').exists()


def test_har_rrv_save_roundtrip(har_file):
    tmp_dir, path = har_file
    volatility._har_rrv_save()
    with open(path) as f:
        assert json.load(f) == volatility._har_rrv_store['symbols']
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []
