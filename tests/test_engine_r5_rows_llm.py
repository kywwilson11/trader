"""ENGINE round-5 W17 — stock buy-row key parity (R5-a) + the ENGINE half of
INTEL's SCOUT_E LLM-call journal fields (research_intel.md § "R4 · Scout E").

R5-a (measurement-only, additive): stock_loop._execute_buys builds its OWN
buy row; it now appends the four R4-a join keys base _place_and_track_buy
writes (order_id of the acquiring bracket parent, decision_bid /
decision_ask / decision_quote_ts of the quote the decision used) via
setdefault after every legacy key — journal_stats' "one buy-row key set"
contract. client_order_id stays OUT (the replay determinism pins).

SCOUT_E, ENGINE half (base_loop._run_llm_analysis; measurement-only):
  * an exception escaping analyze_trades now leaves an 'llm_error' row
    (outcome, error_type, n_symbols_sent, latency_ms) and is RE-RAISED
    unchanged — no state or control-flow change;
  * the no-scores branch's existing 'llm_backoff' row gains outcome /
    n_symbols_sent / latency_ms AFTER its four legacy keys;
  * llm_analysis scores gain s_defaulted (A3) from
    llm_analyst.get_last_analysis_meta()['parse_flags'] (None when unknown);
  * the false "s journaled as null when the provider omitted it" comment is
    reworded (_parse_response substitutes 0.5).
Readers (llm_eval, decision_report, journal_stats, execution_report) give
identical output with and without the new rows/keys.
"""

import copy
import datetime as _dt
import inspect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import base_loop                                   # noqa: E402
import llm_analyst                                 # noqa: E402
import sentiment                                   # noqa: E402
from test_engine_r2_replay_harness import (        # noqa: E402,F401
    replay, _run_stock_day)

NEW_BUY = ('order_id', 'decision_bid', 'decision_ask', 'decision_quote_ts',
           'decision_quote_t')
NEW_ALL = frozenset(NEW_BUY)
STOCK_BUYS = {'AAA', 'BBB', 'DDD', 'EEE'}


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in NEW_ALL} for r in rows]


def _ev(broker):
    return [{k: v for k, v in e.items() if k not in ('t', 'client_order_id')}
            for e in broker.events]


# ===========================================================================
# R5-a — stock buy row carries the base buy row's join keys
# ===========================================================================

def _stock_run(replay, monkeypatch, stub_new=False):
    if stub_new:      # the pre-R5 stock row shape (and R4-less sells)
        monkeypatch.setattr(base_loop, '_order_journal_ids',
                            lambda order, keep_none=True: {})
        monkeypatch.setattr(base_loop, '_decision_quote_journal_keys',
                            lambda quote: {})
    loop, broker, tape = _run_stock_day(replay)
    return loop, broker


def test_stock_buy_rows_carry_order_id_and_decision_quote(replay):
    loop, broker, tape = _run_stock_day(replay)
    buys = replay.rec.rows('buy')
    assert {r['symbol'] for r in buys} == STOCK_BUYS
    for r in buys:
        assert r['entry_tactic'] == 'marketable_bracket'
        for k in NEW_BUY:
            assert k in r, (k, r)
        o = broker.orders[r['order_id']]
        # the acquiring bracket PARENT (not a leg), filled, for this name
        assert o['side'] == 'buy' and o['status'] == 'filled'
        assert o['order_class'] == 'bracket'
        assert o['symbol'].replace('/', '') == r['symbol']
        assert 'client_order_id' not in r
        # the decision quote: decision_price is its midpoint
        assert r['decision_bid'] < r['decision_ask']
        assert r['decision_price'] == pytest.approx(
            (r['decision_bid'] + r['decision_ask']) / 2.0, rel=0, abs=1e-12)
        # stock stamps no '_fetched_ts' -> get_quote's fetched_ts, which the
        # fake clock drives: inside the RTH session of this replay
        (open_utc, close_utc), = replay.sessions
        assert (open_utc.timestamp() <= r['decision_quote_ts']
                < close_utc.timestamp())


def test_stock_buy_row_preexisting_keys_byte_identical_ab(replay, monkeypatch):
    """In-process A/B on the r2 stock RTH replay: run A with the two R4
    helpers stubbed to {} (= the pre-R5 stock row), run B as shipped. Same
    broker traffic/trades/state; every row identical once the new keys are
    stripped; each stock buy row = A's keys in A's ORDER + exactly the four
    new keys appended."""
    out = []
    for stub in (True, False):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()
        with monkeypatch.context() as m:
            loop, broker = _stock_run(replay, m, stub_new=stub)
            out.append((copy.deepcopy(replay.rec.journal),
                        copy.deepcopy(replay.rec.trades), _ev(broker),
                        [c for c in broker.calls],
                        copy.deepcopy(replay.rec.state_blobs)))
    (ja, ta, ea, ca, sa), (jb, tb, eb, cb, sb) = out
    assert not any(NEW_ALL & set(r) for r in ja)
    assert _strip(jb) == ja
    assert (tb, eb, cb, sb) == (ta, ea, ca, sa)
    ba = [r for r in ja if r['action'] == 'buy']
    bb = [r for r in jb if r['action'] == 'buy']
    assert len(ba) == len(bb) == 4
    for a, b in zip(ba, bb):
        assert list(b) == list(a) + list(NEW_BUY)      # appended, in order
        assert [b[k] for k in a] == [a[k] for k in a]


def test_stock_buy_row_join_keys_never_break_the_row(replay, monkeypatch):
    """A helper that raises must not cost the buy row or change anything
    else (the join keys are best-effort)."""
    def boom(*a, **k):
        raise RuntimeError('join-key helper exploded')
    ref = []
    for broken in (False, True):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()
        with monkeypatch.context() as m:
            if broken:
                m.setattr(base_loop, '_decision_quote_journal_keys', boom)
            loop, broker = _stock_run(replay, m)
            ref.append((copy.deepcopy(replay.rec.journal),
                        copy.deepcopy(replay.rec.trades), _ev(broker)))
    (ja, ta, ea), (jb, tb, eb) = ref
    assert (tb, eb) == (ta, ea)
    buys_b = [r for r in jb if r['action'] == 'buy']
    assert {r['symbol'] for r in buys_b} == STOCK_BUYS
    for r in buys_b:
        assert not (NEW_ALL & set(r))                  # dropped, row kept
    assert _strip(jb) == _strip(ja)


def test_base_and_stock_buy_rows_share_the_r4_join_keys():
    """journal_stats.py:13-15 contract: the two buy-row producers carry the
    same R4/R5 join keys through the SAME helpers (one implementation)."""
    import stock_loop
    src = inspect.getsource(stock_loop.StockLoop._execute_buys)
    seg = src[src.index('buy_rec = {"symbol": symbol, "action": "buy"'):]
    seg = seg[:seg.index('log_decision(buy_rec)')]
    assert '_order_journal_ids(result)' in seg
    assert '_decision_quote_journal_keys(quote)' in seg
    assert 'buy_rec.setdefault(k, v)' in seg
    assert 'client_order_id' not in base_loop._JOURNAL_ID_FIELDS[0]


# ===========================================================================
# SCOUT_E — ENGINE half on base_loop._run_llm_analysis
# ===========================================================================

def _llm_self(**over):
    me = SimpleNamespace(
        _expire_llm_scores=lambda: None,
        _llm_backoff_until=0.0, _llm_fail_count=0, _llm_scores_ts=None,
        _last_llm_time=0.0, LLM_INTERVAL_SEC=600, LLM_SCORE_TTL_SEC=7200,
        _check_llm_staleness=lambda: None,
        _build_llm_candidates=lambda preds: [
            {'symbol': 'BTC/USD', 'pred_return': 0.5},
            {'symbol': 'ETH/USD', 'pred_return': -0.2}],
        positions={}, _equity=100000.0, config={},
        get_asset_type=lambda: 'crypto',
        llm_scores={'OLD': {'s': 0.9}}, _veto_strikes={'BTC/USD': 1},
        _last_llm_symbols={'OLD'},
    )
    for k, v in over.items():
        setattr(me, k, v)
    return me


@pytest.fixture
def llm_env(monkeypatch):
    journal = []
    env = SimpleNamespace(journal=journal, analyze=None,
                          meta={})
    monkeypatch.setattr(base_loop, 'load_llm_config', lambda: {'enabled': True})
    monkeypatch.setattr(base_loop, 'log_decision', lambda row: journal.append(row))
    monkeypatch.setattr(base_loop, 'analyze_trades',
                        lambda *a, **k: env.analyze(*a, **k))
    monkeypatch.setattr(sentiment, 'get_fear_greed', lambda: {'value': 50})
    monkeypatch.setattr(llm_analyst, 'get_last_analysis_meta',
                        lambda: dict(env.meta))
    env.run = lambda me: base_loop.BaseTradingLoop._run_llm_analysis(me, {})
    return env


def _state(me):
    return (copy.deepcopy(me.llm_scores), me._llm_scores_ts,
            me._llm_fail_count, me._llm_backoff_until,
            copy.deepcopy(me._veto_strikes), set(me._last_llm_symbols))


def test_analyze_trades_exception_is_journaled_then_reraised(llm_env):
    def raising(*a, **k):
        raise OverflowError('int too large to convert to float')
    llm_env.analyze = raising
    me = _llm_self()
    before = _state(me)
    with pytest.raises(OverflowError, match='int too large'):
        llm_env.run(me)
    # state untouched exactly as before the fix (the exception still
    # propagates to run(); nothing is reset, nothing backs off)
    assert _state(me) == before
    assert me._last_llm_time > 0                  # stamped on ATTEMPT, as before
    row, = llm_env.journal
    assert list(row) == ['action', 'asset_type', 'outcome', 'error_type',
                         'n_symbols_sent', 'latency_ms']
    assert row['action'] == 'llm_error' and row['asset_type'] == 'crypto'
    assert row['outcome'] == 'exception'
    assert row['error_type'] == 'OverflowError'   # class name only, no message
    assert row['n_symbols_sent'] == 2
    assert isinstance(row['latency_ms'], int) and 0 <= row['latency_ms'] < 60000
    json.dumps(row)


def test_exception_row_failure_never_masks_the_original_exception(llm_env,
                                                                  monkeypatch):
    def raising(*a, **k):
        raise ValueError('provider SDK blew up')

    def bad_journal(row):
        raise RuntimeError('journal exploded')
    llm_env.analyze = raising
    monkeypatch.setattr(base_loop, 'log_decision', bad_journal)
    with pytest.raises(ValueError, match='provider SDK'):
        llm_env.run(_llm_self())


def test_keyboard_interrupt_is_not_journaled(llm_env):
    def interrupted(*a, **k):
        raise KeyboardInterrupt
    llm_env.analyze = interrupted
    with pytest.raises(KeyboardInterrupt):
        llm_env.run(_llm_self())
    assert llm_env.journal == []


def test_no_scores_backoff_row_legacy_keys_then_new(llm_env):
    llm_env.analyze = lambda *a, **k: {}
    me = _llm_self()
    llm_env.run(me)
    assert me._llm_fail_count == 1
    assert me._llm_backoff_until == pytest.approx(me._last_llm_time + 600, abs=5)
    assert me.llm_scores == {'OLD': {'s': 0.9}}   # fail-open: untouched
    row, = llm_env.journal
    assert list(row)[:4] == ['action', 'asset_type', 'consecutive_failures',
                             'backoff_s']
    assert [row[k] for k in list(row)[:4]] == ['llm_backoff', 'crypto', 1, 600.0]
    assert list(row)[4:] == ['outcome', 'n_symbols_sent', 'latency_ms']
    assert row['outcome'] == 'no_scores' and row['n_symbols_sent'] == 2
    assert isinstance(row['latency_ms'], int) and row['latency_ms'] >= 0
    # None result takes the same branch
    llm_env.journal.clear()
    llm_env.analyze = lambda *a, **k: None
    me._llm_backoff_until = 0.0
    me._last_llm_time = 0.0
    llm_env.run(me)
    assert llm_env.journal[0]['consecutive_failures'] == 2
    assert llm_env.journal[0]['outcome'] == 'no_scores'


@pytest.mark.parametrize('flags,expect', [
    ({'BTC/USD': {'s_defaulted': True, 'raw_s': None},
      'ETH/USD': {'s_defaulted': False, 'raw_s': 0.7}}, [True, False]),
    ({'BTC/USD': {'s_defaulted': True}}, [True, None]),     # partial flags
    ({'BTC/USD': {'s_defaulted': 1}, 'ETH/USD': 'x'}, [None, None]),  # junk
    (None, [None, None]),                                   # no parse_flags
    ('junk', [None, None]),
])
def test_llm_analysis_scores_carry_s_defaulted(llm_env, flags, expect):
    llm_env.analyze = lambda *a, **k: {'BTC/USD': {'s': 0.5, 'm': 1.0},
                                       'ETH/USD': {'s': 0.7, 'm': 1.2}}
    llm_env.meta = {'model': 'm1', 'prompt_sha256': 'ab' * 32,
                    'dedup_hit': False, 'latency_ms': 42}
    if flags is not None:
        llm_env.meta['parse_flags'] = flags
    me = _llm_self()
    llm_env.run(me)
    row, = llm_env.journal
    assert list(row) == ['action', 'asset_type', 'forward_bars', 'scores',
                         'model', 'prompt_sha256', 'dedup_hit', 'latency_ms']
    assert list(row['scores']) == ['BTC/USD', 'ETH/USD']
    for sym, want in zip(('BTC/USD', 'ETH/USD'), expect):
        sc = row['scores'][sym]
        assert list(sc) == ['s', 'pred', 's_defaulted']   # legacy s,pred first
        assert sc['s_defaulted'] is want
    assert row['scores']['BTC/USD']['s'] == 0.5         # s itself unchanged
    assert row['scores']['ETH/USD']['pred'] == -0.2
    assert me._veto_strikes == {}                     # gate path unchanged
    assert me._llm_fail_count == 0


def test_s_defaulted_none_when_meta_unavailable(llm_env, monkeypatch):
    def boom():
        raise RuntimeError('meta exploded')
    monkeypatch.setattr(llm_analyst, 'get_last_analysis_meta', boom)
    llm_env.analyze = lambda *a, **k: {'BTC/USD': {'s': 0.1}}
    me = _llm_self()
    llm_env.run(me)
    row, = llm_env.journal
    assert row['scores'] == {'BTC/USD': {'s': 0.1, 'pred': 0.5,
                                         's_defaulted': None}}
    assert me._veto_strikes == {'BTC/USD': 2}          # strike logic untouched


def test_false_d33_comment_reworded_and_meeting_point_exists():
    src = inspect.getsource(base_loop.BaseTradingLoop._run_llm_analysis)
    assert 'journaled as null when the provider omitted it' not in src
    assert "get_last_analysis_meta()['parse_flags']" in src
    # INTEL's half of the meeting point (llm_analyst attaches parse_flags
    # carrying s_defaulted to the success meta)
    asrc = inspect.getsource(llm_analyst.analyze_trades)
    assert "_LAST_CALL_META['parse_flags']" in asrc
    assert 's_defaulted' in inspect.getsource(llm_analyst)


# ===========================================================================
# Readers unaffected by the new rows / keys
# ===========================================================================

def _llm_rows(with_new, base):
    t = lambda m: (base + timedelta(minutes=m)).isoformat()   # noqa: E731
    rows = [
        {'action': 'llm_analysis', 'asset_type': 'crypto', 'forward_bars': 24,
         'scores': {'AAA/USD': {'s': 0.6, 'pred': 0.1},
                    'BBB/USD': {'s': 0.1, 'pred': 0.2}},
         'model': 'm1', 'prompt_sha256': 'shaX', 'dedup_hit': False,
         'latency_ms': 900, 'cost_usd': 0.01, 'ts': t(0)},
        {'action': 'llm_backoff', 'asset_type': 'crypto',
         'consecutive_failures': 1, 'backoff_s': 600.0, 'ts': t(10)},
        {'action': 'llm_analysis', 'asset_type': 'crypto', 'forward_bars': 24,
         'scores': {'AAA/USD': {'s': 0.5, 'pred': -0.1}},
         'model': 'm1', 'prompt_sha256': 'shaY', 'dedup_hit': False,
         'latency_ms': 700, 'ts': t(30)},
        {'symbol': 'AAA/USD', 'action': 'buy', 'pred_return': 0.4,
         'final_notional': 500, 'decision_price': 10.0, 'fill_price': 10.01,
         'slippage_bps': 10.0, 'quote_age_s': 1.0, 'entry_tactic': 'marketable',
         'maker': False, 'skip_reason': None, 'ts': t(31)},
        {'symbol': 'AAA/USD', 'action': 'sell', 'exit_reason': 'signal_sell',
         'pnl_pct': 1.0, 'decision_price': 10.1, 'fill_price': 10.1,
         'slippage_bps': 0.0, 'estimated': False, 'ts': t(40)},
    ]
    if with_new:
        rows[0]['scores']['AAA/USD']['s_defaulted'] = False
        rows[0]['scores']['BBB/USD']['s_defaulted'] = True
        rows[1].update(outcome='no_scores', n_symbols_sent=2, latency_ms=45000)
        rows[2]['scores']['AAA/USD']['s_defaulted'] = None
        rows.insert(2, {'action': 'llm_error', 'asset_type': 'crypto',
                        'outcome': 'exception', 'error_type': 'OverflowError',
                        'n_symbols_sent': 2, 'latency_ms': 1200,
                        'ts': t(20)})
    return rows


def _write(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_dt.date.today().isoformat()}.jsonl"
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def _readers(monkeypatch, d, out_dir, now, base):
    import decision_report
    import execution_report
    import journal_stats
    import llm_eval
    import trade_journal
    monkeypatch.setattr(decision_report, 'JOURNAL_DIR', d)
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', d)
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', d)
    monkeypatch.setattr(llm_eval, 'JOURNAL_DIR', d)
    monkeypatch.setattr(llm_eval, 'BASE_DIR', out_dir)
    monkeypatch.setattr(llm_eval, '_read_daily_cost', lambda: None)
    ts_full = base.timestamp() - 48 * 3600 + np.arange(24 * 10) * 3600.0
    closes = 100.0 + np.arange(len(ts_full)) * 0.01
    monkeypatch.setattr(llm_eval, '_bars_lookup',
                        lambda api, symbol, asset_type, start, end:
                        (ts_full, closes))
    out_dir.mkdir(parents=True, exist_ok=True)
    ev = llm_eval.run_eval(days=1, api=object(), out_dir=out_dir)
    ev = json.loads(json.dumps(ev, default=str))
    for k in ('generated_at',):
        ev.pop(k, None)
        (ev.get('meta') or {}).pop(k, None)
    dr_rows = decision_report.load_journal(1)
    trades = journal_stats.load_trades(d)
    stats = journal_stats.compute_stats(trades)
    er = execution_report.run_report(days=1, out_dir=out_dir)
    er.pop('generated_at', None)
    return {
        'llm_eval.run_eval': ev,
        'decision_report.load_journal': dr_rows,
        'decision_report.admitted_k': decision_report.admitted_k_distribution(dr_rows),
        'journal_stats.load_trades': trades,
        'journal_stats.compute_stats': stats,
        'journal_stats.format_summary': journal_stats.format_summary(stats),
        'journal_stats.eod_digest': journal_stats.build_eod_digest(d, now=now),
        'execution_report.run_report': er,
    }


def test_readers_unaffected_by_llm_rows_and_keys(monkeypatch, tmp_path):
    now = _dt.datetime.now().astimezone().replace(microsecond=0)
    base = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=2)
    outs = []
    for with_new in (False, True):
        d = tmp_path / f'j{int(with_new)}'
        _write(d, _llm_rows(with_new, base))
        outs.append(_readers(monkeypatch, d, tmp_path / f'o{int(with_new)}',
                             now, base))
    legacy, new = outs
    assert new == legacy
    # not vacuous: every reader saw the rows under test
    ev = legacy['llm_eval.run_eval']
    assert ev and ev['coverage']['n_rows_scored'] == 3
    la = legacy['execution_report.run_report']['llm_analysis']
    assert la['n_calls'] == 2 and la['n_backoffs'] == 1
    assert 'LLM calls' in legacy['journal_stats.eod_digest']
    assert len(legacy['journal_stats.load_trades']) == 1


# ===========================================================================
# R5-b — get_quote 'quote_t' (exchange quote time) + buy-row decision_quote_t
# ===========================================================================

import order_utils                                 # noqa: E402


class _QApi:
    def __init__(self, q):
        self.q = q

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: self.q}

    def get_latest_quote(self, symbol):
        return self.q


def _utcnow():
    return datetime.now(timezone.utc).replace(microsecond=123456)


@pytest.mark.parametrize('asset', ['crypto', 'stock'])
@pytest.mark.parametrize('form', ['aware_utc', 'naive_utc', 'pd_ny', 'pd_ns'])
def test_get_quote_quote_t_is_the_exchange_epoch(asset, form):
    import pandas as pd
    t = _utcnow() - timedelta(seconds=7)
    raw = {'aware_utc': t,
           'naive_utc': t.replace(tzinfo=None),            # Alpaca = UTC
           'pd_ny': pd.Timestamp(t).tz_convert('America/New_York'),
           'pd_ns': pd.Timestamp(t) + pd.Timedelta(nanoseconds=789)}[form]
    out = order_utils.get_quote(_QApi(SimpleNamespace(bp=100.0, ap=100.1,
                                                      t=raw)), 'X', asset)
    assert out is not None
    assert isinstance(out['quote_t'], float)
    assert out['quote_t'] == t.timestamp()      # ns truncated to µs, as the age
    assert list(out) == ['bid', 'ask', 'spread', 'midpoint', 'spread_pct',
                         'fetched_ts', 'quote_t']          # appended last
    assert out['fetched_ts'] - out['quote_t'] == pytest.approx(7.0, abs=2.0)


class _NoEpoch(datetime):
    """Ages fine (subtraction works) but .timestamp() raises: quote_t must
    be None while the accept verdict is unchanged."""
    def timestamp(self):
        raise OverflowError('no epoch')


def test_get_quote_quote_t_none_safe_and_verdict_unchanged():
    t = _utcnow() - timedelta(seconds=5)
    ok = _NoEpoch(*t.timetuple()[:6], t.microsecond, tzinfo=timezone.utc)
    out = order_utils.get_crypto_quote(_QApi(SimpleNamespace(
        bp=100.0, ap=100.1, t=ok)), 'BTC/USD')
    assert out is not None and out['quote_t'] is None
    assert out['midpoint'] == (100.0 + 100.1) / 2.0
    stale = _utcnow() - timedelta(seconds=400)
    assert order_utils.get_crypto_quote(_QApi(SimpleNamespace(
        bp=100.0, ap=100.1, t=stale)), 'BTC/USD') is None


def test_decision_quote_t_helper_passthrough():
    f = base_loop._decision_quote_journal_keys
    assert f({'bid': 1.0, 'ask': 2.0, 'fetched_ts': 9.0,
              'quote_t': 8.5})['decision_quote_t'] == 8.5
    assert f({'bid': 1.0})['decision_quote_t'] is None


def test_stock_buy_rows_carry_the_exchange_quote_time(replay):
    loop, broker, tape = _run_stock_day(replay)
    buys = replay.rec.rows('buy')
    assert {r['symbol'] for r in buys} == STOCK_BUYS
    for r in buys:
        qt, fts = r['decision_quote_t'], r['decision_quote_ts']
        assert isinstance(qt, float) and isinstance(fts, float)
        assert 0.0 <= fts - qt <= 180.0          # passed the staleness gate


def test_crypto_buy_rows_carry_the_stamped_quotes_quote_t(replay, monkeypatch):
    from test_engine_r4_journal_flatten import _run_cash
    loop, broker, seen = _run_cash(replay, monkeypatch, False)
    buys = replay.rec.rows('buy')
    assert len(buys) == 2
    for r in buys:
        stamped = [q for s, q in seen if s == r['symbol']
                   and q.get('_fetched_ts') == r['decision_quote_ts']]
        assert stamped and isinstance(r['decision_quote_t'], float)
        assert r['decision_quote_t'] == stamped[-1]['quote_t']
