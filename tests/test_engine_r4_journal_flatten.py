"""ENGINE round-4 W13 — journal join keys (R4-a) + flatten-site exit rows (R4-b).

R4-a (measurement-only, additive): base_loop._place_and_track_buy's buy row
now carries the broker ``order_id`` of the order that
ACQUIRED the fill (maker ladder: the rung place_maker_buy returns) and the
decision quote's ``decision_bid`` / ``decision_ask`` / ``decision_quote_ts``
(the quote dict _execute_buys stamped — no new fetch). Sell rows written by
_record_confirmed_exit / _execute_stop_exit add ``order_id`` ONLY when the
exit order carries one (the legacy key-set pin
tests/test_c26_T6.py::TestRecordConfirmedExit stays green).
``client_order_id`` is NOT journaled: its uuid4 tail would break the
harness's journal-determinism pins; the broker order carries it.
``entry_tactic`` already existed and is untouched. Every pre-existing key is
byte-identical (in-process A/B below) and the journal readers produce the
same output with and without the new keys.

R4-b: _check_flatten_request (runs FIRST in the cycle) reuses
_breaker_record_server_fill under BREAKER_SERVER_FILL_ATTRIB (site
'remote_flatten'); OFF is byte-identical (no get_order). The stablecoin
emergency flatten (_update_macro_regime) now journals an estimated
'stablecoin_flatten' exit row per released position (trade memory, same
shape as the breaker's) and saves position state; flag ON = the same
server-fill attribution (site 'stablecoin_flatten'). No order-flow change.
"""

import copy
import datetime as _dt
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import base_loop                                   # noqa: E402
import strategy_config                             # noqa: E402
from types_mod import MacroRegime, Position        # noqa: E402
from fake_alpaca_broker import FakeClock           # noqa: E402
from test_engine_r2_replay_harness import (        # noqa: E402,F401
    replay, CP, SNAPSHOT, UNIVERSE, _load_tape, _crypto_cash_snapshot,
    _preds_b, _run_stock_day)

NEW_BUY = ('order_id', 'decision_bid', 'decision_ask',
           'decision_quote_ts', 'decision_quote_t')   # + r5 exchange time
NEW_ALL = frozenset(NEW_BUY)


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in NEW_ALL} for r in rows]


def _ev(broker):
    # 't' is the fake's wall clock (FakeClock() = real now) and the client
    # ids embed a uuid4 — neither is comparable across two runs.
    return [{k: v for k, v in e.items() if k not in ('t', 'client_order_id')}
            for e in broker.events]


# ===========================================================================
# R4-a — pure helpers
# ===========================================================================

class _Boom:
    def __getattr__(self, name):
        raise RuntimeError('adapter exploded')


def test_order_journal_ids_shapes_and_never_raises():
    f = base_loop._order_journal_ids
    o = SimpleNamespace(id='ord-1', client_order_id='maker-x')
    assert f(o) == {'order_id': 'ord-1'}          # client id NOT journaled
    u = uuid.UUID('12345678-1234-5678-1234-567812345678')
    assert f(SimpleNamespace(id=u)) == {'order_id': str(u)}   # alpaca-py UUID
    for bad in (None, SimpleNamespace(filled_avg_price='1'), _Boom()):
        assert f(bad) == {'order_id': None}
        assert f(bad, keep_none=False) == {}
    assert f(SimpleNamespace(id='a'), keep_none=False) == {'order_id': 'a'}


def test_decision_quote_journal_keys_shapes_and_never_raises():
    f = base_loop._decision_quote_journal_keys
    none = {'decision_bid': None, 'decision_ask': None,
            'decision_quote_ts': None, 'decision_quote_t': None}
    q = {'bid': 99.0, 'ask': 101.0, 'midpoint': 100.0, 'fetched_ts': 5.0,
         '_fetched_ts': 7.0}
    assert f(q) == {'decision_bid': 99.0, 'decision_ask': 101.0,
                    'decision_quote_ts': 7.0,        # the buy row's own stamp
                    'decision_quote_t': None}        # no get_quote quote_t
    assert f({'bid': 1.0, 'ask': 2.0, 'fetched_ts': 5.0})['decision_quote_ts'] == 5.0
    assert f({'midpoint': 50000.0}) == none           # test_grp_loops' stub shape
    for bad in (None, 'q', 3, _Boom()):
        assert f(bad) == none


# ===========================================================================
# R4-a — buy / sell rows on the fake broker
# ===========================================================================

def _record_quotes(loop):
    seen = []
    real = loop.get_quote

    def wrapped(sym):
        q = real(sym)
        if q is not None:
            seen.append((sym, q))
        return q
    loop.get_quote = wrapped
    return seen


def _run_cash(replay, monkeypatch, maker, stub_new=False):
    monkeypatch.setattr(strategy_config, 'MAKER_ENTRIES_ENABLED', maker)
    if stub_new:     # the pre-R4 row shape: no new key is ever produced
        monkeypatch.setattr(base_loop, '_order_journal_ids',
                            lambda order, keep_none=True: {})
        monkeypatch.setattr(base_loop, '_decision_quote_journal_keys',
                            lambda quote: {})
    tape = _load_tape()
    n = len(tape) - 1
    loop, broker = replay.build_crypto(snapshot=_crypto_cash_snapshot(),
                                       tape=tape, preds_fn=_preds_b,
                                       replay_quote_age=True)
    seen = _record_quotes(loop)
    replay.drive(loop, broker, n)
    return loop, broker, seen


def _assert_buy_row_joins(rec, broker, seen):
    buys = rec.rows('buy')
    assert {r['symbol'] for r in buys} == {'SOL/USD', 'XRP/USD'}
    for r in buys:
        # the broker order that filled this entry
        fills = [e for e in broker.events_of('fill', r['symbol'])
                 if e['side'] == 'buy']
        assert r['order_id'] in {e['id'] for e in fills}
        o = broker.orders[r['order_id']]
        assert o['status'] == 'filled' and o['side'] == 'buy'
        assert 'client_order_id' not in r and o['client_order_id']
        # the decision quote: the dict _execute_buys stamped for this name
        stamped = [q for s, q in seen if s == r['symbol'] and '_fetched_ts' in q]
        match = [q for q in stamped
                 if (q['bid'], q['ask'], q['_fetched_ts'])
                 == (r['decision_bid'], r['decision_ask'],
                     r['decision_quote_ts'])]
        assert match, (r, stamped)
        assert r['decision_price'] == match[-1]['midpoint']
    return buys


@pytest.mark.parametrize('maker', [True, False])
def test_buy_row_carries_broker_ids_and_decision_quote(replay, monkeypatch,
                                                       maker):
    loop, broker, seen = _run_cash(replay, monkeypatch, maker)
    buys = _assert_buy_row_joins(replay.rec, broker, seen)
    for r in buys:
        assert r['entry_tactic'] == ('taker_fallback' if maker else 'marketable')
        # the join recovers the tactic tag from the broker order
        assert broker.orders[r['order_id']]['client_order_id'].startswith('mktfb-')


@pytest.mark.parametrize('maker', [True, False])
def test_preexisting_keys_byte_identical_ab(replay, monkeypatch, maker):
    """In-process A/B: run A with the two R4 helpers stubbed to {} (= the
    pre-R4 row shape), run B as shipped. Same broker traffic, same trade
    memory, same state; every row identical once the new keys are
    stripped; B's buy rows = A's key set + exactly the five new keys."""
    out = []
    for stub in (True, False):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()
        with monkeypatch.context() as m:
            loop, broker, seen = _run_cash(replay, m, maker, stub_new=stub)
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
    assert len(ba) == len(bb) == 2
    for a, b in zip(ba, bb):
        assert set(b) == set(a) | NEW_ALL
        assert list(b)[:len(a)] == list(a)          # legacy key ORDER too
    sa_ = [r for r in ja if r['action'] == 'sell']
    sb_ = [r for r in jb if r['action'] == 'sell']
    assert [set(b) - set(a) for a, b in zip(sa_, sb_)] == [
        {'order_id'}]


def test_signal_sell_row_carries_the_sell_order_ids(replay, monkeypatch):
    loop, broker, seen = _run_cash(replay, monkeypatch, True)
    sells = [r for r in replay.rec.rows('sell') if r['symbol'] == 'ETH/USD']
    assert [r['exit_reason'] for r in sells] == ['signal_sell']
    r = sells[0]
    fill = [e for e in broker.events_of('fill', 'ETH/USD')
            if e['side'] == 'sell']
    assert [e['id'] for e in fill] == [r['order_id']]
    assert 'client_order_id' not in r
    assert broker.orders[r['order_id']]['client_order_id'].startswith('csell-')


def test_stock_server_stop_and_eod_rows_carry_order_ids(replay):
    loop, broker, tape = _run_stock_day(replay)
    sells = replay.rec.rows('sell')
    assert {r['exit_reason'] for r in sells} == {'server_stop', 'eod_flatten'}
    for r in sells:
        o = broker.orders[r['order_id']]
        assert o['status'] == 'filled' and o['side'] == 'sell'
        assert _norm(o['symbol']) == _norm(r['symbol'])
        assert 'client_order_id' not in r


def _norm(s):
    return s.replace('/', '')


def test_record_confirmed_exit_without_ids_keeps_legacy_keyset(monkeypatch):
    rows = []
    monkeypatch.setattr(base_loop, 'log_decision', lambda r: rows.append(r))
    monkeypatch.setattr(base_loop, 'record_trade', lambda *a, **k: None)
    inst = SimpleNamespace(_breaker_note_realized=lambda pnl: None)
    pos = Position(qty=1.0, entry_price=100.0, high_water_mark=100.0)
    rce = base_loop.BaseTradingLoop._record_confirmed_exit
    rce(inst, 'BTC/USD', pos, SimpleNamespace(filled_avg_price='99.0'), None,
        exit_reason='signal_sell')
    rce(inst, 'BTC/USD', pos, None, None, exit_reason='signal_sell')
    rce(inst, 'BTC/USD', pos, SimpleNamespace(filled_avg_price='99.0',
                                              id='o-9', client_order_id='c-9'),
        None, exit_reason='signal_sell', extra={'detect_source': 'x'})
    legacy = {'symbol', 'action', 'exit_reason', 'pnl_pct', 'decision_price',
              'fill_price', 'slippage_bps', 'quote_age_s', 'estimated'}
    assert set(rows[0]) == legacy and set(rows[1]) == legacy
    assert set(rows[2]) == legacy | {'detect_source', 'order_id'}
    assert rows[2]['order_id'] == 'o-9'


def test_stop_exit_row_carries_the_exit_order_ids(monkeypatch):
    import order_utils
    rows, trades = [], []
    monkeypatch.setattr(base_loop, 'log_decision', lambda r: rows.append(r))
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    monkeypatch.setattr(order_utils, 'cancel_orders_for_symbol',
                        lambda api, s, timeout=5: True)
    filled = SimpleNamespace(id='o-7', client_order_id='stop-abc',
                             status='filled', filled_avg_price='95.0')
    monkeypatch.setattr(base_loop, 'manage_order_lifecycle',
                        lambda api, oid, **k: filled)

    class Api:
        def submit_order(self, **k):
            return SimpleNamespace(id='o-7', client_order_id=k['client_order_id'])

    class L:
        _execute_stop_exit = base_loop.BaseTradingLoop._execute_stop_exit
        ORDER_TIMEOUT = 1

        def get_asset_type(self):
            return 'crypto'

        def _breaker_note_realized(self, pnl):
            pass

        def _save_hard_stop_lockout(self):
            pass
    inst = L()
    inst.api, inst.llm_scores, inst.last_trade_time = Api(), {}, {}
    inst.hard_stop_lockout = {}
    inst.HARD_STOP_LOCKOUT_HOURS = 24
    pos = Position(qty=1.0, entry_price=100.0, high_water_mark=100.0)
    inst.positions = {'BTC/USD': pos}
    inst._execute_stop_exit('BTC/USD', pos, 'trailing', 96.0)
    assert len(rows) == 1
    r = rows[0]
    assert r['order_id'] == 'o-7' and 'client_order_id' not in r
    assert r['exit_reason'] == 'trailing' and r['fill_price'] == 95.0
    assert 'BTC/USD' not in inst.positions


# ===========================================================================
# R4-a — the journal readers are unaffected by the new keys
# ===========================================================================

def _journal_rows(with_new, ts):
    t = lambda h: (ts - _dt.timedelta(hours=h)).isoformat()   # noqa: E731
    rows = [
        {'action': 'entry_window', 'asset_type': 'crypto', 'n_candidates': 3,
         'admitted_k': 1, 'admitted': ['SOL/USD'],
         'veto_counts': {'no_pred': 1}, 'buys_allowed': True, 'ts': t(5)},
        {'symbol': 'XRP/USD', 'action': 'skip', 'skip_reason': 'cost',
         'pred_return': 0.2, 'spread_pct': 0.1, 'ts': t(5)},
        {'symbol': 'SOL/USD', 'action': 'buy', 'pred_return': 2.0,
         'sentiment_gate': 1.0, 'sentiment_reasons': [], 'llm_multiplier': 1.0,
         'llm_score': 0.5, 'llm_reasoning': '', 'final_notional': 500,
         'decision_price': 120.0, 'fill_price': 120.06, 'slippage_bps': 5.0,
         'quote_age_s': 1.2, 'entry_tactic': 'maker', 'maker': True,
         'skip_reason': None, 'ts': t(4)},
        {'symbol': 'ETH/USD', 'action': 'buy', 'pred_return': 1.0,
         'final_notional': 800, 'decision_price': 2700.0, 'fill_price': 2701.0,
         'slippage_bps': 3.7, 'quote_age_s': 0.5,
         'entry_tactic': 'taker_fallback', 'maker': False, 'skip_reason': None,
         'ts': t(4)},
        {'symbol': 'SOL/USD', 'action': 'sell', 'exit_reason': 'signal_sell',
         'pnl_pct': 2.5, 'decision_price': 123.0, 'fill_price': 123.06,
         'slippage_bps': -4.9, 'quote_age_s': 0.8, 'estimated': False,
         'ts': t(2)},
        {'symbol': 'ETH/USD', 'action': 'sell', 'exit_reason': 'server_stop',
         'pnl_pct': -3.0, 'decision_price': None, 'fill_price': 2620.0,
         'slippage_bps': None, 'quote_age_s': None, 'estimated': False,
         'server_stop_kind': 'hard', 'stop_px': 2620.0, 'ts': t(1)},
    ]
    if with_new:
        rows[2].update(order_id='ord-1',
                       decision_bid=119.9, decision_ask=120.1,
                       decision_quote_ts=1.79e9)
        rows[3].update(order_id='ord-2',
                       decision_bid=2699.0, decision_ask=2701.0,
                       decision_quote_ts=None)
        rows[4].update(order_id='ord-3')
        rows[5].update(order_id='ord-4')
    return rows


def _write_journal(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{_dt.date.today().isoformat()}.jsonl"
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def _reader_outputs(monkeypatch, d, out_dir, now):
    import decision_report
    import execution_report
    import fees
    import journal_stats
    import trade_journal
    monkeypatch.setattr(decision_report, 'JOURNAL_DIR', d)
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', d)
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', d)
    monkeypatch.setattr(fees, '_maker_share_cache', {})
    dr_rows = decision_report.load_journal(1)
    st = {}
    trades = journal_stats.load_trades(d, stats=st)
    stats = journal_stats.compute_stats(trades)
    er = execution_report.run_report(days=1, out_dir=out_dir)
    er.pop('generated_at', None)
    return {
        'decision_report.load_journal': dr_rows,
        'decision_report.admitted_k': decision_report.admitted_k_distribution(dr_rows),
        'journal_stats.load_trades': trades,
        'journal_stats.stats_counts': st,
        'journal_stats.compute_stats': stats,
        'journal_stats.format_summary': journal_stats.format_summary(stats),
        'journal_stats.eod_digest': journal_stats.build_eod_digest(d, now=now),
        'execution_report.run_report': er,
        'fees.maker_share': fees.realized_crypto_maker_share(days=1,
                                                             min_entries=1),
    }


def test_journal_readers_unaffected_by_new_keys(monkeypatch, tmp_path):
    now = _dt.datetime.now().astimezone().replace(microsecond=0)
    outs = []
    for with_new in (False, True):
        d = tmp_path / f'j{int(with_new)}'
        _write_journal(d, _journal_rows(with_new, now))
        outs.append(_reader_outputs(monkeypatch, d,
                                    tmp_path / f'o{int(with_new)}', now))
    legacy, new = outs
    assert new == legacy
    # the comparison is not vacuous: every reader saw the trade rows
    assert len(legacy['journal_stats.load_trades']) == 2
    assert legacy['fees.maker_share'] == 0.5
    assert any(r['action'] == 'buy' for r in legacy['decision_report.load_journal'])
    assert legacy['execution_report.run_report'] != {'window_days': 1}


def test_fill_venue_report_never_reads_the_decision_journal():
    """scripts/fill_venue_slippage_report.py joins BROKER bundles only
    (orders/fills/quotes/bars): its loaders cannot see journal rows, so the
    new keys cannot change its output — they let a later join key live
    rows to its order ids (the broker order's client_order_id classifies
    the tactic)."""
    src = (REPO / 'scripts' / 'fill_venue_slippage_report.py').read_text()
    for token in ('JOURNAL_DIR', 'open_journal', 'iter_journal_rows',
                  'trade_journal', '.jsonl'):
        assert token not in src, token


# ===========================================================================
# R4-b — remote /flatten: server-fill attribution under the SAME flag
# ===========================================================================

def _gap_tape():
    return {s: [CP[s] * m for m in (1.0, 0.93, 0.93)] for s in CP}


def _set_flag(monkeypatch, attrib):
    monkeypatch.delenv('TRADER_BREAKER_SERVER_FILL_ATTRIB', raising=False)
    monkeypatch.setattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB', attrib)


def _wrap_calls(loop, broker, name, sink):
    real = getattr(loop, name)

    def wrapped(*a, **k):
        n0 = len(broker.calls)
        out = real(*a, **k)
        sink.append(list(broker.calls[n0:]))
        return out
    setattr(loop, name, wrapped)


def _remote_flatten_run(replay, monkeypatch, tmp_path, attrib):
    _set_flag(monkeypatch, attrib)
    loop, broker = replay.build_crypto(tape=_gap_tape(), clock=FakeClock())
    calls = []
    _wrap_calls(loop, broker, '_check_flatten_request', calls)

    def before(loop, broker):
        if loop.cycle == 0:     # the request lands with the gap tick
            f = tmp_path / 'flatten_crypto.flag'
            f.write_text('1')
            t = broker.clock().timestamp()
            os.utime(f, (t, t))
    replay.drive(loop, broker, 2, before_cycle=before)
    stop_fills = {e['symbol']: e['fill_price'] for e in broker.events_of('fill')
                  if e['side'] == 'sell' and e['type'] == 'stop_limit'}
    return loop, broker, calls[0], stop_fills


def test_remote_flatten_flag_off_is_legacy_estimated_rows(replay, monkeypatch,
                                                           tmp_path):
    loop, broker, calls, stop_fills = _remote_flatten_run(
        replay, monkeypatch, tmp_path, False)
    rec = replay.rec
    assert len(stop_fills) == 6 and broker.positions == {}
    assert 'list_positions' in [c[0] for c in calls]
    assert [c for c in calls if c[0] == 'get_order'] == []      # no probe
    assert [(a[0], k) for a, k in rec.trades] == [
        (s, {'exit_reason': 'remote_flatten', 'estimated': True})
        for s in UNIVERSE]
    for a, k in rec.trades:      # the quote MID, not the real stop fill
        assert a[3] == pytest.approx(CP[a[0]] * 0.93, rel=1e-9)
        assert a[3] != pytest.approx(stop_fills[a[0]], rel=1e-9)
    assert not rec.rows('sell') and loop.hard_stop_lockout == {}


def test_remote_flatten_flag_on_attributes_filled_server_stops(
        replay, monkeypatch, tmp_path):
    loop, broker, calls, stop_fills = _remote_flatten_run(
        replay, monkeypatch, tmp_path, True)
    rec = replay.rec
    names = [c[0] for c in calls]
    assert names.count('get_order') == 6
    assert names.index('get_order') > names.index('list_positions')
    sells = {r['symbol']: r for r in rec.rows('sell')}
    assert set(sells) == set(UNIVERSE)
    for s, r in sells.items():
        assert r['exit_reason'] == 'server_stop' and r['estimated'] is False
        assert r['detect_source'] == 'remote_flatten'
        assert r['fill_price'] == pytest.approx(stop_fills[s])
        assert r['order_id'] == loop_stop_id(broker, s)
    assert not [k for a, k in rec.trades
                if k.get('exit_reason') == 'remote_flatten']
    assert set(loop.hard_stop_lockout) == set(UNIVERSE)
    assert set(rec.state_blobs[0]['last_trade']) == set(UNIVERSE)


def loop_stop_id(broker, sym):
    ids = [e['id'] for e in broker.events_of('fill', sym)
           if e['side'] == 'sell' and e['type'] == 'stop_limit']
    assert len(ids) == 1
    return ids[0]


# ===========================================================================
# R4-b — stablecoin emergency flatten: exit rows + state save
# ===========================================================================

def _stable_run(replay, monkeypatch, attrib, tape, skip_stops=False):
    _set_flag(monkeypatch, attrib)
    monkeypatch.setattr(base_loop, 'get_macro_regime', lambda api, at: MacroRegime(
        stress_level=1.0, vix=18.0, cape=None, regime_label='contagion',
        sizing_mult=0.0, stablecoin_alert=True))
    loop, broker = replay.build_crypto(tape=tape, clock=FakeClock())
    if skip_stops:
        # A stop that fills AFTER _manage_stops looked (between it and the
        # regime refresh): simulated by skipping the stop pass and the
        # breaker (a -7 % gap would otherwise trip it first).
        loop._manage_stops = lambda: None
        loop._circuit_breaker_check = lambda: False
    calls = []
    _wrap_calls(loop, broker, '_update_macro_regime', calls)
    marks = replay.drive(loop, broker, 2)
    return loop, broker, calls[0], marks


def test_stablecoin_flatten_journals_exits_and_saves_state(replay,
                                                            monkeypatch):
    tape = {s: [CP[s]] * 3 for s in CP}
    loop, broker, calls, marks = _stable_run(replay, monkeypatch, False, tape)
    rec = replay.rec
    assert loop.positions == {} and broker.positions == {}
    # one estimated exit row per released position, at the quote mid,
    # in the breaker's row shape
    assert [(a[0], a[1], a[3], k) for a, k in rec.trades] == [
        (s, 'sell', pytest.approx(CP[s], rel=1e-9),
         {'exit_reason': 'stablecoin_flatten', 'estimated': True})
        for s in UNIVERSE]
    # state saved after the release (was: still six HWMs until next cycle)
    assert rec.state_blobs[0]['hwm'] == {}
    # order flow unchanged: cycle 1's submits are exactly the flatten's six
    # market sells (A/B vs the pre-edit copy: identical broker events)
    # (plus the pre-existing cycle-1 zero-basis stop re-placements, O2)
    subs = [e for e in broker.events[marks[0]:marks[1]] if e['kind'] == 'submit']
    mkt = [e for e in subs if e['type'] == 'market']
    assert sorted(_norm(e['symbol']) for e in mkt) == sorted(map(_norm, UNIVERSE))
    assert all(e['side'] == 'sell' for e in mkt)
    assert all(e['type'] == 'stop_limit' for e in subs if e not in mkt)
    # OFF: the resting stop ids are never probed
    stop_ids = {e['id'] for e in broker.events[:marks[0]]
                if e['kind'] == 'submit' and e['type'] == 'stop_limit'}
    assert len(stop_ids) == 6
    assert not [c for c in calls if c[0] == 'get_order' and c[1] in stop_ids]


def test_stablecoin_flatten_row_failure_never_breaks_the_branch(monkeypatch):
    """test_grp_loops' stub shape (int positions, api None): the new rows
    fail per position, are logged, and the position bookkeeping is the
    same as before."""
    import crypto_loop
    inst = object.__new__(crypto_loop.CryptoLoop)
    inst.api = None
    regime = SimpleNamespace(regime_label='bear', sizing_mult=0.0,
                             stop_mult=1.0, stablecoin_alert=True)
    # Patch the globals the method ACTUALLY resolves (suite-order safe):
    # under the full suite an earlier module can leave crypto_loop bound to
    # a different base_loop module object than `import base_loop` returns,
    # so patching the latter silently let the REAL get_macro_regime run
    # (gate engine-r4: VIX fetched, no flatten, test failed; green alone).
    g = crypto_loop.CryptoLoop._update_macro_regime.__globals__
    monkeypatch.setitem(g, 'get_macro_regime', lambda api, at: regime)
    monkeypatch.setitem(g, 'emergency_flatten',
                        lambda api, symbols=None: ['BTCUSD'])
    trades = []
    monkeypatch.setitem(g, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    inst.get_quote = lambda s: None
    errs = []
    monkeypatch.setattr(g['logger'], 'error',
                        lambda m, *a, **k: errs.append(m % a))
    inst.positions = {'BTC/USD': 1, 'ETH/USD': 2}
    inst._update_macro_regime()
    assert set(inst.positions) == {'BTC/USD'}
    assert trades == [] and len(errs) == 1 and 'ETH/USD' in errs[0]


@pytest.mark.parametrize('attrib', [False, True])
def test_stablecoin_flatten_server_fill_attribution(replay, monkeypatch,
                                                    attrib):
    loop, broker, calls, marks = _stable_run(
        replay, monkeypatch, attrib, _gap_tape(), skip_stops=True)
    rec = replay.rec
    stop_fills = {e['symbol']: e['fill_price'] for e in broker.events_of('fill')
                  if e['side'] == 'sell' and e['type'] == 'stop_limit'}
    assert len(stop_fills) == 6 and loop.positions == {}
    stop_ids = {loop_stop_id(broker, s) for s in UNIVERSE}
    probed = [c[1] for c in calls if c[0] == 'get_order' and c[1] in stop_ids]
    if not attrib:
        assert probed == []
        assert [k for a, k in rec.trades] == [
            {'exit_reason': 'stablecoin_flatten', 'estimated': True}] * 6
        assert not rec.rows('sell') and loop.hard_stop_lockout == {}
    else:
        assert sorted(probed) == sorted(stop_ids)
        sells = {r['symbol']: r for r in rec.rows('sell')}
        assert set(sells) == set(UNIVERSE)
        for s, r in sells.items():
            assert r['exit_reason'] == 'server_stop'
            assert r['detect_source'] == 'stablecoin_flatten'
            assert r['fill_price'] == pytest.approx(stop_fills[s])
        assert not [k for a, k in rec.trades
                    if k.get('exit_reason') == 'stablecoin_flatten']
        assert set(loop.hard_stop_lockout) == set(UNIVERSE)


# ===========================================================================
# R4-b — the shared helper's site labels (breaker text unchanged)
# ===========================================================================

class _Log:
    def __init__(self):
        self.lines = []

    def _rec(self, lvl):
        return lambda m, *a, **k: self.lines.append((lvl, m % a if a else m))

    def __getattr__(self, lvl):
        return self._rec(lvl)


@pytest.mark.parametrize('site,tag,what', [
    (None, '[CIRCUIT BREAKER]', 'breaker flatten'),
    ('breaker', '[CIRCUIT BREAKER]', 'breaker flatten'),
    ('remote_flatten', '[FLATTEN]', 'remote flatten'),
    ('stablecoin_flatten', '[CONTAGION]', 'stablecoin flatten'),
])
def test_server_fill_helper_site_labels(monkeypatch, site, tag, what):
    from test_engine_r3_o3_quote import _book, _stub_loop
    log = _Log()
    monkeypatch.setattr(base_loop, 'logger', log)
    rows = []
    monkeypatch.setattr(base_loop, 'log_decision', lambda r: rows.append(r))
    monkeypatch.setattr(base_loop, 'record_trade', lambda *a, **k: None)
    positions, orders, quotes = _book(base_loop)
    loop, ev = _stub_loop(base_loop, positions, orders, quotes)
    kw = {} if site is None else {'site': site}
    f = loop._breaker_record_server_fill
    assert f('DDD/USD', positions['DDD/USD'], **kw) is False
    assert f('AAA/USD', positions['AAA/USD'], **kw) is True
    assert log.lines[0] == ('debug', f'{tag} DDD/USD: stop s-raise status '
                                     'check failed (HTTP 500) — estimated exit row')
    assert log.lines[1] == ('info', '[STOP-FILL] AAA/USD: resting stop filled '
                                    f'at $93.0 before the {what} — journaled '
                                    'as server_stop')
    assert rows[-1]['detect_source'] == (site or 'breaker')


def test_maker_rung_fill_journals_the_rung_order_id(replay, monkeypatch):
    """Maker ladder: the buy row's order_id is the rung place_maker_buy
    returns (a zero-spread tape makes the first bid-join marketable, so
    rung 1 fills and the ladder returns it with tactic 'maker')."""
    monkeypatch.setattr(strategy_config, 'MAKER_ENTRIES_ENABLED', True)
    snap = copy.deepcopy(SNAPSHOT)
    snap['account'].update(cash='30000', non_marginable_buying_power='30000',
                           buying_power='30000', last_equity='30000',
                           equity='30000', portfolio_value='30000')
    snap['positions'] = []
    mids = {'BTC/USD': 84000.0, 'ETH/USD': 2700.0, 'XRP/USD': 1.5,
            'SOL/USD': 120.0, 'DOGE/USD': 0.125, 'LINK/USD': 14.0}
    loop, broker = replay.build_crypto(
        snapshot=snap, tape={s: [m] * 6 for s, m in mids.items()},
        preds_fn=lambda c: {s: (2.0 if s in ('SOL/USD', 'XRP/USD') else 0.0)
                            for s in UNIVERSE},
        clock=FakeClock(), spread_bps=0.0)
    seen = _record_quotes(loop)
    replay.drive(loop, broker, 3)
    buys = _assert_buy_row_joins(replay.rec, broker, seen)
    for r in buys:
        assert r['entry_tactic'] == 'maker'
        assert broker.orders[r['order_id']]['client_order_id'].startswith('maker-')
        assert broker.orders[r['order_id']]['type'] == 'limit'
        assert r['decision_bid'] == r['decision_ask'] == mids[r['symbol']]
