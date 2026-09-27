"""ENGINE-R2 W9 (2026-09 Jetson campaign) — LLM veto strikes + halt.

1. D13 scope A (class-A fix): a symbol SENT to the LLM this analysis but
   omitted from a successful-but-partial response reads as s=0.5 (no veto)
   and must lose its veto strike — otherwise veto -> omitted -> veto
   liquidates on two NON-consecutive vetoes, contradicting the documented
   "two consecutive vetoing analyses" rule. Scope B (a held symbol NOT sent
   because stocks analyse only top-N) is an owner decision and must keep
   its strike — pinned here as unchanged.
2. strategy_config.HALT_CANCELS_WORKING_BUYS (default OFF): OFF = a halt
   blocks new entries only and leaves working orders alone (byte-identical);
   ON = the first halted cycle cancels this book's open BUY orders once per
   halt epoch, never SELL/stop orders, never the other book's orders.
"""
import ast
import sys
import textwrap
import time as _time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tests'))

BASE_SRC = (REPO / 'base_loop.py').read_text()


# =====================================================================
# 1. D13 scope A — stale veto strike for symbols omitted from a response
# =====================================================================

def _method(name, glb):
    for node in ast.walk(ast.parse(BASE_SRC)):
        if isinstance(node, ast.ClassDef) and node.name == 'BaseTradingLoop':
            for it in node.body:
                if isinstance(it, ast.FunctionDef) and it.name == name:
                    seg = textwrap.dedent(ast.get_source_segment(BASE_SRC, it))
                    ns = dict(glb)
                    exec(compile(seg, f'<{name}>', 'exec'), ns)
                    return ns[name]
    raise AssertionError(f'BaseTradingLoop.{name} not found')


class _Log:
    def __init__(self):
        self.lines = []

    def _r(self, lvl, m, *a):
        self.lines.append((lvl, m % a if a else m))

    def info(self, m, *a):
        self._r('info', m, *a)

    def warning(self, m, *a):
        self._r('warning', m, *a)

    def debug(self, m, *a):
        self._r('debug', m, *a)

    def error(self, m, *a):
        self._r('error', m, *a)


V, OK = {'s': 0.05}, {'s': 0.60}


def _run(monkeypatch, steps):
    """steps: list of (candidate symbols, LLM response). Runs one analysis
    then the veto-sell pass per step; returns (strike trace, sells)."""
    monkeypatch.setitem(sys.modules, 'sentiment', types.SimpleNamespace(
        get_fear_greed=lambda: {'value': 50}))
    monkeypatch.setitem(sys.modules, 'llm_analyst', types.SimpleNamespace(
        get_last_analysis_meta=lambda: {}))
    responses = iter([r for _, r in steps])
    glb = {'load_llm_config': lambda: {'enabled': True},
           'analyze_trades': lambda cands, *a, **k: next(responses),
           'log_decision': lambda row: None, 'logger': _Log(),
           'LLM_VETO_THRESHOLD': 0.15,
           'time': SimpleNamespace(time=_time.time, sleep=lambda s: None),
           'cooldown_ok': lambda *a: True,
           'datetime': __import__('datetime')}
    run = _method('_run_llm_analysis', glb)
    veto = _method('_execute_llm_veto_sells', glb)
    sells = []
    cands = {'now': []}
    me = SimpleNamespace(
        _expire_llm_scores=lambda: None, _llm_backoff_until=0.0,
        _llm_fail_count=0, _llm_scores_ts=None, _last_llm_time=0.0,
        LLM_INTERVAL_SEC=600, LLM_SCORE_TTL_SEC=7200,
        _check_llm_staleness=lambda: None,
        _build_llm_candidates=lambda p: [{'symbol': s} for s in cands['now']],
        positions={'AAA': SimpleNamespace(qty=10, stop_order_id=None,
                                          to_dict=lambda: {})},
        _equity=1e5, config={}, get_asset_type=lambda: 'crypto',
        llm_scores={}, _veto_strikes={}, _last_llm_symbols=set(),
        last_trade_time={}, COOLDOWN_MINUTES=60, get_quote=lambda s: None,
        place_sell_order=lambda s, q, quote: (
            sells.append(s) or SimpleNamespace(filled_avg_price=1.0)),
        _record_confirmed_exit=lambda *a, **k: None)
    trace = []
    for syms, _ in steps:
        cands['now'] = syms
        me._last_llm_time = 0.0
        me._llm_backoff_until = 0.0
        run(me, {})
        veto(me)
        trace.append(me._veto_strikes.get('AAA', 0))
    return trace, sells


BOTH = ['AAA', 'BBB']


def test_d13_veto_omitted_veto_does_not_liquidate(monkeypatch):
    # AAA sent every time; the middle response omits it (partial parse).
    trace, sells = _run(monkeypatch, [(BOTH, {'AAA': V, 'BBB': OK}),
                                      (BOTH, {'BBB': OK}),
                                      (BOTH, {'AAA': V, 'BBB': OK})])
    assert trace == [1, 0, 1]
    assert sells == []


def test_d13_consecutive_vetoes_still_liquidate(monkeypatch):
    trace, sells = _run(monkeypatch, [(BOTH, {'AAA': V, 'BBB': OK}),
                                      (BOTH, {'AAA': V, 'BBB': OK})])
    assert trace == [1, 2]
    assert sells == ['AAA']


def test_d13_outage_keeps_strike_by_design(monkeypatch):
    # Empty response = provider failure (c26 D14): scores AND strikes kept.
    trace, sells = _run(monkeypatch, [(BOTH, {'AAA': V, 'BBB': OK}),
                                      (BOTH, {}),
                                      (BOTH, {'AAA': V, 'BBB': OK})])
    assert trace == [1, 1, 2]
    assert sells == ['AAA']


def test_d13_scope_b_unsent_symbol_keeps_strike(monkeypatch):
    # Scope B (owner decision, NOT changed): AAA leaves the analysed set
    # (stock top-N) — it was never sent, so its strike is kept.
    trace, sells = _run(monkeypatch, [(BOTH, {'AAA': V, 'BBB': OK}),
                                      (['BBB'], {'BBB': OK}),
                                      (BOTH, {'AAA': V, 'BBB': OK})])
    assert trace == [1, 1, 2]
    assert sells == ['AAA']


# =====================================================================
# 2. HALT_CANCELS_WORKING_BUYS
# =====================================================================

base_loop = pytest.importorskip('base_loop')
crypto_loop = pytest.importorskip('crypto_loop')
stock_loop = pytest.importorskip('stock_loop')
import notify                                   # noqa: E402
import macro_calendar                           # noqa: E402
import strategy_config                          # noqa: E402
from fake_alpaca_broker import FakeAlpacaBroker  # noqa: E402

SNAPSHOT = {
    'account': {'cash': '100000', 'last_equity': '100000',
                'buying_power': '100000'},
    'positions': [{'symbol': 'BTCUSD', 'qty': '1', 'avg_entry_price': '60000',
                   'cost_basis': '60000', 'current_price': '60000',
                   'asset_class': 'crypto'}],
    'open_orders': [],
}
TAPE = {'BTC/USD': [60000.0], 'ETH/USD': [3000.0], 'SOL/USD': [150.0],
        'AAPL': [200.0]}


@pytest.fixture
def world(monkeypatch):
    halt = {'on': False}
    monkeypatch.setattr(notify, 'halt_active', lambda: halt['on'])
    monkeypatch.setattr(macro_calendar, 'macro_standdown',
                        lambda *a, **k: (False, None))
    monkeypatch.setattr(macro_calendar, 'calendar_exhausted',
                        lambda *a, **k: False)
    b = FakeAlpacaBroker(SNAPSHOT, tape=TAPE)
    ids = {
        'eth_buy': b.submit_order(symbol='ETH/USD', qty=1, side='buy',
                                  type='limit', limit_price=2000.0,
                                  time_in_force='gtc',
                                  client_order_id='maker-aaa').id,
        'btc_stop': b.submit_order(symbol='BTC/USD', qty=1, side='sell',
                                   type='stop_limit', stop_price=50000.0,
                                   limit_price=49900.0, time_in_force='gtc',
                                   client_order_id='cstop-aaa').id,
        'aapl_buy': b.submit_order(symbol='AAPL', qty=10, side='buy',
                                   type='limit', limit_price=150.0,
                                   time_in_force='day').id,
    }
    return SimpleNamespace(broker=b, halt=halt, ids=ids)


def _crypto(broker):
    inst = object.__new__(crypto_loop.CryptoLoop)
    inst.api = broker
    inst.cycle = 1
    return inst


def _status(b, key, ids):
    return b.orders[ids[key]]['status']


def _flag(monkeypatch, value):
    monkeypatch.setattr(strategy_config, 'HALT_CANCELS_WORKING_BUYS', value)


def test_halt_flag_default_off():
    assert strategy_config.HALT_CANCELS_WORKING_BUYS is False


def test_off_halt_blocks_entries_and_leaves_orders(world):
    b, ids = world.broker, world.ids
    inst = _crypto(b)
    world.halt['on'] = True
    n_calls = len(b.calls)
    for _ in range(3):
        assert inst._entries_allowed() is False
        assert inst._entries_block_info == ('halt', 'trading_halt.flag', None)
    assert all(_status(b, k, ids) == 'new' for k in ids)
    # OFF path makes no broker calls at all (byte-identical to legacy)
    assert b.calls[n_calls:] == []
    world.halt['on'] = False
    assert inst._entries_allowed() is True
    assert b.calls[n_calls:] == []


def test_on_cancels_book_buys_once_and_rearms(world, monkeypatch):
    _flag(monkeypatch, True)
    b, ids = world.broker, world.ids
    inst = _crypto(b)
    assert inst._entries_allowed() is True           # not halted: no-op
    assert all(_status(b, k, ids) == 'new' for k in ids)

    world.halt['on'] = True
    assert inst._entries_allowed() is False
    assert _status(b, 'eth_buy', ids) == 'canceled'
    assert _status(b, 'btc_stop', ids) == 'new'      # exits never touched
    assert _status(b, 'aapl_buy', ids) == 'new'      # other book untouched
    cancels = [c for c in b.calls if c[0] == 'cancel_order']
    assert cancels == [('cancel_order', ids['eth_buy'], 0)]

    # Same halt epoch: no second listing / cancel.
    late = b.submit_order(symbol='SOL/USD', qty=1, side='buy', type='limit',
                          limit_price=1.0, time_in_force='gtc').id
    n_calls = len(b.calls)
    assert inst._entries_allowed() is False
    assert b.calls[n_calls:] == []
    assert b.orders[late]['status'] == 'new'

    # Un-halt re-arms; the next halt cancels again.
    world.halt['on'] = False
    assert inst._entries_allowed() is True
    world.halt['on'] = True
    assert inst._entries_allowed() is False
    assert b.orders[late]['status'] == 'canceled'
    assert _status(b, 'btc_stop', ids) == 'new'


def test_on_stock_book_cancels_only_its_buys(world, monkeypatch):
    _flag(monkeypatch, True)
    monkeypatch.setattr(stock_loop.StockLoop, 'get_symbol_universe',
                        lambda self: ['AAPL', 'MSFT'])
    b, ids = world.broker, world.ids
    inst = object.__new__(stock_loop.StockLoop)
    inst.api = b
    inst.cycle = 1
    world.halt['on'] = True
    assert inst._entries_allowed() is False
    assert _status(b, 'aapl_buy', ids) == 'canceled'
    assert _status(b, 'eth_buy', ids) == 'new'
    assert _status(b, 'btc_stop', ids) == 'new'


class _StubApi:
    def __init__(self, orders, fail_ids=(), list_raises=False):
        self.orders = orders
        self.fail_ids = set(fail_ids)
        self.list_raises = list_raises
        self.cancelled = []

    def list_orders(self, status='open', limit=None, symbols=None):
        if self.list_raises:
            raise RuntimeError('api down')
        return list(self.orders)

    def cancel_order(self, oid):
        if oid in self.fail_ids:
            raise RuntimeError('boom')
        self.cancelled.append(oid)


def _o(oid, symbol, side, type_):
    return SimpleNamespace(id=oid, symbol=symbol, side=side, type=type_)


def test_on_skips_stop_types_and_retries_failed_cancel(monkeypatch, world):
    _flag(monkeypatch, True)
    api = _StubApi([_o('b1', 'ETH/USD', 'buy', 'limit'),
                    _o('b2', 'ETHUSD', 'buy', 'market'),
                    _o('s1', 'ETH/USD', 'buy', 'stop_limit'),
                    _o('x1', 'ETH/USD', 'sell', 'limit')], fail_ids={'b2'})
    inst = _crypto(api)
    world.halt['on'] = True
    assert inst._entries_allowed() is False
    assert api.cancelled == ['b1']                   # b2 raised, s1/x1 skipped
    api.fail_ids.clear()
    assert inst._entries_allowed() is False          # epoch not done: retry
    assert api.cancelled == ['b1', 'b1', 'b2']
    assert inst._entries_allowed() is False          # done now
    assert api.cancelled == ['b1', 'b1', 'b2']


def test_on_cancel_path_failure_never_unblocks_halt(monkeypatch, world):
    _flag(monkeypatch, True)
    world.halt['on'] = True
    inst = _crypto(_StubApi([], list_raises=True))
    assert inst._entries_allowed() is False
    assert inst._entries_allowed() is False

    def boom(self, halted):
        raise RuntimeError('unexpected')
    monkeypatch.setattr(crypto_loop.CryptoLoop, '_halt_cancel_working_buys',
                        boom, raising=False)
    assert inst._entries_allowed() is False
