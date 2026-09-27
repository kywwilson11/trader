"""ENGINE round-7 W20: broker failure modes H1-H5 (research_engine.md § R6
scout Q3, owner item #24) driven through the REAL CryptoLoop / StockLoop
against tests/fake_alpaca_broker.py and its r7 fault switches.

* H5 (class D, LANDED): a NULL ``avg_entry_price`` (not the string "0") is
  read exactly like the paper zero-basis quirk's "0" at all three sites —
  order_utils.reconstruct_positions._entry, base_loop._place_and_track_buy
  and alpaca_compat._shim_position — through order_utils._basis_or_zero.
  Byte-identical for every input float() accepts.
* H1 / H2 / H4 (owner rulings, NOT fixed): each has a CURRENT-BEHAVIOUR pin
  citing the production line, plus a strict xfail stating the intended
  invariant. When the owner rules and the fix lands, the xfail XPASSes
  (strict -> fails) and the pin must be retired together with it.
* H3: with R1 + H5 a filled buy whose position reports a NULL basis is
  entered at the order's fill price (PASSES — it is the fix). The W19
  variant (order filled_avg_price ALSO null) is pinned + strict xfail.
* W16 F2 harness gaps: ns quote timestamps, canceled_at, the live 03:09
  startup configuration (no model + halt flag) and combined-mode startup.

Everything network-touching is stubbed at its seam (sockets blocked); every
file the loops write is redirected to tmp_path (position state, lockout,
journal, prediction cache, halt flag).
"""

import copy
import datetime as _dt
import logging
import os
import socket
import sys
import time
import warnings
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import alpaca_compat                               # noqa: E402
import base_loop                                   # noqa: E402
import crypto_loop                                 # noqa: E402
import stock_loop                                  # noqa: E402
import order_utils                                 # noqa: E402
from types_mod import MacroRegime                  # noqa: E402
from strategy_config import CRYPTO_POLICY          # noqa: E402
from fake_alpaca_broker import (FakeAlpacaBroker, FakeAPIError,  # noqa: E402
                                apply_live_modes)

P = CRYPTO_POLICY

# Real paper-account snapshot (generals/engine/account_snapshot.json,
# read-only GET 2026-09-27 00:20) — same values as test_engine_r1.
SNAPSHOT = {
    'account': {
        'status': 'ACTIVE', 'crypto_status': 'ACTIVE', 'currency': 'USD',
        'buying_power': '374.52', 'regt_buying_power': '187.26',
        'effective_buying_power': '374.52',
        'non_marginable_buying_power': '93.63', 'cash': '93.63',
        'portfolio_value': '121930.35', 'equity': '121930.35',
        'last_equity': '123095.9849438596608',
        'long_market_value': '121836.72', 'multiplier': '1',
        'shorting_enabled': False, 'trading_blocked': False,
    },
    'positions': [
        {'symbol': 'BTCUSD', 'qty': '0.249291739', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '84364.6', 'asset_class': 'crypto'},
        {'symbol': 'DOGEUSD', 'qty': '114913.12732127', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '0.0962855', 'asset_class': 'crypto'},
        {'symbol': 'ETHUSD', 'qty': '7.904614454', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '2692.94', 'asset_class': 'crypto'},
        {'symbol': 'LINKUSD', 'qty': '1576.529633666', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '14.09', 'asset_class': 'crypto'},
        {'symbol': 'SOLUSD', 'qty': '212.082911929', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '120.55', 'asset_class': 'crypto'},
        {'symbol': 'XRPUSD', 'qty': '13641.540903233', 'avg_entry_price': '0',
         'cost_basis': '0', 'current_price': '1.51554', 'asset_class': 'crypto'},
    ],
    'open_orders': [],
}
CP = {p['symbol'][:-3] + '/USD': float(p['current_price'])
      for p in SNAPSHOT['positions']}
QTY = {p['symbol'][:-3] + '/USD': float(p['qty'])
       for p in SNAPSHOT['positions']}
UNIVERSE = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD', 'LINK/USD']
PRICES = dict(CP, **{'AVAX/USD': 20.0})
STOCKS = {'AAA': 50.0, 'BBB': 120.0, 'CCC': 80.0}
SPREAD_BPS = 10.0
ATR_FRAC = 0.01
THRESHOLD = 0.15
FLAT = {s: 0.0 for s in UNIVERSE}


def _rp(p):
    return crypto_loop.CryptoLoop._round_px(p)


def _null_snapshot():
    snap = copy.deepcopy(SNAPSHOT)
    for p in snap['positions']:
        p['avg_entry_price'] = None
        p['cost_basis'] = None
    return snap


class Rec:
    def __init__(self):
        self.journal, self.trades, self.risk, self.notes = [], [], [], []
        self.errors, self.net = [], []

    def rows(self, action):
        return [r for r in self.journal if r.get('action') == action]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """build(...) -> (loop, broker); drive(loop, broker, n, before_cycle)."""
    rec = Rec()
    for k in ('TRADER_ORDER_STREAM', 'TRADER_USE_ALPACA_PY'):
        monkeypatch.delenv(k, raising=False)
    import random as _random
    import time as _time
    monkeypatch.setattr(_time, 'sleep', lambda s=0: None)
    monkeypatch.setattr(_random, 'uniform', lambda a, b: 0.0)

    def _no_net(*a, **k):
        rec.net.append(repr(a)[:120])
        raise OSError('network blocked in the r7 harness')
    monkeypatch.setattr(socket.socket, 'connect', _no_net)
    monkeypatch.setattr(socket, 'create_connection', _no_net)
    monkeypatch.setattr(socket, 'getaddrinfo', _no_net)

    import notify, macro_calendar, trade_journal, portfolio, market_data
    import risk_budget, funding, volatility, meta_label, shadow
    import events_calendar, edgar_events, macro_indicators, panel_ranks
    import predict_now
    monkeypatch.setattr(notify, 'flatten_requested', lambda: False)
    # halt flag: the REAL halt_active reading a tmp flag (absent unless a
    # test calls apply_live_modes(halt=True)) — the repo's own flag can
    # never leak in.
    monkeypatch.setattr(notify, '_HALT_FLAG', str(tmp_path / 'trading_halt.flag'))
    monkeypatch.setattr(notify, 'set_halt', lambda *a, **k: None)
    monkeypatch.setattr(notify, 'notify', lambda *a, **k: rec.notes.append(a))
    monkeypatch.setattr(notify, 'ping_heartbeat', lambda *a, **k: None)
    monkeypatch.setattr(macro_calendar, 'macro_standdown', lambda: (False, ''))
    monkeypatch.setattr(macro_calendar, 'calendar_exhausted', lambda: False)
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', tmp_path)
    monkeypatch.setattr(trade_journal, 'log_decision',
                        lambda row: rec.journal.append(dict(row)))
    monkeypatch.setattr(base_loop, 'log_decision',
                        lambda row: rec.journal.append(dict(row)))
    _rt = lambda *a, **k: rec.trades.append((a, k))     # noqa: E731
    monkeypatch.setattr(base_loop, 'record_trade', _rt)
    monkeypatch.setattr(stock_loop, 'record_trade', _rt)
    monkeypatch.setattr(base_loop, 'compute_kelly_fraction', lambda **k: None)
    monkeypatch.setattr(base_loop, 'get_macro_regime', lambda api, at: MacroRegime(
        stress_level=0.0, vix=18.0, cape=None, regime_label='neutral'))
    monkeypatch.setattr(base_loop, 'get_gpu_temp', lambda: None)
    monkeypatch.setattr(base_loop, 'sentiment_gate', lambda s, a: (1.0, []))
    monkeypatch.setattr(stock_loop, 'sentiment_gate', lambda s, a: (1.0, []))
    monkeypatch.setattr(stock_loop, 'get_market_sentiment', lambda: None)
    monkeypatch.setattr(base_loop, 'load_llm_config', lambda: {'enabled': False})
    monkeypatch.setattr(base_loop, '_flatten_flag_path',
                        lambda book: str(tmp_path / f'flatten_{book}.flag'))
    _atr = lambda api, s, asset_type=None: (                     # noqa: E731
        PRICES.get(s) or STOCKS.get(s) or 100.0) * ATR_FRAC
    monkeypatch.setattr(market_data, 'get_live_atr', _atr)
    monkeypatch.setattr(stock_loop, 'get_live_atr', _atr)
    for fn in ('fetch_bars_alpaca', 'fetch_stock_bars_alpaca',
               'fetch_spy_bars_alpaca'):
        monkeypatch.setattr(market_data, fn, lambda *a, **k: None)
    monkeypatch.setattr(crypto_loop, 'fetch_bars_alpaca', lambda *a, **k: None)
    monkeypatch.setattr(crypto_loop, 'get_fear_greed', lambda *a, **k: None)
    monkeypatch.setattr(crypto_loop, '_PRED_CACHE_FILE',
                        tmp_path / 'crypto_predictions.json')
    monkeypatch.setattr(stock_loop, 'fetch_spy_bars_alpaca', lambda *a, **k: None)
    monkeypatch.setattr(portfolio, 'get_correlation_matrix_cached',
                        lambda *a, **k: {})
    monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                        lambda *a, **k: 1.0)
    monkeypatch.setattr(risk_budget, 'record_book_risk_and_report',
                        lambda book, risks, rho, **k: rec.risk.append(list(risks)))
    monkeypatch.setattr(funding, 'funding_tilt', lambda s: 1.0)
    monkeypatch.setattr(volatility, 'get_crypto_rv_mult',
                        lambda *a, **k: (1.0, 'normal', None))
    monkeypatch.setattr(volatility, 'update_crypto_rv_state', lambda *a, **k: None)
    monkeypatch.setattr(meta_label, 'meta_probability_live', lambda *a, **k: None)
    monkeypatch.setattr(shadow, 'maybe_log_shadow', lambda *a, **k: None)
    monkeypatch.setattr(events_calendar, 'earnings_within_days', lambda *a, **k: False)
    monkeypatch.setattr(events_calendar, 'calendar_available', lambda: True)
    monkeypatch.setattr(events_calendar, 'blocks_overnight_hold', lambda s: False)
    monkeypatch.setattr(events_calendar, 'reported_recently', lambda *a, **k: False)
    monkeypatch.setattr(edgar_events, 'entry_blocked', lambda s: (False, ''))
    monkeypatch.setattr(macro_indicators, 'get_spy_trend_ok', lambda api: True)
    monkeypatch.setattr(panel_ranks, 'compute_live_panel_ranks',
                        lambda *a, **k: {})
    monkeypatch.setattr(predict_now, 'set_panel_features', lambda *a, **k: None)
    monkeypatch.setattr(base_loop.BaseTradingLoop, '_load_hard_stop_lockout',
                        lambda self: None)

    class H:
        pass
    h = H()
    h.rec, h.tmp, h.mp = rec, tmp_path, monkeypatch

    def _common(loop, book, stub_model=True):
        loop._lockout_file = str(tmp_path / f'{book}_hard_stop_lockout.json')
        state = tmp_path / f'{book}_position_state.json'
        loop._position_state_file = lambda: str(state)
        if stub_model:
            loop._hot_reload_check = lambda: None

            def _load_models():
                loop.model = object()
                loop.config = {'trade_threshold': THRESHOLD, 'forward_bars': 24}
                loop.trade_threshold = THRESHOLD
            loop._load_models = _load_models
        return loop

    def build(snapshot=SNAPSHOT, tape=None, preds=FLAT, universe=None,
              stub_model=True, **bkw):
        """Crypto loop over a legacy mid tape; tape: {sym: [mult, ...]}
        (multipliers of PRICES) — symbols absent from it stay flat."""
        tape = {s: [PRICES[s] * m for m in ms] for s, ms in (tape or {}).items()}
        broker = FakeAlpacaBroker(snapshot, tape=tape, spread_bps=SPREAD_BPS,
                                  **bkw)
        for s in PRICES:
            broker.tape.setdefault(s.replace('/', ''), [PRICES[s]])
        monkeypatch.setattr(base_loop, 'get_api', lambda: broker)
        loop = crypto_loop.CryptoLoop()
        _common(loop, 'crypto', stub_model)
        if stub_model:
            loop._get_predictions = lambda bench: (dict(preds), {})
        if universe is not None:
            loop.get_symbol_universe = lambda: list(universe)
        return loop, broker

    def drive(loop, broker, n_cycles, before_cycle=None):
        """Real run(): startup, then n cycles; the broker ticks one step
        BEFORE each cycle (startup = step 0, cycle k = step k). Returns the
        broker event index at the end of startup and of each cycle."""
        marks = []
        real = type(loop)._run_one_cycle

        def one():
            if not marks:
                marks.append(len(broker.events))
            if n_cycles == 0:
                raise KeyboardInterrupt
            broker.tick()
            if before_cycle:
                before_cycle(loop, broker)
            try:
                real(loop)
            except Exception as e:
                rec.errors.append(repr(e))
                raise KeyboardInterrupt
            marks.append(len(broker.events))
            if loop.cycle >= n_cycles:
                raise KeyboardInterrupt
        loop._run_one_cycle = one
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        with pytest.raises(KeyboardInterrupt):
            loop.run()
        assert rec.errors == []
        assert rec.net == [], f'network attempted: {rec.net[:3]}'
        return marks

    h.build, h.drive, h.common = build, drive, _common
    yield h


def _ev(broker):
    """normalized_events() minus the wall-clock 't' stamp (each FakeClock
    starts at real now, so two runs differ only there)."""
    return [{k: v for k, v in e.items() if k != 't'}
            for e in broker.normalized_events()]


def _stops(events):
    return {e['symbol']: e for e in events
            if e['kind'] == 'submit' and e['type'] == 'stop_limit'}


def _buy_direct(h, loop, broker, symbol, notional):
    """Drive the REAL _place_and_track_buy with the entry order reduced to
    one market buy on the fake (fills at the ask) — isolates the
    post-fill tracking/protection path from the maker ladder."""
    loop._execute_entry_order = lambda s, n, q: (broker.submit_order(
        symbol=s, notional=n, side='buy', type='market',
        time_in_force='gtc', client_order_id='t-buy'), 'market')
    q = loop.get_quote(symbol)
    q['_fetched_ts'] = time.time()
    loop._place_and_track_buy(symbol, notional, 2.0, q, 1.0, [], None, 1.0, '')


# ===========================================================================
# Task A — H5: a NULL avg_entry_price is read as the zero-basis "0"
# ===========================================================================

NUMERIC_GRID = ['0', 0, 0.0, '21031.39', 21031.39, 1e-9, '1e-9', '84364.6',
                -1.5, '-0.0', 7, Decimal('3.14159'), '  12.5 ', float('inf'),
                '-inf', 'nan', float('nan'), '1_000.5', True]


@pytest.mark.parametrize('x', NUMERIC_GRID, ids=repr)
def test_basis_or_zero_byte_identical_to_float_on_numeric_inputs(x):
    got = order_utils._basis_or_zero(x)
    assert type(got) is float
    assert repr(got) == repr(float(x))          # repr: nan/-0.0 exact


@pytest.mark.parametrize('x', [None, '', 'null', 'abc', object(), [], {}],
                         ids=repr)
def test_basis_or_zero_missing_or_garbage_is_zero_and_logged(x, caplog):
    with caplog.at_level(logging.DEBUG, logger='order_utils'):
        assert order_utils._basis_or_zero(x) == 0.0
    msgs = [r.getMessage() for r in caplog.records if '[BASIS]' in r.getMessage()]
    assert len(msgs) == 1 and type(x).__name__ in msgs[0]


class _ListApi:
    def __init__(self, positions, listing=True):
        self._p = positions
        if not listing:
            self.list_positions = None

    def list_positions(self):
        return list(self._p)

    def get_position(self, sym):
        for p in self._p:
            if p.symbol == sym:
                return p
        raise FakeAPIError('position does not exist')


def _pos(avg, sym='BTCUSD', qty='0.25', cp='84364.6'):
    return SimpleNamespace(symbol=sym, qty=qty, avg_entry_price=avg,
                           current_price=cp)


@pytest.mark.parametrize('listing', [True, False], ids=['list', 'probe'])
def test_reconstruct_keeps_null_basis_exactly_like_zero(listing):
    """H5 at order_utils.reconstruct_positions._entry: a null basis was a
    TypeError -> 'bad position payload' -> DROPPED (list path) / swallowed
    as a probe failure (probe path). Now identical to the '0' case."""
    want = order_utils.reconstruct_positions(
        _ListApi([_pos('0')], listing), ['BTC/USD'])
    assert want == {'BTC/USD': {'qty': 0.25, 'entry_price': 0.0,
                                'high_water_mark': 84364.6}}
    for raw in (None, '', 'garbage'):
        api = _ListApi([_pos(raw)], listing)
        assert order_utils.reconstruct_positions(api, ['BTC/USD']) == want


def test_reconstruct_qty_none_is_still_a_bad_payload(caplog):
    """Basis fields only: a null QTY keeps failing as today."""
    with caplog.at_level(logging.WARNING, logger='order_utils'):
        out = order_utils.reconstruct_positions(
            _ListApi([_pos('0', qty=None)]), ['BTC/USD'])
    assert out == {}
    assert any('bad position payload' in r.getMessage() for r in caplog.records)


def test_h5_startup_null_basis_book_identical_to_zero_basis_book(env, caplog):
    """The inherited book reported with NULL bases: all six are kept, each
    with base_loop's 'cost basis unknown' WARNING, and the startup stops
    plus the cycle-1 churn are byte-identical to the '0' book (the stop is
    the HWM-anchored 6 % fallback, then the 5 % trail)."""
    runs = []
    for snap in (SNAPSHOT, _null_snapshot()):
        env.rec.__init__()
        caplog.clear()
        loop, broker = env.build(snapshot=snap)
        with caplog.at_level(logging.WARNING, logger='base_loop'):
            marks = env.drive(loop, broker, 1)
        warn = [r.getMessage() for r in caplog.records
                if 'cost basis unknown' in r.getMessage()]
        runs.append((_ev(broker), marks,
                     {s: (p.qty, p.entry_price, p.high_water_mark)
                      for s, p in loop.positions.items()}, len(warn)))
    assert runs[0] == runs[1]
    events, marks, positions, n_warn = runs[1]
    assert set(positions) == set(UNIVERSE) and n_warn == 6
    st = _stops(events[:marks[0]])
    for s in UNIVERSE:
        assert positions[s][1] == 0.0
        assert st[s]['stop_price'] == _rp(CP[s] * (1 - P['stop_fallback_pct']))


def test_h5_filled_buy_null_position_basis_uses_order_fill_price(env):
    """base_loop._place_and_track_buy: pos.avg_entry_price None + order
    filled_avg_price '100.0' -> entry 100.0 and the resting stop at
    100*(1-d) (it raised TypeError after the fill: position untracked)."""
    loop, broker = env.build(universe=UNIVERSE + ['AVAX/USD'])
    env.drive(loop, broker, 1)
    broker.positions['AVAXUSD'] = {
        'symbol': 'AVAXUSD', 'qty': 10.0, 'avg_entry_price': 0.0,
        'cost_basis': 0.0, 'asset_class': 'crypto', 'snapshot_price': 100.0}
    broker._null_basis.add('AVAXUSD')
    assert broker.get_position('AVAX/USD').avg_entry_price is None
    loop._execute_entry_order = lambda s, n, q: (SimpleNamespace(
        id='ord-x', client_order_id='t-x', status='filled', filled_qty='10',
        filled_avg_price='100.0'), 'market')
    q = loop.get_quote('AVAX/USD')
    q['_fetched_ts'] = time.time()
    loop._place_and_track_buy('AVAX/USD', 1000.0, 2.0, q, 1.0, [], None, 1.0, '')
    pos = loop.positions['AVAX/USD']
    assert pos.entry_price == 100.0 and pos.qty == 10.0
    d = min(P['stop_ceil_pct'], max(P['stop_floor_pct'],
                                    20.0 * ATR_FRAC * P['atr_stop_mult'] / 100.0))
    st = [e for e in broker.events_of('submit', 'AVAX/USD')
          if e['type'] == 'stop_limit']
    assert [e['stop_price'] for e in st] == [_rp(100.0 * (1 - d))]
    assert pos.stop_order_id == st[0]['id']
    assert [r['fill_price'] for r in env.rec.rows('buy')] == [100.0]


def test_h5_shim_position_null_basis_is_zero_not_raise():
    p = SimpleNamespace(symbol='BTCUSD', qty='0.249291739', avg_entry_price=None,
                        current_price='84364.6')
    s = alpaca_compat._shim_position(p)
    assert s.avg_entry_price == 0.0 and type(s.avg_entry_price) is float
    assert s.qty == 0.249291739 and s.current_price == 84364.6
    for raw in NUMERIC_GRID:
        s = alpaca_compat._shim_position(SimpleNamespace(
            symbol='X', qty='1', avg_entry_price=raw))
        assert repr(s.avg_entry_price) == repr(float(raw))
    with pytest.raises(TypeError):     # qty is NOT a basis field
        alpaca_compat._shim_position(SimpleNamespace(
            symbol='X', qty=None, avg_entry_price='1'))


def test_h5_shim_verify_position_returns_live_null_basis_position():
    """Through CompatREST.get_position: verify_position must see the live
    position (it returned None -> every rejected exit = false DESYNC)."""
    class _Trading:
        def get_open_position(self, sym):
            if sym != 'BTCUSD':
                raise FakeAPIError('position does not exist')
            return SimpleNamespace(symbol='BTCUSD', qty='0.25',
                                   avg_entry_price=None, current_price='84000')
    api = object.__new__(alpaca_compat.CompatREST)
    api._trading = _Trading()
    pos = order_utils.verify_position(api, 'BTC/USD')
    assert pos is not None and pos.qty == 0.25 and pos.avg_entry_price == 0.0


# ===========================================================================
# Task B — H1..H4
# ===========================================================================

# --- H1: position hidden for one cycle while an exit fires ----------------
# BTC gaps to 0.90 at step 2: the resting 5 % trail stop (0.95) TRIGGERS but
# the bid is under its 2 %-lower limit (0.931) -> it rests unfilled; the loop
# arms the trailing breach in cycle 2 and CONFIRMS it in cycle 3, the cycle
# the broker hides the position. Other names flat (dd 2.7 % < breaker 5 %).
H1_TAPE = {'BTC/USD': [1.0, 1.0, 0.90, 0.90, 0.90, 0.90, 0.90]}


def _h1_run(env, n=6):
    loop, broker = env.build(tape=H1_TAPE)
    broker.hide_position('BTC/USD', cycles=1, at_step=3)
    marks = env.drive(loop, broker, n)
    return loop, broker, marks


def test_h1_current_behaviour_hidden_position_dropped_unprotected(env):
    """CURRENT-BEHAVIOUR pin (owner item #24 H1 — do not 'fix' here).
    Cycle 3: _execute_stop_exit cancels the resting stop FIRST
    (base_loop.py:1605 cancel_orders_for_symbol), the market sell is
    rejected ('available: 0'), verify_position returns None (the position
    is hidden) and the DESYNC branch (base_loop.py:1626-1648) writes an
    estimated 'desync' row and pops the position. When the broker shows it
    again (cycles 4-6) it is untracked with NO resting stop."""
    loop, broker, marks = _h1_run(env)
    c3 = broker.events[marks[2]:marks[3]]
    btc3 = [e for e in c3 if e.get('symbol') == 'BTC/USD']
    assert [(e['kind'], e['type']) for e in btc3] == [
        ('cancel', 'stop_limit'), ('reject', 'market')]
    assert 'available: 0' in btc3[1]['reason']
    desync = [r for r in env.rec.rows('sell') if r['symbol'] == 'BTC/USD']
    assert [(r['exit_reason'], r['estimated']) for r in desync] == [('desync', True)]
    assert [k['exit_reason'] for a, k in env.rec.trades] == ['desync']
    # reappeared, still held at the broker, untracked, unprotected
    assert not broker.is_hidden('BTC/USD')
    assert 'BTCUSD' in {p.symbol for p in broker.list_positions()}
    assert float(broker.get_position('BTC/USD').qty) == pytest.approx(QTY['BTC/USD'])
    assert 'BTC/USD' not in loop.positions
    assert broker.open_orders('BTC/USD') == []
    assert not [e for e in broker.events[marks[3]:]
                if e.get('symbol') == 'BTC/USD']
    # the other five are untouched
    assert set(loop.positions) == set(UNIVERSE) - {'BTC/USD'}


@pytest.mark.xfail(strict=True, reason='owner item #24 H1: a hidden-then-'
                   'visible position must be tracked again or protected by a '
                   'resting stop within 2 cycles (today: dropped, no stop)')
def test_h1_desired_reappeared_position_tracked_or_protected(env):
    loop, broker, marks = _h1_run(env, n=6)      # visible again at step 4
    held = broker.positions.get('BTCUSD', {}).get('qty', 0.0) > 0
    assert held
    assert 'BTC/USD' in loop.positions or broker.open_orders('BTC/USD')


# --- H2: equity == cash with every position hidden ------------------------

def _h2_run(env, n=3):
    loop, broker = env.build()
    broker.equity_equals_cash(cycles=1, at_step=2)
    marks = env.drive(loop, broker, n)
    return loop, broker, marks


def test_h2_current_behaviour_breaker_clears_tracking_sells_nothing(env):
    """CURRENT-BEHAVIOUR pin (owner item #24 H2). Cycle 2: dd =
    (123,095.98 - 93.63)/123,095.98 = 99.9 % trips check_circuit_breaker
    (order_utils.py check_circuit_breaker, called base_loop.py:763);
    emergency_flatten lists NO position -> no cancel, no sell, returns [];
    base_loop.py:854 positions.clear(); six estimated 'circuit_breaker'
    trade rows; halt latched. Cycle 3 (broker restored): the six positions
    are back at the broker, untracked; their six stale stops still rest."""
    loop, broker, marks = _h2_run(env)
    trip = env.rec.rows('circuit_breaker_trip')
    assert len(trip) == 1 and trip[0]['n_positions_at_trip'] == 6
    assert trip[0]['drawdown_pct'] == pytest.approx(99.924, abs=0.01)
    assert trip[0]['equity'] == pytest.approx(93.63)
    assert broker.events[marks[1]:marks[3]] == []      # nothing sold/cancelled
    assert loop.positions == {}
    assert [(k['exit_reason'], k['estimated']) for a, k in env.rec.trades] == \
        [('circuit_breaker', True)] * 6
    assert loop._halted_until is not None and loop._buys_allowed is False
    assert {p.symbol for p in broker.list_positions()} == \
        {s.replace('/', '') for s in UNIVERSE}
    assert sorted(o['symbol'] for o in broker.open_orders()) == sorted(UNIVERSE)


@pytest.mark.xfail(strict=True, reason='owner item #24 H2: a breaker that '
                   'sold NOTHING while positions were hidden must not clear '
                   'tracking (today: base_loop.py:854 clears all six)')
def test_h2_desired_nothing_sold_keeps_tracking(env):
    loop, broker, marks = _h2_run(env)
    assert set(loop.positions) == set(UNIVERSE)


# --- H3: filled buy, position basis NULL ----------------------------------

def test_h3_fill_avg_none_entry_is_order_fill_price(env):
    """R1 + H5: a new AVAX fill whose position reports avg_entry_price
    None (fake fill_avg_none()) enters at the ORDER's fill price and rests
    its stop at fill*(1-d); no reject. (Before H5: TypeError after the fill
    at base_loop.py:3563 -> position untracked, no stop.)"""
    loop, broker = env.build(universe=UNIVERSE + ['AVAX/USD'])
    broker.cash = 10_000.0
    env.drive(loop, broker, 1)
    broker.fill_avg_none()
    _buy_direct(env, loop, broker, 'AVAX/USD', 1000.0)
    fill = [e for e in broker.events_of('fill', 'AVAX/USD') if e['side'] == 'buy']
    assert len(fill) == 1
    px = fill[0]['fill_price']
    assert broker.get_position('AVAX/USD').avg_entry_price is None
    pos = loop.positions['AVAX/USD']
    assert pos.entry_price == pytest.approx(px) and px > 0
    d = min(P['stop_ceil_pct'], max(P['stop_floor_pct'],
                                    20.0 * ATR_FRAC * P['atr_stop_mult'] / px))
    st = [e for e in broker.events_of('submit', 'AVAX/USD')
          if e['type'] == 'stop_limit']
    assert [e['stop_price'] for e in st] == [_rp(px * (1 - d))]
    assert not broker.events_of('reject', 'AVAX/USD')
    assert env.rec.rows('buy')[-1]['fill_price'] == pytest.approx(px)


def _h3b_run(env):
    loop, broker = env.build(universe=UNIVERSE + ['AVAX/USD'])
    broker.cash = 10_000.0
    env.drive(loop, broker, 1)
    broker.report_zero_basis = True                 # position avg "0"
    broker.fill_avg_none(position=False, order=True)  # order fill price null
    _buy_direct(env, loop, broker, 'AVAX/USD', 1000.0)
    return loop, broker


def test_h3b_current_behaviour_no_basis_anywhere_stop_at_zero(env):
    """CURRENT-BEHAVIOUR pin (owner item #24 H3, W19 variant): order
    filled_avg_price None AND position avg "0" -> ofp 0 (base_loop.py
    :3575), entry stays 0, the resting stop is 0*(1-d) = 0 and the broker
    rejects it (crypto_loop.py:201 stop_order_id=None); the buy row
    journals fill_price 0 with no estimated marker."""
    loop, broker = _h3b_run(env)
    pos = loop.positions['AVAX/USD']
    assert pos.entry_price == 0.0 and pos.stop_order_id is None
    rej = broker.events_of('reject', 'AVAX/USD')
    assert [(e['type'], e['reason']) for e in rej] == [
        ('stop_limit', 'limit_price must be > 0')]
    row = env.rec.rows('buy')[-1]
    assert row['fill_price'] == 0.0 and 'estimated' not in row


@pytest.mark.xfail(strict=True, reason='owner item #24 H3: with no basis '
                   'anywhere, no stop_limit may be submitted at <= 0 and the '
                   'buy row must carry an estimated/basis-unknown marker')
def test_h3b_desired_no_zero_stop_and_marked_row(env):
    loop, broker = _h3b_run(env)
    assert not [e for e in broker.events_of('submit', 'AVAX/USD')
                if e['type'] == 'stop_limit' and not e['stop_price'] > 0]
    row = env.rec.rows('buy')[-1]
    assert row.get('estimated') is True or row.get('basis_unknown') is True


# --- H4: add-on while the first lot's stop rests (J1) ---------------------

def _h4_addon(env, reserve=True):
    loop, broker = env.build(universe=UNIVERSE + ['AVAX/USD'])
    broker.cash = 10_000.0
    broker.reserve_qty_on_resting_stops = reserve
    env.drive(loop, broker, 1)
    _buy_direct(env, loop, broker, 'AVAX/USD', 1000.0)
    first = broker.open_orders('AVAX/USD')
    assert len(first) == 1 and first[0]['type'] == 'stop_limit'
    q1 = first[0]['qty']
    broker.tick()
    _buy_direct(env, loop, broker, 'AVAX/USD', 1000.0)
    return loop, broker, first[0], q1


def test_h4_current_behaviour_addon_stop_rejected_then_self_heals(env):
    """CURRENT-BEHAVIOUR pin (owner item #24 H4 / J1). The add-on's
    _after_entry_protection (crypto_loop.py:204-206) submits a full-qty
    stop WITHOUT cancelling the first lot's resting stop -> 'insufficient
    balance for AVAX' (qty_available = qty - resting stop qty);
    _place_resting_stop sets stop_order_id=None (crypto_loop.py:201), so the
    next _manage_stops cannot see a server fill of the old stop
    (base_loop.py:1418 'if pos.stop_order_id') for one cycle; that same
    cycle _maybe_update_resting_stop (crypto_loop.py:227-243) cancels the
    old stop and re-places one full-qty stop (self-heal)."""
    loop, broker, old, q1 = _h4_addon(env)
    pos = loop.positions['AVAX/USD']
    rej = broker.events_of('reject', 'AVAX/USD')
    assert len(rej) == 1 and rej[0]['type'] == 'stop_limit'
    assert rej[0]['reason'].startswith('insufficient balance for AVAX')
    assert rej[0]['qty'] == pytest.approx(pos.qty) and pos.qty > q1
    assert pos.stop_order_id is None
    live = broker.open_orders('AVAX/USD')
    assert [o['id'] for o in live] == [old['id']] and live[0]['qty'] == q1
    assert float(broker.get_position('AVAX/USD').qty_available) == \
        pytest.approx(pos.qty - q1)
    # next cycle: blind to the old stop's status, then self-heal
    n_calls = len(broker.calls)
    broker.tick()
    loop._manage_stops()
    assert ('get_order', old['id']) not in [c[:2] for c in broker.calls[n_calls:]]
    live = broker.open_orders('AVAX/USD')
    assert len(live) == 1 and live[0]['id'] != old['id']
    assert live[0]['qty'] == pytest.approx(pos.qty)
    assert pos.stop_order_id == live[0]['id']
    assert broker.orders[old['id']]['status'] == 'canceled'
    assert len(broker.events_of('reject', 'AVAX/USD')) == 1


def test_h4_control_without_reservation_the_addon_stop_is_accepted(env):
    """Counterfactual control: the reject is exactly the qty reservation
    (reserve_qty_on_resting_stops=False -> accepted; TWO stops rest)."""
    loop, broker, old, q1 = _h4_addon(env, reserve=False)
    assert not broker.events_of('reject', 'AVAX/USD')
    assert len(broker.open_orders('AVAX/USD')) == 2


@pytest.mark.xfail(strict=True, reason='owner item #24 H4 / J1: on an add-on '
                   'the old resting stop must be cancelled BEFORE '
                   '_after_entry_protection -> 0 rejects, one live stop for '
                   'the total qty, tracked, at the end of the add-on')
def test_h4_desired_addon_cancels_old_stop_first(env):
    loop, broker, old, q1 = _h4_addon(env)
    pos = loop.positions['AVAX/USD']
    assert not broker.events_of('reject', 'AVAX/USD')
    live = broker.open_orders('AVAX/USD')
    assert len(live) == 1 and live[0]['qty'] == pytest.approx(pos.qty)
    assert pos.stop_order_id == live[0]['id']


# ===========================================================================
# Task C — W16 F2 harness gaps
# ===========================================================================

def test_fake_default_quote_t_is_stdlib_datetime():
    b = FakeAlpacaBroker(SNAPSHOT, tape={'BTC/USD': [CP['BTC/USD']]})
    t = b.get_latest_crypto_quotes(['BTC/USD'])['BTC/USD'].t
    assert type(t) is _dt.datetime


def test_ns_timestamps_quote_parses_without_nanosecond_warning():
    import pandas as pd
    b = FakeAlpacaBroker(SNAPSHOT, tape={'BTC/USD': [CP['BTC/USD']]},
                         ns_timestamps=True)
    t = b.get_latest_crypto_quotes(['BTC/USD'])['BTC/USD'].t
    assert isinstance(t, pd.Timestamp) and t.nanosecond == 123
    assert t.tzinfo is not None
    with warnings.catch_warnings():
        warnings.simplefilter('error')        # the UserWarning would raise
        q = order_utils.get_crypto_quote(b, 'BTC/USD')
    assert q is not None
    assert q['quote_t'] == pytest.approx(t.timestamp(), abs=1e-6)


def test_ns_timestamps_loop_run_identical_and_warning_free(env):
    runs = []
    for ns in (False, True):
        env.rec.__init__()
        loop, broker = env.build(ns_timestamps=ns)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            env.drive(loop, broker, 2)
        assert not [x for x in w if 'nanosecond' in str(x.message).lower()]
        runs.append(_ev(broker))
    assert runs[0] == runs[1] and runs[0]


def test_cancel_records_canceled_at(env):
    loop, broker = env.build()
    marks = env.drive(loop, broker, 1)
    startup = [e['id'] for e in broker.events[:marks[0]] if e['kind'] == 'submit']
    c1 = [e for e in broker.events[marks[0]:marks[1]] if e['kind'] == 'cancel']
    assert len(c1) == 6
    for e in c1:
        o = broker.get_order(e['id'])
        assert o.canceled_at == e['t'] and o.status == 'canceled'
        assert o.canceled_at >= o.submitted_at
    assert {e['id'] for e in c1} == set(startup)
    for o in broker.open_orders():
        assert broker.get_order(o['id']).canceled_at is None
    # the event dicts themselves are unchanged (no canceled_at key)
    assert all('canceled_at' not in e for e in broker.events)


def test_live_config_no_model_and_halt_matches_flat_case_zero_buys(env, caplog):
    """The live 03:09 configuration: real _load_models on an empty model dir
    (FileNotFoundError -> fail closed) + trading_halt.flag present. Startup
    and cycles 1-2 broker events are byte-identical to the flat case with
    zero buys."""
    env.rec.__init__()
    loop, broker = env.build()
    flat = env.drive(loop, broker, 2), _ev(broker)
    paths = apply_live_modes(env.mp, env.tmp, no_model=True, halt=True)
    assert paths['halt_flag'].exists() and Path.cwd() == paths['model_dir']
    import notify
    assert notify.halt_active() is True
    env.rec.__init__()
    loop2, broker2 = env.build(stub_model=False)
    with caplog.at_level(logging.WARNING):
        marks = env.drive(loop2, broker2, 2)
    assert loop2.model is None
    assert any('Model files not found' in r.getMessage() for r in caplog.records)
    assert (marks, _ev(broker2)) == flat
    assert not [e for e in broker2.events if e.get('side') == 'buy']
    assert list(paths['model_dir'].iterdir()) == []   # nothing written there


def test_live_modes_flag_is_redirected_and_absent_by_default(env):
    import notify
    paths = apply_live_modes(env.mp, env.tmp)
    assert notify._HALT_FLAG == str(paths['halt_flag'])
    assert not paths['halt_flag'].exists() and notify.halt_active() is False


def test_combined_mode_stock_startup_leaves_crypto_stops_alone(env, caplog):
    """run_bots combined mode: CryptoLoop starts first, StockLoop 5 s later
    on the SAME account. The stock startup cleanup is universe-scoped
    (order_utils.cancel_all_open_orders(symbols=...)): 'Canceling 0/6' —
    the six crypto stops stay live with the same ids (live evidence
    2026-09-27 03:09:26)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    sessions = [(now + _dt.timedelta(days=1),
                 now + _dt.timedelta(days=1, hours=6, minutes=30))]
    loop, broker = env.build(sessions=sessions)
    for s, px in STOCKS.items():
        broker.tape[s] = [px]
    env.drive(loop, broker, 1)
    crypto_live = sorted(o['id'] for o in broker.open_orders())
    assert len(crypto_live) == 6
    n_ev = len(broker.events)

    sl = stock_loop.StockLoop()
    env.common(sl, 'stock')
    sl.get_symbol_universe = lambda: list(STOCKS)
    sl.write_prediction_cache = lambda *a, **k: None
    sl._get_predictions = lambda bench: ({s: 0.0 for s in STOCKS}, {})
    with caplog.at_level(logging.INFO, logger='order_utils'):
        env.drive(sl, broker, 1)              # startup + 1 closed-market cycle
    assert any('Canceling 0/6 open order(s)' in r.getMessage()
               for r in caplog.records)
    assert broker.events[n_ev:] == []
    assert sorted(o['id'] for o in broker.open_orders()) == crypto_live
    assert sl.positions == {}
    # the crypto loop still tracks exactly those ids
    assert {loop.positions[s].stop_order_id for s in UNIVERSE} == set(crypto_live)
