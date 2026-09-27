"""ENGINE round-2 W6: recorded-day replay harness for BOTH live loops.

Drives the REAL ``CryptoLoop.run()`` / ``StockLoop.run()`` (startup cancel
-> _reconstruct_positions -> _replace_protective_stops -> cycles) against
tests/fake_alpaca_broker.py:

* crypto: the recorded read-only tape tests/fixtures/crypto_tape_2026-09-27.json
  (six names, latest quote + trade every 30 s, ~10 min) — replayed for its
  full length, one loop cycle per tape row;
* stocks: a synthetic RTH day generated in code (09:30-16:00 ET, a -4 % gap
  at 10:00, a +3 % rally from 14:00 into the close), one cycle per 5 min.

Everything network-touching is stubbed at its real seam (fixed prediction
dicts per cycle — no torch inference; macro neutral; LLM off; journals,
trade memory and state files captured or redirected to tmp_path); sockets
are blocked and any attempt fails the test. The loops' ``datetime``/``time``
module globals are pointed at the fake clock (install_clock) — no
production time call is changed.

Also holds the ENGINE r2 O3 / O2 evidence (see the W6 report):
  O3 — breaker-before-stops mis-attribution on a gap. Landed in ENGINE r3
       (W10) behind strategy_config.BREAKER_SERVER_FILL_ATTRIB (default
       OFF): the CURRENT-BEHAVIOUR pin is now the flag-OFF pin (plus: no
       get_order inside the breaker); the former strict xfail now runs with
       the flag ON and passes.
  O2 — startup 6 % fallback stop then cycle-1 cancel/replace at the 5 % trail
       (orders per restart, unprotected window in REST calls; owner item).
"""

import copy
import datetime as _dt
import json
import os
import socket
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils chain
pytest.importorskip('torch')               # predict_now import chain

import base_loop                                   # noqa: E402
import crypto_loop                                 # noqa: E402
import stock_loop                                  # noqa: E402
import order_utils                                 # noqa: E402
import trading_utils                               # noqa: E402
from types_mod import MacroRegime, Position        # noqa: E402
from strategy_config import CRYPTO_POLICY, STOCK_POLICY  # noqa: E402
from fake_alpaca_broker import (FakeAlpacaBroker, FakeClock, Tape,  # noqa: E402
                                install_clock, synthetic_rth_tape)

TAPE_PATH = HERE / 'fixtures' / 'crypto_tape_2026-09-27.json'
UNIVERSE = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD', 'LINK/USD']
THRESHOLD = 0.15
ATR_FRAC = 0.01

# Real paper-account snapshot (generals/engine/account_snapshot.json, read-only
# GET 2026-09-27 00:20) — same values as tests/test_engine_r1_inherited_positions.
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

STOCKS = {'AAA': 50.0, 'BBB': 120.0, 'CCC': 80.0, 'DDD': 40.0, 'EEE': 25.0}
RTH_DAY = _dt.date(2026, 9, 28)           # a Monday
STOCK_SNAPSHOT = {
    'account': {'status': 'ACTIVE', 'currency': 'USD', 'cash': '200000',
                'buying_power': '400000', 'non_marginable_buying_power': '200000',
                'equity': '200000', 'portfolio_value': '200000',
                'last_equity': '200000', 'multiplier': '2',
                'shorting_enabled': False, 'trading_blocked': False},
    'positions': [], 'open_orders': [],
}


def _load_tape():
    return Tape.from_json(TAPE_PATH)


class Rec:
    def __init__(self):
        self.journal, self.trades, self.risk, self.notes = [], [], [], []
        self.errors, self.net, self.state_blobs = [], [], []
        self.cycle_calls = []

    def rows(self, action):
        return [r for r in self.journal if r.get('action') == action]


@pytest.fixture
def replay(tmp_path, monkeypatch):
    """Returns an object with build_crypto(...), build_stock(...), drive()."""
    rec = Rec()
    for k in ('TRADER_ORDER_STREAM', 'TRADER_USE_ALPACA_PY'):
        monkeypatch.delenv(k, raising=False)
    import random as _random
    monkeypatch.setattr(_random, 'uniform', lambda a, b: 0.0)

    def _no_net(*a, **k):
        rec.net.append(repr(a)[:120])
        raise OSError('network blocked in the replay harness')
    monkeypatch.setattr(socket.socket, 'connect', _no_net)
    monkeypatch.setattr(socket, 'create_connection', _no_net)
    monkeypatch.setattr(socket, 'getaddrinfo', _no_net)

    import notify, macro_calendar, trade_journal, portfolio, market_data
    import risk_budget, funding, volatility, meta_label, shadow
    import events_calendar, edgar_events, macro_indicators, panel_ranks
    import predict_now
    monkeypatch.setattr(notify, 'flatten_requested', lambda: False)
    monkeypatch.setattr(notify, 'halt_active', lambda: False)
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
        CP.get(s) or STOCKS.get(s) or 100.0) * ATR_FRAC
    monkeypatch.setattr(market_data, 'get_live_atr', _atr)
    monkeypatch.setattr(stock_loop, 'get_live_atr', _atr)
    for fn in ('fetch_bars_alpaca', 'fetch_stock_bars_alpaca',
               'fetch_spy_bars_alpaca'):
        monkeypatch.setattr(market_data, fn, lambda *a, **k: None)
    monkeypatch.setattr(crypto_loop, 'fetch_bars_alpaca', lambda *a, **k: None)
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
    h.rec = rec

    def _common(loop, broker, preds_fn, book):
        loop._lockout_file = str(tmp_path / f'{book}_hard_stop_lockout.json')
        state = tmp_path / f'{book}_position_state.json'
        loop._position_state_file = lambda: str(state)
        loop._hot_reload_check = lambda: None
        h.state_path = state

        def _load_models():
            loop.model = object()
            loop.config = {'trade_threshold': THRESHOLD, 'forward_bars': 24}
            loop.trade_threshold = THRESHOLD
        loop._load_models = _load_models
        return loop

    def build_crypto(snapshot=SNAPSHOT, tape=None, preds_fn=None,
                     clock=None, **bkw):
        tape = tape if tape is not None else _load_tape()
        if clock is None:
            st = tape.start_time() if isinstance(tape, Tape) else None
            clock = FakeClock(st)
        broker = FakeAlpacaBroker(snapshot, tape=tape, clock=clock, **bkw)
        install_clock(monkeypatch, clock,
                      [base_loop, crypto_loop, order_utils, trading_utils])
        monkeypatch.setattr(base_loop, 'get_api', lambda: broker)
        loop = crypto_loop.CryptoLoop()
        _common(loop, broker, preds_fn, 'crypto')
        pf = preds_fn or (lambda c: {s: 0.0 for s in UNIVERSE})
        loop._get_predictions = lambda bench: (dict(pf(loop.cycle)), {})
        return loop, broker

    def build_stock(preds_fn, step_s=300, names=STOCKS):
        tape, sessions = synthetic_rth_tape(names, RTH_DAY, step_s=step_s)
        clock = FakeClock(tape.start_time())
        broker = FakeAlpacaBroker(STOCK_SNAPSHOT, tape=tape, clock=clock,
                                  sessions=sessions)
        install_clock(monkeypatch, clock,
                      [base_loop, stock_loop, order_utils, trading_utils])
        monkeypatch.setattr(base_loop, 'get_api', lambda: broker)
        monkeypatch.setattr(base_loop.BaseTradingLoop, '_get_predictions',
                            lambda self, bench: (dict(preds_fn(clock())), {}))
        loop = stock_loop.StockLoop()
        _common(loop, broker, preds_fn, 'stock')
        loop.get_symbol_universe = lambda: list(names)
        loop.write_prediction_cache = lambda *a, **k: None
        h.sessions = sessions
        return loop, broker, tape

    def drive(loop, broker, n_cycles, seconds=30.0, before_cycle=None):
        """Real run(): startup, then n cycles; the tape advances one step
        BEFORE each cycle (startup = step 0, cycle k = step k). After each
        cycle the tmp position-state file is parsed (must be valid JSON).
        Returns the broker event index at the end of startup and each
        cycle."""
        marks = []
        real = type(loop)._run_one_cycle

        def one():
            if not marks:
                marks.append(len(broker.events))
            broker.tick(seconds=seconds)
            if before_cycle:
                before_cycle(loop, broker)
            try:
                real(loop)
            except Exception as e:
                rec.errors.append(repr(e))
                raise KeyboardInterrupt
            marks.append(len(broker.events))
            if h.state_path.exists():
                rec.state_blobs.append(json.loads(h.state_path.read_text()))
            else:
                rec.state_blobs.append(None)
            if loop.cycle >= n_cycles:
                raise KeyboardInterrupt
        loop._run_one_cycle = one
        os.environ.pop('PYTEST_CURRENT_TEST', None)
        with pytest.raises(KeyboardInterrupt):
            loop.run()
        assert rec.errors == []
        assert rec.net == [], f'network attempted: {rec.net[:3]}'
        return marks

    h.build_crypto, h.build_stock, h.drive = build_crypto, build_stock, drive
    yield h


def _submits(broker):
    return [e for e in broker.events if e['kind'] == 'submit']


def _assert_broker_rules(broker, universe):
    """Every order the loop sent obeys the broker rules the fake enforces
    (no reject event = no oversell / no bad price / no bracket-price
    violation) plus the relations checked here."""
    assert broker.events_of('reject') == [], broker.events_of('reject')
    uni = {u.replace('/', '') for u in universe}
    for e in _submits(broker):
        assert e['symbol'].replace('/', '') in uni, e
        assert e['qty'] and e['qty'] > 0, e
        if e['type'] == 'stop_limit':
            assert e['side'] == 'sell'
            assert 0 < e['limit_price'] < e['stop_price'], e
        if e['type'] == 'stop':
            assert e['side'] == 'sell' and e['stop_price'] > 0, e


# ---------------------------------------------------------------------------
# 0. The fixture tape itself
# ---------------------------------------------------------------------------

def test_recorded_tape_fixture_schema_and_size():
    raw = json.loads(TAPE_PATH.read_text())
    assert raw['schema'] == 'fake_alpaca_broker.tape/v1'
    assert TAPE_PATH.stat().st_size < 200_000
    tape = Tape.from_dict(raw)
    assert sorted(tape.symbols) == sorted(UNIVERSE)
    assert len(tape) >= 15                       # ~10 min at 30 s
    for s in UNIVERSE:
        for ts, bid, ask, last, age in tape.rows[s.replace('/', '')]:
            assert 0 < bid <= ask and last > 0
            _dt.datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# 1. CryptoLoop over the recorded tape, full length
# ---------------------------------------------------------------------------

def _crypto_cash_snapshot():
    """Scenario B: $30k cash, holds BTC + ETH with a REAL basis (the tape's
    first mid) so the entry, resting-stop and signal-sell paths all run."""
    tape = _load_tape()
    snap = copy.deepcopy(SNAPSHOT)
    snap['account'].update(cash='30000', non_marginable_buying_power='30000',
                           buying_power='30000')
    keep = []
    for p in snap['positions']:
        sym = p['symbol'][:-3] + '/USD'
        if sym in ('BTC/USD', 'ETH/USD'):
            _, bid, ask, _, _ = tape.row(sym, 0)
            px = round((bid + ask) / 2, 4)
            q = 2000.0 / px
            p.update(qty=repr(q), avg_entry_price=repr(px),
                     cost_basis=repr(q * px), current_price=repr(px))
            keep.append(p)
    snap['positions'] = keep
    # breaker baseline consistent with this book (no trip at startup)
    snap['account'].update(last_equity='34000', equity='34000',
                           portfolio_value='34000')
    return snap


def _preds_b(cycle):
    p = {s: 0.0 for s in UNIVERSE}
    p.update({'SOL/USD': 2.0, 'XRP/USD': 2.0})
    if cycle >= 8:
        p['ETH/USD'] = -2.0                       # signal sell mid-tape
    return p


@pytest.mark.parametrize('scenario', ['inherited_book', 'cash_entries'])
def test_crypto_replay_full_tape_obeys_broker_rules(replay, scenario):
    tape = _load_tape()
    n = len(tape) - 1
    if scenario == 'inherited_book':
        loop, broker = replay.build_crypto(tape=tape, replay_quote_age=True)
    else:
        loop, broker = replay.build_crypto(snapshot=_crypto_cash_snapshot(),
                                           tape=tape, preds_fn=_preds_b,
                                           replay_quote_age=True)
    replay.drive(loop, broker, n)
    rec = replay.rec
    assert loop.cycle == n
    _assert_broker_rules(broker, UNIVERSE)
    # position-state file written (tmp) and valid JSON after EVERY cycle
    assert len(rec.state_blobs) == n and all(isinstance(b, dict)
                                             for b in rec.state_blobs)
    assert all('hwm' in b for b in rec.state_blobs)
    # every tracked position has exactly one resting protective stop
    for s, pos in loop.positions.items():
        live = broker.open_orders(s)
        assert [o['type'] for o in live if o['side'] == 'sell'] == ['stop_limit']
        assert pos.stop_order_id == live[0]['id']
    if scenario == 'cash_entries':
        buys = [e for e in broker.events_of('fill') if e['side'] == 'buy']
        assert {e['symbol'] for e in buys} == {'SOL/USD', 'XRP/USD'}, (
            [(r['action'], r.get('symbol'), r.get('skip_reason'), r.get('veto_counts'))
             for r in rec.journal if r['action'] != 'cycle_latency'][:30],
            broker.events[:30])
        assert 'ETH/USD' not in loop.positions       # signal-sold
        sells = [r for r in rec.rows('sell') if r['symbol'] == 'ETH/USD']
        assert [r['exit_reason'] for r in sells] == ['signal_sell']
        assert sells[0]['estimated'] is False
        # the resting stop was cancelled BEFORE the sell was submitted
        ev = [e for e in broker.events if e.get('symbol') == 'ETH/USD'
              and e['kind'] in ('cancel', 'submit') and e['step'] >= 8]
        assert ev[0]['kind'] == 'cancel' and ev[0]['type'] == 'stop_limit'


def test_crypto_replay_is_deterministic(replay, monkeypatch):
    tape = _load_tape()
    n = len(tape) - 1
    out = []
    for _ in range(2):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()      # run 2 must not inherit run 1's HWM
        loop, broker = replay.build_crypto(snapshot=_crypto_cash_snapshot(),
                                           tape=_load_tape(), preds_fn=_preds_b,
                                           replay_quote_age=True)
        replay.drive(loop, broker, n)
        out.append((broker.normalized_events(),
                    [(a, {k: v for k, v in k_.items()})
                     for a, k_ in replay.rec.trades],
                    replay.rec.journal, replay.rec.state_blobs))
    assert out[0][0] == out[1][0], [
        (r['action'], r.get('symbol'), r.get('skip_reason'), r.get('veto_counts'))
        for r in out[1][2] if r['action'] in ('skip', 'entry_window')][:12]
    assert out[0][1] == out[1][1]                 # trade-memory rows
    assert out[0][2] == out[1][2]                 # decision journal
    assert out[0][3] == out[1][3]                 # position-state blobs
    assert len(out[0][0]) > 0


# ---------------------------------------------------------------------------
# 2. StockLoop over a synthetic RTH day
# ---------------------------------------------------------------------------

def _et(h, m):
    import zoneinfo
    return _dt.datetime(RTH_DAY.year, RTH_DAY.month, RTH_DAY.day, h, m,
                        tzinfo=zoneinfo.ZoneInfo('America/New_York'))


def _stock_preds(now):
    """AAA/BBB bullish all day; CCC below threshold; DDD/EEE bullish from
    14:30 ET; EEE turns mildly negative at 15:45 (not a signal sell, but
    no longer a sleeve keeper) -> the EOD flatten must sell it."""
    et = now.astimezone(_et(9, 30).tzinfo)
    p = {'AAA': 2.0, 'BBB': 2.0, 'CCC': 0.05, 'DDD': 0.0, 'EEE': 0.0}
    if et >= _et(14, 30):
        p.update(DDD=2.0, EEE=2.0)
    if et >= _et(15, 45):
        p['EEE'] = -0.05
    return p


def _run_stock_day(replay):
    loop, broker, tape = replay.build_stock(_stock_preds)
    n = len(tape) - 1
    replay.drive(loop, broker, n, seconds=300.0)
    return loop, broker, tape


def test_stock_replay_rth_day(replay):
    loop, broker, tape = _run_stock_day(replay)
    rec = replay.rec
    (open_utc, close_utc), = replay.sessions
    _assert_broker_rules(broker, list(STOCKS))
    subs = _submits(broker)
    when = lambda e: _dt.datetime.fromisoformat(e['t'])    # noqa: E731
    # Entries only inside market hours — and only inside the configured
    # ET entry windows (09:45-11:00, 14:30-15:30).
    buys = [e for e in subs if e['side'] == 'buy']
    assert {e['symbol'] for e in buys} == {'AAA', 'BBB', 'DDD', 'EEE'}
    for e in buys:
        assert open_utc <= when(e) < close_utc and e['mkt_open'] is True
        et = when(e).astimezone(_et(9, 30).tzinfo)
        assert (_et(9, 45) <= et < _et(11, 0)) or (_et(14, 30) <= et < _et(15, 30))
        # bracket legs on every stock entry
        assert e['order_class'] == 'bracket'
        legs = broker.orders[e['id']]['legs']
        kinds = sorted(broker.orders[i]['leg'] for i in legs)
        assert kinds == ['stop_loss', 'take_profit']
    # Nothing submitted after the close (the loop idles on clock.is_open)
    assert [e for e in subs if when(e) >= close_utc] == []
    assert all(e['mkt_open'] for e in subs)
    # The 10:00 gap fills AAA/BBB server-side stop legs; the loop journals
    # them as server_stop with the real fill and locks the names out.
    ss = {r['symbol']: r for r in rec.rows('sell')
          if r['exit_reason'] == 'server_stop'}
    assert set(ss) == {'AAA', 'BBB'}
    assert all(r['estimated'] is False for r in ss.values())
    assert {'AAA', 'BBB'} <= set(loop.hard_stop_lockout)
    # EOD: EEE (non-keeper) sold by the flatten BEFORE the close; DDD kept
    # as the overnight sleeve with a GTC stop; nothing else left.
    eod = [r for r in rec.rows('sell') if r['exit_reason'] == 'eod_flatten']
    assert [r['symbol'] for r in eod] == ['EEE']
    eee_fill = [e for e in broker.events_of('fill', 'EEE') if e['side'] == 'sell']
    assert len(eee_fill) == 1 and when(eee_fill[0]) < close_utc
    assert set(broker.positions) == {'DDD'} and set(loop.positions) == {'DDD'}
    live = broker.open_orders('DDD')
    assert [(o['type'], o['time_in_force']) for o in live] == [('stop', 'gtc')]
    assert loop.flattened_today is True
    # state file: absent until the first open cycle (_manage_stops writes
    # it), then valid JSON after every cycle (json.loads in drive()).
    first = next(i for i, b in enumerate(rec.state_blobs) if b is not None)
    assert first > 0 and all(isinstance(b, dict)
                             for b in rec.state_blobs[first:])


def test_stock_replay_is_deterministic(replay):
    out = []
    for _ in range(2):
        replay.rec.__init__()
        if getattr(replay, 'state_path', None) is not None and \
                replay.state_path.exists():
            replay.state_path.unlink()
        loop, broker, tape = _run_stock_day(replay)
        out.append((broker.normalized_events(), replay.rec.trades,
                    replay.rec.journal))
    assert out[0] == out[1] and len(out[0][0]) > 20


def test_overnight_keeper_gtc_stop_survives_same_cycle_trailing_upgrade(replay):
    """Class A fix (stock_loop._manage_stops trailing upgrade, ENGINE r2
    W6). Cycle order: flatten_before_close -> _manage_stops. At 15:50 ET
    _prepare_overnight_keepers cancels the keeper's legs, places a GTC
    'stop' and resets trailing_activated=False; DDD is > +1 % over entry,
    so the SAME cycle's trailing upgrade used to cancel that GTC stop and
    submit a DAY trailing_stop, which expires at 16:00 -> the keeper rode
    overnight with no server-side stop (and next morning stop_order_id
    points at the expired order -> cleared -> no server stop at all)."""
    loop, broker, tape = _run_stock_day(replay)
    (open_utc, close_utc), = replay.sessions
    ev = [e for e in broker.events if e.get('symbol') == 'DDD']
    gtc = [e for e in ev if e['kind'] == 'submit' and e['type'] == 'stop'
           and e.get('time_in_force') == 'gtc']
    assert len(gtc) == 1                         # the sleeve prep ran once
    later = ev[ev.index(gtc[0]) + 1:]
    assert [e for e in later if e['kind'] in ('submit', 'cancel')] == []
    assert not broker.events_of('expire', 'DDD')
    o = broker.orders[gtc[0]['id']]
    assert o['status'] == 'new'                  # live after the close
    assert loop.positions['DDD'].stop_order_id == o['id']
    # the keeper WAS trailing before the prep (the upgrade path is live)
    assert [e for e in ev if e['kind'] == 'submit'
            and e['type'] == 'trailing_stop'
            and _dt.datetime.fromisoformat(e['t']) < _dt.datetime.fromisoformat(
                gtc[0]['t'])]


# ---------------------------------------------------------------------------
# 3. O3 — breaker runs before _manage_stops: server-stop fills mis-attributed
# ---------------------------------------------------------------------------

def _o3_setup(replay, monkeypatch, attrib):
    """attrib = strategy_config.BREAKER_SERVER_FILL_ATTRIB for the run (the
    env override is cleared so the constant decides). trip_calls = the
    broker `calls` made INSIDE each _circuit_breaker_check that returned
    True (the trip first; later entries are the latched halt)."""
    import strategy_config
    monkeypatch.delenv('TRADER_BREAKER_SERVER_FILL_ATTRIB', raising=False)
    monkeypatch.setattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB', attrib,
                        raising=False)   # OFF pin also runs pre-flag
    tape = {s: [CP[s] * m for m in (1.0, 0.93, 0.93)] for s in CP}
    loop, broker = replay.build_crypto(tape=tape, clock=FakeClock())
    order = []
    trip_calls = []
    for name in ('_circuit_breaker_check', '_manage_stops'):
        real = getattr(loop, name)

        def wrap(real=real, name=name):
            if name == '_circuit_breaker_check':
                order.append((name, sorted(
                    broker.orders[p.stop_order_id]['status']
                    for p in loop.positions.values())))
                n0 = len(broker.calls)
                out = real()
                if out:
                    trip_calls.append(list(broker.calls[n0:]))
                return out
            order.append((name, None))
            return real()
        setattr(loop, name, wrap)
    replay.drive(loop, broker, 2)
    return loop, broker, order, trip_calls


def test_o3_breaker_journals_estimates_for_already_filled_server_stops(
        replay, monkeypatch):
    """FLAG-OFF pin (BREAKER_SERVER_FILL_ATTRIB=False, the default) — the
    pre-r3 CURRENT-BEHAVIOUR pin, unchanged, plus: the breaker makes no
    get_order call at all when OFF. Cycle order in
    base_loop._run_one_cycle: _circuit_breaker_check -> flatten_before_close
    -> _manage_stops. On the -7 % gap tick all six resting stops fill
    server-side; when the breaker runs, every tracked position's
    stop_order_id is ALREADY 'filled' at the broker (one get_order away),
    yet it journals six estimated 'circuit_breaker' exits at the quote mid;
    _manage_stops then has nothing left to record."""
    loop, broker, order, trip_calls = _o3_setup(replay, monkeypatch, False)
    rec = replay.rec
    assert trip_calls and all(c == [] for c in trip_calls[1:])  # latched: no calls
    assert [c for c in trip_calls[0] if c[0] == 'get_order'] == []
    assert order[0] == ('_circuit_breaker_check', ['filled'] * 6)
    assert order[1] == ('_manage_stops', None)
    fills = {e['symbol']: e['fill_price'] for e in broker.events_of('fill')}
    assert len(fills) == 6 and broker.positions == {}
    assert [k['exit_reason'] for a, k in rec.trades] == ['circuit_breaker'] * 6
    assert all(k['estimated'] is True for a, k in rec.trades)
    # the estimate is the quote MID, not the real server-stop fill (the bid
    # at the gap tick): every recorded exit price differs from the fill.
    for a, k in rec.trades:
        sym, _, _, px = a[0], a[1], a[2], a[3]
        assert px != pytest.approx(fills[sym], rel=1e-9)
    assert not rec.rows('sell') and loop.hard_stop_lockout == {}
    # no order was needed: emergency_flatten found no broker position
    assert not [e for e in broker.events
                if e['kind'] == 'submit' and e['type'] == 'market']


def test_o3_intended_attribution_server_stop_rows_with_real_fills(
        replay, monkeypatch):
    """FLAG-ON (BREAKER_SERVER_FILL_ATTRIB=True): the former strict xfail
    (ENGINE r2 W6), unchanged assertions, now passing. Plus: every
    get_order the breaker makes comes AFTER the flatten's list_positions
    (no latency added to the liquidation), one per released position."""
    loop, broker, order, trip_calls = _o3_setup(replay, monkeypatch, True)
    rec = replay.rec
    assert trip_calls and all(c == [] for c in trip_calls[1:])  # latched: no calls
    names = [c[0] for c in trip_calls[0]]
    assert 'list_positions' in names
    first_go = names.index('get_order')
    assert first_go > names.index('list_positions')
    assert names.count('get_order') == 6
    fills = {e['symbol']: e['fill_price'] for e in broker.events_of('fill')}
    sells = {r['symbol']: r for r in rec.rows('sell')}
    assert set(sells) == set(UNIVERSE)
    for s, r in sells.items():
        assert r['exit_reason'] == 'server_stop' and r['estimated'] is False
        assert r['fill_price'] == pytest.approx(fills[s])
    assert not [k for a, k in rec.trades
                if k.get('exit_reason') == 'circuit_breaker']
    assert set(loop.hard_stop_lockout) == set(UNIVERSE)


# ---------------------------------------------------------------------------
# 4. O2 — restart stop churn, quantified
# ---------------------------------------------------------------------------

def test_o2_restart_churn_orders_and_unprotected_window(replay):
    """Zero-basis inherited book, flat tape: startup places six 6 % stops,
    cycle 1 cancels and re-places all six at the 5 % trail -> 18 order
    writes per restart instead of 6. Each name is unprotected between its
    cancel and its re-submit for exactly: 1 cancel_order + 1 list_orders
    (the await-clear poll, after one 0.5 s sleep) -> the submit; never a
    whole cycle."""
    loop, broker = replay.build_crypto(
        tape={s: [CP[s]] * 3 for s in CP}, clock=FakeClock())
    marks = replay.drive(loop, broker, 2)
    start, c1 = broker.events[:marks[0]], broker.events[marks[0]:marks[1]]
    writes = lambda ev: [e for e in ev if e['kind'] in ('submit', 'cancel')]  # noqa: E731
    assert len(writes(start)) == 6 and len(writes(c1)) == 12
    assert broker.events[marks[1]:marks[2]] == []
    # per-name gap, in broker API calls between its cancel and re-submit
    calls = broker.calls
    for s in UNIVERSE:
        so = [i for i, c in enumerate(calls)
              if c[0] == 'submit_order' and c[1] == s]
        assert len(so) == 2
        cancel = max(i for i, c in enumerate(calls[:so[1]])
                     if c[0] == 'cancel_order')
        between = [c[0] for c in calls[cancel + 1:so[1]]]
        assert between == ['list_orders']
    # Level identity check the owner needs: startup anchor
    # max(entry,hwm)*(1-_stop_distance_for) vs _desired_stop_for.
    loop.macro_regime = None
    P = CRYPTO_POLICY

    def startup(pos):
        return max(pos.entry_price, pos.high_water_mark) * \
            (1 - loop._stop_distance_for(pos))
    same = Position(qty=1, entry_price=100.0, high_water_mark=100.0, entry_atr=1.0)
    assert startup(same) == loop._desired_stop_for(same)[0]      # identical
    j11 = Position(qty=1, entry_price=100.0, high_water_mark=101.0, entry_atr=1.0)
    assert not loop._desired_stop_for(j11)[3]                    # trail off
    assert startup(j11) == pytest.approx(101 * 0.975)            # 98.475
    assert loop._desired_stop_for(j11)[0] == pytest.approx(97.5)  # differs
    no_atr = Position(qty=1, entry_price=100.0, high_water_mark=103.0,
                      entry_atr=None)                          # trail on
    assert startup(no_atr) == pytest.approx(103 * (1 - P['stop_fallback_pct']))
    assert loop._desired_stop_for(no_atr)[0] == pytest.approx(
        103 * (1 - P['trail_fallback_pct']))                    # churns too
