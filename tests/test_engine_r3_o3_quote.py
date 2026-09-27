"""ENGINE round-3 W10 — two live-path fixes.

1. O3 — strategy_config.BREAKER_SERVER_FILL_ATTRIB (default OFF;
   TRADER_BREAKER_SERVER_FILL_ATTRIB wins when set). base_loop's cycle runs
   _circuit_breaker_check BEFORE _manage_stops, so on a gap that already
   FILLED the resting server stops the breaker journaled ESTIMATED
   'circuit_breaker' exits at the quote mid for positions the broker had
   already closed. ON: after the flatten, a released position whose
   stop_order_id is 'filled' is journaled through _manage_stops' own
   server-fill calls (server_stop row, real fill, detect_source='breaker',
   last_trade_time, lockout). OFF: byte-identical — no get_order at all.
   (End-to-end replay pins: tests/test_engine_r2_replay_harness.py O3 pair.)

2. order_utils.get_quote — a quote timestamp that cannot be turned into a
   finite age (present-but-unparsable, or ABSENT: t None / no 't' key)
   FAILS CLOSED (None, like an over-age quote); unparsable used to be
   `except Exception: pass` and absent skipped the check — both accepted
   as FRESH. Every parsable form is pinned unchanged. The census
   classifier agrees with the live verdict.
"""

import dataclasses
import datetime as _dt
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

import order_utils  # noqa: E402
import strategy_config  # noqa: E402
import crypto_quote_staleness_census as cq  # noqa: E402

UTC = _dt.timezone.utc


def _now():
    return _dt.datetime.now(UTC)


# ===========================================================================
# 1. O3 — BREAKER_SERVER_FILL_ATTRIB
# ===========================================================================

@pytest.fixture
def bl(monkeypatch):
    pytest.importorskip('alpaca_trade_api')   # base_loop -> trading_utils
    pytest.importorskip('torch')               # base_loop -> predict_now
    import base_loop
    monkeypatch.delenv('TRADER_BREAKER_SERVER_FILL_ATTRIB', raising=False)
    monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', False)
    return base_loop


def test_constant_default_off():
    assert strategy_config.BREAKER_SERVER_FILL_ATTRIB is False


@pytest.mark.parametrize('const,env,expected', [
    (False, None, False),          # default
    (True, None, True),            # constant ON
    ('absent', None, False),       # constant missing -> OFF
    (False, '1', True), (False, 'true', True), (False, 'YES', True),
    (False, 'on', True),
    (True, '0', False), (True, 'off', False), (True, 'no', False),
    (True, '', True),              # empty env = unset -> constant decides
])
def test_flag_reader(bl, monkeypatch, const, env, expected):
    if const == 'absent':
        monkeypatch.delattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB',
                            raising=False)
    else:
        monkeypatch.setattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB',
                            const)
    if env is not None:
        monkeypatch.setenv('TRADER_BREAKER_SERVER_FILL_ATTRIB', env)
    assert bl._breaker_server_fill_attrib() is expected


class _Api:
    """get_account for the trip journal; get_order per stop id."""

    def __init__(self, orders):
        self.orders = orders
        self.calls = []

    def get_account(self):
        self.calls.append(('get_account', None))
        return SimpleNamespace(equity='90000', last_equity='100000')

    def get_order(self, oid):
        self.calls.append(('get_order', oid))
        o = self.orders[oid]
        if isinstance(o, Exception):
            raise o
        return o


def _stub_loop(bl, positions, orders, quotes):
    """A minimal object carrying the REAL _circuit_breaker_check,
    _breaker_record_server_fill and _record_confirmed_exit; the rest is
    stubbed and recorded."""
    BTL = bl.BaseTradingLoop
    ev = []

    class L:
        CIRCUIT_BREAKER_PCT = 0.05
        _circuit_breaker_check = BTL._circuit_breaker_check
        _record_confirmed_exit = BTL._record_confirmed_exit
        _next_baseline_reset = staticmethod(BTL._next_baseline_reset)
        if hasattr(BTL, '_breaker_record_server_fill'):
            _breaker_record_server_fill = BTL._breaker_record_server_fill

        def get_asset_type(self):
            return 'crypto'

        def get_symbol_universe(self):
            return list(positions)

        def get_quote(self, sym):
            ev.append(('get_quote', sym))
            return {'midpoint': quotes[sym]}

        def _classify_server_stop(self, sym, pos, so):
            ev.append(('classify', sym, so.id))
            return 'hard', float(so.stop_price)

        def _apply_server_stop_lockout(self, sym, kind):
            ev.append(('lockout', sym, kind))
            self.hard_stop_lockout[sym] = 'now'

        def _breaker_note_realized(self, pnl):
            ev.append(('note_realized', round(pnl, 6)))

    loop = L()
    loop.api = _Api(orders)
    loop.positions = dict(positions)
    loop._halted_until = None
    loop._buys_allowed = True
    loop.last_trade_time = {}
    loop.hard_stop_lockout = {}
    loop.llm_scores = {'AAA/USD': {'s': 0.4, 'r': 'why'}}
    return loop, ev


def _book(bl):
    from types_mod import Position
    P = lambda sid: Position(qty=2.0, entry_price=100.0,  # noqa: E731
                             high_water_mark=101.0, stop_order_id=sid)
    positions = {'AAA/USD': P('s-filled'),   # server stop already filled
                 'BBB/USD': P(None),         # no resting stop
                 'CCC/USD': P('s-cancel'),   # flatten cancelled its stop
                 'DDD/USD': P('s-raise')}    # status probe fails
    orders = {
        's-filled': SimpleNamespace(id='s-filled', status='filled',
                                    filled_avg_price='93.0', stop_price='94.0'),
        's-cancel': SimpleNamespace(id='s-cancel', status='canceled',
                                    filled_avg_price=None, stop_price='94.0'),
        's-raise': RuntimeError('HTTP 500'),
    }
    quotes = {'AAA/USD': 95.0, 'BBB/USD': 96.0, 'CCC/USD': 97.0,
              'DDD/USD': 98.0}
    return positions, orders, quotes


def _run_trip(bl, monkeypatch, attrib):
    trades, journal, flat = [], [], []
    monkeypatch.setattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB', attrib,
                        raising=False)   # OFF test also runs pre-flag
    monkeypatch.setattr(bl, 'check_circuit_breaker',
                        lambda api, max_drawdown_pct: (True, 0.10))

    def _flatten(api, symbols=None):
        flat.append(len(api.calls))
        return []
    monkeypatch.setattr(bl, 'emergency_flatten', _flatten)
    monkeypatch.setattr(bl, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    monkeypatch.setattr(bl, 'log_decision', lambda row: journal.append(row))
    import notify
    monkeypatch.setattr(notify, 'notify', lambda *a, **k: None)
    positions, orders, quotes = _book(bl)
    loop, ev = _stub_loop(bl, positions, orders, quotes)
    out = loop._circuit_breaker_check()
    return loop, ev, trades, journal, flat, out


def test_o3_flag_off_is_legacy_and_never_calls_get_order(bl, monkeypatch):
    loop, ev, trades, journal, flat, out = _run_trip(bl, monkeypatch, False)
    assert out is True and loop._buys_allowed is False
    assert loop._halted_until is not None and loop.positions == {}
    assert [c for c in loop.api.calls if c[0] == 'get_order'] == []
    # legacy rows: one estimated circuit_breaker row per released position,
    # priced at the quote MID, in pre-flatten order
    assert [(a[0], a[3], k) for a, k in trades] == [
        ('AAA/USD', 95.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
        ('BBB/USD', 96.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
        ('CCC/USD', 97.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
        ('DDD/USD', 98.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
    ]
    assert [r for r in journal if r.get('action') == 'sell'] == []
    assert loop.hard_stop_lockout == {} and loop.last_trade_time == {}
    assert [e[0] for e in ev] == ['get_quote'] * 4


def test_o3_flag_on_attributes_filled_server_stop(bl, monkeypatch):
    loop, ev, trades, journal, flat, out = _run_trip(bl, monkeypatch, True)
    assert out is True and loop._buys_allowed is False
    assert loop._halted_until is not None and loop.positions == {}
    # every status probe comes AFTER the flatten (no added latency), one
    # per released position that HAS a stop id
    gos = [i for i, c in enumerate(loop.api.calls) if c[0] == 'get_order']
    assert [loop.api.calls[i][1] for i in gos] == ['s-filled', 's-cancel',
                                                   's-raise']
    assert flat and min(gos) >= flat[0]
    # AAA: the server_stop row with the REAL fill, through the same calls
    # as _manage_stops' server-fill branch, plus detect_source='breaker'
    a, k = trades[0]
    assert a[:4] == ('AAA/USD', 'sell', 100.0, 93.0)
    assert a[4] == pytest.approx(-7.0)
    assert k == {'llm_score': 0.4, 'reasoning': 'why',
                 'exit_reason': 'server_stop', 'estimated': False}
    sells = [r for r in journal if r.get('action') == 'sell']
    assert len(sells) == 1
    r = sells[0]
    assert r['symbol'] == 'AAA/USD' and r['exit_reason'] == 'server_stop'
    assert r['fill_price'] == 93.0 and r['estimated'] is False
    assert r['server_stop_kind'] == 'hard' and r['stop_px'] == 94.0
    assert r['detect_source'] == 'breaker'
    assert set(loop.last_trade_time) == {'AAA/USD'}
    assert loop.hard_stop_lockout == {'AAA/USD': 'now'}
    assert ('classify', 'AAA/USD', 's-filled') in ev
    assert ('lockout', 'AAA/USD', 'hard') in ev
    assert ('note_realized', -14.0) in ev          # (93-100) * qty 2
    assert ('get_quote', 'AAA/USD') not in ev
    # everything else keeps today's estimated row at the quote mid
    assert [(a[0], a[3], k) for a, k in trades[1:]] == [
        ('BBB/USD', 96.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
        ('CCC/USD', 97.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
        ('DDD/USD', 98.0, {'exit_reason': 'circuit_breaker', 'estimated': True}),
    ]


def test_o3_env_override_off_wins_over_constant(bl, monkeypatch):
    monkeypatch.setenv('TRADER_BREAKER_SERVER_FILL_ATTRIB', '0')
    loop, ev, trades, journal, flat, out = _run_trip(bl, monkeypatch, True)
    assert [c for c in loop.api.calls if c[0] == 'get_order'] == []
    assert all(k['exit_reason'] == 'circuit_breaker' for a, k in trades)


def test_o3_flatten_failure_keeps_position_and_writes_no_row(bl, monkeypatch):
    """A position the flatten could NOT release stays tracked and gets no
    exit row and no status probe, flag ON (unchanged contract)."""
    monkeypatch.setattr(strategy_config, 'BREAKER_SERVER_FILL_ATTRIB', True)
    monkeypatch.setattr(bl, 'check_circuit_breaker',
                        lambda api, max_drawdown_pct: (True, 0.10))
    monkeypatch.setattr(bl, 'emergency_flatten',
                        lambda api, symbols=None: ['AAAUSD'])
    trades = []
    monkeypatch.setattr(bl, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    monkeypatch.setattr(bl, 'log_decision', lambda row: None)
    import notify
    monkeypatch.setattr(notify, 'notify', lambda *a, **k: None)
    positions, orders, quotes = _book(bl)
    loop, ev = _stub_loop(bl, positions, orders, quotes)
    assert loop._circuit_breaker_check() is True
    assert set(loop.positions) == {'AAA/USD'}
    assert ('get_order', 's-filled') not in loop.api.calls
    assert 'AAA/USD' not in [a[0] for a, k in trades]


def test_o3_record_helper_never_raises_and_falls_back(bl, monkeypatch):
    """_breaker_record_server_fill: False (-> estimated row) when the id is
    missing, get_order raises, the status is not 'filled', or the journal
    write raises; a lockout-write failure still returns True (the row is
    written, the breaker must reach its halt latch)."""
    positions, orders, quotes = _book(bl)
    loop, ev = _stub_loop(bl, positions, orders, quotes)
    f = loop._breaker_record_server_fill
    assert f('BBB/USD', positions['BBB/USD']) is False
    assert f('CCC/USD', positions['CCC/USD']) is False
    assert f('DDD/USD', positions['DDD/USD']) is False
    for st in ('partially_filled', 'new', 'accepted', None):
        loop.api.orders['s-x'] = SimpleNamespace(id='s-x', status=st,
                                                 filled_avg_price='93')
        p = dataclasses.replace(positions['AAA/USD'], stop_order_id='s-x')
        assert f('AAA/USD', p) is False

    def _boom(*a, **k):
        raise OSError('journal disk full')
    loop._record_confirmed_exit = _boom
    assert f('AAA/USD', positions['AAA/USD']) is False
    assert loop.hard_stop_lockout == {}

    loop2, ev2 = _stub_loop(bl, positions, orders, quotes)
    monkeypatch.setattr(bl, 'record_trade', lambda *a, **k: None)
    monkeypatch.setattr(bl, 'log_decision', lambda row: None)

    def _lock_boom(sym, kind):
        raise OSError('lockout file unwritable')
    loop2._apply_server_stop_lockout = _lock_boom
    assert loop2._breaker_record_server_fill(
        'AAA/USD', positions['AAA/USD']) is True
    assert 'AAA/USD' in loop2.last_trade_time


# ===========================================================================
# 2. get_quote — unparsable timestamp fails CLOSED
# ===========================================================================

class _QApi:
    def __init__(self, q):
        self.q = q

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: self.q}

    def get_latest_quote(self, symbol):
        return self.q


def _q(t):
    return SimpleNamespace(bp=100.0, ap=100.1, t=t)


def _sdk_quote(raw_t='<absent>'):
    """The REAL legacy-SDK entity (alpaca_trade_api QuoteV2): q.t is parsed
    inside __getattr__ as pd.Timestamp(raw, tz=America/New_York, unit=ns)."""
    ev2 = pytest.importorskip('alpaca_trade_api.entity_v2')
    raw = {'bp': 100.0, 'ap': 100.1}
    if raw_t != '<absent>':
        raw['t'] = raw_t
    return ev2.QuoteV2(raw)


def _rfc3339_ns(age_s):
    t = _now() - _dt.timedelta(seconds=age_s)
    return t.strftime('%Y-%m-%dT%H:%M:%S.') + f'{t.microsecond:06d}789Z'


_EXPECTED = {'bid': 100.0, 'ask': 100.1, 'spread': 100.1 - 100.0,
             'midpoint': (100.0 + 100.1) / 2.0,
             'spread_pct': (100.1 - 100.0) / ((100.0 + 100.1) / 2.0) * 100.0}


def _assert_legacy_quote(out):
    assert out is not None
    fts = out.pop('fetched_ts')
    assert isinstance(fts, float)
    qt = out.pop('quote_t')            # ENGINE r5 additive key (exchange time)
    assert isinstance(qt, float) and qt == qt
    assert out == _EXPECTED            # exact floats, legacy arithmetic


# -- parsable forms: unchanged (fresh accepted, stale rejected) ------------

_PARSABLE = {
    'aware_utc': lambda age: _now() - _dt.timedelta(seconds=age),
    'aware_ny_pd': lambda age: pd.Timestamp(
        _now() - _dt.timedelta(seconds=age)).tz_convert('America/New_York'),
    'naive_utc': lambda age: (_now() - _dt.timedelta(seconds=age)
                              ).replace(tzinfo=None),
    'naive_pd': lambda age: pd.Timestamp(
        (_now() - _dt.timedelta(seconds=age)).replace(tzinfo=None)),
    'aware_offset': lambda age: (_now() - _dt.timedelta(seconds=age)
                                 ).astimezone(_dt.timezone(_dt.timedelta(hours=-5))),
}


@pytest.mark.parametrize('kind', sorted(_PARSABLE))
@pytest.mark.parametrize('asset', ['crypto', 'stock'])
def test_parsable_timestamps_unchanged(kind, asset, caplog):
    mk = _PARSABLE[kind]
    _assert_legacy_quote(order_utils.get_quote(_QApi(_q(mk(5))), 'X', asset))
    _assert_legacy_quote(order_utils.get_quote(_QApi(_q(mk(170))), 'X', asset))
    with caplog.at_level(logging.WARNING, logger='order_utils'):
        assert order_utils.get_quote(_QApi(_q(mk(400))), 'X', asset) is None
    assert any('stale, ignoring' in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize('age,accepted', [(5, True), (170, True),
                                          (190, False), (900, False)])
def test_real_sdk_entity_rfc3339_nanoseconds_unchanged(age, accepted):
    q = _sdk_quote(_rfc3339_ns(age))
    assert isinstance(q.t, pd.Timestamp) and str(q.t.tz) == 'America/New_York'
    out = order_utils.get_crypto_quote(_QApi(q), 'BTC/USD')
    if accepted:
        _assert_legacy_quote(out)
    else:
        assert out is None


def test_real_sdk_entity_int_nanoseconds_unchanged():
    q = _sdk_quote(int((_now().timestamp() - 5) * 1e9))
    _assert_legacy_quote(order_utils.get_crypto_quote(_QApi(q), 'BTC/USD'))


def test_alpaca_compat_shim_aware_utc_unchanged():
    import alpaca_compat
    fresh = alpaca_compat._shim_quote(SimpleNamespace(
        bid_price=100.0, ask_price=100.1,
        timestamp=_now() - _dt.timedelta(seconds=5)))
    _assert_legacy_quote(order_utils.get_stock_quote(_QApi(fresh), 'TSLA'))
    stale = alpaca_compat._shim_quote(SimpleNamespace(
        bid_price=100.0, ask_price=100.1,
        timestamp=_now() - _dt.timedelta(seconds=600)))
    assert order_utils.get_stock_quote(_QApi(stale), 'TSLA') is None


# -- absent timestamp: fail CLOSED (was: check skipped = FRESH) -------------

@pytest.mark.parametrize('asset', ['crypto', 'stock'])
def test_absent_timestamp_fails_closed(asset, caplog):
    for q in (_q(None), SimpleNamespace(bp=100.0, ap=100.1)):
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger='order_utils'):
            assert order_utils.get_quote(_QApi(q), 'X', asset) is None
        msgs = [r.getMessage() for r in caplog.records
                if 'missing quote timestamp' in r.getMessage()]
        assert len(msgs) == 1 and 'raw type NoneType' in msgs[0]


def test_real_sdk_entity_missing_t_key_fails_closed():
    q = _sdk_quote()
    assert getattr(q, 't', None) is None
    assert order_utils.get_crypto_quote(_QApi(q), 'BTC/USD') is None


def test_alpaca_compat_shim_without_timestamp_fails_closed():
    import alpaca_compat
    shim = alpaca_compat._shim_quote(
        SimpleNamespace(bid_price=100.0, ask_price=100.1))
    assert shim.t is None
    assert order_utils.get_stock_quote(_QApi(shim), 'TSLA') is None


# -- present but unparsable: fail CLOSED -------------------------------------

class _NonFiniteAgeTs:
    """A timestamp whose age comes out NaN without raising (pandas versions
    where NaT arithmetic does not raise) — the isfinite guard."""
    tzinfo = UTC

    def astimezone(self, tz):
        return self

    def __rsub__(self, other):
        return SimpleNamespace(total_seconds=lambda: float('nan'))


_UNPARSABLE = {
    'str_garbage': lambda: 'not-a-time',
    'str_iso_Z': lambda: _now().strftime('%Y-%m-%dT%H:%M:%SZ'),
    'int': lambda: 5,
    'float_nan': lambda: float('nan'),
    'date_only': lambda: _now().date(),
    'pd_NaT': lambda: pd.NaT,
    'magicmock': lambda: MagicMock(),
    'overflow_min_plus1h': lambda: _dt.datetime.min.replace(
        tzinfo=_dt.timezone(_dt.timedelta(hours=1))),
    'nonfinite_age': lambda: _NonFiniteAgeTs(),
}


@pytest.mark.parametrize('kind', sorted(_UNPARSABLE))
@pytest.mark.parametrize('asset', ['crypto', 'stock'])
def test_unparsable_timestamp_fails_closed(kind, asset, caplog):
    t = _UNPARSABLE[kind]()
    with caplog.at_level(logging.DEBUG, logger='order_utils'):
        out = order_utils.get_quote(_QApi(_q(t)), 'X', asset)
    assert out is None
    msgs = [r.getMessage() for r in caplog.records
            if 'missing quote timestamp (raw type' in r.getMessage()]
    assert len(msgs) == 1 and caplog.records[-1].levelno == logging.DEBUG
    assert f'raw type {type(t).__name__}' in msgs[0]


@pytest.mark.parametrize('raw', [None, '', 'garbage', 10 ** 30],
                         ids=['null', 'empty', 'garbage', 'out_of_bounds'])
def test_real_sdk_entity_corrupt_t_fails_closed(raw):
    """W7's probe cases on the REAL SDK entity: raw null/'' -> NaT,
    'garbage' -> DateParseError inside the getattr itself, an out-of-range
    int -> OutOfBoundsDatetime. All were accepted as FRESH before."""
    q = _sdk_quote(raw)
    assert order_utils.get_crypto_quote(_QApi(q), 'BTC/USD') is None


def test_named_exceptions_are_exactly_the_parse_path_set():
    """The clause names what the parse path raises (probe + tests above);
    an unexpected exception type still fails closed via the outer handler
    (warning), never as FRESH."""
    import inspect
    src = inspect.getsource(order_utils.get_quote)
    assert ('except (TypeError, ValueError, AttributeError, OverflowError)'
            in src)
    assert 'pass  # unparseable timestamp' not in src

    class _Weird:
        tzinfo = UTC

        def astimezone(self, tz):
            raise RuntimeError('tz database exploded')
    assert order_utils.get_crypto_quote(_QApi(_q(_Weird())), 'X/USD') is None


# -- census classifier == live verdict for the new cases --------------------

class _CApi:
    def __init__(self, q):
        self.q = q

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: self.q}


@pytest.mark.parametrize('kind', sorted(_UNPARSABLE))
def test_census_classifies_unparsable_as_stale_matching_live(kind):
    rec = cq.sample_once(_CApi(_q(_UNPARSABLE[kind]())), 'DOGE/USD', 0,
                         order_utils.get_crypto_quote)
    assert rec['live_none'] is True
    assert rec['reason'] == 'stale' and rec['age_s'] is None


@pytest.mark.parametrize('raw', [None, '', 'garbage'])
def test_census_real_sdk_corrupt_t_matches_live(raw):
    rec = cq.sample_once(_CApi(_sdk_quote(raw)), 'BTC/USD', 0,
                         order_utils.get_crypto_quote)
    assert rec['live_none'] is True and rec['reason'] == 'stale'
    assert rec['quote_t_utc'] is None


def test_census_absent_is_stale_and_parsable_unchanged():
    for q in (_q(None), SimpleNamespace(bp=100.0, ap=100.1)):
        rec = cq.sample_once(_CApi(q), 'X/USD', 0,
                             order_utils.get_crypto_quote)
        assert rec['reason'] == 'stale' and rec['live_none'] is True
        assert rec['age_s'] is None and rec['quote_t_utc'] is None
    rec = cq.sample_once(_CApi(_q(_now() - _dt.timedelta(seconds=30))),
                         'X/USD', 0, order_utils.get_crypto_quote)
    assert rec['reason'] == 'ok' and 25 < rec['age_s'] < 60


def test_census_none_under_unparsable_is_none_at_every_threshold():
    rec = {'reason': 'stale', 'age_s': None}
    for th in (180.0, 300.0, 600.0, 900.0, 1e9):
        assert cq.none_under(rec, th) is True
    # an age-bearing stale sample is unchanged: strict > per threshold
    rec = {'reason': 'stale', 'age_s': 400.0}
    assert cq.none_under(rec, 300.0) is True
    assert cq.none_under(rec, 600.0) is False
    # a legacy (pre-W10 replay file) absent-ts 'ok' row is never None
    assert cq.none_under({'reason': 'ok', 'age_s': None}, 180.0) is False
