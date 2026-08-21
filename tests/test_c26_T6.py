"""c26 packet T6 — execution quality (D20 + D21 + maker-share truth +
stream stops + latency).

Mac-runnable, no heavy deps: order_utils / alpaca_compat / order_stream /
trade_journal / fees imported directly; base_loop + stock_loop via the
Wave-A sys.modules stub pattern (tests/test_c26_base_loop_functional.py).

Pins (flag OFF = byte-identical):
  T1  _shim_order stop/limit price + _shim_quote timestamp passthrough
  T2  get_quote fetched_ts stamp (additive sixth key)
  T3  lifecycle flag-OFF byte-compat (market fallback even with ioc_fallback)
  T4  lifecycle flag-ON capped IOC (no market ever; no-quote => fail closed;
      exits still market-fallback)
  T5  place_maker_buy 'entry_fills' journaling (fees count-scan isolation)
  T6  realized_crypto_maker_share_notional (thin sample / exact share /
      malformed skip / TTL cache)
  T7  should_trade flag-OFF identity + exact fee-delta shift flag-ON
  T8  _classify_server_stop truth table
  T9  _apply_server_stop_lockout both flag states
  T10 _stream_stop_fallback both flag states
  T11 crypto _manage_stops integration (REST raise + stream cache)
  T12 _record_confirmed_exit extra-merge + legacy key-set + quote_age_s
  T13 _journal_cycle_latency row shape + never-raises
"""

import datetime as _dt
import inspect
import json
import os as _os
import sys
import time as _time
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# base_loop's only Mac-unimportable links are predict_now (joblib/torch)
# and trading_utils's `from dotenv import load_dotenv`. Stub exactly those
# two, import, then RESTORE sys.modules (same pattern as
# tests/test_c26_base_loop_functional.py).
_dv = types.ModuleType('dotenv'); _dv.load_dotenv = lambda *a, **k: None
_pn = types.ModuleType('predict_now')
_pn.load_models = lambda *a, **k: (None, None, {}, None)
sys.modules['dotenv'] = _dv
sys.modules['predict_now'] = _pn
try:
    import base_loop
    import stock_loop
finally:
    for _m in ('dotenv', 'predict_now', 'trading_utils',
               'base_loop', 'stock_loop', 'crypto_loop'):
        sys.modules.pop(_m, None)

import alpaca_compat
import fees
import order_stream
import order_utils
import trade_journal
from types_mod import Position


# ---------------------------------------------------------------------------
# Concrete subclass + factory (Wave-A pattern)
# ---------------------------------------------------------------------------

class _Loop(base_loop.BaseTradingLoop):
    MODEL_PREFIX = ''

    def get_symbol_universe(self):
        return list(self._universe)

    def check_market_hours(self):
        return True

    def get_asset_type(self):
        return 'crypto'

    def get_quote(self, symbol):
        return self._quotes.get(symbol)

    def place_buy_order(self, *a, **k):
        return None

    def place_sell_order(self, *a, **k):
        return None

    def get_benchmark_close(self):
        return None

    def get_headlines(self, symbol):
        return []

    def flatten_before_close(self):
        pass

    def write_prediction_cache(self, preds, **kwargs):
        pass


def _mk(tmp_path, **over):
    inst = object.__new__(_Loop)
    inst.api = None
    inst.positions = {}
    inst.last_trade_time = {}
    inst.hard_stop_lockout = {}
    inst._lockout_file = str(tmp_path / 'hard_stop_lockout.json')
    inst.llm_scores = {}
    inst.macro_regime = None
    inst.corr_matrix = {}
    inst._equity = 100_000.0
    inst.cycle = 1
    inst._pending_breach = {}
    inst._quotes = {}
    inst._universe = ['BTC/USD']
    inst._save_position_state = lambda: None   # never touch the repo
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


@pytest.fixture
def nosleep(monkeypatch):
    monkeypatch.setattr(order_utils.time, 'sleep', lambda s: None)


# ---------------------------------------------------------------------------
# T1 — alpaca_compat shims
# ---------------------------------------------------------------------------

def _enum(v):
    return SimpleNamespace(value=v)


def _full_order(**over):
    kw = dict(id='O1', client_order_id='cid', symbol='AAPL', qty='1.5',
              side=_enum('buy'), order_type=_enum('market'),
              status=_enum('filled'), filled_qty='1.5',
              filled_avg_price='100.25', notional='150.38',
              submitted_at=None, filled_at=None, legs=None)
    kw.update(over)
    return SimpleNamespace(**kw)


class TestShims:
    def test_shim_order_stop_and_limit_price_from_strings(self):
        s = alpaca_compat._shim_order(
            _full_order(stop_price='101.5', limit_price='99.25'))
        assert s.stop_price == 101.5
        assert s.limit_price == 99.25
        # pre-existing surface unchanged
        assert (s.id, s.symbol, s.qty) == ('O1', 'AAPL', 1.5)
        assert (s.side, s.type, s.status) == ('buy', 'market', 'filled')
        assert s.filled_avg_price == 100.25
        assert s.notional == 150.38

    def test_shim_order_absent_prices_none(self):
        s = alpaca_compat._shim_order(_full_order())
        assert s.stop_price is None
        assert s.limit_price is None

    def test_shim_quote_passthrough(self):
        ts = _dt.datetime(2026, 8, 19, tzinfo=_dt.timezone.utc)
        q = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=1.0, ask_price=2.0, timestamp=ts))
        assert (q.bp, q.ap, q.t) == (1.0, 2.0, ts)

    def test_shim_quote_missing_timestamp_none(self):
        q = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=1.0, ask_price=2.0))
        assert q.t is None


# ---------------------------------------------------------------------------
# T2 — get_quote fetched_ts
# ---------------------------------------------------------------------------

class _QuoteAPI:
    def get_latest_crypto_quotes(self, symbols):
        now = _dt.datetime.now(_dt.timezone.utc)
        return {symbols[0]: SimpleNamespace(bp=100.0, ap=100.1, t=now)}


def test_get_quote_stamps_fetched_ts():
    out = order_utils.get_quote(_QuoteAPI(), 'BTC/USD', 'crypto')
    assert out is not None
    for k in ('bid', 'ask', 'spread', 'midpoint', 'spread_pct'):
        assert k in out
    assert out['bid'] == 100.0 and out['ask'] == 100.1
    assert abs(_time.time() - out['fetched_ts']) < 2.0


# ---------------------------------------------------------------------------
# T3/T4 — manage_order_lifecycle ioc_fallback
# ---------------------------------------------------------------------------

class _API:
    def __init__(self):
        self.submitted = []
        self.canceled = []
        self.orders = {}

    def add(self, oid, **kw):
        o = SimpleNamespace(id=oid, status='new', symbol='BTC/USD', qty=1.0,
                            side='buy', filled_qty=0.0, filled_avg_price=None)
        for k, v in kw.items():
            setattr(o, k, v)
        self.orders[oid] = o
        return o

    def get_order(self, oid):
        return self.orders[oid]

    def cancel_order(self, oid):
        self.canceled.append(oid)

    def submit_order(self, **kw):
        self.submitted.append(kw)
        oid = f"sub{len(self.submitted)}"
        filled = kw.get('time_in_force') == 'ioc'
        o = SimpleNamespace(
            id=oid, status=('filled' if filled else 'accepted'),
            symbol=kw.get('symbol'), qty=kw.get('qty'), side=kw.get('side'),
            filled_qty=(kw.get('qty') if filled else 0.0),
            filled_avg_price=(100.0 if filled else None))
        self.orders[oid] = o
        return o


_IOC_QUOTE = {'bid': 99.9, 'ask': 100.1, 'spread': 0.2,
              'midpoint': 100.0, 'spread_pct': 0.2}


def _run_lifecycle(api, **kw):
    return order_utils.manage_order_lifecycle(
        api, 'o1', timeout=2, poll_interval=1, fallback_to_market=True, **kw)


class TestLifecycleIocFallback:
    def test_flag_off_market_fallback_even_with_context(self, nosleep):
        assert order_utils.IOC_ENTRY_CAP_ENABLED is False  # default OFF
        api = _API()
        api.add('o1')
        result = _run_lifecycle(
            api, ioc_fallback={'quote_fn': lambda: dict(_IOC_QUOTE),
                               'cap_bps': None, 'asset_type': 'crypto'})
        markets = [s for s in api.submitted if s.get('type') == 'market']
        assert len(markets) == 1                      # today's behavior
        assert markets[0]['qty'] == pytest.approx(1.0)
        assert not any(s.get('time_in_force') == 'ioc' for s in api.submitted)
        assert result is not None

    def test_flag_on_submits_capped_ioc_never_market(self, nosleep,
                                                     monkeypatch):
        monkeypatch.setattr(order_utils, 'IOC_ENTRY_CAP_ENABLED', True)
        rows = []
        monkeypatch.setattr(trade_journal, 'log_decision', rows.append)
        api = _API()
        api.add('o1')
        result = _run_lifecycle(
            api, ioc_fallback={'quote_fn': lambda: dict(_IOC_QUOTE),
                               'cap_bps': None, 'asset_type': 'crypto'})
        assert not any(s.get('type') == 'market' for s in api.submitted)
        iocs = [s for s in api.submitted if s.get('time_in_force') == 'ioc']
        assert len(iocs) == 1
        cap = order_utils._entry_ioc_cap_bps('crypto')
        expected_limit = order_utils.ioc_limit_price(
            'buy', _IOC_QUOTE, cap, 'crypto')
        assert iocs[0]['type'] == 'limit'
        assert iocs[0]['limit_price'] == pytest.approx(expected_limit)
        assert iocs[0]['qty'] == pytest.approx(1.0)   # the remainder
        assert getattr(result, 'status', None) == 'filled'
        # measurement row for the new path
        j = [r for r in rows if r.get('action') == 'ioc_entry_fallback']
        assert len(j) == 1
        assert j[0]['cap_bps'] == cap
        assert j[0]['symbol'] == 'BTC/USD'

    def test_flag_on_no_quote_no_chase_fail_closed(self, nosleep,
                                                   monkeypatch):
        monkeypatch.setattr(order_utils, 'IOC_ENTRY_CAP_ENABLED', True)
        api = _API()
        api.add('o1')
        result = _run_lifecycle(
            api, ioc_fallback={'quote_fn': lambda: None,
                               'cap_bps': None, 'asset_type': 'crypto'})
        assert api.submitted == []                    # NO submit at all
        assert result is api.orders['o1']             # pre-fallback state

    def test_stock_unparseable_qty_fails_closed(self, nosleep, monkeypatch):
        # Broker qty can be a raw string ('1.5' via the saved_qty fallback);
        # unparseable-to-int ⇒ NO chase, pre-fallback state returned.
        monkeypatch.setattr(trade_journal, 'log_decision', lambda row: None)
        api = _API()
        sentinel = SimpleNamespace(status='canceled', filled_qty=0.0)
        out = order_utils._ioc_entry_fallback(
            api, 'AAPL', 'buy', 'not-a-qty',
            {'quote_fn': lambda: dict(_IOC_QUOTE), 'cap_bps': None,
             'asset_type': 'stock'}, sentinel)
        assert out is sentinel
        assert api.submitted == []
        # float-string qty parses and truncates to int shares
        out2 = order_utils._ioc_entry_fallback(
            api, 'AAPL', 'buy', '2.0',
            {'quote_fn': lambda: dict(_IOC_QUOTE), 'cap_bps': None,
             'asset_type': 'stock'}, sentinel)
        assert len(api.submitted) == 1
        assert api.submitted[0]['qty'] == 2

    def test_flag_on_exits_still_market_fallback(self, nosleep, monkeypatch):
        # Exits never pass ioc_fallback — a liquidation is never capped.
        monkeypatch.setattr(order_utils, 'IOC_ENTRY_CAP_ENABLED', True)
        api = _API()
        api.add('o1')
        _run_lifecycle(api)                           # ioc_fallback=None
        assert any(s.get('type') == 'market' for s in api.submitted)
        assert not any(s.get('time_in_force') == 'ioc' for s in api.submitted)


# ---------------------------------------------------------------------------
# T5 — place_maker_buy entry_fills journaling
# ---------------------------------------------------------------------------

def _read_entry_fills(tmp_path):
    files = list(Path(tmp_path).glob('*.jsonl'))
    lines = []
    for f in files:
        lines += [ln for ln in f.read_text().splitlines() if ln.strip()]
    fills = []
    for ln in lines:
        row = json.loads(ln)
        if row.get('action') == 'entry_fills':
            assert '"entry_tactic"' not in ln     # fees count-scan isolation
            assert row.get('action') != 'buy'
            fills.append(row)
    return fills


class TestMakerBuyJournaling:
    def test_filled_first_rung_journals_maker_notional(self, monkeypatch,
                                                       tmp_path):
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', Path(tmp_path))
        # Lift the under-pytest suppression — this test asserts real rows
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        api = _API()
        monkeypatch.setattr(
            order_utils, 'manage_order_lifecycle',
            lambda *a, **k: SimpleNamespace(status='filled', filled_qty=10.0,
                                            filled_avg_price=100.0))
        quote_fn = lambda: {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
                            'spread_pct': 0.2}
        result, tactic = order_utils.place_maker_buy(api, 'BTC/USD', 1000,
                                                     quote_fn)
        assert tactic == 'maker'
        fills = _read_entry_fills(tmp_path)
        assert len(fills) == 1
        assert fills[0]['tactic'] == 'maker'
        assert fills[0]['maker_notional'] == pytest.approx(1000.0)
        assert fills[0]['taker_notional'] == 0.0

    def test_partial_rung_then_taker_fallback_splits(self, monkeypatch,
                                                     tmp_path):
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', Path(tmp_path))
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        api = _API()
        results = [
            SimpleNamespace(status='canceled', filled_qty=5.0,
                            filled_avg_price=100.0),   # rung 1: half filled
            SimpleNamespace(status='canceled', filled_qty=0.0,
                            filled_avg_price=None),    # rung 2: nothing
            SimpleNamespace(status='filled', filled_qty=5.0,
                            filled_avg_price=100.0),   # taker fallback
        ]
        monkeypatch.setattr(order_utils, 'manage_order_lifecycle',
                            lambda *a, **k: results.pop(0))
        quote_fn = lambda: {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
                            'spread_pct': 0.2}
        result, tactic = order_utils.place_maker_buy(api, 'BTC/USD', 1000,
                                                     quote_fn)
        assert tactic == 'taker_fallback'
        fills = _read_entry_fills(tmp_path)
        assert len(fills) == 1
        assert fills[0]['tactic'] == 'taker_fallback'
        assert fills[0]['maker_notional'] == pytest.approx(500.0)
        assert fills[0]['taker_notional'] == pytest.approx(500.0)

    def test_suppressed_under_pytest_writes_no_rows(self, monkeypatch,
                                                    tmp_path):
        # PYTEST_CURRENT_TEST is set by pytest itself: a suite run must not
        # seed the live journal with synthetic entry_fills rows (they would
        # count toward arming TRADER_MAKER_SHARE_NOTIONAL on the Jetson).
        assert 'PYTEST_CURRENT_TEST' in _os.environ
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', Path(tmp_path))
        api = _API()
        monkeypatch.setattr(
            order_utils, 'manage_order_lifecycle',
            lambda *a, **k: SimpleNamespace(status='filled', filled_qty=10.0,
                                            filled_avg_price=100.0))
        order_utils.place_maker_buy(
            api, 'BTC/USD', 1000,
            lambda: {'bid': 100.0, 'ask': 100.2, 'midpoint': 100.1,
                     'spread_pct': 0.2})
        assert _read_entry_fills(tmp_path) == []

    def test_fees_count_scan_never_sees_entry_fills_rows(self, monkeypatch,
                                                         tmp_path):
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', Path(tmp_path))
        monkeypatch.setattr(fees, '_maker_share_cache', None)
        for _ in range(40):
            trade_journal.log_decision(
                {'action': 'entry_fills', 'symbol': 'BTC/USD',
                 'tactic': 'maker', 'maker_notional': 100.0,
                 'taker_notional': 0.0})
        assert fees.realized_crypto_maker_share() is None   # not entries


# ---------------------------------------------------------------------------
# T6 — realized_crypto_maker_share_notional
# ---------------------------------------------------------------------------

def _write_fills(tmp_path, rows, day_offset=0):
    day = (_dt.date.today() - _dt.timedelta(days=day_offset)).isoformat()
    path = Path(tmp_path) / f"{day}.jsonl"
    with open(path, 'a') as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _fill_row(maker=2.0, taker=1.0, symbol='BTC/USD'):
    return {'action': 'entry_fills', 'symbol': symbol, 'tactic': 'maker',
            'maker_notional': maker, 'taker_notional': taker}


class TestNotionalShare:
    def _reset(self, monkeypatch, tmp_path):
        monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', Path(tmp_path))
        monkeypatch.setattr(order_utils, '_maker_share_notional_cache', None)

    def test_thin_sample_none(self, monkeypatch, tmp_path):
        self._reset(monkeypatch, tmp_path)
        _write_fills(tmp_path,
                     [_fill_row()] * (fees.MAKER_SHARE_MIN_ENTRIES - 1))
        assert order_utils.realized_crypto_maker_share_notional() is None

    def test_exact_weighted_share(self, monkeypatch, tmp_path):
        self._reset(monkeypatch, tmp_path)
        _write_fills(tmp_path,
                     [_fill_row(2.0, 1.0)] * fees.MAKER_SHARE_MIN_ENTRIES)
        share = order_utils.realized_crypto_maker_share_notional()
        assert share == pytest.approx(2.0 / 3.0)

    def test_malformed_and_noncrypto_rows_skipped(self, monkeypatch,
                                                  tmp_path):
        self._reset(monkeypatch, tmp_path)
        rows = ([_fill_row(2.0, 1.0)] * fees.MAKER_SHARE_MIN_ENTRIES
                + [_fill_row(maker='garbage')]           # malformed: skipped
                + [_fill_row(999.0, 0.0, symbol='AAPL')])  # non-crypto
        _write_fills(tmp_path, rows)
        share = order_utils.realized_crypto_maker_share_notional()
        assert share == pytest.approx(2.0 / 3.0)

    def test_cache_honored_inside_ttl(self, monkeypatch, tmp_path):
        self._reset(monkeypatch, tmp_path)
        _write_fills(tmp_path,
                     [_fill_row(2.0, 1.0)] * fees.MAKER_SHARE_MIN_ENTRIES)
        first = order_utils.realized_crypto_maker_share_notional()
        assert first == pytest.approx(2.0 / 3.0)
        # Mutate the journal — the cached value must survive inside the TTL
        _write_fills(tmp_path, [_fill_row(0.0, 1000.0)] * 50)
        assert (order_utils.realized_crypto_maker_share_notional()
                == pytest.approx(2.0 / 3.0))


# ---------------------------------------------------------------------------
# T7 — should_trade consumption
# ---------------------------------------------------------------------------

class TestShouldTradeNotional:
    @pytest.fixture(autouse=True)
    def _det(self, monkeypatch):
        # Deterministic count share: None => live blend prices full taker
        monkeypatch.setattr(fees, 'realized_crypto_maker_share',
                            lambda *a, **k: None)

    def _t0(self, spread=0.1):
        return fees.required_edge_pct('crypto', spread, False, None,
                                      live=True)

    def test_flag_off_identity(self):
        assert order_utils.MAKER_SHARE_NOTIONAL_ENABLED is False  # default
        t0 = self._t0()
        assert order_utils.should_trade(t0 + 0.01, 0.1) is True
        assert order_utils.should_trade(t0 - 0.01, 0.1) is False

    def test_flag_on_exact_fee_delta_shift(self, monkeypatch):
        monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', True)
        monkeypatch.setattr(order_utils,
                            'realized_crypto_maker_share_notional',
                            lambda: 1.0)   # full maker
        t0 = self._t0()
        shift = fees.MIN_EDGE_MULTIPLE * (
            fees.CRYPTO_MAKER_BPS - fees.CRYPTO_TAKER_BPS) / 100.0
        assert shift < 0
        # Threshold moved to exactly t0 + shift (strict > comparison)
        assert order_utils.should_trade(t0 + shift + 0.01, 0.1) is True
        assert order_utils.should_trade(t0 + shift - 0.01, 0.1) is False
        # Still below the shifted threshold: rejected
        assert order_utils.should_trade(t0 + 2 * shift, 0.1) is False

    def test_flag_on_none_share_noop(self, monkeypatch):
        monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', True)
        monkeypatch.setattr(order_utils,
                            'realized_crypto_maker_share_notional',
                            lambda: None)
        t0 = self._t0()
        assert order_utils.should_trade(t0 + 0.01, 0.1) is True
        assert order_utils.should_trade(t0 - 0.01, 0.1) is False

    def test_stock_and_maker_paths_unaffected(self, monkeypatch):
        monkeypatch.setattr(order_utils, 'MAKER_SHARE_NOTIONAL_ENABLED', True)
        calls = []
        monkeypatch.setattr(order_utils,
                            'realized_crypto_maker_share_notional',
                            lambda: calls.append(1) or 1.0)
        order_utils.should_trade(1.0, 0.05, asset_type='stock')
        order_utils.should_trade(1.0, 0.1, asset_type='crypto', maker=True)
        assert calls == []


# ---------------------------------------------------------------------------
# T8 — _classify_server_stop
# ---------------------------------------------------------------------------

def _pos(**over):
    kw = dict(qty=1.0, entry_price=100.0, high_water_mark=100.0,
              entry_atr=3.0)   # stop_dist = 6/100 = 0.06 -> hard = 94.0
    kw.update(over)
    return Position(**kw)


class TestClassifyServerStop:
    def test_stop_above_entry_is_trail(self, tmp_path):
        inst = _mk(tmp_path)
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(), SimpleNamespace(stop_price='101.0'))
        assert (kind, px) == ('trail', 101.0)

    def test_ratcheted_above_hard_level_is_trail(self, tmp_path):
        inst = _mk(tmp_path)
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(), SimpleNamespace(stop_price=97.0))
        assert (kind, px) == ('trail', 97.0)   # 97 > 94*(1+1e-3)

    def test_at_hard_level_is_hard(self, tmp_path):
        inst = _mk(tmp_path)
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(), SimpleNamespace(stop_price=94.0 * 0.999))
        assert kind == 'hard'
        assert px == pytest.approx(94.0 * 0.999)

    def test_resting_stop_px_cache_fallback(self, tmp_path):
        inst = _mk(tmp_path)
        inst._resting_stop_px = {'BTC/USD': 101.0}   # crypto_loop cache
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(), SimpleNamespace(stop_price=None))
        assert (kind, px) == ('trail', 101.0)

    def test_no_evidence_unknown(self, tmp_path):
        inst = _mk(tmp_path)
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(), SimpleNamespace(stop_price=None))
        assert (kind, px) == ('unknown', None)

    def test_trailing_activated_wins_first(self, tmp_path):
        inst = _mk(tmp_path)
        kind, px = inst._classify_server_stop(
            'BTC/USD', _pos(trailing_activated=True),
            SimpleNamespace(stop_price=None))
        assert kind == 'trail'

    def test_garbage_stop_price_never_raises(self, tmp_path):
        inst = _mk(tmp_path)
        for garbage in ('abc', object()):
            kind, px = inst._classify_server_stop(
                'BTC/USD', _pos(), SimpleNamespace(stop_price=garbage))
            assert kind == 'unknown'


# ---------------------------------------------------------------------------
# T9 — _apply_server_stop_lockout
# ---------------------------------------------------------------------------

class TestServerStopLockout:
    def test_flag_off_locks_trail_byte_compat(self, monkeypatch, tmp_path):
        monkeypatch.setattr(base_loop, 'STOP_CLASSIFY_V2', False)
        inst = _mk(tmp_path)
        inst._apply_server_stop_lockout('BTC/USD', 'trail')
        assert 'BTC/USD' in inst.hard_stop_lockout
        with open(inst._lockout_file) as f:
            assert 'BTC/USD' in json.load(f)   # persisted (tmp file only)

    def test_flag_on_trail_exempt_hard_unknown_locked(self, monkeypatch,
                                                      tmp_path):
        monkeypatch.setattr(base_loop, 'STOP_CLASSIFY_V2', True)
        inst = _mk(tmp_path)
        inst._apply_server_stop_lockout('BTC/USD', 'trail')
        assert 'BTC/USD' not in inst.hard_stop_lockout
        inst._apply_server_stop_lockout('BTC/USD', 'hard')
        assert 'BTC/USD' in inst.hard_stop_lockout
        inst2 = _mk(tmp_path)
        inst2._apply_server_stop_lockout('ETH/USD', 'unknown')
        assert 'ETH/USD' in inst2.hard_stop_lockout


# ---------------------------------------------------------------------------
# T10 — _stream_stop_fallback
# ---------------------------------------------------------------------------

class TestStreamStopFallback:
    def test_flag_off_none_even_with_cached_fill(self, monkeypatch,
                                                 tmp_path):
        monkeypatch.setattr(base_loop, 'STREAM_STOP_DETECT', False)
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled',
                                         'filled_qty': 1.0,
                                         'filled_avg_price': 95.0})
        assert _mk(tmp_path)._stream_stop_fallback('o1') is None

    def test_flag_on_cached_filled_recovers(self, monkeypatch, tmp_path):
        monkeypatch.setattr(base_loop, 'STREAM_STOP_DETECT', True)
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled',
                                         'filled_qty': 1.0,
                                         'filled_avg_price': 95.0})
        so = _mk(tmp_path)._stream_stop_fallback('o1')
        assert so is not None
        assert so.status == 'filled'
        assert so.filled_avg_price == 95.0
        assert so.stop_price is None

    def test_flag_on_non_terminal_absent_raising_all_none(self, monkeypatch,
                                                          tmp_path):
        monkeypatch.setattr(base_loop, 'STREAM_STOP_DETECT', True)
        inst = _mk(tmp_path)
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'new'})
        assert inst._stream_stop_fallback('o1') is None
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: None)
        assert inst._stream_stop_fallback('o1') is None

        def _boom(oid):
            raise RuntimeError('cache down')
        monkeypatch.setattr(order_stream, 'get_order_state', _boom)
        assert inst._stream_stop_fallback('o1') is None


# ---------------------------------------------------------------------------
# T11 — crypto _manage_stops integration
# ---------------------------------------------------------------------------

def _stops_env(monkeypatch):
    trades, rows = [], []
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    return trades, rows


class _RaisingAPI:
    def get_order(self, oid):
        raise RuntimeError('REST down')


class TestManageStopsIntegration:
    def test_rest_raise_flag_on_stream_fill_removes_position(
            self, monkeypatch, tmp_path):
        trades, rows = _stops_env(monkeypatch)
        monkeypatch.setattr(base_loop, 'STREAM_STOP_DETECT', True)
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled',
                                         'filled_qty': 1.0,
                                         'filled_avg_price': 95.0})
        pos = _pos(stop_order_id='so1')
        inst = _mk(tmp_path, api=_RaisingAPI(),
                   positions={'BTC/USD': pos})
        inst._manage_stops()
        assert 'BTC/USD' not in inst.positions
        sells = [r for r in rows if r.get('action') == 'sell']
        assert len(sells) == 1
        assert sells[0]['exit_reason'] == 'server_stop'
        assert sells[0]['detect_source'] == 'stream'
        assert 'server_stop_kind' in sells[0]
        assert sells[0]['server_stop_kind'] == 'unknown'  # no px evidence

    def test_rest_raise_flag_off_position_retained(self, monkeypatch,
                                                   tmp_path):
        trades, rows = _stops_env(monkeypatch)
        monkeypatch.setattr(base_loop, 'STREAM_STOP_DETECT', False)
        monkeypatch.setattr(order_stream, 'get_order_state',
                            lambda oid: {'status': 'filled',
                                         'filled_qty': 1.0,
                                         'filled_avg_price': 95.0})
        pos = _pos(stop_order_id='so1')
        inst = _mk(tmp_path, api=_RaisingAPI(),
                   positions={'BTC/USD': pos})
        inst._manage_stops()   # today's behavior: retry next cycle
        assert 'BTC/USD' in inst.positions
        assert [r for r in rows if r.get('action') == 'sell'] == []

    def test_rest_happy_path_always_journals_kind(self, monkeypatch,
                                                  tmp_path):
        trades, rows = _stops_env(monkeypatch)
        filled = SimpleNamespace(status='filled', filled_qty=1.0,
                                 filled_avg_price='94.0', stop_price='94.0')
        api = SimpleNamespace(get_order=lambda oid: filled)
        pos = Position(qty=1.0, entry_price=100.0, high_water_mark=100.0,
                       stop_order_id='so1')   # entry_atr None -> dist 0.05
        inst = _mk(tmp_path, api=api, positions={'BTC/USD': pos})
        inst._manage_stops()
        assert 'BTC/USD' not in inst.positions
        assert 'BTC/USD' in inst.hard_stop_lockout   # flag OFF: locked
        sells = [r for r in rows if r.get('action') == 'sell']
        assert len(sells) == 1
        assert sells[0]['server_stop_kind'] == 'hard'   # 94 < 95*(1+1e-3)
        assert sells[0]['stop_px'] == 94.0
        assert 'detect_source' not in sells[0]           # REST, not stream


# ---------------------------------------------------------------------------
# T12 — _record_confirmed_exit
# ---------------------------------------------------------------------------

class TestRecordConfirmedExit:
    def test_default_call_legacy_keyset_plus_quote_age(self, monkeypatch,
                                                       tmp_path):
        trades, rows = _stops_env(monkeypatch)
        inst = _mk(tmp_path)
        inst._record_confirmed_exit(
            'BTC/USD', _pos(), SimpleNamespace(filled_avg_price='99.0'),
            None, exit_reason='signal_sell')
        assert len(rows) == 1
        assert set(rows[0].keys()) == {
            'symbol', 'action', 'exit_reason', 'pnl_pct', 'decision_price',
            'fill_price', 'slippage_bps', 'quote_age_s', 'estimated'}
        assert rows[0]['quote_age_s'] is None

    def test_quote_fetched_ts_computes_age(self, monkeypatch, tmp_path):
        trades, rows = _stops_env(monkeypatch)
        inst = _mk(tmp_path)
        quote = {'midpoint': 100.0, 'fetched_ts': _time.time() - 5.0}
        inst._record_confirmed_exit(
            'BTC/USD', _pos(), SimpleNamespace(filled_avg_price='99.0'),
            quote, exit_reason='signal_sell')
        assert rows[0]['quote_age_s'] == pytest.approx(5.0, abs=1.0)
        assert rows[0]['decision_price'] == 100.0

    def test_extra_merges_additive_keys(self, monkeypatch, tmp_path):
        trades, rows = _stops_env(monkeypatch)
        inst = _mk(tmp_path)
        inst._record_confirmed_exit(
            'BTC/USD', _pos(), SimpleNamespace(filled_avg_price='99.0'),
            None, exit_reason='server_stop',
            extra={'server_stop_kind': 'trail', 'stop_px': 101.0})
        assert rows[0]['server_stop_kind'] == 'trail'
        assert rows[0]['stop_px'] == 101.0
        assert rows[0]['exit_reason'] == 'server_stop'


# ---------------------------------------------------------------------------
# T13 — _journal_cycle_latency
# ---------------------------------------------------------------------------

class TestCycleLatency:
    _PHASES = dict(stops_s=0.1, maint_s=0.2, fetch_s=0.0, predict_s=1.0,
                   llm_s=0.0, sells_s=0.05, buys_s=0.0)

    def test_row_shape(self, monkeypatch, tmp_path):
        rows = []
        monkeypatch.setattr(base_loop, 'log_decision', rows.append)
        inst = _mk(tmp_path, cycle=7)
        inst._journal_cycle_latency(**self._PHASES)
        assert len(rows) == 1
        row = rows[0]
        assert row['action'] == 'cycle_latency'
        assert row['book'] == 'crypto'
        assert row['cycle'] == 7
        for k, v in self._PHASES.items():
            assert row[k] == round(v, 3)
        assert row['total_s'] == pytest.approx(1.35)

    def test_never_raises_when_journal_broken(self, monkeypatch, tmp_path):
        def _boom(row):
            raise RuntimeError('disk full')
        monkeypatch.setattr(base_loop, 'log_decision', _boom)
        inst = _mk(tmp_path)
        inst._journal_cycle_latency(**self._PHASES)   # must not raise


# ---------------------------------------------------------------------------
# stock_loop mirror pin (source-level: the hand-synced branch uses the
# shared base methods, not a private copy)
# ---------------------------------------------------------------------------

def test_stock_manage_stops_uses_shared_base_methods():
    src = inspect.getsource(stock_loop.StockLoop._manage_stops)
    assert '_stream_stop_fallback' in src
    assert '_apply_server_stop_lockout' in src
    assert '_classify_server_stop' in src
    # inherited, not overridden
    assert 'StockLoop' not in str(stock_loop.StockLoop._classify_server_stop)
    assert stock_loop.StockLoop._classify_server_stop is \
        base_loop.BaseTradingLoop._classify_server_stop
