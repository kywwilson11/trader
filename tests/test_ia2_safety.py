"""IA-2 — undesigned-behavior + safety fixes (2026-08 influence audit).

Pins the four ledger-prescribed repairs from
research/campaign_2026-08/07_decision_influences.md:

1. Exits are NEVER cooldown-gated (§3.4 "Cooldown + trade budget": an
   entry throttle delaying risk reduction inverts the tool's purpose) —
   base_loop._execute_sells, base_loop._execute_llm_veto_sells and
   stock_loop._execute_sells proceed regardless of cooldown state and
   journal a cooldown_bypassed_exit marker (measurement only). The
   ENTRY-side cooldown gate is unchanged.
2. Per-book hard-stop lockout state file (§3.4 "shared unprefixed file
   lets books clobber each other"): {prefix}_hard_stop_lockout.json with
   a one-time legacy-file migration read; saves write only the book's
   own file.
3. Quote-staleness guard under the alpaca_compat shim: the census claim
   ("timestamp dropped") is ALREADY fixed in the tree (_shim_quote
   threads t); pinned here — stale quotes are rejected under the shim
   shape; absent timestamps fail CLOSED (rejected; ENGINE r3 W10 — they
   were accepted before).
4. FOMC/CPI static-table staleness alarm (§3.2): one loud daily warning
   + one notify per process when the table is exhausted; fail-open (gate
   behavior unchanged).

Stub pattern mirrors tests/test_c26_base_loop_functional.py: real
`import base_loop`/`import stock_loop` under dotenv/predict_now stubs,
sys.modules restored afterward.
"""

import datetime as _dt
import inspect
import json
import logging
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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

import macro_calendar          # Mac-importable (stdlib only)
import notify                  # halt/notify seams (function-local imports)
import order_utils             # Mac-importable; get_quote staleness guard
import alpaca_compat           # Mac-importable (alpaca imports are lazy)
from types_mod import Position


# ---------------------------------------------------------------------------
# Concrete subclass + minimal factory (functional-suite pattern)
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

    def place_sell_order(self, symbol, qty, quote):
        self._sold.append(symbol)
        return SimpleNamespace(filled_avg_price='99.0')

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
    inst.model = object(); inst.config = {}
    inst.trade_threshold = 0.15
    inst.positions = {}; inst.last_trade_time = {}
    inst.hard_stop_lockout = {}
    inst._lockout_file = str(tmp_path / 'hard_stop_lockout.json')
    inst._legacy_lockout_file = str(tmp_path / 'hard_stop_lockout.json')
    inst.llm_scores = {}; inst._veto_strikes = {}
    inst.cycle = 1
    inst.macro_regime = None; inst.corr_matrix = {}
    inst._macro_cal_alarm_date = None
    inst._macro_cal_alarm_notified = False
    inst._universe = ['BTC/USD']
    inst._quotes = {'BTC/USD': {'bid': 99.9, 'ask': 100.1, 'spread': 0.2,
                                'midpoint': 100.0, 'spread_pct': 0.2}}
    inst._sold = []
    inst._save_position_state = lambda: None
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


def _pos(entry=100.0):
    return Position(qty=1.0, entry_price=entry, high_water_mark=entry)


@pytest.fixture
def quiet(monkeypatch):
    """No real sleeps, no real journal writes; capture journal rows."""
    rows, trades = [], []
    monkeypatch.setattr(base_loop.time, 'sleep', lambda s: None)
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    return rows, trades


# ---------------------------------------------------------------------------
# 1. Exits are never cooldown-gated
# ---------------------------------------------------------------------------

class TestExitCooldownBypass:
    def test_signal_sell_proceeds_under_cooldown_with_marker(
            self, tmp_path, quiet):
        rows, trades = quiet
        inst = _mk(tmp_path,
                   positions={'BTC/USD': _pos()},
                   last_trade_time={'BTC/USD': _dt.datetime.now()})
        inst._execute_sells({'BTC/USD': -1.0})
        assert inst._sold == ['BTC/USD']          # sale executed in cooldown
        assert 'BTC/USD' not in inst.positions
        assert len(trades) == 1
        row, = rows
        assert row['action'] == 'sell'
        assert row['exit_reason'] == 'signal_sell'
        assert row['cooldown_bypassed_exit'] is True

    def test_signal_sell_no_marker_outside_cooldown(self, tmp_path, quiet):
        rows, _ = quiet
        inst = _mk(tmp_path, positions={'BTC/USD': _pos()},
                   last_trade_time={})
        inst._execute_sells({'BTC/USD': -1.0})
        assert inst._sold == ['BTC/USD']
        row, = rows
        assert 'cooldown_bypassed_exit' not in row   # legacy key-set intact

    def test_llm_veto_sell_proceeds_under_cooldown_with_marker(
            self, tmp_path, quiet):
        rows, _ = quiet
        inst = _mk(tmp_path,
                   positions={'BTC/USD': _pos()},
                   last_trade_time={'BTC/USD': _dt.datetime.now()},
                   llm_scores={'BTC/USD': {'s': 0.05, 'r': 'hack'}},
                   _veto_strikes={'BTC/USD': 2})
        inst._execute_llm_veto_sells()
        assert inst._sold == ['BTC/USD']
        assert 'BTC/USD' not in inst.positions
        row, = rows
        assert row['exit_reason'] == 'llm_veto'
        assert row['cooldown_bypassed_exit'] is True

    def test_llm_veto_strike_rule_still_holds(self, tmp_path, quiet):
        # 2-strike liquidation protocol untouched: one veto never sells.
        inst = _mk(tmp_path,
                   positions={'BTC/USD': _pos()},
                   llm_scores={'BTC/USD': {'s': 0.05, 'r': 'x'}},
                   _veto_strikes={'BTC/USD': 1})
        inst._execute_llm_veto_sells()
        assert inst._sold == []
        assert 'BTC/USD' in inst.positions

    def test_stock_signal_sell_proceeds_under_cooldown_with_marker(
            self, monkeypatch):
        recorded = []
        _tu = types.ModuleType('trading_utils')
        _tu.cooldown_ok = lambda *a, **k: False       # in cooldown
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu)
        monkeypatch.setattr(stock_loop, 'cancel_orders_for_symbol',
                            lambda *a, **k: True)
        monkeypatch.setattr(stock_loop.time, 'sleep', lambda s: None)
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0, qty=5)
        me = SimpleNamespace(
            api=SimpleNamespace(get_position=lambda s: SimpleNamespace(qty='5')),
            positions={'AAPL': info},
            hold_symbols={'AAPL'},
            trade_threshold=0.15,
            last_trade_time={'AAPL': _dt.datetime.now()},
            COOLDOWN_MINUTES=60,
            HOLD_RANK=15,
            llm_scores={},
            get_quote=lambda s: {'midpoint': 100.0},
            place_sell_order=lambda s, q, quote: SimpleNamespace(
                filled_avg_price='99.0'),
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)),
            _journal_external_close=lambda *a, **k: None,
        )
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})
        (args, kwargs), = recorded
        assert kwargs['exit_reason'] == 'signal_sell'
        assert kwargs['extra'] == {'cooldown_bypassed_exit': True}
        assert 'AAPL' not in me.positions

    def test_stock_signal_sell_no_marker_outside_cooldown(self, monkeypatch):
        recorded = []
        _tu = types.ModuleType('trading_utils')
        _tu.cooldown_ok = lambda *a, **k: True        # not in cooldown
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu)
        monkeypatch.setattr(stock_loop, 'cancel_orders_for_symbol',
                            lambda *a, **k: True)
        monkeypatch.setattr(stock_loop.time, 'sleep', lambda s: None)
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0, qty=5)
        me = SimpleNamespace(
            api=SimpleNamespace(get_position=lambda s: SimpleNamespace(qty='5')),
            positions={'AAPL': info}, hold_symbols={'AAPL'},
            trade_threshold=0.15, last_trade_time={}, COOLDOWN_MINUTES=60,
            HOLD_RANK=15, llm_scores={},
            get_quote=lambda s: {'midpoint': 100.0},
            place_sell_order=lambda s, q, quote: SimpleNamespace(
                filled_avg_price='99.0'),
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)),
            _journal_external_close=lambda *a, **k: None,
        )
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})
        (args, kwargs), = recorded
        assert kwargs['extra'] is None

    def test_exit_paths_not_cooldown_gated_source(self):
        for cls, meth in ((base_loop.BaseTradingLoop, '_execute_sells'),
                          (base_loop.BaseTradingLoop, '_execute_llm_veto_sells'),
                          (stock_loop.StockLoop, '_execute_sells')):
            src = inspect.getsource(getattr(cls, meth))
            assert 'cooldown_bypassed' in src, (cls, meth)
            assert 'if not cooldown_ok' not in src, (cls, meth)

    def test_entry_cooldown_gate_unchanged(self):
        bsrc = inspect.getsource(base_loop.BaseTradingLoop._execute_buys)
        assert ("if not cooldown_ok(self.last_trade_time, symbol, "
                "self.COOLDOWN_MINUTES):") in bsrc
        assert "vc['cooldown'] += 1" in bsrc
        ssrc = inspect.getsource(stock_loop.StockLoop._execute_buys)
        assert ("if not cooldown_ok(self.last_trade_time, symbol, "
                "self.COOLDOWN_MINUTES):") in ssrc
        assert "vc['cooldown'] += 1" in ssrc


# ---------------------------------------------------------------------------
# 1b. Stock sentiment gate<=0 dead limb removed (deferred to this packet by
#     IA-1 — same unreachable branch IA-1 deleted from base_loop; the
#     unreachability proof itself is pinned in tests/test_ia1_removals.py,
#     which sweeps sentiment_gate's [0.15, 1.5] clamp for BOTH asset types)
# ---------------------------------------------------------------------------

class TestStockSentimentDeadLimbRemoved:
    def test_dead_branch_removed_multiplier_survives(self):
        src = inspect.getsource(stock_loop.StockLoop._execute_buys)
        assert 'if gate <= 0:' not in src
        assert "'sentiment_block'" not in src
        # Multiplier path intact: gate still fetched and fed into sizing.
        assert "sentiment_gate(symbol, 'stock')" in src
        assert 'sentiment_mult=gate' in src


# ---------------------------------------------------------------------------
# 2. Per-book lockout state file + legacy migration
# ---------------------------------------------------------------------------

def _write_lockout(path, symbols, hours_left=23.0):
    expiry = (_dt.datetime.now()
              + _dt.timedelta(hours=hours_left)).timestamp()
    with open(path, 'w') as f:
        json.dump({s: expiry for s in symbols}, f)


class TestLockoutPerBook:
    def test_init_computes_per_book_filename(self):
        src = inspect.getsource(base_loop.BaseTradingLoop.__init__)
        assert "f'{self.MODEL_PREFIX}_hard_stop_lockout.json'" in src
        assert "_legacy_lockout_file" in src

    def test_migration_read_from_legacy_then_prefixed_write(self, tmp_path):
        legacy = tmp_path / 'hard_stop_lockout.json'
        prefixed = tmp_path / 'stock_hard_stop_lockout.json'
        _write_lockout(legacy, ['AAPL'])
        legacy_before = legacy.read_text()
        inst = _mk(tmp_path, MODEL_PREFIX='stock',
                   _lockout_file=str(prefixed),
                   _legacy_lockout_file=str(legacy))
        inst._load_hard_stop_lockout()
        assert 'AAPL' in inst.hard_stop_lockout       # migrated in
        inst._save_hard_stop_lockout()
        assert prefixed.exists()                       # writes prefixed only
        assert set(json.loads(prefixed.read_text())) == {'AAPL'}
        assert legacy.read_text() == legacy_before     # legacy untouched

    def test_prefixed_file_preferred_over_legacy(self, tmp_path):
        legacy = tmp_path / 'hard_stop_lockout.json'
        prefixed = tmp_path / 'stock_hard_stop_lockout.json'
        _write_lockout(legacy, ['BTC/USD'])
        _write_lockout(prefixed, ['TSLA'])
        inst = _mk(tmp_path, MODEL_PREFIX='stock',
                   _lockout_file=str(prefixed),
                   _legacy_lockout_file=str(legacy))
        inst._load_hard_stop_lockout()
        assert set(inst.hard_stop_lockout) == {'TSLA'}   # no legacy merge

    def test_crypto_same_path_behavior_unchanged(self, tmp_path):
        legacy = tmp_path / 'hard_stop_lockout.json'
        _write_lockout(legacy, ['BTC/USD'])
        inst = _mk(tmp_path)   # _lockout_file == _legacy_lockout_file
        inst._load_hard_stop_lockout()
        assert set(inst.hard_stop_lockout) == {'BTC/USD'}

    def test_books_no_longer_clobber_each_other(self, tmp_path):
        crypto = _mk(tmp_path)
        stock = _mk(tmp_path, MODEL_PREFIX='stock',
                    _lockout_file=str(tmp_path / 'stock_hard_stop_lockout.json'),
                    _legacy_lockout_file=str(tmp_path / 'hard_stop_lockout.json'))
        crypto.hard_stop_lockout = {'BTC/USD': _dt.datetime.now()}
        stock.hard_stop_lockout = {'TSLA': _dt.datetime.now()}
        crypto._save_hard_stop_lockout()
        stock._save_hard_stop_lockout()
        crypto._save_hard_stop_lockout()   # crypto save after stock save
        assert set(json.loads(
            (tmp_path / 'stock_hard_stop_lockout.json').read_text())) == {'TSLA'}
        assert set(json.loads(
            (tmp_path / 'hard_stop_lockout.json').read_text())) == {'BTC/USD'}

    def test_missing_both_files_starts_empty(self, tmp_path):
        inst = _mk(tmp_path, MODEL_PREFIX='stock',
                   _lockout_file=str(tmp_path / 'stock_hard_stop_lockout.json'),
                   _legacy_lockout_file=str(tmp_path / 'hard_stop_lockout.json'))
        inst._load_hard_stop_lockout()
        assert inst.hard_stop_lockout == {}


# ---------------------------------------------------------------------------
# 3. Quote-staleness guard under the alpaca_compat shim shape
# ---------------------------------------------------------------------------

def _api_returning(q):
    return SimpleNamespace(get_latest_quote=lambda s: q)


class TestQuoteStalenessUnderShim:
    def test_shim_threads_timestamp(self):
        ts = _dt.datetime.now(_dt.timezone.utc)
        shim = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=100.0, ask_price=100.1, timestamp=ts))
        assert shim.t is ts
        assert shim.bp == 100.0 and shim.ap == 100.1

    def test_shim_missing_timestamp_yields_none(self):
        shim = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=100.0, ask_price=100.1))
        assert shim.t is None

    def test_stale_shim_quote_rejected(self):
        stale = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=600)
        shim = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=100.0, ask_price=100.1, timestamp=stale))
        assert order_utils.get_quote(_api_returning(shim), 'TSLA',
                                     asset_type='stock') is None

    def test_fresh_shim_quote_accepted(self):
        fresh = _dt.datetime.now(_dt.timezone.utc)
        shim = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=100.0, ask_price=100.1, timestamp=fresh))
        q = order_utils.get_quote(_api_returning(shim), 'TSLA',
                                  asset_type='stock')
        assert q is not None
        assert q['midpoint'] == pytest.approx(100.05)

    def test_absent_timestamp_fails_closed(self):
        # Fail-closed contract (ENGINE r3 W10): a quote that cannot be aged
        # is not fresh — rejected like a stale one (was: accepted).
        shim = alpaca_compat._shim_quote(
            SimpleNamespace(bid_price=100.0, ask_price=100.1))
        q = order_utils.get_quote(_api_returning(shim), 'TSLA',
                                  asset_type='stock')
        assert q is None


# ---------------------------------------------------------------------------
# 4. FOMC/CPI static-table staleness alarm
# ---------------------------------------------------------------------------

class TestMacroCalendarAlarm:
    def test_calendar_exhausted_boundaries(self):
        last = max(max(macro_calendar.FOMC_STATEMENT_DAYS),
                   max(macro_calendar.CPI_RELEASE_DAYS))
        before = _dt.datetime(last[0], last[1], last[2], 12, 0,
                              tzinfo=macro_calendar._ET)
        after = before + _dt.timedelta(days=1)
        assert macro_calendar.calendar_exhausted(before) is False
        assert macro_calendar.calendar_exhausted(after) is True

    def _armed(self, tmp_path, monkeypatch, notified):
        monkeypatch.setattr(notify, 'halt_active', lambda: False)
        monkeypatch.setattr(notify, 'notify',
                            lambda msg, **k: notified.append(msg))
        monkeypatch.setattr(macro_calendar, 'macro_standdown',
                            lambda *a, **k: (False, None))
        monkeypatch.setattr(macro_calendar, 'calendar_exhausted',
                            lambda *a, **k: True)
        return _mk(tmp_path)

    def test_alarm_fail_open_daily_warning_notify_once(
            self, tmp_path, monkeypatch, caplog):
        notified = []
        inst = self._armed(tmp_path, monkeypatch, notified)
        with caplog.at_level(logging.WARNING, logger='base_loop'):
            assert inst._entries_allowed() is True     # fail-open: gate unchanged
        assert sum('STALE CALENDAR' in r.message
                   for r in caplog.records) == 1
        assert len(notified) == 1
        assert 'macro_calendar' in notified[0]
        # Same day, second cycle: no repeat warning, no repeat notify.
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='base_loop'):
            assert inst._entries_allowed() is True
        assert not any('STALE CALENDAR' in r.message for r in caplog.records)
        assert len(notified) == 1
        # Next day: warning fires again; notify still once per process.
        inst._macro_cal_alarm_date = (
            _dt.date.today() - _dt.timedelta(days=1)).isoformat()
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='base_loop'):
            assert inst._entries_allowed() is True
        assert sum('STALE CALENDAR' in r.message
                   for r in caplog.records) == 1
        assert len(notified) == 1

    def test_no_alarm_when_calendar_current(self, tmp_path, monkeypatch,
                                            caplog):
        notified = []
        inst = self._armed(tmp_path, monkeypatch, notified)
        monkeypatch.setattr(macro_calendar, 'calendar_exhausted',
                            lambda *a, **k: False)
        with caplog.at_level(logging.WARNING, logger='base_loop'):
            assert inst._entries_allowed() is True
        assert not any('STALE CALENDAR' in r.message for r in caplog.records)
        assert notified == []

    def test_standdown_still_blocks_entries(self, tmp_path, monkeypatch):
        notified = []
        inst = self._armed(tmp_path, monkeypatch, notified)
        monkeypatch.setattr(macro_calendar, 'macro_standdown',
                            lambda *a, **k: (True, 'CPI stand-down'))
        assert inst._entries_allowed() is False
