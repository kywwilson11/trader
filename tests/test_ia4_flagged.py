"""IA-4 — influence-audit flag family (2026-08 decision-influence ledger,
research/campaign_2026-08/07_decision_influences.md). Seven flag-gated
structural changes, each default-OFF with flag-OFF behavior byte-identical
(pinned here), plus one direct-ship ENB input fix:

1. VIX25_BLOCK_REMOVED — ON skips the VIX>25 non-safe-haven block at both
   sites (ledger §3.3 2-1 remove-candidate: double-charge + de facto halt).
2. CORR_FAMILY_MERGED — ON: ENB budget is the single correlation consumer;
   admission loosens to a CORR_SANITY_MAX sanity bar; f_corr composes at
   1.0 (ledger §3.3 MERGE). Plus the DIRECT covered-none rho fix:
   avg_book_correlation(uncovered=0.5) honors the no-data prior for a
   non-empty matrix covering none of the book's pairs.
3. TRADE_BUDGET_BACKSTOP — ON raises the daily budget to a runaway
   backstop (cap x mult), cooldown the one churn instrument (§3.4 MERGE).
4. CRYPTO_VERTICAL_BARRIER — fb-anchored crypto max-hold in the LOOP layer
   (policy_exits UNTOUCHED); would-fire journaled ALWAYS (§3.7 "the
   absence does not belong").
5. SIGNAL_EXIT_CONFIRM_READS — 1 (default) = today's single-reading exit;
   2 = stop-discipline two consecutive readings, armed/lapsed journaled
   (§3.7 "reconcile the 1-vs-2-reading asymmetry deliberately").
6. KELLY_SAMPLE_GATE — hold kelly_mult neutral until >=50 uncensored
   post-D06 trades/book (trading_utils.uncensored_trade_count) (§3.5).
7. BREAKER_PER_BOOK — per-book P&L attribution + crypto weekend-aware
   baseline window; OFF = account-wide check_circuit_breaker exactly as
   today (§3.2 KEEP-COND).

Stub pattern mirrors tests/test_c26_base_loop_functional.py.
"""

import datetime as _dt
import json
import sys
import time as _time
import types
import zoneinfo
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
    import trading_utils as _tu_real       # real module, dotenv stubbed
finally:
    for _m in ('dotenv', 'predict_now', 'trading_utils',
               'base_loop', 'stock_loop', 'crypto_loop'):
        sys.modules.pop(_m, None)

import strategy_config
import portfolio
import market_data          # get_live_atr / fetch seams (function-local)
import trade_journal        # stock funnel log_decision seam
from types_mod import Position

_UTC = _dt.timezone.utc


# ---------------------------------------------------------------------------
# Factories (test_c26_base_loop_functional pattern)
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


class _StockTyped(_Loop):
    def get_asset_type(self):
        return 'stock'


def _mk(cls=_Loop, **over):
    inst = object.__new__(cls)
    inst.api = None
    inst.model = object(); inst.config = {}; inst.scaler_X = None
    inst.feature_cols = None
    inst.trade_threshold = 0.15
    inst.positions = {}; inst.last_trade_time = {}
    inst.hard_stop_lockout = {}
    inst.llm_scores = {}; inst._veto_strikes = {}
    inst.model_mtime = 0; inst.cycle = 2
    inst.macro_regime = None; inst.corr_matrix = {}
    inst._equity = 100_000.0; inst._peak_equity = 100_000.0
    inst._peak_from_seed = False
    inst._buys_allowed = True; inst._halted_until = None
    inst._daily_trades = {}
    inst._daily_trades_date = _dt.date.today().isoformat()
    inst._pending_breach = {}
    inst._last_meta_p = {}; inst._last_meta_p_cycle = {}
    inst._leveraged_etfs = {}
    inst._last_sizing_detail = None
    inst._universe = ['BTC/USD']
    inst._quotes = {'BTC/USD': {'midpoint': 100.0, 'spread_pct': 0.02}}
    inst._save_position_state = lambda: None
    inst._entries_allowed = lambda: True
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


def _pos(entry=100.0, qty=1.0, **kw):
    return Position(qty=qty, entry_price=entry, high_water_mark=entry, **kw)


@pytest.fixture
def rows(monkeypatch):
    rec = []
    monkeypatch.setattr(base_loop, 'log_decision', rec.append)
    monkeypatch.setattr(trade_journal, 'log_decision', rec.append)
    return rec


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(base_loop.time, 'sleep', lambda s: None)
    monkeypatch.setattr(base_loop.random, 'uniform', lambda a, b: 0.0)
    monkeypatch.setattr(stock_loop.time, 'sleep', lambda s: None)
    monkeypatch.setattr(base_loop, 'record_trade', lambda *a, **k: None)


def _rows_by(rows, **want):
    return [r for r in rows
            if all(r.get(k) == v for k, v in want.items())]


def _stock_regime(block=True):
    return SimpleNamespace(should_halt_stocks=False,
                           should_block_risky_entries=block,
                           vix=28.0, sizing_mult=1.0, stop_mult=1.0)


def _base_stock_funnel(monkeypatch, rows, **over):
    """Drive base_loop._execute_buys on a stock-typed loop up to and past
    the VIX25 gate (downstream gates stubbed)."""
    events = []
    monkeypatch.setattr(base_loop, 'should_trade',
                        lambda pred, spread, **k: True)
    monkeypatch.setattr(base_loop, 'sentiment_gate',
                        lambda sym, at: (1.0, []))
    inst = _mk(_StockTyped,
               macro_regime=_stock_regime(True),
               _universe=['NVDA'],
               _quotes={'NVDA': {'midpoint': 100.0, 'spread_pct': 0.02}},
               **over)
    inst._meta_gate = lambda sym, pred, snaps, rank=None: (True, 1.0)
    inst._compute_position_size = lambda *a, **k: 500
    inst._place_and_track_buy = (
        lambda *a, **k: events.append(a[0]))
    return inst, events


# ===========================================================================
# 1. VIX25_BLOCK_REMOVED
# ===========================================================================

class TestVix25BlockRemoved:
    def test_off_blocks_and_journals(self, monkeypatch, rows, fast):
        inst, buys = _base_stock_funnel(monkeypatch, rows)
        inst._execute_buys({'NVDA': 0.50}, {'NVDA': {}})
        assert buys == []
        assert len(_rows_by(rows, action='skip',
                            skip_reason='vix25_block')) == 1

    def test_on_skips_block_entirely(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'VIX25_BLOCK_REMOVED', True)
        inst, buys = _base_stock_funnel(monkeypatch, rows)
        inst._execute_buys({'NVDA': 0.50}, {'NVDA': {}})
        assert buys == ['NVDA']
        assert _rows_by(rows, action='skip', skip_reason='vix25_block') == []

    def test_on_stock_loop_live_site(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'VIX25_BLOCK_REMOVED', True)
        _tu = types.ModuleType('trading_utils')
        _tu.cooldown_ok = lambda *a, **k: True
        _tu.LLM_VETO_THRESHOLD = 0.15
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu)
        _ev = types.ModuleType('events_calendar')
        _ev.earnings_within_days = lambda *a, **k: False
        monkeypatch.setitem(sys.modules, 'events_calendar', _ev)
        _ed = types.ModuleType('edgar_events')
        _ed.entry_blocked = lambda s: (False, None)
        monkeypatch.setitem(sys.modules, 'edgar_events', _ed)
        import macro_indicators
        monkeypatch.setattr(macro_indicators, 'get_spy_trend_ok',
                            lambda api: True)
        import order_utils
        monkeypatch.setattr(order_utils, 'should_trade',
                            lambda pred, spread, **k: True)
        monkeypatch.setattr(stock_loop, 'sentiment_gate',
                            lambda sym, at: (1.0, []))
        inst = object.__new__(stock_loop.StockLoop)
        for k, v in dict(
                api=None, trade_threshold=0.15, positions={},
                last_trade_time={}, hard_stop_lockout={}, llm_scores={},
                cycle=2, macro_regime=_stock_regime(True), corr_matrix={},
                _equity=100_000.0, _buys_allowed=True, _daily_trades={},
                _daily_trades_date=_dt.date.today().isoformat(),
                _last_meta_p={}, _last_meta_p_cycle={},
                flattened_today=False, top_symbols=['NVDA'],
                hold_symbols=set(), _tp_order_ids={},
                _entries_allowed=lambda: True,
                _in_entry_window=lambda: True,
                _get_current_exposure=lambda: 0.0,
                _bucket_room_ok=lambda s: True,
                _is_hard_stop_locked=lambda s: False,
                _trade_budget_ok=lambda s: True,
                _meta_gate=lambda sym, pred, snaps, rank=None: (True, 1.0),
                _compute_position_size=lambda *a, **k: 0,   # sizing_zero
                get_quote=lambda s: {'midpoint': 100.0,
                                     'spread_pct': 0.02}).items():
            setattr(inst, k, v)
        inst._execute_buys({'NVDA': 0.50}, {'NVDA': {}})
        # Passed the vix25 gate (flag ON) all the way to sizing.
        assert _rows_by(rows, action='skip', skip_reason='vix25_block') == []
        assert len(_rows_by(rows, action='skip',
                            skip_reason='sizing_zero')) == 1


# ===========================================================================
# 2. CORR_FAMILY_MERGED + the direct covered-none rho fix
# ===========================================================================

def _corr_funnel(monkeypatch, rows, corr):
    monkeypatch.setattr(base_loop, 'should_trade',
                        lambda pred, spread, **k: True)
    monkeypatch.setattr(base_loop, 'sentiment_gate',
                        lambda sym, at: (1.0, []))
    inst = _mk(positions={'ETH/USD': _pos()},
               corr_matrix={('BTC/USD', 'ETH/USD'): corr,
                            ('ETH/USD', 'BTC/USD'): corr})
    inst._meta_gate = lambda sym, pred, snaps, rank=None: (True, 1.0)
    inst._compute_position_size = lambda *a, **k: 500
    bought = []
    inst._place_and_track_buy = lambda *a, **k: bought.append(a[0])
    return inst, bought


class TestCorrFamilyMerged:
    def test_off_binary_gate_at_070(self, monkeypatch, rows, fast):
        inst, bought = _corr_funnel(monkeypatch, rows, 0.75)
        inst._execute_buys({'BTC/USD': 0.50}, {'BTC/USD': {}})
        assert bought == []
        skip, = _rows_by(rows, action='skip', skip_reason='correlation')
        assert skip['avg_corr'] == 0.75
        assert 'corr_sanity' not in skip

    def test_on_admits_below_sanity_bar(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'CORR_FAMILY_MERGED', True)
        inst, bought = _corr_funnel(monkeypatch, rows, 0.75)
        inst._execute_buys({'BTC/USD': 0.50}, {'BTC/USD': {}})
        assert bought == ['BTC/USD']       # 0.75 < 0.85 sanity bar
        assert _rows_by(rows, action='skip', skip_reason='correlation') == []

    def test_on_sanity_block_above_085(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'CORR_FAMILY_MERGED', True)
        inst, bought = _corr_funnel(monkeypatch, rows, 0.90)
        inst._execute_buys({'BTC/USD': 0.50}, {'BTC/USD': {}})
        assert bought == []
        skip, = _rows_by(rows, action='skip', skip_reason='correlation')
        assert skip['corr_sanity'] is True

    def test_covered_none_rho_prior(self):
        # Direct-ship fix: a NON-EMPTY matrix covering none of the book's
        # pairs honors the caller's prior instead of fabricating 0.0.
        m = {('X', 'Y'): 0.9, ('Y', 'X'): 0.9}
        portfolio._bookcorr_warn_ts = 0.0
        assert portfolio.avg_book_correlation(
            ['A', 'B'], m, uncovered=0.5) == 0.5
        # Legacy default preserved (pinned by test_portfolio_v3 too).
        portfolio._bookcorr_warn_ts = 0.0
        assert portfolio.avg_book_correlation(['A', 'B'], m) == 0.0
        # Covered pairs unaffected by the parameter.
        assert portfolio.avg_book_correlation(
            ['X', 'Y'], m, uncovered=0.5) == pytest.approx(0.9)
        # Empty matrix / short book keep the 0.0 early return.
        assert portfolio.avg_book_correlation(['A', 'B'], {},
                                              uncovered=0.5) == 0.0

    def test_enb_call_site_passes_prior(self):
        import inspect
        src = inspect.getsource(base_loop.BaseTradingLoop._compute_position_size)
        assert 'uncovered=0.5' in src
        src2 = inspect.getsource(base_loop.BaseTradingLoop._record_account_risk)
        assert 'uncovered=0.5' in src2


class TestCorrSizingFactorMerged:
    def _size(self, monkeypatch, tmp_path):
        monkeypatch.setattr(market_data, 'get_live_atr',
                            lambda *a, **k: None)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda *a, **k: None)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda *a, **k: None)
        monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                            lambda *a, **k: 1.0)
        inst = _mk(positions={'ETH/USD': _pos()},
                   corr_matrix={('BTC/USD', 'ETH/USD'): 0.6,
                                ('ETH/USD', 'BTC/USD'): 0.6})
        n = inst._compute_position_size(
            'BTC/USD', 0.5, {'midpoint': 100.0, 'spread_pct': 0.02})
        return inst._last_sizing_detail, n

    def test_off_f_corr_applies(self, monkeypatch, tmp_path):
        d, _ = self._size(monkeypatch, tmp_path)
        assert 0 < d['corr_mult'] < 1.0
        assert 'corr_family_merged' not in d

    def test_on_f_corr_composes_at_one(self, monkeypatch, tmp_path):
        monkeypatch.setattr(strategy_config, 'CORR_FAMILY_MERGED', True)
        d, _ = self._size(monkeypatch, tmp_path)
        assert 'corr_mult' not in d
        assert d['corr_family_merged'] is True

    def test_v2_shadow_inherits_merge(self, monkeypatch, tmp_path):
        # v2's tilt reads detail.get('corr_mult', 1.0): flag OFF includes
        # the f_corr haircut, flag ON composes it at 1.0 — the merge
        # applies to BOTH compositions (spec: ENB is the single consumer).
        d_off, _ = self._size(monkeypatch, tmp_path)
        monkeypatch.setattr(strategy_config, 'CORR_FAMILY_MERGED', True)
        d_on, _ = self._size(monkeypatch, tmp_path)
        f_corr = d_off['corr_mult']
        assert 0 < f_corr < 1.0
        assert d_off['v2']['tilt_raw'] == pytest.approx(
            d_on['v2']['tilt_raw'] * f_corr, rel=1e-3)


# ===========================================================================
# 3. TRADE_BUDGET_BACKSTOP
# ===========================================================================

class TestTradeBudgetBackstop:
    def test_off_todays_cap(self):
        inst = _mk(_daily_trades={'BTC/USD': 4})
        assert inst._trade_budget_ok('BTC/USD') is False
        inst._daily_trades['BTC/USD'] = 3
        assert inst._trade_budget_ok('BTC/USD') is True

    def test_on_backstop_cap(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'TRADE_BUDGET_BACKSTOP', True)
        inst = _mk(_daily_trades={'BTC/USD': 4})
        assert inst._trade_budget_ok('BTC/USD') is True     # 4 < 12
        inst._daily_trades['BTC/USD'] = 12
        assert inst._trade_budget_ok('BTC/USD') is False    # backstop binds


# ===========================================================================
# 4. CRYPTO_VERTICAL_BARRIER
# ===========================================================================

def _vertical_inst(age_hours=30.0, cls=_Loop, **over):
    inst = _mk(cls,
               positions={'BTC/USD': _pos()},
               config={'forward_bars': 24},
               _position_entry_ts={
                   'BTC/USD': _time.time() - age_hours * 3600},
               **over)
    return inst


class TestCryptoVerticalBarrier:
    def test_would_fire_journaled_flag_off(self, rows, fast):
        inst = _vertical_inst()
        inst._manage_stops()
        ev = _rows_by(rows, action='vertical_barrier')
        assert len(ev) == 1
        assert ev[0]['fired'] is False
        assert ev[0]['forward_bars'] == 24
        assert ev[0]['age_hours'] >= 24
        assert 'BTC/USD' in inst.positions      # flag OFF: no exit
        # One row per position lifetime (dedupe on re-check).
        inst._manage_stops()
        assert len(_rows_by(rows, action='vertical_barrier')) == 1

    def test_young_position_no_row(self, rows, fast):
        inst = _vertical_inst(age_hours=10.0)
        inst._manage_stops()
        assert _rows_by(rows, action='vertical_barrier') == []

    def test_unknown_age_fails_open(self, rows, fast):
        inst = _vertical_inst()
        inst._position_entry_ts = {}
        inst._manage_stops()
        assert _rows_by(rows, action='vertical_barrier') == []
        assert 'BTC/USD' in inst.positions

    def test_flag_on_exits_via_stop_path(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'CRYPTO_VERTICAL_BARRIER', True)
        fired = []
        inst = _vertical_inst()
        inst._execute_stop_exit = (
            lambda sym, pos, reason, px, quote=None:
            fired.append((sym, reason)))
        inst._manage_stops()
        assert fired == [('BTC/USD', 'vertical')]
        ev, = _rows_by(rows, action='vertical_barrier')
        assert ev['fired'] is True

    def test_stock_book_never_fires(self, rows, fast):
        inst = _vertical_inst(cls=_StockTyped)
        inst._universe = ['BTC/USD']    # symbol irrelevant; asset gate rules
        inst._manage_stops()
        assert _rows_by(rows, action='vertical_barrier') == []

    def test_entry_ts_persisted_and_restored(self, monkeypatch, tmp_path,
                                             fast):
        # Save side: entry_ts key lands in the state blob.
        inst = _vertical_inst()
        sf = tmp_path / 'position_state.json'
        inst._position_state_file = lambda: str(sf)
        del inst._save_position_state       # restore real method
        inst._save_position_state()
        data = json.loads(sf.read_text())
        assert 'BTC/USD' in data['entry_ts']
        # Restore side: _reconstruct_positions rehydrates the clock.
        ts = data['entry_ts']['BTC/USD']
        inst2 = _mk()
        inst2._update_equity = lambda: None
        inst2._replace_protective_stops = lambda: None
        inst2._load_position_state = lambda: {'entry_ts': {'BTC/USD': ts}}
        monkeypatch.setattr(base_loop, 'reconstruct_positions',
                            lambda api, symbols: {
                                'BTC/USD': {'qty': 1.0, 'entry_price': 100.0,
                                            'high_water_mark': 100.0}})
        monkeypatch.setattr(market_data, 'get_live_atr',
                            lambda *a, **k: None)
        inst2._reconstruct_positions()
        assert inst2._position_entry_ts['BTC/USD'] == pytest.approx(ts)

    def test_vertical_is_loop_layer_only(self):
        # SACRED: the vertical barrier is a LOOP-layer position-age check;
        # the policy_exits kernel is never imported or invoked by ANY live
        # loop file (docstrings may NAME it — assert on import/call sites,
        # not prose). Source read from disk so crypto_loop's heavy imports
        # don't gate the pin.
        for fname in ('base_loop.py', 'stock_loop.py', 'crypto_loop.py'):
            full = (REPO / fname).read_text()
            assert 'import policy_exits' not in full, fname
            assert 'from policy_exits' not in full, fname
            assert 'exit_walk(' not in full, fname


# ===========================================================================
# 5. SIGNAL_EXIT_CONFIRM_READS
# ===========================================================================

def _sell_inst(**over):
    sold = []
    inst = _mk(positions={'BTC/USD': _pos()}, **over)
    inst.place_sell_order = (
        lambda sym, qty, quote:
        (sold.append(sym) or SimpleNamespace(filled_avg_price='99.0')))
    return inst, sold


class TestSignalExitConfirmReads:
    def test_default_single_reading_byte_identical(self, rows, fast):
        inst, sold = _sell_inst()
        inst._execute_sells({'BTC/USD': -1.0})
        assert sold == ['BTC/USD']
        assert _rows_by(rows, action='signal_exit_reading') == []
        row, = _rows_by(rows, action='sell')
        assert 'signal_exit_readings' not in row

    def test_two_reads_arms_then_confirms(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        inst, sold = _sell_inst()
        inst._execute_sells({'BTC/USD': -1.0})
        assert sold == []                              # armed, not sold
        armed, = _rows_by(rows, action='signal_exit_reading', event='armed')
        assert armed['confirm_reads'] == 2
        inst._execute_sells({'BTC/USD': -1.0})         # consecutive reading
        assert sold == ['BTC/USD']
        row, = _rows_by(rows, action='sell')
        assert row['signal_exit_readings'] == 2

    def test_two_reads_pending_survives_missing_pred(self, monkeypatch,
                                                     rows, fast):
        # pred None = data failure, not recovery: armed state survives
        # (base_loop `continue`s before the pending logic) and the next
        # real negative reading confirms.
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        inst, sold = _sell_inst()
        inst._execute_sells({'BTC/USD': -1.0})         # arm
        inst._execute_sells({})                        # pred gap
        assert 'BTC/USD' in inst._pending_signal_exit
        assert _rows_by(rows, action='signal_exit_reading',
                        event='lapsed') == []
        inst._execute_sells({'BTC/USD': -1.0})         # confirm
        assert sold == ['BTC/USD']

    def test_two_reads_lapse_clears(self, monkeypatch, rows, fast):
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        inst, sold = _sell_inst()
        inst._execute_sells({'BTC/USD': -1.0})         # arm
        inst._execute_sells({'BTC/USD': +0.5})         # recovers
        assert sold == []
        assert len(_rows_by(rows, action='signal_exit_reading',
                            event='lapsed')) == 1
        assert 'BTC/USD' not in inst._pending_signal_exit
        # A fresh breach re-arms from scratch.
        inst._execute_sells({'BTC/USD': -1.0})
        assert sold == []

    def _stock_me(self, monkeypatch, preds_hold=True):
        recorded = []
        _tu = types.ModuleType('trading_utils')
        _tu.cooldown_ok = lambda *a, **k: True
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu)
        monkeypatch.setattr(stock_loop, 'cancel_orders_for_symbol',
                            lambda *a, **k: True)
        monkeypatch.setattr(stock_loop.time, 'sleep', lambda s: None)
        info = SimpleNamespace(stop_order_id=None, entry_price=100.0, qty=5)
        me = SimpleNamespace(
            api=SimpleNamespace(
                get_position=lambda s: SimpleNamespace(qty='5')),
            positions={'AAPL': info},
            hold_symbols={'AAPL'} if preds_hold else set(),
            trade_threshold=0.15, last_trade_time={}, COOLDOWN_MINUTES=20,
            HOLD_RANK=15, llm_scores={}, cycle=3,
            get_quote=lambda s: {'midpoint': 100.0},
            place_sell_order=lambda s, q, quote: SimpleNamespace(
                filled_avg_price='99.0'),
            _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)),
            _journal_external_close=lambda *a, **k: None,
        )
        return me, recorded

    def test_stock_two_reads_signal_exit(self, monkeypatch, rows):
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        me, recorded = self._stock_me(monkeypatch)
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})
        assert recorded == []                          # armed
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})
        (args, kwargs), = recorded
        assert kwargs['extra']['signal_exit_readings'] == 2

    def test_stock_rank_drop_never_deferred(self, monkeypatch, rows):
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        me, recorded = self._stock_me(monkeypatch, preds_hold=False)
        # pred<0 but above -threshold, out of hold set -> rank-drop exit,
        # which keeps its own hysteresis band and fires on ONE reading.
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -0.01})
        (args, kwargs), = recorded
        assert kwargs['exit_reason'] == 'signal_sell'
        assert 'signal_exit_readings' not in (kwargs['extra'] or {})

    def test_stock_default_byte_identical(self, monkeypatch, rows):
        me, recorded = self._stock_me(monkeypatch)
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})
        (args, kwargs), = recorded
        assert kwargs['extra'] is None    # legacy key-set (no cooldown here)

    def test_stock_pending_survives_missing_pred(self, monkeypatch, rows):
        # A missing pred is a data failure, not a recovery: the armed
        # first reading must SURVIVE it (hardener fix — the pop used to
        # run before the pred-is-None check) and confirm on the next real
        # negative reading, mirroring base_loop's pred-None `continue`.
        monkeypatch.setattr(strategy_config, 'SIGNAL_EXIT_CONFIRM_READS', 2)
        me, recorded = self._stock_me(monkeypatch)
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})   # arm
        assert me._pending_signal_exit == {'AAPL': 3}
        stock_loop.StockLoop._execute_sells(me, {})               # pred gap
        assert me._pending_signal_exit == {'AAPL': 3}             # survives
        assert _rows_by(rows, action='signal_exit_reading',
                        event='lapsed') == []
        stock_loop.StockLoop._execute_sells(me, {'AAPL': -1.0})   # confirm
        (args, kwargs), = recorded
        assert kwargs['extra']['signal_exit_readings'] == 2


# ===========================================================================
# 6. KELLY_SAMPLE_GATE
# ===========================================================================

def _trade_file(tmp_path, n_win, n_loss, ts):
    trades = ([{'pnl_pct': 4.0, 'ts': ts}] * n_win
              + [{'pnl_pct': -1.0, 'ts': ts}] * n_loss)
    f = tmp_path / 'trade_memory.json'
    f.write_text(json.dumps({'BTC/USD': trades}))
    return f


class TestUncensoredTradeCount:
    def test_counts_and_filters(self, tmp_path, monkeypatch):
        f = tmp_path / 'trade_memory.json'
        f.write_text(json.dumps({
            'BTC/USD': [{'pnl_pct': 1.0, 'ts': '2026-09-01T00:00:00'},
                        {'pnl_pct': 1.0, 'ts': '2026-07-01T00:00:00'},
                        {'pnl_pct': 1.0, 'ts': '2026-09-02T00:00:00',
                         'estimated': True}],
            'NVDA': [{'pnl_pct': 1.0, 'ts': '2026-09-01T00:00:00'}]}))
        monkeypatch.setattr(_tu_real, '_TRADE_MEMORY_FILE', f)
        assert _tu_real.uncensored_trade_count() == 3
        assert _tu_real.uncensored_trade_count('crypto') == 2
        assert _tu_real.uncensored_trade_count('stock') == 1
        assert _tu_real.uncensored_trade_count(
            'crypto', since_iso='2026-08-22') == 1     # estimated excluded
        assert _tu_real.uncensored_trade_count(
            'stock', since_iso='2026-10-01') == 0

    def test_missing_file_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_tu_real, '_TRADE_MEMORY_FILE',
                            tmp_path / 'absent.json')
        assert _tu_real.uncensored_trade_count() == 0


class TestKellySampleGate:
    def _size(self, monkeypatch, tmp_path, ts, flag):
        f = _trade_file(tmp_path, 55, 5, ts)
        monkeypatch.setattr(_tu_real, '_TRADE_MEMORY_FILE', f)
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu_real)
        if flag:
            monkeypatch.setattr(strategy_config, 'KELLY_SAMPLE_GATE', True)
        monkeypatch.setattr(market_data, 'get_live_atr',
                            lambda *a, **k: None)
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda *a, **k: None)
        monkeypatch.setattr(market_data, 'fetch_stock_bars_alpaca',
                            lambda *a, **k: None)
        monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                            lambda *a, **k: 1.0)
        inst = _mk()
        inst._compute_position_size(
            'BTC/USD', 0.5, {'midpoint': 100.0, 'spread_pct': 0.02})
        return inst._last_sizing_detail

    def test_off_byte_identical(self, monkeypatch, tmp_path):
        d = self._size(monkeypatch, tmp_path, '2026-07-01T00:00:00', False)
        assert d['kelly_mult'] == 1.5           # winner-heavy sample ramps
        assert 'kelly_gate' not in d

    def test_on_holds_neutral_on_censored_history(self, monkeypatch,
                                                  tmp_path):
        # All 60 rows predate KELLY_SAMPLE_SINCE -> gate holds 1.0.
        d = self._size(monkeypatch, tmp_path, '2026-07-01T00:00:00', True)
        assert d['kelly_mult'] == 1.0
        assert d['kelly_gate']['held_neutral'] is True
        assert d['kelly_gate']['n_uncensored'] == 0

    def test_on_releases_at_min_trades(self, monkeypatch, tmp_path):
        d = self._size(monkeypatch, tmp_path, '2026-09-01T00:00:00', True)
        assert d['kelly_mult'] == 1.5
        assert d['kelly_gate']['held_neutral'] is False
        assert d['kelly_gate']['n_uncensored'] == 60


# ===========================================================================
# 7. BREAKER_PER_BOOK
# ===========================================================================

class TestBreakerPerBook:
    def test_off_uses_account_wide_check(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            base_loop, 'check_circuit_breaker',
            lambda api, max_drawdown_pct: (calls.append(1) or (False, 0.01)))
        inst = _mk()
        assert inst._circuit_breaker_check() is False
        assert calls == [1]
        assert inst._buys_allowed is True

    def test_on_dispatches_to_book_check(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', True)
        monkeypatch.setattr(
            base_loop, 'check_circuit_breaker',
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError('account-wide path used under flag')))
        inst = _mk()
        inst._book_breaker_check = lambda: (False, 0.01)
        assert inst._circuit_breaker_check() is False
        assert inst._buys_allowed is True

    def test_book_check_baseline_then_trip(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', True)
        api = SimpleNamespace(get_account=lambda: SimpleNamespace(
            equity='100000'))
        inst = _mk(api=api,
                   positions={'BTC/USD': _pos(entry=100.0, qty=100.0)},
                   _last_marks={'BTC/USD': 100.0})
        tripped, dd = inst._book_breaker_check()
        assert (tripped, dd) == (False, 0.0)           # baseline capture
        assert inst._breaker_base['equity'] == 100000.0
        assert inst._breaker_base['refs']['BTC/USD'] == 100.0
        inst._last_marks['BTC/USD'] = 40.0             # -6000 on 100k
        tripped, dd = inst._book_breaker_check()
        assert tripped is True
        assert dd == pytest.approx(0.06)

    def test_book_check_realized_accumulates(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', True)
        api = SimpleNamespace(get_account=lambda: SimpleNamespace(
            equity='100000'))
        inst = _mk(api=api)
        inst._book_breaker_check()                     # baseline, no marks
        inst._breaker_note_realized(-5500.0)           # realized loss
        tripped, dd = inst._book_breaker_check()
        assert tripped is True
        assert dd == pytest.approx(0.055)

    def test_note_realized_noop_flag_off(self):
        inst = _mk(_breaker_base={'realized': 0.0})
        inst._breaker_note_realized(-9999.0)
        assert inst._breaker_base['realized'] == 0.0   # flag OFF: untouched

    def test_book_check_gain_never_trips(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', True)
        api = SimpleNamespace(get_account=lambda: SimpleNamespace(
            equity='100000'))
        inst = _mk(api=api,
                   positions={'BTC/USD': _pos(entry=100.0, qty=100.0)},
                   _last_marks={'BTC/USD': 100.0})
        inst._book_breaker_check()
        inst._last_marks['BTC/USD'] = 160.0
        tripped, dd = inst._book_breaker_check()
        assert tripped is False
        assert dd < 0                                  # book is UP

    def test_book_check_api_error_fails_closed(self, monkeypatch):
        monkeypatch.setattr(strategy_config, 'BREAKER_PER_BOOK', True)

        def _boom():
            raise RuntimeError('down')
        inst = _mk(api=SimpleNamespace(get_account=lambda: _boom()))
        assert inst._book_breaker_check() == (False, None)
        # Through the dispatcher: dd None -> buys suspended (fail closed).
        assert inst._circuit_breaker_check() is False
        assert inst._buys_allowed is False

    def test_crypto_window_is_weekend_aware(self):
        inst = _mk()
        end = inst._breaker_window_end()
        now = _dt.datetime.now(_UTC)
        assert end.tzinfo is not None
        assert end > now
        assert (end - now) <= _dt.timedelta(days=1)
        assert (end.hour, end.minute) == (0, 0)        # UTC midnight, DAILY

    def test_stock_window_matches_baseline_reset(self):
        inst = _mk(_StockTyped)
        assert inst._breaker_window_end() == inst._next_baseline_reset()


# ===========================================================================
# 8. Flag defaults (the whole family ships OFF/today)
# ===========================================================================

def test_ia4_flag_defaults():
    assert strategy_config.VIX25_BLOCK_REMOVED is False
    assert strategy_config.CORR_FAMILY_MERGED is False
    assert strategy_config.CORR_SANITY_MAX == 0.85
    assert strategy_config.TRADE_BUDGET_BACKSTOP is False
    assert strategy_config.TRADE_BUDGET_BACKSTOP_MULT == 3
    assert strategy_config.CRYPTO_VERTICAL_BARRIER is False
    assert strategy_config.SIGNAL_EXIT_CONFIRM_READS == 1
    assert strategy_config.KELLY_SAMPLE_GATE is False
    assert strategy_config.KELLY_SAMPLE_MIN_TRADES == 50
    assert strategy_config.BREAKER_PER_BOOK is False
