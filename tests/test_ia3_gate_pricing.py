"""IA-3 — price the unpriced gates (2026-08 influence audit, open question #4:
"the highest-impact gates are the least priced"). All measurement-only:
no gate's admit/veto behavior changes; every previously invisible block
now journals a skip/event row decision_report-shaped via the existing
_journal_skip plumbing.

Pins:
1. VIX>35 macro halt journals 'vix_halt' skip rows with pred/rank/vix
   context (base_loop stock-typed branch AND the live stock_loop site);
   the crypto book stays untouched (branch inert, behavior unchanged).
2. VIX>25 non-safe-haven block journals 'vix25_block' skip rows + a
   once-per-day log naming the SAFE_HAVEN-untradable gap (C3). Gate
   behavior unchanged.
3. FOMC/CPI stand-down journals stood-down skips WITH window identity
   (macro_calendar.standdown_window_id — new journal hook), once per
   (window, symbol), threshold-crossing candidates only; manual halt
   journals nothing; macro_standdown's own behavior is byte-equivalent
   (boundaries + reason strings pinned across the refactor).
4. High-VIX RR_5 tiebreak demotion journals every demotion with pre/post
   rank + the counterfactual admission flag ("journal ... within one
   release or delete it" — this is the release), and a priced
   'rr5_demotion' skip row for names pushed out of the top-N.
5. Ranks TOP_N+1..HOLD_RANK journal 'rank_near_miss' skip rows (rank,
   pred, would-be size proxy) only when the entry funnel actually runs,
   once per (day, symbol).
6. Circuit-breaker trips journal a 'circuit_breaker_trip' event row
   (book, baseline used, drawdown at trip) and never block the flatten.

Stub pattern mirrors tests/test_c26_base_loop_functional.py: real
`import base_loop`/`import stock_loop` under dotenv/predict_now stubs,
sys.modules restored afterward.
"""

import datetime as _dt
import sys
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
finally:
    for _m in ('dotenv', 'predict_now', 'trading_utils',
               'base_loop', 'stock_loop', 'crypto_loop'):
        sys.modules.pop(_m, None)

import macro_calendar          # Mac-importable (stdlib only)
import macro_indicators        # get_spy_trend_ok seam (function-local import)
import notify                  # halt/notify seams (function-local imports)
import order_utils             # should_trade seam (module object shared)
import trade_journal           # log_decision seam (function-local imports)
from stock_config import SAFE_HAVEN_SYMBOLS

_ET = zoneinfo.ZoneInfo('America/New_York')


def _et(y, mo, d, h, mi):
    return _dt.datetime(y, mo, d, h, mi, tzinfo=_ET)


# ---------------------------------------------------------------------------
# Concrete subclasses + factories
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
    """Exercises base_loop's stock-typed branches (vix gates)."""

    def get_asset_type(self):
        return 'stock'


def _mk(cls=_Loop, **over):
    """Instance bypassing __init__ (which calls get_api(), reads repo
    state files and creates a thread pool)."""
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
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


def _mk_stock(**over):
    inst = object.__new__(stock_loop.StockLoop)
    inst.api = None
    inst.trade_threshold = 0.15
    inst.positions = {}; inst.last_trade_time = {}
    inst.hard_stop_lockout = {}
    inst.llm_scores = {}
    inst.cycle = 2
    inst.macro_regime = None; inst.corr_matrix = {}
    inst._equity = 100_000.0
    inst._buys_allowed = True
    inst._daily_trades = {}
    inst._daily_trades_date = _dt.date.today().isoformat()
    inst._last_meta_p = {}; inst._last_meta_p_cycle = {}
    inst.flattened_today = False
    inst.top_symbols = []; inst.hold_symbols = set()
    inst._tp_order_ids = {}
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


@pytest.fixture
def rows(monkeypatch):
    """Capture BOTH journal seams: base_loop's module-level log_decision
    binding AND function-local `from trade_journal import log_decision`
    call sites (stock funnel, _journal_rr5_demotions)."""
    rec = []
    monkeypatch.setattr(base_loop, 'log_decision', rec.append)
    monkeypatch.setattr(trade_journal, 'log_decision', rec.append)
    return rec


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(base_loop.time, 'sleep', lambda s: None)
    monkeypatch.setattr(base_loop.random, 'uniform', lambda a, b: 0.0)
    monkeypatch.setattr(stock_loop.time, 'sleep', lambda s: None)


def _rows_by(rows, **want):
    return [r for r in rows
            if all(r.get(k) == v for k, v in want.items())]


# ---------------------------------------------------------------------------
# 1. macro_calendar: window identity hook + refactor byte-equivalence
# ---------------------------------------------------------------------------

class TestStanddownWindowId:
    def test_fomc_window_identity(self):
        assert (macro_calendar.standdown_window_id(_et(2026, 7, 29, 14, 0))
                == 'FOMC-2026-07-29')
        assert (macro_calendar.standdown_window_id(_et(2026, 7, 29, 12, 0))
                == 'FOMC-2026-07-29')   # start inclusive
        assert macro_calendar.standdown_window_id(
            _et(2026, 7, 29, 11, 59)) is None
        assert macro_calendar.standdown_window_id(
            _et(2026, 7, 29, 15, 30)) is None   # half-open end

    def test_cpi_window_identity(self):
        assert (macro_calendar.standdown_window_id(_et(2026, 8, 12, 8, 30))
                == 'CPI-2026-08-12')
        assert macro_calendar.standdown_window_id(
            _et(2026, 8, 12, 9, 30)) is None

    def test_non_event_day(self):
        assert macro_calendar.standdown_window_id(
            _et(2026, 8, 3, 14, 0)) is None

    def test_macro_standdown_behavior_pinned_across_refactor(self):
        # Same reason strings + half-open boundaries as before the
        # _active_window refactor (test_review_b04 pins more; this is the
        # local sentinel).
        blocked, reason = macro_calendar.macro_standdown(_et(2026, 7, 29, 14, 0))
        assert blocked and reason == 'FOMC stand-down (12:00-15:30 ET)'
        blocked, reason = macro_calendar.macro_standdown(_et(2026, 8, 12, 8, 30))
        assert blocked and reason == 'CPI stand-down (06:30-09:30 ET)'
        assert not macro_calendar.macro_standdown(_et(2026, 7, 29, 15, 30))[0]
        assert not macro_calendar.macro_standdown(_et(2026, 8, 3, 14, 0))[0]

    def test_naive_datetime_read_as_utc(self):
        # 18:00 UTC == 14:00 ET on 2026-07-29 (EDT)
        naive = _dt.datetime(2026, 7, 29, 18, 0)
        assert macro_calendar.standdown_window_id(naive) == 'FOMC-2026-07-29'


# ---------------------------------------------------------------------------
# 2. FOMC/CPI stand-down: journaled stood-down skips with window identity
# ---------------------------------------------------------------------------

def _arm_standdown(monkeypatch, wid='CPI-2026-08-12',
                   reason='CPI stand-down (06:30-09:30 ET)'):
    monkeypatch.setattr(notify, 'halt_active', lambda: False)
    monkeypatch.setattr(macro_calendar, 'macro_standdown',
                        lambda now=None: (True, reason))
    monkeypatch.setattr(macro_calendar, 'standdown_window_id',
                        lambda now=None: wid)
    monkeypatch.setattr(macro_calendar, 'calendar_exhausted',
                        lambda now=None: False)


class TestStanddownJournaling:
    def test_stood_down_skips_journaled_with_window_identity(
            self, monkeypatch, rows):
        _arm_standdown(monkeypatch)
        inst = _mk(_universe=['BTC/USD', 'ETH/USD'])
        preds = {'BTC/USD': 0.50, 'ETH/USD': 0.05}   # only BTC >= 0.15
        inst._execute_buys(preds, {})
        skips = _rows_by(rows, action='skip', skip_reason='macro_standdown')
        assert len(skips) == 1
        r = skips[0]
        assert r['symbol'] == 'BTC/USD'
        assert r['pred_return'] == 0.5
        assert r['entry_rank'] == 1
        assert r['standdown_window'] == 'CPI-2026-08-12'
        assert r['standdown_reason'] == 'CPI stand-down (06:30-09:30 ET)'
        # The funnel itself never ran (no entry_window row, no buys)
        assert _rows_by(rows, action='entry_window') == []

    def test_once_per_window_per_symbol(self, monkeypatch, rows):
        _arm_standdown(monkeypatch)
        inst = _mk()
        preds = {'BTC/USD': 0.50}
        inst._execute_buys(preds, {})
        inst._execute_buys(preds, {})   # same window: no second row
        assert len(_rows_by(rows, action='skip',
                            skip_reason='macro_standdown')) == 1
        monkeypatch.setattr(macro_calendar, 'standdown_window_id',
                            lambda now=None: 'FOMC-2026-09-16')
        inst._execute_buys(preds, {})   # new window: journals again
        skips = _rows_by(rows, action='skip', skip_reason='macro_standdown')
        assert len(skips) == 2
        assert skips[1]['standdown_window'] == 'FOMC-2026-09-16'

    def test_manual_halt_journals_nothing(self, monkeypatch, rows):
        monkeypatch.setattr(notify, 'halt_active', lambda: True)
        inst = _mk()
        inst._execute_buys({'BTC/USD': 0.50}, {})
        assert rows == []

    def test_stock_funnel_also_journals_standdown(self, monkeypatch, rows):
        _arm_standdown(monkeypatch, wid='FOMC-2026-10-28',
                       reason='FOMC stand-down (12:00-15:30 ET)')
        # The stock funnel imports trading_utils before the stand-down
        # check — stub its two names (Mac has no dotenv)
        _tu = types.ModuleType('trading_utils')
        _tu.cooldown_ok = lambda *a, **k: True
        _tu.LLM_VETO_THRESHOLD = 0.15
        monkeypatch.setitem(sys.modules, 'trading_utils', _tu)
        inst = _mk_stock()
        inst._execute_buys({'NVDA': 0.40}, {'NVDA': {}})
        skips = _rows_by(rows, action='skip', skip_reason='macro_standdown')
        assert len(skips) == 1
        assert skips[0]['symbol'] == 'NVDA'
        assert skips[0]['standdown_window'] == 'FOMC-2026-10-28'


# ---------------------------------------------------------------------------
# 3. base_loop VIX gates: priced skip rows; crypto book untouched
# ---------------------------------------------------------------------------

def _passthrough_gates(monkeypatch):
    monkeypatch.setattr(base_loop, 'should_trade',
                        lambda pred, spread, **k: True)
    monkeypatch.setattr(base_loop, 'sentiment_gate',
                        lambda sym, at: (1.0, []))


class TestBaseLoopVixGates:
    def test_vix_halt_skip_row(self, monkeypatch, rows, fast):
        _passthrough_gates(monkeypatch)
        inst = _mk(_StockTyped,
                   _entries_allowed=lambda: True,
                   macro_regime=SimpleNamespace(
                       should_halt_stocks=True,
                       should_block_risky_entries=False,
                       vix=41.34, sizing_mult=1.0))
        inst._execute_buys({'BTC/USD': 0.50}, {})
        skips = _rows_by(rows, action='skip', skip_reason='vix_halt')
        assert len(skips) == 1
        assert skips[0]['vix'] == 41.3
        assert skips[0]['pred_return'] == 0.5
        assert skips[0]['entry_rank'] == 1
        win = _rows_by(rows, action='entry_window')[0]
        assert win['veto_counts']['macro_halt'] == 1   # vc key unchanged
        assert win['admitted_k'] == 0

    def test_vix25_block_skip_row(self, monkeypatch, rows, fast):
        _passthrough_gates(monkeypatch)
        inst = _mk(_StockTyped,
                   _entries_allowed=lambda: True,
                   macro_regime=SimpleNamespace(
                       should_halt_stocks=False,
                       should_block_risky_entries=True,
                       vix=27.8, sizing_mult=1.0))
        assert 'BTC/USD' not in SAFE_HAVEN_SYMBOLS
        inst._execute_buys({'BTC/USD': 0.50}, {})
        skips = _rows_by(rows, action='skip', skip_reason='vix25_block')
        assert len(skips) == 1
        assert skips[0]['vix'] == 27.8
        win = _rows_by(rows, action='entry_window')[0]
        assert win['veto_counts']['vix_block'] == 1    # vc key unchanged
        # The daily gap log fired (state marker set)
        assert inst._vix25_gap_log_date == _dt.date.today().isoformat()

    def test_crypto_book_vix_gates_stay_inert(self, monkeypatch, rows, fast):
        """A halting macro regime must NOT veto (or journal vix rows on)
        the crypto book — the branches are stock-typed. Pins behavior
        unchanged: the candidate proceeds all the way to a buy."""
        _passthrough_gates(monkeypatch)
        events = []
        inst = _mk(_Loop,
                   _entries_allowed=lambda: True,
                   macro_regime=SimpleNamespace(
                       should_halt_stocks=True,
                       should_block_risky_entries=True,
                       vix=45.0, sizing_mult=1.0))
        inst._meta_gate = lambda sym, pred, snaps, rank=None: (True, 1.0)
        inst._compute_position_size = lambda *a, **k: 500
        inst._place_and_track_buy = lambda *a, **k: events.append('buy')
        inst._execute_buys({'BTC/USD': 0.50}, {})
        assert events == ['buy']
        assert _rows_by(rows, action='skip', skip_reason='vix_halt') == []
        assert _rows_by(rows, action='skip', skip_reason='vix25_block') == []


class TestVix25GapLog:
    def test_once_per_day_and_names_the_gap(self, monkeypatch):
        logs = []
        monkeypatch.setattr(base_loop.logger, 'warning',
                            lambda msg, *a: logs.append(('warn', msg % a if a else msg)))
        monkeypatch.setattr(base_loop.logger, 'info',
                            lambda msg, *a: logs.append(('info', msg % a if a else msg)))
        inst = _mk(_StockTyped, _universe=['ZZTOP', 'NVDA'])
        inst._log_vix25_gap_once(SAFE_HAVEN_SYMBOLS)
        inst._log_vix25_gap_once(SAFE_HAVEN_SYMBOLS)   # same day: no repeat
        gap = [m for lvl, m in logs if 'gap C3' in m]
        assert len(gap) == 1
        assert 'de facto full book halt' in gap[0]

    def test_tradable_safe_haven_downgrades_to_info(self, monkeypatch):
        logs = []
        monkeypatch.setattr(base_loop.logger, 'warning',
                            lambda msg, *a: logs.append(('warn', msg % a if a else msg)))
        monkeypatch.setattr(base_loop.logger, 'info',
                            lambda msg, *a: logs.append(('info', msg % a if a else msg)))
        inst = _mk(_StockTyped, _universe=['GLD', 'NVDA'])   # GLD is SAFE_HAVEN
        inst._log_vix25_gap_once(SAFE_HAVEN_SYMBOLS)
        assert [lvl for lvl, m in logs] == ['info']
        assert 'GLD' in logs[0][1]


# ---------------------------------------------------------------------------
# 4. stock_loop live VIX gate sites journal the same skip classes
# ---------------------------------------------------------------------------

def _stock_funnel(monkeypatch, rows, macro_regime):
    """Drive the real stock _execute_buys to the VIX gates."""
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
    monkeypatch.setattr(macro_indicators, 'get_spy_trend_ok',
                        lambda api: True)
    monkeypatch.setattr(order_utils, 'should_trade',
                        lambda pred, spread, **k: True)
    inst = _mk_stock(macro_regime=macro_regime,
                     top_symbols=['NVDA'],
                     _entries_allowed=lambda: True,
                     _in_entry_window=lambda: True,
                     _get_current_exposure=lambda: 0.0,
                     _bucket_room_ok=lambda s: True,
                     _is_hard_stop_locked=lambda s: False,
                     _trade_budget_ok=lambda s: True,
                     get_quote=lambda s: {'midpoint': 100.0,
                                          'spread_pct': 0.02})
    return inst


class TestStockLoopVixGates:
    def test_vix_halt_row_at_live_site(self, monkeypatch, rows, fast):
        inst = _stock_funnel(monkeypatch, rows, SimpleNamespace(
            should_halt_stocks=True, should_block_risky_entries=False,
            vix=38.5, sizing_mult=1.0, stop_mult=1.0))
        inst._execute_buys({'NVDA': 0.50}, {'NVDA': {}})
        skips = _rows_by(rows, action='skip', skip_reason='vix_halt')
        assert len(skips) == 1
        assert skips[0]['symbol'] == 'NVDA'
        assert skips[0]['vix'] == 38.5
        assert skips[0]['entry_rank'] == 1
        win = _rows_by(rows, action='entry_window')[0]
        assert win['veto_counts']['macro_halt'] == 1

    def test_vix25_block_row_at_live_site(self, monkeypatch, rows, fast):
        inst = _stock_funnel(monkeypatch, rows, SimpleNamespace(
            should_halt_stocks=False, should_block_risky_entries=True,
            vix=28.0, sizing_mult=1.0, stop_mult=1.0))
        assert 'NVDA' not in SAFE_HAVEN_SYMBOLS
        inst._execute_buys({'NVDA': 0.50}, {'NVDA': {}})
        skips = _rows_by(rows, action='skip', skip_reason='vix25_block')
        assert len(skips) == 1
        assert skips[0]['vix'] == 28.0
        assert inst._vix25_gap_log_date == _dt.date.today().isoformat()


# ---------------------------------------------------------------------------
# 5. RR_5 high-VIX tiebreak demotion journaling
# ---------------------------------------------------------------------------

class TestRR5DemotionJournaling:
    def test_direct_event_and_skip_rows(self, rows):
        inst = _mk_stock()
        # A: pushed out of top-N (pre 1 -> post 10); B: demoted within
        # the window (pre 5 -> post 7, still admitted)
        pre = {'A': 1, 'B': 5}
        post = {'A': 10, 'B': 7}
        rr = {'A': 4.2, 'B': 3.1}
        inst._journal_rr5_demotions({'A', 'B'}, pre, post, rr, 29.7,
                                    {'A': 0.9, 'B': 0.6}, {})
        ev = _rows_by(rows, action='rr5_demotion')
        assert len(ev) == 1
        assert ev[0]['vix'] == 29.7
        demos = {d['symbol']: d for d in ev[0]['demotions']}
        assert demos['A']['admission_lost'] is True
        assert demos['A']['pre_rank'] == 1 and demos['A']['post_rank'] == 10
        assert demos['B']['admission_lost'] is False
        skips = _rows_by(rows, action='skip', skip_reason='rr5_demotion')
        assert [s['symbol'] for s in skips] == ['A']   # only the lost one
        assert skips[0]['pre_rank'] == 1
        assert skips[0]['post_rank'] == 10
        assert skips[0]['rr5'] == 4.2
        assert skips[0]['pred_return'] == 0.9

    def test_wired_into_get_predictions(self, monkeypatch, rows):
        preds = {f'S{i:02d}': round(1.0 - 0.05 * i, 4) for i in range(1, 13)}
        snapshots = {f'S{i:02d}': {'RR_5': (5.0 if i == 1 else -1.0)}
                     for i in range(1, 11)}
        monkeypatch.setattr(base_loop.BaseTradingLoop, '_get_predictions',
                            lambda self, bc: (preds, snapshots))
        monkeypatch.setattr(stock_loop, 'get_market_sentiment', lambda: None)
        _pr = types.ModuleType('panel_ranks')
        _pr.compute_live_panel_ranks = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, 'panel_ranks', _pr)
        _pn2 = types.ModuleType('predict_now')
        _pn2.set_panel_features = lambda x: None
        monkeypatch.setitem(sys.modules, 'predict_now', _pn2)
        inst = _mk_stock(macro_regime=SimpleNamespace(vix=30.0),
                         write_prediction_cache=lambda p, **k: None)
        inst._get_predictions(None)
        # S01 (pre rank 1, RR_5 popper) demoted to the back of the
        # TOP_N+3 window -> rank 10, out of the top-7
        assert 'S01' not in inst.top_symbols
        skips = _rows_by(rows, action='skip', skip_reason='rr5_demotion')
        assert len(skips) == 1
        assert skips[0]['symbol'] == 'S01'
        assert skips[0]['pre_rank'] == 1
        assert skips[0]['post_rank'] == 10
        assert skips[0]['entry_rank'] == 10
        assert skips[0]['vix'] == 30.0
        ev = _rows_by(rows, action='rr5_demotion')
        assert len(ev) == 1
        # Near-miss substrate stashed for the funnel (ranks 8..12),
        # including the demoted S01 at its post rank
        stash = dict((s, r) for r, s, p in inst._near_miss_ranked)
        assert stash['S01'] == 10
        assert all(inst.TOP_N < r <= inst.HOLD_RANK
                   for r, s, p in inst._near_miss_ranked)

    def test_no_demotion_no_rows(self, monkeypatch, rows):
        preds = {f'S{i:02d}': round(1.0 - 0.05 * i, 4) for i in range(1, 13)}
        snapshots = {f'S{i:02d}': {'RR_5': -1.0} for i in range(1, 11)}
        monkeypatch.setattr(base_loop.BaseTradingLoop, '_get_predictions',
                            lambda self, bc: (preds, snapshots))
        monkeypatch.setattr(stock_loop, 'get_market_sentiment', lambda: None)
        _pr = types.ModuleType('panel_ranks')
        _pr.compute_live_panel_ranks = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, 'panel_ranks', _pr)
        _pn2 = types.ModuleType('predict_now')
        _pn2.set_panel_features = lambda x: None
        monkeypatch.setitem(sys.modules, 'predict_now', _pn2)
        inst = _mk_stock(macro_regime=SimpleNamespace(vix=30.0),
                         write_prediction_cache=lambda p, **k: None)
        inst._get_predictions(None)
        assert _rows_by(rows, action='rr5_demotion') == []
        assert inst.top_symbols == [f'S{i:02d}' for i in range(1, 8)]


# ---------------------------------------------------------------------------
# 6. Top-N near-miss skip rows (ranks 8-15)
# ---------------------------------------------------------------------------

class TestRankNearMiss:
    def test_rows_with_would_be_size_proxy(self, rows):
        inst = _mk_stock(_near_miss_ranked=[(8, 'AAA', 0.30),
                                            (9, 'BBB', 0.25)])
        snapshots = {'AAA': {'Close': 100.0, 'ATR': 2.0}, 'BBB': {}}
        inst._journal_rank_near_misses(snapshots)
        skips = _rows_by(rows, action='skip', skip_reason='rank_near_miss')
        assert [s['symbol'] for s in skips] == ['AAA', 'BBB']
        a = skips[0]
        assert a['entry_rank'] == 8
        assert a['pred_return'] == 0.3
        # Expected proxy from the instance's OWN stop constants
        from strategy_config import RISK_PCT_PER_TRADE
        raw = (2.0 * inst.ATR_STOP_MULTIPLIER) / 100.0
        stop_dist = max(inst.ATR_STOP_FLOOR_PCT,
                        min(inst.ATR_STOP_CEIL_PCT, raw))
        expect = min(100_000.0 * RISK_PCT_PER_TRADE / stop_dist,
                     inst.NOTIONAL_PER_SYMBOL)
        assert a['would_be_base_notional'] == round(expect, 2)
        assert a['would_be_stop_dist'] == round(stop_dist, 5)
        # No Close in snapshot -> no proxy fields, row still journaled
        assert 'would_be_base_notional' not in skips[1]

    def test_once_per_day_per_symbol(self, rows):
        inst = _mk_stock(_near_miss_ranked=[(8, 'AAA', 0.30)])
        inst._journal_rank_near_misses({'AAA': {'Close': 100.0, 'ATR': 2.0}})
        inst._journal_rank_near_misses({'AAA': {'Close': 100.0, 'ATR': 2.0}})
        assert len(_rows_by(rows, action='skip',
                            skip_reason='rank_near_miss')) == 1

    def test_only_when_funnel_runs(self, monkeypatch, rows, fast):
        """No near-miss rows when the funnel is pre-empted (flattened /
        stand-down / outside entry window); rows appear once it runs."""
        inst = _stock_funnel(monkeypatch, rows, None)
        inst.top_symbols = []
        inst._near_miss_ranked = [(8, 'AAA', 0.30)]
        inst.flattened_today = True
        inst._execute_buys({}, {})
        assert rows == []
        inst.flattened_today = False
        inst._in_entry_window = lambda: False
        inst._execute_buys({}, {})
        assert rows == []
        inst._in_entry_window = lambda: True
        inst._execute_buys({}, {'AAA': {'Close': 100.0, 'ATR': 2.0}})
        assert len(_rows_by(rows, action='skip',
                            skip_reason='rank_near_miss')) == 1


# ---------------------------------------------------------------------------
# 7. Circuit-breaker trip event journaling
# ---------------------------------------------------------------------------

class TestCircuitBreakerTripRow:
    def _arm(self, monkeypatch, rows, api):
        flat = []
        monkeypatch.setattr(base_loop, 'check_circuit_breaker',
                            lambda a, max_drawdown_pct: (True, 0.06))
        monkeypatch.setattr(base_loop, 'emergency_flatten',
                            lambda a, symbols=None: (flat.append(symbols) or []))
        monkeypatch.setattr(base_loop, 'record_trade', lambda *a, **k: None)
        monkeypatch.setattr(notify, 'notify', lambda *a, **k: None)
        inst = _mk(api=api)
        return inst, flat

    def test_trip_journals_event_row(self, monkeypatch, rows):
        api = SimpleNamespace(get_account=lambda: SimpleNamespace(
            equity='94000', last_equity='100000'))
        inst, flat = self._arm(monkeypatch, rows, api)
        assert inst._circuit_breaker_check() is True
        assert inst._buys_allowed is False
        trips = _rows_by(rows, action='circuit_breaker_trip')
        assert len(trips) == 1
        r = trips[0]
        assert r['asset_type'] == 'crypto'
        assert r['drawdown_pct'] == 6.0
        assert r['baseline_kind'] == 'account_last_equity'
        assert r['baseline_equity'] == 100000.0
        assert r['equity'] == 94000.0
        assert isinstance(r['weekend_window'], bool)
        assert r['n_positions_at_trip'] == 0
        # halted_until matches the breaker's own baseline-reset latch
        assert (_dt.datetime.fromisoformat(r['halted_until'])
                == inst._halted_until)
        assert flat == [['BTC/USD']]   # flatten still ran

    def test_trip_row_never_blocks_flatten(self, monkeypatch, rows):
        def _boom():
            raise RuntimeError('account API down')
        api = SimpleNamespace(get_account=lambda: _boom())
        inst, flat = self._arm(monkeypatch, rows, api)
        assert inst._circuit_breaker_check() is True
        trips = _rows_by(rows, action='circuit_breaker_trip')
        assert len(trips) == 1                    # row still written
        assert trips[0]['baseline_equity'] is None
        assert trips[0]['equity'] is None
        assert trips[0]['drawdown_pct'] == 6.0    # from the check result
        assert flat == [['BTC/USD']]              # flatten still ran

    def test_no_trip_no_row(self, monkeypatch, rows):
        monkeypatch.setattr(base_loop, 'check_circuit_breaker',
                            lambda a, max_drawdown_pct: (False, 0.01))
        inst = _mk()
        assert inst._circuit_breaker_check() is False
        assert inst._buys_allowed is True
        assert _rows_by(rows, action='circuit_breaker_trip') == []


# ---------------------------------------------------------------------------
# 8. decision_report wiring: the new skip classes are replay-priced
# ---------------------------------------------------------------------------

class TestGateReasonsWired:
    """Open question #4's instrument is decision_report.GATE_REASONS —
    skip rows that never reach that list stay unpriced drift. Pins the
    hardener's wiring so a future edit cannot silently orphan the rows."""

    def test_new_skip_reasons_are_priced_classes(self):
        import decision_report as dr
        for g in ('vix_halt', 'vix25_block', 'macro_standdown',
                  'rr5_demotion', 'rank_near_miss'):
            assert g in dr.GATE_REASONS, g
            assert g not in dr.UNPRICED_GATES, g

    def test_vix_vc_counter_keys_unchanged(self):
        # The entry_window veto_counts keys for the two VIX gates keep
        # their legacy names (back-compat with existing journal readers);
        # they are counter keys, never skip_reason row values, so they
        # stay in UNPRICED_GATES and out of GATE_REASONS.
        import decision_report as dr
        assert 'macro_halt' in dr.UNPRICED_GATES
        assert 'vix_block' in dr.UNPRICED_GATES
        assert 'macro_halt' not in dr.GATE_REASONS
        assert 'vix_block' not in dr.GATE_REASONS

    def test_priced_rows_flow_through_gate_attribution(self, monkeypatch):
        """A 'macro_standdown' skip row is replay-priced (appears as a
        per-gate entry), not listed as unclassified producer drift."""
        import decision_report as dr
        import market_data
        import numpy as np
        import pandas as pd
        closes = 100 * np.cumprod(1 + np.full(60, 0.001))
        idx = pd.date_range('2026-06-01', periods=60, freq='h', tz='UTC')
        bars = pd.DataFrame({'Close': closes}, index=idx)
        bars['Open'] = bars['Close'].shift(1).fillna(100.0)
        bars['High'] = bars[['Open', 'Close']].max(axis=1) * 1.001
        bars['Low'] = bars[['Open', 'Close']].min(axis=1) * 0.999
        bars['Volume'] = 1e6
        monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                            lambda api, s, **k: bars)
        rows_in = [{'action': 'skip', 'skip_reason': 'macro_standdown',
                    'symbol': 'AAA/USD', 'ts': str(idx[5])}]
        out = dr.gate_attribution(rows_in, api=object())
        assert 'macro_standdown' in out
        assert out['macro_standdown']['vetoes_priced'] == 1
        assert '_unclassified_skip_reasons' not in out
