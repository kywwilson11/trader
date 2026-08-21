"""c26 packet P6 — base_loop/stock_loop FUNCTIONAL suite (B16 slice).

Real `import base_loop` on the dev Mac under two sys.modules stubs
(dotenv, predict_now — the only two unimportable links, verified);
instances built via object.__new__ on a concrete subclass; every
module-level binding patched at its real seam. Pins POST-P1 behavior:
entry-funnel order, sizing composition incl. the D26 emergency-zero
branch, fail-open/fail-closed contracts, hard-stop-lockout persistence,
D17 per-book flatten consumption, D01 fan-out timeout harvest, D06
TP-leg confirmed-exit journaling.
"""

import concurrent.futures
import datetime as _dt
import json
import os
import sys
import time as _time
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# base_loop's only Mac-unimportable links are predict_now (joblib/torch)
# and trading_utils's `from dotenv import load_dotenv` (verified by import
# probe). Stub exactly those two, import, then RESTORE sys.modules so the
# rest of the suite (baseline import-failure tests, the importorskip
# Jetson tests in test_base_loop_v3.py) sees an unchanged world.
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

import drawdown            # noqa: F401  importable on the Mac; expected math
import meta_label          # importable on the Mac (verified)
import notify              # flatten seam (function-local import in base_loop)
import market_data         # get_live_atr/fetch_bars seams (function-local)
import portfolio           # get_book_vol_scalar_cached seam
import volatility          # get_sigma seam
import order_utils         # cancel_orders_for_symbol seam
from types_mod import Position, MacroRegime
from strategy_config import (RISK_PCT_PER_TRADE, TILT_MAX,
                             MIN_ORDER_NOTIONAL)

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Concrete subclass + factory
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
    """Build a _Loop instance bypassing __init__ (which calls get_api(),
    reads the repo lockout file, and creates a thread pool)."""
    inst = object.__new__(_Loop)
    inst.api = None
    inst.model = object(); inst.config = {}; inst.scaler_X = None
    inst.feature_cols = None
    inst.trade_threshold = 0.15
    inst.positions = {}; inst.last_trade_time = {}
    inst.hard_stop_lockout = {}
    inst._lockout_file = str(tmp_path / 'hard_stop_lockout.json')
    inst.llm_scores = {}; inst._veto_strikes = {}
    inst._last_llm_time = 0.0; inst._last_llm_symbols = set()
    inst._last_stale_force = 0.0; inst._llm_scores_ts = None
    inst._llm_fail_count = 0; inst._llm_backoff_until = 0.0
    inst.model_mtime = 0; inst.cycle = 1
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
    inst._save_position_state = lambda: None     # never touch the repo
    inst._entries_allowed = lambda: True          # bypass halt-flag reads
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


@pytest.fixture
def fast(monkeypatch):
    """Neutralize the jitter sleep (_execute_buys L2306) and the per-buy
    pacing sleep — applied explicitly where a test would otherwise sleep."""
    monkeypatch.setattr(base_loop.time, 'sleep', lambda s: None)
    monkeypatch.setattr(base_loop.random, 'uniform', lambda a, b: 0.0)


# ---------------------------------------------------------------------------
# 2. Entry-funnel ORDER + veto short-circuits (_execute_buys)
# ---------------------------------------------------------------------------

class _RecDict(dict):
    """Records the LLM-gate .get on llm_scores (the only .get on
    llm_scores inside _execute_buys)."""
    events = None

    def get(self, k, d=None):
        self.events.append('llm')
        return super().get(k, d)


def _funnel(monkeypatch, tmp_path, **over):
    """Common arrange for the entry-funnel tests. Returns
    (inst, events, rows, captured)."""
    events = []
    rows = []
    captured = {}
    inst = _mk(tmp_path, **over)
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    monkeypatch.setattr(base_loop, 'should_trade',
                        lambda pred, spread, **k: (events.append('cost') or True))
    monkeypatch.setattr(base_loop, 'sentiment_gate',
                        lambda sym, at: (events.append('sentiment') or (1.0, [])))
    rd = _RecDict()
    rd.events = events
    inst.llm_scores = rd
    inst._meta_gate = (lambda sym, pred, snaps, rank=None:
                       (events.append('meta') or (True, 1.0)))

    def _size(sym, pred, quote, **kw):
        events.append('size')
        captured.update(kw)
        return 500
    inst._compute_position_size = _size
    inst._place_and_track_buy = lambda *a, **k: events.append('buy')
    return inst, events, rows, captured


def _rows_by(rows, **want):
    return [r for r in rows
            if all(r.get(k) == v for k, v in want.items())]


def test_entry_funnel_order_cost_sentiment_llm_meta_size_buy(
        monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert events == ['cost', 'sentiment', 'llm', 'meta', 'size', 'buy']
    win = _rows_by(rows, action='entry_window')
    assert len(win) == 1
    assert win[0]['admitted_k'] == 1
    assert win[0]['n_candidates'] == 1


def test_cost_floor_veto_short_circuits(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    monkeypatch.setattr(base_loop, 'should_trade',
                        lambda pred, spread, **k: (events.append('cost') or False))
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert events == ['cost']
    skips = _rows_by(rows, action='skip', skip_reason='cost_floor')
    assert len(skips) == 1
    assert skips[0]['spread_pct'] == round(0.02, 4)
    assert skips[0]['entry_rank'] == 1
    win = _rows_by(rows, action='entry_window')[0]
    assert win['veto_counts']['cost_floor'] == 1
    assert win['admitted_k'] == 0


def test_llm_veto_blocks_before_meta(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    inst.llm_scores = {'BTC/USD': {'s': 0.05, 'r': 'bad'}}
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert 'meta' not in events
    assert 'size' not in events
    assert 'buy' not in events
    skips = _rows_by(rows, action='skip', skip_reason='llm_veto')
    assert len(skips) == 1
    assert skips[0]['llm_score'] == 0.05


def test_meta_veto_blocks_before_sizing(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    inst._meta_gate = (lambda sym, pred, snaps, rank=None:
                       (events.append('meta') or (False, 1.0)))
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert 'size' not in events
    assert 'buy' not in events
    win = _rows_by(rows, action='entry_window')[0]
    assert win['veto_counts']['meta_veto'] == 1
    # meta_veto's own skip row is written inside _meta_gate (stubbed here),
    # so no skip-row assertion in this test.


def test_sizing_zero_blocks_buy(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)

    def _size0(sym, pred, quote, **kw):
        events.append('size')
        return 0
    inst._compute_position_size = _size0
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert 'buy' not in events
    skips = _rows_by(rows, action='skip', skip_reason='sizing_zero')
    assert len(skips) == 1


def test_missing_pred_fails_closed(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    inst.get_quote = lambda s: pytest.fail('quote fetched without a pred')
    inst._execute_buys({}, {})
    assert events == []
    assert rows == []   # n_candidates == 0 suppresses the window row


def test_missing_quote_fails_closed(monkeypatch, tmp_path, fast):
    inst, events, rows, _ = _funnel(monkeypatch, tmp_path)
    inst._quotes = {}
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert events == []
    assert rows == []   # no skip row; n_candidates == 0 suppresses the window


def test_absent_llm_score_defaults_to_pass(monkeypatch, tmp_path, fast):
    inst, events, rows, captured = _funnel(monkeypatch, tmp_path)
    inst.llm_scores = {}    # plain dict, no score at all
    inst._execute_buys({'BTC/USD': 0.5}, {})
    # 0.5 default score -> llm_mult = 0.5 + 0.5 = 1.0, entry admitted:
    # the LLM layer's fail-open contract at the entry gate.
    assert captured['llm_mult'] == pytest.approx(1.0)
    assert 'buy' in events


def test_sentiment_multiplier_passes_into_sizing(monkeypatch, tmp_path, fast):
    inst, events, rows, captured = _funnel(monkeypatch, tmp_path)
    monkeypatch.setattr(base_loop, 'sentiment_gate',
                        lambda sym, at: (events.append('sentiment')
                                         or (0.15, ['weak'])))
    inst._execute_buys({'BTC/USD': 0.5}, {})
    assert captured['sentiment_mult'] == 0.15
    assert 'buy' in events   # gate <= 0 veto branch not taken


# ---------------------------------------------------------------------------
# 3. _meta_gate contracts (real method; seam = meta_label)
# ---------------------------------------------------------------------------

def test_meta_gate_fail_open_on_exception(monkeypatch, tmp_path):
    inst = _mk(tmp_path)
    monkeypatch.setattr(
        meta_label, 'meta_probability_live',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('schema')))
    inst._last_meta_p = {'BTC/USD': 0.9}
    inst._last_meta_p_cycle = {'BTC/USD': 1}
    assert inst._meta_gate('BTC/USD', 0.5, {}) == (True, 1.0)
    # stale stash popped: a dead gate can't re-journal its last good p
    assert inst._last_meta_p == {}
    assert inst._last_meta_p_cycle == {}


def test_meta_gate_veto_journals_and_blocks(monkeypatch, tmp_path):
    inst = _mk(tmp_path)
    rows = []
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    monkeypatch.setattr(meta_label, 'meta_probability_live',
                        lambda *a, **k: 0.10)
    assert 0.10 < meta_label.META_VETO_PROB   # structural pin (0.30)
    assert inst._meta_gate('BTC/USD', 0.5, {}) == (False, 1.0)
    skips = _rows_by(rows, action='skip', skip_reason='meta_veto')
    assert len(skips) == 1
    assert skips[0]['meta_prob'] == 0.1
    assert inst._last_meta_p['BTC/USD'] == 0.1
    assert inst._last_meta_p_cycle['BTC/USD'] == inst.cycle


def test_meta_gate_pass_returns_size_mult(monkeypatch, tmp_path):
    inst = _mk(tmp_path)
    rows = []
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    monkeypatch.setattr(meta_label, 'meta_probability_live',
                        lambda *a, **k: 0.8)
    allowed, mult = inst._meta_gate('BTC/USD', 0.5, {})
    assert allowed is True
    assert mult == pytest.approx(1.3)   # meta_size_mult clips 2*0.8 to 1.3
    assert _rows_by(rows, action='skip') == []


def test_meta_gate_none_probability_neutral(monkeypatch, tmp_path):
    inst = _mk(tmp_path)
    monkeypatch.setattr(meta_label, 'meta_probability_live',
                        lambda *a, **k: None)
    assert inst._meta_gate('BTC/USD', 0.5, {}) == (True, 1.0)
    assert inst._last_meta_p == {}
    assert inst._last_meta_p_cycle == {}


# ---------------------------------------------------------------------------
# 4. Sizing composition (_compute_position_size, real method)
# ---------------------------------------------------------------------------

def _full_info_macro(sizing_mult=1.0):
    return MacroRegime(stress_level=0.0, vix=12.0, cape=None,
                       regime_label='bull', sizing_mult=sizing_mult,
                       stop_mult=1.0)


def _bars_150():
    # >100 rows so returns is not None, <=200 so the HMM branch never runs
    return pd.DataFrame({'Close': np.linspace(100.0, 115.0, 150)})


def _sizing_seams(monkeypatch, kelly=0.125, bars='full'):
    monkeypatch.setattr(market_data, 'get_live_atr', lambda *a, **k: None)
    monkeypatch.setattr(market_data, 'fetch_bars_alpaca',
                        lambda *a, **k: (_bars_150() if bars == 'full'
                                         else None))
    monkeypatch.setattr(base_loop, 'compute_kelly_fraction',
                        lambda *a, **k: kelly)
    monkeypatch.setattr(portfolio, 'get_book_vol_scalar_cached',
                        lambda api, at: 1.0)
    monkeypatch.setattr(volatility, 'get_sigma', lambda *a, **k: None)


def _full_info_inst(tmp_path, **over):
    # corr_matrix truthy but positions empty: corr factor skipped, yet
    # `not self.corr_matrix` is False (no degraded count from it)
    kw = dict(macro_regime=_full_info_macro(),
              corr_matrix={'ETH/USD': {}})
    kw.update(over)
    return _mk(tmp_path, **kw)


_QUOTE = {'midpoint': 100.0, 'spread_pct': 0.02}


def test_sizing_risk_base_full_information(monkeypatch, tmp_path):
    inst = _full_info_inst(tmp_path)
    _sizing_seams(monkeypatch)
    # base = min(equity * RISK_PCT_PER_TRADE / 0.05, NOTIONAL_PER_SYMBOL)
    expected_base = min(100_000.0 * RISK_PCT_PER_TRADE / 0.05,
                        inst.NOTIONAL_PER_SYMBOL)
    assert expected_base == 1000.0
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 1000
    d = inst._last_sizing_detail
    assert d['base'] == 1000.0
    assert d['kelly_mult'] == 1.0
    assert d['vol_mult'] == 1.0
    assert d['tilt'] == 1.0
    assert 'degraded_inputs' not in d


def test_sizing_tilt_boost_clamped_at_tilt_max(monkeypatch, tmp_path):
    inst = _full_info_inst(tmp_path)
    _sizing_seams(monkeypatch)
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                         llm_mult=1.5)
    assert result == int(1000 * TILT_MAX) == 1300
    d = inst._last_sizing_detail
    assert d['tilt'] == TILT_MAX
    assert d['tilt_raw'] == 1.5


def test_sizing_derisk_floor_0_1(monkeypatch, tmp_path):
    inst = _full_info_inst(tmp_path)
    _sizing_seams(monkeypatch)
    # 0.15 * 0.65 = 0.0975 < 0.1 -> floored at the 0.1 de-risk floor;
    # 100 < MIN_ORDER_NOTIONAL (100) is False, so the order survives
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE),
                                         sentiment_mult=0.15, llm_mult=0.65)
    assert result == int(1000 * 0.1) == 100
    assert result >= MIN_ORDER_NOTIONAL


class _Sentinel:
    def __getattr__(self, name):
        raise AssertionError(f'api touched ({name}) despite emergency zero')


def test_sizing_D26_emergency_zero_before_anything(tmp_path):
    # NO seam patched at all: the return must precede every import/API touch
    inst = _mk(tmp_path, macro_regime=_full_info_macro(sizing_mult=0.0),
               api=_Sentinel())
    assert inst._compute_position_size('BTC/USD', 0.5,
                                       {'midpoint': 100.0}) == 0


def test_sizing_nonzero_macro_emergency_still_floored(monkeypatch, tmp_path):
    # Pins that the emergency branch triggers on EXACTLY 0.0 and any
    # nonzero de-risk saturates the 0.1 floor (documented pre-B7 default).
    inst = _full_info_inst(tmp_path,
                           macro_regime=_full_info_macro(sizing_mult=0.05))
    _sizing_seams(monkeypatch)
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 100


def test_sizing_degraded_mode_caps_tilt(monkeypatch, tmp_path):
    inst = _mk(tmp_path, macro_regime=None, corr_matrix={})
    _sizing_seams(monkeypatch, kelly=None, bars='none')
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 500   # 1000 * 0.5 degraded cap
    d = inst._last_sizing_detail
    assert d['degraded_inputs'] == 4
    assert d['tilt'] == 0.5


def test_sizing_dust_returns_zero(monkeypatch, tmp_path):
    inst = _mk(tmp_path, macro_regime=None, corr_matrix={})
    inst.NOTIONAL_PER_SYMBOL = 150
    _sizing_seams(monkeypatch, kelly=None, bars='none')
    # sized 150 * 0.5 = 75 < MIN_ORDER_NOTIONAL -> 0
    result = inst._compute_position_size('BTC/USD', 0.15, dict(_QUOTE))
    assert result == 0


# ---------------------------------------------------------------------------
# 5. Hard-stop lockout persistence
# ---------------------------------------------------------------------------

def test_lockout_save_load_roundtrip(tmp_path):
    a = _mk(tmp_path)
    now = _dt.datetime.now()
    a.hard_stop_lockout = {'BTC/USD': now}
    a._save_hard_stop_lockout()
    with open(a._lockout_file) as f:
        data = json.load(f)
    assert list(data) == ['BTC/USD']
    expected_expiry = (now + _dt.timedelta(
        hours=a.HARD_STOP_LOCKOUT_HOURS)).timestamp()
    assert data['BTC/USD'] == pytest.approx(expected_expiry, abs=1.0)
    assert a.HARD_STOP_LOCKOUT_HOURS == 24

    b = _mk(tmp_path)
    b._load_hard_stop_lockout()
    assert 'BTC/USD' in b.hard_stop_lockout
    assert b._is_hard_stop_locked('BTC/USD') is True
    restored = b.hard_stop_lockout['BTC/USD']
    assert abs((restored - now).total_seconds()) < 5


def test_lockout_expired_not_loaded(tmp_path):
    inst = _mk(tmp_path)
    with open(inst._lockout_file, 'w') as f:
        json.dump({'BTC/USD': _time.time() - 60}, f)
    inst._load_hard_stop_lockout()
    assert inst.hard_stop_lockout == {}


def test_lockout_expiry_deletes_key_and_rewrites_file(tmp_path):
    inst = _mk(tmp_path)
    inst.hard_stop_lockout = {
        'BTC/USD': _dt.datetime.now() - _dt.timedelta(hours=25)}
    assert inst._is_hard_stop_locked('BTC/USD') is False
    assert 'BTC/USD' not in inst.hard_stop_lockout
    with open(inst._lockout_file) as f:
        assert json.load(f) == {}   # the expiry path SAVES


def test_lockout_corrupt_file_fail_soft(tmp_path):
    inst = _mk(tmp_path)
    with open(inst._lockout_file, 'w') as f:
        f.write('not json{')
    inst._load_hard_stop_lockout()   # must not raise
    assert inst.hard_stop_lockout == {}


def test_hard_stop_exit_sets_lockout_and_persists(monkeypatch, tmp_path):
    trades = []
    rows = []
    monkeypatch.setattr(order_utils, 'cancel_orders_for_symbol',
                        lambda api, sym, timeout=5: True)
    monkeypatch.setattr(base_loop, 'manage_order_lifecycle',
                        lambda *a, **k: SimpleNamespace(
                            status='filled', filled_avg_price='99.0'))
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: trades.append((a, k)))
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    api = SimpleNamespace(
        submit_order=lambda **k: SimpleNamespace(id='o1'))
    pos = Position(qty=1.0, entry_price=100.0, high_water_mark=100.0)
    inst = _mk(tmp_path, api=api, positions={'BTC/USD': pos})

    inst._execute_stop_exit('BTC/USD', pos, 'hard_stop', 100.0)

    assert len(trades) == 1
    args, kwargs = trades[0]
    assert kwargs['exit_reason'] == 'hard_stop'
    assert kwargs['estimated'] is False
    assert args[3] == 99.0                       # real fill price
    assert args[4] == pytest.approx(-1.0)        # pnl_pct
    sells = _rows_by(rows, action='sell')
    assert len(sells) == 1
    assert sells[0]['slippage_bps'] == 100.0     # (100-99)/100*1e4, sell sign
    assert sells[0]['fill_price'] == 99.0
    assert 'BTC/USD' not in inst.positions
    assert 'BTC/USD' in inst.hard_stop_lockout
    with open(inst._lockout_file) as f:
        assert 'BTC/USD' in json.load(f)         # persisted to disk
    fresh = _mk(tmp_path)
    fresh._load_hard_stop_lockout()
    assert fresh._is_hard_stop_locked('BTC/USD') is True


# ---------------------------------------------------------------------------
# 6. D17 per-book flatten consumption (_check_flatten_request)
# ---------------------------------------------------------------------------

def _flatten_seams(monkeypatch, tmp_path, requested, flatten_result):
    """Patch the flatten seams; returns a record dict."""
    rec = {'clear': 0, 'halt': [], 'notify': [], 'flatten_calls': [],
           'trades': []}
    monkeypatch.setattr(base_loop, '_FLATTEN_FLAG_DIR', str(tmp_path))
    monkeypatch.setattr(notify, 'flatten_requested', lambda: requested)
    monkeypatch.setattr(notify, 'clear_flatten_request',
                        lambda: rec.__setitem__('clear', rec['clear'] + 1))
    monkeypatch.setattr(notify, 'set_halt',
                        lambda reason='': rec['halt'].append(reason))
    monkeypatch.setattr(notify, 'notify',
                        lambda msg, **k: rec['notify'].append(msg))

    def _flatten(api, symbols=None):
        rec['flatten_calls'].append(symbols)
        return flatten_result
    monkeypatch.setattr(base_loop, 'emergency_flatten', _flatten)
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: rec['trades'].append((a, k)))
    return rec


def _pos():
    return Position(qty=1.0, entry_price=100.0, high_water_mark=100.0)


def test_flatten_legacy_fans_out_consumes_own_leaves_stock(
        monkeypatch, tmp_path):
    rec = _flatten_seams(monkeypatch, tmp_path, requested=True,
                         flatten_result=[])
    inst = _mk(tmp_path, positions={'BTC/USD': _pos()})
    inst.get_quote = lambda s: None    # px falls back to entry_price
    inst._check_flatten_request()
    assert rec['clear'] == 1
    assert not os.path.exists(tmp_path / 'flatten_crypto.flag')  # consumed
    assert os.path.exists(tmp_path / 'flatten_stock.flag')  # left for stock
    assert rec['flatten_calls'] == [['BTC/USD']]
    assert rec['halt'] == ['remote flatten']
    assert inst.positions == {}
    assert len(rec['trades']) == 1
    args, kwargs = rec['trades'][0]
    assert args[0] == 'BTC/USD'
    assert kwargs['exit_reason'] == 'remote_flatten'
    assert kwargs['estimated'] is True


def test_flatten_stale_own_flag_discarded_not_actioned(monkeypatch, tmp_path):
    rec = _flatten_seams(monkeypatch, tmp_path, requested=False,
                         flatten_result=[])
    flag = tmp_path / 'flatten_crypto.flag'
    flag.write_text(str(_time.time()))
    old = _time.time() - 7200   # 2h > FLATTEN_FLAG_STALE_SEC (3600)
    os.utime(flag, (old, old))
    inst = _mk(tmp_path, positions={'BTC/USD': _pos()})
    inst._check_flatten_request()
    assert rec['flatten_calls'] == []          # never actioned
    assert rec['halt'] == []
    assert not flag.exists()                   # consumed (removed) anyway
    assert inst.positions != {}


def test_flatten_broker_format_failures_keep_positions(monkeypatch, tmp_path):
    rec = _flatten_seams(monkeypatch, tmp_path, requested=True,
                         flatten_result=['BTCUSD'])   # broker format
    inst = _mk(tmp_path,
               positions={'BTC/USD': _pos(), 'ETH/USD': _pos()},
               _universe=['BTC/USD', 'ETH/USD'])
    inst.get_quote = lambda s: None
    inst._check_flatten_request()
    assert set(inst.positions) == {'BTC/USD'}   # failed_norm slash-strip
    assert len(rec['trades']) == 1
    assert rec['trades'][0][0][0] == 'ETH/USD'  # only the released one
    assert any('INCOMPLETE' in m for m in rec['notify'])

    # Sentinel case: unknown broker state keeps ALL positions, journals none
    rec2 = _flatten_seams(monkeypatch, tmp_path, requested=True,
                          flatten_result=['<list_positions failed>'])
    inst2 = _mk(tmp_path,
                positions={'BTC/USD': _pos(), 'ETH/USD': _pos()},
                _universe=['BTC/USD', 'ETH/USD'])
    inst2.get_quote = lambda s: None
    inst2._check_flatten_request()
    assert set(inst2.positions) == {'BTC/USD', 'ETH/USD'}
    assert rec2['trades'] == []


def test_flatten_other_books_flag_ignored(monkeypatch, tmp_path):
    rec = _flatten_seams(monkeypatch, tmp_path, requested=False,
                         flatten_result=[])
    stock_flag = tmp_path / 'flatten_stock.flag'
    stock_flag.write_text(str(_time.time()))
    inst = _mk(tmp_path, positions={'BTC/USD': _pos()})
    inst._check_flatten_request()
    assert rec['flatten_calls'] == []
    assert rec['halt'] == []
    assert stock_flag.exists()                 # stock flag survives
    assert inst.positions != {}


# ---------------------------------------------------------------------------
# 7. D01 fan-out timeout harvest (_get_predictions)
# ---------------------------------------------------------------------------

class _FakePool:
    def __init__(self, futs):
        self._futs = list(futs)

    def submit(self, fn, *a, **k):
        return self._futs.pop(0)


def _pred_env(monkeypatch):
    rows = []
    stub = types.ModuleType('monitor_drift')
    stub_calls = []
    stub.log_predictions = lambda *a, **k: stub_calls.append((a, k))
    monkeypatch.setitem(sys.modules, 'monitor_drift', stub)
    monkeypatch.setattr(base_loop, 'choose_inference_device', lambda: 'cpu')
    monkeypatch.setattr(base_loop, 'log_decision', rows.append)
    return rows


def _done_future(value):
    f = concurrent.futures.Future()
    f.set_result(value)
    return f


def test_fanout_timeout_harvests_done_and_rebuilds_pool(
        monkeypatch, tmp_path):
    rows = _pred_env(monkeypatch)
    f_done = _done_future(('AAA', 0.5, {'x': 1}))
    f_stuck = concurrent.futures.Future()   # never resolves
    fake = _FakePool([f_done, f_stuck])
    inst = _mk(tmp_path, _universe=['AAA', 'BBB'])
    inst._prediction_pool = fake
    inst.PREDICTION_TIMEOUT_SEC = 1
    try:
        preds, snaps = inst._get_predictions(None)
        assert preds == {'AAA': 0.5}
        assert snaps == {'AAA': {'x': 1}}
        assert f_stuck.cancelled() is True
        t_rows = _rows_by(rows, action='pred_fanout_timeout')
        assert len(t_rows) == 1
        assert t_rows[0]['wedged'] == ['BBB']
        assert t_rows[0]['timeout_s'] == 1
        assert isinstance(inst._prediction_pool,
                          concurrent.futures.ThreadPoolExecutor)
        assert inst._prediction_pool is not fake
    finally:
        pool = inst._prediction_pool
        if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
            pool.shutdown(wait=False)


def test_fanout_all_complete_no_rebuild(monkeypatch, tmp_path):
    rows = _pred_env(monkeypatch)
    fake = _FakePool([_done_future(('AAA', 0.5, {'x': 1})),
                      _done_future(('BBB', 0.3, None))])
    inst = _mk(tmp_path, _universe=['AAA', 'BBB'])
    inst._prediction_pool = fake
    inst.PREDICTION_TIMEOUT_SEC = 5
    preds, snaps = inst._get_predictions(None)
    assert preds == {'AAA': 0.5, 'BBB': 0.3}
    assert snaps == {'AAA': {'x': 1}}
    assert _rows_by(rows, action='pred_fanout_timeout') == []
    assert inst._prediction_pool is fake       # untouched


def test_fanout_prunes_departed_meta_stash(monkeypatch, tmp_path):
    _pred_env(monkeypatch)
    fake = _FakePool([_done_future(('AAA', 0.5, None))])
    inst = _mk(tmp_path, _universe=['AAA'])
    inst._prediction_pool = fake
    inst.PREDICTION_TIMEOUT_SEC = 5
    inst._last_meta_p = {'GONE/USD': 0.7, 'AAA': 0.6}
    inst._last_meta_p_cycle = {'GONE/USD': 3, 'AAA': 3}
    inst._get_predictions(None)
    assert set(inst._last_meta_p) == {'AAA'}
    assert set(inst._last_meta_p_cycle) == {'AAA'}


# ---------------------------------------------------------------------------
# 8. D06 TP-leg confirmed-exit journaling (stock_loop)
# ---------------------------------------------------------------------------

def _mk_stock(**over):
    inst = object.__new__(stock_loop.StockLoop)
    inst.llm_scores = {}
    inst._tp_order_ids = {}
    inst.last_trade_time = {}   # stale-stub modernization: F10 time filter
    inst.get_quote = lambda s: {'midpoint': 110.0}
    for k, v in over.items():
        setattr(inst, k, v)
    return inst


def _stock_journal_seams(monkeypatch):
    """Capture BOTH journal paths: confirmed (base_loop namespace) and
    estimated (stock_loop.record_trade + trade_journal's log_decision,
    the estimated path's function-local import)."""
    rec = {'base_trades': [], 'base_rows': [],
           'est_trades': [], 'est_rows': []}
    monkeypatch.setattr(base_loop, 'record_trade',
                        lambda *a, **k: rec['base_trades'].append((a, k)))
    monkeypatch.setattr(base_loop, 'log_decision',
                        lambda r: rec['base_rows'].append(r))
    monkeypatch.setattr(stock_loop, 'record_trade',
                        lambda *a, **k: rec['est_trades'].append((a, k)))
    monkeypatch.setattr(sys.modules['trade_journal'], 'log_decision',
                        lambda r: rec['est_rows'].append(r))
    return rec


def test_external_close_tp_leg_writes_confirmed_row(monkeypatch):
    rec = _stock_journal_seams(monkeypatch)
    api = SimpleNamespace(get_order=lambda oid: SimpleNamespace(
        status='filled', filled_avg_price='103.5'))
    inst = _mk_stock(api=api, _tp_order_ids={'NVDA': 'tp1'})
    info = Position(qty=10, entry_price=100.0, high_water_mark=100.0,
                    stop_order_id=None)
    inst._journal_external_close('NVDA', info)
    assert len(rec['base_trades']) == 1
    args, kwargs = rec['base_trades'][0]
    assert kwargs['exit_reason'] == 'take_profit'
    assert kwargs['estimated'] is False        # Kelly-eligible
    assert args[3] == 103.5
    assert args[4] == pytest.approx(3.5)
    sells = [r for r in rec['base_rows'] if r.get('action') == 'sell']
    assert len(sells) == 1
    assert sells[0]['estimated'] is False
    assert sells[0]['fill_price'] == 103.5
    assert sells[0]['decision_price'] is None  # quote=None recovery path
    assert sells[0]['slippage_bps'] is None
    assert rec['est_trades'] == []             # estimated path never fired


def test_external_close_server_stop_probe(monkeypatch):
    rec = _stock_journal_seams(monkeypatch)
    api = SimpleNamespace(get_order=lambda oid: SimpleNamespace(
        status='filled', filled_avg_price='97.0'))
    inst = _mk_stock(api=api, _tp_order_ids={})
    info = Position(qty=10, entry_price=100.0, high_water_mark=100.0,
                    stop_order_id='st1')
    inst._journal_external_close('NVDA', info)
    assert len(rec['base_trades']) == 1
    args, kwargs = rec['base_trades'][0]
    assert kwargs['exit_reason'] == 'server_stop'
    assert kwargs['estimated'] is False
    assert args[3] == 97.0                     # confirmed stop fill
    assert args[4] == pytest.approx(-3.0)


def test_external_close_closed_orders_fallback(monkeypatch):
    rec = _stock_journal_seams(monkeypatch)
    api = SimpleNamespace(
        get_order=lambda oid: pytest.fail('get_order probed with no ids'),
        list_orders=lambda **k: [SimpleNamespace(
            side='sell', status='filled', filled_avg_price='99.0',
            # stale-stub modernization: F10's time filter skips orders with
            # no verifiable fill time — stamp a fresh one
            filled_at=_dt.datetime.now(_dt.timezone.utc).isoformat())])
    inst = _mk_stock(api=api, _tp_order_ids={})
    info = Position(qty=10, entry_price=100.0, high_water_mark=100.0,
                    stop_order_id=None)
    inst._journal_external_close('NVDA', info)
    assert len(rec['base_trades']) == 1
    args, kwargs = rec['base_trades'][0]
    assert kwargs['exit_reason'] == 'external_close'
    assert kwargs['estimated'] is False
    assert args[3] == 99.0                     # newest filled sell's price


def test_external_close_unrecoverable_falls_back_to_estimate(monkeypatch):
    rec = _stock_journal_seams(monkeypatch)

    def _boom(oid):
        raise RuntimeError('api down')
    api = SimpleNamespace(get_order=_boom, list_orders=lambda **k: [])
    inst = _mk_stock(api=api, _tp_order_ids={'NVDA': 'tp1'})
    info = Position(qty=10, entry_price=100.0, high_water_mark=100.0,
                    stop_order_id=None)
    inst._journal_external_close('NVDA', info)
    assert rec['base_trades'] == []            # confirmed recorders empty
    assert len(rec['est_trades']) == 1
    args, kwargs = rec['est_trades'][0]
    assert kwargs['exit_reason'] == 'external_close'
    assert kwargs['estimated'] is True
    assert args[3] == 110.0                    # quote midpoint estimate
    est_sells = [r for r in rec['est_rows'] if r.get('action') == 'sell']
    assert len(est_sells) == 1
    assert est_sells[0]['estimated'] is True


def test_external_close_tp_id_pruned_when_position_gone():
    inst = _mk_stock(positions={}, _tp_order_ids={'GONE': 'tp9'},
                     _pending_breach={})
    inst._save_position_state = lambda: None   # base tail write suppressed
    inst._manage_stops()
    assert inst._tp_order_ids == {}
