"""2026-07 review batch b01: crypto_loop / stock_loop / crypto_trend fixes.

crypto_trend is pure numpy -> tested by direct import. The two loop modules
need the Jetson stack to import, so (following tests/test_prediction_cache.py)
they are covered by source guards plus extract-and-exec functional tests of
the individual methods with stubbed collaborators.

Covers:
  - crypto_trend: smooth_state persistence=1 off-by-one; fail-HOLD docstring;
    sma_gap NaN-compaction docs.
  - crypto_loop: Volume_Ratio dead-code removal; place_buy_order delegation;
    atomic prediction-cache write; resting-stop price pruning; funding-tilt
    failure logging; docstring/constant dedup.
  - stock_loop: flatten_before_close transient-error retry (P1); _manage_stops
    stop-id retention on transient errors (P2); buy-journal sizing
    decomposition (P2); Stage-0 counter placement + qty_zero attribution;
    external_close log_decision row; atomic cache write; cosmetic rot.
"""
import ast
import datetime
import json
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crypto_trend import sma_gap, hysteresis_state, smooth_state

REPO = Path(__file__).resolve().parent.parent
CRYPTO_SRC = (REPO / "crypto_loop.py").read_text()
STOCK_SRC = (REPO / "stock_loop.py").read_text()


# ---------------------------------------------------------------------------
# Extraction helpers (pattern: exec one method with stubbed globals)
# ---------------------------------------------------------------------------

def _extract_method(src: str, class_name: str, method_name: str,
                    replace: dict | None = None) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, ast.FunctionDef)
                        and item.name == method_name):
                    seg = textwrap.dedent(ast.get_source_segment(src, item))
                    for old, new in (replace or {}).items():
                        assert old in seg, f"{old!r} not in {method_name}"
                        seg = seg.replace(old, new)
                    return seg
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_method(src, class_name, method_name, glb, replace=None):
    seg = _extract_method(src, class_name, method_name, replace=replace)
    ns = dict(glb)
    exec(compile(seg, f"<{method_name}>", "exec"), ns)
    return ns[method_name]


class _Log:
    """Minimal logger stub recording (level, message%args)."""

    def __init__(self):
        self.lines = []

    def _rec(self, level, msg, *args):
        self.lines.append((level, (msg % args) if args else msg))

    def debug(self, msg, *a):
        self._rec('debug', msg, *a)

    def info(self, msg, *a):
        self._rec('info', msg, *a)

    def warning(self, msg, *a):
        self._rec('warning', msg, *a)

    def error(self, msg, *a):
        self._rec('error', msg, *a)


# ---------------------------------------------------------------------------
# crypto_trend — smooth_state persistence=1 off-by-one (behavior fix)
# ---------------------------------------------------------------------------

def test_smooth_state_persistence_one_tracks_raw_exactly():
    # The reviewer's reproducer: pre-fix this returned ['risk_on','risk_on']
    assert smooth_state(['risk_on', 'risk_off'], persistence=1) == \
        ['risk_on', 'risk_off']
    raw = ['risk_on', 'risk_off', 'risk_off', 'risk_on',
           'risk_off', 'risk_on', 'risk_on', 'risk_off']
    assert smooth_state(raw, persistence=1) == raw


def test_smooth_state_persistence_ge2_provably_unchanged():
    # Hand-traced expectations identical to the pre-fix implementation
    # (the else-branch commit check is a no-op when count=1 < persistence).
    assert smooth_state(['risk_on', 'risk_on', 'risk_off', 'risk_on',
                         'risk_on'], persistence=3) == ['risk_on'] * 5
    assert smooth_state(['risk_on', 'risk_off', 'risk_off', 'risk_off',
                         'risk_off'], persistence=3) == \
        ['risk_on', 'risk_on', 'risk_on', 'risk_off', 'risk_off']
    assert smooth_state(['risk_on', 'risk_off'] * 10, persistence=2) == \
        ['risk_on'] * 20
    assert smooth_state(['risk_on', 'risk_off', 'risk_off', 'risk_on'],
                        persistence=2) == \
        ['risk_on', 'risk_on', 'risk_off', 'risk_off']
    assert smooth_state([], persistence=3) == []


# ---------------------------------------------------------------------------
# crypto_trend — docstring corrections (fail-HOLD label, NaN compaction)
# ---------------------------------------------------------------------------

def test_hysteresis_docstring_says_fail_hold_and_behavior_matches():
    doc = hysteresis_state.__doc__
    assert 'fail-hold' in doc.lower()
    assert 'not fail-open' in doc.lower()
    # The behavior the label describes: a dead feed FREEZES the state, so a
    # prior risk_off persists (the restrictive outcome, not fail-open).
    assert hysteresis_state(None, 'risk_off') == 'risk_off'
    assert hysteresis_state(float('nan'), 'risk_off') == 'risk_off'
    assert hysteresis_state(None, 'risk_on') == 'risk_on'


def test_sma_gap_docstring_documents_finite_compaction():
    doc = sma_gap.__doc__
    assert 'FINITE' in doc
    # Behavior the docstring now states: NaNs are compacted out, so the
    # gap equals the gap of the finite subsequence (numerator = last
    # finite close, window stretched across NaN voids).
    base = np.linspace(100.0, 120.0, 200)
    with_tail = np.concatenate([base, [np.nan, np.nan, np.nan]])
    assert sma_gap(with_tail, window=200) == pytest.approx(
        sma_gap(base, window=200))
    void = np.concatenate([base[:100], np.full(150, np.nan), base[100:]])
    assert sma_gap(void, window=200) == pytest.approx(
        sma_gap(base, window=200))


# ---------------------------------------------------------------------------
# crypto_loop — Volume_Ratio dead code removed (fetch + injection + state)
# ---------------------------------------------------------------------------

def test_crypto_volume_ratio_dead_code_removed():
    assert '_volume_ratios' not in CRYPTO_SRC
    assert 'fetch_crypto_volume' not in CRYPTO_SRC
    assert 'Volume_Ratio' not in CRYPTO_SRC


# ---------------------------------------------------------------------------
# crypto_loop — place_buy_order reduced to a delegation stub
# ---------------------------------------------------------------------------

def test_crypto_place_buy_order_delegates_to_entry_order():
    assert 'place_limit_order' not in CRYPTO_SRC  # dead import dropped
    fn = _load_method(CRYPTO_SRC, 'CryptoLoop', 'place_buy_order', {})
    calls = []
    me = SimpleNamespace(
        _execute_entry_order=lambda *a: (calls.append(a) or ('ORDER', 'maker_join')))
    assert fn(me, 'BTC/USD', 1000, {'midpoint': 5.0}) == 'ORDER'
    assert calls == [('BTC/USD', 1000, {'midpoint': 5.0})]


# ---------------------------------------------------------------------------
# crypto_loop + stock_loop — atomic prediction-cache writes (tmp + os.replace)
# ---------------------------------------------------------------------------

def test_write_prediction_cache_source_uses_os_replace():
    for src in (CRYPTO_SRC, STOCK_SRC):
        seg = _extract_method(src, 'CryptoLoop' if src is CRYPTO_SRC
                              else 'StockLoop', 'write_prediction_cache')
        assert 'os.replace' in seg
        assert ".with_suffix('.tmp')" in seg
        # the non-atomic in-place open is gone
        assert "open(_PRED_CACHE_FILE, 'w')" not in seg


def test_crypto_prediction_cache_atomic_write_end_to_end(tmp_path):
    import os
    cache = tmp_path / 'crypto_predictions.json'
    fn = _load_method(CRYPTO_SRC, 'CryptoLoop', 'write_prediction_cache',
                      {'json': json, 'datetime': datetime, 'os': os,
                       'logger': _Log(), '_PRED_CACHE_FILE': cache})
    me = SimpleNamespace(trade_threshold=0.15)
    fn(me, {'BTC/USD': 0.5, 'ETH/USD': -0.3, 'SOL/USD': 0.01, 'DOGE/USD': None})
    data = json.loads(cache.read_text())          # complete, parseable JSON
    assert data['BTC/USD']['signal'] == 'BULL'
    assert data['ETH/USD']['signal'] == 'BEAR'
    assert data['SOL/USD']['signal'] == 'NEUTRAL'
    assert data['DOGE/USD'] == {'pred': None, 'score': 0, 'signal': 'NEUTRAL',
                                'updated': data['DOGE/USD']['updated']}
    assert not (tmp_path / 'crypto_predictions.tmp').exists()  # tmp renamed away


def test_stock_prediction_cache_atomic_write_end_to_end(tmp_path):
    import os
    cache = tmp_path / 'stock_predictions.json'
    fn = _load_method(STOCK_SRC, 'StockLoop', 'write_prediction_cache',
                      {'json': json, 'datetime': datetime, 'os': os,
                       'logger': _Log(), '_PRED_CACHE_FILE': cache})
    me = SimpleNamespace(trade_threshold=0.15, top_symbols=['AAPL'])
    fn(me, {'AAPL': 0.5, 'XOM': -0.4, 'KO': 0.0})
    data = json.loads(cache.read_text())
    assert data['AAPL']['signal'] == 'BULL'
    assert data['XOM']['signal'] == 'BEAR'
    assert data['KO']['signal'] == 'NEUTRAL'
    assert not (tmp_path / 'stock_predictions.tmp').exists()


# ---------------------------------------------------------------------------
# crypto_loop — stale resting-stop prices pruned when positions drop
# ---------------------------------------------------------------------------

def test_crypto_resting_stop_px_pruned_for_dropped_positions():
    fn = _load_method(CRYPTO_SRC, 'CryptoLoop', '_manage_stops', {},
                      replace={'super()._manage_stops()': 'pass'})
    me = SimpleNamespace(
        _resting_stop_px={'BTC/USD': 50_000.0, 'ETH/USD': 3_000.0},
        positions={'ETH/USD': object()})
    fn(me)
    # BTC exited (position gone) -> its level pruned; live ETH kept
    assert me._resting_stop_px == {'ETH/USD': 3_000.0}


# ---------------------------------------------------------------------------
# crypto_loop — funding-tilt failure is logged, not swallowed silently
# ---------------------------------------------------------------------------

def test_crypto_extra_tilt_logs_on_failure():
    seg = _extract_method(CRYPTO_SRC, 'CryptoLoop', '_extra_tilt')
    assert 'except Exception as e:' in seg
    assert 'logger.debug' in seg
    assert '[FUNDING]' in seg


# ---------------------------------------------------------------------------
# crypto_loop — docstring order + base-default constants deduplicated
# ---------------------------------------------------------------------------

def test_crypto_docstring_matches_cycle_order_and_constants_dedup():
    doc = ast.get_docstring(ast.parse(CRYPTO_SRC))
    # base _run_one_cycle manages stops BEFORE predictions
    assert doc.index('stop-loss') < doc.index('predictions')
    assert 'circuit-breaker' in doc
    # redundant restatements of base defaults are gone (ATR block stays)
    for name in ('NOTIONAL_PER_SYMBOL', 'MAX_NOTIONAL_PER_SYMBOL',
                 'ORDER_TIMEOUT =', 'LOOP_INTERVAL',
                 'MAX_PREDICTION_WORKERS', 'LLM_INTERVAL_SEC',
                 'CIRCUIT_BREAKER_PCT'):
        assert name not in CRYPTO_SRC, f"{name} still restated in crypto_loop"
    assert 'ATR_STOP_MULTIPLIER' in CRYPTO_SRC  # policy-driven block kept


# ---------------------------------------------------------------------------
# stock_loop — flatten_before_close (P1): transient get_position errors must
# retry, not silently drop tracking; not-found closes are journaled
# ---------------------------------------------------------------------------

class _FlattenSelf:
    def __init__(self, positions, api):
        self.flattened_today = False
        self.positions = positions
        self.api = api
        self.llm_scores = {}
        self.journaled = []

    def _in_flatten_window(self):
        return True

    def _select_overnight_keepers(self):
        return set()

    def _prepare_overnight_keepers(self, keepers):
        pass

    def get_symbol_universe(self):
        return ['AAPL', 'MSFT']

    def _journal_external_close(self, symbol, info):
        self.journaled.append(symbol)

    def _record_confirmed_exit(self, *a, **k):
        pass


def _flatten_fn(monkeypatch):
    import time as _time
    fake_notify = types.ModuleType('notify')
    fake_notify.notify = lambda *a, **k: None
    monkeypatch.setitem(__import__('sys').modules, 'notify', fake_notify)
    glb = {
        'logger': _Log(), 'time': _time,
        'get_all_positions': lambda api: None,          # no orphans
        'get_stock_quote': lambda api, s: None,
        'place_stock_limit_order': lambda *a, **k: None,
        'manage_order_lifecycle': lambda *a, **k: None,
        'cancel_orders_for_symbol': lambda *a, **k: True,
        'make_client_order_id': lambda p: f'{p}-1',
        'datetime': datetime,
    }
    return _load_method(STOCK_SRC, 'StockLoop', 'flatten_before_close', glb)


def _api_raising(msg):
    def get_position(symbol):
        raise Exception(msg)
    return SimpleNamespace(get_position=get_position)


def test_flatten_transient_error_keeps_tracking_and_retries(monkeypatch):
    fn = _flatten_fn(monkeypatch)
    info = SimpleNamespace(qty=5, entry_price=100.0, stop_order_id=None)
    me = _FlattenSelf({'AAPL': info}, _api_raising('request timed out (429)'))
    fn(me)
    assert me.flattened_today is False        # NOT marked done -> retries
    assert 'AAPL' in me.positions             # tracking kept
    assert me.journaled == []                 # no phantom external_close


def test_flatten_not_found_journals_external_close_and_completes(monkeypatch):
    fn = _flatten_fn(monkeypatch)
    info = SimpleNamespace(qty=5, entry_price=100.0, stop_order_id=None)
    me = _FlattenSelf({'AAPL': info}, _api_raising('position not found (404)'))
    fn(me)
    assert me.journaled == ['AAPL']           # broker-side close journaled
    assert 'AAPL' not in me.positions
    assert me.flattened_today is True         # book empty -> done


# ---------------------------------------------------------------------------
# stock_loop — _manage_stops (P2): keep stop_order_id on transient errors
# ---------------------------------------------------------------------------

def _manage_stops_fn(log):
    glb = {
        'logger': log, 'datetime': datetime,
        'get_stock_quote': lambda api, s: None,   # stop after the id check
        'cancel_orders_for_symbol': lambda *a, **k: True,
        'make_client_order_id': lambda p: f'{p}-1',
    }
    return _load_method(STOCK_SRC, 'StockLoop', '_manage_stops', glb,
                        replace={'super()._manage_stops()': 'pass'})


def _stops_self(api):
    info = SimpleNamespace(stop_order_id='oid-1', qty=5, entry_price=100.0,
                           high_water_mark=100.0, entry_atr=None,
                           trailing_activated=False, take_profit_price=None)
    me = SimpleNamespace(positions={'AAPL': info}, api=api, llm_scores={},
                         last_trade_time={}, hard_stop_lockout={},
                         _save_hard_stop_lockout=lambda: None,
                         _record_confirmed_exit=lambda *a, **k: None)
    return me, info


def test_manage_stops_keeps_stop_id_on_transient_error():
    log = _Log()
    fn = _manage_stops_fn(log)

    def get_order(oid):
        raise Exception('rate limit exceeded')
    me, info = _stops_self(SimpleNamespace(get_order=get_order))
    fn(me)
    assert info.stop_order_id == 'oid-1'      # retained -> retried next cycle
    assert any('get_order failed' in line for _, line in log.lines)


def test_manage_stops_clears_stop_id_only_when_order_gone():
    fn = _manage_stops_fn(_Log())

    def get_order(oid):
        raise Exception('order not found')
    me, info = _stops_self(SimpleNamespace(get_order=get_order))
    fn(me)
    assert info.stop_order_id is None         # genuinely gone -> cleared


# ---------------------------------------------------------------------------
# stock_loop — buy journal attaches the sizing decomposition (P2)
# ---------------------------------------------------------------------------

def test_stock_buy_rec_attaches_sizing_and_entry_tactic():
    seg = _extract_method(STOCK_SRC, 'StockLoop', '_execute_buys')
    assert "_last_sizing_detail" in seg
    assert "buy_rec['sizing'] = sizing_detail" in seg
    assert '"entry_tactic": "marketable_bracket"' in seg
    # the stash must be read AFTER buy_rec is built and BEFORE it is logged
    assert (seg.index('buy_rec = {') < seg.index("buy_rec['sizing']")
            < seg.index('log_decision(buy_rec)'))


# ---------------------------------------------------------------------------
# stock_loop — Stage-0 counters: n_candidates after quote; qty==0 attributed
# ---------------------------------------------------------------------------

def test_stock_candidate_counter_after_quote_and_qty_zero_attributed():
    seg = _extract_method(STOCK_SRC, 'StockLoop', '_execute_buys')
    assert seg.count('n_candidates += 1') == 1
    # count only past pred+quote, matching base_loop's entry_window rows
    assert (seg.index("vc['no_quote']") < seg.index('n_candidates += 1')
            < seg.index("vc['cost_floor']"))
    assert "vc['qty_zero']" in seg
    assert "_journal_skip(symbol, 'qty_zero'" in seg


# ---------------------------------------------------------------------------
# stock_loop — external_close writes BOTH trade_memory and a journal sell row
# ---------------------------------------------------------------------------

def test_journal_external_close_writes_trade_memory_and_decision(monkeypatch):
    recorded, decided = [], []
    fake_tj = types.ModuleType('trade_journal')
    fake_tj.log_decision = lambda rec: decided.append(rec)
    monkeypatch.setitem(__import__('sys').modules, 'trade_journal', fake_tj)
    fn = _load_method(
        STOCK_SRC, 'StockLoop', '_journal_external_close',
        {'record_trade': lambda *a, **k: recorded.append((a, k))})
    me = SimpleNamespace(get_quote=lambda s: {'midpoint': 110.0})
    fn(me, 'AAPL', SimpleNamespace(entry_price=100.0))
    (args, kwargs), = recorded
    assert args == ('AAPL', 'sell', 100.0, 110.0, pytest.approx(10.0))
    assert kwargs == {'exit_reason': 'external_close', 'estimated': True}
    row, = decided
    assert row['action'] == 'sell'
    assert row['exit_reason'] == 'external_close'
    assert row['estimated'] is True
    assert row['pnl_pct'] == pytest.approx(10.0)
    assert row['fill_price'] == 110.0


def test_execute_sells_routes_external_close_through_helper():
    seg = _extract_method(STOCK_SRC, 'StockLoop', '_execute_sells')
    assert '_journal_external_close' in seg
    # the not-found detection is still string-matched before dropping
    for needle in ("'not found'", "'404'", "'no position'"):
        assert needle in seg


# ---------------------------------------------------------------------------
# stock_loop — cosmetic rot: unused import, flatten + ROD docstrings
# ---------------------------------------------------------------------------

def test_stock_cosmetic_rot_fixed():
    assert 'get_correlation_sizing_factor' not in STOCK_SRC
    doc = ast.get_docstring(ast.parse(STOCK_SRC))
    assert 'Flatten all stock positions at 3:50 PM ET' not in doc
    assert 'ACTUAL' in doc and 'overnight sleeve' in doc
    rod = _extract_method(STOCK_SRC, 'StockLoop', '_spy_rod_pm_tilt')
    assert '14:00-16:00' in rod
    assert '14:30-15:30' not in rod
