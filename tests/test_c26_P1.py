"""c26 campaign packet P1 — live-engine safety + journal integrity (B08+B09).

Covers (Mac-runnable, stubs/fakes only, no network, no heavy deps):
  order_utils   D19 confirm-only lifecycle, D18 maker rung abort/deterministic
                ids, D01 session-timeout installer, D19 emergency_flatten.
  trading_utils D01 REST-timeout wiring (source pins — dotenv not on the Mac).
  base_loop     D01 fan-out timeout + pool rebuild, D14 LLM backoff/expiry/
                preload, D15 peak-equity seed, D17 per-book flatten flags,
                D26 macro-emergency sizing guard, D33 journal row, B08 cycle
                order, B19 avg_corr journaling.
  stock_loop    D16 acquired-qty entry confirmation (source pins), D06/B09
                confirmed external-exit recovery, B19 avg_corr.
  llm_analyst   D33 get_last_analysis_meta passthrough.

base_loop/stock_loop cannot be imported on the dev Mac (torch via
predict_now), so functional coverage uses the extract-and-exec pattern from
tests/test_review_b01.py; order_utils and llm_analyst import directly.
"""
import ast
import concurrent.futures
import os
import sys
import textwrap
import time as _time
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import order_utils as ou
import llm_analyst

BASE_SRC = (REPO / "base_loop.py").read_text()
STOCK_SRC = (REPO / "stock_loop.py").read_text()
TRADING_SRC = (REPO / "trading_utils.py").read_text()


# ---------------------------------------------------------------------------
# Extraction helpers (copied pattern: tests/test_review_b01.py)
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


def _order(**kw):
    defaults = dict(id='o1', symbol='BTC/USD', qty='10', side='buy',
                    status='new', filled_qty='0', filled_avg_price=None)
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class _FakeApi:
    """Order-lifecycle fake: get_order serves from a script (or raises)."""

    def __init__(self, orders=None, raise_get=False):
        self._orders = list(orders or [])
        self._raise_get = raise_get
        self.canceled = []
        self.submitted = []

    def get_order(self, order_id):
        if self._raise_get:
            raise RuntimeError('api down')
        if len(self._orders) > 1:
            return self._orders.pop(0)
        return self._orders[0]

    def cancel_order(self, order_id):
        self.canceled.append(order_id)

    def submit_order(self, **kw):
        self.submitted.append(kw)
        return _order(id=f'sub{len(self.submitted)}', **{
            k: v for k, v in kw.items() if k in ('symbol', 'qty', 'side')})


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(ou.time, 'sleep', lambda s: None)


# ---------------------------------------------------------------------------
# order_utils — D19 confirm-only lifecycle
# ---------------------------------------------------------------------------

def test_confirm_only_never_cancels_returns_fetched_state(fast):
    stuck = _order(status='partially_filled', filled_qty='5')
    api = _FakeApi(orders=[stuck])
    result = ou.manage_order_lifecycle(api, 'o1', timeout=1, poll_interval=1,
                                       fallback_to_market=True,
                                       cancel_on_timeout=False)
    assert api.canceled == []
    assert api.submitted == []          # no market fallback either
    assert result is stuck


def test_confirm_only_three_error_giveup_no_cancel(fast, caplog):
    api = _FakeApi(raise_get=True)
    with caplog.at_level('ERROR', logger='order_utils'):
        result = ou.manage_order_lifecycle(api, 'o1', timeout=10,
                                           poll_interval=1,
                                           fallback_to_market=False,
                                           cancel_on_timeout=False)
    assert result is None
    assert api.canceled == []
    # Log contract: an emergency post-mortem must not claim a cancel
    # that confirm-only mode never issues.
    assert 'confirm-only mode' in caplog.text
    assert 'canceling order' not in caplog.text


def test_default_still_cancels_at_timeout(fast):
    # Byte-compat pin: the default path cancels exactly as before.
    api = _FakeApi(orders=[_order(status='new')])
    result = ou.manage_order_lifecycle(api, 'o1', timeout=1, poll_interval=1,
                                       fallback_to_market=False)
    assert api.canceled == ['o1']
    assert result is not None and result.status == 'new'


def test_default_three_error_giveup_still_cancels(fast):
    api = _FakeApi(raise_get=True)
    result = ou.manage_order_lifecycle(api, 'o1', timeout=10, poll_interval=1,
                                       fallback_to_market=False)
    assert result is None
    assert api.canceled == ['o1']


def test_emergency_flatten_phase2_confirm_only(monkeypatch, fast):
    captured = []

    def fake_lifecycle(api, order_id, timeout=30, **kw):
        captured.append((order_id, timeout, kw))
        return SimpleNamespace(status='filled', filled_qty='1')

    monkeypatch.setattr(ou, 'manage_order_lifecycle', fake_lifecycle)
    monkeypatch.setattr(ou, 'cancel_orders_for_symbol',
                        lambda api, sym, timeout=5: True)
    pos = SimpleNamespace(symbol='BTCUSD', qty='1')
    api = _FakeApi()
    api.list_positions = lambda: [pos]
    failures = ou.emergency_flatten(api, symbols=['BTC/USD'])
    assert failures == []
    (order_id, timeout, kw), = captured
    assert timeout == 15
    assert kw.get('fallback_to_market') is False
    assert kw.get('cancel_on_timeout') is False


# ---------------------------------------------------------------------------
# order_utils — D18 maker ladder: unknown outcome aborts, deterministic ids
# ---------------------------------------------------------------------------

def test_maker_rung_none_aborts_ladder_no_fallback(monkeypatch, fast):
    monkeypatch.setattr(ou, 'manage_order_lifecycle',
                        lambda *a, **k: None)
    api = _FakeApi()
    result, tactic = ou.place_maker_buy(api, 'BTC/USD', 100.0,
                                        lambda: {'bid': 10.0})
    assert tactic == 'maker_unknown'
    assert result is None                       # no evidence yet
    assert len(api.submitted) == 1              # exactly one rung, no fallback


def test_maker_rung2_none_returns_rung1_evidence(monkeypatch, fast):
    rung1 = _order(status='canceled', filled_qty='2', filled_avg_price='10')
    results = iter([rung1, None])
    monkeypatch.setattr(ou, 'manage_order_lifecycle',
                        lambda *a, **k: next(results))
    api = _FakeApi()
    result, tactic = ou.place_maker_buy(api, 'BTC/USD', 100.0,
                                        lambda: {'bid': 10.0},
                                        max_reprices=1)
    assert tactic == 'maker_unknown'
    assert result is rung1                      # best evidence kept
    assert len(api.submitted) == 2              # two rungs, no taker fallback


def test_maker_rung_id_deterministic_and_used(monkeypatch, fast):
    a = ou._maker_rung_id('BTC/USD', 123, 0)
    assert a == ou._maker_rung_id('BTC/USD', 123, 0)
    assert a != ou._maker_rung_id('BTC/USD', 123, 1)
    assert a != ou._maker_rung_id('BTC/USD', 124, 0)
    assert a != ou._maker_rung_id('ETH/USD', 123, 0)
    assert a.startswith('maker-') and len(a) <= 48

    monkeypatch.setattr(ou, 'manage_order_lifecycle', lambda *a, **k: None)
    api = _FakeApi()
    t0 = int(_time.time())
    ou.place_maker_buy(api, 'BTC/USD', 100.0, lambda: {'bid': 10.0})
    t1 = int(_time.time())
    coid = api.submitted[0]['client_order_id']
    assert coid in {ou._maker_rung_id('BTC/USD', t, 0)
                    for t in range(t0, t1 + 1)}


def test_make_client_order_id_still_non_idempotent():
    assert ou.make_client_order_id('maker') != ou.make_client_order_id('maker')


# ---------------------------------------------------------------------------
# order_utils — D01 install_session_timeout
# ---------------------------------------------------------------------------

def test_install_session_timeout_injects_and_preserves():
    calls = []

    class S:
        def request(self, *a, **kw):
            calls.append(kw)
            return 'ok'

    s = S()
    assert ou.install_session_timeout(s) is True
    s.request('GET', 'http://x')
    assert calls[-1]['timeout'] == (ou.REST_CONNECT_TIMEOUT_S,
                                    ou.REST_READ_TIMEOUT_S)
    s.request('GET', 'http://x', timeout=5)
    assert calls[-1]['timeout'] == 5            # explicit timeout preserved
    # Idempotent: second install refuses (no double wrap)
    assert ou.install_session_timeout(s) is False
    # No callable .request -> False, never raises
    assert ou.install_session_timeout(SimpleNamespace()) is False
    assert ou.install_session_timeout(SimpleNamespace(request=3)) is False


# ---------------------------------------------------------------------------
# trading_utils — D01 wiring (source pins: dotenv missing on the Mac)
# ---------------------------------------------------------------------------

def test_get_api_installs_timeouts_on_both_sdk_paths():
    start = TRADING_SRC.index('def get_api')
    body = TRADING_SRC[start:TRADING_SRC.index('\ndef ', start + 10)]
    assert body.count('_install_rest_timeouts(api)') == 2
    assert body.index('tradeapi.REST') < body.index('_install_rest_timeouts')
    assert 'CompatREST(key, secret, base_url)' in body


def test_install_rest_timeouts_probes_all_session_shapes():
    start = TRADING_SRC.index('def _install_rest_timeouts')
    body = TRADING_SRC[start:TRADING_SRC.index('\ndef ', start + 10)]
    assert "getattr(api, '_session', None)" in body
    for inner in ('_trading', '_stock_data', '_crypto_data'):
        assert f"'{inner}'" in body
    assert 'install_session_timeout' in body
    assert 'except Exception' in body           # fail-open


# ---------------------------------------------------------------------------
# base_loop — B08/D17 cycle order (source pins)
# ---------------------------------------------------------------------------

def _base_method(name: str) -> str:
    start = BASE_SRC.index(f"def {name}")
    return BASE_SRC[start:BASE_SRC.index("\n    def ", start + 10)]


def test_cycle_order_flatten_first_stops_before_housekeeping():
    body = _base_method('_run_one_cycle')
    assert body.index('_check_flatten_request') < body.index('check_market_hours')
    assert body.index('self._manage_stops()') < body.index('_hot_reload_check')
    assert body.index('self._manage_stops()') < body.index('_update_correlations')
    assert body.count('self._manage_stops()') == 1
    # market-hours pin unchanged: CYCLE banner after the gate
    assert body.index('check_market_hours') < body.index('logger.info("--- CYCLE')


def test_fanout_uses_bounded_as_completed():
    body = _base_method('_get_predictions')
    assert 'timeout=self.PREDICTION_TIMEOUT_SEC' in body
    assert '_rebuild_prediction_pool' in body
    assert 'pred_fanout_timeout' in body


# ---------------------------------------------------------------------------
# base_loop — D01 fan-out timeout (functional)
# ---------------------------------------------------------------------------

def test_fanout_timeout_partial_harvest_and_pool_rebuild():
    resolved = concurrent.futures.Future()
    resolved.set_result(('AAA', 0.5, {'x': 1}))
    stuck = concurrent.futures.Future()   # never resolves
    queue = iter([resolved, stuck])

    class Pool:
        def submit(self, fn, *a, **k):
            return next(queue)

    journal, rebuilt = [], []
    log = _Log()
    fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_get_predictions', {
        'choose_inference_device': lambda: 'cpu',
        'predict_symbol': lambda *a, **k: None,
        'as_completed': concurrent.futures.as_completed,
        'FuturesTimeoutError': concurrent.futures.TimeoutError,
        'TimeoutError': TimeoutError,
        'logger': log,
        'log_decision': lambda row: journal.append(row),
    })
    me = SimpleNamespace(
        model=object(), config={}, scaler_X=None, feature_cols=None,
        api=None, MODEL_PREFIX='', PREDICTION_TIMEOUT_SEC=1,
        _last_meta_p={}, _last_meta_p_cycle={},
        _prediction_pool=Pool(),
        get_symbol_universe=lambda: ['AAA', 'BBB'],
        get_asset_type=lambda: 'crypto',
        _rebuild_prediction_pool=lambda: rebuilt.append(True),
    )
    preds, snapshots = fn(me, None)
    assert preds == {'AAA': 0.5}
    assert snapshots == {'AAA': {'x': 1}}
    assert rebuilt == [True]
    rows = [r for r in journal if r.get('action') == 'pred_fanout_timeout']
    assert rows and rows[0]['wedged'] == ['BBB']
    assert stuck.cancelled()


def test_rebuild_prediction_pool_swaps_executor():
    old_calls = []
    fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_rebuild_prediction_pool',
                      {'ThreadPoolExecutor': concurrent.futures.ThreadPoolExecutor})
    old = SimpleNamespace(
        shutdown=lambda wait, cancel_futures: old_calls.append(
            (wait, cancel_futures)))
    me = SimpleNamespace(_prediction_pool=old, MAX_PREDICTION_WORKERS=1,
                         get_asset_type=lambda: 'crypto')
    fn(me)
    assert old_calls == [(False, True)]
    assert me._prediction_pool is not old
    me._prediction_pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# base_loop — D14 backoff / expiry / preload (functional)
# ---------------------------------------------------------------------------

def _llm_self(**over):
    me = SimpleNamespace(
        _expire_llm_scores=lambda: None,
        _llm_backoff_until=0.0, _llm_fail_count=0, _llm_scores_ts=None,
        _last_llm_time=0.0, LLM_INTERVAL_SEC=600, LLM_SCORE_TTL_SEC=7200,
        _check_llm_staleness=lambda: None,
        _build_llm_candidates=lambda preds: [
            {'symbol': 'BTC/USD', 'pred_return': 0.5}],
        positions={}, _equity=100000.0, config={},
        get_asset_type=lambda: 'crypto',
        llm_scores={}, _veto_strikes={}, _last_llm_symbols=set(),
    )
    for k, v in over.items():
        setattr(me, k, v)
    return me


def _run_llm(analyze, journal, calls=None):
    def _analyze(*a, **k):
        if calls is not None:
            calls.append(1)
        return analyze(*a, **k)
    return _load_method(BASE_SRC, 'BaseTradingLoop', '_run_llm_analysis', {
        'load_llm_config': lambda: {'enabled': True},
        'analyze_trades': _analyze,
        'log_decision': lambda row: journal.append(row),
        'logger': _Log(),
        'LLM_VETO_THRESHOLD': 0.15,
        'time': _time,
    })


def test_llm_outage_backoff_and_attempt_stamp():
    journal, calls = [], []
    fn = _run_llm(lambda *a, **k: {}, journal, calls)
    me = _llm_self()
    t0 = _time.time()
    fn(me, {})
    assert calls == [1]
    assert me._last_llm_time >= t0              # stamped on ATTEMPT
    assert me._llm_fail_count == 1
    assert me._llm_backoff_until == pytest.approx(me._last_llm_time + 600, abs=5)
    rows = [r for r in journal if r.get('action') == 'llm_backoff']
    assert rows and rows[0]['consecutive_failures'] == 1
    assert rows[0]['backoff_s'] == 600.0
    # Inside the backoff window: analyze_trades is never invoked
    fn(me, {})
    assert calls == [1]
    # Second failure doubles the backoff
    me._llm_backoff_until = 0.0
    me._last_llm_time = _time.time() - 601
    fn(me, {})
    assert calls == [1, 1]
    assert me._llm_fail_count == 2
    assert me._llm_backoff_until == pytest.approx(_time.time() + 1200, abs=5)


def test_llm_success_resets_backoff_and_journals_null_s(monkeypatch):
    journal = []
    fn = _run_llm(lambda *a, **k: {'BTC/USD': {'m': 0.9, 'r': ''}}, journal)
    monkeypatch.setattr(llm_analyst, 'get_last_analysis_meta', lambda: {
        'model': 'test-model', 'prompt_sha256': 'ab' * 32,
        'dedup_hit': False, 'latency_ms': 42})
    me = _llm_self(_llm_fail_count=3, _llm_backoff_until=0.0,
                   _veto_strikes={'BTC/USD': 1})
    fn(me, {})
    assert me._llm_fail_count == 0
    assert me._llm_backoff_until == 0.0
    assert me._llm_scores_ts is not None
    row, = [r for r in journal if r.get('action') == 'llm_analysis']
    assert row['scores']['BTC/USD']['s'] is None      # NOT a fabricated 0.5
    assert row['model'] == 'test-model'
    assert row['prompt_sha256'] == 'ab' * 32
    assert row['dedup_hit'] is False
    assert row['latency_ms'] == 42
    assert me._veto_strikes == {}                     # 0.5 default clears strike


def test_expire_llm_scores_ttl():
    fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_expire_llm_scores',
                      {'logger': _Log(), 'time': _time})
    me = SimpleNamespace(llm_scores={'X': {'s': 0.1}},
                         _llm_scores_ts=_time.time() - 7300,
                         LLM_SCORE_TTL_SEC=7200,
                         _veto_strikes={'X': 1})
    fn(me)
    assert me.llm_scores == {} and me._veto_strikes == {}
    # Fresh -> untouched
    me = SimpleNamespace(llm_scores={'X': {'s': 0.1}},
                         _llm_scores_ts=_time.time() - 100,
                         LLM_SCORE_TTL_SEC=7200, _veto_strikes={'X': 1})
    fn(me)
    assert me.llm_scores == {'X': {'s': 0.1}} and me._veto_strikes == {'X': 1}
    # Untimestamped (unit-test stubs) -> untouched
    me = SimpleNamespace(llm_scores={'X': {'s': 0.1}}, _llm_scores_ts=None,
                         LLM_SCORE_TTL_SEC=7200, _veto_strikes={'X': 1})
    fn(me)
    assert me.llm_scores == {'X': {'s': 0.1}}


def test_startup_preload_gated_on_enabled_and_age(monkeypatch):
    def make(enabled, ts):
        fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_print_startup', {
            'load_llm_config': lambda: {'enabled': enabled},
            'logger': _Log(),
        })
        monkeypatch.setattr(llm_analyst, 'load_analysis', lambda: {
            'crypto': {'BTC/USD': {'s': 0.7, 'timestamp': ts}}})
        me = SimpleNamespace(
            get_symbol_universe=lambda: [], get_asset_type=lambda: 'crypto',
            NOTIONAL_PER_SYMBOL=1000, LOOP_INTERVAL=30, COOLDOWN_MINUTES=60,
            LLM_SCORE_TTL_SEC=7200, llm_scores={}, _llm_scores_ts=None)
        fn(me)
        return me

    fresh_ts = datetime.now(timezone.utc).isoformat()
    stale_ts = datetime.fromtimestamp(
        _time.time() - 8000, tz=timezone.utc).isoformat()

    me = make(False, fresh_ts)                  # disabled -> no load
    assert me.llm_scores == {}
    me = make(True, stale_ts)                   # stale -> no load
    assert me.llm_scores == {} and me._llm_scores_ts is None
    me = make(True, fresh_ts)                   # fresh -> loaded + stamped
    assert 'BTC/USD' in me.llm_scores
    assert me._llm_scores_ts is not None


# ---------------------------------------------------------------------------
# base_loop — D15 peak-equity seed (functional)
# ---------------------------------------------------------------------------

def _update_equity_fn():
    return _load_method(BASE_SRC, 'BaseTradingLoop', '_update_equity',
                        {'logger': _Log()})


def test_first_real_equity_read_drops_seed_peak():
    fn = _update_equity_fn()
    me = SimpleNamespace(
        api=SimpleNamespace(get_account=lambda: SimpleNamespace(equity='40000')),
        cycle=1, _equity=100000.0, _peak_equity=100000.0,
        _peak_from_seed=True, _equity_cycle=None)
    fn(me)
    assert me._equity == 40000.0
    assert me._peak_equity == 40000.0           # not pinned to the $100k seed
    assert me._peak_from_seed is False


def test_default_100k_account_peak_unchanged():
    fn = _update_equity_fn()
    me = SimpleNamespace(
        api=SimpleNamespace(get_account=lambda: SimpleNamespace(equity='100000')),
        cycle=1, _equity=100000.0, _peak_equity=100000.0,
        _peak_from_seed=True, _equity_cycle=None)
    fn(me)
    assert me._peak_equity == 100000.0          # numerically identical to old


def test_equity_fetch_failure_keeps_seed_flag():
    fn = _update_equity_fn()

    def boom():
        raise RuntimeError('down')

    me = SimpleNamespace(api=SimpleNamespace(get_account=boom), cycle=1,
                         _equity=100000.0, _peak_equity=100000.0,
                         _peak_from_seed=True, _equity_cycle=None)
    fn(me)
    assert me._peak_from_seed is True           # still a placeholder


def _reconstruct_fn(update_equity):
    return _load_method(
        BASE_SRC, 'BaseTradingLoop', '_reconstruct_positions',
        {
            'reconstruct_positions': lambda api, syms: {},
            'logger': _Log(),
            'datetime': __import__('datetime'),
            'Position': SimpleNamespace,
        },
        replace={'from market_data import get_live_atr':
                 'get_live_atr = lambda *a, **k: None'},
    ), update_equity


def test_restore_honors_saved_peak_with_real_equity():
    def upd(me):
        me._equity = 40000.0
        me._peak_from_seed = False

    fn, _ = _reconstruct_fn(upd)
    me = SimpleNamespace(
        api=None, positions={}, last_trade_time={}, _daily_trades={},
        _daily_trades_date='x', _equity=100000.0, _peak_equity=100000.0,
        _peak_from_seed=True,
        get_symbol_universe=lambda: [],
        _load_position_state=lambda: {'peak_equity': 50000.0},
        _replace_protective_stops=lambda: None,
        get_asset_type=lambda: 'stock')
    me._update_equity = lambda: upd(me)
    fn(me)
    assert me._peak_equity == 50000.0
    assert me._peak_from_seed is False


def test_restore_honors_sub_seed_peak_when_fetch_failed():
    # Equity fetch failed -> _peak_from_seed stays True -> restore against 0
    fn, _ = _reconstruct_fn(None)
    me = SimpleNamespace(
        api=None, positions={}, last_trade_time={}, _daily_trades={},
        _daily_trades_date='x', _equity=100000.0, _peak_equity=100000.0,
        _peak_from_seed=True,
        get_symbol_universe=lambda: [],
        _load_position_state=lambda: {'peak_equity': 50000.0},
        _replace_protective_stops=lambda: None,
        get_asset_type=lambda: 'stock')
    me._update_equity = lambda: None            # fetch failed: no change
    fn(me)
    assert me._peak_equity == 50000.0           # honored, not inflated to 100k


def test_restore_default_paper_account_identical():
    fn, _ = _reconstruct_fn(None)
    me = SimpleNamespace(
        api=None, positions={}, last_trade_time={}, _daily_trades={},
        _daily_trades_date='x', _equity=100000.0, _peak_equity=100000.0,
        _peak_from_seed=True,
        get_symbol_universe=lambda: [],
        _load_position_state=lambda: {'peak_equity': 100000.0},
        _replace_protective_stops=lambda: None,
        get_asset_type=lambda: 'stock')

    def upd():
        me._equity = 100000.0
        me._peak_from_seed = False

    me._update_equity = upd
    fn(me)
    assert me._peak_equity == 100000.0          # pinned: same as before


def test_update_equity_is_first_statement_of_reconstruct():
    body = _base_method('_reconstruct_positions')
    first = [ln for ln in body.splitlines()[1:]
             if ln.strip() and not ln.strip().startswith(('"""', '#'))
             and 'Rebuild positions' not in ln][0]
    assert '_update_equity()' in first


# ---------------------------------------------------------------------------
# base_loop — D17 per-book flatten flags (functional)
# ---------------------------------------------------------------------------

def _flatten_setup(tmp_path, monkeypatch, requested):
    fake_notify = types.ModuleType('notify')
    fake_notify.calls = calls = {'clear': 0, 'halt': [], 'notes': []}
    fake_notify.flatten_requested = lambda: requested[0]
    fake_notify.clear_flatten_request = lambda: calls.__setitem__(
        'clear', calls['clear'] + 1)
    fake_notify.set_halt = lambda reason: calls['halt'].append(reason)
    fake_notify.notify = lambda *a, **k: calls['notes'].append(a)
    monkeypatch.setitem(sys.modules, 'notify', fake_notify)

    flattened = []

    def _path(book):
        return os.path.join(str(tmp_path), f'flatten_{book}.flag')

    fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_check_flatten_request', {
        '_flatten_flag_path': _path,
        'FLATTEN_FLAG_STALE_SEC': 3600,
        'os': os, 'time': _time, 'logger': _Log(),
        'emergency_flatten': lambda api, symbols=None: flattened.append(
            symbols) or [],
        'record_trade': lambda *a, **k: None,
    })
    me = SimpleNamespace(
        api=None, positions={},
        get_asset_type=lambda: 'crypto',
        get_symbol_universe=lambda: ['BTC/USD'],
        get_quote=lambda s: None,
        _save_position_state=lambda: None)
    return fn, me, _path, calls, flattened


def test_flatten_legacy_flag_fans_out_to_both_books(tmp_path, monkeypatch):
    requested = [True]
    fn, me, path, calls, flattened = _flatten_setup(tmp_path, monkeypatch,
                                                    requested)
    fn(me)
    assert calls['clear'] == 1                      # legacy flag cleared
    assert not os.path.exists(path('crypto'))       # own flag consumed
    assert os.path.exists(path('stock'))            # other book's flag intact
    assert flattened == [['BTC/USD']]               # this book flattened
    assert calls['halt'] == ['remote flatten']


def test_flatten_own_flag_only(tmp_path, monkeypatch):
    requested = [False]
    fn, me, path, calls, flattened = _flatten_setup(tmp_path, monkeypatch,
                                                    requested)
    with open(path('crypto'), 'w') as fh:
        fh.write(str(_time.time()))
    fn(me)
    assert flattened == [['BTC/USD']]
    assert not os.path.exists(path('crypto'))


def test_flatten_stale_flag_discarded(tmp_path, monkeypatch):
    requested = [False]
    fn, me, path, calls, flattened = _flatten_setup(tmp_path, monkeypatch,
                                                    requested)
    with open(path('crypto'), 'w') as fh:
        fh.write('x')
    old = _time.time() - 7200
    os.utime(path('crypto'), (old, old))
    fn(me)
    assert flattened == []                          # NOT actioned
    assert not os.path.exists(path('crypto'))       # but consumed
    assert calls['halt'] == []


def test_flatten_other_books_flag_ignored(tmp_path, monkeypatch):
    requested = [False]
    fn, me, path, calls, flattened = _flatten_setup(tmp_path, monkeypatch,
                                                    requested)
    with open(path('stock'), 'w') as fh:
        fh.write(str(_time.time()))
    fn(me)
    assert flattened == []
    assert os.path.exists(path('stock'))            # left for the stock book


# ---------------------------------------------------------------------------
# base_loop — D26 macro-emergency sizing guard
# ---------------------------------------------------------------------------

class _Sentinel:
    def __getattr__(self, name):
        raise AssertionError(f'api touched: {name}')


def test_macro_emergency_forces_zero_size():
    fn = _load_method(BASE_SRC, 'BaseTradingLoop', '_compute_position_size',
                      {'logger': _Log()})
    me = SimpleNamespace(api=_Sentinel(),
                         macro_regime=SimpleNamespace(sizing_mult=0.0),
                         get_asset_type=lambda: 'crypto')
    assert fn(me, 'BTC/USD', 0.5, {'midpoint': 100.0}) == 0


def test_macro_emergency_guard_precedes_normal_path():
    body = _base_method('_compute_position_size')
    assert body.index('sizing_mult == 0.0') < body.index(
        'from strategy_config import')
    # any sizing_mult > 0 falls through to the untouched normal path
    assert 'return 0\n' in body


# ---------------------------------------------------------------------------
# base_loop / stock_loop — B19 avg_corr journaling (source pins)
# ---------------------------------------------------------------------------

def test_avg_corr_journaled_in_both_books():
    for src, accept in ((BASE_SRC, "conv['avg_corr']"),
                        (STOCK_SRC, "buy_rec['avg_corr']")):
        assert 'avg_corr = None' in src
        i = src.index("'correlation'")
        seg = src[i:i + 400]
        assert 'avg_corr=round(avg_corr, 4)' in seg
        assert accept in src
        assert "= round(avg_corr, 4)" in src[src.index(accept):
                                             src.index(accept) + 200]


# ---------------------------------------------------------------------------
# stock_loop — D16 acquired-qty entry confirmation (source pins)
# ---------------------------------------------------------------------------

def test_stock_buy_judges_by_acquired_qty():
    body = _extract_method(STOCK_SRC, 'StockLoop', '_execute_buys')
    assert "if result and result.status == 'filled':" not in body
    assert "partial_qty = float(getattr(result, 'filled_qty', 0) or 0)" in body
    assert "== 'filled'\n                    or partial_qty > 0)" in STOCK_SRC
    # partial branch prefers broker truth
    i = body.index("!= 'filled':")
    seg = body[i:i + 900]
    assert 'verify_position' in seg
    assert 'filled_qty_int = int(float(vp.qty))' in seg
    assert 'Position(\n' in body and 'qty=filled_qty_int' in body
    assert 'filled_qty_int * fill_price' in body   # risk + exposure math
    assert 'tp_leg_id' in body and '_tp_order_ids[symbol]' in body


def test_stock_manage_stops_prunes_tp_ids():
    body = _extract_method(STOCK_SRC, 'StockLoop', '_manage_stops')
    assert "_tp_order_ids" in body
    assert 'if s not in self.positions' in body


# ---------------------------------------------------------------------------
# stock_loop — D06/B09 confirmed external-exit recovery (functional)
# ---------------------------------------------------------------------------

def _recover_fn():
    return _load_method(STOCK_SRC, 'StockLoop', '_recover_external_exit', {})


def test_recover_external_exit_via_tp_leg():
    fn = _recover_fn()
    tp_order = SimpleNamespace(status='filled', filled_avg_price='103.5',
                               side='sell')
    recorded = []
    me = SimpleNamespace(
        api=SimpleNamespace(get_order=lambda oid: tp_order),
        _tp_order_ids={'AAPL': 'tp1'},
        llm_scores={},
        _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)))
    info = SimpleNamespace(stop_order_id=None, entry_price=100.0)
    assert fn(me, 'AAPL', info) is True
    (args, kwargs), = recorded
    assert args[0] == 'AAPL' and args[2] is tp_order
    assert kwargs['exit_reason'] == 'take_profit'


def test_recover_external_exit_unparseable_price_falls_back():
    fn = _recover_fn()
    bad = SimpleNamespace(status='filled', filled_avg_price=None, side='sell')
    me = SimpleNamespace(
        api=SimpleNamespace(get_order=lambda oid: bad,
                            list_orders=lambda **k: []),
        _tp_order_ids={'AAPL': 'tp1'}, llm_scores={},
        _record_confirmed_exit=lambda *a, **k: pytest.fail('must not record'))
    assert fn(me, 'AAPL', SimpleNamespace(stop_order_id=None)) is False


def test_recover_external_exit_no_api_returns_false():
    fn = _recover_fn()
    me = SimpleNamespace(api=None)
    assert fn(me, 'AAPL', SimpleNamespace(stop_order_id=None)) is False


def test_recover_external_exit_closed_orders_fallback():
    # stale-stub modernization (F10): the fallback now requires a verifiable
    # fill time and reads self.last_trade_time — inject datetime and stamp
    # a fresh fill on the fake.
    import datetime as _dt2
    fn = _load_method(STOCK_SRC, 'StockLoop', '_recover_external_exit',
                      {'datetime': _dt2})
    sell = SimpleNamespace(status='filled', filled_avg_price='99.0',
                           side='sell',
                           filled_at=_dt2.datetime.now(
                               _dt2.timezone.utc).isoformat())
    recorded = []
    me = SimpleNamespace(
        api=SimpleNamespace(get_order=lambda oid: pytest.fail('no probes'),
                            list_orders=lambda **k: [sell]),
        _tp_order_ids={}, llm_scores={}, last_trade_time={},
        _record_confirmed_exit=lambda *a, **k: recorded.append((a, k)))
    info = SimpleNamespace(stop_order_id=None, entry_price=100.0)
    assert fn(me, 'AAPL', info) is True
    (args, kwargs), = recorded
    assert kwargs['exit_reason'] == 'external_close'


def test_journal_external_close_bare_stub_still_estimated(monkeypatch):
    # b01-compat: a stub without _recover_external_exit must fall through
    # to the estimated path unchanged (AttributeError swallowed).
    recorded, decided = [], []
    fake_tj = types.ModuleType('trade_journal')
    fake_tj.log_decision = lambda rec: decided.append(rec)
    monkeypatch.setitem(sys.modules, 'trade_journal', fake_tj)
    fn = _load_method(
        STOCK_SRC, 'StockLoop', '_journal_external_close',
        {'record_trade': lambda *a, **k: recorded.append((a, k))})
    me = SimpleNamespace(get_quote=lambda s: {'midpoint': 110.0})
    fn(me, 'AAPL', SimpleNamespace(entry_price=100.0))
    (args, kwargs), = recorded
    assert kwargs == {'exit_reason': 'external_close', 'estimated': True}
    row, = decided
    assert row['estimated'] is True


def test_journal_external_close_prefers_confirmed_recovery(monkeypatch):
    fake_tj = types.ModuleType('trade_journal')
    fake_tj.log_decision = lambda rec: pytest.fail('estimated row written')
    monkeypatch.setitem(sys.modules, 'trade_journal', fake_tj)
    fn = _load_method(
        STOCK_SRC, 'StockLoop', '_journal_external_close',
        {'record_trade': lambda *a, **k: pytest.fail('estimated record')})
    me = SimpleNamespace(get_quote=lambda s: {'midpoint': 110.0},
                         _recover_external_exit=lambda sym, info: True)
    fn(me, 'AAPL', SimpleNamespace(entry_price=100.0))   # no estimated row


# ---------------------------------------------------------------------------
# llm_analyst — D33 metadata passthrough (functional)
# ---------------------------------------------------------------------------

def test_analyze_trades_sets_last_analysis_meta(monkeypatch):
    monkeypatch.setattr(llm_analyst, 'load_llm_config',
                        lambda: {'enabled': True})
    monkeypatch.setattr(llm_analyst, 'get_recommended_model',
                        lambda role: 'model-x')
    monkeypatch.setattr(llm_analyst, 'get_last_model_used',
                        lambda: 'model-x')
    monkeypatch.setattr(llm_analyst, 'call_model', lambda *a, **k: 'x')
    monkeypatch.setattr(llm_analyst, '_parse_response',
                        lambda *a, **k: {'BTC/USD': {'s': 0.6, 'm': 0.9,
                                                     'r': ''}})
    monkeypatch.setattr(llm_analyst, '_save_analysis', lambda *a, **k: None)
    monkeypatch.setattr(llm_analyst, '_journal_replay', lambda *a, **k: None)
    monkeypatch.setattr(llm_analyst, '_LAST_CALL_META', {})
    result = llm_analyst.analyze_trades(
        [{'symbol': 'BTC/USD', 'pred_return': 0.5}], 'crypto')
    assert result == {'BTC/USD': {'s': 0.6, 'm': 0.9, 'r': ''}}
    meta = llm_analyst.get_last_analysis_meta()
    assert meta['dedup_hit'] is False
    assert isinstance(meta['prompt_sha256'], str) and len(
        meta['prompt_sha256']) == 64
    assert isinstance(meta['latency_ms'], int) and meta['latency_ms'] >= 0
    assert meta['model'] == 'model-x'


def test_get_last_analysis_meta_returns_copy():
    llm_analyst._LAST_CALL_META.clear()
    assert llm_analyst.get_last_analysis_meta() == {}
    llm_analyst._LAST_CALL_META.update({'model': 'm'})
    got = llm_analyst.get_last_analysis_meta()
    got['model'] = 'tampered'
    assert llm_analyst._LAST_CALL_META['model'] == 'm'
    llm_analyst._LAST_CALL_META.clear()
