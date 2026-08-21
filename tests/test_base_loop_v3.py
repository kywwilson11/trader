"""2026-07 panel adjudication fixes for base_loop.py (module-improve-v3).

base_loop.py cannot be imported on the dev Mac (torch/joblib via
predict_now), so coverage here is source-text pinning in the style of
tests/test_grp_loops.py; functional coverage runs on Jetson/CI behind
pytest.importorskip.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SRC = (REPO / "base_loop.py").read_text()


def _method(name: str) -> str:
    start = SRC.index(f"def {name}")
    return SRC[start:SRC.index("\n    def ", start + 10)]


# --- P0: remote /flatten keeps failed positions tracked ---

def test_flatten_request_keeps_failed_positions():
    body = _method("_check_flatten_request")
    assert "'<list_positions failed>' in failures" in body
    assert "f.replace('/', '')" in body
    assert "s.replace('/', '') in failed_norm" in body
    assert body.index("'<list_positions failed>'") < body.index("self.positions.clear()")


def test_flatten_request_journals_released_positions():
    body = _method("_check_flatten_request")
    assert "exit_reason='remote_flatten'" in body
    assert body.index("emergency_flatten(") < body.index("record_trade(")


def test_circuit_breaker_journals_after_flatten():
    body = _method("_circuit_breaker_check")
    assert body.index("emergency_flatten(") < body.index("record_trade(")
    assert "will retry next cycle" not in body
    assert "exit_reason='circuit_breaker'" in body
    assert "'<list_positions failed>' in failures" in body


def test_pending_breach_pruned_against_positions():
    body = _method("_manage_stops")
    assert "for s in list(self._pending_breach)" in body
    assert "if s not in self.positions" in body


def test_stop_exit_division_guards():
    body = _method("_execute_stop_exit")
    assert body.count("if entry_price > 0 else 0.0") >= 2
    i = body.index("float(result.filled_avg_price)")
    assert "except (TypeError, ValueError)" in body[i:i + 400]


def test_stop_exit_writes_decision_journal_rows():
    body = _method("_execute_stop_exit")
    assert body.count('"action": "sell"') >= 2
    assert '"exit_reason": stop_reason' in body
    assert '"desync"' in body


def test_meta_gate_logs_fail_open():
    body = _method("_meta_gate")
    assert "except Exception as e:" in body
    assert "failing open" in body
    assert "_journal_skip(symbol, 'meta_veto'" in body
    # Both the p-None path AND the fail-open except path clear the stash
    # and its freshness stamp (a dead gate must not leave a stale stamp).
    assert body.count(
        "getattr(self, '_last_meta_p_cycle', {}).pop(symbol, None)") >= 2


def test_conv_fields_freshness_flag_and_hardening():
    body = _method("_conv_fields")
    assert "_last_meta_p_cycle" in body
    assert "_conviction_journal_on" in body
    assert "except Exception" in body


def test_last_meta_p_pruned_in_get_predictions():
    body = _method("_get_predictions")
    assert "_last_meta_p" in body and "not in uni" in body


def test_no_bare_skip_rows_in_execute_buys():
    body = _method("_execute_buys")
    assert 'log_decision({"symbol": symbol, "action": "skip"' not in body
    for reason in ("'sentiment_block'", "'llm_veto'", "'q10_tail_veto'"):
        assert f"_journal_skip(symbol, {reason}" in body


def test_dead_hard_stop_threshold_removed():
    body = _method("_execute_buys")
    assert "symbol in self.hard_stop_lockout" not in body
    assert "vc['hard_stop_lockout']" in body


def test_crypto_rank_annotation():
    body = _method("_execute_buys")
    assert "rank_map" in body
    assert "rank=rank_map.get(symbol)" in body


def test_quote_age_instrumented():
    assert "_fetched_ts" in _method("_execute_buys")
    assert "quote_age_s" in _method("_place_and_track_buy")


def test_heartbeat_pinged_when_market_closed():
    body = _method("_run_one_cycle")
    closed = body[body.index("check_market_hours"):body.index('logger.info("--- CYCLE')]
    assert "ping_heartbeat" in closed


def test_lockout_tmp_path_is_per_book():
    body = _method("_save_hard_stop_lockout")
    assert "self._lockout_file + '.tmp'" not in body
    assert "MODEL_PREFIX" in body


def test_lockout_load_corruption_logged():
    body = _method("_load_hard_stop_lockout")
    assert "json.JSONDecodeError" in body
    assert "corrupt" in body


def test_hot_reload_failure_backoff():
    assert "_failed_reload" in _method("_hot_reload_check")


def test_update_equity_logs_failure():
    body = _method("_update_equity")
    assert "except Exception:\n            pass" not in body
    assert "logger.warning" in body


def test_macro_regime_failure_warns_with_age():
    body = _method("_update_macro_regime")
    assert 'logger.debug("[MACRO] Regime update failed' not in body
    assert "_macro_regime_ts" in body


def test_llm_empty_result_logged():
    assert "analyze_trades returned no scores" in _method("_run_llm_analysis")


def test_llm_veto_sell_noop_logged():
    assert "did not execute" in _method("_execute_llm_veto_sells")


def test_position_state_dedup_and_prune():
    body = _method("_save_position_state")
    assert "_last_state_blob" in body
    assert "COOLDOWN_MINUTES" in body
    assert "'peak_equity': self._peak_equity" in body


def test_module_docstring_no_false_sync_claim():
    head = SRC[:SRC.index("import json")]
    assert "stay in sync as new features are added" not in head


def test_get_quote_contract_documented():
    body = _method("get_quote")
    assert "midpoint" in body and "spread_pct" in body


# --- Behavior tests (skip on Mac, run on Jetson/CI) ---

def test_flatten_request_functional_keeps_failed(monkeypatch, tmp_path):
    base_loop = pytest.importorskip('base_loop')
    cl = pytest.importorskip('crypto_loop')
    import notify
    inst = object.__new__(cl.CryptoLoop)
    inst.api = None
    P = type('P', (), {'entry_price': 100.0})
    # c26 D17 isolation: the fan-out must not drop real flatten_*.flag
    # files into the repo during a Jetson test run.
    monkeypatch.setattr(base_loop, '_FLATTEN_FLAG_DIR', str(tmp_path))
    monkeypatch.setattr(notify, 'flatten_requested', lambda: True)
    monkeypatch.setattr(notify, 'clear_flatten_request', lambda: None)
    monkeypatch.setattr(notify, 'set_halt', lambda *a, **k: None)
    monkeypatch.setattr(notify, 'notify', lambda *a, **k: None)
    monkeypatch.setattr(base_loop, 'record_trade', lambda *a, **k: None)
    monkeypatch.setattr(cl.CryptoLoop, '_save_position_state', lambda self: None)
    monkeypatch.setattr(cl.CryptoLoop, 'get_quote', lambda self, s: None)
    # broker-format failure keeps the matching universe-format position
    inst.positions = {'BTC/USD': P(), 'ETH/USD': P()}
    monkeypatch.setattr(base_loop, 'emergency_flatten',
                        lambda api, symbols=None: ['BTCUSD'])
    inst._check_flatten_request()
    assert set(inst.positions) == {'BTC/USD'}
    # sentinel keeps everything
    inst.positions = {'BTC/USD': P(), 'ETH/USD': P()}
    monkeypatch.setattr(base_loop, 'emergency_flatten',
                        lambda api, symbols=None: ['<list_positions failed>'])
    inst._check_flatten_request()
    assert set(inst.positions) == {'BTC/USD', 'ETH/USD'}
    # clean flatten clears
    inst.positions = {'BTC/USD': P()}
    monkeypatch.setattr(base_loop, 'emergency_flatten',
                        lambda api, symbols=None: [])
    inst._check_flatten_request()
    assert inst.positions == {}


def test_conv_fields_freshness_functional():
    base_loop = pytest.importorskip('base_loop')

    class _Stub(base_loop.BaseTradingLoop):
        MODEL_PREFIX = ''

        def __init__(self):
            self.trade_threshold = 0.2
            self._last_meta_p = {'NVDA': 0.7}
            self._last_meta_p_cycle = {'NVDA': 5}
            self.cycle = 6

        def get_asset_type(self): return 'stock'
        def check_market_hours(self): return True
        def flatten_before_close(self): pass
        def get_benchmark_close(self): return None
        def get_headlines(self, s): return []
        def get_quote(self, s): return None
        def get_symbol_universe(self): return []
        def place_buy_order(self, *a, **k): return None
        def place_sell_order(self, *a, **k): return None
        def write_prediction_cache(self, *a, **k): pass

    loop = _Stub()
    f = loop._conv_fields('NVDA', 0.4, {})
    assert 'meta_p' not in f and 'conviction_tier' not in f   # stale cycle
    loop.cycle = 5
    f = loop._conv_fields('NVDA', 0.4, {})
    assert f['meta_p'] == 0.7 and f['conviction_tier'] == 'A'
