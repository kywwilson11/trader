"""Loop-group (base/crypto/stock/crypto_trend) design+scout fixes, 2026-07."""
import sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SRC = (REPO / "base_loop.py").read_text()


# --- Source-level tests (Mac-green) ---

def test_stablecoin_flatten_normalizes_failure_symbols():
    start = SRC.index("[CONTAGION] Stablecoin emergency"); body = SRC[start:start + 1200]
    assert "'<list_positions failed>' in failures" in body
    assert "f.replace('/', '')" in body and "s.replace('/', '') in failed_norm" in body


def test_dead_imports_pruned():
    lines = [l.strip() for l in SRC.splitlines()]
    assert 'import gc' not in lines
    for name in ('get_all_positions', 'get_model_mtime', 'get_cached_sigma', 'get_garch_stop',
                 'get_returns_for_symbols', 'compute_correlation_matrix'):
        assert name not in SRC, name


def test_peak_equity_seeded_from_drawdown():
    from drawdown import PEAK_SEED
    assert PEAK_SEED == 100_000
    init = SRC[SRC.index('def __init__'):SRC.index('def get_symbol_universe')]
    assert 'PEAK_SEED' in init and '100_000' not in init


def test_entries_allowed_logs_swallowed_exceptions():
    body = SRC[SRC.index('def _entries_allowed'):SRC.index('def _execute_buys')]
    assert 'except Exception:\n            pass' not in body
    assert 'halt-flag check failed' in body and 'stand-down check failed' in body


def test_stock_bad_price_attributed():
    assert "vc['bad_price']" in (REPO / 'stock_loop.py').read_text()


def test_crypto_trend_docstring_marks_unwired():
    import crypto_trend
    assert 'NOT YET WIRED' in crypto_trend.__doc__


# --- Behavior tests (skip on Mac, run on Jetson/CI) ---

def test_stablecoin_flatten_keeps_failed_positions(monkeypatch):
    base_loop = pytest.importorskip('base_loop'); cl = pytest.importorskip('crypto_loop')
    from types import SimpleNamespace
    inst = object.__new__(cl.CryptoLoop); inst.api = None
    regime = SimpleNamespace(regime_label='bear', sizing_mult=0.0, stop_mult=1.0, stablecoin_alert=True)
    monkeypatch.setattr(base_loop, 'get_macro_regime', lambda api, at: regime)
    # broker-format failure keeps the matching universe-format position
    inst.positions = {'BTC/USD': 1, 'ETH/USD': 2}
    monkeypatch.setattr(base_loop, 'emergency_flatten', lambda api, symbols=None: ['BTCUSD'])
    inst._update_macro_regime(); assert set(inst.positions) == {'BTC/USD'}
    # sentinel keeps everything
    inst.positions = {'BTC/USD': 1, 'ETH/USD': 2}
    monkeypatch.setattr(base_loop, 'emergency_flatten', lambda api, symbols=None: ['<list_positions failed>'])
    inst._update_macro_regime(); assert set(inst.positions) == {'BTC/USD', 'ETH/USD'}
    # clean flatten clears
    inst.positions = {'BTC/USD': 1}
    monkeypatch.setattr(base_loop, 'emergency_flatten', lambda api, symbols=None: [])
    inst._update_macro_regime(); assert inst.positions == {}


def test_unverified_buy_logs_untracked(monkeypatch):
    base_loop = pytest.importorskip('base_loop'); cl = pytest.importorskip('crypto_loop')
    import order_utils, notify
    from types import SimpleNamespace
    inst = object.__new__(cl.CryptoLoop); inst.api = None
    inst.positions = {}; inst.last_trade_time = {}
    order = SimpleNamespace(status='filled', filled_qty='1')
    monkeypatch.setattr(cl.CryptoLoop, '_execute_entry_order', lambda self, s, n, q: (order, 'maker'))
    monkeypatch.setattr(order_utils, 'verify_position', lambda api, sym: None)
    sent = []; monkeypatch.setattr(notify, 'notify', lambda *a, **k: sent.append(a))
    msgs = []; monkeypatch.setattr(base_loop.logger, 'error', lambda m, *a, **k: msgs.append(m % tuple(a) if a else m))
    inst._place_and_track_buy('BTC/USD', 100.0, 0.5, {'midpoint': 50000.0}, 1.0, [], 0.6, 1.1, '')
    assert any('UNTRACKED' in m for m in msgs) and sent
    assert inst.positions == {} and inst.last_trade_time == {}


def test_entries_allowed_fail_open_logs(monkeypatch):
    base_loop = pytest.importorskip('base_loop'); cl = pytest.importorskip('crypto_loop')
    import notify, macro_calendar
    inst = object.__new__(cl.CryptoLoop); inst.cycle = 1
    def boom(*a, **k): raise RuntimeError('disk')
    monkeypatch.setattr(notify, 'halt_active', boom)
    monkeypatch.setattr(macro_calendar, 'macro_standdown', boom)
    msgs = []; monkeypatch.setattr(base_loop.logger, 'warning', lambda m, *a, **k: msgs.append(m % tuple(a) if a else m))
    assert inst._entries_allowed() is True
    assert any('halt-flag check failed' in m for m in msgs)
    assert any('stand-down check failed' in m for m in msgs)
