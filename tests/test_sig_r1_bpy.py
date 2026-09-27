"""SIG-R1-BPY — BARS_PER_YEAR_MEASURED (default OFF) via bars_calendar.

OFF (and before strategy_config grows the constant): every SIGNAL copy keeps
8760/1638 and compute_sharpe / portfolio_backtest defaults are byte-identical
to the pre-flag expressions. ON: stock -> 3827 (census, see bars_calendar),
crypto unchanged.
"""
import ast
import importlib
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
LEGACY = {'crypto': 8760, 'stock': 1638}


@pytest.fixture
def flag_absent(monkeypatch):
    import strategy_config
    monkeypatch.delattr(strategy_config, 'BARS_PER_YEAR_MEASURED',
                        raising=False)


@pytest.fixture
def flag_on(monkeypatch):
    import strategy_config
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True,
                        raising=False)


def _hs():
    pytest.importorskip('torch')
    pytest.importorskip('optuna')
    import hypersearch_v2
    return hypersearch_v2


def _sample():
    rng = np.random.default_rng(11)
    preds = rng.normal(0, 1, 5000)
    actuals = np.sign(preds) * 1.5 + rng.normal(0, 2.0, 5000)
    return preds, actuals


def _legacy_sharpe(hs, preds, actuals, thr, fb, asset):
    """compute_sharpe's pre-flag body, verbatim, with the literal table."""
    tr = hs.simulate_trades(preds, actuals, thr, fb,
                            hs.TXN_COST_PCT.get(asset, 0.6))
    std = tr.std()
    bars_per_year = LEGACY.get(asset, 8760)
    slots_per_year = bars_per_year / fb
    occupancy = min(len(tr) * fb / max(len(preds), 1), 1.0)
    trades_per_year = occupancy * slots_per_year
    return float((tr.mean() / std) * np.sqrt(max(trades_per_year, 1.0)))


# --- OFF: byte pins ---------------------------------------------------------

def test_literal_tables_unchanged():
    pat = re.compile(r"BARS_PER_YEAR\s*=\s*(\{[^}]*\})")
    for rel in ('backtest.py', 'scripts/hypersearch_v2.py',
                'portfolio_backtest.py', 'volatility.py'):
        m = pat.search((REPO / rel).read_text())
        assert m and ast.literal_eval(m.group(1)) == LEGACY, rel


@pytest.mark.parametrize('state', ['absent', 'false'])
def test_function_off(monkeypatch, state):
    import strategy_config
    if state == 'absent':
        monkeypatch.delattr(strategy_config, 'BARS_PER_YEAR_MEASURED',
                            raising=False)
    else:
        monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                            raising=False)
    from bars_calendar import bars_per_year, measured_enabled
    assert measured_enabled() is False
    assert bars_per_year('stock') == 1638 and bars_per_year('crypto') == 8760
    assert bars_per_year('fx') == 8760
    assert bars_per_year('fx', LEGACY, 123) == 123
    own = {'crypto': 8760.0, 'stock': 1638.0}
    v = bars_per_year('stock', own)
    assert v == 1638.0 and type(v) is float           # caller's table wins
    patched = {'stock': 999}
    assert bars_per_year('stock', patched) == 999      # monkeypatch-transparent


@pytest.mark.parametrize('asset', ['stock', 'crypto'])
def test_compute_sharpe_off_byte_identical(flag_absent, asset):
    hs = _hs()
    preds, actuals = _sample()
    for fb in (4, 24):
        got = hs.compute_sharpe(preds, actuals, 0.5, fb, asset)
        assert got == _legacy_sharpe(hs, preds, actuals, 0.5, fb, asset)


def test_portfolio_default_off(flag_absent):
    import portfolio_backtest as pb
    pb = importlib.reload(pb)
    assert pb.DEFAULT_PERIODS_PER_YEAR == 1638.0
    assert type(pb.DEFAULT_PERIODS_PER_YEAR) is float
    assert (inspect.signature(pb.run_policy).parameters['periods_per_year']
            .default == 1638.0)


# --- ON ---------------------------------------------------------------------

def test_function_on(flag_on):
    from bars_calendar import (bars_per_year, measured_enabled,
                               MEASURED_BARS_PER_YEAR)
    assert measured_enabled() is True
    assert MEASURED_BARS_PER_YEAR == {'crypto': 8760, 'stock': 3827}
    assert bars_per_year('stock') == 3827
    assert bars_per_year('stock', LEGACY) == 3827
    assert bars_per_year('crypto', LEGACY) == 8760
    assert bars_per_year('fx', LEGACY, 8760) == 8760


def test_compute_sharpe_on_scales_stock_only(monkeypatch):
    import strategy_config
    hs = _hs()
    preds, actuals = _sample()
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', False,
                        raising=False)
    off_s = hs.compute_sharpe(preds, actuals, 0.5, 24, 'stock')
    off_c = hs.compute_sharpe(preds, actuals, 0.5, 24, 'crypto')
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True)
    on_s = hs.compute_sharpe(preds, actuals, 0.5, 24, 'stock')
    on_c = hs.compute_sharpe(preds, actuals, 0.5, 24, 'crypto')
    assert off_s > 0
    assert on_s == pytest.approx(off_s * math.sqrt(3827 / 1638), rel=1e-12)
    assert on_c == off_c


def test_portfolio_default_on(monkeypatch):
    import strategy_config
    import portfolio_backtest as pb
    monkeypatch.setattr(strategy_config, 'BARS_PER_YEAR_MEASURED', True,
                        raising=False)
    try:
        pb = importlib.reload(pb)
        assert pb.DEFAULT_PERIODS_PER_YEAR == 3827.0
        assert pb.BARS_PER_YEAR == {'crypto': 8760.0, 'stock': 1638.0}
    finally:
        monkeypatch.undo()
        importlib.reload(pb)
    assert pb.DEFAULT_PERIODS_PER_YEAR == 1638.0
