"""SIG-R1-A1: evaluate_on_holdout feeds calendar_effective_n epoch SECONDS.

sample_weights.calendar_effective_n's numeric contract is float HOURS
(datetime64 is converted to hours internally — backtest.py's path).
hypersearch_v2.evaluate_on_holdout passes all_times (int64 epoch seconds),
so concurrency is binned per SECOND: touching/partially-overlapping holds
barely register (wrong n_eff — the DSR input under PROMOTION_GATE_V2, and
the persisted 'n_eff_v2' instrumentation otherwise) and the diff/prefix
arrays scale with the holdout span in seconds (~0.9 GB on the 220-day
crypto holdout).

Runs the REAL evaluate_on_holdout on CPU with a tiny RegressionLSTM and a
synthetic 2-name hourly panel. Module under test: SIG_R1_HS (path to a
hypersearch_v2.py; default = <repo>/scripts/hypersearch_v2.py).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sklearn')
pytest.importorskip('optuna')

REPO = Path(os.environ.get('SIG_R1_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R1_HS',
                         str(REPO / 'scripts' / 'hypersearch_v2.py'))

import strategy_config  # noqa: E402
from objective_utils import simulate_trades_core  # noqa: E402
from sample_weights import calendar_effective_n  # noqa: E402

FB = 4
H = 1000          # hourly bars per name
T0 = 1_760_000_000


@pytest.fixture(scope='module')
def hs():
    spec = importlib.util.spec_from_file_location('hs_sig_r1_a1', HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _panel(tb_span):
    rng = np.random.default_rng(7)
    tickers = ['A', 'B']
    bounds = {'A': (0, H), 'B': (H, 2 * H)}
    times = np.concatenate([T0 + np.arange(H) * 3600,
                            T0 + (np.arange(H) + 1) * 3600]).astype(np.int64)
    feats = rng.normal(size=(2 * H, 3)).astype(np.float32)
    ret = rng.normal(0.1, 1.0, size=2 * H).astype(np.float32)
    ret[H - FB:H] = np.nan
    ret[2 * H - FB:] = np.nan
    returns = {FB: ret, ('tb', FB): ret.copy()}
    spans = {FB: np.full(2 * H, float(tb_span), dtype=np.float32)}
    return tickers, bounds, times, feats, returns, spans


def _run(hs, monkeypatch, gate_v2, target_kind='tb', tb_span=FB):
    from sklearn.preprocessing import RobustScaler
    monkeypatch.setattr(hs, '_objective_v3', lambda: False)
    monkeypatch.setattr(hs, '_fixed_holdout_days', lambda: None)
    monkeypatch.setattr(strategy_config, 'PROMOTION_GATE_V2', gate_v2)
    monkeypatch.setattr(strategy_config, 'KISH_NEFF_ENABLED', False)
    monkeypatch.delenv('TRADER_HOLDOUT_SPAN_BY_TARGET', raising=False)
    tickers, bounds, times, feats, returns, spans = _panel(tb_span)
    cfg = {'seq_len': 8, 'forward_bars': FB, 'trade_threshold': -1e9,
           'target_kind': target_kind, 'hidden_dim': 8, 'num_layers': 1,
           'dropout': 0.0, 'n_heads': 2}
    torch.manual_seed(0)
    state = hs.RegressionLSTM(3, 8, 1, 0.0, 2).state_dict()
    scaler = RobustScaler().fit(feats)
    rep = hs.evaluate_on_holdout(state, scaler, cfg, feats, returns, times,
                                 tickers, bounds, 3, 'crypto', n_trials=10,
                                 all_tb_bars_by_fb=spans)
    # Independent replication of the trades + [entry, exit] rows
    hidx = hs.get_holdout_indices(times, tickers, bounds, cfg['seq_len'])
    key = ('tb', FB) if target_kind == 'tb' else FB
    hidx = hidx[~np.isnan(returns[key][hidx])]
    _, ent = simulate_trades_core(np.ones(len(hidx)), returns[key][hidx],
                                  -1e9, FB, 0.6)
    rows = hidx[ent]
    last = np.where(rows < H, H - 1, 2 * H - 1)
    exits = np.minimum(rows + int(tb_span), last)
    return rep, times[rows], times[exits]


def test_gate_v2_calendar_n_eff_uses_hours(hs, monkeypatch):
    rep, e_s, x_s = _run(hs, monkeypatch, gate_v2=True)
    hours = calendar_effective_n(e_s / 3600.0, x_s / 3600.0)['n_eff']
    secs = calendar_effective_n(e_s, x_s)['n_eff']
    assert abs(hours - secs) > 1.0          # the input discriminates
    assert rep['n_eff_v2'] == pytest.approx(hours)
    assert rep['n_eff'] == pytest.approx(round(min(hours, rep['n_trades']), 2))


def test_legacy_side_by_side_n_eff_v2_uses_hours(hs, monkeypatch):
    rep, e_s, x_s = _run(hs, monkeypatch, gate_v2=False)
    hours = calendar_effective_n(e_s / 3600.0, x_s / 3600.0)['n_eff']
    assert rep['n_eff_v2'] == pytest.approx(hours)


def test_legacy_gate_numbers_untouched(hs, monkeypatch):
    """OFF-path pin: the legacy DSR input (uniqueness -> clustered) is
    unit-invariant and must not move."""
    from sample_weights import clustered_effective_n
    rep, e_s, x_s = _run(hs, monkeypatch, gate_v2=False)
    n_x = clustered_effective_n(e_s, x_s)
    assert n_x == clustered_effective_n(e_s / 3600.0, x_s / 3600.0)
    assert rep['status'] == 'ok' and rep['n_trades'] == len(e_s)
    assert rep['n_eff'] == pytest.approx(max(min(float(n_x), len(e_s)), 10.0))
