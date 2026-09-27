"""SIG-R1-A3: raw-target holdout trades are deflated with the TRIPLE-
BARRIER span.

evaluate_on_holdout reconstructs every holdout trade's [entry, exit]
window from all_tb_bars_by_fb[fb] (TB_Bars — the policy-exit span of the
triple-barrier label) regardless of cfg['target_kind']. A target_kind='raw'
winner books Target_Return_fb — an fb-bar window — so its overlap (the
thing the n_eff deflation measures) is understated: on the live crypto
store TB_Bars averages 0.35 x fb at fb=48 (0.78 x at fb=12). n_eff is
overstated -> the DSR gate is looser than its own design.

Fix behind default-OFF HOLDOUT_SPAN_BY_TARGET / TRADER_HOLDOUT_SPAN_BY_TARGET:
raw target -> span = fb. OFF (and target_kind='tb'): TB_Bars, byte-identical.
Runs the REAL evaluate_on_holdout on CPU (tiny RegressionLSTM, synthetic
2-name panel). Module under test: SIG_R1_HS.
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
from sample_weights import (average_uniqueness, clustered_effective_n,  # noqa: E402
                            effective_n)

FB = 4
TB_SPAN = 1       # triple-barrier exits after 1 bar; the raw label spans FB
H = 1000
T0 = 1_760_000_000


@pytest.fixture(scope='module')
def hs():
    spec = importlib.util.spec_from_file_location('hs_sig_r1_a3', HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(hs, monkeypatch, flag, target_kind):
    from sklearn.preprocessing import RobustScaler
    monkeypatch.setattr(hs, '_objective_v3', lambda: False)
    monkeypatch.setattr(hs, '_fixed_holdout_days', lambda: None)
    monkeypatch.setattr(strategy_config, 'PROMOTION_GATE_V2', False)
    if flag:
        monkeypatch.setenv('TRADER_HOLDOUT_SPAN_BY_TARGET', '1')
    else:
        monkeypatch.delenv('TRADER_HOLDOUT_SPAN_BY_TARGET', raising=False)
        monkeypatch.setattr(strategy_config, 'HOLDOUT_SPAN_BY_TARGET',
                            False, raising=False)
    rng = np.random.default_rng(7)
    tickers = ['A', 'B']
    bounds = {'A': (0, H), 'B': (H, 2 * H)}
    times = np.concatenate([T0 + np.arange(H) * 3600,
                            T0 + np.arange(H) * 3600]).astype(np.int64)
    feats = rng.normal(size=(2 * H, 3)).astype(np.float32)
    ret = rng.normal(0.1, 1.0, size=2 * H).astype(np.float32)
    ret[H - FB:H] = np.nan
    ret[2 * H - FB:] = np.nan
    returns = {FB: ret, ('tb', FB): ret.copy()}
    spans = {FB: np.full(2 * H, float(TB_SPAN), dtype=np.float32)}
    cfg = {'seq_len': 8, 'forward_bars': FB, 'trade_threshold': -1e9,
           'target_kind': target_kind, 'hidden_dim': 8, 'num_layers': 1,
           'dropout': 0.0, 'n_heads': 2}
    torch.manual_seed(0)
    state = hs.RegressionLSTM(3, 8, 1, 0.0, 2).state_dict()
    rep = hs.evaluate_on_holdout(state, RobustScaler().fit(feats), cfg,
                                 feats, returns, times, tickers, bounds, 3,
                                 'crypto', n_trials=10,
                                 all_tb_bars_by_fb=spans)
    key = ('tb', FB) if target_kind == 'tb' else FB
    hidx = hs.get_holdout_indices(times, tickers, bounds, cfg['seq_len'])
    hidx = hidx[~np.isnan(returns[key][hidx])]
    _, ent = simulate_trades_core(np.ones(len(hidx)), returns[key][hidx],
                                  -1e9, FB, 0.6)
    return rep, hidx[ent], bounds, times


def _legacy_n_eff(rows, span, bounds, times):
    """Independent replication of the legacy uniqueness->clustered rule
    at a given per-trade span, then the DSR's [10, n] clamp."""
    masked = np.full(2 * H, np.nan)
    masked[rows] = span
    n_u = effective_n(average_uniqueness(masked, bounds)[rows])
    last = np.where(rows < H, H - 1, 2 * H - 1)
    n_x = clustered_effective_n(times[rows],
                                times[np.minimum(rows + int(span), last)])
    n_eff = float(n_x) if 0 < n_x < (n_u if n_u else len(rows)) else n_u
    return round(min(max(n_eff, 10.0), float(len(rows))), 2)


def test_raw_target_deflates_on_fb_span_when_flag_on(hs, monkeypatch):
    rep, rows, bounds, times = _run(hs, monkeypatch, True, 'raw')
    want, stale = (_legacy_n_eff(rows, FB, bounds, times),
                   _legacy_n_eff(rows, TB_SPAN, bounds, times))
    assert want != stale                       # the input discriminates
    assert rep['n_eff'] == pytest.approx(want)


def test_flag_off_keeps_tb_span(hs, monkeypatch):
    """OFF-path byte-pin (raw target, flag OFF): legacy TB_Bars span."""
    rep, rows, bounds, times = _run(hs, monkeypatch, False, 'raw')
    assert rep['n_eff'] == pytest.approx(
        _legacy_n_eff(rows, TB_SPAN, bounds, times))


def test_tb_target_unaffected_by_flag(hs, monkeypatch):
    rep, rows, bounds, times = _run(hs, monkeypatch, True, 'tb')
    assert rep['n_eff'] == pytest.approx(
        _legacy_n_eff(rows, TB_SPAN, bounds, times))
