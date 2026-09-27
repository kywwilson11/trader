"""SIG-R3-DECOMP: per-fold trade-decomposition user_attrs (instrumentation).

The objective recorded only cfg / regime_sharpes / fold_sharpes / avg_sharpe
/ std_sharpe, so "why is this trial negative?" (no edge vs cost vs scoring
artefact vs threshold mismatch) could not be answered from the study DB
(scripts/trial_score_decomposition.py layer 3 needs these). The patch records,
per fold (lists aligned with fold_sharpes): gross_ret_mean, gross_ret_std,
net_ret_mean, cost_drag, n_trades, hit_rate, mean_hold_bars,
threshold_pass_rate, n_rows — plus regime_trade_decomp for the regime-penalty
slice. Pure instrumentation: the trial value and every pre-existing user_attr
are BIT-IDENTICAL to the pre-patch module (OBJECTIVE_LONG_ONLY off and on).

Module under test: SIG_R3_HS (default: the repo's scripts/hypersearch_v2.py).
Pre-patch reference for the bit-identity pin: SIG_R3_LIVE_HS (default = HS).
"""
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
optuna = pytest.importorskip('optuna')
pytest.importorskip('sklearn')

REPO = Path(os.environ.get('SIG_R3_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R3_HS', str(REPO / 'scripts' / 'hypersearch_v2.py'))
LIVE_HS_PATH = os.environ.get('SIG_R3_LIVE_HS', HS_PATH)

KEYS = ('gross_ret_mean', 'gross_ret_std', 'net_ret_mean', 'cost_drag',
        'n_trades', 'hit_rate', 'mean_hold_bars', 'threshold_pass_rate',
        'n_rows')
LEGACY_ATTRS = ('cfg', 'regime_sharpes', 'fold_sharpes', 'avg_sharpe',
                'std_sharpe')
FB = 4
N = 1200
PARAMS = {'forward_bars': FB, 'seq_len': 8, 'hidden_dim': 64,
          'num_layers': 1, 'n_heads': 2, 'dropout': 0.10,
          'learning_rate': 1e-3, 'batch_size': 512, 'weight_decay': 1e-4,
          'huber_delta': 1.0, 'trade_threshold': 0.05, 'scheduler': 'cosine'}
FOLDS = [(np.arange(8, 520), np.arange(540, 720)),
         (np.arange(8, 800), np.arange(820, 1000))]

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(1)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def hs():
    return _load(HS_PATH, 'hs_sig_r3_decomp')


@pytest.fixture(scope='module')
def hs_live():
    return _load(LIVE_HS_PATH, 'hs_sig_r3_decomp_live')


def _data():
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(N, 3)).astype(np.float32)
    ret = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    times = (1_760_000_000 + np.arange(N) * 3600).astype(np.int64)
    label_times = times[np.minimum(np.arange(N) + FB, N - 1)]
    return feats, ret, times, label_times


def _run(mod, monkeypatch, long_only):
    """One trial through the REAL create_objective closure (CPU, seeded,
    2 epochs, real compute_sharpe / compute_regime_sharpes)."""
    feats, ret, times, label_times = _data()
    monkeypatch.setattr(mod, 'MAX_EPOCHS', 2)
    monkeypatch.setattr(mod, '_training_repairs', lambda: False)
    monkeypatch.setattr(mod, '_objective_v3', lambda: False)
    monkeypatch.setattr(mod, '_trainer_seed', lambda: 11)
    monkeypatch.setattr(mod, '_objective_long_only', lambda: long_only)
    monkeypatch.setattr(mod, 'get_walk_forward_folds',
                        lambda *a, **k: list(FOLDS))
    obj = mod.create_objective(feats, {FB: ret}, times, label_times, ['A'],
                               {'A': (0, N)}, 3, {}, asset_type='crypto',
                               has_multi_horizon=True,
                               adaptive_space={'forward_bars': [FB]},
                               study_name='sig_r3_decomp')
    st = optuna.create_study(direction='maximize')
    st.enqueue_trial(dict(PARAMS))
    t0 = time.perf_counter()
    st.optimize(obj, n_trials=1, catch=())
    return st.trials[-1], time.perf_counter() - t0


@pytest.mark.parametrize('long_only', [False, True])
def test_decomp_attrs_recorded_per_fold(hs, monkeypatch, long_only):
    tr, _ = _run(hs, monkeypatch, long_only)
    assert tr.state == optuna.trial.TrialState.COMPLETE
    ua = tr.user_attrs
    nf = len(ua['fold_sharpes'])
    assert nf == len(FOLDS)
    for k in KEYS:
        assert k in ua, k
        assert len(ua[k]) == nf, (k, ua[k])
    for i, (_, val) in enumerate(FOLDS):
        assert ua['n_rows'][i] == len(val)
        n = ua['n_trades'][i]
        assert 0 < n <= math.ceil(len(val) / FB)
        assert math.isclose(ua['cost_drag'][i], 0.60, abs_tol=1e-6)
        assert math.isclose(ua['gross_ret_mean'][i] - ua['net_ret_mean'][i],
                            0.60, abs_tol=1e-6)
        assert 0.0 <= ua['hit_rate'][i] <= 1.0
        assert 0.0 < ua['threshold_pass_rate'][i] <= 1.0
        assert 1.0 <= ua['mean_hold_bars'][i] <= FB
        # a fold with >= 10 trades has a non-floor Sharpe whose sign is
        # the sign of its net mean (compute_sharpe: mean/std*sqrt(tpy))
        if n >= 10 and ua['fold_sharpes'][i] != 0.0:
            assert np.sign(ua['fold_sharpes'][i]) == np.sign(ua['net_ret_mean'][i])
    assert isinstance(ua.get('regime_trade_decomp'), dict)


@pytest.mark.parametrize('long_only', [False, True])
def test_score_and_legacy_attrs_bit_identical(hs, hs_live, monkeypatch,
                                              long_only):
    new, t_new = _run(hs, monkeypatch, long_only)
    old, t_old = _run(hs_live, monkeypatch, long_only)
    assert new.value == old.value            # bit-identical, not isclose
    for k in LEGACY_ATTRS:
        assert new.user_attrs.get(k) == old.user_attrs.get(k), k
    print(f'[overhead] long_only={long_only} trial wall new={t_new:.2f}s '
          f'old={t_old:.2f}s')


@pytest.mark.parametrize('long_only', [False, True])
def test_gross_equals_cost_zero_walk_and_net_equals_cost_walk(hs, monkeypatch,
                                                              long_only):
    """Gross read back from the entries == the cost-0 walk EXACTLY; the
    cost never moves an entry; net == the cost walk to float32 rounding."""
    monkeypatch.setattr(hs, '_objective_long_only', lambda: long_only)
    rng = np.random.default_rng(3)
    n = 5000
    y = rng.normal(0.02, 2.0, n).astype(np.float32)
    p = (y + rng.normal(0, 2.0, n)).astype(np.float32)
    blocks = np.repeat(np.arange(5), n // 5)
    veto = rng.random(n) < 0.1
    for bid, lv in ((None, None), (blocks, None), (blocks, veto)):
        d = hs.fold_trade_decomposition(p, y, 0.7, 12, 'crypto',
                                        block_ids=bid, long_veto=lv)
        g0, e0 = hs.simulate_trades(p, y, 0.7, 12, 0.0, return_entries=True,
                                    block_ids=bid, long_veto=lv)
        nc, ec = hs.simulate_trades(p, y, 0.7, 12, 0.60, return_entries=True,
                                    block_ids=bid, long_veto=lv)
        assert np.array_equal(e0, ec)
        g0 = np.asarray(g0, dtype=np.float64)
        assert d['n_trades'] == len(g0)
        assert d['gross_ret_mean'] == float(g0.mean())
        assert d['gross_ret_std'] == float(g0.std())
        assert math.isclose(d['net_ret_mean'],
                            float(np.asarray(nc, np.float64).mean()),
                            rel_tol=0, abs_tol=1e-6)
        assert np.allclose(g0 - 0.60, np.asarray(nc, np.float64), atol=1e-6)


def test_gross_read_back_is_exact_for_both_sides(hs, monkeypatch):
    monkeypatch.setattr(hs, '_objective_long_only', lambda: False)
    rng = np.random.default_rng(9)
    y = rng.normal(0, 2.0, 3000)
    p = y + rng.normal(0, 1.0, 3000)
    d = hs.fold_trade_decomposition(p, y, 0.5, 6, 'crypto')
    g0 = np.asarray(hs.simulate_trades(p, y, 0.5, 6, 0.0), np.float64)
    assert d['n_trades'] == len(g0)
    assert d['gross_ret_mean'] == float(g0.mean())
    assert d['gross_ret_std'] == float(g0.std())
    assert d['hit_rate'] == float((g0 > 0).mean())


def test_regime_decomp_optional_and_legacy_default(hs):
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1.0, 900)
    p = y + rng.normal(0, 1.0, 900)
    base = hs.compute_regime_sharpes(p, y, 0.3, 4, 'crypto')
    out = {}
    again = hs.compute_regime_sharpes(p, y, 0.3, 4, 'crypto', decomp_out=out)
    assert base == again
    assert set(out) <= {'bull', 'bear', 'sideways'} and out


def test_decomposition_never_raises(hs):
    d = hs.fold_trade_decomposition(np.zeros(3), np.zeros(4), 0.1, 4)
    assert set(KEYS) <= set(d)
    e = hs.fold_trade_decomposition(np.zeros(10), np.zeros(10), 0.1, 4)
    assert e['n_trades'] == 0 and e['gross_ret_mean'] is None


def test_overhead_one_extra_walk_per_fold(hs):
    rng = np.random.default_rng(2)
    n = 35000   # ~ one crypto fold's val slice
    y = rng.normal(0.05, 3.0, n).astype(np.float32)
    p = (0.1 * y + rng.normal(0, 0.5, n)).astype(np.float32)
    t0 = time.perf_counter()
    hs.compute_sharpe(p, y, 0.4, 24, 'crypto')
    t_sharpe = time.perf_counter() - t0
    t0 = time.perf_counter()
    hs.fold_trade_decomposition(p, y, 0.4, 24, 'crypto')
    t_dec = time.perf_counter() - t0
    print(f'[overhead] 35k rows: compute_sharpe {t_sharpe*1e3:.1f} ms, '
          f'decomposition {t_dec*1e3:.1f} ms')
    assert t_dec < 5 * t_sharpe + 0.05
