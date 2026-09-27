"""SIG-R1-A4: HYPERSEARCH_V3 final-refit epoch budget is one epoch short.

_train_walk_forward records `max(best_epoch, 0)` where best_epoch is the
0-based LOOP INDEX of the best-val-loss epoch — index e means e+1 epochs
had been trained. final_refit trains `for epoch in range(epochs)` with
epochs = refit_epoch_budget(...) = median(recorded) -> one epoch fewer than
the checkpoint it claims to reproduce ("FIXED epoch budget = median of the
winning trial's per-fold best epochs"). The LightGBM analog of the same
collective-early-stopping rule (fixed_boost_rounds) uses best_iteration,
which LightGBM defines as a 1-based COUNT.

Fix: record best_epoch + 1 (a no-best -1 still records 0, as legacy's
max(-1, 0) did). Only consumer: final_refit, which runs only under
HYPERSEARCH_V3 (default OFF) — flag-OFF outputs are unchanged.
Module under test: SIG_R1_HS.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
optuna = pytest.importorskip('optuna')
pytest.importorskip('sklearn')

REPO = Path(os.environ.get('SIG_R1_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R1_HS',
                         str(REPO / 'scripts' / 'hypersearch_v2.py'))

FB = 4
N = 1200
PARAMS = {'forward_bars': FB, 'seq_len': 8, 'hidden_dim': 64,
          'num_layers': 1, 'n_heads': 2, 'dropout': 0.10,
          'learning_rate': 1e-3, 'batch_size': 512, 'weight_decay': 1e-4,
          'huber_delta': 1.0, 'trade_threshold': 0.5, 'scheduler': 'cosine'}
VAL_LOSSES = [3.0, 1.0, 2.0]   # best at loop index 1 == after 2 epochs


@pytest.fixture(scope='module')
def hs():
    spec = importlib.util.spec_from_file_location('hs_sig_r1_a4', HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data():
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(N, 3)).astype(np.float32)
    ret = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    times = (1_760_000_000 + np.arange(N) * 3600).astype(np.int64)
    label_times = times[np.minimum(np.arange(N) + FB, N - 1)]
    return feats, ret, times, label_times


def test_recorded_budget_is_epoch_count_and_refit_trains_it(hs, monkeypatch):
    feats, ret, times, label_times = _data()
    real_huber = torch.nn.functional.huber_loss
    calls = {'n': 0}

    def scripted_huber(inp, tgt, *a, **k):
        if k.get('reduction', 'mean') == 'none':      # training criterion
            return real_huber(inp, tgt, *a, **k)
        v = VAL_LOSSES[calls['n']]                     # legacy val loss
        calls['n'] += 1
        return torch.tensor(v)

    monkeypatch.setattr(torch.nn.functional, 'huber_loss', scripted_huber)
    monkeypatch.setattr(hs, 'MAX_EPOCHS', len(VAL_LOSSES))
    monkeypatch.setattr(hs, 'get_walk_forward_folds', lambda *a, **k: [
        (np.arange(8, 700), np.arange(720, 1000))])   # 280 val rows: 1 batch
    monkeypatch.setattr(hs, 'compute_sharpe', lambda *a, **k: 1.0)
    monkeypatch.setattr(hs, 'compute_regime_sharpes',
                        lambda *a, **k: {'bull': 0.0, 'bear': 0.0,
                                         'sideways': 0.0, 'min': 0.0})
    monkeypatch.setattr(hs, '_training_repairs', lambda: False)
    monkeypatch.setattr(hs, '_objective_v3', lambda: False)
    monkeypatch.setattr(hs, '_trainer_seed', lambda: 11)
    monkeypatch.setattr(hs, '_fixed_holdout_days', lambda: None)
    cache = {}
    obj = hs.create_objective(feats, {FB: ret}, times, label_times, ['A'],
                              {'A': (0, N)}, 3, cache, asset_type='crypto',
                              has_multi_horizon=True,
                              adaptive_space={'forward_bars': [FB]},
                              study_name='sig_r1_a4')
    assert obj(optuna.trial.FixedTrial(dict(PARAMS), number=0)) > 0
    assert calls['n'] == len(VAL_LOSSES)
    recorded = cache[0]['fold_best_epochs']
    assert recorded == [2]          # best checkpoint had 2 epochs of training
    monkeypatch.setattr(torch.nn.functional, 'huber_loss', real_huber)
    out = hs.final_refit(dict(PARAMS), ret, feats, times, label_times,
                         ['A'], {'A': (0, N)}, 3, recorded, [1.0], seed=5)
    assert out is not None and out[2]['epochs'] == 2
