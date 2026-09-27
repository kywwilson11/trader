"""SIG-R1-A2: the regime "30% penalty" rewards negative-scoring trials.

scripts/hypersearch_v2.py _train_walk_forward: `if regime_sharpes['min']
< -0.5: score *= 0.7`. For score < 0, *0.7 moves the score TOWARD zero —
a trial with a losing regime outranks an otherwise-identical one without.
Tonight's 6-trial crypto run: every trial's logged score == 0.700 x
(mean - 0.5*std) with all six negative, i.e. all six were "rewarded".

Fix (behind TRAINING_REPAIRS_V1, the bundle that already rewrites this
penalty's regime mask — L2): score -= 0.3*|score| (identical to *0.7 for
score > 0). Flag OFF: *0.7 verbatim.

Runs the REAL objective closure (create_objective) on CPU for one 1-epoch
fold with compute_sharpe / compute_regime_sharpes pinned, via an Optuna
FixedTrial. Module under test: SIG_R1_HS (default <repo>/scripts/...).
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


@pytest.fixture(scope='module')
def hs():
    spec = importlib.util.spec_from_file_location('hs_sig_r1_a2', HS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _score(hs, monkeypatch, fold_sharpe, regime_min, repairs):
    rng = np.random.default_rng(3)
    feats = rng.normal(size=(N, 3)).astype(np.float32)
    ret = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    times = (1_760_000_000 + np.arange(N) * 3600).astype(np.int64)
    fold = (np.arange(8, 700), np.arange(720, 1000))
    monkeypatch.setattr(hs, 'MAX_EPOCHS', 1)
    monkeypatch.setattr(hs, 'get_walk_forward_folds',
                        lambda *a, **k: [fold])
    monkeypatch.setattr(hs, 'compute_sharpe',
                        lambda *a, **k: float(fold_sharpe))
    monkeypatch.setattr(hs, 'compute_regime_sharpes',
                        lambda *a, **k: {'bull': regime_min,
                                         'bear': regime_min,
                                         'sideways': regime_min,
                                         'min': regime_min})
    monkeypatch.setattr(hs, '_training_repairs', lambda: repairs)
    monkeypatch.setattr(hs, '_objective_v3', lambda: False)
    monkeypatch.setattr(hs, '_trainer_seed', lambda: 11)
    obj = hs.create_objective(feats, {FB: ret}, times, times, ['A'],
                              {'A': (0, N)}, 3, {}, asset_type='crypto',
                              has_multi_horizon=True,
                              adaptive_space={'forward_bars': [FB]},
                              study_name='sig_r1_a2')
    trial = optuna.trial.FixedTrial(dict(PARAMS), number=0)
    score = obj(trial)
    assert trial.user_attrs['regime_sharpes']['min'] == regime_min
    return score


def test_negative_score_is_penalized_not_rewarded(hs, monkeypatch):
    s_bad = _score(hs, monkeypatch, -2.0, -1.0, repairs=True)
    s_ok = _score(hs, monkeypatch, -2.0, 0.0, repairs=True)
    assert s_ok == pytest.approx(-2.0)
    assert s_bad == pytest.approx(-2.6)          # -2.0 - 0.3*2.0
    assert s_bad < s_ok                           # a bad regime never helps


def test_positive_score_penalty_unchanged_under_flag(hs, monkeypatch):
    assert _score(hs, monkeypatch, 2.0, -1.0, repairs=True) == \
        pytest.approx(1.4)


def test_flag_off_legacy_multiplier_pinned(hs, monkeypatch):
    """OFF-path byte-pin: legacy `score *= 0.7` for both signs."""
    assert _score(hs, monkeypatch, -2.0, -1.0, repairs=False) == \
        pytest.approx(-1.4)
    assert _score(hs, monkeypatch, 2.0, -1.0, repairs=False) == \
        pytest.approx(1.4)
