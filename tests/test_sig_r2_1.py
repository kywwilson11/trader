"""SIG-R2-1: failed-trial 0.0 sentinel outranks every negative trial.

scripts/hypersearch_v2.py's objective returns 0.0 for a trial that produced
no honest score — the OOM/RuntimeError handler, `if not folds`, a fold that
timed out before its first checkpoint (fold_sharpe = 0.0 appended), and zero
completed folds (avg_sharpe = 0.0). Optuna records that 0.0 as a COMPLETE
value: with OBJECTIVE_LONG_ONLY most real scores are negative, so the failed
trial becomes study.best_trial, seeds TPE's 'good' set, and inflates the
COMPLETE-count deflation pool / cum_trials.

Fix (default-OFF flag FAILED_TRIAL_PRUNE, env TRADER_FAILED_TRIAL_PRUNE
wins): raise optuna.TrialPruned instead — state PRUNED, excluded from
best_trial and from every `state == COMPLETE` pool count. OFF path: every
failure mode still returns exactly 0.0 and a healthy trial's score is
unchanged vs the pre-patch module (loaded from SIG_R2_LIVE_HS when given).

Module under test: SIG_R2_HS (default: the repo's scripts/hypersearch_v2.py).
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

REPO = Path(os.environ.get('SIG_R2_REPO',
                           Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
HS_PATH = os.environ.get('SIG_R2_HS',
                         str(REPO / 'scripts' / 'hypersearch_v2.py'))
LIVE_HS_PATH = os.environ.get('SIG_R2_LIVE_HS', HS_PATH)

FB = 4
N = 1200
PARAMS = {'forward_bars': FB, 'seq_len': 8, 'hidden_dim': 64,
          'num_layers': 1, 'n_heads': 2, 'dropout': 0.10,
          'learning_rate': 1e-3, 'batch_size': 512, 'weight_decay': 1e-4,
          'huber_delta': 1.0, 'trade_threshold': 0.5, 'scheduler': 'cosine'}
# AUDIT-1 repro R2: six real OBJECTIVE_LONG_ONLY trial scores
REAL_SCORES = [-1.878, -1.503, -1.61, -1.517, -2.819, -1.192]
FOLDS = [(np.arange(8, 500), np.arange(520, 700)),
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
    return _load(HS_PATH, 'hs_sig_r2_1')


@pytest.fixture(scope='module')
def hs_live():
    return _load(LIVE_HS_PATH, 'hs_sig_r2_1_live')


def _data():
    rng = np.random.default_rng(5)
    feats = rng.normal(size=(N, 3)).astype(np.float32)
    ret = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    times = (1_760_000_000 + np.arange(N) * 3600).astype(np.int64)
    label_times = times[np.minimum(np.arange(N) + FB, N - 1)]
    return feats, ret, times, label_times


class _FakeTime:
    """time.time(): first call (trial_start) 0, then far past 900 s."""
    def __init__(self):
        self.n = 0

    def time(self):
        self.n += 1
        return 0.0 if self.n == 1 else 1e6


def _boom(*a, **k):
    raise RuntimeError('CUDA out of memory (synthetic)')


def _objective(mod, monkeypatch, mode):
    """Real create_objective closure, CPU, seeded, heavy seams pinned."""
    feats, ret, times, label_times = _data()
    monkeypatch.setattr(mod, 'MAX_EPOCHS', 2)
    monkeypatch.setattr(mod, 'compute_regime_sharpes',
                        lambda *a, **k: {'bull': 0.0, 'bear': 0.0,
                                         'sideways': 0.0, 'min': 0.0})
    monkeypatch.setattr(mod, '_training_repairs', lambda: False)
    monkeypatch.setattr(mod, '_objective_v3', lambda: False)
    monkeypatch.setattr(mod, '_trainer_seed', lambda: 11)
    folds = FOLDS
    if mode == 'nofolds':
        folds = []
    elif mode == 'zerofolds':          # every fold below the row minimum
        folds = [(np.arange(8, 400), np.arange(420, 480))]
    elif mode == 'oom':
        monkeypatch.setattr(mod, 'RegressionLSTM', _boom)
    elif mode == 'timeout':
        monkeypatch.setattr(mod, 'time', _FakeTime())
    else:
        assert mode == 'ok'
    monkeypatch.setattr(mod, 'get_walk_forward_folds',
                        lambda *a, **k: list(folds))
    return mod.create_objective(feats, {FB: ret}, times, label_times, ['A'],
                                {'A': (0, N)}, 3, {}, asset_type='crypto',
                                has_multi_horizon=True,
                                adaptive_space={'forward_bars': [FB]},
                                study_name='sig_r2_1')


def _study_with_real_trials():
    st = optuna.create_study(direction='maximize')
    for v in REAL_SCORES:
        st.add_trial(optuna.trial.create_trial(params={}, distributions={},
                                               value=v))
    return st


def _run_one(obj):
    """One trial through study.optimize exactly as main() calls it."""
    st = _study_with_real_trials()
    st.enqueue_trial(dict(PARAMS))
    st.optimize(obj, n_trials=1, catch=(Exception,))
    return st, st.trials[-1]


def _off(monkeypatch, mod):
    monkeypatch.delenv('TRADER_FAILED_TRIAL_PRUNE', raising=False)
    import strategy_config
    monkeypatch.delattr(strategy_config, 'FAILED_TRIAL_PRUNE', raising=False)


FAIL_MODES = ['oom', 'nofolds', 'zerofolds', 'timeout']


@pytest.mark.parametrize('mode', FAIL_MODES)
def test_on_failed_trial_is_pruned_not_best_not_pooled(hs, monkeypatch,
                                                       mode):
    monkeypatch.setenv('TRADER_FAILED_TRIAL_PRUNE', '1')
    st, t = _run_one(_objective(hs, monkeypatch, mode))
    assert t.state == optuna.trial.TrialState.PRUNED, (mode, t.state,
                                                       t.value)
    assert 'failed_trial' in t.user_attrs
    assert st.best_trial.value == max(REAL_SCORES)      # a REAL trial wins
    completed = [x for x in st.trials
                 if x.state == optuna.trial.TrialState.COMPLETE
                 and x.value is not None]
    assert len(completed) == len(REAL_SCORES)            # pool not inflated


@pytest.mark.parametrize('mode', FAIL_MODES)
def test_off_failed_trial_keeps_legacy_zero_sentinel(hs, hs_live,
                                                     monkeypatch, mode):
    _off(monkeypatch, hs)
    st, t = _run_one(_objective(hs, monkeypatch, mode))
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert t.value == 0.0 and 'failed_trial' not in t.user_attrs
    # the defect, byte-pinned on the OFF path: the failure is best_trial
    assert st.best_trial.number == t.number
    # identical to the pre-patch module
    st_l, t_l = _run_one(_objective(hs_live, monkeypatch, mode))
    assert (t_l.state, t_l.value) == (t.state, t.value)


def test_config_constant_enables_and_env_zero_overrides(hs, monkeypatch):
    import strategy_config
    monkeypatch.delenv('TRADER_FAILED_TRIAL_PRUNE', raising=False)
    monkeypatch.setattr(strategy_config, 'FAILED_TRIAL_PRUNE', True,
                        raising=False)
    assert hs._failed_trial_prune() is True
    monkeypatch.setenv('TRADER_FAILED_TRIAL_PRUNE', '0')
    assert hs._failed_trial_prune() is False
    monkeypatch.setenv('TRADER_FAILED_TRIAL_PRUNE', 'on')
    monkeypatch.setattr(strategy_config, 'FAILED_TRIAL_PRUNE', False)
    assert hs._failed_trial_prune() is True
    monkeypatch.delenv('TRADER_FAILED_TRIAL_PRUNE')
    monkeypatch.delattr(strategy_config, 'FAILED_TRIAL_PRUNE')
    assert hs._failed_trial_prune() is False           # absent = legacy


@pytest.mark.parametrize('flag', ['off', 'on'])
def test_healthy_trial_value_identical_to_pre_patch(hs, hs_live,
                                                    monkeypatch, flag):
    if flag == 'on':
        monkeypatch.setenv('TRADER_FAILED_TRIAL_PRUNE', '1')
    else:
        _off(monkeypatch, hs)
    v_new = _objective(hs, monkeypatch, 'ok')(
        optuna.trial.FixedTrial(dict(PARAMS), number=0))
    v_old = _objective(hs_live, monkeypatch, 'ok')(
        optuna.trial.FixedTrial(dict(PARAMS), number=0))
    assert np.isfinite(v_new)
    assert v_new == v_old                         # bit-identical score
