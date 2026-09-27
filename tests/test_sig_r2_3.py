"""SIG-R2-3: provable flag state for the trainer.

(a) main() prints ONE `[FLAGS] name=value ...` line (every training-path
    flag via getattr(strategy_config, name, None), env overrides, preset,
    seed base, prefix, mode, --trials, resolved data path) after argparse
    and before the study DB is touched / created.
(b) TRAINING_REPAIRS_V1's L1/L2/L5/L6 branches each print ONE
    `[REPAIRS] Lx ...` line when they take effect (once per trial / call).

Pure instrumentation: every numeric output of the objective closure,
get_walk_forward_folds, compute_regime_sharpes and final_refit is
bit-identical to the pre-patch module (SIG_R2_LIVE_HS) with the repairs
flag OFF and ON; OFF prints no [REPAIRS] line. The banner never matches
run_pipeline's trainer-log regexes (run_pipeline.py:568-592).

Module under test: SIG_R2_HS (default: the repo's scripts/hypersearch_v2.py).
"""
import contextlib
import importlib.util
import io
import os
import re
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
          'huber_delta': 1.3, 'trade_threshold': 0.05, 'scheduler': 'cosine'}
FOLDS = [(np.arange(8, 500), np.arange(520, 700)),
         (np.arange(8, 800), np.arange(820, 1000))]
FLAGS = ('OBJECTIVE_LONG_ONLY', 'HYPERSEARCH_V3', 'OBJECTIVE_V3',
         'TRAINING_REPAIRS_V1', 'BLEND_FIT_ON_REFIT',
         'BLEND_THRESHOLD_RESELECT', 'PROMOTION_GATE_V2', 'KISH_NEFF_ENABLED',
         'LGB_REFIT_FULL', 'HOLDOUT_SPAN_BY_TARGET', 'BARS_PER_YEAR_MEASURED',
         'FAILED_TRIAL_PRUNE')

optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_num_threads(1)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def hs():
    return _load(HS_PATH, 'hs_sig_r2_3')


@pytest.fixture(scope='module')
def hs_live():
    return _load(LIVE_HS_PATH, 'hs_sig_r2_3_live')


def _data(n=N, seed=5):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, 3)).astype(np.float32)
    ret = rng.normal(0.0, 3.0, size=n).astype(np.float32)
    times = (1_760_000_000 + np.arange(n) * 3600).astype(np.int64)
    label_times = times[np.minimum(np.arange(n) + FB, n - 1)]
    return feats, ret, times, label_times


def _captured(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def _repairs_lines(text):
    return [ln for ln in text.splitlines() if ln.startswith('[REPAIRS]')]


# --------------------------------------------------------------------------
# (a) banner
# --------------------------------------------------------------------------

class _Stop(Exception):
    pass


def _run_main_until_load_data(mod, monkeypatch, argv):
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(mod, 'load_adaptive_state',
                        lambda asset: {'mode': 'refine'})
    monkeypatch.setattr(mod, 'get_search_space_for_trial', lambda st: {})
    monkeypatch.setattr(mod, 'get_trial_count', lambda mode, **k: 5)

    def stop(*a, **k):
        raise _Stop()
    monkeypatch.setattr(mod, 'load_data', stop)
    monkeypatch.setattr(mod.optuna, 'create_study', stop)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(_Stop):
        mod.main()
    return buf.getvalue()


def test_banner_one_greppable_line_before_study(hs, monkeypatch):
    import strategy_config
    monkeypatch.delattr(strategy_config, 'KISH_NEFF_ENABLED', raising=False)
    monkeypatch.setenv('TRADER_FAILED_TRIAL_PRUNE', 'yes')
    out = _run_main_until_load_data(
        hs, monkeypatch, ['hypersearch_v2.py', '--trials', '7', '--preset',
                          'stationary', '--no-status', '--prefix', 'stock',
                          '--data', 'stock_training_data.csv'])
    lines = [ln for ln in out.splitlines() if re.match(r'^\[FLAGS\]', ln)]
    assert len(lines) == 1, out
    banner = lines[0]
    toks = dict(t.split('=', 1) for t in banner.split()[1:])
    for name in FLAGS:
        assert name in toks, name
    assert toks['KISH_NEFF_ENABLED'] == 'None'             # absent -> None
    assert toks['OBJECTIVE_LONG_ONLY'] == str(
        getattr(strategy_config, 'OBJECTIVE_LONG_ONLY', None))
    assert toks['env.TRADER_FAILED_TRIAL_PRUNE'] == 'yes'
    assert toks['preset'] == 'stationary'
    assert toks['prefix'] == 'stock' and toks['mode'] == 'refine'
    assert toks['trials'] == '7' and toks['trials_arg'] == '7'
    assert toks['data'].endswith(('stock_training_data.parquet',
                                  'stock_training_data.csv'))
    assert 'seed_base' in toks
    # run_pipeline's trainer-log parsers must never fire on it
    assert re.match(r'\[\s*(\d+)\]', banner) is None
    assert re.match(r'Resuming from (\d+) prior trials', banner) is None
    assert re.match(r'Prior best (?:sharpe|score)=(-?\d+\.\d+)', banner) is None
    assert '** BEST **' not in banner


def test_banner_default_trials_resolves_adaptive_count(hs, monkeypatch):
    out = _run_main_until_load_data(hs, monkeypatch,
                                    ['hypersearch_v2.py', '--no-status'])
    banner = [ln for ln in out.splitlines() if ln.startswith('[FLAGS]')][0]
    toks = dict(t.split('=', 1) for t in banner.split()[1:])
    assert toks['trials'] == '5'                 # adaptive count wins
    assert toks['trials_arg'] == str(hs.NUM_TRIALS)
    assert toks['prefix'] == 'None'


# --------------------------------------------------------------------------
# (b) [REPAIRS] lines + byte-identity vs the pre-patch module
# --------------------------------------------------------------------------

def _objective(mod, monkeypatch, repairs):
    feats, ret, times, label_times = _data()
    monkeypatch.setattr(mod, 'MAX_EPOCHS', 2)
    monkeypatch.setattr(mod, '_training_repairs', lambda: repairs)
    monkeypatch.setattr(mod, '_objective_v3', lambda: False)
    monkeypatch.setattr(mod, '_trainer_seed', lambda: 11)
    monkeypatch.setattr(mod, 'get_walk_forward_folds',
                        lambda *a, **k: list(FOLDS))
    cache = {}
    obj = mod.create_objective(feats, {FB: ret}, times, label_times, ['A'],
                               {'A': (0, N)}, 3, cache, asset_type='crypto',
                               has_multi_horizon=True,
                               adaptive_space={'forward_bars': [FB]},
                               study_name='sig_r2_3')
    trial = optuna.trial.FixedTrial(dict(PARAMS), number=0)
    score, out = _captured(obj, trial)
    return score, trial.user_attrs, cache, out


def _assert_same(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _assert_same(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f'{path}[{i}]')
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), path
    elif hasattr(a, 'center_') or hasattr(a, 'scale_'):   # RobustScaler
        assert np.array_equal(a.center_, b.center_), path
        assert np.array_equal(a.scale_, b.scale_), path
    else:
        assert a == b or (a != a and b != b), (path, a, b)


@pytest.mark.parametrize('repairs', [False, True])
def test_objective_outputs_bit_identical_and_lines(hs, hs_live, monkeypatch,
                                                   repairs):
    s_new, ua_new, c_new, out_new = _objective(hs, monkeypatch, repairs)
    s_old, ua_old, c_old, out_old = _objective(hs_live, monkeypatch, repairs)
    assert np.isfinite(s_new)
    assert s_new == s_old                              # bit-identical score
    # SIG-R3-DECOMP adds instrumentation-only user_attrs (DECOMP_KEYS +
    # regime_trade_decomp): every attr the reference records must be
    # bit-identical; extra keys are allowed.
    assert set(ua_old) <= set(ua_new)
    _assert_same({k: ua_new[k] for k in ua_old}, ua_old)  # cfg/folds/regimes
    _assert_same(c_new, c_old)                         # state/oof/epochs
    lines = _repairs_lines(out_new)
    if not repairs:
        assert lines == []
        return
    tags = [ln.split()[1] for ln in lines]
    assert tags.count('L1') == 1 and tags.count('L5') == 1   # once per trial
    assert tags.count('L2') == 1          # regime penalty: once per trial
    assert 'huber_delta=1.3' in [ln for ln in lines if ' L1 ' in ln][0]


def _folds_input(n=4000):
    rng = np.random.default_rng(1)
    times = (1_760_000_000 + np.arange(n) * 3600).astype(np.int64)
    label_times = times + 48 * 3600
    return times, label_times, ['A'], {'A': (0, n)}


@pytest.mark.parametrize('repairs', [False, True])
def test_folds_regimes_refit_bit_identical_and_lines(hs, hs_live,
                                                     monkeypatch, repairs):
    outs = {}
    for tag, mod in (('new', hs), ('old', hs_live)):
        monkeypatch.setattr(mod, '_training_repairs', lambda: repairs)
        t, lt, tick, tb = _folds_input()
        folds, o1 = _captured(mod.get_walk_forward_folds, t, lt, tick, tb, 8)
        rng = np.random.default_rng(2)
        rs, o2 = _captured(mod.compute_regime_sharpes,
                           rng.normal(0, 1, 800), rng.normal(0, 4, 800),
                           0.3, forward_bars=4)
        feats, ret, times, label_times = _data()
        torch.manual_seed(0)
        ref, o3 = _captured(mod.final_refit, dict(PARAMS), ret, feats, times,
                            label_times, ['A'], {'A': (0, N)}, 3, [2, 2],
                            [1.0, 1.0], seed=5)
        outs[tag] = (folds, rs, ref, o1 + o2 + o3)
    (f_n, r_n, ref_n, out_n), (f_o, r_o, ref_o, _) = outs['new'], outs['old']
    assert len(f_n) == len(f_o) > 0
    for (a, b), (c, d) in zip(f_n, f_o):
        assert np.array_equal(a, c) and np.array_equal(b, d)
    assert r_n == r_o
    assert ref_n is not None
    _assert_same(ref_n[0], ref_o[0])                   # refit state_dict
    _assert_same(ref_n[1], ref_o[1])                   # refit scaler
    _assert_same(ref_n[2], ref_o[2])                   # refit info
    tags = [ln.split()[1] for ln in _repairs_lines(out_n)]
    if repairs:
        assert tags == ['L6', 'L2', 'L5']
    else:
        assert tags == []
