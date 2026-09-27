"""SIG-R3-METADUMP — TRADER_META_FRAME_DUMP instrumentation hook in meta_label.

Pins:
  * env parsing (unset/''/0/false/no/off = OFF; 1/true/yes/on = repo logs dir;
    anything else = a directory; book prefix in the file names);
  * _dump_meta_frame writes the npz frame + JSON sidecar with the documented
    columns/row counts, the sidecar is directly consumable by
    scripts/reliability_report.py when the purged arm exists (rc 0) and is a
    clean NO-DATA (rc 2) when it does not, and the helper never raises;
  * source pin: train_meta reads the env exactly once and calls the dump
    exactly once, behind `if _dump_target is not None:`;
  * end-to-end (lightgbm + sklearn + joblib; skipped without them): train_meta
    on a stubbed synthetic frame returns the same value and stages
    byte-identical artifacts with the env unset vs set, and never touches the
    dump path when unset. With SIG_R3_META_REF=<pre-patch meta_label.py> the
    unset run is also compared byte-for-byte against the pre-patch module.

SIG_R3_META=<path> selects the module under test (default: repo meta_label.py);
SIG_R3_REPO=<repo root> when the file runs from outside tests/ (staging).
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(os.environ.get('SIG_R3_REPO',
                          Path(__file__).resolve().parent.parent))
MOD_PATH = Path(os.environ.get('SIG_R3_META', REPO / 'meta_label.py'))
REF_PATH = os.environ.get('SIG_R3_META_REF')
ENV = 'TRADER_META_FRAME_DUMP'

NPZ_KEYS = {'row', 'ticker', 'entry_time_ns', 'exit_time_ns', 'entry_e',
            'exit_e', 'fold_id', 'split', 'y', 'net_pct', 'raw_score',
            'raw_oof', 'p_served', 'p_legacy', 'p_purged', 'X',
            'feature_names'}


def _load(path, name):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def ml():
    return _load(MOD_PATH, 'meta_label_sig_r3_under_test')


# ---------------------------------------------------------------------------
# env parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('val', [None, '', '0', 'false', 'No', ' OFF '])
def test_dump_target_off(ml, monkeypatch, val):
    if val is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, val)
    assert ml._meta_frame_dump_target('') is None
    assert ml._meta_frame_dump_target('stock') is None


def test_dump_target_on(ml, monkeypatch, tmp_path):
    monkeypatch.setattr(ml, 'BASE_DIR', tmp_path)
    for v in ('1', 'true', 'YES', 'on'):
        monkeypatch.setenv(ENV, v)
        npz, js = ml._meta_frame_dump_target('')
        assert npz == tmp_path / 'logs' / 'meta_frame_dump' / 'meta_frame.npz'
        assert js == tmp_path / 'logs' / 'meta_frame_dump' / 'meta_frame.json'
    monkeypatch.setenv(ENV, str(tmp_path / 'x'))
    npz, js = ml._meta_frame_dump_target('stock')
    assert npz == tmp_path / 'x' / 'stock_meta_frame.npz'
    assert js == tmp_path / 'x' / 'stock_meta_frame.json'
    # no collision with any meta artifact name
    arts = {p.name for p in ml._paths('stock').values()}
    assert npz.name not in arts and js.name not in arts


# ---------------------------------------------------------------------------
# the dump helper (unit)
# ---------------------------------------------------------------------------

class _Booster:
    best_iteration = 17

    def predict(self, X):
        return 1.0 / (1.0 + np.exp(-np.asarray(X, float)[:, 0]))


def _frame(n=300, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 13))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    base = pd.Timestamp('2025-01-01', tz='UTC')
    ts = [base + pd.Timedelta(hours=int(h)) for h in rng.permutation(n)]
    exit_ts = [t + pd.Timedelta(hours=3) for t in ts]
    order = np.argsort(np.asarray(ts))
    tickers = [f'T{i % 3}' for i in range(n)]
    net = list(rng.normal(size=n))
    return X[order], y[order], order, ts, exit_ts, net, tickers


def _call(ml, target, calib_used, oof=True, **kw):
    X, y, order, ts, exit_ts, net, tickers = _frame()
    n = len(y)
    split = int(n * 0.8)
    p_served = np.clip(_Booster().predict(X) * 0.9 + 0.05, 0, 1)
    prov = {'used': calib_used, 'mode_requested': 'purged_oof'}
    args = dict(prefix='stock', booster=_Booster(), X=X, y=y, order=order,
                ts=ts, exit_ts=exit_ts, net_list=net, tickers=tickers,
                split=split, p_served=p_served, calib_prov=prov,
                oof_raw=(np.linspace(0.1, 0.9, n) if oof else None),
                params={'num_leaves': 31})
    args.update(kw)
    return ml._dump_meta_frame(target, **args), n, split, p_served


def _rel_report(js):
    return subprocess.run(
        [sys.executable, str(REPO / 'scripts' / 'reliability_report.py'),
         '--in', str(js)], capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('v2', [False, True])
def test_dump_purged_arm_schema_and_reliability(ml, tmp_path, monkeypatch, v2):
    pytest.importorskip('sklearn')
    # the refit legacy arm follows train_meta's legacy branch: V2 ->
    # calibration.fit_calibrator, else sklearn isotonic
    monkeypatch.setattr(ml, '_calibration_v2_on', lambda: v2)
    target = (tmp_path / 'd' / 'stock_meta_frame.npz',
              tmp_path / 'd' / 'stock_meta_frame.json')
    out, n, split, p_served = _call(ml, target, 'purged_oof')
    assert out == target[0] and target[0].exists() and target[1].exists()
    with np.load(target[0], allow_pickle=False) as z:
        assert set(z.files) == NPZ_KEYS
        for k in NPZ_KEYS - {'X', 'feature_names'}:
            assert len(z[k]) == n, k
        assert z['X'].shape == (n, 13)
        assert list(z['feature_names']) == list(ml.META_FEATURES)
        assert (z['split'] == 'train').sum() == split
        assert (z['split'] == 'earlystop').sum() == n - split
        assert np.all(np.diff(z['entry_time_ns']) >= 0)       # argsort order
        assert np.all(z['exit_time_ns'] - z['entry_time_ns'] == 3 * 3600 * 10**9)
        assert np.allclose(z['entry_e'] * 1e9, z['entry_time_ns'])
        assert sorted(set(z['fold_id'].tolist())) == [0, 1, 2, 3, 4]
        assert np.array_equal(z['p_purged'], p_served)          # served arm
        assert np.isfinite(z['p_legacy']).all()                 # refit arm
        assert not np.array_equal(z['p_legacy'], z['p_purged'])
        assert np.isfinite(z['raw_oof']).all()
        assert set(z['ticker'].tolist()) == {'T0', 'T1', 'T2'}
    sc = json.loads(target[1].read_text())
    assert sc['kfold'] == {'k': 5, 'embargo': 0.05 if v2 else 0.0}
    assert sc['schema'] == 'meta_frame_dump/v1'
    assert sc['counts']['n_rows'] == n and sc['counts']['n_earlystop'] == n - split
    assert sc['counts']['n_oof_finite'] == n
    assert sc['calibration']['used'] == 'purged_oof'
    assert sc['reliability_slice'] == 'earlystop'
    assert sc['reliability_holdout_honest'] is False
    assert len(sc['y']) == len(sc['p_legacy']) == len(sc['p_purged']) == n - split
    r = _rel_report(target[1])
    assert r.returncode == 0, r.stderr
    assert 'VERDICT' in r.stdout


def test_dump_legacy_only_is_clean_no_data(ml, tmp_path):
    target = (tmp_path / 'meta_frame.npz', tmp_path / 'meta_frame.json')
    out, n, split, p_served = _call(ml, target, 'legacy', oof=False)
    assert out == target[0]
    with np.load(target[0], allow_pickle=False) as z:
        assert np.array_equal(z['p_legacy'], p_served)
        assert np.isnan(z['p_purged']).all() and np.isnan(z['raw_oof']).all()
    sc = json.loads(target[1].read_text())
    assert 'p_purged' not in sc and len(sc['p_legacy']) == n - split
    assert sc['counts']['n_oof_finite'] == 0
    r = _rel_report(target[1])
    assert r.returncode == 2 and 'p_purged' in r.stderr


def test_dump_never_raises(ml, tmp_path, capsys):
    blocker = tmp_path / 'file'
    blocker.write_text('x')                      # parent "dir" is a file
    out, *_ = _call(ml, (blocker / 'a.npz', blocker / 'a.json'), 'legacy')
    assert out is None
    out, *_ = _call(ml, (tmp_path / 'b.npz', tmp_path / 'b.json'), 'legacy',
                    tickers=['T0'])              # length mismatch
    assert out is None
    assert not (tmp_path / 'b.npz').exists() and not (tmp_path / 'b.json').exists()
    assert 'frame dump failed' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# source pin
# ---------------------------------------------------------------------------

def test_train_meta_source_pin():
    src = MOD_PATH.read_text()
    body = src.split('\ndef train_meta(')[1].split('\nif __name__')[0]
    assert body.count('_meta_frame_dump_target(') == 1
    assert body.count('_dump_meta_frame(') == 1
    assert 'os.environ' not in body
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if '_dump_meta_frame(' in ln or 'tk_rows.extend(' in ln \
                or 'tk_rows_is.extend(' in ln:
            j = i - 1
            while lines[j].strip().startswith('#'):
                j -= 1                   # skip comment lines
            assert lines[j].strip() == 'if _dump_target is not None:', ln
            assert (len(lines[j]) - len(lines[j].lstrip())
                    < len(ln) - len(ln.lstrip()))    # ln is inside the if
    # the env is read in exactly one place module-wide
    assert len(re.findall(r"os\.environ\.get\(META_FRAME_DUMP_ENV\)", src)) == 1


# ---------------------------------------------------------------------------
# end-to-end train_meta on a stubbed synthetic frame
# ---------------------------------------------------------------------------

def _synthetic_df(n_per=2400, tickers=('AAA', 'BBB', 'CCC', 'DDD', 'EEE'),
                  seed=11):
    frames = []
    idx = pd.date_range('2025-01-01', periods=n_per, freq='h', tz='UTC')
    for k, tk in enumerate(tickers):
        rng = np.random.default_rng(seed + k)
        close = 100.0 * (k + 1) * np.exp(np.cumsum(rng.normal(0, 0.004, n_per)))
        df = pd.DataFrame({
            'Ticker': tk, 'Close': close,
            'High': close * (1 + np.abs(rng.normal(0, 0.002, n_per))),
            'Low': close * (1 - np.abs(rng.normal(0, 0.002, n_per))),
            'Open': np.r_[close[0], close[:-1]], 'ATR': close * 0.01,
            'RSI': rng.uniform(30, 70, n_per), 'ATR_Pct': np.full(n_per, 1.0),
            # deliberately LEAKY feature (forward 6-bar return + noise) so the
            # meta booster has skill and the publish guards pass
            'Return_4h': (np.r_[np.log(close[6:] / close[:-6]), np.zeros(6)]
                          * 100 + rng.normal(0, 0.3, n_per)),
        }, index=idx)
        frames.append(df)
    return pd.concat(frames)


def _fake_backtest():
    m = types.ModuleType('backtest')

    def _load_artifacts(prefix):
        return object(), object(), {'trade_threshold': 0.15, 'seq_len': 10}, ['Close']

    def _predict_ticker(model, scaler, config, feature_cols, tdf,
                        lgb_model=None, q10_model=None):
        seed = int(round(float(tdf['Close'].iloc[0]) * 1000)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        ret4 = tdf['Return_4h'].values
        preds = 0.1 * np.tanh(ret4) + rng.uniform(-0.1, 0.3, len(tdf))
        return preds, None

    m._load_artifacts = _load_artifacts
    m._predict_ticker = _predict_ticker
    m._load_lgb = lambda prefix: None
    m._load_q10 = lambda prefix: None
    m._entry_window_mask = lambda idx: np.ones(len(idx), bool)
    return m


def _run(mod, base, monkeypatch, mode, dump):
    import strategy_config
    base.mkdir(parents=True, exist_ok=True)
    fake_du = types.ModuleType('data_utils')
    fake_du.load_training_data = lambda kind: _synthetic_df()
    monkeypatch.setitem(sys.modules, 'backtest', _fake_backtest())
    monkeypatch.setitem(sys.modules, 'data_utils', fake_du)
    monkeypatch.setattr(strategy_config, 'META_CALIBRATION_MODE', mode)
    monkeypatch.setattr(strategy_config, 'CALIBRATION_V2', False)
    monkeypatch.setattr(strategy_config, 'META_OOF_PRED', False)
    monkeypatch.setattr(strategy_config, 'META_REPLAY_POLICY_PARITY', False)
    monkeypatch.setattr(mod, 'BASE_DIR', base)
    monkeypatch.setattr(mod, '_notify', lambda msg: None)
    if dump is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, str(dump))
    ret = mod.train_meta('', publish=False)
    arts = {}
    for k in ('model', 'calib', 'meta'):
        p = Path(str(mod._paths('')[k]) + '.staged')
        b = p.read_bytes() if p.exists() else None
        if k == 'meta' and b is not None:
            d = json.loads(b)
            d.pop('trained_at')
            b = json.dumps(d, sort_keys=True).encode()
        arts[k] = b
    return ret, arts


@pytest.mark.parametrize('mode', ['legacy', 'purged_oof'])
def test_train_meta_env_unset_vs_set_identical(ml, tmp_path, monkeypatch, mode):
    pytest.importorskip('lightgbm')
    pytest.importorskip('sklearn')
    pytest.importorskip('joblib')

    # Unset: the dump helper must never be entered.
    def _boom(*a, **k):
        raise AssertionError('dump called with env unset')
    with monkeypatch.context() as m:
        m.setattr(ml, '_dump_meta_frame', _boom)
        ret0, arts0 = _run(ml, tmp_path / 'unset', monkeypatch, mode, None)
    assert ret0 is True
    assert all(v is not None for v in arts0.values())
    assert not (tmp_path / 'unset' / 'logs').exists()

    dump_dir = tmp_path / 'dumpdir'
    ret1, arts1 = _run(ml, tmp_path / 'set', monkeypatch, mode, dump_dir)
    assert ret1 == ret0
    for k in arts0:
        assert arts1[k] == arts0[k], f'{k} artifact differs with the dump ON'

    meta = json.loads(arts0['meta'])
    n = meta['n_trades']
    with np.load(dump_dir / 'meta_frame.npz', allow_pickle=False) as z:
        assert set(z.files) == NPZ_KEYS
        assert len(z['y']) == n and z['X'].shape == (n, len(ml.META_FEATURES))
        assert (z['split'] == 'earlystop').sum() == n - int(n * 0.8)
        assert set(z['ticker'].tolist()) <= {'AAA', 'BBB', 'CCC', 'DDD', 'EEE'}
        assert abs(float(z['y'].mean()) - meta['base_win_rate']) < 1e-4
        assert np.isclose(float(z['p_served'].min()),
                          meta['calibration']['p_min'], atol=1e-4)
        if meta['calibration']['used'] == 'purged_oof':
            assert np.isfinite(z['raw_oof']).sum() == meta['calibration']['oof_rows']
            assert np.array_equal(z['p_purged'], z['p_served'])
        else:
            assert np.array_equal(z['p_legacy'], z['p_served'])
    sc = json.loads((dump_dir / 'meta_frame.json').read_text())
    assert sc['counts']['n_rows'] == n
    assert sc['calibration']['used'] == meta['calibration']['used']
    assert sc['params']['num_leaves'] == 31


@pytest.mark.skipif(not REF_PATH, reason='SIG_R3_META_REF not set (proof-only)')
@pytest.mark.parametrize('mode', ['legacy', 'purged_oof'])
def test_unset_byte_identical_to_pre_patch(ml, tmp_path, monkeypatch, mode):
    pytest.importorskip('lightgbm')
    ref = _load(REF_PATH, 'meta_label_sig_r3_reference')
    r_ref, a_ref = _run(ref, tmp_path / 'ref', monkeypatch, mode, None)
    r_new, a_new = _run(ml, tmp_path / 'new', monkeypatch, mode, None)
    assert r_new == r_ref
    for k in a_ref:
        assert a_new[k] == a_ref[k], f'{k} artifact differs from pre-patch'
