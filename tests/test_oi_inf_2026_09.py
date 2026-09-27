"""FIX-R1 (2026-09): OI_Chg_24h +inf rows + the non-finite parity guard.

Root cause: the Binance metrics archive carries glitch prints with
sum_open_interest_value == 0.0; the offline 24h pct_change divided by that
zero 24 rows later (+inf, 74 rows in the 2026-09 crypto harvest), served
-100% at the glitch hour itself and dragged OI_Z's trailing mean/std.

Mac-safe: the OI tests need only numpy/pandas(+pyarrow for the archive
fixture — skipped without it); the sanitizer twins are exercised from their
SOURCE via ast (no torch/optuna/sklearn import); the two end-to-end checks
importorskip their heavy deps.
"""

import ast
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import oi_archive  # noqa: E402  (light: funding/funding_archive/log_config)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_twin(relpath):
    """Exec `_sanitize_nonfinite_features` + NONFINITE_FILL from a module's
    source without importing the module (keeps torch/optuna out)."""
    src = (ROOT / relpath).read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef)
                and n.name == '_sanitize_nonfinite_features')
            or (isinstance(n, ast.Assign)
                and any(getattr(t, 'id', None) == 'NONFINITE_FILL'
                        for t in n.targets))]
    assert len(keep) == 2, f'{relpath}: sanitizer/NONFINITE_FILL not found'
    ns = {'np': np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), relpath, 'exec'), ns)
    return ns['_sanitize_nonfinite_features'], ns['NONFINITE_FILL']


TWINS = {
    'predict_now': _load_twin('predict_now.py'),
    'hypersearch_v2': _load_twin('scripts/hypersearch_v2.py'),
}


@pytest.fixture
def glitch_archive(tmp_path, monkeypatch):
    """20-day hourly archive with a zero-notional glitch print at hour 300
    (the exact shape found in oi_archive.parquet)."""
    pytest.importorskip('pyarrow')
    monkeypatch.setattr(oi_archive, 'ARCHIVE_FILE', tmp_path / 'oi.parquet')
    idx = pd.date_range('2026-05-01', periods=24 * 20, freq='h', tz='UTC')
    rng = np.random.default_rng(0)
    oi_val = 7e9 * (1 + rng.normal(0, 0.002, len(idx))).cumprod()
    oi_val[300] = 0.0
    pd.DataFrame({'symbol': 'BTC/USD', 'ts': idx, 'oi': oi_val / 7e4,
                  'oi_value': oi_val, 'tt_ls_ratio': 1.4,
                  'taker_ratio': 1.3}).to_parquet(tmp_path / 'oi.parquet')
    return idx, oi_val


# ---------------------------------------------------------------------------
# (a) offline OI feature
# ---------------------------------------------------------------------------

def test_offline_zero_denominator_is_nan_not_inf(glitch_archive):
    idx, oi_val = glitch_archive
    f = oi_archive.oi_features_for_index('BTC/USD', idx)
    chg, z = f['OI_Chg_24h'], f['OI_Z']
    assert not np.isinf(chg).any() and not np.isinf(z).any()
    # glitch hour: change ffills over the prior print (no -100% spike)
    assert np.isfinite(chg[300]) and abs(chg[300]) < 5
    # 24 rows later the old code divided by the zero print -> +inf
    assert np.isfinite(chg[324]) and abs(chg[324]) < 5
    # z is NaN (-> the harvest's neutral 0.0) at the glitch hour, and the
    # zero never enters the trailing window
    assert np.isnan(z[300])
    assert np.nanmin(z) > -6
    # rows away from the glitch are unchanged vs the pre-fix formula
    s = pd.Series(oi_val, index=idx)
    ref = (s.pct_change(24, fill_method=None) * 100).values
    far = np.r_[24:300, 325:len(idx)]
    np.testing.assert_array_equal(chg[far], ref[far])


def test_offline_pct_change_backstop_on_nan_denominator(glitch_archive,
                                                        monkeypatch):
    """First print of a name / a NaN reference -> NaN, never inf."""
    idx, oi_val = glitch_archive
    s = pd.Series(oi_val, index=idx)
    s.iloc[:10] = np.nan
    monkeypatch.setattr(oi_archive, 'get_oi_series', lambda sym: s)
    f = oi_archive.oi_features_for_index('BTC/USD', idx)
    assert not np.isinf(f['OI_Chg_24h']).any()
    assert np.isnan(f['OI_Chg_24h'][:34]).all()


def test_offline_harvest_fill_maps_nan_to_neutral_zero():
    """The value the harvest writes for the (now) NaN rows is 0.0 — the
    neutral the sanitizer and live serving also use."""
    src = (ROOT / 'scripts' / 'harvest_crypto_data.py').read_text()
    assert "'OI_Chg_24h', 'OI_Z'" in src
    assert 'df[col] = df[col].fillna(0.0)' in src
    for _, (fn, fill) in TWINS.items():
        assert fill == 0.0


# ---------------------------------------------------------------------------
# (b) live path
# ---------------------------------------------------------------------------

@pytest.fixture
def live_env(tmp_path, monkeypatch):
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE',
                        tmp_path / 'oi_history.json')
    oi_archive._live_cache.clear()
    oi_archive._fail_cache.clear()
    return tmp_path


def test_live_zero_reference_serves_neutral_zero(live_env, monkeypatch):
    now = time.time()
    (live_env / 'oi_history.json').write_text(json.dumps(
        {'BTC/USD': [[now - 86400, 0.0]]}))
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    out = oi_archive.live_oi_features('BTC/USD')
    assert out['OI_Chg_24h'] == 0.0 and math.isfinite(out['OI_Z'])


def test_live_skips_bad_history_sample_for_valid_neighbour(live_env,
                                                           monkeypatch):
    now = time.time()
    (live_env / 'oi_history.json').write_text(json.dumps(
        {'BTC/USD': [[now - 86400, 0.0], [now - 86400 + 3600, 4000.0]]}))
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    out = oi_archive.live_oi_features('BTC/USD')
    assert out['OI_Chg_24h'] == pytest.approx(25.0)


def test_live_z_ignores_persisted_nonfinite_samples(live_env, monkeypatch):
    now = time.time()
    rng = np.random.default_rng(1)
    hist = [[now - (200 - i) * 3600, float(v)]
            for i, v in enumerate(4000 + rng.normal(0, 40, 200))]
    hist[50][1] = float('nan')
    hist[60][1] = 0.0
    # json writes/reads NaN tokens by default, as a pre-fix build would have
    (live_env / 'oi_history.json').write_text(json.dumps({'BTC/USD': hist}))
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 4400.0)
    out = oi_archive.live_oi_features('BTC/USD')
    assert math.isfinite(out['OI_Z']) and out['OI_Z'] > 3


@pytest.mark.parametrize('raw', ['0', '-1', 'NaN', 'inf'])
def test_live_fetch_rejects_unusable_prints(live_env, monkeypatch, raw):
    class _Resp:
        def read(self):
            return json.dumps({'data': [{'oiCcy': raw}]}).encode()
    monkeypatch.setattr(oi_archive.urllib.request, 'urlopen',
                        lambda req, timeout=10: _Resp())
    assert oi_archive._fetch_okx_oi('BTC/USD') is None
    # nothing persisted -> no history poisoning
    assert oi_archive.live_oi_features('BTC/USD') is None
    assert not (live_env / 'oi_history.json').exists()


def test_live_fetch_accepts_positive_print(live_env, monkeypatch):
    class _Resp:
        def read(self):
            return json.dumps({'data': [{'oiCcy': '12345.6'}]}).encode()
    monkeypatch.setattr(oi_archive.urllib.request, 'urlopen',
                        lambda req, timeout=10: _Resp())
    assert oi_archive._fetch_okx_oi('BTC/USD') == pytest.approx(12345.6)


# ---------------------------------------------------------------------------
# (c) sanitizer twins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(TWINS))
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_sanitizer_noop_on_finite_is_bit_identical(name, dtype):
    fn, _ = TWINS[name]
    X = np.random.default_rng(3).normal(size=(64, 7)).astype(dtype)
    before = X.tobytes()
    out, n_bad, n_inf, cols = fn(X)
    assert out is X and out.tobytes() == before
    assert (n_bad, n_inf, cols) == (0, 0, [])
    out2, *_ = fn(X, copy=False)
    assert out2 is X and out2.tobytes() == before


@pytest.mark.parametrize('name', sorted(TWINS))
def test_sanitizer_int_matrix_untouched(name):
    fn, _ = TWINS[name]
    X = np.arange(12).reshape(3, 4)
    out, n_bad, n_inf, cols = fn(X)
    assert out is X and n_bad == 0


@pytest.mark.parametrize('name', sorted(TWINS))
def test_sanitizer_replaces_and_counts(name):
    fn, fill = TWINS[name]
    X = np.ones((5, 4), dtype=np.float32)
    X[1, 2] = np.inf
    X[3, 2] = -np.inf
    X[4, 0] = np.nan
    snap = X.copy()
    out, n_bad, n_inf, cols = fn(X)
    assert (n_bad, n_inf, cols) == (3, 2, [0, 2])
    assert np.isfinite(out).all()
    assert out[1, 2] == fill and out[3, 2] == fill and out[4, 0] == fill
    np.testing.assert_array_equal(out[np.isfinite(snap)],
                                  snap[np.isfinite(snap)])
    # copy=True never mutates the caller's array (df.values may be a view)
    np.testing.assert_array_equal(np.isnan(X), np.isnan(snap))
    # copy=False sanitizes in place (trainer: no extra full-matrix copy)
    out3, *_ = fn(X, copy=False)
    assert out3 is X and np.isfinite(X).all()


def test_twins_behave_identically():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 6))
    X[rng.random(X.shape) < 0.1] = np.inf
    X[rng.random(X.shape) < 0.05] = np.nan
    a = TWINS['predict_now'][0](X)
    b = TWINS['hypersearch_v2'][0](X)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1:] == b[1:]


def test_guards_sit_right_before_the_scaler():
    pn = (ROOT / 'predict_now.py').read_text()
    i = pn.index('_sanitize_nonfinite_features(current_features)')
    assert i < pn.index(
        'sequence = scaler_X.transform(current_features[-seq_len:])')
    hs = (ROOT / 'scripts' / 'hypersearch_v2.py').read_text()
    j = hs.index('all_features = np.vstack(all_features_list)')
    k = hs.index('_sanitize_nonfinite_features(\n        all_features')
    assert j < k < hs.index('scaler.fit(self._all_features[train_indices])')


# ---------------------------------------------------------------------------
# end-to-end (heavy deps — skipped on the dev Mac)
# ---------------------------------------------------------------------------

def _run_predict(monkeypatch, inject):
    pytest.importorskip('torch')
    pytest.importorskip('joblib')
    import torch
    import predict_now

    idx = pd.date_range('2026-06-01', periods=40, freq='h', tz='UTC')
    bars = pd.DataFrame({c: np.full(40, 100.0) for c in
                         ('Open', 'High', 'Low', 'Close', 'Volume')},
                        index=idx)
    feat = bars.copy()
    feat['RSI'] = np.linspace(40, 60, 40)
    feat['OI_Chg_24h'] = 1.5
    if inject:
        feat.iloc[-2, feat.columns.get_loc('OI_Chg_24h')] = np.inf
    monkeypatch.setattr(predict_now, 'fetch_bars_yfinance', lambda s: bars)
    monkeypatch.setattr(predict_now, 'compute_features',
                        lambda df, btc_close=None: feat)
    seen = []

    class Scaler:
        def transform(self, x):
            seen.append(np.array(x, copy=True))
            return np.asarray(x, dtype=np.float32)

    class Model:
        def __call__(self, t):
            return torch.tensor([0.42])

    cfg = {'seq_len': 8, 'trade_threshold': 0.2, 'prefix': 'zz_nope'}
    pred = predict_now.get_live_prediction(
        'BTC-USD', Model(), Scaler(), cfg, ['Close', 'RSI', 'OI_Chg_24h'],
        asset_type='crypto')
    return pred, seen[0]


def test_predict_now_guard_end_to_end(monkeypatch, capsys):
    pred0, clean = _run_predict(monkeypatch, inject=False)
    assert '[FEATURES] BTC-USD' not in capsys.readouterr().out
    pred1, fixed = _run_predict(monkeypatch, inject=True)
    out = capsys.readouterr().out
    assert "1 non-finite input value(s) (1 inf) in ['OI_Chg_24h']" in out
    assert np.isfinite(fixed).all() and fixed[-2, 2] == 0.0
    np.testing.assert_array_equal(np.delete(fixed, -2, axis=0),
                                  np.delete(clean, -2, axis=0))
    assert pred0 is not None and pred1 is not None


def test_hypersearch_load_data_guard_end_to_end(monkeypatch, capsys):
    for mod in ('torch', 'optuna', 'sklearn', 'joblib'):
        pytest.importorskip(mod)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import data_utils
    import indicator_config
    import hypersearch_v2 as hs

    idx = pd.date_range('2026-01-01', periods=300, freq='h', tz='UTC',
                        name='Datetime')
    rng = np.random.default_rng(5)

    def _frame():
        parts = []
        for t in ('BTC-USD', 'ETH-USD'):
            d = pd.DataFrame({'Ticker': t,
                              'RSI': rng.normal(50, 5, len(idx)),
                              'OI_Chg_24h': rng.normal(0, 2, len(idx))},
                             index=idx)
            for fb in hs.FORWARD_BARS:
                d[f'Target_Return_{fb}'] = rng.normal(0, 1, len(idx))
            d['Target_Return'] = d[f'Target_Return_{hs.FORWARD_BARS[0]}']
            parts.append(d)
        return pd.concat(parts)

    monkeypatch.setattr(indicator_config, 'get_preset_features',
                        lambda name: None)
    monkeypatch.setattr(indicator_config, 'load_indicator_config',
                        lambda: {'preset': 'test'})
    base = _frame()
    monkeypatch.setattr(data_utils, 'load_training_data',
                        lambda prefix: base.copy())
    X0 = hs.load_data('training_data.csv', max_rows=10**9)[0]
    assert '[SANITIZE]' not in capsys.readouterr().out
    np.testing.assert_array_equal(
        X0, np.vstack([base[base.Ticker == t][['RSI', 'OI_Chg_24h']]
                       .values.astype(np.float32)
                       for t in base.Ticker.unique()]))

    bad = base.copy()
    bad.iloc[7, bad.columns.get_loc('OI_Chg_24h')] = np.inf
    monkeypatch.setattr(data_utils, 'load_training_data',
                        lambda prefix: bad.copy())
    X1 = hs.load_data('training_data.csv', max_rows=10**9)[0]
    out = capsys.readouterr().out
    assert "[SANITIZE] 1 non-finite feature value(s) (1 inf) in " \
           "['OI_Chg_24h']" in out
    assert np.isfinite(X1).all() and X1[7, 1] == 0.0
    mask = np.ones(X1.shape, bool)
    mask[7, 1] = False
    np.testing.assert_array_equal(X1[mask], X0[mask])
