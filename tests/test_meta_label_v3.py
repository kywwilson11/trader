"""Panel v3 adjudicated spec — meta_label.py. Mac-runnable: numpy/pandas/stdlib
only; heavy deps stubbed via the _paths/_read_artifacts seams (same pattern as
tests/test_grp_models.py).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_label
from meta_label import (META_FEATURES, META_VETO_PROB, meta_size_mult,
                        meta_probability_live, predict_meta_array,
                        build_feature_matrix, _gen_meta_rows, _meta_payload)
from strategy_config import policy_for

REPO = Path(__file__).resolve().parent.parent
SRC = (REPO / 'meta_label.py').read_text()

THRESHOLD = 0.15  # config default in train_meta


def _synthetic_tdf(n=400, with_spread=False, seed=7):
    """Harvest-shaped hourly frame + primary preds with entries to replay.
    (Copied from tests/test_review_b14.py.)"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    df = pd.DataFrame({
        'Close': close, 'High': high, 'Low': low, 'Open': open_,
        'ATR': close * 0.01,
        'RSI': rng.uniform(30, 70, n),
        'ATR_Pct': np.full(n, 1.0),
    }, index=idx)
    if with_spread:
        df['Eff_Spread_Pct'] = rng.uniform(0.02, 0.2, n)
    preds = rng.uniform(-0.1, 0.3, n)  # plenty above 0.5x threshold
    return df, preds


def _touch_both(paths):
    """(Copied from tests/test_grp_models.py.)"""
    paths['model'].write_bytes(b'x')
    paths['calib'].write_bytes(b'x')


def _bump(path):
    """(Copied from tests/test_grp_models.py.)"""
    st = os.stat(path)
    os.utime(path, ns=(st.st_mtime_ns + 10 ** 9,) * 2)


def _mk_paths(tmp_path, tag):
    return {
        'model': tmp_path / f'{tag}_m.txt',
        'calib': tmp_path / f'{tag}_c.pkl',
        'meta': tmp_path / f'{tag}_meta.json',
    }


@pytest.fixture(autouse=True)
def _clear_meta_cache():
    # The _loaded cache and the warn-once flags are process-global module
    # state, not per-test — every test starts and ends on a clean slate.
    meta_label.invalidate_cache()
    yield
    meta_label.invalidate_cache()


# ---------------------------------------------------------------------------
# 1/2: meta_size_mult guards + the base_loop clamp behavior it exists for
# ---------------------------------------------------------------------------

def test_meta_size_mult_guards():
    assert meta_size_mult(float('nan')) == 1.0
    assert meta_size_mult(float('inf')) == 1.0
    assert meta_size_mult(None) == 1.0
    # finite pins unchanged
    assert meta_size_mult(0.0) == 0.6
    assert meta_size_mult(0.3) == 0.6
    assert meta_size_mult(0.5) == 1.0
    assert meta_size_mult(0.65) == pytest.approx(1.3)
    assert meta_size_mult(1.0) == 1.3


def test_nan_tilt_chain_is_why():
    # Documentation pin: the base_loop clamp behavior meta_size_mult's and
    # meta_probability_live's non-finite guards exist to prevent.
    TILT_MAX = 1.30
    assert max(0.1, min(TILT_MAX, float('nan') * 1.0)) == 1.30


# ---------------------------------------------------------------------------
# 3: meta_probability_live serves None (not NaN) on a non-finite calibrated p
# ---------------------------------------------------------------------------

class _FixedBooster:
    def __init__(self, val):
        self._val = val

    def predict(self, x):
        return np.array([self._val])


class _FixedCalib:
    def __init__(self, val):
        self._val = val

    def predict(self, raw):
        return np.array([self._val])


def test_meta_probability_live_nonfinite_returns_none(tmp_path, monkeypatch, capsys):
    paths = _mk_paths(tmp_path, 'v3a')
    monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
    _touch_both(paths)
    monkeypatch.setattr(meta_label, '_read_artifacts',
                        lambda p: (_FixedBooster(0.5), _FixedCalib(float('nan'))))
    monkeypatch.setattr(meta_label, '_nonfinite_p_warned', False)

    result = meta_label.meta_probability_live('v3a', {}, 0.5)
    assert result is None
    out = capsys.readouterr().out
    assert 'non-finite calibrated probability' in out

    # second call: same artifact generation (cache hit) -> warn-once, silent
    result2 = meta_label.meta_probability_live('v3a', {}, 0.5)
    assert result2 is None
    out2 = capsys.readouterr().out
    assert 'non-finite calibrated probability' not in out2

    # swap in a finite calibrator via a fresh artifact generation (mtime bump)
    _bump(paths['calib'])
    monkeypatch.setattr(meta_label, '_read_artifacts',
                        lambda p: (_FixedBooster(0.5), _FixedCalib(0.42)))
    result3 = meta_label.meta_probability_live('v3a', {}, 0.5)
    assert result3 == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# 4: predict_meta_array fails loud (backtest.py swallows the exception silently)
# ---------------------------------------------------------------------------

class _BoomBooster:
    def predict(self, x):
        raise RuntimeError('boom')


def test_predict_meta_array_fail_loud(tmp_path, monkeypatch, capsys):
    paths = _mk_paths(tmp_path, 'v3b')
    monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
    _touch_both(paths)
    monkeypatch.setattr(meta_label, '_read_artifacts',
                        lambda p: (_BoomBooster(), object()))

    tdf, preds = _synthetic_tdf()
    result = meta_label.predict_meta_array('v3b', tdf, preds)
    assert result is None
    out = capsys.readouterr().out
    assert '[META] scoring failed' in out
    assert 'boom' in out


# ---------------------------------------------------------------------------
# 5: build_feature_matrix length guard
# ---------------------------------------------------------------------------

def test_build_feature_matrix_length_guard():
    tdf, preds = _synthetic_tdf()
    with pytest.raises(ValueError, match='preds'):
        build_feature_matrix(tdf, preds[:-1])

    mat = build_feature_matrix(tdf, preds)
    assert mat.shape == (len(tdf), len(META_FEATURES))
    assert META_FEATURES[0] == 'pred'
    assert np.array_equal(mat[:, 0], preds)


# ---------------------------------------------------------------------------
# 6: torn-pair re-stat retry
# ---------------------------------------------------------------------------

def test_load_torn_pair_reread(tmp_path, monkeypatch):
    paths = _mk_paths(tmp_path, 'v3c')
    monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
    _touch_both(paths)

    calls = []

    def stub(p):
        calls.append(1)
        if len(calls) == 1:
            # simulate landing between the publisher's two os.replace calls
            _bump(p['calib'])
            return ('sentinel_1_booster', 'sentinel_1_calib')
        return ('sentinel_2_booster', 'sentinel_2_calib')

    monkeypatch.setattr(meta_label, '_read_artifacts', stub)

    result = meta_label._load('v3c')
    assert result == ('sentinel_2_booster', 'sentinel_2_calib')
    assert len(calls) == 2

    # a further _load call under the now-stable key is served from cache
    result2 = meta_label._load('v3c')
    assert result2 == ('sentinel_2_booster', 'sentinel_2_calib')
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# 7: feature-schema drift tripwire
# ---------------------------------------------------------------------------

def test_feature_schema_drift_warns_but_serves(tmp_path, monkeypatch, capsys):
    paths = _mk_paths(tmp_path, 'v3d')
    monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
    _touch_both(paths)
    paths['meta'].write_text(json.dumps({'features': list(reversed(META_FEATURES))}))
    monkeypatch.setattr(meta_label, '_read_artifacts', lambda p: ('b', 'c'))

    result = meta_label._load('v3d')
    assert result == ('b', 'c')
    out = capsys.readouterr().out
    assert 'feature schema drift' in out

    # matching feature list -> silent
    meta_label.invalidate_cache()
    paths['meta'].write_text(json.dumps({'features': list(META_FEATURES)}))
    result2 = meta_label._load('v3d')
    assert result2 == ('b', 'c')
    out2 = capsys.readouterr().out
    assert 'feature schema drift' not in out2


# ---------------------------------------------------------------------------
# 8: warn-once flags re-arm on a fresh load AND on invalidate_cache
# ---------------------------------------------------------------------------

def test_warn_flags_rearm(tmp_path, monkeypatch):
    paths = _mk_paths(tmp_path, 'v3e')
    monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
    _touch_both(paths)
    monkeypatch.setattr(meta_label, '_read_artifacts', lambda p: ('b', 'c'))

    monkeypatch.setattr(meta_label, '_calib_fallback_warned', True)
    monkeypatch.setattr(meta_label, '_nonfinite_p_warned', True)
    result = meta_label._load('v3e')
    assert result == ('b', 'c')
    assert meta_label._calib_fallback_warned is False
    assert meta_label._nonfinite_p_warned is False

    monkeypatch.setattr(meta_label, '_calib_fallback_warned', True)
    monkeypatch.setattr(meta_label, '_nonfinite_p_warned', True)
    meta_label.invalidate_cache()
    assert meta_label._calib_fallback_warned is False
    assert meta_label._nonfinite_p_warned is False


# ---------------------------------------------------------------------------
# 9: CLI --prefix rejects a typo before any heavy import
# ---------------------------------------------------------------------------

def test_prefix_cli_rejects_typo():
    result = subprocess.run(
        [sys.executable, str(REPO / 'meta_label.py'), '--prefix', 'stocks'],
        capture_output=True, text=True, cwd=str(REPO))
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr


# ---------------------------------------------------------------------------
# 10: replayed rows are independent copies; labels agree with net sign
# ---------------------------------------------------------------------------

def test_replay_rows_are_copies_and_labels_consistent():
    tdf, preds = _synthetic_tdf()
    rows, labels, nets, times, exit_times = _gen_meta_rows(
        tdf, preds, 'crypto', THRESHOLD, policy_for('crypto'))
    assert len(rows) > 0
    assert all(r.base is None for r in rows)
    assert all(bool(lb) == (v > 0) for lb, v in zip(labels, nets))


# ---------------------------------------------------------------------------
# 11: groupby(sort=False) order == unique() order, groups == boolean-mask slices
# ---------------------------------------------------------------------------

def test_groupby_order_matches_unique():
    idx = pd.date_range('2025-01-01', periods=12, freq='h', tz='UTC')
    tickers = ['BBB', 'AAA', 'BBB', 'CCC', 'AAA', 'BBB',
              'CCC', 'AAA', 'BBB', 'CCC', 'AAA', 'BBB']
    df = pd.DataFrame({'Ticker': tickers, 'Close': np.arange(12, dtype=float)},
                      index=idx)

    unique_order = list(df['Ticker'].unique())
    grouped = list(df.groupby('Ticker', sort=False))
    groupby_order = [k for k, _ in grouped]
    assert unique_order == groupby_order

    for k, gdf in grouped:
        expected = df[df['Ticker'] == k]
        # check_freq=False: a boolean-mask slice can coincidentally infer a
        # DatetimeIndex freq (e.g. an evenly-spaced ticker pattern) that the
        # groupby path does not — an irrelevant metadata difference, not a
        # values/order difference (which is what this test is pinning).
        pd.testing.assert_frame_equal(gdf, expected, check_freq=False)


# ---------------------------------------------------------------------------
# 12: _meta_payload schema — legacy keys preserved, additive new keys, JSON-safe
# ---------------------------------------------------------------------------

def test_meta_payload_schema():
    common = dict(
        n_trades=500, base_rate=0.55,
        holdout_cutoff_utc='2026-01-01T00:00:00+00:00',
        n_rows_total=10000, n_rows_pre_cutoff=8800, n_tickers_used=12,
        skipped_tickers=[('XYZ', 'too_short')], zero_filled_features=['Hurst'],
        net_summary={'mean_pct': 0.1, 'median_pct': 0.05,
                    'p10_pct': -1.0, 'p90_pct': 1.0},
        calibration={'used': 'legacy'}, primary=None,
        trained_at='2026-07-25T00:00:00+00:00',
    )
    payload_none = _meta_payload(val_auc=None, **common)
    payload_val = _meta_payload(val_auc=0.6123456, **common)

    legacy_keys = {'features', 'n_trades', 'base_win_rate', 'val_auc',
                  'threshold_fraction'}
    assert legacy_keys <= payload_none.keys()
    assert payload_none['features'] == META_FEATURES
    assert payload_none['val_auc'] is None
    assert payload_val['val_auc'] == pytest.approx(0.6123)

    new_keys = {'trained_at', 'holdout_cutoff_utc', 'n_rows_total',
               'n_rows_pre_cutoff', 'n_tickers_used', 'n_tickers_skipped',
               'skipped_tickers', 'zero_filled_features', 'net_summary',
               'calibration', 'primary'}
    assert new_keys <= payload_none.keys()

    json.dumps(payload_none)
    json.dumps(payload_val)


# ---------------------------------------------------------------------------
# 13: HOLDOUT_FRACTION parity with hypersearch_v2.py
# ---------------------------------------------------------------------------

def test_holdout_fraction_parity():
    pattern = re.compile(r'HOLDOUT_FRACTION\s*=\s*([0-9.]+)')
    m1 = pattern.search(SRC)
    hy_src = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
    m2 = pattern.search(hy_src)
    assert m1 and m2
    assert float(m1.group(1)) == float(m2.group(1))


# ---------------------------------------------------------------------------
# 14: source pins — publish order, no unguarded best_score subscript,
#     no .view('int64'), PID-suffixed tmp names, backup-failure visibility
# ---------------------------------------------------------------------------

def test_source_pins():
    assert (SRC.index("os.replace(tmp_meta") < SRC.index("os.replace(tmp_calib")
            < SRC.index("os.replace(tmp_model"))
    assert "best_score['valid_0']" not in SRC
    assert ".view('int64')" not in SRC
    assert "os.getpid()" in SRC
    assert "backup failed" in SRC
