"""Models-group review pins — meta cache contract, atomic saves, degenerate
guards, fail-closed sizing. Mac-runnable: lightgbm/joblib are stubbed via the
_read_artifacts seam or pinned by source inspection.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import meta_label
import model_lgb
from bet_sizing import afml_bet_size
from calibration import fit_calibrator

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clear_meta_cache():
    meta_label.invalidate_cache()
    yield
    meta_label.invalidate_cache()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _touch_both(paths):
    paths['model'].write_bytes(b'x')
    paths['calib'].write_bytes(b'x')


def _bump(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_mtime_ns + 10 ** 9,) * 2)


def _make_read_stub():
    """Counting stub for meta_label._read_artifacts: fresh sentinel per call."""
    calls = []

    def stub(paths):
        calls.append(1)
        return (f'booster_{len(calls)}', f'calib_{len(calls)}')

    stub.calls = calls
    return stub


# ---------------------------------------------------------------------------
# A1 — mtime-keyed cache contract
# ---------------------------------------------------------------------------

class TestMetaLoadCacheContract:

    def _wire(self, tmp_path, monkeypatch, stub=None):
        paths = {
            'model': tmp_path / 'm.txt',
            'calib': tmp_path / 'c.pkl',
            'meta': tmp_path / 'meta.json',
        }
        monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
        if stub is None:
            stub = _make_read_stub()
        monkeypatch.setattr(meta_label, '_read_artifacts', stub)
        return paths, stub

    def test_missing_files_none_and_no_read(self, tmp_path, monkeypatch):
        paths, stub = self._wire(tmp_path, monkeypatch)
        assert meta_label._load('g1') is None
        assert meta_label._load('g1') is None
        assert len(stub.calls) == 0

    def test_first_train_self_enables(self, tmp_path, monkeypatch):
        paths, stub = self._wire(tmp_path, monkeypatch)
        assert meta_label._load('g1') is None
        _touch_both(paths)
        r1 = meta_label._load('g1')
        assert r1 == ('booster_1', 'calib_1')
        assert len(stub.calls) == 1
        r2 = meta_label._load('g1')
        assert r2 == r1
        assert len(stub.calls) == 1

    def test_retrain_mtime_change_reloads(self, tmp_path, monkeypatch):
        paths, stub = self._wire(tmp_path, monkeypatch)
        _touch_both(paths)
        r1 = meta_label._load('g1')
        assert len(stub.calls) == 1
        _bump(paths['model'])
        r2 = meta_label._load('g1')
        assert r2 == ('booster_2', 'calib_2')
        assert len(stub.calls) == 2

    def test_shadow_deletion_fails_open(self, tmp_path, monkeypatch):
        paths, stub = self._wire(tmp_path, monkeypatch)
        _touch_both(paths)
        r1 = meta_label._load('g1')
        assert r1 is not None
        paths['model'].unlink()
        paths['calib'].unlink()
        r2 = meta_label._load('g1')
        assert r2 is None

    def test_load_failure_cached_until_change(self, tmp_path, monkeypatch, capsys):
        paths = {
            'model': tmp_path / 'm.txt',
            'calib': tmp_path / 'c.pkl',
            'meta': tmp_path / 'meta.json',
        }
        monkeypatch.setattr(meta_label, '_paths', lambda prefix: paths)
        _touch_both(paths)
        calls = []

        def raising_stub(paths_arg):
            calls.append(1)
            raise RuntimeError('boom')

        monkeypatch.setattr(meta_label, '_read_artifacts', raising_stub)

        r1 = meta_label._load('g1')
        assert r1 is None
        captured = capsys.readouterr()
        assert '[META] load failed' in captured.out
        assert len(calls) == 1

        r2 = meta_label._load('g1')
        assert r2 is None
        assert len(calls) == 1  # no retry storm on unchanged mtimes

        _bump(paths['model'])
        r3 = meta_label._load('g1')
        assert r3 is None
        assert len(calls) == 2

    def test_invalidate_cache_forces_reread(self, tmp_path, monkeypatch):
        paths, stub = self._wire(tmp_path, monkeypatch)
        _touch_both(paths)
        r1 = meta_label._load('g1')
        assert len(stub.calls) == 1
        meta_label.invalidate_cache()
        r2 = meta_label._load('g1')
        assert len(stub.calls) == 2


# ---------------------------------------------------------------------------
# A2/A3 — atomic saves
# ---------------------------------------------------------------------------

class TestAtomicSaves:

    def test_save_lgb_model_atomic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_lgb, '_MODEL_DIR', str(tmp_path))
        final = tmp_path / 'grp_lgb_model.txt'
        final.write_text('OLD')

        class StubModel:
            def save_model(self, path):
                Path(path).write_text('NEW')

        model_lgb.save_lgb_model(StubModel(), prefix='grp')
        assert final.read_text() == 'NEW'
        assert list(tmp_path.glob('*.tmp')) == []

    def test_save_lgb_model_failure_preserves_old(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_lgb, '_MODEL_DIR', str(tmp_path))
        final = tmp_path / 'grp_lgb_model.txt'
        final.write_text('OLD')

        class FailingStubModel:
            def save_model(self, path):
                Path(path).write_text('PARTIAL')
                raise IOError('disk full')

        with pytest.raises(IOError):
            model_lgb.save_lgb_model(FailingStubModel(), prefix='grp')
        assert final.read_text() == 'OLD'

    def test_meta_save_is_atomic_and_ordered(self):
        src = (REPO / 'meta_label.py').read_text()
        assert "booster.save_model(tmp_model)" in src
        assert "booster.save_model(str(paths['model']))" not in src
        assert src.index("os.replace(tmp_calib") < src.index("os.replace(tmp_model"), (
            "calib must be os.replace'd before model so a reader waking on "
            "the model swap always sees the paired calibrator already in place"
        )


# ---------------------------------------------------------------------------
# A4 — calibrator degenerate guard
# ---------------------------------------------------------------------------

class TestCalibratorDegenerateGuard:

    def test_constant_scores_return_none(self):
        rng = np.random.default_rng(42)
        y = (rng.random(500) < 0.5).astype(float)
        scores = np.full(500, 0.7)
        assert fit_calibrator(scores, y) is None

    def test_informative_scores_still_fit(self):
        n_half = 750
        scores = np.concatenate([np.full(n_half, 0.3), np.full(n_half, 0.7)])
        # Deterministic (no rng): 0.3 wins 10% of the time, 0.7 wins 90%.
        y_lo = np.where(np.arange(n_half) % 10 == 0, 1.0, 0.0)
        y_hi = np.where(np.arange(n_half) % 10 != 0, 1.0, 0.0)
        y = np.concatenate([y_lo, y_hi])

        cal = fit_calibrator(scores, y)
        assert cal is not None
        preds = cal.predict(np.array([0.3, 0.7]))
        assert np.all((preds >= 0.0) & (preds <= 1.0))
        assert cal.predict(np.array([0.7]))[0] >= cal.predict(np.array([0.3]))[0]


# ---------------------------------------------------------------------------
# A5 — afml_bet_size fail-closed
# ---------------------------------------------------------------------------

class TestAfmlBetSizeFailClosed:

    def test_nonfinite_scalar_sizes_zero(self):
        for v in (np.nan, np.inf, -np.inf):
            r = afml_bet_size(v)
            assert r == 0.0
            assert isinstance(r, float)
            r2 = afml_bet_size(v, step=0.05)
            assert r2 == 0.0

    def test_nonfinite_vector_element(self):
        m = afml_bet_size(np.array([0.6, np.nan, 0.4]))
        assert np.all(np.isfinite(m))
        assert m[1] == 0.0
        assert m[0] > 0 > m[2]

    def test_base_rate_centering_preserved(self):
        assert afml_bet_size(0.55, base_rate=0.55) == 0.0


# ---------------------------------------------------------------------------
# A7 — vectorized hour extraction
# ---------------------------------------------------------------------------

class TestHourVectorization:

    def _frame(self, n=200):
        idx = pd.date_range('2025-01-01', periods=n, freq='h', tz='UTC')
        df = pd.DataFrame({
            'Close': np.linspace(100.0, 110.0, n),
            'RSI': np.linspace(30.0, 70.0, n),
        }, index=idx)
        return df, idx

    def test_hour_matrix_bit_identical_datetimeindex(self):
        df, idx = self._frame()
        preds = np.zeros(len(idx))
        mat = meta_label.build_feature_matrix(df, preds)
        expected = np.array([meta_label._hour_encode(t.hour) for t in idx])
        assert np.array_equal(mat[:, 10], expected[:, 0])
        assert np.array_equal(mat[:, 11], expected[:, 1])

    def test_hour_fallback_object_index(self):
        df, idx = self._frame()
        preds = np.zeros(len(idx))
        mat_dt = meta_label.build_feature_matrix(df, preds)

        obj_idx = pd.Index(list(idx.to_pydatetime()), dtype=object)
        assert not hasattr(obj_idx, 'hour')  # confirms the fallback path fires
        df_obj = df.copy()
        df_obj.index = obj_idx
        mat_obj = meta_label.build_feature_matrix(df_obj, preds)

        assert np.array_equal(mat_dt[:, 10], mat_obj[:, 10])
        assert np.array_equal(mat_dt[:, 11], mat_obj[:, 11])
