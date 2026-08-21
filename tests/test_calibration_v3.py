"""Adjudicated panel-review hardening tests for calibration.py (2026-07).

Pure numpy — no heavy deps, no importorskip needed.
"""
import numpy as np
import pytest

from calibration import (
    _pava,
    IsotonicCalibrator,
    SigmoidCalibrator,
    fit_calibrator,
    purged_kfold_indices,
    crossfit_oof_predict,
    brier,
    reliability_curve,
    expected_calibration_error,
    compare_calibrations,
)


def test_brier_masks_nonfinite():
    assert brier([1.0, np.nan], [1.0, 0.0]) == 0.0


def test_brier_empty_after_mask_returns_nan():
    assert np.isnan(brier([np.nan], [1.0]))


def test_brier_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        brier([0.5], [0.0, 1.0, 1.0, 0.0])


def test_compare_calibrations_one_nan_does_not_flip_verdict():
    rng = np.random.default_rng(12)
    n = 4000
    p_true = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_purged = np.clip(p_true + rng.normal(0, 0.05, n), 0, 1)
    p_legacy = np.clip((p_true - 0.5) * 1.8 + 0.5, 0, 1)

    assert 'safe to flip' in compare_calibrations(p_legacy, p_purged, y)['verdict']

    p_purged[7] = np.nan
    rep = compare_calibrations(p_legacy, p_purged, y)
    assert 'safe to flip' in rep['verdict']
    assert rep['n'] == n - 1
    assert rep['n_dropped'] == 1
    assert np.isfinite(rep['brier_purged'])


def test_compare_calibrations_delta_fields():
    rng = np.random.default_rng(12)
    n = 4000
    p_true = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_purged = np.clip(p_true + rng.normal(0, 0.05, n), 0, 1)
    p_legacy = np.clip((p_true - 0.5) * 1.8 + 0.5, 0, 1)

    rep = compare_calibrations(p_legacy, p_purged, y)
    assert rep['brier_delta'] > 0
    assert rep['brier_delta_se'] > 0
    assert rep['brier_delta_t'] > 0


def test_compare_calibrations_reports_tie():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.1, 0.9, 500)
    y = (rng.uniform(size=500) < p).astype(float)
    rep = compare_calibrations(p, p.copy(), y)
    assert rep['tied'] is True
    assert 'do NOT flip' in rep['verdict']
    assert 'safe to flip' not in rep['verdict']


def test_compare_calibrations_empty_and_all_nan():
    rep = compare_calibrations([np.nan, np.nan], [0.5, 0.5], [1.0, 0.0])
    assert rep['n'] == 0 and rep['n_dropped'] == 2
    assert 'nothing to compare' in rep['verdict']

    rep2 = compare_calibrations([], [], [])
    assert rep2['n'] == 0 and rep2['n_dropped'] == 0


def test_compare_calibrations_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_calibrations([0.5, 0.5], [0.5], [1.0, 0.0])


def test_ece_and_reliability_reject_bad_bins():
    for bad in (0, -3):
        with pytest.raises(ValueError):
            expected_calibration_error([0.2, 0.8], [0.0, 1.0], bad)
        with pytest.raises(ValueError):
            reliability_curve([0.2, 0.8], [0.0, 1.0], bad)
    assert isinstance(expected_calibration_error([0.2, 0.8], [0.0, 1.0], 1), float)


def test_purged_kfold_input_validation():
    with pytest.raises(ValueError):  # unsorted entry
        purged_kfold_indices([3., 1., 2.], [4., 2., 3.])
    with pytest.raises(ValueError):  # non-finite
        entry = np.arange(10.)
        exit_ = entry + 1
        exit_[3] = np.nan
        purged_kfold_indices(entry, exit_)
    with pytest.raises(ValueError):  # negative embargo
        purged_kfold_indices(np.arange(10.), np.arange(10.) + 1, embargo=-1.0)
    with pytest.raises(ValueError):  # inverted span (exit_ < entry)
        purged_kfold_indices([0., 1., 2.], [0.5, 0.5, 2.5])
    with pytest.raises(ValueError):  # shape mismatch
        purged_kfold_indices(np.arange(5.), np.arange(4.))


def test_purged_kfold_clean_path_pinned():
    entry = np.arange(60.)
    exit_ = entry + 3.0
    folds = purged_kfold_indices(entry, exit_, k=5, embargo=0.0)
    assert len(folds) == 5
    # fold 0: test rows 0-11, purged rows 12-14, train rows 15-59
    assert len(folds[0][0]) == 45


def test_crossfit_rejects_length_mismatch():
    stub = lambda a, b, c: 1 / (1 + np.exp(-c[:, 0]))  # noqa: E731
    X = np.random.default_rng(0).normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float)
    with pytest.raises(ValueError):
        crossfit_oof_predict(stub, X, y, np.arange(30.), np.arange(30.) + 1, k=5)


def test_crossfit_rejects_scalar_and_wrong_shape_scores():
    X = np.random.default_rng(0).normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(float)
    entry = np.arange(50.)
    exit_ = entry + 1
    with pytest.raises(ValueError):
        crossfit_oof_predict(lambda a, b, c: 0.42, X, y, entry, exit_, k=5)
    with pytest.raises(ValueError):
        crossfit_oof_predict(lambda a, b, c: np.zeros(3), X, y, entry, exit_, k=5)


def test_crossfit_prints_coverage(capsys):
    # Reuses the n=300 fixture of test_calibration.py::test_crossfit_assigns_every_row_a_score.
    rng = np.random.default_rng(5)
    n = 300
    X = rng.normal(size=(n, 2))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    entry = np.arange(n, dtype=float)
    exit_ = entry + 1.0

    def stub(Xtr, ytr, Xte):
        return 1 / (1 + np.exp(-Xte[:, 0]))

    crossfit_oof_predict(stub, X, y, entry, exit_, k=5)
    out = capsys.readouterr().out
    assert '[CALIB] cross-fit: 5/5 folds usable, 300/300 rows scored' in out


def test_fit_calibrator_rejects_nonbinary_y():
    scores = np.linspace(0.1, 0.9, 50)
    y = np.where(np.arange(50) % 2 == 0, 1.0, -1.0)
    with pytest.raises(ValueError, match='binary'):
        fit_calibrator(scores, y)


def test_fit_calibrator_metadata():
    rng = np.random.default_rng(4)
    scores = rng.uniform(size=205)
    y = (rng.uniform(size=205) < 0.5).astype(float)
    scores[:5] = np.nan
    cal = fit_calibrator(scores, y)
    assert cal.method_ == 'sigmoid'
    assert cal.n_fit_ == 200
    assert cal.n_dropped_ == 5
    assert 0.0 <= cal.base_rate_ <= 1.0


def test_fit_calibrator_decline_prints_reason(capsys):
    assert fit_calibrator([0.1, 0.2], [1.0, 0.0]) is None
    assert '[CALIB] declined' in capsys.readouterr().out


def test_fit_calibrator_warns_on_separable_sigmoid(capsys):
    x = np.concatenate([np.linspace(0.0, 0.45, 100), np.linspace(0.55, 1.0, 100)])
    y = np.concatenate([np.zeros(100), np.ones(100)])
    cal = fit_calibrator(x, y)
    assert isinstance(cal, SigmoidCalibrator)
    assert '[CALIB] WARNING' in capsys.readouterr().out


def test_sigmoid_records_convergence():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-(0.7 + 1.5 * x)))).astype(float)
    cal = SigmoidCalibrator().fit(x, y)
    assert cal.converged_ is True
    assert cal.n_iter_ >= 1


def test_calibrator_fit_guards():
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0.1, np.nan, 0.4], [0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([], [])
    with pytest.raises(ValueError):
        SigmoidCalibrator().fit([0.1, 0.2, np.nan], [0.0, 0.0, 1.0])


def test_isotonic_predict_nan_propagates():
    cal = IsotonicCalibrator().fit([0., 1., 2.], [0., 0., 1.])
    assert np.isnan(cal.predict([np.nan])[0])  # pins the documented contract


def test_pava_rejects_weight_length_mismatch():
    with pytest.raises(ValueError):
        _pava(np.arange(8.), np.ones(2))


def test_isotonic_fit_rejects_weight_shape_mismatch():
    # A too-long w would otherwise be silently truncated by the [order] fancy
    # index and then PASS _pava's shape guard — same hole, one level up.
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0., 1., 2.], [0., 0., 1.], w=np.ones(5))
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit([0., 1., 2.], [0., 0., 1.], w=np.ones(2))
