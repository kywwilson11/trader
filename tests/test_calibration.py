"""Wave-9 #1: leak-free meta-label calibration kernels.

Covers the PAVA isotonic (vs scipy oracle), the small-n sigmoid fallback, the
purged k-fold (no train span overlaps its test fold), the cross-fit orchestrator,
and the headline LEAK-DETECTION property: a calibrator fit on the same
(over-separated) slice the model peeked at is MORE miscalibrated on fresh data
than one fit on honest out-of-fold scores.
"""
import numpy as np
import pytest

from calibration import (
    _pava,
    IsotonicCalibrator,
    SigmoidCalibrator,
    choose_calibration_method,
    fit_calibrator,
    purged_kfold_indices,
    crossfit_oof_predict,
    brier,
    reliability_curve,
    expected_calibration_error,
    compare_calibrations,
    ISOTONIC_MIN_N,
)


def test_pava_matches_scipy_oracle():
    iso = pytest.importorskip("scipy.optimize").isotonic_regression
    rng = np.random.default_rng(1)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = rng.normal(size=200)
        w = rng.uniform(0.5, 2.0, size=200)
        np.testing.assert_allclose(_pava(y, w), iso(y, weights=w).x, rtol=1e-9, atol=1e-9)


def test_isotonic_calibrator_monotone_and_clipped():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 2000)
    y = (rng.uniform(size=2000) < raw).astype(float)   # true prob == raw
    cal = IsotonicCalibrator().fit(raw, y)
    grid = np.linspace(0, 1, 50)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= -1e-9)                  # non-decreasing
    assert p.min() >= 0.0 and p.max() <= 1.0
    # out-of-range inputs clip to the end values, never extrapolate
    assert cal.predict([-5.0])[0] == pytest.approx(p[0], abs=0.2)
    assert cal.predict([5.0])[0] == pytest.approx(cal.y_[-1])
    # roughly recovers the identity calibration map
    assert np.mean(np.abs(p - grid)) < 0.08


def test_sigmoid_calibrator_recovers_logistic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=5000)
    true_p = 1 / (1 + np.exp(-(0.7 + 1.5 * x)))
    y = (rng.uniform(size=5000) < true_p).astype(float)
    cal = SigmoidCalibrator().fit(x, y)
    assert cal.beta_[0] == pytest.approx(0.7, abs=0.2)
    assert cal.beta_[1] == pytest.approx(1.5, abs=0.2)


def test_choose_method_and_fit_calibrator_fallback():
    assert choose_calibration_method(ISOTONIC_MIN_N) == 'isotonic'
    assert choose_calibration_method(ISOTONIC_MIN_N - 1) == 'sigmoid'
    rng = np.random.default_rng(4)
    # thin book -> sigmoid
    small = fit_calibrator(rng.uniform(size=200), (rng.uniform(size=200) < 0.5).astype(float))
    assert isinstance(small, SigmoidCalibrator)
    # degenerate / tiny -> None (fail-open)
    assert fit_calibrator([0.1, 0.2], [1, 0]) is None
    assert fit_calibrator(np.full(50, 0.3), np.ones(50)) is None  # one class


def test_purged_kfold_no_train_span_overlaps_test_fold():
    n = 60
    entry = np.arange(n, dtype=float)
    exit_ = entry + 3.0                       # each label spans 3 bars -> overlaps neighbors
    folds = purged_kfold_indices(entry, exit_, k=5, embargo=0.0)
    assert len(folds) == 5
    seen_test = []
    for train_idx, test_idx in folds:
        seen_test.extend(test_idx.tolist())
        t0, t1 = entry[test_idx].min(), exit_[test_idx].max()
        # NO training row's [entry,exit] may overlap [t0, t1]
        assert np.all((exit_[train_idx] < t0) | (entry[train_idx] > t1))
    assert sorted(seen_test) == list(range(n))   # every row tested exactly once


def test_purged_kfold_tiny_n_single_fold():
    folds = purged_kfold_indices([0.0, 1.0], [0.5, 1.5], k=5)
    assert len(folds) <= 2 and all(len(t) >= 1 for _, t in folds)


def test_crossfit_assigns_every_row_a_score():
    rng = np.random.default_rng(5)
    n = 300
    X = rng.normal(size=(n, 2))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    entry = np.arange(n, dtype=float)
    exit_ = entry + 1.0

    def stub(Xtr, ytr, Xte):                 # honest logistic-ish scorer
        return 1 / (1 + np.exp(-Xte[:, 0]))

    oof = crossfit_oof_predict(stub, X, y, entry, exit_, k=5)
    assert np.isfinite(oof).all()


def test_leak_detection_oof_beats_same_slice_on_fresh_data():
    # honest deployment score (what the live booster gives OOS) ~ true prob + noise
    rng = np.random.default_rng(7)
    N = 6000
    true_p = rng.uniform(0.15, 0.85, N)
    y = (rng.uniform(size=N) < true_p).astype(float)
    honest = np.clip(true_p + rng.normal(0, 0.15, N), 0, 1)
    cal_idx, hold_idx = np.arange(N // 2), np.arange(N // 2, N)

    # The LEAK: on the calibration slice the booster peeked at labels, so its
    # scores over-separate the classes (pushed toward 0/1).
    leaked = np.clip(honest[cal_idx] + 0.45 * (2 * y[cal_idx] - 1), 0, 1)

    cal_leaked = IsotonicCalibrator().fit(leaked, y[cal_idx])
    cal_oof = IsotonicCalibrator().fit(honest[cal_idx], y[cal_idx])  # honest OOF analog

    # On its OWN slice the leaked calibrator looks BETTER (the in-sample optimism)...
    assert brier(cal_leaked.predict(leaked), y[cal_idx]) < \
        brier(cal_oof.predict(honest[cal_idx]), y[cal_idx])

    # ...but on FRESH data (live booster gives honest scores) it is WORSE-calibrated.
    p_leaked = cal_leaked.predict(honest[hold_idx])
    p_oof = cal_oof.predict(honest[hold_idx])
    assert brier(p_oof, y[hold_idx]) < brier(p_leaked, y[hold_idx])
    # The leak's signature (Kull-Filho-Flach): calibrating on the over-separated
    # slice makes the map OVER-CONFIDENT — it pushes fresh predictions toward the
    # extremes (higher dispersion) relative to the honest OOF calibrator.
    assert p_leaked.std() > p_oof.std()


def test_reliability_and_ece_reward_calibration():
    rng = np.random.default_rng(11)
    n = 8000
    p_true = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    # Perfectly-calibrated forecast: obs_freq ~ pred_mean per bin, tiny ECE.
    for b in reliability_curve(p_true, y, n_bins=10):
        if b['n'] > 50:
            assert abs(b['pred_mean'] - b['obs_freq']) < 0.06
    assert expected_calibration_error(p_true, y) < 0.03
    # An OVER-confident forecast (pushed toward 0/1) is worse-calibrated.
    p_over = np.clip((p_true - 0.5) * 2.0 + 0.5, 0, 1)
    assert expected_calibration_error(p_over, y) > expected_calibration_error(p_true, y)


def test_compare_calibrations_flips_only_on_holdout_improvement():
    rng = np.random.default_rng(12)
    n = 8000
    p_true = rng.uniform(0.1, 0.9, n)
    y = (rng.uniform(size=n) < p_true).astype(float)
    p_purged = np.clip(p_true + rng.normal(0, 0.05, n), 0, 1)        # honest
    p_legacy = np.clip((p_true - 0.5) * 1.8 + 0.5, 0, 1)            # over-confident (the leak)
    rep = compare_calibrations(p_legacy, p_purged, y)
    assert rep['brier_purged'] < rep['brier_legacy']
    assert rep['ece_purged'] < rep['ece_legacy']
    assert 'safe to flip' in rep['verdict']
    # When the candidate map is NOT better, the gate refuses to flip.
    rep2 = compare_calibrations(p_true, p_legacy, y)                # arg2 worse than arg1
    assert 'keep legacy' in rep2['verdict']
