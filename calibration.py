"""Leak-free probability calibration for the meta-label gate (wave-9 #1).

The meta-labeler fit its isotonic calibrator on the SAME validation slice the
LightGBM booster early-stopped on (meta_label.py, no purge) — so the calibrator
inherits the booster's in-sample optimism and maps to probabilities "closer to 0
and 1 than they should be" (scikit-learn's own warning). That p drives a LIVE
veto (p<0.30) and a LIVE size multiplier, so the bias fills sub-hurdle trades and
over-sizes losers. Boosted-tree scores already "tend to the extremes"
(Kull-Filho-Flach 2017), which the leak compounds.

The fix (López de Prado AFML ch.7): calibrate on OUT-OF-FOLD predictions from a
PURGED k-fold (no train row's label span [entry,exit] overlaps its test fold),
and use a sigmoid (Platt) fit on thin books where isotonic overfits.

Everything here is pure numpy/scipy (sklearn is absent on the dev Mac). The
cross-fit orchestrator is generic over the model (a fit_predict_fn callable), so
it is unit-testable with a numpy stub here and takes a LightGBM closure on the
Jetson. The calibrators are plain arrays/coeffs — joblib-serialisable, and their
.predict matches the sklearn interface meta_label already loads.
"""
import numpy as np

# Isotonic overfits below ~1000 samples (sklearn guidance); use sigmoid there.
ISOTONIC_MIN_N = 1000


def _pava(y, w):
    """Pool-Adjacent-Violators: weighted L2 isotonic (non-decreasing) fit of y.

    y is assumed already ordered by the calibration score x. Returns the fitted
    values aligned to that order. Matches scipy.optimize.isotonic_regression.
    """
    val, wt, size = [], [], []
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        v, ww, sz = yi, wi, 1
        while val and val[-1] >= v:          # violation -> pool with previous block
            pv, pw, ps = val.pop(), wt.pop(), size.pop()
            v = (v * ww + pv * pw) / (ww + pw)
            ww += pw
            sz += ps
        val.append(v); wt.append(ww); size.append(sz)
    out = np.empty(len(y))
    pos = 0
    for v, sz in zip(val, size):
        out[pos:pos + sz] = v
        pos += sz
    return out


class IsotonicCalibrator:
    """Monotone non-decreasing score->probability map (PAVA), clip out of bounds."""

    def fit(self, raw, y, w=None):
        raw = np.asarray(raw, float)
        y = np.asarray(y, float)
        order = np.argsort(raw, kind='mergesort')
        xs = raw[order]
        ys = y[order]
        ws = np.ones_like(ys) if w is None else np.asarray(w, float)[order]
        fit = _pava(ys, ws)
        # Collapse to unique thresholds for a strictly-increasing np.interp grid.
        ux = np.unique(xs)
        last = np.searchsorted(xs, ux, side='right') - 1
        self.x_ = ux
        self.y_ = np.clip(fit[last], 0.0, 1.0)
        return self

    def predict(self, raw):
        raw = np.asarray(raw, float)
        if len(self.x_) == 1:
            return np.full(raw.shape, self.y_[0])
        return np.interp(raw, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])


class SigmoidCalibrator:
    """Platt scaling p = sigmoid(a + b*score), Newton-IRLS. Robust on small n."""

    def fit(self, raw, y, w=None):
        x = np.asarray(raw, float)
        y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        X = np.column_stack([np.ones_like(x), x])
        beta = np.zeros(2)
        for _ in range(100):
            eta = np.clip(X @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            W = w * p * (1 - p) + 1e-9
            grad = X.T @ (w * (y - p))
            H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-9:
                break
        self.beta_ = beta
        return self

    def predict(self, raw):
        x = np.asarray(raw, float)
        eta = np.clip(self.beta_[0] + self.beta_[1] * x, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))


def choose_calibration_method(n, min_n=ISOTONIC_MIN_N):
    """'isotonic' only when there is enough data; else 'sigmoid' (small-n safe)."""
    return 'isotonic' if n >= min_n else 'sigmoid'


def fit_calibrator(scores, y, min_n=ISOTONIC_MIN_N):
    """Pick + fit a calibrator on (scores, y), ignoring NaN scores. Returns the
    fitted calibrator, or None when there is nothing usable."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, float)
    mask = np.isfinite(scores) & np.isfinite(y)
    scores, y = scores[mask], y[mask]
    if scores.size < 10 or np.unique(y).size < 2:
        return None
    method = choose_calibration_method(scores.size, min_n)
    cal = IsotonicCalibrator() if method == 'isotonic' else SigmoidCalibrator()
    return cal.fit(scores, y)


def purged_kfold_indices(entry, exit_, k=5, embargo=0.0):
    """Purged k-fold (AFML ch.7): contiguous test folds in row order, with train
    rows whose label span [entry,exit] overlaps the test fold's time window (plus
    an embargo) REMOVED, so no leakage from overlapping labels.

    entry/exit are per-row label-span bounds (bar indices or timestamps), already
    sorted by entry. Returns [(train_idx, test_idx), ...]; a single fold for n<k.
    """
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    n = len(entry)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    edges = np.linspace(0, n, k + 1).astype(int)
    folds = []
    for i in range(k):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        test = np.arange(a, b)
        t_start = entry[test].min()
        t_end = exit_[test].max()
        span = max(t_end - t_start, 0.0)
        emb = embargo * span if 0.0 < embargo < 1.0 else embargo
        keep = np.ones(n, dtype=bool)
        keep[test] = False
        overlap = (entry <= (t_end + emb)) & (exit_ >= (t_start - 0.0))
        keep &= ~overlap
        folds.append((np.where(keep)[0], test))
    return folds


def crossfit_oof_predict(fit_predict_fn, X, y, entry, exit_, k=5, embargo=0.0):
    """Out-of-fold scores via a purged k-fold. fit_predict_fn(X_tr,y_tr,X_te)->scores.

    Generic over the model so it is testable with a numpy stub here and a
    LightGBM closure on the Jetson. Rows whose fold has no usable training data
    come back NaN (fit_calibrator drops them).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    for train_idx, test_idx in purged_kfold_indices(entry, exit_, k, embargo):
        if len(train_idx) < 10 or np.unique(y[train_idx]).size < 2:
            continue
        scores = fit_predict_fn(X[train_idx], y[train_idx], X[test_idx])
        oof[test_idx] = np.asarray(scores, float)
    return oof


def brier(p, y):
    """Mean squared error of probabilistic forecasts (lower = better calibrated)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))
