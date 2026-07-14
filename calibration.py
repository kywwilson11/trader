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
    fitted calibrator, or None when there is nothing usable (thin data,
    one-class labels, or constant scores)."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, float)
    mask = np.isfinite(scores) & np.isfinite(y)
    scores, y = scores[mask], y[mask]
    # Degenerate guards: thin data, one-class labels, or CONSTANT scores —
    # a constant-score model has no ranking to calibrate; isotonic collapses
    # to a single p (verified pathological p==1.0 in the 2026-07 review).
    if (scores.size < 10 or np.unique(y).size < 2
            or np.unique(scores).size < 2):
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


def reliability_curve(p, y, n_bins=10):
    """Binned reliability: per equal-width probability bin, the mean predicted
    probability vs the observed win frequency. A well-calibrated forecast has
    obs_freq ~= pred_mean in every populated bin (the diagonal)."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        out.append({'bin': b, 'n': n,
                    'pred_mean': round(float(p[sel].mean()), 4) if n else None,
                    'obs_freq': round(float(y[sel].mean()), 4) if n else None})
    return out


def expected_calibration_error(p, y, n_bins=10):
    """ECE = sum_b (n_b/N) * |pred_mean_b - obs_freq_b| — the n-weighted average
    gap from the diagonal. 0 = perfectly calibrated; bigger = more mis-calibrated."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece, N = 0.0, len(p)
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n:
            ece += (n / N) * abs(float(p[sel].mean()) - float(y[sel].mean()))
    return float(ece)


def compare_calibrations(p_legacy, p_purged, y, n_bins=10):
    """Before/after gate for META_CALIBRATION_MODE='purged_oof' (wave-9 #1).

    Compares the leaked same-slice calibration (p_legacy) against the purged
    out-of-fold one (p_purged) on the SAME held-out outcomes y, by Brier and ECE.
    Flip the flag only when the purged calibrator is at least as well-calibrated
    (lower/equal Brier AND ECE) on the holdout — never on a same-slice score, and
    never if the win-rate floor is threatened (that is decision_report's job).
    """
    y = np.asarray(y, float)
    bl, bp = brier(p_legacy, y), brier(p_purged, y)
    el = expected_calibration_error(p_legacy, y, n_bins)
    ep = expected_calibration_error(p_purged, y, n_bins)
    better = (bp <= bl) and (ep is not None and el is not None and ep <= el)
    return {
        'n': int(np.isfinite(y).sum()),
        'brier_legacy': round(bl, 5), 'brier_purged': round(bp, 5),
        'ece_legacy': round(el, 5) if el is not None else None,
        'ece_purged': round(ep, 5) if ep is not None else None,
        'brier_improved': bool(bp < bl),
        'ece_improved': bool(ep is not None and el is not None and ep < el),
        'verdict': ('purged_oof is better-calibrated on the holdout — safe to flip '
                    'META_CALIBRATION_MODE (then re-certify in shadow)' if better else
                    'no calibration improvement on this holdout — keep legacy / collect more'),
        'reliability_legacy': reliability_curve(p_legacy, y, n_bins),
        'reliability_purged': reliability_curve(p_purged, y, n_bins),
    }
