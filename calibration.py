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

SCOPE OF THE GUARANTEE: the purged cross-fit removes the META booster's
same-slice leak ONLY. The primary model's predictions inside the meta feature
matrix ('pred') — and the entry filter selecting which bars become meta rows —
are generated in-sample w.r.t. the primary's training window (rev-07-01 "meta
in-sample-primary leak"); no fold arrangement here can make them out-of-sample.
A green compare_calibrations verdict certifies the meta-calibration leak is
closed, not that p is unbiased.

Everything here is pure numpy (scipy appears only as the tests' oracle; sklearn
is absent on the dev Mac). The cross-fit orchestrator is generic over the model
(a fit_predict_fn callable), so it is unit-testable with a numpy stub here and
takes a LightGBM closure on the Jetson. The calibrators are plain arrays/coeffs — joblib-serialisable, and their
.predict matches the sklearn interface meta_label already loads.
"""
import numpy as np

# Isotonic overfits below ~1000 samples (sklearn guidance); use sigmoid there.
ISOTONIC_MIN_N = 1000

_LOGIT_EPS = 1e-6


def _logit(p):
    """logit with clipping; propagates NaN (predict contract: callers guard)."""
    p = np.clip(np.asarray(p, float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def _pava(y, w):
    """Pool-Adjacent-Violators: weighted L2 isotonic (non-decreasing) fit of y.

    y is assumed already ordered by the calibration score x. Returns the fitted
    values aligned to that order. Matches scipy.optimize.isotonic_regression.
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    if w.shape != y.shape:
        raise ValueError(f'_pava: y and w must have the same shape '
                         f'(y={y.shape}, w={w.shape})')
    val, wt, size = [], [], []
    for yi, wi in zip(y, w):
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
    """Monotone non-decreasing score->probability map (PAVA), clip out of bounds. The w
    parameter is reserved (no production caller passes it). predict propagates
    non-finite raw as NaN — callers must guard."""

    def __init__(self, pool_ties=False):
        # CALIBRATION_V2: pool tied scores by weighted mean before PAVA.
        self.pool_ties = bool(pool_ties)

    def fit(self, raw, y, w=None):
        raw = np.asarray(raw, float)
        y = np.asarray(y, float)
        if raw.shape != y.shape:
            raise ValueError(f'IsotonicCalibrator.fit: raw/y shape mismatch '
                             f'{raw.shape} vs {y.shape}')
        if raw.size == 0:
            raise ValueError('IsotonicCalibrator.fit: empty input')
        if not (np.isfinite(raw).all() and np.isfinite(y).all()):
            raise ValueError('IsotonicCalibrator.fit: non-finite input '
                             '(fit_calibrator masks NaN before fitting)')
        if w is not None:
            w = np.asarray(w, float)
            if w.shape != raw.shape:
                raise ValueError(f'IsotonicCalibrator.fit: w shape {w.shape} '
                                 f'!= raw shape {raw.shape}')
        order = np.argsort(raw, kind='mergesort')
        xs = raw[order]
        ys = y[order]
        ws = np.ones_like(ys) if w is None else w[order]
        if getattr(self, 'pool_ties', False):    # getattr: legacy pickles lack it
            # B04.2 (de Leeuw 1977 secondary method / sklearn _make_unique):
            # collapse duplicate x to y' = sum(w_i*y_i)/sum(w_i), w' = sum(w_i),
            # run weighted PAVA on the unique grid. Order-independent; a tied
            # score calibrates to its pooled rate, not the last row's block value.
            ux, inv = np.unique(xs, return_inverse=True)
            wsum = np.bincount(inv, weights=ws)
            ysum = np.bincount(inv, weights=ws * ys)
            fit = _pava(ysum / wsum, wsum)
            self.x_ = ux
            self.y_ = np.clip(fit, 0.0, 1.0)
            return self
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
    """Platt scaling p = sigmoid(a + b*score), Newton-IRLS. Robust on small n. Records n_iter_/converged_; predict
    propagates non-finite raw as NaN — callers must guard. w is reserved."""

    def __init__(self, platt_v2=False):
        # CALIBRATION_V2: fit on logit(score) with Platt MAP target smoothing.
        self.platt_v2 = bool(platt_v2)

    def fit(self, raw, y, w=None):
        x = np.asarray(raw, float)
        y = np.asarray(y, float)
        w = np.ones_like(y) if w is None else np.asarray(w, float)
        if x.shape != y.shape or w.shape != y.shape:
            raise ValueError(f'SigmoidCalibrator.fit: shape mismatch raw={x.shape} '
                             f'y={y.shape} w={w.shape}')
        if x.size == 0:
            raise ValueError('SigmoidCalibrator.fit: empty input')
        if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()):
            raise ValueError('SigmoidCalibrator.fit: non-finite input '
                             '(fit_calibrator masks NaN before fitting)')
        if getattr(self, 'platt_v2', False):    # getattr: legacy pickles lack it
            # Kull-Silva Filho-Flach 2017: sigmoid(a+b*p) on a probability is
            # misspecified; Niculescu-Mizil & Caruana fit Platt on log-odds.
            x = _logit(x)
            # Platt 1999 MAP-under-uniform-prior targets: kills quasi-separation.
            n_pos = float((y == 1.0).sum())
            n_neg = float((y == 0.0).sum())
            t = np.where(y == 1.0, (n_pos + 1.0) / (n_pos + 2.0),
                         1.0 / (n_neg + 2.0))
        else:
            t = y
        X = np.column_stack([np.ones_like(x), x])
        beta = np.zeros(2)
        converged = False
        it = 0
        for it in range(1, 101):
            eta = np.clip(X @ beta, -30, 30)
            p = 1.0 / (1.0 + np.exp(-eta))
            W = w * p * (1 - p) + 1e-9
            grad = X.T @ (w * (t - p))  # flag OFF: t IS y (same array)
            H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
            step = np.linalg.solve(H, grad)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-9:
                converged = True
                break
        self.beta_ = beta
        self.n_iter_ = it
        self.converged_ = converged
        return self

    def predict(self, raw):
        x = np.asarray(raw, float)
        if getattr(self, 'platt_v2', False):
            x = _logit(x)   # NaN still propagates — predict contract preserved
        eta = np.clip(self.beta_[0] + self.beta_[1] * x, -30, 30)
        return 1.0 / (1.0 + np.exp(-eta))


def choose_calibration_method(n, min_n=ISOTONIC_MIN_N):
    """'isotonic' only when there is enough data; else 'sigmoid' (small-n safe)."""
    return 'isotonic' if n >= min_n else 'sigmoid'


def fit_calibrator(scores, y, min_n=ISOTONIC_MIN_N, v2=None):
    """Pick + fit a calibrator on (scores, y), ignoring NaN scores. Returns the
    fitted calibrator, or None when there is nothing usable (thin data,
    one-class labels, or constant scores) — the decline reason is printed.
    y must be binary 0/1. The returned calibrator carries fit provenance:
    method_, n_fit_, n_dropped_, base_rate_ (plain joblib-safe scalars)."""
    scores = np.asarray(scores, float)
    y = np.asarray(y, float)
    mask = np.isfinite(scores) & np.isfinite(y)
    scores, y = scores[mask], y[mask]
    u = np.unique(y)
    if u.size and not np.isin(u, (0.0, 1.0)).all():
        raise ValueError(f'fit_calibrator: y must be binary 0/1 labels, '
                         f'got values {u[:5]}')
    # Degenerate guards: thin data, one-class labels, or CONSTANT scores —
    # a constant-score model has no ranking to calibrate; isotonic collapses
    # to a single p (verified pathological p==1.0 in the 2026-07 review).
    n_unique_scores = np.unique(scores).size
    if scores.size < 10 or u.size < 2 or n_unique_scores < 2:
        print(f"[CALIB] declined: n={scores.size} classes={u.size} "
              f"unique_scores={n_unique_scores} — no calibrator fitted")
        return None
    if v2 is None:
        # Call-time flag read (no import cycle; strategy_config is pure
        # constants) so monkeypatched flips need no restart-order care.
        try:
            import strategy_config as _sc
            v2 = bool(getattr(_sc, 'CALIBRATION_V2', False))
        except Exception:
            v2 = False
    method = choose_calibration_method(scores.size, min_n)
    cal = (IsotonicCalibrator(pool_ties=v2) if method == 'isotonic'
           else SigmoidCalibrator(platt_v2=v2))
    cal.fit(scores, y)
    cal.method_ = method
    cal.n_fit_ = int(scores.size)
    cal.n_dropped_ = int((~mask).sum())
    cal.base_rate_ = float(y.mean())
    cal.calibration_v2_ = bool(v2)
    if method == 'sigmoid' and (not cal.converged_ or abs(cal.beta_[1]) > 50.0):
        print(f"[CALIB] WARNING: sigmoid fit suspect (converged={cal.converged_}, "
              f"n_iter={cal.n_iter_}, slope={cal.beta_[1]:.1f}) — likely "
              f"(quasi-)separable calibration sample; p may saturate at 0/1")
    return cal


def purged_kfold_indices(entry, exit_, k=5, embargo=0.0):
    """Purged k-fold (AFML ch.7): contiguous test folds in row order, with train
    rows whose label span [entry,exit] overlaps the test fold's time window (plus
    an embargo) REMOVED, so no leakage from overlapping labels.

    entry/exit are per-row label-span bounds (bar indices or timestamps), and
    MUST be sorted ascending by entry (validated). embargo has DUAL semantics:
    a value in (0, 1) is a FRACTION of the test fold's own [min entry, max exit]
    time span; a value >= 1 is an ABSOLUTE offset in the same units as
    entry/exit (the meta_label call site passes epoch SECONDS, so embargo=1.0
    there means one second, NOT 100%). The embargo is forward-only (AFML: it
    covers observations FOLLOWING the test window; backward leakage is already
    handled by the overlap term). Returns [(train_idx, test_idx), ...] with
    min(k, n) folds — for n < k each fold is a single row, and
    crossfit_oof_predict will then skip them all (it requires >= 10 training
    rows), yielding an all-NaN OOF.
    """
    entry = np.asarray(entry, float)
    exit_ = np.asarray(exit_, float)
    if entry.shape != exit_.shape:
        raise ValueError(f'purged_kfold_indices: entry/exit_ shape mismatch '
                         f'{entry.shape} vs {exit_.shape}')
    if not (np.isfinite(entry).all() and np.isfinite(exit_).all()):
        raise ValueError('purged_kfold_indices: entry/exit_ must be finite — '
                         'a NaN span bound would silently disable purging')
    if np.any(exit_ < entry):
        raise ValueError('purged_kfold_indices: exit_ must be >= entry elementwise')
    if entry.size > 1 and np.any(np.diff(entry) < 0):
        raise ValueError('purged_kfold_indices: entry must be sorted ascending '
                         '(time-ordered rows)')
    if not np.isfinite(embargo) or embargo < 0:
        raise ValueError(f'purged_kfold_indices: embargo must be finite and >= 0, '
                         f'got {embargo!r}')
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
        overlap = (entry <= (t_end + emb)) & (exit_ >= t_start)
        keep &= ~overlap
        folds.append((np.where(keep)[0], test))
    return folds


def crossfit_oof_predict(fit_predict_fn, X, y, entry, exit_, k=5, embargo=0.0):
    """Out-of-fold scores via a purged k-fold. fit_predict_fn(X_tr,y_tr,X_te)->scores.

    Generic over the model so it is testable with a numpy stub here and a
    LightGBM closure on the Jetson. Rows whose fold has no usable training data
    come back NaN (fit_calibrator drops them). Purging guarantees ONLY that no
    training ROW's label span overlaps its test fold: any hyperparameter the
    caller's closure captured from the full sample (e.g. an early-stopped
    n_iter) still leaks into every fold and is the CALLER's responsibility.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)
    if not (len(X) == len(entry) == len(exit_) == n):
        raise ValueError(f'crossfit_oof_predict: length mismatch X={len(X)} '
                         f'y={n} entry={len(entry)} exit_={len(exit_)}')
    oof = np.full(n, np.nan)
    folds = purged_kfold_indices(entry, exit_, k, embargo)
    used = 0
    for train_idx, test_idx in folds:
        if len(train_idx) < 10 or np.unique(y[train_idx]).size < 2:
            continue
        scores = np.asarray(fit_predict_fn(X[train_idx], y[train_idx],
                                           X[test_idx]), float)
        if scores.shape != test_idx.shape:
            raise ValueError(f'crossfit_oof_predict: fit_predict_fn returned '
                             f'shape {scores.shape}, expected {test_idx.shape}')
        oof[test_idx] = scores
        used += 1
    print(f"[CALIB] cross-fit: {used}/{len(folds)} folds usable, "
          f"{int(np.isfinite(oof).sum())}/{n} rows scored")
    return oof


def brier(p, y):
    """Mean squared error of probabilistic forecasts (lower = better calibrated).
    Non-finite (p, y) pairs are dropped — matching expected_calibration_error /
    reliability_curve — and nan is returned when nothing survives."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if p.shape != y.shape:
        raise ValueError(f'brier: shape mismatch p={p.shape} y={y.shape}')
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if p.size == 0:
        return float('nan')
    return float(np.mean((p - y) ** 2))


def reliability_curve(p, y, n_bins=10):
    """Binned reliability: per equal-width probability bin, the mean predicted
    probability vs the observed win frequency. A well-calibrated forecast has
    obs_freq ~= pred_mean in every populated bin (the diagonal)."""
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f'n_bins must be >= 1, got {n_bins}')
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
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f'n_bins must be >= 1, got {n_bins}')
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
    All metrics are computed on the single jointly-finite row set (NaN rows —
    e.g. unscored OOF folds — are dropped and counted in 'n_dropped'), so both
    arms are always scored on identical rows. Flip the flag only when the purged
    calibrator is at least as well-calibrated (lower/equal Brier AND ECE) on the
    holdout — never on a same-slice score, never on an exact tie ('tied': equal
    metrics carry zero evidence), and never if the win-rate floor is threatened
    (that is decision_report's job). brier_delta (>0 = purged improves) comes
    with its exact paired standard error and t-stat as flip-decision evidence.
    NOTE: a green verdict certifies the META-calibration leak is closed; the
    primary model's in-sample 'pred' feature is a separate residual this
    comparison cannot see (module docstring, SCOPE OF THE GUARANTEE).
    """
    pl = np.asarray(p_legacy, float)
    pp = np.asarray(p_purged, float)
    y = np.asarray(y, float)
    if not (pl.shape == pp.shape == y.shape):
        raise ValueError(f'compare_calibrations: shape mismatch '
                         f'p_legacy={pl.shape} p_purged={pp.shape} y={y.shape}')
    m = np.isfinite(pl) & np.isfinite(pp) & np.isfinite(y)
    n_dropped = int((~m).sum())
    pl, pp, y = pl[m], pp[m], y[m]
    if y.size == 0:
        return {
            'n': 0, 'n_dropped': n_dropped,
            'brier_legacy': None, 'brier_purged': None,
            'ece_legacy': None, 'ece_purged': None,
            'brier_improved': False, 'ece_improved': False,
            'brier_delta': None, 'brier_delta_se': None, 'brier_delta_t': None,
            'tied': False,
            'verdict': ('no jointly-finite holdout rows — nothing to compare; '
                        'collect a dump'),
            'reliability_legacy': reliability_curve(pl, y, n_bins),
            'reliability_purged': reliability_curve(pp, y, n_bins),
        }
    bl, bp = brier(pl, y), brier(pp, y)
    el = expected_calibration_error(pl, y, n_bins)
    ep = expected_calibration_error(pp, y, n_bins)
    tied = bool(bp == bl and el is not None and ep is not None and ep == el)
    better = ((bp <= bl) and (ep is not None and el is not None and ep <= el)
              and not tied)
    # Paired per-row Brier delta: > 0 means purged improves. Exact SE, no bootstrap.
    d = (pl - y) ** 2 - (pp - y) ** 2
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else None
    tstat = float(delta / se) if (se is not None and se > 0.0) else None
    if tied:
        verdict = ('tied — no evidence on this holdout; do NOT flip '
                   'META_CALIBRATION_MODE (collect a discriminating dump)')
    elif better:
        verdict = ('purged_oof is better-calibrated on the holdout — safe to flip '
                   'META_CALIBRATION_MODE (then re-certify in shadow)')
    else:
        verdict = ('no calibration improvement on this holdout — keep legacy / '
                   'collect more')
    return {
        'n': int(m.sum()), 'n_dropped': n_dropped,
        'brier_legacy': round(bl, 5), 'brier_purged': round(bp, 5),
        'ece_legacy': round(el, 5) if el is not None else None,
        'ece_purged': round(ep, 5) if ep is not None else None,
        'brier_improved': bool(bp < bl),
        'ece_improved': bool(ep is not None and el is not None and ep < el),
        'brier_delta': delta, 'brier_delta_se': se, 'brier_delta_t': tstat,
        'tied': tied,
        'verdict': verdict,
        'reliability_legacy': reliability_curve(pl, y, n_bins),
        'reliability_purged': reliability_curve(pp, y, n_bins),
    }
