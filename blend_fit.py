"""Out-of-fold stacked LSTM/LGB blend weight (wave-9 #2).

The live ensemble is predict = w*lstm + (1-w)*lgb with a HARDCODED w=0.6
(model_lgb.ensemble_predict) that is never tuned, never holdout-validated, and —
per the repo's own comment (hypersearch_v2: "tree ensembles are the stronger
learner at this data size") — likely UNDER-weights the LightGBM leg.

fit_blend_weight selects w on OUT-OF-FOLD predictions two ways (Breiman stacked
regression, 1-DOF; and a search maximizing the long-only policy Sharpe), then
SHRINKS toward 0.5. The shrinkage is deliberate: the forecast-combination puzzle
(Smith-Wallis 2009; Timmermann 2006) shows estimated weights routinely lose to
the simple average via finite-sample variance, so we cap the downside. Pure
numpy — Mac-testable; the live read is
ensemble_predict(..., lstm_weight=cfg.get('lstm_weight', 0.6)).

References: Wolpert 1992 (stacked generalization); Breiman 1996 (stacked
regressions — non-negativity is crucial); Granger-Ramanathan 1984; Diebold-Shin
2019 (shrink to equal weights).
"""
import numpy as np


def _policy_sharpe(pred, y, threshold):
    """Per-trade Sharpe of the long-only policy "take pred>=threshold"."""
    take = pred >= threshold
    if int(take.sum()) < 5:
        return 0.0
    r = y[take]
    if r.std() < 1e-12:
        return 0.0
    return float(r.mean() / r.std())


def fit_blend_weight(lstm_oof, lgb_oof, y, objective='sharpe', threshold=0.0,
                     shrink_to=0.5, shrink_lambda=0.5):
    """Blend weight w in [0,1] for w*lstm + (1-w)*lgb, shrunk toward shrink_to.

    objective='nnls'   : Breiman stacked regression, 1-DOF convex form
                         (minimize ||y - (w*lstm+(1-w)*lgb)||^2 over w in [0,1]).
    objective='sharpe' : 1-DOF search maximizing the long-only policy Sharpe.
    Returns shrink_to (clipped to [0,1]) on degenerate/thin input — fail-safe to
    the simple average. Unknown objectives raise ValueError.
    """
    a = np.asarray(lstm_oof, float)
    b = np.asarray(lgb_oof, float)
    y = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[m], b[m], y[m]
    shrink_to = min(max(float(shrink_to), 0.0), 1.0)
    if a.size < 20:
        return shrink_to

    if objective == 'nnls':
        d = a - b
        denom = float(d @ d)
        if denom < 1e-12:              # identical legs: every w is the same blend
            return shrink_to
        w = float(((y - b) @ d) / denom)
    elif objective == 'sharpe':
        ws = np.linspace(0.0, 1.0, 101)
        s = np.asarray([_policy_sharpe(w * a + (1.0 - w) * b, y, threshold) for w in ws])
        # Sharpe depends on w only through the take-set, so the grid is piecewise
        # constant with exact-tie plateaus; break ties toward the shrink target
        # instead of np.argmax's leftmost (most-LGB-heavy) grid edge.
        best = np.flatnonzero(s == s.max())
        w = float(ws[best[np.argmin(np.abs(ws[best] - shrink_to))]])
    else:
        raise ValueError(f"unknown objective {objective!r}")

    w = min(max(w, 0.0), 1.0)
    lam = min(max(float(shrink_lambda), 0.0), 1.0)
    return float((1.0 - lam) * w + lam * shrink_to)
