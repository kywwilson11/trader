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
    Returns shrink_to on degenerate/thin input (fail-safe to the simple average).
    """
    a = np.asarray(lstm_oof, float)
    b = np.asarray(lgb_oof, float)
    y = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[m], b[m], y[m]
    if a.size < 20:
        return float(shrink_to)

    if objective == 'nnls':
        d = a - b
        denom = float(d @ d)
        w = 0.5 if denom < 1e-12 else float(((y - b) @ d) / denom)
    else:
        ws = np.linspace(0.0, 1.0, 101)
        sharpes = [_policy_sharpe(w * a + (1.0 - w) * b, y, threshold) for w in ws]
        w = float(ws[int(np.argmax(sharpes))])

    w = min(max(w, 0.0), 1.0)
    lam = min(max(float(shrink_lambda), 0.0), 1.0)
    return float((1.0 - lam) * w + lam * float(shrink_to))
