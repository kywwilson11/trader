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

# The hardcoded serving-side fallback: model_lgb.ensemble_predict(...,
# lstm_weight=0.6) == predict_now/backtest's cfg.get('lstm_weight', 0.6).
# R2C-02a (defect H1): this is now the ONE certificate-visible home of that
# default — whenever boosters ship without a fitted weight, BOTH the holdout
# certificate and the shipped config resolve to this value. B12.2 OWNER NOTE:
# the pending 0.6 -> 0.5 default question is decided HERE (decision-queue
# item), in lockstep with model_lgb.ensemble_predict's keyword default.
DEFAULT_LSTM_WEIGHT = 0.6


def effective_lstm_weight(fitted_weight, ship_boosters, default=None):
    """The ONE effective blend weight for certificate AND deployment (H1).

    Whenever the LGB boosters will ship, live's ensemble_predict blends at
    cfg.get('lstm_weight', DEFAULT_LSTM_WEIGHT) regardless of whether the
    blend fit succeeded — so a None fitted weight must resolve to the SAME
    default for the holdout certificate and the shipped config (cert ==
    deploy; the R2C-02a repair of the blend-fit failure path that
    previously certified a raw LSTM while deploying a 0.6 blend).

    ship_boosters False -> the fitted weight passes through unchanged
    (None stays None: raw-LSTM certificate, no boosters — the legacy path).
    """
    if not ship_boosters:
        return None if fitted_weight is None else float(fitted_weight)
    if fitted_weight is None:
        return float(DEFAULT_LSTM_WEIGHT if default is None else default)
    return float(fitted_weight)


def _policy_sharpe(pred, y, threshold):
    """Per-trade Sharpe of the long-only policy "take pred > threshold".

    Strict '>' matches every deployed path (objective_utils
    simulate_trades_core, backtest.py, predict_now.py) — R2C-02e / L9;
    the old '>=' admitted exact-threshold rows no live gate takes.
    """
    take = pred > threshold
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


def fit_blend_weight_v2(lstm_oof, lgb_oof, y, forward_bars=1, shrink_to=0.5,
                        shrink_lambda=0.5, kish_divisor=None):
    """NNLS blend weight with an overlap-corrected significance gate (T1/B12.2).

    Deploy the estimated weight only when it differs from the simple average
    by more than 2 standard errors — otherwise ship EXACTLY 0.5. The SE uses
    the label-overlap effective sample size n_eff = n / forward_bars
    (overlapping fb-bar labels are ~fb-times over-counted), and a significant
    estimate is still Diebold-Shin shrunk toward shrink_to. Model-averaging
    weights estimated on modest samples routinely lose to the simple average
    (Claeskens et al. 2016 — estimated weights add variance that swamps the
    bias saved; Stock & Watson 2004 forecast-combination puzzle; Diebold &
    Shin 2019 — shrink to equal weights).

    kish_divisor (R2C-02e / L4, default None = legacy byte-identical):
    optional Kish design-effect divisor for pooled CROSS-SECTIONAL rows —
    deff = 1 + (G-1)*rho_bar for G names with intra-timestamp residual
    correlation rho_bar (Kish 1965). The temporal n/fb correction alone
    understates the SE on multi-ticker panels (crypto cross-name rho
    0.7-0.9), firing 'significant' too liberally. When provided (clamped
    to >= 1), n_eff divides by it and the SE grows by sqrt of it. The
    caller stays on the legacy default; activation is a runbook event.

    Returns dict {'w', 'w_raw', 'se', 'significant', 'n', 'n_eff'}; on
    degenerate/thin input w falls back to the clipped shrink target with
    w_raw/se/n_eff None (fail-safe to the simple average, mirroring
    fit_blend_weight).
    """
    a = np.asarray(lstm_oof, float)
    b = np.asarray(lgb_oof, float)
    y = np.asarray(y, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[m], b[m], y[m]
    n = a.size
    shrink_to = min(max(float(shrink_to), 0.0), 1.0)
    d = a - b
    denom = float(d @ d) if n else 0.0
    if n < 20 or denom < 1e-12:
        return {'w': float(shrink_to), 'w_raw': None, 'se': None,
                'significant': False, 'n': int(n), 'n_eff': None}

    w_raw = float(((y - b) @ d) / denom)          # UNCLIPPED estimator
    eps = y - (w_raw * a + (1.0 - w_raw) * b)
    sigma2 = float(eps @ eps) / max(n - 1, 1)
    # Label-overlap correction (B12): n_eff = n / forward_bars, i.e. the
    # OLS variance is multiplied by the overlap factor fb.
    fb = max(int(forward_bars), 1)
    if kish_divisor is None:            # legacy path — byte-identical
        n_eff = n / fb
        se = float(np.sqrt(sigma2 / denom * fb))
    else:
        kd = max(float(kish_divisor), 1.0)
        n_eff = n / (fb * kd)
        se = float(np.sqrt(sigma2 / denom * fb * kd))
    significant = abs(w_raw - 0.5) > 2.0 * se     # tested on the UNSHRUNK w
    if significant:
        lam = min(max(float(shrink_lambda), 0.0), 1.0)
        w_clip = min(max(w_raw, 0.0), 1.0)
        w = min(max((1.0 - lam) * w_clip + lam * shrink_to, 0.0), 1.0)
    else:
        w = 0.5
    return {'w': float(w), 'w_raw': w_raw, 'se': se,
            'significant': bool(significant), 'n': int(n),
            'n_eff': float(n_eff)}


def smooth_across_retrains(w_new, w_prev=None, lo=0.25, hi=0.75):
    """Cross-retrain smoothing of the blend weight (T1/B12).

    Average the fresh estimate with the champion's previously persisted
    weight, then clamp to [lo, hi] — all the safe time-variation at weekly
    cadence with zero live-path code. w_prev None/non-finite -> the fresh
    estimate alone (still clamped).
    """
    w_new = float(w_new)
    if w_prev is None or not np.isfinite(w_prev):
        w = w_new
    else:
        w = 0.5 * (w_new + float(w_prev))
    return float(min(max(w, lo), hi))


def reselect_trade_threshold(preds, y, tt_range, old_threshold, score_fn,
                             step=0.01):
    """Re-select the trade threshold on BLENDED predictions (R2C-02d / H2).

    The Optuna-searched trade_threshold is scored against raw-LSTM fold
    predictions, but serving applies it to w*LSTM + (1-w)*LGB whose
    distribution is variance-compressed by the tree leg — a systematic
    scale mismatch. This grid re-scores score_fn(preds, y, threshold) over
    [tt_range[0], tt_range[1]] at the SAME step Optuna searched (0.01) and
    returns (best_threshold, best_score). Exact score ties break toward
    old_threshold (least policy change — mirrors fit_blend_weight's
    plateau rule). Pure numpy: the caller supplies the scorer (hypersearch
    passes its compute_sharpe closure); deployment of the result is gated
    by strategy_config.BLEND_THRESHOLD_RESELECT (default OFF).
    """
    lo, hi = float(tt_range[0]), float(tt_range[1])
    step = float(step)
    n_steps = max(int(round((hi - lo) / step)), 0)
    grid = np.round(lo + np.arange(n_steps + 1) * step, 10)
    scores = np.asarray([float(score_fn(preds, y, float(t))) for t in grid])
    best = np.flatnonzero(scores == scores.max())
    thr = float(grid[best[np.argmin(np.abs(grid[best]
                                           - float(old_threshold)))]])
    return thr, float(scores.max())
