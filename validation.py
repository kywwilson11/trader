"""Statistical validation for model selection — DSR and overfitting checks.

Running hundreds of Optuna trials and picking the best validation Sharpe
guarantees selection bias: the expected MAXIMUM Sharpe of N skill-less
configs grows with sqrt(2 ln N). The Deflated Sharpe Ratio (Bailey &
Lopez de Prado, 2014, "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality") asks: what is the
probability the observed Sharpe exceeds the expected max under the null?

Promotion gates in this repo:
  - holdout Sharpe > 0 on a final time slice Optuna never saw, AND
  - DSR > DSR_MIN on that holdout (default 0.60)

Also provides a coarse CSCV-style probability-of-backtest-overfitting
estimate from per-trial fold scores.
"""

import math

import numpy as np

EULER_GAMMA = 0.5772156649015329

# Minimum deflated-Sharpe probability for a model to be promoted
DSR_MIN = 0.60


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation; |err| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, sr_std_across_trials: float,
                        sr_mean_across_trials: float = 0.0) -> float:
    """E[max SR] across n_trials configs under the null (no true skill).

    Bailey & Lopez de Prado (2014), eq. for the expected maximum of N
    gaussian draws: mean + std * ((1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(N*e))).
    """
    n = max(int(n_trials), 2)
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return sr_mean_across_trials + sr_std_across_trials * (
        (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(observed_sr: float, benchmark_sr: float,
                          n_obs: int, skew: float = 0.0,
                          kurt: float = 3.0) -> float:
    """Probability that the TRUE Sharpe exceeds benchmark_sr.

    observed_sr / benchmark_sr are per-period (NOT annualized) Sharpe
    ratios over the same n_obs sample; skew/kurt are of the returns.
    Returns a probability in [0, 1]; > 0.95 is strong evidence of skill,
    < 0.5 means the result is indistinguishable from selection luck.
    """
    if n_obs < 10:
        return 0.0
    denom = math.sqrt(max(
        1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2,
        1e-12))
    z = (observed_sr - benchmark_sr) * math.sqrt(n_obs - 1) / denom
    return _norm_cdf(z)


def dsr_from_trade_returns(trade_returns, n_trials: int,
                           sr_std_across_trials: float | None = None) -> dict:
    """End-to-end DSR for a sequence of per-trade returns.

    Args:
        trade_returns: realized per-trade returns (percent or fraction —
            unit cancels in the Sharpe).
        n_trials: number of configurations evaluated during the search
            (the selection pool the winner was picked from).
        sr_std_across_trials: PER-TRADE-period std of trial Sharpe
            estimates. If None (the usual case — trial scores are
            annualized and not commensurate), uses the null sampling std
            of a Sharpe estimator over n observations, 1/sqrt(n): under
            H0 every config's true SR is 0 and its estimate scatters with
            exactly that width (Lopez de Prado's "False Strategy" setup).

    Returns dict: {sr, expected_max_sr, dsr, n}
    """
    r = np.asarray(trade_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std() < 1e-12:
        return {'sr': 0.0, 'expected_max_sr': 0.0, 'dsr': 0.0, 'n': n}
    sr = float(r.mean() / r.std())
    centered = r - r.mean()
    m2 = float((centered ** 2).mean())
    skew = float((centered ** 3).mean() / (m2 ** 1.5 + 1e-18))
    kurt = float((centered ** 4).mean() / (m2 ** 2 + 1e-18))
    if sr_std_across_trials is None:
        sr_std_across_trials = 1.0 / math.sqrt(n)
    sr0 = expected_max_sharpe(n_trials, sr_std_across_trials)
    return {
        'sr': sr,
        'expected_max_sr': sr0,
        'dsr': deflated_sharpe_ratio(sr, sr0, n, skew, kurt),
        'n': n,
    }


def pbo_from_fold_scores(fold_score_rows) -> float | None:
    """Coarse CSCV probability-of-backtest-overfitting from fold scores.

    Args:
        fold_score_rows: list of per-trial fold-score vectors (each length
            n_folds, e.g. the 3 walk-forward fold Sharpes Optuna recorded).

    For every leave-one-fold-out combination, pick the trial with the best
    in-sample (remaining folds) mean and check whether its out-of-sample
    fold ranks in the bottom half. PBO = fraction of combinations where the
    IS winner is a below-median OOS performer. With only 3 folds this is a
    coarse screen, not a substitute for the holdout gate.
    """
    rows = [np.asarray(r, dtype=np.float64) for r in fold_score_rows
            if r is not None and len(r) >= 2 and np.all(np.isfinite(r))]
    if len(rows) < 8:
        return None
    n_folds = min(len(r) for r in rows)
    mat = np.stack([r[:n_folds] for r in rows])  # trials x folds

    below_median = 0
    combos = 0
    for oos_fold in range(n_folds):
        is_folds = [f for f in range(n_folds) if f != oos_fold]
        is_mean = mat[:, is_folds].mean(axis=1)
        winner = int(np.argmax(is_mean))
        oos_scores = mat[:, oos_fold]
        rank = (oos_scores < oos_scores[winner]).mean()  # fraction beaten
        combos += 1
        if rank < 0.5:
            below_median += 1
    return below_median / combos if combos else None
