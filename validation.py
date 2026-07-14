"""Statistical validation for model selection — DSR and overfitting checks.

Running hundreds of Optuna trials and picking the best validation Sharpe
guarantees selection bias: the expected MAXIMUM Sharpe of N skill-less
configs grows with sqrt(2 ln N). The Deflated Sharpe Ratio (Bailey &
Lopez de Prado, 2014, "The Deflated Sharpe Ratio: Correcting for Selection
Bias, Backtest Overfitting and Non-Normality") asks: what is the
probability the observed Sharpe exceeds the expected max under the null?

Promotion gates in this repo:
  - holdout Sharpe > 0 on a final time slice Optuna never saw, AND
  - DSR >= DSR_MIN on that holdout (default 0.60)

Also provides a coarse CSCV-style probability-of-backtest-overfitting
estimate from per-trial fold scores, a proper Combinatorially-Symmetric
Cross-Validation PBO when a finer per-subperiod performance matrix is
available, and a Lo (2002) serial-correlation effective-sample correction.
"""

import itertools
import math

import numpy as np

EULER_GAMMA = 0.5772156649015329

# Minimum deflated-Sharpe probability for a model to be promoted
DSR_MIN = 0.60


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation; relative |err| < 1.15e-9)."""
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
                          kurt: float = 3.0,
                          n_eff: float | None = None) -> float:
    """Probability that the TRUE Sharpe exceeds benchmark_sr.

    observed_sr / benchmark_sr are per-period (NOT annualized) Sharpe
    ratios over the same n_obs sample; skew/kurt are of the returns.
    Returns a probability in [0, 1]; > 0.95 is strong evidence of skill,
    < 0.5 means the result is indistinguishable from selection luck.

    n_eff: effective (independent) sample size. The Sharpe estimator's
        sampling variance is set by the number of INDEPENDENT observations,
        not the raw row count. With overlapping forward-window labels the
        effective count n_eff = sum(average-uniqueness) is < n_obs (see
        sample_weights.py), so the z-statistic's sqrt(n-1) scaling must use
        n_eff. Defaults to n_obs (IID assumption) for backward compatibility.
    """
    if n_obs < 10:
        return 0.0
    if n_eff is not None and not math.isfinite(float(n_eff)):
        n_eff = None  # a NaN slips through the min/max clamps and poisons
        #               z -> dsr=nan; fall back to the IID raw count
    ne = n_obs if n_eff is None else float(n_eff)
    # Never let an effective count exceed the raw count or fall below the
    # 10-sample floor the gate trusts.
    ne = min(max(ne, 10.0), float(n_obs))
    denom = math.sqrt(max(
        1.0 - skew * observed_sr + ((kurt - 1.0) / 4.0) * observed_sr ** 2,
        1e-12))
    z = (observed_sr - benchmark_sr) * math.sqrt(ne - 1) / denom
    return _norm_cdf(z)


def dsr_from_trade_returns(trade_returns, n_trials: int,
                           sr_std_across_trials: float | None = None,
                           n_eff: float | None = None) -> dict:
    """End-to-end DSR for a sequence of per-trade returns.

    Args:
        trade_returns: realized per-trade returns (percent or fraction —
            unit cancels in the Sharpe).
        n_trials: number of configurations evaluated during the search
            (the selection pool the winner was picked from).
        sr_std_across_trials: PER-TRADE-period std of trial Sharpe
            estimates. If None (the usual case — trial scores are
            annualized and not commensurate), uses the null sampling std
            of a Sharpe estimator over n_eff observations, 1/sqrt(n_eff):
            under H0 every config's true SR is 0 and its estimate scatters
            with exactly that width (Lopez de Prado's "False Strategy"
            setup). Using n_eff (not raw n) makes the expected-max null
            WIDER when labels overlap, raising the bar the winner must clear.
        n_eff: effective (independent) sample size from average-uniqueness
            (sample_weights.effective_n). Defaults to the raw finite count
            (IID assumption) — supply the measured value to de-bias the gate.

    Returns dict: {sr, expected_max_sr, dsr, n, n_eff}
    """
    r = np.asarray(trade_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10 or r.std() < 1e-12:
        return {'sr': 0.0, 'expected_max_sr': 0.0, 'dsr': 0.0, 'n': n,
                'n_eff': float(n)}
    if n_eff is not None and not math.isfinite(float(n_eff)):
        n_eff = None  # NaN survives the clamps and yields dsr=nan (a
        #               confusing spurious gate fail); fall back to IID
    ne = float(n) if n_eff is None else min(max(float(n_eff), 10.0), float(n))
    sr = float(r.mean() / r.std())
    centered = r - r.mean()
    m2 = float((centered ** 2).mean())
    skew = float((centered ** 3).mean() / (m2 ** 1.5 + 1e-18))
    kurt = float((centered ** 4).mean() / (m2 ** 2 + 1e-18))
    if sr_std_across_trials is None:
        sr_std_across_trials = 1.0 / math.sqrt(ne)
    sr0 = expected_max_sharpe(n_trials, sr_std_across_trials)
    return {
        'sr': sr,
        'expected_max_sr': sr0,
        'dsr': deflated_sharpe_ratio(sr, sr0, n, skew, kurt, n_eff=ne),
        'n': n,
        'n_eff': round(ne, 2),
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
    rows = []
    for r in fold_score_rows:
        if r is None:
            continue
        try:
            # Coerce first: np.isfinite on a raw row holding a non-numeric
            # entry (e.g. None) raises TypeError instead of filtering it.
            a = np.asarray(r, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if a.ndim == 1 and a.size >= 2 and np.all(np.isfinite(a)):
            rows.append(a)
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


def _sharpe_cols(mat, cols):
    """Per-trial Sharpe over a subset of period-columns (mean/std)."""
    sub = mat[:, cols]
    mu = sub.mean(axis=1)
    sd = sub.std(axis=1)
    out = np.zeros_like(mu)
    nz = sd > 1e-12
    out[nz] = mu[nz] / sd[nz]
    return out


def pbo_cscv(perf_matrix, n_groups: int = 8, perf_fn=None) -> dict | None:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV.

    Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
    Overfitting" (J. Computational Finance). The honest upgrade over the coarse
    3-fold screen in pbo_from_fold_scores.

    Args:
        perf_matrix: array shape [n_trials, T] — each row is one configuration's
            per-subperiod performance series (e.g. per-bar net returns, or
            per-block Sharpe contributions). Needs T >= n_groups and the
            granularity to repartition; feed it from a hypersearch that records
            per-subperiod scores (not just 3 fold means).
        n_groups: S, the number of equal column groups (even). All C(S, S/2)
            ways of choosing the IS half are evaluated; the OOS half is the
            complement — symmetric, so every block serves as both IS and OOS.
        perf_fn: optional (submatrix)->per-trial score; default in-sample and
            out-of-sample Sharpe via mean/std.

    Method: for each split, pick the IS-best trial, find its OOS performance
    RANK omega in (0,1), take the logit lambda = ln(omega/(1-omega)). PBO is
    the fraction of splits where the IS-winner lands at/below the OOS median
    (lambda <= 0) — i.e. in-sample selection did not carry out of sample.

    Rows containing non-finite entries are DROPPED (matching
    pbo_from_oos_blocks) — zero-filling them would silently mutate trial
    performance and shift the OOS ranks.

    Returns {pbo, n_splits, median_logit, mean_oos_rank} or None if the matrix
    is too small / degenerate.
    """
    m = np.asarray(perf_matrix, dtype=np.float64)
    if m.ndim != 2:
        return None
    m = m[np.all(np.isfinite(m), axis=1)]
    n_trials, t = m.shape
    if n_trials < 2 or n_groups < 2 or n_groups % 2 != 0 or t < n_groups:
        return None
    score = perf_fn or _sharpe_cols

    # Equal column groups (drop the remainder so groups are balanced).
    gsz = t // n_groups
    groups = [np.arange(g * gsz, (g + 1) * gsz) for g in range(n_groups)]
    half = n_groups // 2

    logits = []
    oos_ranks = []
    below = 0
    n_splits = 0
    for is_groups in itertools.combinations(range(n_groups), half):
        is_set = set(is_groups)
        is_cols = np.concatenate([groups[g] for g in is_groups])
        oos_cols = np.concatenate([groups[g] for g in range(n_groups)
                                   if g not in is_set])
        is_perf = score(m, is_cols)
        oos_perf = score(m, oos_cols)
        winner = int(np.argmax(is_perf))
        # relative OOS rank of the IS winner in (0,1): fraction of trials it
        # beats OOS, smoothed by (n+1) so the extremes never hit 0 or 1.
        beaten = int(np.sum(oos_perf < oos_perf[winner]))
        omega = (beaten + 1) / (n_trials + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lam = math.log(omega / (1.0 - omega))
        logits.append(lam)
        oos_ranks.append(omega)
        if lam <= 0.0:
            below += 1
        n_splits += 1

    if n_splits == 0:
        return None
    return {
        'pbo': below / n_splits,
        'n_splits': n_splits,
        'median_logit': float(np.median(logits)),
        'mean_oos_rank': float(np.mean(oos_ranks)),
    }


def serial_correlation_factor(returns, max_lag: int | None = None) -> dict:
    """Lo (2002) serial-correlation variance-inflation factor for a Sharpe.

    Lo, "The Statistics of Sharpe Ratios" (FAJ 2002): when per-period returns
    are autocorrelated, the IID Sharpe standard error is wrong. The variance
    inflation factor is
        f = 1 + 2 * sum_{k=1}^{q} (1 - k/(q+1)) * rho_k
    (a Newey-West-style weighting of the sample autocorrelations rho_k). The
    serial-correlation-adjusted effective sample size is n_eff = n / f, and the
    annualized-Sharpe scaling shrinks by 1/sqrt(f) when f > 1 (positive
    autocorrelation, the usual overlap case).

    IMPORTANT: this is a SEPARATE effect from label-overlap uniqueness
    (sample_weights.effective_n). Do NOT stack both n_eff reductions on the
    same DSR — they double-count variance and the sign of rho_k matters. This
    is provided OFF by default; opt in deliberately when serial correlation is
    the dominant non-IID source (e.g. a single contiguous return stream).

    Returns {factor, n_eff, n, max_lag, sharpe_scale} where sharpe_scale =
    1/sqrt(max(factor, eps)) is the multiplier on an IID Sharpe.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 12 or r.std() < 1e-12:
        return {'factor': 1.0, 'n_eff': float(n), 'n': n, 'max_lag': 0,
                'sharpe_scale': 1.0}
    q = max_lag if max_lag is not None else min(n // 4, int(round(n ** (1 / 3))) + 2)
    q = max(1, min(q, n - 1))
    x = r - r.mean()
    denom = float(np.sum(x * x))
    f = 1.0
    for k in range(1, q + 1):
        rho_k = float(np.sum(x[k:] * x[:-k]) / denom)
        f += 2.0 * (1.0 - k / (q + 1.0)) * rho_k
    # A negative factor (strong negative autocorrelation) would imply more
    # independent info than n; floor at a small positive so n_eff stays finite
    # and we never INFLATE n beyond the raw count.
    f = max(f, 1e-6)
    n_eff = min(float(n), n / f)
    return {'factor': f, 'n_eff': n_eff, 'n': n, 'max_lag': q,
            'sharpe_scale': 1.0 / math.sqrt(f)}


def build_oos_blocks(trade_returns, n_blocks: int = 8):
    """Bin a 1-D net-return stream into n_blocks contiguous equal-count block means.

    pbo_cscv needs a rectangular [n_trials, T] matrix, but different
    configurations produce different numbers of trades. This normalizes one
    configuration's variable-length per-trade (or per-bar) return series into a
    FIXED-length vector — each element is the mean net return of a contiguous
    equal-count block — so heterogeneous trials stack into one matrix.

    Returns None when there are fewer finite returns than blocks (a block would
    be empty) — the caller treats None as "cannot judge" and falls back to DSR.
    """
    r = np.asarray(trade_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if n_blocks < 2 or r.size < n_blocks:
        return None
    return np.array([chunk.mean() for chunk in np.array_split(r, n_blocks)])


def pbo_from_oos_blocks(block_rows, n_groups: int = 8) -> dict | None:
    """Fail-open CSCV-PBO over per-trial block vectors (from build_oos_blocks).

    Filters None / wrong-length / non-finite / zero-variance rows, stacks the
    rest into a [n_valid_trials, n_blocks] matrix and runs pbo_cscv. Returns None
    — leaving the promotion decision governed by the DSR gate — whenever there is
    too little to judge honestly: fewer than 2 valid trials, fewer block-columns
    than n_groups, or a width not divisible by n_groups (pbo_cscv would then
    silently drop the trailing — i.e. MOST RECENT — width % n_groups blocks of
    every trial's time-ordered stream). The gate it feeds can only ever get
    STRICTER, never looser, so this never blocks an honest model on a thin/early
    run.
    """
    from collections import Counter
    rows = [np.asarray(r, dtype=np.float64) for r in block_rows if r is not None]
    rows = [r for r in rows
            if r.ndim == 1 and r.size >= 2 and np.all(np.isfinite(r)) and r.std() > 1e-12]
    if len(rows) < 2:
        return None
    width = Counter(r.size for r in rows).most_common(1)[0][0]  # modal length
    rows = [r for r in rows if r.size == width]
    if len(rows) < 2 or width < n_groups or width % n_groups != 0:
        return None
    return pbo_cscv(np.stack(rows), n_groups=n_groups)
