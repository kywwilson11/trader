"""Correlation-aware portfolio management.

Three components, all driven by the same pairwise-|corr| inputs:

1. Correlation gate + sizing (Markowitz 1990 Nobel): reject candidates whose
   avg pairwise correlation with the book exceeds MAX_AVG_CORRELATION, and
   shrink entries by 1/sqrt(1 + n*rho_bar) so a clone's marginal RISK stays
   constant as the book fills (check_portfolio_correlation,
   get_correlation_sizing_factor).

2. Equicorrelation ENB book-risk kernels (diversified_book_risk,
   book_risk_budget): closed-form diversified book stop-risk and the max risk
   a new position may add under the book cap. risk_budget.py imports these to
   net BOTH books at the account level.

3. Account-level realized-vol scalar (get_book_vol_scalar_cached): de-risk-only
   entry multiplier from the EWMA vol of the shared account equity curve.
"""

import time
import numpy as np
from log_config import get_logger

logger = get_logger(__name__)

# Cache (per asset class — the combined bot runner shares this module
# between the crypto and stock threads)
_corr_cache: dict[str, tuple[dict, float]] = {}
_CORR_CACHE_TTL = 3600  # 1 hour

MAX_AVG_CORRELATION = 0.7  # reject if avg pairwise > this

# NOTE: live policy threshold local to this module — NOT in strategy_config.py
# (the repo's declared policy source of truth). Relocating it is an owner call.

# |daily equity return| above this is almost certainly a deposit/withdrawal or
# paper-account reset, not P&L. Mirrors beta_ledger.OUTLIER_DAILY_RETURN (0.15);
# kept as a local copy — never import beta_ledger from a live sizing path.
OUTLIER_DAILY_RETURN = 0.15

# Lazily-resolved LedoitWolf class: None = unresolved, False = sklearn absent.
# Resolved on FIRST use, never at import time — portfolio.py is imported at
# module scope by risk_budget.py, and an eager sklearn import would drag
# ~100 MB RSS into every process that merely touches the risk layer on the
# 8 GB Jetson. Tests may monkeypatch this to False (force the corrcoef path
# deterministically on any machine) or to a stub class.
_LedoitWolf = None


def _resolve_ledoit_wolf():
    """Resolve the optional sklearn LedoitWolf estimator exactly once."""
    global _LedoitWolf
    if _LedoitWolf is None:
        try:
            from sklearn.covariance import LedoitWolf as _LW
            _LedoitWolf = _LW
        except Exception:
            _LedoitWolf = False
    return _LedoitWolf


# Diagnostics from the most recent compute_correlation_matrix call.
# Instrumentation only; shared by both loop threads — races merely garble a
# log field, never a returned value.
_last_corr_diag: dict = {}

# Correlation entry-gate counters (instrumentation only; in-process; counter
# races between the two loop threads can drop an increment — acceptable).
_gate_stats = {'n_checked': 0, 'n_rejected': 0,
               'sum_avg_corr': 0.0, 'max_avg_corr_seen': 0.0}


def correlation_gate_stats(reset: bool = False) -> dict:
    """Snapshot of the in-process correlation-gate counters.

    The gate was the one live rejector with no attribution data: both call
    sites discard avg_corr, so MAX_AVG_CORRELATION had no calibration record.
    """
    snap = dict(_gate_stats)
    if reset:
        _gate_stats['n_checked'] = 0
        _gate_stats['n_rejected'] = 0
        _gate_stats['sum_avg_corr'] = 0.0
        _gate_stats['max_avg_corr_seen'] = 0.0
    return snap


# Rate-limit for the avg_book_correlation coverage warning (it runs per entry
# candidate per cycle; warn at most once per 600 s).
_bookcorr_warn_ts = 0.0


def get_correlation_matrix_cached(api, symbols, asset_type='crypto'):
    """Correlation matrix with a 1h TTL cache.

    WARNING: the cache key is asset_type ONLY — `symbols` is ignored on a
    cache hit. Callers MUST pass the full universe (as
    base_loop._update_correlations does): a subset passed first would pin a
    tiny matrix as the asset class's matrix for the whole TTL, and every
    other pair lookup would silently fail open to 0.0.

    Pairwise correlations over 30 bars barely move cycle to cycle; the old
    path serially refetched the whole universe and re-ran ~1,000
    LedoitWolf fits every 10th cycle while this cache sat unused.
    """
    now = time.monotonic()
    hit = _corr_cache.get(asset_type)
    if hit is not None and (now - hit[1]) < _CORR_CACHE_TTL:
        return hit[0]
    t0 = time.monotonic()
    returns_dict = get_returns_for_symbols(api, symbols, asset_type)
    corr = compute_correlation_matrix(returns_dict) if returns_dict else {}
    n_req = len(symbols) if symbols else 0
    n_got = len(returns_dict)
    if n_req and n_got < 0.9 * n_req:
        missing = [s for s in symbols if s not in returns_dict]
        logger.warning(
            "[PORTFOLIO] %s corr rebuild: only %d/%d symbols fetched — "
            "gate/haircut are neutral (corr 0.0) for the missing names: %s",
            asset_type, n_got, n_req, ', '.join(missing[:8]))
    logger.info(
        "[PORTFOLIO] %s corr rebuild: %d/%d symbols, %d pair entries, "
        "%.1fs, estimator=%s", asset_type, n_got, n_req, len(corr),
        time.monotonic() - t0,
        _last_corr_diag.get('estimator', 'n/a') if corr else 'n/a')
    if corr:
        _corr_cache[asset_type] = (corr, now)
    elif hit is not None:
        logger.warning(
            "[PORTFOLIO] %s corr rebuild returned nothing — callers keep a "
            "matrix built %.0f min ago", asset_type, (now - hit[1]) / 60.0)
    return corr


def compute_correlation_matrix(returns_dict: dict[str, np.ndarray],
                               window: int = 30) -> dict[tuple[str, str], float]:
    """Compute pairwise correlation from recent returns.

    Uses LedoitWolf-shrunk covariance when sklearn is importable (resolved once
    via the module-level _LedoitWolf sentinel), else falls back to np.corrcoef.
    The two estimators are NOT numerically interchangeable — pairwise (p=2)
    LedoitWolf shrinks correlations toward zero, more so for unequal-vol
    pairs; the estimator actually used is recorded in _last_corr_diag and
    logged once per rebuild. Only OFF-DIAGONAL pairs are written (both (s1,s2) and
    (s2,s1)); the diagonal (sym,sym) is intentionally absent, so a self-pair
    lookup falls to the caller's 0.0 default — which under-counts a symbol's
    correlation with itself on add-on entries (see the deferred self-pair item).

    Args:
        returns_dict: {symbol: array_of_returns}
        window: Number of recent bars to use

    Returns:
        Dict mapping (sym1, sym2) -> correlation coefficient
    """
    symbols = sorted(returns_dict.keys())
    corr = {}
    windows = {s: returns_dict[s][-window:] for s in symbols}
    LW = _resolve_ledoit_wolf()
    n_pairs = 0
    n_short = 0
    n_fallback = 0     # LedoitWolf fit failed for the pair -> corrcoef
    n_degenerate = 0   # zero-variance / undefined correlation -> 0.0

    for i, s1 in enumerate(symbols):
        r1 = windows[s1]
        for s2 in symbols[i + 1:]:
            r2 = windows[s2]
            min_len = min(len(r1), len(r2))
            n_pairs += 1
            # Unreachable from the production caller: get_returns_for_symbols
            # admits a symbol only when len(df) > 10, which yields >= 10
            # returns after pct_change().dropna(). Defensive for direct
            # calls only; the REACHABLE degenerate path is a constant series
            # -> NaN -> 0.0 below.
            if min_len < 10:
                n_short += 1
                corr[(s1, s2)] = 0.0
                corr[(s2, s1)] = 0.0
                continue
            c = None
            if LW is not False:
                try:
                    X = np.column_stack([r1[-min_len:], r2[-min_len:]])
                    lw = LW().fit(X)
                    cov = lw.covariance_
                    std = np.sqrt(np.diag(cov))
                    if std[0] > 0 and std[1] > 0:
                        c = cov[0, 1] / (std[0] * std[1])
                    else:
                        c = 0.0
                        n_degenerate += 1
                except Exception:
                    c = None
                    n_fallback += 1
            if c is None:
                with np.errstate(invalid='ignore', divide='ignore'):
                    c = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
            if np.isnan(c):
                n_degenerate += 1
                c = 0.0
            corr[(s1, s2)] = c
            corr[(s2, s1)] = c

    _last_corr_diag.clear()
    _last_corr_diag.update({
        'estimator': 'ledoit-wolf' if LW is not False else 'corrcoef',
        'n_pairs': n_pairs, 'n_short': n_short,
        'n_fallback': n_fallback, 'n_degenerate': n_degenerate,
    })
    if n_degenerate:
        logger.warning(
            "[PORTFOLIO] %d/%d pairs had zero-variance or undefined "
            "correlation (halted/stale feed?) — recorded as 0.0, which the "
            "gate and sizing read as perfectly diversifying",
            n_degenerate, n_pairs)

    return corr


def get_returns_for_symbols(api, symbols: list[str],
                            asset_type: str = 'crypto') -> dict[str, np.ndarray]:
    """Fetch recent returns for a list of symbols.

    Args:
        api: Alpaca API client
        symbols: List of symbols
        asset_type: 'crypto' or 'stock'

    Returns:
        {symbol: array_of_pct_returns}
    """
    from market_data import (fetch_bars_alpaca, fetch_stock_bars_alpaca,
                             closed_bars_v2_enabled)

    returns_dict = {}
    _closed = closed_bars_v2_enabled()
    for sym in symbols:
        try:
            if asset_type == 'crypto':
                df = fetch_bars_alpaca(api, sym, closed_only=_closed)
            else:
                df = fetch_stock_bars_alpaca(api, sym, closed_only=_closed)
            if df is not None and len(df) > 10:
                # pandas pin (c26-T3/B21): explicit ffill + fill_method=None
                # == pandas-2 pad semantics, pandas-3-proof
                rets = df['Close'].ffill().pct_change(fill_method=None).dropna().values
                returns_dict[sym] = rets
        except Exception as e:
            logger.debug("Correlation: failed to fetch %s: %s", sym, e)

    return returns_dict


def _avg_abs_corr(candidate: str,
                  current_positions: list[str],
                  corr_matrix: dict[tuple[str, str], float]) -> float:
    """Mean |corr| between a candidate and each held name.

    Shared by the entry gate and the sizing haircut so they can never drift
    (the parked self-pair decision lands in exactly one place). Missing pairs
    count as 0.0 in the mean (fail-open — parked decision-queue item). abs()
    means a negatively-correlated hedge scores like a clone; deliberate for
    now, flagged as an owner decision before any hedging work.
    """
    correlations = []
    for sym in current_positions:
        c = corr_matrix.get((sym, candidate),
                            corr_matrix.get((candidate, sym), 0.0))
        correlations.append(abs(c))
    return float(np.mean(correlations))


def check_portfolio_correlation(current_positions: list[str],
                                candidate: str,
                                corr_matrix: dict[tuple[str, str], float],
                                max_avg_corr: float = MAX_AVG_CORRELATION) -> tuple[bool, float]:
    """Check if adding a candidate would push avg pairwise correlation too high.

    Args:
        current_positions: List of currently held symbols
        candidate: Symbol we want to add
        corr_matrix: Precomputed correlation dict
        max_avg_corr: Maximum allowed average pairwise correlation

    Returns:
        (allowed: bool, avg_corr: float)

    Missing pairs average in as 0.0 and abs() treats hedges as clones (see _avg_abs_corr).
    """
    if not current_positions:
        return True, 0.0

    avg_corr = _avg_abs_corr(candidate, current_positions, corr_matrix)

    _gate_stats['n_checked'] += 1
    _gate_stats['sum_avg_corr'] += avg_corr
    if avg_corr > _gate_stats['max_avg_corr_seen']:
        _gate_stats['max_avg_corr_seen'] = avg_corr

    if avg_corr > max_avg_corr:
        _gate_stats['n_rejected'] += 1
        logger.info("[PORTFOLIO] %s rejected: avg corr %.2f > %.2f with %s "
                    "(gate: %d rejected / %d checked this process)",
                    candidate, avg_corr, max_avg_corr,
                    ', '.join(current_positions),
                    _gate_stats['n_rejected'], _gate_stats['n_checked'])
        return False, avg_corr

    if avg_corr >= 0.75 * max_avg_corr:
        logger.debug("[PORTFOLIO] %s accepted near the corr cap: avg %.2f "
                     "vs %.2f with %d held", candidate, avg_corr,
                     max_avg_corr, len(current_positions))
    return True, avg_corr


def get_correlation_sizing_factor(candidate: str,
                                  current_positions: list[str],
                                  corr_matrix: dict[tuple[str, str], float]) -> float:
    """Variance-correct sizing factor: 1/sqrt(1 + n*rho_bar).

    In an equicorrelated book of n+1 equal positions at average |corr|
    rho_bar, TOTAL book variance is (n+1)*(1 + n*rho_bar) times one
    standalone variance, so each position's AVERAGE share is (1 + n*rho_bar);
    scaling size by the inverse square root equalizes that average share to
    standalone. (The strict MARGINAL variance a candidate adds is
    (1 + 2*n*rho_bar) — this normalization is the intentionally gentler of
    the two; switching is an owner decision.) The old linear heuristic
    (1 - 0.5*corr) ignored n entirely — the 5th BTC-clone got the same
    haircut as the 1st. Floored at 0.4. Missing pairs average in as 0.0;
    abs() treats hedges like clones (see _avg_abs_corr).
    """
    if not current_positions:
        return 1.0

    avg_corr = _avg_abs_corr(candidate, current_positions, corr_matrix)
    n = len(current_positions)
    return max(0.4, 1.0 / float(np.sqrt(1.0 + n * avg_corr)))


def avg_book_correlation(symbols: list[str],
                         corr_matrix: dict[tuple[str, str], float]) -> float:
    """Average absolute pairwise correlation across a set of symbols.

    Missing pairs are EXCLUDED from the mean (unlike the gate/sizing, which
    default them to 0.0), and a non-empty matrix covering NONE of the book's
    pairs returns 0.0 — callers guarding only on matrix truthiness never see
    their no-data prior (owner decision parked). Coverage gaps are warned
    below, rate-limited.
    """
    if len(symbols) < 2 or not corr_matrix:
        return 0.0
    vals = []
    n_expected = 0
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i + 1:]:
            c = corr_matrix.get((s1, s2), corr_matrix.get((s2, s1)))
            if s1 != s2:
                # (sym,sym) duplicates from pyramiding candidates hit the
                # documented-absent diagonal; not counted as missing.
                n_expected += 1
            if c is not None:
                vals.append(abs(c))
    rho = float(np.mean(vals)) if vals else 0.0
    if n_expected and len(vals) < n_expected:
        global _bookcorr_warn_ts
        now = time.monotonic()
        if now - _bookcorr_warn_ts > 600:
            _bookcorr_warn_ts = now
            logger.warning(
                "[PORTFOLIO] avg_book_correlation: %d of %d pairs missing "
                "from a %d-entry matrix for %s — rho=%.2f may be spuriously "
                "low (missing pairs are EXCLUDED here; a fully-uncovered "
                "book returns 0.0, bypassing the caller's 0.5 no-data "
                "prior)", n_expected - len(vals), n_expected,
                len(corr_matrix), ','.join(symbols[:8]), rho)
    return rho


def _clip_rho(avg_corr) -> float:
    """Clip rho to [0, 1]; non-finite input fails CLOSED to 1.0 (lockstep).

    Python's min/max propagate NaN, so the previous inline clip could return
    NaN — poisoning book risk and making base_loop's `cand_risk > budget`
    comparison silently False (cap skipped). Unreachable from in-repo callers
    today (compute_correlation_matrix NaN-guards its output; base_loop falls
    back to 0.5), so this is value-preserving on every reachable path.
    Mirrors risk_budget.py's finite-guard convention.
    """
    v = float(avg_corr)
    if not np.isfinite(v):
        return 1.0
    return min(max(v, 0.0), 1.0)


def diversified_book_risk(risks: list[float], avg_corr: float) -> float:
    """Book stop-risk under the equicorrelation model.

    With per-position stop-risks r_i (fractions of equity) and constant
    pairwise correlation rho, portfolio risk is exactly
        sqrt((1-rho) * sum(r_i^2) + rho * (sum r_i)^2)
    — between sqrt-sum-of-squares (independent book) and the plain sum
    (lockstep book). This is the effective-number-of-bets view:
    correlated books get less headroom for the same gross risk.
    """
    r = np.asarray([x for x in risks if x and x > 0], dtype=float)
    if r.size == 0:
        return 0.0
    rho = _clip_rho(avg_corr)
    return float(np.sqrt(max(
        (1.0 - rho) * np.sum(r ** 2) + rho * np.sum(r) ** 2, 0.0)))


def book_risk_budget(existing_risks: list[float], avg_corr: float,
                     cap: float) -> float:
    """Max stop-risk (fraction of equity) a NEW position may add.

    Solves diversified_book_risk(existing + [r_c]) = cap for r_c under
    equicorrelation: with S1 = sum(r_i) and A = current book risk^2,
        r_c = -rho*S1 + sqrt(rho^2*S1^2 + cap^2 - A)
    Returns 0 when the cap is already used up.
    """
    cap = float(cap)
    if not np.isfinite(cap) or cap <= 0:
        # Fail closed on a nonsensical cap: cap**2 discards the sign, so a
        # negative cap previously returned a POSITIVE budget. Matches
        # risk_budget.account_risk_budget's guard convention.
        return 0.0
    rho = _clip_rho(avg_corr)
    r = np.asarray([x for x in existing_risks if x and x > 0], dtype=float)
    s1 = float(np.sum(r))
    a = (1.0 - rho) * float(np.sum(r ** 2)) + rho * s1 ** 2  # book risk^2
    if cap ** 2 <= a:
        return 0.0
    return float(-rho * s1 + np.sqrt(rho ** 2 * s1 ** 2 + cap ** 2 - a))


# --- Account-level realized-vol scalar, applied per-book against per-book
# targets (closes the loop that per-asset GARCH targeting leaves open:
# correlation buildup raises BOOK vol even when each position individually
# sits at its own target). api.get_portfolio_history is ACCOUNT-level — both
# books share one equity curve, so crypto-book vol de-risks stock entries and
# vice versa. ---

# The two per-book targets divide the SAME account curve, so the coupling is
# threshold-asymmetric: stock (target 0.18) starts de-risking at roughly half
# the account vol crypto (0.35) needs — "vice versa" above is
# mechanism-symmetric only. PORTFOLIO_VOL_TARGET also drives per-position
# GARCH targeting in volatility.compute_vol_adjusted_size (a second
# multiplier on the same sized product); scope/composition: resolved under
# strategy_config.DERISK_STACK_V2 — this book-level scalar owns
# PORTFOLIO_VOL_TARGET (inside the regime-family MIN) and base_loop composes
# the per-position GARCH ratio at 1.0; legacy (flag OFF) still multiplies both.

_book_vol_cache: dict[str, tuple[float, float]] = {}
_BOOK_VOL_TTL = 3600

EWMA_LAMBDA = 0.94          # RiskMetrics daily decay
# Annualization assumes ~252 daily equity points/year. If Alpaca's 1D
# portfolio history includes weekend points for a crypto-holding account
# (verify from a real payload's timestamps on the Jetson), 365 points/year
# would make this understate realized vol ~17% (sqrt(252/365)) and
# under-de-risk the crypto book — switch to 365 if weekends are present.
_TRADING_DAYS = 252

# Cross-refs: volatility.BARS_PER_YEAR['crypto'] = 8760 already commits the
# same book to a 365-day year, and `python beta_ledger.py --days 90` on the
# Jetson prints obs_per_year for this exact series; the [BOOK-VOL] provenance
# log line below also reports median point spacing and weekend presence.


def _ewma_vol_diag(equity_curve, lam: float = EWMA_LAMBDA,
                   exclude_outliers: bool = False):
    """(annualized_vol_or_None, diagnostics) — see ewma_annualized_vol.

    Diagnostics (instrumentation only): n_raw, n_used, n_dropped, n_returns,
    n_outliers (|ret| > OUTLIER_DAILY_RETURN), max_abs_ret.

    exclude_outliers (D29, deposit contamination): the input is the RAW
    account equity curve, so a deposit/withdrawal or paper-account reset
    prints as a huge fake "return" that inflates the EWMA for the whole 3M
    window and pins the book scalar at its 0.5 floor. When True (wired to
    strategy_config.DERISK_STACK_V2 by get_book_vol_scalar_cached), equity
    points are pre-filtered finite+positive (beta_ledger pattern) and
    |return| > OUTLIER_DAILY_RETURN observations are EXCLUDED from both the
    variance seed and the EWMA recursion; diag gains 'n_excluded'.
    n_outliers/max_abs_ret keep reporting the FULL return series so the
    always-on contamination warning still sees them. Default False is
    byte-identical to the legacy path.
    """
    diag = {'n_raw': 0, 'n_used': 0, 'n_dropped': 0, 'n_returns': 0,
            'n_outliers': 0, 'max_abs_ret': 0.0}
    src = [] if equity_curve is None else list(equity_curve)
    diag['n_raw'] = len(src)
    eq = np.asarray([e for e in src if e], dtype=float)
    if exclude_outliers:
        eq = eq[np.isfinite(eq) & (eq > 0)]
    diag['n_used'] = int(eq.size)
    diag['n_dropped'] = diag['n_raw'] - diag['n_used']
    if eq.size < 11:
        return None, diag
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]
    diag['n_returns'] = int(rets.size)
    if rets.size:
        abs_rets = np.abs(rets)
        diag['max_abs_ret'] = float(abs_rets.max())
        diag['n_outliers'] = int(np.sum(abs_rets > OUTLIER_DAILY_RETURN))
    rets_used = rets
    if exclude_outliers:
        rets_used = rets[np.abs(rets) <= OUTLIER_DAILY_RETURN]
        diag['n_excluded'] = int(rets.size - rets_used.size)
    if rets_used.size < 10:
        return None, diag
    var = float(np.var(rets_used))
    for x in rets_used:
        var = lam * var + (1.0 - lam) * x * x
    return float(np.sqrt(var * _TRADING_DAYS)), diag


def ewma_annualized_vol(equity_curve, lam: float = EWMA_LAMBDA,
                        exclude_outliers: bool = False) -> float | None:
    """Annualized EWMA volatility of a daily equity curve, or None.

    Accepts any iterable of numbers (list/tuple/ndarray/Series) or None —
    the old `equity_curve or []` truthiness raised ValueError on arrays.
    Falsy points (None, 0, 0.0) are dropped and the survivors spliced, so a
    gap becomes a 1-day return. The recursion is SEEDED with the full-window
    sample variance, which retains weight lam**n in the result (54% at the
    10-return minimum, 2% at a full 3M window) — short windows are closer to
    a plain sample variance than an EWMA. exclude_outliers: see
    _ewma_vol_diag (D29 deposit-contamination exclusion; default False is
    byte-identical).
    """
    return _ewma_vol_diag(equity_curve, lam, exclude_outliers)[0]


def _timestamp_diag(timestamps):
    """(median_spacing_days | None, has_weekend_points | None) — best-effort.

    Settles the 252-vs-365 annualization question from production logs (see
    the _TRADING_DAYS comment) and lets the outlier warning be dated. Never
    raises; (None, None) on anything unparseable.
    """
    try:
        if timestamps is None:
            return None, None
        ts = np.asarray(list(timestamps), dtype=float)
        if ts.size < 3:
            return None, None
        spacing = float(np.median(np.diff(ts))) / 86400.0
        has_weekend = None
        if ts[0] > 1e9:  # plausible epoch-seconds
            from datetime import datetime, timezone
            has_weekend = any(
                datetime.fromtimestamp(t, tz=timezone.utc).weekday() >= 5
                for t in ts)
        return spacing, has_weekend
    except Exception:
        return None, None


def get_book_vol_scalar_cached(api, asset_type: str = 'crypto') -> float:
    """De-risk-only sizing scalar from REALIZED account vol (EWMA λ=0.94).

    Moreira-Muir (2017): volatility-managed sizing adds value mostly by
    CUTTING exposure when realized vol runs hot — so this clamps to
    [0.5, 1.0] and never boosts (a mostly-cash paper account shows tiny
    realized vol; boosting on that signal would be spurious). Neutral 1.0
    on any data failure. Cached 1h per asset class.

    Composition caveats (owner decisions parked): base_loop multiplies this
    into a tilt product that is then clamped to TILT_MAX, so on high-tilt
    candidates part or all of the cut can be absorbed by the clamp (compare
    detail['tilt_raw'] vs detail['tilt'] in the conviction journal). The
    input is the RAW account equity curve — deposits/withdrawals read as
    P&L (see the outlier warning) — and the same PORTFOLIO_VOL_TARGET also
    drives per-position GARCH targeting in volatility.py.
    """
    now = time.monotonic()
    hit = _book_vol_cache.get(asset_type)
    if hit is not None and (now - hit[1]) < _BOOK_VOL_TTL:
        return hit[0]
    scalar = 1.0
    try:
        try:
            from strategy_config import DERISK_STACK_V2
        except ImportError:
            DERISK_STACK_V2 = False
        hist = api.get_portfolio_history(period='3M', timeframe='1D')
        realized, diag = _ewma_vol_diag(getattr(hist, 'equity', None),
                                        exclude_outliers=DERISK_STACK_V2)
        spacing_days, has_weekend = _timestamp_diag(
            getattr(hist, 'timestamp', None))
        if diag['n_outliers'] > 0:
            logger.warning(
                "[BOOK-VOL] %s: %d day(s) with |daily return| > %.0f%% "
                "(max %.1f%%) in the 3M equity window — possible "
                "deposit/withdrawal or paper-account reset, not P&L; "
                "realized vol and this sizing scalar are contaminated%s",
                asset_type, diag['n_outliers'], OUTLIER_DAILY_RETURN * 100,
                diag['max_abs_ret'] * 100,
                (" (DERISK_STACK_V2: excluded from the EWMA recursion)"
                 if DERISK_STACK_V2 else ""))
        if realized is not None and realized > 1e-6:
            from strategy_config import PORTFOLIO_VOL_TARGET
            if asset_type not in PORTFOLIO_VOL_TARGET:
                logger.warning(
                    "[BOOK-VOL] unknown asset_type %r — falling back to "
                    "target 0.25 (an arbitrary midpoint, not from "
                    "strategy_config)", asset_type)
            target = PORTFOLIO_VOL_TARGET.get(asset_type, 0.25)
            scalar = min(max(target / realized, 0.5), 1.0)
            if scalar < 1.0:
                logger.info("[BOOK-VOL] %s: realized %.0f%% > target %.0f%% "
                            "-> %.2fx entries", asset_type, realized * 100,
                            target * 100, scalar)
        elif realized is None:
            logger.warning(
                "[BOOK-VOL] %s: insufficient equity history (%d usable of "
                "%d points) — scalar neutral 1.0, de-risk layer inactive",
                asset_type, diag['n_used'], diag['n_raw'])
        logger.info(
            "[BOOK-VOL] %s: series=portfolio_history(3M,1D) points=%d/%d "
            "dropped=%d outliers=%d outliers_excluded=%s spacing_days=%s "
            "weekend_points=%s realized=%s target-scope=book -> scalar %.2f",
            asset_type, diag['n_used'], diag['n_raw'], diag['n_dropped'],
            diag['n_outliers'], DERISK_STACK_V2,
            ('%.2f' % spacing_days) if spacing_days is not None else 'n/a',
            has_weekend if has_weekend is not None else 'n/a',
            ('%.1f%%' % (realized * 100)) if realized is not None else 'n/a',
            scalar)
    except Exception as e:
        logger.warning("[BOOK-VOL] %s: portfolio history unavailable (%s) — "
                       "de-risk layer neutral 1.0 pinned for %ds",
                       asset_type, e, _BOOK_VOL_TTL)
    _book_vol_cache[asset_type] = (scalar, now)
    return scalar
