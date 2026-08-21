"""Edge- and probability-proportional bet sizing (wave-9 #5).

Sizing today collapses to a pooled Kelly multiplier plus a FLAT-TOPPED meta
multiplier (clip(2p, 0.6, 1.3) saturates at 1.3 for every p>=0.65), so a rank-1
high-conviction trade sizes ~the same as a rank-7 marginal one. Kelly (1956):
growth-optimal capital is PROPORTIONAL to edge/odds — the flat top throws that
geometric-growth edge away.

HARD PRECONDITION: these consume a calibrated win probability p, so they are
gated on the wave-9 #1 calibration fix. Kelly over-bets on optimistic p and the
sign flips negative (Chopra-Ziemba 1993: mean/probability errors are ~11x costlier
than variance errors; "slight overbet => eventual ruin"), which is why any live
wiring must be fractional-Kelly + KELLY_CAP + the ENB book cap, and staged.

STATUS: DECLARED-AHEAD, NOT WIRED. No production module imports bet_sizing —
only tests. Activation is gated on EDGE_KELLY_ENABLED (strategy_config.py,
default False), itself hard-gated on META_CALIBRATION_MODE='purged_oof' plus a
Stage-0 rank-gradient measurement; wiring is model-facing (challenger -> shadow
-> promotion path). UNITS DIFFER BY FUNCTION and are NOT interchangeable:
afml_bet_size returns a dimensionless signed size in [-1, 1]; kelly_edge_odds
returns a NOTIONAL fraction of equity (gross leverage — f*·a is the fraction
of equity risked to the stop); the live trading_utils.compute_kelly_fraction is
a fraction-at-risk consumed as a bounded multiplier. Do not cap all three with
the same constant.

Pure numpy/scipy — Mac-testable. References: Kelly 1956; MacLean-Thorp-Ziemba
2011 (fractional Kelly); López de Prado AFML ch.10 (bet size from p:
z=(p-base)/sqrt(p(1-p)), m=2*Phi(z)-1; step discretization; concurrency averaging).
"""
import numpy as np


def afml_bet_size(p, base_rate=0.5, step=0.0):
    """AFML ch.10 bet size in [-1, 1] from a calibrated win probability p.

    z = (p - base_rate)/sqrt(p(1-p)); size = 2*Phi(z) - 1. Centered at base_rate
    (size 0 at the per-book base/break-even probability, NOT hardcoded 0.5), so
    it neither sizes up nor down a coin-flip name. Optional `step` discretizes to
    avoid churning micro-adjustments. Long-only callers take max(0, size).

    RETURNS a dimensionless SIGNED size in [-1, 1] (fraction of the caller's
    maximum position) — not a capital fraction; not interchangeable with
    kelly_edge_odds (notional leverage). WARNING: the base_rate=0.5 DEFAULT is
    AFML's binary-classifier null, NOT this system's economic break-even — with
    tp_rr=2.0 the break-even is breakeven_p(b, a) = 1/3, so a long-only caller
    taking max(0, size) under the default zeroes every +EV trade with p in
    (1/3, 1/2); live wiring must pass the book's base rate explicitly (an open
    owner decision). `step` also acts as a MINIMUM: any |size| < step/2 rounds
    to exactly 0.0 (an entry filter, not just smoothing); a non-finite or
    non-positive step is ignored. Fail-closed: non-finite p sizes exactly 0.0;
    a non-finite base_rate sizes the whole call 0.0; finite base_rate is
    clamped into [1e-6, 1-1e-6] so the size stays monotone in p.
    """
    # ndtr == scipy.stats.norm.cdf bit-identically (norm._cdf IS ndtr), ~250x
    # faster per call, and keeps the ~900-module scipy.stats import off the
    # Jetson live path. Import stays local so importing bet_sizing costs no scipy.
    from scipy.special import ndtr
    p = np.asarray(p, float)
    try:
        br = float(base_rate)
    except (TypeError, ValueError):
        br = float('nan')
    if not np.isfinite(br):
        # Fail closed: an unknown base rate must not produce a bet.
        return 0.0 if np.ndim(p) == 0 else np.zeros_like(p)
    br = min(max(br, 1e-6), 1.0 - 1e-6)
    # Fail closed (matches kelly_edge_odds / breakeven_p): a non-finite p
    # maps to base_rate -> z=0 -> size exactly 0.0, instead of propagating
    # NaN through Phi into the returned size.
    p = np.where(np.isfinite(p), p, br)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (p - br) / np.sqrt(p * (1 - p))
    m = 2.0 * ndtr(z) - 1.0
    if step and np.isfinite(step) and step > 0:
        m = np.round(m / step) * step
        # AFML clips AFTER rounding: steps that don't divide 1 evenly overshoot.
        m = np.clip(m, -1.0, 1.0)
    return float(m) if np.ndim(m) == 0 else m


def kelly_edge_odds(p, b, a, fraction=1.0, cap=1.0):
    """Fractional Kelly fraction for a bet that wins b or loses a (fractions>0).

    f* = p/a - (1-p)/b  (the asymmetric-payoff Kelly), so f*=0 exactly at the
    break-even p = a/(a+b). Scaled by `fraction` (e.g. 0.5 = half-Kelly) and
    clipped to [0, cap] for long-only. b is the take-profit move, a the stop
    move — ARGUMENT ORDER IS (p, b, a): win move before loss move.

    UNITS WARNING: returns a NOTIONAL fraction of equity (gross leverage), not
    a risk fraction — f*·a is the fraction of equity risked to the stop. NOT on
    the same scale as trading_utils.compute_kelly_fraction; do not cap both
    with KELLY_CAP. With ATR-scale odds (a ~ 0.01-0.15, b = tp_rr·a) the raw
    f* is O(1/a), so any cap <= 1 binds almost immediately above breakeven and
    the capped output is ~CONSTANT over the whole admitted p range (measured:
    cap=0.25 with a=0.02, b=0.04 returns exactly 0.25 for every p >= 0.337) —
    the cap space for live wiring is an open owner decision; see the
    EDGE_KELLY_ENABLED block in strategy_config.py.

    Vectorized in p AND in the odds (per-name a/b broadcast). Fail-closed:
    non-finite p sizes 0; non-finite or non-positive a/b sizes 0 elementwise;
    non-finite fraction/cap, fraction < 0, or cap <= 0 zeroes the whole call.
    fraction > 1 (super-Kelly) is passed through — `cap` is the only backstop.
    """
    p = np.asarray(p, float)
    # Fail closed: a non-finite probability sizes to 0, out-of-range p is clipped.
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.0)
    try:
        fraction = float(fraction); cap = float(cap)
    except (TypeError, ValueError):
        return 0.0 if np.ndim(p) == 0 else np.zeros_like(p)
    if not (np.isfinite(fraction) and np.isfinite(cap)) or fraction < 0 or cap <= 0:
        return 0.0 if np.ndim(p) == 0 else np.zeros_like(p)
    try:
        a = np.asarray(a, float); b = np.asarray(b, float)
    except (TypeError, ValueError):
        return 0.0 if np.ndim(p) == 0 else np.zeros_like(p)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = (p / a - (1 - p) / b) * fraction
    f = np.clip(np.where(ok, f, 0.0), 0.0, cap)
    return float(f) if np.ndim(f) == 0 else f


def concurrency_scale(n_concurrent):
    """AFML active-bet averaging: K identical concurrent bets each take ~1/K of
    the single-bet Kelly so total exposure stays growth-optimal, not K*f*.

    NOT WIRED, and 1/K is the PERFECT-CORRELATION bound: this system already
    de-risks concurrency twice (portfolio.py's equicorrelation ENB book cap
    under MAX_BOOK_RISK_PCT, and base_loop's correlation sizing factor) — if
    ever wired this must REPLACE one of those, never stack on top. Entry-time
    1/K is also NOT AFML's average-of-active-bets: four sequential fills each
    scaled 1/k at entry hold 1 + 1/2 + 1/3 + 1/4 = 2.08x what the average form
    holds. n is TRUNCATED toward zero by int() (2.9 -> 1/2, not 1/2.9); n <= 1
    returns 1.0. Fail-closed: non-finite or unparseable input returns 1.0
    (no scaling). Changing the truncation is a sizing-value change (owner path).
    """
    try:
        n = float(n_concurrent)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(n):
        return 1.0
    return 1.0 / max(1, int(n))


def breakeven_p(b, a):
    """Win probability at which the edge/odds Kelly fraction is exactly 0.

    ARGUMENT ORDER IS (b = win/take-profit move, a = loss/stop move) — the same
    order as kelly_edge_odds(p, b, a), NOT alphabetical. Returns a/(a+b):
    breakeven_p(0.04, 0.02) == 1/3 for tp_rr=2.0. The REVERSED call silently
    returns b/(a+b) = 2/3 — plausible and wrong; unlike kelly_edge_odds, a
    swap here does not fail closed.

    Degenerate odds (non-finite, a<=0 or b<=0) return 1.0 — the fail-closed
    "unreachable breakeven", consistent with kelly_edge_odds sizing 0 for the
    same inputs. NOTE: 1.0 is an ATTAINABLE probability (kelly_edge_odds clips
    p>1 down to exactly 1.0), so callers must gate with a STRICT comparison
    (p > breakeven), never >=. Vectorized: array odds return an array with 1.0
    in degenerate elements.
    """
    try:
        a = np.asarray(a, float); b = np.asarray(b, float)
    except (TypeError, ValueError):
        return 1.0
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(ok, a / (a + b), 1.0)
    return float(r) if np.ndim(r) == 0 else r
