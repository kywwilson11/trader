"""Edge- and probability-proportional bet sizing (wave-9 #5).

Sizing today collapses to a pooled Kelly multiplier plus a FLAT-TOPPED meta
multiplier (clip(2p, 0.6, 1.3) saturates at 1.3 for every p>=0.65), so a rank-1
high-conviction trade sizes ~the same as a rank-7 marginal one. Kelly (1956):
growth-optimal capital is PROPORTIONAL to edge/odds — the flat top throws that
geometric-growth edge away.

HARD PRECONDITION: these consume a calibrated win probability p, so they are
gated on the wave-9 #1 calibration fix. Kelly over-bets on optimistic p and the
sign flips negative (Chopra-Ziemba 1993: mean/probability errors are ~11x costlier
than variance errors; "slight overbet => eventual ruin"), which is why the live
wiring is fractional-Kelly + KELLY_CAP + the ENB book cap, and staged.

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
    """
    from scipy.stats import norm
    p = np.asarray(p, float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = (p - float(base_rate)) / np.sqrt(p * (1 - p))
    m = 2.0 * norm.cdf(z) - 1.0
    if step and step > 0:
        m = np.round(m / step) * step
        # AFML clips AFTER rounding: steps that don't divide 1 evenly overshoot.
        m = np.clip(m, -1.0, 1.0)
    return float(m) if np.ndim(m) == 0 else m


def kelly_edge_odds(p, b, a, fraction=1.0, cap=1.0):
    """Fractional Kelly fraction for a bet that wins b or loses a (fractions>0).

    f* = p/a - (1-p)/b  (the asymmetric-payoff Kelly), so f*=0 exactly at the
    break-even p = a/(a+b). Scaled by `fraction` (e.g. 0.5 = half-Kelly) and
    clipped to [0, cap] for long-only. b is the take-profit move, a the stop move.
    """
    a = float(a); b = float(b)
    if a <= 0 or b <= 0:
        return 0.0
    p = np.asarray(p, float)
    # Fail closed: a non-finite probability sizes to 0, out-of-range p is clipped.
    p = np.where(np.isfinite(p), np.clip(p, 0.0, 1.0), 0.0)
    f = (p / a - (1 - p) / b) * float(fraction)
    f = np.clip(f, 0.0, float(cap))
    return float(f) if np.ndim(f) == 0 else f


def concurrency_scale(n_concurrent):
    """AFML active-bet averaging: K identical concurrent bets each take ~1/K of
    the single-bet Kelly so total exposure stays growth-optimal, not K*f*."""
    return 1.0 / max(1, int(n_concurrent))


def breakeven_p(b, a):
    """Win probability at which the edge/odds Kelly fraction is exactly 0.

    Degenerate odds (a<=0 or b<=0) return 1.0 — the fail-closed "unreachable
    breakeven", consistent with kelly_edge_odds sizing 0 for the same inputs.
    """
    a = float(a); b = float(b)
    if a <= 0 or b <= 0:
        return 1.0
    return a / (a + b)
