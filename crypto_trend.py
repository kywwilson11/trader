"""BTC-native trend / TSMOM risk-off gate for the crypto book (wave-9 #7).

The crypto book has no market-wide regime gate: risk-off keys off EQUITY-vol
proxies (VIX/STLFSI2) + a stablecoin-peg halt, all stale on a calm-VIX,
no-depeg, multi-week BTC bleed. On the ~50bps-round-trip book where all coins
move 0.6-0.9 correlated, round-tripping into a downtrend is pure cost bleed.

A graded BTC 200h-SMA gap, debounced with an asymmetric Schmitt trigger (de-risk
FAST, re-arm SLOW) + N-bar persistence, shrinks every crypto entry in risk-off.
It is COST-POSITIVE by construction (it only suppresses/shrinks entries, never
adds a trade) and long-only-safe.

HONEST scope: largely redundant with the shipped realized-vol scaling + the
stablecoin halt + the VIX gate, so the unique edge is a thin, rare residual — a
Sharpe/survival hygiene item, not a P&L driver. The whole point of the
hysteresis is that bare TSMOM is often OOS-negative (whipsaw tax). Validate the
co-fire counterfactual before flipping it on. Pure numpy — Mac-testable.

References: Moskowitz-Ooi-Pedersen 2012 (TSMOM); Detzel et al. 2021 (MAs predict
BTC; rational-learning equilibrium); Faber 2007 (200d SMA cut max-drawdown).
"""
import numpy as np


def sma_gap(closes, window=200):
    """(last close - trailing SMA(window)) / SMA, or None if < window bars."""
    c = np.asarray(closes, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < window:
        return None
    sma = float(c[-window:].mean())
    if sma <= 0:
        return None
    return (float(c[-1]) - sma) / sma


def trend_scalar(gap, floor=0.5, scale=0.05):
    """Graded de-risk scalar in [floor, 1.0] from the SMA gap (tanh transition).

    Deep above the SMA -> ~1.0 (full size); deep below -> floor; at the SMA ~mid.
    Fail-OPEN to 1.0 on a missing/non-finite gap (never silently de-risk on a data
    gap). Floored at 0.5 so it composes with the existing 0.1 tilt floor without
    stacking to a near-zero size.
    """
    if gap is None or not np.isfinite(gap):
        return 1.0
    s = 0.5 * (np.tanh(float(gap) / scale) + 1.0)      # 0..1
    return float(np.clip(floor + (1.0 - floor) * s, floor, 1.0))


def hysteresis_state(gap, prev_state='risk_on', b_lo=-0.02, b_hi=0.01):
    """Asymmetric Schmitt trigger: flip to 'risk_off' when gap < b_lo (de-risk
    FAST), back to 'risk_on' only when gap > b_hi (re-arm SLOW). Between the
    bands hold the prior state — no whipsaw. Fail-open: hold prev on bad input."""
    if gap is None or not np.isfinite(gap):
        return prev_state
    if gap < b_lo:
        return 'risk_off'
    if gap > b_hi:
        return 'risk_on'
    return prev_state


def smooth_state(raw_states, persistence=3):
    """Require `persistence` consecutive raw states before committing a switch —
    a single-bar flip never moves the committed state (whipsaw guard)."""
    out = []
    if not raw_states:
        return out
    committed = raw_states[0]
    candidate = committed
    count = 0
    for s in raw_states:
        if s == committed:
            candidate, count = committed, 0
        elif s == candidate:
            count += 1
            if count >= persistence:
                committed, candidate, count = candidate, candidate, 0
        else:
            candidate, count = s, 1
        out.append(committed)
    return out
