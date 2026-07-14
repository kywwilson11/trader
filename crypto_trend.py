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

STATUS: NOT YET WIRED — no live caller. CRYPTO_TREND_GATE_ENABLED /
CRYPTO_TREND_SMA_WINDOW / CRYPTO_TREND_FLOOR in strategy_config.py are consumed
by NOTHING yet; flipping the flag on the Jetson is currently a silent no-op.
Wiring (CryptoLoop._extra_tilt composing funding_tilt x trend_scalar, bar-fetch
limit >= 220) is the Jetson-gated wave-9 #7 phase 2 — see the review queue.

References: Moskowitz-Ooi-Pedersen 2012 (TSMOM); Detzel et al. 2021 (MAs predict
BTC; rational-learning equilibrium); Faber 2007 (200d SMA cut max-drawdown).
"""
import numpy as np


def sma_gap(closes, window=200):
    """(last close - trailing SMA(window)) / SMA, or None if < window FINITE bars.

    Non-finite closes are compacted out BEFORE windowing: the SMA spans the
    last `window` FINITE closes and the numerator is the last FINITE close.
    So NaN voids inside the window stretch the effective lookback across wall
    time (mixing in older prices), and a trailing-NaN tail leaves the gap
    silently stale — callers cannot distinguish fresh from stale gaps. A
    ~220-bar fetch bounds the stretch: below `window` finite bars this
    returns None and downstream fails open (trend_scalar -> 1.0).
    """
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
    bands hold the prior state — no whipsaw. Fail-HOLD on a missing/non-finite
    gap: keep the prior state (note: NOT fail-open — a stale 'risk_off'
    persists until data returns; the graded trend_scalar path is the one that
    fails open to 1.0)."""
    if gap is None or not np.isfinite(gap):
        return prev_state
    if gap < b_lo:
        return 'risk_off'
    if gap > b_hi:
        return 'risk_on'
    return prev_state


def smooth_state(raw_states, persistence=3):
    """Require `persistence` consecutive raw states before committing a switch —
    a single-bar flip never moves the committed state (whipsaw guard; with
    persistence <= 1 the committed state just tracks the raw states)."""
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
            # Same commit check as above so persistence<=1 commits on THIS
            # bar (count=1 < persistence for any persistence>=2: no-op there)
            if count >= persistence:
                committed, count = candidate, 0
        out.append(committed)
    return out
