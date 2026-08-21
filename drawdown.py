"""Account drawdown de-leveraging ladder + high-water-mark persistence.

Pure functions extracted from base_loop so the restart-survival logic and the
size-ladder arithmetic are unit-testable — base_loop pulls torch/joblib
transitively and cannot import on the dev Mac. The live loop calls these, so the
tested math IS the traded math.

The bug they close (wave-8 #4): base_loop._peak_equity seeded a hardcoded 100k
and was never persisted. After a restart mid-drawdown — routine on an 8GB Jetson
(retrain / OOM / power) — the peak reset to ~current equity, the drawdown snapped
to ~0, and the 25/50/75% de-leveraging ladder silently DISABLED exactly when the
account was underwater (a restarted bot at 15% DD sized 1.0x instead of 0.50x).

Grossman & Zhou (1993): drawdown-control sizing is defined relative to the
running high-water mark — so the HWM MUST persist across sessions for the rule to
be well-defined.
"""

# (drawdown threshold, size multiplier), richest drawdown first.
# Under strategy_config.DERISK_STACK_V2 this ladder deliberately stays OUTSIDE
# the regime-family MIN and keeps composing as a product — it measures the
# ACCOUNT's own state (Grossman-Zhou 1993), not market vol (02_research B06).
DRAWDOWN_LADDER = ((0.20, 0.25), (0.15, 0.50), (0.10, 0.75))

PEAK_SEED = 100_000.0


def update_peak_equity(prev_peak, current_equity):
    """Monotone up-ratchet of the high-water mark (never decreases)."""
    try:
        return max(float(prev_peak), float(current_equity))
    except (TypeError, ValueError):
        return prev_peak


def restore_peak_equity(saved_peak, current_equity, seed=PEAK_SEED):
    """Peak to adopt after a restart.

    max(persisted peak, current equity) — never below current, so a higher prior
    peak survives and the ladder stays ARMED while underwater. Falls back to
    max(seed, current) when the saved value is missing / non-finite / non-positive
    (legacy state files predate this field). Pair with update_peak_equity, which
    only ratchets UP, so a lower current equity never clobbers the restored peak.
    """
    try:
        cur = float(current_equity)
    except (TypeError, ValueError):
        cur = 0.0
    try:
        sp = float(saved_peak)
        # Totality: rejects NaN (all comparisons false), non-positive, and
        # +inf. json round-trips Infinity, so a corrupted state file CAN hand
        # us inf — which max() would then pin forever, silently disabling the
        # ladder (drawdown_fraction(inf, e) -> nan -> floored to 0).
        if not (0.0 < sp < float('inf')):
            sp = float(seed)
    except (TypeError, ValueError):
        sp = float(seed)
    return max(sp, cur)


def drawdown_fraction(peak, equity):
    """(peak - equity) / peak, floored at 0; returns 0 when peak <= 0."""
    try:
        p = float(peak)
        e = float(equity)
    except (TypeError, ValueError):
        return 0.0
    if p <= 0.0:
        return 0.0
    return max(0.0, (p - e) / p)


def drawdown_size_multiplier(dd, ladder=DRAWDOWN_LADDER):
    """Size multiplier for a drawdown fraction; 1.0 when dd is below the
    shallowest rung."""
    for threshold, mult in ladder:
        if dd >= threshold:
            return mult
    return 1.0
