"""Free, offline option-pricing + cost-realism harness for the overnight overlay.

The stock book flattens 100% at 15:50 to dodge overnight gap risk, forfeiting
overnight drift. Wave-7 asked: is a cheap, defined-risk DEBIT VERTICAL a better
overnight carrier than flattening? The user rejects a paid OPRA quote feed, so
this module prices options from FREE inputs (Black-Scholes + an IV bootstrapped
from the in-house HAR-RV realized vol) purely to DECIDE — it holds no position
and touches no trading path.

It is deliberately a measurement instrument with a PRE-REGISTERED expectation
of NO-GO: the corrected round-trip friction is ONE full spread per leg per
round trip (sp * sum(leg premiums)), and on the spec-tech tier that is ~half
the debit for a single overnight hold — no realistic overnight edge clears a
MIN_EDGE_MULTIPLE * friction gate. Framing is tail-INSURANCE economics, never
positive carry: the overnight option BUYER loses on average (Muravyev-Ni), so
nothing here may be read as a carry edge.

Everything is pure numpy/scipy, unit-tested against Hull textbook values and
put-call parity. IV is HAR-RV * an empirical IV/RV ratio and EVERY result is
flagged proxy-dependent — refuse to size off it live until real chains
validate the ratio.
"""

import numpy as np
from scipy.stats import norm

# Minimum edge-to-friction multiple an overlay must clear to be worth running.
MIN_EDGE_MULTIPLE = 2.0

# Empirical index-option IV/RV ratio band (variance risk premium): realized
# vol UNDER-states option-implied vol, so HAR-RV must be scaled up. Mid 1.25.
IV_RV_RATIO = (1.1, 1.4)

# Per-name option bid/ask spread as a FRACTION of option premium, by liquidity
# tier (seeded from the equity Eff_Spread_Pct ranking; options are far wider
# than their underlying). One full spread is crossed per leg per round trip.
SPREAD_TIERS = {
    'A': 0.03,   # NVDA/TSLA/AMD/META/COIN — tightest listed-option markets
    'B': 0.06,   # PLTR/HOOD/SOFI/SMCI
    'C': 0.14,   # ASTS/IONQ/QBTS/RKLB/POET — spec-tech, brutally wide
}
TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Black-Scholes (the only pricer; no chains, no OPRA)
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma):
    S = float(S); K = float(K); T = max(float(T), 1e-9); sigma = max(float(sigma), 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, call=True):
    """Black-Scholes European option price. T in YEARS, sigma annualized.

    Degenerates gracefully at expiry (T<=0) to intrinsic value.
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if call else max(K - S, 0.0)
        return float(intrinsic)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if call:
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S, K, T, r, sigma, call=True):
    """delta, gamma, theta (per YEAR), vega (per 1.00 vol). Dict."""
    if T <= 0 or sigma <= 0:
        return {'delta': float((S > K) if call else -(S < K)), 'gamma': 0.0,
                'theta': 0.0, 'vega': 0.0}
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf = norm.pdf(d1)
    gamma = pdf / (S * sigma * np.sqrt(T))
    vega = S * pdf * np.sqrt(T)
    if call:
        delta = norm.cdf(d1)
        theta = (-S * pdf * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (-S * pdf * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return {'delta': float(delta), 'gamma': float(gamma),
            'theta': float(theta), 'vega': float(vega)}


# ---------------------------------------------------------------------------
# Defined-risk vertical (debit spread) — the gap-PROOF structure
# ---------------------------------------------------------------------------

def vertical_debit(S, K1, K2, T, r, sigma, call=True):
    """Net debit (cost) of a debit vertical. call=True -> bull call spread
    (long K1, short K2>K1); call=False -> bear put spread (long K2, short K1).
    Returns the positive premium paid; max loss == this debit (gap-proof)."""
    if K2 <= K1:
        raise ValueError("need K2 > K1")
    if call:
        return bs_price(S, K1, T, r, sigma, True) - bs_price(S, K2, T, r, sigma, True)
    return bs_price(S, K2, T, r, sigma, False) - bs_price(S, K1, T, r, sigma, False)


def vertical_value(S, K1, K2, T, r, sigma, call=True):
    """Mark value of the long vertical at price S / time-to-expiry T."""
    if call:
        return bs_price(S, K1, T, r, sigma, True) - bs_price(S, K2, T, r, sigma, True)
    return bs_price(S, K2, T, r, sigma, False) - bs_price(S, K1, T, r, sigma, False)


def vertical_payoff_at_expiry(S, K1, K2, call=True):
    """Terminal payoff (value, not P&L) of the long vertical at expiry."""
    if call:
        return max(S - K1, 0.0) - max(S - K2, 0.0)
    return max(K2 - S, 0.0) - max(K1 - S, 0.0)


def option_round_trip_cost(leg_premiums, spread_pct_per_leg):
    """Round-trip option friction in PRICE units (same units as the premiums).

    Crossing the bid/ask costs ~half the spread per fill; a round trip is two
    fills per leg, i.e. ONE full spread per leg. So friction = spread_pct *
    sum(|leg premium|), mirroring fees.round_trip_cost_pct's one-spread-per-
    round-trip convention. (The wave-7 finding's 2x figure double-counted.)
    """
    return float(spread_pct_per_leg) * float(np.sum(np.abs(leg_premiums)))


def friction_fraction_per_night(debit, leg_premiums, spread_pct_per_leg,
                                nights_held):
    """Round-trip friction as a FRACTION of the debit, amortized per night.

    This is the number that kills short-dated overlays: a one-night hold pays
    the full round-trip spread once, so friction/night ~ cost/debit; a 14-night
    hold amortizes it 14x. Returns (friction_total_frac, friction_per_night).
    """
    cost = option_round_trip_cost(leg_premiums, spread_pct_per_leg)
    total_frac = cost / max(abs(debit), 1e-9)
    return total_frac, total_frac / max(int(nights_held), 1)


def required_edge_clears(expected_edge_frac, friction_total_frac,
                         min_multiple=MIN_EDGE_MULTIPLE):
    """Does the overlay's expected edge clear MIN_EDGE_MULTIPLE * friction?

    expected_edge_frac and friction_total_frac are both fractions of the debit.
    Returns (clears: bool, required: float). The PRE-REGISTERED expectation is
    that this returns False for every tier on a single overnight hold.
    """
    required = float(min_multiple) * float(friction_total_frac)
    return bool(expected_edge_frac >= required), required


# ---------------------------------------------------------------------------
# IV bootstrap from the in-house HAR-RV (no options data needed)
# ---------------------------------------------------------------------------

def iv_from_har(rv_sigma_annual, iv_rv_ratio=None):
    """Bootstrap an implied vol from HAR-RV realized vol (volatility.get_sigma).

    Returns (iv_low, iv_mid, iv_high) over the IV/RV band. PROXY — flagged
    everywhere; do not size off it live until real chains validate the ratio.
    """
    lo, hi = iv_rv_ratio or IV_RV_RATIO
    mid = 0.5 * (lo + hi)
    s = float(rv_sigma_annual)
    return s * lo, s * mid, s * hi


# ---------------------------------------------------------------------------
# Overnight overlay simulation (entry-time IV held — no realized-vol look-ahead)
# ---------------------------------------------------------------------------

def overnight_overlay_pnl(close_t, open_t1, K1, K2, T_close, overnight_frac,
                          r, sigma_entry, call=True):
    """P&L (price units) of holding the vertical from 15:50 close to next open.

    The vertical is priced at the 15:50 close (S=close_t, T=T_close) and
    re-priced at the next open (S=open_t1, T=T_close-overnight_frac) using the
    SAME entry-time sigma — pricing the open off the realized next-day vol would
    be look-ahead. Returns mark-to-market P&L of the long vertical (excl.
    friction; apply option_round_trip_cost separately).
    """
    v0 = vertical_value(close_t, K1, K2, T_close, r, sigma_entry, call)
    T1 = max(T_close - overnight_frac, 1e-9)
    v1 = vertical_value(open_t1, K1, K2, T1, r, sigma_entry, call)
    return float(v1 - v0)


def overlay_decision(close_prices, open_prices, tier, rv_sigma_annual,
                     dte=1, r=0.04, call=True, width_frac=0.05,
                     min_multiple=MIN_EDGE_MULTIPLE):
    """Pre-registered NO-GO instrument for the overnight debit-vertical overlay.

    Given a series of consecutive (15:50 close[t], next open[t+1]) pairs for one
    name, its liquidity tier, and its HAR-RV vol, simulate an ATM-ish debit
    vertical (K1=close, K2=close*(1+width_frac) for calls) held overnight with
    entry-time IV, and compare the mean overlay edge to the friction gate.

    Returns a dict the caller logs verbatim: mean overnight overlay P&L as a
    fraction of debit, the friction fraction, the required edge, and the GO/
    NO-GO verdict. INSURANCE framing — a negative mean is expected and fine;
    the question is only whether ANY positive edge could clear friction.
    """
    c = np.asarray(close_prices, dtype=float)
    o = np.asarray(open_prices, dtype=float)
    n = min(len(c), len(o))
    if n < 5:
        return {'verdict': 'INSUFFICIENT_DATA', 'n': int(n)}
    c, o = c[:n], o[:n]
    spread = SPREAD_TIERS.get(tier, SPREAD_TIERS['C'])
    _, iv, _ = iv_from_har(rv_sigma_annual)
    T_close = dte / TRADING_DAYS
    overnight_frac = (17.0 / 24.0) / TRADING_DAYS  # ~15:50->09:30 calendar slice

    pnls, debits = [], []
    for ct, ot in zip(c, o):
        K1 = ct
        K2 = ct * (1.0 + width_frac) if call else ct * (1.0 - width_frac)
        klo, khi = (K1, K2) if call else (K2, K1)
        debit = vertical_debit(ct, klo, khi, T_close, r, iv, call)
        if debit <= 1e-6:
            continue
        pnl = overnight_overlay_pnl(ct, ot, klo, khi, T_close, overnight_frac,
                                    r, iv, call)
        pnls.append(pnl)
        debits.append(debit)
    if not debits:
        return {'verdict': 'NO_VIABLE_STRIKES', 'n': int(n)}

    pnls = np.asarray(pnls); debits = np.asarray(debits)
    mean_debit = float(debits.mean())
    # friction as a fraction of debit (legs = the two option premiums).
    legs = [bs_price(c.mean(), c.mean(), T_close, r, iv, call),
            bs_price(c.mean(), c.mean() * (1 + width_frac), T_close, r, iv, call)]
    friction_frac = option_round_trip_cost(legs, spread) / max(mean_debit, 1e-9)
    mean_edge_frac = float((pnls / debits).mean())
    clears, required = required_edge_clears(mean_edge_frac, friction_frac,
                                            min_multiple)
    return {
        'verdict': 'GO' if clears else 'NO_GO',
        'n': int(len(pnls)),
        'tier': tier,
        'mean_edge_frac_of_debit': round(mean_edge_frac, 4),
        'friction_frac_of_debit': round(friction_frac, 4),
        'required_edge_frac': round(required, 4),
        'mean_debit': round(mean_debit, 4),
        'iv_used': round(iv, 4),
        'note': ('PROXY IV (HAR-RV x IV/RV); tail-INSURANCE economics, NOT '
                 'carry — overnight option buyer loses on average. Do not size '
                 'live until real chains validate the IV/RV ratio.'),
    }
