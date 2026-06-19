"""Per-name effective-spread estimation (Ardia-Guidotti-Kroencke 2024 EDGE).

The offline cost model used a FLAT spread haircut per asset class
(backtest.SPREAD_PCT, meta_label spread=0.05) while the LIVE gate already
sees the real quoted spread (round_trip_cost_pct's spread arg). So the
meta-label / q10 veto / backtest gate / hypersearch objective were all trained
on net-of-0.05%-spread returns but deployed behind a real-spread gate — fake
positive net on wide-spread spec-tech names that the DSR gate then certifies.

EDGE (Ardia, Guidotti, Kroencke 2024, *Journal of Financial Economics*)
estimates the effective bid-ask spread from OHLC bars alone — no quote data,
no cost — so we can stamp a per-bar, point-in-time spread on the SAME free
bars the harvest already pulls and make the offline cost match live.

Units: bidask's `edge`/`edge_rolling` return the effective spread as a FRACTION
of price; this module returns Eff_Spread_Pct in PERCENT (x100) to match the
existing `spread_pct` convention in fees.round_trip_cost_pct.

Point-in-time discipline: the rolling estimate at bar t uses only bars in
[t-window+1, t] (pandas .rolling is strictly trailing), so stamping it during
the harvest introduces no look-ahead — exactly like the _DV30 stamp.
"""

import math

import numpy as np
import pandas as pd

# Clips on the raw estimate (percent of price). EDGE is an RMS estimator and
# can go slightly negative on short windows / halt prints; floor it to a tiny
# positive crossing cost and cap it so a single bad print can't poison costs.
SPREAD_FLOOR_PCT = 0.02    # ~2 bps
SPREAD_CAP_PCT = 1.50      # ~150 bps

# Default trailing window. Hourly bars: ~7 bars/RTH-day, so 5 trading days
# ~= 35 bars — enough for a stable EDGE estimate without smearing regime.
DEFAULT_WINDOW = 35


def edge_spread_series(ohlc_df, window=DEFAULT_WINDOW,
                       floor_pct=SPREAD_FLOOR_PCT, cap_pct=SPREAD_CAP_PCT):
    """Trailing per-bar effective spread (PERCENT of price) for one ticker.

    Args:
        ohlc_df: DataFrame with Open/High/Low/Close (case-insensitive),
            time-ordered for a SINGLE ticker.
        window: trailing bar count for the rolling EDGE estimate.

    Returns a float Series aligned to ohlc_df.index, clipped to
    [floor_pct, cap_pct]. Early rows with insufficient history come back as
    the floor (a conservative non-zero crossing cost), never NaN, so the
    cost choke points always have a usable per-bar number.
    """
    cols = {c.lower(): c for c in ohlc_df.columns}
    need = ['open', 'high', 'low', 'close']
    if not all(k in cols for k in need):
        raise KeyError(f"edge_spread_series needs OHLC columns, got "
                       f"{list(ohlc_df.columns)}")
    df = ohlc_df.rename(columns={cols['open']: 'Open', cols['high']: 'High',
                                 cols['low']: 'Low', cols['close']: 'Close'})
    n = len(df)
    if n < max(window // 2, 5):
        return pd.Series(np.full(n, floor_pct), index=ohlc_df.index)
    try:
        from bidask import edge_rolling
        frac = edge_rolling(df[['Open', 'High', 'Low', 'Close']], window=window)
    except Exception:
        frac = _abdi_ranaldo_rolling(df, window)
    pct = pd.Series(frac, index=ohlc_df.index).astype(float) * 100.0
    # EDGE can emit NaN (first window-1 rows) or non-finite noise -> floor.
    pct = pct.replace([np.inf, -np.inf], np.nan)
    pct = pct.clip(lower=floor_pct, upper=cap_pct)
    pct = pct.fillna(floor_pct)
    return pct


def _abdi_ranaldo_rolling(df, window):
    """Abdi-Ranaldo (2017) close-high-low spread proxy — bidask fallback.

    2*sqrt(max(E[(c-eta)^2], 0)) where eta = (high+low)/2 (log mids). A robust
    secondary estimator for when bidask is unavailable or EDGE degenerates.
    Returns a price-FRACTION Series (caller scales to percent).
    """
    c = np.log(df['Close'].to_numpy(dtype=float))
    h = np.log(df['High'].to_numpy(dtype=float))
    low = np.log(df['Low'].to_numpy(dtype=float))
    mid = (h + low) / 2.0
    n = len(c)
    out = np.full(n, np.nan)
    for t in range(window - 1, n):
        sl = slice(t - window + 1, t + 1)
        cc = c[sl]
        m = mid[sl]
        if len(cc) < 3:
            continue
        # E[(c_t - eta_t)(c_t - eta_{t+1})] form, simplified single-window var
        d = cc - m
        val = np.mean(d * d)
        out[t] = 2.0 * np.sqrt(val) if val > 0 else 0.0
    return out


# --- Square-root market impact (wave-8 #6) -------------------------------- #
# A $100 and a $50k order into a thin name pay the same fee+spread today; real
# impact grows with participation. Almgren et al. (2005) / Kyle lambda:
# temporary impact ~ k * spread * sqrt(notional / ADV). Used to DE-CERTIFY edge
# that only exists because the cost model under-prices size on illiquid names.
# OFF by default (strategy_config.IMPACT_COST_ENABLED); DV30 must be stamped in
# the harvested data and k/notional calibrated on the Jetson before enabling.
IMPACT_CAP_PCT = 2.0   # per-side impact ceiling so a near-zero ADV can't blow up


def market_impact_pct(notional, adv_dollar, spread_pct, k=1.0, sides=2,
                      cap_pct=IMPACT_CAP_PCT):
    """Square-root market-impact haircut, PERCENT of notional (round trip).

    impact_one_side = min(k * spread_pct * sqrt(notional / ADV), cap_pct); the
    round trip is `sides` of that (entry + exit = 2). Fail-open: returns 0.0 when
    ADV/notional/spread is unknown, non-finite, or <= 0, so a missing DV30 leaves
    the spread-only cost untouched.
    """
    n, adv, sp = float(notional), float(adv_dollar), float(spread_pct)
    if not all(np.isfinite([n, adv, sp])) or adv <= 0 or n <= 0 or sp <= 0:
        return 0.0
    one_side = min(k * sp * math.sqrt(n / adv), float(cap_pct))
    return one_side * sides


def impact_inputs_from_df(df):
    """(adv_array, notional, k) for per_bar_round_trip_cost's impact term.

    Returns (None, None, 1.0) — i.e. impact OFF — unless strategy_config enables
    it AND the frame carries a DV30 (30d $ volume) column. Single source so the
    backtest and the meta-labeler stay in lockstep.
    """
    try:
        from strategy_config import (IMPACT_COST_ENABLED, IMPACT_K,
                                      IMPACT_TYPICAL_NOTIONAL)
    except Exception:
        return None, None, 1.0
    if not IMPACT_COST_ENABLED:
        return None, None, 1.0
    for col in ('DV30', '_DV30'):
        if col in df.columns:
            return df[col].values, float(IMPACT_TYPICAL_NOTIONAL), float(IMPACT_K)
    return None, None, 1.0


def per_bar_round_trip_cost(asset_type, spread_pct_array,
                            maker=False, live=False,
                            adv_dollar=None, notional=None, impact_k=1.0):
    """Vectorized round-trip cost (PERCENT) = fee constant + per-bar spread
    [+ optional sqrt market impact].

    round_trip_cost_pct(asset, s) is linear in s: fee_const + max(s, 0). So we
    compute fee_const once at spread=0 and add the per-bar spread array — same
    result, no Python-level per-bar fee call. NaN spreads fall back to the
    asset's flat default so a missing stamp never zeroes the cost.

    When adv_dollar (per-bar ADV $, e.g. DV30) and notional are supplied, a
    square-root market-impact haircut k*spread*sqrt(notional/ADV) (round trip,
    per-side capped) is ADDED per bar — larger for thinner names. adv_dollar=None
    (the default) preserves the exact prior spread-only behavior.
    """
    from fees import round_trip_cost_pct
    fee_const = round_trip_cost_pct(asset_type, 0.0, maker, live)
    flat = 0.10 if asset_type == 'crypto' else 0.05
    s = np.asarray(spread_pct_array, dtype=float)
    s = np.where(np.isfinite(s) & (s >= 0.0), s, flat)
    cost = fee_const + s
    if adv_dollar is not None and notional is not None:
        adv = np.asarray(adv_dollar, dtype=float)
        part = np.where(np.isfinite(adv) & (adv > 0.0), float(notional) / adv, 0.0)
        impact_one_side = np.clip(impact_k * s * np.sqrt(np.clip(part, 0.0, None)),
                                  0.0, IMPACT_CAP_PCT)
        cost = cost + 2.0 * impact_one_side
    return cost
