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


def per_bar_round_trip_cost(asset_type, spread_pct_array,
                            maker=False, live=False):
    """Vectorized round-trip cost (PERCENT) = fee constant + per-bar spread.

    round_trip_cost_pct(asset, s) is linear in s: fee_const + max(s, 0). So we
    compute fee_const once at spread=0 and add the per-bar spread array — same
    result, no Python-level per-bar fee call. NaN spreads fall back to the
    asset's flat default so a missing stamp never zeroes the cost.
    """
    from fees import round_trip_cost_pct
    fee_const = round_trip_cost_pct(asset_type, 0.0, maker, live)
    flat = 0.10 if asset_type == 'crypto' else 0.05
    s = np.asarray(spread_pct_array, dtype=float)
    s = np.where(np.isfinite(s) & (s >= 0.0), s, flat)
    return fee_const + s
