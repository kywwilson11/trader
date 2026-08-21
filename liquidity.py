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

Scope (2026-07): STOCKS ONLY — scripts/harvest_stock_data.py is the sole site
that stamps Eff_Spread_Pct. The crypto harvest stamps no spread, so the crypto
book still prices the flat fees.FLAT_SPREAD_PCT['crypto'] haircut in
backtest.py and meta_label.py; extending the stamp to crypto is a model-facing
owner decision (re-harvest + promotion gate). Note also that EDGE estimates
the EFFECTIVE spread (what trades actually paid) while the live gate prices
the QUOTED spread (order_utils ask - bid), so the stamp closes the per-name
dispersion gap but is a lower bound on the live hurdle, not an equality.
"""

import logging
import math
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Clips on the raw estimate (percent of price). bidask's edge_rolling is
# called with its default sign=False, so the raw estimate is sqrt(|s2|) —
# NEVER negative; the floor is not a negativity guard. It does two jobs:
# (a) lower clip on genuinely tiny finite estimates, and (b) via the fillna
# below, the DEFAULT stamped on every bar EDGE could not estimate (warmup,
# halts, zero-range windows, bad prints). Default behavior is UNCHANGED:
# (b) still prices a no-estimate bar CHEAPER than the flat
# fees.FLAT_SPREAD_PCT fallback (0.05/0.10) used when the stamp is absent
# entirely — the D40 owner decision now has a SHIPPED DARK fix: under
# TRADER_SPREAD_FILL_V2 (model-facing, gotcha-#2 re-harvest event)
# no-estimate bars pay FLAT_SPREAD_PCT and +inf clips to the CAP instead of
# the floor. The cap stops a single bad print from poisoning costs.
SPREAD_FLOOR_PCT = 0.02    # ~2 bps
SPREAD_CAP_PCT = 1.50      # ~150 bps

# Default trailing window. Hourly bars: ~7 bars/RTH-day, so 5 trading days
# ~= 35 bars — enough for a stable EDGE estimate without smearing regime.
DEFAULT_WINDOW = 35

# --- c26-T3 dark flags (B05/D40). All default OFF; OFF is byte-identical. ---
# KILL-ADJACENT NOTICE: the crypto spread-stamp family below is adjacent to
# KILL_LIST.md line 90 ("EDGE-on-hourly inflation fix", wave-7). Research
# (campaign_2026-08/02_research.md B05.1) re-opens only the quote-first /
# minute-bar half; everything ships DARK pending an explicit owner ruling
# plus the gotcha-#2 re-harvest/retrain event.
SPREAD_FILL_V2 = os.getenv('TRADER_SPREAD_FILL_V2', '0').strip().lower() in ('1', 'true', 'yes')
CRYPTO_SPREAD_STAMP = os.getenv('TRADER_CRYPTO_SPREAD_STAMP', '0').strip().lower() in ('1', 'true', 'yes')
STOCK_MINUTE_EDGE = os.getenv('TRADER_STOCK_MINUTE_EDGE', '0').strip().lower() in ('1', 'true', 'yes')
IMPACT_VOLSCALE = os.getenv('TRADER_IMPACT_VOLSCALE', '0').strip().lower() in ('1', 'true', 'yes')


def _no_estimate_fill_pct(floor_pct, asset_type):
    """Value stamped on bars with NO spread estimate. Default (V1): the
    floor — the documented open owner decision (D40). Under
    TRADER_SPREAD_FILL_V2: the asset's flat fees.FLAT_SPREAD_PCT — corrupted
    or absent data must never price CHEAPER than having no stamp at all."""
    if not SPREAD_FILL_V2:
        return floor_pct
    from fees import FLAT_SPREAD_PCT
    return FLAT_SPREAD_PCT['crypto' if asset_type == 'crypto' else 'stock']


def _finalize_spread_pct(pct, floor_pct, cap_pct, fill_pct):
    """Shared clip/fill tail for both estimator paths.
    V1 (default, byte-identical to the pre-c26 inline code): inf->NaN,
    count, clip, fillna(fill_pct==floor). V2 (TRADER_SPREAD_FILL_V2): clip
    BEFORE the inf handling so +inf lands at the CAP, -inf at the floor
    (D40 ordering fix), then NaN -> fill_pct (the flat fallback).
    Returns (pct, raw_nan, at_floor, at_cap)."""
    import numpy as np
    if SPREAD_FILL_V2:
        raw_nan = int(pct.isna().sum())
        at_floor = int((pct < floor_pct).sum())   # NaN/-inf handled by clip
        at_cap = int((pct > cap_pct).sum())       # +inf counts as cap-clipped
        pct = pct.clip(lower=floor_pct, upper=cap_pct)
        pct = pct.fillna(fill_pct)
        return pct, raw_nan, at_floor, at_cap
    pct = pct.replace([np.inf, -np.inf], np.nan)
    raw_nan = int(pct.isna().sum())
    at_floor = int((pct < floor_pct).sum())
    at_cap = int((pct > cap_pct).sum())
    pct = pct.clip(lower=floor_pct, upper=cap_pct)
    pct = pct.fillna(fill_pct)
    return pct, raw_nan, at_floor, at_cap


def edge_spread_series(ohlc_df, window=DEFAULT_WINDOW,
                       floor_pct=SPREAD_FLOOR_PCT, cap_pct=SPREAD_CAP_PCT,
                       *, symbol=None, asset_type='stock'):
    """Trailing per-bar effective spread (PERCENT of price) for one ticker.

    Args:
        ohlc_df: DataFrame with Open/High/Low/Close (case-insensitive),
            time-ordered for a SINGLE ticker. A non-monotonic index logs a
            WARNING: .rolling is positional (row order), so the PIT guarantee
            only holds for time-sorted input.
        window: trailing bar count for the rolling EDGE estimate. Must be an
            int >= 3 (bidask needs at least two usable periods); the default
            35 is the only production value.
        symbol: optional ticker name used ONLY in log messages, so estimator
            warnings can be attributed to a name in a multi-ticker harvest.

    Returns a float Series aligned to ohlc_df.index, clipped to
    [floor_pct, cap_pct], never NaN. Default behavior unchanged: the floor
    is the NO-ESTIMATE default — warmup rows, degenerate windows (halts /
    zero-range stretches / bad prints), and any frame shorter than `window`
    come back at the floor, CHEAPER than the flat fees.FLAT_SPREAD_PCT
    fallback used when the stamp is absent entirely. That D40 owner
    decision now has a SHIPPED DARK fix: under TRADER_SPREAD_FILL_V2
    (model-facing, gotcha-#2 re-harvest event) no-estimate bars pay the
    asset's FLAT_SPREAD_PCT (`asset_type` selects it) and +inf clips to
    the cap instead of the floor (see the constants comment above).
    """
    if not isinstance(window, (int, np.integer)) or window < 3:
        raise ValueError(f"edge_spread_series: window must be an int >= 3 "
                         f"(bidask needs >= 2 usable periods), got {window!r}")
    try:
        floor_pct, cap_pct = float(floor_pct), float(cap_pct)
    except (TypeError, ValueError):
        raise ValueError(f"edge_spread_series: floor_pct/cap_pct must be "
                         f"numeric, got {floor_pct!r} / {cap_pct!r}") from None
    if not (math.isfinite(floor_pct) and math.isfinite(cap_pct)) \
            or floor_pct > cap_pct:
        raise ValueError(f"edge_spread_series: need finite floor_pct <= "
                         f"cap_pct, got {floor_pct!r} / {cap_pct!r}")
    need = ('open', 'high', 'low', 'close')
    cols = {}
    for c in ohlc_df.columns:
        key = str(c).lower()
        if key in need:
            if key in cols:
                raise KeyError(f"edge_spread_series: ambiguous duplicate "
                               f"{key!r} columns ({cols[key]!r} and {c!r})")
            cols[key] = c
    if not all(k in cols for k in need):
        raise KeyError(f"edge_spread_series needs OHLC columns, got "
                       f"{list(ohlc_df.columns)}")
    name = symbol or 'unnamed'
    fill_pct = _no_estimate_fill_pct(floor_pct, asset_type)
    if not ohlc_df.index.is_monotonic_increasing:
        log.warning("edge_spread_series(%s): index is not monotonically "
                    "increasing (%d bars) — .rolling is positional, so the "
                    "strictly-trailing/PIT guarantee does NOT hold for this "
                    "frame", name, len(ohlc_df))
    # Project the four columns out instead of renaming the caller's whole
    # (possibly ~150-column) feature frame; value-identical, and duplicate
    # case-variant columns are impossible past the check above.
    df = ohlc_df[[cols['open'], cols['high'], cols['low'], cols['close']]]
    df = df.set_axis(['Open', 'High', 'Low', 'Close'], axis=1)
    n = len(df)
    if n < max(window, 5):
        # bidask's first estimate lands at row window-1, so a frame shorter
        # than `window` is provably all-NaN -> entirely floored. Skip the
        # wasted estimator run and say so: this spread column is fabricated.
        log.warning("edge_spread_series(%s): %d bars < window=%d — no EDGE "
                    "estimate possible, whole series stamped at the %.2f%% "
                    "floor", name, n, window, fill_pct)
        return pd.Series(np.full(n, fill_pct), index=ohlc_df.index)
    try:
        from bidask import edge_rolling
        frac = edge_rolling(df, window=window)  # df is exactly OHLC by now
    except Exception as exc:
        # The fallback is several-fold upward-biased; a silent estimator swap
        # would shift the whole offline cost surface with zero trace.
        log.warning("bidask EDGE failed for %s on %d-bar frame (%s: %s) — "
                    "using upward-biased AR fallback", name, n,
                    type(exc).__name__, exc)
        frac = _abdi_ranaldo_rolling(df, window)
    # Positional construction on both estimator paths (edge_rolling returns a
    # Series built on df.index, the fallback a bare ndarray) — bit-identical
    # to the old label-aligned form, without dual alignment semantics.
    pct = pd.Series(np.asarray(frac, dtype=float) * 100.0, index=ohlc_df.index)
    # edge_rolling(sign=False) is never negative; NaN marks the first
    # window-1 rows and windows with no usable price variation (halts /
    # zero-range bars). The AR fallback can emit +/-inf on zero or negative
    # prices. All of those end at the floor via the fillna below.
    pct, raw_nan, at_floor, at_cap = _finalize_spread_pct(pct, floor_pct,
                                                          cap_pct, fill_pct)
    post_warmup = max(n - (window - 1), 1)
    nan_beyond_warmup = max(raw_nan - (window - 1), 0)
    log.info("edge_spread_series(%s): %d bars, median %.3f%% — %d raw-NaN "
             "(%d beyond warmup), %d floor-clipped, %d cap-clipped",
             name, n, float(pct.median()), raw_nan, nan_beyond_warmup,
             at_floor, at_cap)
    if nan_beyond_warmup > 0.25 * post_warmup:
        log.warning("edge_spread_series(%s): %d/%d post-warmup bars had no "
                    "EDGE estimate and were stamped at the %.2f%% floor — "
                    "this name's spread stamp is mostly fabricated",
                    name, nan_beyond_warmup, post_warmup, fill_pct)
    return pct


def _abdi_ranaldo_rolling(df, window):
    """Simplified close-vs-mid dispersion proxy — break-glass bidask fallback.

    2*sqrt(max(E[(c-eta)^2], 0)) where eta = (high+low)/2 (log mids). NOT the
    true Abdi-Ranaldo (2017) estimator: AR uses the cross-product
    E[(c_t-eta_t)(c_t-eta_{t+1})], which cancels efficient-price variance; the
    same-bar squared form here is vol-dominated and several-fold UPWARD-biased,
    so treat it as a conservative spread UPPER BOUND, not an estimate.
    Upgrading to the cross-product form changes stamped Eff_Spread_Pct —
    Jetson re-harvest + promotion gate required. Returns a price-FRACTION
    array (caller scales to percent).
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
        if len(cc) < 3:   # only reachable when window < 3 (slice len == window)
            continue
        # SAME-bar squared dispersion E[(c_t - eta_t)^2] (NOT the AR cross-
        # product E[(c_t-eta_t)(c_t-eta_{t+1})] — see docstring); upward-biased
        d = cc - m
        val = np.mean(d * d)
        out[t] = 2.0 * np.sqrt(val) if val > 0 else 0.0
    return out


# --- Crypto spread tiers (B05.1, quote-first; KILL_LIST:90-adjacent) ------ #
# DARK: inert unless TRADER_CRYPTO_SPREAD_STAMP=1. Tier levels are meant to
# come from scripts/crypto_spread_census.py output (Jetson-run owner
# evidence); the defaults below are conservative PLACEHOLDERS, not
# measurements: BTC/ETH at the tight tier's upper end, liquid alts pinned to
# the current flat 0.10% (no cost cut without evidence), long tail wide.
CRYPTO_CENSUS_FILE = os.getenv('TRADER_CRYPTO_CENSUS_FILE', 'crypto_spread_census.json')
CRYPTO_TIER_DEFAULTS_PCT = {
    'BTC/USD': 0.05, 'ETH/USD': 0.05,
    'SOL/USD': 0.10, 'XRP/USD': 0.10, 'DOGE/USD': 0.10, 'LINK/USD': 0.10,
    'AVAX/USD': 0.25, 'BCH/USD': 0.25, 'DOT/USD': 0.25, 'LTC/USD': 0.25,
}
CRYPTO_TIER_FALLBACK_PCT = 0.25   # unknown pair -> wide tier (conservative)
_CENSUS_MEMO = None               # lazy per-process load; tests reset to None


def _load_census():
    """Memoized load of the census JSON (scripts/crypto_spread_census.py
    output). Accepts both the census schema ({'pairs': {sym: stats}}) and a
    flat {sym: pct} map. Any error (missing file included) memoizes and
    returns {} — the tier map then falls back to the placeholder defaults."""
    global _CENSUS_MEMO
    if _CENSUS_MEMO is not None:
        return _CENSUS_MEMO
    import json
    try:
        with open(CRYPTO_CENSUS_FILE) as fh:
            data = json.load(fh)
        _CENSUS_MEMO = data.get('pairs', data) if isinstance(data, dict) else {}
    except Exception as exc:
        log.info("crypto spread census not loaded (%s: %s) — using tier "
                 "defaults", type(exc).__name__, exc)
        _CENSUS_MEMO = {}
    return _CENSUS_MEMO


def get_crypto_spread_tier(symbol):
    """Per-pair crypto spread tier (PERCENT). Census evidence first, then the
    placeholder defaults, then the wide fallback. Never raises."""
    try:
        sym = str(symbol).upper().replace('-', '/')
        entry = _load_census().get(sym)
        if entry is not None:
            val = (entry.get('median_spread_pct') if isinstance(entry, dict)
                   else float(entry))
            if val is not None and math.isfinite(float(val)) and float(val) > 0:
                tier = min(max(float(val), SPREAD_FLOOR_PCT), SPREAD_CAP_PCT)
                log.debug("get_crypto_spread_tier(%s): %.3f%% (source=census)",
                          sym, tier)
                return tier
        return CRYPTO_TIER_DEFAULTS_PCT.get(sym, CRYPTO_TIER_FALLBACK_PCT)
    except Exception as exc:
        log.warning("get_crypto_spread_tier(%r) failed (%s: %s) — wide "
                    "fallback", symbol, type(exc).__name__, exc)
        return CRYPTO_TIER_FALLBACK_PCT


def stamp_crypto_spreads(df, symbol):
    """Harvest-side crypto Eff_Spread_Pct stamp (B05.1, quote-first tiers).

    Flag OFF (default): returns df UNCHANGED — the SAME object (the
    byte-identity contract; the crypto store carries no Eff_Spread_Pct
    column and downstream keeps pricing flat FLAT_SPREAD_PCT['crypto']).
    Flag ON (TRADER_CRYPTO_SPREAD_STAMP — model-facing; KILL_LIST:90 owner
    ruling + gotcha-#2 re-harvest/retrain required; backtest/meta_label
    auto-consume the column on presence): stamps the pair's constant tier.
    Per B05, a pair whose tier sits at SPREAD_CAP_PCT is a CRYPTO_POOL
    delisting candidate (warned below). Fail-open: any exception returns
    df unchanged."""
    if not CRYPTO_SPREAD_STAMP:
        return df
    try:
        tier = float(get_crypto_spread_tier(symbol))
        out = df.copy()
        out['Eff_Spread_Pct'] = tier
        log.info("stamp_crypto_spreads(%s): Eff_Spread_Pct=%.3f%% "
                 "(tier map/census)", symbol, tier)
        if tier >= SPREAD_CAP_PCT:
            log.warning("stamp_crypto_spreads(%s): tier at/above the %.2f%% "
                        "cap — CRYPTO_POOL delisting candidate (B05)",
                        symbol, SPREAD_CAP_PCT)
        return out
    except Exception as exc:
        log.warning("stamp_crypto_spreads(%r) failed (%s: %s) — frame "
                    "unchanged", symbol, type(exc).__name__, exc)
        return df


def edge_spread_daily_from_minute(minute_df, hourly_index, smooth_days=5,
                                  floor_pct=SPREAD_FLOOR_PCT,
                                  cap_pct=SPREAD_CAP_PCT, *, symbol=None,
                                  min_bars_per_day=30, min_days=3):
    """Per-DAY EDGE from 1-minute bars, mapped onto an hourly index (B05.1).

    EDGE at minute frequency is valid where hourly is noise-floor-dominated
    (the JFE authors' own guidance; trust criterion n_bars >=
    (2*sigma_bar/s_expected)^4). One estimate per ticker-DAY from that day's
    1-min bars via bidask.edge(sign=True) — negative estimates -> NaN, never
    abs-folded (B05 sign rule) — spread*100 -> percent, then
    rolling(smooth_days, min_periods=min_days).median() over trading days
    (n ~ 1950 minute bars at the defaults), clipped to [floor, cap], then
    shift(1) BY POSITION over the daily series so day-D hourly rows see only
    data through the prior trading day (a per-day estimate uses that day's
    FULL session — stamping same-day would leak intraday-future bars).

    Returns a float Series aligned to hourly_index; NaN wherever no covered
    prior day exists — the CALLER merges over the hourly stamp; NO fill
    here. Fail-open: any estimator failure returns an all-NaN Series."""
    name = symbol or 'unnamed'
    nan_out = pd.Series(np.full(len(hourly_index), np.nan),
                        index=hourly_index)
    try:
        need = ('open', 'high', 'low', 'close')
        cols = {}
        for c in minute_df.columns:
            key = str(c).lower()
            if key in need and key not in cols:
                cols[key] = c
        if not all(k in cols for k in need):
            raise KeyError(f"edge_spread_daily_from_minute needs OHLC "
                           f"columns, got {list(minute_df.columns)}")
        mdf = minute_df[[cols['open'], cols['high'], cols['low'],
                         cols['close']]]
        mdf = mdf.set_axis(['Open', 'High', 'Low', 'Close'], axis=1)

        def _day_keys(idx):
            d = pd.DatetimeIndex(idx).normalize()
            if d.tz is not None:      # mirror cost_regime.vix_features_for_
                d = d.tz_localize(None)   # index's tz strip
            return d

        mdays = _day_keys(mdf.index)
        from bidask import edge
        daily = {}
        for day, sub in mdf.groupby(mdays):
            if len(sub) < min_bars_per_day:
                continue
            est = edge(sub['Open'].to_numpy(dtype=float),
                       sub['High'].to_numpy(dtype=float),
                       sub['Low'].to_numpy(dtype=float),
                       sub['Close'].to_numpy(dtype=float), sign=True)
            est = float(est) * 100.0
            daily[day] = est if (math.isfinite(est) and est > 0) else np.nan
        if not daily:
            return nan_out
        daily_ser = pd.Series(daily).sort_index()
        smoothed = (daily_ser.rolling(smooth_days, min_periods=min_days)
                    .median().clip(floor_pct, cap_pct).shift(1))
        hdays = _day_keys(hourly_index)
        mapped = pd.Series(hdays).map(smoothed)
        out = pd.Series(mapped.to_numpy(dtype=float), index=hourly_index)
        log.info("edge_spread_daily_from_minute(%s): %d estimated days, "
                 "%d/%d hourly bars covered", name, len(daily_ser),
                 int(out.notna().sum()), len(out))
        return out
    except Exception as exc:
        log.warning("edge_spread_daily_from_minute(%s) failed (%s: %s) — "
                    "all-NaN (hourly stamp retained)", name,
                    type(exc).__name__, exc)
        return nan_out


# --- Square-root market impact (wave-8 #6) -------------------------------- #
# A $100 and a $50k order into a thin name pay the same fee+spread today; real
# impact grows with participation. Almgren et al. (2005) / Kyle lambda:
# temporary impact ~ k * spread * sqrt(notional / ADV). Used to DE-CERTIFY edge
# that only exists because the cost model under-prices size on illiquid names.
# OFF by default (strategy_config.IMPACT_COST_ENABLED). NOTE the real enable
# recipe: the stock harvest deliberately DROPS the DV30 level after the rank
# layer (scripts/harvest_stock_data.py — only its RANK is a feature, never
# the level), so flipping IMPACT_COST_ENABLED alone is a warned no-op.
# Enabling for real needs a cost-only dollar-ADV column the harvest KEEPS and
# hypersearch excludes from features (like Eff_Spread_Pct), plus k/notional
# calibrated on the Jetson.
IMPACT_CAP_PCT = 2.0   # per-side impact ceiling so a near-zero ADV can't blow up

# B05.2 vol-scale re-base (DARK behind TRADER_IMPACT_VOLSCALE): empirical
# sqrt-law calibrations scale by DAILY VOL, not spread (I = Y*sigma_D*
# sqrt(Q/V); Y ~ 0.34-0.69 US large-cap, ~0.9 BTC). sigma_D/spread ~ 40x
# (stocks) / ~20x (crypto), so the spread-scaled default k=1.0 underprices
# impact 10-40x. Keeping the code shape (k*spread*sqrt(N/ADV)), the re-base
# is k = Y * median(sigma_D/spread) per book. Exponent stays 0.5 (do not
# fit); IMPACT_CAP_PCT unchanged.
IMPACT_Y = {'stock': 0.5, 'crypto': 0.9}
IMPACT_K_VOLSCALE = {'stock': 20.0, 'crypto': 18.0}


def volscale_impact_k(asset_type, sigma_daily_pct=None, spread_pct=None):
    """Vol-scaled impact k (B05.2). With finite sigma_daily_pct > 0 and
    spread_pct > 0: IMPACT_Y[book] * sigma_daily_pct / spread_pct; otherwise
    the per-book constant IMPACT_K_VOLSCALE[book]. Fail-open to the constant
    on any bad input."""
    book = 'crypto' if asset_type == 'crypto' else 'stock'
    try:
        sig, sp = float(sigma_daily_pct), float(spread_pct)
        if math.isfinite(sig) and math.isfinite(sp) and sig > 0 and sp > 0:
            return IMPACT_Y[book] * sig / sp
    except (TypeError, ValueError):
        pass
    return IMPACT_K_VOLSCALE[book]


def _infer_asset_type(df):
    """'crypto' when the frame's Ticker column carries '/'-pairs; else
    'stock' (the HIGHER re-based k — conservative fallback)."""
    try:
        if 'Ticker' in df.columns:
            vals = df['Ticker'].dropna()
            if len(vals) and '/' in str(vals.iloc[0]):
                return 'crypto'
    except Exception:
        pass
    return 'stock'


def market_impact_pct(notional, adv_dollar, spread_pct, k=1.0, sides=2,
                      cap_pct=IMPACT_CAP_PCT):
    """Square-root market-impact haircut, PERCENT of notional (round trip).

    Scalar REFERENCE implementation — no production caller; the vectorized
    twin inside per_bar_round_trip_cost reimplements the same formula and the
    two are pinned bar-identical by tests. impact_one_side =
    min(k * spread_pct * sqrt(notional / ADV), cap_pct); the round trip is
    `sides` of that (entry + exit = 2). Fail-open: returns 0.0 when ANY
    input (notional / ADV / spread / k / sides / cap_pct) is None,
    non-numeric, non-finite, or out of range (adv/notional/spread <= 0,
    k < 0, sides < 0), so a missing DV30 or a mis-parsed IMPACT_K leaves the
    spread-only cost untouched — matching the vectorized twin, which skips a
    non-finite k and clips the impact term at zero.
    """
    try:
        n, adv, sp = float(notional), float(adv_dollar), float(spread_pct)
        kf, sd, cap = float(k), float(sides), float(cap_pct)
    except (TypeError, ValueError):
        return 0.0
    if (not all(np.isfinite([n, adv, sp, kf, sd, cap]))
            or adv <= 0 or n <= 0 or sp <= 0 or kf < 0 or sd < 0):
        return 0.0
    one_side = min(kf * sp * math.sqrt(n / adv), cap)
    return max(one_side * sd, 0.0)


def impact_inputs_from_df(df):
    """(adv_array, notional, k) for per_bar_round_trip_cost's impact term.

    Returns (None, None, 1.0) — i.e. impact OFF — unless strategy_config enables
    it AND the frame carries a DV30 (30d $ volume) column. Single source so the
    backtest and the meta-labeler stay in lockstep.

    DV30 must be PER-BAR dollar ADV in DOLLARS — a mis-scaled column (e.g.
    millions) would pin every bar at the impact cap; note panel_ranks.py
    writes a one-value-per-symbol snapshot under the same 'DV30' name, which
    is NOT this input.
    """
    try:
        from strategy_config import (IMPACT_COST_ENABLED, IMPACT_K,
                                      IMPACT_TYPICAL_NOTIONAL)
    except Exception as exc:
        log.warning("impact_inputs_from_df: strategy_config import failed "
                    "(%s: %s) — impact term disabled", type(exc).__name__, exc)
        return None, None, 1.0
    if not IMPACT_COST_ENABLED:
        return None, None, 1.0
    for col in ('DV30', '_DV30'):
        if col in df.columns:
            k = float(IMPACT_K)
            if IMPACT_VOLSCALE:
                k = float(IMPACT_K_VOLSCALE[_infer_asset_type(df)])
                log.info("impact k re-based to vol scale (B05.2): %.1f", k)
            return df[col].values, float(IMPACT_TYPICAL_NOTIONAL), k
    log.warning("IMPACT_COST_ENABLED but no DV30/_DV30 column in frame — "
                "impact term inactive")
    return None, None, 1.0


def per_bar_round_trip_cost(asset_type, spread_pct_array,
                            maker=False, live=False,
                            adv_dollar=None, notional=None, impact_k=1.0):
    """Vectorized round-trip cost (PERCENT) = fee constant + per-bar spread
    [+ optional sqrt market impact].

    round_trip_cost_pct(asset, s) is linear in s: fee_const + max(s, 0). We
    compute fee_const once at spread=0 and add the per-bar spread array —
    identical to the scalar fees path bar-by-bar for FINITE, NON-NEGATIVE
    spreads. For NaN/inf/NEGATIVE spreads this helper is DELIBERATELY more
    conservative than the scalar (which would charge zero spread): it
    substitutes the asset's flat fees.FLAT_SPREAD_PCT so a missing or corrupt
    stamp can never under-price a bar. Do not "restore parity" here — the
    divergence is the protection. A literal 0.0 spread IS accepted as valid
    (warned below; treating it as missing is an open owner decision).

    Contract note: the returned value is a full ROUND-TRIP cost priced off
    THAT bar's spread. backtest.py and meta_label.py index it at the ENTRY
    bar, i.e. the entry bar's spread pays for both crossings; the exit bar's
    spread is never consulted (changing that is an owner decision).

    `maker`/`live` exist for signature parity with fees.round_trip_cost_pct
    and MUST stay False in this module's two real contexts (offline backtest
    gate, meta-label training): fees.py forbids live=True outside live paths,
    and it would read journal files from disk inside this vectorized helper.

    When adv_dollar (per-bar ADV $, e.g. DV30) and notional are supplied, a
    square-root market-impact haircut k*spread*sqrt(notional/ADV) (round
    trip, per-side capped) is ADDED per bar — larger for thinner names.
    adv_dollar=None (the default) preserves the exact prior spread-only
    behavior. A malformed adv_dollar (non-numeric, wrong shape) skips ONLY
    the impact term with a WARNING — the per-bar spread cost is retained,
    never downgraded to flat.
    """
    from fees import FLAT_SPREAD_PCT, round_trip_cost_pct
    fee_const = round_trip_cost_pct(asset_type, 0.0, maker, live)
    flat = FLAT_SPREAD_PCT['crypto' if asset_type == 'crypto' else 'stock']
    s = np.asarray(spread_pct_array, dtype=float)
    finite = np.isfinite(s)
    above_mask = finite & (s > SPREAD_CAP_PCT)
    above_cap = int(above_mask.sum())
    zeros = int((s == 0.0).sum())
    if above_cap or zeros:
        # max over the COUNTED set only — a stray +inf goes down the flat-
        # substitution path below and must not masquerade as the max here.
        log.warning("per_bar_round_trip_cost(%s): suspicious spread stamp — "
                    "%d bars above SPREAD_CAP_PCT (max %.3f%%), %d bars "
                    "exactly 0.0 (the producer floors at %.2f%%) — check the "
                    "Eff_Spread_Pct column", asset_type, above_cap,
                    float(s[above_mask].max()) if above_cap else 0.0, zeros,
                    SPREAD_FLOOR_PCT)
    s = np.where(finite & (s >= 0.0), s, flat)
    cost = fee_const + s
    if adv_dollar is not None and notional is not None:
        try:
            n_dollar, k = float(notional), float(impact_k)
            # Parity with market_impact_pct's fail-open: a non-finite size/k
            # must skip the impact term, not propagate NaN into every bar's
            # cost.
            if math.isfinite(n_dollar) and math.isfinite(k) and n_dollar > 0:
                adv = np.asarray(adv_dollar, dtype=float)
                if adv.shape != s.shape:
                    raise ValueError(f"adv_dollar shape {adv.shape} != "
                                     f"spread shape {s.shape}")
                ok = np.isfinite(adv) & (adv > 0.0)
                part = np.zeros_like(adv)
                np.divide(n_dollar, adv, out=part, where=ok)
                # part >= 0 by construction, so no inner clip is needed
                impact_one_side = np.clip(k * s * np.sqrt(part),
                                          0.0, IMPACT_CAP_PCT)
                cost = cost + 2.0 * impact_one_side
        except Exception as exc:
            # A broken OPTIONAL impact input must not take down the healthy
            # per-bar spread path (the callers' blanket except would fall all
            # the way back to the FLAT cost, discarding a good stamp).
            log.warning("per_bar_round_trip_cost(%s): impact term skipped "
                        "(%s: %s) — spread-only cost retained", asset_type,
                        type(exc).__name__, exc)
    return cost
