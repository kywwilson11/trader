"""Cost-regime META features: PIT VIX history + Amihud illiquidity (wave-6 T2).

Spreads on spec-tech names widen with market stress (commonality in
liquidity), but the meta-learner has no cost-regime feature — only
Volatility_12h / ATR_Pct. This adds two FREE, point-in-time inputs the learner
can use to DISCOVER when to charge more for crossing a wide book:

  - VIX regime, from the FRED VIXCLS daily history (free, zero-auth CSV). Used
    1-day LAGGED and ffilled onto the bar grid, so it never peeks.
  - Amihud (2002) ILLIQ = mean(|return| / dollar_volume) over a trailing
    window — a per-name liquidity gauge straight from OHLCV.

CRITICAL (wave-6 kill-list): this is OPTION B only — a META FEATURE the model
may learn from. Do NOT build option A (regressing realized shortfall on
CONTEMPORANEOUS VIX into the cost gate) — that is a look-ahead / feedback
hazard. Nothing here touches the gate; it only produces lagged columns for
the harvest (VIX_Pctile is self-normalizing; Amihud is a RAW LEVEL and needs
a cross-sectional rank or per-name z-score before pooling across names).

Pure functions are unit-tested; only fetch_fred_vixcls() touches the network
(lazy urllib, like the archive syncs).
"""

import os

import numpy as np

# Fixed VIX regime thresholds (level): calm / normal / stress. Coarse but
# PIT-trivial; the trailing-percentile feature gives the learner a finer,
# self-calibrating alternative.
VIX_CALM = 15.0
VIX_STRESS = 25.0

_FRED_VIXCLS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

# Per-process memo for the FRED VIXCLS history (one fetch per run; a per-name
# harvest loop must not issue one full-history GET per symbol).
_VIXCLS_MEMO = None

# c26-T3 / B21: harvest wiring flag. Model-facing (training-store columns)
# -> default OFF; rides the bundled gotcha-#2 re-harvest/retrain event.
COST_REGIME_FEATURES = os.getenv('TRADER_COST_REGIME_FEATURES', '0').strip().lower() in ('1', 'true', 'yes')


# ---------------------------------------------------------------------------
# Amihud illiquidity (per-name, from OHLCV)
# ---------------------------------------------------------------------------

def amihud_illiq(close, volume, window=21, scale=1e6):
    """Amihud (2002) ILLIQ over a trailing window: mean(|ret| / dollar_vol).

    Higher = more price impact per dollar traded = less liquid. Returns a
    pandas Series aligned to the input (NaN until min_periods =
    min(window, max(2, window // 2)) observations exist — the first
    half-window is a shorter-sample, noisier estimate, a deliberate
    coverage/precision trade).
    `scale` rescales the tiny raw ratio to a workable magnitude (1e6 ->
    impact per $1M). Strictly trailing (rolling) -> point-in-time.

    SINGLE-NAME series only: a stacked long panel (non-unique index after
    pd.concat) computes pct_change ACROSS ticker boundaries and the column
    is pure noise — a warning is printed if the index is non-unique.
    `window` is in BARS, not days: on the hourly harvest grid the default 21
    is ~21 hours (~0.9d crypto / ~3.2 US sessions), not Amihud's canonical
    21 trading days. Non-positive dollar volume (bad data) is masked to NaN.
    Raises ValueError on a close/volume length mismatch (the array path used
    to fabricate finite values for bars with no volume data).
    """
    import pandas as pd
    c_arr = np.asarray(close, dtype=float)
    v_arr = np.asarray(volume, dtype=float)
    if c_arr.shape != v_arr.shape:
        raise ValueError(
            f"amihud_illiq: close/volume length mismatch: "
            f"{c_arr.shape} vs {v_arr.shape}")
    idx = close.index if isinstance(close, pd.Series) else None
    if idx is not None and isinstance(volume, pd.Series) \
            and not volume.index.equals(idx):
        print("[COST-REGIME] amihud_illiq: volume index differs from close "
              "index — values are paired POSITIONALLY, not by label")
    if idx is not None and not idx.is_unique:
        print("[COST-REGIME] amihud_illiq: non-unique index — this function "
              "is SINGLE-NAME only; a stacked panel computes returns ACROSS "
              "names (groupby ticker before calling)")
    c = pd.Series(c_arr, index=idx)
    v = pd.Series(v_arr, index=idx)
    # B21 pandas pin: fill_method=None (the pandas-3 default) — a NaN close
    # yields a NaN return excluded by the rolling min_periods instead of a
    # pad-fabricated 0-return across the gap. Free to adopt no-fill
    # semantics: zero callers/artifacts existed when pinned (B3 verdict).
    ret = c.pct_change(fill_method=None).abs()
    dv = c * v
    dollar_vol = dv.where(dv > 0)  # 0 or negative dollar volume -> NaN
    daily = (ret / dollar_vol) * scale
    daily = daily.where(np.isfinite(daily))  # zero prior close -> inf ratio
    return daily.rolling(window,
                         min_periods=min(window, max(2, window // 2))).mean()


# ---------------------------------------------------------------------------
# FRED VIXCLS history (free daily CSV)
# ---------------------------------------------------------------------------

def parse_fred_vixcls(csv_text):
    """Parse a FRED VIXCLS CSV into a date-indexed float Series, or None.

    FRED marks missing observations with '.', and the date column header is
    'DATE' (legacy) or 'observation_date' (current) — we take the first column
    as the date and the second as the value, coercing '.'/blanks to NaN and
    dropping them. Returns None (never raises, never an empty Series) for
    empty / header-only / all-missing / unparseable input.
    """
    import io
    import pandas as pd
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        # Zero-byte / whitespace-only / ragged bodies must read as failure,
        # matching the documented 'Series or None' contract.
        return None
    if df.shape[1] < 2:
        return None
    date_col, val_col = df.columns[0], df.columns[1]
    idx = pd.to_datetime(df[date_col], errors='coerce')
    val = pd.to_numeric(df[val_col], errors='coerce')
    s = pd.Series(val.values, index=idx).dropna()
    s = s[~s.index.isna()]
    # Header-only / all-'.' CSVs must read as failure, not an empty success —
    # fetch_fred_vixcls's contract is 'parsed Series or None'.
    return s.sort_index() if len(s) else None


def fetch_fred_vixcls():
    """Download the full FRED VIXCLS daily history (lazy network). Returns the
    parsed Series or None on failure. Memoized per process: the first
    successful fetch is cached so a per-name harvest loop costs one GET."""
    global _VIXCLS_MEMO
    if _VIXCLS_MEMO is not None:
        return _VIXCLS_MEMO.copy()
    import urllib.request
    try:
        req = urllib.request.Request(_FRED_VIXCLS_URL,
                                     headers={'User-Agent': 'trader/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read(16 * 1024 * 1024).decode('utf-8')
        s = parse_fred_vixcls(text)
        if s is not None:
            _VIXCLS_MEMO = s
            return s.copy()
        return None
    except Exception as e:
        print(f"[COST-REGIME] FRED VIXCLS fetch failed: {e}")
        return None


def vix_regime_code(level):
    """0 calm (<15), 1 normal, 2 stress (>=25). NaN/None/pd.NA -> 1 (neutral).

    Scalar contract only: the neutral fallback is for live/scalar callers.
    In the vectorized path (vix_features_for_index) this means VIX_Regime is
    1 on the first VIX observation day, where VIX_Level is NaN."""
    import pandas as pd
    if level is None or pd.isna(level):
        return 1
    if level < VIX_CALM:
        return 0
    if level >= VIX_STRESS:
        return 2
    return 1


def vix_features_for_index(vix_daily, index, pct_window=252):
    """Point-in-time VIX meta features aligned to an (hourly) bar index.

    Uses the PRIOR observation's close (shift 1) ffilled onto the grid — a bar
    on day D sees the close of the trading day BEFORE the most recent VIX
    observation on or before D. On weekends/holidays that is one trading day
    staler than the last published close (deliberately conservative; never a
    look-ahead). Sub-daily input is collapsed to one observation per calendar
    day (the day's last) BEFORE the lag, so the shift is a true one-DAY lag.

    PRECONDITIONS: `vix_daily` must carry a datetime-like index (a numeric
    index is refused -> None: pd.DatetimeIndex would reinterpret integers as
    1970-epoch nanoseconds and ffill END-OF-SAMPLE values onto every bar).
    `index` should be tz-naive-as-UTC or in a tz at/west of ~UTC+2:45 — the
    day bucket assumes local midnight on day D falls after the ~16:15 ET
    publication of the close dated D-1; an index further east could peek.

    Returns None if fewer than 5 daily VIX observations are usable — callers
    MUST check. Otherwise returns dict of float64 numpy arrays, positional
    against `index` (pass the destination frame's OWN index):
      VIX_Level     lagged daily VIX; NaN for bars before the first usable
                    observation (a [COST-REGIME] warning reports the fraction)
      VIX_Regime    0/1/2 from fixed thresholds, as float64; NaN before the
                    history; 1 where the lagged level itself is NaN (first
                    observation day) — see vix_regime_code
      VIX_Pctile    trailing-`pct_window` percentile rank in (0, 1] (self-
                    inclusive: the floor is 1/n, 0.0 unreachable; NaN until
                    min(20, pct_window) observations exist, and ranks are
                    over a GROWING window until pct_window fills)
    """
    import pandas as pd
    s = pd.Series(vix_daily).dropna()
    ix = s.index
    if not isinstance(ix, pd.DatetimeIndex):
        if pd.api.types.is_numeric_dtype(ix):
            print("[COST-REGIME] vix_daily has a numeric index, not dates — "
                  "refusing to align (would ffill end-of-sample values onto "
                  "every bar); returning None")
            return None
        try:
            ix = pd.DatetimeIndex(pd.to_datetime(ix, errors='coerce'))
        except (TypeError, ValueError):
            print("[COST-REGIME] vix_daily index is not datetime-like — "
                  "returning None")
            return None
        s.index = ix
        s = s[~s.index.isna()]
    # Chronological sort on FULL timestamps first, so the per-day collapse
    # below deterministically keeps each day's LAST observation.
    s = s.sort_index()
    six = s.index.normalize()
    if six.tz is not None:                # strip tz on the VIX side too —
        six = six.tz_localize(None)       # mirrors the bar-side strip below
    s.index = six
    # One observation per calendar day BEFORE the lag: shift(1) is a row
    # shift, so sub-daily/duplicate-date input would otherwise make "lagged"
    # a SAME-day reading — a real look-ahead.
    s = s[~s.index.duplicated(keep='last')]
    if len(s) < 5:
        return None
    lagged = s.shift(1)                                   # only yesterday's close
    # rank(method='max', pct=True) == (w[-1] >= w).mean() exactly (ties count
    # as <=); method is load-bearing — 'average' would NOT match. ~8x faster
    # than the old Python-lambda apply, but this path is only ~25ms/call on
    # the full FRED history anyway — do not optimize further.
    mp = min(pct_window, max(2, min(20, pct_window)))
    pct = s.rolling(pct_window, min_periods=mp).rank(
        method='max', pct=True).shift(1)                  # trailing rank, lagged
    regime = lagged.map(vix_regime_code)

    idx = pd.DatetimeIndex(index)
    day = idx.normalize()
    if day.tz is not None:
        day = day.tz_localize(None)
    uniq_days = pd.DatetimeIndex(day).unique()
    day_ser = pd.Series(day)

    # _map's ffill extends the last VIX value indefinitely — a truncated FRED
    # response would silently flatline all three features. Logging only.
    gap_days = (day.max() - s.index.max()).days if len(day) else 0
    if gap_days > 5:
        print(f"[COST-REGIME] newest bar is {gap_days} days past the last VIX "
              f"observation — features are stale forward-fills")

    # s (hence lagged/regime/pct) is unique + sorted after the collapse above,
    # so the daily grid is computed once and shared by all three columns.
    grid = s.index.union(uniq_days)

    def _map(series):
        # daily-grid forward-fill, then broadcast to each bar via its day
        # (bars on the same day share one value). astype(float) pins the
        # output dtype: VIX_Regime would otherwise flip int64/float64 with
        # the bar calendar, and nullable inputs would leak ExtensionArrays.
        per_day = series.reindex(grid).ffill().reindex(uniq_days)
        return day_ser.map(per_day).astype(float).to_numpy()

    out = {'VIX_Level': _map(lagged), 'VIX_Regime': _map(regime),
           'VIX_Pctile': _map(pct)}

    # Leading-edge visibility (mirror of the trailing staleness warning):
    # ffill cannot fill backwards, so bars before the first usable VIX
    # observation are NaN in all three columns — and harvests dropna().
    if len(out['VIX_Level']):
        miss = float(np.isnan(out['VIX_Level']).mean())
        if miss > 0.0:
            print(f"[COST-REGIME] {miss:.0%} of bars have no PIT VIX "
                  f"(history {s.index.min().date()}..{s.index.max().date()}, "
                  f"bars {day.min().date()}..{day.max().date()}) — those "
                  f"rows are NaN")
        elif np.isnan(out['VIX_Pctile']).all():
            print(f"[COST-REGIME] VIX_Pctile is 100% NaN — {len(s)} usable "
                  f"VIX observations is too few for the percentile "
                  f"(min_periods {mp} + the 1-day lag)")
    return out


def stamp_cost_regime_features(df, asset_type='stock', vix_daily=None):
    """B21 Option-B harvest wiring — DARK behind TRADER_COST_REGIME_FEATURES.

    Flag OFF (default): returns df UNCHANGED — the SAME object (the
    harvests' byte-identity contract). Flag ON (model-facing; activation
    only via the bundled re-harvest/retrain, gotcha #2, AND after live-serve
    injection parity lands in predict_now — see the packet report handoff):
    adds VIX_Level / VIX_Regime / VIX_Pctile (PIT 1-day-lagged FRED VIXCLS
    via vix_features_for_index; ONE memoized fetch per harvest process via
    the existing _VIXCLS_MEMO) and Amihud_Illiq (per-name trailing ILLIQ,
    ffill then 0.0-neutral-filled so the harvests' dropna() cannot eat rows
    — the ARCHIVE_FEATURES convention: "no data reads as neutral").

    SINGLE-NAME frames only (the harvests call per ticker before concat —
    amihud_illiq's own guard). Option B ONLY per the module header: columns
    for the learner, nothing touches any gate. Fail-open per column family.
    """
    if not COST_REGIME_FEATURES:
        return df
    out = df.copy()
    vix = vix_daily if vix_daily is not None else fetch_fred_vixcls()
    if vix is not None:
        feats = vix_features_for_index(vix, out.index)
        if feats is not None:
            for col, vals in feats.items():
                out[col] = vals
        else:
            print(f"[COST-REGIME] {asset_type}: VIX features skipped "
                  f"(too few usable observations)")
    else:
        print(f"[COST-REGIME] {asset_type}: VIX features skipped (no history)")
    try:
        if 'Close' in out.columns and 'Volume' in out.columns:
            am = amihud_illiq(out['Close'], out['Volume'])
            out['Amihud_Illiq'] = am.ffill().fillna(0.0)
        else:
            print(f"[COST-REGIME] {asset_type}: Amihud skipped (no Close/Volume)")
    except Exception as e:
        print(f"[COST-REGIME] {asset_type}: Amihud skipped ({e})")
    return out
