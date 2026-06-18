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
hazard. Nothing here touches the gate; it only produces lagged, stationary
columns for the harvest.

Pure functions are unit-tested; only fetch_fred_vixcls() touches the network
(lazy urllib, like the archive syncs).
"""

import numpy as np

# Fixed VIX regime thresholds (level): calm / normal / stress. Coarse but
# PIT-trivial; the trailing-percentile feature gives the learner a finer,
# self-calibrating alternative.
VIX_CALM = 15.0
VIX_STRESS = 25.0

_FRED_VIXCLS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


# ---------------------------------------------------------------------------
# Amihud illiquidity (per-name, from OHLCV)
# ---------------------------------------------------------------------------

def amihud_illiq(close, volume, window=21, scale=1e6):
    """Amihud (2002) ILLIQ over a trailing window: mean(|ret| / dollar_vol).

    Higher = more price impact per dollar traded = less liquid. Returns a
    pandas Series aligned to the input (NaN until the window fills). `scale`
    rescales the tiny raw ratio to a workable magnitude (1e6 -> impact per $1M).
    Strictly trailing (rolling) -> point-in-time.
    """
    import pandas as pd
    c = pd.Series(np.asarray(close, dtype=float))
    v = pd.Series(np.asarray(volume, dtype=float))
    ret = c.pct_change().abs()
    dollar_vol = (c * v).replace(0.0, np.nan)
    daily = (ret / dollar_vol) * scale
    return daily.rolling(window, min_periods=max(2, window // 2)).mean()


# ---------------------------------------------------------------------------
# FRED VIXCLS history (free daily CSV)
# ---------------------------------------------------------------------------

def parse_fred_vixcls(csv_text):
    """Parse a FRED VIXCLS CSV into a date-indexed float Series.

    FRED marks missing observations with '.', and the date column header is
    'DATE' (legacy) or 'observation_date' (current) — we take the first column
    as the date and the second as the value, coercing '.'/blanks to NaN and
    dropping them.
    """
    import io
    import pandas as pd
    df = pd.read_csv(io.StringIO(csv_text))
    if df.shape[1] < 2:
        return None
    date_col, val_col = df.columns[0], df.columns[1]
    idx = pd.to_datetime(df[date_col], errors='coerce')
    val = pd.to_numeric(df[val_col], errors='coerce')
    s = pd.Series(val.values, index=idx).dropna()
    s = s[~s.index.isna()]
    return s.sort_index()


def fetch_fred_vixcls():
    """Download the full FRED VIXCLS daily history (lazy network). Returns the
    parsed Series or None on failure."""
    import urllib.request
    try:
        req = urllib.request.Request(_FRED_VIXCLS_URL,
                                     headers={'User-Agent': 'trader/1.0'})
        text = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
        return parse_fred_vixcls(text)
    except Exception as e:
        print(f"[COST-REGIME] FRED VIXCLS fetch failed: {e}")
        return None


def vix_regime_code(level):
    """0 calm (<15), 1 normal, 2 stress (>=25). NaN -> 1 (neutral)."""
    if level is None or (isinstance(level, float) and np.isnan(level)):
        return 1
    if level < VIX_CALM:
        return 0
    if level >= VIX_STRESS:
        return 2
    return 1


def vix_features_for_index(vix_daily, index, pct_window=252):
    """Point-in-time VIX meta features aligned to an (hourly) bar index.

    Uses the PRIOR day's close VIX (shift 1) ffilled onto the grid — a bar on
    day D sees D-1's official close, never D's (no look-ahead). Returns dict:
      VIX_Level     lagged daily VIX
      VIX_Regime    0/1/2 from fixed thresholds
      VIX_Pctile    trailing-`pct_window` percentile rank in [0,1] (self-
                    calibrating, no look-ahead)
    """
    import pandas as pd
    s = pd.Series(vix_daily).dropna().sort_index()
    if len(s) < 5:
        return None
    lagged = s.shift(1)                                   # only yesterday's close
    pct = s.rolling(pct_window, min_periods=20).apply(
        lambda w: (w[-1] >= w).mean(), raw=True).shift(1)  # trailing rank, lagged
    regime = lagged.map(vix_regime_code)

    idx = pd.DatetimeIndex(index)
    day = idx.normalize()
    if day.tz is not None:
        day = day.tz_localize(None)
    lagged.index = pd.DatetimeIndex(lagged.index).normalize()
    regime.index = lagged.index
    pct.index = pd.DatetimeIndex(pct.index).normalize()
    uniq_days = pd.DatetimeIndex(day).unique()
    day_ser = pd.Series(day)

    def _map(series):
        # daily-grid forward-fill on a UNIQUE index, then broadcast to each bar
        # via its day (bars on the same day share one value; no dup-label error)
        s2 = series[~series.index.duplicated(keep='last')].sort_index()
        grid = s2.index.union(uniq_days)
        per_day = s2.reindex(grid).ffill().reindex(uniq_days)
        return day_ser.map(per_day).values

    return {'VIX_Level': _map(lagged), 'VIX_Regime': _map(regime),
            'VIX_Pctile': _map(pct)}
