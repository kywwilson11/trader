"""Cross-sectional rank features — the panel substrate for stock selection.

The wave-3 diagnosis: the app makes a per-hour RELATIVE decision (rank
top-7-of-~50) using models that only ever see each symbol's OWN history.
No pooled model can know whether NVDA's momentum is strong relative to
the other names THIS hour — the one thing that determines top-7
membership. LightGBM splits are global thresholds and cannot reconstruct
within-hour ranks from absolute features.

Fix (Gu-Kelly-Xiu RFS 2020 fn.29; Freyberger-Neuhierl-Weber; replicated
internationally by Tobek-Hronec and in liquid universes by Avramov et
al. 2023): rank-transform each feature CROSS-SECTIONALLY per period to
[-1, 1]. Plus two panel-context scalars every member shares: dispersion
(how separated the cross-section is — Stivers-Sun: dispersion conditions
momentum profitability) and breadth.

TRAIN/SERVE POPULATION PARITY (the red team's survival conditional):
training ranks run over the as-of top-K dollar-volume members of the
~96-name panel — so live ranks must too. The live pre-pass fetches the
FULL panel hourly, applies the same trailing-30d dollar-volume top-K
mask via the same dv30() used by the harvest, and ranks with the same
formula. Ranking only the ~50 traded names would put them at distorted
rank extremes relative to training.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from log_config import get_logger

logger = get_logger(__name__)

# Base columns rank-transformed per timestamp (must exist in
# compute_stock_features output)
CS_RANK_BASE_COLS = [
    'Return_4h', 'Return_12h', 'ROC', 'RSI', 'ATR_Pct', 'Volume_Ratio',
    'Price_SMA20_Ratio', 'RS_vs_SPY', 'Gap_Pct', 'ROD_Ret',
    'RM_252_21',   # residual momentum (Blitz-Huij-Martens)
    'Ret_21d',     # Medhat-Schmeling conditioning base
    'DV30',        # dollar-volume rank = the free turnover proxy
]
CS_CONTEXT_COLS = ['CS_Dispersion', 'CS_Breadth']
# MS_Interact = CS_Rank_Ret_21d x CS_Rank_DV30 (Medhat-Schmeling RFS
# 2022: 1-month return is MOMENTUM in low-turnover names and REVERSAL in
# high-turnover names — the product lets trees split on both regimes)
CS_FEATURE_COLS = ([f'CS_Rank_{c}' for c in CS_RANK_BASE_COLS]
                   + CS_CONTEXT_COLS + ['MS_Interact'])

_LIVE_TTL_SEC = 3300            # ranks change once per hourly bar
_live_cache: tuple[float, dict] | None = None


def dv30(ohlcv: pd.DataFrame) -> pd.Series:
    """Trailing 30d median daily dollar volume, aligned to the bar index.

    ONE implementation shared by the harvest membership mask and the live
    panel mask — population parity by construction.
    """
    daily_dv = (ohlcv['Close'] * ohlcv['Volume']).resample('1D').sum()
    daily_dv = daily_dv[daily_dv > 0]
    med = daily_dv.rolling('30D').median()
    return med.reindex(ohlcv.index, method='ffill')


def _signed_rank(rank: pd.Series, n) -> pd.Series:
    """Map average-ranks 1..n symmetrically to [-1, 1] (bottom=-1, top=+1).
    The naive 2*(rank/n)-1 is asymmetric: top pins at +1 but bottom only
    reaches 2/n-1."""
    return 2.0 * (rank - 1.0) / (n - 1.0) - 1.0


def add_panel_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Harvest-side: per-timestamp cross-sectional ranks over the panel.

    Call AFTER the as-of membership mask (rows present = members). Adds
    CS_Rank_<col> in [-1, 1] and the shared context columns. Single-name
    timestamps rank to 0 (median) — harmless, and matches the live
    neutral fill when the panel is unavailable.
    """
    if 'Ticker' not in df.columns or df.empty:
        return df
    ts = df.index
    for col in CS_RANK_BASE_COLS:
        if col not in df.columns:
            continue
        r = df.groupby(ts)[col].rank(method='average')
        n = df.groupby(ts)[col].transform('count')
        signed = _signed_rank(r, n)
        # A 1-member cross-section has no relative information
        df[f'CS_Rank_{col}'] = signed.where(n > 1, 0.0)
    if 'Return_4h' in df.columns:
        disp = df.groupby(ts)['Return_4h'].transform('std')
        df['CS_Dispersion'] = disp.fillna(0.0)
    if 'Price_SMA20_Ratio' in df.columns:
        above = (df['Price_SMA20_Ratio'] > 1.0).astype(float)
        breadth = above.groupby(ts).transform('mean')
        # Centered: 0 = half the panel above trend (neutral fill value)
        df['CS_Breadth'] = (breadth - 0.5) * 2.0
    if 'CS_Rank_Ret_21d' in df.columns and 'CS_Rank_DV30' in df.columns:
        df['MS_Interact'] = df['CS_Rank_Ret_21d'] * df['CS_Rank_DV30']
    return df


def neutral_fill_cs(df: pd.DataFrame) -> pd.DataFrame:
    """0.0-fill CS columns (0 = median rank / neutral context) so the
    harvest dropna() can't eat rows where a base column was warming up —
    the same train/serve 'no data = neutral' convention the crypto
    archive features use."""
    for col in CS_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


# --- Live pre-pass (hourly, called by the stock loop) ---

def _panel_symbols() -> list[str]:
    from stock_config import load_stock_universe, TRAINING_CANDIDATE_POOL
    uni = [s for s in load_stock_universe() if '/' not in s]
    return sorted(set(uni) | set(TRAINING_CANDIDATE_POOL))


def compute_live_panel_ranks(api, spy_close=None,
                             top_k: int | None = None) -> dict[str, dict]:
    """{symbol: {CS_Rank_*: v, CS_Dispersion: v, CS_Breadth: v}} for the
    CURRENT hour, ranked over the live as-of top-K members of the full
    panel (mirrors the harvest membership mask). Cached ~1h. Returns {}
    on failure — predict_now neutral-fills 0.0, matching training's
    neutral fill.
    """
    global _live_cache
    now = time.monotonic()
    if _live_cache and (now - _live_cache[0]) < _LIVE_TTL_SEC:
        return _live_cache[1]

    from stock_config import AS_OF_TOP_K
    from market_data import fetch_stock_bars_alpaca
    from indicators import compute_stock_features
    if top_k is None:
        top_k = AS_OF_TOP_K

    rows = {}
    dvs = {}
    for sym in _panel_symbols():
        try:
            bars = fetch_stock_bars_alpaca(api, sym)
            if bars is None or len(bars) < 60:
                continue
            feats = compute_stock_features(bars, spy_close=spy_close,
                                           symbol=sym)
            dv = dv30(bars).iloc[-1]
            if not np.isfinite(dv) or dv <= 0:
                continue
            last = feats.iloc[-1].copy()
            last['DV30'] = float(dv)   # ranked as the turnover proxy
            rows[sym] = last
            dvs[sym] = float(dv)
        except Exception as e:
            logger.debug("[PANEL] %s: skipped (%s)", sym, e)

    if len(rows) < 10:
        logger.warning("[PANEL] only %d panel names available — "
                       "cross-sectional ranks neutral this hour", len(rows))
        _live_cache = (now, {})
        return {}

    members = sorted(dvs, key=dvs.get, reverse=True)[:top_k]
    frame = pd.DataFrame({s: rows[s] for s in members}).T

    out: dict[str, dict] = {s: {} for s in members}
    for col in CS_RANK_BASE_COLS:
        if col not in frame.columns:
            continue
        r = frame[col].rank(method='average')
        n = int(r.notna().sum())
        signed = _signed_rank(r, n) if n > 1 else r * 0.0
        for s in members:
            v = signed.get(s)
            out[s][f'CS_Rank_{col}'] = float(v) if pd.notna(v) else 0.0
    disp = float(np.nanstd(frame['Return_4h'].values, ddof=1)) \
        if 'Return_4h' in frame.columns else 0.0  # ddof=1 = harvest's std
    breadth = (float(np.nanmean((frame['Price_SMA20_Ratio'] > 1.0)
                                .astype(float))) - 0.5) * 2.0 \
        if 'Price_SMA20_Ratio' in frame.columns else 0.0
    for s in members:
        out[s]['CS_Dispersion'] = disp if np.isfinite(disp) else 0.0
        out[s]['CS_Breadth'] = breadth if np.isfinite(breadth) else 0.0
        out[s]['MS_Interact'] = (out[s].get('CS_Rank_Ret_21d', 0.0)
                                 * out[s].get('CS_Rank_DV30', 0.0))

    logger.info("[PANEL] live cross-section: %d members ranked "
                "(dispersion=%.2f, breadth=%+.2f)", len(members),
                disp, breadth)
    _live_cache = (now, out)
    return out
