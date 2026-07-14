"""Indicator preset configuration — choose which features to train on.

Harvest scripts always compute ALL indicators to CSV. The preset only filters
which columns scripts/hypersearch_v2.py uses for training. The existing
feature_cols.pkl mechanism ensures inference matches training.

Persists to indicator_config.json (gitignored) via an atomic write. Default
preset: "standard". No heavy imports (json, pathlib, os, tempfile, contextlib
only) so it's safe for the GUI env.
"""

import contextlib
import json
import os
import tempfile
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "indicator_config.json"

_DEFAULTS = {"preset": "standard"}

# --- Hurst input mode (2026-07 review fix, DEFAULT OFF) ---
# R/S analysis is defined on a process's INCREMENTS. The historical call fed
# price LEVELS, so the partial sums integrate the walk and H reads ~0.8 on a
# pure random walk — the feature carried no regime information and the live
# `hurst < 0.45` mean-reversion gate could never fire. True = compute on
# returns (correct math; random walk ≈ 0.5). MODEL-FACING: the feature's
# distribution shifts, so flip ONLY together with a fresh harvest + retrain
# on the Jetson (and CLAUDE.md gotcha #2: delete the study DBs first) —
# never against a deployed levels-trained model/scaler.
HURST_ON_RETURNS = False

# Columns only present in crypto training data
CRYPTO_ONLY_COLS = ["BTC_Return_1h", "BTC_SMA_Ratio", "BTC_RSI",
                    "Funding_Rate_Ann", "Funding_Z", "Funding_Chg_24h",
                    "OI_Chg_24h", "OI_Z", "TT_LS_Z", "Taker_Imb_24h"]

# Columns only present in stock training data
STOCK_ONLY_COLS = ["VWAP", "Price_VWAP_Ratio", "Gap_Pct", "ATR_Pct", "RS_vs_SPY",
                   "ROD_Ret", "Same_Hour_Mean_40d",
                   "CS_Rank_Return_4h", "CS_Rank_Return_12h", "CS_Rank_ROC",
                   "CS_Rank_RSI", "CS_Rank_ATR_Pct", "CS_Rank_Volume_Ratio",
                   "CS_Rank_Price_SMA20_Ratio", "CS_Rank_RS_vs_SPY",
                   "CS_Rank_Gap_Pct", "CS_Rank_ROD_Ret",
                   "CS_Rank_RM_252_21", "CS_Rank_Ret_21d", "CS_Rank_DV30",
                   "CS_Dispersion", "CS_Breadth", "MS_Interact",
                   "RM_252_21", "Ret_21d",
                   "RR_5", "RR_21", "MA_Dist_10d", "MA_Dist_20d", "MA_Dist_50d",
                   "MA_Dist_100d", "MA_Dist_200d", "ON_Mom_21", "ON_Mom_252",
                   "TugOfWar_252", "Pos_Range_20h", "Pos_Range_60h",
                   "MidRange_Gap_20h", "MidRange_Gap_60h", "Pos_Range_20d",
                   "Pos_Range_60d", "SVR_21", "SVR_Z",
                   "CS_Rank_ON_Mom_252", "CS_Rank_RR_5",
                   "CS_Rank_Pos_Range_20d", "CS_Rank_MA_Dist_50d"]

# --- Preset definitions ---
# Each preset lists column names. "full" uses None (all columns).

_MINIMAL_FEATURES = [
    # OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Momentum
    "RSI", "MACD_12_26_9", "MACDh_12_26_9",
    # Volatility
    "ATR",
    # Trend
    "SMA_20", "Price_SMA20_Ratio",
    # Volume
    "Volume_Ratio", "OBV",
    # Oscillator
    "STOCHk_14_3_3",
    # Temporal
    "Hour_sin", "Hour_cos", "Day_sin", "Day_cos",
    # Rate of change
    "ROC",
    # Return-based
    "Return_4h", "Return_12h", "Volatility_12h",
    # Sentiment
    "Daily_Sentiment",
]

_STANDARD_FEATURES = _MINIMAL_FEATURES + [
    # MACD signal line
    "MACDs_12_26_9",
    # Bollinger Bands (no BBM — identical to SMA_20)
    "BBL_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0",
    # Volume moving average
    "Volume_SMA_20",
    # Stochastic %D
    "STOCHd_14_3_3",
    # Hurst exponent (regime awareness)
    "Hurst",
    # Calendar effects
    "Month_sin", "Month_cos", "Turn_of_Month",
    # Perp funding positioning (crypto only)
    "Funding_Rate_Ann", "Funding_Z", "Funding_Chg_24h",
    # Perp open-interest dynamics (crypto only)
    "OI_Chg_24h", "OI_Z",
    # Top-trader long/short positioning (crypto only)
    "TT_LS_Z",
    # Aggressive taker flow imbalance (crypto only)
    "Taker_Imb_24h",
    # Return-of-day + same-hour periodicity (stock only; BDS 2025 / HKS)
    "ROD_Ret", "Same_Hour_Mean_40d",
    # Cross-sectional panel ranks + context (stock only; GKX rank
    # transform — selection is a RELATIVE decision)
    "CS_Rank_Return_4h", "CS_Rank_Return_12h", "CS_Rank_ROC",
    "CS_Rank_RSI", "CS_Rank_ATR_Pct", "CS_Rank_Volume_Ratio",
    "CS_Rank_Price_SMA20_Ratio", "CS_Rank_RS_vs_SPY",
    "CS_Rank_Gap_Pct", "CS_Rank_ROD_Ret",
    "CS_Rank_RM_252_21", "CS_Rank_Ret_21d", "CS_Rank_DV30",
    "CS_Dispersion", "CS_Breadth", "MS_Interact", "RM_252_21", "Ret_21d",
    # Wave 4: residual reversal, HZZ MA distances, session momentum,
    # JKX range-position distillation, FINRA shorting flow
    "RR_5", "RR_21", "MA_Dist_10d", "MA_Dist_20d", "MA_Dist_50d",
    "MA_Dist_100d", "MA_Dist_200d", "ON_Mom_21", "ON_Mom_252",
    "TugOfWar_252", "Pos_Range_20h", "Pos_Range_60h",
    "MidRange_Gap_20h", "MidRange_Gap_60h", "Pos_Range_20d",
    "Pos_Range_60d", "SVR_21", "SVR_Z",
    "CS_Rank_ON_Mom_252", "CS_Rank_RR_5",
    "CS_Rank_Pos_Range_20d", "CS_Rank_MA_Dist_50d",
]

# Stationary features only — no raw prices/volumes that trend over time.
# Cross-asset columns (BTC_Return_1h, RS_vs_SPY, etc.) are already stationary
# and get auto-included via asset-type filtering in hypersearch.
_STATIONARY_FEATURES = [
    # Returns (stationary by construction)
    "Return_4h", "Return_12h", "Volatility_12h", "ROC",
    # Ratios (price-normalized)
    "Price_SMA20_Ratio", "Volume_Ratio", "BBP_20_2.0", "BBB_20_2.0",
    # Oscillators (bounded)
    "RSI", "STOCHk_14_3_3", "STOCHd_14_3_3",
    # MACD (differenced, mean-reverting)
    "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
    # Temporal (cyclical encoding, bounded [-1, 1])
    "Hour_sin", "Hour_cos", "Day_sin", "Day_cos",
    # Calendar effects (cyclical, bounded)
    "Month_sin", "Month_cos", "Turn_of_Month",
    # Hurst exponent (bounded [0, 1], regime awareness)
    "Hurst",
    # Sentiment (bounded)
    "Daily_Sentiment",
    # Perp funding positioning (crypto only; stationary rates/z-scores;
    # BIS 'Crypto Carry': extreme funding marks crowded positioning that
    # precedes crashes at this system's 12-48h horizon)
    "Funding_Rate_Ann", "Funding_Z", "Funding_Chg_24h",
    # Perp open-interest dynamics (crypto only; stationary % change and
    # z-score — rising price on FALLING OI is short covering, not trend)
    "OI_Chg_24h", "OI_Z",
    # Top-trader long/short positioning z (crypto only; direction of
    # crowding — funding prices it, OI sizes it, this signs it)
    "TT_LS_Z",
    # Aggressive taker flow imbalance (crypto only; log 24h mean
    # buy/sell taker ratio — who is hitting the tape)
    "Taker_Imb_24h",
    # Return-of-day + same-hour periodicity (stock only; stationary
    # returns; BDS 2025 loser-bounce / HKS intraday periodicity)
    "ROD_Ret", "Same_Hour_Mean_40d",
    # Cross-sectional panel ranks, bounded [-1,1] + context (stock only;
    # GKX/FNW rank transform: the models finally see each name's
    # standing WITHIN the panel this hour — the relative information
    # the top-7 selection actually turns on)
    "CS_Rank_Return_4h", "CS_Rank_Return_12h", "CS_Rank_ROC",
    "CS_Rank_RSI", "CS_Rank_ATR_Pct", "CS_Rank_Volume_Ratio",
    "CS_Rank_Price_SMA20_Ratio", "CS_Rank_RS_vs_SPY",
    "CS_Rank_Gap_Pct", "CS_Rank_ROD_Ret",
    "CS_Rank_RM_252_21", "CS_Rank_Ret_21d", "CS_Rank_DV30",
    "CS_Dispersion", "CS_Breadth", "MS_Interact", "RM_252_21", "Ret_21d",
    # Wave 4: residual reversal, HZZ MA distances, session momentum,
    # JKX range-position distillation, FINRA shorting flow
    "RR_5", "RR_21", "MA_Dist_10d", "MA_Dist_20d", "MA_Dist_50d",
    "MA_Dist_100d", "MA_Dist_200d", "ON_Mom_21", "ON_Mom_252",
    "TugOfWar_252", "Pos_Range_20h", "Pos_Range_60h",
    "MidRange_Gap_20h", "MidRange_Gap_60h", "Pos_Range_20d",
    "Pos_Range_60d", "SVR_21", "SVR_Z",
    "CS_Rank_ON_Mom_252", "CS_Rank_RR_5",
    "CS_Rank_Pos_Range_20d", "CS_Rank_MA_Dist_50d",
]

# --- stationary_lean (2026-07 indicator review) ---
# stationary minus STRUCTURAL redundancy only — exact duplicates and linear
# identities, no data-mined cuts (empirical pruning belongs to
# indicator_leadlag.py on the Jetson's real panel):
#   ROC           == Return_12h bit-for-bit (same formula, indicators.py) —
#                    every preset shipped one column twice
#   CS_Rank_ROC   == CS_Rank_Return_12h (rank of the identical column)
#   MACDs_12_26_9 == MACD_12_26_9 - MACDh_12_26_9 (exact linear identity;
#                    keeping all three is rank-2 information in 3 columns)
#   STOCHd_14_3_3 == SMA3(STOCHk) — reconstructable from the %K sequence
#                    both the LSTM window and the flattened LGB window see
#   Month_sin/cos + Turn_of_Month — <= 5 observed yearly cycles in the
#                    training span; unlearnable amplitude, pure overfit surface
# Selecting this preset is MODEL-FACING (feature-set change): first Jetson
# retrain after switching => CLAUDE.md gotcha #2 (delete study DBs, reset
# best_score), and it deploys via the challenger -> shadow path as usual.
_LEAN_EXCLUDE = {
    "ROC", "CS_Rank_ROC", "MACDs_12_26_9", "STOCHd_14_3_3",
    "Month_sin", "Month_cos", "Turn_of_Month",
}
_STATIONARY_LEAN = [f for f in _STATIONARY_FEATURES if f not in _LEAN_EXCLUDE]

PRESETS = {
    "minimal": {
        "description": "Core signals only. Fastest training, lowest overfitting risk.",
        "features": _MINIMAL_FEATURES,
    },
    "standard": {
        "description": "Balanced set with Bollinger Bands and full oscillators. Recommended.",
        "features": _STANDARD_FEATURES,
    },
    "stationary": {
        "description": "Stationary features only (returns, ratios, oscillators). "
                       "Best for regression models — no raw price/volume drift.",
        "features": _STATIONARY_FEATURES,
    },
    "stationary_lean": {
        "description": "stationary minus structural redundancy (exact dupes, "
                       "linear identities, unlearnable calendar cycles). "
                       "A/B candidate vs stationary via shadow.",
        "features": _STATIONARY_LEAN,
    },
    "full": {
        "description": "All indicators including divergence and cross-asset signals. "
                       "More data but higher overfitting risk.",
        "features": None,  # None means use all columns
    },
}


def load_indicator_config() -> dict:
    """Load indicator config from disk, falling back to defaults."""
    try:
        with open(_FILE) as f:
            config = json.load(f)
        if isinstance(config, dict) and config.get("preset") in PRESETS:
            return config
    except (OSError, ValueError, TypeError):
        pass
    return dict(_DEFAULTS)


def save_indicator_config(config: dict) -> None:
    """Persist indicator config to disk atomically (tempfile + os.replace)."""
    fd, tmp = tempfile.mkstemp(dir=str(_FILE.parent), prefix=_FILE.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp, _FILE)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def get_preset_name() -> str:
    """Return the active preset name."""
    return load_indicator_config().get("preset", "standard")


def get_preset_features(preset_name: str) -> list[str] | None:
    """Return a copy of the feature list for a preset, or None for 'full' (all columns)."""
    preset = PRESETS.get(preset_name)
    if preset is None:
        return None
    features = preset["features"]
    return list(features) if features is not None else None


def get_all_preset_info() -> dict:
    """Return metadata for all presets (for GUI display).

    Returns dict like:
        {"minimal": {"description": "...", "count": 20, "features": [...]}, ...}
    """
    info = {}
    for name, preset in PRESETS.items():
        features = preset["features"]
        info[name] = {
            "description": preset["description"],
            "count": len(features) if features is not None else None,
            "features": list(features) if features is not None else None,
        }
    return info
