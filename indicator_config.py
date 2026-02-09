"""Indicator preset configuration — choose which features to train on.

Harvest scripts always compute ALL indicators to CSV. The preset only filters
which columns hypersearch_dual.py uses for training. The existing
feature_cols.pkl mechanism ensures inference matches training.

Persists to indicator_config.json (gitignored). Default preset: "standard".
No heavy imports (json, pathlib only) so it's safe for the GUI env.
"""

import json
from pathlib import Path

_FILE = Path(__file__).resolve().parent / "indicator_config.json"

_DEFAULTS = {"preset": "standard"}

# Columns only present in crypto training data
CRYPTO_ONLY_COLS = ["BTC_Return_1h", "BTC_SMA_Ratio", "BTC_RSI"]

# Columns only present in stock training data
STOCK_ONLY_COLS = ["VWAP", "Price_VWAP_Ratio", "Gap_Pct", "ATR_Pct", "RS_vs_SPY"]

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
]

PRESETS = {
    "minimal": {
        "description": "Core signals only. Fastest training, lowest overfitting risk.",
        "features": _MINIMAL_FEATURES,
    },
    "standard": {
        "description": "Balanced set with Bollinger Bands and full oscillators. Recommended.",
        "features": _STANDARD_FEATURES,
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
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return dict(_DEFAULTS)


def save_indicator_config(config: dict) -> None:
    """Persist indicator config to disk."""
    with open(_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_preset_name() -> str:
    """Return the active preset name."""
    return load_indicator_config().get("preset", "standard")


def get_preset_features(preset_name: str) -> list[str] | None:
    """Return the feature list for a preset, or None for 'full' (all columns)."""
    preset = PRESETS.get(preset_name)
    if preset is None:
        return None
    return preset["features"]


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
