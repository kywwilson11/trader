"""Shared utilities for the crypto and stock trading loops.

Centralizes duplicated code: API construction, model hot-reload helpers,
cooldown tracking, inference device selection, the predict_symbol wrapper,
and Kelly criterion position sizing.
"""

import json
import os
import datetime
from pathlib import Path

import numpy as np
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from hw_monitor import is_gpu_available
from predict_now import get_live_prediction

load_dotenv()


# --- SHARED CONSTANTS ---
# These were hardcoded in multiple places; centralizing here prevents drift.

LLM_VETO_THRESHOLD = 0.15      # LLM score below this = catastrophic veto
THERMAL_THROTTLE_TEMP = 75     # GPU temp threshold for throttling (Celsius)
ORDER_TIMEOUT = 30             # Seconds to wait for limit order fill
TEMP_LOG_EVERY_N_CYCLES = 10   # Log GPU temp every N cycles


# --- API CONSTRUCTION ---

def get_api():
    """Build an Alpaca REST client from .env credentials."""
    return tradeapi.REST(
        os.getenv('ALPACA_API_KEY'),
        os.getenv('ALPACA_API_SECRET'),
        os.getenv('ALPACA_BASE_URL'),
        api_version='v2',
    )


# --- MODEL HOT-RELOAD HELPERS ---

def get_model_mtime(path):
    """Get modification time of a model file, or 0 if it doesn't exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


# --- INFERENCE DEVICE ---

def choose_inference_device():
    """Always use CPU for inference in the trading bots.

    The model is ~1.3MB — CPU inference is fast enough for 30-second trading
    cycles (a few ms per symbol). Reserving GPU exclusively for training
    eliminates the CUDA OOM crashes that happen when inference and training
    compete for the Jetson's 8GB unified memory.
    """
    return 'cpu'


# --- COOLDOWN ---

def cooldown_ok(last_trade_time, symbol, cooldown_minutes=30):
    """Return True if the symbol is not in cooldown.

    Args:
        last_trade_time: dict mapping symbol -> datetime of last trade
        symbol: symbol to check
        cooldown_minutes: minimum minutes between trades on the same symbol
    """
    if symbol not in last_trade_time:
        return True
    elapsed = (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
    return elapsed >= cooldown_minutes * 60


# --- PREDICTION WRAPPER ---

def predict_symbol(api, symbol, model, config, scaler_X, feature_cols,
                   inference_device, asset_type='crypto', benchmark_close=None,
                   return_snapshot=False):
    """Run a regression prediction for a single symbol.

    Returns:
        (symbol, pred_return) tuple where pred_return is float or None
        If return_snapshot: (symbol, pred_return, snapshot_dict)
    """
    extra_kwargs = {}
    if asset_type == 'stock':
        extra_kwargs['spy_close'] = benchmark_close
    else:
        extra_kwargs['btc_close'] = benchmark_close

    result = get_live_prediction(
        symbol, model, scaler_X, config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type=asset_type, return_snapshot=return_snapshot,
        **extra_kwargs,
    )
    if return_snapshot:
        pred, snapshot = result if result else (None, None)
        return symbol, pred, snapshot
    return symbol, result


# --- KELLY CRITERION ---

_TRADE_MEMORY_FILE = Path(__file__).resolve().parent / "trade_memory.json"


def compute_kelly_fraction(min_trades: int = 50) -> float | None:
    """Compute half-Kelly fraction from trade history.

    Uses trade_memory.json to calculate:
        Kelly f = (win_rate * avg_win/avg_loss - (1 - win_rate)) / (avg_win/avg_loss)
        Half-Kelly = f / 2

    Args:
        min_trades: Minimum trades required before Kelly activates

    Returns:
        Half-Kelly fraction (0.0 to 1.0), or None if insufficient history.
    """
    try:
        if not _TRADE_MEMORY_FILE.exists():
            return None
        with open(_TRADE_MEMORY_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Flatten all trades across symbols
    all_trades = []
    for trades in data.values():
        all_trades.extend(trades)

    if len(all_trades) < min_trades:
        return None

    # Use last 200 trades for recency
    recent = all_trades[-200:]
    wins = [t for t in recent if t.get('pnl_pct', 0) > 0]
    losses = [t for t in recent if t.get('pnl_pct', 0) < 0]

    if not wins or not losses:
        return None

    win_rate = len(wins) / len(recent)
    avg_win = np.mean([t['pnl_pct'] for t in wins])
    avg_loss = abs(np.mean([t['pnl_pct'] for t in losses]))

    if avg_loss == 0:
        return None

    win_loss_ratio = avg_win / avg_loss
    kelly_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    # Half-Kelly, clamped to [0.05, 0.25]
    half_kelly = max(0.05, min(0.25, kelly_f / 2))
    return half_kelly


def kelly_position_size(base_notional: float, equity: float,
                        min_trades: int = 50) -> float:
    """Compute Kelly-based position size with bounds.

    Args:
        base_notional: Default fixed notional per trade
        equity: Current account equity
        min_trades: Minimum trades before Kelly activates

    Returns:
        Adjusted notional (floored at base, capped at 3x base).
    """
    kelly_f = compute_kelly_fraction(min_trades)
    if kelly_f is None:
        return base_notional

    kelly_notional = kelly_f * equity
    # Floor at base_notional, ceiling at 3x base
    return max(base_notional, min(base_notional * 3, kelly_notional))
