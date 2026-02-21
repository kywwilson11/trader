"""Shared utilities for the crypto and stock trading loops.

Centralizes duplicated code: API construction, model hot-reload helpers,
cooldown tracking, inference device selection, and the predict_symbol wrapper.
"""

import os
import datetime

import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from hw_monitor import is_gpu_available
from predict_now import get_live_prediction

load_dotenv()


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
