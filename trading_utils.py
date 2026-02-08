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
    """Choose inference device: CPU if GPU is busy/unavailable, else default."""
    if not is_gpu_available():
        return 'cpu'
    return None  # None = use default device


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

def predict_symbol(api, symbol, bear_model, bear_config, bull_model, bull_config,
                   scaler_X, feature_cols, inference_device,
                   asset_type='crypto', benchmark_close=None):
    """Run both bear and bull predictions for a single symbol.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca symbol (e.g. 'BTC/USD' or 'TSLA')
        bear_model/bear_config: bear model and its config dict
        bull_model/bull_config: bull model and its config dict
        scaler_X: feature scaler
        feature_cols: list of feature column names
        inference_device: device string or None for default
        asset_type: 'crypto' or 'stock'
        benchmark_close: SPY close Series (stocks) or BTC close Series (crypto)

    Returns:
        (symbol, bear_pred, bull_pred) tuple
    """
    # Route the benchmark to the correct kwarg based on asset type
    extra_kwargs = {}
    if asset_type == 'stock':
        extra_kwargs['spy_close'] = benchmark_close
    else:
        extra_kwargs['btc_close'] = benchmark_close

    bear_pred = get_live_prediction(
        symbol, bear_model, scaler_X, bear_config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type=asset_type, **extra_kwargs,
    )
    bull_pred = get_live_prediction(
        symbol, bull_model, scaler_X, bull_config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type=asset_type, **extra_kwargs,
    )
    return symbol, bear_pred, bull_pred
