# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
import os
import torch
import torch.nn as nn
from model import CryptoLSTM
from indicators import compute_features, compute_stock_features

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'stock_predictor.pth'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'
MODEL_CONFIG_PATH = 'model_config.pkl'

# Dual model paths
BEAR_MODEL_PATH = 'bear_model.pth'
BEAR_CONFIG_PATH = 'bear_config.pkl'
BULL_MODEL_PATH = 'bull_model.pth'
BULL_CONFIG_PATH = 'bull_config.pkl'


def _prefixed_paths(prefix):
    """Return (bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols) paths for a prefix."""
    p = f'{prefix}_' if prefix else ''
    return {
        'bear_model': f'{p}bear_model.pth',
        'bear_config': f'{p}bear_config.pkl',
        'bull_model': f'{p}bull_model.pth',
        'bull_config': f'{p}bull_config.pkl',
        'scaler_X': f'{p}scaler_X.pkl',
        'feature_cols': f'{p}feature_cols.pkl',
        'default_model': MODEL_PATH,
        'default_config': MODEL_CONFIG_PATH,
    }


# --- LOAD MODEL AND SCALERS ONCE ---
def load_model(model_type='default', inference_device=None, prefix=''):
    """Load a model by type: 'default', 'bear', or 'bull'.

    Args:
        model_type: Which model to load
        inference_device: Override device for inference (e.g. 'cpu' for GPU fallback)
        prefix: File prefix (e.g. 'stock' -> stock_bear_model.pth)

    Returns:
        (model, scaler_X, config, seq_len, feature_cols)
    """
    dev = torch.device(inference_device) if inference_device else device
    paths = _prefixed_paths(prefix)

    scaler_X = joblib.load(paths['scaler_X'])
    feature_cols = joblib.load(paths['feature_cols'])

    if model_type == 'bear':
        config = joblib.load(paths['bear_config'])
        model_path = paths['bear_model']
    elif model_type == 'bull':
        config = joblib.load(paths['bull_config'])
        model_path = paths['bull_model']
    else:
        config = joblib.load(paths['default_config'])
        model_path = paths['default_model']

    num_classes = config.get('num_classes', 3)
    model = CryptoLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=num_classes,
    ).to(dev)
    model.load_state_dict(torch.load(model_path, map_location=dev, weights_only=True))
    model.eval()

    # Try to JIT-trace for faster inference (LSTM compatible with trace)
    try:
        dummy = torch.randn(1, config['seq_len'], config['input_dim']).to(dev)
        model = torch.jit.trace(model, dummy)
        print(f"  [JIT] Model traced successfully ({model_type})")
    except Exception as e:
        print(f"  [JIT] Trace failed ({model_type}): {e}, using eager mode")

    return model, scaler_X, config, config['seq_len'], feature_cols


def load_dual_models(inference_device=None, prefix=''):
    """Load both bear and bull models. Falls back to default model for both if dual models don't exist.

    Args:
        inference_device: Override device for inference
        prefix: File prefix (e.g. 'stock' -> stock_bear_model.pth)

    Returns:
        (bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols)
    """
    paths = _prefixed_paths(prefix)
    has_dual = os.path.exists(paths['bear_model']) and os.path.exists(paths['bull_model'])

    pfx_label = f"{prefix} " if prefix else ""
    if has_dual:
        bear_model, scaler_X, bear_config, _, feature_cols = load_model('bear', inference_device, prefix)
        bull_model, _, bull_config, _, _ = load_model('bull', inference_device, prefix)
        print(f"{pfx_label}Dual models loaded: bear(seq={bear_config['seq_len']}, th={bear_config.get('bull_threshold', 0.15):.2f}) "
              f"bull(seq={bull_config['seq_len']}, th={bull_config.get('bull_threshold', 0.15):.2f})")
    else:
        print(f"No {pfx_label}dual models found, falling back to default model for both")
        model, scaler_X, config, _, feature_cols = load_model('default', inference_device, prefix)
        bear_model = model
        bear_config = config
        bull_model = model
        bull_config = config

    return bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols


def _fetch_bars_alpaca(api, symbol, limit=120):
    """Fetch hourly bars from Alpaca's crypto data API.
    symbol: Alpaca format e.g. 'BTC/USD'
    Returns a DataFrame with OHLCV columns or None.
    """
    from datetime import datetime, timedelta, timezone
    try:
        start = datetime.now(timezone.utc) - timedelta(days=6)
        bars = api.get_crypto_bars(symbol, '1Hour', start=start.isoformat(), limit=limit)
        rows = []
        for bar in bars[symbol]:
            rows.append({
                'Open': float(bar.o),
                'High': float(bar.h),
                'Low': float(bar.l),
                'Close': float(bar.c),
                'Volume': float(bar.v),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Create a datetime index from bar timestamps
        timestamps = [bar.t for bar in bars[symbol]]
        df.index = pd.DatetimeIndex(timestamps)
        df.index.name = 'Datetime'
        return df
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def _fetch_bars_yfinance(symbol):
    """Fetch hourly bars from yfinance (standalone/fallback).
    symbol: yfinance format e.g. 'BTC-USD'
    """
    df = yf.download(symbol, period="5d", interval="1h", progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _fetch_stock_bars_alpaca(api, symbol, limit=120):
    """Fetch hourly bars from Alpaca's stock data API.
    symbol: stock symbol e.g. 'TSLA'
    Returns a DataFrame with OHLCV columns or None.
    """
    from datetime import datetime, timedelta, timezone
    try:
        start = datetime.now(timezone.utc) - timedelta(days=6)
        bars = api.get_bars(symbol, '1Hour', start=start.isoformat(), limit=limit)
        rows = []
        timestamps = []
        for bar in bars:
            rows.append({
                'Open': float(bar.o),
                'High': float(bar.h),
                'Low': float(bar.l),
                'Close': float(bar.c),
                'Volume': float(bar.v),
            })
            timestamps.append(bar.t)
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.DatetimeIndex(timestamps)
        df.index.name = 'Datetime'
        return df
    except Exception as e:
        print(f"  [ALPACA BARS] Error fetching {symbol}: {e}")
        return None


def _fetch_spy_bars_alpaca(api, limit=120):
    """Fetch SPY hourly bars for relative strength calculation."""
    return _fetch_stock_bars_alpaca(api, 'SPY', limit)


def get_live_prediction(symbol, model, scaler_X, config_or_seq,
                        seq_len_or_feature_cols=None, feature_cols=None,
                        api=None, inference_device=None,
                        asset_type='crypto', spy_close=None):
    """Get prediction for a symbol. Returns a score:
    - Positive = bullish (higher = more confident)
    - Negative = bearish (lower = more confident)
    - Near zero = neutral

    Args:
        symbol: Ticker symbol (Alpaca format 'BTC/USD' if api provided, else yfinance 'BTC-USD')
        model: CryptoLSTM model
        scaler_X: Feature scaler
        config_or_seq: Config dict (new style) or legacy arg
        seq_len_or_feature_cols: feature_cols when config_or_seq is dict
        feature_cols: Explicit feature_cols override
        api: Alpaca API object — if provided, uses Alpaca bars instead of yfinance
        inference_device: Override device for inference (e.g. 'cpu')
        asset_type: 'crypto' or 'stock' — determines feature computation and data source
        spy_close: SPY close Series for stock relative strength (optional)
    """
    dev = torch.device(inference_device) if inference_device else device

    # Handle different calling conventions
    if isinstance(config_or_seq, dict):
        config = config_or_seq
        seq_len = config['seq_len']
        if feature_cols is None:
            feature_cols = seq_len_or_feature_cols
    else:
        try:
            feature_cols_loaded = joblib.load(FEATURE_COLS_PATH)
            config = joblib.load(MODEL_CONFIG_PATH)
            seq_len = config['seq_len']
            if feature_cols is None:
                feature_cols = feature_cols_loaded
        except FileNotFoundError:
            print(f"  {symbol}: Cannot load config for prediction")
            return None

    print(f"\n--- ANALYZING {symbol} ---")

    # Fetch data: different sources for stock vs crypto
    if asset_type == 'stock':
        if api is not None:
            df = _fetch_stock_bars_alpaca(api, symbol)
        else:
            df = _fetch_bars_yfinance(symbol)
    else:
        if api is not None:
            alpaca_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            df = _fetch_bars_alpaca(api, alpaca_sym)
        else:
            yf_sym = symbol.replace('/', '-') if '/' in symbol else symbol
            df = _fetch_bars_yfinance(yf_sym)

    if df is None or df.empty:
        print("Error: No data found for symbol.")
        return None

    # Use stock features or crypto features
    if asset_type == 'stock':
        df = compute_stock_features(df, spy_close=spy_close)
    else:
        df = compute_features(df)
    df = df.dropna()

    if len(df) < seq_len:
        print(f"  Not enough data for sequence (need {seq_len}, have {len(df)})")
        return None

    try:
        current_features = df[feature_cols].values
    except KeyError as e:
        print(f"  Feature mismatch: {e}")
        return None

    current_features_scaled = scaler_X.transform(current_features)

    sequence = current_features_scaled[-seq_len:]
    sequence = sequence.reshape(1, seq_len, -1)

    tensor_input = torch.tensor(sequence, dtype=torch.float32).to(dev)

    with torch.inference_mode():
        logits = model(tensor_input)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # probs: [bearish, neutral, bullish]
    bear_prob, neut_prob, bull_prob = probs

    # Convert to a score: bull_prob - bear_prob
    # Range: [-1, +1], positive = bullish, negative = bearish
    score = bull_prob - bear_prob

    # Scale score by the model's trained bull_threshold so predicted_return
    # is proportional to the magnitude the model was trained to detect
    bull_threshold = config.get('bull_threshold', 0.15)
    predicted_return = score * bull_threshold

    price = df['Close'].iloc[-1]
    print(f"Current Price:   ${price:.2f}")
    print(f"Probabilities:   Bear={bear_prob:.1%}  Neut={neut_prob:.1%}  Bull={bull_prob:.1%}")
    print(f"Score: {score:+.3f} -> Predicted Return: {predicted_return:+.4f}%")

    if predicted_return > bull_threshold:
        print("Recommendation:  [BUY]")
    elif predicted_return < -bull_threshold:
        print("Recommendation:  [SELL/AVOID]")
    else:
        print("Recommendation:  [HOLD/WEAK]")

    return predicted_return


if __name__ == "__main__":
    print(f"Using device: {device}")

    # Try dual models first, fall back to default
    try:
        bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = load_dual_models()
        dual = bear_model is not bull_model
    except FileNotFoundError:
        try:
            model, scaler_X, config, seq_len, feature_cols = load_model()
            bear_model = bull_model = model
            bear_config = bull_config = config
            dual = False
        except FileNotFoundError:
            print("Error: Model files not found. Did you run hypersearch_dual.py?")
            exit(1)

    symbols = [
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
        'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'BCH-USD',
    ]
    for sym in symbols:
        if dual:
            print(f"\n{'='*40} BEAR MODEL {'='*40}")
            get_live_prediction(sym, bear_model, scaler_X, bear_config, feature_cols)
            print(f"\n{'='*40} BULL MODEL {'='*40}")
            get_live_prediction(sym, bull_model, scaler_X, bull_config, feature_cols)
        else:
            get_live_prediction(sym, bear_model, scaler_X, bear_config, feature_cols)
