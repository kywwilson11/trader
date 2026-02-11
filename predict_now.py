"""ML prediction engine — model loading, dual bear/bull inference, live predictions.

Supports two model versions:
  v1: Dual CryptoLSTM classifiers (bear + bull) with softmax → score → predicted_return
  v2: Single RegressionLSTM that directly outputs predicted return percentage

Loads models from disk, optionally JIT-traces them for faster inference,
and provides get_live_prediction() which fetches recent bars, computes features,
and returns a predicted-return score.

Bar-fetching and ATR logic live in market_data.py.
"""

# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import joblib
import os
import torch
from model import CryptoLSTM
from model_v2 import RegressionLSTM
from indicators import compute_features, compute_stock_features
from market_data import (
    fetch_bars_alpaca, fetch_bars_yfinance,
    fetch_stock_bars_alpaca,
)

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'stock_predictor.pth'
SCALER_X_PATH = 'scaler_X.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'
MODEL_CONFIG_PATH = 'model_config.pkl'

# Dual model paths
BEAR_MODEL_PATH = 'bear_model.pth'
BEAR_CONFIG_PATH = 'bear_config.pkl'
BULL_MODEL_PATH = 'bull_model.pth'
BULL_CONFIG_PATH = 'bull_config.pkl'


def _prefixed_paths(prefix):
    """Return dict of file paths for a given model prefix (e.g. 'stock')."""
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
        # v2 regression model paths
        'v2_model': f'{p}model_v2.pth',
        'v2_config': f'{p}config_v2.pkl',
        'v2_scaler': f'{p}scaler_v2.pkl',
        'v2_features': f'{p}feature_cols_v2.pkl',
    }


# --- MODEL LOADING ---

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
    """Load both bear and bull models. Falls back to default model if dual models don't exist.

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
        bear_fb = bear_config.get('forward_bars', 4)
        bull_fb = bull_config.get('forward_bars', 4)
        print(f"{pfx_label}Dual models loaded: "
              f"bear(seq={bear_config['seq_len']}, th={bear_config.get('bull_threshold', 0.15):.2f}, fb={bear_fb}) "
              f"bull(seq={bull_config['seq_len']}, th={bull_config.get('bull_threshold', 0.15):.2f}, fb={bull_fb})")
    else:
        print(f"No {pfx_label}dual models found, falling back to default model for both")
        model, scaler_X, config, _, feature_cols = load_model('default', inference_device, prefix)
        bear_model = model
        bear_config = config
        bull_model = model
        bull_config = config

    return bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols


# --- V2 REGRESSION MODEL LOADING ---

def load_model_v2(inference_device=None, prefix=''):
    """Load a v2 regression model.

    Returns:
        (model, scaler_X, config, seq_len, feature_cols)
    """
    dev = torch.device(inference_device) if inference_device else device
    paths = _prefixed_paths(prefix)

    config = joblib.load(paths['v2_config'])
    scaler_X = joblib.load(paths['v2_scaler'])
    feature_cols = joblib.load(paths['v2_features'])

    model = RegressionLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        n_heads=config.get('n_heads', 4),
    ).to(dev)
    model.load_state_dict(torch.load(paths['v2_model'], map_location=dev, weights_only=True))
    model.eval()

    try:
        dummy = torch.randn(1, config['seq_len'], config['input_dim']).to(dev)
        model = torch.jit.trace(model, dummy, check_trace=False)
        print(f"  [JIT] v2 model traced successfully")
    except Exception as e:
        print(f"  [JIT] v2 trace failed: {e}, using eager mode")

    return model, scaler_X, config, config['seq_len'], feature_cols


def load_v2_models(inference_device=None, prefix=''):
    """Try to load v2 regression model. Returns None tuple if not found.

    Returns:
        (model, config, scaler_X, feature_cols) or (None, None, None, None)
    """
    paths = _prefixed_paths(prefix)
    if not os.path.exists(paths['v2_model']):
        return None, None, None, None

    pfx_label = f"{prefix} " if prefix else ""
    model, scaler_X, config, seq_len, feature_cols = load_model_v2(inference_device, prefix)
    th = config.get('trade_threshold', 0.15)
    fb = config.get('forward_bars', 4)
    print(f"{pfx_label}v2 regression model loaded: "
          f"seq={seq_len}, th={th:.2f}, fb={fb}, "
          f"heads={config.get('n_heads', 4)}")
    return model, config, scaler_X, feature_cols


# --- LIVE PREDICTION ---

def get_live_prediction(symbol, model, scaler_X, config, feature_cols,
                        api=None, inference_device=None,
                        asset_type='crypto', spy_close=None, btc_close=None):
    """Get prediction for a symbol. Returns a predicted-return score.

    Score interpretation:
        Positive = bullish (higher = more confident)
        Negative = bearish (lower = more confident)
        Near zero = neutral

    Args:
        symbol: Ticker symbol (Alpaca format 'BTC/USD' if api provided, else yfinance 'BTC-USD')
        model: CryptoLSTM model (or JIT-traced variant)
        scaler_X: Feature scaler
        config: Model config dict (must contain 'seq_len', 'bull_threshold')
        feature_cols: List of feature column names
        api: Alpaca API object — if provided, uses Alpaca bars instead of yfinance
        inference_device: Override device for inference (e.g. 'cpu')
        asset_type: 'crypto' or 'stock' — determines feature computation and data source
        spy_close: SPY close Series for stock relative strength (optional)
        btc_close: BTC/USD close Series for crypto cross-asset features (optional)

    Returns:
        float predicted_return, or None on error
    """
    dev = torch.device(inference_device) if inference_device else device
    seq_len = config['seq_len']

    print(f"\n--- ANALYZING {symbol} ---")

    # --- Fetch bars ---
    if asset_type == 'stock':
        if api is not None:
            df = fetch_stock_bars_alpaca(api, symbol)
        else:
            df = fetch_bars_yfinance(symbol)
    else:
        if api is not None:
            alpaca_sym = symbol.replace('-', '/') if '-' in symbol else symbol
            df = fetch_bars_alpaca(api, alpaca_sym)
        else:
            yf_sym = symbol.replace('/', '-') if '/' in symbol else symbol
            df = fetch_bars_yfinance(yf_sym)

    if df is None or df.empty:
        print("Error: No data found for symbol.")
        return None

    # --- Compute technical features ---
    if asset_type == 'stock':
        df = compute_stock_features(df, spy_close=spy_close)
    else:
        df = compute_features(df, btc_close=btc_close)
    df = df.dropna()

    if len(df) < seq_len:
        print(f"  Not enough data for sequence (need {seq_len}, have {len(df)})")
        return None

    # Inject live sentiment if the model was trained with it
    if 'Daily_Sentiment' in feature_cols and 'Daily_Sentiment' not in df.columns:
        try:
            from sentiment_history import get_live_daily_sentiment
            df['Daily_Sentiment'] = get_live_daily_sentiment(symbol, asset_type)
        except Exception as e:
            print(f"  [SENTIMENT] Live sentiment unavailable: {e}")
            df['Daily_Sentiment'] = 0.0

    try:
        current_features = df[feature_cols].values
    except KeyError as e:
        print(f"  Feature mismatch: {e}")
        return None

    # --- Scale and build input tensor ---
    current_features_scaled = scaler_X.transform(current_features)
    sequence = current_features_scaled[-seq_len:]
    sequence = sequence.reshape(1, seq_len, -1)
    tensor_input = torch.tensor(sequence, dtype=torch.float32).to(dev)

    # --- Run inference ---
    model_version = config.get('model_version', 1)

    with torch.inference_mode():
        output = model(tensor_input)

    price = df['Close'].iloc[-1]

    if model_version >= 2:
        # v2 regression: model outputs predicted return directly
        predicted_return = float(output.cpu().item())
        trade_threshold = config.get('trade_threshold', 0.15)

        print(f"Current Price:   ${price:.2f}")
        print(f"Predicted Return: {predicted_return:+.4f}% (threshold={trade_threshold:.2f})")

        if predicted_return > trade_threshold:
            print("Recommendation:  [BUY]")
        elif predicted_return < -trade_threshold:
            print("Recommendation:  [SELL/AVOID]")
        else:
            print("Recommendation:  [HOLD/WEAK]")
    else:
        # v1 classification: softmax → score → predicted_return
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        bear_prob, neut_prob, bull_prob = probs
        score = bull_prob - bear_prob
        bull_threshold = config.get('bull_threshold', 0.15)
        predicted_return = score * bull_threshold

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
        'LINK-USD',
    ]
    for sym in symbols:
        if dual:
            print(f"\n{'='*40} BEAR MODEL {'='*40}")
            get_live_prediction(sym, bear_model, scaler_X, bear_config, feature_cols)
            print(f"\n{'='*40} BULL MODEL {'='*40}")
            get_live_prediction(sym, bull_model, scaler_X, bull_config, feature_cols)
        else:
            get_live_prediction(sym, bear_model, scaler_X, bear_config, feature_cols)
