# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from model import CryptoLSTM
from indicators import compute_features

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'stock_predictor.pth'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'
MODEL_CONFIG_PATH = 'model_config.pkl'


# --- LOAD MODEL AND SCALERS ONCE ---
def load_model():
    scaler_X = joblib.load(SCALER_X_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    config = joblib.load(MODEL_CONFIG_PATH)

    num_classes = config.get('num_classes', 3)
    model = CryptoLSTM(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_classes=num_classes,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Return 5 values for backward compat with crypto_loop.py's unpacking
    # The 4th value is seq_len (was input_dim in old interface)
    return model, scaler_X, config, config['seq_len'], feature_cols


def get_live_prediction(symbol, model, scaler_X, config_or_seq, seq_len_or_feature_cols=None, feature_cols=None):
    """Get prediction for a symbol. Returns a score:
    - Positive = bullish (higher = more confident)
    - Negative = bearish (lower = more confident)
    - Near zero = neutral

    Supports flexible calling conventions for backward compat.
    """
    # Handle different calling conventions
    if isinstance(config_or_seq, dict):
        # New style: get_live_prediction(sym, model, scaler_X, config, seq_len, feature_cols)
        config = config_or_seq
        seq_len = config['seq_len']
        if feature_cols is None:
            feature_cols = seq_len_or_feature_cols
    else:
        # Old style or intermediate: figure out from args
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

    df = yf.download(symbol, period="5d", interval="1h", progress=False)

    if df.empty:
        print("Error: No data found for symbol.")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

    tensor_input = torch.tensor(sequence, dtype=torch.float32).to(device)

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
    try:
        model, scaler_X, config, seq_len, feature_cols = load_model()
    except FileNotFoundError:
        print("Error: Model files not found. Did you run train_model.py?")
        exit(1)

    symbols = [
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
        'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'BCH-USD',
    ]
    for sym in symbols:
        get_live_prediction(sym, model, scaler_X, config, seq_len, feature_cols)
