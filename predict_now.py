# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf
import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'stock_predictor.pth'
SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
FEATURE_COLS_PATH = 'feature_cols.pkl'
MODEL_CONFIG_PATH = 'model_config.pkl'


# --- MODEL DEFINITION (must match train_model.py) ---
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=3):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


# --- INDICATOR FUNCTIONS (matches harvest_data.py exactly) ---
def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, histogram, signal_line


def compute_atr(high, low, close, length=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def compute_bbands(close, length=20, std=2):
    sma = close.rolling(window=length).mean()
    std_dev = close.rolling(window=length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bandwidth = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)
    return lower, sma, upper, bandwidth, pct_b


def compute_stoch(high, low, close, k=14, d=3, smooth_k=3):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def compute_obv(close, volume):
    sign = np.sign(close.diff())
    sign.iloc[0] = 0
    return (sign * volume).cumsum()


def compute_roc(close, length=12):
    return ((close - close.shift(length)) / close.shift(length)) * 100


def compute_features(df):
    """Compute all features matching harvest_data.py exactly."""
    df['RSI'] = compute_rsi(df['Close'], length=14)

    macd_line, macd_hist, macd_signal = compute_macd(df['Close'])
    df['MACD_12_26_9'] = macd_line
    df['MACDh_12_26_9'] = macd_hist
    df['MACDs_12_26_9'] = macd_signal

    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], length=14)

    bb_lower, bb_mid, bb_upper, bb_bw, bb_pct = compute_bbands(df['Close'], length=20, std=2)
    df['BBL_20_2.0'] = bb_lower
    df['BBM_20_2.0'] = bb_mid
    df['BBU_20_2.0'] = bb_upper
    df['BBB_20_2.0'] = bb_bw
    df['BBP_20_2.0'] = bb_pct

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
    df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']

    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['OBV'] = compute_obv(df['Close'], df['Volume'])

    df['ROC'] = compute_roc(df['Close'], length=12)

    stoch_k, stoch_d = compute_stoch(df['High'], df['Low'], df['Close'])
    df['STOCHk_14_3_3'] = stoch_k
    df['STOCHd_14_3_3'] = stoch_d

    idx = df.index
    hour = idx.hour
    day = idx.dayofweek
    df['Hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['Day_sin'] = np.sin(2 * np.pi * day / 7)
    df['Day_cos'] = np.cos(2 * np.pi * day / 7)

    return df


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

    # Map score to a % prediction that crypto_loop.py understands:
    # score > 0 => positive predicted return (buy signal)
    # score < 0 => negative predicted return (sell signal)
    # Scale so that crypto_loop's thresholds (>0.2 buy, <-0.1 sell) work
    predicted_return = score * 0.5  # ±0.5% at max confidence

    price = df['Close'].iloc[-1]
    print(f"Current Price:   ${price:.2f}")
    print(f"Probabilities:   Bear={bear_prob:.1%}  Neut={neut_prob:.1%}  Bull={bull_prob:.1%}")
    print(f"Score: {score:+.3f} -> Predicted Return: {predicted_return:+.4f}%")

    if predicted_return > 0.2:
        print("Recommendation:  [BUY]")
    elif predicted_return < -0.1:
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
