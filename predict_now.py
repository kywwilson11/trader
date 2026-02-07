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

# --- MODEL DEFINITION (must match train_model.py) ---
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- INDICATOR FUNCTIONS (replaces pandas_ta for portability) ---
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

# --- LOAD MODEL AND SCALERS ONCE ---
def load_model():
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    input_dim = scaler_X.n_features_in_
    model = StockPredictor(input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    return model, scaler_X, scaler_y, input_dim

def get_live_prediction(symbol, model, scaler_X, scaler_y, input_dim):
    print(f"\n--- ANALYZING {symbol} ---")

    # 1. Get Recent Data (enough for moving averages to settle)
    df = yf.download(symbol, period="5d", interval="1h", progress=False)

    if df.empty:
        print("Error: No data found for symbol.")
        return None

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Calculate Indicators (MUST match training exactly)
    df['RSI'] = compute_rsi(df['Close'], length=14)
    macd_line, macd_hist, macd_signal = compute_macd(df['Close'])
    df['MACD_12_26_9'] = macd_line
    df['MACDh_12_26_9'] = macd_hist
    df['MACDs_12_26_9'] = macd_signal
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'], length=14)

    # 3. Prepare the last row for prediction
    last_row = df.iloc[-1:]
    feature_cols = [c for c in df.columns if c not in ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']]
    current_features = last_row[feature_cols].select_dtypes(include=[np.number]).values

    if current_features.shape[1] != input_dim:
        print(f"Shape Mismatch! Model expects {input_dim} features, but we found {current_features.shape[1]}.")
        return None

    # 4. Scale and Predict
    current_features_scaled = scaler_X.transform(current_features)
    tensor_input = torch.tensor(current_features_scaled, dtype=torch.float32).to(device)

    with torch.inference_mode():
        prediction_scaled = model(tensor_input)

    # 5. Un-Scale the Output
    prediction_cpu = prediction_scaled.cpu().numpy()
    predicted_return = scaler_y.inverse_transform(prediction_cpu)

    result = predicted_return[0][0]

    print(f"Current Price:   ${last_row['Close'].values[0]:.2f}")
    print(f"Predicted Move:  {result:+.4f}% (Next Hour)")

    if result > 0.2:
        print("Recommendation:  [BUY]")
    elif result < -0.1:
        print("Recommendation:  [SELL/AVOID]")
    else:
        print("Recommendation:  [HOLD/WEAK]")

    return result

if __name__ == "__main__":
    print(f"Using device: {device}")
    try:
        model, scaler_X, scaler_y, input_dim = load_model()
    except FileNotFoundError:
        print("Error: Model or Scaler files not found. Did you run train_model.py?")
        exit(1)

    # Top 10 cryptos (yfinance uses '-' instead of '/')
    symbols = [
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD',
        'LINK-USD', 'AVAX-USD', 'DOT-USD', 'LTC-USD', 'UNI-USD',
    ]
    for sym in symbols:
        get_live_prediction(sym, model, scaler_X, scaler_y, input_dim)
