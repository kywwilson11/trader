import time
import datetime
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import torch
import torch.nn as nn
import joblib
import numpy as np
import alpaca_trade_api as tradeapi

# --- CONFIGURATION ---
API_KEY = 'YOUR_PAPER_API_KEY'
SECRET_KEY = 'YOUR_PAPER_SECRET_KEY'
BASE_URL = 'https://paper-api.alpaca.markets'

# The basket of assets the AI will scan
TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'AVAX-USD']
TRADE_AMOUNT = 100  # Dollars per trade

# Model Files
MODEL_PATH = 'stock_predictor.pth'
SCALER_X = 'scaler_X.pkl'
SCALER_Y = 'scaler_y.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- AI ENGINE ---
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

def load_brain():
    """Loads the model and scalers into memory once."""
    print("Loading AI Model...")
    scaler_x = joblib.load(SCALER_X)
    scaler_y = joblib.load(SCALER_Y)
    input_dim = scaler_x.n_features_in_
    
    model = StockPredictor(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, scaler_x, scaler_y

def get_prediction(symbol, model, scaler_x, scaler_y):
    """Downloads data and asks the AI for a forecast."""
    try:
        # Download last 5 days hourly data
        df = yf.download(symbol, period="5d", interval="1h", progress=False)
        if df.empty: return -999 # Error signal

        # Calculate Indicators (MUST MATCH TRAINING EXACTLY)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Get last row features
        feature_cols = [c for c in df.columns if c not in ['Target_Return', 'Ticker', 'Date', 'NextClose']]
        last_row = df.iloc[-1:][feature_cols].select_dtypes(include=[np.number]).values
        
        # Predict
        if last_row.shape[1] != scaler_x.n_features_in_:
            return -999 # Shape mismatch
            
        features_scaled = scaler_x.transform(last_row)
        tensor_in = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            pred_scaled = model(tensor_in)
            
        pred_real = scaler_y.inverse_transform(pred_scaled.cpu().numpy())
        return pred_real[0][0]
        
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")
        return -999

# --- TRADING ENGINE ---
def run_autonomous_trader():
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    model, scaler_x, scaler_y = load_brain()
    
    last_buy_symbol = None
    
    print("\n--- JETSON AI TRADER ONLINE ---")
    print(f"Monitoring: {TICKERS}")
    
    while True:
        now = datetime.datetime.now()
        
        # Wait for top of the hour (minute 0)
        if now.minute == 0:
            print(f"\n[CYCLE START] {now.strftime('%H:%M')}")
            
            # 1. SELL PREVIOUS
            if last_buy_symbol:
                print(f" -> Selling previous: {last_buy_symbol}")
                try:
                    api.submit_order(symbol=last_buy_symbol, qty=None, notional=TRADE_AMOUNT, side='sell', type='market', time_in_force='gtc')
                except Exception as e:
                    print(f"Sell Error: {e}")
            
            # 2. SCAN & RANK
            print(" -> AI Scanning Market...")
            scores = []
            for ticker in TICKERS:
                score = get_prediction(ticker, model, scaler_x, scaler_y)
                scores.append((ticker, score))
                print(f"    {ticker}: {score:.4f}%")
            
            # Sort by Score (Highest First)
            scores.sort(key=lambda x: x[1], reverse=True)
            best_pick, best_score = scores[0]
            
            # 3. BUY BEST
            print(f" -> WINNER: {best_pick} (Pred: {best_score:.4f}%)")
            try:
                api.submit_order(symbol=best_pick, qty=None, notional=TRADE_AMOUNT, side='buy', type='market', time_in_force='gtc')
                last_buy_symbol = best_pick
                print(f" -> BOUGHT {best_pick}")
            except Exception as e:
                print(f"Buy Error: {e}")
                last_buy_symbol = None

            # Sleep to prevent double-execution
            print("[SLEEPING] Waiting for next hour...")
            time.sleep(120) 
            
        else:
            # Heartbeat every 30 seconds
            time.sleep(30)

if __name__ == "__main__":
    run_autonomous_trader()
