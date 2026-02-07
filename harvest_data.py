import yfinance as yf
import pandas as pd
import pandas_ta as ta # You must install this: pip install pandas_ta

# Stocks that fit your "High Beta" profile
TICKERS = ['NVDA', 'TSLA', 'AMD', 'META', 'AMZN', 'QQQ']

def prepare_data(ticker):
    print(f"Processing {ticker}...")
    
    # 1. Download hourly data (last 730 days is max for hourly usually, but we'll do 60 days for speed)
    df = yf.download(ticker, period="1y", interval="1h", progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    # 2. Calculate Technical Indicators (The "Features")
    # RSI: Relative Strength Index
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD: Momentum
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # ATR: Volatility (How much does it move?)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 3. Create the "Target" (The Answer Key)
    # We want to know: "Did the price go up in the NEXT hour?"
    # We shift the Close price back by 1 to compare current hour to next hour.
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100
    
    # Drop the last row (it has no "NextClose")
    df = df.dropna()
    
    return df

# Main Loop
all_data = []
for t in TICKERS:
    stock_df = prepare_data(t)
    if stock_df is not None:
        stock_df['Ticker'] = t # Label the data
        all_data.append(stock_df)

# Combine and Save
final_df = pd.concat(all_data)
final_df.to_csv('training_data.csv')
print(f"Done! Saved {len(final_df)} rows of training data to training_data.csv")
