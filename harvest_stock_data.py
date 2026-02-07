import yfinance as yf
import pandas as pd
from indicators import compute_stock_features

# ~30 high-beta, liquid stocks (no penny stocks)
STOCK_TICKERS = [
    'AMD', 'PLTR', 'SNAP', 'ROKU', 'AFRM', 'HOOD', 'SHOP', 'NET', 'CRWD', 'COIN',
    'MARA', 'MSTR', 'UBER', 'SOFI', 'ABNB', 'DASH', 'RBLX', 'SMCI', 'MRVL', 'ARM',
    'FSLR', 'ENPH', 'OXY', 'MRNA', 'CRSP', 'ARKK', 'TQQQ', 'SOXL', 'TSLA', 'NVDA',
    'META',
]

BENCHMARK = 'SPY'


def fetch_spy_close():
    """Download SPY hourly data for benchmark relative strength."""
    print(f"Fetching benchmark ({BENCHMARK})...")
    df = yf.download(BENCHMARK, period="1y", interval="1h", prepost=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return None
    return df['Close']


def prepare_stock_data(ticker, spy_close=None):
    print(f"Processing {ticker}...")

    df = yf.download(ticker, period="1y", interval="1h", prepost=True, progress=False)

    # Flatten multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return None

    df = compute_stock_features(df, spy_close=spy_close)

    # Target
    df['NextClose'] = df['Close'].shift(-1)
    df['Target_Return'] = (df['NextClose'] - df['Close']) / df['Close'] * 100

    df = df.dropna()
    return df


# Main
spy_close = fetch_spy_close()

all_data = []
for t in STOCK_TICKERS:
    stock_df = prepare_stock_data(t, spy_close)
    if stock_df is not None:
        stock_df['Ticker'] = t
        all_data.append(stock_df)

# Combine and Save — sort chronologically for time-series split in training
final_df = pd.concat(all_data)
final_df = final_df.sort_index()
final_df.to_csv('stock_training_data.csv')
print(f"Done! Saved {len(final_df)} rows of stock training data to stock_training_data.csv")

# Summary
print(f"\nStocks harvested: {len(all_data)}/{len(STOCK_TICKERS)}")
print(f"Feature columns: {len([c for c in final_df.columns if c not in ['Target_Return', 'Ticker', 'NextClose']])}")
print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
