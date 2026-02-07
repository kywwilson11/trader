import alpaca_trade_api as tradeapi
import time
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# We will rotate these just to test the "New Buy" logic
TEST_TICKERS = ['SPY', 'QQQ'] 

def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def place_order(api, symbol, side, qty=1):
    try:
        # Check if market is open before submitting
        clock = api.get_clock()
        if not clock.is_open:
            print(f"[{side}] Market is closed. Skipping.")
            return False

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f" -> SUCCESS: {side} {qty} {symbol}")
        return True
    except Exception as e:
        print(f" -> ERROR: {e}")
        return False

def run_bot():
    api = get_api()
    
    # State: What do we currently own that needs selling?
    # In a real app, we would save this to a file (JSON/SQLite) so a reboot doesn't wipe it.
    last_hour_buy = None 
    
    print("--- JETSON TRADER STARTED ---")
    print("Waiting for top of the hour...")

    ticker_index = 0

    while True:
        # 1. Get current time
        now = datetime.datetime.now()
        
        # 2. Check if we are at the top of the hour (Minute 00)
        # We add a buffer (0-2 mins) to ensure we don't miss it or double-fire
        if now.minute == 0:
            
            print(f"\n--- EXECUTING HOURLY CYCLE: {now.strftime('%H:%M')} ---")
            
            # A. SELL PREVIOUS (If exists)
            if last_hour_buy:
                print(f"Step 1: Selling previous position ({last_hour_buy})...")
                place_order(api, last_hour_buy, 'sell')
            else:
                print("Step 1: Nothing to sell (First run).")

            # B. BUY NEW (The "Predicted" Winner)
            # Later, this comes from your Neural Network.
            # For now, we pick from the list.
            symbol_to_buy = TEST_TICKERS[ticker_index % len(TEST_TICKERS)]
            
            print(f"Step 2: Buying new position ({symbol_to_buy})...")
            success = place_order(api, symbol_to_buy, 'buy')
            
            if success:
                last_hour_buy = symbol_to_buy
                ticker_index += 1
            
            # C. SLEEP
            # Sleep for 60 seconds to ensure we don't fire again in the same minute
            print("Cycle complete. Sleeping...")
            time.sleep(65) 
            
        else:
            # Check again in 30 seconds
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
