import alpaca_trade_api as tradeapi
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

def get_trading_status():
    # Connect to the API
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

    try:
        # 1. Get Account Info
        account = api.get_account()
        
        # 2. Check Financials
        equity = float(account.equity)
        day_trade_count = int(account.daytrade_count)
        is_pdt_flagged = account.pattern_day_trader
        
        print(f"\n--- ACCOUNT STATUS ---")
        print(f"Equity:       ${equity:,.2f}")
        print(f"Day Trades:   {day_trade_count} / 3 (Rolling 5-day window)")
        print(f"PDT Flagged:  {is_pdt_flagged}")
        print(f"Status:       {account.status}")

        # 3. The Logic Gate
        if equity >= 25000:
            print("\n[DECISION]: UNLIMITED MODE")
            print("Account is over $25k. We can execute the Hourly Strategy.")
            return "UNLIMITED"
            
        else:
            print("\n[DECISION]: CONSERVATIVE MODE")
            print("Account is under $25k.")
            
            # Logic: If we have burned 3 trades, we stop to prevent the 4th (Ban).
            if day_trade_count >= 3:
                print("CRITICAL: You are at the limit (3 trades). HALTING TRADING.")
                return "HALT"
            else:
                remaining = 3 - day_trade_count
                print(f"Safe to trade. You have {remaining} day trades remaining this week.")
                return "SAFE_TO_TRADE"

    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return "ERROR"

if __name__ == "__main__":
    # Run the check
    status = get_trading_status()
