import alpaca_trade_api as tradeapi
import time
import datetime
import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import TimeFrame

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# We use Ethereum for testing because it has high volume 24/7
CRYPTO_SYMBOL = 'ETH/USD' 

def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def place_crypto_order(api, symbol, side, notional_value=50):
    """
    Crypto often requires buying by 'notional' (dollar amount) 
    rather than share count to handle decimals easily.
    We will trade $50 worth of ETH per hour for this test.
    """
    try:
        api.submit_order(
            symbol=symbol,
            notional=notional_value, # Buy/Sell $50 worth
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f" -> SUCCESS: {side} ${notional_value} of {symbol}")
        return True
    except Exception as e:
        print(f" -> ERROR: {e}")
        return False

def run_crypto_bot():
    api = get_api()
    
    # State tracking
    has_position = False 
    
    print("--- JETSON CRYPTO BOT STARTED (WEEKEND MODE) ---")
    print(f"Target: {CRYPTO_SYMBOL}")
    print("Logic: Buy $50 -> Wait 1 Hour -> Sell $50 -> Buy $50")

    # FORCE START: We execute a trade immediately to prove it works, 
    # then settle into the hourly loop.
    print("\n[INITIALIZATION] Executing first trade to test connection...")
    if place_crypto_order(api, CRYPTO_SYMBOL, 'buy', 50):
        has_position = True
    
    while True:
        # Calculate time until next hour
        now = datetime.datetime.now()
        next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        seconds_to_wait = (next_hour - now).total_seconds()
        
        print(f"\n[WAITING] Next trade at {next_hour.strftime('%H:%M')} (in {seconds_to_wait/60:.1f} mins)")
        
        # Sleep until the top of the hour
        time.sleep(seconds_to_wait)
        
        print(f"\n--- EXECUTING HOURLY CYCLE: {datetime.datetime.now().strftime('%H:%M')} ---")
        
        # 1. SELL PREVIOUS
        if has_position:
            print("Step 1: Selling previous hour's position...")
            place_crypto_order(api, CRYPTO_SYMBOL, 'sell', 50)
            time.sleep(5) # Give the API a moment to process
        
        # 2. BUY NEW
        print("Step 2: Buying new position...")
        if place_crypto_order(api, CRYPTO_SYMBOL, 'buy', 50):
            has_position = True
        
        # Buffer to prevent double-firing
        time.sleep(60)

if __name__ == "__main__":
    run_crypto_bot()
