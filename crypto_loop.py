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

# Top 10 cryptos by market cap (USD pairs on Alpaca)
CRYPTO_SYMBOLS = [
    'BTC/USD',
    'ETH/USD',
    'XRP/USD',
    'SOL/USD',
    'DOGE/USD',
    'LINK/USD',
    'AVAX/USD',
    'DOT/USD',
    'LTC/USD',
    'UNI/USD',
]

NOTIONAL_PER_SYMBOL = 25  # $25 per symbol per cycle

def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def place_crypto_order(api, symbol, side, notional_value):
    try:
        api.submit_order(
            symbol=symbol,
            notional=notional_value,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        print(f"  {symbol}: {side} ${notional_value} -> SUCCESS")
        return True
    except Exception as e:
        print(f"  {symbol}: {side} ${notional_value} -> ERROR: {e}")
        return False

def run_crypto_bot():
    api = get_api()

    # Track which symbols we hold positions in
    positions = set()

    print("--- JETSON CRYPTO BOT STARTED ---")
    print(f"Symbols: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"Notional: ${NOTIONAL_PER_SYMBOL} per symbol per cycle")
    print(f"Total per cycle: ${NOTIONAL_PER_SYMBOL * len(CRYPTO_SYMBOLS)}")

    # Initial buy across all symbols
    print("\n[INITIALIZATION] Buying initial positions...")
    for symbol in CRYPTO_SYMBOLS:
        if place_crypto_order(api, symbol, 'buy', NOTIONAL_PER_SYMBOL):
            positions.add(symbol)
        time.sleep(1)

    while True:
        now = datetime.datetime.now()
        next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        seconds_to_wait = (next_hour - now).total_seconds()

        print(f"\n[WAITING] Next cycle at {next_hour.strftime('%H:%M')} (in {seconds_to_wait/60:.1f} mins)")
        time.sleep(seconds_to_wait)

        print(f"\n--- HOURLY CYCLE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ---")

        # 1. SELL all current positions
        if positions:
            print("Step 1: Selling positions...")
            for symbol in list(positions):
                place_crypto_order(api, symbol, 'sell', NOTIONAL_PER_SYMBOL)
                time.sleep(1)
            positions.clear()
            time.sleep(3)

        # 2. BUY fresh positions
        print("Step 2: Buying new positions...")
        for symbol in CRYPTO_SYMBOLS:
            if place_crypto_order(api, symbol, 'buy', NOTIONAL_PER_SYMBOL):
                positions.add(symbol)
            time.sleep(1)

        # Buffer to prevent double-firing
        time.sleep(60)

if __name__ == "__main__":
    run_crypto_bot()
