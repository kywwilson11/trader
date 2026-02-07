import alpaca_trade_api as tradeapi
import time
import datetime
import os
from dotenv import load_dotenv

from order_utils import (
    get_stock_quote, place_stock_limit_order, manage_order_lifecycle,
    verify_position, cancel_all_open_orders,
)

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# We will rotate these just to test the "New Buy" logic
TEST_TICKERS = ['SPY', 'QQQ']

def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

ORDER_TIMEOUT = 30  # seconds to wait for limit fill

def place_order(api, symbol, side, qty=1):
    """Place a limit order for stocks with lifecycle management."""
    try:
        # Check if market is open before submitting
        clock = api.get_clock()
        if not clock.is_open:
            print(f"[{side}] Market is closed. Skipping.")
            return False

        quote = get_stock_quote(api, symbol)
        if quote is None:
            print(f" -> No quote for {symbol}, falling back to market order")
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            print(f" -> SUCCESS: {side} {qty} {symbol} (market)")
            return True

        order = place_stock_limit_order(api, symbol, side, qty, quote,
                                         time_in_force='day')
        if order is None:
            return False

        result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                         fallback_to_market=True)
        if result and result.status == 'filled':
            return True

        # Market fallback may have been submitted
        if result and result.id != order.id:
            time.sleep(2)
            try:
                final = api.get_order(result.id)
                return final.status == 'filled'
            except Exception:
                pass

        return False
    except Exception as e:
        print(f" -> ERROR: {e}")
        return False

def run_bot():
    api = get_api()

    # Cancel stale orders from previous runs
    cancel_all_open_orders(api)

    # State: What do we currently own that needs selling?
    last_hour_buy = None

    # Check if we already hold a position from a previous run
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol in TEST_TICKERS:
                last_hour_buy = pos.symbol
                print(f"Existing position found: {last_hour_buy}")
                break
    except Exception:
        pass

    print("--- JETSON TRADER STARTED (SMART MODE) ---")
    print("Waiting for top of the hour...")

    ticker_index = 0

    while True:
        # 1. Get current time
        now = datetime.datetime.now()

        # 2. Check if we are at the top of the hour (Minute 00)
        if now.minute == 0:

            print(f"\n--- EXECUTING HOURLY CYCLE: {now.strftime('%H:%M')} ---")

            # A. SELL PREVIOUS (If exists)
            if last_hour_buy:
                print(f"Step 1: Selling previous position ({last_hour_buy})...")
                # Verify we actually hold this position
                pos = verify_position(api, last_hour_buy)
                if pos is not None:
                    place_order(api, last_hour_buy, 'sell', qty=int(float(pos.qty)))
                else:
                    print(f"  No actual position found for {last_hour_buy}, skipping sell")
            else:
                print("Step 1: Nothing to sell (First run).")

            # B. BUY NEW (The "Predicted" Winner)
            symbol_to_buy = TEST_TICKERS[ticker_index % len(TEST_TICKERS)]

            print(f"Step 2: Buying new position ({symbol_to_buy})...")
            success = place_order(api, symbol_to_buy, 'buy')

            if success:
                last_hour_buy = symbol_to_buy
                ticker_index += 1

            # C. SLEEP
            print("Cycle complete. Sleeping...")
            time.sleep(65)

        else:
            # Check again in 30 seconds
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
