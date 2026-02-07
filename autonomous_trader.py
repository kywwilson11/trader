# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf

import time
import datetime
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

from predict_now import (
    load_model, get_live_prediction, compute_rsi, compute_macd, compute_atr,
)
from order_utils import (
    get_crypto_quote, place_limit_order, manage_order_lifecycle,
    verify_position, should_trade, cancel_all_open_orders,
)

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# The basket of assets the AI will scan (yfinance format)
TICKERS_YF = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'AVAX-USD']
# Alpaca format mapping
YF_TO_ALPACA = {t: t.replace('-', '/') for t in TICKERS_YF}

TRADE_AMOUNT = 100  # Dollars per trade
ORDER_TIMEOUT = 30  # seconds to wait for limit fill

# --- TRADING ENGINE ---
def run_autonomous_trader():
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

    print("Loading AI Model...")
    try:
        model, scaler_x, scaler_y, input_dim = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model or Scaler files not found. Did you run train_model.py?")
        exit(1)

    # Cancel stale orders from previous runs
    cancel_all_open_orders(api)

    last_buy_symbol = None  # Alpaca format (e.g. BTC/USD)

    # Check if we already hold a position from a previous run
    try:
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol in YF_TO_ALPACA.values():
                last_buy_symbol = pos.symbol
                print(f"Existing position found: {last_buy_symbol}")
                break
    except Exception:
        pass

    print(f"\n--- JETSON AI TRADER ONLINE (SMART MODE) ---")
    print(f"Monitoring: {TICKERS_YF}")

    while True:
        now = datetime.datetime.now()

        # Wait for top of the hour (minute 0)
        if now.minute == 0:
            print(f"\n[CYCLE START] {now.strftime('%H:%M')}")

            # 1. SCAN & RANK all tickers
            print(" -> AI Scanning Market...")
            scores = []
            for ticker_yf in TICKERS_YF:
                score = get_live_prediction(ticker_yf, model, scaler_x, scaler_y, input_dim)
                if score is None:
                    score = -999
                scores.append((ticker_yf, score))
                print(f"    {ticker_yf}: {score:.4f}%")

            # Sort by Score (Highest First)
            scores.sort(key=lambda x: x[1], reverse=True)
            best_pick_yf, best_score = scores[0]
            best_pick_alpaca = YF_TO_ALPACA[best_pick_yf]

            print(f" -> WINNER: {best_pick_yf} (Pred: {best_score:.4f}%)")

            # 2. Check if best pick == current holding (hold-through)
            if best_pick_alpaca == last_buy_symbol:
                print(f" -> Already holding {best_pick_alpaca}, no action needed (hold-through)")
                time.sleep(120)
                continue

            # 3. Check if prediction clears spread threshold
            quote = get_crypto_quote(api, best_pick_alpaca)
            if quote is not None and not should_trade(best_score, quote['spread_pct']):
                print(f" -> Prediction {best_score:+.4f}% too weak vs spread "
                      f"{quote['spread_pct']:.3f}%, holding current position")
                time.sleep(120)
                continue

            # 4. SELL PREVIOUS (if exists)
            if last_buy_symbol:
                print(f" -> Selling previous: {last_buy_symbol}")
                pos = verify_position(api, last_buy_symbol)
                if pos is not None:
                    sell_quote = get_crypto_quote(api, last_buy_symbol)
                    if sell_quote is not None:
                        order = place_limit_order(api, last_buy_symbol, 'sell',
                                                  TRADE_AMOUNT, sell_quote)
                        if order:
                            manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    else:
                        try:
                            api.submit_order(symbol=last_buy_symbol, notional=TRADE_AMOUNT,
                                             side='sell', type='market', time_in_force='gtc')
                        except Exception as e:
                            print(f"    Sell Error: {e}")
                else:
                    print(f" -> No position found for {last_buy_symbol}, skipping sell")
                last_buy_symbol = None
                time.sleep(1)

            # 5. BUY BEST
            if best_score <= 0.1:
                print(f" -> Best prediction {best_score:.4f}% not bullish enough, sitting out")
                time.sleep(120)
                continue

            print(f" -> Buying {best_pick_alpaca}...")
            if quote is None:
                quote = get_crypto_quote(api, best_pick_alpaca)

            if quote is not None:
                order = place_limit_order(api, best_pick_alpaca, 'buy',
                                          TRADE_AMOUNT, quote)
                if order:
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                    fallback_to_market=True)
                    if result and result.status == 'filled':
                        last_buy_symbol = best_pick_alpaca
                        print(f" -> BOUGHT {best_pick_alpaca}")
                    elif result and result.id != order.id:
                        # Market fallback was submitted
                        time.sleep(2)
                        last_buy_symbol = best_pick_alpaca
                        print(f" -> BOUGHT {best_pick_alpaca} (market fallback)")
                    else:
                        print(f" -> Buy failed for {best_pick_alpaca}")
            else:
                # No quote — fall back to market
                try:
                    api.submit_order(symbol=best_pick_alpaca, notional=TRADE_AMOUNT,
                                     side='buy', type='market', time_in_force='gtc')
                    last_buy_symbol = best_pick_alpaca
                    print(f" -> BOUGHT {best_pick_alpaca} (market, no quote available)")
                except Exception as e:
                    print(f"    Buy Error: {e}")

            # Sleep to prevent double-execution
            print("[SLEEPING] Waiting for next hour...")
            time.sleep(120)

        else:
            # Heartbeat every 30 seconds
            time.sleep(30)

if __name__ == "__main__":
    run_autonomous_trader()
