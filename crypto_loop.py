# NOTE: yfinance must be imported BEFORE torch to avoid CUDA's bundled
# SQLite library overriding the system one (breaks yfinance's cache).
import yfinance as yf

import alpaca_trade_api as tradeapi
import time
import datetime
import os
from dotenv import load_dotenv

from order_utils import (
    get_crypto_quote, place_limit_order, manage_order_lifecycle,
    verify_position, get_all_positions, should_trade,
    cancel_all_open_orders,
)
from predict_now import load_model, get_live_prediction

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
    'BCH/USD',
]

# yfinance uses '-' not '/'
YFINANCE_MAP = {sym: sym.replace('/', '-') for sym in CRYPTO_SYMBOLS}

NOTIONAL_PER_SYMBOL = 25  # $25 per symbol per cycle
ORDER_TIMEOUT = 30  # seconds to wait for limit fill
LOOP_INTERVAL = 90  # seconds between checks
COOLDOWN_MINUTES = 30  # min time between trades on same symbol

def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


def place_smart_order(api, symbol, side, notional):
    """Place a limit order with lifecycle management and market fallback."""
    quote = get_crypto_quote(api, symbol)
    if quote is None:
        print(f"  {symbol}: No quote available, skipping {side}")
        return False

    order = place_limit_order(api, symbol, side, notional, quote)
    if order is None:
        return False

    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                    fallback_to_market=True)
    if result and result.status == 'filled':
        return True

    # Market fallback may have been submitted — check if it's a new order
    if result and result.id != order.id:
        # Wait briefly for market order to fill
        time.sleep(2)
        try:
            final = api.get_order(result.id)
            return final.status == 'filled'
        except Exception:
            pass

    return False


def cooldown_ok(last_trade_time, symbol):
    """Return True if the symbol is not in cooldown."""
    if symbol not in last_trade_time:
        return True
    elapsed = (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
    return elapsed >= COOLDOWN_MINUTES * 60


def run_crypto_bot():
    api = get_api()

    # Load prediction model
    print("Loading prediction model...")
    try:
        model, scaler_X, scaler_y, input_dim = load_model()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("WARNING: Model files not found. Running without prediction gating.")
        model = None

    # Cancel any stale orders from previous runs
    cancel_all_open_orders(api)

    # Reconstruct positions from API (survive restarts)
    positions = set()
    existing = get_all_positions(api)
    for sym in CRYPTO_SYMBOLS:
        if sym in existing or sym.replace('/', '') in existing:
            positions.add(sym)
    if positions:
        print(f"Existing positions found: {', '.join(positions)}")

    # Per-symbol cooldown tracking: symbol -> datetime of last trade
    last_trade_time = {}

    print("\n--- JETSON CRYPTO BOT STARTED (CONTINUOUS MODE) ---")
    print(f"Symbols: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"Notional: ${NOTIONAL_PER_SYMBOL} per symbol per trade")
    print(f"Loop interval: {LOOP_INTERVAL}s | Cooldown: {COOLDOWN_MINUTES} min")

    cycle = 0
    while True:
        cycle += 1
        print(f"\n--- CYCLE {cycle}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        # 1. Get predictions for all symbols
        predictions = {}
        if model is not None:
            for symbol in CRYPTO_SYMBOLS:
                yf_sym = YFINANCE_MAP[symbol]
                pred = get_live_prediction(yf_sym, model, scaler_X, scaler_y, input_dim)
                if pred is not None:
                    predictions[symbol] = pred
                time.sleep(0.5)

        # 2. SELL: bearish positions with cooldown expired
        for symbol in list(positions):
            pos = verify_position(api, symbol)
            if pos is None:
                print(f"  {symbol}: No actual position found, removing from tracking")
                positions.discard(symbol)
                continue

            pred = predictions.get(symbol)
            if pred is not None and pred > -0.1:
                print(f"  {symbol}: Prediction {pred:+.4f}%, HOLDING")
                continue

            # Bearish — check cooldown
            if not cooldown_ok(last_trade_time, symbol):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: Bearish but in cooldown ({remaining/60:.1f} min left), skipping sell")
                continue

            reason = f"pred={pred:+.4f}%" if pred is not None else "no prediction"
            print(f"  {symbol}: SELLING ({reason})")

            quote = get_crypto_quote(api, symbol)
            if quote is not None:
                qty = float(pos.qty)
                try:
                    order = api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='limit',
                        limit_price=round(quote['midpoint'] - quote['midpoint'] * 0.0005, 4),
                        time_in_force='gtc',
                    )
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    if result:
                        positions.discard(symbol)
                        last_trade_time[symbol] = datetime.datetime.now()
                except Exception as e:
                    print(f"  {symbol}: Sell error: {e}")
            else:
                try:
                    api.submit_order(symbol=symbol, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='gtc')
                    positions.discard(symbol)
                    last_trade_time[symbol] = datetime.datetime.now()
                except Exception as e:
                    print(f"  {symbol}: Market sell error: {e}")
            time.sleep(1)

        # 3. BUY: bullish symbols we don't hold, with cooldown expired
        for symbol in CRYPTO_SYMBOLS:
            if symbol in positions:
                continue

            if not cooldown_ok(last_trade_time, symbol):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: In cooldown ({remaining/60:.1f} min left), skipping buy")
                continue

            pred = predictions.get(symbol)
            quote = get_crypto_quote(api, symbol)

            if pred is not None and quote is not None:
                if not should_trade(pred, quote['spread_pct']):
                    print(f"  {symbol}: Prediction {pred:+.4f}% too weak vs spread "
                          f"{quote['spread_pct']:.3f}%, skipping")
                    continue
                if pred < 0.1:
                    print(f"  {symbol}: Prediction {pred:+.4f}% not bullish enough, skipping")
                    continue

            if place_smart_order(api, symbol, 'buy', NOTIONAL_PER_SYMBOL):
                positions.add(symbol)
                last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

        print(f"[SLEEP] Next check in {LOOP_INTERVAL}s...")
        time.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    run_crypto_bot()
