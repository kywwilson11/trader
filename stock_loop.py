"""
Stock trading loop — market-hours aware, dual bear/bull models, dynamic top 10 selection.

- Trades only during regular market hours (9:30 AM - 4:00 PM ET)
- Scores all 30 stocks with bull model, trades only top 10 by signal strength
- Flattens all stock positions at 3:50 PM ET to avoid overnight gap risk
- $500 notional per position, max $5k total exposure
- Uses stock-prefixed models (stock_bear_model.pth, stock_bull_model.pth)
- Parallel predictions via ThreadPoolExecutor
- Hot-reload when model files change
"""
import alpaca_trade_api as tradeapi
import time
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from order_utils import (
    get_stock_quote, place_stock_limit_order, manage_order_lifecycle,
    get_all_positions, should_trade, cancel_all_open_orders,
)
from predict_now import load_dual_models, get_live_prediction, _fetch_spy_bars_alpaca
from hw_monitor import get_gpu_temp, is_gpu_available

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_API_SECRET')
BASE_URL = os.getenv('ALPACA_BASE_URL')

# ~30 high-beta, liquid stocks
STOCK_UNIVERSE = [
    'AMD', 'PLTR', 'SNAP', 'ROKU', 'AFRM', 'HOOD', 'SHOP', 'NET', 'CRWD', 'COIN',
    'MARA', 'MSTR', 'UBER', 'SOFI', 'ABNB', 'DASH', 'RBLX', 'SMCI', 'MRVL', 'ARM',
    'FSLR', 'ENPH', 'OXY', 'MRNA', 'CRSP', 'ARKK', 'TQQQ', 'SOXL', 'TSLA', 'NVDA',
    'META',
]

TOP_N = 10                   # Trade only top N stocks by bull signal
NOTIONAL_PER_STOCK = 500     # $500 per position
MAX_EXPOSURE = 5000          # Max total stock exposure
ORDER_TIMEOUT = 30           # Seconds to wait for limit fill
LOOP_INTERVAL = 30           # Seconds between checks
COOLDOWN_MINUTES = 30        # Min time between trades on same symbol
MAX_PREDICTION_WORKERS = 5
TEMP_LOG_EVERY_N_CYCLES = 10
THERMAL_THROTTLE_TEMP = 75
MODEL_PREFIX = 'stock'       # stock_bear_model.pth, stock_bull_model.pth

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
FLATTEN_HOUR = 15
FLATTEN_MINUTE = 50


def get_api():
    return tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')


def _get_eastern_now():
    """Get current time in US/Eastern."""
    import zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo('US/Eastern'))


def _is_market_hours():
    """Check if current time is within regular market hours."""
    now = _get_eastern_now()
    # Weekday check (0=Monday, 6=Sunday)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
    return market_open <= now < market_close


def _is_flatten_time():
    """Check if it's time to flatten all positions (3:50 PM ET)."""
    now = _get_eastern_now()
    flatten_time = now.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
    return flatten_time <= now < market_close


def _get_model_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _choose_inference_device():
    if not is_gpu_available():
        return 'cpu'
    return None


def _get_current_exposure(api):
    """Calculate total stock exposure from current positions."""
    positions = get_all_positions(api)
    total = 0.0
    for sym, pos in positions.items():
        # Only count stock positions (not crypto)
        if '/' not in sym and 'USD' not in sym:
            total += abs(float(pos.market_value))
    return total


def _predict_symbol(api, symbol, bear_model, bear_config, bull_model, bull_config,
                    scaler_X, feature_cols, inference_device, spy_close=None):
    """Run both bear and bull predictions for a single stock symbol."""
    bear_pred = get_live_prediction(
        symbol, bear_model, scaler_X, bear_config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type='stock', spy_close=spy_close,
    )
    bull_pred = get_live_prediction(
        symbol, bull_model, scaler_X, bull_config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type='stock', spy_close=spy_close,
    )
    return symbol, bear_pred, bull_pred


def flatten_all_stocks(api, positions):
    """Sell all stock positions for end-of-day flatten."""
    print("\n[FLATTEN] Selling all stock positions before market close...")
    for symbol in list(positions):
        try:
            pos = api.get_position(symbol)
            qty = int(float(pos.qty))
            if qty <= 0:
                positions.discard(symbol)
                continue

            quote = get_stock_quote(api, symbol)
            if quote is not None:
                order = place_stock_limit_order(api, symbol, 'sell', qty, quote,
                                                time_in_force='day', offset_bps=10)
                if order:
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    if result:
                        positions.discard(symbol)
                        print(f"  [FLATTEN] {symbol}: Sold {qty} shares")
            else:
                # No quote — market sell
                api.submit_order(symbol=symbol, qty=qty, side='sell',
                                type='market', time_in_force='day')
                positions.discard(symbol)
                print(f"  [FLATTEN] {symbol}: Market sold {qty} shares")

        except Exception as e:
            print(f"  [FLATTEN] {symbol}: Error: {e}")
        time.sleep(0.5)

    return positions


def cooldown_ok(last_trade_time, symbol):
    if symbol not in last_trade_time:
        return True
    elapsed = (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
    return elapsed >= COOLDOWN_MINUTES * 60


def run_stock_bot():
    api = get_api()

    # Load stock-specific dual models
    print("Loading stock prediction models...")
    try:
        inference_device = _choose_inference_device()
        bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = \
            load_dual_models(inference_device, prefix=MODEL_PREFIX)
        bear_threshold = bear_config.get('bull_threshold', 0.15)
        bull_threshold = bull_config.get('bull_threshold', 0.15)
        is_dual = bear_model is not bull_model
        print(f"Stock models loaded (dual={is_dual}, bear_th={bear_threshold:.2f}, bull_th={bull_threshold:.2f})")
    except FileNotFoundError:
        print("WARNING: Stock model files not found. Run hypersearch_dual.py --prefix stock first.")
        print("Exiting.")
        return

    # Track model mtimes for hot-reload
    bear_mtime = _get_model_mtime(f'{MODEL_PREFIX}_bear_model.pth')
    bull_mtime = _get_model_mtime(f'{MODEL_PREFIX}_bull_model.pth')

    # Cancel stale orders
    cancel_all_open_orders(api)

    # Reconstruct stock positions from API
    positions = set()
    existing = get_all_positions(api)
    for sym in STOCK_UNIVERSE:
        if sym in existing:
            positions.add(sym)
    if positions:
        print(f"Existing stock positions: {', '.join(positions)}")

    last_trade_time = {}

    print("\n--- STOCK TRADING BOT STARTED ---")
    print(f"Universe: {len(STOCK_UNIVERSE)} stocks, trading top {TOP_N}")
    print(f"Notional: ${NOTIONAL_PER_STOCK}/position, max ${MAX_EXPOSURE} exposure")
    print(f"Loop interval: {LOOP_INTERVAL}s | Cooldown: {COOLDOWN_MINUTES} min")
    print(f"Flatten at: {FLATTEN_HOUR}:{FLATTEN_MINUTE:02d} ET")

    cycle = 0
    flattened_today = False
    last_date = None

    while True:
        cycle += 1
        now = datetime.datetime.now()
        eastern_now = _get_eastern_now()

        # Reset flatten flag on new day
        if last_date != eastern_now.date():
            flattened_today = False
            last_date = eastern_now.date()

        # Check market hours
        if not _is_market_hours():
            if cycle == 1 or cycle % 20 == 0:
                print(f"\n[WAIT] {eastern_now.strftime('%Y-%m-%d %H:%M ET')} — Market closed. "
                      f"Next check in {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)
            continue

        print(f"\n--- CYCLE {cycle}: {eastern_now.strftime('%Y-%m-%d %H:%M:%S ET')} ---")

        # --- Flatten check ---
        if _is_flatten_time() and not flattened_today:
            positions = flatten_all_stocks(api, positions)
            flattened_today = True
            print("[FLATTEN] Done. No more trades today.")
            time.sleep(LOOP_INTERVAL)
            continue

        if flattened_today:
            time.sleep(LOOP_INTERVAL)
            continue

        # --- Hot-reload check ---
        new_bear_mt = _get_model_mtime(f'{MODEL_PREFIX}_bear_model.pth')
        new_bull_mt = _get_model_mtime(f'{MODEL_PREFIX}_bull_model.pth')
        if new_bear_mt != bear_mtime or new_bull_mt != bull_mtime:
            print("[HOT-RELOAD] Stock model files changed, reloading...")
            try:
                inference_device = _choose_inference_device()
                bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = \
                    load_dual_models(inference_device, prefix=MODEL_PREFIX)
                bear_threshold = bear_config.get('bull_threshold', 0.15)
                bull_threshold = bull_config.get('bull_threshold', 0.15)
                bear_mtime = new_bear_mt
                bull_mtime = new_bull_mt
                print(f"[HOT-RELOAD] Success (bear_th={bear_threshold:.2f}, bull_th={bull_threshold:.2f})")
            except Exception as e:
                print(f"[HOT-RELOAD] Failed: {e}, keeping current models")

        # --- Log GPU temp periodically ---
        if cycle % TEMP_LOG_EVERY_N_CYCLES == 0:
            temp = get_gpu_temp()
            if temp is not None:
                print(f"[HW] GPU temp: {temp:.0f}C")

        # --- Fetch SPY bars for relative strength ---
        spy_close = None
        try:
            spy_df = _fetch_spy_bars_alpaca(api)
            if spy_df is not None:
                spy_close = spy_df['Close']
        except Exception as e:
            print(f"  [SPY] Error fetching benchmark: {e}")

        # --- Get predictions for ALL stocks in parallel ---
        bear_preds = {}
        bull_preds = {}
        inference_device = _choose_inference_device()
        if inference_device == 'cpu':
            print("[HW] GPU unavailable, using CPU for inference")

        with ThreadPoolExecutor(max_workers=MAX_PREDICTION_WORKERS) as executor:
            futures = {}
            for symbol in STOCK_UNIVERSE:
                f = executor.submit(
                    _predict_symbol, api, symbol,
                    bear_model, bear_config, bull_model, bull_config,
                    scaler_X, feature_cols, inference_device, spy_close,
                )
                futures[f] = symbol

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, bear_pred, bull_pred = future.result()
                    if bear_pred is not None:
                        bear_preds[sym] = bear_pred
                    if bull_pred is not None:
                        bull_preds[sym] = bull_pred
                except Exception as e:
                    print(f"  {symbol}: Prediction error: {e}")

        # --- Dynamic top N selection by bull signal strength ---
        ranked = sorted(bull_preds.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in ranked[:TOP_N]]
        print(f"[RANK] Top {TOP_N}: {', '.join(f'{s}({bull_preds[s]:+.4f})' for s in top_symbols)}")

        # --- SELL: bearish positions ---
        for symbol in list(positions):
            try:
                pos = api.get_position(symbol)
            except Exception:
                positions.discard(symbol)
                continue

            bear_pred = bear_preds.get(symbol)
            # Sell if: bear signal is strong, OR symbol dropped out of top N
            sell_reason = None
            if bear_pred is not None and bear_pred < -bear_threshold:
                sell_reason = f"bear_pred={bear_pred:+.4f}%"
            elif symbol not in top_symbols and bear_pred is not None and bear_pred < 0:
                sell_reason = f"dropped from top {TOP_N} (bear={bear_pred:+.4f}%)"

            if sell_reason is None:
                continue

            if not cooldown_ok(last_trade_time, symbol):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: Sell signal but in cooldown ({remaining/60:.1f} min left)")
                continue

            print(f"  {symbol}: SELLING ({sell_reason})")
            qty = int(float(pos.qty))
            if qty <= 0:
                positions.discard(symbol)
                continue

            quote = get_stock_quote(api, symbol)
            if quote is not None:
                order = place_stock_limit_order(api, symbol, 'sell', qty, quote,
                                                time_in_force='day')
                if order:
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    if result:
                        positions.discard(symbol)
                        last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(0.5)

        # --- BUY: top N bullish stocks we don't hold ---
        current_exposure = _get_current_exposure(api)
        for symbol in top_symbols:
            if symbol in positions:
                continue

            if current_exposure >= MAX_EXPOSURE:
                print(f"  Max exposure ${MAX_EXPOSURE} reached, no more buys")
                break

            if not cooldown_ok(last_trade_time, symbol):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: In cooldown ({remaining/60:.1f} min left)")
                continue

            bull_pred = bull_preds.get(symbol)
            if bull_pred is None or bull_pred < bull_threshold:
                if bull_pred is not None:
                    print(f"  {symbol}: Bull pred {bull_pred:+.4f}% < {bull_threshold:.2f}, skipping")
                continue

            quote = get_stock_quote(api, symbol)
            if quote is None:
                continue

            if not should_trade(bull_pred, quote['spread_pct']):
                print(f"  {symbol}: Bull pred {bull_pred:+.4f}% too weak vs spread "
                      f"{quote['spread_pct']:.3f}%, skipping")
                continue

            # Calculate qty (whole shares)
            price = quote['midpoint']
            if price <= 0:
                continue
            qty = int(NOTIONAL_PER_STOCK / price)
            if qty <= 0:
                print(f"  {symbol}: Price ${price:.2f} too high for ${NOTIONAL_PER_STOCK} notional")
                continue

            print(f"  {symbol}: BUYING {qty} shares @ ~${price:.2f} (bull={bull_pred:+.4f}%)")
            order = place_stock_limit_order(api, symbol, 'buy', qty, quote,
                                            time_in_force='day')
            if order:
                result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                               fallback_to_market=True)
                if result and result.status == 'filled':
                    positions.add(symbol)
                    last_trade_time[symbol] = datetime.datetime.now()
                    current_exposure += qty * price
            time.sleep(0.5)

        # Thermal throttling
        sleep_interval = LOOP_INTERVAL
        temp = get_gpu_temp()
        if temp is not None and temp > THERMAL_THROTTLE_TEMP:
            sleep_interval = LOOP_INTERVAL * 2
            print(f"[HW] GPU temp {temp:.0f}C > {THERMAL_THROTTLE_TEMP}C, throttling to {sleep_interval}s")

        print(f"[STATUS] Positions: {len(positions)} | Exposure: ~${current_exposure:.0f}")
        print(f"[SLEEP] Next check in {sleep_interval}s...")
        time.sleep(sleep_interval)


if __name__ == "__main__":
    run_stock_bot()
