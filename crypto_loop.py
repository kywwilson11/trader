"""24/7 crypto trading loop — 10 symbols, dual bear/bull models, sentiment-gated.

Runs continuously (crypto markets never close):
  1. Fetch predictions for all symbols in parallel (ThreadPoolExecutor)
  2. Check stop-loss / trailing stop / take-profit on open positions
  3. Sell positions where the bear model signals weakness
  4. Buy symbols where the bull model signals strength (sentiment-gated)
  5. Sleep and repeat

Uses ATR-adaptive stops with fixed-percentage fallbacks.
"""

import time
import datetime
import gc
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from order_utils import (
    get_crypto_quote, place_limit_order, manage_order_lifecycle,
    verify_position, get_all_positions, should_trade,
    cancel_all_open_orders, reconstruct_positions,
    check_circuit_breaker, emergency_flatten,
)
from predict_now import load_dual_models, get_live_prediction
from market_data import fetch_bars_alpaca, get_live_atr
from trading_utils import get_api, get_model_mtime, choose_inference_device, cooldown_ok, predict_symbol
from hw_monitor import get_gpu_temp
from sentiment import sentiment_gate, get_fear_greed

# --- CONFIGURATION ---

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

NOTIONAL_PER_SYMBOL = 250  # $250 per symbol per cycle
ORDER_TIMEOUT = 30  # seconds to wait for limit fill
LOOP_INTERVAL = 30  # seconds between checks
COOLDOWN_MINUTES = 30  # min time between trades on same symbol
MAX_PREDICTION_WORKERS = 5
TEMP_LOG_EVERY_N_CYCLES = 10
THERMAL_THROTTLE_TEMP = 75  # increase sleep if GPU above this

# ATR-based stop-loss / trailing stop / take-profit settings
ATR_STOP_MULTIPLIER = 2.0          # stop = entry - (ATR * 2.0)
ATR_TRAIL_MULTIPLIER = 1.5         # trail = hwm - (ATR * 1.5)
ATR_TRAIL_ACTIVATE_PCT = 0.01      # activate trailing at 1% profit
ATR_STOP_FLOOR_PCT = 0.015         # min stop distance 1.5%
ATR_STOP_CEIL_PCT = 0.08           # max stop distance 8%
CRYPTO_TAKE_PROFIT_RR = 2.0        # 2:1 risk-reward take-profit
CRYPTO_TAKE_PROFIT_CEIL_PCT = 0.12  # max take-profit distance 12%

# Fallback fixed percentages (used when ATR unavailable)
CRYPTO_STOP_LOSS_PCT = 0.04        # 4% hard stop-loss from entry
CRYPTO_TRAIL_PCT = 0.03            # 3% trailing stop from high water mark
CIRCUIT_BREAKER_PCT = 0.05         # 5% daily equity drawdown triggers flatten

_PRED_CACHE_FILE = Path(__file__).resolve().parent / "crypto_predictions.json"


def _write_prediction_cache(bear_preds, bull_preds, bear_threshold, bull_threshold):
    """Write prediction scores to JSON for GUI consumption."""
    try:
        data = {}
        all_syms = set(bear_preds) | set(bull_preds)
        for sym in sorted(all_syms):
            bear = bear_preds.get(sym)
            bull = bull_preds.get(sym)
            score = (bull or 0) - abs(bear or 0)
            if bull is not None and bull >= bull_threshold:
                signal = "BULL"
            elif bear is not None and bear < -bear_threshold:
                signal = "BEAR"
            else:
                signal = "NEUTRAL"
            data[sym] = {
                "bear": round(bear, 6) if bear is not None else None,
                "bull": round(bull, 6) if bull is not None else None,
                "score": round(score, 6),
                "signal": signal,
                "updated": datetime.datetime.now().isoformat(),
            }
        with open(_PRED_CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  [CACHE] Error writing crypto prediction cache: {e}")


# --- ORDER HELPERS ---

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
        time.sleep(2)
        try:
            final = api.get_order(result.id)
            return final.status == 'filled'
        except Exception:
            pass

    return False


# --- MAIN LOOP ---

def run_crypto_bot():
    api = get_api()

    # ── Load prediction models ──
    print("Loading prediction models...")
    try:
        inference_device = choose_inference_device()
        bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = \
            load_dual_models(inference_device)
        bear_threshold = bear_config.get('bull_threshold', 0.15)
        bull_threshold = bull_config.get('bull_threshold', 0.15)
        is_dual = bear_model is not bull_model
        print(f"Models loaded (dual={is_dual}, bear_th={bear_threshold:.2f}, bull_th={bull_threshold:.2f})")
    except FileNotFoundError:
        print("WARNING: Model files not found. Running without prediction gating.")
        bear_model = bull_model = None
        bear_config = bull_config = {}
        bear_threshold = bull_threshold = 0.15
        scaler_X = feature_cols = None
        is_dual = False

    # Track model file mtimes for hot-reload
    bear_mtime = get_model_mtime('bear_model.pth')
    bull_mtime = get_model_mtime('bull_model.pth')
    default_mtime = get_model_mtime('stock_predictor.pth')

    # ── Cancel stale orders from previous runs ──
    cancel_all_open_orders(api)

    # ── Reconstruct positions from API (survive restarts) ──
    positions = reconstruct_positions(api, CRYPTO_SYMBOLS, asset_type='crypto')
    if positions:
        print(f"Existing positions found: {', '.join(positions)}")
        for sym, info in positions.items():
            entry_atr = get_live_atr(api, sym, asset_type='crypto')
            info['entry_atr'] = entry_atr
            tp_price = None
            if entry_atr is not None and info['entry_price'] > 0:
                raw_stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / info['entry_price']
                stop_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_stop_dist))
                tp_dist = min(CRYPTO_TAKE_PROFIT_CEIL_PCT, stop_dist * CRYPTO_TAKE_PROFIT_RR)
                tp_price = info['entry_price'] * (1 + tp_dist)
            info['take_profit_price'] = tp_price
            atr_str = f"ATR=${entry_atr:.4f}" if entry_atr else "ATR=N/A"
            tp_str = f"tp=${tp_price:.4f}" if tp_price else "tp=N/A"
            print(f"  {sym}: qty={info['qty']}, entry=${info['entry_price']:.4f}, hwm=${info['high_water_mark']:.4f}, {atr_str}, {tp_str}")

    last_trade_time = {}

    print("\n--- JETSON CRYPTO BOT STARTED (CONTINUOUS MODE) ---")
    print(f"Symbols: {', '.join(CRYPTO_SYMBOLS)}")
    print(f"Notional: ${NOTIONAL_PER_SYMBOL} per symbol per trade")
    print(f"Loop interval: {LOOP_INTERVAL}s | Cooldown: {COOLDOWN_MINUTES} min")
    print(f"Parallel workers: {MAX_PREDICTION_WORKERS}")
    print(f"Sentiment gating: ENABLED")

    cycle = 0
    while True:
        cycle += 1
        print(f"\n--- CYCLE {cycle}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        # ── Sentiment check (once per cycle, cached 5 min) ──
        fng = get_fear_greed()
        if fng is not None and cycle % TEMP_LOG_EVERY_N_CYCLES == 1:
            print(f"[SENTIMENT] Fear & Greed: {fng['value']} ({fng['label']})")

        # ── Hot-reload check ──
        new_bear_mt = get_model_mtime('bear_model.pth')
        new_bull_mt = get_model_mtime('bull_model.pth')
        new_default_mt = get_model_mtime('stock_predictor.pth')

        if (new_bear_mt != bear_mtime or new_bull_mt != bull_mtime
                or new_default_mt != default_mtime):
            print("[HOT-RELOAD] Model files changed, reloading...")
            try:
                inference_device = choose_inference_device()
                bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = \
                    load_dual_models(inference_device)
                bear_threshold = bear_config.get('bull_threshold', 0.15)
                bull_threshold = bull_config.get('bull_threshold', 0.15)
                is_dual = bear_model is not bull_model
                bear_mtime = new_bear_mt
                bull_mtime = new_bull_mt
                default_mtime = new_default_mt
                print(f"[HOT-RELOAD] Success (dual={is_dual}, bear_th={bear_threshold:.2f}, bull_th={bull_threshold:.2f})")
            except Exception as e:
                print(f"[HOT-RELOAD] Failed: {e}, keeping current models")

        # ── Circuit breaker check ──
        tripped, dd = check_circuit_breaker(api, max_drawdown_pct=CIRCUIT_BREAKER_PCT)
        if tripped:
            print(f"[CIRCUIT BREAKER] Daily drawdown {dd:.1%} >= {CIRCUIT_BREAKER_PCT:.0%}, flattening all positions!")
            emergency_flatten(api)
            positions.clear()
            print("[CIRCUIT BREAKER] Sleeping 1 hour before resuming...")
            time.sleep(3600)
            continue

        # ── Log GPU temp periodically ──
        if cycle % TEMP_LOG_EVERY_N_CYCLES == 0:
            temp = get_gpu_temp()
            if temp is not None:
                print(f"[HW] GPU temp: {temp:.0f}C")

        # ── Software stop-loss / trailing stop / take-profit checks ──
        for symbol in list(positions):
            quote = get_crypto_quote(api, symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            info = positions[symbol]
            entry_price = info['entry_price']
            info['high_water_mark'] = max(info['high_water_mark'], current_price)
            hwm = info['high_water_mark']

            # Determine stop distances based on ATR or fallback
            entry_atr = info.get('entry_atr')
            if entry_atr is not None and entry_price > 0:
                raw_stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / entry_price
                stop_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_stop_dist))
                raw_trail_dist = (entry_atr * ATR_TRAIL_MULTIPLIER) / hwm if hwm > 0 else stop_dist
                trail_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_trail_dist))
            else:
                stop_dist = CRYPTO_STOP_LOSS_PCT
                trail_dist = CRYPTO_TRAIL_PCT

            stop_reason = None
            if current_price <= entry_price * (1 - stop_dist):
                stop_reason = 'hard_stop'
            elif info.get('take_profit_price') and current_price >= info['take_profit_price']:
                stop_reason = 'take_profit'
            elif (hwm >= entry_price * (1 + ATR_TRAIL_ACTIVATE_PCT)
                  and current_price <= hwm * (1 - trail_dist)):
                stop_reason = 'trailing'

            if stop_reason:
                tp_str = f", tp=${info.get('take_profit_price', 0):.4f}" if info.get('take_profit_price') else ""
                print(f"  [STOP] {symbol}: STOPPED OUT at ${current_price:.4f} "
                      f"(entry=${entry_price:.4f}, hwm=${hwm:.4f}, stop_d={stop_dist:.1%}, trail_d={trail_dist:.1%}{tp_str}, reason={stop_reason})")
                try:
                    api.submit_order(
                        symbol=symbol, qty=info['qty'],
                        side='sell', type='market', time_in_force='gtc',
                    )
                    del positions[symbol]
                    last_trade_time[symbol] = datetime.datetime.now()
                except Exception as e:
                    print(f"  [STOP] {symbol}: Sell error: {e}")

        # ── Fetch BTC bars once per cycle for cross-asset features ──
        btc_close = None
        try:
            btc_df = fetch_bars_alpaca(api, 'BTC/USD')
            if btc_df is not None:
                btc_close = btc_df['Close']
        except Exception as e:
            print(f"  [BTC] Error fetching benchmark: {e}")

        # ── Get predictions for all symbols in parallel ──
        bear_preds = {}
        bull_preds = {}
        if bear_model is not None:
            inference_device = choose_inference_device()
            if inference_device == 'cpu':
                print("[HW] GPU unavailable, using CPU for inference")

            with ThreadPoolExecutor(max_workers=MAX_PREDICTION_WORKERS) as executor:
                futures = {}
                for symbol in CRYPTO_SYMBOLS:
                    f = executor.submit(
                        predict_symbol, api, symbol,
                        bear_model, bear_config, bull_model, bull_config,
                        scaler_X, feature_cols, inference_device,
                        asset_type='crypto', benchmark_close=btc_close,
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

            # Free temporary tensors from prediction batch
            gc.collect()

            # Write prediction cache for GUI
            _write_prediction_cache(bear_preds, bull_preds, bear_threshold, bull_threshold)

        # ── SELL: bearish positions with cooldown expired ──
        for symbol in list(positions):
            pos = verify_position(api, symbol)
            if pos is None:
                print(f"  {symbol}: No actual position found, removing from tracking")
                del positions[symbol]
                continue

            bear_pred = bear_preds.get(symbol)
            if bear_pred is not None and bear_pred > -bear_threshold:
                print(f"  {symbol}: Bear pred {bear_pred:+.4f}% > -{bear_threshold:.2f}, HOLDING")
                continue

            if not cooldown_ok(last_trade_time, symbol, COOLDOWN_MINUTES):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: Bearish but in cooldown ({remaining/60:.1f} min left), skipping sell")
                continue

            reason = f"bear_pred={bear_pred:+.4f}%" if bear_pred is not None else "no prediction"
            print(f"  {symbol}: SELLING ({reason})")

            quote = get_crypto_quote(api, symbol)
            info = positions[symbol]
            if quote is not None:
                qty = info['qty']
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
                        del positions[symbol]
                        last_trade_time[symbol] = datetime.datetime.now()
                except Exception as e:
                    print(f"  {symbol}: Sell error: {e}")
            else:
                try:
                    api.submit_order(symbol=symbol, qty=info['qty'],
                                     side='sell', type='market', time_in_force='gtc')
                    del positions[symbol]
                    last_trade_time[symbol] = datetime.datetime.now()
                except Exception as e:
                    print(f"  {symbol}: Market sell error: {e}")
            time.sleep(1)

        # ── BUY: bullish symbols we don't hold, with cooldown expired ──
        for symbol in CRYPTO_SYMBOLS:
            if symbol in positions:
                continue

            if not cooldown_ok(last_trade_time, symbol, COOLDOWN_MINUTES):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: In cooldown ({remaining/60:.1f} min left), skipping buy")
                continue

            bull_pred = bull_preds.get(symbol)
            quote = get_crypto_quote(api, symbol)

            if bull_pred is not None and quote is not None:
                if not should_trade(bull_pred, quote['spread_pct']):
                    print(f"  {symbol}: Bull pred {bull_pred:+.4f}% too weak vs spread "
                          f"{quote['spread_pct']:.3f}%, skipping")
                    continue
                if bull_pred < bull_threshold:
                    print(f"  {symbol}: Bull pred {bull_pred:+.4f}% < {bull_threshold:.2f}, skipping")
                    continue

            # Confidence-based sizing
            if bull_pred is not None and bull_threshold > 0:
                confidence = min(2.0, max(0.5, bull_pred / bull_threshold))
            else:
                confidence = 1.0
            sized_notional = int(NOTIONAL_PER_SYMBOL * confidence)

            # Sentiment gate
            gate, gate_reasons = sentiment_gate(symbol, 'crypto')
            if gate <= 0:
                print(f"  {symbol}: BLOCKED by sentiment ({', '.join(gate_reasons)})")
                continue
            adjusted_notional = int(sized_notional * gate)
            sizing_info = f"conf={confidence:.2f}x"
            if gate != 1.0:
                sizing_info += f", sent={gate:.2f}x"
            if gate_reasons:
                sizing_info += f" ({', '.join(gate_reasons)})"
            print(f"  {symbol}: Sizing ${adjusted_notional} [{sizing_info}]")

            if place_smart_order(api, symbol, 'buy', adjusted_notional):
                fill_price = None
                quote = get_crypto_quote(api, symbol)
                if quote:
                    fill_price = quote['midpoint']
                pos = verify_position(api, symbol)
                if pos:
                    fill_price = float(pos.avg_entry_price)

                    entry_atr = get_live_atr(api, symbol, asset_type='crypto')
                    tp_price = None
                    if entry_atr is not None and fill_price > 0:
                        raw_stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / fill_price
                        stop_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_stop_dist))
                        raw_tp_dist = stop_dist * CRYPTO_TAKE_PROFIT_RR
                        tp_dist = min(CRYPTO_TAKE_PROFIT_CEIL_PCT, raw_tp_dist)
                        tp_price = fill_price * (1 + tp_dist)
                        raw_trail_dist = (entry_atr * ATR_TRAIL_MULTIPLIER) / fill_price
                        trail_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_trail_dist))
                        print(f"  [ATR-STOP] {symbol}: ATR=${entry_atr:.4f}, "
                              f"stop={stop_dist:.1%}, trail={trail_dist:.1%}, tp={tp_dist:.1%}")
                    else:
                        print(f"  [ATR-STOP] {symbol}: ATR unavailable, using fixed stops")

                    positions[symbol] = {
                        'qty': float(pos.qty),
                        'entry_price': fill_price,
                        'high_water_mark': fill_price,
                        'stop_order_id': None,
                        'trailing_activated': False,
                        'entry_atr': entry_atr,
                        'take_profit_price': tp_price,
                    }
                last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

        # ── Thermal throttling ──
        sleep_interval = LOOP_INTERVAL
        temp = get_gpu_temp()
        if temp is not None and temp > THERMAL_THROTTLE_TEMP:
            sleep_interval = LOOP_INTERVAL * 2
            print(f"[HW] GPU temp {temp:.0f}C > {THERMAL_THROTTLE_TEMP}C, throttling to {sleep_interval}s")

        print(f"[SLEEP] Next check in {sleep_interval}s...")
        time.sleep(sleep_interval)

if __name__ == "__main__":
    run_crypto_bot()
