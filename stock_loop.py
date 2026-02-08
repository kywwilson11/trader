"""Stock trading loop — market-hours aware, dual bear/bull models, dynamic top 10 selection.

Trades only during regular market hours (9:30 AM - 4:00 PM ET):
  1. Score all stocks with both models, trade only top N by bull signal strength
  2. Check stop-loss / trailing stop upgrades on open positions
  3. Sell positions where the bear model signals weakness or they drop from top N
  4. Buy top-N bullish stocks (sentiment-gated, confidence-sized)
  5. Flatten all stock positions at 3:50 PM ET to avoid overnight gap risk

Uses stock-prefixed models (stock_bear_model.pth, stock_bull_model.pth).
"""

import json
import time
import datetime
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from order_utils import (
    get_stock_quote, place_stock_limit_order, manage_order_lifecycle,
    get_all_positions, should_trade, cancel_all_open_orders,
    reconstruct_positions, check_circuit_breaker, emergency_flatten,
    compute_limit_price,
)
from predict_now import load_dual_models, get_live_prediction
from market_data import fetch_spy_bars_alpaca, get_live_atr
from trading_utils import get_api, get_model_mtime, choose_inference_device, cooldown_ok, predict_symbol
from hw_monitor import get_gpu_temp
from sentiment import sentiment_gate, get_market_sentiment
from stock_config import load_stock_universe
from llm_config import load_llm_config
from llm_analyst import analyze_trades
from fundamentals import get_fundamentals, get_insider_activity, get_filing_summary, format_fundamentals_for_llm
from trade_journal import log_decision

# --- CONFIGURATION ---

STOCK_UNIVERSE = load_stock_universe()

_PRED_CACHE_FILE = Path(__file__).resolve().parent / "stock_predictions.json"

TOP_N = 10                   # Trade only top N stocks by bull signal
NOTIONAL_PER_STOCK = 2500    # $2,500 per position
MAX_EXPOSURE = 25000         # Max total stock exposure
ORDER_TIMEOUT = 30           # Seconds to wait for limit fill
LOOP_INTERVAL = 30           # Seconds between checks
COOLDOWN_MINUTES = 30        # Min time between trades on same symbol
MAX_PREDICTION_WORKERS = 5
TEMP_LOG_EVERY_N_CYCLES = 10
THERMAL_THROTTLE_TEMP = 75
MODEL_PREFIX = 'stock'       # stock_bear_model.pth, stock_bull_model.pth

# ATR-based stop-loss / trailing stop / take-profit settings
ATR_STOP_MULTIPLIER = 2.0          # stop = entry - (ATR * 2.0)
ATR_TRAIL_MULTIPLIER = 1.5         # trail = hwm - (ATR * 1.5)
ATR_TRAIL_ACTIVATE_PCT = 0.015     # activate trailing at 1.5% profit
ATR_STOP_FLOOR_PCT = 0.015         # min stop distance 1.5%
ATR_STOP_CEIL_PCT = 0.08           # max stop distance 8%
STOCK_TAKE_PROFIT_RR = 2.0         # 2:1 risk-reward take-profit
STOCK_TAKE_PROFIT_CEIL_PCT = 0.15  # max take-profit distance 15%

# Fallback fixed percentages (used when ATR unavailable)
STOCK_STOP_LOSS_PCT = 0.03        # 3% hard stop-loss
STOCK_TRAIL_PCT = 0.02            # 2% trailing stop
STOCK_TP_PCT = 0.10               # 10% take-profit cap
CIRCUIT_BREAKER_PCT = 0.05        # 5% daily equity drawdown triggers flatten

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
FLATTEN_HOUR = 15
FLATTEN_MINUTE = 50


def _write_prediction_cache(bear_preds, bull_preds, top_symbols):
    """Write prediction scores to JSON for GUI consumption (no torch needed)."""
    try:
        data = {}
        all_syms = set(bear_preds) | set(bull_preds)
        for sym in sorted(all_syms):
            bear = bear_preds.get(sym)
            bull = bull_preds.get(sym)
            score = (bull or 0) - abs(bear or 0)
            if sym in top_symbols:
                signal = "BULL"
            elif bear is not None and bear < 0:
                if bull is not None and bull > 0:
                    signal = "DISAGREE"
                else:
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
        print(f"  [CACHE] Error writing prediction cache: {e}")


# --- MARKET HOURS HELPERS ---

def _get_eastern_now():
    """Get current time in US/Eastern."""
    import zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo('US/Eastern'))


def _is_market_hours():
    """Check if current time is within regular market hours."""
    now = _get_eastern_now()
    if now.weekday() >= 5:  # 0=Mon, 6=Sun
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


# --- EXPOSURE TRACKING ---

def _get_current_exposure(api):
    """Calculate total stock exposure from current positions."""
    positions = get_all_positions(api)
    total = 0.0
    for sym, pos in positions.items():
        if '/' not in sym and 'USD' not in sym:
            total += abs(float(pos.market_value))
    return total


# --- END-OF-DAY FLATTEN ---

def flatten_all_stocks(api, positions):
    """Sell all stock positions for end-of-day flatten."""
    print("\n[FLATTEN] Selling all stock positions before market close...")

    # Cancel all outstanding stop orders first
    for symbol, info in list(positions.items()):
        if info.get('stop_order_id'):
            try:
                api.cancel_order(info['stop_order_id'])
                print(f"  [FLATTEN] {symbol}: Canceled stop order {info['stop_order_id']}")
            except Exception:
                pass

    for symbol in list(positions):
        try:
            pos = api.get_position(symbol)
            qty = int(float(pos.qty))
            if qty <= 0:
                del positions[symbol]
                continue

            quote = get_stock_quote(api, symbol)
            if quote is not None:
                order = place_stock_limit_order(api, symbol, 'sell', qty, quote,
                                                time_in_force='day', offset_bps=10)
                if order:
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    if result:
                        del positions[symbol]
                        print(f"  [FLATTEN] {symbol}: Sold {qty} shares")
            else:
                api.submit_order(symbol=symbol, qty=qty, side='sell',
                                type='market', time_in_force='day')
                del positions[symbol]
                print(f"  [FLATTEN] {symbol}: Market sold {qty} shares")

        except Exception as e:
            print(f"  [FLATTEN] {symbol}: Error: {e}")
        time.sleep(0.5)

    return positions


# --- MAIN LOOP ---

def run_stock_bot():
    api = get_api()

    # ── Load stock-specific dual models ──
    print("Loading stock prediction models...")
    try:
        inference_device = choose_inference_device()
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
    bear_mtime = get_model_mtime(f'{MODEL_PREFIX}_bear_model.pth')
    bull_mtime = get_model_mtime(f'{MODEL_PREFIX}_bull_model.pth')

    # ── Cancel stale orders ──
    cancel_all_open_orders(api)

    # ── Reconstruct stock positions from API ──
    positions = reconstruct_positions(api, STOCK_UNIVERSE, asset_type='stock')
    if positions:
        print(f"Existing stock positions: {', '.join(positions)}")
        for sym, info in positions.items():
            entry_atr = get_live_atr(api, sym, asset_type='stock')
            info['entry_atr'] = entry_atr
            atr_str = f"ATR=${entry_atr:.2f}" if entry_atr else "ATR=N/A"
            print(f"  {sym}: qty={info['qty']}, entry=${info['entry_price']:.2f}, hwm=${info['high_water_mark']:.2f}, {atr_str}")

    last_trade_time = {}

    print("\n--- STOCK TRADING BOT STARTED ---")
    print(f"Universe: {len(STOCK_UNIVERSE)} stocks, trading top {TOP_N}")
    print(f"Notional: ${NOTIONAL_PER_STOCK}/position, max ${MAX_EXPOSURE} exposure")
    print(f"Loop interval: {LOOP_INTERVAL}s | Cooldown: {COOLDOWN_MINUTES} min")
    print(f"Flatten at: {FLATTEN_HOUR}:{FLATTEN_MINUTE:02d} ET")
    print(f"Sentiment gating: ENABLED")

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

        # ── Wait for market open ──
        if not _is_market_hours():
            if cycle == 1 or cycle % 20 == 0:
                print(f"\n[WAIT] {eastern_now.strftime('%Y-%m-%d %H:%M ET')} — Market closed. "
                      f"Next check in {LOOP_INTERVAL}s...")
            time.sleep(LOOP_INTERVAL)
            continue

        print(f"\n--- CYCLE {cycle}: {eastern_now.strftime('%Y-%m-%d %H:%M:%S ET')} ---")

        # ── Market sentiment check (logged periodically) ──
        if cycle % TEMP_LOG_EVERY_N_CYCLES == 1:
            mkt = get_market_sentiment()
            if mkt is not None:
                print(f"[SENTIMENT] Market: score={mkt['sentiment_score']:+.2f}, "
                      f"articles={mkt['article_count']}, "
                      f"pos={mkt['positive_ratio']:.0%}/neg={mkt['negative_ratio']:.0%}")

        # ── Flatten check ──
        if _is_flatten_time() and not flattened_today:
            positions = flatten_all_stocks(api, positions)
            flattened_today = True
            print("[FLATTEN] Done. No more trades today.")
            time.sleep(LOOP_INTERVAL)
            continue

        if flattened_today:
            time.sleep(LOOP_INTERVAL)
            continue

        # ── Circuit breaker check ──
        tripped, dd = check_circuit_breaker(api, max_drawdown_pct=CIRCUIT_BREAKER_PCT)
        if tripped:
            print(f"[CIRCUIT BREAKER] Daily drawdown {dd:.1%} >= {CIRCUIT_BREAKER_PCT:.0%}, flattening all positions!")
            emergency_flatten(api)
            positions.clear()
            print("[CIRCUIT BREAKER] Sleeping 1 hour before resuming...")
            time.sleep(3600)
            continue

        # ── Hot-reload check (models + universe) ──
        new_bear_mt = get_model_mtime(f'{MODEL_PREFIX}_bear_model.pth')
        new_bull_mt = get_model_mtime(f'{MODEL_PREFIX}_bull_model.pth')
        if new_bear_mt != bear_mtime or new_bull_mt != bull_mtime:
            print("[HOT-RELOAD] Stock model files changed, reloading...")
            try:
                inference_device = choose_inference_device()
                bear_model, bear_config, bull_model, bull_config, scaler_X, feature_cols = \
                    load_dual_models(inference_device, prefix=MODEL_PREFIX)
                bear_threshold = bear_config.get('bull_threshold', 0.15)
                bull_threshold = bull_config.get('bull_threshold', 0.15)
                bear_mtime = new_bear_mt
                bull_mtime = new_bull_mt
                print(f"[HOT-RELOAD] Success (bear_th={bear_threshold:.2f}, bull_th={bull_threshold:.2f})")
            except Exception as e:
                print(f"[HOT-RELOAD] Failed: {e}, keeping current models")

        # Reload universe each cycle so GUI edits take effect
        STOCK_UNIVERSE = load_stock_universe()

        # ── Log GPU temp periodically ──
        if cycle % TEMP_LOG_EVERY_N_CYCLES == 0:
            temp = get_gpu_temp()
            if temp is not None:
                print(f"[HW] GPU temp: {temp:.0f}C")

        # ── Stop fill detection + trailing stop upgrade ──
        for symbol in list(positions):
            info = positions[symbol]
            if info.get('stop_order_id'):
                try:
                    stop_order = api.get_order(info['stop_order_id'])
                    if stop_order.status == 'filled':
                        print(f"  [STOP-FILL] {symbol}: Stop order filled at ${stop_order.filled_avg_price}")
                        del positions[symbol]
                        last_trade_time[symbol] = datetime.datetime.now()
                        continue
                    elif stop_order.status in ('canceled', 'expired', 'rejected'):
                        info['stop_order_id'] = None
                except Exception:
                    info['stop_order_id'] = None

            quote = get_stock_quote(api, symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            entry_price = info['entry_price']
            info['high_water_mark'] = max(info['high_water_mark'], current_price)

            # Determine ATR-based or fallback trail width
            entry_atr = info.get('entry_atr')
            if entry_atr is not None and entry_price > 0:
                raw_trail_dist = (entry_atr * ATR_TRAIL_MULTIPLIER) / entry_price
                trail_pct = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_trail_dist))
            else:
                trail_pct = STOCK_TRAIL_PCT

            if (not info.get('trailing_activated')
                    and current_price >= entry_price * (1 + ATR_TRAIL_ACTIVATE_PCT)
                    and info.get('stop_order_id')):
                try:
                    api.cancel_order(info['stop_order_id'])
                    time.sleep(0.5)
                    trail_order = api.submit_order(
                        symbol=symbol,
                        qty=int(info['qty']),
                        side='sell',
                        type='trailing_stop',
                        trail_percent=round(trail_pct * 100, 1),
                        time_in_force='day',
                    )
                    info['stop_order_id'] = trail_order.id
                    info['trailing_activated'] = True
                    print(f"  [TRAIL] {symbol}: Upgraded to trailing stop ({trail_pct:.1%}) "
                          f"at ${current_price:.2f} (entry=${entry_price:.2f})")
                except Exception as e:
                    print(f"  [TRAIL] {symbol}: Upgrade error: {e}")

        # ── Fetch SPY bars for relative strength ──
        spy_close = None
        try:
            spy_df = fetch_spy_bars_alpaca(api)
            if spy_df is not None:
                spy_close = spy_df['Close']
        except Exception as e:
            print(f"  [SPY] Error fetching benchmark: {e}")

        # ── Get predictions for ALL stocks in parallel ──
        bear_preds = {}
        bull_preds = {}
        inference_device = choose_inference_device()
        if inference_device == 'cpu':
            print("[HW] GPU unavailable, using CPU for inference")

        with ThreadPoolExecutor(max_workers=MAX_PREDICTION_WORKERS) as executor:
            futures = {}
            for symbol in STOCK_UNIVERSE:
                f = executor.submit(
                    predict_symbol, api, symbol,
                    bear_model, bear_config, bull_model, bull_config,
                    scaler_X, feature_cols, inference_device,
                    asset_type='stock', benchmark_close=spy_close,
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

        # ── Dynamic top N selection by bull signal strength (bear agreement filter) ──
        agree_candidates = {sym: pred for sym, pred in bull_preds.items()
                            if bear_preds.get(sym) is None or bear_preds[sym] >= -bear_threshold}
        n_filtered = len(bull_preds) - len(agree_candidates)
        if n_filtered:
            print(f"[RANK] {n_filtered} symbol(s) excluded by bear disagreement")
        ranked = sorted(agree_candidates.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [sym for sym, _ in ranked[:TOP_N]]
        print(f"[RANK] Top {TOP_N}: {', '.join(f'{s}({bull_preds[s]:+.4f})' for s in top_symbols)}")

        # Write prediction cache for GUI
        _write_prediction_cache(bear_preds, bull_preds, top_symbols)

        # ── LLM pre-trade analysis ──
        llm_scores = {}
        llm_cfg = load_llm_config()
        if llm_cfg.get("enabled"):
            candidates = []
            for symbol in top_symbols:
                if symbol in positions:
                    continue
                bp = bull_preds.get(symbol, 0)
                if bp < bull_threshold:
                    continue
                fund = get_fundamentals(symbol, 'stock')
                insider = get_insider_activity(symbol)
                filing_sum = get_filing_summary(symbol)
                fund_text = format_fundamentals_for_llm(symbol, fund, insider, filing_sum)
                candidates.append({
                    'symbol': symbol,
                    'bull_pred': bp,
                    'bear_pred': bear_preds.get(symbol),
                    'fundamentals_text': fund_text,
                })
            if candidates:
                try:
                    acct = api.get_account()
                    equity = float(acct.equity)
                except Exception:
                    equity = 0
                llm_scores = analyze_trades(
                    candidates, 'stock', equity=equity,
                    positions=list(positions.keys()),
                )
                if llm_scores:
                    print("[LLM] Scores: " + ", ".join(f"{s}={v.get('m', 1.0):.1f}x" for s, v in llm_scores.items()))

        # ── SELL: bearish positions ──
        for symbol in list(positions):
            try:
                pos = api.get_position(symbol)
            except Exception:
                del positions[symbol]
                continue

            bear_pred = bear_preds.get(symbol)
            sell_reason = None
            if bear_pred is not None and bear_pred < -bear_threshold:
                sell_reason = f"bear_pred={bear_pred:+.4f}%"
            elif symbol not in top_symbols and bear_pred is not None and bear_pred < 0:
                sell_reason = f"dropped from top {TOP_N} (bear={bear_pred:+.4f}%)"

            if sell_reason is None:
                continue

            if not cooldown_ok(last_trade_time, symbol, COOLDOWN_MINUTES):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: Sell signal but in cooldown ({remaining/60:.1f} min left)")
                continue

            print(f"  {symbol}: SELLING ({sell_reason})")
            qty = int(float(pos.qty))
            if qty <= 0:
                del positions[symbol]
                continue

            info = positions[symbol]
            if info.get('stop_order_id'):
                try:
                    api.cancel_order(info['stop_order_id'])
                except Exception:
                    pass

            quote = get_stock_quote(api, symbol)
            if quote is not None:
                order = place_stock_limit_order(api, symbol, 'sell', qty, quote,
                                                time_in_force='day')
                if order:
                    result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                                   fallback_to_market=True)
                    if result:
                        del positions[symbol]
                        last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(0.5)

        # ── BUY: top N bullish stocks we don't hold ──
        current_exposure = _get_current_exposure(api)
        for symbol in top_symbols:
            if symbol in positions:
                continue

            if current_exposure >= MAX_EXPOSURE:
                print(f"  Max exposure ${MAX_EXPOSURE} reached, no more buys")
                break

            if not cooldown_ok(last_trade_time, symbol, COOLDOWN_MINUTES):
                remaining = COOLDOWN_MINUTES * 60 - (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
                print(f"  {symbol}: In cooldown ({remaining/60:.1f} min left)")
                continue

            bull_pred = bull_preds.get(symbol)
            if bull_pred is None or bull_pred < bull_threshold:
                if bull_pred is not None:
                    print(f"  {symbol}: Bull pred {bull_pred:+.4f}% < {bull_threshold:.2f}, skipping")
                continue

            # Bear agreement check — skip if bear model disagrees
            bear_pred = bear_preds.get(symbol)
            if bear_pred is not None and bear_pred < -bear_threshold:
                print(f"  {symbol}: Bull {bull_pred:+.4f}% but bear {bear_pred:+.4f}% disagrees, skipping")
                continue

            quote = get_stock_quote(api, symbol)
            if quote is None:
                continue

            if not should_trade(bull_pred, quote['spread_pct']):
                print(f"  {symbol}: Bull pred {bull_pred:+.4f}% too weak vs spread "
                      f"{quote['spread_pct']:.3f}%, skipping")
                continue

            # Confidence-based sizing
            if bull_pred is not None and bull_threshold > 0:
                confidence = min(2.0, max(0.5, bull_pred / bull_threshold))
            else:
                confidence = 1.0
            sized_notional = int(NOTIONAL_PER_STOCK * confidence)

            # Sentiment gate
            gate, gate_reasons = sentiment_gate(symbol, 'stock')
            if gate <= 0:
                print(f"  {symbol}: BLOCKED by sentiment ({', '.join(gate_reasons)})")
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "bull_pred": bull_pred, "bear_pred": bear_preds.get(symbol),
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                              "llm_multiplier": None, "llm_reasoning": None})
                continue
            effective_notional = int(sized_notional * gate)

            # LLM multiplier
            llm_info = llm_scores.get(symbol, {})
            llm_mult = llm_info.get('m', 1.0)
            llm_reason = llm_info.get('r', '')
            if llm_mult <= 0:
                print(f"  {symbol}: BLOCKED by LLM ({llm_reason})")
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "llm_block",
                              "bull_pred": bull_pred, "bear_pred": bear_preds.get(symbol),
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                              "llm_multiplier": llm_mult, "llm_reasoning": llm_reason})
                continue
            effective_notional = int(effective_notional * llm_mult)

            sizing_info = f"conf={confidence:.2f}x"
            if gate != 1.0:
                sizing_info += f", sent={gate:.2f}x"
            if llm_mult != 1.0:
                sizing_info += f", llm={llm_mult:.1f}x"
            if gate_reasons:
                sizing_info += f" ({', '.join(gate_reasons)})"
            print(f"  {symbol}: Sizing ${effective_notional} [{sizing_info}]")

            # Calculate qty (whole shares)
            price = quote['midpoint']
            if price <= 0:
                continue
            qty = int(effective_notional / price)
            if qty <= 0:
                print(f"  {symbol}: Price ${price:.2f} too high for ${effective_notional} notional")
                continue

            print(f"  {symbol}: BUYING {qty} shares @ ~${price:.2f} (bull={bull_pred:+.4f}%)")
            limit_price = compute_limit_price('buy', quote, offset_bps=5)
            limit_price = round(limit_price, 2)

            # Compute ATR-based stop and take-profit
            entry_atr = get_live_atr(api, symbol, asset_type='stock')
            if entry_atr is not None and limit_price > 0:
                raw_stop_dist = (entry_atr * ATR_STOP_MULTIPLIER) / limit_price
                stop_dist = max(ATR_STOP_FLOOR_PCT, min(ATR_STOP_CEIL_PCT, raw_stop_dist))
                raw_tp_dist = stop_dist * STOCK_TAKE_PROFIT_RR
                tp_dist = min(STOCK_TAKE_PROFIT_CEIL_PCT, raw_tp_dist)
                print(f"  [ATR-STOP] {symbol}: ATR=${entry_atr:.2f}, stop={stop_dist:.1%}, tp={tp_dist:.1%}")
            else:
                stop_dist = STOCK_STOP_LOSS_PCT
                tp_dist = STOCK_TP_PCT
                print(f"  [ATR-STOP] {symbol}: ATR unavailable, using fixed stops")

            stop_price = round(limit_price * (1 - stop_dist), 2)
            tp_price = round(limit_price * (1 + tp_dist), 2)
            try:
                order = api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='limit',
                    limit_price=limit_price,
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss={'stop_price': stop_price},
                    take_profit={'limit_price': tp_price},
                )
                print(f"  [ORDER] {symbol}: buy {qty} @ ${limit_price:.2f} "
                      f"(stop=${stop_price:.2f}, tp=${tp_price:.2f})")
            except Exception as e:
                print(f"  [ORDER] {symbol}: Bracket order error: {e}")
                order = None

            if order:
                result = manage_order_lifecycle(api, order.id, timeout=ORDER_TIMEOUT,
                                               fallback_to_market=False)
                if result and result.status == 'filled':
                    child_stop_id = None
                    try:
                        legs = api.list_orders(status='open', symbols=[symbol])
                        for leg in legs:
                            if leg.side == 'sell' and leg.type in ('stop', 'stop_limit'):
                                child_stop_id = leg.id
                                break
                    except Exception:
                        pass

                    fill_price = float(result.filled_avg_price)
                    positions[symbol] = {
                        'qty': qty,
                        'entry_price': fill_price,
                        'high_water_mark': fill_price,
                        'stop_order_id': child_stop_id,
                        'trailing_activated': False,
                        'entry_atr': entry_atr,
                    }
                    log_decision({"symbol": symbol, "action": "buy",
                                  "bull_pred": bull_pred, "bear_pred": bear_preds.get(symbol),
                                  "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                                  "llm_multiplier": llm_mult, "llm_reasoning": llm_reason,
                                  "final_notional": effective_notional, "confidence": confidence,
                                  "skip_reason": None})
                    last_trade_time[symbol] = datetime.datetime.now()
                    current_exposure += qty * fill_price
            time.sleep(0.5)

        # ── Thermal throttling ──
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
