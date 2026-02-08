"""Shared order utilities for limit orders, lifecycle management, and position verification.

Covers the full order lifecycle: quote fetching, limit/market order placement,
fill polling with timeout + market fallback, position verification, circuit
breaker, and emergency flatten.
"""

import time
import math
import datetime


# --- SPREAD / QUOTE HELPERS ---

def get_quote(api, symbol, asset_type='crypto'):
    """Get real-time bid/ask quote for any asset via Alpaca.

    Args:
        api: Alpaca REST API object
        symbol: Alpaca symbol (e.g. 'BTC/USD' for crypto, 'TSLA' for stock)
        asset_type: 'crypto' or 'stock' — determines which Alpaca endpoint to call

    Returns:
        dict with bid, ask, spread, midpoint, spread_pct — or None on error.
    """
    try:
        if asset_type == 'crypto':
            quotes = api.get_latest_crypto_quotes([symbol])
            q = quotes[symbol]
        else:
            q = api.get_latest_quote(symbol)

        bid = float(q.bp)
        ask = float(q.ap)
        spread = ask - bid
        midpoint = (bid + ask) / 2.0
        spread_pct = (spread / midpoint) * 100.0 if midpoint > 0 else 0.0
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'midpoint': midpoint,
            'spread_pct': spread_pct,
        }
    except Exception as e:
        print(f"  [QUOTE] Error fetching quote for {symbol}: {e}")
        return None


def get_crypto_quote(api, symbol):
    """Get real-time bid/ask for a crypto symbol. Wrapper around get_quote()."""
    return get_quote(api, symbol, asset_type='crypto')


def get_stock_quote(api, symbol):
    """Get real-time bid/ask for a stock symbol. Wrapper around get_quote()."""
    return get_quote(api, symbol, asset_type='stock')


def compute_limit_price(side, quote_info, offset_bps=5):
    """Compute a limit price near the midpoint.
    For buys: midpoint + offset (willing to pay slightly above mid).
    For sells: midpoint - offset (willing to sell slightly below mid).
    offset_bps: basis points offset from midpoint (5 bps = 0.05%).
    """
    mid = quote_info['midpoint']
    offset = mid * (offset_bps / 10000.0)
    if side == 'buy':
        return round(mid + offset, 4)
    else:
        return round(mid - offset, 4)


# --- ORDER PLACEMENT ---

def place_limit_order(api, symbol, side, notional, quote_info,
                      time_in_force='gtc', offset_bps=5):
    """Place a limit order. Computes qty from notional/price since Alpaca
    only supports `notional` for market orders.
    Returns the order object or None on error.
    """
    limit_price = compute_limit_price(side, quote_info, offset_bps)
    qty = math.floor((notional / limit_price) * 1e8) / 1e8  # 8 decimal places for crypto

    if qty <= 0:
        print(f"  [ORDER] {symbol}: qty too small (notional=${notional}, price=${limit_price})")
        return None

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        print(f"  [ORDER] {symbol}: {side} {qty} @ ${limit_price} (mid=${quote_info['midpoint']:.4f}, "
              f"spread={quote_info['spread_pct']:.3f}%)")
        return order
    except Exception as e:
        print(f"  [ORDER] {symbol}: {side} LIMIT ERROR: {e}")
        return None


def place_stock_limit_order(api, symbol, side, qty, quote_info,
                            time_in_force='day', offset_bps=5):
    """Place a limit order for stocks (integer qty, day TIF)."""
    limit_price = compute_limit_price(side, quote_info, offset_bps)

    if qty <= 0:
        print(f"  [ORDER] {symbol}: qty must be > 0")
        return None

    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=round(limit_price, 2),
            time_in_force=time_in_force,
        )
        print(f"  [ORDER] {symbol}: {side} {qty} @ ${limit_price:.2f} (mid=${quote_info['midpoint']:.2f}, "
              f"spread={quote_info['spread_pct']:.3f}%)")
        return order
    except Exception as e:
        print(f"  [ORDER] {symbol}: {side} LIMIT ERROR: {e}")
        return None


# --- ORDER LIFECYCLE ---

def manage_order_lifecycle(api, order_id, timeout=30, poll_interval=2, fallback_to_market=True):
    """Poll order status. Cancel if unfilled after timeout.
    If fallback_to_market is True, places a market order after cancellation.
    Returns the final order object.
    """
    elapsed = 0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            order = api.get_order(order_id)
        except Exception as e:
            print(f"  [LIFECYCLE] Error checking order {order_id}: {e}")
            return None

        if order.status == 'filled':
            print(f"  [LIFECYCLE] Order {order_id} FILLED ({order.filled_qty} @ ${order.filled_avg_price})")
            return order
        elif order.status in ('canceled', 'expired', 'rejected'):
            print(f"  [LIFECYCLE] Order {order_id} terminal status: {order.status}")
            return order

    # Timeout reached — cancel
    print(f"  [LIFECYCLE] Order {order_id} unfilled after {timeout}s, canceling...")
    try:
        api.cancel_order(order_id)
        time.sleep(1)  # give cancel time to process
    except Exception as e:
        print(f"  [LIFECYCLE] Cancel error: {e}")

    # Check final state after cancel
    try:
        order = api.get_order(order_id)
        if order.status == 'filled':
            print(f"  [LIFECYCLE] Order filled during cancel (race condition), keeping.")
            return order
    except Exception:
        pass

    if fallback_to_market:
        print(f"  [LIFECYCLE] Falling back to market order...")
        try:
            market_order = api.submit_order(
                symbol=order.symbol,
                qty=order.qty,
                side=order.side,
                type='market',
                time_in_force='gtc',
            )
            print(f"  [LIFECYCLE] Market fallback submitted: {market_order.id}")
            return market_order
        except Exception as e:
            print(f"  [LIFECYCLE] Market fallback error: {e}")
            return None

    return None


# --- POSITION VERIFICATION ---

def verify_position(api, symbol):
    """Check actual position via API. Returns position object or None if no position.
    Handles Alpaca's crypto symbol format (BTC/USD -> BTCUSD).
    """
    # Try the symbol as-is first, then without the slash
    candidates = [symbol]
    if '/' in symbol:
        candidates.append(symbol.replace('/', ''))
    for sym in candidates:
        try:
            pos = api.get_position(sym)
            qty = float(pos.qty)
            if qty > 0:
                return pos
        except Exception:
            continue
    return None


def get_all_positions(api):
    """Returns a dict of symbol -> position object for all current positions."""
    try:
        positions = api.list_positions()
        return {pos.symbol: pos for pos in positions}
    except Exception as e:
        print(f"  [POSITIONS] Error listing positions: {e}")
        return {}


# --- TRADE GATING ---

def should_trade(predicted_return, spread_pct, min_edge=2.0):
    """Only trade if predicted return > min_edge * round-trip spread cost.
    predicted_return: expected % move (e.g. 0.5 means +0.5%)
    spread_pct: current spread as % of price
    min_edge: multiplier — predicted return must exceed this * spread to trade
    """
    round_trip_cost = spread_pct  # approximate: pay ~half spread each way
    threshold = round_trip_cost * min_edge
    return abs(predicted_return) > threshold


# --- CLEANUP ---

def cancel_all_open_orders(api):
    """Cancel all open orders. Call on startup to clean stale state."""
    try:
        orders = api.list_orders(status='open')
        if orders:
            print(f"  [CLEANUP] Canceling {len(orders)} open order(s)...")
            api.cancel_all_orders()
            time.sleep(1)
        else:
            print("  [CLEANUP] No open orders to cancel.")
    except Exception as e:
        print(f"  [CLEANUP] Error canceling orders: {e}")


# --- POSITION RECONSTRUCTION ---

def reconstruct_positions(api, symbols, asset_type='crypto'):
    """Rebuild position dict from Alpaca API (survive restarts).
    Returns: {symbol: {qty, entry_price, high_water_mark, stop_order_id, trailing_activated}}
    """
    positions = {}
    for sym in symbols:
        candidates = [sym]
        if '/' in sym:
            candidates.append(sym.replace('/', ''))
        for candidate in candidates:
            try:
                pos = api.get_position(candidate)
                qty = float(pos.qty)
                if qty > 0:
                    entry_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    positions[sym] = {
                        'qty': qty,
                        'entry_price': entry_price,
                        'high_water_mark': max(entry_price, current_price),
                        'stop_order_id': None,
                        'trailing_activated': False,
                    }
                    break
            except Exception:
                continue
    return positions


# --- CIRCUIT BREAKER ---

def check_circuit_breaker(api, max_drawdown_pct=0.05):
    """Check if daily equity drawdown exceeds threshold.
    Returns (tripped: bool, drawdown_pct: float).
    """
    account = api.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)  # previous close
    if last_equity <= 0:
        return False, 0.0
    drawdown = (last_equity - equity) / last_equity
    return drawdown >= max_drawdown_pct, drawdown


# --- EMERGENCY FLATTEN ---

def emergency_flatten(api):
    """Market-sell all positions immediately."""
    print("[EMERGENCY] Flattening all positions...")
    try:
        api.cancel_all_orders()
        time.sleep(1)
    except Exception as e:
        print(f"  [EMERGENCY] Cancel orders error: {e}")

    try:
        all_positions = api.list_positions()
    except Exception as e:
        print(f"  [EMERGENCY] List positions error: {e}")
        return

    for pos in all_positions:
        try:
            side = 'sell' if float(pos.qty) > 0 else 'buy'
            api.submit_order(
                symbol=pos.symbol,
                qty=abs(float(pos.qty)),
                side=side,
                type='market',
                time_in_force='gtc',
            )
            print(f"  [EMERGENCY] {pos.symbol}: Market {side} {pos.qty}")
        except Exception as e:
            print(f"  [EMERGENCY] {pos.symbol}: {e}")
