"""Shared order utilities for limit orders, lifecycle management, and position verification.

Covers the full order lifecycle: quote fetching, limit/market order placement,
fill polling with timeout + market fallback, position verification, circuit
breaker, and emergency flatten.
"""

import time
import math
import datetime
import uuid


def make_client_order_id(tag: str) -> str:
    """Generate a unique client_order_id (Alpaca cap: 48 chars).

    Tagging orders lets us (a) avoid duplicate submissions on retry and
    (b) scope cleanup to this bot's own orders instead of the whole account.
    """
    return f"{tag[:14]}-{uuid.uuid4().hex[:20]}"


def _symbol_variants(symbol: str) -> set[str]:
    """All spellings Alpaca may use for a symbol ('BTC/USD' <-> 'BTCUSD')."""
    variants = {symbol}
    if '/' in symbol:
        variants.add(symbol.replace('/', ''))
    return variants


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
        if midpoint <= 0 or bid <= 0 or ask <= 0:
            # Degenerate quote (halted/stale feed). Treating it as valid leads
            # to division-by-zero limit prices downstream — reject instead.
            print(f"  [QUOTE] {symbol}: degenerate quote bid={bid} ask={ask}, ignoring")
            return None

        # STALENESS: a frozen feed means stops silently never fire — the
        # loop happily compares positions against a price that stopped
        # updating. Reject quotes older than 3 minutes (and let callers'
        # quote-unavailable paths handle it loudly).
        try:
            qt = getattr(q, 't', None)
            if qt is not None:
                if hasattr(qt, 'to_pydatetime'):
                    qt = qt.to_pydatetime()
                age = (datetime.datetime.now(datetime.timezone.utc)
                       - qt.astimezone(datetime.timezone.utc)).total_seconds()
                if age > 180:
                    print(f"  [QUOTE] {symbol}: quote is {age:.0f}s stale, ignoring")
                    return None
        except Exception:
            pass  # unparseable timestamp — don't block on the check itself
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
    """Compute a limit price near the midpoint with spread-aware offset.

    For buys: midpoint + offset (willing to pay slightly above mid).
    For sells: midpoint - offset (willing to sell slightly below mid).
    offset_bps: basis points offset from midpoint (5 bps = 0.05%).

    Spread-aware: when spread is wide (> 0.1%), use a proportional offset
    instead of the fixed offset to avoid crossing the spread unnecessarily.
    """
    mid = quote_info['midpoint']
    spread_pct = quote_info.get('spread_pct', 0)

    # Dynamic offset: tight spread → fixed bps, wide spread → proportional
    if spread_pct > 0.1:
        # Use 20% of half-spread as offset (more conservative for wide spreads)
        effective_offset = mid * (spread_pct / 100.0) * 0.1
    else:
        effective_offset = mid * (offset_bps / 10000.0)

    if side == 'buy':
        return round(mid + effective_offset, 4)
    else:
        return round(mid - effective_offset, 4)


def _round_price_band(px, asset_type='stock'):
    """Per-price-band tick rounding (wave-7): equities round to 2dp at/above
    $1 and 4dp below $1 — sub-penny limits on >=$1 names are rejected and
    become silent non-fills. Crypto keeps finer resolution."""
    px = float(px)
    if asset_type == 'crypto':
        return round(px, 6 if px < 1 else 2)
    return round(px, 2 if px >= 1.0 else 4)


def _bid_ask(quote_info):
    """Best bid/ask from a quote dict, deriving from midpoint+spread_pct when
    explicit bid/ask are absent (spread_pct is the FULL spread, % of price)."""
    bid = quote_info.get('bid')
    ask = quote_info.get('ask')
    if bid and ask:
        return float(bid), float(ask)
    mid = float(quote_info['midpoint'])
    half = mid * (float(quote_info.get('spread_pct', 0.0)) / 100.0) / 2.0
    return mid - half, mid + half


def ioc_limit_price(side, quote_info, cap_bps, asset_type='stock'):
    """Marketable-limit price for an IOC: cross the touch but cap how far past
    it we will pay. Buys lift the ask up to ask*(1+cap); sells hit the bid down
    to bid*(1-cap). Band-rounded so the limit is never sub-penny-rejected."""
    bid, ask = _bid_ask(quote_info)
    if side == 'buy':
        return _round_price_band(ask * (1.0 + cap_bps / 1e4), asset_type)
    return _round_price_band(bid * (1.0 - cap_bps / 1e4), asset_type)


def place_marketable_ioc(api, symbol, side, qty, quote_info, cap_bps,
                         asset_type='stock'):
    """Submit a slippage-capped marketable IOC (limit + time_in_force='ioc').

    Replaces uncapped type='market' fallbacks: a market order in a thin
    spec-tech book can fill tens-to-hundreds of bps through the quote on the
    exact bad days slippage concentrates. The IOC takes liquidity up to
    cap_bps past the touch, then auto-cancels any unfilled remainder (the
    caller re-chases for entries, or escalates to a true-market backstop for
    exits/flatten so a stop can never silently fail). Returns the order or
    None on submit failure.
    """
    limit = ioc_limit_price(side, quote_info, cap_bps, asset_type)
    try:
        return api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit',
            limit_price=limit, time_in_force='ioc')
    except Exception as e:
        print(f"[IOC] {side} {symbol} submit failed ({e})")
        return None


# --- ORDER PLACEMENT ---

def place_maker_buy(api, symbol, notional, quote_fn, stage_timeout=25,
                    max_reprices=1):
    """Crypto entry ladder: join the bid (maker fee + half-spread saved),
    reprice once at the fresh bid, then go marketable for any remainder.

    Alpaca crypto charges 15bps maker vs 25bps taker per side; a bid-join
    that fills passively also avoids paying the half-spread. With a
    12-48h signal horizon, risking ~1 minute of passive waiting for
    10bps+spread is positive expectancy. Safe to wait now that entries
    get a resting server-side stop the moment they're tracked.

    quote_fn: zero-arg callable returning a fresh quote dict (or None) —
    each rung reprices off live data, not the stale decision quote.

    Returns (final_order_or_None, tactic) where tactic is one of
    'maker', 'maker_reprice', 'maker_partial', 'taker_fallback',
    'unfilled'. The caller must judge success by acquired quantity
    (filled_qty / live position), NOT by the last order's status — a
    95%-filled maker rung that times out is a success.
    """
    remaining = float(notional)
    last = None
    for attempt in range(1 + max_reprices):
        quote = quote_fn()
        bid = (quote or {}).get('bid')
        if not bid or bid <= 0:
            break
        qty = math.floor((remaining / bid) * 1e8) / 1e8
        if qty <= 0:
            break
        try:
            order = api.submit_order(
                symbol=symbol, qty=qty, side='buy', type='limit',
                limit_price=round(bid, 4), time_in_force='gtc',
                client_order_id=make_client_order_id('maker'))
        except Exception as e:
            print(f"  [MAKER] {symbol}: bid-join error: {e}")
            break
        print(f"  [MAKER] {symbol}: joining bid @ ${bid} "
              f"(${remaining:.0f}, attempt {attempt + 1})")
        result = manage_order_lifecycle(api, order.id, timeout=stage_timeout,
                                        fallback_to_market=False)
        if result is not None and getattr(result, 'status', None) == 'filled':
            return result, ('maker' if attempt == 0 else 'maker_reprice')
        # Chase only the unfilled remainder on the next rung
        try:
            filled_qty = float(getattr(result, 'filled_qty', 0) or 0)
            px = float(getattr(result, 'filled_avg_price', 0) or 0) or bid
            remaining -= filled_qty * px
        except (TypeError, ValueError):
            pass
        if result is not None:
            last = result
        if remaining < 10:  # dust left — maker rungs covered the entry
            return last, 'maker_partial'

    # Marketable fallback for whatever's left
    quote = quote_fn()
    if quote is None:
        return last, 'unfilled'
    order = place_limit_order(api, symbol, 'buy', remaining, quote)
    if order is None:
        return last, 'unfilled'
    result = manage_order_lifecycle(api, order.id, timeout=stage_timeout + 5,
                                    fallback_to_market=True)
    return result, 'taker_fallback'


def place_limit_order(api, symbol, side, notional, quote_info,
                      time_in_force='gtc', offset_bps=5,
                      client_order_id=None):
    """Place a limit order. Computes qty from notional/price since Alpaca
    only supports `notional` for market orders.
    Returns the order object or None on error.
    """
    try:
        limit_price = compute_limit_price(side, quote_info, offset_bps)
        if limit_price <= 0:
            print(f"  [ORDER] {symbol}: invalid limit price {limit_price}")
            return None
        qty = math.floor((notional / limit_price) * 1e8) / 1e8  # 8 dp for crypto

        if qty <= 0:
            print(f"  [ORDER] {symbol}: qty too small (notional=${notional}, price=${limit_price})")
            return None

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=limit_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id or make_client_order_id('trader'),
        )
        print(f"  [ORDER] {symbol}: {side} {qty} @ ${limit_price} (mid=${quote_info['midpoint']:.4f}, "
              f"spread={quote_info['spread_pct']:.3f}%)")
        return order
    except Exception as e:
        print(f"  [ORDER] {symbol}: {side} LIMIT ERROR: {e}")
        return None


def place_stock_limit_order(api, symbol, side, qty, quote_info,
                            time_in_force='day', offset_bps=5,
                            client_order_id=None):
    """Place a limit order for stocks (integer qty, day TIF)."""
    if qty <= 0:
        print(f"  [ORDER] {symbol}: qty must be > 0")
        return None

    try:
        limit_price = compute_limit_price(side, quote_info, offset_bps)
        if limit_price <= 0:
            print(f"  [ORDER] {symbol}: invalid limit price {limit_price}")
            return None
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=round(limit_price, 2),
            time_in_force=time_in_force,
            client_order_id=client_order_id or make_client_order_id('trader'),
        )
        print(f"  [ORDER] {symbol}: {side} {qty} @ ${limit_price:.2f} (mid=${quote_info['midpoint']:.2f}, "
              f"spread={quote_info['spread_pct']:.3f}%)")
        return order
    except Exception as e:
        print(f"  [ORDER] {symbol}: {side} LIMIT ERROR: {e}")
        return None


# --- ORDER LIFECYCLE ---

def manage_order_lifecycle(api, order_id, timeout=30, poll_interval=2,
                           fallback_to_market=True, time_in_force='gtc'):
    """Poll order status. Cancel if unfilled after timeout.
    If fallback_to_market is True, places a market order for the REMAINING
    (unfilled) quantity after cancellation — never the full original qty,
    which would double-buy any partially filled portion.
    Returns the final order object.
    """
    # Save order params upfront for market fallback (in case later get_order fails)
    saved_symbol = saved_qty = saved_side = None
    saved_filled = 0.0
    consecutive_errors = 0
    elapsed = 0
    while elapsed < timeout:
        # If the trade_updates stream already saw a terminal state, skip the
        # wait and make one authoritative REST fetch instead of N polls
        try:
            from order_stream import get_order_state, TERMINAL
            streamed = get_order_state(order_id)
        except Exception:
            streamed = None
        if streamed is None or streamed.get('status') not in TERMINAL:
            time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            order = api.get_order(order_id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"  [LIFECYCLE] Error checking order {order_id} ({consecutive_errors}x): {e}")
            if consecutive_errors >= 3:
                # Don't leave a live working order orphaned — best-effort cancel
                print(f"  [LIFECYCLE] Giving up after {consecutive_errors} consecutive errors, canceling order")
                try:
                    api.cancel_order(order_id)
                except Exception:
                    pass
                return None
            continue

        # Cache order params on first successful poll
        if saved_symbol is None:
            saved_symbol = order.symbol
            saved_qty = order.qty
            saved_side = order.side
        try:
            saved_filled = float(order.filled_qty or 0)
        except (TypeError, ValueError):
            pass

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

    # Check final state after cancel (may have filled during the race)
    try:
        order = api.get_order(order_id)
        if order.status == 'filled':
            print(f"  [LIFECYCLE] Order filled during cancel (race condition), keeping.")
            return order
        try:
            saved_filled = float(order.filled_qty or 0)
        except (TypeError, ValueError):
            pass
    except Exception:
        pass

    if fallback_to_market and saved_symbol:
        # Only chase the unfilled remainder
        try:
            remaining = float(saved_qty) - saved_filled
        except (TypeError, ValueError):
            remaining = None
        if remaining is not None and remaining <= 0:
            print(f"  [LIFECYCLE] Fully filled during cancel ({saved_filled}), no fallback needed.")
            try:
                return api.get_order(order_id)
            except Exception:
                return None
        print(f"  [LIFECYCLE] Falling back to market order ({remaining or saved_qty} remaining)...")
        try:
            market_order = api.submit_order(
                symbol=saved_symbol,
                qty=remaining if remaining is not None else saved_qty,
                side=saved_side,
                type='market',
                time_in_force=time_in_force,
                client_order_id=make_client_order_id('mktfb'),
            )
            print(f"  [LIFECYCLE] Market fallback submitted: {market_order.id}")
            # Poll briefly for fill
            for _ in range(3):
                time.sleep(1)
                try:
                    mkt = api.get_order(market_order.id)
                    if mkt.status == 'filled':
                        print(f"  [LIFECYCLE] Market fallback FILLED ({mkt.filled_qty} @ ${mkt.filled_avg_price})")
                        return mkt
                except Exception:
                    pass
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
    """Returns a dict of symbol -> position object, or None on API error."""
    try:
        positions = api.list_positions()
        return {pos.symbol: pos for pos in positions}
    except Exception as e:
        print(f"  [POSITIONS] Error listing positions: {e}")
        return None


# --- TRADE GATING ---

def should_trade(predicted_return, spread_pct, min_edge=2.0,
                 asset_type='crypto', maker=False):
    """Only trade if predicted return clears min_edge x the FULL round-trip
    cost: venue fees (crypto: 25 bps taker / 15 bps maker per side) plus
    spread. The old spread-only hurdle ignored the dominant crypto cost.

    predicted_return: expected % move (e.g. 0.5 means +0.5%)
    spread_pct: current spread as % of price
    min_edge: multiplier — predicted return must exceed this x cost to trade
    asset_type: 'crypto' or 'stock' (fee schedule differs ~10x)
    maker: True when entries rest as maker limit orders
    """
    from fees import required_edge_pct
    # live=True: the LIVE gate blends the crypto entry fee by REALIZED
    # maker share (journals) so an overstated taker assumption doesn't
    # reject genuinely positive-edge entries. Backtest/training paths
    # call fees directly with the conservative static model.
    threshold = required_edge_pct(asset_type, spread_pct, maker, min_edge,
                                  live=True)
    return abs(predicted_return) > threshold


# --- CLEANUP ---

def cancel_all_open_orders(api, symbols=None):
    """Cancel open orders. Call on startup to clean stale state.

    Args:
        symbols: if given, cancel ONLY orders for these symbols (this bot's
            universe). Both bots share one account — an account-wide
            cancel_all_orders() from one bot strips the other bot's
            protective bracket/stop legs.
    """
    try:
        orders = api.list_orders(status='open')
        if not orders:
            print("  [CLEANUP] No open orders to cancel.")
            return
        if symbols is None:
            print(f"  [CLEANUP] Canceling {len(orders)} open order(s) (account-wide)...")
            api.cancel_all_orders()
            time.sleep(1)
            return
        allowed = set()
        for s in symbols:
            allowed |= _symbol_variants(s)
        mine = [o for o in orders if getattr(o, 'symbol', None) in allowed]
        print(f"  [CLEANUP] Canceling {len(mine)}/{len(orders)} open order(s) in this bot's universe...")
        for o in mine:
            try:
                api.cancel_order(o.id)
            except Exception as e:
                print(f"  [CLEANUP] Cancel {o.symbol} {o.id}: {e}")
        if mine:
            time.sleep(1)
    except Exception as e:
        print(f"  [CLEANUP] Error canceling orders: {e}")


def cancel_orders_for_symbol(api, symbol, timeout=5, poll_interval=0.5):
    """Cancel all open orders for one symbol and WAIT until they are gone.

    Alpaca cancellation is async (`pending_cancel`); selling shares that are
    still reserved by a live stop/take-profit leg rejects with
    'insufficient qty'. Returns True when no open orders remain for the
    symbol, False if any survive past the timeout.
    """
    variants = _symbol_variants(symbol)
    try:
        open_orders = [o for o in api.list_orders(status='open')
                       if getattr(o, 'symbol', None) in variants]
    except Exception as e:
        print(f"  [CANCEL] {symbol}: list_orders error: {e}")
        return False
    if not open_orders:
        return True

    for o in open_orders:
        try:
            api.cancel_order(o.id)
        except Exception as e:
            # Already terminal is fine; anything else we'll catch on re-poll
            print(f"  [CANCEL] {symbol}: cancel {o.id}: {e}")

    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            remaining = [o for o in api.list_orders(status='open')
                         if getattr(o, 'symbol', None) in variants]
            if not remaining:
                return True
        except Exception:
            pass
    print(f"  [CANCEL] {symbol}: open orders still pending after {timeout}s")
    return False


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

    Returns (tripped: bool, drawdown_pct: float | None).
    drawdown_pct is None when the account API is unreachable — callers should
    treat that as "unknown risk state" and skip NEW entries (fail closed)
    rather than trade as if everything is fine.
    """
    try:
        account = api.get_account()
        equity = float(account.equity)
        last_equity = float(account.last_equity)  # previous close
        if last_equity <= 0:
            return False, 0.0
        drawdown = (last_equity - equity) / last_equity
        return drawdown >= max_drawdown_pct, drawdown
    except Exception as e:
        print(f"  [CIRCUIT BREAKER] API error: {e}")
        return False, None


# --- EMERGENCY FLATTEN ---

def emergency_flatten(api, symbols=None, tif_for_symbol=None):
    """Market-sell positions immediately and VERIFY the sells went through.

    Args:
        symbols: if given, flatten ONLY positions in these symbols (this
            bot's universe). Both bots share one account — an unscoped
            flatten from the crypto loop would liquidate the stock book
            (and vice versa).
        tif_for_symbol: optional callable symbol -> time_in_force. Defaults
            to 'gtc' for crypto-style symbols and 'day' for stocks.

    Returns:
        list of symbols whose flatten could not be confirmed.
    """
    scope = "scoped" if symbols is not None else "ALL"
    print(f"[EMERGENCY] Flattening {scope} positions...")

    allowed = None
    if symbols is not None:
        allowed = set()
        for s in symbols:
            allowed |= _symbol_variants(s)

    try:
        all_positions = api.list_positions()
    except Exception as e:
        print(f"  [EMERGENCY] List positions error: {e}")
        return ['<list_positions failed>']

    targets = [p for p in all_positions
               if allowed is None or getattr(p, 'symbol', None) in allowed]

    # Cancel working orders first (scoped) so shares aren't reserved
    if allowed is None:
        try:
            api.cancel_all_orders()
            time.sleep(1)
        except Exception as e:
            print(f"  [EMERGENCY] Cancel orders error: {e}")
    else:
        for pos in targets:
            cancel_orders_for_symbol(api, pos.symbol, timeout=5)

    failures = []
    for pos in targets:
        sym = pos.symbol
        try:
            qty = float(pos.qty)
            side = 'sell' if qty > 0 else 'buy'
            if tif_for_symbol is not None:
                tif = tif_for_symbol(sym)
            else:
                tif = 'gtc' if ('/' in sym or sym.endswith('USD') and len(sym) > 5) else 'day'
            order = api.submit_order(
                symbol=sym,
                qty=abs(qty),
                side=side,
                type='market',
                time_in_force=tif,
                client_order_id=make_client_order_id('flatten'),
            )
            print(f"  [EMERGENCY] {sym}: Market {side} {pos.qty}")
            # Verify the sell reached a terminal good state
            result = manage_order_lifecycle(api, order.id, timeout=15,
                                            fallback_to_market=False)
            status = getattr(result, 'status', None)
            if status != 'filled':
                print(f"  [EMERGENCY] {sym}: flatten NOT confirmed (status={status})")
                failures.append(sym)
        except Exception as e:
            print(f"  [EMERGENCY] {sym}: {e}")
            failures.append(sym)

    if failures:
        print(f"  [EMERGENCY] UNCONFIRMED flattens: {', '.join(failures)}")
    return failures
