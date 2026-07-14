"""Shared order utilities for limit orders, lifecycle management, and position verification.

Covers the full order lifecycle: quote fetching, limit/market order placement,
fill polling with timeout + market fallback, position verification, circuit
breaker, and emergency flatten.
"""

import time
import math
import datetime
import uuid

from log_config import get_logger

logger = get_logger(__name__)


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
            logger.warning("[QUOTE] %s: degenerate quote bid=%s ask=%s, ignoring",
                           symbol, bid, ask)
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
                if qt.tzinfo is None:
                    # Alpaca timestamps are UTC by definition — a naive value
                    # must not be read as machine-local time (astimezone would),
                    # which skews the age by the UTC offset in either direction.
                    qt = qt.replace(tzinfo=datetime.timezone.utc)
                age = (datetime.datetime.now(datetime.timezone.utc)
                       - qt.astimezone(datetime.timezone.utc)).total_seconds()
                if age > 180:
                    logger.warning("[QUOTE] %s: quote is %.0fs stale, ignoring",
                                   symbol, age)
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
        logger.warning("[QUOTE] Error fetching quote for %s: %s", symbol, e)
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

    Intended to replace uncapped type='market' fallbacks (wave-7; NOT YET
    WIRED — manage_order_lifecycle still submits plain market fallbacks, see
    execution_policy.py): a market order in a thin spec-tech book can fill
    tens-to-hundreds of bps through the quote on the exact bad days slippage
    concentrates. The IOC takes liquidity up to cap_bps past the touch, then
    auto-cancels any unfilled remainder (the caller re-chases for entries, or
    escalates to a true-market backstop for exits/flatten so a stop can never
    silently fail). Returns the order or None on submit failure.
    """
    limit = ioc_limit_price(side, quote_info, cap_bps, asset_type)
    try:
        return api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit',
            limit_price=limit, time_in_force='ioc')
    except Exception as e:
        logger.warning("[IOC] %s %s submit failed (%s)", side, symbol, e)
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
            logger.warning("[MAKER] %s: bid-join error: %s", symbol, e)
            break
        logger.info("[MAKER] %s: joining bid @ $%s ($%.0f, attempt %d)",
                    symbol, bid, remaining, attempt + 1)
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
            logger.warning("[ORDER] %s: invalid limit price %s", symbol, limit_price)
            return None
        qty = math.floor((notional / limit_price) * 1e8) / 1e8  # 8 dp for crypto

        if qty <= 0:
            logger.warning("[ORDER] %s: qty too small (notional=$%s, price=$%s)",
                           symbol, notional, limit_price)
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
        logger.info("[ORDER] %s: %s %s @ $%s (mid=$%.4f, spread=%.3f%%)",
                    symbol, side, qty, limit_price,
                    quote_info['midpoint'], quote_info['spread_pct'])
        return order
    except Exception as e:
        logger.error("[ORDER] %s: %s LIMIT ERROR: %s", symbol, side, e)
        return None


def place_stock_limit_order(api, symbol, side, qty, quote_info,
                            time_in_force='day', offset_bps=5,
                            client_order_id=None):
    """Place a limit order for stocks (integer qty, day TIF)."""
    if qty <= 0:
        logger.warning("[ORDER] %s: qty must be > 0", symbol)
        return None

    try:
        limit_price = compute_limit_price(side, quote_info, offset_bps)
        if limit_price <= 0:
            logger.warning("[ORDER] %s: invalid limit price %s", symbol, limit_price)
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
        logger.info("[ORDER] %s: %s %s @ $%.2f (mid=$%.2f, spread=%.3f%%)",
                    symbol, side, qty, limit_price,
                    quote_info['midpoint'], quote_info['spread_pct'])
        return order
    except Exception as e:
        logger.error("[ORDER] %s: %s LIMIT ERROR: %s", symbol, side, e)
        return None


# --- ORDER LIFECYCLE ---

def manage_order_lifecycle(api, order_id, timeout=30, poll_interval=2,
                           fallback_to_market=True, time_in_force='gtc'):
    """Poll order status. Cancel if unfilled after timeout.
    If fallback_to_market is True, places a market order for the REMAINING
    (unfilled) quantity after cancellation — never the full original qty,
    which would double-buy any partially filled portion.
    Returns the final order object — including a canceled order that carries
    partial fills (callers judge acquired quantity by filled_qty, not by
    status). Returns None only when the order state could never be fetched
    or the market fallback failed to submit.
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
            logger.warning("[LIFECYCLE] Error checking order %s (%dx): %s",
                           order_id, consecutive_errors, e)
            if consecutive_errors >= 3:
                # Don't leave a live working order orphaned — best-effort cancel
                logger.error("[LIFECYCLE] Giving up after %d consecutive errors, canceling order",
                             consecutive_errors)
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
            logger.info("[LIFECYCLE] Order %s FILLED (%s @ $%s)",
                        order_id, order.filled_qty, order.filled_avg_price)
            return order
        elif order.status in ('canceled', 'expired', 'rejected'):
            logger.info("[LIFECYCLE] Order %s terminal status: %s", order_id, order.status)
            return order

    # Timeout reached — cancel
    logger.info("[LIFECYCLE] Order %s unfilled after %ss, canceling...", order_id, timeout)
    try:
        api.cancel_order(order_id)
        time.sleep(1)  # give cancel time to process
    except Exception as e:
        logger.warning("[LIFECYCLE] Cancel error: %s", e)

    # Check final state after cancel (may have filled during the race).
    # Keep the fetched object: a canceled order still carries filled_qty,
    # which callers (maker-ladder remainder math, stop-exit partial-fill
    # checks) need — returning None here would silently discard it.
    final_order = None
    try:
        final_order = api.get_order(order_id)
        if final_order.status == 'filled':
            logger.info("[LIFECYCLE] Order filled during cancel (race condition), keeping.")
            return final_order
        try:
            saved_filled = float(final_order.filled_qty or 0)
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
            logger.info("[LIFECYCLE] Fully filled during cancel (%s), no fallback needed.",
                        saved_filled)
            try:
                return api.get_order(order_id)
            except Exception:
                return None
        logger.info("[LIFECYCLE] Falling back to market order (%s remaining)...",
                    remaining or saved_qty)
        try:
            market_order = api.submit_order(
                symbol=saved_symbol,
                qty=remaining if remaining is not None else saved_qty,
                side=saved_side,
                type='market',
                time_in_force=time_in_force,
                client_order_id=make_client_order_id('mktfb'),
            )
            logger.info("[LIFECYCLE] Market fallback submitted: %s", market_order.id)
            # Poll briefly for a fill; return the FRESHEST fetched state, not
            # the submit-time snapshot (status='accepted', filled_qty=0) — a
            # slow/partial market fill would otherwise look unacquired to the
            # caller and the position would go untracked and unprotected.
            latest = market_order
            for _ in range(3):
                time.sleep(1)
                try:
                    latest = api.get_order(market_order.id)
                    if latest.status == 'filled':
                        logger.info("[LIFECYCLE] Market fallback FILLED (%s @ $%s)",
                                    latest.filled_qty, latest.filled_avg_price)
                        return latest
                except Exception:
                    pass
            return latest
        except Exception as e:
            logger.error("[LIFECYCLE] Market fallback error: %s", e)
            return None

    return final_order


# --- POSITION VERIFICATION ---

def verify_position(api, symbol):
    """Check actual position via API. Returns the position object, or None
    when there is no LONG position — qty <= 0 (flat or short inventory) is
    treated as no position; the live book is long-only, so short wiring must
    NOT reuse this check as-is.
    Handles Alpaca's crypto symbol format (BTC/USD -> BTCUSD).
    """
    for sym in _symbol_variants(symbol):
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
        logger.warning("[POSITIONS] Error listing positions: %s", e)
        return None


# --- TRADE GATING ---

def should_trade(predicted_return, spread_pct, min_edge=None,
                 asset_type='crypto', maker=False):
    """Only trade if predicted return clears min_edge x the FULL round-trip
    cost: venue fees (crypto: 25 bps taker / 15 bps maker per side) plus
    spread. The old spread-only hurdle ignored the dominant crypto cost.

    predicted_return: expected % move (e.g. 0.5 means +0.5%)
    spread_pct: current spread as % of price
    min_edge: multiplier — predicted return must exceed this x cost to trade.
        None (the default used by both live entry gates) resolves to
        fees.MIN_EDGE_MULTIPLE so tuning the canonical multiple can never
        silently drift the live gate from the backtest/training cost model.
    asset_type: 'crypto' or 'stock' (fee schedule differs ~10x)
    maker: True when entries rest as maker limit orders
    """
    from fees import MIN_EDGE_MULTIPLE, required_edge_pct
    if min_edge is None:
        min_edge = MIN_EDGE_MULTIPLE
    # live=True: the LIVE gate blends the crypto entry fee by REALIZED
    # maker share (journals) so an overstated taker assumption doesn't
    # reject genuinely positive-edge entries. Backtest/training paths
    # call fees directly with the conservative static model.
    threshold = required_edge_pct(asset_type, spread_pct, maker, min_edge,
                                  live=True)
    return abs(predicted_return) > threshold


# --- CLEANUP ---

def _list_open_orders(api):
    """All open orders with an explicit high limit. Without `limit` both SDKs
    fall back to the server default page of 50 — with >50 open orders (both
    books' resting stops + working entries + bracket legs) the cancel/cleanup
    paths would silently miss the rest."""
    try:
        return api.list_orders(status='open', limit=500)
    except TypeError:
        # Shims/fakes without a `limit` kwarg (both real SDKs accept it)
        return api.list_orders(status='open')


def cancel_all_open_orders(api, symbols=None):
    """Cancel open orders. Call on startup to clean stale state.

    Args:
        symbols: if given, cancel ONLY orders for these symbols (this bot's
            universe). Both bots share one account — an account-wide
            cancel_all_orders() from one bot strips the other bot's
            protective bracket/stop legs.
    """
    try:
        orders = _list_open_orders(api)
        if not orders:
            logger.info("[CLEANUP] No open orders to cancel.")
            return
        if symbols is None:
            logger.info("[CLEANUP] Canceling %d open order(s) (account-wide)...",
                        len(orders))
            api.cancel_all_orders()
            time.sleep(1)
            return
        allowed = set()
        for s in symbols:
            allowed |= _symbol_variants(s)
        mine = [o for o in orders if getattr(o, 'symbol', None) in allowed]
        logger.info("[CLEANUP] Canceling %d/%d open order(s) in this bot's universe...",
                    len(mine), len(orders))
        for o in mine:
            try:
                api.cancel_order(o.id)
            except Exception as e:
                logger.warning("[CLEANUP] Cancel %s %s: %s", o.symbol, o.id, e)
        if mine:
            time.sleep(1)
    except Exception as e:
        logger.warning("[CLEANUP] Error canceling orders: %s", e)


def cancel_orders_for_symbol(api, symbol, timeout=5, poll_interval=0.5):
    """Cancel all open orders for one symbol and WAIT until they are gone.

    Alpaca cancellation is async (`pending_cancel`); selling shares that are
    still reserved by a live stop/take-profit leg rejects with
    'insufficient qty'. Returns True when no open orders remain for the
    symbol, False if any survive past the timeout.
    """
    variants = _symbol_variants(symbol)
    try:
        open_orders = [o for o in _list_open_orders(api)
                       if getattr(o, 'symbol', None) in variants]
    except Exception as e:
        logger.warning("[CANCEL] %s: list_orders error: %s", symbol, e)
        return False
    if not open_orders:
        return True

    for o in open_orders:
        try:
            api.cancel_order(o.id)
        except Exception as e:
            # Already terminal is fine; anything else we'll catch on re-poll
            logger.warning("[CANCEL] %s: cancel %s: %s", symbol, o.id, e)

    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            remaining = [o for o in _list_open_orders(api)
                         if getattr(o, 'symbol', None) in variants]
            if not remaining:
                return True
        except Exception:
            pass
    logger.warning("[CANCEL] %s: open orders still pending after %ss", symbol, timeout)
    return False


# --- POSITION RECONSTRUCTION ---

def reconstruct_positions(api, symbols, asset_type='crypto'):
    """Rebuild position dict from Alpaca API (survive restarts).

    asset_type is accepted for caller compatibility but unused — both books
    reconstruct identically (base_loop restores stop/trailing state from its
    own persisted files, not from here).
    Returns: {symbol: {qty, entry_price, high_water_mark}} — long (qty > 0)
    positions only.
    """
    positions = {}
    for sym in symbols:
        for candidate in _symbol_variants(sym):
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
        logger.warning("[CIRCUIT BREAKER] API error: %s", e)
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
    logger.warning("[EMERGENCY] Flattening %s positions...", scope)

    allowed = None
    if symbols is not None:
        allowed = set()
        for s in symbols:
            allowed |= _symbol_variants(s)

    try:
        all_positions = api.list_positions()
    except Exception as e:
        logger.error("[EMERGENCY] List positions error: %s", e)
        return ['<list_positions failed>']

    targets = [p for p in all_positions
               if allowed is None or getattr(p, 'symbol', None) in allowed]

    # Cancel working orders first (scoped) so shares aren't reserved
    if allowed is None:
        try:
            api.cancel_all_orders()
            time.sleep(1)
        except Exception as e:
            logger.warning("[EMERGENCY] Cancel orders error: %s", e)
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
            logger.warning("[EMERGENCY] %s: Market %s %s", sym, side, pos.qty)
            # Verify the sell reached a terminal good state
            result = manage_order_lifecycle(api, order.id, timeout=15,
                                            fallback_to_market=False)
            status = getattr(result, 'status', None)
            if status != 'filled':
                logger.error("[EMERGENCY] %s: flatten NOT confirmed (status=%s)",
                             sym, status)
                failures.append(sym)
        except Exception as e:
            logger.error("[EMERGENCY] %s: %s", sym, e)
            failures.append(sym)

    if failures:
        logger.error("[EMERGENCY] UNCONFIRMED flattens: %s", ', '.join(failures))
    return failures
