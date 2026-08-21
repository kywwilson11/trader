"""Shared order utilities for limit orders, lifecycle management, and position verification.

Covers the full order lifecycle: quote fetching, limit/market order placement,
fill polling with timeout + market fallback, position verification, circuit
breaker, and emergency flatten.
"""

import os
import time
import math
import datetime
import hashlib
import uuid

from log_config import get_logger

logger = get_logger(__name__)

# c26 T6 flags (module-level env-var booleans, the funding.py/shadow.py
# pattern; tests toggle the module attribute). Both default OFF with
# flag-OFF behavior byte-identical (pinned by tests/test_c26_T6.py).
# IOC_ENTRY_CAP: entry-order market fallbacks become slippage-capped
# marketable IOCs (D21); exits/flatten are NEVER capped.
IOC_ENTRY_CAP_ENABLED = os.environ.get(
    'TRADER_IOC_ENTRY_CAP', '0').strip().lower() in ('1', 'true', 'yes')
# MAKER_SHARE_NOTIONAL: should_trade's live crypto threshold uses the
# notional-weighted maker share instead of fees' count-based one.
MAKER_SHARE_NOTIONAL_ENABLED = os.environ.get(
    'TRADER_MAKER_SHARE_NOTIONAL', '0').strip().lower() in ('1', 'true', 'yes')

# Capped-IOC entry fallback: brief fill poll (mirrors the market-fallback
# poll inside manage_order_lifecycle).
IOC_ENTRY_POLL_S = 1
IOC_ENTRY_POLLS = 3


def make_client_order_id(tag: str) -> str:
    """Generate a unique client_order_id (Alpaca cap: 48 chars).

    A fresh uuid4 per call: the tag prefix makes this bot's orders
    identifiable in logs and order history, but the id is NOT an idempotency
    key — a retry mints a NEW id, so the broker's duplicate-client_order_id
    rejection can never dedup a resubmission. Submit paths must not
    blind-retry on ambiguous failures on the strength of this tag.
    """
    return f"{tag[:14]}-{uuid.uuid4().hex[:20]}"


def _symbol_variants(symbol: str) -> set[str]:
    """All spellings Alpaca may use for a symbol ('BTC/USD' <-> 'BTCUSD').

    Genuinely bidirectional: list_positions reports crypto slashless
    ('BTCUSD') while orders are submitted with the universe spelling
    ('BTC/USD') — emergency_flatten feeds BROKER symbols in here, so the
    reverse expansion is what lets its pre-flatten cancel find the resting
    stop. The slashless->slashed heuristic (endswith USD, len > 5) is the
    same one the flatten TIF branch already trusts; no US stock ticker
    longer than 5 chars ends in 'USD', so equities stay single-variant.
    Returns a set — callers fold it with `allowed |= _symbol_variants(s)`.
    """
    variants = {symbol}
    if '/' in symbol:
        variants.add(symbol.replace('/', ''))
    elif symbol.endswith('USD') and len(symbol) > 5:
        variants.add(symbol[:-3] + '/USD')
    return variants


_NOT_FOUND_SIGS = ('position does not exist', 'not found', '404', 'no position')


def _is_not_found(exc) -> bool:
    """True when an exception reads like Alpaca's 'no such position' (both
    SDKs), as opposed to a transient API failure (429/timeout/5xx).
    Mirrors the classifier stock_loop's flatten path already uses."""
    msg = str(exc).lower()
    return any(s in msg for s in _NOT_FOUND_SIGS)


def _filled_qty(order) -> float:
    """Best-effort filled quantity of an order object (0.0 for None/absent/unparseable)."""
    try:
        return float(getattr(order, 'filled_qty', 0) or 0)
    except (TypeError, ValueError):
        return 0.0


REST_CONNECT_TIMEOUT_S = 10.0
REST_READ_TIMEOUT_S = 30.0


def install_session_timeout(session, connect_s=REST_CONNECT_TIMEOUT_S,
                            read_s=REST_READ_TIMEOUT_S) -> bool:
    """Wrap a requests.Session-like object so every .request() carries a
    default (connect, read) timeout unless the caller passed one (c26
    D01: neither Alpaca SDK sets one, so a half-open socket blocks a
    prediction worker forever). Idempotent. Returns True when installed,
    False when the object has no callable .request or is already wrapped.
    Fail-open: never raises."""
    orig = getattr(session, 'request', None)
    if not callable(orig) or getattr(session, '_trader_timeout_wrapped', False):
        return False
    def _timed_request(*args, **kwargs):
        kwargs.setdefault('timeout', (connect_s, read_s))
        return orig(*args, **kwargs)
    try:
        session.request = _timed_request
        session._trader_timeout_wrapped = True
    except Exception:
        return False
    return True


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
        if (not (math.isfinite(bid) and math.isfinite(ask))
                or midpoint <= 0 or bid <= 0 or ask <= 0):
            # Degenerate quote (halted/stale feed/NaN). NaN compares False to
            # everything, so without the finiteness check a NaN quote sailed
            # through and surfaced as spread_pct=0.0 — the loosest possible
            # cost-gate input — while freezing stop comparisons downstream.
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
        spread_pct = (spread / midpoint) * 100.0
        if ask < bid:
            logger.warning("[QUOTE] %s: CROSSED quote bid=%s ask=%s "
                           "(spread_pct=%.4f)", symbol, bid, ask, spread_pct)
        # fetched_ts (c26 T6): decision-time stamp for slippage-vs-quote-age
        # decomposition. Additive sixth key — consumers key-access only the
        # five legacy keys; buy rows keep their own later '_fetched_ts'
        # stamp set in _execute_buys.
        return {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'midpoint': midpoint,
            'spread_pct': spread_pct,
            'fetched_ts': time.time(),
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
    try:
        limit = ioc_limit_price(side, quote_info, cap_bps, asset_type)
        return api.submit_order(
            symbol=symbol, qty=qty, side=side, type='limit',
            limit_price=limit, time_in_force='ioc')
    except Exception as e:
        logger.warning("[IOC] %s %s submit failed (%s)", side, symbol, e)
        return None


def _entry_ioc_cap_bps(asset_type):
    """Cap for a capped-IOC ENTRY fallback (c26 T6 / D21).

    The name_class seed table is unbuilt (execution_policy.py), so stocks
    take the 'mid' cap and crypto the widest ('spec') — Alpaca crypto books
    run wide. Lazy strategy_config import (READ-ONLY, the same way every
    module consumes it); the literal fallback keeps this module import-safe.
    """
    try:
        from strategy_config import IOC_CAP_BPS
        return (IOC_CAP_BPS.get('spec', 40) if asset_type == 'crypto'
                else IOC_CAP_BPS.get('mid', 20))
    except Exception:
        return 40 if asset_type == 'crypto' else 20


def _ioc_entry_fallback(api, symbol, side, qty, ioc_fallback, final_order):
    """Capped-IOC replacement for the uncapped market ENTRY fallback (D21).

    Only reached flag-ON (order_utils.IOC_ENTRY_CAP_ENABLED) from
    manage_order_lifecycle when the caller passed an ioc_fallback context —
    entries only; exits/flatten never pass it. No quote => NO chase
    (entries fail closed): never submit an uncapped market entry here.
    """
    try:
        quote = ioc_fallback['quote_fn']()
    except Exception:
        quote = None
    if quote is None:
        logger.warning("[IOC] %s: no quote for capped entry fallback — "
                       "remainder NOT chased (entries fail closed)", symbol)
        return final_order
    asset_type = ioc_fallback.get('asset_type', 'crypto')
    cap = ioc_fallback.get('cap_bps') or _entry_ioc_cap_bps(asset_type)
    if asset_type == 'stock':
        # qty can arrive as the broker's raw string (saved_qty fallback when
        # the remaining computation failed) — int('5.0') raises, and an
        # exception here would escape into the live entry path. Unparseable
        # ⇒ no chase (entries fail closed), same as the no-quote branch.
        try:
            qty = int(float(qty))
        except (TypeError, ValueError):
            return final_order
        if qty <= 0:
            return final_order
    order = place_marketable_ioc(api, symbol, side, qty, quote, cap,
                                 asset_type)
    if order is None:
        return final_order
    # Brief fill poll — return the FRESHEST fetched state (mirror the
    # market-fallback poll in manage_order_lifecycle). On total fetch
    # failure return the submit-time object.
    latest = order
    for _ in range(IOC_ENTRY_POLLS):
        time.sleep(IOC_ENTRY_POLL_S)
        try:
            latest = api.get_order(order.id)
            if getattr(latest, 'status', None) == 'filled':
                break
        except Exception:
            pass
    # Measurement of the new path (only fires flag-ON); best-effort.
    try:
        from trade_journal import log_decision
        log_decision({'action': 'ioc_entry_fallback', 'symbol': symbol,
                      'side': side, 'cap_bps': cap,
                      'limit_price': ioc_limit_price(side, quote, cap,
                                                     asset_type),
                      'filled_qty': _filled_qty(latest),
                      'status': getattr(latest, 'status', None)})
    except Exception:
        pass
    return latest


# --- ORDER PLACEMENT ---

def _maker_rung_id(symbol: str, ladder_ts: int, rung: int) -> str:
    """Deterministic client_order_id for one maker rung (c26 D18): a
    resend of the same rung within one ladder invocation collides at
    the broker (duplicate-client_order_id rejection) instead of
    silently doubling exposure. Distinct rungs/timestamps still get
    distinct ids (this is NOT make_client_order_id, which is
    deliberately non-idempotent)."""
    h = hashlib.sha256(f"{symbol}|{ladder_ts}|{rung}".encode()).hexdigest()[:20]
    return f"maker-{h}"


def _journal_entry_fills(symbol, tactic, maker_notional, taker_notional):
    """One 'entry_fills' measurement row per place_maker_buy invocation
    (c26 T6): the maker/taker NOTIONAL split behind the pinned count-based
    share in fees.realized_crypto_maker_share. Key name is 'tactic' NOT
    'entry_tactic', and action != 'buy' — fees' count-scan prefilter
    ('"entry_tactic"' substring + action=='buy') must never see these rows
    as entries. Best-effort; never raises.

    Suppressed under pytest (PYTEST_CURRENT_TEST): pre-existing order-path
    tests exercise place_maker_buy against fakes WITHOUT stubbing
    trade_journal, and a suite run on the Jetson must not seed the LIVE
    journal with synthetic rows that realized_crypto_maker_share_notional
    would count toward arming TRADER_MAKER_SHARE_NOTIONAL. Tests that
    assert on real rows delenv the marker (tests/test_c26_T6.py)."""
    if 'PYTEST_CURRENT_TEST' in os.environ:
        return
    try:
        from trade_journal import log_decision
        log_decision({'action': 'entry_fills', 'symbol': symbol,
                      'tactic': tactic,
                      'maker_notional': round(maker_notional, 2),
                      'taker_notional': round(taker_notional, 2)})
    except Exception:
        pass


def place_maker_buy(api, symbol, notional, quote_fn, stage_timeout=25,
                    max_reprices=1):
    """Crypto entry ladder: join the bid (maker fee + half-spread saved),
    reprice once at the fresh bid, then fall back to a spread-aware limit
    that escalates to market (via manage_order_lifecycle) for any remainder.

    Alpaca crypto charges 15bps maker vs 25bps taker per side; a bid-join
    that fills passively also avoids paying the half-spread. With a
    12-48h signal horizon, risking ~1 minute of passive waiting for
    10bps+spread is positive expectancy. Safe to wait now that entries
    get a resting server-side stop the moment they're tracked.

    quote_fn: zero-arg callable returning a fresh quote dict (or None) —
    each rung reprices off live data, not the stale decision quote.

    Returns (final_order_or_None, tactic) where tactic is one of
    'maker', 'maker_reprice', 'maker_partial', 'maker_unknown',
    'taker_fallback', 'unfilled'. 'maker_unknown' means a rung's outcome
    could not be determined — the ladder aborts and does NOT escalate
    (normally the remainder escalates to the taker fallback).
    The caller must judge success by acquired quantity
    (filled_qty / live position), NOT by the last order's status — a
    95%-filled maker rung that times out is a success.
    The returned object is the best acquisition EVIDENCE observed:
    whichever order carries the largest filled_qty — a later zero-fill
    rung or a failed fallback never erases an earlier rung's partial fill.
    """
    remaining = float(notional)
    last = None
    maker_notional = 0.0   # c26 T6: notional-weighted fill attribution
    taker_notional = 0.0
    ladder_ts = int(time.time())
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
                client_order_id=_maker_rung_id(symbol, ladder_ts, attempt))
        except Exception as e:
            logger.warning("[MAKER] %s: bid-join error: %s", symbol, e)
            break
        logger.info("[MAKER] %s: joining bid @ $%s ($%.0f, attempt %d)",
                    symbol, bid, remaining, attempt + 1)
        result = manage_order_lifecycle(api, order.id, timeout=stage_timeout,
                                        fallback_to_market=False)
        if result is None:
            # UNKNOWN outcome (c26 D18): the rung may still be a live
            # working GTC bid. Unknown is NOT zero-fill — re-sending the
            # remainder (or escalating to the taker fallback) can multiply
            # intended exposure precisely during API instability. Abort the
            # ladder and skip the fallback; the caller judges acquisition by
            # the best evidence returned; a restart reconstructs anything
            # the working bid buys later.
            logger.error("[MAKER] %s: rung %d outcome UNKNOWN (lifecycle "
                         "returned None) — aborting ladder, skipping taker "
                         "fallback", symbol, attempt + 1)
            _journal_entry_fills(symbol, 'maker_unknown',
                                 maker_notional, taker_notional)
            return last, 'maker_unknown'
        if result is not None and getattr(result, 'status', None) == 'filled':
            logger.info("[MAKER] %s: rung %d filled @ %s (posted bid %s)",
                        symbol, attempt + 1,
                        getattr(result, 'filled_avg_price', None), bid)
            maker_notional += _filled_qty(result) * (
                float(getattr(result, 'filled_avg_price', 0) or 0) or bid)
            tactic = 'maker' if attempt == 0 else 'maker_reprice'
            _journal_entry_fills(symbol, tactic,
                                 maker_notional, taker_notional)
            return result, tactic
        # Chase only the unfilled remainder on the next rung
        try:
            filled_qty = float(getattr(result, 'filled_qty', 0) or 0)
            px = float(getattr(result, 'filled_avg_price', 0) or 0) or bid
            remaining -= filled_qty * px
            maker_notional += filled_qty * px
        except (TypeError, ValueError):
            pass
        if result is not None and _filled_qty(result) >= _filled_qty(last):
            last = result
        if remaining < 10:  # dust left — maker rungs covered the entry
            _journal_entry_fills(symbol, 'maker_partial',
                                 maker_notional, taker_notional)
            return last, 'maker_partial'

    # Fallback for whatever's left: a spread-aware limit that escalates to
    # market via manage_order_lifecycle's own timeout (above a 0.1% spread
    # the initial limit posts INSIDE the quote — see compute_limit_price).
    # D21 (c26 T6): the wide-spread inside-quote limit is deliberately KEPT
    # as the passive first leg; under TRADER_IOC_ENTRY_CAP its escalation
    # is a capped IOC instead of an uncapped market order — that is the D21
    # resolution of the wide-spread branch. compute_limit_price itself is
    # NOT touched (flag-OFF pricing stays byte-identical).
    quote = quote_fn()
    if quote is None:
        _journal_entry_fills(symbol, 'unfilled', maker_notional, taker_notional)
        return last, 'unfilled'
    order = place_limit_order(api, symbol, 'buy', remaining, quote)
    if order is None:
        _journal_entry_fills(symbol, 'unfilled', maker_notional, taker_notional)
        return last, 'unfilled'
    result = manage_order_lifecycle(
        api, order.id, timeout=stage_timeout + 5, fallback_to_market=True,
        ioc_fallback={'quote_fn': quote_fn, 'cap_bps': None,
                      'asset_type': 'crypto'})
    # The fallback is a NEW order, so its whole fill is attributed here.
    # Conservative: the fallback's own passive limit fill is classed taker
    # (matches the pinned count-share bias in fees).
    try:
        fb_px = (float(getattr(result, 'filled_avg_price', 0) or 0)
                 or float(quote.get('midpoint') or 0))
        taker_notional += max(0.0, _filled_qty(result)) * max(0.0, fb_px)
    except (TypeError, ValueError):
        pass
    _journal_entry_fills(symbol, 'taker_fallback',
                         maker_notional, taker_notional)
    # Never hand back LESS acquisition evidence than a maker rung already
    # proved: base_loop judges `acquired` by the returned object's
    # filled_qty (then verifies the true total against the broker), so a
    # None/zero-fill fallback result would leave real coins untracked,
    # unstopped and unalerted.
    if _filled_qty(result) >= _filled_qty(last):
        return result, 'taker_fallback'
    logger.warning("[MAKER] %s: fallback under-reports acquisition "
                   "(fallback filled=%s < rung filled=%s) — returning the "
                   "rung order as acquisition evidence", symbol,
                   _filled_qty(result), _filled_qty(last))
    return last, 'taker_fallback'


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
                    quote_info.get('midpoint', float('nan')),
                    quote_info.get('spread_pct', float('nan')))
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
                    quote_info.get('midpoint', float('nan')),
                    quote_info.get('spread_pct', float('nan')))
        return order
    except Exception as e:
        logger.error("[ORDER] %s: %s LIMIT ERROR: %s", symbol, side, e)
        return None


# --- ORDER LIFECYCLE ---

def manage_order_lifecycle(api, order_id, timeout=30, poll_interval=2,
                           fallback_to_market=True, time_in_force='gtc',
                           cancel_on_timeout=True, ioc_fallback=None):
    """Poll order status. Cancel if unfilled after timeout.
    If fallback_to_market is True, places a market order for the REMAINING
    (unfilled) quantity after cancellation — never the full original qty,
    which would double-buy any partially filled portion.
    Returns the final order object — including a canceled order that carries
    partial fills (callers judge acquired quantity by filled_qty, not by
    status). Returns None only when the order state could never be fetched
    or the market fallback failed to submit.

    cancel_on_timeout=False is CONFIRM-ONLY mode for liquidation orders
    (emergency flatten, stop exits): the function never cancels the order —
    not at timeout and not on the 3-consecutive-error give-up — and never
    submits a market fallback (an uncanceled order plus a fallback would
    double-sell). At timeout it returns the freshest fetched order state
    (possibly still working), or None when the state could never be
    fetched. Default True is byte-identical to the previous behavior.

    ioc_fallback (c26 T6 / D21): entry-only capped-IOC fallback context
    {'quote_fn', 'cap_bps', 'asset_type'} — honored only when the module
    flag IOC_ENTRY_CAP_ENABLED is on AND fallback_to_market; exits/flatten
    never pass it (a liquidation is never capped). Default None (or flag
    OFF) is byte-identical: the plain market fallback below runs.
    """
    # Save order params upfront for market fallback (in case later get_order fails)
    saved_symbol = saved_qty = saved_side = None
    saved_filled = 0.0
    consecutive_errors = 0
    elapsed = 0
    # One pre-loop import: the stream cache is consulted every tick, but the
    # import machinery must not re-run once per tick (hot path on the Jetson).
    try:
        from order_stream import get_order_state, TERMINAL
    except Exception:
        get_order_state, TERMINAL = None, ()
    stream_skip_used = False
    while elapsed < timeout:
        # If the trade_updates stream already saw a terminal state, skip the
        # wait ONCE for one authoritative REST fetch instead of N polls.
        # Only once: if REST lags the stream (or raises), every later
        # iteration must pace normally — otherwise the loop degrades into an
        # unthrottled REST burst and the 3-error give-up window collapses
        # from seconds to milliseconds.
        streamed = None
        if get_order_state is not None and not stream_skip_used:
            try:
                streamed = get_order_state(order_id)
            except Exception:
                streamed = None
        if streamed is not None and streamed.get('status') in TERMINAL:
            stream_skip_used = True
        else:
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
                if cancel_on_timeout:
                    logger.error("[LIFECYCLE] Giving up after %d consecutive errors, canceling order",
                                 consecutive_errors)
                    try:
                        api.cancel_order(order_id)
                    except Exception as ce:
                        logger.warning("[LIFECYCLE] Best-effort cancel of %s failed:"
                                       " %s — order may still be working",
                                       order_id, ce)
                else:
                    logger.error("[LIFECYCLE] %s: confirm-only mode — leaving"
                                 " order working after %d fetch errors",
                                 order_id, consecutive_errors)
                if saved_filled > 0:
                    logger.error("[LIFECYCLE] %s: giving up with filled_qty=%s"
                                 " already observed — position may be"
                                 " untracked/unprotected until restart",
                                 order_id, saved_filled)
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
    if cancel_on_timeout:
        logger.info("[LIFECYCLE] Order %s unfilled after %ss, canceling...", order_id, timeout)
        try:
            api.cancel_order(order_id)
            time.sleep(1)  # give cancel time to process
        except Exception as e:
            logger.warning("[LIFECYCLE] Cancel error: %s", e)
    else:
        logger.info("[LIFECYCLE] Order %s unfilled after %ss — confirm-only"
                    " mode, leaving it working", order_id, timeout)

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

    if fallback_to_market and saved_symbol and cancel_on_timeout:
        # Only chase the unfilled remainder
        try:
            remaining = float(saved_qty) - saved_filled
            # Match the module's qty convention (8dp floor, never up): a raw
            # float subtraction can carry 16+ decimals into submit_order and
            # a precision rejection here would turn a real partial fill into
            # a None ('nothing acquired') for the caller.
            remaining = math.floor(remaining * 1e8) / 1e8
        except (TypeError, ValueError):
            remaining = None
        if remaining is not None and remaining <= 0:
            logger.info("[LIFECYCLE] Fully filled during cancel (%s), no fallback needed.",
                        saved_filled)
            try:
                return api.get_order(order_id)
            except Exception:
                # We may already hold the post-cancel state (final_order from the
                # fetch above) — a failed redundant re-fetch must not discard it;
                # None only when the state could never be fetched (docstring).
                return final_order
        # D21 (flag-gated; reads the module global at call time so tests
        # can monkeypatch): capped-IOC entry fallback instead of the
        # uncapped market chase below. Flag OFF (or no context passed):
        # the plain market fallback runs — byte-identical.
        if ioc_fallback is not None and IOC_ENTRY_CAP_ENABLED:
            return _ioc_entry_fallback(
                api, saved_symbol, saved_side,
                remaining if remaining is not None else saved_qty,
                ioc_fallback, final_order)
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
            logger.error("[LIFECYCLE] Market fallback error: %s (order %s,"
                         " filled_qty=%s on the canceled limit)",
                         e, order_id, saved_filled)
            return None

    if final_order is None and saved_filled > 0:
        logger.error("[LIFECYCLE] %s: post-cancel state unknown with"
                     " filled_qty=%s observed in-loop — returning None;"
                     " position may be untracked/unprotected until restart",
                     order_id, saved_filled)
    return final_order


# --- POSITION VERIFICATION ---

def verify_position(api, symbol):
    """Check actual position via API. Returns the position object, or None
    when there is no LONG position — qty <= 0 (flat or short inventory) is
    treated as no position; the live book is long-only, so short wiring must
    NOT reuse this check as-is.
    Handles Alpaca's crypto symbol format (BTC/USD -> BTCUSD).
    """
    errs = []
    for sym in _symbol_variants(symbol):
        try:
            pos = api.get_position(sym)
            qty = float(pos.qty)
            if qty > 0:
                return pos
        except Exception as e:
            if not _is_not_found(e):
                errs.append(f"{sym}: {e}")
            continue
    if errs:
        # Still returns None (callers treat that as 'really gone' — see
        # ledger P1), but a transient blip that is about to drop tracking
        # must at least be visible in the log. Emitted only when NO variant
        # found the position: a variant that failed while another variant
        # succeeded is not a drop, and logging it mid-loop asserted the
        # opposite of what happened.
        logger.warning("[VERIFY] %s: get_position failed (%s) — "
                       "treated as no position", symbol, '; '.join(errs))
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

# realized_crypto_maker_share_notional cache: (mono_ts, share) or None.
# TTL mirrors fees._MAKER_SHARE_TTL (local constant — fees' is private).
_maker_share_notional_cache = None
_MAKER_NOTIONAL_TTL = 3600


def realized_crypto_maker_share_notional():
    """NOTIONAL-weighted crypto maker share from 'entry_fills' journal rows
    (c26 T6). The count-based fees.realized_crypto_maker_share treats a
    95%-maker-filled entry that fell back for dust as full taker; this
    weights by dollars actually filled. Window/min-sample constants come
    from fees at call time (one source of truth). None until at least
    MAKER_SHARE_MIN_ENTRIES rows exist in the window — thin samples must
    not move the live gate. Cached 1h.

    Failure semantics mirror fees' contract: malformed rows are skipped per
    row, but ANY exception aborts the whole scan to None (all-or-nothing
    fail-closed — a partially-readable window must not move the live gate).
    """
    global _maker_share_notional_cache
    import time as _t
    now = _t.monotonic()
    hit = _maker_share_notional_cache
    if hit is not None and (now - hit[0]) < _MAKER_NOTIONAL_TTL:
        return hit[1]
    share = None
    try:
        from fees import MAKER_SHARE_WINDOW_DAYS, MAKER_SHARE_MIN_ENTRIES
        from trade_journal import JOURNAL_DIR
        import json as _json
        n = 0
        maker_sum = 0.0
        taker_sum = 0.0
        today = datetime.date.today()
        for d in range(MAKER_SHARE_WINDOW_DAYS + 1):
            path = (JOURNAL_DIR
                    / f"{(today - datetime.timedelta(days=d)).isoformat()}.jsonl")
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    if '"entry_fills"' not in line:
                        continue
                    try:
                        e = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if not isinstance(e, dict):
                        continue
                    sym = e.get('symbol', '')
                    if (e.get('action') != 'entry_fills'
                            or not isinstance(sym, str) or '/' not in sym):
                        continue
                    try:
                        m = float(e.get('maker_notional', 0) or 0)
                        t = float(e.get('taker_notional', 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    n += 1
                    maker_sum += m
                    taker_sum += t
        total = maker_sum + taker_sum
        if n >= MAKER_SHARE_MIN_ENTRIES and total > 0:
            share = maker_sum / total
    except Exception as exc:
        logger.warning("[MAKER-SHARE] notional journal scan failed (%s: %s)"
                       " — notional share unavailable",
                       type(exc).__name__, exc)
        share = None
    _maker_share_notional_cache = (now, share)
    return share


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
    NOTE: compares abs(predicted_return) — direction-blind. In the
    long-only live books this gate is only meaningful AFTER a pred > 0
    check; the two loops currently order the two checks differently,
    which skews cost_floor skip attribution in decision_report.
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
    # c26 T6 (flag-gated, default OFF => byte-identical): substitute the
    # NOTIONAL-weighted maker share for fees' count-based one in the live
    # entry fee only, via the round_trip linearity contract (fees.py
    # round_trip_cost_pct docstring): threshold moves by exactly
    # min_edge * (fee_notional - fee_count)/100. All fee numbers come from
    # fees — no duplicated schedule; the clamped blend keeps the delta
    # inside [maker, taker]. HANDOFF (recorded, fees.py out of this
    # packet's scope): the cleaner long-term home is a share= parameter on
    # fees.crypto_entry_fee_bps, collapsing this delta arithmetic.
    if MAKER_SHARE_NOTIONAL_ENABLED and asset_type == 'crypto' and not maker:
        try:
            sh = realized_crypto_maker_share_notional()
            if sh is not None:
                from fees import (CRYPTO_MAKER_BPS, CRYPTO_TAKER_BPS,
                                  crypto_entry_fee_bps)
                sh = min(max(sh, 0.0), 1.0)
                fee_n = CRYPTO_MAKER_BPS * sh + CRYPTO_TAKER_BPS * (1.0 - sh)
                threshold += min_edge * (fee_n
                                         - crypto_entry_fee_bps(live=True)) / 100.0
        except Exception:
            pass
    return abs(predicted_return) > threshold


# --- CLEANUP ---

def _list_open_orders(api, symbols=None):
    """All open orders with an explicit high limit. Without `limit` both SDKs
    fall back to the server default page of 50 — with >50 open orders (both
    books' resting stops + working entries + bracket legs) the cancel/cleanup
    paths would silently miss the rest.

    symbols: optional server-side narrowing (both SDKs accept the kwarg;
    alpaca_compat passes it through). The caller's client-side filter stays
    authoritative — on ANY failure of the filtered call this falls back to
    the unfiltered listing, so narrowing can never hide orders."""
    orders = None
    if symbols:
        try:
            orders = api.list_orders(status='open', limit=500,
                                     symbols=list(symbols))
        except TypeError:
            orders = None  # shim/fake without the `symbols` kwarg
        except Exception as e:
            logger.warning("[CLEANUP] symbol-filtered list_orders failed (%s)"
                           " — retrying unfiltered", e)
            orders = None
    if orders is None:
        try:
            orders = api.list_orders(status='open', limit=500)
        except TypeError:
            # Shims/fakes without a `limit` kwarg (both real SDKs accept it)
            orders = api.list_orders(status='open')
    if orders is not None and len(orders) >= 500:
        logger.warning("[CLEANUP] open-order listing hit the 500-order page"
                       " cap — some open orders may not have been seen")
    return orders


def _await_orders_clear(api, variants, timeout, poll_interval):
    """Poll until no open order for `variants` remains (True) or the timeout
    expires (False). Listing errors during the wait are retried, not fatal —
    identical semantics to the wait loop this was extracted from."""
    symbols_hint = sorted(variants)
    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval
        try:
            remaining = [o for o in _list_open_orders(api, symbols=symbols_hint)
                         if getattr(o, 'symbol', None) in variants]
            if not remaining:
                return True
        except Exception:
            pass
    return False


def cancel_all_open_orders(api, symbols=None, timeout=5):
    """Cancel open orders. Call on startup to clean stale state.

    Args:
        symbols: if given, cancel ONLY orders for these symbols (this bot's
            universe). Both bots share one account — an account-wide
            cancel_all_orders() from one bot strips the other bot's
            protective bracket/stop legs.
        timeout: scoped path only — bounded wait for the cancels to actually
            clear (Alpaca cancellation is async; a replacement stop submitted
            while the old one is pending_cancel rejects with insufficient qty).
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
        if mine and not _await_orders_clear(api, allowed, timeout, 0.5):
            logger.warning("[CLEANUP] open order(s) still pending cancel after"
                           " %ss — later submits may reject with insufficient"
                           " qty", timeout)
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
        open_orders = [o for o in _list_open_orders(api,
                                                    symbols=sorted(variants))
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

    if _await_orders_clear(api, variants, timeout, poll_interval):
        return True
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
    def _entry(pos):
        qty = float(pos.qty)
        if qty <= 0:
            return None
        entry_price = float(pos.avg_entry_price)
        # current_price can be None under the alpaca-py adapter
        # (the shim passes it through) — float(None) raised and
        # silently dropped a LIVE position from startup tracking.
        # Fall back to entry for the HWM; base_loop ratchets it up.
        cp = getattr(pos, 'current_price', None)
        current_price = float(cp) if cp is not None else entry_price
        return {
            'qty': qty,
            'entry_price': entry_price,
            'high_water_mark': max(entry_price, current_price),
        }

    positions = {}
    # Primary path: ONE list_positions call instead of up to 2 get_position
    # probes per universe symbol (46-name stock startup = 46+ round trips of
    # mostly 404s on the Jetson). A missing key here is unambiguously 'no
    # position'; the per-symbol probe loop below survives as the fallback
    # for shims/fakes without list_positions or a transient listing failure.
    try:
        pos_map = {getattr(p, 'symbol', None): p for p in api.list_positions()}
    except Exception:
        pos_map = None
    if pos_map is not None:
        for sym in symbols:
            for candidate in _symbol_variants(sym):
                pos = pos_map.get(candidate)
                if pos is None:
                    continue
                try:
                    info = _entry(pos)
                except Exception as e:
                    logger.warning("[RECONSTRUCT] %s: bad position payload (%s)"
                                   " — a live position may be untracked",
                                   candidate, e)
                    continue
                if info is not None:
                    positions[sym] = info
                    break
        return positions

    for sym in symbols:
        for candidate in _symbol_variants(sym):
            try:
                info = _entry(api.get_position(candidate))
                if info is not None:
                    positions[sym] = info
                    break
            except Exception as e:
                if not _is_not_found(e):
                    logger.warning("[RECONSTRUCT] %s: get_position failed (%s)"
                                   " — a live position may be untracked", candidate, e)
                continue
    return positions


# --- CIRCUIT BREAKER ---

def check_circuit_breaker(api, max_drawdown_pct=0.05):
    """Check if daily equity drawdown exceeds threshold.

    SCOPE: the measured drawdown is ACCOUNT-WIDE — both books share one
    Alpaca account, so a drawdown driven entirely by one book trips the
    other book's caller too; there is no per-book attribution here.
    BASELINE: account.last_equity is the previous EQUITY-TRADING-DAY close,
    so for the 24/7 crypto book the window spans weekends/holidays (a
    Sat+Sun drift is measured against Friday's close).

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
            logger.warning("[CIRCUIT BREAKER] last_equity=%s — drawdown not"
                           " computable this cycle (breaker effectively"
                           " disabled)", last_equity)
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
        list of symbols whose flatten could not be confirmed. When the
        position listing itself fails, returns the sentinel list
        ['<list_positions failed>'] — base_loop string-matches that exact
        literal, so it must never change.
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
            if not cancel_orders_for_symbol(api, pos.symbol, timeout=5):
                logger.error("[EMERGENCY] %s: resting orders still pending"
                             " cancel — the market flatten may reject with"
                             " insufficient qty", pos.symbol)

    # Phase 1 — submit EVERY market flatten first: whole-book exposure closes
    # after one submit round-trip each, instead of each position waiting out
    # the previous one's confirmation poll (identical orders, same TIF/tags).
    failures = []
    submitted = []
    for pos in targets:
        sym = pos.symbol
        try:
            qty = float(pos.qty)
            if qty == 0:
                continue  # already flat — a 0-qty submit only manufactures a rejection
            side = 'sell' if qty > 0 else 'buy'
            if tif_for_symbol is not None:
                tif = tif_for_symbol(sym)
            else:
                tif = 'gtc' if ('/' in sym or (sym.endswith('USD') and len(sym) > 5)) else 'day'
            order = api.submit_order(
                symbol=sym,
                qty=abs(qty),
                side=side,
                type='market',
                time_in_force=tif,
                client_order_id=make_client_order_id('flatten'),
            )
            logger.warning("[EMERGENCY] %s: Market %s %s", sym, side, pos.qty)
            submitted.append((sym, order.id))
        except Exception as e:
            logger.error("[EMERGENCY] %s: %s", sym, e)
            failures.append(sym)

    # Phase 2 — verify each flatten reached a terminal good state
    for sym, order_id in submitted:
        try:
            # confirm-only: a slow flatten must never cancel its own liquidation (c26 D19)
            result = manage_order_lifecycle(api, order_id, timeout=15,
                                            fallback_to_market=False,
                                            cancel_on_timeout=False)
            status = getattr(result, 'status', None)
            if status != 'filled':
                logger.error("[EMERGENCY] %s: flatten NOT confirmed"
                             " (status=%s, filled_qty=%s)",
                             sym, status, getattr(result, 'filled_qty', None))
                failures.append(sym)
        except Exception as e:
            logger.error("[EMERGENCY] %s: %s", sym, e)
            failures.append(sym)

    if failures:
        logger.error("[EMERGENCY] UNCONFIRMED flattens: %s", ', '.join(failures))
    return failures
