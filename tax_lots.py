"""Tax-lot estimation, extracted from gui.py so the matching logic is
Mac-testable (pure stdlib: datetime + collections only — no PySide6, no Qt,
no numpy). gui.py owns fetching orders from Alpaca and painting the result;
this module owns the MinTax lot-matching arithmetic.

IMPORTANT — MinTax lot selection (as implemented here) is an *optimistic
specific-identification* assumption: for every sell it picks losses first,
then long-term gains, then short-term gains, and within each tier the
highest-cost-basis lot — the ordering that minimizes tax *if* the filer has
validly elected specific-lot identification with the broker. This is NOT the
IRS default (FIFO) and the resulting figures are indicative estimates for a
paper account, not tax advice and not a filing-ready number.
"""

import datetime as dt
from collections import defaultdict

# Defaults mirror gui.py's former FED_SHORT_TERM / FED_LONG_TERM / STATE_RATE
# module constants — same numbers, now parameters so callers (and tests) can
# override them instead of monkeypatching module globals.
DEFAULT_FED_SHORT_TERM = 0.37
DEFAULT_FED_LONG_TERM = 0.20
DEFAULT_STATE_RATE = 0.05

# IRS long-term boundary: a lot must be held for MORE than one year (366+
# calendar days out) to qualify for long-term treatment; exactly 365 days is
# still short-term. gui.py's original used `>= 365`, off by one.
LONG_TERM_DAYS = 365


def _field(order, key, default=None):
    """Duck-typed field access: mapping `.get()` first, else `getattr`.

    gui.py's DataFetcher.fetch_orders() passes plain dicts today (that's the
    only shape that has ever reached estimate_taxes() in production), but
    accepting attribute-style objects too (e.g. a raw Alpaca SDK Order, or
    types.SimpleNamespace in tests) costs nothing and matches the "duck-typed
    order objects/dicts" contract this module is specified against. For a
    plain dict this is byte-identical to `order.get(key, default)`.
    """
    if isinstance(order, dict):
        return order.get(key, default)
    getter = getattr(order, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(order, key, default)


def _parse_time(value):
    """Parse an Alpaca-style ISO8601 timestamp (trailing 'Z' or explicit
    offset). Returns None if missing/unparseable — callers decide the
    fallback, matching gui.py's original silent-except behavior."""
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _is_long_term(buy_time, sell_time):
    """True iff held strictly MORE than LONG_TERM_DAYS days. Missing/
    unparseable timestamps count as short-term — same net effect as gui.py's
    original `except Exception: days_held = 0` fallback (0 >= 365 was always
    False, so treating it as short-term here is behavior-preserving)."""
    if buy_time is None or sell_time is None:
        return False
    return (sell_time - buy_time).days > LONG_TERM_DAYS


def _mintax_sort_key(lot, sell_price, sell_time_str):
    """Sort key for MinTax lot selection: losses first (highest cost basis),
    then long-term gains, then short-term gains — each sub-sorted by highest
    cost basis first, to minimize the gain recognized within its tier."""
    gain = sell_price - lot["price"]
    is_loss = gain < 0
    long_term = _is_long_term(_parse_time(lot["time"]), _parse_time(sell_time_str))
    if is_loss:
        tier = 0
    elif long_term:
        tier = 1
    else:
        tier = 2
    return (tier, -lot["price"])


def estimate_taxes(
    orders,
    *,
    fed_short=DEFAULT_FED_SHORT_TERM,
    fed_long=DEFAULT_FED_LONG_TERM,
    state_rate=DEFAULT_STATE_RATE,
    crypto_symbols=frozenset(),
    now=None,
    window_truncated=False,
):
    """MinTax lot matching over filled orders. Returns a dict of realized
    tax info for gui.py's tax card (see module docstring for the MinTax
    caveat — this is an indicative estimate, not a filing-ready number).

    Lots are matched in tax-optimal order: losses first (highest cost basis),
    then long-term gains, then short-term gains — each sub-sorted by highest
    cost basis to minimize realized gain within the tier.

    Args:
        orders: iterable of order records shaped like gui.py's
            DataFetcher.fetch_orders() output — dicts (or attribute-style
            objects; see _field) with symbol/side/status/filled_at/
            filled_qty (falls back to qty)/filled_avg_price. Only fills
            (status == "filled" with a filled_avg_price) are considered.
        fed_short: federal short-term capital-gains rate (ordinary income).
        fed_long: federal long-term capital-gains rate.
        state_rate: flat state rate added on top of both tiers.
        crypto_symbols: symbols to exclude from any wash-sale adjustment,
            since crypto has no US wash-sale rule. NOTE: the gui.py logic
            this module ports never had wash-sale adjustment of any kind
            (verified: no "wash" logic anywhere in gui.py), so this
            parameter is currently inert — kept in the signature so it does
            not need to change if wash-sale handling is ever added for the
            stock book.
        now: reserved clock-injection point (e.g. for a future open-lot
            aging feature). The realized-gain computation below is fully
            determined by historical order timestamps and does not consume
            it today; defaulted so the parameter is always well-formed.
        window_truncated: pass True when `orders` is known to be a
            truncated window (e.g. a paginated fetch capped at N orders) —
            forces basis_complete False even if every sell in this window
            happened to match.

    Returns:
        dict with keys:
            realized_gain, short_term_gain, long_term_gain (float): realized
                P&L broken out by holding-period tier.
            estimated_tax (float): tax on realized gains only (tier losses
                net against tier gains but a net-negative tier contributes
                no rebate — matches the original gui.py arithmetic).
            net_after_tax (float): realized_gain minus estimated_tax.
            num_trades (int): number of matched-lot fills (a multi-lot sell
                counts once per lot it consumes, same as the original).
            basis_complete (bool): False if any sell could not be fully
                matched against known buy-lot history, or if
                window_truncated was passed True.
            unmatched_sell_qty (float): total sell quantity (across all
                symbols) that found no/insufficient buy-lot history —
                silently dropped in the original gui.py code; counted here.
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    buys = defaultdict(list)
    realized = []
    unmatched_sell_qty = 0.0

    filled = [
        o for o in orders
        if _field(o, "status") == "filled" and _field(o, "filled_avg_price")
    ]
    filled.sort(key=lambda o: _field(o, "filled_at", "") or "")

    for o in filled:
        sym = _field(o, "symbol")
        try:
            qty = abs(float(_field(o, "filled_qty") or _field(o, "qty") or 0))
            price = float(_field(o, "filled_avg_price"))
        except (TypeError, ValueError):
            continue
        if qty == 0:
            continue

        filled_at = _field(o, "filled_at", "") or ""
        side = _field(o, "side")

        if side == "buy":
            buys[sym].append({"qty": qty, "price": price, "time": filled_at})
        elif side == "sell":
            remaining = qty
            buys[sym].sort(key=lambda lot: _mintax_sort_key(lot, price, filled_at))
            while remaining > 0 and buys[sym]:
                lot = buys[sym][0]
                matched = min(remaining, lot["qty"])
                gain = (price - lot["price"]) * matched
                long_term = _is_long_term(_parse_time(lot["time"]), _parse_time(filled_at))
                realized.append({
                    "symbol": sym, "gain": gain, "qty": matched,
                    "long_term": long_term,
                })
                lot["qty"] -= matched
                remaining -= matched
                if lot["qty"] <= 0:
                    buys[sym].pop(0)
            if remaining > 0:
                unmatched_sell_qty += remaining

    total_gain = sum(r["gain"] for r in realized)
    st_gain = sum(r["gain"] for r in realized if not r["long_term"])
    lt_gain = sum(r["gain"] for r in realized if r["long_term"])
    st_tax = max(0, st_gain) * (fed_short + state_rate)
    lt_tax = max(0, lt_gain) * (fed_long + state_rate)

    basis_complete = (not window_truncated) and (unmatched_sell_qty == 0)

    return {
        "realized_gain": total_gain,
        "short_term_gain": st_gain,
        "long_term_gain": lt_gain,
        "estimated_tax": st_tax + lt_tax,
        "net_after_tax": total_gain - (st_tax + lt_tax),
        "num_trades": len(realized),
        "basis_complete": basis_complete,
        "unmatched_sell_qty": unmatched_sell_qty,
    }
