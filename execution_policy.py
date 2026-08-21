"""Explicit, calibrated entry-tactic selection (wave-7, Finding 1 UPGRADE).

`compute_limit_price` has been spread-aware since commit 8c9d860, but its
decision lives in buried magic constants (a 0.1% spread cutoff and a
20%-of-half-spread offset) that the backtester does not share. This module
lifts that decision into one pure, table-driven function with named
thresholds in strategy_config, so that — once wired — the live loop and the
policy backtest will choose the SAME tactic and the thresholds are auditable.

Tactic vocabulary:
  cross  — take liquidity now (marketable). Tight spreads, mega-caps, or thin
           edge where a missed passive fill would forfeit the trade.
  post   — rest a passive limit inside the quote. Genuinely wide spreads on
           names that fill passively, when edge headroom can absorb a non-fill.
  ladder — split the difference: a passive rung that re-chases. The middle band.
           The table names the tactic ONLY — rung count / stage timeout live
           with the caller (order_utils.place_maker_buy's stage_timeout /
           max_reprices, strategy_config.MAKER_STAGE_TIMEOUT) and are not
           carried in the returned dict. No stock-side passive executor
           exists: live stock entries are atomic bracket parents
           (stock_loop.place_buy_order), so the stock 'post'/'ladder'
           branches are unexecutable until the bracket-parent redesign lands.

Crypto is always `post`: Alpaca's maker vs taker fee gap (~10bps/side — a
reduced maker FEE, not a rebate) dwarfs the 1-3bps BTC/ETH spread, so posting
is cheaper CONDITIONAL ON FILLING. Live, crypto 'post' denotes
order_utils.place_maker_buy's bid-join ladder (join the bid -> reprice once
-> marketable remainder): the taker fallback bounds the non-fill cost and a
wired caller must never drop it; post_offset_pct=0.0 there means 'join the
touch', not 'rest inside the quote'.

CALLER CONTRACT (read before wiring):
  * This table answers HOW to enter, never WHETHER. Callers MUST still honor
    strategy_config.MAKER_ENTRIES_ENABLED (the crypto passive-entry kill
    switch crypto_loop checks today) and, for stocks, ENTRY_WINDOWS_ENABLED,
    before acting on a post/ladder result — the table reads neither flag.
  * The returned `tactic` (cross/post/ladder) is an ex-ante DECISION
    vocabulary, NOT the journal's `entry_tactic` field. The journal's
    ex-post REALIZED values ('maker'/'maker_reprice'/'maker_partial'/
    'marketable'/'marketable_bracket'/'taker_fallback') are prefix-matched
    on 'maker' by fees.realized_crypto_maker_share and execution_report,
    feeding the LIVE crypto entry cost gate. Writing cross/post/ladder into
    `entry_tactic` would zero the realized maker share and silently revert
    that gate to full-taker pricing — journal the decision under a separate
    key and keep `entry_tactic` as the realized outcome.

NOTHING here is tuned on realized P&L — thresholds are microstructure priors +
the offline Eff_Spread_Pct ranking (name_class). Pure function, no I/O beyond
a warning log, no model; unit-tested in isolation. NOT YET WIRED: neither
order_utils (live tactics) nor backtest.py (replay) consumes this table —
both wirings are separate (Jetson-validated) steps.
"""

import logging
import math

from strategy_config import (EXEC_TAKER_FLOOR_PCT, EXEC_WIDE_SPREAD_PCT,
                             EXEC_POST_INSIDE_FRAC, EXEC_EDGE_HEADROOM_MULT)

log = logging.getLogger(__name__)

# name_class WILL BE seeded offline from the per-name Eff_Spread_Pct ranking
# (liquidity.py) with a weekly refresh — not from live quotes. The seed table
# + refresh job are not built yet, so wired callers would currently receive
# the 'mid' default for every symbol (disabling the mega/spec rows).
# TRAP: the repo's only existing per-symbol class lookup
# (borrow_proxy.class_lookup_from_config -> stock_config.SECTOR_BUCKETS)
# emits a DISJOINT vocabulary ('megacap_tech', 'spec_growth', 'semis', ...);
# passing those values here silently coerces every symbol to 'mid'. Do not
# wire it in — the returned name_class/name_class_coerced keys expose the
# coercion when it happens.
VALID_CLASSES = ('mega', 'mid', 'spec')


def _finite_or_none(value):
    """Coerce to a finite float; None for anything else (missing data).

    None, non-numeric types (dict/list/str garbage) and non-finite floats
    (nan/inf) all mean MISSING, not a market state: a raise out of a pure
    lookup would kill the caller's whole entry cycle, and a NaN would
    survive the band comparisons and poison a limit price once the module
    is wired.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def choose_entry_tactic(asset_type, live_spread_pct, pred_return=None,
                        edge_floor=None, name_class='mid'):
    """Pick {cross|post|ladder} for an entry. Pure lookup.

    Args:
        asset_type: 'crypto' (always post) or 'stock'. The match is EXACT and
            case-sensitive; any other label is routed through the stock table
            with a log.warning (mirroring fees.round_trip_cost_pct) — the
            routing itself is deliberately unchanged.
        live_spread_pct: current quoted bid/ask spread, PERCENT of the quote
            MIDPOINT (the same denominator as order_utils.get_quote's
            spread_pct). None / non-numeric / NaN / inf are MISSING data and
            fail closed to cross; a finite negative value is a locked/crossed
            quote and also fails closed to cross. There is NO upper sanity
            bound — callers must reject implausible quotes upstream (a
            bid 0.01 / ask 1.00 book passes get_quote's degeneracy checks
            and arrives here as spread_pct ~196).
        pred_return: model's predicted return (PERCENT), for edge headroom.
            SIGNED, for a LONG entry — a negative (short) prediction always
            fails the headroom test, so short callers must pass the edge
            magnitude (abs) instead. Non-numeric and non-finite values fail
            the headroom test (same as None, never a raise): missing data
            must not fabricate headroom.
        edge_floor: the cost/edge floor (PERCENT) the entry must clear.
            UNPINNED CONTRACT — OPEN OWNER DECISION: the repo has two
            quantities answering this description, 2.0x apart —
            fees.round_trip_cost_pct (raw cost) and fees.required_edge_pct
            (= cost x MIN_EDGE_MULTIPLE=2.0, the value bound to `edge_floor`
            in backtest.py and enforced by the live should_trade gate). The
            two readings produce OPPOSITE tactics for the same entry, and
            under the raw-cost reading the headroom test is vacuous for any
            live-admitted trade (EXEC_EDGE_HEADROOM_MULT=1.5 < 2.0). Pass
            pred_return and edge_floor from the SAME gate evaluation in the
            SAME units. edge_floor <= 0, None, non-numeric or non-finite
            means 'cost floor unavailable' and DISABLES the post branch for
            non-spec names by design (fail closed against an unpriced
            passive non-fill) — it is NOT read as infinite headroom.
        name_class: 'mega'|'mid'|'spec' from the offline liquidity
            (Eff_Spread_Pct) ranking. NOT stock_config.SECTOR_BUCKETS /
            borrow_proxy's name_class vocabulary, and not normalized: any
            unknown value (including 'MEGA'/' mega') coerces to 'mid'.

    Returns dict {tactic, post_offset_pct, name_class, name_class_coerced,
    reason}. post_offset_pct is how far inside our side of the quote to
    rest, as a PERCENT of the quote midpoint: a long passive entry rests at
    bid + midpoint * post_offset_pct/100 (sell mirror: ask - ...). It always
    equals EXEC_POST_INSIDE_FRAC x the half-spread, so at the shipped 0.40
    it never reaches the mid. 0.0 with tactic 'post' (crypto) means join the
    touch; 0.0 with 'cross' means not applicable. Callers MUST pass the
    resulting price through order_utils._round_price_band before submitting:
    below roughly price*spread_pct = 2.5 the offset is smaller than the
    equity penny tick and legitimately rounds to a touch-join. name_class is
    the RESOLVED class the table used; name_class_coerced is True when the
    input was not in VALID_CLASSES.
    """
    nc = name_class if name_class in VALID_CLASSES else 'mid'
    nc_coerced = name_class not in VALID_CLASSES

    # Spread sanitation — two degraded states are tracked so `reason` can
    # tell a feed outage from a genuinely tight book (indistinguishable in a
    # journal otherwise):
    #   missing — None / non-numeric / NaN / inf spread: data unavailable.
    #   crossed — finite negative spread: locked/crossed quote (bid > ask).
    # Both fail CLOSED to the cross path (sp = 0.0); see _finite_or_none.
    sp = _finite_or_none(live_spread_pct)
    spread_missing = sp is None
    spread_crossed = (not spread_missing) and sp < 0.0
    sp = 0.0 if spread_missing else max(sp, 0.0)

    # Crypto: the maker fee gap dominates the spread -> always post (live,
    # this is place_maker_buy's bid-join ladder incl. its taker fallback).
    if asset_type == 'crypto':
        return {'tactic': 'post', 'post_offset_pct': 0.0,
                'name_class': nc, 'name_class_coerced': nc_coerced,
                'reason': 'crypto_maker_rebate_dominates_spread'}

    if asset_type != 'stock':
        # Mirror fees.round_trip_cost_pct: an unrecognized label must not
        # silently pick a tactic table. Routing is unchanged (stock table).
        log.warning("choose_entry_tactic: unknown asset_type %r routed "
                    "through the stock tactic table", asset_type)

    # Mega-caps trade in tight, deep books: crossing is cheap and certain;
    # never post (and never treat a transient wide quote as 'wide').
    if nc == 'mega':
        return {'tactic': 'cross', 'post_offset_pct': 0.0,
                'name_class': nc, 'name_class_coerced': nc_coerced,
                'reason': 'mega_tight_book_cross'}

    # Tight live spread -> crossing costs ~nothing; don't risk a non-fill.
    # Missing/crossed quotes were coerced to sp=0.0 above and land here too —
    # their reasons name the true state, not a measured tight spread.
    if sp <= EXEC_TAKER_FLOOR_PCT:
        if spread_missing:
            reason = 'spread_unavailable_cross'
        elif spread_crossed:
            reason = 'crossed_quote_cross'
        else:
            reason = 'spread_below_taker_floor'
        return {'tactic': 'cross', 'post_offset_pct': 0.0,
                'name_class': nc, 'name_class_coerced': nc_coerced,
                'reason': reason}

    # Genuinely wide spread: posting saves real money, but a passive non-fill
    # forfeits the trade — only post when the name fills passively (spec) or
    # the edge is fat enough to absorb a miss. Missing / non-numeric /
    # non-finite pred/floor fail the test: bad edge data must not fabricate
    # headroom (the mirror of the spread guard above).
    pred = _finite_or_none(pred_return)
    floor = _finite_or_none(edge_floor)
    has_headroom = (pred is not None and floor is not None and floor > 0
                    and pred >= EXEC_EDGE_HEADROOM_MULT * floor)
    # ONE shared offset expression for post AND ladder (deliberate — the two
    # branches must not drift): EXEC_POST_INSIDE_FRAC of the half-spread.
    inside = round(sp * 0.5 * EXEC_POST_INSIDE_FRAC, 4)
    if sp >= EXEC_WIDE_SPREAD_PCT and (nc == 'spec' or has_headroom):
        return {'tactic': 'post', 'post_offset_pct': inside,
                'name_class': nc, 'name_class_coerced': nc_coerced,
                'reason': 'wide_spread_post_inside'}

    # Fallthrough: a genuine middle band, OR a wide spread that failed the
    # spec/headroom test — distinct reasons, same ladder tactic.
    reason = ('wide_no_headroom_ladder' if sp >= EXEC_WIDE_SPREAD_PCT
              else 'mid_band_ladder')
    return {'tactic': 'ladder', 'post_offset_pct': inside,
            'name_class': nc, 'name_class_coerced': nc_coerced,
            'reason': reason}
