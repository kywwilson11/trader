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

Crypto is always `post`: Alpaca's maker vs taker fee gap (~10bps/side) dwarfs
the 1-3bps BTC/ETH spread, so posting is strictly cheaper regardless of width.

NOTHING here is tuned on realized P&L — thresholds are microstructure priors +
the offline Eff_Spread_Pct ranking (name_class). Pure function, no I/O, no
model; unit-tested in isolation. NOT YET WIRED: neither order_utils (live
tactics) nor backtest.py (replay) consumes this table — both wirings are
separate (Jetson-validated) steps.
"""

import math

from strategy_config import (EXEC_TAKER_FLOOR_PCT, EXEC_WIDE_SPREAD_PCT,
                             EXEC_POST_INSIDE_FRAC, EXEC_EDGE_HEADROOM_MULT)

# name_class WILL BE seeded offline from the per-name Eff_Spread_Pct ranking
# (liquidity.py) with a weekly refresh — not from live quotes. The seed table
# + refresh job are not built yet, so wired callers would currently receive
# the 'mid' default for every symbol (disabling the mega/spec rows).
VALID_CLASSES = ('mega', 'mid', 'spec')


def choose_entry_tactic(asset_type, live_spread_pct, pred_return=None,
                        edge_floor=None, name_class='mid'):
    """Pick {cross|post|ladder} for an entry. Pure lookup.

    Args:
        asset_type: 'crypto' (always post) or 'stock'.
        live_spread_pct: current quoted bid/ask spread, PERCENT of price.
        pred_return: model's predicted return (PERCENT), for edge headroom.
            SIGNED, for a LONG entry — a negative (short) prediction always
            fails the headroom test, so short callers must pass the edge
            magnitude (abs) instead.
        edge_floor: the cost/edge floor (PERCENT) the entry must clear.
        name_class: 'mega'|'mid'|'spec' from the offline liquidity ranking.

    Returns dict {tactic, post_offset_pct, reason}. post_offset_pct is how far
    inside our side of the quote to rest (PERCENT of price); 0 for cross.
    """
    nc = name_class if name_class in VALID_CLASSES else 'mid'
    sp = float(live_spread_pct or 0.0)
    if not math.isfinite(sp):
        # NaN/inf spread is MISSING data, not a wide market: fail closed to
        # the same path as None (-> cross). NaN survives `or 0.0` (truthy)
        # and both band comparisons, and would otherwise emit a NaN offset
        # that poisons the limit price.
        sp = 0.0
    sp = max(sp, 0.0)

    # Crypto: maker rebate dominates the spread -> always post.
    if asset_type == 'crypto':
        return {'tactic': 'post', 'post_offset_pct': 0.0,
                'reason': 'crypto_maker_rebate_dominates_spread'}

    # Mega-caps trade in tight, deep books: crossing is cheap and certain;
    # never post (and never treat a transient wide quote as 'wide').
    if nc == 'mega':
        return {'tactic': 'cross', 'post_offset_pct': 0.0,
                'reason': 'mega_tight_book_cross'}

    # Tight live spread -> crossing costs ~nothing; don't risk a non-fill.
    if sp <= EXEC_TAKER_FLOOR_PCT:
        return {'tactic': 'cross', 'post_offset_pct': 0.0,
                'reason': 'spread_below_taker_floor'}

    # Genuinely wide spread: posting saves real money, but a passive non-fill
    # forfeits the trade — only post when the name fills passively (spec) or
    # the edge is fat enough to absorb a miss.
    has_headroom = (pred_return is not None and edge_floor is not None
                    and edge_floor > 0
                    and pred_return >= EXEC_EDGE_HEADROOM_MULT * edge_floor)
    if sp >= EXEC_WIDE_SPREAD_PCT and (nc == 'spec' or has_headroom):
        return {'tactic': 'post',
                'post_offset_pct': round(sp * 0.5 * EXEC_POST_INSIDE_FRAC, 4),
                'reason': 'wide_spread_post_inside'}

    # Middle band -> ladder a passive rung that re-chases.
    return {'tactic': 'ladder',
            'post_offset_pct': round(sp * 0.5 * EXEC_POST_INSIDE_FRAC, 4),
            'reason': 'mid_band_ladder'}
