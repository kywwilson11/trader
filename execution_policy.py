"""Explicit, calibrated entry-tactic selection (wave-7, Finding 1 UPGRADE).

`compute_limit_price` has been spread-aware since commit 8c9d860, but its
decision lives in buried magic constants (a 0.1% spread cutoff and a
"10%-of-half-spread" offset whose comment says 20%) that the backtester does
not share. This module lifts that decision into one pure, table-driven
function with named thresholds in strategy_config, so the live loop and the
policy backtest choose the SAME tactic and the thresholds are auditable.

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
model; unit-tested in isolation. Live wiring of these tactics into order_utils
is a separate (Jetson-validated) step.
"""

from strategy_config import (EXEC_TAKER_FLOOR_PCT, EXEC_WIDE_SPREAD_PCT,
                             EXEC_POST_INSIDE_FRAC, EXEC_EDGE_HEADROOM_MULT)

# name_class is seeded OFFLINE from the per-name Eff_Spread_Pct ranking
# (liquidity.py), refreshed weekly — not from live quotes.
VALID_CLASSES = ('mega', 'mid', 'spec')


def choose_entry_tactic(asset_type, live_spread_pct, pred_return=None,
                        edge_floor=None, name_class='mid'):
    """Pick {cross|post|ladder} for an entry. Pure lookup.

    Args:
        asset_type: 'crypto' (always post) or 'stock'.
        live_spread_pct: current quoted bid/ask spread, PERCENT of price.
        pred_return: model's predicted return (PERCENT), for edge headroom.
        edge_floor: the cost/edge floor (PERCENT) the entry must clear.
        name_class: 'mega'|'mid'|'spec' from the offline liquidity ranking.

    Returns dict {tactic, post_offset_pct, reason}. post_offset_pct is how far
    inside our side of the quote to rest (PERCENT of price); 0 for cross.
    """
    nc = name_class if name_class in VALID_CLASSES else 'mid'
    sp = max(float(live_spread_pct or 0.0), 0.0)

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
