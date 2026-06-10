"""Transaction-cost model — the binding constraint on this strategy.

Alpaca crypto charges 15 bps maker / 25 bps taker PER SIDE at tier 1
(<$100k 30-day volume; see docs.alpaca.markets/us/docs/crypto-fees), so a
taker round trip costs 50 bps before spread. US equities are commission-free
but pay SEC ($20.60 per $1M, sells) + FINRA TAF (~$0.000195/share, sells)
plus spread/slippage.

The old gate compared predictions against spread alone (and training assumed
5 bps round trip) — admitting structurally negative-expectancy crypto trades.
Every entry gate and the training objective should price costs through this
module so live and simulated economics agree.
"""

# Alpaca crypto fee schedule, tier 1, bps per side
CRYPTO_TAKER_BPS = 25.0
CRYPTO_MAKER_BPS = 15.0

# US equities (per round trip): regulatory fees on the sell side are tiny
# (~0.2-0.3 bps); slippage allowance covers marketable-limit/market fills.
STOCK_REGULATORY_BPS = 0.3   # sell-side SEC + TAF, expressed per round trip
STOCK_SLIPPAGE_BPS_PER_SIDE = 3.0

# Entry gate: predicted move must exceed this multiple of round-trip cost
MIN_EDGE_MULTIPLE = 2.0


def round_trip_cost_pct(asset_type: str, spread_pct: float = 0.0,
                        maker: bool = False) -> float:
    """Estimated round-trip cost as a PERCENT of notional (0.5 == 0.5%).

    Args:
        asset_type: 'crypto' or 'stock'
        spread_pct: current bid/ask spread as percent of price; crossing the
            spread costs roughly one full spread per round trip
        maker: True if entries rest as maker limit orders (crypto only —
            15 bps vs 25 bps per side)
    """
    spread_cost = max(spread_pct, 0.0)
    if asset_type == 'crypto':
        fee_side_bps = CRYPTO_MAKER_BPS if maker else CRYPTO_TAKER_BPS
        return (2 * fee_side_bps) / 100.0 + spread_cost
    return (STOCK_REGULATORY_BPS + 2 * STOCK_SLIPPAGE_BPS_PER_SIDE) / 100.0 + spread_cost


def required_edge_pct(asset_type: str, spread_pct: float = 0.0,
                      maker: bool = False,
                      min_edge: float = MIN_EDGE_MULTIPLE) -> float:
    """Minimum predicted move (percent) for a trade to clear costs."""
    return round_trip_cost_pct(asset_type, spread_pct, maker) * min_edge
