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

# Realized maker-share feedback (LIVE gate only)
MAKER_SHARE_WINDOW_DAYS = 14
MAKER_SHARE_MIN_ENTRIES = 30
_MAKER_SHARE_TTL = 3600

_maker_share_cache: tuple[float, float | None] | None = None  # (mono_ts, share)


def realized_crypto_maker_share(days: int = MAKER_SHARE_WINDOW_DAYS,
                                min_entries: int = MAKER_SHARE_MIN_ENTRIES
                                ) -> float | None:
    """Fraction of recent crypto entries that filled via maker tactics.

    Read from the decision journals (entry_tactic logged per buy by the
    maker ladder). None until at least min_entries crypto entries exist
    in the window — thin samples must not move the cost model. Cached 1h.
    """
    import json
    import time as _time
    from datetime import datetime, timedelta

    global _maker_share_cache
    now = _time.monotonic()
    if _maker_share_cache and (now - _maker_share_cache[0]) < _MAKER_SHARE_TTL:
        return _maker_share_cache[1]

    share = None
    try:
        from trade_journal import JOURNAL_DIR
        tactics = []
        today = datetime.now().date()
        for d in range(days + 1):
            path = JOURNAL_DIR / f"{(today - timedelta(days=d)).isoformat()}.jsonl"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sym = e.get('symbol', '')
                    if (e.get('action') == 'buy' and '/' in sym
                            and e.get('entry_tactic')):
                        tactics.append(e['entry_tactic'])
        if len(tactics) >= min_entries:
            share = sum(1 for t in tactics if t.startswith('maker')) / len(tactics)
    except Exception:
        share = None
    _maker_share_cache = (now, share)
    return share


def crypto_entry_fee_bps(live: bool = False) -> float:
    """Expected crypto ENTRY-side fee in bps.

    Static contexts (training objective, backtest gate, meta replay)
    price the conservative taker fee — a model that clears taker costs
    also clears maker costs, and feeding realized fills back into the
    GATES would loosen them exactly when recent fills look good.
    The LIVE entry gate may blend by realized maker share so an
    overstated cost floor doesn't reject genuinely positive-edge trades.
    """
    if not live:
        return CRYPTO_TAKER_BPS
    share = realized_crypto_maker_share()
    if share is None:
        return CRYPTO_TAKER_BPS
    return CRYPTO_MAKER_BPS * share + CRYPTO_TAKER_BPS * (1.0 - share)


def round_trip_cost_pct(asset_type: str, spread_pct: float = 0.0,
                        maker: bool = False, live: bool = False) -> float:
    """Estimated round-trip cost as a PERCENT of notional (0.5 == 0.5%).

    Args:
        asset_type: 'crypto' or 'stock'
        spread_pct: current bid/ask spread as percent of price; crossing the
            spread costs roughly one full spread per round trip
        maker: True forces the maker fee on the entry side (crypto only)
        live: True lets the crypto ENTRY side blend maker/taker by the
            REALIZED maker share from the journals (exit side is always
            taker — exits are market/stop-market). Training and gate
            contexts must leave this False.
    """
    spread_cost = max(spread_pct, 0.0)
    if asset_type == 'crypto':
        if maker:
            entry_bps = CRYPTO_MAKER_BPS
        else:
            entry_bps = crypto_entry_fee_bps(live=live)
        return (entry_bps + CRYPTO_TAKER_BPS) / 100.0 + spread_cost
    return (STOCK_REGULATORY_BPS + 2 * STOCK_SLIPPAGE_BPS_PER_SIDE) / 100.0 + spread_cost


def required_edge_pct(asset_type: str, spread_pct: float = 0.0,
                      maker: bool = False,
                      min_edge: float = MIN_EDGE_MULTIPLE,
                      live: bool = False) -> float:
    """Minimum predicted move (percent) for a trade to clear costs."""
    return round_trip_cost_pct(asset_type, spread_pct, maker, live) * min_edge
