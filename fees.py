"""Transaction-cost model — the binding constraint on this strategy.

Alpaca crypto charges 15 bps maker / 25 bps taker PER SIDE at tier 1
(<$100k 30-day volume; see docs.alpaca.markets/us/docs/crypto-fees), so a
taker round trip costs 50 bps before spread. US equities are commission-free
but pay small sell-side regulatory fees (SEC Section 31 + FINRA TAF), order of
magnitude ~0.2-0.3 bps per round trip; both rates are reset periodically by
rule, so STOCK_REGULATORY_BPS is an allowance, not a live rate. Plus
spread/slippage.

The old gate compared predictions against spread alone (and training assumed
5 bps round trip) — admitting structurally negative-expectancy crypto trades.

The cost-multiple ladder (one economic quantity, three multiples):
  1.0x  round_trip_cost_pct — raw cost CHARGED to P&L. Consumers: backtest
        net P&L, meta_label labels, decision_report counterfactuals,
        short_cost base, llm_analyst's prompt line.
  2.0x  required_edge_pct (= cost x MIN_EDGE_MULTIPLE) — the ADMISSION floor
        both entry gates enforce: backtest.py binds it to `edge_floor`; the
        live order_utils.should_trade compares abs(pred) against it.
  1.5x  on top — execution_policy's EXEC_EDGE_HEADROOM_MULT, whose base
        quantity (raw cost vs the admission floor, 2.0x apart) is an OPEN
        OWNER DECISION documented in execution_policy.choose_entry_tactic.
Consumers that do NOT import this module and carry their own copies, kept
honest by source-text tests only: scripts/hypersearch_v2.TXN_COST_PCT (the
Optuna training objective), backtest.SPREAD_PCT and meta_label's inline
spread literal (copies of FLAT_SPREAD_PCT), llm_analyst's prompt spread
literal. The meta-label replay applies NO cost floor at all (deliberate —
documented in meta_label._gen_meta_rows).
"""

import json
import logging
import math
import time as _time
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

# Alpaca crypto fee schedule, tier 1, bps per side
CRYPTO_TAKER_BPS = 25.0
CRYPTO_MAKER_BPS = 15.0

# US equities (per round trip): regulatory fees on the sell side are tiny
# (~0.2-0.3 bps); slippage allowance covers marketable-limit/market fills.
STOCK_REGULATORY_BPS = 0.3   # sell-side SEC + TAF, expressed per round trip
STOCK_SLIPPAGE_BPS_PER_SIDE = 3.0

# Flat per-round-trip spread haircut (PERCENT) when no per-bar estimate exists.
# Canonical copy — backtest.SPREAD_PCT, meta_label's inline fallback, the
# spread baked into scripts/hypersearch_v2.TXN_COST_PCT (the training
# objective's own cost copy) and llm_analyst's prompt literal must stay
# consistent with these values (tests/test_review_b10.py + tests/test_fees_v3.py
# cross-check their sources until they import this dict directly).
FLAT_SPREAD_PCT = {'crypto': 0.10, 'stock': 0.05}

# Entry gate: predicted move must exceed this multiple of round-trip cost
MIN_EDGE_MULTIPLE = 2.0

# Realized maker-share feedback (LIVE gate only)
MAKER_SHARE_WINDOW_DAYS = 14
MAKER_SHARE_MIN_ENTRIES = 30
_MAKER_SHARE_TTL = 3600

# (days, min_entries) -> (mono_ts, share); tests reset it to None (tolerated).
# Unbounded by construction, bounded in practice: production only ever uses
# the default-args key.
_maker_share_cache: dict[tuple[int, int], tuple[float, float | None]] | None = None


def realized_crypto_maker_share(days: int | None = None,
                                min_entries: int | None = None
                                ) -> float | None:
    """Fraction of recent crypto entries that filled via maker tactics.

    Read from the decision journals (entry_tactic logged per buy by the
    maker ladder). None until at least min_entries crypto entries exist
    in the window — thin samples must not move the cost model. Cached 1h.

    Definition (pinned): an ORDER-COUNT share, not notional-weighted, pooled
    across all crypto symbols. Conservatively biased DOWN: 'taker_fallback'
    zeroes an entry that may have filled mostly passively, while
    'maker_partial' counts as maker only because <$10 of dust was never
    bought (order_utils.place_maker_buy vocabulary — a cleared suspicion,
    not a bug). Window = today plus the previous `days` files (days+1
    calendar files), matching trade_journal.iter_journal_rows. As a trailing
    statistic it lags execution-policy changes (e.g. flipping
    strategy_config.MAKER_ENTRIES_ENABLED) by up to the full window.
    days/min_entries default to the module constants resolved at CALL time.

    Failure semantics (deliberate): malformed rows are skipped per row, but
    any file-level I/O error aborts the WHOLE scan to None (full taker) —
    all-or-nothing fail-closed, unlike trade_journal.iter_journal_rows'
    per-file tolerance; a partially-readable window must not move the live
    gate. None means thin sample, missing journals, or scan failure alike;
    the per-recompute log line disambiguates.
    """
    global _maker_share_cache
    # Resolve at CALL time (order_utils.should_trade's min_edge=None
    # convention) so a runtime rebinding of the constants cannot drift this
    # function from consumers that read them at call time.
    if days is None:
        days = MAKER_SHARE_WINDOW_DAYS
    if min_entries is None:
        min_entries = MAKER_SHARE_MIN_ENTRIES
    now = _time.monotonic()
    if not isinstance(_maker_share_cache, dict):
        _maker_share_cache = {}
    key = (days, min_entries)
    hit = _maker_share_cache.get(key)
    if hit is not None and (now - hit[0]) < _MAKER_SHARE_TTL:
        return hit[1]

    share = None
    try:
        # Late import ON PURPOSE: tests monkeypatch trade_journal.JOURNAL_DIR
        # and only a call-time import sees the patched value.
        from trade_journal import JOURNAL_DIR
        n_total = 0
        n_maker = 0
        today = datetime.now().date()
        for d in range(days + 1):
            path = JOURNAL_DIR / f"{(today - timedelta(days=d)).isoformat()}.jsonl"
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    # Cheap superset prefilter: json.dumps (ensure_ascii)
                    # always emits the key literally, so no countable row can
                    # lack this substring; non-matching rows skip the parse.
                    if '"entry_tactic"' not in line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(e, dict):
                        continue  # malformed row (bare scalar/list) — skip, don't abort
                    sym = e.get('symbol', '')
                    tac = e.get('entry_tactic')
                    if (e.get('action') == 'buy' and isinstance(sym, str)
                            and '/' in sym and isinstance(tac, str) and tac):
                        n_total += 1
                        if tac.startswith('maker'):
                            n_maker += 1
        if n_total >= min_entries:
            share = n_maker / n_total
        log.info("maker-share: %d crypto entries in %dd window (min %d) -> "
                 "share=%s", n_total, days, min_entries,
                 ('%.3f' % share) if share is not None
                 else 'None (thin sample — live gate prices full taker)')
    except Exception as exc:
        # Fail-safe to full-taker pricing, but leave a trace: a silent None
        # here disables the maker-share feedback with zero operator signal.
        log.warning("maker-share journal scan failed (%s: %s) — live crypto "
                    "entry gate prices full taker", type(exc).__name__, exc)
        share = None
    _maker_share_cache[key] = (now, share)
    return share


def crypto_entry_fee_bps(live: bool = False) -> float:
    """Expected crypto ENTRY-side fee in bps.

    Static contexts (training objective, backtest gate, meta replay)
    price the conservative taker fee — a model that clears taker costs
    also clears maker costs, and feeding realized fills back into the
    GATES would loosen them exactly when recent fills look good.
    The LIVE entry gate may blend by realized maker share so an
    overstated cost floor doesn't reject genuinely positive-edge trades.
    The blend is a trailing ORDER-COUNT statistic (see
    realized_crypto_maker_share) and is clamped to the fee schedule:
    never below maker, never above taker.
    """
    if not live:
        return CRYPTO_TAKER_BPS
    share = realized_crypto_maker_share()
    if share is None:
        return CRYPTO_TAKER_BPS
    # Defensive clamp: the blend must stay inside the fee schedule whatever
    # a future share definition produces (mirrors short_cost's htb clamp —
    # a backstop must never price below the cheapest real fee).
    share = min(max(share, 0.0), 1.0)
    return CRYPTO_MAKER_BPS * share + CRYPTO_TAKER_BPS * (1.0 - share)


def round_trip_cost_pct(asset_type: str, spread_pct: float = 0.0,
                        maker: bool = False, live: bool = False) -> float:
    """Estimated round-trip cost as a PERCENT of notional (0.5 == 0.5%).

    This is the raw cost CHARGED to P&L — NOT the admission floor;
    required_edge_pct() (this x MIN_EDGE_MULTIPLE) is what both entry
    gates compare predictions against.

    CONTRACT (load-bearing): linear in spread_pct for spread_pct >= 0,
    i.e. fee_const + spread. liquidity.per_bar_round_trip_cost vectorizes
    on exactly this (fee_const computed once at spread=0, spread array
    added) — any spread-dependent fee term added here must update that
    module in lockstep.

    Args:
        asset_type: 'crypto' or 'stock' (anything else warns and prices
            as stock — the more expensive fallback direction is crypto's,
            but an unknown label must never silently understate stock).
        spread_pct: FULL bid/ask spread as a percent of the quote MIDPOINT
            (order_utils.get_quote convention; liquidity's Eff_Spread_Pct
            uses the same units). Charged ONCE per round trip. NOT
            sanitized here: a negative value (crossed/locked quote) clamps
            to ZERO crossing cost — the loosest floor; NaN propagates NaN
            (backtest's `p < edge_floor` then fails OPEN); +inf propagates
            an infinite floor (rejects everything — fails CLOSED);
            None raises TypeError. The vectorized twin substitutes
            FLAT_SPREAD_PCT for bad values instead — aligning this scalar
            path is an open owner decision; both states warn below.
        maker: True forces the maker fee on the entry side (crypto only;
            no-op for stock). FEE-ONLY: the full spread is still charged —
            a deliberate conservative upper bound for a passive entry (do
            NOT 'fix' by simulating passive fills; killed, wave-7). Takes
            precedence over `live` and is deliberately NOT gated by it:
            the one way an offline context prices below full taker. No
            production caller passes it today.
        live: True lets the crypto ENTRY side blend maker/taker by the
            REALIZED maker share from the journals (exit side is always
            priced taker — crypto exits are marketable limits with market
            fallback plus a resting stop_limit; conservative). Crypto-only.
            Training and gate contexts must leave this False —
            order_utils.should_trade is the ONLY production live=True
            caller (pinned by tests/test_fees_v3.py).
    """
    if not (0.0 <= spread_pct < math.inf):
        # Negative (crossed/locked quote) clamps to zero crossing cost — the
        # LOOSEST floor; NaN propagates NaN (fails the gate OPEN), +inf
        # propagates inf (fails it CLOSED). Values unchanged pending an owner
        # ruling; warn so a degenerate spread reaching the cost model is
        # visible. (The chained form is False for negative, NaN AND +inf —
        # a bare `spread_pct >= 0.0` check would let +inf through silently.)
        log.warning("round_trip_cost_pct: degenerate spread_pct=%r for %s — "
                    "negative clamps to 0.0, NaN propagates NaN, "
                    "inf propagates inf", spread_pct, asset_type)
    spread_cost = max(spread_pct, 0.0)
    if asset_type == 'crypto':
        if maker:
            entry_bps = CRYPTO_MAKER_BPS
        else:
            entry_bps = crypto_entry_fee_bps(live=live)
        return (entry_bps + CRYPTO_TAKER_BPS) / 100.0 + spread_cost
    if asset_type != 'stock':
        # Stock pricing is ~8x cheaper than crypto — an unrecognized label
        # must not silently understate the cost floor.
        log.warning("round_trip_cost_pct: unknown asset_type %r priced as stock",
                    asset_type)
    return (STOCK_REGULATORY_BPS + 2 * STOCK_SLIPPAGE_BPS_PER_SIDE) / 100.0 + spread_cost


def required_edge_pct(asset_type: str, spread_pct: float = 0.0,
                      maker: bool = False,
                      min_edge: float | None = None,
                      live: bool = False) -> float:
    """The ADMISSION floor (PERCENT): round_trip_cost_pct x min_edge.

    NOT the break-even — a trade breaks even at round_trip_cost_pct();
    this returns min_edge times that. min_edge=None (the default) resolves
    to MIN_EDGE_MULTIPLE at CALL time, so a retune moves every defaulting
    caller (backtest's edge_floor included) together with the live gate —
    the same convention as order_utils.should_trade. Known comparison
    asymmetries, deliberate-by-omission: the live gate rejects
    pred == floor (strict >, on abs(pred)); the backtest replay admits it
    (skips only p < floor, on the signed p). min_edge <= 0 disables the
    gate entirely (floor <= 0); values in (0, 1) admit below-cost trades.
    """
    if min_edge is None:
        min_edge = MIN_EDGE_MULTIPLE
    return round_trip_cost_pct(asset_type, spread_pct, maker, live) * min_edge
