"""Stock trading loop — subclass of BaseTradingLoop.

Trades only during regular market hours (9:30 AM - 4:00 PM ET):
  1. Score all stocks with the model, trade only top N by signal strength
  2. Check stop-loss / trailing stop upgrades on open positions
  3. Sell positions where the model signals weakness or they drop from top N
  4. Buy top-N bullish stocks (sentiment-gated, bracket orders with stops)
  5. Flatten all stock positions at 3:50 PM ET to avoid overnight gap risk

Enhanced with GARCH volatility, macro regime, correlation-aware sizing,
Kelly criterion, HMM regime detection, and VIX-based risk scaling.
"""

import json
import time
import datetime
import zoneinfo
from pathlib import Path

from base_loop import BaseTradingLoop
from order_utils import (
    get_stock_quote, place_stock_limit_order, manage_order_lifecycle,
    get_all_positions, compute_limit_price, cancel_orders_for_symbol,
    make_client_order_id,
)
from market_data import fetch_spy_bars_alpaca, get_live_atr
from sentiment import sentiment_gate, get_market_sentiment, get_recent_headlines
from stock_config import load_stock_universe
from fundamentals import get_fundamentals, get_insider_activity, get_filing_summary, format_fundamentals_for_llm
from log_config import get_logger
from trade_memory import record_trade

logger = get_logger(__name__)

_PRED_CACHE_FILE = Path(__file__).resolve().parent / "stock_predictions.json"

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
FLATTEN_HOUR = 15
FLATTEN_MINUTE = 50


class StockLoop(BaseTradingLoop):
    """Market-hours stock trading loop."""

    # Rank hysteresis (Garleanu-Pedersen no-trade band): enter only the
    # strongest names, but keep holding well past the entry cutoff so a
    # name oscillating around the boundary doesn't churn round-trip costs
    TOP_N = 7          # entry: rank <= 7
    HOLD_RANK = 15     # hold:  rank <= 15
    NOTIONAL_PER_SYMBOL = 5000
    MAX_NOTIONAL_PER_SYMBOL = 5000
    MAX_EXPOSURE = 50000
    ORDER_TIMEOUT = 30
    LOOP_INTERVAL = 30
    MAX_PREDICTION_WORKERS = 5
    LLM_INTERVAL_SEC = 600
    CIRCUIT_BREAKER_PCT = 0.05
    MODEL_PREFIX = 'stock'

    # ATR stops — values come from strategy_config so the backtester
    # validates the SAME policy the bot trades
    from strategy_config import STOCK_POLICY as _P
    ATR_STOP_MULTIPLIER = _P['atr_stop_mult']
    ATR_TRAIL_MULTIPLIER = _P['atr_trail_mult']
    ATR_TRAIL_ACTIVATE_PCT = _P['trail_activate_pct']
    ATR_STOP_FLOOR_PCT = _P['stop_floor_pct']
    ATR_STOP_CEIL_PCT = _P['stop_ceil_pct']
    TAKE_PROFIT_RR = _P['tp_rr']
    TAKE_PROFIT_CEIL_PCT = _P['tp_ceil_pct']
    STOP_LOSS_PCT = _P['stop_fallback_pct']
    TRAIL_PCT = _P['trail_fallback_pct']
    COOLDOWN_MINUTES = _P['cooldown_min']
    HARD_STOP_LOCKOUT_HOURS = _P['lockout_hours']
    del _P

    def __init__(self):
        super().__init__()
        self.flattened_today = False
        self.last_date = None
        self.top_symbols: list[str] = []
        self.hold_symbols: set[str] = set()
        self._clock_cache: tuple[float, object] = (0.0, None)
        self._last_preds: dict[str, float] = {}

    def get_symbol_universe(self) -> list[str]:
        return [s for s in load_stock_universe() if '/' not in s]

    def _get_clock(self):
        """Alpaca market clock, cached ~60s.

        The clock knows about holidays AND early closes (1:00 PM on
        Nov 27 / Dec 24 etc.) — wall-clock 9:30-16:00 logic does not,
        which previously left positions held overnight on exactly the
        days with the worst gap risk.
        """
        now = time.monotonic()
        ts, clock = self._clock_cache
        if clock is not None and (now - ts) < 60:
            return clock
        try:
            clock = self.api.get_clock()
            self._clock_cache = (now, clock)
            return clock
        except Exception as e:
            logger.warning("[CLOCK] get_clock failed (%s) — falling back to wall clock", e)
            return None

    def check_market_hours(self) -> bool:
        now = self._get_eastern_now()
        # Reset flatten flag on new day
        if self.last_date != now.date():
            self.flattened_today = False
            self.last_date = now.date()

        clock = self._get_clock()
        if clock is not None:
            return bool(clock.is_open)

        # Fallback: wall-clock schedule (no holiday/half-day awareness)
        if now.weekday() >= 5:
            return False
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
        return market_open <= now < market_close

    def get_asset_type(self) -> str:
        return 'stock'

    def get_quote(self, symbol: str) -> dict | None:
        return get_stock_quote(self.api, symbol)

    def place_buy_order(self, symbol, qty, quote, stop_price=None, tp_price=None):
        limit_price = compute_limit_price('buy', quote, offset_bps=5)
        limit_price = round(limit_price, 2)
        try:
            kwargs = {
                'symbol': symbol, 'qty': qty, 'side': 'buy',
                'type': 'limit', 'limit_price': limit_price,
                'time_in_force': 'day',
            }
            if stop_price and tp_price:
                kwargs['order_class'] = 'bracket'
                kwargs['stop_loss'] = {'stop_price': stop_price}
                kwargs['take_profit'] = {'limit_price': tp_price}

            order = self.api.submit_order(**kwargs)
            return order
        except Exception as e:
            logger.error("[ORDER] %s: bracket order error: %s", symbol, e)
            return None

    def place_sell_order(self, symbol, qty, quote):
        """Sell and confirm. Returns the FILLED order object, or None."""
        if quote is not None:
            order = place_stock_limit_order(self.api, symbol, 'sell', int(qty), quote,
                                            time_in_force='day')
            if order:
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=True,
                                                time_in_force='day')
                if result is not None and getattr(result, 'status', None) == 'filled':
                    return result
        return None

    def get_benchmark_close(self):
        try:
            spy_df = fetch_spy_bars_alpaca(self.api)
            if spy_df is not None:
                return spy_df['Close']
        except Exception as e:
            logger.error("[SPY] Benchmark error: %s", e)
        return None

    def get_headlines(self, symbol: str) -> list[str]:
        return get_recent_headlines(symbol, 'stock')

    def _in_flatten_window(self) -> bool:
        """True within the last 10 minutes before today's ACTUAL close."""
        clock = self._get_clock()
        if clock is not None:
            try:
                next_close = clock.next_close
                if hasattr(next_close, 'to_pydatetime'):
                    next_close = next_close.to_pydatetime()
                now_utc = datetime.datetime.now(datetime.timezone.utc)
                close_utc = next_close.astimezone(datetime.timezone.utc)
                # Only meaningful when the close is TODAY's session close
                if bool(clock.is_open):
                    return (close_utc - now_utc) <= datetime.timedelta(minutes=10)
                return False
            except Exception as e:
                logger.warning("[FLATTEN] clock parse failed (%s) — wall-clock fallback", e)
        now = self._get_eastern_now()
        flatten_time = now.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)
        return flatten_time <= now < market_close

    def _in_entry_window(self) -> bool:
        """True when new entries are allowed (open/close windows, ET).

        Gao-Han-Li-Zhou (2018): intraday equity predictability concentrates
        in the first and last half-hours; midday is mostly noise that pays
        spread. Exits/stops run all day — only ENTRIES are windowed.
        """
        from strategy_config import STOCK_ENTRY_WINDOWS_ET, ENTRY_WINDOWS_ENABLED
        if not ENTRY_WINDOWS_ENABLED:
            return True
        now = self._get_eastern_now()
        minutes = now.hour * 60 + now.minute
        for start_s, end_s in STOCK_ENTRY_WINDOWS_ET:
            sh, sm = map(int, start_s.split(':'))
            eh, em = map(int, end_s.split(':'))
            if sh * 60 + sm <= minutes < eh * 60 + em:
                return True
        return False

    def _select_overnight_keepers(self) -> set[str]:
        """Pick the capped overnight sleeve (Lou-Polk-Skouras 2019).

        Essentially all of the equity premium accrues overnight; flattening
        100% of the book every day pays the spread twice AND forfeits that
        premium. Keep up to OVERNIGHT_SLEEVE_MAX_POSITIONS positions that
        (a) the model still predicts up, (b) fit under the per-name equity
        cap, and (c) are not leveraged ETFs (daily-reset decay products).
        GTC stops replace the day-TIF legs before the close
        (_prepare_overnight_keepers).
        """
        from strategy_config import (OVERNIGHT_SLEEVE_ENABLED,
                                     OVERNIGHT_SLEEVE_MAX_POSITIONS,
                                     OVERNIGHT_SLEEVE_MAX_PCT_EQUITY,
                                     OVERNIGHT_SLEEVE_MIN_PRED)
        if not OVERNIGHT_SLEEVE_ENABLED or not self.positions:
            return set()

        # EARNINGS FAIL-CLOSED: GTC stops cannot protect against gaps (they
        # fill at the open, wherever that lands). If the earnings calendar
        # is unavailable we cannot prove a keeper is safe — keep nothing.
        from events_calendar import calendar_available, blocks_overnight_hold
        if not calendar_available():
            logger.warning("[SLEEVE] earnings calendar unavailable — "
                           "fail closed, no overnight keepers tonight")
            return set()

        candidates = []
        for sym, info in self.positions.items():
            if self._leveraged_etfs.get(sym, 1) > 1:
                continue
            if blocks_overnight_hold(sym):
                logger.info("[SLEEVE] %s excluded — earnings print ahead", sym)
                continue
            pred = self._last_preds.get(sym)
            if pred is None or pred < OVERNIGHT_SLEEVE_MIN_PRED:
                continue
            value = info.qty * info.entry_price
            if value > OVERNIGHT_SLEEVE_MAX_PCT_EQUITY * max(self._equity, 1):
                continue
            # Loser-bounce exclusion (Baltussen-Da-Soebhag 2025 Table 6):
            # a day-loser that bounced hard into the close is transitory
            # liquidity pressure that fully reverts by the next OPEN —
            # exactly the wrong name to hold for the overnight premium
            if self._is_bounced_loser(sym):
                logger.info("[SLEEVE] %s: excluded — EOD loser-bounce "
                            "reverts at the open", sym)
                continue
            candidates.append((pred, sym))
        if not candidates:
            return set()
        # Rank-blend: 70% model prediction, 30% overnight-component
        # momentum (wave 4 / Lou-Polk-Skouras: the momentum premium
        # accrues OVERNIGHT — the sleeve is exactly where that component
        # belongs). Falls back to pure pred ordering when ON_Mom_252 is
        # missing from snapshots.
        snaps = getattr(self, '_last_snapshots', {}) or {}
        n = len(candidates)
        pred_rank = {s: r for r, (_, s) in
                     enumerate(sorted(candidates), 1)}
        on_vals = {s: snaps.get(s, {}).get('ON_Mom_252') for _, s in candidates}
        if n > 1 and any(v is not None for v in on_vals.values()):
            on_sorted = sorted((v if v is not None else -1e9, s)
                               for s, v in on_vals.items())
            on_rank = {s: r for r, (_, s) in enumerate(on_sorted, 1)}
            scored = [(0.7 * pred_rank[s] + 0.3 * on_rank[s], s)
                      for _, s in candidates]
        else:
            scored = [(pred_rank[s], s) for _, s in candidates]
        scored.sort(reverse=True)   # higher combined rank = better
        return {sym for _, sym in scored[:OVERNIGHT_SLEEVE_MAX_POSITIONS]}

    def flatten_before_close(self):
        """Flatten stock positions ~10 min before the actual market close.

        Every stock buy is a bracket order, so the shares are RESERVED by
        the live stop/TP leg — selling without canceling those legs first
        rejects with 'insufficient qty'. Cancel-and-confirm per symbol,
        then sell with fill confirmation. flattened_today is only set once
        the book is actually empty, so failures retry next cycle.
        """
        if self.flattened_today:
            return
        if not self._in_flatten_window():
            return

        keepers = self._select_overnight_keepers()
        if keepers:
            logger.info("[FLATTEN] Overnight sleeve keeps: %s", ', '.join(sorted(keepers)))
            self._prepare_overnight_keepers(keepers)

        # Include ORPHANS: broker positions in our universe that tracking
        # lost (e.g. after a desync) must not ride overnight either.
        universe = set(self.get_symbol_universe())
        targets = dict(self.positions)
        broker_positions = get_all_positions(self.api)
        if broker_positions:
            for sym, pos in broker_positions.items():
                if sym in universe and sym not in targets and '/' not in sym:
                    logger.warning("[FLATTEN] Orphan broker position %s (%s shares) — flattening too",
                                   sym, pos.qty)
                    targets[sym] = None  # not tracked; sell whatever broker reports

        failures = []
        for symbol in targets:
            if symbol in keepers:
                continue
            try:
                try:
                    pos = self.api.get_position(symbol)
                    qty = int(float(pos.qty))
                except Exception:
                    self.positions.pop(symbol, None)
                    continue
                if qty <= 0:
                    self.positions.pop(symbol, None)
                    continue

                # Free the shares: bracket/trailing legs hold them. Alpaca
                # cancellation is async — wait until confirmed before selling.
                if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                    logger.error("[FLATTEN] %s: legs still pending cancel — will retry", symbol)
                    failures.append(symbol)
                    continue
                tracked = self.positions.get(symbol)
                if tracked is not None:
                    tracked.stop_order_id = None

                quote = get_stock_quote(self.api, symbol)
                order = None
                if quote is not None:
                    order = place_stock_limit_order(self.api, symbol, 'sell', qty, quote,
                                                    time_in_force='day', offset_bps=10)
                if order is None:
                    order = self.api.submit_order(
                        symbol=symbol, qty=qty, side='sell',
                        type='market', time_in_force='day',
                        client_order_id=make_client_order_id('flatten'))
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=True,
                                                time_in_force='day')
                if result is not None and getattr(result, 'status', None) == 'filled':
                    if tracked is not None:
                        llm_info = self.llm_scores.get(symbol, {})
                        self._record_confirmed_exit(symbol, tracked, result, quote,
                                                    exit_reason='eod_flatten',
                                                    llm_score=llm_info.get('s'),
                                                    reasoning=llm_info.get('r', ''))
                    self.positions.pop(symbol, None)
                    logger.info("[FLATTEN] %s: Sold %d shares", symbol, qty)
                else:
                    logger.error("[FLATTEN] %s: sell unconfirmed (status=%s) — will retry",
                                 symbol, getattr(result, 'status', None))
                    failures.append(symbol)
            except Exception as e:
                logger.error("[FLATTEN] %s: Error: %s", symbol, e)
                failures.append(symbol)
            time.sleep(0.5)

        remaining = [s for s in self.positions if s not in keepers]
        if failures or remaining:
            stuck = sorted(set(failures) | set(remaining))
            logger.error("[FLATTEN] Incomplete (%d unconfirmed) — retrying next cycle, NOT marking done",
                         len(stuck))
            try:
                from notify import notify
                notify(f"EOD FLATTEN INCOMPLETE: {', '.join(stuck)} still open "
                       f"near the close — positions may ride overnight",
                       level='critical', dedupe_key='flatten-incomplete')
            except Exception:
                pass
            return
        self.flattened_today = True
        logger.info("[FLATTEN] Done. No more entries today.")

    def _prepare_overnight_keepers(self, keepers: set[str]):
        """Replace day-TIF protective legs with GTC stops for kept names.

        Day-TIF bracket legs expire at the close, which would leave sleeve
        positions unprotected overnight. (Implementation used by the
        overnight sleeve, Phase 3c.)
        """
        for symbol in keepers:
            info = self.positions.get(symbol)
            if info is None:
                continue
            try:
                if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                    logger.error("[SLEEVE] %s: could not clear day legs", symbol)
                    continue
                entry_atr = info.entry_atr
                if entry_atr is not None and info.entry_price > 0:
                    raw = (entry_atr * self.ATR_STOP_MULTIPLIER) / info.entry_price
                    stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw))
                else:
                    stop_dist = self.STOP_LOSS_PCT
                stop_price = round(info.entry_price * (1 - stop_dist), 2)
                stop_order = self.api.submit_order(
                    symbol=symbol, qty=int(info.qty), side='sell',
                    type='stop', stop_price=stop_price,
                    time_in_force='gtc',
                    client_order_id=make_client_order_id('onstop'))
                info.stop_order_id = stop_order.id
                info.trailing_activated = False
                logger.info("[SLEEVE] %s: GTC stop @ $%.2f for overnight hold", symbol, stop_price)
            except Exception as e:
                logger.error("[SLEEVE] %s: GTC stop placement failed: %s — flattening instead", symbol, e)
                keepers.discard(symbol)

    def write_prediction_cache(self, preds, **kwargs):
        top_symbols = kwargs.get('top_symbols', self.top_symbols)
        try:
            data = {}
            for sym in sorted(preds):
                pred = preds[sym]
                if sym in top_symbols:
                    signal = "BULL"
                elif pred is not None and pred < -self.trade_threshold:
                    signal = "BEAR"
                else:
                    signal = "NEUTRAL"
                data[sym] = {
                    "pred": round(pred, 6) if pred is not None else None,
                    "score": round(pred, 6) if pred is not None else 0,
                    "signal": signal,
                    "updated": datetime.datetime.now().isoformat(),
                }
            with open(_PRED_CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("[CACHE] Error writing stock prediction cache: %s", e)

    def _build_llm_candidates(self, preds: dict) -> list[dict]:
        """Stock-specific: include insider activity and filing summaries."""
        candidates = []
        for symbol in self.top_symbols:
            fund = get_fundamentals(symbol, 'stock')
            insider = get_insider_activity(symbol)
            filing_sum = get_filing_summary(symbol)
            fund_text = format_fundamentals_for_llm(symbol, fund, insider, filing_sum)
            headlines = self.get_fresh_headlines(symbol)
            candidates.append({
                'symbol': symbol,
                'pred_return': preds.get(symbol),
                'fundamentals_text': fund_text,
                'news_headlines': headlines,
            })
        return candidates

    def _get_predictions(self, benchmark_close):
        """Override to add top-N ranking."""
        # Panel pre-pass (wave-3 flagship): compute live cross-sectional
        # ranks over the SAME as-of top-K panel training ranked over, and
        # register them for predict_now's CS_* injection. Hourly-cached
        # inside; fail-open to neutral ranks.
        try:
            from panel_ranks import compute_live_panel_ranks
            from predict_now import set_panel_features
            set_panel_features(compute_live_panel_ranks(
                self.api, spy_close=benchmark_close))
        except Exception as e:
            logger.warning("[PANEL] live pre-pass failed (%s) — "
                           "neutral ranks this cycle", e)
        preds, snapshots = super()._get_predictions(benchmark_close)
        self._last_preds = dict(preds)  # sleeve selection reads these
        self._last_snapshots = dict(snapshots)

        # Dynamic top N selection with hold buffer
        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        # High-VIX dip-preference tiebreak (wave 4 / Nagel: short-term
        # reversal is the return to liquidity provision, priced UP when
        # intermediaries are constrained). Among the model's own top
        # candidates, demote names whose 5d residual return is a recent
        # POP — tiebreak only, never overrides the model/meta/q10 gates.
        vix = self.macro_regime.vix if self.macro_regime else None
        if vix is not None and vix > 25 and len(ranked) > self.TOP_N:
            window = ranked[:self.TOP_N + 3]
            rr = {s: (snapshots.get(s, {}) or {}).get('RR_5') for s, _ in window}
            vals = [v for v in rr.values() if v is not None]
            if len(vals) >= 4:
                import numpy as _np
                pop_cut = float(_np.percentile(vals, 80))
                poppers = {s for s, v in rr.items()
                           if v is not None and v >= pop_cut and v > 0}
                if poppers:
                    window.sort(key=lambda kv: (kv[0] in poppers, -kv[1]))
                    ranked = window + ranked[self.TOP_N + 3:]
                    logger.info("[RR-TIEBREAK] VIX %.0f: demoted recent "
                                "poppers %s", vix, ', '.join(sorted(poppers)))
        self.top_symbols = [sym for sym, _ in ranked[:self.TOP_N]]
        self.hold_symbols = {sym for sym, _ in ranked[:self.HOLD_RANK]}
        if self.top_symbols:
            logger.info("[RANK] Top %d: %s", self.TOP_N,
                        ', '.join(f'{s}({preds[s]:+.4f})' for s in self.top_symbols))

        self.write_prediction_cache(preds, top_symbols=self.top_symbols)

        # Log market sentiment periodically
        if self.cycle % 10 == 1:
            mkt = get_market_sentiment()
            if mkt is not None:
                logger.info("[SENTIMENT] Market: score=%+.2f, pos=%.0f%%/neg=%.0f%%",
                            mkt['sentiment_score'],
                            mkt['positive_ratio'] * 100,
                            mkt['negative_ratio'] * 100)

        return preds, snapshots

    def _execute_sells(self, preds: dict):
        """Stock-specific: also sell positions that drop from top N."""
        from trading_utils import cooldown_ok

        for symbol in list(self.positions):
            try:
                pos = self.api.get_position(symbol)
            except Exception as e:
                err_str = str(e).lower()
                if 'not found' in err_str or '404' in err_str or 'no position' in err_str:
                    # Position closed at the broker outside our tracking
                    # (e.g. TP leg filled between cycles) — journal it so
                    # the Kelly sample isn't censored of these exits
                    info = self.positions[symbol]
                    quote = self.get_quote(symbol)
                    px = quote['midpoint'] if quote else info.entry_price
                    pnl = ((px - info.entry_price) / info.entry_price * 100
                           if info.entry_price > 0 else 0.0)
                    record_trade(symbol, 'sell', info.entry_price, px, pnl,
                                 exit_reason='external_close', estimated=True)
                    del self.positions[symbol]
                continue

            sell_reason = None
            pred = preds.get(symbol)
            if pred is not None and pred < -self.trade_threshold:
                sell_reason = f"pred={pred:+.4f}%"
            elif (symbol not in self.hold_symbols
                  and pred is not None and pred < 0):
                sell_reason = (f"fell below hold rank {self.HOLD_RANK} "
                               f"(pred={pred:+.4f}%)")

            if sell_reason is None:
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue

            logger.info("%s: SELLING (%s)", symbol, sell_reason)
            qty = int(float(pos.qty))
            if qty <= 0:
                del self.positions[symbol]
                continue

            info = self.positions[symbol]
            # Bracket/trailing legs reserve the shares; wait until the
            # cancel is CONFIRMED or the sell will reject.
            if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                logger.warning("%s: legs still pending cancel — retrying next cycle", symbol)
                continue
            info.stop_order_id = None

            quote = self.get_quote(symbol)
            order = self.place_sell_order(symbol, qty, quote)
            if order:
                llm_info = self.llm_scores.get(symbol, {})
                self._record_confirmed_exit(symbol, info, order, quote,
                                            exit_reason='signal_sell',
                                            llm_score=llm_info.get('s'),
                                            reasoning=llm_info.get('r', ''))
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(0.5)

    def _bucket_room_ok(self, symbol: str) -> bool:
        """Sector-bucket notional cap. Conservative: blocks when one more
        full-size entry would breach the bucket's share of MAX_EXPOSURE."""
        from stock_config import SECTOR_BUCKETS, BUCKET_CAP_FRACTION
        bucket = SECTOR_BUCKETS.get(symbol)
        if bucket is None:
            return True
        cap = self.MAX_EXPOSURE * BUCKET_CAP_FRACTION.get(
            bucket, BUCKET_CAP_FRACTION['default'])
        held = sum(p.qty * p.entry_price for s, p in self.positions.items()
                   if SECTOR_BUCKETS.get(s) == bucket)
        if held + self.NOTIONAL_PER_SYMBOL > cap:
            logger.info("[BUCKET] %s: %s bucket at $%.0f/$%.0f — entry blocked",
                        symbol, bucket, held, cap)
            return False
        return True

    def _execute_buys(self, preds: dict, snapshots: dict):
        """Stock-specific: bracket orders with server-side stops, exposure tracking."""
        from trading_utils import cooldown_ok
        from order_utils import should_trade
        from trade_journal import log_decision
        from trading_utils import LLM_VETO_THRESHOLD
        from portfolio import check_portfolio_correlation, get_correlation_sizing_factor

        if self.flattened_today:
            return

        # Macro-event stand-down (FOMC/CPI windows)
        if not self._entries_allowed():
            return

        # Entry windows: predictability lives in the open/close half-hours
        if not self._in_entry_window():
            return

        # Faber 200-day trend filter: below trend, only safe havens
        from macro_indicators import get_spy_trend_ok
        from stock_config import SAFE_HAVEN_SYMBOLS
        trend_ok = get_spy_trend_ok(self.api)

        current_exposure = self._get_current_exposure()
        if current_exposure is None:
            logger.warning("[EXPOSURE] API error, skipping buys")
            return

        from collections import Counter
        vc = Counter()          # veto attribution for the window summary
        admitted = []           # ranked names that cleared every gate
        n_candidates = 0        # ranked candidates evaluated past mechanical gates
        for rank, symbol in enumerate(self.top_symbols, 1):
            if symbol in self.positions:
                vc['already_held'] += 1
                continue

            if current_exposure >= self.MAX_EXPOSURE:
                logger.info("Max exposure $%d reached, no more buys", self.MAX_EXPOSURE)
                vc['max_exposure'] += 1
                break

            # Sector-bucket cap: ranked entries cluster in one theme on
            # exactly the days that theme is running hot (factor crowding)
            if not self._bucket_room_ok(symbol):
                vc['bucket_cap'] += 1
                self._journal_skip(symbol, 'bucket_cap', rank=rank,
                                   pred=preds.get(symbol),
                                   snapshot=snapshots.get(symbol))
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                vc['cooldown'] += 1
                continue

            if self._is_hard_stop_locked(symbol):
                vc['hard_stop_lockout'] += 1
                continue

            if not self._trade_budget_ok(symbol):
                vc['trade_budget'] += 1
                continue

            # No new entries within a day of a known earnings print
            # (fail OPEN on calendar outage — the sleeve is fail-closed)
            try:
                from events_calendar import earnings_within_days
                if earnings_within_days(symbol, days=1):
                    logger.info("%s: blocked — earnings within 1 day", symbol)
                    vc['earnings'] += 1
                    self._journal_skip(symbol, 'earnings', rank=rank,
                                       pred=preds.get(symbol),
                                       snapshot=snapshots.get(symbol))
                    continue
            except Exception:
                pass

            # Corporate-event veto: fresh 8-K solvency/credibility items
            # and pending-M&A targets (price pinned to deal terms) — the
            # model's technical signals are invalid in both states
            try:
                from edgar_events import entry_blocked
                ev_blocked, ev_reason = entry_blocked(symbol)
                if ev_blocked:
                    vc['edgar_event'] += 1
                    rec = {"symbol": symbol, "action": "skip",
                           "skip_reason": "edgar_event", "detail": ev_reason,
                           "entry_rank": rank}
                    log_decision(rec)
                    continue
            except Exception:
                pass

            pred = preds.get(symbol)
            if pred is None:
                vc['no_pred'] += 1
                continue
            n_candidates += 1
            snapshot = snapshots.get(symbol, {})
            if pred < self.trade_threshold:
                vc['below_threshold'] += 1
                self._journal_skip(symbol, 'below_threshold', rank=rank,
                                   pred=pred, snapshot=snapshot)
                continue

            quote = self.get_quote(symbol)
            if quote is None:
                vc['no_quote'] += 1
                continue

            if not should_trade(pred, quote['spread_pct'], asset_type='stock'):
                vc['cost_floor'] += 1
                self._journal_skip(symbol, 'cost_floor', rank=rank, pred=pred,
                                   snapshot=snapshot)
                continue

            # Winner's curse filter
            sma20 = snapshot.get('SMA_20')
            atr = snapshot.get('ATR')
            if sma20 and atr and quote['midpoint'] > sma20 + 2 * atr:
                required = self.trade_threshold * 1.5
                if pred < required:
                    logger.info("%s: Winner's curse filter, need %.2f got %.4f",
                                symbol, required, pred)
                    vc['winners_curse'] += 1
                    self._journal_skip(symbol, 'winners_curse', rank=rank,
                                       pred=pred, snapshot=snapshot)
                    continue

            # Correlation check
            if self.corr_matrix and self.positions:
                allowed, avg_corr = check_portfolio_correlation(
                    list(self.positions.keys()), symbol, self.corr_matrix)
                if not allowed:
                    vc['correlation'] += 1
                    self._journal_skip(symbol, 'correlation', rank=rank,
                                       pred=pred, snapshot=snapshot)
                    continue

            # Macro regime halt
            if self.macro_regime and self.macro_regime.should_halt_stocks:
                logger.info("%s: Halted by VIX > 35", symbol)
                vc['macro_halt'] += 1
                continue

            # VIX > 25: block risky entries, allow safe-havens
            if self.macro_regime and self.macro_regime.should_block_risky_entries:
                if symbol not in SAFE_HAVEN_SYMBOLS:
                    logger.info("%s: Blocked — VIX > 25 defensive (non-safe-haven)", symbol)
                    vc['vix_block'] += 1
                    continue

            # SPY below its 200d SMA: block non-safe-haven entries (Faber)
            if trend_ok is False and symbol not in SAFE_HAVEN_SYMBOLS:
                logger.info("%s: Blocked — SPY below 200d SMA (trend filter)", symbol)
                vc['trend_filter'] += 1
                self._journal_skip(symbol, 'trend_filter', rank=rank,
                                   pred=pred, snapshot=snapshot)
                continue

            # Sentiment gate (veto first; multiplier folds into sizing tilt)
            gate, gate_reasons = sentiment_gate(symbol, 'stock')
            if gate <= 0:
                vc['sentiment_block'] += 1
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "pred_return": pred, "entry_rank": rank,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons})
                continue

            # LLM gate (veto first; multiplier folds into sizing tilt)
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            llm_reason = llm_info.get('r', '')
            if llm_s < LLM_VETO_THRESHOLD:
                vc['llm_veto'] += 1
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "llm_veto",
                              "pred_return": pred, "entry_rank": rank,
                              "llm_score": llm_s, "llm_reasoning": llm_reason})
                continue
            llm_mult = 0.5 + llm_s

            # Meta-labeling gate (veto + bounded sizing multiplier)
            meta_ok, meta_mult = self._meta_gate(symbol, pred, snapshots, rank=rank)
            if not meta_ok:
                vc['meta_veto'] += 1
                continue

            # q10 tail veto (fat left tail despite bullish mean)
            q10 = snapshot.get('Q10')
            q10_floor = snapshot.get('Q10_Floor')
            if q10 is not None and q10_floor is not None and q10 < q10_floor:
                vc['q10_tail_veto'] += 1
                log_decision({"symbol": symbol, "action": "skip",
                              "skip_reason": "q10_tail_veto",
                              "pred_return": pred, "entry_rank": rank,
                              "q10": round(q10, 4),
                              "q10_floor": round(q10_floor, 4)})
                continue

            # Single risk-based sizing call (all bounds enforced inside)
            sized_notional = self._compute_position_size(
                symbol, pred, quote, sentiment_mult=gate, llm_mult=llm_mult,
                meta_mult=meta_mult)
            if sized_notional <= 0:
                vc['sizing_zero'] += 1
                self._journal_skip(symbol, 'sizing_zero', rank=rank, pred=pred,
                                   snapshot=snapshot)
                continue

            # Calculate qty (whole shares)
            price = quote['midpoint']
            if price <= 0:
                continue
            qty = int(sized_notional / price)
            if qty <= 0:
                continue

            # ATR-based stop and take-profit for bracket order
            entry_atr = get_live_atr(self.api, symbol, asset_type='stock')
            limit_price = compute_limit_price('buy', quote, offset_bps=5)
            limit_price = round(limit_price, 2)

            if entry_atr is not None and limit_price > 0:
                raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / limit_price
                stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
                tp_dist = min(self.TAKE_PROFIT_CEIL_PCT, stop_dist * self.TAKE_PROFIT_RR)
            else:
                stop_dist = self.STOP_LOSS_PCT
                tp_dist = self.TAKE_PROFIT_CEIL_PCT

            # Apply macro regime stop tightening
            if self.macro_regime and self.macro_regime.stop_mult < 1.0:
                stop_dist *= self.macro_regime.stop_mult

            stop_price = round(limit_price * (1 - stop_dist), 2)
            tp_price = round(limit_price * (1 + tp_dist), 2)

            logger.info("%s: BUYING %d @ ~$%.2f (pred=%+.4f%%, stop=$%.2f, tp=$%.2f)",
                        symbol, qty, price, pred, stop_price, tp_price)

            admitted.append(symbol)

            import random
            time.sleep(random.uniform(0, 5))

            order = self.place_buy_order(symbol, qty, quote, stop_price, tp_price)
            if order:
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                               fallback_to_market=False)
                if result and result.status == 'filled':
                    child_stop_id = None
                    try:
                        legs = self.api.list_orders(status='open', symbols=[symbol])
                        for leg in legs:
                            if leg.side == 'sell' and leg.type in ('stop', 'stop_limit'):
                                child_stop_id = leg.id
                                break
                    except Exception:
                        pass

                    fill_price = float(result.filled_avg_price)
                    from types_mod import Position
                    self.positions[symbol] = Position(
                        qty=qty,
                        entry_price=fill_price,
                        high_water_mark=fill_price,
                        stop_order_id=child_stop_id,
                        trailing_activated=False,
                        entry_atr=entry_atr,
                        take_profit_price=tp_price,
                    )
                    decision_price = quote['midpoint']
                    slippage_bps = ((fill_price - decision_price) / decision_price * 1e4
                                    if decision_price > 0 else None)
                    # Conviction context + realized book stop-risk of this
                    # fill (wave-5 Tier1-1): book_risk_pct lets Stage-0
                    # measure how often the $5k notional cap binds before
                    # the 0.5%-risk sizing does (notional-cap-bind audit).
                    book_risk_pct = (round(qty * fill_price * stop_dist
                                           / max(self._equity, 1) * 100, 4))
                    buy_rec = {"symbol": symbol, "action": "buy",
                               "pred_return": pred,
                               "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                               "llm_multiplier": llm_mult, "llm_score": llm_s,
                               "llm_reasoning": llm_reason,
                               "final_notional": sized_notional,
                               "decision_price": decision_price,
                               "fill_price": fill_price,
                               "slippage_bps": round(slippage_bps, 2) if slippage_bps is not None else None,
                               "book_risk_pct": book_risk_pct,
                               "skip_reason": None}
                    buy_rec.update(self._conv_fields(symbol, pred, snapshot,
                                                     rank=rank))
                    log_decision(buy_rec)
                    self.last_trade_time[symbol] = datetime.datetime.now()
                    self._count_trade(symbol)
                    estimated_exposure = current_exposure + qty * fill_price

                    # After fill confirmation (None on API error — fall back
                    # to the local estimate instead of `None > int` crashing)
                    refreshed = self._get_current_exposure()
                    current_exposure = refreshed if refreshed is not None else estimated_exposure
                    if current_exposure > self.MAX_EXPOSURE:
                        logger.warning("[EXPOSURE] Exceeded cap after fill: $%.0f > $%.0f",
                                       current_exposure, self.MAX_EXPOSURE)
                        self._journal_entry_window(n_candidates, admitted, vc)
                        return  # Stop placing more orders this cycle
            time.sleep(0.5)

        self._journal_entry_window(n_candidates, admitted, vc)

    def _manage_stops(self):
        """Stock-specific: check server-side stop fills, upgrade to trailing stops."""
        for symbol in list(self.positions):
            info = self.positions[symbol]
            if info.stop_order_id:
                try:
                    stop_order = self.api.get_order(info.stop_order_id)
                    if stop_order.status == 'filled':
                        logger.info("[STOP-FILL] %s: Stop filled at $%s",
                                    symbol, stop_order.filled_avg_price)
                        # Server-side exits were previously never journaled,
                        # censoring exactly the losing trades out of the
                        # Kelly sizing sample.
                        llm_info = self.llm_scores.get(symbol, {})
                        self._record_confirmed_exit(symbol, info, stop_order, None,
                                                    exit_reason='server_stop',
                                                    llm_score=llm_info.get('s'),
                                                    reasoning=llm_info.get('r', ''))
                        del self.positions[symbol]
                        self.last_trade_time[symbol] = datetime.datetime.now()
                        self.hard_stop_lockout[symbol] = datetime.datetime.now()
                        self._save_hard_stop_lockout()
                        continue
                    elif stop_order.status in ('canceled', 'expired', 'rejected'):
                        info.stop_order_id = None
                except Exception:
                    info.stop_order_id = None

            quote = get_stock_quote(self.api, symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            entry_price = info.entry_price
            info.high_water_mark = max(info.high_water_mark, current_price)

            # Trailing stop upgrade
            entry_atr = info.entry_atr
            if entry_atr is not None and entry_price > 0:
                raw_trail_dist = (entry_atr * self.ATR_TRAIL_MULTIPLIER) / entry_price
                trail_pct = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_trail_dist))
            else:
                trail_pct = self.TRAIL_PCT

            if (not info.trailing_activated
                    and current_price >= entry_price * (1 + self.ATR_TRAIL_ACTIVATE_PCT)
                    and info.stop_order_id):
                try:
                    # Canceling one bracket leg cancels the whole OCO group;
                    # wait for confirmation or the trailing submit rejects
                    # with shares-held.
                    if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                        logger.warning("[TRAIL] %s: legs pending cancel — retry next cycle", symbol)
                        continue
                    trail_order = self.api.submit_order(
                        symbol=symbol, qty=int(info.qty), side='sell',
                        type='trailing_stop',
                        trail_percent=round(trail_pct * 100, 1),
                        time_in_force='day',
                        client_order_id=make_client_order_id('trail'),
                    )
                    info.stop_order_id = trail_order.id
                    info.trailing_activated = True
                    logger.info("[TRAIL] %s: Upgraded to trailing stop (%.1f%%) at $%.2f",
                                symbol, trail_pct * 100, current_price)
                except Exception as e:
                    logger.error("[TRAIL] %s: Upgrade error: %s", symbol, e)
                    info.stop_order_id = None

        # Also run base class stop management for software stops
        super()._manage_stops()

    def _extra_tilt(self, symbol: str) -> float:
        """Stock-specific sizing tilts (post-print vol + PM-tape ROD)."""
        tilt = 1.0
        # Post-print vol haircut: ATR(14)/GARCH lag the 3-5x realized-vol
        # expansion after an earnings report by ~1 day — halve the first
        # post-print day's size instead of trading at stale risk numbers
        try:
            from events_calendar import reported_recently
            if reported_recently(symbol):
                tilt *= 0.5
        except Exception:
            pass
        # SPY rest-of-day momentum gate for the PM window: on a strongly
        # DOWN tape (|ROD| > 0.5x daily vol), the documented last-hour
        # continuation runs against fresh longs — halve PM entries.
        # Strong-up tapes ride at full size (continuation helps longs).
        try:
            tilt *= self._spy_rod_pm_tilt()
        except Exception:
            pass
        return tilt

    def _spy_rod_pm_tilt(self) -> float:
        """0.5 when entering the 14:30-15:30 window into a strong-down
        SPY tape; 1.0 otherwise. Uses the cached SPY benchmark series."""
        now = self._get_eastern_now()
        if not (14 <= now.hour < 16):
            return 1.0
        spy = self.get_benchmark_close()
        if spy is None or len(spy) < 200:
            return 1.0
        dates = spy.index.normalize()
        last_date = dates[-1]
        prior = spy[dates < last_date]
        if prior.empty:
            return 1.0
        prev_close = float(prior.iloc[-1])
        rod = (float(spy.iloc[-1]) / prev_close - 1.0) * 100
        daily = spy.groupby(dates).last().pct_change().dropna() * 100
        vol = float(daily.tail(20).std())
        if vol <= 0:
            return 1.0
        if rod < -0.5 * vol:
            logger.info("[ROD] SPY %.2f%% (-%.1f sigma-day) — PM entries "
                        "halved (last-hour continuation risk)", rod,
                        abs(rod) / vol)
            return 0.5
        return 1.0

    def _is_bounced_loser(self, symbol: str) -> bool:
        """True when today's day-loser bounced hard in the late session
        (prev-close-to-15:00 strongly negative AND 15:00-to-now strongly
        positive) — the BDS transitory-pressure pattern."""
        try:
            from market_data import fetch_stock_bars_alpaca
            bars = fetch_stock_bars_alpaca(self.api, symbol)
            if bars is None or len(bars) < 30:
                return False
            close = bars['Close']
            dates = close.index.normalize()
            last_date = dates[-1]
            today = close[dates == last_date]
            prior = close[dates < last_date]
            if len(today) < 2 or prior.empty:
                return False
            prev_close = float(prior.iloc[-1])
            rod_now = float(today.iloc[-1]) / prev_close - 1.0
            rod_early = float(today.iloc[-2]) / prev_close - 1.0
            bounce = rod_now - rod_early
            # Day loser (<-1.5%) that bounced >+0.75% in the final stretch
            return rod_early < -0.015 and bounce > 0.0075
        except Exception:
            return False

    def _replace_protective_stops(self):
        """Re-place a server-side stop for every reconstructed stock position.

        Startup cancels this bot's working orders (incl. bracket legs);
        without re-placement, positions would depend on 30s software
        polling — i.e. on this process staying alive — for protection.
        """
        for symbol, info in self.positions.items():
            try:
                entry_atr = info.entry_atr
                if entry_atr is not None and info.entry_price > 0:
                    raw = (entry_atr * self.ATR_STOP_MULTIPLIER) / info.entry_price
                    stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw))
                else:
                    stop_dist = self.STOP_LOSS_PCT
                # Anchor to the HWM so a restart doesn't widen an
                # already-tightened trail back to entry-based distance
                anchor = max(info.entry_price, info.high_water_mark)
                stop_price = round(anchor * (1 - stop_dist), 2)
                qty = int(float(info.qty))
                if qty <= 0:
                    continue
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side='sell',
                    type='stop', stop_price=stop_price,
                    time_in_force='day',
                    client_order_id=make_client_order_id('restop'),
                )
                info.stop_order_id = order.id
                logger.info("[RECONSTRUCT] %s: protective stop re-placed @ $%.2f", symbol, stop_price)
            except Exception as e:
                logger.error("[RECONSTRUCT] %s: stop re-placement failed: %s "
                             "(software stops still active)", symbol, e)

    def _get_current_exposure(self) -> float | None:
        """Calculate total stock exposure."""
        positions = get_all_positions(self.api)
        if positions is None:
            return None
        total = 0.0
        for sym, pos in positions.items():
            if '/' not in sym and 'USD' not in sym:
                total += abs(float(pos.market_value))
        return total

    def _get_eastern_now(self):
        return datetime.datetime.now(zoneinfo.ZoneInfo('US/Eastern'))


def run_stock_bot():
    loop = StockLoop()
    loop.run()


if __name__ == "__main__":
    run_stock_bot()
