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
    get_all_positions, compute_limit_price,
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

    TOP_N = 10
    NOTIONAL_PER_SYMBOL = 5000
    MAX_NOTIONAL_PER_SYMBOL = 5000
    MAX_EXPOSURE = 50000
    ORDER_TIMEOUT = 30
    LOOP_INTERVAL = 30
    COOLDOWN_MINUTES = 20
    MAX_PREDICTION_WORKERS = 5
    LLM_INTERVAL_SEC = 600
    CIRCUIT_BREAKER_PCT = 0.05
    MODEL_PREFIX = 'stock'

    # ATR stops (stock-specific)
    ATR_STOP_MULTIPLIER = 2.0
    ATR_TRAIL_MULTIPLIER = 1.5
    ATR_TRAIL_ACTIVATE_PCT = 0.015
    ATR_STOP_FLOOR_PCT = 0.02
    ATR_STOP_CEIL_PCT = 0.10
    TAKE_PROFIT_RR = 3.0
    TAKE_PROFIT_CEIL_PCT = 0.20
    STOP_LOSS_PCT = 0.03
    TRAIL_PCT = 0.02

    def __init__(self):
        super().__init__()
        self.flattened_today = False
        self.last_date = None
        self.top_symbols: list[str] = []

    def get_symbol_universe(self) -> list[str]:
        return [s for s in load_stock_universe() if '/' not in s]

    def check_market_hours(self) -> bool:
        now = self._get_eastern_now()
        # Reset flatten flag on new day
        if self.last_date != now.date():
            self.flattened_today = False
            self.last_date = now.date()

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

    def place_sell_order(self, symbol, qty, quote) -> bool:
        if quote is not None:
            order = place_stock_limit_order(self.api, symbol, 'sell', int(qty), quote,
                                            time_in_force='day')
            if order:
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=True)
                return result and getattr(result, 'status', None) == 'filled'
        return False

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

    def flatten_before_close(self):
        """Flatten all stock positions at 3:50 PM ET."""
        if self.flattened_today:
            return

        now = self._get_eastern_now()
        flatten_time = now.replace(hour=FLATTEN_HOUR, minute=FLATTEN_MINUTE, second=0)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)

        if not (flatten_time <= now < market_close):
            return

        logger.info("[FLATTEN] Selling all stock positions before market close...")
        for symbol in list(self.positions):
            try:
                pos = self.api.get_position(symbol)
                qty = int(float(pos.qty))
                if qty <= 0:
                    del self.positions[symbol]
                    continue

                quote = get_stock_quote(self.api, symbol)
                if quote is not None:
                    order = place_stock_limit_order(self.api, symbol, 'sell', qty, quote,
                                                    time_in_force='day', offset_bps=10)
                    if order:
                        result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                       fallback_to_market=True)
                        if result:
                            del self.positions[symbol]
                            logger.info("[FLATTEN] %s: Sold %d shares", symbol, qty)
                else:
                    self.api.submit_order(symbol=symbol, qty=qty, side='sell',
                                         type='market', time_in_force='day')
                    del self.positions[symbol]
            except Exception as e:
                logger.error("[FLATTEN] %s: Error: %s", symbol, e)
            time.sleep(0.5)

        self.flattened_today = True
        logger.info("[FLATTEN] Done. No more trades today.")

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
            headlines = self.get_headlines(symbol)
            candidates.append({
                'symbol': symbol,
                'pred_return': preds.get(symbol),
                'fundamentals_text': fund_text,
                'news_headlines': headlines,
            })
        return candidates

    def _get_predictions(self, benchmark_close):
        """Override to add top-N ranking."""
        preds, snapshots = super()._get_predictions(benchmark_close)

        # Dynamic top N selection
        ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        self.top_symbols = [sym for sym, _ in ranked[:self.TOP_N]]
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
                    del self.positions[symbol]
                continue

            sell_reason = None
            pred = preds.get(symbol)
            if pred is not None and pred < -self.trade_threshold:
                sell_reason = f"pred={pred:+.4f}%"
            elif symbol not in self.top_symbols and pred is not None and pred < 0:
                sell_reason = f"dropped from top {self.TOP_N} (pred={pred:+.4f}%)"

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
            if info.stop_order_id:
                try:
                    self.api.cancel_order(info.stop_order_id)
                except Exception:
                    pass

            quote = self.get_quote(symbol)
            if self.place_sell_order(symbol, qty, quote):
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(0.5)

    def _execute_buys(self, preds: dict, snapshots: dict):
        """Stock-specific: bracket orders with server-side stops, exposure tracking."""
        from trading_utils import cooldown_ok
        from order_utils import should_trade
        from trade_journal import log_decision
        from trading_utils import LLM_VETO_THRESHOLD
        from portfolio import check_portfolio_correlation, get_correlation_sizing_factor

        if self.flattened_today:
            return

        current_exposure = self._get_current_exposure()
        if current_exposure is None:
            logger.warning("[EXPOSURE] API error, skipping buys")
            return

        for symbol in self.top_symbols:
            if symbol in self.positions:
                continue

            if current_exposure >= self.MAX_EXPOSURE:
                logger.info("Max exposure $%d reached, no more buys", self.MAX_EXPOSURE)
                break

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue

            if self._is_hard_stop_locked(symbol):
                continue

            pred = preds.get(symbol)
            if pred is None or pred < self.trade_threshold:
                continue

            quote = self.get_quote(symbol)
            if quote is None:
                continue

            if not should_trade(pred, quote['spread_pct']):
                continue

            # Winner's curse filter
            snapshot = snapshots.get(symbol, {})
            sma20 = snapshot.get('SMA_20')
            atr = snapshot.get('ATR')
            if sma20 and atr and quote['midpoint'] > sma20 + 2 * atr:
                required = self.trade_threshold * 1.5
                if pred < required:
                    logger.info("%s: Winner's curse filter, need %.2f got %.4f",
                                symbol, required, pred)
                    continue

            # Correlation check
            if self.corr_matrix and self.positions:
                allowed, avg_corr = check_portfolio_correlation(
                    list(self.positions.keys()), symbol, self.corr_matrix)
                if not allowed:
                    continue

            # Macro regime halt
            if self.macro_regime and self.macro_regime.should_halt_stocks:
                logger.info("%s: Halted by VIX > 35", symbol)
                continue

            # VIX > 25: block risky entries, allow safe-havens
            if self.macro_regime and self.macro_regime.should_block_risky_entries:
                from stock_config import SAFE_HAVEN_SYMBOLS
                if symbol not in SAFE_HAVEN_SYMBOLS:
                    logger.info("%s: Blocked — VIX > 25 defensive (non-safe-haven)", symbol)
                    continue

            # Compute position size
            sized_notional = self._compute_position_size(symbol, pred, quote)

            # Sentiment gate
            gate, gate_reasons = sentiment_gate(symbol, 'stock')
            if gate <= 0:
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "pred_return": pred,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons})
                continue
            sized_notional = int(sized_notional * gate)

            # LLM gate
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            llm_reason = llm_info.get('r', '')
            if llm_s < LLM_VETO_THRESHOLD:
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "llm_veto",
                              "pred_return": pred,
                              "llm_multiplier": llm_s, "llm_reasoning": llm_reason})
                continue
            llm_mult = 0.5 + llm_s
            sized_notional = int(sized_notional * llm_mult)

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
                    log_decision({"symbol": symbol, "action": "buy",
                                  "pred_return": pred,
                                  "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                                  "llm_multiplier": llm_mult, "llm_score": llm_s,
                                  "llm_reasoning": llm_reason,
                                  "final_notional": sized_notional,
                                  "skip_reason": None})
                    self.last_trade_time[symbol] = datetime.datetime.now()
                    current_exposure += qty * fill_price

                    # After fill confirmation
                    current_exposure = self._get_current_exposure()
                    if current_exposure > self.MAX_EXPOSURE:
                        logger.warning("[EXPOSURE] Exceeded cap after fill: $%.0f > $%.0f",
                                       current_exposure, self.MAX_EXPOSURE)
                        break  # Stop placing more orders this cycle
            time.sleep(0.5)

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
                        del self.positions[symbol]
                        self.last_trade_time[symbol] = datetime.datetime.now()
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
                    self.api.cancel_order(info.stop_order_id)
                    time.sleep(0.5)
                    trail_order = self.api.submit_order(
                        symbol=symbol, qty=int(info.qty), side='sell',
                        type='trailing_stop',
                        trail_percent=round(trail_pct * 100, 1),
                        time_in_force='day',
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
