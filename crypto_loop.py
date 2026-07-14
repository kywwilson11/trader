"""24/7 crypto trading loop — subclass of BaseTradingLoop.

Runs continuously (crypto markets never close). Each cycle
(base_loop._run_one_cycle):
  1. Handle remote flatten requests + circuit-breaker check
  2. Check stop-loss / trailing stop / take-profit on open positions
  3. Fetch predictions for all symbols in parallel
  4. Sell positions where the model signals weakness (+ LLM veto sells)
  5. Buy symbols where the model signals strength (sentiment-gated)
  6. Sleep and repeat

Enhanced with GARCH volatility, macro regime, correlation-aware sizing,
Kelly criterion, HMM regime detection, and stablecoin contagion detection.
"""

import os
import json
import datetime
from pathlib import Path

from base_loop import BaseTradingLoop
from order_utils import (
    get_crypto_quote, manage_order_lifecycle,
)
from market_data import fetch_bars_alpaca
from sentiment import get_fear_greed, get_recent_headlines
from stock_config import CRYPTO_SYMBOLS
from log_config import get_logger

logger = get_logger(__name__)

_PRED_CACHE_FILE = Path(__file__).resolve().parent / "crypto_predictions.json"


class CryptoLoop(BaseTradingLoop):
    """24/7 crypto trading loop.

    Notional/timeout/interval/worker/LLM/breaker settings inherit the
    BaseTradingLoop defaults.
    """

    # ATR stops — values come from strategy_config so the backtester
    # validates the SAME policy the bot trades
    from strategy_config import CRYPTO_POLICY as _P
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
        self._resting_stop_px: dict[str, float] = {}

    def _extra_tilt(self, symbol: str) -> float:
        """Perp funding-rate positioning tilt (crowded-long de-risk).

        BIS/Management Science 'Crypto Carry': extreme positive funding
        marks crowded longs that precede crashes. Sourced live from OKX
        public endpoints (Binance live REST is geo-blocked from US IPs).
        """
        try:
            from funding import funding_tilt
            return funding_tilt(symbol)
        except Exception as e:
            # Fail-open by design (advisory input), but never SILENTLY:
            # funding_tilt only logs when it tilts, so without this line a
            # broken feed disables the de-risk with zero log evidence.
            logger.debug("[FUNDING] tilt unavailable for %s: %s", symbol, e)
            return 1.0

    def get_symbol_universe(self) -> list[str]:
        return CRYPTO_SYMBOLS

    def check_market_hours(self) -> bool:
        return True  # Crypto markets never close

    def get_asset_type(self) -> str:
        return 'crypto'

    def get_quote(self, symbol: str) -> dict | None:
        return get_crypto_quote(self.api, symbol)

    def place_buy_order(self, symbol, notional, quote, stop_price=None, tp_price=None):
        # Satisfies the base abstractmethod only — no crypto code path calls
        # it (stock_loop is the sole caller of its own override). Crypto
        # entries flow exclusively through _execute_entry_order.
        return self._execute_entry_order(symbol, notional, quote)[0]

    def _execute_entry_order(self, symbol, notional, quote):
        """Crypto entries try the maker ladder first (15bps maker vs
        25bps taker, plus the half-spread when joined passively). Falls
        back to the marketable path when disabled."""
        from strategy_config import MAKER_ENTRIES_ENABLED, MAKER_STAGE_TIMEOUT
        if not MAKER_ENTRIES_ENABLED:
            return super()._execute_entry_order(symbol, notional, quote)
        from order_utils import place_maker_buy
        return place_maker_buy(self.api, symbol, notional,
                               lambda: self.get_quote(symbol),
                               stage_timeout=MAKER_STAGE_TIMEOUT)

    def place_sell_order(self, symbol, qty, quote):
        """Sell and confirm. Returns the FILLED order object, or None."""
        from order_utils import make_client_order_id
        if quote is not None:
            try:
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side='sell', type='limit',
                    limit_price=round(quote['midpoint'] - quote['midpoint'] * 0.0005, 4),
                    time_in_force='gtc',
                    client_order_id=make_client_order_id('csell'),
                )
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=True)
                if result is not None and getattr(result, 'status', None) == 'filled':
                    return result
            except Exception as e:
                logger.error("%s: Sell error: %s", symbol, e)
        else:
            try:
                order = self.api.submit_order(symbol=symbol, qty=qty,
                                              side='sell', type='market', time_in_force='gtc',
                                              client_order_id=make_client_order_id('csell'))
                # Confirm the market sell actually filled instead of
                # assuming success at submit time
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=False)
                if result is not None and getattr(result, 'status', None) == 'filled':
                    return result
            except Exception as e:
                logger.error("%s: Market sell error: %s", symbol, e)
        return None

    def get_benchmark_close(self):
        try:
            btc_df = fetch_bars_alpaca(self.api, 'BTC/USD')
            if btc_df is not None:
                return btc_df['Close']
        except Exception as e:
            logger.error("[BTC] Benchmark error: %s", e)
        return None

    def get_headlines(self, symbol: str) -> list[str]:
        return get_recent_headlines(symbol, 'crypto')

    def flatten_before_close(self):
        pass  # Crypto doesn't flatten

    # ------------------------------------------------------------------
    # Resting server-side protection (GTC stop_limit)
    #
    # Alpaca crypto supports market/limit/STOP_LIMIT with gtc — so every
    # position gets a RESTING stop that survives process death, OOM kills,
    # and network loss. Historical motivation: May 2021 (BTC -31% intraday
    # with major venues down) and Aug 2024 (BTC -11% overnight) would have
    # realized -15-25% on a dead 30s software loop instead of the designed
    # ATR stop. The limit leg sits 2% below the stop so slippage is BOUNDED
    # but a gap can't leave an unfilled naked stop-market chase.
    # ------------------------------------------------------------------

    RESTING_STOP_LIMIT_GAP = 0.02   # limit 2% below stop trigger
    RESTING_STOP_MIN_IMPROVE = 1.01  # re-place only if 1%+ tighter

    @staticmethod
    def _round_px(p: float) -> float:
        return round(p, 6 if p < 1 else 4)

    def _stop_distance_for(self, pos) -> float:
        if pos.entry_atr is not None and pos.entry_price > 0:
            raw = (pos.entry_atr * self.ATR_STOP_MULTIPLIER) / pos.entry_price
            return max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw))
        return self.STOP_LOSS_PCT

    def _place_resting_stop(self, symbol, pos, stop_price: float) -> bool:
        from order_utils import make_client_order_id
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=pos.qty, side='sell',
                type='stop_limit',
                stop_price=self._round_px(stop_price),
                limit_price=self._round_px(stop_price * (1 - self.RESTING_STOP_LIMIT_GAP)),
                time_in_force='gtc',
                client_order_id=make_client_order_id('cstop'),
            )
            pos.stop_order_id = order.id
            self._resting_stop_px[symbol] = stop_price
            logger.info("[RESTING-STOP] %s: GTC stop_limit @ $%s (limit $%s)",
                        symbol, self._round_px(stop_price),
                        self._round_px(stop_price * (1 - self.RESTING_STOP_LIMIT_GAP)))
            return True
        except Exception as e:
            logger.error("[RESTING-STOP] %s: placement failed: %s "
                         "(software stops still active)", symbol, e)
            pos.stop_order_id = None
            return False

    def _after_entry_protection(self, symbol, pos):
        stop_price = pos.entry_price * (1 - self._stop_distance_for(pos))
        self._place_resting_stop(symbol, pos, stop_price)

    def _replace_protective_stops(self):
        """Re-place resting stops for every reconstructed crypto position
        (startup cancels this bot's working orders first)."""
        for symbol, pos in self.positions.items():
            # Anchor to the persisted HWM so restarts don't widen a
            # trail that had already tightened
            anchor = max(pos.entry_price, pos.high_water_mark)
            stop_price = anchor * (1 - self._stop_distance_for(pos))
            self._place_resting_stop(symbol, pos, stop_price)

    def _maybe_update_resting_stop(self, symbol, pos, desired_stop_price):
        """Tighten the resting stop as the software trail rises (churn-
        limited: only when 1%+ tighter than the current resting level)."""
        current = self._resting_stop_px.get(symbol)
        if pos.stop_order_id and current is not None \
                and desired_stop_price < current * self.RESTING_STOP_MIN_IMPROVE:
            return
        if pos.stop_order_id is None and current is None:
            # No resting protection at all (earlier placement failed) — place
            self._place_resting_stop(symbol, pos, desired_stop_price)
            return
        from order_utils import cancel_orders_for_symbol
        if not cancel_orders_for_symbol(self.api, symbol, timeout=5):
            return  # retry next cycle
        pos.stop_order_id = None
        self._resting_stop_px.pop(symbol, None)
        self._place_resting_stop(symbol, pos, desired_stop_price)

    def _manage_stops(self):
        # Exits (signal sells, LLM vetoes, server fills, flattens) drop
        # positions in base_loop without touching this bookkeeping — prune
        # stale levels so a re-entry whose stop placement failed doesn't
        # take the cancel-and-replace branch on a leftover price.
        for symbol in list(self._resting_stop_px):
            if symbol not in self.positions:
                self._resting_stop_px.pop(symbol, None)
        super()._manage_stops()

    def write_prediction_cache(self, preds, **kwargs):
        try:
            data = {}
            for sym in sorted(preds):
                pred = preds[sym]
                if pred is not None and pred > self.trade_threshold:
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
            # Atomic write (tmp + rename): the GUI polls this file, and an
            # in-place write can hand it a torn read
            tmp = _PRED_CACHE_FILE.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _PRED_CACHE_FILE)
        except Exception as e:
            logger.error("[CACHE] Error writing crypto prediction cache: %s", e)

    def _get_predictions(self, benchmark_close):
        """Override to add cache writing and periodic sentiment logging."""
        preds, snapshots = super()._get_predictions(benchmark_close)

        # Write prediction cache for GUI
        self.write_prediction_cache(preds)

        # Log Fear & Greed periodically
        if self.cycle % 10 == 1:
            fng = get_fear_greed()
            if fng is not None:
                logger.info("[SENTIMENT] Fear & Greed: %s (%s)",
                            fng['value'], fng['label'])

        return preds, snapshots


def run_crypto_bot():
    loop = CryptoLoop()
    loop.run()


if __name__ == "__main__":
    run_crypto_bot()
