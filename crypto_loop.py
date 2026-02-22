"""24/7 crypto trading loop — subclass of BaseTradingLoop.

Runs continuously (crypto markets never close):
  1. Fetch predictions for all symbols in parallel
  2. Check stop-loss / trailing stop / take-profit on open positions
  3. Sell positions where the model signals weakness
  4. Buy symbols where the model signals strength (sentiment-gated)
  5. Sleep and repeat

Enhanced with GARCH volatility, macro regime, correlation-aware sizing,
Kelly criterion, HMM regime detection, and stablecoin contagion detection.
"""

import json
import datetime
from pathlib import Path

from base_loop import BaseTradingLoop
from order_utils import (
    get_crypto_quote, place_limit_order, manage_order_lifecycle,
)
from market_data import fetch_bars_alpaca, fetch_crypto_volume
from sentiment import get_fear_greed, get_recent_headlines
from stock_config import CRYPTO_SYMBOLS
from log_config import get_logger

logger = get_logger(__name__)

_PRED_CACHE_FILE = Path(__file__).resolve().parent / "crypto_predictions.json"


class CryptoLoop(BaseTradingLoop):
    """24/7 crypto trading loop."""

    NOTIONAL_PER_SYMBOL = 1000
    MAX_NOTIONAL_PER_SYMBOL = 3000
    ORDER_TIMEOUT = 30
    LOOP_INTERVAL = 30
    COOLDOWN_MINUTES = 60
    MAX_PREDICTION_WORKERS = 5
    LLM_INTERVAL_SEC = 600
    CIRCUIT_BREAKER_PCT = 0.05

    # ATR stops (crypto-specific)
    ATR_STOP_MULTIPLIER = 2.0
    ATR_TRAIL_MULTIPLIER = 1.5
    ATR_TRAIL_ACTIVATE_PCT = 0.01
    ATR_STOP_FLOOR_PCT = 0.03
    ATR_STOP_CEIL_PCT = 0.10
    TAKE_PROFIT_RR = 3.0
    TAKE_PROFIT_CEIL_PCT = 0.25
    STOP_LOSS_PCT = 0.04
    TRAIL_PCT = 0.03

    def get_symbol_universe(self) -> list[str]:
        return CRYPTO_SYMBOLS

    def check_market_hours(self) -> bool:
        return True  # Crypto markets never close

    def get_asset_type(self) -> str:
        return 'crypto'

    def get_quote(self, symbol: str) -> dict | None:
        return get_crypto_quote(self.api, symbol)

    def place_buy_order(self, symbol, notional, quote, stop_price=None, tp_price=None):
        order = place_limit_order(self.api, symbol, 'buy', notional, quote)
        if order is None:
            return None
        result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                        fallback_to_market=True)
        return result

    def place_sell_order(self, symbol, qty, quote) -> bool:
        if quote is not None:
            try:
                order = self.api.submit_order(
                    symbol=symbol, qty=qty, side='sell', type='limit',
                    limit_price=round(quote['midpoint'] - quote['midpoint'] * 0.0005, 4),
                    time_in_force='gtc',
                )
                result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                                fallback_to_market=True)
                return result and getattr(result, 'status', None) == 'filled'
            except Exception as e:
                logger.error("%s: Sell error: %s", symbol, e)
        else:
            try:
                self.api.submit_order(symbol=symbol, qty=qty,
                                      side='sell', type='market', time_in_force='gtc')
                return True
            except Exception as e:
                logger.error("%s: Market sell error: %s", symbol, e)
        return False

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
            with open(_PRED_CACHE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("[CACHE] Error writing crypto prediction cache: %s", e)

    def _get_predictions(self, benchmark_close):
        """Override to add crypto volume injection and cache writing."""
        preds, snapshots = super()._get_predictions(benchmark_close)

        # Write prediction cache for GUI
        self.write_prediction_cache(preds)

        # Fetch real crypto volume (Alpaca reports zero)
        try:
            vol_ratios = fetch_crypto_volume(self.get_symbol_universe())
            for sym, ratio in vol_ratios.items():
                if sym in snapshots:
                    snapshots[sym]['Volume_Ratio'] = ratio
                else:
                    snapshots[sym] = {'Volume_Ratio': ratio}
        except Exception as e:
            logger.debug("[VOLUME] CryptoCompare error: %s", e)

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
