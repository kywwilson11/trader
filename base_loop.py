"""Template Method base class for trading loops.

Extracts the shared skeleton of crypto_loop.py and stock_loop.py into a
reusable base class. Subclasses override only asset-specific behavior:
symbol universe, market hours, order types, and flatten logic.

This eliminates ~400 lines of duplicated code and ensures both loops
stay in sync as new features are added.
"""

import json
import os
import time
import datetime
import gc
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from log_config import get_logger
from types_mod import Position, MacroRegime
from order_utils import (
    manage_order_lifecycle, get_all_positions, should_trade,
    cancel_all_open_orders, reconstruct_positions,
    check_circuit_breaker, emergency_flatten,
)
from predict_now import load_models
from trading_utils import (
    get_api, get_model_mtime, choose_inference_device, cooldown_ok,
    predict_symbol, kelly_position_size, compute_kelly_fraction,
    LLM_VETO_THRESHOLD, THERMAL_THROTTLE_TEMP, TEMP_LOG_EVERY_N_CYCLES,
)
from hw_monitor import get_gpu_temp
from sentiment import sentiment_gate
from llm_config import load_llm_config
from llm_analyst import analyze_trades
from fundamentals import get_fundamentals, format_fundamentals_for_llm
from trade_journal import log_decision
from trade_memory import record_trade
from macro_indicators import get_macro_regime
from volatility import get_cached_sigma, compute_vol_adjusted_size, get_garch_stop
from portfolio import (
    get_returns_for_symbols, compute_correlation_matrix,
    check_portfolio_correlation, get_correlation_sizing_factor,
)
from regime_detector import get_cached_regime

logger = get_logger(__name__)


class BaseTradingLoop(ABC):
    """Abstract base for crypto and stock trading loops."""

    # --- Configuration (override in subclasses) ---
    NOTIONAL_PER_SYMBOL: float = 1000
    MAX_NOTIONAL_PER_SYMBOL: float = 3000
    ORDER_TIMEOUT: int = 30
    LOOP_INTERVAL: int = 30
    COOLDOWN_MINUTES: int = 60
    MAX_PREDICTION_WORKERS: int = 5
    LLM_INTERVAL_SEC: int = 600
    CIRCUIT_BREAKER_PCT: float = 0.05
    MODEL_PREFIX: str = ''

    # ATR stops
    ATR_STOP_MULTIPLIER: float = 2.0
    ATR_TRAIL_MULTIPLIER: float = 1.5
    ATR_TRAIL_ACTIVATE_PCT: float = 0.01
    ATR_STOP_FLOOR_PCT: float = 0.03
    ATR_STOP_CEIL_PCT: float = 0.10
    TAKE_PROFIT_RR: float = 3.0
    TAKE_PROFIT_CEIL_PCT: float = 0.25
    STOP_LOSS_PCT: float = 0.04  # fallback fixed
    TRAIL_PCT: float = 0.03
    HARD_STOP_LOCKOUT_HOURS: int = 24  # cooldown after hard stop

    def __init__(self):
        self.api = get_api()
        self.model = None
        self.config = {}
        self.scaler_X = None
        self.feature_cols = None
        self.trade_threshold = 0.15
        self.positions: dict[str, Position] = {}
        self.last_trade_time: dict[str, datetime.datetime] = {}
        self.hard_stop_lockout: dict[str, datetime.datetime] = {}
        self._lockout_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'hard_stop_lockout.json')
        self._load_hard_stop_lockout()
        self.llm_scores: dict = {}
        self._last_llm_time = 0.0
        self.model_mtime = 0
        self.cycle = 0
        self.macro_regime: MacroRegime | None = None
        self.corr_matrix: dict = {}
        self._equity: float = 100_000
        from stock_config import LEVERAGED_ETFS
        self._leveraged_etfs = LEVERAGED_ETFS

    # --- Abstract methods (subclasses must implement) ---

    @abstractmethod
    def get_symbol_universe(self) -> list[str]:
        """Return list of tradeable symbols."""

    @abstractmethod
    def check_market_hours(self) -> bool:
        """Return True if market is open for trading."""

    @abstractmethod
    def get_asset_type(self) -> str:
        """Return 'crypto' or 'stock'."""

    @abstractmethod
    def get_quote(self, symbol: str) -> dict | None:
        """Get bid/ask quote for a symbol."""

    @abstractmethod
    def place_buy_order(self, symbol: str, notional_or_qty, quote: dict,
                        stop_price: float | None = None,
                        tp_price: float | None = None) -> object | None:
        """Place a buy order. Returns order object or None."""

    @abstractmethod
    def place_sell_order(self, symbol: str, qty, quote: dict | None) -> bool:
        """Place a sell order. Returns True if filled."""

    @abstractmethod
    def get_benchmark_close(self):
        """Fetch benchmark close prices (BTC for crypto, SPY for stocks)."""

    @abstractmethod
    def get_headlines(self, symbol: str) -> list[str]:
        """Get recent news headlines for a symbol."""

    @abstractmethod
    def flatten_before_close(self):
        """Flatten positions before market close (stocks only, no-op for crypto)."""

    @abstractmethod
    def write_prediction_cache(self, preds: dict, **kwargs):
        """Write predictions to JSON for GUI consumption."""

    # --- Template Method: main loop ---

    def run(self):
        """Main trading loop skeleton."""
        self._load_models()
        cancel_all_open_orders(self.api)
        self._reconstruct_positions()
        self._print_startup()

        while True:
            self.cycle += 1

            if not self.check_market_hours():
                if self.cycle == 1 or self.cycle % 20 == 0:
                    logger.info("[WAIT] Market closed. Next check in %ds...", self.LOOP_INTERVAL)
                time.sleep(self.LOOP_INTERVAL)
                continue

            logger.info("--- CYCLE %d: %s ---", self.cycle,
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            # Pre-trade checks
            if self._circuit_breaker_check():
                continue

            self.flatten_before_close()

            # Hot-reload model
            self._hot_reload_check()

            # Update macro regime (every 10 cycles to save API calls)
            if self.cycle % 10 == 1:
                self._update_macro_regime()
                self._update_equity()
                self._update_correlations()

            # Log GPU temp periodically
            if self.cycle % TEMP_LOG_EVERY_N_CYCLES == 0:
                temp = get_gpu_temp()
                if temp is not None:
                    logger.info("[HW] GPU temp: %.0fC", temp)

            # Stop-loss management
            self._manage_stops()

            # Predictions
            benchmark = self.get_benchmark_close()
            if benchmark is None:
                logger.warning("[BENCHMARK] Benchmark data unavailable — predictions will lack relative strength")
            preds, snapshots = self._get_predictions(benchmark)

            # LLM analysis (throttled)
            self._run_llm_analysis(preds)

            # Sell bearish positions
            self._execute_sells(preds)

            # LLM veto sells
            self._execute_llm_veto_sells()

            # Buy
            self._execute_buys(preds, snapshots)

            # Thermal throttling
            self._sleep()

    # --- Shared implementations ---

    def _load_models(self):
        """Load ML prediction model."""
        logger.info("Loading prediction models...")
        try:
            inference_device = choose_inference_device()
            self.model, self.config, self.scaler_X, self.feature_cols = \
                load_models(inference_device, prefix=self.MODEL_PREFIX)
            self.trade_threshold = self.config.get('trade_threshold', 0.15)
            logger.info("Model loaded (trade_threshold=%.2f)", self.trade_threshold)
        except FileNotFoundError:
            logger.warning("Model files not found. Running without prediction gating.")

        model_file = f'{self.MODEL_PREFIX}_model_v2.pth' if self.MODEL_PREFIX else 'model_v2.pth'
        self.model_mtime = get_model_mtime(model_file)

    def _reconstruct_positions(self):
        """Rebuild positions from API (survive restarts)."""
        from market_data import get_live_atr
        symbols = self.get_symbol_universe()
        raw_positions = reconstruct_positions(self.api, symbols, asset_type=self.get_asset_type())
        for sym, info in raw_positions.items():
            entry_atr = get_live_atr(self.api, sym, asset_type=self.get_asset_type())
            tp_price = None
            if entry_atr is not None and info['entry_price'] > 0:
                raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / info['entry_price']
                stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
                tp_dist = min(self.TAKE_PROFIT_CEIL_PCT, stop_dist * self.TAKE_PROFIT_RR)
                tp_price = info['entry_price'] * (1 + tp_dist)
            self.positions[sym] = Position(
                qty=info['qty'],
                entry_price=info['entry_price'],
                high_water_mark=info['high_water_mark'],
                entry_atr=entry_atr,
                take_profit_price=tp_price,
            )

        if self.positions:
            logger.info("Existing positions: %s", ', '.join(self.positions))

    def _print_startup(self):
        """Print startup configuration."""
        symbols = self.get_symbol_universe()
        logger.info("--- %s BOT STARTED ---", self.get_asset_type().upper())
        logger.info("Symbols: %d | Notional: $%d | Loop: %ds | Cooldown: %d min",
                     len(symbols), self.NOTIONAL_PER_SYMBOL, self.LOOP_INTERVAL,
                     self.COOLDOWN_MINUTES)
        logger.info("Risk management: GARCH + Macro regime + Correlation + Kelly")

        # Pre-load cached LLM scores from disk
        try:
            from llm_analyst import load_analysis
            data = load_analysis()
            section = data.get(self.get_asset_type(), {})
            if section:
                self.llm_scores = section
                logger.info("[LLM] Loaded %d cached scores from disk", len(section))
        except Exception:
            pass

    def _circuit_breaker_check(self) -> bool:
        """Check circuit breaker. Returns True if tripped (caller should continue)."""
        try:
            tripped, dd = check_circuit_breaker(self.api, max_drawdown_pct=self.CIRCUIT_BREAKER_PCT)
        except Exception as e:
            logger.error("[CIRCUIT BREAKER] API error: %s", e)
            return False

        if tripped:
            logger.warning("[CIRCUIT BREAKER] Drawdown %.1f%% >= %.0f%%, flattening!",
                           dd * 100, self.CIRCUIT_BREAKER_PCT * 100)
            emergency_flatten(self.api)
            self.positions.clear()
            logger.info("[CIRCUIT BREAKER] Sleeping 1 hour...")
            time.sleep(3600)
            return True
        return False

    def _hot_reload_check(self):
        """Check if model files changed and reload."""
        model_file = f'{self.MODEL_PREFIX}_model_v2.pth' if self.MODEL_PREFIX else 'model_v2.pth'
        new_mtime = get_model_mtime(model_file)
        if new_mtime != self.model_mtime:
            logger.info("[HOT-RELOAD] Model files changed, reloading...")
            try:
                inference_device = choose_inference_device()
                self.model, self.config, self.scaler_X, self.feature_cols = \
                    load_models(inference_device, prefix=self.MODEL_PREFIX)
                self.trade_threshold = self.config.get('trade_threshold', 0.15)
                self.model_mtime = new_mtime
                logger.info("[HOT-RELOAD] Reloaded (threshold=%.2f)", self.trade_threshold)
            except Exception as e:
                logger.error("[HOT-RELOAD] Failed: %s", e)

    def _update_macro_regime(self):
        """Fetch and cache current macro regime."""
        try:
            self.macro_regime = get_macro_regime(self.api, self.get_asset_type())
            logger.info("[MACRO] Regime: %s (sizing=%.2fx, stops=%.2fx)",
                        self.macro_regime.regime_label,
                        self.macro_regime.sizing_mult,
                        self.macro_regime.stop_mult)

            # Emergency stablecoin flatten for crypto
            if self.macro_regime.stablecoin_alert and self.get_asset_type() == 'crypto':
                if self.macro_regime.sizing_mult == 0:
                    logger.warning("[CONTAGION] Stablecoin emergency! Flattening crypto...")
                    emergency_flatten(self.api)
                    self.positions.clear()
        except Exception as e:
            logger.debug("[MACRO] Regime update failed: %s", e)

    def _update_equity(self):
        """Update cached equity."""
        try:
            acct = self.api.get_account()
            self._equity = float(acct.equity)
        except Exception:
            pass

    def _update_correlations(self):
        """Update correlation matrix for portfolio management."""
        try:
            symbols = self.get_symbol_universe()
            returns_dict = get_returns_for_symbols(self.api, symbols, self.get_asset_type())
            if returns_dict:
                self.corr_matrix = compute_correlation_matrix(returns_dict)
        except Exception as e:
            logger.debug("[PORTFOLIO] Correlation update failed: %s", e)

    def _manage_stops(self):
        """Check stop-loss, trailing stop, and take-profit on open positions."""
        for symbol in list(self.positions):
            quote = self.get_quote(symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            pos = self.positions[symbol]
            pos.high_water_mark = max(pos.high_water_mark, current_price)
            hwm = pos.high_water_mark
            entry_price = pos.entry_price

            # Determine stop distances
            entry_atr = pos.entry_atr
            if entry_atr is not None and entry_price > 0:
                raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / entry_price
                stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
                raw_trail_dist = (entry_atr * self.ATR_TRAIL_MULTIPLIER) / hwm if hwm > 0 else stop_dist
                trail_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_trail_dist))
            else:
                stop_dist = self.STOP_LOSS_PCT
                trail_dist = self.TRAIL_PCT

            # Apply macro regime stop tightening
            if self.macro_regime and self.macro_regime.stop_mult < 1.0:
                stop_dist *= self.macro_regime.stop_mult
                trail_dist *= self.macro_regime.stop_mult

            stop_reason = None
            if current_price <= entry_price * (1 - stop_dist):
                stop_reason = 'hard_stop'
            elif pos.take_profit_price and current_price >= pos.take_profit_price:
                stop_reason = 'take_profit'
            elif (hwm >= entry_price * (1 + self.ATR_TRAIL_ACTIVATE_PCT)
                  and current_price <= hwm * (1 - trail_dist)):
                stop_reason = 'trailing'

            if stop_reason:
                logger.info("[STOP] %s: %s at $%.4f (entry=$%.4f, hwm=$%.4f, "
                            "stop_d=%.1f%%, trail_d=%.1f%%, reason=%s)",
                            symbol, stop_reason, current_price, entry_price, hwm,
                            stop_dist * 100, trail_dist * 100, stop_reason)
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=pos.qty,
                        side='sell', type='market', time_in_force='gtc',
                    )
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    llm_info = self.llm_scores.get(symbol, {})
                    record_trade(symbol, 'sell', entry_price, current_price,
                                 pnl_pct, llm_score=llm_info.get('s'),
                                 reasoning=llm_info.get('r', ''),
                                 exit_reason=stop_reason)
                    del self.positions[symbol]
                    self.last_trade_time[symbol] = datetime.datetime.now()
                    if stop_reason == 'hard_stop':
                        self.hard_stop_lockout[symbol] = datetime.datetime.now()
                        self._save_hard_stop_lockout()
                        logger.info("[LOCKOUT] %s: %dh lockout after hard stop",
                                    symbol, self.HARD_STOP_LOCKOUT_HOURS)
                except Exception as e:
                    err_msg = str(e).lower()
                    logger.error("[STOP] %s: Sell error: %s", symbol, e)
                    # Position no longer exists at broker — remove from tracking
                    if ('insufficient qty' in err_msg
                            or 'position does not exist' in err_msg
                            or 'not found' in err_msg
                            or 'available: 0' in err_msg):
                        logger.warning("[DESYNC] %s: Position gone at broker, removing from tracking", symbol)
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        llm_info = self.llm_scores.get(symbol, {})
                        record_trade(symbol, 'sell', entry_price, current_price,
                                     pnl_pct, llm_score=llm_info.get('s'),
                                     reasoning='position desync — broker qty=0',
                                     exit_reason='desync')
                        if symbol in self.positions:
                            del self.positions[symbol]
                        self.last_trade_time[symbol] = datetime.datetime.now()

    def _get_predictions(self, benchmark_close) -> tuple[dict, dict]:
        """Get predictions for all symbols in parallel."""
        preds = {}
        snapshots = {}
        if self.model is None:
            return preds, snapshots

        inference_device = choose_inference_device()
        symbols = self.get_symbol_universe()

        with ThreadPoolExecutor(max_workers=self.MAX_PREDICTION_WORKERS) as executor:
            futures = {}
            for symbol in symbols:
                f = executor.submit(
                    predict_symbol, self.api, symbol,
                    self.model, self.config, self.scaler_X, self.feature_cols,
                    inference_device, asset_type=self.get_asset_type(),
                    benchmark_close=benchmark_close,
                    return_snapshot=True,
                )
                futures[f] = symbol

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    sym, pred, snapshot = future.result()
                    if pred is not None:
                        preds[sym] = pred
                    if snapshot is not None:
                        snapshots[sym] = snapshot
                except Exception as e:
                    logger.error("%s: Prediction error: %s", symbol, e)

        return preds, snapshots

    def _run_llm_analysis(self, preds: dict):
        """Run LLM pre-trade analysis if interval elapsed."""
        now_ts = time.time()
        if now_ts - self._last_llm_time < self.LLM_INTERVAL_SEC:
            # Even if interval hasn't elapsed, check for stale disk cache
            # and warn if scores are being used from hours-old analysis
            self._check_llm_staleness()
            return

        llm_cfg = load_llm_config()
        if not llm_cfg.get("enabled"):
            return

        candidates = self._build_llm_candidates(preds)
        if not candidates:
            return

        new_scores = analyze_trades(
            candidates, self.get_asset_type(), equity=self._equity,
            positions=list(self.positions.keys()),
            model_config=self.config,
        )
        if new_scores:
            self.llm_scores = new_scores
            self._last_llm_time = now_ts
            logger.info("[LLM] Scores: %s",
                        ", ".join(f"{s}={v.get('s', 0.5):.2f}" for s, v in self.llm_scores.items()))

    def _check_llm_staleness(self):
        """Warn and force refresh if LLM scores on disk are stale (> 2 hours)."""
        if not self.llm_scores:
            return
        from llm_analyst import load_analysis
        from datetime import datetime, timezone
        try:
            data = load_analysis()
            section = data.get(self.get_asset_type(), {})
            if not section:
                return
            # Check oldest timestamp in section
            oldest_ts = None
            for sym, entry in section.items():
                ts_str = entry.get('timestamp', '')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if oldest_ts is None or ts < oldest_ts:
                            oldest_ts = ts
                    except ValueError:
                        pass
            if oldest_ts is not None:
                age_hours = (datetime.now(timezone.utc) - oldest_ts).total_seconds() / 3600
                if age_hours > 2:
                    logger.warning("[LLM] Scores are %.1fh stale — forcing refresh next cycle",
                                   age_hours)
                    self._last_llm_time = 0  # Force refresh
        except Exception:
            pass

    def _build_llm_candidates(self, preds: dict) -> list[dict]:
        """Build candidate list for LLM analysis. Override for stock-specific fundamentals."""
        candidates = []
        for symbol in self.get_symbol_universe():
            fund = get_fundamentals(symbol, self.get_asset_type())
            fund_text = format_fundamentals_for_llm(symbol, fund)
            headlines = self.get_headlines(symbol)
            candidates.append({
                'symbol': symbol,
                'pred_return': preds.get(symbol),
                'fundamentals_text': fund_text,
                'news_headlines': headlines,
            })
        return candidates

    def _execute_sells(self, preds: dict):
        """Sell bearish positions."""
        for symbol in list(self.positions):
            pred = preds.get(symbol)
            if pred is not None and pred > -self.trade_threshold:
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue

            reason = f"pred={pred:+.4f}%" if pred is not None else "no prediction"
            logger.info("%s: SELLING (%s)", symbol, reason)

            quote = self.get_quote(symbol)
            if quote is None:
                logger.warning("%s: Skipping sell — quote unavailable", symbol)
                continue
            pos = self.positions[symbol]
            if self.place_sell_order(symbol, pos.qty, quote):
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

    def _execute_llm_veto_sells(self):
        """Sell positions with catastrophic LLM scores."""
        for symbol in list(self.positions):
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            if llm_s >= LLM_VETO_THRESHOLD:
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                logger.info("%s: LLM VETO (%.2f) but in cooldown", symbol, llm_s)
                continue

            logger.info("%s: LLM VETO SELL (%.2f — %s)", symbol, llm_s,
                        llm_info.get('r', ''))
            pos = self.positions[symbol]
            quote = self.get_quote(symbol)
            if self.place_sell_order(symbol, pos.qty, quote):
                pnl_pct = 0.0
                if quote:
                    pnl_pct = ((quote['midpoint'] - pos.entry_price) / pos.entry_price) * 100
                record_trade(symbol, 'sell', pos.entry_price,
                             quote['midpoint'] if quote else pos.entry_price,
                             pnl_pct, llm_score=llm_s,
                             reasoning=llm_info.get('r', ''),
                             exit_reason='llm_veto')
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

    def _compute_position_size(self, symbol: str, pred_return: float | None,
                               quote: dict) -> int:
        """Compute final position size with all adjustments.

        Applies: confidence scaling, Kelly criterion, GARCH vol targeting,
        macro regime, correlation, sentiment gate, LLM multiplier.
        """
        # Base: Kelly or fixed
        base = kelly_position_size(self.NOTIONAL_PER_SYMBOL, self._equity)

        # Confidence scaling
        if pred_return is not None and self.trade_threshold > 0.001:
            confidence = min(2.0, max(0.5, pred_return / self.trade_threshold))
        else:
            confidence = 1.0
        sized = base * confidence

        # Fetch bars once for both GARCH and HMM
        returns = None
        try:
            from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
            if self.get_asset_type() == 'crypto':
                df = fetch_bars_alpaca(self.api, symbol)
            else:
                df = fetch_stock_bars_alpaca(self.api, symbol)
            if df is not None and len(df) > 100:
                returns = df['Close'].pct_change().dropna().values * 100
        except Exception:
            pass

        # GARCH vol-targeted sizing
        if returns is not None:
            try:
                sigma = get_cached_sigma(symbol, returns)
                if sigma is not None:
                    sized = compute_vol_adjusted_size(sized, sigma)
            except Exception:
                pass

        # Macro regime multiplier
        if self.macro_regime:
            sized *= self.macro_regime.sizing_mult

        # Correlation-based reduction
        if self.corr_matrix and self.positions:
            corr_factor = get_correlation_sizing_factor(
                symbol, list(self.positions.keys()), self.corr_matrix)
            sized *= corr_factor

        # HMM regime multiplier
        if returns is not None and len(returns) > 200:
            try:
                regime = get_cached_regime(symbol, returns)
                sized *= regime['sizing_mult']
            except Exception:
                pass

        # Leveraged ETF scaling: divide by leverage factor
        leverage = self._leveraged_etfs.get(symbol, 1)
        if leverage > 1:
            sized /= leverage

        return max(1, int(sized))

    def _load_hard_stop_lockout(self):
        """Load hard-stop lockout state from disk (survive restarts)."""
        try:
            with open(self._lockout_file, 'r') as f:
                data = json.load(f)
            now = datetime.datetime.now().timestamp()
            for symbol, expiry_ts in data.items():
                if expiry_ts > now:
                    self.hard_stop_lockout[symbol] = datetime.datetime.fromtimestamp(
                        expiry_ts - self.HARD_STOP_LOCKOUT_HOURS * 3600)
            if self.hard_stop_lockout:
                logger.info("[LOCKOUT] Loaded %d lockout(s) from disk: %s",
                            len(self.hard_stop_lockout),
                            ', '.join(self.hard_stop_lockout.keys()))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except Exception as e:
            logger.warning("[LOCKOUT] Failed to load lockout file: %s", e)

    def _save_hard_stop_lockout(self):
        """Persist hard-stop lockout state to disk (atomic write)."""
        try:
            data = {}
            for symbol, lockout_time in self.hard_stop_lockout.items():
                expiry_ts = (lockout_time + datetime.timedelta(
                    hours=self.HARD_STOP_LOCKOUT_HOURS)).timestamp()
                data[symbol] = expiry_ts
            tmp = self._lockout_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self._lockout_file)
        except Exception as e:
            logger.warning("[LOCKOUT] Failed to save lockout file: %s", e)

    def _is_hard_stop_locked(self, symbol: str) -> bool:
        """Check if symbol is in hard-stop lockout period."""
        if symbol not in self.hard_stop_lockout:
            return False
        elapsed = (datetime.datetime.now() - self.hard_stop_lockout[symbol]).total_seconds()
        if elapsed >= self.HARD_STOP_LOCKOUT_HOURS * 3600:
            del self.hard_stop_lockout[symbol]
            self._save_hard_stop_lockout()
            return False
        return True

    def _execute_buys(self, preds: dict, snapshots: dict):
        """Buy bullish symbols with all risk checks."""
        symbols = self.get_symbol_universe()
        for symbol in symbols:
            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue

            if self._is_hard_stop_locked(symbol):
                continue

            # Position cap check
            if symbol in self.positions:
                existing_value = self.positions[symbol].qty * self.positions[symbol].entry_price
                if existing_value >= self.MAX_NOTIONAL_PER_SYMBOL:
                    continue

            pred_return = preds.get(symbol)
            quote = self.get_quote(symbol)

            # Prediction gate
            if pred_return is not None and quote is not None:
                if not should_trade(pred_return, quote['spread_pct']):
                    continue
                if pred_return < self.trade_threshold:
                    continue

            if quote is None:
                continue

            # Winner's curse filter: if price > SMA20 + 2*ATR, require higher threshold
            snapshot = snapshots.get(symbol, {})
            sma20 = snapshot.get('SMA_20')
            atr = snapshot.get('ATR')
            if sma20 and atr and quote['midpoint'] > sma20 + 2 * atr:
                required = self.trade_threshold * 1.5
                if pred_return is not None and pred_return < required:
                    logger.info("%s: Winner's curse filter (extended move), need %.2f got %.4f",
                                symbol, required, pred_return)
                    continue

            # Correlation check
            if self.corr_matrix and self.positions:
                allowed, avg_corr = check_portfolio_correlation(
                    list(self.positions.keys()), symbol, self.corr_matrix)
                if not allowed:
                    continue

            # Macro regime halt check
            if self.macro_regime and self.macro_regime.should_halt_stocks and self.get_asset_type() == 'stock':
                logger.info("%s: Halted by VIX > 35", symbol)
                continue

            # VIX > 25: block risky entries, allow safe-havens
            if (self.macro_regime and self.macro_regime.should_block_risky_entries
                    and self.get_asset_type() == 'stock'):
                from stock_config import SAFE_HAVEN_SYMBOLS
                if symbol not in SAFE_HAVEN_SYMBOLS:
                    logger.info("%s: Blocked — VIX > 25 defensive (non-safe-haven)", symbol)
                    continue

            # Compute position size with all adjustments
            sized_notional = self._compute_position_size(symbol, pred_return, quote)

            # Sentiment gate
            gate, gate_reasons = sentiment_gate(symbol, self.get_asset_type())
            if gate <= 0:
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "pred_return": pred_return,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons})
                continue
            sized_notional = int(sized_notional * gate)

            # LLM gate
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            llm_reason = llm_info.get('r', '')
            if llm_s < LLM_VETO_THRESHOLD:
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "llm_veto",
                              "pred_return": pred_return,
                              "llm_multiplier": llm_s, "llm_reasoning": llm_reason})
                continue
            llm_mult = 0.5 + llm_s
            sized_notional = int(sized_notional * llm_mult)

            # Order timing jitter (prevent pattern detection)
            import random
            time.sleep(random.uniform(0, 5))

            logger.info("%s: Sizing $%d (pred=%.4f)", symbol, sized_notional,
                        pred_return if pred_return else 0)

            self._place_and_track_buy(symbol, sized_notional, pred_return, quote,
                                      gate, gate_reasons, llm_s, llm_mult, llm_reason)
            time.sleep(1)

    def _place_and_track_buy(self, symbol, notional, pred_return, quote,
                             gate, gate_reasons, llm_s, llm_mult, llm_reason):
        """Place buy order and update position tracking. Override in subclasses for bracket orders."""
        from market_data import get_live_atr
        from order_utils import place_limit_order

        order = place_limit_order(self.api, symbol, 'buy', notional, quote)
        if order is None:
            return

        result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                        fallback_to_market=True)
        if result and getattr(result, 'status', None) == 'filled':
            from order_utils import verify_position
            pos = verify_position(self.api, symbol)
            if pos:
                fill_price = float(pos.avg_entry_price)
                total_qty = float(pos.qty)
                entry_atr = get_live_atr(self.api, symbol, asset_type=self.get_asset_type())

                tp_price = None
                if entry_atr is not None and fill_price > 0:
                    raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / fill_price
                    stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
                    tp_dist = min(self.TAKE_PROFIT_CEIL_PCT, stop_dist * self.TAKE_PROFIT_RR)
                    tp_price = fill_price * (1 + tp_dist)

                # Compute GARCH sigma for this position
                garch_sigma = None
                try:
                    from market_data import fetch_bars_alpaca, fetch_stock_bars_alpaca
                    if self.get_asset_type() == 'crypto':
                        df = fetch_bars_alpaca(self.api, symbol)
                    else:
                        df = fetch_stock_bars_alpaca(self.api, symbol)
                    if df is not None and len(df) > 100:
                        returns = df['Close'].pct_change().dropna().values * 100
                        garch_sigma = get_cached_sigma(symbol, returns)
                except Exception:
                    pass

                is_add = symbol in self.positions
                hwm = fill_price
                if is_add:
                    hwm = max(self.positions[symbol].high_water_mark, fill_price)

                self.positions[symbol] = Position(
                    qty=total_qty,
                    entry_price=fill_price,
                    high_water_mark=hwm,
                    entry_atr=entry_atr,
                    take_profit_price=tp_price,
                    garch_sigma=garch_sigma,
                )
                log_decision({"symbol": symbol, "action": "buy",
                              "pred_return": pred_return,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                              "llm_multiplier": llm_mult, "llm_score": llm_s,
                              "llm_reasoning": llm_reason,
                              "final_notional": notional,
                              "skip_reason": None})
                self.last_trade_time[symbol] = datetime.datetime.now()

    def _sleep(self):
        """Sleep with thermal throttling."""
        sleep_interval = self.LOOP_INTERVAL
        temp = get_gpu_temp()
        if temp is not None and temp > THERMAL_THROTTLE_TEMP:
            sleep_interval = self.LOOP_INTERVAL * 2
            logger.info("[HW] GPU temp %.0fC > %dC, throttling to %ds",
                        temp, THERMAL_THROTTLE_TEMP, sleep_interval)

        # Add small random jitter (±5s) to prevent pattern detection
        import random
        jitter = random.uniform(-5, 5)
        sleep_interval = max(10, sleep_interval + jitter)

        logger.info("[SLEEP] Next check in %.0fs...", sleep_interval)
        time.sleep(sleep_interval)
