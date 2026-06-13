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
    predict_symbol, compute_kelly_fraction,
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
    ATR_TRAIL_MULTIPLIER: float = 2.0
    ATR_TRAIL_ACTIVATE_PCT: float = 0.01
    ATR_STOP_FLOOR_PCT: float = 0.05
    ATR_STOP_CEIL_PCT: float = 0.10
    TAKE_PROFIT_RR: float = 3.0
    TAKE_PROFIT_CEIL_PCT: float = 0.25
    STOP_LOSS_PCT: float = 0.05  # fallback fixed
    TRAIL_PCT: float = 0.04
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
        self._last_llm_symbols: set[str] = set()
        self._last_stale_force = 0.0
        self.model_mtime = 0
        self.cycle = 0
        self.macro_regime: MacroRegime | None = None
        self.corr_matrix: dict = {}
        self._equity: float = 100_000
        self._peak_equity: float = 100_000
        self._buys_allowed: bool = True
        self._halted_until: datetime.datetime | None = None
        # Per-symbol daily trade budget: caps fee bleed from signal jitter
        # re-trading the same name all day
        self._daily_trades: dict[str, int] = {}   # symbol -> count (today)
        self._daily_trades_date: str = datetime.date.today().isoformat()
        # Stop-breach confirmation state (2-consecutive-reading rule)
        self._pending_breach: dict[str, str] = {}
        # Conviction instrumentation (wave-5 Tier1-1): last meta probability
        # per symbol, stashed by _meta_gate so buy/skip rows can record it
        # without recomputing. Measurement-only — never gates.
        self._last_meta_p: dict[str, float] = {}
        # LLM veto strikes: liquidation needs 2 consecutive vetoing analyses
        self._veto_strikes: dict[str, int] = {}
        # Persistent prediction pool: rebuilding an executor every cycle
        # forced fresh thread spawns + per-thread SQLite connections 2,880x/day
        self._prediction_pool = ThreadPoolExecutor(
            max_workers=self.MAX_PREDICTION_WORKERS,
            thread_name_prefix=f'{self.get_asset_type()}-pred')
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
    def place_sell_order(self, symbol: str, qty, quote: dict | None):
        """Place a sell order. Returns the FILLED order object, or None.

        Returning the order (not a bool) lets callers journal the real
        fill price instead of a pre-trade quote estimate.
        """

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
        try:
            from order_stream import start_order_stream
            start_order_stream()  # no-op unless TRADER_ORDER_STREAM=1
        except Exception:
            pass
        self._load_models()
        # Scope cleanup to this bot's own universe — both bots share one
        # account, and an account-wide cancel would strip the other bot's
        # protective bracket/stop legs.
        cancel_all_open_orders(self.api, symbols=self.get_symbol_universe())
        self._reconstruct_positions()
        self._print_startup()

        while True:
            try:
                self._run_one_cycle()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # A single bad cycle (API blip, parse error) must never kill
                # the process: a crash-restart cancels working orders and
                # resets in-memory state, which is worse than skipping a beat.
                logger.exception("[CYCLE] Unhandled error in cycle %d — continuing", self.cycle)
                try:
                    from notify import notify
                    notify(f"{self.get_asset_type()} loop cycle error: {e}",
                           level='warning',
                           dedupe_key=f'cycle-error-{self.get_asset_type()}')
                except Exception:
                    pass
                time.sleep(self.LOOP_INTERVAL)

    def _run_one_cycle(self):
        """One iteration of the trading loop."""
        self.cycle += 1

        if not self.check_market_hours():
            if self.cycle == 1 or self.cycle % 20 == 0:
                logger.info("[WAIT] Market closed. Next check in %ds...", self.LOOP_INTERVAL)
            time.sleep(self.LOOP_INTERVAL)
            return

        logger.info("--- CYCLE %d: %s ---", self.cycle,
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Remote /flatten request (Telegram kill switch or GUI)
        self._check_flatten_request()

        # Pre-trade checks (also refreshes self._buys_allowed)
        self._circuit_breaker_check()

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

        # Challenger shadow predictions (hourly side-by-side log; no
        # trading impact — promotion is decided by the daily DM test)
        try:
            from shadow import maybe_log_shadow
            maybe_log_shadow(self, preds, benchmark)
        except Exception:
            pass

        # LLM analysis (throttled)
        self._run_llm_analysis(preds)

        # Sell bearish positions
        self._execute_sells(preds)

        # LLM veto sells
        self._execute_llm_veto_sells()

        # Buy (suppressed while halted or when risk state is unknown)
        if self._buys_allowed:
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
            logger.warning("Model files not found. Buys are DISABLED until a model exists "
                           "(fail closed); exits and stops still run.")

        from trading_utils import model_reload_key
        self.model_mtime = model_reload_key(self.MODEL_PREFIX)

    def _position_state_file(self) -> str:
        name = f'{self.MODEL_PREFIX}_position_state.json' if self.MODEL_PREFIX else 'position_state.json'
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)

    def _save_position_state(self):
        """Persist high-water marks and cooldown timers across restarts.

        Without this, every restart resets trailing-stop HWMs to
        max(entry, current) — loosening stops — and clears cooldowns,
        allowing immediate re-entries after a crash loop.
        """
        try:
            data = {
                'hwm': {s: p.high_water_mark for s, p in self.positions.items()},
                'trailing': {s: p.trailing_activated for s, p in self.positions.items()},
                'last_trade': {s: t.timestamp() for s, t in self.last_trade_time.items()},
                'daily_trades': {'date': self._daily_trades_date,
                                 'counts': self._daily_trades},
            }
            tmp = self._position_state_file() + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, self._position_state_file())
        except Exception as e:
            logger.debug("[STATE] Failed to save position state: %s", e)

    def _load_position_state(self) -> dict:
        try:
            with open(self._position_state_file()) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception as e:
            logger.debug("[STATE] Failed to load position state: %s", e)
            return {}

    def _reconstruct_positions(self):
        """Rebuild positions from API (survive restarts)."""
        from market_data import get_live_atr
        symbols = self.get_symbol_universe()
        raw_positions = reconstruct_positions(self.api, symbols, asset_type=self.get_asset_type())
        saved = self._load_position_state()
        saved_hwm = saved.get('hwm', {})
        saved_trailing = saved.get('trailing', {})
        for sym, info in raw_positions.items():
            entry_atr = get_live_atr(self.api, sym, asset_type=self.get_asset_type())
            tp_price = None
            if entry_atr is not None and info['entry_price'] > 0:
                raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / info['entry_price']
                stop_dist = max(self.ATR_STOP_FLOOR_PCT, min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
                tp_dist = min(self.TAKE_PROFIT_CEIL_PCT, stop_dist * self.TAKE_PROFIT_RR)
                tp_price = info['entry_price'] * (1 + tp_dist)
            # Restore the persisted high-water mark so trailing stops don't
            # loosen across restarts
            hwm = max(info['high_water_mark'], float(saved_hwm.get(sym, 0.0)))
            self.positions[sym] = Position(
                qty=info['qty'],
                entry_price=info['entry_price'],
                high_water_mark=hwm,
                entry_atr=entry_atr,
                take_profit_price=tp_price,
                trailing_activated=bool(saved_trailing.get(sym, False)),
            )

        # Restore cooldown timers (in-memory only before; a crash-restart
        # cleared them and allowed instant re-trades)
        for sym, ts in saved.get('last_trade', {}).items():
            try:
                self.last_trade_time[sym] = datetime.datetime.fromtimestamp(float(ts))
            except (TypeError, ValueError, OSError):
                pass

        # Restore today's per-symbol trade counts (budget survives restarts)
        dt = saved.get('daily_trades', {})
        if dt.get('date') == datetime.date.today().isoformat():
            self._daily_trades = {k: int(v) for k, v in (dt.get('counts') or {}).items()}
            self._daily_trades_date = dt['date']

        self._replace_protective_stops()

        if self.positions:
            logger.info("Existing positions: %s", ', '.join(self.positions))

    def _replace_protective_stops(self):
        """Re-place server-side protection after a restart (subclass hook).

        Startup cancels this bot's working orders (including bracket legs),
        and reconstruction previously left stop_order_id=None forever.
        Stocks re-place a stop order; crypto re-places a resting GTC
        stop_limit (Alpaca crypto supports market/limit/STOP_LIMIT — an
        earlier comment here claimed otherwise, which left 24/7 crypto
        positions dependent on this process staying alive).
        """

    def _after_entry_protection(self, symbol: str, pos):
        """Place server-side protection right after a confirmed entry
        (subclass hook). Stocks use bracket legs at order time; crypto
        overrides this to place a resting GTC stop_limit."""

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
        """Check circuit breaker and update self._buys_allowed.

        Trip behavior: flatten THIS bot's book once, then halt new entries
        until the daily baseline (Alpaca last_equity) resets at the next
        market close — the old sleep(3600) re-tripped hourly against the
        same baseline and re-flattened/re-cancelled all day while also
        blocking stop management.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        if self._halted_until and now < self._halted_until:
            self._buys_allowed = False
            return True
        self._halted_until = None

        try:
            tripped, dd = check_circuit_breaker(self.api, max_drawdown_pct=self.CIRCUIT_BREAKER_PCT)
        except Exception as e:
            logger.error("[CIRCUIT BREAKER] API error: %s", e)
            self._buys_allowed = False  # unknown risk state — fail closed
            return False

        if dd is None:
            # Account API unreachable: keep managing exits, skip new entries
            logger.warning("[CIRCUIT BREAKER] Risk state unknown (API error) — buys suspended this cycle")
            self._buys_allowed = False
            return False

        if tripped:
            logger.warning("[CIRCUIT BREAKER] Drawdown %.1f%% >= %.0f%%, flattening %s book!",
                           dd * 100, self.CIRCUIT_BREAKER_PCT * 100, self.get_asset_type())
            try:
                from notify import notify
                notify(f"CIRCUIT BREAKER tripped: {self.get_asset_type()} book "
                       f"down {dd * 100:.1f}% — flattening and halting entries "
                       f"until baseline reset", level='critical',
                       dedupe_key=f'breaker-{self.get_asset_type()}')
            except Exception:
                pass
            # Journal the forced exits (estimated — emergency fills aren't
            # individually confirmed here) so Kelly's sample isn't censored
            # of exactly the worst trades
            for sym, pos in list(self.positions.items()):
                quote = self.get_quote(sym)
                px = quote['midpoint'] if quote else pos.entry_price
                pnl = ((px - pos.entry_price) / pos.entry_price * 100
                       if pos.entry_price > 0 else 0.0)
                record_trade(sym, 'sell', pos.entry_price, px, pnl,
                             exit_reason='circuit_breaker', estimated=True)
            failures = emergency_flatten(self.api, symbols=self.get_symbol_universe())
            if failures:
                logger.error("[CIRCUIT BREAKER] Unconfirmed flattens: %s — will retry next cycle",
                             ', '.join(failures))
                # Keep failed symbols tracked so stops still manage them
                self.positions = {s: p for s, p in self.positions.items() if s in failures}
            else:
                self.positions.clear()
            self._halted_until = self._next_baseline_reset()
            self._buys_allowed = False
            logger.info("[CIRCUIT BREAKER] Halting new entries until %s (baseline reset)",
                        self._halted_until.isoformat())
            return True

        self._buys_allowed = True
        return False

    @staticmethod
    def _next_baseline_reset() -> datetime.datetime:
        """Next time Alpaca's last_equity baseline rolls (~16:05 ET), in UTC."""
        import zoneinfo
        et = zoneinfo.ZoneInfo('US/Eastern')
        now_et = datetime.datetime.now(et)
        reset = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
        if now_et >= reset:
            reset += datetime.timedelta(days=1)
        return reset.astimezone(datetime.timezone.utc)

    def _hot_reload_check(self):
        """Check if model files changed and reload (keyed on the manifest)."""
        from trading_utils import model_reload_key
        new_mtime = model_reload_key(self.MODEL_PREFIX)
        if new_mtime != self.model_mtime:
            logger.info("[HOT-RELOAD] Model files changed, reloading...")
            try:
                inference_device = choose_inference_device()
                self.model, self.config, self.scaler_X, self.feature_cols = \
                    load_models(inference_device, prefix=self.MODEL_PREFIX)
                self.trade_threshold = self.config.get('trade_threshold', 0.15)
                self.model_mtime = new_mtime
                # New champion artifacts -> the per-prefix LGB/q10 booster
                # caches pair with the OLD weights; drop them for reload
                try:
                    import predict_now
                    predict_now._lgb_models.pop(self.MODEL_PREFIX, None)
                    predict_now._q10_models.pop(self.MODEL_PREFIX, None)
                except Exception:
                    pass
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
                    failures = emergency_flatten(self.api, symbols=self.get_symbol_universe())
                    self.positions = {s: p for s, p in self.positions.items() if s in failures}
        except Exception as e:
            logger.debug("[MACRO] Regime update failed: %s", e)

    def _update_equity(self):
        """Update cached equity."""
        try:
            acct = self.api.get_account()
            self._equity = float(acct.equity)
            if self._equity > self._peak_equity:
                self._peak_equity = self._equity
        except Exception:
            pass

    def _update_correlations(self):
        """Update correlation matrix for portfolio management (1h cache)."""
        try:
            from portfolio import get_correlation_matrix_cached
            symbols = self.get_symbol_universe()
            corr = get_correlation_matrix_cached(self.api, symbols,
                                                 self.get_asset_type())
            if corr:
                self.corr_matrix = corr
        except Exception as e:
            logger.debug("[PORTFOLIO] Correlation update failed: %s", e)

    def _maybe_update_resting_stop(self, symbol, pos, desired_stop_price):
        """Tighten the server-side resting stop as the trail rises
        (subclass hook; crypto overrides). Base: no-op."""

    def _extra_tilt(self, symbol: str) -> float:
        """Asset-specific advisory tilt hook (crypto: funding). Base: 1.0."""
        return 1.0

    def _desired_stop_for(self, pos) -> tuple[float, float, float, bool]:
        """Current protective stop for a position — ONE source of truth.

        Returns (desired_stop, stop_dist, trail_dist, trailing_active).
        Used by stop management AND book-risk accounting so the cap can
        never disagree with the stops actually being enforced.
        """
        entry_price = pos.entry_price
        hwm = pos.high_water_mark
        entry_atr = pos.entry_atr
        if entry_atr is not None and entry_price > 0:
            raw_stop_dist = (entry_atr * self.ATR_STOP_MULTIPLIER) / entry_price
            stop_dist = max(self.ATR_STOP_FLOOR_PCT,
                            min(self.ATR_STOP_CEIL_PCT, raw_stop_dist))
            raw_trail_dist = ((entry_atr * self.ATR_TRAIL_MULTIPLIER) / hwm
                              if hwm > 0 else stop_dist)
            trail_dist = max(self.ATR_STOP_FLOOR_PCT,
                             min(self.ATR_STOP_CEIL_PCT, raw_trail_dist))
        else:
            stop_dist = self.STOP_LOSS_PCT
            trail_dist = self.TRAIL_PCT

        # Apply macro regime stop tightening
        if self.macro_regime and self.macro_regime.stop_mult < 1.0:
            stop_dist *= self.macro_regime.stop_mult
            trail_dist *= self.macro_regime.stop_mult

        trailing_active = hwm >= entry_price * (1 + self.ATR_TRAIL_ACTIVATE_PCT)
        desired_stop = entry_price * (1 - stop_dist)
        if trailing_active:
            desired_stop = max(desired_stop, hwm * (1 - trail_dist))
        return desired_stop, stop_dist, trail_dist, trailing_active

    def _book_stop_risks(self) -> list[float]:
        """Open stop-risk per position as a fraction of equity.

        Initial-risk bookkeeping (anchored at entry): once the trail moves
        a stop above entry, that position's principal risk is 0 and it
        stops consuming book risk budget. Giveback of unrealized gains is
        the drawdown ladder's job, not this cap's.
        """
        if self._equity <= 0:
            return []
        risks = []
        for pos in self.positions.values():
            try:
                stop, *_ = self._desired_stop_for(pos)
                risks.append(max(0.0, pos.entry_price - stop)
                             * pos.qty / self._equity)
            except Exception:
                risks.append(0.0)
        return risks

    def _manage_stops(self):
        """Check stop-loss, trailing stop, and take-profit on open positions."""
        for symbol in list(self.positions):
            pos = self.positions[symbol]

            # Resting protective order (crypto GTC stop_limit) — detect
            # server-side fills the loop would otherwise miss
            if pos.stop_order_id and self.get_asset_type() == 'crypto':
                try:
                    so = self.api.get_order(pos.stop_order_id)
                    status = getattr(so, 'status', None)
                    if status == 'filled':
                        logger.info("[STOP-FILL] %s: resting stop filled at $%s",
                                    symbol, so.filled_avg_price)
                        llm_info = self.llm_scores.get(symbol, {})
                        self._record_confirmed_exit(symbol, pos, so, None,
                                                    exit_reason='server_stop',
                                                    llm_score=llm_info.get('s'),
                                                    reasoning=llm_info.get('r', ''))
                        del self.positions[symbol]
                        self.last_trade_time[symbol] = datetime.datetime.now()
                        self.hard_stop_lockout[symbol] = datetime.datetime.now()
                        self._save_hard_stop_lockout()
                        continue
                    if status in ('canceled', 'expired', 'rejected'):
                        pos.stop_order_id = None
                except Exception:
                    pass

            quote = self.get_quote(symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            pos.high_water_mark = max(pos.high_water_mark, current_price)
            hwm = pos.high_water_mark
            entry_price = pos.entry_price

            (desired_stop, stop_dist, trail_dist,
             trailing_active) = self._desired_stop_for(pos)

            # Tighten the resting server-side stop as the trail rises
            self._maybe_update_resting_stop(symbol, pos, desired_stop)

            stop_reason = None
            if current_price <= entry_price * (1 - stop_dist):
                stop_reason = 'hard_stop'
            elif pos.take_profit_price and current_price >= pos.take_profit_price:
                stop_reason = 'take_profit'
            elif (trailing_active
                  and current_price <= hwm * (1 - trail_dist)):
                stop_reason = 'trailing'

            # Two-consecutive-reading confirmation: a single anomalous
            # quote on Alpaca's thin crypto venue must not market-dump a
            # healthy position. Costs one 30s cycle; genuine fast crashes
            # are covered by the resting server-side stop, which has no
            # such delay.
            if stop_reason:
                if self._pending_breach.get(symbol) == stop_reason:
                    self._pending_breach.pop(symbol, None)
                    logger.info("[STOP] %s: %s CONFIRMED at $%.4f (entry=$%.4f, hwm=$%.4f, "
                                "stop_d=%.1f%%, trail_d=%.1f%%)",
                                symbol, stop_reason, current_price, entry_price, hwm,
                                stop_dist * 100, trail_dist * 100)
                    self._execute_stop_exit(symbol, pos, stop_reason, current_price)
                else:
                    self._pending_breach[symbol] = stop_reason
                    logger.info("[STOP] %s: %s breach at $%.4f — awaiting "
                                "confirmation next cycle", symbol, stop_reason,
                                current_price)
            else:
                self._pending_breach.pop(symbol, None)

        # Persist HWM / cooldown state each cycle (tiny atomic JSON write)
        self._save_position_state()

    def _execute_stop_exit(self, symbol, pos, stop_reason, current_price):
        """Sell a position for a stop/TP/trailing exit and confirm the fill.

        Any resting order for this symbol (bracket stop/TP leg, trailing
        stop) holds the shares — selling around it rejects with
        'insufficient qty'. Cancel symbol-scoped orders first, confirm the
        sell filled, and record the trade at the REAL fill price.
        """
        from order_utils import cancel_orders_for_symbol, make_client_order_id, verify_position

        entry_price = pos.entry_price
        tif = 'gtc' if self.get_asset_type() == 'crypto' else 'day'

        # Free shares reserved by any working order (bracket leg, trailing stop)
        if not cancel_orders_for_symbol(self.api, symbol, timeout=5):
            logger.warning("[STOP] %s: working orders still pending cancel — retrying next cycle", symbol)
            return
        pos.stop_order_id = None

        try:
            order = self.api.submit_order(
                symbol=symbol, qty=pos.qty,
                side='sell', type='market', time_in_force=tif,
                client_order_id=make_client_order_id('stop'),
            )
        except Exception as e:
            err_msg = str(e).lower()
            logger.error("[STOP] %s: Sell error: %s", symbol, e)
            if ('insufficient qty' in err_msg
                    or 'position does not exist' in err_msg
                    or 'not found' in err_msg
                    or 'available: 0' in err_msg):
                # Only treat as desync if the broker REALLY has no position —
                # the same rejection fires when shares are merely reserved by
                # an order we failed to cancel.
                if verify_position(self.api, symbol) is None:
                    logger.warning("[DESYNC] %s: Position gone at broker, removing from tracking", symbol)
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    llm_info = self.llm_scores.get(symbol, {})
                    record_trade(symbol, 'sell', entry_price, current_price,
                                 pnl_pct, llm_score=llm_info.get('s'),
                                 reasoning='position desync — broker qty=0',
                                 exit_reason='desync', estimated=True)
                    self.positions.pop(symbol, None)
                    self.last_trade_time[symbol] = datetime.datetime.now()
                else:
                    logger.warning("[STOP] %s: position EXISTS but shares unavailable "
                                   "(reserved by an open order?) — retrying next cycle", symbol)
            return

        result = manage_order_lifecycle(self.api, order.id, timeout=self.ORDER_TIMEOUT,
                                        fallback_to_market=False, time_in_force=tif)
        status = getattr(result, 'status', None)
        if status == 'filled':
            fill_price = float(result.filled_avg_price)
            estimated = False
        else:
            logger.warning("[STOP] %s: exit fill unconfirmed (status=%s) — recording estimate",
                           symbol, status)
            fill_price = current_price
            estimated = True
            if status not in ('filled', 'partially_filled') and verify_position(self.api, symbol) is not None:
                # Sell didn't go through and we still hold it — keep tracking
                return

        pnl_pct = ((fill_price - entry_price) / entry_price) * 100
        llm_info = self.llm_scores.get(symbol, {})
        record_trade(symbol, 'sell', entry_price, fill_price,
                     pnl_pct, llm_score=llm_info.get('s'),
                     reasoning=llm_info.get('r', ''),
                     exit_reason=stop_reason, estimated=estimated)
        self.positions.pop(symbol, None)
        self.last_trade_time[symbol] = datetime.datetime.now()
        if stop_reason == 'hard_stop':
            self.hard_stop_lockout[symbol] = datetime.datetime.now()
            self._save_hard_stop_lockout()
            logger.info("[LOCKOUT] %s: %dh lockout after hard stop",
                        symbol, self.HARD_STOP_LOCKOUT_HOURS)

    def _get_predictions(self, benchmark_close) -> tuple[dict, dict]:
        """Get predictions for all symbols in parallel."""
        preds = {}
        snapshots = {}
        if self.model is None:
            return preds, snapshots

        inference_device = choose_inference_device()
        symbols = self.get_symbol_universe()

        futures = {}
        for symbol in symbols:
            f = self._prediction_pool.submit(
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

        # Rolling prediction log for the PSI drift monitor (one line per
        # cycle; monitor_drift.py prunes it to 7 days)
        try:
            from monitor_drift import log_predictions
            log_predictions(self.MODEL_PREFIX, preds)
        except Exception:
            pass

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

        fng_value = None
        try:
            from sentiment import get_fear_greed
            fd = get_fear_greed()
            fng_value = fd.get('value') if isinstance(fd, dict) else fd
        except Exception:
            pass

        new_scores = analyze_trades(
            candidates, self.get_asset_type(), equity=self._equity,
            positions=list(self.positions.keys()),
            position_details={s: p.to_dict() for s, p in self.positions.items()},
            fng_value=fng_value,
            model_config=self.config,
        )
        self._last_llm_symbols = {c.get('symbol') for c in candidates}
        if new_scores:
            self.llm_scores = new_scores
            self._last_llm_time = now_ts
            # Veto strikes: forced LIQUIDATION needs two consecutive
            # vetoing analyses. Headlines are untrusted text concatenated
            # into the prompt — a single injected/anomalous score must not
            # be able to dump a position (measured attack vector:
            # arXiv 2601.13082). New-entry blocking stays immediate.
            for sym, v in new_scores.items():
                if v.get('s', 0.5) < LLM_VETO_THRESHOLD:
                    self._veto_strikes[sym] = self._veto_strikes.get(sym, 0) + 1
                else:
                    self._veto_strikes.pop(sym, None)
            logger.info("[LLM] Scores: %s",
                        ", ".join(f"{s}={v.get('s', 0.5):.2f}" for s, v in self.llm_scores.items()))
            # Journal every scored candidate (not just traded ones) so
            # llm_eval.py can measure whether the gate predicts returns —
            # the system previously had no way to know if the LLM helped
            preds_by_symbol = {c['symbol']: c.get('pred_return') for c in candidates}
            log_decision({
                "action": "llm_analysis",
                "asset_type": self.get_asset_type(),
                "forward_bars": self.config.get('forward_bars', 24) if self.config else 24,
                "scores": {sym: {"s": v.get('s', 0.5),
                                 "pred": preds_by_symbol.get(sym)}
                           for sym, v in new_scores.items()},
            })

    def _check_llm_staleness(self):
        """Force a refresh if the scores for CURRENT candidates are stale.

        Scoped to the symbols this bot actually analyzes: the stock bot only
        refreshes its top-N, while the GUI writes the whole universe into
        llm_analysis.json — judging staleness by the OLDEST entry in the
        section made departed symbols permanently stale, collapsing the
        600s cadence to ~60s (10x quota burn + loop stalls). Forced
        refreshes are additionally rate-limited to one per LLM interval.
        """
        if not self.llm_scores:
            return
        now_ts = time.time()
        # Rate-limit both the disk parse and the forced refresh
        if now_ts - self._last_stale_force < self.LLM_INTERVAL_SEC:
            return
        from llm_analyst import load_analysis
        from datetime import datetime, timezone
        try:
            data = load_analysis()
            section = data.get(self.get_asset_type(), {})
            if not section:
                return
            relevant = self._last_llm_symbols or set(self.llm_scores.keys())
            newest_ts = None
            for sym in relevant:
                ts_str = (section.get(sym) or {}).get('timestamp', '')
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if newest_ts is None or ts > newest_ts:
                            newest_ts = ts
                    except ValueError:
                        pass
            if newest_ts is not None:
                age_hours = (datetime.now(timezone.utc) - newest_ts).total_seconds() / 3600
                if age_hours > 2:
                    logger.warning("[LLM] Scores are %.1fh stale — forcing refresh next cycle",
                                   age_hours)
                    self._last_llm_time = 0  # Force refresh
                    self._last_stale_force = now_ts
        except Exception:
            pass

    def get_fresh_headlines(self, symbol: str) -> list[str]:
        """Headlines with stale reprints filtered out (novelty.py).

        Fresh news carries ~1.7x the price response of reprints; feeding
        the LLM the same wire story five times re-counts one event five
        times. Fail open to the raw list.
        """
        headlines = self.get_headlines(symbol)
        try:
            from novelty import filter_novel
            return filter_novel(symbol, headlines)
        except Exception:
            return headlines

    def _build_llm_candidates(self, preds: dict) -> list[dict]:
        """Build candidate list for LLM analysis. Override for stock-specific fundamentals."""
        candidates = []
        for symbol in self.get_symbol_universe():
            fund = get_fundamentals(symbol, self.get_asset_type())
            fund_text = format_fundamentals_for_llm(symbol, fund)
            headlines = self.get_fresh_headlines(symbol)
            candidates.append({
                'symbol': symbol,
                'pred_return': preds.get(symbol),
                'fundamentals_text': fund_text,
                'news_headlines': headlines,
            })
        return candidates

    def _record_confirmed_exit(self, symbol, pos, order, quote, exit_reason,
                               llm_score=None, reasoning=''):
        """Journal an exit using the order's real fill price when available."""
        fill_price = None
        try:
            fp = getattr(order, 'filled_avg_price', None)
            if fp is not None:
                fill_price = float(fp)
        except (TypeError, ValueError):
            pass
        estimated = fill_price is None
        if fill_price is None:
            fill_price = quote['midpoint'] if quote else pos.entry_price
        pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100 \
            if pos.entry_price > 0 else 0.0
        record_trade(symbol, 'sell', pos.entry_price, fill_price, pnl_pct,
                     llm_score=llm_score, reasoning=reasoning,
                     exit_reason=exit_reason, estimated=estimated)
        # Implementation-shortfall journal: decision price (quote when the
        # exit fired) vs realized fill. Sells slip NEGATIVE of buys: a fill
        # below the decision mid costs money, so flip the sign so positive
        # slippage_bps always means "paid more than the decision price".
        decision_price = quote['midpoint'] if quote else None
        slippage_bps = None
        if decision_price and decision_price > 0 and not estimated:
            slippage_bps = round((decision_price - fill_price) / decision_price * 1e4, 2)
        log_decision({"symbol": symbol, "action": "sell",
                      "exit_reason": exit_reason,
                      "pnl_pct": round(pnl_pct, 4),
                      "decision_price": decision_price,
                      "fill_price": fill_price,
                      "slippage_bps": slippage_bps,
                      "estimated": estimated})

    def _execute_sells(self, preds: dict):
        """Sell bearish positions."""
        for symbol in list(self.positions):
            pred = preds.get(symbol)
            if pred is None:
                # Missing prediction = data failure, NOT a sell signal.
                # Stops continue to protect the position; liquidating the
                # book on a transient fetch error is how rate-limit bursts
                # turn into forced sales.
                continue
            if pred > -self.trade_threshold:
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                continue

            logger.info("%s: SELLING (pred=%+.4f%%)", symbol, pred)

            quote = self.get_quote(symbol)
            if quote is None:
                logger.warning("%s: Skipping sell — quote unavailable", symbol)
                continue
            pos = self.positions[symbol]
            # A resting protective order HOLDS the qty — selling around it
            # rejects with 'insufficient quantity available'
            if pos.stop_order_id:
                from order_utils import cancel_orders_for_symbol
                if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                    logger.warning("%s: resting orders pending cancel — retry next cycle", symbol)
                    continue
                pos.stop_order_id = None
            llm_info = self.llm_scores.get(symbol, {})
            order = self.place_sell_order(symbol, pos.qty, quote)
            if order:
                self._record_confirmed_exit(symbol, pos, order, quote,
                                            exit_reason='signal_sell',
                                            llm_score=llm_info.get('s'),
                                            reasoning=llm_info.get('r', ''))
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

    def _execute_llm_veto_sells(self):
        """Sell positions with catastrophic LLM scores.

        Liquidation requires the veto to PERSIST across two consecutive
        analyses (strike count from _run_llm_analysis). One anomalous or
        injected score blocks new entries immediately but cannot force a
        sale on its own.
        """
        for symbol in list(self.positions):
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            if llm_s >= LLM_VETO_THRESHOLD:
                continue
            if self._veto_strikes.get(symbol, 0) < 2:
                logger.info("%s: LLM VETO (%.2f) strike %d/2 — blocking entries, "
                            "liquidation needs a second consecutive veto",
                            symbol, llm_s, self._veto_strikes.get(symbol, 0))
                continue

            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                logger.info("%s: LLM VETO (%.2f) but in cooldown", symbol, llm_s)
                continue

            logger.info("%s: LLM VETO SELL (%.2f — %s)", symbol, llm_s,
                        llm_info.get('r', ''))
            pos = self.positions[symbol]
            # Free shares held by any resting protective order first
            if pos.stop_order_id:
                from order_utils import cancel_orders_for_symbol
                if not cancel_orders_for_symbol(self.api, symbol, timeout=8):
                    logger.warning("%s: resting orders pending cancel — retry next cycle", symbol)
                    continue
                pos.stop_order_id = None
            quote = self.get_quote(symbol)
            order = self.place_sell_order(symbol, pos.qty, quote)
            if order:
                self._record_confirmed_exit(symbol, pos, order, quote,
                                            exit_reason='llm_veto',
                                            llm_score=llm_s,
                                            reasoning=llm_info.get('r', ''))
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            time.sleep(1)

    def _meta_gate(self, symbol: str, pred_return, snapshots: dict,
                   rank=None):
        """Meta-labeling gate: calibrated P(this trade profits net of costs).

        Returns (allowed, meta_mult). No trained meta model -> neutral.
        Stashes the computed probability in self._last_meta_p for the
        conviction journaling (wave-5 Tier1-1), and tags the meta_veto
        skip row with the entry rank when supplied.
        """
        try:
            from meta_label import (meta_probability_live, meta_size_mult,
                                    META_VETO_PROB)
            p = meta_probability_live(self.MODEL_PREFIX,
                                      snapshots.get(symbol, {}) or {},
                                      pred_return)
            if p is None:
                self._last_meta_p.pop(symbol, None)
                return True, 1.0
            self._last_meta_p[symbol] = float(p)
            if p < META_VETO_PROB:
                rec = {"symbol": symbol, "action": "skip",
                       "skip_reason": "meta_veto",
                       "pred_return": pred_return,
                       "meta_prob": round(p, 4)}
                if rank is not None:
                    rec["entry_rank"] = rank
                log_decision(rec)
                return False, 1.0
            return True, meta_size_mult(p)
        except Exception:
            return True, 1.0

    # --- Conviction instrumentation (wave-5 Tier1-1, measurement-only) ---
    # These NEVER affect control flow. They make every entry window's
    # admitted-k and per-candidate veto attribution reconstructable from
    # the journals, the substrate for the Stage-0 experiments that gate
    # the conviction/concentration flagship. Disable via
    # strategy_config.CONVICTION_JOURNAL_ENABLED.

    @staticmethod
    def _conviction_journal_on() -> bool:
        try:
            from strategy_config import CONVICTION_JOURNAL_ENABLED
            return bool(CONVICTION_JOURNAL_ENABLED)
        except Exception:
            return True

    def _conviction_tier(self, pred, meta_p) -> str:
        """Provisional conviction tier. Tier1-1 only PLACES the field so
        downstream code and analysis have it; the real tier definition and
        its sizing land in Tier1-3 AFTER Stage-0 measures the rank-edge
        gradient. A = strong signal AND strong meta; B = one of them;
        C = neither (rare among fills since both clear their vetoes)."""
        thr = self.trade_threshold or 0.0
        strong_pred = thr > 0 and pred is not None and pred >= 1.5 * thr
        strong_meta = meta_p is not None and meta_p >= 0.60
        if strong_pred and strong_meta:
            return 'A'
        if strong_pred or strong_meta:
            return 'B'
        return 'C'

    def _conv_fields(self, symbol, pred, snapshot, rank=None) -> dict:
        """Conviction context shared by skip and buy journal rows."""
        f = {}
        if rank is not None:
            f['entry_rank'] = rank
        thr = self.trade_threshold or 0.0
        if pred is not None and thr > 0:
            f['pred_thresh_ratio'] = round(float(pred) / thr, 3)
        snap = snapshot or {}
        if snap.get('Q10') is not None:
            f['q10'] = round(float(snap['Q10']), 4)
        if snap.get('Q10_Floor') is not None:
            f['q10_floor'] = round(float(snap['Q10_Floor']), 4)
        mp = self._last_meta_p.get(symbol)
        if mp is not None:
            f['meta_p'] = round(float(mp), 4)
            f['conviction_tier'] = self._conviction_tier(pred, mp)
        return f

    def _journal_skip(self, symbol, reason, rank=None, pred=None,
                      snapshot=None, **extra):
        """Journal a per-candidate veto row for counterfactual pricing.

        Only for the conviction/risk gates worth replaying — mechanical
        skips (already-held, cooldown, budget) are counted in the window
        summary but not priced per-symbol."""
        if not self._conviction_journal_on():
            return
        rec = {"symbol": symbol, "action": "skip", "skip_reason": reason}
        if pred is not None:
            rec["pred_return"] = round(float(pred), 4)
        rec.update(self._conv_fields(symbol, pred, snapshot, rank=rank))
        rec.update(extra)
        try:
            log_decision(rec)
        except Exception:
            pass

    def _journal_entry_window(self, n_candidates, admitted, veto_counts):
        """One summary row per entry-window cycle: admitted-k + the veto
        breakdown across the candidate set. The cheap aggregate that
        reconstructs the admitted-k distribution for Stage-0."""
        if not self._conviction_journal_on() or n_candidates <= 0:
            return
        try:
            log_decision({
                "action": "entry_window",
                "asset_type": self.get_asset_type(),
                "n_candidates": int(n_candidates),
                "admitted_k": len(admitted),
                "admitted": list(admitted),
                "veto_counts": dict(veto_counts),
                "buys_allowed": bool(self._buys_allowed),
            })
        except Exception:
            pass

    def _compute_position_size(self, symbol: str, pred_return: float | None,
                               quote: dict, sentiment_mult: float = 1.0,
                               llm_mult: float = 1.0,
                               meta_mult: float = 1.0) -> int:
        """Risk-based position sizing.

        Replaces the old ~10-factor multiplier soup (which could stack to
        ~32x base with per-symbol caps unenforced) with a structure where
        each component has ONE job and a hard bound:

          1. RISK BASE — risk a fixed fraction of equity to the stop:
                 notional = equity * RISK_PCT_PER_TRADE / stop_dist
             capped at NOTIONAL_PER_SYMBOL. Size now scales with the
             account and with how far the stop is — the textbook
             definition the old code never had.
          2. KELLY — fractional Kelly (<= KELLY_CAP) from CONFIRMED trade
             history, as a bounded multiplier [0.5, 1.5]. The old floor
             meant Kelly could only ever INCREASE size.
          3. VOL TARGET — GARCH per-bar sigma vs the annualized portfolio
             target (volatility.py), bounded [0.5, 1.5].
          4. TILT — everything advisory (confidence, macro, HMM,
             correlation, drawdown ladder, VIX, sentiment, LLM) multiplies
             into ONE tilt. Boosts are clamped at TILT_MAX (1.3x);
             de-risking is honored down to 0.1x (a drawdown ladder that
             says cut 75% must not be clipped).
          5. CAPS — MAX_NOTIONAL_PER_SYMBOL enforced INCLUDING the new
             order; orders below MIN_ORDER_NOTIONAL return 0 (fees eat dust).
        """
        from strategy_config import (RISK_PCT_PER_TRADE, KELLY_CAP,
                                     TILT_MAX, MIN_ORDER_NOTIONAL)
        from market_data import get_live_atr

        # --- 1. Risk base: equity at risk / stop distance ---
        entry_atr = get_live_atr(self.api, symbol, asset_type=self.get_asset_type())
        price = quote['midpoint']
        if entry_atr is not None and price > 0:
            raw_stop = (entry_atr * self.ATR_STOP_MULTIPLIER) / price
            stop_dist = max(self.ATR_STOP_FLOOR_PCT,
                            min(self.ATR_STOP_CEIL_PCT, raw_stop))
        else:
            stop_dist = self.STOP_LOSS_PCT
        risk_dollars = self._equity * RISK_PCT_PER_TRADE
        risk_notional = risk_dollars / max(stop_dist, 1e-4)
        base = min(risk_notional, self.NOTIONAL_PER_SYMBOL)

        # --- 2. Fractional Kelly from confirmed history (bounded) ---
        kelly_mult = 1.0
        kelly_f = compute_kelly_fraction(asset_type=self.get_asset_type())
        if kelly_f is not None:
            kelly_f = min(kelly_f, KELLY_CAP)
            # 0.125 (mid of the [0.05, 0.25] clamp) maps to 1.0x
            kelly_mult = max(0.5, min(1.5, kelly_f / 0.125))

        # --- 3. GARCH vol targeting (bounded inside the helper) ---
        vol_mult = 1.0
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
        if returns is not None:
            try:
                from volatility import get_sigma
                sigma = get_sigma(symbol, returns, bars=df,
                                  asset_type=self.get_asset_type())
                if sigma is not None:
                    vol_mult = compute_vol_adjusted_size(
                        1.0, sigma, asset_type=self.get_asset_type())
            except Exception:
                pass

        # --- 4. Advisory tilt (single product, asymmetric clamp) ---
        tilt = 1.0
        # Signal confidence (tamed: was 0.5-2.0)
        if pred_return is not None and self.trade_threshold > 0.001:
            tilt *= min(1.25, max(0.75, pred_return / self.trade_threshold))
        # VIX ladder (arXiv 2508.16598: shrink Kelly in high vol)
        vix = self.macro_regime.vix if self.macro_regime else None
        if vix is not None:
            tilt *= 0.3 if vix > 35 else 0.5 if vix > 25 else 0.7 if vix > 15 else 1.0
        # Drawdown de-leveraging ladder
        if self._peak_equity > 0:
            dd = (self._peak_equity - self._equity) / self._peak_equity
            if dd >= 0.20:
                tilt *= 0.25
            elif dd >= 0.15:
                tilt *= 0.50
            elif dd >= 0.10:
                tilt *= 0.75
        # Macro regime
        if self.macro_regime:
            tilt *= self.macro_regime.sizing_mult
        # Correlation with existing book
        if self.corr_matrix and self.positions:
            tilt *= get_correlation_sizing_factor(
                symbol, list(self.positions.keys()), self.corr_matrix)
        # HMM regime (advisory) + disagreement penalty
        hmm_label = 'unknown'
        if returns is not None and len(returns) > 200:
            try:
                regime = get_cached_regime(symbol, returns)
                tilt *= regime['sizing_mult']
                hmm_label = regime.get('label', 'unknown')
            except Exception:
                pass
        votes = []
        if self.macro_regime:
            votes.append(-1 if self.macro_regime.sizing_mult < 0.6
                         else 1 if self.macro_regime.sizing_mult > 0.9 else 0)
        if hmm_label == 'bull':
            votes.append(1)
        elif hmm_label == 'bear':
            votes.append(-1)
        elif hmm_label != 'unknown':
            votes.append(0)
        if len(votes) >= 2 and len(set(votes)) > 1:
            tilt *= 0.8
        # Regime-conditional Kelly cap: never let recent-win streaks scale
        # size ABOVE baseline in stressed/bear regimes (procyclicality fix)
        if (vix is not None and vix > 25) or hmm_label == 'bear':
            kelly_mult = min(kelly_mult, 1.0)
        # Sentiment gate, LLM conviction, meta-label probability
        # (vetoes handled by callers)
        tilt *= sentiment_mult
        tilt *= llm_mult
        tilt *= meta_mult
        # Asset-specific extra tilt (crypto: funding-rate positioning)
        tilt *= self._extra_tilt(symbol)
        # Book-level realized-vol scalar (EWMA of the account equity
        # curve; de-risk only — catches correlation buildup that
        # per-position GARCH targeting can't see)
        try:
            from portfolio import get_book_vol_scalar_cached
            tilt *= get_book_vol_scalar_cached(self.api, self.get_asset_type())
        except Exception:
            pass
        # Boosts capped; de-risking honored
        tilt = max(0.1, min(TILT_MAX, tilt))

        # DEGRADED-MODE CLAMP: the advisory gates share upstream data
        # sources (yfinance, Alpaca bars), and each one silently defaults
        # to NEUTRAL when its feed fails — so a yfinance outage during a
        # VIX-40 crash made the system size at full tilt with its risk
        # ladders blind. When 2+ advisory inputs are unavailable, treat
        # the risk state as unknown and cap the tilt at 0.5x.
        missing = sum([
            vix is None,
            returns is None,                # GARCH + HMM both blind
            not self.corr_matrix,
            kelly_f is None,
        ])
        if missing >= 2:
            if tilt > 0.5:
                logger.warning("[SIZING] %s: %d advisory inputs unavailable — "
                               "degraded mode, tilt capped at 0.5x", symbol, missing)
            tilt = min(tilt, 0.5)

        sized = base * kelly_mult * vol_mult * tilt

        # --- 4b. Correlation-adjusted BOOK risk cap (ENB) ---
        # Per-trade risk alone lets N near-lockstep positions stack
        # N*0.5% of equity behind one factor move. Cap the
        # equicorrelation book risk at MAX_BOOK_RISK_PCT by shrinking
        # the entry into the remaining budget (0 budget = no entry).
        if self.positions and self._equity > 0:
            try:
                from portfolio import avg_book_correlation, book_risk_budget
                from strategy_config import MAX_BOOK_RISK_PCT
                book = list(self.positions.keys())
                # No correlation data -> assume a fairly correlated book
                # (crypto pairwise corr typically 0.6-0.9) rather than 0
                rho = (avg_book_correlation(book + [symbol], self.corr_matrix)
                       if self.corr_matrix else 0.5)
                budget = book_risk_budget(self._book_stop_risks(), rho,
                                          MAX_BOOK_RISK_PCT)
                cand_risk = sized * stop_dist / self._equity
                if cand_risk > budget:
                    scaled = budget * self._equity / max(stop_dist, 1e-4)
                    if scaled < MIN_ORDER_NOTIONAL:
                        logger.info("[BOOK-RISK] %s: book stop-risk budget "
                                    "exhausted (rho=%.2f, %d positions) — "
                                    "entry blocked", symbol, rho,
                                    len(self.positions))
                        return 0
                    logger.info("[BOOK-RISK] %s: entry shrunk to fit book "
                                "risk cap ($%d -> $%d, rho=%.2f)",
                                symbol, int(sized), int(scaled), rho)
                    sized = scaled
            except Exception:
                pass

        # --- 5. Hard caps including the NEW order ---
        existing_value = 0.0
        if symbol in self.positions:
            existing_value = self.positions[symbol].qty * self.positions[symbol].entry_price
        room = self.MAX_NOTIONAL_PER_SYMBOL - existing_value
        sized = min(sized, max(room, 0))

        # Leveraged ETF scaling: divide by leverage factor
        leverage = self._leveraged_etfs.get(symbol, 1)
        if leverage > 1:
            sized /= leverage

        if sized < MIN_ORDER_NOTIONAL:
            return 0
        return int(sized)

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

    def _roll_trade_budget_date(self):
        today = datetime.date.today().isoformat()
        if today != self._daily_trades_date:
            self._daily_trades = {}
            self._daily_trades_date = today

    def _count_trade(self, symbol: str):
        self._roll_trade_budget_date()
        self._daily_trades[symbol] = self._daily_trades.get(symbol, 0) + 1
        self._save_position_state()

    def _trade_budget_ok(self, symbol: str) -> bool:
        """True if the symbol hasn't used up today's entry budget."""
        from strategy_config import MAX_TRADES_PER_SYMBOL_PER_DAY
        self._roll_trade_budget_date()
        cap = MAX_TRADES_PER_SYMBOL_PER_DAY.get(self.get_asset_type(), 4)
        if self._daily_trades.get(symbol, 0) >= cap:
            return False
        return True

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

    def _check_flatten_request(self):
        """Honor a remote /flatten: liquidate this book, halt, clear flag.

        The flag is cleared BEFORE flattening so a partial failure can't
        re-trigger liquidation every 30s; failures are notified instead.
        """
        try:
            from notify import (flatten_requested, clear_flatten_request,
                                set_halt, notify)
            if not flatten_requested():
                return
        except Exception:
            return
        logger.warning("[FLATTEN] remote flatten requested — liquidating %s book",
                       self.get_asset_type())
        clear_flatten_request()
        set_halt('remote flatten')
        try:
            failures = emergency_flatten(self.api,
                                         symbols=self.get_symbol_universe())
            self.positions.clear()
            self._save_position_state()
            notify(f"FLATTEN {self.get_asset_type()}: done "
                   f"({'failures: ' + ', '.join(failures) if failures else 'all positions closed'}). "
                   f"Trading halted — /resume to re-enable entries.",
                   level='critical', dedupe_key=f'flatten-{self.get_asset_type()}')
        except Exception as e:
            logger.error("[FLATTEN] failed: %s", e)
            try:
                from notify import notify as _n
                _n(f"FLATTEN {self.get_asset_type()} FAILED: {e} — "
                   f"intervene manually", level='critical')
            except Exception:
                pass

    def _entries_allowed(self) -> bool:
        """Book-level entry gate shared by both loops (exits never gated).

        Gates: manual halt flag (Telegram /halt, GUI, or `touch
        trading_halt.flag`), then the scheduled macro-event stand-down
        (FOMC/CPI windows) — an hourly-bar model has no edge against an
        8:30 CPI print, in stocks OR crypto.
        """
        try:
            from notify import halt_active
            if halt_active():
                if self.cycle % 10 == 1:
                    logger.warning("[HALT] trading_halt.flag active — "
                                   "entries blocked (/resume to clear)")
                return False
        except Exception:
            pass
        try:
            from macro_calendar import macro_standdown, calendar_exhausted
            blocked, reason = macro_standdown()
            if blocked:
                if self.cycle % 10 == 1:
                    logger.info("[MACRO] entries paused: %s", reason)
                return False
            if calendar_exhausted() and self.cycle % 2000 == 1:
                logger.warning("[MACRO] static FOMC/CPI table has no future "
                               "events — refresh macro_calendar.py")
        except Exception:
            pass
        return True

    def _execute_buys(self, preds: dict, snapshots: dict):
        """Buy bullish symbols with all risk checks."""
        if not self._entries_allowed():
            return
        from collections import Counter
        vc = Counter()          # veto attribution for the window summary
        admitted = []           # symbols that cleared every gate
        n_candidates = 0        # symbols evaluated past the mechanical gates
        symbols = self.get_symbol_universe()
        for symbol in symbols:
            if not cooldown_ok(self.last_trade_time, symbol, self.COOLDOWN_MINUTES):
                vc['cooldown'] += 1
                continue

            if self._is_hard_stop_locked(symbol):
                vc['hard_stop_lockout'] += 1
                continue

            if not self._trade_budget_ok(symbol):
                vc['trade_budget'] += 1
                continue

            # Position cap check
            if symbol in self.positions:
                existing_value = self.positions[symbol].qty * self.positions[symbol].entry_price
                if existing_value >= self.MAX_NOTIONAL_PER_SYMBOL:
                    vc['position_cap'] += 1
                    continue

            # FAIL CLOSED: no prediction or no quote means we don't know
            # enough to buy. (Previously both gates sat inside an
            # `if pred is not None` block, so a missing model or a data
            # outage bought the whole universe ungated.)
            pred_return = preds.get(symbol)
            if pred_return is None:
                vc['no_pred'] += 1
                continue
            quote = self.get_quote(symbol)
            if quote is None:
                vc['no_quote'] += 1
                continue
            n_candidates += 1   # a real, evaluatable candidate this window
            snapshot = snapshots.get(symbol, {})

            # Prediction gate (higher bar if recently hard-stopped or mean-reverting)
            effective_threshold = self.trade_threshold
            if symbol in self.hard_stop_lockout:
                effective_threshold = self.trade_threshold * 1.5
            # Hurst < 0.45 = mean-reverting; momentum signals less reliable
            hurst = snapshots.get(symbol, {}).get('Hurst')
            if hurst is not None and hurst < 0.45:
                effective_threshold = max(effective_threshold,
                                          self.trade_threshold * 1.3)
            if not should_trade(pred_return, quote['spread_pct'],
                                asset_type=self.get_asset_type()):
                vc['cost_floor'] += 1
                self._journal_skip(symbol, 'cost_floor', pred=pred_return,
                                   snapshot=snapshot)
                continue
            if pred_return < effective_threshold:
                vc['below_threshold'] += 1
                self._journal_skip(symbol, 'below_threshold', pred=pred_return,
                                   snapshot=snapshot)
                continue

            # Winner's curse filter: if price > SMA20 + 2*ATR, require higher threshold
            sma20 = snapshot.get('SMA_20')
            atr = snapshot.get('ATR')
            if sma20 and atr and quote['midpoint'] > sma20 + 2 * atr:
                required = self.trade_threshold * 1.5
                if pred_return is not None and pred_return < required:
                    logger.info("%s: Winner's curse filter (extended move), need %.2f got %.4f",
                                symbol, required, pred_return)
                    vc['winners_curse'] += 1
                    self._journal_skip(symbol, 'winners_curse',
                                       pred=pred_return, snapshot=snapshot)
                    continue

            # Correlation check
            if self.corr_matrix and self.positions:
                allowed, avg_corr = check_portfolio_correlation(
                    list(self.positions.keys()), symbol, self.corr_matrix)
                if not allowed:
                    vc['correlation'] += 1
                    self._journal_skip(symbol, 'correlation', pred=pred_return,
                                       snapshot=snapshot)
                    continue

            # Macro regime halt check
            if self.macro_regime and self.macro_regime.should_halt_stocks and self.get_asset_type() == 'stock':
                logger.info("%s: Halted by VIX > 35", symbol)
                vc['macro_halt'] += 1
                continue

            # VIX > 25: block risky entries, allow safe-havens
            if (self.macro_regime and self.macro_regime.should_block_risky_entries
                    and self.get_asset_type() == 'stock'):
                from stock_config import SAFE_HAVEN_SYMBOLS
                if symbol not in SAFE_HAVEN_SYMBOLS:
                    logger.info("%s: Blocked — VIX > 25 defensive (non-safe-haven)", symbol)
                    vc['vix_block'] += 1
                    continue

            # Sentiment gate (veto first; multiplier folds into sizing tilt)
            gate, gate_reasons = sentiment_gate(symbol, self.get_asset_type())
            if gate <= 0:
                vc['sentiment_block'] += 1
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "sentiment_block",
                              "pred_return": pred_return,
                              "sentiment_gate": gate, "sentiment_reasons": gate_reasons})
                continue

            # LLM gate (veto first; multiplier folds into sizing tilt)
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            llm_reason = llm_info.get('r', '')
            if llm_s < LLM_VETO_THRESHOLD:
                vc['llm_veto'] += 1
                log_decision({"symbol": symbol, "action": "skip", "skip_reason": "llm_veto",
                              "pred_return": pred_return,
                              "llm_score": llm_s, "llm_reasoning": llm_reason})
                continue
            llm_mult = 0.5 + llm_s

            # Meta-labeling gate (veto + bounded sizing multiplier)
            meta_ok, meta_mult = self._meta_gate(symbol, pred_return, snapshots)
            if not meta_ok:
                vc['meta_veto'] += 1
                continue

            # q10 tail veto: a bullish mean prediction with a fat left
            # tail (10th-pct regression below the calibrated floor) is a
            # bad bet at 2:1 reward:risk even when the mean clears the bar
            q10 = snapshot.get('Q10')
            q10_floor = snapshot.get('Q10_Floor')
            if q10 is not None and q10_floor is not None and q10 < q10_floor:
                vc['q10_tail_veto'] += 1
                log_decision({"symbol": symbol, "action": "skip",
                              "skip_reason": "q10_tail_veto",
                              "pred_return": pred_return,
                              "q10": round(q10, 4),
                              "q10_floor": round(q10_floor, 4)})
                continue

            # Single risk-based sizing call (all bounds enforced inside)
            sized_notional = self._compute_position_size(
                symbol, pred_return, quote,
                sentiment_mult=gate, llm_mult=llm_mult, meta_mult=meta_mult)
            if sized_notional <= 0:
                vc['sizing_zero'] += 1
                self._journal_skip(symbol, 'sizing_zero', pred=pred_return,
                                   snapshot=snapshot)
                continue

            # Order timing jitter (prevent pattern detection)
            import random
            time.sleep(random.uniform(0, 5))

            logger.info("%s: Sizing $%d (pred=%.4f)", symbol, sized_notional,
                        pred_return if pred_return else 0)

            admitted.append(symbol)
            conv = self._conv_fields(symbol, pred_return, snapshot)
            self._place_and_track_buy(symbol, sized_notional, pred_return, quote,
                                      gate, gate_reasons, llm_s, llm_mult, llm_reason,
                                      conv=conv)
            time.sleep(1)

        self._journal_entry_window(n_candidates, admitted, vc)

    def _execute_entry_order(self, symbol, notional, quote):
        """Place and confirm the entry order. Returns (final_order, tactic).

        Base: marketable limit at mid+offset with market fallback.
        Crypto overrides with the maker bid-join ladder.
        """
        from order_utils import place_limit_order
        order = place_limit_order(self.api, symbol, 'buy', notional, quote)
        if order is None:
            return None, 'marketable'
        result = manage_order_lifecycle(self.api, order.id,
                                        timeout=self.ORDER_TIMEOUT,
                                        fallback_to_market=True)
        return result, 'marketable'

    def _place_and_track_buy(self, symbol, notional, pred_return, quote,
                             gate, gate_reasons, llm_s, llm_mult, llm_reason,
                             conv=None):
        """Place buy order and update position tracking. Override in subclasses for bracket orders.

        conv: optional conviction-context dict (wave-5 Tier1-1) merged
        into the buy journal row.
        """
        from market_data import get_live_atr

        result, entry_tactic = self._execute_entry_order(symbol, notional, quote)
        # Judge by ACQUIRED qty, not final order status: a partially
        # filled rung that timed out still bought coins that must be
        # tracked (and protected) — the old status=='filled' check left
        # partial fills invisible to stop management.
        try:
            partial_qty = float(getattr(result, 'filled_qty', 0) or 0)
        except (TypeError, ValueError):
            partial_qty = 0.0
        acquired = result is not None and (
            getattr(result, 'status', None) == 'filled' or partial_qty > 0)
        if acquired:
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
                        from volatility import get_sigma
                        garch_sigma = get_sigma(symbol, returns, bars=df,
                                                asset_type=self.get_asset_type())
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
                # Server-side disaster backstop (crypto: resting GTC stop_limit)
                self._after_entry_protection(symbol, self.positions[symbol])
                decision_price = quote['midpoint']
                slippage_bps = ((fill_price - decision_price) / decision_price * 1e4
                                if decision_price > 0 else None)
                buy_rec = {"symbol": symbol, "action": "buy",
                           "pred_return": pred_return,
                           "sentiment_gate": gate, "sentiment_reasons": gate_reasons,
                           "llm_multiplier": llm_mult, "llm_score": llm_s,
                           "llm_reasoning": llm_reason,
                           "final_notional": notional,
                           "decision_price": decision_price,
                           "fill_price": fill_price,
                           "slippage_bps": round(slippage_bps, 2) if slippage_bps is not None else None,
                           "entry_tactic": entry_tactic,
                           "maker": entry_tactic.startswith('maker'),
                           "skip_reason": None}
                if conv:
                    buy_rec.update(conv)
                log_decision(buy_rec)
                self.last_trade_time[symbol] = datetime.datetime.now()
                self._count_trade(symbol)

    def _sleep(self):
        """Sleep with thermal throttling."""
        try:
            from notify import ping_heartbeat
            ping_heartbeat(self.get_asset_type())
        except Exception:
            pass
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
