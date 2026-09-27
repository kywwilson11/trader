"""Template Method base class for trading loops.

Extracts the shared skeleton of crypto_loop.py and stock_loop.py into a
reusable base class: the cycle skeleton, stop management, sizing, the
meta/LLM gating, journaling and restart persistence.

NOTE (2026-07): the two loops do NOT stay in sync automatically.
stock_loop reimplements _execute_buys/_execute_sells with no super()
call and keeps its own buy/bracket path in sync with
_place_and_track_buy BY HAND; the entry-anchored stop arithmetic is
hand-duplicated at 7 sites (see the STOP-MATH SYNC CONTRACT in
_desired_stop_for). Changes on either side must be mirrored manually.
"""

import json
import os
import time
import datetime
import random
from abc import ABC, abstractmethod
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from log_config import get_logger
from types_mod import Position, MacroRegime
from order_utils import (
    manage_order_lifecycle, should_trade,
    cancel_all_open_orders, reconstruct_positions,
    check_circuit_breaker, emergency_flatten,
)
from predict_now import load_models
from trading_utils import (
    get_api, choose_inference_device, cooldown_ok,
    predict_symbol, compute_kelly_fraction,
    LLM_VETO_THRESHOLD, THERMAL_THROTTLE_TEMP, TEMP_LOG_EVERY_N_CYCLES,
)
from hw_monitor import get_gpu_temp
from sentiment import sentiment_gate
from llm_config import load_llm_config
from llm_analyst import analyze_trades, rich_context_enabled, build_compact_evidence
from fundamentals import get_fundamentals, format_fundamentals_for_llm
from trade_journal import log_decision
from trade_memory import record_trade
from macro_indicators import get_macro_regime
from volatility import compute_vol_adjusted_size
from portfolio import (
    check_portfolio_correlation, get_correlation_sizing_factor,
)
from regime_detector import get_cached_regime

logger = get_logger(__name__)


# c26 D17: per-book flatten flags. Writers (Telegram /flatten, GUI)
# still write notify's legacy shared flag; each loop fans that out to
# BOTH books and then consumes only its own flag.
_FLATTEN_FLAG_DIR = os.path.dirname(os.path.abspath(__file__))
FLATTEN_FLAG_STALE_SEC = 3600

# c26 T6 flags (module-level env-var booleans, the funding.py/shadow.py
# pattern; tests toggle the module attribute). Both default OFF with
# flag-OFF behavior byte-identical (pinned by tests/test_c26_T6.py).
# STOP_CLASSIFY_V2: 24h re-entry lockout only for server-stop fills
# classified 'hard'/'unknown' — 'trail' (ratcheted, usually profitable)
# exempt. The CLASSIFICATION itself always runs (measurement, DIRECT).
STOP_CLASSIFY_V2 = os.environ.get(
    'TRADER_STOP_CLASSIFY_V2', '0').strip().lower() in ('1', 'true', 'yes')
# STREAM_STOP_DETECT: when the REST resting-stop probe RAISES, a cached
# order_stream 'filled' event (terminal, monotonic-safe) recovers the fill
# instead of retrying next cycle indefinitely. Polling stays primary and
# authoritative. Requires TRADER_ORDER_STREAM=1 + combined-bots mode.
STREAM_STOP_DETECT = os.environ.get(
    'TRADER_STREAM_STOP_DETECT', '0').strip().lower() in ('1', 'true', 'yes')


def _breaker_server_fill_attrib() -> bool:
    """ENGINE r3 O3: strategy_config.BREAKER_SERVER_FILL_ATTRIB (default
    False), read at CALL time. TRADER_BREAKER_SERVER_FILL_ATTRIB
    ('1'/'true'/'yes'/'on'; any other non-empty value = off) wins over the
    constant when set. Never raises — anything unreadable is OFF."""
    v = os.environ.get('TRADER_BREAKER_SERVER_FILL_ATTRIB')
    if v not in (None, ''):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        import strategy_config as _sc
        return bool(getattr(_sc, 'BREAKER_SERVER_FILL_ATTRIB', False))
    except Exception:
        return False


# client_order_id is deliberately NOT journaled yet: make_client_order_id
# mints a uuid4 tail, which breaks the replay harness's journal-determinism
# pins (tests/test_engine_r2_replay_harness.py *_is_deterministic); the
# broker order (GET /v2/orders/{order_id}) carries it for any join.
_JOURNAL_ID_FIELDS = (('id', 'order_id'),)


def _order_journal_ids(order, keep_none=True) -> dict:
    """ENGINE r4 R4-a (measurement-only): the broker order id behind a
    journal row, {'order_id': str} (None when the object lacks one).
    keep_none=False drops an unknown id — sell rows keep their pinned
    legacy key-set when no order object is in hand. Lets
    scripts/fill_venue_slippage_report.py join live rows to broker fills.
    Never raises."""
    out = {}
    for attr, key in _JOURNAL_ID_FIELDS:
        try:
            v = getattr(order, attr, None)
            v = None if v is None else str(v)
        except Exception:
            v = None
        if v is not None or keep_none:
            out[key] = v
    return out


def _decision_quote_journal_keys(quote) -> dict:
    """ENGINE r4 R4-a (measurement-only): the quote the entry decision used
    (no new fetch) — decision_bid / decision_ask, and decision_quote_ts =
    the epoch-seconds fetch stamp the buy row's quote_age_s is measured
    from ('_fetched_ts', else get_quote's 'fetched_ts'); ENGINE r5:
    decision_quote_t = get_quote's 'quote_t', the EXCHANGE quote time
    (epoch s). None when unknown. Never raises."""
    out = {'decision_bid': None, 'decision_ask': None,
           'decision_quote_ts': None, 'decision_quote_t': None}
    try:
        if isinstance(quote, dict):
            out['decision_bid'] = quote.get('bid')
            out['decision_ask'] = quote.get('ask')
            out['decision_quote_ts'] = (quote.get('_fetched_ts')
                                        or quote.get('fetched_ts'))
            out['decision_quote_t'] = quote.get('quote_t')
    except Exception:
        pass
    return out


# Log labels per _breaker_record_server_fill call site (ENGINE r4 R4-b).
_SERVER_FILL_SITES = {
    'breaker': ('[CIRCUIT BREAKER]', 'breaker flatten'),
    'remote_flatten': ('[FLATTEN]', 'remote flatten'),
    'stablecoin_flatten': ('[CONTAGION]', 'stablecoin flatten'),
}

# c26 S3: one-time DERISK_STACK_V2 activation announcement (per process).
_derisk_v2_logged = False


def _flatten_flag_path(book: str) -> str:
    return os.path.join(_FLATTEN_FLAG_DIR, f'flatten_{book}.flag')


class BaseTradingLoop(ABC):
    """Abstract base for crypto and stock trading loops."""

    # --- Configuration (override in subclasses) ---
    NOTIONAL_PER_SYMBOL: float = 1000
    MAX_NOTIONAL_PER_SYMBOL: float = 3000
    ORDER_TIMEOUT: int = 30
    LOOP_INTERVAL: int = 30
    COOLDOWN_MINUTES: int = 60
    MAX_PREDICTION_WORKERS: int = 5
    PREDICTION_TIMEOUT_SEC: int = 45  # fan-out hard deadline (c26 D01)
    LLM_INTERVAL_SEC: int = 600
    LLM_SCORE_TTL_SEC: int = 7200  # stale scores expire (fail-open: no veto) — c26 D14
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
        # Per-book lockout file (2026-08 influence audit §3.4: "shared
        # unprefixed file lets books clobber each other"). Naming mirrors
        # _position_state_file: crypto keeps the legacy unprefixed name,
        # stock gets 'stock_hard_stop_lockout.json'. A prefixed book whose
        # file does not exist yet migrates ONCE from the legacy shared
        # file on load; saves only ever write the book's own file.
        _here = os.path.dirname(os.path.abspath(__file__))
        _lk = (f'{self.MODEL_PREFIX}_hard_stop_lockout.json'
               if self.MODEL_PREFIX else 'hard_stop_lockout.json')
        self._lockout_file = os.path.join(_here, _lk)
        self._legacy_lockout_file = os.path.join(_here, 'hard_stop_lockout.json')
        self._load_hard_stop_lockout()
        self.llm_scores: dict = {}
        self._last_llm_time = 0.0
        self._last_llm_symbols: set[str] = set()
        self._last_stale_force = 0.0
        self._llm_scores_ts: float | None = None   # last refresh (epoch)
        self._llm_fail_count: int = 0
        self._llm_backoff_until: float = 0.0
        self.model_mtime = 0
        self.cycle = 0
        self.macro_regime: MacroRegime | None = None
        # Static FOMC/CPI table staleness alarm state (one loud warning
        # per day, one notify per process — 2026-08 influence audit).
        self._macro_cal_alarm_date: str | None = None
        self._macro_cal_alarm_notified: bool = False
        # Gate-pricing instrumentation (2026-08 influence audit, open
        # question #4: "the highest-impact gates are the least priced").
        # _entries_block_info: why _entries_allowed last said False —
        # ('halt'|'macro_standdown', reason, window_id). _standdown_journaled:
        # (window_id, symbol) pairs already journaled (one priced skip row
        # per candidate per stand-down window). _vix25_gap_log_date: one
        # log per day for the SAFE_HAVEN-untradable gap (C3).
        self._entries_block_info: tuple | None = None
        self._standdown_journaled: set[tuple[str, str]] = set()
        self._vix25_gap_log_date: str | None = None
        self.corr_matrix: dict = {}
        from drawdown import PEAK_SEED
        self._equity: float = PEAK_SEED
        self._peak_equity: float = PEAK_SEED
        self._peak_from_seed = True  # placeholder until the first REAL equity read (c26 D15)
        self._buys_allowed: bool = True
        self._halted_until: datetime.datetime | None = None
        # Per-symbol daily trade budget: caps fee bleed from signal jitter
        # re-trading the same name all day
        self._daily_trades: dict[str, int] = {}   # symbol -> count (today)
        self._daily_trades_date: str = datetime.date.today().isoformat()
        # Stop-breach confirmation state (2-consecutive-reading rule)
        self._pending_breach: dict[str, str] = {}
        # IA-4 flag-family state (2026-08 influence audit; every consumer
        # getattr-guards these — unit-test stubs bypass __init__):
        # _pending_signal_exit: symbol -> cycle of an ARMED first signal-
        #   exit reading (consulted only when SIGNAL_EXIT_CONFIRM_READS>=2).
        # _position_entry_ts: symbol -> epoch of first entry (persisted;
        #   the crypto vertical-barrier age anchor).
        # _vertical_journaled: (symbol, int(entry_ts)) already journaled as
        #   vertical_barrier would-fire events (one row per position).
        # _last_marks: symbol -> last midpoint seen by _manage_stops (the
        #   BREAKER_PER_BOOK mark source; <= one cycle stale).
        # _breaker_base: per-book breaker baseline window (BREAKER_PER_BOOK).
        self._pending_signal_exit: dict[str, int] = {}
        self._position_entry_ts: dict[str, float] = {}
        self._vertical_journaled: set[tuple[str, int]] = set()
        self._last_marks: dict[str, float] = {}
        self._breaker_base: dict | None = None
        # Conviction instrumentation (wave-5 Tier1-1): last meta probability
        # per symbol, stashed by _meta_gate so buy/skip rows can record it
        # without recomputing. Measurement-only — never gates.
        self._last_meta_p: dict[str, float] = {}
        # LLM veto strikes: liquidation needs 2 consecutive vetoing analyses
        self._veto_strikes: dict[str, int] = {}
        # Freshness stamp for _last_meta_p (journal emission only): cycle
        # number when _meta_gate last computed the probability. _conv_fields
        # emits meta_p/conviction_tier only for same-cycle values — gates
        # earlier in the funnel were journaling values from arbitrarily
        # older cycles as if current (2026-07 panel). _last_meta_p itself
        # stays float-valued: the subclass prediction-cache writers and
        # tests pin that shape.
        self._last_meta_p_cycle: dict[str, int] = {}
        # 2026-07 panel instrumentation state: last successful macro
        # refresh, last successful equity read (cycle), hot-reload failure
        # backoff, position-state write dedup.
        self._macro_regime_ts: float | None = None
        self._equity_cycle: int | None = None
        self._failed_reload: tuple = (None, 0.0)
        self._last_state_blob: str | None = None
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
        """Get a quote dict for a symbol, or None when unavailable.

        Callers hard-require at least {'midpoint', 'spread_pct'}
        (canonical shape: types_mod.Quote). None fails closed on the buy
        path (no entry).
        """

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

        # Remote /flatten first, BEFORE the market-hours gate: the stock
        # book must be able to liquidate off-hours (day orders queue for
        # the open) — c26 D17.
        self._check_flatten_request()

        if not self.check_market_hours():
            if self.cycle == 1 or self.cycle % 20 == 0:
                logger.info("[WAIT] Market closed. Next check in %ds...", self.LOOP_INTERVAL)
            # Dead-man's-switch keepalive: without this the stock book's
            # heartbeat goes silent ~17.5h/day plus weekends, forcing a
            # watchdog grace period so wide it can't catch a real death.
            # ping_heartbeat is self-rate-limited (1/min) and never raises.
            try:
                from notify import ping_heartbeat
                ping_heartbeat(self.get_asset_type())
            except Exception:
                pass
            time.sleep(self.LOOP_INTERVAL)
            return

        logger.info("--- CYCLE %d: %s ---", self.cycle,
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Pre-trade checks (also refreshes self._buys_allowed)
        self._circuit_breaker_check()

        self.flatten_before_close()

        # Stop-loss management FIRST — before hot-reload and the hourly
        # maintenance block (correlation rebuild alone is 11-28s serial),
        # so protective exits are never queued behind housekeeping (c26 B08).
        # NOTE: on the very first cycle stops now run before the first
        # macro-regime fetch, so stop_mult tightening starts one cycle later
        # at process start; the 2-reading breach confirmation makes that a
        # non-event.
        # c26 T6: per-phase time.monotonic() checkpoints (measurement,
        # DIRECT) — no statement moved; the phase sums land in one
        # cycle_latency journal row before the sleep.
        _t = time.monotonic()
        self._manage_stops()
        stops_s = time.monotonic() - _t

        _t = time.monotonic()
        # Hot-reload model
        self._hot_reload_check()

        # Update macro regime (every 10 cycles to save API calls)
        if self.cycle % 10 == 1:
            self._update_macro_regime()
            self._update_equity()
            self._update_correlations()
            self._record_account_risk()

        # Log GPU temp periodically
        if self.cycle % TEMP_LOG_EVERY_N_CYCLES == 0:
            temp = get_gpu_temp()
            if temp is not None:
                logger.info("[HW] GPU temp: %.0fC", temp)
        maint_s = time.monotonic() - _t

        # Predictions
        _t = time.monotonic()
        benchmark = self.get_benchmark_close()
        if benchmark is None:
            logger.warning("[BENCHMARK] Benchmark data unavailable — predictions will lack relative strength")
        fetch_s = time.monotonic() - _t

        _t = time.monotonic()
        preds, snapshots = self._get_predictions(benchmark)

        # Challenger shadow predictions (hourly side-by-side log; no
        # trading impact — promotion is decided by the daily DM test)
        try:
            from shadow import maybe_log_shadow
            maybe_log_shadow(self, preds, benchmark)
        except Exception:
            pass
        predict_s = time.monotonic() - _t

        # LLM analysis (throttled)
        _t = time.monotonic()
        self._run_llm_analysis(preds)
        llm_s = time.monotonic() - _t

        _t = time.monotonic()
        # Sell bearish positions
        self._execute_sells(preds)

        # LLM veto sells
        self._execute_llm_veto_sells()
        sells_s = time.monotonic() - _t

        # Buy (suppressed while halted or when risk state is unknown)
        # NOTE: the gates-vs-orders split inside _execute_buys is
        # deliberately folded into buys_s — the per-entry jitter sleeps
        # (random.uniform(0,5) + 1s pacing) dominate and stock_loop
        # reimplements _execute_buys separately. ~2880 rows/day crypto
        # ≈ 0.4MB/day, pruned with the journals.
        buys_s = 0.0
        if self._buys_allowed:
            _t = time.monotonic()
            self._execute_buys(preds, snapshots)
            buys_s = time.monotonic() - _t

        self._journal_cycle_latency(stops_s=stops_s, maint_s=maint_s,
                                    fetch_s=fetch_s, predict_s=predict_s,
                                    llm_s=llm_s, sells_s=sells_s,
                                    buys_s=buys_s)

        # Thermal throttling
        self._sleep()

    def _journal_cycle_latency(self, **phases):
        """One cycle_latency journal row per trading cycle (c26 T6,
        measurement — DIRECT). Never raises."""
        try:
            row = {'action': 'cycle_latency', 'book': self.get_asset_type(),
                   'cycle': self.cycle}
            row.update({k: round(float(v), 3) for k, v in phases.items()})
            row['total_s'] = round(sum(float(v) for v in phases.values()), 3)
            log_decision(row)
        except Exception:
            pass

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
        except Exception as e:
            # A corrupt/mismatched artifact must fail CLOSED like the FNF
            # branch, not escape run() — in combined mode that crash took
            # the healthy book down too (2026-09 G1 F2). Retry is handed to
            # _hot_reload_check's existing >=300 s backoff: model_mtime=None
            # differs from the reload key, and _failed_reload holds it.
            logger.error("Model load FAILED (%s: %s) — buys DISABLED (fail closed); "
                         "exits and stops still run; retrying via hot-reload backoff",
                         type(e).__name__, e)
            self.model = None
            self.config = {}
            self.scaler_X = None
            self.feature_cols = None
            from trading_utils import model_reload_key
            self._failed_reload = (model_reload_key(self.MODEL_PREFIX), time.time())
            self.model_mtime = None
            return

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
            # Prune cooldown timers past 2x the cooldown window before
            # persisting: cooldown_ok treats absent == expired, so this
            # provably cannot change any cooldown verdict — it only stops
            # the dict and the state file growing monotonically across
            # universe rotations (2,880 rewrites/day/book on flash).
            cutoff_dt = (datetime.datetime.now()
                         - datetime.timedelta(minutes=self.COOLDOWN_MINUTES * 2))
            for s in [s for s, t in self.last_trade_time.items() if t < cutoff_dt]:
                self.last_trade_time.pop(s, None)
            data = {
                'hwm': {s: p.high_water_mark for s, p in self.positions.items()},
                'trailing': {s: p.trailing_activated for s, p in self.positions.items()},
                'last_trade': {s: t.timestamp() for s, t in self.last_trade_time.items()},
                'daily_trades': {'date': self._daily_trades_date,
                                 'counts': self._daily_trades},
                # Persist the account high-water mark so the drawdown
                # de-leveraging ladder survives restarts (wave-8 #4). Without it
                # a restart mid-drawdown reset the peak to ~current equity and
                # silently disabled the ladder while underwater.
                'peak_equity': self._peak_equity,
                # Entry timestamps for held positions (IA-4): the crypto
                # vertical-barrier age anchor must survive restarts or a
                # crash would reset every position's max-hold clock.
                'entry_ts': {s: t for s, t in
                             getattr(self, '_position_entry_ts', {}).items()
                             if s in self.positions},
            }
            # Skip the write when nothing changed (the common empty-book
            # case previously rewrote an identical file every 30s cycle).
            blob = json.dumps(data)
            if blob == getattr(self, '_last_state_blob', None):
                return
            tmp = self._position_state_file() + '.tmp'
            with open(tmp, 'w') as f:
                f.write(blob)
            os.replace(tmp, self._position_state_file())
            self._last_state_blob = blob
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
        self._update_equity()  # real equity BEFORE the peak restore (c26 D15): the restore ratchet must never run against the $100k seed
        from market_data import get_live_atr
        symbols = self.get_symbol_universe()
        raw_positions = reconstruct_positions(self.api, symbols)
        saved = self._load_position_state()
        # Restore the account high-water mark BEFORE any equity ratchet so the
        # drawdown ladder stays armed across a restart-mid-drawdown (wave-8 #4).
        # restore_peak_equity never drops below current equity, and the later
        # _update_equity ratchet only moves UP, so a higher prior peak survives.
        if 'peak_equity' in saved:
            from drawdown import restore_peak_equity
            # When the equity fetch failed, restore against 0 so a
            # sub-seed saved peak is honored instead of inflated to the
            # placeholder; a degenerate save keeps the seed path.
            cur = 0.0 if getattr(self, '_peak_from_seed', False) else self._equity
            restored = restore_peak_equity(saved.get('peak_equity'), cur,
                                           seed=cur)
            if restored > 0:
                self._peak_equity = restored
                self._peak_from_seed = False
        saved_hwm = saved.get('hwm', {})
        saved_trailing = saved.get('trailing', {})
        for sym, info in raw_positions.items():
            if not info['entry_price'] > 0:
                # Log-only (the consequences are owner items, ENGINE r1 W1):
                # with no basis the entry-anchored stop/TP/risk math degrades.
                logger.warning("[RECONSTRUCT] %s: cost basis unknown "
                               "(avg_entry_price=%s) — hard stop and TP "
                               "disabled, protection is a pure %.0f%% trail "
                               "from the high-water mark, book stop-risk "
                               "counts it as 0", sym, info['entry_price'],
                               self.TRAIL_PCT * 100)
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

        # Restore entry timestamps for still-held positions (IA-4 vertical-
        # barrier age anchor). Unknown-age positions (pre-upgrade state
        # files) simply have no barrier clock — fail-open, journal nothing.
        try:
            self._position_entry_ts = {
                s: float(t) for s, t in (saved.get('entry_ts') or {}).items()
                if s in self.positions}
        except (TypeError, ValueError):
            self._position_entry_ts = {}

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

        # Pre-load cached LLM scores from disk — only when the LLM gate is
        # enabled, and only entries younger than LLM_SCORE_TTL_SEC (c26 D14:
        # a week-old cached veto must not gate entries after a restart).
        try:
            if load_llm_config().get("enabled"):
                from llm_analyst import load_analysis
                from datetime import datetime as _dt, timezone as _tz
                section = load_analysis().get(self.get_asset_type(), {})
                fresh, newest = {}, None
                for sym, entry in section.items():
                    ts_str = (entry or {}).get('timestamp', '')
                    try:
                        ts = _dt.fromisoformat(ts_str)
                    except (TypeError, ValueError):
                        continue
                    age = (_dt.now(_tz.utc) - ts).total_seconds()
                    if age <= self.LLM_SCORE_TTL_SEC:
                        fresh[sym] = entry
                        newest = ts if newest is None or ts > newest else newest
                if fresh:
                    self.llm_scores = fresh
                    self._llm_scores_ts = newest.timestamp()
                    logger.info("[LLM] Loaded %d fresh cached score(s) from disk",
                                len(fresh))
                elif section:
                    logger.info("[LLM] disk cache present but stale (> %dh)"
                                " — starting with no scores (fail-open)",
                                self.LLM_SCORE_TTL_SEC // 3600)
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

        # BREAKER_PER_BOOK (IA-4, ledger §3.2 KEEP-COND): OFF = account-wide
        # Alpaca last_equity baseline exactly as today; ON = this book's own
        # P&L vs a per-book, crypto-weekend-aware baseline window.
        try:
            from strategy_config import BREAKER_PER_BOOK
        except Exception:
            BREAKER_PER_BOOK = False
        try:
            if BREAKER_PER_BOOK:
                tripped, dd = self._book_breaker_check()
            else:
                tripped, dd = check_circuit_breaker(
                    self.api, max_drawdown_pct=self.CIRCUIT_BREAKER_PCT)
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
            pre_flatten = dict(self.positions)
            # Trip-event journal row (2026-08 influence audit §3.2 circuit
            # breaker KEEP-COND: "Per-book (or weekend-aware) baselines +
            # trip journaling"). Records the baseline actually used —
            # account-wide last_equity = previous EQUITY-TRADING-DAY close,
            # so crypto weekend trips are judged against Friday's close —
            # the evidence substrate for the per-book-baseline decision.
            # Measurement-only; never blocks the flatten.
            # Halt latch end: flag OFF = the account baseline reset
            # (~16:05 ET, unchanged); BREAKER_PER_BOOK = this book's own
            # window end (crypto: next UTC midnight — weekend-aware).
            try:
                _halt_until = (self._breaker_window_end() if BREAKER_PER_BOOK
                               else self._next_baseline_reset())
            except Exception:
                _halt_until = self._next_baseline_reset()
            try:
                _base_equity = _cur_equity = None
                try:
                    _acct = self.api.get_account()
                    _cur_equity = float(_acct.equity)
                    _base_equity = float(_acct.last_equity)
                except Exception:
                    pass
                _base_kind = 'account_last_equity'
                if BREAKER_PER_BOOK:
                    _base_kind = 'book_window_v2'
                    _bb = getattr(self, '_breaker_base', None) or {}
                    if _bb.get('equity') is not None:
                        _base_equity = _bb['equity']
                import zoneinfo
                _now_et = datetime.datetime.now(
                    zoneinfo.ZoneInfo('US/Eastern'))
                log_decision({
                    'action': 'circuit_breaker_trip',
                    'asset_type': self.get_asset_type(),
                    'drawdown_pct': round(dd * 100, 3),
                    'threshold_pct': round(self.CIRCUIT_BREAKER_PCT * 100, 2),
                    'baseline_kind': _base_kind,
                    'baseline_equity': _base_equity,
                    'equity': _cur_equity,
                    'weekend_window': _now_et.weekday() >= 5,
                    'n_positions_at_trip': len(pre_flatten),
                    'halted_until': _halt_until.isoformat(),
                })
            except Exception:
                pass
            failures = emergency_flatten(self.api, symbols=self.get_symbol_universe())
            if failures:
                logger.error("[CIRCUIT BREAKER] Unconfirmed flattens: %s — kept "
                             "tracked; stops manage them while entries stay halted",
                             ', '.join(failures))
                # Keep failed symbols tracked so stops still manage them.
                # emergency_flatten reports BROKER-format symbols ('BTCUSD')
                # while self.positions is keyed universe-format ('BTC/USD') —
                # compare on the normalized form, or the filter keeps NOTHING
                # for crypto and an unflattened, unprotected position silently
                # vanishes from tracking (2026-07 review P1). The
                # list-positions sentinel means broker state is UNKNOWN:
                # keep every position tracked rather than guess.
                if '<list_positions failed>' in failures:
                    pass  # unknown broker state — keep all positions tracked
                else:
                    failed_norm = {f.replace('/', '') for f in failures}
                    self.positions = {s: p for s, p in self.positions.items()
                                      if s.replace('/', '') in failed_norm}
            else:
                self.positions.clear()
            # Journal ONLY the exits the flatten actually released. The old
            # pre-flatten loop fabricated a 'circuit_breaker' sell row for
            # positions whose flatten then FAILED (double-counted round
            # trip) and added N quote round-trips of latency BEFORE the
            # liquidation. NOTE: estimated=True rows are EXCLUDED from
            # Kelly's sample by compute_kelly_fraction — these rows feed
            # the trade log/GUI, not sizing (the old comment claimed the
            # opposite).
            # BREAKER_SERVER_FILL_ATTRIB (ENGINE r3 O3, default OFF): this
            # check runs BEFORE _manage_stops, so on a gap the resting
            # server stops may already be FILLED — ON, such a position is
            # journaled as the server_stop it was (real fill, lockout)
            # instead of an estimated row. Checked only AFTER the flatten:
            # zero added latency to the liquidation.
            _attrib = _breaker_server_fill_attrib()
            for sym, pos in pre_flatten.items():
                if sym in self.positions:
                    continue    # flatten failed/unknown — still open, no exit row
                if _attrib and self._breaker_record_server_fill(sym, pos):
                    continue
                quote = self.get_quote(sym)
                px = quote['midpoint'] if quote else pos.entry_price
                pnl = ((px - pos.entry_price) / pos.entry_price * 100
                       if pos.entry_price > 0 else 0.0)
                record_trade(sym, 'sell', pos.entry_price, px, pnl,
                             exit_reason='circuit_breaker', estimated=True)
            self._halted_until = _halt_until
            self._buys_allowed = False
            logger.info("[CIRCUIT BREAKER] Halting new entries until %s (baseline reset)",
                        self._halted_until.isoformat())
            return True

        self._buys_allowed = True
        return False

    def _breaker_record_server_fill(self, sym, pos, site='breaker') -> bool:
        """BREAKER_SERVER_FILL_ATTRIB ON only (ENGINE r3 O3).

        site (ENGINE r4 R4-b): the flatten that released the position —
        'breaker' (default, byte-identical), 'remote_flatten' or
        'stablecoin_flatten'; sets detect_source and the log labels.

        True iff pos.stop_order_id is 'filled' at the broker and the exit
        was journaled through the SAME calls as _manage_stops' server-fill
        branch (_classify_server_stop -> _record_confirmed_exit
        'server_stop' -> last_trade_time -> _apply_server_stop_lockout),
        plus detect_source='breaker'. False (caller writes today's
        estimated circuit_breaker row) when there is no id, get_order
        raises, the status is anything but 'filled', or the journal write
        raises. Never raises: the breaker's halt latch is set after the
        caller's loop.
        """
        tag, what = _SERVER_FILL_SITES.get(site, (f'[{site}]', site))
        oid = getattr(pos, 'stop_order_id', None)
        if not oid:
            return False
        try:
            so = self.api.get_order(oid)
        except Exception as e:
            logger.debug("%s %s: stop %s status check failed "
                         "(%s) — estimated exit row", tag, sym, oid, e)
            return False
        if getattr(so, 'status', None) != 'filled':
            return False
        logger.info("[STOP-FILL] %s: resting stop filled at $%s before the "
                    "%s — journaled as server_stop", sym,
                    getattr(so, 'filled_avg_price', None), what)
        llm_info = (getattr(self, 'llm_scores', None) or {}).get(sym, {})
        kind, stop_px = self._classify_server_stop(sym, pos, so)
        try:
            self._record_confirmed_exit(
                sym, pos, so, None, exit_reason='server_stop',
                llm_score=llm_info.get('s'),
                reasoning=llm_info.get('r', ''),
                extra={'server_stop_kind': kind, 'stop_px': stop_px,
                       'detect_source': site})
        except Exception as e:
            logger.error("%s %s: server_stop journal failed "
                         "(%s) — estimated exit row instead", tag, sym, e)
            return False
        self.last_trade_time[sym] = datetime.datetime.now()
        try:
            self._apply_server_stop_lockout(sym, kind)
        except Exception as e:
            logger.error("%s %s: lockout write failed (%s)",
                         tag, sym, e)
        return True

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

    # --- BREAKER_PER_BOOK (IA-4, 2026-08 influence audit §3.2) ------------
    # OFF (default): none of these run on the money path — the breaker uses
    # order_utils.check_circuit_breaker exactly as today. ON: each book
    # measures ITS OWN P&L against a baseline captured at the book's own
    # window roll (crypto: UTC midnight daily, weekends included; stock:
    # the ~16:05 ET account baseline reset).

    def _breaker_window_end(self) -> datetime.datetime:
        """End of the current per-book breaker window, UTC (v2 only)."""
        if self.get_asset_type() == 'crypto':
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now + datetime.timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        return self._next_baseline_reset()

    def _breaker_note_realized(self, pnl_dollars: float):
        """Accumulate this book's realized P&L into the current breaker
        window (called from the confirmed-exit chokepoints). No-op unless
        BREAKER_PER_BOOK is on and a baseline window exists. Never raises.

        Approximation, documented: a position entered BEFORE the baseline
        roll books its full entry-to-exit P&L into this window (its ref
        would ideally be the baseline mark) — overstates an already-losing
        position's in-window loss, i.e. errs toward tripping EARLIER.
        """
        try:
            from strategy_config import BREAKER_PER_BOOK
            if not BREAKER_PER_BOOK:
                return
            base = getattr(self, '_breaker_base', None)
            if base is not None:
                base['realized'] = base.get('realized', 0.0) + float(pnl_dollars)
        except Exception:
            pass

    def _book_breaker_check(self) -> tuple[bool, float | None]:
        """(tripped, drawdown) under BREAKER_PER_BOOK.

        Book drawdown = -(realized-in-window + open-position mark drift
        since the window roll) / baseline equity. Marks come from
        _manage_stops' stash (<= one cycle stale); a position entered after
        the roll anchors at its entry price; positions with no mark yet
        contribute 0 (under-counts briefly at process start — the breaker
        is a backstop, not the sizing stack). Baseline equity is the
        ACCOUNT equity at the roll (the shared denominator both books'
        percentages are judged against). Returns (False, None) when the
        account API is unreachable — the caller fails closed exactly as on
        the legacy path.

        Documented approximation: the baseline lives in process memory
        only — a mid-window RESTART re-captures it at current equity,
        forgetting the window's accumulated loss (errs toward NOT
        tripping until the next roll; the legacy account-wide baseline
        survives restarts via the broker's last_equity). Acceptable for
        a backstop; persist the window blob if Jetson trip rows ever
        show restart-straddled drawdowns mattering.
        """
        try:
            acct = self.api.get_account()
            equity = float(acct.equity)
        except Exception as e:
            logger.warning("[CIRCUIT BREAKER] book-v2 equity fetch failed: %s", e)
            return False, None
        window_end = self._breaker_window_end()
        marks = getattr(self, '_last_marks', {}) or {}
        base = getattr(self, '_breaker_base', None)
        if not base or base.get('window_end') != window_end.isoformat():
            # Window roll: capture baseline equity + per-position refs.
            self._breaker_base = {
                'window_end': window_end.isoformat(),
                'equity': equity,
                'refs': {s: marks.get(s, p.entry_price)
                         for s, p in self.positions.items()},
                'realized': 0.0,
            }
            return False, 0.0
        unreal = 0.0
        for s, p in self.positions.items():
            ref = base['refs'].get(s)
            if ref is None:
                # Entered after the roll: anchor at entry.
                ref = p.entry_price
                base['refs'][s] = ref
            mark = marks.get(s)
            if mark is None or not ref or ref <= 0:
                continue
            unreal += (float(mark) - float(ref)) * p.qty
        pnl = unreal + base.get('realized', 0.0)
        be = base.get('equity') or 0.0
        if be <= 0:
            return False, 0.0
        dd = -pnl / be
        return dd >= self.CIRCUIT_BREAKER_PCT, dd

    def _hot_reload_check(self):
        """Check if model files changed and reload (keyed on the manifest)."""
        from trading_utils import model_reload_key
        new_mtime = model_reload_key(self.MODEL_PREFIX)
        if new_mtime != self.model_mtime:
            # Backoff: a corrupt artifact otherwise re-runs the full
            # torch/joblib load every 30s cycle forever (8GB Jetson). A
            # genuinely new/repaired artifact changes the reload key and
            # is picked up immediately; the same failed key retries at
            # >=300s.
            failed_key, failed_at = getattr(self, '_failed_reload', (None, 0.0))
            if new_mtime == failed_key and time.time() - failed_at < 300:
                return
            logger.info("[HOT-RELOAD] Model files changed, reloading...")
            try:
                inference_device = choose_inference_device()
                self.model, self.config, self.scaler_X, self.feature_cols = \
                    load_models(inference_device, prefix=self.MODEL_PREFIX)
                self.trade_threshold = self.config.get('trade_threshold', 0.15)
                self.model_mtime = new_mtime
                self._failed_reload = (None, 0.0)
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
                self._failed_reload = (new_mtime, time.time())
                logger.error("[HOT-RELOAD] Failed: %s", e)

    def _update_macro_regime(self):
        """Fetch and cache current macro regime."""
        try:
            self.macro_regime = get_macro_regime(self.api, self.get_asset_type())
            self._macro_regime_ts = time.time()
            logger.info("[MACRO] Regime: %s (sizing=%.2fx, stops=%.2fx)",
                        self.macro_regime.regime_label,
                        self.macro_regime.sizing_mult,
                        self.macro_regime.stop_mult)

            # BTC trailing-RV regime input (c26 S3 / B06): warm the state on
            # every regime refresh (~5 min) so the v2 shadow journal — and any
            # later DERISK_STACK_V2 flip — has history from day one. Usually a
            # _bar_cache hit (the prediction path fetches the same bars).
            # Measurement-only while the flag is OFF.
            if self.get_asset_type() == 'crypto':
                try:
                    from market_data import fetch_bars_alpaca
                    from volatility import update_crypto_rv_state
                    btc_bars = fetch_bars_alpaca(self.api, 'BTC/USD')
                    if btc_bars is not None:
                        update_crypto_rv_state('BTC/USD', btc_bars)
                except Exception as e:
                    logger.debug("[DERISK-V2] BTC RV state update failed: %s", e)

            # Emergency stablecoin flatten for crypto
            if self.macro_regime.stablecoin_alert and self.get_asset_type() == 'crypto':
                if self.macro_regime.sizing_mult == 0:
                    logger.warning("[CONTAGION] Stablecoin emergency! Flattening crypto...")
                    pre_flatten = dict(self.positions)
                    failures = emergency_flatten(self.api, symbols=self.get_symbol_universe())
                    # Same failure-tracking contract as the circuit-breaker
                    # branch (2026-07 review P1): failures come back
                    # BROKER-format ('BTCUSD'), positions are keyed
                    # universe-format ('BTC/USD'), and the list-positions
                    # sentinel means broker state is UNKNOWN — keep all.
                    if '<list_positions failed>' in failures:
                        pass  # unknown broker state — keep all positions tracked
                    elif failures:
                        failed_norm = {f.replace('/', '') for f in failures}
                        self.positions = {s: p for s, p in self.positions.items()
                                          if s.replace('/', '') in failed_norm}
                    else:
                        self.positions.clear()
                    # ENGINE r4 R4-b (journal/state only; no order-flow
                    # change): persist the release and journal each released
                    # position like the breaker/remote-flatten sites — an
                    # estimated row at the quote mid (excluded from Kelly),
                    # or, BREAKER_SERVER_FILL_ATTRIB ON, the server_stop its
                    # already-filled resting stop was. Was: no exit row and
                    # no state save.
                    self._save_position_state()
                    _attrib = _breaker_server_fill_attrib()
                    for sym, pos in pre_flatten.items():
                        if sym in self.positions:
                            continue    # flatten failed/unknown — still open
                        try:
                            if _attrib and self._breaker_record_server_fill(
                                    sym, pos, site='stablecoin_flatten'):
                                continue
                            q = self.get_quote(sym)
                            px = q['midpoint'] if q else pos.entry_price
                            pnl = ((px - pos.entry_price) / pos.entry_price * 100
                                   if pos.entry_price > 0 else 0.0)
                            record_trade(sym, 'sell', pos.entry_price, px, pnl,
                                         exit_reason='stablecoin_flatten',
                                         estimated=True)
                        except Exception as je:
                            logger.error("[CONTAGION] %s: exit journal "
                                         "failed (%s)", sym, je)
        except Exception as e:
            # Escalated from debug (2026-07 panel): a silent failure here
            # freezes sizing_mult/stop_mult/vix at the LAST GOOD read, and
            # the degraded-mode clamp keys on `vix is None` so it cannot
            # see a stale-but-present regime. TTL/expiry is an owner
            # decision; visibility ships now. Runs every 10 cycles, so
            # this is inherently rate-limited.
            ts = getattr(self, '_macro_regime_ts', None)
            if ts is None:
                logger.warning("[MACRO] Regime update failed (%s) — no regime "
                               "available", e)
            else:
                logger.warning("[MACRO] Regime update failed (%s) — reusing "
                               "regime from %.0f min ago", e,
                               (time.time() - ts) / 60)

    def _update_equity(self):
        """Update cached equity."""
        try:
            acct = self.api.get_account()
            self._equity = float(acct.equity)
            self._equity_cycle = self.cycle
            if getattr(self, '_peak_from_seed', False):
                # First REAL equity read: drop the $100k placeholder peak —
                # it pinned every sub-$100k account into a permanent
                # drawdown-ladder haircut (c26 D15).
                self._peak_equity = self._equity
                self._peak_from_seed = False
            from drawdown import update_peak_equity
            self._peak_equity = update_peak_equity(self._peak_equity, self._equity)
        except Exception as e:
            # The only swallowed exception feeding a live sizing
            # denominator (risk base + book-risk cap). Runs every 10
            # cycles, so inherently rate-limited. Fail-closed handling is
            # an owner decision; visibility ships now (2026-07 panel).
            logger.warning("[EQUITY] account refresh failed (%s) — sizing on "
                           "equity from cycle %s ($%.0f)", e,
                           getattr(self, '_equity_cycle', None) or 'seed',
                           self._equity)

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

        STOP-MATH SYNC CONTRACT (2026-07 review): the entry-anchored
        stop-distance arithmetic (entry_ATR * ATR_STOP_MULTIPLIER / price,
        clamped [ATR_STOP_FLOOR_PCT, ATR_STOP_CEIL_PCT], fallback
        STOP_LOSS_PCT) is duplicated at base_loop._reconstruct_positions,
        base_loop._compute_position_size, base_loop._place_and_track_buy,
        crypto_loop._stop_distance_for, stock_loop._execute_buys (bracket),
        stock_loop._prepare_overnight_keepers and
        stock_loop._replace_protective_stops. Change one -> change ALL
        (helper consolidation queued for owner review).
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

    def _classify_server_stop(self, symbol, pos, order):
        """('hard'|'trail'|'unknown', stop_px) for a filled server stop.

        Measurement-only (c26 T6); never raises. Order of evidence:
        (1) stock trailing upgrade — pos.trailing_activated means
        stop_order_id IS a native trailing_stop -> 'trail'; (2) the fill's
        stop_price (via the alpaca_compat shim / legacy SDK); (3) crypto's
        _resting_stop_px cache via getattr (crypto_loop is out of packet
        scope — read-only). The hard level is the entry-anchored
        entry*(1-stop_dist) from _desired_stop_for — the ONE source of
        stop arithmetic (STOP-MATH SYNC CONTRACT: no 8th copy). 'trail'
        iff px >= entry or px > hard*(1+1e-3). CAVEAT: macro stop_mult
        drift between placement and fill can misclassify near the
        boundary; 'unknown' and boundary cases stay conservative.
        """
        try:
            px = None
            try:
                sp = getattr(order, 'stop_price', None)
                if sp is not None:
                    px = float(sp)
            except (TypeError, ValueError):
                px = None
            if getattr(pos, 'trailing_activated', False):
                return 'trail', px
            if px is None:
                cache = getattr(self, '_resting_stop_px', None)
                if isinstance(cache, dict):
                    cpx = cache.get(symbol)
                    try:
                        px = float(cpx) if cpx is not None else None
                    except (TypeError, ValueError):
                        px = None
            if px is None:
                return 'unknown', None
            try:
                _, stop_dist, _, _ = self._desired_stop_for(pos)
                hard = pos.entry_price * (1 - stop_dist)
            except Exception:
                return 'unknown', px
            if px >= pos.entry_price or (hard > 0 and px > hard * (1 + 1e-3)):
                return 'trail', px
            return 'hard', px
        except Exception:
            return 'unknown', None

    def _apply_server_stop_lockout(self, symbol, kind):
        """Post-server-stop-fill lockout (c26 T6).

        Flag OFF: unconditional lockout — byte-identical to today.
        TRADER_STOP_CLASSIFY_V2: lockout ONLY for 'hard' and 'unknown'
        ('unknown' stays conservative); a trailing (ratcheted, usually
        profitable) exit no longer suppresses a winning name for 24h.
        """
        if STOP_CLASSIFY_V2 and kind == 'trail':
            logger.info("[LOCKOUT] %s: trailing server-stop exit — no "
                        "lockout (STOP_CLASSIFY_V2)", symbol)
            return
        self.hard_stop_lockout[symbol] = datetime.datetime.now()
        self._save_hard_stop_lockout()

    def _stream_stop_fallback(self, order_id):
        """TRADER_STREAM_STOP_DETECT failure-path assist (c26 T6): when
        the REST stop-status probe raises, a cached trade_updates 'filled'
        event (terminal, monotonic — cannot regress) recovers the fill
        instead of waiting out the REST outage. Only 'filled' is accepted;
        every other/absent state returns None = today's retry-next-cycle.
        Self-gated on the module flag (callers invoke unconditionally in
        their except branches). Never raises.
        """
        if not STREAM_STOP_DETECT:
            return None
        try:
            from order_stream import get_order_state
            st = get_order_state(order_id)
        except Exception:
            return None
        if st and st.get('status') == 'filled':
            logger.warning("[STOP-CHECK] %s: REST failed but stream cache "
                           "shows FILLED — using streamed fill", order_id)
            return SimpleNamespace(status='filled',
                                   filled_qty=st.get('filled_qty'),
                                   filled_avg_price=st.get('filled_avg_price'),
                                   stop_price=None)
        return None

    def _record_account_risk(self):
        """GATE-1 measurement (wave-8 #7): journal the COMBINED cross-book
        stop-risk.

        The per-book ENB cap runs independently in each loop process, so the
        stock book (crypto-proxies) and crypto book (spot) can stack toward the
        same factor. This writes THIS book's diversified stop-risk to a shared
        registry, reads the other book's, and journals the combined account
        risk vs the cap. MEASUREMENT ONLY — it takes no trading action; the live
        clamp is gated to the Jetson once the journals show the books do stack.
        """
        try:
            from risk_budget import record_book_risk_and_report
            from portfolio import avg_book_correlation
            from strategy_config import CROSS_BOOK_RHO
            names = list(self.positions.keys())
            # uncovered=0.5: keep the journaled rho consistent with the
            # live ENB budget's no-data prior (IA-4 covered-none fix).
            rho_b = (avg_book_correlation(names, self.corr_matrix,
                                          uncovered=0.5)
                     if (self.corr_matrix and names) else 0.5)
            rep = record_book_risk_and_report(
                self.get_asset_type(), self._book_stop_risks(), rho_b,
                rho_cross=CROSS_BOOK_RHO)
            if rep is not None:
                log_decision({'action': 'account_risk',
                              'book': self.get_asset_type(), **rep})
        except Exception as e:
            logger.debug("[ACCT-RISK] gate-1 measurement skipped: %s", e)

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
        # Exits (signal sells, LLM vetoes, server fills, flattens, breaker)
        # drop positions without touching breach-confirmation state — prune
        # stale entries so a re-entry can't have its 2-consecutive-reading
        # confirmation short-circuited to ONE reading by a dead position's
        # armed breach (2026-07 panel; mirrors crypto_loop._manage_stops'
        # _resting_stop_px prune, which cites the same base_loop exit paths).
        for s in list(self._pending_breach):
            if s not in self.positions:
                self._pending_breach.pop(s, None)
        # IA-4 state prune (mirrors the breach prune above): entry-ts /
        # vertical-journal / armed-signal-exit keys for departed positions.
        for _d in (getattr(self, '_position_entry_ts', None),
                   getattr(self, '_pending_signal_exit', None)):
            if _d:
                for s in [s for s in _d if s not in self.positions]:
                    _d.pop(s, None)
        _vj = getattr(self, '_vertical_journaled', None)
        if _vj:
            for k in [k for k in _vj if k[0] not in self.positions]:
                _vj.discard(k)
        for symbol in list(self.positions):
            pos = self.positions[symbol]

            # Resting protective order (crypto GTC stop_limit) — detect
            # server-side fills the loop would otherwise miss
            if pos.stop_order_id and self.get_asset_type() == 'crypto':
                so = None
                via_stream = False
                try:
                    so = self.api.get_order(pos.stop_order_id)
                except Exception as e:
                    # Stream assist (self-gated on STREAM_STOP_DETECT
                    # inside the method; flag OFF returns None).
                    so = self._stream_stop_fallback(pos.stop_order_id)
                    via_stream = so is not None
                    if so is None:
                        # Keep the id and retry next cycle (stock_loop logs
                        # the same case); a transient get_order failure must
                        # not blind server-fill detection silently.
                        logger.debug("[STOP-CHECK] %s: resting-stop status check "
                                     "failed (%s) — retry next cycle", symbol, e)
                if so is not None:
                    status = getattr(so, 'status', None)
                    if status == 'filled':
                        logger.info("[STOP-FILL] %s: resting stop filled at $%s",
                                    symbol, so.filled_avg_price)
                        llm_info = self.llm_scores.get(symbol, {})
                        # c26 T6: the resting stop may sit at the TRAILING
                        # level (ratcheted by _maybe_update_resting_stop) —
                        # classify hard-vs-trail (always journaled) and let
                        # _apply_server_stop_lockout decide; flag OFF the
                        # lockout stays unconditional as before.
                        kind, stop_px = self._classify_server_stop(symbol, pos, so)
                        extra = {'server_stop_kind': kind, 'stop_px': stop_px}
                        if via_stream:
                            extra['detect_source'] = 'stream'
                        self._record_confirmed_exit(symbol, pos, so, None,
                                                    exit_reason='server_stop',
                                                    llm_score=llm_info.get('s'),
                                                    reasoning=llm_info.get('r', ''),
                                                    extra=extra)
                        del self.positions[symbol]
                        self.last_trade_time[symbol] = datetime.datetime.now()
                        self._apply_server_stop_lockout(symbol, kind)
                        continue
                    if status in ('canceled', 'expired', 'rejected'):
                        pos.stop_order_id = None

            quote = self.get_quote(symbol)
            if quote is None:
                continue
            current_price = quote['midpoint']
            # Mark stash (IA-4): the BREAKER_PER_BOOK unrealized-P&L
            # source. Measurement infrastructure — always on, never read
            # by the legacy breaker path.
            try:
                if getattr(self, '_last_marks', None) is None:
                    self._last_marks = {}
                self._last_marks[symbol] = current_price
            except Exception:
                pass
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
                    self._execute_stop_exit(symbol, pos, stop_reason,
                                            current_price, quote=quote)
                else:
                    self._pending_breach[symbol] = stop_reason
                    logger.info("[STOP] %s: %s breach at $%.4f — awaiting "
                                "confirmation next cycle", symbol, stop_reason,
                                current_price)
            else:
                self._pending_breach.pop(symbol, None)
                # Crypto fb-anchored vertical barrier (IA-4, ledger §3.7):
                # only reachable with NO stop/TP/trail firing — i.e. the
                # position drifting between the barriers past the horizon
                # its labels were trained on. Would-fire journaled ALWAYS;
                # the exit itself is gated on CRYPTO_VERTICAL_BARRIER.
                if self.get_asset_type() == 'crypto':
                    self._check_vertical_barrier(symbol, pos, current_price,
                                                 quote)

        # Persist HWM / cooldown state each cycle (tiny atomic JSON write)
        self._save_position_state()

    def _check_vertical_barrier(self, symbol, pos, current_price, quote):
        """fb-anchored crypto max-hold check (IA-4, 2026-08 influence audit
        §3.7: "Labels are triple-barrier with an fb vertical; live crypto
        holds are unlimited ... the meta model learns a horizon the
        deployed book doesn't enforce").

        LOOP-layer age check only — the policy_exits kernel is UNTOUCHED
        (its offline vertical stays the label/backtest semantics; this is
        the live mirror). Horizon = the deployed model config's
        forward_bars, read the way the labels read it (hourly bars, 24/7
        book => fb bars == fb hours). Journals ONE would-fire row per
        position lifetime ALWAYS (fired=false while the flag is OFF — the
        flip evidence); under CRYPTO_VERTICAL_BARRIER the position exits
        through the existing stop-exit path with exit_reason='vertical'
        (retried next cycle on a failed sell). Never raises.
        """
        try:
            entry_ts = getattr(self, '_position_entry_ts', {}).get(symbol)
            if entry_ts is None:
                return      # unknown age (pre-upgrade state) — fail open
            fb = int((self.config or {}).get('forward_bars', 24))
            if fb <= 0:
                return
            age_h = (time.time() - float(entry_ts)) / 3600.0
            if age_h < fb:
                return
            from strategy_config import CRYPTO_VERTICAL_BARRIER
            seen = getattr(self, '_vertical_journaled', None)
            if seen is None:
                seen = self._vertical_journaled = set()
            key = (symbol, int(entry_ts))
            if key not in seen:
                seen.add(key)
                pnl_pct = ((current_price - pos.entry_price)
                           / pos.entry_price * 100
                           if pos.entry_price > 0 else 0.0)
                try:
                    log_decision({'action': 'vertical_barrier',
                                  'symbol': symbol,
                                  'asset_type': self.get_asset_type(),
                                  'age_hours': round(age_h, 1),
                                  'forward_bars': fb,
                                  'pnl_pct': round(pnl_pct, 4),
                                  'fired': bool(CRYPTO_VERTICAL_BARRIER)})
                except Exception:
                    pass
            if CRYPTO_VERTICAL_BARRIER:
                logger.info("[VERTICAL] %s: held %.0fh >= forward_bars=%d "
                            "with price between barriers — max-hold exit",
                            symbol, age_h, fb)
                self._execute_stop_exit(symbol, pos, 'vertical',
                                        current_price, quote=quote)
        except Exception as e:
            logger.debug("[VERTICAL] check failed for %s: %s", symbol, e)

    def _execute_stop_exit(self, symbol, pos, stop_reason, current_price,
                           quote=None):
        """Sell a position for a stop/TP/trailing exit and confirm the fill.

        Any resting order for this symbol (bracket stop/TP leg, trailing
        stop) holds the shares — selling around it rejects with
        'insufficient qty'. Cancel symbol-scoped orders first, confirm the
        sell filled, and record the trade at the REAL fill price.

        quote (c26 T6, measurement): the confirmation-cycle quote dict —
        only its fetched_ts is read, for the quote_age_s journal key.
        """
        from order_utils import cancel_orders_for_symbol, make_client_order_id, verify_position

        def _quote_age_s():
            if quote is None:
                return None
            fts = quote.get('_fetched_ts') or quote.get('fetched_ts')
            return round(time.time() - fts, 2) if fts else None

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
                    pnl_pct = (((current_price - entry_price) / entry_price) * 100
                               if entry_price > 0 else 0.0)
                    llm_info = self.llm_scores.get(symbol, {})
                    record_trade(symbol, 'sell', entry_price, current_price,
                                 pnl_pct, llm_score=llm_info.get('s'),
                                 reasoning='position desync — broker qty=0',
                                 exit_reason='desync', estimated=True)
                    # Decision-journal row (2026-07 panel): desync exits
                    # previously reached trade_memory only, invisible to
                    # decision_report/journal_stats. log_decision never
                    # raises (own try/except).
                    log_decision({"symbol": symbol, "action": "sell",
                                  "exit_reason": "desync",
                                  "pnl_pct": round(pnl_pct, 4),
                                  "decision_price": current_price,
                                  "fill_price": current_price,
                                  "slippage_bps": None,
                                  "quote_age_s": _quote_age_s(),
                                  "estimated": True})
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
            # alpaca adapters have returned None in filled_avg_price before
            # (order_utils.reconstruct_positions ledger P1) — a crash here
            # fires AFTER the sell filled and leaves a sold position still
            # tracked; fall back to the estimated path instead.
            try:
                fill_price = float(result.filled_avg_price)
                estimated = False
            except (TypeError, ValueError):
                logger.warning("[STOP] %s: filled but no usable fill price — "
                               "recording estimate", symbol)
                fill_price = current_price
                estimated = True
        else:
            logger.warning("[STOP] %s: exit fill unconfirmed (status=%s) — recording estimate",
                           symbol, status)
            fill_price = current_price
            estimated = True
            if status not in ('filled', 'partially_filled') and verify_position(self.api, symbol) is not None:
                # Sell didn't go through and we still hold it — keep tracking
                return

        pnl_pct = (((fill_price - entry_price) / entry_price) * 100
                   if entry_price > 0 else 0.0)
        self._breaker_note_realized((fill_price - entry_price) * pos.qty)
        llm_info = self.llm_scores.get(symbol, {})
        record_trade(symbol, 'sell', entry_price, fill_price,
                     pnl_pct, llm_score=llm_info.get('s'),
                     reasoning=llm_info.get('r', ''),
                     exit_reason=stop_reason, estimated=estimated)
        # Decision-journal row (2026-07 panel): hard_stop/take_profit/
        # trailing were the ONLY exits never journaled (every other path
        # goes through _record_confirmed_exit, which writes both). Same
        # row shape and sell-side slippage sign convention as
        # _record_confirmed_exit.
        decision_price = current_price
        slippage_bps = None
        if decision_price and decision_price > 0 and not estimated:
            slippage_bps = round((decision_price - fill_price) / decision_price * 1e4, 2)
        stop_row = {"symbol": symbol, "action": "sell",
                    "exit_reason": stop_reason,
                    "pnl_pct": round(pnl_pct, 4),
                    "decision_price": decision_price,
                    "fill_price": fill_price,
                    "slippage_bps": slippage_bps,
                    "quote_age_s": _quote_age_s(),
                    "estimated": estimated}
        # ENGINE r4 R4-a: broker ids, additive and only when known.
        for k, v in _order_journal_ids(result if result is not None
                                       else order, keep_none=False).items():
            stop_row.setdefault(k, v)
        log_decision(stop_row)
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
        # Prune the conviction stash for symbols that left the universe —
        # measurement-only dict with NO control-flow reader (verified:
        # _conv_fields + the subclass prediction-cache writers only), so
        # this cannot change a trade. Without it a rotated-out symbol
        # keeps a stale meta_p forever (2026-07 owner seed e).
        # _veto_strikes is deliberately NOT pruned here: strikes gate
        # liquidation (control flow) — owner decision.
        uni = set(symbols)
        for s in [s for s in self._last_meta_p if s not in uni]:
            self._last_meta_p.pop(s, None)
            self._last_meta_p_cycle.pop(s, None)

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

        def _harvest(future, symbol):
            try:
                sym, pred, snapshot = future.result()
                if pred is not None:
                    preds[sym] = pred
                if snapshot is not None:
                    snapshots[sym] = snapshot
            except Exception as e:
                logger.error("%s: Prediction error: %s", symbol, e)

        pending = set(futures)
        try:
            for future in as_completed(futures,
                                       timeout=self.PREDICTION_TIMEOUT_SEC):
                pending.discard(future)
                _harvest(future, futures[future])
        except (FuturesTimeoutError, TimeoutError):
            # One half-open socket must not wedge stops/sells/breaker/
            # flatten for the rest of the process lifetime (c26 D01).
            # Completed-but-unyielded futures are still harvested; wedged
            # symbols simply have no prediction this cycle — fail-closed
            # for entries (no_pred), exits/stops unaffected.
            for f in list(pending):
                if f.done():
                    pending.discard(f)
                    _harvest(f, futures[f])
                else:
                    f.cancel()
            wedged = sorted(futures[f] for f in pending)
            logger.error("[PRED] fan-out timed out after %ds — no "
                         "prediction for %d symbol(s): %s — rebuilding "
                         "worker pool", self.PREDICTION_TIMEOUT_SEC,
                         len(wedged), ', '.join(wedged))
            self._rebuild_prediction_pool()
            try:
                log_decision({'action': 'pred_fanout_timeout',
                              'asset_type': self.get_asset_type(),
                              'timeout_s': self.PREDICTION_TIMEOUT_SEC,
                              'wedged': wedged})
            except Exception:
                pass

        # Rolling prediction log for the PSI drift monitor (one line per
        # cycle; monitor_drift.py prunes it to 7 days)
        try:
            from monitor_drift import log_predictions
            log_predictions(self.MODEL_PREFIX, preds)
        except Exception:
            pass

        # Stash for the LLM candidate builder's rich-context evidence block
        # (stock_loop already does this; setting it in the base too gives
        # the crypto loop access — harmless when rich_context_enabled=False).
        self._last_snapshots = snapshots

        return preds, snapshots

    def _rebuild_prediction_pool(self):
        """Drop a wedged executor and start fresh (c26 D01): with five
        workers blocked on dead sockets the pool is bricked permanently.
        Old threads unwind when their REST timeouts fire.

        Rate-limited to one rebuild per 10 min: shutdown() cannot stop
        threads already blocked on dead sockets, so every rebuild during a
        persistent outage would strand another MAX_PREDICTION_WORKERS
        threads (each with a per-thread SQLite connection) on the 8 GB
        Jetson. While rate-limited, predictions fail closed (no entries);
        exits/stops are unaffected."""
        import time as _t
        now = _t.monotonic()
        last = getattr(self, '_pool_rebuild_ts', None)
        if last is not None and (now - last) < 600.0:
            try:
                logger.warning("[PRED] pool rebuild rate-limited (%.0fs since "
                               "last) — keeping current executor", now - last)
            except Exception:
                pass
            return
        self._pool_rebuild_ts = now
        try:
            self._prediction_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self._prediction_pool = ThreadPoolExecutor(
            max_workers=self.MAX_PREDICTION_WORKERS,
            thread_name_prefix=f'{self.get_asset_type()}-pred')

    def _expire_llm_scores(self):
        """Fail-open expiry (c26 D14): scores older than LLM_SCORE_TTL_SEC
        must not keep vetoing entries (or holding veto strikes) through an
        outage. Only expires timestamped score sets — unit-test stubs that
        set llm_scores directly are untouched."""
        ts = getattr(self, '_llm_scores_ts', None)
        if (self.llm_scores and ts is not None
                and time.time() - ts > self.LLM_SCORE_TTL_SEC):
            logger.warning("[LLM] scores %.1fh stale — expiring (fail-open:"
                           " no LLM veto until a fresh analysis lands)",
                           (time.time() - ts) / 3600)
            self.llm_scores = {}
            self._veto_strikes.clear()

    def _run_llm_analysis(self, preds: dict):
        """Run LLM pre-trade analysis if interval elapsed."""
        self._expire_llm_scores()
        now_ts = time.time()
        if now_ts < getattr(self, '_llm_backoff_until', 0.0):
            return          # outage backoff (c26 D14) — forced-refresh cannot bypass
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
        self._last_llm_time = now_ts  # stamp on ATTEMPT, not success (c26 D14): an outage must not collapse the 600s cadence into a 30s retry storm

        fng_value = None
        try:
            from sentiment import get_fear_greed
            fd = get_fear_greed()
            fng_value = fd.get('value') if isinstance(fd, dict) else fd
        except Exception:
            pass

        _call_t0 = time.time()
        try:
            new_scores = analyze_trades(
                candidates, self.get_asset_type(), equity=self._equity,
                positions=list(self.positions.keys()),
                position_details={s: p.to_dict() for s, p in self.positions.items()},
                fng_value=fng_value,
                model_config=self.config,
            )
        except Exception as e:
            # SCOUT_E A1, ENGINE half (measurement-only): an exception that
            # escapes analyze_trades (e.g. _parse_response's OverflowError on
            # a huge-integer "s") used to leave no journal row. Journal the
            # attempt, then RE-RAISE unchanged — run()'s handling of the
            # cycle is untouched (fail-open here is an owner decision).
            try:
                log_decision({'action': 'llm_error',
                              'asset_type': self.get_asset_type(),
                              'outcome': 'exception',
                              'error_type': type(e).__name__,
                              'n_symbols_sent': len(candidates),
                              'latency_ms': int(round(
                                  (time.time() - _call_t0) * 1000))})
            except Exception:
                pass
            raise
        self._last_llm_symbols = {c.get('symbol') for c in candidates}
        if new_scores:
            self.llm_scores = new_scores
            self._llm_scores_ts = now_ts
            self._llm_fail_count = 0
            self._llm_backoff_until = 0.0
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
            # A symbol SENT this analysis but omitted from the (partial)
            # response reads as s=0.5 (no veto) — its strike must reset too,
            # or veto -> omitted -> veto liquidates on two NON-consecutive
            # vetoes (D13 scope A). Symbols never sent (stock top-N misses)
            # keep their strikes: see _get_predictions (owner decision).
            for sym in self._last_llm_symbols.difference(new_scores):
                self._veto_strikes.pop(sym, None)
            logger.info("[LLM] Scores: %s",
                        ", ".join(f"{s}={v.get('s', 0.5):.2f}" for s, v in self.llm_scores.items()))
            # Journal every scored candidate (not just traded ones) so
            # llm_eval.py can measure whether the gate predicts returns —
            # the system previously had no way to know if the LLM helped
            preds_by_symbol = {c['symbol']: c.get('pred_return') for c in candidates}
            row = {
                "action": "llm_analysis",
                "asset_type": self.get_asset_type(),
                "forward_bars": self.config.get('forward_bars', 24) if self.config else 24,
                # s is null only if a result dict lacks 's' (c26 D33). The
                # live parser never produces that: _parse_response writes
                # 0.5 for a missing/non-numeric/non-finite s, and a symbol
                # the provider omitted is simply absent here. s_defaulted
                # (SCOUT_E A3, measurement-only) flags that 0.5 fallback:
                # True/False from get_last_analysis_meta()['parse_flags'],
                # None when unknown (dedup hit, no flags).
                "scores": {sym: {"s": v.get('s'),
                                 "pred": preds_by_symbol.get(sym),
                                 "s_defaulted": None}
                           for sym, v in new_scores.items()},
            }
            try:
                from llm_analyst import get_last_analysis_meta
                meta = get_last_analysis_meta() or {}
                for k in ('model', 'prompt_sha256', 'dedup_hit',
                          'latency_ms', 'cost_usd'):
                    if meta.get(k) is not None:
                        row[k] = meta[k]
                pf = meta.get('parse_flags')
                if isinstance(pf, dict):
                    for sym, sc in row['scores'].items():
                        f = pf.get(sym)
                        if isinstance(f, dict) and isinstance(
                                f.get('s_defaulted'), bool):
                            sc['s_defaulted'] = f['s_defaulted']
            except Exception:
                pass    # metadata is best-effort — never blocks the journal
            log_decision(row)
        else:
            # Provider failure / empty parse: llm_scores are deliberately
            # untouched (fail-open — no gate value changes while the
            # provider is down). The throttle stamp now advances on ATTEMPT
            # and consecutive failures back off exponentially (c26 D14).
            logger.warning("[LLM] analyze_trades returned no scores "
                           "(provider failure/empty parse) — keeping "
                           "previous scores")
            self._llm_fail_count = getattr(self, '_llm_fail_count', 0) + 1
            backoff = min(3600.0, self.LLM_INTERVAL_SEC
                          * (2 ** (self._llm_fail_count - 1)))
            self._llm_backoff_until = now_ts + backoff
            logger.warning("[LLM] consecutive failures=%d — next attempt in"
                           " %.0fs", self._llm_fail_count, backoff)
            try:
                # outcome / n_symbols_sent / latency_ms: SCOUT_E A1, ENGINE
                # half (measurement-only, appended). The failure CLASS
                # (transport/parse/refusal) is llm_analyst's llm_calls row.
                log_decision({'action': 'llm_backoff',
                              'asset_type': self.get_asset_type(),
                              'consecutive_failures': self._llm_fail_count,
                              'backoff_s': round(backoff, 1),
                              'outcome': 'no_scores',
                              'n_symbols_sent': len(candidates),
                              'latency_ms': int(round(
                                  (time.time() - _call_t0) * 1000))})
            except Exception:
                pass

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
        rich = rich_context_enabled()
        for symbol in self.get_symbol_universe():
            fund = get_fundamentals(symbol, self.get_asset_type())
            fund_text = format_fundamentals_for_llm(symbol, fund)
            headlines = self.get_fresh_headlines(symbol)
            candidate = {
                'symbol': symbol,
                'pred_return': preds.get(symbol),
                'fundamentals_text': fund_text,
                'news_headlines': headlines,
            }
            if rich:
                # Fail-open: evidence is an optional enrichment — an error
                # here must never abort the cycle (sells/buys run after us).
                try:
                    prof = build_compact_evidence(
                        symbol,
                        getattr(self, '_last_snapshots', {}).get(symbol),
                        fund,
                        position=(self.positions[symbol].to_dict()
                                  if symbol in self.positions else None),
                        asset_type=self.get_asset_type())
                    if prof:
                        candidate['profile'] = prof
                except Exception:
                    pass
            candidates.append(candidate)
        return candidates

    def _record_confirmed_exit(self, symbol, pos, order, quote, exit_reason,
                               llm_score=None, reasoning='', extra=None):
        """Journal an exit using the order's real fill price when available.

        extra (c26 T6): optional dict of ADDITIVE journal keys merged into
        the sell row (server_stop_kind / stop_px / detect_source). Default
        None keeps the legacy key-set (plus quote_age_s, measurement).
        """
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
        self._breaker_note_realized((fill_price - pos.entry_price) * pos.qty)
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
        # quote_age_s (c26 T6, measurement): decision-quote age at journal
        # time — separates execution slippage from decision->fill staleness
        # on sell rows (buy rows already carry it).
        quote_age_s = None
        if quote is not None:
            fts = quote.get('_fetched_ts') or quote.get('fetched_ts')
            if fts:
                quote_age_s = round(time.time() - fts, 2)
        row = {"symbol": symbol, "action": "sell",
               "exit_reason": exit_reason,
               "pnl_pct": round(pnl_pct, 4),
               "decision_price": decision_price,
               "fill_price": fill_price,
               "slippage_bps": slippage_bps,
               "quote_age_s": quote_age_s,
               "estimated": estimated}
        if extra:
            row.update(extra)
        # ENGINE r4 R4-a: broker ids, additive and only when known.
        for k, v in _order_journal_ids(order, keep_none=False).items():
            row.setdefault(k, v)
        log_decision(row)

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
            pending = getattr(self, '_pending_signal_exit', None)
            if pending is None:
                pending = self._pending_signal_exit = {}
            if pred > -self.trade_threshold:
                if pending.pop(symbol, None) is not None:
                    # Armed first reading lapsed (pred recovered) — the
                    # jitter evidence the 1-vs-2-reading decision needs.
                    try:
                        log_decision({'action': 'signal_exit_reading',
                                      'symbol': symbol, 'event': 'lapsed',
                                      'pred_return': round(float(pred), 4)})
                    except Exception:
                        pass
                continue

            # SIGNAL_EXIT_CONFIRM_READS (IA-4, ledger §3.7): default 1 =
            # today's single-reading exit, byte-identical. At 2 the signal
            # exit matches stop-confirmation discipline: the first reading
            # ARMS (journaled, no sale); a subsequent reading confirms.
            # Pending survives a missing-pred cycle (fail toward exiting,
            # mirroring _pending_breach's quote-less behavior).
            try:
                from strategy_config import SIGNAL_EXIT_CONFIRM_READS
                confirm_reads = int(SIGNAL_EXIT_CONFIRM_READS)
            except Exception:
                confirm_reads = 1
            if confirm_reads >= 2 and symbol not in pending:
                pending[symbol] = self.cycle
                logger.info("%s: signal exit reading 1/2 (pred=%+.4f%%) — "
                            "awaiting confirmation next cycle", symbol, pred)
                try:
                    log_decision({'action': 'signal_exit_reading',
                                  'symbol': symbol, 'event': 'armed',
                                  'pred_return': round(float(pred), 4),
                                  'confirm_reads': confirm_reads})
                except Exception:
                    pass
                continue
            pending.pop(symbol, None)

            # Exits are NEVER cooldown-gated (2026-08 influence audit §3.4:
            # "an entry throttle delaying risk reduction inverts the tool's
            # purpose"). Record when this sell would previously have been
            # blocked — measurement only.
            cooldown_bypassed = not cooldown_ok(
                self.last_trade_time, symbol, self.COOLDOWN_MINUTES)

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
                extra = {}
                if cooldown_bypassed:
                    extra['cooldown_bypassed_exit'] = True
                if confirm_reads >= 2:
                    extra['signal_exit_readings'] = confirm_reads
                self._record_confirmed_exit(
                    symbol, pos, order, quote,
                    exit_reason='signal_sell',
                    llm_score=llm_info.get('s'),
                    reasoning=llm_info.get('r', ''),
                    extra=(extra or None))
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

            # Exits are NEVER cooldown-gated (2026-08 influence audit §3.4)
            # — holding vetoed risk to honor an entry throttle inverted the
            # tool's purpose. Marker is measurement only.
            cooldown_bypassed = not cooldown_ok(
                self.last_trade_time, symbol, self.COOLDOWN_MINUTES)

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
                self._record_confirmed_exit(
                    symbol, pos, order, quote,
                    exit_reason='llm_veto',
                    llm_score=llm_s,
                    reasoning=llm_info.get('r', ''),
                    extra=({'cooldown_bypassed_exit': True}
                           if cooldown_bypassed else None))
                del self.positions[symbol]
                self.last_trade_time[symbol] = datetime.datetime.now()
            else:
                # stock place_sell_order returns None on a missing/stale
                # quote — previously this announced 'LLM VETO SELL' and then
                # silently did nothing, every cycle. Log-only: do NOT add a
                # quote-None guard above (crypto deliberately falls back to
                # a market sell when the quote is missing).
                logger.warning("%s: LLM veto sell did not execute (quote=%s) "
                               "— retrying next cycle", symbol,
                               'unavailable' if quote is None else 'ok')
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
                getattr(self, '_last_meta_p_cycle', {}).pop(symbol, None)
                return True, 1.0
            self._last_meta_p[symbol] = float(p)
            try:
                self._last_meta_p_cycle[symbol] = self.cycle
            except AttributeError:
                pass    # unit-test stubs bypass __init__
            if p < META_VETO_PROB:
                # Routed through _journal_skip (2026-07 panel) so the row
                # carries the full conviction context and honors the
                # CONVICTION_JOURNAL_ENABLED switch. meta_prob kept as an
                # extra key — decision_report falls back to it (legacy name).
                self._journal_skip(symbol, 'meta_veto', rank=rank,
                                   pred=pred_return,
                                   snapshot=snapshots.get(symbol, {}) or {},
                                   meta_prob=round(p, 4))
                return False, 1.0
            return True, meta_size_mult(p)
        except Exception as e:
            # Fail-open PRESERVED (an error here must never block a trade),
            # but no longer silent: a feature-schema mismatch after a
            # retrain used to disable the meta veto for every trade in both
            # books with zero log output. Pop the stash (and its freshness
            # stamp) so a dead gate can't keep re-journaling its last good
            # probability.
            self._last_meta_p.pop(symbol, None)
            getattr(self, '_last_meta_p_cycle', {}).pop(symbol, None)
            if getattr(self, 'cycle', 1) % 10 == 1:
                logger.warning("[META] gate unavailable for %s (%s) — "
                               "failing open, meta veto INACTIVE", symbol, e)
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
        """Conviction context shared by skip and buy journal rows.

        meta_p/conviction_tier are emitted only when the stashed meta
        probability was computed THIS cycle (_meta_gate stamps
        _last_meta_p_cycle) — gates earlier in the funnel were journaling
        a value from an arbitrarily older cycle as if current, poisoning
        the Stage-0 substrate. Returns {} when the conviction journal is
        disabled, and never raises: measurement code must never abort a
        buy (the documented invariant this helper previously violated on
        the buy path).
        """
        if not self._conviction_journal_on():
            return {}
        f = {}
        try:
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
            # Blend legs (c26 T4 handoff): lets Stage-0 answer the
            # static-vs-fitted blend question from live journals.
            if snap.get('LSTM_Pred') is not None:
                f['lstm_pred'] = round(float(snap['LSTM_Pred']), 4)
            if snap.get('LGB_Pred') is not None:
                f['lgb_pred'] = round(float(snap['LGB_Pred']), 4)
            mp = self._last_meta_p.get(symbol)
            if mp is not None:
                stamp = getattr(self, '_last_meta_p_cycle', {}).get(symbol)
                if stamp is None or stamp == getattr(self, 'cycle', stamp):
                    f['meta_p'] = round(float(mp), 4)
                    f['conviction_tier'] = self._conviction_tier(pred, mp)
        except Exception as e:
            logger.debug("[CONV] context build failed for %s: %s", symbol, e)
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
        try:
            if pred is not None:
                rec["pred_return"] = round(float(pred), 4)
            rec.update(self._conv_fields(symbol, pred, snapshot, rank=rank))
            rec.update(extra)
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

    def _journal_standdown_skips(self, preds: dict, snapshots: dict):
        """Journal FOMC/CPI stood-down entry skips with window identity.

        2026-08 influence audit §3.2 (J2 dissent) / open question #4: the
        stand-down "fires before journaling — structurally unpriceable".
        Called by _execute_buys when _entries_allowed() said False; only
        acts when the recorded cause is the macro stand-down (manual halt
        stays unjournaled — no counterfactual worth pricing there).

        Volume discipline (Jetson journals): one row per (window, symbol),
        and only candidates at/above the entry threshold — the priceable
        counterfactual — not the whole universe every cycle. Rows go
        through _journal_skip so they carry the same pred/rank/conviction
        context every other priced skip class carries. Measurement-only.
        """
        block = getattr(self, '_entries_block_info', None)
        if not block or block[0] != 'macro_standdown':
            return
        if not self._conviction_journal_on():
            return
        _, reason, wid = block
        if wid is None:
            wid = 'unknown-window'
        seen = getattr(self, '_standdown_journaled', None)
        if seen is None:
            seen = self._standdown_journaled = set()
        try:
            ranked = sorted((s for s in preds if preds[s] is not None),
                            key=lambda s: preds[s], reverse=True)
            for rank, symbol in enumerate(ranked, 1):
                pred = preds[symbol]
                if pred < self.trade_threshold:
                    continue
                key = (wid, symbol)
                if key in seen:
                    continue
                seen.add(key)
                self._journal_skip(symbol, 'macro_standdown', rank=rank,
                                   pred=pred,
                                   snapshot=(snapshots or {}).get(symbol),
                                   standdown_window=wid,
                                   standdown_reason=reason)
        except Exception:
            pass

    def _log_vix25_gap_once(self, safe_haven_symbols):
        """One log per day stating whether the VIX>25 'safe-haven
        carve-out' is satisfiable at all: when no SAFE_HAVEN symbol is in
        the tradable universe the block is a de facto full book halt
        nobody designed (2026-08 influence audit gap C3). Log-only —
        behavior unchanged; the block's fate is a separate flag decision.
        """
        today = datetime.date.today().isoformat()
        if getattr(self, '_vix25_gap_log_date', None) == today:
            return
        self._vix25_gap_log_date = today
        try:
            tradable = sorted(set(self.get_symbol_universe())
                              & set(safe_haven_symbols))
            if tradable:
                logger.info("[VIX25] defensive block active — tradable "
                            "safe havens: %s", ', '.join(tradable))
            else:
                logger.warning(
                    "[VIX25] defensive block active and NO SAFE_HAVEN "
                    "symbol is in the tradable universe — de facto full "
                    "book halt (influence-audit gap C3)")
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

        strategy_config.DERISK_STACK_V2 (c26 S3 / 02_research B06) selects
        between two compositions: legacy (this product) and v2 (regime family
        {VIX tier / BTC-RV, stress, book-vol scalar} aggregated by MIN, product
        only across families; HMM + per-position vol_mult excluded; the
        pseudo-CAPE haircut was DELETED repo-wide 2026-08-22 — owner ruling,
        08_removed_code.md). Both are always computed and journaled — the legacy keys
        always describe the legacy product, detail['v2'] the v2 composition,
        and detail['stack'] names which one was actually applied to the size.
        """
        if (self.macro_regime is not None
                and self.macro_regime.sizing_mult == 0.0):
            # Emergency halt (stablecoin depeg sets sizing_mult *= 0.0):
            # returning here means the 0.1 tilt floor below can never
            # resurrect a 10%-size re-buy into the contagion (c26 D26).
            logger.warning("[SIZING] %s: macro emergency (sizing_mult=0.0)"
                           " — entry size forced to 0", symbol)
            return 0
        from strategy_config import (RISK_PCT_PER_TRADE, KELLY_CAP,
                                     TILT_MAX, MIN_ORDER_NOTIONAL,
                                     DERISK_STACK_V2)
        try:
            from strategy_config import CORR_FAMILY_MERGED
        except Exception:
            CORR_FAMILY_MERGED = False
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
        # KELLY_SAMPLE_GATE (IA-4, ledger §3.5 kelly_mult KEEP-COND): the
        # pre-D06 history is winner-censored; hold neutral 1.0 until the
        # book accumulates KELLY_SAMPLE_MIN_TRADES uncensored trades
        # recorded on/after KELLY_SAMPLE_SINCE. Gate state journaled in
        # the sizing detail while ON; OFF = byte-identical.
        kelly_gate_info = None
        try:
            from strategy_config import (KELLY_SAMPLE_GATE,
                                         KELLY_SAMPLE_MIN_TRADES,
                                         KELLY_SAMPLE_SINCE)
            if KELLY_SAMPLE_GATE:
                from trading_utils import uncensored_trade_count
                n_unc = uncensored_trade_count(
                    self.get_asset_type(), since_iso=KELLY_SAMPLE_SINCE)
                held = n_unc < KELLY_SAMPLE_MIN_TRADES
                if held:
                    kelly_mult = 1.0
                kelly_gate_info = {'n_uncensored': n_unc,
                                   'min_trades': KELLY_SAMPLE_MIN_TRADES,
                                   'held_neutral': held}
        except Exception:
            pass

        # --- 3. GARCH vol targeting (bounded inside the helper) ---
        vol_mult = 1.0
        returns = None
        try:
            from market_data import (fetch_bars_alpaca, fetch_stock_bars_alpaca,
                                     closed_bars_v2_enabled)
            _closed = closed_bars_v2_enabled()
            if self.get_asset_type() == 'crypto':
                df = fetch_bars_alpaca(self.api, symbol, closed_only=_closed)
            else:
                df = fetch_stock_bars_alpaca(self.api, symbol,
                                             closed_only=_closed)
            if df is not None and len(df) > 100:
                # pandas pin (c26-T3/B21): ffill + fill_method=None ==
                # pandas-2 pad semantics, pandas-3-proof
                returns = df['Close'].ffill().pct_change(fill_method=None).dropna().values * 100
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
        # Each factor is computed into a named local and recorded into the
        # sizing-decomposition journal (2026-07 indicator review): the
        # de-risk layers share drivers (VIX enters BOTH the ladder and
        # macro sizing_mult; vol enters ATR base, vol_mult, VIX, HMM and
        # the book scalar), and only per-fill decomposition rows can show
        # how often they co-fire and pin size at the 0.1 floor.
        # Measurement-only — the arithmetic is unchanged.
        tilt = 1.0
        detail = {'stop_dist': round(stop_dist, 5),
                  'base': round(base, 2),
                  'kelly_mult': round(kelly_mult, 4),
                  'vol_mult': round(vol_mult, 4)}
        if kelly_gate_info is not None:
            detail['kelly_gate'] = kelly_gate_info
        # Signal confidence (tamed: was 0.5-2.0)
        if pred_return is not None and self.trade_threshold > 0.001:
            f_conf = min(1.25, max(0.75, pred_return / self.trade_threshold))
            tilt *= f_conf
            detail['signal_conf'] = round(f_conf, 4)
        # VIX ladder (arXiv 2508.16598: shrink Kelly in high vol)
        vix = self.macro_regime.vix if self.macro_regime else None
        if vix is not None:
            f_vix = 0.3 if vix > 35 else 0.5 if vix > 25 else 0.7 if vix > 15 else 1.0
            tilt *= f_vix
            detail['vix_tilt'] = f_vix
            detail['vix'] = round(vix, 1)
        # Drawdown de-leveraging ladder (peak persisted across restarts, wave-8 #4)
        from drawdown import drawdown_fraction, drawdown_size_multiplier
        f_dd = drawdown_size_multiplier(
            drawdown_fraction(self._peak_equity, self._equity))
        tilt *= f_dd
        detail['dd_mult'] = round(f_dd, 4)
        # Macro regime
        if self.macro_regime:
            tilt *= self.macro_regime.sizing_mult
            detail['macro_mult'] = round(self.macro_regime.sizing_mult, 4)
        # Correlation with existing book. CORR_FAMILY_MERGED (IA-4, ledger
        # §3.3): ON bypasses the f_corr haircut — the ENB book-risk cap
        # below is the single correlation consumer (v2's composition reads
        # detail['corr_mult'] with a 1.0 default, so it merges too).
        if self.corr_matrix and self.positions:
            if not CORR_FAMILY_MERGED:
                f_corr = get_correlation_sizing_factor(
                    symbol, list(self.positions.keys()), self.corr_matrix)
                tilt *= f_corr
                detail['corr_mult'] = round(f_corr, 4)
            else:
                detail['corr_family_merged'] = True
        # HMM regime (advisory) + disagreement penalty
        hmm_label = 'unknown'
        if returns is not None and len(returns) > 200:
            try:
                regime = get_cached_regime(symbol, returns)
                tilt *= regime['sizing_mult']
                hmm_label = regime.get('label', 'unknown')
                detail['hmm_mult'] = round(regime['sizing_mult'], 4)
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
            detail['disagree_mult'] = 0.8
        # Regime-conditional Kelly cap: never let recent-win streaks scale
        # size ABOVE baseline in stressed/bear regimes (procyclicality fix)
        if (vix is not None and vix > 25) or hmm_label == 'bear':
            kelly_mult = min(kelly_mult, 1.0)
            detail['kelly_mult'] = round(kelly_mult, 4)
        # Sentiment gate, LLM conviction, meta-label probability
        # (vetoes handled by callers)
        tilt *= sentiment_mult
        tilt *= llm_mult
        tilt *= meta_mult
        detail['sentiment_mult'] = round(sentiment_mult, 4)
        detail['llm_mult'] = round(llm_mult, 4)
        detail['meta_mult'] = round(meta_mult, 4)
        # Asset-specific extra tilt (crypto: funding-rate positioning)
        f_extra = self._extra_tilt(symbol)
        tilt *= f_extra
        if f_extra != 1.0:
            detail['extra_tilt'] = round(f_extra, 4)
        # Book-level realized-vol scalar (EWMA of the account equity
        # curve; de-risk only — catches correlation buildup that
        # per-position GARCH targeting can't see)
        try:
            from portfolio import get_book_vol_scalar_cached
            f_bookvol = get_book_vol_scalar_cached(self.api, self.get_asset_type())
            tilt *= f_bookvol
            detail['book_vol_mult'] = round(f_bookvol, 4)
        except Exception:
            pass
        # Boosts capped; de-risking honored
        detail['tilt_raw'] = round(tilt, 4)
        tilt = max(0.1, min(TILT_MAX, tilt))
        detail['tilt'] = round(tilt, 4)

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
            detail['degraded_inputs'] = missing
            detail['tilt'] = round(tilt, 4)

        # --- DERISK_STACK_V2 composition (c26 S3 / 02_research B06) ---
        # ALWAYS computed; shadow-journaled while the flag is OFF so the
        # Jetson produces before/after evidence from day one
        # (scripts/sizing_cofire_report.py reads both). Regime family
        # aggregates by MIN (comonotone reads of one latent risk-off state);
        # product only ACROSS families. EXCLUDED from v2 by design:
        # HMM mult (kill-recommended; inverted
        # smoothing — see regime_detector.py), the duplicate inline VIX
        # ladder + macro VIX tiers (macro_indicators.vix_tier_mult_v2 owns
        # the ONE map), the 0.8 disagreement penalty (min already resolves
        # disagreement), and the per-position vol_mult (PORTFOLIO_VOL_TARGET
        # owner = book scalar, inside the family min). The degraded-mode
        # missing-count definition above is shared verbatim (still counts
        # `vix is None` even for crypto-v2 — failure-path safety).
        v2 = {}
        try:
            from macro_indicators import regime_family_mults_v2
            family = regime_family_mults_v2(self.macro_regime,
                                            self.get_asset_type())
            if self.get_asset_type() == 'crypto':
                from volatility import get_crypto_rv_mult
                rv_mult, rv_state, rv_pct = get_crypto_rv_mult()
                family['btc_rv'] = rv_mult
                v2['btc_rv_state'] = rv_state
                if rv_pct is not None:
                    v2['btc_rv_pctile'] = round(rv_pct, 1)
            family['bookvol'] = detail.get('book_vol_mult', 1.0)
            f_regime = min(family.values()) if family else 1.0
            v2['family'] = {k: round(v, 4) for k, v in family.items()}
            v2['f_regime_min'] = round(f_regime, 4)
            v2['min_src'] = (min(family, key=family.get) if family else None)
            tilt_v2 = (f_regime
                       * detail.get('signal_conf', 1.0)
                       * detail['dd_mult']
                       * detail.get('corr_mult', 1.0)
                       * sentiment_mult * llm_mult * meta_mult
                       * detail.get('extra_tilt', 1.0))
            v2['tilt_raw'] = round(tilt_v2, 4)
            tilt_v2 = max(0.1, min(TILT_MAX, tilt_v2))
            if missing >= 2:
                tilt_v2 = min(tilt_v2, 0.5)   # same degraded contract
                v2['degraded'] = missing
            v2['tilt'] = round(tilt_v2, 4)
            # (g) ONE vol-target scope: book scalar (in the min) owns
            # PORTFOLIO_VOL_TARGET; per-position ratio composes at 1.0.
            v2['sized_pre_caps'] = round(base * kelly_mult * tilt_v2, 2)
            if DERISK_STACK_V2:
                global _derisk_v2_logged
                if not _derisk_v2_logged:
                    _derisk_v2_logged = True
                    logger.warning(
                        "[DERISK-V2] ACTIVE: regime family aggregated by MIN; "
                        "HMM multiplier EXCLUDED (KILL_LIST); "
                        "PORTFOLIO_VOL_TARGET owned by book scalar; legacy "
                        "product journaled as shadow")
        except Exception as e:
            tilt_v2 = None
            logger.warning("[DERISK-V2] v2 composition failed (%s) — "
                           "legacy product retained", e)
        detail['v2'] = v2
        detail['stack'] = ('v2' if (DERISK_STACK_V2 and tilt_v2 is not None)
                           else 'legacy')

        # Fail-open: any v2 error falls back to the legacy product even when
        # the flag is ON — an internal error must never zero/black-hole an
        # entry. Fail-closed paths (emergency zero, MIN_ORDER_NOTIONAL,
        # book-risk budget=0) are untouched and apply identically after
        # selection.
        if DERISK_STACK_V2 and tilt_v2 is not None:
            sized = base * kelly_mult * 1.0 * tilt_v2
        else:
            sized = base * kelly_mult * vol_mult * tilt
        detail['sized_pre_caps'] = round(sized, 2)
        # Stash for the buy journal (measurement-only; see _execute_buys).
        # The dict keeps being mutated below (book-risk shrink, leverage),
        # and the journal write reads the same object at fill time, so the
        # logged row carries the final values.
        try:
            from strategy_config import CONVICTION_JOURNAL_ENABLED
            self._last_sizing_detail = (detail if CONVICTION_JOURNAL_ENABLED
                                        else None)
        except Exception:
            self._last_sizing_detail = None

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
                # (crypto pairwise corr typically 0.6-0.9) rather than 0.
                # uncovered=0.5 (IA-4, ledger §3.3 ENB input defect,
                # direct-ship): a NON-EMPTY matrix covering none of the
                # book's pairs previously returned rho=0.0 — bypassing
                # this same no-data prior and pricing a fully-unknown
                # book as perfectly diversifying.
                rho = (avg_book_correlation(book + [symbol], self.corr_matrix,
                                            uncovered=0.5)
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
                    detail['book_risk_scale'] = round(scaled / max(sized, 1e-9), 4)
                    sized = scaled
            except Exception:
                pass

        # --- 5. Hard caps including the NEW order ---
        existing_value = 0.0
        if symbol in self.positions:
            existing_value = self._cap_value(self.positions[symbol])
        room = self.MAX_NOTIONAL_PER_SYMBOL - existing_value
        sized = min(sized, max(room, 0))

        # Leveraged ETF scaling: divide by leverage factor
        leverage = self._leveraged_etfs.get(symbol, 1)
        if leverage > 1:
            sized /= leverage
            detail['leverage_div'] = leverage

        if sized < MIN_ORDER_NOTIONAL:
            return 0
        return int(sized)

    @staticmethod
    def _cap_value(pos) -> float:
        """Notional a held position counts against MAX_NOTIONAL_PER_SYMBOL.

        Cost basis (qty * entry_price) exactly as before whenever
        entry_price > 0. An unknown basis (entry_price <= 0 — the Alpaca
        paper avg_entry_price=0 quirk on inherited positions) valued the
        position at $0, so the cap admitted add-ons on a $21k position
        every cycle (ENGINE r1 W1); value it at the high-water mark (the
        tracked mark, >= the last quote) instead.
        """
        if pos.entry_price > 0:
            return pos.qty * pos.entry_price
        return pos.qty * pos.high_water_mark

    def _load_hard_stop_lockout(self):
        """Load hard-stop lockout state from disk (survive restarts).

        Per-book file; when this book's prefixed file does not exist yet,
        fall back ONCE to the legacy shared file (migration read — the
        next save writes the prefixed file and the legacy copy is never
        written again by this book).
        """
        path = self._lockout_file
        legacy = getattr(self, '_legacy_lockout_file', path)
        if path != legacy and not os.path.exists(path) and os.path.exists(legacy):
            logger.info("[LOCKOUT] no %s yet — migration read from legacy "
                        "shared %s", os.path.basename(path),
                        os.path.basename(legacy))
            path = legacy
        try:
            with open(path, 'r') as f:
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
        except FileNotFoundError:
            pass
        except json.JSONDecodeError as e:
            # A torn/corrupt file silently discarded ALL persisted lockouts
            # for both books — make it visible (2026-07 panel).
            logger.warning("[LOCKOUT] lockout file corrupt (%s) — starting "
                           "with no persisted lockouts", e)
        except Exception as e:
            logger.warning("[LOCKOUT] Failed to load lockout file: %s", e)

    def _save_hard_stop_lockout(self):
        """Persist hard-stop lockout state to disk (atomic write)."""
        try:
            data = {}
            for symbol, lockout_time in self.hard_stop_lockout.items():
                # epoch + seconds, not naive wall-clock + timedelta: the
                # latter is 23 h / 25 h of real time across DST (G2-3).
                expiry_ts = (lockout_time.timestamp()
                             + self.HARD_STOP_LOCKOUT_HOURS * 3600)
                data[symbol] = expiry_ts
            # Final paths are per-book now (2026-08 influence audit), but
            # the per-book TEMP suffix stays: combined-bots mode runs both
            # books in one process and a shared temp name could still tear
            # under interleaved open(tmp,'w')/os.replace (2026-07 panel).
            tmp = f"{self._lockout_file}.{self.MODEL_PREFIX or 'crypto'}.tmp"
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
        """True if the symbol hasn't used up today's entry budget.

        TRADE_BUDGET_BACKSTOP (IA-4, ledger §3.4 cooldown+budget MERGE):
        ON demotes the daily budget to a runaway backstop (cap x mult) so
        cooldown is the ONE churn instrument; OFF = today's cap.
        """
        from strategy_config import MAX_TRADES_PER_SYMBOL_PER_DAY
        self._roll_trade_budget_date()
        cap = MAX_TRADES_PER_SYMBOL_PER_DAY.get(self.get_asset_type(), 4)
        try:
            from strategy_config import (TRADE_BUDGET_BACKSTOP,
                                         TRADE_BUDGET_BACKSTOP_MULT)
            if TRADE_BUDGET_BACKSTOP:
                cap = cap * int(TRADE_BUDGET_BACKSTOP_MULT)
        except Exception:
            pass
        if self._daily_trades.get(symbol, 0) >= cap:
            return False
        return True

    def _is_hard_stop_locked(self, symbol: str) -> bool:
        """Check if symbol is in hard-stop lockout period."""
        if symbol not in self.hard_stop_lockout:
            return False
        # .timestamp() difference, not naive local subtraction: stamps are
        # naive LOCAL wall-clock (now()/fromtimestamp set .fold), so this is
        # true elapsed seconds across a DST transition (2026-09 G2-3).
        elapsed = (datetime.datetime.now().timestamp()
                   - self.hard_stop_lockout[symbol].timestamp())
        if elapsed >= self.HARD_STOP_LOCKOUT_HOURS * 3600:
            del self.hard_stop_lockout[symbol]
            self._save_hard_stop_lockout()
            return False
        return True

    def _check_flatten_request(self):
        """Honor a remote /flatten: liquidate this book, halt, clear flag.

        c26 D17: writers (Telegram /flatten, GUI) still set notify's legacy
        SHARED flag; whichever loop sees it first fans it out into per-book
        flag files (flatten_crypto.flag / flatten_stock.flag) and clears the
        legacy flag, then each loop consumes ONLY its own flag — so BOTH
        books flatten, and the stock book picks its flag up even when the
        request landed off-hours (this check now runs before the
        market-hours gate). A per-book flag older than
        FLATTEN_FLAG_STALE_SEC is discarded, not actioned.

        The per-book flag is consumed BEFORE flattening so a partial
        failure can't re-trigger liquidation every 30s; failures are
        notified instead.
        """
        my_flag = _flatten_flag_path(self.get_asset_type())
        try:
            from notify import (flatten_requested, clear_flatten_request,
                                set_halt, notify)
            if flatten_requested():
                # Fan the shared request out to BOTH books, then clear it —
                # previously whichever book cycled first consumed the flag
                # and the other book stayed fully invested.
                for book in ('crypto', 'stock'):
                    try:
                        with open(_flatten_flag_path(book), 'w') as fh:
                            fh.write(str(time.time()))
                    except Exception:
                        pass
                clear_flatten_request()
            if not os.path.exists(my_flag):
                return
            age = time.time() - os.path.getmtime(my_flag)
            # Consume BEFORE flattening (unchanged contract: a partial
            # failure must not re-trigger every 30s).
            os.remove(my_flag)
            if age > FLATTEN_FLAG_STALE_SEC:
                logger.warning("[FLATTEN] stale per-book flatten flag "
                               "(%.0f min old) — discarded, not actioned",
                               age / 60)
                return
        except Exception:
            return
        logger.warning("[FLATTEN] remote flatten requested — liquidating %s book",
                       self.get_asset_type())
        set_halt('remote flatten')
        try:
            pre_flatten = dict(self.positions)
            failures = emergency_flatten(self.api,
                                         symbols=self.get_symbol_universe())
            # Same failure-tracking contract as the circuit-breaker and
            # stablecoin flatten sites (2026-07 review P1; this third site
            # was missed): failures come back BROKER-format ('BTCUSD'),
            # positions are keyed universe-format ('BTC/USD'), and the
            # list-positions sentinel means broker state is UNKNOWN — keep
            # every position tracked rather than guess.
            if '<list_positions failed>' in failures:
                pass  # unknown broker state — keep all positions tracked
            elif failures:
                failed_norm = {f.replace('/', '') for f in failures}
                self.positions = {s: p for s, p in self.positions.items()
                                  if s.replace('/', '') in failed_norm}
            else:
                self.positions.clear()
            self._save_position_state()
            # Journal only the positions the flatten actually released
            # (estimated fills; excluded from Kelly). A remote flatten
            # previously produced zero trade records at all.
            # BREAKER_SERVER_FILL_ATTRIB ON (ENGINE r4 R4-b): this check
            # runs FIRST in the cycle, before _manage_stops — a resting
            # stop the broker already FILLED is journaled as the
            # server_stop it was (detect_source='remote_flatten'), same
            # helper as the breaker; OFF = no extra broker call.
            _attrib = _breaker_server_fill_attrib()
            _attributed = False
            for sym, pos in pre_flatten.items():
                if sym in self.positions:
                    continue
                if _attrib and self._breaker_record_server_fill(
                        sym, pos, site='remote_flatten'):
                    _attributed = True
                    continue
                q = self.get_quote(sym)
                px = q['midpoint'] if q else pos.entry_price
                pnl = ((px - pos.entry_price) / pos.entry_price * 100
                       if pos.entry_price > 0 else 0.0)
                record_trade(sym, 'sell', pos.entry_price, px, pnl,
                             exit_reason='remote_flatten', estimated=True)
            if _attributed:
                self._save_position_state()   # persist the new cooldown stamps
            if self.positions:
                notify(f"FLATTEN {self.get_asset_type()}: INCOMPLETE — "
                       f"{len(self.positions)} position(s) STILL OPEN and "
                       f"tracked ({', '.join(sorted(self.positions))}). "
                       f"Trading halted — /resume to re-enable entries.",
                       level='critical',
                       dedupe_key=f'flatten-{self.get_asset_type()}')
            else:
                notify(f"FLATTEN {self.get_asset_type()}: done (all "
                       f"positions closed). Trading halted — /resume to "
                       f"re-enable entries.",
                       level='critical',
                       dedupe_key=f'flatten-{self.get_asset_type()}')
        except Exception as e:
            logger.error("[FLATTEN] failed: %s", e)
            try:
                from notify import notify as _n
                _n(f"FLATTEN {self.get_asset_type()} FAILED: {e} — "
                   f"intervene manually", level='critical')
            except Exception:
                pass

    def _halt_cancel_working_buys(self, halted: bool) -> None:
        """strategy_config.HALT_CANCELS_WORKING_BUYS (default OFF — no-op).

        ON: the first halted cycle cancels this book's open BUY orders
        (symbol in this bot's universe, side 'buy', not a stop type) once
        per halt epoch; an un-halted cycle re-arms. A halt otherwise blocks
        only NEW entries — a working buy left by an unconfirmed cancel
        (maker rung 'maker_unknown', lifecycle give-up, stock 'day' parent)
        keeps working and can fill during the halt. SELL / stop orders are
        never touched (exits are never gated). The epoch is marked done
        only when the listing and every cancel succeeded — anything that
        raised retries on the next halted cycle. Never raises.
        """
        try:
            from strategy_config import HALT_CANCELS_WORKING_BUYS
        except ImportError:
            return
        if not HALT_CANCELS_WORKING_BUYS:
            return
        if not halted:
            self._halt_buys_cancelled = False
            return
        if getattr(self, '_halt_buys_cancelled', False):
            return
        try:
            from order_utils import _list_open_orders, _symbol_variants
            allowed = set()
            for s in self.get_symbol_universe():
                allowed |= _symbol_variants(s)
            orders = _list_open_orders(self.api, symbols=sorted(allowed)) or []
        except Exception as e:
            logger.warning("[HALT] open-order listing failed (%s) — working"
                           " buys NOT cancelled, retrying next cycle", e)
            return
        ok = True
        n = 0
        for o in orders:
            if (getattr(o, 'symbol', None) not in allowed
                    or str(getattr(o, 'side', '')).lower() != 'buy'
                    or 'stop' in str(getattr(o, 'type', '')).lower()):
                continue
            try:
                self.api.cancel_order(o.id)
                n += 1
            except Exception as e:
                ok = False
                logger.warning("[HALT] cancel %s %s failed: %s — retrying"
                               " next cycle", o.symbol, o.id, e)
        if n:
            logger.warning("[HALT] cancelled %d working buy order(s) for the"
                           " %s book", n, self.get_asset_type())
        self._halt_buys_cancelled = ok

    def _entries_allowed(self) -> bool:
        """Book-level entry gate shared by both loops (exits never gated).

        Gates: manual halt flag (Telegram /halt, GUI, or `touch
        trading_halt.flag`), then the scheduled macro-event stand-down
        (FOMC/CPI windows) — an hourly-bar model has no edge against an
        8:30 CPI print, in stocks OR crypto.

        Records WHY entries were blocked in self._entries_block_info so
        _execute_buys can journal stood-down skips with window identity
        (2026-08 influence audit §3.2: the stand-down "fires before
        journaling — structurally unpriceable" without this).
        """
        self._entries_block_info = None
        try:
            from notify import halt_active
            halted = halt_active()
            try:    # never lets the cancel path decide whether halt blocks
                self._halt_cancel_working_buys(halted)
            except Exception as e:
                logger.warning("[HALT] working-buy cancel failed: %s", e)
            if halted:
                self._entries_block_info = ('halt', 'trading_halt.flag', None)
                if self.cycle % 10 == 1:
                    logger.warning("[HALT] trading_halt.flag active — "
                                   "entries blocked (/resume to clear)")
                return False
        except Exception as e:
            if self.cycle % 10 == 1:
                logger.warning("[HALT] halt-flag check failed (%s) — "
                               "failing open, entries allowed", e)
        try:
            from macro_calendar import (macro_standdown, calendar_exhausted,
                                        standdown_window_id)
            blocked, reason = macro_standdown()
            if blocked:
                try:
                    wid = standdown_window_id()
                except Exception:
                    wid = None
                self._entries_block_info = ('macro_standdown', reason, wid)
                if self.cycle % 10 == 1:
                    logger.info("[MACRO] entries paused: %s", reason)
                return False
            if calendar_exhausted():
                # Static-table staleness alarm (2026-08 influence audit
                # §3.2: FOMC/CPI stand-down "needs a staleness alarm for
                # the static date table"). One loud warning per day + one
                # notify per process; fail-open — the gate's behavior is
                # unchanged, the alarm IS the fix.
                today = datetime.date.today().isoformat()
                if getattr(self, '_macro_cal_alarm_date', None) != today:
                    self._macro_cal_alarm_date = today
                    logger.warning(
                        "[MACRO] STALE CALENDAR: static FOMC/CPI table has "
                        "no future events — the stand-down gate can no "
                        "longer fire; refresh macro_calendar.py")
                    if not getattr(self, '_macro_cal_alarm_notified', False):
                        self._macro_cal_alarm_notified = True
                        try:
                            from notify import notify
                            notify("macro_calendar.py FOMC/CPI table "
                                   "exhausted — refresh needed "
                                   f"({self.get_asset_type()} book)")
                        except Exception:
                            pass
        except Exception as e:
            if self.cycle % 10 == 1:
                logger.warning("[MACRO] stand-down check failed (%s) — "
                               "failing open, entries allowed", e)
        return True

    def _execute_buys(self, preds: dict, snapshots: dict):
        """Buy bullish symbols with all risk checks."""
        if not self._entries_allowed():
            # Priced skip rows for FOMC/CPI stood-down candidates
            # (measurement-only; manual halt journals nothing)
            self._journal_standdown_skips(preds, snapshots)
            return
        from collections import Counter
        # IA-4 flag family (all default-OFF/today; see strategy_config).
        try:
            from strategy_config import (VIX25_BLOCK_REMOVED,
                                         CORR_FAMILY_MERGED, CORR_SANITY_MAX)
        except Exception:
            VIX25_BLOCK_REMOVED, CORR_FAMILY_MERGED = False, False
            CORR_SANITY_MAX = 0.85
        vc = Counter()          # veto attribution for the window summary
        admitted = []           # symbols that cleared every gate
        n_candidates = 0        # symbols evaluated past the mechanical gates
        symbols = self.get_symbol_universe()
        # Pred-descending entry rank across this cycle's evaluable
        # candidates — ANNOTATION ONLY, iteration order is unchanged.
        # Gives the crypto book the same entry_rank instrumentation the
        # stock book already journals (Stage-0 rank-gradient substrate;
        # any change to actual entry ORDERING is a deferred owner
        # decision gated on these journals).
        ranked = sorted((s for s in symbols if preds.get(s) is not None),
                        key=lambda s: preds[s], reverse=True)
        rank_map = {s: i + 1 for i, s in enumerate(ranked)}
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
                existing_value = self._cap_value(self.positions[symbol])
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
            # Decision-quote fetch time: slippage_bps currently conflates
            # execution slippage with decision->order staleness (full
            # sizing + a 0-5s jitter sleep sit between this fetch and the
            # order). quote_age_s in the buy row lets execution analysis
            # separate the two. No decision reads this key.
            quote['_fetched_ts'] = time.time()
            n_candidates += 1   # a real, evaluatable candidate this window
            snapshot = snapshots.get(symbol, {})

            # Prediction gate. Two former threshold modifiers were removed
            # as provably-dead code:
            # - the "recently hard-stopped -> 1.5x" bump was unreachable:
            #   _is_hard_stop_locked either `continue`s above or DELETES
            #   the expired key before returning False (2026-07 panel; a
            #   REAL post-lockout elevated bar is an owner decision);
            # - the Hurst<0.45 -> 1.3x mean-reversion shift could never
            #   fire: live Hurst is computed on price LEVELS (reads ~0.8+
            #   regardless of regime — HURST_ON_RETURNS in indicator_config
            #   is the legitimate future form, via harvest+retrain). Deleted
            #   2026-08-22 per the decision-influence ledger (unanimous);
            #   verbatim code in research/campaign_2026-08/08_removed_code.md.
            if not should_trade(pred_return, quote['spread_pct'],
                                asset_type=self.get_asset_type()):
                vc['cost_floor'] += 1
                # spread_pct journaled so decision_report can price this
                # veto at the REAL reject-time spread: cost_floor fires on
                # wide-spread names by construction, and a flat-spread
                # counterfactual reads it as "charging admission" falsely.
                self._journal_skip(symbol, 'cost_floor', rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot,
                                   spread_pct=round(quote['spread_pct'], 4))
                continue
            if pred_return < self.trade_threshold:
                vc['below_threshold'] += 1
                self._journal_skip(symbol, 'below_threshold',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot)
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
                                       rank=rank_map.get(symbol),
                                       pred=pred_return, snapshot=snapshot)
                    continue

            # Correlation check. CORR_FAMILY_MERGED (IA-4, ledger §3.3
            # MERGE): flag OFF = the binary >0.7 admission gate as today;
            # ON = the ENB stop-risk budget is the single correlation
            # consumer and only a loose sanity block (avg|corr| >
            # CORR_SANITY_MAX) remains at admission.
            avg_corr = None
            if self.corr_matrix and self.positions:
                _corr_kw = ({'max_avg_corr': CORR_SANITY_MAX}
                            if CORR_FAMILY_MERGED else {})
                allowed, avg_corr = check_portfolio_correlation(
                    list(self.positions.keys()), symbol, self.corr_matrix,
                    **_corr_kw)
                if not allowed:
                    vc['correlation'] += 1
                    self._journal_skip(symbol, 'correlation',
                                       rank=rank_map.get(symbol),
                                       pred=pred_return, snapshot=snapshot,
                                       avg_corr=round(avg_corr, 4),
                                       **({'corr_sanity': True}
                                          if CORR_FAMILY_MERGED else {}))
                    continue

            # Macro regime halt check. Skip rows journaled as 'vix_halt' —
            # now a priced skip class in decision_report.GATE_REASONS
            # (2026-08 influence audit §3.3: keep as THE one extreme VIX
            # read; it was previously the highest-impact stock gate with
            # the least evidence). vc key stays 'macro_halt' (entry_window
            # veto_counts back-compat). Measurement-only.
            if self.macro_regime and self.macro_regime.should_halt_stocks and self.get_asset_type() == 'stock':
                logger.info("%s: Halted by VIX > 35", symbol)
                vc['macro_halt'] += 1
                _vix = getattr(self.macro_regime, 'vix', None)
                self._journal_skip(symbol, 'vix_halt',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot,
                                   vix=(round(_vix, 1) if _vix is not None
                                        else None))
                continue

            # VIX > 25: block risky entries, allow safe-havens. Skip rows
            # journaled as 'vix25_block' + one daily log naming the
            # SAFE_HAVEN-untradable gap (C3) so the de facto halt is
            # visible. VIX25_BLOCK_REMOVED (IA-4, ledger §3.3 2-1
            # remove-candidate): ON skips the block entirely — the graded
            # VIX tier map owns the 25-35 band (one graded instrument per
            # latent, one extreme hard stop: the >35 halt above).
            if (not VIX25_BLOCK_REMOVED and self.macro_regime
                    and self.macro_regime.should_block_risky_entries
                    and self.get_asset_type() == 'stock'):
                from stock_config import SAFE_HAVEN_SYMBOLS
                if symbol not in SAFE_HAVEN_SYMBOLS:
                    logger.info("%s: Blocked — VIX > 25 defensive (non-safe-haven)", symbol)
                    vc['vix_block'] += 1
                    _vix = getattr(self.macro_regime, 'vix', None)
                    self._journal_skip(symbol, 'vix25_block',
                                       rank=rank_map.get(symbol),
                                       pred=pred_return, snapshot=snapshot,
                                       vix=(round(_vix, 1) if _vix is not None
                                            else None))
                    self._log_vix25_gap_once(SAFE_HAVEN_SYMBOLS)
                    continue

            # Sentiment gate: multiplier only — sentiment.sentiment_gate
            # clamps to [0.15, 1.5], so a gate<=0 veto branch was
            # mathematically unreachable and was deleted 2026-08-22 per the
            # decision-influence ledger (zero behavior change; verbatim code
            # in research/campaign_2026-08/08_removed_code.md). Making
            # sentiment a REAL veto remains an owner decision.
            gate, gate_reasons = sentiment_gate(symbol, self.get_asset_type())

            # LLM gate (veto first; multiplier folds into sizing tilt)
            llm_info = self.llm_scores.get(symbol, {})
            llm_s = llm_info.get('s', 0.5)
            llm_reason = llm_info.get('r', '')
            if llm_s < LLM_VETO_THRESHOLD:
                vc['llm_veto'] += 1
                self._journal_skip(symbol, 'llm_veto',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot,
                                   llm_score=llm_s,
                                   llm_reasoning=llm_reason)
                continue
            llm_mult = 0.5 + llm_s

            # Meta-labeling gate (veto + bounded sizing multiplier)
            meta_ok, meta_mult = self._meta_gate(symbol, pred_return, snapshots,
                                                 rank=rank_map.get(symbol))
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
                # q10/q10_floor arrive via _conv_fields (same snapshot
                # source, same rounding as the old inline row)
                self._journal_skip(symbol, 'q10_tail_veto',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot)
                continue

            # Single risk-based sizing call (all bounds enforced inside)
            sized_notional = self._compute_position_size(
                symbol, pred_return, quote,
                sentiment_mult=gate, llm_mult=llm_mult, meta_mult=meta_mult)
            if sized_notional <= 0:
                vc['sizing_zero'] += 1
                self._journal_skip(symbol, 'sizing_zero',
                                   rank=rank_map.get(symbol),
                                   pred=pred_return, snapshot=snapshot)
                continue

            # Order timing jitter (prevent pattern detection)
            time.sleep(random.uniform(0, 5))

            logger.info("%s: Sizing $%d (pred=%.4f)", symbol, sized_notional,
                        pred_return if pred_return else 0)

            admitted.append(symbol)
            conv = self._conv_fields(symbol, pred_return, snapshot,
                                     rank=rank_map.get(symbol))
            if avg_corr is not None:
                conv['avg_corr'] = round(avg_corr, 4)
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
        # ioc_fallback (c26 T6 / D21): inert flag-OFF — the kwarg is only
        # honored when order_utils.IOC_ENTRY_CAP_ENABLED is on.
        result = manage_order_lifecycle(
            self.api, order.id, timeout=self.ORDER_TIMEOUT,
            fallback_to_market=True,
            ioc_fallback={'quote_fn': (lambda: self.get_quote(symbol)),
                          'cap_bps': None,
                          'asset_type': self.get_asset_type()})
        return result, 'marketable'

    def _place_and_track_buy(self, symbol, notional, pred_return, quote,
                             gate, gate_reasons, llm_s, llm_mult, llm_reason,
                             conv=None):
        """Place buy order and update position tracking (base/crypto path).

        stock_loop does NOT override this — it reimplements the whole buy
        path inside its own _execute_buys and keeps the journal row in
        sync with this method BY HAND (see stock_loop.py ~line 1003).

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
            from order_utils import verify_position, _basis_or_zero
            pos = verify_position(self.api, symbol)
            if pos:
                # A null basis (not "0") raised here AFTER a real fill and
                # left the position untracked with no stop; it now takes
                # the zero-basis fallback below (ENGINE r7 H5).
                fill_price = _basis_or_zero(pos.avg_entry_price)
                if not fill_price > 0:
                    # Paper zero-basis quirk: the position reports
                    # avg_entry_price=0 for a filled buy. Entry 0 put the
                    # resting stop at 0*(1-d) = $0 (broker-rejected) and
                    # journaled fill_price 0 (ENGINE r1 W1). The order's
                    # own filled_avg_price is the honest basis; unchanged
                    # whenever the broker reports a positive average.
                    try:
                        ofp = float(getattr(result, 'filled_avg_price', 0) or 0)
                    except (TypeError, ValueError):
                        ofp = 0.0
                    if ofp > 0:
                        logger.warning("[BUY] %s: broker avg_entry_price=%s —"
                                       " using the order fill price $%s as the"
                                       " entry basis", symbol,
                                       pos.avg_entry_price, ofp)
                        fill_price = ofp
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
                    from market_data import (fetch_bars_alpaca,
                                             fetch_stock_bars_alpaca,
                                             closed_bars_v2_enabled)
                    _closed = closed_bars_v2_enabled()
                    if self.get_asset_type() == 'crypto':
                        df = fetch_bars_alpaca(self.api, symbol,
                                               closed_only=_closed)
                    else:
                        df = fetch_stock_bars_alpaca(self.api, symbol,
                                                     closed_only=_closed)
                    if df is not None and len(df) > 100:
                        # pandas pin (c26-T3/B21): ffill + fill_method=None ==
                        # pandas-2 pad semantics, pandas-3-proof
                        returns = df['Close'].ffill().pct_change(
                            fill_method=None).dropna().values * 100
                        from volatility import get_sigma
                        garch_sigma = get_sigma(symbol, returns, bars=df,
                                                asset_type=self.get_asset_type())
                except Exception:
                    pass

                is_add = symbol in self.positions
                hwm = fill_price
                if is_add:
                    hwm = max(self.positions[symbol].high_water_mark, fill_price)

                # Vertical-barrier age anchor (IA-4): stamp the FIRST
                # entry time; add-ons keep the original clock (max-hold
                # anchored at first entry, matching the label's entry bar).
                try:
                    if getattr(self, '_position_entry_ts', None) is None:
                        self._position_entry_ts = {}
                    if not is_add or symbol not in self._position_entry_ts:
                        self._position_entry_ts[symbol] = time.time()
                except Exception:
                    pass

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
                           "quote_age_s": (round(time.time() - quote['_fetched_ts'], 2)
                                           if quote.get('_fetched_ts') else None),
                           "entry_tactic": entry_tactic,
                           "maker": entry_tactic.startswith('maker'),
                           "skip_reason": None}
                if conv:
                    buy_rec.update(conv)
                # Sizing decomposition (2026-07 review): which de-risk
                # layers actually moved this fill's size, for the vol-stack
                # co-fire measurement (VIX is counted in BOTH the ladder
                # and macro_mult today).
                sizing_detail = getattr(self, '_last_sizing_detail', None)
                if sizing_detail:
                    buy_rec['sizing'] = sizing_detail
                # ENGINE r4 R4-a (measurement-only, additive; entry_tactic
                # above already names the tactic): broker ids of the order
                # that ACQUIRED the fill (maker ladder: the rung returned as
                # acquisition evidence) and the decision quote's bid/ask/
                # fetch stamp — joins the row to broker fills.
                for k, v in {**_order_journal_ids(result),
                             **_decision_quote_journal_keys(quote)}.items():
                    buy_rec.setdefault(k, v)
                log_decision(buy_rec)
                self.last_trade_time[symbol] = datetime.datetime.now()
                self._count_trade(symbol)
            else:
                # Coins were ACQUIRED but the position endpoint can't see
                # them (lag or transient error): the position is UNTRACKED —
                # no stop management, no journal row, no cooldown/budget
                # stamp — until a restart reconstructs it. Tracking-retry is
                # deferred (2026-07 review P2); surface it loudly.
                logger.error("[BUY] %s: acquired qty=%s but verify_position "
                             "returned None — position UNTRACKED until "
                             "restart (order status=%s)", symbol, partial_qty,
                             getattr(result, 'status', None))
                try:
                    from notify import notify
                    notify(f"BUY {symbol}: fill acquired but position "
                           f"unverified — UNTRACKED until restart",
                           level='warning',
                           dedupe_key=f'untracked-buy-{symbol}')
                except Exception:
                    pass

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
        jitter = random.uniform(-5, 5)
        sleep_interval = max(10, sleep_interval + jitter)

        logger.info("[SLEEP] Next check in %.0fs...", sleep_interval)
        time.sleep(sleep_interval)
