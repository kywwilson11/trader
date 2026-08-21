"""Shared utilities for the crypto and stock trading loops.

Centralizes duplicated code: API construction, model hot-reload helpers,
cooldown tracking, inference device selection, the predict_symbol wrapper,
and Kelly criterion position sizing.
"""

import json
import os
import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# NOTE: predict_now (and through it torch, ~300MB RSS) is imported lazily
# inside predict_symbol() — harvest scripts and the GUI construct API
# clients through this module and must NOT pay the torch import.

load_dotenv()


# --- SHARED CONSTANTS ---
# Single source of truth for LLM_VETO_THRESHOLD, THERMAL_THROTTLE_TEMP and
# TEMP_LOG_EVERY_N_CYCLES (imported by base_loop/stock_loop). ORDER_TIMEOUT
# is NOT centralized here: the live loops use their own class attributes
# (base_loop.BaseTradingLoop.ORDER_TIMEOUT, overridden in stock_loop) —
# keep those in sync by hand.

LLM_VETO_THRESHOLD = 0.15      # LLM score below this = catastrophic veto
THERMAL_THROTTLE_TEMP = 75     # GPU temp threshold for throttling (Celsius)
ORDER_TIMEOUT = 30             # Seconds to wait for limit order fill
TEMP_LOG_EVERY_N_CYCLES = 10   # Log GPU temp every N cycles


# --- API CONSTRUCTION ---

def _install_rest_timeouts(api) -> None:
    """Best-effort request timeouts on the SDK's requests.Session(s)
    (c26 D01). Legacy alpaca-trade-api exposes api._session; the
    alpaca-py CompatREST shim holds three inner clients each with a
    _session. Fail-open: a shape mismatch leaves the SDK untouched but
    is logged (hung-socket protection off)."""
    try:
        from order_utils import install_session_timeout
        candidates = [
            getattr(api, '_session', None),
            getattr(getattr(api, '_trading', None), '_session', None),
            getattr(getattr(api, '_stock_data', None), '_session', None),
            getattr(getattr(api, '_crypto_data', None), '_session', None),
        ]
        n = sum(1 for s in candidates
                if s is not None and install_session_timeout(s))
        if n == 0:
            print("[API] WARNING: could not install REST request "
                  "timeouts (unknown SDK session shape) — a hung "
                  "socket can wedge a worker thread")
    except Exception as e:
        print(f"[API] WARNING: REST timeout install failed: {e}")


def get_api():
    """Build an Alpaca REST client from .env credentials.

    Prefers the legacy alpaca-trade-api SDK (battle-tested in this repo)
    but transparently switches to the maintained alpaca-py SDK via
    alpaca_compat.CompatREST when:
      - TRADER_USE_ALPACA_PY=1 is set (opt-in / testing), or
      - alpaca-trade-api can no longer be imported (it is unmaintained
        since 2022 and its dependency pins conflict with modern packages —
        rot will eventually land).
    """
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_API_SECRET')
    base_url = os.getenv('ALPACA_BASE_URL')

    # Fail loud on misconfiguration: None credentials otherwise surface as
    # confusing 401s deep in the SDKs, and an unset base_url makes BOTH SDKs
    # default to the LIVE trading endpoint (this system is paper-only).
    for name, val in (('ALPACA_API_KEY', key), ('ALPACA_API_SECRET', secret)):
        if not val:
            print(f"[API] ERROR: {name} is not set — requests will fail "
                  f"with 401 (check .env)")
    if not base_url:
        print("[API] WARNING: ALPACA_BASE_URL is not set — the SDKs "
              "default to the LIVE trading endpoint, not paper")

    if os.environ.get('TRADER_USE_ALPACA_PY') != '1':
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(key, secret, base_url, api_version='v2')
            _install_rest_timeouts(api)
            return api
        except ImportError:
            print("[API] alpaca-trade-api unavailable — using alpaca-py adapter")

    from alpaca_compat import CompatREST
    api = CompatREST(key, secret, base_url)
    _install_rest_timeouts(api)
    return api


# --- MODEL HOT-RELOAD HELPERS ---

def get_model_mtime(path):
    """Get modification time of a model file, or 0 if it doesn't exist."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def model_reload_key(model_prefix=''):
    """Reload key for hot-reload checks.

    Prefers the save manifest (written LAST, after all four artifacts are
    atomically in place) so a bot never reloads mid-save and pairs new
    weights with an old scaler. Falls back to the .pth mtime for models
    saved before manifests existed.
    """
    p = f'{model_prefix}_' if model_prefix else ''
    # get_model_mtime (not exists+getmtime) so a manifest vanishing between
    # the two calls degrades to the .pth fallback instead of raising.
    m = get_model_mtime(f'{p}model_v2.manifest.json')
    return m if m else get_model_mtime(f'{p}model_v2.pth')


# --- INFERENCE DEVICE ---

def choose_inference_device():
    """Always use CPU for inference in the trading bots.

    The model is ~1.3MB — CPU inference is fast enough for 30-second trading
    cycles (a few ms per symbol). Reserving GPU exclusively for training
    eliminates the CUDA OOM crashes that happen when inference and training
    compete for the Jetson's 8GB unified memory.
    """
    return 'cpu'


# --- COOLDOWN ---

def cooldown_ok(last_trade_time, symbol, cooldown_minutes=30):
    """Return True if the symbol is not in cooldown.

    Args:
        last_trade_time: dict mapping symbol -> datetime of last trade
        symbol: symbol to check
        cooldown_minutes: minimum minutes between trades on the same symbol
    """
    if symbol not in last_trade_time:
        return True
    elapsed = (datetime.datetime.now() - last_trade_time[symbol]).total_seconds()
    return elapsed >= cooldown_minutes * 60


# --- PREDICTION WRAPPER ---

def predict_symbol(api, symbol, model, config, scaler_X, feature_cols,
                   inference_device, asset_type='crypto', benchmark_close=None,
                   return_snapshot=False):
    """Run a regression prediction for a single symbol.

    Returns:
        (symbol, pred_return) tuple where pred_return is float or None
        If return_snapshot: (symbol, pred_return, snapshot_dict)
    """
    from predict_now import get_live_prediction

    extra_kwargs = {}
    if asset_type == 'stock':
        extra_kwargs['spy_close'] = benchmark_close
    else:
        extra_kwargs['btc_close'] = benchmark_close

    result = get_live_prediction(
        symbol, model, scaler_X, config, feature_cols,
        api=api, inference_device=inference_device,
        asset_type=asset_type, return_snapshot=return_snapshot,
        **extra_kwargs,
    )
    if return_snapshot:
        pred, snapshot = result if result else (None, None)
        return symbol, pred, snapshot
    return symbol, result


# --- KELLY CRITERION ---

_TRADE_MEMORY_FILE = Path(__file__).resolve().parent / "trade_memory.json"


def compute_kelly_fraction(min_trades: int = 50,
                           asset_type: str | None = None) -> float | None:
    """Compute half-Kelly fraction from trade history.

    Uses trade_memory.json to calculate:
        Kelly f = (win_rate * avg_win/avg_loss - (1 - win_rate)) / (avg_win/avg_loss)
    with win_rate and payoff ratio first SHRUNK toward a skeptical prior
    (50 pseudo-trades at breakeven: win rate 0.5, payoff 1.0), then halved.

    Args:
        min_trades: Minimum trades required before Kelly activates
        asset_type: 'crypto'/'stock' restricts the sample to that book —
            the two books have different edge distributions (fee scale,
            horizon, vol), so a hot stock fortnight must not ramp crypto
            sizing (or vice versa). None pools everything.

    Returns:
        Half-Kelly fraction clamped to [0.05, 0.25], or None if history is
        insufficient/degenerate. The 0.05 floor applies even when the raw
        Kelly is negative — a losing edge surfaces as the floor, never 0.
        Consumer contract: base_loop maps 0.125 (the clamp midpoint) to a
        1.0x sizing multiplier, bounded to [0.5x, 1.5x].
    """
    try:
        if not _TRADE_MEMORY_FILE.exists():
            return None
        with open(_TRADE_MEMORY_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # Flatten trades across symbols (optionally one book). Exclude
    # 'estimated' records — unconfirmed exits journaled at pre-slippage
    # quote midpoints (worst on exactly the stop-outs) inflate
    # avg_win/avg_loss and thus Kelly.
    all_trades = []
    for symbol, trades in data.items():
        is_crypto = '/' in symbol
        if asset_type == 'crypto' and not is_crypto:
            continue
        if asset_type == 'stock' and is_crypto:
            continue
        all_trades.extend(t for t in trades if not t.get('estimated'))

    if len(all_trades) < min_trades:
        return None

    # Recency means TIME: the flat list is grouped by symbol, so slicing
    # without this sort kept "all trades of whichever symbols happened to
    # iterate last" instead of the newest 200
    all_trades.sort(key=lambda t: t.get('ts', ''))
    recent = all_trades[-200:]
    wins = [t for t in recent if t.get('pnl_pct', 0) > 0]
    losses = [t for t in recent if t.get('pnl_pct', 0) < 0]

    if not wins or not losses:
        return None

    win_rate = len(wins) / len(recent)
    avg_win = np.mean([t['pnl_pct'] for t in wins])
    avg_loss = abs(np.mean([t['pnl_pct'] for t in losses]))

    if avg_loss == 0:
        return None

    # SHRINKAGE toward a skeptical prior (50 pseudo-trades at breakeven:
    # win_rate 0.5, payoff ratio 1.0). Raw Kelly from a hot recent sample
    # ramps size at exactly the wrong time — regime tops — and Rising &
    # Wyner show fractional Kelly is equivalent to shrinking the estimated
    # edge toward a prior; this makes the shrinkage explicit so a 1.5x
    # multiplier requires SUSTAINED evidence, not one lucky fortnight.
    n = len(recent)
    prior_n = 50
    win_rate = (win_rate * n + 0.5 * prior_n) / (n + prior_n)
    win_loss_ratio = (avg_win / avg_loss * n + 1.0 * prior_n) / (n + prior_n)
    kelly_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    # Half-Kelly, clamped to [0.05, 0.25]
    half_kelly = max(0.05, min(0.25, kelly_f / 2))
    return half_kelly


def kelly_position_size(base_notional: float, equity: float,
                        min_trades: int = 50) -> float:
    """Compute Kelly-based position size with bounds.

    Args:
        base_notional: Default fixed notional per trade
        equity: Current account equity
        min_trades: Minimum trades before Kelly activates

    Returns:
        Adjusted notional (floored at base, capped at 3x base).
    """
    kelly_f = compute_kelly_fraction(min_trades)
    if kelly_f is None:
        return base_notional

    kelly_notional = kelly_f * equity
    # Floor at base_notional, ceiling at 3x base
    return max(base_notional, min(base_notional * 3, kelly_notional))
