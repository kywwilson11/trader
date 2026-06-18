"""Single source of truth for strategy/policy parameters.

The trading loops AND the backtester read these — if they drift apart, the
backtest validates a different policy than the one trading. Keep every
tunable that affects entries/exits/sizing here.

Stop-distance floors: the old 5%/6% floors swallowed the ATR logic for
most names (raw 2-2.5x hourly ATR is ~0.6-3%), making stops effectively
fixed-percent and pushing the 3:1-RR take-profit to an unreachable 15-30%.
Floors now only guard against degenerate sub-spread stops; ATR does the
work. TP ratio lowered to 2:1 accordingly.
"""

CRYPTO_POLICY = {
    'atr_stop_mult': 2.5,
    'atr_trail_mult': 2.0,
    'trail_activate_pct': 0.015,
    'stop_floor_pct': 0.015,   # was 0.06 — floor only vs degenerate stops
    'stop_ceil_pct': 0.15,
    'tp_rr': 2.0,              # was 3.0 — 2:1 reachable within the horizon
    'tp_ceil_pct': 0.30,
    'stop_fallback_pct': 0.06,
    'trail_fallback_pct': 0.05,
    'cooldown_min': 60,
    'lockout_hours': 24,
}

STOCK_POLICY = {
    'atr_stop_mult': 2.0,
    'atr_trail_mult': 2.0,
    'trail_activate_pct': 0.01,
    'stop_floor_pct': 0.01,    # was 0.05
    'stop_ceil_pct': 0.10,
    'tp_rr': 2.0,              # was 3.0
    'tp_ceil_pct': 0.15,
    'stop_fallback_pct': 0.05,
    'trail_fallback_pct': 0.04,
    'cooldown_min': 20,
    'lockout_hours': 24,
}

# --- Sizing (risk-based; replaces the unbounded multiplier soup) ---
RISK_PCT_PER_TRADE = 0.005       # 0.5% of equity at risk per trade (to the stop)
MAX_BOOK_RISK_PCT = 0.025        # correlation-adjusted stop-risk cap per book
                                 # (equicorrelation ENB model in portfolio.py)
KELLY_CAP = 0.25                 # fractional Kelly ceiling (MacLean-Thorp-Ziemba)
PORTFOLIO_VOL_TARGET = {         # annualized portfolio volatility targets
    'crypto': 0.35,
    'stock': 0.18,
}
TILT_MIN, TILT_MAX = 0.70, 1.30  # combined regime/sentiment/LLM tilt bounds
HAR_VOL_ENABLED = True           # HAR-RV (realized range) sigma with GARCH
                                 # fallback — set False to force GARCH-only
CONVICTION_JOURNAL_ENABLED = True  # wave-5 Tier1-1: per-candidate veto
                                 # attribution + entry-window summaries in
                                 # the decision journal (measurement-only)
MIN_ORDER_NOTIONAL = 100         # skip dust orders that fees would eat

# Per-symbol DAILY entry budget: signal jitter re-trading the same name all
# day is pure fee bleed (each crypto round trip costs ~0.6%). Exits/stops
# are never budget-limited — only new entries.
MAX_TRADES_PER_SYMBOL_PER_DAY = {
    'crypto': 4,
    'stock': 3,
}

# --- Crypto maker entries (Alpaca fees: 15bps maker / 25bps taker) ---
MAKER_ENTRIES_ENABLED = True
MAKER_STAGE_TIMEOUT = 25         # seconds per bid-join rung (2 rungs max)

# --- Entry-tactic table (wave-7: replaces compute_limit_price's buried magic
# constants with explicit, calibrated thresholds the backtester reads too).
# Thresholds are from microstructure priors + the offline Eff_Spread_Pct
# ranking — NEVER tuned on realized P&L. Spreads are PERCENT of price.
EXEC_TAKER_FLOOR_PCT = 0.05      # spread <= this -> just cross (passive saves ~nothing, risks non-fill)
EXEC_WIDE_SPREAD_PCT = 0.15      # spread >= this -> candidate to POST inside the quote
EXEC_POST_INSIDE_FRAC = 0.40     # post this fraction of the half-spread inside from our side
EXEC_EDGE_HEADROOM_MULT = 1.5    # need pred >= this * edge_floor to risk a passive non-fill

# Marketable-IOC slippage caps (bps past the touch a taker order may pay before
# it cancels). ENTRY caps are tight (re-chase next loop); EXIT/flatten caps are
# WIDE with a true-market backstop so a stop can never silently fail to fill.
# Per name_class from the offline Eff_Spread_Pct ranking.
IOC_CAP_BPS = {'mega': 8, 'mid': 20, 'spec': 40}
IOC_EXIT_CAP_BPS = {'mega': 15, 'mid': 35, 'spec': 50}

# --- Stock entry windows (Gao-Han-Li-Zhou 2018: intraday predictability
# concentrates in the first/last half-hours; midday is noise+costs).
# Start at 9:45, not 9:30: the first 15 minutes are dominated by the
# opening auction unwind — wide spreads and quote-driven adverse
# selection that an hourly-bar model has no edge against. ---
STOCK_ENTRY_WINDOWS_ET = [
    ('09:45', '11:00'),
    ('14:30', '15:30'),
]
ENTRY_WINDOWS_ENABLED = True

# --- Overnight sleeve (Lou-Polk-Skouras 2019: equity premium accrues
# overnight; a small capped sleeve harvests it without full gap exposure) ---
OVERNIGHT_SLEEVE_ENABLED = True
OVERNIGHT_SLEEVE_MAX_POSITIONS = 2
OVERNIGHT_SLEEVE_MAX_PCT_EQUITY = 0.05   # per kept position
OVERNIGHT_SLEEVE_MIN_PRED = 0.0          # only keep names still predicted up


def policy_for(asset_type: str) -> dict:
    return CRYPTO_POLICY if asset_type == 'crypto' else STOCK_POLICY
