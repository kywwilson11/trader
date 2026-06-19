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

# --- Square-root market-impact cost (wave-8 #6) ---
# A $100 and a $50k order into a thin spec name cost the same bps in the offline
# cost model today; real impact grows ~ sqrt(notional/ADV). OFF by default —
# enabling it ADDS a per-name impact haircut to the OFFLINE backtest/meta net
# P&L (strictly higher cost, never live behavior), de-certifying edge that only
# survives because size is under-priced on illiquid names. Flip on only after
# stamping DV30 into the harvested data and calibrating k / typical notional on
# the Jetson (see liquidity.market_impact_pct).
IMPACT_COST_ENABLED = False
IMPACT_K = 1.0                   # sqrt-impact coefficient (Almgren/Kyle)
IMPACT_TYPICAL_NOTIONAL = 25_000 # representative $ order size for the %-return replay

# --- Average-uniqueness training weights (wave-8 #1) ---
# The DSR gate already deflates by effective-n, but the trainers still over-count
# overlapping hourly labels ~k times. OFF by default — flip on the Jetson AFTER
# scripts/wave6_stage0.py measures u_bar per book (crypto u_bar<0.30 -> ship;
# stock EOD-capped near-IID -> near no-op). Use PURE uniqueness (never blend
# uniqueness x |return|), and delete v2_study.db before the first weighted retrain
# (the loss change makes old Optuna scores incomparable — CLAUDE.md gotcha #2).
UNIQUENESS_WEIGHTS_ENABLED = False

# --- Meta-label probability calibration (wave-9 #1) ---
# 'legacy'     = original isotonic fit on the same val slice the booster early-
#                stopped on (a leak -> upward-biased p that gates cost + sizes bets).
# 'purged_oof' = calibrate on PURGED out-of-fold predictions (leak-free; picks
#                sigmoid on thin books). Flip on the Jetson after a reliability /
#                Brier-before-after check, and re-certify in shadow BEFORE enabling
#                any p-consuming sizing lever (edge-Kelly, conviction tiers).
META_CALIBRATION_MODE = 'legacy'

# --- Edge/probability bet sizing (wave-9 #5) ---
# OFF by default and HARD-GATED on META_CALIBRATION_MODE='purged_oof' being live
# and certified: edge-Kelly over-bets on an optimistic p (Chopra-Ziemba). When
# enabled (Jetson, after Stage-0 shows a real rank gradient) it replaces the
# flat-topped clip(2p,0.6,1.3) with bet_sizing.afml_bet_size / kelly_edge_odds,
# kept inside KELLY_CAP + the ENB book cap, and must move OUTSIDE the TILT_MAX
# clamp or it is dead on arrival.
EDGE_KELLY_ENABLED = False

# --- Crypto cross-sectional rank tilt (wave-9 #6) ---
# OFF by default. A SOFT [0.90,1.10] size tilt toward the relative-strength
# leader (never an exclusion — every laggard already cleared the 2x cost floor).
# Gate by dispersion so pure-BTC-beta hours are no-ops. Model-facing (the crypto
# panel changes); enable on the Jetson after a retrain on the full coin set + a
# Stage-0 measurement that the laggard actually realizes lower net P&L.
CRYPTO_CS_RANK_ENABLED = False
CRYPTO_CS_DISPERSION_FLOOR = 0.01

# --- BTC trend / TSMOM risk-off gate (wave-9 #7) ---
# OFF by default. Graded BTC-200h-SMA de-risk scalar via CryptoLoop._extra_tilt,
# debounced (Schmitt + persistence). Wiring MUST floor the COMBINED macro x HMM x
# book-vol x trend product (4 de-risk terms can stack-collapse size) and run the
# co-fire counterfactual (if it fires with the shipped vol-scaler >70% of the
# time the marginal edge is ~0 -> kill it). Needs 220-bar BTC history on the Jetson.
CRYPTO_TREND_GATE_ENABLED = False
CRYPTO_TREND_SMA_WINDOW = 200
CRYPTO_TREND_FLOOR = 0.5

# --- Conviction-gated dynamic top-K + tier sizing flagship (wave-9 #4) ---
# OFF by default. Admit fewer/higher-conviction names and concentrate the top
# tier by edge. STRICTLY Stage-0-gated on the Jetson: certify via
# portfolio_backtest.compare_deflated (DSR after deflation by #policy-configs)
# AND decision_report rank-1-3 net >= ~2x rank-6-7 in BOTH live journals and the
# holdout, else it is regime-mining. When CONCENTRATION_ENABLED=False the
# conviction walk reduces EXACTLY to the incumbent flat top-K (no-op kill switch).
# Tier-A concentration is also inert until the $5k notional cap is raised (which
# must clear the wave-8 market-impact model). Sequence AFTER the calibration fix.
CONCENTRATION_ENABLED = False
CONVICTION_K_MAX = 7
CONVICTION_K_MIN = 3            # Statman diversification floor
CONVICTION_SIGNAL_FLOOR = None  # None = floor not applied
CONVICTION_META_FLOOR = None
CONVICTION_RATIO_FLOOR = None
TIER_SIZING_ENABLED = False
TIER_A_K = 3                   # top-K names that get edge-proportional concentration

# --- Bar-keyed prediction cache (wave-8 #5) ---
# Hourly bars + 30s loop => ~119/120 inference cycles recompute a bit-identical
# feature+LSTM+LGB result. Memoizing on the latest closed-bar timestamp skips
# them — the biggest idle-CPU win on the 8GB Jetson. OFF by default; enable on
# the Jetson after confirming bit-identical predicted_return on real symbols and
# a clean cache-clear on model hot-reload (see prediction_cache.py).
PREDICTION_CACHE_ENABLED = False

# --- Cross-book account stop-risk (wave-8 #7) ---
# The per-book ENB cap (MAX_BOOK_RISK_PCT) runs independently, so the stock book
# (COIN/MSTR/MARA) and crypto book (spot BTC/ETH) can each run to 2.5% behind the
# SAME factor — ~5% combined vs the intended ~3%. CROSS_BOOK_RHO is the assumed
# cross-book correlation for the GATE-1 measurement journal; 1.0 = the risk-off
# lockstep worst case. The measurement only LOGS; the live clamp (Jetson) will
# replace this with the realized cross-book correlation it accumulates.
CROSS_BOOK_RHO = 1.0


def policy_for(asset_type: str) -> dict:
    return CRYPTO_POLICY if asset_type == 'crypto' else STOCK_POLICY
