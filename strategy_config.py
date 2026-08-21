"""Single source of truth for strategy/policy parameters.

The trading loops AND the backtester read these — if they drift apart, the
backtest validates a different policy than the one trading. Keep every
tunable that affects entries/exits/sizing here.

Stop-distance floors: the old 5%/6% floors swallowed the ATR logic for
most names (raw 2-2.5x hourly ATR is ~0.6-3%), making stops effectively
fixed-percent and pushing the 3:1-RR take-profit to an unreachable 15-30%.
Floors now only guard against degenerate sub-spread stops; ATR does the
work. TP ratio lowered to 2:1 accordingly.

Related constants defined elsewhere (siblings — check them when editing here):
  - fees.FLAT_SPREAD_PCT (canonical flat spread) and its copy backtest.SPREAD_PCT
    (drift guarded by tests/test_review_b10.py).
  - cooldown_bars = max(1, ceil(cooldown_min/60)) is derived verbatim in BOTH
    meta_label.py and backtest.py (drift guarded by tests/test_improve_stratcfg.py).
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
TILT_MAX = 1.30    # combined regime/sentiment/LLM tilt BOOST cap (enforced in base_loop)
TILT_MIN = 0.70    # UNUSED/reserved — NOT enforced anywhere. The live de-risk floor is a
                   # hardcoded 0.1 in base_loop (tilt = max(0.1, min(TILT_MAX, tilt))),
                   # i.e. de-risking down to 10% is honored by design. Do not wire this
                   # in without an explicit decision (raising the floor 0.1 -> 0.70 is
                   # model-facing).
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

# --- Entry-tactic table (wave-7: named thresholds INTENDED to replace
# compute_limit_price's buried magic constants and be shared by the live
# loop and the backtester once wired).
# Thresholds are from microstructure priors + the offline Eff_Spread_Pct
# ranking — NEVER tuned on realized P&L. Spreads are PERCENT of price.
# DECLARED-AHEAD: execution_policy.choose_entry_tactic implements the table
# but has NO production caller yet — live stock entries still use
# order_utils.compute_limit_price's own constants, live crypto branches on
# MAKER_ENTRIES_ENABLED, and backtest.py reads none of these. Tuning these
# values changes nothing today.
EXEC_TAKER_FLOOR_PCT = 0.05      # spread <= this -> just cross (passive saves ~nothing, risks non-fill)
EXEC_WIDE_SPREAD_PCT = 0.15      # spread >= this -> candidate to POST inside the quote
EXEC_POST_INSIDE_FRAC = 0.40     # post this fraction of the half-spread inside from our side
EXEC_EDGE_HEADROOM_MULT = 1.5    # need pred >= this * edge_floor to risk a passive non-fill

# Marketable-IOC slippage caps (bps past the touch a taker order may pay before
# it cancels). ENTRY caps are tight (re-chase next loop); EXIT/flatten caps are
# WIDE with a true-market backstop so a stop can never silently fail to fill.
# Per name_class from the offline Eff_Spread_Pct ranking.
# DECLARED-AHEAD: order_utils.ioc_limit_price/place_marketable_ioc implement the
# mechanics but have NO production caller yet — no live order is IOC-capped today.
# Wiring these into the order path is owned by the execution/order_utils track.
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

# --- Earnings trading-day windows (D07, 2026-08 campaign, DEFAULT OFF) ---
# When ON, events_calendar's earnings buffers walk TRADING days (weekend +
# static NYSE-holiday aware): Friday entries/overnight holds are protected
# against Monday prints, and Monday gets the post-print size tilt after a
# Friday-AMC/weekend report. Only ever blocks MORE than calendar mode.
# Entry-gating change -> default OFF; flip on the Jetson after review.
EVENTS_TRADING_DAY_WINDOWS = False

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

# --- Promotion-gate v2: effective-n + selection-pressure accounting (2026-08 Q1) ---
# OFF (default): gate numerics BYTE-IDENTICAL to today; only side-by-side logging of the
# v2 calendar n_eff, the cum_trials deflation-pool line, study-DB deletion events, and
# MinTRL-on-failure reporting run (instrumentation). ON: (1) calendar-concurrency
# average-uniqueness n_eff (AFML ch.4, across ALL names) REPLACES both the per-ticker
# uniqueness and the connected-components cluster count — exactly ONE non-IID correction,
# never stacked with the Lo-2002 serial factor (CLAUDE.md gotcha #4); (2) n_eff < 10 fails
# CLOSED (dsr=0.0, status 'insufficient_effective_n') instead of being silently clamped up
# to 10; (3) DSR deflation pools unify on the persisted cumulative/overlap-weighted trial
# count (adaptive_state cum_trials / trial_history) for BOTH the fit gate and
# backtest --gate; (4) Thresholdout-shaped noisy best_score ratchet. Gate-behavior change:
# flip on the Jetson only, and expect the promotion bar to MOVE on first flip.
PROMOTION_GATE_V2 = False
# Kish design-effect softening of the calendar concurrency (per-hour weight
# 1/(1+(c_t-1)*rho_bar)) for books where measured pairwise rho < 1 makes lockstep 1/c_t
# over-harsh. Read ONLY when PROMOTION_GATE_V2 is ON. rho floors are the conservative
# lower bounds (rho_bar=1.0 reproduces the plain 1/c_t default).
KISH_NEFF_ENABLED = False
KISH_RHO_FLOOR = {'crypto': 0.5, 'stock': 0.25}

# --- Challenger-targeted policy gate (2026-08 Q2, defect D03) ---
# OFF (default): byte-identical legacy wiring. Under the DEFAULT shadow-mode weekly
# retrain, hypersearch saves the fresh model to the CHALLENGER slot while backtest --gate
# replays the CHAMPION: the model that will actually deploy is never policy-gated (it
# promotes on the shadow DM forecast test alone) and a gate failure rolls the INNOCENT
# live champion back to a STALE .prev. run_pipeline logs this loudly every weekly run
# while OFF. ON: the weekly gate passes --model-prefix <challenger slot> so backtest.py
# scores CHALLENGER artifacts on the champion's book data (same thresholds: net Sharpe
# > 0, DSR >= DSR_MIN, n >= 10); exit 3 then means HOLD the challenger — champion and
# its .prev are never touched, the challenger keeps shadowing, and the verdict lands in
# {slot}_policy_gate.json for the shadow-side promotion pre-flight (future shadow.py
# change; see backtest.py docstring). Gate-behavior change: flip on the Jetson only.
GATE_TARGETS_CHALLENGER = False

# --- Long-only objective scoring (2026-07 review, DEFAULT OFF) ---
# hypersearch's simulate_trades historically booked a SHORT leg (-r - cost on
# p < -threshold) into the trial score AND the holdout DSR, but the live book
# is long-only: a model whose certified edge is carried by bear-side accuracy
# deploys only its weak long side. True = score longs only (the deployable
# policy). Flipping this changes trial scores — old Optuna scores become
# incomparable, so flip ONLY on the Jetson together with CLAUDE.md gotcha #2
# (delete v2_study.db + stock_v2_study.db, reset the adaptive best_score).
OBJECTIVE_LONG_ONLY = False

# --- Hypersearch model-fit honesty v3 (2026-08 T1, D22/D23/D25 / 02_research B12) ---
# OFF (default): search/gate/save flow BYTE-IDENTICAL to today (fold-max checkpoint
# ships, holdout gate scores raw LSTM, LGB trains after the save, lstm_weight stays
# the hardcoded 0.6 default). ON (Jetson): (1) ONE final refit of the winning config
# on ALL pre-holdout data (train purged so label windows complete before the holdout
# boundary; scaler refit on the full region; FIXED epoch budget = median of the
# winning trial's per-fold best epochs, no early stopping; SWA tail soup = uniform
# average of the LAST 4 epoch checkpoints; regime tripwire warns — never blocks —
# when the newest fold Sharpe is negative while the trial mean is positive);
# (2) the LGB mean+q10 legs train BEFORE the holdout gate, on the SHIPPING scaler
# (predict_now feeds both legs one scaler — train/serve parity), with NOTHING
# written to disk until the gate passes; (3) blend_fit.fit_blend_weight_v2 (NNLS
# estimator + label-overlap SE significance gate + Diebold-Shin shrink 0.5/0.5,
# cross-retrain smoothing vs the champion's previous weight) writes
# config['lstm_weight'] — the key predict_now.py:405 / backtest.py already read;
# (4) the holdout DSR certificate is issued against the BLENDED predictor with the
# q10 tail veto applied to long entries — the certified predictor IS the deployed
# predictor (the ~10-15 min refit runs sequentially under the existing GPU lock;
# memory profile unchanged).
# RUNBOOK (gotcha #2 — ONE study-reset retrain event): flip HYPERSEARCH_V3 +
# OBJECTIVE_V3 TOGETHER; delete v2_study.db + stock_v2_study.db, reset the adaptive
# best_score, and reset cum_trials via the B-1 sanctioned gotcha-#2 reset.
# Optionally fold OBJECTIVE_LONG_ONLY and --preset stationary_lean into the SAME
# event (owner's call — this spec flips neither). NOTE: OBJECTIVE_V3 changes the
# trade_threshold Optuna distribution — reusing an old study DB would make Optuna
# reject the changed distribution; the study reset is mandatory, not optional.
HYPERSEARCH_V3 = False

# --- Objective scoring v3 (2026-08 T1, D24-part + D05-threshold / 01_state_map) ---
# OFF (default): trial scoring BYTE-IDENTICAL. ON: (1) simulate_trades resets the
# position walk at ticker-block boundaries (a hold entered near the end of one
# ticker's block no longer swallows the first bars of the NEXT ticker's block);
# (2) the Optuna trade_threshold range is floor-anchored to the book's deployment
# edge: [0.8x, 2.5x] fees.required_edge_pct(asset, FLAT_SPREAD_PCT[asset]), upper
# clamped to 2.0 (objective_utils.v3_trade_threshold_range: crypto [0.96, 2.0],
# stock [0.18, 0.57]) — replacing the legacy [0.05, 1.0] that sits ENTIRELY below
# the 1.20% crypto floor; (3) walk-forward VAL rows whose label windows cross into
# the holdout are purged (they previously leaked holdout returns into checkpoint /
# threshold selection). Changes trial SCORES — same runbook as HYPERSEARCH_V3
# above; the adaptive state's own trade_threshold range/edge-expansion is ignored
# (overridden) while this flag is ON.
OBJECTIVE_V3 = False

# --- Meta-label probability calibration (wave-9 #1) ---
# 'legacy'     = original isotonic fit on the same val slice the booster early-
#                stopped on (a leak -> upward-biased p that gates cost + sizes bets).
# 'purged_oof' = calibrate on PURGED out-of-fold predictions (leak-free; picks
#                sigmoid on thin books). Flip on the Jetson after a reliability /
#                Brier-before-after check, and re-certify in shadow BEFORE enabling
#                any p-consuming sizing lever (edge-Kelly, conviction tiers).
META_CALIBRATION_MODE = 'legacy'

# --- Calibration mechanics v2 (2026-08 R1, defect D13c / 02_research B04.2) ---
# OFF (default): every calibrator output byte-identical to legacy (pinned).
# ON (Jetson, after a scripts/reliability_report.py before/after check):
#   (1) isotonic pools tied scores by weighted mean BEFORE PAVA (de Leeuw 1977
#       "secondary method" — fixes the order-dependent tie collapse that can
#       calibrate a true-10% bucket to p~0.90);
#   (2) Platt fits on the LOGIT of the score with (N+1)/(N+2) target smoothing
#       (Platt 1999; Niculescu-Mizil & Caruana 2005 — bounded p, no
#       quasi-separation divergence);
#   (3) the purged-OOF calibration split gets a real embargo (0.05 of the test
#       fold's time span; today it runs with embargo=0.0), and
#   (4) the legacy same-slice calibration branch routes through
#       calibration.fit_calibrator's size-aware chooser (sigmoid below 1000
#       points) instead of raw sklearn isotonic on a 40-100-point slice.
# Model-facing: changes calibrated p -> veto/size. Certify BEFORE flipping
# META_CALIBRATION_MODE='purged_oof' — that A/B is uninterpretable on the
# tie-collapsing isotonic.
CALIBRATION_V2 = False

# --- Meta OOF primary predictions (2026-08 R2, defect D12 / 02_research B04.1) ---
# hypersearch_v2 now ALWAYS persists the winning config's purged walk-forward
# VALIDATION-fold predictions as {prefix}oof_preds.npz (instrumentation, direct;
# fingerprinted to the manifest's saved_at+score; NEVER contains holdout rows),
# and train_meta ALWAYS stamps pred_source ('in_sample'|'oof') into meta_meta.json.
# This flag gates CONSUMPTION only.
# OFF (default): train_meta keeps the current IN-SAMPLE primary 'pred' path,
# byte-identical, with a LOUD "[META] pred feature is IN-SAMPLE" breadcrumb.
# ON (Jetson): when the npz exists AND its fingerprint matches the current
# champion manifest, OOF predictions drive BOTH the 'pred' feature and the entry
# filter; rows outside OOF coverage are DROPPED (never backfilled). Starvation
# tiers (B04.3): n>=1000 full booster params; 200<=n<1000 shrunk tier
# (num_leaves=8, max_depth=3, min_data_in_leaf=max(20,n//20), feature_fraction=0.6);
# n<200 falls back to the in-sample path with the LOUD warning (no hard refusal).
# Model-facing: changes the trained meta artifact. A/B before trusting: honest
# val AUC is EXPECTED lower; flip only if honest holdout veto precision at
# p<0.30 >= the leaked variant's (02_research B04.1).
# KNOWN COMPOSITION SEAM: the persisted OOF preds are the LSTM leg only,
# while live serves the meta gate the lstm_weight blend (and HYPERSEARCH_V3
# refits that weight per retrain) — the mandated holdout-veto-precision A/B is
# the guard; meta_meta.json stamps pred_composition.
META_OOF_PRED = False

# --- Meta replay policy parity (2026-08 R2, defect D05-meta) ---
# OFF (default): _gen_meta_rows' row population byte-identical (admission = 0.5x
# threshold + cooldown + EOD only). Row-count diagnostics (rows_legacy /
# rows_parity, per-condition first-fail drop counts) are ALWAYS computed and
# stamped into meta_meta.json (instrumentation, direct); q10 is only scored when
# the flag is ON (extra booster inference — Jetson memory priority).
# ON (Jetson): the replay applies the SAME admission conditions the deployed
# policy enforces — required_edge_pct cost floor on the flat spread
# (backtest.simulate_ticker convention; live uses the real quote), the
# max(cooldown, lockout_hours-in-bars) wait after hard-stop exits, the stock
# entry-window mask (ENTRY_WINDOWS_ENABLED semantics via backtest's
# _entry_window_mask), and the q10 tail veto where {prefix}lgb_q10.txt exists.
# Model-facing: changes the meta training population -> the veto/size-tilt.
META_REPLAY_POLICY_PARITY = False

# --- De-risk multiplier stack v2 (2026-08 S3, defects D10/D29 / 02_research B06) ---
# OFF (default): sizing arithmetic BYTE-IDENTICAL to today (pinned by
# tests/test_c26_base_loop_functional.py + test_c26_S3.py); the v2 composition is
# computed and journaled SHADOW-only in the buy row's sizing detail, and the BTC
# trailing-RV history file warms in the background (instrumentation, direct).
# ON (Jetson, after reviewing scripts/sizing_cofire_report.py evidence):
#   (a) regime family {VIX tier, STLFSI2 stress, book-vol scalar; BTC-RV for crypto}
#       aggregates by MINIMUM (Frechet bound — comonotone estimates of ONE latent
#       risk-off state), product kept only ACROSS families (drawdown ladder,
#       correlation, alpha tilts);
#   (b) exactly ONE VIX tier map (macro_indicators.vix_tier_mult_v2: <25 -> 1.0,
#       25-35 -> 0.5, >35 -> 0.3, hysteresis enter 25/35 exit 22/31) — base_loop's
#       inline 15/25/35 ladder and the macro sizing_mult VIX tiers do NOT also apply;
#   (c) modal regime (VIX 15-25) sizes at exactly 1.0 (BKvD 2020: modal-state cuts
#       are pure foregone exposure);
#   (d) crypto book replaces VIX with BTC's own trailing Parkinson-RV percentile
#       state (volatility.get_crypto_rv_mult; VIX stays stock-only);
#   (e) pseudo-CAPE multiplier EXCLUDED (KILL_LIST item; code retained pending
#       owner deletion) and (f) HMM multiplier EXCLUDED (kill-recommended;
#       inverted smoothing documented in regime_detector.py) — both still
#       computed and journaled;
#   (g) PORTFOLIO_VOL_TARGET applied at exactly ONE scope: the book-level scalar
#       (portfolio.get_book_vol_scalar_cached, inside the family min); the
#       per-position GARCH ratio composes at 1.0 (the ATR risk base already
#       normalizes per-position vol via stop distance)
#       (under TRADER_HAR_DAILY_FEED this makes the HAR sizing feed
#       journaled-only while v2 is ON — see market_data.har_daily_feed_enabled);
#   (i) deposit-contaminated |daily return| > 0.15 outliers EXCLUDED from the
#       book-vol EWMA recursion (beta_ledger finite+positive pattern).
# Hard floors are UNTOUCHED in both modes: macro emergency zero (D26) and
# MIN_ORDER_NOTIONAL; the 0.1 advisory floor keeps applying ONCE to the composed
# tilt. Model-facing: changes admitted sizes -> flip on the Jetson only.
DERISK_STACK_V2 = False

# BTC trailing-RV regime state constants (read only by volatility.py; B06:
# enter immediately, exit slowly — asymmetric Schmitt per crypto_trend.py).
CRYPTO_RV_ENTER_HIGH_PCT = 80.0     # BKvD top-quintile
CRYPTO_RV_ENTER_CRISIS_PCT = 95.0
CRYPTO_RV_EXIT_HIGH_PCT = 65.0      # hold below this CRYPTO_RV_EXIT_HOLD_EVALS new bars
CRYPTO_RV_EXIT_CRISIS_PCT = 90.0    # crisis -> high (immediate)
CRYPTO_RV_EXIT_HOLD_EVALS = 12      # consecutive new-hourly-bar evaluations
CRYPTO_RV_MIN_HISTORY_DAYS = 90     # below this the state is 'unknown' (fail-OPEN 1.0)
CRYPTO_RV_HIGH_MULT = 0.5
CRYPTO_RV_CRISIS_MULT = 0.3

# ============================================================================
# WAVE-9 FORWARD-DECLARED FLAGS — RESERVED / NOT YET WIRED.
# None of the constants from here through TIER_A_K has a production reader:
# the kernels (bet_sizing.afml_bet_size/kelly_edge_odds, panel_ranks.cs_size_tilt,
# crypto_trend.trend_scalar, portfolio_backtest.conviction_gated) take their
# thresholds as FUNCTION ARGUMENTS. Flipping any flag below is a SILENT NO-OP
# today. Activation = the Jetson wiring step, which must import these constants
# at the call sites. tests/test_improve_stratcfg.py asserts they stay default-off
# until that wiring lands (update the test in the same change that wires them).
# ============================================================================
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
# OFF by default. Graded BTC-200h-SMA de-risk scalar to be COMPOSED INTO
# CryptoLoop._extra_tilt — which today returns the perp FUNDING tilt
# (funding.funding_tilt); the two must MULTIPLY, not overwrite —
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
