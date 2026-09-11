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
  - fees.FLAT_SPREAD_PCT (canonical flat spread) and its TWO copies,
    backtest.SPREAD_PCT and meta_label._gen_meta_rows' inline spread literal
    (both drift-guarded by tests/test_review_b10.py).
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
# {slot}_policy_gate.json for the shadow-side promotion pre-flight (consumed by
# shadow._gate_preflight). Gate-behavior change: flip on the Jetson only.
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
# config['lstm_weight'] — the key predict_now.get_live_prediction / backtest.py
# already read;
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

# --- Blend-coherence sub-flags (2026-08 R2-C packet R2C-02, defects M2 + H2) ---
# Both act ONLY inside the HYPERSEARCH_V3 save path (inert while it is False) and
# ride Chain-2's single gotcha-#2 study-reset event — never a reset of their own.
# BLEND_FIT_ON_REFIT — OFF (default): the DEPLOYED lstm_weight comes from the
#   legacy "stale" fit (the fold-souped trial checkpoint's val preds under the
#   FOLD scaler vs ship-scaler LGB preds — mixed inputs, defect M2); the
#   refit-state fit (ship_state + ship_scaler over folds[-1] val rows, both legs
#   under ONE scaler) is still computed and logged side-by-side ("[BLEND] M2"
#   line + blend_diag w_stale/w_refit). ON: the refit-based fit becomes the
#   deployment source. Flip only after one Jetson cycle's side-by-side log and a
#   non-inferior identical-holdout blend DSR (06 plan §4.11).
BLEND_FIT_ON_REFIT = False
# BLEND_THRESHOLD_RESELECT — OFF (default): trade_threshold stays the
#   Optuna-searched value (selected on raw-LSTM fold predictions — defect H2:
#   the served blend is variance-compressed, so the searched threshold implies a
#   different trade frequency than every trial scored); the blend-optimal
#   threshold and old/new n_trades are still logged ("[BLEND] threshold" line).
# ON: trade_threshold is re-selected on the BLENDED folds[-1] val predictions
#   over objective_utils.v3_trade_threshold_range's grid BEFORE certification —
#   certificate and shipped config carry the reselected value (cert == deploy).
#   Same flip evidence as BLEND_FIT_ON_REFIT.
BLEND_THRESHOLD_RESELECT = False

# --- LGB full refit (2026-08 R2-C packet R2C-03, defects M3 + M4-floor) ---
# OFF (default): both LGB legs (mean + q10 tail veto) ship trained only on
#   folds[-1] train — a window ENDING at the 0.85 quantile of the search region
#   — while the LSTM final-refits on ALL purged pre-holdout data (M3: the leg
#   the code calls "the stronger learner at this data size" deploys blind to
#   the newest ~15% of the window every retrain), and the q10 veto floor is
#   calibrated on the same val slice the booster early-stopped on (M4).
# ON: fold training supplies each leg's best_iteration ONLY; both boosters are
#   then RETRAINED on all purged pre-holdout rows (final_refit's purge,
#   NaN-filtered, identical LGB_MAX_ROWS / LGB_X_BYTE_BUDGET most-recent-first
#   cap) at that FIXED round count with no early stopping — the collective-
#   early-stopping analog of the LSTM refit. The veto floor is recomputed as
#   percentile-15 of the REFIT q10's predictions on the original fold-val rows
#   (in-sample for the refit — caveat recorded in lgb_q10_meta.json; a q10
#   holdout-coverage check (10% +/- 3pp) has NO tool yet — scripts/reliability_report.py
#   is the META-calibration Brier/ECE report — so it stays an open Jetson item). Same
#   caveat for the V3 blend-weight fit: its folds[-1]-val anchor rows sit
#   inside the refit train window too, so the LGB leg's fit inputs turn
#   in-sample under this flag — the identical-holdout A/B below is the guard
#   for BOTH (the holdout stays untouched either way). Acts only
#   at save time in train_lgb_ensemble — trial scores untouched, so no study
#   reset of its own; it rides Chain-2's single gotcha-#2 event. Flip only
#   after the Jetson identical-holdout blend-DSR A/B (fold-LGB vs refit-LGB,
#   same LSTM, same cum_trials), then backtest.py --gate, challenger -> shadow
#   (06 plan R2C-03).
LGB_REFIT_FULL = False

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

# --- Trainer seed (2026-08 R2-C packet R2C-04, defect L3) ---
# None (default): legacy fully-unseeded training — torch init/dropout, the
#   np.random.permutation batch shuffles and the Optuna TPESampler all draw
#   from ambient RNG state (two identical retrains produce different
#   artifacts; FR-02's "identical seed" arms and FR-13's seed ensembles are
#   impossible to run as specced).
# int: every RNG consumer in scripts/hypersearch_v2.py is seeded via
#   objective_utils.derive_seed(TRAINER_SEED, study_name, trial.number,
#   fold) — per-fold torch.manual_seed + np.random.default_rng batch
#   permutations in _train_walk_forward, a 'refit' sub-seed for
#   final_refit(seed=), and a 'sampler' sub-seed for the TPESampler.
#   Opt-in plumbing only: it changes no score MATH, but seeding obviously
#   changes which draws occur, so leave None except for reproducibility
#   experiments. Env override: TRADER_TRAINER_SEED (wins over this
#   constant). The determinism check (two refits at the same seed
#   byte-compare) is a Jetson step — 06 plan §4.8.
TRAINER_SEED = None

# --- Training-loop repairs v1 (2026-08 R2-C packet R2C-04, L1+L2+L5+L6) ---
# OFF (default): trial scoring BYTE-IDENTICAL, including four verified
#   defects: (L1) validation loss uses default huber_loss (delta=1.0,
#   unweighted) while training optimizes the trial's huber_delta with
#   |return|+1 weights — checkpoint soup admission, early stopping and the
#   refit epoch budget select on a mismatched criterion; (L2) the
#   regime-penalty "trailing" series is a trailing mean of FORWARD fb-bar
#   returns, so the bull/bear mask at t embeds returns through t+fb and
#   rows 0..48 carry under-scaled partial sums; (L5) the OOM probe batch
#   performs a real unclipped optimizer step before epoch 0; (L6) the
#   embargo is denominated in calendar seconds — ~11 RTH bars instead of
#   seq_len bars on the stock book.
# ON: val loss is computed with the trial's criterion (same delta, same
#   weights); the regime mask uses objective_utils.lagged_regime_series
#   (only returns completed by t; NaN warmup rows excluded from every
#   regime); the OOM probe restores pristine init state + a fresh
#   optimizer/scheduler/grad-scaler after probing; the embargo counts
#   DISTINCT bars via objective_utils.embargo_end_time. ALL FOUR change
#   trial scores and/or fold composition -> old Optuna scores become
#   incomparable: this flag rides Chain-2's single gotcha-#2 study-reset
#   event (flip with HYPERSEARCH_V3/OBJECTIVE_V3, delete both study DBs,
#   reset adaptive best_score + cum_trials) — NEVER a reset of its own.
TRAINING_REPAIRS_V1 = False

# --- Fixed-calendar holdout span (2026-08 R2-C packet R2C-05, FR-01) ---
# None (default): legacy PROPORTIONAL holdout — the final 12% of the pooled
#   calendar span (hypersearch HOLDOUT_FRACTION quantile), byte-identical.
#   Under it the holdout's calendar width scales with the training span
#   (a 1Y window arm gets ~44 days, full history ~200), so cross-arm DSRs
#   are incomparable and the promotion gate's n_eff >= 10 fail-closed floor
#   flunks short windows on gate MECHANICS, not skill — the FR-02/FR-09
#   training-window-experiment blocker.
# float/int days (suggested 60 for crypto): the holdout becomes the fixed
#   trailing span max_time - days*86400 (objective_utils.holdout_boundary);
#   every consumer (folds, refit purge, blend M4 guard, OOF pack, the
#   certificate) inherits through hypersearch's get_holdout_boundary choke
#   point, so successive retrains certify on a consistent-width holdout.
#   Changes fold layout AND what the gate scores -> old Optuna scores become
#   incomparable: activation rides Chain-2's single gotcha-#2 study-reset
#   event, NEVER a reset of its own. Until then the dual-boundary
#   instrumentation line prints on every holdout evaluation (06 plan §4.2:
#   require >= 10 calendar-effective trades at 60d before any window
#   experiment). Env override: TRADER_FIXED_HOLDOUT_DAYS (wins over this
#   constant; also the knob for the one-off "re-run evaluate_on_holdout
#   under both boundaries and report the DSR delta" comparison).
FIXED_HOLDOUT_DAYS = None

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
#   (e) pseudo-CAPE multiplier DELETED repo-wide 2026-08-22 (owner ruling on
#       KILL_LIST ask #3; verbatim code archived in
#       research/campaign_2026-08/08_removed_code.md) and (f) HMM multiplier
#       EXCLUDED (kill-recommended; inverted smoothing documented in
#       regime_detector.py) — HMM still computed and journaled in legacy;
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

# ============================================================================
# INFLUENCE-AUDIT FLAG FAMILY (2026-08 packet IA-4 — the decision-influence
# ledger's split-verdict / behavior-loosening prescriptions made FLIPPABLE).
# Every flag below defaults to today's behavior (flag-OFF byte-identical,
# pinned by tests/test_ia4_flagged.py). Each comment records the ledger
# verdict (research/campaign_2026-08/07_decision_influences.md), the flip
# criterion, and the instrument that decides it. Flip on the Jetson only.
# ============================================================================

# --- VIX>25 non-safe-haven block removal (ledger §3.3, 2-1 "NO"/"suspect") ---
# OFF (default): the block fires exactly as today (journaled as vix25_block
# skip rows since IA-3). ON: the block is skipped entirely — the 25-35 band
# is priced ONCE by the graded VIX tier map (a hard block stacked on a graded
# cut over the same variable double-charges the state; and with no SAFE_HAVEN
# name in the tradable universe the block is a de facto full book halt nobody
# designed — gap C3). Flip criterion: one Jetson read of the vix25_block skip
# rows (decision_report GATE_REASONS) confirming the double-charge and the de
# facto halt cost. Instrument: decision_report.py gate attribution.
VIX25_BLOCK_REMOVED = False

# --- Correlation family merge (ledger §3.3 "MERGE -> ENB budget") ---
# OFF (default): three consumers of one matrix — binary admission gate
# (>0.7), f_corr sizing haircut, ENB stop-risk budget — all fire as today.
# ON: the ENB book stop-risk budget is the SINGLE correlation consumer
# (risk-denominated, continuous, shrink-to-fit); the admission gate loosens
# to a sanity block at avg|corr| > CORR_SANITY_MAX and the f_corr haircut
# composes at 1.0 (in legacy AND v2 tilt). Flip criterion: Jetson
# correlation-skip rows + account_risk journals showing the ENB cap binds
# first (the gate/f_corr only re-charge what the budget already prices).
# Instrument: decision_report 'correlation' skips + the account_risk journal.
CORR_FAMILY_MERGED = False
CORR_SANITY_MAX = 0.85           # loose admission sanity bar when merged

# --- Daily trade budget -> runaway backstop (ledger §3.4 cooldown+budget
# MERGE: "keep cooldown; budget as a runaway backstop it should never
# touch" — two instruments for one churn latent) ---
# OFF (default): MAX_TRADES_PER_SYMBOL_PER_DAY caps as today. ON: the cap
# is multiplied by TRADE_BUDGET_BACKSTOP_MULT, leaving cooldown as the ONE
# churn instrument and the budget as a runaway backstop. Flip criterion:
# Jetson trade_budget veto counts showing the budget fires on names the
# cooldown alone would have throttled (double-charging one jitter latent).
# Instrument: entry_window veto_counts['trade_budget'] + journal_stats churn.
TRADE_BUDGET_BACKSTOP = False
TRADE_BUDGET_BACKSTOP_MULT = 3

# --- Crypto fb-anchored vertical barrier (ledger §3.7: "NO (the absence
# does not belong)" — labels are triple-barrier with an fb vertical; live
# crypto holds are unlimited, so the meta model learns a horizon the
# deployed book doesn't enforce) ---
# OFF (default): no exit fires; the would-fire event is journaled ALWAYS
# (action='vertical_barrier', fired=false) so the flip has evidence from
# day one. ON: a crypto position older than the deployed forward_bars
# horizon (read from the model config, the way the labels read it) whose
# price sits between the barriers gets a max-hold exit with
# exit_reason='vertical' — implemented in the LOOP's position-management
# layer (base_loop._check_vertical_barrier); the policy_exits kernel is
# UNTOUCHED. Flip criterion: Jetson vertical_barrier would-fire rows
# showing stale holds with flat/negative drift past fb bars. Instrument:
# the vertical_barrier journal rows + journal_stats hold-time distribution.
CRYPTO_VERTICAL_BARRIER = False

# --- Signal-exit confirmation reads (ledger §3.7 KEEP-COND: "reconcile the
# 1-vs-2-reading asymmetry deliberately" — stops need 2 consecutive
# readings, the signal exit fires on 1; rev-07-02 conflict #2) ---
# 1 (default): today's behavior — the signal exit fires on a single
# reading (the sell row itself is the first-reading record). 2: the signal
# exit requires two CONSECUTIVE readings (pred < -threshold twice),
# matching stop-confirmation discipline; armed/lapsed first readings are
# journaled (action='signal_exit_reading') and confirmed sells carry
# signal_exit_readings=2. Applies to the pred-based signal exit only —
# the stock rank-drop exit keeps its own hysteresis band. Flip criterion:
# Jetson signal_sell outcomes showing sub-hour flip-sells that round-trip
# fees (see also cooldown_bypassed_exit markers). Instrument:
# llm_eval/journal_stats signal_sell outcome attribution.
SIGNAL_EXIT_CONFIRM_READS = 1

# --- Kelly sample gate (ledger §3.5 kelly_mult KEEP-COND: "poisoned sample
# (D06 winner-censoring — recovery in tree, history still biased to the
# floor exactly when winning). Hold neutral until ~50 uncensored
# trades/book rebuild") ---
# OFF (default): kelly_mult computes from trade_memory as today (the D06
# recovery writes confirmed rows going forward, but the historical sample
# is still winner-censored). ON: kelly_mult holds neutral 1.0 until the
# book has >= KELLY_SAMPLE_MIN_TRADES uncensored (non-estimated) trades
# recorded on/after KELLY_SAMPLE_SINCE (set this to the Jetson deploy date
# of the D06-fix wave — rows before it are the censored history). Gate
# state is journaled in the sizing detail ('kelly_gate') while ON.
# Flip criterion: immediate on deploy (the gate IS the repair; it releases
# itself once the clean sample accumulates). Instrument:
# trading_utils.uncensored_trade_count via the sizing journal.
KELLY_SAMPLE_GATE = False
KELLY_SAMPLE_MIN_TRADES = 50
KELLY_SAMPLE_SINCE = '2026-08-22'   # ISO date; rows with ts >= this count

# --- Per-book circuit-breaker baseline (ledger §3.2 KEEP-COND: "the
# account-wide scope cross-contaminates books and crypto weekends are
# judged against Friday's close. Per-book (or weekend-aware) baselines +
# trip journaling") ---
# OFF (default): account-wide Alpaca last_equity baseline exactly as today
# (trip journaling landed in IA-3). ON: each book measures ITS OWN P&L
# (realized exits + open-position mark drift, accumulated in-loop) against
# a baseline equity captured at the book's own window roll — crypto rolls
# at UTC midnight EVERY day (weekend-aware), stock at the ~16:05 ET
# baseline reset — and halts until its own window end on a trip. Flip
# criterion: circuit_breaker_trip journal rows (IA-3) showing cross-book
# contamination or weekend-stale baselines caused/missed trips.
# Instrument: the circuit_breaker_trip event rows.
BREAKER_PER_BOOK = False

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
