# FLAGS.md — every flag, policy constant and environment variable

*Single home for the flag inventory (`docs/README.md` reading order rule: other docs point here,
they do not repeat these numbers). Code-verified 2026-09-08 against the working tree at `master`
+ the in-flight uncommitted diff; line numbers are a snapshot of that date, names are stable.*
*2026-09-26 Jetson pass: the `TRADER_INDICATORS_C` / `TRADER_PYBIN` rows, the `CUDA_VISIBLE_DEVICES`,
`TORCH_NUM_THREADS`, `OMP_NUM_THREADS`, `LD_LIBRARY_PATH`, `SUDO_USER` rows, the `TRADER_SHADOW_MODE`
cites and the `setup_jetson_system.sh` cites were re-verified against the working tree; other line
cites were not re-swept (that day's edits shifted `run_pipeline.py` by +19…+24 lines below `:288`).*
*2026-09-27 cite sweep (INTEL W20): every `file:NNN` cite was re-verified against the working
tree; stale read-site cites became durable anchors (a file plus a backticked function or
constant name, e.g. volatility.py `get_sigma()`), and the Defined / First-read-at / §6 Where
columns + §4 section ranges keep line numbers (per the §7 recipe), regenerated that day. Line
numbers drift with every edit; the anchor names do not.*

## Philosophy — read this before flipping anything

1. **Default-OFF is the contract.** Every model-facing or gate-behavior change from the 2026-08
   campaign onward ships behind a flag whose default reproduces today's behavior exactly. 82
   distinct flags exist: **69 at an OFF/neutral default, 12 default-ON, 1 a neutral mode** (the
   `indicator_config.json` feature preset). CLAUDE.md's "~32 new default-OFF flags" counts the
   2026-08 campaign wave only.
2. **OFF paths are byte-pinned, not just value-pinned.** A "byte-pin" test asserts the flag-OFF
   code path is byte-identical to the pre-flag code (e.g. `tests/test_c26_S3.py` §E, the verbatim
   legacy-expression pins in `tests/test_r2c_training_repairs.py`), not merely that the default
   literal is unchanged. Where the only guard is the literal, this doc says "(value pin)".
3. **Facing** — `model` changes predictions/labels/features/trained artifacts · `gate` changes
   admission, sizing, exits or the promotion verdict without a retrain · `measurement` touches only
   journals/reports/offline kernels · `ops` is process/infra.
4. **Promotion path.** `model`- and `gate`-facing flips ship only through
   **challenger → shadow → DM-HLN** (`shadow.dm_hln`, `shadow.py:310`; promotion decided at
   `shadow.evaluate_and_maybe_promote`). Measurement-only flags ship directly.
5. **Gotcha #2 applies to every `model` flip**: the first Jetson retrain after an objective/feature
   change must delete `v2_study.db` + `stock_v2_study.db` and reset the adaptive `best_score` —
   old Optuna scores are incomparable.
6. **Flip criteria and evidence gates live in the code comment above each flag**, not here — see
   `strategy_config.py:<line>` in the Defined column, and `research/campaign_2026-08/03_jetson_runbook.md`
   for the activation sequence. Restating them here would guarantee drift.

## Env-var parsing — two incompatible rules, know which you are using

| Rule | Accepts | Used by |
|---|---|---|
| **strict `'1'`** — `os.environ.get(NAME) != '1'` | `1` only | `TRADER_ORDER_STREAM` (`order_stream.py:97`), `TRADER_USE_ALPACA_PY` (`trading_utils.py:88`), `TRADER_INDICATORS_C` (`indicators.py:15`, `== '1'`; added 2026-09-26) |
| **2026-08 family** — `os.getenv(NAME,'0').strip().lower() in ('1','true','yes')` | `1`, `true`, `yes` (case/whitespace-insensitive) | all 16 c26/R2C dark flags: `base_loop.py` `STOP_CLASSIFY_V2`, `STREAM_STOP_DETECT`; `order_utils.py` `IOC_ENTRY_CAP_ENABLED`, `MAKER_SHARE_NOTIONAL_ENABLED`; `shadow.py` `DM_V2_ENABLED`; `funding.py` `FUNDING_Z_TIME_THINNING`; `cost_regime.py` `COST_REGIME_FEATURES`; `liquidity.py` `SPREAD_FILL_V2`, `CRYPTO_SPREAD_STAMP`, `STOCK_MINUTE_EDGE`, `IMPACT_VOLSCALE`; `market_data.py` `closed_bars_v2_enabled()`, `daily_feature_restore_enabled()`, `har_daily_feed_enabled()`; `data_sources.py` `_yf_window_slice_enabled()`; `data_utils.py` `raw_sidecar_enabled()` |
| **default-ON negation** — `os.environ.get(NAME,'1').strip().lower() not in ('0','false','no')` | `0`, `false`, `no` disable | `TRADER_BOTS_OPS` (`run_bots.py` `_OPS_ENABLED`), `TRADER_EOD_DIGEST` (`run_pipeline.py` `EOD_DIGEST_ENABLED`) |
| **literal `'0'` only** — `os.getenv(NAME,'1') != '0'` | `0` disables; anything else is ON | `TRADER_SHADOW_MODE` (`run_pipeline.py` `_build_training_phases()`) |
| **int with guard** | `try: int(...) except ValueError: <fallback>` | `TRADER_EOD_DIGEST_HOUR` (→16), `TRADER_JOURNAL_ROTATE_DAYS` (→0). **`TRADER_MINUTE_EDGE_DAYS` (`scripts/harvest_stock_data.py` `MINUTE_EDGE_DAYS`) has NO guard — a non-numeric value crashes at import.** |

> **`TRADER_ORDER_STREAM=true` is a silent no-op.** So is `TRADER_USE_ALPACA_PY=true`, `=yes`, or
> `=1 ` with trailing whitespace. Those two flags compare against the literal `'1'` with no
> `.strip().lower()`, unlike every flag added in 2026-08. Nothing warns. Use `=1` exactly.

---

## 1. Master flag table

Legend — **Status**: `LIVE` = a production reader exists and a flip changes behavior ·
`LIVE·inert until X` = a reader exists but a flip alone does nothing until X is also on ·
`DECLARED-AHEAD` = no production reader; the kernel is staged ahead of its wiring (§2) ·
`DEAD-zero-refs` = no reference anywhere, tests included.
`gate†` = the facing it *would* have once wired. Env flags appear here as the **module attribute**
tests monkeypatch; their env-var spelling and parsing rule is in §5.

#### strategy_config.py (47)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `HAR_VOL_ENABLED` | `strategy_config.py:64` | `True` | bool | gate | volatility.get_sigma uses HAR-RV sigma with GARCH fallback (volatility.py `get_sigma()`); False force… | `volatility.py` | **none** | LIVE·inert until TRADER_HAR_DAILY_FEED=1 |
| `CONVICTION_JOURNAL_ENABLED` | `strategy_config.py:66` | `True` | bool | measurement | per-candidate veto attribution + sizing detail written to the decision journal (base_loop.py `_conviction_journal_on()`)… | `base_loop.py`, `decision_report.py` | `test_conviction_journal.py`, `test_decision_report_v3.py` | LIVE |
| `MAKER_ENTRIES_ENABLED` | `strategy_config.py:80` | `True` | bool | gate | crypto passive bid-join entry ladder (crypto_loop.py:101-102); False = taker only | `crypto_loop.py` | `test_execution_policy_v3.py` | LIVE |
| `ENTRY_WINDOWS_ENABLED` | `strategy_config.py:117` | `True` | bool | gate | entry-window mask applied (backtest.py `_entry_window_mask()`, stock_loop.py:223); False = all RTH hours | `stock_loop.py`, `backtest.py` | **none** | LIVE |
| `OVERNIGHT_SLEEVE_ENABLED` | `strategy_config.py:121` | `True` | bool | gate | keep up to N stock positions overnight instead of EOD-flattening everything | `stock_loop.py` | **none** | LIVE |
| `EVENTS_TRADING_DAY_WINDOWS` | `strategy_config.py:132` | `False` | bool | gate | earnings buffers walk TRADING days (weekend/holiday aware) instead of calendar days (events_cal… | `events_calendar.py` | `test_c26_P4.py` | LIVE |
| `IMPACT_COST_ENABLED` | `strategy_config.py:142` | `False` | bool | gate | adds a sqrt(notional/ADV) impact haircut to OFFLINE backtest/meta net P&L (liquidity.py `impact_inputs_from_df()`) | `liquidity.py`, `backtest.py` | `test_c26_T3.py`, `test_market_impact.py`, `test_improve_stratcfg.py`, `test_review_b10.py` | LIVE·inert until a DV30 column |
| `UNIQUENESS_WEIGHTS_ENABLED` | `strategy_config.py:153` | `False` | bool | model | average-uniqueness sample weights in the LSTM loss (scripts/hypersearch_v2.py `train_lgb_ensemble()`) | `scripts/hypersearch_v2.py` | `test_improve_stratcfg.py` | LIVE |
| `PROMOTION_GATE_V2` | `strategy_config.py:167` | `False` | bool | gate | calendar-concurrency n_eff replaces per-ticker uniqueness + cluster count, n_eff<10 fails CLOSE… | `backtest.py`, `run_pipeline.py`, `scripts/hypersearch_v2.py` +1 | `test_c26_Q1.py`, `test_c26_X1.py` | LIVE |
| `KISH_NEFF_ENABLED` | `strategy_config.py:172` | `False` | bool | gate | Kish design-effect softening of the calendar concurrency (backtest.py `aggregate_metrics()`); read only when PROM… | `backtest.py`, `scripts/hypersearch_v2.py` | `test_c26_Q1.py` | LIVE·inert until PROMOTION_GATE_V2 |
| `GATE_TARGETS_CHALLENGER` | `strategy_config.py:187` | `False` | bool | gate | weekly policy gate replays the CHALLENGER artifacts instead of the champion (run_pipeline.py `_build_training_phases()`)… | `run_pipeline.py`, `shadow.py`, `backtest.py` | `test_c26_Q2.py`, `test_c26_R1.py` | LIVE |
| `OBJECTIVE_LONG_ONLY` | `strategy_config.py:202` | `True` (flipped 2026-09-27 at the clean-rebuild gotcha-#2 event; was `False`) | bool | model | hypersearch simulate_trades scores ONLY the deployable long leg (scripts/hypersearch_v2.py `_objective_long_only()`)… | `scripts/hypersearch_v2.py` | `test_improve_stratcfg.py` | LIVE |
| `HYPERSEARCH_V3` | `strategy_config.py:231` | `True` (flipped 2026-09-27, founder; Phase-3 bundle) | bool | model | final refit on all pre-holdout data + pre-gate LGB + blend-weight fit + blended holdout certifi… | `scripts/hypersearch_v2.py` | `test_c26_T1.py`, `test_r2c_blend_coherence.py`, `test_r2c_lgb_refit.py` | LIVE |
| `BLEND_FIT_ON_REFIT` | `strategy_config.py:244` | `False` | bool | model | deployed lstm_weight comes from the refit-state fit instead of the stale fold-scaler fit (scrip… | `scripts/hypersearch_v2.py` | `test_r2c_blend_coherence.py` | LIVE·inert until HYPERSEARCH_V3 |
| `BLEND_THRESHOLD_RESELECT` | `strategy_config.py:254` | `False` | bool | model | trade_threshold re-selected on BLENDED val predictions before certification (scripts/hypersearc… | `scripts/hypersearch_v2.py`, `blend_fit.py (kernel reselect_trade_threshold)` | `test_r2c_blend_coherence.py` | LIVE·inert until HYPERSEARCH_V3 |
| `LGB_REFIT_FULL` | `strategy_config.py:281` | `False` | bool | model | both LGB legs retrain on all purged pre-holdout rows at a fixed round count; q10 veto floor rec… | `scripts/hypersearch_v2.py`, `objective_utils.py (lgb_refit_indices kernel)` | `test_r2c_lgb_refit.py` | LIVE |
| `OBJECTIVE_V3` | `strategy_config.py:296` | `True` (flipped 2026-09-27, founder; Phase-3 bundle) | bool | model | ticker-block position reset + edge-anchored trade_threshold range + holdout-crossing val purge … | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_c26_T1.py`, `test_r2c_blend_coherence.py`, `test_r2c_holdout_boundary.py` | LIVE |
| `TRAINER_SEED` | `strategy_config.py:314` | `None` | int? | model | seeds torch init/dropout, batch permutations and the TPESampler via objective_utils.derive_seed… | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_r2c_training_repairs.py` | LIVE |
| `TRAINING_REPAIRS_V1` | `strategy_config.py:338` | `True` (flipped 2026-09-27, founder; Phase-3 bundle) | bool | model | L1 val-loss criterion match + L2 lagged regime mask + L5 pristine OOM probe + L6 bar-denominate… | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_r2c_training_repairs.py` | LIVE |
| `HOLDOUT_SPAN_BY_TARGET` | `strategy_config.py:351` | `False` | bool | gate | holdout n_eff spans raw-target trades over their real fb-bar window instead of the TB_Bars exit span (scripts/hypersearch_v2.py evaluate_on_holdout, SIG-R1-A3); env `TRADER_HOLDOUT_SPAN_BY_TARGET` wins | `scripts/hypersearch_v2.py` | `test_sig_r1_a3.py` | LIVE |
| `BARS_PER_YEAR_MEASURED` | `strategy_config.py:369` | `False` | bool | model | `bars_calendar.bars_per_year()`/`bars_per_day()` return the census-measured stock calendar (3827 bars/yr = 3827/252 bars/day, extended session) instead of 1638/6.5 to `hypersearch_v2.compute_sharpe`, `portfolio_backtest` AND (ENGINE-R2) `volatility.py`'s live sizing (compute_vol_adjusted_size per-bar target; HAR daily→per-bar sigma — HAR-sourced vol ratio is calendar-invariant, GARCH-sourced ×√(1638/3827)); crypto unchanged; no env var | `bars_calendar.py`, `scripts/hypersearch_v2.py`, `portfolio_backtest.py`, `volatility.py` | `test_sig_r1_bpy.py`, `test_engine_r2_flags.py` | LIVE (flip only in a study-reset event) |
| `FAILED_TRIAL_PRUNE` | `strategy_config.py:388` | `False` | bool | model | a trial with no honest score (OOM/RuntimeError, no folds, zero completed folds, fold timeout before its first checkpoint) raises optuna.TrialPruned (PRUNED, user_attr `failed_trial`) instead of the 0.0 COMPLETE sentinel that outranked every negative trial and inflated the COMPLETE-only pool (scripts/hypersearch_v2.py:603 `_failed_trial_prune`, SIG-R2-1); read per trial at scoring time, so a recorded study keeps its stored COMPLETE/best_trial history; env `TRADER_FAILED_TRIAL_PRUNE` wins | `scripts/hypersearch_v2.py` | `test_sig_r2_1.py` | LIVE (flip at a study reset) |
| `OBJECTIVE_SESSION_MASK` | `strategy_config.py:406` | `False` | bool | model | stock LONG entries scored only where the live book may enter (bar open-time in `STOCK_ENTRY_WINDOWS_ET` on a weekday; RTH if `ENTRY_WINDOWS_ENABLED` is False) in every trainer scorer — pruning, folds, regime Sharpes, holdout certificate, threshold reselect (scripts/hypersearch_v2.py:659 `_objective_session_mask` → `_session_entry_ok` → `objective_utils.session_entry_mask`, SIG-R2-MASK); crypto never masked; env `TRADER_OBJECTIVE_SESSION_MASK` wins | `scripts/hypersearch_v2.py`, `objective_utils.py (session_entry_mask kernel)`, `scripts/session_mask_holdout_ab.py` | `test_sig_r2_mask.py` | LIVE (flip only in a study-reset event; evidence = `session_mask_holdout_ab.py` verdict) |
| `WICK_PRINT_FILTER` | `strategy_config.py:423` | `False` | bool | model | crypto harvest repairs (never drops) wick-only bad prints — Low/High ≥15 % beyond Open on a bar with \|C−O\|/O < 3 % — to min/max(O,C) ∓ the PIT 24-bar median true range before features + TB labels (data_utils.py:53 `wick_print_filter_enabled` / `repair_wick_prints`, called at scripts/harvest_crypto_data.py:155, SIG-R2-X6); raw sidecar keeps the unrepaired bars; env `TRADER_WICK_PRINT_FILTER` wins; precondition: ENGINE live-bar parity repair at market_data.py:244 | `data_utils.py`, `scripts/harvest_crypto_data.py` | `test_sig_r2_x6_wick_filter.py` | LIVE (flip = re-harvest + study reset, after serving parity) |
| `CRYPTO_QUOTE_MAX_AGE_SEC` | `strategy_config.py:441` | `None` | float? | gate | crypto quote-staleness limit in `order_utils.get_quote` (via `_quote_max_age_sec`, read at call time): None = legacy 180 s; a number = that many seconds (invalid → 180); stocks stay 180 s. A stale None skips every software exit for the symbol that cycle (base_loop._manage_stops) and blocks entries (ENGINE-R2 O8); no env var | `order_utils.py` | `test_engine_r2_flags.py` | LIVE (flip only via the owner flip proposal: ≥24 h weekday+weekend staleness census; move `crypto_quote_staleness_census.LIVE_MAX_AGE_S` in the same change) |
| `HALT_CANCELS_WORKING_BUYS` | `strategy_config.py:457` | `False` | bool | gate | on the first halted `base_loop._entries_allowed` call, cancel this book's open BUY orders (universe symbols, side buy, non-stop types) once per halt epoch via `_halt_cancel_working_buys`; un-halting re-arms; SELL/stop orders never touched. OFF = halt blocks new entries only (a working buy — unconfirmed-cancel maker rung, lifecycle give-up, stock day parent — can still fill); read at call time; no env var | `base_loop.py` | `test_engine_r2_strikes_halt.py` | LIVE (flip only via the owner flip proposal, ENGINE W9: buy fills observed while `trading_halt.flag` was active) |
| `BREAKER_SERVER_FILL_ATTRIB` | `strategy_config.py:481` | `False` | bool | gate | after a circuit-breaker flatten (`base_loop._circuit_breaker_check`, which runs BEFORE `_manage_stops`) — and, ENGINE-R4 R4-b, after a remote `/flatten` (`_check_flatten_request`, first in the cycle; `detect_source='remote_flatten'`) or a stablecoin flatten (`_update_macro_regime`; `detect_source='stablecoin_flatten'`, whose OFF path now writes estimated `stablecoin_flatten` rows) — each released position with a `stop_order_id` gets one `get_order`; `'filled'` → journaled via the `_manage_stops` server-fill calls (`server_stop` sell row with the real fill, `detect_source='breaker'`, `last_trade_time`, 24 h lockout) instead of an estimated `circuit_breaker` row at the quote mid; anything else keeps the estimated row. OFF = no extra broker call. Moves real fills into Kelly's sample and arms the lockout (ENGINE-R3 O3); env `TRADER_BREAKER_SERVER_FILL_ATTRIB` wins when set; read at call time | `base_loop.py` | `test_engine_r3_o3_quote.py`, `test_engine_r2_replay_harness.py`, `test_engine_r4_journal_flatten.py` | LIVE (flip only via the owner flip proposal, ENGINE W10: ≥1 live `circuit_breaker` exit row whose stop order Alpaca reports `filled`) |
| `RESTART_STOP_ANCHOR_DESIRED` | `strategy_config.py:508` | `False` | bool | gate | after a restart, `crypto_loop`/`stock_loop._replace_protective_stops` place the server stop at exactly `base_loop._desired_stop_for(pos)[0]` (stocks rounded to cents; crypto `_resting_stop_px` records it, so cycle 1 does not cancel/re-place) instead of the legacy `max(entry, hwm) * (1 - hard stop_dist)`; stocks keep a PLAIN stop even with a restored `trailing_activated=True` (J2). OFF = legacy, byte-identical. Moves the resting stop LEVEL for every position with hwm > entry: looser (down to the entry-anchored hard stop) while the trail is not armed, tighter-or-equal once it is (ENGINE-R3 O2+J11); inherits `_desired_stop_for`'s trail denominator; read at call time; no env var | `crypto_loop.py`, `stock_loop.py` | `test_engine_r3_restart_anchor.py` | LIVE (flip only via the owner flip proposal, ENGINE W11: accept restart server stop == software stop, before the first restart with open positions) |
| `FIXED_HOLDOUT_DAYS` | `strategy_config.py:531` | `None` | int? | model | holdout becomes a fixed trailing calendar span instead of the 12% quantile (scripts/hypersearch… | `scripts/hypersearch_v2.py`, `objective_utils.py (holdout_boundary)`, `scripts/window_ab.py` | `test_r2c_holdout_boundary.py` | LIVE |
| `META_CALIBRATION_MODE` | `strategy_config.py:540` | `legacy` | str | model | 'legacy' same-slice isotonic vs 'purged_oof' leak-free calibration of the meta probability | `meta_label.py`, `calibration.py` | `test_improve_stratcfg.py` | LIVE |
| `CALIBRATION_V2` | `strategy_config.py:559` | `False` | bool | model | tie-pooled PAVA + logit-Platt MAP smoothing + OOF embargo + size-aware chooser (calibration.py:… | `calibration.py`, `meta_label.py` | `test_c26_R1.py` | LIVE |
| `META_OOF_PRED` | `strategy_config.py:582` | `False` | bool | model | consumes {prefix}oof_preds.npz for the meta 'pred' feature + entry filter (meta_label.py:854) | `meta_label.py`, `scripts/meta_learning_curve.py` | `test_c26_R2.py` | LIVE |
| `META_REPLAY_POLICY_PARITY` | `strategy_config.py:597` | `False` | bool | model | meta replay applies the deployed admission conditions (cost floor, cooldown/lockout, entry wind… | `meta_label.py`, `scripts/meta_learning_curve.py` | `test_c26_R2.py` | LIVE |
| `DERISK_STACK_V2` | `strategy_config.py:632` | `False` | bool | gate | regime family aggregated by MIN, one VIX tier map, modal regime = 1.0, crypto BTC-RV replaces V… | `base_loop.py`, `portfolio.py` | `test_c26_S3.py` | LIVE |
| `VIX25_BLOCK_REMOVED` | `strategy_config.py:652` | `False` | bool | gate | skips the VIX>25 non-safe-haven entry block (base_loop.py / stock_loop.py `_execute_buys()`) | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `CORR_FAMILY_MERGED` | `strategy_config.py:664` | `False` | bool | gate | ENB stop-risk budget becomes the single correlation consumer; admission loosens to CORR_SANITY_… | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `TRADE_BUDGET_BACKSTOP` | `strategy_config.py:676` | `False` | bool | gate | multiplies MAX_TRADES_PER_SYMBOL_PER_DAY by the backstop mult, leaving cooldown as the one chur… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `CRYPTO_VERTICAL_BARRIER` | `strategy_config.py:693` | `False` | bool | gate | fb-anchored max-hold exit for crypto positions in the LOOP layer (base_loop.py `_check_vertical_barrier()`); would-fir… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `SIGNAL_EXIT_CONFIRM_READS` | `strategy_config.py:708` | `1` | int | gate | 1 = signal exit fires on one reading; 2 = requires two consecutive readings (base_loop.py `_execute_sells()`)… | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `KELLY_SAMPLE_GATE` | `strategy_config.py:724` | `False` | bool | gate | holds kelly_mult neutral at 1.0 until the book has enough uncensored post-D06 trades (base_loop… | `base_loop.py`, `trading_utils.py (uncensored_trade_count substrate)` | `test_ia4_flagged.py` | LIVE |
| `BREAKER_PER_BOOK` | `strategy_config.py:741` | `False` | bool | gate | per-book P&L circuit-breaker baseline + window roll instead of the account-wide Alpaca last_equ… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `EDGE_KELLY_ENABLED` | `strategy_config.py:771` | `False` | bool | gate† | would replace clip(2p,0.6,1.3) with bet_sizing.afml_bet_size / kelly_edge_odds | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CRYPTO_CS_RANK_ENABLED` | `strategy_config.py:779` | `False` | bool | gate† | would apply a soft [0.90,1.10] crypto cross-sectional rank size tilt | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CRYPTO_TREND_GATE_ENABLED` | `strategy_config.py:790` | `False` | bool | gate† | would compose a graded BTC-200h-SMA de-risk scalar into CryptoLoop._extra_tilt | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONCENTRATION_ENABLED` | `strategy_config.py:803` | `False` | bool | gate† | would enable conviction-gated dynamic top-K admission | **none** | `test_improve_stratcfg.py`, `test_conviction_ab.py` | DECLARED-AHEAD |
| `CONVICTION_SIGNAL_FLOOR` | `strategy_config.py:806` | `None` | float? | gate† | None = signal floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONVICTION_META_FLOOR` | `strategy_config.py:807` | `None` | float? | gate† | None = meta-probability floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONVICTION_RATIO_FLOOR` | `strategy_config.py:808` | `None` | float? | gate† | None = edge/cost ratio floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `TIER_SIZING_ENABLED` | `strategy_config.py:809` | `False` | bool | gate† | would concentrate the top tier by edge | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `PREDICTION_CACHE_ENABLED` | `strategy_config.py:818` | `False` | bool | gate | bar-keyed memo of the feature+LSTM+LGB inference result (predict_now.py `get_live_prediction()`) | `predict_now.py`, `prediction_cache.py` | `test_improve_stratcfg.py` | LIVE |

#### indicator_config.py / stock_config.py (3)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `HURST_ON_RETURNS` | `indicator_config.py:33` | `False` | bool | model | Hurst R/S computed on RETURNS instead of price LEVELS (indicators.py `compute_features()`) | `indicators.py` | `test_ia1_removals.py`, `test_live_feature_parity.py` | LIVE |
| `TRADABLE_POOL_ENABLED` | `stock_config.py:194` | `False` | bool | gate† | would promote candidate-pool names into the live selectable set | **none** | `test_grp_ops.py` | DECLARED-AHEAD |
| `indicator_config.json:preset` | `indicator_config.py:22` | `standard` | str | model | which feature columns scripts/hypersearch_v2.py trains on (get_preset_features) | `scripts/hypersearch_v2.py`, `gui.py`, `indicator_leadlag.py` | `test_indicator_config.py` | LIVE |

#### llm_config.json keys (12)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `llm_config.json:enabled` | `llm_config.py:196` | `True` | bool | gate | master on/off for the whole LLM stack | `llm_client.py` | **none** | LIVE |
| `llm_config.json:provider` | `llm_config.py:196` | `auto` | str | gate | legacy single-provider pin; consulted only when selection_mode=='single' | `llm_client.py` | `test_llm_config.py` | LIVE |
| `llm_config.json:selection_mode` | `llm_config.py:196` | `auto` | str | gate | provider-chain builder: auto \| single \| free-only \| best-free | `llm_client.py (resolve_provider_chain)` | `test_llm_routing.py` | LIVE |
| `llm_config.json:analyst_model_override` | `llm_config.py:196` | `null` | str? | gate | pin the analyst model (None = smart routing) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:sentiment_model_override` | `llm_config.py:196` | `null` | str? | gate | pin the sentiment model (None = smart routing) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:anthropic_cache_system_ttl` | `llm_config.py:196` | `"" (off)` | str | gate | Anthropic prompt-cache breakpoint on the static system prompt (llm_client.py `_call_anthropic()`); "" \| … | `llm_client.py` | `test_c26_S2.py` | LIVE |
| `llm_config.json:tier_override` | `llm_config.py:196` | `null` | str? | ops | manual Gemini tier override ('free'\|'paid'\|None) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:journal_enabled` | `llm_config.py:196` | `True` | bool | measurement | LLM call journaling | `trade_journal.py`, `llm_client.py` | **none** | LIVE |
| `llm_config.json:rich_context_enabled` | `llm_config.py:196` | `False` | bool | gate | attach the compact quant evidence block to live LLM candidates (llm_analyst.py `rich_context_enabled()`) | `llm_analyst.py`, `base_loop.py` / `stock_loop.py` `_build_llm_candidates()` | `test_llm_advice.py` | LIVE |
| `llm_config.json:replay_capture_enabled` | `llm_config.py:196` | `True` | bool | measurement | journal full candidate cycles to journals/llm_replay/ (llm_analyst.py `_journal_replay()`) | `llm_analyst.py` | `test_llm_dossier_persist.py` | LIVE |
| `llm_config.json:advisor_v2_enabled` | `llm_config.py:196` | `False` | bool | measurement | structured decision dossier: extended prompt/schema/shadow journal (llm_analyst.py `analyze_trades()`) | `llm_analyst.py`, `llm_eval.py` | `test_llm_dossier_persist.py` | LIVE |
| `llm_config.json:analyst_dedup_ttl_sec` | `llm_config.py:196` | `0` | int | gate | evidence-hash call-dedup TTL for analyze_trades (llm_analyst.py `analyze_trades()`); clamped to [0,7000] | `llm_analyst.py` | **none** | LIVE |

#### env-flag module attributes (28)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `base_loop.STOP_CLASSIFY_V2` | `base_loop.py:66` | `False` | bool | gate | 24h re-entry lockout only for server-stop fills classified 'hard'/'unknown'; 'trail' exempt (ba… | `base_loop.py` | `test_c26_T6.py` | LIVE |
| `base_loop.STREAM_STOP_DETECT` | `base_loop.py:72` | `False` | bool | gate | cached order_stream fill recovers a stop when the REST probe raises (base_loop.py `_stream_stop_fallback()`) | `base_loop.py` | `test_c26_T6.py` | LIVE |
| `order_utils.IOC_ENTRY_CAP_ENABLED` | `order_utils.py:24` | `False` | bool | gate | entry-order market fallbacks become slippage-capped marketable IOCs (order_utils.py `manage_order_lifecycle()`) | `order_utils.py`, `base_loop.py` `_execute_entry_order()` | `test_c26_T6.py` | LIVE |
| `order_utils.MAKER_SHARE_NOTIONAL_ENABLED` | `order_utils.py:28` | `False` | bool | gate | should_trade's live crypto threshold uses the notional-weighted maker share (order_utils.py `should_trade()`) | `order_utils.py` | `test_c26_T6.py` | LIVE |
| `shadow.DM_V2_ENABLED` | `shadow.py:85` | `False` | bool | gate | DM v2 (two-look Lan-DeMets alpha budget, IM blocks) DECIDES promotion instead of legacy (shadow… | `shadow.py` | `test_c26_Q3.py` | LIVE |
| `funding.FUNDING_Z_TIME_THINNING` | `funding.py:42` | `False` | bool | model | time-thinned funding history append + archive-preferred z baseline (funding.py `get_funding_rate()`, `funding_tilt()`) | `funding.py` | `test_c26_P5.py` | LIVE |
| `cost_regime.COST_REGIME_FEATURES` | `cost_regime.py:42` | `False` | bool | model | B21 cost-regime meta features written into the harvest store (cost_regime.py `stamp_cost_regime_features()`) | `cost_regime.py`, `scripts/harvest_stock_data.py` `prepare_stock_data()`, `scripts/harvest_crypto_data.py` `prepare_data()` | `test_c26_T3.py` | LIVE |
| `liquidity.SPREAD_FILL_V2` | `liquidity.py:69` | `False` | bool | model | no-estimate bars are stamped with the flat FLAT_SPREAD_PCT instead of the floor, +inf clipped t… | `liquidity.py` | `test_c26_T3.py` | LIVE |
| `liquidity.CRYPTO_SPREAD_STAMP` | `liquidity.py:70` | `False` | bool | model | per-pair crypto spread tier stamp replaces the flat 0.10% (liquidity.py `stamp_crypto_spreads()`) | `liquidity.py`, `scripts/harvest_crypto_data.py` `prepare_data()` | `test_c26_T3.py` | LIVE |
| `liquidity.STOCK_MINUTE_EDGE` | `liquidity.py:71` | `False` | bool | model | per-day EDGE spread computed from 1-min bars during the stock harvest (scripts/harvest_stock_da… | `liquidity.py`, `scripts/harvest_stock_data.py` | `test_c26_T3.py` | LIVE |
| `liquidity.IMPACT_VOLSCALE` | `liquidity.py:72` | `False` | bool | gate | empirical vol-scale re-base of the sqrt market-impact model (liquidity.py `impact_inputs_from_df()`) | `liquidity.py` | `test_c26_T3.py` | LIVE |
| `liquidity.CRYPTO_CENSUS_FILE` | `liquidity.py:253` | `crypto_spread_census.json` | str | ops | path of the crypto spread-tier census consumed in liquidity.py `_load_census()` | `liquidity.py` | `test_c26_T3.py (attribute)` | LIVE |
| `trade_journal.JOURNAL_ROTATE_DAYS` | `trade_journal.py:68` | `0` | int | ops | gzip day-files older than N days (trade_journal.py `log_decision()`, `rotate_old_journals()`) | `trade_journal.py` | `test_c26_T7.py` | LIVE |
| `run_bots._OPS_ENABLED` | `run_bots.py:64` | `True` | bool | ops | ops thread in standalone run_bots.py: Telegram kill-switch polling + daily drift check (run_bot… | `run_bots.py` | **none** | LIVE |
| `run_pipeline.EOD_DIGEST_ENABLED` | `run_pipeline.py:144` | `True` | bool | ops | once-daily EOD digest notification (run_pipeline.py `_maybe_send_eod_digest()`) | `run_pipeline.py` | `test_c26_T7.py (attribute)` | LIVE |
| `run_pipeline.EOD_DIGEST_HOUR` | `run_pipeline.py:146` | `16` | int | ops | local hour after which the EOD digest may fire (run_pipeline.py `_maybe_send_eod_digest()`) | `run_pipeline.py` | `test_c26_T7.py (attribute)` | LIVE |
| `run_pipeline.BOT_RESTART_BACKOFF` (env `TRADER_BOT_RESTART_BACKOFF`, 2026-08-family parse, same ops-constant pattern as `EOD_DIGEST_ENABLED` `run_pipeline.py:144`) | `run_pipeline.py:1074` | `False` | bool | ops | crash-restart backoff in `_check_restart_bots`: k-th identical crash (innermost-frame signature from the bot log tail) waits 60·2^k s capped 960 s; 5 identical crashes in 60 min latch the bot down with ONE critical notify until `start_bot` (`_start_bots_now`) or the weekly `_restart_bots`. Acts on exited processes only; never touches orders. Acceptance (April replay, `tests/fixtures/april_crashloop_2026.json`): 1,616 → 12 restarts (99.3 % avoided), 3 critical alerts | `run_pipeline.py` | `test_engine_r5_restart_backoff.py` (OFF path sha-pinned) | LIVE |
| `scripts/harvest_stock_data.MINUTE_EDGE_DAYS` | `scripts/harvest_stock_data.py:91` | `120` | int | model | trailing window of 1-min bars fetched for the per-day EDGE stamp | `scripts/harvest_stock_data.py` | **none** | LIVE |
| `market_data.closed_bars_v2_enabled()` | `market_data.py:24` | `False` | bool | model | live bar fetches drop the forming partial bar (market_data.py `get_live_atr()`, panel_ranks.py:164, base_loo… | `market_data.py`, `panel_ranks.py`, `base_loop.py` +1 | `test_c26_T2.py`, `test_c26_X1.py`, `test_panel_ranks.py` | LIVE |
| `market_data.daily_feature_restore_enabled()` | `market_data.py:37` | `False` | bool | model | real live values for the 9 daily-window features instead of warmup-fill constants (predict_now.… | `market_data.py`, `predict_now.py`, `panel_ranks.py` +1 | `test_c26_W1.py`, `test_c26_X1.py`, `test_review_b16.py` | LIVE |
| `market_data.har_daily_feed_enabled()` | `market_data.py:60` | `False` | bool | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches GARCH -> HAR (vol… | `market_data.py`, `volatility.py`, `predict_now.py` `get_live_prediction()` | `test_c26_W1.py` | LIVE |
| `volatility.har_daily_feed_enabled()` | `volatility.py:259` | `False` | bool | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches GARCH -> HAR (vol… | `market_data.py`, `volatility.py`, `predict_now.py` `get_live_prediction()` | `test_c26_W1.py` | LIVE |
| `data_sources._yf_window_slice_enabled()` | `data_sources.py:18` | `False` | bool | model | slices the yfinance response to the requested window (data_sources.py:196) | `data_sources.py` | `test_c26_T2.py` | LIVE |
| `data_utils.raw_sidecar_enabled()` | `data_utils.py:34` | `False` | bool | ops | harvest writes/reads a raw-OHLCV sidecar (scripts/harvest_*_data.py) | `data_utils.py`, `scripts/harvest_stock_data.py` / `scripts/harvest_crypto_data.py` `main()` | `test_c26_T2.py` | LIVE |
| `scripts/hypersearch_v2._fixed_holdout_days()` | `scripts/hypersearch_v2.py:391` | `None` | int? | model | fixed trailing holdout span in days (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_holdout_boundary.py` | LIVE |
| `scripts/hypersearch_v2._trainer_seed()` | `scripts/hypersearch_v2.py:594` | `None` | int? | model | base training seed (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_training_repairs.py` | LIVE |
| `indicators._HAS_C` | `indicators.py:14-20` | `False` | bool | ops | opt-in C backend for the TA kernels: tried only when `TRADER_INDICATORS_C=1` AND `import indicators_c` succeeds; not model-facing (≤1e-12 rel parity with numba) | `indicators.py:325,341,358,377` | `test_indicators_parity.py`, `test_indicators.py` (monkeypatch `False`) | LIVE·inert until the archived `.so` is fixed and restored (`archive/README.md` c_ext row) — added 2026-09-26 |
| `STAGE0_DUMP_DEFAULT` | `backtest.py:97` | `True` | bool | measurement | Stage-0 predictions dump + hourly MTM equity (measurement-only; never touches admission/exits/S… | `backtest.py` | **none** | LIVE |

Notes on the table above:
- `market_data.har_daily_feed_enabled()` materializes twice (`market_data.py` `har_daily_feed_enabled()` and the
  `volatility.py` `har_daily_feed_enabled()` proxy) — one env var, two attributes; that is why the env-attribute block
  holds 27 rows for 26 env flags plus `backtest.STAGE0_DUMP_DEFAULT` (`indicators._HAS_C` added
  2026-09-26).
- `TRAINER_SEED` and `FIXED_HOLDOUT_DAYS` are constants whose **env twin wins**
  (`scripts/hypersearch_v2.py` `_trainer_seed()` and `_fixed_holdout_days()`); both are listed in §5.
- `SIGNAL_EXIT_CONFIRM_READS` (int) and `META_CALIBRATION_MODE` / `indicator_config.json:preset` /
  `llm_config.json:selection_mode` (strings) are **mode switches**, not booleans.
- Flags/constants with **no test at all**: `HAR_VOL_ENABLED`, `ENTRY_WINDOWS_ENABLED`,
  `OVERNIGHT_SLEEVE_*` (all four), `MAKER_STAGE_TIMEOUT`, `KISH_RHO_FLOOR`,
  `CRYPTO_CS_DISPERSION_FLOOR`, `CRYPTO_TREND_SMA_WINDOW`, `CRYPTO_TREND_FLOOR`,
  `AS_OF_TRADABLE_TOP_K`, `TRADABLE_K_ENTER`, `TRADABLE_K_HOLD`, `TRADER_BOTS_OPS`,
  `TRADER_USE_ALPACA_PY`, `TRADER_MINUTE_EDGE_DAYS`, `TRADER_HEALTHCHECK_URL_{NAME}`,
  `FINNHUB_API_KEY`.
  For the c26 family the *module attribute* is pinned but the **env-var parsing itself** is only
  covered for `TRADER_CLOSED_BARS_V2`, `TRADER_RAW_SIDECAR`, `TRADER_YF_WINDOW_SLICE`,
  `TRADER_DAILY_FEATURE_RESTORE`, `TRADER_HAR_DAILY_FEED`, `TRADER_JOURNAL_ROTATE_DAYS`,
  `TRADER_FIXED_HOLDOUT_DAYS`, `TRADER_TRAINER_SEED`.

---

## 2. Declared-ahead / no production reader (17 flags + companions)

These are **not orphaned constants** — each one's own comment says it is staged ahead of its
wiring, and `tests/test_improve_stratcfg.py` asserts they stay at their defaults until wiring
lands. A flip today is a silent no-op. Root cause is module-level, not constant-level: the modules
that would read them have **no production importer at all**.

| Unwired module | Imported by | Flags it would gate |
|---|---|---|
| `bet_sizing.py` | tests only | `EDGE_KELLY_ENABLED` |
| `crypto_trend.py` | tests only | `CRYPTO_TREND_GATE_ENABLED`, `CRYPTO_TREND_SMA_WINDOW`, `CRYPTO_TREND_FLOOR` |
| `execution_policy.py` | `tests/test_execution_policy.py`, `tests/test_execution_policy_v3.py`, `tests/test_review_b02.py` | `EXEC_TAKER_FLOOR_PCT`, `EXEC_WIDE_SPREAD_PCT`, `EXEC_POST_INSIDE_FRAC`, `EXEC_EDGE_HEADROOM_MULT` |
| `portfolio_backtest.py` | tests + the measurement CLI `scripts/rank_gradient_report.py` | `CONCENTRATION_ENABLED`, `CONVICTION_K_MAX/K_MIN`, `CONVICTION_SIGNAL_FLOOR/META_FLOOR/RATIO_FLOOR`, `TIER_SIZING_ENABLED`, `TIER_A_K` |
| `panel_ranks.live_tradable_members` (`panel_ranks.py:260`) | `tests/test_universe_promotion.py` only | `TRADABLE_POOL_ENABLED`, `AS_OF_TRADABLE_TOP_K`, `TRADABLE_K_ENTER`, `TRADABLE_K_HOLD` |
| `panel_ranks.cs_size_tilt` (`panel_ranks.py:342`) | `tests/test_review_b16.py` only — and it takes `dispersion_floor` as an argument | `CRYPTO_CS_RANK_ENABLED`, `CRYPTO_CS_DISPERSION_FLOOR` |
| — (no kernel; live floor is a hardcoded `0.1` in `base_loop`) | — | `TILT_MIN` |
| — (`order_utils` reads only `IOC_CAP_BPS`) | — | `IOC_EXIT_CAP_BPS` |

**The two worst offenders:**

1. **`IOC_EXIT_CAP_BPS`** (`strategy_config.py:106`) — its sibling `IOC_CAP_BPS` *does* have a
   production reader (`order_utils.py` `_entry_ioc_cap_bps()`), so the pair reads as a
   matched set while only half is wired. Exits and flattens are deliberately never slippage-capped
   (`order_utils.py` comment at `:22-23`), so the value is inventory, not a bug — but any doc that
   implies exit IOCs are capped is wrong.
2. **`CRYPTO_CS_DISPERSION_FLOOR`** (`strategy_config.py:726`) — the only forward-declared constant
   with **zero references repo-wide, tests included**. Every sibling in the wave-9 block is at least
   value-pinned by `tests/test_improve_stratcfg.py`; this one is not, so nothing would notice if it
   changed. Adding it to that pin list is the cheap fix.

Also **inert-by-dependency** (a reader exists, but a flip alone does nothing) — see the
`LIVE·inert until X` rows in §1: `BLEND_FIT_ON_REFIT` / `BLEND_THRESHOLD_RESELECT` (need
`HYPERSEARCH_V3`), `IMPACT_COST_ENABLED` (warned no-op without a `DV30` column,
`liquidity.py` `impact_inputs_from_df()`), `IOC_CAP_BPS` (needs `TRADER_IOC_ENTRY_CAP=1`), `KISH_NEFF_ENABLED`
(read only under `PROMOTION_GATE_V2`), and `HAR_VOL_ENABLED` — default **True** yet structurally
unreachable live until `TRADER_HAR_DAILY_FEED=1` (the `market_data.py` `har_daily_feed_enabled()` docstring says so explicitly).

---

## 3. `TRADER_SHADOW_MODE` — one default, in both readers (**fixed 2026-09-08**)

| Reader | Expression | Unset | `=0` | `=1` |
|---|---|---|---|---|
| `run_pipeline._build_training_phases` (**authoritative**) | `os.getenv('TRADER_SHADOW_MODE','1') != '0'` | **ON** | OFF | ON |
| `gui.py` settings chip (display only) | `os.getenv('TRADER_SHADOW_MODE','1') != '0'` | **ON** | OFF | ON |

**What the flag does:** its only effect is routing the weekly retrain's save into the **challenger**
slot instead of the champion. It suppresses no orders and gates no entry — trading is live either
way.

*Both readers agreed as of 2026-09-08.* They did not before this cleanup pass: the GUI chip read
`bool(os.getenv('TRADER_SHADOW_MODE'))`, inverted in both directions (OFF when unset, ON for the
documented `=0`, since `bool("0") is True`), and its label claimed "SHADOW MODE — orders
suppressed". The expression now matches `_build_training_phases` exactly and the chip reads
"SHADOW MODE — weekly retrain saves to the challenger slot" / "Immediate promotion (shadow off)".
`gui.py` is PySide6-gated, so the corrected chip is `py_compile`-checked here and **visually
verified on the Jetson**.

Every other env var read from two places agrees on its default: `CUDA_VISIBLE_DEVICES`
(`predict_now.py:20` / `hw_monitor.py:137`), `TORCH_NUM_THREADS`, `LD_LIBRARY_PATH`,
`TRADER_WEBHOOK_URL`, `TRADER_TELEGRAM_*`, `TRADER_HEALTHCHECK_URL` (all "unset ⇒ off").

---

## 4. Non-flag constants — the policy surface

Same columns as §1 minus the flag-specific ones. These are values, not switches: changing one is a
policy change with the same promotion obligations as flipping a gate-facing flag.


### Exit policy dicts (`strategy_config.py:21-47`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_POLICY` | `strategy_config.py:21` | `{"atr_stop_mult": 2.5, "atr_trail_mult": 2.…` | gate | crypto ATR stop/trail/TP/cooldown/lockout policy consumed by policy_exits.exit_… | `crypto_loop.py`, `gui.py`, `backtest.py (via policy_for)` +3 | LIVE |
| `STOCK_POLICY` | `strategy_config.py:35` | `{"atr_stop_mult": 2.0, "atr_trail_mult": 2.…` | gate | stock ATR stop/trail/TP/cooldown/lockout policy | `stock_loop.py`, `gui.py`, `gap_audit.py` +1 | LIVE |

### Sizing / risk (`:49-77`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `RISK_PCT_PER_TRADE` | `strategy_config.py:50` | `0.005` | gate | fraction of equity risked to the stop on each entry | `base_loop.py`, `stock_loop.py`, `gui.py` | LIVE |
| `MAX_BOOK_RISK_PCT` | `strategy_config.py:51` | `0.025` | gate | correlation-adjusted (ENB) per-book stop-risk cap | `base_loop.py`, `gui.py` | LIVE |
| `KELLY_CAP` | `strategy_config.py:53` | `0.25` | gate | fractional-Kelly ceiling on the size multiplier | `base_loop.py` | LIVE |
| `PORTFOLIO_VOL_TARGET` | `strategy_config.py:54` | `{"crypto": 0.35, "stock": 0.18}` | gate | annualized book vol targets driving the book-vol scalar | `portfolio.py`, `volatility.py` | LIVE |
| `TILT_MAX` | `strategy_config.py:58` | `1.3` | gate | upper clamp on the composed regime/sentiment/LLM size tilt | `base_loop.py`, `scripts/sizing_cofire_report.py` | LIVE |
| `TILT_MIN` | `strategy_config.py:59` | `0.7` | gate† | NOTHING - reserved; the live de-risk floor is a hardcoded 0.1 in base_loop | **none** | DECLARED-AHEAD |
| `MIN_ORDER_NOTIONAL` | `strategy_config.py:69` | `100` | gate | dust-order floor below which an entry is skipped | `base_loop.py`, `gui.py` | LIVE |
| `MAX_TRADES_PER_SYMBOL_PER_DAY` | `strategy_config.py:74` | `{"crypto": 4, "stock": 3}` | gate | per-symbol daily NEW-ENTRY budget (exits never limited) | `base_loop.py` | LIVE |

### Execution — maker ladder, entry tactics, IOC caps (`:79-106`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `MAKER_STAGE_TIMEOUT` | `strategy_config.py:81` | `25` | gate | seconds per bid-join rung (2 rungs max) | `crypto_loop.py` | LIVE |
| `EXEC_TAKER_FLOOR_PCT` | `strategy_config.py:93` | `0.05` | gate† | spread <= this -> cross immediately | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_WIDE_SPREAD_PCT` | `strategy_config.py:94` | `0.15` | gate† | spread >= this -> candidate to post inside the quote | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_POST_INSIDE_FRAC` | `strategy_config.py:95` | `0.4` | gate† | fraction of the half-spread to post inside | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_EDGE_HEADROOM_MULT` | `strategy_config.py:96` | `1.5` | gate† | required pred/edge_floor headroom before risking a passive non-fill | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `IOC_CAP_BPS` | `strategy_config.py:105` | `{"mega": 8, "mid": 20, "spec": 40}` | gate | per-name-class bps cap past the touch for ENTRY marketable-IOC orders (order_ut… | `order_utils.py` | LIVE·inert until TRADER_IOC_ENTRY_CAP=1 |
| `IOC_EXIT_CAP_BPS` | `strategy_config.py:106` | `{"mega": 15, "mid": 35, "spec": 50}` | gate† | per-name-class bps cap for EXIT/flatten IOCs | **none** | DECLARED-AHEAD |

### Stock entry windows + overnight sleeve (`:108-124`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `STOCK_ENTRY_WINDOWS_ET` | `strategy_config.py:113` | `[["09:45", "11:00"], ["14:30", "15:30"]]` | gate | allowed intraday entry windows (ET) for the stock book | `stock_loop.py`, `backtest.py` | LIVE |
| `OVERNIGHT_SLEEVE_MAX_POSITIONS` | `strategy_config.py:122` | `2` | gate | max positions held overnight | `stock_loop.py` | LIVE |
| `OVERNIGHT_SLEEVE_MAX_PCT_EQUITY` | `strategy_config.py:123` | `0.05` | gate | per-kept-position equity cap overnight | `stock_loop.py` | LIVE |
| `OVERNIGHT_SLEEVE_MIN_PRED` | `strategy_config.py:124` | `0.0` | gate | min predicted return to keep a name overnight | `stock_loop.py` | LIVE |

### Square-root market impact (`:134-144`) + promotion gate v2 (`:155-173`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `IMPACT_K` | `strategy_config.py:143` | `1.0` | gate | sqrt-impact coefficient (Almgren/Kyle) | `liquidity.py` | LIVE·inert until IMPACT_COST_ENABLED |
| `IMPACT_TYPICAL_NOTIONAL` | `strategy_config.py:144` | `25000` | gate | representative order size for the %-return replay | `liquidity.py` | LIVE·inert until IMPACT_COST_ENABLED |
| `KISH_RHO_FLOOR` | `strategy_config.py:173` | `{"crypto": 0.5, "stock": 0.25}` | gate | per-book conservative rho lower bounds for the Kish softening (backtest.py `aggregate_metrics()`) | `backtest.py`, `scripts/hypersearch_v2.py` | LIVE·inert until KISH_NEFF_ENABLED |

### IA-4 influence-audit companions (`:580-687`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CORR_SANITY_MAX` | `strategy_config.py:665` | `0.85` | gate | loose admission sanity bar on avg\|corr\| when the correlation family is merged… | `base_loop.py`, `stock_loop.py` | LIVE·inert until CORR_FAMILY_MERGED |
| `TRADE_BUDGET_BACKSTOP_MULT` | `strategy_config.py:677` | `3` | gate | multiplier applied to the daily cap when the backstop flag is on (base_loop.py:… | `base_loop.py` | LIVE·inert until TRADE_BUDGET_BACKSTOP |
| `KELLY_SAMPLE_MIN_TRADES` | `strategy_config.py:725` | `50` | gate | uncensored-trade threshold releasing the Kelly gate (base_loop.py `_compute_position_size()`) | `base_loop.py` | LIVE·inert until KELLY_SAMPLE_GATE |
| `KELLY_SAMPLE_SINCE` | `strategy_config.py:726` | `2026-08-22` | gate | ISO date from which trades count as uncensored (base_loop.py `_compute_position_size()`) | `base_loop.py` | LIVE·inert until KELLY_SAMPLE_GATE |

### BTC trailing-RV regime ladder (`:689-698`) — `volatility.py` only

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_RV_ENTER_HIGH_PCT` | `strategy_config.py:745` | `80.0` | gate | BTC trailing-RV percentile entering the HIGH state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_ENTER_CRISIS_PCT` | `strategy_config.py:746` | `95.0` | gate | percentile entering the CRISIS state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_HIGH_PCT` | `strategy_config.py:747` | `65.0` | gate | percentile below which HIGH may release | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_CRISIS_PCT` | `strategy_config.py:748` | `90.0` | gate | percentile at which CRISIS drops to HIGH | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_HOLD_EVALS` | `strategy_config.py:749` | `12` | gate | consecutive new-bar evaluations required to leave HIGH | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_MIN_HISTORY_DAYS` | `strategy_config.py:750` | `90` | gate | below this the RV state is 'unknown' and fails OPEN at 1.0 | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_HIGH_MULT` | `strategy_config.py:751` | `0.5` | gate | size multiplier in the HIGH RV state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_CRISIS_MULT` | `strategy_config.py:752` | `0.3` | gate | size multiplier in the CRISIS RV state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |

### WAVE-9 forward-declared companions (`:700-756`) — see §2

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_CS_DISPERSION_FLOOR` | `strategy_config.py:780` | `0.01` | gate† | dispersion floor below which the crypto CS rank tilt is a no-op | **none** | DEAD-zero-refs |
| `CRYPTO_TREND_SMA_WINDOW` | `strategy_config.py:791` | `200` | gate† | SMA window for the BTC trend gate | **none** | DECLARED-AHEAD |
| `CRYPTO_TREND_FLOOR` | `strategy_config.py:792` | `0.5` | gate† | floor of the BTC trend de-risk scalar | **none** | DECLARED-AHEAD |
| `CONVICTION_K_MAX` | `strategy_config.py:804` | `7` | gate† | max names in the conviction walk | **none** | DECLARED-AHEAD |
| `CONVICTION_K_MIN` | `strategy_config.py:805` | `3` | gate† | Statman diversification floor for the conviction walk | **none** | DECLARED-AHEAD |
| `TIER_A_K` | `strategy_config.py:810` | `3` | gate† | top-K names receiving edge-proportional concentration | **none** | DECLARED-AHEAD |

### Cross-book account stop-risk (`:766-773`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CROSS_BOOK_RHO` | `strategy_config.py:827` | `1.0` | measurement | assumed cross-book correlation for the GATE-1 account-risk measurement journal … | `base_loop.py`, `risk_budget.py (argument)` | LIVE |

### Constants in the other committed config modules


**`stock_config.py` — universe / sector policy**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `LEVERAGED_ETFS` | `stock_config.py:73` | `{"TQQQ": 3, "SOXL": 3}` | gate | ticker -> leverage multiplier; position sizes divided by it | `base_loop.py` | LIVE |
| `SAFE_HAVEN_SYMBOLS` | `stock_config.py:82` | `"{'PEP', 'WMT', 'COPX', 'MRK', 'T', 'KO', '…` | gate | names still tradable during the VIX>25 defensive block | `base_loop.py`, `stock_loop.py` | LIVE |
| `ETF_TICKERS` | `stock_config.py:87` | `"{'IWM', 'COPX', 'SPY', 'PPLT', 'ARKK', 'TQ…` | model | ETFs whose residual momentum is hard-zeroed | `indicators.py` | LIVE |
| `TRAINING_CANDIDATE_POOL` | `stock_config.py:99` | `["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"…` | model | sector-diverse liquid pool added to the harvest for survivorship mitigation (NO… | `panel_ranks.py`, `short_flow.py`, `scripts/harvest_stock_data.py` | LIVE |
| `AS_OF_TOP_K` | `stock_config.py:122` | `60` | model | as-of membership mask: keep a training row only when the name ranked top-K by t… | `panel_ranks.py`, `scripts/harvest_stock_data.py` | LIVE |
| `CANDIDATE_START` | `stock_config.py:126` | `2021-01-01` | model | earliest fetch date for candidate-pool names | `scripts/harvest_stock_data.py` | LIVE |
| `SECTOR_BUCKETS` | `stock_config.py:135` | `{"COIN": "crypto_proxy", "MSTR": "crypto_pr…` | gate | ticker -> factor bucket for the crowding notional cap | `stock_loop.py`, `borrow_proxy.py` | LIVE |
| `BUCKET_CAP_FRACTION` | `stock_config.py:178` | `{"crypto_proxy": 0.2, "default": 0.35}` | gate | bucket notional caps as a fraction of MAX_EXPOSURE | `stock_loop.py` | LIVE |
| `AS_OF_TRADABLE_TOP_K` | `stock_config.py:195` | `20` | gate† | as-of top-K for the tradable pool | **none** | DECLARED-AHEAD |
| `TRADABLE_K_ENTER` | `stock_config.py:196` | `20` | gate† | hysteresis enter rank for tradable promotion | **none** | DECLARED-AHEAD |
| `TRADABLE_K_HOLD` | `stock_config.py:197` | `28` | gate† | hysteresis hold rank for tradable promotion | **none** | DECLARED-AHEAD |
| `CRYPTO_SYMBOLS` | `stock_config.py:206` | `["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD"…` | gate | live crypto trading + panel set (6 coins) | `crypto_loop.py`, `gui.py`, `llm_analyst.py` | LIVE |
| `CRYPTO_POOL` | `stock_config.py:221` | `<expr: CRYPTO_SYMBOLS + ['AVAX/USD', 'BCH/U…` | measurement | intended full 10-coin set (declaration) | `gui.py`, `liquidity.py`, `scripts/crypto_spread_census.py` | LIVE |

**`indicator_config.py` — feature presets**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_ONLY_COLS` | `indicator_config.py:36` | `["BTC_Return_1h", "BTC_SMA_Ratio", "BTC_RSI…` | model | columns present only in crypto training data (asset-type filtering) | `gui.py`, `scripts/hypersearch_v2.py (indirect)` | LIVE |
| `STOCK_ONLY_COLS` | `indicator_config.py:41` | `["VWAP", "Price_VWAP_Ratio", "Gap_Pct", "AT…` | model | columns present only in stock training data | `gui.py` | LIVE |
| `PRESETS` | `indicator_config.py:207` | `<expr: {'minimal': {'description': 'Core si…` | model | feature-preset registry: minimal \| standard (code default) \| stationary \| st… | `gui.py`, `scripts/hypersearch_v2.py (get_preset_features)`, `indicator_leadlag.py` | LIVE |

**`adaptive_config.py` — Optuna search-space governor**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `DEFAULT_SEARCH_SPACE` | `adaptive_config.py:20` | `{"forward_bars": [12, 18, 24, 32, 48], "seq…` | model | Optuna search-space defaults seeding adaptive_state_{asset}.json | `scripts/hypersearch_v2.py`, `scripts/wave6_stage0.py` | LIVE |
| `CATEGORICAL_PARAMS` | `adaptive_config.py:35` | `"{'forward_bars', 'n_heads', 'batch_size'}"` | model | params whose space is a discrete list (edge detection + expansion) | `adaptive_config.py (internal)` | LIVE |
| `RANGE_PARAMS` | `adaptive_config.py:38` | `"{'hidden_dim', 'learning_rate', 'dropout',…` | model | params whose space is a [min,max] range | `adaptive_config.py (internal)` | LIVE |
| `EXPANSION_POOLS` | `adaptive_config.py:45` | `{"forward_bars": {"low": [8], "high": [64, …` | model | new boundary values added when an edge is detected | `adaptive_config.py (internal)` | LIVE |
| `HARD_LIMITS` | `adaptive_config.py:60` | `{"forward_bars": {"min": 8, "max": 96}, "se…` | model | absolute bounds a search space may never exceed | `adaptive_config.py (internal)` | LIVE |
| `TRIAL_COUNTS` | `adaptive_config.py:75` | `{"initial": 200, "refine": 70, "explore": 120}` | model | trials per mode: initial/refine/explore | `adaptive_config.py (internal)`, `run_pipeline.py (via get_trial_count)` | LIVE |
| `STAGNATION_CYCLES` | `adaptive_config.py:82` | `3` | model | cycles without >5% improvement before switching to explore | `adaptive_config.py (internal)` | LIVE |
| `IMPROVEMENT_THRESHOLD` | `adaptive_config.py:83` | `0.05` | model | relative improvement counted as progress | `adaptive_config.py (internal)` | LIVE |
| `FLOAT_EDGE_FRACTION` | `adaptive_config.py:86` | `0.1` | model | fraction of a range within which a value counts as at-edge | `adaptive_config.py (internal)` | LIVE |
| `BASE_DIR` | `adaptive_config.py:17` | `<expr: Path(__file__).resolve().parent>` | ops | directory used to build adaptive_state_{asset_type}.json paths | `adaptive_config.py (internal)` | LIVE |

**`llm_config.py` — non-flag `llm_config.json` keys**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `llm_config.json:models` | `llm_config.py:196` | `{'gemini': 'gemini-2.5-flash-lite', 'claude…` | gate | per-provider {api_key, model} | `llm_client.py` | LIVE |
| `llm_config.json:provider_preference` | `llm_config.py:196` | `['anthropic', 'openai', 'gemini']` | gate | order 'auto' mode tries native providers in | `llm_client.py` | LIVE |
| `llm_config.json:endpoints` | `llm_config.py:196` | `[]` | gate | OpenAI-compatible endpoint list {name,base_url,api_key,model,free,enabled} | `llm_client.py` | LIVE |
| `llm_config.json:pricing` | `llm_config.py:196` | `{}` | measurement | per-MTok price corrections winning over llm_client built-ins | `llm_client.py` | LIVE |
| `llm_config.json:pricing_cache_multipliers` | `llm_config.py:196` | `{'anthropic': [1.25, 0.1], 'gemini': [1.0, …` | measurement | cache-billing multipliers vs input price (llm_client.py `_cache_multipliers()`) | `llm_client.py` | LIVE |
| `llm_config.json:detected_tier` | `llm_config.py:196` | `null` | ops | auto-detected Gemini tier ('free'\|'paid') | `llm_client.py` | LIVE |
| `llm_config.json:fmp_api_key` | `llm_config.py:196` | `""` | ops | Financial Modeling Prep key (unrelated to provider selection) | `fundamentals.py` | LIVE |
| `llm_config.json:max_llm_latency_sec` | `llm_config.py:196` | `30` | ops | per-call timeout budget in seconds | `llm_client.py` | LIVE |
| `LLM_CONFIG_FILE` | `llm_config.py:194` | `<expr: Path(__file__).resolve().parent / 'l…` | ops | path of the gitignored llm_config.json | `llm_config.py (internal)` | LIVE |
| `FREE_CANDIDATE_PRESETS` | `llm_config.py:272` | `[{"name": "openrouter", "base_url": "https:…` | measurement | registry metadata for scripts/llm_qualify.py; NOT merged into _DEFAULTS | `scripts/llm_qualify.py` | LIVE |

**`design_tokens.py` — GUI tokens (no trading effect)**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `TYPE` | `design_tokens.py:79` | `{"display": [24, 700], "heading": [15, 600]…` | ops | GUI type scale | `gui.py` | LIVE |
| `NUMERIC_FAMILY` | `design_tokens.py:87` | `IBM Plex Mono` | ops | mono-tabular numeric font family | `gui.py` | LIVE |
| `UI_FAMILY` | `design_tokens.py:88` | `Inter` | ops | UI font family | `gui.py` | LIVE |
| `FALLBACKS` | `design_tokens.py:89` | `["Segoe UI", "Roboto", "DejaVu Sans", "sans…` | ops | font fallback stack | `gui.py` | LIVE |
| `SPACE` | `design_tokens.py:94` | `{"s1": 4, "s2": 8, "s3": 12, "s4": 16, "s5"…` | ops | GUI spacing grid | `gui.py` | LIVE |
| `RADIUS` | `design_tokens.py:95` | `{"control": 4, "input": 6, "card": 8, "pane…` | ops | GUI corner radii | `gui.py` | LIVE |
| `SOURCE_DEFAULTS` | `design_tokens.py:105` | `{"green": "#00c853", "red": "#ff4444", "yel…` | ops | source color defaults for token resolution | `gui.py` | LIVE |

`adaptive_config` also carries the persisted-state schema (`_default_state`,
`adaptive_config.py:93-113`): `asset_type`, `best_score` (0.0), `best_params`, `search_space`,
`mode` (`'refine'`), `cycles_without_improvement`, `cum_trials`, `cum_holdout_gates`,
`trial_history`, `db_deletions`, `last_updated`. `load_adaptive_state` back-fills missing keys and
**raises on a corrupt file (fail-closed)**.

---

## 5. Environment variables — complete census

60 distinct read patterns: 36 `TRADER_*` (37 concrete names once `TRADER_HEALTHCHECK_URL_{NAME}`
expands to `_CRYPTO`/`_STOCK`) plus 24 others. Of the 36 `TRADER_*`, 26 are flags and 10 are plain
settings. Parsing rules: the "Env-var parsing" table in the preamble.

### 5.1 `TRADER_*`

| Env var | First read at | Default when unset | Facing | What it does | Readers | Test |
|---|---|---|---|---|---|---|
| `TRADER_STOP_CLASSIFY_V2` | `base_loop.py:66` | `0 (OFF)` | gate | 24h re-entry lockout only for server-stop fills classified 'hard'/'unknown'; 't… | `base_loop.py` | `test_c26_T6.py` |
| `TRADER_STREAM_STOP_DETECT` | `base_loop.py:72` | `0 (OFF)` | gate | cached order_stream fill recovers a stop when the REST probe raises (base_loop.… | `base_loop.py` | `test_c26_T6.py` |
| `TRADER_IOC_ENTRY_CAP` | `order_utils.py:24` | `0 (OFF)` | gate | entry-order market fallbacks become slippage-capped marketable IOCs (order_util… | `order_utils.py`, `base_loop.py` `_execute_entry_order()` | `test_c26_T6.py` |
| `TRADER_MAKER_SHARE_NOTIONAL` | `order_utils.py:28` | `0 (OFF)` | gate | should_trade's live crypto threshold uses the notional-weighted maker share (or… | `order_utils.py` | `test_c26_T6.py` |
| `TRADER_SHADOW_DM_V2` | `shadow.py:85` | `0 (OFF)` | gate | DM v2 (two-look Lan-DeMets alpha budget, IM blocks) DECIDES promotion instead o… | `shadow.py` | `test_c26_Q3.py` |
| `TRADER_FUNDING_Z_TIME_THINNING` | `funding.py:42` | `0 (OFF)` | model | time-thinned funding history append + archive-preferred z baseline (funding.py:… | `funding.py` | `test_c26_P5.py` |
| `TRADER_COST_REGIME_FEATURES` | `cost_regime.py:42` | `0 (OFF)` | model | B21 cost-regime meta features written into the harvest store (cost_regime.py `stamp_cost_regime_features()`)… | `cost_regime.py`, `scripts/harvest_stock_data.py` `prepare_stock_data()`, `scripts/harvest_crypto_data.py` `prepare_data()` | `test_c26_T3.py` |
| `TRADER_SPREAD_FILL_V2` | `liquidity.py:69` | `0 (OFF)` | model | no-estimate bars are stamped with the flat FLAT_SPREAD_PCT instead of the floor… | `liquidity.py` | `test_c26_T3.py` |
| `TRADER_CRYPTO_SPREAD_STAMP` | `liquidity.py:70` | `0 (OFF)` | model | per-pair crypto spread tier stamp replaces the flat 0.10% (liquidity.py `stamp_crypto_spreads()`) | `liquidity.py`, `scripts/harvest_crypto_data.py` `prepare_data()` | `test_c26_T3.py` |
| `TRADER_STOCK_MINUTE_EDGE` | `liquidity.py:71` | `0 (OFF)` | model | per-day EDGE spread computed from 1-min bars during the stock harvest (scripts/… | `liquidity.py`, `scripts/harvest_stock_data.py` | `test_c26_T3.py` |
| `TRADER_IMPACT_VOLSCALE` | `liquidity.py:72` | `0 (OFF)` | gate | empirical vol-scale re-base of the sqrt market-impact model (liquidity.py `impact_inputs_from_df()`) | `liquidity.py` | `test_c26_T3.py` |
| `TRADER_CRYPTO_CENSUS_FILE` | `liquidity.py:253` | `'crypto_spread_census.json'` | ops | path of the crypto spread-tier census consumed in liquidity.py `_load_census()` | `liquidity.py` | `test_c26_T3.py (attribute)` |
| `TRADER_CLOSED_BARS_V2` | `market_data.py:33` | `0 (OFF)` | model | live bar fetches drop the forming partial bar (market_data.py `get_live_atr()`, panel_ranks.… | `market_data.py`, `panel_ranks.py`, `base_loop.py` +1 | `test_c26_T2.py`, `test_c26_X1.py`, `test_panel_ranks.py` |
| `TRADER_DAILY_FEATURE_RESTORE` | `market_data.py:56` | `0 (OFF)` | model | real live values for the 9 daily-window features instead of warmup-fill constan… | `market_data.py`, `predict_now.py`, `panel_ranks.py` +1 | `test_c26_W1.py`, `test_c26_X1.py`, `test_review_b16.py` |
| `TRADER_HAR_DAILY_FEED` | `market_data.py:86` | `0 (OFF)` | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches G… | `market_data.py`, `volatility.py`, `predict_now.py` `get_live_prediction()` | `test_c26_W1.py` |
| `TRADER_YF_WINDOW_SLICE` | `data_sources.py:20` | `0 (OFF)` | model | slices the yfinance response to the requested window (data_sources.py:196) | `data_sources.py` | `test_c26_T2.py` |
| `TRADER_RAW_SIDECAR` | `data_utils.py:36` | `0 (OFF)` | ops | harvest writes/reads a raw-OHLCV sidecar (scripts/harvest_*_data.py) | `data_utils.py`, `scripts/harvest_stock_data.py` / `scripts/harvest_crypto_data.py` `main()` | `test_c26_T2.py` |
| `TRADER_JOURNAL_ROTATE_DAYS` | `trade_journal.py:68` | `'0' (OFF)` | ops | gzip day-files older than N days (trade_journal.py `log_decision()`, `rotate_old_journals()`) | `trade_journal.py` | `test_c26_T7.py` |
| `TRADER_LOG_DIR` | `log_config.py:121` (in `_log_paths()`, read at the first `get_logger` / `_setup` call, not at import) | `unset -> <repo>/logs` (production value: unset; `run_pipeline`, `run_bots` and the systemd unit do not set it) | ops | directory for `trader.log` (+ `.1`-`.5` and the `trader.log.lock` flock sibling). Absolute, or relative to the REPO ROOT (never the CWD); empty = unset. Uncreatable/unwritable -> default path + one stderr line, never raises. Set after the handler exists -> no re-pointing. A module-level `_LOG_DIR`/`_LOG_FILE` reassignment (tests' monkeypatch) wins over it. Not model-facing. Tests: set by `tests/conftest.py` to a per-session temp dir so pytest stops writing into the production log (R5-W16) | `log_config.py` | `test_engine_r6_log_dir.py` |
| `TRADER_ORDER_STREAM` | `order_stream.py:97` | `unset (OFF)` | ops | starts the alpaca-py TradingStream order-event listener | `order_stream.py` | `test_review_b02.py` |
| `TRADER_USE_ALPACA_PY` | `trading_utils.py:88` | `unset (legacy SDK)` | ops | forces the alpaca-py CompatREST adapter instead of alpaca-trade-api | `trading_utils.py` | **none** |
| `TRADER_BOTS_OPS` | `run_bots.py:64` | `'1' (ON)` | ops | ops thread in standalone run_bots.py: Telegram kill-switch polling + daily drif… | `run_bots.py` | **none** |
| `TRADER_EOD_DIGEST` | `run_pipeline.py:144` | `'1' (ON)` | ops | once-daily EOD digest notification (run_pipeline.py `_maybe_send_eod_digest()`) | `run_pipeline.py` | `test_c26_T7.py (attribute)` |
| `TRADER_EOD_DIGEST_HOUR` | `run_pipeline.py:146` | `'16'` | ops | local hour after which the EOD digest may fire (run_pipeline.py `_maybe_send_eod_digest()`) | `run_pipeline.py` | `test_c26_T7.py (attribute)` |
| `TRADER_BOT_RESTART_BACKOFF` | `run_pipeline.py:1074` | `0 (OFF)` | ops | crash-restart backoff (60·2^k s, cap 960 s) + give-up latch after 5 identical crashes in 60 min, one critical notify; cleared by start_bot / weekly `_restart_bots` (see §1 row) | `run_pipeline.py` | `test_engine_r5_restart_backoff.py` |
| `TRADER_SHADOW_MODE` | `run_pipeline.py:1425` | `'1' (ON) in both readers (§3; the gui's old UNSET==OFF chip was fixed 2026-09-08)` | gate | weekly retrain saves into the CHALLENGER slot instead of promoting immediately | `run_pipeline.py` `_build_training_phases()`, `gui.py` `_build_settings_tab()` | `test_c26_Q2.py`, `test_c26_X1.py` |
| `TRADER_FIXED_HOLDOUT_DAYS` | `scripts/hypersearch_v2.py:400` | `unset -> strategy_config.FIXED_HOLDOUT_DAYS…` | model | fixed trailing holdout span in days (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_holdout_boundary.py` |
| `TRADER_TRAINER_SEED` | `scripts/hypersearch_v2.py:602` | `unset -> strategy_config.TRAINER_SEED (None)` | model | base training seed (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_training_repairs.py` |
| `TRADER_MINUTE_EDGE_DAYS` | `scripts/harvest_stock_data.py:91` | `'120'` | model | trailing window of 1-min bars fetched for the per-day EDGE stamp | `scripts/harvest_stock_data.py` | **none** |
| `TRADER_INDICATORS_C` | `indicators.py:15` | `unset (OFF → numba > pure)` | ops | opt-in: only `=1` makes `indicators.py` try `import indicators_c` (the C ext archived 2026-09-26 to `archive/c_ext/` after a heap overflow on short frames) and prefer it over numba. Not model-facing (≤1e-12 relative parity with numba; models were trained on numba). Enable only after the C source is fixed — see the `archive/README.md` c_ext row | `indicators.py` | **none** for the env parse; `test_indicators_parity.py` forces `_HAS_C=False` |
| `TRADER_PYBIN` | `scripts/setup_jetson_system.sh:170` | `/home/kyle/miniforge3/envs/jetson/bin/python` (= `run_pipeline.PYTHON`, `run_pipeline.py:44`) | ops | **install-time only**: the interpreter written into the `trader.service` `ExecStart` (`scripts/setup_jetson_system.sh:316`); step 0 exits 2 (`[python] FATAL`) unless it is executable and can `import pyarrow, dotenv, torch` (`scripts/setup_jetson_system.sh:175-184`). `sudo` strips the caller's env, so pass it as `sudo TRADER_PYBIN=… bash …` (`scripts/setup_jetson_system.sh:25-26`). Added 2026-09-26; never read at runtime | `scripts/setup_jetson_system.sh` | `test_jetson_ops_2026_09.py` |
| `TRADER_WEBHOOK_URL` | `notify.py:77` | `unset (channel off)` | ops | Discord/Slack-style webhook target for notify() | `notify.py:77,249`, `gui.py` `_build_settings_tab()`, `_on_test_notify()` | `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_TELEGRAM_BOT_TOKEN` | `notify.py:80` | `unset (channel off)` | ops | Telegram bot token for alerts + the /halt kill switch | `notify.py:80,195,250`, `gui.py` `_build_settings_tab()`, `_on_test_notify()` | `test_notify.py`, `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_TELEGRAM_CHAT_ID` | `notify.py:81` | `unset (channel off)` | ops | Telegram chat id; also the authorization check for inbound commands (notify.py:… | `notify.py:81,196,251`, `gui.py` `_build_settings_tab()`, `_on_test_notify()` | `test_notify.py`, `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_HEALTHCHECK_URL` | `notify.py:110` | `unset (no ping)` | ops | generic healthcheck ping URL (heartbeat) | `notify.py:110`, `gui.py` `_build_settings_tab()` | `test_review_b18.py` |
| `TRADER_HEALTHCHECK_URL_{NAME}` | `notify.py:109` | `unset -> falls back to TRADER_HEALTHCHECK_URL` | ops | per-book healthcheck URL; f-string over name.upper() (CRYPTO / STOCK) | `notify.py:109`, `gui.py` `_build_settings_tab()` | **none** |
| `<NAME>_API_KEY` | `llm_client.py:664` | `''` | ops | per-endpoint key convention: endpoint['name'].upper()+'_API_KEY' (e.g. OPENROUT… | `llm_client.py` `_endpoint_api_key()`, `scripts/llm_qualify.py` `_candidate_api_key()` | **none** |

### 5.2 Everything else

| Env var | First read at | Default when unset | Facing | What it does | Readers | Test |
|---|---|---|---|---|---|---|
| `ALPACA_API_KEY` | `trading_utils.py:73` | `none (loud error)` | ops | Alpaca broker/data credential | `trading_utils.py:73`, `order_stream.py:105`, `gui.py` `main()` +3 | `test_review_b02.py`, `test_review_b03.py` |
| `ALPACA_API_SECRET` | `trading_utils.py:74` | `none (loud error)` | ops | Alpaca broker/data credential | `trading_utils.py:74`, `order_stream.py:106`, `gui.py` `main()` +3 | `test_review_b02.py`, `test_review_b03.py` |
| `ALPACA_BASE_URL` | `trading_utils.py:75` | `none (loud WARNING)` | ops | paper vs live endpoint; unset makes BOTH SDKs default to LIVE (trading_utils.py… | `trading_utils.py:75`, `order_stream.py:111` | `test_review_b02.py`, `test_review_b03.py` |
| `FINNHUB_API_KEY` | `sentiment.py:39` | `none -> feature silently disabled` | ops | Finnhub news + earnings-calendar client | `sentiment.py:39`, `sentiment_history.py` `_get_finnhub()`, `events_calendar.py` `_fetch_calendar()` | **none** |
| `ANTHROPIC_API_KEY` | `llm_client.py:1527` | `''` | ops | fallback for llm_config models.claude.api_key | `llm_client.py` | `test_c26_S2.py`, `test_llm_claude.py`, `test_llm_providers.py` |
| `OPENAI_API_KEY` | `llm_client.py:1534` | `''` | ops | fallback for llm_config models.openai.api_key | `llm_client.py` | `test_llm_providers.py` |
| `CUDA_VISIBLE_DEVICES` | `predict_now.py:20` | `None (unset)` | ops | '' means CPU-only: predict_now caps torch threads, hw_monitor short-circuits GP…; since 2026-09-26 `run_pipeline._training_env` deletes an inherited `''` for training children (`TRAIN_ENV`) while `BOT_ENV` still sets `''`, and the systemd unit no longer sets it | `predict_now.py:20`, `hw_monitor.py:137`, `run_bots.py:45 (setdefault)`, `run_pipeline.py` `_training_env()`, `BOT_ENV` | `test_hw_monitor.py`, `test_jetson_ops_2026_09.py` |
| `TORCH_NUM_THREADS` | `predict_now.py:21` | `'2'` | ops | torch intra-op thread cap for bot inference | `predict_now.py:21`, `run_bots.py:46 (setdefault)`, `run_pipeline.py` `BOT_ENV` | **none** |
| `OMP_NUM_THREADS` | `run_bots.py:47` | `'2'` | ops | OpenMP thread cap (set only, never read by repo code) | `run_bots.py:47`, `run_pipeline.py` `BOT_ENV` | **none** |
| `LD_LIBRARY_PATH` | `run_pipeline.py:308` | `''` | ops | prefixed with Jetson CUDA/cusparselt lib dirs for child processes (gui's `_engine_env` drops empty elements since 2026-09-26) | `run_pipeline.py` `ENV`, `gui.py` `_engine_env()` | **none** |
| `LD_PRELOAD` | `run_pipeline.py:310` | `set only` | ops | Jetson libstdc++ preload for child processes | `run_pipeline.py` | **none** |
| `PYTHONUNBUFFERED` | `run_pipeline.py:311` | `'1'` | ops | unbuffered child-process logging | `run_pipeline.py` | **none** |
| `NOTIFY_SOCKET` | `run_pipeline.py:90` | `unset (no-op)` | ops | systemd Type=notify watchdog socket | `run_pipeline.py` | **none** |
| `PYTORCH_CUDA_ALLOC_CONF` | `scripts/hypersearch_v2.py:44` | `'expandable_segments:True' (setdefault)` | ops | CUDA allocator config for training | `scripts/hypersearch_v2.py` | **none** |
| `APCA_RETRY_WAIT` | `scripts/harvest_crypto_data.py:66` | `'10' (setdefault)` | ops | Alpaca SDK internal retry backoff | `scripts/harvest_crypto_data.py:66`, `scripts/harvest_stock_data.py:61` | **none** |
| `APCA_RETRY_MAX` | `scripts/harvest_crypto_data.py:67` | `'5' (setdefault)` | ops | Alpaca SDK internal retry count | `scripts/harvest_crypto_data.py:67`, `scripts/harvest_stock_data.py:62` | **none** |
| `PY_COLORS` | `scripts/ab_check.sh:36` | `forced to '0' (exported)` | ops | disables pytest ANSI so the FAILED/ERROR name greps cannot be blinded | `scripts/ab_check.sh` | **none** |
| `AB_CHECK_TIMEOUT_S` | `scripts/ab_check.sh:27` | `900` | ops | hard wall for the full pytest run | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `AB_CHECK_RERUN_TIMEOUT_S` | `scripts/ab_check.sh:28` | `300` | ops | wall for the targeted flaky/persistent rerun | `scripts/ab_check.sh` | **none** |
| `AB_CHECK_MIN_PASSED` | `scripts/ab_check.sh:29` | `1500` | ops | launch-sanity floor: fewer passed => refuse to report PASS | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `AB_CHECK_PYTEST` | `scripts/ab_check.sh:30` | `'python3 -m pytest'` | ops | pytest invocation override, only so the gate itself is testable | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `TMPDIR` | `scripts/ab_check.sh:48` | `/tmp` | ops | mktemp location for the ab_check work files | `scripts/ab_check.sh` | **none** |
| `SUDO_USER` | `scripts/setup_jetson_system.sh:303` | `$(whoami)` | ops | systemd unit User= for the Jetson install | `scripts/setup_jetson_system.sh` | **none** |

Names that look like env vars but are **not**: `TRADER_DIR` / `TRADER_USER` are shell locals in
`scripts/setup_jetson_system.sh:302-303` (as are `PYBIN` / `UNIT_LD_PRELOAD` / `UNIT_LD_LIBRARY_PATH`,
`:170-174` — only `TRADER_PYBIN` is read from the environment) and `scripts/backup_state.sh:15`; `TRADER_LOG_ROLE` exists
only as a proposal inside `research/module_review_2026-07.json`. **There is no `GEMINI_API_KEY`
env var** — the Gemini key comes only from `llm_config.json` `models.gemini.api_key`
(`llm_client.py` `probe_tier()`, `resolve_provider_chain()`, `call_gemini()`, `probe_available_models()`).

The documented `.env` template (`scripts/setup.sh:59-67`, `README.md`) covers only
`ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`, `FINNHUB_API_KEY` — four of the ~10
credentials the code can consume; the LLM keys are not templated.

---

## 6. SSOT exceptions — policy that lives outside `strategy_config.py`

`strategy_config.py` is the declared single source of truth for policy, and both the live loops and
the backtester read it. These constants are policy-shaped but live elsewhere. Two are guarded by a
test; the rest can drift silently.

| Constant | Where | Value | Guard |
|---|---|---|---|
| `fees.FLAT_SPREAD_PCT` vs `backtest.SPREAD_PCT` | `fees.py:55` (canonical) / `backtest.py:91` (verbatim copy) | `{'crypto':0.10,'stock':0.05}` | ✅ `tests/test_review_b10.py:154-182` regex-reads `backtest.py` and asserts equality |
| `cooldown_bars = max(1, ceil(policy['cooldown_min']/60))` | derived verbatim twice: `backtest.py:349`, `meta_label.py:738` | — | ✅ `tests/test_improve_stratcfg.py:156-167` asserts exactly one verbatim occurrence in each |
| `risk_budget.ACCOUNT_RISK_CAP` | `risk_budget.py:49` | `0.03` | ✗ account-wide stop-risk cap; the per-book `MAX_BOOK_RISK_PCT` is in `strategy_config` |
| `portfolio.MAX_AVG_CORRELATION` | `portfolio.py:31` | `0.7` | ✗ admission bar on avg pairwise correlation; unrelated to `strategy_config.CORR_SANITY_MAX` (`0.85`) despite doing a similar job |
| `blend_fit.DEFAULT_LSTM_WEIGHT` | `blend_fit.py:29` | `0.6` | ✗ `strategy_config.py:207` merely *describes* it as "the hardcoded 0.6 default" |
| `CORR_SANITY_MAX` re-typed as an import-failure fallback | `base_loop.py:3240`, `stock_loop.py:990` (both `CORR_SANITY_MAX = 0.85`) vs `strategy_config.py:611` | `0.85` ×3 | ✗ changing the config value silently diverges from both fallbacks |
| Holdout fraction `0.12` — **three independent spellings** | `meta_label.py:60` `HOLDOUT_FRACTION = 0.12`; `scripts/hypersearch_v2.py:378` `HOLDOUT_FRACTION = 0.12`; `adaptive_config.py:239` `holdout_span_days: float = 43.8` (= 0.12·365) | — | partial: `tests/test_r2c_holdout_boundary.py:217` regex-pins the hypersearch literal only |

Other policy-shaped constants outside `strategy_config` (inventory, not defects):
`fees.py:41-63` (fee/spread/edge-multiple table), `validation.py:41` `DSR_MIN = 0.6` +
`validation.py:47` `MAX_CSCV_SPLITS`, `meta_label.py:58-93,145-147` (veto prob 0.30, threshold
fraction 0.5, guard rails, OOF starvation tiers), `shadow.py:75-94` (shadow window, promote
p-values, DM-v2 constants), `backtest.py` `BARS_PER_YEAR`, `retrain_ledger.py:35-38`,
`naive_baseline.py:28-29`, `horizon_transfer.py:28`, `serving_cache.py:30`,
`liquidity.py` `CRYPTO_TIER_DEFAULTS_PCT` / `CRYPTO_TIER_FALLBACK_PCT` (crypto spread tiers).

Verified *not* duplicated: `RISK_PCT_PER_TRADE` (0.005) and `MAX_BOOK_RISK_PCT` (0.025) appear as
literals nowhere else in production code; `KELLY_CAP` (0.25) appears only in `bet_sizing.py`
docstrings and test arguments.

---

## 7. Regenerating this census

The tables were built by AST extraction, not grep — the generator lived in the session scratchpad
and was not committed. To redo it on the dev Mac (no heavy imports needed):

1. **Enumerate definitions:** `ast.parse` each of `strategy_config.py`, `indicator_config.py`,
   `stock_config.py`, `adaptive_config.py`, `llm_config.py`, `design_tokens.py`; take every
   module-level `ast.Assign` → `(name, lineno, ast.literal_eval(value) or ast.unparse(value))`.
   `llm_config._DEFAULTS` needs one extra level: its dict keys are the `llm_config.json` flags.
2. **Resolve readers:** `ast.parse` every `.py` in the tree (top level + `scripts/` + `tests/`); record `ImportFrom(module=<config>)`
   names and `Import(<config> as alias)` + every `Attribute(value=Name(alias))`. That gives real
   reads; a plain grep cannot separate them from docstring mentions.
3. **Catch dynamic reads:** regex-sweep `getattr\(\s*(strategy_config|_sc|sc)\s*,\s*['"](\w+)['"]`
   — several campaign flags are read only this way (`backtest.py` `aggregate_metrics()`, `run_pipeline.py` `_build_training_phases()`,
   `meta_label.py:854-855`, `events_calendar.py:57`).
4. **Env vars:** regex `os\.(environ\.get|getenv|environ\[)` over `*.py` + `*.sh`, then read each
   hit's default and comparison to classify parsing rule and polarity.
5. **Status column:** a name whose reader set is empty or tests-only is `DECLARED-AHEAD`; confirm by
   checking whether the *module* that would read it has any production importer
   (`grep -rl '^\(from\|import\) <module>' --include='*.py' . | grep -v tests/`).

Cross-check any rewrite against `research/campaign_2026-08/03_jetson_runbook.md` (activation
sequence) and `research/KILL_LIST.md` (some dark flags are kill-adjacent and need an owner ruling
before activation, e.g. the crypto spread-stamp family — `liquidity.py:64-68`).
