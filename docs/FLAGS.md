# FLAGS.md — every flag, policy constant and environment variable

*Single home for the flag inventory (`docs/README.md` reading order rule: other docs point here,
they do not repeat these numbers). Code-verified 2026-09-08 against the working tree at `master`
+ the in-flight uncommitted diff; line numbers are a snapshot of that date, names are stable.*

## Philosophy — read this before flipping anything

1. **Default-OFF is the contract.** Every model-facing or gate-behavior change from the 2026-08
   campaign onward ships behind a flag whose default reproduces today's behavior exactly. 81
   distinct flags exist: **68 at an OFF/neutral default, 12 default-ON, 1 a neutral mode** (the
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
| **strict `'1'`** — `os.environ.get(NAME) != '1'` | `1` only | `TRADER_ORDER_STREAM` (`order_stream.py:97`), `TRADER_USE_ALPACA_PY` (`trading_utils.py:88`) |
| **2026-08 family** — `os.getenv(NAME,'0').strip().lower() in ('1','true','yes')` | `1`, `true`, `yes` (case/whitespace-insensitive) | all 16 c26/R2C dark flags: `base_loop.py:67,73`, `order_utils.py:25,29`, `shadow.py:86`, `funding.py:40`, `cost_regime.py:42`, `liquidity.py:67-70`, `market_data.py:31,54,84`, `data_sources.py:21`, `data_utils.py:37` |
| **default-ON negation** — `os.environ.get(NAME,'1').strip().lower() not in ('0','false','no')` | `0`, `false`, `no` disable | `TRADER_BOTS_OPS` (`run_bots.py:58`), `TRADER_EOD_DIGEST` (`run_pipeline.py:118`) |
| **literal `'0'` only** — `os.getenv(NAME,'1') != '0'` | `0` disables; anything else is ON | `TRADER_SHADOW_MODE` (`run_pipeline.py:1016`) |
| **int with guard** | `try: int(...) except ValueError: <fallback>` | `TRADER_EOD_DIGEST_HOUR` (→16), `TRADER_JOURNAL_ROTATE_DAYS` (→0). **`TRADER_MINUTE_EDGE_DAYS` (`scripts/harvest_stock_data.py:89`) has NO guard — a non-numeric value crashes at import.** |

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

#### strategy_config.py (41)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `HAR_VOL_ENABLED` | `strategy_config.py:63` | `True` | bool | gate | volatility.get_sigma uses HAR-RV sigma with GARCH fallback (volatility.py:452-455); False force… | `volatility.py` | **none** | LIVE·inert until TRADER_HAR_DAILY_FEED=1 |
| `CONVICTION_JOURNAL_ENABLED` | `strategy_config.py:65` | `True` | bool | measurement | per-candidate veto attribution + sizing detail written to the decision journal (base_loop.py:21… | `base_loop.py`, `decision_report.py` | `test_conviction_journal.py`, `test_decision_report_v3.py` | LIVE |
| `MAKER_ENTRIES_ENABLED` | `strategy_config.py:79` | `True` | bool | gate | crypto passive bid-join entry ladder (crypto_loop.py:101-102); False = taker only | `crypto_loop.py` | `test_execution_policy_v3.py` | LIVE |
| `ENTRY_WINDOWS_ENABLED` | `strategy_config.py:116` | `True` | bool | gate | entry-window mask applied (backtest.py:240, stock_loop.py:223); False = all RTH hours | `stock_loop.py`, `backtest.py` | **none** | LIVE |
| `OVERNIGHT_SLEEVE_ENABLED` | `strategy_config.py:120` | `True` | bool | gate | keep up to N stock positions overnight instead of EOD-flattening everything | `stock_loop.py` | **none** | LIVE |
| `EVENTS_TRADING_DAY_WINDOWS` | `strategy_config.py:131` | `False` | bool | gate | earnings buffers walk TRADING days (weekend/holiday aware) instead of calendar days (events_cal… | `events_calendar.py` | `test_c26_P4.py` | LIVE |
| `IMPACT_COST_ENABLED` | `strategy_config.py:141` | `False` | bool | gate | adds a sqrt(notional/ADV) impact haircut to OFFLINE backtest/meta net P&L (liquidity.py:507) | `liquidity.py`, `backtest.py` | `test_c26_T3.py`, `test_market_impact.py`, `test_improve_stratcfg.py`, `test_review_b10.py` | LIVE·inert until a DV30 column |
| `UNIQUENESS_WEIGHTS_ENABLED` | `strategy_config.py:152` | `False` | bool | model | average-uniqueness sample weights in the LSTM loss (scripts/hypersearch_v2.py:1336-1339) | `scripts/hypersearch_v2.py` | `test_improve_stratcfg.py` | LIVE |
| `PROMOTION_GATE_V2` | `strategy_config.py:166` | `False` | bool | gate | calendar-concurrency n_eff replaces per-ticker uniqueness + cluster count, n_eff<10 fails CLOSE… | `backtest.py`, `run_pipeline.py`, `scripts/hypersearch_v2.py` +1 | `test_c26_Q1.py`, `test_c26_X1.py` | LIVE |
| `KISH_NEFF_ENABLED` | `strategy_config.py:171` | `False` | bool | gate | Kish design-effect softening of the calendar concurrency (backtest.py:424); read only when PROM… | `backtest.py`, `scripts/hypersearch_v2.py` | `test_c26_Q1.py` | LIVE·inert until PROMOTION_GATE_V2 |
| `GATE_TARGETS_CHALLENGER` | `strategy_config.py:186` | `False` | bool | gate | weekly policy gate replays the CHALLENGER artifacts instead of the champion (run_pipeline.py:10… | `run_pipeline.py`, `shadow.py`, `backtest.py` | `test_c26_Q2.py`, `test_c26_R1.py` | LIVE |
| `OBJECTIVE_LONG_ONLY` | `strategy_config.py:196` | `False` | bool | model | hypersearch simulate_trades scores ONLY the deployable long leg (scripts/hypersearch_v2.py:476-… | `scripts/hypersearch_v2.py` | `test_improve_stratcfg.py` | LIVE |
| `HYPERSEARCH_V3` | `strategy_config.py:224` | `False` | bool | model | final refit on all pre-holdout data + pre-gate LGB + blend-weight fit + blended holdout certifi… | `scripts/hypersearch_v2.py` | `test_c26_T1.py`, `test_r2c_blend_coherence.py`, `test_r2c_lgb_refit.py` | LIVE |
| `BLEND_FIT_ON_REFIT` | `strategy_config.py:237` | `False` | bool | model | deployed lstm_weight comes from the refit-state fit instead of the stale fold-scaler fit (scrip… | `scripts/hypersearch_v2.py` | `test_r2c_blend_coherence.py` | LIVE·inert until HYPERSEARCH_V3 |
| `BLEND_THRESHOLD_RESELECT` | `strategy_config.py:247` | `False` | bool | model | trade_threshold re-selected on BLENDED val predictions before certification (scripts/hypersearc… | `scripts/hypersearch_v2.py`, `blend_fit.py (kernel reselect_trade_threshold)` | `test_r2c_blend_coherence.py` | LIVE·inert until HYPERSEARCH_V3 |
| `LGB_REFIT_FULL` | `strategy_config.py:273` | `False` | bool | model | both LGB legs retrain on all purged pre-holdout rows at a fixed round count; q10 veto floor rec… | `scripts/hypersearch_v2.py`, `objective_utils.py (lgb_refit_indices kernel)` | `test_r2c_lgb_refit.py` | LIVE |
| `OBJECTIVE_V3` | `strategy_config.py:288` | `False` | bool | model | ticker-block position reset + edge-anchored trade_threshold range + holdout-crossing val purge … | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_c26_T1.py`, `test_r2c_blend_coherence.py`, `test_r2c_holdout_boundary.py` | LIVE |
| `TRAINER_SEED` | `strategy_config.py:306` | `None` | int? | model | seeds torch init/dropout, batch permutations and the TPESampler via objective_utils.derive_seed… | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_r2c_training_repairs.py` | LIVE |
| `TRAINING_REPAIRS_V1` | `strategy_config.py:330` | `False` | bool | model | L1 val-loss criterion match + L2 lagged regime mask + L5 pristine OOM probe + L6 bar-denominate… | `scripts/hypersearch_v2.py`, `objective_utils.py` | `test_r2c_training_repairs.py` | LIVE |
| `FIXED_HOLDOUT_DAYS` | `strategy_config.py:353` | `None` | int? | model | holdout becomes a fixed trailing calendar span instead of the 12% quantile (scripts/hypersearch… | `scripts/hypersearch_v2.py`, `objective_utils.py (holdout_boundary)`, `scripts/window_ab.py` | `test_r2c_holdout_boundary.py` | LIVE |
| `META_CALIBRATION_MODE` | `strategy_config.py:362` | `legacy` | str | model | 'legacy' same-slice isotonic vs 'purged_oof' leak-free calibration of the meta probability | `meta_label.py`, `calibration.py` | `test_improve_stratcfg.py` | LIVE |
| `CALIBRATION_V2` | `strategy_config.py:381` | `False` | bool | model | tie-pooled PAVA + logit-Platt MAP smoothing + OOF embargo + size-aware chooser (calibration.py:… | `calibration.py`, `meta_label.py` | `test_c26_R1.py` | LIVE |
| `META_OOF_PRED` | `strategy_config.py:404` | `False` | bool | model | consumes {prefix}oof_preds.npz for the meta 'pred' feature + entry filter (meta_label.py:854) | `meta_label.py`, `scripts/meta_learning_curve.py` | `test_c26_R2.py` | LIVE |
| `META_REPLAY_POLICY_PARITY` | `strategy_config.py:419` | `False` | bool | model | meta replay applies the deployed admission conditions (cost floor, cooldown/lockout, entry wind… | `meta_label.py`, `scripts/meta_learning_curve.py` | `test_c26_R2.py` | LIVE |
| `DERISK_STACK_V2` | `strategy_config.py:454` | `False` | bool | gate | regime family aggregated by MIN, one VIX tier map, modal regime = 1.0, crypto BTC-RV replaces V… | `base_loop.py`, `portfolio.py` | `test_c26_S3.py` | LIVE |
| `VIX25_BLOCK_REMOVED` | `strategy_config.py:474` | `False` | bool | gate | skips the VIX>25 non-safe-haven entry block (base_loop.py:3095, stock_loop.py:1154) | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `CORR_FAMILY_MERGED` | `strategy_config.py:486` | `False` | bool | gate | ENB stop-risk budget becomes the single correlation consumer; admission loosens to CORR_SANITY_… | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `TRADE_BUDGET_BACKSTOP` | `strategy_config.py:498` | `False` | bool | gate | multiplies MAX_TRADES_PER_SYMBOL_PER_DAY by the backstop mult, leaving cooldown as the one chur… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `CRYPTO_VERTICAL_BARRIER` | `strategy_config.py:515` | `False` | bool | gate | fb-anchored max-hold exit for crypto positions in the LOOP layer (base_loop.py:1387); would-fir… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `SIGNAL_EXIT_CONFIRM_READS` | `strategy_config.py:530` | `1` | int | gate | 1 = signal exit fires on one reading; 2 = requires two consecutive readings (base_loop.py:1934-… | `base_loop.py`, `stock_loop.py` | `test_ia4_flagged.py` | LIVE |
| `KELLY_SAMPLE_GATE` | `strategy_config.py:546` | `False` | bool | gate | holds kelly_mult neutral at 1.0 until the book has enough uncensored post-D06 trades (base_loop… | `base_loop.py`, `trading_utils.py (uncensored_trade_count substrate)` | `test_ia4_flagged.py` | LIVE |
| `BREAKER_PER_BOOK` | `strategy_config.py:563` | `False` | bool | gate | per-book P&L circuit-breaker baseline + window roll instead of the account-wide Alpaca last_equ… | `base_loop.py` | `test_ia4_flagged.py` | LIVE |
| `EDGE_KELLY_ENABLED` | `strategy_config.py:593` | `False` | bool | gate† | would replace clip(2p,0.6,1.3) with bet_sizing.afml_bet_size / kelly_edge_odds | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CRYPTO_CS_RANK_ENABLED` | `strategy_config.py:601` | `False` | bool | gate† | would apply a soft [0.90,1.10] crypto cross-sectional rank size tilt | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CRYPTO_TREND_GATE_ENABLED` | `strategy_config.py:612` | `False` | bool | gate† | would compose a graded BTC-200h-SMA de-risk scalar into CryptoLoop._extra_tilt | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONCENTRATION_ENABLED` | `strategy_config.py:625` | `False` | bool | gate† | would enable conviction-gated dynamic top-K admission | **none** | `test_improve_stratcfg.py`, `test_conviction_ab.py` | DECLARED-AHEAD |
| `CONVICTION_SIGNAL_FLOOR` | `strategy_config.py:628` | `None` | float? | gate† | None = signal floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONVICTION_META_FLOOR` | `strategy_config.py:629` | `None` | float? | gate† | None = meta-probability floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `CONVICTION_RATIO_FLOOR` | `strategy_config.py:630` | `None` | float? | gate† | None = edge/cost ratio floor not applied | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `TIER_SIZING_ENABLED` | `strategy_config.py:631` | `False` | bool | gate† | would concentrate the top tier by edge | **none** | `test_improve_stratcfg.py` | DECLARED-AHEAD |
| `PREDICTION_CACHE_ENABLED` | `strategy_config.py:640` | `False` | bool | gate | bar-keyed memo of the feature+LSTM+LGB inference result (predict_now.py:238-242) | `predict_now.py`, `prediction_cache.py` | `test_improve_stratcfg.py` | LIVE |

#### indicator_config.py / stock_config.py (3)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `HURST_ON_RETURNS` | `indicator_config.py:31` | `False` | bool | model | Hurst R/S computed on RETURNS instead of price LEVELS (indicators.py:573) | `indicators.py` | `test_ia1_removals.py`, `test_live_feature_parity.py` | LIVE |
| `TRADABLE_POOL_ENABLED` | `stock_config.py:192` | `False` | bool | gate† | would promote candidate-pool names into the live selectable set | **none** | `test_grp_ops.py` | DECLARED-AHEAD |
| `indicator_config.json:preset` | `indicator_config.py:20` | `standard` | str | model | which feature columns scripts/hypersearch_v2.py trains on (get_preset_features) | `scripts/hypersearch_v2.py`, `gui.py`, `indicator_leadlag.py` | `test_indicator_config.py` | LIVE |

#### llm_config.json keys (12)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `llm_config.json:enabled` | `llm_config.py:192` | `True` | bool | gate | master on/off for the whole LLM stack | `llm_client.py` | **none** | LIVE |
| `llm_config.json:provider` | `llm_config.py:192` | `auto` | str | gate | legacy single-provider pin; consulted only when selection_mode=='single' | `llm_client.py` | `test_llm_config.py` | LIVE |
| `llm_config.json:selection_mode` | `llm_config.py:192` | `auto` | str | gate | provider-chain builder: auto \| single \| free-only \| best-free | `llm_client.py (resolve_provider_chain)` | `test_llm_routing.py` | LIVE |
| `llm_config.json:analyst_model_override` | `llm_config.py:192` | `null` | str? | gate | pin the analyst model (None = smart routing) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:sentiment_model_override` | `llm_config.py:192` | `null` | str? | gate | pin the sentiment model (None = smart routing) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:anthropic_cache_system_ttl` | `llm_config.py:192` | `"" (off)` | str | gate | Anthropic prompt-cache breakpoint on the static system prompt (llm_client.py:1152-1157); "" \| … | `llm_client.py` | `test_c26_S2.py` | LIVE |
| `llm_config.json:tier_override` | `llm_config.py:192` | `null` | str? | ops | manual Gemini tier override ('free'\|'paid'\|None) | `llm_client.py` | **none** | LIVE |
| `llm_config.json:journal_enabled` | `llm_config.py:192` | `True` | bool | measurement | LLM call journaling | `trade_journal.py`, `llm_client.py` | **none** | LIVE |
| `llm_config.json:rich_context_enabled` | `llm_config.py:192` | `False` | bool | gate | attach the compact quant evidence block to live LLM candidates (llm_analyst.py:1137-1144) | `llm_analyst.py`, `base_loop.py:1821`, `stock_loop.py:515` | `test_llm_advice.py` | LIVE |
| `llm_config.json:replay_capture_enabled` | `llm_config.py:192` | `True` | bool | measurement | journal full candidate cycles to journals/llm_replay/ (llm_analyst.py:1317) | `llm_analyst.py` | `test_llm_dossier_persist.py` | LIVE |
| `llm_config.json:advisor_v2_enabled` | `llm_config.py:192` | `False` | bool | measurement | structured decision dossier: extended prompt/schema/shadow journal (llm_analyst.py:500) | `llm_analyst.py`, `llm_eval.py` | `test_llm_dossier_persist.py` | LIVE |
| `llm_config.json:analyst_dedup_ttl_sec` | `llm_config.py:192` | `0` | int | gate | evidence-hash call-dedup TTL for analyze_trades (llm_analyst.py:503); clamped to [0,7000] | `llm_analyst.py` | **none** | LIVE |

#### env-flag module attributes (26)

| Flag | Defined | Default | Type | Facing | Gates (one line) | Readers | OFF-path pinned by | Status |
|---|---|---|---|---|---|---|---|---|
| `base_loop.STOP_CLASSIFY_V2` | `base_loop.py:66` | `False` | bool | gate | 24h re-entry lockout only for server-stop fills classified 'hard'/'unknown'; 'trail' exempt (ba… | `base_loop.py` | `test_c26_T6.py` | LIVE |
| `base_loop.STREAM_STOP_DETECT` | `base_loop.py:72` | `False` | bool | gate | cached order_stream fill recovers a stop when the REST probe raises (base_loop.py:1140) | `base_loop.py` | `test_c26_T6.py` | LIVE |
| `order_utils.IOC_ENTRY_CAP_ENABLED` | `order_utils.py:24` | `False` | bool | gate | entry-order market fallbacks become slippage-capped marketable IOCs (order_utils.py:762) | `order_utils.py`, `base_loop.py:3194` | `test_c26_T6.py` | LIVE |
| `order_utils.MAKER_SHARE_NOTIONAL_ENABLED` | `order_utils.py:28` | `False` | bool | gate | should_trade's live crypto threshold uses the notional-weighted maker share (order_utils.py:963) | `order_utils.py` | `test_c26_T6.py` | LIVE |
| `shadow.DM_V2_ENABLED` | `shadow.py:85` | `False` | bool | gate | DM v2 (two-look Lan-DeMets alpha budget, IM blocks) DECIDES promotion instead of legacy (shadow… | `shadow.py` | `test_c26_Q3.py` | LIVE |
| `funding.FUNDING_Z_TIME_THINNING` | `funding.py:39` | `False` | bool | model | time-thinned funding history append + archive-preferred z baseline (funding.py:122,226) | `funding.py` | `test_c26_P5.py` | LIVE |
| `cost_regime.COST_REGIME_FEATURES` | `cost_regime.py:42` | `False` | bool | model | B21 cost-regime meta features written into the harvest store (cost_regime.py:305) | `cost_regime.py`, `scripts/harvest_stock_data.py:244`, `scripts/harvest_crypto_data.py:186` | `test_c26_T3.py` | LIVE |
| `liquidity.SPREAD_FILL_V2` | `liquidity.py:67` | `False` | bool | model | no-estimate bars are stamped with the flat FLAT_SPREAD_PCT instead of the floor, +inf clipped t… | `liquidity.py` | `test_c26_T3.py` | LIVE |
| `liquidity.CRYPTO_SPREAD_STAMP` | `liquidity.py:68` | `False` | bool | model | per-pair crypto spread tier stamp replaces the flat 0.10% (liquidity.py:315) | `liquidity.py`, `scripts/harvest_crypto_data.py:176` | `test_c26_T3.py` | LIVE |
| `liquidity.STOCK_MINUTE_EDGE` | `liquidity.py:69` | `False` | bool | model | per-day EDGE spread computed from 1-min bars during the stock harvest (scripts/harvest_stock_da… | `liquidity.py`, `scripts/harvest_stock_data.py` | `test_c26_T3.py` | LIVE |
| `liquidity.IMPACT_VOLSCALE` | `liquidity.py:70` | `False` | bool | gate | empirical vol-scale re-base of the sqrt market-impact model (liquidity.py:512) | `liquidity.py` | `test_c26_T3.py` | LIVE |
| `liquidity.CRYPTO_CENSUS_FILE` | `liquidity.py:252` | `crypto_spread_census.json` | str | ops | path of the crypto spread-tier census consumed at liquidity.py:272 | `liquidity.py` | `test_c26_T3.py (attribute)` | LIVE |
| `trade_journal.JOURNAL_ROTATE_DAYS` | `trade_journal.py:63` | `0` | int | ops | gzip day-files older than N days (trade_journal.py:111,152,159) | `trade_journal.py` | `test_c26_T7.py` | LIVE |
| `run_bots._OPS_ENABLED` | `run_bots.py:58` | `True` | bool | ops | ops thread in standalone run_bots.py: Telegram kill-switch polling + daily drift check (run_bot… | `run_bots.py` | **none** | LIVE |
| `run_pipeline.EOD_DIGEST_ENABLED` | `run_pipeline.py:118` | `True` | bool | ops | once-daily EOD digest notification (run_pipeline.py:175) | `run_pipeline.py` | `test_c26_T7.py (attribute)` | LIVE |
| `run_pipeline.EOD_DIGEST_HOUR` | `run_pipeline.py:120` | `16` | int | ops | local hour after which the EOD digest may fire (run_pipeline.py:178) | `run_pipeline.py` | `test_c26_T7.py (attribute)` | LIVE |
| `scripts/harvest_stock_data.MINUTE_EDGE_DAYS` | `scripts/harvest_stock_data.py:89` | `120` | int | model | trailing window of 1-min bars fetched for the per-day EDGE stamp | `scripts/harvest_stock_data.py` | **none** | LIVE |
| `market_data.closed_bars_v2_enabled()` | `market_data.py:21` | `False` | bool | model | live bar fetches drop the forming partial bar (market_data.py:691, panel_ranks.py:164, base_loo… | `market_data.py`, `panel_ranks.py`, `base_loop.py` +1 | `test_c26_T2.py`, `test_c26_X1.py`, `test_panel_ranks.py` | LIVE |
| `market_data.daily_feature_restore_enabled()` | `market_data.py:34` | `False` | bool | model | real live values for the 9 daily-window features instead of warmup-fill constants (predict_now.… | `market_data.py`, `predict_now.py`, `panel_ranks.py` +1 | `test_c26_W1.py`, `test_c26_X1.py`, `test_review_b16.py` | LIVE |
| `market_data.har_daily_feed_enabled()` | `market_data.py:57` | `False` | bool | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches GARCH -> HAR (vol… | `market_data.py`, `volatility.py`, `predict_now.py:198` | `test_c26_W1.py` | LIVE |
| `volatility.har_daily_feed_enabled()` | `volatility.py:250` | `False` | bool | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches GARCH -> HAR (vol… | `market_data.py`, `volatility.py`, `predict_now.py:198` | `test_c26_W1.py` | LIVE |
| `data_sources._yf_window_slice_enabled()` | `data_sources.py:18` | `False` | bool | model | slices the yfinance response to the requested window (data_sources.py:196) | `data_sources.py` | `test_c26_T2.py` | LIVE |
| `data_utils.raw_sidecar_enabled()` | `data_utils.py:34` | `False` | bool | ops | harvest writes/reads a raw-OHLCV sidecar (scripts/harvest_*_data.py) | `data_utils.py`, `scripts/harvest_stock_data.py:418`, `scripts/harvest_crypto_data.py:265` | `test_c26_T2.py` | LIVE |
| `scripts/hypersearch_v2._fixed_holdout_days()` | `scripts/hypersearch_v2.py:341` | `None` | int? | model | fixed trailing holdout span in days (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_holdout_boundary.py` | LIVE |
| `scripts/hypersearch_v2._trainer_seed()` | `scripts/hypersearch_v2.py:525` | `None` | int? | model | base training seed (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_training_repairs.py` | LIVE |
| `STAGE0_DUMP_DEFAULT` | `backtest.py:94` | `True` | bool | measurement | Stage-0 predictions dump + hourly MTM equity (measurement-only; never touches admission/exits/S… | `backtest.py` | **none** | LIVE |

Notes on the table above:
- `market_data.har_daily_feed_enabled()` materializes twice (`market_data.py:57` and the
  `volatility.py:250` proxy) — one env var, two attributes; that is why the env-attribute block
  holds 26 rows for 25 env flags plus `backtest.STAGE0_DUMP_DEFAULT`.
- `TRAINER_SEED` and `FIXED_HOLDOUT_DAYS` are constants whose **env twin wins**
  (`scripts/hypersearch_v2.py:533` and `:350`); both are listed in §5.
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

1. **`IOC_EXIT_CAP_BPS`** (`strategy_config.py:105`) — its sibling `IOC_CAP_BPS` *does* have a
   production reader (`order_utils._ioc_cap_bps`, `order_utils.py:291-293`), so the pair reads as a
   matched set while only half is wired. Exits and flattens are deliberately never slippage-capped
   (`order_utils.py` comment at `:22-23`), so the value is inventory, not a bug — but any doc that
   implies exit IOCs are capped is wrong.
2. **`CRYPTO_CS_DISPERSION_FLOOR`** (`strategy_config.py:602`) — the only forward-declared constant
   with **zero references repo-wide, tests included**. Every sibling in the wave-9 block is at least
   value-pinned by `tests/test_improve_stratcfg.py`; this one is not, so nothing would notice if it
   changed. Adding it to that pin list is the cheap fix.

Also **inert-by-dependency** (a reader exists, but a flip alone does nothing) — see the
`LIVE·inert until X` rows in §1: `BLEND_FIT_ON_REFIT` / `BLEND_THRESHOLD_RESELECT` (need
`HYPERSEARCH_V3`), `IMPACT_COST_ENABLED` (warned no-op without a `DV30` column,
`liquidity.py:513-516`), `IOC_CAP_BPS` (needs `TRADER_IOC_ENTRY_CAP=1`), `KISH_NEFF_ENABLED`
(read only under `PROMOTION_GATE_V2`), and `HAR_VOL_ENABLED` — default **True** yet structurally
unreachable live until `TRADER_HAR_DAILY_FEED=1` (`market_data.py:64-67` says so explicitly).

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
(`predict_now.py:19` / `hw_monitor.py:137`), `TORCH_NUM_THREADS`, `LD_LIBRARY_PATH`,
`TRADER_WEBHOOK_URL`, `TRADER_TELEGRAM_*`, `TRADER_HEALTHCHECK_URL` (all "unset ⇒ off").

---

## 4. Non-flag constants — the policy surface

Same columns as §1 minus the flag-specific ones. These are values, not switches: changing one is a
policy change with the same promotion obligations as flipping a gate-facing flag.


### Exit policy dicts (`strategy_config.py:20-46`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_POLICY` | `strategy_config.py:20` | `{"atr_stop_mult": 2.5, "atr_trail_mult": 2.…` | gate | crypto ATR stop/trail/TP/cooldown/lockout policy consumed by policy_exits.exit_… | `crypto_loop.py`, `gui.py`, `backtest.py (via policy_for)` +3 | LIVE |
| `STOCK_POLICY` | `strategy_config.py:34` | `{"atr_stop_mult": 2.0, "atr_trail_mult": 2.…` | gate | stock ATR stop/trail/TP/cooldown/lockout policy | `stock_loop.py`, `gui.py`, `gap_audit.py` +1 | LIVE |

### Sizing / risk (`:48-76`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `RISK_PCT_PER_TRADE` | `strategy_config.py:49` | `0.005` | gate | fraction of equity risked to the stop on each entry | `base_loop.py`, `stock_loop.py`, `gui.py` | LIVE |
| `MAX_BOOK_RISK_PCT` | `strategy_config.py:50` | `0.025` | gate | correlation-adjusted (ENB) per-book stop-risk cap | `base_loop.py`, `gui.py` | LIVE |
| `KELLY_CAP` | `strategy_config.py:52` | `0.25` | gate | fractional-Kelly ceiling on the size multiplier | `base_loop.py` | LIVE |
| `PORTFOLIO_VOL_TARGET` | `strategy_config.py:53` | `{"crypto": 0.35, "stock": 0.18}` | gate | annualized book vol targets driving the book-vol scalar | `portfolio.py`, `volatility.py` | LIVE |
| `TILT_MAX` | `strategy_config.py:57` | `1.3` | gate | upper clamp on the composed regime/sentiment/LLM size tilt | `base_loop.py`, `scripts/sizing_cofire_report.py` | LIVE |
| `TILT_MIN` | `strategy_config.py:58` | `0.7` | gate† | NOTHING - reserved; the live de-risk floor is a hardcoded 0.1 in base_loop | **none** | DECLARED-AHEAD |
| `MIN_ORDER_NOTIONAL` | `strategy_config.py:68` | `100` | gate | dust-order floor below which an entry is skipped | `base_loop.py`, `gui.py` | LIVE |
| `MAX_TRADES_PER_SYMBOL_PER_DAY` | `strategy_config.py:73` | `{"crypto": 4, "stock": 3}` | gate | per-symbol daily NEW-ENTRY budget (exits never limited) | `base_loop.py` | LIVE |

### Execution — maker ladder, entry tactics, IOC caps (`:78-105`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `MAKER_STAGE_TIMEOUT` | `strategy_config.py:80` | `25` | gate | seconds per bid-join rung (2 rungs max) | `crypto_loop.py` | LIVE |
| `EXEC_TAKER_FLOOR_PCT` | `strategy_config.py:92` | `0.05` | gate† | spread <= this -> cross immediately | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_WIDE_SPREAD_PCT` | `strategy_config.py:93` | `0.15` | gate† | spread >= this -> candidate to post inside the quote | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_POST_INSIDE_FRAC` | `strategy_config.py:94` | `0.4` | gate† | fraction of the half-spread to post inside | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `EXEC_EDGE_HEADROOM_MULT` | `strategy_config.py:95` | `1.5` | gate† | required pred/edge_floor headroom before risking a passive non-fill | `execution_policy.py (NO production importer - tests only)` | DECLARED-AHEAD |
| `IOC_CAP_BPS` | `strategy_config.py:104` | `{"mega": 8, "mid": 20, "spec": 40}` | gate | per-name-class bps cap past the touch for ENTRY marketable-IOC orders (order_ut… | `order_utils.py` | LIVE·inert until TRADER_IOC_ENTRY_CAP=1 |
| `IOC_EXIT_CAP_BPS` | `strategy_config.py:105` | `{"mega": 15, "mid": 35, "spec": 50}` | gate† | per-name-class bps cap for EXIT/flatten IOCs | **none** | DECLARED-AHEAD |

### Stock entry windows + overnight sleeve (`:107-123`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `STOCK_ENTRY_WINDOWS_ET` | `strategy_config.py:112` | `[["09:45", "11:00"], ["14:30", "15:30"]]` | gate | allowed intraday entry windows (ET) for the stock book | `stock_loop.py`, `backtest.py` | LIVE |
| `OVERNIGHT_SLEEVE_MAX_POSITIONS` | `strategy_config.py:121` | `2` | gate | max positions held overnight | `stock_loop.py` | LIVE |
| `OVERNIGHT_SLEEVE_MAX_PCT_EQUITY` | `strategy_config.py:122` | `0.05` | gate | per-kept-position equity cap overnight | `stock_loop.py` | LIVE |
| `OVERNIGHT_SLEEVE_MIN_PRED` | `strategy_config.py:123` | `0.0` | gate | min predicted return to keep a name overnight | `stock_loop.py` | LIVE |

### Square-root market impact (`:133-143`) + promotion gate v2 (`:154-172`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `IMPACT_K` | `strategy_config.py:142` | `1.0` | gate | sqrt-impact coefficient (Almgren/Kyle) | `liquidity.py` | LIVE·inert until IMPACT_COST_ENABLED |
| `IMPACT_TYPICAL_NOTIONAL` | `strategy_config.py:143` | `25000` | gate | representative order size for the %-return replay | `liquidity.py` | LIVE·inert until IMPACT_COST_ENABLED |
| `KISH_RHO_FLOOR` | `strategy_config.py:172` | `{"crypto": 0.5, "stock": 0.25}` | gate | per-book conservative rho lower bounds for the Kish softening (backtest.py:425) | `backtest.py`, `scripts/hypersearch_v2.py` | LIVE·inert until KISH_NEFF_ENABLED |

### IA-4 influence-audit companions (`:456-563`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CORR_SANITY_MAX` | `strategy_config.py:487` | `0.85` | gate | loose admission sanity bar on avg\|corr\| when the correlation family is merged… | `base_loop.py`, `stock_loop.py` | LIVE·inert until CORR_FAMILY_MERGED |
| `TRADE_BUDGET_BACKSTOP_MULT` | `strategy_config.py:499` | `3` | gate | multiplier applied to the daily cap when the backstop flag is on (base_loop.py:… | `base_loop.py` | LIVE·inert until TRADE_BUDGET_BACKSTOP |
| `KELLY_SAMPLE_MIN_TRADES` | `strategy_config.py:547` | `50` | gate | uncensored-trade threshold releasing the Kelly gate (base_loop.py:2360) | `base_loop.py` | LIVE·inert until KELLY_SAMPLE_GATE |
| `KELLY_SAMPLE_SINCE` | `strategy_config.py:548` | `2026-08-22` | gate | ISO date from which trades count as uncensored (base_loop.py:2360) | `base_loop.py` | LIVE·inert until KELLY_SAMPLE_GATE |

### BTC trailing-RV regime ladder (`:565-574`) — `volatility.py` only

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_RV_ENTER_HIGH_PCT` | `strategy_config.py:567` | `80.0` | gate | BTC trailing-RV percentile entering the HIGH state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_ENTER_CRISIS_PCT` | `strategy_config.py:568` | `95.0` | gate | percentile entering the CRISIS state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_HIGH_PCT` | `strategy_config.py:569` | `65.0` | gate | percentile below which HIGH may release | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_CRISIS_PCT` | `strategy_config.py:570` | `90.0` | gate | percentile at which CRISIS drops to HIGH | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_EXIT_HOLD_EVALS` | `strategy_config.py:571` | `12` | gate | consecutive new-bar evaluations required to leave HIGH | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_MIN_HISTORY_DAYS` | `strategy_config.py:572` | `90` | gate | below this the RV state is 'unknown' and fails OPEN at 1.0 | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_HIGH_MULT` | `strategy_config.py:573` | `0.5` | gate | size multiplier in the HIGH RV state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |
| `CRYPTO_RV_CRISIS_MULT` | `strategy_config.py:574` | `0.3` | gate | size multiplier in the CRISIS RV state | `volatility.py` | LIVE·inert until DERISK_STACK_V2 |

### WAVE-9 forward-declared companions (`:576-632`) — see §2

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_CS_DISPERSION_FLOOR` | `strategy_config.py:602` | `0.01` | gate† | dispersion floor below which the crypto CS rank tilt is a no-op | **none** | DEAD-zero-refs |
| `CRYPTO_TREND_SMA_WINDOW` | `strategy_config.py:613` | `200` | gate† | SMA window for the BTC trend gate | **none** | DECLARED-AHEAD |
| `CRYPTO_TREND_FLOOR` | `strategy_config.py:614` | `0.5` | gate† | floor of the BTC trend de-risk scalar | **none** | DECLARED-AHEAD |
| `CONVICTION_K_MAX` | `strategy_config.py:626` | `7` | gate† | max names in the conviction walk | **none** | DECLARED-AHEAD |
| `CONVICTION_K_MIN` | `strategy_config.py:627` | `3` | gate† | Statman diversification floor for the conviction walk | **none** | DECLARED-AHEAD |
| `TIER_A_K` | `strategy_config.py:632` | `3` | gate† | top-K names receiving edge-proportional concentration | **none** | DECLARED-AHEAD |

### Cross-book account stop-risk (`:642-649`)

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CROSS_BOOK_RHO` | `strategy_config.py:649` | `1.0` | measurement | assumed cross-book correlation for the GATE-1 account-risk measurement journal … | `base_loop.py`, `risk_budget.py (argument)` | LIVE |

### Constants in the other committed config modules


**`stock_config.py` — universe / sector policy**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `LEVERAGED_ETFS` | `stock_config.py:71` | `{"TQQQ": 3, "SOXL": 3}` | gate | ticker -> leverage multiplier; position sizes divided by it | `base_loop.py` | LIVE |
| `SAFE_HAVEN_SYMBOLS` | `stock_config.py:80` | `"{'PEP', 'WMT', 'COPX', 'MRK', 'T', 'KO', '…` | gate | names still tradable during the VIX>25 defensive block | `base_loop.py`, `stock_loop.py` | LIVE |
| `ETF_TICKERS` | `stock_config.py:85` | `"{'IWM', 'COPX', 'SPY', 'PPLT', 'ARKK', 'TQ…` | model | ETFs whose residual momentum is hard-zeroed | `indicators.py` | LIVE |
| `TRAINING_CANDIDATE_POOL` | `stock_config.py:97` | `["JPM", "BAC", "WFC", "GS", "MS", "V", "MA"…` | model | sector-diverse liquid pool added to the harvest for survivorship mitigation (NO… | `panel_ranks.py`, `short_flow.py`, `scripts/harvest_stock_data.py` | LIVE |
| `AS_OF_TOP_K` | `stock_config.py:120` | `60` | model | as-of membership mask: keep a training row only when the name ranked top-K by t… | `panel_ranks.py`, `scripts/harvest_stock_data.py` | LIVE |
| `CANDIDATE_START` | `stock_config.py:124` | `2021-01-01` | model | earliest fetch date for candidate-pool names | `scripts/harvest_stock_data.py` | LIVE |
| `SECTOR_BUCKETS` | `stock_config.py:133` | `{"COIN": "crypto_proxy", "MSTR": "crypto_pr…` | gate | ticker -> factor bucket for the crowding notional cap | `stock_loop.py`, `borrow_proxy.py` | LIVE |
| `BUCKET_CAP_FRACTION` | `stock_config.py:176` | `{"crypto_proxy": 0.2, "default": 0.35}` | gate | bucket notional caps as a fraction of MAX_EXPOSURE | `stock_loop.py` | LIVE |
| `AS_OF_TRADABLE_TOP_K` | `stock_config.py:193` | `20` | gate† | as-of top-K for the tradable pool | **none** | DECLARED-AHEAD |
| `TRADABLE_K_ENTER` | `stock_config.py:194` | `20` | gate† | hysteresis enter rank for tradable promotion | **none** | DECLARED-AHEAD |
| `TRADABLE_K_HOLD` | `stock_config.py:195` | `28` | gate† | hysteresis hold rank for tradable promotion | **none** | DECLARED-AHEAD |
| `CRYPTO_SYMBOLS` | `stock_config.py:204` | `["BTC/USD", "ETH/USD", "XRP/USD", "SOL/USD"…` | gate | live crypto trading + panel set (6 coins) | `crypto_loop.py`, `gui.py`, `llm_analyst.py` | LIVE |
| `CRYPTO_POOL` | `stock_config.py:218` | `<expr: CRYPTO_SYMBOLS + ['AVAX/USD', 'BCH/U…` | measurement | intended full 10-coin set (declaration) | `gui.py`, `liquidity.py`, `scripts/crypto_spread_census.py` | LIVE |

**`indicator_config.py` — feature presets**

| Constant | Defined | Default | Facing | What it controls | Readers | Status |
|---|---|---|---|---|---|---|
| `CRYPTO_ONLY_COLS` | `indicator_config.py:34` | `["BTC_Return_1h", "BTC_SMA_Ratio", "BTC_RSI…` | model | columns present only in crypto training data (asset-type filtering) | `gui.py`, `scripts/hypersearch_v2.py (indirect)` | LIVE |
| `STOCK_ONLY_COLS` | `indicator_config.py:39` | `["VWAP", "Price_VWAP_Ratio", "Gap_Pct", "AT…` | model | columns present only in stock training data | `gui.py` | LIVE |
| `PRESETS` | `indicator_config.py:203` | `<expr: {'minimal': {'description': 'Core si…` | model | feature-preset registry: minimal \| standard (code default) \| stationary \| st… | `gui.py`, `scripts/hypersearch_v2.py (get_preset_features)`, `indicator_leadlag.py` | LIVE |

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
| `llm_config.json:models` | `llm_config.py:192` | `{'gemini': 'gemini-2.5-flash-lite', 'claude…` | gate | per-provider {api_key, model} | `llm_client.py` | LIVE |
| `llm_config.json:provider_preference` | `llm_config.py:192` | `['anthropic', 'openai', 'gemini']` | gate | order 'auto' mode tries native providers in | `llm_client.py` | LIVE |
| `llm_config.json:endpoints` | `llm_config.py:192` | `[]` | gate | OpenAI-compatible endpoint list {name,base_url,api_key,model,free,enabled} | `llm_client.py` | LIVE |
| `llm_config.json:pricing` | `llm_config.py:192` | `{}` | measurement | per-MTok price corrections winning over llm_client built-ins | `llm_client.py` | LIVE |
| `llm_config.json:pricing_cache_multipliers` | `llm_config.py:192` | `{'anthropic': [1.25, 0.1], 'gemini': [1.0, …` | measurement | cache-billing multipliers vs input price (llm_client.py:266) | `llm_client.py` | LIVE |
| `llm_config.json:detected_tier` | `llm_config.py:192` | `null` | ops | auto-detected Gemini tier ('free'\|'paid') | `llm_client.py` | LIVE |
| `llm_config.json:fmp_api_key` | `llm_config.py:192` | `""` | ops | Financial Modeling Prep key (unrelated to provider selection) | `fundamentals.py` | LIVE |
| `llm_config.json:max_llm_latency_sec` | `llm_config.py:192` | `30` | ops | per-call timeout budget in seconds | `llm_client.py` | LIVE |
| `LLM_CONFIG_FILE` | `llm_config.py:190` | `<expr: Path(__file__).resolve().parent / 'l…` | ops | path of the gitignored llm_config.json | `llm_config.py (internal)` | LIVE |
| `FREE_CANDIDATE_PRESETS` | `llm_config.py:268` | `[{"name": "openrouter", "base_url": "https:…` | measurement | registry metadata for scripts/llm_qualify.py; NOT merged into _DEFAULTS | `scripts/llm_qualify.py` | LIVE |

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

56 distinct read patterns: 32 `TRADER_*` (33 concrete names once `TRADER_HEALTHCHECK_URL_{NAME}`
expands to `_CRYPTO`/`_STOCK`) plus 24 others. Of the 32 `TRADER_*`, 24 are flags and 8 are plain
settings. Parsing rules: the "Env-var parsing" table in the preamble.

### 5.1 `TRADER_*`

| Env var | First read at | Default when unset | Facing | What it does | Readers | Test |
|---|---|---|---|---|---|---|
| `TRADER_STOP_CLASSIFY_V2` | `base_loop.py:66` | `0 (OFF)` | gate | 24h re-entry lockout only for server-stop fills classified 'hard'/'unknown'; 't… | `base_loop.py` | `test_c26_T6.py` |
| `TRADER_STREAM_STOP_DETECT` | `base_loop.py:72` | `0 (OFF)` | gate | cached order_stream fill recovers a stop when the REST probe raises (base_loop.… | `base_loop.py` | `test_c26_T6.py` |
| `TRADER_IOC_ENTRY_CAP` | `order_utils.py:24` | `0 (OFF)` | gate | entry-order market fallbacks become slippage-capped marketable IOCs (order_util… | `order_utils.py`, `base_loop.py:3194` | `test_c26_T6.py` |
| `TRADER_MAKER_SHARE_NOTIONAL` | `order_utils.py:28` | `0 (OFF)` | gate | should_trade's live crypto threshold uses the notional-weighted maker share (or… | `order_utils.py` | `test_c26_T6.py` |
| `TRADER_SHADOW_DM_V2` | `shadow.py:85` | `0 (OFF)` | gate | DM v2 (two-look Lan-DeMets alpha budget, IM blocks) DECIDES promotion instead o… | `shadow.py` | `test_c26_Q3.py` |
| `TRADER_FUNDING_Z_TIME_THINNING` | `funding.py:39` | `0 (OFF)` | model | time-thinned funding history append + archive-preferred z baseline (funding.py:… | `funding.py` | `test_c26_P5.py` |
| `TRADER_COST_REGIME_FEATURES` | `cost_regime.py:42` | `0 (OFF)` | model | B21 cost-regime meta features written into the harvest store (cost_regime.py:30… | `cost_regime.py`, `scripts/harvest_stock_data.py:244`, `scripts/harvest_crypto_data.py:186` | `test_c26_T3.py` |
| `TRADER_SPREAD_FILL_V2` | `liquidity.py:67` | `0 (OFF)` | model | no-estimate bars are stamped with the flat FLAT_SPREAD_PCT instead of the floor… | `liquidity.py` | `test_c26_T3.py` |
| `TRADER_CRYPTO_SPREAD_STAMP` | `liquidity.py:68` | `0 (OFF)` | model | per-pair crypto spread tier stamp replaces the flat 0.10% (liquidity.py:315) | `liquidity.py`, `scripts/harvest_crypto_data.py:176` | `test_c26_T3.py` |
| `TRADER_STOCK_MINUTE_EDGE` | `liquidity.py:69` | `0 (OFF)` | model | per-day EDGE spread computed from 1-min bars during the stock harvest (scripts/… | `liquidity.py`, `scripts/harvest_stock_data.py` | `test_c26_T3.py` |
| `TRADER_IMPACT_VOLSCALE` | `liquidity.py:70` | `0 (OFF)` | gate | empirical vol-scale re-base of the sqrt market-impact model (liquidity.py:512) | `liquidity.py` | `test_c26_T3.py` |
| `TRADER_CRYPTO_CENSUS_FILE` | `liquidity.py:252` | `'crypto_spread_census.json'` | ops | path of the crypto spread-tier census consumed at liquidity.py:272 | `liquidity.py` | `test_c26_T3.py (attribute)` |
| `TRADER_CLOSED_BARS_V2` | `market_data.py:30` | `0 (OFF)` | model | live bar fetches drop the forming partial bar (market_data.py:691, panel_ranks.… | `market_data.py`, `panel_ranks.py`, `base_loop.py` +1 | `test_c26_T2.py`, `test_c26_X1.py`, `test_panel_ranks.py` |
| `TRADER_DAILY_FEATURE_RESTORE` | `market_data.py:53` | `0 (OFF)` | model | real live values for the 9 daily-window features instead of warmup-fill constan… | `market_data.py`, `predict_now.py`, `panel_ranks.py` +1 | `test_c26_W1.py`, `test_c26_X1.py`, `test_review_b16.py` |
| `TRADER_HAR_DAILY_FEED` | `market_data.py:83` | `0 (OFF)` | gate | feeds volatility.get_sigma a complete-day RRV series so sizing sigma switches G… | `market_data.py`, `volatility.py`, `predict_now.py:198` | `test_c26_W1.py` |
| `TRADER_YF_WINDOW_SLICE` | `data_sources.py:20` | `0 (OFF)` | model | slices the yfinance response to the requested window (data_sources.py:196) | `data_sources.py` | `test_c26_T2.py` |
| `TRADER_RAW_SIDECAR` | `data_utils.py:36` | `0 (OFF)` | ops | harvest writes/reads a raw-OHLCV sidecar (scripts/harvest_*_data.py) | `data_utils.py`, `scripts/harvest_stock_data.py:418`, `scripts/harvest_crypto_data.py:265` | `test_c26_T2.py` |
| `TRADER_JOURNAL_ROTATE_DAYS` | `trade_journal.py:63` | `'0' (OFF)` | ops | gzip day-files older than N days (trade_journal.py:111,152,159) | `trade_journal.py` | `test_c26_T7.py` |
| `TRADER_ORDER_STREAM` | `order_stream.py:97` | `unset (OFF)` | ops | starts the alpaca-py TradingStream order-event listener | `order_stream.py` | `test_review_b02.py` |
| `TRADER_USE_ALPACA_PY` | `trading_utils.py:88` | `unset (legacy SDK)` | ops | forces the alpaca-py CompatREST adapter instead of alpaca-trade-api | `trading_utils.py` | **none** |
| `TRADER_BOTS_OPS` | `run_bots.py:58` | `'1' (ON)` | ops | ops thread in standalone run_bots.py: Telegram kill-switch polling + daily drif… | `run_bots.py` | **none** |
| `TRADER_EOD_DIGEST` | `run_pipeline.py:118` | `'1' (ON)` | ops | once-daily EOD digest notification (run_pipeline.py:175) | `run_pipeline.py` | `test_c26_T7.py (attribute)` |
| `TRADER_EOD_DIGEST_HOUR` | `run_pipeline.py:120` | `'16'` | ops | local hour after which the EOD digest may fire (run_pipeline.py:178) | `run_pipeline.py` | `test_c26_T7.py (attribute)` |
| `TRADER_SHADOW_MODE` | `run_pipeline.py:1016` | `'1' (ON) at run_pipeline; UNSET==OFF at gui…` | gate | weekly retrain saves into the CHALLENGER slot instead of promoting immediately | `run_pipeline.py:1016`, `gui.py:7489` | `test_c26_Q2.py`, `test_c26_X1.py` |
| `TRADER_FIXED_HOLDOUT_DAYS` | `scripts/hypersearch_v2.py:350` | `unset -> strategy_config.FIXED_HOLDOUT_DAYS…` | model | fixed trailing holdout span in days (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_holdout_boundary.py` |
| `TRADER_TRAINER_SEED` | `scripts/hypersearch_v2.py:533` | `unset -> strategy_config.TRAINER_SEED (None)` | model | base training seed (overrides the config constant) | `scripts/hypersearch_v2.py` | `test_r2c_training_repairs.py` |
| `TRADER_MINUTE_EDGE_DAYS` | `scripts/harvest_stock_data.py:89` | `'120'` | model | trailing window of 1-min bars fetched for the per-day EDGE stamp | `scripts/harvest_stock_data.py` | **none** |
| `TRADER_WEBHOOK_URL` | `notify.py:77` | `unset (channel off)` | ops | Discord/Slack-style webhook target for notify() | `notify.py:77,249`, `gui.py:7459,7707` | `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_TELEGRAM_BOT_TOKEN` | `notify.py:80` | `unset (channel off)` | ops | Telegram bot token for alerts + the /halt kill switch | `notify.py:80,195,250`, `gui.py:7460,7709` | `test_notify.py`, `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_TELEGRAM_CHAT_ID` | `notify.py:81` | `unset (channel off)` | ops | Telegram chat id; also the authorization check for inbound commands (notify.py:… | `notify.py:81,196,251`, `gui.py:7461,7710` | `test_notify.py`, `test_c26_P2.py`, `test_review_b18.py` |
| `TRADER_HEALTHCHECK_URL` | `notify.py:110` | `unset (no ping)` | ops | generic healthcheck ping URL (heartbeat) | `notify.py:110`, `gui.py:7462` | `test_review_b18.py` |
| `TRADER_HEALTHCHECK_URL_{NAME}` | `notify.py:109` | `unset -> falls back to TRADER_HEALTHCHECK_URL` | ops | per-book healthcheck URL; f-string over name.upper() (CRYPTO / STOCK) | `notify.py:109`, `gui.py:7463-7464` | **none** |
| `<NAME>_API_KEY` | `llm_client.py:404` | `''` | ops | per-endpoint key convention: endpoint['name'].upper()+'_API_KEY' (e.g. OPENROUT… | `llm_client.py:404`, `scripts/llm_qualify.py:393` | **none** |

### 5.2 Everything else

| Env var | First read at | Default when unset | Facing | What it does | Readers | Test |
|---|---|---|---|---|---|---|
| `ALPACA_API_KEY` | `trading_utils.py:73` | `none (loud error)` | ops | Alpaca broker/data credential | `trading_utils.py:73`, `order_stream.py:105`, `gui.py:10501` +3 | `test_review_b02.py`, `test_review_b03.py` |
| `ALPACA_API_SECRET` | `trading_utils.py:74` | `none (loud error)` | ops | Alpaca broker/data credential | `trading_utils.py:74`, `order_stream.py:106`, `gui.py:10502` +3 | `test_review_b02.py`, `test_review_b03.py` |
| `ALPACA_BASE_URL` | `trading_utils.py:75` | `none (loud WARNING)` | ops | paper vs live endpoint; unset makes BOTH SDKs default to LIVE (trading_utils.py… | `trading_utils.py:75`, `order_stream.py:111` | `test_review_b02.py`, `test_review_b03.py` |
| `FINNHUB_API_KEY` | `sentiment.py:39` | `none -> feature silently disabled` | ops | Finnhub news + earnings-calendar client | `sentiment.py:39`, `sentiment_history.py:274`, `events_calendar.py:108` | **none** |
| `ANTHROPIC_API_KEY` | `llm_client.py:1041` | `''` | ops | fallback for llm_config models.claude.api_key | `llm_client.py` | `test_c26_S2.py`, `test_llm_claude.py`, `test_llm_providers.py` |
| `OPENAI_API_KEY` | `llm_client.py:1048` | `''` | ops | fallback for llm_config models.openai.api_key | `llm_client.py` | `test_llm_providers.py` |
| `CUDA_VISIBLE_DEVICES` | `predict_now.py:19` | `None (unset)` | ops | '' means CPU-only: predict_now caps torch threads, hw_monitor short-circuits GP… | `predict_now.py:19`, `hw_monitor.py:137`, `run_bots.py:42 (setdefault)` +1 | `test_hw_monitor.py` |
| `TORCH_NUM_THREADS` | `predict_now.py:20` | `'2'` | ops | torch intra-op thread cap for bot inference | `predict_now.py:20`, `run_bots.py:43 (setdefault)`, `run_pipeline.py:292 (BOT_ENV)` | **none** |
| `OMP_NUM_THREADS` | `run_bots.py:44` | `'2'` | ops | OpenMP thread cap (set only, never read by repo code) | `run_bots.py:44`, `run_pipeline.py:292` | **none** |
| `LD_LIBRARY_PATH` | `run_pipeline.py:282` | `''` | ops | prefixed with Jetson CUDA/cusparselt lib dirs for child processes | `run_pipeline.py:282`, `gui.py:108` | **none** |
| `LD_PRELOAD` | `run_pipeline.py:284` | `set only` | ops | Jetson libstdc++ preload for child processes | `run_pipeline.py` | **none** |
| `PYTHONUNBUFFERED` | `run_pipeline.py:285` | `'1'` | ops | unbuffered child-process logging | `run_pipeline.py` | **none** |
| `NOTIFY_SOCKET` | `run_pipeline.py:64` | `unset (no-op)` | ops | systemd Type=notify watchdog socket | `run_pipeline.py` | **none** |
| `PYTORCH_CUDA_ALLOC_CONF` | `scripts/hypersearch_v2.py:42` | `'expandable_segments:True' (setdefault)` | ops | CUDA allocator config for training | `scripts/hypersearch_v2.py` | **none** |
| `APCA_RETRY_WAIT` | `scripts/harvest_crypto_data.py:64` | `'10' (setdefault)` | ops | Alpaca SDK internal retry backoff | `scripts/harvest_crypto_data.py:64`, `scripts/harvest_stock_data.py:60` | **none** |
| `APCA_RETRY_MAX` | `scripts/harvest_crypto_data.py:65` | `'5' (setdefault)` | ops | Alpaca SDK internal retry count | `scripts/harvest_crypto_data.py:65`, `scripts/harvest_stock_data.py:61` | **none** |
| `PY_COLORS` | `scripts/ab_check.sh:29` | `forced to '0' (exported)` | ops | disables pytest ANSI so the FAILED/ERROR name greps cannot be blinded | `scripts/ab_check.sh` | **none** |
| `AB_CHECK_TIMEOUT_S` | `scripts/ab_check.sh:20` | `900` | ops | hard wall for the full pytest run | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `AB_CHECK_RERUN_TIMEOUT_S` | `scripts/ab_check.sh:21` | `300` | ops | wall for the targeted flaky/persistent rerun | `scripts/ab_check.sh` | **none** |
| `AB_CHECK_MIN_PASSED` | `scripts/ab_check.sh:22` | `1500` | ops | launch-sanity floor: fewer passed => refuse to report PASS | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `AB_CHECK_PYTEST` | `scripts/ab_check.sh:23` | `'python3 -m pytest'` | ops | pytest invocation override, only so the gate itself is testable | `scripts/ab_check.sh` | `test_imports_v3.py` |
| `TMPDIR` | `scripts/ab_check.sh:41` | `/tmp` | ops | mktemp location for the ab_check work files | `scripts/ab_check.sh` | **none** |
| `SUDO_USER` | `scripts/setup_jetson_system.sh:155` | `$(whoami)` | ops | systemd unit User= for the Jetson install | `scripts/setup_jetson_system.sh` | **none** |

Names that look like env vars but are **not**: `TRADER_DIR` / `TRADER_USER` are shell locals in
`scripts/setup_jetson_system.sh:154-155` and `scripts/backup_state.sh:15`; `TRADER_LOG_ROLE` exists
only as a proposal inside `research/module_review_2026-07.json`. **There is no `GEMINI_API_KEY`
env var** — the Gemini key comes only from `llm_config.json` `models.gemini.api_key`
(`llm_client.py:367,454,477,485,528,854,1494`).

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
| `fees.FLAT_SPREAD_PCT` vs `backtest.SPREAD_PCT` | `fees.py:55` (canonical) / `backtest.py:88` (verbatim copy) | `{'crypto':0.10,'stock':0.05}` | ✅ `tests/test_review_b10.py:154-182` regex-reads `backtest.py` and asserts equality |
| `cooldown_bars = max(1, ceil(policy['cooldown_min']/60))` | derived verbatim twice: `backtest.py:345`, `meta_label.py:738` | — | ✅ `tests/test_improve_stratcfg.py:156-167` asserts exactly one verbatim occurrence in each |
| `risk_budget.ACCOUNT_RISK_CAP` | `risk_budget.py:49` | `0.03` | ✗ account-wide stop-risk cap; the per-book `MAX_BOOK_RISK_PCT` is in `strategy_config` |
| `portfolio.MAX_AVG_CORRELATION` | `portfolio.py:31` | `0.7` | ✗ admission bar on avg pairwise correlation; unrelated to `strategy_config.CORR_SANITY_MAX` (`0.85`) despite doing a similar job |
| `blend_fit.DEFAULT_LSTM_WEIGHT` | `blend_fit.py:29` | `0.6` | ✗ `strategy_config.py:205` merely *describes* it as "the hardcoded 0.6 default" |
| `CORR_SANITY_MAX` re-typed as an import-failure fallback | `base_loop.py:2950`, `stock_loop.py:978` (both `CORR_SANITY_MAX = 0.85`) vs `strategy_config.py:487` | `0.85` ×3 | ✗ changing the config value silently diverges from both fallbacks |
| Holdout fraction `0.12` — **three independent spellings** | `meta_label.py:60` `HOLDOUT_FRACTION = 0.12`; `scripts/hypersearch_v2.py:328` `HOLDOUT_FRACTION = 0.12`; `adaptive_config.py:239` `holdout_span_days: float = 43.8` (= 0.12·365) | — | partial: `tests/test_r2c_holdout_boundary.py:217` regex-pins the hypersearch literal only |

Other policy-shaped constants outside `strategy_config` (inventory, not defects):
`fees.py:41-63` (fee/spread/edge-multiple table), `validation.py:41` `DSR_MIN = 0.6` +
`validation.py:47` `MAX_CSCV_SPLITS`, `meta_label.py:58-93,145-147` (veto prob 0.30, threshold
fraction 0.5, guard rails, OOF starvation tiers), `shadow.py:75-94` (shadow window, promote
p-values, DM-v2 constants), `backtest.py:86` `BARS_PER_YEAR`, `retrain_ledger.py:35-38`,
`naive_baseline.py:28-29`, `horizon_transfer.py:28`, `serving_cache.py:30`,
`liquidity.py:253-258` (crypto spread tiers).

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
   — several campaign flags are read only this way (`backtest.py:423`, `run_pipeline.py:1020`,
   `meta_label.py:854-855`, `events_calendar.py:57`).
4. **Env vars:** regex `os\.(environ\.get|getenv|environ\[)` over `*.py` + `*.sh`, then read each
   hit's default and comparison to classify parsing rule and polarity.
5. **Status column:** a name whose reader set is empty or tests-only is `DECLARED-AHEAD`; confirm by
   checking whether the *module* that would read it has any production importer
   (`grep -rl '^\(from\|import\) <module>' --include='*.py' . | grep -v tests/`).

Cross-check any rewrite against `research/campaign_2026-08/03_jetson_runbook.md` (activation
sequence) and `research/KILL_LIST.md` (some dark flags are kill-adjacent and need an owner ruling
before activation, e.g. the crypto spread-stamp family — `liquidity.py:61-66`).
