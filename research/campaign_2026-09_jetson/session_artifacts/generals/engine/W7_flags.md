# W7 — ENGINE R2 flags (O8 staleness flag, BARS_PER_YEAR lockstep, 24 h census) — 2026-09-27
## LANDED (pre-edit copies + mutation dir: w7/orig/, w7/mut/)
- strategy_config.py:371-387 NEW `CRYPTO_QUOTE_MAX_AGE_SEC = None` (house-pattern comment: effect, evidence, flip only via this proposal, move census LIVE_MAX_AGE_S in the same change). :362-368 BARS_PER_YEAR_MEASURED comment rewritten (volatility.py now covered — was "NOT covered"; the constant is unchanged).
- order_utils.py:116-134 NEW `_quote_max_age_sec(asset_type)`: reads the flag at CALL time via getattr, crypto only. It returns the int literal 180 when the flag is None/absent, when the asset is stock, or when the value is invalid (str, bool, NaN/inf, ≤0). It never raises, because a raise inside the staleness try-block would skip the check (fail-open). :185 `age > 180` → `age > _quote_max_age_sec(asset_type)`.
- bars_calendar.py:55-80 (SIGNAL's shared helper) NEW `bars_per_day(asset_type, legacy, default)` + `LEGACY_BARS_PER_DAY`, `TRADING_DAYS_PER_YEAR={'crypto':365,'stock':252}`. ON = MEASURED_BARS_PER_YEAR/days → stock 3827/252 = 15.1865, crypto 24.0 exactly. It is derived, not a second copy, so bpy == bpd×days holds exactly in both states. test_sig_r1_bpy.py: 9/9 still pass.
- volatility.py:145-164 NEW `_bars_per_year/_bars_per_day` (call-time bars_calendar routing; ImportError → the legacy `.get`). :236 HAR per-bar divisor and :756 vol-target read sites rerouted. Legacy dicts :141-142 kept, so the source-text pins in b17 / portfolio_v3 / sig_r1_bpy still hold. The measured numbers do not appear in volatility.py (test-pinned). Lockstep consequence (proved by test): the HAR-sourced vol ratio = annual/(√252·σ_daily) is calendar-invariant under ON. Only GARCH-sourced per-bar σ re-scales (×√(1638/3827) = 0.654 on the stock target). The per-bar σ has one live consumer: compute_vol_adjusted_size (base_loop.py:2426). Position.garch_sigma (:3353) has no reader.
- docs/FLAGS.md:84 BPY row: volatility.py and the new test added. :85 NEW CRYPTO_QUOTE_MAX_AGE_SEC row. The table header count was already stale (41 labelled vs 43 rows); it now reads (44).
- NEW tests/test_engine_r2_flags.py (52 tests). OFF pins, run against the real get_crypto_quote with a stub api at 179/181/299/301 s:
  - flag None/absent: 180 behaviour.
  - flag 300: flips only at 301.
  - stocks stay at 180 when the flag is on.
  - invalid values fall back to 180.
  - the flag is read at call time.
  vol_adjusted OFF matches the verbatim legacy body on a 28-σ grid × 3 assets. The HAR OFF path equals the legacy-expression fallback. ON: stock target = annual/√3827; HAR σ scales by √(6.5/(3827/252)); crypto is byte-identical; the invariance test above passes.
  Mutation check against the pre-edit copies: 28 fail (every flag/ON/helper test), 24 pass (every OFF behaviour pin — pre-edit code satisfies them, as a byte-identical OFF path should). Scratch w7/offgrid.py: 1,464 exact-equal comparisons of pre-edit vs post-edit (HAR ± shrink/c_scale on 40 seeds; vol_adjusted on a 204-σ grid).
## FLIP PROPOSAL — O8 (owner; exits/entries-facing, runbook Phase 2 §6 "execution set", no retrain)
- Candidates:
  - 300: in W4's 10 min census, all six names at 0 % None (ETH 15 %→0). In W3's Sunday probe, ETH 0/24 (max 294.7 s) but SOL still ≥12/24 None (p50 age 370.6 s).
  - 600: both samples 0 % on every name (max seen 486.7 s).
  - 240 would already clear W4 (ETH max 227 s) but not W3.
- Instrument: the running 24 h census → `$JPY scripts/crypto_quote_staleness_census.py --replay w7/census_24h_NN.json --thresholds 180,240,300,600,900`. Also scratch `w7/flip_rule.py census_24h_*.json`, which evaluates the rule over all 24 files together (validated on W4's JSON → "insufficient sample, keep None").
- PRE-REGISTERED RULE: T* = the smallest T in {240, 300, 600, 900} such that EVERY name meets all of the following. Otherwise keep None (180).
  (1) None-rate at T < 1 %, and longest None streak ≤ 2 polls (60 s without exits).
  (2) p90 spread of admitted-stale samples (180 < age ≤ T) ≤ fresh p90 + 5 bps. I use p90 rather than p99 because each name's stale bucket has ~10² samples, so p99 would just be its max. 5 bps is set just above W4's widest within-name fresh p50→p90 dispersion (XRP 4.5; others 0.8–3.2), so a larger gap means the stale quotes are systematically wider, i.e. the book is abandoned rather than quiet.
  (3) Median |Δmid| at the first update after a >180 s silence ≤ 2× the median after a ≤180 s silence. This tests W3's "stamped only on touch change" inference. W4: ETH long 1.91 bps vs short 2.63 bps (n=2, supportive only).
  Sample requirement: ≥ 24 distinct UTC hours, with ≥ 6 weekday and ≥ 6 weekend hours. This run (Sun 06:16Z → Mon ~06:20Z) gives ≈18 weekend + 6 weekday hours. Expected outcome from current evidence: 600 (300 fails SOL's Sunday stretch). With the flip, the resting GTC stop_limit still guards the hard stop.
## FOUND-NOT-FIXED
- order_utils.py:176-190 fail-open, REPORTED ONLY. With the real SDK entity (alpaca_trade_api `_Timestamped` → `pd.Timestamp(raw, tz=NY)`), a raw `t` of `null`, `''` or `'garbage'` gives a quote accepted as FRESH (probe run in the jetson env; `'2020-…Z'` → None). Mechanism: NaT compares False, or ValueError hits `except: pass`. A missing `t` also skips the check (`if qt is not None`).
  Why I did not fix it:
  - Tests pin it: tests/test_crypto_quote_staleness_census.py:70 (`unparseable_ts` → 'ok', parity-asserted against the real function at :87-92); test_ia2_safety.py:19 ("absent timestamps fail safe (accepted)"); test_order_utils_v3.py:173 stubs `t=None`.
  - It has never been observed live: W4 had 0 disagreements and every age parsed.
  - "Fail-closed" here means None, which SKIPS exits. That is not obviously safer on the exit path, so it is an owner decision.
- On any flip, scripts/crypto_quote_staleness_census.py:50 `LIVE_MAX_AGE_S = 180.0` must follow. test_live_constant_matches_real_get_crypto_quote (:48-53) will fail if strategy_config is edited without it — a useful tripwire.
- Pre-existing drift (not mine): docs/FLAGS.md `strategy_config.py:NNN` refs after :365 were already ~35 lines off (e.g. PREDICTION_CACHE_ENABLED cited :640, actually :675 pre-edit). My block adds another +22.
## CENSUS (Item 3)
- The script has no checkpointing, so it runs as 24 × 1 h runs in a loop: w7/census_loop.sh → w7/census_24h_00..23.json, log w7/census_24h.log, `--thresholds 180,240,300,600,900`, nice 15, setsid+nohup, started 06:16:27Z. Loop PID 115632 (session leader; kill that to stop). First python child PID 115635; a new child starts each hour. RSS 105 MB, flat at 9 m 45 s. Poll 10 already shows an ETH None.
## VERIFIED-CLEAN
- The real SDK `q.t` (RFC-3339 ns) parses to a tz-aware NY pd.Timestamp, and get_quote's UTC age is correct (jetson probe). The only BARS_PER_YEAR/BARS_PER_DAY reads of volatility's dicts are :236/:756 (repo grep); test_new_modules.py:105 only imports the dict for its crypto expectation.
## TEST RUNS (each via hwlock heavy, nice 10, CUDA_VISIBLE_DEVICES='', one file per process)
- py_compile of all 5 touched .py files: OK. test_engine_r2_flags 52 passed. test_order_utils 16, test_order_utils_v3 40, test_sig_r1_bpy 9, test_review_b17 24, test_portfolio_backtest_v3 31, test_c26_T7 37, test_crypto_quote_staleness_census 26, test_review_b02 27, test_ia2_safety 24, test_c26_W1 36, test_wave4 14, test_new_modules 36, test_grp_macro 13, test_g5_fixes_2026_09 59+2 skipped, test_c26_T6 43, test_c26_R2 23, test_improve_stratcfg 10, test_execution_policy_v3 33 — all passed, 0 failed. No test_volatility*/test_har* files exist.
