# Campaign 2026-08 — Phase 6: Signal-Model Plan (R2-B adjudication)

**Produced:** 2026-08-20 by the R2-B adjudicator, from three independent lens reviews of the
signal path (model_v2.py, model_lgb.py, blend_fit.py, scripts/hypersearch_v2.py, predict_now.py
and their feature/gate interfaces). Every defect claim below was re-verified line-by-line against
the current tree before adjudication; nothing was edited (read-only pass). Binding inputs:
`research/KILL_LIST.md` (including the pending-asks appendix), the frontier synthesis
`05_frontier_research.md` (FR-01..FR-20, negatives N1–N10), and `02_research.md` B12/B03.
Kill-list compliance was checked on every item: no Mamba/KAN/TSFM/E2E-Sharpe/IPCA, no conformal
wrap on any quantile work, no online blend weights, sLSTM only in the capped FR-15 cell form.

**Headline findings.** The panel that model_v2.py and model_lgb.py never received found the model
files themselves clean (the 42-line LSTM+MHA stack is internally consistent; LightGBM's
early-stopping usage is correct) — but the SEAMS around them carry three high-severity defects:
(H3) the live champion serves a NEW LSTM blended with the PREVIOUS week's LGB/q10 boosters for up
to a week after every legacy retrain; (H1) the HYPERSEARCH_V3 blend-fit failure path re-opens D25
by certifying a raw LSTM while shipping boosters that live blends at 0.6; and (H2) the
trade_threshold is selected on raw-LSTM predictions but served (and, under V3, certified) against
the blend. All three are gate/serving-integrity defects, exactly the class this system's
promotion discipline exists to prevent.

---

## 1. Verified defects and dispositions

All 16 distinct claims (after merging duplicates across lenses) were CONFIRMED against the
current tree. None were dropped. Dispositions: FIX-NOW (packet), FIX-AT-CHAIN-2 (flagged rider),
MEASURE-FIRST (probe before any code flip), or GUARD/DOCUMENT.

### High severity

**H1 — Blend-fit failure path deploys an uncertified predictor (re-opens D25 through the V3
flag-ON path).** Verified: `evaluate_on_holdout` blends only when `lgb_booster is not None and
lstm_weight is not None` (hypersearch_v2.py:1396); both the blend-fit exception path
(:2111-2114) and the "OOF arrays unavailable" path (:2108-2110) leave `lstm_weight = None`; yet
`extra_artifacts` ships the boosters on `_v3 and lgb_pack and lgb_pack[0] is not None` alone
(:2222-2238) and `config['lstm_weight']` is written only when non-None (:2176-2177), so
predict_now.py:447 and backtest.py (~:219, :834) serve a 0.6/0.4 blend plus a live q10 veto the
certificate never scored. This violates the V3 spec's own stated invariant ("the certified
predictor IS the deployed predictor", strategy_config.py:214-215). Latent today (HYPERSEARCH_V3
defaults False) but MUST land before the Chain-2 activation flips the flag. **Disposition:
FIX-NOW, packet R2C-02.**

**H2 — trade_threshold scale mismatch: selected on raw-LSTM predictions, deployed on the
blend.** Verified: the threshold is suggested at :649-656 and scored against LSTM-only fold
predictions (:875-879, :936-939), written unchanged into the config (:2166), then applied to
`w*LSTM+(1-w)*LGB` at the holdout gate (:1413-1434) and in serving (predict_now.py:446-482) and
the replay (backtest.py:219, :346). Tree-regression predictions are variance-compressed relative
to the LSTM leg, so the blended prediction distribution under the searched threshold differs
systematically from what every trial score measured. Legacy (flag OFF, live today) is worse:
threshold selected LSTM-only, served against the hardcoded 0.6 blend with no blend certificate
at all. **Disposition: FIX at the V3 save path (blend-consistent threshold re-selection behind a
default-OFF sub-flag), packet R2C-02; rides Chain-2's study reset.**

**H3 — Legacy champion hot-reload race: the live book serves a NEW LSTM with LAST WEEK's LGB and
q10 boosters (and floor) after every retrain.** Verified end-to-end: under the current default
flags the legacy flow writes the manifest first and trains the LGB legs minutes later
(hypersearch_v2.py:2240-2254); the bot hot-reloads on manifest mtime and pops the per-prefix
booster caches at that moment (base_loop.py:709-716); predict_now's caches are presence-keyed
(`if pfx not in _lgb_models`, predict_now.py:429, :452) with a permanent cache-None-on-failure
hazard (:433-434), so the first ~30s prediction cycle after reload loads the OLD booster files
still on disk and no further invalidation event exists until the next retrain's manifest. The
stale booster was also trained against the previous scaler. shadow.py:220-233 patched exactly
this race for the CHALLENGER slot only — proof the mechanism is real and the champion repair was
missed. The V3 extra_artifacts path (boosters written before the manifest,
hypersearch_v2.py:1650-1656, 1686-1689) closes the race only once V3 activates. **Disposition:
FIX-NOW as a serving-only repair (mtime-keyed caches), packet R2C-01. This is the highest-value
live fix in the wave.**

### Medium severity

**M1 — One-bar entry-timing skew between live serving and training/backtest.** Verified:
training/holdout/backtest windows use `offsets = np.arange(-seq_len, 0)` (hypersearch_v2.py:727,
1225, 1374; backtest.py:202) with the label at row i anchored Close[i]→Close[i+fb]
(harvest_crypto_data.py:195-197, harvest_stock_data.py:214-216) — the window excludes the entry
bar. Live drops only the forming bar (predict_now.py:209) and slices
`current_features[-seq_len:]` (:417), so the window INCLUDES the last closed bar and entry
happens ~immediately; mapped into training space that window belongs to row T+1 whose entry
anchor is the forming bar's close, up to an hour away. The parity comment at predict_now.py:204-209
claims equivalence and is wrong. Every offline certificate therefore scores a one-bar-delayed
entry live does not execute. **Disposition: MEASURE-FIRST — packet R2C-06 ships the probe (score
existing stage0 predictions against both label anchors + journal fill-gap join); the offsets flip
(`WINDOW_INCLUDES_ENTRY_BAR`, moving OFFLINE to match LIVE, labels untouched) is DEFERRED to
Chain-2 and only on a material probe delta. Either way the false comment gets corrected.**

**M2 — The deployed blend weight is fitted on the WRONG LSTM (and mixed scalers).** Verified:
the fit consumes the fold-souped trial checkpoint's val predictions (`oof_preds[-1]`,
hypersearch_v2.py:2049-2050, cached at :943-944 under the fold scaler) while the shipped LSTM is
the final-refit SWA state under a refit scaler (:2015-2021, scaler at :1221); lgb_oof is
meanwhile computed with ship_scaler (:2055-2062), so the two legs' fit inputs are not even under
one scaler, and the LGB side's early stopping ran on the very rows the weight is fitted on. The
certificate (scoring refit-LSTM + w) catches only gross failure. **Disposition: FIX at the V3
path — one batched inference pass of ship_state/ship_scaler over folds[-1] val rows, log stale-w
vs refit-w side-by-side, deployment source switch behind a default-OFF sub-flag. Packet R2C-02.**

**M3 — D22's recency fix is asymmetric: the LGB mean leg and q10 veto still train only on
folds[-1] train (ending at the 0.85 quantile of the search region) while the LSTM refits on all
pre-holdout data.** Verified: `train_idx, val_idx = folds[-1]` (hypersearch_v2.py:1093), fold-2
train_end_pct = 0.85 (:359); final_refit retrains only the LSTM (:1181-1328); the q10 floor is
calibrated on the truncated fold's val slice (:1155-1156). The leg the repo's own comment calls
"the stronger learner at this data size" deploys blind to the newest ~15% of the window every
retrain. **Disposition: FIX-NOW behind default-OFF `LGB_REFIT_FULL` (collective-early-stopping
analog: fold training finds best_iteration, then retrain both boosters on all purged pre-holdout
rows at that fixed round count under the same byte budget), packet R2C-03.**

**M4 — q10 floor calibrated on the early-stopping slice, and HYPERSEARCH_V3/OBJECTIVE_V3 are
uncoupled in code.** Verified: floor = percentile-15 of q10.predict(X_val) on the same X_val the
booster early-stopped on (:1151-1156) — the wave-9 meta-calibration-leak pattern FR-12 forbids;
and `purge_val_labels=_objective_v3()` (:1090, :723) follows OBJECTIVE_V3 while the blend
fit/q10 calibration belong to HYPERSEARCH_V3 — the two constants are coupled only by a runbook
comment (strategy_config.py:217-224), so V3-on/objective-off would fit the deployed w and floor
partly on label windows crossing into the holdout the certificate then scores. **Disposition:
FIX-NOW — a code guard forcing purge_val_labels=True in the V3 blend-fit/LGB fold construction
(packet R2C-02) and honest floor recalibration inside LGB_REFIT_FULL (packet R2C-03).**

### Low severity (all confirmed)

**L1 — Validation loss ignores the searched huber_delta AND the |return|+1 training weights**
(`nn.functional.huber_loss(vo, yvb)` at :862 vs the training criterion :762/:835-838), so
checkpoint-soup admission, early stopping, and the refit epoch budget are selected on a
mismatched criterion and the huber_delta hyperparameter barely influences selection.
FIX-AT-CHAIN-2 behind `TRAINING_REPAIRS_V1` (packet R2C-04).

**L2 — Regime-penalty look-ahead:** the "trailing" regime series is a trailing mean of FORWARD
fb-bar returns (:504-508), so the bull/bear mask at t embeds returns through t+fb; rows 0–48
carry under-scaled partial sums; boolean-masked subsets also break simulate_trades' contiguity
assumption (:523-527). Selection-shaping only (never serves live). FIX-AT-CHAIN-2 in
`TRAINING_REPAIRS_V1` (packet R2C-04).

**L3 — No seed control anywhere in the torch training path** (verified by grep: the only 'seed'
is the noisy-ratchet Laplace draw :1969-1980; np.random.permutation at :824/:1286 and the
TPESampler at :1883 are unseeded). Blocks FR-02's "identical seed" arms and FR-13's
"differing only in seed" ensembles as specced. FIX-NOW as opt-in plumbing (`TRAINER_SEED`,
default None = legacy unseeded), packet R2C-04.

**L4 — blend_fit's significance SE corrects only temporal overlap** (n_eff = n/fb,
blend_fit.py:110-114) on pooled multi-ticker rows with cross-name residual rho 0.7–0.9 — the SE
is understated and "significant" fires too liberally. Damage bounded by shrink 0.5 + the
[0.25,0.75] clamp. FIX in packet R2C-02 (optional Kish breadth divisor, default legacy).

**L5 — The OOM probe batch performs a REAL, unclipped optimizer step** before epoch 0 in both
the fold loop (:779-791) and final_refit (:1250-1263). Negligible impact; folded into
`TRAINING_REPAIRS_V1` (packet R2C-04) since any change alters trial scores.

**L6 — Stock embargo denominated in calendar seconds** (`seq_len * EMBARGO_MULTIPLIER * 3600`,
:356, EMBARGO_MULTIPLIER=1): ~40 calendar hours ≈ 11 RTH bars for a seq_len=40 stock model, not
40 bars. The hard label purge (:366) is intact; only the serial-correlation buffer is weakened,
stock book only. FIX-AT-CHAIN-2 as a bars-denominated flag in packet R2C-04.

**L7 — TB_Bars_* positional spans are stamped BEFORE dropna + the as-of tradability mask**
(harvest_stock_data.py:214-252) against policy_exits' own documented invalidity caveat
(policy_exits.py:464-468); consumers treat spans as row offsets in the filtered frame
(hypersearch_v2.py:1470-1477; sample_weights uniqueness). Today's removals are per-ticker
prefixes (offset-preserving) but nothing enforces that. GUARD-NOW: a prefix/suffix-only
assertion in the harvest (packet R2C-06); a re-stamp-after-filtering is deferred pending the
assert's Jetson evidence.

**L8 — The indicator preset picker is dead for training:** hypersearch's `--preset` default
'stationary' (:125-126) always overrides `load_indicator_config()` (:188), run_pipeline
hardcodes '--preset stationary' (:1032, :1077), and indicator_config's default is 'standard'
(:20). Train/serve parity unaffected (predict_now reads the pickled feature_cols). FIX-NOW as a
behavior-preserving truth fix (CLI default None resolving to the config file, run_pipeline
unchanged) + docstring correction, packet R2C-04.

**L9 — blend_fit's Sharpe-grid diagnostic takes entries at `pred >= threshold`**
(blend_fit.py:25) while every deployed path uses strict `>` (objective_utils.py:67,
backtest.py:346, predict_now.py:482). Diagnostic-only today. FIX in packet R2C-02.

### Lens claims verified as NON-defects (recorded so they are not re-hunted)

The flatten ordering equals `windows.reshape(rows,-1)` on all three consumer paths; the
challenger slot's booster caches are correctly invalidated by shadow.py; LightGBM's
early_stopping ignores the training set in valid_sets; model_v2's eval() precedes every
inference and the JIT trace; the V3 LGB legs train on the shipping refit scaler (parity);
backtest.main already reads cum_trials (B03.2 landed, :1029-1048). FR-spec corrections found:
FR-14's "~40 features" is actually ~72 under the stationary preset (hypersearch_v2.py:1044-1045
byte-budget comment); FR-04's stage0 dump carries no closes (stage0_preds.py:131-150), so the
naive-baseline driver must join per-name hourly closes; FR-09's "Optuna categorical" framing is
selection-inert because the trial objective trains no LGB leg (:618-1033) — the honest form is
an explicit arm grid at the retrain event with trials counted; FR-08's "incumbent = .prev" is
wrong under --shadow (fresh artifact lands in the challenger slot; the incumbent is the
champion).

---

## 2. Build packets (R2-C implementation wave)

Rules applied: Mac-buildable and Mac-testable now (pure kernels, flags, wiring verified by
py_compile + extracted-pure-logic tests); every model-facing change is default-OFF with the
flag-OFF path byte-pinned; nothing self-activates; hypersearch_v2.py is owned by one serialized
chain. Chains: **1** = predict_now.py; **2** = scripts/hypersearch_v2.py (+ blend_fit.py,
objective_utils.py, strategy_config.py flag block); **3** = new measurement scripts +
stage0_preds.py + harvest guard; **4** = backtest.py.

**R2C-01 (chain 1) — Serving booster-cache integrity.** Key predict_now's `_lgb_models` /
`_q10_models` lazy caches on booster-file mtime (store (mtime, obj); reload on change) and stop
permanently caching None on transient failure (retry with backoff). Restores the intended
new-stack-together semantics the shadow slot already has; closes H3 for the champion under the
current default flags. Tests: stub files with changing mtimes; None-eviction; challenger-slot
behavior unchanged. Gate: ab_check clean; Jetson observation — one legacy retrain cycle's log
shows the champion picking up the fresh boosters when they land.

**R2C-02 (chain 2, first) — Blend/certificate coherence in the V3 save path.** (a) H1 parity:
`lstm_weight_eff = lstm_weight if lstm_weight is not None else 0.6` used by BOTH
evaluate_on_holdout and config whenever boosters will ship (cert==deploy always), plus a
failure-path unit test with stub boosters; resolves the pending B12.2 0.6→0.5 default question
into a certificate-visible constant (owner note). (b) M4 guard: force purge_val_labels=True for
the V3 blend-fit/LGB fold construction when `_hypersearch_v3() and not _objective_v3()`, loudly.
(c) M2: re-predict folds[-1] val rows with ship_state/ship_scaler and log the refit-based blend
fit next to the stale one; deployment source switches behind default-OFF `BLEND_FIT_ON_REFIT`.
(d) H2: re-select trade_threshold on the BLENDED folds[-1] val predictions over the
v3_trade_threshold_range grid behind default-OFF `BLEND_THRESHOLD_RESELECT`, logging old/new
n_trades. (e) L4/L9 in blend_fit.py: optional Kish cross-sectional SE divisor (default legacy)
and `>=`→`>` in _policy_sharpe. Gate: flag-OFF byte-identity pinned; the failure-path
cert==deploy test; Jetson identical-holdout A/B for (c)/(d) — adopt only on non-inferior blend
DSR with n_trades moving toward the trial-scored frequency; sub-flags ride Chain-2.

**R2C-03 (chain 2) — LGB full refit + honest q10 floor.** Under default-OFF `LGB_REFIT_FULL` in
train_lgb_ensemble: fold training supplies best_iteration for mean and q10; retrain both on all
purged pre-holdout rows (final_refit's purge, same LGB_MAX_ROWS/byte-budget cap most-recent-first)
at the fixed round count with no early stopping; recompute the floor as percentile-15 of the
refit q10's predictions on the original fold-val rows, with the in-sample caveat recorded in the
q10 meta and coverage verified downstream. Return through the existing save=False tuple; the
atomic-save path is unchanged. Gate: pure numpy tests for the cap/purge index math; flag-OFF
pinned; Jetson identical-holdout blend-DSR A/B (fold-LGB vs refit-LGB, same LSTM, same
cum_trials) + q10 holdout coverage 10% ± 3pp via scripts/reliability_report.py; then
backtest.py --gate; challenger→shadow.

**R2C-04 (chain 2) — Training-loop repairs + seed plumbing + config truth.** (a) `TRAINER_SEED`
(config/env, default None = legacy unseeded): torch.manual_seed + np.random.default_rng-derived
permutations threaded through _train_walk_forward and final_refit(seed=), seeds derived from
(study_name, trial.number, fold/k) — the prerequisite for FR-02 and FR-13. (b) Default-OFF
`TRAINING_REPAIRS_V1` bundling the selection-shaping repairs that change trial scores: validation
loss computed with the trial's criterion (delta + |return|+1 weights) (L1); regime series lagged
so the mask at t uses only returns completed by t, with proper warmup, via a pure helper hoisted
to objective_utils for Mac tests (L2); OOM probe made side-effect-free (L5); embargo optionally
denominated in bars (L6). (c) L8: `--preset` default None resolving to load_indicator_config()
inside load_data (run_pipeline passes 'stationary' explicitly — behavior preserved) + corrected
docstrings. Gate: pure-helper unit tests (regime-lag kernel on hand-built arrays; embargo bar
counting on synthetic timestamps); flag-OFF byte-pin; TRAINING_REPAIRS_V1 rides Chain-2's single
study reset (never its own); seeded-run determinism check is a Jetson step.

**R2C-05 (chain 2) — FR-01 fixed holdout + FR-02 window A/B runner.** Hoist the boundary rule
into pure objective_utils (`holdout_boundary(all_times, fixed_days=None)`: legacy quantile when
None, else max−days·86400); get_holdout_boundary delegates, reading
strategy_config.FIXED_HOLDOUT_DAYS (default None = byte-identical) with a TRADER_FIXED_HOLDOUT_DAYS
env override; all five call sites inherit through the one choke point; instrumentation prints
both boundaries + holdout row/trade counts side-by-side (ships direct). FR-02: load_data gains
window_days masking (pure cutoff helper, Mac-tested) and scripts/window_ab.py loads the champion
config_v2.pkl and calls final_refit + train_lgb_ensemble(save=False) + the blend fit +
evaluate_on_holdout directly — no Optuna, no ratchet, no saves, identical cum_trials passed to
every arm, fixed TRAINER_SEED, per-arm stage0-schema dump of holdout predictions. Gate:
boundary/cutoff unit tests on synthetic timestamps incl. purge interaction; flag-OFF pinned;
Jetson: FR-01's dual-boundary DSR delta on the saved winner and ≥10 calendar-effective trades at
60d before FR-02 arms run.

**R2C-06 (chain 3) — Measurement kernel suite (ships direct, no flags).** (a) FR-04:
naive_baseline.py (strictly-trailing EWMA-momentum / trailing vol, hand-checked synthetic tests)
+ scripts/naive_vs_blend.py joining per-name closes to the stage0 dump (the dump lacks closes —
verified), side-by-side purged IC + DSR at n_trials=cum_trials vs 1; additive optional 'close'
field in stage0_preds.build_rows so future dumps are self-sufficient. (b) FR-07-A:
horizon_transfer.py computing rho(r^δ, r^Δ) per name/pooled vs the IID sqrt(δ/Δ) null with
weekly-block-bootstrap SEs + scripts/horizon_transfer_report.py (stock TB degeneracy caveat
printed per policy_exits.py:460-463). (c) FR-03: scripts/funding_drift_audit.py — PSI/KS +
split purged IC (2026-01 split) for the Funding_* family on HARVEST columns, flag table to
research/funding_drift_2026-08.json. (d) M1 probe: scripts/entry_timing_probe.py scoring stage0
predictions against both label anchors (overlap-adjusted, per book) + a journal join for the
realized fill-time gap. (e) L7 guard: harvest_stock_data assertion that post-stamp row removals
are per-ticker prefix/suffix only, loud warning otherwise. Gate: each kernel carries synthetic
fixtures (AR(1) transfer curve, shifted-PSI distributions, hand-computed EWMA); ab_check; the
real numbers are Jetson runs.

**R2C-07 (chain 2, last) — FR-05 rank-IC certificate lines + FR-08 retrain-gain ledger.**
FR-05: pure `cs_rank_ic(preds, y, group_ids)` kernel in objective_utils (per-timestamp rank
Pearson, groups ≥5 names, mean/SE + 4 sub-period splits); in evaluate_on_holdout keep
`lstm_preds = preds.copy()` before the blend overwrite and emit cs_rank_ic for LSTM/LGB/blend
INSIDE the `if blended:` branch only (flag-OFF report stays key-for-key identical per the D25
convention, enforced at :1629); plus a gate-time per-fold LSTM cs-IC print from the OOF arrays;
crypto's 6-name-wide CI caveat printed. FR-08: retrain_ledger.py — pure paired_scores kernel
(Mac-tested) + a Jetson-only scorer loading the incumbent stack (champion artifacts when
save_prefix != prefix, .prev on same-slot saves — the shadow-slot correction), scoring both
stacks on the trailing ~7 days of purged bars, appending capped rows to adaptive state;
one fail-soft call site after save_model_atomically. Gate: kernel property tests
(perfect rank→1.0, anti→−1.0, <5-name groups skipped; ledger append round-trip); report
key-compat test; the readouts are the measurement (FR-05 decides FR-06's fate; FR-08 needs ≥12
weekly rows then B03.3's IM block-t).

**R2C-08 (chain 4) — FR-16 breakeven fee-multiplier sweep.** backtest.py: keyword-only
`fee_mult=1.0` threaded to simulate_ticker scaling only the charged cost legs (rt_cost /
rt_cost_arr), leaving edge_floor/threshold/entries fixed so λ* measures the CURRENT policy's
cost headroom; `--fee-mult` and `--fee-sweep '1.0,1.5,2,3,4,6'` in main with per-book/per-name
λ* by zero-crossing interpolation; `ap.error` on --gate combined with either flag; the 3-arg
run_backtest positional seam untouched (test-pinned). Gate: default-path byte-identity pinned;
simulate_ticker fee_mult unit test on synthetic OHLC via the pure policy_exits fallback; λ*
interpolation unit test; the sweep itself is the metric (champion + challengers side-by-side on
the Jetson; the "challenger must not reduce λ*" acceptance rule is a documented owner
convention).

**Chain-2 execution order:** R2C-02 → R2C-03 → R2C-04 → R2C-05 → R2C-07 (serialized; each ends
with `bash scripts/ab_check.sh`).

---

## 3. Deferred (with triggers)

1. **WINDOW_INCLUDES_ENTRY_BAR offsets flip (M1 fix).** Trigger: the R2C-06 entry-timing probe
   shows a material IC delta (weekly-block-bootstrap 2·SE rule). Then a Chain-2 rider with full
   promotion path; on a null, the finding downgrades to correcting the false parity comment and
   documenting the accepted one-bar convention.
2. **FR-14 VSN front-end and FR-15 capped sLSTM cell.** Both are Chain-2 Optuna categoricals
   whose torch code cannot be executed (only py_compiled) on the Mac and whose search-space slots
   should be spent on evidence. Trigger: FR-04 and FR-05 results in hand at Chain-2 assembly
   time; build then, with FR-15's ≤25% trial-share cap, the MAX_TRIAL_SECONDS truncation-asymmetry
   check, and its pre-registered kill entry on failure.
3. **FR-13 seed-ensemble vs SWA-soup A/B (and the _SeedMean serving wrapper).** Trigger:
   TRAINER_SEED plumbing (R2C-04) verified deterministic on one Jetson retrain; build at the B12
   refit event. The third-leg diversity rule stays binding: no new blend leg until the B02
   per-leg journals measure rho(lstm_err, lgb_err).
4. **FR-12 multi-quantile head (q05/q25/q50, journal-only).** Hard prerequisite verified still
   open: the q10 veto is uncertified and its floor is early-stopping-slice-calibrated (M4).
   Trigger: R2C-02/03 landed + one Jetson reliability_report showing q10 holdout coverage inside
   10% ± 3pp. No conformal wrap ever (kill-list boundary).
5. **FR-09 decay-on-cumulative-uniqueness weights.** Strictly sequenced after FR-02; skip
   entirely on an FR-02 null. Implement as the corrected explicit arm grid at the retrain event
   (the Optuna-categorical framing is score-inert — recorded above), trials counted into
   cum_trials.
6. **FR-06 LambdaRankIC port (stock LGB leg).** Trigger: FR-05's champion-LGB cross-sectional
   rank IC near zero or negative; strongly positive ⇒ deprioritize before it spends a Chain-2
   slot.
7. **FR-07-B..E (horizon probe and exploit).** Trigger: FR-07-A curves (R2C-06) suggest
   off-diagonal structure; stage-B decision rule max_{δ<Δ} Ĵ(δ) − Ĵ(Δ) > 2·SE with probe
   trainings incremented into cum_trials.
8. **TB label re-stamp after filtering (L7 full fix).** Trigger: the R2C-06 harvest assert ever
   fires on real data (interior removals exist). Shared policy_exits/label semantics ⇒ gotcha #2
   + owner.
9. **B12.2's hardcoded default lstm_weight 0.6 → 0.5.** Owner item — it interlocks with
   R2C-02(a)'s cert-visible default; flag for the decision queue rather than changing silently.
10. **Deeper preset/GUI rewiring (L8 beyond the truth fix).** Owner call on whether the GUI
    picker should reach training at all.
11. **FR-10 news census.** Blocked on kill-list Ask A (owner ruling on the Alpaca News data
    dependency).
12. **Embargo-in-bars default flip (L6).** Code ships OFF in R2C-04; flipping changes fold
    composition ⇒ Chain-2 event only.

---

## 4. Ordered Jetson experiment sequence (what the new code enables)

Run in this order; each carries its decision rule from the FR measurement plans.

1. **Serving-integrity observation (R2C-01):** one legacy retrain cycle; confirm the champion
   log shows fresh boosters loading when their files land (no week-long stale blend). Pass =
   the race is closed; nothing further.
2. **FR-01 instrumentation (R2C-05):** per book, print quantile vs fixed-60d boundaries with
   holdout row/trade counts; re-run evaluate_on_holdout on the same saved winner under both and
   report the DSR delta. Requirement: ≥10 calendar-uniqueness effective trades at 60d before any
   window experiment.
3. **FR-04 naive baseline (R2C-06):** blend vs the Nagel one-liner on identical stage0 rows.
   Decision: blend must beat it on BOTH purged IC and DSR (cum_trials deflation vs n_trials=1);
   a within-noise result is an owner report, never an auto-action.
4. **FR-05 rank-IC certificate pass (R2C-07):** champion LGB/LSTM/blend cross-sectional rank IC.
   Decision: strongly-positive LGB rank IC ⇒ FR-06 deprioritized; near-zero/negative ⇒ the
   in-house justification for the port.
5. **FR-03 funding drift audit (R2C-06), before the next retrain:** flag any Funding_* feature
   with PSI > 0.25 or an IC sign flip with non-overlapping CIs; the flag table attaches to the
   retrain notes (a flag means the retrain re-fits on the shifted distribution — no feature
   removal; kill-list survivor #1 boundary).
6. **Entry-timing probe (R2C-06):** IC against both label anchors + realized fill-gap join.
   Decision: material delta (2·SE, weekly blocks) ⇒ schedule WINDOW_INCLUDES_ENTRY_BAR at
   Chain-2; else record the negative and fix the comment only.
7. **FR-07-A transfer curves (R2C-06):** per-book rho(r^δ, r^Δ) vs the IID null. Decision:
   flat/on-diagonal ⇒ kill the horizon topic cheaply; otherwise sequence FR-07-B's ~8 LGB
   probes (trials counted).
8. **Seed determinism check (R2C-04):** two refits at the same TRAINER_SEED byte-compare.
   Pass = FR-02 and FR-13 preconditions met.
9. **FR-02 window A/B (R2C-05), crypto book:** full / 2Y / 1Y / 18-month PT-2007 arms at
   identical cum_trials, fixed seed, FR-01 fixed holdout. Decision: adopt shorter ONLY on both
   the fold-objective AND holdout-DSR wins, winner through backtest.py --prefix '' --days 60
   --gate; else keep full history and record the durable negative (which also skips FR-09).
10. **LGB_REFIT_FULL A/B (R2C-03):** identical-holdout blend DSR fold-LGB vs refit-LGB (same
    LSTM, same cum_trials) + q10 coverage 10% ± 3pp via reliability_report. Decision:
    non-inferior DSR + sane coverage ⇒ backtest --gate ⇒ challenger→shadow; else stays OFF.
11. **Blend-coherence A/B (R2C-02 sub-flags):** one cycle logging stale-w vs refit-w and
    searched vs blend-reselected threshold side-by-side. Decision: flip BLEND_FIT_ON_REFIT /
    BLEND_THRESHOLD_RESELECT only on non-inferior identical-holdout blend DSR with the
    certificate's n_trades moving toward the trial-scored trade frequency.
12. **FR-16 fee sweep (R2C-08) + FR-08 ledger accumulation (R2C-07):** λ* per book/name at
    --days 180 becomes the standing challenger acceptance metric ("must not reduce λ*"); the
    retrain ledger accumulates ≥12 weekly paired rows, then the B03.3 IM block-t decides the
    cadence question — an owner decision, never automatic.

*End of Phase 6 plan. The three lens transcripts remain the detailed record; this file is the
owner surface for the R2-C build wave.*
