# SIGNAL research log (append-only)

Scope: research & training path (hypersearch_v2 / model_v2 / model_lgb / blend_fit / objective_utils /
validation / sample_weights / calibration / meta_label / backtest / policy_exits). Entries are dated and
never edited in place — later rounds append. Nothing here is implemented; every model-facing item rides
default-OFF flag -> challenger -> shadow (DM-HLN). research/KILL_LIST.md is binding.

## 2026-09-27 R1 scout brief (SCOUT-1)

Method: web survey (WebSearch/WebFetch, 2024-12 .. 2026-09) filtered against 05_frontier_research.md
(FR-01..FR-20, N1..N10) and 06_signal_model_plan.md §4 so nothing already surveyed is re-sold. Code
facts cite file:line in the live tree (2026-09-27). Two numeric checks were run on the Jetson
(pure numpy/scipy, <100 MB, <5 s; scripts in the session scratchpad `scout1/`): (i) repo DSR vs the
2025 Lopez de Prado-Lipton-Zoonekynd formulas, (ii) the holdout gate's power table. Labels:
OBJECTIVE = one correct answer; JUDGMENT = two good engineers could disagree (=> owner item).

### Ranking (expected value / cost on the 8 GB Jetson)

| Rank | ID | Item | Ship mode | Cost | EV |
|---|---|---|---|---|---|
| 1 | S1-01 | LGB leg is byte-budget starved: trains on ~3–45% of rows | step 0 measure; then default-OFF `LGB_LAG_SUBSET` | M | high |
| 2 | S1-02 | Holdout DSR gate power is ~0–15% for realistic edges | measurement-only `gate_power` block; OWNER | S | high |
| 3 | S1-03 | Optuna 5.0 (2026-09-07) silently flips TPE defaults; `optuna>=4.0` unpinned | pin / explicit kwargs (hygiene) | XS | med-high |
| 4 | S1-04 | Meta calibrator: Beta calibration arm (Venn-Abers = kill-adjacent ask) | default-OFF sub-flag | S | medium |
| 5 | S1-05 | Is the LSTM leg (and torch in the bots) paying for its RAM? | measurement-only, then OWNER | S | medium |
| 6 | S1-06 | LightGBM quantized-gradient training (up to ~2x CPU) | default-OFF `LGB_QUANT_GRAD` | S | medium |
| 7 | S1-07 | CPCV/PBO audit of the search's winner selection | measurement-only CLI | L | medium |
| 8 | S1-08 | Joint window x complexity (ATOMS) as a lens on FR-02 | FR-02 extension, report-only | M | low-med |
| 9 | S1-09 | Agentic-discovery trial ledger (the campaign itself is a search) | measurement-only | XS | low-med |
| 10 | S1-10 | TabICLv2 license verified clean; TFMs still lose under temporal shift | FR-11 arm only | — | low |
| — | S1-11..16 | Checked, not adopted (incl. one kill corroboration) | — | — | — |

---

### S1-01 — The LightGBM leg trains on a small, most-recent slice because of the flatten byte budget (OBJECTIVE fact, JUDGMENT remedy)

**Code fact.** `train_lgb_ensemble` flattens the full window (seq_len x n_features float32) and caps rows at
`min(LGB_MAX_ROWS=120_000, max(20_000, LGB_X_BYTE_BUDGET=600e6 // row_bytes))`, keeping the most recent
(scripts/hypersearch_v2.py:1286-1293, 1360-1368). At 72 features: seq 8/16 -> 120k rows; seq 24 -> 86.8k;
seq 40 (top of the default range, adaptive_config.py:22) -> 52.1k; seq 64 (expansion max) -> 32.6k.
Against ~264k crypto / ~1.54M stock rows that is ~20–45% of crypto and ~3–8% of stock history — the
"stronger learner" (hypersearch_v2.py docstring of train_lgb_ensemble) sees the least data, and its effective
training window is set by seq_len, a hyperparameter searched for the LSTM. Consequence for FR-02 (window
A/B): at seq ≥ 24 the LGB leg is ALREADY on a short recent window in every arm, so FR-02's LGB comparison is
confounded unless each arm logs the LGB effective row count and calendar span.
**Literature.** Trees scale with rows, not raw lag depth: TabArena (arXiv 2506.16791, 2025-06) and the M5
winners (sparse lags + rolling summaries) — cited in 05 §N2 / model docstrings; Bysik & Ślepaczuk
(arXiv 2606.00060, 2026-05-19; 70k hourly BTC-USDT, 27-fold walk-forward) got their best net-of-10bp
long-only result from gradient boosting on engineered lag/summary features, not full windows.
**Experiment (Jetson-later, idle box; not during a hypersearch).** Step 0 (free): the next retrain log
already prints `[LGB] row cap ...` — record rows, span and pre-holdout fraction per book. Step 1: on the
same winner cfg + same LSTM + same seed (TRAINER_SEED) + FR-01 fixed holdout, three LGB arms:
A = legacy flatten (byte-capped); B = lag subset L={0,1,2,3,5,8,12,23 ∩ <seq_len} x F at the same 120k cap;
C = B with the row cap lifted to the byte budget (crypto: 600e6/(8·72·4) ≈ 260k = all pre-holdout rows).
**Pre-registered rule.** Statistic 1: LGB-leg holdout pooled Spearman IC vs forward return, weekly-block
bootstrap (2000 reps) 95% CI of ΔIC(arm − A). Statistic 2: blend holdout DSR at identical cum_trials.
Statistic 3: peak RSS (`/usr/bin/time -v`). Adopt C (else B) iff CI lower bound of ΔIC > 0 on that book AND
blend DSR ≥ DSR(A) − 0.02 AND peak RSS ≤ RSS(A) + 100 MB. Otherwise record a durable negative.
**Ship.** Default-OFF `LGB_LAG_SUBSET` touching hypersearch_v2 + predict_now/`model_lgb.flatten_sequence`
parity (serving must build the identical lag subset; parity test required) -> challenger -> shadow.
Kill-list: NOT the killed "pruned feature core" [wave-6] (that was an inference-cost claim; this is a
training-row claim). Features unchanged, so no new-oscillator issue.

### S1-02 — The holdout DSR gate has near-zero power at realistic per-trade Sharpes (OBJECTIVE arithmetic, JUDGMENT remedy => OWNER)

**Source.** Lopez de Prado, Lipton & Zoonekynd, "How to Use the Sharpe Ratio", SSRN 5520741, 2025-09-23
(code: github.com/zoonek/2025-sharpe-ratio `functions.py`: SR variance with skew/kurt/autocorrelation,
PSR, MinTRL, power, pFDR/oFDR, variance of the max of K). Companion: "Sharpe Ratio Inference: A New Standard
for Decision-Making and Reporting" (SSRN 5950754). Claim: the SR is only safe with an inference framework
that reports power and false-discovery probability, not just a deflated p-value.
**Check 1 (no-op verdict, OBJECTIVE).** The repo's `deflated_sharpe_ratio` (validation.py:129-173) equals
the paper's PSR at K=1 with variance evaluated at SR0 to ±0.01 on six (n, SR, skew, kurt, N) cases
(e.g. n=60, SR=.40, skew 1.5, kurt 8, N=300: repo .596 vs paper-K1 .597). The paper's variance-of-the-max
(K>1) form sharpens the test (same case: .737) but it models the IN-SAMPLE max; the gated holdout SR is one
out-of-sample draw of the fold-selected winner, so it does not apply there. Its `rho` term is a serial
correction — never combine with n_eff (gotcha #4).
**Check 2 (the finding).** The gate passes iff DSR ≥ 0.60 (hypersearch_v2.py:2038) with null width
1/sqrt(n_eff) (hypersearch_v2.py:2018-2026 -> validation.py dsr_from_trade_returns, no sr_std supplied).
Normal-returns power table (true per-trade SR1; N = n_trials):

| n_eff | N=100 α / power@.1/.2/.3 | N=500 | N=2000 |
|---|---|---|---|
| 30 | .003 / .01 .05 .13 | .0005 / .00 .01 .05 | .0001 / .00 .00 .02 |
| 60 | .003 / .02 .11 .33 | .0005 / .01 .04 .17 | .0001 / .00 .02 .09 |
| 120 | .003 / .05 .28 .69 | .0005 / .01 .13 .49 | .0001 / .00 .07 .34 |
| 250 | .003 / .12 .65 .97 | .0005 / .04 .44 .92 | .0001 / .02 .30 .85 |

A per-trade SR of 0.1 is already ~2 annualized at ~500 trades/yr, yet with tens of effective holdout trades
and a cumulative pool of hundreds (B03.2 cum_trials grows every week) it passes <5% of the time, while a
skill-less model on an untouched holdout passes ~0.05–0.3%. The gate is power-starved, not FDR-starved, and
power decays monotonically as cum_trials accumulates (a ratchet toward "nothing ever promotes").
JUDGMENT (owner): the E[max of N] deflation targets in-sample selection; on a holdout its honest pool is the
number of HOLDOUT evaluations (B03.2's `cum_holdout_gates` + reuse overlap), not the Optuna trial count —
but B03.2 deliberately chose the conservative pool. Do not stack a remedy on the anti-stacking guard.
**Experiment / rule.** Measurement-only: add a `gate_power` block to the holdout certificate
(validation.py helper + hypersearch/backtest print): α under true SR=0, power at SR1 ∈ {0.05, 0.10, 0.20},
MinTRL, and the same numbers re-computed with n_trials = cum_holdout_gates. Pre-registered escalation:
if power at SR1 = 0.10 is < 0.20 on ≥ 3 of the next 4 certificates of a book, file an OWNER item choosing
exactly ONE of {holdout pool = holdout-evaluation count; longer FR-01 fixed holdout to raise n_eff;
accept and rely on shadow DM-HLN}. No automatic action.

### S1-03 — Optuna 5.0.0 (released 2026-09-07) flips TPESampler defaults; the Jetson pin is `optuna>=4.0` (OBJECTIVE)

**Source.** Optuna v5.0.0 release notes (github.com/optuna/optuna/releases/tag/v5.0.0): TPESampler
`multivariate=True` and `constant_liar=True` by default (#6746, #6738), new bandwidth rule (Watanabe 2023),
`set_system_attr`/`system_attrs` removed, UTC timestamps in RDBStorage. PyPI: 4.7.0 installed (jetson env),
4.8.0 2026-03-16, 4.9.0 2026-06-01, 5.0.0 2026-09-07.
**Why it matters.** hypersearch_v2.py:2371-2375 builds `TPESampler(n_startup_trials=..., seed=...)` relying
on defaults; requirements-jetson.txt pins only `optuna>=4.0`. Any `pip install -U` silently changes the
sampler: R2C-04 seed determinism breaks across versions, and a study resumed from a 4.x sqlite DB continues
under a different proposal distribution while cum_trials keeps counting it as one family. grep found no
`set_system_attr`/`multi_objective` use (no crash risk; behaviour risk only).
**Action (no experiment).** Either pin `optuna>=4.0,<5` (requirements files are ops-owned — report) or pass
`multivariate=False, constant_liar=False` explicitly in hypersearch_v2 (SIGNAL landing protocol). Adopting
multivariate TPE deliberately = gotcha-#2 study reset + cum_trials family reset, an owner decision.
**Env horizon.** Python 3.10 reaches EOL ~2026-10 (PEP 619); the Jetson AI Lab torch wheels are cp310, so the
env is frozen on 3.10. Upstream is leaving: scikit-learn 1.9.x requires py≥3.11 (PyPI metadata);
LightGBM 4.7.0 (2026-07-18) and numba 0.67.0 (2026-08-11) still support 3.10; pyarrow 25.0.1 (2026-08-10)
requires ≥3.10. Installed: lightgbm 4.6.0, numba 0.63.1, pyarrow 23.0.1, torch 2.8.0, sklearn 1.7.2,
numpy 1.26.1, pandas 2.3.3. Recommend (ops) a `pip freeze` lock snapshot so a rebuild is reproducible.

### S1-04 — Meta-probability calibration: add a Beta-calibration arm; Venn-Abers only via owner ask (JUDGMENT)

**Sources.** Manokhin & Grønhaug, "Classifier Calibration at Scale", arXiv 2601.19944, 2026-01-19 (21
classifiers incl. LightGBM, 5 post-hoc methods): Venn-Abers gives the largest mean log-loss reduction,
Beta calibration (Kull, Silva Filho & Flach 2017) is a close second and improves most often; Platt and
isotonic "can systematically degrade proper scoring performance for strong modern tabular models".
Meyer, Barziy & Joubert, "Meta-Labeling: Calibration and Position Sizing", JFDS 5(2):23-40, 2023: calibration
materially helps FIXED sizing functions; data-fitted sizers (ECDF, sigmoid-optimal) gain nothing from it.
**Why it applies.** Live sizing is a fixed function of p: veto p<0.30 and tilt clip(2p, 0.6, 1.3)
(meta_label.py:640 `meta_size_mult`) — exactly the class Meyer et al. find calibration-sensitive.
calibration.py uses isotonic at n≥1000 else Platt (calibration.py:186-191). Beta calibration is a 3-parameter
logistic on [ln p, −ln(1−p)] — pure numpy, Mac-testable, not conformal.
**Kill-list.** Venn-Abers (IVAP) is a conformal-family CALIBRATOR, not abstention, but it sits next to
"Conformal abstention (CQR/ACI)" [wave-4] — include it only after an explicit owner ruling (ask filed here).
**Experiment (Jetson-later: no meta_* artifacts exist today).** On the purged OOF meta frame
(calibration.crossfit_oof_predict, embargo per CALIBRATION_V2), calibrators fit INSIDE each purged fold.
Arms: raw, Platt, isotonic, Beta (+ IVAP if ruled in). Rule: switch the incumbent to Beta iff the paired
OOF log-loss improvement's weekly-block-bootstrap 95% CI excludes 0 on BOTH books AND ECE (10 bins) is not
worse AND the p<0.30 veto flip-rate vs incumbent is < 10%. Ship: default-OFF sub-flag of CALIBRATION_V2.

### S1-05 — Does the LSTM leg (and `import torch` in the bots) pay for its RAM? (JUDGMENT => OWNER)

**Sources.** Bysik & Ślepaczuk, arXiv 2606.00060 (2026-05-19): hourly BTC, XGBoost vs LSTM vs
iTransformer, 27-fold walk-forward — XGBoost "descriptively stronger ... although bootstrap evidence does not
support formal statistical dominance"; "the main obstacle ... is the way forecasts are converted into
trades" (a cost-aware filter restores profitability). Consistent with 05 §N2 (TabArena).
**Why it applies.** Priority #1 is Jetson memory; torch is the largest import in the bot process. blend_fit
already estimates w with an overlap-corrected SE (blend_fit.py:111-171) and stage0 dumps carry
`lstm_pred`/`lgb_pred` separately (stage0_preds.py:22,149-150), so the evidence is almost free.
**Rule.** Measurement-only first: per retrain, test H0: w_raw = 0 (LSTM weight) with the existing SE.
If w_raw − 2·se ≤ 0 on 2 consecutive retrains of a book AND LGB-only holdout DSR ≥ blend DSR − 0.02 AND
LGB-only λ* (backtest --fee-sweep) ≥ blend λ*, file an OWNER item for a default-OFF `SERVE_LGB_ONLY`
(torch-free bot path) quoting the measured `import torch`+model RSS (measure under hwlock). Interacts with
S1-01 (a better-fed LGB leg makes this more likely) — run S1-01 first.

### S1-06 — LightGBM quantized-gradient training (JUDGMENT, cheap)

**Source.** Shi, Ke et al., "Quantized Training of Gradient Boosting Decision Trees", NeurIPS 2022
(arXiv 2207.09682): 2–5-bit gradient quantization, up to ~2x speedup (CPU incl.) at comparable accuracy.
In LightGBM since 4.0.0: `use_quantized_grad` (cpu/cuda), `num_grad_quant_bins=4`,
`stochastic_rounding=true` (LightGBM Parameters docs). model_lgb.train_lgb sets none of them (model_lgb.py:69-79).
**Why it applies.** 6 ARM cores; the LGB mean + q10 legs, the meta booster and S1-01/S1-07 re-fits are all
CPU-bound. Speed buys rows (S1-01 arm C) or CPCV splits (S1-07) at unchanged wall.
**Rule.** Fixed winner cfg, 5 seeds per arm: adopt iff median wall-time ratio ≤ 0.70 AND holdout IC
difference passes TOST equivalence at ±0.005 (90% CI inside) AND q10 empirical coverage stays in 10% ± 3pp
(reliability_report). UNVERIFIED: determinism with stochastic_rounding under a fixed seed (must pass the
R2C-04 byte-compare), and quantile-objective accuracy under quantization. Ship: default-OFF `LGB_QUANT_GRAD`.

### S1-07 — CPCV/PBO audit of winner selection (measurement-only)

**Source.** Arian, Norouzi & Seco, "Backtest overfitting in the machine learning era: a comparison of
out-of-sample testing methods in a synthetic controlled environment", Knowledge-Based Systems 305 (2024),
112477 (SSRN 4686376): combinatorial purged CV shows lower PBO and a better DSR statistic than walk-forward;
walk-forward shows "notable shortcomings in false discovery prevention". Synthetic data; abstract-level read.
**Why it applies.** Selection runs on 3 expanding walk-forward folds (hypersearch_v2 docstring);
`validation.pbo_cscv` exists with no production caller (CLAUDE.md). A post-hoc audit needs no search change.
**Experiment.** New `scripts/cpcv_pbo_report.py`: top-K=16 completed trials from the study DB; LGB leg only
(LSTM too costly) at each trial's (seq_len, forward_bars, threshold); CPCV N=6 groups, k=2 (15 splits,
5 paths), purge = forward_bars, embargo = seq_len bars; per-path policy Sharpe via
objective_utils.simulate_trades_core; PBO via validation.pbo_cscv on the K x path matrix.
**Rule.** PBO > 0.50 on a book => OWNER report ("fold-score selection is no better than chance"); PBO < 0.20 =>
record as supporting evidence for the current search; in between => report only. ~240 LGB fits: hours;
idle-Jetson only, after S1-06 if adopted.

### S1-08 — Window length and model complexity must be chosen jointly (lens for FR-02)

**Source.** Capponi, Huang, Sidaoui, Wang & Zou, "The Nonstationarity-Complexity Tradeoff in Return
Prediction", arXiv 2512.23596 (2025-12-29, rev. 2026-08-13): longer windows help complex models but import
stale regimes; ATOMS = pairwise tournament with adaptive validation look-back; +14% OOS R² vs fixed-window
and regime-switching baselines (17 industry portfolios, monthly horizon — far from hourly).
**Why it applies (partly).** FR-02 A/Bs window at fixed complexity; S1-01 shows the LGB leg's window is
already coupled to seq_len. **Rule.** Extend scripts/window_ab.py arms with a complexity axis
(LGB num_leaves ∈ {15, 63}) and log LGB effective rows per arm; adoption rule stays FR-02's (both fold
objective AND holdout DSR win; trials counted). ATOMS itself is not built — report-only lens.

### S1-09 — The research campaign is itself a discovery system: keep a trial ledger (measurement-only)

**Sources.** Pan, Ding & Giesecke, "Agentic Empirical Asset Pricing: Methodological Foundations",
arXiv 2609.00731 / SSRN 7359503 (2026-07-22): evaluate the discovery SYSTEM by rolling re-execution, not its
outputs. Tang et al., AlphaAgent (KDD 2025, arXiv 2502.16789): unconstrained LLM factor mining p-hacks and
decays. **Why it applies.** Multi-agent campaigns here run model-facing A/Bs against the same holdout
(FR-02, FR-04, S1-01 ...). Harvey-Liu's judgment-inclusive count (B03.2) should see them.
**Rule.** Append-only `research/trial_ledger.jsonl` {date, book, experiment id, arms, holdout span}; the
certificate prints ledger arms since the last family reset beside cum_trials. Same pool, not a second
correction (anti-stacking guard respected). Feeds S1-02's holdout-evaluation count. No automatic action.

### S1-10 — TabICLv2 is license-clean; tabular foundation models still lose under temporal shift

**Facts (OBJECTIVE, checked 2026-09-27).** tabicl 2.2.0 on PyPI: BSD-3-Clause, python ≥3.10, torch ≥2.2
(install on the Jetson UNVERIFIED); HF `jingang/TabICL` card license bsd-3-clause, checkpoints
`tabicl-{classifier,regressor}-v2-20260212.ckpt`. This closes 05 §N8's open "verify TabICLv2's license".
tabpfn 6.0.0 shipped 2025-11-06 (the TabPFN-2.5 week) and is now 9.0.0 (2026-09-15): FR-11 should pin
tabpfn<6 for v2 weights (default-checkpoint mapping UNVERIFIED — FR-11 step 0 must confirm).
**Evidence against.** Purucker et al., "Beyond IID: How General Are Tabular Foundation Models, Really?",
arXiv 2606.30410 (2026-06-29, 142 datasets): TFMs excel on small IID data; trees dominate non-IID incl.
temporal. Loza et al., arXiv 2607.26000 (2026-07-28): all nine TFMs tested degrade under shift.
**Rule.** No new build. If FR-11 runs, add TabICLv2 as one more arm under FR-11's own temporal-split rule
(≥ 0.01 AUC over LGB at the books' actual n).

### Checked, not adopted

- **S1-11 Conformal Kelly** (Ryan, arXiv 2608.01494, 2026-08-02): conformal-interval-scaled fractional Kelly;
  dev window SR 1.34, but the PRE-REGISTERED 2022+ test fell below passive benchmarks while coverage stayed
  calibrated (74.5% vs 75%). Corroborates KILL "Conformal abstention (CQR/ACI)" [wave-4]. Adaptive conformal
  VaR on crypto (MPRA 121214) and nonstationary portfolio-VaR conformal (arXiv 2602.03903) are risk-coverage
  results, not decision P&L. **No re-open ask.** FR-12's no-conformal boundary stands.
- **S1-12 Decision-focused / SPO** (arXiv 2601.04062, 2026-01; IPMO arXiv 2512.11273): monthly-ETF, linear
  predictors, mean-variance layers — nothing hourly, single-name or long-only-threshold. 05 §N3 stands.
  Bysik & Ślepaczuk (S1-05) instead corroborate the EXISTING cost-aware entry gate.
- **S1-13 Post-selection Sharpe** (Pav, arXiv 2606.01650, 2026-06): James-Stein best estimates the true SR of
  the in-sample winner. Could print a shrunk fold score beside best_score; B03.2's Thresholdout ratchet
  already prices this. Report-only, low EV.
- **S1-14 "Utility-weighted calibration under trading frictions"** (arXiv 2601.07852, 2026-01, single
  author): reports t = −30.31 and Sharpe −3.62 -> −2.29; not credible evidence. No action.
- **S1-15 Complexity debate** (Nagel, NBER w34104, 2025; Kelly-Malamud reply 2025; Capponi et al. S1-08):
  no change to 05 lesson 8 (modest supervised blend).
- **S1-16 Ensemble meta-labeling** (Thumm, Barucca & Joubert, JFDS 5(1):10-26, 2023): gains need
  multi-regime data AND sample; meta is starvation-tier (B04.3). Not now; revisit only with n ≥ ~5k meta rows.

### Owner items raised (asks only)

1. S1-02: choose ONE holdout-power remedy (pool definition / longer fixed holdout / rely on shadow).
2. S1-04: is Venn-Abers (a conformal-family calibrator, not abstention) inside the [wave-4] kill boundary?
3. S1-03: pin optuna <5 or make the TPE kwargs explicit (ops vs SIGNAL landing).
4. S1-05: torch-free serving is a Jetson-memory decision gated on the pre-registered evidence.

### Not done this round

No code, no tests, no training/harvest/backtest runs. Full-text reads were blocked (HTTP 403) for SSRN
5520741 / 6543458 and the KBS article — those claims are abstract/code-level. Torch RSS, TabICL install,
and LightGBM quantized determinism are UNVERIFIED.

## 2026-09-27 R2 scout brief (SCOUT-2)

Scope: S1-01/S1-04 → build-blind designs + release notes. Live tree re-read ~02:00; numbers from parquet metadata /
column projection (hwlock) and the running crypto log. No code changed. OBJECTIVE vs JUDGMENT labelled.

### Corrections to R1 (OBJECTIVE)

- **C1: S1-01 row fractions were wrong.**
  - Features: `--preset stationary` gives 30 on crypto and 65 on stock (log: `Preset: stationary (30 features)`;
    72 is the preset LIST length; hypersearch_v2.py:221-236).
  - Stock loads `--max-rows 200000` (run_pipeline.py:1243; hypersearch_v2.py:258-270): 191,251 rows, not 1.54M.
    LGB trains on `folds[-1]` train only (1392; labels ≤ the 0.85 search-region quantile, 474-481): 196,953
    rows on crypto (log), ~143k on stock.
  - Cap = `min(120k, max(20k, 600e6//(seq·F·4)))` (1322, 1330, 1395-1403). Share of the pool seen: crypto 61%
    at seq ≤ 41 (the 120k ceiling binds, not the byte budget), 53% at 48, 40% at 64; stock 84% at ≤ 18, 67% at
    24, 54% at 30, 40% at 40, 25% at 64.
  - Starvation is moderate on crypto, real on stock only at seq ≥ 24. Crypto spans 2021-01→2026-09 (50,121 h,
    6 tickers); the cap drops 2021-22.
- **C2: X_val (≤ 30k rows, 1404-1410) is outside LGB_X_BYTE_BUDGET:** stock +0.26 GB at seq 40, +0.42 GB at 64.
- **C3: S1-04 named the wrong shipped calibrator.** Defaults `META_CALIBRATION_MODE='legacy'`
  (strategy_config.py:435) + `CALIBRATION_V2=False` (:454) mean sklearn IsotonicRegression fit on the same 20%
  slice the meta booster early-stopped on (meta_label.py:1008, 1103). The fit_calibrator chooser (isotonic at
  n≥1000, else Platt; calibration.py:32, 186-234) runs only on purged_oof (1041-1053) or under V2 (1086-1101).
  02_research.md:269 (B04.2) already deferred Beta "unless post-fix reliability curves still show S-curvature",
  and puts the slice at 40–100 points, i.e. ~200–500 meta trades per book.
- **C4: the Beta A/B is a nested 1-df test.** `SigmoidCalibrator(platt_v2=True)` fits sigmoid(a·logit(s)+c)
  (calibration.py:147-156, plus target smoothing), which by Kull 2017 Prop. 1 is exactly beta[a=b].

### D1 — LGB lag-subset A/B (S1-01 → design)

**Parity surface (OBJECTIVE).** Nine sites build the LGB input as `windows.reshape(n,-1)` in flatten_sequence
order: predict_now.py:516 (mean) and :547 (q10); backtest.py:218 (meta_label reaches it via `_predict_ticker`,
meta_label.py:838/929); retrain_ledger.py:243; scripts/window_ab.py:146; hypersearch_v2.py:1409-1410 (fold), 1500
(R2C-03 refit), 1850 (holdout), 2623 (blend OOF). Serving holds exactly `seq_len` bars, so the lags are forced to
L ∩ [0, seq_len). Index map t = seq_len−1−lag (model_lgb.py:31-38); columns stay oldest-first, features inner.

**Arms.** L = {0,1,2,3,5,8,12,23} (8 lags at seq ≥ 24); features, params (model_lgb.py:69-79), seed and horizon
unchanged. A = legacy flatten + legacy cap. B = L at A's rows (lag effect). C = L on the whole fold-train pool,
uncapped (row effect, B→C).

**Data and folds.** Winner cfg, loaded with `hypersearch_v2.load_data` under production flags. Folds:
`get_walk_forward_folds(..., purge_val_labels=True)` (435). In each val window, early-stop on the first half,
embargo seq_len bars, score the second half (avoids production's X_val = stop set = q10-floor set leak). Pool the
3 scored halves. The holdout (1 fit per arm on folds[-1]) is secondary only, being the gate's own sample.

**Statistic.** Primary: paired ΔIC = Spearman(pred_arm, fwd ret) − Spearman(pred_A) on the same rows; stock also
reports mean per-bar cross-sectional rank IC. Weekly-block bootstrap resampling calendar weeks jointly across
tickers: block = 1 week (168 crypto bars ≥ fb 48), 2 weeks for stock when fb ≥ 32, **B = 2000**. Seed check:
2 extra seeds of A; ΔIC counts as noise if |ΔIC| < 2·SD_seed.

**Power (assumptions flagged).** The "~20k effective labels" figure is 264k/fb12 with independent tickers. With
cross-ticker ρ≈0.75 (assumed) there are 6/(1+5ρ)=1.26 effective names, so the pooled scored halves (~9.9k hours)
give n_eff ≈ 520 (fb 24) to 1,040 (fb 12). With arm-prediction correlation r = 0.8–0.9 (assumed; step 0 measures
it), SE(ΔIC) ≈ sqrt(2(1−r)/n_eff) ≈ 0.014–0.028, so the **80% MDE is ≈ 0.04–0.08 IC** (≈ 0.012 even at the naive
20k). Realistic effects are ≤ 0.01, so **S1-01's "CI lower bound ΔIC > 0" is effectively unpassable**, and the
design is re-cast as resource-motivated non-inferiority.

**Budget (X = rows·lags·F·4 B; histogram work ∝ cells = rows·cols).**

| book, seq | A rows / X_train / cells | B X / cells | C rows / X / cells | A X_val |
|---|---|---|---|---|
| crypto 24 | 120k / 346 MB / 86M | 115 MB / 29M | 197k / 189 MB / 47M | 86 MB |
| crypto 40 | 120k / 576 MB / 144M | 115 MB / 29M | 197k / 189 MB / 47M | 144 MB |
| stock 24 | 96k / 600 MB / 150M | 250 MB / 62M | 143k / 298 MB / 74M | ~156 MB |
| stock 40 | 58k / 600 MB / 150M | 250 MB / 62M | 143k / 298 MB / 74M | ~260 MB |

C uses less memory than A and ~0.3–0.5x the cells at seq ≥ 24. Peak/fit ≈ X_train + X_val + bins (~1 B/cell) +
all_scaled (32/50 MB) + the loader (= the production search's footprint, crypto RSS 2.4 GB): **idle Jetson only,
via hwlock heavy.** 14 mean-leg fits/book (3 arms x 3 folds + 3 holdout + 2 seeds); wall = 14x the per-fit time
from the next retrain's `[LGB] Training on` lines. **Training** cost claim only; KILL_LIST.md:104 stands.

**Rule (pre-registered, per book).** Adopt C (else B) iff (i) point ΔIC ≥ 0 and the one-sided 90% lower bound is
≥ −0.01; (ii) holdout blend DSR(C) ≥ DSR(A) − 0.05 at equal cum_trials; (iii) peak RSS(C) ≤ RSS(A) and
wall(C) ≤ wall(A) (`/usr/bin/time -v`); (iv) q10 floor coverage no worse. If (i) is inconclusive (point ≥ 0,
bound < −0.01), record "no harm detected, underpowered" and let challenger → shadow DM-HLN decide. No retries with
other lag sets (forking paths; log them in the S1-09 ledger).

**Kill.** Point ΔIC < 0 on both books; any lower bound < −0.02; RSS(C) > RSS(A); or SD_seed > |ΔIC| on both
books (FR-02 then owns the window question). Run with LGB_REFIT_FULL OFF (strategy_config.py:281); its pool is all
purged pre-holdout rows (objective_utils.py:198-215), so C ≈ "REFIT_FULL rows at lag-subset width".

**Ship.** Measurement-only `scripts/lgb_lag_ab.py` (patterned on window_ab.py `run_arm`), then
- default-OFF `LGB_LAG_SUBSET = None` (+ `TRADER_LGB_LAG_SUBSET`) holding the lag tuple. One pure helper
  `model_lgb.lag_select(windows3d, lags)` serves all nine sites; `lags=None` returns `windows.reshape(n,-1)`
  byte-identically (pinned test).
- ON: `{p}lgb_lags.json` is a save_model_atomically extra artifact, same generation as the booster. predict_now
  reads it under the R2C-01 cache key and asserts `booster.num_feature() == len(L)·F`. File absent → legacy
  flatten; mismatch → loud LSTM-only fallback (the existing failure path).
- Parity test: `lag_select` == `flatten_sequence(seq)[cols(L)]` == backtest/holdout builders. Model-facing ⇒
  challenger → shadow; no LSTM study reset needed.

### D2 — Beta-calibration arm for the meta probability (S1-04 → design)

**How p is used (read-only, OBJECTIVE).** `_meta_gate` (base_loop.py:2083-2117) vetoes at p < 0.30
(meta_label.py:58; base_loop.py:2107), else returns clip(2p, .6, 1.3) (meta_label.py:640-653). That multiplies the
tilt (base_loop.py:2519; v2 path 2598), clamped to [0.1, TILT_MAX] (2540), so the **decision band is
p ∈ [0.30, 0.65]**. The calibrator is a joblib pickle (meta_label.py:490-496) shared by backtest
`predict_meta_array` (622-637); a load failure disables meta, fail-open (542-543).

**Preconditions (all, else do not run).** P1: CALIBRATION_V2 and purged_oof live and certified (B04.2 order).
P2: reliability_report.py on the post-fix calibrator still shows curvature in 0.3–0.65. P3: OOF meta rows ≥ 1000
(OOF_FULL_PARAMS_MIN_ROWS, meta_label.py:146). With ~200–500 trades today (C3), **"starved, not run" is the
expected and valid outcome.**

**Arms (all out of fold).** R = raw. I = incumbent-as-planned, `fit_calibrator(v2=True)` (isotonic(pool_ties) at
n ≥ 1000, else Platt-v2). P = Platt-v2 forced (= beta[a=b]). Beta = Kull 3-parameter logistic on
[ln s, −ln(1−s)], s clipped to [1e-6, 1−1e-6]; a negative coefficient is fixed to 0 and refit (Kull §3.3);
Platt target smoothing kept for parity with P. IVAP only after an owner ruling.

**Data and scheme.**
- train_meta builds its frame inline (argsort at meta_label.py:1003-1006), so an instrumentation hook is needed:
  env `TRADER_META_FRAME_DUMP=<npz>` writes (X, y, entry_e, exit_e, n_iter, params) before calibration. Staged
  landing, byte-identical when unset; it rides the next production meta phase at zero extra cost.
- Level 1: OOF raw scores via `calibration.crossfit_oof_predict` (k=5, purged, embargo 0.05 = V2's). Level 2: in
  each purged fold, fit every calibrator on the other folds' OOF scores and predict the held-out fold (OOF-of-OOF).
  Light (≤ ~10k x ~15, < 1 min, < 300 MB); calibrator math Mac-testable.

**Statistic.** Primary: mean paired per-trade log-loss d = LL_I − LL_Beta; weekly-block bootstrap by entry week
(≥ the 48-bar max label span), **B = 5000**, two-sided 95%. Secondary: ECE (10 equal-mass bins), Brier, veto
flip rate, mean |Δ clip(2p,.6,1.3)|, and decision P&L Σ 1[p≥.3]·clip(2p,.6,1.3)·r_net (paired, bootstrapped).
Nested prerequisite: full-sample LR(Beta vs P) ≥ 3.84 (χ²₁), i.e. mean gain > 1.92/n nats (0.0019 at n=1000).
Power: MDE = 2.8σ_d/√n = 0.089σ_d at n=1000, 0.063σ_d at n=2000. Manokhin-Grønhaug's gains are vs UNcalibrated
scores; vs Platt-v2 the increment is one parameter, so the prior favours a null result.

**Rule (per book; both books to flip).** Beta replaces only the branch it beats (n<1000: Platt-v2; n≥1000: must
beat isotonic) iff LR ≥ 3.84, the 95% CI of d excludes 0 in Beta's favour, ECE(Beta) ≤ ECE(I) + 0.005, veto flip
rate < 10%, and decision-P&L difference ≥ 0.

**Kill.** P1–P3 fail; LR < 3.84 on both books; d's CI includes 0 on either; decision P&L < 0; >1/5 folds refit.

**Ship.**
- Default-OFF `CALIBRATION_BETA = False` (+ `TRADER_CALIBRATION_BETA`), read only in `calibration.fit_calibrator`
  and effective only when v2 is True (ON with V2 OFF: ignored with a warning). New `BetaCalibrator`
  (method_='beta'; `dropped_coef_`) beside SigmoidCalibrator.
- OFF path byte-pinned (fit_calibrator outputs identical for v2∈{False, True} on a fixed synthetic set).
  Rollback caveat: a Beta pickle cannot unpickle on pre-Beta code (meta silently fail-open), so code and artifact
  roll back together (runbook line).
- Results go to a measurement-only `scripts/meta_calib_ab.py` or a `--arms` option on reliability_report.py
  (owner picks; reliability_report is the incumbent flip harness).

**Kill-list boundary (JUDGMENT).** Beta is a parametric logistic regression on two log-transformed score
features (Kull §3.3, Prop. 2) and a strict generalisation of the V2 Platt map: no conformal/Venn machinery, no
abstention, no set output. So it is **OUTSIDE** KILL_LIST.md:105 ("Conformal abstention (CQR/ACI) — red-team
rejected [wave-4]"). IVAP refits isotonic with the test point appended under each label (Venn prediction), and
generalized Venn-Abers "recovers conformal prediction as a special case" (van der Laan & Alaa 2025). IVAP is
therefore **adjacent**, and the S1-04 owner ask stands.

### Sources (dated; access noted; no 403s this round)

- Kull, Silva Filho & Flach, EJS 11(2):5052-80, 2017, doi 10.1214/17-EJS1338SI (**full PDF read**): Prop. 1;
  §3.3 negative-coefficient elimination; §4.2 Beta beats logistic on LL for 4/7 learners, never significantly
  beaten; §4.4 Beta fits well "regardless of dataset size".
- Manokhin & Grønhaug, arXiv 2601.19944 (2026-01; html): Beta and Venn-Abers "most consistently improve"; **no
  per-model, small-n or imbalance breakdown**, so R1's LightGBM-level reading is unverifiable.
  Manokhin, arXiv 2605.03816 (2026-05-05; abstract): Venn-Abers −6.5..−12.6% LL on under-confident models, +2.1%
  on over-confident; "no order-preserving post-hoc calibrator can add discriminatory power".
- van der Laan & Alaa, arXiv 2502.05676 (2025-02, v3 2025-07, ICML 2025; abstract): the conformal special case.
- Lags: Mesfin, arXiv 2605.17724 (2026-05-18; abstract): on 5-min futures GB ≥ LSTM, neither significant, so data
  is the limit. Sharma et al., arXiv 2509.20244 (2025-09-24; abstract): dynamic lags ~5% MAPE, e-commerce (weak
  transfer). FETS, arXiv 2604.22328 (2026-04; snippet): SHAP lag selection for GBDTs. Negative prior: GBDTs are
  robust to redundant lags (Biogeosciences 20:897, 2023; snippet), so expect any gain from ROWS (arm C). Gap: no
  2025-26 lag-subset A/B on hourly financial panels found; arXiv 2503.17290 abstract-only, unused.

### Release notes (≤ 15 lines)

- **Resumed crypto study (OBJECTIVE).** DB and install are both optuna 4.7.0, so no drift.
  - Trial #4 is orphaned RUNNING (started 01:01:58). Harmless under 4.7 (TPE ignores RUNNING without
    constant_liar; pools count COMPLETE only, hypersearch_v2.py:2432, 2486-2505). Under 5.0 (#6738) it becomes a
    permanent worst-value liar, so fail it before any upgrade.
  - `PRUNE_STARTUP_TRIALS=60` (:100) is BOTH TPE's and MedianPruner's `n_startup_trials` (2419-2424). With ≤ 44
    COMPLETE, **tonight's 40 trials are pure random search with no pruning** despite the "TPE + pruning" banner.
  - TRAINER_SEED=None (strategy_config.py:314), and re-seeding on resume means R2C-04 determinism never spans a
    stop/resume.
- Optuna 4.8.0 (2026-03-16): only #6505 (TPE multivariate+liar fix). 4.9.0 (2026-06-01): deprecations (#6635, none
  used). 5.0.0 (2026-09-07): multivariate/constant_liar defaults (#6746/#6738); UTC timestamps (#6776), so a
  `storage upgrade` is needed; system_attrs removed (#6834). S1-03 stands.
- LightGBM 4.7.0 (2026-07-18; installed 4.6.0 is from 2025-02-15): no quantized or determinism items, so S1-06
  stays UNVERIFIED (byte-compare with `deterministic=true`+`force_row_wise=true` still required). #7224 fixes the
  weighted percentile for quantile/l1; on 4.6.0 a weighted q10 (UNIQUENESS_WEIGHTS_ENABLED, default False,
  strategy_config.py:153) is affected, so add a runbook precondition. Minimum Python is 3.10 (#7276), OK.

## 2026-09-27 R3 scout brief (SCOUT-3)

Scope: (1) S1-02 → build-ready `gate_power` spec (measurement-only); (2) the `PRUNE_STARTUP_TRIALS` question
(owner item; standalone note `startup_trials_owner_item.md`). Live tree re-read ~02:20; Optuna source read from
the installed 4.7.0 (jetson env). One light numeric prototype (pure numpy, 15 s, imports the live `validation`;
scratchpad `scout3/gate_power_proto.py`, not repo code). No code changed. OBJECTIVE vs JUDGMENT labelled.

### G1 — How the holdout certificate is produced today (OBJECTIVE, live lines)

- Pool: `n_trials_pool = max(len(COMPLETE-with-value trials in the study), 2)` legacy; under PROMOTION_GATE_V2
  (default False, strategy_config.py:167) the overlap-weighted cum pool (hypersearch_v2.py:2476-2507).
  PRUNED and FAIL trials are never counted (every filter is `state == COMPLETE`).
- The holdout is scored ONLY when the winner beats the stored score: legacy `accept_new = new_score > existing`
  (:2525; V2 ratchet :2519). Tonight's crypto best is −0.763 vs stored 0.0, so **no certificate will be issued
  tonight** (no `[HOLDOUT]` lines exist in phase5/train_crypto.log; the 6-trial excerpt has none either).
- `evaluate_on_holdout(..., n_trials)` (:1774) → `dsr_from_trade_returns(trade_returns, n_trials, n_eff=…)`
  (:2066-2076) → report keys `sharpe, dsr, dsr_min, n_trades, n_eff, n_eff_v2, status, min_trl, n_trials_pool,
  u_bar_mean, trade_returns, n_rows, hit_rate, pred_deciles` (:2093-2117; + blend keys when blended). Gate:
  `sharpe > 0 and dsr >= dsr_min` (:2883-2885). MinTRL print on failure (:2084-2092).
- DSR as implemented (validation.py): ne = n_eff clamped to [10, n] (legacy) or upper-clamp + fail-closed <10
  (V2) (:267-272); null width σ0 = 1/sqrt(ne) (:297); SR0 = `expected_max_sharpe(N, σ0)` =
  σ0·[(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(N·e))], N clamped to [2, 1e15] (:81-100, :302);
  DSR = Φ((ŝ−SR0)·sqrt(ne−1)/sqrt(v(ŝ))), v(s) = max(1 − γ3·s + (γ4−1)/4·s², 1e-12) with γ4 Pearson
  (non-excess) kurtosis — the moment term is evaluated at the OBSERVED ŝ, not at SR0 (:129-173).
- **Side finding (OBJECTIVE):** `adaptive_state['cum_holdout_gates']` is declared/back-filled
  (adaptive_config.py:109, :136) but **never incremented** anywhere (grep: no writer in any .py). S1-02's
  "re-compute with n_trials = cum_holdout_gates" would silently read 0.

### G2 — `validation.gate_power` spec (build-ready; instrumentation ⇒ ships directly next window)

Signature (pure, numpy/math only, Mac-testable):
`gate_power(sr_true, n_eff, n_trials, dsr_min=DSR_MIN, skew=0.0, kurt=3.0, *, n_trades=None,
fail_closed_floor=False) -> float | None` = P(gate passes | true per-trade SR = sr_true), under the DSR's
own normal approximation.
1. ne_gate = n_eff if fail_closed_floor else max(n_eff, 10) (mirrors :267-272); N = max(int(n_trials), 2)
   (mirrors the production clamp); SR0 = expected_max_sharpe(N, 1/sqrt(ne_gate)); z* = Φ⁻¹(dsr_min)
   (= 0.2533 at 0.60).
2. Acceptance threshold c = the root > SR0 of (c−SR0)·sqrt(ne_gate−1) = z*·sqrt(v(c)) (bisection on
   [SR0, SR0+10], 200 iters; v uses the supplied skew/kurt); then c = max(c, 0) for the `sharpe > 0` leg.
3. Sampling law at the TRUE effective count: ŝ ~ N(sr_true, v(sr_true)/(n_eff−1)) (Mertens/Opdyke variance,
   the same one the DSR z uses). Power = 1 − Φ((c − sr_true)/sqrt(v(sr_true)/(n_eff−1))).
Fail-safe (never raises; body in try/except → None): any non-finite / non-coercible input → None;
dsr_min ∉ (0,1) → None; n_trials < 1 → None (a missing pool must not certify a power); n_eff ≤ 1, or
n_trades < 10 (dsr_from_trade_returns' insufficient_n), or fail_closed_floor with n_eff < 10 → 0.0.
Companion `gate_power_mc(sr_true, n_eff, n_trials, dsr_min=DSR_MIN, B=2000, seed=0)`: draws B samples of
round(n_eff) iid N(sr_true, 1) returns and runs the PRODUCTION `dsr_from_trade_returns(r, n_trials)`;
returns the pass fraction (`sr > 0 and dsr >= dsr_min`). Tests + the optional CLI only — not in the certificate.

Certificate block (new key, report-only; appended after `min_trl`, never read by the gate):
`holdout['gate_power'] = {'method': 'analytic_normal_v1', 'sr_grid': [0.05, 0.10, 0.15, 0.20],
'power': [...], 'alpha': gate_power(0.0, ...), 'sr0': ..., 'c_accept': ..., 'n_eff_used', 'n_trials_used',
'skew', 'kurt', 'power_holdout_pool': [...] | None}` with n_eff/skew/kurt/pool taken from the SAME `dsr`
dict (`dsr['n_eff']`, `dsr['skew']`, `dsr['kurt']`, `dsr['n_trials']`; skew/kurt None ⇒ 0/3).
`power_holdout_pool` recomputes with N = cum_holdout_gates and stays None until that counter has a writer
(Round-4 one-liner: +1 in adaptive state each time evaluate_on_holdout returns non-None; instrumentation).
Print (one line, always, after the DSR line): `  [HOLDOUT] gate power (N=44, n_eff=60.0, SR0=0.287,
accept ŝ>=0.320): alpha=0.007 | SR .05:0.02 .10:0.04 .15:0.09 .20:0.18`. Wrapped in try/except: a failure
prints `[HOLDOUT] gate power unavailable (<err>)` and omits the key — the certificate is never degraded.
Flag: none (report-only key; the flag-OFF key-for-key comment at :2115 is about `blended`, so the implementer
must extend any "exact key set" test to accept `gate_power`). backtest.py `--gate` may carry the same block
as `dsr_gate_power` beside `dsr_min_trl` (backtest.py:573) — optional second consumer.

Worked example (tonight's realistic numbers; no certificate exists yet — see G1). Legacy pool at the end of
tonight's run = 4 prior + 40 new COMPLETE = 44 (study DB read-only: 12 COMPLETE + 2 RUNNING at 02:19).
Analytic power at N = 44 (MC cross-check B=2000 in brackets):

| n_eff | SR0 | α (SR 0) | SR .05 | SR .10 | SR .15 | SR .20 |
|---|---|---|---|---|---|---|
| 10 | .704 | .008 | .012 | .018 [.043] | .026 | .037 |
| 30 | .407 | .007 | .014 | .028 [.043] | .051 | .086 |
| 60 | .287 | .007 | .019 | .045 [.046] | .095 | .178 |
| 120 | .203 | .007 | .027 | .084 [.082] | .202 | .386 |

The only real `[HOLDOUT]` lines on this box (signal_r1/phase3 synthetic-slice runs: 6 trades, n_eff legacy
2.0 → clamped to 10, v2 calendar 4.2) give power 0.000 under V2 (fail-closed) and, under the legacy clamp,
α ≈ 0.21 (indicative only — the normal approximation is meaningless at n_eff 2; it re-states the known
anti-conservative clamp that V2 closes). Pool sensitivity at n_eff 60 (power @0.10 / @0.20): N 44 .045/.178;
70 .031/.137; 100 .023/.111; 200 .013/.073; 480 (≈ initial + 4 refines) .006/.042; 1000 .003/.026.
Agrees with R1's S1-02 table (n_eff 60, N 100: .02/.11). Skew/kurt barely move it (n_eff 60, N 200, SR .10:
(0,3) .013, (1.5,8) .010, (−1,6) .015). **Verdict at realistic n_eff ≤ 120 and N ≥ 44: power at SR 0.10 is
≤ 0.09 — the S1-02 escalation condition (< 0.20) will be met on essentially every certificate.**

Analytic vs MC agreement (normal returns, B=2000, seed 7): max |Δ| over SR ∈ {0,…,.20} = .029 (n_eff 30,
N 45), .024 (60/45), .014 (120/45), .024 (60/200), .015 (120/270), .015 (250/500). The analytic value is
slightly LOW (conservative) at small n_eff because the MC also carries sample-moment noise and ddof=0 std.

Required tests (Round-4 implementer; `tests/test_gate_power.py`, pure, Mac-runnable, < 20 s total):
1. Agreement: n_eff ∈ {30, 60, 120}, N ∈ {44, 200}, SR ∈ {0, .05, .10, .20}: |analytic − MC(B=4000, seed
   fixed)| ≤ 0.03 + 3·sqrt(p(1−p)/4000).
2. Monotone non-decreasing in sr_true (grid 0…0.5 step .01) and in n_eff (10…500) at fixed N; monotone
   non-increasing in n_trials (2…1e6) at fixed n_eff; α(sr_true=0) ≤ 1 − dsr_min.
3. Pin: n_eff 60, N 100, normal → (.023, .111) at SR (.10, .20) ± 0.002 (reproduces R1).
4. Fail-safe: n_trials ∈ {0, −1, None, nan, 'x'} → None; n_eff ∈ {0, 1} → 0.0; n_trades 9 → 0.0;
   fail_closed_floor + n_eff 9.9 → 0.0; dsr_min ∈ {0, 1, nan} → None; never raises (hypothesis-style fuzz
   over floats incl. ±inf).
5. Consistency: with sr_true set to the observed ŝ and v at ŝ, c_accept reproduces the gate —
   `deflated_sharpe_ratio(c_accept, SR0, n, …, n_eff=ne) == dsr_min` to 1e-9.
6. Certificate wiring (source-text or stubbed evaluate_on_holdout): key present, never read by `gate_ok`,
   exception path omits the key and still returns the report; the `trade_returns`-stripped `_hr` print
   unchanged.

Pre-registered owner rule (restated from S1-02, now numeric): if `gate_power.power[SR=0.10] < 0.20` on ≥ 3
of the next 4 certificates of a book, the owner picks exactly ONE of {holdout pool = holdout-evaluation
count (needs the cum_holdout_gates writer); longer FR-01 fixed holdout to raise n_eff (prototype: n_eff 320
→ power .245 at N 44, SR .10); accept and rely on
shadow DM-HLN}. Anti-stacking: never combined with a DSR_MIN change (validation.py:30-41 comment). Note the rule
counts ISSUED certificates only; with negative search scores (tonight) none is issued, so "no certificate"
must be logged as its own outcome, not as a pass.

### G3 — Startup count (OWNER ITEM; full note: `startup_trials_owner_item.md`)

OBJECTIVE facts (installed Optuna 4.7.0 read, not docs):
- `PRUNE_STARTUP_TRIALS = 60` (hypersearch_v2.py:100) feeds BOTH MedianPruner (:2419) and TPESampler
  (:2422). History: 50 at dd58a02 (2026-02-11, "add regression hypersearch"), 60 at 570081a (2026-02-21,
  "comprehensive codebase review", in the same hunk that added NUM_TRIALS = 300). **Neither message nor any
  comment gives a rationale.** 60 matches the folk "rule of 59" (best of 59 random draws is in the top 5 %
  with p ≥ 0.95) and 20 % of 300 — provenance UNVERIFIED.
- TPE startup counts COMPLETE + PRUNED trials of the whole study DB (optuna/samplers/_tpe/sampler.py:446-450,
  462-466); FAIL (our `catch=(Exception,)`, :2464) and RUNNING do not count. MedianPruner counts COMPLETE only
  (optuna/pruners/_percentile.py:173-180), then needs step ≥ n_warmup_steps (:186-188) and compares the
  trial's BEST intermediate value over ALL its steps against the median of COMPLETE trials at that step
  (:196-207; n_min_trials=1).
- Our steps are `fold_idx·60 + epoch` (hypersearch_v2.py:1152) and the code guard is `epoch >= 12` per fold
  (:1154), so Optuna's `n_warmup_steps=12` binds only on fold 0. Because "best over steps" includes earlier
  folds, a trial with a good fold 0 is almost unprunable in folds 1–2; late-epoch medians thin out as COMPLETE
  trials early-stop (patience 10). Pruning signal = raw epoch Sharpe, not the soup/regime-penalised score.
- Search space: 13 dims (hypersearch_v2.py:842-879): 8 numeric (seq_len, hidden_dim, num_layers, dropout, lr,
  weight_decay, huber_delta, trade_threshold) + 5 categorical (forward_bars, n_heads, batch_size, scheduler,
  target_kind — present whenever TB labels exist, true on both books). OBJECTIVE_V3 only re-ranges
  trade_threshold (:865-870); HYPERSEARCH_V3 adds no suggest call. Effective d = 13.
- Trial budgets: TRIAL_COUNTS initial 200 / refine 70 / explore 120 (adaptive_config.py:75-79); first pipeline
  run passes `--trials 200 --mode initial` (run_pipeline.py:1531, 1618-1619); weekly retrain = max adaptive
  count over books (:1280-1289); `--trials` wins when ≠ 300 (hypersearch_v2.py:2272-2276); tonight 40.

Consequence (per run on a FRESH study DB, assuming no FAIL): 40 → 40 random / 0 TPE / pruning never;
70 → 60 random / 10 TPE (14 %) / pruning eligible for the last 10; 200 → 60 / 140 (70 %) / 140 eligible.
On a RESUMED DB that already holds ≥ 60 COMPLETE, every trial is TPE-steered and prunable — so the constant
only bites after a DB reset (gotcha #2, categorical expansion adaptive_config.py:520-550, `--fresh`). Tonight's
DB: 4 prior COMPLETE + 40 = 44 → the whole run is random search with no pruning (SCOUT-2 confirmed).

Pool interaction (JUDGMENT): median-pruned trials leave the deflation pool (COMPLETE-only filters), yet
`dsr_from_trade_returns`' own docstring says the pool must include "pruned/failed trials" (validation.py:
191-195) — a pre-existing doc-vs-code conflict (also IMPL-A owner item (d); reviews_2026-07 B1). SCOUT-3
view: a MEDIAN-pruned trial was compared on validation Sharpe and discarded ⇒ a selection event ⇒ count it;
a SIG-R2-1 failure-pruned trial produced no score ⇒ correctly excluded (distinguishable by its
`failed_trial` user_attr). Cost of counting is small: power at SR .10, n_eff 60 falls .031 → .023 when N goes
70 → 100 (G2), because SR0 grows only like sqrt(2 ln N).

Literature: Optuna defaults TPE n_startup_trials=10, MedianPruner 5/0 (4.7 source; 5.0 docs, 2026-09);
hyperopt-compat TPE default 20 (installed sampler.py:658); Watanabe arXiv 2304.11127 (v1 2023-04-21, v5
2026-05-31) runs n_startup=10 on 5/10/30-D and does NOT ablate or scale it with D; recommends multivariate=True;
BOHB (Falkner et al., ICML 2018) needs ≥ d+1 points per KDE (recalled, not re-read); Bossek, Doerr & Kerschke
GECCO 2020 (arXiv 2003.13826): small initial designs preferable for EGO, with cases where random wins;
Bergstra & Bengio JMLR 13 (2012) random-search baseline. No 2024-26 study of TPE warm-up size vs dimension
found (gap). constant_liar matters only for parallel/orphaned RUNNING trials (sequential here; S1-03).

Options, recommendation and the pre-registered measurement: see the owner note (short form: recommend B =
sampler 20 / pruner 20 COMPLETE behind a default-OFF flag, landed only after SIG-R2-1, with median-pruned
trials counted in the pool; keep A if the owner values random-heavy exploration of a very noisy objective).

### Not done

No code or tests written; no search run; Watanabe/BOHB full texts not re-read beyond the HTML summary; the
power numbers come from the scratchpad prototype, not a landed function. The prototype lives only in the scratchpad.

## 2026-09-27 R4 scout brief (SCOUT-4)

Scope: (1) the (d) lever — should the trainer's searched `trade_threshold` be tied to the live admission floor;
(2) an honest meta-calibrator holdout. I re-read the live tree at ~06:35 (hypersearch_v2 / objective_utils / meta_label /
strategy_config all have mtime 06:03). The stock study DB was read from a `cp` only. No code changed.
Labels: OBJECTIVE = one correct answer; JUDGMENT = owner item.

### T1 — Threshold floor vs live admission (the (d) lever)

**Code facts (OBJECTIVE).**
- The trial draws `trade_threshold ~ U[tt_range]` with step 0.01 (hypersearch_v2.py:1055-1062).
  - Legacy range is [0.05, 1.0] (adaptive_config.py:31). The edge expansion can reach 0.03 / 1.5 (:56); HARD_LIMITS are [0.01, 2.0] (:71).
  - OBJECTIVE_V3 (False, strategy_config.py:296) substitutes `v3_trade_threshold_range` (objective_utils.py:166-184):
    [round(0.8·F), min(2.5·F, 2.0)], with F = `fees.required_edge_pct(book, FLAT_SPREAD_PCT)`.
  - Evaluated now: crypto round trip 0.60 → F 1.20 → [0.96, 2.0]; stock 0.113 → F 0.226 → [0.18, 0.57].
  - **So 0.96 = 0.8 × the 1.20 % floor.** The stated reason is "so TPE sees the gradient across" the floor (:173-174).
    07_decision_influences.md:158 had asked for a margin ABOVE the floor instead.
- The trainer's cost is the literal `TXN_COST_PCT = {'crypto': 0.60, 'stock': 0.11}` (hypersearch_v2.py:530).
  That is the 1.0× break-even of the fees.py:14-21 ladder (stock 0.11 vs fees' 0.113). The trainer never applies the 2.0× floor.
- **Live and promotion gates both require pred ≥ max(thr, F).**
  - Live: `should_trade` checks pred > F on the live quote spread, with the maker blend (order_utils.py:1029-1055).
    It is called at base_loop.py:3347 and stock_loop.py:1103, followed by `pred ≥ trade_threshold` (base_loop.py:3359).
  - Backtest: skips if `p < threshold or p < edge_floor` (backtest.py:321, :365).
- **The trainer's scorers (fold / pruning / regime / holdout / H2 reselect) apply `threshold` only** (compute_sharpe :742-768;
  fold :1420; reselect :3108-3112). The trainer therefore scores "p > thr" while the book trades "p ≥ max(thr, F)".
  This is the same family as OBJECTIVE_LONG_ONLY and OBJECTIVE_SESSION_MASK (now landed, default False, :406).
- `compute_sharpe` returns 0.0 below 10 trades (:753-754), so an abstaining fold beats every negative fold.
  A trial must score > 0 to be saved (F2).
- Recorded data: crypto study (decomp copy) 16/16 COMPLETE thr in 0.12–0.94, all < F; 8/16 < the 0.60 break-even.
  Legacy [0.05, 1.0] puts 100 % of crypto draws below F by construction (stock: 18.5 % < F, 6.6 % < break-even); the
  current stock run (pid 202303, 06:29, 25 trials, all flags OFF) drew trial #0 at thr 0.05.
- **No re-scoring is possible.** Studies store no predictions. Winner OOF preds are persisted only at save (score > 0)
  to `{p}oof_preds.npz` (:2493-2516), and none exists on the box.

**Literature.**
- Bysik & Ślepaczuk, arXiv 2606.00060 (2026-05-19; html read): hourly BTC, 27 WF folds; naive sign rules die at 10 bp;
  |r̂| > λ·c·|Δpos| with **λ = 2.0 FIXED, not tuned** "sharply reduces turnover and restores profitability". λ = 2.0 is our
  MIN_EDGE_MULTIPLE (fees.py:58) — independent support for anchoring on F, not searching below it.
- Jadouli, arXiv 2607.19453 (2026-07-21; abstract): Binance spot, AUC 0.87–0.90 yet negative after 31 bp.
  Every decision came out NO_TRADE — abstention is a common honest outcome.
- Lim, Zohren & Roberts, JFDS 2019 (arXiv 1904.04912): Sharpe-loss LSTM wins only to 2–3 bp; they add a turnover
  regulariser, i.e. cost must be inside the objective.
- SPO / decision-focused work (arXiv 2601.04062; 2605.01176, 2026-05, on SPO turnover inflation; DeePM 2601.05975;
  abstracts): monthly, multi-asset, optimisation layers. S1-12 stands.
- Grinold–Kahn IR ≈ IC·√BR·TC, with the transfer coefficient from Clarke, de Silva & Thorley, FAJ 58(5) 2002 (recalled).
  Scoring breadth the book cannot take (TC = 0 below F) overstates BR — that is the formal name of this mismatch.
- Gârleanu & Pedersen, JF 68(6) 2013 (recalled): the optimal policy has a no-trade region; F is its crude boundary.

**Answer (JUDGMENT → OWNER): tie it, in the SCORER, not the range.**
- Proposed flag: default-OFF `OBJECTIVE_THRESHOLD_FLOOR_LIVE` (+ `TRADER_OBJECTIVE_THRESHOLD_FLOOR_LIVE`).
- ON: every trainer trade scorer uses thr_eff = max(thr, F_book), with F_book = `required_edge_pct(book, FLAT_SPREAD_PCT[book])`.
  That is exactly backtest.py:321/:365.
  Implement as `long_veto |= p < F_book` via the SIG-R2-MASK plumbing (:1018, every `long_veto=`); save
  `config['trade_threshold'] = thr_eff`; OFF byte-identical (pinned).
- Why the scorer, not the range: exact parity whatever the range; composes with legacy, V3 and adaptive expansion; no
  Optuna distribution touched. It still changes scores ⇒ gotcha-#2 reset; ride the V3 event.
- Interaction with V3:
  - Both ON: V3's [0.96, 1.20) band (23 % of the range) becomes a flat plateau at 1.20 — harmless.
    Owner option: under FLOOR, set V3 lo = F (07:158's "margin above").
  - FLOOR ON with V3 OFF: the crypto thr dimension becomes fully inert. Honest, but wasteful.
- **F1 / abstention hazard (OBJECTIVE mechanics).** PHASE3-1's blended preds peaked at 0.64–0.73.
  - Under FLOOR (or V3) such models make 0 trades and score 0.0, which beats every negative trial.
  - Nothing ships (F2), but selection then favours low-pred-scale models.
  - Tonight it is moot: every run is random search (PRUNE_STARTUP_TRIALS 60 at :100, more than 25/40 trials). It matters past 60 trials.
  - Companion owner option: score an abstaining fold as NaN plus an `abstained` attr, not 0.0.

**Experiment.** Instrumentation first — the only way to get counterfactuals without stored predictions.
- **Step 0: staged landing `SIG-R4-FLOORDECOMP`** (score byte-identical): a second `fold_trade_decomposition` at
  threshold = max(thr, F_book) beside :1420-1428 (same preds, one O(n) walk) → `floor_{n_trades, gross_ret_mean,
  net_ret_mean, sharpe, pass_rate}` user_attrs. Pin: objective return unchanged; attrs == independent replay.
- **Data:** the next ≥ 20-COMPLETE study per book.
- **Statistics (pre-registered):**
  - **M1** = share of trials with thr < F. This is arithmetic: 100 % crypto-legacy, ~18 % stock.
  - **M2** = among those, share with `cost_drag ≥ gross_ret_mean` (i.e. net ≤ 0; cost_drag = gross − net, :841) on ≥ 2/3 folds.
  - **M3 (primary)** = per below-F trial, Δ = mean over folds of (floor_sharpe − fold_sharpe). One-sided sign test,
    with Wilcoxon and per-trade net as secondaries.
  - **A** = share of trials whose median floor_n_trades < 10.
- **Rule:**
  - **PROPOSE** the flag (owner flips at the next reset) iff M1 ≥ 25 %, M2 ≥ 60 %, M3 p < 0.05 with Δ > 0 on ≥ 60 %, and A < 50 %.
  - **HOLD + OWNER** ("fix pred scale / abstention scoring first") iff A ≥ 50 %.
  - Otherwise **RECORD NULL**: parity remains OBJECTIVE, so the flag stays available at low urgency.
  - Stock (~5 below-F trials) reaches p < 0.05 only at 5/5 — report it as underpowered. Never pool books (different F).
- **Ship:** Step 0 is instrumentation → then default-OFF `OBJECTIVE_THRESHOLD_FLOOR_LIVE` (model-facing, V3 reset).
  Needs a strategy_config constant, a FLAGS.md row and BANNER_FLAGS — other departments.
- **Kill list / removed code: clear.** KILL_LIST.md has no threshold/admission/cost-floor entry; 07:158 KEEPs the cost
  floor (unanimous), 07:159 trade_threshold KEEP-COND ("re-anchor above"); 01_state_map.md:97 D05 = `deferred_owner`;
  08_removed_code.md:172-195 removed only live-loop bumps (Hurst 1.3×, hard-stop 1.5×) — no trainer floor ever removed.

### T2 — An honest meta-calibration holdout

**Facts (OBJECTIVE).**
- `train_meta` drops rows after the **full-frame** 0.88 time quantile (meta_label.py:1119, :1141), chrono-splits 80/20
  (:1235), early-stops on the 20 % (:1251) and fits the shipped legacy isotonic (META_CALIBRATION_MODE 'legacy',
  strategy_config.py:540) on that same slice (:1332). No row is held out from both.
- METADUMP is **landed** (06:03, :829-1043). It writes X, y, entry/exit_e, fold_id (purged k 5), split, net_pct, raw_score,
  raw_oof, p_served/p_legacy/p_purged, plus params and best_iteration, with `reliability_holdout_honest: False` (:1014).
  - raw_score is the full-sample booster, so it is in-sample on every row.
  - In legacy mode raw_oof and p_purged are NaN, and the fold embargo is 0.0 (:134-138), whereas B04.1 binds 0.05.
- reliability_report.py compares POINT Brier/ECE ("lower-or-equal", :8-12) with no interval.
- The meta `pred` feature is primary-IN-SAMPLE unless META_OOF_PRED (False, strategy_config.py:582) is on.
- **Stock boundary mismatch (PROXY, UNVERIFIED — the index read needs a parquet slot, blocked by HW PAUSE).**
  - The stock primary trains on the per-ticker newest `--max-rows 200000` rows (hypersearch_v2.py:261-263) with its own
    boundary (:415-431). Meta's full-frame cutoff over 1.54 M rows is probably much EARLIER.
  - If so, the stock meta rows mostly predate the primary, and meta's "post-cutoff" rows include the primary's training window.
  - A holdout must therefore use the primary's boundary, not meta's cutoff.

**Literature.**
- López de Prado, AFML ch. 7 (2018): purged/embargoed k-fold.
- Bates, Hastie & Tibshirani, "Cross-validation: what does it estimate and how well does it do it?", JASA 119(546):1434-45,
  2024 (arXiv 2104.00673; abstract): naive CV intervals under-cover, and nested CV yields honest SEs.
- Varma & Simon, BMC Bioinf. 7:91 2006; Cawley & Talbot, JMLR 11 2010 (recalled): fitting a tuning step on the evaluation
  rows biases the estimate. That is exactly the legacy slice.
- Chidambaram & Ge, ICLR 2025 (arXiv 2406.04068, v2 2025-02-23; abstract): ECE alone crowns trivial recalibrators;
  always pair it with NLL.
- Roelofs et al., AISTATS 2022 (PMLR 151:4036-54): equal-mass-bin ECE is less biased.
- Rahman & Tabassum, arXiv 2604.02351 (2026-03; abstract, credit panel): measure reliability per forward window
  (forward-chaining).
- Kull 2017's inner-split protocol was not re-verified.

**Designs.**
- **(a) Final-K % replay holdout inside train_meta.**
  - Cost: +12/88 = **+13.6 %** of primary replay inference (+ seq_len warm-up per ticker).
    Crypto adds 31.7 k to 232 k rows; stock adds 184 k to 1.35 M (train_meta loads the full store, :1098).
  - Yield ≈ 13.6 % × n_meta trades. n_meta is UNVERIFIED; B04.2 put the 20 % slice at 40–100, so expect tens of trades.
  - Strength: it is the only design where the primary is out of sample, i.e. the deployment condition, provided it uses
    the primary's boundary.
  - Better shipped standalone as **(a′) `scripts/meta_holdout_replay.py`**: champion + published meta artifacts, with the
    primary boundary re-derived via `load_data(--max-rows/--preset)` + `get_holdout_boundary`. Idle Jetson, hwlock.
- **(b) OOF-of-OOF over the METADUMP npz — ship first.**
  - Outer loop: purged k = 5 on entry/exit with embargo **0.05** set explicitly.
  - Inside each outer-train set, re-run each recipe exactly and score the untouched outer-test rows:
    - L = legacy: 80/20 split, early stop, isotonic on the 20 %;
    - V = that slice through `fit_calibrator(v2=True)`;
    - O = purged_oof: inner `crossfit_oof_predict` k 5, then `fit_calibrator`;
    - R = raw.
  - **Zero training-path change: YES.** The npz holds X, y, entry/exit, params and best_iteration, and the script refits
    the booster (raw_score is in-sample; raw_oof is NaN in legacy).
  - Runs on the Jetson (lightgbm); light at ≤ ~10 k × 15 features, 30 boosters.
  - Honest for the META layer only (the `pred` feature stays in-sample).
  - Sensitivity arm: forward-chaining outer folds. A sign disagreement with the main arm ⇒ "drift — inconclusive".
- **(c) D2 on the same npz:** the same machinery with arms {R, I, P, Beta}, and it presumes V2 is certified. (b) is its
  missing prerequisite: first show legacy loses honestly, then pick its replacement.

**Rule (pre-registered; `scripts/meta_calib_nested.py`, measurement-only, JSON out).**
- Primary statistic: paired per-row log-loss d = LL_L − LL_arm on outer-test rows, weekly-block bootstrap by entry ISO week
  (≥ the 48-bar label span), B = 5000, two-sided 95 %.
- Secondaries: Brier; equal-mass 10-bin ECE with bootstrap CI; veto flip rate at p 0.30; paired decision P&L
  Σ 1[p ≥ .3]·clip(2p, .6, 1.3)·net_pct.
- Precondition: n_meta ≥ 500 per book, else "starved, not decided".
- **Switch** META_CALIBRATION_MODE → 'purged_oof' and/or CALIBRATION_V2 → True (global ⇒ both books agree) iff d's CI
  excludes 0 in the arm's favour on ≥ 1 book with point ≥ 0 on the other, ECE(arm) ≤ ECE(L) + 0.005 and P&L diff ≥ 0 on
  both, and flip rate < 10 %. If O and V both pass, take the lower pooled LL.
- **No-go:** any CI favours L, or P&L < 0.
- **Inconclusive:** keep legacy and follow the B04.2 order.
- (a′) then confirms at holdout n ≥ 200: it must not contradict the sign, and it is logged in the S1-09 ledger as a holdout use.
- OWNER note: reliability_report's point "≤" verdict should quote (b)'s intervals.

### Stock-DB residue check (06:33 `cp` → scratchpad/scout4/stock_v2_study_copy_0633.db; live file never opened)

- One fresh study `stock_v2_search` (id 1, schema v12, optuna 4.7.0), with no study attrs.
- Exactly one trial: #0 **RUNNING**, started 06:29:53 = the 06:29 relaunch (pid 202303, `--trials 25 --mode initial --shadow`).
  It has 19 intermediate values and thr 0.05 on U[0.05, 1.0].
- No FAIL, no second RUNNING, no COMPLETE, and no -wal/-journal file. The ~06:02 kill left no study or trial residue.
- 25 < PRUNE_STARTUP_TRIALS 60 ⇒ random search, no pruning.

### Not done

- No code, tests or heavy runs. The stock boundary claim is a proxy (needs a hwlock index read); n_meta and METADUMP output
  are unverified (no meta run since landing); Clarke–de Silva–Thorley, Gârleanu–Pedersen, Varma–Simon, Cawley–Talbot recalled.
