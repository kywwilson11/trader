# Campaign 2026-08 — Phase 2 Research Synthesis

**Produced:** 2026-08-18, synthesizing 12 round-1 research reports (24 seed topics) and 8 round-2
branch dives against `research/KILL_LIST.md` (read in full) and
`research/campaign_2026-08/01_state_map.md` (defect IDs D01–D40, build packets B02–B24). This file
is written so a reader who never saw the round-1 reports can act on it directly. Where a
recommendation changes model, gate, or sizing behavior it is marked **model-facing** and ships only
through the repo's default-OFF-flag / owner-ruling / challenger-shadow path; **measurement-only**
items ship directly per the deployment convention. Confidence labels (high / medium / low) are the
research agents' own, preserved.

Two packets received no external research this round: **B14 (funding z-score baseline, defect D28)**
— the fix is a code-behavior repair with no literature question attached — and **B21 (cost_regime.py
wiring)** — pure activation of already-certified code. They proceed on the state map's own spec.

---

## B02 — Per-bar predictions dump from backtest.py

**What the research says.** The dump is the single highest-leverage unbuilt producer (both Stage-0
holdout gates wait on it), and two independent research threads add requirements to its schema so it
is built once, correctly:

1. **Emit an hourly mark-to-market book equity series from the policy replay.** The current
   `max_drawdown_pct` is an equal-weight trade-ordinal cumsum, not a sized portfolio path. An hourly
   marked-to-market equity series is the prerequisite for (a) an honest sized-portfolio drawdown and
   (b) the entire class of block-bootstrap Sharpe inference (Ledoit-Wolf 2008 studentized
   circular/stationary bootstrap with Politis-White automatic block length), which wants a
   regularly-spaced return series rather than the irregular per-trade stream. Confidence: medium
   (as a diagnostic; high that the replay change is small).
2. **Persist both blend legs, not just the blended prediction.** `lstm_pred` and `lgb_pred` are
   already computed as locals in `predict_now.py` (lines ~384 and ~402) and are never persisted
   anywhere. Adding the two floats per decision row costs zero extra inference and is the only way
   the static-vs-online blend-weight question (see B12) ever becomes answerable from in-house data.
   Confidence: high. Measurement-only; ships directly.

**Design implication.** The dump schema should carry, per bar and per symbol: timestamp, symbol,
blended prediction, `lstm_pred`, `lgb_pred`, the admission-gate context needed by the rank-gradient
and IC-by-name harnesses, and (per replay run) the hourly mark-to-market equity of the simulated
book. Getting the schema right the first time avoids re-running the Jetson producer.

**Citations.** Ledoit & Wolf 2008 (J. Empirical Finance); Politis & White 2004 with the
Patton-Politis-White 2009 correction (block-length rule D_SB = 2·ĝ²(0)); repo `predict_now.py`.

---

## B03 — Gate repair (effective-n, selection-pressure accounting, shadow promotion test)

This packet got the deepest treatment: two round-1 reports plus two branch dives, one of which is a
Mac-runnable Monte Carlo at the shadow gate's exact geometry (scripts preserved at `/tmp/dm_sim/`).

### B03.1 Effective-n for the Deflated Sharpe gate (defect D02)

**What the literature says.** The current `clustered_effective_n` (sample_weights.py:217) is a
transitive connected-components count: on a book whose trades tile the calendar it collapses toward
the number of idle gaps, and it is non-monotone — adding one bridging trade DECREASES n_eff, so a
busier backtest gets a harsher null. Then `validation.deflated_sharpe_ratio` silently clamps any
n_eff back UP to 10 (line 134), loosening the gate exactly where it should refuse to judge. The
defensible replacement is López de Prado's average-uniqueness construction (AFML ch. 4) applied
once, across names, on calendar-interval concurrency of realized trades.

**Exact formulas.**
- Discretize the union of trade [entry, exit] intervals to hourly bins. Let c_t = number of trades
  open (any name) at hour t. For trade i with open-hour span span_i:
  u_i = (1/|span_i|)·Σ_{t∈span_i} 1/c_t, and **n_eff = Σ_i u_i**. Degenerate spans (exit < entry)
  become point intervals, matching current behavior. This is monotone non-decreasing in added
  trades, reduces to the existing within-ticker average uniqueness in the single-name case, and
  reuses the tested `_avg_uniqueness_block` machinery on a calendar axis (~30 lines of numpy).
  Confidence: high.
- **Fail-closed floor**: delete the upward clamp `ne = min(max(ne, 10), n)`; keep `ne = min(ne, n)`.
  If n_eff < 10, return dsr = 0.0 with status `insufficient_effective_n` (mirrors the existing
  n_obs < 10 branch) and keep echoing the pre-floor value for audit. Gate-behavior change →
  default-OFF flag or decision-queue entry. Confidence: high.
- **Minimum Track Record Length on every failed gate** (pure instrumentation, ships directly):
  MinTRL = 1 + (1 − γ₃·SR + ((γ₄−1)/4)·SR²)·(z_{0.95}/(SR − SR₀*))², denominated in EFFECTIVE
  trades (use n_eff; SR₀* = the `expected_max_sharpe` output). Print "need ≈X more effective trades
  at this SR". Confidence: high.
- **Kish design-effect softening, config-gated, default OFF** (for the stock book, where measured
  pairwise hourly ρ ≈ 0.3–0.5 makes lockstep 1/c_t over-harsh): u_i uses
  1/(1 + (c_t − 1)·ρ̄) with ρ̄ = mean pairwise correlation of hourly close-to-close returns among
  names open in the window, floored at 0.5 (crypto) / 0.25 (stock). ρ̄ = 1 reproduces the
  conservative default. Confidence: medium.
- **Single-correction invariant**: the calendar-concurrency n_eff SUPERSEDES both existing
  corrections (per-ticker uniqueness AND cluster count) — remove the min()-of-two logic, do not add
  a third path. Lo-2002 `serial_correlation_factor` stays OFF and mutually exclusive (CLAUDE.md
  gotcha #4). Confidence: high.

**Citations.** López de Prado 2018 AFML ch. 4; Bailey & López de Prado 2014 (SSRN 2460551); Ledoit
& Wolf 2008; Politis & White 2004 / Patton-Politis-White 2009; Portfolio Optimizer PSR/MinTRL note.

### B03.2 Selection pressure across research iterations (trial pools, holdout reuse, the ratchet)

**What the literature says.** Harvey & Liu 2015 require the trial count to be the CUMULATIVE,
judgment-inclusive count of everything ever tried ("the number does not have to be exact — just a
ball park"), and Bailey-LdP's E[max SR] grows only like sqrt(2·ln N), so conservative counting is
cheap: N = 100 → 2000 moves the deflation bar from ≈ 2.53 to ≈ 3.45 null-widths (+36%). The current
code leaks selection pressure three verified ways: (1) `evaluate_on_holdout` deflates against the
CURRENT study DB's trial count while `adaptive_config.update_after_search` deletes that DB on
categorical expansion but the best_score ratchet persists — pressure survives, the pool count
resets; (2) `backtest.py --gate` deflates against a hardcoded `--trials` default of 100, a
different pool than the fit gate; (3) the 12% calendar holdout of a rolling ~1Y window shifts ~1
week per weekly retrain, so successive winners are re-gated on an ~84–92%-overlapping holdout under
a ratchet — classic adaptive holdout reuse (Dwork et al. 2015).

The round-2 dive settled the "one principled mechanism" question: **online-FDR wealth rules
(SAFFRON / ADDIS / alpha-investing) are REJECTED for this stream.** They require independent (or at
best positively-locally-dependent, mFDR-only) p-values and are documented as most powerful at
T > 1000 tests; this stream is ~52 tests/yr/book at ~85–92% dependence, and the only
dependence-valid variants (LOND, e-LOND) are power-infeasible (a 28-day shadow cannot reach the
required e-value ≈ 40–100). The correct decomposition is: across-weeks stream → cumulative-pool
deflation; within-shadow peeks → group-sequential alpha spending (B03.3); holdout reuse → a
Thresholdout-shaped noisy ratchet at the best_score comparison.

**Exact parameters.**
- **Cumulative trial counter** (confidence: high): persist `cum_trials` in
  `adaptive_state_{asset}.json`; increment by newly completed trials at the END of every hypersearch
  run, BEFORE any DB deletion; `evaluate_on_holdout` gets n_trials = max(cum_trials,
  len(completed_trials), 2). Reset only with the gotcha-#2 best_score reset (objective change = new
  family). Also persist `cum_holdout_gates` (+1 per winner actually scored on the holdout).
- **Unify the policy gate's pool** (confidence: high): `backtest.py` defaults `--trials` from the
  saved model config's holdout report (or reads adaptive-state `cum_trials` directly) instead of
  the hardcoded 100; `run_pipeline` passes the same value to both gates.
- **Overlap-weighted pool for the weekly reuse** (confidence: medium): persist
  [(week_age_k, n_k)] and compute n_trials_eff = Σ_k max(0, 1 − 7k/43.8)·n_k, where 43.8 =
  0.12·365 days is the holdout span. At a steady 100 trials/week this gives n_trials_eff ≈ 360 vs
  100 today (bar up ~13% via sqrt(ln 360 / ln 100)).
- **Do NOT build effective-trials clustering (ONC)** for the TPE pool — it can only lower the bar
  and adds Jetson-hostile machinery; raw cumulative counting is the blessed conservative choice.
  Confidence: high.
- **Thresholdout-shaped noisy ratchet at the best_score comparison** (hypersearch_v2.py ~line 1498;
  confidence: medium): σ_score = std(fold Sharpes)/√NUM_FOLDS from the winner's own folds; accept
  iff new_score > best_score_stored + 2·σ_score + η with η ~ Laplace(scale = σ_score/2) drawn fresh
  per comparison; on accept, store best_score_stored = new_score + Laplace(σ_score/2). At ≤ 52
  comparisons/yr the Dwork budget never binds; log the noise draw (seeded from study name + date).
- **Anti-stacking guard** (confidence: high): adopt EITHER the cumulative-pool deflation OR any
  ad-hoc DSR_MIN "freshness bump" — never both (they price the same reuse twice; document next to
  the existing gotcha-#4 warning in validation.py). If n_trials_eff ships, DSR_MIN stays 0.60.
- **Instrumentation now, before any bar moves** (ships directly; confidence: high): when
  `update_after_search` deletes a study DB, persist {deleted_at, trials_lost, best_score_retained}
  into expansion_history; hypersearch prints "deflating against N cumulative trials (M this
  study)" at gate time; log holdout `fresh_frac` (fraction of holdout calendar unseen by prior
  promotions) for several cycles.

**Citations.** Harvey & Liu 2015 (SSRN 2345489); Bailey & López de Prado 2014; Dwork et al. 2015
(NeurIPS + Science reusable holdout, Thresholdout pseudocode); Robertson, Wason & Ramdas 2023
(Statistical Science, online multiple testing); Ramdas et al. 2018 (SAFFRON); Tian & Ramdas 2019
(ADDIS); Xu & Ramdas 2024 (e-LOND).

### B03.3 Shadow promotion test rebuild (defects D34 and D03)

**What the measurement says.** A numpy-only Monte Carlo at the exact gate geometry (T = 336–672
hourly observations, forecast horizon h = 24 with overlapping losses, cross-sectional correlation
0.3–0.9; scripts at /tmp/dm_sim/) measured the CURRENT pooled dm_hln at 16–20% false-promote for
crypto (K = 6 symbols) and 33–39% for stocks (K = 56) at nominal one-sided 5%. The record-unit
pooling, not the HLN correction, is the broken part. Qu-Timmermann-Zhu 2023 is the definitive
treatment: the panel DM variance must come from the time series of per-period cross-sectional
averages, never from pooled records.

**Exact design (all promote-behavior changes → owner ruling / default-OFF).**
- **Collapse first** (confidence: high): d̄_t = mean over symbols of (e²_champ/v_c − e²_chall/v_x)
  per hourly timestamp; all inference runs on the T-length series; dependence lag = h_max − 1 in
  HOURS. This one change cuts false-promote to 5.4–8.1% across all measured configurations.
- **Promote statistic = Ibragimov-Müller cluster t on block means** (confidence: high): block
  length 2·h_max hours (48h at fb = 24); q = floor(T/block); require q ≥ 6;
  t = √q · mean(block_means)/sd(block_means, ddof = 1), one-sided against t_{q−1} (critical value
  1.943 at q = 7, 1.771 at q = 14). Measured size 5.5–7.5%; power 0.36 at a 10% MSE edge
  (T = 336, K = 56). No LRV estimation, no scipy. Do NOT use 24h blocks (size degrades to
  6.9–9.3%). Keep the collapsed DM-HLN as a logged diagnostic only — under H0 it promotes
  ~1.3–1.6x nominal exactly where IM holds level.
- **If a DM-form statistic is kept**, it must use bandwidth M = 2·h_max with Kiefer-Vogelsang
  fixed-b critical values q(b) = 1.6449 + 2.1859b + 0.3142b² − 0.3427b³, b = M/T (measured size
  5.6–8.3%); the Coroneo-Iacone default M = floor(√T) truncates inside the MA(h−1) band and runs
  7–11%. Confidence: high.
- **Looks**: replace ~14 unadjusted daily peeks (37–57% cumulative false-promote under H0) with a
  Lan-DeMets O'Brien-Fleming-type spending schedule. Two calibrated options: interim at day 21 at
  α = 0.025 plus final at day 28 at α = 0.10 with mean_d > 0 (measured 15–16% overall, floored by
  the owner's final-0.10 rule itself), or final α = 0.05 for ~8–9%. The spending-function form
  A(t) = 2 − 2Φ(Φ⁻¹(1 − α_total/2)/√t) with information fraction t = min(age_days, 28)/28 is
  valid for any look schedule. Daily evaluation continues writing shadow_status.json but never
  promotes. Confidence: high for the mechanism; the alphas are an owner budget choice.
- **Minimums in the right units** (confidence: high): replace MIN_OBS = 200 pooled records (~4
  hours of stock data) with resolved collapsed TIMESTAMPS ≥ 8·h_max per book (192 at fb = 24), and
  require q ≥ 6 IM blocks.
- **Stock-book geometry is unsound at 28 days** (confidence: high): RTH logging gives T ≈ 130 bars
  with h = 24 (h/T ≈ 0.18), where EVERY candidate statistic is oversized (10–12%) and IM is
  infeasible (q ≤ 2). Owner options: a stock-specific MAX_SHADOW_DAYS ≈ 56 calendar days
  (T ≈ 260, q = 5), or stock promotes only at max duration with the fixed-b statistic and a
  documented ~7–10% real level. The same rule blocks 14-day early promotion for fb = 48 crypto
  models.
- **Mixed-horizon comparability** (confidence: high): the homegrown e²/var(r) skill score is the
  right idea (Quaedvlieg 2021 aSPA/uSPA does not apply — it needs a joint multi-horizon forecast
  path — and "Grant 2026" does not resolve to any real paper), but var(r_h) must be FROZEN from a
  long trailing window (≥ 90 days of bars per horizon: ≥ 2160 crypto hours, ~630 stock RTH bars),
  never estimated inside the shadow window: the in-window version injects a Jensen/small-n bias of
  mean_d ≈ +0.0425 at T = 336 favoring the shorter-horizon side (~40% of a genuine 10% MSE edge).
  Quick patch if a long window is unavailable: multiply each side's normalized loss by
  (n_eff − 2)/n_eff with n_eff = T/h_side.
- **The D03 fix — challenger policy-replay gate** (confidence: high): in
  `evaluate_and_maybe_promote`, when decision == 'promote', run backtest.py's replay loading MODEL
  artifacts from the challenger prefix but data/universe from the champion prefix (~20-line
  adapter: `--model-prefix` separate from `--data-prefix`); require net Sharpe > 0 AND
  DSR ≥ DSR_MIN (0.60) on the same windows run_pipeline already uses; on fail, HOLD the challenger
  until the terminal look (do not discard). This keeps the kill-list boundary intact: the shadow
  DM stays a forecast-error test on models; policy economics are gated by the policy replay.
- **Report-only extra**: Pesaran-Timmermann directional-accuracy significance on the collapsed
  series next to the existing hit rates. Confidence: medium.

**Citations.** Qu, Timmermann & Zhu 2023 (Panel DM, eqs. 5–7, 9, Thm 2); Coroneo & Iacone 2020 (J.
Applied Econometrics); Kiefer & Vogelsang 2005; Harvey, Leybourne & Newbold 1997; Ibragimov &
Müller 2010 (JBES); Quaedvlieg 2021 (JBES); Giacomini & White 2006 (Econometrica); Lan & DeMets
1983; Spotify Engineering 2023 (GST vs always-valid comparison); Monte Carlo scripts
/tmp/dm_sim/shadow_dm_size_sim.py and power_and_variants.py (seeds fixed).

---

## B04 — Meta-label and calibration package

### B04.1 Out-of-fold primary predictions (defect D12)

**What the literature says.** The stacking literature is unambiguous (Wolpert 1992; AFML ch. 3;
Joubert 2022): a meta-learner trained on the primary's in-sample predictions learns to trust an
optimism level the live score never exhibits (practitioner reports: 10–20% inflated meta validation
scores). In `meta_label.py` today the 'pred' input comes from scoring the deployed primary on its
own training window, poisoning BOTH channels: the 'pred' feature value AND the entry-population
selection (pred ≥ 0.5x threshold). The infrastructure for the fix already 80% exists:
`hypersearch_v2.py` computes honest purged+embargoed val-fold predictions for the winning config
and throws them away.

**Exact design (model-facing → default-OFF flag `META_OOF_PRED_MODE`, legacy path byte-identical).**
- Persist the 3 walk-forward folds' val predictions at `save_model_atomically` time as
  `{prefix}oof_preds.npz` with arrays ticker, ts_ns (int64), oof_pred (float32), keyed to the
  primary manifest fingerprint (saved_at + score, which meta_meta.json already stores). NEVER
  include holdout-slice predictions — the final 12% must stay out of meta training. Size: a few MB.
  Confidence: high.
- Rework `_gen_meta_rows` to replay per contiguous OOF-covered segment (≤ 3 runs/ticker), using OOF
  predictions for BOTH entry selection and the 'pred' feature; rows outside coverage are DROPPED,
  never backfilled with in-sample values. Log n_oof_rows/n_total; keep the 200-trade floor and fail
  gracefully when coverage starves it. Expect roughly 40–45% of current meta rows to survive.
  Confidence: high.
- Set embargo = 0.05 in the meta calibration cross-fit call (`crossfit_oof_predict(..., k=5)`):
  purged_kfold_indices' fractional semantics are per-test-fold span, and 5% of a k = 5 fold span ≈
  AFML's ~1%-of-sample default. Confidence: medium.
- Close the residual n_iter leak in the fold closures: per-fold early stopping on the last 20% of
  train rows (time-ordered) with lgb.early_stopping(30), min 50 rows, falling back to the captured
  full-sample n_iter when the fold is too thin. Confidence: medium.
- A/B protocol before any flip: train both variants on the same harvest; score both on
  holdout-slice replayed trades; expect the honest val AUC to be LOWER (that is the point); flip
  only if honest holdout veto precision at p < 0.30 ≥ the leaked variant's. Confidence: high.

### B04.2 Calibration mechanics (defect D13, wave-9 activation blocker)

**What the literature says and what was reproduced.** Three canonical fixes, all pure numpy, all in
the non-default `purged_oof` path so they ship as safe pre-flip hardening (the legacy default uses
sklearn isotonic, which already pools ties):
- **Isotonic tie handling**: reproduced in-repo — `IsotonicCalibrator.fit` on scores
  [0.4, 0.5, 0.5, 0.6] with labels [0, 0, 1, 1] maps the tied score 0.5 → p = 1.0 and the answer
  FLIPS with within-tie row order. Canonical fix (de Leeuw 1977 "secondary method", sklearn
  `_make_unique`): collapse duplicate x before PAVA with y' = Σwᵢyᵢ/Σwᵢ, w' = Σwᵢ, run weighted
  PAVA on the unique grid, delete the searchsorted fit[last] hack (~6 lines). Test oracle:
  scipy PAVA on the pooled grid. Confidence: high.
- **Platt target smoothing** (fixes quasi-separation divergence): before IRLS set
  t₊ = (N₊+1)/(N₊+2), t₋ = 1/(N₋+2) (Platt 1999 MAP-under-uniform-prior targets); keep the
  |b| > 50 warning; optional step-halving per Lin-Weng-Keerthi 2007. Confidence: high.
- **Fit the sigmoid on the logit scale**: transform z = logit(clip(p, 1e-6, 1−1e-6)) inside
  `SigmoidCalibrator` fit AND predict — fitting sigmoid(a + b·p) on a probability is misspecified
  (Kull-Silva Filho-Flach 2017: can be worse than the raw scores; Niculescu-Mizil & Caruana fit
  Platt on log-odds for boosting). Confidence: high.
- **Keep ISOTONIC_MIN_N = 1000** — exactly the Niculescu-Mizil & Caruana crossover (Platt wins
  below ~1000–2000 calibration points). Confidence: high. **Defer beta calibration** unless
  post-fix reliability curves still show S-curvature. Confidence: medium.
- **Route the DEFAULT calibration path through `calibration.fit_calibrator`** (round-2 dive): the
  legacy branch at meta_label.py:588 fits raw sklearn isotonic on the 20% slice — 40–100 points at
  current row counts, squarely in the isotonic-overfit regime that produces the D13 constant-p
  hazard. The chooser (sigmoid below 1000) already exists; only the non-default branch uses it.
  Sequence this BEFORE or WITH the OOF flip — a starved retrain with legacy isotonic is the worst
  ordering. Confidence: high.
- Flip criterion unchanged: `compare_calibrations` on the holdout dump, flip only on
  brier_purged ≤ brier_legacy AND ece_purged ≤ ece_legacy (not tied), then re-certify in shadow.

### B04.3 Small-sample starvation policy (round-2 dive on OOF row loss)

**What the literature says.** Clinical minimum-sample theory (Riley et al. pmsampsize criteria)
puts an honest floor for the current 13-feature, 31-leaf/depth-5 booster + isotonic architecture at
n ≈ 1,100–2,300 rows even under logistic-equivalent assumptions, and tree ensembles empirically
need ~10x the events-per-variable of logistic regression (van der Ploeg 2014: RF/SVM/NN unstable
even at 200 EPV). With the OOF restriction cutting rows ~55–60%, the stock book needs ≥ 450–500
pre-OOF rows to keep 200.

**Exact design.**
- Keep the 200-row floor; make starvation LOUD: on len(X) < 200, publish meta_meta.json with
  {veto_disabled: true, n_trades, n_pre_oof_rows, oof_coverage_frac} and surface it in
  pipeline_status instead of the silent return-False (live already fails open to neutral 1.0x).
  First action is simply reading n_trades from the two meta_meta.json files on the Jetson.
  Confidence: high.
- Tiered capacity, model-facing: full current params only at n ≥ 1000; at 200 ≤ n < 1000 a shrunk
  tier (num_leaves = 8, max_depth = 3, min_data_in_leaf = max(20, n//20), feature_fraction = 0.6)
  or a ridge-logistic on 4 features (pred, ATR_Pct, Volatility_12h, hour_sin/cos — Riley gives
  n ≈ 250–400 for 3–4 parameters); no publish below 200. Confidence: medium.
- Set the REAL floor empirically (measurement-only, ships directly): crypto-book subsampling
  learning curve, n ∈ {100, 200, 400, 800, 1600, 3200, all} × 20 seeds, temporal block
  subsampling; floor = smallest n with (plateau_AUC − mean_AUC) < 0.01 AND cross-seed veto
  flip-rate < 10%; extrapolate to the stock book with an inverse-power-law fit (Figueroa 2012).
  Confidence: high.
- Population alignment the cheap, exact way — deployment admissibility here is a DETERMINISTIC
  function of observables (cost floor, lockouts, entry windows, q10 veto), so no IPW is needed and
  there is no positivity violation (training is a superset of deployment): stamp a per-row
  `admissible` flag in `_gen_meta_rows`, keep training the booster on ALL rows (keeps n and the
  deliberate 0.5x-threshold teaching signal), but fit the CALIBRATOR on admissible rows only
  (sigmoid needs ~100 points; record fallback in calib_prov). Report base rate / AUC split by
  admissibility first (measurement, ships now). Confidence: medium.
- Cross-book pooling (crypto+stock booster, book-indicator feature, per-book sample weights
  w_book = N_total/(2·N_book), per-book sigmoid recalibration) ONLY as a gated challenger and only
  if the learning curve proves the stock book cannot self-support. Confidence: low.

**Citations (B04).** López de Prado AFML ch. 3, 4, 7; Joubert 2022 (JFDS); Singh & Joubert 2022;
Meyer, Barziy & Joubert 2023 (JFDS); Wolpert 1992; de Leeuw 1977; scikit-learn isotonic source;
Platt 1999; Lin, Weng & Keerthi 2007; Niculescu-Mizil & Caruana 2005 and the boosting-calibration
paper; Kull, Silva Filho & Flach 2017 (AISTATS); Riley et al. 2019/2020 (pmsampsize); van der Ploeg
et al. 2014 (BMC); Lakkaraju et al. 2017 (KDD selective labels); Figueroa et al. 2012.

---

## B05 — Cost truth (spread stamps, market impact, conditional exit costs)

### B05.1 Spread stamps for both books (defect D04 crypto; stock stamp inflation)

**What the literature says.** EDGE (Ardia-Guidotti-Kroencke, JFE 2024) estimates s² with
sd ~ σ_bar²/√n; the repo's sign=False sqrt(|s²|) folds that noise positive, creating a spurious
floor of roughly σ_bar·(k/n)^(1/4). At hourly frequency with window 35 this floor is ~0.25–0.45%
for BTC (5–25x any plausible true spread) and ~0.10–0.20% for mega-cap stocks (5–10x true 1–5bp
spreads, conservative direction but universe-narrowing). The authors' own guidance: use the highest
frequency with ≥ 2 trades per bar; their minute-vs-daily benchmark correlation vs TAQ moves 56.17%
→ 88.79% precisely in the small-spread sample. **Trust criterion for any window choice: believe an
estimate only where n_bars ≥ (2·σ_bar/s_expected)⁴** — minute bars satisfy this in under a day for
spreads ≥ 3bp; hourly bars would need ~2.4 years for BTC. Hourly EDGE for tight-spread assets is
unsalvageable; this is the same phenomenon behind KILL_LIST line 90, now with the mechanism
quantified (see the kill-list asks section).

The round-2 venue dive removed the need to argue from anecdotes: Alpaca's v1beta3 API serves FREE
historical crypto quotes (bp/ap/bs/as) and trades for venues us / us-1 / us-2 / eu-1 (us-1 quote
history from 2025-10-14), so the crypto ground truth is directly measurable. The 2.5%-spread horror
stories all date to the pre-2022 single-liquidity-provider era, before Alpaca's own orderbook
exchange; no public post-2022 venue statistics exist — the stamp levels MUST come from a one-day
Jetson measurement script (see New Opportunities, item 1).

**Exact design (model-facing; one gotcha-#2 re-harvest event shared with B12/B17).**
- **Crypto stamp, tiered, quote-first** (confidence: medium pending the census):
  Tier 1 (history covered by /quotes): Eff_Spread_Pct = trailing median of minute quoted
  (ap−bp)/mid·100 over the past 24h (1440 obs), floored at 0.02%. Tier 2 (pre-quote history, pairs
  with measured mean trades/min ≥ 2): `edge_rolling` on 1-min bars, window 1440, stamped trailing
  onto hourly rows. Tier 3 (pairs failing density): a per-pair constant equal to that pair's
  measured median quoted spread — NOT the global flat 0.10%. Keep SPREAD_CAP_PCT = 1.5% but log
  cap-hit share (a pair persistently at cap is a CRYPTO_POOL delisting candidate: its admission
  floor is already ≥ 4% predicted move).
- **Stock stamp from minute bars** (confidence: medium): one EDGE estimate per ticker-day from the
  ~390 RTH 1-min bars, smoothed with a 5-day rolling median (n ≈ 1950), stamped trailing onto
  hourly rows; keep floor 0.02% / cap 1.5%. Expected effect: mega-cap round-trip cost drops from
  ~16bp to ~8–9bp, re-widening the tradable universe. Prerequisite check: confirm Alpaca
  Basic-plan minute bars are SIP-sourced, not IEX-only.
- **Zero-cost confirmation FIRST** (confidence: high): the stock harvest already prints
  "[SPREAD] {ticker}: median X% floor-hit Y% cap-hit Z%" per name — grep the last Jetson harvest
  log for AAPL/MSFT/NVDA/SPY. Median ≥ 0.05% on 3+ mega-caps confirms the inflation; median
  ≈ 0.02% means the floor, not EDGE noise, is the binding distortion.
- **sign=True everywhere estimates are aggregated or validated**; negatives → NaN → existing
  fillna(floor), never abs-folded into medians (removes the documented small-sample upward bias).
  Confidence: high.
- **Validation before promotion** (confidence: high): compare per-symbol median stamped spread vs
  median live Alpaca quoted spread captured by order_utils over ≥ 14 days; accept if within ~2x
  (EDGE measures effective; the live gate prices quoted — expect stamp ≤ quoted).
- **Units sanity asserts baked into the measurement script** (confidence: high): bidask returns
  FRACTIONS (0.01 = 1%) and liquidity.py multiplies by 100 — assert measured BTC median quoted
  spread ∈ [0.005%, 0.5%], any alt ∈ [0.01%, 5%], post-fix mega-cap stamps ∈ [0.02%, 0.10%], so a
  wave-7-style units slip is caught mechanically.

### B05.2 Square-root market impact activation (wave-8 survivor code)

**What the literature says.** The square-root law I = Y·σ_daily·√(Q/V_daily) is universal and
calibrated: Y ≈ 0.34–0.69 in 2024–25 US large-cap data (AAPL bias-corrected 0.34, raw 0.69) and
Y ≈ 0.9 for Bitcoin, where it holds over four decades of order sizes including the smallest — no
small-order reprieve for crypto. At the repo's $25k typical notional this is ~1–2bp/side for
DV30 ≥ $1B (a no-op on megacaps) and material only for DV30 ≤ ~$10M or Alpaca alt books (~30bp/side
into a ~$2M/day book). **Calibration defect confirmed in the built code:** `market_impact_pct`
scales by SPREAD (k·spread·√(N/ADV), k = 1.0) instead of daily volatility; since σ_D/spread ≈ 40x
for stocks and ~20x for crypto, the default underprices impact 10–40x versus every empirical
calibration.

**Exact parameters.**
- Re-base IMPACT_K to the volatility scale keeping the code shape: k = Y · median(σ_daily/spread)
  per book → IMPACT_K_STOCK ≈ 20 (Y = 0.5, σ_D ≈ 2%, spread ≈ 0.05%), IMPACT_K_CRYPTO ≈ 18
  (Y = 0.9, σ_D ≈ 3%, spread ≈ 0.15%). Cleaner alternative: switch the multiplier column to a
  stamped trailing 20d daily-return std and set k = Y directly (0.5 stock / 0.9 crypto). Fix the
  exponent at 0.5 (do not fit it). Keep IMPACT_CAP_PCT = 2.0. Confidence: high.
- Harvest keeps DV30 as a COST-ONLY column excluded from features (exact Eff_Spread_Pct pattern);
  crypto V_D = Alpaca-VENUE-local 30d dollar volume from /v1beta3/crypto/us/bars — NEVER the
  training store's volume column, which the yfinance leg overwrites with Yahoo GLOBAL composite
  volume (~7 orders of magnitude off venue volume; defect D08). Supplement with collect-forward
  orderbook depth-within-25bp snapshots (~1 REST call/cycle). Confidence: high.
- Activation A/B expectation: near-zero P&L shift on mid/mega-cap stocks; real haircuts only on
  DV30 < ~$10M names and Alpaca alts. Any other name flipping certification indicates a mis-scaled
  DV30 column. Accept the small double-count with the half-spread rather than modeling the linear
  small-participation regime (not worth the complexity). Confidence: high / medium respectively.

### B05.3 Conditional (stressed-exit) transaction costs

**What the literature says.** Entry-bar-only round-trip pricing (`rt_cost_arr[entry_bar]` in both
backtest.py and meta_label.py) is systematically optimistic on exactly the losing trades: price
jumps coincide with significant spread widening that decays within ~30–60 min (Boudt-Petitjean
2014; Będowska-Sójka 2016), BIS stress studies show 2–4x expansion, and the Oct-2025 crypto cascade
showed 30x over 40 minutes with 98% depth evaporation. On Alpaca's already-wide venue, 1.5–2x
(stocks) and 2–3x (crypto) are the defensible stop-exit multipliers. The trailing 35-bar stamp
cannot capture a 1-bar stress event (diluted ~1/35), so condition on EXIT REASON, which
`exit_walk` already returns.

**Exact parameters (model-facing → default-OFF `STRESS_EXIT_COST_ENABLED`).**
- cost = fee_const + 0.5·s[entry_bar] + 0.5·m(reason)·s[exit_bar], with m = 1.75 for reasons 1
  (hard stop) and 3 (trailing stop) on stocks, m = 2.0 crypto, m = 1.0 for TP/signal/EOD/vertical.
  For the crypto flat path pre-stamp: stop-exit round trip = fee_const + 0.10%·(0.5 + 0.5·2.0) =
  fee_const + 0.15%. Confidence: medium.
- PREREQUISITE per the wave-5 measurement rule (confidence: high): calibrate m from the account's
  own journals — regress realized stop-exit fill-vs-quote-mid shortfall against entry-bar spread
  over ≥ 60 stop exits on the Jetson; replace the literature m with the measured ratio.
- Keep the entry-side admission floor UNCHANGED — the 2.0x MIN_EDGE_MULTIPLE already embeds
  headroom; adding stress to admission would double-count with the realized-exit charge. Only
  realized P&L and meta labels see the conditional cost. Confidence: medium.
- Do not rely on an exit-bar EDGE stamp alone to capture stress (it lags by construction).
  Confidence: high.

**Citations (B05).** Ardia, Guidotti & Kroencke 2024 (JFE 161:103916) + bidask FAQ/vignette/repo
data; Alpaca v1beta3 crypto quotes/trades/orderbooks docs; Nadtochiy et al. 2026 (arXiv
2606.24019); Donier & Bonart 2014 (arXiv 1412.4503); Bucci et al. 2023; Sato & Kanazawa 2024;
Almgren et al. 2005; Boudt & Petitjean 2014 (J. Financial Markets); Będowska-Sójka 2016; BIS Papers
No 2; Amberdata Oct-2025 crash post-mortem; Osler (NY Fed SR 150); SEC 2024 tick-increment rule
(level anchor only).

---

## B06 — De-risk stack consolidation (defect D10)

**What the literature says.** The regime family (VIX ladder, macro sizing_mult, stress rule, HMM,
book-vol scalar) are all noisy estimates of ONE latent risk-off state; multiplying them compounds
the same evidence — under comonotonicity the calibrated joint de-risk factor is the MINIMUM, not
the product (Fréchet bound argument). The code confirms the pathology: VIX enters one entry three
times on identical 15/25/35 breakpoints, so the MODAL regime (VIX 15–25) sizes at 0.56x and crisis
saturates the 0.1 floor, making every other signal inert. On what vol-conditioning is worth:
Bongaerts-Kang-van Dijk (FAJ 2020) show the realized-vol→future-return correlation is ~0.00 in
medium-vol states and −0.10 to −0.22 only in the TOP-QUINTILE state, and CONDITIONAL targeting
(scale only in extremes) beats continuous scaling on Sharpe, max drawdown (−6.6% average,
consistent across 9/10 markets), and turnover. Cederburg et al. (JFE 2020, 103 strategies) find no
systematic OOS Sharpe gain from vol management — reinforcing the repo's Moreira-Muir kill — so the
success metric is TAIL statistics, not Sharpe.

**Exact design (model-facing → default-OFF flag, byte-compat test; informed by the co-fire journal
before the flip).**
- ONE VIX tier map only (delete the duplicate macro sizing_mult application): VIX < 25 → 1.0,
  25–35 → 0.5, > 35 → 0.3. The modal 15–25 band moves from a combined ~0.56x to 1.0x (BKvD:
  modal-state cuts are pure foregone exposure). The VIX > 25 Kelly cap may stay (it bounds
  procyclicality, a different term). Confidence: high (collapse) / medium (modal-neutral).
- Aggregate the regime FAMILY by minimum; keep product only ACROSS families:
  f_regime = min(f_vix, f_stress, f_hmm, f_bookvol); tilt = f_regime · f_dd(ladder) · f_corr ·
  alpha-tilts. The drawdown ladder stays product — it measures the account's own state
  (Grossman-Zhou), not market vol. Drop the 0.8 disagreement penalty (min already resolves
  disagreement). Confidence: medium.
- Hysteresis at the tier boundaries: enter at 25/35, exit at 22/31 (~12% gap), or require 2
  consecutive hourly reads both to enter and to leave; instrument tier-flip counts in the sizing
  journal before/after. Confidence: medium.
- Fix the inverted HMM smoothing bug REGARDLESS of the layer's pending cut (regime_detector.py
  200–218): on a label change with count < N, return the PREVIOUS regime (not neutral); require
  N = 3 consecutive observations of the NEW label before switching (the current code does the
  exact opposite). Confidence: high.
- Success metric = tails: MDD reduction 5–10% relative, ES99 1–3%, floor-saturation events → ~zero
  outside VIX > 35, modal-state median tilt ~0.5 → ~0.9–1.0. Sharpe gain is a bonus, never the
  test. Confidence: medium.

**Crypto book: replace VIX with BTC's own trailing-RV extreme state** (round-2 dive; model-facing).
VIX is wrong for crypto on three grounds: BTC-equity correlation DIPS precisely in risk-off
episodes (Wu 2025); by mid-2026 BTC's macro linkage regime-flipped outright (easing-breadth
correlation +0.21 → −0.778; BTC fell ~50% through three 75bp cuts); and VIX is stale ~2/7 of crypto
hours. Design: state = HIGH when today's BTC daily Parkinson realized-range percentile vs trailing
365 days > 80 (BKvD top quintile) → 0.5x; CRISIS > 95 → 0.3x; no boost below the 20th percentile;
TRAILING (not expanding) window because BTC RV structurally declines. Hysteresis reuses the
asymmetric-Schmitt pattern already written in crypto_trend.py: enter HIGH immediately, exit only at
percentile < 65 held ≥ 12 consecutive hourly evaluations; CRISIS exits into HIGH at < 90. Fail-OPEN
to 1.0 on missing RV. The input series (daily Parkinson realized range, 250-day cache) already
exists in volatility.py. Delete the crypto-side VIX ladder AND macro VIX tiers (keep the
stablecoin-peg halt and STLFSI2 — not VIX); the account-level book-vol EWMA stays (a different,
complementary object). VIX remains on the stock book only, collapsed to ONE read. Confidence: high
for the replacement direction, medium for the hysteresis constants. An optional signed refinement
(apply HIGH only when downside semivariance share > 0.55 over 24–72h, because BTC's leverage
effect is inverted and unsigned triggers de-risk blow-off rallies) ships default-OFF as a
measured hypothesis (confidence: low; dovetails with B24).

**Macro release windows stay** (see B10 for the earnings/NFP work): keep FOMC 12:00–15:30 ET and
CPI 06:30–09:30 ET for both books — elevated vol with NO significant return drift (Nazaruk 2025:
62 CPI + 42 FOMC events 2020–2025, all CARs insignificant) is exactly the donate-spread regime the
stand-down assumes, at a cost of ~64 entry-hours/yr. Add a decay tripwire (annual comparison of
BTC |day-0 return| on the last 12 CPI days vs non-CPI days; queue owner removal only after 12
consecutive quiet months — mid-2026 shows 3 quiet prints, not yet grounds). Confidence: high.

**Citations (B06).** Bongaerts, Kang & van Dijk 2020 (FAJ); Cederburg, O'Doherty, Wang & Yan 2020
(JFE); Harvey et al. 2018 (JPM vol-targeting); Grossman & Zhou 1993; Wu 2025 (arXiv 2501.09911);
Nazaruk 2025 (KSE thesis, events tables); crypto.news 2026-08 CPI-decay report; PLOS One 2021
(inverted leverage effect); Grayscale momentum-signals report.

---

## B07 — LLM package (spend authorization statistics + cost engineering)

### B07.1 Fix the llm_eval keep/kill inference (defect D09) — measurement-only, ships directly

**What the literature says.** The b2 spend verdict is currently void two ways: HAC lag counted in
ROW units on a panel with ~6+ rows per timestamp (10–53% false-'keep' in three independent null
simulations) and row-count degrees of freedom. The exact estimator for "general forms of
cross-sectional and temporal dependence" is Driscoll-Kraay 1998, implemented as
cluster-by-timestamp-then-Newey-West-over-clusters; it nests the current `_newey_west_se` as the
one-row-per-timestamp degenerate case. The long-horizon literature (Boudoukh-Israel-Richardson JFE
2022; Hodrick 1992) warns that even correct HAC over-rejects when h is large relative to the
effective sample, so the estimator must be paired with a hard power gate. Hodrick reverse-regression
standard errors are NOT adoptable here (they need a single common horizon; llm_eval pools mixed
horizons).

**Exact parameters.**
- Group rows by t0 hour into G clusters; h_g = Σ_{i∈g} x_i·e_i (k-vector);
  S = Σ_g h_g h_g' + Σ_{l=1..L} (1 − l/(L+1))·(G_l + G_l') with G_l = Σ_g h_g h_{g−l}' stepping l
  over sorted cluster index; Cov = pinv(X'X)·S·pinv(X'X)·G/(G−1); p-value from t_{G−1}. Lag
  L = max(forward_bars) − 1 counted in cluster (hour) steps; rename hac_lag_rows →
  hac_lag_hours. ~25 lines of numpy. Confidence: high.
- Make the pseudo-replication check a HARD gate: if effective_n = span_hours/forward_bars < 20 or
  n_distinct_t0 < 25, verdict = insufficient_power regardless of the p-value (replaces the current
  warning-string suffix). Preferred operating floor for a keep/kill spend verdict: effective_n ≥
  30. Confidence: high.
- Re-base the power floors: keep MIN_POWER_N = 60 rows but ADD MIN_POWER_T0 = 120 distinct t0
  hours AND the effective_n ≥ 20 gate; mirror all three in prompt_ab.decide_adopt. Confidence:
  medium.
- Ibragimov-Müller time-block cross-check: K = 8 contiguous blocks, per-block OLS, t-test the K b2
  estimates with t_{K−1}; report as b2_im_p next to the DK p; disagreement = distrust the DM.
  Confidence: medium.
- Also fix the stock horizon bar-count-as-hours slip (~3.4x stretch) flagged in D09 while in the
  file.

### B07.2 LLM cost engineering (keep the gate measurable without tripping the cap)

**What the research established.** Anthropic prompt caching is NOT the lever for this workload:
minimum cacheable prefixes are model-dependent and non-monotonic — claude-haiku-4-5 (the current
default analyst when an ANTHROPIC_API_KEY exists) requires 4096 tokens before cache_control does
anything, and the analyst's static prefix is under 1000 tokens, so a marker silently no-ops. Even
on Sonnet 5 (min 1024) the savings ceiling is ~20% of input cost and the 600s cadence outlives the
5-minute default TTL (pure write-premium loss). The REAL levers, ranked:
- **Pin the analyst role to gemini-2.5-flash-lite** (confidence: high): at ~183–288 calls/day ×
  ~2.8K input + ~400 output tokens, Haiku 4.5 costs ~$0.88–1.38/day and trips the $1.00
  _DAILY_COST_LIMIT mid-day — silencing the gate AND starving the n ≥ 60 llm_eval sample;
  flash-lite is ~$0.08–0.13/day (~10x headroom). 2024–26 benchmarks show mini/haiku-class parity
  with frontier models on bounded-schema financial scoring. Keep Anthropic/OpenAI as fallback
  chain.
- **Activate the already-built evidence-hash dedup cache** (confidence: high):
  analyst_dedup_ttl_sec = 3600 (config-only; veto-boundary bypass already engineered). Largest
  savings in static overnight hours. Note the state map's D33: the journal identity fields must
  land so cached re-serves stop counting as fresh observations.
- **Make _record_cost cache-aware and fix placeholder prices** (confidence: medium;
  instrumentation, ships directly): Anthropic cost = in·p_in + 1.25·cache_write·p_in +
  0.10·cache_read·p_in + out·p_out (the normalization currently discards the cache fields); Gemini:
  subtract 0.75·cachedContentTokenCount·p_in (implicit caching is default-on for 2.5 models — the
  ledger currently overbills). Correct the self-declared gpt-5.4 placeholder rows via
  config['pricing'] once verified.
- Do NOT add cache_control under the Haiku default; revisit only if the analyst moves to Sonnet 5
  AND the static prefix exceeds 1024 tokens — then system-block cache_control with ttl '1h' and
  verify usage.cache_read_input_tokens > 0. Confidence: high.
- Calendar note: claude-sonnet-5 intro pricing ($2/$10 per MTok) ends 2026-08-31, then $3/$15 —
  the repo table's steady-state rows are already correct; don't "fix" them downward from an August
  bill. Confidence: medium.

**Citations (B07).** Driscoll & Kraay 1998 (REStat); Hoechle 2007 (xtscc); Boudoukh, Israel &
Richardson 2022 (JFE / NBER w27410); Hodrick 1992; Qu, Timmermann & Zhu 2023 (IM cluster theorem);
Anthropic prompt-caching docs (per-model minimum prefix table); Google implicit-caching and Batch
Mode docs; AIMultiple 2026 sentiment benchmark; Google AI forum flash-lite caching thread.

---

## B08 — Execution safety (idempotency, cancel races, reconciliation) — defects D16/D18/D19

**What the practice literature says.** The codebase already implements several best practices
(judge-by-filled_qty in base_loop, fill-during-cancel race check, scoped cancels, 2-phase verified
flatten) but has four textbook gaps, each mapping to a standard broker-API safety pattern. These
are execution-reliability changes, not model-facing, but they alter live behavior — ship with tests
and fail-closed conventions, not silently.

**Exact designs.**
- **Idempotent submit-recovery (D18)** (confidence: high): one client_order_id per INTENT, minted
  once, reused across retries. Classify submit exceptions TERMINAL (4xx validation: never retry)
  vs AMBIGUOUS (timeout / reset / 429 / 5xx: may have landed). On AMBIGUOUS:
  api.get_order_by_client_order_id(coid), 2 attempts at t+1s and t+3s; found → adopt the order;
  404 both times → resubmit with the SAME coid (max 1); a 422 "must be unique" on resubmit means
  the original landed → re-query. Apply to entry submits, `_execute_stop_exit` market sells, and
  `emergency_flatten`. The dedup window is active-orders-only, so query-before-retry is the
  load-bearing half.
- **Never cancel market orders at timeout (D19)** (confidence: high): the timeout-cancel branch in
  `manage_order_lifecycle` applies only to LIMIT orders; market orders poll every 2s up to 60s to
  a terminal state (filled/canceled/expired/rejected — pending_cancel is NOT terminal), no cancel
  ever; if still non-terminal, return freshest state and resolve via position qty; never resubmit
  while a prior order is non-terminal.
- **Periodic broker-vs-local reconciliation sweep** (confidence: high): every ~15 cycles, one
  scoped list_positions + one scoped open-orders list diffed against self.positions and tracked
  stop_order_ids. Rules: (a) broker long in-universe not tracked → adopt via the
  reconstruct_positions shape, place protection immediately, alert once; (b) tracked position
  absent at broker → 2-consecutive-miss rule before recording a desync exit (a transient
  verify_position error can no longer drop a live position); (c) qty mismatch > 0.5% → sync to
  broker; (d) ORPHAN ORDERS: any open in-universe order with no tracked position and no tracked
  stop id → cancel + alert (kills the crypto GTC resting-stop double-sell risk, the missing
  intra-day orphan sweep from the state map).
- **Liquidation confirmed by position state, not order status (D16-adjacent)** (confidence:
  medium): after any unconfirmed exit, authority is broker position qty; distinguish not-found
  (flat, authoritative) from transient error (unknown) — 3 attempts 2s apart, consecutive
  not-found required; emergency_flatten retries with FRESH qty, never stale.
- **Consume order_stream's already-recorded server-side stop/bracket fill events** (confidence:
  medium): behind TRADER_ORDER_STREAM=1, consult the cache before the REST get_order in
  `_manage_stops` (cache is hint, REST stays authoritative); feed terminal cache hits into the
  reconciliation sweep. The write path already exists; consumption is the "future work" its
  docstring names.

**Citations.** Alpaca order-lifecycle and error docs (422 duplicate client_order_id semantics,
get_order_by_client_order_id); Hummingbot connector architecture (streaming primary + periodic
REST reconciliation backstop); standard idempotency-key / query-before-retry patterns.

---

## B09 — Kelly repair (de-censoring plus small-sample shrinkage) — defect D06 adjacent

**What the literature says.** The optimal Kelly fraction under estimation error is not a fixed 1/2
but a signal-to-noise shrink: Rising & Wyner (IEEE ISIT 2012) prove fractional Kelly equals full
Kelly on a shrunk edge with optimal weight collapsing, in the scalar per-book case, to
**c\* = max(0, 1 − 1/t²)** where t = (mean trade edge)/(SE of that mean). At this book's scale
(n = 50–200 trades, mean pnl ~0.2–0.5%, sd ~2%) t ≈ 1–2, so c\* ≈ 0–0.75 — the fixed half-Kelly
OVER-bets noisy windows and under-bets after sustained evidence. The existing prior_n = 50
pseudo-trade shrink is exactly a Beta(25,25) conjugate posterior mean and handles the MEAN bias;
c\* supplies the missing VARIANCE-aware term. On censoring, the first-order fixes are structural,
not model-based: the 'estimated'-record exclusion (worst on stock TP winners — defect D06) deletes
winners preferentially, and the residual right-censoring is handled by sampling trades by ENTRY
time and marking still-open positions to market.

**Exact parameters (sizing changes → default-OFF flag + Stage-0 measurement per the wave-5 rule).**
- t-stat-adaptive fraction in `compute_kelly_fraction`: on the confirmed-trade window (last 200 by
  ts, min 50): m = mean(pnl_pct), s = std(ddof=1), t = m/(s/√n); c\* = clip(1 − 1/t², 0, 1);
  return clip(c\*·kelly_f_raw, 0.05, 0.25) where kelly_f_raw is the already-shrunk full Kelly.
  t ≤ 1 → c\* = 0 → the 0.05 floor applies. Confidence: high.
- Keep the pseudo-count prior EXACTLY as-is (it is already the Bayesian answer); do not stack
  another prior on top of c\* (double-shrink). Confidence: high.
- De-censor: Kelly stats over trades ENTERED in the lookback, appending open positions as
  pseudo-closed at last price with estimated exit cost from fees.round_trip_cost_pct
  (measurement-only sampling change, ships directly). The D06 data-side fix (fetch the TP leg's
  real fill via its order id) restores the winners themselves. Confidence: medium.
- Do not shrink the window below 50 trades (SE of a win rate at n = 50 is ~7pp; c\* makes thin
  windows self-enforcing). Confidence: high.

**Citations.** Rising & Wyner 2012; Baker & McHale 2013 (Decision Analysis); Meyer-Barziy-Joubert
(calibration improves fixed sizing maps); Chopra & Ziemba 1993 (already cited in repo).

---

## B10 — Scheduled-event risk windows (earnings weekend leak D07; NFP stand-down)

**What the literature says.** Managers strategically shift BAD news to after-hours and Fridays
(deHaan-Shevlin-Thornock 2015; Michaely-Rubin-Vedrashko 2016), so weekend-adjacent prints are
negatively selected — the worst possible population for a long-only book whose calendar-day
buffers provably leak over weekends (Friday's today+2 window ends Sunday and misses Monday-BMO
reporters). Release-day risk premia concentrate on CPI, FOMC, AND NFP days (Fed IFDP 1376), with
400–600% vol spikes in the 15 minutes post-NFP; the pre-FOMC drift collapsed post-2015
(Kurov-Wolfe-Gilbert: 44bp → 9bp), so the existing FOMC stand-down forfeits no drift.

**Exact designs (both cheap; entry-gating changes, not model-facing).**
- **Trading-day-aware earnings buffers** (confidence: high): add a pure helper
  next_trading_days(date, n) skipping Sat/Sun plus a static _NYSE_HOLIDAYS_2026 tuple (9 dates,
  refreshed annually like macro_calendar's tables; no API dependency, fail-open/fail-closed
  semantics untouched). blocks_overnight_hold: block if any report date d satisfies
  today ≤ d ≤ today + 2 TRADING days (Friday maps to {Fri…Tue}; weekdays byte-identical).
  earnings_within_days(n): horizon = today + n trading days. reported_recently: true if
  prev_trading_day ≤ d < today (weekend dates included) or d == today with hour 'bmo' — Monday now
  catches Friday-AMC and weekend reporters.
- **NFP stand-down** (confidence: high): add ('NFP', NFP_RELEASE_DAYS, 06:30, 09:30 ET) to
  macro_calendar._WINDOWS — identical shape to CPI; 12 static BLS Employment Situation dates for
  2026, verified against bls.gov before shipping. Cost: ZERO stock RTH entry hours (window closes
  at the open); crypto ~36 h/yr (~0.4%). The crypto-side benefit is protective-tail-only (NY Fed
  Benigno-Rosa: BTC responds mainly to CPI), justified by near-zero cost.
- **Leave FOMC at 12:00–15:30 ET** (confidence: medium): the drift is dead so the stand-down
  donates nothing; the optional 13:00 trim reclaims 8 entry-hours/yr and is an owner taste call.
- Do NOT widen any window past 09:30 ET on literature alone — measure release-day first-half-hour
  RV and entry outcomes in-house first (extend to 10:00 only if RV ratio > ~1.5x AND outcomes are
  measurably worse). Confidence: low (deliberately).

**Citations.** deHaan, Shevlin & Thornock 2015 (JAE); Michaely, Rubin & Vedrashko 2016 (JAE);
Kurov, Wolfe & Gilbert 2021 (FRL); Lucca & Moench 2015 (JF); Benigno & Rosa 2023 (NY Fed SR 1052);
Fed IFDP 1376; Nazaruk 2025.

---

## B11 — Daily-bars / HAR-RV activation (defect D30)

**What the literature says.** The standard HAR estimation window is 1000 days
(Bollerslev-Patton-Quaedvlieg 2016; Clements-Preve 2021), with 250–756 as the low end; 60 days is
defensible only as a hard FLOOR paired with variance-reduction remedies — and the repo already has
the two most important ones (log transform, BPQ insanity filter). What actually blocks activation
is plumbing: live fetches supply ~10 crypto days / < 60 stock trading days, so `har_forecast_sigma`
returns None on every call and sizing silently uses the inferior GARCH while burning CPU refitting
it hourly. The one genuine soundness gap is stocks-only: Parkinson range over 6.5 RTH hours misses
overnight variance (~20–40% of whole-day variance).

**Exact design.**
- **Fix the data starvation via a persisted per-symbol daily-RRV store** (confidence: high):
  one-time seed (crypto ~1500 hourly bars; stock fetch limit ~450–500), then append one RRV row
  per symbol at each day-roll, capped at _HAR_WINDOW = 250 rows (~10 symbols × 250 floats,
  negligible; also removes the hourly GARCH-refit burn once the daily cache hits).
- **Exclude the forming/partial day from `daily_realized_range`** before both fitting and the
  forecast regressors (drop days with bar count < 24 crypto / < 6 stock RTH, or drop the last
  calendar day unless complete); key the daily cache on the last COMPLETE day. Mirrors
  `drop_forming_bar` at the day level and fixes the whole-day caching bug in the same motion.
  Confidence: high.
- **Whole-day scaling for stocks (Hansen-Lunde 2005 via the Martens scaling estimator)**
  (confidence: high): multiply the RRV series (equivalently rrv_hat) by
  c = Σ_d (r_cc,d − r̄)² / Σ_d RRV_d over the same trailing ≥ 60-day window, clamped to
  [1.0, 2.5]; expected c ≈ 1.25–1.5 for liquid US names. One constant jointly corrects the
  overnight omission AND the Parkinson discretization bias. Crypto: c = 1 identically. Refresh at
  the daily cache roll. Per-bar sigma = √(c·rrv_hat)/√6.5 stays consistent with
  compute_vol_adjusted_size's √1638 convention.
- **Shrink coefficients toward a canonical log-HAR prior until the window matures** (confidence:
  low, flagged as synthesis not published result): beta_used = λ·beta_ols + (1−λ)·beta_prior with
  λ = n_rows/(n_rows + 120); prior (b_d, b_w, b_m) = (0.40, 0.30, 0.25), intercept from the
  sample mean. Effect vanishes by n ≈ 250. Verify the ≥ 38-row guard at exactly 60 days
  (off-by-one territory).
- **Keep the log-HAR spec; do NOT add WLS or HARQ** (Clements-Preve: log is the best transform and
  WLS gains largely substitute for it; HARQ was outperformed by the simple remedies; the BPQ clamp
  already guards pathological forecasts). Confidence: high.
- The activation is model-facing sizing behavior (GARCH → HAR switch): default-OFF flag / owner
  path. A one-shot offline QLIKE gate on the harvest parquet (HAR-with-c-scaling vs current
  EGARCH, per symbol) converts the certification from literature-backed to self-measured before
  the flip (measurement-only, ships directly).

**Citations.** Clements & Preve 2021 (J. Banking & Finance); Hansen & Lunde 2005 (J. Financial
Econometrics); Martens & van Dijk 2007 (J. Econometrics); Bollerslev, Patton & Quaedvlieg 2016;
Corsi 2009; Parkinson 1980 corpus; MDPI IJFS 2026 (HAR vs GARCH on crypto high-frequency).

---

## B12 — Model-fit retrain batch (final refit, blend weight, holdout certificate)

This is the bundle that shares ONE gotcha-#2 re-harvest/retrain event with B05's stamps and B17's
universe work (study DBs deleted, adaptive best_score reset).

### B12.1 Final refit on the full window (defect D22)

**What the literature says.** Standard practice (sklearn refit=True convention; AutoGluon
refit_full; the CV-selection literature) is: select the CONFIG by cross-validation, refit ONE model
on all pre-holdout data, verify once on the untouched holdout. hypersearch_v2 currently deploys
`best_fold_state` — the max-val-Sharpe fold's checkpoint — which (1) can ship a model trained on
only the OLDEST ~48% of the year (recency loss under drift) and (2) inflates the expected deployed
edge by E[max of 3 correlated folds] ≈ 0.85·std_sharpe (computable with the repo's own
`expected_max_sharpe` at n_trials = 3), selecting the checkpoint that got the easiest val regime.

**Exact design (model-facing → challenger/shadow; first Jetson retrain needs the gotcha-#2 study
reset).**
- After Optuna picks the winner, train one final model on ALL search-region data (everything ≤
  `get_holdout_boundary`, purged so label windows complete before the boundary); scaler refit on
  the full region; epoch budget FIXED with no early stopping: epochs_refit = int(median of the
  winning trial's per-fold best epochs) — the "collective early stopping" workaround AutoGluon
  documents. The existing `evaluate_on_holdout` gate (Sharpe > 0, DSR ≥ 0.60) then scores THAT
  artifact. Confidence: high.
- Checkpoint soup adapts to the no-validation refit: uniform-average the LAST K = 4 epoch
  checkpoints (SWA tail averaging, Izmailov 2018) instead of K-best-val-loss; NEVER soup across
  folds (different scalers = different input spaces). Confidence: high.
- Report a curse-corrected expected edge wherever best_fold_sharpe is surfaced: subtract
  ≈ 0.85·std_sharpe; treat avg_sharpe, not best_fold_sharpe, as the deployable-edge point
  estimate. Confidence: high.
- Regime tripwire: if fold_sharpes[-1] < 0 while the trial mean is positive, flag the winner for
  owner review before the holdout gate (refit-on-all would bake in a stale-regime model); warn,
  do not auto-block. Confidence: medium.
- Keep the Optuna objective exactly as-is (mean − 0.5·std is already curse-resistant at the trial
  level). Jetson memory: refit region ≈ fold-2 train + ~15%; budget ≈ 1.15GB host numpy — inside
  the envelope, verify against MemoryMax=6G on first run. Confidence: high / medium.

### B12.2 Blend weight (defects D23 and D25)

**What the literature says.** The forecast-combination puzzle is settled for the 2-model case:
estimated weights beat 0.5 only when the true weight is far from 1/2 AND estimation noise is small
(Claeskens et al. 2016; Stock & Watson 2004 — the LEAST adaptive schemes win, and discounted-MSFE
with delta 0.9/0.95 was "typically no better, sometimes worse" than delta = 1). The round-2 dive
REJECTED the online discounted-MSFE tracker: the regime-flip premise is unsupported
(Wong-Barahona 2023: GBDT ≥ NN in BOTH vol regimes; Remlinger 2023: online mixture value was
drawdown insurance across 13 experts, not a 2-model case), and an online weight would break
train/serve parity, escape the DSR certificate (worsening D25), and learn from ~8 effective
observations/week. Additionally the hardcoded 0.6 LSTM tilt has the prior's sign BACKWARDS
(Grinsztajn et al. 2022: trees remain state-of-the-art at this tabular data size).

**Exact design (model-facing; wire at the B12 retrain event).**
- Wire `blend_fit.fit_blend_weight` into the retrain path with objective='nnls' as the ESTIMATOR
  (w_hat = ((y − lgb)·d)/(d·d), d = lstm_oof − lgb_oof, clipped [0,1]); the 101-point Sharpe grid
  becomes a logged DIAGNOSTIC only (its argmax over correlated candidates is an unshrunk winner's
  curse). Write 'lstm_weight' into the model config predict_now/backtest already read (zero
  changes there). Confidence: high.
- Significance gate on deployment: SE(ŵ) = √(σ̂_ε²/Σd_i²) with the variance multiplied by the
  label-overlap correction (n_eff = n_oof/forward_bars); deploy the shrunk ŵ only when
  |ŵ − 0.5| > 2·SE(ŵ), else return exactly 0.5. Keep shrink_to = 0.5, shrink_lambda = 0.5
  (effective w ∈ [0.25, 0.75], both legs always alive). Confidence: high / medium.
- OOF symmetry is mandatory: the LGB leg needs its own matching 3-fold walk-forward OOF pass (same
  fold boundaries, purge, embargo; holdout excluded) — honest-LSTM vs in-sample-LGB would
  spuriously over-weight LGB. Confidence: high.
- Change the hardcoded default lstm_weight 0.6 → 0.5. Confidence: high.
- Smooth ACROSS retrains instead of tracking within them: w_used = 0.5·w_new + 0.5·w_prev
  (persist w_prev), clamped [0.25, 0.75] — all the safe time-variation at weekly cadence with zero
  live-path code. Confidence: medium.
- **Close D25 for the weight**: issue the holdout DSR certificate against the BLENDED predictor
  with the fitted w (score w·lstm_oof + (1−w)·lgb_oof; gate promotion on the blend's DSR).
  Confidence: high. (The q10 tail veto remains uncertified — an open follow-up.)
- Forecast-encompassing check at the same event: report the unshrunk NNLS w; if < 0.15 on BOTH
  books, flag for the owner — the real prize would be dropping torch from the live loops (Jetson
  memory priority #1). Do not auto-drop. Confidence: medium.
- Revisit-online trigger (documented, not built): only if the per-leg journals (B02) later show
  the counterfactual discounted path (Stock-Watson eq. 4; per-hour delta for half-lives 2/6/13
  weeks = 0.99794/0.99931/0.99968) beating the static weekly-refit weight by > 2% relative MSE
  AND higher policy-replay Sharpe, with w_t persistently outside [0.35, 0.65] for ≥ 4 weeks.

**Citations (B12).** Stock & Watson 2004 (J. Forecasting); Claeskens, Magnus, Vasnev & Wang 2016
(IJF); Diebold & Shin 2019; Wang, Hyndman, Li & Kang 2022 review; Grinsztajn, Oyallon & Varoquaux
2022 (NeurIPS); Wong & Barahona 2023; Remlinger et al. 2023; AutoGluon refit_full docs; Izmailov
et al. 2018 (SWA); Wortsman et al. 2022 (model soups); Bailey & López de Prado 2014.

---

## B13 — Sentiment repair (fix, don't delete)

**What the literature says.** Verdict: FIX the stack, demote the lexicon to fallback-only, measure
the gate before trusting its multipliers. (1) Staleness: Tetlock (RFS 2011) shows price reactions
to STALE news (textual similarity to the prior 10 stories) partially REVERSE within a week —
trading on reprints is actively harmful, and the repo already owns the right tool (novelty.py's
Jaccard shingles) but only as a filter, not a weight. (2) Decay: salient news is priced within a
session intraday; for an hourly gate, headlines older than ~2–3 days are noise. (3) Lexicons
plateau at ~0.50–0.70 accuracy vs FinBERT/LLM 0.72–0.97; the verified in-repo bug is exactly the
documented phrase/word interaction failure: `_score_text` scores the phrase "rate cut" +1.0 then
re-tokenizes the FULL text and scores "cut" −1.0, nulling rate-cut headlines.

**Exact designs.**
- Phrase-mask fix (~5 lines): after each accepted Phase-1 phrase match, blank the matched span
  (replace with spaces) before Phase-2 tokenization; negation window unchanged. Changes
  Daily_Sentiment values → model-facing: default-OFF flag with a byte-compat test, or owner queue.
  Confidence: high.
- Novelty as a staleness WEIGHT in aggregation: w = 1 − max_jaccard (novelty.py already computes
  it); hard floor w = 0 above jaccard 0.6; retention window ≥ 72h. Confidence: high.
- Exponential age decay in `get_news_sentiment`/`_aggregate_scores`: score_i ×= exp(−age_h/H) with
  half-life 12h for the hourly gate; cap lookback at 72h; the daily backfill keeps its 1-day-lag
  semantics. Confidence: medium.
- Instrument `sentiment_gate()` BEFORE tuning it (measurement-only, ships directly): journal
  (symbol, multiplier, component reasons, subsequent 12/24/48h return); score it llm_eval-style
  with a keep/kill verdict at n ≥ 60 gated decisions. The 0.15x/0.35x/1.35x steps are hand-set and
  unmeasured — freeze until measured. Confidence: high.
- Do NOT expand the keyword lexicon (dictionary ceiling ~0.50–0.70 vs the integrated LLM scorers
  0.74–0.95); keep the LLM 0.7 / keyword 0.3 blend; no local FinBERT without a separate Jetson
  memory feasibility pass. Confidence: high.
- Note the state-map coupling: the D27 refresh fix re-arms the sentiment triple-count — land the
  gate-overlap adjudication with it.

**Citations.** Tetlock 2011 (RFS); Loughran & McDonald 2016; MDPI IJFS 2025 head-to-head (LM 0.501
vs OPT 0.744); Kirtac & Germano 2024; FinLlama 2024 (ICAIF); Cognitive Computation 2021 (negation);
MDPI Systems 2025 (3-day sentiment half-life); EJF 2024 night-and-day.

---

## B15 — Data-store integrity (forming-bar discipline; the found parity hole)

**What the audit established.** The primary prediction path is CLEAN (predict_now calls
drop_forming_bar; daily-window features are shift(1)-mapped), but ONE concrete hole was found:
`panel_ranks.compute_live_panel_ranks` (~line 160) fetches bars and computes features WITHOUT
drop_forming_bar, so during market hours every live CS_Rank_* / CS_Dispersion / CS_Breadth /
MS_Interact value is drawn from a partial-bar distribution the model never trained on, and is
timestamped one hour ahead of the prediction frame it is injected into. dv30 also includes the
partial bar's volume in membership ranking. This is one line to fix
(`bars = drop_forming_bar(bars, bar_seconds=3600)` after the fetch) but it changes live feature
VALUES → route as a decision-queue/owner item, not a silent ship. Confidence: high.

Two forward-looking conventions come with it: the day-level forming-DAY exclusion for HAR lives in
B11; and IF any minute-derived feature is ever built (see B24's verdict that none is currently
needed), the closed-window convention must be established at harvest BEFORE the feature ships —
retrofitting parity is far more expensive. The remaining D38 sites (live ATR/GARCH partial bar,
shadow evaluator, Stage-0 replays) are covered by the state map's own spec; no external literature
was needed for them.

**Citations.** Zinkevich Rules of ML #32 (shared transform code); Vertex AI training-serving skew
docs; TradeThatSwing partial-candle note.

---

## B16 — Verification stack (golden-row parity; ab_check/CI context)

**What the practice literature says.** Distribution monitoring must be paired with shared-transform
discipline and a ROUND-TRIP parity test (Rules of ML #32; feature-store practice runs per-feature
parity in CI at 1e-6 tolerance). The repo's strongest asset here is already having ONE
compute_features for harvest and live; the missing piece is the harness that proves it stays true.

**Exact design (rides with the D36/D37 ab_check + CI hardening the state map specifies).**
- **Golden-row parity test** (confidence: medium): at harvest, persist the last 5 computed feature
  rows + their raw input bars for one symbol per book; at pipeline start on the Jetson, recompute
  those rows through the LIVE path (fetch → drop_forming_bar → compute_features → injections) on
  the same raw bars and assert max|diff| < 1e-6 per column; a failing column NAMES the divergent
  transform. Write the pure comparison kernel + synthetic test on the Mac; the harness itself is
  Jetson-gated.
- The per-feature PSI monitor and degenerate-value alarm that complement this at runtime live in
  B19.

**Citations.** Zinkevich Rules of ML #32; APXML feature-store skew-diagnosis notes (1e-6 CI parity
convention); Google Cloud Vertex AI skew-monitoring blog.

---

## B17 — Crypto breadth and as-of universe membership

**What the literature says.** Crypto survivorship/delisting bias is 0.93%/yr value-weighted but
62.19%/yr equal-weighted (Ammann-Burdorf-Liebi-Stöckl 2022, 3,904 coins); the EW number is a
micro-cap artifact and does NOT transfer to a 6-name mega-cap book, but the current harvest
hardcodes TODAY'S six winners chosen in 2026 with a 2021 start — training-panel
selection-on-performance, plausibly worse than the stock side's measured +1.6–4.9pp/yr because the
historical top-10 contained LUNA (top-10 April 2022, −99.99% in a week) and the mid-2023 Alpaca
regulatory delistings. On breadth 6 → 10: diversification value is almost nil (pairwise ρ ≈ 0.7+
⇒ effective bets move ~1.26 → 1.30); the real value is ~67% more training rows and cross-sectional
rank granularity (Grinold: IR ∝ IC·√breadth) — expect a modest rank-quality-shaped gain, not a
Sharpe jump.

**Exact design (model-facing; same B12/B05 re-harvest event).**
- Per-bar `in_universe` FLAG (not row filtering): membership = (coin in top-10 by mcap at the most
  recent CoinMarketCap weekly historical snapshot) AND Alpaca-tradeable at that date. CMC weekly
  snapshots are FREE HTML back to April 2013 (~260 pages for 2021–2026; one-time scrape cached to
  a small csv), applied with a 1-week publication lag for PIT safety. Panel ranks rank only
  in-universe names per bar; features/labels stay computed for all harvested names. Confidence:
  high.
- Harvest the widest retrievable Alpaca USD-pair set (~20 names) for the training panel while
  trading only CRYPTO_POOL; additions need ≥ 180d of bars before entering ranks. ~20 names × ~5y
  hourly ≈ 0.9M rows — within the Jetson pandas budget. Must verify on the Jetson whether Alpaca
  still serves bars for pairs delisted in 2023; fall back to yfinance for delisted windows only
  (with source-provenance stamped, given D08). Confidence: medium.
- Quantify the bias in-house: rerun the crypto policy backtest winners-only vs as-of and report
  the spread (expected +1–5pp/yr optimistic bias; never quote the 62%/yr EW number for this book).
  Confidence: medium.
- Report effective bets in decision_report (measurement-only, ships directly):
  N_eff = N/(1 + (N−1)·ρ̄) with ρ̄ = trailing 90d mean pairwise hourly correlation; expect
  ≈ 1.2–1.4 — MAX_BOOK_RISK_PCT should keep treating crypto as essentially one correlated
  position, and breadth is NOT grounds to revisit the HRP/ERC kill. Confidence: high.

**Citations.** Ammann, Burdorf, Liebi & Stöckl 2022 (SSRN 4287573); Concretum 2025 PIT-crypto
method; CoinMarketCap historical snapshots; StratBase dead-coins note; FMPM 2025 and AUT
crypto-momentum papers (majors: limited dispersion, short-side alphas); Alpaca coin-pair FAQ.

---

## B18 — Execution quality (no-trade bands, signal-exit asymmetry, churn brakes)

**What the literature says.** For PROPORTIONAL costs (Alpaca's case) the correct closed form is the
cube-root no-trade band: half-width in weight space Δπ = (3/(2γ)·π²(1−π)²·λ)^(1/3) with λ = one-way
proportional cost (Muhle-Karbe/Reppen/Soner eq. 4.15). Crypto numbers (λ = 0.003, γ = 3, π = 0.25)
give Δπ ≈ 0.037 — ±15% relative — which VALIDATES the existing design of never resizing open
positions and of discrete enter/hold/exit with cooldowns (a no-trade band in time). The genuine
defect the band math exposes is ASYMMETRY: entry requires pred ≥ ~1.2% (crypto, cost-floored) but
the signal exit fires at pred ≤ −0.15% with NO cost term — exits trigger ~8x more cheaply relative
to cost than entries, and an exit-plus-re-entry costs a full ~0.6% round trip. The full
Garleanu-Pedersen aim-portfolio machinery is a dynamic mean-variance optimizer and is
kill-adjacent — do NOT build it; stay at threshold/band/debounce level.

**Exact designs (policy-facing → default-OFF flags; MUST be implemented in the shared
strategy_config-driven path so backtest.py and meta labels replay identical semantics).**
- Cost-scaled signal-exit threshold: exit only when the forecast decline exceeds the one-way cost
  of getting out: exit_threshold = max(trade_threshold,
  ONE_WAY_EXIT_COST_MULT · required_edge_pct(asset, spread)/MIN_EDGE_MULTIPLE/2), start
  ONE_WAY_EXIT_COST_MULT = 1.0 (≈ −0.30–0.35% crypto, ≈ −0.15% floor stocks). Stops/TP/trailing
  untouched — ONLY the pred-driven signal_sell. Measure via policy replay before any live flip.
  Confidence: high.
- Two-reading confirmation on the signal exit (time-axis band widening; resolves the rev-07-02
  1-reading-exit vs 2-reading-stops conflict): require pred ≤ −exit_threshold on 2 consecutive
  hourly cycles; one flag per position, cleared on a non-confirming read. Confidence: medium.
- Prediction EMA smoothing (GP's "down-weight fast signals" in cheapest form), GATED on a Stage-0
  measurement of hourly pred lag-1 autocorrelation from the journals: pred_smooth = α·pred +
  (1−α)·prev with half-life 2–3 bars. If preds are near-white, smoothing destroys timing — measure
  first. Confidence: medium.
- Keep (do not loosen) the discrete turnover brakes — cooldowns, per-symbol daily trade caps,
  MIN_ORDER_NOTIONAL — they ARE the band's time-axis implementation; any relaxation proposal must
  clear the same band arithmetic. Confidence: high.
- The IOC price-cap activation (D21, built and wired to nothing) rides this packet per the state
  map; no external literature was needed for it.

**Citations.** Muhle-Karbe, Reppen & Soner 2016 (arXiv 1612.01302, eq. 4.14–4.15); Garleanu &
Pedersen 2013 (JF); Janecek & Shreve 2004; Whalley & Wilmott 1997; NBIM Discussion Note 1/2018.

---

## B19 — Ops and monitoring (the input-side third leg)

**What the practice literature says.** Today's alarms are output-side only (prediction-PSI + CUSUM);
there is NO input-side per-feature check, and predict_now's neutral-fill (0.0) safety nets actively
MASK pipeline breakage from prediction-PSI because 0.0 IS the trained neutral — a dead injection
path shifts predictions toward the training median, which PSI reads as calm. Industry practice
(Vertex AI) is per-feature monitoring against training-time baselines paired with shared-transform
discipline.

**Exact designs (measurement-only, ship directly; Jetson-cheap: O(60 × histogram) numpy daily).**
- Per-feature PSI monitor: (a) harvest/hypersearch saves 'feature_deciles' {col: 11 quantile edges
  of the UNSCALED training matrix} into the model manifest next to holdout.pred_deciles; (b) live
  logs each symbol's final closed-bar feature vector per cycle to {prefix}feat_history.jsonl
  (same flock pattern and 7-day prune as log_predictions); (c) daily run_check reuses compute_psi
  per column pooled across symbols over 24h, MIN_LIVE_SAMPLES = 50; warn > 0.10, action > 0.25,
  top-5 offenders reported. Multiple-testing guard across ~60 features: page only when ≥ 3
  features exceed 0.25 or any one exceeds 0.50 for 2 consecutive days. Confidence: high.
- Degenerate-value (dead-injection) alarm — the failure PSI under-detects: for each
  injected/centered feature (Funding_*, OI_*, TT_LS_Z, Taker_Imb_24h, SVR_*, CS_*,
  Daily_Sentiment), track the live fraction of values exactly equal to the neutral fill vs the
  training fraction (store train_neutral_frac in the manifest); alert when live − train > 0.30
  sustained 24h. Catches a silently-dead archive/API path within a day. Confidence: high.
- Keep PSI (not Jensen-Shannon) — one metric, one interpretation, shared kernel, established
  0.10/0.25 thresholds and the 2-consecutive-day convention. Confidence: medium.
- D32's drift-churn fixes (warn-day re-trigger; pred_history not cleared on deploy) proceed on the
  state map's spec; the PSI additions above give the retrain trigger an input-side reason code.

**Citations.** Google Cloud Vertex AI model-monitoring blog + OneUptime 2026 walk-through;
Zinkevich Rules of ML #32; APXML skew-diagnosis notes.

---

## B20 — SPY hedge (low-beta roadmap step 2; kill-list survivor #12)

**What the research established.** The round-1 blocking premise is FALSE: the 100-share round-lot
rule applies only to manual locate requests for HARD-to-borrow securities. SPY is ETB — no locate
call at all, $0 borrow, and the real minimum hedge increment is ONE whole share (~$640; fractional
shorting is unsupported). Constraints that do bind: margin account (paper is margin-enabled; $2k
floor), Reg-T ~50% initial / 30% maintenance buying-power consumption, RTH/extended-hours
adjustment only, and the crypto book's BTC beta is unhedgeable on Alpaca. Economics: a CONTINUOUS
hedge costs ~β·(rf + ERP) ≈ 7–8.5%/yr of hedged notional at β = 0.9 because Alpaca pays nothing on
short proceeds — so the round-1 gate (alpha > rf·β) was actually optimistic. The
TREND-CONDITIONAL hedge (ON only when SPY < 200d SMA, ~25–30% of days) cuts the certain rf-forgone
leg to ~0.27x and removes ~63% of beta-driven variance (below-trend vol ~28–30% vs ~13–14% above),
consistent with Goulding-Harvey-Mazzoleni's state-conditional beta economics. beta_ledger.py
already computes every input — but its trend state is NOT lagged (a PIT violation) and must be
fixed before the same series feeds a live trigger.

**Exact design (sequenced behind real Jetson beta-ledger numbers, per the roadmap).**
- Instrument: short whole SPY shares direct (never inverse ETFs — that kill stands); ETB check via
  the Assets endpoint easy_to_borrow flag before each adjustment. Confidence: high.
- Sizing: qty = round(d · beta_smooth · invested_book_equity / SPY_price) with dampening d = 0.5
  (range 0.4–0.6; the Quantitativo replication: beta 0.57 → 0.27 with alpha retained), beta_smooth
  = 20d EMA of the rolling 63-day JOINT-regression SPY Dimson beta (widen ROLLING_BETA_WINDOW from
  30 — 30d rolling betas are the worst-performing estimator class; use the joint SPY+BTC
  regression's SPY coefficient so correlated BTC variance is not double-hedged). Confidence:
  medium.
- Trigger: hedge ON iff PRIOR-DAY SPY close < SMA200 (t−1 lag fixes the PIT flag) with a whipsaw
  guard (2 consecutive closes across the line OR a ±1% band). Confidence: medium.
- Activation gate (owner decision, evidence-backed): activate only if the ledger shows the
  below-200d-trend SPY beta significantly positive (t ≥ 2 at n ≥ MIN_BUCKET_OBS = 15) AND
  annualized HAC alpha > f_hedged·β·rf ≈ 0.27·0.9·4% ≈ 1.0%/yr (replacing the round-1 full
  rf·β ≈ 3.6–4% hurdle). Until then, DE-SIZE — costless beta reduction. Re-evaluate monthly on
  --days 90 ledger runs. Confidence: medium-high.
- Rebalancing: weekly checks; trade only on > 10% notional drift or |Δbeta_smooth| > 0.10; explicit
  maintenance < 5bp/yr of hedge notional. Confidence: high.
- Paper-fidelity corrections (measurement honesty): Alpaca paper does not debit short dividends
  (~1.15%/yr SPY yield) and simulates no interest at all — accrue the dividend cost explicitly in
  the dormant short_cost stack and book the rf-forgone term offline; never read hedge economics
  off raw paper P&L. Confidence: medium.
- Risk accounting: wire the hedge sleeve into risk_budget with NEGATIVE correlation to the stock
  book (the code already supports signed rho, "books can hedge") rather than as a new gross-risk
  consumer. Confidence: medium.
- Fix beta_ledger's trend-state lag (shift the px > sma200 state by 1 trading day) NOW —
  measurement-only, ships directly. Confidence: high.

**Note for the owner:** the trigger's resemblance to the KILLED "SPY<200d short-activation gate"
(wave-5) is addressed in the kill-list asks section — the killed item is a short-ALPHA trigger;
this is timed BETA REMOVAL. Explicit blessing requested before any build.

**Citations.** Alpaca margin/short-selling, HTB-locate, fractional-trading, and fee docs; Israelov
2019 "Pathetic Protection" (JAI); Quantitativo beta-hedging replication; Goulding, Harvey &
Mazzoleni 2023 (FAJ) and Momentum Turning Points (JFE); Asness, Krail & Liew 2001 (Dimson betas);
JFQA 2024 machine-learning betas (short-window rolling betas worst; Vasicek shrinkage); SSGA SPY
lending note; Theta Trend above/below-200d conditional moments.

---

## B22 — Reporting last mile (instruments the other packets emit)

No standalone literature seed targeted B22, but the research round specifies several report-only
instruments that belong to it; collecting them here so the packet has one list (all
measurement-only, ship directly):
- MinTRL ("need ≈X more effective trades at this SR") on every failed DSR gate (B03.1).
- "Deflating against N cumulative trials (M this study)" at gate time, study-deletion events into
  expansion_history, and holdout fresh_frac logging (B03.2).
- Curse-corrected expected edge (best_fold_sharpe − 0.85·std_sharpe) wherever fold results are
  surfaced (B12.1).
- Effective-bets N_eff ≈ 1.2–1.4 for the crypto book in decision_report (B17).
- The sentiment-gate outcome journal and its n ≥ 60 keep/kill scorecard (B13).
- The cache-aware LLM spend ledger and corrected pricing rows (B07.2).
- Per-leg (lstm/lgb) prediction journaling in the decision journal (B02/B12.2).
- Sizing-journal tier-flip counts before/after the B06 hysteresis change (B06).
- The admissibility-split base-rate/AUC report in meta_meta.json (B04.3).
These ride their parent packets' code changes; B22's own work (report consumers, GUI gate-box
fields, freshness strip) proceeds on the state map's spec.

---

## B23 — Wave-8/9 activations touched by research

**crypto_trend gate (BTC 200h-SMA de-risk; kill-list survivor-#10 sign-off required).** Bare MA
rules on BTC decayed post-2022 (median returns ~0 in whipsaw years), but risk-ADJUSTED evidence
stays positive and the module's asymmetric Schmitt trigger + 0.5 floor + only-shrink design is
precisely the prescribed whipsaw mitigation; the 200h window sits inside Detzel et al.'s
significant 1–4-week band. Wire it exactly as designed and validate with the co-fire
counterfactual BEFORE enabling: log trend_scalar every cycle for 2–4 weeks with the flag still
off, then require (a) unique-fire rate > ~5% of risk-off hours beyond what vol-scaling + VIX +
stablecoin halt already shrink, AND (b) counterfactual P&L on suppressed size ≥ 0 net; otherwise
leave it off — the standalone alpha is ~0 post-2022, only drawdown hygiene. Confidence: high.

**Do NOT build BTC-lagged alt-alpha features (decision support for D31).** Post-2022 evidence:
BTC→alt transmission in MAJORS completes within minutes (inside one hourly bar); the seesaw effect
(largest coins NEGATIVELY predict smaller next period) makes the sign regime-dependent; and the
cross-crypto predictability that survives costs is harvested by broad long-SHORT portfolios, not a
long-only 6-name book. Expected hourly IC on our majors ≈ 0, sign-unstable. If the owner wants
in-house proof first: run the shipped indicator_leadlag.py with BTC_Return_1h as candidate leader;
acceptance bar |IC| ≥ 0.02 stable in sign across 2023/2024/2025 at some lag ≥ 1h (prediction: it
fails). Keep the existing contemporaneous BTC context features unchanged; note D31's separate
finding that the production preset currently drops them — restoring them is the D31 ruling, not a
lag-stack build. Confidence: high.

**bet_sizing meta-tilt re-map (wave-9 dormant code).** Replace the fixed live map clip(2p, 0.6,
1.3) with the AFML bet size already implemented in bet_sizing.afml_bet_size:
tilt(p) = clip(1 + 0.75·afml_bet_size(p, base_rate), 0.6, 1.3), step 0.05, with base_rate set to
the ECONOMIC breakeven a/(a+b) (= 1/3 at tp_rr = 2), not 0.5 — removing the flat top that sizes a
rank-1 trade like a rank-7 one. PRECONDITION: purged-OOF calibration live (Meyer-Barziy-Joubert:
fixed maps only gain from CALIBRATED probabilities) — so this sequences strictly after B04.
Confidence: medium.

**Citations.** Asia-Pacific Financial Markets 2026 (BTC→alt minute-scale transmission); Jia, Wu,
Yan & Yin 2023 (J. Empirical Finance, seesaw); Guo, Sang, Tu & Wang 2024 (JEDC); Detzel et al.
2021; Grayscale momentum-signals; CoinGecko/CoinDesk dominance-regime notes; López de Prado AFML
ch. 10; Meyer, Barziy & Joubert 2023.

---

## B24 — Literature builds (signed-jump / MAX family) — SCOPE SHRUNK

**What the two-round verdict is.** The round-2 dive split B24 and killed most of its cost: the
minute-bar harvest leg and the EQUITY signed-jump feature are permanently SKIPPED, and the
surviving build is a crypto-only feature computed from hourly bars already on disk.

- **Why the equity leg dies** (confidence: medium): signed-jump variation SJ = RV⁺ − RV⁻ isolates
  jumps only with enough intraday observations (Bollerslev-Li-Zhao used 78/day); stocks at hourly
  bars give ~7/day, where the residual is weekly return asymmetry — heavily spanned by short-term
  reversal (corr 0.37), MAX (0.24), MIN (0.32), and realized skew (0.93), i.e. by RR_5/RR_21
  (shipped) plus the MAX survivor. No direct US replication exists on post-2013 data (RSJ needs
  TAQ, so it is absent from the open replication universes); adverse priors: ~40–50%
  post-publication decay base rate, one published out-of-era replication where the sign FLIPPED,
  concentration in illiquid names this book cannot hold, and long-only truncation (the profitable
  leg is the short leg). Reopen trigger: a credible US result on ≥ 2014 data showing VW long-leg
  alpha ≥ 10bps/week net in liquid names.
- **The crypto leg survives cheaply** (confidence: high for the build, unknown sign until
  measured): crypto keeps 24 obs/day (same order as BLZ's 78) and a far larger jump share (~28% of
  BTC daily variance). Feature: daily RV⁺_d = Σ r_h²·1[r_h>0] over the 24 hourly log returns;
  RSJ_d = (RV⁺ − RV⁻)/(RV⁺ + RV⁻) ∈ [−1, 1]; feature = trailing 7-day mean (equivalently
  rolling-168h (Σ r²·sign)/(Σ r²)); require ≥ 120 of 168 hourly returns present else NaN;
  winsorize at ±0.5. Enters the blend sign-agnostically (Lee-Wang 2024 finds high jump variance →
  LOWER next-week crypto returns, but the sample ends ~2021 and BTC jump probability halved
  post-2020).
- **Gate before any harvest-column inclusion** (confidence: high): the feature goes into the
  already-sequenced indicator_leadlag.py IC run on real Jetson panels FIRST, tested jointly with
  its redundancy cluster (crypto RR_5 analog, trailing 7d return, 168h return skewness — corr with
  RSJ ~0.9 — and MAX/MIN_24h once built); keep only if it survives the overlap-adjusted FDR at
  1–48h AND shows post-2022 subsample IC distinguishable from zero.
- **MAX / lottery-demand features from existing hourly data** (confidence: medium): MAX_20d (max
  daily return over trailing 20 trading days, equities) and MAX_7d (crypto); optional MAX5 (mean
  of 5 largest). Feed as CS_Rank features only — NEVER hard-code a sign: equity evidence says
  penalize high MAX (EW −0.96%/mo t = −3.64 but VW only −0.61 t = −1.96, short-leg concentrated),
  crypto evidence is contradictory (MAX positive 2016–2019 weekly; jump-variance negative
  2015–2023). Expectation-setting: assume ≤ half the published equity spread survives
  post-publication and a breadth haircut ~√(N_names/100); if Stage-0 IC is indistinguishable from
  zero at ≥ 6 months of hourly panel, KILL rather than tune.

**Citations.** Bollerslev, Li & Zhao 2020 (JFQA); Lee & Wang 2024 (JFQA); Amaya et al. 2015 (JFE);
Bali, Cakici & Whitelaw 2011 (JFE); Grobys & Junttila 2021 (JIFMIM); Financial Innovation 2021
(crypto MAX); Chen & Zimmermann open-source asset pricing + 2022 publication-bias decay rates;
Rehman et al. 2023 (sign-flip replication); Patton & Sheppard 2015 (REStat); Liu, Patton & Sheppard
2015 (J. Econometrics); Zhang & Zhao 2023 (IRFA); arXiv 2510.14435 (BTC jump shares).

---

## NEW opportunities surfaced by the research (kill-screened)

Each item below was screened against `research/KILL_LIST.md`; none rebuilds a killed item. Screen
notes are inline.

1. **Alpaca crypto venue census script** (measurement-only, ships directly; one day of Jetson
   runtime). GET /v1beta3/crypto/{us,us-1}/trades and /quotes for all traded pairs, trailing 30
   days: per-pair trades-per-minute density, share of minute bars with ≥ 2 and ≥ 1 trades,
   time-weighted quoted spread median/p75/p95, /orderbooks depth at touch and within 10bp/25bp,
   and how far back /quotes actually returns for loc=us. This single script decides the entire
   B05 crypto stamp design (quote-stamp vs minute-EDGE vs per-pair constant, per pair) and
   calibrates the validation threshold. Kill screen: the depth snapshots feed the impact model's
   V_D denominator (cost measurement), NOT an entry filter — the killed order-book-imbalance
   filters are untouched; every endpoint is free, so the paid-data kills are untouched.
2. **Hourly mark-to-market equity emission from the policy replay** (B02 schema requirement).
   Unlocks the honest sized-portfolio drawdown and the whole block-bootstrap inference class for a
   small replay change. Kill screen: clean — validation-inference tooling, distinct from the
   killed sequential-bootstrap-for-LGB-bagging (a training-weighting substitute).
3. **Stationary-bootstrap Sharpe p-value as a shadow diagnostic** (measurement-only): Politis-
   Romano stationary bootstrap, B = 1000 resamples, studentized per Ledoit-Wolf 2008, mean block
   length from the Patton-Politis-White-corrected automatic rule; logged NEXT TO the analytic DSR
   for a few cycles before any further gate change. ~40 lines of numpy; requires item 2. Kill
   screen: clean (same note as item 2).
4. **Durable intent journal (write-ahead order log)**: a tiny append-only jsonl of {client_order_id,
   symbol, side, notional, ts} written BEFORE each submit, replayed at startup against
   get_order_by_client_order_id — closes the crash-window between submit and tracking that the B08
   reconciliation sweep cannot reach (the restart currently reconstructs positions but not
   in-flight intents). Standard write-ahead pattern; touches restart semantics in both loops, so it
   deserves its own design pass. Kill screen: clean — broker-reliability mechanics, no strategy
   content.
5. **Downside realized semibeta for hedge sizing** (Bollerslev-Patton-Quaedvlieg JFE 2022):
   semibetas are the one part of the semivariance literature proven to work from DAILY returns, so
   beta_ledger could decompose the book's SPY beta into down/up semibetas at zero data cost and
   the B20 hedge could size on the DOWNSIDE semibeta instead of the plain Dimson beta if they
   differ materially. Sequenced with/after the first real beta-ledger run. Kill screen: clean —
   hedging machinery under survivor #12, not a new anomaly book.
6. **BTC-dominance / alt-season slow regime input** — an ASK, not a build: the post-2022 evidence
   locates alt-beta variation in the dominance regime, not BTC-lag returns; Δ(BTC dominance, 7d)
   lagged 1 day from free CoinMarketCap/CoinGecko data would be a weekly-granularity conditioning
   column. Kill screen: partially implicated — the on-chain-flow kill's rationale (new
   unreplicated data dependency) applies in spirit, so this requires an explicit owner
   data-dependency ruling before any build; recorded here as an opportunity with its blocking
   question attached.
7. **Free-endpoint LLM qualification pass for the analyst role**: llm_config already supports
   free-only/best-free selection modes and endpoint configs with none enabled; a measured
   qualification (schema-compliance rate, latency vs the 45s budget, ~1 week of llm_eval score
   agreement vs flash-lite in shadow) could take the analyst to $0/day and free the whole $1 cap.
   429 behavior and strict-schema support must be tested before trusting a gate to a free tier.
   Kill screen: clean.
8. **Signed refinement of the crypto extreme-vol de-risk state**: gate the B06 HIGH-RV multiplier
   on downside-semivariance share RS⁻/(RS⁻+RS⁺) > 0.55 over 24–72h, because BTC's leverage effect
   is inverted and an unsigned trigger cuts size in blow-off rallies a long-only book earns.
   Default-OFF, measured via the sizing-decomposition journal; merges naturally with the B24
   crypto RSJ feature (same kernel). Kill screen: clean — a de-risk trigger refinement under
   survivor #8's boundary (managing vol for risk, not betting on vol as alpha).

---

## Kill-list asks (evidence challenging or clarifying an existing entry — asks only, owner decides)

Per KILL_LIST.md's own rule, an item leaves the list only by explicit owner decision; these are
the round's grounds to ASK. Nothing was dropped or rebuilt pending these rulings.

1. **"Honest-OHLC passive-fill simulator + EDGE-on-hourly inflation fix — cited medians were wrong
   ~70-95x" [wave-7].** Two independent agents converged: (a) the ~70–95x error factor is within
   rounding of the exact 100x percent-vs-fraction slip (bidask returns FRACTIONS, liquidity.py
   multiplies by 100), supporting a units slip in the wave-7 FINDING rather than in wave-6
   liquidity.py; (b) the killed item's directional premise is nonetheless REAL and now quantified —
   hourly EDGE on tight-spread assets has a noise floor ≈ σ_bar·n^(−1/4) (the JFE authors' own
   minute-vs-daily benchmark correlation is 56.17% → 88.79% in the small-spread sample), a genuine
   ~5–10x inflation for mega-caps, nowhere near 70–95x. ASK: re-open ONLY the "stamp from minute
   bars instead of hourly" half (which is what B05 implements through the normal gotcha-#2 gate);
   the honest-OHLC passive-fill SIMULATOR half stays dead (fees.py's "do NOT fix by simulating
   passive fills" note is untouched by every recommendation this round).
2. **"SPY<200d short-activation gate — contradicts SYY, short alpha follows froth not bears"
   [wave-5].** The B20 trend-conditional hedge uses the same 200-day trigger for a DIFFERENT
   mechanism: timing beta REMOVAL (grounded in Goulding-Harvey-Mazzoleni state-conditional beta
   economics plus beta_ledger's own measured below-trend beta), not harvesting short ALPHA (the
   SYY-based claim the kill rejected). ASK: explicitly bless this distinction — and ideally record
   it as a new "commonly confused survivors" entry — before any hedge build, so the trigger is
   never mistaken for a rebuild of the wave-5 kill.
3. **Pseudo-CAPE (SPY P/E×1.6) [rev-07-01, econ-07, nobel-07] — an ENFORCEMENT ask, not a
   challenge.** The killed indicator still multiplies ~every stock entry by 0.7x in
   get_macro_regime; its removal is queued under D10 as an owner decision. The B06 research adds
   one clarification: min-aggregation does NOT launder it (a fake signal inside a min() still
   binds whenever it is the minimum). ASK: rule on the queued removal when adjudicating B06, so
   the consolidation doesn't ship around a kill-listed input.
4. **"BTC-lagged spillover for alts" (commonly-confused survivor #10 — UNDECIDED, needs user
   sign-off).** This round supplies the decision support for the D31 ruling: the evidence says the
   alt-ALPHA lag-feature version fails for hourly majors (minute-scale absorption plus
   sign-unstable seesaw dynamics — recommend AGAINST building it), while the BTC-native trend GATE
   (crypto_trend.py, wired per B23 with the co-fire counterfactual) is a separate, defensible
   sizing-hygiene item, and restoring the existing CONTEMPORANEOUS BTC context features (which the
   production preset silently drops — D31's false-comment finding) is a third, distinct object.
   ASK: rule on the three objects separately so the undecided entry stops blocking the two
   defensible ones.

---

*End of Phase 2 synthesis. Build packets should cite this file's per-packet sections in their
specs; the round-1 reports and branch-dive transcripts remain the detailed record of why.*
