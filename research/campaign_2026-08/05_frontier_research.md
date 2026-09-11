# Campaign 2026-08 — Phase 5: Frontier Research Synthesis

**Produced:** 2026-08-20. Synthesizes 6 round-1 frontier reports (12 topics: TSFMs, tabular/sequence
architectures, objectives & labels, ensembling, cross-sectional ML, anomaly decay, regime detection,
nonstationarity, equity market structure, crypto market structure, top-journal economics, LLM signal
extraction) plus 6 deep dives (LambdaRankIC, label-horizon decoupling, PFN/TabPFN serving,
training-window recency, xLSTM, news-archive census). Everything below was screened against
`research/KILL_LIST.md` (in full, including the PENDING OWNER ASKS appendix) and against
`research/campaign_2026-08/02_research.md` — **nothing here re-recommends what the August campaign
already specced (B02–B24, NEW-#1..#8)**; where a candidate touches an August packet it is explicitly
a complement or a rider, and says so.

**How to read this file.** Section 1 is the digest of what changed in the markets this book trades.
Section 2 is the ranked build-candidate list; every candidate carries a **CORE** or **ADJACENT** tag
relative to the **signal model** (the RegressionLSTM + LightGBM blend, its labels, objectives,
training data, and output heads) because a follow-up wave will deep-dive the signal model and
**consumes this file as binding input** — CORE items are that wave's menu; ADJACENT items route to
ops/cost/risk work. Section 3 records the honest negatives (frontier areas checked and rejected),
which are as load-bearing as the builds: they close questions so no future agent re-derives them.
Section 4 is the kill-list asks. Section 5 is sequencing. Effort: S = under a day, M = 1–3 days,
L = multi-day plus shadow weeks. Confidence labels are the research agents' own, preserved.

Repo facts verified while assembling this file (2026-08-20): `stage0_preds.py`,
`portfolio_backtest.py`, `scripts/ic_by_name.py`, `scripts/rank_gradient_report.py`,
`scripts/reliability_report.py`, `scripts/meta_learning_curve.py`, and
`scripts/crypto_spread_census.py` all exist — the Stage-0 evaluator stack the older waves demanded
is built. No fee-multiplier/stress knob exists in `backtest.py` or `fees.py` (FR-16 is genuinely
new). `HOLDOUT_FRACTION = 0.12` is a quantile of the data span (`scripts/hypersearch_v2.py:299,314`),
which is the mechanical comparability defect FR-01 fixes. There is no retrain-gain ledger in-repo
(FR-08 is new).

---

## 1. Market-lessons digest — what changed in the world this book trades (2025–2026)

1. **Stress liquidity now has a measured signature: depth evaporates ~10x faster than volume
   rises.** In the April 2025 tariff crash (S&P −11% over Apr 2–4; SF Fed Economic Letter,
   2025-10), CME measured E-mini volume +99% vs Q1 average while book depth fell 68%, with fill
   quality recovering only by Apr-21 (CME Group, 2025). March 2026 (Iran risk-off, VIX peak 31.65
   on 2026-03-27; Cboe Index Insights, 2026-03) and the June-6-2026 semiconductor "crash-up"
   (record 7.8M SPX options contracts; CNBC, 2026-06-06) repeated the pattern. These are the first
   severe stress episodes inside our own harvested data — they turn the B05.3 stressed-exit
   multiplier from a literature prior into a measurable quantity (FR-17). As of 2026-08-17 the VIX
   sits at a 2026 low of 14.2 with the S&P at record highs (CNBC, 2026-08-17) — complacent-regime
   entry conditions going into the fall.

2. **24-hour US equities is real and accelerating, and it erodes the overnight-sleeve premise.**
   Alpaca has offered 24/5 trading since ~Feb-2026 (Sun 8PM–Fri 8PM ET, LIMIT-only overnight,
   free real-time overnight quote feed; Alpaca docs accessed 2026-08). The SEC approved Nasdaq
   23/5 on 2026-04-10 (Simpson Thacher memo, 2026-04-29); NYSE Arca targets a 2026-12-06 launch;
   NSCC targeted 24x5 clearing for June 2026 (DTCC). Blue Ocean ATS carried ~$375B notional across
   2025. Implications: overnight gap risk is becoming partially hedgeable (limit-only), and the
   Lou-Polk-Skouras close-to-open premium underlying `OVERNIGHT_SLEEVE` may be migrating into
   tradeable sessions — a measurement question our own data answers (FR-18), not a rebuild. Do not
   trade the overnight sessions themselves (thin, limit-only books); do use the free feed as
   telemetry.

3. **US equity microstructure has a dated calendar item and a dominant new flow.** The half-penny
   tick (Rule 612) was delayed by SEC exemptive order on 2025-10-31 to the first business day of
   November 2026 (SEC press 2025-130) — when it lands, quoted spreads on tick-constrained
   mega-caps roughly halve, so the EDGE spread stamps and `IOC_CAP_BPS['mega']` need a scheduled
   re-stamp (FR-20d). 0DTE options are now 50–63% of SPX volume (Cboe data, 2025–2026), with
   regime-dependent dealer-gamma pin/trend effects concentrated in the final two hours (Božović,
   SSRN 5223127, 2025) — free gamma data does not exist, so the honest response is to re-verify
   that our own 14:30–15:30 entry window still carries IC in 2025–2026 data (FR-18), not to build
   a gamma feature.

4. **Crypto is in a confirmed post-ETF bear, and defense outranks any new alpha.** BTC peaked at
   $126,198 in Oct-2025, cascaded on 2025-10-10, flushed again Jan–Feb 2026 (~$9B liquidations,
   OI −21.7%), broke to a 21-month low near $59.3k in June 2026 (CoinDesk, 2026-06-24), and sits
   ~$64.5k (−49% from peak) as of early Aug-2026, on record ETF outflows and long-term-holder
   capitulation. For a long-only hourly momentum book, the de-risk stack and crypto_trend gate
   (B06/B23) are the highest-value machinery we own; no 2025–2026 alpha paper outranks "don't be
   fully long in this tape."

5. **The Oct-10-2025 $19B liquidation cascade left permanent microstructure damage.** ~1.6M
   traders liquidated, alts wicked −20 to −80%, USDe printed $0.65 on Binance as a venue-oracle
   artifact while staying ~110% collateralized (CoinDesk Research, 2025-10; AMBCrypto, 2025-11).
   Market makers were left stuffed with inventory after ADL closed their hedges; alt 1%-depth
   halved ($2.5M → $1.3M) and was still ~$1M below pre-crash into 2026 — the thinnest books since
   2022 (CoinDesk, 2025-11-15 and 2026-01-08; Amberdata). Two consequences: (a) spread/impact
   stamps calibrated on pre-Oct-2025 data understate today's alt costs — the already-built
   `scripts/crypto_spread_census.py` run should include a pre/post-2025-10-10 split (FR-17); (b)
   the cascade is a documented structural break in the training data — the direct motivation for
   the training-window work (FR-01/FR-02/FR-09).

6. **The crypto funding regime flipped.** BTC perp funding has been persistently negative through
   2026 — a 46-day negative 30-day-average streak as of 2026-04-15, the longest since Nov-2022
   (Phemex, 2026-04-15). The `Funding_*` features (kill-list survivor #1) were trained mostly on a
   positive-funding world: that is a train/serve distribution shift to audit before the next
   retrain (FR-03), not a feature to remove.

7. **Price discovery moved to US/ETF hours, and the altseason rotation story died.** Spot Bitcoin
   ETFs lead spot ~85% of the time (Kia, Financial Review 61(2), 2026); US sessions now carry ~47%
   of global spot volume vs 38% pre-ETF, liquidity peaks ~11:00 UTC and is ~42% thinner by 21:00
   UTC, and weekends run on thin crypto-native books (DailyCoin, 2025). BTC dominance fell below
   the 57% "rotation trigger" (56.5%, Aug-2026) with the Altcoin Season Index stuck in the
   high-40s — ETF capital structurally does not rotate into alts. This modulates our costs and
   possibly our IC by session-of-week (FR-18) and weakens pending kill-list ask #6 (Section 4).

8. **The academic frontier's 2025–2026 message is conservative, and it vindicates this system's
   posture.** The "complexity wins" thesis was dismantled: Nagel (BFI WP 2025-104, 2025-08) shows
   over-parameterized RFF return forecasts are volatility-timed momentum in disguise, and Buncic
   (SSRN 5239006, 2025-09) shows a simple shrunk linear model beats the most complex
   Kelly-Malamud-Zhou configuration. Time-series foundation models fail at return forecasting
   (DM-significant beats of random walk in only ~2 of 25 cases — arXiv 2606.27100, 2026-06;
   off-the-shelf TSFMs poor zero-shot AND fine-tuned — arXiv 2511.18578, 2025-11). The net
   anomaly premium sat in unharvestable short legs (Muravyev-Pearson-Pollet, Journal of Finance,
   2025: 162 anomalies, +0.14%/mo pre-borrow, −0.01%/mo net) — the long-only design is where the
   harvestable premium lives. Reported LLM trading alpha is substantially memorization
   (Look-Ahead-Bench, 2026: +20.7% in-window → −1.0% post-cutoff; Lopez-Lira-Tang-Zhu, 2025-04).
   A modest supervised blend, long-only, measured in-house, is the defensible frontier position —
   the frontier's offer this round is mostly calibration data and cheap falsification tools, not a
   new model.

---

## 2. Ranked build candidates

Every candidate names its exact in-repo measurement; per the wave-5 kill-list rule, nothing ships
on literature priors alone, and every model-facing item rides the default-OFF-flag →
challenger → shadow path with cumulative-trial accounting (B03.2). "Chain-2" refers to the single
already-planned gotcha-#2 re-harvest/study-reset event shared by B05/B12/B17 — candidates that
need a study reset ride it rather than triggering their own.

---

### FR-01 — Pin the holdout to a fixed calendar span (prerequisite gate repair) — CORE, S, confidence: high

**Technique.** Replace the proportional 12%-of-span holdout (`HOLDOUT_FRACTION = 0.12`, a quantile
of the data span at `hypersearch_v2.py:299,314`) with a fixed trailing window (suggested
`TRADER_FIXED_HOLDOUT_DAYS = 60` for crypto), keeping the existing purge of label windows crossing
the boundary. Flag OFF = byte-identical legacy quantile path.

**Why now.** Any training-window or decay experiment (FR-02, FR-09) is uninterpretable under the
proportional rule: a 1Y arm gets a ~44-day holdout while a full-history arm gets ~200 days, so the
PROMOTION_GATE_V2 fail-closed n_eff ≥ 10 floor systematically flunks short windows on gate
*mechanics*, not skill, and cross-arm DSRs are not comparable. This is also a standing weekly
benefit: successive retrains get certified on a consistent-width holdout.

**Expected gain.** Comparability, not P&L: it removes a mechanical bias that would otherwise force
every recency experiment toward long windows.

**Measurement plan.** Instrumentation first (ships directly): print both boundaries plus holdout
row/trade counts side-by-side for one Jetson run per book, confirming the 60d span yields ≥ 10
calendar-uniqueness effective trades for the current champion (the `calendar_effective_n` output
already prints at `hypersearch_v2.py:~1501`). Then validate the flag by re-running
`evaluate_on_holdout` on the same saved winner under both boundaries and reporting the DSR delta;
`bash scripts/ab_check.sh` for suite regressions. Compatible with B03.2's overlap-weighted pool —
this changes the holdout's *span*, not the reuse accounting.

**Jetson honesty.** Trivial compute. Known cost: a fixed 60d holdout on the full-history arm
discards ~140 days the quantile rule would have scored, widening long-arm DSR CIs — acceptable,
because cross-arm comparability is the point of a promotion gate.

**Citations.** Bailey & López de Prado 2014 (SSRN 2460551, DSR); López de Prado, Lipton &
Zoonekynd 2025-09, "How to Use the Sharpe Ratio" (SSRN 5520741) — small SR deltas across arms are
noise without a comparable inference frame.

---

### FR-02 — Fixed-config training-window A/B for the crypto book (full vs 2Y vs 1Y vs 18-month "PT-2007" arm) — CORE, M, confidence: high

**Technique.** Retrain the CURRENT champion config — no Optuna search, therefore zero new
selection pressure — on three or four data spans: full history (~2021+), trailing 2Y (17,520h),
trailing 1Y (8,760h), and optionally an 18-month arm (8 months pre-break + everything after
2025-10-10, per Pesaran-Timmermann 2007's result that the optimal post-break window *keeps* some
pre-break data). Identical purged folds where spans overlap, identical FIXED holdout (FR-01
prerequisite), identical seed and epoch budget.

**Why.** The 2025-12 frontier moved past "more data always": Capponi-Huang-Sidaoui-Wang-Zou prove
window length and complexity must be jointly chosen — their error bound adds a total-variation
drift cost that *rises* with window length, and short-window simple models win precisely in
high-drift regimes (+14% OOS R², gains concentrated in stress periods). The Oct-2025 cascade is a
documented structural break (lesson 5), so the drift term is live for crypto. Kelly's own school
conceded the boundary the same month (Chen-Kelly-Malamud, "Limits To (Machine) Learning").

**Expected gain.** Either a better-adapted crypto model (if a shorter window wins on both the fold
objective and holdout DSR at equal n_trials) or a durable in-house negative that closes the
recency question and blocks future re-litigation. Decision rule: adopt shorter only on BOTH wins;
else keep full history and ship nothing.

**Measurement plan.** Jetson: one fixed-config run per arm (bypass the TPE loop; call the trial
trainer + final refit + `evaluate_on_holdout` directly). Metrics, all in-repo: (a) holdout DSR at
IDENTICAL n_trials per arm (same cum_trials passed, so deflation is equal and the comparison is
pure SR ordering under one null); (b) the per-fold purged val Sharpe vector; (c) purged IC per
name via the `stage0_preds.py` dump + `scripts/ic_by_name.py` on the holdout slice; (d) winner arm
through `backtest.py --prefix '' --days 60 --gate` before any swap.

**Jetson honesty.** 3–4 fixed-config trains ≈ tens of minutes each on the Orin GPU — one evening
total. Shorter windows *reduce* peak RAM. Honest caveat: the 1Y arm trains the LSTM on ~8.8k
bars/name — near the floor for sequence models; expect the blend to lean on the LGB leg there,
which is informative, not disqualifying.

**Citations.** Capponi, Huang, Sidaoui, Wang & Zou 2025-12 (rev. 2026-08), "The
Nonstationarity-Complexity Tradeoff in Return Prediction" (arXiv 2512.23596); Chen, Kelly &
Malamud 2025-12, "Limits To (Machine) Learning" (arXiv 2512.12735); Pesaran & Timmermann 2007
(J. Econometrics) — prior art, optimal window keeps pre-break data; CoinDesk 2025-11-15 /
2026-01-08 + Amberdata (persistent post-cascade depth damage = the drift evidence).

---

### FR-03 — Funding-feature regime-shift audit before the next retrain — CORE, S, confidence: high

**Technique.** The `Funding_Rate_Ann/Z/Chg_24h` and `CS_Rank_Funding_Z` features SURVIVE
(kill-list survivor #1) but were fit mostly on a positive-funding world, and 2026's funding is
persistently negative (lesson 6). Compute PSI and KS statistics of each funding feature's
training-window distribution vs its trailing-90d live distribution, plus per-feature purged IC
full-sample vs 2026-01-onward subsample. Flag = PSI > 0.25 or an IC sign flip with non-overlapping
CIs.

**Expected gain.** Prevents the next retrain from silently inheriting a mis-scaled live feature
family; produces an audited go/no-go note, not a code change. If flagged, the scheduled retrain
handles it naturally (gotcha #2 applies only if a feature transform changes).

**Measurement plan.** Jetson: recompute the features from the harvest parquet (pure pandas); drift
stats via `monitor_drift.py` conventions; per-feature purged IC via the `stage0_preds.py` dump +
`scripts/ic_by_name.py` with a timestamp split at 2026-01-01. Deliverable = the flag table
attached to the next retrain notes. Complementary to B19's forward-looking per-feature PSI monitor
(that one watches live drift going forward; this one audits the historical training envelope once).

**Jetson honesty.** Cheap. Small-data honesty: the negative-funding regime spans only ~7 months ×
6–10 names of hourly bars (~30k rows) — enough to detect a sign flip, not to re-estimate
magnitudes. The deliverable is a flag, not a refit.

**Citations.** Phemex 2026-04-15 (46-day negative 30d-average funding streak, longest since
Nov-2022); kill-list survivor #1 boundary (features survive; the killed carry-harvest /
carry-rank / basis-tilt trio stays dead).

---

### FR-04 — Nagel naive-baseline benchmark: is the blend's complexity a window artifact? — CORE, S, confidence: high

**Technique.** A pure-numpy script computing a recency-weighted, vol-scaled momentum one-liner per
name-hour — EWMA(returns, half-life ≈ the tuned 12–48h horizon) / trailing realized vol, zero
fitting, zero dependencies beyond numpy/pandas — scored head-to-head against the deployed blend on
identical windows and admission gates. Nagel (2025) proved over-parameterized return forecasts
collapse to exactly this predictor; if the blend cannot beat it, the blend's claimed edge is
suspect.

**Expected gain.** Falsification insurance on the whole modeling stack for ~a day of work. The
July compilation prescribed this as cheap-diagnostic #6; grep confirms nobody built it.

**Measurement plan.** Consume `{prefix}stage0_preds.json` through `scripts/ic_by_name.py`'s
loader; emit side-by-side purged IC (blend vs naive) per name, and a DSR on each via
`validation.deflated_sharpe_ratio` with the campaign's cumulative trial count for the blend and
n_trials = 1 for the naive rule (it was never searched). Success criterion for the blend: IC and
DSR strictly above the one-liner on the same holdout windows. If the advantage is within noise
(stationary-bootstrap CI from the B02 equity series), report to owner — a promotion-culture
finding, never an auto-action.

**Jetson honesty.** Kernel + synthetic test are Mac-buildable today; the real run needs the
Jetson-produced stage0 dump. Zero new dependencies, zero model-facing changes.

**Citations.** Nagel 2025-08, "Seemingly Virtuous Complexity in Return Prediction" (NBER WP 34104
/ BFI WP 2025-104); Buncic 2025-09 (SSRN 5239006) — simple shrunk linear SR 0.699 > KMZ complex
0.485.

---

### FR-05 — Champion cross-sectional rank-IC baseline in the holdout report — CORE, S, confidence: high

**Technique.** Add a per-fold cross-sectional Spearman line for the CURRENT champion LGB leg, the
LSTM leg, and the blend to the holdout certificate report (~30 lines in the holdout scoring path,
report-additive only, flag-OFF report keys preserved per the D25 convention).

**Why.** It is the cheapest possible go/no-go evidence for FR-06: LambdaRankIC's strongest
transferable finding is that GBDT *regression* on noisy returns can underperform even OLS on rank
IC (their XGB-regression baseline: 0.0418 vs OLS 0.0471). If our champion LGB leg's cross-sectional
IC is already strongly positive, the expected FR-06 delta shrinks and the owner can deprioritize
before spending a study-reset slot; if it is near zero or negative, that is the strongest in-house
justification to proceed.

**Measurement plan.** Its own output IS the measurement — reuse the per-timestamp grouping +
`scipy.stats.spearmanr` kernel from `ic_by_name`; print champion-LGB, LSTM, and blend rank IC per
fold at the next Jetson report pass.

**Jetson honesty.** Seconds of compute inside an existing report pass.

**Citations.** Lin, Su & Yang 2026-05, "LambdaRankIC: Directly Optimizing Rank IC for Financial
Prediction" (arXiv 2605.00501, v1 2026-05-01) — Table: regression-GBDT below OLS on rank IC.

---

### FR-06 — LambdaRankIC rank-objective challenger for the STOCK LGB leg, with a mandatory score→return calibration layer — CORE, L, confidence: medium

**Technique.** Port the LambdaRankIC objective to LightGBM as a pure-numpy custom objective
(`rankic_objective.py`, Mac-testable): per date-group of size n, for each pair with ỹ_i > ỹ_j,
Δ = 12·|r̂_j − r̂_i|·|ỹ_i − ỹ_j| / (n(n²−1)); lambda gradients with sigmoid p and hessian
2·p·(1−p)·|Δ|; all pairs per group (n = 55 ⇒ 1,485 pairs — no sampling needed); avg-uniqueness
weights applied manually inside the objective (LightGBM does NOT auto-apply Dataset weights to
custom fobj gradients); early stopping on a per-group Spearman feval. **No public code exists**
(verified 2026-08-20: arXiv abs page lists no repo; reference implementation is a custom XGBoost
objective) — the port is ~150 lines from confirmed formulas. LightGBM's built-in `lambdarank` is
the wrong target (integer relevance labels, NDCG top-heaviness; the paper shows NDCG-LTR at rank
IC 0.0863 vs the IC objective's 0.1148). Flag `LGB_RANKIC_OBJECTIVE`, stock prefix only — crypto's
6–10-name cross-sections are too thin for a stable rank objective and are deliberately excluded
(record this scoping in the wave file). Because ranker scores are unitless and every downstream
gate consumes percent-return magnitudes, the build includes the calibration layer: raw score →
within-group cross-sectional percentile → OOF purged isotonic fit to realized forward returns
(reuse `calibration.py`'s pure-numpy PAVA + `purged_kfold_indices`; NEVER fit on the
early-stopping slice — that is the wave-9 meta-calibration leak), persisted as a sidecar json,
fail-closed (no calibrator ⇒ no rankic artifact ⇒ champion stays live). This honors the wave-3
rule "never feed ordinal scores to %-return gates" and discharges the wave-3 open LambdaMART item
with a 2026-vintage objective.

**Expected gain.** A rank-IC-shaped improvement on the stock book or a publishable negative that
closes the wave-3 open item either way. Honest transfer prior: the paper's +175% rank-IC gain is
on monthly cross-sections of hundreds-to-thousands of US names over 61 years; at n = 55 hourly the
red-team "possibly-zero delta" prior stands.

**Measurement plan.** (1) Mac: synthetic recovery test (gradient direction raises per-group
Spearman, ρ > 0.95 noiseless), finite-difference gradient check to 1e-5, grad-sums-to-zero
property test, all under `ab_check.sh`. (2) Jetson A/B on IDENTICAL folds/rows/features/seed:
per-fold purged cross-sectional Spearman of leg and refit blend, requiring ≥ 4/5 folds
IC_challenger ≥ IC_champion, and no name flipping from significantly-positive to
significantly-negative IC (`scripts/ic_by_name.py --time-key ts`). (3) Calibration quality:
monotone realized net return across buckets via `scripts/rank_gradient_report.py --strict`.
(4) Holdout blend DSR ≥ champion (DSR_MIN = 0.60 binding), `backtest.py --prefix stock --days N
--gate`, then challenger → shadow → the B03.3 promote statistic. Any single failure ⇒ archive with
numbers.

**Jetson honesty.** Training-time only: ~3.2M pair-ops/round vectorized ≈ 50–200 ms/round → +1–2
min per LGB train at the 120k-row cap; pair-index arrays ≈ 26 MB. Inference is a plain booster —
unchanged sub-ms cost, identical artifact size. Rides the Chain-2 study reset (objective change =
gotcha #2); do NOT trigger a separate reset for this alone.

**Citations.** Lin, Su & Yang 2026-05 (arXiv 2605.00501; formulas confirmed from the HTML full
text — Eq. 20 surrogate, Prop. 1 ΔRankIC, Alg. 1); Burges 2010, "From RankNet to LambdaRank to
LambdaMART" (MSR TR, prior art); LightGBM 4.7 docs (custom-objective signature; lambdarank label
constraints); Molinaro 2025 (SSRN 5402583) — independent 2025 evidence rank objectives compete
cross-sectionally.

---

### FR-07 — Label-horizon decoupling: measure Ĵ(δ) first, exploit as a horizon-diverse LGB leg second — CORE, M for the probe (L for the full path), confidence: high (probe) / medium (exploit)

**Technique.** The Label Horizon Paradox (Song-Liu-Chen, arXiv 2602.03395, v1 2026-02-03, v5
2026-08-19) claims the optimal *supervision* horizon δ* often differs from the *trading* horizon Δ
(signal-realization vs noise-accumulation tradeoff; for interday targets δ* ≪ Δ). Our Optuna
search is structurally coupled: one `forward_bars` categorical sets label, objective, live
horizon, and gate scale simultaneously (`hypersearch_v2.py:95,627`) — the hypothesis is genuinely
untested in-repo, though narrower than the paper's (the TB labels already have endogenous
effective horizons, and stock TB labels are horizon-degenerate above one session). Staged plan:
**(A)** mechanical horizon-transfer curves ρ(r^δ, r^Δ) from the existing multi-horizon
`Target_Return_{12,18,24,32,48}` columns vs the IID null √(δ/Δ), overlap-adjusted (S effort,
ships directly). **(B)** THE DECISIVE PROBE: train the existing `model_lgb` leg once per δ on
`Target_Return_δ` with identical features/folds/purge, score every probe against the champion's
policy-horizon r^Δ — this measures J(δ) directly for ~8 cheap LGB trainings (~minutes each),
honestly incremented into cum_trials. Kill the topic if the curve is flat or peaks at δ = Δ.
**(C)** add LABEL-ONLY columns `Target_Return_{4,8}` (+ TB variants for crypto) at the Chain-2
re-harvest, excluded from the Optuna choice set via an explicit `LABEL_ONLY_BARS` set, so the grid
extends below the current 12h floor where the paper predicts the peak. **(D)** exploit, only on a
significant off-diagonal peak: train the production LGB leg at δ* while the LSTM stays at Δ,
re-project to Δ-scale with a purged OOF OLS (a, b — two floats), and blend through the settled
B12.2 NNLS + shrink-to-0.5 machinery. Diversity-by-construction: same-horizon legs share label
noise; a δ*-leg decorrelates errors mechanically. The deployed prediction target stays Δ, so cost
gates, exits, and shadow comparability are untouched. **(E)** the paper's bi-level λ-weighted
trainer inside `model_v2` is deliberately LAST and flag-gated (confidence: low) — implement from
equations only if A2/A4 both prove out; its real value is that label choice becomes ~7 learned
parameters in one run (B03.2-friendly), but it has a documented collapse-to-shortest-horizon
failure mode without warm-up, and the paper is a 5x-revised, uncited, code-less preprint — treat
every parameter as hypothesis.

**Expected gain.** If Ĵ(δ) peaks off-diagonal: a cheap, genuinely diverse second LGB leg with
blend-DSR improvement. If flat: a durable negative for ~8 LGB trainings.

**Measurement plan.** Probe predictions in the `stage0_preds.py` row schema → purged per-name IC
via `scripts/ic_by_name.py` + `scripts/rank_gradient_report.py`; continue-decision =
max_{δ<Δ} Ĵ(δ) − Ĵ(Δ) > 2·SE (weekly block bootstrap). Exploit gates: holdout DSR on the BLENDED
predictor via `evaluate_on_holdout` (A/B same-horizon vs horizon-diverse on the identical holdout),
`backtest.py --gate`, then shadow with the B03.3 statistic.

**Jetson honesty.** Probe: minutes per LGB training, no torch. Exploit: one extra LGB training per
retrain, zero live-loop memory change. Bi-level (if ever): ~2x forward/backward per step — apply
only to the final refit, never all trials.

**Citations.** Song, Liu & Chen 2026-02 (v5 2026-08-19), arXiv 2602.03395 — flagged: unreviewed,
no venue, no code, zero independent citations as of 2026-08-20; López de Prado 2018 AFML ch. 3
(prior art: label ≠ raw fixed-horizon return; the repo's TB labels are an existing
endogenous-horizon answer); Claeskens, Magnus, Vasnev & Wang 2016 (IJF, combination-puzzle);
Elliott & Liao 2025 + Lee & Lee 2025 (UCR WP 202514) — equal/shrunk weights and subset averaging
at small samples.

---

### FR-08 — Retrain-gain ledger: measure whether weekly retraining actually pays — CORE, S, confidence: high

**Technique.** At each weekly retrain, score BOTH the incumbent (from the `.prev` artifacts) and
the fresh artifact on the same most-recent ~7 days of purged bars (drop the last `forward_bars`
hours), and persist {date, mse_inc, mse_new, ic_inc, ic_new, n_bars} into the adaptive-state
ledger. Verified: no such ledger exists in-repo.

**Expected gain.** After ≥ 12 weeks, the in-house evidence that decides the cadence question: if
the fresh model's median improvement is indistinguishable from zero, weekly is too frequent (drift
triggers alone suffice, saving Jetson thermal budget); if strongly positive, weekly is justified
and biweekly experiments are permanently off the table. Complements — does not duplicate — the
shadow DM (which compares champion vs promoted challenger, not incumbent vs every weekly refit).

**Measurement plan.** The ledger IS the harness: 12+ weekly rows of paired deltas; significance
via the same Ibragimov-Müller block-t machinery B03.3 standardizes. Cadence relaxation, if the
data supports it, is an owner decision.

**Jetson honesty.** Two extra inference passes over ~168 hourly bars per book per retrain —
seconds, inside the existing retrain job. Zero live-path change.

**Citations.** Regol, Schwinn, Sprague, Coates & Markovich 2025-05, "When to retrain a machine
learning model" (arXiv 2505.14903) — cost-aware retraining beats schedules only when retraining is
expensive (ours is near-free, hence measure rather than adopt); Lyu et al. 2023-11 (arXiv
2311.03213, prior art) — periodic ≈ drift-informed on accuracy.

---### FR-09 — Time-decay training weights, AFML decay-on-cumulative-uniqueness form, as one Optuna categorical (crypto first; sequenced AFTER FR-02) — CORE, M, confidence: medium

**Technique.** w_i = u_i · d(U_i) where U_i is the cumulative sum of avg-uniqueness over panel
time and d runs linearly from c to 1 (newest); categorical c ∈ {1.0 (=off), 0.75, 0.5, 0.25} on
the LGB legs only (where uniqueness weights are already plumbed); explicitly NOT naive
calendar-exponential decay multiplied onto u-bar (that double-shrinks old overlapped rows — AFML
snippet 4.11 semantics compose overlap and staleness exactly once). Kish equivalence for
reporting: exponential half-life h ⇒ ESS ≈ 2.9h; report per-fold ESS = (Σw)²/Σw². Rides the
Chain-2 study reset (objective-family change); trials increment cum_trials. Ships together with a
documentation invariant (S, ships directly): comment next to `validation.py`'s anti-stacking note
and `sample_weights.py`'s gotcha-#4 note that **training sample weights (uniqueness and/or decay)
NEVER enter the DSR n_eff** — the holdout n_eff is computed solely from realized holdout trade
calendar concurrency; Kish ESS of training weights is not a DSR input; decay is never applied to
holdout scoring — plus one negative test asserting `evaluate_on_holdout`'s n_eff is invariant to a
synthetic training-weight vector. (Adjudicated against the code: there is NO present double-count;
the invariant prohibits the two future entry points.)

**Expected gain.** The soft-window middle ground between FR-02's hard truncation arms and the
status quo. Honest prior: Stock-Watson 2004 found discounted schemes "typically no better,
sometimes worse," so the expected winner is c = 1.0 — and that null closes the recency question
with in-house evidence. Skip entirely if FR-02 shows full ≈ 1Y.

**Measurement plan.** Optuna search with the categorical live; winner gated by `evaluate_on_holdout`
holdout DSR ≥ DSR_MIN on the FR-01 fixed holdout with cum_trials deflation, then `backtest.py
--gate`, then challenger → shadow. Per-arm purged IC by name distinguishes "decay helps the model"
from "decay helps one lucky name." Stock book only if crypto shows a win (halflife-90-equivalent
decay cuts the stock book's effective n ~40%, worsening B04.3 starvation — the DSR gate punishes
that automatically).

**Jetson honesty.** Zero serving cost (train-time weights); O(n) float64 at train time. Real cost
is search-space growth on one axis — hence categorical-not-continuous and the shared reset event.

**Citations.** López de Prado 2018 AFML ch. 4 (prior art, snippet 4.11); Kish 1965 (ESS, prior
art); Stock & Watson 2004 (J. Forecasting, prior art — least adaptive wins); arXiv 2602.03903 v2,
2026-02 ("Taming Tail Risk": time-decay a robust default in *calibration* — weak transfer to
training weights, cited as the supportive-but-thin frontier edge).

---

### FR-10 — Alpaca News archive census: the go/no-go gate for the news-embedding feature class — CORE (gates a future signal-model feature family), M, confidence: high — BLOCKED on kill-list ask #1 (Section 4)

**Technique.** `scripts/news_census.py` (measurement-only): page the full Alpaca News API archive
(v1beta1, Benzinga-sourced, history to Jan 2015, per-symbol filtering for stocks AND crypto
tickers, `created_at` at RFC-3339 second resolution, 50 articles/page, free on the existing
ALPACA key at 100 req/min) for all 45 stock + 10 crypto names; persist per-article {symbol, id,
created_at, updated_at, headline_len, source}; emit per name: headlines/week, earliest date,
median inter-headline gap, the `updated_at ≠ created_at` revision rate (PIT-text hazard), and
**bar coverage** = fraction of that name's hourly bars in the training parquet with ≥ 1 headline
in the trailing 24h. The round-1 kill criterion becomes evaluable: per-name coverage < 30% ⇒ that
name is OUT of any future news-embedding column; a mostly-failing book ⇒ record the negative and
STOP (no harvest change, no embedding build). Companion S-items: (a) persist full intraday
timestamps in the existing Finnhub cache (`sentiment_history.py` currently truncates unix
`datetime` to date — additive `ts_utc` column, so the running cache accumulates A/B-grade data
from today); (b) score crypto per-name coverage as a byproduct (the crypto book's only historical
sentiment today is market-wide Fear & Greed). Source ruling to record: Alpaca News = designated
archive; Finnhub = live gate + daily sentiment only (free tier is a 1-year *rolling*
NA-equities-only window, no crypto history); GDELT = recorded negative (Section 3, N9).

**Expected gain.** Converts the round-1 "coin flip" verdict on news-embedding features
(Chen-Kelly-Xiu: LLM text embeddings beat sentiment scores — but on 3000+-name daily
cross-sections) into a definite, dated, per-name evidence base before any gotcha-#2 harvest cost
is paid. Expect a per-name coverage MASK: mega-caps and crypto majors pass; small-caps (POET, RDW,
SERV, PRME, QBTS, ASTS) plausibly fail — with the honest tension that CKX's strongest multi-day
drift lives in exactly the small names where coverage is thinnest.

**Measurement plan.** The census IS the measurement; output lands as
`research/news_census_2026-08.json`, a committed dated artifact. Only if a book passes does the
downstream A/B get sequenced (its own gates: purged IC via `indicator_leadlag.py` on real panels
first, then holdout DSR via the blend gate after a Chain-2-style harvest+retrain). Expectation-
setter for the eventual feature: best one-day rank IC 0.0143 for FinBERT on Benzinga headlines
(arXiv 2608.04200, 2026-08) — tiny per-headline effect, further shrunk by ~45-name breadth.

**Jetson honesty.** requests + pandas only, no torch; well under 2h for the whole universe;
pagination and the bar-join kernel are Mac-unit-testable on synthetic fixtures. Embedder choice
(ChronoBERT-class PIT-clean vs API batch) is a later, separate decision with its own B13-style
memory feasibility pass.

**Citations.** Alpaca Historical News docs + API reference, accessed 2026-08-20 (2015 depth,
created_at/updated_at, 100 req/min); Alpaca forum thread 13576, 2024-01 (staff-verified JPM depth
20+ articles/month through 2015–2016); Chen, Kelly & Xiu, SSRN 4416687, latest rev. 2026-02;
arXiv 2608.04200, 2026-08 (QLoRA/Benzinga rank-IC expectation-setter); He, Lv, Manela & Wu
2025-02, ChronoBERT (arXiv 2502.21206); Finnhub docs accessed 2026-08-20 (1-year rolling free
tier); GDELT DOC 2.0 contract + live probe 2026-08-20 (3-month artlist cap; 1-req/5s throttle).

---

### FR-11 — TabPFN v2 teacher → LightGBM student distillation for the meta-label gate (license-clean path) — ADJACENT (meta gate, not the primary blend), M, confidence: medium

**Technique.** The starved meta classifier (B04.3: honest floor n ≈ 1,100–2,300 for the current
booster; actual rows 200–2,500) is squarely in the regime where prior-fitted networks excel. The
route the frontier actually permits: **TabPFN v2** (Hollmann et al., Nature, 2025-01-08; Prior
Labs License = Apache-2.0 + attribution, production-clean) used ONLY at train time as a teacher,
distilled into the EXISTING LightGBM artifact via stratified out-of-fold soft labels — the exact
recipe validated at scale by "Pocket Foundation Models" (arXiv 2605.18654, 2026-05: GBDT student
retains ~96.5% of teacher AUC at 1.9 ms CPU; teacher edge concentrates on < 21-feature datasets —
our slot is 13 features; OOF teacher labeling is MANDATORY, converging with B04.1's own OOF
design). **The TabPFN-2.5/2.6/3 route is license-dead**: their weights bar "the model, its
derivatives, and its outputs" from production/internal-commercial-decision-making, and the
built-in distillation engine is proprietary (Section 3, N8). Serving changes NOTHING — the shipped
artifact stays a LightGBM booster in the current `meta_label` format (kills the latency/RAM
objection). Order of operations: (0) 30-minute license + dependency preflight on the Jetson (pin
the tabpfn pip version whose default checkpoint is v2, record checkpoint hash + attribution,
verify torch compatibility) — items below do not start until this note exists; (1) direct-arm
benchmark: LGB vs TabPFN-v2-direct vs distilled student on the identical post-OOF meta frame,
under TEMPORAL-block purged CV with embargo (never random folds — Drift-Resilient TabPFN, NeurIPS
2024, shows vanilla TabPFN loses more than trees under temporal shift), including a shift-stress
split (oldest 70% context → newest 30% test) and per-tier n ∈ {200, 400, 800, 1600, all} via the
existing `scripts/meta_learning_curve.py` protocol; (2) distillation behind default-OFF
`META_PFN_DISTILL`, sequenced strictly AFTER B04.1 OOF + B04.2 calibration land (a PFN teacher fed
leaked 'pred' features inherits the poisoning). Optional narrow fallback if the full A/B is flat:
TabPFN posterior-predictive probabilities as a *calibrator* for the sub-400-row tier only (a Jul
2026 eval found ECE 2.1–5.3x lower than baselines — arXiv 2607.11007). Bundled probe (CORE,
confidence: low, expected FAIL, run only if idle Jetson hours exist): a FinPFN-style
cross-sectional in-context IC probe on the stock panel (off-the-shelf TabPFN v2 regressor, context
= last K ∈ {24,72,168} hourly cross-sections, strictly trailing) scored via overlap-adjusted
purged IC and orthogonality to the blend — the paper's gains (Wang & Lera, J. Financial Markets,
2025) are at monthly/daily with ~500-name cross-sections; the probe exists to buy a durable
negative rather than leave the branch open. No serving plan under any circumstances until it
passes.

**Expected gain.** If the PFN advantage exists at our n under temporal splits: better veto
precision and calibration at zero serving cost. If not: the idea dies cleanly at zero serving
cost, recorded.

**Measurement plan.** Extend the B04.3 harness: 3 arms × temporal-block purged 5-fold CV × 20
seeds; metrics = fold-OOF AUC + Brier + ECE via `calibration.reliability_curve` /
`scripts/reliability_report.py`, and B04.1's decision metric (holdout veto precision at p < 0.30).
Proceed to distillation only if the PFN arm beats LGB by ≥ 0.01 AUC at the books' actual n under
the TEMPORAL split; flip criterion for the student: ≥ LGB on AUC AND ≤ on ECE on BOTH books,
cross-seed veto flip-rate < 10%; then challenger → shadow.

**Jetson honesty.** Teacher runs only in the retrain window: TabPFN v2 ≈ 17M params (~70 MB),
attention over a ≤ 2500×13 context — seconds on the Orin GPU. Real risk is the dependency pin
(tabpfn requires torch ≥ 2.1 on py3.10 — verify against the Jetson CUDA torch build FIRST). Served
artifact byte-format-identical to today's booster: zero live RAM/latency delta.

**Citations.** Hollmann, Müller et al. 2025-01 (Nature, TabPFN v2); Prior Labs tabpfn_2_5 HF card,
2025-11 (license text barring production outputs) + TabPFN-3 technical report, 2026-05-12 (same
bar); Tanna et al. 2026-05, "Pocket Foundation Models" (arXiv 2605.18654); Helli et al., NeurIPS
2024, "Drift-Resilient TabPFN" (arXiv 2411.10634 — temporal-shift boundary); arXiv 2607.11007,
2026-07 (calibration eval); Wang & Lera 2025 (SSRN 5022829 / JFM; FinPFN, code BSD-3); Ye et al.
2025-05, "Realistic Evaluation of TabPFN v2 in Open Environments" (arXiv 2505.16226 — trees remain
optimal under shift; the reason the temporal split is mandatory).

---

### FR-12 — Multi-quantile head extension of the live q10 tail veto (q05/q10/q25/q50, journaled before any gate change) — CORE, M, confidence: medium

**Technique.** Three extra LightGBM boosters (objective='quantile', alpha ∈ {0.05, 0.25, 0.50}) on
the same shipping scaler/features as the existing `lgb_q10.txt`; non-crossing via post-hoc
monotone rearrangement (sort the four quantile predictions — the cheap, theory-backed alternative
to a non-crossing network, cf. arXiv 2504.08215, 2025-04). Journal the q-spread (q50 − q10) per
decision row; NO sizing/veto change initially. Two hard boundaries: (a) **prerequisite** — the q10
veto itself is flagged uncertified by the holdout DSR gate (B12.2's open follow-up); certify the
incumbent before extending it; (b) **kill-list boundary** — "Conformal abstention (CQR/ACI)" is
KILLED [wave-4]; this head must NOT be wrapped in conformal calibration; any future proposal to
conformalize these quantiles is a rebuild of the killed item and needs an owner ask first.

**Expected gain.** A measurable tail-calibration story for the veto (and later, possibly, a
width-conditioned veto) grounded in the 2025 distributional-forecasting evidence
(Barunik-Hronec-Tobek v2, 2025-08) — as an extension of a live survivor, not a new architecture.

**Measurement plan.** Tail calibration via `scripts/reliability_report.py`: empirical coverage of
q10 (target 10% ± 3pp) and q05 on holdout decision rows. Only if calibrated: A/B the veto
(q10-only vs q10 + width condition) through `backtest.py --gate` policy replay and the holdout DSR
gate. Stocks: pool across names as the current q10 already does (per-name q05 is fragile at ~2k
RTH rows/name-year).

**Jetson honesty.** Negligible: three small boosters, ~ms inference, a few MB on disk.

**Citations.** Barunik, Hronec & Tobek 2024-08 (v2 2025-08), arXiv 2408.07497; "Deep
Distributional Learning with Non-crossing Quantile Network," arXiv 2504.08215, 2025-04 (the
rearrangement justification); arXiv 2508.18921, 2025-08 (distributional return forecasting with
DNNs).

---

### FR-13 — Seed-ensemble vs SWA-soup A/B at the B12 refit event, plus the third-leg diversity rule — CORE, M, confidence: medium

**Technique.** Train K = 3 LSTMs differing only in seed (identical config/scaler/epochs_refit per
B12.1's median-of-fold-best), uniform-average predictions (never weight-fit across seeds), and
compare against the single SWA-tail-soup model B12.1 ships. The 2025 deep-ensemble evidence (DSL,
arXiv 2503.13544, rev. 2025-10, CIKM FinAI wksp) shows ensemble size buys allocation *stability*
(variance reduction) — but SWA tail-soup already harvests most of that at zero extra training
cost, so the readout is churn, not just DSR. Bundled binding rule for the signal-model wave (S,
zero code): **no third blend leg until the B02 per-leg journals measure realized error correlation
ρ(lstm_err, lgb_err)** — combination gain scales with (1 − ρ); if ρ ≥ ~0.85 no similar-type third
leg can pay, and only a structurally different leg (FR-07's horizon-diverse leg, or a ridge/linear
leg) qualifies as a candidate. Until those journals exist this topic has no in-house measurement —
which is itself the finding: build the journals, not the leg. If the pool ever reaches 3+
candidate legs, the 2025 subset-averaging results (Elliott & Liao; Lee & Lee UCR WP 202514) say
trim-then-equal-weight, never fit weights — which also decides whether the same-horizon LGB leg
retires if a δ*-leg enters (fewer artifacts = the Jetson memory priority).

**Expected gain.** Materially lower retrain-to-retrain prediction churn at non-inferior DSR, or a
clean negative confirming the SWA soup suffices. Honest prior: the incremental over SWA is small.

**Measurement plan.** Identical-holdout comparison through `evaluate_on_holdout` (seed-mean vs
SWA-soup vs single best); purged IC via `stage0_preds` + `ic_by_name`; churn metric = correlation
of consecutive weeks' predictions. Adopt only if holdout DSR is non-inferior AND churn drops
materially.

**Jetson honesty.** 3x LSTM training at the weekly retrain (sequential, off-peak — acceptable);
inference = 3 small forward passes/hour/book, loading sequentially, < 100 MB total.

**Citations.** arXiv 2503.13544 (rev. 2025-10, CIKM FinAI wksp), "Decision by Supervised Learning
with Deep Ensembles"; Izmailov et al. 2018 (SWA, prior art); Lakshminarayanan et al. 2017 (prior
art); Liu 2024 (Oxford Bulletin, doi 10.1111/obes.12590, double shrinkage); Elliott & Liao 2025 +
Lee & Lee 2025 (UCR WP 202514) — subset equal-weighting beats fitted weights; Wood, Roberts &
Zohren 2026-01, DeepPM (arXiv 2601.05975 — seed-position ensembling as a turnover device, the
mechanism inside the xLSTM benchmark's cost result).

---

### FR-14 — VSN (variable-selection network) gating front-end as an Optuna categorical on the LSTM leg — CORE, M, confidence: medium

**Technique.** A TFT-style per-timestep learned softmax gate over the ~40 input features
(per-feature GRN encoders, hidden 8–16; output = weighted feature sum) feeding the existing
RegressionLSTM + MHA stack unchanged; ~20–60k extra params; searched as an Optuna categorical
(vsn on/off) so it pays the same selection-pressure deflation as everything else. Evidence tier
upgraded by the dive: VSN+LSTM was the actual WINNER of the only significance-controlled 2026
finance deep-learning benchmark (Sharpe 2.40, HAC t 8.81, 1.14M params, daily futures 2010–2025) —
feature-selection-in-front-of-LSTM is that paper's strongest transferable conclusion, far more so
than its xLSTM headline.

**Expected gain.** Possibly a wash (the benchmark is daily futures); adopt only if holdout DSR
holds. The honest draw is that it is the top risk-adjusted architecture family in the best
available benchmark at near-zero serving cost.

**Measurement plan.** Standard promotion path, no new harness: Optuna trial family vsn on/off
under the identical purged walk-forward objective; winner faces `evaluate_on_holdout`
(Sharpe > 0, DSR ≥ 0.60 on the blended predictor) and `stage0_preds` + `ic_by_name` per-name IC;
then challenger → shadow → B03.3.

**Jetson honesty.** Trivially servable (tens of k params, same torch stack, no new dependency).
Real cost = Optuna search-space growth and one gotcha-#2 reset if the feature-tensor shape
changes — ride Chain-2.

**Citations.** Saly-Kaufmann, Wood, Peter-Calliess & Zohren 2026-03-02, "Deep Learning for
Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance" (arXiv 2603.01820) —
VSN+LSTM top Sharpe with HAC significance, 50-seed protocol; Lim, Arik, Loeff & Pfister 2019 (TFT,
prior art for the VSN block).

---

### FR-15 — sLSTM cell swap as a low-share Optuna categorical (the honest residue of the xLSTM story) — CORE, M, confidence: low

**Technique.** Replace only the `nn.LSTM` recurrence with a self-contained ~120-line pure-PyTorch
sLSTM cell (exponential input/forget gating with log-domain running-max stabilizer, normalizer
state, per Beck et al. 2024, eqs 15–17), parameter-neutral at equal hidden size, keeping the MHA +
FC head and everything downstream untouched; `rnn_cell` categorical {'lstm','slstm'} capped at
~25–30% of trials (hand-rolled recurrence loses cuDNN fusion, ~2–4x slower/epoch). Explicitly NOT
the full xLSTM residual-block stack and NOT the NX-AI package (Section 3, N2's architecture
boundary): the only small-scale 2026 study (Mathematics 14(8):1282, 2026-04) found the full stack
slowest to train and *underperforming* its own constituent cells for short-term financial
forecasting, and the official package needs a custom CUDA kernel compile (sLSTM) or Triton (fast
mLSTM) — JetPack toolchain risk for zero benefit at H ≤ 384. No hourly-bar financial xLSTM
evidence exists anywhere (negative search finding, 2026-08-20).

**Expected gain.** Low — one cheap shot justified purely because the measurement is free (existing
holdout-DSR gate + shadow path). Pre-registered: kill the arm if holdout DSR does not improve
within one retrain cycle, no tuning rescue; on failure, recommend recording "xLSTM/sLSTM
architecture family at hourly scale" as a new kill entry so the daily-futures headline (Sharpe
1.79 at 34x the parameters of the 74k-param LSTM baseline it barely beat) does not resurrect it.

**Measurement plan.** (1) hypersearch holdout-DSR gate on identical purged folds (slstm arm must
beat the lstm arm's holdout DSR, not just CV); (2) `stage0_preds` + `ic_by_name` breadth check;
(3) `backtest.py --gate`; (4) challenger → shadow.

**Jetson honesty.** Training-only cost bounded by the trial-share cap; O(H) state; served artifact
same size as today. Skipping the official package avoids both Jetson landmines (CUDA compile,
Triton on aarch64).

**Citations.** Beck et al. 2024-05, xLSTM (NeurIPS 2024, arXiv 2405.04517 — the equations);
arXiv 2603.01820, 2026-03 (xLSTM turnover/breakeven-cost result and its 2,507,269-vs-73,729 param
asymmetry); "Beyond xLSTM," Mathematics 14(8):1282, 2026-04 (small-scale counter-evidence);
NX-AI/xlstm repository, checked 2026-08 (kernel requirements).

---

### FR-16 — Breakeven-cost sweep on the policy replay (`--fee-mult`): adopt the benchmark's best idea, not its architecture — ADJACENT, S, confidence: high

**Technique.** A lambda grid {1.0, 1.5, 2, 3, 4, 6} scaling all `fees.py` spread+fee legs
uniformly in `backtest.py`, reporting per-book and per-name λ* where net replay P&L crosses zero.
Verified 2026-08-20: no such knob exists in `backtest.py` or `fees.py`. Measurement-only
instrumentation, ships directly (~30–50 lines).

**Expected gain.** A standing cost-headroom metric: (a) a permanent acceptance rule for future
challengers — must not reduce λ* even when holdout DSR ties (cost resilience is exactly the
dimension the 2026 benchmark showed differentiates models that tie on Sharpe); (b) a standing
owner answer to "how much fee/spread deterioration can the current book absorb" — relevant the
day Alpaca crypto spreads widen (lesson 5) or the Nov-2026 tick change lands (lesson 3).

**Measurement plan.** The sweep is the metric: `backtest.py --prefix {'',stock} --days 180` per λ,
champion and challengers side-by-side.

**Jetson honesty.** Pure replay compute, minutes per sweep; no model, no memory, no live-loop
impact.

**Citations.** arXiv 2603.01820, 2026-03 (the breakeven-transaction-cost metric c* and why it
separates models); Alpaca crypto fee schedule accessed 2026-08 (25/15 bps taker/maker tier-1
unchanged — the λ = 1 anchor).

---

### FR-17 — Named stress-window replay panel + bad-print stop-out audit (both books), feeding the B05.3 multiplier calibration — ADJACENT, M, confidence: high (panel) / medium (audit)

**Technique.** Add fixed, dated 2025–2026 stress windows to the policy-replay reporting so every
future backtest run reports per-window drawdown, exit behavior, and realized-vs-modeled cost gap.
Equities: 2025-04-02..2025-04-21 (tariff crash through the CME-measured fill-quality recovery),
2026-03-01..2026-03-31 (Iran risk-off, VIX 25–31.65), 2026-06-01..2026-06-15 (semi unwind).
Crypto: 2025-10-10..2025-10-12 (cascade), 2026-01-15..2026-02-28 (flush), 2026-06-01..2026-06-30
(breakdown). These are the first severe stress episodes inside our own harvested data — use them
to calibrate the B05.3 stressed-exit multiplier m *empirically* (realized/modeled cost ratio per
window) instead of from literature priors; this complements, and does not re-spec, B05.3's
journal-regression calibration. Crypto companion audit (the USDe-at-$0.65 lesson): for every
hard-stop/trailing exit in journals + replay, fetch minute bars around the exit and flag exits
where price reverted > 50% of the stop-triggering move within 30 minutes; report the flagged
fraction and its P&L drag. Also fold in the liquidity re-baseline: when
`scripts/crypto_spread_census.py` (already built; campaign NEW-#1) next runs, add a
pre/post-2025-10-10 split of time-weighted spread and depth, since independent evidence says alt
depth is still ~40–50% below pre-crash.

**Expected gain.** A measured stress cost multiplier per window feeding B05.3; confirmation (or
not) that the exit stack's Mar-2026 drawdown sits inside the DSR gate's assumed tail; a quantified
answer on whether reverting-wick stop-outs are a real tax (> 10% of stop P&L ⇒ evidence for a
confirmation-delay stop *design question*, which would be its own owner-gated, replay-gated pass
since `policy_exits` semantics are shared with labels — gotcha-#2 territory). Honest caveat: our
exits run on hourly bars and Alpaca spot never printed the worst Binance-perp wicks, so the
artifact fraction may be near zero — a valid, valuable negative that closes the question before
anyone proposes exit-stack surgery.

**Measurement plan.** Jetson: `backtest.py --prefix stock --days 500` and `--prefix '' --days 400`
with the B02 per-bar dump, sliced by window (max DD, per-trade slippage vs the `fees.py` model,
gate/exit attribution via `decision_report.py`); the audit script joins journal exits to Alpaca
minute bars.

**Jetson honesty.** Jetson-only (models + parquet); the windows are date constants; pure reporting
change, no model-facing risk.

**Citations.** CME Group 2025, "Reassessing Liquidity: Beyond Order Book Depth" (Apr-2025 volume
+99% / depth −68% / recovery Apr-21); SF Fed Economic Letter 2025-10; Cboe Index Insights 2026-03;
CNBC 2026-06-06; CoinDesk Research 2025-10 ($19B cascade anatomy); AMBCrypto 2025-11 (USDe
venue-oracle artifact); CoinDesk 2025-11-15 / 2026-01-08 + Amberdata (persistent depth damage);
Boudt & Petitjean 2014 (prior art, spread widening at jumps — already in B05.3).

---

### FR-18 — Market-structure re-audits: overnight premium, entry-window IC in the 0DTE era, crypto session-of-week — ADJACENT, M, confidence: medium

**Technique.** Three cheap conditioning audits driven by lessons 2, 3, and 7. **(a) Overnight
premium re-verification:** decompose close-to-open vs open-to-close returns per stock name,
rolling 6-month windows 2024-01..2026-08 (the 24/5 expansion may be eroding the premium
`OVERNIGHT_SLEEVE` harvests); sleeve replay A/B (`OVERNIGHT_SLEEVE_MAX_POSITIONS` 2 vs 0) through
`backtest.py --prefix stock --days 250 --gate`; plus free telemetry — two Alpaca overnight-feed
quote snapshots/day (20:05, 03:55 ET) for held sleeve positions, evaluated after ~60 sessions for
gap-predictive value (fraction of |gap| > 1.5x ATR opens where the 03:55 quote had already moved
> 50% of the gap). An overnight protective-limit EXIT is explicitly NOT proposed — that would be
its own default-OFF, owner-gated design. **(b) Entry-window re-validation:** per-hour purged-IC
buckets (09:45–11:00, 11:00–14:30, 14:30–15:30 ET), full sample vs 2025-01-onward, given 0DTE
gamma effects concentrate in the final two hours; if the late window's 2025+ IC is
indistinguishable from zero, replay a morning-only variant (ENTRY_WINDOWS is policy-facing →
owner path). **(c) Crypto session-of-week:** bucket-split costs (census + journals) and purged IC
across US-RTH / off-hours-weekday / weekend, with a default-OFF weekend size damp (×0.7, one
parameter) tested through the replay gate only if the cost evidence justifies it — costs need far
fewer observations than IC and a cost-only justification suffices for a damp.

**Expected gain.** Keeps three shipped policy choices (sleeve sizing, entry windows, 24/7 crypto
sizing) honest against dated structural change, at measurement cost only.

**Measurement plan.** All via existing harnesses: `stage0_preds.py` dump + bucket-split purged IC;
`decision_report.py` cost attribution; `backtest.py --gate` for any variant. Small-data honesty:
~20k obs per stock bucket (sign, not fine structure — report CIs); weekend crypto IC CIs will be
wide on 6 names.

**Jetson honesty.** Cheap pandas passes + a ~30-line snapshot logger in `stock_loop`'s off-hours
path; RAM-negligible.

**Citations.** Alpaca 24/5 docs accessed 2026-08 (limit-only overnight; free overnight feed);
Simpson Thacher 2026-04-29 (Nasdaq 23/5 approved 2026-04-10); NYSE Extended-Hours FAQ v4.0,
2026-08 (Arca 2026-12-06 target); DTCC "The Shift to 24x5" (NSCC June-2026 target); Božović, SSRN
5223127, 2025 (0DTE jump clustering; 0DTE = 50–63% of SPX volume per Cboe 2025–2026); Kia 2026
(Financial Review 61(2), ETFs lead spot ~85%); DailyCoin 2025 (US-hours liquidity concentration;
11:00 UTC peak, −42% by 21:00 UTC).

---

### FR-19 — Pure-numpy statistical jump model as an OFFLINE regime diagnostic (calibrates B06's hand-set hysteresis; the named HMM-slot replacement candidate) — ADJACENT, S, confidence: medium

**Technique.** ~200 lines of numpy, Mac-runnable: K = 2 discrete jump model (k-means-style state
means + dynamic-programming state assignment with per-transition penalty λ, features = EWMA return
and EWMA downside deviation at halflives {5, 21, 63}d, z-scored trailing-only) fit on daily BTC
and SPY via yfinance; λ over ~15 log-spaced points selected by purged walk-forward CV on OOS
next-24h realized-vol separation (HAC t). Compare its state sequence against B06's
BTC-RV-percentile + hysteresis states and the VIX tiers: dwell times, flips/yr, forward-vol
separation, % agreement. The jump penalty and B06's Schmitt-trigger hysteresis are two
parameterizations of the same persistence prior — the JM's cross-validated λ yields a data-driven
target dwell distribution instead of the hand-set 65-percentile/12h constants. This is the only
2025-sweep regime tool with an honest net-of-cost OOS design behind it (JM beats both
buy-and-hold and HMM out-of-sample 1990–2023 net of 10bp one-way costs on vol/MDD/Sharpe).

**Expected gain.** Calibration evidence for the flag-ON B06 constants; and the recorded
replacement candidate for the kill-pending HMM slot (Section 4, ask #4) — with the default
recommendation still the CUT (a JM would mostly re-learn the RV state).

**Measurement plan.** Offline script reporting per candidate state series: flips/yr, dwell
distribution, purged HAC t of next-24h Parkinson-RV difference, % agreement with B06. Decision
metrics per B06's own success criteria: feed each state series as the de-risk input into the
`backtest.py` policy replay (Jetson) and compare TAIL stats (relative MDD 5–10%, ES99 1–3%,
tier-flip counts in the sizing journal) — Sharpe is a bonus, never the test (Cederburg 2020
kill-aligned).

**Jetson honesty.** Diagnostic runs on the Mac (numpy + yfinance daily bars); only the tail-stat
replay needs the Jetson. Zero live-path code; nothing activates without the B06/D10 owner ruling.
Reimplement rather than pip-install (the jump-models package pulls sklearn).

**Citations.** Shu, Yu & Mulvey 2024-09 (v3), "Downside Risk Reduction Using Regime-Switching
Signals: A Statistical Jump Model Approach" (arXiv 2402.05272; JFDS 2024); Aydinhan, Kolm, Mulvey
& Shu 2024-06 (Annals of OR — JM stability vs HMM); Blanchard 2025 (ASMBI — false-alarm cost
control is the binding constraint, i.e., what hysteresis already is); Cederburg et al. 2020 (JFE,
prior art anchor).

---

### FR-20 — Standing hygiene tools (four S-effort ship-direct items) — ADJACENT, S, confidence: high

**(a) JKP factor-family drift check.** A pandas-only Mac script reading the free jkpfactors.com US
theme returns (updated through 2025-12; release 2026-04) and reporting trailing 24m/36m mean,
t-stat, and Sharpe for the families our features proxy (momentum, short-term reversal,
seasonality, low-risk, quality) into a one-page table in `research/`. Used as a PRIOR in Stage-0
feature-retention runs (a family dead in JKP needs stronger in-house purged IC to retain); never a
direct signal (monthly, long-short, US-wide — coarse family-level context only). Acceptance test:
two known anchors reproduce (momentum positive 2023–2025; average anomaly ~0 net). Refresh on
demand, not scheduled (gotcha #5).

**(b) Long-leg-only anomaly triage.** Fetch Muravyev-Pearson-Pollet's per-anomaly long-leg
decomposition (JF 2025, doi 10.1111/jofi.13501) and store the which-anomalies-retain-long-leg-alpha
shortlist in `research/` as the stock-side feature-family whitelist prior; add a "long-leg
evidence" line to the feature-proposal template. Reorders the queue only — never substitutes for
in-house measurement (wave-5 rule).

**(c) Look-Ahead-Bench cutoff guard in `llm_eval`.** ~10 lines + a unit test: refuse (or loudly
flag) any evaluation row whose evidence timestamp predates the serving model's published training
cutoff (model id + cutoff from `llm_config.json`); live-shadow rows are post-cutoff by
construction. Codifies the memorization-defense rule so no future agent backtests LLM outputs on
historical text. Mac-buildable end-to-end.

**(d) Nov-2026 tick-regime calendar item.** After the first business day of Nov 2026 (the delayed
Rule 612 date — SEC press 2025-130, 2025-10-31): re-run the offline EDGE spread ranking
(`liquidity.py`), re-derive name_class boundaries, recalibrate `IOC_CAP_BPS['mega']` (currently
likely too generous post-reform), archive the pre/post table so the cost model is regime-dated via
`cost_regime.py`. Flagging now prevents a silent cost-model drift in November. Until then: do
nothing (the 2024 rules are still not live).

**Citations.** Jensen, Kelly & Pedersen 2023 (JF) + jkpfactors.com data release notes 2026-04;
Muravyev, Pearson & Pollet 2025 (JF, doi 10.1111/jofi.13501); Benhenda 2026, Look-Ahead-Bench
(+20.73% in-window → −1.04% post-cutoff; verify the primary arXiv source before quoting in an
owner decision — flagged branch); Lopez-Lira, Tang & Zhu 2025-04, "The Memorization Problem"; SEC
press 2025-130, 2025-10-31.

**Deliberately demoted from the candidate list** (recorded so the choice is visible): the TTM-R2
zero-shot TSFM sanity baseline (round-1, confidence medium) — FR-04's Nagel one-liner delivers the
same falsification insurance with zero new pip dependencies, so the TSFM baseline is filed as a
fallback if FR-04's result is ever contested; and the auxiliary direction head on the LSTM trunk
(round-1, confidence low) — legitimate but small; the signal-model wave may pick it up if FR-14/
FR-15 leave trial budget on the table (pre-registered: a null result kills it).

---

## 3. Honest negatives — frontier areas checked and rejected

Each entry records the verdict, the dated evidence, and the revisit trigger, so the question is
not re-derived next campaign. None of these are kill-list entries yet (Section 4 notes the one
recommended addition candidate); they are this round's research verdicts.

**N1 — Time-series foundation models in the model path (adopt/fine-tune/distill): NO.** The two
best finance-specific evaluations converge: TSFMs beat random walk with Diebold-Mariano
significance in only ~2 of 25 model-asset cases on liquid US equities ("not universal engines for
statistically reliable alpha generation" — arXiv 2606.27100, 2026-06-25), and off-the-shelf TSFMs
perform poorly on daily excess returns both zero-shot AND fine-tuned; only from-scratch pretraining
on large financial panels helps (Rahimikia-Ni-Wang, arXiv 2511.18578, 2025-11-23) — a corpus and
compute regime our books do not reach. Distillation targets (TTM, Chronos-Bolt) are generic-corpus
models, exactly the failing class. Revisit trigger: a peer-reviewed 2026+ study showing a TSFM
beating a tuned supervised baseline on HOURLY returns with DM significance after costs.

**N2 — Architecture surgery on the blend (TabM/RealMLP/GRANDE for the LGB leg; Mamba/SSM; KAN;
the full xLSTM residual stack): NO.** TabArena (arXiv 2506.16791, 2025-06, ~25M runs): deep
tabular models reach parity only after heavy tuning PLUS post-hoc ensembling (a Jetson-hostile
4–8x serve-memory multiplier); CatBoost/LGBM-class wins under conventional budgets, and TabPFN v2
degrades precisely under temporal shift ("tree-based models remain the optimal choice... in open
environments" — arXiv 2505.16226, 2025-05). Mamba/Mamba2 posted 0.64/0.78 Sharpe — bottom tier,
below plain LSTM's 1.48 — in the significance-controlled 2026 finance benchmark (arXiv
2603.01820), independently corroborating the rejection. KAN ≈ MLP under fair comparison (2024
fair-comparison line + arXiv 2510.16940, 2025-10). The full xLSTM stack underperforms its own
constituent cells at small scale (Mathematics 14(8):1282, 2026-04) and has zero hourly-bar
financial evidence (negative search, 2026-08-20) — only FR-15's ~120-line cell swap earns one
capped shot. Revisit triggers: single-model (un-ensembled) TabArena-style wins at 50–100k rows
under shift; a peer-reviewed risk-adjusted finance benchmark win for SSMs with cost/seed controls.

**N3 — Differentiable-Sharpe / utility / end-to-end position-output losses as the primary
objective: NO.** Every 2025 headline win (e.g., arXiv 2502.17493) is a daily cross-sectional
backtest with no cost model, no deflation, no purged CV, and author-reported training instability;
none replicates at hourly single-name scale, and at ~8 effective independent portfolio-level
observations per week the non-convex loss is exactly where E2E overfits. E2E losses also require
batch-level portfolio simulation in the training loop — real Jetson cost for unevidenced gain. The
existing holdout DSR gate is the kill screen if anyone ever insists.

**N4 — Conditional latent-factor models (IPCA / conditional autoencoders) and complexity-scaling
at N ≈ 55: NO.** Latent-factor estimation needs cross-sectional averaging a 55-name book does not
have (KPS's own small-sample power caveat); the zero-pricing-error claims are fragile even at CRSP
scale (Zhang, SSRN 4802936, rev. 2025-09), and CAE gains shrink drastically ex-microcaps and net
of costs (Int. Review of Financial Analysis 107, 2025) — our liquid large-caps are precisely the
subsample where CAE adds least. Complexity-scaling is refuted by Nagel 2025-08 and Buncic 2025-09
(lesson 8). Deliberately unmeasurable in-repo: no in-repo measurement could reach power at N = 55,
which is the reason not to build. Usable only as offline risk-attribution research, never the
alpha head.

**N5 — BMA / online adaptive blend weighting / fitted stacking beyond K = 2: NO (frontier-
reconfirmed).** The 2024–2025 combination literature (Liu 2024, Oxford Bulletin double-shrinkage;
Lee & Lee 2025; Elliott & Liao 2025 subset averaging) converges on shrink-toward-equal with
selection — exactly B12.2's NNLS-shrunk-to-0.5 with a significance gate. Nothing beats it at
K = 2; estimation noise kills fitted weights beyond K ≈ 2–3 at our effective sample. B12.2's
documented revisit-online trigger remains the only reopening route.

**N6 — Bayesian online changepoint detection as a live regime input, and regime-conditional model
selection (separate blends per regime): NO.** The 2025 BOCPD evidence (ACM CSMT 2025) is
retrospective event alignment with no net-of-cost trading result; Blanchard 2025 (ASMBI) shows
false-alarm transaction costs are the binding constraint — which our hysteresis already controls.
Per-regime model fits would fragment the stock book's already-starvation-level samples (B04.3's
n ≥ 1000 tiers) below any honest floor. Burden of proof: an hourly-scale net-of-cost OOS result;
none exists as of 2026-08.

**N7 — Cost-aware retraining-decision forecasting, continual learning / warm-start / replay-buffer
adaptation, and drift-triggered-ONLY retraining: NO, all three.** Regol 2025-05's advantage exists
only when retraining is expensive (ours is overnight-free, already hybridized with PSI/CUSUM
triggers — the literature's own recommended operating point); weekly from-scratch retraining at
our model sizes makes catastrophic forgetting a non-problem and CL machinery pure train/serve-
parity risk; drift-only scheduling would miss label-side drift the weekly schedule catches for
free. FR-08's ledger is the standing instrument that could overturn the cadence half with in-house
data.

**N8 — TabPFN-2.5 / 2.6 / 3 and the proprietary distillation engine: license-dead for this
system.** The 2.5+ weights ship under licenses barring "the model, its derivatives, and its
outputs" from "any commercial or production purpose... including internal commercial
decision-making" (tabpfn_2_5 HF card, 2025-11; TabPFN-3 report, 2026-05-12), which plausibly
covers even a personal trading system once real money is at stake; the distillation engine is
enterprise-only. Pin TabPFN v2 (Nature 2025-01, Apache-2.0-derived) as FR-11 specifies; treat
2.5+ as unusable, not merely unnecessary. TabICLv2 is a possible license-clean stronger teacher —
verify its license before ever preferring it.

**N9 — GDELT as a headline archive: NO, definitively.** Free DOC 2.0 artlist mode structurally
returns only the most recent 3 months of any search window (a per-headline 1Y archive is
unretrievable without the heavy GKG/BigQuery pipeline); no ticker tagging (hopeless disambiguation
for SERV/RDW/QS-class names); live probe 2026-08-20 hit a persistent 1-request-per-5-seconds
throttle even at 40+s spacing; and it is a genuinely new archive dependency tripping the
on-chain-flow kill's rationale. Recommend recording as a named negative (Section 4, ask #1) so no
future agent re-proposes it.

**N10 — LLM analyst upsizing, historical-news backtesting of LLM outputs, and prompted headline-
score alpha: NO.** Reported LLM alpha is substantially memorization (+20.73% in-window → −1.04%
post-cutoff, largest models decaying worst — Look-Ahead-Bench 2026; Lopez-Lira-Tang-Zhu 2025-04;
"The Alpha Illusion," arXiv 2605.16895, 2026-05); residual post-cutoff headline predictability
concentrates in small caps and negative news — not our liquid large-cap long-only book; and small
chronologically-clean models match Llama-scale for the bounded scoring the gate does (ChronoBERT,
arXiv 2502.21206, 2025-02). Keep the flash-lite pin (B07.2), keep the analyst as veto/size-tilt
only, decide spend by the llm_eval b2 verdict, and never evaluate LLM outputs against pre-cutoff
text (FR-20c enforces this mechanically).

**Also checked, nothing to adopt (one line each).** Momentum-crash management 2025-flavor (crash
risk lives in the short leg and rebounding bears; the long-only ATR/HAR/de-risk stack already
holds the deliverable; strategy-level vol overlays are kill-adjacent [wave-6]). Calendar
seasonality, PEAD, 52-week-high: no 2025–2026 revival — and JFE 2025's "Warp speed price moves"
(after-hours earnings discovery at millisecond speed) independently *reinforces* the PEAD kill.
Label smoothing under noise: no replicated finance win post-2024 (survives only as the ε = 0.05
option on a BCE auxiliary head, if that demoted item is ever built). The 2025 Nobel (Mokyr /
Aghion-Howitt, announced 2025-10-13) has no operational relevance; no 2026 laureate exists yet.
The multiple-testing frontier adds nothing beyond B03's already-specced machinery — the
anti-stacking guard is the binding constraint on any future methodology adoption. Sequential
bootstrap stays dead [wave-6]; FR-13's seed ensembling is plain seed variation on the LSTM leg,
not a resampling scheme.

---

## 4. Kill-list asks

Per KILL_LIST.md's own rule, entries leave (or are annotated) only by explicit owner decision.
The four asks already filed by the August campaign (02_research.md's asks section) stand
unchanged; this round adds four more. Nothing was built pending these rulings.

**Ask A — Alpaca News archive: data-dependency ruling (blocks FR-10), plus record GDELT as a named
negative.** The on-chain-flow kill [econ-07] rejects NEW unreplicated data dependencies. Research
judgment: the Alpaca News API (v1beta1) is a FREE endpoint of the ALREADY-INTEGRATED broker on the
EXISTING API key — the same posture as the blessed v1beta3 crypto-quote venue census — so it does
NOT trip the rationale; but the boundary is the owner's to draw. Requested: (a) confirm Alpaca
News as the designated archive so `scripts/news_census.py` may run; (b) record GDELT as a named
do-not-build negative (evidence in N9) so the question closes.

**Ask B — Append a dated corroboration to the "Anomaly-short decile book" kill [wave-5, line
~40].** Muravyev, Pearson & Pollet 2025 (Journal of Finance, doi 10.1111/jofi.13501) now provides
top-journal numbers for exactly that kill: across 162 anomalies, long-short returns are +0.14%/mo
BEFORE short-sale costs and −0.01%/mo after actual borrow fees — the entire net anomaly premium
sat in unharvestable short legs. No status change; the ask is to append the citation to the entry
so the kill carries published ammunition.

**Ask C — Deprioritize PENDING OWNER ASK #6 (BTC-dominance / alt-season regime input).** The 2026
data weakens the ask's premise: dominance broke below the 57% "rotation trigger" (56.5%, Aug-2026)
with the Altcoin Season Index stuck in the high-40s, and ETF flow — the marginal price setter —
structurally does not rotate into alts (lesson 7). The dominance-regime conditioning column has
less discriminating power than the 2025 evidence suggested. Recommendation: deprioritize (the ask
remains an ask; nothing built either way).

**Ask D — Annotate the pending HMM-layer cut [rev-07-01] with its sole frontier-validated
replacement candidate.** When adjudicating the D10/B06 HMM cut, record the discrete statistical
jump model (FR-19's object; Shu-Yu-Mulvey 2024 — beats HMM out-of-sample net of 10bp costs on
vol/MDD/Sharpe, 1990–2023) as the only replacement candidate IF any model-based regime slot is
retained inside the B06 min() family. The research default remains the CUT (the BTC-RV percentile
state likely captures the same information at near-zero complexity); the JM enters only via
challenger → replay tail-stats, and only on an explicit owner decision.

---

## 5. Sequencing notes

- **Run this week (ship-direct, no flags):** FR-03 (before the next retrain), FR-01
  instrumentation, FR-04, FR-05, FR-08, FR-16, FR-20a–c, FR-07-A (transfer curves). FR-17's window
  constants can land with the next replay run.
- **Chain-2 riders (the single shared gotcha-#2 study-reset event from B05/B12/B17 — do NOT mint
  new reset events):** FR-06 (rankic objective), FR-07-C (LABEL_ONLY_BARS {4,8} columns), FR-09
  (decay categorical), FR-14 (VSN categorical), FR-15 (sLSTM categorical). All of their trials
  increment `cum_trials` per B03.2; the anti-stacking guard applies.
- **Strict orderings:** FR-01 before FR-02; FR-02 before FR-09 (skip FR-09 entirely on a null);
  FR-05 before funding FR-06's Jetson slot; FR-07-B before FR-07-D before FR-07-E; B04.1 + B04.2
  before FR-11's distillation arm; q10 certification (B12.2 open follow-up) before FR-12's veto
  A/B; Ask A before FR-10; B02 per-leg journals before any third-leg proposal (FR-13's rule).
- **Owner-gated:** FR-10 (Ask A), FR-19's any-live-use (Ask D / D10), FR-18b's window change
  (policy-facing), any stop-confirmation design FR-17's audit might motivate (shared
  `policy_exits` semantics = gotcha #2 + owner).
- **Calendar:** FR-20d fires the first week of November 2026.
- **For the signal-model deep-dive wave (binding):** the CORE menu is FR-01..FR-10, FR-12..FR-15;
  the binding constraints are Section 3's N1–N5 (no TSFMs, no architecture surgery beyond the
  named categoricals, no E2E objectives, no latent-factor heads, no online weights), FR-13's
  third-leg diversity rule, FR-12's no-conformal boundary, and the wave-5 measure-in-house rule
  throughout. The wave should treat the FR-02/FR-07-B/FR-04/FR-05 measurements as its opening
  moves — they are the cheap experiments that decide which heavier candidates deserve their
  Chain-2 slots.

*End of Phase 5 synthesis. The round-1 reports and dive transcripts remain the detailed record of
why; this file is the actionable owner surface.*
