# Campaign 2026-08 — Phase 7: Decision-Influence Ledger

> ## ⚠️ ANALYSIS ONLY — NOTHING CHANGED
> This document is a READ-ONLY audit. No code file was modified, no flag flipped, no config
> touched. The owner's instruction, verbatim: *"There are many influences to a decision. Do all
> belong? To what degree? Change nothing but just consider this."* This is the owner's thinking
> companion, **not a work order**. Every "remove", "merge", or "retire" below is a verdict on
> paper awaiting the owner's ruling and, in almost every case, Jetson evidence that does not yet
> exist. Produced 2026-08-22 by a 3-census / 3-judge / 1-synthesizer panel against the current
> working tree (uncommitted state atop `c7f846e`). Citations are working-tree `file:line`.

**Method.** Three census agents walked (1) the entry-funnel gates, (2) the sizing path, and
(3) the signal-formation/feature/exit paths, cataloguing every influence with mechanism, origin,
bounds, what it actually proxies, and its evidence pointer. Three judges then independently ruled
on every influence. This synthesis merges the three verdicts; where judges disagreed, the
disagreement is presented, not papered over. Priors honored: the 2026-07-02 review (VIX
double-count, sentiment triple-count, winner's-curse anchor conflict), the D10/D26/D29 de-risk
findings, `research/KILL_LIST.md` (read in full), and the wave-5 rule — **no composition/sizing
change ships on literature priors; in-house measurement first.**

---

## 1. The funnel as it stands

### 1.1 Cycle level (both books, `base_loop._run_one_cycle`)

| # | Influence | Mechanism | Live? | Source |
|---|---|---|---|---|
| 0 | Remote /flatten check + halt latch | exit/halt | LIVE (per-book fan-out = c26 D17 fix, in tree) | base_loop.py:276, 2292-2390 |
| 1 | Market-hours gate | hard gate | LIVE (crypto always True) | base_loop.py:278; stock_loop.py:119-135 |
| 2 | Circuit breaker (5% daily, account-wide) | hard gate + one-shot flatten | LIVE; fail-closed on API error | base_loop.py:590-675 |
| 3 | Stock 15:50 EOD flatten (+ only orphan sweep) | exit rule | LIVE | stock_loop.py:300-415 |
| — | stops → reload → macro → predictions → LLM → **sells → LLM-veto sells → buys** | ordering | LIVE (exits free risk budget before buys) | base_loop.py:357-375 |

### 1.2 Crypto per-symbol entry funnel (`base_loop._execute_buys:2427`, universe order — rank_map is annotation-only)

halt/stand-down → cooldown → 24h lockout → trade budget → position cap → **no_pred**
(fail-closed) → **no_quote** (fail-closed) → Hurst<0.45 threshold×1.3 (**dead** — levels bug) →
**cost floor** (`should_trade`, 2.0× round-trip on live quotes) → **below_threshold** →
winner's-curse (SMA20+2ATR ⇒ 1.5× bar) → correlation (>0.7 block) → macro_halt/vix_block
(stock-typed, inert here) → sentiment gate≤0 (**unreachable**) → LLM veto (s<0.15) → meta veto
(p<0.30) → q10 tail veto → `_compute_position_size` (→ sizing_zero) → order.

### 1.3 Stock per-symbol entry funnel (`stock_loop._execute_buys:796`, hand-duplicated, iterates only top-7 by pred rank)

flattened_today → FOMC/CPI stand-down → entry window (09:45-11:00, 14:30-15:30) → SPY-trend
fetch → exposure fetch (fail-closed) → per name: already_held → MAX_EXPOSURE ≥$50k (**break**)
→ bucket cap → cooldown → lockout → budget → earnings≤1d → EDGAR 8-K/M&A → no_pred →
**below_threshold** (BEFORE quote; NO Hurst shift — order diverges from crypto, skewing
cost_floor attribution) → no_quote → cost floor → winner's-curse → correlation → **VIX>35 halt**
→ **VIX>25 non-safe-haven block** (SAFE_HAVEN names not in universe ⇒ de facto full halt, gap C3)
→ SPY<200d trend filter → sentiment → LLM → meta → q10 → sizing → bracket order.
Upstream ordering bias: top-N=7 / hold-rank=15 construction + high-VIX RR_5 demotion
(stock_loop.py:559-581).

### 1.4 Sizing path (`_compute_position_size`, legacy-live)

(0) **stablecoin emergency zero** (short-circuit before everything — D26 fix in tree,
base_loop.py:1902-1909) → (1) base = min(equity×0.005/stop_dist, NOTIONAL_PER_SYMBOL) →
(2) kelly_mult [0.5,1.5] (outside tilt clamp) → (3) vol_mult [0.5,1.5] → (4) tilt product:
signal_conf × **vix_tilt** × dd_mult × **macro_mult**(VIX tiers × STLFSI2 × **pseudo-CAPE** ×
stablecoin) × f_corr × **hmm_mult** × **disagree 0.8** × sentiment × llm × meta × extra_tilt
(funding) × book_vol_mult, with kelly retro-capped ≤1.0 at VIX>25/HMM-bear → clamp [0.1, 1.3] →
degraded-mode min(·,0.5) → (5) sized = base×kelly×vol×tilt → ENB book-risk shrink
(MAX_BOOK_RISK_PCT=0.025) → room cap → leveraged-ETF divisor → MIN_ORDER_NOTIONAL=100 dust gate.

**Between admission and the wire a legacy entry passes ~22 distinct influences, of which only
two (signal_conf, meta_mult) read the model at all.**

### 1.5 Exit stack

Hard stop / trailing (2-reading confirm, client) / TP 2:1 / server resting stops (no delay) /
signal exit (**1-reading, cooldown-gated** — the unreconciled rev-07-02 asymmetry) / stock
rank-drop exit (no offline counterpart) / EOD flatten + sleeve / 24h lockout (currently also
locking out trailing WINS, D20). **Crypto has no vertical barrier while its labels train on
fb-bar verticals, and no orphan sweep (C4).**

### 1.6 Legacy-live vs flag-gated (all flags default OFF today)

Live today = everything above: hardcoded 0.6/0.4 blend, raw-LSTM-searched threshold (D05/H2),
leaky in-sample meta calibration (D12/D13), unconditional lockout incl. trailing wins (D20),
calendar-day earnings windows (D07), dead Hurst gate, dead stock Daily_Sentiment (D27), ~14
constant daily features (D11), kill-listed calendar columns, pseudo-CAPE cutting live entries.
Built-but-OFF: DERISK_STACK_V2 (shadow-journaled on every fill), STOP_CLASSIFY_V2,
EVENTS_TRADING_DAY_WINDOWS, HYPERSEARCH_V3 / OBJECTIVE_V3 / BLEND_FIT_ON_REFIT /
BLEND_THRESHOLD_RESELECT / LGB_REFIT_FULL, CALIBRATION_V2 / META_OOF_PRED /
META_REPLAY_POLICY_PARITY, HURST_ON_RETURNS, DAILY_RESTORE, FUNDING_Z_TIME_THINNING,
TRADER_COST_REGIME_FEATURES, stationary_lean preset, CRYPTO_TREND_GATE (unwired no-op),
edge-Kelly (unwired), GATE-1 cross-book clamp (measurement-only).

---

## 2. The true dimensions of the decision

The ~45 censused influences measure roughly **eight** underlying things. The system's degrees of
freedom exceed its information content by ~5×.

| Dimension | Current readers | Redundancy verdict |
|---|---|---|
| **1. Equity fear / risk-off regime** | VIX ×5-6: >35 halt, >25 block, inline vix_tilt ladder, macro_mult tiers, Kelly regime cap, RR_5 demotion trigger; + SPY-200d filter (slow correlate), macro stop_mult, STLFSI2 (distinct-but-correlated) | Worst multi-count in the system (D10). Deserves **2 reads**: one extreme hard halt (>35, journaled) + one graded sizing tier map with hysteresis (vix_tier_mult_v2); crypto's regime read should be BTC-RV, not equity VIX. v2 consolidates the *sizing* reads only — the two admission blocks were **never adjudicated by anyone**. |
| **2. Realized / forecast volatility** | ATR in the risk base, vol_mult (GARCH; HAR dead per D30), book_vol_mult, HMM high_vol, btc_rv (v2), stop/trail/TP geometry, winner's-curse anchor, vol features into both models | Deserves **2**: ATR-in-the-stop (per-name — the base already vol-normalizes) + one book-level scalar owning PORTFOLIO_VOL_TARGET. The current per-position+book double-apply is D29's 0.25× worst case. |
| **3. Model confidence / edge** | One `pred` read ×6-7: trade_threshold, signal_conf, meta feature, conviction tier, stock ordering + rank-drop exit, sleeve keeper rank, signal exit; plus meta_p, q10, LLM echo of the same state | Deserves **3**: one cost-anchored admission bar, one calibrated size mapping (edge-Kelly replacing signal_conf+meta_mult), one tail check (q10, once certified). The four-bar family (threshold / cost floor / meta / q10) is not redundancy but **mutual miscalibration** — each bar calibrated on a different population than the one it gates (theme C2). |
| **4. Transaction cost / churn** | cost floor (live quotes), cooldown, daily budget, hold-rank hysteresis, dust gate | The cost floor is the one honest instrument in the whole system and should be the **reference frame** the threshold is expressed in. Cooldown+budget are two instruments for one jitter latent; cooldown also leaks into gating EXITS (base_loop.py:1625, 1673) — an entry-economics argument applied to holding vetoed risk. |
| **5. Crowding / concentration** | One correlation matrix ×3 (admission gate, f_corr, ENB budget) + bucket cap + MAX_EXPOSURE + funding tilt + sentiment-crowd | Deserves the **ENB stop-risk budget as sole owner** (risk-denominated, continuous, shrink-to-fit), plus the funding tilt once its z baseline is honest (D28: 2.8-day baseline vs intended ~90). Notional caps stay only as dumb backstops. |
| **6. Account health** | Circuit breaker (acute daily), dd_mult (chronic high-water-mark), kelly_mult (realized edge), book_vol partly | The one defensibly multi-read dimension — three different timescales. Verdict: all three stay, **never a fourth**. Inputs were the problem (D15 peak seed — fixed; D06 winner-censored Kelly sample — recovery in tree, history still poisoned; D29 deposit-contaminated equity). |
| **7. Event risk (scheduled + unscheduled)** | FOMC/CPI stand-down, earnings-1d block, EDGAR veto, EOD flatten/sleeve pair | Genuinely orthogonal to everything else; the cleanest family. Flaws are bounds bugs (D07 calendar-days), not redundancy. |
| **8. Operational availability / epistemic health** | halt flags, market hours, no_pred/no_quote, staleness guard, degraded-mode clamp | Non-market, non-negotiable, fail-closed on the money path. Note: consolidation elsewhere makes the degraded contract MORE load-bearing (today 12 overlapping advisory reads accidentally insure each other's fail-open defaults). |

**Two form errors generate most of the excess:**
- **Multiplying comonotone estimates.** The legacy tilt multiplies 3-4 readers of one risk-off
  latent as if independent → modal entries measured at 0.39-0.56× intent, crisis products of
  ~2e-6 rescued ~45,000× by the 0.1 floor, where every differentiating signal goes inert exactly
  when it matters. Min-within-family (Fréchet bound) is the correct aggregation — DERISK_STACK_V2
  implements it, for sizing only.
- **Hard blocks stacked on graded cuts over the same variable.** Sequential vetoes already
  compose as a min; a VIX>25 block PLUS a VIX>25 graded 0.5× charges the state twice and costs
  breadth. Rule the stack currently violates: per latent state, at most one graded instrument
  plus at most one extreme-tail hard stop.

Also symmetric and worth stating: with a theoretical 4.4× boost product above a 1.3 clamp, the
entire boost side of sentiment/LLM/meta/HMM/conf is largely decorative. **Both ends of the
advisory stack are clamp-dominated — most tilt DOFs are already functionally dead**, which is
the strongest argument that consolidating them (v2) changes less than feared.

---

## 3. The ledger

Verdict key — **KEEP**: belongs as-is. **KEEP-COND**: belongs, at reduced/repaired degree.
**MERGE**: belongs only as part of a consolidated family. **UNPROVEN**: keep only while its
instrument accumulates evidence (some with removal as the null hypothesis). **NO**: does not
belong (dead / fake / duplicate as a matter of code fact — the only class ruleable without
Jetson numbers, still owner-gated).

### 3.1 Structural / ops (all KEEP)

| Influence | Degree | Evidence |
|---|---|---|
| /flatten + halt latch, market-hours, EOD latch, stablecoin emergency zero | Hard, binary, first in order. D17/D26 fixes in tree; the zero's pre-floor short-circuit is exactly right | n/a by design; no depeg on record (insurance may live unfired) |
| Fail-closed presence guards (no model/no_pred/no_quote, 180s staleness) | Hard, unconditional | Staleness guard **inert under alpaca_compat** (timestamp dropped); quote_age_s journaled, read by nothing. Presence ≠ validity: “pred present” masks D11 |
| Degraded-mode 0.5 cap | Keep both stacks; becomes MORE load-bearing after any consolidation | journaled (degraded_inputs), zero rows |
| Clamp family (TILT_MAX 1.3 / 0.1 floor / dust gate) | Keep. The floor's 45,000× rescue indicted the legacy PRODUCT, not the floor. Forward constraint: **edge-Kelly must sit outside TILT_MAX or it is dead on arrival** (strategy_config.py:481) | floor-binding frequency = the single best over-crowding statistic; zero rows |

### 3.2 Event-risk family

| Influence | Verdict | Degree / note |
|---|---|---|
| FOMC/CPI stand-down | KEEP (J2 dissent: *unproven* — fires before journaling, so its windows are structurally unpriceable) | Hard, narrow. Needs a staleness alarm for the static date table + journaled stood-down skips so the counterfactual becomes priceable |
| Earnings-within-1-day block | KEEP | Hard; land trading-day windows (D07, the standing P0, fix built OFF). Fail-open for entries is right (sleeve keeps fail-closed) |
| EDGAR 8-K/M&A veto | KEEP | Nearly free, rare-fire; may live permanently on priors — acceptable for regime-break insurance |
| Circuit breaker | KEEP-COND | Keep hard/fail-closed; the account-wide scope cross-contaminates books and crypto weekends are judged against Friday's close. Per-book (or weekend-aware) baselines + trip journaling. A breaker tripping >1-2×/yr is doing the sizing stack's job at infinite cost |
| EOD flatten + overnight sleeve | KEEP-COND | Flatten hard; sleeve capped 2/5%. Sleeve's 30% ON_Mom_252 rank leg is a D11 constant (silently pure-pred); D07 hole open; no instrument prices forfeited premium vs sleeve capture — the flatten/sleeve boundary is faith-based today |

### 3.3 Admission gates

| Influence | Verdict | Degree / note |
|---|---|---|
| **Cost/edge floor** (2.0× live-quote round trip) | **KEEP** — unanimous; the best-justified influence in the system | Promote to the reference frame: the threshold should be a searched margin ABOVE it (OBJECTIVE_V3). Its defect is its mirror — training/backtest price flat 0.10% (D04/D05): the gate is right, the rest of the world is wrong. Unify gate order across books |
| trade_threshold | KEEP-COND | One signal bar, re-anchored: crypto search range [0.05,1.0] sits entirely below the ~1.20% floor (D05) and was selected on raw LSTM but served on the blend (H2). Until Chain-2, the cost floor is the de facto crypto threshold and this parameter is mostly dead weight |
| Hurst threshold shift (×1.3) | **NO — unanimous** | Dead three ways: levels-not-returns input (never fires), absent from stock path, journal-invisible. Delete the branch; regime conditioning, if wanted, ships as the corrected FEATURE through retrain (gotcha #2), not a hand rule |
| Winner's-curse anchor (SMA20+2ATR ⇒ 1.5×) | UNPROVEN, **null = removal** | Re-reads two model features; rev-07-02 anchor-horizon conflict open ~7 weeks. It IS journaled — price its skips in one decision_report run, then rule |
| Correlation admission gate (>0.7) + f_corr | MERGE → ENB budget | Three consumers of one matrix; this pair is the weaker two (missing-pairs→0.0 loosens when sparse; abs() clones hedges; binary cliff). One estimate, one consumer with teeth |
| **ENB book stop-risk cap** (2.5%) | **KEEP** — the honest instrument | Merge target for MAX_EXPOSURE, bucket cap, corr gate, f_corr. Fix D29 equity denominator + the covered-none rho=0.0 prior bypass |
| Static notional caps (per-symbol, room, MAX_EXPOSURE $50k, bucket, ETF divisor, dust) | MERGE | ETF divisor (pure unit correction) + dust gate + per-symbol backstops keep as-is; MAX_EXPOSURE and bucket cap are notional-denominated duplicates of the ENB cap — demote to backstops once account_risk journals show the risk cap binds first. Static $ constants silently revert risk-sizing to fixed-notional on a larger account |
| VIX>35 halt (stock) | KEEP-COND | Keep as **the ONE** extreme VIX read: cycle-level, journaled as a priced skip class (today it is not in GATE_REASONS — the highest-impact stock gate has the least evidence) |
| VIX>25 non-safe-haven block | **NO** (J1, J3) / UNPROVEN-suspect (J2) | 25-35 is exactly the band the graded tier map prices — hard block + graded cut = double-charge; and SAFE_HAVEN names are not in the tradable universe (C3), so this is a full book halt nobody designed. If a defensive tilt survives, it is an ordering preference, not a veto — and requires the safe-haven names actually being tradable |
| SPY 200d trend filter | UNPROVEN | In GATE_REASONS (priceable, zero rows). Slower latent than VIX but co-fires in every drawdown. Decide **jointly with the B20 SPY hedge** (pending ask #2): if the hedge lands, trend-conditioned hedging beats trend-conditioned entry-blocking and this gate double-counts |
| Sentiment gate (veto + multiplier) | KEEP-COND / UNPROVEN-suspect (J3) | Veto branch mathematically unreachable (clamp floor 0.15) — delete the dead limb. Channel budget must be decided BEFORE fixing D27 (dead stock feature): fixing it silently re-arms the rev-07-02 triple-count (feature + gate + LLM dossier reading the same headlines). One channel + at most one advisory read; no attribution instrument exists (gap #5) |
| LLM veto + multiplier | UNPROVEN — judges split on interim form | **Disagreement, honest:** J1 would demote the veto to advisory NOW (an unproven influence should not wield hard blocks); J2/J3 keep current bounded form strictly until a repaired llm_eval reaches n≥60 valid rows, then honor the pre-registered keep/kill verdict *including the kill branch*. Common ground: zero valid measurement exists (D09 statistically void; D33 fabricated neutrals — both producer fixes in tree), it is the third read of the same headlines, and no expansion of LLM influence before the b2 number exists |
| Meta-label veto + tilt | KEEP-COND | The one calibrated-probability stage the minimal funnel keeps — but today p proxies the primary's in-sample optimism (D12 in-sample pred feature, leaky same-slice isotonic, floorless replay), so the second opinion re-approves the first. Restore authority via META_OOF_PRED + CALIBRATION_V2 + replay parity; the flat-topped clip(2p,0.6,1.3) discards exactly the gradation that justifies the layer — replaced by edge-Kelly, never coexisting with it |
| q10 tail veto | UNPROVEN | Strongest new rationale among model gates (model-conditional left tail — nothing else measures it), weakest certification: floor calibrated on the booster's own early-stopping slice (M4, the forbidden pattern), never holdout-certified (D25). Keep only through the R2C-03 gate: LGB_REFIT_FULL + reliability_report coverage 10%±3pp; fails ⇒ retire (a miscalibrated tail veto is a random ~15% entry tax). Longer-term: grade it, don't cliff it |

### 3.4 Ordering / selection (stock)

| Influence | Verdict | Degree / note |
|---|---|---|
| Top-N=7 rank admission | KEEP-COND | A hard gate wearing ordering clothes: ranks 8+ never journaled (no counterfactual, ever), ranked partly on D11 noise, winner's-curse of ranking correlated predictions unaccounted (same sin as D22). Journal ranks 8-15 as skip rows; do not tune N until D11 is fixed and rank_gradient is D35-repaired |
| Hold-rank hysteresis (15) + rank-drop exit | KEEP | The most principled influence in the funnel — a Garleanu-Pedersen no-trade band pricing turnover cost, which nothing else captures. Give the rank-drop exit its own exit_reason (currently conflated with signal_sell) and either add it to the replay kernel or document the certify/deploy divergence |
| High-VIX RR_5 tiebreak demotion | UNPROVEN-suspect → removal presumption (all three judges) | Fourth VIX consumer; re-reads a model feature as a hand rule; produces only log lines — structurally unmeasurable forever as built. Journal it within one release or delete it |
| Entry windows | UNPROVEN | Forfeits ~72% of session on a wave-8 prior the kill list itself undercuts (first-half-hour momentum killed as weak at hourly single-name); overlaps the cost floor's spread dimension. Populations currently non-comparable across books — fix journal comparability, then one Jetson month decides. If the effect is mostly spread, the cost floor already prices it |
| Cooldown + trade budget | MERGE | One churn instrument (keep cooldown; budget as a runaway backstop it should never touch). **Remove cooldown gating from EXITS** — an entry throttle delaying risk reduction inverts the tool's purpose (base_loop.py:1625, 1673) |
| Hard-stop 24h lockout | KEEP-COND | As-built it enforces the defect list in reverse: trailing server-stop WINS get the 24h ban (D20 — anti-momentum, wrong sign); shared unprefixed file lets books clobber each other. Hard-stop-only (flip STOP_CLASSIFY_V2 once server_stop_kind journals confirm the misclassification rate), per-book prefix, mirror into the meta replay. The 24h span itself is a guess — measurable now |

### 3.5 Sizing

| Influence | Verdict | Degree / note |
|---|---|---|
| Risk base (equity×0.005/stop_dist) | KEEP | The correct spine — and it ALREADY vol-normalizes per name via the stop, which convicts vol_mult. Input repairs: D15 (fixed), D38 forming-bar ATR, and the stop_mult ordering bug (size computed on the un-tightened stop) |
| kelly_mult + VIX/HMM regime cap | KEEP-COND | Right family (shrunk half-Kelly, skeptical prior, outside the clamp); poisoned sample (D06 winner-censoring — recovery in tree, history still biased to the floor exactly when winning). Hold neutral until ~50 uncensored trades/book rebuild; fold the regime cap into the ONE VIX map; the HMM trigger dies with the HMM |
| vol_mult (per-position GARCH/HAR) | **NO** (J1) / MERGE→retire (J2, J3) — net: retire per v2 | Triple-counted vol: base already normalizes per name; PORTFOLIO_VOL_TARGET double-applied with book_vol (D29, 0.25× worst case); HAR structurally dead (D30) so silently GARCH-on-a-forming-bar (D38). Kill list already rejected per-ticker vol-timing as invisible. v2 pins it to 1.0 — correct |
| book_vol_mult | KEEP-COND | The sole legitimate portfolio-vol instrument (catches correlation buildup per-name vol can't see); de-risk-only [0.5,1.0] is right. Legacy input measured-poisoned (D29 deposit contamination — the series that fabricated the beta-ledger alpha; one transfer pins 0.5× for ~3 months). v2's outlier exclusion mandatory; sole owner of the vol target |
| signal_conf | MERGE | Sixth read of one pred, linear in a value the threshold just gated, denominated in a miscalibrated threshold. Interim keep at narrow bounds (nearly decorative under the clamp); **replaced by edge-Kelly together with meta_mult** — never three confidence multipliers |
| vix_tilt inline ladder | **NO — unanimous** | Literal duplicate of macro_mult's VIX tiers on the same reading at near-identical breakpoints: modal market pays 0.7×0.8=0.56× from ONE number (the measured heart of D10). No hysteresis; applied to crypto where equity vol is the wrong state variable. Dies with the v2 flip |
| macro_mult composite | **NO as packaging** | Four unrelated quantities pre-multiplied into one opaque, journal-inseparable scalar. Members adjudicated separately: STLFSI2 stress **survives** (distinct funding-system latent, correctly a min-family member in v2); VIX tiers → the one map; stablecoin → already a hard gate; CAPE → below |
| pseudo-CAPE 0.7× | **NO — unanimous, the cleanest verdict in the audit** | Kill-listed by three sources ("fake data driving a real haircut"), still cutting ~every stock entry 30% when its fabricated z>1.5. **Min-aggregation does NOT launder it** (a fake signal inside a min still binds when it is the minimum) — v2's exclusion plus code deletion (pending ask #3) is the only sound treatment. Needs no measurement: the input is fictional by construction |
| hmm_mult + disagree_mult | **NO — unanimous** | HMM: kill-recommended, smoothing inverted (switches on ONE observation), re-infers state from bars GARCH already read, and is the only de-risk that can BOOST (procyclical leverage from a noisy estimator). The 0.8× disagreement penalty is a tax on the stack's own internal redundancy — remove the redundancy and the penalty has nothing to arbitrate. Both excluded from v2; execution = the pending rev-07-01 owner ratification |
| dd_mult ladder | KEEP | The account's OWN state — genuinely distinct from every market read ("the world is risky" vs "we are wounded"); correctly its own product family outside the v2 min. D15 fixed. Rungs are owner risk preference, not measurement targets |
| extra_tilt (funding crowding; stock event/tape tilts) | KEEP-COND | Real dimension, kill-list-respecting design (de-risk only, carry stays dead) — but the live z baseline holds ~2.8 days not ~90 (D28): the cuts fire on noise, a persistent size leak on the 24/7 book. Suspend-or-fix first (FUNDING_Z_TIME_THINNING + funding_drift_audit); stock tilts get journaled counterfactuals or retirement |
| **Legacy multiplicative composition itself** | **NO — unanimous** | The central error, independent of members: multiplying 12-14 factors that collapse to ~5 latents, fear and vol priced 3-4× each. Measured: 0.39-0.56× modal drag; crisis floor-rescue where "does it belong" becomes literally unanswerable because nothing differentiates. Replace with v2 **on the sizing_cofire evidence** per the wave-5 rule — the direction is not in doubt; the flip still waits for the numbers |
| **DERISK_STACK_V2** | KEEP as the correct future — flip gated | The direct answer to this audit's question, for sizing: one latent counted once (min within {vix-tier w/ hysteresis \| btc_rv, stress, bookvol}), modal regime sized 1.0, crypto de-risked on its OWN vol, fakes excluded, one vol-target scope, fail-open, always-shadow-journaled. Zero real rows. Scope caveat carried to the ruling: **v2 does not adjudicate the VIX admission blocks** — extend the one-count ruling to them in the same sitting |

### 3.6 Signal formation and features

| Influence | Verdict | Degree / note |
|---|---|---|
| LSTM leg | KEEP | Model file clean per the R2-B panel; defects are selection-machinery (D22 winner's-curse checkpoint, M1) — all flag-built repairs |
| LGB leg + 0.6/0.4 blend weight | KEEP-COND | A second leg = legitimate variance compression; a NEVER-fitted weight (D23 — zero callers, docstring calls 0.6 wrong) certified on a predictor that never trades (D25) is an influence running on default, not evidence. BLEND_FIT_ON_REFIT + BLEND_THRESHOLD_RESELECT + cert==deploy (H1) — one coordinated Chain-2 activation. **Deepest open question: the blend has never faced its null** — run naive_vs_blend (FR-04) before tuning anything downstream |
| Feature hygiene (D11 constants, ROC≡Return_12h + MACDs/STOCHd dupes, Month_sin/cos + Turn_of_Month, dead Daily_Sentiment, absent BTC context D31) | **NO** for dupes/calendar/constants-as-served | Kill-listed columns still shipping; ~14 features served as constants training saw as real (every live prediction sits in a training-atypical region — corrupting the pred that six downstream influences consume). One bundled gotcha-#2 event at Chain-2: stationary_lean + DAILY_RESTORE + the D31 contemporaneous-BTC restoration on survivor-#10 sign-off. Absences are belonging failures too: the crypto model is blind to its main common factor |
| Meta feature recycling (10 primary features + pred into the secondary model) | KEEP (principled) | Deliberate AFML conditioning, not accidental double-count — but only after the pred feature goes OOF (D12); today the second pass mostly re-approves the first |

### 3.7 Exits

| Influence | Verdict | Degree / note |
|---|---|---|
| ATR exit family (hard stop, trailing+activation, 2:1 TP, server resting stops, 2-reading confirm) | **KEEP** — the parsimony success story | One parameter family (entry ATR) generates stop/trail/TP coherently; one shared kernel (policy_exits) makes label==backtest==live the closest thing to *certified* in this repo. TP:stop being one belief, not two, is a feature |
| Macro stop_mult tightening | **NO in current form** (J3) / fix-ordering (J1) | Incoherent as ordered: size computed on the un-tightened stop, then the traded stop tightened — under-deploys risk AND raises stop-out probability, exactly in stress; also another VIX-derived read reaching a third subsystem. Either tighten BEFORE sizing or remove it and let the regime family's size cut be the whole stress response — the current ordering must not survive |
| Signal exit (1-reading, cooldown-gated) | KEEP-COND | Un-gate from cooldown; reconcile the 1-reading vs 2-reading asymmetry deliberately (rev-07-02 conflict #2 — accident, not design); note it applies a long-searched threshold symmetrically (D24-adjacent) |
| Missing crypto vertical barrier + orphan sweep | **NO (the absence does not belong)** | Labels are triple-barrier with an fb vertical; live crypto holds are unlimited and untracked coins ride stopless (C4) — the meta model learns a horizon the deployed book doesn't enforce. Add an fb-anchored max-hold (the offline kernel already supports it) or re-label; plus the sweep |
| Server resting stops / brackets | KEEP | Same stop math, no-latency backstop; D20 classification fix pending its evidence gate |

---

## 4. Minimal-sufficient-funnel sketch — a DISCUSSION OBJECT, not a proposal

If every verdict above were adopted (it should not be, wholesale — most await evidence), the
~45 influences reduce to **~15**, with every latent read once:

- **Structural:** presence guards · ops/venue gates · event stand-down family (FOMC/CPI +
  earnings + EDGAR, one family) · circuit breaker (per-book) · one churn control (entries only).
- **Admission:** ONE cost-anchored signal bar (threshold = searched margin above the live-quote
  floor) · ONE calibrated probability gate (repaired meta) · ONE tail check (certified q10) ·
  ONE extreme-regime halt (VIX>35, journaled) · ENB risk family (per-name + book, absorbing the
  correlation gate, f_corr, MAX_EXPOSURE, bucket cap) · top-N with journaled counterfactual +
  the hysteresis band.
- **Sizing:** risk base × bounded Kelly × ONE min-aggregated regime family (vix-tier|btc_rv,
  stress, book-vol) × dd ladder × ONE edge-proportional confidence leg (edge-Kelly, outside the
  clamp, replacing signal_conf+meta_mult) × honest funding tilt — under the existing
  floor/cap/degraded clamps.
- **Exits:** the ATR family · un-cooldowned signal exit · vertical/EOD on both books · lockout
  on true hard stops only.

Everything else either merges into these or must **earn re-entry with a journaled
counterfactual**. Corollary rule for the dormant backlog: activation is itself a parsimony
decision — edge-Kelly REPLACES signal_conf+meta_mult; GATE-1 becomes the outer bound of the ENB
family, never a parallel cap; the crypto trend gate wires only with the survivor-#10 sign-off
and the >70% co-fire kill test (and must not stack on btc_rv without floor logic); COST_REGIME's
VIX feature waits behind the D10 consolidation or it extends the multi-count into the signal.
Score every activation by net readers-per-latent: the funnel should end each change with fewer.

**Honest caveats on this sketch.** (1) It is built almost entirely on structural judgment —
not one shipped instrument has produced a real number; the wave-5 rule blocks every graded
verdict until Chain-0 runs. (2) Judges did not fully agree: the LLM veto's interim form (advisory
now vs keep-until-measured), the VIX>25 block (delete vs suspect-and-measure), vol_mult (delete
vs merge), and the FOMC stand-down (keep vs unproven) are genuine splits recorded above.
(3) Consolidation removes the accidental insurance of redundant fail-open reads — the degraded
contract and per-input health checks become first-class invariants afterward, which v2 already
treats correctly.

**Aggregate distribution** (~45 influence groups): ~11 keep as-is (backstops, unit corrections,
the parity-certified exit family, the cost floor) · ~7 keep-reduced · ~6 merge (correlation,
VIX, vol, notional-cap families) · ~10 unproven-keep-measuring · ~5 unproven-suspect
(removal-presumption) · ~6 do-not-belong (legacy composition, pseudo-CAPE, HMM+disagree, dead
Hurst gate, vix_tilt duplicate, kill-listed columns — plus two *absences* that don't belong:
the missing crypto vertical and the untradable SAFE_HAVEN set). The owner's suspicion is
confirmed: roughly a third of the decision's influences are dead, fake, duplicate, or
kill-listed-but-live — and **not one advisory influence has a positive marginal-value
measurement**.

---

## 5. Open questions → the in-repo instrument that answers each

| # | Question | Instrument (exists unless noted) | Blocked on |
|---|---|---|---|
| 1 | Does the blend beat a naive trailing-EWMA baseline at all? (everything downstream is second-order to this) | `scripts/naive_vs_blend.py` (FR-04) | one Jetson run |
| 2 | Legacy vs v2 sizing: co-fire matrix, floor/clamp saturation, modal drag, v2-vs-legacy per fill | `scripts/sizing_cofire_report.py` + the per-fill `sizing.*`/`sizing.v2.*` journal | Jetson fills; gates the DERISK_STACK_V2 flip AND the pseudo-CAPE deletion ruling (KILL_LIST ask #3) |
| 3 | Does the LLM spend earn its place beyond echoing pred+sentiment? | `llm_eval.py --days N` b2 echo-gap at n≥60 | its own D09 repair first (verdict void until then); D33 producer fix in tree |
| 4 | What do the UNPRICED book-wide gates cost (VIX>35/25, stand-down, entry windows, ranks 8+, breaker windows, RR demotions)? | `decision_report.py` GATE_REASONS — **after** these gates gain journaled skip rows (they have none today; the highest-impact gates are the least priced) | small journaling additions + Jetson days |
| 5 | Does rank 1-3 beat rank 6-7 net (is the N=7 cliff right)? | `decision_report.py` rank cut + `rank_gradient.py` | D35 repair + authoring the stage0 predictions dump (gap #1) + D11 restore |
| 6 | How often does the lockout punish trailing WINS (flip STOP_CLASSIFY_V2)? | `server_stop_kind` journal fields (always-on since c26 T6) | one Jetson read |
| 7 | Is the funding z baseline honest (re-arm the crowding tilt)? | `scripts/funding_drift_audit.py` (R2C-06) + FUNDING_Z_TIME_THINNING | Jetson run |
| 8 | Is q10 coverage actually 10%±3pp on holdout (keep or retire the tail veto)? | `scripts/reliability_report.py` after LGB_REFIT_FULL refit | Chain-2 retrain event |
| 9 | Do the two books stack on one factor (activate GATE-1's cross-book clamp)? | `account_risk` registry (base_loop.py:954-979) + gate1 report | Jetson cycles |
| 10 | Is the meta gate's p monotone vs outcomes once honestly calibrated (restore its veto authority; unlock edge-Kelly)? | `decision_report.py` conviction calibration + `meta_meta.json` diagnostics | CALIBRATION_V2 + META_OOF_PRED + replay-parity flips (D13 isotonic fix first) |

**The single highest-leverage act for this entire audit is running the Chain-0 Jetson runbook
(`03_jetson_runbook.md`).** Every instrument above is shipped; none has produced a number. One
execution converts roughly fifteen of this ledger's verdicts from structural inference into
measurement — it dominates every other action in expected information per hour.

*— end of analysis; nothing in the tree was changed by this audit.*
