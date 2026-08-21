# KILL LIST — trader

**Purpose:** the ONE canonical do-not-rebuild list for this system. Every future research or build
agent MUST check this file before proposing a strategy/overlay, short-side mechanism, options play,
feature/indicator, data source, execution/cost-model idea, portfolio-construction technique, or
validation method. It consolidates — with original source tags preserved — every KILLED / REJECTED /
refuted / do-not-build verdict scattered across the wave2–9 memory archives, the 2026-07-01/02 reviews,
and the 2026-07 econ/Nobel literature research. Per-wave memory files remain the detailed record of
*why*; this file is the fast pre-flight check. **An item leaves this list only by explicit owner
decision** — a compelling new paper or a re-reading of an old one is grounds to *ask*, not to quietly
drop an entry.

**Last updated:** 2026-07-21

**Source key:**
- `wave-2` … `wave-7` — the wave{2..7} memory archives (`~/.claude/projects/-Users-kywwilson-Desktop-Projects-trader/memory/wave{2-roadmap,3-selection-timing,4-patterns-ta-leading,5-conviction-shorts-options,6-integrity-cost-validation,7-execution-shorts-options-carry}.md`), each wave's own "RED-TEAM KILLS" / "KILLED" / "Research-REJECTED" section. wave-8 (activation) and wave-9 (money-alpha) memories were also checked — they added no new kills, only referenced/reinforced older ones.
- `rev-07-01` — `review-2026-07-01-full-system.md` memory (six-agent full-system review; ML-verdict "cut" recommendations, not adversarially red-teamed like the wave kills — flagged as pending below).
- `rev-07-02` — `review-2026-07-02-indicators-decision.md` memory (indicator/decision-algorithm review; has its own explicit "KILL (evidence)" list).
- `econ-07` — `research/econ_research_2026-07.json`, the `killed_overlaps` array.
- `nobel-07` — `research/nobel_modern_research_2026-07.md`, Section 1 & 3 SKIP entries.
- `research/module_review_2026-07.json` was checked and is **NOT** a source here — it is a code-defect
  (P0–P3 bug) review with no research kill/reject entries; its 90-item owner decision queue is a
  separate artifact (render with `/decision-queue`).

---

## Strategies / overlays

- **VRP overlay** (selling variance risk premium) — sign-flipped post-2016, t=-2.84 — [wave-3, econ-07]
- **Hard EOD reversal rules** — extreme-decile spread misread as winner's magnitude — [wave-3, rev-07-02]
- **VIX/VIX3M backwardation overnight hard gate** — deep bucket is actually positive — [wave-3]
- **Inverse-ETF hedge sleeve** — rejected, no detailed reason recorded — [wave-2]
- **SGOV idle-cash parking** — rejected, no detailed reason recorded — [wave-2]
- **Crypto cross-pair cointegration / pairs trading** — unstable, too few correlated majors — [econ-07, nobel-07]
- **Cross-sectional crypto carry rank** — belongs to funding-harvest trade Alpaca spot can't do — [wave-7, econ-07]
- **Spot-only basis-aware timing tilt** — duplicate of funding signal, zero incremental EV — [wave-7, econ-07]

## Short-side

- **Anomaly-short decile book** — 162 anomalies net ≈0 after retail costs — [wave-5]
- **SPY<200d short-activation gate** — contradicts SYY, short alpha follows froth not bears — [wave-5]
- **Sign-flipped long meta reused for shorts** — mis-calibrated by construction — [wave-5]
- **Short overnight sleeve v1** — ETB→HTB transitions plus HTB fees — [wave-5]
- **Broker ETB flags as training features** — PIT violation — [wave-5]
- **Froth-regime short-activation gate** — SYY is overnight/monthly/small-cap, horizon mismatch — [wave-7]
- **SSR SHORT_SPREAD_MULT** — inverts its own citation — 2008 ban, not Rule 201 — [wave-7]
- **Coinbase-Binance premium as short gauge** — signed backwards, it's bullish accumulation — [wave-7]
- **Aggregate short-interest index** — superseded by per-name FINRA SVR_21/SVR_Z — [wave-4]
- **Attention-factor market-neutral stat-arb construction** — needs long-short, system is long-only — [nobel-07]

## Options

- **Naked/single-leg options + weeklies** — Alpaca bars naked options anyway; VRP bleed — [wave-5]
- **Immediate OPRA subscription** — $99/mo ≈119bps/yr; earlier 12bps estimate was a 10x error — [wave-5]
- **Earnings straddle-avoidance / long straddles** — strawman, already rejected; avoids VRP bleed — [wave-5, wave-7]

## Features / indicators

- **Post-earnings-announcement drift (PEAD/SUE)** — dead post-2006 in large caps — [wave-2, econ-07, nobel-07]
- **52-week-high anchoring** — t=0.43 at horizon, HXZ replication found no effect — [wave-3]
- **ROD_Ret hard winner-leg rule** — refuted, effect ≈0; soft feature survived — [wave-3]
- **Heston-Sadka/KLN broader seasonality feature family** — 0.2-0.3bps/hr vs 100-200bps label noise — [wave-3]
- **HY-OAS credit overlay** — red-team rejected, no detailed reason recorded — [wave-4]
- **HYG/LQD credit-tape overlay** — red-team rejected, no detailed reason recorded — [wave-4]
- **Jobless-claims momentum** — red-team rejected, no detailed reason recorded — [wave-4]
- **SOFR-IORB funding-stress gate** — "net liquidity" salvage attempt, lowest confidence tier — [wave-4]
- **Cross-industry lead-lag features (equities)** — red-team rejected, no detailed reason recorded — [wave-4]
- **Foreign-signal radar** — red-team rejected, no detailed reason recorded — [wave-4]
- **Round-number / reference-level microstructure** — red-team rejected, no detailed reason recorded — [wave-4]
- **CNN1D-on-images challenger head** (JKX chart-pattern CNN) — compute + transfer evidence failed — [wave-4]
- **New TA oscillators** (any additional beyond current set) — A-minus grade negative evidence post-cost — [rev-07-02]
- **VWAP-deviation strategy** — grade D evidence — [rev-07-02]
- **ROC ≡ Return_12h, MACDs, STOCHd** — bit-identical/exact-linear duplicate columns, double-counted — [rev-07-02]
- **Month_sin/cos, Turn_of_Month seasonality** — post-2015 decay, dropped from leaner preset — [rev-07-02]
- **OBV** — redundant vs other volume features, cut recommended (pending) — [rev-07-01]
- **Pseudo-CAPE** (SPY P/E×1.6 proxy) — fake data driving a real haircut — [rev-07-01, econ-07, nobel-07]
- **HMM regime layer** — redundant regime signal, cut recommended (pending) — [rev-07-01]
- **Analyst-revision / investor-attention momentum** — free/proxy data too slow, new dependency — [wave-2, econ-07]
- **Intraday first-half-hour momentum** — index-level 30-min effect, weak at hourly single-name — [econ-07, nobel-07]

## Data / feeds

- **Order-book imbalance filters** — infeasible at retail execution latency — [wave-2]
- **Cboe options put/call O/S ratio** — collect-forward only, no historical backfill — [wave-4]
- **On-chain flow features** — new unreplicated data dependency, arXiv-only evidence — [econ-07]
- **Auction-imbalance / exchange-netflow features** — paid data — [rev-07-02]

## Execution / cost model

- **Honest-OHLC passive-fill simulator + "EDGE-on-hourly inflation fix"** — cited medians were wrong ~70-95x — [wave-7]
- **Adverse-selection contrarian-posting guard** — net-negative even with maker rebate — [wave-7]

## Portfolio construction / risk sizing

- **HRP/ERC portfolio construction** — not enough names (N≤10) for stable weights — [wave-2]
- **Ledoit-Wolf shrinkage / mean-variance cross-asset optimizer** — system uses a risk cap instead — [wave-2, wave-6, econ-07]
- **Marchenko-Pastur denoising** — denoises a target that doesn't exist here — [wave-6]
- **Strategy-level vol-targeting / Moreira-Muir vol-timing-as-alpha** — constant scalar mathematically invisible per-ticker — [wave-6, econ-07]

## Validation / method

- **Realized-shortfall feedback re-pricing offline cost** — phantom EDGE, empty journals, look-ahead — [wave-6]
- **Sequential bootstrap for LGB bagging** — substitute not complement for weighting — [wave-6]
- **"Halve Jetson inference via pruned feature core"** cost claim — LGB inference is column-count-independent — [wave-6]
- **Conformal abstention (CQR/ACI)** — confidence 2/5, red-team rejected — [wave-4]
- **Using shadow DM-HLN to gate POLICY changes** — category error, it tests forecast errors — [wave-5]
- **Shipping concentration/sizing changes on literature priors alone** — must measure in-house first — [wave-5]
- **GMM (Hansen 1982) macro estimation** — redundant, purged-WF+DSR already covers this — [nobel-07]

---

## Commonly confused survivors

Items adjacent to a kill that are themselves alive — do not over-kill these when you see the
neighboring entry above.

1. **Funding-rate features** (`Funding_Rate_Ann/Z/Chg_24h`, `CS_Rank_Funding_Z`) SURVIVE — only the
   carry-*harvest* trade, the cross-sectional carry *rank*, and the basis-timing *tilt* built on top of
   them are killed (wave-7).
2. **Basis-archive features** (`Basis_Bps/Z/Chg_24h`, `Basis_minus_Funding`, `basis_archive.py`)
   SHIPPED (wave-7 Tier-2) — the killed items are the carry *sleeve* and the basis-aware *timing tilt*,
   not the underlying features.
3. **Squeeze features** (`Funding_x_OI`, `Squeeze_Setup`, `squeeze_features.py`) SHIPPED (wave-7) —
   reuse funding+OI inputs but are not the killed carry trade.
4. **JKX distillation features** (`Pos_Range`/`MidRange_Gap`, `MA_Dist_*`) SHIPPED (wave-4) — only the
   CNN1D-on-images model architecture that inspired them was killed, not the hand-crafted features.
5. **`Same_Hour_Mean_40d`** (HKS same-hour periodicity) SHIPPED — and the broader Heston-Sadka/KLN
   seasonality feature *family* was ALSO evaluated and killed in the very same wave (wave-3, not a
   later one) as too small vs label noise. Same literature, two different implementations, two
   different verdicts — don't let one justify reviving or removing the other.
6. **`ROD_Ret`** as a soft feature + the loser-bounce sleeve exclusion SHIPPED (wave-3) — only the hard
   winner-leg *rule* version was refuted (effect ≈0).
7. **VIX-conditioned residual reversal** (`RR_5`/`RR_21`) SHIPPED (wave-4) — unconditional/multi-year
   reversal (De Bondt-Thaler horizon) was never built and is wrong-horizon for this book; that is not
   the same as RR_5/21 being killed.
8. **HAR-RV volatility forecasting** (`volatility.py`, `HAR_VOL_ENABLED`) is live and first-class —
   distinct from "vol-timing as return alpha" and "strategy-level vol-targeting," both killed
   (wave-6, econ-07). Forecasting vol for sizing is not the same as betting on vol as an alpha source.
9. **`options_overlay.py`** (BSM pricer + defined-risk verticals) is LIVE measurement code that
   currently self-gates to NO_GO — that is a data-driven negative verdict from active instrumentation,
   not a kill. Distinct from the killed naked-options/weeklies and immediate-OPRA-subscription items.
10. **"BTC-lagged spillover for alts"** is flagged kill-*adjacent* (rev-07-02) to the killed equity
    cross-industry lead-lag (wave-4) but explicitly needs USER SIGN-OFF — treat as undecided, not
    killed, unlike crypto pairs-trading/cointegration which IS killed (econ-07).
11. The killed **52-week-high anchoring** feature (wave-3, a proposed CS indicator) is unrelated to the
    live **"winner's-curse" 20-hour SMA20+2×ATR anchor gate** flagged as an open conflict in rev-07-02 —
    different anchors, different status; do not conflate the two.
12. The **direct SPY short hedge via Alpaca ETB** ($0 borrow) SURVIVES — it is step 2 of the
    rev-07-01 low-beta roadmap (beta ledger → SPY hedge → ETB CS long-short) and was never killed.
    The killed wave-2 item is the **inverse-ETF product sleeve** (SH/SDS-style instruments with
    their daily-rebalance decay), not hedging itself.

---

## PENDING OWNER ASKS (2026-08 campaign — asks only; NOTHING has left this list)

Per this file's own rule, entries leave only by explicit owner decision. The 2026-08 campaign's
web research produced four formal asks (evidence in `research/campaign_2026-08/02_research.md`
and the campaign report; activation context in `03_jetson_runbook.md` Phase 5):

1. **"Honest-OHLC passive-fill simulator + EDGE-on-hourly inflation fix" [wave-7] (line ~90):**
   ask to re-open ONLY the stamp-from-minute-bars half. Two independent agents traced the cited
   "~70–95x" discrepancy to a probable percent-vs-fraction (100x) slip in the wave-7 *finding*,
   not in the wave-6 code; the honest quantified hourly-EDGE degradation is ~5–10x on tight-spread
   names (noise floor σ·n^-¼). The passive-fill SIMULATOR half stays dead; fees.py's "do NOT fix
   by simulating passive fills" note is untouched. Code for the minute-bar/quote-first stamps is
   built DARK behind `TRADER_STOCK_MINUTE_EDGE` / `TRADER_CRYPTO_SPREAD_STAMP` — nothing activates
   without this ruling plus the census evidence.
2. **"SPY<200d short-activation gate" [wave-5] (line ~41):** ask to bless the distinction between
   that killed short-ALPHA trigger and the B20 trend-conditional beta HEDGE using the same 200-day
   trigger for beta REMOVAL (Goulding-Harvey-Mazzoleni state-conditional beta economics + the
   ledger's own measured trend-conditional beta). If blessed, record as commonly-confused-survivor
   #13 so the hedge trigger is never mistaken for a rebuild of the kill.
3. **Pseudo-CAPE [rev-07-01, econ-07, nobel-07] (line ~76):** enforcement ask, not a challenge —
   rule on the queued code REMOVAL when adjudicating the DERISK_STACK_V2 flip. Research
   clarification: min-aggregation does NOT launder it (a fake signal inside a min() still binds
   whenever it is the minimum). Under the flag it is excluded from composition; the code remains
   until this ruling.
4. **"BTC-lagged spillover for alts" (commonly-confused survivor #10, UNDECIDED):** ask to split
   the ruling three ways — (a) BTC-lagged alt-ALPHA features: research recommends AGAINST
   (transmission completes within one hourly bar post-2022; lag sign regime-dependent);
   (b) the BTC-native crypto_trend sizing GATE: recommend wiring per B23 with the co-fire
   counterfactual acceptance test; (c) restoring the existing CONTEMPORANEOUS BTC context columns
   the production preset silently drops (D31's false auto-include comment): a preset repair, not
   a lag build — still requires this sign-off because of the survivor-#10 boundary.
