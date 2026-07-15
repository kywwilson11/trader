# Nobel-Laureate & Modern Research Compilation — 2026-07

Compiler synthesis of 45 findings from three searchers, deduped, cross-checked, kill-list-screened,
and ranked by **evidence grade × fit-to-this-system**. This system = LSTM+LightGBM on hourly bars,
6 large-cap cryptos + ~45 mostly-liquid US stocks, **LONG-ONLY**, retail costs (~50–60bps rt crypto,
~15–25bps stocks), Jetson 8 GB prod, honest validation (purged CV, Deflated Sharpe DSR_MIN=0.60,
CSCV-PBO, shadow A/B).

## Method notes

- **Dedup:** 45 → 44 unique. McLean & Pontiff (2016) appeared twice (identical paper, two URLs) —
  **merged** into one entry.
- **Verification (WebFetch spot-checks of the off-looking / load-bearing A/B claims):**
  - **Chen & Welch (2026), "What Useful Alphas?", arXiv:2607.06502** — future-dated, so verified.
    **CONFIRMED real** (submitted 2026-07-07, Andrew Y. Chen & Ivo Welch). The finding was
    *conservative*: paper reports median anomaly return falling from **48bp/mo (through 2005) → 19bp
    (post-2005 all stocks) → 7bp (post-2005, non-micro-cap)**, "eliminated even by modest allowances
    for luck or transaction costs." Grade B upheld (unrefereed working paper, but authoritative
    authors and directly on-point).
  - **Nagel (2025), NBER WP 34104** — **CONFIRMED real** (Stefan Nagel, Aug 2025); claim that RFF
    complexity collapses to a recency-weighted, vol-timed momentum average is accurately stated.
    Grade B upheld.
  - **Frazzini-Israel-Moskowitz (SSRN 2294498)** — paywalled (HTTP 403); canonical AQR paper, claim
    left as-is at grade B.
  - Barroso–Santa-Clara (2015) and Chen–Velikov (2023) headline figures ("~doubles Sharpe";
    "~4bp/mo net, best combos ~20bp") are canonical and consistent with the literature — accepted at
    grade A without re-fetch.
- **Kill-list screen:** One finding violated the kill list — Garfinkel et al. PEAD/SUE was
  originally graded ACTIONABLE, but PEAD is wave-2 Research-REJECTED ("dead post-2006 large caps");
  flipped to **SKIP** in pre-commit review (Section 3 #7), matching the
  `econ_research_2026-07.json` killed_overlaps entry. Every other SKIP item (Black-Scholes/
  options, Engle-Granger pairs, Shiller CAPE, Gao intraday-momentum, Epstein market-neutral statarb,
  overnight-anomaly expansion) is already correctly self-flagged as non-transferable; the SKIPs are
  retained only as one-line citations / guardrails.
- **Grading:** No material mis-grades found. Kelly-Malamud-Zhou (A, "complexity is fine") is presented
  as a **matched pair** with its direct rebuttal Nagel (B) — for *this* system the caution dominates.

---

## Section 1 — Laureate Foundations
*Retrospective validation of the existing architecture. These are theory, not new builds — they
confirm which existing modules are well-grounded, with one cheap actionable extension (Markowitz).*

| Rank | Source (Nobel/laureate) | Grade | Fit verdict |
|---|---|---|---|
| 1 | **Engle (1982) ARCH/GARCH** — Nobel 2003 | A | **USE (already built).** Volatility clusters and is forecastable from past squared returns → the exact grounding for realized-vol sizing + `HAR_VOL_ENABLED`. Keep vol-target sizing **first-class, not optional**. A light GARCH(1,1)/HAR-RV hourly proxy is within the evidence base. |
| 2 | **Markowitz (1952) Portfolio Selection** — Nobel 1990 | A | **USE — cheap ACTIONABLE extension.** Portfolio variance is driven by pairwise covariances, not just per-asset variance. Implies `MAX_BOOK_RISK_PCT` / per-name sizing should be **covariance-aware across the combined 6-crypto + 45-stock book**, not just aggregate-capped. The **beta ledger already measures the needed correlations** → this is a measurement extension, not new infra. |
| 3 | **Sharpe (1964) CAPM** — Nobel 1990 | A | **USE (already implemented).** `beta_ledger.py` realized SPY/BTC betas + the Sharpe/DSR promotion gate are direct descendants. Keep the beta ledger as the primary "is this book secretly just beta?" check. (Fama-French 1992 already showed the raw linear beta-return prediction is weak — don't lean on it as a signal.) |
| 4 | **Asness, Moskowitz & Pedersen (2013) "Value & Momentum Everywhere"** — laureate-class | B | **USE (retrospective).** Value/momentum premia are correlated across 8 markets/asset classes → strong evidence they're not pure data-mining. Supports keeping momentum features **enabled jointly** across crypto+stock books rather than as fully independent per-book hyperparameter searches. Cross-sectional panel ranks already cover this. |
| 5 | **Kahneman & Tversky (1979) Prospect Theory / Shefrin-Statman (1985) disposition effect** — Nobel 2002 | B | **USE (design rationale).** Loss aversion → disposition effect (hold losers, sell winners). This is *why* the mechanical ATR stop/exit stack (`policy_exits.py`) must override discretion. Confirms the **no-override design is correct** — never add "let it come back" loosened-trailing-stop logic. No new code. |
| 6 | **Fama & French (1993) 3-factor / Fama (1970) EMH** — Nobel 2013 | B | **PARTIAL / SKIP new build.** Cross-sectional panel ranks already capture size/value-style structure. The EMH half is the strongest argument for *why this book's edge is necessarily thin and cost-sensitive* → validates prioritizing cost-aware gating over chasing new anomaly features. Do **not** add book-to-market factor portfolios at hourly frequency (wrong horizon). |
| 7 | **Lo & MacKinlay (1988) variance-ratio / Lo (2004) Adaptive Markets** — laureate-class | C | **USE (architecture justification).** Short-horizon returns show significant serial correlation (violates random walk); AMH says edges decay and rotate. This is the theoretical case for the **champion/challenger shadow A/B + continuous re-validation** already built. Not itself an hourly rule. |

**SKIP but cited (correctly self-flagged as wrong-horizon or kill-list-adjacent):**

- **De Bondt & Thaler (1985) overreaction/reversal** — Nobel 2017 — C. Multi-*year* reversal; wrong
  horizon. Useful one-liner: "limits to arbitrage" justifies why a small, cost-aware retail book can
  still harvest a compressed edge institutions have arbitraged down. No code.
- **Engle & Granger (1987) cointegration** — Nobel 2003 — C. **SKIP pairs trading** (6 cryptos too
  small/correlated for a robust cointegrating vector; equity cross-industry lead-lag is on the kill
  list). The Granger-causality half is exactly what shipped `indicator_leadlag.py` already does —
  retrospective validation of existing tooling only.
- **Duffie (2010) slow-moving capital** — laureate-class — C. Institutional block-trade scale.
  One-line citation for the cost-aware gating philosophy: frictions create the dislocations a nimble
  retail account can sometimes capture faster than slow institutional capital.
- **Cochrane (2011) "Discount Rates"** — laureate-class — C. Time-varying risk premia, but at
  quarterly-to-multiyear horizons on dividend yields. The regime-diagnostics layer already embodies
  the "time-varying premia" idea. No new work.
- **Shiller (1981) excess volatility / CAPE** — Nobel 2013 — D. **SKIP** — multi-year valuation mean
  reversion, same wrong-horizon problem as the already-killed 52-week-high anchoring.
- **Hansen (1982) GMM** — Nobel 2013 — D. **SKIP** — low-frequency macro asset-pricing estimation
  methodology; `validation.py` (purged WF + DSR) already fills the "honest estimation under uncertain
  distribution" role here.
- **Black-Scholes-Merton (1973)** — Nobel 1997 — D. **SKIP explicitly** — options overlays are on the
  kill list; no listed-options data harvested.
- **Mokyr, Aghion & Howitt (2025 Prize) innovation-driven growth** — D. **SKIP** — decade-plus
  economy-wide macro growth theory, zero mechanical link to an hourly signal. Confirms **there is no
  new finance-relevant Nobel to react to as of 2026-07**; the 2026 prize is not yet announced.

---

## Section 2 — Modern Research
*ML / validation / integrity / cost literature. This is the **financial-soundness** tier — the
highest-priority category per the user's stated ordering (after Jetson memory/perf). Ranked
evidence × fit.*

| Rank | Source | Grade | Fit verdict |
|---|---|---|---|
| 1 | **Chen & Velikov (2023) "Zeroing In on the Expected Returns of Anomalies," JFQA** | A | **ACTIONABLE — cost benchmark.** Net of realistic effective spreads + post-pub decay, the average long-short anomaly nets **~4bp/mo** (best ~10, best combos ~20), because anomaly books overweight wide-spread names and turn over ~40%/mo. Judge any new signal against a **~4–20bp/mo net hurdle**, not gross backtest; keep turnover low — the stock book's 15–25bps cost is already the same order as the whole edge. |
| 2 | **Lalwani, Meshram & Jindal (2024) "…Role of Research Design Choices," Eur. Fin. Mgmt** | A | **ACTIONABLE — non-standard errors.** Across 5,376 ML-strategy portfolios over 8 defensible design choices, **variation from design choices ran up to 5× the standard error** of any single estimate. A single config passing `DSR_MIN=0.60` is *one draw*. Before promotion, **perturb window length / universe filters** and require the DSR pass to survive — robustness to design choice, not just to the one search run. |
| 3 | **Jensen, Kelly & Pedersen (2023) "Is There a Replication Crisis in Finance?" J. Finance** | A | **USE — counterweight.** Bayesian replication across 153 characteristics / 13 themes / 93 countries: most factors replicate, cluster into economic themes, and hold OOS internationally. Mild pushback on blanket "everything is data-mined." Supports keeping a **moderate, theme-diversified factor set** (as cross-sectional panel ranks already do) rather than collapsing to 1–2 factors — though effect sizes at this cost profile stay small (per Chen-Velikov). |
| 4 | **Kelly, Malamud & Zhou (2024) "The Virtue of Complexity in Return Prediction," J. Finance** | A | **USE (paired with Nagel below).** Ridgeless P≫T models beat simple/OLS OOS, mainly by de-risking ahead of recessions. → Do **not** shrink the LSTM+LightGBM ensemble on parameter-count grounds alone; complexity per se isn't the risk. **But** their setup is *monthly market-timing on a few macro predictors*, not hourly cross-sectional selection — it does **not** license adding features without the cost/DSR gates. |
| 5 | **McLean & Pontiff (2016) "Does Academic Research Destroy Stock Return Predictability?" J. Finance** *(merged dupe)* | A | **USE — expectation-setter.** Anomaly returns are ~26% lower OOS pre-publication, ~58% lower post-publication. **Haircut any newly-published academic signal by roughly half before it is allowed to size** — and per Chen-Velikov/Chen-Welch the non-micro-cap/post-2005 decay is likely *worse* than this 2016 estimate. |
| 6 | **Chen & Welch (2026) "What Useful Alphas?" arXiv:2607.06502** *(WebFetch-verified)* | B | **USE — reinforces the gated-beta verdict.** Post-2005, non-micro-cap published anomalies net **~7bp/mo — "useless… eliminated by modest allowances for luck or transaction costs."** Directly reinforces the prior red-team finding that the ~45-stock book is **gated-beta, not alpha**. Treat any new academic cross-sectional/TA feature for the stock leg as very likely **net-zero-to-negative after this system's own costs** unless proven on *this system's* purged/cost-aware validation. |
| 7 | **Nagel (2025) "Seemingly Virtuous Complexity in Return Prediction," NBER 34104** *(WebFetch-verified)* | B | **ACTIONABLE — cheap diagnostic.** RFF P≫T models mechanically collapse to a recency-weighted, vol-timed momentum average, and "discover" reversal on synthetic reversal data — apparent complexity gains can be a window artifact. **Before trusting any DSR-passing config with many features and a short retrain window, benchmark it against a naive recency-momentum baseline on the same window.** If it tracks the naive rule, the ensemble is being credited for what a one-liner captures. |
| 8 | **Arian, Norouzi M. & Seco (2024) "Backtest Overfitting in the ML Era," Expert Systems w/ Apps** | B | **ACTIONABLE — activate CSCV-PBO.** In a synthetic ground-truth environment, **CPCV shows markedly lower Probability of Backtest Overfitting and better DSR calibration** than K-Fold/Purged-KFold/walk-forward. If `validation.py`'s CSCV-PBO isn't already the *primary* promotion gate, make it one — walk-forward alone is the weaker overfitting detector. |
| 9 | **Meyer, Barziy & Joubert (2023) "Meta-Labeling: Calibration and Position Sizing," JFDS** | B | **ACTIONABLE — reinforces wave-9.** Across 6 sizing algorithms, **probability calibration** (not classifier complexity) is what converts meta-labeling into realized Sharpe/drawdown gains for fixed/rule-based sizers. **Prioritize calibration diagnostics (reliability curves, Brier) over adding meta-label input features.** Directly backs the existing wave-9 calibration-reliability gate work. |
| 10 | **Lopez-Lira, Tang & Zhu (2025) "Can ChatGPT Forecast Stock Price Movements?" arXiv:2304.07619** | B | **USE (LLM sizing caution).** LLM-headline-sentiment Sharpe decayed **6.54 (2021Q4) → 3.68 → 2.33 → 1.22 (Jan–May 2024)** as the technique spread. By 2026 the realistic edge is likely smaller than the 1.22 endpoint. **Keep the LLM/sentiment gate (`llm_analyst.py`) as a veto/sizing overlay, not a primary alpha source.** Do not up-size it on the strength of the 2023 paper. |
| 11 | **Gao, Jiang & Yan (2025) "Detecting Lookahead Bias in LLM Forecasts," arXiv:2512.23847** | C | **ACTIONABLE — cheap guardrail.** Lookahead Propensity (LAP) metric spikes for events inside the model's training window and "collapses to ~zero" right after the cutoff; LLM forecast gains vanish once contamination is controlled. **Before trusting any backtest of the LLM gate on historical news, verify news dates are strictly after that specific provider/version's training cutoff** — add a LAP-style check alongside existing PIT discipline. (Grade C: brand-new unreviewed preprint, but the mechanism is straightforward to verify.) |
| 12 | **Cakici et al. (2024) "ML and the Cross-Section of Crypto Returns," IRFA** | B | **USE (crypto caution).** ML crypto gains are real gross, and survive costs despite high turnover — **but the abnormal returns are concentrated in the long leg and depend on extreme returns of small, illiquid, volatile coins.** Not the large liquid majors this book trades → expect the realized edge on BTC/ETH-tier names to be **far weaker** than the pooled result. Consistent with "beta not alpha" for the crypto leg. |
| 13 | **Mercik et al. (2025) "Cross-Sectional Interactions in Crypto Returns," IRFA** | B | **USE (same crypto caution).** Return interactions from liquidity × risk × past-return, demonstrated on 500+ coins. Low transferability to 6 majors; a reminder not to expect published crypto-anomaly magnitudes to survive down-selection to a narrow, liquid universe. |
| 14 | **Epstein, Wang, Choi & Pelger (2025) "Attention Factors for Statistical Arbitrage," ICAIF/arXiv:2510.11616** | C | **SKIP (does not transfer).** Gross Sharpe >4 / net 2.3 comes from a **market-neutral long-short residual** design this system explicitly cannot run (long-only, no shorting). The transferable kernel (conditioning factor exposure on characteristic embeddings via attention) is a feature-engineering idea, low priority given the prior kill of cross-industry lead-lag work. |

---

## Section 3 — Proven Strategies
*Trend / momentum / quality — the **trading-strategy** tier (lowest of the four priorities, but where
the concrete new levers live). Ranked evidence × fit.*

| Rank | Source | Grade | Fit verdict |
|---|---|---|---|
| 1 | **Barroso & Santa-Clara (2015) "Momentum has its moments," JFE** | A | **ACTIONABLE — top strategy lever.** Scaling a momentum book by the inverse of its **own** trailing realized variance **~doubles Sharpe and virtually eliminates momentum crashes**, robust across subsamples/countries. **Throttle position conviction by the recent volatility of the model's own signal/momentum leg** — distinct from the price-vol sizing already built. Cheap, well-replicated. |
| 2 | **Asness, Frazzini & Pedersen (2019) "Quality Minus Junk," Rev. Acct. Studies** | A | **ACTIONABLE — long-only compatible.** A quality tilt (profitability, growth, safety, payout) earns risk-adjusted returns across US + 24 countries and subsumes the low-beta anomaly. A **cheap, low-turnover, no-shorting quality screen/tilt on the ~45-stock universe** is a natural complement to the momentum-driven LSTM signal — no change to the long-only constraint. |
| 3 | **Daniel & Moskowitz (2016) "Momentum Crashes," JFE** | A | **ACTIONABLE — regime guard.** Momentum crashes are forecastable and concentrated in post-crash "panic" rebounds. The short-call mechanism doesn't apply to a long-only book, but the lesson does: **long momentum is fragile right after a sharp drawdown/rebound** → add a **post-crash cooling-off / regime check before re-loading long momentum** in either book. |
| 4 | **Hurst, Ooi & Pedersen (2017) "Century of Evidence" / Moskowitz, Ooi & Pedersen (2012) TSMOM, JFE** | A | **USE — sets the horizon.** Time-series momentum (1/3/12-month blend) delivers Sharpe ~0.4–0.5 across 58–67 markets over 100+ years, strong in 8/10 largest 60/40 drawdowns. Confirms **medium-horizon (weeks-to-months) trend is the durable speed, not daily/hourly.** Informs LSTM feature horizon; it's a multi-asset long-short overlay, so don't transplant the program directly. |
| 5 | **Bogousslavsky et al. / "TSMOM & reversal from realized semivariance" (J. Empirical Finance, 2023)** | B | **ACTIONABLE — pairs with #4.** Very short-horizon (daily/weekly) momentum weakens once microstructure frictions (bid-ask bounce, dealer inventory) are netted out; the 3–12-month signal is stable. **Hourly bar-to-bar "momentum" features are likely bounce/noise → compute momentum features on daily+ aggregation** even though execution stays hourly. Reinforces cost-aware gating. |
| 6 | **Cederburg, O'Doherty, Wang & Yan (2020) "Performance of volatility-managed portfolios," JFE** | B | **USE (caution on vol-targeting).** Real-time-implementable vol-managed portfolios **do not reliably beat the unmanaged originals OOS** (8/103 significant). Naive 1/realized-vol scaling is **not** a free Sharpe lift. Any vol-targeting overlay must be validated through this system's purged-CV/DSR pipeline, not trusted from in-sample spanning-regression alphas. (Tempers, does not negate, #1 — Barroso scales by the signal's *own* vol, a narrower and better-replicated claim.) |
| 7 | **Garfinkel, Hribar & Hsiao (2024) PEAD (UCLA Anderson Review)** | B | **SKIP — kill list.** SUE-decile long-short ~5%/3mo, but the market-wide t-stat drops 2.18 → 1.43 once microcaps are excluded — drift is concentrated in small names. PEAD is **wave-2 Research-REJECTED** ("dead post-2006 large caps"), and this paper's own microcap-exclusion decay confirms that verdict for the ~45-name mostly mid/large-cap universe. Retained as a citation only — do **not** build a SUE/PEAD feature. |
| 8 | **Morgan Stanley IM (2024) "Momentum Ruled in 2024, But Reversal Likely in 2025"** | B | **USE — current-decade case for a quality veto.** Momentum was the top US factor in 2023–24 (+44% in 2024, ~2σ, AI/growth mega-cap-driven), then reversed sharply in early 2025. **Pair the momentum model with a quality/defensive veto or exposure cap** to avoid concentrating the long book in high-beta, low-quality names during speculative run-ups. (Complements #2.) |
| 9 | **Zarattini, Pagani & Barbon (2025) "Catching Crypto Trends," SSRN/CHF** | C | **USE (benchmark/challenger).** Donchian-channel trend ensemble with vol-based sizing, rotated across top-20 liquid coins, net Sharpe >1.5 / 10.8% annual alpha vs BTC, explicitly turnover-controlled. The **most directly comparable-scale finding** — a long-only-capable, cost-aware trend strategy on *liquid majors* (unlike the ML-crypto papers whose edge lives in illiquid names). Worth using as an **external benchmark/challenger** for the crypto book's trend features. Unreviewed practitioner paper. |
| 10 | **Crypto momentum studies (2023–25) — TS vs CS comparison** | B | **USE — favor cross-sectional.** Crypto momentum persists post-2022 but with **short 1–4-week formation windows** (vs equity 6–12mo); pure TS momentum went negative in choppy 2022–23 while cross-sectional (relative-strength) limited losses. With only 6 coins, **favor the already-built cross-sectional panel-rank approach over pure directional trend, with short (days-weeks) formation windows** matching hourly retrain cadence. |
| 11 | **"Crypto Risk-Managed Momentum Strategies" (2025) + critique** | C | **USE (skeptical).** Vol-scaling weekly CS crypto momentum lifted Sharpe 1.12→1.42, but via *added return* not crash mitigation (crypto lacks the equity momentum-crash pattern); a companion paper disputes clean transfer of the Barroso mechanism. Studied on many liquid-but-volatile alts → **expect a much smaller edge on 6 majors.** Supports using the crypto momentum signal's own realized vol as a **low-weight** sizing input, re-verified OOS. |
| 12 | **MacLean, Thorp & Ziemba (2010/11) "Kelly Capital Growth Criterion"** | B | **USE — validates existing config, no change.** Full Kelly's growth curve is flat near the optimum while drawdown rises sharply; practitioners use ¼–½ Kelly (half-Kelly ≈ 8% less terminal growth for ~half the max drawdown). **Directly validates `KELLY_CAP=0.25`** — already in the safe range. No change. |
| 13 | **Frazzini, Israel & Moskowitz "Trading Costs of Asset Pricing Anomalies," AQR WP** *(SSRN paywalled — canonical)* | B | **USE (cuts both ways).** Real institutional algo-execution costs are ~an order of magnitude smaller than academic assumptions → high-turnover momentum isn't fatal in principle. **But** this system's retail costs (50–60bps crypto rt, 15–25bps stocks) are far above those institutional costs → **keep sizing on this system's own `fees.py`/`liquidity.py` model**, not these lower assumptions. |
| 14 | **Gao, Han, Li & Zhou (2018) "Market Intraday Momentum," JFE** | B | **SKIP.** First-half-hour predicts last-half-hour on liquid index ETFs, but OOS R² only ~1.2–3.3% — a 30-min near-zero-cost ETF-arb effect. This system's hourly granularity + 15–25bps costs are too coarse/expensive to extract it. (Not a killed EOD-reversal *rule* — just uneconomic here.) |
| 15 | **Overnight return anomaly (Alpha Architect replication)** | C | **USE (keep behind flag).** The overnight (close-to-open) premium is large gross but its robustness **collapses net of realistic costs.** Consistent with keeping the overnight sleeve behind its default-off/measured flag; re-verify net-of-cost edge on this system's actual 15–25bps model before expanding overnight-holding logic. |

---

## TOP 10 ACTIONABLE FOR THIS SYSTEM
*Ranked by evidence × fit × cheapness-to-ship, weighted toward the user's priority order
(financial soundness › LLM utilization › trading strategy). Each is either a new lever or a
concrete gate/discipline — none violate the kill list; none are pure "already-built" restatements.*

1. **Signal-vol scaling of conviction** — throttle position size by the trailing realized volatility of the model's *own* momentum/signal leg (distinct from price-vol sizing already built). *Barroso & Santa-Clara (2015, JFE) — A.*
2. **Bake a net-cost hurdle into every new-signal gate** — require ~4–20bp/month net-of-EDGE-cost, judged on this system's own cost model, before any signal is allowed to size. *Chen & Velikov (2023, JFQA) — A.*
3. **Add a low-turnover long-only quality tilt** (profitability/growth/safety/payout) to the ~45-stock book — long-only-compatible complement to the momentum LSTM, doubles as a defensive veto vs speculative run-ups. *Asness, Frazzini & Pedersen (2019, RAS) — A; Morgan Stanley IM (2024) — B.*
4. **Require promotions to survive design-choice perturbation** — re-run the DSR gate under reasonable window-length / universe-filter variations; a single passing config is one draw from a distribution up to 5× wider than its standard error. *Lalwani, Meshram & Jindal (2024, EFM) — A.*
5. **Make CSCV-PBO the primary promotion gate** (not walk-forward + DSR alone) — it detects backtest overfitting markedly better and calibrates DSR. *Arian, Norouzi M. & Seco (2024, Expert Systems w/ Apps) — B.*
6. **Benchmark every DSR-passing config against a naive recency-momentum baseline** on the same window — if it tracks the one-liner, the ensemble's "edge" is a window artifact. *Nagel (2025, NBER 34104) — B (WebFetch-verified).*
7. **Prioritize meta-label probability calibration over more meta features** — reliability curves + Brier score are what convert meta-labeling into realized Sharpe; reinforces the existing wave-9 calibration gate. *Meyer, Barziy & Joubert (2023, JFDS) — B.*
8. **Add a post-crash/rebound cooling-off regime check** before re-loading long momentum in either book — long momentum is fragile in panic-rebound regimes. *Daniel & Moskowitz (2016, JFE) — A.*
9. **Compute momentum features on daily+ aggregation, not hourly bar-to-bar** — sub-daily "momentum" is largely bid-ask bounce/noise; the durable signal lives at weeks-to-months even though execution stays hourly. *Bogousslavsky et al. (2023, J. Emp. Fin.) — B; Hurst, Ooi & Pedersen (2017) — A.*
10. **Add a lookahead/training-cutoff contamination check for LLM-gate backtests** — verify news dates strictly post-date the model version's training cutoff before trusting any historical LLM-analyst backtest; cheap guardrail on top of existing PIT discipline. *Gao, Jiang & Yan (2025, arXiv:2512.23847) — C.*

*Runner-up (cheap, laureate-grounded): covariance-aware cross-book sizing using the beta ledger's
already-measured pairwise correlations — Markowitz (1952) — A.*

*Deployment note: items 1, 3, 8, 9 are model-facing → challenger → shadow → DM-HLN promotion path.
Items 2, 4, 5, 6, 7, 10 are instrumentation / validation-gate / measurement changes → shippable
directly. All heavy-dep work (retrain, harvest, Stage-0) is Jetson-gated.*
