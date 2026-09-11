# Nobel-Laureate & Modern Research Compilation — 2026-08 (Round 3)

**Produced 2026-08-22.** Successor to `nobel_modern_research_2026-07.md` (round 1, 44 findings) and
complement to `../campaign_2026-08/05_frontier_research.md` (round 2, ML frontier). This round mined
only the GAPS: laureate/nominee-tier economics not yet touched, 2025–26 economic research outside
the ML-architecture ground round 2 closed, and strategy families not yet screened. **Nothing here
re-recommends a prior verdict or a killed item**; every proposal was screened against
`research/KILL_LIST.md` including the PENDING OWNER ASKS appendix.

## Method & verification discipline (read before trusting any number)

Five research lanes ran (laureates; modern econ; strategies; plus two data-feasibility probes and a
top-journal sweep spawned beneath them). **One lane self-reported a mid-run integrity failure**: its
first report presented sub-agent results that had never arrived; it issued a correction separating
verified from unverified claims. This file inherits that discipline:

- **Every repo-code claim below was re-verified in-session against the source by the compiler**
  (file:line cited). None rest on an agent's word.
- Literature is tiered: **[V]** = primary text/DOI/publisher record fetched this session;
  **[E]** = existence + venue confirmed, headline numbers NOT independently verified;
  **[Q]** = quarantined (Section 6) — do not act on.
- Two web-search budgets exhausted mid-run; coverage gaps are declared, not papered over
  (JF/RFS advance lists, JFQA FirstView, Review of Finance: **unswept**).

System constraints applied throughout: hourly bars, LONG-ONLY live, ~6–10 crypto majors +
~46 US names, retail costs (~50–60bps rt crypto assumed, ~15–25bps stocks), free data only,
Jetson 8GB, purged-WF + DSR ≥ 0.60 + CSCV-PBO + challenger→shadow promotion; nothing ships on
literature priors alone.

---

## Section 0 — Verified code findings surfaced by the research

The highest-value output of the round. Each verified by the compiler against source this session.

**D1 — Backtest/live macro stand-down parity gap.** `base_loop.py:2412-2413` calls
`macro_standdown()` and blocks live entries in FOMC/CPI windows; `backtest.py` contains **zero**
references to `macro_calendar` while it *does* mirror `ENTRY_WINDOWS` (backtest.py:239-245,
865-874). The promotion-gate replay therefore validates a policy that trades through windows the
live bots sit out — the exact `strategy_config`-drift class CLAUDE.md warns about. (Related in-code
admission: `strategy_config.py:90` notes backtest reads none of the maker-tactic flags either.)
**Fix class:** mirroring the stand-down into the replay aligns the gate with live reality
(gate-honesty repair, same class the 2026-08 campaign shipped directly).

**D2 — `macro_calendar.py` has no NFP.** Stand-down list = FOMC + CPI only (macro_calendar.py:32-47).
Londono & Samadi (FEDS Note, 2025-10-03) [V]: annualized event premia Sept-2023→Jul-2025 — CPI
45bp (t=10.4), FOMC 49bp (t=4.7), **NFP 53bp (t=8.8)** — NFP the *largest*, same 08:30-ET shock
class as CPI. Knox, Londono, Samadi & Vissing-Jorgensen (Fed WP, SSRN 4773692) [V]: the *ex-ante*
option-implied premium is only **7.35bp per forward period** — far below our round trip ⇒ **there
is no tradeable event tilt, only a stand-down.** Adding the NFP date list (BLS schedule, free,
static, annual refresh) widens live gate behavior ⇒ default-OFF flag or owner sign-off, per
convention. Ships naturally with D1.

**D3 — Vol-target sizing splits daily vol flat across the day.** `volatility.py:227` returns
`sigma_daily / sqrt(BARS_PER_DAY)` ({crypto 24, stock 6.5}, line 155) — no intraday periodicity,
both books. Evidence the shape matters: Dumitru, Hizmeri & Izzeldin, *J. Banking & Finance*
170:107342 (2025-01) [V] — periodicity-filtered HAR cuts OOS forecast loss **6.6–7.3%** at
1-day/1-week and ranks first in the Model Confidence Set; Martins et al. (Örebro WP 14/2025) [V]
— conditioning the diurnal shape on lagged same-hour volume deviation is the best variant;
Yatawara (SSRN 6997366, 2026-06) [V] — *when* variance arrives in the day predicts next-day RV
(~6%/SD), unrecoverable from daily RV alone. **Proposed lever:** per-book diurnal vector `p_h`
(trailing ~60d median hour-h Parkinson share, normalized), `sigma_h = sqrt(rrv_hat·p_h/BARS_PER_DAY)`,
behind default-OFF `HAR_DIURNAL_ENABLED` next to `HAR_VOL_ENABLED`; hourly Parkinson components
already exist (`daily_realized_range`, volatility.py:171). Offline gate: QLIKE of realized |hourly
return| vs flat/adjusted sigma, then `backtest.py --gate` + `--fee-sweep`. Crypto expected to gain
more (24 buckets, US-hours concentration). **Kill-screen: PASSES** — survivor #8 protects
HAR-for-sizing; this must NEVER be motivated by vol-managed-portfolio Sharpe (that is the killed
Moreira-Muir lever). The campaign's "B11 binding: keep log-HAR; no WLS, no HARQ" is untouched —
periodicity is a different axis than quarticity.

**D4 — LLM spend ledger never nets benefit against cost (Grossman-Stiglitz gap).** `llm_eval.py`
reports benefit as `llm_tilt_bps_per_trade` (:908) and cost as `daily_cost_usd` /
`window_journaled_cost_usd` (:901,:904) — different units, never netted; no `benefit_usd` exists
(grep-verified). `llm_client._DAILY_COST_LIMIT = 1.00` (:173) = **146 bps/yr of equity at $25k**
(36.5 at $100k). Grossman-Stiglitz (1980, AER) is the doctrine: paid information survives only
where gross value strictly exceeds cost — and the binding risk here is *negative gross value*
(the gate costing return), not the fee. **Fix:** journals already carry `final_notional` on the
rows llm_eval walks ⇒ add `benefit_usd = Σ tilt·realized·notional`, `net_usd`, break-even
bps/trade and annualized bps-of-equity next to the b2 verdict. ~20 lines pure numpy,
measurement-only ⇒ ships direct.

**D5 — `_DV30` horizon mismatch latent in the dark impact path.** The harvest stamps `_DV30` as
trailing 30d median **daily** dollar volume (harvest_stock_data.py:373-378); the square-root
impact function requires "**PER-BAR** dollar ADV in DOLLARS" and warns the panel_ranks snapshot
under the same name "is NOT this input" (liquidity.py:495-498). Orders fill inside one hourly bar
⇒ a naive enable would understate participation ~6.5× and impact ~√6.5 ≈ 2.5× (stocks; √24 ≈ 4.9×
crypto). The enable-recipe comment (liquidity.py:411-419) already declares flag-alone a warned
no-op and demands a cost-only dollar-ADV column. **Record as the horizon-consistency requirement
for the eventual `IMPACT_COST_ENABLED` activation** (per-bar ADV, not daily), riding the Chain-2
harvest.

**D6 — The stablecoin peg halt is not cascade protection.** `macro_indicators.check_stablecoin_pegs`
monitors USDT/USDC off Alpaca quotes with warn/emergency bands (macro_indicators.py:~200-230) and
feeds a hard `sizing_mult=0.0` halt (base_loop.py:1904). The Oct-10-2025 cascade ($19B liquidated,
~1.6M accounts, $3.21B in one 60-second window; USDe printed ~$0.65 **on Binance only** — an
internal-collateral artifact, not a cross-venue depeg) **would not have tripped it** — wrong asset,
wrong venue, and the 40-minute core is sub-hourly. Theory agrees no predictor should be built:
the mechanism was Kiyotaki-Moore collateral amplification, and Diamond-Dybvig multiple equilibria
mean a directional cascade predictor is unsound by construction. **Action: re-label the halt as a
narrow USDT/USDC tripwire in docs/comments; route tail-cost work to FR-17 (already specced); do
NOT add a USDe/venue feed** (same rationale that killed on-chain flows and GDELT).

---

## Section 1 — Laureate & nominee-tier foundations (round 3)

| # | Source | Grade | Verdict for this system |
|---|---|---|---|
| 1 | **Grossman-Stiglitz (1980, AER)** | A | **ACTIONABLE** → D4: dollarize the LLM keep/kill ledger. The information-cost equilibrium is the formal doctrine behind `llm_eval`'s b2 verdict. |
| 2 | **Menkveld et al. (2024, JF 79:2339) nonstandard errors + Angrist-Card-Imbens design** | A | **ACTIONABLE.** 164 teams on identical data ⇒ researcher-choice dispersion is large and self-underestimated. The 12-step plan in `06_signal_model_plan.md` is already pre-registered (credit it). Missing: **(a) a placebo arm** — block-permute training targets (blocks ≥ label horizon) through the identical `window_ab.py` refit→blend→holdout path; pass = placebo DSR ≈ 0, else the harness manufactures edge and every arm comparison is void. **(b) Spec-curve on steps 9–10 only** (≥5 seeds × 2 holdout widths; adopt on median + fraction-positive). **(c)** Label llm_eval/decision_report estimands as complier-average (LATE) — fired-trades only, not the vetoed population. |
| 3 | **Roll (1984) → Kyle (1985) → Glosten-Milgrom (1985)** lineage behind the EDGE cost model | A | **ACTIONABLE ×2.** (a) *Agent-measured participation census (yfinance 60d hourly, 2026-08-22, repo's own IMPACT_Y=0.5, $25k notional)*: **20 of 46 stock names ≥3bps/side** √-law impact in a median hour; PRME ≈23bps/side, PALL 17, SERV 14, PPLT 13 (mega-caps <1bp). Four names carry impact as large as the entire modeled 15–25bps cost. Feeds the `IMPACT_COST_ENABLED` enable decision + D5. (b) **Maker markout measurement**: `MAKER_ENTRIES_ENABLED=True` live (strategy_config.py:79) because crypto maker/taker = 15/25bps, but the adverse-selection cost of posting is **never measured** (zero markout code). Extend `scripts/entry_timing_probe.py`'s journal join to signed markouts at k∈{1,3,6} bars split by `entry_tactic`; rule of thumb: E[m|maker]−E[m|taker] < −10bps ⇒ maker entries net-negative. **Kill-boundary flag:** wave-7 killed the adverse-selection contrarian-posting *guard* (a mechanism); a markout *measurement* of a live default is not that mechanism — flagged as ask-shaped, owner confirms the boundary. Kyle-Obizhaeva invariance: nothing at $25k — SKIP. |
| 4 | **Campbell & Thompson (2008, RFS)** | A | **USE-confirms + small add.** Both C-T restrictions are already structurally enforced (long-only kills negative forecasts; tilt hard-clamped to [0.1, 1.30], base_loop.py:2081) — do not rebuild. Add their **R²_OOS vs trailing-mean statistic to `scripts/naive_vs_blend.py`** (~15 lines) so the repo emits the literature-comparable number. Their sentence "a variable is quite likely to have poor out-of-sample performance for an extended period even when it genuinely predicts" is the citation for FR-04's existing rule that a single-window negative is a report, not a kill. |
| 5 | **Realized semivariance / signed jumps — Barndorff-Nielsen et al. (2010); Patton & Sheppard (2015, REStat 97(3):683–697) [V — tables read from the authors' PDF after an initial retraction]; Bollerslev-Li-Zhao (2020, JFQA 55(3)) [Q]** | A (vol) / open (returns) | **MEASURE-first, sharpened.** P&S verified: SPDR h=1 RS⁻ coef 1.182 (t=13.0) vs RS⁺ −0.024 (t=−0.5), R² 0.532→0.611; **but the 105-single-name panel — the relevant column for this book — has BOTH legs significant** (RS⁻ 0.704 t=24.5, RS⁺ 0.268 t=15.6; RS⁻ ~2.6×): don't expect the index result. Signed jump φ_J −0.572 (t=−7.7) SPDR / −0.215 (t=−10.5) panel; jumps ≈2% of QV for SPDR but **≈13% for the average single name**. Two independent lines draw the same boundary: (a) *first-hand simulation (pure-diffusion null)* — corr(**SJ**, hourly return)=0.92 ⇒ SJ-as-alpha is a duplicate column (ROC≡Return_12h class); (b) *P&S §V-B* — adding the Black-1976 leverage dummy to the RS⁻ model gains R² of only 0.001 ⇒ **RS⁻ is NOT a re-encoding of return sign** (the 0.92 applies to SJ, not the RS⁻ level), and RS⁻ **subsumes the leverage-effect feature** (closes the item-8 Black-1976 thread — build nothing separate). Table 1: corr(RS⁺,RS⁻)=0.824 with SJ essentially non-persistent (AC₁ −0.112) vs RS⁻ the most persistent series — same split. Expectation-setter: real-data corr(RV,RS⁻)=0.943 ⇒ RS⁻ adds ~11% orthogonal content, not the ~40% an early constant-vol simulation implied — formally informative (φ_d⁺=φ_d⁻ rejected at 63/66 SPDR horizons and ALL panel horizons) but a MODEST add; size the effort accordingly. **Sampling (corrected, reverses earlier guidance): trailing-24h RS⁻ from 5-MINUTE bars** — n=288 ⇒ relSE(RS⁻)≈0.13 (2× better than P&S's own daily-estimator noise) AND worst-name bounce inflation drops 1.53×→**1.11×** (PRME; SERV 1.06×, mega-caps ≈1.00×), removing most of the liquidity-correlated bias that would turn a CS RS⁻ rank into a spread proxy; also cuts the crypto minute harvest ~5.3M→1.05M bars/yr. **Verdict:** trailing-24h RS⁻ from 5-min bars as a downside-vol input to HAR/sizing — at this sampling the bounce bias is small enough (≤1.11× worst name) that the earlier liquid-subset restriction is DROPPED: whole universe usable, ~78 bars/day/name stocks / 288 crypto (streaming-friendly on 8GB); **reject SJ as an alpha feature**; transfer caveat: P&S forecast *daily* vol — the evidence justifies the trailing-24h aggregation (which restores their effective sampling count), NOT an hourly RS⁻ term; and since RS⁺+RS⁻=RV, adding RS⁻ next to the existing RV inputs carries the full decomposition (the panel's significant RS⁺ needs no second column). Still **BLOCKED on pending kill-list ask #1** (minute-data harvesting, `TRADER_STOCK_MINUTE_EDGE`). BLZ's cross-sectional RSJ return premium stays an **unverified open question** (Section 6). |
| 6 | **Diamond-Dybvig (1983) + Bernanke (1983) + Kiyotaki-Moore (1997)** | A theory / B mapping | **USE-confirms + D6.** No directional run predictor (multiple equilibria); the K-M generalizing variable is pre-state *leverage* (OI/mcap, funding extremes — already harvested); note for FR-17, not a build (n=3 stress windows has no power). |
| 7 | **Koijen-Yogo (2019, JPE 127(4), DOI 10.1086/701683) / Gabaix-Koijen (NBER 28967, unpublished 5 yrs)** inelastic markets | B, decaying | **SKIP crypto; low-priority equity ASK.** Aggregate multiplier ≈5 ($3–8) is a *quarterly, aggregate-only* object — **micro/single-name elasticity ≈ 1, which kills any per-name flow feature on theory alone**; Bouchaud (Quant. Fin. 2022): short-horizon impact is transient √-law. Erosion, dated [V]: Greenwood & Sammon (JF 80(2)) index-add **7.4% → <1%**; Haddad-Huebner-Loualiche (AER 115(3), 2025) ~2/3 strategic offset; Wardlaw (JF 2020) mechanical-return contamination. Honest upgrade: short-horizon *aggregate* evidence exists at our horizon (Ben-Rephael-Kandel-Wohl JFQA 2011: ~half of flow impact reverses in 10d; Brown-Davies-Ringgenberg RoF 2021 ETF-flow reversal; Lim 2026 next-day t=3.12, flow-autocorrelation-masked reversal) — but data probes (Section 5, receipts) leave no lawful free daily path for crypto and only **SSGA navhist** for equities (daily NAV/SO, **SPY 5,091 rows back to Jun-2006**, keyless, robots-permitted, **snapshot file ⇒ strict PIT still needs collect-forward self-archiving**) — covers GLD alone among our 46 names ⇒ book-level series at best, redistribution-prohibited ToS. **Ask, ranked low.** |
| 8 | One-liners | — | **Merton/Samuelson**: SKIP — hedging demand ≈ 0 at hours-to-days; CRRA+iid ⇒ risky share horizon-independent ⇒ no horizon-based sizing justified. **Holmström informativeness**: already implemented twice (llm_eval b2 conditional test; `retrain_ledger.py` paired same-window scoring = relative performance evaluation). **Kydland-Prescott commitment**: confirms the no-discretionary-override exit stack. **Gabaix et al. cubic law (2003, Nature)**: α≈3 ⇒ variance exists ⇒ vol targeting well-posed, 4th moment doesn't ⇒ the existing clamps (vol/kelly mult ∈[0.5,1.5], KELLY_CAP=0.25, TILT_MAX=1.30) are the correct response — nothing to add. **Nobel scan**: 2026 prize announces **2026-10-12** — nothing to react to; 2025 Clarivate econ laureates (Autor & Katz, Bertrand & Mullainathan, Bloom) zero finance; only uncovered finance name 2002–2025 = Ross APT — SKIP (adds nothing over beta ledger + panel ranks). |

---

## Section 2 — Modern economic research 2025–26

**Costs (the load-bearing cluster).**
- **Schwarz, Barber, Huang, Jorion & Odean, "The 'Actual Retail Price' of Equity Trades," JF 80(5),
  2025, 2507–2541, DOI 10.1111/jofi.13467** [V] — ~85,000 *simultaneous* real orders across 5
  brokers: account-level round-trip **7–46bps ex-commission, 6.6× dispersion on identical trades**;
  PFOF doesn't explain it. → The 15–25bps central assumption survives; the tail doesn't. **Action:
  run the stock `--fee-sweep` with a 46bps stress arm.**
- **Dyhrberg, Shkilko & Werner, JFE 168 (2025-06)** [V] — wholesaler price improvement = **24% of
  quoted spread**. → Quoted/EDGE-style spread can *overstate* realized retail cost in liquid names.
  Context for pending kill-list ask #1 (EDGE stamps); not itself a gate change.
- **Rösch, Shohfi, Stanco & Walz, SSRN 6992598 (2026-06)** [E — existence+venue verified by the
  journal sweep; numbers from abstract] — 496 real retail trades, Apr–Sep-2025,
  Coinbase/Kraken/Crypto.com/Robinhood: all-in round-trip **253–834bps**; corroborated by
  Frankfurt School/intas.tech (2026-03): 0.53–6.45% across 9 MiCAR venues. **Alpaca is in neither
  sample — which is exactly why `scripts/crypto_spread_census.py` (already built) is the decisive
  measurement.** If the census lands anywhere near these numbers, the crypto book's 50–60bps gating
  world is fiction; the λ* fee-sweep quantifies survival.
- **Fieberg, Liedtke, Poddig, Walker & Zaremba, JFQA 60 (2025), crypto trend factor** [E] — trend
  survives costs *in the big/liquid-coin subsample* (sample ends May-2022, pre-ETF, pre-cascade);
  its breakeven-cost figure is the natural yardstick for the census — verify the table in one pass
  before quoting numbers.

**Volatility.** HAR stands. Kilic (FEDS WP 2025-061) [V]: regime-switching HAR beats an ML suite on
accuracy, risk forecasts *and* realized utility; ML "no consistent advantage." Rough-vol H≈0.1
remains contested as a microstructure artifact with no forecasting win for an hours-to-days sizing
consumer. **Confirms the campaign's B11 binding (log-HAR, no WLS/HARQ). The one supported upgrade
axis is diurnal periodicity → D3.**

**Announcement economics.** Londono-Samadi + Knox et al. → D1/D2: realized event-day premia are
partly ex-post news, the ex-ante priced premium (7.35bp) is un-harvestable at our costs ⇒
stand-down is the only defensible response. Barardehi & Bernhardt (JFM 74, 2025, DOI
10.1016/j.finmar.2025.100971) [V abstract]: in *trade time*, volatility and Kyle's λ fall
monotonically open→close — the textbook ∪ is an aggregation artifact ⇒ the 14:30–15:30 entry
window may be *cheaper* than assumed, folding into FR-18(b)'s per-hour IC/cost buckets.

**Method / integrity.**
- **Barde, J. Econometrics 253:106123 (2026-01)** [V] — fast Model Confidence Sets: O(M³)→O(M²),
  post-hoc model addition, demonstrated on 4,800 models. **ACTIONABLE (Mac-buildable, pure numpy):**
  emit the *set* of statistically indistinguishable Optuna trials instead of one winner —
  complements DSR (which deflates the max) with an honest near-tie region; slots into the FR-13/
  subset-averaging doctrine (equal-weight the MCS survivors, never fit weights).
- **Yin et al., arXiv 2603.20319 (2026-03)** [V, preprint] — 5 engines × identical strategies:
  exact agreement at zero cost, up to 3.71% total-return divergence under costs, all defects in
  cost/infrastructure code. **Empirical case for the shared-kernel invariant** (`policy_exits.py`/
  `fees.py` shared by backtest and live) — and for D1 being worth fixing.
- **Leakage hygiene (citation pending, discipline valid):** two 2025 top-journal retractions-in-
  spirit — a t≈6 ML alpha traced entirely to look-ahead, and "pockets of predictability" traced to
  a centered kernel — motivate one cheap sweep: grep every rolling smoother in-repo (regime
  hysteresis, adaptive weights, smoothed vol) for accidental centering. Mac-doable; numbers
  quarantined (Section 6) but the audit costs minutes.
- **Simon, Weibels & Zimmermann, Mgmt Sci advance (2026-06), DOI 10.1287/mnsc.2025.00721** [V] —
  deep *parametric portfolio policies* add 43–102bps/mo CE over linear, robust to costs and
  long-only-compatible. Context only: their engine is a large monthly cross-section; at N≈46
  hourly it's unproven — watch, don't build.

**Crypto majors (verdicts, all consistent).** Lee & Wang (JFQA 60(4), 2025) [E]: the RV→next-week
negative relation lives in small/illiquid/retail coins — weakest on majors. Borri, Liu, Tsyvinski
& Wu (arXiv 2510.14435, rev 2026-03) [V]: crypto-carry Sharpe 6.45 full-sample → **negative in
2025** — the funding premium inverted; carry stays killed, funding *features* stay (survivor #1)
with the FR-03 drift audit before retrain. Kim & Hansen (arXiv 2607.09426) [V]: quarter-hour
opening-imbalance predictability ≈ 0.5bp/boundary — **fails cost by ~10× by the authors' own
arithmetic**. Baquero (arXiv 2606.00071 survey) [V]: nothing robustly beats a naive baseline at
short horizons across regimes — corroborates `naive_baseline.py` (FR-04). **Funding→spot: no
peer-reviewed test exists (dated negative). Nothing on majors clears 50–60bps standalone.**

**One overnight warning for FR-18(a).** Day/night decomposition results (Barardehi-Bogousslavsky-
Muravyev, RFS forthcoming [E]; Wang, JFQA FirstView [E]) imply close-to-close IC can be an
overnight artifact that a bot entering next-bar cannot harvest — add an
overnight-vs-intraday split to the FR-18(a) audit before trusting any overnight-adjacent IC.

---

## Section 3 — Strategy families (round 3 screen)

1. **Single-metric quality feature — gross profitability (GP/A), not a QMJ composite.**
   Novy-Marx & Medhat (NBER w33601 / SSRN 5190788, 2025-03) [E]: profitability subsumes quality
   composites AND defensive/low-vol returns. Sharpens the July actionable (still unbuilt):
   build **one** cross-sectional rank column `CS_Rank_GP_Assets` (quarterly-refreshing) on the ~46
   names. **Data ruling:** PIT-clean fundamentals only via **SEC EDGAR XBRL companyfacts/frames**
   (per-fact `filed` dates); **yfinance `.info` is a current snapshot ⇒ PIT violation — banned for
   this.** (`fundamentals.py` today feeds only the LLM prompt — compiler-verified consumers:
   llm_analyst/base_loop/stock_loop; its EDGAR EFTS path 500s per agent probe.) As a *feature* the
   incremental turnover ≈ 0 (it reweights entries the blend already takes); the overlay variant is
   strictly worse — don't build it. Measure: stage0 dump → `ic_by_name` incremental rank-IC over
   the existing CS_Rank block → `indicator_leadlag.py` redundancy → `backtest.py --prefix stock
   --gate`. Model-facing ⇒ rides Chain-2 (gotcha #2). Size expectations for a multi-year factor
   drawdown (2025 = worst quality year since index inception).
2. **Momentum-horizon redundancy pass (feature-pruning, not feature-adding).** Etienne et al.
   (arXiv 2510.23150, 2025-10) [E]: adjacent trend horizons correlate ~0.8; the middle band is
   cost without return; barbell (fast+slow) beats ladder. Transfer is NOT automatic (daily futures
   → 12–48h labels) ⇒ measure ours: `indicator_leadlag.py` already emits |Spearman| clusters over
   the momentum family (Return_4h/12h, Ret_21d, RM_252_21, MA_Dist_*, Pos_Range_20d + CS twins).
   If ~0.8 replicates, add a "barbell" arm next to `stationary_lean` and A/B fixed-config. Payoff
   is Jetson feature-count/memory as much as Sharpe. Same redundancy logic that killed
   ROC≡Return_12h — kill-screen passes.
3. **Exit-cap (tp_rr) skew diagnostic.** Zarattini, Pagani & Wilcox (SSRN 5084316, 2025-01) [E]:
   66k+ long-only trend trades — **<7% of trades drive cumulative profitability** (gross-only
   headline). Both books hard-cap winners at `tp_rr = 2.0` (strategy_config.py:26,40 — "was 3.0");
   if long-only P&L is a thin right tail, a 2R cap structurally removes what funds the other 93%.
   Replay `tp_rr ∈ {2, 3, 4, trail-only}` with everything else fixed via `backtest.py --days N`
   (+ exit-reason distribution + realized skew from journals; `--fee-sweep` per arm). NOT the
   Donchian/ATH entry half (already graded; 52-week-high kill-adjacent).
4. **Signal-exit hysteresis (θ_out < θ_in).** Tranching evidence (Gabriel-Pagani-Zarattini, SSRN
   5230603, 2025-05 [E]) translated to our retail reality: at 50–60bps rt, threshold flip-flop is
   the largest controllable cost. Scope honesty (reconciling two agents): the system has **no
   resize/rebalance path** (entries one-shot, exits = ATR stack — verified negative for a
   Davis-Norman no-trade band), so hysteresis applies **only to the signal-exit threshold** in the
   exit stack. Measure: replay with/without a band via `--fee-sweep`; adopt only if λ* rises
   materially, then `--gate`. Execution-policy change, zero training cost.
5. **Pre-FOMC drift, VIX-conditional — probe rider only.** Published negative (Kurov et al.: gone
   post-2015) vs a 2025 replication (alive at Sharpe 0.5–0.6 in-market ~5% of days, **near-zero in
   low-VIX, strong in high-VIX**) [E]. Reconciled by VIX conditioning. Tag FOMC windows in
   `scripts/entry_timing_probe.py` (calendar is PIT-clean by construction) and report our own
   IC/hit/net-P&L in the 24h pre-decision window split by the existing VIX regime state. Distinct
   from D1/D2 (which are about the *post*-announcement stand-down). No rule without a positive,
   VIX-conditional, in-house result.
6. **Earnings windows: counterfactual before widening the block.** The announcement *premium*
   (positive, idiosyncratic-vol-driven, strong in large caps [E]; explicitly NOT the killed PEAD)
   says blanket avoidance may cost mean return. Before flipping `EVENTS_TRADING_DAY_WINDOWS`
   (currently False, strategy_config.py:131) either way: replay the trades the earnings buffer
   *blocked* (journals + backtest). If blocked-trade mean net P&L is positive ⇒ propose
   size-haircut instead of block. Measurement only.
7. **Net-of-cost objective question (owner-level).** Baldi-Lanfranchi (SSRN 4737166) [E] +
   Jensen-Kelly-Malamud-Pedersen (RFS advance 2026-03) [E]: cost-aware *construction* beats
   cost-aware *filtering*; at retail scale magnitudes shrink but the direction stands. In-repo
   question: does `hypersearch_v2`'s objective score net-of-fee P&L or gross? If gross, moving
   fees inside the objective is the largest single lever surfaced this round — and it is an
   **objective change ⇒ gotcha #2** (study-db deletes, Chain-2) ⇒ owner decision, never an agent
   action. First step costs nothing: read the objective, report which it is.

**Honest negatives from this lane:** low-vol/BAB long-leg tilt (a ~34%-bonds duration bet whose
residual is profitability — build item 1 instead; the free diagnostic is adding IEF/TLT as a third
beta-ledger regressor); post-CPI/NFP *post*-release drift at hourly granularity (falsified —
tradable part lives in the first ~25 minutes; arXiv 2605.04004 [V]); CTREND cross-sectional crypto
factor at N=6–10 (breadth); post-ETF "TSMOM improved" claim (p=0.58 — no evidence);
triple-witching/rebalance-day effects (no credible net-of-cost evidence found — closed as
not-proposed).

---

## Section 4 — TOP ACTIONABLES (ranked, with ship class)

Ship classes: **[M]** measurement/instrumentation — ships direct; **[F]** model/gate-facing —
default-OFF flag + challenger→shadow; **[A]** owner ask/decision first.

1. **[M] Run the crypto spread census + λ* fee-sweep against the 2026 retail-cost evidence**
   (Section 2 costs; stock sweep gains a 46bps stress arm). Everything about the crypto book's
   viability hangs here. Already-built tools; Jetson, hours.
2. **[M→A] D1 parity fix: mirror `macro_standdown()` into `backtest.py`** (gate honesty, aligns
   replay with live); **[F/A] D2: add NFP dates** (widens live gating ⇒ flag/sign-off). Trivial code.
3. **[M] D4: dollarize the LLM ledger** (benefit_usd/net_usd/break-evens; ~20 lines). Decides the
   LLM-spend keep/kill with G-S economics instead of unit-mismatched telemetry.
4. **[M] Placebo arm in `window_ab.py` + spec-curve rule for plan steps 9–10 + LATE labels**
   (Menkveld/ACI). Integrity insurance for the entire 12-step Jetson sequence; one arm's compute.
5. **[F] D3: `HAR_DIURNAL_ENABLED` diurnal vol vector** (QLIKE offline gate first; crypto book
   first). Three concordant sources; sizing-path only; kill-screen clean.
6. **[M] C-T R²_OOS line in `naive_vs_blend.py`** (~15 lines) — literature-comparable falsification
   number alongside FR-04's IC/DSR.
7. **[F] GP/A quality rank feature via EDGAR XBRL** (Section 3 #1; rides Chain-2; IC-gated).
8. **[M/A] Maker markout measurement in `entry_timing_probe.py`** (Section 1 #3b; ask-shaped
   boundary vs the wave-7 kill — owner confirms, then it's a journal join).
9. **[M] tp_rr exit-cap replay** {2,3,4,trail-only} + exit-reason skew (Section 3 #3).
10. **[M] Momentum-horizon redundancy pass** via `indicator_leadlag.py`; possible lean-preset
    barbell arm (Section 3 #2).
11. **[M] Signal-exit hysteresis replay A/B** via `--fee-sweep` λ* (Section 3 #4).
12. **[M] Barde fast-MCS module** (pure numpy, Mac-buildable) emitting the near-tie set of Optuna
    trials next to the DSR winner (Section 2 method).
13. **[M] Earnings-blocked-trades counterfactual replay** (Section 3 #6).
14. **[M] FOMC pre-drift VIX-conditional probe rider** (Section 3 #5).
15. **[M] Leakage-hygiene sweep**: audit every rolling smoother for accidental centering; plus the
    participation-census read → owner decision on thin-name universe/impact activation (D5).

*Blocked/deferred:* RS⁻ downside-vol input (blocked on pending ask #1 minute bars); SSGA book-level
flow series (low-priority ask, Section 1 #7).

**Sequencing note:** nothing above displaces the 12-step plan in `06_signal_model_plan.md`. Items
1–4 are pre-Jetson or same-trip additions; item 4 (placebo) should run **before** steps 9–10 of
that plan; item 5 can join the same trip as the FR experiments; item 7 waits for Chain-2.

---

## Section 5 — Honest negatives & proposed kill-list additions

**Dated negatives established this round (with receipts — do not re-derive):**
- **Crypto ETF-flow features** — evidence: next-day t=3.12 lives in one unpublished single-author
  WP with bidirectional Granger and no cost test (Lim [V abstract]); causality substantially
  price→flow (Oefele, Econ. Letters 2025 [E]); the famous R²=95% is a cointegration artifact
  (levels on cumulative flows). Data: **fails free-PIT everywhere** — Farside is Cloudflare-403 to
  code with no ToS page (one probe reported a Wayback-diff freshest-row revision; the parent lane
  could not re-substantiate it — revision behavior UNCONFIRMED; the 403 alone fails it);
  CoinGlass paid (401 keyless, $29/mo floor); SoSoValue 1-month history; yfinance
  `get_shares_full` returns length-0 for IBIT/SPY (two independent probes) and its snapshot SO ran
  ~17% off the issuer's own file; issuer CSVs are snapshot-only, zero backfill (the exact wave-4
  Cboe-put/call rationale). **ETF flows are NOT the on-chain kill by data type, but its stated
  rationale ("new unreplicated data dependency") applies verbatim.**
- **Equity flow/demand-system features** — micro/single-name elasticity ≈ 1 kills per-name flow
  features on theory (KY/GK identification is quarterly + aggregate-only); index-reconstitution
  flow decayed 7.4%→<1% (Greenwood & Sammon, JF 2025 [V]) with ~2/3 strategic offset (Haddad et
  al., AER 2025 [V]); short-horizon *aggregate* reversal evidence exists but is unbuildable free
  (Section 1 #7); sole surviving free daily-history file = SSGA
  navhist (SPY/sector complex only, 1/46 names, redistribution-prohibited) ⇒ book-level ask at
  most.
- **Stablecoin flow aggregates** — fails on data before evidence (supply not PIT: restatements,
  chain migrations, treasury supply; exchange balances paywalled); only direct study is the
  already-killed on-chain dependency class; BIS work targets FX parity, not BTC returns.
- **Funding→spot direction on majors** — no peer-reviewed test exists (2025–26); best industry
  test: T→T+1 R²≈0. Keep existing funding features (survivor #1), build nothing, run FR-03 first.
- **Retail order-flow / BJZZ-Mroib** — algorithm mis-signs 28% of identified trades against
  ground truth (Barber et al., JF 2024 [V]); needs paid TAQ; effect ~10bps/week inside our cost
  band. Adjacent to killed order-book-imbalance + auction-imbalance entries.
- **Intermediary/dealer risk-appetite overlays** — free daily proxies are T+2-lagged composites of
  inputs we already gate on (VIX, realized vol) ⇒ third correlated stress input + the rev-07-02
  VIX double-count; He-Kelly-Manela refreshes months late. Consistent with the killed wave-4
  macro-overlay family.
- **Low-vol/BAB long-leg tilt; post-release macro drift at hourly; CTREND at N≤10; post-ETF TSMOM
  claim; triple-witching** — Section 3 negatives.
- **SJ-as-alpha; Kyle-λ estimator; Kyle-Obizhaeva at $25k; directional cascade
  predictor; USDe/venue peg feed; Hansen-Sargent min-over-models sizing; no-trade band for
  resizing (no resize path exists); Merton horizon-based sizing; MAD-for-σ swap** — Sections 1–2.
- **Quarter-hour opening-imbalance effect** — fails cost ~10× by its own authors' arithmetic.
- **2025 Clarivate cohort** — zero finance; nothing to mine.

**Proposed NEW kill-list entries (asks — entries change only by owner decision):**
1. **ETF net-flow features (crypto AND equity)** — rationale above; carve-out note: a future
   SSGA-navhist *book-level regime* series would need its own ask.
2. **Retail order-flow / Mroib-style features** — mis-signing + paid TAQ + sub-cost effect; file
   under the existing order-book-imbalance heading.
3. **Index-reconstitution flow trades** — Greenwood-Sammon decay to ~0.
(Also: record "funding→spot predictor on majors" and "stablecoin-flow aggregates" as dated
negatives inside the wave file if not promoted to the list.)

---

## Section 6 — Quarantine (do NOT act; one verification pass would clear or kill)

From the corrected lane — bibliographically real or plausible but with **numbers unverified**:
Chu-Shen-Zhu (JFQA accepted) model-disagreement veto idea + Bali et al. RFS "Machine Forecast
Disagreement" (the *veto-only* framing was sound: any disagreement signal must never become a long
tilt — the long leg is the insignificant one); Dai-Shi-Zhang JFM spread-estimator blend (+10pp
claim measured off CHL, not EDGE); Zhang-Zhu-Linnainmaa RFS leakage magnitudes; Cakici et al. JF
centered-kernel magnitudes; Aleti-Bollerslev-Siggaard Ross-bound/jump-contamination numbers (paper
itself [V], those numbers not); Branco et al. JEF vol-ML result; Schwenkler et al. crypto
vendor-data quality; Oefele's specific reverse-causality coefficients; Mesfin's exact t-stats are
[V] but single-preprint; Lacava et al. "Realized Illiquidity" (a realized-Amihud computable from
hourly bars — the one lead worth a follow-up search); McLean-Pontiff-Reilly JFE 2025-11, Cao et
al. JFE 2026, Easley-O'Hara crypto-microstructure JFM 2026 (exist; contents unread).
Strategy-lane items 1/4/7 (Novy-Marx, Zarattini-stocks, Baldi-Lanfranchi) draw numbers from
abstracts/secondary pages — one primary-table pass recommended before any build ships.
Retraction history, resolved: the Patton-Sheppard coefficients were struck mid-round (their
sub-agent never reported), then **recovered and verified directly from the authors' PDF** — now
cited [V] in Section 1 #5; two figures claimed earlier ("OOS R² 40.4→53.9", "DM=0.02") do NOT
appear in the paper and stay struck. **All Bollerslev-Li-Zhao figures remain unverified**; the
**BLZ RSJ cross-sectional return premium** is an open question, not a graded negative — one
literature pass closes it either way before any build or formal kill.

**Coverage honesty:** JF + RFS advance-access lists, JFQA FirstView, and Review of Finance were
never fully swept (budget exhaustion; RoF ISSN lookup failed) — treat "no relevant JF/RFS 2026
paper" as *unknown*, not established. No new live-vs-paper slippage study was found in what WAS
swept.

---

*Compiled by round-3 research (3 lanes + 3 sub-sweeps + 2 data probes), adjudicated against
`KILL_LIST.md`, `05_frontier_research.md` §3 negatives, and the repo source. Code claims
compiler-verified 2026-08-22. This file is research + measurement plans only — no code was
changed in this round.*
