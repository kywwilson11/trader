# Campaign 2026-08 — Phase 1 State Map

**Produced:** 2026-08-18, from 15 subsystem reader reports covering every production module, the
tooling, and the research corpus, synthesized against `research/KILL_LIST.md` (read in full) with
spot-verification of the highest-stakes claims against the current working tree. Working tree =
uncommitted state on top of HEAD `c7f846e`. This file is the campaign's shared ground truth for
Phase 2 (external research fan-out) and Phase 3 (build). Defect IDs (D01–D40) and build IDs
(B01–B24) here match the structured seeds handed to the orchestrator.

---

## A. One-page system truth (as of the current working tree)

The system is a paper-trading engine with one RegressionLSTM + LightGBM blend per book, a stack of
admission gates (cost floor, meta-label veto, sentiment, LLM), ATR exit management with resting
server-side stops, and a promotion pipeline (Optuna search → holdout Deflated Sharpe gate → policy
replay gate → champion/challenger shadow). Roughly three weeks of verified panel-hardening work (19
modules through the module-improve-v3 panels, the full GUI overhaul, ~40 modified files, 19 new v3
test files) sits **uncommitted** and has **never produced a single number on the Jetson**. Ten
things define the current truth:

1. **The promotion machinery is not currently trustworthy in either direction.** The effective-n
   fed to the Deflated Sharpe gate is a connected-components count that collapses ~600-trade books
   to ~1–21 "independent" observations (making the gate near-unpassable), while a silent 10-sample
   floor rescues degenerate values (making it passable on 4 real observations). On top of that, the
   weekly policy gate replays the **champion** while the freshly trained **challenger** deploys via
   the shadow forecast-error test alone — the one mechanism the kill list brands a category error
   for policy decisions — and a gate failure rolls back the innocent champion.
2. **The crypto book — the higher-fee book — runs on fictional costs.** It was never stamped with
   per-bar effective spreads (flat 0.10% haircut everywhere), the admission floors diverge three
   ways between backtest, live, and meta-label training (the meta layer applies no floor at all),
   and the Optuna threshold search range sits entirely below the 1.20% floor the deployed policy
   enforces, so model selection optimizes a population the book can never trade.
3. **The stock book's status is unknown pending one Jetson check.** The code-side fix for the
   wave-4 "predictions all None" P0 is in the tree but unverified with real data; and even if it
   works, ~14 long-window daily features (12-1 momentum, 200-day trend distance, etc.) are served
   as constants live while training saw real values — the certified model and the served model see
   different worlds.
4. **The de-risk stack chronically destroys size.** VIX is read three times into one entry's size,
   the KILL-LISTED pseudo-CAPE still cuts most stock entries 30%, the portfolio vol target is
   applied twice, and the raw multiplier product saturates the 0.1 floor in stress — the modal
   normal-market entry is sized at roughly 0.39–0.56x of intent, and in a crisis all differentiating
   signals go inert at the floor. The batch (B7) composed to adjudicate this stack never ran.
5. **Several measurement instruments that decide real money are themselves broken.** The LLM
   spend keep/kill verdict (llm_eval's b2 significance) is statistically void (10–53% false-keep
   under the null); the conviction-flagship A/B passes floors on absent fields; the rank-gradient
   gate can CONFIRM on noise; stock take-profit winners are censored out of the Kelly sizing sample
   while stop losers are counted; and both Stage-0 holdout gates are inert because the per-bar
   predictions dump they consume was never authored.
6. **The live engine has real tail-risk liveness holes.** The prediction fan-out has no timeout
   anywhere (one hung socket freezes stops, sells, the circuit breaker, and the kill switch), the
   emergency flatten cancels its own liquidation orders at timeout, the /flatten kill switch reaches
   only one book, a stock bracket partial fill leaves real untracked shares, and the maker ladder
   can triple intended exposure on an ambiguous order outcome.
7. **A very large activation backlog of built, tested, dormant code exists.** cost_regime.py,
   basis_archive.py, squeeze_features.py, blend_fit.py, bet_sizing.py, execution_policy.py, the IOC
   price cap, the account-level risk cap, CSCV-PBO, the prediction cache, crypto_trend.py, the
   SAFE_HAVEN and CRYPTO_POOL / TRADABLE_POOL expansions, and most wave-8 flags all have zero
   production consumers. The waves' own verdict stands: the highest-ROI work is activation, not
   invention.
8. **Data plumbing silently rewrites history.** Incremental crypto harvests overwrite the trailing
   ~730 days with Yahoo composite prices/volumes; incremental harvests erode ~112 bars of head per
   run; interior fetch holes are permanent; and forming (partial) bars leak into live ATR, GARCH,
   cross-sectional ranks, the shadow evaluator, and Stage-0 replays.
9. **The verification stack cannot catch a money-path regression.** ab_check can print PASS having
   run zero tests, the baseline masks the live loops' import failures and the shadow-promotion
   tests by name, base_loop has essentially zero functionally-executed coverage on the Mac, and CI
   never installs lightgbm or the production numpy/pandas pins — no environment other than the
   production Jetson has ever run the production stack.
10. **The prior verdict stands until the beta ledger runs:** the book is a gated-BETA book
    (~85–95% invested variance is factor) until proven otherwise, and the sanctioned path out
    (beta ledger → SPY hedge via easy-to-borrow shares at $0 borrow → cross-sectional long-short)
    is measurement-ready but has produced no Jetson numbers yet.

**The single most important operational fact:** none of the shipped instruments (beta ledger,
decision report, sizing-decomposition journal, n_eff provenance fields, entry-rank journals) has
run on real data. Step one of everything is: commit the tree, sync, and run the one-look Jetson
diagnostics. Most owner decisions below are deliberately blocked on exactly those numbers.

---

## B. Ranked defect-candidate table (money impact first)

Severity: how much money the defect loses, risks, or blocks. Status: `new` = found this campaign;
`known_open` = documented in code/reports but undecided/unfixed; `deferred_owner` = sitting in an
owner-decision queue. Sources = number of independent readers that flagged it (cross-module views
of one defect merged). Fix shape notes whether the change is measurement-only (ships directly per
the deployment convention) or model/gate-facing (default-OFF flag, owner ruling, and/or the
challenger → shadow path, with gotcha #2 study resets where scoring changes).

| # | Where | What it does to money | Sev | Status | Src |
|---|---|---|---|---|---|
| D01 | base_loop.py:1025 (+trading_utils.py:67) | Prediction fan-out has no timeout and the REST client no request timeout: one half-open socket wedges the entire cycle — stops, sells, circuit breaker, and the /flatten kill switch all stop running; five wedged workers brick the pool permanently. The largest single tail-loss mechanism in the engine. | critical | deferred_owner | 1 |
| D02 | sample_weights.py:279 + backtest.py:424 + validation.py:134 + hypersearch_v2.py:1179 | Promotion-gate effective-n is broken in both directions: the connected-components estimator collapses calendar-tiling books to ~1–21 clusters (gate near-unpassable, good retrains rolled back weekly), while the silent 10-sample floor and the unguarded 0.0 sentinel rescue degenerate values (gate passable on ~4 real observations). Current gate verdicts are uninterpretable until both are ruled together. | critical | deferred_owner | 3 |
| D03 | run_pipeline.py:977/1003 + backtest.py:18 | Under default shadow mode the weekly policy gate replays the CHAMPION while the fresh model sits in the challenger slot: the model that will deploy is never policy-gated (it promotes on the shadow forecast-error test alone — the kill-listed category error), and a gate failure rolls back the innocent live champion. | critical | known_open | 2 |
| D04 | scripts/harvest_crypto_data.py (no spread stamp site) | The crypto book was never stamped with per-bar effective spreads; every crypto gate, label, and certification prices a flat 0.10% spread while the book pays 0.50% fees plus real, several-fold-dispersed spreads. Wide-spread pairs get certified on fake positive net; tight pairs are over-taxed offline. Kill-adjacent (KILL_LIST line 90) — owner sign-off plus the hourly-EDGE level reproduction required first. | critical | known_open | 3 |
| D05 | backtest.py:277 + meta_label.py:401 + order_utils.py:636 + adaptive_config.py:29 | The unified edge-floor divergence: backtest admits on the flat spread, live admits on the real quote, meta-label training applies no floor at all (16x below the crypto deployment floor), and the Optuna trade_threshold range [0.05, 1.0] sits entirely below the 1.20% crypto floor — model selection and the meta veto are calibrated on populations the deployed policy cannot trade. One owner ruling, four bindings; the range fix is a scoring change (gotcha #2). | critical | deferred_owner | 3 |
| D06 | stock_loop.py:961/667 | Stock take-profit bracket-leg exits (the book's natural winners) are journaled as estimated external closes at the current midpoint and are therefore excluded from Kelly sizing and the CUSUM monitor, while server-stop losers are confirmed and included. Kelly is biased toward the 0.05 floor — the stock book systematically undersizes exactly when it is winning — and the drift monitor false-alarms. The TP leg's order id is never stored, though the real fill is one REST call away. | critical | new | 1 |
| D07 | events_calendar.py:157/182/196 | Earnings proximity uses calendar days: Friday entries and Friday→Monday overnight-sleeve holds are unprotected against Monday prints (a large fraction of reporters), and Monday gets no post-print caution after a Friday-evening report. This is the standing P0 and the single largest tail in the stock book; its entry in the owner queue is literally a placeholder with no description. | critical | known_open | 2 |
| D08 | data_sources.py:175 | Incremental crypto harvests run the yfinance leg every time with period='max' (ignores start_date) and the store merge keeps 'last': the trailing ~730 days of the crypto training store converge to Yahoo composite prices and USD volume (~7 orders of magnitude off venue volume). The model trains and certifies on prices from a venue the bot does not trade. | critical | deferred_owner | 1 |
| D09 | llm_eval.py:411/141 | The LLM keep/kill verdict is statistically void two ways: the HAC lag is counted in rows on a sample with ~10 pseudo-replicated rows per timestamp (10–53% false-'keep' rates in three independent null simulations), and the stock horizon treats a bar count as wall-clock hours (~3.4x horizon stretch, biasing b2 toward 'keep'). Every dollar of daily LLM spend is currently authorized by this number. | critical | known_open | 2 |
| D10 | base_loop.py:1592–1665 + macro_indicators.py:268–315 + regime_detector.py:200 + portfolio.py:447 | The de-risk multiplier stack: VIX is read three times into one entry's size (ladder + macro sizing_mult on identical breakpoints + Kelly cap), the KILL-LISTED pseudo-CAPE (SPY P/E×1.6) still cuts ~every stock entry 0.7x, the redundant HMM layer's smoothing is inverted (switches after ONE observation; returns neutral instead of holding the previous regime), and the vol target is applied twice. Modal-regime entries run at 0.39–0.56x intent; in crisis the product saturates the 0.1 floor (~45,000x rescue) making every differentiating signal inert. The B7 panel composed to adjudicate this never ran. | high | deferred_owner | 3 |
| D11 | predict_now.py:222 + indicators.py:889 + market_data.py:214 | ~14 trained stock features (RM_252_21, MA_Dist_50/100/200d, ON_Mom_252, TugOfWar_252, Same_Hour_Mean_40d, Pos_Range_60d + their cross-sectional ranks) are permanently constant 0.0/0.5 at inference because the live frame is 45 days — training and the promotion backtest see real values. The stock book's likely strongest signals contribute nothing live, and every live prediction sits in a training-atypical input region. | high | new | 3 |
| D12 | meta_label.py:497/488 | Meta in-sample-primary leak (confirmed still open): train_meta loads the deployed primary and scores exactly its own training window, so the 'pred' feature and the row-selection filter are in-sample fitted values. The veto/size-tilt layer learns to trust optimism it will never see live. No calibration mode can remove it; requires persisting out-of-fold predictions from hypersearch. | high | deferred_owner | 2 |
| D13 | meta_label.py:584 + shadow.py:497 + calibration.py:93 | Meta/calibration publish is ungated: the default 'legacy' branch has none of the degenerate-calibrator guards, so a one-class calibration tail can publish a constant p=0.0 that vetoes the entire book for a week (or ~1.0 boosting everything 1.3x); shadow's post-promotion background meta retrain reaches live with zero checks; and the isotonic tie-collapse bug (order-dependent maximum instead of weighted mean) both risks mis-sizing and confounds the purged-OOF calibration flip — wave-9 activation item #1 is blocked on it. | high | deferred_owner | 3 |
| D14 | base_loop.py:1086/467/1112 | LLM outage cluster: the throttle stamp advances only on success so an outage collapses the 600s cadence into a 30s retry storm (~20x spend, added stop-path latency); startup preloads llm_analysis.json with no enabled/age check so a disabled or long-dead LLM keeps vetoing entries from an arbitrarily old disk cache; and a stale veto (plus its accumulated strikes) persists through the whole outage, blocking entries on old evidence. | high | known_open | 3 |
| D15 | base_loop.py:396/105 | Peak-equity restore runs on the $100k placeholder before any real equity fetch: on any account below $100k the drawdown ladder pins every entry at ~0.25x forever (invisible on the default paper account); the stock book also trades its first minutes after the open on seed equity with no macro regime. A silent, permanent P&L haircut on any real-sized account. | high | deferred_owner | 1 |
| D16 | stock_loop.py:957 | The stock entry judges the bracket parent by status=='filled' only, violating the lifecycle contract ('judge by filled_qty'): a partially-filled-then-canceled parent leaves real acquired shares completely untracked — no Position, no stops, no journal row, no risk-cap accounting — until the 15:50 orphan sweep. base_loop fixed its copy of this exact bug; the hand-duplicated stock path never got the mirror. | high | new | 1 |
| D17 | base_loop.py:1832 | The /flatten kill switch flattens only ONE book: the shared flag file is consumed by whichever book cycles first (crypto virtually always, in combined mode), and the stock book can never see an off-hours request because the market-hours return happens before the flatten check. The operator's emergency exit leaves the stock book fully invested. | high | known_open | 1 |
| D18 | order_utils.py:285 | place_maker_buy treats a lifecycle None (unknown outcome — possibly a live GTC bid still working) as zero-fill: the next rung AND the taker fallback re-send the full remaining notional with fresh order ids that can never collide as duplicates — up to ~3x intended exposure past every risk cap, precisely during API instability. | high | deferred_owner | 1 |
| D19 | order_utils.py:957/479 | emergency_flatten confirms liquidations through manage_order_lifecycle, which CANCELS the order at its timeout: a slow flatten in a fast market (or any off-hours stock flatten) withdraws its own liquidation with the protective legs already canceled — naked exposure from the one path whose job is guaranteed exit. | high | deferred_owner | 1 |
| D20 | base_loop.py:811 + stock_loop.py:1047 | Server-side resting-stop fills at the TRAILING (ratcheted, often profitable) level are journaled as hard stops and given the 24h re-entry lockout: winning momentum names are suppressed for a day right after wins, and exit-reason attribution in every downstream report is poisoned. Fix requires exposing stop_price through the alpaca_compat shim (currently absent). | high | deferred_owner | 1 |
| D21 | order_utils.py:165/527 | Above a 0.1% spread the entry limit is never marketable, so every wide-spread entry becomes a ~30s passive wait then an UNCAPPED market order — slippage concentrates on exactly the thin-book bad days. The IOC price-cap machinery (place_marketable_ioc, IOC_CAP_BPS) is fully built, tested, and wired to nothing. | high | deferred_owner | 1 |
| D22 | scripts/hypersearch_v2.py:901 | The deployed artifact is the single best-validation-Sharpe fold's checkpoint soup — never refit on the full pre-holdout window: the champion can be a model trained only on the OLDEST ~55% of data, selected by a winner's-curse max across correlated folds, deployed blind to ~5 months of the most recent regime. | high | new | 1 |
| D23 | blend_fit.py:34 + predict_now.py:405 | The wave-9 blend-weight tuner has zero callers and nothing ever writes 'lstm_weight' into the config: the live blend is permanently the hardcoded 0.6/0.4 that the module's own docstring says under-weights the stronger LightGBM leg. Prediction accuracy left on the table on every trade. | high | new | 2 |
| D24 | scripts/hypersearch_v2.py:431/424 | The trial objective still scores the un-deployable short leg (OBJECTIVE_LONG_ONLY=False), so a bear-carried model can win the search and clear the holdout while the long-only book earns ~zero; and simulate_trades walks the ticker-concatenated panel as one serial stream (positions bleed across ticker boundaries; concurrent breadth is never scored). The search's definition of 'best' is not the book's. | high | deferred_owner | 1 |
| D25 | scripts/hypersearch_v2.py:1149/1575 | The holdout Deflated-Sharpe certificate is issued to the LSTM leg alone, but the deployed predictor is the 0.6/0.4 blend plus the q10 tail veto (trained after the save, never holdout-scored): the core anti-overfit gate never evaluates what actually trades. | high | known_open | 2 |
| D26 | macro_indicators.py:310 + base_loop.py:1665 | The stablecoin-emergency halt (sizing_mult = 0.0) is defeated by the 0.1 tilt floor: during an active depeg the crypto loop flattens, then can immediately re-buy at 10% size into the contagion. The one scenario this module exists to prevent. | high | new | 1 |
| D27 | sentiment_history.py:344 | Stock sentiment articles are never incrementally refreshed (a ticker refetches only when it has ZERO cached articles in the trailing year), so Daily_Sentiment decays to 0.0 for all recent training bars and live serving — a dead feature plus a permanent train/serve shift, which also silently re-arms the sentiment triple-count the moment anyone fixes it without noticing the gate overlap. | high | deferred_owner | 1 |
| D28 | funding.py:102 | The funding z-score baseline holds ~2.8 days, not the intended ~90 (the 15-minute poll appends the continuously-drifting predicted rate on essentially every fetch): the 0.6x/0.25x crowding de-risks fire routinely on noise — a persistent size leak on the 24/7 book. | high | new | 1 |
| D29 | portfolio.py:491/447 + volatility.py:268 | The account-level book-vol scalar consumes deposit-contaminated raw equity (one transfer pins all entries toward 0.5x for ~3 months — the same series that fabricated +4.2%→+43.6%/yr alpha in the beta-ledger simulation), and PORTFOLIO_VOL_TARGET is applied twice (per-position GARCH multiplier and account-level scalar), worst case 0.25x for one condition. | high | deferred_owner | 3 |
| D30 | volatility.py:164/204 | HAR-RV — the certified-better wave-4 vol forecaster, flag-enabled — is structurally dead live: it needs 60 daily observations but live fetches provide ~10 (crypto) / ~31 (stock) days, so every sizing decision silently falls back to the inferior GARCH while burning CPU refitting it hourly. The partial-day regressor and whole-day caching bugs must be fixed with the activation. | high | known_open | 2 |
| D31 | indicator_config.py:125 + indicators.py:634 | The config comment claiming cross-asset columns are 'auto-included via asset-type filtering' is false — no such mechanism exists: under the production preset the crypto model trains with NO BTC context features (BTC_Return_1h/SMA_Ratio/RSI) and the stock model loses the raw session features; separately RS_vs_SPY divides signed returns (sign-incoherent, explodes near zero) and its rank IS in the production preset. Adding BTC features touches the undecided 'BTC-lagged spillover' item (kill-list commonly-confused survivor #10) — owner sign-off required. | high | new | 1 |
| D32 | monitor_drift.py:398/388 | Drift-triggered retrain churn: the retrain flag is re-written on 'warn' days after a consumed trigger (only a full 'ok' day resets the streak), and pred_history is never cleared on model deploy, so the first post-retrain check compares the OLD model's predictions against the NEW model's deciles. Spurious retrains burn Jetson GPU days and churn the champion. | high | new | 1 |
| D33 | base_loop.py:1107 | The v1 llm_analysis journal row fabricates s=0.5 for any missing score and carries no dedup/model/prompt identity: llm_eval counts fabricated neutrals and cached re-serves as independent fresh observations — the producer-side half of the void spend verdict (D09), and the blocker on enabling the dedup cache that would cut most analyst spend. | high | known_open | 2 |
| D34 | shadow.py:585/261 | The shadow promotion test is anti-conservative three ways (Newey-West long-run variance truncated in pooled-record units, ~14 unadjusted daily peeks from day 14, MIN_OBS counted in pooled records ≈ 4 hours of stock data): challengers can promote on noise, replacing a working champion with a worse model for at least a week. | high | known_open | 1 |
| D35 | portfolio_backtest.py:73 + rank_gradient.py:72 | The conviction-flagship evidence chain is invalid end-to-end: conviction floors PASS on absent fields (the production panel never carries meta_p/pred_thresh_ratio, so the flagship A/B would report 'floors change nothing' on floors that never executed), and the rank-gradient verdict CONFIRMS on point-estimate means with no n floor and no confidence interval — it can green-light concentration and edge-Kelly on pure noise. | high | deferred_owner | 2 |
| D36 | scripts/ab_check.sh:37 + tests/baseline_failures.txt:13 | The regression gate every campaign relies on can false-PASS: pytest stderr is discarded and zero-tests-run yields 'NEW failures: none' → exit 0, there is no timeout, and the baseline masks by name the live loops' import failures and the shadow-promotion tests — a new breakage of the money engine or the promotion path is invisible to the Mac gate. | high | new | 1 |
| D37 | .github/workflows/ci.yml:34 + requirements-ci.txt + requirements-jetson.txt | CI never exercises the production stack: lightgbm is installed nowhere off the Jetson (the LGB half of the blend, the q10 veto, and the meta trainer are only ever tested against stubs), numpy/pandas are unpinned (no leg matches the Jetson's numpy<2), alpaca-py (the auto-fallback broker SDK) is never imported in any test environment, and requirements-ci.txt is dead config (used only as a cache key). Production dependencies also have no lockfile. | high | new | 1 |
| D38 | panel_ranks.py:160 + market_data.py:455 + shadow.py:311 + decision_report.py:292 | Forming-bar leaks at four-plus consumer sites: live cross-sectional ranks are computed from the in-progress bar and attached to the previous closed bar (train/serve skew on the wave-3 flagship inputs), live ATR and GARCH include the partial bar (stops set tighter and size larger than certified early in each hour), and the shadow evaluator and Stage-0 replays price against partial closes. | high | new | 1 |
| D39 | scripts/harvest_crypto_data.py:162 + scripts/harvest_stock_data.py:159 | Incremental head-creep: every incremental harvest re-extracts OHLCV from the post-dropna feature file and the recomputed 100-bar-warmup features die in dropna again — permanently deleting ~112 bars of head history per run (~245 days/year at weekly cadence, faster if the pipeline restarts daily). The training set silently shrinks with no alarm. | high | deferred_owner | 1 |
| D40 | liquidity.py:51/145 | Every bar where the spread estimator had no estimate (warmup, halts, zero-range) is stamped at the 0.02% FLOOR — ~26% cheaper than the flat fallback — on exactly the least-liquid bars; and infinities are converted to the floor instead of the cap. Cheap fabricated costs inflate certified edge precisely where fills are worst; measured to flip meta labels in the marginal band. | high | known_open | 1 |

### Below-the-line defects (real, smaller, or folded into the bundles above)

These are confirmed and tracked but did not make the 40-slot structured list; each is one owner
sentence or rides an existing bundle. — **Sizing/exits:** macro stop_mult tightens the traded stop
after size was computed on the un-tightened stop (risk under-deployed and stop-outs raised in
stress); the 1-reading signal exit vs 2-reading stop confirmation asymmetry (rev-07-02 conflict #2)
is still unreconciled; hard_stop_lockout.json is shared/unprefixed so each book's save clobbers the
other's lockouts. **LLM:** a model-less bot can still LLM-veto-liquidate positions while the same
missing model blocks all entries; OpenAI and custom-endpoint models inherit a 50-requests/day
default budget (silently disabled by mid-morning) and unknown models are priced at Gemini-Pro rates
so FREE endpoints burn the $1/day cap; one model's 429 cools down the whole provider including its
own fallbacks; prompt_ab permanently censors one-sided provider failures out of the A/B sample.
**Execution/plumbing:** the alpaca_compat quote shims drop the timestamp so the 180s staleness
guard is inert under the adapter; crossed quotes clamp to zero crossing cost (loosest floor on the
least trustworthy quotes); unknown asset_type is priced on the cheap stock side in a live admission
path; order_stream has no guard against the forbidden two-process mode. **Data/features:**
stock harvest exits 0 on total fetch failure and ignores the save return (trains on stale data with
no retry/notify); a transient chunk error permanently deletes up to 6 months of one ticker
(incremental never revisits interior gaps); the stale-adjustment-basis split/dividend cliff at
incremental boundaries has no detector; the bad-print filter eats the first bar of a genuine crash
at the frame tail; Volume_Ratio has no zero guard (+inf reaches the scaler); Hurst on price levels
keeps the live mean-reversion gate dead; the production preset still ships the kill-listed
duplicate columns (ROC, MACDs, STOCHd, Month_sin/cos, Turn_of_Month) with stationary_lean built and
unadopted; OI features are trained in USD notional but served in coin units; sentiment phrase/word
double-scoring nulls 'rate cut' headlines and substring matching pollutes SOL/DOT/LINK relevance;
the FMP fallback renders revenue-per-share as 'RevGrowth=2500%' in the LLM dossier. **Journals/
ops/reporting:** stock buy rows journal requested not filled notional (dollar P&L overstated up to
one share's price); journals and three pipeline logs grow unboundedly on the Jetson SD card;
standalone run_bots mode has no Telegram kill-switch polling and no drift check; notify records the
dedupe timestamp before sending so a failed CRITICAL alert is suppressed 10 minutes; the novelty
store clobbers itself across the two default separate processes; shadow's .prev backups are
best-effort so a disk-full Jetson forfeits every rollback path; the GUI gate box ignores the
verdict/insufficient-n/representative fields it was hardened to produce and misattributes stale
causes; chart_core's median-spacing annualization inflates the displayed Sharpe ~20% vs
beta_ledger's fixed 252; beta_ledger's trend-conditional state is not lagged (PIT violation) and
its HAC lacks the finite-sample correction; the Total-P&L baseline decays into trailing-1Y P&L once
the account is older than a year. **Corpus/meta:** session-state.md is 3.5 weeks stale and
contradicts the campaign plan (wrong resume point); the module-review queue's ONLY P0 has a
placeholder description; entry-window candidate populations are not comparable across the two books
(weak-signal windows censored); the committed permission allowlist pre-approves the git-stash
operation the agent brief forbids; connection_test.py always exits 0.

---

## C. Gap / opportunity list (things that don't exist yet, or exist and are wired to nothing)

1. **The per-bar predictions dump from backtest.py** — the one unauthored producer both finished
   Stage-0 holdout gates (rank-gradient and IC-by-name harnesses) wait on. The entire wave-9
   activation chain (breadth promotion, concentration flagship, edge-Kelly) is blocked here. The
   single highest-leverage unbuilt piece in the repo.
2. **The dormant-code activation backlog** (all built, tested, zero consumers): cost_regime
   features into both harvests; basis_archive sync + features + live premium serving;
   squeeze_features columns; blend_fit into hypersearch; IOC price cap into the entry fallback;
   account-level cross-book cap clamp; CSCV-PBO into the retrain report; prediction cache (needs an
   injected-feature-aware key first); crypto_trend gate (flag currently a silent no-op — flagged
   for the survivor-#10 sign-off); options_overlay needs an actual runner to produce its
   pre-registered GO/NO_GO verdict; execution_policy pending its edge-floor source ruling.
3. **Declared-but-unwired breadth**: CRYPTO_POOL (6→10 coins), TRADABLE_POOL, SAFE_HAVEN defensives
   (in config, not in the tradable universe — on risk-off days the book has almost nothing
   entrable). Sequenced behind honest crypto costs and the IC-by-name gate.
4. **Missing reconciliation/liveness machinery**: no intra-day crypto orphan sweep (untracked
   crypto positions ride 24/7 stopless between restarts); no graceful drain on SIGTERM (bots are
   killed mid-order twice per retrain); bots fully stop for the Saturday retrain (an exits-only
   skeleton mode would preserve stop management); stop management runs after the once-an-hour 11–28s
   maintenance block; quotes are fetched one REST call per symbol instead of batched.
5. **Missing measurement**: cycle-latency instrumentation (wedges/storms/throttles leave no trace);
   sizing co-fire aggregator over the shipped decomposition journal (the evidence every de-risk
   ruling waits on); sentiment-gate P&L attribution; LLM spend-vs-benefit dollar ledger; promotion
   ledger + post-promotion P&L attribution (the system cannot answer 'have promotions ever made
   money?'); slippage-vs-quote-age decomposition (quote_age_s is journaled, nothing reads it);
   end-of-harvest spread-health summary; source-provenance stamp on training rows; daily EOD P&L
   digest through the existing notify channel.
6. **Missing last-mile reporting**: llm_eval_report.json, llm_advisor_report.json, and
   execution_report.json are produced and read by nothing; the GUI gate box discards the anti-noise
   verdict fields; no reports-freshness strip.
7. **Graded literature survivors not yet built** (kill-list screened): MAX/lottery-demand
   cross-sectional feature; realized semivariance / signed-jump from minute bars (the standing #1
   build from rev-07-02); no-trade band / partial-adjustment sizing; momentum-crash de-risk flag
   (measurement-first). All gated on an indicator lead/lag IC run over real Jetson panels first.
8. **The SPY hedge (low-beta roadmap step 2)** — kill-list survivor #12, sequenced after the beta
   ledger produces real numbers and the deposit-contamination fix lands, with the dormant
   short-cost stack as its honest cost model.

---

## D. Cross-cutting themes

1. **The de-risk multiplier stack is the largest chronic P&L leak.** One reading (VIX) can enter an
   entry's size three times; a kill-listed fake indicator still fires; two vol targets compound;
   twelve advisory multipliers share one floor that erases their differences exactly in stress. The
   whole family needs ONE adjudication (the never-run B7 decision) informed by the co-fire
   aggregator, not twelve independent patches.
2. **Certify ≠ deploy ≠ label — the measurement-validity family.** The holdout gate certifies a
   model (LSTM-only) that never trades; the policy gate replays a book (champion, flat spreads,
   bars-unit lockouts) that isn't the one deployed; labels and meta rows are generated under
   admission rules live never uses; live features (long-window constants, forming bars, dead
   sentiment) diverge from training. Every Sharpe/DSR/b2 number currently describes a different
   book than the one trading. Most fixes are cheap; the discipline is the point.
3. **Instruments that decide money are themselves broken.** Kelly's sample is censored; llm_eval's
   inference is void; the conviction gates pass on absent fields; the journals fabricate neutral
   scores; the regression gate can false-PASS. Fixing measurement FIRST is the campaign's stated
   sequencing and the readers' unanimous conclusion — otherwise every later keep/kill ruling is
   noise.
4. **Selection pressure is unaccounted at every level.** Best-of-3-folds checkpoint shipping,
   cumulative Optuna pools reset by study deletion, weekly re-tests of a ~92%-overlapping holdout
   under a ratchet, 14 unadjusted shadow peeks, A/B samples censored by one-sided provider
   failures. The system's anti-overfitting math is sound locally and undercounted globally.
5. **The activation backlog beats new alpha.** Five consecutive waves concluded the same thing;
   this campaign's readers confirmed it with greps: the highest expected-value-per-effort work is
   wiring what exists (spread stamps, cost-regime features, blend weights, IOC caps, Stage-0 dumps)
   and executing decisions already queued.
6. **Shared-state and two-process hazards.** One Alpaca account, two books, shared flag/lockout/
   novelty/journal files, separate-process defaults, and a doc/flag mismatch about which mode is
   default: several defects (flatten one-book, lockout clobber, novelty clobber, stream conflicts)
   are one design decision — per-book-prefix everything — applied consistently.
7. **The verification stack must be made sound before the build phase leans on it.** ab_check
   hardening, baseline shrink, a CI leg that runs the production pins and lightgbm, and functional
   (not text-pin) coverage of base_loop are prerequisites for shipping the fixes above at campaign
   speed.
8. **Nothing has run on the Jetson.** Committing and executing the runbook converts ~15 owner
   decisions from stalled to evidence-backed. It is the first step of every chain below.

---

## E. Already-queued vs genuinely NEW

**Already sitting in owner-decision queues** (the 90-item module-review queue plus ~150 verbatim
panel deferrals in the batch reports): D01 (fan-out timeout), D02 (both n_eff halves), D03, D04,
D05, D07 (the placeholder P0), D08, D09 (llm_eval fixes were specified by the panel), D10 (VIX
stack, pseudo-CAPE, HMM cut pending), D12, D13, D14, D15, D17, D18, D19, D20, D21, D24 (the
long-only flip), D25, D27, D29, D33, D34, D35 (both halves flagged by batch-0), D39, D40, plus the
below-the-line signal-exit conflict, stop-units mismatch, lockout clobber, OI units, crossed-quote
policy, maker-share weighting, and the get_live_atr forming bar. **The money here is in EXECUTING
the queue in the panels' own dependency order, not re-finding it.**

**Genuinely NEW this campaign** (found in never-panel-reviewed files or paths): D06 (stock TP
Kelly censoring — arguably the biggest new find), D16 (stock partial-fill untracked position), D22
(no final refit / winner's-curse checkpoint), D23 (blend_fit orphaned), D26 (stablecoin halt
floored), D28 (funding z-baseline), D30's live-path deadness (the 60-day vs 45-day arithmetic), D31
(false auto-include comment), D32 (drift churn), D36 (ab_check false-PASS + baseline masking), D37
(CI production-stack hole), D38's panel_ranks/shadow sites, the holdout boundary label leak, the
cross-ticker objective bleed, the LLM budget/pricing registry holes, provider-wide 429, stale-veto
persistence, prompt_ab censoring, run_bots' missing kill switch, journal tearing/retention/
final_notional, notify dedupe-before-send, novelty cross-process clobber, shadow .prev best-effort,
alpaca_compat missing quote timestamp, chart_core annualization split, the Total-P&L baseline
decay, sentiment phrase/word and substring defects, Volume_Ratio infinity, connection_test exit 0,
the git-stash allowlist contradiction, and the session-state/P0-placeholder corpus rot.

---

## F. Kill-list screening record

`research/KILL_LIST.md` was read in full and every reader idea screened. **Nothing proposed
rebuilds a killed item.** Outcomes: (1) the pseudo-CAPE finding is kill-list ENFORCEMENT — a killed
item still running in production (D10). (2) The crypto EDGE stamp is kill-ADJACENT (line 90's
killed 'EDGE-on-hourly inflation fix'); it mirrors the live, sanctioned stock stamp, but ships only
with owner sign-off and the hourly-level reproduction — noting B3's finding that the kill's cited
medians may rest on a percent-vs-basis-points slip (grounds to ask, not to drop the entry). (3)
Adding BTC cross-asset features to the crypto preset and wiring the crypto_trend BTC gate both
touch the UNDECIDED 'BTC-lagged spillover for alts' (commonly-confused survivor #10) — flagged for
explicit sign-off, not assumed alive. (4) The NFP stand-down is an entry gate, distinct from the
killed jobless-claims-momentum FEATURE. (5) The shadow economic-loss arm evaluates MODEL forecasts
and stays on the legal side of the killed 'DM-HLN for policy changes'; the challenger policy-gate
(D03's fix) is the kill-consistent replacement for that category error. (6) The HMM cut executes a
pending rev-07-01 kill recommendation. (7) Funding/basis/squeeze features, HAR-RV, JKX
distillations, RR_5/21, options_overlay-as-instrumentation, and the SPY ETB hedge are all
commonly-confused SURVIVORS and are used as such. (8) All concentration/sizing changes (rank-order
admission, edge-Kelly, meta tilt re-map, min-aggregation) are gated on in-house measurement per the
wave-5 rule — none ship on literature priors.

---

## G. Sequencing skeleton for Phases 2–3 (informing, not binding)

Chain 0 (unblocks everything): commit tree → Jetson runbook one-looks (stock preds non-null;
n_eff_clustered vs n_trades; beta ledger; decision report; bidask presence; spread health).
Chain 1 (make the gate honest): n_eff repair + challenger policy-gate + trial-pool accounting →
meta publish gating → calibration package → OOF preds + replay parity → purged-OOF flip.
Chain 2 (make the costs honest): edge-floor ruling + crypto EDGE stamp + threshold range + study
reset, bundled into ONE re-harvest/retrain with the long-only flip, final-refit, blend-gate, and
preset fixes (one gotcha-#2 event, not five).
Chain 3 (stop the bleed): liveness/safety bundle; Kelly un-censoring; earnings buffers; de-risk
consolidation after the co-fire evidence.
Chain 4 (then go get money): stock feature restoration + HAR activation; breadth + cross-sectional
activation; SPY hedge; literature survivors — each behind its measured gate.
