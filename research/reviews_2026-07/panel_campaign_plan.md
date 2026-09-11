# Panel-review campaign plan (module-improve-v3) — 2026-07-22, re-paced 2026-07-26

Batches of **3 modules × 5 Opus reviewers (~29 agents each)**. Re-paced down from 5-module batches
after batch A's final two agents died on a session limit: a 5-module batch runs ~5.8M subagent tokens
in one shot, a 3-module batch ~4M. **No module was dropped** — the same coverage is spread across more,
smaller runs.

**Runs must stay sequential** — each batch ends with a serialized full-suite `ab_check`, and a
concurrent batch's implementers writing files mid-suite produce phantom failures.

Launch procedure per batch:
1. Copy that batch's config over `.claude/workflows/modules-v3.run.json` (args are not delivered to
   workflows — the file is the only channel).
2. `Workflow({scriptPath: ".claude/workflows/module-improve-v3.js"})` — **never by name**
   (name-resolution serves a stale cached script).
3. When it finishes: independently re-run `bash scripts/ab_check.sh` and the batch's new
   `tests/test_*_v3.py`, then write the next batch's config (seeds should absorb what this one found).

## Status

| # | Batch | Modules | Report | Status |
|---|---|---|---|---|
| 0 | Measurement contracts | decision_report, llm_eval, beta_ledger | `module_improve_v3_report.md` | ✅ 266 findings, 61 shipped, 30 owner decisions, gate PASS |
| A | Live engine & order path | base_loop, order_utils, execution_policy, risk_budget, bet_sizing | `..._batchA.md` | ✅ 384 findings, 67 shipped, 36 rejected, **54 owner decisions**; gate re-run manually = PASS, 142 new tests pass (the workflow's own gate+report agents died on a session limit; report reconstructed from the surviving output, all deferral lines diff-verified verbatim) |
| B1 | Validation core | validation, sample_weights, calibration | `..._batchB1.md` | ✅ 191 findings, 33 shipped, 26 rejected; gate PASS (workflow's own + independent re-run), 71 new tests pass. **P0 found: `clustered_effective_n` is a connected-components count, not independent draws** |
| B2 | Promotion gate | meta_label, backtest, policy_exits | `..._batchB2.md` | ✅ 224 findings, 35 shipped, 20 rejected; gate PASS, 105 new tests. **policy_exits kernel proven bit-identical vs HEAD across 12,000 synthetic invocations** (docs-only, as fenced). **rev-07-01 in-sample-primary leak CONFIRMED still open (5/5)** |
| B3 | Cost model | fees, liquidity, cost_regime | `..._batchB3.md` | ✅ (2nd attempt; 1st aborted on a session limit with zero footprint) 190 findings, 35 shipped, 22 rejected; gate PASS, 88 new tests. **Crypto book was never EDGE-stamped at all (5/5)**; **`pct_change` default differs pandas 2 vs 3 → different Amihud per machine (5/5)** |
| B4 | Portfolio risk | portfolio, ~~drawdown~~, portfolio_backtest | `..._batchB4.md` | ⚠️ **2 of 3 done** — 160 findings, 32 shipped, 19 rejected; gate PASS, 49 new tests. **`drawdown` chain DIED** (a reviewer exceeded the StructuredOutput retry cap) — zero footprint, rescheduled into B7 |
| B5 | Model artifacts | model_v2, model_lgb, blend_fit | `..._batchB5.md` | running |
| B6 | Prediction path | predict_now, prediction_cache, market_data | `..._batchB6.md` | queued |
| B7 | **De-risk multiplier stack** | drawdown, macro_indicators, regime_detector | `..._batchB7.md` | queued — see note below |
| B8 | Feature kernel & panels | indicators, volatility, panel_ranks | `..._batchB8.md` | queued |
| B9 | Diagnostics & journals | rank_gradient, ic_diagnostic, trade_journal | `..._batchB9.md` | queued |
| B10 | Data & ops | monitor_drift, data_sources, notify | `..._batchB10.md` | queued |
| B11 | LLM + remaining measurement | llm_client, execution_report, gap_audit | `..._batchB11.md` | queued |

**B7 was deliberately re-composed.** `drawdown` needed a re-run anyway, and four separate panels have
now found pieces of ONE question — how many independent de-risk multipliers co-fire on the same
event? Batch A: VIX is read three times (base_loop `f_vix`, `macro_indicators.sizing_mult` on the SAME
breakpoints, plus a `kelly_mult` clamp) compounding to 0.09 at VIX>35, below the 0.1 floor. B3:
`cost_regime`'s 3-tier VIX feature vs the live 4-tier ladder. B4: `PORTFOLIO_VOL_TARGET` applied twice
(volatility.py per-position × portfolio.py `f_bookvol`), worst case 0.25×. And `dd_mult` is a
candidate further co-firing term. Putting `drawdown` + `macro_indicators` + `regime_detector` in front
of ONE panel lets it adjudicate the whole stack instead of adding a fifth partial answer.

**40 modules total** — 8 reviewed, 32 remaining across 11 batches (10×3 + 1×2).

Two modules were **added** to the original plan on evidence, not scope creep: `macro_indicators`
(batch A proved its `sizing_mult` VIX ladder is the second of three co-firing VIX reads — it cannot be
resolved without reviewing it) and `ic_diagnostic` (a crucial measurement module dropped from the
first draft by oversight).

## Off-limits to every batch (uncommitted, under separate owner review)

- **GUI campaign:** `gui.py`, `chart_core.py`, `crypto_loop.py`, `stock_loop.py`, `llm_analyst.py`,
  `run_pipeline.py`, `shadow.py`, `tests/test_chart_core.py`, + new `tax_lots.py`,
  `journal_stats.py`, `design_tokens.py`, assets `fonts/`, `logos/96/`.
- **Panel batch 0:** `decision_report.py`, `llm_eval.py`, `beta_ledger.py` + their `tests/test_*_v3.py`.
- **Panel batch A:** `base_loop.py`, `order_utils.py`, `execution_policy.py`, `risk_budget.py`,
  `bet_sizing.py` + their `tests/test_*_v3.py`.

Agents may READ these (working-tree version — `HEAD` is stale) but never edit, revert, or stash them.

## Findings that must propagate into later batches' seeds

- **Batch 0 → B4:** `portfolio.py`'s live `book_vol_scalar` consumes the same raw Alpaca equity series
  that `beta_ledger` proved is deposit-contaminated (one +$50k top-up fabricated +4.2%→+43.6%/yr).
  Live sizing path — strictly an owner decision there.
- **Batch 0 → B1:** `llm_eval`'s overlapping-observation inference bug (10–53% false "keep it" rates,
  5/5 reviewers + 3 null simulations). Check whether the repo's effective-n discipline is applied
  consistently wherever overlapping windows are regressed.
- **Batch A → B7:** the confirmed VIX triple-read. `macro_indicators.sizing_mult` and
  `base_loop.f_vix` use the SAME breakpoints on the SAME reading and both multiply into tilt → 0.09 at
  VIX>35, below the 0.1 floor, making every other advisory multiplier inert precisely in a crisis.
- **Batch A → B4/B5:** the edge-Kelly cap-space error (f\* is notional leverage, `KELLY_CAP` a risk
  fraction) is a units mismatch class — watch for the same confusion anywhere leverage and
  fraction-at-risk meet.
- **Batch 0 → B6:** `market_data.drop_forming_bar` is not applied to `decision_report`'s replay frames;
  review whether other consumers have the same gap.
- **B1 → B2 (P0, routed):** `sample_weights.clustered_effective_n` counts CONNECTED COMPONENTS of the
  overlap graph, not independent draws — non-monotone, and it collapsed 521-1080 realistic crypto
  trades to 1-12 clusters. It is live in `backtest.py --gate` (weekly retrain, rc=3 = rollback) and
  hypersearch's holdout gate, and B1's R4 showed the per-ticker `effective_n` branch there is a
  structural no-op — so **~100% of the promotion gate's overlap correction flows through the defective
  function**. Owner's one-command Jetson check: read `n_eff_clustered` vs `n_trades` in
  `backtest_report.json` / `backtest_stock_report.json`; `n_eff_clustered < n_trades/5` means the
  clustering, not the model, is deciding promotions.
- **B1 → B2 (activation blocker, routed):** `calibration.IsotonicCalibrator`'s tie-collapse takes the
  tie group's MAXIMUM and is order-dependent (can calibrate a true-10% bucket to p=0.90). Legacy
  sklearn pools ties correctly, so flipping `META_CALIBRATION_MODE='purged_oof'` — wave-9 activation
  item #1 — changes the ALGORITHM and the DATA at once. **Fix the tie handling before running that
  A/B or the result is uninterpretable.**
- **B1 ledger correction:** the module-review entry claiming a "missing guard at calibration.py:115"
  is STALE — the unique-scores guard has existed since `ca00b16`.
- **A + B2 → B3 (three-way convergence, routed):** three independent panels have now hit the SAME
  unresolved question from different sides — what exactly is "the edge floor"?
  (i) batch A/execution_policy: `round_trip_cost_pct` (raw) vs `required_edge_pct` (~3.0x raw, which
  is what `backtest.py:222` actually binds); if raw is chosen, `EXEC_EDGE_HEADROOM_MULT` must exceed
  `MIN_EDGE_MULTIPLE=2.0` or the headroom branch is vacuous for every live-admitted trade.
  (ii) batch A: backtest feeds the FLAT `SPREAD_PCT` while live feeds the per-quote spread — the same
  function answering two different questions. (iii) batch B2/meta_label: `_gen_meta_rows` omits the
  cost/edge floor entirely (order-of-magnitude gap for crypto vs the 0.5x threshold), so the meta veto
  trains on trades live can never take. B3 must return ONE coherent owner decision covering all three.
- **B2 standing defects (owner queue):** the rev-07-01 **in-sample-primary leak is confirmed still
  open** (5/5 reviewers — `_load_artifacts` scores exactly the pre-cutoff window the primary trained
  on, so both the `pred` feature and the row-selection filter are in-sample); the DEFAULT `legacy`
  calibration branch has **none** of `fit_calibrator`'s degenerate guards, so a one-class calibration
  tail can publish a constant p=0.0 that vetoes the entire book (or ~1.0 that boosts everything), and
  `shadow.py`'s post-promotion background retrain reaches live with **no gate at all**.

## B3 findings that need an owner call (beyond the batch report)

- **The per-name EDGE cost model was never applied to the crypto book (5/5).**
  `scripts/harvest_stock_data.py:120` is the repo's ONLY `Eff_Spread_Pct` stamp site, and the
  consumer's `if 'Eff_Spread_Pct' in columns` guard fails silently — so crypto (0.50% fee load, the
  widest per-pair spread dispersion) has been running the flat 0.10% haircut that wave-6 built EDGE
  to replace, with no log ever emitted.
- **Cross-machine numerical divergence (5/5).** `amihud_illiq` inherits `pct_change`'s version-dependent
  `fill_method` default: pandas 2.x (Jetson) pads NaN closes, pandas 3.x (this Mac) does not — the same
  OHLCV yields different Amihud values per machine. Zero callers/artifacts today, so it is free to pin
  now. The panel recommends a repo-wide sweep of ~30 other unpinned `pct_change` sites plus a pandas
  upper bound in `requirements-jetson.txt`.
- **A KILL_LIST rationale may rest on a unit error — owner decision, nothing rebuilt (3/5).** Two
  reviewers independently reproduced the shipped EDGE estimator on 60d/1h bars at NVDA ≈ **0.30 percent
  (30 bps)** median. The wave-7 refutation recorded at `research/KILL_LIST.md:90` cites "0.3" with the
  unit label as bps, and 25.3/0.3 = 84x falls inside its own quoted "~70-95x" discrepancy — consistent
  with a percent/bps slip in the *rationale*, not in the code (a DGP control showed the estimator
  itself is accurate). 30-second Mac reproduction is in the batch report. Per repo rules the entry
  stays killed until the owner decides otherwise; the panel proposed no revival.
- **Before any Jetson `pip install`:** run `python -c 'import bidask'` there first. If it is absent,
  every `Eff_Spread_Pct` in the current training parquet came from the several-fold upward-biased AR
  fallback, and installing it silently changes stamped costs on the next harvest — that is a
  model-facing change requiring re-harvest + the promotion gate, not a routine dependency install.

## Expected shape of results

Modules deep in the model path (**B7 above all**) should produce **mostly owner deferrals, not shipped
code** — nearly everything there changes feature values, which is model-facing by definition. A batch
that ships little but returns a sharp deferral list is a success, not a failure.
