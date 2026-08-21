# Module improvement campaign — panel review (v3)

This module-improve-v3 panel review covered 3 measurement/reporting-contract modules — `beta_ledger`, `decision_report`, and `llm_eval` — with 5 independent Opus reviewers per module (266 raw findings total: 81 / 86 / 99 respectively). All three modules were adjudicated **approved-after-hardening**: 61 findings were accepted and shipped (23 / 17 / 21) against 18 rejected (6 / 7 / 5); the remainder of the raw findings consolidated into 30 owner-decision items (below) rather than being auto-fixed, since they change estimator definitions, sample semantics, or cross-file/model-facing behavior. No modules were dropped for budget. Three commits shipped, one per module, covering measurement-only bug fixes, perf cleanups, and new instrumentation — nothing was weakened or reverted to make the gate pass. The regression gate (`bash scripts/ab_check.sh`) **passed clean**: zero NEW failures, zero DISAPPEARED failures, same 21-failed/7-error baseline name-set as `tests/baseline_failures.txt`.

## Per-module results

| Module | Raw findings | Accepted | Rejected | Verdict | Bugs fixed | Perf | Modernized |
|---|---|---|---|---|---|---|---|
| beta_ledger | 81 | 23 | 6 | approved-after-hardening | 4 | 1 | 1 |
| decision_report | 86 | 17 | 7 | approved-after-hardening | 2 | 2 | 1 |
| llm_eval | 99 | 21 | 5 | approved-after-hardening | 3 | 2 | 2 |

## Shipped

- `feat: beta_ledger v3 — exact AKL summed-beta HAC t, bounded alignment, bucket floors, data-quality/coverage instrumentation, JSON-safe atomic --json (+38 tests)`
- `fix: decision_report 2026-07b — full-frame ATR counterfactuals, tz-safe dedup, out-of-window exclusion, verdict/bucket floors, atomic stale-aware reports, fetch-once cache`
- `feat: llm_eval v3 — coverage/realization/panel diagnostics, degenerate+ANTI verdicts, calibration denominators, atomic no_data stubs`

## OWNER DECISIONS (deferred — model-facing or policy-semantic)

### beta_ledger
- PIT trend-state conditioning (4 reviewers): state at day t uses SPY's close(t) — the END of the return period it classifies — so SMA-crossing days are sorted into the bucket their own move created. Sketch: state = (px > sma200).where(sma200.notna()).shift(1); rollout by emitting BOTH trend_conditional (current keys, GUI-compatible) and trend_conditional_lagged for one measurement cycle, then pick the canonical one. Estimator-definition change; do not ship from this pipeline.
- HAC finite-sample d.o.f. correction (3 reviewers): ols_hac's sandwich lacks the standard n/(n-k) scaling (statsmodels default), understating every SE by ~3% at n=90/k=5 — anti-conservative on alpha_t specifically. Sketch: S *= n / float(n - k) before forming cov, documented next to the NW-1994 plug-in note. Moves every published t-stat; owner decision.
- ANNUALIZATION_DAYS=252 vs the actual equity grid (4 reviewers): if Alpaca returns calendar days for this crypto-enabled account, alpha_annual/ann_return are understated ~31% and vol/Sharpe ~17%. The shipped period.obs_per_year diagnostic makes this a one-look decision: run `python beta_ledger.py --days 90 --json /tmp/b.json` once on the Jetson and read period.obs_per_year; then decide 252 vs inferred grid (reconcile with chart_core.perf_stats, which derives ann from median sample spacing and documents matching beta_ledger's ddof convention).
- Excess-return (Jensen) alpha (R5): the intercept currently absorbs rf·(1−Σβ) ≈ +2.2%/yr of pure T-bill carry at rf=4.3%/β=0.5. Sketch: --rf-annual flag (default 0.0 keeps today's numbers), regress excess returns, emit NEW keys alpha_annual_excess/alpha_t_excess/sharpe_excess alongside the unchanged raw keys. Estimator definition; also applies to chart_core.perf_stats' matching Sharpe convention.
- Transfer-clean return source (R4): switch load_equity_alpaca to hist.profit_loss_pct (or profit_loss[i]/equity[i-1]) so deposits/resets stop fabricating alpha (measured +4.2%→+43.6%/yr from one +$50k top-up). Jetson check first: print hist.profit_loss[:5]/profit_loss_pct[:5]/base_value for timeframe='1D' to confirm Alpaca populates them. The shipped |r|>15% outlier warning flags contamination in the meantime. NOTE: portfolio.py's live book_vol_scalar consumes the same raw equity — strictly an owner decision there (live sizing path).
- Raise the joint regression's observation floor to max(20, 5·k) so --lags 3 on a short window refuses to fit 9 regressors on ~22 points (2 reviewers). Changes which invocations succeed; the shipped underpowered flag surfaces the condition without refusing.
- Regular-grid reindex before the lag shift so 'lag 1' is a fixed horizon by construction instead of 'previous surviving row' across gaps (2 reviewers). The shipped index_regularity gap stats make the current behavior auditable. Decide together with the PIT trend-state item.
- Equity-clock vs BTC-bar-clock skew (R4, low confidence): Alpaca 1D points are stamped near US close (~20-21:00 UTC) while yfinance BTC-USD daily bars close 00:00 UTC, attenuating BTC's lag-0 beta into lag-1. One-line Jetson check: value_counts of the raw portfolio-history timestamps' time-of-day. Only the summed AKL beta is clock-robust; re-sourcing BTC bars is an input change.
- Hard refusal (nonzero exit) when the SPY benchmark is entirely absent from the loaded set — R5's position; currently shipped as a stderr warning + excluded/degenerate instrumentation so a BTC-only report is at least loudly labeled.

### decision_report
- Ledger #59 — legacy tz-naive journal rows are localized as UTC but were written in box-local time; localizing them to the box's local tz (dt.datetime.now().astimezone().tzinfo) in _replay_grouped and the dedup would re-align pre-2026-07-13 rows by the UTC offset. Numbers-moving reinterpretation of historical data; the crash fix deliberately keeps the UTC convention so both code paths agree.
- Replay horizon should track the deployed forward_bars (Optuna-tuned over [12..48], journaled on llm_analysis rows) instead of the hardcoded MAX_HOLD_BARS=24 (R3). Sketch: read the most recent llm_analysis row per asset_type from the loaded rows, use its forward_bars with fallback 24, emit horizon_bars/horizon_source in the JSON. Methodology change to what the counterfactual measures — owner call.
- Reason-aware horizon-pending guard: a stock EOD-flatten (reason 5) landing on the frame's last bar is a resolved exit but is counted horizon-pending; capturing the discarded reason and exempting code 5 changes sample composition (R3 instrumentation vs R4 unclear — routed to owner). Sketch: `if j == n-1 and int(_reason[0]) == 6 and (n-1) < max_hold: return None`.
- market_data.drop_forming_bar is not applied to replay frames, so exits landing on the forming trailing bar price against a partial bar and are not reproducible run-to-run (R5, 1/5). Sketch: drop_forming_bar(bars) once per symbol after fetch; affected rows become horizon-pending. Sample-defining — owner call.
- Ledger #52 — import rank_gradient.DEFAULT_BUCKETS instead of inline literals: requires rewriting tests/test_review_b16.py::TestBucketParityTripwire (another batch's file, source-text pinned). Coordinate as one change.
- Ledger #60 — full migration of load_journal to trade_journal.iter_journal_rows and pointing JOURNAL_DIR at trade_journal's: the in-place filter in this spec removes the memory risk; the migration changes iteration order and the tests' JOURNAL_DIR monkeypatch seam, so it stays a deliberate follow-up.
- Crypto book journals no entry_rank (base_loop.py:1857 calls _conv_fields without rank=), so the CONCENTRATION_ENABLED rank certification is structurally stock-only. Either rank crypto candidates by pred_return before the gate loop and pass rank=, or amend strategy_config.py:203 to scope the certification to the stock book. Cross-file producer decision.
- base_loop._journal_entry_window returns early when n_candidates <= 0, blinding pct_windows_zero to total prediction outages (the standing 'stock preds None since wave-4' P0 scenario). Sketch: journal the row with admitted_k=0 and veto_counts intact. Cross-file producer decision; this spec ships the in-module disclosure note only.
- Consumers should adopt the new fields: gui.py's gate panel should render verdict/insufficient_n/quality.representative instead of coloring by raw mean sign, and rank_gradient_verdict should refuse buckets below a minimum n. Both files are outside this module's ownership (gui.py is in the separately-owned uncommitted campaign).
- A journaled spread_pct of exactly 0.0 is honored as a zero-spread cost rather than treated as missing (R4 secondary). Changing it alters the cost charged in counterfactuals — cost-semantics owner call.
- After this lands, every previously-taken reading off decision_report.json (including any CONCENTRATION_ENABLED / gate-loosening evidence) is void: the ATR fix and out-of-window exclusion move every published number by design. The banner announces the break; re-run on the Jetson before acting on any figure.

### llm_eval
- HAC/inference redefinition (5/5 reviewers, 3 independent null simulations showing 10-53% false 'keep it' rates): choose (a) lag scaled to rows-per-bar x forward_bars, (b) cluster/Driscoll-Kraay by t0, or (c) collapse to one obs per (symbol, non-overlapping forward_bars block) reusing the repo's existing effective-n discipline (sample_weights.py); then gate MIN_POWER_N on effective-n, not raw rows. Until then no b2 p-value from this module should be quoted as evidence — the shipped diagnostics (effective_n_hint, rows_per_t0, caveat) make this visible.
- Stock horizon semantics (4/5): forward_bars is a BAR count (policy_exits vertical barrier, EOD-capped) consumed as wall-clock HOURS — stock realized returns span ~7 bars, not 24, attenuating the pred control and biasing b2 upward. Fix sketch: `i1 = i0 + horizon` bar-stepping (matches TB labels; identical for 24/7 crypto) behind a horizon_unit param defaulting to current behavior; the shipped bars_spanned/elapsed_hours diagnostics quantify it first.
- Exclude dedup_hit=True rows (and/or down-weight) from the advisor incremental/calibration sample — they are cached re-serves, not observations (4/5). Shipped accounting (n_dedup_hit, dedup_hit_frac, per-sha grouping) sizes the problem; exclusion changes the sample.
- Restrict the advisor incremental regression to p_up-present rows, or make incremental_p_up_only the primary (3/5) — shipped as an additive secondary block; swapping primaries changes the verdict definition.
- Alignment tolerance: reject rows with entry_lag > ~2 bars or elapsed > 1.5x horizon (3/5) — changes which rows enter the verdict; shipped diagnostics count them first.
- Pooled-book regression: add a book dummy / per-book pred columns or suppress the pooled verdict when books mix (R5's 42%-false-keep simulation) — shipped incremental_by_asset + pooled_books flag; changing the pooled estimator is definitional.
- two_by_two_grid: s>=0.5 counts the neutral/parse-default 0.5 as LLM-bull (2/5) — options: strict '>', or a 3-state grid with an explicit neutral band; shipped n_s_exactly_half counter sizes it.
- Newey-West small-sample d.f. correction (n/(n-k), ~2.6% at the n=60 floor, direction favors 'keep') — would move every p-value and requires updating the statsmodels oracle test's use_correction pin in lockstep.
- Economic-significance floor + spend figure: report implied bps/trade at deployed sizing (llm_mult = 0.5 + s) and read llm_client.get_daily_cost() fail-soft, so the keep/kill decision sees both sides of the ledger; gating the verdict on effect size is a decision-rule change.
- Cross-file (report-only, do not edit from this module's pass): (a) base_loop.py llm_analysis rows carry no dedup_hit field and journal missing s as 0.5 instead of null — run_eval can never detect replays or parse failures; (b) scripts/prompt_ab.py should import VETO_THRESHOLD and MIN_POWER_N from llm_eval instead of its own literals (three-way manual sync today); (c) a true veto P&L counterfactual requires joining skip_reason=='llm_veto' / buy(llm_multiplier, final_notional) / sell(pnl_pct) rows already in the same journal.

## Verification

Ran `cd /Users/kywwilson/Desktop/Projects/trader && bash scripts/ab_check.sh`. Verbatim verdict:

```
ab_check: suite summary: ====== 21 failed, 2177 passed, 16 skipped, 2 warnings, 7 errors in 14.49s ======
NEW failures: none
DISAPPEARED failures: none
ab_check: PASS — zero regressions vs tests/baseline_failures.txt
```

Exit status: PASS (exit 0). No NEW failure names exist, so there is nothing to attribute to beta_ledger, decision_report, or llm_eval. Raw pass/skip counts (2177 passed / 16 skipped) differ slightly from the older CLAUDE.md figure (1887 passed / 15 skipped, 2026-07-15) — expected drift from the suite growing since then (new untracked test files such as test_journal_stats.py, test_tax_lots.py, test_design_tokens.py are present per git status). ab_check.sh judges by FAILED/ERROR test NAMES against `tests/baseline_failures.txt`, not counts, and confirmed the identical 21-failed/7-errored name set with no additions or removals. No files were edited for verification and the baseline was not regenerated.

**Standing reminders:** nothing in this campaign has been committed — the three commit_lines above are proposals for the owner to review, stage, and commit (or reject), alongside deciding the 30 owner-decision items listed above. And per the two-machine split, this dev Mac's verification is bounded by its installed deps (no torch/lightgbm/optuna/joblib/numba/sklearn/dotenv/alpaca/finnhub/PySide6) — `ab_check.sh` above is the full extent of what could be confirmed here; anything gated on the missing deps remains unverified until it runs on the Jetson stack.

Generated by module-improve-v3 (Opus panel -> Fable spec -> Sonnet implement -> Fable harden).
