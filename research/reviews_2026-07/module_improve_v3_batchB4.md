# Module improvement campaign — panel review (v3)

Batch B4 ran a 5-reviewer-per-module panel review on the two portfolio-risk modules — `portfolio.py` (book vol scalar, correlation gating, ENB sizing) and `portfolio_backtest.py` (policy backtest / conviction-gated & edge-weighted engines) — under the campaign "feat: panel-review batch B4 — portfolio risk (book vol scalar / drawdown / portfolio backtest)". Both modules were adjudicated **approved-after-hardening**: `portfolio.py` shipped provenance/instrumentation and fail-closed hardening only (13/69 raw findings accepted, 8 rejected, zero bug/perf/modernization changes — every substantive finding was deferred to the owner as model-facing or policy-semantic), and `portfolio_backtest.py` shipped 2 real bug fixes plus 3 perf improvements and 3 modernization/dedup changes (19/91 raw findings accepted, 11 rejected). No modules were dropped for budget. The zero-regression gate (`scripts/ab_check.sh`) **PASSED** — zero new failures, zero disappeared failures — so this batch is regression-clean on the dev Mac.

## Per-module summary

| Module | Raw findings | Accepted | Rejected | Verdict | Bugs fixed / perf / modernized |
|---|---|---|---|---|---|
| `portfolio.py` | 69 | 13 | 8 | approved-after-hardening | 0 / 0 / 0 |
| `portfolio_backtest.py` | 91 | 19 | 11 | approved-after-hardening | 2 / 3 / 3 |

## Shipped

- feat: portfolio.py v3 — corr/vol provenance instrumentation, lazy LedoitWolf sentinel, fail-closed ENB guards (+18 tests)
- feat: 2026-07 B4 portfolio_backtest — fail-loud panel contract, DSR provenance + weighted-arm certification, vectorized panel build (60 tests)

## OWNER DECISIONS (deferred — model-facing or policy-semantic)

### portfolio.py

- Vol-scalar input series is raw deposit-contaminated account equity (5/5, the routed batch-0 item): switch returns to profit_loss/(equity-profit_loss) — alpaca_compat._shim_portfolio_history already carries profit_loss through — or pass Alpaca's cashflow-exclusion params, or winsorize |ret| at OUTLIER_DAILY_RETURN=0.15. All three change live sizing on both books -> challenger/shadow path. The shipped outlier warning + provenance line produce the before/after evidence.
- Correlation estimator pin (5/5): production runs pairwise LedoitWolf (shrinks corr toward zero 14-65%, worse with unequal vols — a true-0.8 pair can pass the 0.7 gate that the tested corrcoef path rejects ~90% of the time); every test exercises corrcoef. Recommend dropping per-pair LW for np.corrcoef (p=2, T=30 needs no conditioning) or recalibrating MAX_AVG_CORRELATION to the LW scale (~0.55-0.60). KILL_LIST 'Ledoit-Wolf shrinkage' [wave-2, wave-6, econ-07] naming overlap is unadjudicated — owner must rule whether this pairwise use is the killed technique.
- Degenerate-series policy (3/5): a halted/frozen symbol currently enters the matrix as 0.0 (perfect diversifier, full size). Sketch: omit zero-variance pairs from the matrix (read as unknown) or reject rets.std()==0 symbols in get_returns_for_symbols. Changes live gate/sizing values.
- avg_book_correlation missing-pair semantics + the three-way inconsistency (5/5): gate/sizing average missing pairs in as 0.0, avg_book_correlation excludes them, and a zero-coverage non-empty matrix returns 0.0 past base_loop's truthiness guard (~2.9x ENB budget inflation at 5 positions). Sketch: one shared missing-pair policy; have avg_book_correlation return None (or take default=0.5) on zero coverage so base_loop's stated 0.5 prior is reachable; both call sites (base_loop.py:750, 1713) change together.
- Vol-scalar scope mismatch (4/5): per-BOOK target (0.35/0.18) divided by ACCOUNT-level realized vol — crypto de-risk structurally near-dead (needs >100% invested to trip), stock absorbs crypto vol at ~1.94x, and PORTFOLIO_VOL_TARGET is applied twice (volatility.py per-position vol_mult x portfolio.py f_bookvol, worst case 0.25x) plus f_corr — the recorded 'VIX double-count' defect class. Sketch: one scope owns the target (account-level target, per-book realized from journals, or divide realized by deployment fraction); decide jointly with the KILL_LIST 'strategy-level vol-targeting vs defensible book-level version' overlap.
- _book_vol_cache failure policy (5/5 flagged): a transient API error caches neutral 1.0 for the full 1h TTL (write outside the try), opposite to the sibling corr cache's success-only policy — fail-open in a live path. Sketch: cache success only, or (scalar, ts, ok) with a short (~120-300s) failure TTL. Changes when the scalar can drop below 1.0 -> model-facing.
- f_bookvol is composed INSIDE base_loop's TILT_MAX-clamped tilt product, so the full 0.5x cut can be absorbed on exactly the high-conviction candidates (1/5, mechanism verified; strategy_config already states this hazard for edge-Kelly). Sketch: apply as a post-clamp multiplier on sized, like the leverage divisor. Absorption is measurable offline today via detail['tilt_raw'] vs detail['tilt'].
- abs() in gate/sizing treats a -0.8 hedge as a +0.8 clone (3/5) — blocks the first hedge instrument the recorded low-beta roadmap (SPY hedge) would add, and conflicts with risk_budget's signed rho_cross convention. Sketch: gate on SIGNED average; keep the sizing formula on max(0, rho_bar) so 1/sqrt(1+n*rho) stays defined.
- Equity-curve hygiene bundle (3/5, decide together with the input-series item): `if e` admits negatives/near-zero/NaN (one 0.01 point -> absurd vol -> 0.5 floor for an hour) and gap-splices dropped points into fake 1-day returns; the EWMA seed double-counts the window (lam**n weight, 54% at the 10-return minimum) and 11 points suffice to halve live sizing. Sketch: adopt beta_ledger.py:509's finite-and-positive filter verbatim, make returns timestamp-aware, seed from the first k returns, raise the minimum to ~30.
- 252-vs-365 annualization (3/5): _TRADING_DAYS=252 vs volatility.BARS_PER_YEAR['crypto']=8760 (365d) for the same book; settle from the new [BOOK-VOL] spacing/weekend log line or beta_ledger's obs_per_year, then flip if warranted (rescales the crypto scalar ~0.83 -> model-facing).
- Cross-file, for the base_loop/stock_loop owners (5/5): journal avg_corr into _journal_skip on rejects and into the sizing detail on accepts — in BOTH independently-maintained _execute_buys copies (base_loop.py:2042, stock_loop.py:833) in lockstep; count book-vol blindness in the degraded-inputs clamp (base_loop.py:1674); consider ordering _update_correlations after _manage_stops (the 56-symbol serial rebuild can block the stop path 11-28s once an hour).
- MAX_AVG_CORRELATION=0.7 lives outside strategy_config.py, the declared policy source of truth (1/5): move it (value byte-identical) or cross-reference it there; note the estimator noise floor at window=30 (SE ~0.18 near r=0) makes accept/reject near the threshold materially noisy — window change is model-facing.

### portfolio_backtest.py

- conviction_gated missing-field semantics (5 findings' P0): a SET meta_floor/ratio_floor is silently satisfied by an ABSENT field, and the documented production panel ({ts,symbol,signal,fwd_return}) never carries the fields — the flagship gate would read 'conviction changes nothing' on a floor that never ran. Sketch: change the defaults at portfolio_backtest.py:53/55 from 1.0 to float('nan') so absent==NaN==fail-closed under the existing `not (x >= floor)` idiom (3 reviewers' preference), OR raise 'meta_floor set but no candidate carries meta_p — build the panel with extra_cols=["meta_p"]'. Either changes admitted sets; also wire extra_cols=['meta_p','pred_thresh_ratio'] into scripts/rank_gradient_report.py when a conviction panel is intended.
- run_policy cost basis (5/5): charge cost on the equal-weight book's WEIGHT turnover — cost_pct * sum_s max(1/k_cur - w_prev(s), 0) — the formula already used at line 261; provably reduces to today's len(new)/k at constant K (incumbent top_k numbers unchanged byte-for-byte), and re-prices only dynamic-K policies. The shipped weight_turnover key quantifies the currently-uncharged component per run.
- hit_rate redefinition (5/5): make the headline hit_rate the INVESTED-period definition (mean(nets[k>0] > 0), None when never invested) — every reviewer independently concluded this; the calendar version stays reconstructible from pct_periods_cash. hit_rate_invested ships now as an additive key; retiring/redefining the existing key is the owner's call.
- CONVICTION_K_MIN support (3 reviewers, semantics split): conviction_gated has no k_min parameter, so the declared Statman floor (strategy_config.py:216) cannot be expressed and the gate certifies a more aggressive policy than the one that would deploy. Owner must choose the set-behavior: (a) fail-closed — admit NOTHING when fewer than k_min clear the floors (R3's position, matches the module's posture), or (b) backfill from the top of the sorted rejects up to k_min (R1/R2). Deferred precisely because the panel itself split on the semantics.
- Overlapping-panel accounting (R4): on the every-bar-sampled fb-bar-fwd_return panel the module documents as its primary input, gross is credited ~fwd_bars times while entry cost is charged once and sharpe annualizes at 1-bar spacing (measured cost-drag 3.7% vs 26.8% of gross for the same strategy sampled non-overlapping). Sketch: give run_policy/compare a fwd_bars argument and replay every fwd_bars-th period (also retires the n_eff division), or scale charged cost by fwd_bars and annualize by periods_per_year/fwd_bars. The shipped fwd_bars_defaulted flag is the visibility half.
- net_total is an arithmetic sum, flattering high-vol concentrated books (R1's verified sign flip via vol drag): add net_compounded — but ONLY after a units contract (percent vs fraction) is plumbed, since prod(1+r) vs prod(1+r/100) differ; consider making the compounded figure the headline for concentration A/Bs.
- edge_proportional_weights NaN-with-negative-floor (R2): the finite-mask runs BEFORE the floor subtraction, so with floor<0 a NaN signal receives positive weight (verified [0.4545,0.5455]). Sketch: e = clip(raw - floor, 0, None); e = where(isfinite(raw), e, 0.0). Changes weights => deferred; behavior now pinned by test.
- run_policy_weighted cannot express gross exposure < 1 (R5): weights always renormalize to a fully-invested book, and the no-positive-edge fallback equal-weights FULL size across the names the floor just rejected — so the engine cannot measure edge-Kelly's central de-grossing claim. Sketch: a normalize='cap' mode (weights = clip(edge/scale, 0, cap), un-renormalized) with the shipped avg_gross_exposure reporting. NOT a vol-target overlay (KILL_LIST.md:98 boundary noted).
- Making fwd_bars a required positional of compare_deflated (fail-closed API): breaks three existing call sites; the shipped fwd_bars_defaulted flag is the compatible half.
- Redefining/renaming compare_deflated's 'turnover' (per-name count) to the book-fraction measure or names_traded_per_period (R2/R3): the shipped avg_entry_fraction_delta + docstring note carry the information; renaming the existing key is a report-contract change.
- Terminal-book exit flush (R5): n_exits_total += len(prev) after the loop would make entries==exits by construction but changes reported exits/turnover; the convention ('exits count only within-panel transitions') is now documented and pinned instead.
- Time-based signal lag + lagging extra_cols (R2/R3/R5): the row-based shift is now documented and pinned (always backward — PIT holds — but a gapped ticker gets unbounded extra lag, and extra_cols are never lagged, so a lagged conviction A/B currently gates on one bar of look-ahead the incumbent lacks). Sketch: lag_cols=... shifting named extras with the same per-ticker shift, and/or reindex-to-bar-grid / merge_asof-with-tolerance shifting. Model-facing: changes the panel values the gate scores.

## Verification

Gate: `scripts/ab_check.sh` — **PASS**, 0 new failures, 0 disappeared failures.

Ran `cd /Users/kywwilson/Desktop/Projects/trader && bash scripts/ab_check.sh` (read-only, no edits made, baseline not touched).

Verbatim output:
```
ab_check: running python3 -m pytest tests/ --continue-on-collection-errors -q ...

ab_check: suite summary: ====== 21 failed, 2632 passed, 18 skipped, 3 warnings, 7 errors in 15.22s ======

NEW failures: none

DISAPPEARED failures: none

ab_check: PASS — zero regressions vs tests/baseline_failures.txt
```
Exit code: 0.

Verdict: PASS — zero regressions. NEW failures: none. DISAPPEARED failures: none.

Note: total passed count (2632) is higher than the CLAUDE.md-documented 2026-07-15 baseline snapshot (1887 passed) — expected, since several new test files exist in the working tree (test_command_ack.py, test_design_tokens.py, test_journal_stats.py, test_llm_dossier_persist.py, test_prediction_cache_context.py, test_shadow_status_persist.py, test_tax_lots.py) that add passing tests. This doesn't affect the verdict: ab_check.sh diffs FAILED/ERROR test NAMES only, never counts, against tests/baseline_failures.txt, and confirmed no new names and no disappeared names.

Since there are 0 NEW failure names, there is nothing to attribute to portfolio or portfolio_backtest — neither module introduced any new failing/erroring test in this run.

Nothing in this batch is committed — per project convention the owner reviews and commits everything. Verification above ran on the dev Mac only (numpy/pandas/scipy/pytest); this campaign did not exercise the full Jetson stack (torch/lightgbm/optuna/joblib/numba/sklearn/dotenv), so anything gated on those deps is confirmed only as far as this dev Mac allows.

*Generated by module-improve-v3 (Opus panel -> Fable spec -> Sonnet implement -> Fable harden).*
