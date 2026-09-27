# FIX_D — measurement-only fixes (2026-09-26, Jetson)

All edits are measurement-only and touch no order path. Nothing was committed. Raw outputs are in `scratchpad/FIXD_runs/`.
The report JSONs the re-runs wrote to the repo root (`decision_report.json`, `execution_report.json`, `llm_eval_report.json`) were moved there too.

## Changes (file → what)
1. **decision_report.py**
   - `conviction_calibration` now counts buys that have a symbol but `pred_return: null` in a new `_dropped_null_pred` key, on every return path. They are still not replayed.
   - `run_report` adds them to the unpriced denominator and adds `quality.dropped_null_pred`. `quality.horizon_pending` now subtracts them.
   - The existing WARNING line now names the null-pred count, and a new banner line follows it.
   - The stock-window NOTE uses a new constant `STOCK_FRAME_MIN_DAYS = 26`: 320 bars ÷ ~16 extended-hours bars per day ≈ 20 trading days, so ≥26 calendar days (~30 observed). The NOTE fires at `days > 26` and now says what actually binds: `fetch_stock_bars_alpaca .tail(320)`, not the 45-day fetch start.
2. **beta_ledger.py** adds `drop_glitch_days(equity, profit_loss, max_move=0.50, revert_days=3)`.
   - **Rule:** drop a day when equity leaves the band [last_good×0.5, last_good×2] and comes back inside it within 3 days. A crash that never reverts is kept.
   - **P&L handling:** the dropped days' `profit_loss` is folded into the next kept day, so the transfer-clean leg sees the true 2-day P&L (−83,180 + 82,683).
   - **Why not the "equity == cash" rule:** portfolio history has no per-day cash, so it cannot be applied historically.
   - **CLI:** `--drop-glitch-days` / `--no-drop-glitch-days` (ON by default), plus `--glitch-max-move` and `--glitch-revert-days`. Dropped dates are printed, put first in the warnings, and stored in `data_quality.glitch_days_dropped` and `data_quality.drop_glitch_days`.
3. **execution_report.py**
   - In the notional block, a buy with neither `maker_notional` nor `entry_tactic` is now `unknown`. It goes to `crypto_unknown_tactic_notional`, not into the taker denominator. With no known tactic at all, the block prints "n/a" and emits no share.
   - `_write_json` is now atomic: a pid-unique tmp file plus `os.replace`, with cleanup. It takes an optional `out` path.
4. **scripts/sizing_cofire_report.py**
   - `--json [PATH]`: a PATH writes the file atomically (then prints the human report). A bare `--json` (or `--json -`) still goes to stdout, which keeps `tests/test_c26_S3.py` and existing users working.
   - Argparse errors now exit 2 and `--help` exits 0. An unexpected runtime failure (for example an unwritable PATH) exits 1. The docstring was updated.
5. **llm_eval.py**
   - `_write_report` writes through `tempfile.mkstemp` in the same directory, then `os.replace`, with cleanup.
   - The n<60 verdict now also says ≥120 clusters and effective_n ≥20.
   - After `Verdict:`, the CLI prints the full power floor with this run's n, cluster count and effective_n.
   - The module docstring now states that n ≥ 60 alone is not enough. The power gate logic itself is unchanged.
6. **gui.py `_run_gap_audit_clicked`** (one surgical replace; the other hunks in gui.py's diff belong to the concurrent GUI agent). It now passes only the names the sleeve could hold:
   - **Pool:** `load_stock_universe()` minus any name containing '/', which is StockLoop.get_symbol_universe's own rule, minus `stock_config.LEVERAGED_ETFS`, the sleeve's `_leveraged_etfs` exclusion.
   - **Ranking:** by `stock_predictions.json` pred (the sleeve's primary key), names ≥ `OVERNIGHT_SLEEVE_MIN_PRED` first.
   - **Cap:** `OVERNIGHT_SLEEVE_MAX_POSITIONS` (2).
7. **chart_core.gate_panel_model**: a stale stub with `api_available is None` now reads "no journal rows in the report window — nothing to price". `False`, or the key missing, still reads "no API when generated — counterfactuals not priced", which `tests/test_c26_U1.py` pins.
8. **Docs (one-line edits):** the n≥60 claim is replaced with "≥120 distinct hourly t0 clusters and n_eff ≥ 20, ≈20+ days of LLM cycles". Edited lines:
   - CLAUDE.md (llm_eval row)
   - docs/MAP.md (3 places: :608, :773, :1019)
   - docs/GLOSSARY.md:60
   - docs/MODULES.md:778
   - 03_jetson_runbook.md:44

   The sizing_cofire `--json [PATH]` row was updated in docs/MODULES.md Appendix A and scripts/README.md. The beta_ledger Appendix A row gained the glitch flags.
9. **New test:** `tests/test_measurement_fixes_2026_09.py`, 21 tests, all Mac-safe.

## Verification
- `py_compile` passes with $JPY on every touched .py file.
- Targeted tests: **354 passed**. The files run were:
  - test_decision_report, test_decision_report_v3
  - test_beta_ledger, test_beta_ledger_v3
  - test_llm_eval, test_llm_eval_v3
  - test_grp_reports, test_chart_core, test_gap_audit
  - test_c26_S3, test_c26_U1
  - the new file
- A further **65 passed** in test_c26_T7, test_grp_ops and test_review_b17, which reference execution_report and gap_audit.
- There is no test_execution_report*.py. The full suite was not run.

## Real-journal re-runs (--days 200)
**beta_ledger** (the headline figure, before → after):
```
before (--no-drop-glitch-days): strategy: +127789.0%/yr at 106275.0% vol (Sharpe 1.20); alpha +122396.1%/yr (t +1.01); R^2 0.00
after  (default):  [beta_ledger] dropped 1 glitch day(s) (spike beyond 50% that reverted within 3d): 2026-08-13
                   strategy: +37.8%/yr at 39.1% vol (Sharpe 0.97); alpha +44.4%/yr (t +1.08); R^2 0.30
                   beta[BTC] summed(AKL) +0.623 (t +3.51); CLEAN alpha +44.4%/yr (contamination delta -0.0%/yr)
```
**decision_report:**
```
WARNING: 99% of rows unpriced (0 fetch failures, 23 out-of-window, 153 null-pred buys) — sections below are NOT representative
WARNING: 153 buy row(s) dropped from conviction calibration for pred_return=null (cannot be bucketed by prediction; counted as unpriced above)
NOTE: stock replay frames hold only the last 320 hourly bars (market_data.fetch_stock_bars_alpaca .tail(320), extended hours included) ≈ 26-30 calendar days — older stock rows are counted _out_of_window, not priced.
quality: {'priced': 2, 'unpriced': 176, 'out_of_window': 23, 'dropped_null_pred': 153, 'unpriced_rate': 0.989, 'representative': False}
```
The banner used to say 92% over 25 rows. It now says 99% over 178.

**execution_report**, before and after:
```
before: Crypto maker NOTIONAL share: 0.0% of $260,472 entered
after:  Crypto maker NOTIONAL share: n/a — all $260,472 of crypto entries carry no entry_tactic/maker_notional (pre-maker-ladder journals)
```
**sizing_cofire_report:**
- `--json <path>`: exit 0, and the file is written (`FIXD_runs/sizing.json`).
- `--bogus`: **exit 2**, where it used to exit 0.

**llm_eval:** there are still no `llm_analysis` rows, so the stub path is unchanged and the new power-floor line cannot print on real data. The new test covers it.

**gap_audit in the GUI form:** I ran the same selection logic on the real files. Pool = 44 stock names out of 56 (crypto and TQQQ/SOXL removed), and the chosen names are CRSP and MRVL:
```
SLEEVE TOTAL: forfeited_drift=$4,546/yr  gap_through=$4,732/yr     (was $110,591 / $125,901 over 56 names)
```

## Caveats / not done (outside my ownership)
- `stock_predictions.json` is from 2026-05-07, so the GUI's top-2 pick uses stale preds. When the file has no preds it falls back to universe order. The sleeve's real pick is dynamic (held positions × preds × ON_Mom); this is a representative 2-name audit.
- `gap_audit.py:170-181` still sums whatever it is given. It was not edited because I don't own it.
- Not asked, left open:
  - sizing_cofire `:97` header ("0 buy rows" when 177 buys lack `sizing`)
  - execution_report's silent omission of the shortfall section when buys have no `slippage_bps`
  - rank_gradient_report ignoring `quality.representative`
  - no producer for reliability_report's input
