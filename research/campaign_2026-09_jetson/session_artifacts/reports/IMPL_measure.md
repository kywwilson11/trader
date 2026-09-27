# IMPL_measure — G6 measurement-shelf fixes (2026-09-27, Jetson)

All fixes are measurement-only and ship directly (AGENT_CONTEXT rule 2). Nothing was committed.
BARS_PER_YEAR and indicator_leadlag's effective-n were not touched.
Raw outputs are in `scratchpad/impl_measure/`. Every report JSON that the re-runs wrote to the repo root was moved there:
- decision_report.json
- execution_report.json
- llm_eval_report.json (twice)
- llm_advisor_report.json. The `--advisor` run overwrote this gitignored file (mtime 00:11:21); its content was a `no_data` stub.

## Changes (file:line → what)
1. **G6-1 · llm_eval.py `realize_scored_rows` (~:230-245) + new `_clamp_stock_end` (~:175).**
   - **Stock:** `end = max(t0) + (max_h//7 + 5) days`, passed through `market_data._clamp_sip_end(end, 'stock')`. That call adds the SIP delay plus FIX_R2's in-session hour floor. If market_data cannot be imported, a plain 16-min clamp is used instead.
   - **Crypto:** the window is byte-identical (a test pins start and end).
   - **Live read-only proof:** SPY, Thu 2026-09-24 10:05 ET, fb=24 (`g61_live.out`).
     - Pre-fix window: 24 bars → realized **None**.
     - Fixed: realized **+1.015%**, 24 bars spanned.
2. **G6-2 · naive journal ts is now read as LOCAL time.**
   - **decision_report.py:** new `_LOCAL_TZ` hook plus `_naive_local_to_utc` (~:218-238). It is used by `_dedup_first_per_day` (sort key) and `_replay_grouped` (replay ts). The docstrings and the two comments that said "naive treated as UTC" were updated.
   - **scripts/sizing_cofire_report.py `_parse_ts`:** now `.astimezone()`, with the same `_LOCAL_TZ` hook.
   - Offset-aware rows are unchanged, and a CI box running in UTC behaves exactly as before.
   - **Real-journal A/B,** with the old reading emulated via `_LOCAL_TZ=utc` (`decision_report_priced_ab.out`). Only 2 crypto buys price in 200 d:
     - new: entries at 15:21Z / 16:05Z, net **−0.29% / −2.10%**
     - old: entries at 10:21Z / 11:05Z, net −0.23% / −0.58%
     - The report JSON does not change, because n=2 buckets are suppressed.
3. **C-1 · scripts/llm_qualify.py `run_qualification` / `verdict_for`.**
   - **schema_valid_pct:** the denominator is now `n_attempts − rate_limit_events`, so timeouts and 5xx count as invalid. It stays None when n_completed=0.
   - **Latency:** status-less failures (timeouts, transport errors) enter the latency sample censored at `max(latency, BUDGET_S)`. New keys: `n_timeouts` and `p95_censored`.
   - **Verdict:** `verdict_for` treats a censored p95 as +inf, which fails both ceilings.
   - The docstring now documents this.
   - **Tests:** 5/10 timeouts → schema 50%, **failed** (it was "qualified"). 2/20 timeouts → schema 90% would give marginal, but the censored p95 makes it **failed** (the latency gate fails on its own). All-answered calls are unchanged.
4. **C-1 · column projection** in scripts/horizon_transfer_report.py, wave6_stage0.py and funding_drift_audit.py.
   - The three scripts have identical local copies of a `_projected_columns(book, keep)` helper. It reads the parquet SCHEMA and returns Ticker plus the matching names. It returns None (meaning today's full load) when:
     - the parquet is absent,
     - `data_utils._csv_is_fresher` (its print is silenced),
     - or pyarrow is missing.
   - The result is passed to the public `load_training_data(book, columns=...)`, which already accepted `columns=`. data_utils was not edited; the helper reads data_utils' private `_stem`, `_BASE_DIR` and `_csv_is_fresher`, read-only.
   - wave6's defending comment was replaced with the measured truth.
   - Measured on the real stores (`proj_crypto.out`, `proj_stock.out`), sha of all outputs:

     | script | crypto full RSS | crypto projected RSS | sha | stock projected RSS |
     |---|---|---|---|---|
     | horizon_transfer_report | 765 MB | 214 MB | identical | 516 MB |
     | wave6_stage0 | 543 MB | 301 MB | identical | 532 MB |
     | funding_drift_audit | 531 MB | 249 MB | identical | — |

     The full stock loads were not run, because they are projected at about 3.8 GB.
   - The synthetic-store tests assert sha-identical output (stdout included), the exact projected column sets, and the CSV-fresher / no-store fallbacks.
5. **C-2 · scripts/train_lexicon.py.**
   - Days are now `idx.tz_convert(args.session_tz).date`. A naive index is localized to UTC first.
   - New flag `--session-tz` (default America/New_York; pass UTC for a crypto store).
   - The test shows no phantom Saturday: Friday's close is the 19:00 ET bar and Monday's open is Monday's 04:00 ET bar. `--session-tz UTC` reproduces the old buckets.
6. **B-2 · scripts/rank_gradient_report.py.**
   - These `--preds` inputs now print "no rows: … empty Stage-0 dump" to stderr and exit **2**:
     - an empty dump (`[]`),
     - a 0-byte CSV,
     - a header-only CSV.
   - A non-empty dump with no `ts` column also exits 2.
   - A new `EXIT_CODES_EPILOG` documents 0/1/2 in `--help`, and the docstring's exit sentence was updated.
7. **G6A-1 · execution_report.py.**
   - When no fill carries slippage, the report now prints `shortfall section skipped: 0/N buys carry slippage_bps (...)`.
   - The "Compare against the backtest" footer prints only when fills exist.
   - Real run, 200 d: `0/169 buys`.
8. **B-1 · scripts/reliability_report.py.**
   - Brier/ECE for the labels and `tied` are now computed on compare_calibrations' jointly-finite mask. When the mask is empty the label is n/a.
   - On the hunter's NaN payload the Brier line now prints "(worse)".
9. **C-3/4/5 · scripts/sizing_cofire_report.py.**
   - **C-3, header:** `_load_rows` also counts buys without sizing (it now returns a 5-tuple; main is its only caller). The header reads "`N buy rows, M with sizing`".
     - The JSON gains `n_buy_rows_without_sizing`; `n_buy_rows` keeps its meaning.
     - With 0 sizing rows and >0 plain buys, the report prints "no rows with sizing — the N buy row(s) … pre-date the sizing producer". The substring pinned by test_c26_S3:802 is kept.
     - Real run: "169 buy rows, 0 with sizing".
   - **C-4:** flips are ordered by parsed datetime. The DST test now counts 2 flips (it was 1).
   - **C-5:** `--json PATH` cleans up its tmp file in a `finally`. The PATH=directory case now exits 1 and leaves no leftover tmp.

## Tests
- New file `tests/test_g6_fixes_2026_09.py`: 29 tests, Mac-safe. The parquet tests importorskip pyarrow, and zones come from zoneinfo via the `_LOCAL_TZ` hooks, never the process TZ.
- `tests/test_measurement_fixes_2026_09.py` was not changed. Its sizing and execution tests pass as they are.
- Required set: **593 passed** (`impl_measure/pytest_required.log`). It covered g6, measurement_fixes, decision_report(+v3), llm_eval(+v3), c26_S3, c26_P2, grp_reports, c26_V1 and r2c_*.
  - There is no test_execution_report*.py.
- Related modules: **259 passed** (`pytest_extra.log`). They were c26_T7, c26_U1, c26_V2, review_b17, b20, b21 and b22, grp_ops, cs_neff, llm_advice and c26_S1.
- `py_compile` passes on all 11 files plus the new test, under both python3 and $JPY. No full suite was run.

## Real-journal re-runs (--days 200)
- **decision_report:** quality `{priced 2, unpriced 167, out_of_window 15, dropped_null_pred 152, unpriced_rate 0.988}`. The gate sections price 0 rows, because every skip reason is an unknown `llm_below_buy_min (x<y)` variant, as before. See item 2 for the A/B.
- **llm_eval** (plain, `--asset stock`, `--advisor`): still no `llm_analysis` / `llm_advisor_v2` rows, so the stub path ran and every run exited 0. G6-1 is proven by the live SPY row above instead.

## For the orchestrator (outside my ownership)
- **docs/MODULES.md** (the CLI census) and **scripts/README.md** need three updates:
  - train_lexicon `--session-tz` (default America/New_York)
  - rank_gradient_report exit code 2 for an empty `--preds` dump
  - sizing_cofire's new JSON key `n_buy_rows_without_sizing`
- `scripts/train_lexicon.py` still loads the whole stock parquet with `pd.read_parquet(args.data)`, although it needs only Ticker/Open/Close. The same projection could apply, but it was not in scope.
- The sibling string-sorts flagged under C-4 (llm_qualify `_load_replay_cycles`, prompt_ab `load_replay_cycles`) were left alone: they are negligible, and prompt_ab is not mine.
- `_projected_columns` exists as three identical copies, because no shared module is owned here. A `columns_matching(book, predicate)` helper in data_utils would dedupe them.
