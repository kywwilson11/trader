# G8: operator console hunt (gui.py, chart_core.py, design_tokens.py, tax_lots.py), 2026-09-26

Hunt phase only. **No production file was edited.** Everything I ran lives under `scratchpad/hunt/g8/`.

How the runs were made:
- The GUI ran under the base env (`/home/kyle/miniforge3/bin/python`, `QT_QPA_PLATFORM=offscreen`, `CUDA_VISIBLE_DEVICES=''`), driven by `hunt/g8/g8_driver.py`, which was derived from `F/fix_driver.py`.
- In that driver, `gui_settings.json` and `news_cache.json` were redirected to `hunt/g8/runs/`. `_maybe_write_baseline` and the report dialogs were stubbed, and `DataFetcher.fetch_news` was disabled, so there was no Finnhub or LLM traffic.
- The only network use was read-only Alpaca calls.
- `pipeline_command.json` was never created (checked before and after the run).

I read FIX_F and FIX_R3 first. D1, D2, D3, D4/D5, D10 and M3 are already fixed and are not re-reported. D6, D7, D8 and D9 are listed in the audit and are not re-reported either.

Findings are ranked by severity.

---

## G8-1 · gui.py:2174 and gui.py:5855 · class A (+B): the llm_analysis.json section merge picks the stale entry. Proven on the real file.

**Defect.** Both readers of `llm_analysis.json` flatten `{"crypto": {…}, "stock": {…}}` with a naive `dict.update`, in file order:
- `DataFetcher.fetch_stocks`, gui.py:2167-2177
- `TradingDashboard._reload_llm_from_disk`, gui.py:5848-5859

The rule in the memory note `feedback_llm_analysis_sections.md` is to keep the newer timestamp when the same symbol appears in both sections. That rule is **absent from both readers**. `git log -G` shows it was never committed.

On this box, a stale batch left crypto symbols in the `stock` section. Because `stock` is merged after `crypto`, the stale copy wins.

**Proof.**
- `hunt/g8/repro_llm_merge.py` runs read-only on the real `llm_analysis.json` (May 7).
  - 6 symbols render the stale entry: BTC, DOGE, ETH, LINK, SOL, XRP/USD.
  - Example: BTC/USD shows `2026-03-30T23:49:10`, `gemini-3-flash-preview`. The crypto section's entry is `2026-05-05T04:29:52`, `gemini-3.1-pro-preview`.
- The same result appears inside the running GUI (`hunt/g8/runs/g8_result.json → llm`):
  - `_llm_analysis_cache['BTC/USD'].timestamp == '2026-03-30T23:49:10+00:00'`.
  - The Markets **LLM Age cell shows "180d"**; the correct age is 144 d.
  - Calling `_reload_llm_from_disk()` directly gives the same stale timestamp.
- Live consequence: the exact bug the memory note describes is back.
  - "Refresh Selected" on BTC/USD runs `refresh_one(sym, 'crypto')` (gui.py:5674), which writes the **crypto** section.
  - Both readers then still show the stale **stock**-section copy.
  - The Refresh-All confirm dialog's "N still fresh" count (gui.py:5764-5774) is wrong for the same reason.

**Fix.**
- Add one module-level pure helper `_flatten_llm_sections(raw)`, used by both readers. For each symbol it keeps the entry whose `fromisoformat(timestamp)` is greatest; a missing or unparseable timestamp loses.
- A reference implementation is `flatten_newest` in the repro. It is identical to today's output for every symbol that is not duplicated, which the repro asserts.
- The two copies of the loop become one, which is the B part of this finding.

**Blast radius.** Display only: the stance table, the dossier panel and the Refresh-All count. No test references either reader (grepped `tests/`). `base_loop` reads its own section only and is unaffected.

**Why indisputable.** A documented rule, a real file and the rendered GUI cell all show the older analysis winning over the newer one.

---

## G8-2 · gui.py:8069 and gui.py:8074 · class A: crypto positions get the STOCK exit policy and lose their trailing ratchet in the Positions table

**Defect.**
- `_exit_levels_raw` decides crypto vs stock with `"/" in str(symbol)` and looks up `pstates.get(symbol)`.
- The Positions table feeds it Alpaca position symbols. **Alpaca returns crypto positions slash-less.** A read-only probe (`hunt/g8/pos_probe.py`) returned `BTCUSD, DOGEUSD, ETHUSD, LINKUSD, SOLUSD, XRPUSD`.
- So every crypto row uses `STOCK_POLICY` (stop floor 0.010), not `CRYPTO_POLICY` (0.015).
- It also misses `position_state.json`, whose keys are `BTC/USD`: base_loop keys positions by universe symbol (order_utils.reconstruct_positions → base_loop:547/472). A trailing ratchet is therefore never applied.
- The chart guide lines for the same position (`_update_position_lines`, gui.py:6229) pass the combo text `BTC/USD`, so the table and the chart disagree.
- The code already knows about the two spellings: `_position_row` is slash-insensitive, and `CRYPTO_SYMBOL_SET` contains both forms.

**Proof.** From `g8_result.json → exit`, real GUI methods, entry 80 000, hwm 90 000, trailing on:

| path | entry | est. stop | est. TP |
|---|---|---|---|
| table (`'BTCUSD'`) | 80000 | **79 200** | **81 600** |
| chart (`'BTC/USD'`) | 80000 | 88 650 | 82 400 |

The actual `on_positions` render of an Alpaca-shaped row shows `~$79,200.00 · ~$81,600.00 · ~+6.8%`. The "%→Stop" column, which turns red below 1 %, is therefore computed against the wrong stop.

**Fix.**
- In `_exit_levels_raw`:
  - use `pol = CRYPTO_POLICY if ('/' in s or s.upper() in CRYPTO_SYMBOL_SET) else STOCK_POLICY`, where `s = str(symbol)`;
  - resolve `st` slash-insensitively, for example by normalising the keys once in `_load_position_states`.
- Nothing else changes for stock symbols or for `BTC/USD` callers.

**Blast radius.** Display only: the table columns 8-10 and the chart guide lines. No test references `_exit_levels_raw` or `_compute_exit_levels`.

**Why indisputable.** One position, one policy, two different displayed stops, on real Alpaca symbol formats.

---

## G8-3 · tax_lots.py:157-160 (+ gui.py:8906) · class A: real fills on `canceled` orders are dropped from tax lots and from Recent Fills

**Defect.**
- `estimate_taxes` only considers `status == "filled"`.
- Alpaca reports a partially-filled order that was later canceled as `status="canceled"` with `filled_qty > 0`, a `filled_avg_price` and a `filled_at`. This is common for the loops' GTC limit entries.
- Those shares were really bought or sold, but they never become lots or matched sells.
- The Recent Fills table has the same filter (gui.py:8906), so those executions never appear there either.

**Proof.**
- Read-only dump of the full order history: `hunt/g8/orders_dump.py` → `orders_all.json`, 1 680 orders.
  - **54 canceled orders carry real fills (31 buys, 23 sells, $53 795 notional).** All 54 have `filled_at` and `filled_avg_price`.
- `hunt/g8/tax_real.py`, full history:
  - realized −$4 487.39 → **−$5 761.66** when these fills are counted;
  - unmatched sell qty 5 801.8 → 3 248.3.
- In the GUI's 1 000-order window: 51 such orders ($44 821); realized −3 318.41 → −3 355.55; unmatched 5 434.6 → 3 015.0; matched lots 273 → 323.
- Minimal repro, `hunt/g8/repro_tax_lots.py` case 2: a canceled SOL buy with a 14.72 fill, then a sell. Result: realized 0, unmatched 14.72, basis incomplete. The correct result is +50.72, 0, complete.

**Fix.**
- tax_lots: count an order when `status == "filled"`, **or** when `status in {"canceled","expired","partially_filled","done_for_day"}` and `float(filled_qty) > 0` with a `filled_avg_price`.
- `test_non_filled_orders_are_ignored` uses `status="new"` and stays green.
- gui.py:8906 Recent Fills: apply the same predicate.

**Blast radius.**
- Tax cards and Recent Fills only.
- The pinned tests in `tests/test_tax_lots.py` (TestFilteringAndOrdering) use `"new"`, so they are unaffected.
- Current suite: `test_tax_lots.py`, `test_chart_core.py` and `test_design_tokens.py` together give 201 passed.

**Why indisputable.** Executed shares from the live account are silently excluded from the cost basis.

---

## G8-4 · gui.py:9036-9038 · class C: first view of a bot log reads the whole file (115 MB) on the UI thread to keep a 200 KB tail

**Defect.**
- `_on_log_selected` does `_trim_to_newline(path.read_text(errors="replace"))` whenever that log's buffer is empty. The buffer is empty on the first selection when the tailer has not seen new lines, which is always the case for a quiet or stopped bot.
- `_trim_to_newline` keeps only the last 200 000 characters.
- On this box `stock_bot_output.log` is **115 MB** and `crypto_bot_output.log` is **60 MB**.

**Measurement.** `hunt/g8/log_tail_measure.py` copies `_trim_to_newline` verbatim, runs each variant in a separate process, and **asserts that the resulting buffers are identical**, which they are for all 4 logs:

| log | current: time / peak RSS | bounded tail read: time / peak RSS |
|---|---|---|
| stock_bot_output.log (115 MB) | **427 ms / 565 MB** | 6.3 ms / 19 MB |
| crypto_bot_output.log (60 MB) | **268 ms / 300 MB** | 4.6 ms / 18 MB |
| pipeline_output.log (1.1 MB) | 4.3 ms / 19 MB | 5.1 ms / 18 MB |

The current path is roughly a 0.4 s UI freeze and a +546 MB transient RSS spike in the GUI process, which already sits at about 330 MB, on the 8 GB box that also runs the bots and training.

**Fix.** Seek to `max(0, size - (4*LOG_BUFFER_MAXLEN + 8))` in `'rb'` mode, decode with `'utf-8', errors='replace'`, translate `\r\n`/`\r` to `\n` (as `read_text` does), then `_trim_to_newline`.
- A UTF-8 character is at most 4 bytes, so the last 200 000 characters always lie inside that window. The result is bit-identical, as asserted on the real logs.
- Add a unit test that compares both paths on a synthetic multibyte log larger than 800 KB.

**Blast radius.** One call site. No test pins it.

**Why indisputable.** Byte-identical output with a measured 60–70× time and about 30× memory reduction.

---

## G8-5 · gui.py:377-384 and gui.py:432-460 · class D: gui_settings.json and news_cache.json are truncated in place, not atomically replaced

**Defect.**
- `_save_gui_settings` and `_save_news_cache` both do `open(path,'w')` + `json.dump`, and swallow every exception.
- Any failure after the truncate leaves a torn file:
  - ENOSPC (the GUI's own Disk gauge comment says "a full SD silently breaks status writes");
  - SIGKILL or power loss mid-dump;
  - the closeEvent race: step 5 (gui.py:10645-10649) writes `news_cache.json` from the main thread even when the slow thread was declared stuck (`threads_stuck`) and may itself be inside `_save_news_cache`.
- The loaders swallow the `JSONDecodeError` and return `{}` or `None`.
- The next read-modify-write, for example `_on_theme_changed`, then persists only the key it touched.

**Proof.** `hunt/g8/repro_settings_torn.py` uses the real gui functions against a scratch path, with ENOSPC injected after 10 bytes:
- settings `{theme, cadences, ov_sma20, chart_default_zoom}` → on disk `'{\n  "theme'` → `_load_gui_settings() == {}`;
- after the next theme change the file is `{'theme': 'Batman'}`, so every other setting is gone.
- For `news_cache.json`, a torn file means the next boot drops the cache, which re-bills LLM scoring for up to 200 headlines and leaves the News tab blank for about 70 s (see audit D6).

**Fix.** Use tmp + `os.replace`, the pattern gui.py already uses at three sites (1353-1357, 8361-8364, 9092-9095). Use a per-writer tmp name, e.g. `path.with_suffix('.json.tmp')`, and keep the existing swallow.

**Blast radius.** The two writer functions only. No test pins them.

**Why indisputable.** It adopts the file's own atomic-write idiom; the output is unchanged except that a failed write no longer destroys the previous file.

---

## G8-6 · tax_lots.py:28,64-71 · class A: a lot held exactly one year across 29 Feb is classified long-term

**Defect.** `_is_long_term` is `(sell - buy).days > 365`. IRS long-term treatment requires holding **more than one year**. When the holding period spans 29 Feb, the exact one-year anniversary is 366 days, so it is wrongly treated as long-term.

**Proof.** `hunt/g8/repro_tax_lots.py` case 1: buy 2024-01-15, sell 2025-01-15 (days = 366).
- Current output: `_is_long_term` True, LT gain 100, tax **25.0**.
- Correct: short-term, tax 100×(0.37+0.05) = **42.0**.

**Fix.** Compare calendar dates: `sell.date() > anniv(buy.date())`, where `anniv` is `replace(year=+1)` with 29 Feb mapped to 28 Feb.
- The existing boundary tests (`BASE=2023-01-01`: +365 → short, +366 → long) stay green, because 2023 is not a leap year.
- Add the leap-year case as a test.

**Blast radius.** The Est. Tax card only. Pinned by `TestLongTermBoundary`, which still passes.

**Why indisputable.** One calendar year is not a fixed 365-day count, and the module's own docstring states the "more than one year" rule.

---

## G8-7 · gui.py:2312-2318 · class A (low): LogTailer mixes a byte cursor with a character read and corrupts or duplicates lines

**Defect.**
- `check_logs` stats the file to get a size in **bytes**, then in text mode does `f.read(size - last_pos)`. That argument is a count of **characters**.
- It then sets `_positions[name] = size`.
- The bot logs contain multibyte UTF-8 (3 620 lines in crypto_bot_output.log, 4 132 in stock_bot_output.log have non-ASCII such as `—` `×` `→`). So if the writer appends between `stat()` and `read()`, the read over-runs `size`, and the next tick re-reads those bytes.

**Proof.** `hunt/g8/repro_logtailer.py` uses the real `gui.LogTailer`, with the writer simulated by appending right after `stat()`:
- file lines: `LINE-A …`, `LINE-B …`, `LINE-C written during the tick`;
- tailer emits `…, 'LINE-C writLINE-C written during the tick'`, a corrupted line.

**Fix.** Open in `'rb'`, read exactly `size - last_pos` bytes, and decode with `errors='replace'`. Optionally hold back a trailing partial line or UTF-8 sequence until the next tick.

**Blast radius.** The Logs tab view only. No test pins it.

**Why indisputable.** The cursor and the read length use different units; the repro shows garbled output from the real class.

---

## G8-8 · tax_lots.py:190-195 · class A (low): float dust flips `basis_complete` to False

**Defect.** Lot consumption is exact float subtraction, and `remaining > 0` has no tolerance.

**Proof.** `hunt/g8/repro_tax_lots.py` case 3: buy 0.3, sell 0.1, sell 0.2 gives `unmatched_sell_qty = 2.78e-17` and `basis_complete = False`, so the card shows "(incomplete basis)". Correct: 0 and True.
- Honesty note: 0 occurrences in the real 1 680-order history (`hunt/g8/tax_dust.py`), because Alpaca quantities are decimal strings that sum cleanly. Synthetic proof only.

**Fix.** Treat `remaining <= 1e-9 * max(1.0, qty)` as 0. Pop lots with `lot["qty"] <= 1e-9 * max(1.0, orig)`.

**Blast radius.** The tax card only. No test pins the zero comparison.

**Why indisputable.** A fully matched sell is reported as unmatched purely from rounding. Ranked low because the real history never triggers it.

---

## Hand-computed tax_lots check (requested)

`hunt/g8/repro_tax_lots.py` case 0 asserts all 8 output keys.
- Buys: AAA 10@100 (2023-01-10), 5@120 (2024-03-01), 5@90 (2024-06-01).
- Sell 12@110 (2024-07-01):
  - loss tier first: the 120-lot, 5×(−10) = −50 ST;
  - then long-term gain: the 100-lot, 7×10 = +70 LT.
- Sell 8@80 (2024-08-01):
  - loss tier, highest basis first: the 100-lot, 3×(−20) = −60 LT;
  - the 90-lot, 5×(−10) = −50 ST.
- Expected: realized −90, ST −100, LT +10, tax 10×0.25 = 2.5, net −92.5, 4 lots, complete.
- **The module matches exactly.** The MinTax ordering and the arithmetic are correct; the defects are only G8-3, G8-6 and G8-8.

---

## Verified clean (nothing to report)

- **Qt objects touched from worker threads: none.**
  - Runtime check: every one of the TradingDashboard methods was wrapped with a thread-ident check during a 150 s live run covering Cockpit, Markets, Models and Logs. **0 off-thread calls, 0 uncaught exceptions** (`g8_result.json`).
  - Static check:
    - DataFetcher and LogTailer only use `self.api`, their own caches and `emit`.
    - The three daemon threads (journal-stats, llm-test, notify-test) only emit queued signals.
    - Fetcher timers are touched only via `invokeMethod(set_interval/stop_timers)`.
    - Emitted payloads are fresh containers, and the shared inner lists are never mutated on either side.
- **`avg_entry_price=0` division: none remain.**
  - `_exit_levels_raw` guards `entry > 0`.
  - P&L % comes from Alpaca `unrealized_plpc`.
  - The risk gauge guards `equity > 0`.
  - `day_pct` and `tot_pct` guard their denominators, and HW % guards `total`.
  - The live account really has avg_entry 0 on all 6 positions (probe), and the GUI renders dashes without error.
- **chart_core numerics:** `hunt/g8/fuzz_chart_core.py` gives **0 failures**.
  - LTTB invariants hold (length `n_out`, strictly increasing, includes 0 and n−1) for n = 0..59, 100, 1000, 1501, 5000 × 13 values of `n_out`.
  - `ohlc_aggregate` equals a naive loop for n = 0..39 × factor ∈ {0, 1, 2, 3, 5, 7, 40, 41}, including the partial last bucket.
  - `build_price_view` was run on 0/1/2/3/14/15/20/50/299/300/301/600/601/1000 bars × 6 zooms × OHLC on/off × volume on/off × NaN first/last. All array lengths are consistent and overlays line up bar-for-bar.
  - `build_equity_view` on empty, single-point, zero-first and NaN input, with mismatched P&L lengths, always returns finite-or-None stats.
  - Empty-input `bar_widths`, `nearest_index`, `trailing_sma`, `wilder_atr` (length > m), `align_benchmark` and `obs_per_year` behave correctly.
  - Alpaca `profit_loss` checked live: 64/64 and 174/174 points equal `equity[i] − equity[i−1]`. So Daily P&L bars and `total_return`/`best_day`/`worst_day`/`win_rate` read the right quantity.
- **design_tokens.py:** `SOURCE_DEFAULTS` equals the AST-parsed `THEMES['Dark']` for all 13 keys. Every export is used by gui.py.
- **Timers on hidden tabs:** measured over 150 s, none is material.
  - `on_hw` (Models gauges and sparklines, every 5 s): median 7.7 ms, about 0.15 % of one core.
  - `_refresh_cockpit`: 6.6 ms.
  - `on_positions`: 11 ms.
  - `on_stocks`: 62 ms median, and it is visibility-throttled.
  - `_chart_timer` is already gated on the Markets tab.
- **Duplicated JSON helpers:** the halt-reason readers (gui.py:3774 and 10254) are identical, not diverged. The only diverged pair is G8-1.

## Judgment calls (not proposed)

1. **Optuna on the UI thread.**
   - `_get_best_score` imports optuna and runs `load_study` inside `_refresh_models_tab`. That cost 1.22–1.65 s during window construction.
   - While a hypersearch is writing `v2_study.db` (live tonight), it costs about 50–160 ms per 60 s tick: `_refresh_models_tab` median 162 ms, against about 10 ms otherwise.
   - Fixing it means moving the load off-thread, which is a design change.
2. **`_rerender_log_view` takes about 606 ms per render** (200 KB buffer, one `appendHtml` per line). It is wired to `QLineEdit.textChanged` (gui.py:7136), so every keystroke in the log filter re-renders. A debounce timer would fix it, but it is a UX change.
3. **First selection of a log skips history when lines are already buffered.** If the tailer buffered new lines before the first selection, `_on_log_selected` shows only post-start lines, not the historical tail (the `if not buf` guard). That is inconsistent with the idle path in G8-4, but whether to re-read from disk is a design choice.
4. **`_refresh_last_actions` runs `journal_stats.load_trades` on the UI thread every 60 s.** That is 49 ms on the last live week of journals, while `_refresh_journal_analytics` deliberately runs the same call off-thread. The cost is modest.
5. **`_combined_bots_running` forks `pgrep` on the UI thread** every 60 s in combined mode. `_restart_poll_stop` SIGKILLs every `pgrep -f run_pipeline\.py` match, which could include an editor that has the file open.
6. **Network calls and a shared session.**
   - `api.cancel_order` runs on the UI thread (gui.py:8997).
   - The two fetcher threads and the UI thread share one `alpaca_trade_api.REST` session.
7. **Manual trade sizing uses the same slash test as G8-2.** `_size_by_policy` and `_manual_trade` classify crypto with `'/' in symbol`. Typing `BTCUSD`, the spelling the Positions table shows, sizes with the 1 % stock floor, giving 1.5× the policy notional. This is sizing math, so it is excluded, but it has the same root as G8-2.
8. **`chart_core.heatmap_style(nan)` raises `ValueError`** via `int(round(nan))`. No live path produces a NaN `change_pct`.
9. **NaN sort keys break `NumericTableItem` ordering.** A NaN UserRole key, for example a NaN `pred` in predictions JSON, would make `NumericTableItem.__lt__` a non-strict-weak order. No NaN producer was observed.
10. **Two chart_core journal readers lack the `.jsonl.gz` fallback** (`load_trade_markers`, `sizing_stack_summary`). This is already documented at trade_journal.py:54-61, and rotation is off by default.
11. **tax_lots tax-date and short-sale limitations.**
    - Trade dates are taken in UTC; the IRS uses the trade date.
    - Short sells are unsupported. The real history contains shorts (for example ASTS sold 22 before buying 22 on 2026-03-31), which explains most of the remaining 32 unmatched sells after G8-3.
12. **The price-chart fingerprint omits o/h/l.** A repaint could only be skipped if H or L changed with an identical close and volume, which is not realistic.

## Coverage: "no issues found" per file

- **design_tokens.py:** no issues.
- **chart_core.py:** no issues. Fuzzed and checked live; see above.
- **tax_lots.py:** G8-3, G8-6 and G8-8 only; the MinTax ordering and arithmetic were verified by hand.
- **gui.py:** G8-1, G8-2, G8-3 (Recent Fills), G8-4, G8-5 and G8-7. Thread affinity, zero-cost-basis division, timer costs, report/LLM subprocess lifecycle, orders merge (post-R3), `fetch_chart` cache/eviction, and the heatmap/stance-table sort keys were reviewed with no further indisputable defect.

## Artifacts (`scratchpad/hunt/g8/`)

| purpose | files |
|---|---|
| repros | `repro_llm_merge.py`, `repro_tax_lots.py`, `repro_logtailer.py`, `repro_settings_torn.py` |
| measurement | `log_tail_measure.py`, `measure_last_actions.py`, `best_score_probe.py` |
| fuzz | `fuzz_chart_core.py` |
| live GUI driver | `g8_driver.py` → `runs/g8_result.json` |
| read-only Alpaca probes | `pos_probe.py`, `orders_dump.py` → `orders_all.json`, `ph_probe.py` |
| order-history analysis | `tax_real.py`, `tax_dust.py` |
