# IMPL_gui — G8 hunt fixes (gui.py, tax_lots.py), 2026-09-27

## Scope and what was touched
- **Files edited:** `gui.py` (11 hunks, +193/−26), `tax_lots.py` (6 hunks) and `tests/test_tax_lots.py` (append-only; no existing line removed).
- **New file:** `tests/test_g8_fixes_2026_09.py`. It is PySide6-free: gui.py helpers, methods and `LogTailer` are pulled out of the AST and exec'd against stubs.
- `tests/test_gui_fixes_2026_09.py` was **not** changed. The G8 tests live in the new file, and the existing file stays green.
- **Earlier hunks are intact:** FIX_F (D1/D2/D3/D4-5/D10) and FIX_R3 (M3: `_walk_orders_pages`, `ORDERS_FULL_WALK_SEC`). My diff against the pre-edit copy touches none of their lines.
- **Pre-edit copies:** `IMPL_gui/{gui,tax_lots,test_tax_lots,test_gui_fixes_2026_09}.py.pre`.
- **My diffs:** `IMPL_gui/{gui,tax_lots,test_tax_lots}.G8.diff`.
- Nothing was staged or committed. `pipeline_command.json` was never created. The repo's `gui_settings.json` / `news_cache.json` are untouched (mtimes are still Mar/May).

## Fixes
| id | where | change |
|---|---|---|
| G8-1 | new pure `_llm_entry_ts` + `_merge_llm_sections(doc)`; used by `DataFetcher.fetch_stocks` and `_reload_llm_from_disk` | Flattens the {crypto, stock} sections so the newest timestamp wins per symbol. A missing or unparseable timestamp loses. On a tie the later section wins, which is the old file-order behaviour, so symbols that appear only once come out identical. `Z` suffix parses on py3.10. A non-dict doc returns {}. |
| G8-2 | new pure `_is_crypto_symbol`, `_lookup_position_state`; `_exit_levels_raw` | Crypto is detected by `'/'` or membership in `CRYPTO_SYMBOL_SET`, which holds both spellings. `position_state` is looked up exactly first, then slash- and case-insensitively. Result: `BTCUSD` gets `CRYPTO_POLICY` plus the `BTC/USD` trailing ratchet. Stock rows are unchanged. |
| G8-3 | `tax_lots.order_has_fill` + `FILL_BEARING_STATUSES = {partially_filled, canceled, expired, done_for_day}`; `estimate_taxes`; gui Recent Fills (`_apply_trade_filter`) | An order counts as a fill if it has `filled_avg_price` AND (status is `filled`, OR a fill-bearing status with `filled_qty > 0`). One predicate is shared by the tax kernel and the Recent Fills table. |
| G8-4 | new `_read_text_tail` / `_read_log_tail`; `_on_log_selected` | Reads a bounded window at the end of the file (4·maxlen+8 bytes), decodes it as UTF-8 with `errors='replace'` and universal newlines, then applies `_trim_to_newline`. |
| G8-5 | new `_atomic_write_json`; `_save_gui_settings`, `_save_news_cache` | Writes to a tmp file and then `os.replace`. The tmp name is unique per writer (`<name>.<pid>.<thread-id>.tmp`). The tmp file is removed if the write fails. The existing exception swallow is kept. The output bytes are the same as before (`indent=2` for settings). |
| G8-6 | `tax_lots._one_year_after` + `_is_long_term` | Long-term means the UTC sale date is strictly after the calendar anniversary of the purchase. A 29 Feb purchase has its anniversary on 28 Feb, so 1 Mar is long-term. A mix of naive and aware timestamps no longer raises. `LONG_TERM_DAYS` is kept, but only as a reference constant. |
| G8-7 | new `_utf8_complete_len`, `_decode_log_bytes`, `_read_log_increment`; `LogTailer.check_logs` | Reads exactly the byte range `[last_pos, size)`, decodes it, and advances the cursor in bytes. A trailing incomplete UTF-8 sequence is held back until the next tick. Invalid bytes still decode to U+FFFD. |
| G8-8 | `tax_lots.QTY_EPS = 1e-9` | The loop runs while `remaining > EPS`. A lot is popped at `qty <= EPS`. Anything unmatched is counted only if it is `> EPS`. |

- **G8-3 decision:** `replaced` is deliberately **not** in the fill-bearing set. The fill history continues on the replacement order, so counting both could double-count.
- **G8-3 test constraint:** `new`/`accepted` are also excluded, even when they carry `filled_qty`. `test_non_filled_orders_are_ignored` builds status `new` with `filled_qty=10`, and it must stay green. It does.

## Verification
- **Compile:** `/home/kyle/miniforge3/bin/python -m py_compile gui.py tax_lots.py` succeeds.
- **Tests:** ran `CUDA_VISIBLE_DEVICES='' $JPY -m pytest tests/test_g8_fixes_2026_09.py tests/test_gui_fixes_2026_09.py tests/test_tax_lots.py tests/test_chart_core.py tests/test_gui_contracts.py tests/test_gui_charts.py tests/test_c26_U1.py tests/test_review_b03.py -q -p no:cacheprovider`. Result: **367 passed**.
  - `test_g8_fixes_2026_09.py` has 69 tests.
  - `test_tax_lots.py` went from 24 to 40 tests. The new classes are `TestG8FillsOnNonFilledStatus`, `TestG8LeapYearLongTerm` and `TestG8FloatDust`.
- **Mutation checks:**
  - Against the pre-edit `tax_lots.py`, **11 of the 16** new tax tests fail. The 5 that pass pin behaviour that should not change: zero or missing `filled_qty`, new/replaced statuses, and a real shortfall above tolerance.
  - The pre-edit `LogTailer` on the hunter's interleave emits `'LINE-C writLINE-C written during the tick'`. The new one emits exactly the file's lines.
  - Script: `IMPL_gui/mut_check.py`.
- **G8-4 on the real logs (read-only):** the bounded read matches `_trim_to_newline(read_text())` exactly on all three logs.

  | log | old read | new read |
  |---|---|---|
  | stock_bot_output.log (115 MB) | 427 ms | 3.9 ms |
  | crypto_bot_output.log (60 MB) | 233 ms | 2.8 ms |
  | pipeline_output.log | identical | identical |

  The unit tests cover every window offset through a 4-, 3- and 2-byte character, CRLF and a lone CR, invalid bytes, an all-4-byte worst case, and a file larger than 800 KB.
- **Headless GUI run:** driver `IMPL_gui/g8_verify_driver.py`, base env, offscreen, with the same redirection and stubs as `F/fix_driver.py`. Results are in `IMPL_gui/runs/g8v_result.json`.
  - **Tabs:** all 8 tabs × 2 themes (Batman, Paper) rendered with **0 uncaught exceptions**.
  - **Screenshots:** 16 tab shots plus `g8_2_positions_BTCUSD_crypto_policy.png`, all in `scratchpad/g8_shots/`.
  - **G8-2:** the real `on_positions` render of an Alpaca-shaped `BTCUSD` row (entry 80k, current 85k, `BTC/USD` hwm 90k, trailing on) now shows stop `~$88,650.00`, TP `~$82,400.00`, `~-4.3%`. That equals the chart path (`BTC/USD`). Before, it showed $79,200 / $81,600. `AAPL` is unchanged at $198 / $204.
  - **G8-2 live positions:** the 6 real crypto positions still show "—", because Alpaca reports avg_entry 0 for them, as the hunter noted.
  - **G8-1:** `_llm_analysis_cache['BTC/USD']` is now 2026-05-05 (LLM Age **145d**; it was 180d). ETH, DOGE, LINK, SOL and XRP are also fixed. AVAX, DOT and LTC correctly keep the stock-section copy, which is the newer one for those three. `_reload_llm_from_disk` gives the same results.
  - **G8-3:** 1088 cached orders → 471 `filled` / **523 fills**.
  - **G8-3 Realized Gains card:** now **−$5,160.62**; it was −$3,950.69. The same 1088-order window, computed offline with the old and new modules, gives exactly that before/after (`IMPL_gui/tax_compare.py`).
  - **G8-3 full history:** −4,487.39 → −5,761.66, which matches the hunt. Real data is moved only by G8-3; G8-6 and G8-8 do not change it.
  - **G8-5:** theme changes wrote the scratch `gui_settings.json` atomically with no `.tmp` residue.
- The full suite was not run, as instructed.

## Notes for the owner
- The Cockpit positions table is still narrow at 1280×800, so the screenshot truncates the Stop/TP cells (FIX_F already recorded this squeeze). The exact cell text is in the result JSON.
- First view of a log is now about 520 ms, and that time is almost entirely the render (`_rerender_log_view`, the hunter's judgment call #2). The file read itself is about 4 ms.
- Out of scope, same root as G8-2: `_size_by_policy` and `_manual_trade` still classify crypto by `'/'` (judgment call 7, which touches sizing math).
