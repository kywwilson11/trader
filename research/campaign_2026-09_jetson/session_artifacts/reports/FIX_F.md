# FIX_F — gui.py fixes for audit F (D1, D2, D3, D4/D5, D10), 2026-09-26

Files edited: `gui.py` (+137/-19) and a new `tests/test_gui_fixes_2026_09.py` (33 tests, PySide6-free).
`chart_core.py` was NOT edited by me. Its working-tree diff at `gate_panel_model`, the D8 stale-reason split, came from another agent. I left it untouched.
Pristine copy of gui.py before my edits: `scratchpad/F/gui.py.orig`.

## Diff summary (gui.py)

### D1 — HIGH: orders pagination cursor (FIXED, verified live)
- New module-level pure helper `_alpaca_until(ts)`, defined after `_engine_env`.
  - Accepts a datetime, a pandas Timestamp or an ISO string: space or `T` separator, `Z` or an offset, 1–9 fractional digits.
  - Naive values are treated as UTC. Returns RFC3339 UTC `YYYY-MM-DDTHH:MM:SSZ`, or None when the input is empty or unparseable. The existing `new_until is None` guard then stops the walk.
- **Sub-second values are rounded UP to the next second, not truncated.** A live probe showed Alpaca's `until` is exclusive. Truncating `19:45:17.419822` to `19:45:17Z` would silently skip older orders placed within the same second. The one-order overlap that rounding up re-fetches is already removed by the `seen_ids` dedupe in `fetch_orders`.
- `fetch_orders` is the only `until=` pagination site in gui.py. It now does `new_until = _alpaca_until(oldest.submitted_at)`.
- Live results (read-only):
  - Before: `API: orders ERR×1`, zero emits, `stream_health.orders.fails` reached 4.
  - After: `orders_updated` emitted **1088 orders** (truncated=True, hit the 1000 cap) at boot and again on a forced fetch, with `API: OK` and fails=0.
  - Recent Fills and the Est. Tax cards now populate. The tax row shows realized −$3,950.69 and "incomplete basis" because the cap cut history short.
  - Open Orders shows 0 rows, which is correct: no bot is running.

### D2 — HIGH: NumericTableItem.__lt__ recursion (FIXED)
- `super().__lt__` is gone. The new logic:
  - If both items have a UserRole payload, compare them as floats. A non-numeric payload falls through.
  - Otherwise compare the plain `text()` strings, the same lexical order Qt's default operator< uses.
  - `other=None` is safe.
- Driver: RecursionErrors went from **7 (before) to 0 (after)**.
  - `_stock_table.signalsBlocked()` was True after boot and after each theme switch before the fix. After the fix it is False throughout, so `_sync_stock_table` and `_restyle` now run to completion.

### D10 — LOW: LD_LIBRARY_PATH trailing ':' (FIXED)
- `_engine_env` now drops empty path elements, so the result is `jetson/lib[:cusparselt]` plus the inherited non-empty elements. The cwd is never on the search path.

### D3 — MED: price y-axis pinned at 0 (FIXED)
Changes:
- `_build_stocks_tab` now adds the ATR `FillBetweenItem` with `ignoreBounds=True` and hides it at construction. The fill always lies between the two band lines, and those still drive autorange.
- `_apply_chart_zoom` shows the fill only when the band has data.
- `_clear_price_items` hides it again.

BTC/USD y-range, measured live:

| state | before | after |
|---|---|---|
| ATR band off | [-6252, 93637] | [74154, 88140] |
| ATR band on | — | [69919, 92805], fill visible |
| ATR band toggled off again | — | [74154, 88140] |

Screenshots: `before__d3_chart_BTC-USD.png`, where the candles are squashed at the top of a 0–85k axis, and `after__d3_chart_BTC-USD.png`, where the candles fill the plot. Also `*_atr_on.png`.

### D4/D5 — MED: clipping at 1280×800 (FIXED for Performance, Markets and Models)
Changes:
- New staticmethod `TradingDashboard._scroll_wrap(inner)`. It is the Settings-tab recipe: a frameless, widget-resizable `QScrollArea`.
- Performance, Markets and Models are now added through it.
- `_markets_tab_index` now uses `indexOf(markets_page)`, i.e. the scroll page that is actually in the tab widget. Every other tab lookup is by index, and nothing uses `tabs.widget()` or `currentWidget()`, so this was the only reference to update.
- Two follow-ups the screenshots showed were needed:
  - **Models Reports row:** 8 buttons needed about 1300 px, which forced a horizontal scrollbar. They now sit in two rows of four, and the inner minimum width dropped from 1318 to 770.
  - **Performance plots:** inside a scroll area the plots collapsed to their 69 px minimum. They now have minimum heights: equity 220, daily P&L 130.

Inner minimum size vs. the 1267×645 viewport:

| tab | inner minimum (w×h) | result |
|---|---|---|
| Performance | 1228×890 | scrolls vertically |
| Markets | 675×734 | scrolls vertically |
| Models | 770×1183 | scrolls vertically, no horizontal bar |

- Theme check, looked at myself: Batman, Space and Joker (dark) and **Paper** (light). The scroll area inherits each theme's background, borders and group titles. Nothing is stuck on another palette.
- The stat-card values are no longer clipped. Both Models boxes render in full with no overlapping text: the table shows both the Crypto and Stock rows, and the Shadow/Meta boxes are readable.

## Verification
- `/home/kyle/miniforge3/bin/python -m py_compile gui.py`: OK.
- Headless driver `scratchpad/F/fix_driver.py before|after`. It is based on the audit driver, with the same redirection of gui_settings.json and news_cache.json to scratch, the account_baseline write intercepted, dialogs suppressed and no LLM spend.
  - "before" runs the pristine gui.py from `F/orig_mod/`, with BASE_DIR pinned to the repo.
  - The after run covered all 8 tabs × 4 themes (Batman, Paper, Space, Joker). Results: **0 uncaught exceptions, 0 RecursionErrors**, orders count 1088 > 0, and `pipeline_command.json` and `account_baseline.json` were never created.
  - Logs: `F/fix_{before,after}.{out,err}`, `F/fix_result_{before,after}.json`. Size probe: `F/size_probe.py` → `F/sizeprobe.out`.
- Tests (jetson env):
  - `$JPY -m pytest tests/test_gui_fixes_2026_09.py tests/test_chart_core.py tests/test_design_tokens.py -q -p no:cacheprovider` → **210 passed in 5.27s**.
  - Also re-run, because they source-inspect the builders I touched: test_gui_contracts, test_gui_charts, test_c26_U1 and test_review_b03 → **118 passed**.
  - The full suite was NOT run, as instructed.
- What the new tests cover:
  - `_alpaca_until`: extracted via AST with 17 cases, including the exact rejected string, pandas Timestamp including a pure-ns fraction, offsets, naive input, the midnight round-up, and unparseable → None.
  - A **standalone reproduction of the PySide6 6.8 recursion**. A stub base class whose `__lt__` re-dispatches to the override makes the old code raise RecursionError. The extracted new `NumericTableItem` sorts numerically and by text without recursing.
  - `_engine_env` with no, empty and messy inherited `LD_LIBRARY_PATH`.
  - Source contracts for D3, D4/D5, the two report-button rows and the plot minimum heights.

## Screenshots (`scratchpad/F/fix_shots/`)
- D3: `before__d3_chart_BTC-USD.png` vs `after__d3_chart_BTC-USD.png`, plus `after__d3_chart_BTC-USD_atr_on.png`.
- D4/D5:
  - `before__Batman__{2_Performance,4_Markets,5_Models}.png` and `before__Paper__*.png`.
  - vs `after__{Batman,Paper,Space,Joker}__{2_Performance,4_Markets,5_Models}.png`.
- D1: `after__Joker__1_Trading.png` (Recent Fills populated), `after__Paper__2_Performance.png` (tax cards populated).
- The full after grid is 32 PNGs; the before grid covers Batman and Paper.

## Deferred / not done
- **D6, D7, D8, D9: skipped, report only.**
  - D7: on first visit to Markets the chart still shows "ABNB — Loading…", because nothing requests a chart on tab show.
  - D8 appears to have been handled by another agent in chart_core.py; that diff is not mine.
- **Other tabs still squeeze at 1280×800 (not in my scope list):**
  - Trading: Recent Fills shows about 1 row, and Open Orders and the gate box are compressed.
  - Cockpit: the positions header text is clipped.
  - Both could get the same `_scroll_wrap`, plus minimum table heights. I did not do this because it was not requested.
- **Markets:** the stance table still shows about 1 row inside the now-scrollable page, and the LLM detail pane sits below the fold. A minimum table height (for example about 180 px) would help, but it would lengthen the scroll. Left for owner taste.
- **Performance:** at the 1100 px minimum window width, the Est. Tax group (minimum 1210 px) would get a horizontal scrollbar. That is better than clipping, and it does not happen at the 1280 default.
- **Operational note (from the audit, now live):** with D1 fixed, `fetch_orders` walks about 11 pages (up to the 1000 cap) every 30 s. That is roughly 20 Alpaca requests/min on the key the bots share. I recommend a longer orders cadence, or caching the tax history.
- Nothing was committed or staged.
