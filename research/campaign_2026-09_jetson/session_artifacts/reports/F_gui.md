# F — gui.py headless exercise on the Jetson (2026-09-26)

## Verdict

**The GUI works: it launches, stays up and renders all 8 tabs × 12 themes. But it has two real functional bugs and several layout defects at its default window size.**

- The window builds in about 4 s under the base env (py3.12, PySide6 6.8.0.2, pyqtgraph 0.14.0) with `QT_QPA_PLATFORM=offscreen`. It connects to Alpaca read-only (equity $122k, 6 crypto positions).
- All 96 tab×theme screenshots were captured. No tab raises an exception on its own, and all 12 themes apply their colours consistently.
- The report launcher works end to end. `pipeline_command.json` was never written. Idle CPU is about 1–2 % of one core and peak RSS (VmHWM) is 332–349 MB.
- **Bug 1 (HIGH): the orders stream is dead on this account.** `fetch_orders` builds its pagination cursor as `str(pandas.Timestamp)`, and Alpaca rejects that format (gui.py:1562).
- **Bug 2 (HIGH): `NumericTableItem.__lt__` recurses forever on PySide6 6.8.** This raised 16 uncaught `RecursionError`s in about 5 minutes. It breaks every text-column sort. It also aborts `_sync_stock_table` and `_restyle` partway through, leaving the Markets table with its signals blocked (gui.py:65-70).

## Method
- Driver: `scratchpad/F/driver.py`, run with the base interpreter `/home/kyle/miniforge3/bin/python`, `QT_QPA_PLATFORM=offscreen` and `CUDA_VISIBLE_DEVICES=''`. It copies `gui.main()` except for `app.exec()`, then pumps `QEventLoop`s.
- Logs: `F/driver_stdout.log`, `F/driver_stderr.log`, `F/driver_result.json`.
- Run 1 died from a bug in my driver (`plotItem` attribute) after the grid. Its logs and images are in `F/shots_run1/`, `F/driver_*_run1.log`. Run 2 finished cleanly (rc=0). `F/probe2.py` was a follow-up probe.
- Safety patches applied inside the driver process only:
  - `sentiment._llm_score_batch` and `try_llm_upgrade` were set to return None, so no LLM spend (the news tab used keyword scoring).
  - `gui.GUI_SETTINGS_FILE` and `NEWS_CACHE_FILE` were redirected to scratch copies.
  - `_maybe_write_baseline` was intercepted. It would have written `account_baseline.json` with baseline_equity=100000.0.
  - `QMessageBox`/`QDialog.exec` were suppressed and the report dialog text was captured to a file.
  - `open`/`os.replace` were traced in-process, and the repo mtimes were snapshotted before and after.
- Nothing was clicked except the Decision Report button. No order, close, flatten or halt path was touched.

## Import census (base env, py3.12)
- Every module gui needs imports: `dotenv`, PySide6, pyqtgraph, numpy 2.2.6, pandas 2.3.3, alpaca_trade_api 3.2.0, finnhub, optuna 4.7.
- Also present in the base env: torch 2.10 (GPU-broken build, never loaded by gui), sklearn, numba 0.61.2.
- Missing in the base env: `alpaca` (alpaca-py), lightgbm, pyarrow, arch, hmmlearn. gui imports none of them.
- `import gui` takes 0.9–2.3 s and loads these repo modules: chart_core, design_tokens, hw_monitor, journal_stats, log_config, notify, stock_config, strategy_config, tax_lots, trading_utils. The only heavy dependency it pulls is `dotenv`.
- Lazy imports inside gui also succeed: sentiment, llm_client, llm_config, indicator_config, optuna.
- Nothing gui imports fails in the base env.

## 1. Launch
| step | result |
|---|---|
| get_api (legacy `alpaca_trade_api.rest.REST`) | 1.0 s; account ACTIVE, equity $122,089 |
| `TradingDashboard(api, app)` | 3.7–4.1 s; RSS 271→283 MB; 10 threads |
| after a 20 s pump | status bar `API: orders ERR×1`, `Pos: 6`, GPU 49 °C, RAM 2.5/7.6 GB; `FnG: —` until the news fetch lands |
| stderr | only the RecursionError tracebacks (below) and one alpaca 429 "sleep 3 seconds and retrying" on the snapshot call during the probe; there are no Qt warnings |

How the GUI handles stale or missing inputs:
- It degrades gracefully. The Cockpit shows `Crypto STALE (no ping)`, `Stock off-hours`, `journal: 142d ago`, "no risk data" and "no closed trades in 7d".
- The Models tab shows Crypto `Stale` 168 d and "no drift data".
- The Total P&L card is prefixed `~` because `account_baseline.json` is absent.
- The Markets heatmap and the LLM column fill from `llm_analysis.json` (LLM Age 180–188 d, shown in red).
- Pred and Meta-p show `—` except where the stale prediction cache has a value (BTC/USD Pred +1.8347).
- Initial screenshot: `F/shots/00_initial_Cockpit.png`.

## 2. Tabs × themes — all 96 kept
Screenshots are `F/shots/<Theme>__<i>_<Tab>.png`. There is one contact sheet per tab across the 12 themes: `F/shots/sheet_<i>_<Tab>.png`. Image statistics are in `driver_result.json → grid` (dominant-colour fraction and colour count over the body).

| Theme | Cockpit | Trading | Performance | News | Markets | Models | Logs | Settings |
|---|---|---|---|---|---|---|---|---|
| Batman | ok | ok¹ | ok² | EMPTY³ | ok⁴ | ok⁵ | ok | ok |
| Joker | ok | ok¹ | ok² | EMPTY³ | ok⁴ | ok⁵ | ok | ok |
| Harley Quinn | ok | ok¹ | ok² | EMPTY³ | ok⁴ | ok⁵ | ok | ok |
| Two-Face | ok | ok¹ | ok² | EMPTY³ | ok⁴ | ok⁵ | ok | ok |
| Salander | ok | ok¹ | ok² | EMPTY³ | ok⁴ | ok⁵ | ok | ok |
| Black Metal | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Bubblegum Goth | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Dark | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Space | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Money | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Terminal | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |
| Paper (light) | ok | ok¹ | ok² | ok | ok⁴ | ok⁵ | ok | ok |

"ok" means it rendered, had content and raised no exception attributable to that tab. The RecursionError fires on every theme switch after the first (11×) and from the fetcher's `on_stocks` (5×), whichever tab is showing.

Footnotes:
1. Open Orders and Recent Fills are empty because of Bug 1.
2. The stat-card values are clipped vertically.
3. The News tab is empty for about 70 s after a cold start (defect D6).
4. The chart is empty until a symbol is changed (D7), the table shows one row, and labels are clipped.
5. The Models tab content is squashed, with overlapping text (D5).

Theme colours: I checked the Cockpit, Markets and Models contact sheets. The palette follows the theme everywhere, including the heatmap, the chart background, the accent tab underline and the zoom-button highlight. Black Metal is intentionally monochrome and Paper (light) is readable. I saw no widget stuck on the previous theme's colours. Note that D2c means the P&L cards are re-tinted only on the next account tick.

## Defects (file:line → evidence)
| # | sev | defect | where | screenshot / evidence |
|---|---|---|---|---|
| D1 | HIGH | **The orders pagination cursor uses the wrong format.** `new_until = str(oldest.submitted_at)` gives `'2026-04-07 19:45:17.419822+00:00'`, a pandas Timestamp with a space. Alpaca answers `APIError("invalid format for until; format: '2006-01-02T15:04:05Z'")`, reproduced standalone. The account has more than 1,100 orders, so page 2 always fails and the whole fetch is dropped. As a result: Open Orders and Recent Fills are empty, the tax cards show `—`, the status bar reads `API: orders ERR×N`, and the alert feed says "orders stream failing". The timer backs off from 30 s to 120 s. A `datetime` from alpaca-py would hit the same problem, since `str()` also puts a space in it. | gui.py:1562 (in `fetch_orders`, ~1518-1576) | `shots/Batman__1_Trading.png`, `Batman__2_Performance.png` (tax row), `00_initial_Cockpit.png` status bar; `stream_health.orders.fails=5` |
| D2 | HIGH | **`NumericTableItem.__lt__` falls back to `super().__lt__(other)`.** On PySide6 6.8.0.2 that call dispatches back into the Python override, giving "RecursionError … maximum recursion depth exceeded" and a False result. Reproduced standalone with two items and no UserRole. Consequences: **(a)** every text-column sort is wrong — the Positions table header shows `Symb ▲` but the rows are XRP, SOL, LINK, ETH, DOGE, BTC. **(b)** `_sync_stock_table` raises at `tbl.setSortingEnabled(True)` (gui.py:6318), so `tbl.blockSignals(False)` (6319), the selection restore and the scroll restore never run. The probe measured `_stock_table.signalsBlocked()==True` both after boot and after a theme switch; a later row-select still worked, so the lock is intermittent. **(c)** `_restyle` aborts at gui.py:3197 on every theme switch, so the P&L card re-tint after it (3199-3215) is skipped. | gui.py:65-70 (raised via 6318, 3197) | `00_initial_Cockpit.png` (unsorted positions); `F/driver_stderr.log` (16 tracebacks) |
| D3 | MED | **The price chart's y-axis is pinned to 0.** BTC candles are squashed into the top 5 % of an axis running 0–85k. The likely cause is the always-visible, empty `FillBetweenItem` ATR band. In a synthetic pyqtgraph test an empty fill moves autorange from [80k, 85k] to [-4k, 89k]; a hidden InfiniteLine does not. The in-GUI confirmation was inconclusive because the chart had not loaded within the probe window. | gui.py:5024-5027 | `shots/chart_BTC-USD.png` |
| D4 | MED | **Layout breaks at the default size of 1280×800 (minimum 1100×700).** Only the Settings tab has a `QScrollArea` (gui.py:7095). Performance: stat-card values clipped at the bottom. Markets: the stance table shows only one row, the last heatmap row is cut off, and the zoom buttons show one glyph each. Cockpit: positions headers are clipped ("ymb", "rrent Pr", "1kt Valu"). Models: report buttons clipped ("dicator Lead/Lag (Crypt"), plus the Models crunch in D5. | gui.py:2427-2428 (size); tab builders 3290-4777, 4905-6609, 6609-6960 | `Batman__2_Performance.png`, `Batman__4_Markets.png`, `00_initial_Cockpit.png` |
| D5 | MED | **The Models tab is crushed vertically**, identically in all 12 themes. The Model Status table shows only the Crypto row (Stock is hidden). The Shadow/Promotion and Meta-gate boxes show overlapping, clipped text. The pipeline progress bar overlaps its buttons. The Hardware box is empty. "Cost: $0.000/$1.00" overlaps "Pro: 1000/1000". | `_build_models_tab` gui.py:6609-6960 (no scroll area) | `Batman__5_Models.png`, `sheet_5_Models.png` |
| D6 | LOW | **The News tab is blank for about 70 s after a cold start.** `news_cache.json` (May) is older than `NEWS_CACHE_MAX_AGE_DAYS=7` and is dropped. The first fetch then scores 74 articles with keywords plus full-text fetch (or with the LLM, in production) on the slow thread, and chart/stock requests queue behind it. | gui.py:309 (`NEWS_CACHE_MAX_AGE_DAYS = 7`), fetch_news 1636-1790, burst order 1400-1410 | `Batman__3_News.png` (empty) vs `Space__3_News.png` (filled) |
| D7 | LOW | **The Markets chart is blank on first visit.** No chart is requested at startup or on tab show. Only a symbol change or the 120 s `_chart_timer` calls `_request_chart`. All 12 Markets screenshots show an empty 0–1 plot with no title. | gui.py:2617-2619, 5898-5901 | `sheet_4_Markets.png` |
| D8 | LOW | **The stale-report reason is mislabelled.** `gate_panel_model` labels any stale report "no API when generated — counterfactuals not priced". Here the real cause was `api_available=None` (no journal rows in the window; decision_report.py:775-777). The GUI button hardcodes `--days 30`, but the journals are 142 d old, so on this device it always produces the empty stub. | chart_core.py:1109-1114; gui.py:6913-6915 | driver_result.json `report.gate_widget__gate_attr_label`; `shots/report_trading_tab_gate_attr.png` |
| D9 | LOW | **Positions with zero cost basis are not flagged.** Alpaca returns `avg_entry_price=0`, `cost_basis=0` for all 6 crypto positions (verified via the API, so this is an upstream data anomaly). The GUI shows Avg Entry $0.00, Unrealized equal to full market value, and P&L % +0.00%. The status bar then reads `Unr: $121,996`. | gui.py:7993-8000, 1494 | `00_initial_Cockpit.png` |
| D10 | LOW | **The subprocess `LD_LIBRARY_PATH` gets a trailing `:`** when the parent has none set. The empty element means cwd (BASE_DIR) is searched for shared libraries. This is a hygiene issue only. | gui.py:107-108 | — |

Already-documented items I saw but did not re-count: the `_THEME_IMAGES["Salander"]` fallback renders an SVG logo (fine), and the 8.2 MB app icon loads lazily in 0.01 s.

## 3. Report launchers — environment audit
All jetson-python launches go through `_engine_python()` (gui.py:93-95 → `/home/kyle/miniforge3/envs/jetson/bin/python` when present) and `_engine_env()` (gui.py:98-109).

| site | gui.py lines | what it runs | LD_LIBRARY_PATH | LD_PRELOAD | works? |
|---|---|---|---|---|---|
| Refresh Selected (LLM) | 5541-5557 | `-c llm_analyst.refresh_one(sym, type)` | jetson/lib | **no** | yes (see below) |
| Refresh All (LLM) | 5650-5662 | `llm_analyst.py --refresh-all` | jetson/lib | **no** | yes |
| Report runner `_run_report_clicked` | 8985-9013 | decision_report, beta_ledger (`--json beta_report.json`), indicator_leadlag ×2, gap_audit (`--json tmp`), llm_eval ×2, execution_report — buttons at 6913-6941, gap audit via 9159-9169 | jetson/lib | **no** | yes |
| Pipeline restart `_restart_launch` | 9455-9465 | `run_pipeline.py … --skip-harvest --bot-only` | jetson/lib + cusparselt | **no** | yes (import-verified, not launched) |

No site sets `LD_PRELOAD`. The memory note says both variables are required and that the env should be copied from `_launch_training`, but that function no longer exists; every site now uses `_engine_env`.

I tested the env `_engine_env()` actually builds, spawned from the base env:
- The jetson python passes `import torch; import sqlite3` (the CXXABI crash case), in both orders.
- All seven launched modules and `run_pipeline` import cleanly.
- With a bare env (no `LD_LIBRARY_PATH`), `import torch; import sqlite3` reproduces `CXXABI_1.3.15 not found (libicui18n.so.78)`.

So `LD_LIBRARY_PATH=jetson/lib` is sufficient on its own and every site is functionally correct. The memory note is stale on this point.

Two side notes:
- None of the report CLIs imports torch; each import peaks at 33–98 MB.
- The subprocesses inherit the GUI's env without `CUDA_VISIBLE_DEVICES=''`. That is harmless for the report CLIs, and intended for the pipeline restart.

**Triggered through the GUI's own path:** `win._decision_report_btn.click()`.
- It spawned `['/home/kyle/miniforge3/envs/jetson/bin/python','-u','/home/kyle/trader/decision_report.py','--days','30']`, exited rc 0 after 1.9 s, and the status read "Decision Report (30d) complete".
- The dialog text (captured in `F/report_dialog_Decision_Report_(30d).txt`) was "No journal entries found."
- `decision_report.json` was rewritten (mtime changed; `generated 20:23:02`, `days 30`, stale stub).
- The GUI then re-read it: the freshness strip showed `decision_report: 1s` and the Trading-tab gate box showed the STALE banner (`shots/report_trading_tab_gate_attr.png`, `report_models_tab_after.png`). **So the output lands where the GUI reads it.**
- I then restored the prior content, a 109-byte stub from another agent (D), backed up at `F/decision_report.json.before_gui_click`.
- Later, `decision_report.json`, `llm_eval_report.json` and `execution_report.json` were all gone from the repo root. I did not do that — my process only wrote `decision_report.json` (the restore) and `logs/trader.log`. Agent D appears to have cleaned up its own artifacts.

## 4. Control channel
- `pipeline_command.json`: absent before and after. There was no `.tmp` either. The in-process write tracer shows no attempt to write it.
- `gui_settings.json` (mtime 2026-03-31) and `news_cache.json` (2026-05-05) were unchanged because they were redirected. Without the redirect, cycling the 12 themes would have rewritten `gui_settings.json` 11 times (`_on_theme_changed`, gui.py:2926-2928), and closing would have rewritten `news_cache.json`.
- `account_baseline.json` was intercepted, not written.
- `retrain_trigger.json` and `trading_halt.flag`: never touched.
- The repo mtime diff over the run also showed `.gpu.lock`, `logs/trader.log` and some py312 `__pycache__` files (run_pipeline, shadow, monitor_drift …). gui never imports those modules, and they are not in my process's write trace, so they came from concurrent agents. The one exception is `logs/trader.log`, which the gui import touches through log_config (known smell).

## 5. Charts
- `tests/test_chart_core.py` in the jetson env: **96 passed** (0.82 s).
- Also run: `test_design_tokens.py`, `test_gui_charts.py` and `test_gui_contracts.py`. With test_chart_core that makes **130 passed** in the jetson env; the base env has no pytest.
- Markets tab, BTC/USD fetched read-only: daily payload of 365 bars in 2.0 s, titled "BTC/USD (1M)", candles, volume, last-price line and LLM detail panel rendered (`shots/chart_BTC-USD.png`). The y-axis defect is D3.
- AAPL is not in the universe combo (56 names), so I charted no stock symbol.
- Performance equity curve and daily P&L bars rendered from Alpaca portfolio history (`shots/chart_performance_equity.png`, `Batman__2_Performance.png`). The equity plot is squeezed to about 30 px tall (D4).

## 6. Resources
- VmHWM after all 96 tab/theme visits, charts and the report: **332 MB** in run 2 (349 MB in run 1). VmRSS was 332 MB at the end. There were 10–14 threads (the main thread, the hot/slow fetchers, the log tailer and Qt workers). Construction alone takes about 283 MB.
- Idle CPU, measured with getrusage across all threads:

| window | wall | CPU s | % of one core |
|---|---|---|---|
| Cockpit | 60 s | 1.14 | **1.9 %** |
| Markets (stocks fetch active) | 60 s | 0.86 | **1.4 %** |
| Logs | 30 s | 0.24 | **0.8 %** |

Timers in gui.py (values measured live match the code):
- Hot fetcher: account 10 s, positions 5 s, orders 30 s (backed off to 120 s after failures), hw 5 s. Defaults are at gui.py:337-340 and the timers start at 1361-1375.
- Slow fetcher: news 300 s, stocks 30 s (throttled to 120 s when the Markets tab is not visible; 1384-1390).
- LogTailer: 2 s (stat of 3 files; 2172-2174).
- Other timers: `_model_timer` 60 s (2600-2602), `_perf_timer` 300 s (2606-2608), `_clock_timer` 30 s (2611-2613), `_chart_timer` 120 s (2617-2619), `_chart_stale_timer` 30 s (2622-2624).
- One-shot baseline fetch at 8 s (2589). Report/LLM/restart poll timers run at 1–2 s only while a child process is alive.

**Verdict on polling:** CPU is negligible for an 8 GB box shared with the bots and training. The real cost is Alpaca API traffic on the key the bots share:
- About 18 requests/min from account plus positions.
- Snapshots every 30 s.
- Once D1 is fixed, `fetch_orders` walks up to 10 pages (1,000-order cap) every 30 s, which is up to about 20 requests/min just to feed the tax card.

A 429 retry was already observed during the probe while other agents were active. I'd consider stretching the orders cadence, or caching tax history, once D1 is fixed.

## Files
- Report: `scratchpad/reports/F_gui.md`
- Driver and probes: `scratchpad/F/driver.py`, `probe2.py`, `census.py`, `census2.py`, `envprobe.py`, `orders_probe.py`, `sheet.py`
- Results: `scratchpad/F/driver_result.json`, `driver_stdout.log`, `driver_stderr.log`, `grid_table.md`
- Screenshots: `scratchpad/F/shots/` (96 grid PNGs, 8 contact sheets, chart, report and probe grabs); run-1 set in `F/shots_run1/`

No production files were edited.
