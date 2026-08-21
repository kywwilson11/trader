# GUI Review — 2026-07-21

> **IMPLEMENTATION STATUS (updated 2026-07-22): ALL 6 PHASES COMPLETE — uncommitted, awaiting owner
> review + Jetson visual pass.** The §11 roadmap was executed end to end. gui.py 5654→10261 lines;
> chart_core 612→928. New pure modules (all Mac-tested): `tax_lots.py`, `journal_stats.py`,
> `design_tokens.py`. New producers (behavior-neutral instrumentation): shadow_status.json,
> enriched prediction caches, persisted LLM v2 dossier, command_result.json ack, per-book heartbeats.
> Assets: `fonts/` (Inter + IBM Plex Mono, OFL) + `logos/96/` (prescaled). Verification: every gui
> batch kept `tests/test_gui_contracts.py` + `tests/test_gui_charts.py` green and self-ran
> `scripts/ab_check.sh`; final consolidated gate = **298 passed / 1 skipped** on new+contract tests,
> full-suite ab_check **zero regressions**, all files py_compile. **Not yet verified: live rendering
> on the Jetson (no PySide6 on the dev Mac)** — see the per-phase Jetson checklists collected in
> §13. Nothing committed (repo convention: owner commits).



**Scope:** `gui.py` (5,654 lines) + `chart_core.py` (612 lines). Full-system deep review, function by function.
**Method:** 6 per-cluster code critiques + 1 best-in-class benchmark (web-researched) + 1 mechanical
data-surface catalog, synthesized with cross-cluster verdicts. Charts math was *executed and verified*
against synthetic numpy on the dev Mac. The 2026-07 module review never covered gui.py/chart_core.py —
this is its GUI counterpart. GUI work is instrumentation (safe to ship directly per conventions);
items marked **[cross-file]** touch producers outside gui.py and need coordination; items marked
**[Jetson]** can only be visually verified on the Jetson.

---

## 0. Executive verdict

**The GUI is a competent Alpaca account viewer wearing ten costumes, bolted to the most interesting
trading system it never talks about.**

What's genuinely good: `chart_core.py` math is verified correct (LTTB, OHLC aggregation, max-DD,
WCAG contrast math — no P0s); the Jetson-aware rendering architecture (fingerprint memo, `QPicture`
candle cache, LTTB downsampling) is the *right* architecture; the GUI↔pipeline JSON IPC is careful
(atomic writes, TOCTOU-safe renames, stale thresholds); the Performance tab (drawdown shading + HWM +
linked P&L bars) is near-Koyfin-grade; the chart staleness/status UX is a real strength; retrain
queueing logic is thoughtful.

What's wrong, in five diagnoses (§9): **(1)** it surfaces almost nothing that makes this system unique —
no heartbeat, no risk headroom, no shadow/promotion status, no gate attribution, no conviction/sizing,
no journal analytics (the data-surface catalog proves ~20 artifact families are produced and never read);
**(2)** data is discarded at the last mile — the LLM dossier, `exit_pnl` on markers, prediction-cache
richness, shadow reports all die one step before the screen; **(3)** three control paths can hurt the
book — manual trade (no confirm, bypasses all sizing policy), Restart (duplicate-orchestrator race),
Flatten (fire-and-hope flag no one may consume); **(4)** some displayed numbers are wrong-by-construction —
Total P&L hardcodes a $100k baseline, tax basis silently truncates at 100 orders; **(5)** "beautiful"
fails on fundamentals, not taste — declared fonts are never loaded (Jetson renders DejaVu Sans), no
tabular numerals anywhere, widgets and charts literally use two different greens.

**Benchmark score: 7 clean passes, 16 partial, 22 absent — of 45 best-in-class checklist items (§1).**

**Platform verdict: KEEP PySide6 + pyqtgraph.** Every cluster and the benchmark agree: the constraint
set (single user, desktop, 8 GB shared) rewards native Qt, and the expensive-looking gaps (crosshair
readout, cursor zoom, linked panels, command palette) are all Qt-cheap. No web rewrite.

---

## 1. Benchmark scorecard (45 items, from TradingView / thinkorswim / IBKR / Koyfin / Bloomberg / FreqUI / Hummingbot / Jesse / OctoBot / 3Commas / Grafana)

✅ pass ◐ partial ❌ absent

**Overview / cockpit**
| # | Item | Score |
|---|---|---|
| 1 | Bot heartbeat + last-tick age visible in a 1-second glance | ❌ |
| 2 | Mode (paper/live/shadow/halt) unambiguous at all times | ◐ halt state only on Models tab |
| 3 | Equity curve including unrealized P&L | ✅ |
| 4 | Equity curve with drawdown shading / max-DD marker | ✅ |
| 5 | Range chips (1D/1W/1M/…) on time-series | ✅ |
| 6 | Today's P&L + open risk + net-liq tiles up top | ◐ no open-risk, no % |
| 7 | Per-symbol/pair performance table | ❌ |
| 8 | Top/bottom performers surfaced | ❌ |
| 9 | Both books unified in one overview | ◐ one account blob, no split |

**Charts**
| # | Item | Score |
|---|---|---|
| 10 | Crosshair with live OHLC + indicator readout | ❌ date+close only |
| 11 | Cursor-anchored scroll-zoom + drag-pan + auto-fit | ❌ mouse disabled on all plots |
| 12 | Trade markers (entry/exit) on price chart | ✅ |
| 13 | Markers link to journal/decision entry | ❌ |
| 14 | Model prediction/signal/gate state on chart timeline | ❌ |
| 15 | Multi-pane indicator subgraphs | ◐ volume pane only |

**Positions / orders**
| # | Item | Score |
|---|---|---|
| 16 | Positions show %-to-stop / ATR exit levels | ❌ |
| 17 | Closed-trades journal, filterable/sortable by metric | ◐ 50-row fills list, no analytics |
| 18 | Manual actions require confirm + impact preview | ◐ close yes; manual trade NO; resume-halt no |
| 19 | Order shows position & risk impact before commit | ❌ |
| 20 | Per-trade gate attribution (why entered/vetoed) | ❌ |

**Bot-ops / model-ops**
| # | Item | Score |
|---|---|---|
| 21 | Pipeline job status with timestamps | ✅ genuinely good |
| 22 | Model freshness + champion-vs-challenger slot | ◐ challenger = mtime only |
| 23 | Data-staleness indicators | ◐ charts excellent, streams none |
| 24 | HW panel with threshold colors/gauges | ◐ good gauges, no disk, no per-process RAM |
| 25 | GPU-lock / process health | ◐ pgrep-based, no heartbeat |
| 26 | Drift-monitor status visible | ❌ |
| 27 | In-UI log viewer | ✅ mechanically sound |
| 28 | Backtest / policy-replay results in-app | ❌ raw stdout dialog only |
| 29 | Promotion/gate outcomes (DSR/Sharpe pass-fail) shown | ◐ badge without metrics |

**News / research**
| # | Item | Score |
|---|---|---|
| 30 | News linked to selected symbol | ❌ |
| 31 | LLM verdict per symbol with rationale | ◐ m/s/r only; dossier discarded |
| 32 | Calibration / decision-report surfaced visually | ❌ |

**Alerts**
| # | Item | Score |
|---|---|---|
| 33 | Persistent alert center with history + ack | ❌ |
| 34 | Alerts for ops failures (bot down, stale data, thermal) | ❌ |
| 35 | External notify (Telegram/webhook) status in-UI | ❌ |

**Design polish**
| # | Item | Score |
|---|---|---|
| 36 | Elevated-gray dark theme | ✅ |
| 37 | Tabular/monospace numerals, right-aligned, fixed decimals | ❌ |
| 38 | P&L sign not color-only (CVD-safe) | ◐ ± present, no luminance separation |
| 39 | CVD-safe categorical palette | ◐ hollow candles yes; heatmap red-green |
| 40 | Sparklines/micro-charts in tables & tiles | ❌ |
| 41 | Linked panels (symbol drives everything) | ◐ three inputs, three different rules |
| 42 | Global symbol search / command palette | ❌ |
| 43 | Keyboard shortcuts for core flows | ❌ |
| 44 | Density toggle / progressive disclosure | ❌ |
| 45 | Informative empty/loading/stale states | ◐ charts yes, elsewhere blank |

---

## 2. Functional group map

| Group | Contents | Overall verdict |
|---|---|---|
| A. Core monitoring | Dashboard tab, Performance tab, DataFetcher, status bar, lifecycle | Dashboard **OVERHAUL**, fetcher **OVERHAUL**, Performance **KEEP/IMPROVE** |
| B. Trading + News | Positions/orders/manual trade/tax, News tab | Manual trade **OVERHAUL**, tax **FIX**, news **IMPROVE** |
| C. Stocks/Markets | Watchlist, heatmap, LLM columns, universe mgmt | Table update path + LLM surface **OVERHAUL** |
| D. Charting | CandlestickItem, crosshair, chart_core, zoom/staleness | Math **KEEP**, interaction **OVERHAUL**, overlays **MISSING** |
| E. Ops/Models/Logs | Models tab, pipeline/bot controls, kill switches, reports, HW, logs | Controls **IMPROVE/FIX races**, shadow panel **MISSING**, Logs **OVERHAUL** |
| F. Theming + Settings | 10 themes, stylesheets, palette math, Settings tab | Color system **KEEP**, tokens/type/fonts **OVERHAUL**, Settings **EXPAND** |

---

## 3. Group A — Core monitoring & data plumbing

| Function | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| `NumericTableItem` | 47 | KEEP | — | correct numeric sort |
| `fmt_money`/`fmt_pct` | 815–830 | IMPROVE | P2 | fetch failure renders as a real "$0.00" — use "—" sentinel for missing |
| `fmt_time*`, `pnl_color` | 833–872 | KEEP | — | |
| `make_card`/`_set_card` | 874, 4648 | IMPROVE | P3 | inline stylesheet keeps stale theme color until next data tick |
| status/command IO | 894–930 | KEEP | — | atomic tmp-rename writes — solid |
| **DataFetcher threading** | 931–1008 | **OVERHAUL** | **P1** | ALL streams on one QThread event loop; `fetch_news` (Finnhub + up to 10 sync LLM calls, admits "minutes") **freezes account/positions/orders while it runs**. Split hot (account/pos/orders/hw) vs slow (news/stocks/history) workers |
| fetchers + error paths | 1009–1083 | IMPROVE | P2 | one shared `_error_count`, reset by any account success → a permanently-dead stream is masked. Per-stream last-ok + fail count. No backoff (P3). ~22 req/min steady — no rate risk |
| init/toolbar/clock | 1901–2064 | KEEP/IMPROVE | P3 | seconds wall-clock → replace with market clock (session open/closed + next-open countdown via Alpaca `get_clock()`); two status labels both prefixed "GPU:" |
| **Dashboard tab** | 2307–2354 | **OVERHAUL** | **P1** | landing = 5 cards + positions + **tax estimation for a paper account**. No heartbeat, no risk, no alerts, no recent actions. See cockpit spec §10 |
| `on_account` | 4200–4244 | **FIX** | **P1** | `total_pl = equity − 100_000` hardcoded baseline — wrong after any reset/`.clean_slate`; "Day P&L" = since prior *stock* close, mislabeled for the 24/7 crypto book; no % anywhere |
| `on_positions` | 4245–4305 | IMPROVE | P2 | full table rebuild + fresh Close-button lambdas every 5s (drops selection 12×/min); ignores Alpaca's `market_value`, recomputes qty×price |
| Performance tab | 2529–2634, 4312–4370 | KEEP/IMPROVE | P2 | strongest screen (HWM + drawdown fill + linked P&L bars). Missing: intraday "today" view (min zoom = 1W@1D), live current-DD badge, benchmark overlay (→ §6) |
| `closeEvent`/`main` | 5571–5654 | KEEP | P3 | textbook shutdown; minor news-cache write race with a wedged fetcher (atomic writer → lost update at worst) |

**Missing (A):** bot-heartbeat strip **P1** · open-risk gauge vs `MAX_BOOK_RISK_PCT` **P1** · alerts feed
(halt is currently invisible from the overview) **P1** · last-actions/fills feed **P1** · today sparkline +
session boundary **P2** · live drawdown badge **P2** · realized-vs-unrealized per book **P2** · per-stream
freshness ages **P2** · real connection health via `get_clock()` **P2**.

---

## 4. Group B — Trading tab + News tab

| Function | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| **`_manual_trade`** | 2441–2494 | **OVERHAUL** | **P0** | one click fires a market order: **no confirmation**, **bypasses the entire sizing/risk policy** (no `MIN_ORDER_NOTIONAL`, no book-risk cap, no buying-power check), no qty sanity bound, silent qty-over-notional precedence, and a SELL with no position **opens a short on a long-only book**. Spec: confirm dialog (symbol/side/resolved qty↔$/est cost/current pos/post-trade BP) + "size by policy" button reusing `_compute_position_size` + short-block + marketable-limit default |
| `_close_position` | 2495–2528 | IMPROVE | P1 | confirm exists, but status text goes to a *Trading-tab* label while the button lives on *Dashboard* — feedback invisible; confirm lacks qty/P&L; no partial close |
| `_apply_trade_filter` | 4656–4705 | IMPROVE | P1/P2 | hardcoded crypto set drifts from `stock_config` (pool coins misclassified); **no cancel-order action** on the open-orders blotter; side-color painted on every cell |
| **`estimate_taxes`** | 1637–1725 | **FIX** | **P1** | cost basis from an **unpaginated 100-order window** — older buy lots fall out, SELLs match empty lots and are *silently dropped*: the tax number is wrong-by-construction beyond ~100 orders. Also: MinTax isn't the IRS default, wash-sale applied to crypto (has none), flat 37/20/5% rates, `>=365` LT boundary off by one. Paginate, relabel "indicative (paper)", make rates editable, exclude crypto wash-sale — and move OFF the landing screen |
| News tab build | 2635–2748 | IMPROVE | P2 | no symbol column (tracked internally!), sentiment not sortable, no scale legend |
| `fetch_news` | 1110–1280 | IMPROVE | **P1** | company news hardcoded to 10 tickers, **decoupled from `load_stock_universe()`** — hold AAPL, never see AAPL news, while the "My Universe" filter filters a feed that never contained your names. Combined sentiment dominated by those 10 high-beta names. LLM upgrades run inside the fetcher thread (see A). Feed failures swallowed silently |
| news cache | 183–260 | KEEP | — | lean, atomic, capped — good |
| `on_news` | 4450–4523 | KEEP | — | sensible thresholds, defensive FnG fallback |
| `_apply_news_filter` | 4524–4612 | IMPROVE | P2 | O(articles×symbols) regex per refresh on the UI thread; 1–2-char tickers (`A`, `ON`, `IT`) false-match prose |

**Missing (B):** manual-trade confirm+policy sizing **P0** · **exit-distance columns (stop/trail/TP/%-to-stop)
on positions** — highest-value read-only add, data in `policy_exits` + policy config + `position_state.json`
**P1** · journal analytics (win rate, expectancy, profit factor, per-symbol) **P1** · "why did the bot
trade this" gate attribution (decision_report data) **P1** · cancel-order + order lifecycle **P1** ·
position drill-down **P2** · feed-staleness badge **P2** · book-risk headroom readout **P2**.

---

## 5. Group C — Stocks/Markets tab

**Headline: the LLM produces a rich decision dossier that never reaches this screen.** `analyze_trades`
computes `p_up, conviction, abstain, key_risks, event_flags` (llm_analyst.py:1032–1036) but
`_save_analysis` persists only `m/s/r/bull/bear/ts/model` (llm_analyst.py:637–645). The GUI reads only
`llm_analysis.json` → it *cannot* show the advisor's actual decision payload. With LLM utilization an
owner priority, this is the single highest-ROI fix in the review. **[cross-file]**

| Function | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| `_build_stocks_tab` | 2749–2928 | IMPROVE | P2 | chart dominates, the "what does the system think" table is a cramped bottom strip — inverted hierarchy vs Koyfin/TradingView |
| watchlist add/remove | 2929–2974 | IMPROVE | P2 | no ticker validation (garbage flows into an LLM subprocess); remove acts on combo not table selection; no undo |
| **`on_stocks` table path** | 3393–3565 | **OVERHAUL** | **P1** | full teardown every 30s: recreates every cell, **destroys selection/scroll while you read a dossier**, cache replaced wholesale (torn read → all LLM columns blank for a cycle). Spec: stable sym→row map, in-place updates, change-flash, selection preserved by symbol |
| heatmap | 3402–3435, 3289 | IMPROVE | P2 | flat equal-weight day-chg grid; no sector grouping/cap sizing (Finviz); click moves chart but not table/detail |
| row select/detail | 3018–3080 | IMPROVE | P1 | detail = three prose lines (bull/bear/summary); age math duplicated in 3 places; dblclick reaches into Trading tab internals |
| **LLM refresh subprocess** | 3081–3288 | **OVERHAUL** | **P1** | user-typed symbol **f-string-spliced into `python -c`** (injection; fix = argv); refresh-one and refresh-all have **no cross-guard** — two heavy Jetson-env interpreters can run beside the bots on 8 GB; "Refresh All" = whole-universe real-money spend from one click, no cost estimate/confirm/progress; likely doesn't request v2 anyway |
| `fetch_stocks` | 1400–1503 | IMPROVE | **P1** | **not gated by tab visibility** (chart auto-refresh is!) — Alpaca snapshots + full rebuild every 30s while you're on Logs. One-line class of fix |

**Missing (C):** persist+show the v2 dossier (conviction chips, p_up, abstain, risks, event badges) **P1
[cross-file]** · model stance columns — meta-label p, cost-gate, regime (loops' `write_prediction_cache`
drops them today: crypto_loop.py:246–270) **P1 [cross-file]** · "do I hold this?" position join **P1** ·
refresh-stale-only + $ cost meter **P1** · filter/search + sector/asset grouping **P2** · echo-gap
(LLM p_up vs model pred agreement) inline **P2** · earnings/FOMC badges (already computed in
`_compute_event_lines`) **P2** · sparkline column **P3**.

---

## 6. Group D — Charting subsystem

**Math verified by execution on this Mac: LTTB endpoints/monotonicity, OHLC bucket boundaries,
max-DD=50% test case, contrast ratios 9.15/5.04 — all correct. No P0s. The gap is interactivity +
overlays, not rendering.** The irony: interaction was disabled wholesale
(`setMouseEnabled(False)` at 2564/2576/2816/2838), but the `QPicture` candle cache + fingerprint memo
that make interaction *cheap* are already built.

| Piece | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| `ChartCrosshair` | 434–486 | IMPROVE | P1 | snap logic good; readout = **date + close only** on a candle chart — needs O/H/L/C/%chg/volume + y-axis price label |
| `CandlestickItem` | 487–547 | KEEP | — | cached QPicture, hollow-up/filled-down (CVD-safe) — well designed |
| chart_core math | 56–224 | KEEP | — | verified correct |
| `perf_stats` | 128–151 | IMPROVE | P1 | only total/best/worst/maxDD — **no Sharpe/Sortino/win-rate/vol/CAGR** (pure-numpy, Mac-testable adds) |
| `build_price_view` | 335–447 | IMPROVE | P1 | no overlay plumbing at all |
| palette/contrast | 449–532 | KEEP | P2 | genuinely good color science; raise floor 2.5→3.0, make CVD luminance separation always-on |
| `load_trade_markers` | 536–612 | KEEP | P2 | solid, mtime-cached; **`exit_pnl` parsed then never used** — wins and losses render identically |
| `fetch_chart` | 1281–1399 | IMPROVE | P2 | no live tick — the forming candle never updates; fixed zoom→resolution map |
| staleness/status UX | 2276–2306 | KEEP | — | a real strength — keep |
| zoom/refresh flow | 2983–3392 | **OVERHAUL** | **P1** | 5 preset buttons swap window+resolution; **no pan, no wheel-zoom, no box-zoom anywhere**. Enable bounded x-pan + cursor-anchored wheel-zoom on the ViewBox; keep buttons for resolution |
| equity chart path | 4322–4370 | IMPROVE | P1 | no benchmark overlay, no log scale, thin stats |

**Missing (D):** **position entry/stop/TP lines on the live chart** (a trader cannot see where the stop
sits; data = positions + `strategy_config` mults + `position_state.json` trailing/hwm) **P1** · OHLC+vol
crosshair readout **P1** · fluid pan/wheel-zoom **P1** · SMA/ATR-band overlays from the system's own
features (**no VWAP — killed rev-07-02**) **P1** · SPY/BTC benchmark overlay on equity ("are we beating
buy-and-hold" = the low-beta thesis; `beta_ledger` logic exists) **P1** · Sharpe/Sortino/win-rate tiles
**P1** · exit-marker win/loss color + P&L tooltip (data already loaded!) **P2** · log-scale + %-return
toggle **P2** · last-price line **P2** · prediction/conviction overlay **P2** · session-gap compression
for stocks **P3** · underwater subpane **P3** · resolution decoupled from window **P3** · trade replay **P3**.

---

## 7. Group E — Models/Ops tab + Logs + hardware

| Function | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| model-status table | 3574, 5154–5222 | IMPROVE | P1 | `_model_deployed_ts` manifest-first logic is smart; but **`optuna.load_study()` + pickle unpickle for both books every 60s** on the shared 8 GB box — cache by mtime; ">3 = green" magic constant |
| **challenger cell** | 5169–5177 | **MISSING panel** | **P1** | shows "in shadow since ⟨mtime⟩" while `shadow.py:evaluate_shadow` computes DM-HLN stat, **p-value, n vs MIN_OBS=200, window age, hit-rate delta, promote/discard decision** — none persisted or shown. "A model registry with the evaluation column blanked out." **[cross-file: persist `shadow_status.json` each eval]** |
| pipeline-status panel | 5225–5435 | KEEP | P2 | genuinely good ops instrumentation (tri-state scores, rollback badges, stale bands). Fix: renders **bear/neutral/bull per-class accuracy** — the 3-class ensemble is gone (CLAUDE.md) — dead UI or stale key; add trials/sec ETA |
| retrain controls | 4738–4797, 5437–5498 | KEEP | P3 | confirmation + queue semantics + stale-trigger expiry — thoughtful |
| bot start/stop | 5054–5123 | IMPROVE | **P1** | **command rejection invisible** (pipeline rejects start-during-training with only a log line; GUI already said "Starting…"); **`_stop_bot` doesn't check pipeline-running** — a dead pipeline leaves the stop command on disk to fire on *next* startup; no double-click debounce. Needs `command_result.json` ack **[cross-file]** |
| halt toggle | 5124–5137 | IMPROVE | P1 | halting-fast-no-confirm is right; but **Resume is also one silent click** — confirm the un-halt |
| **flatten** | 5138–5153 | **FIX** | **P1** | writes a flag **only a running bot consumes** — if bots are wedged/dead: no halt, no liquidation, GUI says "requested". Fix: GUI-flatten also calls `set_halt()` directly + verifies positions→0 + shows pending state (`notify.flatten_requested()` exists, never called) |
| **restart pipeline** | 4948–5041 | **OVERHAUL** | **P1** | SIGTERM → 5s wait → **unconditionally launches new pipeline**. Mid-training shutdown legitimately takes >10s (Optuna child wait(10) then bot stop) → **two orchestrators, duplicate order flow**; wedged old = permanent dual. Also runs pgrep/sleep loops on the GUI thread. Fix: SIGKILL escalation + confirm-dead + confirm-alive, off-thread |
| reports runner | 4803–4915 | KEEP | P3 | tempfile-not-PIPE reasoning is sound; no cancel; `--days` hardcoded; leadlag hardcodes crypto parquet (stock unreachable); raw-stdout dialog is 2005-era (§10: parse the JSON artifacts instead) |
| HW gauges | 3722, 4370–4449, 1503–1584 | KEEP | P2 | clean sysfs readers, threshold colors. **No disk gauge** (full SD silently breaks status writes — `write_status` swallows OSError); no per-process RAM split (bots vs training — the number that matters on 8 GB); add ring-buffer sparklines |
| **Logs tab** | 3810–3837, 1590–1636 | **OVERHAUL** | **P1** | tailer mechanics sound (rotation, offsets, caps). As a console: **no severity coloring, no search/filter, no level filter, no pause, no jump-to-latest, no multi-file merge**; buffer trimmed mid-line. This is the incident surface and it's the barest tab. Cheapest 5× upgrade in the review |
| IPC (cross-cutting) | — | IMPROVE | P1 | atomic + TOCTOU-safe (good). Gaps: **no command acknowledgement** (write-and-forget, optimistic UI) and **liveness = status-file mtime only** (a wedged-but-alive bot shows green — no per-book heartbeat) |

**Missing (E):** shadow/DM-HLN panel **P1 [cross-file]** · in-app alert center (notify.py already emits
to Telegram/webhook; mirror in-GUI with unread badge + ack) **P1** · bot heartbeat + stale detection **P1
[cross-file: base_loop writes per-book heartbeat]** · log severity+filter **P1** · disk gauge **P2** ·
command audit trail (`gui_actions.log`) **P2** · retrain ETA **P2** · promotion-gate metrics (why a
rollback happened: Sharpe/DSR values into `phase_results`) **P2**.

---

## 8. Group F — Theming/design system + Settings

**Verdict: a color system, not a design system — "enthusiastically themed," not yet "professionally
beautiful."** The 13-role color contract per theme + `derive_chart_palette` (computed luminance,
`ensure_contrast`, hue-distance CVD guard) is the strongest part. Tokens stop at color: no type scale,
no spacing scale, ~136 scattered `setStyleSheet` strings.

| Feature | Lines | Verdict | Pri | Key finding |
|---|---|---|---|---|
| THEMES dict | 261–417 | IMPROVE | P1 | good 13-role contract; all 10 themes are dark high-chroma character pieces — zero restrained "terminal pro," zero light theme |
| **typography** | 1752 | **OVERHAUL** | **P1** | font stack (`Inter, SF Pro…`) is **never loaded** — zero `addApplicationFont` calls → Jetson renders DejaVu Sans. **No tabular figures on any money number** — digits jitter as values tick. Bundle Inter + Plex Mono, add a `numeric` role |
| color semantics | various | FIX | P2 | BUY/SELL hardcode Material `#2e7d32`/`#c62828` ignoring the theme; **table P&L uses raw `T["green"]` while charts use contrast-adjusted `pal["up"]` — two greens for one semantic on one screen**; stray `#444`/`#111111`/`gray`/`orange` literals (2915, 4294, 5337, 5344, 2641, 4076, 4081) |
| `apply_theme`/`_restyle` | 1726–2264 | IMPROVE | P1 | ~40 inline re-style blocks over hand-maintained widget lists — new widgets silently miss theming. Move to global QSS attribute selectors (`QFrame[card="true"]`, `[numeric="true"]`) — deletes most of `_restyle` |
| `make_card` | 874–891 | IMPROVE | P2 | sets properties no selector reads; themed via positional `layout().itemAt(0)` — brittle |
| **theme logos** | 764–809 | **OVERHAUL** | **P1** | **62 MB of full-res PNGs** (batman 7.9 MB, black_metal 9 MB) decoded uncached to draw an **80 px icon** on an 8 GB box. Pre-scale to a shipped `@96` set + memoize |
| contrast math | cc 449–532 | KEEP (tune) | P2 | floor 2.5 < WCAG-AA-graphical 3.0; CVD guard only fires on rare hue-collapse — make luminance separation always-on |
| Settings tab | 3838–4199 | EXPAND | **P0/P1** | clean structure, auto-save right, async LLM test well done. **But the shipped multi-provider LLM (Claude/OpenAI) is invisible** — only Gemini+FMP keys wired (3882–3894), routing combos Gemini-only (3955), success string hardcodes "Connected to Gemini" (4162), while `llm_config.py:7–53` documents the full provider-preference contract |

**Missing (F):** multi-provider LLM settings **P0** · refresh-cadence spinboxes (a direct Jetson load
lever) **P1** · Notifications group (Telegram/webhook/healthcheck set + "send test") **P1** · visible
safe-mode / shadow-mode toggle **P1** · chart defaults **P2** · logo on/off **P3**.

**Design language spec (adopt incrementally, byte-compatible first):** color tokens `bg.base/raised/
overlay/inset · text.hi/mid/dim · accent(+soft/glow) · pnl.up/down/warn/info` (ONE semantic set feeding
both widgets AND charts via `derive_chart_palette`); type scale `24/700 · 15/600 · 13/500 · 11/600 +
mono-tabular numeric role`; spacing grid `4·8·12·16·24`; radii `4/6/8/10`. Themes become color packs
over this spine; then add one restrained pro-dark + one paper-light so the set spans professional, not
just theatrical. All stylesheet/string work + one font load — no per-frame cost.

---

## 9. Cross-cutting diagnoses (the five that matter)

1. **It's a quote screen, not a mission control.** The catalog (§below) proves ~20 artifact families are
   produced and never read: journal skip/conviction/sizing rows, `position_state.json`, shadow preds +
   reports, drift state + PSI, `decision_report.json` (launched as text, JSON never parsed),
   `llm_eval_report.json` + `llm_advisor_report.json` (the ONLY report family with no GUI presence at
   all), `execution_report.json`, `account_risk_registry.json`/GATE-1 headroom, heartbeats, adaptive
   state. The GUI shows what any broker app shows and almost nothing this system uniquely knows.
2. **Data dies at the last mile.** Five independent instances of the same pattern: LLM dossier discarded
   at `_save_analysis`; `exit_pnl` loaded then unused; prediction cache drops meta_p/conviction/gates;
   `shadow_report` written into the manifest but never read; beta-ledger launched without `--json`.
   The pipes exist; the final connection was never made.
3. **Control paths need real-money discipline.** Manual trade (P0), Restart race (P1), Flatten
   fire-and-hope (P1), silent command rejection + unguarded stop (P1), one-click un-halt (P1).
4. **Wrong numbers on a money screen.** `equity − 100_000`, 100-order tax basis, mislabeled Day P&L,
   sentiment average dominated by 10 hardcoded tickers. Wrong beats missing for damage.
5. **Beauty fails on fundamentals, not taste.** Unloaded fonts, proportional jittering digits, two
   greens, hardcoded hex escapes, all-dark novelty theme set, no type/space tokens. All fixable
   cheaply; none needs a rewrite.

---

## 10. Target information architecture

Eight tabs stay (muscle memory), three get renamed/refocused; global affordances added:

- **Cockpit** (was Dashboard): mode banner (PAPER + halt/flatten state, always visible) · per-book
  heartbeat strip (bot alive? last cycle age? journal write age?) · equity + today-sparkline + live-DD
  badge · open-risk gauge (GATE-1 headroom, positions/book, largest name) · alert feed (halt, crash,
  stale, thermal, rollback — ack-able) · last-10-actions feed. Tax leaves.
- **Book** (was Trading): positions **with stop/trail/TP/%-to-stop columns** + per-position gate/
  conviction context · open orders with **Cancel** · fills · journal analytics block (win rate,
  expectancy, profit factor, per-symbol) · manual ticket behind confirm+policy-sizing.
- **Markets** (was Stocks): the "system's stance per name" table — price/chg + pred + meta-p + gates +
  conviction chip + LLM p_up/verdict + held? + event badges · linked selection (one click drives table,
  chart, news, detail everywhere) · sector-grouped heatmap · stale-only LLM refresh with cost meter.
- **Charts** (inside Markets): interactive (pan/wheel-zoom) · OHLC crosshair readout · position
  entry/stop/TP lines · SMA/ATR overlays · win/loss-colored trade markers with P&L tooltips ·
  last-price line.
- **Performance**: adds SPY/BTC benchmark overlay, Sharpe/Sortino/win-rate/CAGR tiles, log toggle,
  underwater subpane, per-book split.
- **Models**: adds shadow/DM-HLN promotion panel (p, n/200, window, decision trajectory) · drift panel
  (PSI per label) · retrain ETA · command ack states · report views that parse the JSON artifacts
  (decision_report, llm_eval, execution_report, beta --json, gap_audit) instead of stdout dumps.
- **Logs**: severity coloring, filter box, level filter, pause, jump-to-latest.
- **Settings**: + multi-provider LLM, cadences, notifications+test, safe-mode, chart defaults.
- **Global**: Ctrl-K command palette (symbol jump + actions) · keyboard shortcuts · per-stream
  freshness · design tokens + bundled fonts + tabular numerals everywhere.

## 11. Roadmap (phased; S/M/L effort)

**Phase 0 — Safety & truth** (do first; all Jetson-verify)
| # | Item | Effort | Notes |
|---|---|---|---|
| 0.1 | Manual-trade confirm + policy sizing + short-block + min-notional | M | P0 |
| 0.2 | Restart handshake: SIGKILL escalation, confirm-dead/alive, off-thread | M | P1 safety |
| 0.3 | Flatten self-halts + verifies + pending indicator | S | P1 safety |
| 0.4 | Command ack (`command_result.json` echo) + stop-bot guard + debounce | S+S | [cross-file run_pipeline] |
| 0.5 | Total-P&L baseline + Day-P&L labeling + % deltas | S | correctness |
| 0.6 | Tax: paginate basis, relabel indicative, crypto no-wash-sale, LT boundary | M | correctness |
| 0.7 | LLM refresh: argv (kill injection), single-flight lock, stale-only default | S–M | P1 |
| 0.8 | Settings: expose Anthropic/OpenAI providers + fix "Connected to Gemini" | M | P0 |

**Phase 1 — Cockpit + fetcher + logs**
| # | Item | Effort |
|---|---|---|
| 1.1 | Dashboard → Cockpit (heartbeat [cross-file base_loop], risk gauge, alerts v1, last-actions, today sparkline, DD badge) | L |
| 1.2 | Hot/slow DataFetcher split | M |
| 1.3 | Per-stream health + freshness ages | S |
| 1.4 | Logs overhaul (severity, filter, pause, jump) | M |
| 1.5 | Gate `fetch_stocks` on tab visibility | S |

**Phase 2 — Make the machine legible** (the differentiator)
| # | Item | Effort |
|---|---|---|
| 2.1 | Persist + render LLM dossier (conviction/p_up/risks/events) | S [cross-file llm_analyst] + M GUI |
| 2.2 | Shadow/DM-HLN panel (`shadow_status.json`) | S [cross-file shadow.py] + M GUI |
| 2.3 | Prediction-cache enrichment (meta_p, gates, conviction) + stance columns | S [cross-file loops] + M GUI |
| 2.4 | Exit-distance columns (policy + `position_state.json`) | M |
| 2.5 | Journal analytics view | M–L |
| 2.6 | Gate-attribution panel (parse `decision_report.json`) | M |
| 2.7 | Drift panel (`drift_state.json` PSI) | S |
| 2.8 | Structured report views + beta `--json` + gap_audit launcher | M |
| 2.9 | Cache Optuna best-score by mtime (kill 60s load_study) | S |

**Phase 3 — Charts become instruments**
3.1 position lines (M) · 3.2 OHLC crosshair readout (S–M) · 3.3 pan/wheel-zoom (S–M) ·
3.4 exit-marker P&L colors+tooltips (S) · 3.5 SMA/ATR overlays (M) · 3.6 equity benchmark overlay +
Sharpe/Sortino/win-rate (M; `perf_stats` adds Mac-testable) · 3.7 log toggle (S) · 3.8 last-price
line (S) · 3.9 volume-axis SI units (S)

**Phase 4 — Design system**
4.1 bundle Inter+Plex Mono, `addApplicationFont`, tabular numeric role (M) · 4.2 type/space tokens +
global QSS attribute selectors, shrink `_restyle` (L) · 4.3 one pnl.up/down semantic feeding widgets
AND charts; fold in stray hex (S) · 4.4 logo pre-scale@96 + memoize (S) · 4.5 contrast floor 3.0 +
always-on CVD luminance separation (S) · 4.6 pro-dark + light themes (M)

**Phase 5 — Productivity & polish**
5.1 unified linked selection (M) · 5.2 Ctrl-K palette + shortcuts (M) · 5.3 diff-update all tables +
change flash (M–L) · 5.4 news from real universe + symbol column + sortable sentiment (M) ·
5.5 Settings ops page (cadences/notifications/safe-mode) (M) · 5.6 sector heatmap (M) · 5.7 close-
position feedback + partial close (S) · 5.8 cancel-order buttons (S) · 5.9 market clock (S) ·
5.10 disk gauge + HW sparklines (S)

**What NOT to do:** no web/electron rewrite (unanimous — Qt is the right platform here) · no VWAP
overlay (killed rev-07-02) · no new TA indicator families on charts beyond what the pipeline already
computes (kill list) · don't rebuild the bear/bull per-class UI — remove or fix the stale key ·
no Grafana/external-stack dependency for the ops panels.

## 12. Kill-list & conventions compliance

All six clusters checked `research/KILL_LIST.md`; every proposal surfaces *existing computed outputs*
(exits, journals, shadow, drift, risk registry, llm_eval, beta ledger) or standard UX patterns — no
killed strategy/feature/data-source is rebuilt. VWAP explicitly avoided. GUI changes are
instrumentation/measurement-only → ship directly (no promotion gate). Items tagged **[cross-file]**
(llm_analyst persistence, loop prediction caches, shadow status file, base_loop heartbeats,
run_pipeline command ack) are also instrumentation but touch shared producers — coordinate, don't
silently drift, and verify on the Jetson.

**Review manifest:** 6× Opus 4.8 cluster critiques (core monitoring · trading+news · markets · charts ·
ops · design) + 1× Opus 4.8 web benchmark + 1× Sonnet 5 data-surface catalog; Fable synthesis,
scorecard, IA, and roadmap. Charts math verified by execution; other clusters code-read with line-ref
evidence; ops cluster verified the IPC contract against `run_pipeline.py`/`shadow.py`/`notify.py`.

---

## 13. Implementation log & Jetson visual-pass runbook (2026-07-22)

**All changes uncommitted. GUI logic verified by py_compile + Mac-runnable source-contract tests
(test_gui_contracts.py / test_gui_charts.py) + pure-module unit tests; live rendering needs the
Jetson.** Verify on-device in this order — each line is a thing to look at:

**Phase 0 — safety & truth**
- Manual trade (Trading tab): BUY/SELL now open a confirm dialog (symbol/side/resolved qty↔$/est
  cost/current pos/buying-power before→after/"MARKET order"); "Size by policy" fills notional;
  a SELL with no/insufficient position is hard-blocked (long-only); below-MIN_ORDER_NOTIONAL blocked;
  >50% equity blocked, ≥10% warned.
- Restart Pipeline (Models tab): stop→confirm-dead→launch→confirm-alive; kill a wedged pipeline and
  confirm it does NOT start a second orchestrator (SIGKILL at 10s, ABORT at 15s).
- Flatten: sets halt immediately + shows "FLATTEN PENDING (n)" banner until positions→0.
- Resume-halt now confirms; stop-bot with pipeline down warns; command REJECTED shows in the pipeline
  panel (start a bot during training → see the red ack).
- Cockpit cards: Total P&L no longer hardcodes 100k (writes account_baseline.json from the 1A history
  ~8s after launch; shows "~" until then); Day card reads "P&L (since prior close)"; both show $ (±%).
- Tax (now on Performance tab): "Est. Tax — MinTax, indicative (paper)"; " (incomplete basis)" appears
  only when >1000-order/365d truncation hit.
- Settings: Anthropic + OpenAI key rows + provider-selection + cross-provider role pickers; LLM test
  reports the provider/model that actually answered.

**Phase 1 — cockpit, fetcher, logs**
- Two fetcher threads: news/LLM churn must NOT freeze account/positions (watch during a news refresh).
- Status "API:" shows worst-stream health with a per-stream tooltip; a killed feed stays flagged.
- Cockpit: mode/halt banner, Crypto/Stock heartbeat chips (stock = gray "off-hours" outside RTH, NOT
  red), risk gauge from account_risk_registry.json, alerts feed, last-actions feed (last 7d journals),
  today sparkline, DD badge.
- Logs: ERROR red / WARNING yellow; regex + level filter; Pause; Jump-to-latest; no mid-line cut.

**Phase 2 — legibility**
- Markets table: 13 cols (Meta-p, Conv, Rank, Gate, Held); selection + scroll survive the 30s refresh
  while reading a dossier; price change-flash; regime chips; dossier panel shows v2 p_up/conviction/
  risks/events + echo-gap ("LLM agrees/fights model"); watchlist add validates + rejects dupes.
- Models tab: Shadow/DM-HLN panel (decision, n/min_obs, day X of Y, p-value, hit-rates) once
  shadow_status.json exists; Drift panel (PSI, thresholds 0.10/0.25); Optuna study not reloaded unless
  the db changed.
- Cockpit positions: Stop/TP (~estimates) + %→Stop (red <1%).
- Trading tab: gate-attribution box (after a Decision Report run) + journal-analytics (win-rate/
  expectancy/profit-factor cards + summary, loaded off-thread).
- Reports: Beta Ledger + Gap Audit show a parsed summary above stdout; Lead/Lag has Crypto/Stock.

**Phase 3 — charts**
- Price chart: drag-pan + cursor wheel-zoom (can't fly past data ±5%); "Reset view" + zoom presets
  snap back; a manual pan survives a data refresh. OHLC+%+volume crosshair readout + right-edge price
  tag. Entry/~stop/~TP lines when a position is open (vanish when flat). Exit markers green/red by P&L
  with "SOLD … (+x%)" tooltips. SMA20/SMA50/ATR-band toggles (persist; no draw across warmup).
  Last-price dashed line. Volume axis in K/M/B.
- Performance: Benchmark None/SPY/BTC overlay (normalized dashed); 5 new tiles (Sharpe/Sortino/Win/
  Vol/CAGR); log-scale toggle (hides DD wash while on).

**Phase 4 — design system**
- Fonts actually load (check a startup log line): Inter for UI, IBM Plex Mono for every numeric cell/
  card value/status readout — digits are tabular (don't jitter as they tick). Theme logos load from
  logos/96 (no multi-MB decode). All 12 themes recolor buttons/lines/tiles; two NEW themes: Terminal
  (pro-dark) and Paper (light — confirm charts + heatmap are legible on white). One P&L green/red
  across tables AND charts. Theme switch: cards re-tint instantly (no flash).

**Phase 5 — productivity & polish**
- Ctrl+K palette (symbols + actions; actions still fire their confirms); Ctrl+1..8 tabs; Ctrl+L logs;
  Ctrl+F filter; F5 refresh. Linked selection: heatmap/table/combo/detail/chart move together.
- News: company news rotates the real universe; Sym column; sortable sentiment; "Tracked avg" label.
- Settings ops: live refresh-cadence spinboxes; notification test; safe-mode chip + halt mirror; chart
  defaults. Heatmap: Crypto/Stocks grouping (sector data NOT available to the GUI — asset-class
  fallback) + Day%/Pred/Meta-p metric combo. Partial close 25/50/100%. Cancel-order buttons. Market
  session clock. Disk gauge (red <1 GB) + GPU-temp/RAM sparklines.

**Known / deferred (owner queue):**
- `_last_meta_p` never pruned (pre-existing; a delisted stock keeps its last meta-p in the cache) —
  surfaced by the prediction-cache lane, not fixed (out of that task's ownership).
- Partial-close uses 3dp for stocks (fractional day orders); a non-fractionable symbol surfaces the
  broker rejection in the status label.
- Card frame radius kept at 10 (panel) for geometry byte-compat; move to RADIUS["card"]=8 if desired.
- Reports still have no cancel button (P3, explicitly out of scope).
- Two-new-theme palettes and the light-theme chart legibility were verified by contrast/luminance MATH,
  not by eye — confirm on-device.

**Commit grouping suggestion** (owner decides; convention = owner commits): the producers +
command-ack + heartbeats are behavior-neutral instrumentation and could ship in one commit ahead of
the GUI; the pure modules (tax_lots/journal_stats/design_tokens/chart_core) + their tests are
independently valuable; gui.py is one large feature commit. Suggested message stem:
`feat: 2026-07 GUI overhaul — cockpit, model-registry legibility, interactive charts, design system`.
