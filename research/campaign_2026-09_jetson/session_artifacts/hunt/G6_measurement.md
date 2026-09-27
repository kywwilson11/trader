# G6 — measurement shelf: indisputable-improvement hunt (2026-09-26, Jetson, read-only)

Scope: decision_report, beta_ledger, execution_report, llm_eval, journal_stats, gap_audit, indicator_leadlag,
ic_diagnostic, rank_gradient, stage0_preds, naive_baseline, horizon_transfer, meta_curve, portfolio_backtest,
options_overlay, plus scripts/{ic_by_name, rank_gradient_report, reliability_report, sizing_cofire_report,
wave6_stage0, cscv_audit, funding_drift_audit, entry_timing_probe, horizon_transfer_report, naive_vs_blend,
meta_learning_curve, window_ab, crypto_spread_census, llm_qualify, prompt_ab, train_lexicon, connection_test}.
The orchestrator of this group reviewed decision_report, beta_ledger and llm_eval, and adjudicated the open items.
Three sub-hunts covered the rest: partA, partB and partC. Their full write-ups are included verbatim below.
Each sub-hunt finding was re-checked against its repro output before it was listed here.
All repros are in `hunt/G6/` (partB's are in `hunt/G6/partB_repro/`). No repo file was edited. No pytest was run.
Network use was limited to 2 read-only Alpaca bar requests (G6-1).

## Ranked index

| # | id | file:line | class | sev | one line |
|---|---|---|---|---|---|
| 1 | **G6-1** | llm_eval.py:217-219 | A | **high** | Stock fetch window is wall-clock `t0+(fb+6)h` while the horizon is stepped in bars. The SIP `end` is not clamped either. A run while the bots trade realizes **0 of 48** stock rows; a later run permanently loses the newest ~day per symbol. This also breaks `--advisor` and prompt_ab. |
| 2 | **G6-2** | decision_report.py:354-355 (+:246-253); scripts/sizing_cofire_report.py:57-61 | A | medium | Legacy naive `ts` (the writer used local `datetime.now()`, box = America/Chicago) is read as UTC. Every legacy row is replayed 5 h early. llm_eval, journal_stats and chart_core read it correctly. |
| 3 | C-1 (partC) | scripts/llm_qualify.py:606-649, :80, :306 | A | medium | The p95 is computed over completed calls only, and `P95_MAX_S == BUDGET_S`, so the latency gate can't fail. `schema_valid_pct` also excludes timeouts. With 5/10 calls timed out the verdict is still `qualified`. |
| 4 | C-1 (partB) | scripts/horizon_transfer_report.py:42, wave6_stage0.py:78, funding_drift_audit.py:213 | C | medium | Full-store parquet loads where only 2-9 columns are used. Crypto measured at 773 MB → 205 MB with sha256-identical output. Stock full load is projected at about 3.8 GB (not run). |
| 5 | C-2 (partC) | scripts/train_lexicon.py:91-104 | A | low-med | Stock days are grouped by UTC date: 3,057 phantom Saturdays, and 23% of days open on the previous evening's bar. The output has no consumer. |
| 6 | B-2 (partB) | scripts/rank_gradient_report.py:178-180 | A | low | backtest.py:912 can legitimately write `[]`. `--preds` then crashes with `KeyError: 'ts'` and exits 1, which the contract reserves for "gate said no". |
| 7 | G6A-1 (partA) | execution_report.py:93-97, :106-135, :245-249 | A | low | Buys without `slippage_bps` silently drop the shortfall section, yet the "compare realized shortfall" footer still prints. This is today's real data: 264 buys, 0 with slippage. |
| 8 | B-1 (partB) | scripts/reliability_report.py:85-89 | A | low | Labels are computed on per-arm rows while the gate uses jointly-finite rows, so "(better)" can print next to a keep-legacy verdict. Unreachable until a producer exists. |
| 9 | C-3 (partC) | scripts/sizing_cofire_report.py:103, :141, :295-300 | A | low | Header says "0 buy rows … no rows" when 169 buys are in the window but none carries `sizing`. The fix is additive. |
| 10 | C-4 (partC) | scripts/sizing_cofire_report.py:242 | A | low | v2 flip count sorts ts as strings. Across the DST fall-back hour the true count is 2 and it reports 1. |
| 11 | C-5 (partC) | scripts/sizing_cofire_report.py:362-365 | D | low | `--json PATH` leaks `<path>.<pid>.tmp` when `os.replace` fails. execution_report's writer already has the try/finally. |

## Adjudication of the brief's named open items
- **llm_eval SIP end bug at :218**: **promoted as G6-1.** It is worse than described. The unclamped `end` is only half of it; the wall-clock window cannot contain a bar-stepped horizon for stocks at all.
- **Naive vs aware ts, the 5-hour disagreement**: **promoted as G6-2.** The writer's source (`git show 8c9d860:trade_journal.py`: `entry["ts"] = datetime.datetime.now().isoformat()`) plus the NYSE-RTH test (11,905/11,906 stock rows fall in RTH read as local, 22.9% read as UTC) proves the **local** readers right: llm_eval, journal_stats and chart_core. decision_report and sizing_cofire are wrong. partC showed that on today's data sizing_cofire's output is unaffected (no naive row carries `sizing`). decision_report's replay is affected on every legacy row. One fix covers both.
- **sizing_cofire "0 buy rows" header**: promoted as C-3 (low). The header text is wrong output; the JSON key is as documented. The fix is additive and keeps the `"no rows"` substring pinned by test_c26_S3.py:802.
- **execution_report silently skipping the shortfall section**: promoted as G6A-1 (low, stdout only).
- **rank_gradient ignoring quality.representative**: **not promoted** (judgment call). `representative` describes the whole report, not the rank buckets. MIN_BUCKET_N suppression is the per-bucket guard. Recommendation: warn always, refuse under `--strict`. partB found a different, real defect in the same script (B-2).
- **reliability_report has no producer**: **not promoted**. It is a gap, not a bug. partB's grep shows `p_purged`/`calib_holdout` appear only in the script, calibration.py and tests. B-1 is a latent defect in its consumer logic.

## Escalations (model-facing, owner decision, not proposed)
- **`BARS_PER_YEAR['stock'] = 1638`** assumes 6.5 RTH bars/day. The stock store is extended-hours hourly: about 16 bars/session, median 3,558/yr and up to 4,026 (partA `measure_stock_bars_per_year.py`). Stock per-bar Sharpe annualised with 1638 is understated by up to √(4026/1638) ≈ 1.57×. The same constant sets volatility's per-bar target and hypersearch `compute_sharpe`. Parity is pinned by tests/test_portfolio_backtest_v3.py and tests/test_review_b17.py. It is model-facing, so it goes to the owner.
- **indicator_leadlag pooled effective-n** treats tickers as independent. partB measured 30-39% false positives at nominal 5% for a cross-sectionally common feature. This bears directly on D_measurement's "Daily_Sentiment is leading". It is a methodology change, so not proposed.

---

## G6-1 · llm_eval.py:217-219 (realize_scored_rows) · class A · stock rows fetched over a wall-clock window that cannot contain the bar-stepped horizon, and with an unclamped SIP end

**Defect.** For every (symbol, asset) group the bar request ends at `max(t0) + (max_h + 6)` WALL-CLOCK hours
(`llm_eval.py:218`), but since c26 D09.b the horizon is stepped in BARS (`_realized_forward_return`,
`i1 = i0 + horizon_bars`). Stock hourly bars are 16 per weekday (04:00-20:00 ET) and 0 on weekends, so the
fetched frame never holds `i0 + fb` bars for the newest rows of each symbol; and for any symbol scored in the
last `fb+6` h the `end` is in the future, which this account's SIP feed rejects for the WHOLE request
(`market_data.py:540-548` documents it; FIX_C), so every row of that symbol — back to the window start — is dropped.
The file never calls `market_data._clamp_sip_end`.

**Proof.**
- `hunt/G6/r_sip_future_end.py` (2 read-only Alpaca calls) → `r_sip_future_end.out`:
  `[BARS] SPY: subscription does not permit querying recent SIP data` / `llm_eval shape end=now+20h bars=0`;
  `clamped end bars=32`.
- `hunt/G6/r_llm_eval_stock_window.py` (single row, fake ext-hours bars) → `.out`: for fb ∈ {12,18,24,32,48} × t0 ∈
  {04:30, 10:30, 15:30 ET}, 14/15 rows return `realized=None` even though the exit bar exists (e.g.
  `fb=24 t0=10:30ET realized=None fetched_to=Wed 16:30ET exit_bar_needed=Wed 08-26 19:00ET`).
- `hunt/G6/r_llm_eval_stock_panel.py` (60 RTH-hourly rows over 10 trading days, fb=24, fake SIP that rejects
  end > now-15m exactly like the real feed) → `.out`:
  ```
  A: run 3d after last row: rows=60 realized_by_llm_eval=49 truly_realizable_by_now=54
  B: run 2h after last row: rows=60 realized_by_llm_eval=0 truly_realizable_by_now=48
  A fixed: realized=54 truly_realizable=54
  B fixed: realized=48 truly_realizable=48
  ```
  Case B is the normal daily run while the bots trade: the stock book's keep/kill scorecard (and
  `--asset stock`, and `--advisor`, and scripts/prompt_ab.py which shares `realize_scored_rows`) gets **n=0**
  realized rows, silently (the only trace is a `[BARS]` print and `symbols_all_unrealized`).

**Fix** (stock branch only; crypto request byte-identical):
```python
        end = datetime.fromtimestamp(max(t0s), tz=timezone.utc) + timedelta(hours=max_h + 6)
        if asset != 'crypto':
            # horizon is in BARS (<=16 ext-hours bars/weekday, none on weekends/holidays):
            # fetch a calendar allowance, then respect the SIP 15-min delay.
            from market_data import _clamp_sip_end
            end = _clamp_sip_end(datetime.fromtimestamp(max(t0s), tz=timezone.utc)
                                 + timedelta(days=max_h // 7 + 5), 'stock')
```
Over-fetching is harmless: realization indexes bars, and `_realized_forward_return` already returns None
when `i1` is beyond the frame. Add a test: stock row with fb=24, fake `_bars_lookup` capturing `end` →
`end <= now-16min` and the frame spans ≥ fb bars.

**Blast radius.** Callers: `llm_eval.run_eval`, `llm_eval.advisor_report`, `scripts/prompt_ab.py`. Tests touching
`realize_scored_rows` (`tests/test_llm_eval_v3.py::test_realize_window_is_utc_aware` pins only `start`;
`tests/test_llm_advice.py::TestRealizeScoredRows` stubs `_bars_lookup`; `tests/test_c26_S1.py` stubs it) — none
pins `end`. Measurement-only; nothing model-facing.

**Why indisputable.** A request the vendor rejects outright, and a frame that is provably shorter than the horizon
the same function then steps, cannot be the intended behaviour; the fix reproduces the ground truth exactly.

## G6-2 · decision_report.py:354-355 (+ :246-253) and scripts/sizing_cofire_report.py:57-61 · class A · legacy naive journal ts read as UTC, but the writer stamped LOCAL wall-clock

**Which reader is right.** Before 20a41db (2026-08-20) `trade_journal.log_decision` stamped
`entry["ts"] = datetime.datetime.now().isoformat()` (`git show 8c9d860:trade_journal.py`, also 5b49718/6bb38e7) —
naive LOCAL time; the box is `America/Chicago` (`/etc/timezone`). The current writer's own comment
(`trade_journal.py:100-104`) names the disagreement. So llm_eval (`fromisoformat().timestamp()`, :983/:1211),
journal_stats (:191) and chart_core (:952) are RIGHT; decision_report (`tz_localize('UTC')`) and sizing_cofire
(`replace(tzinfo=utc)`) are WRONG by the local UTC offset (−5 h CDT / −6 h CST).

**Proof.** `hunt/G6/r_legacy_ts_tz.py` (read-only over journals/) → `r_legacy_ts_tz.out`:
```
naive ts rows=12113  offset-aware rows=0
stock rows=11906: inside NYSE RTH if read as LOCAL = 11905 (100.0%); if read as UTC = 2723 (22.9%)
file-name date == naive wall-clock date: 11906/11906
row ts 2026-05-07T08:31:58: decision_report reads 2026-05-07 08:31:58+00:00 = 04:31 ET; writer meant 2026-05-07 13:31:58+00:00 = 09:31 ET; error = -1 days +19:00:00
sizing_cofire._parse_ts(2026-05-07T08:31:58) = 2026-05-07 08:31:58+00:00  (true 2026-05-07 13:31:58+00:00)
```
Every on-box journal row (all 12,113) is naive. decision_report therefore replays every legacy row 5 bars
early (crypto) / in the pre-market (stock) — the counterfactual P&L for every legacy gate/buy is priced at the
wrong entry bar.

**Fix.** One helper, used in both places in decision_report and in sizing_cofire:
```python
# decision_report._replay_grouped / _dedup_first_per_day
if ts.tz is None:
    ts = pd.Timestamp(ts.to_pydatetime().astimezone())   # naive = writer's LOCAL wall clock (pre-2026-08-20 rows)
# sizing_cofire_report._parse_ts
if dt.tzinfo is None:
    dt = dt.astimezone()
```
(`datetime.astimezone()` on a naive value applies the system zone incl. DST — exactly the inverse of the old
`datetime.now()` writer, and identical to what llm_eval/journal_stats/chart_core already do.) Update the two
"naive treated as UTC — the SAME convention" comments (decision_report.py:233, :250).

**Blast radius.** Only naive-ts rows change. Tests: `tests/test_decision_report*.py` build ts from tz-aware UTC
indexes (`tests/test_decision_report.py:33`) — unaffected; `tests/test_decision_report_v3.py::TestDedupTz::test_mixed_naive_aware_no_raise`
asserts only no-raise and len==2 — unaffected. CI runs in UTC, where the new rule equals the old one. Offset-aware
rows (every row since 2026-08-20) are byte-identical.

**Why indisputable.** 11,905/11,906 stock rows fall in NYSE RTH under the local reading vs 22.9% under the UTC
reading; the writer's source code settles it.

### Orchestrator judgment calls (decision_report / beta_ledger / llm_eval), not proposed
- llm_eval `_bars_lookup` has no closed-bar filter, unlike decision_report's `closed_only=True` (c26 D38). A crypto row whose `i1` is the still-forming bar is realized from a partial close. This affects at most the rows whose horizon ends in the last hour. It is a reproducibility issue.
- llm_eval: each `--asset` run overwrites the one `llm_eval_report.json` (D_measurement). This is a design choice.
- llm_eval DK lag is `forward_bars-1` in *cluster steps*, keyed `hac_lag_hours`. The two are equal for hourly crypto. For stock RTH clusters the lag over-covers (conservative), so it is not an under-correction.
- decision_report `replay_entry` enters at the close of the first bar whose OPEN is ≥ ts, i.e. about 1-2 h after the decision. That is a convention. `MAX_HOLD_BARS=24` is fixed while the deployed fb ranges over {12..48}.
- decision_report conviction buckets bootstrap by value, but buys are not deduped. Overlapping 24-bar replays of repeat buys are dependent (the docstring's independence rationale covers only the gate sections).
- beta_ledger `drop_glitch_days` cannot drop a bad print on the LAST observation. This mirrors the day-0 ambiguity FIX_R2 documented.
- beta_ledger lag columns are `shift(lag)` after `dropna()`, which is positional. Since `align_benchmark_returns` ffills, rows drop only at the edges. That is harmless today, but a mid-series NaN would pair non-adjacent days.
- `.jsonl`-only readers (decision_report, llm_eval, sizing_cofire, prompt_ab) go blind to rotated `.gz` days. This is already documented at trade_journal.py:54-61, and rotation is OFF by default.

### Orchestrator "no issues found"
- **decision_report**: bootstrap CI, verdict gates, dedup sort key (apart from G6-2's naive convention), unpriced/quality arithmetic (horizon_pending = unpriced − fetch − oow − null reconciles), atomic pid-unique writer, admitted_k tolerance.
- **beta_ledger**: `ols_hac` implements the documented NW-1994 plug-in `floor(4(n/100)^(2/9))` Bartlett sandwich. The summed-beta SE comes from the full covariance. Annualisation of 252 matches the observed grid (`obs_per_year` 252.2) and there is a guard. Up/down and conditional floors are sound. The writer is atomic.
- **llm_eval**: Newey-West and Driscoll-Kraay sandwiches match their docstrings (Bartlett weights, G/(G-1)). The IM block test, partial Spearman, calibration bins and Brier are correct. `sizing_formula 0.5+s` matches base_loop.py:3131 and stock_loop.py:1194. The writer is atomic (mkstemp + replace).

---

# partA (verbatim)
# G6 partA — measurement shelf hunt (read-only)

Files: journal_stats.py, execution_report.py, gap_audit.py, portfolio_backtest.py, options_overlay.py,
naive_baseline.py, horizon_transfer.py, meta_curve.py. No repo file was edited and pytest was not run.
All repros live in `scratchpad/hunt/G6/`. Their outputs are saved next to them as `*.out`.

**Result: 1 finding meets the bar.** It is low severity: CLI stdout only, and the GUI summary already
handles the case. Everything else checked out; the large cross-cutting item is in the appendix (J1)
because it is model-facing and pinned by tests.

---

## G6A-1 · execution_report.py:93-97 + :106-135 + :245-249 · class A (dead notice branch / silent omission)

**Defect.** The "No fills with slippage data yet" notice only fires when `rows` is empty. Since c26 T7,
`_load` also collects every `buy`, `llm_analysis` and `llm_backoff` row, with or without
`slippage_bps`. So a window holding any such row but no fills gets two things wrong:
- the implementation-shortfall section is **skipped with no notice at all**;
- the trailing footer "Compare against the backtest's assumptions … if realized shortfall is persistently
  higher …" still prints, about a shortfall that was never shown.

This is exactly the real-journal shape on this box. The census over `journals/*.jsonl` found:
`('buy', no slippage)=264, ('skip')=11818, ('short')=31`, and zero rows carrying `slippage_bps`.

**Proof.** Repro: `hunt/G6/repro_exec_shortfall_silent.py`, output in `repro_exec_shortfall_silent.out`.
It monkeypatches `JOURNAL_DIR`/`BASE_DIR` to a scratch directory and writes nothing to the repo.
```
=== case A: empty journal (reference behaviour) ===
No fills with slippage data yet — the loops journal decision_price/fill_price on every confirmed fill.
=== case B: 3 legacy buys, no slippage_bps (real-journal shape) ===
Crypto maker NOTIONAL share: n/a — all $300 of crypto entries carry no entry_tactic/maker_notional (pre-maker-ladder journals)
Compare against the backtest's assumptions (backtest.py SPREAD_PCT haircuts: ...) If realized shortfall is persistently higher, ...
notice printed? False
shortfall header printed? False
```
Correct output for case B: the "no fills with slippage data" notice (the same text case A prints), and
no "compare against the backtest" footer.

**Fix** (stdout only; the JSON is unchanged):
```python
    if fills:
        ...                                   # unchanged shortfall block
    else:
        print("No fills with slippage data in window — the loops journal "
              "decision_price/fill_price on every confirmed fill "
              f"({len(rows)} buy/LLM row(s) without slippage_bps read).")
    ...
    if fills:                                 # guard the footer (:245)
        print("Compare against the backtest's assumptions ...")
```
The early-return branch at :93-97 can stay as it is (an empty window still writes the stub JSON).

**Blast radius.**
- **Callers:** the CLI, and the GUI `_run_report` via `["execution_report.py", "--days", "14"]`
  (gui.py:7083). The GUI renders the JSON through `chart_core.format_execution_summary`, which already
  prints "no fills with slippage data in window" (chart_core.py:1329-1331). So only the raw stdout pane
  and the CLI user are misled, and the fix aligns them with the GUI.
- **Tests:**
  - `tests/test_c26_T7.py::test_llm_only_window_no_crash` asserts only on JSON keys, so it stays green.
  - `test_legacy_keys_unchanged_for_pure_fills` asserts that 'IMPLEMENTATION SHORTFALL' is in stdout when
    fills exist, which is unaffected.
  - `tests/test_review_b17.py` and `test_measurement_fixes_2026_09.py` do not assert on the absence of
    the notice.
  - No test pins the silence.

**Why indisputable.** The notice exists to say "no fills". It cannot fire in the only situation where
it matters (buy rows exist but carry no slippage), and the report then prints advice about a number it
did not print. The fix is measurement-only stdout and cannot change a decision.

---

## Appendix — judgment calls (not proposed)

**J1 (escalate to owner; cross-cutting, model-facing).** `BARS_PER_YEAR['stock'] = 1638` (= 252×6.5 RTH)
is used in:
- portfolio_backtest.py:39-40 (`DEFAULT_PERIODS_PER_YEAR`, commented "stock RTH hourly bars/yr");
- backtest.py:89, volatility.py:156, and scripts/hypersearch_v2.py:522 (`compute_sharpe` → objective);
- all byte-pinned equal by tests/test_portfolio_backtest_v3.py:60-74 and tests/test_review_b17.py:87-97.

The stock store is **extended-hours hourly**, not RTH (repro `measure_stock_bars_per_year.py`, `.out`):
```
median bars/session per ticker: mean 14.8  max 16
bars/yr per ticker: median 3558  max 4026  (BARS_PER_YEAR stock = 1638)
ET bar-open hour histogram: {4: 17090, 5: 15912, ... 19: 18036, 20: 10}
Sharpe scale error if an every-bar stock panel is annualised with 1638: sqrt(4026/1638) = 1.568
```
FIX_D independently records the same thing for the live frame: 320 bars ≈ 20 trading days at ~16
bars/day.

Consequences:
- Any per-bar stock Sharpe annualised with 1638 is understated about 1.57×.
- `volatility.compute_vol_adjusted_size`'s per-bar target (`annual/√1638`) is overstated by the same
  factor if its inputs are per extended-hours bar.
- hypersearch's `slots_per_year` is similarly understated.

Why this is not proposed: the constant is shared with training-path and sizing code (model-facing:
objective, vol-target sizing). It is pinned by two tests and it lives in do-not-touch modules. The fix
is an owner decision plus a gotcha-#2 retrain, and it may be intentional if some consumers are
RTH-filtered. In portfolio_backtest itself, `run_policy` / `compare_deflated` have no production
caller: only tests use them, and scripts/rank_gradient_report.py uses only `panel_from_frame`. So there
is no live misreport today.

**J2 · gap_audit.py:142-143.** `auto_adjust=False` means the Open/Close pair is split-adjusted but
**not dividend-adjusted**. The ex-date overnight gap therefore counts the dividend drop as forfeited
drift, even though a holder at the prior close is paid that dividend. Forfeited drift is understated by
roughly the dividend yield × notional (about $50–100/yr per $5k name at a 1–2% yield). Fixing it
changes the output, needs network to prove, and reviewers could argue it.

**J3 · gap_audit.py:17-18.** The module docstring says "gap stats are computed on .shift(1)-lagged
history", but no shift exists anywhere in the module. This is docstring drift and not in bar scope. The
statistics are a descriptive full-sample audit, so no PIT harm follows.

**J4 · options_overlay.py:320 vs :187, :196-197, :342.** `run_verdict` computes one full-sample
`realized_vol_annual(closes)` per name. It includes closes *after* each simulated entry. That
contradicts the section header's "entry-time IV held — no realized-vol look-ahead" and the
`rv_proxy` text "trailing". The verdict is a static offline NO-GO instrument, and a rolling RV would
change its output, so this is not proposed.

**J5 · meta_curve.py:94-109.** `build_subsample_plan` partitions only `n_pool // block_len` full
segments. The final `n_pool % block_len` rows (up to 49, and the *most recent* ones if rows are
time-ordered) can never be drawn into any sub-`n_pool` subsample. Changing this changes the drawn plans,
which tests/test_c26_V3.py pins by seed determinism, so it is a judgment call.

**J6 · horizon_transfer.py:80-83.** Weekly bootstrap blocks are `epoch_ns // WEEK_NS`. That makes them
Thursday-00:00-UTC-anchored 7-day blocks, not calendar (Mon–Sun) weeks as the docstring's "Calendar-week"
suggests. The SE is valid either way, and changing the anchor changes the output.

**J7 · execution_report.py:36 (and fees.py:122, llm_eval.py:91, decision_report.py:117,
trade_journal.py:228).** `--days N` reads N+1 calendar files and applies no `ts` filter. That is up to
N+1 days of rows. The convention is repo-wide and consistent, so it is not an off-by-one to fix in one
reader.

**J8 · journal_stats.py:35-37.** The docstring says every row carries an offset-aware ISO `ts`. In
reality all 12,113 on-disk rows are naive local time (legacy). The code is correct for both:
`fromisoformat(...).timestamp()` treats naive values as local, which is the legacy writer's clock on
this America/Chicago box, and aware values are exact. The date fast-path's "file D ⇒ local date D"
holds for both. This is doc drift only.

## Appendix — no issues found (per file)

Sanity repro: `check_kernels_partA.py` (`.out`). Empty-input probes: `check_empty_partA.py` (`.out`).

- **journal_stats.py:**
  - `ts` parsing is correct for both naive-local (legacy) and offset-aware rows.
  - The date fast-path is correct: `date.fromtimestamp` in the reader tz matches the writer's
    `now.astimezone().date()` filename.
  - LIFO pairing matches the documented writer semantics.
  - Empty dir, missing dir and `compute_stats([])` are all safe.
  - A read-only run on the real journals gives `files_read 51, rows_seen 264, trades 0`, and the EOD
    digest renders.
  - It writes no files, so atomicity is not applicable.
- **execution_report.py** (beyond FIX_D and G6A-1):
  - Slippage sign conventions agree with every writer: base_loop.py:1510/1881 for sells,
    :3294 for buys, and stock_loop.py:1344.
  - `_write_json` is atomic after FIX_D.
  - The notional/count blocks are consistent after FIX_D.
- **gap_audit.py:**
  - overnight/intraday alignment is verified by hand-computed values.
  - `gap_through` is the mean excess over all nights × 252 on daily bars, which is consistent.
  - `--json` is written non-atomically, but its only reader (gui.py:9151-9160, 9231-9240) opens it after
    the subprocess exits, so there is no race.
  - A 1-row or empty frame is safe.
- **portfolio_backtest.py:**
  - Sharpe = mean/std×√ppy on per-period nets, verified numerically.
  - DSR is fed per-period nets with an overlap-derived `n_eff`, which is scale-invariant.
  - The empty panel, `compare_deflated([])` and `panel_from_frame(empty)` are all safe.
  - There is no non-atomic I/O. The annualisation concern is J1.
- **options_overlay.py:**
  - The bear-put strike ordering and the friction legs are verified by reading the code; b11 tests pin them.
  - The time units are consistent: T in years on a 252-day basis.
  - The verdict JSON has no in-repo reader, so a non-atomic write is not a torn-read risk.
  - `run_verdict({})` is safe.
- **naive_baseline.py:**
  - `bar_returns` equals pandas `pct_change×100`.
  - `ewma_momentum` equals `ewm(halflife, adjust=False)`.
  - `trailing_vol` equals `rolling(w).std()`.
  - Causality holds: `signal[:i]` is unaffected by later closes.
  - There is no annualisation in this module. Its consumer `scripts/naive_vs_blend.py` feeds DSR with
    stage0 dump rows, which are non-overlapping by `stage0_preds` construction, so `iid_nonoverlapping`
    is honest.
- **horizon_transfer.py:**
  - `forward_returns` equals the harvest `Target_Return_h` convention (`(Close.shift(-h)-Close)/Close*100`)
    for h = 1, 4 and 24, so there is no off-by-one.
  - The minimum stride gap equals Delta (24), so windows do not overlap.
  - On IID data, rho(4,16) = 0.502 against the null of 0.500.
  - Empty inputs are safe.
- **meta_curve.py:**
  - `rank_auc` equals sklearn `roc_auc_score` under heavy ties.
  - Seeds are unique per (n, seed).
  - The `m > n_segments` fallback is exercised and returns exactly n rows.
  - The report is JSON-safe.
  - Empty inputs are safe.
  - There are no horizon shifts in this module.
- **Dead code:** no function in the eight files has zero callers; each is used by tests and/or
  scripts/gui/run_pipeline. **Bare `except:`, mutable defaults, argparse misuse:** none found.


---

# partB (verbatim)
# G6 / partB — measurement shelf (IC / rank / Stage-0 / audits) — read-only hunt, 2026-09-26

Repros live in `hunt/G6/partB_repro/` (`*.py` plus the matching `*.out`). All ran in the jetson env with
`CUDA_VISIBLE_DEVICES=''`. No repo file was edited and no pytest was run. The one process that went over
the 600 MB cap was the crypto full-frame load in C1 (773 MB peak), and that overage is the finding. The
stock full-frame load was NOT run because it would exceed the cap by several GB.

## Findings (ranked)

### B-1 · scripts/reliability_report.py:85-89 · class A — wrong label printed next to the gate's numbers
**Defect.** The script prints the gate's Brier/ECE (from `calibration.compare_calibrations`, which scores
only the JOINTLY-finite rows). It then computes its own better/worse/tie labels and the `tied` override
with `brier()` / `expected_calibration_error()` on the FULL arrays, and those functions drop non-finite
rows per arm. When `p_purged` has NaN rows, the two arms are scored on different row sets. The label
can then contradict both the numbers printed beside it and the verdict. The code comment at :82-84 says
the labels "must match the <= criterion compare_calibrations actually gated on". compare_calibrations'
own docstring names NaN rows (unscored OOF folds) as an expected input.

**Proof.** `partB_repro/rel_nan_rowset.py` → `rel_nan_rowset.out`. The input has 400 rows, 100 of them
NaN in p_purged.
```
GATE (jointly-finite rows): n=300 n_dropped=100 brier legacy=0.09 purged=0.0951 ...
script-side (per-arm rows): brier legacy=0.3175 purged=0.0951 ...
  Brier  legacy 0.09  ->  purged 0.0951  (better)      <- 0.0951 > 0.09: should be "worse"
  VERDICT: no calibration improvement on this holdout — keep legacy / collect more
```
The same row-set mismatch feeds `tied` at :89. So if the per-arm metrics happened to tie while the
joint ones did not, the verdict would be wrongly overwritten to "tied".

**Fix.** Score the labels on compare_calibrations' own row set:
```python
    y_arr = np.asarray(y, float)
    pl_arr, pp_arr = np.asarray(p_legacy, float), np.asarray(p_purged, float)
    jm = np.isfinite(pl_arr) & np.isfinite(pp_arr) & np.isfinite(y_arr)  # == compare_calibrations' mask
    bl, bp = brier(pl_arr[jm], y_arr[jm]), brier(pp_arr[jm], y_arr[jm])
    el = expected_calibration_error(pl_arr[jm], y_arr[jm], args.bins)
    ep = expected_calibration_error(pp_arr[jm], y_arr[jm], args.bins)
```
Add a test: the NaN payload above must print `(worse)` on the Brier line.

**Blast radius.** CLI only, with no importers. The pinning tests (`tests/test_review_b22.py`
TestReliabilityTieSemantics, `tests/test_grp_reports.py:188-240`) use only all-finite arrays, where
jm is all True, so their output is unchanged.

**Why indisputable.** A printed "better" beside a strictly larger Brier is wrong output, and the code's
own comment states the intended invariant.

### B-2 · scripts/rank_gradient_report.py:178-180 · class A — crash on an empty Stage-0 dump, exit 1 reads as "gate ran and said no"
**Defect.** `backtest.py:912` always writes the dump, including `[]` when no row qualifies
(`stage0_preds.write_rows([])`). Given that dump, `--preds` builds `pd.DataFrame([])` and dies on
`df['ts']` with a KeyError traceback, exit status 1. The docstring's exit contract (:121-125) reserves
1 for "ran-but-no-go", so scripted use reads the crash as a real negative verdict. The sibling consumers
handle the same file cleanly: `ic_by_name.py` exits 0 with an empty table, and `naive_vs_blend` /
`entry_timing_probe` refuse explicitly.

**Proof.** `partB_repro/rgr_empty_dump.py` → `rgr_empty_dump.out`:
```
scripts/rank_gradient_report.py --preds: exit=1  last stderr line: KeyError: 'ts'
scripts/ic_by_name.py --in: exit=0
scripts/rank_gradient_report.py --buckets: exit=2  ... must be a JSON object ...
```

**Fix.** Right after the DataFrame is built (:178-179), before `set_index`:
```python
        if df.empty:
            print(f"ERROR: {args.preds} holds no prediction rows (empty stage0 dump) — "
                  f"nothing to gate", file=sys.stderr)
            return 2
```
Extend the docstring's exit-2 sentence to cover "an empty --preds dump".

**Blast radius.** CLI only. The tests (`tests/test_grp_reports.py:108-126`, `tests/test_review_b22.py:189-235`,
`tests/test_c26_P2.py:330-366`) all use non-empty frames, so their behaviour is unchanged.

**Why indisputable.** An uncaught traceback on the producer's own documented empty output yields an exit
code the contract assigns to a different outcome.

### C-1 · scripts/horizon_transfer_report.py:42, scripts/wave6_stage0.py:78, scripts/funding_drift_audit.py:213 · class C — full-store loads where 2–9 of 72/110 columns are used (multi-GB on the stock store)
**Defect.** Each of these scripts calls `load_training_data(book)` / `(book, columns=None)` and loads every
column, but uses only a few:
- horizon_transfer_report uses `Ticker` + `Target_Return_*`.
- wave6 uses `Ticker` + `TB_Bars_*`, sliced at :104.
- funding_drift uses `Ticker` + `*Funding*` + `Target_Return_*`.

The stores were re-harvested today:
- crypto: 263,889 rows × 72 columns
- stock: 1,536,356 rows × 110 columns, including five string `TB_Reason_*` columns

**Proof.**
- **Crypto, measured, output bit-identical.** `partB_repro/htr_load_mem.py` → `htr_load_mem.out`
  (sha256 over every per-name array plus the time vectors):
  ```
  full   crypto ... sha256 7aa5778d48643dfe peak_rss_MB 772.7
  pruned crypto ... sha256 7aa5778d48643dfe peak_rss_MB 205.1
  pruned stock  ... names 90 sha256 9b682bd5c321f7a1 peak_rss_MB 511.2
  ```
- **Stock, projected (not measured).** The full stock load was not run because of the hunt's RSS cap.
  Parquet metadata gives 728 MB of uncompressed column data for stock vs 108 MB for crypto. Crypto's
  measured full-load overhead is about 5.3× its uncompressed size (568 MB). That projects a full stock
  load to roughly 3.8 GB above baseline. `free -m` showed about 3.7 GB available during the hunt, so
  `horizon_transfer_report.py --prefix stock` and `wave6_stage0.py` (both books by default) risk heavy
  swap or OOM on the 8 GB Jetson. With the pruned load, stock measures 511 MB.
- **wave6 and funding_drift.** They issue the same `load_training_data(<book>)` full call. They slice the
  columns immediately (wave6:104) or read only those names (funding_drift:125-155), so pruning is
  output-identical by construction.

**Fix.** Peek the parquet schema before loading. This reads no data. Keep the exact current behaviour
whenever the loader would not serve the parquet:
```python
def _pruned_columns(book, keep):          # keep: predicate on a column name
    """Columns to request, or None (= today's full load) when the loader would not
    serve the parquet (absent, or CSV fresher -> data_utils._csv_is_fresher)."""
    try:                                   # pyarrow is absent on the dev Mac -> full load
        import data_utils, pyarrow.parquet as pq
        stem = data_utils._stem(book)
        pqp, csvp = data_utils._BASE_DIR / f'{stem}.parquet', data_utils._BASE_DIR / f'{stem}.csv'
        if not pqp.exists() or data_utils._csv_is_fresher(pqp, csvp):
            return None
        cols = [c for c in pq.read_schema(pqp).names if c == 'Ticker' or keep(c)]
        return cols if len(cols) > 1 else None
    except Exception:
        return None
```
Then:
- horizon_transfer_report: `keep=lambda c: c.startswith('Target_Return_')`
- wave6: `keep=lambda c: c.startswith('TB_Bars_')`
- funding_drift: `keep=lambda c: 'Funding' in c or c.startswith('Target_Return_')`

wave6's comment at :72-76 ("a pruned parquet read raises on missing columns") is addressed, because the
names come from the file's own schema.

**Blast radius.** Three measurement CLIs, no importers.
- `tests/test_review_b20.py:227-260` TestWave6Horizons monkeypatches `data_utils.load_training_data`
  with `lambda prefix, columns=None: _fake_panel()`. With the fix, the schema peek still reads the REAL
  parquet on the Jetson, and the patched loader then ignores `columns` and returns the fake panel.
  wave6 discovers horizons from the LOADED frame's columns, so the tests still see [24, 64]. On the
  Mac/CI, with no parquet or no pyarrow, the peek returns None, which is today's call.
- `tests/test_r2c_measurement_kernels.py` exercises only `fda.*` kernels (`psi_from_train_deciles`,
  `spearman_with_ci`, `sign_flip_disjoint`, `audit_frame`) and `horizon_transfer`, never `load_per_name`
  or `main`.

**Why indisputable.** The output is bit-identical (hash-checked). The only behavioural difference is
memory, and it is measured (568 MB saved on crypto). On stock it is the difference between fitting in
RAM and not.

**Caveat for the adjudicator.** The fix reads data_utils' private `_stem`, `_BASE_DIR` and
`_csv_is_fresher` (read-only; data_utils is not edited). Using them is what keeps the CSV-fresher path
byte-identical. Without the guard, a stale parquet schema could drop columns that only a fresher CSV
has.

## Judgment calls (not proposed)

1. **indicator_leadlag.py:103-138: pooled t ignores cross-sectional dependence.** The largest
   statistical issue on this shelf, but a methodology choice, not a docstring contradiction.
   - The n/h time-overlap correction is implemented as documented. The null size is conservative:
     `leadlag_null_size.out` shows 0.00–0.037 at a nominal 0.05.
   - Summing n_eff across tickers treats names as independent. For a market-wide feature (identical
     across names, e.g. daily sentiment / F&G) with crypto-like cross-correlation, the null rejection
     rate is 0.30 at ρ=0.5 and 0.39 at ρ=0.8, vs 0.053 at ρ=0 (`leadlag_xsec_size.out`).
   - This bears directly on D_measurement §2, where Daily_Sentiment was the top "leading" feature.
   - Candidate fixes (all change output): cluster by timestamp, divide Σn_eff by an effective number of
     independent names, or test market-wide features on one series only.
2. **scripts/rank_gradient_report.py:205: `stale` checked, `quality.representative` ignored** (the open
   item). `representative` (decision_report.py:979) is a report-wide flag: priced > 0, no fetch
   failures, and unpriced_rate ≤ 0.30 across ALL gates, including horizon-pending rows. It is not a
   rank-bucket property, and the script's documented contract (:121-125) covers only stale/non-object.
   `--strict` already enforces n ≥ 30 per bucket plus the ci90. Whether to refuse (exit 2), warn, or
   refuse only under `--strict` is a design decision → judgment call. My recommendation is a WARN line
   always, plus a refusal under `--strict`.
3. **scripts/reliability_report.py: no producer for `{p_legacy, p_purged, y}`** (the open item).
   Confirmed by grep across `*.py`, `*.md`, `*.json` and `*.sh`: `p_purged` and `calib_holdout` appear
   only in the script itself, `calibration.compare_calibrations` (the consumer kernel) and tests. It is a
   gap, not a bug; building the producer means writing a meta holdout dump that runs both calibrators.
   Two stale comments (`strategy_config.py:265`, `scripts/hypersearch_v2.py:1494`) and `06` plan §4 step 10
   also cite this script for q10 coverage, which it does not do (already noted in docs/MODULES.md:143,363).
4. **scripts/entry_timing_probe.py:51 `BAR_ANCHOR_MIN['stock']=30` and docstring :23-25 ("stock hourly bars
   on the half-hour").** The store is 100% :00-aligned (1,536,356/1,536,356 rows), and live stock bars
   come from Alpaca `get_bars('1Hour')` (hour-aligned); only the yfinance fallback is :30. Real stock
   fills (58 journal buys) have a median of 526 s past :30 vs 2156 s past :00 (`etp_stock_anchor.out`).
   Which anchor is "right" depends on whether the live window used a forming or a closed bar
   (`closed_bars_v2_enabled()` at base_loop.py:2382/3249), so it is arguable. The docstring's factual
   claim about bar alignment is false for this data either way.
5. **horizon_transfer.weekly_block_ids (used by entry_timing_probe / horizon_transfer_report).** `epoch //
   WEEK_NS` blocks start Thursday 00:00 UTC, so each stock trading week is split Mon–Wed | Thu–Fri. The
   block bootstrap is still valid, but "news-sharing rows move together" holds less well. Changing the
   anchor to Monday changes SE values.
6. **scripts/funding_drift_audit.py.**
   - The "sign flip with disjoint CIs" test compares the full sample with a subsample of itself, so the
     two CIs are not independent.
   - The Fisher CI uses the Pearson SE 1/√(n−3) for a Spearman ρ; Fieller's ≈√(1.06/(n−3)) is slightly
     wider.
   - Anchors pooled across the 6 crypto names are cross-sectionally correlated, which is the same issue
     as item 1.
   - All three are methodology choices.
7. **scripts/window_ab.py:75-76/383.** `--prefix` expects the internal form `stock_`, while every sibling
   CLI (`hypersearch_v2`, `naive_vs_blend`, `horizon_transfer_report`, `entry_timing_probe`) takes `stock`.
   `--prefix stock` dies at `joblib.load('stockconfig_v2.pkl')`. The help text documents `stock_`, so this
   is UX, not a bug.
8. **scripts/crypto_spread_census.py:178.**
   - The JSON write is non-atomic. Its reader `liquidity._load_census` runs in the harvest process only
     under the dark `TRADER_CRYPTO_SPREAD_STAMP` flag; a torn read memoizes `{}` and falls back to the
     defaults.
   - The census file is written, and consumed by liquidity, even when `sanity.ok` is false or n < 30.
   - Polling "latest quote" every 5 s counts an unchanged quote repeatedly (time-weighted, not
     quote-weighted).
   - The flag is dark and model-facing, so this goes to the owner.
9. **scripts/ic_by_name.py:16-17.** A stale doc says "pass --pred-key signal to share one dump", but the
   stage0 dump now carries both `pred` and `signal`. Doc drift only.
10. **scripts/meta_learning_curve.py:125.** It also does a full-store load, but it genuinely needs
    feature_cols + OHLC(+ATR) for `_predict_ticker` / `_gen_meta_rows`, so pruning needs a
    column-dependency audit of backtest/meta_label (training-path, out of bounds this round). Not
    measured.

## No issues found (per file)

- **indicator_leadlag.py**
  - `bh_fdr` matches statsmodels `fdr_bh` on 3000 random cases with ties and NaNs (NaN excluded from m;
    step-up is monotone).
  - `spearman` matches `scipy.stats.spearmanr` on 2000 cases with ties and NaNs.
  - Forward (t, t+h] and past (t−h, t] targets are exact, with no off-by-one.
  - Alignment between `groupby.indices` and the per-group target arrays holds on an interleaved panel.
  - The n/h overlap adjustment works as documented, and the size is conservative (`leadlag_stats_check.out`,
    `leadlag_null_size.out`).
  - See judgment call 1.
- **ic_diagnostic.py**: rank_ic, sub-periods, and promote_set's t = IC·√(n_finite−1) all behave as documented.
- **rank_gradient.py**: bucket assignment, the n/fwd_bars overlap-widened ci90 and the verdict guards all
  behave as documented.
- **stage0_preds.py**
  - `select_row_indices` enforces i+h ≤ n−1 and a per-name ≥h spacing.
  - The anchor grid is correct.
  - `mtm_equity` handles its step and open marks as documented.
  - The writes are atomic.
  - tz handling is consistent: the UTC store, with `.value` and `index_ns` both in UTC ns.
- **scripts/ic_by_name.py**: sound apart from judgment call 9.
- **scripts/rank_gradient_report.py**: sound apart from B-2 and judgment call 2.
- **scripts/reliability_report.py**: sound apart from B-1 and judgment call 3.
- **scripts/wave6_stage0.py**: `_ticker_boundaries` (`value_counts(sort=False)` order) matches the stable
  Ticker sort on the real stores (verified). Apart from that, see C-1.
- **scripts/cscv_audit.py**: argparse guards match `pbo_from_oos_blocks` / `pbo_cscv` preconditions, and
  the printed PBO semantics (λ ≤ 0) match `validation.pbo_cscv`.
- **scripts/funding_drift_audit.py**: PSI (decile edges, ±inf ends, actual ref histogram), the strided
  anchors and the tz-localized split are all correct. Apart from that, see C-1 and judgment call 6.
- **scripts/entry_timing_probe.py**: the anchor arithmetic (train Close[i]→Close[i+h], live Close[i−1]→…)
  is exact and the weekly-block bootstrap is of the delta. Apart from that, see judgment calls 4–5.
- **scripts/horizon_transfer_report.py** (and the horizon_transfer kernel): the IID null √(δ/Δ), strided
  non-overlap and block-bootstrap SE are correct. Apart from that, see C-1.
- **scripts/naive_vs_blend.py**: exact-timestamp joins, the identical-row comparison set and strict `>`
  admission all behave as documented.
- **scripts/meta_learning_curve.py**: the seed index is in range(n_seeds), the eval slice is fixed and
  chronological, and the write is atomic. See judgment call 10.
- **scripts/window_ab.py**: the stage0 dump's non-overlap and UTC-naive ts strings are correct, and
  identical n_trials are used across arms. See judgment call 7.
- **scripts/crypto_spread_census.py**: `quote_spread_pct` / `summarize` / `sanity_check` are correct. See
  judgment call 8.

Across the 16 files: no bare `except:`, no mutable default arguments, and no annualisation constants.
Every broad `except Exception` is either fail-soft by design in a measurement path or re-raises. None
of these files uses a Newey-West/HAC estimator.


---

# partC (verbatim)
# G6 partC — llm_qualify, prompt_ab, train_lexicon, connection_test, sizing_cofire_report

**Scope and method.** This was a read-only hunt on the Jetson, 2026-09-26. No repo file was edited and no pytest was run. No LLM or network call was made: C-1 uses a local 127.0.0.1 stub server. Every repro lives in this directory as `partC_*.py`, with its captured output in `partC_*.out`. None of these five files appears in KILL_LIST.md or 08_removed_code.md.

Findings are ranked by severity.

---

## C-1 · scripts/llm_qualify.py:606-649 + :80/:306 · class A (wrong verdict) · MEDIUM
**Defect.** The latency criterion is censored at the very threshold it tests, so a model that times out on most calls is still "qualified".

**How it happens:**
- Every qualification call is sent with `timeout=BUDGET_S` (45 s, :611).
- Latency is appended only for completed calls (:630).
- The pass test is `p95 <= P95_MAX_S`, and `P95_MAX_S == BUDGET_S` (:80, :306).
- So any call slower than the budget times out and is simply dropped. The p95 of the survivors is ≤ the budget by construction.
- `schema_valid_pct` has the same problem: its denominator is `n_completed` (:648), so timeouts and 5xx never count against it.
- Result: a model that answers 1 of 20 calls in time and fails the other 19 by timeout or 5xx gets **"qualified"**.
- The "marginal: p95 ≤ 60 s" band is also practically unreachable, because the 45 s transport timeout caps every sample.
- This contradicts the module docstring (:8-13, :72-73): "a candidate that can't answer inside it is not a stand-in".

**Proof.** `partC_qualify_censored_latency.py` runs the real `llm_client._call_openai` over urllib against a local stub that sleeps 3 s on every second call. BUDGET_S and P95_MAX_S are scaled together from 45 s to 1 s, keeping prod's relation `P95_MAX_S = BUDGET_S`.
```
{'n_attempts': 10, 'n_completed': 5, 'schema_valid_pct': 100.0, 'p50_latency_s': 0.0035, 'p95_latency_s': 0.0042, 'verdict': 'qualified'}
errors: ['attempt 2 (F02): timed out', 'attempt 4 (F04): timed out', 'attempt 6 (F06): timed out']
=> 5/10 attempts exceeded the 1.0s budget, yet verdict='qualified' (p95 over survivors only)
p95 over ALL attempts (timeouts as >budget) = 1.000 -> verdict 'marginal'
```
**Correct output.** Half of the calls blew the budget, so the verdict must not be "qualified".

**Exact fix (minimal).** Score schema validity over the attempts that were not rate-limited instead of over completed calls. A call that produced no response did not produce a schema-valid response.
```python
# :648
denom = q["n_attempts"] - q["rate_limit_events"]
q["schema_valid_pct"] = (100.0 * n_valid / denom) if denom > 0 else None
```
- The repro then gives 5/10 = 50 %, which is "failed".
- `parse_fallback_pct` stays over `n_completed`.
- Optionally, also record `max(r["latency_s"], BUDGET_S)` for non-429, status-less failures (timeouts) in `latencies`, so p95 reflects them.

**Blast radius.**
- Only `main()` → `run_qualification` calls it. No live caller; the output goes only to `journals/llm_qualify/llm_qualify_report.json` and its `config_patch`.
- Tests in `tests/test_c26_V1.py`:
  - `test_qualify_end_to_end_stubbed`: 5 attempts, 1×429, 4 valid. It asserts schema_valid_pct 100.0 and "qualified", which stay 4/4 = 100 % under the fix.
  - `test_verdict_boundaries`: feeds synthetic dicts, so it is unaffected.
  - `test_qualify_sustained_429_aborts` and `test_transport_exception_fail_open`: already "failed", so unaffected.
- Nothing pins the censoring.

**Why indisputable.** The pass/fail statistic is computed only over the calls that passed, with a cut-off equal to the pass threshold. The criterion can never fail on a slow model, which is the model it exists to catch.

---

## C-2 · scripts/train_lexicon.py:91-104 · class A (wrong daily bars) · MEDIUM (dark research artifact, not model-facing)
**Defect.** Hourly stock bars are bucketed into "trading days" by **UTC** date (`idx.tz_convert('UTC').date`). In EST months the 19:00 ET post-market bar opens at 00:00 UTC of the next date.

**Effects on the real `stock_training_data.parquet`:**
- Every Friday-evening bar creates a phantom **Saturday** trading day: 3,057 of 115,426 price rows.
- 26,212 day-rows (23 %) take their `open` (the entry price) from the prior evening's 19:00 ET bar.
- `learned_lexicon.attach_forward_returns` works in "trading-day index space". With phantom Saturdays, h-day forward returns (h = 1, 3, 5) span the wrong number of real sessions.

**Proof.** `partC_lexicon_utc_days.py` runs the exact aggregation lines from `main()` on the real parquet, read-only.
```
UTC              : price rows=115426  Saturday rows=3057  Sunday rows=0
America/New_York : price rows=112437  Saturday rows=0  Sunday rows=0
example phantom Saturday rows:  AAPL 2023-11-11 close 184.02 open 183.79 ; 2023-11-18 ; 2023-12-02 ; 2023-12-09
UTC-dated days whose first bar is 00:00 UTC (=19:00 ET prev evening, EST): 26212 of 115426
```
**Exact fix.**
```python
dates = idx.tz_convert('America/New_York').date
```
- Apply it at :93, and use the same conversion in the naive-index branch after `tz_localize('UTC')`.
- No look-ahead is added. Articles are UTC-publication-dated (sentiment_history.py:614-620), and entry is still the first ET session strictly after that date.
- **Caveat:** if someone passes `--data training_data.parquet` (crypto, 24/7), ET day boundaries are a convention change. Gate the ET conversion to the stock default, or add a `--session-tz` flag defaulting to America/New_York.

**Blast radius.**
- Nothing imports the script. Its outputs, `learned_lexicon.json` and `lexicon_eval_report.json`, are consumed by nothing (grep-verified; learned_lexicon.py:10-11).
- `tests/test_c26_V2.py:641` smoke-runs the CLI but does not pin the day bucketing.
- The only thing that changes is the numbers in a research report.

**Why indisputable.** Saturday is never a US equity session, and a one-bar post-market "day" is not a trading day.

---

## C-3 · scripts/sizing_cofire_report.py:103, :141, :295-300 · class A (wrong output, human header) · LOW
This is the open item FIX_D left.

**Defect.** `n_buy_rows` counts only buys that carry a `sizing` dict. The header therefore prints "0 buy rows … no rows — nothing to report" on the real journals, where the same window holds 169 buy rows (none of them pre-2026-07 rows with sizing).

The GUI's sibling instrument, `chart_core.sizing_stack_summary` (chart_core.py:998-1035), uses the **same key name** `n_buy_rows` for ALL buys, plus a separate `n_with_sizing`. The two instruments over the same journal disagree on what `n_buy_rows` means.

**Proof.** `partC_sizing_header.py`, read-only over journals/:
```
sizing_cofire header : sizing co-fire report — last 200d, book=all: 0 buy rows (51 files, 0 malformed lines skipped)
sizing_cofire line 2 : no rows — nothing to report
buy rows in the same 200d window: 169  (with sizing dict: 0)
chart_core.sizing_stack_summary same window: n_buy_rows=177 n_with_sizing=0
```
The 177 vs 169 gap is chart_core's whole-day-file window granularity, not a parsing difference. See the naive-ts appendix item.

**My judgment on the parent's question: yes, this meets class A "wrong output", in the header text only.** The printed sentence asserts a false fact: there are 169 buy rows in the window, not 0. "No rows" also hides the real diagnosis: the journals pre-date the sizing producer.

The JSON value is internally consistent with the docstring's scope ("BUY rows (the nested 'sizing' decomposition dict)"). **Do not change the JSON key's meaning.**

**Exact fix (additive; no existing key changes).**
- In `_load_rows`, count `elif action == 'buy': n_buy_nosizing += 1` and return it.
- In `build_report`, add `rep['n_buy_rows_without_sizing']`.
- In `print_report`, print:
  `'%d buy rows with a sizing dict (+%d older buy rows without one, skipped)'`.
- When `n_buy_rows == 0 and n_buy_rows_without_sizing > 0`, print
  `'no rows with sizing{} — journals pre-date the sizing producer (base_loop, 2026-07)'`.
- Keep the literal "no rows" substring, which `tests/test_c26_S3.py:802` pins.
- `_load_rows` has no caller outside this file (grep).

**Blast radius.**
- The tests pin `n_buy_rows` values: test_c26_S3.py:755 (6) and :795 (2), and test_measurement_fixes_2026_09.py:358 and :368 (1). All their fixtures carry sizing, so they are unchanged.
- :802 pins "no rows", which the fix preserves.

**Why indisputable (with the caveat stated).** The header claims 0 buy rows when 169 exist. The fix only adds a count and wording. If the adjudicator reads the header as "scoped by docstring", move this to judgment calls. The additive count is still zero-risk.

---

## C-4 · scripts/sizing_cofire_report.py:242 · class A (wrong count, rare) · LOW
**Defect.** The v2 hysteresis flip count orders rows by the ts **string**. The writer stamps local offset-aware ISO (`datetime.now().astimezone().isoformat()`, trade_journal.py:106,115). Across the DST fall-back hour, string order ≠ time order, so `flip_counts` is wrong for rows in that hour.

**Proof.** `partC_sizing_dst_sort.py` uses three rows on 2026-11-01 in America/Chicago:
```
2026-11-01T01:50:00-05:00 = 06:50Z calm | 01:10:00-06:00 = 07:10Z stress | 01:30:00-06:00 = 07:30Z calm
report vix_tier_total = 1
true time order tiers = ['calm', 'stress', 'calm'] -> flips = 2
```
**Exact fix.**
```python
_MIN = _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)
ordered = sorted(v2_rows, key=lambda r: _parse_ts(r.get('ts')) or _MIN)
```
Rows that reach it via `_load_rows` already parsed, so the fallback only matters for direct `build_report` callers.

**Blast radius.**
- No other caller.
- test_c26_S3.py:783-796 fixtures use one uniform offset (`_ts(minute)`), so the parsed order equals the string order and their results do not change.

**Sibling string-sorts** have the same cause but negligible effect: they only affect which cycles count as the "newest N", within one hour per year.
- llm_qualify.py `_load_replay_cycles` (`out.sort(key=str(ts))`)
- prompt_ab.py `load_replay_cycles` (`cycles.sort(key=r["ts"])`)

List them with the fix only if the fixer touches those files anyway.

**Why indisputable.** Sorting mixed-offset ISO strings lexicographically is not chronological order.

---

## C-5 · scripts/sizing_cofire_report.py:362-365 · class A/D (tmp-file leak on error path) · LOW
**Defect.** `--json PATH` writes `PATH.<pid>.tmp` and then calls `os.replace`, with no cleanup. When the replace fails, the script exits 1 (correctly) but leaves the tmp file behind. FIX_D's sibling `execution_report._write_json` (:62-78) wraps the same pattern in `try/finally: tmp.unlink()`: two copies of the same FIX_D atomic-write pattern, one without cleanup.

**Proof.** `partC_sizing_tmp_leak.py`, with PATH set to an existing directory:
```
exit = 1 | stderr: sizing_cofire_report failed: [Errno 21] Is a directory: '.../target_is_dir.60486.tmp' -> '.../target_is_dir'
leftover tmp files: ['target_is_dir.60486.tmp']
```
**Exact fix.** Wrap the write:
```python
try:
    with open(tmp, 'w') as f:
        json.dump(...)
    os.replace(tmp, args.json)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)   # inside try/except OSError
```
**Blast radius.**
- `test_measurement_fixes_2026_09.py:358` asserts no leftover tmp on success, which is unchanged.
- :383 (unwritable dir → exit 1) is unchanged, because the tmp open fails first there.

**Why indisputable.** It is a resource leak on an error path, and the sibling copy already has the fix.

---

## Judgment calls (not proposed)
1. **prompt_ab.py:380-381 `veto_rate_a/_b` divide by `n_total`**, which counts every row, including rows where that variant returned no score.
   - When A and B fail on different rows, the two rates have different effective denominators.
   - One could argue this is the effective fail-open veto rate, since a missing score means no veto live.
   - Arguable, so not proposed. tests/test_llm_advice.py:378-379 would still pass under either definition.
2. **prompt_ab.decide_adopt compares b2_A and b2_B estimated on non-identical subsamples** (`samples_a` and `samples_b` differ when one variant failed), while the docstring says "paired A/B comparison". The `paired` block itself is correctly restricted to pairs. Whether the b2 comparison should be restricted to the paired set is a design choice.
3. **Naive-ts basis (the parent's question).** sizing_cofire `_parse_ts` (:57-64) reads naive legacy ts as UTC, but the pre-6bb38e7 writer stamped naive LOCAL time (trade_journal.py:101-104 admits this). It is **unreachable** on real data:
   - The `sizing` dict and aware ts both landed in the same commit, 6bb38e7 (2026-07-13).
   - `sizing_zero` skips date from 2026-06-12, but the bots have not run since 2026-05-07.
   - So 0 naive rows carry sizing or `sizing_zero` (census: 264 buys, 31 shorts, 11,818 skips, all naive, none with sizing).
   - Window membership of naive buys is identical under both readings (`partC_sizing_naive_ts.out`: 169 vs 169 at 200 d).
   - The UTC reading also matches decision_report's convention. Not a finding. It would only matter if C-3's additive count were later used for window-edge accounting.
4. **The same naive-ts split inside prompt_ab.py.** `load_replay_cycles` reads naive as UTC (:95-96), while `cmd_run` computes `t0 = fromisoformat(ts).timestamp()`, which reads it as local (:249-250). The replay writer has been aware-only since inception (llm_analyst.py:1325), and no `journals/llm_replay/` exists on the Jetson, so this is unreachable.
5. **llm_qualify.schema_check** accepts `s` as a numeric string ("0.7") or a bool (`true` → 1.0) as "strict-schema valid" (`float(entry.get("s"))`). Whether "strict" should require a JSON number is arguable, since the live parser also coerces.
6. **llm_qualify docstring** says "main() always exits 0". Argparse errors exit 2 (SystemExit is not caught). The behaviour is better than the docstring; docs only.
7. **sizing_cofire / prompt_ab / llm_qualify read `*.jsonl` only** (no `.jsonl.gz`). This is already documented as a known blind spot at trade_journal.py:54-61, and rotation defaults OFF.
8. **connection_test.py:58-60** discards the returned status (HALT/ERROR/…) and always exits 0, so a failed credential check exits 0. It also prints "OK TO TRADE… intraday margin" without checking the account type (cash vs margin).
   - docs/MODULES.md:900 still lists an "unused `import os`" that no longer exists (docs drift).
9. **prompt_ab `--max-cycles 0`** means "all" (`if args.max_cycles:`). `cmd_score` writes `llm_prompt_ab_report.json` non-atomically, but no other process reads it (grep), so class D does not apply.
10. **sizing_cofire flip counting** interleaves all symbols and books in one sequence. Whether VIX-tier and BTC-RV flips should be per-book is a design question.

## No issues found (per file)
- **scripts/llm_qualify.py**: checked
  - `agreement_stats` (neutral band, veto side, n=0 → None)
  - `_percentile` / `latency_stats` (linear interpolation, empty list)
  - `pricing_zero`, `discover_candidates` (the `--models` filter, unknown names, the gemini pseudo-candidate)
  - `_transport_call` (never raises; retry-after parse)
  - the 429 abort logic
  - `assemble_report` (merge semantics; `partition('/')` is safe for OpenRouter ids containing '/')
  - `_write_report` (tmp + replace; single writer and reader)
  - no bare `except:`, no mutable defaults
  - never writes llm_config.json
- **scripts/prompt_ab.py**: checked
  - `load_existing_pairs` (float t0 round-trips exactly through JSON)
  - the resumability skip and the empty-both-variants skip
  - `pair_variant_samples` (deltas and flips only on pairs; spearman NaN guard)
  - the `decide_adopt` rule order, which matches its docstring
  - `cmd_score` sharing one realization pass (a `zip` with `realize_scored_rows`, same length by contract)
  - `persist=False` on both `analyze_trades` calls
  - no bare `except:`, no mutable defaults
- **scripts/train_lexicon.py**: apart from C-2, checked
  - close/open alignment (both groupbys iterate the same sorted keys)
  - the SystemExit re-raise
  - atomic outputs via `ll.write_json_atomic`
  - output files have no consumers
  - the argument parsing of horizons and the lambda grid (ValueError → exit 1)
- **scripts/connection_test.py**: nothing beyond judgment item 8. `get_api` is shared, and the file has no mutable defaults or bare excepts.
- **scripts/sizing_cofire_report.py**: apart from C-3, C-4 and C-5, checked
  - empty input (no division by zero: `p_both_fire` is only computed when a pair exists; `fires_given_floor` is guarded)
  - `_stats` / `_median` on empty input
  - the `marginal_effect` exclusion of kelly and vol, and `cf > 0`
  - the exit-code handling (FIX_D)
  - the book classification (no slashless crypto symbols in the real journals)
