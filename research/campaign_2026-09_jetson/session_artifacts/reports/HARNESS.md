# HARNESS — the phase-5 paper-bot observation harness (2026-09-26)

The harness was built under `scratchpad/phase5/harness/`. Nothing was written in the repo. **No
bots were started** and **no orders were placed**. All Alpaca calls were read-only (account,
positions, open orders, order history). No repo flag file was created: halt and resume were tested
against a temporary directory.

## Files

| path (under `scratchpad/phase5/harness/`) | purpose |
|---|---|
| `start_bots.sh [--crypto-only\|--stock-only] [--dry-run]` | Runs `run_bots.py` in combined mode under run_pipeline's `ENV`+`BOT_ENV` values (the `LD_LIBRARY_PATH` order matches run_pipeline.py:276-284). Launched with `setsid` + `nohup`, cwd = `/home/kyle/trader`, log → `phase5/bots_stdout.log`, pid → `phase5/bots.pid`. It also writes `phase5/bots_started.json` and snapshots `llm_cost.json` → `phase5/llm_cost_start.json`. **Refuses** in two cases: the pid file points at a live process, or any `run_bots.py`/`crypto_loop.py`/`stock_loop.py`/`run_pipeline.py` process exists. It **warns** if hypersearch, harvest, meta_label or backtest is running. `--dry-run` runs every check and launches nothing. |
| `stop_bots.sh [stop\|halt\|resume\|status]` | `stop`: SIGTERM, wait up to 15 s, then SIGKILL (+5 s). It refuses a pid whose cmdline is not `run_bots.py`, prints `EXITED CLEANLY` (rc 0) or `KILLED (SIGKILL)` (rc 3), and appends to `phase5/bots_stop.log`. `halt`/`resume`: write/remove `/home/kyle/trader/trading_halt.flag`, using the same JSON payload as `notify.set_halt`. `status`: liveness, VmRSS/VmHWM, and any control flags present. |
| `monitor.py --minutes N [--interval 60] [--pid P] [--alpaca-every 10] [--no-alpaca] [--log-from-now]` | Appends one JSON line per tick to `phase5/monitor.jsonl` and prints one summary line per tick. Every probe runs in its own try/except. A 45 s watchdog thread bounds the Alpaca call. |
| `summarize.py [--monitor F] [--log F] [--last-run] [--journal-date D]… [--journal F]… [--since ISO\|none] [--pred-dir D] [--alpaca]` | Produces the end-of-window report. It is stdlib-only; `--alpaca` needs the jetson env. |
| `README.md` | Usage in order, and when to use halt vs stop. |
| `verify/` | Evidence: fake `run_bots.py` stubs, monitor jsonl/stdout, summarize outputs. |

## Verification output (all on this device)

**`bash -n`**: both scripts OK. `py_compile`: both .py files OK.

**start_bots.sh `--dry-run`**:
- It prints `cmd : …/envs/jetson/bin/python -u run_bots.py` (`--stock-only` is appended when
  passed).
- It prints the env `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2
  PYTHONUNBUFFERED=1`, plus `LD_PRELOAD` and `LD_LIBRARY_PATH`.
- It ends with `DRY RUN — all checks passed, nothing launched.`
- Bad arguments exit with rc 2 in both cases tested (both book flags together, and `--combined`).
- A pid file pointing at a live `sleep` gives `REFUSING … rc=1`.
- A stale pid file prints a note and is overwritten.

**Full launch/stop mechanics.** These used the real scripts with test-only
`HARNESS_REPO`/`HARNESS_PHASE5` overrides and a fake stdlib `run_bots.py` (`verify/fake_start.txt`,
`verify/stop_verify.txt`):
- The pid file holds the python pid exactly (pid == sid == pgid, so it is in its own session and
  reparented).
- `/proc/<pid>/environ` shows `CUDA_VISIBLE_DEVICES=` (empty), `OMP_NUM_THREADS=2`,
  `TORCH_NUM_THREADS=2`, `PYTHONUNBUFFERED=1` and `LD_PRELOAD`.
- cwd is correct and `--crypto-only` was passed through.
- A second start was refused.
- stop printed `EXITED CLEANLY after SIGTERM in 0 s` (rc 0).
- A SIGTERM-ignoring fake gave `still alive after 15 s — SIGKILL` / `KILLED` (rc 3).
- No stray processes were left.

**monitor.py** ran for 1 minute at a 20 s interval against the fake bot pid, and for 30 s at a
10 s interval against the dummy pid `$$`. Excerpts:
```
22:15:37 #0 pid=36451 UP rss=10.6MB hwm=10.6MB cpu=None% avail=4555MB swap=75MB gpu=46.906C cpuT=47.156C load=0.55 log+0 err+0 cyc+0 preds=c0/s0 jrn[-] flags=- | acct eq=122415.48 cash=93.63 pos=6 open_ord=0
22:15:57 #1 pid=36451 UP rss=10.6MB hwm=10.6MB cpu=1.0% ... log+20 err+0 cyc+4 ...
22:17:08 #0 pid=36653 UP rss=3.0MB ... log+107 err+2 cyc+20 ...            (banner rewind: startup ERROR+Traceback counted)
22:17:28 #2 pid=36653 UP ... preds=c2/s0 jrn[crypto:cycle_latency=1,crypto:skip=1,stock:skip=1] flags=['halt']
```
Every probe produced a value on this device:
- `/proc` RSS, VmHWM and CPU%;
- MemAvailable (≈4.2–4.8 GB free now) and swap;
- the GPU temperature through `hw_monitor.get_gpu_temp`, plus CPU/Tj temperatures from sysfs
  (`hw_monitor` has no CPU reader);
- load average;
- the log line, error and cycle deltas;
- the journal-row deltas by `book:action` (rows already in the file were correctly ignored);
- the pred_history deltas;
- the flag detection;
- `llm_cost.json`;
- the read-only Alpaca snapshot (`equity 122415.48, cash 93.63, buying_power 374.52, ACTIVE,
  6 positions MV 122,321.85, 0 open orders`; probe 1.5 s).

rc was 0 in both runs, with no `*_err` keys.

**summarize.py:**
- **Empty defaults (no run yet):** every section degrades gracefully ("no monitor records", "log
  not found", …), rc 0.
- **Fake-bot outputs:** it found the 28 CYCLE lines, 28 `Predicted Return:` lines, 28 placed and
  28 filled orders, skip reasons by book (`llm_below_buy_min (0.5<0.6)` normalised to
  `llm_below_buy_min`), the cycle_latency row, 2 preds, and the one ZeroDivisionError traceback
  (first 3 lines + exception line + the preceding ERROR line).
- **May data** (`--journal-date 2026-05-07 --since none --log stock_bot_output.log`; 115 MB,
  4.19 M lines, 13.6 s): 1213 stock `llm_below_buy_min` skips; 17,543 CYCLE lines; 6,341 WAIT
  lines; order errors (4,675 stop-sell insufficient qty, 3,488 bracket insufficient buying power,
  2,894 PDT rejections …). It found **one distinct traceback × 1,778**:
  `stock_loop.py line 574 … ValueError: Unknown format code 'f' for object of type 'str'`. That
  independently reproduces E_ops' 1,778-crash finding.
- **`--alpaca`** (read-only `list_orders(status='all', after=…)`): since 2026-04-20 it returned
  n=4, split as 2 crypto buy market filled and 2 crypto buy limit canceled. That matches the 2
  journal buy rows of 04-26. There are 0 broker orders since 2026-05-01.

## Repo facts the orchestrator must know before launch

1. **`run_bots.py` is combined by default.** Its only arguments are `--crypto-only` and
   `--stock-only` (argparse; `--combined` / `--combined-bots` would error). `--combined-bots` is a
   **run_pipeline** flag. `run_bots` itself `setdefault`s `CUDA_VISIBLE_DEVICES=''` and the
   thread counts to 2 (run_bots.py:38-40), so the harness env is belt-and-braces.
2. **Control flag filenames**, all in the repo root:
   - `trading_halt.flag` (notify.py:131). It blocks entries only. Both loops check it via
     `base_loop._entries_allowed` (base_loop.py:2884; stock_loop.py:986).
   - `flatten_request.flag` (notify.py:133). This is the shared request that Telegram and the GUI
     write.
   - `flatten_crypto.flag` / `flatten_stock.flag` (base_loop.py:79-80). These are per-book, fanned
     out by whichever loop sees the shared flag first. They are consumed before flattening, and a
     flag older than 3600 s is discarded (base_loop.py:2769-2810). A flatten also sets the halt
     flag.
   - The task's `flatten_crypto.flag`/`flatten_stock.flag` exist, but **no writer creates them
     directly**. Operators touch `flatten_request.flag`.
3. **Journal rows are a tagged union on `action`, not `kind`** (trade_journal.py:9-22). Live
   producers write:
   - `skip`, `buy`, `sell`, `entry_window`, `account_risk`, `llm_analysis`;
   - `cycle_latency` (one per completed cycle, **with `book`**, base_loop.py:420-430);
   - `signal_exit_reading`, `vertical_barrier`, `pred_fanout_timeout`, `llm_backoff`,
     `circuit_breaker_trip`, `rr5_demotion`, `ioc_entry_fallback`, `entry_fills`.

   Most rows carry no book field; the harness infers the book from `book`/`asset_type` or `/` in
   the symbol. File = `journals/<local date>.jsonl`, and `ts` is offset-aware.
   `CONVICTION_JOURNAL_ENABLED=True` (strategy_config.py:66), so gate skips are journaled. A
   **manual halt journals nothing**.
4. **Cycle lines cannot be split by book in combined mode.** `--- CYCLE N` is logged by the
   `base_loop` logger for both books (base_loop.py:328), with no thread or book tag in the format
   (log_config.py:18). Per-book cycles come only from the journal `cycle_latency` rows.
   - The stock book logs no CYCLE line while the market is closed. It logs
     `[WAIT] Market closed` on cycle 1 and every 20th cycle.
   - The next US open is **Mon 2026-09-28 09:30 ET**. A weekend window exercises only crypto.
   - Cadence: LOOP_INTERVAL = 30 s ± 5 s jitter plus the cycle work
     (`[SLEEP] Next check in Ns`). It doubles if the GPU temperature exceeds 75 °C.
5. **Predictions:** `monitor_drift.log_predictions` appends one line per cycle to the repo-root
   `pred_history.jsonl` / `stock_pred_history.jsonl`, holding only non-null preds. Neither file
   exists today. `predict_now` also `print`s `Predicted Return:` per symbol to stdout.
6. **Logging:** the StreamHandler goes to **stderr** (the harness captures it with `2>&1`), and
   every line is duplicated into the shared `logs/trader.log` (DEBUG). The harness log is **not**
   `crypto_bot_output.log`, and `pipeline_status.json` is not updated. The GUI will therefore show
   the bots as not running. That is expected.
7. **Standalone ops thread:** `run_bots` starts an ops thread (TRADER_BOTS_OPS default ON) while
   no fresh `pipeline_status.json` heartbeat exists. It does three things:
   - polls Telegram, which is a no-op because no `TRADER_TELEGRAM_*` is set;
   - runs a once-daily PSI drift check (`monitor_drift.run_check`), which reads pred_history;
   - runs journal rotation, a no-op at the default `TRADER_JOURNAL_ROTATE_DAYS=0`.
8. **On startup each loop runs `cancel_all_open_orders` for its own universe**, then
   reconstructs positions from the broker (base_loop.py:277-282). SIGTERM does **not** cancel
   resting orders, and the daemon loop threads die mid-cycle.
9. **Expected memory: about 650 MB** for the combined bots (E_ops measured 649–650 MB for a full
   prediction pass with the C extension, 688 MB without). Add about 60 MB once `lgb_model.txt`
   exists. Budget 0.75–1.0 GB, and do not run a stock hypersearch alongside.
10. **Expected noise:** if the rebuild leaves no q10 booster, each book prints
    `[LightGBM] [Fatal] Could not open {,stock_}lgb_q10.txt` **once** (B_serving §2). monitor
    counts that line in `log_err_lines`; it is benign.
11. **LLM spend:** `llm_cost.json` holds only the cumulative cost for the current
    **America/Los_Angeles** day (llm_client.py:781-805). The harness snapshots it at start and on
    every tick, and summarize sums the per-day maxima minus the baseline. The value was
    $0.000092 on 2026-09-26.

## Launch blockers / risks seen while building (not fixed; owner/orchestrator decisions)

- **The account cannot fund any entry.** Re-checked read-only just now: equity $122,415, **cash
  $93.63, buying power $374.52**. There are 6 crypto positions (BTC, DOGE, ETH, LINK, SOL, XRP;
  MV $122.3k) and **0 open orders**, so there are no resting stops.
  - A paper run as-is will mostly produce entry-order rejections. In May that was `insufficient
    balance`, `bracket: insufficient buying power` and PDT rejections; see
    `verify/summarize_may.txt`.
  - The observation window will exercise predictions, gates, stops and exits, **not entries**.
- **`avg_entry_price = 0` and `cost_basis = 0` on all six positions** (confirmed just now).
  - `order_utils.reconstruct_positions` passes `entry_price = 0` through (order_utils.py:1121).
  - The P&L divisions are all guarded (`if entry_price > 0 else 0.0`: base_loop.py:772, 1375,
    1443, 1494, 1868, 2843), so there is no crash from that. Journaled `pnl_pct` for these six will
    read 0.0.
  - The **stop math changes behaviour.** In `base_loop._desired_stop_for` (base_loop.py:1027-1068),
    `trailing_active = hwm >= 0 * (1 + x)`, which is **True immediately**. The entry-ATR branch is
    skipped, so the crypto trail is the fixed fallback of **5%** (CRYPTO_POLICY
    `trail_fallback_pct`, strategy_config.py:30).
  - No `position_state.json` exists, so HWM = the current price at reconstruct.
  - Net effect: on start, the crypto book arms a trailing stop about 5% under the current price on
    **all six positions** (about $122k). Any 5% pullback from the running high liquidates them.
  - This comes from reading the code; it was not exercised. A macro `stop_mult < 1` would tighten
    the trail further.
  - Decide before launch whether that is acceptable, or whether to launch `--stock-only`.
- The untracked C extension crashes on frames under ~99 bars (B_serving). A symbol with few bars
  would abort the **whole combined process**. monitor shows this as `alive=false`, with the abort
  message in the log tail.
