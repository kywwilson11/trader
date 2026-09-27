# phase5 bot-observation harness (scratchpad only — nothing here lives in the repo)

`H=/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/phase5/harness`
`P5=$H/..`  (outputs: `bots.pid`, `bots_stdout.log`, `bots_started.json`, `llm_cost_start.json`,
`monitor.jsonl`, `bots_stop.log`)

| file | what |
|---|---|
| `start_bots.sh [--crypto-only\|--stock-only] [--dry-run]` | `run_bots.py` (combined: both loops as threads in one process) under run_pipeline's BOT_ENV, `setsid nohup`, cwd = repo, log → `$P5/bots_stdout.log`, pid → `$P5/bots.pid` |
| `stop_bots.sh [stop\|halt\|resume\|status]` | `stop` = SIGTERM, 15 s grace, SIGKILL. `halt` / `resume` = create / remove `/home/kyle/trader/trading_halt.flag` |
| `monitor.py --minutes N [--interval 60]` | one JSON line per tick → `$P5/monitor.jsonl`, plus a one-line summary on stdout |
| `summarize.py [--alpaca]` | report at the end of the window |
| `verify/` | self-test evidence: fake bots, dummy-pid monitor runs, summarize runs on the May data |

## Order of use

```bash
S=/tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad
H=$S/phase5/harness; source $S/jenv.sh

# 0. pre-flight (launches nothing): checks the pid file, other bot/pipeline processes, and heavy jobs
bash $H/start_bots.sh --dry-run            # add --crypto-only / --stock-only if the gate passed only one book

# 1. start
bash $H/start_bots.sh [--crypto-only|--stock-only]
bash $H/stop_bots.sh status                # PID alive, VmRSS/VmHWM, any control flags

# 2. monitor (background; must use the jetson env so the Alpaca probe can import trading_utils)
CUDA_VISIBLE_DEVICES='' nohup $JPY $H/monitor.py --minutes 1440 --interval 60 \
    > $S/phase5/monitor_stdout.log 2>&1 &
tail -f $S/phase5/monitor_stdout.log       # one line per tick
tail -f $S/phase5/bots_stdout.log          # raw bot output

# 3. if anything looks wrong while the bots are running
bash $H/stop_bots.sh halt                  # no NEW entries from the next cycle; exits/stops/breaker keep running
bash $H/stop_bots.sh resume                # re-enable entries

# 4. end of window
bash $H/stop_bots.sh                       # SIGTERM → 15 s → SIGKILL; prints EXITED CLEANLY or KILLED
CUDA_VISIBLE_DEVICES='' $JPY $H/summarize.py --alpaca     # --alpaca = read-only broker order counts
```

## halt vs stop: which to use

- **halt** (`trading_halt.flag`). Use it to stop taking on new risk while still observing, and
  while still protecting the positions already open. Both loops check the flag before any entry
  (`base_loop._entries_allowed`, which `stock_loop` also calls). Stop management, signal sells,
  LLM-veto sells and the circuit breaker all keep running. The bot logs `[HALT] trading_halt.flag
  active` every 10 cycles. The halt is **not** journaled as skip rows. Use halt first when anything
  looks odd.
- **stop**. Use it at the end of the window, and in a memory or thermal emergency, a crash loop,
  or any case where the process itself is the problem. `run_bots`' SIGTERM handler sets
  `_shutdown`, and `main()` returns within about 5 s. The loop threads are daemons, so they die
  mid-cycle. Resting broker orders are **not** cancelled.
- **flatten** has no sub-command on purpose, because it *sells the book*. If it is ever needed:
  `touch /home/kyle/trader/flatten_request.flag`. Each loop fans that flag out to
  `flatten_crypto.flag` / `flatten_stock.flag`, liquidates its own book within one cycle, and then
  sets `trading_halt.flag`. Per-book flags older than 1 h are discarded.

## How to read monitor fields

- `rss_mb`/`hwm_mb` come from `/proc/<pid>/status`.
- `cpu_pct` is the percentage of **one** core; `cpu_pct_of_box` divides it by 6.
- `log_*` fields count lines added since the previous tick. The first tick rewinds to the last
  `start_bots.sh` banner, so startup errors are counted. `log_err_lines` also matches the
  `[LightGBM] [Fatal] Could not open …lgb_q10.txt` line. That line is expected **once per book**
  whenever the q10 booster is absent, and it means the book serves LSTM-only.
- `journal_by_action` keys look like `book:action`. The book comes from `book`, `asset_type`, or
  the symbol (a `/` means crypto).
- `pred_new` is read from `pred_history.jsonl` / `stock_pred_history.jsonl` (monitor_drift): one
  line per cycle that produced any non-null prediction.
- `alpaca` is a read-only snapshot taken on tick 0 and then every 10th tick.

Per-book cycle counts come only from the journal `cycle_latency` rows. The `--- CYCLE N` log
lines carry no book, because both loops log as `[base_loop]` in combined mode.
