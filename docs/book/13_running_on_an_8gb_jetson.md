# Chapter 13. Running on an 8 GB Jetson

## 13.1 The idea

This whole system, data harvesting, neural-network training, two trading loops, a GUI and a
supervisor, runs on one NVIDIA Jetson Orin Nano: a credit-card-sized computer with six ARM cores,
a small GPU and **8 GB of memory shared between the CPU and the GPU**. There is no cloud server
behind it. When the Jetson runs out of memory, overheats, fills its disk or loses power, the
trading stops, or worse, half-stops.

This chapter is about operating a trading system as a piece of machinery: how memory is budgeted,
which processes may use the GPU, how the service is supervised and restarted, what happens when
memory runs out, how the logs are kept from filling the disk, and how the owner is told when
something breaks. Operations is not glamorous, but an unattended trading system is only as good as
its worst night.

Some vocabulary.

- **RSS** (resident set size) is the physical memory a process is using right now.
- **Swap** is disk space used as overflow memory. It prevents crashes but is far slower.
- The **OOM killer** ("out of memory") is the Linux kernel's last resort: when memory is
  exhausted it kills a process to save the machine.
- **systemd** is the Linux service manager. A **unit** is its description of a service: what to
  run, how to restart it, what limits apply.
- A **watchdog** is a timer the service must keep resetting; if it stops, systemd assumes the
  service is hung and restarts it.

## 13.2 Why it matters financially

Operational failures cost money in ways that never appear in a backtest.

- A bot that crashes while holding positions leaves them managed only by whatever stop orders are
  resting at the broker. Software-side trailing stops, signal exits and the end-of-day flatten all
  stop.
- A bot that crash-loops can do damage by restarting: each restart cancels and re-places orders,
  and each gap between cancel and re-place is a moment without protection.
- An alert that never arrives turns a five-minute problem into a five-week one. This system's bots
  were down from 2026-05-07 to 2026-09-26 and nothing told anyone.
- A training run that starves the bots of memory can make them miss exits during exactly the hours
  the retrain runs.

Professional trading firms treat these as risk controls, not IT chores. The Futures Industry
Association's guidance on automated trading, cited in the INTEL research log, describes a kill
switch that blocks new orders *and* cancels working ones, and warns the operator of the
consequences. That standard is useful to measure this system against.

## 13.3 How this system does it

### The memory budget

The machine reports 7,619 MB of total memory and a 12 GB swapfile on the NVMe drive
([`scripts/setup_jetson_system.sh`](../../scripts/setup_jetson_system.sh) replaces the default
compressed-RAM swap with this swapfile, because compressed swap costs CPU and memory). The big
consumers, as measured by the ENGINE department on this box:

- **PyTorch itself.** Importing `torch` adds about 331 MB of resident memory to a process. The
  trading model is tiny by comparison: 293,761 parameters, about 1.2 MB. Quantizing the weights
  therefore saves almost nothing; the runtime is the cost.
- **A CUDA context.** Each process that touches the GPU reserves roughly 300 MB for it
  (the comment above `BOT_ENV` in [`run_pipeline.py`](../../run_pipeline.py)).
- **Training data.** The trainer keeps one scaled feature matrix per fold in memory at a time
  (`ScaledCache`); the current stock run logs about 50 MB per matrix.

The design responses follow directly:

1. **Bots never touch the GPU.** `run_pipeline.BOT_ENV` sets `CUDA_VISIBLE_DEVICES=''`,
   `OMP_NUM_THREADS=2` and `TORCH_NUM_THREADS=2` for every bot process, and `run_bots.py` itself
   defaults `CUDA_VISIBLE_DEVICES` to empty. Inference of a small LSTM on two CPU threads takes tens of milliseconds, well
   inside a 30 second cycle, and saves the CUDA context and GPU memory for training.
2. **Both books in one process.** Combined mode (`--combined-bots`) runs both loops as threads in
   one Python process, so PyTorch and the libraries are loaded once instead of twice.
3. **Training holds a lock.** [`gpu_lock.py`](../../gpu_lock.py) provides an exclusive file lock
   (`fcntl.flock` on `.gpu.lock`). The trainer must acquire it before using the GPU, so two
   training runs can never share the 8 GB. The operating system releases the lock automatically if
   the holder dies, so there are no stale locks to clean up.
4. **The weekly retrain stops the bots.** `_stop_bots` sends each bot SIGTERM, waits 10 seconds,
   then kills it, and sleeps 3 seconds so the kernel can reclaim the memory before training starts.
   Its docstring states the reason: each bot's PyTorch import reserves 300 to 500 MB, and stopping
   them frees about a gigabyte for CUDA. The cost is that the books are unmanaged by software
   during the retrain (Chapter 10).
5. **A measured alternative is parked.** The ENGINE department built a pure-numpy replica of the
   LSTM's forward pass. On 3,207 real feature windows it changed zero trading decisions and would
   cut about 349 MB from each bot process by never importing PyTorch. It is not wired in, because
   its pre-registered accuracy bar (1e-6) turned out to be stricter than PyTorch's own
   float32 noise; re-registering that bar is an owner decision (ENGINE owner item 14).

### Heat

Two thermal rules exist. Before any training phase, `run_pipeline._bounded_thermal_wait` waits
for the GPU to cool below 70 °C, for at most 30 minutes. In the bots, `_sleep` doubles the cycle
interval when the GPU is above `THERMAL_THROTTLE_TEMP = 75` °C. The temperature comes from the
Linux thermal sysfs files via [`hw_monitor.py`](../../hw_monitor.py) `get_gpu_temp`.

### The service

Since 09:22 on 2026-09-27 the system runs as a **systemd user unit**, which needs no administrator
password. Its key lines (from `~/.config/systemd/user/trader.service`):

    Type=notify
    ExecStart=/home/kyle/miniforge3/envs/jetson/bin/python -u run_pipeline.py --combined-bots --bot-only
    EnvironmentFile=-/home/kyle/trader/.env
    Restart=on-failure
    RestartSec=30
    WatchdogSec=900
    OOMPolicy=continue
    OOMScoreAdjust=200
    MemoryMax=6G

A drop-in override in `trader.service.d/override.conf` replaces `ExecStart` to add `--stock-only`,
recording the founder's decision without editing the repository's generic unit. **Linger** is
enabled (`loginctl enable-linger kyle`), so the user's service manager starts at boot and keeps
running after logout.

What each line buys:

- `Type=notify` and `WatchdogSec=900`: `run_pipeline` sends `READY=1` when it is up, and a
  heartbeat thread sends `WATCHDOG=1`, but **only if the main loop has stamped progress within the
  last 600 seconds** (`WATCHDOG_STALL_SEC`). Before this rule, the heartbeat kept the watchdog
  happy even when the main thread was stuck forever on a blocked read, so the watchdog only caught
  outright death. Now a real hang lets the lease expire and systemd restarts the service. A thermal
  wait counts as progress, so cooling down is not mistaken for hanging.
- `Restart=on-failure` with `RestartSec=30`: a crashed pipeline comes back in 30 seconds.
- `MemoryMax=6G`: the whole unit, bots and any training it launches, may not exceed 6 GB, which
  leaves room for the operating system and the GUI.
- `OOMScoreAdjust=200`: if the kernel must kill something, it should prefer the pipeline to a
  random system process.
- `OOMPolicy=continue`: when the OOM killer kills one *child* (say, a training phase), systemd does
  not tear down the entire unit; the pipeline's own retry and bot-restart logic handles it.

### The internal supervisor and the crash loop

Inside the unit, `run_pipeline` checks its bots every 60 seconds and restarts any that exited
(`_check_restart_bots`). The historical problem: it restarted forever at a fixed 60 seconds, with
no backoff and no notion that the same crash was repeating. In April 2026 one deterministic bug (a
string formatted as a number in `fundamentals.py`, since fixed) crashed the stock bot **1,778
times**. A backoff design is now in the tree behind `TRADER_BOT_RESTART_BACKOFF` (default off):
repeated identical crashes wait 60, 120, 240 seconds and so on up to 960, and after five identical
crashes within an hour the supervisor stops restarting that bot and sends one critical alert.
Replayed against the real April history it cut 1,616 restarts to 12. Since June 2026, errors inside
a cycle are caught in-process instead of crashing the bot, which changes the failure mode: the
April bug would now repeat every cycle with a deduplicated warning instead of restarting.

### Logs and journals

Three kinds of files grow.

- **`logs/trader.log`**, the structured log from [`log_config.py`](../../log_config.py): rotating,
  10 MB per file, five backups (`_MAX_BYTES`, `_BACKUP_COUNT`), with a lock so several processes
  can share it safely. Tests used to write into this production log (33,452 lines in one night were
  test noise), which made forensics impossible; the campaign added `TRADER_LOG_DIR` so tests log
  elsewhere.
- **Bot stdout logs** (`crypto_bot_output.log`, `stock_bot_output.log`; combined mode writes to the
  crypto one). These are rotated one-deep at 20 MB, but only when a bot is launched
  (`_rotate_log`). A bot that runs for months never rotates. The file `stock_bot_output.log` on
  this box is 115 MB, last written on 2026-05-07, left over from the crash loop.
- **Journals** (`journals/`): 5.5 MB across 53 entries today. Compression of old journals exists
  (`TRADER_JOURNAL_ROTATE_DAYS`), but is off by default because several readers open the plain
  files directly and would go blind to compressed days.

The authoritative list of every generated file, who writes it and who reads it is
[`docs/STATE_FILES.md`](../STATE_FILES.md).

### Alerts

[`notify.py`](../../notify.py) sends alerts to a webhook (`TRADER_WEBHOOK_URL`, Discord- or
Slack-compatible) or Telegram (`TRADER_TELEGRAM_BOT_TOKEN` plus `TRADER_TELEGRAM_CHAT_ID`). It
never blocks a trading path (sends happen on a background thread), deduplicates each alert key to
once per 10 minutes, and retries a failed critical send once. With no channel configured it does
nothing, silently.

## 13.4 A worked example: the memory ledger during a retrain

At 09:57 on 2026-09-27, with the Phase-3 stock retrain running and the stock-only service up, the
machine looked like this (`free -m` and `/proc/<pid>/smaps_rollup`):

| Item | Memory |
|---|---:|
| Total physical memory | 7,619 MB |
| Used | 5,086 MB |
| Available (free plus reclaimable cache) | 2,296 MB |
| Swap in use | 2,976 MB of 12,001 MB |
| `hypersearch_v2.py` (training, on the GPU) | 1,357 MB RSS |
| `run_bots.py --stock-only` | 95 MB RSS |
| `run_pipeline.py` (the supervisor) | 28 MB RSS |

How to read it. First, the trainer is the elephant: over a gigabyte, and on this unified-memory
machine its GPU allocations come out of the same 7.6 GB. Second, the bot looks small at 95 MB, but
that is because the market is closed on a Sunday and much of the idle process has been pushed to
swap; the ENGINE measurement of a live bot with PyTorch loaded is several hundred megabytes. When
the market opens and the bot touches those pages again, they must come back from disk, which is
why swap is a safety net and not a budget. Third, nearly 3 GB of swap in use means the machine has
been under pressure at some point during the night; "available" is the number to watch, and 2.3 GB
is comfortable but not generous. Finally, this particular retrain was launched by hand for the
campaign, outside the service, so it is not counted against the unit's `MemoryMax=6G`. A retrain
launched by `run_pipeline` itself would be.

## 13.5 What the evidence says

- **The most valuable operational fix costs one minute.** The `.env` on this box contains only the
  Alpaca and Finnhub keys; no webhook or Telegram variables. Every breaker trip, crash and halt
  alert has therefore been silent. The ENGINE department estimated 232 alerts would have fired
  during April's crash loop; none was sent (ENGINE owner item 17).
- **The user unit is installed but not yet proven by a reboot.** The pre-registered acceptance test
  (reboot without logging in and see the service come up within 180 seconds; `kill -9` the main
  process and see it return within 60; freeze it with `kill -STOP` and see the watchdog restart it)
  is written in [`research_engine.md`](../../research/campaign_2026-09_jetson/research_engine.md)
  as experiment X8. Until it is run, treat automatic recovery as expected, not demonstrated. Note
  too that this box runs systemd 249, which lacks the newer escalating restart-delay options, so
  backoff has to live in the pipeline.
- **Halt is entries-only.** Creating `trading_halt.flag` blocks new entries but does not cancel
  working orders, and the LLM keeps running. Whether a halt should also cancel working buys is
  staged as `HALT_CANCELS_WORKING_BUYS` (default off) and awaits the owner.
- **Crash-loop backoff is designed and replay-tested but off.** The numbers (5 crashes, 60
  minutes, 960 seconds) are judgment calls, and the choice between giving up permanently and
  probing again after an hour is an owner decision.

## 13.6 Further reading

- Betsy Beyer, Chris Jones, Jennifer Petoff and Niall Richard Murphy (eds.), *Site Reliability
  Engineering: How Google Runs Production Systems*, O'Reilly, 2016. Monitoring, alerting and
  post-mortems for systems that must not stop.
- Michael T. Nygard, *Release It! Design and Deploy Production-Ready Software*, 2nd ed., Pragmatic
  Bookshelf, 2018. Circuit breakers, timeouts, bulkheads and the "steady state" pattern, which is
  exactly the log and journal problem above.
- The systemd manual pages for `systemd.service` and `systemd.resource-control`
  (freedesktop.org/software/systemd/man), for `WatchdogSec`, `OOMPolicy` and `MemoryMax`; and
  `loginctl(1)` for linger.
- In this repository: [`docs/MAP.md`](../MAP.md) section 4i,
  [`docs/STATE_FILES.md`](../STATE_FILES.md) section 10 (logs), and the ENGINE research log
  sections E5 and E6.
