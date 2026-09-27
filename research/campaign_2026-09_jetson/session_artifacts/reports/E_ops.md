# E — Ops audit: can this Jetson Orin Nano run trader unattended? (2026-09-26)

## Verdict

**Not as the setup script would install it today. It is safe once five fixes to the unit and environment are made.**
Memory is not the problem. The bots, the parent process and the backfill worker together use about 1 GB, which fits
comfortably. The weekly retrain fits only if the desktop is turned off. The problems are in how the service is
configured:

1. The unit's `ExecStart` interpreter is `$(command -v python3)` run under `sudo`. That resolves to **/usr/bin/python3**
   (no pyarrow/dotenv/yfinance/torch). Result: **every weekly retrain crashes the pipeline before the bots are
   stopped**, systemd restarts it with `--bot-only`, and the next retrain is scheduled a week later. The model never
   retrains, and the daily shadow evaluation fails silently.
2. `Environment=CUDA_VISIBLE_DEVICES=` on the unit also hides the GPU from **training**. The weekly search would run on
   CPU at about 820 s per epoch. That means roughly 1 epoch per fold (the trial timeout cuts the rest), about 80 h per
   book, and bots offline most of the week.
3. systemd's `DefaultOOMPolicy=stop` means any OOM-kill of a child process stops the **whole** unit. The pipeline's own
   3× phase retry and its bot crash-restart never get to run.
4. There is **no alert channel at all.** `.env` has only the Alpaca and Finnhub keys, so `notify` does nothing and the
   /halt and /flatten kill switch is unreachable.
5. The desktop is still on (graphical.target, about 1 GB). Unified memory was already oversubscribed during past
   retrains: the log has 77 `NvMap … error 12` lines, 4 trials killed by OOM, and 5 batch-size halvings.

Also out of scope, but it blocks go-live: the paper account holds **6 crypto positions ($121.9k market value, $93 cash,
$374 buying power) with 0 open orders** (no resting stops at the broker), and **avg_entry_price = 0 on all six**.

## Measurements (bot environment: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2`, jetson py3.10)

Each scenario ran in a fresh subprocess. VmHWM and ru_maxrss agreed to within 1 MB in every case. Scripts are in
`scratchpad/ops/mem_probe.py` and `scratchpad/ops/cycle_probe.py`; logs are `scratchpad/ops/mem_*.log` and
`cycle_*.log`.

| Scenario | Peak RSS (VmHWM) |
|---|---|
| bare interpreter | 10 MB |
| (a) `import predict_now` | **519 MB** (anon 284 / file 232) |
| (b) `import base_loop, crypto_loop, stock_loop` | **520 MB** (`import run_bots` alone: 13 MB, because the loops are imported lazily) |
| (c) + `load_models()` crypto and stock (both load; JIT trace OK) | 599 → **600 MB** |
| (c′) + a second crypto model copy (what `shadow.maybe_log_shadow` does in the bot process) | 604 MB (+4 MB) |
| + lightgbm (the bots import it once `lgb_model.txt` exists; none exists today) | about +60 MB (475 → 533 MB in a separate run) |
| one full read-only prediction pass: 6 crypto + 46 stocks, 5 threads, C extension ON | **649–650 MB** (2 runs, no crash) |
| same pass with the C extension OFF (numba path, 6+20 names) | 688 MB |
| (d) RegressionLSTM CPU forward, crypto champion (seq 18, h 288, 2 layers, 2 heads, 23 features, 1.38 M params) | 407 MB process; **b1 10.3 ms, b8 23.2 ms, JIT b1 10.2 ms** |
| (d) stock champion (seq 20, h 160, 2 layers, 8 heads, 0.44 M params) | 405 MB; **b1 7.9 ms, b8 14.7 ms** |
| run_pipeline parent after every import it makes in-process (notify, monitor_drift, shadow, trading_utils, market_data, sentiment_history) + `_needs_force_harvest` | **177 MB** |
| sentiment backfill worker (imports) | 28 MB |
| crypto training frame load (`load_training_data('crypto')`, 234,822 × 51) | 96 → **361 MB** (+265 MB for a 93 MB dense frame, about 2.85×) |
| stock training frame (1,236,699 × 54, 405 MB uncompressed) | not loaded (per-process cap). Estimated **about +1.4 GB peak** at 2.85× |
| CPU training step (synthetic, bs 2048, AdamW, 6 threads, other agents running) | **11.6 s/step (h 288) and 7.4 s/step (h 160)**, about 824 and 521 s per epoch at 145k rows |

LSTM inference cost is negligible. Bot memory is dominated by the import stack (torch + pandas + numba + yfinance),
about 520 MB.

## Memory budget (7,620 MB physical; swap 12 GB = 8 GB /swapfile + 6×635 MB zram)

| Consumer | Trading | Trading + promotion | Weekly retrain (bots stopped, per code) |
|---|---|---|---|
| Desktop (graphical.target, measured user PSS ~0.73 GB + Xorg/gdm) | ~1.0 GB | ~1.0 GB | ~1.0 GB (0 if headless) |
| Kernel unreclaimable + CMA | ~0.35 GB | ~0.35 GB | ~0.35 GB |
| run_pipeline parent | 0.18 | 0.18 | 0.18 |
| backfill worker | 0.03 | 0.03 | 0.03 (paused, still resident) |
| combined bots (measured 0.65; + lgb, challenger stacks, runtime caches) | **0.75–1.0** | 0.75–1.0 | **0 (stopped, run_pipeline.py:1792 `_stop_bots`; restarted :1811)** |
| `meta_label.py` spawned by `shadow.promote_challenger` (shadow.py:875), loads the full frame | — | **~1.5 (stock)** | — |
| hypersearch host side: torch 0.39 + stock frame 1.4 transient + ScaledCache ~0.4 (log: "349+54 MB") + LGB X ≤0.6 + Dataset copy ≤0.6 | — | — | **2.0–2.5 peak** |
| CUDA context (code comments say 0.6–1.2 GB; not measured, GPU off-limits) | — | — | ~0.8 |
| GPU tensors, capped by `set_per_process_memory_fraction(0.40)` (hypersearch_v2.py:114-115) | — | — | ≤3.05 |
| **Total** | **~2.3–2.6 GB** | **~3.8–4.1 GB** | **~7.2–7.9 GB with desktop / ~6.2–6.9 headless** |

- **Trading is safe.** There is more than 5 GB of headroom. `MemAvailable` right now is 5.3 GB.
- **The daily shadow flow is safe.** The challenger is loaded **in the bot process**, once per book per challenger
  mtime (shadow.py:199-214), and costs about +5 MB per model. The parent's daily `evaluate_and_maybe_promote`
  (run_pipeline.py:150) loads **no** model: it uses numpy plus close fetches (parent peak 177 MB). The only large
  item is the background `meta_label.py` that promotion spawns (about 1.5 GB for stock), which is fine next to the
  bots.
- **The weekly retrain is marginal with the desktop on.** GPU pages on the Jetson are pinned nvmap memory and cannot
  swap. The pipeline log has 77 `NvMapMemAllocInternalTagged … error 12` lines, 4 `[OOM] Trial` kills and 5
  `[OOM-RETRY]` batch halvings. The earliest ones came right after the GUI stopped the bots (pipeline_output.log
  lines 37-297), so they happened even without bots running. Headless gives about 1 GB of margin. Running the bots
  during training would need headless mode **and** a lower fraction (about 0.30).
- **`MemoryMax=6G`** covers the host memory of parent + children. Hypersearch's host side (about 2.5–3.3 GB) fits.
  Whether nvmap/GPU pages are charged to the memory cgroup could not be verified without root or debugfs; I think it
  is likely they are not. So the physical/nvmap limit binds before 6G does. If 6G ever does bind, finding 3 applies.
- **Swap as it is today:** zram (priority 5) is used before the swapfile (priority −2). zram compresses into the same
  RAM that nvmap needs, and `vm.swappiness=60`. `setup_jetson_system.sh` was never run.

## Findings, ranked

**P0-1 — The unit interpreter is wrong, so the weekly retrain crashes the pipeline every week.**
`scripts/setup_jetson_system.sh:156` sets `PYBIN="$(command -v python3)"` and runs under `sudo`. Ubuntu's
`secure_path` has no conda, and `which -a python3` shows only miniforge-base, /usr/bin and /bin, so PYBIN becomes
`/usr/bin/python3` (3.10.12). I could not confirm this directly because sudo needs a password. Measured under
/usr/bin/python3:
- `run_pipeline._needs_force_harvest(True,True)` raises `ModuleNotFoundError: pyarrow` at run_pipeline.py:1146.
- It is called uncaught at :1750, before `_stop_bots` at :1792. `main` unwinds, the `finally` block kills the bots,
  the process exits 1, and `Restart=on-failure` restarts `--bot-only`.
- `_next_retrain_time` (:916) then schedules +7 days, so **the retrain never happens**.
- `import trading_utils` fails on dotenv and `market_data` fails on yfinance, so the daily shadow evaluation logs a
  warning and returns None (shadow.py:639-645). **No challenger is ever promoted.**
- `promote_challenger` relaunches `meta_label.py` with `sys.executable` (shadow.py:875), which would also be the
  system python.

The miniforge base python (3.12) fails the same way on pyarrow.
Fix: `ExecStart=/home/kyle/miniforge3/envs/jetson/bin/python -u run_pipeline.py …`.

**P0-2 — `Environment=CUDA_VISIBLE_DEVICES=` on the unit hides the GPU from training.**
setup_jetson_system.sh:169. `ENV = {**os.environ, …}` (run_pipeline.py:276) carries the empty value into every
training child. Only the bots get it on purpose, through `BOT_ENV` at :291. hypersearch then picks
`device = cuda if available else cpu` (hypersearch_v2.py:106). Measured: `torch.cuda.is_available()` returns False
with `CUDA_VISIBLE_DEVICES=''`.
- CPU training runs at 11.6 s per step, about 824 s per epoch at the h 288 champion's size.
- `MAX_TRIAL_SECONDS=900` (:756) is checked once per epoch (:1089), so each fold stops after about 1 epoch. That gives
  near-untrained models and about 40 min per trial.
- At 120 trials that is roughly 80 h per book. For comparison, the GPU runs in history took 525–1094 min per book.
- The bots stay stopped the whole time.

The parent never imports torch (measured: `torch` is not in sys.modules), so it does not need the GPU hidden.
Fix: remove the line. `BOT_ENV` and the `run_bots.py` setdefaults already hide the GPU from the bots.

**P1-3 — systemd `DefaultOOMPolicy=stop` (verified with `systemctl show`; systemd 249).**
An OOM kill of any process in the unit (hypersearch at nvmap/cgroup pressure, or the bots) stops and restarts the
whole service. The in-pipeline `MAX_PHASE_RETRIES=3` (run_pipeline.py:1156) and `_check_restart_bots` never run, and
the week's retrain is lost. Fix: `OOMPolicy=continue`.

**P1-4 — Unified-memory oversubscription during retrain (evidence above).**
The setup script was never applied:
- `systemctl get-default` returns graphical.target.
- nvzramconfig is enabled.
- swappiness is 60.
- The existing /swapfile is **8 GB**. The script's "12 GB" `fallocate` is skipped because the file already exists.

Fix: go headless (about 1 GB freed), disable zram, and set swappiness to 15, as the script intends.

**P1-5 — No alerting and no backups.**
- `.env` holds only `ALPACA_*` and `FINNHUB_API_KEY`. No `TRADER_TELEGRAM_*`, `TRADER_WEBHOOK_URL` or
  `TRADER_HEALTHCHECK_URL` is set anywhere (env, .env, ~/.bashrc, ~/.profile). `notify` (notify.py:77-110, which uses
  `os.getenv` only) does nothing, and Telegram /halt, /flatten and /status do not exist.
- Even once added to `.env`, the **parent** would not see them until something imports `trading_utils`, the only
  `load_dotenv()` call (trading_utils.py:20). That happens at the first EOD digest or the first shadow evaluation with
  rows. So the parent's kill-switch poller and crash alerts start out blind. Put them in the unit through
  `EnvironmentFile=`.
- No backup cron exists (`crontab -l` has only the NVIDIA updater) and there is no `~/trader_backups`.

History shows what this costs: **1,778 stock-bot crash-restarts** (pipeline_output.log:5387–9269, about 60 s apart,
about 30 h in total). The cause was `ValueError: Unknown format code 'f' for object of type 'str'` at
fundamentals.py:339 (`f"P/E={pe:.1f}"` with a yfinance string PE). Nobody was alerted.
- That crash is now **contained** by the per-cycle `try/except` in `base_loop.run`.
- Line 339 itself is unchanged: `pe_ratio = info.get("trailingPE") …` at :69 is still not coerced. The failure mode
  has moved from crashing the bot to losing the LLM candidate list for that cycle.

**P1-6 — The weekly retrain takes the bots offline for 20–33 h.**
Measured phase durations: 04-04 1094+896 min; 04-11 630+598; 04-18 790+888; 04-25 525+647; 05-02 740+885. The current
pipeline adds harvest, meta and gate phases on top of these. The crypto book is dark for about 12–20% of each week.
Crypto resting GTC stop-limits (crypto_loop.py:181-201) survive `_stop_bots`, because `run_bots` does not cancel them
on SIGTERM. But **the account today has 0 open orders against 6 open crypto positions**. This is an owner decision:
with the desktop off, bots plus training could share memory if the GPU fraction is lowered.

**P2-7 — Watchdog margin.**
- READY=1 is sent at run_pipeline.py:1444.
- WATCHDOG=1 is sent from the heartbeat thread every 30 s (:352-365), but only while `mark_progress()` was stamped
  within 600 s (:313).
- Progress is stamped on every phase output line (:473), every 60 s monitor cycle (:870) and every thermal-wait poll
  (:342).

So a phase is killed after about 600 + 900 = **~25 min without output**. On GPU, the longest silence in history is
**13.5 min** (trial 113, 2026-04-18 06:18:54 → 06:32:24). The per-fold `[CACHE]` line is skipped when the cache is
reused, and there are no per-epoch prints. The new refit, OOF and meta/gate phases have no history, so their quiet
periods are unmeasured. On CPU (P0-2), one epoch alone takes about 13.7 min.
Fix: add a per-epoch or keep-alive print in hypersearch, or raise `WATCHDOG_STALL_SEC`.

**P2-8 — Bot crash-restart has no backoff or escalation** (run_pipeline.py:868-913). It restarts every 60 s forever
and sends only a deduped `notify`.

**P2-9 — Combined-mode status is always "not running"** (MAP §9 row 22, confirmed). `_update_per_bot_status`
(run_pipeline.py:757-763) matches only `'Crypto'`/`'Stock'`, never `'Bots'`. So `/status`, `pipeline_status.json` and
the GUI all report both bots not running in production mode. A related smaller issue: in combined mode,
`suspend_and_start_bot` starts split `crypto_loop.py`/`stock_loop.py` (`_start_single_bot` :735).

**P2-10 — Logs.** Totals are bounded, but there are four problems:
- **Bot stdout rotation.** Rotation (20 MB, one-deep, :576) happens only in `_start_bot`, so in steady state it runs
  about once a week. Measured volume is up to 3.1 MB/day per book (stock mean 1.67, crypto 1.58 over their active
  days). Combined mode writes both books into `crypto_bot_output.log`, about 6 MB/day, which bounds the pair at
  roughly 125 MB.
- **Stale split-mode logs.** `stock_bot_output.log` (110 MB, 2026-02-23 → 05-07) and `crypto_bot_output.log` (57 MB)
  are left over from split mode; each run lasted weeks between rotations.
- **Every logger line is written twice**: to stdout through the StreamHandler (log_config.py:38-41) and to
  `logs/trader.log`.
- **Several processes rotate `logs/trader.log` at once.** Every process has its own `RotatingFileHandler` on the same
  file (10 MB × 5), and the backup sizes are irregular (`.1` = 2.5 MB, `.5` = 1.6 MB). A handler still attached to a
  renamed or deleted inode can keep disk space until that process restarts.

`pipeline_output.log` (1.15 MB over about 10 weeks, about 6 MB/yr), `backfill_output.log`, `sentiment_fetch.log` and
`meta_retrain.log` are never rotated but stay small.

**P2-11 — `journals/` is unrotated** (`TRADER_JOURNAL_ROTATE_DAYS` defaults to 0, trade_journal.py:63). Across the
May files the rate was 0.24–0.82 MB/day. There is now also a `cycle_latency` row per cycle per book (base_loop.py:420),
which I estimate at +0.9 MB/day. That gives about **0.4–0.6 GB/yr**. Disk is not at risk (382 GB free on /), and
readers are windowed (the EOD digest reads 7 days).

**P2-12 — `sentiment_cache.db`** is 40 MB in WAL mode (no `-wal` file present now) and is never pruned. It holds
63,743 articles, all fetched in Feb 2026, 2020-07-22 → 2026-02-21. Growth is small.

**P2-13 — Weekly dual write.** `save_training_data` (data_utils.py:152) writes the parquet (389 MB) **and** the 1.1 GB
stock CSV on every harvest. That is about 78 GB/yr of redundant writes plus 1.3 GB of duplicate storage. NVMe
endurance is fine; the extra minutes of CPU fall inside the retrain window.

**P2-14 — The nvpmodel guidance is wrong for this board.** The script says `-m 1` = 15 W and `-m 2` = 25 W
(setup_jetson_system.sh:198-199). This board's `/etc/nvpmodel.conf` has 0 = 15 W, 1 = 25 W, 2 = MAXN_SUPER, 3 = 7 W.
The active mode is MAXN_SUPER (2).
- Idle `tegrastats`: VDD_IN 6.0–6.3 W, CPU 1–27% at 729–883 MHz, GR3D 12–20%, cpu/gpu/tj about 50 °C,
  RAM 2361/7620 MB, swap 1 MB.
- Fan: PWM 74/255, 1542 rpm.
- soctherm OC events: 0/0/0.

The CPU governor is schedutil, so staying on MAXN_SUPER costs little while idle.

**P2-15 — The parent's LD_PRELOAD works by luck.** In the jetson environment without LD_PRELOAD, `import pandas` (or
`market_data`, or `torch`) followed by `import sqlite3` fails with `CXXABI_1.3.15`. The parent survives only because
Phase B imports `sentiment_history` (and with it sqlite3) before anything imports pandas (run_pipeline.py:1514).
Fix: set LD_PRELOAD and LD_LIBRARY_PATH in the unit.

**P2-16 — Docs drift.**
- docs/MODULES.md:869 says the unit is "installed but deliberately not enabled". It is **not installed**
  (`Unit trader.service could not be found`).
- MODULES.md:992 says "12 GB swapfile". The swapfile is 8 GB, plus zram.
- The script's "no RTC battery" rationale: this board exposes rtc0/rtc1 (nvvrs-pseq-rtc).
- The setup script and MODULES cite `hw_monitor.wait_for_cool_gpu`, which has **no production caller**; the real gate
  is `run_pipeline._bounded_thermal_wait`.

**P2-17 — Invisible warnings under systemd.** `_print` is silent when stdout is not a TTY (run_pipeline.py:49-55). The
D03 gate warning in `_build_training_phases` and the harvest-skip messages therefore never reach any log in service
mode. Also, SIGHUP exits with status 0 (:1394 → :1360), so `Restart=on-failure` does **not** restart after a HUP.

**Checks that passed:**
- **hw_monitor on JetPack 6.2:** the zone scan finds `thermal_zone1` = `gpu-thermal` (50.25 °C). The cv0-2 zones
  return EAGAIN on `temp`, but the scan reads only `type`. `get_ram_usage` returns (2386, 7620) MB.
  `is_gpu_available` short-circuits to False under the bot environment. Trip points are passive 70/99 °C and critical
  104 °C. The code's thresholds (70 °C search gate, 75 °C bot throttle at trading_utils.py:31) are sensible.
  hw_monitor has **no fan reader** (the fan is at `/sys/class/hwmon/hwmon0/pwm1` and `hwmon2/rpm`).
- **gpu_lock round trip in a subprocess:** before = free. Held = locked by `ops_audit_probe` with the info JSON
  written. A second process trying `LOCK_EX|LOCK_NB` was **blocked**, as it should be. After release = free and
  `.gpu_lock_info.json` was removed. **`.gpu.lock` stays by design**: it is the 0-byte flock inode, it existed before
  my test (2026-05-02), and it is gitignored. Deleting it would break mutual exclusion, so I left it. Its mtime was
  bumped by the `open('w')`.
- **Time:** systemd-timesyncd is active with NTPSynchronized=yes. chrony is not installed (the script would install
  it). TZ is America/Chicago, so the retrain runs Saturday 02:00 CDT. Alpaca's clock agrees with local UTC.
- **DNS:** 75.75.75.75 and 75.75.76.76, fallback 8.8.8.8. paper-api.alpaca.markets and data.alpaca.markets resolve.
- **Alpaca (read-only, `scripts/connection_test.py`):** status ACTIVE, `trading_blocked=False`,
  `account_blocked=False`, equity $122,017.08, buying power $374.52, 6 positions (all crypto), 0 open orders. Market
  closed; next open 2026-09-28 09:30 ET.
- **The C extension does not crash on the live path.** Two full prediction passes (52 symbols, 5 threads) succeeded
  with `_HAS_C=True`, with predictions identical to the numba path. The SIGABRT seen in pytest was not reproduced
  here.

## Recommended unit and environment

```ini
[Unit]
Description=Trader pipeline (bots + weekly retrain)
After=network-online.target time-sync.target
Wants=network-online.target

[Service]
Type=notify
NotifyAccess=all
User=kyle
WorkingDirectory=/home/kyle/trader
# Jetson env ONLY — /usr/bin/python3 and miniforge base both lack pyarrow (P0-1)
ExecStart=/home/kyle/miniforge3/envs/jetson/bin/python -u run_pipeline.py --combined-bots --bot-only
Environment=PYTHONUNBUFFERED=1
Environment=LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6
Environment=LD_LIBRARY_PATH=/home/kyle/miniforge3/envs/jetson/lib:/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/nvidia/cusparselt/lib
# NO CUDA_VISIBLE_DEVICES here (P0-2): BOT_ENV + run_bots.py already hide the GPU from bots; training needs it.
Environment=TRADER_JOURNAL_ROTATE_DAYS=30
# Makes TRADER_TELEGRAM_* / TRADER_HEALTHCHECK_URL visible to the PARENT too (P1-5). Add them to .env first.
EnvironmentFile=/home/kyle/trader/.env
Restart=on-failure
RestartSec=30
WatchdogSec=900
OOMPolicy=continue          # let run_pipeline's 3x phase retry / bot crash-restart handle a killed child (P1-3)
OOMScoreAdjust=200
MemoryMax=6G                # headless: 6.5G is reasonable; GPU/nvmap pages are likely not charged here

[Install]
WantedBy=multi-user.target
```

Also do these (they need root, so they are owner actions):
- `systemctl set-default multi-user.target`
- disable nvzramconfig
- `vm.swappiness=15`
- add a 02:30 `backup_state.sh` cron
- set `TRADER_TELEGRAM_BOT_TOKEN`, `TRADER_TELEGRAM_CHAT_ID` and `TRADER_HEALTHCHECK_URL`
- fix the nvpmodel IDs in the script's guidance
- before enabling the service, sort out the 6 stop-less, zero-cost-basis crypto positions

In code (reported only, not implemented):
- coerce `pe_ratio` in fundamentals.py
- add backoff to `_check_restart_bots`
- teach `_update_per_bot_status` about `'Bots'`
- add a hypersearch keep-alive print per epoch
- rotate `crypto_bot_output.log` on size from the wait loop, not only at launch

No production files were edited. Nothing was trained, harvested or ordered, and no bots or services were started.
