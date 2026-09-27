# G3 — ops & orchestration: indisputable-improvement hunt (2026-09-26)

**Scope:** run_pipeline.py (including today's FIX_E edits), notify.py, gpu_lock.py, hw_monitor.py,
log_config.py, trade_journal.py, adaptive_config.py, monitor_drift.py, retrain_ledger.py,
scripts/{setup_jetson_system,backup_state,ab_check}.sh. I also verified each E_ops P2 item.

**How this was done:** read-only; no production file was edited. Every repro sits in
`<scratchpad>/hunt/G3/` and redirects every file it writes (status, command result, logs) into a
scratch temp dir. The repros use **fake bot processes**, so no bot, pipeline, training or order ran.
Jetson env, `CUDA_VISIBLE_DEVICES=''`, each process <100 MB and <60 s.

**Line numbers:** the cites are for the current working tree, after FIX_E.

---

## Findings, ranked by severity

### G3-1 · run_pipeline.py:1653-1657, :1828-1832 (and :754-772) · class A · combined mode: "suspend training → start bot" launches a split `crypto_loop.py`; a second GUI click then adds `run_bots.py`, so **two processes trade crypto**. "Stop Crypto" cannot stop the split loop.

**Defect.** After a GUI suspend, both post-suspend blocks call `_start_single_bot(bots, 'Crypto'|'Stock')`.
- That function always spawns the split `crypto_loop.py` / `stock_loop.py` and has no `_COMBINED_BOTS` branch (:754-772).
- In combined mode (the systemd unit's mode):
  - `_handle_command('start_bot')` checks only for a live `'Bots'` process (:841). It then launches `run_bots.py` with both books next to the split loop.
  - `stop_bot` stops only `'Bots'` (:814).
- The code's own invariant is at :843-845: "Starting a per-book loop here would duplicate order flow alongside the already-running combined process." The GUI enforces the same rule at gui.py:9723.

**Proof.** `hunt/G3/r1_r2_combined_suspend.py` → `r1_r2.out`:
```
after suspend block : [('Crypto', 'crypto_loop.py')]
after start_bot     : [('Crypto', 'crypto_loop.py'), ('Bots', 'run_bots.py')]
processes trading crypto: 2 ['crypto_loop.py', 'run_bots.py']
after stop_bot crypto: [('Crypto', 'crypto_loop.py')]   # the user's "stop crypto" killed the stock book, crypto keeps trading
```

**Reachability with the real GUI** (two clicks):
1. During training, click Start Crypto. `_combined_bots_running()` is False because the bots are stopped for the retrain, so the GUI writes `suspend_and_start_bot`.
2. Click Start Stock. pgrep finds no `run_bots.py` and the phase is `trading`, so the GUI writes `start_bot`.

No bot has a single-instance lock (grep of run_bots/base_loop/crypto_loop/stock_loop: no flock or pidfile). The split loop is also crash-restarted as a split loop (:891-894).

**Fix.** Add one helper and use it at both post-suspend sites:
```python
def _start_pending_bots(bots, log_fh, pending):
    if not (pending.get('crypto') or pending.get('stock')):
        return
    if _COMBINED_BOTS:          # same semantics as start_bot's combined branch (:840-858)
        if not any(n == 'Bots' and p.poll() is None for n, p, _ in bots):
            _launch_bots(bots, log_fh, *_BOT_SCOPE, verb='started')
        return
    if pending.get('crypto'): _start_single_bot(bots, 'Crypto', log_fh)
    if pending.get('stock'):  _start_single_bot(bots, 'Stock', log_fh)
```
Optional defence in depth: `_start_single_bot` returns early (with a log line) when `_COMBINED_BOTS`.

**Blast radius.**
- Changes only the two post-suspend blocks in main().
- `_start_single_bot` callers: :861/:864 (split mode only) and the two sites above.
- Tests: `tests/test_command_ack.py` never hits these blocks. No test pins split starts in combined mode.

**Why indisputable.** It violates an invariant the code and the GUI both state, and it is shown end to end with the real functions.

---

### G3-2 · run_pipeline.py:869-883 (latch), :551-564 (consumer) · class A · a `suspend_and_start_bot` consumed by the wait loop latches `_suspend_requested`, which **aborts the next weekly retrain at its first output line**, days later.

**Defect.**
- `_handle_command` has exactly two callers, :1578 and :1704. Both are the trading wait loops, where no phase is running.
- Its `suspend_and_start_bot` branch sets the global `_suspend_requested = True` and `_pending_bot_start`, then acks `accepted`. It starts nothing.
- The flag is cleared only inside `_run_training` after a `-99` (:1233).
- So the next `run_phase` (the Saturday harvest) sees the latch on its first line, terminates the child and returns -99. `_run_training` returns `'suspended'`, the retrain is skipped for the week, and the stale `_pending_bot_start` bots are started. In combined mode those are split loops, which is G3-1.

**Reachability.** The GUI reads `phase` before its modal "Suspend training?" `QMessageBox` (gui.py:9728-9747). If the last phase finishes while the dialog is open, the command lands in the wait loop.

**Proof.** `r1_r2.out`:
```
_suspend_requested after wait-loop command: True | bots started now: []
run_phase rc = -99 (-99 = aborted as suspended)
phase log tail: [..., 'line 0', '[SUSPEND] Training suspended by GUI command']
```

**Fix.** In the `suspend_and_start_bot` branch, handle the no-training-phase case, which with today's callers is the only reachable case:
- if `status.get('phase','') in ('trading','idle','failed','complete','suspended','')`, run the `start_bot` branch body (the same bot-start logic, including its combined-mode guard) and ack with the original command name;
- keep the latch only when the phase really is a training phase.

**Blast radius.** `tests/test_command_ack.py::test_suspend_and_start_bot_accepted` uses `phase='crypto_search'`, so it keeps passing unchanged. No other caller.

**Why indisputable.** A command that can only arrive when nothing is running must not arm a trap that kills an unrelated future run.

---

### G3-3 · run_pipeline.py:478-486, :567-570 · class A/D · any exception in `run_phase`'s read loop is swallowed, then `proc.wait()` **deadlocks** on a child whose stdout pipe is no longer drained.

**Defect.**
- `except Exception as e: _print(...)` produces no output when stdout is not a TTY (see G3-6).
- `finally: proc.wait()` then waits on a child that blocks in `write()` once the 64 KiB pipe fills.

**Triggers.**
- One undecodable byte in the child's output (`Popen(text=True)`, strict decoding).
- A mid-phase `log_fh.write/flush` OSError, such as ENOSPC while a harvest writes its 1.1 GB CSV.

**Consequence.**
- Under systemd: `mark_progress` stops, and the watchdog restarts the unit after about 25 min, killing the retrain.
- Under the GUI launcher (no watchdog): a permanent hang **with the bots stopped**, because `_stop_bots` ran before `_run_training`.

**Proof.** `hunt/G3/r6_run_phase_deadlock.py` → `r6.out`:
- unmodified: `exit=124` (killed by `timeout 20`, i.e. deadlocked);
- with `errors='replace'`: `run_phase returned rc=0 after 0.1s`.

**Fix.**
1. `subprocess.Popen(..., text=True, errors='replace')`. Output is decoded only for logging and regex progress parsing, so no decision reads it.
2. In the `except` branch, drain before waiting: `try: [None for _ in proc.stdout] except Exception: pass`, so the child can finish and its exit code still decides retry/success. Also write `e` to `log_fh` with a try/except (see G3-6).

**Blast radius.**
- Only `run_phase` changes.
- Tests: `tests/test_pipeline.py` and `tests/test_c26_*` exercise run_phase through fake cmds that emit ASCII only, so they are unaffected.

**Why indisputable.** A log-tailing loop must never turn a logging failure into a deadlock of the thing it tails.

---

### G3-4 · run_pipeline.py:406-411 (with :371-379) · class D · the heartbeat thread and the main thread write `pipeline_status.json` through **one shared tmp path** (`.tmp.<pid>`, same pid, and the main thread never takes `_heartbeat_lock`). A collision publishes a **torn file**.

**Defect.**
- Two `open(tmp,'w')` calls on the same path share one inode.
- The first `os.replace` publishes it, then the second thread's buffered bytes land at offset 0 of the already-published file.
- The second `os.replace` fails with FileNotFoundError, which is swallowed.
- `grep _heartbeat_lock` shows use only at :374. The docstring at :390 says callers should hold it; none of the roughly 40 main-thread callers do.

**Proof.** `hunt/G3/r3_status_tmp_race.py` → `r3.out`. The interleaving is forced by pausing only thread A after its `json.dump`:
```
RESULT: TORN pipeline_status.json -> Extra data: line 8 column 2 (char 184)
GUI _read_pipeline_status() -> {}
```

**Consequences.**
- The GUI (gui.py:1340-1344) sees `{}`: phase unknown, and it treats training as not running.
- On restart, main() `prev_status` (:1429-1434) drops `crypto_final_score` / `stock_final_score`.
- The file self-heals on the next write.
- In production, collisions need both threads inside `write_status` together. That is rare but unbounded over weeks of per-line forced writes during training.

**Fix (one line).** `tmp = STATUS_FILE + f'.tmp.{os.getpid()}.{threading.get_ident()}'`. Each writer then has its own inode and `os.replace` stays atomic; last writer wins. Taking `_heartbeat_lock` inside `write_status` would also work, but it is a wider change.

**Blast radius.** write_status only. Tests: test_command_ack monkeypatches STATUS_FILE and reads the final file; unaffected. A stray tmp is possible only on a crash between open and replace, the same as today.

**Why indisputable.** Two concurrent writers on one tmp path defeat the atomic-replace pattern the function exists to provide.

---

### G3-5 · log_config.py:47-48 · class A · every process attaches its own `RotatingFileHandler` to the one `logs/trader.log`. Rotations race: records are **dropped** ("--- Logging error ---" FileNotFoundError in `doRollover`) and **rotations cascade**, so backups are undersized and history falls short of the 60 MB design.

**Writers.** The parent, the bots, the backfill worker and every phase child (harvest, hypersearch, meta_label, backtest) all call `get_logger`. The brief also notes that nearly every import touches trader.log.

**Evidence on the box.** `logs/`:
- `trader.log.2` = 10.49 MB (May 2 00:57);
- `trader.log.1` = **2.59 MB**, rotated 58 min later (01:55);
- `trader.log.5` = **1.63 MB**.

A single writer only rotates at 10 MB, so a second process rotated a 2.5 MB file.

**Proof.** `hunt/G3/r4_multiproc_rotation.py` → `r4.out` uses the same handler class and params with maxBytes shrunk to 20 KB:

| Writers | Backup sizes | Logging errors (each is a dropped record plus a stderr traceback) |
|---|---|---|
| 1 | 5 × 19,951 B | 0 |
| 2 | 8,946 / 19,951 / **17,395** / … | yes |
| 4 | 4,189 / **14,200** / 19,951 / **13,490** / … | **31** |

**Fix** (stdlib, ~20 lines, tested in `r4_fix_multiproc_rotation.py` → `r4_fix.out`). A `RotatingFileHandler` subclass:
1. `shouldRollover`: if `os.stat(baseFilename).st_ino != os.fstat(stream.fileno()).st_ino`, another process has rotated, so reopen the stream.
2. `doRollover`: run under `fcntl.flock` on `trader.log.lock` and re-check the inode inside the lock. If someone else rotated, just reopen.

Result with the fix, 1/2/4 writers: **0 logging errors** and every backup full-size (19,951–20,022 B). Cost: one `stat` per record.

**Blast radius.** Only log_config `_setup`. Readers of trader.log (the GUI log tab) are unchanged. A new 0-byte `logs/trader.log.lock` appears; `logs/` is gitignored. No test pins the handler class.

**Why indisputable / caveat.** The defect is measured and matches the prod file sizes. Python's own docs say multi-process logging to one file is unsupported. The adjudicator may prefer another fix, such as per-process files, but that changes the path the GUI reads. The flock-and-inode handler keeps every path and format.

---

### G3-6 · run_pipeline.py:365, :568, :983, :994, :1049 · class A (a branch meant to run whose output is unreachable) · five `_print`-only messages are emitted **nowhere** in both production launch modes.

**Defect.** `_print` writes only when stdout is a TTY (:49-55). That is meant to avoid doubling lines that are *also* written to `log_fh`. These five sites never write to `log_fh`:
- :1049 — the D03 gate warning;
- :568 — the swallowed read-loop exception (G3-3);
- :365 — "GPU still hot … proceeding anyway";
- :983 and :994 — harvest skipped.

**Where their output goes:**
- systemd unit: stdout is the journal (not a TTY), so they are dropped.
- GUI launcher (gui.py:9641-9645): `stdout=pipeline_output.log` (not a TTY), so they are dropped.

**Proof.** `hunt/G3/r8_print_silent.py` → `r8.out`:
- `GATE_TARGETS_CHALLENGER=False` (the default) and `TRADER_SHADOW_MODE` is unset (default ON), so D03's condition is true on **every** weekly retrain;
- with stdout redirected, **0 bytes** are captured;
- on a pty, the warning appears.

**Fix.** At those five sites, replace `_print(...)` with `print(..., flush=True)`, unconditionally:
- GUI mode: the line lands in pipeline_output.log exactly once (log_fh never got it);
- systemd: it goes to the journal;
- TTY: unchanged.

The two signal-handler sites (:1352, :1373) are deliberately left out (see the appendix: print inside a signal handler).

**Blast radius.** Output only. No test asserts that these lines are absent.

**Why indisputable.** A warning that fires every week and is written nowhere is a no-op.

---

### G3-7 · scripts/backup_state.sh:22-27 · class A · **the `sqlite3` CLI does not exist on the prod Jetson**, so the Optuna study DBs are never backed up. The only trace is one WARN line in a cron log.

**Proof.**
- `command -v sqlite3` is empty. It is absent from /usr/bin, miniforge base and the jetson env; `dpkg -l sqlite3` reports none.
- `hunt/G3/r5/step1.sh` is lines 21-27 verbatim, run under a cron PATH (→ `r5.out`):
```
step1.sh: line 7: sqlite3: command not found
WARN: sqlite backup failed for v2_study.db
script exit path reached; staged files: 0
```
- The proposed fallback (stdlib online backup API) staged 100/100 rows.

**Fix.** Keep the CLI when it exists, and otherwise fall back to the stdlib online-backup API (the same `.backup` semantics, WAL-safe):
```bash
PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"
if command -v sqlite3 >/dev/null 2>&1; then sqlite3 "$db" ".backup '$STAGE/$db'"
else "$PYBIN" -c 'import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()' "$db" "$STAGE/$db"
fi || echo "WARN: sqlite backup failed for $db"
```
A bare `import sqlite3` in the jetson python needs no LD_PRELOAD; the CXXABI clash needs pandas or torch imported first. The `TRADER_PYBIN` default matches setup_jetson_system.sh:51.

**Blast radius.** Backup script only. There are no tests. The backup cron is not installed yet (E_ops P1-5), so this should be fixed before the owner installs it.

**Why indisputable.** The script's only DB-backup mechanism cannot run on the one machine it targets.

---

### G3-8 · hw_monitor.py:153-172 · class B · `wait_for_cool_gpu` has zero callers.

**Proof.** `grep -rn wait_for_cool_gpu --include=*.py` finds only the definition and a comment at run_pipeline.py:344 ("Replaces a direct hw_monitor.wait_for_cool_gpu() call"). No production caller and no test caller exist. docs/MODULES.md:888 already lists it as "OBJECTIVE-fix-pending (dead code)", superseded by `run_pipeline._bounded_thermal_wait`.

**Fix.** Remove it per repo convention:
- archive it verbatim in `research/campaign_2026-08/08_removed_code.md`;
- drop it from MODULES.md:887 "Key API" and :888.

**Blast radius.** None: no importer uses it. `tests/test_hw_monitor.py` imports only `get_ram_usage`, `get_gpu_temp` and `is_gpu_available`.

**Why indisputable.** It is superseded and unbounded (it can block forever), and it is already flagged as dead in the owning doc.

---

## Verification of the areas the task named

- **systemd notify/watchdog protocol.** Correct.
  - `_sd_notify` delivered `READY=1` and `WATCHDOG=1` over both filesystem and abstract (`@`) sockets, and a dead socket does not raise (`r7.out`).
  - READY=1 is sent at :1468, right after the heartbeat thread starts, well inside the default TimeoutStartSec of 90 s. WATCHDOG=1 is sent every 30 s while progress is under 600 s old (verified: 599 s → ping, 601 s → no ping).
  - The unit has `Type=notify`, `NotifyAccess=all` and `WatchdogSec=900`. The effective hang budget is 600 + 900 = about 25 min. The longest measured silence during a GPU phase is 13.5 min (E_ops P2-7).
  - Progress is stamped on every phase output line (:492), every 60 s monitor cycle (:894) and every thermal poll (:361).
  - A 2 h phase is fine if it prints at least every 25 min. Hypersearch prints per trial, and `MAX_TRIAL_SECONDS=900` bounds a trial.
  - Not proven failing, so it is in the appendix, not a finding.
- **Heartbeat exception safety.** `write_status` swallows OSError; the loop swallows the dict-mutation RuntimeError. Every status value is a JSON primitive (checked every assignment site), so no TypeError or ValueError can escape. Not reachable. The real defect there is G3-4.
- **Signal handling and zombies.**
  - SIGTERM/SIGINT/SIGHUP terminate the in-flight phase child (wait 10 s, then kill) and the bots (wait 3 s), then exit.
  - Crashed bots are reaped by `poll()` in `_check_restart_bots`.
  - The sentiment-fetch and backfill children can each stay a zombie until exit, and shadow's detached `meta_label` Popen until the next Popen (`subprocess._cleanup`). That is bounded at a few, so it is not a leak.
  - SIGHUP exiting 0: appendix.
- **Atomic IPC files.**
  - The GUI writes `pipeline_command.json` and `retrain_trigger.json` via tmp + `os.replace` (gui.py:1353-1357, :9090-9094).
  - The pipeline consumes them with rename-then-read (:237-251, :260-274), and `command_result.json` is written tmp + replace by the main thread only.
  - No TOCTOU there. The only torn-file path is G3-4.
- **Journal append atomicity (two loop threads).** Each row is one `write()` on an O_APPEND fd, including rows over the 8 KiB buffer. 2 threads × 3000 rows (1/7 of them 9 KB) gave **6000 lines, 0 torn** (`r7.out`).
- **adaptive_config round trip.** `update_after_search`, then save, then load, is equal to the in-memory state. Floats are bit-exact (0.4012345678901234, 0.1000000000000001, 1e-17). The only difference is `expansion_history[*].edges` tuples becoming lists, which is log-only and never read as tuples. No tmp left behind (`r7.out`).

---

## E_ops P2 items: verdicts

| E_ops item | Verdict |
|---|---|
| restart backoff (P2-8) | appendix: design choice, no wrong output |
| watchdog keep-alive during hypersearch (P2-7) | appendix: margin, not a proven failure (see above) |
| bot log rotation (P2-10) | appendix: bounded (~125 MB/wk measured by E) |
| `_print` silent under systemd (P2-17a) | **promoted → G3-6** |
| SIGHUP exit 0 (P2-17b) | appendix |
| `suspend_and_start_bot` in combined mode (P2-9b) | **promoted → G3-1**, plus the related latch bug **G3-2** |
| double-written logger lines (P2-10) | appendix: bot stderr → bot log is intended as that bot's console |
| several processes rotating trader.log (P2-10) | **promoted → G3-5** |

---

## Judgment calls (not proposed)

1. **Bot crash-restart has no backoff** (:886-931). It restarts every 60 s forever, with the notify deduped per 10 min. History shows 1,778 restarts, each re-importing torch (~520 MB). A backoff is sensible but is a policy choice.
2. **Watchdog margin during quiet phases.** The budget is 25 min without output. Phases added since the last measured history (refit/OOF/meta/gate) have no measured silence. Options are a per-epoch keep-alive print in hypersearch (a training-path file, off-limits this round) or a larger `WATCHDOG_STALL_SEC`.
3. **SIGHUP → exit 0** (:1418 → :1384). `Restart=on-failure` will not restart after a HUP. Under systemd, HUP only comes from an operator, and a graceful exit 0 is defensible.
4. **Signal-handler `_print` sites** (:1352, :1373) are silent in non-TTY modes. Making them `print` risks "reentrant call inside BufferedWriter" if the signal interrupts another stdout write, so they are left out of G3-6.
5. **`_signal_handler` waits only 3 s for bots and never kills them**; bots slower than that are orphaned (they still got SIGTERM). Under systemd, the cgroup kill covers it.
6. **The weekly retrain restarts manually stopped bots.** `_restart_bots` ignores `_manually_stopped` (:1660, :1835). Arguably intended: a retrain resets to the configured scope.
7. **An uncaught `_needs_force_harvest` exception** (a corrupt parquet → `pq.read_schema`, :1171) propagates out of main(). The pipeline exits, `finally` kills the bots, and the systemd restart comes up `--bot-only`, so that week's retrain is skipped. With the correct interpreter it needs a corrupt store.
8. **`trade_journal.log_decision` re-reads llm_config.json for every row** (trade_journal.py:84, via `load_llm_config`, uncached). I measured no impact; an mtime cache is a class-C candidate only with a measurement.
9. **backup_state.sh omits `sentiment_cache.db`** (40 MB of LLM-scored articles that cost money to rebuild) and the `.env` secrets. That is a content policy choice.
10. **Drift retrain flags are never consumed in `--no-retrain` mode.** `_check_drift_trigger` is only called in Phase C. That is consistent with "no auto-retrain" there.
11. **Combined-mode bot stdout rotation only at launch** (`_rotate_log` in `_start_bot`). This is bounded; see E_ops.

---

## No issues found (per file)

| File | Checked | Result |
|---|---|---|
| run_pipeline.py | FIX_E `_training_env` / `TRAIN_ENV` / `_BOT_SCOPE` status; phase retry and gate rc==3; rename-then-read IPC; `_untrack_handle`; `_next_retrain_time` (DST never lands on a Saturday); watchdog gating | G3-1..4, G3-6 only |
| notify.py | dedupe (+prune >200), critical single retry on the daemon thread, telegram offset persistence (tmp + replace) and chat-id filter, halt/flatten flags, never-raise | no issues |
| gpu_lock.py | `acquired` guard on release, atomic info file, non-blocking-then-blocking flock, shared probe | no issues |
| hw_monitor.py | zone scan (thread-safe publish order), 20 s cache TTL vs the 30 s poll, one-shot warn, meminfo parse | G3-8 only |
| log_config.py | double-checked `_setup` lock, handler build before attach | G3-5 only |
| trade_journal.py | single-clock ts/filename, O_APPEND atomicity (measured), verified-copy gzip rotation with its lock, readers tolerate torn lines | no issues |
| adaptive_config.py | round trip (measured), pid-unique tmp, forward-compat backfill, study-DB sidecar removal ordering | no issues |
| monitor_drift.py | sidecar flock across append/prune/clear, manifest-hash fence, action-streak day logic, CUSUM ts compare (trade_memory ts is uniformly 25-char `+00:00`, 123/123 rows) | no issues |
| retrain_ledger.py | `trailing_purged_rows` bar-count purge, `stack_valid_rows`, `window_gather_plan` offsets, `all_times` = epoch seconds (hypersearch_v2.py:298), returns are ndarrays | no issues |
| scripts/setup_jetson_system.sh | step-0 interpreter check before any change; unit notify/watchdog/OOMPolicy/EnvironmentFile consistent with run_pipeline | no issues (FIX_E + REVIEW L4/L5 already cover it) |
| scripts/ab_check.sh | `-q` + pyproject `addopts=-v` → bordered summary line (verified: bare `-q` would not be bordered, but addopts restores it); watchdog, sanity floor, NAME diff | no issues. Note: `python3` on the Jetson is miniforge base without pytest, so it fails loudly (use `AB_CHECK_PYTEST`) |
| scripts/backup_state.sh | set -euo, nullglob, retention prune | G3-7 only |

**Repro index** (`<scratchpad>/hunt/G3/`):
- `r1_r2_combined_suspend.py` / `r1_r2.out` (G3-1, G3-2)
- `r3_status_tmp_race.py` / `r3.out` (G3-4)
- `r4_multiproc_rotation.py` / `r4.out`, `r4_fix_multiproc_rotation.py` / `r4_fix.out` (G3-5)
- `r5/step1.sh` / `r5.out` (G3-7)
- `r6_run_phase_deadlock.py` / `r6.out` (G3-3)
- `r7_nonissues.py` / `r7.out` (negative checks)
- `r8_print_silent.py` / `r8.out` (G3-6)
