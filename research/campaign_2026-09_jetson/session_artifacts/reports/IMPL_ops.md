# IMPL_ops — G3 ops & orchestration fixes (2026-09-26/27)

Files touched: run_pipeline.py, log_config.py, scripts/backup_state.sh,
tests/test_jetson_ops_2026_09.py (extended), tests/test_g3_ops_2026_09.py (new, 43 tests).
**NOT touched (G3-8 blocked, see below):** hw_monitor.py, research/campaign_2026-08/08_removed_code.md, docs/MODULES.md.
Nothing was started, trained, committed or installed. No pipeline or bot ran; the bots in the tests are fakes.

## Implemented

### G3-1: combined-mode post-suspend start
- New helper `run_pipeline._start_bots_now(bots, log_fh, want_crypto, want_stock) -> (result, reason)`.
  - It holds the start_bot body verbatim. Combined mode: if a 'Bots' process is alive, reject; otherwise `_launch_bots(*_BOT_SCOPE)`. Split mode: start the per-book loops.
- New helper `_start_pending_bots(bots, log_fh, pending)`.
  - It is a no-op when nothing is pending; otherwise it calls `_start_bots_now`.
  - It replaces both post-suspend blocks in main() (the no-retrain branch and the Phase C branch). main() no longer calls `_start_single_bot` anywhere.
- Defence in depth: `_start_single_bot` refuses, with a log line, when `_COMBINED_BOTS` is set.
  - No combined-mode caller remains. The split-mode start_bot and the crash-restart paths are unchanged.

### G3-2: suspend_and_start_bot with nothing to suspend
- New `_IDLE_PHASES = ('trading','idle','failed','complete','suspended','')`. It is the same tuple start_bot's guard used before, and start_bot now uses it too.
- A `suspend_and_start_bot` that arrives while the phase is idle is now handled as start_bot:
  - it goes through `_start_bots_now`, so the combined-mode guard applies;
  - the ack uses the original command name;
  - `_suspend_requested` and `_pending_bot_start` are never set.
- During a training phase the latch is unchanged. `test_command_ack::test_suspend_and_start_bot_accepted` (phase='crypto_search') still passes.

### G3-3: run_phase deadlock
- The Popen now passes `errors='replace'`.
- New `_drain_phase_output(proc)`. It reads `proc.stdout.buffer` to EOF with `read1` (no decoding), stamping `mark_progress()` per chunk. If even the drain raises, it kills the child: a -9 retry is better than a deadlock.
- The `except` branch in run_phase now:
  - calls `_announce(...)`;
  - does a best-effort `log_fh.write`;
  - drains the pipe, then `finally: proc.wait()`.
- The footer `log_fh.write` is now try/except, so a broken log cannot turn a finished phase into a crash.
- rc is still the child's exit code.

### G3-4: status tmp race
- `tmp = STATUS_FILE + f'.tmp.{os.getpid()}.{threading.get_ident()}'`.
- Added `except BaseException`: it unlinks the partial tmp and re-raises. This is the heartbeat's RuntimeError path, so no per-thread tmp litter is left and behaviour is otherwise unchanged.
- I did NOT add `_heartbeat_lock` to the main-thread writer. The forced-interleave proof does not need it: separate inodes mean last writer wins, atomically.

### G3-5: log_config
- New `SharedRotatingFileHandler(RotatingFileHandler)`. It is the hunter's tested design:
  - an inode check in `shouldRollover` reopens after a foreign rotation;
  - `doRollover` runs under `fcntl.flock` on `trader.log.lock`, with the inode re-checked inside the lock;
  - on non-POSIX there is a lockless fallback.
- It keeps the same path, maxBytes, backupCount, encoding and format.
- `_setup` uses the subclass unless the module-level `RotatingFileHandler` has been swapped. `tests/test_review_b20.py::test_setup_failure_does_not_latch` injects an exploding handler through that name, and the swap is honoured so that test stays green.
- New `get_file_logger(name)`: a logger with `propagate=False` that holds only the shared trader.log handler (used by G3-6).
- New runtime file: a 0-byte `logs/trader.log.lock` appears on the first rotation. `logs/` is gitignored and nothing in the repo globs `trader.log*`.

### G3-6: five TTY-only messages
- `_print` gains `always=False`.
- New `_announce(msg, level='warning')`:
  - calls `_print(msg, always=True, flush=True)`, which is `print(msg, flush=True)`;
  - then `get_file_logger('run_pipeline').<level>(msg)`;
  - never raises.
- I routed it through `_print` rather than a bare `print` because `tests/test_c26_Q2.py::_spy_print` spies on `rp._print` to assert that the D03 warning fires. A bare print would have broken a test I don't own.
- Sites converted: GPU-still-hot (thermal), the read-loop error (G3-3), crypto/stock harvest-skipped (both `level='info'`) and D03.
- The two signal-handler `_print` sites are untouched.
- I used a file-only logger rather than `logger.warning` on the root logger. Root carries a stderr console handler, and both systemd and the GUI launcher merge stderr into the same sink as stdout, so the line would have been written twice.

### G3-7: backup_state.sh
- `PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"`. It is the same line as setup_jetson_system.sh, and the default equals run_pipeline.PYTHON.
- New `sqlite_backup()` uses the `sqlite3` CLI when `command -v` finds it, and otherwise the stdlib `Connection.backup` via `$PYBIN`.
- It keeps the `|| echo "WARN: sqlite backup failed for $db"` continue-on-failure behaviour. `bash -n` passes.

## BLOCKED: G3-8 (hw_monitor.wait_for_cool_gpu removal): not done
The hunter's "zero callers" proof missed a test pin. `tests/test_review_b19.py::test_hw_monitor_cosmetic_pins` asserts:
- :352 `'print("[HW] Cannot read GPU temp' in src` (that line exists only inside wait_for_cool_gpu);
- :358 `'_TEMP_CACHE_TTL' in hw_monitor.wait_for_cool_gpu.__doc__`.

Removing the function makes that test fail, and test_review_b19.py is outside my ownership. So hw_monitor.py, 08_removed_code.md and docs/MODULES.md are untouched. The archived block would be false if the function still exists.

To finish G3-8, the orchestrator needs to grant edit rights to tests/test_review_b19.py to drop those two asserts, with an archive note. Then:
- remove hw_monitor.py:153-169;
- append the verbatim block to 08_removed_code.md;
- edit MODULES.md :887 (Key API) and :888 (Known issues). That is two lines, so it is not the single one-line doc change the task allowed.

## Tests
tests/test_g3_ops_2026_09.py (43 tests). run_pipeline and log_config are imported directly (stdlib-only; importorskip guard). Every file lives in tmp_path, the bots are fakes, and the only real subprocesses are tiny `sys.executable -c` children and bash.

| Fix | What the tests cover |
|---|---|
| G3-1 | The hunter's two-click sequence leaves exactly 1 crypto trader, and the start_bot ack is 'rejected'; stop_bot{crypto} then leaves 0. Launches follow `_BOT_SCOPE`: both books, `--crypto-only`, `--stock-only`. No duplicate when a combined process is alive. Split mode is parametrized and unchanged. `_start_single_bot` refuses in combined mode. main() uses the helper twice and never calls `_start_single_bot`. |
| G3-2 | Every idle phase maps to start_bot with no latch, and the ack is `suspend_and_start_bot`/accepted. Combined mode launches run_bots.py, and a second command is rejected. A training phase still latches. The hunter's R2: the next real phase returns rc 0 with no `[SUSPEND]`. |
| G3-3 | A driver subprocess under `timeout=90` covers three cases: an undecodable byte gives rc 0 with U+FFFD and the full output logged; a log write that fails with ENOSPC gives rc 0; the same with a child exiting 7 gives rc 7. Source pin on `errors='replace'` and the drain. The drain kills the child when unreadable. |
| G3-4 | Both forced interleaves (heartbeat paused while main writes a longer status, and the mirror) produce valid JSON with no tmp litter. Source pin on the tmp name. A partial tmp is removed on RuntimeError. |
| G3-5 | 4 writer processes with maxBytes 20 KB: 0 "Logging error", 5 backups each ≥ 90 % of maxBytes, every line intact. A two-handler foreign-rotation test covers follow-the-new-inode and no double rotation; it would fail with the stdlib handler. `get_logger` installs the subclass on the same path and params. `get_file_logger` does not propagate and is idempotent. |
| G3-6 | Source-text: all 5 messages go only through `_announce`, and `_signal_handler` does not call it. Behavioural: output reaches a non-TTY stdout and the file logger at WARNING/INFO. It never raises. |
| G3-7 | The DB step is extracted verbatim and run against WAL temp DBs holding 100 rows: with PATH = an empty dir (no CLI), both DBs are staged with 100 rows and no WARN. A fake CLI in PATH is preferred. With no CLI and a bad PYBIN: 2 WARNs and the script continues. `bash -n` passes. |

Two tests were added to tests/test_jetson_ops_2026_09.py:
- the backup PYBIN line is identical to the setup script's, its default equals run_pipeline.PYTHON, and the CLI is tried before the fallback;
- `bash -n` on backup_state.sh.

## Verify
```
py_compile run_pipeline.py log_config.py hw_monitor.py tests/test_g3_ops_2026_09.py tests/test_jetson_ops_2026_09.py -> PYC_OK
bash -n scripts/backup_state.sh -> BASH_OK
$JPY -m pytest tests/test_g3_ops_2026_09.py tests/test_jetson_ops_2026_09.py tests/test_pipeline.py \
  tests/test_command_ack.py tests/test_c26_T1.py tests/test_c26_T4.py tests/test_hw_monitor.py \
  tests/test_gpu_lock.py -q -p no:cacheprovider                     -> 174 passed in 5.80s
(no tests/test_log_config*.py exists; log_config is pinned in test_review_b20/test_new_modules)
Neighbours that pin the touched code (PYTHONPATH=noc):
$JPY -m pytest tests/test_c26_P2.py test_c26_Q2.py test_c26_T7.py test_c26_X1.py test_c26_R1.py \
  test_new_modules.py test_gate_protocol.py test_review_b06.py test_review_b19.py test_review_b20.py \
  test_review_b21.py test_r2c_training_repairs.py test_imports.py   -> 353 passed in 16.16s
```
The full suite was not run, per the brief.

## Notes
- `_announce` writes to logs/trader.log. As a result, tests that exercise `_build_training_phases` (the D03 path) or `_bounded_thermal_wait` now add a `[run_pipeline] WARNING` line to the repo's real `logs/trader.log`: 7 such lines were seen after these runs. This is the same pre-existing smell as "importing almost any module touches trader.log", and `logs/` is gitignored.
- Calling `_announce` in the pipeline parent runs `log_config._setup()` there. That adds the root console and file handlers if no lazily imported module had done so already.
- Not in scope, unchanged: bot restart backoff, SIGHUP exit 0, the signal-handler `_print` sites, and the watchdog margin. See the G3 appendix.
