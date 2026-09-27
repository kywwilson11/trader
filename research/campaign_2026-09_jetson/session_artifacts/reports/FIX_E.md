# FIX_E — objective ops fixes (2026-09-26)

Files touched (only these): scripts/setup_jetson_system.sh, run_pipeline.py, fundamentals.py,
docs/MODULES.md (2 sentences), tests/test_jetson_ops_2026_09.py (new). No sudo; nothing installed,
started, trained, committed.

## Diff summary

### scripts/setup_jetson_system.sh (+~55/-8)
- New step 0, run BEFORE any system change: `PYBIN="${TRADER_PYBIN:-/home/kyle/miniforge3/envs/jetson/bin/python}"`
  (default == run_pipeline.PYTHON). Exits 2 with `[python] FATAL` if not executable, or if
  `env LD_PRELOAD=… LD_LIBRARY_PATH=… CUDA_VISIBLE_DEVICES= $PYBIN -c 'import pyarrow, dotenv, torch'` fails.
  Verified live: the jetson python passes; /usr/bin/python3 fails on pyarrow. Usage note: sudo strips env, so
  write `sudo TRADER_PYBIN=… bash …`. The old `PYBIN="$(command -v python3)"` is removed.
- Unit: `Environment=CUDA_VISIBLE_DEVICES=` removed, with a comment citing run_pipeline.py:310 (BOT_ENV) and
  run_bots.py:45 (setdefault ''). Added `Environment=PYTHONUNBUFFERED=1`, `Environment=LD_PRELOAD=` and
  `Environment=LD_LIBRARY_PATH=` (both set to the exact run_pipeline.ENV values, with no inherited suffix),
  `EnvironmentFile=-${TRADER_DIR}/.env` and `OOMPolicy=continue`. I checked that .env is systemd-compatible:
  no `export` prefix, no quotes, every line KEY=VALUE. The rendered unit was inspected.
- "Installed, NOT enabled" is unchanged. The "already exists" message now says to move the file aside to regenerate.
- Swap: the sizing logic is unchanged. /swapfile is 8 GB with mtime 2026-01-13, which predates the script's first
  commit (ccdc068, 2026-06-10), so it is a pre-existing file that the `[[ ! -f ]]` guard skipped. The script did
  not cause it. Only the messages changed: the header notes that an existing file is kept, and the "already
  exists" line prints its size and "NOT resized to 12GB".
- nvpmodel guidance: verified from /etc/nvpmodel.conf (0=15W, 1=25W, 2=MAXN_SUPER, 3=7W; `nvpmodel -q` = MAXN_SUPER/2).
  Trading is now `-m 0`, retrain is `-m 1` or `-m 2`. The thermal-gate reference changed from the callerless
  `wait_for_cool_gpu` to `run_pipeline._bounded_thermal_wait`.

### run_pipeline.py (+25/-1)
- New pure `_training_env(base_env)`: it copies the env and drops `CUDA_VISIBLE_DEVICES` only when the value is ''.
  A non-empty value such as '0' passes through. `TRAIN_ENV = _training_env(ENV)` is used by the
  `run_phase` Popen, which runs every harvest/hypersearch/meta_label/backtest phase. BOT_ENV is unchanged and
  still sets ''. The sentiment/backfill Popens stay on ENV (they are not training).
- `_update_per_bot_status`: a live `'Bots'` process (combined mode) now ORs `_BOT_SCOPE` into
  crypto/stock_bot_running, and bots_running follows. The keys are unchanged. The consumers are compatible:
  gui.py:10036-10042/10330 already ORs its own pgrep heuristic, and Telegram /status (:106) just reads the keys.

### fundamentals.py (+~45/-15)
- Added `_safe_float(v)`, which never raises (None, bool, non-numeric str, NaN and ±inf all become None), and
  `_fmt_num(v, spec)`, which returns the number or the placeholder "n/a".
- Every format site in `format_fundamentals_for_llm` is now fixed the same way: P/E (:339 crash), P/B, MktCap
  (the `>=` compare also raised on str), RevGrowth, EPS, DivYield (the `> 1` compare), Beta and 52wk (skipped
  unless both values are numeric).
- A None value is still omitted, as before. Numeric output is byte-identical; the existing tests pass.

### docs/MODULES.md
- The §ops Goal sentence (formerly "…CUDA_VISIBLE_DEVICES= and is installed but deliberately not enabled")
  now describes the new unit contents and says trader.service is **not installed** on the prod box as of 2026-09-26.
- The setup_jetson_system.sh row now says the 12 GB swapfile is created only if /swapfile is absent (the prod box
  keeps its pre-existing 8 GB file plus zram), and that the unit is not yet installed on prod. It also gives the
  nvpmodel IDs, and the resolved OBJECTIVE-fix-pending note about wait_for_cool_gpu was replaced with
  `_bounded_thermal_wait`. An unrelated indicators_c edit in the same file comes from another agent.

## Tests (tests/test_jetson_ops_2026_09.py — Mac-safe: text/AST only + stdlib fundamentals import)
- (a) Unit heredoc: no non-comment CUDA_VISIBLE_DEVICES; OOMPolicy=continue and EnvironmentFile are present; the
  LD_* values equal run_pipeline.ENV (extracted via AST); still not enabled.
- (b) The exact PYBIN line; its default equals run_pipeline.PYTHON; no `command -v python3` in the code; the
  check comes before step 1. The step-0 block is executed with a fake failing interpreter and with a missing path,
  and the test asserts a non-zero exit plus FATAL. `bash -n` passes.
- (c) `_training_env`, extracted and exec'd alone: '' is dropped, '0', '0,1' and '-1' pass through, the input is
  not mutated. The run_phase Popen env is TRAIN_ENV, BOT_ENV CUDA is '', and _start_bot uses BOT_ENV.
  `_update_per_bot_status` is covered for the combined scopes (both / crypto-only / stock-only / dead) and for
  split mode (unchanged).
- (d) P/E: 12.345, '12.345', '', 'N/A', 'Infinity' and NaN all format without raising; None is omitted.
  Neighbouring fields handle strings; the full numeric output string is pinned; `_safe_float` has a table test.

Verify:
```
py_compile run_pipeline.py fundamentals.py tests/test_jetson_ops_2026_09.py -> PYC_OK ; bash -n -> BASH_OK
$JPY -m pytest tests/test_jetson_ops_2026_09.py tests/test_pipeline.py tests/test_fundamentals.py \
  tests/test_c26_T{1,2,3,4,6,7}.py tests/test_command_ack.py -q -p no:cacheprovider
======================== 294 passed, 1 warning in 6.38s ========================
```
(The one warning is a pre-existing pandas SettingWithCopyWarning in data_sources.py:203.) The full suite was
not run, per the brief.

## Deferred / not in my ownership
- docs/FLAGS.md:486,489 cite `setup_jetson_system.sh:154-155` (now shifted). The new `TRADER_PYBIN` env var
  needs a FLAGS.md row. I don't own FLAGS.md.
- shadow.py:875 relaunches meta_label.py with `sys.executable` and the parent env (os.environ). With the fixed
  unit that is the jetson python, and no CUDA_VISIBLE_DEVICES is inherited any more. It does not use
  _training_env, which is harmless now that the unit no longer sets ''.
- The setup script's "no RTC battery" chrony rationale (audit P2-16) is left as is. rtc0/rtc1 exist, but whether
  they are battery-backed was not verified.
- Audit items not in scope: `_check_restart_bots` backoff (P2-8), a hypersearch per-epoch keep-alive / watchdog
  margin (P2-7), bot-log size rotation from the wait loop (P2-10), `_print` being silent under systemd and SIGHUP
  exiting 0 (P2-17), and combined-mode `suspend_and_start_bot` starting split loops (P2-9, second half).
- Owner/root actions (unchanged): run the script, headless, zram off, swappiness, backup cron, Telegram and
  healthcheck keys in .env, and the 6 stop-less, zero-cost-basis crypto positions before enabling.
