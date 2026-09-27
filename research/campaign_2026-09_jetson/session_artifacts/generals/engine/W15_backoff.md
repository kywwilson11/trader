# W15 — E6 crash-loop backoff + give-up latch (ENGINE R5) — class D, ops flag default OFF
## LANDED
- run_pipeline.py:1053 `BOT_RESTART_BACKOFF` <- `TRADER_BOT_RESTART_BACKOFF` ('0'; 2026-08 house parse `.strip().lower() in ('1','true','yes')`), an ops constant like `EOD_DIGEST_ENABLED` (:144).
- Pure functions, no I/O or clock reads: `crash_signature` :1063 (innermost `file:line | exception line`, digits masked, <=160 chars, 'unknown' for empty or garbage input); `next_restart_delay` :1098 (60*2^k capped at 960, k=0 -> 60); `consecutive_same` :1112 (windowed); `should_give_up` :1127 (>= 5 crashes with the latest signature inside 3600 s).
- Wiring (ON only): `_check_restart_bots` :1251. Stderr is merged into the bot log (`_start_bot` :691 `stderr=STDOUT`), so the crash tail is read from that log. It starts at the size recorded when this process was first seen (`_backoff_note_proc` :1146, `_read_crash_tail` :1158), which keeps older in-cycle tracebacks (base_loop.py:356) out. If no traceback is found the signature is `exit_rc_<code>`.
- Delay semantics: the delay is counted from the pass before the detection pass (the earliest possible crash time). A first or distinct-signature crash is therefore restarted on the detection pass, exactly as today. Repeats wait +60/+180/+420/+900 s after detection.
- Give-up (`_backoff_verdict` :1174): logs `[BOTS] giving up on <bot> after N identical crashes (<sig>)`, sends ONE `notify(level='critical', dedupe_key='bot-giveup-<bot>')` (notify.py:245, 600 s dedupe :32), then drops the dead entry. Any internal error fails open to today's immediate restart.
- Un-latch: `_backoff_release` :1229 runs from `_start_bots_now` (:914/:920/:925, the start_bot path) and the weekly `_restart_bots` (:803). It also drops a dead entry that is still waiting, so a deferred restart cannot duplicate an operator start.
- docs/FLAGS.md:157: one row in §1 env-flag table.
- tests/fixtures/april_crashloop_2026.json (47.6 KB): built by streaming stock_bot_output.log through the new `crash_signature`. It gives 1,778 crashes, 1 signature (`fundamentals.py:258 | ValueError…`), and timestamps that match the W14 census 1:1. Build script: w15/build_fixture.py.
- tests/test_engine_r5_restart_backoff.py: 28 tests.
  - April replay reproduces the acceptance numbers exactly: current 1,616 -> ON 12 restarts, 3 give-ups, 99.3 % avoided.
  - The real `_check_restart_bots` on a fake clock over all 3 April episodes: 12 restarts and 3 critical alerts ON; OFF avoids >= 99 % fewer.
  - Distinct signatures restart on the detection pass (delay = base).
  - The latch never touches a running bot. start_bot and `_restart_bots` both un-latch and reset history.
  - No duplicate process if start_bot arrives during a wait; errors fail open; the tail reader ignores output from before the process started.
  - OFF pin: with the guarded blocks stripped, the three touched functions match sha256 of the pre-edit source; behavioural check that the process restarts on every pass.
- Mutation check: against the scratch original, the 24 feature tests FAIL. The 4 OFF-path tests PASS, which confirms the pre-edit behaviour and bytes.
## FOUND-NOT-FIXED
- F1 (class A race, OFF path, existing; not in my remit). If start_bot arrives after a crash but before the next 60 s pass, `_start_single_bot` (:855) / `_start_bots_now` combined check alive only. They launch a second process, and the next pass also restarts the dead entry, giving two loops for one book (duplicate order flow). Fix: drop dead entries before launching. This is `_backoff_release(bots, names)` without the flag guard.
- F2 docs: FLAGS.md §5.1 needs a `TRADER_BOT_RESTART_BACKOFF` row and the §1 heading count "(27)" -> "(28)". I skipped both because of the one-row limit. The EOD rows at :155/:460/:41 still cite the stale `:118` (real line :144).
- F3 JUDGMENT: `should_give_up` follows the task spec (same signature in the window, not necessarily consecutive). An alternating A/B loop therefore latches after about 9 crashes; the scout's variant never did. The consecutive count is windowed (a quiet hour resets it) instead of K8s' 1800 s healthy reset. It is the same on April.
- F4 JUDGMENT: backoff alone with this ladder gives 188 restarts (88.4 % avoided; scout 169 / 89.5 %) and fails the 90 % bar. Only backoff plus give-up passes.
## FLIP PROPOSAL (owner; runbook phase = bot activation, 03_jetson_runbook)
- Pre-registered acceptance: the fixture replay (12 restarts / 99.3 % / 3 alerts) plus the OFF sha pin, both green on the Jetson full stack.
- Owner choice 1: permanent latch (shipped) vs a 3,600 s parked probe (scout replay: 148 restarts, 90.8 %). A probe needs a per-bot `probe_at` timer in `_backoff_state`, one restart per hour while latched, notify muted while parked, and an un-latch after the probe survives >= 1800 s. Risk to weigh: an Alpaca outage (5 identical ConnectionErrors) latches the bots after the outage ends.
- Owner choice 2: in combined mode, giving up on 'Bots' also takes crypto down (run_bots.py:163-175).
- Owner choice 3: configure a notify channel; today notify.py:259 returns early, so the critical alert goes nowhere.
## VERIFIED-CLEAN
- run_pipeline import is stdlib-only: 12 MB maxrss, no torch/pandas/base_loop.
- `_check_restart_bots` is called only from the trading wait loops (:1721/:1844-era call sites), never during training.
- `_stop_bots` clears `bots`, so a pending wait cannot fire during a retrain.
## TEST RUNS (each `CUDA_VISIBLE_DEVICES='' nice -n 10 $JPY -m pytest <f> -q -p no:cacheprovider`, one at a time; maxrss <= 108 MB, conftest numpy+pandas ~90 MB)
- test_engine_r5_restart_backoff 28 passed; test_pipeline 9; test_command_ack 13; test_jetson_ops_2026_09 36; test_g3_ops_2026_09 43; test_c26_T1 30 (it does not touch run_pipeline). py_compile OK.
## DEFERRED to HW: resume
- tests/test_gate_protocol.py: TestCheckRestartBotsScope pins `_check_restart_bots` and imports backtest+pandas. Expected green because the OFF path is byte-identical.
- tests/test_c26_T4.py (backtest), test_c26_Q2.py, test_c26_X1.py, test_c26_T7.py, test_imports.py.
- `bash scripts/ab_check.sh`.

## ADDENDUM (follow-up: F1 landed, FLAGS §5.1)
- Line shift: every run_pipeline.py cite above from :1053 onward is now +21. Flag :1074, crash_signature :1084, _backoff_verdict :1195, _backoff_release :1250, _check_restart_bots :1265; _restart_bots release :823, _start_bots_now releases :935/:941/:946.
- F1 LANDED (class A, flag-independent). New `_drop_dead_entries(bots, names)` at run_pipeline.py:758 closes, untracks and pops exited entries of the books about to launch.
  - Called in `_launch_bots` before each launch (:784 'Bots', :800 'Crypto', :810 'Stock'). That covers combined start_bot via `_start_bots_now` and the weekly `_restart_bots`.
  - Called in `_start_single_bot` (:880, split start_bot) only when it will actually launch, i.e. after its alive check.
  - `_backoff_release` now reuses it; behaviour is unchanged.
- The sha pin was not changed and still passes. The three pinned functions (`_check_restart_bots`, `_start_bots_now`, `_restart_bots`) did not change, because the fix lives in `_launch_bots` and `_start_single_bot`, which are deliberately unpinned. Their OFF path changes by exactly this defect fix.
- Test `test_start_inside_crash_window_leaves_one_live_process[{split,combined,weekly_restart}-{flag OFF,ON}]` (6 cases). Setup: crash, then start_bot or `_restart_bots` before the monitor runs, then 5 passes on a fake clock with fake Popen. It asserts exactly ONE live process for the book, one launch, and the dead handle closed.
- Mutation check against the scratch original: all 6 FAIL with `assert 2 == 1` (two live processes for one book, the defect reproduced). After the fix they pass. The 4 OFF-path pins still PASS on the original.
- FLAGS.md: added the §5.1 `TRADER_BOT_RESTART_BACKOFF` row; bumped the §1 heading from (27) to (28); updated the §1 row cite to :1074. Left the stale EOD :118 cites alone.
- TEST RUNS (light, one at a time, maxrss <= 108 MB):
  - test_engine_r5_restart_backoff: 34 passed
  - test_pipeline: 9
  - test_command_ack: 13
  - test_jetson_ops_2026_09: 36
  - test_g3_ops_2026_09: 43
  - py_compile OK
- Still DEFERRED to HW: test_gate_protocol.py (it calls `_check_restart_bots` and the three launch paths indirectly), test_c26_T4/Q2/X1/T7, test_imports, and `bash scripts/ab_check.sh`.
