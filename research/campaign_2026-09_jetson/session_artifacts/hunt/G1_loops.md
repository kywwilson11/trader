# G1 — live decision engine (base_loop.py, crypto_loop.py, stock_loop.py, run_bots.py)

Hunt agent G1, 2026-09-26, on the Jetson. This was read-only: no repo file was edited, no order was placed, and no bot was started.
All repros are extract-and-exec. They pull the REAL method source out of the repo file by AST and run it against stubs, so no heavy import is needed. They live in
`<scratchpad>/hunt/G1/` (helper: `_extract.py`). Run them with `source <scratchpad>/jenv.sh; CUDA_VISIBLE_DEVICES='' $JPY <file>`.
Each process stays under 100 MB RSS and runs for 10 s or less. The one exception is r3, which does 300 iterations in about 20 s.

Already known or already queued, so NOT re-reported here:
- D13 veto strikes (FIX_G).
- The partial-fill exit handling in `_execute_stop_exit` (module_review_2026-07 P2).
- The keeper cancel-failure branch (P2).
- The server-stop lockout on trailing fills (P2).
- `avg_entry_price=0` → immediate 5% trail (HARNESS).
- SIGTERM not cancelling resting orders (E_ops/HARNESS).
- Combined-mode log lines carry no book tag (HARNESS).
- `_load_models` catching only FileNotFoundError: B_serving §3 noticed this. The combined-process consequence below (F2) is new.

---

## Findings (ranked by severity)

### F1 · stock_loop.py:431 (+ :456) · class A · `_prepare_overnight_keepers` mutates the set it is iterating, so ONE failed GTC-stop placement raises RuntimeError out of `flatten_before_close` and aborts the EOD flatten and the whole cycle

**Defect.**
- The loop is `for symbol in keepers:`, and its except branch does `keepers.discard(symbol)` (:456).
- CPython raises `RuntimeError: Set changed size during iteration` on the next `next()`. That happens even when the failing keeper is the last element.
- The raise sits outside the try, so it propagates:
  - through `flatten_before_close` (:323), before the orphan sweep and before any non-keeper is sold;
  - into `_run_one_cycle`, where `flatten_before_close()` runs BEFORE `_manage_stops()` (base_loop.py:334 vs :347);
  - up to `run()`'s per-cycle `except`.
- So the "— flattening instead" path the log line promises never executes.
- Every cycle in the 10-minute window repeats this while the failure persists. Consequences:
  - Software stops never run.
  - Nothing is flattened.
  - At 16:00 every position rides overnight. The non-keepers' day-TIF bracket legs expire at the close.
  - The failing keeper's legs were already cancelled, so it is naked.
- This path is live today: `OVERNIGHT_SLEEVE_ENABLED = True` (strategy_config.py:121, docs/FLAGS.md "LIVE"), with MAX_POSITIONS=2.
- No test runs the real method. tests/test_review_b01.py:278 stubs it out.

**Proof.** `hunt/G1/r1_keepers.py`, run against the real source:
```
keepers=['AAA'] failing=['AAA']            -> RuntimeError ESCAPED: Set changed size during iteration
keepers=['AAA','BBB'] failing=both         -> RuntimeError ESCAPED
keepers=['AAA','BBB'] failing=[]           -> returned normally
== flatten_before_close, keeper AAA (GTC stop fails) + non-keeper CCC ==
  RuntimeError ESCAPED flatten_before_close: Set changed size during iteration
  orders submitted for flatten: [] | still tracked: ['AAA','CCC'] | flattened_today: False
```
`hunt/G1/r1_keepers_fixed.py` runs the same harness with the one-line fix. Results:
- keepers become `[]` after the failures;
- `flatten_before_close` sells CCC ("[FLATTEN] CCC: Sold 10 shares") and attempts AAA;
- flattened_today stays False only because the stub also rejects AAA's sell.

**Fix.**
```diff
-        for symbol in keepers:
+        for symbol in list(keepers):
```
- When there is no failure, the iteration order is bit-identical: `list(set)` preserves set order.
- The discard still reaches the caller's set, which is the documented intent.

**Blast radius.**
- The only caller is `flatten_before_close` (:323).
- No test exercises the real body.
- NOTE: the queued module_review P2 fix for the cancel-failure branch ("keepers.discard on cancel failure") would hit this same RuntimeError unless F1 lands first.

**Why indisputable.** Mutating a set while iterating it is a guaranteed CPython RuntimeError. The fix changes no decision on the success path and makes the failure path do what its own log line says.

### F2 · base_loop.py:437-445 (`_load_models`) + run_bots.py:160-171 · class A · a corrupt or mismatched model artifact in ONE book crashes the combined process and takes the healthy book down with it, in a 60 s crash-restart loop

**Defect.**
- `_load_models` catches only `FileNotFoundError`. It is called outside `run()`'s per-cycle try (:277).
- A truncated or shape-mismatched `.pth`/`.pkl` makes torch/joblib raise `RuntimeError`/`UnpicklingError`/`EOFError`. That exception escapes `run()`.
- In combined mode (the production systemd unit passes `--combined-bots`), the sequence is:
  1. `_run_loop` logs the exception.
  2. Its `finally` sets `_shutdown`.
  3. `main()` returns (rc=0, see F4).
  4. The OTHER book's daemon thread is killed wherever it is.
  5. run_pipeline `_check_restart_bots` restarts the process every 60 s (run_pipeline.py:1579/1705). On every restart the healthy book runs `cancel_all_open_orders` again, which cancels its resting GTC stops. If the kill lands mid-startup, that happens before `_replace_protective_stops` has run.
- The failing book's own documented contract is violated. The FNF branch says "Buys are DISABLED … (fail closed); exits and stops still run". `_hot_reload_check` already treats any load exception as fail-closed with a 300 s retry (:919-921).
- The two copies of this load logic have diverged.

**Proof.** `hunt/G1/r2_startup_crash.py` uses the REAL `run_bots.main` and the REAL `_load_models` source. The loop classes are stubs, and the stock loader raises a state_dict size mismatch.
```
t=0.0s crypto: cancel_all_open_orders(universe) -> resting GTC stops CANCELLED
t=5.0s stock: _load_models() ...
[run_bots] ERROR: [stock] Trading loop crashed ... RuntimeError: Error(s) in loading state_dict ...
[run_bots] INFO: [BOTS] 2 loop(s) running in one process      <- logged AFTER the crash
t=10.0s run_bots.main() returned rc=0; process would now exit (daemon threads die)
crypto startup completed before exit? False
```
- The t=10 s exit is deterministic: `_shutdown` is already set when main's loop starts.
- Whether the crypto thread has already re-placed its stops by then depends on its startup time: model load, a scoped cancel with up to 5 s await, then an ATR fetch per position.

**Fix.** Fail closed and hand the retry to the existing backoff:
```diff
         except FileNotFoundError:
             logger.warning("Model files not found. Buys are DISABLED until a model exists "
                            "(fail closed); exits and stops still run.")
+        except Exception as e:
+            logger.error("Model load FAILED (%s) — buys DISABLED (fail closed); exits and "
+                         "stops still run; retrying via hot-reload backoff", e)
+            from trading_utils import model_reload_key
+            self._failed_reload = (model_reload_key(self.MODEL_PREFIX), time.time())
+            self.model_mtime = None    # != key -> _hot_reload_check retries every >=300 s
+            return
```
`hunt/G1/r2_fixed.py` exercises the patched source together with the REAL `_hot_reload_check`:
- startup does not raise, and model=None;
- at t+30 s there is no reload attempt (backoff holds);
- at t+330 s, once the artifact is repaired, it reloads (`model=MODEL thr=0.2`).

**Blast radius.**
- Callers: `run()` only.
- No test pins `_load_models` (grep of tests/: only test_predict_now tests `predict_now.load_models`).
- Split-process mode also improves: a crash-loop becomes a fail-closed book that retries.

**Why indisputable.**
- The failure path already exists and is documented (fail closed; exits and stops run), and the sibling hot-reload path already implements it.
- This removes a crash that couples an unrelated book's liveness to another book's artifact.
- No success-path behaviour changes.

### F3 · llm_analyst.py:753 (cross-group; reached from base_loop.py:1689 in BOTH loop threads) · class D · `llm_analysis.json` is written through ONE fixed tmp name with no lock, so in combined mode two loop threads can install a torn file

**Defect.**
- `base_loop._run_llm_analysis` calls `analyze_trades(...)` with `persist=True` by default (llm_analyst.py:441), which calls `_save_analysis`.
- That function writes `llm_analysis.json.tmp` and then calls `os.replace`.
- In combined mode, the crypto and stock threads can both be inside `_save_analysis` at once. Both `open(tmp,'w')` the SAME inode (O_TRUNC under the other writer), so:
  - interleaved bytes get `os.replace`d in as `llm_analysis.json`;
  - the second replace raises FileNotFoundError, which is caught and printed, so that book's analysis is silently not persisted.
- The code comment claims the atomic write guarantees "never a half-written/corrupt one". That holds only for distinct writers.
- Downstream effects:
  - `load_analysis()` returns `{}` on the torn file.
  - The next `_save_analysis` then rebuilds from `{}` and wipes the other book's section.
  - `_print_startup`'s fresh-score preload is lost.
  - `_check_llm_staleness` sees stale timestamps and forces an extra, paid refresh.
  - The GUI goes blank.
- base_loop already fixed this exact pattern for its lockout file (per-book tmp, base_loop.py:2719). So did risk_budget (:378) and events_calendar (:162).

**Proof.** `hunt/G1/r3_llm_analysis_tmp.py` runs the REAL `_save_analysis`/`load_analysis` source against a scratch path, with two threads released together by a barrier and 300 iterations:
```
replace failures (FileNotFoundError on shared tmp): 291
torn llm_analysis.json observed by a concurrent reader: 80   ('Extra data: line 544 ...')
part2: after 300 simultaneous save pairs -> torn file LEFT INSTALLED 28x; parseable-but-missing-a-book 3x
```
`hunt/G1/r3_fixed.py` swaps in a per-writer tmp name:
```
replace failures: 0 | torn observed: 0 | torn LEFT INSTALLED 0x | missing-a-book 1x (lost update, see below)
```
The forced-simultaneity harness shows the mechanism. In production a collision needs both books' 600 s analyses to land within a few ms, so it is low-probability per event but unbounded over weeks. The GUI refresh subprocess writes the same tmp name as a third writer.

**Fix (in llm_analyst.py — owner of that file, not G1).**
```diff
-    tmp_path = _ANALYSIS_FILE.with_name(_ANALYSIS_FILE.name + ".tmp")
+    tmp_path = _ANALYSIS_FILE.with_name(
+        f"{_ANALYSIS_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
```
Add `import threading`. Optionally, also wrap the whole load→modify→write in a module `threading.Lock()` to close the in-process lost update, which the file's comment already accepts only for the cross-process case.

**Blast radius.**
- Only `_save_analysis`.
- Readers are unchanged.
- The tmp name is not referenced anywhere else (grep).

**Why indisputable.** This is the textbook atomic-write precondition (a unique tmp per writer). The repo already applies it at three other sites. It changes no content, only prevents torn installs.

### F4 · run_bots.py:160-171, 213-223 · class D · the combined runner exits 0 after a trading loop crashed (and logs "2 loop(s) running" after one already died)

**Defect.**
- `_run_loop` swallows the loop's exception (logged) and sets `_shutdown`.
- `main()` then returns 0 unconditionally. run_pipeline reports "Bots bot crashed (exit 0)".
- Any supervisor keyed on exit status sees a clean exit. For example, `Restart=on-failure` would not restart a unit that ran `run_bots.py` directly.
- The "[BOTS] %d loop(s) running" line is printed even when a thread has already exited.

**Proof.** The r2 output above shows `rc=0` after the crash traceback, and "[BOTS] 2 loop(s) running" logged 5 s after "[stock] Loop thread exiting".

**Fix.** Add `_crashed = threading.Event()`. Call `_crashed.set()` in `_run_loop`'s `except Exception` branch. In main, use `return 1 if _crashed.is_set() else 0`. Optionally use `sum(t.is_alive() for t in threads)` in the running log line.

**Blast radius.**
- `run_pipeline._check_restart_bots` restarts on any exit, so behaviour there is unchanged apart from the message text.
- tests/test_c26_T7.py imports run_bots but does not pin rc.

**Why indisputable.** A process whose worker crashed must not report success, and this cannot change a trading decision.

---

## Requested checks — results

- **30 s hot path, disk re-reads.** Measured on this device (see J9). Every per-cycle file read in these four modules is ≤ ~100 µs. None is material on the Jetson, so no class-C finding. The real per-cycle costs are network calls (J7, J8), which cannot be deduplicated bit-identically.
- **Does one thread's exception take the other book down?**
  - A per-cycle exception does not: `run()` catches it at base_loop.py:290.
  - A startup exception does (F2). By design, run_bots stops the whole process when one loop dies (J5).
  - The only startup call that can raise is `_load_models`. `cancel_all_open_orders`, `reconstruct_positions`, `_update_equity`, `_replace_protective_stops` and `_print_startup` are all guarded.
- **Position rebuild at startup.** Correct for the persisted fields. Two stale-state issues are in the appendix (J2, J11).
- **Hard-stop lockout persistence.** Clean:
  - per-book file with a per-book tmp;
  - atomic;
  - expiry stored as an absolute timestamp and restored against the class's own hours;
  - naive-local on both sides;
  - the legacy migration read is harmless (the other book's symbols are never queried).
- **SIGTERM path.**
  - The handler only sets `_shutdown`, and main returns within ≤5 s.
  - The interpreter then joins the (non-daemon) ThreadPoolExecutor workers before freezing the daemon loop threads.
  - State files are atomic, so none tears.
  - In-flight GTC crypto entries survive (J4).
- **Thread safety across the loop threads and the ops thread.** Surveyed every module-level shared writer reached from both loops. Locked or per-writer-tmp (OK):
  - trade_memory (thread + flock)
  - novelty
  - funding
  - events_calendar
  - risk_budget
  - market_data bar cache
  - monitor_drift (per-prefix flock)
  - llm_client cost (FIX_G locks)
  - notify
  - trade_journal (append + rotate lock)
  - the per-book position/lockout/prediction-cache files

  Shared fixed-tmp writers:
  - llm_analyst (F3);
  - volatility `_har_rrv_save` and llm_config `save_llm_config`, both flag-off or one-shot (J16).

  predict_now's `_panel_features`/`_lgb_models` are keyed by symbol or prefix, so there is no cross-book bleed.
- **The 8 mirrored stop-arithmetic copies.** The guard and clamp expressions were compared at every site:
  - base :540, :1048, :2337, :3238;
  - crypto :177;
  - stock :441, :1245, :1596.

  All are `entry_atr is not None and price > 0` → `max(FLOOR, min(CEIL, atr*M/price))`, with fallback `STOP_LOSS_PCT`. The argument order is identical, so NaN also resolves identically (to CEIL). No pair that must agree already differs.
  - Macro `stop_mult` is applied only at the software stop (:1060) and the stock bracket (:1253). That split is consistent with "backstop vs enforced" and not a must-agree pair.
  - The trailing-denominator asymmetry is the known OWNER item and is not proposed.
  - The only diverged copy is the measurement-only 9th copy at stock_loop.py:694 (J10).

---

## Judgment calls (not proposed)

- **J1 · base_loop.py:3267-3290 (crypto add-on buy).**
  - What happens: `_place_and_track_buy` REPLACES the Position on an add. The old resting GTC stop's id is dropped, while that order still reserves the old qty. `_after_entry_protection` then submits a stop for the full qty, which Alpaca crypto should reject (the logs show "insufficient balance … available: 0" when qty is reserved).
  - It self-heals next cycle: `_maybe_update_resting_stop` cancels and re-places.
  - But if the old stop fills inside that window, its fill is never detected: qty goes stale and exits retry "insufficient qty" forever.
  - Not proposed because it needs a paper-account order test to prove the rejection. Fix idea: cancel the symbol's resting orders before `_after_entry_protection` on an add.
- **J2 · base_loop.py:553 + stock_loop.py:1592-1617.**
  - What happens: the restart restores `trailing_activated=True` for stocks, but `cancel_all_open_orders` has already cancelled the native trailing stop and `_replace_protective_stops` places a PLAIN stop without resetting the flag. (`_prepare_overnight_keepers` :452 does reset it.)
  - Effects: no re-upgrade to trailing after a restart; `_classify_server_stop` labels the plain stop 'trail' (wrong journal kind, and wrong lockout under STOP_CLASSIFY_V2); the GUI shows trailing=True.
  - Not proposed because resetting the flag changes server-side stop levels (a re-upgrade can LOOSEN versus the HWM-anchored plain stop), which is arguable.
- **J3 · base_loop.py:1469.** `_execute_stop_exit` uses `manage_order_lifecycle` with the default `cancel_on_timeout=True`, so an unfilled market stop-exit is cancelled at 30 s. The order_utils.py:616 docstring says confirm-only mode is "for liquidation orders (emergency flatten, stop exits)". D19 was applied only to emergency_flatten. Owner decision (execution behaviour); the docstring drift should at least be corrected.
- **J4 · run_bots SIGTERM.** Crypto entry orders are GTC (maker ladder order_utils:439; marketable :533). A SIGTERM mid-entry leaves a working buy that can fill during a 10-20 h weekly retrain with no resting stop until restart. Cancelling open BUY orders on shutdown is a design choice.
- **J5 · run_bots.py:171.** One loop's death deliberately stops both books ("should surface, not hide"). Keeping the surviving book alive is a design choice. F2 removes the main trigger.
- **J6 · base_loop.py:950-966.** The stablecoin emergency-flatten copy of the flatten/failure-tracking block has diverged from the breaker (:738-764) and remote-flatten (:2818-2846) copies. It records no `record_trade` rows and does not save state. The rows feed LLM "lessons" (trade_memory.get_lesson_summary), so adding them is prompt-facing. A shared helper is the right shape, but that is an owner call.
- **J7 · stock_loop.py:1462 + base_loop.py:1276.** Each held stock gets two REST quotes per cycle: the stock pass, then base `_manage_stops`. Sharing one quote drops an HWM sample, so it is not bit-identical.
- **J8 · base_loop.py:2992.** Crypto `_execute_buys` fetches a quote (REST) for every symbol with a pred BEFORE the threshold check, which means 6 quote calls per cycle even when nothing clears the threshold. Reordering changes skip attribution (cost_floor versus below_threshold rows).
- **J9 · hot-path file reads (measured here).**
  - `load_llm_config()` costs 71.7 µs per call and runs on EVERY `log_decision` row (trade_journal.py:88).
  - `load_stock_universe()` costs 95.5 µs per call, about 1-2 calls per stock cycle.
  - `notify.halt_active()` costs 3.5 µs.
  - Total is under 10 ms per 30 s cycle, which is immaterial. The universe re-read is intentional (GUI edits it live). Not proposed.
- **J10 · stock_loop.py:693.** The near-miss would-be stop copy guards with `if atr:` where every other copy uses `is not None`. At ATR==0 it journals `STOP_LOSS_PCT` instead of FLOOR. This is measurement-only and trivially rare.
- **J11 · crypto_loop.py:214 / stock_loop.py:1602.** Restart re-placement anchors at `max(entry, hwm)*(1-stop_dist)` even when the trail is not active. The resting or server stop therefore sits above the software hard stop by up to `trail_activate_pct`, and `_classify_server_stop` then says 'trail'. Exit-level change; owner.
- **J12 · base_loop.py:537-543 vs stock_loop.py:1245-1250.** When ATR is unavailable, the stock bracket sets TP at `TAKE_PROFIT_CEIL_PCT`, while the base reconstruct and `_place_and_track_buy` set TP=None. A restart can therefore drop a stock position's software TP. Exit rule; owner.
- **J13 · base_loop.py:2790-2808.** The two threads' shared-flag fan-out can re-create a per-book flag after that book consumed it, so the book flattens again one cycle later (already flat and halted, so benign).
- **J14 · base_loop.py:2650.** The ENB book-risk cap is wrapped in `except Exception: pass`. It fails open silently with no log. A log line would be visibility-only, but no realistic raise path was found.
- **J15 · stock_loop.py:1498-1500, :810-816.** A failed trailing upgrade, or a failed signal sell after the legs were cancelled, leaves NO server-side stop for the rest of the day (software stops only). Owner.
- **J16.** Other fixed-tmp writers reachable from both loop threads have the same pattern as F3:
  - `volatility._har_rrv_save` (:320), only when the D30 HAR feed flag is ON (default OFF);
  - `llm_config.save_llm_config`, called from `load_llm_config` only on a one-time key migration.
- **J17 · stock_loop.py:1626.** `_get_current_exposure` excludes any symbol containing "USD". This is latent: no current stock ticker contains it.

## No issues found (coverage)

- **base_loop.py:**
  - `run()` per-cycle containment;
  - flatten-request consumption and staleness;
  - circuit-breaker fail-closed paths and failure-set normalization (all 3 sites);
  - `_desired_stop_for`;
  - `_classify_server_stop`;
  - hot-reload backoff;
  - prediction fan-out timeout and rate-limited pool rebuild;
  - LLM TTL expiry, backoff and staleness rate-limit (D13 excluded);
  - position-state save (atomic, deduped) and restore (tz-consistent cooldowns, peak restore ordering);
  - lockout load/save;
  - trade-budget roll;
  - `_entries_allowed`;
  - every container iteration that mutates uses `list(...)` (checked all 7 bare iterations: none mutate);
  - conviction/journal helpers never raise.
- **crypto_loop.py:**
  - resting-stop placement, ratchet and prune;
  - `place_sell_order` confirm-on-fill;
  - atomic prediction-cache write;
  - `_stop_distance_for` agrees with the other copies.
- **stock_loop.py:**
  - clock cache and early-close handling;
  - orphan sweep;
  - get_position error classification (not-found vs transient);
  - bracket leg capture;
  - partial-parent tracking;
  - external-close recovery time filter;
  - near-miss dedup bound;
  - RR-tiebreak journaling;
  - exposure re-check after fill.
- **run_bots.py:**
  - env setdefaults before imports;
  - SIGTERM handler installed in the main thread;
  - ops-thread per-op isolation and pipeline-liveness deferral;
  - lazy loop imports (13 MB for `import run_bots`).
