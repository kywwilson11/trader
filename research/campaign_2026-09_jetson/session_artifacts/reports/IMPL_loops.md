# IMPL_loops — G1 F1/F2/F4 + G2-3/G2-4 (2026-09-27)

Files edited: base_loop.py, stock_loop.py, run_bots.py; new tests/test_loop_fixes_2026_09.py.
crypto_loop.py: NOT edited (no elapsed-time/cooldown copy there; grep of now()/timedelta/total_seconds finds none).
Pre-edit snapshots: <scratchpad>/impl_loops_orig/. Full diff: <scratchpad>/reports/IMPL_loops.diff
(base_loop +26/-3, stock_loop +21/-9, run_bots +7/-2). No threshold, duration or success-path decision changed.

## Changes
1. F1 stock_loop.py:436 `_prepare_overnight_keepers`: `for symbol in list(keepers):` (discard still hits the caller's set).
2. F2 base_loop.py:446-462 `_load_models`: new `except Exception as e` after the FNF branch -> logger.error with type+msg,
   model/config/scaler_X/feature_cols reset to None/{} (fail closed: `_get_predictions` returns {} when model is None,
   exits/stops keep running), `_failed_reload=(model_reload_key(prefix), time.time())`, `model_mtime=None`, return.
   The existing `_hot_reload_check` 300 s backoff retries; its success path clears `_failed_reload`.
3. F4 run_bots.py: module `_crashed = threading.Event()`, set in `_run_loop`'s `except Exception`; `main()` returns
   `1 if _crashed.is_set() else 0`; "[BOTS] %d loop(s) running" now counts `t.is_alive()`.
4. G2-4 stock_loop.py:355 (EOD flatten) and :727 (`_execute_sells`): inline `'not found'/'404'/'no position'` checks
   replaced by `order_utils._is_not_found(e)` (function-local import, so extract-and-exec harnesses without the name in
   globals still work). :1418 (order-not-found) left as is, per the hunt report.
5. G2-3 (lockout/window math only; durations unchanged):
   - base_loop.py:2785 `_is_hard_stop_locked`: `now().timestamp() - stamp.timestamp()`.
   - base_loop.py:2729 `_save_hard_stop_lockout`: expiry = `stamp.timestamp() + HOURS*3600` (was naive
     `stamp + timedelta(hours)` -> 23 h/25 h real across DST, corrupting the restored stamp after a restart).
   - stock_loop.py:939 `_recover_external_exit` 24 h window: `.timestamp()` difference; its `_order_ts_local` helper now
     converts aware->naive-local via `fromtimestamp(ts.timestamp())` (keeps `.fold` in the repeated hour; same wall value).

## Tests — tests/test_loop_fixes_2026_09.py (23 tests, extract-and-exec, Mac-safe; notify stubbed autouse)
F1 x5 (3 failing-keeper parametrizations, success path unchanged, real flatten_before_close sells non-keeper + failed
keeper), F2 x3 (corrupt artifact -> no raise, `_failed_reload` set, backoff holds at +30 s, repaired reload at +330 s
clears it; FNF path unchanged; REAL `run()` startup: no escape, scoped cancel + reconstruct still run), F4 x2 (crash ->
rc 1 and "0 loop(s) running"; clean -> rc 0), G2-4 x5 ('position does not exist' treated as gone at both sites;
429 keeps tracking), G2-3 x7 (spring-forward still locked at 23.5 h real; fall-back unlocked at 24.5 h real; fold in the
repeated hour; off-DST boundary exact at 24 h; persisted expiry = real 24 h; external-close window both directions).
DST cases need local zone America/Chicago: in-process when already so / via tzset; the jetson conda python has NO
time.tzset, so a fallback test re-runs `-k g23` in a child with TZ=America/Chicago (here the box is already Chicago, so
they ran in-process).
Mutation check: against the pre-edit snapshots, 17/22 targeted tests FAIL (all F1, F2, F4, G2-4, G2-3 cases).

## Verify
py_compile base_loop.py stock_loop.py crypto_loop.py run_bots.py tests/test_loop_fixes_2026_09.py -> OK
pytest test_loop_fixes_2026_09 c26_base_loop_functional c26_P1 ia2_safety ia4_flagged review_b01 c26_T6 grp_loops:
  FAILED tests/test_review_b01.py::test_execute_sells_routes_external_close_through_helper
  1 failed, 242 passed, 1 skipped in 14.33s
Also green, one file per process: ia3_gate_pricing 28, base_loop_v3 26, journal_stats 20, c26_X1 25, llm_eval 9,
predict_now 11, c26_T7 37.

## Needs the test owner (NOT in my file set)
tests/test_review_b01.py:436-441 is a SOURCE-TEXT pin of the old inline copy G2-4 removes:
`for needle in ("'not found'", "'404'", "'no position'"): assert needle in seg`. Its intent ("not-found detection before
dropping") is now carried by `_is_not_found(e)`. Proposed one-line update:
    -    for needle in ("'not found'", "'404'", "'no position'"):
    -        assert needle in seg
    +    assert '_is_not_found(e)' in seg
(I did not game it with a comment listing the strings.)

## Residuals (not changed)
- Root of G2-3 is trading_utils.cooldown_ok:153 (naive subtraction) — outside my file set; G2-3.diff still applies there.
- stock_loop.py:936 `ots < entry_ref - timedelta(minutes=5)` is a naive comparison (DST-ambiguous only for fills in
  01:00-03:00 local; stocks never trade then). Left untouched.
