# IMPL_exec: G2-1, G2-3 and F3 implemented (2026-09-27)

Files touched: order_utils.py, trading_utils.py, llm_analyst.py, tests/test_c26_T6.py (fake-broker only),
and a new file tests/test_exec_fixes_2026_09.py. There were no pre-existing working-tree changes in any of
them. No git operations were run, no orders were placed, no LLM calls were made, and CUDA_VISIBLE_DEVICES=''
was set throughout.

## Diff summary

### G2-1: order_utils.py (+45)
- There is a new module constant `_SETTLED_STATUSES = ('filled','canceled','expired','rejected')`, placed just
  above `_maker_rung_id`.
- **`manage_order_lifecycle`**
  - A new `cancel_failed` flag records whether the timeout `cancel_order` raised.
  - A new guard runs before the fallback, when `fallback_to_market and cancel_on_timeout`.
  - If the post-cancel fetch shows a non-settled status (the cancel raised, the order is still
    `pending_cancel`, or it is `new`/`partially_filled`), it logs an ERROR and returns the fetched order. It does
    not send a market or IOC fallback. The caller judges the result by filled_qty, as it already does.
  - **Beyond the hunter's diff:** if the cancel raised AND the post-cancel fetch also failed, it returns None
    with no fallback. This follows D18's rule that an unknown outcome is not a zero fill. The task said "never
    send a second order while the first may still be working", and this is that case.
- **`place_maker_buy`**
  - A new check sits right after the existing `result is None` abort.
  - When the rung's status is not settled, it keeps the result as `last` if the result has more fill, journals
    `maker_unknown`, and returns `(last, 'maker_unknown')`.
  - So no next rung is sent and no taker fallback is stacked.

### G2-1 test fix: tests/test_c26_T6.py (+4)
- The fake `_API.cancel_order` now sets status to `'canceled'` unless the order is already `'filled'`, as the real
  broker does.
- No assertion was changed.

### G2-3: trading_utils.cooldown_ok (+5/-1)
- It now computes `elapsed = now().timestamp() - last.timestamp()`.
- This honours `.fold` and works for tz-aware stamps too. Off a DST boundary the duration is unchanged.

### F3: llm_analyst.py
- It now does `import threading`, and there is a module-level `_ANALYSIS_LOCK = threading.Lock()`.
- The whole load → modify → write in `_save_analysis` runs under that lock. The body was re-indented; its
  logic is unchanged.
- The tmp name is now per writer: `llm_analysis.json.{pid}.{thread_ident}.tmp`, installed with `os.replace`.
- If the write fails, the per-writer tmp is unlinked and the function returns.
- After a successful replace, a legacy fixed-name `llm_analysis.json.tmp` is swept, best-effort.
  - No writer uses that name any more, so a copy left by a crash under the old code is garbage.
  - This keeps `test_llm_analyst.py::TestSaveAnalysisAtomicWrite::test_survives_garbage_preexisting_tmp_file`
    green without editing that test. That file is not mine.
- Section semantics are preserved: `{crypto:{...}, stock:{...}}`, each save touches only its own `asset_type`
  section, and the record keys are unchanged.
- The cross-process lost update (the GUI refresh subprocess) is still accepted, as the existing comment says.
- `*.tmp` is already gitignored (.gitignore:153).

## New tests: tests/test_exec_fixes_2026_09.py (21 tests, Mac-safe, stub broker)
- **G2-1**
  - Uses the hunter's fake broker, with cancel modes `raise` and `pending_cancel`.
  - Lifecycle: no market fallback, exactly one live order, and the fetched order is returned, including partial
    fill evidence.
  - Cancel raised and the fetch failed: returns None with no fallback.
  - Maker ladder: `maker_unknown`, one submit, one live order, and the partial fill is kept as best evidence.
  - Happy path (the cancel settles the order): the lifecycle market fallback is still sent for the full qty, or
    only the remainder after a partial fill. The ladder still reprices and then falls back.
- **G2-3**
  - The DST cases run in a child interpreter with `TZ=America/Chicago`, because the Jetson conda py3.10 has no
    `time.tzset`.
  - Spring-forward: 15 real minutes gives cooldown_ok 15m/30m/60m = T/F/F.
  - Fall-back: 40 real minutes gives 30m/60m = T/F.
  - The `fromtimestamp` restore path keeps fold=1.
  - Off-DST behaviour is unchanged, and aware stamps work.
- **F3**
  - The hunter's 300-iteration two-thread stress, plus a concurrent reader and a per-round check. The result: 0
    replace failures, 0 torn files (neither seen by the reader nor left installed), both sections present every
    round, and no tmp left behind.
  - Also covered: section preservation, the per-writer tmp name, and that a failed replace cleans up its tmp.
- **Discrimination check:** the same test file was run against the HEAD versions of the 3 modules, in the
  scratchpad copy `impl_orig/`. **14 of the 21 tests fail** there, which covers every fix-specific test.

## Verification (Jetson env, one process at a time, -q -p no:cacheprovider)
py_compile of all 5 files: OK.
```
test_exec_fixes_2026_09.py   21 passed
test_order_utils.py          16 passed
test_c26_T6.py               43 passed
test_c26_T7.py               37 passed
test_review_b02.py           27 passed
test_trading_utils.py         9 passed
test_llm_analyst.py          28 passed   (1 failure before the legacy-tmp sweep was added; fixed in code, not test)
test_llm_advisor.py          65 passed
test_c26_V1.py               34 passed
extra: test_c26_P1 44, test_llm_dossier_persist 11, test_c26_X1 25, test_fault_injection 13, test_grp_exec 7,
       test_ioc_helper 9, test_order_utils_v3 40, test_new_modules 38, test_execution_policy_v3 33 — all passed
```
The full suite was not run.

## Notes for the orchestrator
- **Behaviour change for sell callers** (crypto_loop/stock_loop `place_sell_order` and the stock flatten): an
  unconfirmed cancel now returns the still-working order, or None. The caller treats that as a failed sell and
  retries next cycle, instead of sending a second sell.
- Stock bracket entries are unaffected (they pass `fallback_to_market=False`).
- **Siblings not fixed, because they are outside my files:**
  - G2-3's naive subtraction also appears at base_loop.py:2762 (hard_stop_lockout) and stock_loop.py:929.
  - The fixed-tmp pattern also appears in volatility._har_rrv_save and llm_config.save_llm_config (G1 J16).
- G2-2 and G2-5 were not in scope and were not implemented.
