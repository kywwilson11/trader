# ENGINE W9 — D13 strikes, halt → working buys, trailing-denominator table
Pre-edit copies are in w9/ (base_loop.orig.py, strategy_config.orig.py, FLAGS.orig.md). New test file: tests/test_engine_r2_strikes_halt.py (10 tests).

## LANDED
- **Item 1: D13 scope A (class A).** `base_loop.py:1738-1744`: in the `if new_scores:` branch, `for sym in self._last_llm_symbols.difference(new_scores): self._veto_strikes.pop(sym, None)`. This clears the strike of a symbol that was sent to the LLM but left out of the response. It can only remove strikes. The outage path (`{}`) and scope B are untouched.
  - Tests: `test_d13_veto_omitted_veto_does_not_liquidate` (strikes 1→0→1, no sell); `test_d13_consecutive_vetoes_still_liquidate` (1→2, sold); `test_d13_outage_keeps_strike_by_design` (1→1→2, sold, c26 D14); `test_d13_scope_b_unsent_symbol_keeps_strike` (not in candidates ⇒ 1→1→2, sold, i.e. scope B unchanged).
- **Item 2: `HALT_CANCELS_WORKING_BUYS = False`.**
  - `strategy_config.py:388-403` defines it in the house comment pattern.
  - New helper `base_loop._halt_cancel_working_buys` at `:2923-2975`, called from `_entries_allowed` at `:2992-2997` inside its own try/except, so the cancel path can never decide whether a halt blocks.
  - ON: on the first halted call, list open orders (`order_utils._list_open_orders`) and cancel those that are in this book's universe variants, have side == 'buy' and are not a stop type. This runs once per halt epoch, and an un-halted call re-arms it. The epoch counts as done only if the list call and every cancel succeeded; otherwise it retries the next cycle.
  - OFF: returns before any broker call. `test_off_halt_blocks_entries_and_leaves_orders` pins zero extra `broker.calls`, all orders 'new', and the same block info.
  - ON tests: `test_on_cancels_book_buys_once_and_rearms` (ETH buy cancelled, BTC sell stop_limit and the other book's AAPL buy untouched, once per epoch, re-arm works); `test_on_stock_book_cancels_only_its_buys`; `test_on_skips_stop_types_and_retries_failed_cancel`; `test_on_cancel_path_failure_never_unblocks_halt`; `test_halt_flag_default_off`.
  - `docs/FLAGS.md:86` has the new row.
- **Mutation check.** Against the pre-edit base_loop (w9/mut/, with a sanity test that confirms the mutant module was loaded): 4 failed, 7 passed. The 4 failures are exactly the D13 omitted-case test and the three ON-cancel tests. The OFF, scope-B, consecutive and outage tests pass both before and after, as they should since they pin unchanged behaviour.
- **Item 3.** Appended "## Trailing-denominator divergence table (W9, 2026-09-27)" to `research/campaign_2026-09_jetson/research_engine.md`. Script and output are in w9/trail_table.{py,out}. No code change.

## FOUND-NOT-FIXED
1. **Trail asymmetry (owner).** Sites:
   - Kernel, `policy_exits.py:153`: td = C(m·ATR/entry).
   - Stock, `stock_loop.py:1484`: the same, sent as a server `trailing_stop` with trail_percent rounded to 0.1 %.
   - Crypto, `base_loop.py:1076-1079`: C(m·ATR/**hwm**).
   Gap = rE·[C(m·a) − C(m·a/r)] ≥ 0, so the live crypto (HWM) stop is always tighter or equal. Inside the clamps the gap is m·ATR·(r−1). On the grid it runs from 0 bps (a=0.5 %: both stops sit on the floor) to 133 bps (r=1.2, a=4 %), and it is the same for both books. Stocks also feed the HWM form into `_book_stop_risks` (`:1225`), because StockLoop does not override it, but that is risk accounting only.
   - Tonight: nothing is affected. The six crypto positions are zero-basis, so `:1073` skips the ATR branch and neither convention applies. The live rule is a pure 5 % HWM trail (W1 item 1).
   - Resolution: changing the loop to divide by entry (`:1077`) is the only fix that keeps labels == backtest == live without a retrain. Changing the kernel to divide by the HWM changes every label, and the stock server trailing_stop cannot express it.
   - Pin: a parity test that imports both kernel and loop (allowed: `test_vertical_is_loop_layer_only` only forbids the loop files from importing the kernel).
2. **Scope of the halt-cancel hook.** `_entries_allowed` runs only inside `_execute_buys`, which runs only when `_buys_allowed` is set and, for stocks, the market is open and `flattened_today` is false (`base_loop.py:313-326` and `:406-409`, `stock_loop.py:990-999`). So a halt that coincides with a circuit-breaker stop gets its cancel deferred to the first cycle where buys are allowed. Two judgment calls, not proposed: moving the hook to the top of the cycle, and the unverified question of what Alpaca does with bracket legs when a partially filled parent is cancelled.
3. **Filtering choice (item 2).** I did NOT filter by client_order_id prefix. Stock bracket parents carry no client_order_id (`stock_loop.py:153-165`), so a prefix filter would make the flag a silent no-op for stocks. The books stay isolated because the universes are disjoint (the `_symbol_variants` rule), and the test pins that AAPL and ETH do not cross. A side effect: a manual operator buy on a universe symbol would also be cancelled.
4. **FLAGS.md line drift.** My 16-line insert shifts every `strategy_config.py:NNN` citation above :388 in FLAGS.md (38 rows). Those citations were already stale before I started: FIXED_HOLDOUT_DAYS is cited at :353, is really at :426, and was at :410 before my edit. That is a doc-owner sweep, not my one row.
5. **Test gap.** fake_alpaca_broker exposes no `type` for buy stops, so the stop-type exclusion is covered with a stub API instead. The fake was not modified.

## FLIP PROPOSAL — HALT_CANCELS_WORKING_BUYS
- **Which buys can still be working when the next cycle sees the halt.** Normal entries settle inside the cycle: maker rungs and the fallback are cancelled on timeout by `manage_order_lifecycle`, and IOC orders self-cancel. What survives is failure residue only:
  - (a) a maker GTC rung, `maker-*`, whose outcome is unknown or whose cancel did not confirm (`order_utils.py:476-500`);
  - (b) a GTC `trader-*` fallback limit whose cancel did not confirm (`:787-799`);
  - (c) any buy after a lifecycle give-up following 3 fetch errors with a best-effort cancel (`:708-725`);
  - (d) a stock 'day' bracket parent in the same states (`stock_loop.py:1278-1281`);
  - (e) a halt set while a ladder is mid-flight inside the cycle. The flag does NOT cover (e).
- **Evidence count = 0 buys filled while `trading_halt.flag` was active.**
  - The flag file is absent.
  - "trading_halt.flag active" appears 0 times across logs/trader.log* (2026-04-08..2026-09-27).
  - The 21 `/halt` lines, dated 2026-09-26/27, are test residue: no bots have run since 2026-05-07.
  - The journals (51 files, 2026-02-23..2026-05-07) have no halt rows, because a manual halt journals nothing by design.
  - The residue signatures (outcome UNKNOWN, still working after cancel, cancel not confirmed, Giving up after) appear 0 times outside the 09-26/27 test residue.
- **Instrument (measurement-only, before any flip).** Journal `halt_on` / `halt_off` rows with a timestamp. On each halted cycle, journal the count and ids of open universe BUY orders. Offline, join Alpaca closed orders whose `filled_at` falls inside a halt window.
- **Pre-registered rule.** Flip to ON if at least 1 buy fill is observed inside a halt window, or if at least 1 halted cycle is seen with an open universe BUY. Otherwise keep OFF, but review after 30 halt epochs. The rule is asymmetric because the ON path can only cancel buys, never exits.
- **Runbook phase.** At bot activation (03_jetson_runbook), next to the Telegram /halt ops check.

## VERIFIED-CLEAN
- The existing pins are unaffected: test_c26_P1:449-489 (the success path clears the strike for the scored symbol), test_ia2_safety:177-190 and test_prediction_cache_context:73.
- The test_grp_loops text pin still holds: no `except Exception: pass` in `_entries_allowed`.
- test_entries_allowed_fail_open_logs: the OFF path makes no extra attribute reads on an `object.__new__` instance.

## TEST RUNS
All runs used `hwlock.sh heavy engine-w9-<f> -- $JPY -m pytest tests/<f>.py -q -p no:cacheprovider`, with CUDA_VISIBLE_DEVICES='' and one file per process. py_compile passed on base_loop, strategy_config and the test file.

| Test file | Result |
|---|---|
| test_engine_r2_strikes_halt | 10 passed (mutant: 4 failed, 7 passed) |
| test_c26_P1 | 44 passed |
| test_ia2_safety | 24 passed |
| test_prediction_cache_context | 4 passed |
| test_engine_r1_inherited_positions | 11 passed |
| test_engine_r2_replay_harness | 9 passed, 1 xfailed |
| test_c26_base_loop_functional | 37 passed |
| test_loop_fixes_2026_09 | 22 passed, 1 skipped |
| test_review_b01 | 21 passed |
| test_grp_loops | 9 passed |
| test_engine_r2_flags | 52 passed |
| test_ia4_flagged | 43 passed |
