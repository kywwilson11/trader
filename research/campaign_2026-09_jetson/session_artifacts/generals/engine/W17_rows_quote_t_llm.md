# ENGINE r5 W17: stock buy-row parity, quote_t (STOPPED), SCOUT_E ENGINE half. Sandbox generals/engine/w17/ (orig/, preload/, ab/, quote_t/)
LANDED. All changes are additive and measurement-only. base_loop was written atomically (py_compile, then os.replace).
- R5-a, stock_loop.py:1387-1399. Before `log_decision(buy_rec)`, it does a local import of base_loop's `_order_journal_ids(result)` and `_decision_quote_journal_keys(quote)`, then setdefault. This is wrapped in try/except, so a failing helper drops the keys but keeps the row. `result` is the acquired bracket PARENT; it is never None in that branch. client_order_id stays OUT.
  - Scratch A/B vs orig/stock_loop.py on the r2 stock RTH replay (ab/pre.json vs post.json): journal equal with the new keys stripped (144 rows, value AND key order); trades 3, events 33, calls 383 and state 84 identical. Each of the AAA/BBB/DDD/EEE buy rows has exactly the 4 keys appended after the legacy keys.
- SCOUT_E, ENGINE half (base_loop._run_llm_analysis):
  - :1877-1903. analyze_trades is wrapped. An exception now writes `{'action':'llm_error', asset_type, outcome:'exception', error_type:<class name only>, n_symbols_sent, latency_ms}` and is then RE-RAISED unchanged: no state change, and run()'s handling is identical. BaseException/KeyboardInterrupt is not caught. A journal failure can never mask the original exception.
  - :1984-1989. The existing `llm_backoff` row gains `outcome:'no_scores'`, `n_symbols_sent` and `latency_ms` after its 4 legacy keys, which are unchanged in value and order.
  - :1936-1961. Each llm_analysis `scores[sym]` gains `s_defaulted` after s/pred. It is filled from `get_last_analysis_meta()['parse_flags'][sym]['s_defaulted']` (INTEL W15's meeting point) only when that value is a bool, and is None otherwise (dedup hit, no flags, meta failure). `s` and the veto-strike logic are untouched.
  - Latency uses time.time(), not monotonic, because test_engine_r2_strikes_halt injects a `time` namespace that has no monotonic.
  - The false D33 comment is reworded to the truth: `_parse_response` writes 0.5, and an omitted symbol is simply absent. No behaviour change. test_c26_P1's `s is None` pin still holds, since it only applies when a result dict lacks `s`.
- The A1 fields that only llm_analyst can fill (requested_model, model_used, path, finish class, response_chars, max_tokens, prompt_sha256 of the failed attempt) are NOT duplicated. They already exist in INTEL's `journals/llm_calls` row, which joins on ts+asset_type. `_LAST_CALL_META` is stale on failure, so none of it is copied. The A2 finish/block reason is still null until INTEL's llm_client accessor lands (INTEL's half).
- trade_journal.py:14-35. Schema docstring only: buy-row r5 note, llm_analysis `s_defaulted` plus the optional meta keys, and the new `llm_backoff` and `llm_error` entries.
- New tests/test_engine_r5_rows_llm.py (16 tests). Coverage:
  - stock buy rows: order_id is the filled bracket parent, bid/ask/mid match, fetch ts is inside RTH;
  - in-process stub A/B: legacy keys byte-identical, key order kept, identical broker traffic, trades and state;
  - a raising helper never breaks the row;
  - a source pin that both producers share the helpers;
  - the llm_error row plus re-raise with identical state;
  - a failing journal cannot mask the exception, and KeyboardInterrupt is not journaled;
  - backoff row key order;
  - s_defaulted in 5 flag shapes, plus the meta-exception case;
  - a comment/meeting-point pin;
  - READERS UNAFFECTED: llm_eval.run_eval, decision_report.load_journal/admitted_k, journal_stats load_trades/compute_stats/format_summary/eod_digest and execution_report.run_report give identical output with and without the new rows/keys. The test is non-vacuous: 3 scored rows, 2 calls, 1 backoff, 1 trade.
- Mutation check (`-p w17_orig` preloads orig base_loop+stock_loop): 12 FAIL for the right reasons, 4 PASS. The 4 are pins that hold on both versions: never-break, never-mask, KeyboardInterrupt, readers.
FOUND-NOT-FIXED
- ITEM 2 (quote_t) STOPPED: existing tests pin the exact dict shapes.
  - Proof: scratch order_utils with `'quote_t'` (diff in w17/quote_t/order_utils.diff) run against the pins → tests/test_engine_r3_o3_quote.py 14 FAILED at :359 `_assert_legacy_quote`: `out == _EXPECTED` after popping only fetched_ts. The other 7 get_quote pin files pass (quote_t/runs.txt).
  - `decision_quote_t` would likewise break test_engine_r4_journal_flatten.py:92-101 (exact `none`/expected dicts) and :193 (`set(b) == set(a) | NEW_ALL`).
  - No production consumer serialises or compares the whole quote dict (grep for `**quote`, `.items()`, `json.dumps(quote`, `==`: none).
  - To land it: (a) the order_utils.diff (qt.timestamp() in its own try, after the staleness gate, so the verdict cannot change); (b) `_decision_quote_journal_keys`: add `'decision_quote_t': None` plus `out['decision_quote_t'] = quote.get('quote_t')`; (c) tests: r3_o3:357 add `out.pop('quote_t', None)`; r4 add 'decision_quote_t' to NEW_BUY and to the :92/:96 dicts; r5 (mine) NEW_BUY.
- Owner item: an exception escaping analyze_trades (e.g. INTEL's OverflowError finding) still aborts the rest of the cycle (sells, veto sells, buys) through run()'s handler. It does not fail open, and fail count/backoff are not incremented. Fail-open (treat it as no_scores) changes a decision, so it is left unimplemented; the row now makes it countable.
- Doc drift (not mine): journal_stats.py:13-15 says the two buy rows have the "identical key set". Still false: base has quote_age_s, while stock has book_risk_pct and no quote_age_s. Stock has no `_fetched_ts` stamp, so decision_quote_ts = get_quote fetched_ts.
VERIFIED-CLEAN: the llm_error/llm_backoff/s_defaulted additions are ignored by every reader (action filters: execution_report.py:56/244, journal_stats.py:533, llm_eval.py:1166, learned_lexicon.py:876, llm_eprocess.py:526). The llm_eval needle scan cannot match 'llm_error'. The exec-harness tests (P1, strikes_halt, V2, base_loop_v3) still pass with the new names.
TEST RUNS (each `TRADER_LOG_DIR=…/engine/testlogs hwlock.sh heavy engine-w17-<t> -- $JPY -m pytest tests/<t>.py -q -p no:cacheprovider`, CUDA_VISIBLE_DEVICES=''; py_compile OK on all 4 files; log in w17/test_runs.txt)
- r5_rows_llm 16, r4_journal_flatten 23, r2_replay_harness 10 (determinism pins incl.), r1_inherited_positions 11, c26_base_loop_functional 37, c26_T6 43, loop_fixes_2026_09 22+1 skip, review_b01 21, c26_P1 44, ia2_safety 24, llm_eval 9, journal_stats 20, sizing_signal_instruments 6, r2_strikes_halt 10, base_loop_v3 26, c26_V2 32, trade_journal 3, review_b18 29, grp_loops 9, r5_live_invariants 5. ALL PASS.
- The get_quote pins were not run against the tree because order_utils.py is unchanged. They were run against the quote_t scratch variant only; results above.

## ADDENDUM: item 2 LANDED on the general's authorisation (w17/orig_tests/ = pre-item-2 copies of the 3 test files + base_loop)
- order_utils.py:220-226, 243. `quote_t = float(qt.timestamp())` is computed in its own try (None on any failure) AFTER the staleness try-block. Accept/reject cannot change. It is the additive 7th key, appended after fetched_ts. Return type and every existing key are unchanged.
- base_loop.py:117-135. `_decision_quote_journal_keys` also emits `decision_quote_t = quote.get('quote_t')` (None when absent). Both buy rows (base :3631 and stock :1394 via the shared helper) therefore append 5 keys. trade_journal.py docstring updated.
- Pins modernised only for the additive key, nothing weakened:
  - test_engine_r3_o3_quote.py:357-360: `_assert_legacy_quote` pops quote_t and asserts it is a non-NaN float, then keeps the exact `out == _EXPECTED`.
  - test_engine_r4_journal_flatten.py:53-54 (NEW_BUY +decision_quote_t) and :92-99 (expected dicts +decision_quote_t None).
  - My r5 NEW_BUY.
- New r5 tests (+12, file now 28):
  - get_quote quote_t equals the exchange epoch for aware-UTC / naive-UTC / pandas-NY / pandas-ns, crypto and stock. The key is appended last.
  - A `.timestamp()`-raising datetime gives quote_t None while the quote is still ACCEPTED; a stale quote is still None.
  - Helper passthrough.
  - Stock buy rows: 0 ≤ decision_quote_ts − decision_quote_t ≤ 180.
  - Crypto buy rows: decision_quote_t equals the stamped decision quote's quote_t.
- Mutation (`-p w17b_orig`: pre-item-2 order_utils + base_loop): r5 14 FAIL (all quote_t/decision_quote_t tests, KeyError / key-set) and 14 pass (the item 1/3 tests); r3_o3 14 FAIL (KeyError 'quote_t'); r4 3 FAIL (helper dict, A/B key sets). Log: w17/mutation_item2.txt.
- No production consumer serialises, iterates or compares the whole quote dict (grep for `**quote`, `.items()`, `json.dumps(quote`, `==`, copy: none). This is unchanged from the report above.
- TEST RUNS (arbiter, TRADER_LOG_DIR set, one per process; log w17/test_runs_item2.txt): r5_rows_llm 28, r3_o3_quote 74, r4_journal_flatten 23, order_utils 16, order_utils_v3 40, r2_flags 52, r4_ts_warning 7, crypto_quote_staleness_census 26, r1_cost_kernels 68+15 xfail, r2_replay_harness 10 (determinism incl.), c26_T7 37, c26_T6 43, ia2_safety 24, review_b02 27, r1_inherited_positions 11, c26_base_loop_functional 37. ALL PASS. py_compile OK: order_utils, base_loop, trade_journal and the 3 test files.
