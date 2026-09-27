# ENGINE r4 W13 — R4-a journal join keys + R4-b flatten-site exit rows (sandbox generals/engine/w13/: orig/, ab/, preload/)
LANDED (tree green at every step; base_loop edits applied atomically via os.replace after py_compile)
- R4-a base_loop.py:95-134. `_order_journal_ids(order, keep_none)` returns {'order_id': str|None}. `_decision_quote_journal_keys(quote)` returns decision_bid, decision_ask, decision_quote_ts (epoch s; '_fetched_ts', else 'fetched_ts'). Neither helper raises.
  - Buy row, :3626-3634. Adds order_id, decision_bid, decision_ask, decision_quote_ts via setdefault, so keys are only appended and never overwrite. order_id comes from `result` (what _execute_entry_order returned); for the maker ladder that is the rung place_maker_buy returned. The quote is the dict _execute_buys stamped, not a new fetch.
  - Sell rows. _record_confirmed_exit :2095 and _execute_stop_exit :1694-1705 (result, else the submitted order) add order_id ONLY when the order carries an id. This is required by tests/test_c26_T6.py:750, which pins the legacy key-set for an id-less order.
  - `entry_tactic` already existed (:3614), so it is left as is. No tactic key was added.
- R4-b, remote flatten, base_loop.py:3057-3078. Under BREAKER_SERVER_FILL_ATTRIB it reuses `_breaker_record_server_fill` with the new `site=` kwarg (:888; labels in `_SERVER_FILL_SITES` :137). With site='remote_flatten', detect_source is 'remote_flatten' and the log says [FLATTEN]. When ON attributed a row, the state is saved again. The breaker's default site keeps its log text byte-identical (pinned).
- R4-b, stablecoin flatten, :1100-1150 (confirmed: inside `_update_macro_regime`'s depeg branch). It now:
  - takes a pre_flatten snapshot and calls `_save_position_state()` after the release;
  - writes one `record_trade(..., exit_reason='stablecoin_flatten', estimated=True)` row per released position at the quote mid, the same shape as the breaker's :825 `'circuit_breaker'` row, with a per-position try/except;
  - with the flag ON, uses the same helper (site 'stablecoin_flatten').
  journal_stats has no known-reason list: its Counter counts any reason (journal_stats.py:326). These rows go to trade_memory (record_trade), exactly like the breaker and remote rows.
- Docs:
  - The schema's one home is trade_journal.py:14-20 (module docstring). docs/STATE_FILES.md has no key list, so it is untouched.
  - strategy_config.py:405: one header line rewritten in place to name all three sites. The line count is unchanged, so the constant stays at :427.
  - docs/FLAGS.md:87: row text updated and my test file added.
- New tests/test_engine_r4_journal_flatten.py (23 tests). The core harness tests use the fake broker and the r2 `replay` fixture:
  - buy rows (maker ON/OFF plus a zero-spread tape where maker rung 1 fills): order_id is the broker's filled buy, and bid/ask/ts match the stamped quote;
  - an in-process A/B with the helpers stubbed to {}: identical broker events/calls/trades/state, stripped rows identical, key order preserved;
  - sell/stop/stock rows carry order_id;
  - readers unaffected: decision_report.load_journal + admitted_k, journal_stats load_trades/compute_stats/format_summary/eod_digest, execution_report.run_report and fees.realized_crypto_maker_share give identical output with and without the new keys;
  - remote flatten OFF: 0 get_order, six estimated remote_flatten rows at the mid. ON: six server_stop rows with detect_source 'remote_flatten', the real fills, lockouts;
  - stablecoin: six estimated rows; the state blob after cycle 1 has hwm {} (pre-edit: six); cycle-1 market submits are exactly the six flatten sells. Also ON vs OFF, with stops filled after _manage_stops;
  - never-raise on test_grp_loops' int-position stub; site log labels.
- Mutation check (`-p w13_orig` preloads orig/base_loop.py): 19 FAIL, 4 PASS. The 4 are byte-identity pins that hold on both versions: reader identity, fill_venue independence, remote OFF legacy, and breaker default-site log text.
- Scratch A/B vs orig (ab/pre0.json vs post1.json, PYTHONHASHSEED=0; 7 scenarios):
  - cash maker ON/OFF, stock day, remote OFF: journal (new keys stripped), broker events, calls, trade memory and state are all identical.
  - stablecoin OFF: broker events identical. The only added calls are 6 read-only get_latest_crypto_quotes; trades gain the 6 rows; state is saved.
FOUND-NOT-FIXED
- client_order_id is NOT journaled (STOP-and-report). make_client_order_id adds a uuid4 tail (order_utils.py:37). The tree was green; my first cut added the key, which failed tests/test_engine_r2_replay_harness.py::test_crypto_replay_is_deterministic and ::test_stock_replay_is_deterministic (they compare raw journals across two runs). I withdrew the key rather than edit those tests; see base_loop.py:91-95 `_JOURNAL_ID_FIELDS`. order_id is enough for the join (GET /v2/orders/{id} returns client_order_id). To enable it:
  - add `('client_order_id', 'client_order_id')` to `_JOURNAL_ID_FIELDS`;
  - in both determinism tests, reduce journal rows' client_order_id to its tag, like fake_alpaca_broker.normalized_events:777;
  - loosen the four `'client_order_id' not in r` asserts in my test.
- stock_loop.py:1364-1387 builds its own buy row. Per journal_stats.py:13-15 it must keep the key set identical to base's, and it now lacks the 4 keys. The file is not mine. Exact diff, inserted before :1387 `log_decision(buy_rec)`:
  `from base_loop import _order_journal_ids, _decision_quote_journal_keys` and `for k, v in {**_order_journal_ids(result if result is not None else order), **_decision_quote_journal_keys(quote)}.items(): buy_rec.setdefault(k, v)` (verify `result`/`order` are in scope, :1278-1280).
- decision_quote_ts is the fetch stamp, not the exchange quote time: get_quote drops `t` (order_utils.py:217-224). A 'quote_t' key there would give the true arrival time (order_utils owner).
- journal_stats.py:17-24 docstring drift: it says circuit-breaker exits go through _record_confirmed_exit. They are record_trade-only, and so are remote_flatten and stablecoin_flatten; none reach the decision journal. (INTEL/measure.)
VERIFIED-CLEAN: fill_venue_slippage_report never reads journals (broker bundle only; pinned by a source-token test); log_decision and the new key builders never raise; flag-OFF paths make zero extra broker calls at both new sites.
TEST RUNS (each `hwlock.sh heavy engine-w13-<t> -- nice -n 10 $JPY -m pytest tests/<t>.py -q -p no:cacheprovider`, CUDA_VISIBLE_DEVICES=''; py_compile OK)
- r4_journal_flatten 23, r3_o3_quote 74, r2_replay_harness 10, r1_inherited_positions 11, c26_base_loop_functional 37, c26_T6 43, loop_fixes_2026_09 22+1 skip, review_b01 21, journal_stats 20, decision_report 19, decision_report_v3 30, fill_venue_slippage_report 13, trade_journal 3, grp_loops 9, base_loop_v3 26, r3_restart_anchor 17, r2_strikes_halt 10, r2_flags 52, c26_T7 37, fees_feedback 8, fees_v3 29, conviction_journal 12, improve_stratcfg 10, ia4 43, ia3 28, ia2 24, c26_X1 25, c26_P1 44, review_b07 55. All pass.
