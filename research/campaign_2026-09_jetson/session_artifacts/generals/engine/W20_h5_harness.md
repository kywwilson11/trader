# ENGINE W20 (R7): H5 class-D fix, H1–H4 harness pins, W16 F2 harness gaps. No orders placed, bots not touched, crypto_loop.py unedited.
## LANDED
1. H5, class D (objective: a null basis crashed or dropped the position; now it follows the existing "0" path). New helper `order_utils._basis_or_zero` (order_utils.py:914-932) is float(x) for every input float() accepts. None, '' and garbage (TypeError/ValueError) return 0.0 and write one debug line with the raw type. It is used at three sites:
   - order_utils.py:1249: `reconstruct_positions._entry`. Before, float(None) led to "bad position payload" and the position was DROPPED; the probe path swallowed the error.
   - base_loop.py:3560-3566: `_place_and_track_buy`. Before, it raised after a real fill, leaving the position untracked with no stop. Now it takes R1's filled_avg_price fallback.
   - alpaca_compat.py:59-73: `_shim_position` (lazy import). avg_entry_price is the shim's ONLY basis field, since cost_basis is not exposed. qty keeps float(), so a null qty is still a bad payload (pinned).
   Tests: basis grid ×19 ('0', 0, '21031.39', 1e-9, Decimal, nan/inf, -0.0 …) byte-identical by repr; garbage ×7; reconstruct (list and probe) null == '0'; the null-basis book gives startup and cycle-1 events byte-identical to the '0' book, with 6 "cost basis unknown" WARNINGs and a 6% HWM-anchored stop; null position basis + order fill 100.0 gives entry 100.0 and a stop at 100·(1−d); shim returns 0.0 with no raise; CompatREST→verify_position returns the live position.
   Mutation check (pre-edit copies in w20/orig, run through the w20/mirror tree): 33 failed / 13 passed / 4 xfailed. 14 of the failures are the fix-specific tests: base_loop.py:3563 and alpaca_compat.py:65 TypeError, the reconstruct/verify drops, H3. The other 19 are grid cases failing only because the helper is missing. All the pins and xfails behave the same on both trees.
2. tests/fake_alpaca_broker.py: fault switches, all default-OFF and documented in the docstring under "CONTROLLABLE FAULTS":
   - hide_position(sym, cycles, at_step): list/get omit the position, a new sell sees 0 available, resting orders untouched, equity intact.
   - equity_equals_cash(cycles, at_step): equity == cash and every position hidden.
   - fill_avg_none(position=, order=): null basis on the position and/or null filled_avg_price on the buy order.
   - reserve_qty_on_resting_stops: default True = the existing rule; False is a control.
   - ns_timestamps: pandas Timestamp +123 ns.
   - canceled_at stamped by cancel, the OCO/parent cascade and cancel_all. Order attribute only; event dicts are unchanged.
   - apply_live_modes(monkeypatch, workdir, no_model, halt): chdir to an empty dir, so the real load_models raises FileNotFoundError (its paths are cwd-relative); the halt flag is redirected into the tmp workdir and created when halt=True.
   All seven fake consumers pass unchanged.
3. tests/test_engine_r7_broker_failure_modes.py: 50 tests (46 pass, 4 strict xfail). Sockets are blocked. Every file the loops write goes to tmp (including crypto_predictions.json and trading_halt.flag).
## H1–H4: today's behaviour pinned vs strict-xfail desired (owner item #24; no production change)
| mode | today's behaviour, PINNED (passes) | DESIRED, strict xfail (fails on its stated assertion under --runxfail) |
|---|---|---|
| H1 hide BTC at the confirm cycle | cancel resting stop (base_loop.py:1605), then market sell rejected 'available: 0', then verify_position None, then DESYNC estimated row and pop (:1626-1648). After it reappears: held at the broker, untracked, 0 open orders | reappeared position is tracked, or has a resting stop, within 2 cycles |
| H2 equity==cash 1 cycle | dd 99.924% trips the breaker (base_loop.py:763); flatten lists nothing, so 0 events; positions.clear() (:854); 6 estimated circuit_breaker rows; halted; restored positions untracked and 6 stale stops still resting | a breaker that sold nothing while positions were hidden keeps tracking |
| H3 position basis null | PASSES (the fix): entry = order fill, stop at fill·(1−d), 0 rejects | — |
| H3b order fill null + avg "0" (W19 variant) | entry 0; stop_limit rejected 'limit_price must be > 0', so stop_order_id None (crypto_loop.py:201); buy row fill_price 0, no estimated key | no stop ≤ 0 submitted; row marked estimated/basis_unknown |
| H4 add-on over a resting stop | full-qty stop rejected 'insufficient balance for AVAX' (crypto_loop.py:204-206); stop_order_id None (:201); next cycle skips get_order on the old stop (base_loop.py:1418, blind), then _maybe_update_resting_stop (crypto_loop.py:227-243) cancels it and re-places one full-qty stop. Control with reserve=False: accepted, 2 stops | old stop cancelled before _after_entry_protection (J1): 0 rejects, 1 tracked stop for the total qty |
## W16 F2 harness gaps (pinned)
ns quotes parse with no nanosecond warning, and 2 cycles are event-identical to the stdlib-datetime run. canceled_at == the cancel event's t. Live 03:09 configuration (real _load_models fails closed + halt flag): startup and 2 cycles are event-identical to the flat case, 0 buys, nothing written to the model dir. Combined mode: the stock startup logs 'Canceling 0/6', emits 0 broker events, and the 6 crypto stop ids are unchanged and still tracked.
## FOUND-NOT-FIXED
- stock_loop.py:1306 `fill_price = float(vp.avg_entry_price)` (partial-fill stock buy path; no enclosing try) raises on a null basis and escapes the cycle. Cross-file, not mine to edit. Proposed diff: `from order_utils import verify_position, _basis_or_zero` … `fill_price = _basis_or_zero(vp.avg_entry_price)`. The existing `not fill_price or fill_price <= 0` guard at :1314 then handles it.
- Decisions for H1/H2/H3b/H4 are owner rulings (#24). If a ruling lands, retire its pin together with its xfail.
## VERIFIED-CLEAN
alpaca-py shim: the only other basis-like field is none (cost_basis is not shimmed). No other float(avg_entry_price) in the live loops except stock_loop:1306.
## TEST RUNS (each through hwlock, CUDA_VISIBLE_DEVICES='', TRADER_LOG_DIR=testlogs, one file per process)
py_compile OK (order_utils, base_loop, alpaca_compat, fake, new test). New file: 46 passed, 4 xfailed; mirror: 33 failed/13 passed/4 xfailed; `--runxfail -k desired`: 4 failed on their stated assertions.
r1_inherited 11, r2_replay 10, r3_restart_anchor 17, r3_o3_quote 74, r4_journal_flatten 23, r5_rows_llm 28, r2_strikes_halt 10, test_order_utils 16, order_utils_v3 40, c26_T6 43, c26_base_loop_functional 37, review_b01 21, review_b03 38, ia2_safety 24, r4_ts_warning 7: all passed, 0 failed (w20/verify_results.txt).
