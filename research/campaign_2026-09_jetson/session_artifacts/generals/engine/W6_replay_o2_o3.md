# ENGINE r2 W6 — replay harness (both loops) + O3/O2 verdicts. Proofs/sandbox: generals/engine/w6/
LANDED
- stock_loop.py:1489-1498 (class A, +1 condition): the trailing upgrade in `_manage_stops` is skipped once `flattened_today` is set. Found BY THE HARNESS. Cycle order: flatten_before_close (base :334) -> _manage_stops (:347). At 15:50 ET `_prepare_overnight_keepers` (:426) cancels the keeper's legs, places a GTC `stop` and resets trailing_activated=False (:457). In the SAME cycle the upgrade (keeper >= +1 %) cancelled that GTC stop and sent a DAY trailing_stop, which expired at 16:00. Result: every profitable sleeve keeper rode overnight with NO server stop. Next morning stop_order_id pointed at the expired order, got cleared, and no server stop at all remained (software only). The fix keeps exactly the order `_prepare_overnight_keepers` documents. Positions/levels are otherwise untouched: after flattened_today only keepers remain, and entries are off. getattr form, because test_review_b01 runs this source on a bare namespace.
  Tests: test_overnight_keeper_gtc_stop_survives_same_cycle_trailing_upgrade and test_stock_replay_rth_day. Mutation: both FAIL against w6/stock_loop.py.pre (sandbox w6/mut); pass after. JUDGMENT flag for the general: it changes order flow (one cancel plus one day-trailing submit fewer per keeper per day), so revert it if you class that as an exit-rule change.
- tests/fake_alpaca_broker.py extended; the 11 r1 consumers still pass. Plus tests/test_engine_r2_replay_harness.py (10 tests: 9 pass + 1 strict xfail = O3) and the recorded fixture tests/fixtures/crypto_tape_2026-09-27.json (11.7 KB; 6 names x 21 rows at 30 s, 06:17-06:27 UTC; read-only GETs via `record_crypto_tape`, run under the arbiter).
  - The tape shows real staleness: ETH quote age max 264 s, XRP 200 s. The replay reproduces the None-quote cycles (O8 evidence).
  - Runtime: the whole file takes 5.2 s; the stock day takes 0.16 s.
HARNESS CAPABILITIES (API -> rule)
- API->rule: get_clock: is_open/next_open/next_close from `sessions` (RTH); a tick crossing a close EXPIRES open stock day orders ; submit_order bracket: Alpaca buy-bracket price rules enforced (tp >= base+0.01, stop <= base-0.01). Two legs 'held' until the parent fills, reserve qty once, OCO (fill or cancel of one kills the sibling), `legs` on the order object ; stop / trailing_stop: trigger on `last` (Tape) or on bid (legacy mid tape, unchanged); fill at the bid; trailing uses the HWM since submit ; stop_limit sell: triggers when trigger <= stop, fills at the bid if bid >= limit, else rests. Deviation from the brief: NOT min(limit, next last), because a sell limit can never fill below its limit
  market / limit: touch fills; stocks fill only while the session is open (orders are accepted when closed, flag `mkt_open`) ; sell qty / buy BP: oversell rejects: crypto "insufficient balance for X", stock "insufficient qty available". BP: crypto non-marginable cash, stock buying_power ; list_orders: open (insertion order), closed/all (newest first), symbols filter ; get_latest_quote / crypto_quotes: t = clock (optional recorded age); `calls` logs every API call (round-trip counter)
Other parts:
- Clock: `install_clock` swaps the datetime/time module globals of base_loop, crypto/stock_loop, order_utils and trading_utils for shims driven by the FakeClock. No production time call changed.
- `Tape` (schema fake_alpaca_broker.tape/v1, documented in the docstring) and `synthetic_rth_tape`: -4 % gap at 10:00, +3 % rally from 14:00.
- Sockets are blocked, and any network attempt fails a test.
- No get_calendar, replace_order or close_position: no consumer calls them (census in the docstring).
FOUND-NOT-FIXED (owner items)
- O3 verdict (OBJECTIVE bug, owner item, not landed):
  - Cause: cycle order base :332 breaker -> :347 _manage_stops. On the -7 % gap all six stop_order_ids are ALREADY 'filled' at the broker when the breaker runs (pinned by test_o3_breaker_journals_estimates_...).
  - Current output: emergency_flatten (:764) finds no positions and sends no orders. The loop at :793-801 then record_trade()s six estimated=True 'circuit_breaker' exits at the quote MID, not the real bid fills. No sell rows and no lockouts, whereas the :1277-1297 server_stop branch would have written them.
  - Why it is not class A: the fix is journal-only for already-closed positions, but it moves 6 rows into Kelly's sample (estimated rows are excluded at trading_utils:240) and adds a 24 h lockout plus a 60-min cooldown. The breaker halt ends at the next 16:05 ET, i.e. < 24 h, so a later BUY changes. The r1 test_minus7_gap... also pins the current behaviour.
  - Flag design `TRADER_BREAKER_SERVER_FILL_ATTRIB` (default OFF), placed AFTER the flatten so the liquidation gains no latency:
    - for each released position with stop_order_id: get_order; if 'filled', write _classify_server_stop + _record_confirmed_exit('server_stop', extra detect_source='breaker') + last_trade_time + _apply_server_stop_lockout, then continue; otherwise write today's estimated row.
    - Exact patch: w6/o3_patch.py (patched copy w6/base_loop.o3_design.py).
    - Sandbox: flag=0 -> pin passes, xfail holds (byte-identical). flag=1 -> pin fails and the strict xfail XPASSes.
  - Same pattern at the remote-flatten site base :2864-2893 (runs first in the cycle, estimated 'remote_flatten' rows). The stablecoin flatten :979 journals nothing.
- O2 numbers:
  - 18 order writes per restart instead of 6: 6 startup stop_limits at 6 %, then in cycle 1 6 cancels + 6 re-submits at the 5 % trail (test_o2_...).
  - Unprotected per name: exactly one list_orders call between cancel and re-submit (after one 0.5 s poll sleep), about 0.5 s + 2 RTT. 0 whole cycles. Plus a 30 s window with the looser 6 % level.
  - Startup-with-_desired_stop_for is byte-identical ONLY when hwm == entry (entry>0, trail off; macro_regime is None at startup). Pinned counterexamples:
    - entry 100 / hwm 101 / ATR 1: startup 98.475 vs desired 97.5 = J11 (known owner item).
    - trail active + no ATR (entry>0): churns exactly like zero-basis.
  - So no stop-level change was shipped. Owner write-up: pick an anchor rule for J11+O2 together (startup = _desired_stop_for fixes both, but it widens J11 stops to the entry-anchored level).
- For INTEL: tests/README.md rows for test_engine_r2_replay_harness.py (10 tests) and for tests/fixtures/ (new dir).
VERIFIED-CLEAN
- The crypto replay over the full recorded tape runs clean in both scenarios (zero-basis book; $30k cash with entries + a signal sell): no exception, no reject, all stop_limit limit<stop, universe-only symbols, the state file is valid JSON every cycle, and there is 1 resting stop per position.
- Stock day checks: buys only within 09:45-11:00 / 14:30-15:30 ET, every entry is a bracket with 2 legs, nothing is submitted after the close, the gap fills record as server_stop with real fills and lockouts, and the non-keeper is flattened before 16:00.
- Two runs are identical (events, trades, journal, state) for both books once the tmp state file is reset. The HWM persists across runs by design.
TEST RUNS (each `hwlock.sh heavy engine-w6-<t> -- $JPY -m pytest tests/<t>.py -q -p no:cacheprovider`, one per process; py_compile OK)
- r2_replay_harness 9 passed + 1 xfailed; r1_inherited_positions 11; c26_base_loop_functional 37; loop_fixes_2026_09 22 + 1 skip; review_b01 21; c26_T6 43.
- Also: c26_P1 44, c26_X1 25, c26_S3 55, grp_loops 9, ia3 28, ia2 24, ia1 21, ia4 43, g3_ops 43, review_b03 38, imports 21, improve_stratcfg 10, prediction_cache_context 4, execution_policy_v3 33. All pass.
