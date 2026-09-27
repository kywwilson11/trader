# ENGINE — ROUND 7 (FINAL, wind-down per founder directive) — 2026-09-27 07:10–

Workers in flight at the directive: W20 (H5 fix + H1–H4 harness pins/xfails + W16 harness gaps) and W21 (phantom-exit dry-run audit + X12
paper-quirk census incl. the read-only #23 asset-id table). Both were spawned before the directive and are allowed to finish; NO new workers.
Gate: engine-r5 queued memory-first at 07:05 (box at ~1.5 GB free with the stock search running) — GATE_RESULT_PLACEHOLDER
W20 LANDED: H5 class-D fix (`order_utils._basis_or_zero` :914-932; reconstruct :1249, base_loop.py:3560-3566, alpaca_compat.py:59-73 — null basis follows the "0" path; float()-identical otherwise; mutation-checked); fake broker fault controls (hide_position, equity_equals_cash, fill_avg_none, reserve_qty_on_resting_stops, ns_timestamps, canceled_at, apply_live_modes); H1/H2/H3b/H4 pinned today + strict xfails of the intended invariant (owner #24). tests/test_engine_r7_broker_failure_modes.py 46 + 4 xfail; 15 neighbour files green. Cross-dept: stock_loop.py:1306 has the same bare float(vp.avg_entry_price) on the partial-fill path (use `_basis_or_zero`; the :1314 guard then handles it) — not landed under wind-down.
W21 LANDED (measurement-only): scripts/trade_memory_phantom_audit.py (dry-run, NO write mode) + scripts/paper_quirk_census.py (X12) + 51 tests. Real runs: 63 phantom `broker_stop` exits CONFIRMED (no sell fill within 5.35 d; none carries `estimated`) ⇒ crypto Kelly 0.1995 on 73 rows ⇒ a 1.5× crypto sizing multiplier TODAY (base_loop.py:2587-2591; KELLY_SAMPLE_GATE False) — repair list w21/repair_real.json NOT applied (founder); census: basis lost + asset-id split on all six (position id ≠ resting-stop id ≠ current asset id), stops DO reserve the old-id qty, fill-reduces-position still unverified; no vanish, no orphan reservation. trade_memory.json sha 87d23f21 unchanged.

## Where everything is
- Round reports: <scratchpad>/generals/engine/ROUND_1..7.md; worker reports W1–W21_*.md (+ w*/ proofs, pre-edit copies, mutation dirs).
- Owner items (24 + standing instruments): <scratchpad>/generals/engine/OWNER_ITEMS.md — § A joint ENGINE→SIGNAL cost-model item, § B five in-tree
  default-OFF flags (CRYPTO_QUOTE_MAX_AGE_SEC, HALT_CANCELS_WORKING_BUYS, BREAKER_SERVER_FILL_ATTRIB, RESTART_STOP_ANCHOR_DESIRED, BARS_PER_YEAR
  lockstep) + TRADER_BOT_RESTART_BACKOFF (run_pipeline env flag), § C unflagged decisions, § D cross-dept, § E standing instruments.
- Per-change log: research/campaign_2026-09_jetson/CHANGELOG.md (ENGINE R1–R7 lines). Research log: research_engine.md (R1, X7, W9 table, R3, R4, R6).
- Census evaluator for the founder: <scratchpad>/generals/engine/w7/flip_rule.py over w7/census_24h_*.json (complete ≈ 06:16Z 09-28).

## Session totals (ENGINE, all through failing-before/passing-after tests; every earlier gate GREEN except where noted in ROUND_4/5/6)
Class-A fixes landed: zero-basis cap no-op + $0 stop at buy (R1), overnight keeper left with no server stop (R2), D13 stale veto strike (R2),
fail-closed quote timestamp (R3), double-loop restart race (R5). Class-D: nanosecond warning, TRADER_LOG_DIR, H5 (if W20 lands). Flags (all OFF,
byte-pinned): 6. Measurement-only modules: crypto_quote_staleness_census, fill_venue_slippage_report, lstm_numpy_serve (research), setup --user,
(+ W21's two). Test architecture: fake_alpaca_broker replay harness (crypto + stock, recorded tape, live-invariant fixture); ~20 new test files.
