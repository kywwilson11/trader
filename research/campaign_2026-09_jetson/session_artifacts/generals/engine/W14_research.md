# W14 — R4 research scout: E6 crash-loop backoff and E3 drawdown ladder (2026-09-27)
Full section: research_engine.md:531-642 (appended; lines 1-530 byte-identical). Scratch: w14/. No production edits; only read-only GETs.

## Ranked by value/cost
1. OWNER, 1 minute, ops. No notify channel is configured in .env, so notify.py:259 returns early.
   The April loop would have sent 232 alerts and sent 0. Every breaker/crash alert on this box is silent.
2. E6 `BOT_RESTART_BACKOFF` (S, ops flag default OFF, run_pipeline constant ← `TRADER_BOT_RESTART_BACKOFF`,
   following the EOD_DIGEST pattern at run_pipeline.py:144).
   - Design: pure functions, crash signature = innermost frame + exception line; a give-up latch after 5 identical crashes in 60 min.
   - Replay of the real 1,778-crash history (one signature, 100 % repeats; p50 lifetime 42 s): 1,616 → 12 restarts (99.3 % avoided).
   - Distinct signatures add 0 s: PASS. Backoff alone reaches only 89.5 % (FAIL); the latch does the work.
   - Owner policy: latch vs parked 3,600 s probe (90.8 %). A transient-outage startup crash would latch bots down.
3. Alert-only companion (S, class D). One critical alert after 20 identical consecutive cycle errors.
   Since edf3151 (2026-06-10, base_loop.py:347-363) the M7-type crash loops IN-process: a deduped warning forever, with no escalation.
4. X11 ladder/breaker census (measurement, needs 60 RTH days of bots). Pre-registered rules:
   floor masking ≥ 25 % → owner item; per-book peak divergence ≥ 1 % → `DD_PEAK_ACCOUNT_SHARED` (default OFF).
5. B3 placeholder-equity window ($100k for ≤ 10 cycles after 2 bracketing account-read failures). Already an owner decision (base_loop.py:1175-1179).

## Key facts
- The process-level crash surface is now startup only (base_loop.py:332-345). In combined mode a stock startup crash kills the crypto loop too (run_bots.py:163-175,:228), costing a cancel plus restart stops each time.
  Six zero-basis crypto positions are held now.
- systemd cannot see bot crash loops (mark_progress at run_pipeline.py:1034). Default start limits never trip at ~74 s spacing. RestartSteps needs ≥ 254; this box runs 249.
- E3, real equity (Alpaca daily; the hourly series was rejected as corrupt: 71/173 points diverge):
  - While the bots ran (02-24→05-07): max DD 6.13 %, ladder never below 1.0, 0 breaker trips.
  - Untended after May: ladder at 0.25 for 52 days, 4 close-to-close breaker days, 3 of them compounding.
- Objective findings, not fixed:
  - B1: the 0.1 tilt floor masks the 20 % rung (it acts as 0.255 in April's regime).
  - B2: two per-book peaks for one account, and the stock book samples in RTH only.
  - Stale doc cite: docs/FLAGS.md:155,:459 say run_pipeline.py:118, but the line is now :144.
- Kill list / 08_removed_code / 07 ledger checked: nothing on supervision. The ladder rungs are owner preference and "never a fourth" is respected.
