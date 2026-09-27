# ENGINE — ROUND 4 (2026-09-27 03:05–04:10) — general: Fable; workers W13–W14 (Opus, ≤2 live) + one general-direct fix

Worker reports: <scratchpad>/generals/engine/W13_journal_flatten.md, W14_research.md. Consolidated owner items: OWNER_ITEMS.md (19 items + joint
ENGINE→SIGNAL section A with W12's fill evidence). Paper bots LIVE since 03:09 on the harness-validated tree (exits-only); startup matched the R1 prediction.
Gate: engine-r4 (03:22 CDT) — RED by ONE test: 1 failed, 5398 passed, 7 skipped, 64 xfailed. The failure is W13's own `test_stablecoin_flatten_row_failure_never_breaks_the_branch` (green alone, red under suite order: it patched `base_loop.get_macro_regime` but the method resolved the REAL one — captured log shows VIX fetched, no flatten). Production code is not implicated. Test fixed to patch `CryptoLoop._update_macro_regime.__globals__` (suite-order safe); py_compile OK; RE-RUN + GATE PENDING "HW: resume" (importing base_loop is heavy).

## LANDED (failing-before/passing-after; CHANGELOG.md: 4 ENGINE R4 lines)
1. R4-a journal keys (W13, measurement-only, additive): buy rows carry `order_id` (the filled order / maker rung), `decision_bid`, `decision_ask`,
   `decision_quote_ts` (base_loop.py:3626-3634; helpers :95-134); sell/stop rows carry `order_id` when the order has one (:2095, :1694-1705). Readers
   (decision_report, journal_stats, execution_report, fees) pinned identical with/without the keys; pre-existing keys byte-identical (value AND order).
   Schema home: trade_journal.py:14-20 docstring. `client_order_id` deliberately NOT added (random uuid tail breaks the r2 determinism tests; `order_id`
   suffices for the broker join — enable path in W13 § FOUND-NOT-FIXED 1).
2. R4-b remote flatten under BREAKER_SERVER_FILL_ATTRIB (W13; base_loop.py:3057-3078 reuses `_breaker_record_server_fill(site=)` :888, detect_source
   'remote_flatten'); OFF = zero get_order + same rows. Stablecoin flatten (:1100-1150) now writes one estimated `stablecoin_flatten` exit row per released
   position and calls `_save_position_state()` (was: no row, no state); broker events identical (A/B). strategy_config.py:405 comment + FLAGS.md:87 list all
   three sites. tests/test_engine_r4_journal_flatten.py (23; 19 fail on the pre-edit copy).
3. Nanosecond warning (general-direct, class D, CEO-observed live): order_utils.py ~:190 `to_pydatetime(warn=False)` (TypeError fallback) — the pandas
   "Discarding nonzero nanoseconds" UserWarning no longer fires per symbol per cycle; µs truncation identical, verdicts byte-identical at 179/181 s.
   tests/test_engine_r4_ts_warning.py (7; pre-edit module fails 2 under warnings-as-errors). Takes effect at the CEO's next bot restart.
4. Research (W14): research_engine.md § R4 scout — E6 crash-loop census + replay, E3 drawdown-ladder as-is table + real-equity replay + X11 rule.

## RESEARCH VERDICTS (W14)
- Crash loop (April, real logs): 1,778 crashes / 1,808 starts, ONE signature (innermost frame + exception line needed to distinguish), median lifetime
  42 s, market hours only, self-recovered 04-23. **0 alerts SENT — no notify channel is configured** (notify.py:259 returns early; 232 would have fired).
  Replay: backoff 60→960 s alone avoids 89.5 % (fails ≥ 90 %); backoff + give-up after 5 identical/60 min avoids 99.3 % (12 restarts, 3 alerts, gives up
  ≈ 15 min per episode; distinct signatures still restart ≤ 60 s). Proposal `BOT_RESTART_BACKOFF` ← TRADER_BOT_RESTART_BACKOFF (run_pipeline env
  pattern, default OFF); owner picks permanent give-up vs 3,600 s parked probe. Since edf3151 cycle errors are caught in-process (base_loop.py:347-363) —
  the April crash would now loop forever with a deduped warning; companion alert-only proposal: one critical alert after 20 identical cycle errors.
  Combined mode: a stock STARTUP crash still takes the crypto loop down (run_bots.py:163-175/:228) and each restart re-places the six crypto stops.
- Drawdown ladder (real daily equity; hourly series corrupt — 71/173 points off > 2 %): bots live 02-24→05-07 max DD 6.13 %, ladder never < 1.0, 0 trips;
  untended 52 days at rung 0.25, 3 days both mechanisms active. Objective findings: B1 the 0.1 floor masks the 20 % rung (acts 0.255); B2 per-book peaks
  + RTH-only stock sampling → books on different rungs (proposal `DD_PEAK_ACCOUNT_SHARED`, OFF, after X11); B3 $100k placeholder sizing up to 10 cycles
  when account reads fail (owner note base_loop.py:1175-1179). X11: measurement-only, needs 60 RTH days of live bots.

## OWNER / CROSS-DEPT (all in OWNER_ITEMS.md; new this round)
- #17 configure a notify channel (one minute; every breaker/crash/halt alert is silent today). #18 backoff design. #19 ladder B1–B3.
- stock_loop.py:1364-1387 builds its own buy row and lacks the 4 new keys (ENGINE, next round; diff in W13 § 2). `decision_quote_ts` is fetch time, not
  exchange time (get_quote drops `t`, order_utils.py:217-224 — a `quote_t` key next round). journal_stats.py:17-24 docstring wrong about breaker exits (INTEL).
- FLAGS.md rows 155/459 cite run_pipeline.py:118 for the EOD digest flag; it is at :144 (INTEL doc owner).
- Deferred: R4-c census evaluation (files complete ≈ 06:16Z 09-28; w7/flip_rule.py).

## NEXT ROUND PLAN
R5-a stock buy-row parity (4 keys) + `quote_t` in get_quote (additive) — measurement-only.
R5-b `BOT_RESTART_BACKOFF` as pure functions in run_pipeline behind the env flag (OFF byte-pinned; replay test on the real April history as fixture).
R5-c live-bot evidence read from <scratchpad>/phase5/monitor.jsonl + bots_stdout.log with the harness: compare predicted vs observed cycle behaviour, file any divergence.
R5-d research third: E1 follow-up on Alpaca paper quirks (avg_entry_price=0 root cause, position disappearance incidents) + census evaluation when due.
