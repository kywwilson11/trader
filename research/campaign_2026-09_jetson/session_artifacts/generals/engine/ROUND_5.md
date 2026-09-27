# ENGINE — ROUND 5 (2026-09-27 03:45–, under HW pause) — general: Fable; workers W15–W16 (light-only) + W17 queued for "HW: resume"

Worker reports: <scratchpad>/generals/engine/W15_backoff.md (+ addendum), W16_live_evidence.md. Owner items: OWNER_ITEMS.md (now 21 items).
Gate: engine-r5 (06:04 snapshot) — RED, 8 failed: 5 = ENGINE tests/test_engine_r5_live_invariants.py, cause = its fixture `live_bots_2026-09-27_excerpt.log` is gitignored (.gitignore:29 `*.log`) so the rsync snapshot (and any clean checkout) lacks it — FIXED: renamed to .txt (not ignored), test repointed, 5 passed; 3 = SIGNAL (test_c26_W1 ×2: compute_stock_features byte-identity + HAR sigma; test_raw_sidecar_reload_2026_09 ×1) — indicators.py / harvest_crypto_data.py / volatility.py were being landed by SIGNAL at snapshot time; not ENGINE. Resume pins all green (W13 file 23; test_gate_protocol 28, c26_T4 25, c26_Q2 28, c26_X1 25, c26_T7 37, test_imports 21, test_new_modules 36). Next gate after W17 lands.
globals); (2) W15's deferred heavy pins (test_gate_protocol, c26_T4/Q2/X1/T7, test_imports); (3) suite gate engine-r5; (4) W17.

## LANDED (light-verified: stdlib-only tests + py_compile; suite-level verification pending)
1. `TRADER_BOT_RESTART_BACKOFF` (W15, default OFF; run_pipeline.py flag :1074, pure fns `crash_signature` :1084 / `next_restart_delay` / `consecutive_same`
   / `should_give_up`, `_backoff_verdict` :1195, `_backoff_release` :1250, wiring in `_check_restart_bots` :1265; FLAGS.md §1 row :157 + §5.1 row).
   OFF path sha-pinned (three touched functions byte-identical with the guarded blocks stripped). ON: 60·2^k s capped 960 for repeated identical
   signatures; give-up after 5 identical in 3,600 s with ONE critical notify (dedupe 600 s) and a `[BOTS] giving up…` line; latch cleared by start_bot and the
   weekly `_restart_bots`. Fixture tests/fixtures/april_crashloop_2026.json (1,778 crashes, ONE signature `fundamentals.py:258 | ValueError…`, timestamps ==
   W14 census): replay 1,616 → 12 restarts, 99.3 % avoided, 3 alerts — reproduced exactly through the REAL `_check_restart_bots` on a fake clock.
   tests/test_engine_r5_restart_backoff.py (34).
2. Double-loop race (W15 F1, class A, flag-independent): a start_bot / weekly restart arriving inside the ≤ 60 s window after a crash launched a SECOND
   process while the monitor also restarted the dead entry → one book on two loops (mutation: `assert 2 == 1` in all 6 cases: split/combined/weekly ×
   OFF/ON). Fix `_drop_dead_entries` (run_pipeline.py:758) before every launch (`_launch_bots` :784/:800/:810, `_start_single_bot` :880).
3. Live evidence (W16, read-only, 03:09–03:39, 56 cycles + broker order listing): EVERY observable R1 prediction matched — six RECONSTRUCT warnings; six GTC
   stops at current × 0.94 (limit = stop × 0.98); cycle-1 six cancels then six re-places at HWM × 0.95 (0.67 s unprotected per name, 5.8 s total, cancel
   always first); no later ratchets (max +0.47 %); book stop-risk 0.0 in all 5 rows; no exits/buys; RSS flat 512–565 MiB (≈ 349 MB torch with no model
   loaded). Divergences all explained: breaker now needs −5.80 % (equity +0.85 % vs last_equity) so the 5 % trail fires first until the 16:05 ET reset;
   ETH staleness 2/56 cycles (203/209 s). Invariants I1–I4 pinned: tests/test_engine_r5_live_invariants.py + tests/fixtures/live_bots_2026-09-27_excerpt.txt.

## OWNER ITEMS new this round (in OWNER_ITEMS.md)
- #20 the LLM still runs every 600 s under the halt flag and a 2-strike veto can still SELL (base_loop.py:162/:465/:2190); $0.0126 spent in exits-only mode.
- #21 pytest pollutes the production logs/trader.log (33,452 fake lines since 03:09; log_config.py:21-25 hard-coded path) — `TRADER_LOG_DIR` proposal (INTEL conftest + ENGINE log_config).
- Backoff owner choices (W15): permanent latch (shipped, OFF) vs a 3,600 s parked probe (90.8 %); in combined mode giving up on 'Bots' also stops crypto;
  the critical alert goes nowhere until a notify channel exists (#17). Judgment: `should_give_up` counts same-signature crashes inside the window (not
  strictly consecutive) — an alternating A/B loop latches after ≈ 9; escalation resets after a quiet hour.
- Harness gaps (W16 F2, next round): fake `_quote` returns a plain datetime (ns path untested); `cancel_order` records no `canceled_at`; no-model + halt
  startup and combined-mode startup are not modelled. `[BARS] Dropped 1 outlier` every cycle without the symbol (SIGNAL/market_data, cosmetic).

## QUEUED (W17, on resume): R5-a stock_loop.py:1364-1387 buy-row parity (4 keys) + additive `quote_t` in get_quote; INTEL Scout-E fields A1–A4 on the ENGINE
half of the LLM-call journal (spec research_intel.md:424; fix the "journaled as null" comment now at base_loop.py:1917); readers pinned unaffected.
R5-d census evaluation ≈ 06:16Z (w7/flip_rule.py).
