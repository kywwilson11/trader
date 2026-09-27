# ENGINE — ROUND 2 (2026-09-27 01:10–02:05) — general: Fable; workers W6–W9 (Opus, ≤2 live under HW throttle)

Worker reports: <scratchpad>/generals/engine/W6_replay_o2_o3.md, W7_flags.md, W8_numpy_serve.md, W9_strikes_halt_trail.md.
Gate: engine-r2 — GREEN (Sun Sep 27 01:53:39 AM CDT 2026): 4946 passed, 7 skipped, 19 xfailed, 50 warnings in 299.29s (0:04:59)

## LANDED (failing-before/passing-after; CHANGELOG.md: 7 ENGINE R2 lines)
1. stock_loop.py:1489-1498 (W6, class A): the same-cycle trailing upgrade after the EOD flatten cancelled an overnight
   keeper's just-placed GTC stop for a DAY trailing stop that expired at 16:00 → every profitable keeper rode overnight with
   NO server-side stop. Upgrade now skipped once `flattened_today` (:423; reset per day :129). tests/test_engine_r2_replay_harness.py.
2. Replay harness (W6): tests/fake_alpaca_broker.py now drives StockLoop + CryptoLoop (RTH clock, brackets/OCO, stop/
   trailing_stop, day-order expiry, Tape schema, socket-blocked, deterministic) + tests/fixtures/crypto_tape_2026-09-27.json
   (recorded read-only, 11.7 KB). 9 pass + 1 strict xfail (O3 correct attribution).
3. `CRYPTO_QUOTE_MAX_AGE_SEC = None` (W7, strategy_config.py:371-387; order_utils.py:116-134 `_quote_max_age_sec`, :185 read
   site; OFF = literal 180 s byte-pinned at 179/181/299/301 s; crypto only). docs/FLAGS.md row. tests/test_engine_r2_flags.py.
4. BARS_PER_YEAR lockstep (W7): volatility.py:236 (HAR per-bar sigma) and :756 (vol target) route through bars_calendar at
   call time under SIGNAL's existing `BARS_PER_YEAR_MEASURED`; bars_calendar.py (+`bars_per_day`, `TRADING_DAYS_PER_YEAR`,
   derived from the per-year tables — SIGNAL file touched, its 9 pins green). OFF exact on a 1,464-point grid. ON: only
   GARCH-sourced stock sizing rescales (×0.654); HAR-sourced sizing is invariant (ratio cancels). FLAGS.md row updated.
5. D13 scope A (W9, class A): base_loop.py:1738-1744 clears the veto strike of a symbol SENT to the LLM but omitted from
   the response (veto→omitted→veto no longer liquidates; scope B and the outage path unchanged, both pinned).
6. `HALT_CANCELS_WORKING_BUYS = False` (W9, strategy_config.py:388-403; base_loop.py:2923-2997): OFF = zero broker calls,
   pinned; ON = once per halt epoch cancel this book's open universe BUY orders (never sells/stops; retry; re-arm).
7. Research: lstm_numpy_serve.py (W8, research-only, nothing imports it — pinned) + research_engine.md § X7 and
   § trailing-denominator table (W9).

## VERDICTS / OWNER ITEMS (proof in the cited reports)
- O3 CONFIRMED (W6): breaker (:332) runs before `_manage_stops` (:347); on a −7 % gap all six resting stops are already
  filled, `emergency_flatten` finds nothing, :793-801 journals six ESTIMATED `circuit_breaker` exits at mid (no sell rows,
  no lockouts). Not class A: the fix moves 6 rows into Kelly's sample and adds lockout+cooldown. Flag design
  `TRADER_BREAKER_SERVER_FILL_ATTRIB` (default OFF, sandbox-proven: OFF matches today; ON passes the strict xfail). Same
  pattern at the remote-flatten site (:2864-2893); stablecoin flatten (:979) journals nothing.
- O2 quantified (W6): 18 order writes per restart instead of 6; per-name unprotected gap ≈ 0.5 s + 2 round trips (never a
  cycle), plus a 30 s window at the looser 6 %. Must be settled together with J11 (restart anchoring) — one anchor rule.
- O8 flip proposal (W7): candidates 300 (clears W4's 10-min census; SOL still ≥12/24 None in W3's Sunday probe) vs 600
  (clears both, oldest quote 486.7 s). Pre-registered rule: smallest T ∈ {240,300,600,900} with every name None-rate <1 %,
  longest None streak ≤2 polls, p90 spread of 180 s–T-old quotes ≤ fresh p90 + 5 bps, first-update mid move ≤ 2× baseline,
  over ≥24 UTC hours (≥6 weekday, ≥6 weekend). 24 h census RUNNING: loop PID 115632, nice 15, RSS 105 MB, hourly files
  w7/census_24h_HH.json; evaluator w7/flip_rule.py. Flip also updates LIVE_MAX_AGE_S in the census script (tripwire test).
- HALT flip proposal (W9): evidence today = 0 buys filled under a halt (flag never active in logs since 2026-04-08; no bots
  since 05-07). Rule: flip ON on the first observed buy fill inside a halt window OR first halted cycle with an open
  universe BUY; else review after 30 halt epochs. Residue that can survive a cycle: maker rung 'maker_unknown', lifecycle
  give-up, unconfirmed cancels, stock 'day' parents. Judgment: hook lives in `_entries_allowed` (buys path only) — a halt
  during a breaker stop defers the cancel; no client_order_id filter (stock bracket parents carry none) → a manual
  operator buy on a universe symbol would also be cancelled when ON.
- X7 numpy serving (W8): NOT FEASIBLE under the registered rule — max|Δ| fp32 5.7e-6 (crypto) / 6.0e-6 (stock) vs the
  1e-6 bar, which sits BELOW torch's own fp32-vs-fp64 error (8.0e-6 / 1.17e-5); op mapping exact (fp64 1.6e-14); 0 sign /
  threshold / rank changes on 1,575 + 1,632 real windows; −349 MB RSS per bot process; numpy 1-thread faster per forward.
  Owner decision: re-register 1a as "max|Δ| ≤ torch's own fp32 error AND zero decision changes" — I did not move it.
  Doc drift found: real crypto champion has 1,378,497 params (not 293,761 — that was the scout's synthetic model).
- Trailing-denominator table (W9, research_engine.md): kernel policy_exits.py:153 + stock_loop.py:1484 divide by ENTRY,
  base_loop.py:1076-1079 by HWM; gap = m·ATR·(r−1) inside the clamps, 0–133 bps on the grid, HWM form always tighter-or-
  equal. Tonight: unaffected (zero-basis book skips the ATR branch). Only loop→entry keeps labels==backtest==live without a
  retrain; the stock server trailing_stop cannot express the HWM form. Pin: kernel-vs-loop parity test (import allowed).
- W7 reported, not fixed: get_crypto_quote's `except: pass` treats an unparsable timestamp as FRESH (fail-open) — pinned
  by two existing tests, never seen live; failing closed would skip exits, so owner.

## CROSS-DEPT
- SIGNAL: bars_calendar.py gained `bars_per_day`/`TRADING_DAYS_PER_YEAR` (ENGINE edit, pins green). predict_now.py:16/:110/
  :484/:487 import torch unconditionally — relevant only if X7 is re-registered.
- INTEL: tests/README.md rows for test_engine_r2_*.py, test_lstm_numpy_serve.py, tests/fixtures/; docs/FLAGS.md
  strategy_config line refs are stale by ~35–50 lines across 38 rows (pre-existing); docs/MODULES.md row for lstm_numpy_serve.py.

## NEXT ROUND PLAN
R3-a land O3 behind `TRADER_BREAKER_SERVER_FILL_ATTRIB` (W6's sandbox patch) if the CEO agrees the flag is the right shape.
R3-b one anchor rule for O2+J11 as a flag-gated proposal with the harness quantifying order counts and stop levels.
R3-c evaluate the 24 h census with flip_rule.py when the hourly files complete (≈ 06:16Z + 24 h).
R3-d research third: scout E2/E4 follow-ups (X2 venue-of-fill vs realized slippage on journals; X4 post-fill returns).
