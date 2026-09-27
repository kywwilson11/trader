# ENGINE — ROUND 3 (2026-09-27 02:05–03:05) — general: Fable; workers W10–W12 (Opus, ≤2 live under HW throttle)

Worker reports: <scratchpad>/generals/engine/W10_o3_flag_quote_parse.md, W11_restart_anchor.md, W12_research.md.
Gate: engine-r3 — GREEN (Sun Sep 27 02:43:57 AM CDT 2026): 5351 passed, 7 skipped, 64 xfailed, 55 warnings in 360.30s (0:06:00)

## LANDED (failing-before/passing-after; CHANGELOG.md: 4 ENGINE R3 lines)
1. O3 behind `BREAKER_SERVER_FILL_ATTRIB = False` (W10; strategy_config.py:405-427, base_loop.py:76/:808-819/:835,
   FLAGS.md:87). ON: a resting stop the broker already FILLED before the breaker flatten is journaled `server_stop` through
   the same calls `_manage_stops` uses (detect_source='breaker') with the lockout armed; OFF: zero extra broker calls,
   byte-pinned. The r2 strict xfail is now the ON test; the old pin is the OFF pin. tests/test_engine_r3_o3_quote.py.
2. get_quote fails CLOSED on an unparsable OR missing timestamp (W10, class A per the fail-closed bar; CEO-approved):
   order_utils.py:176-208 — named exceptions from a real-SDK probe (TypeError/ValueError/AttributeError/OverflowError) +
   isfinite guard; parsable forms byte-identical (aware-UTC/NY, naive, pandas, ISO 'Z', ns string, shim). Pins modernised,
   not weakened: tests/test_ia2_safety.py:399 (accepted → None), test_crypto_quote_staleness_census.py:69-70 ('ok' → 'stale'),
   stubs in test_order_utils_v3.py:173 / test_engine_r1_cost_kernels.py:89 given a fresh timestamp (assertions untouched).
   Census classifier kept in agreement with the live verdict (crypto_quote_staleness_census.py:98-101/:178/:315/:326).
3. `RESTART_STOP_ANCHOR_DESIRED = False` (W11; strategy_config.py:428-454, crypto_loop.py:208-225, stock_loop.py:1604-1646,
   FLAGS.md:88): ON = restart server stop == `_desired_stop_for(pos)[0]` for both books (18 → 6 order writes per zero-basis
   restart, 0 cycle-1 cancels; J11 restart stop = software hard stop, labelled 'hard'); OFF = legacy formula verbatim,
   golden-pinned on four harness runs. tests/test_engine_r3_restart_anchor.py (17).
4. Research: scripts/fill_venue_slippage_report.py (W12, measurement-only, 13 tests) + research_engine.md § R3 scout.

## FLIP PROPOSALS (each has instrument + pre-registered rule + phase in its worker report)
- BREAKER_SERVER_FILL_ATTRIB: instrument w10/o3_flip_check.py (read-only: circuit_breaker rows in trade_memory.json vs Alpaca
  closed sell-stop fills in [−900 s, +60 s]); rule = flip on the first mis-attributed row observed live; today 0 rows/0 trips.
  Owner accepts: real fills enter Kelly's sample; the 24 h lockout can outlast the breaker halt.
- RESTART_STOP_ANCHOR_DESIRED: policy identity (server stop on restart == software stop), acceptance = the ON tests green on
  the Jetson; phase = bot activation, before the first restart with open positions. Level table in W11: crypto r=1.01
  (trail not yet active) old is TIGHTER by 90–98 bps and mislabelled 'trail'; r≥1.015 ON tighter-or-equal by 0–300 bps.
  Caveat (W11): in the first 30 s a zero-basis book's 5 % stop_limit (limit 0.931) would not fill on a −7 % gap where
  today's 6 % stop (limit 0.921) would; from cycle 1 both carry the same exposure. J2 (native trailing re-upgrade) excluded.
- CRYPTO_QUOTE_MAX_AGE_SEC (r2): 24 h census still running (hourly files w7/census_24h_HH.json; evaluate with w7/flip_rule.py
  after ≈ 06:16Z 09-28). R3-c deferred to that time.

## RESEARCH VERDICTS (W12; paper fills only — nothing licenses a live change)
- X2 venue-of-fill: 686 real paper crypto fills (2026-02-07..04-26) fill at Alpaca's `us` touch (median −0.4 bps vs `us`
  ask; +12.7 bps beyond Kraken `us-1`; all 199 market buys at/through the Kraken ask). `us` half-spread predicts realized
  slippage better (10.2 vs 16.2 bps mean error, every month) — except SOL (stale `us` quote overstates cost; unverified).
  Venue gap is structural (weeknights: XRP ~38, DOGE ~32–35, LINK ~14–16, SOL ~6, BTC/ETH ~3 bps). Kill-list ask #1: the
  spread census stays on `--loc us` (already default, crypto_spread_census.py:126); passive-fill simulator stays dead.
- X4 maker markout: n_maker = 0 (place_maker_buy postdates the last bot run 2026-05-07). Toxic-fill rule pre-registered:
  maker−taker 30-min markout net of fees, ≥50 maker / ≥20 taker buys, 95 % bootstrap CI wholly < 0 → owner item on
  MAKER_ENTRIES_ENABLED only. Offline-rebuildable from Alpaca history; no in-loop instrumentation needed.

## OWNER / CROSS-DEPT
- Remote flatten (`_check_flatten_request`, base_loop.py:326 → :2956-2972) has the SAME O3 mis-attribution (runs first in
  the cycle); stablecoin flatten (:1050-1065) journals NO exit row and saves no state — separate owner item.
- Crypto spread assumptions look low vs realized paper taker round-trips (liquidity.py:254 / fees.py:55 flat 0.10 % vs
  LINK 30, XRP 26, DOGE 34 bps) — model-facing, part of the ask #1 ruling + gotcha #2 (SIGNAL/owner).
- Alpaca fee records imply 22 bps market / 12–13.5 bps limit vs published 25/15 over 71 matched orders — unverified.
- Journal buy rows lack broker order_id and bid/ask (base_loop.py:3496-3509) — measurement-only add, next round.
- W11 residuals (flag-independent): restart runs with macro_regime None (multiplier applied from cycle 1); stock restart
  stop is DAY TIF (stock_loop.py:1639) — no server stop next day; two J2 labelling cases.
- INTEL: tests/README.md rows for test_engine_r3_o3_quote.py, test_engine_r3_restart_anchor.py, test_fill_venue_slippage_report.py;
  harness row now "10 tests, 0 xfail"; scripts/README.md inventory now 29 .py (count line fixed twice this session).

## NEXT ROUND PLAN
R4-a journal buy rows: add broker order_id + decision bid/ask (measurement-only; enables X2/X4 on live fills).
R4-b remote-flatten mis-attribution under the SAME flag (BREAKER_SERVER_FILL_ATTRIB) + stablecoin-flatten exit rows (owner ask first).
R4-c R3-c census evaluation when the 24 files exist; R4-d research third: E3/E6 (crash-loop backoff design with evidence; drawdown ladder review).
