# ENGINE r3 W11 — RESTART_STOP_ANCHOR_DESIRED (O2 + J11 + J2: one anchor rule). Scratch: generals/engine/w11/
## LANDED (default OFF; exit-level change, so flag-only)
- strategy_config.py:428-454 `RESTART_STOP_ANCHOR_DESIRED = False` (house comment); docs/FLAGS.md:88 row.
- crypto_loop.py:208-225 (flag read at call time via getattr): ON places `self._desired_stop_for(pos)[0]` through `_place_resting_stop`, which records that level in `_resting_stop_px`, so cycle 1's `_maybe_update_resting_stop` sees no ≥1 % improvement and does not churn. OFF = the legacy lines, verbatim.
- stock_loop.py:1604-1646: ON places `round(self._desired_stop_for(info)[0], 2)`, still a plain DAY `stop`.
- J2 decision: EXCLUDED. ON places no native trailing_stop and leaves the restored `trailing_activated` as-is.
  - A native trail re-anchors at the submit-time price with the entry denominator (stock_loop.py:1480-1484), so it can never equal `_desired_stop_for`.
  - It is looser by (hwm−px)/hwm whenever px<hwm, and it would re-create a cycle-1 cancel/replace.
  - With the flag True, hwm ≥ 1.01·entry, so desired = the trail level > hard. The 'trail' label is therefore correct.
- tests/test_engine_r3_restart_anchor.py (17 tests).
  - OFF: literal goldens captured from the pre-edit copies (w11/orig → w11/golden.json) for r1 zero-basis (18 writes), r2 O2 (18), crypto J11 (stop 83078.0399, fill 'trail') and stock keeper (50.47). Plus A/B == the verbatim legacy bodies over the full recorded tape (both r2 scenarios) and a stock day.
  - ON, zero-basis: 6 writes in startup + cycle 1 and 0 cancels, each at `_desired_stop_for` (5 % trail); `_resting_stop_px` == desired.
  - ON, crypto J11: the stop rests at 0.975·E, not 0.98475·E. No cycle-1 write, and the gap fill journals 'hard'.
  - ON, crypto unit and grid: classify gives 97.5→'hard' and 98.475→'trail'. On the 24-cell grid ON == desired and OFF == legacy.
  - ON, stock keeper: one plain stop at 50.50 (legacy 50.47); no trailing_stop; the flag stays True; the fill is labelled 'trail'. ON, stock J11 (hwm 50.4): 49.00 'hard' vs legacy 49.39 'trail'. Call-time flag read pinned.
- Mutation check (sandbox w11/mut, pre-edit modules): 8 FAIL (every ON/constant test), 9 pass (OFF pins, plus the J2 no-upgrade pin, which the flag does not change).
## TABLE — restart stop, legacy vs ON (Δ = legacy−ON level, bps of entry; full grid w11/table.txt)
- CRYPTO, r=1.0: identical.
- CRYPTO, r=1.01 (trail NOT armed, J11): legacy TIGHTER by +98/+97/+95/+90 bps at ATR 0.5/1/2/4 %; legacy labels 'trail', ON labels 'hard'.
- CRYPTO, armed (r≥1.015): ON tighter or equal, both 'trail'. Δ at 1.015: 0/−54/−108/−215; 1.03: 0/−58/−115/−230; 1.05: 0/−62/−125/−250; 1.10: 0/−75/−150/−300. The ATR 0.5 % column is floor-clamped, hence equal.
- STOCK, r≥1.01 (all armed, 1 % activation): ON tighter or equal, both 'trail'. Δ = 0…−8 bps at 1.01 up to 0…−80 at 1.10 (= −2·ATR·(r−1)/entry).
- STOCK J11 band is r∈(1, 1.01) only: e.g. 1.008 / ATR 1 % gives legacy 49.39 'trail' vs ON 49.00 'hard' (78 bps looser).
- Denominator: ON inherits `_desired_stop_for`'s trail denominator (HWM today, base_loop.py:1149). A later loop→entry decision (research_engine.md § W9) flows through here automatically: one anchor, one place.
## FOUND-NOT-FIXED (owner / judgment; all pre-existing, same under OFF)
- Macro: at restart `macro_regime` is None (base_loop.py:157), so the ON level ignores `stop_mult` until cycle 1. Crypto then ratchets once if the tightening is ≥1 %. The stock plain stop never re-places.
- The stock restart stop is DAY TIF (stock_loop.py:1639), so there is no server stop the next day (J15-adjacent).
- J2 residue, restored trailing=True with no saved hwm: hwm < activation, so desired = hard, but the label is 'trail' (base_loop.py:1191). J2 residue, restored False with hwm ≥ activation: still upgraded in cycle 1 to a native trail re-anchored lower.
- ON zero-basis, first 30 s: the 5 % stop's limit is 0.931, so a −7 % gap leaves it resting unfilled. Legacy's 6 % stop (limit 0.9212) fills. From cycle 1 on this is the same exposure as legacy (r1 `test_minus7_gap` pins OFF).
## FLIP PROPOSAL (RESTART_STOP_ANCHOR_DESIRED → True)
- Evidence instrument: the deterministic harness order counts (18 → 6 writes per zero-basis restart; the per-name ~0.5 s + 2-RTT unprotected window is gone) plus the level table above. No live data is needed: this is a policy identity, not a tuning.
- Pre-registered rule: flip when the owner accepts "the restart server stop equals the software stop `_desired_stop_for`". Acceptance = tests/test_engine_r3_restart_anchor.py green on the Jetson. Runbook phase: bot activation, BEFORE the first restart with open positions.
- Tonight's zero-basis book: levels are identical (5 % HWM trail both ways); the flip only removes the 12-order cycle-1 churn.
- Real-basis books: looser by ≤~1 % of entry in the unarmed band (now the validated hard stop, labelled 'hard'), and tighter by up to 300 bps once the trail is armed.
## VERIFIED-CLEAN
- `_desired_stop_for` is pure.
- `_place_resting_stop` stores the unrounded level, so ON bookkeeping == desired.
- No test pins the edited source text.
- c26_P1 and ia4 stub `_replace_protective_stops`.
## TEST RUNS (`hwlock.sh heavy engine-w11-<t> -- $JPY -m pytest tests/<t>.py -q -p no:cacheprovider`, one per process; py_compile OK)
- restart_anchor 17, r1_inherited 11, r2_replay_harness 10, c26_base_loop_functional 37, c26_T6 43, loop_fixes_2026_09 22+1s, review_b01 21, grp_loops 9, ia4_flagged 43, c26_P1 44, improve_stratcfg 10, engine_r2_flags 52, engine_r3_o3_quote 74, imports 21. All pass.
