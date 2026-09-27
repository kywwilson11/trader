# ENGINE — ROUND 1 (2026-09-27 00:15–01:40) — general: Fable; workers W1–W5 (Opus)

Worker reports: <scratchpad>/generals/engine/W{1..5}_*.md; proofs in w1/..w5/. Real paper-account snapshot
(read-only, 00:20): generals/engine/account_snapshot.json — cash $93.63, equity $121,930, 0 open orders, six
crypto positions all avg_entry_price=0 / cost_basis=0.

## LANDED (failing-before/passing-after tests; gate engine-r1 GREEN 01:07 CDT: 4771 passed, 7 skipped, 18 xfailed, 0 failed)
1. base_loop.py zero-basis fixes (W1, class A, byte-identical for entry_price>0):
   - `_cap_value` (:2697) used by the per-symbol cap at :2682 (`_compute_position_size`) and :3026 (`_execute_buys`):
     unknown basis valued at qty×HWM instead of $0. Before: MAX_NOTIONAL_PER_SYMBOL ($3k) admitted an ADD-ON on every
     ~$21k inherited position every 30 s cycle (with $93 cash: two broker-rejected buys per name per cycle, forever).
   - `_place_and_track_buy` (:3279-3296): a filled buy whose broker avg_entry_price is 0 takes the order's
     `filled_avg_price`. Before: resting stop computed at $0 (broker-rejected), buy row journaled fill 0.
   - `_reconstruct_positions` (:555): one WARNING per unknown-basis position (log-only).
   Tests: tests/test_engine_r1_inherited_positions.py (11; 3 fail on the pre-edit copy) + tests/fake_alpaca_broker.py
   (stdlib fake of the 10 REST methods the loops use — seed of the replay harness, suggestion (2)).
2. tests/test_engine_r1_cost_kernels.py (W2, 83 tests: 68 pass / 15 strict=False xfail, every xfail verified to
   flip with the owner fixes applied in a sandbox). Cost chain is in PERCENT end to end; no unit bug found.
3. research/campaign_2026-09_jetson/research_engine.md (W3; 16 dated sources, 7 device measurements, X1–X7).
4. scripts/crypto_quote_staleness_census.py + tests (W4, measurement-only; --replay offline).
5. scripts/setup_jetson_system.sh `--user` + `--print-unit` (W5; root path golden-pinned byte-identical;
   systemd-analyze --user verify exit 0). Docs: scripts/README.md rows + count, docs/MODULES.md row,
   docs/STATE_FILES.md §4 row. CHANGELOG.md: 5 ENGINE lines appended.

## FIRST-CYCLE NARRATIVE for tonight's inherited book (W1, proven on the fake with the real snapshot)
Startup: 6 GTC stop_limit sells 6 % under current (fallback, entry=0 → no ATR anchor): BTC 79,302.72 ETH 2,531.36
SOL 113.317 LINK 13.2446 XRP 1.4246 DOGE 0.090508. Cycle 1: breaker OK (−0.95 %); all six cancelled and re-placed at
the 5 % trail (trailing is ALWAYS active at entry 0; hard stop and TP can never fire); book stop-risk journals 0 for
$121.8k; no buys (cap now blocks add-ons; a NEW name with $93 cash would be rejected every cycle). Cycle 2: no
orders. Thereafter: stops ratchet on ≥1 % new highs; a −5 % from high exits via server stop, 24 h lockout, P&L 0.
Account-wide breaker trips at ≈ −4.09 % across the book before the next 16:05 ET baseline reset → flattens all six.

## OWNER ITEMS (proof in the cited report; none shipped — each changes a stop level, sizing, gate or journal)
- O1 (W1) Zero-basis stop/risk semantics: anchor entry at the mark for entry<=0 positions (flag design
  `TRADER_ZERO_BASIS_MARK_ANCHOR`, default OFF) — fixes both "pure trail" and "book risk 0".
- O2 (W1) Startup stop churn: 6 % fallback stop then cycle-1 cancel/replace at 5 % → 12 extra orders + a gap per name.
- O3 (W1) Breaker fires before the trail (−4.09 % book vs −5 % trail) and, on a gap where server stops already filled,
  journals 6 estimated `circuit_breaker` exits instead of the real fills (no sell rows, no lockouts).
- O4 (W1) Cash-starved entries retry every cycle with no cooldown/count/row (`TRADER_BUY_REJECT_BACKOFF` design).
- O5 (W1) Zero-basis exits journal P&L 0.0 as real trades (dilutes Kelly). Same cap defect in stock_loop.py:271/:970.
- O6 (W2) G2-2 crypto tick: price_increment is 1e-9 on all six; 4-dp rounding puts the DOGE maker rung 1.5 bps
  BEHIND the bid it claims to join, can cross the spread at tight quotes, and the (flag-OFF) IOC cap is breached on
  XRP (+66 bps vs 40). Flag design `TRADER_CRYPTO_FINE_TICK` (default OFF) in W2 report; needs ONE 9-dp paper order.
- O7 (W2) liquidity.py:243 NaN variance → 0.0 (silences the fabricated-spread warning; wrong under SPREAD_FILL_V2).
- O8 (W4/W3) 180 s quote-staleness rule: 10-min census — ETH 15 % None cycles (max age 227 s), all names 0 % at
  ≥300 s; W3's Sunday probe: SOL 18/24 polls >180 s. Each None skips ALL software exits for that name that cycle
  (base_loop.py:1293-1295). Owner needs the multi-hour rerun (command in W4 report) before picking a number.
- O9 (W5/W3) Supervision never gives up: bot crash loop invisible to the systemd watchdog (heartbeat keeps sending),
  60 s restart with no backoff (1,778 historical restarts). `StartLimit*`/backoff = owner policy.
- O10 (W5) Install path for tonight is ready but NOT RUN: `setup_jetson_system.sh --user --print-unit` → `--user` →
  `loginctl enable-linger kyle` (owner) → `systemctl --user start trader`. Never also start bots by hand/GUI.
- Cross-dept: SIGNAL — ≥40 bad-print hourly bars in training_data.parquet (Low ≥15 % under Open, Close≈Open; 11 carry
  hard_stop labels; inflate the Parkinson HAR cap) — W3 X6. INTEL — tests/README.md rows for the 4 new test files.

## REJECTED / NOT DONE
- No change to thresholds, rounding, staleness, or breaker order (all owner). G3-8 still blocked on test_review_b19.
- Suggestion (3) restart backoff: verdicts stand as owner policy (O9); no proven wrong output.

## NEXT ROUND PLAN
R2-a grow tests/fake_alpaca_broker.py into the recorded-day replay harness (StockLoop too; RTH clock; bracket legs).
R2-b prove O3's exit-attribution defect (server fill vs breaker estimate) and O2's churn as class A/D candidates.
R2-c research third: X7 numpy-LSTM serving (−250 MB RSS, |Δpred|≤1e-6 on ≥1000 real windows) feasibility spike; X1 24 h census.
R2-d trailing-denominator asymmetry: build the exact divergence table (kernel/entry vs base_loop/HWM) for the owner.
