# ENGINE — consolidated owner items (kept current each round; lift verbatim into the morning brief). Updated: FINAL (wind-down), 2026-09-27 (24 items + § E standing instruments) (last gate engine-r3 GREEN)

Nothing below is shipped as behaviour. Flag-gated items are IN the tree default-OFF with byte-pinned OFF paths; evidence lives in
scratchpad/generals/engine/W*.md + w*/ and research/campaign_2026-09_jetson/research_engine.md. Harness for any re-check: tests/fake_alpaca_broker.py.

## A. Joint ENGINE→SIGNAL — crypto cost model understates realised taker cost (model-facing; gotcha #2; kill-list ask #1)
- **Constants:** fees.py:55 `FLAT_SPREAD_PCT['crypto'] = 0.10` and liquidity.py:254 `CRYPTO_TIER_DEFAULTS_PCT` (BTC/ETH 0.05, SOL/XRP/DOGE/LINK 0.10 — commented
  as "conservative PLACEHOLDERS, not measurements"). These feed `should_trade`'s cost floor, the backtest's per-trade charge (backtest.py:320/383), meta-label
  costs (meta_label.py:721/810) and the trainer's TXN_COST_PCT sibling — so a low spread admits entries and labels trades as winners that the venue would not pay.
- **Evidence (W12, real PAPER fills, read-only broker reconstruction, 686 filled crypto orders 2026-02-07..04-26; script scripts/fill_venue_slippage_report.py, bundle
  w12/bundle_2026-02-07_04-26.json, replay-reproducible):** taker fills price off Alpaca's own `us` book (median −0.4 bps vs the `us` ask; +12.7 bps beyond
  Kraken; 199/199 market buys at/through the Kraken ask). Realised round-trip taker crossing: **LINK ≈ 30 bps, XRP ≈ 26, DOGE ≈ 34** vs the modelled 0.10 % = 10 bps
  spread. The `us` − `us-1` gap is structural, not a weekend artefact (Tue–Thu nights: XRP ~38, DOGE ~32–35, LINK ~14–16, SOL ~6, BTC/ETH ~3 bps). Corroborated
  by W2's 2-min quote sample (DOGE 30–36, XRP 38–40, LINK 15–22, SOL 12 bps) and W3's Sunday probe (XRP/DOGE ~35, LINK ~19).
- **Caveat:** paper has no queue/impact; this is a floor on live cost, not a ceiling. Alpaca fee records imply 22 / 12–13.5 bps vs published 25 / 15 (n=71, unverified).
- **ASK:** rule on ask #1's crypto spread-stamp half with these numbers: (i) re-run `scripts/crypto_spread_census.py --loc us` (default) for ≥ 24 h weekday+weekend
  and let it write the census file liquidity.py already prefers over the placeholders; (ii) decide whether FLAT_SPREAD_PCT['crypto'] follows (backtest/meta-label
  parity) — a gotcha-#2 study-reset event, so it belongs in the Phase-3 rebuild bundle, not a hot edit. Decision-neutral until flipped: nothing in the tree changed.

## B. Flag-gated, in the tree default-OFF — each needs one owner word (+ proposal in the cited report)
1. **CRYPTO_QUOTE_MAX_AGE_SEC** (R2-W7; order_utils.py:116-134). Live rule: a crypto quote older than 180 s is None ⇒ that name gets NO software exit check that
   cycle (base_loop.py:1293-1295). Census: ETH 15 % None cycles in 10 min (max 227 s); Sunday SOL 18/24 polls > 180 s (max 486 s); recorded tape: ETH 264 s.
   Candidates 300 vs 600; pre-registered rule + evaluator w7/flip_rule.py; **24 h census completes ≈ 06:16Z 09-28** (hourly w7/census_24h_HH.json). ASK after that.
2. **HALT_CANCELS_WORKING_BUYS** (R2-W9; base_loop.py:2923-2997). Today a halt blocks entries only; residue buys (maker rung `maker_unknown`, unconfirmed cancels,
   stock DAY parents) can fill under a halt. Evidence today: 0 fills under a halt (flag never active since 04-08). Rule: flip on first observed fill-in-halt or
   halted cycle with an open universe BUY. Side effect ON: a manual operator buy on a universe symbol is cancelled too.
3. **BREAKER_SERVER_FILL_ATTRIB** (R3-W10; base_loop.py:76/:808-835; R4 extends to remote flatten). The breaker runs before `_manage_stops`; stops the broker
   already filled are journaled as ESTIMATED `circuit_breaker` exits (no sell rows, no lockouts, excluded from Kelly). ON re-labels only positions the broker has
   closed. Check w10/o3_flip_check.py; rule: first mis-attributed live row. Owner accepts: real fills enter Kelly; the 24 h lockout can outlast the breaker halt.
4. **RESTART_STOP_ANCHOR_DESIRED** (R3-W11; crypto_loop.py:208-225, stock_loop.py:1604-1646). ON: restart server stop == software stop (`_desired_stop_for`);
   18 → 6 order writes per zero-basis restart, J11 fills labelled 'hard' not 'trail'. Level table in W11: crypto r=1.01 old is TIGHTER by 90–98 bps; r ≥ 1.015 ON
   tighter-or-equal by 0–300 bps. Caveat: first 30 s of a zero-basis book, the 5 % stop_limit's limit (0.931) would not fill on a −7 % gap where today's 6 % (0.921)
   would. J2 (native trailing re-upgrade) deliberately excluded. Policy identity — one word.
5. **BARS_PER_YEAR_MEASURED** (SIGNAL's flag; ENGINE lockstep R2-W7 in volatility.py:236/:756). ON: only GARCH-sourced stock sizing rescales (×0.654); HAR-sourced
   invariant. Flip only in the SIGNAL gotcha-#2 event.

## C. Owner decisions without a flag yet (proof in the cited report; nothing changed)
6. **Zero-basis positions (tonight's six, avg_entry_price=0)** — R1-W1. Policy degrades to a pure 5 % HWM trail (no hard stop, no TP), `_book_stop_risks` = 0 for
   $121.8k, zero-basis exits journal P&L 0.0 (dilutes Kelly). Proposed flag `TRADER_ZERO_BASIS_MARK_ANCHOR` (anchor entry at the mark for entry ≤ 0). The cap
   no-op and the $0-stop-at-buy were fixed as class A; same cap defect remains in stock_loop.py:271/:970.
7. **Breaker fires before the trail** on the inherited book: account breaker at ≈ −4.09 % across the book (before the 16:05 ET baseline reset) flattens all six
   before any 5 % trail acts (R1-W1). Whether that ordering is intended is a policy question.
8. **Cash-starved entries retry every cycle** with no cooldown/count/row (R1-W1; proposed `TRADER_BUY_REJECT_BACKOFF`). Today $93 cash ⇒ 2 rejected orders per
   qualifying name per 30 s until cash exists.
9. **G2-2 crypto tick** (R1-W2): price_increment 1e-9 on all six; 4-dp rounding puts the DOGE maker rung 1.5 bps BEHIND the bid it claims to join, can cross the
   spread at tight quotes, and the (OFF) IOC cap is breached on XRP (+66 bps vs 40). Design `TRADER_CRYPTO_FINE_TICK` in W2 report; needs ONE 9-dp paper order.
10. **liquidity.py:243 NaN variance → 0.0** (W2-F1): silences the fabricated-spread warning; wrong under SPREAD_FILL_V2. One-line fix in w2/W2-F1.diff (model-facing ON path).
11. **Trailing-denominator asymmetry** (R2-W9 table in research_engine.md): kernel + stock_loop divide by entry, base_loop by HWM; gap = m·ATR·(r−1) inside the
    clamps, 0–133 bps; loop→entry is the only fix keeping labels == backtest == live (stock server trailing_stop cannot express the HWM form). Tonight unaffected.
12. **Supervision never gives up** (R1-W3/W5): 60 s bot restart with no backoff (1,778 identical restarts historically); bot crash loop invisible to the systemd
    watchdog. R4-W14 is producing the replayed backoff design (`BOT_RESTART_BACKOFF`, default OFF) — numbers are policy.
13. **User-mode systemd unit** ready, NOT installed (R1-W5): `setup_jetson_system.sh --user --print-unit` → `--user` → `loginctl enable-linger kyle` →
    `systemctl --user start trader`. No network-online ordering in user mode; never also start bots by hand/GUI.
14. **X7 numpy LSTM serving** (R2-W8): −349 MB RSS per bot process, 0 decision changes on 3,207 real windows, op mapping exact to 1.6e-14 — but the registered
    1e-6 fp32 bar sits below torch's own fp32 noise (8e-6 / 1.2e-5). ASK: re-register 1a as "max|Δ| ≤ torch's own fp32-vs-fp64 error AND zero decision changes"
    or park. lstm_numpy_serve.py stays research-only (pinned).
15. **Stock restart stop is DAY TIF** (stock_loop.py:1639) — no server stop the day after a restart-with-positions (R3-W11 residual); restart runs with
    macro_regime None (multiplier applied from cycle 1); stablecoin flatten journals nothing (R4-W13 adding the row, journal-only).
16. **Remote flatten** shares O3's mis-attribution (R4-W13 extending the flag); **J1** crypto add-on replaces the Position and drops the old resting stop id
    (self-heals next cycle; a fill inside the window is missed) — needs a paper-order test.

17. **No notify channel is configured** (R4-W14): `notify.py:259` returns early with no `TRADER_TELEGRAM_*`/`TRADER_WEBHOOK_URL` in .env, so every breaker/crash/halt
    alert on this box is SILENT (232 alerts would have fired in April's crash loop; 0 sent). One-minute owner action; the highest-value item on this list.
18. **Crash-loop backoff** (R4-W14, replay of the real 1,778-restart history): backoff 60→960 s alone avoids 89.5 % (fails the ≥ 90 % bar); backoff + give-up after
    5 identical crashes in 60 min avoids 99.3 % (12 restarts, 3 critical alerts, gives up ≈ 15 min into each episode; distinct signatures still restart within 60 s).
    Owner choices: permanent give-up vs a 3,600 s parked probe (90.8 %); flag `BOT_RESTART_BACKOFF` ← `TRADER_BOT_RESTART_BACKOFF` (run_pipeline env pattern).
    Note: since edf3151 (2026-06) cycle errors are caught in-process (base_loop.py:347-363) — the April crash would now repeat every cycle forever with a deduped
    warning; companion alert-only proposal: one critical alert after 20 identical consecutive cycle errors. In combined mode a stock STARTUP crash still takes the
    crypto loop down (run_bots.py:163-175/:228) and each restart cancels/re-places the six crypto stops.
19. **Drawdown ladder findings** (R4-W14, drawdown.py + real daily equity 02-24→05-07: max DD 6.13 %, ladder never < 1.0, 0 breaker trips; untended 52 days at
    rung 0.25 with 3 days both mechanisms active): B1 the 0.1 floor on the combined sizing multiplier masks the 20 % rung (acts as 0.255); B2 each book keeps its
    own peak and the stock book samples equity only in RTH, so the books can sit on different rungs (proposal `DD_PEAK_ACCOUNT_SHARED`, default OFF, after X11);
    B3 sizing can run on the $100k placeholder for up to 10 cycles when account reads fail at startup (already an owner note at base_loop.py:1175-1179).

20. **LLM still runs under the halt flag** (R5-W16 live evidence): halt blocks entries only; the analyst still fires every 600 s (base_loop.py:162) and a
    2-strike veto can still SELL (:465, :2190) — $0.0126 spent so far in exits-only mode, incl. one truncated response billed then discarded. Whether a
    halt should also silence the LLM (spend) and/or its veto sells (exit policy) is an owner ruling; exits-only mode tonight is running with it ON.
21. **Test runs pollute the production `logs/trader.log`** (R5-W16, class D, cross-dept): 33,452 lines since 03:09 are almost all pytest fakes;
    log_config.py:21-25 hard-codes the path with no pid → live forensics impossible. Proposal: `TRADER_LOG_DIR` env read at log_config.py:21, set to a
    tmp dir in tests/conftest.py (INTEL owns conftest; ENGINE owns log_config). Not shipped: touches every test process's log path.

22. **63 phantom `broker_stop` exits in trade_memory.json — CONFIRMED by the dry-run audit (R7-W21, 878 broker orders):** none has a sell fill within
    5.35 days; none of the 123 rows carries `estimated` (older schema) so Kelly counts them: crypto Kelly 0.1995 on 73 rows (63 phantom, avg +2.33 %) ⇒
    **a 1.5× crypto entry-sizing multiplier TODAY** (base_loop.py:2587-2591; KELLY_SAMPLE_GATE False at strategy_config.py:724). After repair only 10 real
    rows remain (< 50 minimum) ⇒ Kelly off, 1.0×. ASK (changes sizing — founder): apply `w21/repair_real.json` (63 ids → estimated=True; script has NO
    write mode) OR set KELLY_SAMPLE_GATE=True. 12 older `desync` rows also lack the flag (reported separately, not in the list).
23. **Zero-basis root cause is the broker's asset-id split** (R6-W19): all six positions sit on an `asset_id` none of our 1,320 orders ever used (forum
    Aug-2025 incident pattern; likely restore event 2026-08-13 when equity printed == cash). UNVERIFIED and material: whether a sell on the current id
    reduces the old-id position. Read-only census (R7-W21, 14:05Z): for EVERY name the position sits on one asset_id while its resting stop and the
    current /v2/assets id use another (BTC 64bbff51 vs 276e2673, DOGE a3ba8ac0 vs 03e005f7, ETH 35f33a69 vs a1733398, LINK 71a012ba vs faf30512,
    SOL 1cf35270 vs 9226ef75, XRP 88a31675 vs 85cbcac6); the stops DO reserve the old-id qty (qty_available 0). Whether a FILL reduces it stays
    unverified — the next `scripts/paper_quirk_census.py` row after the first real stop fill will show it. No probing order was placed.
24. **Unhandled broker failure modes H1–H5** (R6-W19, harness tests specified, not written): H1 position hidden one cycle → false desync drops it
    untracked with no stop; H2 equity==cash day → breaker sees a 99.9 % drawdown, flattens nothing, `positions.clear()` (base_loop.py:854), positions
    return untracked; H3 `filled_avg_price` None + broker "0" → stop at $0; H4 add-on with resting stop → new stop rejected, `stop_order_id=None`
    blinds fill detection one cycle (crypto_loop.py:201); H5 `avg_entry_price` null (not "0") → legacy SDK drops the position at startup
    (order_utils.py:1224/:1254-1256), base_loop.py:3563 raises after a fill, alpaca-py shim raises in get_position (alpaca_compat.py:65) → false DESYNC.
    H5 is a class-D candidate for R7; H1/H2 need an owner ruling on "hidden position" semantics. X12 paper-quirk census (scripts/, read-only, 900 s) proposed.

## D. Cross-department (for the CEO to route)
- SIGNAL: ≥ 40 bad-print hourly bars in training_data.parquet (Low ≥ 15 % under Open, Close ≈ Open; 11 carry hard_stop labels; inflate the Parkinson HAR cap) — W3 X6.
  predict_now.py imports torch unconditionally (:16/:110/:484/:487) — only if X7 is re-registered. bars_calendar.py gained `bars_per_day` (ENGINE edit, pins green).
- INTEL: tests/README.md rows for test_engine_r{1,2,3}_*.py, test_lstm_numpy_serve.py, test_fill_venue_slippage_report.py, test_crypto_quote_staleness_census.py,
  test_setup_jetson_user_unit.py, tests/fixtures/; docs/FLAGS.md strategy_config line refs stale across ~40 rows (pre-existing); MODULES.md row for lstm_numpy_serve.py.

## E. Standing instruments the founder can run without an agent (all read-only)
- **24 h crypto quote-staleness census** (owner item B.1 / O8): loop PID 115632 (`w7/census_loop.sh`, nice 15, ~105 MB) writes hourly files
  `<scratchpad>/generals/engine/w7/census_24h_HH.json` until ≈ 06:16Z 2026-09-28. Evaluate with the pre-registered rule:
  `python3 <scratchpad>/generals/engine/w7/flip_rule.py <scratchpad>/generals/engine/w7/census_24h_*.json` (prints the smallest passing T ∈ {240,300,600,900}
  or "keep None"); per-file tables: `scripts/crypto_quote_staleness_census.py --replay <file> --thresholds 180,240,300,600`. Flipping = set
  `CRYPTO_QUOTE_MAX_AGE_SEC` in strategy_config.py AND `LIVE_MAX_AGE_S` in the census script (tripwire test).
- **Live-bot vs harness check:** `tests/test_engine_r5_live_invariants.py` (fixture excerpt) and the replay harness `tests/fake_alpaca_broker.py`.
- **Breaker mis-attribution check** (B.3): `<scratchpad>/generals/engine/w10/o3_flip_check.py` (read-only GET orders vs trade_memory rows).
- **Fill venue / markout:** `scripts/fill_venue_slippage_report.py --pull … / --replay …` (X2/X4 rules in research_engine.md § R3).
- **Phantom exits (#22) and paper-quirk census (X12/#23):** `scripts/trade_memory_phantom_audit.py` (dry-run only, no write mode) and
  `scripts/paper_quirk_census.py` — landed by W21 in round 7 if its report exists (W21_phantoms_census.md); otherwise see w21/ for the drafts.
- Research log (append-only, cited): research/campaign_2026-09_jetson/research_engine.md; per-change log: research/campaign_2026-09_jetson/CHANGELOG.md.
