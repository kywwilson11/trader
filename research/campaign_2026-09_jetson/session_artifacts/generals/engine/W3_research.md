# W3 research scout — ENGINE (2026-09-27)
Deliverable: /home/kyle/trader/research/campaign_2026-09_jetson/research_engine.md (267 lines, 16 sources, 7 local measurements M1–M7; scripts+outputs in engine/w3/). No production/test edits.

## LANDED
None (research only). New file above: purpose, sources table, per-topic claim/applies/experiment/kill-check.

## Ranked experiments (EV / cost)
1. X1 quote-age census (S, measurement-only). M3: Alpaca `us` crypto quote timestamps advance only on BBO change; Sun 05:24 UTC, ETH 12/24 and SOL 18/24 polls were >180 s old (max 486 s) with the spread unchanged; Kraken `us-1` was always <14 s. order_utils.py:164 then returns None, and base_loop.py:1293-1295 `continue`s past trailing/TP/vertical/signal exits for that cycle (only the resting GTC stop protects). Rule: >5% of polls >180 s for any name in a 24 h weekday or weekend run → owner item for a default-OFF liveness rule (TRADER_CRYPTO_QUOTE_LIVENESS); <1% → close.
2. X9 supervisor backoff (S, class-D candidate). M7: 1,778 restarts of one deterministic Stock-bot crash (pipeline_output.log lines 5387–9269; since-fixed fundamentals P/E format bug). run_pipeline.py:1032-1076 restarts every 60 s with no backoff and no give-up; notify dedupe 10 min ⇒ ~178 alerts. Proposed: doubling 60→960 s, give up and send one error alert after 5 identical-signature crashes in 60 min (thresholds are JUDGMENT).
3. E1.4(a) avg_entry_price=0 (S, class-A candidate). base_loop.py:3270 uses verify_position().avg_entry_price with no filled_avg_price fallback, so 0 ⇒ entry_price=0 (:3323) ⇒ crypto_loop.py:205 resting stop at $0. Alpaca paper had exactly this incident (forum, 2025-08-12→28); the API field value (0 vs null) during it is unverified. Repro: a fake verify_position returning '0'.
4. X8 user systemd unit (S, owner action). M6: systemd 249, Linger=no, polkit set-self-linger allow_any=yes ⇒ `loginctl enable-linger` needs no sudo (local override unverified); user@1000 has the memory+pids controllers delegated ⇒ MemoryMax works in a --user unit. RestartSteps needs systemd ≥254, so it is unavailable here. Acceptance: up ≤180 s after a reboot with no login; restart after kill -9; watchdog catches SIGSTOP.
5. X6 bad prints in training_data.parquet (M, cross-dept → SIGNAL). ≥40 wick-only bars with Low ≥15% under Open and Close≈Open (BTC Low $8,200 vs $65.6k on 2021-10-21; BTC ~$34.5k lows vs ~$42k Dec-23/Jan-24; DOGE 2025-10-10). 11 carry TB hard_stop labels; they also inflate the Parkinson HAR RRV max clamp (volatility.py:158-216). Rule: ≥10 confirmed by a second source → owner item for a default-OFF harvest wick filter (gotcha #2).
6. X2 us vs us-1 spread-as-cost (S build, needs ≥30 fills). `us` spreads are XRP/DOGE ≈35 bps, LINK 19 bps; Kraken's are <3 bps. Which book the orders actually fill against is unverified. Rule: the ½-spread predictor with ≥25% lower MAE vs realized slippage wins the crypto stamp (feeds kill-list ask #1).
7. X7 numpy LSTM serving (M, class-C candidate, default-OFF TRADER_NUMPY_LSTM_SERVE). M1: torch import = +331 MB RSS; model is 293,761 params (~1.2 MB). M2 replica: max Δ 1.6e-7, 54 ms/forward. qint8 is rejected (Δ up to 3.6e-3, saves ~1 MB). Acceptance: real champion, ≥1,000 windows, |Δ| ≤1e-6, zero gate flips, RSS −250 MB or more.
8. X5 resting stop_limit gap audit (S+M). The 2% limit gap is at crypto_loop.py:168. Hourly proxy post-2025-10-10: 0.20–0.55% of name-hours drop ≥4%, 0.012–0.155% drop ≥8%. Rule: ≥10% of ≥20 triggers unfilled or filled >1% below trigger → owner review.
9. X4 maker markout (S, needs ≥50 maker fills). Albers et al. 2025 found maker fill probability anti-correlated with post-fill return. This is measurement-only; the contrarian-posting guard stays KILLED [wave-7].

## FOUND-NOT-FIXED (for the general)
- #3 above (base_loop.py:3270/3323 + crypto_loop.py:205): a $0 stop on avg_entry_price=0 — ENGINE file owner.
- #1: exits skipped on stale-rejected quotes (base_loop.py:1293-1295) — this is a behaviour choice, so it goes to the owner.
- #5: bad prints — SIGNAL/data owner (harvest_crypto_data, policy_exits labels, volatility.py).

## VERIFIED-CLEAN
fees.py:40-42 (25/15 bps) matches Alpaca tier 1. The crypto order types and TIF used (limit gtc/ioc, stop_limit gtc) are supported. Paper price_increment is 1e-9, so _round_px never gets an increment rejection. Kelly drops estimated/desync rows (trading_utils.py:230-240). fundamentals.py:367 P/E crash is fixed. hw_monitor torch import short-circuits under CUDA_VISIBLE_DEVICES=''.
## NOTHING NEW (searched)
E3 vol/Kelly/drawdown: Conformal Kelly (arXiv 2608.01494) failed out of sample, which reinforces the CQR/ACI kill; the crypto-HAR 7/30 windows claim is unverified. E4: hour-of-day depth is already in 05/FR-18; weekends are under-represented in ≥4% drops (19% vs 29% of hours). Websocket limits are undocumented. TorchScript is deprecated in 2.10 (Jetson pinned to 2.8, informational).
## TEST RUNS
No pytest. hwlock heavy: torch_mem.py, np_lstm.py, wick_census.py, badprint.py (all exit 0). Light: quote_age_probe.py, loc_probe.py, asset_probe.py (GET-only Alpaca).
