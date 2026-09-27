# W12 — ENGINE R3 research scout: X2 venue-of-fill, X4 post-fill returns (2026-09-27)
Appended: research/campaign_2026-09_jetson/research_engine.md § "R3 scout — E2/E4 follow-ups …" (119 lines, 6 sources W12-S1..S6).
New (measurement-only): scripts/fill_venue_slippage_report.py + tests/test_fill_venue_slippage_report.py (13 pass) + scripts/README.md row (Inventory header 24→29 .py; it was already stale at 28 rows).
No production code/tests edited. All Alpaca calls were read-only GETs on paper. No orders.

## LANDED
- X2(b) journals: n=0. 51 files, 264 buy rows, 0 sell rows. None has decision_price/fill_price/tactic (the schema postdates them), so I stopped there as instructed.
- Broker reconstruction instead: 686 filled crypto PAPER orders (2026-02-07..04-26) from /v2/orders + FILL activities, plus historical arrival quotes on loc=us AND us-1 for all 686.
  - Taker fills (n=310) price off the Alpaca `us` book. VWAP is −0.4 bps vs the us ask (median), +12.7 bps beyond the Kraken ask, and 199/199 market buys filled at/through the Kraken ask.
  - X2 rule: MAE of ½spread_us is 10.2 vs 16.2 bps for ½spread_us1, so `us` wins (≥25% margin). Every month agrees. Per symbol: BTC/ETH/LINK/DOGE → us, XRP no_change, SOL → us-1 (its us quote goes stale and overstates cost).
- X2(a) docs: orders trade on the "Alpaca Exchange" (docs + staff forum post). No order or fill field names a venue (verified on 36 order keys + 13 fill keys). Why us-1 maps to 23 states is undocumented, and state-based routing is unverified.
- X2(c) probe: Sunday 07:30 UTC live, plus Tue–Thu nights from historical quotes. The us − us-1 spread gap is structural: XRP ~38, DOGE ~32–35, LINK ~14–16, SOL ~6, BTC/ETH ~3 bps. A weekday is within 4 bps of Sunday. Max us quote age 181 s (ETH).
- X4: n_maker = 0. The bid-join ladder postdates the last bot run: no `maker-` ids, and legacy limits filled at mid +3.7 bps.
  - Descriptive result (confounded): legacy limit minus market buys, us-1 move after fill: +30 min −6.8 bps, CI [−16.1, +2.7]; all horizons' CIs include 0.
  - Markouts are reconstructable offline: Alpaca still serves Feb-2026 us-1 quotes and 1-min bars. So no loop row is needed; an optional `fill_markout` row is specified with its cost.

## PRE-REGISTERED RULES (in the append)
- X2: ≥30 taker fills; the venue whose half-spread has ≤0.75× the other's MAE vs realized slippage wins. `us` → ask-#1 census must stay `--loc us` (the default). `us-1` on LIVE → owner flag TRADER_CRYPTO_COST_QUOTE_LOC (not built). Runbook Phase 1 → feeds the Phase 5 ask-#1 ruling.
- X4: net 30-min markout (−fee); TOXIC iff n_maker≥50, n_taker≥20 and the 95% bootstrap CI of maker−taker lies wholly below 0 → owner item on the EXISTING MAKER_ENTRIES_ENABLED only. Posting is never changed; the wave-7 contrarian guard and wave-2 imbalance filters stay killed.

## FOUND-NOT-FIXED / OWNER
- Placeholder crypto tiers (liquidity.py:254, 0.10%) and FLAT_SPREAD_PCT crypto 0.10 (fees.py:55) are below realized paper round-trip taker crossing: LINK 30, XRP 26, DOGE 34 bps. Model-facing, so this is for the ask-#1 ruling (gotcha #2).
- Journal buy rows lack the broker order_id and bid/ask. Adding them is measurement-only instrumentation (ENGINE/base_loop.py:3496-3509 owner).
- CFEE implies 22/12–13.5 bps vs the published 25/15 — unexplained, n=71 orders, unverified.
- Paper ≠ live: Alpaca paper has no queue and no impact, so no paper verdict licenses a live change.
- INTEL: tests/README.md row for test_fill_venue_slippage_report.py.

## TEST RUNS
hwlock heavy: `$JPY -m pytest tests/test_fill_venue_slippage_report.py -q -p no:cacheprovider` → 13 passed. py_compile OK.
`--replay` on the scratch bundle (w12/bundle_2026-02-07_04-26.json) under both jetson py3.10 and base py3.12 reproduces X2=us, X4=insufficient_n.
