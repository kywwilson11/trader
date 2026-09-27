# W2 — cost kernels + G2-2 crypto tick (ENGINE r1, 2026-09-27) — Scratch: generals/engine/w2/ (crypto_asset_meta.json, quote_samples.json, repro_ar_nan.{py,out}, W2-F1.diff, G2-2_applied.diff, sandbox/)

## LANDED
- NEW tests/test_engine_r1_cost_kernels.py (83 tests: 68 pass, 15 xfail). No production file edited, so no mutation check was needed. Discriminator check: in sandbox/, with G2-2.diff + W2-F1.diff applied, `--runxfail` gives 83/83 pass. So every xfail flips when the fix lands.
  Covers: units (a $10k crypto taker round trip costs $50 = 2x$25; get_quote->should_trade->required_edge chain; impact; bidask fraction->percent; backtest $60 charge; short_cost base; execution_report fee identity); monotonicity (spread, taker fee, maker share, notional/ADV/k, the impact cap, amihud, VIX code); identity (zero fee + zero spread = 0 exactly, RT = 2x one-way, linearity, branch selection); parity (per_bar vs scalar, vector vs scalar impact, should_trade notional-share delta vs fees blend on a grid). Also 8 degenerate frames x {bidask, fallback} x {V1, V2}: never NaN, inf or negative. Seeded sweeps only, Mac-safe.

## FOUND-NOT-FIXED
- **W2-F1 · liquidity.py:243 · class A, but its flag-ON path is model-facing, so it goes to the owner.** `_abdi_ranaldo_rolling` sets `out[t] = ... else 0.0`, which turns a NaN variance (zero, NaN or negative price in the window) into 0.0 instead of NaN. Repro in repro_ar_nan.out: one zero print gives 20 raw 0.0 bars.
  - V1 (default): the stamped values are identical (clip-to-floor and fill-to-floor both give 0.02). But raw_nan beyond warmup reads 0 instead of 20, so the "mostly fabricated" warning (liquidity.py:207) is silenced.
  - TRADER_SPREAD_FILL_V2: these bars are stamped at 0.02 instead of FLAT 0.05. That breaks V2's own contract (liquidity.py:78).
  - Fix (one line, W2-F1.diff): `... if val > 0 else (np.nan if np.isnan(val) else 0.0)`. Pinned by the xfail test `test_fallback_corrupt_window_pays_flat_under_v2`.
  - Reached only if bidask raises. bidask 2.1 IS installed in the jetson env now, so the brief's "not installed" is stale; the fallback was forced via sys.modules.
- **G2-2 (owner item) · order_utils.py:222/224, :447 (maker rung), :233 (_round_price_band crypto) + crypto_loop.py:116 (not my file).**
  - (a) Live read-only get_asset at 05:20:53Z: all six coins have price_increment = 1e-9 and min_trade_increment = 1e-9. min_order_size: BTC 0.000011834, ETH 0.000370014, SOL 0.008224698, LINK 0.070521861, XRP 0.655737704, DOGE 10.374520178.
  - (b) Tonight's quotes over a 2-minute sample: DOGE spread 30-36 bps, XRP 38-40, LINK 15-22, SOL 12.
    - DOGE maker rung: bid 0.0959143 posts at 0.0959, i.e. -1.49 bps BEHIND the touch it claims to join (0.09581 -> -1.04 bps).
    - XRP rung: 1.51194 -> 1.5119 (-0.26 bps). BTC, ETH, SOL and LINK quotes carry <= 4 dp, so no effect.
    - DOGE compute_limit_price at a 30.3 bps spread: the intended ±3.03 bps offset becomes +4.20 (buy) and -6.21 bps (sell).
    - On DOGE, 4 dp = 10.4 bps per tick. The tight-spread 5 bps offset therefore lands anywhere in [-0.2, +10.2] bps: it can round to zero.
    - The rung crosses the spread when ask - bid < the upward rounding. Example: bid 0.09577 / ask 0.09578 posts 0.0958 >= ask, so it fills as taker at 25 bps instead of 15. Tonight's spreads were too wide for that; how often it happens is unverified.
    - Flag-gated IOC (TRADER_IOC_ENTRY_CAP, default OFF), _round_price_band crypto at 2 dp: XRP ask 1.51 with a 40 bps cap posts +66.2 bps (the cap is breached); ask 1.0149 posts +50.3 bps.
  - (c) The test: the `test_g2_2_*` tests use the recorded metadata as an inline fixture. Four of them are xfail with 'G2-2 owner item': rung joins the bid exactly, rung is never marketable, rounding is never coarser than price_increment/2, and the IOC stays inside the cap.
  - Proposed flag design, default OFF: `CRYPTO_FINE_TICK = os.getenv('TRADER_CRYPTO_FINE_TICK','0')...` (2026-08 parse family) in order_utils.
    - `_round_price_band` crypto: `round(px, 9)` when ON.
    - Maker rung: `_round_price_band(bid,'crypto')` when ON, else `round(bid,4)`.
    - compute_limit_price gets a `price_dp=4` kwarg; place_limit_order (crypto-only in practice) passes 9 when ON. Stock pricing then stays byte-identical even with the flag ON, and test_order_utils::test_rounds_to_four_decimals keeps passing.
    - crypto_loop.py:116 `round(mid*0.9995, 4)` -> the same flag (its owner).
  - The hunt's G2-2.diff (no flag, 9 dp) is equivalent when ON. Before flipping, the owner must confirm with one 9-dp paper order that Alpaca accepts it; the asset metadata alone does not prove it.

## VERIFIED-CLEAN
- Every cost is in PERCENT end to end: fees.py:246/252; order_utils.py:170 (spread_pct), :997; backtest.py:320/383; meta_label.py:721/810; decision_report.py:216; short_cost.py:76; objective_utils.py:179; execution_report.py:163-164. No bps/fraction mix-up found. Two copies are already pinned elsewhere: hypersearch TXN_COST_PCT (0.11 vs 0.113, within the 0.005 tolerance in test_fees_v3) and meta_label's inline flat spread (pinned in test_review_b10).
- fees.py has no volume-tier lookup (tier 1 only), so the tier-monotonicity invariant is vacuous; that fact is pinned by a test. Unknown or mis-cased asset labels price as stock and WARN. All producers emit lowercase literals.
- execution_policy returns offsets, not costs. Its post offset (0.4x half-spread, inside from the bid) and compute_limit_price's (0.2x half-spread, from mid) are different quantities, not duplicated arithmetic.
- The should_trade notional-share delta equals fees' direct blend exactly, including clamping of shares outside [0,1].
- amihud_illiq is never negative or inf; vix_regime_code is monotone.

## TEST RUNS (each via hwlock heavy, CUDA_VISIBLE_DEVICES='', one file per process)
- `$JPY -m pytest tests/test_engine_r1_cost_kernels.py -q -p no:cacheprovider` -> 68 passed, 15 xfailed. Sandbox with both patches, `--runxfail` -> 83 passed.
- Pins, all green: fees_feedback 8, fees_v3 29, liquidity 13, liquidity_v3 28, cost_regime 9, cost_regime_v3 31, execution_policy 9, execution_policy_v3 33, order_utils 16, order_utils_v3 40, ioc_helper 9, c26_T7 37, market_impact 8, cost_per_bar 4.
- Disclosure: the first read-only get_asset fetch (tiny, about 1 s) ran WITHOUT the arbiter by mistake. The quote sampler ran under it. No orders were placed.
