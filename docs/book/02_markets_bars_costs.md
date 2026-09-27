# Chapter 2. Markets, bars and costs

*Why a crypto trade must be predicted to move more than 1.2% before this system will even consider
it, and what last night's fill study says about whether even that is enough.*

---

## 1. The idea in plain words

A market is a place where buyers and sellers leave prices. At any moment there is a best price
someone will pay (the **bid**) and a best price someone will accept (the **ask**). The ask is always a
little above the bid. The gap is the **spread**, and the point halfway between is the **midpoint**
("mid"). If Bitcoin shows a bid of $59,990 and an ask of $60,010, the spread is $20 and the mid is
$60,000. We usually express the spread as a percent of the mid: $20 / $60,000 = 0.033%, or 3.3
**basis points** (a basis point, "bp" or "bps", is one hundredth of one percent, 0.01%).

If you buy at the ask and immediately sell at the bid, you lose the whole spread without the price
having moved at all. That is the first cost of trading. The second is **fees**, which the broker or
exchange charges per order. The third is **slippage**: the difference between the price you expected
(say, the mid when you decided) and the price you actually got.

A **bar** is a summary of trading over a fixed period. An **hourly bar** records four prices, the
**open** (first trade of the hour), **high**, **low** and **close** (last trade), plus the **volume**
traded. Together these are called **OHLCV**. This system thinks only in hourly bars. It never looks
at individual trades or the order book.

Two kinds of order matter here:

- a **taker** order crosses the spread and fills immediately against someone else's resting order
  (a market order, or a limit order priced at or through the far side);
- a **maker** order rests on the book at your chosen price and waits for someone else to come to you.
  Makers usually pay lower fees, but they may never fill, and when they do fill it is often because
  the price is about to move against them (this is called **adverse selection**).

## 2. Why it matters financially

Costs are paid on every trade, win or lose, while edge is only an average. That asymmetry decides
almost everything about which strategies can work.

**The crypto arithmetic.** At Alpaca's tier-1 crypto schedule a taker pays 25 bps per side
(`fees.CRYPTO_TAKER_BPS = 25.0`). A round trip (buy then sell) therefore costs 50 bps in fees. Add a
spread allowance of 0.10% (`fees.FLAT_SPREAD_PCT['crypto'] = 0.10`) and the model's cost of one round
trip is

  0.25% + 0.25% + 0.10% = **0.60%** of the position.

So a trade whose true expected move is +0.50% loses 0.10% on average. A strategy with a genuine,
repeatable +0.50% predictive edge per trade, which would be a very good forecaster, is a losing
strategy in crypto at these fees. This is the "0.6% edge" you will see quoted all over the repo: it
is the **break-even** move for a crypto trade.

**The stock arithmetic.** US stocks at Alpaca carry no commission, only tiny sell-side regulatory fees
(`fees.STOCK_REGULATORY_BPS = 0.3` per round trip) plus a slippage allowance of 3 bps per side
(`STOCK_SLIPPAGE_BPS_PER_SIDE = 3.0`) and a default spread of 0.05%
(`FLAT_SPREAD_PCT['stock'] = 0.05`):

  0.003% + 0.03% + 0.03% + 0.05% = **0.113%** per round trip.

Stocks are about five times cheaper to trade than crypto here. That single fact explains a lot of
last night's results.

**Why demand twice the cost.** A forecast is noisy. If you admit every trade whose predicted move is
just above break-even, half of them will in reality fall below it, and the average admitted trade
will barely pay its costs. The system therefore demands that the predicted move be at least **twice**
the round-trip cost before entering (`fees.MIN_EDGE_MULTIPLE = 2.0`). For crypto that is a 1.20%
**admission floor**; for stocks, 0.226%. A 2026 hourly-Bitcoin study (Bysik and Ślepaczuk, arXiv
2606.00060) independently found that requiring the predicted move to exceed a fixed 2.0 times cost
"sharply reduces turnover and restores profitability," which the campaign notes as outside support
for this exact multiple (`research/campaign_2026-09_jetson/research_signal.md`, SCOUT-4 T1).

**What goes wrong without it.** Before the cost model was rebuilt, the training objective assumed a
5 bp round trip (`scripts/hypersearch_v2.py` comment above `TXN_COST_PCT`), about a tenth of crypto
reality. It selected models whose "edge" could not survive live fees. The `fees.py` module docstring
records the older live gate compared predictions against the spread alone, "admitting structurally
negative-expectancy crypto trades."

## 3. How this system does it

### Bars: closed, aligned, point-in-time

- `market_data.drop_forming_bar` removes the hour that is still in progress. The model only ever sees
  **closed** bars, because a half-formed bar's "close" is not a close at all.
- Crypto bars come from Alpaca with a yfinance fallback; stock bars come from Alpaca, with yfinance
  used only as a fallback because its hourly stock bars are stamped on the half hour (:30) while
  Alpaca's are on the hour (:00) (`docs/MAP.md` §4a).
- Crypto has 24 x 365 = 8,760 hourly bars per year. The trainer annualizes stocks with 1,638 bars per
  year (`hypersearch_v2.BARS_PER_YEAR`), which is 6.5 regular-session hours on 252 days. Last night's
  campaign measured that the rebuilt stock store actually holds extended-hours bars, about 3,827 per
  year (`bars_calendar.py`, flag `BARS_PER_YEAR_MEASURED`, default OFF). This matters because a
  Sharpe ratio is annualized with the square root of bars per year; using the wrong count mis-scales
  every stock Sharpe by about sqrt(3827/1638) = 1.5.

### One cost function for everyone

The whole system asks a single module what a trade costs:

- `fees.round_trip_cost_pct(asset_type, spread_pct, maker, live)` returns the raw cost **charged to
  P&L**: fee constant plus the full spread, charged once per round trip. For crypto the entry fee is
  taker (25 bps) unless the maker flag is set or, in the live gate only, blended by the recent share
  of entries that actually filled as maker (`fees.crypto_entry_fee_bps(live=True)` reads the journals;
  with fewer than 30 recent entries it assumes all taker). The exit side is always priced taker.
- `fees.required_edge_pct(...)` returns the **admission floor**: that cost times
  `MIN_EDGE_MULTIPLE`, resolved at call time so every caller moves together.

The docstring of `fees.py` calls this a "cost-multiple ladder": 1.0x cost is charged to P&L in
backtests and meta-labels, 2.0x cost is the entry floor, and an extra 1.5x headroom is declared for a
future passive-order tactic. Two deliberate copies exist and are pinned by tests rather than imported:
the trainer's `TXN_COST_PCT = {'crypto': 0.60, 'stock': 0.11}` and `backtest.SPREAD_PCT`.

### The live cost gate

In the live loop, `order_utils.should_trade(pred_return, spread_pct, asset_type=...)` computes the
floor from the **live quoted spread** (`required_edge_pct(..., live=True)`, the only production
`live=True` caller) and rejects the candidate if the predicted move does not exceed it. The rejection
is journaled as a `cost_floor` skip with the spread at that moment, so the decision report can later
price what was given up (`base_loop._execute_buys`).

Quotes themselves are sanity-checked in `order_utils.get_quote`: a quote older than 180 seconds, or
non-positive, or NaN is rejected, and a missing quote means no entry (fail-closed).

### Estimating the spread from bars: EDGE

Historical quotes are not always available, but historical OHLC bars are. The **EDGE** estimator
(Ardia, Guidotti and Kroencke, *Journal of Financial Economics*, 2024) infers the effective spread
from the pattern of open, high, low and close prices: roughly, bid/ask bounce leaves a fingerprint in
how closes relate to the surrounding highs and lows. `liquidity.edge_spread_series(ohlc_df,
window=35)` computes a trailing, per-bar estimate with the `bidask` package and clips it to
[`SPREAD_FLOOR_PCT = 0.02`, `SPREAD_CAP_PCT = 1.50`] percent. The stock harvest stamps it into the
column `Eff_Spread_Pct` so that backtests charge each historical trade the spread of its own time and
name rather than one flat number (the two-tier cost model: the flat spread sets the admission floor,
the per-bar spread is what gets charged, `docs/MAP.md` §4c).

An honest limit, from the 2026-08 literature pass (`research/campaign_2026-08/02_research.md` B05.1):
EDGE's statistical noise on hourly bars creates a spurious floor of roughly 0.10 to 0.20% for
mega-cap stocks, five to ten times their true 1 to 5 bp spreads. That errs on the conservative side
(it overstates cost), but it narrows the tradable universe. The fix, estimating from one-minute bars,
is built dark behind `TRADER_STOCK_MINUTE_EDGE` and awaits an owner ruling. Crypto is not stamped at
all today; it uses the flat 0.10%.

### Order placement

Crypto entries use a **maker ladder** when `MAKER_ENTRIES_ENABLED = True`: the bot first joins the
bid and waits up to `MAKER_STAGE_TIMEOUT = 25` seconds per rung (two rungs), then falls back to a
taker order (`order_utils.place_maker_buy`). Stock entries are **bracket** limit orders that carry a
stop-loss leg and a take-profit leg with them. `execution_report.py` measures **implementation
shortfall**, the difference between the price at the moment of decision and the realized fill
(Perold, 1988).

### A worked example

It is 14:00 UTC. The model predicts that ETH will rise 0.90% over its horizon. The live quote shows
a spread of 0.024% (2.4 bps, typical of ETH on Alpaca, see section 4).

1. Round-trip cost, all taker: 0.25 + 0.25 + 0.024 = 0.524%.
2. Admission floor: 2 x 0.524 = 1.048%.
3. The prediction of 0.90% is below 1.048%, so `should_trade` returns False and the loop journals a
   `cost_floor` skip.

The same forecast on a US stock with a 0.03% quoted spread faces a cost of 0.063 + 0.03 = 0.093% and
a floor of 0.186%, which 0.90% clears easily. The model is equally "confident" in both cases; only
the market's price of admission differs.

## 4. What the evidence says so far

**The admission floor and the model's own scale.** Last night's diagnosis of the crypto search
(DECOMP-1, `research/campaign_2026-09_jetson/README.md` §4) found that 44 of 48 fold Sharpes were
negative and that **every threshold the search tried was below the live 1.2% floor**. In other words
the search was selecting models whose trades would be charged more than they could earn. SCOUT-4
(`research_signal.md` T1) confirmed the mechanism in code: the trainer scores "pred > threshold" with
a cost of 0.60%, while the live book and the backtest require "pred >= max(threshold, 1.20%)." The
legacy threshold range [0.05, 1.0] puts 100% of crypto draws below the floor by construction. The
Phase-3 bundle now switched on (`OBJECTIVE_V3 = True`) re-anchors the searched range to 0.8x to 2.5x
the floor (`objective_utils.v3_trade_threshold_range`); for stocks that is 0.18% to 0.57%, and the
stock retrain running this morning is indeed drawing thresholds in that band.

**The real price of crossing in crypto.** The ENGINE department reconstructed **686 real paper crypto
fills** from February to April 2026 using the broker's own records and historical quotes on two
Alpaca venues (`research_engine.md` §X2). Findings:

- Market-order fills sit at the Alpaca `us` venue's touch (median 0.4 bps better than its ask), and
  about 12.7 bps beyond the Kraken-routed `us-1` ask. Paper crypto is priced off the `us` book.
- The `us` book's spreads are much wider than `us-1` for some coins, and the gap is structural, not a
  weekend effect: XRP about 39 bps versus under 1 bp; DOGE about 33 versus under 1; LINK about 20
  versus 4; BTC and ETH only 2 to 4 bps.
- Measured round-trip crossing cost (twice the median taker slippage): **LINK 30 bps, XRP 26 bps,
  DOGE 34 bps**, against the model's flat 10 bps spread allowance.
- The fee records imply a median of about 22 bps on market orders and 12 to 13.5 bps on limit fills,
  not the published 25/15. The report marks this "unexplained, unverified" and does not use it.
- Maker fills: zero. The maker ladder was added after the last time the bots ran, so there is no
  evidence yet on whether maker entries save money after adverse selection.

The honest summary: for BTC and ETH the cost model is about right; for the smaller coins it
understates the true spread cost by a factor of two to three. Combined with the search result, this
is why the founder directed the effort toward the stock book, where the cost of admission is about a
fifth as large. These are paper fills, so none of this licenses a change to live cost settings; the
study's pre-registered rule says to rerun it on at least 30 live fills if the account ever goes live.

## 5. Further reading

- Larry Harris, *Trading and Exchanges: Market Microstructure for Practitioners*, Oxford University
  Press, 2003. Chapters on bid/ask spreads, order types and transaction costs.
- David Ardia, Emanuele Guidotti and Tim A. Kroencke, "Efficient Estimation of Bid-Ask Spreads from
  Open, High, Low, and Close Prices," *Journal of Financial Economics* 161, 103916, 2024. The EDGE
  estimator used in `liquidity.py`.
- Richard Roll, "A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market,"
  *Journal of Finance* 39(4), 1984. The original idea that bid/ask bounce is visible in prices.
- André F. Perold, "The Implementation Shortfall: Paper versus Reality," *Journal of Portfolio
  Management* 14(3), 1988. Why the price you decided at is not the price you get.
- Andrea Frazzini, Ronen Israel and Tobias J. Moskowitz, "Trading Costs," SSRN working paper 3229719,
  2018. What trading really costs a large institution, measured from $1.7 trillion of executions.
