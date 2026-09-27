# Chapter 8. Sizing, exits and risk

*How much to buy, when to get out, and the layers of limits that keep one bad day from becoming a bad
year. Includes a cautionary tale about 63 trades that never happened.*

---

## 1. The idea in plain words

A trading system makes three money decisions per trade, and the forecast drives only one of them:

1. **Whether** to trade (Chapters 5 to 7).
2. **How much** to trade: **position sizing**.
3. **When to get out**: the **exit rules**.

Professional traders often say sizing and exits matter more than entries, and the arithmetic backs
them: a mediocre entry with disciplined sizing survives; a brilliant entry with reckless sizing can
wipe out an account on one unlucky sequence.

Terms used below:

- **Stop-loss (stop):** a price below the entry at which you sell to cap the loss.
- **Take-profit (TP):** a price above the entry at which you sell to lock in a gain.
- **Trailing stop:** a stop that moves up as the price rises (it trails the **high-water mark**, the
  highest price since entry), but never moves down.
- **Drawdown:** the percentage fall from the account's highest value to its current value.
- **Kelly criterion:** a formula for the fraction of capital to bet that maximizes long-run growth,
  given your win probability and payoff.
- **Volatility targeting:** sizing so that each position contributes roughly the same amount of
  expected fluctuation, smaller in wild assets and larger in calm ones.

## 2. Why it matters financially

**Losses are asymmetric.** Lose 10% and you need +11% to recover; lose 25%, +33%; lose 50%, +100%.
Deep drawdowns are disproportionately hard to climb out of, so a rule that cuts size as drawdown grows
protects the account's ability to recover.

**Kelly and why nobody bets full Kelly.** For a bet that wins with probability p and pays b times the
amount lost when it loses, the Kelly fraction is

  f* = p - (1 - p) / b.

With p = 0.55 and b = 1.2 (average win 1.2%, average loss 1.0%): f* = 0.55 - 0.45 / 1.2 = **0.175**,
meaning risk 17.5% of capital per bet. Two facts make that dangerous in practice. First, full Kelly
has brutal swings: a well-known property is that it has about a one-in-two chance of halving the
account at some point along the way. Second, p and b are *estimated*, and if your true edge is half
what you estimated, betting "full Kelly" on the estimate means betting twice the true Kelly, where
long-run growth falls to about zero. Hence the practitioner's habit of **fractional Kelly** (a half or
a quarter), which gives up a little growth for much smaller drawdowns (MacLean, Thorp and Ziemba,
2011).

**Stops must be sized to the asset.** A fixed 2% stop is too tight for a coin that routinely moves 2%
in an hour (you get stopped out by noise) and too loose for a sleepy utility stock. Scaling stops by
**ATR** (Chapter 4) adapts to each asset's normal range.

## 3. How this system does it

### Sizing: `base_loop._compute_position_size`

One function sizes every entry in both books. Its docstring says it replaced an "unbounded multiplier
soup" that could stack to about 32 times the base size. Each component now has one job and a hard
bound.

**Step 1, risk base.** Risk a fixed fraction of equity to the stop:

  notional = equity x `RISK_PCT_PER_TRADE` (0.005) / stop distance,

capped at `NOTIONAL_PER_SYMBOL` ($1,000 for crypto from the base loop, $5,000 for stocks from
`stock_loop.StockLoop`). The stop distance is ATR x the policy multiple / price, clipped to the
policy floor and ceiling (Chapter 4).

**Step 2, Kelly multiplier.** `trading_utils.compute_kelly_fraction` reads the book's confirmed
trades from `trade_memory.json`, needs at least 50, uses the most recent 200, **shrinks** the win rate
and payoff toward a skeptical prior of 50 pseudo-trades at breakeven (win rate 0.5, payoff 1.0),
computes Kelly, halves it, and clamps to [0.05, 0.25] (`KELLY_CAP = 0.25`). The loop maps 0.125 to a
1.0x multiplier and bounds the result to [0.5, 1.5]. Worked example with 100 trades at 55% wins and a
1.2 payoff: shrunk win rate (55 + 25) / 150 = 0.533, shrunk payoff (120 + 50) / 150 = 1.133, Kelly
0.122, half-Kelly 0.061, multiplier 0.061 / 0.125 = 0.49, clamped to **0.5x**. Notice that a
respectable track record maps to the *minimum* multiplier: reaching 1.0x requires a half-Kelly of
0.125, i.e. a full Kelly of 0.25, which is a very strong edge. With fewer than 50 trades the function
returns None and the multiplier is neutral 1.0. When VIX is above 25 or the HMM regime model says
"bear," the Kelly multiplier is capped at 1.0.

**Step 3, volatility target.** `volatility.compute_vol_adjusted_size` compares the asset's forecast
hourly volatility (HAR-RV on intraday ranges, with GARCH as fallback; `volatility.get_sigma`, output
in decimal per bar) with the book's target converted to hourly units:
`PORTFOLIO_VOL_TARGET` = 35% a year for crypto and 18% for stocks, so the crypto target is
0.35 / sqrt(8760) = 0.374% per hour. The ratio target / forecast is clamped to [0.5, 1.5]. Worked
example: SOL forecast at 0.90% per hour gives 0.374 / 0.90 = 0.42, clamped to **0.5x**.

**Step 4, the tilt product.** Every advisory input multiplies into one number (`detail` in the sizing
journal records each factor):

| Factor | Rule |
|---|---|
| signal confidence | prediction / threshold, clipped to [0.75, 1.25] |
| VIX ladder | 1.0 up to 15, 0.7 above 15, 0.5 above 25, 0.3 above 35 |
| drawdown ladder | 0.75 at 10% drawdown, 0.50 at 15%, 0.25 at 20% (`drawdown.DRAWDOWN_LADDER`) |
| macro regime | `macro_regime.sizing_mult` (stress, stablecoin peg, SPY trend) |
| correlation | haircut if correlated with current holdings |
| HMM regime | 1.2 bull, 0.3 bear, 0.5 high-volatility (`regime_detector.py`) |
| disagreement | 0.8 if the macro and HMM "votes" disagree |
| sentiment, LLM, meta | from Chapter 7 |
| book-specific | crypto: the perpetual-funding crowding tilt (`crypto_loop._extra_tilt`) |
| book volatility | account-level realized-volatility scalar (`portfolio.get_book_vol_scalar_cached`) |

The product is clamped to [0.1, `TILT_MAX = 1.30`]: boosts are capped at 1.3x, but de-risking is
honored down to 0.1x, so a drawdown ladder that says "cut 75%" is not overridden. If two or more
advisory inputs are missing (VIX, return history, correlation matrix, Kelly history), the tilt is
capped at 0.5 ("degraded mode"). An emergency (for example a stablecoin losing its peg sets the macro
multiplier to 0) returns size 0 before any of this.

**Step 5, the book risk budget.** With other positions open, the loop computes how much more stop-risk
the book can take under an equal-correlation model (`portfolio.book_risk_budget`, enforced in
`base_loop`, cap `MAX_BOOK_RISK_PCT = 0.025`). The formula solves for the new risk r that keeps
sqrt((1 - rho) x sum(r_i^2) + rho x (sum r_i)^2) at the cap. Worked example: two open positions each
risking 0.5% of equity, average correlation 0.5: the new position may risk up to about **1.90%**,
far above the 0.5% base, so the budget does not bind here; with six correlated positions it would.

**Step 6, hard caps.** The order is limited by the remaining room under `MAX_NOTIONAL_PER_SYMBOL`
($3,000 crypto, $5,000 stocks), divided by the leverage of leveraged ETFs, and dropped if below
`MIN_ORDER_NOTIONAL = $100` ("fees eat dust").

**A full worked example (stocks).** Equity $122,202. A $100 stock with an entry ATR of $0.60.

1. Stop distance 2.0 x 0.60 / 100 = 1.2%; risk notional $611 / 0.012 = $50,917; capped base **$5,000**.
2. Kelly: no confirmed history yet, so 1.0.
3. Volatility: forecast 0.50% per hour against 0.18 / sqrt(1638) = 0.445%, ratio **0.89**.
4. Tilt: confidence 0.40 / 0.30 = 1.33, clipped to 1.25; VIX 18 gives 0.7; no drawdown 1.0; HMM bull
   1.2; LLM s = 0.6 gives 1.1; everything else 1.0. Product 1.25 x 0.7 x 1.2 x 1.1 = **1.155**.
5. Sized: $5,000 x 1.0 x 0.89 x 1.155 = $5,140.
6. Cap: room under $5,000 is $5,000. **Final order: $5,000**, risking $60 (0.05% of equity).

The caps did almost all the work, which is why `docs/MAP.md` §5-10 says the book runs at roughly a
third to a fifth of its configured risk.

### Exits

- **Stops, trailing and take-profit** follow the policy in Chapter 4: for stocks, stop 2.0 ATR, TP at
  2 times the stop distance, trail 2.0 ATR arming at +1%. For the example: stop $98.80, TP $102.40,
  trail arms at $101.00.
- **Protection at the broker.** Crypto positions carry a resting good-till-cancelled stop-limit order
  at Alpaca, which survives if the bot crashes; stock entries are **bracket** orders whose stop leg
  is upgraded to a native `trailing_stop` once the trail arms (`stock_loop._manage_stops`).
- **Local exit loop.** `base_loop._manage_stops` runs first in every cycle: hard stop, take-profit,
  trailing, requiring **two consecutive readings** to confirm a breach so one bad quote cannot
  trigger a sale. Signal sells fire when the prediction turns sufficiently negative
  (`_execute_sells`); stocks also sell a holding that drops out of the top 15 while its prediction is negative.
- **End of day.** Stocks are flattened within about 10 minutes of the close, except up to two
  overnight-sleeve positions of at most 5% of equity each that are still predicted up
  (`OVERNIGHT_SLEEVE_*`); the sleeve fails closed without a live earnings calendar.

### Account-level brakes

- **Circuit breaker** (`base_loop._circuit_breaker_check`): if the account is down 5% on the day
  against Alpaca's `last_equity` baseline (`CIRCUIT_BREAKER_PCT = 0.05`), the bot flattens its book,
  sends an alert, and halts entries until the baseline resets. If the risk state cannot be read, buys
  are suspended for that cycle (fail-closed).
- **Drawdown ladder** persistence: `drawdown.restore_peak_equity` keeps the high-water mark across
  restarts. Before this fix a restart mid-drawdown reset the peak and silently disabled the ladder
  exactly when the account was underwater (`drawdown.py` docstring, wave-8 #4).
- **Cooldowns and lockouts**: see the funnel in Chapter 7.
- **Cross-book cap**: `risk_budget.ACCOUNT_RISK_CAP = 0.03` is currently **journaled, not enforced**
  (`_record_account_risk`); the per-book 2.5% cap is the enforced one (`docs/MAP.md` §9 item 10).

## 4. What the evidence says so far

- **The phantom-trade Kelly problem.** The most instructive finding of the night (ENGINE R6 and R7,
  `research/campaign_2026-09_jetson/CHANGELOG.md`). In April 2026 a bug made the crypto bot believe
  positions had vanished ("DESYNC"), and it bought again and again: 67 false "Position gone" lines,
  $95.8k of buys and no sells. The trade memory recorded **63 phantom `broker_stop` exits** that never
  happened, and none is marked `estimated`, so Kelly counts them. The result: crypto Kelly computes to
  0.1995 on 73 rows, 63 of them phantom, which maps to a **1.5x** crypto sizing multiplier today. A
  repair list exists but was not applied, and the sample gate that would hold Kelly neutral
  (`KELLY_SAMPLE_GATE`) is off. The lesson: a sizing rule that learns from its own history is only as
  honest as that history. With the crypto book now on hold and the account liquidated, the exposure is
  moot for the moment, but it must be repaired before crypto trades again.
- **Broker quirks touch risk too.** Six inherited crypto positions had zero cost basis on an asset id
  none of the bot's 1,320 orders had used, so it was unverified whether the resting stops protected
  them at all. The founder liquidated the paper account to cash this morning, which removed the
  question (README §6).
- **Double counting in sizing.** VIX enters both the ladder and the macro multiplier; volatility
  enters the ATR base, the vol target, the VIX ladder, the HMM and the book scalar. A default-off
  alternative composition (`DERISK_STACK_V2`, which takes the minimum within a family of regime
  signals instead of the product) is computed and journaled on every entry as a shadow, so the two
  can be compared on real fills before anyone flips it.
- **Alerting was silent.** No Telegram or webhook was configured, so every breaker, crash or halt
  alert before last night went nowhere (README §6 item 4).
- **Kill list:** strategy-level volatility targeting as a source of *alpha* is killed; forecasting
  volatility for *sizing* (HAR-RV) survives and is live (`research/KILL_LIST.md`, commonly confused
  survivor #8).

## 5. Further reading

- John L. Kelly Jr., "A New Interpretation of Information Rate," *Bell System Technical Journal* 35(4),
  1956.
- Edward O. Thorp, "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market," in
  *Handbook of Asset and Liability Management*, Volume 1, Elsevier, 2006.
- Leonard C. MacLean, Edward O. Thorp and William T. Ziemba (eds.), *The Kelly Capital Growth
  Investment Criterion: Theory and Practice*, World Scientific, 2011.
- Sanford J. Grossman and Zhongquan Zhou, "Optimal Investment Strategies for Controlling Drawdowns,"
  *Mathematical Finance* 3(3), 1993.
- Alan Moreira and Tyler Muir, "Volatility-Managed Portfolios," *Journal of Finance* 72(4), 2017.
- Fulvio Corsi, "A Simple Approximate Long-Memory Model of Realized Volatility," *Journal of Financial
  Econometrics* 7(2), 2009. The HAR-RV model behind `volatility.py`.
