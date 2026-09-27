# Chapter 3. Features: turning prices into numbers

*What the model actually sees, why most of it is ratios and ranks rather than prices, and the story
of a one-day leak that sat inside every model trained before last night.*

---

## 1. The idea in plain words

A forecasting model cannot look at a chart. It needs a table of numbers. Each row of the table is one
moment in time for one asset (for example "SOL at the close of the 14:00 UTC bar"), and each column is
a **feature**: a number computed from information available at that moment. The model learns a
mapping from a row of features to the future return that followed.

Three rules govern good features:

1. **Stationarity.** A series is **stationary** when its statistical behavior (average, spread) does
   not drift over time. The raw price of Bitcoin is not stationary: it was $30,000 in one year and
   $100,000 in another, so a model that learns "high price means X" learns nothing that transfers.
   A return ("up 1.2% over 4 hours") or a ratio ("2% above its 20-hour average") behaves similarly in
   any year. This system trains only on the stationary kind.
2. **Point-in-time (PIT) discipline.** Every feature must be computable using only data that existed
   at that moment. If a feature quietly uses tomorrow's information, the backtest will look brilliant
   and the live system will not. That mistake is called **look-ahead bias** or **leakage**.
3. **Train/serve parity.** The number the model sees in live trading must be computed exactly the way
   it was computed during training. Two slightly different formulas for "RSI" in two places is enough
   to make a good model behave badly.

## 2. Why it matters financially

**Leakage produces fake edge.** Here is the arithmetic of how little leakage it takes. Suppose a
sentiment score for day D is accidentally built from news published up to the end of day D, and the
model is predicting returns during day D. Even if the score explains only a sliver of the day's move,
say one twentieth of the variance of a 24-hour return, that is a correlation of about
sqrt(0.05) = 0.22 between feature and target. A real, honest hourly forecasting signal typically has
a correlation (**information coefficient**, IC) of 0.01 to 0.05 with forward returns. So one leaked
column can dwarf everything honest in the table, and the model will lean on it. In live trading the
leaked information does not exist yet, and the edge vanishes on the first day.

**Survivorship bias produces fake edge too.** If you train only on stocks that are in today's index,
you have silently selected companies that survived and grew. A model trained on survivors learns that
"beaten-down stocks bounce," because the ones that did not bounce were delisted and are missing from
your data. The fix is an **as-of universe**: at each historical date, use only the names that would
have qualified on that date.

**Non-stationary features overfit.** A model given raw prices can memorize "BTC around $40,000 in
2023 tended to go up," which is a fact about one period, not a rule.

## 3. How this system does it

### One function, two callers

All price-based features come from `indicators.compute_features(df, btc_close)` (the crypto and base
block) and `indicators.compute_stock_features(df, spy_close, symbol)` (base plus session and daily
features for stocks). The harvest scripts call them to build the training store, and
`predict_now.get_live_prediction` calls the **same functions** on live bars. Warm-up gaps (rows where a
long window has not filled yet) are filled with the same neutral values on both sides by
`indicators.fill_warmup_features` (0.0 for most, 0.5 for range-position features). That is what makes
parity structural rather than a matter of discipline (`docs/MODULES.md` §3). Changing any feature's
values is **model-facing**: it requires a fresh harvest and a fresh training run.

### The stationary preset

The harvest computes every indicator; a **preset** chooses which columns the model trains on.
Production always trains with `--preset stationary` (hardcoded in `run_pipeline.py`, so the GUI's
preset picker only affects manual runs). The list lives in `indicator_config._STATIONARY_FEATURES`.
The trainer intersects it with the columns present in each book's store; the training logs show 30
features for crypto and 65 for stocks (`research_signal.md` SCOUT-2 C1). The families:

**Returns and volatility.** `Return_4h` and `Return_12h` are percent changes over 4 and 12 bars;
`Volatility_12h` is the standard deviation of the last 12 hourly returns. `ROC` is a 12-bar rate of
change, which the 2026-07 review found to be bit-for-bit identical to `Return_12h` (a duplicate the
`stationary_lean` preset removes).

**Price ratios.** `Price_SMA20_Ratio` is the close divided by its 20-bar simple moving average (1.02
means 2% above the average). `Volume_Ratio` is volume over its 20-bar average. Bollinger bands are a
20-bar average plus and minus 2 standard deviations; `BBP_20_2.0` says where the close sits inside
the band (0 at the lower band, 1 at the upper), and `BBB_20_2.0` is the band's width.

**Oscillators.** Bounded indicators from classic technical analysis:

- `RSI` (Relative Strength Index, Wilder, 14 bars): 100 - 100 / (1 + RS), where RS is the average gain
  divided by the average loss. Worked example: if the smoothed average hourly gain is 0.30% and the
  average loss is 0.20%, RS = 1.5 and RSI = 100 - 100/2.5 = 60. Readings above 70 are
  conventionally "overbought," below 30 "oversold."
- `STOCHk_14_3_3` and `STOCHd_14_3_3` (stochastic oscillator): where the close sits within the last
  14 bars' high-low range, smoothed.
- MACD (12, 26, 9): the difference between a 12-bar and a 26-bar exponential moving average
  (`MACD_12_26_9`), its 9-bar signal line (`MACDs`), and their difference, the histogram (`MACDh`).
  Note that MACDs = MACD - MACDh exactly, so three columns carry two numbers of information.

**Time and calendar.** Hours and weekdays are cyclical, so they are encoded as a point on a circle:
`Hour_sin = sin(2 pi h / 24)`, `Hour_cos = cos(2 pi h / 24)`. Worked example: 14:00 gives
sin = -0.500 and cos = -0.866. The benefit is that 23:00 and 00:00 end up close together, as they are
in reality, instead of 23 units apart. `Day_sin/cos`, `Month_sin/cos` and `Turn_of_Month` follow the
same idea. (The kill list records that month-of-year seasonality has decayed since 2015 and that a
training span holds at most about five yearly cycles, which is why the lean preset drops it.)

**Hurst exponent.** `Hurst` is a rolling 100-bar estimate of whether a series trends (above 0.5) or
mean-reverts (below 0.5). A code comment in `indicators.compute_features` is candid: the historical
call feeds price *levels*, for which a pure random walk reads about 0.8 regardless of regime, so the
feature "carried no regime information." The correct version (on returns) is behind
`indicator_config.HURST_ON_RETURNS = False` and would require a retrain.

**Crypto positioning.** From perpetual-futures markets (which the system does not trade, but reads):
`Funding_Rate_Ann` (the 8-hour funding rate annualized, rate x 3 x 365), `Funding_Z` (its z-score
against the trailing 90 prints), `Funding_Chg_24h`, open-interest change and z-score (`OI_Chg_24h`,
`OI_Z`), a top-trader long/short z (`TT_LS_Z`) and taker flow imbalance (`Taker_Imb_24h`). A
**z-score** is (value minus its recent average) divided by its recent standard deviation, so "+2"
means "two standard deviations above normal." Very positive funding means crowded long positioning,
which the cited research links to crash risk at this system's horizon.

**Stock session and daily features.** Price relative to VWAP, the overnight gap, ATR as a percent of
price, relative strength against SPY, `ROD_Ret` (return since the prior session's close),
`Same_Hour_Mean_40d` (the trailing 40-session average return of this same clock hour, shifted so the
current bar is excluded, after Heston, Korajczyk and Sadka 2010), residual reversal (`RR_5`, `RR_21`),
distances from 10- to 200-day moving averages, overnight momentum, range position, and FINRA short
volume (`SVR_21`, the 21-day share of volume that was short sales, and its z-score `SVR_Z`).

**Cross-sectional panel ranks (stocks).** The stock book chooses the top 7 of many names each hour,
so what matters is how a stock compares with the others *right now*. `panel_ranks.add_panel_ranks`
ranks each base feature across all member stocks at the same timestamp and maps the rank to [-1, +1]
with `2 (rank - 1) / (n - 1) - 1` (`panel_ranks._signed_rank`). Worked example: a stock ranked 45th of
60 by 4-hour return gets 2 x 44 / 59 - 1 = +0.49, meaning "better than about three quarters of the
panel." Context columns add the cross-sectional dispersion (`CS_Dispersion`) and breadth
(`CS_Breadth`, the centered share of names above their 20-bar average). The raw dollar volume is
dropped after ranking so only its rank is a feature. Ranking follows Gu, Kelly and Xiu (2020), who
found rank-transformed characteristics work well for machine-learning return prediction.

**Sentiment.** `Daily_Sentiment`: for crypto, the Crypto Fear and Greed index mapped from 0 to 100 onto
-1 to +1 (`sentiment_history._fng_value_to_score`, (value - 50) / 50); for stocks, a daily per-symbol
news score from Finnhub headlines.

### Point-in-time machinery

- **As-of universe (stocks):** at each date only the top 60 names by trailing 30-day dollar volume,
  with a $5M dollar-volume and $3 price floor, are members
  (`scripts/harvest_stock_data._asof_tradability_mask`, `_asof_membership_mask`, `panel_ranks.dv30`).
  `docs/MAP.md` §6 calls these masks "the only place survivorship is prevented."
- **Publication lags:** sentiment is lagged to its publication date; short interest to its FINRA
  publication date; closed bars only (`market_data.drop_forming_bar`).

### The sentiment leak story

This is the most instructive bug in the repository, because it was subtle, it was real, and it was
found by an audit rather than by bad live results.

The crypto Fear and Greed value is published by alternative.me once a day at 00:00 UTC. The cache
table `fng_daily` stored each value under a calendar date. Before a fix, rows were dated in the
Jetson's **local time zone**, America/Chicago. 00:00 UTC is 19:00 or 18:00 the previous evening in
Chicago, so a value published at the start of UTC day D+1 was filed under date D. Every hourly
training bar on day D therefore saw a sentiment value that would not exist until the end of that day:
up to 24 hours of look-ahead. The audit verified this on 1,875 of 1,875 cached days against the API
(comment above `sentiment_history._migrate_fng_date_basis`). Worse, a later fix to the insert code
only governed new rows; cached rows were never rewritten, so every re-harvest re-leaked.

The stock side had a smaller cousin of the same bug: the news cache is Chicago-dated, so the aggregate
for calendar day X includes articles up to about 05:00 to 06:00 UTC on X+1. The rule "use yesterday's
score" was safe for regular-hours bars but not for the extended-hours bar at 00:00 UTC.

The repairs (`research/campaign_2026-09_jetson/README.md` §2 and §3):

- `sentiment_history._migrate_fng_date_basis` moves every legacy row into a side table (delete
  nothing), refills the table with UTC-publication-dated values in one transaction, and sets a marker
  so it runs once. It refuses to run if the refill covers less than 90% of the legacy rows, so a
  truncated API response can never replace real data with zeros.
- `sentiment_history.stock_sentiment_lookup_dates` keys each stock bar to `((t - 6h).date() - 1 day)`
  in UTC, which sends every bar before 06:00 UTC one more day back so the looked-up bucket is strictly
  in the past.

The April 2026 models had trained on the leaked column, which is one reason last night's clean
rebuild was mandatory rather than optional.

## 4. What the evidence says so far

- **Leaks and data faults found and fixed last night** (`research/campaign_2026-09_jetson/README.md`):
  the sentiment look-ahead above; the stock harvest silently losing its newest three months (the
  consolidated-tape end date was not clamped); open-interest zero prints now masked; triple-barrier
  labels now stamped after row filters (Chapter 4). A C-language speed-up of the indicator kernels
  was found to overflow memory on short frames and was archived, now opt-in via
  `TRADER_INDICATORS_C=1`; the Numba version is the default and numerically identical.
- **The rebuilt stores:** crypto 263,889 rows by 71 columns; stocks 1,536,356 rows by 110 columns over
  92 names, all from Alpaca bars, with `Eff_Spread_Pct`, the `CS_*` panel ranks and labels
  re-derived.
- **Features are not edge.** On those clean stores, the model built from these features showed no
  measurable selection skill in either book (Chapter 6). That does not prove the features are
  useless; it proves that this model, objective and cost structure did not turn them into
  profitable trades. `indicator_leadlag.py`, the tool that measures each feature's predictive
  correlation with forward returns at 1 to 48 hours with false-discovery control, has not yet been
  run to a verdict on the new stores.
- **The kill list constrains new ideas.** `research/KILL_LIST.md` rules out, among others, any
  additional technical oscillators ("A-minus grade negative evidence post-cost"), post-earnings drift,
  52-week-high anchoring, and on-chain flow features. A new feature idea must be checked there first.
- **Known skews awaiting the owner:** `oi_archive.py` open-interest units differ between the training
  archive (Binance, dollar notional) and the live source (OKX, coin units); the funding z-score live
  baseline spans only about 2.8 days unless `TRADER_FUNDING_Z_TIME_THINNING` is enabled
  (`docs/MAP.md` §9, `docs/MODULES.md` §3). Both are crypto-side and now on hold.

## 5. Further reading

- John J. Murphy, *Technical Analysis of the Financial Markets*, New York Institute of Finance, 1999.
  Clear definitions of RSI, MACD, Bollinger bands and stochastics.
- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems*, Trend Research, 1978. The original
  RSI and ATR.
- Shihao Gu, Bryan Kelly and Dacheng Xiu, "Empirical Asset Pricing via Machine Learning," *Review of
  Financial Studies* 33(5), 2020. Rank-transformed features and what machine learning can and cannot
  extract from them.
- Steven L. Heston, Robert A. Korajczyk and Ronnie Sadka, "Intraday Patterns in the Cross-section of
  Stock Returns," *Journal of Finance* 65(4), 2010. The same-hour periodicity feature.
- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, chapter 5
  (stationarity and memory in features).
- Cheol-Ho Park and Scott H. Irwin, "What Do We Know About the Profitability of Technical Analysis?"
  *Journal of Economic Surveys* 21(4), 2007. A sober survey of the evidence.
