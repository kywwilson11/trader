# Chapter 11. Measuring yourself honestly

## 11.1 The idea

The previous chapter was about testing a model before it trades. This chapter is about testing the
*system* after it trades: its gates, its market exposure, and its most expensive advisor, the LLM.
The questions are simple to ask and hard to answer honestly:

- Did the gates that blocked trades save money, or did they block winners?
- Is the profit (or loss) skill, or is it just the market going up (or down)?
- Does the LLM's opinion add anything the model did not already know?
- And for each of those: do we have enough data to say?

The last question is the one people skip. The rest of this chapter is largely about it.

Three terms.

- **Alpha** is return that is not explained by exposure to the market. **Beta** is the part that
  is: a beta of 1.0 to SPY means the book tends to move one for one with the S&P 500. You can buy
  beta for almost nothing with an index fund. Paying for complexity only makes sense for alpha.
- The **effective sample size** (`n_eff`) is the number of *independent* observations hiding
  inside a larger number of correlated ones.
- A **verdict** is what a measurement tool concludes: keep, kill, review, or "cannot conclude".
  A **void** verdict is one that must not be used at all, because its inputs do not describe the
  system you are asking about.

## 11.2 Why it matters financially

People are generous graders of their own work. A trader who made 20 % in a year when the market
made 25 % may feel skilled; a trader who lost 5 % in a crash may feel unlucky. Both need the same
correction: separate what the market did from what the decisions did.

The second trap is counting. Suppose an LLM scores six coins every hour and each score is judged
against the next 24 hours of return. In a month that is over four thousand scored rows. But
adjacent hours share 23 of their 24 hours of future return, and the six coins mostly move
together. Four thousand rows are really closer to thirty independent observations. Statistics
computed as if you had four thousand will declare noise to be signal, and a decision to "keep
paying for this" or "keep this gate" will be made on nothing.

## 11.3 How this system does it

### The decision report: pricing every veto

[`decision_report.py`](../../decision_report.py) reads the journals (Chapter 9) and, for every
skipped trade, asks what would have happened if the gate had not fired. It replays the vetoed
entry through **the same exit kernel** (`policy_exits`) that prices real exits, charges the same
round-trip costs, and reports for each gate: how many vetoes, and the mean counterfactual net
return. A gate whose vetoed trades would have *lost* 0.8 % on average saved 0.8 % per veto. A gate
whose vetoed trades would have *made* money is charging admission, not providing protection.

Three honesty mechanisms are built in:

1. **Episode deduplication.** A symbol skipped every 30 seconds for a whole day would otherwise
   produce hundreds of overlapping replays of the same price move. Rows are collapsed to one per
   (symbol, reason, calendar day).
2. **Confidence intervals and a minimum count.** Every mean carries a 90 % bootstrap interval,
   and no REVIEW / OK / CHANGE verdict fires below `MIN_VERDICT_N = 9`. If the interval includes
   zero, the answer is "cannot conclude".
3. **Stale and unrepresentative flags.** If prices cannot be fetched or nothing was priced, the
   report is written with `stale: true` so the GUI and downstream tools refuse to trust it, and a
   `representative` flag goes false when fetch failures occurred.

The 2026-09 campaign added a report-only second interval that resamples whole calendar days
instead of rows (`ci90_dayclust`), because on a given day all stocks tend to move together; the
report now prints how often the two intervals disagree.

### The beta ledger: alpha versus beta

[`beta_ledger.py`](../../beta_ledger.py) regresses the account's daily equity returns on SPY and
BTC. Its docstring cites the key lesson from Asness, Krail and Liew (2001): funds that looked
market-neutral had near-zero same-day beta but large **lagged** beta, because their positions were
priced with a delay. Summing the same-day and lagged betas roughly doubled the measured exposure
and erased much of the apparent alpha. So the ledger reports:

- lagged (summed) betas, not just same-day betas;
- an alpha estimate with a heteroskedasticity-and-autocorrelation-robust (HAC) t-statistic;
- up-market and down-market betas (market timing shows up as an asymmetry);
- betas conditional on the market's trend state;
- since 2026-09, a robust **winsorized** beta after Welch (2022), a `beta_stable` flag, and
  `alpha_mintrl_years`, the number of years of data needed before an alpha of the observed size
  could reach a t-statistic of 2.

### The LLM scorecard

[`llm_eval.py`](../../llm_eval.py) asks whether the LLM's score `s` adds information *beyond* the
model's prediction. This is subtle because the LLM's prompt shows it the model's prediction, so
its scores can simply echo the model. The primary test is a regression

    realized return = a + b1 x (model prediction) + b2 x (standardized LLM score)

and the question is whether `b2` is reliably positive. The standard errors are Driscoll-Kraay
(1998), clustered by the hour the forecast was made, to cope with overlapping horizons and with
all symbols moving together. No verdict is issued unless **all** of these hold (constants at the
top of the file): at least `MIN_POWER_N = 60` realized rows, at least `MIN_POWER_T0 = 120`
distinct forecast hours, and at least `MIN_EFFECTIVE_N = 20` effective observations, computed as
the time span divided by the forecast horizon.

### The evidence-reads wrapper

[`scripts/evidence_reads.py`](../../scripts/evidence_reads.py) runs the whole measurement shelf in
one command and prints a table with READY, NOT YET, NO DATA, FAILED or SKIPPED for each read,
together with each read's pre-registered sample-size rule and, where history exists, an estimated
number of days until it becomes READY.

### The anytime-valid ledger

[`llm_eprocess.py`](../../llm_eprocess.py) is the newest instrument and the most unusual. Ordinary
statistical tests assume you decide the sample size in advance and look once. In practice you look
every day, and each look is another chance to be fooled. An **e-process** is a running measure of
evidence that stays valid no matter how often you look or when you stop.

The intuition is a betting game. Start with $1 of imaginary capital and, each day, bet a fraction
of it that the LLM's size tilt added value that day. If the tilt adds nothing, the game is fair or
unfavourable, and a theorem (Ville's inequality) says the chance that your capital ever reaches $40
is at most 1 in 40. So if it does reach $40, that is strong evidence, however many times you
peeked.

The ledger computes a daily mark-to-market value of the LLM tilt, net of fees and of the LLM bill,
in basis points of equity, and runs three one-sided bets: KEEP (the tilt adds value), KILL-HARM
(it destroys value) and KILL-FUTILITY (it adds less than 1 basis point per day, about $10 a day on
$100,000, ten times the spending cap). Any bet whose capital reaches 40 decides. The first decision
may come after 10 test days and at least 30 trades; the horizon is 180 days. In simulation the
design has an 86.7 % chance to detect a true +1 basis point per day by day 180, but only 38.7 % by
day 90. The module **refuses to run on live data** (exit code 3) until the owner signs its
pre-registration sheet,
[`research/campaign_2026-09_jetson/llm_eprocess_params.json`](../../research/campaign_2026-09_jetson/llm_eprocess_params.json),
because thresholds chosen after seeing the data are not evidence.

### Effective sample sizes, and a rule against double counting

Overlap is handled in several places: `sample_weights.average_uniqueness` and
`calendar_effective_n` for trades that overlap in time, the span-over-horizon rule in `llm_eval`,
and day-clustering in `decision_report`. `CLAUDE.md` gotcha #4 states the rule: use **one**
correction for overlap. The uniqueness-based `n_eff` and the Lo (2002) serial-correlation factor
are two estimates of the same inflation; applying both punishes the data twice.

### What "void" means

A verdict is void when the data behind it came from a different system than the one you are
judging. The clearest case in this repository: the April and May 2026 journals were written while
a since-removed rule blocked buys whenever the LLM score was below 0.60. Of 11,818 skip rows in
those journals, 11,815 were that one gate. Today's code only vetoes below 0.15. Any gate
attribution, LLM verdict or spend ledger computed from those journals describes a policy that no
longer exists, so the runbook declares pre-campaign `decision_report` figures void and the LLM
ledger excludes journals through 2026-05-07. Void is stronger than "not significant": a
not-significant result is weak evidence about the current system, while a void result is no
evidence about it at all.

## 11.4 A worked example: 720 rows or 30?

Suppose one month of hourly LLM scores for one coin: 30 days x 24 hours = 720 scored rows, each
judged against the next 24 hours. You find a correlation of 0.10 between score and outcome.

The usual t-statistic for a correlation is r x sqrt(n - 2) / sqrt(1 - r^2).

- Treating the rows as independent, n = 720: t = 0.10 x 26.8 / 0.995 = **2.69**. That clears the
  conventional bar of 2 and looks like a finding.
- But each 24-hour outcome overlaps the next 23. The effective count is the span divided by the
  horizon, 720 / 24 = 30. With n = 30: t = 0.10 x 5.29 / 0.995 = **0.53**. That is nothing.

Same data, same correlation, opposite conclusions. The `MIN_EFFECTIVE_N = 20` floor in `llm_eval`
exists to prevent the first reading. It also explains why "we have lots of rows" is never a reason
to trust an LLM verdict after a few days.

A second example, from the live beta ledger run during the campaign (INTEL owner item 10). The
book's summed lagged BTC beta was **+1.18**, while the same-day beta was +0.25 and a plain
regression gave +0.44. Looking only at same-day numbers, you would have concluded the crypto book
carried a quarter of Bitcoin's risk; it carried more than all of it. The same run printed
`alpha_mintrl_years` of 2.0 against a 0.4 year window: the alpha is not estimable yet, and "beta
only" is the honest summary.

## 11.5 What the evidence says

- **The LLM keep/kill test was too eager.** The INTEL department simulated the scorecard's test on
  synthetic data with no true effect, shaped like this system's data (six symbols, 24-hour
  horizon). At a nominal 5 % false-alarm rate, the Driscoll-Kraay test with its t(G-1) reference
  rejected a true null 13.0 %, 11.0 % and 9.6 % of the time at 20, 30 and 60 effective
  observations. The Ibragimov-Müller alternative (split the sample into 8 blocks, estimate in each,
  and t-test the 8 estimates) gave 5.6 %, 6.2 % and 5.2 %, the only estimator inside the
  pre-registered 3 to 7 % band. Both p-values are now printed side by side; making Ibragimov-Müller
  the deciding test is an open owner decision
  ([INTEL owner items](../../research/campaign_2026-09_jetson/session_artifacts/generals/intel/OWNER_ITEMS.md), item 1).
- **There is no LLM verdict yet.** The scorecard stands at 0 of 120 required forecast-hour
  clusters on journals written by the current code. At one book scored every 10 minutes, reaching
  the floor takes weeks of normal trading, and the clock only starts once a certified model trades.
- **Alpha cannot be established on a paper-trading horizon.** An earlier INTEL estimate put the
  time needed for a t-statistic of 2 at about 4.3 years, even at a Sharpe near 1. Beta, by
  contrast, is measurable within months. The recommended policy is to report beta and state that
  alpha is not estimable, rather than to report a noisy alpha.
- **The anytime-valid ledger has a known weakness.** With realistic day-to-day autocorrelation in
  the daily statistic (0.3), the simulated false-fire rate rose to 3.5 to 5.4 % against a 2.5 %
  target. The INTEL department asked the owner to rule on a guard before signing the sheet.

## 11.6 Further reading

- Andrew W. Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4), 2002,
  36 to 52. How uncertain a Sharpe ratio really is, and why serial correlation matters.
- Clifford S. Asness, Robert Krail and John M. Liew, "Do Hedge Funds Hedge?" *Journal of Portfolio
  Management* 28(1), 2001, 6 to 19. Lagged betas.
- John C. Driscoll and Aart C. Kraay, "Consistent Covariance Matrix Estimation with Spatially
  Dependent Panel Data," *Review of Economics and Statistics* 80(4), 1998, 549 to 560.
- Rustam Ibragimov and Ulrich K. Müller, "t-Statistic Based Correlation and Heterogeneity Robust
  Inference," *Journal of Business and Economic Statistics* 28(4), 2010, 453 to 468.
- Aaditya Ramdas, Peter Grünwald, Vladimir Vovk and Glenn Shafer, "Game-Theoretic Statistics and
  Safe Anytime-Valid Inference," *Statistical Science* 38(4), 2023, 576 to 601.
- Ian Waudby-Smith and Aaditya Ramdas, "Estimating Means of Bounded Random Variables by Betting,"
  *Journal of the Royal Statistical Society Series B* 86(1), 2024, 1 to 27.
- In this repository: [`docs/MAP.md`](../MAP.md) section 4g (what each instrument is for) and
  [`research/campaign_2026-09_jetson/research_intel.md`](../../research/campaign_2026-09_jetson/research_intel.md)
  (Scouts B and C).
