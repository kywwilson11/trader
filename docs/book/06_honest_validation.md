# Chapter 6. Honest validation

*How to tell skill from luck when you have tried many things on the same history. This is the most
important chapter in Part One, and it ends with last night's verdict: no selection skill, explained
kindly.*

---

## 1. The idea in plain words

A model's quality can only be judged on data it has not seen. Everyone knows that. The hard part is
that in finance "data it has not seen" is surprisingly difficult to arrange, for three reasons:

1. **Time leaks.** If you shuffle hourly rows randomly into a training pile and a test pile, the test
   row for 15:00 sits next to a training row for 14:00 that shares almost all of its future. The model
   has effectively seen the answer. Tests on financial data must respect time: train on the past,
   test on the later future.
2. **Labels leak across the boundary.** Even with a clean time split, a training row from 23:00 the
   night before the test period has a 24-hour label that reaches *into* the test period. That row's
   answer overlaps with test answers. The fix is **purging** (remove training rows whose label window
   crosses the boundary) and an **embargo** (skip a short gap after the boundary before testing, so
   slow-moving features cannot carry information across).
3. **You leak through your own choices.** Every time you try a configuration, look at its test score
   and try another, the test set becomes a little more like training data. After 40 tries, the best
   test score is partly a measure of luck. The fix is to **count your tries** and demand more of the
   winner the more you tried, and to keep one final slice of data, the **holdout**, that the search
   never touches.

A few definitions used throughout:

- **Sharpe ratio**: average return divided by the standard deviation of returns. A strategy earning
  0.2% per trade with a 1% standard deviation has a **per-trade Sharpe** of 0.2. To compare across
  frequencies it is **annualized** by multiplying by the square root of trades (or periods) per year:
  0.2 x sqrt(100 trades a year) = 2.0.
- **Walk-forward validation**: train on an early window, validate on the next window, move forward,
  repeat.
- **Deflated Sharpe Ratio (DSR)**: the probability that the true Sharpe is above what the best of N
  skill-less attempts would show by luck (Bailey and López de Prado, 2014). A DSR of 0.95 is strong
  evidence; 0.5 means indistinguishable from luck.

## 2. Why it matters financially

**The lucky-winner arithmetic.** Take 40 "strategies" that are pure coin flips: each makes 60 trades
with returns drawn from a distribution with mean zero. None has any skill. Pick the one with the best
per-trade Sharpe. A simulation of that experiment (2,000 repetitions, run for this chapter with the
repo's own `validation.expected_max_sharpe` as a cross-check) gives a typical best per-trade Sharpe of
about **0.28**. If those 60 trades happened over a year, the annualized Sharpe is
0.28 x sqrt(60) = **2.2**. That is a number that would impress most investors, produced by selecting
the luckiest of 40 random guesses.

The formula behind it: the expected maximum of N independent standard normal draws grows roughly like
sqrt(2 ln N). `validation.expected_max_sharpe` gives, in units of one standard deviation:

| attempts N | 2 | 10 | 40 | 100 | 1000 |
|---|---|---|---|---|---|
| expected best | 0.52 | 1.58 | 2.19 | 2.53 | 3.26 |

Trying 25 times more (40 to 1000) only raises the luck bar by half again, which is why counting
generously costs little.

**What goes wrong without it.** Deploy the lucky winner and its live performance will regress toward
the true value, zero, minus costs. The account pays fees for the privilege of discovering this. The
2026-07 review expected a 50 to 73% haircut from backtest Sharpe to live Sharpe even for a model that
passes (`docs/MAP.md` §9).

## 3. How this system does it

All of this lives in `scripts/hypersearch_v2.py` (the trainer), `validation.py` (the mathematics),
`sample_weights.py` (effective sample size) and `backtest.py` (the policy gate).

### The holdout: the last 12%, touched once

`hypersearch_v2.get_holdout_boundary` (via `objective_utils.holdout_boundary`) sets the boundary at the
88th percentile of all row timestamps (`HOLDOUT_FRACTION = 0.12`); an option `FIXED_HOLDOUT_DAYS`
pins it to a fixed number of days instead. Every later step (folds, refits, blend fit, the
certificate) inherits the same boundary. The search never sees data after it.

### Purged, embargoed walk-forward folds

`hypersearch_v2.get_walk_forward_folds` builds three **expanding** folds over the search region (the
88% before the holdout). Fold training windows end at the 55th, 70th and 85th percentiles of the search
timestamps; each validation window covers the next 15%. Two protections:

- **Purge:** a training row is kept only if its label window *completes* before the boundary
  (`all_label_times <= t_train_end`).
- **Embargo:** validation starts `seq_len` bars after the boundary. With `TRAINING_REPAIRS_V1` (now
  ON) the gap is counted in actual bars on the data's own grid; the older rule counted calendar hours,
  which for stocks shrank a 40-bar embargo to about 11 trading hours.

The docstring records why folds are split by calendar time: the old code split rows by position, and
since rows are stored ticker by ticker, "train on the first 60%" meant training on some coins' entire
history and validating on other coins over the same dates. With cryptos correlated 0.7 to 0.9, that
was "near-direct leakage."

### The score the search maximizes

For each trial and fold, `hypersearch_v2.compute_sharpe` simulates non-overlapping trades (enter when
the prediction exceeds the trial's threshold, hold `fb` bars) net of `TXN_COST_PCT` and annualizes by
the occupied share of the year. Fewer than 10 trades scores 0.0. The trial score is

  score = mean(fold Sharpes) - 0.5 x std(fold Sharpes),

which rewards consistency across folds. A regime penalty then applies if any market regime's Sharpe
is below -0.5. A small but telling bug: the legacy penalty multiplied the score by 0.7, which for a
negative score moves it *toward zero*, rewarding a trial for doing badly in some regime. Under
`TRAINING_REPAIRS_V1` the penalty is `score -= 0.3 x |score|`, which always makes it worse.

### Effective sample size

Trades that overlap in time are not independent. `sample_weights.calendar_effective_n` gives each
trade a **uniqueness** u equal to the average, over the hours it was open, of 1 / (number of trades
open at that hour), and sums them: n_eff = sum of u. Worked example: three trades on three correlated
coins opened in the same hour and closed together each have u = 1/3, so together they count as
**one** effective observation, not three. Last night SIGNAL found and fixed a units bug in the
holdout's n_eff (a seconds-versus-hours mix-up), which the campaign README lists among the
night's highlights.

### The Deflated Sharpe gate

`validation.dsr_from_trade_returns(trade_returns, n_trials, n_eff=...)` computes:

1. the observed per-trade Sharpe s on the holdout trades;
2. the luck bar SR0 = `expected_max_sharpe(N, 1/sqrt(n_eff))`, the Sharpe the best of N skill-less
   configurations would show with this many effective trades;
3. DSR = Phi( (s - SR0) x sqrt(n_eff - 1) / sqrt(1 - skew x s + (kurt - 1)/4 x s^2) ), where Phi is
   the normal cumulative distribution and the skew/kurtosis term corrects for fat-tailed returns
   (`validation.deflated_sharpe_ratio`).

A model is saved only if holdout Sharpe > 0 **and** DSR >= `DSR_MIN = 0.60`. The code comment is
candid that 0.60 "sits well below" the conventional 0.95: it is a screen for a challenger, not a
publication-grade test.

Worked example, computed with the repo's own functions: N = 40 trials, n_eff = 60 effective trades.
The luck bar SR0 is **0.283** per trade. Then:

| observed per-trade Sharpe | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|
| DSR | 0.08 | 0.27 | 0.55 | 0.81 |

Only a per-trade Sharpe near 0.35 or above passes. `validation.min_track_record_length` turns the
same idea into "how many effective trades would I need": to show a per-trade Sharpe of 0.10 is above
zero at 95% confidence takes about **273** effective trades.

### PBO, the ratchet and the second gate

- **PBO** (probability of backtest overfitting) asks how often the configuration that was best in one
  half of the data ranks below median in the other half. The search prints a coarse version from fold
  scores (`validation.pbo_from_fold_scores`); the full combinatorial version (`validation.pbo_cscv`,
  Bailey et al. 2017) is implemented and tested but has **no production caller**. `docs/MAP.md` §4b
  asks that anyone citing "CSCV-PBO" say so.
- **The ratchet:** the holdout is scored only when the search winner beats the stored best score
  (`accept_new = new_score > existing_score` in `hypersearch_v2.main`). A new study starts from a
  stored best of 0.0, so a search where every trial is negative never touches the holdout at all.
  A noisier, Thresholdout-style ratchet (Dwork et al., 2015) exists behind `PROMOTION_GATE_V2`, off.
- **The policy gate:** `backtest.py --gate` replays the saved model through the real exit kernel and
  costs and requires `n_trades >= 10`, Sharpe > 0 and DSR >= 0.60; on failure it rolls back to the
  previous model (`.prev` files).
- **The live test:** a retrained model goes to the **challenger** slot and runs in **shadow**
  (predicting, not trading) next to the **champion**. `shadow.dm_hln` compares their forecast errors
  with the Diebold-Mariano test (Harvey-Leybourne-Newbold correction); promotion needs at least 200
  observations, 14 days and p < 0.05 (`shadow.MIN_OBS`, `MIN_SHADOW_DAYS`, `EARLY_PROMOTE_P`).

## 4. What the evidence says so far

### Last night's verdict, explained kindly

Here is what happened on the clean, leak-free stores (`research/campaign_2026-09_jetson/README.md` §4):

- **Crypto:** 44 trials, 43 negative, one at +0.03. Its holdout produced 3 trades on 31,662 rows:
  "Model NOT saved: insufficient_n." The DECOMP-1 analysis found 44 of 48 fold Sharpes negative, the
  scoring arithmetic was not the cause, and every searched threshold was below the 1.2% admission
  floor, so trades lost after cost. "More trials of the same design won't help."
- **Stocks:** 25 trials, best score +0.024. Holdout: 20 trades, n_eff 14, Sharpe 0.13, DSR **0.058**
  against 0.60: "Model NOT saved: failed holdout gate." DECOMP-2 on the first 11 trials: no trial beat a
  zero-skill ranker by more than one standard error; a random ranker's Sharpe predicted the observed
  ordering of trials (rank correlation 0.65, p = 0.03); fold 1 was at or below zero in all 11 trials;
  and the study's "best" entry was the 0.0 score of a configuration that barely traded.

The kind reading is this. The system was asked "is there skill here?" and answered "not that I can
detect," with a precise diagnosis of why. Nothing unskilled was saved, nothing unskilled traded, no
money was lost, and the diagnosis names concrete levers (a minimum-trade rule, scoring relative to
a benchmark or drift, the penalty form, conditioning on regime or entry time, thresholds anchored to
cost). A validation pipeline that says "yes" to everything is worthless; this one has now proved it
can say "no." That is the foundation every future "yes" will stand on.

### How strict is the gate, really?

SCOUT-3 computed the gate's **power**: the chance it passes a model that truly has a given edge
(`research_signal.md` G2). At n_eff = 60 and N = 44 trials, a model with a real per-trade Sharpe of
0.10 (about 2 annualized at 500 trades a year) passes only about **4.5%** of the time; at 0.20, about
18%. A skill-less model passes about 0.7% of the time. The gate is safe against false positives but
"power-starved," and power falls further as the cumulative trial count grows. The note proposes that
the owner pick exactly one remedy (count holdout evaluations rather than trials, lengthen the
holdout, or rely on the live shadow test) and warns never to combine it with lowering `DSR_MIN`.

### Search mechanics worth knowing

`PRUNE_STARTUP_TRIALS = 60` sets both the optimizer's random start-up phase and the pruner's warm-up,
so a 40-trial run is **pure random search with no pruning** despite the "TPE + pruning" banner
(`research_signal.md` SCOUT-2, and the owner note `startup_trials_owner_item.md`). This morning's
70-trial stock retrain will steer only for its last 10 trials.

## 5. Further reading

- David H. Bailey and Marcos López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* 40(5), 2014.
- David H. Bailey, Jonathan M. Borwein, Marcos López de Prado and Qiji Jim Zhu, "The Probability of
  Backtest Overfitting," *Journal of Computational Finance* 20(4), 2017.
- Campbell R. Harvey and Yan Liu, "Backtesting," *Journal of Portfolio Management* 42(1), 2015. How
  to haircut Sharpe ratios for multiple testing.
- Andrew W. Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4), 2002.
- Cynthia Dwork et al., "The Reusable Holdout: Preserving Validity in Adaptive Data Analysis,"
  *Science* 349(6248), 2015.
- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, chapters 7 (purged
  cross-validation) and 11 to 14 (the dangers of backtesting).
