# Chapter 14. What the evidence says today, and what would change it

## 14.1 The idea

Every earlier chapter described machinery. This one reports the result of running it. On the night
of 2026-09-26 to 27, for the first time, the current code was run end to end on the production
Jetson, against clean data, with every guard switched on. The honest headline is short:

**On clean, leak-free data, the current model and objective showed no measurable ability to pick
winners in either book.**

That is not a failure of the pipeline. It is the pipeline doing its job: refusing to certify
something that has not been shown to work. This chapter explains how that conclusion was reached,
why it is more useful than it sounds, what levers could change it, what the retrain running as this
book is written is testing, how to read the certificate it will produce, and why the crypto book was
set aside.

Terms:

- **Selection skill** means the model's ranking of opportunities is better than a random ranking.
  Without it, trading is paying costs to hold a random portfolio.
- A **zero-skill ranker** is a benchmark: a random choice of trades with the same frequency and the
  same costs as the model's. If the model cannot beat it, the model has no skill at that task.
- A **lever** is a design change that could plausibly create or reveal skill, as opposed to simply
  searching the same design longer.

## 14.2 Why it matters financially

Most people who build trading systems never get a clean negative result. They get a noisy positive
one, trade it, and learn the truth from their account statement. A clean negative is valuable for
three reasons.

1. **It is cheap.** It cost a night of GPU time and no capital.
2. **It is specific.** The diagnosis below names *why* the scores were negative, which points at
   what to change.
3. **It sets the prior.** The repository's own map ([`docs/MAP.md`](../MAP.md) section 9, item 1)
   had already estimated, from a 2026-07 review, that even a model passing every gate had perhaps a
   40 to 55 % chance of being genuinely profitable, and that live Sharpe ratios typically come in 50
   to 73 % below backtest. Knowing that before money is at stake is the point.

## 14.3 How the result was reached

### The two searches

Both stores were rebuilt from scratch (Chapter 10), and both books were searched with the
long-only objective.

- **Crypto**: 44 trials in the study; 43 scored negative and one scored +0.03. The winner's holdout
  produced **3 trades** in 31,662 rows, so the gate refused to save it (`insufficient_n`).
- **Stock**: 25 trials; the best score was +0.024. Its holdout had 20 trades (14 effective), a
  Sharpe of 0.13 and a DSR of 0.058 against the 0.60 bar, so it was not saved.

A search that finds nothing proves little by itself: maybe it needed more trials. So the SIGNAL
department decomposed the scores, trial by trial and fold by fold, using attributes the trainer now
records (gross return, cost drag, trade count, hit rate, threshold pass rate per fold), and
pre-registered the classification rules before looking.

### What the decomposition found

- **Crypto (DECOMP-1, 16 trials):** 44 of 48 fold Sharpes were below zero. Every searched trade
  threshold sat below the live admission floor of about 1.2 %, the minimum predicted return the
  live cost gate demands, and half sat below the 0.60 % round-trip cost itself. In plain words: the
  search kept choosing rules that trade on forecasts too small to pay for the trade. More trials of
  the same design would not fix that.
- **Stock (DECOMP-2, first 11 trials):** no trial beat the zero-skill ranker by more than one
  standard error. The random ranker's expected Sharpe even predicted the *ordering* of the trials
  (rank correlation 0.65, p = 0.03; for crypto 0.51): trials scored well or badly for reasons a
  random ranker shares, such as how many trades they happened to take, not because of what they
  predicted. The first fold was zero or negative in all 11 trials. Unlike crypto, 9 of 11 stock
  thresholds cleared the live floor, and the raw drift of the target exceeded costs; the sign was
  lost through the scoring formula's `- 0.5 x std` term and a regime penalty.
- **The study's "best" stock trial was a trade that never happened.** Its score was exactly 0.0,
  produced by `compute_sharpe`'s rule of returning 0.0 when a fold has fewer than 10 trades (it
  took 0, 0 and 6). In a study where everything else is negative, zero wins. The fix,
  `MIN_TRADE_PRUNE`, is staged and proven in the landing queue but not yet in the tree.
- **A penalty with the wrong sign.** The objective multiplies a trial's score by 0.7 when its worst
  regime is bad. For a positive score that is a penalty. For a negative score, multiplying by 0.7
  moves it *toward* zero, which is a reward. It fired on 15 of 15 negative crypto trials and 6 of
  11 negative stock trials. `TRAINING_REPAIRS_V1` makes it sign-correct.

### The cost side

Separately, the ENGINE department reconstructed 686 filled crypto paper orders from February to
April 2026. The realized round-trip cost of crossing the spread on LINK, XRP and DOGE was about 30,
26 and 34 basis points, against the 10 basis points (`FLAT_SPREAD_PCT['crypto'] = 0.10` in
`fees.py`) the backtester, the meta-labeler and the cost gate assume. Paper trading has no queue or
market impact, so these are floors on real cost. A model trained and certified on a cost three
times too low will admit trades the venue does not pay for.

## 14.4 The levers

Each lever below is staged behind a default-off flag or written up as an owner proposal. None is a
promise; each is a hypothesis with a test.

| Lever | What it changes | Status |
|---|---|---|
| Minimum-trade prune | a trial with too few trades is pruned instead of scoring 0.0 | staged (`MIN_TRADE_PRUNE`), not landed |
| Cost-anchored thresholds | the searched threshold range starts at the cost | in `OBJECTIVE_V3`, now on |
| Live-floor threshold | the trainer applies the same admission floor as the live gate | proposal (`OBJECTIVE_THRESHOLD_FLOOR_LIVE`) |
| Drift- or benchmark-relative score | judge trades against the market's drift, not zero | proposal |
| Penalty form | fix the sign of the regime penalty, reconsider `- 0.5 x std` | sign fix in `TRAINING_REPAIRS_V1`, now on |
| Session mask | score stock entries only inside the live entry windows | `OBJECTIVE_SESSION_MASK`, off |
| Startup count | 60 random trials before the sampler steers; a 25 or 40 trial run never steers | proposal (`startup_trials_owner_item.md`) |
| Honest crypto costs | spread measured, not assumed | census tool built; ruling pending |

## 14.5 The Phase-3 retrain

On the morning of 2026-09-27, on the founder's explicit instruction, the Phase-3 bundle from the
runbook was switched on: `HYPERSEARCH_V3` (final refit, the LightGBM leg before the gate, a fitted
blend weight, certification of the blended model), `OBJECTIVE_V3` (per-ticker resets,
cost-anchored threshold range) and `TRAINING_REPAIRS_V1` (validation loss on the trial's own
criterion, the regime-penalty sign fix, a look-ahead repair and bar-based embargoes), with
`OBJECTIVE_LONG_ONLY` already on. The old studies were archived, and a 70-trial stock search began
at 09:21.

At 10:07 eight trials had reported. Six scored between -0.39 and +0.15. One (trial 3) was pruned
after a CUDA memory allocator error on the Jetson; because `FAILED_TRIAL_PRUNE` was active for this
run, the failure was recorded as pruned rather than as a fake 0.0 score, which is exactly the
protection the previous section called for. At about five minutes per trial the search will run
for several hours. Remember that with `PRUNE_STARTUP_TRIALS = 60`, the first 60 trials are random,
so the sampler will steer for only the last 10.

Do not read the in-search scores as results. A +0.15 in-search score is exactly the kind of number
selection bias produces. Only the holdout certificate counts.

## 14.6 How to read the next certificate

When the search ends, the trainer prints either `Model NOT saved` with the holdout report, or
`New best model` with the certificate. The certificate lives in `stock_config_v2.pkl` under
`holdout`. Read it in this order.

1. **`n_trades` and `n_eff`.** How many holdout trades, and how many effectively independent ones.
   Below 10 effective trades nothing else matters. Below about 100, expect wide uncertainty.
2. **`sharpe`.** Must be positive. Note whether it is per trade or annualized in the printout.
3. **`n_trials_pool`.** How many configurations the winner was chosen from. This sets the luck bar.
4. **`dsr`** against `DSR_MIN = 0.60`. The probability that the Sharpe beats the best of that many
   skill-less trials.
5. **`min_trl`.** The minimum track record length: how many effective trades would be needed for
   this Sharpe to clear the bar. If it is far above `n_eff`, the certificate is thin.
6. **`hit_rate` and `pred_deciles`.** Do the model's highest predictions actually earn the most?
   A real ranker shows outcomes rising across prediction deciles, not a single lucky decile.
7. For a blended certificate: **`lstm_weight`** (how much the neural net contributes versus
   LightGBM), **`q10_vetoed`**, and **`cs_rank_ic`** for each leg (the cross-sectional rank
   correlation between predictions and outcomes; positive and stable is what skill looks like).

Then remember what comes after: the policy gate replay (`backtest.py --gate`, stock over 60 days,
at least 10 trades, Sharpe above zero, DSR at least 0.60), and, once a champion exists, 14 to 28
days of shadow comparison for any challenger. A certificate is a ticket to the next test, not a
verdict of profitability.

## 14.7 A worked example: the arithmetic of skill

Grinold and Kahn's "fundamental law of active management" gives a rough rule: the
information ratio (risk-adjusted excess return per year) is about the **information coefficient**
(IC, the correlation between forecasts and outcomes) times the square root of **breadth** (the
number of independent bets per year).

- An IC of 0.02, which is typical of a real but weak signal, with 2,500 independent bets a year
  gives 0.02 x 50 = **1.0** before costs. That would be excellent.
- The same IC with 100 independent bets a year gives 0.02 x 10 = **0.2**. That is barely
  distinguishable from noise over several years.
- Now subtract costs. If each trade's expected gross edge is 0.3 % and the round trip costs 0.3 %,
  the net edge is zero regardless of breadth. If the true crypto cost is 0.3 % where 0.1 % was
  assumed, a model that looked profitable in the backtest is not.

This is why the diagnosis above keeps returning to two things: whether the model has any IC at all
(the zero-skill comparison) and whether its trades clear their cost (the threshold versus the
admission floor). Breadth cannot rescue a zero IC, and IC cannot rescue a trade that loses its edge
to the spread.

## 14.8 The crypto decision

At 08:55 on 2026-09-27 the founder directed that effort stop on the crypto book and concentrate on
stocks. The evidence behind it:

- 43 of 44 crypto trials negative, with an ordering explained by a random ranker;
- every crypto threshold below the live admission floor;
- realized crypto taker costs two to three times the modelled cost;
- six inherited crypto positions with a zero cost basis on a broker asset id that none of the
  system's orders had used, raising the question whether stop orders protected them at all.

The mechanics chosen are reversible: the service runs `run_pipeline.py --stock-only`, the paper
account was liquidated to about $122,000 in cash at 09:21, and nothing was deleted. Crypto-specific
owner items (the spread census, funding and open-interest features, crypto de-risking) are on hold.
If crypto is ever revisited, the first step is honest costs, not more trials.

## 14.9 What would change the verdict

Pre-registration means deciding in advance what evidence would change your mind. For this system:

- **Toward "there is skill":** a holdout certificate with DSR at or above 0.60 on a meaningful
  `n_eff`; positive cross-sectional rank IC in the Stage-0 dump from the weekly backtest; a policy
  gate pass; and, over months, shadow and live measurements that agree.
- **Toward "stop":** the Phase-3 search, with the levers now on, again failing to beat a zero-skill
  ranker; or a certificate that passes but whose live results over the pre-registered horizon fall
  inside what luck predicts.
- **Not evidence either way:** in-search scores, a single good week of paper trading, or any
  verdict computed on journals written by an older policy.

## 14.10 Further reading

- Campbell R. Harvey, Yan Liu and Heqing Zhu, "... and the Cross-Section of Expected Returns,"
  *Review of Financial Studies* 29(1), 2016, 5 to 68. Why the bar for a new finding should be a
  t-statistic near 3, not 2.
- Richard C. Grinold and Ronald N. Kahn, *Active Portfolio Management*, 2nd ed., McGraw-Hill, 2000.
  The fundamental law.
- Robert Novy-Marx and Mihail Velikov, "A Taxonomy of Anomalies and Their Trading Costs," *Review of
  Financial Studies* 29(1), 2016, 104 to 147. How costs erase high-turnover strategies.
- Ryan Sullivan, Allan Timmermann and Halbert White, "Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap," *Journal of Finance* 54(5), 1999, 1647 to 1691.
- In this repository: [`research/campaign_2026-09_jetson/README.md`](../../research/campaign_2026-09_jetson/README.md)
  (the night's report), [`research_signal.md`](../../research/campaign_2026-09_jetson/research_signal.md),
  the stock decomposition in
  [`session_artifacts/decomp/RESULT_stock.md`](../../research/campaign_2026-09_jetson/session_artifacts/decomp/RESULT_stock.md),
  and the SIGNAL and ENGINE owner items under
  [`session_artifacts/generals/`](../../research/campaign_2026-09_jetson/session_artifacts/generals/signal/OWNER_ITEMS.md).
