# Chapter 10. From data to a deployed model

## 10.1 The idea

A model does not deserve money because it was trained. It deserves money only after it has
survived a sequence of tests, each designed to catch a different way that a model can look good
without being good. This chapter walks the whole path a model takes in this system:

**harvest** (build the data) → **train** (search for a model) → **certify** (test it once on data
it never saw) → **gate** (replay the real trading policy) → **champion or challenger** (decide
which slot it occupies) → **shadow** (let it predict silently next to the current model on live
data) → **promote** (swap it in) → **hot reload** (the running bots pick it up).

Two vocabulary words carry the chapter.

- The **champion** is the model the bots trade with. Its files are named `model_v2.*` for crypto
  and `stock_model_v2.*` for stocks.
- A **challenger** is a newer model waiting to prove itself. Its files carry a `challenger_`
  prefix. It trades nothing. It only predicts, and its predictions are compared with the
  champion's.

## 10.2 Why it matters financially

The single most expensive mistake in systematic trading is deploying a model whose backtest was an
accident. If you try many model configurations and keep the best, the best will look impressive
even when none of them has any skill, simply because you picked the luckiest. This is
**selection bias**, and it is the reason most published trading strategies decay after
publication (McLean and Pontiff 2016 measured a 58 % drop in returns after publication, of which
a meaningful part is data mining).

Every stage in this pipeline exists to shrink that risk, and each one costs something: time,
data, or false rejections of models that were actually fine. A good pipeline accepts that it will
sometimes refuse a good model, because the alternative (sometimes accepting a lucky bad one)
loses real money.

## 10.3 How this system does it

### Stage 1: Harvest

[`scripts/harvest_crypto_data.py`](../../scripts/harvest_crypto_data.py) and
[`scripts/harvest_stock_data.py`](../../scripts/harvest_stock_data.py) build the two **feature
stores**, `training_data.parquet` and `stock_training_data.parquet`. For each book they fetch
hourly price bars, compute features with **the same functions the live loop calls**
(`indicators.compute_features` and `compute_stock_features`), merge point-in-time external data
(funding rates and open interest for crypto, short-sale volume for stocks), and apply an as-of
universe mask for stocks: a name is in the universe on a given date only if it was among the top
60 by 30-day dollar volume *on that date*. That mask is where survivorship bias is removed
(Chapter 15 explains the trap).

Then come the **labels**, the answers the model is trained to predict: the return from holding
exactly `fb` bars, and a "triple-barrier" return from `policy_exits.compute_tb_labels`, which
walks each hypothetical trade through the same exit rules (stop, take-profit, trailing stop, time
limit) that the backtester and the live loop use. Using one exit kernel everywhere is what lets
the documentation say "labels equal backtest".

The clean rebuild on 2026-09-26 produced a crypto store of 263,889 rows by 71 columns and a stock
store of 1,536,356 rows by 110 columns (92 names, all bars from Alpaca). It also fixed a real
look-ahead leak: the `Daily_Sentiment` column had been dated so that each row held the next day's
value. Every store built before that date carried the leak.

### Stage 2: Train

[`scripts/hypersearch_v2.py`](../../scripts/hypersearch_v2.py) runs an Optuna search. Each
**trial** picks a configuration (forecast horizon `forward_bars`, input window `seq_len`, network
size, dropout, learning rate, the trade threshold, and whether to use the raw or triple-barrier
target) and trains a RegressionLSTM (a small recurrent neural network) on three **walk-forward
folds**: train on the past, validate on the period right after, then roll forward. Between train
and validation it **purges** rows whose label window overlaps the validation period and adds an
**embargo** gap, so information cannot leak across the boundary.

The trial's score is computed in `_train_walk_forward` as

    score = mean(fold Sharpe) - 0.5 x std(fold Sharpe)

where each fold Sharpe comes from simulating non-overlapping trades net of costs
(`compute_sharpe`). A model that does well in one fold and badly in another is penalised for the
inconsistency. `compute_sharpe` returns exactly 0.0 when a fold produces fewer than 10 trades, a
detail that matters in Chapter 14.

The first `PRUNE_STARTUP_TRIALS = 60` trials are effectively random search: Optuna's TPE sampler
needs a warm-up before it starts steering, and this constant sets both the sampler's and the
pruner's warm-up. A 25 or 40 trial run is therefore pure random search.

### Stage 3: Certify on the holdout

Before the search starts, the last slice of time (by default about 12 % of timestamps,
`objective_utils.holdout_boundary`) is set aside as a **holdout**. The search never sees it. After
the search, the winning configuration is retrained and evaluated once on the holdout. The
certificate records the holdout Sharpe, the number of trades, the **effective** number of
independent trades (`n_eff`, which is lower than the raw count when trades overlap in time), and
the **Deflated Sharpe Ratio** (DSR).

The DSR, from Bailey and López de Prado (2014), answers one question: given how many
configurations were tried, what is the probability that this Sharpe ratio reflects real skill
rather than the luck of picking the best of many? It compares the observed Sharpe with the Sharpe
you would *expect* from the best of N skill-less trials. The model is saved only if (the
`gate_ok` expression in `hypersearch_v2.py`)

    holdout Sharpe > 0  and  DSR >= DSR_MIN (0.60, in validation.py)

Otherwise the log prints `Model NOT saved: failed holdout gate` and, pointedly, "A higher
in-search score that cannot clear unseen data is selection bias, not skill."

### Stage 4: The policy gate

The holdout test uses a simplified trade simulation. The **policy gate**, `backtest.py --gate`,
replays the saved model through the real trading policy: the same exit kernel, the two-tier cost
model, the meta-label veto and the q10 veto. [`run_pipeline.py`](../../run_pipeline.py) runs it
over the last 44 days for crypto (inside the holdout) and 60 days for stocks. It passes only if
`n_trades >= 10`, Sharpe > 0 and DSR >= 0.60. A failure exits with code 3, a deterministic
rejection that the pipeline never retries; for the champion slot it restores the previous model
from its `.prev` backup files.

### Stage 5: Champion, challenger, shadow

If a champion already exists, a retrain saves its result into the challenger slot instead
(`--shadow`, controlled by `TRADER_SHADOW_MODE`, default on). The bots then log, once an hour,
both models' predictions for every symbol into `{prefix}shadow_preds.jsonl`
(`shadow.maybe_log_shadow`, every `SHADOW_LOG_INTERVAL_SEC = 3590` seconds).

Once a day, [`shadow.py`](../../shadow.py) `evaluate_and_maybe_promote` waits until each logged
prediction's forecast horizon has passed, computes each model's squared error, and runs a
**Diebold-Mariano test with the Harvey-Leybourne-Newbold correction** (`dm_hln`). The DM test
asks whether one forecaster's errors are smaller than another's by more than chance; the HLN
correction makes it behave better in small samples. The legacy rule, in constants at the top of
`shadow.py`:

- promote early if at least `MIN_OBS = 200` paired observations, at least `MIN_SHADOW_DAYS = 14`
  days old, and p < 0.05;
- at `MAX_SHADOW_DAYS = 28` days, promote if the mean loss difference favours the challenger and
  p < 0.10, otherwise discard it.

The module's own docstring lists why this legacy test is too generous (it pools correlated
records as if independent and re-tests daily without adjustment). A corrected version, DM v2,
exists behind `TRADER_SHADOW_DM_V2` (default off).

### Stage 6: Promotion and hot reload

`shadow.promote_challenger` backs up every champion file to `.prev`, copies the challenger over,
and writes the manifest file `{prefix}model_v2.manifest.json` **last**. The bots check
`trading_utils.model_reload_key`, which is the manifest's modification time, on every cycle
(`_hot_reload_check`). When it changes, they reload. Writing the manifest last means a bot can
never load new network weights paired with an old scaler halfway through a copy.

### The weekly retrain is a cold restart

Distinguish two hand-offs. The daily promotion above is a genuine hot reload: the bots keep
running. The **weekly retrain** is not. `run_pipeline.main` computes the next retrain time with
`_next_retrain_time` (defaults `--retrain-day 5`, Saturday, and `--retrain-hour 2`), then calls
`_stop_bots` (SIGTERM, wait 10 seconds, then kill), runs training on the GPU, and calls
`_restart_bots`. The bots read the champion slot at startup. The reason is memory: on an 8 GB
device, training and bots cannot comfortably coexist (Chapter 13).

### Why flags default OFF, and the runbook

Every change that could alter predictions, labels or trading behaviour ships behind a **flag**
whose default reproduces the old behaviour exactly, and a test pins the old code path
byte-for-byte ([`docs/FLAGS.md`](../FLAGS.md), "Philosophy"). Flags are then turned on in an order
set by [`research/campaign_2026-08/03_jetson_runbook.md`](../../research/campaign_2026-08/03_jetson_runbook.md):
Phase 0 baseline reads, Phase 1 run-once instruments, Phase 2 evidence-gated flips (each one only
after its instrument's output has been read), Phase 3 one bundled retrain, Phase 4 LLM economics,
Phase 5 strategic follow-ons.

Why so much ceremony? Because if you change three things at once and results move, you cannot tell
which change did it; and because any change to the objective or features makes old Optuna scores
incomparable. That is "gotcha #2" in `CLAUDE.md`: after such a change the study databases must be
reset. The runbook therefore bundles all such changes into **one** retrain event.

## 10.4 A worked example: why 20 trades cannot certify anything

Here is the DSR gate computed with the production function `validation.dsr_from_trade_returns`.
In every row the model's true per-trade Sharpe ratio is the same, 0.20 (a respectable edge: the
average trade earns one fifth of a standard deviation). Only the amount of evidence changes.

| Holdout trades | Effective trades | Trials searched | Best Sharpe expected from luck | DSR | Passes 0.60? |
|---:|---:|---:|---:|---:|:---:|
| 20 | 14 | 25 | 0.534 | 0.13 | no |
| 20 | 14 | 70 | 0.642 | 0.07 | no |
| 200 | 140 | 25 | 0.169 | 0.64 | yes |
| 200 | 140 | 70 | 0.203 | 0.49 | no |

Read the rows slowly. With only 14 effective trades, the best of 25 *skill-less* trials would
be expected to show a per-trade Sharpe of about 0.53 by luck alone, so an honest 0.20 looks like
nothing. With ten times the trades, luck's bar falls to 0.17 and the same 0.20 edge passes. Search
harder (70 trials) and the bar rises again. The gate is a trade-off between how hard you searched
and how much evidence you have. You cannot buy certification by searching more; you can only earn
it with more independent trades.

This is not hypothetical. On 2026-09-27 the stock search (25 trials) produced a holdout of 20
trades with 14 effective, Sharpe 0.13 and DSR 0.058, and the model was not saved.

## 10.5 What the evidence says

- **The pipeline refuses correctly.** In the overnight campaign both books were searched on the
  clean stores. Crypto: 44 trials, 43 negative, and the winner produced 3 holdout trades, so the
  gate failed closed on too few trades. Stock: 25 trials, best score +0.024, failed DSR. No model
  was saved, the meta-label and policy-gate phases ran their "no champion" paths cleanly, and the
  bots stayed unable to buy. That is the system working as intended.
- **The gate has little power at realistic edges.** SIGNAL's analysis: with `DSR_MIN = 0.60`, a
  model with a true per-trade Sharpe of 0.10 passes only 1 to 8 % of the time at 10 to 120
  effective trades and a pool of 44 trials. The gate protects against luck but will also reject
  most genuinely modest edges. Remedies (a longer fixed holdout, a holdout-only deflation pool, or
  relying on shadow evidence) are an owner decision.
- **The policy gate has a structural caveat.** In the default shadow mode, the weekly policy gate
  replays the *champion*, while the fresh model sits in the challenger slot without a policy
  replay, unless `GATE_TARGETS_CHALLENGER` (default off) is on. The runbook's Phase 2 flips this
  after one observed cycle.
- **The Phase-3 bundle is now on.** On the morning of 2026-09-27, on the founder's instruction,
  `HYPERSEARCH_V3`, `OBJECTIVE_V3` and `TRAINING_REPAIRS_V1` were set to True (with
  `OBJECTIVE_LONG_ONLY` already True), the old studies were archived, and a 70-trial stock retrain
  was launched. Its `[FLAGS]` banner line records exactly which flags were in force. Chapter 14
  explains how to read the certificate it produces.

## 10.6 Further reading

- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018. Purging,
  embargoes, triple-barrier labels, meta-labeling: most of this pipeline's vocabulary.
- David H. Bailey and Marcos López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* 40(5), 2014.
- David H. Bailey, Jonathan Borwein, Marcos López de Prado and Qiji Jim Zhu, "The Probability of
  Backtest Overfitting," *Journal of Computational Finance* 20(4), 2017, 39 to 69.
- Francis X. Diebold and Roberto S. Mariano, "Comparing Predictive Accuracy," *Journal of Business
  and Economic Statistics* 13(3), 1995, 253 to 263; David Harvey, Stephen Leybourne and Paul
  Newbold, "Testing the Equality of Prediction Mean Squared Errors," *International Journal of
  Forecasting* 13(2), 1997, 281 to 291.
- R. David McLean and Jeffrey Pontiff, "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance* 71(1), 2016, 5 to 32.
- In this repository: [`docs/MAP.md`](../MAP.md) sections 4a to 4d,
  [`docs/FLAGS.md`](../FLAGS.md), and the runbook linked above.
