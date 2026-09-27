# Chapter 5. The forecaster

*Two models, one number. What the neural network and the tree ensemble each learn, how they are
blended, what the "q10" veto is, and why a healthy prediction is a surprisingly small number.*

---

## 1. The idea in plain words

For every symbol, every hour, the forecaster produces one number: its best guess of the percent
return over the chosen horizon (12 to 48 hours; Chapter 4). A prediction of `0.35` means "I expect
+0.35%." That number is the blend of two very different models:

- **An LSTM with attention** (`model_v2.RegressionLSTM`). A **neural network** is a large, flexible
  function with many adjustable weights that are tuned by showing it examples. An **LSTM** (Long
  Short-Term Memory network) is a kind of neural network built for sequences: it reads the last
  `seq_len` hourly rows of features one at a time, carrying a running memory forward, much as you
  would read a chart left to right. **Attention** is an added layer that lets the network look back
  over all those hours and weight the ones that matter most for this prediction.
- **A gradient-boosted tree ensemble** (LightGBM, `model_lgb.py`). A **decision tree** asks a series
  of yes/no questions ("is RSI above 62? is the 4-hour return below -1%?") and outputs an average for
  each final branch. **Gradient boosting** builds hundreds of small trees, each one trained to correct
  the errors of all the trees before it. Trees are excellent at thresholds and interactions in
  table-shaped data and need no scaling of inputs.

A third model rides along: a **quantile** version of the tree ensemble (the "q10" model) that predicts
not the average outcome but a pessimistic one, the 10th percentile. It is used only to veto trades
whose downside looks unusually bad.

## 2. Why it matters financially

### Why blend at all

Two models that make different kinds of mistakes, averaged, usually make smaller mistakes than either
alone. This is the **forecast combination** result (Timmermann, 2006): if two forecasts have errors
that are not perfectly correlated, a weighted average has lower error variance. A classic finding in
that literature (the "forecast combination puzzle") is that simple fixed weights such as 50/50 often
beat weights estimated from data, because estimated weights are themselves noisy. This system's
blend code is built around that warning.

### Why predictions are tiny

New readers are often surprised that a "good" prediction is +0.2% rather than +5%. The reason is
arithmetic, and it matters for every threshold in the system.

The model is trained to minimize prediction error, and the prediction that minimizes squared-style
error is the **conditional average**: the average outcome among all past situations that looked like
this one. Most of a price move is noise that no feature can foresee. Suppose a 24-hour return has a
standard deviation of 3% and the features explain a correlation of 0.05 with it (an **information
coefficient**, IC, of 0.05, which would be excellent for hourly data). Then the best possible
predictions have a standard deviation of only about

  0.05 x 3% = **0.15%**,

and a prediction two standard deviations out, a strong signal, is about +0.30%. The forecaster is
correct to be timid: large predictions would be overconfident.

Now put that next to Chapter 2's costs. The crypto admission floor is 1.20%. A healthy crypto
forecaster almost never predicts a move that large, so an honest crypto model will rarely be allowed
to trade at all. For stocks the floor is about 0.23%, which a strong signal can clear. This is why the
threshold search, the cost floor and the prediction scale must be designed together.

## 3. How this system does it

### The LSTM leg

`model_v2.RegressionLSTM(input_dim, hidden_dim, num_layers, dropout, n_heads)` is 42 lines:

1. a stacked `nn.LSTM` reads the window `(seq_len, features)`;
2. `nn.MultiheadAttention` lets every time step attend to every other;
3. the attention output is added back to the LSTM output (a **residual** connection) and normalized
   (`LayerNorm`);
4. the steps are averaged (**mean pooling**) into one vector;
5. a small head (`Linear -> ReLU -> Dropout -> Linear`) outputs one number, the predicted return in
   percent.

The trainer (`scripts/hypersearch_v2.py`) searches its size: `hidden_dim` 64 to 384, 1 or 2 layers, 2
or 4 attention heads, dropout 0.10 to 0.40, window `seq_len` 8 to 40 bars, learning rate, batch size,
weight decay, and the horizon `forward_bars` (`adaptive_config.DEFAULT_SEARCH_SPACE`). The search uses
**Optuna**, a library that tries configurations one after another and, after a start-up phase of
random tries, steers toward promising regions (the TPE method). Chapter 6 explains why every one of
those tries is counted against the winner.

Training details that matter:

- **Loss function.** The error measure is the **Huber loss**, which is squared error for small
  mistakes and absolute error for large ones, so a few wild hours cannot dominate the fit. Its
  switch-over point `huber_delta` is searched between 0.5 and 2.0. Each row's loss is weighted by
  `clamp(|y| + 1, max=50)`, so rows with big realized moves count more (`hypersearch_v2.py`, the
  `torch.clamp(torch.abs(yb) + 1.0, max=50.0)` lines).
- **Scaling.** Features are scaled with a `RobustScaler` (median and interquartile range, so outliers
  do not distort it), fitted on each fold's **training rows only**.
- **Early stopping and a checkpoint soup.** Training stops when validation error stops improving, and
  the weights of the best `SOUP_K = 4` epochs are averaged into one model, which tends to generalize a
  little better than any single checkpoint.
- **Serving.** `predict_now.load_model` loads the network and traces it with `torch.jit.trace` for
  faster CPU inference on the Jetson.

### The LightGBM leg

The tree model sees the same window, **flattened** into one long row: `seq_len x features` columns in
a fixed order (`model_lgb.flatten_sequence`, pinned to equal `windows.reshape(-1)` so training and
serving agree). Its defaults in `model_lgb.train_lgb`: `num_leaves = 63`, `learning_rate = 0.05`,
`feature_fraction = 0.7` (each tree sees a random 70% of columns), `bagging_fraction = 0.8`, up to 500
rounds with early stopping after 20 rounds without improvement.

Because flattening multiplies columns by `seq_len`, memory on the 8 GB Jetson caps how many rows the
tree leg can train on: `min(LGB_MAX_ROWS = 120,000, max(20,000, LGB_X_BYTE_BUDGET = 600 MB / row
bytes))`, keeping the most recent rows. SCOUT-2 computed what that means in practice
(`research_signal.md` C1): the crypto tree leg sees about 61% of its available training pool at short
windows; the stock leg sees 84% at `seq_len` up to 18 but only 25% at 64. The "stronger learner" can
end up seeing the least history, and its effective training window is set by a hyperparameter chosen
for the LSTM. A lag-subset redesign is written up but not built.

### The blend

Serving: `predict_now.get_live_prediction` computes both legs and calls
`model_lgb.ensemble_predict(lstm_pred, lgb_pred, lstm_weight=config.get('lstm_weight', 0.6))`, i.e.

  prediction = w x LSTM + (1 - w) x LightGBM, with w = 0.6 by default.

The default lives in one place, `blend_fit.DEFAULT_LSTM_WEIGHT = 0.6`. Under `HYPERSEARCH_V3`
(switched ON this morning) the trainer also estimates w from out-of-fold predictions with
`blend_fit.fit_blend_weight_v2`: a non-negative least-squares fit, a significance test that respects
overlapping labels, shrinkage halfway toward the simple average, and smoothing against the previous
champion's weight within [0.25, 0.75] (`blend_fit.smooth_across_retrains`). Under V3 the holdout
certificate is also issued on the **blended** predictor with the q10 veto applied, so "the certified
predictor IS the deployed predictor" (`strategy_config.py` comment on `HYPERSEARCH_V3`). The
sub-flag `BLEND_FIT_ON_REFIT = False` means the deployed weight still comes from the older of two
fits; both are logged side by side.

Worked example: the LSTM predicts +0.50% and the tree model +0.20%. With w = 0.6 the blend is
0.6 x 0.50 + 0.4 x 0.20 = **+0.38%**. For a stock with a 0.23% floor and a searched threshold of
0.30%, that clears both; for crypto it is far below 1.20%.

### The q10 tail veto

A **quantile regression** predicts a chosen percentile of the outcome rather than its average. The
trainer fits a LightGBM model with `objective='quantile', alpha=0.10`
(`hypersearch_v2.train_lgb_ensemble`), so its output answers "in the worst 10% of similar situations,
what return did we see?" The **floor** is the 15th percentile of that model's predictions on the
validation rows (`floor = np.percentile(q10_val, 15)`), saved in `{prefix}lgb_q10_meta.json`. Live,
`predict_now` puts `Q10` and `Q10_Floor` in the snapshot, and `base_loop._execute_buys` skips any buy
where `Q10 < Q10_Floor`, journaling a `q10_tail_veto`. In words: if the downside for this setup looks
worse than for 85% of historical setups, pass, even if the average prediction is positive.

Worked example: the blend says +0.38%, but the q10 model says -1.40% against a floor of -1.10%. The
average case looks fine; the bad case looks unusually bad. The trade is vetoed.

### Serving hygiene

`predict_now` uses closed bars only. The boosters are cached by file modification time
(`serving_cache.py`), fixing a past race where a new LSTM could be served with last week's trees.
`prediction_cache.py` reuses a result within the same bar, so about 119 of 120 cycles per hour skip
recomputation. A missing artifact means no prediction and therefore no entry.

## 4. What the evidence says so far

- **The April models were not what the docs described.** The 2026-09-26 serving audit found the April
  artifacts served **LSTM-only**, silently: there were no tree or q10 boosters and no manifest on the
  box (`research/campaign_2026-09_jetson/README.md` §2). A full rebuild was mandatory.
- **No certified forecaster exists yet.** Crypto: 43 of 44 trials negative. Stocks: best holdout
  Sharpe 0.13, Deflated Sharpe 0.058 against 0.60 (README §4). The 70-trial stock retrain with the
  Phase-3 bundle is running as this is written.
- **Prediction scale versus floor, measured.** A Phase-3 verification slice found blended predictions
  peaking at only 0.64 to 0.73 (`research_signal.md`, SCOUT-4 T1). On crypto that means a correctly
  cost-anchored model will make zero trades and score 0.0, which the search treats as better than any
  negative score. The note flags this "abstention hazard" as an owner item.
- **Is the LSTM earning its memory?** The literature the campaign surveyed leans toward trees on
  hourly crypto data: Bysik and Ślepaczuk (arXiv 2606.00060, 2026) found gradient boosting
  "descriptively stronger" than an LSTM on hourly Bitcoin, though not with formal statistical
  dominance, and that the main obstacle was "the way forecasts are converted into trades."
  Grinsztajn, Oyallon and Varoquaux (2022) found tree models still beat deep learning on typical
  tabular data. PyTorch is also the largest memory item in the bot process. SCOUT-1 (S1-05) wrote a
  pre-registered test: if the fitted LSTM weight is not significantly above zero on two consecutive
  retrains and a tree-only model is no worse, propose a torch-free serving path. Nothing has been
  decided.
- **The q10 floor is in-sample and its coverage is untested.** The floor is computed on validation
  rows that, under the full-refit option, sit inside the training window (a code comment in
  `train_lgb_ensemble` says so), and "q10 holdout coverage has NO tool yet."

## 5. Further reading

- Sepp Hochreiter and Jürgen Schmidhuber, "Long Short-Term Memory," *Neural Computation* 9(8), 1997.
- Ashish Vaswani et al., "Attention Is All You Need," *Advances in Neural Information Processing
  Systems* 30 (NeurIPS), 2017.
- Guolin Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural
  Information Processing Systems* 30 (NeurIPS), 2017.
- Roger Koenker and Gilbert Bassett Jr., "Regression Quantiles," *Econometrica* 46(1), 1978. The idea
  behind the q10 model.
- Allan Timmermann, "Forecast Combinations," in *Handbook of Economic Forecasting*, Volume 1, Elsevier,
  2006. Why and how to blend forecasts, and why simple weights are hard to beat.
- Léo Grinsztajn, Edouard Oyallon and Gaël Varoquaux, "Why Do Tree-Based Models Still Outperform Deep
  Learning on Typical Tabular Data?" NeurIPS Datasets and Benchmarks Track, 2022.
