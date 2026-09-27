# Chapter 4. Labels: what "the model predicts" means

*A forecast is only as meaningful as the question it answers. This chapter is about choosing the
question, and about why the question used in training must be the same one the backtest scores.*

---

## 1. The idea in plain words

In supervised machine learning, every training row has an **input** (the features of Chapter 3) and an
**answer**, called the **label** or **target**. The model learns to produce answers like the labels
from inputs like the features. Whatever the label measures, that is what "the model predicts."

This system offers the trainer two kinds of label, and the training search picks one per trial
(`target_kind` in `scripts/hypersearch_v2.py`):

1. **The fixed-horizon forward return** ("raw"). For a horizon of `fb` bars,
   `Target_Return_fb = (close[t + fb] - close[t]) / close[t] x 100`, the percent change from this bar's
   close to the close `fb` hours later (`scripts/harvest_crypto_data.py`; the stock harvest is the
   same). The searched horizons are 12, 18, 24, 32 and 48 bars (`adaptive_config.DEFAULT_SEARCH_SPACE
   ['forward_bars']`). This label answers: "if I bought now and held exactly `fb` hours, what would I
   make?"
2. **The triple-barrier return** ("tb"). Imagine three barriers around an entry: an upper barrier
   (take profit), a lower barrier (stop loss), and a vertical barrier (a time limit). The trade ends at
   whichever barrier price touches first. The label `TB_Ret_fb` is the gross percent return realized
   that way. This label answers: "if I bought now and let my actual exit rules run, what would I
   make?"

The second question is the one a trader cares about, because nobody in this system holds for exactly
24 hours. The bot exits on stops, trailing stops, take-profits, signal flips and, for stocks, the end
of the day.

## 2. Why it matters financially

A worked example shows how far apart the two questions can be.

Bitcoin closes an hour at **$60,000**. Its **ATR** (Average True Range, the average size of an hourly
bar's full range including gaps, 14 bars) is **$500**. The crypto exit policy in
`strategy_config.CRYPTO_POLICY` says: stop at 2.5 ATR below entry, take profit at 2 times the stop
distance, trailing stop at 2.0 ATR that arms once price is 1.5% above entry. So:

- stop distance = 2.5 x 500 / 60,000 = 2.083%, stop price **$58,750**;
- take-profit distance = 2 x 2.083% = 4.167%, take-profit price **$62,500**;
- trailing distance = 2.0 x 500 / 60,000 = 1.667%, arming level 60,000 x 1.015 = **$60,900**.

Now the price path. In hour 3 the high reaches $61,400, which arms the trail. In hour 4 the bar opens
at $60,900 and trades down to $60,200. The trailing stop sits at 61,400 x (1 - 0.01667) = **$60,377**,
so the position exits there. Over the next day Bitcoin drifts down and closes hour 24 at **$59,400**.

- The raw 24-bar label says: (59,400 - 60,000) / 60,000 = **-1.00%**. "Bad entry."
- The triple-barrier label says: (60,377 - 60,000) / 60,000 = **+0.63%** gross. "Good entry" (before
  the 0.60% round-trip cost, which is charged downstream, so +0.03% net).

A model trained on the first label is taught to avoid this entry; a model trained on the second is
taught to take it. Only one of them describes what the bot would actually have experienced. Train on
the wrong question and even a perfect forecaster of it can lose money, because it optimizes a game
you are not playing.

The same logic forces a second requirement: **the labels, the backtest and the live exits must all
use the same exit rules.** If the backtest scored trades with different stops than the labels
assumed, the backtest would be grading the model on a different game from the one it studied, and a
promotion decision based on it would be noise.

## 3. How this system does it

### One exit kernel, several consumers

`policy_exits.exit_walk` is the single implementation of the exit stack. For an entry at **every**
bar's close it walks forward bar by bar and records the exit bar, price and reason. It is compiled
with Numba for speed and has an identical pure-Python fallback. Its consumers (`policy_exits.py`
module docstring, `docs/MAP.md` §4f):

| Consumer | Parameters | Question answered |
|---|---|---|
| harvest labels, `policy_exits.compute_tb_labels` | vertical barrier = `fb` bars, no signal exit | "what does the exit stack realize from here within `fb` hours?" |
| promotion backtest, `backtest.simulate_ticker` | no time limit, signal exit on | "what would this model's actual policy have made?" |
| meta-labeler, `meta_label._gen_meta_rows` | same as the backtest | "did acting on this signal make money after costs?" (Chapter 7) |
| gate attribution, `decision_report.replay_entry` | 24-bar limit, no signal exit | "what did a skipped trade give up?" |

Because they call one kernel with the same policy numbers from `strategy_config.py`, labels and
backtest agree by construction. The labels are **gross** (before costs); each consumer charges costs
itself using `fees.py` (Chapter 2).

### The rules, precisely

The kernel's docstring lists "normative" rules that any reimplementation must match. The important
ones, in plain words:

1. **Entry is at the bar's close**, and the walk starts on the next bar. The entry bar's own high and
   low are never examined (you cannot be stopped out by prices from before you bought).
2. **Each bar is checked in a fixed order:** stop first (the hard stop, or the trailing stop if it is
   armed and higher), then take-profit, then update the high-water mark and possibly arm the trail,
   then the signal-flip exit, then the end-of-day flatten. Checking the stop before the take-profit
   when both are touched in the same bar is the **conservative** choice: with only hourly data you
   cannot know which happened first, so assume the bad one.
3. **Gaps are honored.** If a bar opens below the stop, the fill is at the (worse) open, not at the
   stop price: `exit = min(open, stop)`.
4. **The high-water mark (HWM)** is the highest price since entry. It updates only after that bar's
   stop and take-profit checks, so a bar that arms or raises the trail can never also fire it.
5. **The vertical barrier** loses every same-bar tie: if a stop, signal or end-of-day exit lands on
   the last allowed bar, that reason wins.

The stop distances come from the policy dictionaries, clipped to sane bounds. For crypto
(`CRYPTO_POLICY`): stop 2.5 ATR, trail 2.0 ATR, floor 1.5%, ceiling 15%, take-profit 2.0 times the
stop distance capped at 30%, trail arming at +1.5%. For stocks (`STOCK_POLICY`): stop 2.0 ATR, trail
2.0 ATR, floor 1.0%, ceiling 10%, take-profit 2.0 times capped at 15%, trail arming at +1.0%. If ATR is
missing the kernel falls back to fixed percentages (6% and 5% stop and trail for crypto).

### Stocks: the end of day is a barrier too

`compute_tb_labels` treats each stock session's last bar as a barrier (an approximation of the live
flatten near 15:50 New York time). A consequence documented in its docstring: because the day always
ends within one session, every horizon longer than a session produces **identical** stock labels.
For stocks, "predict 32 hours ahead" and "predict 48 hours ahead" are the same question under the
triple-barrier label.

### Live is a mirror, not a caller

The live loops do **not** import the kernel (a test, `tests/test_ia4_flagged.py`, pins this), because
they must act on one 30-second quote at a time rather than on a completed history. They re-implement
the same rules in `base_loop._desired_stop_for` and `base_loop._manage_stops` (plus copies in
`stock_loop`), reading the same numbers from `strategy_config.py`. So "label equals backtest" holds
exactly, and "live equals kernel" holds up to a documented list of differences in the
`policy_exits.py` docstring. The notable ones:

- live confirms a stop breach on **two consecutive readings** before selling;
- live tracks the high-water mark on quote **midpoints**, the kernel on bar **highs**;
- in risk-off macro regimes live tightens stops by a multiplier;
- the **trailing denominator**: the kernel sets the trail distance as ATR x 2.0 / *entry* price and
  enforces HWM x (1 - distance), while `base_loop._desired_stop_for` divides by the *HWM*. In the
  example above, with the same $500 entry ATR, the kernel's trail is $60,377 and the crypto live trail
  is 61,400 x (1 - 1000/61,400) = $60,400, about $23 (0.04%) tighter. Small, but it is a real
  difference between what the labels assumed and what the bot does, and `stock_loop` agrees with the
  kernel, not with `base_loop`. It is an open owner decision, not a bug fix anyone may make alone.

### Why overlapping labels need special care

A 24-hour label computed at 14:00 and another computed at 15:00 share 23 of their 24 hours. They are
not independent observations. Treating 1,000 such rows as 1,000 independent facts overstates how much
you know. `sample_weights.py` measures each row's **average uniqueness** (how much of its label window
it shares with others) and turns that into an **effective sample size**, which Chapter 6 uses to keep
the validation honest.

## 4. What the evidence says so far

- **Labels were rebuilt exactly.** The clean rebuild re-derived the triple-barrier labels on both
  stores (`research/campaign_2026-09_jetson/README.md` §4).
- **A real label bug was fixed.** `TB_Bars_fb` (bars held) is a positional offset in the frame passed
  to the kernel, and the `compute_tb_labels` docstring warns it becomes invalid after any row
  filtering. Last night's fix round moved the triple-barrier stamping to after the row filters (as-of
  masks, drop of incomplete rows), so the holding-period counts that feed the uniqueness weights are
  correct (README §3).
- **The search uses both label kinds.** In this morning's stock retrain log the trials so far draw
  `target_kind` both `raw` and `tb`. With 60 random start-up trials before the optimizer steers
  (`hypersearch_v2.PRUNE_STARTUP_TRIALS = 60`), a 70-trial run is still mostly random search, so which
  label "wins" tonight is not evidence of anything yet.
- **Cost-aware labels versus cost-aware scoring.** Labels are gross; the trainer's score charges
  `TXN_COST_PCT` per trade. SCOUT-4 found the trainer scored trades at "pred > threshold" while the
  live book trades at "pred >= max(threshold, admission floor)" (Chapter 2). The labels themselves are
  sound; the scoring had drifted from the policy, and the Phase-3 bundle now running narrows that gap.
- **The live mirror has never been reconciled on real fills in this code version.** The bots have not
  traded with a certified model since the rebuild, so the size of the live-versus-kernel difference in
  practice is unmeasured.

## 5. Further reading

- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, chapter 3 (labeling
  and the triple-barrier method) and chapter 4 (sample weights and average uniqueness).
- Robert Pardo, *The Evaluation and Optimization of Trading Strategies*, 2nd edition, Wiley, 2008.
  Why a strategy must be tested with its real entry and exit rules.
- Ernest P. Chan, *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*, Wiley,
  2008 (2nd edition 2021). A practitioner's view of backtest and live consistency.
- J. Welles Wilder Jr., *New Concepts in Technical Trading Systems*, Trend Research, 1978. The source of
  the Average True Range used to size every stop here.
