# Chapter 7. The gate stack and meta-labeling

*Between "the model says buy" and "an order goes out" sit about eighteen checks. This chapter walks
through them, explains meta-labeling (a second model that judges the first), and explains why some
gates fail open and others fail closed.*

---

## 1. The idea in plain words

A forecast is not a decision. Even a good forecaster should not act on every prediction: some are too
small to pay for their costs, some arrive at moments when the market is dominated by a scheduled news
event, some conflict with what the account already holds, and some come from setups where the model
has historically been wrong more than it is right. A **gate** is a check that can stop a trade (a
**veto**) or shrink it (a **multiplier** or **tilt**). The **gate stack** is the ordered list of those
checks.

Two ideas make the stack more than a pile of rules:

- **Every veto is written down with its reason.** When a gate says no, the loop appends a `skip` row
  to the day's journal (`base_loop._journal_skip`). Later, `decision_report.py` replays what the
  skipped trade would have done using the same exit kernel as the backtest, so each gate's cost and
  benefit can be measured. A gate that blocks more winners than losers is costing money.
- **Meta-labeling.** The primary model (Chapter 5) decides *direction and size of move*. A second
  model, the **meta-labeler**, is trained to answer a narrower question: "given that the primary model
  wants to buy here, what is the probability this trade actually makes money after costs?" This idea
  comes from López de Prado (2018) and was formalized by Joubert (2022). It lets a simple, fairly
  sensitive primary signal be paired with a filter that learns when that signal tends to be wrong.

## 2. Why it matters financially

**Loss avoidance compounds.** Suppose the primary model's candidates split into two groups: 60 that
average +0.30% net and 40 that average -0.50% net. Taking all 100 averages
(60 x 0.30 - 40 x 0.50) / 100 = **-0.02%** per trade: a losing system. A filter that removes
three-quarters of the bad group and one-sixth of the good group leaves 50 good and 10 bad, averaging
(50 x 0.30 - 10 x 0.50) / 60 = **+0.17%**: a winning one, from the same forecasts. That is the
financial case for gates. The 2026-07 review judged that loss avoidance through the gate stack, plus
cost engineering, is the most reliably valuable part of the current system (`docs/MAP.md` §1).

**Over-gating is a cost too.** Every gate also removes some good trades. A stack of ten gates, each
blocking a "mere" 10% of good trades, passes only 0.9^10 = 35% of them. And gates that react to the
same underlying condition (for example several that all respond to high volatility) can silently
multiply into a near-total halt. That is why each gate is journaled and why the campaign built
`scripts/sizing_cofire_report.py` to measure how often multipliers fire together.

**Fail-open versus fail-closed is a financial choice.** If a data feed breaks:

- failing **closed** (no trade) costs you opportunities but cannot lose money through ignorance;
- failing **open** (trade as if neutral) keeps the book running but acts without that input.

The rule here (`research/AGENT_CONTEXT.md`, convention 3; `docs/MAP.md` §6): **core inputs fail
closed** (no model, no prediction or no quote means no entry), while **optional advisors fail open**
(an LLM outage or a missing meta-model must never freeze the whole book).

## 3. How this system does it

### Book-level gates (before any symbol is examined)

`base_loop._entries_allowed` checks, in order: the manual halt file (`trading_halt.flag`); the
**macro stand-down** from `macro_calendar.macro_standdown`, which blocks new entries from 12:00 to
15:30 New York time on FOMC statement days and from 06:30 to 09:30 on CPI release days (exits keep
running); and a warning if the event calendar has run out. The stock book adds: already flattened
today, inside an entry window (09:45 to 11:00 and 14:30 to 15:30), the SPY 200-day trend computed, and
current exposure known (fail-closed). The **circuit breaker** (Chapter 8) sits above all of these.

Note a gap the docs flag honestly: there is **no stand-down for the monthly jobs report (NFP)**,
though the 2026-08 literature pass recommended one (`docs/MAP.md` §9 item 14).

### The per-symbol funnel

`base_loop._execute_buys` walks each candidate through this order (the stock loop iterates only its
top-7 ranked names and evaluates the threshold before the cost gate, a documented difference):

| # | Gate | Rule in code | Kind |
|---|---|---|---|
| 1 | cooldown | no re-entry within `cooldown_min` (crypto 60, stock 20 minutes) | veto |
| 2 | hard-stop lockout | 24 hours after a hard-stop exit on that name (`lockout_hours`) | veto |
| 3 | daily budget | at most 4 crypto / 3 stock entries per name per day (`MAX_TRADES_PER_SYMBOL_PER_DAY`) | veto |
| 4 | position cap | already at `MAX_NOTIONAL_PER_SYMBOL` | veto |
| 5, 6 | prediction and quote present | missing means no entry | fail-closed veto |
| 7 | cost gate | `order_utils.should_trade`: prediction must beat 2x round-trip cost at the live spread (Chapter 2) | veto |
| 8 | threshold | prediction >= the certified `trade_threshold` | veto |
| 9 | winner's curse | if price > 20-bar average + 2 ATR, require 1.5x the threshold | veto |
| 10 | correlation | `portfolio.check_portfolio_correlation` against current holdings | veto |
| 11 | VIX halt | VIX above 35 | veto |
| 12 | VIX-25 block | VIX above 25 blocks everything except a safe-haven list (stocks) | veto |
| stock | trend filter | SPY below its 200-day average blocks non-safe-havens | veto |
| 13 | sentiment | `sentiment.sentiment_gate`, clamped to [0.15, 1.5] | multiplier only |
| 14 | LLM | veto if score s < 0.15, else multiplier 0.5 + s | veto + multiplier |
| 15 | meta-label | veto if p < 0.30, else multiplier clip(2p, 0.6, 1.3) | veto + multiplier |
| 16 | q10 tail | `Q10 < Q10_Floor` (Chapter 5) | veto |
| 17 | sizing | the size works out to zero (Chapter 8) | veto |

The **VIX** is the Cboe Volatility Index, the market's implied expectation of S&P 500 volatility over
the next 30 days, often called the "fear gauge." Readings around 12 to 20 are calm; above 30 is
stress. The thresholds live in `types_mod.MacroRegime`. The list of safe havens (gold, silver,
consumer staples, telecoms and similar) is `stock_config.SAFE_HAVEN_SYMBOLS`.

### Meta-labeling in this system

`meta_label.py` builds its training set by **replaying** the primary model's predictions through the
same exit kernel the backtest uses (Chapter 4), at **half** the live threshold
(`META_THRESHOLD_FRACTION = 0.5`) so the meta-model sees many marginal trades it should learn to
reject. Each replayed trade is labeled 1 if it made money net of costs, else 0. The final ~12% of rows
are excluded so the policy gate stays honest. The classifier is a LightGBM model on a small feature
set (`META_FEATURES`: the primary prediction, RSI, price-to-average ratio, Bollinger width, 12-hour
volatility, 4- and 12-hour returns, ATR percent, sentiment, Hurst, the hour of day and a few others),
and its raw score is turned into a **calibrated probability** (a number that means what it says: of
trades given 0.60, about 60% should win).

Live, `base_loop._meta_gate` calls `meta_label.meta_probability_live`. If p < `META_VETO_PROB = 0.30`
the trade is skipped (`meta_veto`); otherwise `meta_label.meta_size_mult(p) = clip(2p, 0.6, 1.3)` is
multiplied into the size. Worked examples:

- p = 0.25: vetoed.
- p = 0.40: allowed, size x 0.8.
- p = 0.50: neutral, size x 1.0.
- p = 0.70: size x 1.3 (the cap; 2 x 0.70 = 1.4 is clipped).

If no meta-model exists or it fails to load, `meta_probability_live` returns None and the gate passes
with a neutral 1.0: fail-open by documented intent.

### The LLM gate

`llm_analyst.analyze_trades` sends one batched, schema-enforced request per cycle of
`LLM_INTERVAL_SEC = 600` seconds, containing the candidates with their headlines, fundamentals, macro
context and the Fear and Greed reading, and asks for a conviction score s between 0 and 1 per symbol.
The LLM has exactly three powers (`docs/MODULES.md` §8):

1. **veto** a new entry when s < `trading_utils.LLM_VETO_THRESHOLD = 0.15`;
2. **force a sale** of a held position after **two consecutive** vetoing analyses
   (`base_loop._execute_llm_veto_sells`); two readings are required because news headlines are an
   untrusted channel that could carry adversarial text (**prompt injection**);
3. **tilt size** by `llm_mult = 0.5 + s` (so s = 0.5 is neutral and s = 0.8 gives 1.3x before the
   overall cap).

Any failure returns an empty result, the loop falls back to the prior score or s = 0.5, and the trade
proceeds: fail-open. Spending is capped at $1 per day with a cross-process ledger (`llm_cost.json`).
`llm_client.py` can speak to Gemini, Anthropic and OpenAI-compatible endpoints. The 2026-09-26 audit
found this box configured with a Gemini key only, so the analyst runs on `gemini-2.5-flash-lite`, and
a test call cost $0.000021 (the LLM audit summarized in `research/campaign_2026-09_jetson/README.md` §2).

### Fail-open versus fail-closed, collected

| Fails closed (no entry) | Fails open (neutral) |
|---|---|
| missing model, prediction, or quote; stale quote over 180 s | LLM provider error (s = 0.5) |
| circuit-breaker API error (buys suspended) | meta-model absent (multiplier 1.0) |
| stock exposure unknown | earnings calendar and SEC EDGAR event checks for new entries |
| overnight sleeve without an earnings calendar | macro stand-down check on exception |

## 4. What the evidence says so far

- **The historical journals are too old to judge the gates.** The decision journals on the box run
  from 2026-02-23 to 2026-05-07 and hold 11,818 skip rows, 264 buy rows and 0 sell rows in an older
  format without fill prices (`research_engine.md` §X2(b)). The campaign's measurement audit found
  that instrument verdicts on those journals are void because of schema drift (README §2). No gate has
  yet been priced on data from the current code.
- **No meta-model exists on the box**, so today the meta gate is always neutral (the device facts list
  "NO meta_* artifacts anywhere"). Last night meta-label training ran only its no-champion path,
  because there was no certified primary model to replay.
- **A meta-calibration leak was identified.** In the default ("legacy") mode the isotonic calibrator
  is fit on the same 20% slice the meta booster used for early stopping, so no row is held out from
  both (`research_signal.md` SCOUT-2 C3 and SCOUT-4 T2). The campaign landed a measurement-only tool,
  `scripts/meta_calib_nested.py`, for an honest nested comparison; switching calibration mode awaits
  its result on real data. The meta-labeler's `pred` feature is also the primary model's in-sample
  fit unless `META_OOF_PRED` is enabled (default off).
- **The VIX-25 block is effectively a stock halt** in stress, because the safe-haven names are mostly
  not in the tradable top-7 (`docs/MAP.md` §9 item 9); a flag to remove it (`VIX25_BLOCK_REMOVED`)
  exists, default off, pending measurement.
- **Sentiment is counted three times** (as a model feature, as a size multiplier, and inside the LLM
  prompt), and **VIX twice** in sizing. Both are documented as unmeasured over-counting
  (`docs/MAP.md` §9 item 11).
- **Is the LLM worth it?** `llm_eval.py` regresses realized returns on the model's prediction and the
  LLM score; the keep-or-kill verdict needs at least 120 distinct hourly clusters (about 20 days of
  live LLM cycles). It has never been reached. An INTEL audit found the production test
  over-rejects at small samples and proposed an alternative estimator (README §5).
- **Kill list reminder:** using the shadow DM-HLN test to approve *policy* changes (such as a new
  gate) is killed as a category error: that test compares forecast errors, not trading rules
  (`research/KILL_LIST.md`).

## 5. Further reading

- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018, section 3.6
  (meta-labeling).
- Jacques Francois Joubert, "Meta-Labeling: Theory and Framework," *Journal of Financial Data Science*
  4(3), 2022.
- Michael Meyer, Illya Barziy and Jacques Francois Joubert, "Meta-Labeling: Calibration and Position
  Sizing," *Journal of Financial Data Science* 5(2), 2023.
- Alexandru Niculescu-Mizil and Rich Caruana, "Predicting Good Probabilities with Supervised
  Learning," *Proceedings of the 22nd International Conference on Machine Learning* (ICML), 2005.
  Platt scaling versus isotonic calibration.
- David O. Lucca and Emanuel Moench, "The Pre-FOMC Announcement Drift," *Journal of Finance* 70(1),
  2015. Why scheduled Fed announcements dominate price action around them.
- Robert E. Whaley, "The Investor Fear Gauge," *Journal of Portfolio Management* 26(3), 2000. What the
  VIX measures.
