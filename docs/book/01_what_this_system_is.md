# Chapter 1. What this system is, and what it is not

*Part One of "the book": an explanatory guide to this trading system and the finance behind it.
Written 2026-09-27 against the live tree on the Jetson. When this chapter and the code disagree, the
code wins; where the older docs and the code disagree, the chapter says so.*

---

## 1. The idea in plain words

This repository is a robot that tries to make small, repeated, disciplined bets on prices going up.
It watches two groups of assets, which the code calls **books**:

- the **crypto book**: six coins traded against the US dollar (BTC, ETH, XRP, SOL, DOGE, LINK), open
  24 hours a day, 7 days a week;
- the **stock book**: US stocks, traded only during regular market hours (9:30 to 16:00 New York
  time), chosen each hour from a larger panel of liquid names.

Every 30 seconds the robot wakes up, looks at the most recent completed hourly price bars, asks a
forecasting model "how much is this asset likely to move over the next day or so?", and then pushes
each candidate trade through a long list of checks before it ever sends an order. Most candidates
are rejected. The ones that survive are bought in small sizes, protected by stop orders, and sold
later by a fixed set of exit rules.

A few terms, defined once:

- **Paper trading.** Trading with pretend money at a real broker. The broker here is Alpaca. Orders
  are matched against real market quotes, but no real dollars move. Alpaca's own documentation says
  paper orders fill against the best available quote, a limit buy fills only when the limit is at or
  above the ask, about 10% of orders get random partial fills, and there is no market impact or queue
  position (recorded in `research/campaign_2026-09_jetson/research_engine.md`, source W12-S2). So
  paper fills are, if anything, kinder than real ones.
- **Long-only.** The system only ever buys something and later sells what it owns. It never
  **short-sells** (borrows shares and sells them, hoping to buy back cheaper). A short-side exit
  kernel exists in `policy_exits.py` (`side=-1`) but has no production caller.
- **Book.** A self-contained trading program with its own universe, hours, rules and model. The two
  books share one Alpaca account and one equity number.
- **Edge.** The average amount a trade makes after all costs. A system has an edge only if that
  average is reliably above zero. Everything in this book circles back to that one word.

What it is **not**:

- It is not a proven money machine. As of today it has no certified forecasting model in either book
  (section 4 explains why that is the honest result).
- It is not high-frequency trading. It reasons on **hourly bars** (one summary of price per hour) and
  holds positions for hours to about two days.
- It does not use leverage on purpose, options, or shorting. Position sizes are capped in dollars
  (section 3).
- It is not "run by an AI chatbot." Large language models (LLMs) appear only as a bounded advisor
  that can veto an entry or nudge its size (Chapter 7). The core forecast is a conventional
  statistical model (Chapter 5).

## 2. Why it matters financially

Why be so explicit about what the system is? Because most retail trading systems fail for one of
three reasons, and each one has a direct counterpart here.

**Failure 1: confusing market exposure with skill.** Suppose the stock book holds seven positions of
$5,000 each ($35,000 total) in high-beta names. **Beta** measures how much a stock tends to move when
the whole market moves: a beta of 1.5 means that when the S&P 500 (tracked by the ETF **SPY**) falls
2%, the stock tends to fall about 3%. Then on a day when SPY drops 2%, this book loses roughly

  $35,000 x 1.5 x 2% = $1,050

without any forecast being wrong. In a rising year the same arithmetic produces gains that look like
genius. The 2026-07 six-agent review concluded that about 85 to 95% of the invested variance in this
system is exactly this kind of **factor variance** (moves explained by the market or by "crypto as a
whole"), not stock-specific skill (`docs/MAP.md` §1). A system that does not measure this will
believe it is smart in bull markets and unlucky in bear markets. That is why `beta_ledger.py` exists:
it regresses daily equity on SPY and BTC and reports the **alpha** (the return left over after the
market's share is removed) with an honest t-statistic.

**Failure 2: ignoring costs.** Every round trip (buy, then sell) in crypto costs about 0.60% of the
position at this broker once fees and the bid/ask spread are included (Chapter 2 derives the number).
A model that is right on average by 0.40% per trade loses money. A system that "trades often because
the model is confident" can burn an account through costs alone.

**Failure 3: fooling yourself in testing.** If you try 40 random strategies on the same history, the
best one will look good by pure luck. Chapter 6 shows the arithmetic: the luckiest of 40 coin-flip
strategies with 60 trades each typically shows an annualized Sharpe ratio around 2, which is a number
people quit their jobs over.

The design of this repository is a set of defenses against those three failures, in that order of
subtlety. Paper trading is the fourth defense: nothing here should touch real money until the
measurements say it has earned it.

## 3. How this system does it

### The two machines and the one box that matters

Production runs on an **NVIDIA Jetson Orin Nano**, a small computer with 8 GB of memory shared between
its processor and its graphics chip (GPU). The GPU trains models, one training process at a time
(`gpu_lock.py` holds a file lock so two trainers never share the 8 GB). The trading bots run on the
processor only: `CUDA_VISIBLE_DEVICES=''`, `TORCH_NUM_THREADS=2`, `OMP_NUM_THREADS=2` (`docs/MAP.md`
§4i). A development Mac is used for code and tests that need no heavy libraries. The owner's stated
priority order is: **Jetson memory and performance, then financial soundness, then LLM utilization,
then trading strategy** (`CLAUDE.md`). Notice that "trading strategy" is last. The system is built to
be a trustworthy instrument first and a strategy second.

### The process tree

- `run_pipeline.py` is the orchestrator. Its phases: **harvest** data
  (`scripts/harvest_crypto_data.py`, `scripts/harvest_stock_data.py`), **train** a model
  (`scripts/hypersearch_v2.py`), train the **meta-labeler** (`meta_label.py`), run the **promotion
  gate** (`backtest.py --gate`), then launch the bots and wait. Weekly it stops the bots, retrains, and
  restarts them (`run_pipeline._stop_bots` then `_restart_bots`).
- `run_bots.py` runs both trading loops as threads in one process, which saves about 0.5 to 0.8 GB of
  duplicated library memory on the Jetson.
- `base_loop.BaseTradingLoop` is the shared engine; `crypto_loop.CryptoLoop` and
  `stock_loop.StockLoop` fill in the book-specific parts (hours, universe, order style). The pattern is
  called **Template Method**: the base class owns the skeleton of each cycle, the subclasses override
  specific steps.

### One cycle, in order

`base_loop._run_one_cycle` does roughly this every `LOOP_INTERVAL = 30` seconds (`base_loop.py`,
`stock_loop.py`): check for a remote flatten request, check market hours, check the **circuit
breaker** (a 5% daily account drawdown stops new buys, `CIRCUIT_BREAKER_PCT = 0.05`), manage exits on
open positions **first**, hot-reload a newly promoted model if one exists, fetch predictions for every
symbol, run the LLM advisor on its own 600-second timer (`LLM_INTERVAL_SEC = 600`), execute signal
sells, then evaluate new buys through the gate funnel (Chapter 7) and size them (Chapter 8). Exits
come before entries by design: protecting what you own matters more than adding to it.

### The universe of each book

- Crypto: `stock_config.CRYPTO_SYMBOLS` lists the six coins above.
- Stocks: the harvest builds a panel of liquid US names (92 in the 2026-09-27 store) and applies an
  **as-of membership mask**: at each date, only the top 60 names by trailing 30-day dollar volume
  (`stock_config.AS_OF_TOP_K = 60`) and only names with at least $5M daily dollar volume and a price of
  at least $3 count as members (`scripts/harvest_stock_data._asof_tradability_mask`,
  `_asof_membership_mask`). Live, the stock loop ranks candidates each hour and considers entries only
  for the top 7 (`stock_loop.StockLoop.TOP_N = 7`), holding while a name stays in the top 15
  (`HOLD_RANK = 15`).
- Stock entries are further restricted to two windows, 09:45 to 11:00 and 14:30 to 15:30 New York time
  (`strategy_config.STOCK_ENTRY_WINDOWS_ET`, `ENTRY_WINDOWS_ENABLED = True`), and positions are
  flattened near the close except for a small **overnight sleeve** of at most two positions
  (`OVERNIGHT_SLEEVE_MAX_POSITIONS = 2`).

### Money limits that bind before anything clever happens

- Risk per trade: `RISK_PCT_PER_TRADE = 0.005`, i.e. 0.5% of equity lost if the stop is hit.
- Dollar caps per position: the base loop's `NOTIONAL_PER_SYMBOL = 1000` and
  `MAX_NOTIONAL_PER_SYMBOL = 3000` apply to crypto; the stock loop sets both to 5000.
- Per-book correlation-aware risk cap: `MAX_BOOK_RISK_PCT = 0.025` (2.5% of equity summed across a
  book's stops, adjusted for how correlated they are).

Worked example: with $122,202 of equity (the paper account's cash after this morning's liquidation,
per `research/campaign_2026-09_jetson/README.md` §6), 0.5% risk is $611. If a stock's stop sits 1.2%
below the entry, the "risk-based" size would be $611 / 1.2% = $50,917. The $5,000 cap cuts that to
$5,000, which risks only $5,000 x 1.2% = $60, about 0.05% of equity. In practice the dollar caps bind
first, so the book runs a fraction of its configured risk (`docs/MAP.md` §5-10 says roughly one third
to one fifth). That is deliberate caution, and it also means the account cannot be hurt quickly while
the system is unproven.

### Every decision is written down

`trade_journal.log_decision` appends one JSON line per decision to `journals/YYYY-MM-DD.jsonl`: every
buy, every sell, and every **skip** with the reason it was skipped. The measurement tools
(`decision_report.py`, `beta_ledger.py`, `llm_eval.py`, `execution_report.py` and others) read these
journals later. The principle is that every strategy change should be an **evidence-gated flag flip**:
new behavior ships switched off (a "default-OFF flag" in `strategy_config.py`) and is switched on only
when a pre-registered measurement says so (`docs/FLAGS.md`, `research/campaign_2026-08/03_jetson_runbook.md`).

### Fail-closed and fail-open

Two phrases you will meet in every chapter:

- **Fail-closed**: when information is missing, do nothing. A missing prediction, missing quote or
  missing model means **no new entry** (`docs/MAP.md` §6, invariant 3).
- **Fail-open**: when an optional advisor breaks, carry on as if it had said "neutral." The LLM gate
  and the meta-label gate are fail-open, so a provider outage can never block the whole book.

## 4. What the evidence says so far

Honesty first, because the rest of the book depends on it.

- **The 2026-07 review verdict still stands.** The deployed policy is "a gated, conditional-beta,
  long-only book": six correlated cryptos plus the top 7 of a set of high-beta stocks. Positive
  expected value (**+EV**) is unproven; the validation reviewer's honest probability that a full gate
  pass reflects real skill was about 40 to 55%, and a live Sharpe typically comes in 50 to 73% below
  its backtest (`docs/MAP.md` §1 and §9 item 1). The parts of the system that are reliably valuable
  today are **cost engineering** (cheaper order placement) and **loss avoidance** (the gates).
- **The device was stale until last night.** The bots had not run since 2026-05-07; the April models
  were LSTM-only with no manifest; the data stores ended in February on an old schema; and an audit
  found a one-day look-ahead leak in the sentiment feature that the April models had trained on
  (`research/campaign_2026-09_jetson/README.md` §1 and §2; Chapter 3 tells that story).
- **The clean rebuild found no selection skill.** After rebuilding both data stores from scratch
  (263,889 crypto rows; 1,536,356 stock rows over 92 names), a 40-trial crypto search produced 43
  negative trials out of 44, and the best trial's holdout produced just 3 trades, too few to certify, so nothing was saved.
  A 25-trial stock search reached a holdout Sharpe of 0.13 with a Deflated Sharpe of 0.058 against a
  required 0.60, so nothing was saved either. The campaign's summary: "on clean, leak-free data the
  current LSTM + objective design shows no measurable selection skill in either book" (README §4).
  Chapter 6 explains, kindly, why that is a good outcome for the process even though it is a
  disappointing one for the strategy.
- **Operational state this morning** (README §6): on the founder's instruction the paper account was
  liquidated to cash ($122,202), a stock-only systemd user service was installed and started, and a
  70-trial stock retrain was launched with the "Phase-3 bundle" of training fixes switched on
  (`strategy_config.HYPERSEARCH_V3 = True`, `OBJECTIVE_V3 = True`, `TRAINING_REPAIRS_V1 = True`,
  `OBJECTIVE_LONG_ONLY = True`). Note for readers of the older docs: `docs/MODULES.md` §5 still says
  these are "all default OFF"; the code now has them ON, and the code wins. At the time of writing
  that retrain was a handful of trials in, still in its random-search phase, with no result yet.
  Until a model passes the gates, the stock bot runs but cannot buy: no model means no prediction,
  and no prediction means no entry (fail-closed).
- **Direction:** the founder has asked to stop spending effort on the crypto book and concentrate on
  stocks, supported by the crypto search result and by a slippage study showing real crypto crossing
  costs well above the model's assumption (Chapter 2).

The honest goal, then, is not "make money next week." It is: build an instrument that can tell the
difference between skill and luck, that pays only for edge it can measure, and that runs unattended
on an 8 GB box without hurting the account while it learns. Measured against that goal, last night
was a success: the instrument worked and correctly refused to certify a model that had no edge.

## 5. Further reading

- Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018. The single most
  influential source for this repository's labeling, sample weighting, validation and meta-labeling.
- Larry Harris, *Trading and Exchanges: Market Microstructure for Practitioners*, Oxford University
  Press, 2003. How markets, orders, spreads and brokers actually work.
- Ernest P. Chan, *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley, 2013. A
  practical, honest introduction to building and testing retail-scale trading systems.
- David H. Bailey, Jonathan M. Borwein, Marcos López de Prado and Qiji Jim Zhu, "Pseudo-Mathematics
  and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance,"
  *Notices of the American Mathematical Society* 61(5), 2014. Why impressive backtests are cheap.
- Campbell R. Harvey, Yan Liu and Heqing Zhu, "...and the Cross-Section of Expected Returns,"
  *Review of Financial Studies* 29(1), 2016. How many published "discoveries" are false positives.
