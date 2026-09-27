# The Trader Book

A book about this trading system, written for its founder: a smart, motivated reader who is not a
quant and wants to become more financially capable by understanding what the system does, why,
and what the evidence says about it. Chapters are plain Markdown; `build_pdf.sh` assembles all of
them, in filename order, into one PDF (`Trader_Book.pdf`).

<!-- PART ONE BLOCK START (owned by the Part One agent; edit only between these markers) -->
## Part One: the system and the finance behind it

Every chapter follows the same five sections: the idea in plain words; why it matters financially
(with numbers); how this system does it (file.py:function, constants, flags); what the evidence says
so far; further reading.

1. [What this system is, and what it is not](01_what_this_system_is.md): paper trading, long-only,
   the two books, the Jetson, and the honest goal.
2. [Markets, bars and costs](02_markets_bars_costs.md): hourly bars, spreads, the EDGE estimator,
   fees, why a crypto trade needs a 0.6% break-even and a 1.2% admission floor, and last night's
   686-fill slippage study.
3. [Features: turning prices into numbers](03_features.md): the stationary preset, technical
   indicators, time encodings, funding and open interest, panel ranks, point-in-time discipline and
   the sentiment leak story.
4. [Labels: what "the model predicts" means](04_labels.md): forward returns, triple-barrier labels,
   the shared exit kernel, and why labels must equal the backtest.
5. [The forecaster](05_forecaster.md): the LSTM with attention, the LightGBM leg, the blend weight,
   the q10 tail veto, and why predictions are tiny numbers.
6. [Honest validation](06_honest_validation.md): leakage, purged walk-forward folds, the embargo,
   the holdout, the Deflated Sharpe ratio, effective sample size, PBO, the ratchet, and last night's
   "no selection skill" verdict.
7. [The gate stack and meta-labeling](07_gate_stack.md): the cost gate, thresholds, the meta-label
   and LLM vetoes, VIX and macro stand-downs, fail-open versus fail-closed.
8. [Sizing, exits and risk](08_sizing_exits_risk.md): Kelly and its cap, volatility targeting, the
   tilt product, the drawdown ladder, ATR stops and trailing, the circuit breaker and the book risk
   cap.
<!-- PART ONE BLOCK END -->

<!-- PART TWO BLOCK START (owned by the Part Two agent; edit only between these markers) -->
## Part Two: running it, measuring it, and learning from it

Same five-part shape as Part One (idea, why it matters financially, how this system does it,
what the evidence says, further reading), plus one worked example per chapter built from real
numbers on the production Jetson.

9. [The life of a trading day](09_life_of_a_trading_day.md): the 30-second cycle and why exits run
   first, market hours and the Alpaca clock, entry windows, macro stand-downs, the per-symbol
   funnel, the end-of-day flatten and the overnight sleeve, and what the journals record.
10. [From data to a deployed model](10_from_data_to_a_deployed_model.md): harvest, train, certify,
    the policy gate, champion and challenger, the shadow DM-HLN test, promotion and hot reload, the
    weekly cold restart, and why flags default OFF; a DSR table showing why 20 trades certify nothing.
11. [Measuring yourself honestly](11_measuring_yourself_honestly.md): the decision report, the
    realised-beta ledger and alpha versus beta, the LLM scorecard and its statistics, effective
    sample sizes, the anytime-valid ledger, and what "void" verdicts mean.
12. [The LLM as a bounded advisor](12_the_llm_as_a_bounded_advisor.md): roles, schema-enforced
    outputs, fail-open, the one-dollar cap, the free-first plan, and the bill from the night of
    2026-09-27.
13. [Running on an 8 GB Jetson](13_running_on_an_8gb_jetson.md): the memory budget, GPU versus CPU
    processes, the systemd user unit and its watchdog, OOM policy, the crash loop, logs and journals,
    and alerts.
14. [What the evidence says today](14_what_the_evidence_says_today.md): no selection skill on clean
    data, the diagnosis and the levers, the Phase-3 retrain, how to read the next certificate, and
    the crypto decision.
15. [A reading path to financial literacy](15_a_reading_path_to_financial_literacy.md): a glossary
    crosswalk, a ten-item curriculum, and the five traps (overfitting, look-ahead, survivorship,
    cost blindness, small samples) with examples from this repository.
<!-- PART TWO BLOCK END -->

## Building the PDF

    bash docs/book/build_pdf.sh

The script converts every `NN_*.md` chapter in this directory, ordered by filename, into one HTML
file with a table of contents (`md2html.py`, standard library only), then renders it with
LibreOffice (`soffice --headless --convert-to pdf`). Running it twice gives the same result.
