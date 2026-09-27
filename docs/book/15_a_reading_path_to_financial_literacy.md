# Chapter 15. A reading path to financial literacy for this system

## 15.1 The idea

This book has explained one system. The ideas underneath it are general, and they are the ideas
that separate people who understand markets from people who are merely exposed to them: what a
return is made of, what trading costs, how to tell skill from luck, and how data lies. This last
chapter is a map for going further. It has three parts: a crosswalk from the vocabulary of this
book to the repository's glossary, a ten-item reading curriculum with a paragraph on why each item
earns its place, and a field guide to the five traps that ruin most trading research, each shown
with an example from this very repository.

## 15.2 Why it matters financially

Financial capability is not knowing which stock will rise. Nobody reliably knows that. It is the
ability to ask the right questions of any claim, including your own: *Compared with what? After
costs? On how much independent evidence? Using only information available at the time? Out of how
many attempts?* A reader who asks those five questions by reflex will avoid most of the expensive
mistakes in investing, whether the claim comes from a fund manager, a newsletter, an AI model or a
backtest on a Jetson in a spare room.

## 15.3 How this system's vocabulary maps to the glossary

The repository's [`docs/GLOSSARY.md`](../GLOSSARY.md) defines every term as it is used *here*, and
names the module that owns it. The table points from the words used in Part Two to the glossary's
alphabetical sections, where the entries live.

| Term in this book | Glossary entry | Section | One-line meaning |
|---|---|---|---|
| book | book | [B](../GLOSSARY.md#b) | one family of positions run by one loop (crypto or stock) |
| cycle, cooldown, lockout | cooldown; lockout | [C](../GLOSSARY.md#c), [L](../GLOSSARY.md#l) | the 30 second loop and its per-symbol waiting rules |
| entry windows, EOD flatten | entry windows; EOD flatten | [E](../GLOSSARY.md#e) | when stock entries are allowed; the close-out before 16:00 ET |
| overnight sleeve | overnight sleeve | [O](../GLOSSARY.md#o) | up to two stock positions held overnight under strict rules |
| journal | journal | [J](../GLOSSARY.md#j) | the append-only JSONL diary of every decision |
| fail-open, fail-closed | fail-open / fail-closed | [F](../GLOSSARY.md#f) | what a component does when it breaks |
| champion, challenger, shadow | champion; challenger; shadow slot | [C](../GLOSSARY.md#c), [S](../GLOSSARY.md#s) | the trading model, its silent rival, and where the rival lives |
| DM-HLN | DM-HLN | [D](../GLOSSARY.md#d) | the forecast-comparison test used for promotion |
| DSR | DSR (Deflated Sharpe) / `DSR_MIN` | [D](../GLOSSARY.md#d) | probability a Sharpe beats the best of N lucky trials |
| effective n | effective-n (`n_eff`) | [E](../GLOSSARY.md#e) | independent observations hidden in correlated ones |
| purge, embargo | purge; embargo | [P](../GLOSSARY.md#p), [E](../GLOSSARY.md#e) | the gaps that stop training data leaking into validation |
| triple-barrier label | triple-barrier (TB) label | [T](../GLOSSARY.md#t) | a label computed by walking the real exit rules |
| hot reload | hot-reload | [H](../GLOSSARY.md#h) | bots swapping in a newly promoted model without restarting |
| gotcha #2 | gotcha #2 | [G](../GLOSSARY.md#g) | reset the Optuna studies after any objective or feature change |
| default-OFF, evidence gate | default-OFF; evidence gate | [D](../GLOSSARY.md#d), [E](../GLOSSARY.md#e) | how model-facing changes are shipped and switched on |
| beta ledger, AKL lagged beta | beta ledger; AKL lagged beta | [B](../GLOSSARY.md#b), [A](../GLOSSARY.md#a) | measuring market exposure, including delayed exposure |
| `b2`, echo gap | `b2`; echo gap | [B](../GLOSSARY.md#b), [E](../GLOSSARY.md#e) | whether the LLM adds anything beyond the model |
| LLM veto, veto strike, `llm_mult` | LLM veto; veto strike; `llm_mult` | [L](../GLOSSARY.md#l), [V](../GLOSSARY.md#v) | the LLM's bounded powers |
| selection mode, free-first | selection mode; free-first | [S](../GLOSSARY.md#s), [F](../GLOSSARY.md#f) | how the LLM provider is chosen |
| round-trip cost, required edge | round-trip cost; required edge | [R](../GLOSSARY.md#r) | what a trade must earn to break even |
| survivorship, PIT | survivorship; PIT (point-in-time) | [S](../GLOSSARY.md#s), [P](../GLOSSARY.md#p) | two of the traps in section 15.5 |
| MinTRL | MinTRL | [M](../GLOSSARY.md#m) | how long a track record must be to prove a Sharpe ratio |

## 15.4 A ten-item curriculum

The order runs from foundations to the specific methods this system uses. None requires advanced
mathematics beyond what the chapters of this book have already shown; several are written for
practitioners. Read them in order if you can; each one makes the next easier.

**1. Zvi Bodie, Alex Kane and Alan J. Marcus, *Investments* (McGraw-Hill, any recent edition).**
The standard university textbook. Read the chapters on risk and return, the capital asset pricing
model, index models and market efficiency. It gives you the language of this book (return,
volatility, beta, alpha, diversification) with worked problems, and it teaches the discipline of
asking what a return is *compensation for*. Everything else on this list assumes it.

**2. Larry Harris, *Trading and Exchanges: Market Microstructure for Practitioners* (Oxford
University Press, 2003).** How orders actually become trades: limit and market orders, spreads,
dealers, informed and uninformed traders, and why the price you see is not the price you get. After
this book the cost gate, the maker ladder, stale quotes and the crypto spread finding in Chapter 14
will make intuitive sense.

**3. Antti Ilmanen, *Expected Returns: An Investor's Guide to Harvesting Market Rewards* (Wiley,
2011).** A survey of where long-run returns come from across asset classes and strategies: equity
premium, value, carry, momentum, volatility selling, illiquidity. It teaches humility about
forecasting and respect for simple, well-documented premia, which is the right frame for judging
whether an hourly LSTM should be expected to add anything.

**4. Richard C. Grinold and Ronald N. Kahn, *Active Portfolio Management*, 2nd ed. (McGraw-Hill,
2000).** The quantitative manager's handbook. Its fundamental law (Chapter 14's worked example)
explains why small forecasting skill applied widely beats large skill applied rarely, and why
costs and breadth dominate the arithmetic. Dense in places; the early chapters and the chapter on
the fundamental law are the essentials.

**5. Andrew W. Lo, "The Statistics of Sharpe Ratios," *Financial Analysts Journal* 58(4), 2002.**
A short, readable paper that shows the Sharpe ratio is an *estimate* with a standard error, that
annualizing it by a square-root rule is often wrong, and that serial correlation changes
everything. It is the natural bridge from Chapter 11 to the more technical papers below.

**6. David Aronson, *Evidence-Based Technical Analysis* (Wiley, 2006).** Written for traders, not
academics, it explains the scientific method applied to trading rules and devotes a chapter to
data-mining bias, "the fool's gold of objective technical analysis". It is the gentlest complete
introduction to why testing many rules and keeping the best one guarantees a disappointing future.

**7. David H. Bailey and Marcos López de Prado, "The Deflated Sharpe Ratio," *Journal of
Portfolio Management* 40(5), 2014.** The source of this system's promotion gate. Once you have read
Lo and Aronson, this paper's single idea (compare your Sharpe with the best that luck would produce
from as many trials as you ran) will feel inevitable. Pair it with the worked table in Chapter 10.

**8. Marcos López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018).** The
practitioner's reference for most of this repository's methods: triple-barrier labels,
meta-labeling, sample uniqueness and effective sample size, purged and embargoed cross-validation,
backtest overfitting. Read it as a manual of hazards rather than a recipe book; its value is in
the failure modes it names.

**9. Campbell R. Harvey, Yan Liu and Heqing Zhu, "... and the Cross-Section of Expected Returns,"
*Review of Financial Studies* 29(1), 2016.** Hundreds of published return predictors, examined as a
multiple-testing problem. The conclusion (a new factor should clear a t-statistic of about 3, not 2)
is the academic version of this system's DSR gate. Read with McLean and Pontiff (2016), who measured
how much published anomalies shrink after publication.

**10. Ernest P. Chan, *Quantitative Trading: How to Build Your Own Algorithmic Trading Business*,
2nd ed. (Wiley, 2021).** A small operator's view: data, backtesting pitfalls, execution, risk,
capacity and the practical business of running a strategy. It is the closest book on this list to
the founder's actual situation, a one-person system with real infrastructure constraints, and it
is candid about how often strategies fail.

A bonus for the habit of mind rather than the mathematics: Daniel Kahneman, *Thinking, Fast and
Slow* (Farrar, Straus and Giroux, 2011), on why people see patterns in noise and trust small
samples. Every trap below is a cognitive bias before it is a statistical one.

## 15.5 The five traps, with examples from this repository

### Trap 1: Overfitting

*What it is.* Fitting the noise in past data so well that the model fails on new data. Its common
form in trading is selection: try many configurations, keep the best.

*Where it appeared here.* The overnight crypto search ran 44 configurations. One scored slightly
positive. Without the DSR gate that one would have looked like "the model that works".

*The defence.* Walk-forward folds with purging and embargo, a holdout never seen by the search, the
DSR gate with the true number of trials, and shadow comparison on live data.

*The question to ask.* "Out of how many attempts?"

### Trap 2: Look-ahead bias

*What it is.* Using information in a test that was not available at the moment of the decision.

*Where it appeared here.* The `Daily_Sentiment` feature was keyed by host-local date, so every row
held the next day's Fear and Greed value. Every training store built before 2026-09-26 carried a
one-day look-ahead, and the April models were trained on it. It was found by an audit and repaired
in the clean rebuild. LLMs add a subtler version: a model scoring a 2023 headline may already know
what happened next.

*The defence.* Strictly trailing features computed by the same functions at harvest and live,
publication-date lags for sentiment and short interest, and judging LLM opinions only on live,
post-cutoff decisions.

*The question to ask.* "Could I have known this at the time?"

### Trap 3: Survivorship bias

*What it is.* Testing on the names that exist today, which silently excludes the ones that failed.
Brown, Goetzmann, Ibbotson and Ross (1992) showed it can even create the appearance of
predictability where there is none.

*Where it is prevented here.* The stock harvest keeps a name on a given date only if it was among
the top 60 by 30-day dollar volume *on that date* (`_asof_membership_mask`), not today. The
repository's invariants call this mask the only place survivorship is prevented and forbid
"simplifying" it.

*A worked example.* Take 100 stocks. Over a year, 30 fall 50 % and 70 rise 21.4 %. The average
stock returned 0.7 x 21.4 % + 0.3 x (-50 %) = 0 %. Now suppose the fallers dropped out of the
top-volume list and you build your test universe from today's list. Your universe holds mostly
risers, and "buy everything" shows about +21 %, from a strategy with no skill at all.

*The question to ask.* "Which names were in the universe *then*?"

### Trap 4: Cost blindness

*What it is.* Ignoring or understating the cost of trading, which punishes high-turnover
strategies most (Novy-Marx and Velikov 2016).

*Where it appeared here.* The crypto cost model assumes a 10 basis point spread; reconstructed
paper fills showed 26 to 34 basis points round trip on three of the six coins (Chapter 14).
Separately, the crypto search kept choosing thresholds below the live admission floor, so its
simulated trades were ones the live cost gate would never have taken.

*The defence.* One cost function (`fees.round_trip_cost_pct`) shared by the backtester, the
meta-labeler and the live gate; realised-cost reports (`execution_report.py`,
`scripts/fill_venue_slippage_report.py`); and the rule that a new gate re-deriving cost is a bug.

*The question to ask.* "After all costs, including the spread I actually paid?"

### Trap 5: Small samples

*What it is.* Drawing conclusions from too little independent evidence, often disguised by many
correlated rows.

*Where it appeared here.* The stock holdout had 20 trades, 14 effectively independent (Chapter 10).
The LLM scorecard's original test rejected a true null two to three times too often at 20 to 60
effective observations (Chapter 11). An alpha estimate needed about two years of data against a
window under half a year.

*The defence.* Effective-n everywhere, pre-registered minimum counts before any verdict,
"cannot conclude" as an honest output, and anytime-valid methods when you must look repeatedly.

*The question to ask.* "How many *independent* observations is this?"

## 15.6 What the evidence says about learning this way

There is no controlled trial showing that reading these ten items makes anyone a better investor.
What the evidence does show is how often the five traps above explain published and professional
failures: McLean and Pontiff's post-publication decay, Harvey, Liu and Zhu's multiple-testing
arithmetic, Sullivan, Timmermann and White's finding that data-snooping explains much of the
apparent success of technical rules, and this repository's own overnight result. Learning to
recognise those traps is the most reliable financial skill there is, because it protects capital
regardless of what the market does next.

## 15.7 Further reading

The curriculum above is the further reading. Two companions for the traps:

- Stephen J. Brown, William Goetzmann, Roger G. Ibbotson and Stephen A. Ross, "Survivorship Bias
  in Performance Studies," *Review of Financial Studies* 5(4), 1992, 553 to 580.
- R. David McLean and Jeffrey Pontiff, "Does Academic Research Destroy Stock Return
  Predictability?" *Journal of Finance* 71(1), 2016, 5 to 32.

And inside the repository, in reading order for a human: [`docs/README.md`](../README.md), then
[`docs/MAP.md`](../MAP.md) sections 1 and 2, then [`docs/GLOSSARY.md`](../GLOSSARY.md) as needed,
and [`research/KILL_LIST.md`](../../research/KILL_LIST.md) for the ideas that were tried and
retired, with the reasons.
