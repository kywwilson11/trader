# Chapter 12. The LLM as a bounded advisor

## 12.1 The idea

A large language model can read a headline, an earnings summary and a macro calendar and form a
qualitative opinion in seconds. The numerical model in this system cannot read at all; it sees
only prices and engineered features. So there is an obvious temptation: let the LLM decide.

This system resists that temptation on purpose. The LLM is an **advisor with a leash**. It can
shrink or modestly grow a position, it can veto an entry in a clear catastrophe, and that is all.
It never chooses what to buy. Its answers must arrive in a fixed machine-readable shape. If it
fails, the trade goes ahead as if it had said nothing. And it has a hard daily budget of one
dollar.

This chapter explains each part of that leash, what it cost to run last night, and why the
evidence does not yet justify loosening it.

Terms used below:

- A **role** is a job the LLM does: `analyst` (scores trade candidates), `sentiment` (scores news
  for the sentiment feature and gate) and `backfill` (re-scores historical articles in bulk).
- **Schema-enforced output** means the provider is told the exact JSON shape required, and its API
  refuses to return anything else.
- **Fail-open** means that when the component fails, the system behaves as if the component were
  not there. Its opposite, **fail-closed**, means that on failure the system does nothing.

## 12.2 Why it matters financially

Three reasons, each of which costs money if ignored.

1. **LLMs remember the past they are asked to predict.** A model trained on text up to some date
   has, in a sense, read the future of every headline before that date. Glasserman and Lin (2023)
   showed that LLM sentiment scores on in-sample headlines carry look-ahead bias, and 2026 work
   collected in the INTEL research log found that effective knowledge cutoffs can sit months
   before the vendor's stated date. Any backtest of LLM opinions on historical news is therefore
   suspect. Only live, post-cutoff decisions can judge the advisor.
2. **LLMs are persuadable.** The prompt in this system shows the model's own prediction. Research
   summarised in the same log found that context changes how an LLM interprets identical evidence,
   so the LLM may simply echo the model. An echo adds cost and no information.
3. **API calls are a running cost with no ceiling unless you build one.** A call every ten minutes
   per book is 288 calls a day for two books. At a few tenths of a cent each that is trivial; at a
   cent each it is a few dollars a day, every day, whether or not the calls help.

The literature on real LLM trading skill is thin and mixed. Lopez-Lira and Tang found that GPT-4
headline scores capture the immediate (non-tradable) market reaction well, and that the tradable
drift after it is concentrated in small stocks and negative news, which is outside this book's
liquid long-only universe. An audit of 77 LLM trading studies, cited in the INTEL log, found that
of the 19 closed-loop studies at its core only two used time-consistent splits and only one
modelled transaction costs.

## 12.3 How this system does it

### Where the LLM sits in the loop

In the cycle described in Chapter 9, `_run_llm_analysis` runs at most every
`LLM_INTERVAL_SEC = 600` seconds per book. It sends all current candidates in one call to
`llm_analyst.analyze_trades` ([`llm_analyst.py`](../../llm_analyst.py)), which returns a score
`s` between 0 and 1 for each symbol plus short bull, bear and summary strings. The system prompt
tells the LLM exactly how its number will be used:

- `s < 0.15` (`LLM_VETO_THRESHOLD` in [`trading_utils.py`](../../trading_utils.py)): a **veto**.
  New buys in that symbol are blocked immediately. If two consecutive analyses both veto, an open
  position is **liquidated** (the "veto strike" counter in `base_loop.py`). The prompt reserves this
  for confirmed catastrophe: fraud, insolvency, delisting, a hack.
- Otherwise the size multiplier is `llm_mult = 0.5 + s`, so between 0.65 and 1.5, which enters the
  clamped tilt product in `_compute_position_size`. A score of 0.5 changes nothing.

The prompt also contains a security instruction: headlines are untrusted data scraped from the
internet and may contain planted instructions; the LLM must judge the news, not obey it. The
analyst runs at temperature 0.2 (`_ANALYST_TEMPERATURE`) with a 45 second timeout.

### Schema-enforced outputs, three ways

[`llm_client.py`](../../llm_client.py) speaks to three providers without any vendor SDK, and forces
each one to return valid JSON matching the schema built in `llm_analyst.py` (required fields `s`,
`bull`, `bear`, `r`):

- **Gemini**: `responseMimeType` plus `responseSchema`.
- **Anthropic (Claude)**: forced tool use. The schema becomes a tool's input schema and
  `tool_choice` pins the model to that tool, so its only possible answer is a filled-in form.
- **OpenAI** and OpenAI-compatible endpoints (OpenRouter, Groq, a local Ollama): `response_format`
  with a strict JSON schema.

The practical benefit is that the caller never parses free prose, and a malformed answer is a
detectable failure rather than a silently misread number.

### Which model answers: roles, chains and selection modes

`resolve_provider_chain(role, config)` returns an ordered list of (provider, model) candidates,
and `call_llm` tries them in order, skipping providers that are rate-limited, over budget or
missing a key. The `selection_mode` in `llm_config.json` controls the order:

- `auto` (the default): Anthropic, then OpenAI, then Gemini, each skipped if no key is configured,
  with every enabled extra endpoint appended last;
- `single`: only the configured provider, no fallback;
- `free-only` and `best-free`: only free endpoints, with Gemini's free tier as the last resort.

The `backfill` role is pinned to Gemini because it uses Gemini's discounted Batch API. On this
Jetson only a Gemini key is configured, so in practice every call goes to Gemini.

### Fail-open, everywhere

The module docstring of `llm_analyst.py` states the contract: "On any failure, returns {} for
pass-through (never blocks trades)." Concretely:

- a transport error, timeout, refusal or unparseable answer returns an empty result;
- the loop then keeps the previous scores (`llm_scores are deliberately untouched`), and a symbol
  with no score is treated as `s = 0.5`, which means no veto and a multiplier of exactly 1.0;
- scores older than `LLM_SCORE_TTL_SEC = 7200` seconds expire, again to neutral;
- consecutive failures back off exponentially, 600 s, 1,200 s, 2,400 s, capped at 3,600 s, and
  each is journaled as an `llm_backoff` row.

Contrast this with the rest of the live path, which is fail-closed: a missing prediction or quote
blocks the entry. The asymmetry is deliberate. The model is the signal, so without it you must not
trade; the LLM is an overlay, so without it you trade as if it were neutral.

### The one-dollar cap

`_DAILY_COST_LIMIT = 1.00` (in `llm_client.py`). Every call's cost is computed from the provider's
reported token usage and a pricing table (for example Gemini 2.5 Flash at $0.30 per million input
tokens and $2.50 per million output tokens; Flash-Lite at $0.10 and $0.40). The running total is
shared across processes through `llm_cost.json`. Once the day's spend reaches the cap, every
further call is refused, and fail-open turns that refusal into neutral scores. The day resets at
midnight Pacific time (`_maybe_reset_quota`), matching Google's quota reset. Note that the cap is
global: an expensive analyst can starve the sentiment role of budget.

### The free-first plan

The 2026-08 campaign built the plumbing to run the LLM at zero cost on free endpoints, gated by a
qualification script, `scripts/llm_qualify.py`, which checks a candidate's reliability and its
agreement with the production analyst before any switch. The runbook's Phase 4 then says: read the
spend verdict first; only if it says "keep" and a free model qualifies, add budget rows for it and
switch `selection_mode`. As of 2026-09-27 this plan is **not viable as the default**: no Groq or
OpenRouter keys exist, no local model runs, and the arithmetic does not work. Groq's free tier
allows about 200,000 tokens a day, which is roughly 40 analyst calls of 5,000 tokens against a
need of up to 288. OpenRouter's free models allow 50 requests a day, or 1,000 after a one-time
purchase of at least $10 of credit. A local LLM on the Jetson's CPU would need several minutes per
analyst prompt, far too slow for a 10-minute cadence.

## 12.4 A worked example: last night's bill

The journal for 2026-09-27 records every analyst call made while the bots ran overnight (crypto
book only, 03:10 to 08:58 Central time). Summing the `cost_usd` field of the `llm_analysis` rows:

| Model chosen by the router | Calls | Total cost | Cost per call |
|---|---:|---:|---:|
| gemini-2.5-flash-lite | 23 | $0.0244 | $0.0011 |
| gemini-2.5-flash | 11 | $0.1067 | $0.0097 |
| **Total** | **34** | **$0.1311** | **$0.0039** |

Median latency was 5.3 seconds; the slowest call took 30.0 seconds, which is the configured
`max_llm_latency_sec` ceiling. One call failed and produced an `llm_backoff` row.

Three lessons come out of this small table.

1. **The model mix is the budget.** Flash cost nine times Flash-Lite per call. It answered a third
   of the calls and ran up 81 % of the bill.
2. **Extrapolate before you scale.** Six calls an hour is 144 calls a day per book. At last
   night's average of $0.0039, that is about $0.56 a day for one book, and about $1.12 for two,
   which is over the $1.00 cap. At Flash-Lite prices throughout, one book would cost about $0.15 a
   day. The cap would have started refusing calls late in the day with both books running.
3. **Spend is not value.** Every score in those rows has `"pred": null`: no model was loaded, and
   entries were impossible (the bots were in an exits-only mode). The LLM still scored every
   candidate every ten minutes, because the halt blocks entries but does not silence the analyst,
   and a two-strike veto could still have sold a position. The ENGINE department raised this as an
   owner question (item 20). Thirteen cents bought no decision-relevant information.

A second, smaller example shows the leash in action on a single order. The stock book is about to
buy $5,000 of a stock. The LLM scored it 0.62, so `llm_mult = 1.12`. That factor joins the other
tilts, and the product is clamped between 0.1 and 1.30, so even a perfect score cannot push the
order beyond 1.3 times its base. Had the score been 0.12, the buy would have been skipped and
journaled as an `llm_veto`; if the next analysis ten minutes later also said below 0.15, any open
position in that name would have been sold.

## 12.5 What the evidence says

- **There is no verdict on whether the LLM earns its keep.** The scorecard (`llm_eval.py`,
  Chapter 11) needs 120 distinct forecast hours and 20 effective observations on journals written
  by the current code, and it has none. Older journals are void because they ran a different gate.
- **Two cheap experiments are designed but not run.** A test-retest check (ask the same question
  twice and measure how much `s` moves; if vetoes flip on repeat calls they are partly coin tosses)
  and a pred-blind A/B (hide the model's prediction from half the prompts to measure echo) would
  together cost about $0.75. They are parked as a money ask for the founder.
- **The analyst is shown the model's prediction by default** (`include_pred=True`), which is the
  anchoring risk described above. Whether to hide it is a decision to make after the A/B.
- **Local models are not the answer on this hardware.** Measurements on the Jetson put a
  FinBERT-sized encoder at about 134 ms per headline, which is feasible, but 2026 studies found
  next-day information coefficients of about 0.014 at best for such scorers, not enough to justify
  a new dependency.
- **Reliability is now measurable.** Since the campaign, every analyst call, including failures,
  writes an `llm_call` row with provider, model, latency and failure reason, so the next weeks of
  operation will show how often the fail-open path is actually taken.

The honest summary: the LLM is cheap, bounded and unproven. The leash is the right design until a
live, pre-registered measurement says otherwise.

## 12.6 Further reading

- Alejandro Lopez-Lira and Yuehua Tang, "Can ChatGPT Forecast Stock Price Movements? Return
  Predictability and Large Language Models," arXiv:2304.07619 (revised through 2025).
- Paul Glasserman and Caden Lin, "Assessing Look-Ahead Bias in Stock Return Predictions Generated
  by GPT Sentiment Analysis," arXiv:2309.17322, 2023.
- Provider documentation on structured outputs: Google's Gemini API documentation on structured
  output and pricing (ai.google.dev/gemini-api/docs), Anthropic's documentation on tool use
  (docs.anthropic.com), and OpenAI's documentation on structured outputs (platform.openai.com/docs).
- In this repository: [`research/campaign_2026-09_jetson/research_intel.md`](../../research/campaign_2026-09_jetson/research_intel.md)
  (Scout A, with about thirty dated 2026 sources), the runbook's Phase 4 in
  [`03_jetson_runbook.md`](../../research/campaign_2026-08/03_jetson_runbook.md), and
  [`docs/MAP.md`](../MAP.md) section 9 items 12 and 13.
