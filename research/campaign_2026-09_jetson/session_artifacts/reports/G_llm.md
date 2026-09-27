# G — LLM layer on the Jetson: does it work today, and what would it cost? (2026-09-26)

## Verdict

| Provider | Status today | Evidence |
|---|---|---|
| **Gemini** | **WORKS** | One live, schema-enforced call through `llm_client.call_gemini`: HTTP 200, 660 ms, `modelVersion=gemini-2.5-flash-lite`, `serviceTier=standard`, finish STOP, contract satisfied. The ledger recorded **$0.000021** (usage math: $0.0000206). The May 429 storm is **gone**. |
| **Anthropic** | **NOT CONFIGURED** (no key in `llm_config.json`, no `ANTHROPIC_API_KEY` in env). No call made. | The configured id `claude-opus-4-6` is still a valid model but is missing from every table. The transport would break on several current models (D4, D5). |
| **OpenAI** | **NOT CONFIGURED** (no key, no `OPENAI_API_KEY`). No call made. | The configured id `gpt-4.1` is still valid (not deprecated) but is missing from every table. |

| Role | Effective model | Status |
|---|---|---|
| analyst (entry veto / size tilt) | `gemini-2.5-flash-lite` (routing table). Fallback `call_llm` chain: flash → flash-lite → pro | WORKS |
| advisor-v2 | Same transport and model as analyst (extended schema). `advisor_v2_enabled=False` | Inert (off) |
| sentiment (tiered scoring) | `[('gemini-2.5-flash-lite', 1.0)]` | WORKS (the runner's 2 live calls succeeded) |
| backfill (Batch API) | `gemini-2.5-flash-lite` via `get_recommended_model('backfill')` | Not exercised. **Accounting degraded:** Batch spend never reaches the $1 cap ledger (D11) |

**Overall: WORKS on Gemini only, with degraded accounting and several latent traps.** If the owner simply pastes an Anthropic or OpenAI key into today's config, it routes the analyst to an untabled model: 50 RPD per process, and the cap bills at a wrong price.

## Spend this task caused (exact, from `llm_cost.json`)
- The ledger was `{"date":"2026-05-07","cost":0.034652}`. The first offline `get_recommended_model` call rolled it to `{"2026-09-26", 0.0}`. This is the normal midnight-PT reset in `_maybe_reset_quota`, llm_client.py:708-725. The original is backed up to `scratchpad/G/llm_cost.json.orig`.
- Gemini smoke call: +$0.000021.
- `tests/test_sentiment_headlines.py`: +$0.000040. The runner made **2** Gemini `generateContent` calls through `llm_client` (`_score_articles` validation cases), not the "one fail-soft HTTP call" that CLAUDE.md claims (doc drift, D16).
- **Final ledger: `{"date": "2026-09-26", "cost": 6.1e-05}` = $0.000061 total.** No Anthropic or OpenAI calls.
- `llm_cost.json.lock` was created by the flock. It is gitignored (`*.json.lock`).

## 1. Config migration (subprocess, `scratchpad/G/effective_config.py` → `.out`)
- **The loader did NOT rewrite `llm_config.json`.** sha256 `50a9ed81…` is identical before and after load and at the end; `cmp` against the scratchpad backup is identical.
  - `load_llm_config` saves only when an `_MIGRATE_KEYS` legacy key (`analyst_model`/`sentiment_model`) is present (llm_config.py:300-326). The on-disk file already uses `*_override`, so `migrated=False`.
  - The other writer, `_capture_rate_limit_headers` (llm_client.py:329-358), never fired: Gemini returned **no `x-ratelimit-*` headers** (D9).
- On-disk keys: provider, enabled, models{gemini,claude,openai}, fmp_api_key, max_llm_latency_sec, journal_enabled, analyst/sentiment_model_override (null), detected_tier (null), tier_override (null).
- Filled **in memory only** from `_DEFAULTS`: selection_mode=`auto`, provider_preference=`[anthropic, openai, gemini]`, endpoints=`[]` (**no free presets enabled**), pricing=`{}`, pricing_cache_multipliers, anthropic_cache_system_ttl=`""`, rich_context_enabled=False, replay_capture_enabled=True, advisor_v2_enabled=False, analyst_dedup_ttl_sec=0.
- The legacy `"provider": "gemini"` is **ignored** under `auto`. Only `single` mode reads it.
- Usable keys: gemini only (39-char key; FMP key also set). Tier resolves to `paid` (default, never detected), which means 30 RPM and `_PAID_TIER_BUDGETS`.
- Chains:

  | Role | `resolve_provider_chain` | `get_recommended_model` |
  |---|---|---|
  | analyst | gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro | gemini-2.5-flash-lite |
  | sentiment | gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro | gemini-2.5-flash-lite |
  | advisor | gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro | gemini-2.5-flash-lite |
  | backfill | `[gemini-2.5-flash]` | gemini-2.5-flash-lite (**disagree**, D11) |

- **The configured `models.gemini.model` ("gemini-2.5-flash") is not the analyst's model.** `get_recommended_model` falls through to the routing table, where every bracket is flash-lite (llm_client.py:596-605, 262-271). The configured id only heads the `call_llm` fallback chain (D12).

## 2. Budget and pricing tables vs configured models

Paid tier is the effective tier.

| Model | Budget RPD | Price in table | Real list price (2026-09) | Valid id? |
|---|---|---|---|---|
| gemini-2.5-flash | 2000 | 0.30/2.50 | 0.30/2.50 ✓ | yes, but "2.5 access limited to users who have actively used them in the past" (Google models page); no shutdown date |
| gemini-2.5-flash-lite | 5000 | 0.10/0.40 | 0.10/0.40 ✓ | same |
| gemini-2.5-pro | 1000 | 1.25/10 | 1.25/10 ✓ | same |
| **claude-opus-4-6** (configured) | **50 (default)** | **missing → 1.25/10** | **5/25** | valid (Active) |
| **gpt-4.1** (configured) | **50 (default)** | **missing → 1.25/10** | **2/8** | valid (only the gpt-4.1-nano snapshot retires 2026-10-23) |
| claude-haiku-4-5 | 5000 | 1/5 | 1/5 ✓ | valid (snapshot `-20251001` is **not** tabled → 50 RPD, 1.25/10) |
| claude-sonnet-5 | 2000 | **3/15** | **2/10** | valid; over-recorded 1.5× (D1) |
| claude-opus-4-8 | (price only, no budget → 50) | 5/25 | 5/25 ✓ | valid |
| claude-opus-5-5 / claude-fable-5-1 / claude-opus-5 | **missing** | fallback 1.25/10 | 4/20, 10/50, 5/25 | valid. **opus-5-5 and fable-5-1 reject forced tool use (400)** (D5) |
| gpt-5.4 / -mini / -nano | 1000/2000/5000 | 5/15, 1/4, 0.25/1 | **2.50/15, 0.75/4.50, 0.20/1.25** | valid; placeholders wrong (D2) |
| gemini-3.x (3.5-flash-lite, 3.1-flash-lite, 3.8-flash…) | **missing** | fallback 1.25/10 | 3.5/3.1-flash-lite are **free of charge** on the free tier; 3.8-flash 0.75/3.75 | current Google recommendation for new work |

**What the current config allows per day:**
- As configured (Gemini only), the analyst on flash-lite has 5000 RPD **per process**. The counters in `_model_calls` are in-memory and per process, reset on restart (D7), and shared with sentiment and global_context calls in that process.
- Cadence is `LLM_INTERVAL_SEC=600`, so at most 144 calls/day per book, 288 combined. The budget never binds; the $1 cap is the real governor.
- **Historical spend (May-era code, from the `[LLM] … ($X today)` lines, combined across both bot logs):**

  | Month | Median $/day | Max $/day | Successful calls | 429 lines | $/call |
  |---|---|---|---|---|---|
  | Feb | 0.126 | 0.184 | 991 | — | ≈0.00056 |
  | Mar | 0.094 | 0.193 | 5542 | 174 | ≈0.00041 |
  | Apr | 0.080 | 0.399 | 7063 | **105,480** | ≈0.00038 |
  | May (7 days) | 0.042 | 0.180 | 2163 | 29,504 | ≈0.00018 |

  The cap was **never hit** (0 "Daily cost limit" lines). Total ≈$6/month. Per-day detail is in `scratchpad/G/log_daily.txt`.
- **If the owner adds an Anthropic key without changing the model id:**
  - `auto` puts Anthropic first, and the analyst goes to `claude-opus-4-6` with 50 RPD.
  - In combined-bots mode (12 calls/h) the budget is gone in about 4 h. Split mode gets 50 per process, about 8 h.
  - After that, `call_model` returns None and `call_llm` skips to claude-haiku-4-5, so the gate is **degraded, not dead**.
  - Meanwhile the ledger bills Opus at $1.25/$10 against a real $5/$25. The **$1 cap then permits roughly $2.5–4 of real spend**.
- **Rough paid-analyst estimate (not measured):** the prompt is about 5k input tokens; output is about 150–400 tokens per symbol (`max_tok=max(4096, n*400)`).
  - claude-haiku-4-5: about $0.01–0.025 per call, so 288 calls/day ≈ **$3–7/day**. The global $1 cap trips after ~40–100 calls.
  - The cap is **global across roles**, so once it trips, sentiment scoring and global_context stop for the rest of the day too.

## 3. Live smoke (one call per configured provider)
- **Gemini** (`scratchpad/G/smoke_gemini.py` → `.out`):
  - The call went through `llm_client.call_gemini` (the analyst's own transport) with an analyst-shaped Gemini `responseSchema` and temperature 0.
  - `urlopen` was wrapped to allow **exactly one** request, and `_429_MAX_WAIT_PRIMARY=0` so it could not retry. The guard recorded 1 invocation.
  - Result: HTTP 200, 660 ms, model echoed `gemini-2.5-flash-lite`, usage 14 in / 48 out, response `{"TEST":{"s":0.0,"bull":"No","bear":"News","r":"N/A"}}`. The schema keys and range were enforced.
  - Ledger delta **$0.000021**, matching the usage math. `record_call` incremented. No tier detected (no headers). Config untouched.
  - **Not quota-exhausted today.**
- **Anthropic and OpenAI:** no key, so no call. The forced-tool-use and strict contracts are code-read only (llm_client.py:1172-1178, 1271-1279).
- **429 caching:**
  - `_429_cooldown_until` is per provider, 30 s (llm_client.py:121-122), in-memory only (not shared across processes, lost on restart).
  - It is set **only** by `call_llm` (:1025). `call_gemini`/`call_claude`/`call_openai` never set it (D8).
  - A daily-quota body (`limit: 0`) gets the same 30 s cooldown as a per-minute burst.
  - The bot-level protection is `base_loop`'s exponential backoff: 600 → 3600 s (base_loop.py:1742-1751).
- **May 429 storm, explained:** the code running then was an older, uncommitted `llm_client` using gemini-3.1-pro-preview, 3-flash-preview and 3.1-flash-lite-preview. 429s started on 2026-04-07 (3k–9k lines/day per book) while some calls still succeeded. That pattern fits a rate- or spend-limit `429 RESOURCE_EXHAUSTED`, not prepay exhaustion (which is HTTP 402 per Google's billing docs). Current HEAD has none of those model ids.

## 4. Fail-open chain (read, plus the offline probe `scratchpad/G/failopen_probe.py` → `.out`)
- `llm_analyst.analyze_trades`:
  - Transport exceptions are caught for `call_model` (llm_analyst.py:575-583) and for the `call_llm` fallback (:584-591). `if not response: return {}` at :592-593.
  - `_parse_response` returns `{}` on bad JSON (:1079-1085), and a missing `s` becomes 0.5 (:1094-1098).
- `base_loop._run_llm_analysis`:
  - On `{}`, scores are **untouched**, strikes are **not advanced**, and failures back off 600 s then ×2 up to 3600 s (base_loop.py:1737-1758).
  - Consumers read `llm_info.get('s', 0.5)` (base_loop.py:2002, 3121). `llm_mult = 0.5 + s` gives 1.0 at 0.5.
  - Stale scores expire after `LLM_SCORE_TTL_SEC=7200`, which clears scores and strikes (:1646-1658).
- Probe results (zero network):
  - (a) with `urlopen` raising, `analyze_trades` returns `{}`. Note that **one failed cycle issues 4 requests**: call_model flash-lite, then the chain flash, flash-lite, pro.
  - (b) with `call_model` and `call_llm` both raising, it returns `{}`.
  - (c) scores stay unchanged, strikes stay `{AAA:1}`, backoff is 600 s, and an `llm_backoff` row is journaled.
  - (c2) after a second failure, strikes are still `{AAA:1}` and backoff is 1200 s.
  - (d) with a held AAA at prior s=0.10 and strike 1, **0 sells attempted** after two provider failures.
  - (e) after the TTL, scores and strikes are empty and the gate reads s=0.50.
- **Proven:** a provider error never creates a block and cannot advance the 2-strike liquidation.
- **Nuance:** "fail-open" means *prior* score. An existing veto keeps blocking **new entries** for that symbol for up to 2 h during an outage (by design).
- **Edge defect D13 (base_loop.py:1698-1711):** a symbol omitted from a successful but partial response loses its score but **keeps its strike**. A later single veto then reaches strike 2 even though the two vetoing analyses were not consecutive.

## 5. Sentiment and lexicon (offline)
- `tests/test_sentiment_headlines.py`: **1034/1035 (99.9%)**, rc 0. Scoring 1020/1021; the one miss is `"The market hasn't shown any signs of recovery"` at +0.420 (expected negative). Validation 14/14.
- `learned_lexicon.json` **does not exist** on the device, and neither does `lexicon_eval_report.json`. `sentiment.py` has **no reader** for it (grep: only `learned_lexicon.py` and `scripts/train_lexicon.py` reference it), so the static lexicon is what runs.
- MAP §5(12)'s "learned lexicon over the static one" is doc drift; MODULES correctly says DORMANT.
- **B13 confirmed** (`sentiment._score_text`; `'cut'`/`'cuts'` ∈ `_NEGATIVE`, sentiment.py:109; phrase `('rate cut', 1.0)`, :135). Phase-2 word scoring re-scores the phrase tokens (no masking):

  | Headline | Score | Effect |
  |---|---|---|
  | "Fed signals rate cut in September" | **+0.000** | cancelled; identical to "…rate pause…" |
  | "Fed delivers rate cut" | **+0.000** | cancelled |
  | "Powell hints at rate cuts" | **+0.000** | cancelled |
  | "Fed cuts rates by 50 basis points" | +0.235 | attenuated (+1.5 − 1.0) |
  | "Analysts expect earnings beat" | +0.919 | **amplified** (phrase +1.5 plus word `beat` +1) |
  | "Company slashes price target" | −0.956 | amplified |
  | "Fed signals rate hike in September" | 0.000 | lexicon gap |

  The fix is model-facing (it changes the `Daily_Sentiment` feature and the gate), so it needs an owner decision behind a flag.

## 6. Cost ledger
- Current state: `{"date":"2026-09-26","cost":6.1e-05}` (all from this task); the cap is `_DAILY_COST_LIMIT=1.00` (llm_client.py:175), reset at midnight America/Los_Angeles.
- `_record_cost` does its read-modify-write under the thread lock plus `fcntl.flock(llm_cost.json.lock)` (:167-190, 785-792), so concurrent increments are not lost.
- **Gaps:**
  - The day-rollover `_save_shared_cost()` in `_maybe_reset_quota` (:725) runs **outside** the flock. A process rolling over can overwrite another process's first spend of the new day (tiny window).
  - `_cost_ok` reads the in-memory total, refreshed from the file only on rollover or on this process's own record (:795-800). Each process can overshoot the cap by about one call.
  - Values are rounded to 6 decimals per write (fine).
  - `scripts/llm_qualify.py` bypasses the ledger by design, and the Batch backfill bypasses it by omission (D11).

## Defects (file:line) — none fixed; findings only
- **D1** llm_client.py:226 — `claude-sonnet-5` priced 3/15; list price is 2/10.
- **D2** llm_client.py:233-235 — gpt-5.4 family placeholders are wrong (real: 2.50/15, 0.75/4.50, 0.20/1.25). Output under-billed for mini and nano.
- **D3** llm_client.py:257, 820 — unknown ids fall back to 1.25/10 and 50 RPD. This covers both configured ids (`claude-opus-4-6`, `gpt-4.1`), the whole Claude 5 family, the dated Haiku id and all Gemini 3.x. `KNOWN_MODELS` (:92) also rejects them as role overrides.
- **D4** llm_client.py:1171 — `temperature` is always sent to Anthropic, but the analyst passes 0.2. Sonnet 5, Opus 4.7/4.8/5/5.5 and Fable **reject sampling params (400)**. So `claude-sonnet-5`, the Anthropic fallback in `_ANTHROPIC_FALLBACK_CHAIN` (:79), is broken for the analyst. Only Haiku 4.5 and Opus 4.6 accept it. The same risk likely applies at :1269 for OpenAI gpt-5.x (reasoning models accept only temperature=1 per community reports; verify with `llm_qualify`).
- **D5** llm_client.py:1178 — forced `tool_choice {"type":"tool"}` returns 400 on `claude-opus-5-5` and `claude-fable-5-1`. Do not configure either for the analyst without switching to `auto` + `strict` or `output_config.format`.
- **D6** llm_client.py:880/998/1357/1431 — cost is recorded only `if result`. Billed MAX_TOKENS truncations and safety-blocked or empty responses (usage is returned by the transports) never reach the ledger.
- **D7** llm_client.py:152-156, 815-826 — RPD budgets are per process, in-memory and reset on restart. They do not model the provider's per-project RPD.
- **D8** llm_client.py:910 (and the call_claude/call_openai twins) — the direct single-model calls never trigger the 429 cooldown. The sentiment tier path (sentiment.py:713 → `call_model`) keeps retrying through a storm. The cooldown is a flat 30 s even for a daily-quota (`limit: 0`) body (:659-662).
- **D9** llm_client.py:329-358 — tier auto-detection is dead: the live Gemini response carried no `x-ratelimit-*` headers, so `detected_tier` stays null and `paid` is assumed forever. On a free-tier key, flash budgets are overstated (2000 vs ~250).
- **D10** llm_client.py:725 — the rollover write is outside the flock (see §6).
- **D11** sentiment_history.py:679-708 — Batch backfill spend is never recorded in `llm_cost.json`. The backfill pin also disagrees: `resolve_provider_chain('backfill')` gives `models.gemini.model`, while `get_recommended_model('backfill')` gives the flash-lite routing table.
- **D12** llm_client.py:596-605 — `get_recommended_model` ignores `models.gemini.model` for Gemini-headed chains, so the owner's configured Gemini id is never the analyst primary. On failure the analyst double-dials: `call_model`, then the full `call_llm` chain, repeating flash-lite (llm_analyst.py:575-591).
- **D13** base_loop.py:1698-1711 — stale strikes on partial responses (see §4).
- **D14** llm_client.py:764-768 — one cache-write multiplier (1.25) is used for both `5m` and `1h` TTLs; 1 h writes bill at 2×. Latent: the flag is off.
- **D15** sentiment.py:197-284 — B13 phrase/word double-scoring: cancellation and amplification (see §5). Model-facing, owner queue.
- **D16** Doc drift:
  - CLAUDE.md says "one fail-soft HTTP call" for the sentiment runner; it makes 2 billed Gemini calls.
  - MAP §5(12) says "learned lexicon over the static one"; nothing reads it.
  - llm_config.py docstring:56 says the Gemini default is `gemini-2.5-flash-lite`; that holds only for new configs.
- **Side note:** the untracked `global_context.py:283-287` calls `call_gemini(model=get_recommended_model("analyst"))`. With an Anthropic-headed chain it would send a `claude-*` id to Gemini (404). It was live in the May bots (GLOBAL-CTX lines).

## Recommendation (owner config; no edits made)
1. **Now, with no code change and $0 risk:** stay Gemini-only; it works today at about $0.05–0.20/day. Hygiene-patch `llm_config.json` so a future key paste is safe:
   - `models.claude.model` → `claude-haiku-4-5`. It is the only current Anthropic id compatible with this transport (forced tool use plus temperature), and its table row is correct.
   - `models.openai.model` → `gpt-5.4-nano`, adding `"pricing": {"gpt-5.4-nano": [0.20, 1.25], "gpt-5.4-mini": [0.75, 4.50], "gpt-5.4": [2.50, 15.0], "claude-sonnet-5": [2.0, 10.0]}`. Or keep OpenAI keyless until D4 is qualified.
   - Optionally persist `"selection_mode": "auto"` explicitly so the ignored legacy `"provider": "gemini"` stops misleading readers.
2. **Free-first (runbook Phase 4, MAP §9 #13) is the right next step, because 2.5 access is legacy-gated.** The free-of-charge candidates are `gemini-3.5-flash-lite` / `gemini-3.1-flash-lite` (Google's current recommendation), then OpenRouter/Groq presets.
   - This needs code, not just config: add 3.x rows to `_PRICING`, both budget tables, `_FREE_QUALITY_RANK`, the routing tables and `sentiment._get_scoring_tiers` (all hard-code 2.5 ids).
   - Then run `scripts/llm_qualify.py` plus a week of `--shadow` agreement, and `prompt_ab` before any flip. A model swap moves `s`, so it is gate-behaviour-changing.
3. **Paid Anthropic analyst** (only if `llm_eval` b2 at n≥60 justifies spend):
   - Use `claude-haiku-4-5` only, after fixing D4 (drop `temperature` for models that reject it) and D5.
   - Expect about $3–7/day at 288 calls, which **exceeds the global $1 cap**. Either raise the cap or give the analyst its own sub-budget, so a paid analyst does not starve Gemini sentiment scoring.
   - Before any paid provider goes live, fix D1/D2/D3 pricing (config `pricing` overrides suffice), D6, and D11, so the cap measures real money.

Artifacts: `scratchpad/G/{effective_config.out, smoke_gemini.out, failopen_probe.out, sentiment_runner.out, b13.out, log_daily.txt, llm_config.json.orig, llm_cost.json.orig}`.
Provider sources (fetched 2026-09-26): ai.google.dev/gemini-api/docs/{models,pricing,deprecations,billing,rate-limits}; developers.openai.com/api/docs/{pricing,deprecations}; Anthropic model/pricing tables from the claude-api skill (cached 2026-06-24); GPT-5 temperature: community.openai.com/t/gpt-5-models-temperature/1337957.
