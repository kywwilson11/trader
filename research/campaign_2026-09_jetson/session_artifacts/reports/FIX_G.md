# FIX_G — LLM transport + accounting fixes (2026-09-26)

Files touched: `llm_client.py` (edited), `tests/test_llm_fixes_2026_09.py` (new, 56 tests).
`llm_config.py` is unchanged because its defaults are valid ids. `llm_analyst.py`, `base_loop.py` and `llm_config.json` were not touched.
Original backup: `scratchpad/FIX_G/llm_client.py.orig` (== HEAD). Full diff: `scratchpad/FIX_G/llm_client.diff` (+475/−81).

## ⚠ Verify is NOT fully green: 2 existing tests pin the wrong prices this task was told to fix
The task asks for two things that conflict. It says to fix the D1/D2 prices, and it also says `tests/test_llm*.py` must keep passing unchanged. Two existing assertions pin exactly the defective values:
- `tests/test_llm_claude.py::test_pricing_config_override_wins` (line 215) pins `claude-sonnet-5 == (3.0, 15.0)`. The list price is 2/10 (D1).
- `tests/test_llm_providers.py::test_call_openai_records_cost_nano_pricing` (lines 170-171) pins nano at 0.25/1.00. The published price is 0.20/1.25 (D2).

I applied the fixes and did **not** edit those tests. The exact two-hunk update is ready in `scratchpad/FIX_G/proposed_test_pin_update.diff`, awaiting orchestrator/owner OK.
If you'd rather keep the suite green without touching the tests, revert the two `_PRICING` rows (`claude-sonnet-5`, `gpt-5.4-nano`) and put the corrections in `llm_config.json` `pricing` instead.
The two other pins that looked at risk are preserved **legitimately**:
- `_pricing('unknown-model'/'mystery-model-x') == (1.25, 10)`, because the fallback is now the per-family ceiling and the Gemini ceiling is 2.5-pro's 1.25/10.
- `_cache_multipliers('anthropic') == (1.25, 0.10)`, because the 1 h multiplier is a separate accessor.

## Diff summary (llm_client.py)
- **D4, temperature.** New `_anthropic_accepts_sampling(model)` plus `_CLAUDE_ID_RE` / `_claude_family_version`. It sends `temperature` only to Opus <4.7, Sonnet <5, Haiku <5 and legacy `claude-3/2-*`.
  - It is dropped for Fable 5/5.1, Mythos, Opus 4.7/4.8/5/5.5, Sonnet 5 and any unparseable id. Source: skill "Thinking & Effort" table, Sampling column.
  - Haiku 4.5, the only Anthropic model that worked before, gets a byte-identical body (pinned by the existing full-body test).
- **D5, forced tool use.** New `_anthropic_accepts_forced_tool(model)`, which is False for Opus ≥5.5 and Fable/Mythos ≥5.1 (plus forward guesses Sonnet ≥6 and Haiku ≥5, and unparseable ids). Those models go through the skill's recommended path:
  - `tool_choice {"type":"auto"}`, and the tool gets `strict: true` with the `additionalProperties:false` normalization.
  - The instruction "Respond by calling the emit_json tool exactly once…" is added as a trailing **user-turn** text block, so the system/cache prefix is untouched.
  - The answer is checked client-side against the schema (`_json_matches_schema`: type/required/properties/items/enum). If the tool input fails, a JSON text answer (fences stripped) is accepted when it validates.
  - Otherwise there is **one** validation retry, skipped on `max_tokens`/`refusal`. If there is still no valid answer it returns `None`, so the analyst fails open.
  - Usage is summed across both billed requests. A retry that raises still returns the first request's usage.
  - Forced-capable models keep the old byte-identical forced path. The analyst therefore still receives schema-shaped JSON text or None.
  - A test proves `llm_analyst._parse_response` parses the auto-path output for the real `_response_schema`.
- **Pricing (D1/D2/D3).** New rows are listed in the table below. `_fallback_price` bills an unknown id at the element-wise **max over the tabled rows of its provider family** (anthropic $10/$50, openai $2.50/$15, gemini $1.25/$10), falling back to the whole-table max. The warning is still logged once per id and now prints the ceiling used.
- **Budgets.** Every id that has a `_PRICING` row now has a row in both RPD tables. The 50-RPD unknown default is kept, and `get_budget` now logs once per process which id fell through (`_unknown_budget_warned`).
- **D10, ledger rollover.** New `_rollover_cost_locked(today)`. `_maybe_reset_quota` now runs the rollover inside `_cost_file_lock()` (thread lock outer, flock inner, same ordering as before), wrapped in a never-raise.
  - Sibling bug fixed: `_record_cost` also rolls over under the same lock before `+= cost`. A call that straddled midnight PT used to add today's cost to yesterday's in-memory total and stamp the sum with today's date.
- **D6, discarded responses.** New `_usage_billed` / `_charge_discarded`, called before every `if result:` at all 6 transport sites: call_gemini ×2, call_llm ×2, call_claude, call_openai.
  - Truncated, blocked, empty or invalid responses that carry billed usage are now charged to the ledger.
  - Control flow, return values, `record_call` (RPD) and `_last_model_used` are unchanged.
- **D14, 1 h cache writes.** Added `_CACHE_WRITE_1H_MULT_DEFAULTS = {"anthropic": 2.0}` (skill prompt-caching "Economics": 5 m writes 1.25×, 1 h writes 2×) and `_cache_write_1h_multiplier()`. The config can override it with an optional 3rd element: `pricing_cache_multipliers.anthropic = [w5m, read, w1h]`.
  - `_anthropic_usage` records `cacheWrite1hTokenCount` from `usage.cache_creation.ephemeral_1h_input_tokens`. If that breakdown is absent and the request used the 1 h TTL, it infers the count.
  - The field is only added when nonzero, so the 3-key usage pin holds.
- Refactor: `_call_anthropic` was split into `_anthropic_post` / `_anthropic_usage` / `_anthropic_extract_validated`, with no behaviour change on the forced path.

Gate decision behaviour is unchanged. Scores, prompts, parsing, routing, `KNOWN_MODELS` and fallback chains are all untouched. The only request-body changes are for models that previously 400'd.

## Model ids now covered
| id | price in/out $/MTok | RPD free/paid | source |
|---|---|---|---|
| gemini-2.5-pro / -flash / -flash-lite | 1.25/10, 0.30/2.50, 0.10/0.40 (unchanged) | unchanged | ai.google.dev/gemini-api/docs/pricing (fetched 2026-09-26, re-confirmed) |
| **gemini-3.5-flash-lite** (stable) | 0.30/2.50 | 1000*/5000 | same page: paid tier; free tier "Free of charge" |
| **gemini-3.1-flash-lite** (stable) | 0.25/1.50 (audio input 0.50) | 1000*/5000 | same page |
| claude-haiku-4-5 | 1/5 (unchanged) | 5000/5000 | claude-api skill model table (cached 2026-06-24) |
| **claude-haiku-4-5-20251001** | 1/5 | 5000/5000 | skill shared/models.md (dated alias of haiku-4-5) |
| **claude-sonnet-4-6** | 3/15 | 2000/2000 | skill |
| **claude-sonnet-5** | **2/10** (was 3/15, D1) | 2000/2000 | skill |
| **claude-opus-4-6** (currently configured) | 5/25 | 1000/1000 | skill |
| **claude-opus-4-7** / claude-opus-4-8 / **claude-opus-5** | 5/25 | 1000/1000 | skill |
| **claude-opus-5-5** | 4/20 | 1000/1000 | skill |
| **claude-fable-5** / **claude-fable-5-1** | 10/50 | 500/500 | skill |
| **gpt-5.4** | **2.50/15** (was 5/15) | 1000/1000 | developers.openai.com/api/docs/pricing (fetched 2026-09-26) |
| **gpt-5.4-mini** | **0.75/4.50** (was 1/4) | 2000/2000 | same |
| **gpt-5.4-nano** | **0.20/1.25** (was 0.25/1) | 5000/5000 | same |
| **gpt-4.1** (currently configured) | 2.00/8.00 | 1000/1000 | same |
| unknown claude-* | 10/50 (family ceiling) | 50, logged once | — |
| unknown gpt-*/o* | 2.50/15 (family ceiling) | 50, logged once | — |
| unknown other (Gemini/endpoints) | 1.25/10 (family ceiling) | 50, logged once | — |

\* **Gemini free-tier RPD is UNVERIFIED.** ai.google.dev/gemini-api/docs/rate-limits says limits "can be viewed in Google AI Studio" and publishes no per-model numbers. 1000 is copied from the 2.5-flash-lite row as a labelled estimate.

Caveats:
- The Gemini family ceiling (1.25/10) is **not** a ceiling over all of Google's lineup. `gemini-3.1-pro-preview` is 2/12 (4/18 above 200k tokens) and was deliberately not added, which keeps the ceiling and the existing pins. If a 3.x Pro is ever routed, add a `pricing` entry.
- The OpenAI ceiling is 2.50/15, but o1-pro lists at 150/600. So "conservative" means "≥ every tabled sibling", not "≥ everything the provider sells".
- Cache-read multiplier stays 0.10 for all Claude models. Real reads are 0.05× on Opus 5.5 and 0.025× on Fable 5.1, so this over-bills, which is the conservative direction.

## Test tail
```
$ CUDA_VISIBLE_DEVICES='' $JPY -m py_compile llm_client.py llm_config.py tests/test_llm_fixes_2026_09.py   -> OK
$ $JPY -m pytest tests/test_llm*.py tests/test_c26_V1.py tests/test_grp_sentiment.py tests/test_c26_S2.py -q -p no:cacheprovider
FAILED tests/test_llm_claude.py::test_pricing_config_override_wins        (pins old sonnet-5 3/15 — see top)
FAILED tests/test_llm_providers.py::test_call_openai_records_cost_nano_pricing  (pins old nano 0.25/1.00 — see top)
2 failed, 362 passed
$ $JPY -m pytest tests/test_c26_S1.py tests/test_c26_P1.py tests/test_review_b04.py tests/test_parse_scores.py tests/test_sentiment_scoring.py tests/test_imports.py (+S2) -> 150 passed
```
- New file alone: 56 passed.
- Mutation check against the original `llm_client.py`: 43 of the 56 fail. The 13 that still pass are "behaviour kept" pins: haiku/opus-4-6 temperature, the forced path on accepting models, the no-retry-on-max_tokens case, and the analyst-parser round trip.
- Full suite not run, per the brief.

## Live call (one, Gemini, through the client)
- What ran: `call_model(model='gemini-2.5-flash-lite', json_schema=llm_analyst._response_schema(['TEST']), temperature=0)`, with urlopen guarded to exactly one request and `_429_MAX_WAIT_PRIMARY=0`.
- Result: HTTP 200 in 683 ms, schema-valid JSON, and `_parse_response` produced `{'TEST': {s:0.5,…}}`. `last_model=gemini-2.5-flash-lite`.
- **Cost: $0.000031**. Ledger went from $0.000061 to $0.000092 (`llm_cost.json` = `{"date":"2026-09-26","cost":9.2e-05}`).
- `llm_config.json` sha256 unchanged (`50a9ed81…`). No Anthropic/OpenAI calls.
- Output: `scratchpad/FIX_G/smoke_gemini.out`.

## Recommended owner config change (not applied, owner's file)
In `llm_config.json`: `"models.claude.model": "claude-opus-4-6"` → **`"claude-haiku-4-5"`**. It is the cheapest valid Anthropic id ($1/$5) and the only one that keeps both forced tool use and temperature, i.e. a byte-identical body.
After this fix `claude-opus-4-6` is also priced and budgeted correctly ($5/$25, 1000 RPD). Haiku is recommended on cost.
`llm_config.py` needs no change: its defaults `gemini-2.5-flash-lite` / `claude-haiku-4-5` / `gpt-5.4-nano` are valid.
Doc drift to note: llm_config.py docstring lines 103-104 still say an unknown model "bills at llm_client's $1.25/$10". It is now the per-family ceiling. I left it alone, since I was not to touch llm_config.py.

## Not fixed, out of scope, flagged
- `KNOWN_MODELS` (role-override whitelist) still lacks the new ids (D3 part 2). That is routing, not accounting.
- RPD `record_call` is not incremented for discarded-but-billed responses.
- D7/D8/D9/D11/D12/D15 untouched.
- On Opus 5.5 / Fable 5.1, thinking is always on and counts against the analyst's `max_tokens=max(4096, n*400)`. Truncation is handled (it returns None, fails open and is billed), but anyone routing there should consider `output_config.effort="low"`. That is a behaviour choice for the owner.

## D13 — OWNER DECISION (gate behaviour; NOT applied)
**Defect** (base_loop.py:1698-1711, `_run_llm_analysis`): the strike loop only visits symbols **present** in `new_scores`.
- A symbol that was a candidate but was omitted from a successful partial response loses its score, because `self.llm_scores = new_scores` replaces the dict. It **keeps** its stale strike.
- Sequence: veto (strike 1), then omitted, then veto. That reaches strike 2 and triggers forced liquidation, even though the two vetoing analyses were not consecutive.
- The "two consecutive vetoes" guard (the injected-headline defence, arXiv 2601.13082) is weakened.

**Proposed patch** (narrow: only symbols that were *asked about* and omitted; non-candidates keep today's behaviour):
```diff
@@ base_loop.py  _run_llm_analysis  (after the strike loop, ~line 1711)
             for sym, v in new_scores.items():
                 if v.get('s', 0.5) < LLM_VETO_THRESHOLD:
                     self._veto_strikes[sym] = self._veto_strikes.get(sym, 0) + 1
                 else:
                     self._veto_strikes.pop(sym, None)
+            # D13: a candidate the provider OMITTED from a successful but
+            # partial response breaks the "two CONSECUTIVE vetoes" chain —
+            # its score is already gone (llm_scores replaced above), so its
+            # strike must go too, or a later single veto liquidates.
+            for sym in self._last_llm_symbols - set(new_scores):
+                self._veto_strikes.pop(sym, None)
```
- Broader alternative: clear strikes for **every** symbol not in `new_scores`, including held names that were not candidates this cycle. It is stricter about consecutiveness but changes liquidation timing for positions that drop out of the candidate set.
- Either variant only ever *removes* strikes. It cannot create a block or a sell, so it moves in the fail-open direction.
- It is still a gate-behaviour change, so it needs owner sign-off plus a unit test: veto, then omitted, then veto → 0 sells.
