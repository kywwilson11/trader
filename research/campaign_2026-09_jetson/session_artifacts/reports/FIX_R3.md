# FIX_R3: review items L1, L2 and M3 (2026-09-26)

Files edited: `llm_client.py`, `gui.py`, `tests/test_llm_fixes_2026_09.py` and `tests/test_gui_fixes_2026_09.py`.
- In `gui.py` only three things changed: two new module constants above `class DataFetcher`, `DataFetcher.fetch_orders`, and a new sibling method `_walk_orders_pages`. No other hunk in gui.py was touched. I checked this by diffing against the pre-edit copy.
- The tests were only extended. One existing source-contract test was re-pointed; see M3.
- Nothing was staged or committed.

Pre-edit copies: `scratchpad/FIX_R3/{llm_client,gui}.py.pre`.
My diffs: `scratchpad/FIX_R3/{llm_client,gui}.R3.diff`.

## L2: non-canonical Claude ids (llm_client.py)
- **New `_canonical_claude_id(model)`** (next to `_provider_for`). It returns the base first-party alias, or None when the id is not a Claude id. Steps, in order:
  1. lowercase;
  2. drop everything before `claude` (Bedrock `us.`/`eu.`/`global.` + `anthropic.`, or router paths like `anthropic/…`);
  3. strip a trailing `[..]` tag, Bedrock `-vN:M`/`:N`, Vertex `@…`, `-latest` and `-YYYYMMDD`;
  4. turn a dotted version into a dashed one (`4.6` → `4-6`).
- **Classification and pricing only.** The raw id is still what goes on the wire; a test pins this.
- **`_provider_for`**: any id containing `claude` is now Anthropic. OpenAI and Gemini rules are unchanged.
- **`_pricing`** lookup order:
  1. config override for the raw id;
  2. config override for the canonical id;
  3. table row for the raw id;
  4. table row for the canonical id;
  5. the family ceiling.

  An unknown Claude id therefore always bills at the Anthropic ceiling ($10/$50), never the Gemini one.
- **`get_budget`**: an untabled spelling now uses its base model's RPD row. The call counter stays keyed on the raw id.
- **`_claude_family_version`** and the legacy `claude-3`/`claude-2` checks now run on the canonical id. So `us.anthropic.claude-opus-4-6` gets exactly the canonical body: forced tool plus temperature.

Probe results:

| id | before | after |
|---|---|---|
| `us.anthropic.claude-opus-4-6` | gemini, $1.25/$10, auto path, temperature dropped | anthropic, $5/$25, forced path, temperature kept |
| `claude-haiku-4-5-latest` | auto path, $10/$50 | Haiku 4.5, $1/$5, forced path |

- **Still open (not in scope):** `claude-opus-4-5` and `claude-sonnet-4-5` have no table row, so they bill at the $10/$50 ceiling. That is conservative. `KNOWN_MODELS` is untouched.

## L1: validation retry pre-flight and RPD (llm_client.py)
- **`_record_cost` split:** its pure cost arithmetic is now `_cost_of(...)`. The numbers are unchanged, and `_record_cost` keeps every never-raise fallback.
- **New `_retry_preflight(model, pending_usage)`.** It runs the same gates the first request passes, in this order:
  1. provider 429 cooldown;
  2. `_cost_ok()`;
  3. `_daily_cost` plus the cost of the first request (billed but not yet on the ledger) must be under the cap;
  4. RPD remaining must be at least 2 (the first request is not counted yet);
  5. `_rate_limit_ok()`, placed last because a pass consumes a slot.

  It never raises. A refusal prints `validation retry refused (<reason>)`, and `_call_anthropic` returns `(None, first_usage)`. The fail-open semantics are unchanged: the caller charges the first request via `_charge_discarded` and returns None.
- **The retry counts against RPD:** `record_call(model)` runs inside the transport right after the retry response arrives.
- **Discarded billed responses now count against RPD at all 6 transport sites:** `_charge_discarded` calls `record_call` whenever it charges. Unbilled responses (no usage) are still not counted.
- **Resulting counts:**

  | Outcome | RPD counted |
  |---|---|
  | invalid, then valid | 2 |
  | invalid, then invalid | 2 |
  | retry refused | 1 |
  | forced path | 1 |

## M3: gui.fetch_orders cadence (gui.py)
- **New module constants:** `ORDERS_FULL_WALK_SEC = 600` and `ORDERS_REFRESH_PAGES = 1`. The quick cadence is still the existing user-tunable `orders` timer (30 s default). No new timer was added; the full walk is decided by elapsed monotonic time on each tick.
- **The page walk moved verbatim into `_walk_orders_pages(after, max_pages=None)`.** It returns `(orders, truncated, exhausted)`. The RFC3339 `_alpaca_until` cursor, `seen_ids` dedupe, 1000/365-day caps and stuck-cursor guard are unchanged.
- **When `fetch_orders` does a full walk:**
  - at boot, when there is no cache;
  - every 600 s;
  - when the `.clean_slate` cutoff changes;
  - on a **gap**: the refresh page is full and shares no id with the cache, meaning more than 100 new orders arrived since the last tick.
- **Otherwise it fetches page 1 and merges.**
  - Fresh rows come first and win on status or fill updates.
  - Cached rows are then appended if their id was not in the fresh page. Id-less cached rows are dropped until the next walk.
  - `truncated` is carried over from the last full walk.
- **Errors keep the existing `_stream_result` back-off.**
  - With a warm cache, a failed scheduled walk is retried on the 600 s cadence, not every tick. The cache is kept.
  - A failed boot walk is retried on the next tick.
- **Known staleness:** a status change on an order older than the newest 100 shows up at the next full walk, at most 10 minutes later.

### Headless proof (base env, `QT_QPA_PLATFORM=offscreen`, 180 s from `show()`, live read-only Alpaca)
- Driver: `scratchpad/FIX_R3/orders_driver.py`, adapted from `F/fix_driver.py`. It uses the same redirection of gui_settings.json and news_cache.json into scratch, intercepts the baseline write, suppresses dialogs and makes no LLM calls. `api.list_orders`, `get_account` and `list_positions` are wrapped with counters.
- "before" is the pre-edit gui.py (`FIX_R3/orig_mod/`, with `BASE_DIR` pinned to the repo).
- Output: `FIX_R3/runs/{before,after}.{out,err}` and `orders_result_{before,after}.json`.

| | before | after |
|---|---|---|
| list_orders requests in 3 min | **67** (22.3/min) | **17** (5.7/min) = 1 boot walk (11 pages) + 6 page-1 refreshes |
| orders emits | 6 × 1088 rows, truncated=True | 6 × 1088 rows, truncated=True |
| Recent Fills rows / Open Orders rows | 50 / 0 | 50 / 0 |
| Est. Tax "Realized Gains" card | -$3,950.69 | -$3,950.69 |
| orders stream / status bar | fails 0 / "API: OK" | fails 0 / "API: OK" |
| uncaught exceptions | 0 | 0 |
| account_baseline.json / pipeline_command.json created | no / no | no / no |

- Steady state (after the boot walk) at the 30 s default is about 2 req/min, plus about 11 requests every 10 min (about 1.1/min averaged), for roughly 3 req/min in total. Before, it was about 22 req/min. At the 10 s minimum cadence it is about 7 req/min, against about 66 before.
- The 600 s re-walk is not visible in a 3-minute window. It is covered by the unit tests below.

## Tests
- **`tests/test_llm_fixes_2026_09.py`:** 56 → 96 tests.
  - L2: a table of 16 id spellings covering canonical form, provider, family version and price. It includes `us.anthropic.claude-opus-4-6`, `claude-haiku-4-5-latest`, `claude-opus-4-5@20251101`, Bedrock `-v1:0`/`-v2:0`, `[1m]`, dotted versions and `anthropic/…`.
  - L2 also checks: non-Claude ids are unchanged; unknown Claude ids bill at the Anthropic ceiling (never Gemini); a config override applies through the canonical id (an exact-id override still wins); the budget uses the base row; the prefixed id's request body equals the canonical one apart from `model`; `call_model` routing; prefixed ids use the Anthropic cache multipliers.
  - L1: the retry takes a rate-limit slot, and RPD counts 2 for both valid-after-retry and invalid-twice. The retry is refused by each gate: cost cap including the first request, cap hit mid-flight, RPD with room for only 1, rate limiter full, and 429 cooldown. Also: the pre-flight never raises, the forced path counts 1 request, and discarded billed responses count RPD for Gemini and Claude but unbilled ones do not.
  - Mutation check: against the pre-edit `llm_client.py`, **37 of the 40 new tests fail**. The 3 that pass are "unchanged behaviour" pins.
- **`tests/test_gui_fixes_2026_09.py`:** 33 → 44 tests.
  - The new tests are PySide6-free. They exec `fetch_orders` and `_walk_orders_pages` from the AST against a fake paginating Alpaca API (exclusive RFC3339 `until`) and a fake clock.
  - They pin the constant values (600 / 1), and check:
    - boot does a full walk;
    - a refresh is 1 request that merges by id and newest-first order is kept;
    - a new full walk happens at 600 s;
    - a gap forces a full walk;
    - an error backs off and keeps the cache;
    - a failed scheduled walk is not retried every tick, while a failed boot walk is;
    - a clean-slate change forces a full walk;
    - `truncated` is carried over;
    - the 3-minute budget is 10 + 6 requests, against 70 before.
  - The existing `TestAlpacaUntil::test_fetch_orders_uses_helper` source contract now reads `_walk_orders_pages`, where the loop moved, with the same assertions. It also asserts that `fetch_orders` calls the walk and never calls `list_orders` directly. This relocates the test; it does not weaken it.

## Verify
```
$JPY -m py_compile llm_client.py tests/test_llm_fixes_2026_09.py tests/test_gui_fixes_2026_09.py   OK
/home/kyle/miniforge3/bin/python -m py_compile gui.py                                              OK
CUDA_VISIBLE_DEVICES='' $JPY -m pytest tests/test_llm*.py tests/test_gui_fixes_2026_09.py tests/test_c26_S1.py tests/test_c26_S2.py tests/test_review_b04.py -q -p no:cacheprovider
448 passed in 17.28s
```
Also green:
- test_c26_P1, V1, test_imports, test_parse_scores and test_sentiment_scoring: 129 passed.
- Tests that inspect GUI or order source (gui_contracts, gui_charts, c26_U1, review_b03, tax_lots, review_b02, c26_X1): 186 + 76 passed.

The full suite was not run.

**LLM spend: $0.** I made no Gemini, Anthropic or OpenAI calls. The pre-flight is proven by unit tests with faked transport.

## Notes for the owner
- The L1 accounting change makes RPD counters rise faster: discarded and billed responses, and retries, now count. This can exhaust a model's RPD slightly earlier on heavy-truncation days, which is the intended conservative direction.
- The L2 `get_budget` canonical fallback goes slightly beyond the review's ask. It fixes the matching 50-RPD mis-budget for prefixed ids.
- `scripts/llm_qualify.py` calls `_call_anthropic` directly. Its validation retry now also passes the pre-flight, which only matters if the $ cap is hit during a qualification run.
