# IMPL_g4 — G4 hunt fixes (LLM + sentiment layer), 2026-09-27

Owned files edited: sentiment_history.py, sentiment.py, novelty.py, llm_config.py, llm_client.py
New test: tests/test_g4_fixes_2026_09.py (11 tests; Mac-safe: temp sqlite, faked Finnhub/requests/json.dump, no network).
NOT touched: llm_analyst.py (owned by another implementer), tests/test_llm_analyst.py, _score_text (G4-05 deferred).
No git operations. No full suite.

## Changes
- G4-01 sentiment_history.fetch_stock_sentiment_history: `db.commit()` at the end of each 30-day window body
  (before `window_start` advances), so no write txn spans the next company_news call, the 25/min sleep or a
  429 back-off. The per-ticker commit is kept (no-op when nothing is pending). Rows/aggregation/return identical.
- G4-02 fetch_crypto_sentiment_history (migrated path): `result[date_str] = score` moved BEFORE the INSERT try;
  a failed cache INSERT now only loses the cache row. [train] on failure path only.
- G4-08 same function's 429 loop: on attempt == 2 it logs and `break`s instead of sleeping 248 s. Same 3 requests,
  sleeps now [62, 124].
- G4-03 sentiment_history (headline/summary/url) and sentiment.get_news_sentiment crypto filter (:1082-1083):
  `(a.get(k) or '')`, identical to the guarded get_recent_headlines site. Non-None inputs byte-identical.
  [gate] (crypto symbol-news factor now computed instead of neutral on a None field) / [train] (ticker no longer skipped).
- G4-04 novelty._save and llm_config.save_llm_config: tmp = `{path}.{pid}.{thread_ident}.tmp` + os.replace
  (trade_memory idiom). llm_config also now unlinks its tmp on a failed save (per-writer names would otherwise
  accumulate; novelty already did this). Added `import threading` to llm_config.
- G4-10 llm_config.load_llm_config: `copy.deepcopy` for missing top-level defaults, missing provider rows and
  missing per-provider keys. Values identical; only identity changes.
- G4-09 llm_client: module `_rate_lock = threading.Lock()`; `_rate_limit_ok` does prune/check/append (and takes
  `now`) under it. `_get_rate_limit_rpm()` (may read llm_config from disk) is called before the lock; it does not
  depend on the deque, so single-thread behaviour is identical.

## Verification
- py_compile: all 5 modules + new test OK.
- Tests prove the bugs: the new test file run against reverse-patched copies of the 5 modules
  (scratchpad/g4_orig/) -> 9 failed / 2 passed (the 2 that pass on old code: the scaled-down rate-limit stress,
  which does not reproduce at test size — the deterministic lock-interleave test covers G4-09 — and the
  failed-save-no-tmp test, since the reverse copy kept the new unlink). Against the fixed tree: 11 passed.
- Requested set (`$JPY -m pytest` tests/test_g4_fixes_2026_09.py test_sentiment_history.py test_sentiment_pit_2026_09.py
  test_sentiment.py test_sentiment_gate.py test_sentiment_scoring.py test_novelty.py test_llm_config.py
  test_llm_fixes_2026_09.py test_llm_client.py test_c26_P5.py): **262 passed**.
- Adjacent: test_review_b04, test_grp_sentiment, test_c26_T2, test_llm_routing, test_c26_S2, test_llm_providers,
  test_llm_claude: **144 passed** (1 pre-existing pandas SettingWithCopyWarning from data_sources.py:203).
- `$JPY tests/test_sentiment_headlines.py`: **GRAND TOTAL 1034/1035 (99.9%)** — unchanged.

## Notes / follow-ups (not done — out of my file/item scope)
- tests/test_review_b04.py:274 asserts `store.json.tmp` absent — still passes but is now vacuous (should glob `*.tmp`).
  The new test file asserts `glob('*.tmp') == []` for both novelty and llm_config.
- sentiment._try_llm_retry (:930-933) has the same unlocked check-then-popleft pattern (hunter suggested
  `try: popleft() except IndexError: return`); not in my assigned G4-09 scope (llm_client only).
- G4-07 sentiment.py:610 / sentiment_history.py:1012 NaN clamps were not assigned to me; untouched.
- Queued for after the llm_analyst owner: G4-04 (llm_analyst.py:753 + test_llm_analyst tmp assertion), G4-06, G4-07.
- No stale `*.tmp` files exist in the repo root.
