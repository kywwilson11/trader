# G4 — LLM + sentiment layer: indisputable-improvement hunt (2026-09-26, Jetson, read-only)

**Scope.** Files: llm_client.py, llm_analyst.py, llm_config.py, sentiment.py, sentiment_history.py,
novelty.py, fundamentals.py, learned_lexicon.py, trade_memory.py. Read in full, against the working
tree (which includes today's FIX_G / FIX_R3 / FIX_H / FIX_R2 / FIX_E).

**Already covered elsewhere, so NOT re-reported here:**
- Reports: G_llm D1–D16, FIX_G, FIX_R3 (L1/L2), FIX_H, FIX_R2 (M2), FIX_E.
- REVIEW_fixes.
- Owner queue in research/module_review_2026-07.json. Those items are listed in Appendix A.1 for traceability.
- B13 (phrase/word double count), per the brief.

**How the proofs were produced.**
- Every repro is under `scratchpad/hunt/g4/` and runs with `CUDA_VISIBLE_DEVICES='' $JPY <file>`.
- Each one ran in a few seconds and stayed under 100 MB RSS. The exception is `rate_limit_race.py`, which takes about 60 s.
- No network was used: Finnhub, alternative.me, FMP, yfinance and urlopen are all faked.
- The live `sentiment_cache.db` was opened only as `mode=ro`, and only in curly.py.
- No production file was edited.

Findings are ranked by severity. Tags:
- **[train]** — the fix changes stored training data at the next fetch or harvest.
- **[gate]** — the fix changes a live gate input.
- **[prompt]** — the fix changes LLM prompt text.

All three are called out explicitly, as the brief requires.

---

## G4-01 · sentiment_history.py:568-638 · class D (+A) · a SQLite write transaction is held open across Finnhub network calls and the 25/min rate-limit sleep

**Defect.**
- `fetch_stock_sentiment_history` INSERTs each 30-day window's articles on the thread-local connection. The first INSERT implicitly opens a transaction.
- It commits only once per ticker, at :638, after **all** windows.
- So the write lock is held during:
  - every later `client.company_news` call (Finnhub timeout 10 s);
  - the rate-limit `time.sleep` at :576-580 (up to 60 s);
  - the 429 back-off at :595-597 (62, then 124, then 248 s).
- Every other writer on `sentiment_cache.db` waits out the 60 s busy timeout and then raises `sqlite3.OperationalError: database is locked`. The other writers are:
  - the crypto harvest's `_migrate_fng_date_basis` (BEGIN IMMEDIATE) and its FnG INSERTs;
  - the backfill worker's UPDATE and `_batch_state_set`;
  - `set_live_mode`.

**Proof.** `hunt/g4/fetch_stocks_lock_hold.py`, scaled down: the window-2 call sleeps 3 s in place of `sleep(62)`, and the other writer's timeout is 1 s in place of 60 s.
```
other writer: OperationalError database is locked
fetch connection in_transaction during window-2 network call: [True]
```

**Why this happens in production.**
- `run_pipeline.py:1504` launches `sentiment_history.py --fetch-stocks` in the background, then immediately runs `_run_training(phases)`, which includes `harvest_crypto_data.py`.
- The crypto harvest calls `fetch_crypto_sentiment_history` (a writer) while the fetch is still running.
- A single 429 back-off (62 s or more, inside the open transaction) is longer than the 60 s busy timeout.

**Consequences, all verified in code:**
- (a) On an unmigrated DB, the one-time UTC migration is refused (`except Exception`, `sentiment_history.py:403-417`). The harvest then trains on the **legacy 1-day-leaked FnG values** that FIX_H exists to remove.
- (b) On a migrated DB, the day's INSERT is swallowed and the day becomes 0.0. See G4-02.
- (c) The backfill worker (`run_backfill_worker`, :1032-1141) has no exception guard, so it dies. `run_pipeline` never polls or restarts `backfill_proc` (grep: only :1535-1549 reference it).

**Fix.** Commit each window's inserts before the next network call or sleep:
```diff
@@ sentiment_history.py fetch_stock_sentiment_history, end of the per-window body (~:633)
                 ticker_articles += cur.rowcount

+            db.commit()   # never hold the write lock across the next network call / sleep
             window_start = window_end + datetime.timedelta(days=1)
```
- Keep the per-ticker commit at :638 as well. It becomes a no-op when nothing is pending.
- Pair this with G4-08, which removes the pointless final sleep.

**Blast radius.**
- Rows, aggregation and return value are identical. Aggregation still runs after all windows.
- The only semantic change: a crash mid-ticker leaves earlier windows committed. The exception path already does this today; see G4-03(b).
- Tests: `test_c26_P5.py` and `test_c26_T2.py` exercise the function. None pins the commit granularity (grep for commit / in_transaction).

**Why indisputable.** Holding a SQLite write lock across a network call and a sleep of 62 s or more, against a documented multi-writer DB (WAL is enabled precisely for concurrency) with a 60 s busy timeout, is a textbook lock-scope bug. The fix only narrows the lock scope.

---

## G4-02 · sentiment_history.py:427-441 · class A [train] · a cache-INSERT failure silently zeroes a fetched FnG day

**Defect.**
- In the migrated (UTC) path, `result[date_str] = score` sits inside the `try` after `db.execute(INSERT OR IGNORE ...)`, and `except sqlite3.Error: pass` swallows the failure.
- A value the function has **already fetched** is therefore dropped from the returned dict whenever its cache write fails. `database is locked` from G4-01 is the realistic cause.
- The crypto harvest does `.map(sentiment).fillna(0.0)` (`harvest_crypto_data.py:458-460`), so that day's `Daily_Sentiment` becomes 0.0 in the training store.

**Proof.** `hunt/g4/fng_insert_swallow.py`: the fetch is stubbed, and the INSERT for 2026-03-02 raises `database is locked`.
```
returned: {'2026-03-01': -0.6, '2026-03-03': -0.2}
fetched value for 2026-03-02 was FnG=75 -> score 0.5 ; harvest reads res.get(day, 0.0) -> 0.0
```

**Fix.** Set the result from the fetched value regardless of the cache write:
```diff
-        try:
-            db.execute("INSERT OR IGNORE INTO fng_daily ...", (date_str, value, score))
-            result[date_str] = score
-            inserted += 1
-        except sqlite3.Error:
-            pass
+        result[date_str] = score          # the fetched value is known either way
+        try:
+            db.execute("INSERT OR IGNORE INTO fng_daily ...", (date_str, value, score))
+            inserted += 1
+        except sqlite3.Error:
+            pass                          # cache miss only; next run re-inserts
```

**Blast radius.**
- Output changes only on the failure path, where 0.0 becomes the true value.
- Tests `test_sentiment_pit_2026_09.py` and `test_sentiment_history.py` cover the success path, which is unchanged.
- **[train]**: this takes effect at the next crypto harvest. That harvest is already a gotcha-#2 rebuild event because of FIX_H.

**Why indisputable.** A write-through cache failure must not erase data the function holds in memory. The value is PIT-identical either way; it just came from the network instead of the cache.

---

## G4-03 · sentiment_history.py:608-612 and sentiment.py:1082-1083 · class A + B · Finnhub None-valued fields crash two unguarded copies of the article-field access (the copies have diverged)

**Defect.**
- The codebase already treats None-valued Finnhub keys as real:
  - the comment at `sentiment._build_score_prompt` (:571-572);
  - the owner-queue P2 fix for that function;
  - `_deduplicate_articles` (:424);
  - `get_recent_headlines` (:1255-1256), which uses `(a.get('headline') or '')`.
- Two sibling copies still use `a.get(k, '')`, which returns **None** when the key is present with a null value. They then call `.lower()` or `.strip()` on it:
  - **(a)** `sentiment.get_news_sentiment` (crypto relevance filter, :1082-1083) raises AttributeError. The outer `except` returns **None** for the whole symbol: the symbol-news factor in `sentiment_gate` becomes neutral 1.0. Nothing is cached, so every later call refetches Finnhub.
    - Note the `or` short-circuit: a None *summary* crashes on any article whose headline does not contain the base ticker.
  - **(b)** `sentiment_history.fetch_stock_sentiment_history` (:608/611/612) raises. The per-ticker `except` skips the ticker. The INSERTs it had already executed stay in the connection's open transaction and are committed by the *next* ticker's `db.commit()` **without ever being aggregated**.
    - A later incremental run restarts from the newest cached article date (:531-536, `fetch_start = max_dt`), so the earlier days are never aggregated either.
    - `daily_sentiment` stays missing for days that do have articles, so the stock harvest's `cached_only` read stamps them `Daily_Sentiment = 0.0`.

**Proof.** `hunt/g4/finnhub_none_fields.py`, with fake clients and the LLM path stubbed to keyword scoring.
```
(1) get_recent_headlines  -> ['Bitcoin surges past 100k on ETF inflows', 'Crypto market wrap: altcoins slide']
[SENTIMENT] News error for BTC/USD: 'NoneType' object has no attribute 'lower'
(1) get_news_sentiment    -> None          cached? -> False
[SENTIMENT_HIST] Skipping AAA: 'NoneType' object has no attribute 'strip'
(2) AAA articles committed: [('2026-01-01',), ('2026-01-05',), ('2026-01-20',)]
    AAA daily_sentiment   : []
    after rerun AAA daily : [('2026-01-20',)]
    get_daily_sentiment(AAA, 2026-01-05) = 0.0 (article keyword_score = 0.5597...)
```

**Fix.** Use the idiom `get_recent_headlines` already uses:
```diff
@@ sentiment.py:1082-1083
-                        if base in a.get('headline', '').lower()
-                        or base in a.get('summary', '').lower()]
+                        if base in (a.get('headline') or '').lower()
+                        or base in (a.get('summary') or '').lower()]
@@ sentiment_history.py:608-612
-                headline = a.get('headline', '').strip()
+                headline = (a.get('headline') or '').strip()
-                summary = a.get('summary', '').strip()
-                url = a.get('url', '').strip()
+                summary = (a.get('summary') or '').strip()
+                url = (a.get('url') or '').strip()
```

**Blast radius.**
- Non-None inputs produce byte-identical results.
- The matching semantics (the owner-queue P2 "substring relevance" item) are untouched.
- **[gate]** for (a): the symbol-news multiplier is computed instead of defaulting to neutral.
- **[train]** for (b): the next `--fetch-stocks` plus harvest store articles that were previously lost.
- No test pins the crash.

**Why indisputable.** It is the same field access in the same module family. One copy is already guarded and the other raises. The guard's own comment documents that the input occurs.

---

## G4-04 · llm_analyst.py:753, novelty.py:87, llm_config.py:337 · class D · a shared `<file>.tmp` path turns the "atomic write" into a torn-file generator under concurrent writers

**Defect.**
- Three writers use `tmp = <target> + ".tmp"`, a single name shared by every writer.
- Writer A opens tmp (truncate) and writes part of its data. Writer B opens the *same* tmp, truncates it, writes, and `os.replace`s it live. A then keeps writing through its still-open fd into the inode that is now the **live** file. A's own `os.replace` fails with ENOENT.
- The live file is a splice of two writers.
- `trade_memory._save` fixed exactly this bug with a per-writer name (`trade_memory.py:104-108`, `{pid}.{thread_ident}.tmp`). These three copies of the idiom have diverged from it.

**Who collides:**
- **`llm_analysis.json`**: the crypto and stock loops (two threads in combined mode, two processes in `run_pipeline`'s default split mode) and the GUI "Refresh LLM" subprocess (`refresh_all`, which writes one batch after another).
  - A torn file makes `load_analysis()` return `{}`.
  - The next `_save_analysis` writes only its own section, so the other book's section is lost.
  - On the weekly-retrain **cold restart**, `base_loop.py:612-637` pre-loads cached LLM scores from this file, so they are lost.
- **`novelty_store.json`**: the two bot processes in split mode. `_LOCK` is thread-only.
  - A torn store makes `_load()` reset to `{}`, which erases all 7-day reprint history for both books.
- **`llm_config.json`**: the GUI settings autosave, `load_llm_config`'s one-time migration save (any process), and `_capture_rate_limit_headers` (bots; dormant today because Gemini sends no headers).
  - A torn file makes every process fall back to defaults. The file holds every API key.

**Proof, deterministic interleave (a json.dump shim pauses writer A mid-write):**
- `hunt/g4/save_analysis_race.py` (threads, as in combined mode):
  ```
  llm_analysis.json TORN: JSONDecodeError Extra data: line 13 column 2 (char 245)
  load_analysis() -> {}
  PATCHED: llm_analysis.json parses OK; sections = ['crypto'] ; leftover tmp files = []
  ```
  The second half re-runs the same interleave against the fix, exec'd from the real function source with only the tmp line changed.
- `hunt/g4/novelty_xproc_race.py` (two forked processes, as in split mode):
  ```
  novelty_store.json TORN: JSONDecodeError Extra data: line 1 column 419 (char 418)
  reloaded store symbols -> []
  ```

**Fix.** The trade_memory idiom, one line per site:
```diff
@@ llm_analyst.py:753
-    tmp_path = _ANALYSIS_FILE.with_name(_ANALYSIS_FILE.name + ".tmp")
+    tmp_path = _ANALYSIS_FILE.with_name(
+        f"{_ANALYSIS_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
@@ novelty.py:87
-    tmp = str(_STORE_FILE) + '.tmp'
+    tmp = f"{_STORE_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
@@ llm_config.py:337
-        tmp = str(LLM_CONFIG_FILE) + ".tmp"
+        tmp = f"{LLM_CONFIG_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
```
- llm_analyst additionally needs `import threading`.
- In `_save_analysis`, widen the unlink to cover the error path as well (`except OSError` → also unlink the tmp), as novelty already does.
- Cross-process last-writer-wins on *content* remains. That is documented and accepted in `_save_analysis`'s comment; this fix removes only the tear.

**Blast radius.**
- `tests/test_llm_analyst.py::TestSaveAnalysisAtomicWrite::test_survives_garbage_preexisting_tmp_file` asserts that a garbage file at the old fixed `.tmp` path is gone afterwards. With per-writer names that stale file is simply ignored, so the assertion needs re-pointing (assert the target is valid JSON; the stale file is harmless).
- `tests/test_review_b04.py:274` (`store.json.tmp` absent) still passes, but becomes vacuous; it should glob `*.tmp`.
- `test_writes_valid_json_no_tmp_residue` still passes.

**Why indisputable.** It is the same bug that was already fixed, and documented as fixed, in trade_memory. The fix is a name change.

---

## G4-05 · sentiment.py:202 (`_score_text`) · class A (scoring-path defect; **model-facing [train][gate]**) · a typographic apostrophe (U+2019) defeats every negator

**Defect.**
- Negation relies on three things matching:
  - `_NEGATORS` / `word.endswith("n't")` in phase 2;
  - `_NEG_PREFIX` in phase 1;
  - the negated phrases (`"wouldn't touch"` and similar).
- All three assume an ASCII `'`. Headlines from news wires routinely use `’`.
- `_PUNCT` (`[^\w\s'-]`) deletes `’`, so `isn’t` becomes the token `isnt`, which matches nothing. The negation is dropped and the sentiment **flips sign**.
- The same text with a straight apostrophe scores correctly.
- This is not B13, and it is not a lexicon *expansion*: no entry is added.

**Proof.** `hunt/g4/curly.py`:
```
+0.3983 curly | Here’s Why ABNB Stock Isn’t a Buy        -0.6711 straight
+0.5122 curly | Analysts don’t expect growth             -0.7914 straight
+0.4749 curly | Bitcoin won’t rally this week            -0.7450 straight
+0.8884 curly | Company doesn’t beat expectations        -0.9401 straight
articles with U+2019: 8088; headline keyword score differs when normalized: 2578   (of 105,753 in sentiment_cache.db)
```
A read-only count on the live DB: 652 headlines contain `n’t`, against 1,371 with `n't`. So about one third of all contraction negations are currently lost.

**Fix.** Normalize at the top of `_score_text`:
```diff
-    text_lower = text.lower()
+    text_lower = text.lower().replace('’', "'").replace('‘', "'")
```

**Blast radius.**
- **Model-facing.**
  - `keyword_score` for new article rows (`sentiment_history._keyword_score`) is a training feature.
  - The sentiment gate's keyword fallback is a live gate.
  - GUI keyword scores change.
  - Already-stored `keyword_score` rows are unchanged unless they are rescored.
- It must ship only inside the owner-approved rebuild window, the same way as FIX_H.
- Pinned tests: the `test_grp_sentiment.py` T1 goldens contain no U+2019, so they are unaffected. `tests/test_sentiment_headlines.py` contains no U+2019 (grep count 0).

**Why indisputable.** The same sentence scores with the opposite sign depending on the apostrophe glyph. That is a parsing defect in the negator match, not a modelling choice.

---

## G4-06 · llm_analyst.py:1243-1268 (`build_compact_evidence`) and :860-882 (`_build_symbol_profiles`) · class A [prompt] · FIX_E's remaining fundamentals format sites

**Defect.**
- FIX_E hardened `fundamentals.format_fundamentals_for_llm` against yfinance's string values (`'Infinity'`, `'N/A'`, `''`). The fundamentals dict that `get_fundamentals` caches still carries those raw strings.
- Two more sites format the same raw values with `:.1f`, `>= 1e12` and `* 100`:
  - **`build_compact_evidence`** raises ValueError. `base_loop.py:1834-1847` and `stock_loop.py:530-541` catch it and drop the **entire** evidence block, including the technical-snapshot lines that have nothing to do with P/E.
  - **`_build_symbol_profiles`** (used by `refresh_one` / `refresh_all`, the GUI Refresh buttons): its per-symbol `except` drops the whole profile (price, technicals, news).

**Proof.** `hunt/g4/fundamentals_sites.py`:
```
(c) pe=21.3        -> block OK (98 chars)
(c) pe='Infinity'  -> ValueError: Unknown format code 'f' for object of type 'str'  (whole block dropped by caller)
(d) trailingPE=21.3        -> profile present: True
(d) trailingPE='Infinity'  -> profile present: False
```

**Fix.** Reuse FIX_E's single source of truth, `fundamentals._safe_float`:
- `build_compact_evidence`: `pe = _safe_float(fundamentals.get('pe_ratio'))`, and the same for pb, market_cap, revenue_growth, beta, week52_high and week52_low.
- `_build_symbol_profiles` fundamentals loop: `v = _safe_float(info.get(key))` before the formatting branch. `sector` stays a string.
- A None value is omitted, exactly as the existing `is not None` checks already do.

**Blast radius.**
- For numeric inputs the output is byte-identical.
- For string inputs the block is kept instead of dropped. That is prompt text, but only in the case that currently crashes.
- `build_compact_evidence` is live only when `rich_context_enabled=True` (default False). `_build_symbol_profiles` runs on the GUI Refresh path.
- Tests: `tests/test_llm_advice.py` (numeric inputs) is unaffected.

**Why indisputable.** It is the exact crash class FIX_E fixed, at the two remaining sites that read the same dict.

---

## G4-07 · llm_analyst.py:1094/1108/1114, sentiment.py:610, sentiment_history.py:1012 · class A (low) · a NaN score clamps to MOST BULLISH, and `conviction: Infinity` raises into the trading cycle

**Defect.**
- Every clamp has the form `max(lo, min(hi, float(x)))`. Python's `min`/`max` keep the first argument when the comparison involves NaN, so NaN becomes `hi`:
  - the analyst: `s` becomes 1.0, which is the 1.5× size tilt;
  - sentiment: +1.0.
- The fail-open value is 0.5 for the analyst and None (keyword gap-fill) for sentiment.
- `json.loads` accepts the bare `NaN` / `Infinity` tokens, and `float('nan')` accepts the string.
- Separately, in the advisor-v2 path, `int(float('inf'))` raises **OverflowError**, which `except (TypeError, ValueError)` does not catch. It propagates out of `_parse_response` and `analyze_trades` into `base_loop.run_cycle` (:389) before sells run.

**Proof.** `hunt/g4/nan_clamp.py`, plus an inline check:
```
analyst  s=NaN   -> 1.0 (m = 1.5 )  expected neutral 0.5
analyst  s="nan" -> 1.0
sentiment _parse_scores({"1": NaN, "2": 0.1}) -> [1.0, 0.1]  expected [None, 0.1]
_parse_response(..., conviction: Infinity, extended=True) -> RAISES OverflowError
```

**Fix.** Treat non-finite values as unparseable. Add `if not math.isfinite(v): <fallback>` after `float(...)` at each site: s → 0.5, p_up → None, sentiment → None, batch → skip. Add `OverflowError` to the conviction `except`.

**Blast radius.**
- Finite inputs are unchanged.
- Reachability is low: schema-enforced Gemini/OpenAI output cannot emit NaN, but the free-text sentiment path and OpenAI-compatible endpoints can.
- Advisor v2 is off by default.
- Tests `test_parse_scores.py` and `test_llm_analyst.py` contain no NaN or inf cases.

**Why indisputable.** A missing or invalid score must never map to the maximum bullish value, and a parse helper documented as fail-open must not raise.

---

## G4-08 · sentiment_history.py:584-601 · class B/C (low) · the 429 retry loop sleeps 248 s after its LAST attempt

**Defect.** `for attempt in range(3)`: on a 429 it sleeps `62 * 2**attempt` and loops. On `attempt == 2` it still sleeps 248 s, then the loop ends with `articles = None` and no further request. Today that sleep also runs inside the open write transaction (G4-01).

**Proof.** `hunt/g4/finnhub_final_sleep.py` (sleep recorded, not executed):
```
requests at t = [0.0, 62.0, 186.0] | sleeps = [62, 124, 248] | sleep after the last request = 248.0 s
```

**Fix.** Skip the sleep on the last attempt: `if attempt < 2: time.sleep(wait)`. Better, `continue` only while `attempt < 2`.

**Blast radius.**
- The set of requests and the output are identical. The wall time of a fully rate-limited window drops by 248 s.
- `test_c26_P5.py:284` stubs `time.sleep`, and nothing asserts the sleep sequence.

**Why indisputable.** A back-off that precedes no retry buys nothing.

---

## G4-09 · llm_client.py:813-823 (`_rate_limit_ok`) · class A (low) · an unsynchronised check-then-popleft on a shared deque

**Defect.**
- `while _call_timestamps and _call_timestamps[0] < cutoff: _call_timestamps.popleft()` runs with no lock.
- In combined-bot mode, two loop threads reach it: each runs the analyst gate and tiered sentiment scoring.
- It can raise `IndexError: pop from an empty deque`, and the `len >= rpm` / `append` pair can over-admit.
- In `call_gemini`, `call_claude` and `call_openai` the check sits outside the transport `try`, so the error surfaces to callers:
  - the analyst catches it and re-dials through `call_llm`;
  - sentiment's `get_news_sentiment` returns None for the symbol.
- `sentiment._try_llm_retry` (:930-933, `if not _llm_retry_queue: return` then `popleft()`) has the same pattern on the same thread pair. It was not separately reproduced.

**Proof.** `hunt/g4/rate_limit_race.py`: 6 threads × 60k calls, `sys.setswitchinterval(1e-6)`, and a clock with periodic full-window expiry.
```
exceptions: {'IndexError: pop from an empty deque': 2}
```
Production probability is tiny: the calls run at a cadence of minutes.

**Fix.**
- Add a module-level `_rate_lock = threading.Lock()` and wrap the body of `_rate_limit_ok` in it.
- `_get_rate_limit_rpm()` does not touch `_quota_lock`, so no lock-order issue arises.
- The same one-liner applies to `_try_llm_retry`: `try: popleft() except IndexError: return`.

**Blast radius.** The single-thread behaviour is identical. Tests `test_llm_routing.py` and `test_llm_client.py` call it single-threaded.

**Why indisputable.** It is a reproduced data race on shared mutable state. The fix is a lock with no semantic change.

---

## G4-10 · llm_config.py:314-322 · class D (low) · `load_llm_config` hands out the `_DEFAULTS` objects themselves, and the GUI mutates them

**Defect.**
- Missing keys are filled with `config[key] = default` and `config["models"][provider] = pdefault`, with no copy. The loaded config therefore aliases `_DEFAULTS["models"][...]`, `_DEFAULTS["pricing"]`, `["endpoints"]`, `["provider_preference"]` and `["pricing_cache_multipliers"]`.
- `gui._on_settings_changed` (gui.py:7699-7700) does `config.setdefault("models", {}).setdefault(provider, {})["api_key"] = ...`. That writes the typed API key into the module-level defaults for the rest of the GUI process.
- Every later `load_llm_config()` of a file lacking that row returns the key.

**Proof.** `hunt/g4/llm_config_alias.py`:
```
models.claude is _DEFAULTS[models][claude]: True
_DEFAULTS mutated: True -> {'api_key': 'sk-ant-TYPED-IN-GUI', 'model': 'claude-haiku-4-5'}
fresh load of a key-less file -> claude api_key = sk-ant-TYPED-IN-GUI
```

**Fix.** Use `config[key] = copy.deepcopy(default)` and `config["models"][provider] = dict(pdefault)`, with `import copy`.

**Blast radius.**
- Returned values are unchanged. Only object identity differs.
- No test asserts identity (grep `_DEFAULTS` in tests shows value reads only).

**Why indisputable.** It is the mutable-default-sharing hazard the brief lists under D, with a reproduced mutation from existing caller code.

---

## Appendix A — judgment calls and known items (NOT proposed)

### A.1 Already in the owner queue (module_review_2026-07.json) or pinned as known
I reproduced several of these today; they are listed only so nothing is lost.
- **fundamentals `_fetch_fmp_metrics` (:118): `revenuePerShare` is stored as `revenue_growth`.** Reproduced: `RevGrowth=2530.0%` (`hunt/g4/fundamentals_sites.py` (a)). Owner queue P2.
- **fundamentals `get_insider_activity` (:154-158): net subtracts A-Award, M-Exempt and similar.** Reproduced: `0 buys, 0 sells (net -70000 shares)` ((b)). Owner queue P2.
- The following owner-queue P2 items were not re-examined:
  - DivYield ~100× misformat;
  - `_get_scoring_tiers` provider switch is inert;
  - crypto substring relevance filter;
  - `article_date` fallback PIT issue.
- The duplicate `'price target slashed'` phrase (-1.5 and -2.0) is pinned as "known, deferred to owner" by `tests/test_grp_sentiment.py:74`.
- B13 is excluded by the brief.

### A.2 Judgment calls
- **trade_memory.get_lesson_summary (:193)**: `max(set(exit_reasons), key=exit_reasons.count)` breaks ties by the order of a hash-randomised `set`, so the analyst prompt text differs between processes for the same data.
  - Reproduced with PYTHONHASHSEED 1..5, which gives `signal` / `take_profit` alternating (`hunt/g4/lesson_tie.py`).
  - `collections.Counter(...).most_common(1)` would be deterministic.
  - Not proposed, because it changes prompt text.
- **llm_analyst._build_symbol_profiles (:798-801)**: the performance windows are off by one. `close.iloc[-days]` makes "1w" a 4-session return, and "1y" uses `iloc[1]`, not `iloc[0]`. Prompt-only, on the GUI refresh path.
- **llm_client.call_llm (:1240)**: consumes ONE rate-limiter slot for up to N chain dispatches. The analyst then double-dials `call_model` followed by the full chain (G_llm D12). This is a routing and design choice.
- **Negative results are not cached**: `get_fear_greed`, `get_news_sentiment`, `get_market_sentiment`.
  - During a provider outage every `sentiment_gate` call pays up to 5 + 10 + 10 s of timeouts per buy candidate.
  - Negative caching would change which cycle first sees recovered data, so it is not decision-neutral.
- **fundamentals.get_fundamentals** caches an all-None dict for 4 h after a transient yfinance error.
- **novelty**:
  - An exact repeat never refreshes its timestamp, so a daily wire reprint becomes "novel" again after 7 days.
  - A non-Latin-script headline has no `[a-z0-9]` words, so it scores novelty 0.0. This divergence from `learned_lexicon.offline_novelty` (1.0) is documented.
  - Both are design choices.
- **sentiment_history._warn_fng_coverage (:309)** counts a genuine FnG = 50 (score 0.0) as missing. Warning-only.
- **run_backfill_worker**: its loop has no exception guard, and run_pipeline never restarts `backfill_proc`. A guard plus sleep would be D, but it is a lifecycle policy. G4-01 removes the main trigger.
- **sentiment.get_news_sentiment**: `_try_llm_retry()` runs inside the same `try`, so an exception in the retry path returns None even though `result` was computed and cached.
- **llm_analyst._save_analysis**:
  - `load_analysis()` returning valid-JSON-but-not-a-dict raises AttributeError out of `analyze_trades` into run_cycle.
  - A non-OSError from `json.dump` leaks the tmp file.
  - Both need a hand-corrupted file.
- **llm_client `_normalize_schema_for_anthropic/_openai`** would mis-handle a *property* named `type` or `propertyOrdering`. Symbol keys never collide today, so this is latent.
- **Gemini default-model divergence in `resolve_provider_chain`**: `single`/`auto` modes default to `gemini-2.5-flash`, while backfill and free-only default to `-flash-lite`. This is routing.

## Appendix B — coverage ("no qualifying issue found beyond the above")
- **llm_client.py**, re-read after FIX_G and FIX_R3:
  - Transports fail open through their callers. Malformed or unexpected JSON (list, missing candidates, missing content, empty choices) either returns `(None, usage)` or raises into a catch-all in `call_gemini`, `call_claude`, `call_openai` and `call_llm`; `analyze_trades` wraps both entry points.
  - Every urlopen call has a timeout. The 429 waits are bounded (≤10 s primary, ≤5 s fallback, one retry).
  - Cost-ledger writes run under the thread lock plus flock. The rollover (D10) is correctly inside the flock. `_save_shared_cost`'s shared tmp name is safe because every writer holds the flock.
  - Only G4-09 qualifies.
- **llm_analyst.py**: dedup TTL clamp [0, 7000] and the veto-margin bypass are correct; replay prune; prompt sanitizer. Only G4-04, G4-06 and G4-07 qualify.
- **llm_config.py**: G4-04 (tmp) and G4-10.
- **sentiment.py**:
  - `_article_cache` eviction, TTL jitter and `_fetch_full_texts` hard deadline are correct.
  - The phrase-table invariants hold.
  - G4-03, G4-05, G4-07 and G4-09 qualify; B13 is excluded.
- **sentiment_history.py**:
  - FIX_H / FIX_R2 migration: re-check under BEGIN IMMEDIATE, verify-before-delete and the refill ratio guard are correct.
  - `stock_sentiment_lookup_dates` is correct.
  - The batch ingest key match is correct.
  - G4-01, G4-02, G4-03, G4-07 and G4-08 qualify.
- **novelty.py**:
  - Shingle math verified: 3-word crc32 shingles; the fallback for fewer than 3 words is one shingle; Jaccard is `|A∩B| / (|A|+|B|-|A∩B|)`.
  - `learned_lexicon._shingles` / `_jaccard` are exact mirrors.
  - Prune and sweep are correct. Only G4-04 (tmp) qualifies.
- **fundamentals.py**: the FIX_E sites are correct and all urlopen calls have timeouts. The remaining defects are owner-queue items (A.1) or G4-06, which lives in llm_analyst.
- **learned_lexicon.py**: offline and dormant. Checked:
  - PIT entry is strictly after the publication day;
  - folds screen on train only, with a leak guard;
  - `uniqueness_weights` diff-array;
  - ridge centering;
  - `write_json_atomic` (single writer, so the fixed tmp is fine).
  - No issues.
- **trade_memory.py**: per-writer tmp, thread lock plus flock, quarantine and never-raise are all correct. Only A.2 (tie nondeterminism), which is not proposed.
