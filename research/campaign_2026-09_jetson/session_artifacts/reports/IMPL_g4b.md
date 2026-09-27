# IMPL_g4b — G4 second batch (llm_analyst / sentiment / sentiment_history / volatility), 2026-09-27

Owned files edited: llm_analyst.py, sentiment.py, sentiment_history.py, volatility.py,
tests/test_review_b04.py (:274 only), NEW tests/test_g4b_fixes_2026_09.py (46 tests, Mac-safe:
numpy/pandas, temp sqlite, faked yfinance / call_model / batch HTTP, no network).
No git ops, no full suite, no LLM calls, CUDA_VISIBLE_DEVICES=''.

## Changes
1. G4-06 [prompt, crash-path only] llm_analyst: `from fundamentals import _safe_float` (top-level; fundamentals
   imports only stdlib + llm_config). build_compact_evidence (:1293-1320): pe/pb/market_cap/revenue_growth/beta/
   week52_high/week52_low go through _safe_float -> non-numeric/NaN/inf omitted (same as the existing None path).
   _build_symbol_profiles: fundamentals loop `v = _safe_float(info.get(key))` (:903), plus the two other numeric
   yfinance fields formatted there, twoHundredDayAverage (:846) and targetMeanPrice (:917). Sector stays a string.
   Finite numeric output byte-identical (pinned by test_compact_evidence_numeric_output_unchanged /
   test_symbol_profile_numeric_fundamentals_formatted).
2. G4-07 llm_analyst._parse_response (:1122-1165): float(s) first; non-finite -> 0.5, i.e. the exact value of the
   existing missing/malformed-"s" path (test pins missing == NaN result), logged; then clamp. p_up non-finite -> None.
   conviction: `except OverflowError` (int(inf)) -> None, logged; int(nan) was already ValueError -> None.
   analyze_trades with conviction Infinity no longer raises (end-to-end test with stubbed call_model, persist=False).
   sentiment._parse_scores (:606-618): non-finite -> None gap (KW fallback) and NOT counted as matched (so an all-NaN
   chunk fails like an unparseable one instead of becoming +1.0s). sentiment_history.poll_and_ingest_batch
   (:1026-1033): non-finite s -> entry skipped (article stays unscored); OverflowError added for `int(entry['i'])`
   with i=Infinity. Finite values unchanged everywhere.
3. sentiment._try_llm_retry: check-then-popleft replaced by `try: popleft() except IndexError: return` (deque
   popleft is atomic, so EAFP closes the window; single-thread behaviour identical).
4. volatility._har_rrv_save: tmp = `{path}.{pid}.{thread_ident}.tmp` + os.replace; tmp removed on any failure.
5. tests/test_review_b04.py:274: now `assert sorted(p.name for p in nov_sandbox.glob('*.tmp')) == []`.

## Verification
- py_compile of all 5 edited files + new test: OK.
- New tests prove the bugs: run against copies with my edits reverse-applied (scratchpad/g4b_orig/, modules
  pre-imported ahead of the repo): 33 failed / 13 passed — every bug test fails (incl. interleaved-writer torn
  HAR file, empty-deque race, NaN batch ingest, conviction OverflowError); the 13 passing are the byte-identical
  pins plus cases the old code already handled (None fundamentals, plain empty queue). Fixed tree: 46 passed.
- Requested set (test_g4b_fixes_2026_09, test_llm_analyst, test_llm_advisor, test_llm_dossier_persist, test_c26_V1,
  test_review_b04, test_sentiment{,_gate,_history,_pit_2026_09,_scoring}, test_g4_fixes_2026_09, test_new_modules,
  test_grp_risk): **370 passed**.
- Adjacent (test_parse_scores, test_grp_sentiment, test_c26_W1, test_llm_advice): **100 passed**.
- `$JPY tests/test_sentiment_headlines.py`: GRAND TOTAL **1034/1035 (99.9%)** — unchanged.

## Notes / not done (outside scope)
- volatility._rv_save (:~570, crypto RV store) still uses a fixed `_CRYPTO_RV_FILE + '.tmp'`; lower risk (BTC-only
  writer) but the same idiom — not assigned, left as is.
- tests/test_c26_W1.py:510 asserts the retired fixed `_HAR_RRV_FILE + '.tmp'` is absent — still passes but is now
  vacuous; should glob `*.tmp` (not my file). The new test file covers the glob for _har_rrv_save.
- build_compact_evidence's snapshot `_f` lets +/-inf through (formats as "inf", no crash) — untouched.
- Tag summary: G4-06 = [prompt] on the previously-crashing path only; G4-07 = [gate] on non-finite input only
  (NaN no longer 1.5x tilt / +1.0 sentiment); sentiment_history batch = [train] only for non-finite LLM output.
