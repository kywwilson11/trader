# ENGINE — ROUND 6 (2026-09-27 04:30–07:10) — general: Fable; workers W17–W19 (≤2 live; HW pause until 06:03, then one heavy slot)

Worker reports: <scratchpad>/generals/engine/W17_rows_quote_t_llm.md (+ addendum), W18_log_dir.md, W19_research.md. Owner items: OWNER_ITEMS.md (24).
Gate: GATE_RESULT_PLACEHOLDER
(Context: engine-r5 at 06:04 was RED 8 = 5 ENGINE [gitignored `.log` fixture absent from the snapshot — renamed .txt, 5 passed] + 3 SIGNAL; the
06:38 engine-r5b run was killed by the CEO while parked inside the suite lock by a gate.sh defect; INTEL's 06:27 snapshot caught W17 mid-edit —
the finished file is 28 passed on the current tree.)

## LANDED (each failing-before/passing-after; CHANGELOG lines R5-W17 ×2, R6-W18, R6-W19)
1. Stock buy-row key parity (W17; stock_loop.py:1387-1399): the stock row now carries the same join keys as the base row — `order_id`, `decision_bid`,
   `decision_ask`, `decision_quote_ts`, and (item 2) `decision_quote_t`; A/B 144 rows byte-identical once the new keys are stripped.
2. Scout-E ENGINE half (W17; base_loop.py:1877-1903/:1936-1961/:1984-1989): `llm_error` row on an `analyze_trades` exception (outcome, error_type,
   n_symbols_sent, latency_ms; exception re-raised unchanged); `llm_backoff` row gains outcome/n_symbols_sent/latency_ms; `llm_analysis` scores gain
   `s_defaulted` (from llm_analyst's parse_flags); the false "journaled as null" comment corrected. Readers (llm_eval, decision_report, journal_stats,
   execution_report) pinned identical. tests/test_engine_r5_rows_llm.py (28).
3. `quote_t` (W17; order_utils.py:220-226/:243): exchange quote time (epoch s) as an additive last key on the get_quote dict, computed AFTER the
   staleness verdict (None on failure) → `decision_quote_t` on both buy rows. ENGINE's own exact-dict pins modernised for the additive key only.
4. `TRADER_LOG_DIR` (W18; log_config.py:102-135 `_log_paths()`, FLAGS.md:457 + §5 counts 60/36, STATE_FILES.md:264): env override of the trader.log
   directory, read at first `_setup()`, byte-identical handler fingerprint when unset/empty; production unchanged. tests/test_engine_r6_log_dir.py (12).
   INTEL to add at conftest MODULE level: `os.environ.setdefault('TRADER_LOG_DIR', tempfile.mkdtemp(prefix='trader-test-logs-'))`.
5. Research (W19; research_engine.md § R6, 116 lines, 6 new sources): root cause of the six zero-basis positions = three separate things (below).

## VERDICTS (W19 — read-only broker history, 1,320 orders)
- SIZE (~$21k each) is OURS: an April-2026 DESYNC buy-loop of then-uncommitted code (67 "Position gone at broker" lines; $95.8k buys, 0 sells);
  63 phantom `broker_stop` exits still sit in trade_memory.json unflagged and Kelly counts them (owner #22).
- QTY DRIFT (−10.5 %…+6.2 %) is the broker's, already present 2026-04-11; no reset (account 2026-01-18), no non-fill activity.
- ZERO BASIS is the broker's asset-id split (all six on an asset_id none of our orders used; Alpaca forum Aug-2025 pattern; likely restore event
  2026-08-13 when equity printed == cash $93.63). UNVERIFIED and material: whether a sell on the current id reduces the old-id position (owner #23).
- Docs: crypto has no OTO/OCO (so `_after_entry_protection` stays), replace does not guarantee the old order is gone (no fix for O2), `qty_available`
  == 0 under full-qty resting stops (J1 confirmed), `avg_entry_price` comes back as the string "0".
- Five unhandled failure modes H1–H5 with harness tests specified (owner #24); H5 (`avg_entry_price` null through either SDK) is a class-D candidate.

## CROSS-DEPT
- SIGNAL: engine-r5 reds test_c26_W1 ×2 + test_raw_sidecar_reload_2026_09 (their landing; .git now in the snapshot per CEO). INTEL: conftest line above;
  tests/README.md rows for test_engine_r5_rows_llm / r5_live_invariants / r5_restart_backoff / r6_log_dir + fixtures (.txt/.json); journal_stats.py:13-15
  docstring ("identical key set" is still false: base has quote_age_s, stock has book_risk_pct).

## NEXT ROUND PLAN (R7, on the CEO's go; heavy work after the stock chain ends ~08:00)
R7-a H5 class-D fix (null avg_entry_price through both SDKs) + harness tests for H1–H5 (tests only for H1–H4; owner ruling first on semantics).
R7-b W16 harness gaps (fake `_quote` ns timestamps, `canceled_at`, no-model + halt startup, combined-mode startup).
R7-c #22 data repair script (dry-run only; owner approves the write). R7-d census evaluation at 06:16Z 09-28. R7-e research third: X12 paper-quirk census design → measurement-only script.
