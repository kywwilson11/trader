# INTEL — ROUND 5 (2026-09-27, Jetson; started inside the CEO's "HW: pause heavy") — general: Fable; workers: Opus W15/W16/W17

## Status legend: IMPLEMENTED-UNVERIFIED = code + tests written, py_compile clean, tests skip-marked "intel R5 pending verification"; verification runs after "HW: resume".

## Implemented (unverified until resume)
- W16 beta_ledger.py:385-520 (`welch_winsorize`, `half_life_weights`, `wls_slope`, `robust_betas`, `alpha_mintrl_years`, `_add_robust_leg`; called last in
  `beta_report` :740): per benchmark `beta_ols_univariate`, `beta_winsor` (Welch 2022, δ=3 ⇒ clip to [−2,+4]× the benchmark return — the paper's constant, one
  `WELCH_DELTA`), `beta_winsor_wls` (120-obs half-life), `beta_stable` (|β_OLS−β_winsor|<0.15 AND n≥60, Scout B E4 pre-registration) + `winsor` detail;
  `strategy.alpha_mintrl_years` = (2/|SR|)² with `alpha_estimable`. New printed lines only AFTER every existing line; no existing key/value/order changed
  (golden test pending). (d) AKL winsorized leg declined with reason (would clip the lagged response the sum measures). Stdlib sim (2000 seeds): winsor
  move ≤0.13× the OLS move under an injected 40% print; verdict flips every time. Owner caveat: Welch's band shrinks toward β=1, so a low-beta, high-idio book
  can read UNSTABLE for that reason — read `n_clipped` beside the verdict. tests/test_intel_beta_2026_09.py (12, module-skipped; golden digests None until
  computed from the pre-edit module on resume). docs/MODULES.md beta_ledger section updated.
- W15 llm_analyst.py (additive; 2 pre-edit lines touched: `_journal_replay` signature + one call passing `call_meta=`): every non-dedup `analyze_trades`
  attempt — failures included — now writes one `action:"llm_call"` row to `journals/llm_calls/<date>.jsonl` (NEW journal, deliberately NOT llm_replay/ because
  prompt_ab/llm_qualify read every llm_replay line as a scored cycle); outcomes ok/partial/parse_fail/not_object/empty/transport_discard/transport_error +
  SCOUT_E A1 fields + defaulted/non-finite/out-of-range counts; the replay record gains per-symbol `parse_flags`, `prompt_sha256`, `latency_ms`, `dedup_hit`,
  `fence_stripped` AFTER the unchanged legacy keys; same persist/replay_capture gates (persist=False writes nothing). ENGINE meeting point:
  `get_last_analysis_meta()['parse_flags'][sym]['s_defaulted']`. A2 half-done: llm_client exposes no last-call meta, so finish/block reason are null until an
  llm_client accessor exists (INTEL, next round). PRE-EXISTING BUG found (not fixed — behaviour change): `_parse_response` raises OverflowError on a
  huge-integer "s"/"p_up", escaping `analyze_trades`' fail-open contract (smoke-confirmed) → owner/next-round fix candidate (class A). Static child-process
  census: 13 spawning test groups, none reaches the real ledger (dynamic tracer post-resume). tests/test_intel_llm_journal_2026_09.py (20 fns/23 cases,
  module-skipped). STATE_FILES §5 + MODULES row updated.
- W17 decision_report.py:424-586 (`_day_cluster_ci`, `_row_day`, `_dayclust_fields`, `verdict_disagreement`, `E3_RULE`): beside every `ci90` (gates, signal-exit
  audit, conviction buckets) additive `ci90_dayclust` (calendar-day cluster bootstrap, same day bucketing as `_dedup_first_per_day` :283, same n_boot/alpha/seed),
  `n_days`, `dayclust_few_days`, `ci90_dayclust_reason` when null; `verdict_dayclust` from the SAME verdict function; report-level `verdict_disagreement_rate`
  + one new printed line carrying the pre-registered E3 rule. No existing key renamed/moved (gui/chart_core/evidence_reads read only old keys). tests/
  test_intel_decision_ci_2026_09.py (13: 7 pure-logic active, 6 skip-marked incl. golden). docs/MODULES.md:718 row extended.
- W19 llm_client.py:243-345 `get_last_call_meta()` (thread-local copy of {provider, model, finish_reason, block_reason, http_status, latency_ms, attempt_index,
  fallback_used, ts}; recorded per HTTP attempt in call_gemini/claude/openai/call_llm incl. 429 re-sends and the fallback chain; parsers get read-only `_note_*`
  hooks; only 2 `except: pass` blocks altered to record meta) → W15's llm_calls rows now fill finish/block reason + http_status so `transport_discard` can fire.
  Class-A crash-path fix: `_parse_response` catches OverflowError (huge-integer "s"/"p_up" JSON) → s via the existing non-finite path to 0.5, p_up None;
  `_parse_diagnostics` (:1534) sets `s_defaulted` on overflow; finite inputs byte-identical (stdlib smoke on pre/post copies: identical on all 7 stubbed
  transports). tests/test_intel_callmeta_2026_09.py (33 fns/35 cases, module-skipped). CROSS-TEST NOTE: W15's test file will be order-dependent (meta left by
  real-client tests leaks into stubbed runs) → fix at verification time by clearing the thread-local meta in the conftest ledger sandbox (INTEL; W18/W21 scope).
- W18 tests/conftest.py:6-28 — `TRADER_LOG_DIR` set at module level right after `import os`, before every repo import (`setdefault`; empty caller value treated
  as unset because log_config reads '' as production; source recorded on config); production-log guard (inode/mtime/size at sessionstart → "production log
  untouched: yes / no / unattributable" — the bots pids 166905/166986/169981 hold the file open and it grew 144 B in 20 s, so the guard attributes changes via
  /proc before blaming the test process); W13 sandbox also resets llm_client's `_call_meta_tls` per test (W19 order-dependence closed). tests/
  test_intel_logdir_2026_09.py (11; incl. two subprocess mini-projects proving the env precedes a logger-at-import module). Docs: tests/README item 2 fixed,
  STATE_FILES §9/§10 → log_config.py:22-25/46-99/102-135/162-168, MAP.md:810 + row 16. General: scripts/ab_check.sh awk vocabulary extended for the two new lines.

## Landed (docs, no verification needed)
- W20 docs/FLAGS.md verified cite pass (CEO-routed from SIGNAL/ENGINE): 454 line-cite spans, ~340 stale → 189 renumbered (the file's Philosophy #6 / §7 recipe
  keep line numbers in Defined / First-read-at / §6 Where / §4 ranges; incl. HYPERSEARCH_V3→231 … TRAINING_REPAIRS_V1→338, the +124 block, EOD digest 118→144,
  TRADER_SHADOW_MODE 1040→1425), 122 converted to durable function/constant anchors, 143 already correct; final checker: 0 stale, 121/121 anchors resolve.
  Only cites + the header note changed. ROWS PROVEN WRONG (reported, not fixed — facts, not cites): UNIQUENESS_WEIGHTS_ENABLED says "LSTM loss" but its only
  reader is `hypersearch_v2 train_lgb_ensemble()` (LightGBM leg); TRADER_PYBIN "never read at runtime" but scripts/backup_state.sh:26 reads it;
  TRADER_HOLDOUT_SPAN_BY_TARGET and TRADER_BREAKER_SERVER_FILL_ATTRIB lack §5 rows (the "36 TRADER_*" count is off); TRADER_TESTS_STRICT_CLEAN has no row
  (INTEL — will add next round). Checker + pre-edit copies in w/W20/.

## Verified after "HW: resume" (06:03)
- W16 VERIFIED: 20F/1S against the pre-edit module → 21 passed; pre-edit digests of existing report keys (3fbef712…) and printed text (f15af2b1…) reproduced
  by the edited module and pinned; test_beta_ledger 9, _v3 29, c26_U1 31, measurement_fixes 30, evidence_reads 66 all green. Real `--days 90`: 89 days
  (2026-08-13 glitch dropped), SR 1.42, alpha +22%/yr at t +0.66 → "alpha not estimable in this horizon (MinTRL 2.0 y > window 0.4 y)"; SPY β OLS .292 /
  winsor .384 / WLS .411 (54/88 days clipped — Welch's band is centred on β=1, so STABLE there is partly the pull toward 1); BTC .443 / .470 / .469
  (30/89 clipped), STABLE. OWNER LOOK: BTC summed lagged beta +1.18 vs same-day +0.25 — read before any hedge is sized.
- W17 VERIFIED: pre-edit copy → ImportError at collection; golden stdout/JSON from the pre-edit module reproduced by the edited one (minus the one new line /
  new keys) and pinned; 13/13 (one-row-per-day == iid CI exactly; iid width ratio within [0.75,1.33]; same-day correlation ratio >1.3); test_decision_report 19,
  _v3 30, g6_fixes 29 green. Real `--days 200 --out`: 11,984 rows, 2 priced → disagreement rate n/a as expected; root decision_report.json NOT written.
  NOTE for the CEO/owner: ~12k legacy rows carry the score inside skip_reason (`llm_below_buy_min (0.58<0.60)`) → `_unclassified_skip_reasons`; the
  producer no longer exists (removed s<0.60 gate) — those rows must be excluded (Scout C boundary 2026-05-07), not parsed.
- W18 VERIFIED: logdir 10, testarch 12, ledger_sandbox 10, llm_fixes 96, engine_r6_log_dir 12 all green; guard verdicts yes/unattributable/yes/yes/yes (the
  unattributable run: live bot pids held the log open; the +729 B raw delta was read byte-by-byte — all bot cycle lines); `grep -c intel-w18-` = 0 in
  trader.log/.1, marker present in /tmp/trader-test-logs-*/trader.log; pre-edit proof 9F/1P (the pass only sets state). ab_check vocabulary already extended by
  the general (the worker's cross-file note is closed).
- W15 VERIFIED: pre-edit harness 19F/5P (the 5 are pre-edit invariants) → 24 passed; test_llm_analyst 28, llm_advice 37, llm_advisor 65, dossier_persist 11,
  c26_P1 44, g4b 46 green; no journals/llm_calls/ created in the real tree; dynamic tracer over 7 child-spawning files (216 tests): all 14 trace events from
  the parent pytest processes, no child imported llm_client, llm_cost.json mtime/size identical — child-process ledger coverage question CLOSED (round-4 item).
  Note: the live run_bots.py (pid 166905, started 03:09) runs the pre-edit llm_analyst until its next restart.
- W19 VERIFIED: 35 passed (2 first-run failures were test bugs — per-symbol flags live on the replay record, not the llm_call row — test-only fix); pre-edit
  proof 32F/3P (20 AttributeErrors = no accessor; the OverflowError e2e crash reproduced; the 3 passes are the byte-pin goldens); test_llm_client 24, llm_fixes 96,
  g4b 46, llm_analyst 28, llm_providers 20, llm_claude 13, intel_llm_journal 24 (last) green; hygiene clean every run.

## Gate
- (filled after `hwlock.sh suite intel-4cert`)
