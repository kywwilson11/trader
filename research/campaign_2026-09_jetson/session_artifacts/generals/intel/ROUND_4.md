# INTEL — ROUND 4 (2026-09-27, Jetson, HW throttle ≤2 workers) — general: Fable; workers: Opus W13/W14

## Landed
- R4-W14 research/campaign_2026-09_jetson/mac_dotenv_fix_runbook.md (new, 263 lines): Mac-side ATOMIC procedure for owner item #6 — W1's T2/T3 diffs
  verbatim (cmp-identical), `patch --dry-run` clean on the current tree, patched files byte-equal to W1's fixed copies, `comm`-verified expectation that
  exactly 7 names appear (all missing-`dotenv`: test_gpu_lock::test_choose_inference_device_always_cpu, test_new_modules::TestKelly ×3 + ::TestKellyScoping ×2,
  test_wave4::TestWarmupFill), baseline regen keeping the header, ab_check, CLAUDE.md count update from what the run prints, rollback, single-writer caveat.
  tests/README.md item 1 points at it (still OWNER DECISION). Doc drift fixed by the general: CLAUDE.md:90 said "3-line header"; the baseline file has an
  8-line header comment block.
- Scout E (spec only, appended to research_intel.md; deferred per CEO until journals/llm_replay exists): `scripts/llm_schema_reliability.py` replay instrument —
  BLOCKING FACT: failed LLM calls are never journaled (llm_analyst.py:611; base_loop.py:1795, :1843-1850) and truncation/refusal reasons only reach `print`,
  so the headline rates cannot be computed from today's journals; four journal fields requested (A1–A4 with producer file:line) BEFORE the bots restart.
  Also: base_loop.py:1827-1828's comment says a missing score is journaled as null — it never is, `_parse_response` fills 0.5 (llm_analyst.py:1125) (ENGINE comment).
- R4-W13 tests/conftest.py:52-97 suite-wide autouse LLM cost-ledger sandbox (option 1: guarded import of llm_client — 111 ms / ~11 MB, zero file/network/process
  calls at import; llm_client's only non-stdlib import is the repo's own llm_config, so it is dev-Mac-safe and degrades to a no-op if unimportable). Per test,
  `_COST_FILE` → a fresh EXISTING temp dir (a missing dir would let the fail-soft handlers at llm_client.py:287/:867 swallow writes — exactly the masking the CEO
  ruled out) + `_cost_reset_date`/`_daily_cost` reset via monkeypatch; a test's own `_COST_FILE` still wins. PROOF tests/test_intel_ledger_sandbox_2026_09.py
  (10; 7F+3E against the pre-edit conftest): real ledger calls incl. a day rollover land all four files in the sandbox and read back; root ledger mtimes/sizes
  unchanged and nothing opened for write in the root; no cross-test leak; conftest loads with llm_client/llm_config unimportable (subprocess mini-project).
  The ten previously unsandboxed files: only test_llm_batch_scoring touched the root (2 touches) → now 0; the six per-file fixtures kept (redundant, harmless).
  Side effect (good): test_llm_routing's direct ledger writes no longer leak into later tests. Not covered: child processes importing llm_client.
  tests/README.md § Conventions :177; hygiene item 3 CLOSED.

## Gate
- intel-4 (02:57): RED, 4 failures, all source-text tests on base_loop/strategy_config (test_ia2_safety ×3, test_ia4_flagged ×1) — ENGINE modified
  base_loop.py at 02:58:44 (48 s after lock) and strategy_config.py/trade_journal.py at 03:02:25; both files 67/67 in isolation afterwards.
- intel-4b (03:10): RED, 11 failures, ALL in the brand-new ENGINE file tests/test_engine_r4_journal_flatten.py (created 03:11:17, one minute after lock;
  base_loop.py/trade_journal.py modified 03:10:08); it imports no INTEL change (grep: no llm_client/conftest/ledger refs) and passes 23/23 in isolation at 03:17.
  The CEO's paper bots also started during this window (run_bots.py pid 166905) — bot runtime files now appear in the hygiene section, as expected.
- intel-4c (03:28): RED, ONE failure — again ENGINE's tests/test_engine_r4_journal_flatten.py::test_stablecoin_flatten_row_failure_never_breaks_the_branch
  (`assert {'BTC/USD','ETH/USD'} == {'BTC/USD'}` at :563); that file was modified AGAIN at 03:30:09, 74 s after the lock (03:28:55). 5398 passed otherwise.
  No INTEL coupling (the file references no llm_client/ledger/conftest symbol). The CEO's "HW: pause heavy" arrived while 4c ran, so NO further gate was
  started. **Verdict: INTEL's round-4 changes are proven per-file (all targeted test files green under hwlock, pre-edit failing runs recorded); the
  full-suite green certificate is pending a post-resume gate. Three consecutive gates went red ONLY because another department landed edits after the
  suite lock was taken — the suite lock does not freeze the tree.** PROCESS ASK for the CEO: treat `gates/.suite.lock` as a landing freeze (a department
  checks `hwlock.sh status` and defers production/test edits while a suite is running), or have gate.sh snapshot the tree into a temp worktree.
  Logs: gates/intel-4_20260927_025756.log, intel-4b_20260927_030442.log, intel-4c_20260927_031717.log.

## Owner items — OWNER_ITEMS.md refreshed; new this round
- #9 PRE-RESTART journal ask (Scout E): failed/truncated/refused LLM calls are never journaled (llm_analyst.py:611 INTEL; base_loop.py:1795, :1843-1850
  ENGINE) — four fields requested before the bots restart so the schema-reliability replay can ever be computed (measurement-only producer change; both
  departments; needs the CEO's routing).

## Cross-file notes for the CEO (not made)
- base_loop.py:1827-1828 comment claims a missing score is journaled as null; `_parse_response` always fills 0.5 (llm_analyst.py:1125) — comment drift (ENGINE).
- research/campaign_2026-09_jetson/README.md index rows (CEO said end of night): research_intel.md, llm_eprocess_params.json, mac_dotenv_fix_runbook.md.

## Flip proposals
- None.

## Held for the founder's rulings (not implemented, per CEO): llm_eprocess ρ-guard (#3), alarm/P&L token split + palette tuning (#8).

## Next round candidates (throttle-aware; all measurement/test-architecture)
1. Child-process ledger coverage: trace tests that spawn python subprocesses importing llm_client (hygiene tracer) — likely zero, prove it.
2. Journal producer fields for Scout E in llm_analyst.py:611 (INTEL half) once the CEO routes the ENGINE half — flag-free, additive JSON keys only.
3. beta_ledger robust leg (Scout B E4: Welch-winsorized beta + `alpha_mintrl_years`, report-only) — the one Scout B experiment not yet built.
4. decision_report day-cluster bootstrap CI beside the iid CI (Scout B E3, report-only).
