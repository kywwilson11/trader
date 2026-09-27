# INTEL — ROUND 2 (2026-09-27, Jetson, under HW throttle ≤2 workers) — general: Fable; workers: Opus W6/W7/W8 + Scout C

## Research (Scout C, appended to research/campaign_2026-09_jetson/research_intel.md; adjudicated)
- Design A `llm_eprocess.py` (measurement-only, NOT built yet): daily mark-to-market value of the LLM size-tilt vs a multiplier frozen from the first 10 days,
  net of fees + LLM cost in bp of equity; three one-sided betting tests (KEEP / KILL-HARM / KILL-FUTILITY at δ=1 bp/day), e ≥ 40 (α .025) + a .025 fixed-N
  fallback; first decision at 10 test days & ≥30 lots; horizon 180 d (sim: 90 d gives KEEP power 43% at 1 bp/day, 180 d gives 91%; false-fire ≤1.7% at every
  null boundary, R=300–400). Corrects Scout B F4: daily blocks are disjoint only when marked to market daily. Verified code facts: llm_cost.json holds today only
  (llm_client.py:861 overwrite, :879-887 LA-midnight reset) → cost bracketed [journaled cost_usd, $1 cap]; cost ≤0.1 bp/day vs ~2.8 bp daily noise; Apr–May
  journals must be excluded (11,815/11,818 skips came from the removed `llm_below_buy_min (s<0.60)` gate; live code has only the s<0.15 veto, base_loop.py:3170-3178).
- Design B: recommend DM v2's two scheduled looks, NO shadow anytime monitor (at h=24 a stopped e-value is known ~23 h late; decision would take 60–150 d vs
  28/56-d windows; the challenger is silent, shadow.py:14-15). 17-row pre-registration sheet for the owner's signature is in SCOUT_C.md § Pre-registration.
  OWNER asks: sign the sheet before post-retrain journals accrue; choose the inconclusive default and the capital basis; cross-file measurement-only ask —
  append each day's cost to `llm_cost_history.jsonl` at rollover (llm_client.py:884-885; INTEL file — will do next round if approved); DM v2 flip per runbook.

## Landed
- R2-W6 llm_eval.py (:390-480 helpers, :676-681 wiring, :969 `_report_path`, :1277 printed size-audit line, :1660 `--out`) + `scripts/har_size_audit.py` +
  `tests/test_intel_llm_eval_2026_09.py` (10; 9F/1P against the pre-edit copy; golden report byte-identical minus the 5 new keys). Verdict carrier unchanged
  (llm_eval.py:657 still DK/t_{G−1}). Size audit on this device (K=6, fb=24, AR(1) .9, R=500, 121 s, 83 MB), rejection of a TRUE null at nominal 5%
  for n_eff 20/30/60: DK 13.0/11.0/9.6% · DK fixed-b 9.0/8.2/6.4% · EWC 10.6/8.0/6.2% · IM K=8 5.6/6.2/5.2% — under the pre-registered [3%,7%] rule only
  IM is fit to carry the keep/kill verdict (OWNER ask #1 stands; nothing flipped). The n_eff=20 cell reproduces Scout B exactly (same seed/draw order).
- R2-W7 decision_report.py (`_report_path` :804; stale :818/:826 + normal :1051 writes honour `--out`), execution_report.py (:84, `--out` :280), default root
  paths unchanged (gui.py:365/:4330 read them); scripts/evidence_reads.py now passes `--out <out>/<step>` to the five root-writing steps — real run:
  `root_files_written == []`, `find -newer` shows none of the four JSONs in the root. tests/test_g4b_fixes_2026_09.py:181 fixture sandboxes
  `llm_client._COST_FILE` (hygiene: `MOD llm_cost.json.lock` → clean; no assertion changed). NEW `scripts/lexicon_fix_impact.py` (read-only DB, 46 s,
  55 MB): reproduces W5 exactly (12,570 rows changed = 11.89%, 372 sign flips, 2.46%/0.25% of ticker-day cells moved >0.05/>0.2, runner 1034/1035 for every
  variant); pre-registered E3 verdict = **BUNDLE** — the phrase-mask fix alone moves 2.72% of non-zero ticker-days >0.05 (apostrophe alone 0.69%, under the
  1% bar). Tests: tests/test_intel_reports_2026_09.py (21; 8 fail pre-edit), test_evidence_reads_2026_09.py (48; 4 fail against the old wrapper).
  Not sandboxed (reported, not edited): test_llm_advice / test_llm_client / test_llm_dossier_persist / test_llm_advisor / test_llm_eval_v3 still touch the ledger.
- R2-W8 chart_core.py (`artifact_validity` :111 with producer flags cited at :77-99; `freshness_state` :184) + gui.py (`_refresh_reports_freshness` :10382,
  `_report_validity` :10424 stat-gated re-read; readiness panel on the Models tab, refreshed only by the existing 60 s timer :2974 → :11053) +
  tests/test_intel_gui2_2026_09.py (28; 28 fail against the pre-edit copies). Headless: fresh-looking no_data/stale stubs 3/3 → 0/3 false-fresh, 2 themes,
  0 exceptions, 307 MB peak. Two judgement calls flagged (representative=false ⇒ VOID; execution report with buys but no slippage fills ⇒ VOID) — both are
  display-only. docs/MODULES.md chart_core row updated by the general.

## Gate
- `hwlock.sh suite intel-2` (01:58, after W6/W7/W8 landed): **GREEN — 4946 passed, 7 skipped, 19 xfailed, 0 failed/errors, 302 s**
  (log `scratchpad/gates/intel-2_20260927_015518.log`). Hygiene on that run: `MOD llm_cost.json` + `llm_cost.json.lock` (the five un-sandboxed
  ledger-touching test files listed under owner items — next round), `MOD sentiment_cache.db-shm` (read-only WAL opener), `MOD v2_study.db` (CEO hypersearch).
  Post-gate edits: none to code (only this report).

## Owner items (new this round)
- IM as the keep/kill verdict carrier (ask #1): now backed by an in-repo instrument (`scripts/har_size_audit.py`) — DK over-rejects 2–2.6×; IM is the only
  estimator within [3%,7%]. Changing the carrier amends the pre-registered read at runbook 03:48 → owner decision; nothing flipped.
- KW_SCORER_V2: the rerunnable instrument now says BUNDLE (mask fix 2.72% of non-zero ticker-days >0.05). Still a default-OFF proposal for the Phase-3 bundle.
- llm_eprocess pre-registration sheet (SCOUT_C.md § Pre-registration, 17 rows) to sign BEFORE post-retrain journals accrue; plus the measurement-only ask to
  append daily cost to `llm_cost_history.jsonl` at rollover (llm_client.py:884-885 — INTEL file; not done pending approval since it adds a runtime file).
- Cost-ledger sandboxing still missing in test_llm_advice / test_llm_client / test_llm_dossier_persist / test_llm_advisor / test_llm_eval_v3 (INTEL tests; next round)
  and the GUI's LLM-usage display touches llm_cost.json.lock (gui.py ~11007) — expected, it reads the shared ledger under the file lock.

## Flip proposals
- None (no new flags this round).

## Next round plan (throttle-aware)
1. Sandbox the five remaining ledger-touching test files; hygiene should then read clean on a quiet box.
2. Build `llm_eprocess.py` skeleton (measurement-only, parameters read from a frozen JSON the owner signs) with synthetic tests — no live read until signed.
3. Console UX #1/#4 (kill-switch consequence read-back needs ENGINE's halt ruling; alarm hierarchy on the Cockpit feed is INTEL-only — do #4).
4. Scout D: operator-console accessibility contrast test across the 12 themes (design_tokens) + ISA-101 "Ops" theme feasibility.
5. If approved: `llm_cost_history.jsonl` rollover append + evidence_reads readiness ETA from accrual history.
