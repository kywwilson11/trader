# INTEL — ROUND 3 (2026-09-27, Jetson, HW throttle ≤2 workers) — general: Fable; workers: Opus W9/W10/W11/W12 + Scout D

## Landed
- R3-W9 tests/test_llm_{advice,client,dossier_persist,advisor,eval_v3}.py: autouse `_sandboxed_cost_ledger` fixture (advice:34, client:21, dossier_persist:25,
  advisor:45, eval_v3:33) — hygiene `MOD llm_cost.json.lock` 5/5 → 0/5, tracer root touches → 0, counts unchanged (37/24/11/65/22). llm_client.py: approved
  measurement-only rollover append `_append_cost_history` (:901-939, called :963 under the caller-held flock): one ≤200 B line
  {date, cost, src, mem_date, mem_cost, reset_at, pid} to `llm_cost_history.jsonl` (path derived from `_COST_FILE`, so sandboxes cover it); file-first cost
  (a process started after midnight had `_daily_cost=0` and used to overwrite yesterday's total silently); fail-soft; ledger JSON bytes pinned unchanged.
  docs/STATE_FILES.md:215 new row (stale :214 line refs fixed by the general → :264/:286); docs/MODULES.md:818. tests/test_intel_ledger_2026_09.py (17; 14 fail pre-edit).
  CROSS-FILE (CEO): `.gitignore` needs a line for `llm_cost_history.jsonl` (only exact `llm_cost.json` is ignored at :71). Ten further test files reach llm_client
  without sandboxing (base_loop_v3, llm_routing, review_b04, …) — some test rolled the root ledger at 02:02 CDT; after a PT midnight one would now also create the
  root history file. Candidate fix: a conftest autouse sandbox (INTEL) — next round, needs proof it cannot mask a test's intent.
- R3-W10 `llm_eprocess.py` (new, numpy/stdlib; Scout C's aGRAPA formulas cited in docstrings) + UNSIGNED parameter sheet
  research/campaign_2026-09_jetson/llm_eprocess_params.json (Scout C's 17 rows + legacy-exclusion boundary; every tunable read from the sheet) +
  tests/test_llm_eprocess_2026_09.py (18). Live mode (`--journals DIR --days N`) exits 3 unless signed_by/signed_at/registration_sha are set and the sha matches
  (unsigned and tampered sheets tested); nothing in the repo reads the verdict (grep: 0 production importers). `--selftest` (R=2000, 2.4 s, 156 MB): false-fire at
  the three null boundaries KEEP 0.4% / HARM 0.2% / FUTILITY 0.8% (target 2.5%); power at +1 bp/day 86.7% by day 180 vs 38.7% by day 90 (Scout C: 91/43 —
  direction holds). OWNER finding: with AR(1) ρ=0.3 daily noise the sequential false-fire rises to 3.5–5.4% — a sequential guard must be ruled on BEFORE the
  sheet is signed (added to OWNER_ITEMS #3). Correction to Scout C: legacy-exclusion boundary is 2026-05-07 (the last journal still has 1,213 s<0.60 skips).
  docs: MODULES row + census; STATE_FILES row for logs/llm_eprocess_report.json (general); index header corrected 119 → 121 rows (general).
  CROSS-FILE (CEO): research/campaign_2026-09_jetson/README.md does not yet index the params sheet.

## Research (Scout D, appended to research_intel.md)
- Contrast measured from token values with the repo's own `chart_core.contrast_ratio` (verified against an independent WCAG 2.x implementation): 19 pairs × 12
  themes = 228 combos; 46 fail today — chip boundaries fail 3:1 in ALL 12 themes (1.38–2.05); danger/stale red weakest in Black Metal 2.89 / Dark 3.49 /
  Two-Face 3.69 / Terminal 4.02; muted text fails in Black Metal, Two-Face, Salander, Paper; Paper fails 5 pairs (warn 2.70). Shipped as
  tests/test_intel_contrast_2026_09.py (214 pass, 46 strict xfails with the ratio in the reason; a stale-list guard; any NEW theme must pass every pair).
- ISA-101/18.2 "Ops" theme proposal (dark neutral grey, 13 tokens, all 19 pairs pass — lowest text 4.63, lowest non-text 4.40; nominal saturation ≤0.25) pinned
  in the test as `OPS_PROPOSAL`. Blocking structural fact: the 13-key token schema uses ONE `red` for alarms and P&L losses and ONE `green` for healthy and profit,
  so "alarm colours used for nothing else" needs two new tokens (`ok_nominal`, `loss`) + rerouting 10 gui.py call sites (listed with file:line; worst: the
  shadow-mode chip shows alarm yellow in its default ON state, gui.py:~8029). OWNER asks A1–A4 added to OWNER_ITEMS.
- R3-W11 chart_core.py (`alert_priority` :362 — all 11 `_push_alert` callers mapped P1/P2/P3 with a "[Pn]" text tag, WCAG 1.4.1; `AlertLedger` :387 — 10-min
  per-kind dedupe ×N, flood collapse >10/10 min into ONE expandable row that takes the P1 tag so a P1 is never hidden, ack + 30-min shelve, in-memory) +
  gui.py (`_push_alert` :3854 → `_render_alerts` :3869; callers' signatures unchanged; no new timer) + ETA column (`format_eta` :267,
  `evidence_readiness_rows(with_eta=)` :285; default 6-tuple unchanged so W8 tests pass untouched). tests/test_intel_gui3_2026_09.py (17; 17 fail pre-edit).
  Headless (Batman, Paper): 30 mixed alerts/min → 1 collapse row, expands to 13 rows whose ×N sum to 30; byte-pin held live; 0 exceptions; 289 MB peak.
  Behaviour change to note (display-only): the old "skip an identical newest alert" became the ×N counter. Not exercised headless: the right-click menu (blocks).
- R3-W12 scripts/evidence_reads.py (`--history-dir` default = parent of --out, `--eta-window` 5 at :722-741; history + ETA logic :338-457; schema fill :771-787):
  `eta_days`/`eta_basis` per readiness row from prior summary.json runs (same --days only; linear from earliest/latest of the window; null when <2 timestamps
  ≥6 h apart or no accrual); corrupt history skipped with a stderr warning; summary sha byte-identical minus the two new keys (golden pinned). +18 tests
  (17 fail pre-edit; 66/66 total). Real run: exit 0, nothing written to the repo root, every row "no history" (logs/evidence_reads/ does not exist yet — W2/W7
  ran into scratch dirs; a real linear ETA needs two runs ≥6 h apart while journals grow).

## Gate
- `hwlock.sh suite intel-3` (02:38, after W9/W10/W11/W12 + Scout D landed): **GREEN — 5351 passed, 7 skipped, 64 xfailed (46 are Scout D's strict
  contrast xfails), 0 failed/errors, 336 s** (log `scratchpad/gates/intel-3_20260927_023804.log`). Hygiene on the run: `llm_cost.json{,.lock}` NO LONGER
  appear (W9's sandboxing worked box-wide on this run); remaining `MOD sentiment_cache.db-shm` (read-only WAL opener), `MOD v2_study.db` (CEO hypersearch),
  `MOD tests/test_fill_venue_slippage_report.py` (another department's in-flight edit during the run). Post-gate edits: none to code.

## Owner items — consolidated in OWNER_ITEMS.md (8 decisions, 3 parked money asks); new this round
- #3 llm_eprocess: sequential false-fire 3.5–5.4% under AR(1) ρ=0.3 daily noise → rule on an autocorrelation guard BEFORE signing the sheet.
- #8 console contrast (46/228 failing pairs; Ops theme + two new tokens `ok_nominal`/`loss` + 10 call-site reroutes).

## Cross-file notes for the CEO (not made)
- `.gitignore`: add `llm_cost_history.jsonl` (only the exact `llm_cost.json` is ignored at :71).
- research/campaign_2026-09_jetson/README.md: index `llm_eprocess_params.json` (unsigned sheet) and `research_intel.md`.
- Ten more test files reach llm_client's ledger unsandboxed (base_loop_v3, llm_routing, review_b04, …); after a PT midnight one would create the root history
  file. Proposed: an autouse ledger sandbox in tests/conftest.py (INTEL) — round 4, with proof it cannot mask a ledger test's intent.

## Flip proposals
- None (no flags added; llm_eprocess is signature-gated measurement, not a flag).

## Next round plan (throttle-aware)
1. conftest autouse ledger sandbox (proves the ten remaining files clean) + hygiene expected-clean on a quiet box.
2. If the owner rules on the ρ-guard: implement it in llm_eprocess (params-driven) + left-skew/TF sweep sims (Scout C's plan).
3. Console: palette tuning for the 46 failing pairs is an owner call; INTEL can land the P&L/alarm token split behind the existing theme table if A2 is approved.
4. Scout E: LLM structured-output reliability across providers (schema drift, refusal/cutoff rates) as an offline replay instrument once llm_replay journals exist.
5. Prepare the Mac-side atomic dotenv-leak fix + baseline regen instructions (owner item #6).
