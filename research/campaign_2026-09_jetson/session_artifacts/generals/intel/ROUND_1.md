# INTEL — ROUND 1 (2026-09-27, Jetson) — general: Fable; workers: 6 Opus (3 implementers, 1 analyst, 2 scouts)

## Landed (all measurement/robustness/console; nothing model-facing; every change has failing-before/passing-after tests)
- R1-W2 `scripts/evidence_reads.py` (new, stdlib-only) + `tests/test_evidence_reads_2026_09.py` (47 tests). One command runs the runbook
  Phase 0 §5 / Phase 1 reads sequentially (per-step timeout, capture, peak RSS) into `logs/evidence_reads/<ts>/` (.gitignore:31) and prints a
  readiness table from `READINESS_RULES` (evidence_reads.py:95). Only the LLM rule has a documented source (runbook 03:48; llm_eval.py:73-77);
  the rest are labelled "provisional, evidence_reads default". Real run tonight: beta READY (87 obs, but a flat idle curve), every journal read
  NO DATA at 30 d; at 200 d decision_report NOT YET (priced 2 < 30) = IMPL_measure exactly. Docs: scripts/README.md + docs/MODULES.md rows,
  plus the three IMPL_measure items (train_lexicon --session-tz, rank_gradient_report exit 2, sizing_cofire n_buy_rows_without_sizing).
- R1-W3 `gui.py` + `tests/test_intel_gui_2026_09.py` (25 tests; 13F+8E against the pre-edit file). Cockpit (gui.py:3807) and Trading
  (:4305-4309, index via indexOf(page) — indexOf(tab) would return −1 and kill the lazy first paint at :6296) now scroll via `_scroll_wrap`;
  Trading at 1280×800 shows 2.7/5.3 rows of orders/fills instead of 1/1; 8 tabs × 2 themes × 2 sizes, 0 uncaught exceptions before/after.
  `optuna.load_study` moved off the UI thread (daemon Thread + Signal, :488-573, :10400-10432; same (path,mtime,size) cache key): refresh
  inside window build 1467–1572 ms → 35 ms, window build 3.8 s → 2.4 s, cold refresh ~165 → ~26 ms.
- R1-W1 `tests/conftest.py` (:79-197) + `scripts/ab_check.sh` (:17-23, :121-142, :196) + `tests/README.md` + `tests/test_intel_testarch_2026_09.py`
  (12 tests; 12/12 fail against the pre-edit copies). Repo-root hygiene instrument: `== repo-root hygiene ==` lists NEW/MOD files in the repo root
  and tests/ per session; report-only by default, `TRADER_TESTS_STRICT_CLEAN=1` forces exit 1 only when the run would otherwise pass,
  `TRADER_TESTS_HYGIENE_JSON` dumps it; empty allowlist (an allowlisted file would still print as ALW). First finding: `tests/test_g4b_fixes_2026_09.py:181`
  → `llm_analyst.py:579` → `llm_client.py:782/:901/:286` writes `<repo>/llm_cost.json.lock` (README hygiene item 3 re-opened; fix = monkeypatch
  `llm_client._COST_FILE` in that test — INTEL test file, next round). README: family `2026_09` inventory (collected counts), dotenv-leak note corrected.

## Gate
- `hwlock.sh suite intel-1` (00:42, after all three implementers landed): **GREEN — 4759 passed, 7 skipped, 18 xfailed, 0 failed/errors, 237 s**
  (log `scratchpad/gates/intel-1_20260927_004248.log`). Hygiene section on that run: `MOD tests/README.md` (this general's edit during the run),
  `MOD v2_study.db` (the CEO's hypersearch pid 82746), `MOD sentiment_cache.db-shm` (a concurrent read-only opener — WAL readers touch -shm; unverified which agent).
  `llm_cost.json.lock` did NOT show on this full run although W1's single-file run flagged it (mtime-only detection of a truncate-to-same-size write is not
  guaranteed on ext4 — unverified; the JSON/`ALW` path is unaffected). Instrument is report-only; nothing forced.

## Research (2 scouts = 1/3 of capacity; briefs appended verbatim to research/campaign_2026-09_jetson/research_intel.md; adjudication below)
- Scout A (LLM-as-analyst + sentiment; 24 sources, $0 spent). Verified claims: analyst cadence is 600 s not hourly (base_loop.py:94 `LLM_INTERVAL_SEC=600`);
  the prompt includes the ML pred by default (llm_analyst.py:448 `include_pred=True`); cap `_DAILY_COST_LIMIT=1.00` is at llm_client.py:261 (G_llm's :175 is stale);
  no FR-20c cutoff guard in llm_eval.py (grep: 0 hits). Measured: sentiment_cache.db = 105,753 articles through 2026-09-26; U+2019 hits ~31.5% of negated-
  contraction headlines / ~45% of summaries; a FinBERT-class encoder costs ~134 ms/headline on 2 ARM threads (fits) but 2026 evidence gives 1-day rank IC ≤0.014,
  so no build; a local LLM judge is infeasible on the Orin CPU (4–13 tok/s ⇒ ~6 min to read a 5k-token prompt). Free tiers cannot carry 288 calls/day
  (Groq ~40 calls/day; OpenRouter 50 RPD, 1,000 after a one-time ≥$10 credit). ACCEPTED experiments: E3 `scripts/lexicon_fix_impact.py` ($0, offline,
  rule: bundle the fix iff ticker-day aggregate moves >0.05 on ≥1% of non-zero ticker-days) and E4 train_lexicon go/no-go (rule: 1-day IC 95% CI lower
  bound >0.01 on ≥5,000 ticker-days, else NEGATIVE and no encoder build). E1 (test-retest noise, ~$0.20) and E2 (hide-pred A/B, ~$0.55) are OWNER asks
  (spend outside the ledger) — not run.
- Scout B (small-sample statistics + console UX; 25 sources). KEY MEASURED FINDING (synthetic, R=500/cell, K=6, fb=24, nominal 5%): llm_eval's production
  Driscoll-Kraay verdict (Bartlett lag 23, t_{G−1}; llm_eval.py:532-537) rejects a TRUE null 12.4–13.8% at n_eff 20/30/60 and does not improve with T;
  the existing report-only Ibragimov-Müller K=8 cross-check (llm_eval.py:333-369) is the only estimator on nominal (4.0–6.2%). Verified: validation.py:647-649
  returns block length 1.0 (iid) below 20 obs; shadow.py:36-41 documents the ~14 unadjusted daily looks with DM v2 behind a default-OFF flag. ACCEPTED for
  round 2 (measurement-only, INTEL files): E1 `scripts/har_size_audit.py` + report-only `b2_im_p`/fixed-b fields in llm_eval (verdict carrier change =
  owner ask #1, pre-registered read in runbook 03:48); E3 day-cluster bootstrap CI beside decision_report's iid CI (decision_report.py:76, 398-420);
  E4 Welch-winsorized beta + `alpha_mintrl_years` in beta_ledger (report-only). E2 `llm_eprocess.py` (anytime-valid spend ledger) needs its parameters frozen
  by the owner BEFORE post-retrain journals accrue (ask #2). Console UX ranked list (kill-switch consequences read-back; evidence-readiness panel fed by
  evidence_reads; content-aware freshness; alarm hierarchy; structured veto reasons; ISA-101 theme + contrast test) — UX #1 depends on an ENGINE decision
  (halt is entries-only and fails open, base_loop.py:2922-2932 — reported, not touched). Not measured under the HW throttle: decision_report iid-CI
  false-verdict rate (exact command recorded in SCOUT_B.md F5).

## Owner items (written up with numbers in scratchpad/generals/intel/w/W5_owner.md; nothing shipped)
- B13 + G4-05 (sentiment._score_text): ONE fix behind default-OFF `KW_SCORER_V2` (apostrophe normalisation + "mask2" = mask the scored phrase and the
  negator phase-1 consumed, keep negators inside phrases), shipped only inside the Phase-3 rebuild with a `--rescore-keywords` step. Measured read-only on
  the live DB (105,753 articles): stored keyword_score == today's recompute on every row and daily_sentiment on all 9,989 cells (OFF path reproduces
  exactly; rollback clean); the fix changes 12,570 article scores (11.9%), flips 372 signs, moves 2.46% of ticker-day cells by >0.05 and 0.25% by >0.2
  (last 365 d); headline runner stays 1034/1035 (plain masking / the 02_research blank-span design drop to 1014). Full diff: w/W5/b13_g405_proposed.diff.
  DECISION: approve KW_SCORER_V2 + rescore in the Phase-3 bundle (gotcha #2 event already scheduled).
- D13 veto strikes (ENGINE file — reported, not touched): re-verified at base_loop.py:1686/:1706/:1709/:1719-1728/:2027; repro sells after veto → omitted
  cycle → veto. Scope A one-liner (`for sym in self._last_llm_symbols.difference(new_scores): self._veto_strikes.pop(sym, None)`) restores the documented
  "two CONSECUTIVE vetoes" rule — indisputable, route to ENGINE; scope B (clear strikes when a stock leaves the top-N) overlaps the owner note at :1558 — rule.
- Free-first LLM qualification (runbook Phase 4): NO-GO by construction today — no groq/openrouter keys, ollama not running, `journals/llm_replay` absent,
  scorecard readiness 0/120 clusters (zero llm_analysis journal rows); groq/openrouter/ollama lack budget+pricing rows and gemini-3.x is absent from
  GEMINI_MODELS/KNOWN_MODELS. `--help`/`--report` verified at $0. Exact command sequence + go/no-go rule in W5_owner.md ITEM 3. DECISIONS: Scout A's
  A1 (~$0.75 for E1+E2, outside the ledger), A2 ($10 OpenRouter credit or drop it), A3 (analyst sub-budget before any paid 3.x/Haiku analyst), A4 (exclude
  in-window llm_score rows from training aggregates at the re-harvest); Scout B's #1 (IM as verdict carrier), #2 (freeze llm_eprocess params now), #3
  (halt should cancel working orders — ENGINE), #4 (beta-only ledger policy), #5 (shadow anytime-valid monitor vs DM v2).
- dotenv stub leak (tests/test_c26_T2.py:24, T3.py:28): masks 7 names on the Mac, not 5; fix diffs ready in w/W1/fix/{T2,T3}.diff — must be applied
  atomically with a Mac baseline regeneration (adds ~7 baseline names). Owner item as tests/README.md records.
- Cross-department reports: the four root-writing report CLIs (decision_report.py:1036, execution_report.py:67, llm_eval.py:1180/:1495) have no
  output-path flag — INTEL files, will add `--out` next round if the CEO agrees; `_projected_columns` exists as 3 copies pending a data_utils helper (SIGNAL).
  gui: positions cells still elide at ~68 px/col and Batman→Paper leaves dark text until the next update (pre-existing) — round 2 candidates.

## Flip proposals
- None this round (no flag added). KW_SCORER_V2 is proposed above as a Phase-3-bundled flag, not yet written.

## Next round plan (under the HW throttle: ≤2 workers, research/writing first)
1. llm_eval verdict honesty (Scout B E1): `scripts/har_size_audit.py` + report-only `b2_im_p` / fixed-b fields; failing-before test on synthetic nulls.
2. `--out` flags for the four root-writing report CLIs + monkeypatch `_COST_FILE` in tests/test_g4b_fixes_2026_09.py:181 (hygiene item 3).
3. Scout A E3 `scripts/lexicon_fix_impact.py` ($0) reusing W5's variant scorers, so the KW_SCORER_V2 decision has a rerunnable instrument.
4. Console: evidence-readiness panel (Models tab, fed by evidence_reads summary.json) + content-aware freshness (UX #2/#3); positions cell formatting.
5. Scout C (research third): sequential/anytime-valid evaluation design for the shadow + LLM ledger (E2 parameter freeze proposal for the owner).
