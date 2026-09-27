# INTEL — consolidated owner items (kept current each round; lift verbatim into the morning brief). Updated: FINAL — wind-down, 2026-09-27 ~07:10 (rounds 4+5 certified-pending the intel-5cert snapshot gate)

## Decisions requested (nothing below is shipped; all evidence is in scratchpad/generals/intel/w/ and research/campaign_2026-09_jetson/research_intel.md)
1. **IM as the keep/kill-LLM-spend verdict carrier** (llm_eval.py:657 still uses Driscoll-Kraay/t_{G−1}). Instrument `scripts/har_size_audit.py` (this box,
   K=6, fb=24, R=500): the current test rejects a TRUE null 13.0/11.0/9.6% at n_eff 20/30/60 for nominal 5%; Ibragimov-Müller K=8 gives 5.6/6.2/5.2% — the only
   estimator inside the pre-registered [3%,7%] band. Both p-values are now printed side by side (report-only). Switching the carrier amends the read
   pre-registered at 03_jetson_runbook.md:48. ASK: approve IM as carrier (DK printed alongside) before post-retrain journals accrue.
2. **KW_SCORER_V2** (sentiment._score_text: U+2019/U+2018 apostrophe normalisation + "mask2" phrase masking) as ONE default-OFF flag bundled into the Phase-3
   rebuild with a `--rescore-keywords` step. Live DB (105,753 articles, read-only): stored scores == today's recompute on every row and all 9,989 daily cells
   (OFF path exact, rollback clean); the fix changes 12,570 article scores (11.9%), flips 372 signs, moves 2.46% of ticker-day cells >0.05 (0.25% >0.2);
   headline runner stays 1034/1035. Rerunnable instrument `scripts/lexicon_fix_impact.py` → pre-registered E3 verdict **BUNDLE** (mask fix alone moves 2.72%
   of non-zero ticker-days >0.05; apostrophe alone 0.69%). Diff: w/W5/b13_g405_proposed.diff. ASK: approve for the Phase-3 bundle (gotcha-#2 event already scheduled).
3. **llm_eprocess pre-registration sheet** (SCOUT_C.md § Pre-registration, 17 rows): daily MTM value of the LLM size-tilt vs a multiplier frozen from the first
   10 days, net of fees + cost, bp of equity; three one-sided betting tests (KEEP / KILL-HARM / KILL-FUTILITY, δ=1 bp/day), e ≥ 40, first decision at 10 test days
   & ≥30 lots, horizon 180 d (90 d ⇒ KEEP power 43%; 180 d ⇒ 91%; false-fire ≤1.7%). ASKS: sign the sheet BEFORE post-retrain journals accrue; choose the
   inconclusive default; choose the capital basis (paper P&L vs real LLM spend). Round 3 built the skeleton (`llm_eprocess.py`, `--selftest` only; live mode exits 3
   until the sheet at research/campaign_2026-09_jetson/llm_eprocess_params.json is signed and sha-matched). Selftest R=2000: false-fire ≤0.8% at every null
   boundary, power 86.7% at +1 bp/day by day 180 (38.7% by day 90). NEW FINDING: with AR(1) ρ=0.3 daily noise the sequential false-fire is 3.5–5.4% vs the 2.5%
   target — ASK: rule on a sequential autocorrelation guard (e.g. block-aggregate d_t or a ρ-triggered α haircut) BEFORE signing. Boundary correction: exclude
   journals through 2026-05-07 (not 05-06).
4. **D13 veto strikes** — ROUTED TO ENGINE by the CEO (scope A one-liner indisputable; scope B = clearing strikes when a stock leaves the top-N needs a ruling vs
   the owner note at base_loop.py:1558).
5. **Halt should also cancel working entry orders** (base_loop.py:2922-2932 fails open; FIA §1.5) — ROUTED TO ENGINE.
6. **dotenv stub leak** (tests/test_c26_T2.py:24, T3.py:28) masks 7 Mac-baseline names. Copy-pasteable atomic Mac procedure now at
   research/campaign_2026-09_jetson/mac_dotenv_fix_runbook.md (diffs verbatim, baseline regen, ab_check, CLAUDE.md count, rollback). ASK: run it on the Mac (single-writer tree).
7. **In-window `llm_score` rows** (Scout A A4): exclude article rows scored by a model whose training cutoff post-dates the article (cutoff − 8 months margin)
   from `daily_sentiment` training aggregates at the next re-harvest — a look-ahead channel into `Daily_Sentiment`. ASK: rule before the Phase-3 harvest.
   (Round 5 note: SCOUT_E A1–A4 journal fields are LANDED on the INTEL side — journals/llm_calls/<date>.jsonl + call meta; the running bot picks them up at its next restart.)
8. **Console contrast + Ops theme (Scout D, tests/test_intel_contrast_2026_09.py):** 46 of 228 (theme, pair) combos fail WCAG 2.x today — chip boundaries
   fail 3:1 in ALL 12 themes; danger red 2.89 (Black Metal) / 3.49 (Dark) / 3.69 (Two-Face) / 4.02 (Terminal); Paper fails 5 pairs. ASKS: A1 palette tuning
   for the failing pairs (the test's strict xfails flip to failures when fixed, forcing mark removal); A2 two new tokens (`ok_nominal`, `loss`) so alarm
   red/yellow are used for nothing else (10 gui.py call sites rerouted; shadow-mode chip shows alarm yellow in its default ON state); A3 Ops theme as dark
   grey (proposed, all pairs pass) or classic ISA light grey; A4 non-colour cue on the alert feed (round 3 W11 adds "[Pn]" text tags — check its report).
9. **PRE-RESTART journal fields (Scout E spec, research_intel.md § Scout E):** failed / truncated / refused LLM calls are never journaled
   (llm_analyst.py:611; base_loop.py:1795, :1843-1850) and truncation/refusal reasons only reach `print`, so provider reliability (schema-invalid,
   cutoff, refusal, fallback-0.5 rates) can never be measured from journals. ASK: approve adding four additive JSON fields to the analysis journal
   producer BEFORE the bots restart (measurement-only; INTEL + ENGINE halves; exact fields A1–A4 with file:line in the spec).
10. **BTC lagged beta (beta_ledger, real 90-day run):** summed lagged (AKL) BTC beta +1.18 vs same-day +0.25/OLS +0.44 — read before any SPY/BTC hedge
    is sized (Scout B F9 + E4). New report-only fields: `beta_winsor` (Welch δ=3), `beta_stable`, `alpha_mintrl_years` (2.0 y > 0.4 y window ⇒ "alpha not
    estimable in this horizon"). ASK: confirm δ=3 (paper) vs δ=1; note SPY clips 61% of days (Welch's band is centred on β=1), so its STABLE is partly artefact.
11. **Legacy journal rows with the score inside skip_reason** (~12k rows, `llm_below_buy_min (0.58<0.60)`, producer removed): they land in
    decision_report's `_unclassified_skip_reasons`. ASK: confirm exclusion by the Scout C boundary (journals through 2026-05-07) rather than parsing them.
12. **FLAGS.md facts proven wrong by the W20 cite pass (not fixed — facts, not cites):** UNIQUENESS_WEIGHTS_ENABLED row says "LSTM loss" but its only reader is
    `hypersearch_v2 train_lgb_ensemble()` (LightGBM leg); TRADER_PYBIN "never read at runtime" but scripts/backup_state.sh:26 reads it; TRADER_HOLDOUT_SPAN_BY_TARGET
    and TRADER_BREAKER_SERVER_FILL_ATTRIB lack §5 rows. ASK: approve the three row corrections (INTEL/SIGNAL/ENGINE facts).

## Money asks — PARKED for the founder (spend nothing beyond the one-minimal-call rule)
- A1 ≈ $0.75 for Scout A E1 (test-retest noise of the analyst score, ~300 calls) + E2 (hide-pred anchoring A/B, ~800 calls); `llm_qualify` bypasses the ledger.
- A2 one-time ≥ $10 OpenRouter credit (50 → 1,000 RPD) or drop OpenRouter from the free-first plan (Groq free ≈ 40 calls/day; analyst needs up to 288).
- A3 analyst sub-budget / cap raise before any PAID gemini-3.x flash-lite or Haiku analyst (estimated $0.58–1.30/day vs `_DAILY_COST_LIMIT=1.00`, llm_client.py:261).

## Standing facts the owner should know (no decision needed)
- Free-first qualification is NO-GO today by construction: no groq/openrouter keys, ollama not running, `journals/llm_replay` absent, LLM scorecard 0/120 clusters;
  groq/openrouter/ollama lack budget+pricing rows and gemini-3.x is absent from GEMINI_MODELS/KNOWN_MODELS (W5_owner.md ITEM 3 has the exact command plan).
- The analyst runs every 600 s (base_loop.py:94), not hourly; the prompt includes the ML pred by default (llm_analyst.py:448).
- A local LLM judge is infeasible on the Orin CPU; a FinBERT-class encoder fits (~134 ms/headline) but 2026 IC evidence (≤0.014) does not justify building it.
