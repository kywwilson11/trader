# Indisputable-improvement hunt — shared brief (2026-09-26 night)

Read <scratchpad>/CAMPAIGN_BRIEF.md first (env, rules), then research/AGENT_CONTEXT.md, then docs/MODULES.md
for your modules. Then read `git diff --stat` and <scratchpad>/reports/FIX_*.md titles so you know what was
already changed today (do not re-report those).

## The bar: 100 of 100 senior engineers would agree, without discussion
A finding qualifies ONLY if it is one of:
  A. **A proven bug**: wrong output, crash, exception swallowed on a live path, resource leak, race, wrong
     unit/timezone/sign, off-by-one, dead branch that was meant to run, a test that cannot fail. You must
     show the failing input and the correct output (a runnable repro in the scratchpad, or a cited test).
  B. **A provably equivalent simplification**: duplicated logic that has already diverged (cite both
     copies and the divergence), dead code with zero callers (prove with grep + the import graph), an
     unreachable branch, a re-derivation of a value that a single source of truth already provides.
  C. **A bit-identical (or float-tolerance-tested) speedup or memory reduction that matters on the
     8 GB Jetson**: O(n²) → O(n) on a hot path, repeated file/JSON reads inside a 30 s cycle, a full
     DataFrame copy where a view suffices, per-cycle re-imports, unbounded caches. You must measure
     before/after on this device and assert equality in a test.
  D. **Robustness that cannot change a decision**: atomic writes where a torn file would be read by
     another process, missing timeouts on network calls that already have a documented budget, a
     bare `except:` that hides KeyboardInterrupt/SystemExit, mutable default arguments.
DOES NOT qualify: anything model-facing (feature values, labels, thresholds, gate order, sizing math,
exit rules, prompts), anything on research/KILL_LIST.md or in 08_removed_code.md, style, naming,
type hints, docstrings, "would be cleaner", alternative algorithms whose output differs, anything two
reviewers could argue about. If in doubt, it does not qualify — put it in a separate "judgment calls
(not proposed)" appendix so nothing is lost.

## Machine etiquette (a harvest/training run shares this box)
- Read first; run Python only when you need a repro or a measurement. CUDA_VISIBLE_DEVICES='' always.
- At most ONE pytest process at a time, single files only, `-q -p no:cacheprovider`, never the full suite.
- Keep every process under ~600 MB RSS and under 5 minutes. No network calls except read-only Alpaca.
- Do NOT edit anything in the hunt phase (read-only). Fixes come later under one owner per file.
- Do NOT touch the training-path modules at all this round (another process is about to run them):
  scripts/hypersearch_v2.py, meta_label.py, backtest.py, model_v2.py, model_lgb.py, blend_fit.py,
  objective_utils.py, validation.py, sample_weights.py, policy_exits.py, data_utils.py, indicators.py,
  scripts/harvest_*.py, predict_now.py, serving_cache.py, shadow.py, panel_ranks.py, calibration.py.

## Deliverable format — <scratchpad>/hunt/<GROUP>.md
For each finding: id · file:line · class (A/B/C/D) · one-sentence defect · the proof (repro path or
cited test/measurement + numbers) · the exact fix (small diff or precise description) · blast radius
(callers, tests that pin the current behaviour) · why it is indisputable (one line). Rank by severity.
End with the "judgment calls (not proposed)" appendix and a "no issues found" list per file so
coverage is visible. Findings without proof will be discarded by the adjudicator.
