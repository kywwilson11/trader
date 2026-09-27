# FIX_B — March-2026 residue moved to archive/ (2026-09-26)

## Pre-move verification
- `git log --all -- <path>` returned nothing for all 19 files, so none was ever tracked.
  `git log --all -S global_context` / `-S mem_diagnostic` / `-S generate_manual` were also empty.
- `grep -rn` over the whole repo (excluding .git): no importer and no string reference outside the
  residue set itself. **global_context.py: no importer, no string ref.** Its only hits are its own
  source, so the move went ahead. Not in crontab and not in any systemd user unit.
  Within the residue: `generate_manual.py` names `Trader_System_Manual.pdf` (its hard-coded output path),
  `manual_expanded.py` names `research_gap_analysis.md` (in prose), and `research_trading_strategies_2026_03.txt`
  names its predecessor.
- `manual_expanded.py` is imported by nothing, not even `generate_manual.py`, and has no `__main__`,
  so it was the unreachable module. `generate_manual.py` needs `fpdf`, which is absent from the jetson,
  base and system pythons.
- Baseline before the moves: `repo_graph.py --check` → `CHECK FAILED: 1 unreachable module(s) ... ['scripts/manual_expanded.py']`, exit 1.
- `docs/graphs/README.md`: `--check` analyses the live tree and `tests/test_repo_graph.py` writes its JSON to a tmp `--out`.
  Neither compares against the committed `docs/graphs/import_graph.json`, which already contains none of
  these modules (grep count 0). **The JSON was not regenerated** because nothing needs it.

## Moves (plain `mv -n`, since all were untracked)
| From | To | Ignored at destination? |
|---|---|---|
| Trader_System_Manual.pdf, scripts/generate_manual.py, scripts/manual_expanded.py | archive/jetson_residue_2026-03/ | no (shows as `??`) |
| scripts/mem_diagnostic.py | archive/jetson_residue_2026-03/ | no |
| global_context.py, global_context.json | archive/jetson_residue_2026-03/ | no |
| research_code_quality.txt, research_nobel_economics.txt, research_trading_strategies.txt, research_trading_strategies_2026_03.txt | archive/jetson_residue_2026-03/ | no |
| review_findings.md, research_gap_analysis.md | archive/jetson_residue_2026-03/ | no |
| stock_{config,feature_cols,scaler}_v2.pkl.backup_score10, stock_model_v2.pth.backup_score10 | archive/local_residue/ | yes: `.gitignore:162 archive/local_residue/*.backup_score10` |
| stock_v2_study.db.bak, stock_v2_study.db.bak_pre_fresh_20260322, v2_study.db.bak | archive/local_residue/ | yes: `.gitignore:163 archive/local_residue/*_study.db.bak*` |

Note: `archive/local_residue/` did **not** exist on the Jetson. The rows already in the README describe
dev-Mac files. I created the directory.

## Kill-list check (research_gap_analysis.md / review_findings.md)
- review_findings.md lists code defects only, so it has no kill-list overlap. It is superseded by `research/module_review_2026-07.json`.
- research_gap_analysis.md overlaps the kill list in these places:
  - Gap #7 (Ledoit-Wolf shrinkage) is KILLED.
  - Its "implemented" table lists CAPE and Month_sin/cos. Pseudo-CAPE is KILLED and its code was deleted 2026-08-22. Month_sin/cos is KILLED.
  - Gaps #11 and #12 build on the HMM layer, which is marked "cut recommended (pending)".
  - Gap #9 (Hurst-gated threshold) is the branch `07_decision_influences.md` ruled dead ("NO — unanimous"), and IA removed it.
- All of this is recorded in the archive/README.md row.

## Edits (append-only)
- `.gitignore`: +5 lines (comment + 2 anchored patterns).
- `archive/README.md`: +8 lines, i.e. 7 rows inserted after the other agent's c_ext row. The file was
  re-read immediately before insertion, and nothing else was changed.
- `docs/STATE_FILES.md` §11: +10 lines, one paragraph appended at the end of the section.

## Verification
- `git status --short`: every root residue name is gone. The only new untracked entries are `?? archive/jetson_residue_2026-03/`,
  `?? archive/c_ext/` (the other agent's) and `?? tests/test_market_data_sip_clamp_2026_09.py` (the other agent's).
  `archive/local_residue/` shows as `!!` (ignored). The other ` M` files belong to other agents.
- `$JPY scripts/repo_graph.py --check` → `CHECK PASSED: 0 unreachable modules, 0 EAGER import cycles.` (exit 0)
- `$JPY -m pytest tests/test_repo_graph.py -q -p no:cacheprovider` → `4 passed in 19.59s`
- The full suite was not run, as instructed.

## Notes / skipped
- The Group A files are untracked but not ignored, so they will appear as `??` until the owner
  commits them or adds an ignore rule. This follows the `commit_msg.txt` precedent. No broad pattern was added.
- The README paragraph "All three `local_residue/` patterns are depth-agnostic…" still describes only the
  original three rows. My two new rows state their own anchored patterns, and I left the paragraph
  unedited because the task was append-only.
- The other agent's c_ext row ends with "the untracked `scripts/generate_manual.py` text". That file now lives at
  `archive/jetson_residue_2026-03/generate_manual.py`. The row is theirs, so I left it unedited.
- `docs/STATE_FILES.md` §11's heading still reads "what moved on the dev Mac". The new paragraph is
  labelled "Jetson, 2026-09-26" and the heading was not renamed.
