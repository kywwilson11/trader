# Jetson test & improvement campaign — shared agent brief (2026-09-26)

READ FIRST: /home/kyle/trader/research/AGENT_CONTEXT.md, then /home/kyle/trader/CLAUDE.md.
This session runs ON THE JETSON ORIN NANO (prod box), not the dev Mac. The full dependency stack is
available, real data/journals/models are present. The CLAUDE.md two-machine table's "Mac cannot"
column does NOT apply here — but its conventions (delete nothing, never commit/push, default-OFF for
model-facing changes, fail-closed live paths, kill list) apply in full.

## How to run Python here (mandatory)
    source /tmp/claude-1000/-home-kyle-trader/eb151c9d-a6af-415b-be38-65efdadeb88b/scratchpad/jenv.sh
    CUDA_VISIBLE_DEVICES='' $JPY <script>        # $JPY = /home/kyle/miniforge3/envs/jetson/bin/python (3.10)
jenv.sh sets LD_PRELOAD (conda libstdc++ — without it `import torch` then `import sqlite3` dies
with CXXABI_1.3.15) and LD_LIBRARY_PATH (cusparselt). ALWAYS keep CUDA_VISIBLE_DEVICES='' — never
touch the GPU. The GUI (PySide6) lives ONLY in the base env: /home/kyle/miniforge3/bin/python (3.12).

## Hard rules for this campaign
- Do NOT run the full pytest suite (the orchestrator owns it; a run is in progress). Single test
  files are fine: `$JPY -m pytest tests/test_x.py -q -p no:cacheprovider`.
- Do NOT pip install / uninstall anything. Do NOT git add/commit/stash/mv. Do NOT place orders,
  do NOT start bots or run_pipeline, do NOT start training (hypersearch), do NOT run harvest that
  writes the training stores. Read-only API calls (account, bars, quotes) are fine.
- Do NOT edit production files unless your task explicitly says so. Findings are the deliverable.
- Write your report to the path your task names (under the scratchpad). Cite file.py:line.
- Machine: 6-core ARM, 7.4 GB RAM (+12 GB swap), other agents run concurrently — keep any single
  process under ~1.5 GB RSS and under ~10 minutes wall unless told otherwise.

## Known device facts (verified by the orchestrator, do not re-derive)
- Bots have NOT run since 2026-05-07; trader.service is not installed; nothing trades now.
- Training stores: training_data.parquet (crypto, 6 names, ends 2026-02-24, 51 cols) and
  stock_training_data.parquet (46 names, ends 2026-02-19, 53 cols) — both OLD schema
  (no Eff_Spread_Pct / CS_* / Funding_* / TB_* columns).
- Model artifacts: model_v2.pth+pkl (2026-04-11), stock_* (2026-04-25). NO *model_v2.manifest.json,
  NO *lgb_model.txt / lgb_q10*, NO oof_preds.npz, NO meta_* artifacts anywhere.
- Missing from the jetson env: bidask, hypothesis, alpaca-py, fastparquet. Present: torch 2.8.0
  (CUDA ok), lightgbm, optuna, numba 0.63.1, sklearn, joblib, pyarrow 23, arch, hmmlearn, dotenv,
  alpaca-trade-api 3.2.0, finnhub. `pip check`: alpaca-trade-api wants websockets<11, have 16.0.
- An UNTRACKED C extension `indicators_c.cpython-310-aarch64-linux-gnu.so` (source c_ext/, built
  2026-02-28) sits in the repo root and indicators.py auto-uses it (`_HAS_C`). With it loaded the
  pytest suite ABORTS (SIGABRT, heap corruption) in tests/test_indicator_reindex.py ->
  indicators.compute_features. To disable it for a run: PYTHONPATH=<scratchpad>/noc (a sitecustomize
  that sets sys.modules['indicators_c']=None).
- With the C ext disabled, tests/test_indicators_parity.py::test_compute_stock_features_golden_fingerprint
  FAILS on the Jetson (numba path, numpy<2, pandas 2): 422055739916008620 != golden 8972321854121808304.
- Import census: all 116 root+scripts modules import in the jetson env except gui (PySide6, base
  env only) and scripts/generate_manual.py (untracked residue, needs fpdf). Importing almost any
  module touches logs/trader.log (known smell).
- Untracked residue in the root from March 2026 (not in git): Trader_System_Manual.pdf,
  scripts/generate_manual.py, scripts/manual_expanded.py, scripts/mem_diagnostic.py,
  global_context.py(+json, no importer), research_*.txt, review_findings.md, research_gap_analysis.md,
  *.backup_score10, *_study.db.bak*. Leave them alone.
