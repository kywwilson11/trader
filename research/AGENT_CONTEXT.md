# Agent brief — read this first (canonical subagent context)

Purpose: every spawned agent working this repo reads THIS file instead of a
re-typed context paragraph. Orchestrator prompts should say "Read
research/AGENT_CONTEXT.md first" and add only the task itself.

## The system, one line
Autonomous Alpaca **paper-trading**: one RegressionLSTM+LightGBM blend per book
(crypto 24/7, US stocks RTH), long-only, meta-label + cost + LLM gates, honest
validation (purged walk-forward, Deflated Sharpe, shadow DM-HLN), prod on a
Jetson Orin Nano 8GB.

## Where to look things up (do not re-derive)
`docs/MAP.md` — how the repo is laid out and how the pieces fit; read it to
locate anything before grepping blind. `docs/MODULES.md` — per-module goal,
API, callers, CLI flags. `docs/FLAGS.md` — every flag and its default.
`docs/STATE_FILES.md` — every generated/runtime file. `docs/GLOSSARY.md` —
the vocabulary. `CLAUDE.md` — how to run and test, plus the authoritative
two-machine table and the numbered gotchas.

## Two-machine reality (plan around it)
- **This dev Mac (py3.13)**: numpy/pandas/scipy/statsmodels/yfinance/bidask/
  bs4/yaml/requests/pytest. NO torch, lightgbm, optuna, joblib, numba, sklearn,
  dotenv, alpaca, finnhub, PySide6/pyqtgraph, pyarrow, arch, hmmlearn. (Full
  table: `CLAUDE.md` § Two-machine reality.) Numba-decorated repo code has
  pure-python fallbacks. NEVER import heavy modules — read the code instead;
  test pure parts.
- **Jetson (py3.10)**: full stack. Training, live loops, GUI rendering, and
  anything needing real data/journals happen THERE, by the owner.

## Non-negotiable conventions
1. **Train/serve parity is sacred.** Features are computed by the same
   functions at harvest and live; changing a feature's VALUES is model-facing.
2. **Model-facing / gate-behavior changes never ship silently**: default-OFF
   flag with byte-compat pinned by a test, or a decision-queue entry for the
   owner. Measurement/instrumentation ships directly.
3. **Fail-open for the LLM gate** (an error can never block a trade);
   **fail-closed for live trading paths** (missing pred/quote ⇒ no entry).
4. **Do not commit or push.** Ever. The owner commits after review.
5. **Ownership discipline**: edit only the files your task names, plus your
   own new test file. Cross-file fixes get REPORTED, not implemented.
   **Delete nothing** — anything that must leave its place is MOVED (`git mv`
   when tracked) to `archive/` or a dated `research/` subdirectory, with a row
   added to the receiving README (`archive/README.md`).
6. **Do not rebuild killed ideas**: check `research/KILL_LIST.md` before
   proposing any strategy/feature/overlay (including its PENDING OWNER ASKS
   appendix — asked-about entries are still killed until the owner rules).
6b. **The 2026-08 campaign context**: `research/campaign_2026-08/` holds the
   defect map (`01_state_map.md`), the literature parameters (`02_research.md`),
   the activation runbook (`03_jetson_runbook.md`), the 2025-26 frontier survey
   (`05_frontier_research.md`), the signal-model plan and its 12-step Jetson
   sequence (`06_signal_model_plan.md`), the decision-influence ledger
   (`07_decision_influences.md`), and `08_removed_code.md` — the verbatim
   archive of every removed block, which you MUST read before "re-adding"
   anything that looks missing. Every model-facing change from that campaign is
   behind a default-OFF flag (strategy_config constants + `TRADER_*` env vars)
   whose OFF path is byte-pinned by a test; the verified inventory is
   `docs/FLAGS.md` — never quote a flag count from prose. Never flip a flag or
   change a flag-ON path without reading the runbook's evidence-gate for it.
7. Another Claude session may work this tree concurrently: re-read files
   immediately before editing, never `git stash` a shared tree for A/B
   (reconstruct baselines from `git show HEAD:<file>` instead), and prefer
   quiescence over racing.

## Verification (the standard)
- `python3 -m py_compile <touched files>` — always.
- Your own test file + any existing Mac-passing test module covering your
  files. Never weaken an existing test.
- **Regression check: `bash scripts/ab_check.sh`** — diffs the full suite's
  failure NAMES against `tests/baseline_failures.txt` (the known dev-Mac
  missing-dependency set). Exit 0 = clean. Any NEW name is yours to fix or
  revert. (The Jetson/CI suite state is recorded in `CLAUDE.md` § Running tests.)

## Style
Match surrounding code; comments only for constraints code can't show; be
honest in reports (deferred ≠ done; a clean-module no-op is a valid result).
