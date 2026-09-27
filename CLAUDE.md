# CLAUDE.md — trader

Autonomous **paper-trading** system (Alpaca) for **crypto** (24/7) and **US stocks** (market hours).
One **RegressionLSTM + LightGBM blend per book** (the old dual bear/bull ensemble is gone —
"bear/bull" survives only as regime diagnostics and the champion/challenger shadow slots)
+ meta-labeling, Numba TA features, honest validation (purged walk-forward + Deflated Sharpe),
cost-aware gating, and a policy backtester that replays *real* exits before any model is promoted. Runs in production on an **NVIDIA Jetson
Orin Nano (8 GB)**. Research is organized into numbered "waves" (`research/waves/wave1_eval.json`
+ `research/waves/wave{2..9}_research.json`; index: `research/README.md`).

> **This file is the operational source of truth** for how to run, test, and work in this repo —
> if it and the code disagree, trust the code. The architecture narrative (how every piece fits,
> and why) lives in `docs/MAP.md`. Current rolling state lives in the `session-state` memory,
> which is out of the repo and this-Mac-only.

---

## Repository map

Facts have **one home**. Before restating a number or a list, check whether one of these owns it:

| Where | Owns |
|---|---|
| `docs/MAP.md` | how the whole system fits together — layout, pipeline, promotion path, invariants, history |
| `docs/MODULES.md` | per-module reference + the full CLI/argparse census for every entry point |
| `docs/FLAGS.md` | every `strategy_config.py` gate constant and `TRADER_*` env var: default, read site, model-facing?, evidence gate |
| `docs/STATE_FILES.md` | every generated/runtime file — who writes it, who reads it, gitignore status, which machine |
| `docs/GLOSSARY.md` | the vocabulary (book, champion/challenger, shadow, DM-HLN, DSR, PBO, effective-n, Stage-0, packet IDs…) |
| `docs/README.md` | index of `docs/` + reading order for humans vs agents |
| `archive/` | **moved, not deleted** — stale scratch/config that left the working tree keeps a row in `archive/README.md` |
| per-directory READMEs | `research/README.md`, `research/campaign_2026-08/README.md`, `research/campaign_2026-09_jetson/README.md`, `scripts/README.md`, `tests/README.md`, `.claude/README.md`, `logos/README.md`, `fonts/README.md` |

**Nothing in this repo is deleted.** Anything that must leave its place is moved (`git mv` when
tracked) and recorded in the receiving directory's README.

---

## ⚠️ Two-machine reality (the #1 operational fact)

Work happens across two machines with **different installed dependencies**:

| | **This dev Mac** | **Jetson Orin Nano (prod)** |
|---|---|---|
| Python | 3.13.5, framework `python3` (no venv) | 3.10, `/home/kyle/miniforge3/envs/jetson/bin/python` |
| Installed | numpy, pandas, scipy, statsmodels, bidask, yfinance, bs4, yaml, requests, hypothesis, **pytest** | full stack incl. torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, CUDA |
| **NOT installed** | **torch, lightgbm, optuna, joblib, numba, sklearn, dotenv, finnhub, alpaca, alpaca_trade_api, PySide6, pyqtgraph, pyarrow, fastparquet, arch, hmmlearn** | — |
| Can do | pure-algorithm code, synthetic-data unit tests, web research | training, harvest, Stage-0 measurement, live trading, GUI |
| **Cannot do** | model training, data harvest, parquet round-trips, anything importing the heavy deps, run the bots/GUI | — |

**Running anything on the Jetson by hand:** the jetson interpreter must be invoked with the conda
libstdc++ preloaded (`LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6`) and the
cusparselt dir on `LD_LIBRARY_PATH`, exactly as `run_pipeline.ENV` and the systemd unit set them —
otherwise `import torch` followed by `import sqlite3` fails with `CXXABI_1.3.15 not found`
(verified 2026-09-26). Tests there: same wrapper, `CUDA_VISIBLE_DEVICES=''`, ~2 min for the suite.

**Implication for how I plan work:** on the Mac, only build/verify things provable with
numpy/pandas/scipy/bidask + synthetic data. Anything needing torch/lightgbm/joblib/dotenv/real
journals/parquet is **Jetson-gated** — write it, unit-test the pure parts, and flag it for the
user to run on the Jetson. The Mac↔Jetson sync is user-driven and **not encoded in the repo** —
do not invent ssh/rsync steps.

---

## Running tests

**Canonical dev-Mac command** (some test modules can't import their heavy deps, so always continue past
collection errors):

```bash
python3 -m pytest tests/ --continue-on-collection-errors -q
```

**Current baseline (verified 2026-09-08): `16 failed, 3833 passed, 25 skipped, 7 errors`.**
The 16 failures + 7 errors are the **23-name pre-existing missing-dependency set** pinned in
`tests/baseline_failures.txt`, **not regressions**. The deps actually implicated are
pyarrow/fastparquet (12), joblib (3), hmmlearn (3), torch (2), arch (2), dotenv (1);
lightgbm / numba / optuna / sklearn are also absent here but cause **no** baseline name — those
paths are guarded (`importorskip`, pure-python fallbacks, source-text-only tests). On the full Jetson stack
(verified 2026-09-26) it was 3919 passed / 5 failed — the untracked C extension, the Mac-recorded golden
fingerprint test, untracked residue and the missing `bidask` — all addressed in the 2026-09-26 campaign. `ab_check.sh` is hardened (launch-sanity floor ≥1500 passed, watchdog timeout,
flaky/persistent triage of NEW names, forced `PY_COLORS=0`); it judges by failure NAMES, never counts.
This line is the **only** place a suite count is quoted — every other doc points here.

**Verify a change introduced no regressions — the one-command way:** `bash scripts/ab_check.sh`.
It reruns the canonical command above, diffs the FAILED/ERROR test *names* (never counts) against
`tests/baseline_failures.txt`, prints NEW vs DISAPPEARED names separately, and exits 0 iff no NEW
names appear. Regenerate the baseline after an intentional change with the command in that file's
own header comment — **keep `PY_COLORS=0`**, or ANSI codes defeat the anchored `grep` and it
silently emits nothing:

```bash
PY_COLORS=0 python3 -m pytest tests/ --continue-on-collection-errors -q 2>/dev/null \
  | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort -u
```

That command prints only the 23 name lines — re-add the file's 8-line header comment block by hand. The baseline
is dev-Mac-specific (missing-dep failures only) and must never be ported to the Jetson or CI — their
current state is the verified-2026-09-26 sentence in the baseline paragraph above, the only place it is quoted.

**Verify a change introduced no regressions — the from-scratch method** (use when
`tests/baseline_failures.txt` itself might be stale, or to regenerate it): A/B with `git stash` —
run the command above, `git stash`, run again, compare the failed/errored set. Identical set ⇒
zero regressions. Never treat the 23-name baseline as "broken." **Only on a single-writer tree** —
when another session shares the tree, reconstruct the baseline with `git show HEAD:<file>` instead.
The `/regression-ab` skill runs this ritual end-to-end and diffs failure *names*, not counts.

- `pytest tests/ --collect-only -q` still reports collection errors for the heavy-dep test modules (the
  collected count includes tests inside modules that then error at import).
- `tests/test_sentiment_headlines.py` is a **standalone runner**, not a pytest module
  (`tests/conftest.py` ignores it): 1035 scored checks behind a ≥99 % statistical gate, not
  per-case asserts. It **does** run on this Mac — `python3 tests/test_sentiment_headlines.py`
  → 1034/1035, writes nothing — but it makes one fail-soft outbound HTTP call.
- New heavy-dep tests should `pytest.importorskip` at module level so they SKIP rather than ERROR.
- CI (`.github/workflows/ci.yml`): Ubuntu, two legs — py3.10 `jetson-parity` (installs
  `requirements-ci.txt`: the prod pins, `numpy<2`, `pandas<3`, lightgbm, alpaca-py) and py3.12
  `modern` (unpinned forward-compat). Both: `py_compile` of top-level + `scripts/` → the sentiment
  runner → `pytest tests/ -v --tb=short -x` (note `-x`, and no `--continue-on-collection-errors`).

---

## Running the system

All entry points are plain `python <file>.py`. On this Mac most require the Jetson stack; the daily
ones are below with their **verified** flags. **The complete CLI census (every entry point, every
`add_argument`, every default) is `docs/MODULES.md`** — add new flags there, not here.

| Command | What it does |
|---|---|
| `python run_pipeline.py` | Orchestrator: harvest → train → gate → launch bots → weekly retrain (**the retrain STOPS the bots and restarts them** — `_stop_bots` → `_restart_bots`; see § Weekly retrain vs hot-reload). Flags: `--trials N` (200), `--retrain-trials N` (100), `--retrain-day N` (5), `--retrain-hour N` (2), `--no-retrain`, `--bot-only`, `--skip-harvest`, `--combined-bots`, `--crypto-only`, `--stock-only` |
| `python run_bots.py [--crypto-only\|--stock-only]` | Live bots only — runs BOTH loops in one process (saves ~0.5–0.8 GB RAM on Jetson). **`run_pipeline`'s own default is one process per bot**; combined mode is opt-in via `--combined-bots`, which is what the systemd unit in `scripts/setup_jetson_system.sh` passes — so production runs combined and a bare `python run_pipeline.py` does not |
| `python scripts/harvest_crypto_data.py` / `harvest_stock_data.py` | 1Y hourly OHLCV + features → `*training_data.{csv,parquet}`. No flags (neither script uses argparse) |
| `python scripts/hypersearch_v2.py --trials N [--prefix stock] [--data F] [--fresh] [--shadow] [--preset P] [--max-rows N] [--mode {refine,explore,initial}] [--no-status]` | Optuna TPE search (LSTM + LightGBM leg), holdout DSR gate. `--preset` defaults to `None` = whatever `load_indicator_config()` says |
| `python backtest.py --prefix {''\|stock} --days N [--gate] [--min-sharpe X] [--min-dsr X] [--model-prefix P] [--trials N] [--no-stage0-dump] [--fee-mult X] [--fee-sweep '1.0,1.5,…']` | Policy replay (real entries/exits/fees; `''` = crypto); `--gate` rolls back to `.prev` on Sharpe/DSR fail; `--fee-sweep` reports breakeven cost headroom λ* |
| `python decision_report.py --days N` | Per-trade gate attribution + conviction calibration (Stage-0 measurement) |
| `python llm_eval.py --days N [--asset {crypto,stock}] [--advisor]` | LLM-gate scorecard: veto/size-tilt outcome attribution + echo-gap regression (b2 significance with ≥60 rows AND ≥120 distinct hourly t0 clusters AND n_eff ≥ 20 — ≈20+ days of LLM cycles — = the keep/kill-LLM-spend verdict); `--advisor` scores the shadow advisor-v2 dossiers. Measurement-only |
| `python beta_ledger.py --days N [--lags N] [--equity-csv F] [--benchmarks-csv F] [--json F]` | Realized-beta ledger: daily equity vs SPY+BTC (lagged AKL betas, HAC alpha t-stat, up/down + trend-conditional betas). Measurement-only |
| `python indicator_leadlag.py --data F [--preset P] [--features L] [--horizons 1,4,12,24,48] [--fdr-q Q] [--json F]` | Per-feature leading/lagging diagnostic: predictive IC vs reactive coupling at 1–48h (overlap-adjusted, FDR), redundancy clusters + exact dupes. Measurement-only |
| `python gui.py` | PySide6 dashboard (8 tabs, 12 themes); reads `pipeline_status.json` + logs. Chart math lives in pure-numpy `chart_core.py` — testable on this Mac |

### Weekly retrain vs hot-reload — two different hand-offs

Older docs said "hot-reload, bots never stop". That is wrong for the weekly path and right for the
daily one:

- **Weekly retrain = a cold restart.** `run_pipeline.main`'s PHASE C calls `_stop_bots` (SIGTERM →
  10 s → kill), runs `_run_training(..., is_retrain=True)`, then `_restart_bots` → `_launch_bots`.
  The new champion is picked up by `BaseTradingLoop._load_models` at startup, not by a reload.
- **Daily shadow promotion = the actual hot-reload.** `shadow.evaluate_and_maybe_promote` (the
  pipeline's daily check) runs while the bots trade; `shadow.promote_challenger` copies challenger
  artifacts over the champion and writes `{p}model_v2.manifest.json` **last**.
  `trading_utils.model_reload_key` keys on that manifest's mtime, so `base_loop._hot_reload_check`
  swaps models on the next 30 s cycle (and drops the `predict_now` booster caches).

---

## Architecture in brief

Full narrative — layout, lifecycle, the goal of each piece — is `docs/MAP.md`. The short version:

**Pipeline:** data → features → RegressionLSTM+LightGBM blend → cost gate → meta-label gate →
sentiment/LLM gate → order → ATR-based exits → cross-book risk cap.

**Single source of truth — `strategy_config.py`.** `CRYPTO_POLICY`/`STOCK_POLICY` (ATR mults, stop
floors, TP RR, cooldowns), `RISK_PCT_PER_TRADE=0.005`, `MAX_BOOK_RISK_PCT=0.025`, `KELLY_CAP=0.25`,
vol targets, entry windows, overnight sleeve, execution tactics, IOC caps. **Both the live loops AND
the backtester read it** — drift here means the backtest validates a different policy than trades.
Never mutate the shared policy dicts at runtime; read them, copy before adapting.

**Shared kernels (one implementation, many consumers — keep them in sync):**
- `policy_exits.py` — Numba exit-stack kernel (hard/trailing/TP/signal/EOD/vertical). Called by
  `backtest.simulate_ticker`, the harvest triple-barrier labels
  (`scripts/harvest_*_data.py` → `compute_tb_labels`), `meta_label._gen_meta_rows` and
  `decision_report`. So **labels == backtest exactly** — same kernel, same policy dicts.
  **Live == kernel by MIRROR, not by call:** no live loop imports the kernel
  (`tests/test_ia4_flagged.py::test_vertical_is_loop_layer_only` pins that), because the loops
  must act on one bar at a time; `base_loop._desired_stop_for` / `_manage_stops` (and
  `stock_loop`'s copies) re-implement the same stack, equal up to the divergence list in
  `policy_exits`' own docstring — including one open trailing-denominator asymmetry (kernel and
  `stock_loop._manage_stops` divide by entry, `base_loop._desired_stop_for` by the HWM), queued as
  an owner decision. `exit_walk(side=+1)` is the long path; `side=-1` is the offline short mirror.
- `fees.py` + `liquidity.py` (`bidask` EDGE per-name spread) — the cost model every gate shares.
- `base_loop.py` — Template-Method base for `crypto_loop.py` + `stock_loop.py`.

**Validation/promotion:** `validation.py` (Deflated Sharpe, `DSR_MIN=0.60`; CSCV-PBO; Lo-2002 serial
factor), `sample_weights.py` (avg-uniqueness → effective-n), `backtest.py` (policy-replay promotion
gate), `meta_label.py` (secondary classifier; veto p<0.30). *What is actually WIRED:*
`dsr_from_trade_returns` + `DSR_MIN` decide promotion (`backtest.py`, `scripts/hypersearch_v2.py`,
`portfolio_backtest.py`), and `pbo_from_fold_scores` prints a coarse PBO diagnostic at the end of a
search — nothing else. The full `pbo_cscv`, the Lo-2002 `serial_correlation_factor` and
`stationary_bootstrap_sharpe_pvalue` are implemented and tested but have **no production caller**;
they are available, not enforcing. Detail in `docs/MAP.md`.

**Module families** (per-module goal, API, and imported-by: `docs/MODULES.md`) — signals/features,
execution, costs/risk, models, LLM, ops. The one worth stating here because it is easy to get
wrong: `llm_client.py`/`llm_analyst.py` speak to **Gemini + Anthropic/Claude + OpenAI and
OpenAI-compatible endpoints**, all schema-enforced (Gemini responseSchema, Claude forced tool use —
auto + strict tool + client-side validation on models that reject forcing, see `docs/MODULES.md`
§llm_client — OpenAI strict structured outputs), with the provider switch, per-role overrides and pricing
corrections in `llm_config.json`, cross-provider fallback, `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`
accepted from env — and the backfill role pinned to Gemini for its Batch API.

**PIT discipline (do not break):** all features strictly trailing; sentiment/short-interest lagged
to publication date; borrow cost regime-dated; universe membership as-of (no survivorship).

---

## Conventions

- **Commit/push ONLY when the user explicitly asks.** The user reviews everything first; check
  `session-state` for what is currently in flight/uncommitted.
- **Commit style:** conventional prefixes seen in history — `feat:`, `fix:`, `docs:`, `test:`,
  imperative mood, often `feat: wave-N <scope> — <detail>`. Branch is `master`; remote `origin`.
- **Deployment gate:** every **model-facing** change ships only through the
  **challenger → shadow → DM-HLN** promotion path. **Instrumentation/measurement-only** changes
  (journals, reports, offline research kernels) are safe to ship directly.
- **Research → ship:** completed wave research lives in `research/waves/` (committed; index and
  frozen/live status in `research/README.md`). Implement `mac_now` items on the Mac, defer
  `jetson_later` items. The consolidated, canonical kill list across all waves/reviews/research is
  `research/KILL_LIST.md` — every research or build agent MUST check it before proposing anything;
  entries leave only by explicit owner decision.
- **2026-07 module review is DONE** (69 modules, 600 fns): the 90-item owner decision queue (1 P0/21 P1/68 P2)
  lives in `research/module_review_2026-07.json` — render with `/decision-queue`; queue items are owner decisions, do NOT auto-fix.
  The panel-review corpus behind it is frozen in `research/reviews_2026-07/`.
- **2026-08 comprehensive campaign is DONE** (10 waves: understand → research → fix → build →
  bug-hunt; every wave gated by ab_check). Canonical docs `01`–`08` in `research/campaign_2026-08/`
  (indexed in its README): `01_state_map.md` 40-defect map, `02_research.md` literature parameters,
  **`03_jetson_runbook.md` the activation sequence — read before flipping ANYTHING**, `04` frozen
  commit drafts, `05_frontier_research.md` 2025-26 frontier, **`06_signal_model_plan.md` signal-model
  defects + the 12-step Jetson sequence (§4 continues the runbook)**, `07_decision_influences.md`
  influence ledger, **`08_removed_code.md` verbatim archive of every removed block — read it before
  "re-adding" anything** (cited from production code). Every model-facing change from the campaign
  sits behind a **default-OFF** flag (strategy_config constants + `TRADER_*` env vars) with the
  flag-OFF path byte-pinned by a test — verified inventory in `docs/FLAGS.md`, never a count in prose.
  The campaign also added a shelf of measurement-only CLIs (window A/B, naive-vs-blend
  falsification, horizon transfer, funding drift, spread census, LLM qualification, learning
  curves, `backtest.py --fee-sweep`) — catalogued in `scripts/README.md`; the weekly backtest
  auto-emits `{slot}_stage0_preds.json` for `scripts/ic_by_name.py` + `scripts/rank_gradient_report.py`.
- **Since the campaign commit (`20a41db`), UNCOMMITTED and awaiting owner review:** the **R2**
  signal-model round (R2-A frontier research → R2-B panel → R2-C build, 8 packets behind their own
  default-OFF flags, `tests/test_r2c_*.py`), the **R3** econ/Nobel research round
  (`research/literature/nobel_modern_research_2026-08.md`, research-only), and the
  **IA-1..IA-4 decision-influence implementation** (`07_decision_influences.md` → pseudo-CAPE
  deleted, dead Hurst/sentiment-gate branches and exit-path cooldown removed, seven IA-4 flags
  default OFF; every removal archived verbatim in `08_removed_code.md`; `tests/test_ia{1..4}_*.py`).
  Do not describe any of this as shipped. *(Annotation 2026-09-26: all of the above was committed
  2026-09-10 as `438f56a`. What is uncommitted now is the 2026-09-26 on-Jetson campaign —
  `docs/MAP.md` §8.)*
- **Claude Code assets** (`.claude/` is COMMITTED as of 2026-07-21 — only `settings.local.json` stays
  gitignored, so skills/workflows/hooks/settings/agents travel to the Jetson; see `.claude/README.md`):
  skills `/regression-ab`, `/decision-queue`, `/improve`, `/panel-improve`; workflows
  `group-improve-v2` (broad cheap sweep — one designer per group) and `module-improve-v3` (deep
  panel pass — N independent Opus reviewers per module on an identical brief → Fable adjudicates
  into a verified spec → Sonnet implements → Fable hardens → one serialized ab_check gate);
  agent definition `.claude/agents/fable-high.md` (Fable at effort-high); blocking py-syntax
  PostToolUse hook.
- **Spawned agents:** point them at `research/AGENT_CONTEXT.md` (the canonical brief — two-machine
  table, conventions, ab_check verification standard) plus the task itself; don't re-type context.
- **Effort:** the user wants real max-effort, literature-grounded, multi-agent work and accepts long
  runs. Priorities, in order: **Jetson 8 GB memory/perf › financial soundness › LLM utilization ›
  trading strategy.** See `user-working-style` memory.

---

## Environment & secrets

- **Secrets** (in `.env`, gitignored): `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`,
  `FINNHUB_API_KEY`.
- **Ops env vars:** `TRADER_USE_ALPACA_PY` (use alpaca-py adapter), `TRADER_SHADOW_MODE`
  (**default ON**; its ONLY effect is that the weekly retrain saves into the challenger slot —
  `run_pipeline._build_training_phases`; it suppresses no orders), `TRADER_ORDER_STREAM`,
  `TORCH_NUM_THREADS`
  (=2 for bots), `CUDA_VISIBLE_DEVICES` (='' → bots CPU-only).
  Notifications: `TRADER_TELEGRAM_*`, `TRADER_WEBHOOK_URL`, `TRADER_HEALTHCHECK_URL`.
- **Behaviour flags — the full, verified inventory is `docs/FLAGS.md`** (name, default, read site,
  model-facing?, evidence gate). The ones worth knowing by heart, all in `strategy_config.py`:
  `HAR_VOL_ENABLED`, `CONVICTION_JOURNAL_ENABLED`, `MAKER_ENTRIES_ENABLED`, `ENTRY_WINDOWS_ENABLED`,
  `OVERNIGHT_SLEEVE_ENABLED`; `OBJECTIVE_LONG_ONLY` (default off — score only the deployable long
  side in hypersearch; flipping = objective change ⇒ gotcha #2). In `indicator_config.py`:
  `HURST_ON_RETURNS` (default off — correct R/S input; model-facing, flip only with
  harvest+retrain ⇒ gotcha #2).

## What's committed vs generated

**Committed:** all `*.py` (root, `scripts/`, `tests/`), `docs/`, `research/` (`.md` + `.json`,
including `campaign_2026-08/`), `archive/`, `.claude/` (skills, workflows, hooks, agents,
`settings.json`), `.github/workflows/ci.yml`, `requirements*.txt`, `pyproject.toml`,
`stock_universe.json`, `strategy_config.py` & friends, `fonts/`, `logos/`, `README.md`, `CLAUDE.md`.
**Gitignored (generated/secret):** `.env`, `.claude/settings.local.json`, `*.pth`/`*.pkl`/`models/`,
`*_study.db`, `*.log`/`logs/`, training `*.csv`/`*.parquet`, `*_archive.parquet`, `journals/`,
`*_predictions.json`, `llm_config.json`, `indicator_config.json`, `sentiment_cache.db*`,
`pipeline_status.json`, `adaptive_state*.json`, `*.prev`.
So: code + docs + research + policy config are versioned; all data/models/runtime state/secrets are
local. **The authoritative per-file table — writer, reader, machine, ignore status — is
`docs/STATE_FILES.md`.**

## Gotchas

1. **Docs drift.** When any doc and the code disagree, trust the code — then fix the doc in the one
   place that owns that fact (see § Repository map).
2. **First Jetson retrain after objective/feature changes:** delete `v2_study.db` + `stock_v2_study.db`
   (old Optuna scores incomparable) and reset the adaptive `best_score`.
3. **Package pins are not reality on the Mac:** `requirements.txt` says `numpy<2` but this Mac has
   2.4.6, and `requirements-{ci,jetson}.txt` pin `pandas<3` while this Mac has 3.0.3 (py3.13 forces
   both). Fine for the pure modules — the `pct_change` fill_method divergence is pinned in code —
   but don't trust either pin as a description of this machine.
4. **Don't stack variance corrections:** `validation.serial_correlation_factor` (Lo-2002) is OFF by
   default and is *separate* from the label-overlap uniqueness effective-n — using both double-counts.
5. **No background automation is scheduled.** A prior timed workflow hop-chain is dead; do **not**
   create scheduled wakeups for this project unless the user asks.
6. **Delete nothing.** Stale files move to `archive/` (or a dated `research/` subdirectory) with a
   row in the receiving README; `git mv` for tracked files so history follows.
7. **`pyarrow` is declared but not installed here.** It was added (unversioned) to
   `requirements.txt`, `requirements-ci.txt` and `requirements-jetson.txt` on 2026-09-08 — parquet
   is a first-class path (`run_pipeline` imports it directly; seven modules read/write parquet).
   This Mac still has no engine, so parquet round-trips are Jetson/CI-only and six of the 23
   dev-Mac baseline names are exactly that. Assume nothing about parquet availability here.

---

*Memory (the index, `session-state`, `user-working-style`, `trader-architecture-truth`, the wave
archives) lives outside the git repo at
`~/.claude/projects/-Users-kywwilson-Desktop-Projects-trader/memory/` — this Mac only. Anything an
agent on another machine needs must live in `docs/`, not there.*
