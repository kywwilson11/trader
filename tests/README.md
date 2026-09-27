# `tests/` — how the suite is organized and how to run it

**219 `test_*.py` inventoried** (as of 2026-09-27), plus `tests/__init__.py` (empty) and
`tests/conftest.py`. 218 of the 219 are collected by pytest — `test_sentiment_headlines.py` is a
standalone runner (see below). The 2026-09 Jetson campaign is still landing files; any
`test_*_2026_09.py` not in the appendix below is in flight and not yet inventoried.
The suite is prefix-organized by *provenance*: each campaign or research wave
dropped its own family of files rather than editing the older ones, so a file's prefix tells you
which pass wrote it and what contract it was pinning.

> **Where the numbers live.** Pass/fail/skip figures for the current dev-Mac run live in
> `CLAUDE.md` § *Running tests* and are not repeated here. This file owns the *inventory*
> (files, families, per-file purpose) and the explanation of `tests/baseline_failures.txt`.

---

## Running the suite

**Canonical dev-Mac command** — several test modules cannot import their heavy deps, so always
continue past collection errors:

```bash
python3 -m pytest tests/ --continue-on-collection-errors -q
```

**The regression check is `bash scripts/ab_check.sh`.** It reruns exactly that command and diffs
the `FAILED`/`ERROR` test **names** against `tests/baseline_failures.txt`. It **judges by names,
never by counts** — it prints NEW vs DISAPPEARED names separately and exits 0 iff no NEW name
appears. It also forces `PY_COLORS=0`, applies a launch-sanity floor and a watchdog timeout, and
re-runs NEW names to separate flaky from persistent. A NEW name is a regression; a DISAPPEARED
name means the baseline needs regenerating.

**Regenerating the baseline** (only after an intentional change, and always paired with a
`ab_check.sh` re-run):

```bash
PY_COLORS=0 python3 -m pytest tests/ --continue-on-collection-errors -q 2>/dev/null \
  | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort -u
```

`PY_COLORS=0` is **required**: without it ANSI colour codes precede `FAILED`/`ERROR` and the
anchored `grep` silently returns nothing. The command emits only the name lines, so re-add the
file's header comment by hand instead of redirecting straight over the file (`ab_check.sh` strips
`^#`, so a header-less file still *works* — you would just lose the machine-specificity warning).

**Configuration.** `pyproject.toml` sets `testpaths = ["tests"]` and `addopts = "-v --tb=short"`.
`tests/conftest.py` puts `<repo>` and `<repo>/scripts` on `sys.path` (anchored to `__file__`,
never to cwd), sets `collect_ignore` for the headline runner, provides the two shared fixtures
`sample_ohlcv_df` (120-row seeded OHLCV frame) and `tmp_json_file` (dict → temp JSON factory), and
installs the **repo-root hygiene instrument** (session hooks — see *Conventions → Tests must leave
the repo clean*). Its first statement after `import os` points `TRADER_LOG_DIR` at a per-session
temp dir so no test process writes the production `logs/trader.log` (*Known hygiene items* → 2).

**Single-file runs are not baseline-reproducible** — see *Known hygiene items* below.

---

## The dev-Mac baseline — `tests/baseline_failures.txt`

23 names (7 `ERROR`, 16 `FAILED`) across 10 files. **Every one is a missing-dependency outcome**,
reproduced by running each file standalone and reading the actual exception. On the Jetson and in
CI, where the full stack is installed, the suite state is recorded in `CLAUDE.md` § Running tests — this file is dev-Mac-only and must
never be ported.

| Missing dep | Names | Where | Why |
|---|---:|---|---|
| `pyarrow` / `fastparquet` | 12 | `test_data_utils` (2), `test_oi_archive` (2 ERROR + 4 FAILED), `test_wave4::TestShortFlow` (4) | pandas has no parquet engine on this Mac, so every parquet round-trip / archive read fails |
| `joblib` | 3 | `test_conviction_journal` (ERROR), `test_predict_now` (ERROR), `test_panel_ranks::TestBouncedLoser` (1) | all three reach `predict_now`, which imports `joblib` |
| `hmmlearn` | 3 | `test_new_modules::TestRegimeDetector::test_fit_hmm`, `::TestSharpeWithTxnCosts` (2) | `regime_detector` HMM fit; the Sharpe split consumes its regime labels |
| `torch` | 2 | `test_model_v2` (ERROR), `test_hypersearch_v2` (ERROR) | `model_v2` / `scripts/hypersearch_v2.py` import `torch` at module scope |
| `arch` | 2 | `test_new_modules::TestVolatility::test_fit_garch`, `::test_forecast_volatility` | `volatility` GARCH fit/forecast |
| `dotenv` | 1 | `test_asof_universe` (ERROR) | `stock_config` → `universe_selector` → `trading_utils` → `from dotenv import load_dotenv` |

The two `test_oi_archive` `ERROR` entries are *fixture*-level errors (the `archive` fixture calls
`pd.read_parquet`), not module-level import errors.

**`lightgbm`, `numba`, `optuna` and `sklearn` are also absent on this Mac but produce no baseline
name.** Those paths are guarded — `pytest.importorskip` (22 files use it), pure-python fallbacks
for the Numba kernels, or source-text-only tests that never import the module. Do not list them
as baseline causes.

---

## The standalone runner — `tests/test_sentiment_headlines.py`

Not a pytest module: it has **zero `def test_` functions** and `tests/conftest.py`'s
`collect_ignore` excludes it. Run it directly:

```bash
python3 tests/test_sentiment_headlines.py
```

- **Mac-runnable** (`sentiment.py` wraps `load_dotenv` in `try/except ImportError`; the rest is
  stdlib + `requests`). Last measured result: **1034/1035 (99.9 %)** — 1020/1021 scoring plus
  14/14 validation.
- Under test: `sentiment._score_text`, `_validate_text`, `_score_articles`, `_aggregate_scores`.
- It is a **statistical gate**, not per-case assertions: `main()` returns success iff the scoring
  pass-rate is ≥ 99.0 % *and* both validation suites are 100 %. That is why it cannot become a
  pytest module without being rewritten.
- Its cases are `HAND_WRITTEN` (121 hand-labelled headlines) + `TEMPLATES` (generated by nested
  loops over stock/verb/reason families) + 10 `VALIDATION_TESTS`.
- **It makes one real outbound HTTP attempt** via the `_score_articles` path (fail-soft: it prints
  `Fetched 0/1 article bodies` and continues). CI runs this file, so CI depends on that fetch
  failing gracefully. It writes nothing into the repo.

---

## Family map — which pass wrote which files

| Prefix | Files | Tests | Provenance |
|---|---:|---:|---|
| *(unprefixed, module-named)* — `core` | 88 | 1131 | The organic per-module suites, the oldest layer. Named after the module they cover (`test_indicators.py`, `test_portfolio.py`, …). `test_model_v2.py` / `test_hypersearch_v2.py` take their `v2` from the *module* name, not from a wave. |
| `test_c26_*.py` | 26 | 708 | **2026-08 comprehensive campaign** packets, lettered by wave: P1/P2/P4/P5, Q1–Q3, R1–R2, S1–S3, T1–T4/T6/T7, U1, V1–V4, W1, X1, plus `c26_base_loop_functional`. Committed in `20a41db`. Docs: `research/campaign_2026-08/`. |
| `test_review_bNN.py` | 23 | 531 | **2026-07 module review**, batches b01–b22 + `review_final`. Committed in `6bb38e7`. Owner queue: `research/module_review_2026-07.json`. |
| `*_v3.py` | 19 | 519 | **`module-improve-v3` panel pass** (N Opus reviewers → Fable spec → Sonnet implement → Fable harden), one file per module reviewed. Docstrings date the pass to 2026-07; first committed with the 2026-08 campaign (`20a41db`). |
| `test_r2c_*.py` | 8 | 311 | **R2 signal-model wave** (2026-08/09). Densest family (~39 tests/file). Currently **uncommitted**. |
| `test_llm*.py` | 11 (+1) | 233 (+22) | LLM client / analyst / eval / provider / routing suites, grown across passes. The `+1` is `test_llm_eval_v3.py`, produced by the v3 panel pass. |
| `test_grp_*.py` | 12 | 120 | **`group-improve-v2` campaign** (2026-07), one file per module *group*: data, deriv, exec, loops, macro, models, ops, reports, risk, sentiment, training, validation. Committed in `ca00b16`. |
| `test_ia1_*` … `test_ia4_*` | 4 | 111 | **2026-08 decision-influence audit**: ia1 removals (16), ia2 safety (24), ia3 gate-pricing (28), ia4 flagged behaviour (43). Spec: `research/campaign_2026-08/07_decision_influences.md` + `08_removed_code.md`. Currently **uncommitted**. |
| `test_improve_*.py` | 5 | 39 | An **earlier 2026-07 per-module improvement pass** ("spec stage 3"): harvest, indcfg, pbacktest, stratcfg, sweights. Landed in the same commit as the module review (`6bb38e7`). Note: this is *not* `module-improve-v3` — that family is the `_v3` **suffix** above. |
| `test_wave4.py` | 1 | 14 | The only surviving `waveN`-named file (wave 4: HAR-RV, JKX/HZZ/session/reversal features, FINRA short flow). Committed 2026-06-12. |
| `test_*_2026_09.py` — `2026_09` | 18 | 409 (625 collected) | **2026-09 Jetson test & improvement campaign** — suffix `_2026_09`. Producers: hunts G1–G8 + FIX_A–H / R1–R5, plus the INTEL W1 test-architecture file; docs in `research/campaign_2026-09_jetson/`. First family written *on the Jetson* against the full stack. Currently **uncommitted**. |

*Counts are `def test*` **definitions** counted by an AST walk — 198 files / 3739 definitions for
the rows above the `2026_09` row (that row and `test_repo_graph.py`, 4 definitions, were added on
2026-09-27; `core` still reads 88 files).
`@pytest.mark.parametrize` expands some of these at run time, which is why the collected count is
higher; see `CLAUDE.md` § Running tests for the run result.*

---

## Conventions

**Source-text contract tests.** 84 files call `read_text()` on a production module and assert on
its *source*. This is the dev-Mac workaround for modules that cannot be imported here
(`base_loop`, `stock_loop`, `predict_now`, `model_v2`, `scripts/hypersearch_v2.py` …): the test
pins a structural invariant — an ordering, a guard, a keyword being passed — without executing
the heavy import. 64 files anchor those reads with a `REPO` constant
(`Path(__file__).resolve().parent.parent`, or `.parents[1]` in three of them).

**Extract-and-exec.** Four files (`test_c26_P1.py` and peers, pattern from `test_review_b01.py`)
go further: they pull a single method out of an un-importable module with `ast`, `exec` it against
an injected globals dict of stubs, and call it. That gives real functional coverage of loop code
on the Mac. **Caveat, learned the hard way:** a *function-local* `from x import y` inside the
extracted method rebinds the name locally and the injected globals cannot intercept it — the real
module runs. Sandbox the real module's path constants in that case.

**Byte-pins for flag-OFF paths.** ~25 files assert that a flag-gated change is *byte-identical* to
the old path when its flag is OFF. Every model-facing change in this repo ships behind a
default-OFF flag; these tests are what make "default-OFF" verifiable rather than aspirational.
Never relax a byte-pin to make a new feature fit.

**Fingerprint / identity pins.** Where an artifact's identity matters (`shadow._manifest_fingerprint`,
`chart_core`'s repaint fingerprint), tests assert both stability (same input → same fingerprint)
and sensitivity (changed input → changed fingerprint). Both halves are required.

**Stubs are modernized, not weakened.** 27 files build fake modules with `types.ModuleType` and
15 install them into `sys.modules` to stand in for absent heavy deps. When production changes, the
stub must be updated to match the new real interface — never trimmed down or made permissive so
the assertion keeps passing. Same rule for `pytest.importorskip`: it marks a test as
Jetson-only, it is not a way to silence a failure.

**No cwd dependence.** Audited across all 198 files: zero bare `open('<relative>')`, zero
`pd.read_*('<relative>')`, zero `Path('<relative-literal>')`, zero `os.getcwd()` / `Path.cwd()`.
Everything is `tmp_path`-rooted or `__file__`-anchored, and every subprocess CLI run passes
`cwd=str(REPO)` explicitly. Production path literals are `Path(__file__)`-anchored too — which is
exactly why `monkeypatch.chdir` is the *wrong* tool for keeping a test's writes out of the repo,
and per-module path monkeypatching is the right one.

**Tests must leave the repo clean.** Any test that reaches a production write path monkeypatches
that module's path constant into `tmp_path` (`gpu_lock._LOCK_FILE`/`_INFO_FILE`,
`monitor_drift.BASE_DIR`, `adaptive_config.BASE_DIR`, `llm_client._COST_FILE`,
`backtest.BASE_DIR`, `oi_archive._LIVE_HISTORY_FILE`, `log_config._LOG_DIR`). To check a file you
touched: `touch /tmp/marker`, run it with `-p no:cacheprovider`, then
`find <repo> -maxdepth 1 -newer /tmp/marker -type f` must list nothing. Watch for lock sidecars
too — they are derived from the same constant at call time, so patching the constant catches both.

**The LLM cost ledger is sandboxed suite-wide (`tests/conftest.py`, 2026-09 INTEL W13).** An
autouse, function-scoped fixture, `_llm_cost_ledger_sandbox`, monkeypatches
`llm_client._COST_FILE` into a fresh, **existing** per-test directory under pytest's basetemp
(`<basetemp>/llm_cost_ledger/t*/`, deliberately *not* the test's `tmp_path`, so `tmp_path`
listings are unchanged) and resets `_daily_cost` / `_cost_reset_date` to `0.0` / `''`. That
redirects every ledger file, because each one is derived from the constant at call time:
`llm_cost.json`, its `.lock` and `.tmp` siblings, and the rollover's `llm_cost_history.jsonl`.
It also swaps in a fresh `llm_client._call_meta_tls` (`threading.local`, INTEL W19's per-attempt
transport meta), so `get_last_call_meta()` starts at `None` in every thread of every test and a
stubbed test never reads meta left by an earlier real-client test.
Everything goes through monkeypatch, so it is restored after each test. The ledger code
still runs for real, and writes land in the sandbox and read back. Request the fixture by name to
get the directory, or use `Path(llm_client._COST_FILE).parent`.

A test, a module fixture or a requested fixture that sets `_COST_FILE` itself runs *after* the
autouse fixture, so its value wins. `test_intel_ledger_sandbox_2026_09.py` pins that, plus
readback, an untouched production root, and no state leaking between tests.

The real module is captured when conftest is imported. That import is stdlib plus `llm_config`
only, with no file, network or subprocess activity. As a result, a stub placed in
`sys.modules['llm_client']` cannot divert the sandbox. If `llm_client` cannot be imported, the
fixture returns `None` and collection is unaffected.

Two limits:
- Only the in-process module object is covered. A spawned Python subprocess, or a re-imported
  fresh module object, is not; no test does either today.
- **Opting out:** no test needs the real root ledger (none found, 2026-09-27). A test that ever
  truly did would re-point `_COST_FILE` itself, via monkeypatch in the test body. That requires a
  review note, because it writes the production spend ledger.

**The repo-root hygiene instrument (`tests/conftest.py`, 2026-09).** Every pytest session now
snapshots the regular files directly in `<repo>/` and directly in `<repo>/tests/` (no recursion;
`__pycache__`, `.pytest_cache`, `*.pyc` and symlinks skipped) at session start and again at
session finish, and prints a terminal section just before the short test summary:

```
============================== repo-root hygiene ===============================
clean                               <- or one line per file:
NEW <path>                          <- created during the session
MOD <path>                          <- mtime or size changed during the session
```

Deleted files are ignored; subdirectories (`logs/`, `models/`, `journals/` …) are **out of scope**
of the NEW/MOD list; the production `logs/trader.log` has its own two closing lines instead
(`test logs: …`, `production log untouched: …` — hygiene item 2). The allowlist
(`_HYGIENE_ALLOWLIST`) is **empty** — an entry needs a file:line citation of a writer a test
legitimately must reach, and even then it prints as `ALW <path>` rather than disappearing.
Two env vars:

- `TRADER_TESTS_HYGIENE_JSON=<path>` — also write the result (`root`, `watched`, `clean`, `strict`,
  `exitstatus_forced`, `entries`) as JSON. Point it **outside** the repo; unset = nothing on disk.
- `TRADER_TESTS_STRICT_CLEAN=1` — **opt-in** enforcement: a dirty root turns an otherwise-passing
  session (`OK` / `NO_TESTS_COLLECTED`) into exit status **1** (`pytest.ExitCode.TESTS_FAILED`) and
  says so in the section. Unset (the default) = report-only; the exit status is never touched.

`scripts/ab_check.sh` echoes the section after its suite summary. Its verdict is still failure
**names** vs the baseline; only when the caller sets `TRADER_TESTS_STRICT_CLEAN=1` does a `NEW`/`MOD`
line also make it exit 1. **Read it under quiescence:** the snapshot sees the whole shared tree,
so a concurrent pytest or bot writing into the root shows up as this session's `MOD`/`NEW`.

**No test reads anything under `research/`.** The seven `research/…` references in `tests/` are all
docstring/comment pointers to the spec a file implements; moving `research/` breaks no test.

---

## Known hygiene items

**1. `sys.modules['dotenv']` leak — `test_c26_T2.py` / `test_c26_T3.py` (OWNER DECISION, do not
silently fix).** Both install a `dotenv` stub into `sys.modules` at module-import time and never
restore it. Every sibling that does this *does* restore — `test_c26_base_loop_functional.py`,
`test_c26_S3.py`, `test_c26_T6.py`, `test_ia2_safety.py` pop the key in a `finally`;
`test_decision_report.py` / `test_decision_report_v3.py` use `monkeypatch.setitem`. And
`test_decision_report_v3.py` states the broken invariant out loud: *other files
(`test_imports`, `test_gpu_lock`, `test_new_modules`) deliberately exercise the real dev-Mac
`ModuleNotFoundError` path and must see it regardless of what ran first.*

Consequences, each reproduced with `-p no:cacheprovider`:

| Run | Result |
|---|---|
| `pytest tests/test_gpu_lock.py` | 1 failed, 6 passed (`test_choose_inference_device_always_cpu`) |
| `pytest tests/test_c26_T2.py tests/test_gpu_lock.py` | 55 passed, 0 failed |
| `pytest tests/test_new_modules.py` | 10 failed |
| `pytest tests/test_c26_T2.py tests/test_new_modules.py` | 5 failed (= the 5 baseline names) |
| `pytest tests/test_wave4.py` | 5 failed |
| `pytest tests/test_c26_T2.py tests/test_wave4.py` | 4 failed (= the 4 baseline names) |

So **5 of the 23 baseline outcomes are what they are only because T2/T3 leak the stub earlier in
the session**, and `test_gpu_lock.py::test_choose_inference_device_always_cpu` currently asserts
against a stub rather than the real path it was written to check. `ab_check.sh` is unaffected in
practice (it always runs the whole directory in one order), but the suite is **not
order-independent**: `-p xdist`, `-k` and single-file runs give different failure sets. Fixing the
leak is objectively correct but would **add 5 names to `tests/baseline_failures.txt`** — it must
be one atomic change (fix + regenerate the baseline + re-run `ab_check.sh`) and is the owner's
call, not a cleanup-pass edit.
*2026-09 Jetson re-measurement* (dotenv blocked via a `sitecustomize` to mimic the Mac; T2/T3
fixed copies in scratch, not applied): in isolated pairs the leak masks **7** names, not 5 —
`test_new_modules.py::TestKelly` (3) + `::TestKellyScoping` (2), `test_gpu_lock.py::test_choose_inference_device_always_cpu`,
`test_wave4.py::TestWarmupFill::test_long_warmup_features_survive_dropna` — and, from reading
`test_imports.py:44-52` (not measured), dotenv-only skips there become passes. The exact full-suite delta must be taken on the Mac with `ab_check.sh`. The
Jetson and CI have `python-dotenv`, so the stub never installs there (zero blast radius).
*If the owner approves:* the copy-paste Mac procedure is
`research/campaign_2026-09_jetson/mac_dotenv_fix_runbook.md`. It applies both diffs verbatim, checks that the
delta is exactly the 7 names, regenerates the baseline with its header kept, runs `ab_check.sh`, updates the docs
and gives a rollback. Nothing has been applied. This item stays an OWNER DECISION until the owner runs it.

**2. `logs/trader.log` was written by the test suite — fixed 2026-09-27 (INTEL W18).**
`log_config._setup()` runs on the first `get_logger()` call, and 22 modules call it at *module*
scope, so merely importing any of them used to open the **production** `<repo>/logs/trader.log`
for append; fault-injection output then landed in the log the Jetson forensics and live bots use
(33,452 fake lines in one night) and counted against the 5 × 10 MB rotation budget. Now
`tests/conftest.py` sets `TRADER_LOG_DIR` at **module scope, right after `import os`** — before its
own `import llm_client` and every other repo import — to a per-session
`tempfile.mkdtemp(prefix='trader-test-logs-')`; `log_config._log_paths()` (`log_config.py:102-135`)
reads it at the first `_setup()`, so every record of the test process (and of subprocesses that
inherit the environment) goes there. A **non-empty** caller value wins; an empty one counts as
unset. The directory is left in `$TMPDIR`; the repo-root hygiene section ends with
`test logs: <dir> (TRADER_LOG_DIR, set by conftest|caller)` and
`production log untouched: yes | no (...) | unattributable (...)` — the session-start
(inode, mtime, size) of `logs/trader.log` vs finish; `unattributable` = it changed while other
processes (the live bots on the Jetson) held it open and this process has no handler on it.
Report-only: the exit status is never touched, and `ab_check.sh`'s block filter does not echo
these two lines. Per-test `_LOG_DIR`/`_LOG_FILE` monkeypatches (`test_review_b20.py`,
`test_g3_ops_2026_09.py`) still win over the variable. **Production is unchanged**: nothing outside
tests sets it, so bots and the pipeline log to `<repo>/logs/trader.log` exactly as before. Not
covered: a subprocess started with a scrubbed environment. Pinned by `test_intel_logdir_2026_09.py`.

**3. `llm_client._COST_FILE` is order-dependent.** `_maybe_reset_quota()` persists the shared spend
ledger on the first call of a new calendar day, and it is reached from six *read-shaped* functions
(`_cost_ok`, `get_daily_cost`, `get_budget`, `record_call`, `select_model`, `get_routing_info`).
Whichever unsandboxed test reaches one of them first becomes the writer of `<repo>/llm_cost.json`.
`test_c26_P1.py::test_analyze_trades_sets_last_analysis_meta` (today's first caller) now
monkeypatches the constant, as do `test_c26_S2.py`, `test_llm_claude.py` and
`test_llm_providers.py`; `test_llm_routing.py` closes the gate by setting `_cost_reset_date`.
A future test that reaches a cost getter without sandboxing will re-open the hole. **Re-opened
2026-09:** the hygiene instrument reports `MOD llm_cost.json.lock` for
`test_g4b_fixes_2026_09.py::test_analyze_trades_conviction_infinity_does_not_raise` (via
`llm_analyst.analyze_trades` → `get_routing_info` → `_maybe_reset_quota` → `_cost_file_lock`,
which opens `<repo>/llm_cost.json.lock` for write). Static candidates with no `_COST_FILE` /
`_cost_reset_date` sandbox: `test_llm_advice.py`, `test_llm_client.py`,
`test_llm_dossier_persist.py`, `test_llm_advisor.py`. **Closed 2026-09-27 (INTEL W13):** the suite-wide
autouse fixture `_llm_cost_ledger_sandbox` in `tests/conftest.py` (see *Conventions*) now repoints
`_COST_FILE` per test. The per-file copies of the fixture in the five `test_llm_*` files and in `test_g4b_fixes_2026_09.py` are
redundant but kept, because they are harmless and override it. Measured before the change: only `test_llm_batch_scoring.py` still
reached the root ledger (`get_recommended_model` → `_maybe_reset_quota`). Measured after: 0 root touches.

**4. Gitignore gaps for two sidecars — resolved.** `.gitignore` now lists `.gpu_lock_info.json` and
`*.json.lock` (checked 2026-09-27 with `git check-ignore -v`). Historical note: `.gpu_lock_info.json` (`gpu_lock`) and `llm_cost.json.lock`
(`llm_client`) were not ignored (`.gitignore` then covered only `.gpu.lock` and `*.jsonl.lock`).
`llm_cost.json.lock` does exist in the Jetson root — ignored now, but see item 3 for why.

**5. One genuinely environment-dependent test.** `test_c26_W1.py` pins module source via
`git show HEAD:<path>` — it needs a git worktree and compares against the **committed** version,
so with a large uncommitted tree it is testing HEAD, not what you just edited.

---

## Appendix — full per-file inventory

One row per `test_*.py`, grouped by family. **Mac** = `yes` (green on the dev Mac) / `partial`
(some tests `importorskip` a heavy dep) / `BASELINE` (has entries in `tests/baseline_failures.txt`)
/ `not collected`. **Modules under test** = repo modules the file imports *or* reads as source.
**#** = `def test*` definitions.

### Family `core` — unprefixed / module-named — the organic per-module suites (oldest layer)

*88 files, 1131 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_adaptive_config.py` | 40 | yes | `adaptive_config` | Tests for adaptive_config.py — edge detection, expansion, mode decisions. |
| `test_asof_universe.py` | 8 | BASELINE — dotenv (collection ERROR: stock_config -> universe_selector -> trading_utils -> `from dotenv import load_dotenv`) | `harvest_crypto_data`, `harvest_stock_data` | Tests for the as-of training universe (tradability + membership masks). |
| `test_backtest_q10.py` | 2 | yes | `backtest`, `strategy_config` | Gate/live parity: the backtest must apply the q10 tail veto. |
| `test_basis_archive.py` | 7 | yes | `basis_archive`, `funding_archive` | Wave-7 Finding 9: spot-perp basis archive (basis_archive). |
| `test_bet_sizing.py` | 7 | yes | `bet_sizing` | Wave-9 #5: edge/probability bet-sizing kernels. |
| `test_beta_ledger.py` | 9 | yes | `beta_ledger` | beta_ledger regression-core tests — synthetic data, pure numpy/pandas. |
| `test_blend_fit.py` | 7 | yes | `blend_fit` | Wave-9 #2: OOF stacked blend-weight selection (un-hardcode the 0.6/0.4). |
| `test_calibration.py` | 10 | yes | `calibration` | Wave-9 #1: leak-free meta-label calibration kernels. |
| `test_chart_core.py` | 79 | yes | `chart_core`, `gui` | Synthetic-data unit tests for chart_core.py — pure numpy/stdlib, runs on the dev Mac (no PySide6/pyqtgraph/pandas anywhere in this module). |
| `test_command_ack.py` | 13 | yes | — | Tests for run_pipeline.py's command-acknowledgement instrumentation. |
| `test_conviction_ab.py` | 6 | yes | `portfolio_backtest` | Wave-9 #4: conviction A/B evaluator + the multiple-testing deflation fix. |
| `test_conviction_journal.py` | 12 | BASELINE — joblib (collection ERROR: conviction_journal -> predict_now -> joblib) | `base_loop`, `decision_report`, `meta_label`, `strategy_config` | Tests for wave-5 Tier1-1 conviction instrumentation (measurement-only). |
| `test_cost_per_bar.py` | 4 | yes | `backtest`, `fees`, `liquidity`, `strategy_config` | Wave-6 Tier-1: per-bar effective-spread cost threaded into the backtest. |
| `test_cost_regime.py` | 9 | yes | `cost_regime` | Wave-6 Tier-2: cost-regime META features (cost_regime). |
| `test_crypto_panel_ranks.py` | 6 | yes | `panel_ranks` | Wave-9 #6: crypto cross-sectional rank + the soft, cost-neutral size tilt. |
| `test_crypto_trend.py` | 6 | yes | `crypto_trend` | Wave-9 #7: BTC trend / TSMOM risk-off gate (graded, debounced, fail-open). |
| `test_cs_neff.py` | 10 | yes | `backtest`, `portfolio_backtest`, `sample_weights` | Cross-sectional effective-n tests (2026-07 review). |
| `test_cscv_gate.py` | 7 | yes | `validation` | Wave-8 #2: the CSCV-PBO offline gate (build_oos_blocks + pbo_from_oos_blocks). |
| `test_data_sources.py` | 6 | yes | `data_sources` | Tests for data_sources.py — CryptoCompare fetch and fallback chain. |
| `test_data_utils.py` | 9 | BASELINE — pyarrow/fastparquet (pandas has no parquet engine) | `data_utils` | Tests for data_utils.py — Parquet I/O, CSV fallback, append, validation. |
| `test_decision_report.py` | 19 | yes | `decision_report`, `market_data`, `trading_utils` | Tests for gate attribution / conviction calibration replays. |
| `test_design_tokens.py` | 22 | yes | `design_tokens`, `gui` | design_tokens.py contract tests — pure stdlib, no PySide6 anywhere. |
| `test_edgar_events.py` | 10 | yes | `edgar_events` | Tests for the EDGAR 8-K item veto + M&A blacklist rules. |
| `test_execution_policy.py` | 9 | yes | `execution_policy` | Wave-7 Finding 1: calibrated entry-tactic table (execution_policy). |
| `test_fault_injection.py` | 13 | yes | `order_utils` | Fault-injection ("chaos") tests for the order seams. |
| `test_fees_feedback.py` | 8 | yes | `fees`, `trade_journal` | Tests for the realized maker-share fee feedback (LIVE gate only). |
| `test_fundamentals.py` | 18 | yes | `fundamentals` | Tests for fundamentals.py — formatting and caching. |
| `test_gap_audit.py` | 11 | yes | `gap_audit` | Wave-7: overnight forfeited-drift + GTC gap-through audit (gap_audit). |
| `test_gate_protocol.py` | 28 | yes | `backtest`, `hw_monitor`, `run_pipeline` | Tests for the backtest.py <-> run_pipeline.py gate exit-code protocol and the run_pipeline integrity fixes from the 2026-07 deep review: |
| `test_gpu_lock.py` | 7 | yes | `gpu_lock`, `trading_utils` | Tests for GPU lock coordination. |
| `test_gui_charts.py` | 25 | yes | `gui` | Chart-overhaul source contracts: pure source inspection, PySide6-free. |
| `test_gui_contracts.py` | 20 | yes | `beta_ledger`, `decision_report`, `gui`, `indicator_leadlag` | GUI source contracts (U1-U5): pure source inspection, PySide6-free. |
| `test_hw_monitor.py` | 11 | yes | `hw_monitor` | Tests for hw_monitor.py — GPU temp (sysfs), RAM usage, GPU availability. |
| `test_hypersearch_v2.py` | 15 | BASELINE — torch (collection ERROR: scripts/hypersearch_v2 -> torch) | — | Tests for hypersearch_v2.py — walk-forward CV, Sharpe computation, weighted loss. |
| `test_ic_diagnostic.py` | 4 | yes | `ic_diagnostic` | Wave-9 #3 gate: per-name IC diagnostic that decides universe promotion. |
| `test_imports.py` | 1 | partial — 7 skipped — heavy-dep import probes | — | Smoke test: every module except gui.py imports without error. |
| `test_indicator_config.py` | 13 | yes | `indicator_config` | Tests for indicator_config.py — preset management. |
| `test_indicator_leadlag.py` | 10 | yes | `indicator_leadlag`, `indicators` | indicator_leadlag tests — synthetic panels with planted lead/lag structure. |
| `test_indicator_reindex.py` | 4 | yes | `indicators` | Tests for indicators.py reindex behavior — btc_close and spy_close alignment. |
| `test_indicators.py` | 50 | yes | `indicators` | Tests for indicators.py — pure numeric functions. |
| `test_indicators_parity.py` | 12 | partial — 11 skipped — numba not installed (kernel-vs-fallback parity is Jetson-only) | `indicators` | Kernel-vs-fallback parity harness for indicators.py. |
| `test_ioc_helper.py` | 9 | yes | `order_utils`, `strategy_config` | Wave-7 Finding 2: marketable-IOC slippage-cap helper (order_utils). |
| `test_journal_stats.py` | 20 | yes | `journal_stats` | Synthetic-data unit tests for journal_stats.py — pure stdlib, runs on the dev Mac (no numpy/pandas/torch/etc. anywhere in this module or its target). |
| `test_liquidity.py` | 13 | yes | `fees`, `liquidity` | Wave-6 Tier-1: per-name EDGE effective-spread cost (liquidity.py). |
| `test_live_feature_parity.py` | 9 | yes | `indicator_config`, `indicators`, `market_data` | Live-path feature parity tests (2026-07 review P0). |
| `test_market_data.py` | 2 | yes | `market_data` | Tests for market_data.py — yfinance column flattening. |
| `test_market_impact.py` | 8 | yes | `liquidity`, `strategy_config` | Wave-8 #6: square-root market-impact cost term. |
| `test_model_v2.py` | 9 | BASELINE — torch (collection ERROR: model_v2 -> torch) | `model_v2` | Tests for model_v2.py — RegressionLSTM architecture. |
| `test_monitor_drift.py` | 18 | yes | `monitor_drift`, `trade_memory` | Tests for the PSI drift monitor. |
| `test_new_modules.py` | 38 | BASELINE — hmmlearn (regime_detector HMM fit) + arch (volatility GARCH) via the 5 baseline names; the 5 EXTRA standalone failures are dotenv and disappear in a full run (stub leak, see report) | `hypersearch_v2`, `indicators`, `log_config`, `model_lgb`, `order_utils`, `portfolio`, `regime_detector`, `strategy_config` +3 | Tests for new modules: volatility, macro_indicators, portfolio, regime_detector, model_lgb, types_mod, log_config, and indicator enhancements. |
| `test_notify.py` | 5 | yes | `notify` | Tests for notify.py kill-switch controls (halt/flatten flags, Telegram polling). |
| `test_novelty.py` | 10 | yes | `novelty` | Tests for the headline novelty filter (shingle-Jaccard staleness). |
| `test_oi_archive.py` | 18 | BASELINE — pyarrow/fastparquet (oi_archive parquet archive) | `oi_archive` | Tests for the open-interest archive + live features. |
| `test_options_overlay.py` | 17 | yes | `options_overlay` | Wave-7 flagship: free offline option-pricing + overlay decision harness. |
| `test_order_utils.py` | 16 | yes | `order_utils` | Tests for order_utils.py — pure computation functions. |
| `test_panel_ranks.py` | 17 | BASELINE — joblib (bounced-loser detector -> predict_now) | `indicators`, `market_data`, `panel_ranks`, `stock_loop` | Tests for cross-sectional panel ranks + ROD/periodicity features. |
| `test_parse_scores.py` | 18 | yes | `sentiment` | Tests for _parse_scores() and _parse_llm_json() — LLM response parsing edge cases. |
| `test_peak_equity_persistence.py` | 7 | yes | `base_loop`, `drawdown` | Wave-8 #4: the drawdown ladder must survive restarts. |
| `test_pipeline.py` | 9 | yes | `run_pipeline` | Tests for run_pipeline.py — schedule and phase-building logic. |
| `test_policy_exits.py` | 13 | yes | `meta_label`, `policy_exits`, `strategy_config` | Tests for policy_exits — the shared exit-stack kernel. |
| `test_portfolio.py` | 20 | yes | `portfolio` | Tests for the portfolio layer: variance-correct correlation tilt, equicorrelation book-risk cap, and the EWMA book-vol scalar. |
| `test_portfolio_backtest.py` | 9 | yes | `portfolio_backtest` | Wave-5 Tier1-2: cross-sectional A/B policy engine (portfolio_backtest). |
| `test_predict_now.py` | 11 | BASELINE — joblib (collection ERROR: predict_now -> joblib) | `model_v2`, `predict_now` | Tests for predict_now.py — path generation and model loading. |
| `test_prediction_cache.py` | 9 | yes | `predict_now`, `prediction_cache` | Wave-8 #5: bar-keyed prediction cache semantics. |
| `test_prediction_cache_context.py` | 4 | yes | `crypto_loop`, `stock_loop` | GUI review 2026-07 §5/§11 Phase 2.3 (producer side): prediction-cache decision-context enrichment (meta_p, conviction, regime, llm_gate, rank). |
| `test_repo_graph.py` | 4 | yes | `repo_graph` | Tests for scripts/repo_graph.py — one subprocess covers `--summary`, `--json`, `--check` (added 2026-09, not in the 88-file core count above). |
| `test_rank_gradient.py` | 5 | yes | `rank_gradient` | Wave-9 #4/#5 gate: rank-gradient Stage-0 verdict (holdout + live). |
| `test_risk_budget.py` | 17 | yes | `portfolio`, `risk_budget` | Wave-6 Tier-2: cross-book account risk cap + two-book equity simulator. |
| `test_risk_budget_gate1.py` | 6 | yes | `risk_budget` | Wave-8 #7: cross-book account stop-risk GATE-1 measurement. |
| `test_sample_weights.py` | 18 | yes | `sample_weights`, `validation` | Wave-6 Tier-1: average-uniqueness sample weights + effective-n DSR. |
| `test_scaler_slice_equiv.py` | 3 | yes | `predict_now` | Wave-8 #8: slicing the window BEFORE scaler.transform is bit-identical. |
| `test_sentiment.py` | 40 | yes | `sentiment` | Tests for sentiment.py — keyword scoring, text validation, article dedup, and LLM retry queue. |
| `test_sentiment_gate.py` | 13 | yes | `sentiment` | Tests for sentiment trade gating, Fear & Greed endpoints, and CNN FnG. |
| `test_sentiment_headlines.py` | 0 | not collected — excluded by tests/conftest.py:17 collect_ignore — standalone runner | `sentiment` | Comprehensive sentiment scoring test suite — 1000+ headlines. |
| `test_sentiment_history.py` | 13 | yes | `sentiment_history` | Tests for sentiment_history.py — FnG normalization and keyword scoring. |
| `test_sentiment_scoring.py` | 12 | yes | `sentiment` | Tests for LLM/KW article scoring — score_article_batch, try_llm_upgrade, _llm_score_chunk. |
| `test_shadow.py` | 11 | partial — 2 skipped — importorskip joblib | `shadow` | Tests for challenger shadow mode (DM-HLN test, evaluation, promotion). |
| `test_shadow_status_persist.py` | 9 | partial — 1 skipped — importorskip joblib | `shadow` | Tests for shadow-status persistence (Phase 2.2 producer side — research/gui_review_2026-07.md §7 challenger cell / promotion story). |
| `test_short_cost.py` | 14 | yes | `borrow_proxy`, `fees`, `short_cost` | Wave-7 Finding 5: regime-dated short cost + likely-shortable proxy. |
| `test_short_kernel.py` | 12 | yes | `policy_exits` | Wave-5 T1-4: offline side-aware short exit kernel (policy_exits). |
| `test_sizing_signal_instruments.py` | 6 | yes | `base_loop`, `decision_report`, `market_data` | 2026-07 decision-audit instruments: sizing decomposition + signal-exit counterfactual. |
| `test_squeeze_features.py` | 5 | yes | `squeeze_features` | Wave-7 Finding 10 (feature half): crypto squeeze interaction columns. |
| `test_stock_config.py` | 10 | yes | `stock_config` | Tests for stock_config.py — symbol universe management. |
| `test_tax_lots.py` | 24 | yes | `tax_lots` | Synthetic-data unit tests for tax_lots.py — pure stdlib (datetime + collections), runs on the dev Mac. No Alpaca/Qt/pandas anywhere. |
| `test_trade_journal.py` | 3 | yes | `trade_journal` | Tests for trade_journal.py — JSONL logging and summaries. |
| `test_trading_utils.py` | 9 | yes | `trading_utils` | Tests for trading_utils.py — cooldown and model mtime helpers. |
| `test_uniqueness_wiring.py` | 8 | yes | `model_lgb`, `sample_weights` | Wave-8 #1: fold_train_weights — the missing train-side uniqueness wiring. |
| `test_universe_promotion.py` | 7 | yes | `panel_ranks`, `stock_config` | Wave-9 #3: live tradable-universe promotion + sector-bucket completeness. |
| `test_validation_hardening.py` | 10 | yes | `validation` | Validation hardening: CSCV PBO + Lo-2002 serial-correlation factor. |

### Family `v3-panel` — `_v3` suffix — 2026-07 module-improve-v3 panel pass (one file per module reviewed)

*19 files, 519 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_backtest_v3.py` | 28 | yes | `backtest`, `fees`, `strategy_config` | Tests for backtest.py's 2026-07 instrumentation & hardening pass. |
| `test_base_loop_v3.py` | 26 | partial — 2 skipped — importorskip base_loop/crypto_loop (No module named 'joblib') | `base_loop`, `notify` | 2026-07 panel adjudication fixes for base_loop.py (module-improve-v3). |
| `test_bet_sizing_v3.py` | 17 | yes | `bet_sizing` | Panel campaign (module-improve-v3) Batch A: bet_sizing.py hardening. |
| `test_beta_ledger_v3.py` | 29 | yes | `beta_ledger` | beta_ledger v3 hardening tests — pure numpy/pandas synthetic data. |
| `test_calibration_v3.py` | 23 | yes | `calibration` | Adjudicated panel-review hardening tests for calibration.py (2026-07). |
| `test_cost_regime_v3.py` | 31 | yes | `cost_regime` | Pins the adjudicated 2026-07 cost_regime.py panel fixes (module-improve-v3). |
| `test_decision_report_v3.py` | 30 | yes | `base_loop`, `decision_report`, `market_data`, `stock_loop`, `strategy_config`, `trading_utils` | 2026-07b decision_report.py hardening tests. |
| `test_execution_policy_v3.py` | 33 | yes | `backtest`, `base_loop`, `crypto_loop`, `execution_policy`, `fees`, `order_utils`, `stock_config`, `stock_loop` +1 | Panel adjudication (2026-07, batch A) — execution_policy hardening pins. |
| `test_fees_v3.py` | 29 | yes | `backtest`, `decision_report`, `fees`, `hypersearch_v2`, `liquidity`, `llm_analyst`, `meta_label`, `order_utils` +2 | fees.py v3 contract tests — panel-adjudicated definition pins + guards. |
| `test_imports_v3.py` | 11 | partial — 2 skipped — importorskip lightgbm / alpaca | `model_lgb` | P0 verification-stack tests (campaign 2026-08, B16 core). |
| `test_liquidity_v3.py` | 28 | yes | `fees`, `harvest_stock_data`, `liquidity` | Panel-review v3 hardening tests for liquidity.py (2026-07). |
| `test_meta_label_v3.py` | 14 | yes | `hypersearch_v2`, `meta_label`, `strategy_config` | Panel v3 adjudicated spec — meta_label.py. Mac-runnable: numpy/pandas/stdlib only; heavy deps stubbed via the _paths/_read_artifacts seams (same pattern as tests/test_grp_models.py). |
| `test_order_utils_v3.py` | 40 | yes | `order_stream`, `order_utils` | Panel batch v3 — order_utils adjudicated fixes (keep-best maker evidence, NaN/crossed quote guard, stream pacing, bidirectional symbol variants, symbols-filtered listings, list_positions ... |
| `test_policy_exits_v3.py` | 55 | yes | `policy_exits`, `strategy_config` | Tests for policy_exits — panel batch B2 (SACRED KERNEL, docstring/comment edits only; zero executable-code changes). |
| `test_portfolio_backtest_v3.py` | 31 | yes | `backtest`, `portfolio_backtest`, `volatility` | module-improve-v3 batch B4: portfolio_backtest.py hardening + instrumentation. |
| `test_portfolio_v3.py` | 18 | yes | `market_data`, `portfolio` | portfolio.py panel-review v3: instrumentation, estimator sentinel, fail-closed guards. Pure numpy/pandas — Mac-runnable; _LedoitWolf is forced to False (corrcoef) by the autouse fixture s... |
| `test_risk_budget_v3.py` | 28 | yes | `portfolio`, `risk_budget` | Panel-review batch A (2026-07): risk_budget.py hardening locks — GATE-1 report honesty (rho bracket, missing/unknown/skewed books, ages), write-path hygiene (anchored path, bounded flock,... |
| `test_sample_weights_v3.py` | 30 | yes | `sample_weights` | Stage-3(v3) hardening batch for sample_weights.py. |
| `test_validation_v3.py` | 18 | yes | `validation` | Panel-review v3 regression tests for validation.py (adjudicated spec). |

### Family `review` — `review_bNN` — 2026-07 module review, 22 batches + a final sweep

*23 files, 531 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_review_b01.py` | 21 | yes | `crypto_loop`, `crypto_trend`, `stock_loop` | 2026-07 review batch b01: crypto_loop / stock_loop / crypto_trend fixes. |
| `test_review_b02.py` | 27 | yes | `execution_policy`, `fees`, `order_stream`, `order_utils` | Review batch b02 — order_utils / execution_policy / order_stream fixes. |
| `test_review_b03.py` | 38 | yes | `alpaca_compat`, `crypto_loop`, `trading_utils`, `types_mod` | 2026-07 review batch b03: trading_utils / alpaca_compat / types_mod fixes. |
| `test_review_b04.py` | 30 | yes | `llm_client`, `macro_calendar`, `novelty`, `sentiment` | 2026-07 review batch b04: sentiment / novelty / macro_calendar fixes. |
| `test_review_b05.py` | 21 | yes | `data_sources`, `data_utils`, `sentiment_history` | Review batch b05 — data_utils, data_sources, sentiment_history fixes. |
| `test_review_b06.py` | 10 | yes | `edgar_events`, `fundamentals` | Review-batch b06 tests — fundamentals.py + edgar_events.py fixes. |
| `test_review_b07.py` | 50 | yes | `adaptive_config`, `macro_indicators`, `regime_detector` | Review-batch b07 regression tests: macro_indicators, regime_detector, adaptive_config. |
| `test_review_b08.py` | 29 | yes | `funding`, `funding_archive`, `oi_archive` | Review batch b08: funding.py / funding_archive.py / oi_archive.py. |
| `test_review_b09.py` | 8 | yes | `short_flow`, `squeeze_features` | Review batch b09: short_flow diagnostics/robustness + squeeze_features docstring. |
| `test_review_b10.py` | 31 | yes | `backtest`, `cost_regime`, `fees`, `liquidity`, `meta_label`, `strategy_config`, `trade_journal` | Review batch b10 — fees.py / liquidity.py / cost_regime.py fixes. |
| `test_review_b11.py` | 29 | yes | `borrow_proxy`, `options_overlay`, `short_cost`, `stock_config` | Review batch b11 — regression pins for short_cost / borrow_proxy / options_overlay fixes. |
| `test_review_b12.py` | 18 | yes | `drawdown`, `portfolio`, `risk_budget` | Review batch b12: portfolio.py / risk_budget.py / drawdown.py fixes. |
| `test_review_b13.py` | 11 | yes | `bet_sizing`, `blend_fit` | Review batch b13: bet_sizing.py + blend_fit.py fixes. |
| `test_review_b14.py` | 12 | yes | `backtest`, `liquidity`, `meta_label`, `strategy_config` | Review batch b14 — meta_label.py fixes. |
| `test_review_b15.py` | 26 | yes | `ic_diagnostic`, `market_data`, `predict_now`, `shadow`, `validation` | Review batch b15 regression tests. |
| `test_review_b16.py` | 26 | yes | `decision_report`, `indicators`, `market_data`, `panel_ranks`, `policy_exits`, `rank_gradient`, `stock_config`, `strategy_config` | Review batch b16 — regression guards for policy_exits / panel_ranks / rank_gradient. |
| `test_review_b17.py` | 19 | yes | `backtest`, `execution_report`, `fees`, `gap_audit`, `trade_journal`, `volatility` | Review batch b17 — volatility.py, gap_audit.py, execution_report.py fixes. |
| `test_review_b18.py` | 29 | yes | `notify`, `trade_journal`, `trade_memory` | Review batch b18 — trade_journal.py, trade_memory.py, notify.py fixes. |
| `test_review_b19.py` | 24 | yes | `gpu_lock`, `hw_monitor`, `monitor_drift`, `trade_memory` | Review batch b19 — monitor_drift.py, gpu_lock.py, hw_monitor.py. |
| `test_review_b20.py` | 21 | yes | `data_utils`, `harvest_crypto_data`, `log_config`, `stock_config`, `wave6_stage0` | Review batch b20 — log_config, stock_config, scripts/wave6_stage0. |
| `test_review_b21.py` | 26 | yes | `cscv_audit`, `harvest_crypto_data`, `ic_by_name`, `ic_diagnostic`, `validation` | Review batch b21 — scripts/harvest_crypto_data.py, scripts/cscv_audit.py, scripts/ic_by_name.py. |
| `test_review_b22.py` | 21 | yes | `rank_gradient_report`, `reliability_report` | Review batch b22 — scripts/reliability_report.py, scripts/rank_gradient_report.py. |
| `test_review_final.py` | 4 | yes | `base_loop`, `market_data` | Close-out fixes from the 2026-07 module-review workflow (b23 P1s). |

### Family `grp` — `grp_*` — 2026-07 group-improve-v2 campaign, one file per module GROUP

*12 files, 120 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_grp_data.py` | 16 | yes | `data_utils`, `trade_journal`, `trade_memory` | Tests for the 'data' review group — data_sources, data_utils, trade_journal, trade_memory. Focused on GAPS not already covered by test_data_utils.py / test_data_sources.py / test_trade_jo... |
| `test_grp_deriv.py` | 7 | yes | `basis_archive`, `funding`, `funding_archive`, `oi_archive`, `short_flow` | Behaviour-neutral scout fixes for the funding/OI/basis/short-flow group. |
| `test_grp_exec.py` | 7 | yes | `order_utils` | Tests for the order-path group's 2026-07 review edits (order_utils.py): |
| `test_grp_loops.py` | 9 | partial — 3 skipped — importorskip base_loop/crypto_loop (joblib) | `base_loop`, `crypto_trend`, `drawdown`, `macro_calendar`, `notify`, `order_utils`, `stock_loop` | Loop-group (base/crypto/stock/crypto_trend) design+scout fixes, 2026-07. |
| `test_grp_macro.py` | 13 | yes | `macro_indicators`, `volatility` | Group-macro doc/comment hygiene regression tests. |
| `test_grp_models.py` | 16 | yes | `bet_sizing`, `calibration`, `meta_label`, `model_lgb` | Models-group review pins — meta cache contract, atomic saves, degenerate guards, fail-closed sizing. Mac-runnable: lightgbm/joblib are stubbed via the _read_artifacts seam or pinned by so... |
| `test_grp_ops.py` | 4 | yes | `execution_report`, `fees`, `hw_monitor`, `monitor_drift`, `stock_config` | Tests for the 2026-07 GRP-ops behavior-neutral fixes: monitor_drift's load_holdout_hit_rate AttributeError escape, hw_monitor's get_ram_usage OSError contract, execution_report's fees-der... |
| `test_grp_reports.py` | 21 | yes | `cscv_audit`, `ic_by_name`, `rank_gradient_report`, `reliability_report` | Gate-Report scripts: robustness / verdict-clarity / arg-validation tests. |
| `test_grp_risk.py` | 4 | yes | `fees`, `liquidity`, `risk_budget`, `short_cost` | Cost/risk-kernel group (fees/liquidity/cost_regime/short_cost/borrow_proxy/ portfolio/risk_budget/drawdown) design+scout locks, 2026-07. Mac-green. |
| `test_grp_sentiment.py` | 9 | yes | `edgar_events`, `events_calendar`, `sentiment`, `sentiment_history` | Group tests for the sentiment/events review group (S1-S12 changes). |
| `test_grp_training.py` | 8 | yes | `adaptive_config`, `harvest_crypto_data`, `hypersearch_v2`, `wave6_stage0` | Source-contract pins for the training group (2026-07 design pass): scripts/hypersearch_v2.py, scripts/harvest_crypto_data.py, scripts/wave6_stage0.py. |
| `test_grp_validation.py` | 6 | yes | `backtest`, `hypersearch_v2`, `panel_ranks`, `predict_now`, `shadow` | Cross-module tripwires for the validation/shadow/panel group. |

### Family `improve` — `improve_*` — earlier module-improvement pass

*5 files, 39 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_improve_harvest.py` | 5 | yes | `harvest_stock_data` | Tests for the 2026-07 harvest_stock_data cleanup (spec stage 3). |
| `test_improve_indcfg.py` | 11 | yes | `indicator_config` | Tests for indicator_config.py behavior-neutral improvements (2026-07): atomic save, defensive-copy get_preset_features, and existing invariants (dup-free presets, disjoint only-cols, lean... |
| `test_improve_pbacktest.py` | 6 | yes | `portfolio_backtest` | Stage-3 improvement tests for portfolio_backtest.py: |
| `test_improve_stratcfg.py` | 10 | yes | `backtest`, `base_loop`, `crypto_loop`, `meta_label`, `portfolio`, `predict_now`, `stock_loop`, `strategy_config` | Stage-3 improvement batch — strategy_config.py comment/docstring pins. |
| `test_improve_sweights.py` | 7 | yes | `sample_weights` | Stage-3 improvement batch for sample_weights.py. |

### Family `c26` — `c26_*` — 2026-08 comprehensive campaign packets (P/Q/R/S/T/U/V/W/X waves)

*26 files, 708 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_c26_P1.py` | 44 | yes | `base_loop`, `llm_analyst`, `notify`, `order_utils`, `stock_loop`, `trade_journal`, `trading_utils` | c26 campaign packet P1 — live-engine safety + journal integrity (B08+B09). |
| `test_c26_P2.py` | 24 | yes | `monitor_drift`, `notify`, `portfolio_backtest`, `rank_gradient`, `rank_gradient_report` | Campaign Wave A / packet P2 — measurement-stack integrity. |
| `test_c26_P4.py` | 9 | yes | `events_calendar`, `strategy_config` | P4 (D07/B10) — earnings trading-day windows, default-OFF flag. |
| `test_c26_P5.py` | 14 | yes | `funding`, `funding_archive`, `sentiment_history` | c26-P5: funding.py time-thinning flag (D28) + sentiment_history.py incremental refresh (D27 dark half). |
| `test_c26_Q1.py` | 29 | yes | `adaptive_config`, `backtest`, `sample_weights`, `validation` | Packet c26_Q1 (D02/B03): effective-n + selection-pressure accounting. |
| `test_c26_Q2.py` | 19 | yes | `backtest`, `meta_label`, `run_pipeline`, `shadow`, `strategy_config` | Packet Q2 (campaign 2026-08, defect D03) — challenger-targeted policy gate. |
| `test_c26_Q3.py` | 27 | yes | `market_data`, `shadow` | Packet Q3 — Shadow DM v2 (D34 rebuild): per-hour collapse, IM cluster t, scheduled looks, frozen/patched skill-score variance. Mac-runnable (numpy/pandas only). Flag OFF must be byte-iden... |
| `test_c26_R1.py` | 25 | yes | `calibration`, `meta_label`, `shadow`, `strategy_config` | Packet R1 — calibration mechanics v2 (D13c, default-OFF CALIBRATION_V2), ungated-publish closure (D13a/D13b, direct-ship safety), and the shadow promote pre-flight (Q2 hook, under existin... |
| `test_c26_R2.py` | 21 | yes | `fees`, `meta_label`, `policy_exits`, `strategy_config` | Packet R2 (2026-08 campaign): OOF primary persistence (D12) + meta replay parity (D05-meta). |
| `test_c26_S1.py` | 13 | yes | `llm_client`, `llm_eval`, `prompt_ab` | c26 packet S1: llm_eval inference rebuild + journal-consumer integrity + spend ledger (D09 + D33-consumer + B07). |
| `test_c26_S2.py` | 12 | yes | `llm_client`, `llm_config` | Campaign 2026-08 packet S2 — LLM spend engineering (B07). |
| `test_c26_S3.py` | 37 | yes | `base_loop`, `macro_indicators`, `market_data`, `portfolio`, `sizing_cofire_report`, `strategy_config`, `types_mod`, `volatility` | c26 packet S3 — de-risk multiplier stack consolidation (D10 + D29 + B06). |
| `test_c26_T1.py` | 30 | yes | `adaptive_config`, `blend_fit`, `fees`, `hypersearch_v2`, `objective_utils`, `strategy_config` | Packet T1 — model-fit honesty (D22/D23/D24-part/D25/D05-threshold, B12). |
| `test_c26_T2.py` | 48 | yes | `data_sources`, `data_utils`, `funding_archive`, `harvest_crypto_data`, `harvest_stock_data`, `indicators`, `market_data`, `oi_archive` +4 | Packet T2 tests — data-store integrity (D39 sidecar, D08 provenance, B15 merge guard/exit codes, D38 closed-bar enforcement). |
| `test_c26_T3.py` | 37 | yes | `beta_ledger`, `cost_regime`, `crypto_spread_census`, `harvest_crypto_data`, `harvest_stock_data`, `indicator_leadlag`, `indicators`, `liquidity` +4 | Packet T3 tests — cost truth (B05 + B21 + D40). |
| `test_c26_T4.py` | 25 | yes | `backtest`, `ic_diagnostic`, `portfolio_backtest`, `predict_now`, `stage0_preds` | Packet T4 (campaign 2026-08, B02): Stage-0 predictions dump + hourly MTM equity + blend-leg persistence. |
| `test_c26_T6.py` | 43 | yes | `alpaca_compat`, `base_loop`, `fees`, `order_stream`, `order_utils`, `stock_loop`, `trade_journal`, `types_mod` | c26 packet T6 — execution quality (D20 + D21 + maker-share truth + stream stops + latency). |
| `test_c26_T7.py` | 37 | yes | `execution_report`, `journal_stats`, `monitor_drift`, `notify`, `options_overlay`, `run_bots`, `run_pipeline`, `shadow` +2 | c26 packet T7 — ops/measurement remainder tests. |
| `test_c26_U1.py` | 25 | yes | `beta_ledger`, `chart_core`, `decision_report`, `execution_report`, `gui`, `indicator_leadlag`, `llm_eval` | Packet c26 U1 — GUI decision truth + measurement reconciliation (B22). |
| `test_c26_V1.py` | 20 | yes | `llm_config`, `llm_qualify` | c26 packet V1: Free-LLM qualification harness (scripts/llm_qualify.py) + inert FREE_CANDIDATE_PRESETS registry in llm_config.py. |
| `test_c26_V2.py` | 31 | yes | `learned_lexicon`, `sentiment_history`, `train_lexicon` | Packet V2 tests — learned sentiment lexicon (learned_lexicon.py + scripts/train_lexicon.py). |
| `test_c26_V3.py` | 32 | yes | `meta_curve`, `meta_label`, `meta_learning_curve` | c26 packet V3 — meta-label learning-curve harness (B04.3), Mac-runnable. |
| `test_c26_V4.py` | 10 | yes | `validation` | Campaign 2026-08 packet V4 — stationary-bootstrap Sharpe diagnostic. |
| `test_c26_W1.py` | 34 | yes | `indicators`, `market_data`, `volatility` | Wave C-3 packet W1 tests — daily-bars cache + stock feature restoration (D11) + HAR-RV daily feed (D30). |
| `test_c26_X1.py` | 25 | yes | `backtest`, `base_loop`, `indicators`, `market_data`, `meta_label`, `notify`, `order_utils`, `panel_ranks` +6 | c26 final bug-hunt wave (X1) — cross-packet fixes. |
| `test_c26_base_loop_functional.py` | 37 | yes | `base_loop`, `drawdown`, `market_data`, `meta_label`, `monitor_drift`, `notify`, `order_utils`, `portfolio` +4 | c26 packet P6 — base_loop/stock_loop FUNCTIONAL suite (B16 slice). |

### Family `r2c` — `r2c_*` — R2 signal-model wave (2026-08/09, uncommitted)

*8 files, 311 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_r2c_blend_coherence.py` | 39 | yes | `backtest`, `blend_fit`, `hypersearch_v2`, `model_lgb`, `objective_utils`, `predict_now`, `strategy_config` | Packet R2C-02 — blend/certificate coherence (H1, H2, M2, M4-guard, L4, L9). |
| `test_r2c_fee_sweep.py` | 26 | yes | `backtest`, `fees`, `strategy_config` | R2C-08 (FR-16): breakeven fee-multiplier sweep on the policy replay. |
| `test_r2c_holdout_boundary.py` | 41 | yes | `hypersearch_v2`, `objective_utils`, `strategy_config`, `window_ab` | R2C-05 (2026-08 R2-C wave): FR-01 fixed-calendar holdout boundary + FR-02 training-window cutoff / A/B runner. |
| `test_r2c_lgb_refit.py` | 30 | yes | `hypersearch_v2`, `objective_utils`, `strategy_config` | Packet R2C-03 — LGB full refit + honest q10 floor (M3, M4-floor). |
| `test_r2c_measurement_kernels.py` | 40 | yes | `entry_timing_probe`, `funding_drift_audit`, `harvest_stock_data`, `horizon_transfer`, `horizon_transfer_report`, `naive_baseline`, `naive_vs_blend`, `stage0_preds` | R2C-06 measurement kernel suite tests (FR-04, FR-07-A, FR-03, M1, L7). |
| `test_r2c_rankic_ledger.py` | 56 | yes | `adaptive_config`, `hypersearch_v2`, `objective_utils`, `retrain_ledger` | R2C-07 (2026-08 R2-C wave): FR-05 cross-sectional rank-IC certificate lines + FR-08 retrain-gain ledger. |
| `test_r2c_serving_cache.py` | 34 | yes | `base_loop`, `predict_now`, `serving_cache`, `shadow` | R2C-01 (H3) — serving booster-cache integrity tests. |
| `test_r2c_training_repairs.py` | 45 | yes | `hypersearch_v2`, `objective_utils`, `run_pipeline`, `strategy_config` | R2C-04 (2026-08 R2-C wave): training-loop repairs + seed plumbing + config truth — L1, L2, L3, L5, L6, L8. |

### Family `ia1` — `ia1_*` — decision-influence audit: removals

*1 files, 16 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_ia1_removals.py` | 16 | yes | `base_loop`, `indicator_config`, `indicators`, `macro_indicators`, `sentiment`, `stock_loop`, `types_mod` | IA-1 influence-implementation removals (2026-08-22, owner-ruled wave). |

### Family `ia2` — `ia2_*` — decision-influence audit: safety

*1 files, 24 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_ia2_safety.py` | 24 | yes | `alpaca_compat`, `base_loop`, `macro_calendar`, `notify`, `order_utils`, `stock_loop`, `trading_utils`, `types_mod` | IA-2 — undesigned-behavior + safety fixes (2026-08 influence audit). |

### Family `ia3` — `ia3_*` — decision-influence audit: gate pricing

*1 files, 28 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_ia3_gate_pricing.py` | 28 | yes | `base_loop`, `decision_report`, `edgar_events`, `events_calendar`, `macro_calendar`, `macro_indicators`, `market_data`, `notify` +8 | IA-3 — price the unpriced gates (2026-08 influence audit, open question #4: "the highest-impact gates are the least priced"). All measurement-only: no gate's admit/veto behavior changes; ... |

### Family `ia4` — `ia4_*` — decision-influence audit: flagged behaviour

*1 files, 43 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_ia4_flagged.py` | 43 | yes | `base_loop`, `crypto_loop`, `edgar_events`, `events_calendar`, `macro_indicators`, `market_data`, `order_utils`, `portfolio` +5 | IA-4 — influence-audit flag family (2026-08 decision-influence ledger, research/campaign_2026-08/07_decision_influences.md). Seven flag-gated structural changes, each default-OFF with fla... |

### Family `llm` — `llm*` — LLM client/analyst/eval/provider suites

*11 files, 233 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_llm_advice.py` | 37 | yes | `events_calendar`, `llm_analyst`, `llm_config`, `llm_eval`, `prompt_ab` | Tests for the LLM conviction-gate rich-context + offline prompt A/B harness additions: |
| `test_llm_advisor.py` | 65 | yes | `events_calendar`, `fees`, `llm_analyst`, `llm_config`, `llm_eval`, `macro_calendar`, `trade_journal` | Tests for the advisor-v2 decision dossier (llm_analyst.py) and its measurement harness (llm_eval.py). |
| `test_llm_analyst.py` | 30 | yes | `llm_analyst` | Tests for llm_analyst.py — response parsing and prompt building. |
| `test_llm_batch_scoring.py` | 8 | yes | `sentiment` | Tests for _llm_score_batch() — tiered scoring with gap-fill and model tagging. |
| `test_llm_claude.py` | 13 | yes | `llm_client` | Anthropic (Claude) support in llm_client — 2026-07. |
| `test_llm_client.py` | 24 | yes | `llm_client` | Tests for llm_client.py — Gemini-only LLM client with quota tracking. |
| `test_llm_config.py` | 4 | yes | `llm_config` | Tests for llm_config.py — configuration loading with defaults. |
| `test_llm_dossier_persist.py` | 11 | yes | — | Tests for llm_analyst.py's Phase 2.1 fix (2026-07 GUI review §5/§11): the advisor-v2 decision dossier (p_up, conviction, abstain, key_risks, event_flags) computed by analyze_trades used t... |
| `test_llm_eval.py` | 9 | yes | `llm_eval` | Wave-8 #3: the incremental-over-pred LLM statistics. |
| `test_llm_providers.py` | 20 | yes | `llm_client` | Multi-provider selection engine — 2026-07. |
| `test_llm_routing.py` | 12 | yes | `llm_client` | Tests for LLM smart routing, tier detection, and tier-aware budgets. |

### Family `llm/v3-panel` — `llm_eval_v3` — LLM suite produced by the v3 panel pass

*1 files, 22 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_llm_eval_v3.py` | 22 | yes | `llm_eval` | Tests for the llm_eval.py v3 hardening pass: |

### Family `wave` — `waveN` — legacy research-wave suites

*1 files, 14 tests*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_wave4.py` | 14 | BASELINE — pyarrow/fastparquet (short_flow FINRA parquet archive) | `harvest_stock_data`, `indicators`, `short_flow`, `volatility` | Tests for wave 4: HAR-RV, feature suite, shorting flow, sleeve blend. |

### Family `2026_09` — `_2026_09` suffix — 2026-09 Jetson test & improvement campaign

*20 files, 697 collected tests (453 `def test*`). Unlike the rest of this appendix, **#** here is the
`pytest --collect-only` count on the Jetson (parametrize expanded). **Mac** = `claimed` (docstring
says Mac-safe, not re-run on the Mac), `partial` (module-level `importorskip`), `unverified`. Producers:
hunts G1–G8 + FIX_A–H / R1–R5, docs in `research/campaign_2026-09_jetson/`.*

| File | # | Mac | Modules under test | Purpose |
|---|---:|---|---|---|
| `test_exec_fixes_2026_09.py` | 21 | claimed | `llm_analyst`, `order_utils`, `trading_utils` | Execution fixes (hunt G2-1, G2-3, G1 F3); stub broker objects only, no network/orders/LLM calls. |
| `test_g3_ops_2026_09.py` | 43 | partial — importorskip run_pipeline | `crypto_loop`, `log_config`, `run_bots`, `run_pipeline`, `stock_loop` | G3 ops & orchestration fixes; every file a test writes is redirected into a tmp dir. |
| `test_g4_fixes_2026_09.py` | 11 | claimed | `llm_client`, `llm_config`, `novelty`, `sentiment`, `sentiment_history` | G4 hunt, batch 1: LLM + sentiment-layer robustness (e.g. per-window SQLite commits in sentiment history). |
| `test_g4b_fixes_2026_09.py` | 46 | claimed | `llm_analyst`, `llm_config`, `sentiment`, `sentiment_history`, `volatility` | G4 hunt, batch 2: non-finite fundamentals/LLM fields coerced safely; retry-queue race. Used to write `<repo>/llm_cost.json.lock` (hygiene item 3, closed by the suite-wide sandbox). |
| `test_g5_fixes_2026_09.py` | 61 | partial — importorskip arch, pyarrow | `edgar_events`, `events_calendar`, `funding`, `funding_archive`, `macro_indicators`, `oi_archive`, `short_flow`, `stock_config` +1 | G5 risk / non-model-gate fixes (HAR-RV truncated-day merge, funding/OI/short-flow/events guards). |
| `test_g6_fixes_2026_09.py` | 29 | partial — importorskip pyarrow | `decision_report`, `execution_report`, `llm_eval`, `market_data`, `data_utils`, `wave6_stage0` + 8 report scripts | G6 measurement-shelf fixes (llm_eval bar windows, report-script correctness). |
| `test_g8_fixes_2026_09.py` | 69 | claimed | `gui`, `strategy_config`, `tax_lots` | G8 operator-console fixes; gui pieces tested as pure helpers / AST-extracted methods (no PySide6). |
| `test_gui_fixes_2026_09.py` | 44 | partial — importorskip pandas | `gui` | GUI fixes from the headless audit (F_gui D1/D2/D3/D4/D5/D10), PySide6-free. |
| `test_jetson_ops_2026_09.py` | 36 | partial — importorskip fundamentals | `run_pipeline` (+ `scripts/setup_jetson_system.sh` as text) | Jetson ops audit fixes as source/AST contracts. |
| `test_llm_fixes_2026_09.py` | 96 | unverified | `llm_analyst`, `llm_client` | FIX_G LLM transport/accounting fixes (Anthropic temperature gating and peers); sandboxes `_COST_FILE`. |
| `test_loop_fixes_2026_09.py` | 23 | claimed | `base_loop`, `order_utils`, `run_bots`, `stock_loop` | Live-loop fixes (G1 F1/F2/F4, G2-3, G2-4) via extract-and-exec of the real loop source. |
| `test_market_data_sip_clamp_2026_09.py` | 30 | claimed | `market_data` | Stock historical harvest clamps `end` outside the 15-min SIP delay (Alpaca Basic plan). |
| `test_measurement_fixes_2026_09.py` | 30 | claimed | `beta_ledger`, `chart_core`, `decision_report`, `execution_report`, `gui`, `llm_eval`, `sizing_cofire_report` | D_measurement fixes: honest drop counts / replay windows, vendor spike filtering in beta_ledger. |
| `test_oi_inf_2026_09.py` | 23 | partial — importorskip joblib, pyarrow, torch | `oi_archive`, `harvest_crypto_data`, `data_utils`, `indicator_config`, `hypersearch_v2`, `predict_now` | FIX-R1: `OI_Chg_24h` +inf rows (zero-OI glitch prints) + the non-finite parity guard. |
| `test_raw_sidecar_reload_2026_09.py` | 12 | claimed | `data_utils`, `harvest_crypto_data`, `harvest_stock_data` | R5: raw-OHLCV sidecar reload hands `compute_features` a tz-aware UTC DatetimeIndex. |
| `test_sentiment_pit_2026_09.py` | 25 | claimed | `harvest_stock_data`, `sentiment_history` | PIT repair of Daily_Sentiment (FnG publication-date migration, marker-keyed, one-time). |
| `test_tb_restamp_2026_09.py` | 14 | claimed | `harvest_crypto_data`, `harvest_stock_data`, `panel_ranks`, `policy_exits`, `strategy_config` | R4: triple-barrier labels stamped AFTER every row filter (L7 full fix; TB-GUARD re-harvest finding). |
| `test_intel_testarch_2026_09.py` | 12 | claimed (pytest + stdlib; loads conftest → numpy/pandas) | `tests/conftest.py` (hygiene hooks), `scripts/ab_check.sh` | INTEL W1: repo-root hygiene instrument — pure helpers, subprocess mini-project NEW/MOD/strict, ab_check echo + strict passthrough. |
| `test_evidence_reads_2026_09.py` | 47 | claimed | `scripts/evidence_reads` (stubbed subprocess instruments) | INTEL R1: one-command runbook Phase 0/1 evidence reads — step table, parsers, readiness verdicts, exit codes, dry-run. |
| `test_intel_gui_2026_09.py` | 25 | claimed | `gui` (AST-extracted, PySide6-free) | INTEL R1: Cockpit/Trading `_scroll_wrap` + indexOf(page) invariant; off-thread optuna best-score load (cache key, routing). |
| `test_intel_ledger_sandbox_2026_09.py` | 10 | claimed (pytest + stdlib; loads conftest → numpy/pandas) | `tests/conftest.py` (`_llm_cost_ledger_sandbox`), `llm_client` | INTEL W13: suite-wide LLM cost-ledger sandbox. Real ledger writes land in the sandbox and read back; the production-root ledger is unchanged; per-test isolation; a test's own `_COST_FILE` wins; conftest is a no-op without llm_client (subprocess mini-project). |
| `test_intel_logdir_2026_09.py` | 10 | claimed (pytest + stdlib; loads conftest → numpy/pandas) | `tests/conftest.py` (`TRADER_LOG_DIR` block, production-log guard, sandbox meta reset), `log_config` | INTEL W18: test logging goes to the per-session `trader-test-logs-*` dir, never `<repo>/logs/trader.log` — effective dir, source-order guard, production-log snapshot/verdict, a real record lands in the temp file, subprocess mini-project (env set before the first repo import; caller value wins; `production log untouched:` line); per-test `get_last_call_meta()` reset. |
