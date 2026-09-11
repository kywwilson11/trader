# scripts/ — CLIs that are not part of the live trading path

Everything here is invoked as `python scripts/<name>.py` (or `bash scripts/<name>.sh`) **from the
repo root**. All 20 `.py` scripts insert the repo root on `sys.path` themselves
(`sys.path.insert(0, BASE_DIR)`), so they import the root modules directly; they are still written
to be run from the root because most default their inputs/outputs to `BASE_DIR`-relative paths.
`scripts/__pycache__/` is gitignored.

Three families live here:

- **Pipeline phases** — `harvest_*_data.py`, `hypersearch_v2.py`. Spawned by `run_pipeline.py`.
- **Measurement / research CLIs** — everything else `.py`. Offline, no orders, no live-path writes.
  Most print to stdout; the few that persist say so in the table.
- **Shell ops** — `ab_check.sh` (regression gate), `setup.sh` / `setup_jetson_system.sh` (install),
  `backup_state.sh` (prod backup).

**Machine column** (see `CLAUDE.md` § two-machine reality): *Jetson* means it imports a heavy dep
(torch / lightgbm / optuna / joblib / sklearn / dotenv), reads parquet (needs `pyarrow`), or needs
real journals / API keys — none of which exist on the dev Mac. *Mac* means pure
numpy/pandas/scipy/stdlib and it runs here today. *both* means the imports are Mac-safe but you
must supply a real input file that is produced on the Jetson.

## Inventory (20 `.py` + 4 `.sh`)

| Script | Goal | Machine | Key flags (verified against `add_argument`) | Reads → writes | Serves |
|---|---|---|---|---|---|
| `harvest_crypto_data.py` | 1Y hourly crypto OHLCV + features + triple-barrier labels | Jetson (`dotenv` unguarded) | *none* | Alpaca/yfinance/CryptoCompare → `training_data.parquet` + `.csv` (`data_utils.save_training_data`) | `run_pipeline` harvest phase |
| `harvest_stock_data.py` | Same for the US stock book (+ `panel_ranks`, `short_flow`) | Jetson (`dotenv` unguarded) | *none* | Alpaca/yfinance → `stock_training_data.parquet` + `.csv` | `run_pipeline` harvest phase |
| `hypersearch_v2.py` | Optuna TPE search over the LSTM + LightGBM legs, purged walk-forward + holdout DSR gate | Jetson (torch/optuna/sklearn/joblib unguarded) | `--trials --fresh --data --prefix --shadow --preset --max-rows --mode --no-status` | `{stock_}training_data.csv` → `{p}model_v2.pth`, `{p}config_v2.pkl`, `{p}scaler_v2.pkl`, `{p}feature_cols_v2.pkl`, `{p}lgb_model.txt`, `{p}model_v2.manifest.json` (written LAST), `{p}v2_study.db`, `pipeline_status.json` | `run_pipeline` train phase; Chain-2 of `06_signal_model_plan` |
| `window_ab.py` | Fixed-config training-window A/B (`full,730,365,pt2007`) at identical `cum_trials`/seed — no selection pressure | Jetson (torch/joblib) | `--prefix --data --preset --max-rows --arms --seed --cum-trials --epochs --out` | `{stock_}training_data.csv` → `window_ab_summary_<book>.json` | `06_signal_model_plan` step 9 (FR-02) |
| `naive_vs_blend.py` | Deployed blend vs the Nagel naive one-liner on identical Stage-0 rows (falsification baseline) | Jetson (needs the Stage-0 dump + parquet closes) | `--preds --prefix --half-life --vol-window --cum-trials` | `{slot}_stage0_preds.json` + training data → stdout | `06_signal_model_plan` step 3 (FR-04) |
| `funding_drift_audit.py` | PSI / KS + IC-sign regime audit of the `Funding_*` features before a retrain | Jetson (parquet) | `--split --trailing-days --fwd-bars --out` | training parquet → `research/funding_drift_2026-08.json` | `06_signal_model_plan` step 5 (FR-03) |
| `entry_timing_probe.py` | IC against both label anchors + realized fill-gap join from the journals | Jetson (journals + parquet) | `--preds --prefix --journal-dir --n-boot --seed` | Stage-0 preds + `journals/*.jsonl(.gz)` → stdout | `06_signal_model_plan` step 6 (M1 probe) |
| `horizon_transfer_report.py` | Horizon-transfer curves ρ(r^δ, r^Δ) vs the IID null, from harvest labels | Jetson (parquet) | `--prefix --n-boot --seed --stride --per-name` | training data → stdout | `06_signal_model_plan` step 7 (FR-07-A) |
| `meta_learning_curve.py` | Meta-label learning curve → `floor.honest_floor` per book | Jetson (lightgbm) | `--prefix --seeds --block-len --grid --eval-fraction --no-tiering --out --base-seed` | training data + meta artifacts → `{prefix_}meta_curve_report.json` | `03_jetson_runbook` Phase 1; precondition for `META_OOF_PRED` |
| `reliability_report.py` | Calibration reliability (bins, Brier, ECE) before vs after the gate | Mac (numpy + `calibration.py`) | `--in --bins` | a preds JSON → stdout | runbook Phase 2 §4a (`CALIBRATION_V2`); wave-9 #1; q10 coverage check in `06` step 10 |
| `ic_by_name.py` | Per-name IC / consistency / t-stat — the universe-promotion gate | both (Mac-safe; needs a real dump) | `--in --name-key --pred-key --fwd-key --time-key --min-ic --min-consistency --min-t --subperiods` | `{slot}_stage0_preds.json` → stdout | runbook Phase 1 (wave-9 #3) |
| `rank_gradient_report.py` | Monotone rank-gradient Stage-0 gate for breadth / concentration / edge-Kelly | both (Mac-safe; needs a real dump) | `--preds --buckets --signal-lag --cost-pct --fwd-bars --extra-cols --strict` | Stage-0 preds → stdout | runbook Phase 1 (wave-9 #4/#5) |
| `cscv_audit.py` | Offline CSCV Probability-of-Backtest-Overfitting audit | Mac (`validation.py` only) | `--blocks --returns --n-blocks --n-groups --pbo-max` | block/returns JSON → stdout | wave-8 #2 |
| `wave6_stage0.py` | Wave-6 Stage-0 measurement (uniqueness / effective-n, offline, no live data) | both (Mac-safe; needs training data) | `--book --fb --json` | training data → stdout, optional `--json` path | wave-6 Stage-0 |
| `sizing_cofire_report.py` | Per-multiplier bind rates, co-fire matrix, worst composed product, v2-vs-legacy tilt | Jetson (journals) | `--days --journal-dir --book --json` | `journals/` sizing rows → stdout / JSON | runbook Phase 1 (B7); precondition for `DERISK_STACK_V2` |
| `crypto_spread_census.py` | Poll live crypto venue spreads to census the `Eff_Spread_Pct` stamp against reality | Jetson (needs `ALPACA_API_KEY`/`_SECRET`) | `--minutes --interval --loc --symbols --out` | Alpaca crypto quotes → `crypto_spread_census.json` | runbook Phase 1 (B05.1) |
| `llm_qualify.py` | Qualification harness for free LLM endpoints in the analyst role (+ shadow scoring) | Jetson (provider keys + journals) | `--models --n --spacing --out --shadow --replay --days --max-cycles --report` | provider endpoints + journals → `journals/llm_qualify/` (`shadow_scores.jsonl`, `llm_qualify_report.json`) | runbook Phase 1 (c26 packet V1) |
| `prompt_ab.py` | Offline A/B of analyst prompt variants against journal outcomes | Jetson (keys + journals) | `--days --asset --system-b --hide-pred-b --rich-context-b --model --max-cycles --sleep-sec --out --dry-run --in --min-n` | journals + endpoints → `llm_prompt_ab_report.json`, `llm_prompt_ab_scores.jsonl` | LLM economics (pairs with `llm_eval.py`) |
| `train_lexicon.py` | Offline trainer for the learned sentiment lexicon — DARK ARTIFACT, never auto-deployed | Jetson (`sentiment_cache.db` + stock parquet ⇒ `pyarrow`) | `--db --data --horizons --fit-horizon --start --end --min-df --screen-t --embargo-days --folds --lambda-grid --journal-days --no-novelty --recompute-kw --out --report --seed` | `sentiment_cache.db` + `stock_training_data.parquet` → `learned_lexicon.json`, `lexicon_eval_report.json` | runbook Phase 1 |
| `connection_test.py` | Alpaca credential + account-status check | Jetson (`dotenv` unguarded, needs keys) | *none* | `.env` → stdout | first-run install check |
| `ab_check.sh` | **The regression gate.** Runs the suite once, diffs FAILED/ERROR *names* against the baseline | both (the baseline file is dev-Mac-only) | env: `AB_CHECK_TIMEOUT_S` (900), `AB_CHECK_RERUN_TIMEOUT_S` (300), `AB_CHECK_MIN_PASSED` (1500), `AB_CHECK_PYTEST` | `tests/baseline_failures.txt` → stdout + `$TMPDIR` temp files only (never touches git state) | the verification standard (`research/AGENT_CONTEXT.md` § Verification; runbook Phase 0 step 3) |
| `setup.sh` | Install the Python stack, create the `.env` template, verify imports + CUDA | Jetson / Ubuntu desktop | `--jetson` | `requirements.txt` or `requirements-jetson.txt` → `.env` template (if absent) | first-time install |
| `setup_jetson_system.sh` | One-time OS prep: headless target, 12 GB swap, CUDA libs, `jetson-stats`, chrony, `trader.service` | Jetson only (`sudo bash scripts/setup_jetson_system.sh`, not chmod +x) | `--skip-headless --skip-swap` | writes `/etc/fstab`, `/etc/sysctl.conf`, `/swapfile`, `/usr/local/cuda/lib64/*`, `/etc/chrony/chrony.conf`, `/etc/systemd/system/trader.service` (installed, deliberately **not** enabled) | prod bring-up |
| `backup_state.sh` | Consistent snapshot: `sqlite3 .backup` of the Optuna DBs, then JSON state, model artifacts, `journals/` | Jetson (cron) | *none*; env `RESTIC_REPOSITORY` | `*_study.db`, `*_state*.json`, `*.pth/*.pkl/*lgb*`, `journals/` → restic repo, else `~/trader_backups/trader_state_<ts>.tar.gz` (keeps newest 14) | prod ops (cron line printed by `setup_jetson_system.sh`) |

## How the pipeline calls these

`run_pipeline.py` is the only module that `Popen`s anything in `scripts/`. Its phase chain, as the
command literals spell it out (see `docs/MAP.md` / D13 process graph):

```
run_pipeline.py
├─ scripts/harvest_crypto_data.py                 (crypto book)
├─ scripts/harvest_stock_data.py                  (stock book)
├─ scripts/hypersearch_v2.py --trials N --preset stationary --no-status
│      stock leg adds: --data stock_training_data.csv --prefix stock --max-rows 200000
├─ meta_label.py [--prefix stock]                 (root module, not scripts/)
├─ backtest.py --days 44 --gate    |  --prefix stock --days 60 --gate
└─ bots: run_bots.py [--crypto-only|--stock-only]  or crypto_loop.py / stock_loop.py
```

Only harvest and hypersearch live here; `meta_label.py`, `backtest.py` and the loops are root
modules. `gui.py` spawns its report CLIs (`decision_report.py`, `beta_ledger.py`,
`indicator_leadlag.py`, `llm_eval.py`, `execution_report.py`, `gap_audit.py`) — all root modules,
none from `scripts/`. `shadow.py` spawns `meta_label.py`.

## Verification standard

Any change under `scripts/` is verified with `bash scripts/ab_check.sh` — it judges by failure
**names** against `tests/baseline_failures.txt`, never by counts. Exit 0 = no NEW failing names.
Suite numbers live in one place only: `CLAUDE.md` § Running tests.

## Known defects

- `setup.sh`'s closing "Next steps" block printed `python connection_test.py`; the file is
  `scripts/connection_test.py`. Corrected in the 2026-09-08 docs/layout pass.
- `setup.sh` names its repo-root variable `SCRIPT_DIR` (it is assigned `dirname $0/..`, i.e. the
  repo root — the value is right, the name is wrong). Cosmetic; not changed.
