<p align="center">
  <img src="logos/circuit_bull.png" alt="Trader" width="200">
</p>

<h1 align="center">Trader</h1>

<p align="center">
  <a href="https://github.com/kywwilson11/trader/actions/workflows/ci.yml">
    <img src="https://github.com/kywwilson11/trader/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
</p>

<p align="center">
  Autonomous paper-trading system for US stocks (market hours), with a crypto book (24/7) that is
  built but currently idle. One RegressionLSTM + LightGBM blend per book, a meta-labeling veto,
  cost-aware gates, honest validation (purged walk-forward + Deflated Sharpe), and a policy
  backtester that replays real exits before any model is promoted. Runs in production on an NVIDIA
  Jetson Orin Nano (8 GB) and trades through the <a href="https://alpaca.markets/">Alpaca</a> paper API.
</p>

> Orientation for a new reader. Facts have one home in this repo, so this page points at the file
> that owns a number instead of repeating it; when any doc and the code disagree, trust the code.

## How to read this repo

1. **This README**: what the system is, where it stands today, how to check on it.
2. **[`docs/book/README.md`](docs/book/README.md)**: the explanatory book, a guided walk through
   the design and the reasoning behind it (being written now, 2026-09).
3. **[`CLAUDE.md`](CLAUDE.md)**: the operational source of truth: how to run and test, the
   two-machine table, conventions, gotchas.
4. **[`docs/MAP.md`](docs/MAP.md)**: the architecture narrative: layout, the full lifecycle, the
   goal of each piece, invariants, known gaps. [`docs/README.md`](docs/README.md) indexes the
   reference censuses behind it ([`MODULES`](docs/MODULES.md), [`FLAGS`](docs/FLAGS.md),
   [`STATE_FILES`](docs/STATE_FILES.md), [`GLOSSARY`](docs/GLOSSARY.md)).

## Where things stand (2026-09-27)

**Production is the Jetson.** Everything that trains, harvests, trades or draws the GUI runs there,
under the `jetson` conda env (Python 3.10). That env needs the conda `libstdc++` preloaded
(`LD_PRELOAD`) or `import torch` followed by `import sqlite3` fails; `run_pipeline.py` (its `ENV`
block) and the systemd unit set it for their children, and the environment table is in
[`CLAUDE.md`](CLAUDE.md) § Two-machine reality. The GUI is the one exception: it runs under the
base env, the only one with PySide6.

**The book runs stock-only.** On the founder's direction (2026-09-27) effort on the crypto book has
stopped and the service trades stocks only. Nothing crypto was removed: the loop, harvester, flags
and the `--crypto-only` / two-book modes all still work; they are simply not running.

**The service is a user-level systemd unit**, installed without sudo by
`bash scripts/setup_jetson_system.sh --user` (with `loginctl enable-linger` so it survives logout).
The repo's unit stays generic (`run_pipeline.py --combined-bots --bot-only`); the stock-only choice
is a local drop-in override at `~/.config/systemd/user/trader.service.d/override.conf` that adds
`--stock-only`. `--bot-only` skips the initial harvest and training and goes straight to the bots;
the weekly retrain cycle still runs (it stops the bots, retrains, restarts them). Once the unit
runs, never also start bots by hand or from the GUI.

**Models come from the Phase-3 trainer bundle.** `HYPERSEARCH_V3`, `OBJECTIVE_V3` and
`TRAINING_REPAIRS_V1` are now `True` in `strategy_config.py` (flipped together at the 2026-09-27
clean-rebuild study reset, with `OBJECTIVE_LONG_ONLY` already on). What each flag does, and its
evidence gate, is in [`docs/FLAGS.md`](docs/FLAGS.md) and
[`research/campaign_2026-08/03_jetson_runbook.md`](research/campaign_2026-08/03_jetson_runbook.md).

**The bot fails closed without a certified model.** The trainer writes model artifacts only when a
candidate passes its holdout Deflated-Sharpe gate. If no artifact exists (or one fails to load),
`BaseTradingLoop._load_models` disables buys; exits and stops keep running, and the hot-reload
check keeps retrying. So a running service with no certified model places no new entries.

### State of the strategy, honestly

There is no certified edge yet. The 2026-09 clean rebuild retrained both books on leak-free,
re-harvested data, and no candidate in either book passed the holdout gate; the campaign's per-fold
diagnosis is that the current LSTM plus objective design shows no measurable selection skill over a
zero-skill ranker. The evidence, the numbers and the candidate levers are in
[`research/campaign_2026-09_jetson/README.md`](research/campaign_2026-09_jetson/README.md) §4. The
older structural verdict still stands too: the deployed policy is a gated, conditional-beta,
long-only book whose positive expected value is unproven ([`docs/MAP.md`](docs/MAP.md) §1 and §9).
A Phase-3 stock retrain was launched on 2026-09-27; its gate result is recorded in that same §4 when
it lands. Until something passes, this is a well-instrumented paper system waiting for evidence,
not a profitable strategy.

Everything from the 2026-09 Jetson campaign is uncommitted and awaiting owner review; its change log
is [`research/campaign_2026-09_jetson/CHANGELOG.md`](research/campaign_2026-09_jetson/CHANGELOG.md).

## Quick start on the Jetson today

Check the install (the env wrapper below is what `run_pipeline.ENV` sets; keep
`CUDA_VISIBLE_DEVICES=''` for anything interactive so the GPU stays free for training):

```bash
export LD_PRELOAD=/home/kyle/miniforge3/envs/jetson/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/kyle/miniforge3/envs/jetson/lib/python3.10/site-packages/nvidia/cusparselt/lib:/home/kyle/miniforge3/envs/jetson/lib:${LD_LIBRARY_PATH:-}
JPY=/home/kyle/miniforge3/envs/jetson/bin/python
cd ~/trader
CUDA_VISIBLE_DEVICES='' $JPY -c "import torch, lightgbm, optuna, numba, pyarrow, bidask, dotenv, alpaca_trade_api, sqlite3; print('ok', torch.__version__)"
```

Check the service:

```bash
systemctl --user status trader          # should show run_bots.py --stock-only under run_pipeline.py
journalctl --user-unit=trader -f        # live log (this Jetson's journal is volatile)
cat ~/.config/systemd/user/trader.service.d/override.conf   # the stock-only drop-in
```

Open the dashboard (base env, not the jetson env; it reads `pipeline_status.json` and the logs, and
launches jetson-env subprocesses with the right environment itself):

```bash
/home/kyle/miniforge3/bin/python gui.py
```

Read the evidence (measurement-only: runs the runbook's instruments one at a time and prints a
READY / NOT YET / NO DATA table; output goes under `logs/evidence_reads/`):

```bash
CUDA_VISIBLE_DEVICES='' $JPY scripts/evidence_reads.py --dry-run   # show the commands, write nothing
CUDA_VISIBLE_DEVICES='' $JPY scripts/evidence_reads.py             # run them
```

Its flags are in [`scripts/README.md`](scripts/README.md). Before flipping anything, read the
activation runbook,
[`research/campaign_2026-08/03_jetson_runbook.md`](research/campaign_2026-08/03_jetson_runbook.md).

**On the dev Mac** only pure-algorithm work happens: no torch, lightgbm, alpaca or GUI, so no
training, harvest, bots or parquet round-trips. Use the canonical test command and
`bash scripts/ab_check.sh` from [`CLAUDE.md`](CLAUDE.md) § Running tests; the Mac and Jetson sync
is user-driven and not encoded in the repo.

## What it is

- Two independent books sharing one Template-Method engine (`base_loop.py`); only stocks run today.
- **One** `RegressionLSTM` (`model_v2.py`) + LightGBM (`model_lgb.py`) blend per book, producing
  multi-horizon hourly return forecasts. The old dual bear/bull 3-class ensemble is retired;
  "bear/bull" survives only as regime diagnostics and as the champion/challenger shadow-slot names.
- A meta-label veto (`meta_label.py`, a secondary classifier) and a shared cost model (`fees.py` +
  `liquidity.py`, `bidask` EDGE per-name spread) that every gate uses.
- Honest validation: purged walk-forward with embargo, an untouched holdout, Deflated Sharpe, and
  label-overlap effective-n (`sample_weights.py`). What is actually wired into promotion versus
  merely available is spelled out in [`CLAUDE.md`](CLAUDE.md) § Architecture in brief.
- A policy-replay promotion gate (`backtest.py`) that rolls a bad promotion back to `.prev`; every
  **model-facing** change ships only through challenger, shadow, then DM-HLN deployment.
- Fail-closed execution and strict point-in-time (PIT) discipline throughout.

## Architecture

```
run_pipeline.py                    Orchestrator: harvest -> train -> gate -> launch bots -> weekly
                                    retrain (bots are STOPPED for the retrain, restarted after)
|-- scripts/harvest_stock_data.py     hourly OHLCV + features -> stock_training_data.{csv,parquet}
|-- scripts/harvest_crypto_data.py    hourly OHLCV + features -> training_data.{csv,parquet} (idle)
|-- scripts/hypersearch_v2.py         Optuna search: RegressionLSTM + LightGBM leg, holdout DSR gate
|-- backtest.py                       Policy-replay promotion gate (real entries/exits/fees)
|-- stock_loop.py                     Market-hours stock trading
|-- crypto_loop.py                    24/7 crypto trading (idle)
+-- run_bots.py                       Live bots in one process (--stock-only in production)
```

The **weekly retrain is a cold restart** (bots stopped, retrained, restarted); the **daily shadow
promotion is the hot-reload** (manifest written last, running bots swap on its mtime): see
[`CLAUDE.md`](CLAUDE.md) § Weekly retrain vs hot-reload. `run_pipeline` defaults to one process per
bot; `--combined-bots` (what the systemd unit passes) hosts the loops in `run_bots.py`.

Three shared kernels keep the system honest, and every consumer must use them:
`strategy_config.py` (the single source of truth for policy: both the live loops and the
backtester read it, so drift means the backtest validates something other than what trades),
`policy_exits.py` (one Numba exit stack shared by `backtest.py`, the harvest triple-barrier labels
and `meta_label.py`; the live loops mirror it bar by bar rather than calling it, with the known
divergences listed in its docstring), and `fees.py` + `liquidity.py` (the cost model every gate
shares). Around them sit the models, the validation stack, a multi-provider schema-enforced LLM
overlay (`llm_client.py` / `llm_analyst.py`, with cross-provider fallback) and the `gui.py`
dashboard; per-module reference: [`docs/MODULES.md`](docs/MODULES.md).

## How a trade happens, and where to start it

data -> features -> RegressionLSTM+LightGBM blend -> cost gate -> meta-label gate -> sentiment/LLM
gate -> order -> ATR-based exits -> per-book risk cap (full gate order: [`docs/MAP.md`](docs/MAP.md)
§2; sizing constants live in `strategy_config.py`, trust it over any prose copy). The daily commands
and their verified flags are in [`CLAUDE.md`](CLAUDE.md) § Running the system; **every** entry point
and flag is catalogued in [`docs/MODULES.md`](docs/MODULES.md) Appendix A and
[`scripts/README.md`](scripts/README.md). The ones met first: `run_pipeline.py`, `run_bots.py`,
`scripts/hypersearch_v2.py`, `backtest.py`, `decision_report.py`, `llm_eval.py`, `beta_ledger.py`,
`scripts/evidence_reads.py` and `gui.py`.

## Testing & CI

Suite counts live only in [`CLAUDE.md`](CLAUDE.md) § Running tests, together with the canonical
dev-Mac command and the missing-dependency baseline. The standard regression check is
**`bash scripts/ab_check.sh`**, which judges by failure *names* against
`tests/baseline_failures.txt`, never by counts; a `git stash` A/B is the fallback for when that
baseline may itself be stale (see `CLAUDE.md` and the `/regression-ab` skill). On the Jetson, run
single test files with the env wrapper above rather than the whole suite while training is running.
`tests/test_sentiment_headlines.py` is a standalone runner, not a pytest module; run it with
`python tests/test_sentiment_headlines.py`. Suite conventions: [`tests/README.md`](tests/README.md).

CI (`.github/workflows/ci.yml`) runs a py3.10 jetson-parity leg and a py3.12 modern leg; what each
installs and runs is in [`CLAUDE.md`](CLAUDE.md) § Running tests.

## Generated and local files

Code, tests, docs, research, `.claude/`, `archive/`, requirements and policy config are committed;
models, study DBs, training data, journals, logs, caches, `.env` and `llm_config.json` (it holds API
keys) are gitignored and live only on the machine that made them. Authoritative per-file table:
[`docs/STATE_FILES.md`](docs/STATE_FILES.md). Nothing here is deleted: files that outlive their
place are moved to `archive/` or a dated `research/` subdirectory and recorded in
[`archive/README.md`](archive/README.md).

## Research process

Research is a record, not a backlog: every round is dated and frozen, and indexed by
[`research/README.md`](research/README.md). In order: waves 1 to 9 (2026-06, `research/waves/`),
the 2026-07 module review (its owner decision queue is `research/module_review_2026-07.json`,
rendered with `/decision-queue`), the 2026-08 comprehensive campaign (`research/campaign_2026-08/`,
including the Jetson activation runbook), the R2 signal-model and R3 literature rounds, and the
2026-09 Jetson test and improvement campaign (`research/campaign_2026-09_jetson/`, the first run of
the current code against the production stack, data, journals and broker). Two rules bind every
round: [`research/KILL_LIST.md`](research/KILL_LIST.md) is the consolidated do-not-rebuild list, and
[`research/AGENT_CONTEXT.md`](research/AGENT_CONTEXT.md) is the brief every spawned agent reads
first.

## Appendix: fresh install

For a new machine. The prod Jetson is already set up; its environment table is in
[`CLAUDE.md`](CLAUDE.md) § Two-machine reality.

```bash
git clone git@github.com:kywwilson11/trader.git
cd trader
./scripts/setup.sh            # desktop
./scripts/setup.sh --jetson   # Jetson Orin Nano (JetPack 6.x)
```

`scripts/setup.sh --jetson` installs PyTorch 2.8.0 from the Jetson AI Lab wheels, then
`requirements-jetson.txt`; PyTorch 2.9.1 is broken on the Jetson (missing `libcudss.so.0`).

**Configure.** Create `.env` (or let `scripts/setup.sh` create the template) with `ALPACA_API_KEY`,
`ALPACA_API_SECRET`, `ALPACA_BASE_URL` (`https://paper-api.alpaca.markets` for paper) and
`FINNHUB_API_KEY`. Alpaca: a free paper account at [alpaca.markets](https://alpaca.markets/).
Finnhub: a free key at [finnhub.io](https://finnhub.io/) (optional, stock news sentiment). LLM
analysis (optional): the GUI Settings tab, or `llm_config.json` directly (gitignored, it holds API
keys). Alerts are silent until `TRADER_TELEGRAM_*` or `TRADER_WEBHOOK_URL` is set in `.env`
([`docs/FLAGS.md`](docs/FLAGS.md)).

**Verify connectivity** with `python scripts/connection_test.py`. Read its output: it prints
failures but still exits 0.

**Service.** No sudo: `bash scripts/setup_jetson_system.sh --user` installs and enables the user
unit (add `--print-unit` to preview it without touching anything), then
`loginctl enable-linger $USER` and `systemctl --user start trader`. With sudo, the same script also
does the one-time system setup (headless desktop, NVMe swap, cuDSS/cuSPARSELt installed
system-wide, jtop): `sudo bash scripts/setup_jetson_system.sh [--skip-headless] [--skip-swap]`; the
unit's interpreter defaults to the jetson env and can be overridden with
`sudo TRADER_PYBIN=/path/python bash ...` ([`docs/FLAGS.md`](docs/FLAGS.md) §5). Its header comment
documents every step. Without the service, `python run_pipeline.py` runs the full pipeline in the
foreground (harvest, train, trade, weekly retrain) and `--stock-only` restricts it to the stock book.

## Appendix: research-record details

Carried over from the previous README, which was their only home: the 2026-07 module review also
applied 280 safe fixes alongside the owner decision queue it left behind, and each of research
waves 1 to 9 carries its own survivors and its own kill list in `research/waves/`.
