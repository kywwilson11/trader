# Phase-3 bundle run: what a correctly-flagged trainer log must show (LANDKIT-2, 2026-09-27)
Format derived from the composed mirror (signal_r3/mirror4, kit landed): scripts/hypersearch_v2.py `flags_banner` (:2433-2463),
printed once at :2509, after `[SEED]` and before the study is created or `--fresh` deletes anything.
**Precondition:** the three flips (HYPERSEARCH_V3 / OBJECTIVE_V3 / TRAINING_REPAIRS_V1 = True) are NOT in the kit yet (see the LANDKIT-2 handback).
If the banner still shows `=False` for any of them, stop the run.

## 1. `[FLAGS]` banner: one line; tokens are space-separated, and spaces inside values become `_`
Stock search (run_pipeline's form, run_pipeline.py:1242; a hand launch MUST pass `--data stock_training_data.csv`, because
`--prefix stock` alone loads the crypto store, :2524-2525 `load_data(args.data)`, default `training_data.csv`):
```
[FLAGS] OBJECTIVE_LONG_ONLY=True HYPERSEARCH_V3=True OBJECTIVE_V3=True TRAINING_REPAIRS_V1=True BLEND_FIT_ON_REFIT=False BLEND_THRESHOLD_RESELECT=False PROMOTION_GATE_V2=False KISH_NEFF_ENABLED=False LGB_REFIT_FULL=False HOLDOUT_SPAN_BY_TARGET=False BARS_PER_YEAR_MEASURED=False FAILED_TRIAL_PRUNE=False OBJECTIVE_SESSION_MASK=False WICK_PRINT_FILTER=False HOLDOUT_SPAN_BY_TARGET.eff=False FAILED_TRIAL_PRUNE.eff=True env.TRADER_HOLDOUT_SPAN_BY_TARGET=None env.TRADER_FAILED_TRIAL_PRUNE=1 env.TRADER_TRAINER_SEED=None env.TRADER_FIXED_HOLDOUT_DAYS=None env.TRADER_OBJECTIVE_SESSION_MASK=None env.TRADER_WICK_PRINT_FILTER=None preset=stationary preset_arg=stationary seed_base=None fixed_holdout_days=None prefix=stock mode=<mode> trials=<N> trials_arg=<N|200> data=/home/kyle/trader/stock_training_data.parquet data_arg=stock_training_data.csv
```
- Must hold: the four model flags `=True`; stock `FAILED_TRIAL_PRUNE.eff=True` + `env.TRADER_FAILED_TRIAL_PRUNE=<exported value>`; every other token as above (config FAILED_TRIAL_PRUNE stays `False`; only `.eff` resolves the env).
- Crypto rerun: same, but `prefix=None data=/home/kyle/trader/training_data.parquet data_arg=training_data.csv`; its env PRUNE is the CEO's call. `trials_arg=200` = argparse default; `trials` = the resolved count.

## 2. `[REPAIRS]` lines (TRAINING_REPAIRS_V1 ON; print-only, SIG-R2-3). None of them may be missing.
- `[REPAIRS] L6 embargo in bars: <seq_len*EMBARGO_MULTIPLIER> distinct bars (fold-0 val starts +<h>h after train end; legacy calendar rule +<h>h); folds=<k>`
  - Once per fold build (every trial :1045 + save-time LGB :1528). Stock: bar gap > calendar gap; crypto: equal.
- `[REPAIRS] L5 trial <t> fold <f>: memory probe undone (init weights restored; fresh optimizer/scheduler/grad-scaler)`: once per trial.
- `[REPAIRS] L1 trial <t> fold <f>: val loss on the trial criterion (huber_delta=<d>, |y|+1 weights cap 50) epoch0 val_loss=<v>`: once per trial.
- `[REPAIRS] L2 regime masks from lagged completed-by-t returns: bull=<n> bear=<n> sideways=<n> warmup_excluded=<n>`: once per `compute_regime_sharpes` call.
- At save: `[REPAIRS] L5 final_refit: memory probe undone (init weights restored; fresh optimizer/scheduler/grad-scaler)`.

## 3. V3-path lines (verbatim from PHASE3-1 signal_r1/phase3/run_on.log:40-66, crypto slice)
They appear only if a trial scored > 0 AND beat the ratchet. See F2 below.
```
[REFIT] final refit: 10344 rows, 1 fixed epochs (SWA tail soup K=4)
[LGB-Q10] tail model trained, save deferred to the gated atomic save (veto floor -2.0341%)
[BLEND] M2 side-by-side: stale-w=0.2500 (raw=-0.25712879506932795) vs refit-w=0.5000 (raw=1.0064157858597935)
[BLEND] w_raw=-0.2571 se=0.3256 significant=True -> w_fit=0.25 smoothed w=0.25 (prev=None, grid diag=0.41)
[BLEND] threshold reselect (H2): searched=1.53 (n_trades=0 on blended val) -> blend-optimal=1.53 (n_trades=0, sharpe=0.00) [logged only]
  [HOLDOUT] FR-01 boundary quantile=1789441632 (1440 rows) vs fixed-60d=1785121200 (8640 rows); active=quantile, scored rows=1440, trades=0
```
- `certified='blend'` is NOT a log line. It appears in `{p}model_v2.manifest.json` and the holdout report (`"certified": "blend"`, plus `lstm_weight` and `q10_vetoed`).
- Must NOT appear: `[LGB] M4 guard: HYPERSEARCH_V3 without OBJECTIVE_V3` (half-flipped). Trial `th=` must lie in crypto [0.96, 2.0] / stock [0.18, 0.57].

## 4. CAUTIONS
- **F1:** OBJECTIVE_V3 anchors the crypto trade_threshold range to [0.96, 2.0].
  - PHASE3-1's blended holdout preds peaked at 0.64–0.73 on its slice, which could give 0 certificate trades, and then the gate fails closed.
  - The reselect grid starts at the same floor, so it cannot rescue this. Still informative: fold scores, `[REPAIRS]`, SIG-R3-DECOMP attrs; read reselect `n_trades` + holdout `trades=`.
- **F2:** if every trial scores ≤ 0 (like the 40-trial crypto pass), `_state_cache` stays empty (:1262 score>0), so V3 never runs and the log says "No new best found". That is expected, not a defect.
- **F3:** a study DB that was not reset does NOT fail loudly. Optuna 4.7.0 resumes a changed-range study silently (phase3/optuna_distchange.txt). The reset below is mandatory.

## 5. Gotcha-#2 reset (the CEO performs it, trainer stopped, BEFORE the first flagged run; move aside, never delete)
- Move aside `v2_study.db` and `stock_v2_study.db`, plus any `-wal`/`-shm` files. `--fresh` also deletes the DB and records it in `db_deletions`.
- Move aside `adaptive_state_crypto.json` + `adaptive_state_stock.json` (sanctioned: adaptive_config.py:132-136, `cum_trials` resets ONLY by removing the file; also resets `best_score`=0.0, `cum_holdout_gates`=0).

## 6. Rollback of the flips
- `bash <scratchpad>/signal_r3/land_backup/<ts>/restore.sh`, using the `<ts>` of the run that printed `installed:`. It restores strategy_config.py, FLAGS.md and CHANGELOG from `orig/`.
- Studies run ON are incomparable to OFF ones: a rollback needs another gotcha-#2 reset. After the reset expect `Resuming from 0 prior trials`.
