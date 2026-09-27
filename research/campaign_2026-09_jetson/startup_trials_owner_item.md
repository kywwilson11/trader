# OWNER ITEM — `PRUNE_STARTUP_TRIALS = 60` drives both the TPE warm-up and the pruner (SCOUT-3, 2026-09-27)

Status: owner decision, nothing implemented. Evidence: `research_signal.md` "R3 scout brief (SCOUT-3)" G3.
Model-facing, so any change goes through a default-OFF flag, then challenger, then shadow.

## Facts (OBJECTIVE)
- One constant sets both `n_startup_trials` values (scripts/hypersearch_v2.py:100, :2419, :2422). It was
  50 in dd58a02 (2026-02-11) and became 60 in 570081a (2026-02-21), in the same edit that set NUM_TRIALS = 300.
  No rationale was recorded (it fits the folk "rule of 59" and 20 % of 300; that is unverified).
- Installed Optuna 4.7.0 behaviour. TPE runs random until the study holds 60 COMPLETE+PRUNED trials.
  MedianPruner stays off until the study holds 60 COMPLETE trials. FAIL/RUNNING trials count for neither.
  The count covers the whole study DB, not one run.
- Fresh DB (after gotcha-#2 reset, categorical expansion or `--fresh`):
  | budget | random | TPE-steered | prunable |
  |---|---|---|---|
  | 40 (tonight) | 40 | 0 | none |
  | 70 (refine) | 60 | 10 | last 10 |
  | 200 (initial) | 60 | 140 | 140 |
  If the DB is resumed and already holds 60 or more, every trial is TPE and prunable. **So the constant only
  matters after a reset. That happens tonight (a 44-trial DB) and on every first retrain after a reset.**
- 13 search dimensions (hypersearch_v2.py:842-879; OBJECTIVE_V3 re-ranges one and adds none). Median-pruned trials are left out of the DSR pool, because every counter filters on COMPLETE. The DSR
  docstring (validation.py:191-195) says they should be counted. The code and the docstring already conflict.
## Options (JUDGMENT)
- **A — keep 60/60.** Runs up to 60 are pure random search. That is robust on a very noisy objective (fold
  std 0.26–1.08 tonight) and needs no change. Cost: 40- and 70-trial runs after a reset get almost no TPE
  steering and no pruning savings (GPU-hours on the 8 GB box).
- **B — sampler 20, pruner 20 COMPLETE, warm-up 12 unchanged (recommended).** 20 is at least d+1 = 14 (the
  BOHB KDE minimum), matches hyperopt's own default (Optuna sampler.py:658) and gives 1.5·d points.
  Resulting splits: 40 = 20 random + 20 TPE; 70 = 20 + 50; 200 = 20 + 180. Pruning starts after 20 COMPLETE.
  Needs: a default-OFF `HPO_STARTUP_V2` flag (strategy_config is owner-owned), so the OFF path stays
  byte-identical. It lands **after SIG-R2-1**, so failure-pruned trials carry `failed_trial` and can be told
  apart from median-pruned ones. Count median-pruned trials in the pool (COMPLETE + PRUNED − failed_trial).
  This is the conservative convention the docstring already asks for. It costs little: power at SR .10 with
  n_eff 60 drops from .031 to .023 when N goes from 70 to 100.
- **C — Optuna defaults 10/5.** With gamma = ceil(0.1·n), l(x) sits on 1–2 points up to trial 20. On this
  noisy 13-D objective that means chasing noise. Not recommended.
- **D — decouple only the pruner (sampler 60, pruner 20).** Saves compute and leaves the proposals unchanged.
  It is the fallback if the owner wants A's exploration but not A's wall time.

Caveats. Changing the sampler does not change the objective, so old scores stay comparable and no DB reset
is needed. But the constant has no effect on an already-resumed DB with ≥ 60 trials. Seeded runs (R2C-04)
will not reproduce across settings. Counting pruned trials changes what `cum_trials` means from the flip
date on, so log that date in `trial_history`. Optuna 5.0 (S1-03) independently flips multivariate and
constant_liar. Decide S1-03 first so this A/B is not confounded by a version change.
## Pre-registered measurement (idle Jetson only, via hwlock heavy)
Run crypto only, in an ISOLATED copy of the tree: its own `v2_study.db`/adaptive state, never the production
directory, no model save. Same data snapshot and `TRADER_TRAINER_SEED=<fixed>`. Two fresh 70-trial searches:
arm A = 60/60, arm B = 20/20. Both use the same seed base and study name, so the first 20 proposals match
(a paired start). Record per arm:
(1) the best-COMPLETE and median-of-top-5 score trajectories at trials 20/40/70;
(2) total wall time and the number of PRUNED trials;
(3) if both winners score > 0, a holdout certificate each (DSR + `gate_power`).
Rule:
- **Adopt B** iff wall(B) ≤ 0.80·wall(A), AND top-5 median(B) ≥ top-5 median(A) − 0.25 (≈ half tonight's
  typical fold std: non-inferiority, because one pair cannot show superiority), AND (when both certificates
  exist) DSR(B) ≥ DSR(A) − 0.05.
- **Adopt D instead** if B fails only on the score criterion.
- **Keep A** if wall(B) > 0.90·wall(A) (pruning does not pay).
- No re-runs with other values; log both arms in the S1-09 ledger.
- Budget: ~70 × 5–15 min per arm (6–18 h). Never run it alongside production training.
