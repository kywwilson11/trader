# Signal-evidence runbook — the offline half of the 12-step Jetson sequence

*Research note, 2026-09-27 (RUNBOOK-1, SIGNAL dept). Measurement-only. Steps and decision rules come from
`research/campaign_2026-08/06_signal_model_plan.md` §4 (lines 347-393). Every command below was taken from the
script's own `add_argument` calls, not from docs. Nothing here has been committed or flipped.*

**Stores (rebuilt 2026-09-26, new schema):** crypto `training_data.parquet` 263,889 rows × 71 cols (6 names, 5 TB
horizons, Funding_*/OI_*); stock `stock_training_data.parquet` 1,536,356 × 110 (Eff_Spread_Pct, CS_*, TB_*, SVR_*).
The CSVs are 312 s (crypto 30 s) newer than the parquets — under `_STALE_PARQUET_SLACK_S=3600` (data_utils.py:42,58)
— so every loader serves the **parquet** and column projection works.

**Tonight's trainer:** `hypersearch_v2.py --trials 40 --preset stationary --no-status --mode initial --shadow`
(crypto), logging to `<scratchpad>/phase5/train_crypto.log`, about 8.5 min per trial (~5.7 h). No
`*model_v2.manifest.json` exists in the root, so `shadow.champion_exists('')` is False (shadow.py:115-116) and
**the winner saves into the CHAMPION slot (`''`), not the challenger** (hypersearch_v2.py:2883-2889).

## 0. Two facts that change the plan (OBJECTIVE, read before step 4)

1. **`HYPERSEARCH_V3 = False`** (strategy_config.py:231), and the running process has no `TRADER_*` override. The
   blend certificate (`blended=True`) needs `lgb_booster`+`lstm_weight`, which only the V3 path passes
   (hypersearch_v2.py:1803, 2828). So tonight's champion will **not** emit:
   the FR-05 `cs-rank-IC` lines / `holdout.cs_rank_ic` (:2066-2112, inside `if blended`); the M2 `stale-w vs
   refit-w` and H2 `threshold reselect` lines (:2621, :2727, inside `if _v3:` at :2495); or `config['refit']`
   (set only from V3's `final_refit`, :2491/2516/2877-2878).
   **Steps 4, 10 and 11 therefore need a V3 retrain first** — model-facing, so an owner call. `window_ab` on this
   champion needs an explicit `--epochs` (window_ab.py:401-405).
2. **The retrain ledger cannot write row 1 tonight.** There is no champion and no `.prev` incumbent
   (retrain_ledger.py:159 `incumbent_paths`, fail-soft at :336-338). The first paired row lands at retrain **#2**.

## 1. Per-step table

Readiness: **NOW** = new stores only · **AFTER-CH** = needs the named champion-cycle artifact · **BLOCKED** = no
producer exists. Producers: **Stage-0 dump** `{slot}_stage0_preds.json` = backtest.py:909-912 via the weekly gate
`backtest.py --days 44 --gate` / `--prefix stock --days 60 --gate` (run_pipeline.py:1218-1265) · **OOF**
`{p}oof_preds.npz` = hypersearch_v2.py:2180 · **manifest** = :2190-2203 (last) · **`cum_trials`** =
adaptive_config.py:488 · **meta** `{p}meta_model.txt/meta_calib.pkl/meta_meta.json` = meta_label.py:79-82.

| # | Step | Instrument | Readiness today |
|---|---|---|---|
| 1 | R2C-01 serving integrity | live champion log (bots) — live observation, not offline | needs bots running |
| 2 | FR-01 boundaries | hypersearch log line `[HOLDOUT] FR-01 boundary` (hypersearch_v2.py:1846-1869) | AFTER-CH (log grep); DSR-delta re-score **BLOCKED** (no tool) |
| 3 | FR-04 naive baseline | `scripts/naive_vs_blend.py` | AFTER-CH (Stage-0 dump + cum_trials) |
| 4 | FR-05 rank-IC certificate | hypersearch `[HOLDOUT] FR-05` lines + manifest `holdout.cs_rank_ic` | **BLOCKED under V3=OFF** (fact 0.1) |
| 5 | FR-03 funding drift | `scripts/funding_drift_audit.py` | **NOW** |
| 6 | M1 entry timing | `scripts/entry_timing_probe.py` | AFTER-CH (Stage-0 dump); fill-gap half needs new journals |
| 7 | FR-07-A transfer curves | `scripts/horizon_transfer_report.py` | **NOW** |
| 8 | Seed determinism | none dedicated | **BLOCKED** (no byte-compare tool; proxy below) |
| 9 | FR-02 window A/B | `scripts/window_ab.py` | AFTER-CH (config_v2.pkl + adaptive best_params) + `--epochs`; training-class |
| 10 | LGB_REFIT_FULL A/B | two hypersearch retrains + q10 coverage | blend DSR needs V3; **q10-coverage tool BLOCKED** |
| 11 | Blend coherence | hypersearch `[BLEND]` M2/H2 lines | **BLOCKED under V3=OFF** |
| 12 | FR-16 λ* + FR-08 ledger | `backtest.py --fee-sweep`; `retrain_ledger` rows | sweep AFTER-CH; ledger from retrain #2; IM reader **BLOCKED** |

**Supplementary:** `wave6_stage0.py` **NOW** (crypto run in §3) · `ic_by_name.py` / `rank_gradient_report.py`
AFTER-CH (dump) · `meta_learning_curve.py` AFTER-CH (primary + meta) · `cscv_audit.py` **BLOCKED** (no
`oos_block_perf` producer — hypersearch records only `fold_sharpes`, hypersearch_v2.py:1258; the only hit is the
cscv_audit.py:15 docstring) · `reliability_report.py` **BLOCKED** (no code writes `{p_legacy,p_purged,y}`).

## 2. Step by step

**Step 1 — R2C-01.** Live only: in one legacy retrain cycle with bots up, the champion log must show fresh boosters loading.

**Step 2 — FR-01 instrumentation.** Read: `grep "FR-01 boundary" <scratchpad>/phase5/train_crypto.log` (repeat on
the stock log). The line prints the quantile vs fixed-60d boundary with row counts, then `scored rows`/`trades` for
the active (quantile) boundary. The "re-run evaluate_on_holdout on the same saved winner under both" half has
**no tool** — `evaluate_on_holdout` is called only inside hypersearch and window_ab. Closest proxy (refits, so not
the same weights; GPU/training class; owner-scheduled):
`TRADER_FIXED_HOLDOUT_DAYS=60 python scripts/window_ab.py --arms full --epochs N --out <scratch>/fr01_fixed.json`
vs the same command without the env var → `<scratch>/fr01_quant.json`.
Rule (§4.2): "≥10 calendar-uniqueness effective trades at 60d before any window experiment." Runtime: instant.

**Step 3 — FR-04 naive baseline.**
```
python scripts/naive_vs_blend.py --preds stage0_preds.json --prefix ''            # crypto
python scripts/naive_vs_blend.py --preds stock_stage0_preds.json --prefix stock   # stock
  [--half-life 24.0 --vol-window 72 --cum-trials N]
```
Inputs: Stage-0 dump plus `adaptive_state_{book}.json` `cum_trials` — if absent it WARNs and falls back to 100
(naive_vs_blend.py:114-127), so pass `--cum-trials` = the cumulative pool explicitly. Writes nothing (stdout);
loads only `Ticker,Close` (:53). Light: under 1 min and well under 600 MB for crypto; stock unmeasured.
Rule (§4.3): "blend must beat it on BOTH purged IC and DSR (cum_trials deflation vs n_trials=1); a within-noise
result is an owner report, never an auto-action."

**Step 4 — FR-05 rank-IC certificate.** Produced only on a V3 retrain (fact 0.1). Read: `grep "FR-05" <trainer log>`
or `python -c "import json;print(json.load(open('model_v2.manifest.json'))['holdout'].get('cs_rank_ic'))"`.
Until then the nearest read is per-name time-series IC — **not** the FR-05 cross-sectional statistic:
`python scripts/ic_by_name.py --in stage0_preds.json --time-key ts` (defaults `--min-ic 0.0 --min-consistency 0.6
--min-t 2.0 --subperiods 4`; stdout only). Rule (§4.4): "strongly-positive LGB rank IC ⇒ FR-06 deprioritized;
near-zero/negative ⇒ the in-house justification for the port."

**Step 5 — FR-03 funding drift (NOW).**
```
python scripts/funding_drift_audit.py --out <scratch>/funding_drift_audit.json \
    [--split 2026-01-01 --trailing-days 90 --fwd-bars <h>]
```
Crypto store only, projected to `Ticker + *Funding* + Target_Return_*` (funding_drift_audit.py:244-246); the new
store has 3 such columns (`Funding_Rate_Ann`, `Funding_Z`, `Funding_Chg_24h`). **Always pass `--out`** — the
default writes `research/funding_drift_2026-08.json` inside the repo (:236-238, :286-290). `--fwd-bars` defaults
to the shortest horizon (12). Runtime: seconds; RSS ~250-300 MB (same loader shape as wave6 crypto, 294 MB). Run it
**before the next retrain**.
Rule (§4.5): "flag any Funding_* feature with PSI > 0.25 or an IC sign flip with non-overlapping CIs; the flag
table attaches to the retrain notes (a flag means the retrain re-fits on the shifted distribution — no feature
removal; kill-list survivor #1 boundary)."
**Observed 2026-09-27: not run (HW throttle, CEO directive) — run it when the box is idle.**

**Step 6 — M1 entry-timing probe.**
```
python scripts/entry_timing_probe.py --preds stage0_preds.json --prefix '' \
    [--journal-dir journals --n-boot 500 --seed 0]
```
`--preds` is **required** (entry_timing_probe.py:169), so there is no pre-champion run. Loads `Ticker,Close` only
(:184); writes nothing. The fill-gap half scans `journals/*.jsonl` buys; the only journals are 51 files ending
2026-05-07, so today it reports the old-era gap. Rule (§4.6): "material delta (2·SE, weekly blocks) ⇒ schedule
WINDOW_INCLUDES_ENTRY_BAR at Chain-2; else record the negative and fix the comment only." That flag exists nowhere
in code yet — it is a Chain-2 build item.

**Step 7 — FR-07-A transfer curves (NOW).**
```
python scripts/horizon_transfer_report.py --prefix ''  [--per-name --n-boot 200 --seed 0 --stride <bars>]
python scripts/horizon_transfer_report.py --prefix stock
```
Projected to `Ticker + Target_Return_*` (horizon_transfer_report.py:73-74), so stock reads 6 of 110 columns;
stdout only. Runtime: crypto well under 1 min, ~300 MB; stock unmeasured (1.54M rows × 6 cols). For stock it prints
a CAVEAT: raw Target_Return, not the TB family. The script flags `|rho−null| > 2SE` per pair.
Rule (§4.7): "flat/on-diagonal ⇒ kill the horizon topic cheaply; otherwise sequence FR-07-B's ~8 LGB probes
(trials counted)." **Observed 2026-09-27: not run (HW throttle) — run it when the box is idle.**

**Step 8 — seed determinism.** No tool byte-compares two refits. Two `hypersearch --trials 1` runs would overwrite
the champion slot and bump `cum_trials` — do not. Proxy (no model files; still writes `window_ab_crypto_full_stage0.json` to the root, :344; compares holdout numbers, not bytes):
`TRADER_TRAINER_SEED=42 python scripts/window_ab.py --arms full --seed 42 --epochs N --out <scratch>/seedA.json`,
run twice (`seedA`/`seedB`), `diff` the `arms[0].holdout` blocks. Rule (§4.8): "Pass = FR-02 and FR-13 preconditions met."

**Step 9 — FR-02 window A/B (crypto).**
```
TRADER_FIXED_HOLDOUT_DAYS=60 python scripts/window_ab.py --prefix '' --arms full,730,365,pt2007 \
    --seed 42 --cum-trials <cum> --epochs <N> --out <scratch>/window_ab_crypto.json
    [--data training_data.csv --preset stationary --max-rows 500000]
```
- **Inputs:** `config_v2.pkl` (serving keys); `adaptive_state_crypto.json` `best_params` + `cum_trials`;
  `refit.epochs`, which exists only after a V3 retrain — so pass `--epochs` today.
- **Writes:** the summary at `--out` (default `window_ab_summary_<book>.json` in cwd, window_ab.py:465), and
  **per-arm `window_ab_<book>_<arm>_stage0.json`, always into cwd = the repo root** (:344) — no redirect flag.
- **Cost:** holds the GPU training lock (`acquire_for_training`) and trains one full refit per arm (hours per arm).
  Loads the **whole** store just to read the index max (:425-427, no projection) — the full 705 MB parquet for stock.
**Rule (§4.9):** "adopt shorter ONLY on both the fold-objective AND holdout-DSR wins, winner through backtest.py
--prefix '' --days 60 --gate; else keep full history and record the durable negative (which also skips FR-09)."

**Step 10 — LGB_REFIT_FULL A/B.** Two retrains, `LGB_REFIT_FULL` False vs True (strategy_config.py:281); both need V3
for an identical-holdout *blend* DSR. **No q10-coverage tool exists** — hypersearch_v2.py:1492-1495 says so.
`reliability_report.py` is the META Brier/ECE report (`--in <json with p_legacy,p_purged,y> [--bins 10]`), so
§4.10's "via reliability_report" is plan drift. Rule (§4.10): "non-inferior DSR + sane coverage ⇒ backtest --gate ⇒
challenger→shadow; else stays OFF."

**Step 11 — blend coherence.** On a V3 retrain: `grep "\[BLEND\] M2 side-by-side\|threshold reselect (H2)" <trainer
log>`. Flags: `BLEND_FIT_ON_REFIT` (strategy_config.py:244), `BLEND_THRESHOLD_RESELECT` (:254). `blend_fit.py` has
no CLI; it is the kernel (`fit_blend_weight_v2` :109, `reselect_trade_threshold` :191). Rule (§4.11): "flip
BLEND_FIT_ON_REFIT / BLEND_THRESHOLD_RESELECT only on non-inferior identical-holdout blend DSR with the
certificate's n_trades moving toward the trial-scored trade frequency."

**Step 12 — FR-16 fee sweep + FR-08 ledger.**
```
cp stage0_preds.json <scratch>/stage0_preds.gate44d.json      # FIRST — see "Overwrite" below
python backtest.py --prefix '' --days 180 --fee-sweep '1.0,1.5,2,3,4,6' [--model-prefix challenger]
python backtest.py --prefix stock --days 180 --fee-sweep '1.0,1.5,2,3,4,6'
```
- **Writes (repo root):** `backtest_[<slot>_]fee_sweep.json` (backtest.py:1127-1129) and
  `backtest_[<slot>_]stress_report.json` (:941-943).
- **Overwrite:** it also replaces `{slot}_stage0_preds.json`. Pass 1 keeps the dump on (only k>0 disables it,
  :1099-1100), so the 44-day gate dump becomes a 180-day dump — on crypto mostly **inside the search region**
  (run_pipeline.py:1222-1228). Copy the gate dump first, or pass `--no-stage0-dump`.
- **Cost:** each pass reloads the **full** store, unprojected (backtest.py:728). Crypto is fine; stock means 6
  sequential whole-store loads (multi-GB, unmeasured) — idle box only.
- **Ledger:** accrues automatically into `adaptive_state_{book}.json['retrain_ledger']` (retrain_ledger.py:325-326)
  from retrain #2. No reader exists for the B03.3 IM block-t; the kernel is `shadow.im_cluster_t(dbar, block)`
  (shadow.py:408), which refuses below `V2_MIN_BLOCKS=6` (:93).

Rule (§4.12): "λ* per book/name at --days 180 becomes the standing challenger acceptance metric ('must not reduce
λ*'); the retrain ledger accumulates ≥12 weekly paired rows, then the B03.3 IM block-t decides the cadence question
— an owner decision, never automatic."

## 3. Observed 2026-09-27 on the new stores

`wave6_stage0.py --book crypto --json <scratch>/signal_r1/runbook/wave6_stage0_crypto.json`, via the hwlock arbiter:
exit code 0, peak RSS 294 MB, 3.0 s wall. It writes only the `--json` path (wave6_stage0.py:213-217), and
`git status --ignored` before/after showed no file of ours.
```
=== CRYPTO book: 263889 rows, 6 tickers ===
  fb   labels  u_bar_mean  u_bar_med   u_p10   u_p90  N_eff/N  hold_med
  12   263889      0.0996     0.0846   0.077   0.142    0.100      12.0
  18   263889      0.0823     0.0684   0.053   0.128    0.082      13.0
  24   263889      0.0750     0.0618   0.042   0.123    0.075      13.0
  32   263889      0.0704     0.0577   0.034   0.120    0.070      13.0
  48   263889      0.0666     0.0545   0.028   0.117    0.067      13.0
crypto mean u-bar = 0.079 -> NON-IID: ship uniqueness weights + effective-n DSR (Tier-1)
```
- **OBJECTIVE:** mean ū = 0.079 is far below the script's 0.30 line (wave6_stage0.py:186) — crypto TB labels overlap
  heavily. At fb=24 that is ≈ 263,889 × 0.075 ≈ 19.8k effective labels, so any IID-n significance on crypto rows is
  overstated about 13×.
- **JUDGMENT:** the median hold is 12-13 bars at every horizon from 18 to 48. Barriers resolve most crypto labels
  before the vertical, so the TB family is close to horizon-degenerate at the median above fb≈13 — the crypto
  analogue of the stock EOD caveat. Expect step 7 (raw Target_Return) and the TB labels the model trains on to
  disagree; compare the two before sequencing FR-07-B.
- **Stock: not run.** The arbiter queue timed out behind a 10-minute census, then the HW throttle stopped further
  runs. Its projected load (`Ticker + TB_Bars_*`) was measured on 2026-09-26 at 532 MB (wave6_stage0.py:106-108),
  inside the 600 MB cap.
- **Not run (HW throttle, CEO directive) — run when the box is idle:** funding_drift_audit (step 5),
  horizon_transfer_report crypto + stock (step 7), `wave6_stage0 --book stock`. `<scratch>/signal_r1/runbook/chain.sh`
  runs the first three in sequence with all outputs in the scratchpad; wrap it in `hwlock.sh heavy`.

## 4. Order of operations after the champion lands (one screen)

1. `grep "FR-01 boundary" train_crypto.log` (step 2); confirm `Saved:` / `Manifest: model_v2.manifest.json` and
   `oof_preds.npz`.
2. If not done yet, `funding_drift_audit.py --out <scratch>/…` before the backtest (step 5); attach to retrain notes.
3. `python meta_label.py` — creates the meta artifacts, so the dump carries `meta_p`.
4. `python backtest.py --days 44 --gate` (stock: `--prefix stock --days 60 --gate`) → `stage0_preds.json`
   (`stock_stage0_preds.json`). **Copy it to the scratchpad immediately.**
5. `naive_vs_blend.py --preds <copy> --prefix '' --cum-trials <adaptive cum_trials>` (step 3).
6. `entry_timing_probe.py --preds <copy> --prefix ''` (step 6).
7. `ic_by_name.py --in <copy> --time-key ts`, then
   `rank_gradient_report.py --preds <copy> --fwd-bars 1 --strict --extra-cols meta_p,pred_thresh_ratio`.
8. `backtest.py --prefix '' --days 180 --fee-sweep '1.0,1.5,2,3,4,6'` (step 12); record book + per-name λ* as the
   baseline.
9. Owner decision: a V3 retrain. It unlocks steps 4, 10, 11 and `refit.epochs` for step 9. Then run steps 8 → 9 (seed
   proxy, then window A/B) on an idle box with the GPU.
10. `meta_learning_curve.py --out <scratch>/meta_curve.json` (crypto only — it loads the whole store,
    meta_learning_curve.py:125, so stock is not safe under the 600 MB cap).

## 5. Not answerable until live journals accumulate

- **Step 1 serving integrity:** bots on through one retrain cycle (~1 week). **Step 6 fill gap:** new-era buy rows
  (days for any, ~2-4 weeks for a stable median/p90). **Conviction gate, live side** (`rank_gradient_report
  --buckets decision_report.json`): "≥20-30d of live journals" (its docstring).
- **Shadow DM promotion / challenger→champion:** weeks of shadow preds (`V2_FROZEN_MIN_BARS` crypto 2160 ≈ 90 d,
  shadow.py:94). **FR-08 cadence verdict:** ≥12 ledger rows = ≥13 weekly retrains (~3 months) + an IM reader that
  does not exist. **LLM keep/kill, decision_report, beta ledger:** `D_measurement.md` §4 (≥20-30 d; beta void
  until ~2026-11-11).

## 6. Doc/plan drift and defects found (file:line)

- **06 plan §4.10** says q10 coverage comes "via reliability_report" — no such tool (hypersearch_v2.py:1492-1495;
  MODULES.md:364). **§4.4/§4.10/§4.11** read as if any retrain emits their evidence; it is V3-only (fact 0.1).
- **window_ab.py:344** per-arm Stage-0 dumps always go to cwd (repo root); `--out` redirects only the summary.
  **:75-76** `--prefix` convention is `stock_` (builds `f'{prefix}config_v2.pkl'`, :383) while every other
  instrument takes `stock`; the MODULES.md:1064 census row does not say so. **:425-427** full unprojected store
  load just to read `index.max()`.
- **backtest.py:728, meta_learning_curve.py:125** full-store loads, no projection (stock = multi-GB).
  **backtest.py:1099-1100** the fee sweep clobbers the gate's Stage-0 dump on pass 1.
  **funding_drift_audit.py:236-238** default output lands in the committed `research/` tree.
- **MODULES.md census (lines 1016-1064) vs code:** otherwise consistent for every instrument above (nit:
  meta_learning_curve `--prefix` is `choices=['','stock']`).
