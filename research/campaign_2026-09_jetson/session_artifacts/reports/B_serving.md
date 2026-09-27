# B — Can HEAD 438f56a serve the on-device April-2026 artifacts?

Agent B, 2026-09-26, Jetson, CPU only (`CUDA_VISIBLE_DEVICES=''`), every probe in its own subprocess.
Raw logs: `scratchpad/B/` (`probe_crypto.log`, `probe_stock.log`, `bt_crypto.log`, `bt_stock.log`,
`bt_outputs/`, `cext_*.log`). No production file edited, no training run, no orders placed, no bots started.

## Verdict

1. **Mechanically, yes.** Both books load (`predict_now.load_models`) and produce a live number on
   real Alpaca bars. The number is **LSTM-only**. The LGB mean leg, the q10 tail veto and the
   meta-label gate are all missing, and each one silently goes neutral. **No application log line
   says "LSTM-only".** The only trace is LightGBM's own C++ stderr line
   `[LightGBM] [Fatal] Could not open lgb_q10.txt`, printed once per process, plus snapshots that
   have no `LGB_Pred` / `Q10` keys.
2. **Financially, no.** The April stack never carried a certificate: `config_v2.pkl` has no
   `holdout` and no `lstm_weight`, and the April/May pipeline ran only the 2 search phases, with no
   meta phase and no gate. A 30-day policy replay of it (no `--gate`) gives:
   - **crypto:** n=118, Sharpe **−14.93**, DSR **0.0001**, net −102 %
   - **stock:** n=256, Sharpe **−11.86**, DSR **0.0005**, net −126 %

   Both fail the promotion bar (n≥10 ∧ Sharpe>0 ∧ DSR≥0.60) by a wide margin.
3. **No `.prev` exists anywhere.** A `--gate` run on the April stack would therefore print
   `NOT ROLLED BACK` and leave it deployed.
4. **Nothing in the live path refuses a stale, uncertified champion:**
   - The loops never read the manifest.
   - `model_reload_key` falls back to the constant `.pth` mtime, so there is no reload and no
     error.
   - The meta gate fails open to neutral and says nothing.
5. **`run_pipeline` never detects the missing manifest/LGB legs.**
   - With the systemd unit's `--bot-only` it launches bots straight onto the April stack.
   - A bare run trains unconditionally, but the resumed Optuna studies and the adaptive
     `best_score` ratchet (5.032 crypto / 11.404 stock, set under the old objective) will very
     likely block any save.
   - In **every** outcome it launches the bots afterwards.
6. **Bring-up needs all four stages** — fresh harvest, hypersearch (both books), meta_label and
   `backtest --gate`. Before them come three owner prerequisites:
   - the gotcha-#2 study/`best_score` reset;
   - a bidask decision (runbook Phase 0 step 2 says STOP before harvest);
   - the untracked C extension, which **crashes on frames under 99 bars** (new finding, below).

   Estimated first-training wall time is **≈ 45–61 h of search alone** (200 trials × 2 books).
   Estimated peak RAM is **≈ 3.5–4.5 GB** for the stock search process. No RSS telemetry exists, so
   that is an estimate.

---

## 1. `load_models` + artifact contents + feature compatibility

Command: `CUDA_VISIBLE_DEVICES='' $JPY scratchpad/B/serve_probe.py crypto ETH/USD` (and `stock AAPL`).

```
[JIT] Model traced successfully
Model loaded: seq=18, threshold=0.51, fb=64, heads=2                  (crypto)
stock Model loaded: seq=20, threshold=1.31, fb=96, heads=8            (stock)
config keys: ['dropout','forward_bars','hidden_dim','huber_delta','indicator_preset','input_dim',
              'model_version','n_heads','num_layers','prefix','seq_len','trade_threshold']
lstm_weight in config: False | holdout cert: False
```

| | crypto `config_v2.pkl` | stock `stock_config_v2.pkl` |
|---|---|---|
| seq_len / forward_bars | 18 / 64 | 20 / 96 |
| trade_threshold | 0.51 | 1.31 |
| hidden/layers/heads/dropout | 288/2/2/0.25 | 160/2/8/0.10 |
| lstm_weight | **absent** (live and backtest default to 0.6, used only if an LGB leg exists) | absent |
| holdout certificate / blend_diag / target_kind | **absent** | absent |
| prefix | `''` | `'stock'` (correct form for `predict_now.py:449-450`) |

**Feature list.** Both pickles hold the same 23 names: the first 23 entries of today's
`_STATIONARY_FEATURES` (`indicator_config.py:130-155`), ending with `Hurst, Month_*, Turn_of_Month,
Daily_Sentiment`.
- All 23 are present in both parquets (crypto 52 cols, stock 54 cols).
- All 23 are still in today's `stationary` preset.
- Today's `compute_features` / `compute_stock_features` emit 22 of them. `Daily_Sentiment` is
  injected at `predict_now.py:299-318`, as it always was.
- So **no name incompatibility**, in either direction, for serving the old pickles.

Going the other way, the April models do not use features today's code adds:
- **Stock:** today's code emits 20 stationary columns that are in neither the pkl nor the old
  parquet: `ROD_Ret, Same_Hour_Mean_40d, RM_252_21, Ret_21d, RR_5, RR_21, MA_Dist_{10,20,50,100,200}d,
  ON_Mom_21/252, TugOfWar_252, Pos_Range_20h/60h/20d/60d, MidRange_Gap_20h/60h`.
  (CS_*/SVR are panel/archive injections.)
- **Crypto:** today's `compute_features` emits nothing new. Funding/OI/TT/Taker are live
  injections that appear only after a new harvest.
- A retrain on a re-harvested store therefore **changes input_dim**. That is gotcha #2, and the
  April scaler/pkl cannot be reused.

**Value parity: does today's code reproduce the values the April model trained on?**
Script: `scratchpad/B/parity2.py`. It recomputes features from the parquet's own OHLCV and
compares them with the stored columns.

- **Crypto: essentially reproducible.**
  - ETH-USD on the last 400 bars: stored `ROC == pct_change(12)` and
    `Return_4h == pct_change(4)` on 100 % of rows.
  - Over 1500 rows, residual mismatches are 0.3–5 % of rows (RSI/Hurst more, especially BTC-USD).
  - That pattern is consistent with the stored OHLCV having been revised after features were
    stamped (01_state_map **D08**, the Yahoo overwrite), not with a code change.
- **Stock: NOT reproducible.**
  - NVDA, last 400 bars: stored `ROC` matches `pct_change(12)` of stored Close on only
    **11.3 %** of rows. `Return_4h` matches on 65 %.
  - Over 1500 rows: RSI mismatches on 100 % of rows (median |Δ| 0.48), Hurst on 97 %,
    Volume_Ratio on 42 %.
  - The store holds a mixed grid, including a `2026-02-19 20:30 UTC` half-hour bar among hourly
    bars.
  - So the stored stock features were computed on a different bar set than the OHLCV now stored.
    The numba and C paths agree with each other here, so this is **data-store drift, not
    code-path drift**.
  - Consequence: the next harvest ("Recompute ALL features on full history",
    `harvest_stock_data.py:187`) will rewrite the historical stock feature values. This is another
    reason the April stock scaler/model cannot be paired with a new store. **Handoff to the
    harvest/data agent.**

## 2. Live prediction (`trading_utils.predict_symbol` → `predict_now.get_live_prediction`)

The live entry is `base_loop._get_predictions` (`base_loop.py:1547-1555`) →
`trading_utils.predict_symbol` (`trading_utils.py:159-187`) → `get_live_prediction`. The probe
called exactly that, with the loops' benchmark (BTC/USD close for crypto, SPY close for stock) and
read-only Alpaca bars.

```
--- ANALYZING ETH/USD ---
[LightGBM] [Fatal] Could not open lgb_q10.txt
Current Price:   $2693.77
Predicted Return: +0.0408% (threshold=0.51)
Recommendation:  [HOLD/WEAK]
RESULT pred = 0.04081900417804718
LGB_Pred in snap: None | LSTM_Pred: 0.04081900417804718 | Q10: None
predict_now._lgb_models = {'': None} | _q10_models = {'': None}
lstm-only? True
meta_probability_live -> None
wall 7.6s maxrss 630 MB

--- ANALYZING AAPL ---
  [DAILY-FEATURES] AAPL: 12/20 daily-window cols warmup-filled constants
[LightGBM] [Fatal] Could not open stock_lgb_q10.txt
Predicted Return: -0.7070% (threshold=1.31)
RESULT pred = -0.7070011496543884   (== LSTM_Pred; no LGB_Pred, no Q10)
wall 7.8s maxrss 629 MB
```

**It produces a number; it does not fail closed. It is LSTM-only, and the log does not say so.**
- **LGB mean leg.** `model_lgb.load_lgb_model` (`model_lgb.py:131-132`) returns `None` on a missing
  file, with no log line. `predict_now.py:461-464` then keeps `predicted_return = lstm_pred`.
- **q10 leg.** The loader at `predict_now.py:479-485` calls `lgb.Booster(model_file=...)` with no
  existence check. The resulting `[LightGBM] [Fatal]` line (LightGBM's own stderr) is the **only**
  trace of the missing legs.
- **Caching.** `serving_cache.cache_get` caches `None` under key `None`
  (`serving_cache.py:69-76`), so it never retries, and never re-logs, until the files appear. The
  line therefore appears once per process per prefix.
- **Downstream effects.** The snapshot carries `LSTM_Pred` but no `LGB_Pred` / `Q10` / `Q10_Floor`,
  so the q10 entry veto never fires. Journals lack `lgb_pred` / `q10` (`base_loop._conv_fields`).
- AAPL is not in the 46-name trained stock universe. Irrelevant for this check: the April model
  takes no symbol-specific input.

## 3. Hot-reload with no manifest

- `trading_utils.model_reload_key` (`trading_utils.py:113-127`) returns
  `get_model_mtime('{p}model_v2.manifest.json')` (0 when missing), falling back to the `.pth`
  mtime. `get_model_mtime` swallows `OSError` (`:105-110`). Probe output:
  `model_reload_key -> 1775928622.9051356 | .pth mtime 1775928622.9051356` (crypto), and the stock
  value equals `stock_model_v2.pth`'s mtime.
- `base_loop._load_models` (`base_loop.py:434-448`, called at `:277`) catches **only**
  `FileNotFoundError`. When files are missing it sets model=None, and buys are disabled (fail
  closed, `:1531`). It then stores `self.model_mtime = model_reload_key(...)` = the `.pth` mtime.
  Other load exceptions, such as a state-dict mismatch or a corrupt pickle, are not caught there.
- `base_loop._hot_reload_check` (`base_loop.py:890-925`, called each cycle at `:352`) compares the
  key and reloads only if it changed.
- **Behaviour with no manifest:** the key is constant, so there is no reload, no exception and no
  log spam. The first manifest written by a future hypersearch changes the key and triggers a
  reload, which pops the booster caches (`:914-920`). A failed reload backs off 300 s per key
  (`:895-902`). Behaviour is benign.

## 4. `backtest.py` without `--gate`

**Proof that no-gate mode is write-free for champion artifacts**, from reading `backtest.main`
(`backtest.py:1152-1438`):
- `restore_previous_model` (the only code that renames or removes artifacts, `:633-690`) is reached
  only inside `if args.gate:` (`:1323` → `:1390`).
- `notify`, `_patch_report_gate_block` and `_write_policy_gate_sidecar` are likewise gate-only
  (`:1308-1320`, `:1323-1436`).
- `run_backtest` (`:693-973`) only *reads* artifacts: `_load_artifacts` / `_load_lgb` /
  `_load_q10`, which use `map_location='cpu'`.
- It writes exactly two files: `{slot}stage0_preds.json` (`:910-913`) and
  `backtest[_slot]_report.json` (`:944-959`). Both are gitignored (`.gitignore:79,114`) and neither
  existed before.
- Verification: the `ls` of `*.pth` / `*.pkl` taken before and after the runs was byte-identical
  (`ARTIFACTS_UNCHANGED`), and no `.prev` was created. I moved my four generated report files to
  `scratchpad/B/bt_outputs/`, so the repo root is back to its prior state.

Command: `CUDA_VISIBLE_DEVICES='' $JPY scratchpad/B/rss_wrap.py backtest.py --prefix '' --days 30`
(then the same with `--prefix stock`).

| | crypto | stock |
|---|---|---|
| data loaded | 234,822 rows (6 names, 2021-01-13 → 2026-02-24) | 1,236,699 rows (46 names, 2016 → 2026-02-19) |
| period | 2026-01-25 → 2026-02-24 | 2026-01-20 → 2026-02-19 |
| n_trades | **118** | **256** |
| Sharpe | **−14.933** | **−11.857** |
| DSR (n_trials=100) | **0.0001** | **0.0005** |
| n_eff clustered / calendar | 23 / 50.8 | 18 / 24.3 |
| net / gross / fees % | −102.2 / −31.4 / 70.8 | −126.2 / −97.3 / 28.9 |
| win rate | 0.271 | 0.402 |
| meta_veto_active / LGB / q10 | False / absent / absent (`[LightGBM] [Fatal] Could not open …lgb_q10.txt`) | same |
| coverage | 6/6 | 46/46 |
| wall / peak RSS | 16.7 s / **1006 MB** | 32.3 s / **1644 MB** |

- The replay runs cleanly on the old artifacts plus the old parquet.
- **Gross** P&L is negative in both books, so the April models show no edge before costs.
- `Eff_Spread_Pct` is absent from the old stores, so the flat-cost fallback was used
  (`backtest.py:330-341`).

## 5. Shadow evaluation with no challenger

- `evaluate_and_maybe_promote` returns `None` at `shadow.py:1042-1043`
  (`if not challenger_manifest(prefix).exists()`), before any status or ledger write.
- `evaluate_shadow` → `_load_rows` (`shadow.py:556-576`) returns `[]` when there is no
  `{p}shadow_preds.jsonl`, and the function then returns `None` (`:634-636`).
- Both are side-effect-free here, so I ran them:
  ```
  crypto champion_exists= False challenger_manifest exists= False
    evaluate_shadow -> None / evaluate_and_maybe_promote -> None
  stock  (identical)
  ```
- **Note:** `champion_exists()` keys on the **manifest** (`shadow.py:115-116`), so the April
  artifacts do **not** count as a champion. Any `hypersearch_v2 --shadow` (the weekly retrain
  path) therefore saves into the **champion** slot, not the challenger slot
  (`scripts/hypersearch_v2.py:2839-2848`). The shadow/DM-HLN path cannot engage until one
  manifest-bearing champion exists.

## 6. Meta gate with no meta artifacts

- `meta_label._load` (`meta_label.py:519-544`) computes a stat key containing `None`, skips loading
  and returns `None`.
- `meta_probability_live` then returns `None` (`:601-603`). The probe confirms
  `meta_probability_live -> None` for both books.
- `base_loop._meta_gate` (`base_loop.py:2050-2068`) maps `None` to `(True, 1.0)`. The result is
  **fail-open, neutral, and silent.** The warning branch (`:2085-2098`) fires only on exceptions,
  not on absence.
- **Documented intent:**
  - fail-open/neutral: `base_loop._meta_gate` docstring ("No trained meta model -> neutral"),
    `meta_label.meta_probability_live` docstring ("strictly optional and absent models leave
    behavior unchanged"), and `docs/MAP.md:558` (§4e gate table row 15: "fail open").
  - **Doc inconsistency:** `docs/MAP.md` §5 (8) (≈ line 744) says "the LLM gate is fail-*open* …
    every other gate is fail-closed". That contradicts the meta gate's actual and documented
    fail-open behaviour. It should list the meta gate as a second fail-open gate.

---

## New finding: the untracked C extension overflows the heap on short frames

`indicators_c.cpython-310-aarch64-linux-gnu.so` (built 2026-02-28 15:51, after the 02:05 harvest)
is auto-used (`indicators.py:14`, `_HAS_C=True`). Its NaN-prefix loops write `window-1` slots
regardless of `n`:

```c
for (npy_intp i = 0; i < window - 1; i++)   out[i] = NAN;   // c_ext/indicators_c.c:89,106,117,427
```

- **Trigger** (`scratchpad/B/cext_small_*.log`, ETH frames, 3 `compute_features` calls each):
  - n = 0, 1, 5, 13, 27, 60, 65, 75, 85, 92, 96 → SIGSEGV/SIGABRT
    (`double free or corruption`, `array is too big`);
  - n = 99, 100, 101, 120, 200, 248 → survive.
  - The threshold is the 100-bar window.
- **Serving risk.** Live fetches request 250 (crypto) / 320 (stock) bars (`market_data.py:195,267`),
  so a normal cycle is safe. Any symbol returning under ~99 bars (new listing, halt, API partial)
  would corrupt the heap. Under `--combined-bots` that takes down both books, or corrupts memory
  silently.
- **Harvest risk.** Any universe name with fewer than 99 rows would do the same to the harvest.
- **Recommendation (owner):** move it to `archive/` before the bring-up. Delete nothing. The
  numba path is the one the tests pin.
- This is also the root cause of the orchestrator's pytest SIGABRT: my first parity probe died the
  same way when handed an empty frame.

## Other side effect observed

`sentiment_cache.db` shows mtime 20:13:22, and its `-wal`/`-shm` files are gone (checkpointed). The
stock probe's `get_live_daily_sentiment` opens that DB in WAL mode (`sentiment_history.py:76-91`),
so this is most likely my read connection's close-checkpoint. A concurrent agent is also possible.
The content was read-only.

---

## What `run_pipeline` does on first start with these artifacts

`run_pipeline.py` has **no** artifact detection: no reference to manifest, `lgb_model` or `.pth`
anywhere. There are two cases.

**(a) The shipped systemd unit** (`scripts/setup_jetson_system.sh:170`:
`run_pipeline.py --combined-bots --bot-only`):
- Phase A is skipped (`run_pipeline.py:1450`) and the bots launch immediately (`:1503`) on the
  April stack: LSTM-only, no meta, no q10.
- On Saturday 02:00 the weekly retrain (`_build_training_phases(..., shadow=True)`) runs harvest →
  hypersearch `--shadow`. Because `champion_exists` is False, that saves into the champion slot,
  followed by meta and the champion-slot `--gate`.

**(b) A bare `python run_pipeline.py`:**
- Phase A always runs: harvest (both stores are more than 24 h old, `:945-978`), then 200-trial
  `hypersearch --mode initial` with no `--shadow`, then meta_label, then `backtest --gate`
  (`:1002-1111`, `:1452-1497`).
- The bots launch afterwards **whatever the outcome** (`:1503`). A failed phase logs "continuing to
  bot phase with existing models" (`:1261-1262`).
- A gate rc=3 is final and counted as success (`:1215-1220`).

In both cases the first search resumes the old studies (`load_if_exists=True`,
`hypersearch_v2.py:2322`):
- The seeded "prior best" is 5.032 / 11.404 from 384 / 669 old-objective trials
  (`:2350-2357`; `v2_study.db` has 200 COMPLETE + 184 PRUNED, and `stock_v2_study.db` has
  599 COMPLETE + 69 PRUNED + 1 stuck RUNNING).
- The save also has to beat `adaptive_state.best_score` = 5.0316 / 11.4045 under the legacy strict
  ratchet (`:2379`, `:2430`), and then pass the holdout DSR gate (`:2788-2797`).
- In the May cycle the old objective itself failed to beat these (pipeline_output.log:9795
  `No new best found (prior best score=5.032)`, :10481 same for stock). A changed objective makes a
  save even less likely.

**Most likely outcome of (a) or (b) without the reset:**
1. The search runs for about a day per book and saves nothing (rc 0).
2. meta_label trains a meta layer on the **April** primary (it tolerates a missing manifest:
   `meta_label.py:859-865,1147-1158`).
3. `--gate` replays the April stack and fails (see §4).
4. `restore_previous_model` finds no `.prev` → `[GATE] *** NOT ROLLED BACK … still deployed ***`
   plus a critical notify (`backtest.py:1388-1418`), exit 3.
5. The bots launch on the April stack, now with a meta layer.

**If a new model *does* save** (champion slot):
- `save_model_atomically` copies the April four to `.prev` (`hypersearch_v2.py:2106-2119`).
- If the new model then fails the gate, `restore_previous_model` restores the April four **and
  deletes the new manifest/LGB/q10/oof/meta as never-gated orphans** (`backtest.py:633-690`). That
  returns the device to exactly today's state, and the bots launch on it.

## Recommended bring-up sequence (minimum)

**Is full harvest + hypersearch (both books) + meta_label + `backtest --gate` mandatory?**
- To *serve*, no. The code serves the April stack today.
- To serve **what the code is built around, with any evidence of edge**, yes, all four:
  - There is no supported way to fit LGB/q10 boosters to an existing LSTM:
    `train_lgb_ensemble` runs only inside hypersearch.
  - The old stores lack Eff_Spread/CS_*/Funding_*/TB_* and are 7 months stale.
  - The April LSTMs replay at Sharpe −15 / −12 and gross-negative P&L.

**Owner prerequisites, before anything runs:**
1. **Gotcha #2 reset** (runbook Phase 3): move `v2_study.db` and `stock_v2_study.db` to
   `archive/`. Moving is better than `--fresh`, which `os.remove`s the file at
   `hypersearch_v2.py:2210-2215`. Also reset `best_score` and `cum_trials` in
   `adaptive_state_{crypto,stock}.json`. Without this, the first search almost certainly saves
   nothing.
2. **bidask:** absent from the jetson env. Runbook Phase 0 step 2 says STOP before any harvest,
   because otherwise `Eff_Spread_Pct` comes from the upward-biased AR fallback
   (`liquidity.py:181-191`). Install it before the first new-schema harvest, or accept the fallback
   explicitly.
3. **C extension:** move `indicators_c*.so` (and optionally `c_ext/`) to `archive/` (see finding
   above).
4. **Decide the Phase-3 flag bundle now** (`HYPERSEARCH_V3`, `OBJECTIVE_V3`, optional
   `OBJECTIVE_LONG_ONLY`/lean preset). This first retrain *is* the gotcha-#2 event.
   - With V3 OFF, the legacy flow writes the manifest before the LGB boosters.
   - With V3 OFF, the config gets no `lstm_weight`, so the blend runs at the 0.6 default.
5. **Keep the April four in place.** They become the `.prev` rollback target. Moving them away
   means a failing new model would stay deployed ("NOT ROLLED BACK").

**Then run each step by hand, not via `run_pipeline`**, because Phase B launches bots regardless of
gate outcome:
1. `scripts/harvest_crypto_data.py`, then `scripts/harvest_stock_data.py`. These are incremental,
   cover the 7-month gap, and recompute all features on the full history.
2. `scripts/hypersearch_v2.py --trials 200 --preset stationary --mode initial` for crypto, and the
   same plus `--data stock_training_data.csv --prefix stock --max-rows 200000` for stock. No
   `--shadow` is needed: with no manifest it saves to the champion slot either way.
3. `meta_label.py` and `meta_label.py --prefix stock`.
4. `backtest.py --days 44 --trials 200 --gate` and
   `backtest.py --prefix stock --days 60 --trials 200 --gate`.
5. Start bots only for books whose gate **PASSED**:
   `run_pipeline.py --combined-bots --bot-only [--crypto-only|--stock-only]`.
   - A book that fails gets rolled back to the April stack (Sharpe −15/−12). It should stay off:
     do not trade it.
   - A single-book start still leaves the weekly retrain covering only that book.
   - Only after one gated champion exists does the shadow/DM-HLN path (`--shadow` → challenger)
     engage.

## First full-training cost on this box (evidence)

**Wall time**, from `pipeline_output.log` (GPU, old code, 23 features). Phase durations as logged:

| run | trials | min | min/trial |
|---|---|---|---|
| crypto 2026-04-04 | 120 | 1093.6 (`:3685`) | 9.1 |
| crypto 2026-04-11 | 70 | 629.9 (`:4870`) | 9.0 |
| crypto 2026-04-18 | 120 | 789.7 (`:6150`) | 6.6 |
| crypto 2026-04-25 | 70 | 525.0 (`:7579`) | 7.5 |
| crypto 2026-05-02 | 120 | 740.3 (`:9792`) | 6.2 |
| stock 2026-04-04 | 120 | 895.3 (`:4389`) | 7.5 |
| stock 2026-04-11 | 70 | 597.3 (`:5350`) | 8.5 |
| stock 2026-04-18 | 120 | 887.1 (`:6832`) | 7.4 |
| stock 2026-04-25 | 70 | 646.2 (`:8015`) | 9.2 |
| stock 2026-05-02 | 120 | 884.5 (`:10478`) | 7.4 |

Cross-checks:
- Optuna DB per-trial durations: crypto median 340 s, p90 903 s, 44.7 h over 384 trials; stock
  median 398 s, p90 813 s, 84.6 h over 669 trials.
- `hypersearch_v2_log.json`: 121 entries in 740.3 min (complete-trial median 609 s, pruned 147 s).
- `hypersearch_stock_v2_log.json`: 121 entries in 884.5 min (complete median 382 s).

Estimates for `--trials 200` per book (run_pipeline's initial default):
- crypto ≈ 1,240–1,820 min (**21–30 h**);
- stock ≈ 1,480–1,840 min (**25–31 h**);
- search total **≈ 45–61 h**.

On top of that:
- harvest: never timed in any log; a 7-month incremental gap plus full-history recompute;
- meta_label: minutes, not measured;
- two gates: about 17 s and 32 s each, measured on CPU today.

Expect the upper end or beyond, because:
- the K=4 checkpoint soup and V3 refit/LGB-pre-gate (if flipped) add work;
- stock input_dim grows from 23 to about 60+ stationary columns after re-harvest (rows stay capped
  at 200k), and crypto from 23 to about 31 with Funding/OI/TT/Taker;
- crypto rows grow about 13 % (uncapped, `--max-rows` 500k default).

**Peak RAM: no RSS telemetry exists in any log.** Evidence:
- Measured today: full-frame load plus torch on CPU is 1,644 MB for the stock store
  (1.24 M × 54 cols) and 1,006 MB for crypto. A live probe is 630 MB.
- Per-fold scaled caches: up to `506+78 MB` (stock, seq 32; training_test.log) and `337+52 MB`
  (crypto, pipeline_output.log:3149).
- GPU unified-memory pressure: 154 `NvMapMemAllocInternalTagged … error 12` lines and 20 `OOM`
  lines in pipeline_output.log. Examples: `[OOM-RETRY] fold 0: batch 4096→2048` (`:5222`);
  a failed 1.07 GB NvMap allocation in training_test.log.
- `hypersearch_v2.load_data` loads the **full** frame before the row cap
  (`hypersearch_v2.py:156-158` → `data_utils.load_training_data`, no column subset).

Estimate after a new-schema harvest (stock ≈ 1.3 M rows × ~100 cols ≈ 1.1 GB of float64 plus
pandas copies):
- stock hypersearch: **≈ 3.5–4.5 GB** peak (2–3 GB transient load, then CUDA context ≈ 1 GB plus
  fold caches ≈ 0.6 GB plus activations up to the NvMap ceiling);
- stock meta_label/backtest: ≈ 2–3 GB;
- crypto: ≈ 1.5–2.5 GB.

That fits in 7.4 GB (+12 GB swap) with the bots stopped. Running the bots alongside, especially the
stock search, is the risk.
