# STATE_FILES.md — every runtime file the system reads or writes

*Single home for the runtime-file inventory (other docs point here; they do not repeat these
paths). 88 canonical paths + their prefix variants, code-verified 2026-09-08; `git check-ignore`
re-run against the working-tree `.gitignore` on the same date. Line numbers are a snapshot; the
`module.function` anchors are stable.*

## Where state lives

**The repo root IS the state directory.** Every runtime path is anchored to
`Path(__file__).resolve().parent` / `os.path.dirname(os.path.abspath(__file__))` — deliberately, so
the two loop processes, `run_pipeline.py` and `gui.py` agree on one file regardless of launch CWD
(`risk_budget.py` states the convention above `ACCOUNT_RISK_REGISTRY`). There is **no `state/` or
`var/` subdirectory**, and no `models/` directory despite the `.gitignore` rule for one. Only two
runtime *directories* exist: `journals/` (created lazily by `trade_journal.log_decision`) and
`logs/` (created by `log_config._setup` on the first `get_logger` call).

**Prefix convention** — one artifact family, four slots:

| Prefix | Slot | Produced by |
|---|---|---|
| `''` (none) | crypto champion | `data_utils._FILE_STEMS['crypto'] = 'training_data'`, `scripts/hypersearch_v2.py` |
| `stock_` | stock champion | same, `--prefix stock` |
| `challenger_` | crypto shadow slot | `shadow.challenger_prefix('')` |
| `stock_challenger_` | stock shadow slot | `shadow.challenger_prefix('stock')` |

`shadow.challenger_prefix` is the one authority (`shadow.py`); every artifact path is then built as
`f'{prefix}_'` + stem via `shadow._p` / `meta_label._paths` / `monitor_drift._p`. The crypto book's
training data is `training_data.parquet`, **not** `crypto_training_data.parquet` — a name nothing
writes (`gui.py:6922` passes it to the lead/lag button and that button therefore always fails).

**Write conventions observed**

- **Atomic JSON writes** — sibling `.tmp` + `os.replace()` in `data_utils._atomic_to_disk`,
  `decision_report._write_json`, `llm_analyst._save_analysis`, `meta_label._write_refusal`,
  `gpu_lock._write_info`, `stage0_preds.write_rows`, `learned_lexicon.write_json_atomic`,
  `stock_config.save_stock_universe`, `indicator_config.save_indicator_config`,
  `llm_config.save_llm_config`, `funding._save_history`, `novelty._save`, `beta_ledger`,
  `llm_eval`, `risk_budget.write_registry` (per-writer tmp keyed by pid **and** thread id, because
  `--combined-bots` runs both loops as threads in one process).
- **Advisory `flock` sidecars** (`fcntl`, optional import, no-op where unavailable) —
  `risk_budget.py` (`account_risk_registry.json.lock`, bounded non-blocking acquire so a stalled
  peer can never park the trading loop), `trade_memory.py`, `monitor_drift.py`
  (`pred_history.jsonl.lock`), `llm_client.py` (`llm_cost.json.lock`), and `gpu_lock.py` itself
  (`.gpu.lock`, `LOCK_EX`/`LOCK_SH`).
- **Two-phase promotion** — `<artifact>.staged` → `os.replace` → live, with `<artifact>.prev` kept
  for rollback and `<artifact>.stale` as a quarantine marker (`meta_label.py`, `shadow.py`,
  `backtest.restore_previous_model`).

**Gitignore rule: every generated file must be ignored.** As of 2026-09-08 that holds, with two
deliberate exceptions: `stock_universe.json` (hand-edited, committed config read by
`stock_config.load_stock_universe`) and `research/funding_drift_2026-08.json`
(`scripts/funding_drift_audit.py`, which writes into the intentionally-committed `research/` tree).
Sixteen patterns were missing before this pass and are marked **added 2026-09-08** below; they sit
in a labelled block at the end of `.gitignore` (`:131` onward). `.git/info/exclude` additionally
hides `.claude/RESUME.md` — machine-local, so it does **not** travel to the Jetson.

---

## 1. Training artifacts — weekly retrain, Jetson only

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `model_v2.pth` — `stock_model_v2.pth`, `challenger_model_v2.pth`, `stock_challenger_model_v2.pth` | `scripts/hypersearch_v2.py:2091 (save_model_atomically)` | `predict_now.py:81`, `backtest.py:114`, `trading_utils.py:125`, `gui.py:140-141`, `retr… | jetson | weekly retrain | `*.pth` :21 · `challenger_*` :122 · `stock_challenger_*` :123 |
| `config_v2.pkl` — `stock_config_v2.pkl`, `challenger_config_v2.pkl` | `scripts/hypersearch_v2.py:2092` | `predict_now.py:82`, `backtest.py:107`, `gui.py:135-136`, `retrain_ledger.py:182`, `sha… | jetson | weekly retrain | `*.pkl` :22 · `challenger_*` :122 |
| `scaler_v2.pkl` — `stock_scaler_v2.pkl`, `challenger_scaler_v2.pkl` | `scripts/hypersearch_v2.py:2093` | `predict_now.py:83`, `backtest.py:108`, `retrain_ledger.py:183` | jetson | weekly retrain | `*.pkl` :22 · `challenger_*` :122 |
| `feature_cols_v2.pkl` — `stock_feature_cols_v2.pkl`, `challenger_feature_cols_v2.pkl` | `scripts/hypersearch_v2.py:2094` | `predict_now.py:84`, `backtest.py:109`, `retrain_ledger.py:184` | jetson | weekly retrain | `*.pkl` :22 · `challenger_*` :122 |
| `model_v2.manifest.json` — `stock_model_v2.manifest.json`, `challenger_model_v2.manifest.json`, `stock_challenger_model_v2.manifest.json` | `scripts/hypersearch_v2.py:2100,2142` | `backtest.py:880,892,1011`, `shadow.py:116,120,862,1114`, `meta_label.py:359,862,1154`,… | jetson | weekly retrain | `*model_v2.manifest.json` :135 **added 2026-09-08** |
| `lgb_model.txt` — `stock_lgb_model.txt`, `challenger_lgb_model.txt`, `stock_challenger_lgb_model.txt` | `scripts/hypersearch_v2.py:2101,2868; model_lgb.py:114,129` | `backtest.py:129`, `predict_now.py:458`, `retrain_ledger.py:268-271`, `shadow.py:228` | jetson | weekly retrain | `*lgb_model.txt` :136 **added 2026-09-08** |
| `lgb_q10.txt` — `stock_lgb_q10.txt`, `challenger_lgb_q10.txt` | `scripts/hypersearch_v2.py:1469,2102,2872` | `backtest.py:144`, `predict_now.py:482,489`, `shadow.py:231` | jetson | weekly retrain | `*lgb_q10.txt` :137 **added 2026-09-08** |
| `lgb_q10_meta.json` — `stock_lgb_q10_meta.json`, `challenger_lgb_q10_meta.json` | `scripts/hypersearch_v2.py:1473,2103,2888` | `backtest.py:149`, `predict_now.py:483,489`, `shadow.py:232`, `scripts/backup_state.sh:… | jetson | weekly retrain | `*lgb_q10_meta.json` :138 **added 2026-09-08** |
| `oof_preds.npz` — `stock_oof_preds.npz`, `challenger_oof_preds.npz` | `scripts/hypersearch_v2.py:2104,2125` | `meta_label.py:867`, `shadow.py:101`, `backtest.py:596`, `scripts/meta_learning_curve.p… | jetson | weekly retrain | `*oof_preds.npz*` :106 · `challenger_*` :122 |
| `meta_model.txt` — `stock_meta_model.txt` | `meta_label.py:79 (_paths) / train pipeline` | `meta_label.py:79`, `shadow.py:103 (_STALE_META_SUFFIXES)`, `backtest.py:591` | jetson | weekly retrain (post-LSTM) | `*meta_model.txt` :86 |
| `meta_calib.pkl` — `stock_meta_calib.pkl` | `meta_label.py:80` | `backtest.py:591`, `shadow.py:103` | jetson | weekly retrain | `*meta_calib.pkl` :87 |
| `meta_meta.json` — `stock_meta_meta.json` | `meta_label.py:81,303` | `backtest.py:592`, `gui.py:173-174`, `shadow.py:103` | jetson | weekly retrain | `*meta_meta.json` :88 |
| `meta_refused.json` — `stock_meta_refused.json` | `meta_label.py:82,304 (_write_refusal)` | `gui.py:175-176` | jetson | weekly retrain (on publish-guard refusal) | `*meta_refused.json` :103 |
| `<artifact>.staged` | `meta_label.py:329,1216 (--train-staged)` | `meta_label.py:1255 (--promote-staged)` | jetson | weekly retrain (two-phase promote) | `*.staged` :101 |
| `<artifact>.prev` | `scripts/hypersearch_v2.py:2107; meta_label.py:375; retrain_le…` | `backtest.py:647,1375 (restore_previous_model)`, `retrain_ledger.py:271` | jetson | weekly retrain (rollback pair) | `*.prev` :81 |
| `<artifact>.stale` | `meta_label.py:394; shadow.py:799` | `—` | jetson | promotion (quarantine of stale meta leg) | `*.stale` :102 |
| `v2_study.db` — `stock_v2_study.db` | `optuna via scripts/hypersearch_v2.py:2191` | `gui.py:196-198`, `scripts/backup_state.sh:22` | jetson | weekly retrain (append trials) | `*_study.db` :26 |
| `adaptive_state_crypto.json` — `adaptive_state_stock.json`, `adaptive_state_testq1.json (TEST RESIDUE)` | `adaptive_config.py:90 (save_adaptive_state); retrain_ledger.p…` | `run_pipeline.py:36-37`, `scripts/hypersearch_v2.py`, `scripts/naive_vs_blend.py:112` | both | weekly retrain + test runs | `adaptive_state*.json` :34 |
| `hypersearch_v2_log.json` — `hypersearch_stock_v2_log.json` | `scripts/hypersearch_v2.py:2302,3033` | `—` | jetson | weekly retrain | `hypersearch_*_log.json` :37 |

The four **champion** halves (`*model_v2.manifest.json`, `*lgb_model.txt`, `*lgb_q10.txt`,
`*lgb_q10_meta.json`) were the highest-value gitignore gap: their `challenger_*` /
`stock_challenger_*` halves were already ignored by `.gitignore:122-123`, so the same
`scripts/hypersearch_v2.py` save block produced half-tracked, half-ignored output.

`{prefix}model_v2.manifest.json` is the **certificate**: the fingerprint every downstream consumer
checks (`backtest.py`, `shadow.py`, `meta_label.py`, `monitor_drift.py`, `trading_utils.py`,
`gui.py`, `scripts/meta_learning_curve.py`). `{prefix}oof_preds.npz` is fingerprinted to that
manifest's `saved_at`+`score`, so a stale npz is detected rather than silently consumed.
Blend weight and trade threshold are **not** separate files — they live inside
`{prefix}config_v2.pkl` (`lstm_weight`, `trade_threshold`; `blend_fit.effective_lstm_weight`).

## 2. Harvest data + feature archives

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `training_data.parquet` — `training_data.csv` | `data_utils.save_training_data (data_utils.py:151-166) via scr…` | `data_utils.get_data_path:65`, `scripts/hypersearch_v2.py:124`, `scripts/window_ab.py:3… | jetson | harvest (weekly / pipeline start) | `training_data.parquet` :17 · `training_data.csv` :15 |
| `stock_training_data.parquet` — `stock_training_data.csv` | `data_utils.save_training_data via scripts/harvest_stock_data.…` | `scripts/hypersearch_v2.py`, `scripts/window_ab.py:370`, `scripts/train_lexicon.py:37`,… | jetson | harvest (weekly / pipeline start) | `stock_training_data.parquet` :18 · `stock_training_data.csv` :16 |
| `raw_ohlcv.parquet` — `stock_raw_ohlcv.parquet` | `data_utils.save_raw_ohlcv / raw_sidecar_path (data_utils.py:2…` | `data_utils.load_raw_ohlcv:206` | jetson | harvest (incremental) | `*raw_ohlcv.parquet` :116 |
| `funding_archive.parquet` | `funding_archive.py:29 (ARCHIVE_FILE)` | `harvest / features` | jetson | per harvest sync | `funding_archive.parquet` :89 |
| `oi_archive.parquet` — `oi_history.json` | `oi_archive.py:49-50` | `harvest / features` | jetson | per harvest sync (MAX_FILES_PER_SYNC=2000) | `oi_archive.parquet` :126 · `oi_history.json` :127 |
| `basis_archive.parquet` | `basis_archive.py:31 (ARCHIVE_FILE)` | `harvest / features` | jetson | per harvest sync | `basis_archive.parquet` :147 **added 2026-09-08** |
| `short_flow.parquet` | `short_flow.py:39 (ARCHIVE_FILE, FINRA RegSHO)` | `harvest / features` | jetson | per harvest sync (MAX_FILES_PER_SYNC=150) | `short_flow.parquet` :128 |
| `funding_history.json` | `funding.py:25-26 (_HISTORY_FILE)` | `funding.py cache read` | jetson | per funding poll (TTL 900s) | `funding_history.json` :83 |
| `daily_bars_cache.json` | `market_data.py:378-379 (_DAILY_CACHE_FILE)` | `market_data daily-bar path` | jetson | daily (_DAILY_CACHE_REFRESH_SEC=86400) | `daily_bars_cache.json` :118 |
| `har_rrv_history.json` | `volatility.py:269-270 (_HAR_RRV_FILE)` | `volatility HAR-RV` | jetson | per bar | `har_rrv_history.json` :119 |
| `crypto_rv_history.json` | `volatility.py:500-501 (_CRYPTO_RV_FILE)` | `volatility.py:304 (seeds har_rrv)` | jetson | per bar | `crypto_rv_history.json` :108 |
| `earnings_calendar.json` | `events_calendar.py:30-31 (_CACHE_FILE)` | `events_calendar lookup` | jetson | per calendar refresh | `earnings_calendar.json` :84 |
| `edgar_cache.json` — `edgar_tickers.json` | `edgar_events.py:43-44` | `edgar_events veto path` | jetson | per EDGAR poll | `edgar_cache.json` :95 · `edgar_tickers.json` :96 |

`basis_archive.parquet` is written by `basis_archive.sync` — a module with **no production
importer** (`grep -rl basis_archive --include='*.py'` → `basis_archive.py`, `tests/test_basis_archive.py`,
`tests/test_grp_deriv.py`). Its `Basis_*` features are not wired into any harvest, so the file is
produced only by running `python basis_archive.py` by hand.

## 3. Live trading state — per bar / per trade

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `crypto_predictions.json` — `stock_predictions.json` | `crypto_loop.py:32 / stock_loop.py:41 (write_prediction_cache)` | `gui.py:2038` | jetson | per bar (~hourly, written each cycle) | `*_predictions.json` :47 |
| `position_state.json` — `stock_position_state.json` | `base_loop.py:451 (_position_state_file / _save_position_state)` | `base_loop.py:451 (load on startup)`, `gui.py:7877-7878` | jetson | per trade / per cycle | `position_state.json` :72 · `*_position_state.json` :73 |
| `hard_stop_lockout.json` — `stock_hard_stop_lockout.json` | `base_loop.py:128-131` | `base_loop.py:_load_hard_stop_lockout` | jetson | per hard-stop fill | `hard_stop_lockout.json` :75 · `*_hard_stop_lockout.json` :76 |
| `account_risk_registry.json` — `account_risk_registry.json.lock` | `risk_budget.py:332,380 (write_registry, flock sidecar)` | `gui.py:84`, `risk_budget.read_registry:335` | jetson | per cycle (both books) | `account_risk_registry.json` :74 · `*.json.lock` :152 **added 2026-09-08** |
| `trade_memory.json` — `trade_memory.json.lock` | `trade_memory.py:24,43 (flock sidecar)` | `trading_utils.py:190` | jetson | per trade | `trade_memory.json` :142 **added 2026-09-08** · `*.json.lock` :152 **added 2026-09-08** |
| `shadow_v2_state.json` — `stock_shadow_v2_state.json` | `shadow.py:146 (_v2_state_file)` | `shadow._v2_look1_done` | jetson | one-shot per challenger | `*shadow_v2_state.json` :105 |
| `drift_state.json` | `monitor_drift.py:56 (_STATE_FILE)` | `gui.py:162` | jetson | per drift check (daily) | `drift_state.json` :92 |
| `telegram_offset.json` | `notify.py:136` | `notify.poll_telegram_commands` | jetson | per Telegram poll | `telegram_offset.json` :120 |
| `novelty_store.json` | `novelty.py:34 (_STORE_FILE, atomic _save)` | `novelty.py load` | jetson | per LLM/sentiment cycle | `novelty_store.json` :121 |
| `pipeline_status.json` | `run_pipeline.py:40 (write_status, 30s heartbeat thread); scri…` | `gui.py:1288,8898,9299,9487,10018`, `run_bots.py:60` | jetson | every ~30s while the pipeline runs | `pipeline_status.json` :45 |
| `pipeline_command.json` | `gui.py:201 (PIPELINE_COMMAND — GUI writes commands)` | `run_pipeline.py:46` | jetson | operator action | `pipeline_command.json` :144 **added 2026-09-08** |
| `command_result.json` | `run_pipeline.py:47` | `gui.py:10221` | jetson | per command ack | `command_result.json` :145 **added 2026-09-08** |
| `retrain_trigger.json` | `gui.py:8895,9512,10268` | `run_pipeline.py:45` | jetson | operator action | `retrain_trigger.json` :46 |

`pipeline_command.json` (GUI → pipeline) and `command_result.json` (pipeline → GUI) are the
operator IPC pair; `retrain_trigger.json` is the third leg. All three are plain files in the repo
root with no lock — last-writer-wins is accepted.

## 4. Locks and operator flags

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `.gpu.lock` — `.gpu_lock_info.json` | `gpu_lock.py:33-34,85,125 (fcntl advisory lock + metadata)` | `gpu_lock.py:66 (read_info)` | both | process startup (train/serve arbitration) | `.gpu.lock` :70 · `.gpu_lock_info.json` :143 **added 2026-09-08** |
| `trading_halt.flag` | `notify.py:132 (Telegram /halt); gui.py:3658,10074; `touch` ov…` | `base_loop.py:2886-2888` | jetson | operator action | `trading_halt.flag` :97 |
| `flatten_request.flag` — `flatten_crypto.flag`, `flatten_stock.flag` | `notify.py:134 (legacy shared) -> base_loop.py:80 fans out per…` | `base_loop.py:_flatten_flag_path:80` | jetson | operator action | `flatten_*.flag` :99 |
| `retrain_requested.flag` — `stock_retrain_requested.flag` | `monitor_drift.py:68 (retrain_flag_file)` | `run_pipeline.py:216-217` | jetson | drift breach (PSI>0.25 x2 days) | `retrain_requested.flag` :93 · `stock_retrain_requested.flag` :94 |
| `.clean_slate` | `hand-created by the operator / GUI reset` | `gui.py:1520`, `alpaca_compat.py:218 (comment)` | jetson | operator action (one-shot) | `.clean_slate` :40 |
| `crypto_heartbeat` — `stock_heartbeat` | `notify.py:106 (ping_heartbeat, 1/min rate-limited)` | `gui.py:79-82 (HEARTBEAT_FILES)` | jetson | per cycle (<=1/min) | `*_heartbeat` :85 |
| `~/.config/systemd/user/trader.service` (outside the repo) | `scripts/setup_jetson_system.sh --user` (temp file + `mv`; rolled back if `systemctl --user enable` fails) | the `kyle` user manager (`systemctl --user`) | jetson | operator action (one-shot install; `loginctl enable-linger` printed, never run) | n/a — not in the tree **added 2026-09-27** |

`trading_halt.flag` blocks **entries only** — exits keep running. `flatten_request.flag` is the
legacy shared name, fanned out per book to `flatten_{book}.flag` by `base_loop._flatten_flag_path`,
and goes stale after `FLATTEN_FLAG_STALE_SEC = 3600`.

## 5. Journals — append-only

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `journals/YYYY-MM-DD.jsonl` — `journals/YYYY-MM-DD.jsonl.gz` | `trade_journal.py:116 (log_decision); rotation at trade_journa…` | `decision_report.py:111`, `fees.py:123`, `execution_report.py:36`, `llm_eval.py:89`, `c… | jetson | per trade / per skip decision | `journals/` :64 |
| `journals/llm_replay/YYYY-MM-DD.jsonl` | `llm_analyst.py:45,1681 (_journal_replay; replay_capture_enabled, default ON)` — successful scored cycles only; since INTEL W15 (2026-09-27) each record also carries `prompt_sha256`, `latency_ms`, `dedup_hit`, `fence_stripped`, `parse_flags{sym:{raw_s,s_defaulted,s_nonfinite,s_clamped[,raw_p_up,p_up_nonfinite,p_up_clamped,raw_conviction,conviction_nonfinite,conviction_clamped]}}` appended after the unchanged legacy keys | `scripts/prompt_ab.py:51`, `scripts/llm_qualify.py:71,673` | jetson | per LLM cycle | `journals/` :64 |
| `journals/llm_calls/YYYY-MM-DD.jsonl` | `llm_analyst.py:1659 (_write_llm_call_row, via _journal_replay; same gates: persist + replay_capture_enabled)` — one `action:"llm_call"` row per non-dedup `analyze_trades` attempt, **failures included**: `outcome` ∈ ok/partial/parse_fail/not_object/empty/transport_discard/transport_error, `path`, `requested_model`, `model_used`, `n_symbols_sent`/`_returned`, `latency_ms`, `response_chars`, `max_tokens`, `temperature`, `prompt_sha256`, `advisor_v2`, `fence_stripped`, `n_s_defaulted`/`n_nonfinite`/`n_out_of_range`, `transport_errors` (exception class names only), `finish_reason`/`block_reason`/`usage_in`/`usage_out`/`http_status` (None until `llm_client` exposes `get_last_call_meta`), `cost_usd` | none yet (planned `scripts/llm_schema_reliability.py`, SCOUT_E spec) | jetson | per LLM attempt | `journals/` :64 |
| `journals/llm_qualify/shadow_scores.jsonl` | `scripts/llm_qualify.py:68,70` | `—` | jetson | one-shot report run | `journals/` :64 |
| `shadow_preds.jsonl` — `stock_shadow_preds.jsonl` | `shadow.py:124 (shadow_log_file)` | `shadow.evaluate_and_maybe_promote` | jetson | per shadow cycle (~hourly, SHADOW_LOG_INTERVA… | `*shadow_preds.jsonl` :124 |
| `pred_history.jsonl` — `stock_pred_history.jsonl`, `pred_history.jsonl.lock` | `monitor_drift.py:64,94 (history_file + flock sidecar)` | `monitor_drift PSI window` | both | per prediction cycle; HISTORY_KEEP_DAYS=7 | `pred_history.jsonl` :90 · `stock_pred_history.jsonl` :91 · `*.jsonl.lock` :100 |

**Rotation policy: OFF by default and deliberately so.** `trade_journal.rotate_old_journals` gzips
day-files older than `TRADER_JOURNAL_ROTATE_DAYS`, whose default is `'0'` = disabled
(`trade_journal.py:63`). The module comment (`trade_journal.py:55-61`) gives the reason: **eight
readers open the plain `.jsonl` path directly** — `fees.py`, `llm_eval.py`, `decision_report.py`,
`chart_core.py`, `gui.py`'s staleness glob, `llm_analyst.py`, `scripts/prompt_ab.py`,
`scripts/sizing_cofire_report.py` — and only `trade_journal.open_journal` has a `.gz` fallback.
Turning rotation on today would make those eight go blind past the horizon. Consequence:
**journals grow unbounded on the Jetson**. The two exceptions are `journals/llm_replay/` and
`journals/llm_calls/`, which `llm_analyst` prunes past `max_age_days` (45) on its own.

## 6. Reports and ledgers — one-shot CLI or GUI button (measurement-only)

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `shadow_status.json` — `stock_shadow_status.json` | `shadow.py:132 (shadow_status_file, written on every daily eva…` | `gui.py:157-158,188-189` | jetson | daily eval | `*shadow_status.json` :140 **added 2026-09-08** |
| `promotion_ledger.jsonl` — `stock_promotion_ledger.jsonl` | `shadow.py:140 (promotion_ledger_file)` | `gui.py:192` | jetson | per promotion decision | `*promotion_ledger.jsonl` :115 |
| `challenger_policy_gate.json` — `stock_challenger_policy_gate.json` | `backtest.py:1010 (_write_policy_gate_sidecar; only when targe…` | `shadow.py:894,1118 (_gate_preflight)`, `gui.py:179-180` | jetson | weekly retrain gate | `challenger_*` :122 · `stock_challenger_*` :123 |
| `backtest_report.json` — `backtest_stock_report.json`, `backtest_challenger_report.json`, `backtest_<slot>_<stress>report.json` | `backtest.py:941,982 + _patch_report_gate_block:982` | `gui.py backtest panel` | jetson | weekly gate run / manual --days | `backtest_*report.json` :79 |
| `backtest_fee_sweep.json` — `backtest_stock_fee_sweep.json` | `backtest.py:1125 (--fee-sweep)` | `—` | jetson | one-shot report | `backtest_*fee_sweep.json` :80 |
| `stage0_preds.json` — `stock_stage0_preds.json`, `challenger_stage0_preds.json` | `backtest.py:907 via stage0_preds.write_rows:160` | `scripts/ic_by_name.py --in`, `scripts/rank_gradient_report.py --preds` | jetson | weekly backtest | `*stage0_preds.json` :114 · `challenger_*` :122 |
| `decision_report.json` | `decision_report.py:767 (_write_stale_report), 981 (run_report)` | `gui.py:183,3946-3953`, `scripts/rank_gradient_report.py --buckets` | jetson | one-shot CLI / GUI button | `decision_report.json` :129 |
| `execution_report.json` | `execution_report.py:62 (_write_json)` | `gui.py:186,9103`, `chart_core.py:1308` | jetson | one-shot CLI / GUI button | `execution_report.json` :82 |
| `llm_eval_report.json` | `llm_eval.py:941,1036,1131 (+ .json.tmp at 804)` | `gui.py:184,9092` | jetson | one-shot CLI / GUI button | `llm_eval_report.json` :77 |
| `llm_advisor_report.json` | `llm_eval.py:1169,1224,1269,1446 (--advisor)` | `gui.py:185,9090`, `chart_core.py:1274` | jetson | one-shot CLI / GUI button | `llm_advisor_report.json` :78 |
| `beta_report.json` | `beta_ledger.py --json <path>; the GUI passes BETA_REPORT_FILE…` | `gui.py:171,187,9172` | jetson | GUI Beta Ledger button | `beta_report.json` :109 |
| `meta_curve_report.json` — `stock_meta_curve_report.json` | `scripts/meta_learning_curve.py:283` | `—` | jetson | one-shot report | `*meta_curve_report.json` :110 |
| `learned_lexicon.json` — `stock_learned_lexicon.json` | `scripts/train_lexicon.py:56 (--out)` | `—` | jetson | one-shot report | `*learned_lexicon.json` :111 |
| `lexicon_eval_report.json` | `scripts/train_lexicon.py:57 (--report)` | `—` | jetson | one-shot report | `*lexicon_eval_report.json` :112 |
| `llm_qualify_report.json` | `scripts/llm_qualify.py:69` | `—` | jetson | one-shot report | `llm_qualify_report.json` :113 |
| `llm_prompt_ab_report.json` — `llm_prompt_ab_scores.jsonl` | `scripts/prompt_ab.py:52-53` | `—` | jetson | one-shot report | `llm_prompt_ab_*` :149 **added 2026-09-08** |
| `options_overlay_verdict.json` | `options_overlay.py:367 (--out default)` | `—` | jetson | one-shot report | `options_overlay_verdict.json` :148 **added 2026-09-08** |
| `window_ab_summary_<book>.json` — `window_ab_<book>_<label>_stage0.json` | `scripts/window_ab.py:462, 344` | `—` | jetson | one-shot A/B report | `window_ab_*` :150 **added 2026-09-08** |
| `crypto_spread_census.json` | `scripts/crypto_spread_census.py:130,177 (--out default)` | `liquidity.py:252 (CRYPTO_CENSUS_FILE, env TRADER_CRYPTO_CENSUS_FILE)` | jetson | one-shot census | `crypto_spread_census.json` :117 |
| `research/funding_drift_2026-08.json` | `scripts/funding_drift_audit.py:207 (--out default)` | `—` | jetson | one-shot audit | n/a — lands in the committed `research/` tree (intentional) |

`crypto_spread_census.json` is the only runtime path whose location is env-overridable
(`TRADER_CRYPTO_CENSUS_FILE`, `liquidity.py:252`). `{slot}_policy_gate.json` can only ever exist
with a `challenger`/`stock_challenger` slot name — `backtest._write_policy_gate_sidecar` writes it
only when `targets_challenger` is set — so `.gitignore:122-123` covers every reachable variant.
`learned_lexicon.json` / `lexicon_eval_report.json` are **self-declared dark artifacts**: the
producing script's own docstring (`scripts/train_lexicon.py:32-35`) says nothing consumes them.

## 7. LLM and sentiment caches + the cost ledger

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `llm_analysis.json` | `llm_analyst.py:34 (_ANALYSIS_FILE)` | `gui.py:2051,5707-5709` | jetson | per LLM analyst cycle | `llm_analysis.json` :141 **added 2026-09-08** |
| `llm_cost.json` — `llm_cost.json.lock` | `llm_client.py:264 (_COST_FILE), 286 (.lock)` | `llm_client daily budget check` | both | per LLM call | `llm_cost.json` :71 · `*.json.lock` :152 **added 2026-09-08** |
| `llm_cost_history.jsonl` (2026-09-27, INTEL W9) | `llm_client.py:881 (_cost_history_path), 901 (_append_cost_history), 963 (call in _rollover_cost_locked)` — path derived from `_COST_FILE`, so a sandboxed `_COST_FILE` sandboxes it; one append-only JSON line `{date, cost, src, mem_date, mem_cost, reset_at, pid}` per observed daily rollover, under `_cost_file_lock`, fail-soft | none yet (future LLM-spend ledger reader; dedupe by `date` is the reader's job) | both | once per daily cost rollover (first ledger touch after midnight PT) | **NO** — only the exact name `llm_cost.json` is ignored (:71); needs its own line |
| `logs/llm_eprocess_report.json` (2026-09-27, INTEL W10) | `llm_eprocess.py:89 (DEFAULT_LIVE_OUT); written only by live mode, which exits 3 until `research/campaign_2026-09_jetson/llm_eprocess_params.json` carries signed_by/signed_at/registration_sha` | none (measurement-only; nothing in the repo reads its verdict) | Jetson | per operator run | yes — under `logs/` | Anytime-valid LLM-spend ledger (Scout C Design A); `--selftest` writes nothing |
| `llm_config.json` | `llm_config.py:190 (LLM_CONFIG_FILE)` | `decision_report.py:961`, `trade_journal.py:49`, `llm_client`, `llm_analyst` | jetson | on settings change | `llm_config.json` :50 |
| `sentiment_cache.db` — `sentiment_cache.db-wal`, `sentiment_cache.db-shm` (2026-09-26: gains table `fng_daily_legacy_localtz` + `state.fng_date_basis` on the first crypto F&G fetch, via `_migrate_fng_date_basis`) | `sentiment_history.py:32 (_DB_PATH)` | `scripts/train_lexicon.py:36`, `learned_lexicon.py:147 (read-only URI)` | jetson | per sentiment fetch | `sentiment_cache.db` :56 · `sentiment_cache.db-wal` :58 · `sentiment_cache.db-shm` :57 |

`llm_config.json` **contains API keys** and is gitignored at `.gitignore:50`; `llm_cost.json` is
the daily spend ledger (not configuration) guarded by `llm_cost.json.lock`.

## 8. GUI settings and committed config

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `gui_settings.json` | `gui.py:310,324 (_save_gui_settings)` | `gui.py:313,2497` | jetson (GUI) | on settings change | `gui_settings.json` :43 |
| `news_cache.json` | `gui.py:308,379 (_save_news_cache)` | `gui.py:360,1399,1724` | jetson (GUI) | per news refresh | `news_cache.json` :44 |
| `account_baseline.json` | `gui.py:8195,8202` | `gui.py:7791-7797` | jetson (GUI) | boot-time equity fetch | `account_baseline.json` :146 **added 2026-09-08** |
| `indicator_config.json` | `indicator_config.py:18 (_FILE)` | `indicators.py`, `harvest scripts` | jetson | on preset change | `indicator_config.json` :53 |
| `stock_universe.json` | `COMMITTED (hand-edited)` | `stock_config.py:13,52` | both | n/a — versioned config | n/a — **committed on purpose** (hand-edited config) |
| `.env` | `operator-created` | `dotenv consumers` | jetson | n/a — secrets | `.env` :2 |

---

## 9. Files that tests write into the repo root on the dev Mac

Running `python3 -m pytest tests/` on this Mac used to leave five paths behind. All were
gitignored, so none could pollute a commit; the problem was that they are the **live runtime
paths**, written into the repo root instead of a `tmp_path`, and one accumulated across runs.
**Four of the five were fixed on 2026-09-08**; the fifth, `logs/trader.log`, on 2026-09-27 (INTEL W18) — a suite run no longer writes any of them.

| Residue | Responsible test(s) | Why it landed in the repo root | Fix status |
|---|---|---|---|
| `.gpu.lock` (0 B) | `tests/test_gpu_lock.py` | `gpu_lock._LOCK_FILE` is anchored to the module directory; the test exercises the real acquire path | **FIXED** — autouse `_lock_sandbox` fixture monkeypatches `_LOCK_FILE`/`_INFO_FILE` into `tmp_path` (same pattern as `tests/test_review_b19.py::lock_sandbox`) |
| `adaptive_state_testq1.json` (~2.4 KB) | `tests/test_c26_Q1.py` (`AC.record_trials('testq1', …)`) | `adaptive_config.save_adaptive_state` writes `adaptive_state_{asset_type}.json` into `adaptive_config.BASE_DIR` | **FIXED** — the `mock.patch('adaptive_config.BASE_DIR', tmp_path)` context was widened to cover the `record_trials` calls too. `.gitignore`'s comment still anticipates it ("test residue uses adaptive_state_test*") |
| `llm_cost.json` (+`.lock`) | `tests/test_llm_providers.py`, `tests/test_llm_claude.py`, `tests/test_c26_S2.py`, `tests/test_c26_P1.py` | `llm_client._COST_FILE` / `_record_cost` write the real ledger | **FIXED** — every one of those modules now monkeypatches `llm_client._COST_FILE` into `tmp_path` |
| `pred_history.jsonl` (+`.lock`) | `tests/test_review_b19.py`, `tests/test_c26_P1.py`, `tests/test_c26_P2.py`, `tests/test_monitor_drift.py` | `monitor_drift.history_file('')` without redirecting `BASE_DIR`; **accumulated across runs** (~3.6 KB of `{"preds":{"AAA":0.5}}` rows since 2026-08-19) | **FIXED** — `monkeypatch.setattr(monitor_drift, 'BASE_DIR', tmp_path)` at the writing tests (`BASE_DIR` is read at call time, so the redirect takes) |
| `logs/trader.log` | *any* test that imports a module importing `log_config` | `log_config._setup` runs at first `get_logger` (22 modules call it at import) and opened `<repo>/logs/trader.log` — the production log the live bots write | **FIXED 2026-09-27 (INTEL W18)** — `tests/conftest.py` sets `TRADER_LOG_DIR` to a per-session `mkdtemp(prefix='trader-test-logs-')` at module scope, before any repo import (a non-empty caller value wins); `log_config._log_paths()` (`log_config.py:102-135`) reads it at the first `_setup()`. Logging is still configured on first `get_logger` — production behaviour is unchanged (the variable is unset there → `<repo>/logs/trader.log`). The hygiene section prints `test logs: <dir>` and `production log untouched: yes / no / unattributable` (the last = changed while live bots held it open); pinned by `tests/test_intel_logdir_2026_09.py` |

## 10. Logs — what rotates and what does not

| Path (canonical — prefix variants) | Writer | Readers | Machine | When written | Gitignored? |
|---|---|---|---|---|---|
| `logs/trader.log` — `logs/trader.log.1 .. .5` | `log_config.py:22-25,162-168 (SharedRotatingFileHandler 10MB x5; dir from _log_paths() :102-135)` | `operator / gui log tail` | both | every logger call | `logs/` :31 |
| `pipeline_output.log` — `crypto_bot_output.log`, `stock_bot_output.log`, `backfill_output.log`, `sentiment_fetch.log` +3 | `run_pipeline.py:41-43,1478,1516; gui.py:5549,5658; shadow.py:…` | `gui.py:111-114 (LOG_FILES)` | jetson | append while the process runs | `*.log` :29 · `sentiment_fetch.log` :61 · `meta_retrain.log` :125 |

| | Path(s) | Handler | Rotation |
|---|---|---|---|
| **Rotated** | `logs/trader.log` (+ `.1`…`.5`) | `log_config.SharedRotatingFileHandler` (a multi-process-safe `RotatingFileHandler` subclass, `log_config.py:46-99`; built at :162-168 from `_log_paths()` :102-135) | `_MAX_BYTES = 10 MB` × `_BACKUP_COUNT = 5` → **~60 MB worst case** |
| **Unrotated** | `pipeline_output.log`, `crypto_bot_output.log`, `stock_bot_output.log`, `backfill_output.log`, `sentiment_fetch.log`, `meta_retrain.log`, `llm_refresh.log`, `llm_refresh_one.log` | raw `open(..., 'a')` in `run_pipeline.py`, `gui.py`, `shadow.py` | **none — unbounded** |

**Override (2026-09-27):** `TRADER_LOG_DIR` (unset in production) moves `trader.log`, its `.1`-`.5` backups and `trader.log.lock` to another directory (absolute or repo-root-relative; unusable -> default + one stderr line) — `log_config._log_paths()` (`log_config.py:102-135`), read at the first `get_logger` call; `tests/conftest.py` sets it at module scope to a per-session `trader-test-logs-*` temp dir so pytest no longer writes into the production `logs/trader.log` (see `docs/FLAGS.md` §5.1 and §9 above).

`log_config` pins `urllib3` / `httpx` / `httpcore` / `websockets` / `yfinance` / `numba` /
`charset_normalizer` to WARNING so numba's per-deploy byteflow dumps cannot eat the rotation budget
(console INFO, file DEBUG).

**Jetson implication (8 GB device, priority #1 in CLAUDE.md):** the bounded budget is
`logs/` ≈ 60 MB. Everything else grows without a ceiling — the eight raw-append `*.log` files and
the unrotated `journals/*.jsonl` (§5). On a long-running Jetson those two families, not the model
artifacts, are what fills the disk. Neither has a monitor.

## 11. Archive note — what moved into `archive/` (dev Mac 2026-09-08, Jetson 2026-09-26)

The 2026-09-08 cleanup moved stale local artifacts into a new root `archive/` tree rather than
deleting them (owner rule: delete nothing). Nothing in `archive/` is read by any code path — every
consumer of these files is existence-guarded and simply regenerates.

| Moved to | File | Why it was stale |
|---|---|---|
| `archive/local_residue/` | `backtest_report.json` | dev-Mac output from 2026-07-11, `n_trades: 0`, `prefix: ""`. `backtest._patch_report_gate_block` returns early when the path is missing; the GUI panel is existence-guarded. The Jetson has its own. |
| `archive/local_residue/` | `oi_history.json` | dev-Mac output from 2026-06-10, one `BTC/USD` row. `oi_archive` treats a missing file as "no live history" and refetches (TTL 900 s). |
| `archive/local_residue/` | `adaptive_state_testq1.json`, `.gpu.lock`, `llm_cost.json`, `pred_history.jsonl` (+`.lock`) | dev-Mac test residue (§9) — moved after the four test-side fixes landed; verified by a full suite run with a timestamp marker (no root file regenerated) |
| `archive/commit_messages/` | `commit_msg.txt` (untracked) | the already-used commit message of `ca00b16`; zero references repo-wide |
| `archive/commit_messages/` | `commit_msg_round2.txt` (**tracked**, `git mv`) | the commit message of `c7f846e`, accidentally committed by that same commit; zero references |
| `archive/claude_workflow_runs/` | `.claude/workflows/modules-v3.run.json` (**tracked**, `git mv`) | run-config pinned to batch B5, whose `report_path` `research/module_improve_v3_batchB5.md` does not exist (A, B1–B4 do) |

Left in place on purpose: `logs/trader.log` and its `.1`/`.2` rotation slots (the handler renames
them in place, and importing any logged module recreates the file — §9); the empty `journals/`
directory — `trade_journal` recreates it anyway. On the Jetson none of this applies: those paths
are live runtime state there and must stay where the code expects them.

See `archive/README.md` for the moves as executed.

**Jetson, 2026-09-26.** A second pass moved the untracked March-2026 Jetson residue out of the
working tree (plain `mv`; none of it was ever in git, and nothing imports or names it): the PDF
manual and its generator (`scripts/generate_manual.py`, `scripts/manual_expanded.py` — the latter
was the one unreachable module failing `scripts/repo_graph.py --check`), `scripts/mem_diagnostic.py`,
`global_context.py`/`.json`, the `research_*.txt` dumps and the `review_findings.md` /
`research_gap_analysis.md` notes → `archive/jetson_residue_2026-03/` (untracked, deliberately not
ignored — the owner decides whether to commit them); the `stock_*_v2.*.backup_score10` model
backups and `*_study.db.bak*` Optuna backups → `archive/local_residue/`, ignored by two
`archive/local_residue/`-anchored `.gitignore` patterns. The same day the untracked
`indicators_c` C extension (`c_ext/indicators_c.c`, `c_ext/build.py` and the root `.so`) moved to
`archive/c_ext/` (untracked, not ignored); `indicators.py` loads it only with `TRADER_INDICATORS_C=1`.
Rows and restore notes: `archive/README.md`.
