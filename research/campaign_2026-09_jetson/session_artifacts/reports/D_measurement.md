# D — Measurement-only instruments vs the real journals (Jetson, 2026-09-26)

**Scope.** I ran every measurement-only CLI against the on-box journals. The day-files run
2026-02-23 → 2026-05-07; 51 files hold 11,994 rows. The auxiliary inputs were trade_memory.json,
the *_predictions.json files, llm_cost.json and the training parquet.

**How each run was set up.**
- Every run used the jetson env with `CUDA_VISIBLE_DEVICES=''` and `timeout 600`, one subprocess per run.
- Each run went through a wrapper that records peak child RSS and wall time.
- Each run also carried a **Python audit hook**, which records every write-mode `open`,
  `os.replace/rename/remove/mkdir` and `sqlite3.connect`. The "files it wrote" column therefore
  comes from what was observed at run time, not only from reading the source.
- Raw artefacts are in `scratchpad/D_runs/` (first pass) and `scratchpad/D_runs/r2/` (the audited pass).
- Most runs used `--days 200` (back to 2026-03-10). The newest journal is 142 days old, so any
  `--days` ≤ 140 finds **no rows at all**. For example, `decision_report.py --days 30` prints
  "No journal entries found." and writes a `stale:true` report.

Peak RSS stayed under 700 MB for every instrument (indicator_leadlag 441 MB, wave6 367–422 MB,
decision_report 218–253 MB). Wall time was 0.3–22 s. None came close to the Jetson budget.

## 1. Results table

| instrument | command | result class | key output lines | files it wrote (audit-hook observed) | defect (file:line) |
|---|---|---|---|---|---|
| decision_report | `decision_report.py --days 200` | **runs; verdict void** | `WARNING: 92% of rows unpriced (0 fetch failures, 23 out-of-window)`; `WARNING: journaled skip_reasons unknown to GATE_REASONS (producer drift?): {'llm_below_buy_min (0.58<0.60)': 1831, …}` (11,815 rows); `CONVICTION CALIBRATION (n=2 priced entries … out-of-window=22)`; `quality.representative=false`, `admitted_k={}`, `signal_exit.n_signal_sells=0` | `decision_report.json` (repo root, atomic via `decision_report.json.<pid>.tmp`), `logs/trader.log` append | (a) **153 of 177 buy rows dropped silently.** `decision_report.py:592-593` filters `pred_return is not None` without a counter, and every legacy crypto buy has `pred_return: null`. The "92% unpriced" banner is computed over 25 rows, not 177. (b) The stock replay frame is ~30 days, not the ~45 the NOTE claims: `fetch_stock_bars_alpaca` fetches 45 days, then `.tail(320)` (`market_data.py:294,320`); the frame observed starts 2026-08-27. The NOTE fires only when `days > 42` (`decision_report.py:846`). Out-of-window rows are counted, not silent. |
| decision_report (runbook form) | `decision_report.py --days 30` | runs; empty (no journals ≤140 days old) | `No journal entries found.` → stale report | `decision_report.json` (stale:true) | — |
| beta_ledger | `beta_ledger.py --days 200 --json <scratch>` and `--days 90` | **runs; verdict degenerate** | `strategy: +127789.0%/yr at 106275.0% vol`; `alpha +122396%/yr (t +1.01)`; `beta[SPY] -140.9`; `WARNING: 2 day(s) with |daily return| > 15% (2026-08-13, 2026-08-14)`. The 90-day window gives `+247055.8%/yr`. | only the `--json` path (scratch); `~/.cache/py-yfinance/*.db` (yfinance cache); `logs/trader.log` | **Cause: a vendor bad print.** Alpaca portfolio history reports equity **93.63 on 2026-08-13**, which equals today's cash of $93.63, so position value was dropped, then 82,777 the next day. `profit_loss` mirrors the glitch (−83,180 / +82,683), so the "CLEAN (transfer-free)" leg does **not** remove it. beta_ledger only warns (`beta_ledger.py:372-377`, `OUTLIER_DAILY_RETURN` :47) and still regresses on it. **Instrument gap:** there is no exclusion or winsorize option for flagged days. With 08-13 dropped (in-process rerun) the output is sane: `+37.8%/yr, 39.1% vol, alpha t +1.08, beta[BTC] summed +0.623 (t +3.51), R² 0.30`. |
| execution_report | `execution_report.py --days 200` | **runs; one figure fabricated** | `Crypto maker NOTIONAL share: 0.0% of $260,472 entered`; no shortfall section | `execution_report.json` (root, **non-atomic** `open('w')`), `logs/trader.log` | (a) **`execution_report.py:172-181` treats a buy with no `entry_tactic` as taker.** The count-based block at :132-136 correctly requires `entry_tactic`. So 0.0% is fabricated from legacy rows that carry no tactic. (b) The "No fills with slippage data yet" message fires only when `rows` is empty (:79), so 177 buy rows without `slippage_bps` just omit the shortfall section silently. (c) The report write at `:61-64` is not atomic, although gui.py and chart_core.py:1308 read the file. |
| gap_audit | `gap_audit.py --symbols SPY QQQ AAPL --json <scratch>` | runs to a verdict | `SPY overnight_mean=5.58bps … forfeited_drift=$703/yr gap_through=$104/yr`; `SLEEVE TOTAL: forfeited_drift=$1,454/yr gap_through=$931/yr` | only `--json` (scratch); yfinance cache | — (pure yfinance; journals not used) |
| gap_audit (GUI form) | `gap_audit.py --symbols <load_stock_universe(): 56 names>` | runs; **total misleading** | 10 × `XRP/USD: no data` (crypto names; yfinance `TypeError('Response' object is not subscriptable)` caught); `SLEEVE TOTAL: forfeited_drift=$110,591/yr gap_through=$125,901/yr` | same | `gui.py:9154-9169` passes the whole universe, crypto included, to gap_audit. `gap_audit.py:170-181` sums every name as the "SLEEVE TOTAL", but the sleeve holds at most `OVERNIGHT_SLEEVE_MAX_POSITIONS=2` (`strategy_config.py:122`). The GUI's go/no-go total is ~23× too large. |
| llm_eval | `llm_eval.py --days 200` / `--asset crypto` / `--asset stock` | **runs; void (no input schema)** | `No llm_analysis journal entries found …` | `llm_eval_report.json` no_data stub (atomic, but a **fixed** tmp name `llm_eval_report.json.tmp`, `llm_eval.py:804`). Each `--asset` run overwrites the same file. | May journals contain **zero** `llm_analysis` rows. **Reconstruction probe:** 446 legacy skip/buy rows → 328 realized with real Alpaca bars through `realize_scored_rows` + `compute_incremental_report`. It ran cleanly to `insufficient_power (n_clusters=91 < 120 …)`, `partial_spearman -0.07`. The realization and statistics path works on real bars. Doc drift: CLAUDE.md:127, MAP.md:608/1018, GLOSSARY.md:60, MODULES.md:778 and runbook :44 say "n≥60". The code also needs **≥120 distinct t0 clusters and effective_n ≥ 20** (`llm_eval.py:69-73,575-585`), which is about ≥20 days of hourly LLM cycles at fb=24. |
| llm_eval --advisor | `llm_eval.py --days 200 --advisor` | runs; void | `No llm_advisor_v2 journal entries found` | `llm_advisor_report.json` stub (fixed tmp) | void by schema (advisor_v2 never ran) |
| journal_stats (no CLI; API) | `load_trades` / `compute_stats` / `format_summary` / `build_eod_digest` via probe | runs; void | `load stats {'files_read': 39, 'rows_seen': 169, 'corrupt_lines': 0, 'trades': 0}` → `No closed trades in range.` EOD digest for 2026-05-07: `1213 skips, 0 LLM calls` | none (verified: stdlib, read-only) | void by schema: legacy journals contain **no `sell` rows at all**; exits lived only in trade_memory.json (120 sells + 3 covers) |
| sizing_cofire_report | `scripts/sizing_cofire_report.py --days 200` and `--json` | runs; void | `0 buy rows (51 files, 0 malformed lines skipped) — no rows` | none | (a) `scripts/sizing_cofire_report.py:97` drops buys that lack a `sizing` dict, and the header reports "0 buy rows" when 177 buys exist. It should say "177 buys, 0 carry sizing". (b) `--json` is `store_true` (stdout), **not a path**. The brief's `--json <path>` form gets an argparse error that is swallowed, and the script **exits 0** (`:356-363`, by design "ALWAYS exits 0"), so wrappers cannot detect misuse. |
| monitor_drift | CLI **not run**: `main()`→`run_check` always may write `drift_state.json` (`:251-262`) / `retrain_requested.flag` (`:459`) / prune `pred_history`. Pure functions were called via probe instead. | runs (pure); void | manifest hash / ref deciles / holdout hit_rate = **None** for both books; no `pred_history.jsonl`; `check_drift -> None`. trade_memory outcomes: crypto 73 exits, 65.8% hit; **stock 50 exits, 22.0% hit**. A hypothetical in-memory CUSUM at baseline 0.55 would alarm on stock (2 alarms). PSI kernel sanity: same-dist 0.032 / shifted 1.00. | none from the probe (`logs/trader.log` append on import) | no read-only CLI mode exists. With no manifest on disk the CLI would actually be a no-op today, but that is not guaranteed. Minor: `_live_outcomes` counts legacy short `cover` rows into the long hit rate (`monitor_drift.py:317`, via the `exit` fallback). |
| retrain_ledger | no CLI / no read mode. Probe: `incumbent_paths`, `load_adaptive_state`, `_load_stack(champion)`, `_score_stack`, `paired_scores` on trailing purged rows of training_data.parquet | runs (mechanics); void | `ledger rows on disk: 0` (both books); no `.prev`, no challenger. Champion loads on CPU (fb 64, seq_len 18, booster None: no lgb_model.txt). 624 trailing purged rows (2026-02-17..21); `paired_scores(champion,champion)` IC 0.627 (in-sample, same model twice, which only proves the plumbing) | none (no `save_adaptive_state` called) | — (probe peak 1.07 GB, from my full-panel float64 X, not the module) |
| indicator_leadlag | `indicator_leadlag.py --data training_data.parquet --horizons 1,4 --json <scratch>` | **runs to a verdict** | `{'momentum-carrier': 10, 'leading': 6, 'inert': 7}`; `EXACT DUPLICATES: Return_12h == ROC`; 3 redundancy clusters; top "leading" = `Daily_Sentiment +0.060 @4h` | only `--json` (scratch) | — (see §2 observation on Daily_Sentiment) |
| wave6_stage0 | `scripts/wave6_stage0.py --book crypto --json <scratch>` | runs; void by design (exit 1) | `[crypto] no TB_Bars_* columns harvested` → `NOTHING MEASURED — do not act on this run` | `--json` (scratch, `[]`) | — (old-schema parquet; the orchestrator already knew it has no TB_* columns) |
| rank_gradient_report | `--buckets decision_report.json` (a decision_report.json existed only because I had just produced it) | runs; correct refusals | stale 30d report → `REFUSED: … STALE decision_report` (exit 2); 200d report → `insufficient rank coverage` (exit 1) | none | `scripts/rank_gradient_report.py:103` checks only `stale`. It ignores `quality.representative=false` (92% unpriced), so a non-representative report carrying buckets would be gated on (MIN_BUCKET_N suppression is the only guard). |
| ic_by_name, rank_gradient `--preds` | — | **not runnable** | no `*stage0_preds.json` anywhere on disk (and no backtest report) | — | expected: the Stage-0 dump appears only after a weekly backtest |
| reliability_report | — | **not runnable** | needs `{p_legacy, p_purged, y}` (`calib_holdout.json`) | — | **Gap:** no code in the repo produces this file. Only calibration.py/tests reference `p_purged`, and runbook Phase 2 §4a depends on it. No meta artefacts exist either. |

## 2. Journal schema drift (May-era rows vs today's producers/readers)

The census covers 2026-04/05 files: 11,886 rows, of which 11,815 are `skip`, 45 `buy` and 26 `short`.
The Feb/Mar files add 219 `buy`, 5 `short` and 3 `skip`. **Only three `action` values exist**, and
there is no `kind` or `event` key.

| action | keys present in May rows | fields current readers expect that the May rows lack |
|---|---|---|
| `skip` | symbol, action, skip_reason, pred_return, llm_multiplier, llm_reasoning, ts | **skip_reason vocabulary.** 100% of April/May skips are the free-text `llm_below_buy_min (x<0.60)`, which is in neither `GATE_REASONS` nor `UNPRICED_GATES`. They are unpriceable and land in `_unclassified_skip_reasons`. Missing entirely: spread_pct, meta_p/meta_prob, entry_rank, pred_thresh_ratio, conviction_tier, lstm_pred/lgb_pred, q10/q10_floor, _fetch_failed. |
| `buy` | symbol, action, pred_return (**null on every crypto buy**, 204/210), sentiment_gate, sentiment_reasons, llm_multiplier, llm_score, llm_reasoning, final_notional, skip_reason, ts | decision_price, fill_price, **slippage_bps**, quote_age_s, **entry_tactic**, maker, maker_notional, **sizing{}** (sizing_cofire), meta_p, entry_rank, conviction fields |
| `short` | symbol, action, pred_return, llm_score, qty, price, ts | no longer produced (long-only). No reader consumes it: dropped by `_KEPT_ACTIONS` and ignored by journal_stats. |
| *(absent)* | — | **`sell`** (journal_stats, decision_report signal_exit, execution_report exit slippage); **`entry_window`** (admitted_k); **`llm_analysis`** (llm_eval, execution_report LLM economics); `llm_advisor_v2`; `account_risk`; `cycle_latency`; `signal_exit_reading`; `vertical_barrier`; `entry_fills`; `ioc_entry_fallback`; `llm_backoff`; `rr5_demotion`; `circuit_breaker_trip`; `pred_fanout_timeout` |
| `ts` on all rows | naive local time (`2026-05-07T08:31:58` = 13:31 UTC, the stock open) | offset-aware ISO (`trade_journal.py:106,115`). Legacy naive ts are read as **UTC** by decision_report (`_replay_grouped`, `decision_report.py:344-348`) and sizing_cofire (`:56`), but as **local** by llm_eval (`fromisoformat().timestamp()`). That is a 5 h disagreement on every legacy row, acknowledged at `trade_journal.py:103-104`. |

This is the substance behind the runbook's "pre-campaign decision_report figures are void"
(`03_jetson_runbook.md:41`). The current journals cannot feed any post-campaign instrument except
the legacy `pred_return`/`llm_multiplier` pairs, which is why the llm_eval reconstruction probe worked.

**Observation outside the brief, needs a PIT check.** `Daily_Sentiment` has IC(fwd 4h) of
+0.041 to +0.083 across UTC hour buckets, with a similar IC against the *past* 4h return. The
fwd IC is weakest late in the UTC day (+0.041 for 18–24 h vs +0.065 for 0–6 h). That fits a
same-day sentiment aggregate being joined to all hours of its day, i.e. same-day lookahead, but it
is not conclusive. Worth a harvest-side PIT audit before this feature is trusted.

**Operational fact seen along the way (read-only calls).** The paper account is **not flat**. It
holds 6 crypto positions worth about $122k (BTC, DOGE, ETH, LINK, SOL, XRP), cash $93.63, and has
been unmanaged since 2026-05-07. That is why equity keeps moving, from 82k in August to 123k now.

## 3. Genuine code defects vs stale-data artefacts

**Genuine code defects.** All are low-to-medium severity and none is on the live path.
1. `decision_report.py:592-593`: buys with `pred_return=None` are dropped with no counter, so the
   unpriced banner and `quality` block understate the loss (153/177 here). Current producers write
   `pred_return` on buys (`base_loop.py:3297`), so this bites only on null predictions going forward.
2. `execution_report.py:172-181`: the notional maker share counts tactic-less buys as taker. It
   reports a fabricated 0.0%, inconsistent with the count block at :132-136.
3. `execution_report.py:61-64`: a non-atomic report write, while gui.py and chart_core.py read the
   file. `llm_eval.py:804` uses a fixed `.json.tmp` name, the same GUI-vs-CLI race that decision_report
   already fixed with pid-unique tmp files (`decision_report.py:746-756`).
4. `gui.py:9154-9169` + `gap_audit.py:170-181`: the GUI Gap Audit feeds crypto names and the whole
   56-name universe, and the "SLEEVE TOTAL" sums them all against a 2-slot sleeve. The go/no-go
   number is inflated ~23×.
5. `beta_ledger.py:372-377`: outlier days are warned about but still regressed. `clean_returns_from_pl`
   cannot fix vendor bad prints because Alpaca's `profit_loss` carries them too. An exclusion flag
   is needed (for example `--exclude-dates` or auto-drop of |r| > 15% day pairs that reverse).
6. `scripts/sizing_cofire_report.py:97` (no "buys without sizing" count) and `:356-363`
   (argparse errors exit 0).
7. `scripts/rank_gradient_report.py:103` ignores `quality.representative`.
8. `decision_report.py:846`: the NOTE threshold (days > 42) and its "~45 days" text do not match
   the effective 320-bar (~30-day) stock frame (`market_data.py:320`).
9. **Missing producer:** nothing writes reliability_report's `{p_legacy, p_purged, y}` input.
10. **Doc drift:** llm_eval's keep/kill gate is n ≥ 60 **and** ≥ 120 t0-clusters **and**
    effective_n ≥ 20, not "n ≥ 60". Affects CLAUDE.md:127, docs/MAP.md:608,1018,
    docs/GLOSSARY.md:60, docs/MODULES.md:778 and runbook:44.

**Stale-data artefacts, not code bugs.**
- No `sell`, `entry_window`, `llm_analysis`, `llm_advisor_v2` or `sizing` rows. This voids
  journal_stats, sizing_cofire, llm_eval, the signal_exit/admitted_k sections, and the execution
  shortfall section.
- 100% of April/May skips carry the retired free-text reason `llm_below_buy_min (…)`.
- Naive-local `ts` on legacy rows.
- No model manifest, pred_history, `.prev`/challenger, or ledger rows, so monitor_drift and
  retrain_ledger are void.
- Old-schema parquet with no TB_* columns, so wave6 is void.
- No Stage-0 dump.
- The Alpaca 2026-08-13 equity bad print, which beta_ledger surfaces but does not handle (defect 5).

## 4. Readiness for the post-retrain evidence reads (`03_jetson_runbook.md` Phase 0 §5 / Phase 1)

| read | ready? | condition |
|---|---|---|
| `beta_ledger.py --days 90 --json …` (Phase 0) | **Mechanically ready; the output is void today.** | Any window containing 2026-08-13/14 (every `--days 90` run until about 2026-11-11) is dominated by the bad print. Until then run it on an `--equity-csv` with 08-13 removed, or add an exclusion option (defect 5). |
| `decision_report.py --days 30` (Phase 0) | **Ready.** | Needs ≥ ~20–30 days of *new* journals. Stock rows older than ~30 days are structurally out-of-window (320-bar frame). Watch that `_unclassified_skip_reasons` is empty. |
| `llm_eval.py --days 30` (Phase 0) | **Ready.** The realization and statistics path was proven on reconstructed real rows. | A keep/kill verdict needs ≥ 120 hourly LLM-cycle clusters and ≥ ~20 days of span at fb=24, not "n ≥ 60". `--asset` runs overwrite one report file. |
| `scripts/sizing_cofire_report.py` (Phase 1 B7) | **Ready.** | It reads only buys carrying `sizing{}`. Use `--json` as a flag (stdout), not a path. |
| `ic_by_name.py` / `rank_gradient_report.py --preds` (Phase 1) | **Blocked on input.** | They need `{slot}_stage0_preds.json` from the first weekly backtest. `rank_gradient --buckets` is ready but should also check `quality.representative`. |
| `reliability_report.py` (Phase 2 §4a) | **Not ready.** | No producer for its input and no meta artefacts. |
| `execution_report.py` | Ready for new rows. | Fix the notional-share default (defect 2) before trusting maker share on mixed-era windows. |
| `monitor_drift` / `retrain_ledger` | Ready but inert. | Both come alive only after a retrain writes `*model_v2.manifest.json` (deciles, hit_rate) and hypersearch calls `record_retrain_gain`. |
| `indicator_leadlag.py`, `gap_audit.py` (CLI with an explicit short list) | **Ready now.** | Use gap_audit with the actual sleeve names, not the GUI universe button. |
| `wave6_stage0.py` | Blocked. | Needs a harvest with TB_* labels. |

## Housekeeping

- This run created four gitignored root files, none of which existed before:
  `decision_report.json`, `execution_report.json`, `llm_eval_report.json`, `llm_advisor_report.json`.
  They were **moved** (not deleted) to `scratchpad/D_runs/root_outputs/`.
- The repo root now matches the pre-run snapshot, apart from `logs/trader.log` appends (a known
  import side effect) and the items below.
- I did **not** create `llm_cost.json.lock`, the rewrite of `llm_cost.json` (now `{"date":
  "2026-09-26","cost":2.1e-05}`), or the `sentiment_cache.db` mtime changes. The audit logs show no
  instrument touching them, so a concurrent agent did; they are left in place.
- No production file was edited.
