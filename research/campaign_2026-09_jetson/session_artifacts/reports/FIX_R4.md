# FIX_R4 — TB labels stamped AFTER every row filter (L7 full fix)

Owner of edits: scripts/harvest_stock_data.py, scripts/harvest_crypto_data.py, NEW tests/test_tb_restamp_2026_09.py.
Running harvest (pid 36639) and stores untouched. Backups + diffs: <scratchpad>/R4/{harvest_*_data.py.orig, stock.diff, crypto.diff}.

## 1. Root cause (measured, not inferred)

**Stock — `RS_vs_SPY` NaN, not the top-60 cut.** Probe (<scratchpad>/R4/probe.py, read-only Alpaca fetch, replays
prepare_stock_data up to the dropna): AAPL 22,756 raw bars → 22,577 after dropna; 99 prefix rows (SMA_100/Hurst warmup),
**32 interior rows, every one NaN in exactly one column: `RS_vs_SPY`**; 0 of them missing from SPY's index;
tradability removed 0. Cause: `indicators.compute_relative_strength` (indicators.py:640-643) does
`stock_roc / bench_roc.replace(0, np.nan)` — whenever SPY's 12-bar ROC is *exactly* 0 (flat 12-bar window) the feature is NaN,
and `df.dropna()` then deletes a real price bar from the interior of every name (~30-35/name: AAPL 32, ABBV 30, ABNB 35, AFRM 33 in the live log).
Secondary sources (same effect): `_asof_tradability_mask`'s bar-granular `Close >= $3` floor (QBTS: 15 interior discontinuities)
and the day-granular cross-sectional `_asof_membership_mask` (names near the top-60 cut).

**Crypto has the same defect, silently** (it has no guard, so "no TB-GUARD line" was vacuous). Comparing the 22:16 crypto store with
raw_ohlcv.parquet: BTC/ETH/XRP 0 interior removals; **DOGE 37, LINK 166, SOL 252 interior bars removed** (3/22/11 gaps), all NaN only in
`Volume_Ratio` = `Volume / Volume_SMA_20` (indicators.py:522) → 0/0 across zero-volume stretches (all removed bars have Volume==0).
In that store LINK has 215-425 rows per horizon whose TB_Bars offset lands on the wrong stored row.

## 2. Decision

Labels are stamped on **exactly the rows the store keeps** (after feature dropna, tradability and membership), with the walk
**continued past the LAST stored row into the real bars after it** (the Target_Return-NaN tail — reproduces the old series-end
behaviour, no extra tail loss). Rationale:
- backtest.simulate_ticker (backtest.py:352) and meta_label (meta_label.py:746) run `exit_walk` over the **stored** frame, so this is the
  only ordering that gives label == backtest exactly (MAP §6 invariant 1) and makes every TB_Bars a valid row offset for
  sample_weights uniqueness / hypersearch n_eff.
- (a) NaN-driven drops: the removed bar IS a real price bar, but the backtester cannot see it either; making labels see it while the
  backtest does not would break parity. The right place to keep those bars is upstream (neutral-fill RS_vs_SPY when SPY ROC==0 and
  Volume_Ratio on zero-volume stretches) — a feature-value change in indicators.py, **owner decision, not done here**.
- (b) membership drops are day-granular and the stock EOD barrier caps every walk inside its session (both keyed on the UTC date),
  so re-stamping changes only the label of the EOD-bar entry right before a gap — an entry backtest/meta_label never take. Re-stamped for span validity + parity.
- Rows whose walk never crossed a removed bar keep **byte-identical** labels (pinned by test).

## 3. Diff summary

harvest_stock_data.py
- New (before prepare_stock_data): `TBSpanError`, `TB_SPAN_EXIT_CODE = 3`, `_TB_PRICE_COLS`, `_TB_WALK_BARS` registry,
  `_tb_price_frame`, `_stamp_tb_labels(stored, bars, asset_type)` (checks sorted+unique index, walks stored rows + real bars after the
  last stored row, restores the legacy column order — TB_* right after the last Target_Return_{fb}, so the store schema is unchanged), `_tb_cols`.
- prepare_stock_data (:318-345): TB stamp moved from before short-flow/cost-regime to AFTER `dropna()` + `_asof_tradability_mask`;
  the dropna now runs before TB exists, so it keeps exactly the rows the old stamp-then-dropna kept. Then TB-NaN suffix drop, guard →
  `raise TBSpanError` on violation; records the ticker's real bars in `_TB_WALK_BARS` (≈60-120 MB total, OHLC+ATR only).
- New `_restamp_after_membership` (:424): re-stamps every ticker the membership mask removed rows from; prints `[TB-RESTAMP] <t>` for interior cases; raises if an interior-hit ticker has no walk bars.
- main(): clears the registry; `TBSpanError` from prepare → `FATAL [TB-GUARD] … NO training store written`, `sys.exit(3)` (:644);
  re-stamp after the membership mask (:677); **final fail-loud guard right before `save_training_data`** (:722) — any interior removal
  after the last stamp (panel ranks, fills, sentiment) → exit 3, no store write, no sidecar write.
- `_warn_tb_span_violation` / `_tb_membership_guard` keep their report-only contracts (pinned by test_r2c_measurement_kernels);
  callers now turn False into fatal. Message + R2C-06 comment block updated. The "check itself failed" fail-soft branch is unchanged
  (pinned), but non-unique/unsorted indexes now raise earlier in `_stamp_tb_labels`.

harvest_crypto_data.py — same reorder in prepare_data (:206-230) + twin helpers (`TBSpanError`, `_stamp_tb_labels`,
`_removals_prefix_suffix_only`) + exit 3 in main (:420). No masks/post-concat row removal in crypto (only the sentiment column is added), so no re-stamp stage.

## 4. Verification on real data (same bars, old vs new code, read-only)

| name | rows equal | features equal | cols/order equal | TB rows changed |
|---|---|---|---|---|
| AAPL | yes (22,577) | yes | yes | 147 (0.65 %) |
| QBTS | yes (7,522) | yes | – | 37 (0.49 %) |
| BTC / ETH / XRP | yes | yes* | – | 0 |
| DOGE / LINK / SOL | yes | yes* | – | 87 (0.17 %) / 419 (0.84 %) / 198 (0.50 %) |

New TB_* == compute_tb_labels(stored rows) away from the series end (AAPL, QBTS). *crypto feature diffs vs the 22:16 store were only
OI_Chg_24h/OI_Z (inf in the store) — the oi_archive fix landed after that harvest, unrelated to R4.

## 5. Model-facing? Crypto?

**Yes, model-facing**: TB_Ret/TB_Bars/TB_Reason values change for rows whose walk crossed a removed bar (~0.2-0.8 % of rows of affected
names), and TB_Bars-derived uniqueness weights / n_eff change with them. Rows, features and column order are unchanged. It lands inside
this rebuild (gotcha #2 already applies).
**Crypto needs it too and got it** — the 22:16 crypto store is invalid for DOGE/LINK/SOL: **rerun the crypto harvest as well** (sidecar is present, so it is incremental).

Follow-ups for the owner / docs agent: (1) upstream neutral-fill of RS_vs_SPY (indicators.py:643) and Volume_Ratio (:522) so real bars stay in the store (feature change, train/serve parity with predict_now);
(2) docs/MODULES.md §harvest_crypto_data / §harvest_stock_data "Known issues" (L7 lines) and 06_signal_model_plan deferred item 8 now stale;
(3) Target_Return_{fb} is still shift(-fb) on the unfiltered frame and hypersearch's purge uses positional label_idx on stored rows — conservative (over-purges) for names with interior removals, not a leak;
(4) run_pipeline treats exit 3 like any nonzero harvest exit (retry/notify).

## 6. Tests

New tests/test_tb_restamp_2026_09.py (14 tests, Mac-safe): 3-ticker synthetic panel — AAA interior NaN feature row, BBB raw price/timestamp gap
+ −8 % overnight jump, CCC membership gap. Asserts: spans positionally valid (re-walk == stored), label == backtest replay (exit_walk on stored
rows), no [TB-GUARD] through prepare and through main() end-to-end, `[TB-RESTAMP] CCC`, byte-identical legacy labels where the walk never
crossed a removed bar, legacy column order, old ordering provably invalid on the same fixture; fail-loud: prepare raises TBSpanError on
post-stamp interior removal (stock + crypto), main exits 3 with no save when prepare raises and when a downstream step eats an interior row,
re-stamp without walk bars raises, unsorted/duplicate index raises.

py_compile both scripts: OK.
`CUDA_VISIBLE_DEVICES='' $JPY -m pytest tests/test_tb_restamp_2026_09.py tests/test_improve_harvest.py tests/test_policy_exits*.py tests/test_sample_weights*.py tests/test_c26_Q*.py tests/test_r2c_measurement_kernels.py tests/test_c26_T2.py tests/test_grp_training.py tests/test_review_b21.py tests/test_review_b20.py tests/test_asof_universe.py tests/test_oi_inf_2026_09.py tests/test_c26_T3.py tests/test_imports.py -q -p no:cacheprovider`
(no tests/test_harvest*.py exist)
```
tests/test_oi_inf_2026_09.py .......................                     [ 87%]
tests/test_c26_T3.py .....................................               [ 95%]
tests/test_imports.py .....................                              [100%]
================== 459 passed, 1 skipped, 1 warning in 11.89s ==================
```
(the one warning is a pre-existing SettingWithCopyWarning in data_sources.py:203)
