# OBJECTIVE_SESSION_MASK — proposal and pre-registered flip rule (SIG-R2-MASK, 2026-09-27)

Status: **STAGED, not landed, default OFF.** Patch + tests + driver live in the SIGNAL landing area
(`landing/SIG-R2-MASK/`); nothing here changes a trial score until the owner flips the flag at a
gotcha-#2 study reset. Model-facing.

## The mismatch (OBJECTIVE)

The stock store is extended-session: hourly bars stamped with their OPEN time, 04:00–19:00 ET,
~49 % of rows pre/post-market, zero weekend rows (`scripts/bars_per_year_census.py`). The trainer's
objective (`objective_utils.simulate_trades_core`) may open a long on **every** row. The live stock
book opens longs only with the market clock open (`base_loop.py:313` →
`stock_loop.check_market_hours` :125-141) **and** inside `STOCK_ENTRY_WINDOWS_ET`
= `[('09:45','11:00'), ('14:30','15:30')]` (`strategy_config.py:113-117`, `ENTRY_WINDOWS_ENABLED=True`;
`stock_loop._in_entry_window` :215-230, enforced :1003). So the search ranks, and the holdout DSR
certifies, entries the book can never take. This is the same kind of mismatch as the short leg that
`OBJECTIVE_LONG_ONLY` fixed. Crypto has no mask: `crypto_loop.check_market_hours` returns True
(`crypto_loop.py:82-83`) and the crypto loop never calls an entry window.

Kill list / removed-code check: `research/KILL_LIST.md` has no session/RTH/entry-window entry.
`08_removed_code.md:14` lists the "entry-window fate" as a pending evidence-gated verdict. The mask
reads the live window constants at call time, so it follows any future window decision.

## Layer consistency today

| Layer | Entry gating vs live | Cite |
|---|---|---|
| live stock_loop | clock open + entry window (wall-clock minute) | stock_loop.py:125-141, :215-230, :1003 |
| backtest.simulate_ticker (promotion replay) | **consistent**: `_entry_window_mask` on the bar open-time | backtest.py:240-274, :315-318, :367 |
| meta_label._gen_meta_rows | mask built, but enforced **only** under `META_REPLAY_POLICY_PARITY` (default OFF) | meta_label.py:772-786, :932 |
| hypersearch fold / pruning score | **inconsistent** (all rows) | hypersearch_v2.py:1148, :1209 |
| hypersearch regime penalty | **inconsistent** | :1269 |
| hypersearch holdout certificate | **inconsistent** (the q10 veto is the only long veto) | :1874, :1877 |
| threshold reselect (H2) + Sharpe-grid diagnostic | **inconsistent** | :2688, :2756-2769 |
| blend NNLS weight (`fit_blend_weight_v2`) | regression over all validation rows, not a trade walk. Left unchanged (JUDGMENT) | :2663-2668 |
| q10 veto floor | 15th percentile of q10 over all validation rows, not a trade scorer. Left unchanged | :1551 |

## The change (behind `OBJECTIVE_SESSION_MASK`, default False)

- **Resolution:** `TRADER_OBJECTIVE_SESSION_MASK` (1/true/yes/on) wins over
  `getattr(strategy_config, 'OBJECTIVE_SESSION_MASK', False)`.
- **When ON, stock book only:** the mask `entry_ok` is True when the bar's open-time (UTC → America/New_York) falls in a window,
  using the same `start <= minute < end` rule as `stock_loop._in_entry_window`, and on a weekday.
  If `ENTRY_WINDOWS_ENABLED` is False the mask falls back to RTH 09:30–16:00, which is the live clock's wall-clock fallback.
  The mask is built once per data load in `create_objective`. Every trade scorer then receives
  `long_veto = ~entry_ok`: the pruning and fold scores, the regime penalty, the holdout certificate (OR'd with the q10 veto),
  the threshold reselect and the Sharpe-grid diagnostic.
- **When OFF:** every veto is None, so the scoring is byte-identical. This is pinned by comparing the holdout report against the pre-patch module.

## Size of the effect on the real store (measured through hwlock, index + Ticker columns only)

| Rule | Full store (1,536,356 rows) | `--max-rows 200000` cap (stock pipeline) |
|---|---|---|
| **Point rule** (the implemented one; open-time in window = backtest parity) | 14.49 % (open hours 10:00, 15:00 ET) | 13.65 % |
| Interval rule (the live decision hour [open, open+60 min) meets a window) | 28.98 % (09:00, 10:00, 14:00, 15:00) | 27.29 % |

Flipping the flag therefore leaves about 1 row in 7 as an eligible long entry. Per-trial n_trades, the fold Sharpe
annualization (occupancy = n_trades·fb / n_rows) and the holdout DSR pool all change, so old Optuna scores
become incomparable. **The flip must be paired with a gotcha-#2 study reset.**

## OWNER items (JUDGMENT)

1. **Decision-hour offset.** `predict_now` serves closed bars only (`market_data.drop_forming_bar` :343-366).
   The live decision for training row *i* is therefore made while bar *i* is forming, during wall-clock
   [open_i, open_i+60). The live 30 s loop can enter on the 09:00 row (09:45–10:00) and the 14:00 row
   (14:30–15:00), which the point rule excludes. The point rule matches `backtest._entry_window_mask` and
   the meta parity mask, so all three offline layers under-admit by about 2×. Switching all three to the interval rule is one owner decision
   across SIGNAL files (backtest.py, meta_label.py, this mask). This patch deliberately keeps
   backtest parity.
2. `long_veto` blocks longs only. With `OBJECTIVE_LONG_ONLY=True` (strategy_config.py:202, current) no
   shorts are scored. If that flag is ever reverted, off-window shorts would be scored again.
3. `backtest._entry_window_mask` returns all-True when `ENTRY_WINDOWS_ENABLED=False`. That admits pre/post-market
   entries the live clock forbids. The flag is non-default, so the issue is latent.

## Pre-registered decision rule for the flip

Run on the **same saved stock winner** with the mask OFF vs ON (`scripts/session_mask_holdout_ab.py`, which calls the real
`evaluate_on_holdout` twice). Report holdout Sharpe, n_trades, n_eff (DSR) and n_eff_v2 (calendar) for each.

- **ESCALATE (flip REQUIRED, owner):** the unmasked certificate passes (Sharpe > 0 and DSR ≥ DSR_MIN = 0.60) but
  the masked DSR is < 0.60. The certified edge lives outside the tradable hours.
- **PROPOSE flip:** both (i) masked n_eff_v2 ≥ 10 (the PROMOTION_GATE_V2 floor) and (ii) mean masked per-trade
  return − mean unmasked ≥ −1 SE, where the SE comes from a paired weekly-block bootstrap over the ISO weeks of the
  entry times (2000 draws, seed 0).
- **HOLD** otherwise.

Command, on the Jetson with **no trainer running** (load_data holds the capped panel, ~1.5–2.5 GB):

    source <scratchpad>/jenv.sh; cd /home/kyle/trader
    CUDA_VISIBLE_DEVICES='' TORCH_NUM_THREADS=1 OMP_NUM_THREADS=1 $JPY scripts/session_mask_holdout_ab.py \
        --artifact-prefix stock_challenger_ --max-rows 200000 --json session_mask_ab_stock.json
    # champion slot: --artifact-prefix stock_ ; raw-LSTM certificate: add --no-blend

`--max-rows` and `--preset` must match the run that produced the winner. The driver refuses to score
when the reloaded feature columns differ from the saved ones, or when the mask is not landed. No
`*model_v2*` stock artifacts exist on the box today (clean rebuild), so this runs after the next
stock search.
