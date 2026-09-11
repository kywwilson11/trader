"""FR-02 fixed-config training-window A/B runner (R2C-05, Jetson-only).

Retrains the CURRENT champion config — no Optuna search, therefore zero new
selection pressure — on several trailing data spans and scores each arm on
the IDENTICAL holdout with IDENTICAL DSR deflation:

  - no Optuna, no ratchet, no adaptive-state writes, no model artifacts on
    disk (final_refit + train_lgb_ensemble(save=False) + blend fit +
    evaluate_on_holdout are called DIRECTLY);
  - identical cum_trials passed to every arm, so deflation is equal and the
    comparison is pure SR ordering under one null (FR-02 measurement plan);
  - fixed TRAINER_SEED (derived per arm via objective_utils.derive_seed) —
    run the R2C-04 seed-determinism check first (06 plan §4.8);
  - per-arm stage0-schema dump of the holdout blend predictions
    (window_ab_<book>_<arm>_stage0.json) for scripts/ic_by_name.py
    --time-key ts.

FR-01 prerequisite: run with TRADER_FIXED_HOLDOUT_DAYS set (e.g. 60) so all
arms share one fixed-width holdout — under the legacy proportional rule a
1Y arm gets a ~44-day holdout vs ~200 for full history and the arms are not
comparable. The runner warns loudly when the fixed span is not active.

Decision rule (06 plan §4.9): adopt a shorter window ONLY on both the
fold-objective AND holdout-DSR wins; winner through
`backtest.py --prefix '' --days 60 --gate`; else keep full history and
record the durable negative (which also skips FR-09).

Usage (crypto book):
    TRADER_FIXED_HOLDOUT_DAYS=60 python scripts/window_ab.py \
        --arms full,730,365,pt2007 --seed 42
Arms: 'full' (no mask), an integer day count (trailing window), or
'pt2007' (Pesaran-Timmermann 2007 arm: 8 months pre-break + everything
after the 2025-10-10 cascade, i.e. window start 2025-02-10).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gc
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch

from adaptive_config import load_adaptive_state
from blend_fit import fit_blend_weight_v2, effective_lstm_weight
from gpu_lock import acquire_for_training
from model_v2 import RegressionLSTM
from objective_utils import derive_seed
from stage0_preds import write_rows

import hypersearch_v2 as hs

# Pesaran-Timmermann 2007 arm: keep 8 months pre-break + everything after
# the 2025-10-10 cascade => window start 2025-02-10 (frontier plan FR-02).
PT2007_START_TS = int(pd.Timestamp('2025-02-10').timestamp())

# Training-only params final_refit needs that the serving config_v2.pkl
# does not carry; adaptive best_params normally supplies them — these
# fallbacks only apply (loudly) when that record is missing.
_TRAIN_PARAM_FALLBACKS = {'batch_size': 1024, 'learning_rate': 1e-3,
                          'weight_decay': 1e-4, 'scheduler': 'plateau'}


def parse_args():
    ap = argparse.ArgumentParser(
        description='FR-02 fixed-config training-window A/B (no Optuna, '
                    'no ratchet, no model saves)')
    ap.add_argument('--prefix', type=str, default='',
                    help="Book prefix ('' = crypto, 'stock_' = stocks)")
    ap.add_argument('--data', type=str, default='training_data.csv',
                    help='Training CSV fallback path (parquet-first via '
                         'data_utils, same as hypersearch)')
    ap.add_argument('--preset', type=str, default=None,
                    help='Indicator preset override (None = '
                         'load_indicator_config(); run_pipeline uses '
                         '"stationary")')
    ap.add_argument('--max-rows', type=int, default=500_000)
    ap.add_argument('--arms', type=str, default='full,730,365,pt2007',
                    help="Comma list: 'full', integer trailing days, "
                         "'pt2007'")
    ap.add_argument('--seed', type=int, default=42,
                    help='Fixed TRAINER_SEED base (per-arm sub-seeds via '
                         'derive_seed)')
    ap.add_argument('--cum-trials', type=int, default=None,
                    help='Deflation pool for every arm (default: adaptive '
                         'state cum_trials — identical across arms either '
                         'way)')
    ap.add_argument('--epochs', type=int, default=None,
                    help='Refit epoch budget override (default: the '
                         "champion config's refit.epochs record)")
    ap.add_argument('--out', type=str, default=None,
                    help='Summary JSON path (default: '
                         'window_ab_summary_<book>.json)')
    return ap.parse_args()


def _arm_window_days(arm, data_end_ts):
    """Trailing window_days for an arm token ('full' -> None)."""
    if arm == 'full':
        return None
    if arm == 'pt2007':
        return max((int(data_end_ts) - PT2007_START_TS) / 86400.0, 1.0)
    return float(int(arm))


def _predict_rows(state, scaler, cfg, all_features, rows, input_dim,
                  lgb_booster=None, q10_booster=None, scaled=None):
    """LSTM (and optional LGB mean / q10) predictions for global rows —
    the exact evaluate_on_holdout / predict_now inference contract (same
    gather_windows/flatten ordering, same batching). Both boosters score
    the same flattened windows in ONE pass. `scaled` (optional) reuses a
    precomputed scaler.transform of the full panel: run_arm calls this up
    to NUM_FOLDS+2 times per arm and the full-panel transform is the
    dominant repeated cost on the Jetson."""
    seq_len = cfg['seq_len']
    offsets = np.arange(-seq_len, 0)
    _own_scaled = scaled is None
    if _own_scaled:
        scaled = scaler.transform(all_features).astype(np.float32)
    mdl = RegressionLSTM(input_dim, cfg['hidden_dim'], cfg['num_layers'],
                         cfg['dropout'], cfg['n_heads']).to(hs.device)
    mdl.load_state_dict(state)
    mdl.eval()
    use_amp = hs.device.type == 'cuda'
    lstm_preds = []
    lgb_preds = (np.empty(len(rows), dtype=np.float64)
                 if lgb_booster is not None else None)
    q10_preds = (np.empty(len(rows), dtype=np.float64)
                 if q10_booster is not None else None)
    with torch.inference_mode():
        for i in range(0, len(rows), 1024):
            vidx = rows[i:i + 1024]
            w = hs.gather_windows(scaled, vidx, offsets)
            xb = torch.from_numpy(w).to(hs.device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                vo = mdl(xb)
            lstm_preds.append(vo.cpu().numpy())
            if lgb_booster is not None or q10_booster is not None:
                flat = w.reshape(len(vidx), -1)
                if lgb_booster is not None:
                    lgb_preds[i:i + len(vidx)] = lgb_booster.predict(flat)
                if q10_booster is not None:
                    q10_preds[i:i + len(vidx)] = q10_booster.predict(flat)
    del mdl
    if _own_scaled:
        del scaled
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(lstm_preds), lgb_preds, q10_preds


def _stage0_dump(path, holdout_idx, blend, lstm_p, lgb_p, q10_p, returns,
                 all_times, tickers, ticker_boundaries, fb, threshold):
    """Per-arm stage0-schema dump of the holdout predictions (bare JSON
    list, ic_by_name's contract). fwd_return comes straight from the
    harvest Target_Return arrays (already percent — same units as pred);
    'close' is not available from the training arrays, and stage0
    consumers tolerate its absence. Rows are spaced >= fb bars per name
    (global row spacing == bar spacing inside a contiguous ticker block),
    matching the producer's non-overlap guarantee."""
    pos_of = {int(r): k for k, r in enumerate(holdout_idx)}
    rows_out = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        t_rows = [int(r) for r in holdout_idx if start <= r < end]
        last = None
        for r in t_rows:
            k = pos_of[r]
            if not np.isfinite(blend[k]) or not np.isfinite(returns[r]):
                continue
            if last is not None and r - last < fb:
                continue
            p = float(blend[k])
            rows_out.append({
                'ts': str(pd.Timestamp(int(all_times[r]), unit='s')),
                'symbol': str(ticker),
                'pred': round(p, 6),
                'signal': round(p, 6),
                'fwd_return': round(float(returns[r]), 6),
                'horizon_bars': int(fb),
                'lstm_pred': round(float(lstm_p[k]), 6),
                'lgb_pred': (round(float(lgb_p[k]), 6)
                             if lgb_p is not None else None),
                'meta_p': None,
                'q10': (round(float(q10_p[k]), 4)
                        if q10_p is not None else None),
                'pred_thresh_ratio': (round(p / float(threshold), 4)
                                      if threshold and float(threshold) > 0
                                      else None),
            })
            last = r
    write_rows(rows_out, path)
    return len(rows_out)


def run_arm(label, window_days, args, cfg, refit_epochs, n_trials,
            asset_type, book):
    """One fixed-config arm: load windowed data -> final_refit -> LGB legs
    (save=False) -> blend fit on folds[-1] val -> evaluate_on_holdout ->
    stage0 dump. Returns the summary dict (fail-soft per arm).

    `label` is the arm TOKEN ('full', '730', 'pt2007') — used for the seed
    derivation, prints and the dump filename; `window_days` is the resolved
    trailing day count (None = full history; main resolves 'pt2007' — all
    arms share the same panel end, trailing masks only trim the front)."""
    summary = {'arm': str(label), 'status': 'failed'}
    data = hs.load_data(args.data, preset_override=args.preset,
                        max_rows=args.max_rows,
                        window_days=window_days)
    (all_features, all_returns_by_fb, all_times, all_label_times, tickers,
     ticker_boundaries, feature_cols, input_dim, preset_name,
     has_multi_horizon, all_tb_bars_by_fb) = data
    summary['window_days'] = window_days
    summary['n_rows'] = int(len(all_times))
    summary['span_days'] = round(
        (int(all_times.max()) - int(all_times.min())) / 86400.0, 1)

    fb = cfg.get('forward_bars', 24)
    key = ('tb', fb) if cfg.get('target_kind') == 'tb' else fb
    returns = all_returns_by_fb.get(key)
    if returns is None:
        returns = all_returns_by_fb.get(fb)
    if returns is None:
        returns = next(v for k, v in all_returns_by_fb.items()
                       if not isinstance(k, tuple))

    # 1) Final refit at a fixed epoch budget, fixed derived seed. No
    #    fold-max fallback exists here — a refit failure fails the arm.
    refit = hs.final_refit(
        cfg, returns, all_features, all_times, all_label_times, tickers,
        ticker_boundaries, input_dim, [refit_epochs], [],
        seed=derive_seed(args.seed, 'window_ab', str(label), 'refit'))
    if refit is None:
        print(f"[ARM {label}] final_refit failed — arm skipped")
        return summary
    state, scaler, refit_info = refit
    summary['refit'] = refit_info

    # 2) LGB legs on the shipping scaler, save=False (nothing touches disk)
    lgb_pack = hs.train_lgb_ensemble(
        args.prefix, scaler, cfg, all_features, all_returns_by_fb,
        all_times, all_label_times, tickers, ticker_boundaries,
        all_tb_bars_by_fb=all_tb_bars_by_fb, save=False)
    booster = q10b = q10f = None
    if lgb_pack and lgb_pack[0] is not None:
        booster, q10b, q10f = lgb_pack[0], lgb_pack[1], lgb_pack[2]

    # ONE full-panel transform per arm, reused by every _predict_rows call
    # below (blend fit + per-fold vector + stage0 dump — Jetson perf).
    scaled = scaler.transform(all_features).astype(np.float32)

    # 3) Blend-weight fit on folds[-1] val (both legs' shipping inputs;
    #    val labels purged of holdout crossings — the M4 guard is
    #    unconditional in this offline runner). NO cross-retrain
    #    smoothing: each arm's w comes from its own data only.
    lstm_weight = None
    fold_sharpes = []
    folds = hs.get_walk_forward_folds(all_times, all_label_times, tickers,
                                      ticker_boundaries, cfg['seq_len'],
                                      purge_val_labels=True)
    if folds and booster is not None:
        rows = folds[-1][1]
        rows = rows[~np.isnan(returns[rows])]
        if len(rows) > 30_000:
            rows = rows[np.argsort(all_times[rows])][-30_000:]
        if len(rows):
            lstm_val, lgb_val, _ = _predict_rows(state, scaler, cfg,
                                                 all_features, rows,
                                                 input_dim,
                                                 lgb_booster=booster,
                                                 scaled=scaled)
            fit = fit_blend_weight_v2(lstm_val, lgb_val, returns[rows],
                                      forward_bars=fb)
            lstm_weight = fit['w']
            summary['blend'] = {
                'w': round(float(fit['w']), 4),
                'w_raw': (round(float(fit['w_raw']), 4)
                          if fit['w_raw'] is not None else None),
                'se': (round(float(fit['se']), 4)
                       if fit['se'] is not None else None),
                'significant': bool(fit['significant']),
                'n_fit': int(fit['n'])}
    # (b) per-fold purged-val Sharpe vector scored with the REFIT state —
    # the fold-objective half of the FR-02 decision rule. HONESTY: the
    # refit trained on ALL purged pre-holdout rows, so these fold-val
    # slices are IN-SAMPLE for it (levels are inflated); every arm is
    # equally in-sample, so the CROSS-ARM ordering is the usable readout,
    # never the absolute values.
    for tr_idx, va_idx in folds:
        va = va_idx[~np.isnan(returns[va_idx])]
        if len(va) < 200:
            fold_sharpes.append(None)
            continue
        lstm_va, _, _ = _predict_rows(state, scaler, cfg, all_features, va,
                                      input_dim, scaled=scaled)
        # Same scoring rule as the trial objective (block-boundary resets
        # under OBJECTIVE_V3) so the "fold-objective wins" half of the
        # decision rule matches what the search itself would score.
        _bids = (hs.ticker_block_ids(va, ticker_boundaries)
                 if hs._objective_v3() else None)
        fold_sharpes.append(round(float(hs.compute_sharpe(
            lstm_va, returns[va], cfg['trade_threshold'], forward_bars=fb,
            asset_type=asset_type, block_ids=_bids)), 4))
    summary['fold_val_sharpes'] = fold_sharpes
    lstm_weight = effective_lstm_weight(lstm_weight, booster is not None)

    # 4) The certificate: identical n_trials for every arm.
    report = hs.evaluate_on_holdout(
        state, scaler, cfg, all_features, all_returns_by_fb, all_times,
        tickers, ticker_boundaries, input_dim, asset_type,
        n_trials=n_trials, all_tb_bars_by_fb=all_tb_bars_by_fb,
        lgb_booster=booster, q10_booster=q10b, q10_floor=q10f,
        lstm_weight=lstm_weight)
    if report is not None:
        summary['holdout'] = {k: v for k, v in report.items()
                              if k != 'trade_returns'}

    # 5) Stage0-schema dump of the holdout blend predictions for
    #    scripts/ic_by_name.py (purged IC per name — FR-02 metric (c)).
    try:
        holdout_idx = hs.get_holdout_indices(all_times, tickers,
                                             ticker_boundaries,
                                             cfg['seq_len'])
        holdout_idx = holdout_idx[~np.isnan(returns[holdout_idx])]
        lstm_h, lgb_h, q10_h = _predict_rows(state, scaler, cfg,
                                             all_features, holdout_idx,
                                             input_dim,
                                             lgb_booster=booster,
                                             q10_booster=q10b,
                                             scaled=scaled)
        if booster is not None and lstm_weight is not None:
            blend = (float(lstm_weight) * lstm_h
                     + (1.0 - float(lstm_weight)) * lgb_h)
        else:
            blend = lstm_h
        dump_path = f"window_ab_{book}_{label}_stage0.json"
        n_dumped = _stage0_dump(dump_path, holdout_idx, blend, lstm_h,
                                lgb_h, q10_h, returns, all_times, tickers,
                                ticker_boundaries, fb,
                                cfg['trade_threshold'])
        summary['stage0_dump'] = dump_path
        print(f"[ARM {label}] stage0 dump: {n_dumped} rows -> {dump_path}")
    except Exception as e:
        print(f"[ARM {label}] stage0 dump failed (non-fatal): {e}")

    summary['status'] = 'ok'
    del all_features, state, scaler, scaled
    gc.collect()
    return summary


def main():
    args = parse_args()
    book = (args.prefix.rstrip('_') or 'crypto')
    asset_type = 'stock' if 'stock' in (args.prefix or '') else 'crypto'

    # load_data derives its book from the DATA PATH string — with
    # --prefix stock_ and the crypto default --data it would silently
    # load the crypto panel against the stock champion config. Keep the
    # two in lockstep unless --data was given explicitly.
    if asset_type == 'stock' and args.data == 'training_data.csv':
        args.data = 'stock_training_data.csv'
        print("[WINDOW-AB] --prefix is a stock book: defaulting --data to "
              "stock_training_data.csv")

    # FR-01 prerequisite check (comparability warning, never a blocker)
    if hs._fixed_holdout_days() is None:
        print("[WINDOW-AB] WARNING: FIXED_HOLDOUT_DAYS not active — the "
              "legacy proportional holdout gives every arm a DIFFERENT "
              "calendar width and the arms are NOT comparable. Set "
              "TRADER_FIXED_HOLDOUT_DAYS (e.g. 60) — 06 plan §4.2/§4.9.")

    # Champion config: serving keys from config_v2.pkl (the truth for what
    # is deployed), training-only keys from adaptive best_params.
    config = joblib.load(f'{args.prefix}config_v2.pkl')
    adaptive = load_adaptive_state(asset_type)
    train_params = dict(adaptive.get('best_params') or {})
    cfg = {**train_params,
           **{k: config[k] for k in
              ('seq_len', 'hidden_dim', 'num_layers', 'n_heads', 'dropout',
               'huber_delta', 'trade_threshold', 'forward_bars',
               'target_kind') if k in config}}
    for k, v in _TRAIN_PARAM_FALLBACKS.items():
        if k not in cfg:
            print(f"[WINDOW-AB] WARNING: '{k}' missing from adaptive "
                  f"best_params — using fallback {v!r}")
            cfg[k] = v

    # Fixed refit epoch budget: the champion's own refit record, else
    # --epochs (must then be given explicitly).
    refit_epochs = args.epochs
    if refit_epochs is None:
        refit_epochs = (config.get('refit') or {}).get('epochs')
    if refit_epochs is None:
        raise SystemExit('[WINDOW-AB] no refit.epochs in the champion '
                         'config and no --epochs given — refusing to '
                         'guess the epoch budget')

    # Identical deflation pool for every arm (FR-02 metric (a)).
    cum = args.cum_trials
    if cum is None:
        cum = int(adaptive.get('cum_trials', 0) or 0)
    if int(cum) < 2:
        print(f"[WINDOW-AB] WARNING: deflation pool cum_trials={cum} "
              f"(adaptive state missing?) — floored to 2, which barely "
              f"deflates; pass --cum-trials to match the champion's real "
              f"search size")
    n_trials = max(int(cum), 2)
    print(f"[WINDOW-AB] book={book} seed={args.seed} n_trials={n_trials} "
          f"refit_epochs={refit_epochs} threshold="
          f"{cfg.get('trade_threshold')}")

    # Resolve arm tokens; 'pt2007' needs the panel end — peek it cheaply
    # from the parquet index via a full load inside the first arm run is
    # wasteful, so probe once here.
    arm_tokens = [a.strip() for a in args.arms.split(',') if a.strip()]
    from data_utils import load_training_data
    _probe = load_training_data('stock' if asset_type == 'stock'
                                else 'crypto')
    if _probe.empty:
        _probe = pd.read_csv(args.data, index_col=0, parse_dates=True)
    data_end_ts = int(_probe.index.max().timestamp())
    del _probe
    gc.collect()

    results = []
    for arm in arm_tokens:
        wd = _arm_window_days(arm, data_end_ts)
        wd = None if wd is None else round(wd, 1)
        print(f"\n=== ARM {arm} (window_days="
              f"{'full' if wd is None else wd}) ===")
        try:
            results.append(run_arm(arm, wd, args, dict(cfg), refit_epochs,
                                   n_trials, asset_type, book))
        except Exception as e:
            print(f"[ARM {arm}] failed: {e}")
            results.append({'arm': str(arm), 'status': 'failed',
                            'error': str(e)})

    # Summary table + JSON (measurement artifact; no model files written)
    print("\n=== WINDOW A/B SUMMARY (identical holdout, identical "
          f"n_trials={n_trials}) ===")
    for r in results:
        h = r.get('holdout') or {}
        print(f"  {r['arm']:>8}: status={r['status']} "
              f"rows={r.get('n_rows')} span_d={r.get('span_days')} "
              f"sharpe={h.get('sharpe')} dsr={h.get('dsr')} "
              f"n_trades={h.get('n_trades')} n_eff={h.get('n_eff')} "
              f"w={(r.get('blend') or {}).get('w')} "
              f"folds={r.get('fold_val_sharpes')}")
    print("Decision rule (06 plan §4.9): adopt shorter ONLY on both the "
          "fold-objective AND holdout-DSR wins; winner through backtest.py "
          "--gate; else keep full history and record the durable negative.")
    out = args.out or f'window_ab_summary_{book}.json'
    payload = {'generated_at': datetime.now().isoformat(), 'book': book,
               'seed': args.seed, 'n_trials': n_trials,
               'refit_epochs': int(refit_epochs),
               'fixed_holdout_days': hs._fixed_holdout_days(),
               'arms': results}
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Summary: {out}")


if __name__ == '__main__':
    with acquire_for_training('window_ab'):
        main()
