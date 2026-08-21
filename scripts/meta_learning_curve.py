"""B04.3 learning-curve harness (campaign 2026-08, packet V3) — Jetson CLI.

Measures the meta-label honest sample floor on the CURRENT meta-row
population — i.e. the population the next `python meta_label.py` retrain
would train on, including the META_OOF_PRED / META_REPLAY_POLICY_PARITY
flag state — BEFORE the OOF flip cuts rows. Subsamples the pool with
temporal blocks (n x seeds per meta_curve.build_subsample_plan), fits a
small LightGBM per draw, scores a FIXED chronological eval slice with the
pure rank AUC, and reports the B04.3 floor: smallest n with
(plateau_AUC - mean_AUC) < 0.01 AND cross-seed veto flip-rate < 10%,
plus inverse-power-law extrapolations (Figueroa 2012).

Measurement-only: writes {prefix_}meta_curve_report.json and NEVER
touches live/staged meta artifacts. Run once per book:

    python scripts/meta_learning_curve.py                # crypto
    python scripts/meta_learning_curve.py --prefix stock

Cross-book extrapolation is operational, not coded: the crypto fit's
exponent b is the transferable quantity; compare each report's
floor.honest_floor with its own n_rows_assembled.

Style precedent: scripts/reliability_report.py (lazy heavy imports so
--help works on the dev Mac, which has no lightgbm).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import meta_curve  # noqa: E402  (pure numpy + stdlib — Mac-safe)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prefix', default='', choices=['', 'stock'],
                    help="'' = crypto book, 'stock' = stock book")
    ap.add_argument('--seeds', type=int, default=meta_curve.DEFAULT_N_SEEDS,
                    help='subsample draws per grid point')
    ap.add_argument('--block-len', type=int,
                    default=meta_curve.DEFAULT_BLOCK_LEN,
                    help='rows per contiguous temporal block')
    ap.add_argument('--grid', default='100,200,400,800,1600,3200',
                    help='csv of subsample sizes (the full pool is always '
                         'appended as the "all" point)')
    ap.add_argument('--eval-fraction', type=float, default=0.2,
                    help='final chronological fraction held out as the '
                         'fixed eval slice')
    ap.add_argument('--no-tiering', action='store_true',
                    help='force the full 31-leaf params at every n '
                         '(default mirrors the deployed oof_starvation_tier '
                         'shrink for n<1000)')
    ap.add_argument('--out', default=None,
                    help='report path (default: '
                         '{prefix_}meta_curve_report.json in the repo root)')
    ap.add_argument('--base-seed', type=int,
                    default=meta_curve.DEFAULT_BASE_SEED)
    args = ap.parse_args()

    try:
        grid = tuple(int(v) for v in args.grid.split(',') if v.strip())
    except ValueError:
        ap.error(f'--grid must be a csv of ints, got {args.grid!r}')
    if not grid:
        ap.error('--grid is empty')

    t0 = time.time()

    # --- lazy heavy imports (mirror meta_label.train_meta, incl. graceful
    # returns) — everything below needs the Jetson stack -------------------
    try:
        import lightgbm as lgb
    except ImportError:
        print('[CURVE] lightgbm not installed')
        return 1
    import datetime as _dt

    import meta_label as ml
    from backtest import (_load_artifacts, _predict_ticker, _load_lgb,
                          _load_q10, _entry_window_mask)
    from data_utils import load_training_data
    from strategy_config import policy_for
    import strategy_config as _sc
    import calibration

    prefix = args.prefix
    asset_type = prefix or 'crypto'
    try:
        model, scaler, config, feature_cols = _load_artifacts(prefix)
    except FileNotFoundError:
        print('[CURVE] primary model artifacts missing — train the primary first')
        return 1
    lgb_primary = _load_lgb(prefix)
    threshold = config.get('trade_threshold', 0.15)
    policy = policy_for(asset_type)

    _oof_on = bool(getattr(_sc, 'META_OOF_PRED', False))
    _parity_on = bool(getattr(_sc, 'META_REPLAY_POLICY_PARITY', False))
    p_pre = f'{prefix}_' if prefix else ''
    _manifest = None
    try:
        with open(BASE_DIR / f'{p_pre}model_v2.manifest.json') as _mf:
            _manifest = json.load(_mf)
    except Exception:
        _manifest = None
    oof_pack, oof_status = ml.load_oof_npz(BASE_DIR / f'{p_pre}oof_preds.npz',
                                           _manifest)
    # Measured population == what the next retrain would use.
    use_oof = _oof_on and oof_status == 'ok'
    q10_model, q10_floor = None, None
    if _parity_on:
        pack_ = _load_q10(prefix)
        if pack_:
            q10_model, q10_floor = pack_

    df = load_training_data('stock' if prefix == 'stock' else 'crypto')
    if df.empty:
        print('[CURVE] no training data')
        return 1

    # Hypersearch-holdout exclusion — same row time-quantile as train_meta.
    times_ns = df.index.astype('int64')
    cutoff = np.quantile(times_ns, 1.0 - ml.HOLDOUT_FRACTION)

    X, y, ts, sym = [], [], [], []
    for ticker, tdf in df.groupby('Ticker', sort=False):
        tdf = tdf.sort_index()
        tdf = tdf[tdf.index.astype('int64') <= cutoff]
        if len(tdf) < config['seq_len'] + 50:
            continue
        if any(c not in tdf.columns for c in feature_cols):
            continue
        preds, q10_arr = _predict_ticker(model, scaler, config, feature_cols,
                                         tdf, lgb_model=lgb_primary,
                                         q10_model=q10_model)
        entry_ok = (_entry_window_mask(tdf.index)
                    if asset_type == 'stock' else None)
        oof_row_preds = (ml.join_oof_to_index(tdf.index.astype('int64'),
                                              str(ticker), oof_pack)
                         if use_oof else None)
        r, l, nr, t, et = ml._gen_meta_rows(
            tdf, preds, asset_type, threshold, policy,
            entry_preds=oof_row_preds, parity=_parity_on, entry_ok=entry_ok,
            q10_preds=q10_arr, q10_floor=q10_floor, diag=None)
        X.extend(r); y.extend(l); ts.extend(t)
        sym.extend([str(ticker)] * len(r))   # the one addition vs train_meta:
        # per-row symbol for the symbol-level flip-rate grouping
    # Free the harvest frame + primary before LightGBM allocates (Jetson 8GB).
    del df, model, scaler, lgb_primary, q10_model, oof_pack

    if len(X) < 50:
        print(f'[CURVE] only {len(X)} replayed trades — nothing to curve')
        return 1

    # Same unstable sort as train_meta (measurement-only, but keep the mirror).
    order = np.argsort(np.asarray(ts))
    X = np.asarray(X, float)[order]
    y = np.asarray(y, float)[order]
    sym_arr = np.asarray(sym)[order]

    n_rows = len(X)
    n_eval = max(30, int(n_rows * args.eval_fraction))
    if n_eval >= n_rows:
        print(f'[CURVE] eval slice ({n_eval}) swallows the pool ({n_rows})')
        return 1
    X_ev, y_ev, sym_ev = X[-n_eval:], y[-n_eval:], sym_arr[-n_eval:]
    X_pool, y_pool = X[:-n_eval], y[:-n_eval]
    n_pool = len(X_pool)
    if np.unique(y_ev).size < 2:
        print('[CURVE] WARNING: eval slice is one-class — AUC will be None '
              '(flip rate / frac_below_veto still measured)')

    plan = meta_curve.build_subsample_plan(n_pool, grid, args.seeds,
                                           args.block_len, args.base_seed)
    n_values = sorted({d['n'] for d in plan})
    p_mats = {n: np.full((args.seeds, n_eval), np.nan) for n in n_values}
    records = []
    for draw in plan:
        n, s, idx = draw['n'], draw['seed'], draw['idx']
        rec = {'n': n, 'seed': s, 'auc': None, 'frac_below_veto': None,
               'p_q10': None, 'p_median': None, 'p_q90': None,
               'calib': None, 'error': None}
        try:   # fail-open: a failed draw becomes an error record
            Xd, yd = X_pool[idx], y_pool[idx]
            # mirror of meta_label.train_meta params — keep in sync
            params = {
                'objective': 'binary', 'metric': 'auc',
                'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.05,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8,
                'bagging_freq': 5, 'verbose': -1, 'n_jobs': 4,
            }
            if not args.no_tiering:
                tier, ov = ml.oof_starvation_tier(n)
                if tier == 'shrunk':
                    params.update(ov)
                elif tier == 'starved':
                    # oof_starvation_tier's 'starved' verdict is a PUBLISH
                    # policy, not a measurement bound — the curve must
                    # measure below the floor (that is its purpose), so
                    # apply the shrunk overrides here anyway.
                    params.update(dict(ml.OOF_SHRUNK_PARAMS,
                                       min_data_in_leaf=max(20, n // 20)))
            params.update({'seed': s, 'bagging_seed': s,
                           'feature_fraction_seed': s, 'data_random_seed': s,
                           'deterministic': False})
            split_d = int(n * 0.8)
            fit_mode = 'early_stop'
            if np.unique(yd[split_d:]).size < 2 or split_d < 1:
                # one-class internal val slice: no early stopping possible
                fit_mode = 'fixed200'
                dtr = lgb.Dataset(Xd, label=yd,
                                  feature_name=ml.META_FEATURES)
                booster = lgb.train(params, dtr, num_boost_round=200)
            else:
                dtr = lgb.Dataset(Xd[:split_d], label=yd[:split_d],
                                  feature_name=ml.META_FEATURES)
                dva = lgb.Dataset(Xd[split_d:], label=yd[split_d:],
                                  reference=dtr)
                booster = lgb.train(
                    params, dtr, num_boost_round=400, valid_sets=[dva],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            raw_ev = booster.predict(X_ev)
            # Calibrate on the draw's internal 20% slice (pure-numpy
            # calibration.fit_calibrator; declines on degenerate input).
            cal = None
            if split_d < n:
                raw_slice = booster.predict(Xd[split_d:])
                cal = calibration.fit_calibrator(raw_slice, yd[split_d:])
            if cal is not None:
                p_ev = np.clip(cal.predict(raw_ev), 0.0, 1.0)
                rec['calib'] = type(cal).__name__
            else:
                p_ev = np.clip(raw_ev, 0.0, 1.0)
                rec['calib'] = 'raw'
            # Rank metric on raw scores — calibration-invariant, comparable
            # across n and seeds on the fixed eval slice.
            rec['auc'] = meta_curve.rank_auc(raw_ev, y_ev)
            rec['frac_below_veto'] = float(np.mean(p_ev < meta_curve.VETO_PROB))
            q10_, q50_, q90_ = np.quantile(p_ev, (0.10, 0.50, 0.90))
            rec['p_q10'], rec['p_median'], rec['p_q90'] = (
                float(q10_), float(q50_), float(q90_))
            rec['fit_mode'] = fit_mode
            p_mats[n][s] = p_ev
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        records.append(rec)

    flip_by_n = {n: meta_curve.veto_flip_rate(p_mats[n], groups=sym_ev)
                 for n in n_values}
    n_err = sum(1 for r in records if r['error'] is not None)
    meta = {
        'prefix': prefix,
        'generated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'n_rows_assembled': int(n_rows),
        'n_pool': int(n_pool),
        'n_eval': int(n_eval),
        'pred_source': 'oof' if use_oof else 'in_sample',
        'oof_status': oof_status,
        'parity': _parity_on,
        'tiering': not args.no_tiering,
        'block_len': int(args.block_len),
        'seeds': int(args.seeds),
        'grid': list(grid),
        'base_seed': int(args.base_seed),
        'elapsed_s': round(time.time() - t0, 1),
        'n_draws_ok': len(records) - n_err,
        'n_draws_err': n_err,
        'primary': {'saved_at': (_manifest or {}).get('saved_at'),
                    'score': (_manifest or {}).get('score')},
    }
    report = meta_curve.assemble_report(records, flip_by_n, meta=meta)

    out = Path(args.out) if args.out else (
        BASE_DIR / f'{p_pre}meta_curve_report.json')
    tmp = f'{out}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    # --- human summary -----------------------------------------------------
    print(f'\n=== Meta-label learning curve ({asset_type}, '
          f'pred_source={meta["pred_source"]}, parity={_parity_on}) ===')
    print(f'  pool={n_pool}  eval={n_eval}  draws ok/err='
          f'{meta["n_draws_ok"]}/{n_err}')
    print(f'  {"n":>6}  {"AUC mean±std":>16}  {"flip rate":>10}  '
          f'{"frac<veto":>10}')
    for g in report['grid']:
        am = ('n/a' if g['auc_mean'] is None
              else f"{g['auc_mean']:.4f}±{g['auc_std']:.4f}")
        fr = 'n/a' if g['flip_rate'] is None else f"{g['flip_rate']:.3f}"
        fv = ('n/a' if g['frac_below_veto_mean'] is None
              else f"{g['frac_below_veto_mean']:.3f}")
        print(f'  {g["n"]:>6}  {am:>16}  {fr:>10}  {fv:>10}')
    for name in ('powerlaw_auc', 'powerlaw_flip'):
        f_ = report[name]
        if f_['ok']:
            print(f'  {name}: err = {f_["a"]:.4g} * n^(-{f_["b"]:.3f}) '
                  f'+ {f_["c"]:.4g}  (r2={f_["r2"]:.3f})')
        else:
            print(f'  {name}: fit declined ({f_["reason"]})')
    fl = report['floor']
    print(f'  floors: empirical={fl["empirical_n"]}  '
          f'extrapolated_auc={fl["extrapolated_n_auc"]}  '
          f'extrapolated_flip={fl["extrapolated_n_flip"]}')
    hf = fl['honest_floor']
    verdict = ('n/a — no floor criterion satisfied on this grid (starvation)'
               if hf is None else
               f'honest floor ≈ {hf} rows; this book currently assembles '
               f'{n_rows} — self-supporting: {"yes" if n_rows >= hf else "no"}')
    print(f'  VERDICT: {verdict}')
    print('  (cross-book: run once per --prefix; the crypto fit\'s exponent b '
          'is the transferable quantity)')
    print(f'  report -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
