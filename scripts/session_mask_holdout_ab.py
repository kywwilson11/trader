"""SIG-R2-MASK measurement: the saved stock winner's holdout certificate
with OBJECTIVE_SESSION_MASK OFF vs ON (measurement-only, read-only).

Loads {artifact_prefix}model_v2.pth / config_v2.pkl / scaler_v2.pkl /
feature_cols_v2.pkl (+ lgb_model.txt / lgb_q10.txt / lgb_q10_meta.json
when the config carries a fitted lstm_weight — the HYPERSEARCH_V3
blend certificate), reloads the panel exactly as the trainer does
(hypersearch_v2.load_data, same --max-rows / preset), and calls the
REAL hypersearch_v2.evaluate_on_holdout twice — env
TRADER_OBJECTIVE_SESSION_MASK=0 then =1. Writes nothing but --json.

Pre-registered decision rule (research/campaign_2026-09_jetson/
objective_session_mask_proposal.md):
  ESCALATE (flip REQUIRED, owner) : unmasked certificate passes
      (sharpe > 0 and DSR >= DSR_MIN) but the masked DSR < DSR_MIN —
      the certified edge lives outside the tradable hours.
  PROPOSE flip : masked calendar n_eff (n_eff_v2) >= 10 AND
      mean(masked per-trade) - mean(unmasked per-trade) >= -1 SE
      (paired weekly-block bootstrap over ISO weeks of entry time).
  HOLD otherwise.

Memory: load_data holds the capped panel like a hypersearch run does
(~1.5-2.5 GB on the stock store) — run only while no trainer runs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

NEFF_FLOOR = 10.0


def week_ids(epoch_s):
    """ISO-week bucket (Monday 00:00 UTC based) of epoch seconds."""
    t = np.asarray(epoch_s, dtype=np.int64)
    # 1970-01-01 was a Thursday: shift so buckets start on Monday.
    return (t + 3 * 86400) // (7 * 86400)


def weekly_block_se_diff(ret_off, t_off, ret_on, t_on, n_boot=2000, seed=0):
    """SE of mean(ret_on) - mean(ret_off) under a PAIRED weekly-block
    bootstrap: resample calendar weeks with replacement (the same draw
    for both certificates), recompute both means on the drawn weeks.
    Returns (diff, se, n_weeks); se is nan when < 2 weeks or a side is
    empty in every draw."""
    ret_off = np.asarray(ret_off, float)
    ret_on = np.asarray(ret_on, float)
    w_off, w_on = week_ids(t_off), week_ids(t_on)
    weeks = np.union1d(w_off, w_on)
    diff = (float(ret_on.mean()) if ret_on.size else np.nan) - \
           (float(ret_off.mean()) if ret_off.size else np.nan)
    if weeks.size < 2 or ret_on.size == 0 or ret_off.size == 0:
        return diff, float('nan'), int(weeks.size)
    pos_off = np.searchsorted(weeks, w_off)
    pos_on = np.searchsorted(weeks, w_on)
    k = weeks.size
    s_off = np.bincount(pos_off, weights=ret_off, minlength=k)
    c_off = np.bincount(pos_off, minlength=k).astype(float)
    s_on = np.bincount(pos_on, weights=ret_on, minlength=k)
    c_on = np.bincount(pos_on, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(int(n_boot), k))
    cnt = np.stack([np.bincount(d, minlength=k) for d in draws])
    n_off, n_on = cnt @ c_off, cnt @ c_on
    ok = (n_off > 0) & (n_on > 0)
    if ok.sum() < 2:
        return diff, float('nan'), int(k)
    d = (cnt @ s_on)[ok] / n_on[ok] - (cnt @ s_off)[ok] / n_off[ok]
    return diff, float(np.std(d, ddof=1)), int(k)


def decide(off, on, diff, se, dsr_min):
    """Pre-registered verdict from the two holdout reports."""
    off_pass = bool(off.get('sharpe', 0) > 0 and off.get('dsr', 0) >= dsr_min)
    on_dsr = float(on.get('dsr', 0.0))
    n_eff_on = on.get('n_eff_v2')
    n_eff_on = float(n_eff_on) if n_eff_on is not None else float('nan')
    c1 = bool(np.isfinite(n_eff_on) and n_eff_on >= NEFF_FLOOR)
    c2 = bool(np.isfinite(diff) and np.isfinite(se) and diff >= -se)
    if off_pass and on_dsr < dsr_min:
        verdict = 'ESCALATE'
    elif c1 and c2:
        verdict = 'PROPOSE'
    else:
        verdict = 'HOLD'
    return {'verdict': verdict, 'unmasked_passes': off_pass,
            'masked_dsr_below_min': on_dsr < dsr_min,
            'c1_masked_neff_ge_10': c1, 'c2_edge_within_1se': c2,
            'n_eff_v2_masked': n_eff_on, 'edge_diff': diff, 'edge_se': se}


def run(hs, cfg, state, scaler, data, n_trials, lgb=None, n_boot=2000,
        seed=0):
    """Two real evaluate_on_holdout calls (mask OFF, ON) on one loaded
    panel; returns {'off', 'on', 'decision'}. `data` = load_data's tuple.
    lgb = (booster, q10_booster, q10_floor, lstm_weight) or None."""
    if not hasattr(hs, '_session_entry_ok'):
        raise SystemExit('SIG-R2-MASK is not landed in hypersearch_v2 '
                         '(no _session_entry_ok) — the ON run would equal '
                         'OFF; refusing to report.')
    (all_features, all_returns_by_fb, all_times, _alt, tickers,
     ticker_boundaries, _fc, input_dim, _preset, _mh, tb_bars) = data
    from validation import DSR_MIN
    booster, q10b, q10f, w = lgb if lgb else (None, None, None, None)
    captured = {}
    real_sim = hs.simulate_trades
    real_hidx = hs.get_holdout_indices

    def _hidx(*a, **k):
        captured['hidx_raw'] = real_hidx(*a, **k)
        return captured['hidx_raw']

    def _sim(*a, **k):
        out = real_sim(*a, **k)
        if k.get('return_entries'):
            captured['entries'] = np.asarray(out[1])
        return out

    fb = cfg.get('forward_bars', 24)
    key = ('tb', fb) if cfg.get('target_kind') == 'tb' else fb
    returns = all_returns_by_fb.get(key)
    if returns is None:
        returns = all_returns_by_fb.get(fb)
    out = {}
    prev = os.environ.get('TRADER_OBJECTIVE_SESSION_MASK')
    hs.simulate_trades, hs.get_holdout_indices = _sim, _hidx
    try:
        for tag, val in (('off', '0'), ('on', '1')):
            os.environ['TRADER_OBJECTIVE_SESSION_MASK'] = val
            captured.clear()
            rep = hs.evaluate_on_holdout(
                state, scaler, cfg, all_features, all_returns_by_fb,
                all_times, tickers, ticker_boundaries, input_dim, 'stock',
                n_trials=n_trials, all_tb_bars_by_fb=tb_bars,
                lgb_booster=booster, q10_booster=q10b, q10_floor=q10f,
                lstm_weight=w)
            if rep is None or 'hidx_raw' not in captured:
                raise SystemExit(f'holdout evaluation failed ({tag})')
            h = captured['hidx_raw']
            h = h[~np.isnan(returns[h])]
            ent = captured.get('entries', np.zeros(0, np.int64))
            out[tag] = {'report': rep, 'entry_t': all_times[h[ent]]}
    finally:
        hs.simulate_trades, hs.get_holdout_indices = real_sim, real_hidx
        if prev is None:
            os.environ.pop('TRADER_OBJECTIVE_SESSION_MASK', None)
        else:
            os.environ['TRADER_OBJECTIVE_SESSION_MASK'] = prev
    off, on = out['off']['report'], out['on']['report']
    diff, se, n_weeks = weekly_block_se_diff(
        off['trade_returns'], out['off']['entry_t'],
        on['trade_returns'], out['on']['entry_t'], n_boot=n_boot, seed=seed)
    dec = decide(off, on, diff, se, DSR_MIN)
    dec['n_weeks'] = n_weeks
    return {'off': off, 'on': on, 'decision': dec}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--artifact-prefix', default='stock_',
                    help="artifact file prefix: 'stock_' (champion) or "
                         "'stock_challenger_' (shadow slot)")
    ap.add_argument('--max-rows', type=int, default=200_000,
                    help="MUST equal the training run's --max-rows "
                         "(run_pipeline's stock_search passes 200000)")
    ap.add_argument('--preset', default=None,
                    help="default: the saved config's indicator_preset")
    ap.add_argument('--n-trials', type=int, default=None,
                    help="DSR pool; default: the saved holdout's "
                         "n_trials_pool")
    ap.add_argument('--no-blend', action='store_true',
                    help='score the raw-LSTM certificate even if boosters '
                         'exist')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json', default=None, help='write the result here')
    a = ap.parse_args()

    import joblib
    import torch
    import hypersearch_v2 as hs
    p = a.artifact_prefix
    cfg = joblib.load(f'{p}config_v2.pkl')
    scaler = joblib.load(f'{p}scaler_v2.pkl')
    feature_cols = joblib.load(f'{p}feature_cols_v2.pkl')
    state = torch.load(f'{p}model_v2.pth', map_location='cpu')
    preset = a.preset or cfg.get('indicator_preset')
    data = hs.load_data('stock_training_data.csv', preset_override=preset,
                        max_rows=a.max_rows)
    if list(data[6]) != list(feature_cols):
        raise SystemExit('feature columns of the reloaded panel differ from '
                         'the saved winner — stale artifacts or wrong '
                         '--preset; refusing to score')
    n_trials = a.n_trials or int((cfg.get('holdout') or {})
                                 .get('n_trials_pool') or 2)
    lgb = None
    if not a.no_blend and cfg.get('lstm_weight') is not None \
            and os.path.exists(f'{p}lgb_model.txt'):
        import lightgbm
        booster = lightgbm.Booster(model_file=f'{p}lgb_model.txt')
        q10b = q10f = None
        if os.path.exists(f'{p}lgb_q10.txt') and \
                os.path.exists(f'{p}lgb_q10_meta.json'):
            q10b = lightgbm.Booster(model_file=f'{p}lgb_q10.txt')
            with open(f'{p}lgb_q10_meta.json') as f:
                q10f = float(json.load(f)['floor'])
        lgb = (booster, q10b, q10f, float(cfg['lstm_weight']))
    res = run(hs, cfg, state, scaler, data, n_trials, lgb=lgb,
              n_boot=a.boot, seed=a.seed)
    for tag in ('off', 'on'):
        r = res[tag]
        print(f"[AB] mask {tag.upper():3s}: sharpe={r['sharpe']} "
              f"dsr={r['dsr']} n_trades={r['n_trades']} "
              f"n_eff={r['n_eff']} n_eff_v2={r.get('n_eff_v2')} "
              f"rows_ok={r.get('session_rows_ok', r['n_rows'])}/"
              f"{r['n_rows']}")
    d = res['decision']
    print(f"[AB] edge diff (on-off) = {d['edge_diff']:.4f}% per trade, "
          f"weekly-block SE = {d['edge_se']:.4f} ({d['n_weeks']} weeks)")
    print(f"[AB] VERDICT: {d['verdict']}  {json.dumps(d)}")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == '__main__':
    main()
