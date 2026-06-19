"""Event-driven backtest of the ACTUAL trading policy, net of real costs.

Until now, models were promoted on fit metrics from a simulator that bears
no resemblance to live execution. This module replays the real policy —
threshold + cost-gate entries, ATR stop / trailing / take-profit exits,
EOD flatten for stocks, cooldowns, hard-stop lockouts — bar by bar over
the saved model's predictions, charging venue fees + spread on every fill.

Usage:
    python backtest.py --prefix stock --days 60          # report only
    python backtest.py --prefix stock --days 60 --gate   # restore .prev model on fail

The --gate mode is wired into run_pipeline's weekly retrain: if the
freshly-saved model's policy backtest fails (net Sharpe <= 0 or DSR below
threshold), the previous model artifacts are restored and the retrain is
effectively rejected at the POLICY level, not just the fit level.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from fees import round_trip_cost_pct, required_edge_pct
from strategy_config import policy_for
from validation import dsr_from_trade_returns, DSR_MIN

BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
# Assumed spread haircut applied per round trip (percent), on top of fees
SPREAD_PCT = {'crypto': 0.10, 'stock': 0.05}


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def _load_artifacts(prefix: str):
    import joblib
    import torch
    from model_v2 import RegressionLSTM

    p = f'{prefix}_' if prefix else ''
    config = joblib.load(BASE_DIR / f'{p}config_v2.pkl')
    scaler = joblib.load(BASE_DIR / f'{p}scaler_v2.pkl')
    feature_cols = joblib.load(BASE_DIR / f'{p}feature_cols_v2.pkl')
    model = RegressionLSTM(
        input_dim=config['input_dim'], hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'], dropout=config['dropout'],
        n_heads=config.get('n_heads', 4))
    state = torch.load(BASE_DIR / f'{p}model_v2.pth',
                       map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, config, feature_cols


def _load_lgb(prefix: str):
    try:
        from model_lgb import load_lgb_model
        return load_lgb_model(prefix=prefix)
    except Exception:
        return None


def _load_q10(prefix: str):
    """(booster, veto_floor) for the q10 tail model, or None."""
    try:
        import json as _json
        import lightgbm as _lgb
        p = f'{prefix}_' if prefix else ''
        booster = _lgb.Booster(model_file=f'{p}lgb_q10.txt')
        with open(f'{p}lgb_q10_meta.json') as f:
            floor = float(_json.load(f)['floor'])
        return booster, floor
    except Exception:
        return None


def _predict_ticker(model, scaler, config, feature_cols, tdf, lgb_model=None,
                    q10_model=None):
    """Predictions for every bar of one ticker (CPU, batched).

    Mirrors the LIVE inference path: LSTM prediction, ensembled 0.6/0.4
    with LightGBM when the booster exists (predict_now.get_live_prediction
    does the same — the backtest must validate the policy that trades).
    """
    import torch
    seq_len = config['seq_len']
    feats = tdf[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(feats).astype(np.float32)
    n = len(scaled)
    preds = np.full(n, np.nan, dtype=np.float64)
    q10 = np.full(n, np.nan, dtype=np.float64) if q10_model is not None else None
    if n <= seq_len:
        return preds, q10
    idx = np.arange(seq_len, n)
    offsets = np.arange(-seq_len, 0)
    with torch.inference_mode():
        for i in range(0, len(idx), 1024):
            chunk = idx[i:i + 1024]
            windows = scaled[chunk[:, None] + offsets[None, :]]
            out = model(torch.from_numpy(windows)).numpy().astype(np.float64)
            if lgb_model is not None or q10_model is not None:
                # flatten_sequence ordering == windows.reshape(rows, -1)
                flat = windows.reshape(len(chunk), -1)
                if lgb_model is not None:
                    lgb_out = lgb_model.predict(flat)
                    # Blend weight tunable (wave-9 #2); matches predict_now's live path.
                    _w = config.get('lstm_weight', 0.6)
                    out = _w * out + (1.0 - _w) * np.asarray(lgb_out, dtype=np.float64)
                if q10_model is not None:
                    q10[chunk] = np.asarray(q10_model.predict(flat),
                                            dtype=np.float64)
            preds[chunk] = out
    return preds, q10


# ---------------------------------------------------------------------------
# Policy simulation
# ---------------------------------------------------------------------------

def _entry_window_mask(times) -> np.ndarray:
    """True where the bar's ET time falls in a configured stock entry window."""
    from strategy_config import STOCK_ENTRY_WINDOWS_ET, ENTRY_WINDOWS_ENABLED
    if not ENTRY_WINDOWS_ENABLED:
        return np.ones(len(times), dtype=bool)
    import zoneinfo
    et = zoneinfo.ZoneInfo('US/Eastern')
    windows = []
    for start_s, end_s in STOCK_ENTRY_WINDOWS_ET:
        sh, sm = map(int, start_s.split(':'))
        eh, em = map(int, end_s.split(':'))
        windows.append((sh * 60 + sm, eh * 60 + em))
    mask = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        try:
            local = t.astimezone(et) if t.tzinfo else t
            minutes = local.hour * 60 + local.minute
            mask[i] = any(s <= minutes < e for s, e in windows)
        except Exception:
            mask[i] = True
    return mask

def simulate_ticker(tdf, preds, asset_type: str, threshold: float,
                    policy: dict, meta_probs=None, q10_preds=None,
                    q10_floor=None) -> list[dict]:
    """Replay the live exit stack on one ticker. Returns trade dicts.

    The exit walk itself runs in policy_exits.exit_walk — the SAME kernel
    that generates triple-barrier training labels and the meta-labeling
    dataset, so the backtest cannot drift from label semantics. This
    function keeps only the ENTRY policy: threshold + cost-floor gates,
    stock entry windows, cooldowns/lockouts, the meta-probability veto,
    and the q10 tail veto (each applied when its trained model produced
    inputs, mirroring the live loop).

    Bar-level approximations of the 30s loop: entries at the signal bar's
    close; gap-aware stops (filled at min(stop, open)); stop checked
    before TP when both touch in one bar (conservative).
    """
    from policy_exits import exit_walk, eod_mask_from_index, REASON_NAMES
    from meta_label import META_VETO_PROB

    closes = tdf['Close'].values
    highs = tdf['High'].values if 'High' in tdf.columns else closes
    lows = tdf['Low'].values if 'Low' in tdf.columns else closes
    opens = tdf['Open'].values if 'Open' in tdf.columns else closes
    atr = tdf['ATR'].values if 'ATR' in tdf.columns else np.full(len(closes), np.nan)
    times = tdf.index

    is_eod = eod_mask_from_index(times, asset_type)
    if asset_type == 'stock':
        entry_ok = _entry_window_mask(times)
    else:
        entry_ok = np.ones(len(times), dtype=bool)

    rt_cost = round_trip_cost_pct(asset_type, SPREAD_PCT[asset_type])
    edge_floor = required_edge_pct(asset_type, SPREAD_PCT[asset_type])
    # Per-bar effective-spread cost when harvested (wave 6) — the net P&L is
    # charged the real per-name spread, matching the live gate, instead of the
    # flat SPREAD_PCT. The entry edge_floor stays on the (conservative) flat
    # spread so admission is unchanged; only realized net cost is per-bar.
    rt_cost_arr = None
    if 'Eff_Spread_Pct' in tdf.columns:
        try:
            from liquidity import per_bar_round_trip_cost, impact_inputs_from_df
            # Optional sqrt market-impact haircut (wave-8 #6) — off unless
            # IMPACT_COST_ENABLED and a DV30 column is present, so net P&L is
            # unchanged by default.
            adv_arr, impact_notional, impact_k = impact_inputs_from_df(tdf)
            rt_cost_arr = per_bar_round_trip_cost(
                asset_type, tdf['Eff_Spread_Pct'].values,
                adv_dollar=adv_arr, notional=impact_notional, impact_k=impact_k)
        except Exception:
            rt_cost_arr = None
    cooldown_bars = max(1, int(math.ceil(policy['cooldown_min'] / 60)))
    lockout_bars = int(policy['lockout_hours'])

    exit_idx, exit_px, reason_code = exit_walk(
        closes, highs, lows, opens, atr, is_eod, policy,
        preds=preds, threshold=threshold, cooldown_bars=cooldown_bars,
        max_hold=0, use_signal_exit=True)

    trades = []
    n = len(closes)
    i = 0
    next_entry_allowed = 0
    while i < n - 1:
        p = preds[i]
        if (np.isnan(p) or i < next_entry_allowed
                or p < threshold or p < edge_floor
                or is_eod[i]
                or not entry_ok[i]
                or (meta_probs is not None
                    and not np.isnan(meta_probs[i])
                    and meta_probs[i] < META_VETO_PROB)
                or (q10_preds is not None and q10_floor is not None
                    and not np.isnan(q10_preds[i])
                    and q10_preds[i] < q10_floor)):
            i += 1
            continue

        entry_price = closes[i]
        j = int(exit_idx[i])
        exit_price = float(exit_px[i])
        exit_reason = REASON_NAMES.get(int(reason_code[i]), 'end_of_data')

        gross = (exit_price - entry_price) / entry_price * 100.0
        cost_i = rt_cost if rt_cost_arr is None else float(rt_cost_arr[i])
        net = gross - cost_i
        trades.append({
            'entry_time': str(times[i]), 'exit_time': str(times[j]),
            'entry': float(entry_price), 'exit': exit_price,
            'bars_held': int(j - i), 'gross_pct': round(gross, 4),
            'net_pct': round(net, 4), 'reason': exit_reason,
        })

        cooldown_after = cooldown_bars
        if exit_reason == 'hard_stop':
            cooldown_after = max(cooldown_bars, lockout_bars)
        next_entry_allowed = j + cooldown_after
        i = j + 1

    return trades


def aggregate_metrics(all_trades: list[dict], asset_type: str,
                      span_days: float, n_search_trials: int = 100) -> dict:
    if not all_trades:
        return {'n_trades': 0, 'sharpe': 0.0, 'dsr': 0.0, 'net_total_pct': 0.0,
                'win_rate': 0.0, 'max_drawdown_pct': 0.0, 'avg_hold_bars': 0.0,
                'fees_paid_pct': 0.0}
    rets = np.array([t['net_pct'] for t in all_trades])
    # Exact realized cost = gross - net summed over trades (per-bar spread
    # aware); replaces the flat rt_cost * n_trades estimate.
    gross_total = float(sum(t['gross_pct'] for t in all_trades))
    fees_paid = gross_total - float(rets.sum())

    ordered = sorted(all_trades, key=lambda t: t['exit_time'])
    equity = np.cumsum([t['net_pct'] for t in ordered])
    running_max = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    max_dd = float(np.max(running_max - equity)) if len(equity) else 0.0

    trades_per_year = len(rets) * 365.0 / max(span_days, 1)
    sharpe = 0.0
    if rets.std() > 1e-9:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(max(trades_per_year, 1)))
    dsr = dsr_from_trade_returns(rets, n_trials=max(n_search_trials, 2))

    return {
        'n_trades': len(rets),
        'sharpe': round(sharpe, 3),
        'dsr': round(dsr['dsr'], 4),
        'net_total_pct': round(float(rets.sum()), 2),
        'gross_total_pct': round(gross_total, 2),
        'win_rate': round(float((rets > 0).mean()), 3),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_hold_bars': round(float(np.mean([t['bars_held'] for t in all_trades])), 1),
        'fees_paid_pct': round(fees_paid, 2),
        'trades_per_year': round(trades_per_year, 1),
        'exit_reasons': {r: sum(1 for t in all_trades if t['reason'] == r)
                         for r in {t['reason'] for t in all_trades}},
    }


# ---------------------------------------------------------------------------
# Model restore (gate failure path)
# ---------------------------------------------------------------------------

ARTIFACT_SUFFIXES = ['model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl',
                     'feature_cols_v2.pkl', 'model_v2.manifest.json',
                     'lgb_model.txt', 'meta_model.txt', 'meta_calib.pkl',
                     'meta_meta.json', 'lgb_q10.txt', 'lgb_q10_meta.json']


def restore_previous_model(prefix: str) -> bool:
    """Roll back to the .prev artifacts saved before the last promotion."""
    p = f'{prefix}_' if prefix else ''
    prevs = [(BASE_DIR / f'{p}{s}.prev', BASE_DIR / f'{p}{s}')
             for s in ARTIFACT_SUFFIXES]
    if not all(src.exists() for src, _ in prevs[:4]):  # manifest optional
        print("[GATE] No .prev artifacts to restore — keeping current model")
        return False
    for src, dst in prevs:
        if src.exists():
            os.replace(src, dst)
    print("[GATE] Restored previous model artifacts")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(prefix: str = '', days: int = 60,
                 n_search_trials: int = 100) -> dict:
    from data_utils import load_training_data

    asset_type = prefix or 'crypto'
    model, scaler, config, feature_cols = _load_artifacts(prefix)
    lgb_model = _load_lgb(prefix)
    q10_pack = _load_q10(prefix)
    q10_model, q10_floor = q10_pack if q10_pack else (None, None)
    threshold = config.get('trade_threshold', 0.15)
    policy = policy_for(asset_type)

    df = load_training_data('stock' if prefix == 'stock' else 'crypto')
    if df.empty:
        raise SystemExit("No training data found for backtest")
    cutoff = df.index.max() - timedelta(days=days)
    df = df[df.index >= cutoff]

    all_trades = []
    tickers = df['Ticker'].unique()
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        if len(tdf) < config['seq_len'] + 10:
            continue
        missing = [c for c in feature_cols if c not in tdf.columns]
        if missing:
            print(f"  [SKIP] {ticker}: missing features {missing[:3]}...")
            continue
        preds, q10_preds = _predict_ticker(model, scaler, config,
                                           feature_cols, tdf,
                                           lgb_model=lgb_model,
                                           q10_model=q10_model)
        # Meta-labeling parity: the live loops veto entries with low
        # calibrated meta probability, so the gate must too
        meta_probs = None
        try:
            from meta_label import predict_meta_array
            meta_probs = predict_meta_array(prefix, tdf, preds)
        except Exception:
            pass
        trades = simulate_ticker(tdf, preds, asset_type, threshold, policy,
                                 meta_probs=meta_probs, q10_preds=q10_preds,
                                 q10_floor=q10_floor)
        all_trades.extend(trades)
        print(f"  {ticker}: {len(trades)} trades")

    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400
    metrics = aggregate_metrics(all_trades, asset_type, span_days,
                                n_search_trials)
    metrics['period'] = f"{df.index.min()} .. {df.index.max()}"
    metrics['threshold'] = threshold
    metrics['prefix'] = prefix
    metrics['generated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')

    report_path = BASE_DIR / f"backtest_{f'{prefix}_' if prefix else ''}report.json"
    tmp = str(report_path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'metrics': metrics, 'trades': all_trades[-500:]}, f, indent=2)
    os.replace(tmp, report_path)

    print(f"\n=== POLICY BACKTEST ({asset_type}, last {days}d) ===")
    for k, v in metrics.items():
        if k != 'exit_reasons':
            print(f"  {k}: {v}")
    print(f"  exits: {metrics.get('exit_reasons')}")
    print(f"  report: {report_path}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Backtest the actual trading policy')
    ap.add_argument('--prefix', default='', help="'' for crypto, 'stock' for stocks")
    ap.add_argument('--days', type=int, default=60, help='lookback window (days)')
    ap.add_argument('--trials', type=int, default=100,
                    help='search-pool size for DSR deflation')
    ap.add_argument('--gate', action='store_true',
                    help='restore the previous model if the backtest fails')
    ap.add_argument('--min-sharpe', type=float, default=0.0)
    ap.add_argument('--min-dsr', type=float, default=0.5)
    args = ap.parse_args()

    try:
        metrics = run_backtest(args.prefix, args.days, args.trials)
    except FileNotFoundError as e:
        print(f"[BACKTEST] Missing artifact ({e}) — nothing to gate")
        return 0

    if args.gate:
        ok = (metrics['n_trades'] >= 10
              and metrics['sharpe'] > args.min_sharpe
              and metrics['dsr'] >= args.min_dsr)
        if not ok:
            print(f"\n[GATE] FAILED: n={metrics['n_trades']}, "
                  f"sharpe={metrics['sharpe']} (min {args.min_sharpe}), "
                  f"dsr={metrics['dsr']} (min {args.min_dsr})")
            restored = restore_previous_model(args.prefix)
            try:
                from notify import notify
                notify(f"Backtest gate FAILED for {args.prefix or 'crypto'} "
                       f"(sharpe={metrics['sharpe']}, dsr={metrics['dsr']}) — "
                       f"{'previous model restored' if restored else 'no rollback available'}",
                       level='warning', dedupe_key=f'gate-{args.prefix}')
            except Exception:
                pass
        else:
            print(f"\n[GATE] PASSED: sharpe={metrics['sharpe']}, dsr={metrics['dsr']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
