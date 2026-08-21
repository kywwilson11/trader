"""Event-driven backtest of the ACTUAL trading policy, net of real costs.

Until now, models were promoted on fit metrics from a simulator that bears
no resemblance to live execution. This module replays the real policy —
threshold + cost-gate entries, ATR stop / trailing / take-profit exits,
EOD flatten for stocks, cooldowns, hard-stop lockouts — bar by bar over
the saved model's predictions, charging venue fees + spread on every fill.

Usage:
    python backtest.py --prefix stock --days 60          # report only
    python backtest.py --prefix stock --days 60 --gate   # restore .prev model on fail
    python backtest.py --prefix stock --days 60 --gate --model-prefix stock_challenger  # gate the challenger slot

The --gate mode is wired into run_pipeline's weekly retrain: if the
freshly-saved model's policy backtest fails (net Sharpe <= 0 or DSR below
threshold), the previous model artifacts are restored and the retrain is
effectively rejected at the POLICY level, not just the fit level.

NOTE (2026-07 adjudication): under run_pipeline's weekly retrain the
search runs in SHADOW mode by default, so hypersearch saves the fresh
model into the challenger slot while this gate is invoked on the champion
prefix — the gate then re-validates the deployed CHAMPION, not the model
just trained, and a failure rolls the champion back. Re-pointing the gate
at the challenger now exists behind strategy_config.GATE_TARGETS_CHALLENGER
(default OFF) — when run_pipeline passes --model-prefix <challenger slot>,
this module scores challenger artifacts on champion book data, a failure
HOLDS the challenger (no champion rollback), and the verdict is persisted
to {slot}_policy_gate.json for the shadow-side promotion pre-flight; the
report's artifact_manifest_saved_at field and the 'NEWER challenger
manifest' warning remain the flag-OFF breadcrumb.

Exit codes (main()): 0 = gate passed, or nothing to gate (no --gate flag,
or a FileNotFoundError before any model was evaluated); 3 = --gate FAILURE
— a deterministic policy rejection, the model was rolled back to .prev;
any other nonzero code means the process crashed before reaching a verdict.
run_pipeline treats 3 specially: it is a final, non-retryable outcome for
the *_backtest_gate phase (retrying a deterministic rejection is useless),
and it does not abort the rest of that retrain run since the model has
already been safely rolled back. 2 = bad CLI arguments (argparse, e.g.
--days < 1). A SystemExit raised for a missing/stale data window
propagates uncaught (nonzero; run_pipeline retries it as a crash). A
FileNotFoundError raised while all four core artifacts ARE present on
disk re-raises as a crash instead of reporting 'nothing to gate'. With
--model-prefix targeting the challenger slot, exit 3 still means
deterministic policy rejection, but the action is HOLD-challenger
(nothing on disk is rolled back).

Coverage caveat: this replay evaluates the threshold/cost-floor entry
gate, the ATR exit stack (hard stop / trailing / take-profit / signal /
EOD / vertical), stock entry windows, cooldowns/lockouts, the meta-label
veto, and the q10 tail veto — the same gates the live loop enforces before
sizing a trade. It does NOT model the live-only rejectors: winner's-curse
derating, cross-asset correlation caps, macro/VIX halts, sentiment/LLM
vetoes, or effective-threshold escalation under drawdown. Those layers
only ever make live trading MORE conservative than this replay, so the
replay is a permissive superset admission surface by design — extending
it to model those rejectors too is a deliberate policy decision for a
future change, not an oversight in this one.
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
# Attribute reads at call time (never `from ... import PROMOTION_GATE_V2`)
# so tests can monkeypatch the flag on the module.
import strategy_config as _strategy_config
from strategy_config import policy_for
from validation import dsr_from_trade_returns, DSR_MIN

# NOT used by this module's Sharpe — aggregate_metrics annualizes from the
# pooled calendar span (trades/year), never from bar counts. Kept only as
# the third copy of the cross-module bars-per-year constant pinned equal to
# volatility.py and scripts/hypersearch_v2.py by tests/test_review_b17.py
# (consolidation into strategy_config deferred — see the b17 ledger).
BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}
# Assumed spread haircut applied per round trip (percent), on top of fees
SPREAD_PCT = {'crypto': 0.10, 'stock': 0.05}

# Stage-0 predictions dump + hourly MTM equity (B02, measurement-only).
# Default ON: additive report/metrics keys only, never touches admission,
# exits, Sharpe/DSR, or the gate verdict. CLI --no-stage0-dump flips this
# module global (main() must not widen its pinned 3-arg run_backtest calls).
STAGE0_DUMP_DEFAULT = True


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
    """LightGBM booster leg, or None. Mirrors meta_label._load: a missing
    file is expected (silent None) but a PRESENT, unloadable file is a
    real problem — logged loudly so a corrupt artifact isn't mistaken for
    "no LightGBM leg trained" (the backtest would otherwise silently
    evaluate the LSTM-only blend against a policy tuned for the ensemble).
    """
    p = f'{prefix}_' if prefix else ''
    path = BASE_DIR / f'{p}lgb_model.txt'
    try:
        from model_lgb import load_lgb_model
        return load_lgb_model(prefix=prefix)
    except Exception as e:
        if path.exists():
            print(f"[GATE] {path.name} present but failed to load ({e})"
                  f" — evaluating WITHOUT this leg")
        return None


def _load_q10(prefix: str):
    """(booster, veto_floor) for the q10 tail model, or None. Same
    present-but-unloadable distinction as _load_lgb."""
    p = f'{prefix}_' if prefix else ''
    path = BASE_DIR / f'{p}lgb_q10.txt'
    try:
        import json as _json
        import lightgbm as _lgb
        booster = _lgb.Booster(model_file=str(path))
        with open(BASE_DIR / f'{p}lgb_q10_meta.json') as f:
            floor = float(_json.load(f)['floor'])
        if not math.isfinite(floor):
            raise ValueError(f'non-finite q10 floor {floor!r}')
        return booster, floor
    except Exception as e:
        if path.exists():
            print(f"[GATE] {path.name} present but failed to load ({e})"
                  f" — evaluating WITHOUT this leg")
        return None


# Peak-memory knob for the batched inference below: the window gather is
# chunk_rows x seq_len x n_features float32 and LightGBM adds its own
# float64 copy on predict. A bare 1024-row chunk scales with the winning
# config's seq_len and feature count — byte-budget it instead (identical
# outputs at any chunk size; this only bounds the transient allocation on
# the 8 GB Jetson).
PRED_X_BYTE_BUDGET = 64_000_000

def _pred_chunk_rows(seq_len: int, n_features: int) -> int:
    """Rows per inference chunk: <= 1024, >= 64, byte-budgeted."""
    row_bytes = max(int(seq_len) * int(n_features) * 4, 1)
    return int(min(1024, max(64, PRED_X_BYTE_BUDGET // row_bytes)))


def _predict_ticker(model, scaler, config, feature_cols, tdf, lgb_model=None,
                    q10_model=None, legs_out=None):
    """Predictions for every bar of one ticker (CPU, batched).

    Mirrors the LIVE inference path: LSTM prediction, ensembled 0.6/0.4
    with LightGBM when the booster exists (predict_now.get_live_prediction
    does the same — the backtest must validate the policy that trades).

    legs_out (optional dict) is filled with per-bar 'lstm'/'lgb' leg arrays
    (NaN where unavailable); the (preds, q10) return shape is pinned by
    tests/test_review_b14 and must not change.
    """
    import torch
    seq_len = config['seq_len']
    feats = tdf[feature_cols].values.astype(np.float32)
    scaled = scaler.transform(feats).astype(np.float32)
    n = len(scaled)
    preds = np.full(n, np.nan, dtype=np.float64)
    q10 = np.full(n, np.nan, dtype=np.float64) if q10_model is not None else None
    lstm_leg = lgb_leg = None
    if legs_out is not None:
        lstm_leg = np.full(n, np.nan, dtype=np.float64)
        lgb_leg = np.full(n, np.nan, dtype=np.float64)
        legs_out['lstm'] = lstm_leg
        legs_out['lgb'] = lgb_leg
    if n <= seq_len:
        return preds, q10
    idx = np.arange(seq_len, n)
    offsets = np.arange(-seq_len, 0)
    chunk_rows = _pred_chunk_rows(seq_len, scaled.shape[1])
    with torch.inference_mode():
        for i in range(0, len(idx), chunk_rows):
            chunk = idx[i:i + chunk_rows]
            windows = scaled[chunk[:, None] + offsets[None, :]]
            out = model(torch.from_numpy(windows)).numpy().astype(np.float64)
            if lstm_leg is not None:
                lstm_leg[chunk] = out
            if lgb_model is not None or q10_model is not None:
                # flatten_sequence ordering == windows.reshape(rows, -1)
                flat = windows.reshape(len(chunk), -1)
                if lgb_model is not None:
                    lgb_out = lgb_model.predict(flat)
                    if lgb_leg is not None:
                        lgb_leg[chunk] = np.asarray(lgb_out, dtype=np.float64)
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

_TZ_NAIVE_WARNED = False  # reset per run by run_backtest


def _entry_window_mask(times) -> np.ndarray:
    """True where the bar's ET time falls in a configured stock entry window."""
    global _TZ_NAIVE_WARNED
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
    if getattr(times, 'tz', 'n/a') is None and not _TZ_NAIVE_WARNED:
        print("[GATE] WARNING: stock index is tz-NAIVE — entry-window "
              "minutes are being read as raw clock hours, not converted "
              "from UTC to ET; if the harvest stamps UTC every window is "
              "shifted 4-5h")
        _TZ_NAIVE_WARNED = True
    mask = np.zeros(len(times), dtype=bool)
    n_failed = 0
    for i, t in enumerate(times):
        try:
            local = t.astimezone(et) if t.tzinfo else t
            minutes = local.hour * 60 + local.minute
            mask[i] = any(s <= minutes < e for s, e in windows)
        except Exception:
            mask[i] = True
            n_failed += 1
    if n_failed:
        print(f"[GATE] WARNING: {n_failed} bars failed entry-window "
              f"evaluation and were admitted fail-open")
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
    before TP when both touch in one bar (conservative); the hard-stop
    lockout is applied in BARS (int(lockout_hours)) while live enforces
    wall-clock hours — equal for 24/7 crypto, ~3.4x longer than live for
    stocks at ~7 RTH bars/session (a units fix changes admission = owner
    decision); a position still open at the last bar is closed there with
    reason 'end_of_data'.
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
    # flat SPREAD_PCT. The entry edge_floor stays on the FLAT spread —
    # permissive for names whose true spread exceeds SPREAD_PCT, strict for
    # tighter ones (live admission uses the real quote via
    # order_utils.should_trade); switching it is an owner decision. Only
    # realized net cost is per-bar.
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
        except Exception as e:
            print(f"[GATE] per-bar spread cost failed ({e}) — flat-cost fallback")
            rt_cost_arr = None
    cooldown_bars = max(1, int(math.ceil(policy['cooldown_min'] / 60)))
    lockout_bars = int(policy['lockout_hours'])

    exit_idx, exit_px, reason_code = exit_walk(
        closes, highs, lows, opens, atr, is_eod, policy,
        preds=preds, threshold=threshold, cooldown_bars=cooldown_bars,
        max_hold=0, use_signal_exit=True)

    trades = []
    tkr = str(tdf['Ticker'].iloc[0]) if 'Ticker' in tdf.columns else ''
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
            'ticker': tkr,
        })

        cooldown_after = cooldown_bars
        if exit_reason == 'hard_stop':
            cooldown_after = max(cooldown_bars, lockout_bars)
        next_entry_allowed = j + cooldown_after
        i = j + 1

    return trades


def aggregate_metrics(all_trades: list[dict], asset_type: str,
                      span_days: float, n_search_trials: int = 100) -> dict:
    """Pool per-trade returns across every name into gate metrics.

    Conventions: `sharpe` is the POOLED per-trade Sharpe annualized by
    sqrt(trades/year) on a 365-day calendar — pooling K concurrently
    trading names inflates it ~sqrt(K) vs one name; `sharpe_raw`/`dsr_raw`
    are the unrounded values (the gate compares the ROUNDED ones — a
    recorded owner-decision item); `dsr_sr` is the per-trade Sharpe the
    DSR judged. `max_drawdown_pct` is an equal-weight, exit-ordered
    cumulative SUM of per-trade percents — a trade-sequence statistic,
    not a sized portfolio drawdown. A position still open at the window
    end is closed at the final bar (reason 'end_of_data');
    `censored_trade_frac` reports that share. `n_eff_clustered` is the
    PRE-clamp cluster count (or n_trades when no correction applied);
    `dsr_n_eff_used` is the POST-clamp value the DSR consumed (validation
    floors it at 10); `n_eff_source` names the branch; `dsr_iid` is the
    same DSR under the IID null — dsr_iid high with dsr low means the
    clustering, not the model, decided the verdict. `asset_type` selects
    the KISH_RHO_FLOOR entry when the Kish softening is enabled under
    PROMOTION_GATE_V2 (otherwise unused). Under PROMOTION_GATE_V2 the DSR
    consumes the calendar-concurrency average-uniqueness n_eff
    (`n_eff_calendar`, the ONE non-IID correction, fail-closed below 10
    effective trades); `n_eff_clustered` is then reporting-only.
    """
    gate_v2 = bool(getattr(_strategy_config, 'PROMOTION_GATE_V2', False))
    kish = bool(getattr(_strategy_config, 'KISH_NEFF_ENABLED', False))
    floors = getattr(_strategy_config, 'KISH_RHO_FLOOR', {})
    if not all_trades:
        return {
            'n_trades': 0, 'n_eff_clustered': 0, 'n_eff_source': 'no_trades',
            'n_eff_calendar': 0.0, 'gate_v2_active': gate_v2,
            'sharpe': 0.0, 'sharpe_raw': 0.0,
            'dsr': 0.0, 'dsr_raw': 0.0, 'dsr_iid': 0.0,
            'dsr_sr': 0.0, 'dsr_expected_max_sr': 0.0, 'dsr_n_eff_used': 0.0,
            'dsr_n_trials': 0, 'dsr_n_dropped': 0,
            'dsr_sr_std_null': None, 'dsr_skew': None, 'dsr_kurt': None,
            'dsr_status': 'no_trades', 'dsr_min_trl': None,
            'n_nonfinite_trades': 0,
            'net_total_pct': 0.0, 'gross_total_pct': 0.0, 'win_rate': 0.0,
            'max_drawdown_pct': 0.0, 'avg_hold_bars': 0.0,
            'fees_paid_pct': 0.0, 'trades_per_year': 0.0,
            'span_days': round(float(span_days), 2),
            'exit_reasons': {}, 'censored_trade_frac': 0.0,
        }
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

    n_nonfinite = int((~np.isfinite(rets)).sum())
    if n_nonfinite:
        print(f"[GATE] WARNING: {n_nonfinite}/{len(rets)} trades have "
              f"non-finite net_pct — rets.std() is NaN, sharpe is reported "
              f"as 0.0 and a --gate run will fail on it")

    # Cross-sectional effective-n (2026-07 review): the replay pools trades
    # across every name in the book, but same-hour trades on correlated names
    # are not independent draws. Cluster overlapping [entry, exit] calendar
    # intervals across names (rho=1 worst case) and hand the DSR the cluster
    # count as its effective breadth — never loosens the gate, only widens
    # the null when trades crowd the same hours. Falls back to IID on any
    # parse failure.
    n_eff = None
    n_eff_status = 'iid_unavailable'
    cal = None
    try:
        import pandas as pd
        from sample_weights import clustered_effective_n, calendar_effective_n
        ets = pd.to_datetime([t['entry_time'] for t in all_trades]).values
        xts = pd.to_datetime([t['exit_time'] for t in all_trades]).values
        n_clusters = clustered_effective_n(ets, xts)
        if 0 < n_clusters < len(rets):
            n_eff = float(n_clusters)
            n_eff_status = 'clustered'
        elif n_clusters == 0:
            n_eff_status = 'iid_degenerate'
        else:
            n_eff_status = 'iid_no_overlap'
        # v2 calendar-concurrency estimator: side-by-side instrumentation
        # always; the DSR input under PROMOTION_GATE_V2.
        try:
            rho = None
            if gate_v2 and kish:
                rho = floors.get(asset_type)
            cal = calendar_effective_n(ets, xts, rho_bar=rho)
            print(f"[GATE] n_eff legacy clustered={n_clusters} vs v2 "
                  f"calendar={cal['n_eff']:.1f} (max_concurrency="
                  f"{cal['max_concurrency']}, n_trades={len(rets)})")
        except Exception as ce:
            print(f"[GATE] calendar n_eff unavailable ({ce})")
            cal = None
    except Exception as e:
        print(f"[GATE] cross-sectional n_eff unavailable ({e}) — "
              f"falling back to the IID null")
        n_eff = None
        n_eff_status = 'iid_unavailable'

    if n_eff is not None and n_eff < len(rets) / 5:
        print(f"[GATE] WARNING: cross-sectional clustering collapsed "
              f"{len(rets)} trades to {int(n_eff)} clusters "
              f"({n_eff / len(rets):.1%}) — the DSR null is being set by "
              f"trade crowding, not by the model")

    if gate_v2 and cal is not None:
        # PROMOTION_GATE_V2: the calendar-concurrency n_eff is the ONE
        # non-IID correction (supersedes uniqueness AND cluster count;
        # never stacked with Lo-2002 — gotcha #4). Fails closed below 10
        # effective trades. n_clusters stays reported for comparison only.
        n_eff_status = 'calendar_uniqueness'
        dsr = dsr_from_trade_returns(rets, n_trials=max(n_search_trials, 2),
                                     n_eff=float(cal['n_eff']),
                                     n_eff_source='calendar_uniqueness',
                                     fail_closed_floor=True)
        dsr_iid = dsr_from_trade_returns(rets,
                                         n_trials=max(n_search_trials, 2),
                                         n_eff=None)
    elif gate_v2:
        # v2 with the calendar estimator unavailable: mirror hypersearch's
        # degraded path — IID null WITH the closed floor, never the legacy
        # clustered n_eff + silent 10-sample rescue (D02) while the report
        # claims gate_v2_active.
        n_eff_status = 'calendar_unavailable_iid'
        dsr = dsr_from_trade_returns(rets, n_trials=max(n_search_trials, 2),
                                     n_eff=None, fail_closed_floor=True)
        dsr_iid = dsr
    else:
        dsr = dsr_from_trade_returns(rets, n_trials=max(n_search_trials, 2),
                                     n_eff=n_eff,
                                     n_eff_source=('clustered'
                                                   if n_eff is not None
                                                   else None))
        dsr_iid = (dsr if n_eff is None else
                   dsr_from_trade_returns(rets,
                                          n_trials=max(n_search_trials, 2),
                                          n_eff=None))

    exit_reasons = {r: sum(1 for t in all_trades if t['reason'] == r)
                     for r in {t['reason'] for t in all_trades}}

    return {
        'n_trades': len(rets),
        'n_eff_clustered': (int(n_eff) if n_eff is not None else len(rets)),
        'n_eff_source': n_eff_status,
        'n_eff_calendar': (float(cal['n_eff']) if cal is not None else None),
        'gate_v2_active': gate_v2,
        'sharpe': round(sharpe, 3),
        'sharpe_raw': float(sharpe),
        'dsr': round(dsr['dsr'], 4),
        'dsr_raw': float(dsr['dsr']),
        'dsr_iid': round(float(dsr_iid['dsr']), 4),
        'dsr_sr': float(dsr['sr']),
        'dsr_expected_max_sr': float(dsr['expected_max_sr']),
        'dsr_n_eff_used': float(dsr['n_eff']),
        'dsr_n_trials': int(dsr['n_trials']),
        'dsr_n_dropped': int(dsr['n_dropped']),
        'dsr_sr_std_null': dsr['sr_std_null'],
        'dsr_skew': dsr['skew'],
        'dsr_kurt': dsr['kurt'],
        'dsr_status': dsr.get('status'),
        'dsr_min_trl': dsr.get('min_trl'),
        'n_nonfinite_trades': n_nonfinite,
        'net_total_pct': round(float(rets.sum()), 2),
        'gross_total_pct': round(gross_total, 2),
        'win_rate': round(float((rets > 0).mean()), 3),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_hold_bars': round(float(np.mean([t['bars_held'] for t in all_trades])), 1),
        'fees_paid_pct': round(fees_paid, 2),
        'trades_per_year': round(trades_per_year, 1),
        'span_days': round(float(span_days), 2),
        'censored_trade_frac': round(
            exit_reasons.get('end_of_data', 0) / max(len(rets), 1), 4),
        'exit_reasons': exit_reasons,
    }


# ---------------------------------------------------------------------------
# Model restore (gate failure path)
# ---------------------------------------------------------------------------

ARTIFACT_SUFFIXES = ['model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl',
                     'feature_cols_v2.pkl', 'model_v2.manifest.json',
                     'lgb_model.txt', 'meta_model.txt', 'meta_calib.pkl',
                     'meta_meta.json', 'lgb_q10.txt', 'lgb_q10_meta.json',
                     # OOF predictions (D12): optional leg — restored from
                     # .prev on rollback, or deleted as a never-gated orphan
                     # (a fingerprint-stale npz would only fail-soft anyway).
                     'oof_preds.npz']


def _resolve_model_slot(data_prefix: str, requested: str,
                        challenger_core_present: bool) -> tuple:
    """Which artifact slot the gate scores (pure decision, D03).

    Returns (slot, reason). reason in {'legacy', 'challenger',
    'fallback_champion'}. An empty/equal request is the legacy path
    (slot == book == data_prefix, byte-identical behavior). A distinct
    requested slot is honored only when its 4 core artifacts exist;
    otherwise fall back to the champion slot — on a first-ever deploy
    (no champion yet) hypersearch --shadow saves to the CHAMPION slot,
    and silently 'gating' an empty challenger slot would return
    'nothing to gate' (exit 0) and let the fresh champion deploy
    ungated.
    """
    if not requested or requested == data_prefix:
        return data_prefix, 'legacy'
    if challenger_core_present:
        return requested, 'challenger'
    return data_prefix, 'fallback_champion'


def _report_slot(data_prefix: str, model_prefix: str) -> str:
    """Report-file identity: challenger-targeted runs write
    backtest_<slot>_report.json so they never clobber the champion
    book report the GUI reads."""
    if model_prefix and model_prefix != data_prefix:
        return model_prefix
    return data_prefix


def restore_previous_model(prefix: str) -> bool:
    """Roll back to the .prev artifacts saved before the last promotion.

    The first 4 suffixes (LSTM leg) are required — restore is a no-op
    unless all 4 have a .prev. Suffixes from index 4 on (manifest, LGB,
    meta, q10) are optional legs: restored from .prev when one exists;
    when a leg's .prev is MISSING but its current file EXISTS, that leg
    was never gated by a prior promotion (e.g. LightGBM added after the
    last promoted LSTM) — it is deleted so the post-restore disk state is
    always exactly one previously-gated artifact set, never a stale mix
    of old-LSTM + new-never-gated legs.

    On-disk guarantee only: a RUNNING bot re-reads the LGB/q10 boosters
    when its manifest-keyed hot reload fires (base_loop pops
    predict_now._lgb_models/_q10_models) — which is why the manifest is
    restored LAST here, mirroring save_model_atomically.
    """
    p = f'{prefix}_' if prefix else ''
    prevs = [(BASE_DIR / f'{p}{s}.prev', BASE_DIR / f'{p}{s}')
             for s in ARTIFACT_SUFFIXES]
    if not all(src.exists() for src, _ in prevs[:4]):  # manifest optional
        print("[GATE] No .prev artifacts to restore — keeping current model")
        return False
    # Manifest (index 4) is deliberately processed LAST: it is the bots'
    # hot-reload key (save_model_atomically writes it last for the same
    # reason), so every other leg must already be on disk when it flips.
    order = [i for i in range(len(prevs)) if i != 4] + [4]
    restored_names = []
    n_planned = sum(1 for src, _ in prevs if src.exists())
    try:
        for i in order:
            src, dst = prevs[i]
            if src.exists():
                os.replace(src, dst)
                restored_names.append(dst.name)
            elif i >= 4 and dst.exists():
                os.remove(dst)
                print(f"[GATE] removed never-gated orphan {dst.name}")
    except OSError as e:
        print(f"[GATE] CRITICAL: rollback FAILED partway ({e}) — "
              f"{len(restored_names)}/{n_planned} artifacts restored "
              f"({restored_names}); the {prefix or 'crypto'} artifact set "
              f"on disk is now a MIX of old and new and its .prev files "
              f"are partially consumed — manual repair required")
        try:
            from notify import notify
            notify(f"Backtest gate rollback FAILED partway for "
                   f"{prefix or 'crypto'} — artifact set is MIXED on disk",
                   level='critical', dedupe_key=f'rollback-corrupt-{prefix}')
        except Exception:
            pass
        raise
    print("[GATE] Restored previous model artifacts")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backtest(prefix: str = '', days: int = 60,
                 n_search_trials: int = 100, *,
                 model_prefix: str | None = None,
                 stage0_dump: bool | None = None) -> dict:
    global _TZ_NAIVE_WARNED
    _TZ_NAIVE_WARNED = False
    from data_utils import load_training_data

    model_prefix = prefix if model_prefix is None else model_prefix
    challenger_run = (model_prefix != prefix)
    if challenger_run:
        print(f"[GATE] scoring MODEL slot '{model_prefix}' on "
              f"{prefix or 'crypto'} book data (challenger policy gate, D03)")
    asset_type = prefix or 'crypto'
    model, scaler, config, feature_cols = _load_artifacts(model_prefix)
    lgb_model = _load_lgb(model_prefix)
    q10_pack = _load_q10(model_prefix)
    q10_model, q10_floor = q10_pack if q10_pack else (None, None)
    threshold = config.get('trade_threshold', 0.15)
    policy = policy_for(asset_type)

    df = load_training_data('stock' if prefix == 'stock' else 'crypto')
    if df.empty:
        raise SystemExit("No training data found for backtest")
    cutoff = df.index.max() - timedelta(days=days)
    df = df[df.index >= cutoff]
    if df.empty:
        # Re-checked AFTER the cutoff filter: a nonempty file whose most
        # recent bar is still older than `days` ago (stale harvest, a
        # thinly-covered ticker-only slice) used to fall through to a
        # 'NaT .. NaT' report with n_trades=0 — a silent, misleading pass
        # surface. Fail loud instead so the gate treats it as an error.
        raise SystemExit(f"No training data in the last {days}d window")

    all_trades = []
    tickers = df['Ticker'].unique()
    do_dump = (bool(STAGE0_DUMP_DEFAULT) if stage0_dump is None
               else bool(stage0_dump))
    s0_rows: list[dict] = []
    price_series: dict[str, tuple] = {}
    grid_ns = anchor_ns = None
    horizon = max(1, int(config.get('forward_bars', 4) or 4))
    _s0 = None
    if do_dump:
        try:
            import stage0_preds as _s0
            grid_ns = np.unique(_s0.index_ns(df.index))
            anchor_ns = _s0.global_anchor_ns(grid_ns, horizon)
        except Exception as e:
            print(f"[GATE] stage0 dump disabled ({e}) — measurement only, "
                  f"gate unaffected")
            _s0 = None
    n_evaluated = 0
    skipped_tickers = []
    n_skipped_short_history = 0
    n_skipped_missing_features = 0
    meta_warned = False
    meta_active = False
    if challenger_run:
        print("[GATE] challenger slot has no meta artifacts (trained "
              "post-promotion) — replaying WITHOUT the meta gate, matching "
              "immediate post-promotion live behavior")
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        if len(tdf) < config['seq_len'] + 10:
            skipped_tickers.append(ticker)
            n_skipped_short_history += 1
            continue
        missing = [c for c in feature_cols if c not in tdf.columns]
        if missing:
            print(f"  [SKIP] {ticker}: missing features {missing[:3]}...")
            skipped_tickers.append(ticker)
            n_skipped_missing_features += 1
            continue
        legs = None
        _pt_kw = {}
        if _s0 is not None:
            try:
                import inspect
                if 'legs_out' in inspect.signature(_predict_ticker).parameters:
                    legs = {}
                    _pt_kw['legs_out'] = legs
            except (TypeError, ValueError):
                legs = None
        preds, q10_preds = _predict_ticker(model, scaler, config,
                                           feature_cols, tdf,
                                           lgb_model=lgb_model,
                                           q10_model=q10_model, **_pt_kw)
        # Meta-labeling parity: the live loops veto entries with low
        # calibrated meta probability, so the gate must too
        meta_probs = None
        try:
            from meta_label import predict_meta_array
            meta_probs = predict_meta_array(model_prefix, tdf, preds)
            if meta_probs is not None:
                meta_active = True
        except Exception as e:
            if not meta_warned:
                print(f"[GATE] meta veto unavailable ({e}) — replaying "
                      f"WITHOUT the meta gate (more permissive than live)")
                meta_warned = True
        trades = simulate_ticker(tdf, preds, asset_type, threshold, policy,
                                 meta_probs=meta_probs, q10_preds=q10_preds,
                                 q10_floor=q10_floor)
        all_trades.extend(trades)
        n_evaluated += 1
        if _s0 is not None:
            try:
                t_ns = _s0.index_ns(tdf.index)
                idx_sel = _s0.select_row_indices(t_ns, preds, horizon,
                                                 anchor_ns=anchor_ns)
                s0_rows.extend(_s0.build_rows(
                    tdf.index, str(ticker), preds,
                    tdf['Close'].to_numpy(dtype=float), horizon, idx_sel,
                    lstm=(legs or {}).get('lstm'),
                    lgb=(legs or {}).get('lgb'),
                    meta_probs=meta_probs, q10=q10_preds,
                    threshold=threshold))
                price_series[str(ticker)] = (
                    t_ns, tdf['Close'].to_numpy(dtype=float))
            except Exception as e:
                print(f"[GATE] stage0 rows failed for {ticker} ({e})")
        print(f"  {ticker}: {len(trades)} trades")

    span_days = (df.index.max() - df.index.min()).total_seconds() / 86400
    metrics = aggregate_metrics(all_trades, asset_type, span_days,
                                n_search_trials)
    metrics['period'] = f"{df.index.min()} .. {df.index.max()}"
    metrics['threshold'] = threshold
    metrics['prefix'] = prefix
    if challenger_run:
        metrics['model_prefix'] = model_prefix
        metrics['gate_target'] = 'challenger'
    metrics['generated_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    # Coverage self-description (2026-07 review): a replay that silently
    # skipped most of the universe (missing features, too-short history)
    # can still report a passing Sharpe/DSR on the handful of names left —
    # surface the coverage so a passing gate on 2/40 names is visible.
    n_skipped = len(skipped_tickers)
    metrics['n_tickers_evaluated'] = n_evaluated
    metrics['n_tickers_skipped'] = n_skipped
    metrics['skipped_tickers'] = list(skipped_tickers[:20])
    metrics['meta_veto_active'] = meta_active
    metrics['n_skipped_short_history'] = n_skipped_short_history
    metrics['n_skipped_missing_features'] = n_skipped_missing_features
    metrics['coverage_frac'] = round(
        n_evaluated / max(n_evaluated + n_skipped, 1), 3)
    if n_skipped > n_evaluated:
        print(f"[GATE] WARNING: replay covered only "
              f"{n_evaluated}/{n_evaluated + n_skipped} names")

    try:
        from strategy_config import ENTRY_WINDOWS_ENABLED
        from meta_label import META_VETO_PROB
        metrics['policy_values'] = {
            'policy': dict(policy),
            'spread_pct_flat': SPREAD_PCT[asset_type],
            'meta_veto_prob': META_VETO_PROB,
            'q10_floor': q10_floor,
            'threshold': threshold,
            'lstm_weight': config.get('lstm_weight', 0.6),
            'entry_windows_enabled': bool(ENTRY_WINDOWS_ENABLED),
        }
    except Exception as e:
        print(f"[GATE] policy_values unavailable ({e})")
    try:
        p_ = f'{model_prefix}_' if model_prefix else ''
        mpath = BASE_DIR / f'{p_}model_v2.manifest.json'
        if mpath.exists():
            with open(mpath) as f:
                man = json.load(f)
            metrics['artifact_manifest_saved_at'] = man.get('saved_at')
            metrics['artifact_manifest_score'] = man.get('score')
        if not challenger_run:
            # shadow.challenger_prefix naming: '' -> 'challenger',
            # 'stock' -> 'stock_challenger'; manifest = <that>_model_v2.manifest.json
            # (flag-OFF breadcrumb — meaningless when this run IS gating
            # the challenger)
            ch = f'{prefix}_challenger' if prefix else 'challenger'
            ch_path = BASE_DIR / f'{ch}_model_v2.manifest.json'
            if (ch_path.exists() and mpath.exists()
                    and ch_path.stat().st_mtime > mpath.stat().st_mtime):
                print(f"[GATE] WARNING: a NEWER challenger manifest exists "
                      f"({ch_path.name}) — this run is gating the CHAMPION, "
                      f"not the model just trained (shadow-mode wiring; see "
                      f"module docstring)")
    except Exception as e:
        print(f"[GATE] manifest provenance unavailable ({e})")

    rslot = _report_slot(prefix, model_prefix)
    mtm = None
    if _s0 is not None:
        try:
            dump_path = (BASE_DIR /
                         f"{f'{rslot}_' if rslot else ''}stage0_preds.json")
            _s0.write_rows(s0_rows, dump_path)
            mtm = _s0.mtm_equity(all_trades, price_series, grid_ns)
            metrics['mtm_max_drawdown_pct'] = round(
                float(mtm['max_drawdown_pct']), 2)
            _mvp = None
            try:
                from meta_label import META_VETO_PROB as _mvp
            except Exception:
                pass
            metrics['stage0_dump'] = {
                'path': dump_path.name, 'n_rows': len(s0_rows),
                'horizon_bars': horizon, 'non_overlapping': True,
                'units': 'percent', 'threshold': threshold,
                'edge_floor_pct': required_edge_pct(
                    asset_type, SPREAD_PCT[asset_type]),
                'meta_veto_prob': _mvp,
                'n_unmarked_trades': int(mtm.get('n_unmarked_trades', 0)),
            }
            print(f"[GATE] stage0 dump: {dump_path.name} rows={len(s0_rows)} "
                  f"horizon={horizon} (non-overlapping; consumers: "
                  f"ic_by_name --time-key ts / rank_gradient_report --preds "
                  f"--fwd-bars 1)")
        except Exception as e:
            print(f"[GATE] stage0 dump/MTM unavailable ({e}) — measurement "
                  f"only, gate unaffected")
            mtm = None
    report_path = BASE_DIR / f"backtest_{f'{rslot}_' if rslot else ''}report.json"
    ordered_trades = sorted(all_trades, key=lambda t: t['exit_time'])
    persisted = ordered_trades[-500:]
    tmp = str(report_path) + '.tmp'
    try:
        with open(tmp, 'w') as f:
            payload = {'metrics': metrics,
                       'n_trades_total': len(all_trades),
                       'n_trades_persisted': len(persisted),
                       'trades': persisted}
            if mtm is not None:
                payload['mtm_equity_hourly'] = {
                    'ts': mtm['ts'], 'equity_pct': mtm['equity_pct']}
            json.dump(payload, f, indent=2)
        os.replace(tmp, report_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    print(f"\n=== POLICY BACKTEST ({asset_type}, last {days}d) ===")
    for k, v in metrics.items():
        if k != 'exit_reasons':
            print(f"  {k}: {v}")
    print(f"  exits: {metrics.get('exit_reasons')}")
    print(f"  report: {report_path}")
    return metrics


def _patch_report_gate_block(prefix: str, block: dict) -> None:
    """Best-effort: record the gate verdict in the already-written report.

    run_backtest writes the report BEFORE main() decides, so without this
    the persisted report never says whether the gate passed, which
    thresholds applied, or whether a rollback actually happened. Never
    raises — the verdict/exit code must not depend on report patching.
    """
    try:
        report_path = (BASE_DIR /
                       f"backtest_{f'{prefix}_' if prefix else ''}report.json")
        if not report_path.exists():
            return
        with open(report_path) as f:
            report = json.load(f)
        report['gate'] = block
        tmp = str(report_path) + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(report, f, indent=2)
            os.replace(tmp, report_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except Exception as e:
        print(f"[GATE] could not record gate verdict in report ({e})")


def _write_policy_gate_sidecar(model_slot: str, data_prefix: str,
                               gate_block: dict, metrics: dict) -> None:
    """Persist the challenger gate verdict to {slot}_policy_gate.json (tmp +
    os.replace, never raises) so the shadow-side promotion pre-flight
    (shadow._gate_preflight, landed in Wave B-2) can require passed=True with a
    fresh challenger_manifest_mtime before any promote. Instrumentation only."""
    try:
        path = BASE_DIR / f'{model_slot}_policy_gate.json'
        man = BASE_DIR / f'{model_slot}_model_v2.manifest.json'
        payload = {
            'passed': bool(gate_block.get('passed')),
            'gate': gate_block, 'data_prefix': data_prefix,
            'sharpe': metrics.get('sharpe'), 'dsr': metrics.get('dsr'),
            'n_trades': metrics.get('n_trades'),
            'checked_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'challenger_manifest_mtime': (int(man.stat().st_mtime)
                                          if man.exists() else None),
        }
        tmp = str(path) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        print(f"[GATE] challenger verdict recorded: {path.name} "
              f"passed={payload['passed']}")
    except Exception as e:
        print(f"[GATE] could not write policy-gate sidecar ({e})")


def main():
    ap = argparse.ArgumentParser(description='Backtest the actual trading policy')
    ap.add_argument('--prefix', default='', help="'' for crypto, 'stock' for stocks")
    ap.add_argument('--days', type=int, default=60,
                    help='lookback window (days), must be >= 1; run_pipeline '
                         'passes 44 (crypto) / 60 (stock) to stay inside the '
                         'untouched 12%% holdout — the bare default 60 '
                         'reaches INTO the crypto search region')
    ap.add_argument('--trials', type=int, default=None,
                    help='search-pool size for DSR deflation — must be the '
                         'CUMULATIVE selection pool of the resumed Optuna '
                         "study, not just this run's budget; undercounting "
                         'silently lowers the bar (default: the persisted '
                         'cumulative pool (adaptive_state cum_trials) under '
                         'PROMOTION_GATE_V2, else 100)')
    ap.add_argument('--gate', action='store_true',
                    help='restore the previous model if the backtest fails')
    ap.add_argument('--min-sharpe', type=float, default=0.0)
    # Default aligned to validation.DSR_MIN (2026-07 review: the gate ran at
    # 0.5 while the documented promotion bar was 0.60 — a silent drift)
    ap.add_argument('--min-dsr', type=float, default=DSR_MIN)
    ap.add_argument('--model-prefix', default='',
                    help="artifact slot to score: '' = the --prefix champion slot "
                         "(legacy); 'challenger'/'stock_challenger' = the shadow "
                         "challenger slot, replayed on the champion's book data "
                         "(GATE_TARGETS_CHALLENGER wiring, defect D03). A gate "
                         "failure on a challenger slot HOLDS the challenger — no "
                         "champion rollback. Falls back to the champion slot "
                         "(loudly) when the requested slot's core artifacts are "
                         "absent (first deploy).")
    ap.add_argument('--no-stage0-dump', action='store_true',
                    help='skip the per-bar Stage-0 predictions dump + hourly '
                         'MTM equity (B02, measurement-only; default ON)')
    args = ap.parse_args()

    global STAGE0_DUMP_DEFAULT
    if args.no_stage0_dump:
        STAGE0_DUMP_DEFAULT = False

    if args.days < 1:
        ap.error('--days must be >= 1 (a degenerate window produces a '
                 'zero-trade replay and, with --gate, a spurious rollback)')

    # Resolve the deflation pool: an explicit --trials always wins (both
    # modes, operator override). Otherwise read the persisted cumulative
    # pool so this gate deflates against the SAME selection pressure as
    # the fit gate (B03.2) — flag OFF keeps the legacy 100 byte-identical.
    if args.trials is not None:
        resolved_trials = args.trials
    else:
        cum = 0
        try:
            import adaptive_config
            cum = int(adaptive_config.load_adaptive_state(
                args.prefix or 'crypto').get('cum_trials', 0) or 0)
        except Exception:
            cum = 0
        _gate_v2 = bool(getattr(_strategy_config, 'PROMOTION_GATE_V2', False))
        print(f"[GATE] deflation pool: cum_trials={cum} "
              f"(legacy default 100; PROMOTION_GATE_V2={_gate_v2})")
        if _gate_v2 and cum > 0:
            resolved_trials = max(int(cum), 2)
        else:
            resolved_trials = 100

    requested = args.model_prefix or ''
    if requested and requested != args.prefix:
        rq = f'{requested}_'
        core_present = all((BASE_DIR / f'{rq}{s}').exists()
                           for s in ARTIFACT_SUFFIXES[:4])
    else:
        core_present = False
    model_slot, slot_reason = _resolve_model_slot(args.prefix, requested,
                                                  core_present)
    targets_challenger = (slot_reason == 'challenger')
    if slot_reason == 'fallback_champion':
        print(f"[GATE] --model-prefix '{requested}' requested but its core "
              f"artifacts are missing — falling back to the "
              f"{args.prefix or 'crypto'} champion slot (first deploy or "
              f"non-shadow save); a failure rolls that slot back only on a "
              f"genuine first deploy (no .prev) — an established champion "
              f"is HELD (D03)")
    p = f'{model_slot}_' if model_slot else ''
    try:
        try:
            if targets_challenger:
                metrics = run_backtest(args.prefix, args.days, resolved_trials,
                                       model_prefix=model_slot)
            else:
                # Legacy call kept positional-only: model_slot == args.prefix
                # (run_backtest defaults model_prefix to prefix), and the
                # 3-arg seam is monkeypatched by existing tests.
                metrics = run_backtest(args.prefix, args.days, resolved_trials)
        except FileNotFoundError as e:
            core = [BASE_DIR / f'{p}{s}' for s in ARTIFACT_SUFFIXES[:4]]
            if all(c.exists() for c in core):
                # The excuse below exists for ABSENT artifacts. All four
                # core artifacts are present, so this FNF came from
                # somewhere else (sidecar, data path, report write) — a
                # crash, not "nothing to gate". Returning 0 here would
                # report PASS for a gate that never ran.
                raise
            print(f"[BACKTEST] Missing artifact ({e}) — nothing to gate")
            return 0
    except BaseException as e:
        if args.gate:
            print(f"[GATE] CRASHED before a verdict "
                  f"({type(e).__name__}: {e}) — NO rollback was performed; "
                  f"the {args.prefix or 'crypto'} artifacts on disk are "
                  f"UNGATED by this run. Re-run backtest.py --gate manually.")
            try:
                from notify import notify
                notify(f"Backtest gate CRASHED for {args.prefix or 'crypto'} "
                       f"({type(e).__name__}) — no verdict, no rollback",
                       level='critical',
                       dedupe_key=f'gate-crash-{args.prefix}')
            except Exception:
                pass
        raise

    if args.gate:
        ok = (metrics['n_trades'] >= 10
              and metrics['sharpe'] > args.min_sharpe
              and metrics['dsr'] >= args.min_dsr)
        gate_block = {
            'enabled': True, 'min_sharpe': args.min_sharpe,
            'min_dsr': args.min_dsr, 'min_trades': 10,
            'passed': bool(ok), 'restored': None,
            'min_trl': metrics.get('dsr_min_trl'),
            'dsr_status': metrics.get('dsr_status'),
            'failed_checks': [c for c, bad in (
                ('n_trades', metrics['n_trades'] < 10),
                ('sharpe', not metrics['sharpe'] > args.min_sharpe),
                ('dsr', not metrics['dsr'] >= args.min_dsr)) if bad],
        }
        if targets_challenger:
            gate_block['gate_target'] = model_slot
        if not ok:
            print(f"\n[GATE] FAILED: n={metrics['n_trades']}, "
                  f"sharpe={metrics['sharpe']} (min {args.min_sharpe}), "
                  f"dsr={metrics['dsr']} (min {args.min_dsr})")
            # MinTRL instrumentation: how many more EFFECTIVE trades this
            # SR would need before the deflation bar could be cleared.
            mtl = metrics.get('dsr_min_trl')
            if mtl is not None and math.isfinite(mtl):
                print(f"[GATE] MinTRL: need "
                      f"~{max(0.0, mtl - metrics.get('dsr_n_eff_used', 0)):.0f} "
                      f"more effective trades at this SR "
                      f"(min_trl={mtl:.0f})")
            if targets_challenger:
                gate_block['action'] = 'hold_challenger'
                gate_block['restored'] = False
                print(f"[GATE] CHALLENGER HELD: '{model_slot}' failed the "
                      f"policy gate — the {args.prefix or 'crypto'} champion "
                      f"and its .prev files were NOT touched; the challenger "
                      f"keeps shadowing but must not promote until a passing "
                      f"replay (D03)")
                try:
                    from notify import notify
                    notify(f"Backtest gate FAILED for the "
                           f"{args.prefix or 'crypto'} CHALLENGER "
                           f"(sharpe={metrics['sharpe']}, dsr={metrics['dsr']}) "
                           f"— challenger held, champion untouched",
                           level='warning',
                           dedupe_key=f'gate-challenger-{args.prefix}')
                except Exception:
                    pass
            else:
                # D03 (GATE_TARGETS_CHALLENGER wiring): fallback_champion with
                # .prev present means the champion being replayed is an
                # ESTABLISHED model this run never touched (the challenger slot
                # was simply empty — promoted/discarded, or hypersearch saved
                # nothing). Rolling it back would restore an even older, ungated
                # .prev. Hold instead. On a genuine first deploy no .prev exists
                # and the legacy call (a loud no-op there) runs unchanged.
                _prev_ok = (slot_reason == 'fallback_champion' and
                            all((BASE_DIR / f'{p}{s}.prev').exists()
                                for s in ARTIFACT_SUFFIXES[:4]))
                if _prev_ok:
                    restored = False
                    gate_block['restored'] = False
                    gate_block['action'] = 'hold_champion_no_challenger'
                    print(f"[GATE] CHAMPION HELD: fallback replay of the "
                          f"established {args.prefix or 'crypto'} champion "
                          f"failed, but no challenger slot existed this run — "
                          f"NOT rolling back to a stale .prev (D03)")
                else:
                    restored = restore_previous_model(args.prefix)
                    gate_block['restored'] = bool(restored)
                if not restored and not _prev_ok:
                    print("[GATE] *** NOT ROLLED BACK — no .prev artifacts "
                          "existed; the model that just FAILED this gate is "
                          "still deployed ***")
                try:
                    from notify import notify
                    if _prev_ok:
                        # Deliberate hold, not a missing rollback: warning,
                        # never a critical page claiming an ungated deploy.
                        notify(f"Backtest gate FAILED for "
                               f"{args.prefix or 'crypto'} "
                               f"(sharpe={metrics['sharpe']}, "
                               f"dsr={metrics['dsr']}) — established champion "
                               f"HELD, nothing rolled back (no challenger "
                               f"slot this run)",
                               level='warning',
                               dedupe_key=f'gate-{args.prefix}')
                    else:
                        notify(f"Backtest gate FAILED for {args.prefix or 'crypto'} "
                               f"(sharpe={metrics['sharpe']}, dsr={metrics['dsr']}) — "
                               f"{'previous model restored' if restored else 'no rollback available'}",
                               level=('warning' if restored else 'critical'),
                               dedupe_key=f'gate-{args.prefix}')
                except Exception:
                    pass
            # Distinct exit code: a deterministic policy rejection, not a
            # crash. run_pipeline treats 3 as final for *_backtest_gate
            # phases — no retry (retrying a deterministic rejection is
            # useless) — and non-fatal to the rest of the retrain run
            # (the model is already rolled back).
            gate_block['exit_code'] = 3
            _patch_report_gate_block(_report_slot(args.prefix, model_slot),
                                     gate_block)
            if targets_challenger:
                _write_policy_gate_sidecar(model_slot, args.prefix,
                                           gate_block, metrics)
            return 3
        else:
            print(f"\n[GATE] PASSED: sharpe={metrics['sharpe']}, dsr={metrics['dsr']}")
            gate_block['exit_code'] = 0
            _patch_report_gate_block(_report_slot(args.prefix, model_slot),
                                     gate_block)
            if targets_challenger:
                _write_policy_gate_sidecar(model_slot, args.prefix,
                                           gate_block, metrics)
    return 0


if __name__ == '__main__':
    sys.exit(main())
