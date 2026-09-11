"""R2C-07 (FR-08): retrain-gain ledger — does the weekly retrain pay?

At each accepted retrain, score BOTH the incumbent stack and the freshly
saved stack on the same trailing ~7 days of purged bars and append one
paired row to the adaptive-state ledger. After >= 12 weekly rows the
B03.3 Ibragimov-Muller block-t decides the cadence question (an owner
decision, never automatic). Measurement-only: nothing here touches the
live serving path, the gate, or any score.

Incumbent resolution (the --shadow slot correction, mirroring
shadow.py's champion/challenger split):
  - save_prefix != prefix  -> the save went to the CHALLENGER slot; the
    champion's artifacts at `prefix` are untouched and ARE the incumbent
    (reading `.prev` here would compare against the challenger's own
    previous incarnation, not the model actually serving).
  - save_prefix == prefix  -> same-slot save; save_model_atomically just
    backed the outgoing stack up as `.prev` — that is the incumbent.

Two-machine reality: the pure kernels (paired_scores, stack_valid_rows,
trailing_purged_rows, append_ledger_row, window_gather_plan,
incumbent_paths) are numpy-only and Mac-tested;
record_retrain_gain / _load_stack / _score_stack lazily import
torch/joblib/lightgbm and run only on the Jetson, inside hypersearch's
post-save call site (fail-soft — a ledger failure never blocks a save).

Prefixes throughout are hypersearch's FILENAME prefixes, trailing
underscore included ('', 'stock_', 'challenger_', 'stock_challenger_').
"""
import os
from datetime import datetime

import numpy as np

# ~2 years of weekly retrains per book; the ledger is evidence, not a log.
LEDGER_CAP = 104
LEDGER_KEY = 'retrain_ledger'
# Fewer than a day of hourly bars is not a scoring window.
MIN_LEDGER_BARS = 24

_CORE_FILES = ('model_v2.pth', 'config_v2.pkl', 'scaler_v2.pkl',
               'feature_cols_v2.pkl')


# ---------------------------------------------------------------------------
# Pure kernels (Mac-testable)
# ---------------------------------------------------------------------------

def paired_scores(pred_a, pred_b, y):
    """Paired MSE + Pearson IC for two prediction vectors against one y.

    Rows enter ONLY where all three of (pred_a, pred_b, y) are finite —
    both stacks are scored on the identical bar set, which is what makes
    the per-retrain delta a paired observation for the block-t later.

    Returns {'mse_a', 'mse_b', 'ic_a', 'ic_b', 'n'}; MSEs are None when
    n == 0, ICs are None when n < 3 or the relevant variance is zero.
    """
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(yy)
    a, b, yy = a[m], b[m], yy[m]
    n = int(a.size)
    out = {'mse_a': None, 'mse_b': None, 'ic_a': None, 'ic_b': None, 'n': n}
    if n == 0:
        return out
    out['mse_a'] = float(np.mean((a - yy) ** 2))
    out['mse_b'] = float(np.mean((b - yy) ** 2))
    if n >= 3:
        sy = yy.std()
        for key, v in (('ic_a', a), ('ic_b', b)):
            sv = v.std()
            if sv > 0.0 and sy > 0.0:
                out[key] = float(np.mean((v - v.mean()) * (yy - yy.mean()))
                                 / (sv * sy))
    return out


def stack_valid_rows(ticker_boundaries, seq_len):
    """Rows with >= seq_len bars of same-ticker history — the mirror of
    hypersearch's _valid_indices, on {ticker: (start, end)} dicts or
    (start, end) iterables over the contiguous per-ticker block layout."""
    if isinstance(ticker_boundaries, dict):
        blocks = sorted(ticker_boundaries.values())
    else:
        blocks = sorted(ticker_boundaries)
    sl = int(seq_len)
    out = [np.arange(start + sl, end)
           for start, end in blocks if end - start > sl]
    return (np.concatenate(out).astype(np.int64) if out
            else np.array([], dtype=np.int64))


def trailing_purged_rows(times, days=7.0, forward_bars=24):
    """Boolean mask over `times`: the trailing `days` calendar window,
    PURGED of the final forward_bars label BARS.

    Kept: t > max(times) - days*86400  AND  t <= u[-(forward_bars+1)],
    where u is the DISTINCT pooled bar grid — every kept row's fb-bar
    forward label window completes inside the available data, so neither
    stack is scored on a label that was still open when the harvest
    ended. Counting bars (not calendar hours) matches the L6 lesson: on
    the stock RTH grid a 24-calendar-hour cut removes only ~7 trading
    bars; on crypto's continuous hourly grid the two rules coincide.
    Empty times -> empty mask; fewer than forward_bars+1 distinct bars
    -> all-False (the purge swallows the window).
    """
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    u = np.unique(t)
    fb = int(forward_bars)
    if fb < 0:
        fb = 0
    if u.size <= fb:
        return np.zeros(t.size, dtype=bool)
    lo = u[-1] - float(days) * 86400.0
    hi = u[u.size - fb - 1]
    return (t > lo) & (t <= hi)


def append_ledger_row(state, row, cap=LEDGER_CAP):
    """Append one ledger row to state[LEDGER_KEY], keeping the newest
    `cap` rows (mutates and returns state)."""
    rows = state.setdefault(LEDGER_KEY, [])
    rows.append(dict(row))
    if len(rows) > int(cap):
        del rows[:len(rows) - int(cap)]
    return state


def window_gather_plan(rows, seq_len):
    """(src, remap) plan for a windowed gather that touches ONLY the rows
    the windows actually need (FR-08's Jetson-honesty budget: two
    inference passes over ~a week of bars must not transform the whole
    500k-row panel twice).

    Each scored row r consumes source rows r-seq_len..r-1 (hypersearch's
    gather_windows offsets convention, np.arange(-seq_len, 0)). Returns:
      src   — sorted unique source-row indices (int64),
      remap — (len(rows), seq_len) indices into src such that, for any
              per-row matrix M over the full panel,
              M[src][remap] == M[rows[:, None] + offsets[None, :]].
    Empty rows -> (empty, (0, seq_len)) — shapes stay consistent.
    """
    rows = np.asarray(rows, dtype=np.int64)
    sl = int(seq_len)
    offsets = np.arange(-sl, 0, dtype=np.int64)
    need = rows[:, None] + offsets[None, :]
    src = np.unique(need)
    remap = np.searchsorted(src, need)
    return src, remap


# ---------------------------------------------------------------------------
# Jetson-only scorer (torch / joblib / lightgbm — lazy imports)
# ---------------------------------------------------------------------------

def incumbent_paths(prefix, save_prefix):
    """(artifact-path dict, source label) for the incumbent stack.

    Challenger-slot save (save_prefix != prefix): the champion's live
    artifacts at `prefix` (source 'champion'). Same-slot save: the
    freshly written `.prev` backups (source 'prev').
    """
    if save_prefix != prefix:
        return ({f: f'{prefix}{f}' for f in _CORE_FILES}, 'champion')
    return ({f: f'{prefix}{f}.prev' for f in _CORE_FILES}, 'prev')


def _load_stack(paths, lgb_path, label):
    """Load one (LSTM, config, scaler, feature_cols, booster-or-None)
    stack from disk. Raises on a missing/corrupt CORE artifact (the
    caller is fail-soft); a missing LGB booster degrades to raw LSTM —
    the same fallback the live loop applies."""
    import joblib
    import torch
    from model_v2 import RegressionLSTM
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'{label} stack incomplete: {missing}')
    config = joblib.load(paths['config_v2.pkl'])
    scaler = joblib.load(paths['scaler_v2.pkl'])
    fcols = joblib.load(paths['feature_cols_v2.pkl'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionLSTM(config['input_dim'], config['hidden_dim'],
                           config['num_layers'], config['dropout'],
                           config.get('n_heads', 4)).to(dev)
    model.load_state_dict(torch.load(paths['model_v2.pth'],
                                     map_location=dev, weights_only=True))
    model.eval()
    booster = None
    if lgb_path and os.path.exists(lgb_path):
        try:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=lgb_path)
        except Exception as e:
            print(f"[LEDGER] {label} LGB booster load failed ({e}) — "
                  f"scoring raw LSTM")
    return {'model': model, 'config': config, 'scaler': scaler,
            'feature_cols': fcols, 'booster': booster, 'device': dev}


def _score_stack(stack, all_features, feature_cols, rows):
    """Deployed-predictor predictions for `rows` under ONE stack's OWN
    scaler/config: w*LSTM + (1-w)*LGB when its booster exists (predict_now
    semantics, w = config.get('lstm_weight', 0.6)), raw LSTM otherwise.
    Columns are re-indexed BY NAME into the stack's own training order;
    a feature the current harvest no longer carries raises (caller skips,
    fail-soft) — silently zero-filling would score a chimera.

    Memory (Jetson 8 GB): only the unique window-source rows are gathered
    and scaled (window_gather_plan) — for a trailing week that is a few
    hundred rows per ticker, never the full 500k-row panel."""
    import torch
    cfg = stack['config']
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    missing = [c for c in stack['feature_cols'] if c not in col_idx]
    if missing:
        raise KeyError(f'features absent from current harvest: {missing}')
    idx = np.asarray([col_idx[c] for c in stack['feature_cols']],
                     dtype=np.int64)
    seq_len = int(cfg['seq_len'])
    rows = np.asarray(rows, dtype=np.int64)
    src, remap = window_gather_plan(rows, seq_len)
    scaled = stack['scaler'].transform(
        all_features[np.ix_(src, idx)]).astype(np.float32)
    dev = stack['device']
    preds = []
    with torch.inference_mode():
        for i in range(0, len(rows), 1024):
            windows = scaled[remap[i:i + 1024]]
            out = stack['model'](torch.from_numpy(windows).to(dev))
            preds.append(out.cpu().numpy())
    lstm_preds = (np.concatenate(preds) if preds
                  else np.zeros(0, dtype=np.float64))
    if stack['booster'] is None:
        return np.asarray(lstm_preds, dtype=np.float64)
    lgb_preds = np.empty(len(rows), dtype=np.float64)
    for i in range(0, len(rows), 1024):
        ri = rows[i:i + 1024]
        # predict_now's flatten_sequence contract: == windows.reshape(-1)
        X = scaled[remap[i:i + 1024]].reshape(len(ri), -1)
        lgb_preds[i:i + len(ri)] = stack['booster'].predict(X)
    w = float(cfg.get('lstm_weight', 0.6))
    return w * np.asarray(lstm_preds, dtype=np.float64) + (1.0 - w) * lgb_preds


def record_retrain_gain(prefix, save_prefix, asset_type,
                        all_features, feature_cols, all_returns_by_fb,
                        all_times, tickers, ticker_boundaries,
                        days=7.0, cap=LEDGER_CAP, state=None):
    """Score incumbent vs fresh stack on the trailing week and append one
    FR-08 row {date, mse_inc, mse_new, ic_inc, ic_new, n_bars, ...} to
    the adaptive-state ledger. Fail-soft: returns the row dict on
    success, None on any failure (printed, never raised past here).

    state: the CALLER'S live adaptive-state dict when it holds one.
    Hypersearch main loads adaptive_state early and re-saves it via
    update_after_search AFTER this runs — appending into a freshly
    loaded copy here would be clobbered by that later save (the row
    would never survive a run). Passing the caller's object appends the
    row in-memory too, so the later save PRESERVES it; None (standalone
    use) loads/saves the state file directly."""
    try:
        new_paths = {f: f'{save_prefix}{f}' for f in _CORE_FILES}
        inc_paths, inc_source = incumbent_paths(prefix, save_prefix)
        new_stack = _load_stack(new_paths, f'{save_prefix}lgb_model.txt',
                                'fresh')
        inc_lgb = (f'{prefix}lgb_model.txt' if inc_source == 'champion'
                   else f'{prefix}lgb_model.txt.prev')
        inc_stack = _load_stack(inc_paths, inc_lgb, 'incumbent')

        # y under the NEW config's key logic (evaluate_on_holdout's exact
        # fallback chain); a horizon change is recorded, not hidden.
        fb_new = int(new_stack['config'].get('forward_bars', 24))
        fb_inc = int(inc_stack['config'].get('forward_bars', 24))
        rkey = (('tb', fb_new)
                if new_stack['config'].get('target_kind') == 'tb' else fb_new)
        returns = all_returns_by_fb.get(rkey)
        if returns is None:
            returns = all_returns_by_fb.get(fb_new)
        if returns is None:
            returns = next(v for k, v in all_returns_by_fb.items()
                           if not isinstance(k, tuple))

        mask = trailing_purged_rows(all_times, days=days,
                                    forward_bars=max(fb_new, fb_inc))
        valid_new = stack_valid_rows(ticker_boundaries,
                                     new_stack['config']['seq_len'])
        valid_inc = stack_valid_rows(ticker_boundaries,
                                     inc_stack['config']['seq_len'])
        rows = np.intersect1d(valid_new, valid_inc)
        rows = rows[mask[rows] & np.isfinite(
            np.asarray(returns, dtype=np.float64)[rows])]
        if len(rows) < MIN_LEDGER_BARS:
            print(f"[LEDGER] only {len(rows)} scorable trailing bars "
                  f"(< {MIN_LEDGER_BARS}) — no ledger row")
            return None

        pred_inc = _score_stack(inc_stack, all_features, feature_cols, rows)
        pred_new = _score_stack(new_stack, all_features, feature_cols, rows)
        scores = paired_scores(pred_inc, pred_new, returns[rows])

        row = {
            'date': datetime.now().isoformat(),
            'asset_type': asset_type,
            'slot_new': save_prefix,
            'incumbent_source': inc_source,
            'mse_inc': (round(scores['mse_a'], 8)
                        if scores['mse_a'] is not None else None),
            'mse_new': (round(scores['mse_b'], 8)
                        if scores['mse_b'] is not None else None),
            'ic_inc': (round(scores['ic_a'], 4)
                       if scores['ic_a'] is not None else None),
            'ic_new': (round(scores['ic_b'], 4)
                       if scores['ic_b'] is not None else None),
            'n_bars': scores['n'],
            'fb_inc': fb_inc,
            'fb_new': fb_new,
            'blend_inc': inc_stack['booster'] is not None,
            'blend_new': new_stack['booster'] is not None,
        }
        from adaptive_config import load_adaptive_state, save_adaptive_state
        if state is None:
            state = load_adaptive_state(asset_type)
        append_ledger_row(state, row, cap=cap)
        save_adaptive_state(state)
        n_rows = len(state.get(LEDGER_KEY, []))
        print(f"[LEDGER] FR-08 row {n_rows}/{cap} ({asset_type}, "
              f"incumbent={inc_source}): mse_inc={row['mse_inc']} vs "
              f"mse_new={row['mse_new']}, ic_inc={row['ic_inc']} vs "
              f"ic_new={row['ic_new']} on {row['n_bars']} bars"
              + (" — >=12 rows: run the B03.3 IM block-t (owner decision)"
                 if n_rows >= 12 else ""))
        return row
    except Exception as e:
        print(f"[LEDGER] retrain-gain scoring failed (non-fatal): {e}")
        return None
