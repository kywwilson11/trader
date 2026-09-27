"""
Regression-based Optuna hyperparameter search with walk-forward cross-validation.

Trains a single RegressionLSTM (no separate bear/bull models) per book that
predicts continuous returns, plus a LightGBM ensemble leg, searched via
Optuna TPE over a walk-forward cross-validated objective.

Key properties:
  - Single model (no separate bear/bull)
  - Huber loss with return-weighted emphasis
  - Walk-forward CV with expanding windows (3 folds)
  - Risk-adjusted objective: mean fold Sharpe − 0.5·std, cost-aware, holdout
    Deflated-Sharpe gate before any save
  - FP16 mixed precision for ~2x speedup on Jetson tensor cores
  - Feature set follows the indicator preset: --preset overrides, else
    indicator_config.load_indicator_config() decides (run_pipeline passes
    --preset stationary explicitly — stationary features, no raw
    price/volume drift, on the production path)

Usage:
    python scripts/hypersearch_v2.py --trials 200 --data training_data.csv --preset stationary
    python scripts/hypersearch_v2.py --trials 50 --prefix stock --data stock_training_data.csv

Note: data loads parquet-first via data_utils, but the STORE is chosen by the
--data PATH ('stock' in --data selects the stock panel, else crypto — see
load_data); --data itself is only the direct-CSV fallback. --prefix names the
OUTPUT artifacts, it does not select the input store.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import hashlib
import json
import math
import os
import time
from datetime import datetime

# Must be set BEFORE any torch.cuda call lazily initializes the allocator —
# the old placement (after set_per_process_memory_fraction) meant
# expandable_segments never actually applied.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler
import joblib
import optuna
from optuna.pruners import MedianPruner

from model_v2 import RegressionLSTM
from gpu_lock import acquire_for_training
from adaptive_config import (load_adaptive_state, get_search_space_for_trial,
                              update_after_search, get_trial_count)
from objective_utils import (simulate_trades_core, ticker_block_ids,
                             v3_trade_threshold_range, refit_epoch_budget,
                             lgb_refit_indices, fixed_boost_rounds,
                             derive_seed, lagged_regime_series,
                             embargo_end_time, holdout_boundary,
                             window_cutoff, cs_rank_ic)

_STATUS_FILE = Path(__file__).resolve().parent.parent / 'pipeline_status.json'


_SKIP_STATUS = False  # Set by --no-status flag


def _write_pipeline_status(status):
    """Write pipeline_status.json, merging with existing data to preserve prior results."""
    if _SKIP_STATUS:
        return
    # Read existing status to preserve fields like crypto_final_score
    existing = {}
    try:
        with open(_STATUS_FILE) as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass
    existing.update(status)
    existing['updated_at'] = datetime.now().isoformat()
    tmp = str(_STATUS_FILE) + f'.tmp.{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, str(_STATUS_FILE))
    except OSError:
        pass


NUM_TRIALS = 300
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10
SOUP_K = 4  # checkpoint-soup size (avg weights of the K best epochs)
PRUNE_WARMUP_EPOCHS = 12
PRUNE_STARTUP_TRIALS = 60
NUM_FOLDS = 3
EMBARGO_MULTIPLIER = 1  # embargo = seq_len * this

FORWARD_BARS = [12, 18, 24, 32, 48]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# NOTE: cudnn.benchmark deliberately NOT enabled — this model has no
# convolutions (it gains nothing) and benchmark mode is a documented source
# of transient cuDNN workspace OOM spikes during Optuna trials.

# Cap CUDA allocator to prevent fatal kernel-level OOM on Jetson (unified memory).
# Without this, OOM triggers NvMapMemAllocInternalTagged errors that corrupt CUDA
# context and kill the process. With the cap, PyTorch raises catchable OutOfMemoryError.
if device.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.40)  # ~3GB of 7.6GB for CUDA

print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description='Regression hyperparameter search (v2)')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS,
                        help=f'Number of trials (default: {NUM_TRIALS})')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing study DB and start fresh')
    parser.add_argument('--data', type=str, default='training_data.csv',
                        help='Path to training CSV (default: training_data.csv)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output files (e.g. "stock" -> stock_model_v2.pth)')
    parser.add_argument('--shadow', action='store_true',
                        help='Save a gated new model as CHALLENGER when a '
                             'champion exists (shadow mode; promotion via '
                             'live DM test in shadow.py)')
    parser.add_argument('--preset', type=str, default=None,
                        help='Indicator preset override (default: None = the '
                             'indicator_config.load_indicator_config() preset; '
                             'run_pipeline passes "stationary" explicitly)')
    parser.add_argument('--max-rows', type=int, default=500_000,
                        help='Max total rows to load (default: 500000)')
    parser.add_argument('--mode', type=str, default='',
                        choices=['', 'refine', 'explore', 'initial'],
                        help='Adaptive mode (default: auto-detect from state)')
    parser.add_argument('--no-status', action='store_true',
                        help='Skip writing pipeline_status.json (parent handles it)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# FIX-R1 (2026-09) non-finite parity guard — behaviour-identical twin of
# predict_now._sanitize_nonfinite_features (kept local so the trainer does
# not import the live-serving module and its torch thread side effect);
# tests/test_oi_inf_2026_09.py pins both on the same inputs. Fill = 0.0, the
# neutral value harvest_crypto_data._fill_archive_features writes for a NaN
# archive feature and live serves when a value is unavailable.
NONFINITE_FILL = 0.0


def _sanitize_nonfinite_features(X, fill=NONFINITE_FILL, copy=True):
    """Replace +-inf (-> NaN) and NaN in a feature matrix with `fill`.

    Returns (X_out, n_nonfinite, n_inf, bad_col_idx) — bad_col_idx indexes
    the last axis (feature columns). NOT model-facing for finite input:
    an all-finite (or non-float) matrix is returned as the SAME object,
    untouched (bit-identical scaler fits and fold matrices).
    """
    X = np.asarray(X)
    if X.dtype.kind not in 'fc':
        return X, 0, 0, []
    bad = ~np.isfinite(X)
    n_bad = int(np.count_nonzero(bad))
    if n_bad == 0:
        return X, 0, 0, []
    n_inf = int(np.count_nonzero(np.isinf(X)))
    bad_cols = np.flatnonzero(
        bad.reshape(-1, X.shape[-1]).any(axis=0)).tolist()
    if copy:
        X = X.copy()
    X[bad] = fill
    return X, n_bad, n_inf, bad_cols


def load_data(data_path='training_data.csv', preset_override=None,
              max_rows=500_000, window_days=None):
    print("Loading data...")
    # Try Parquet first (faster), fall back to CSV
    from data_utils import load_training_data
    pq_prefix = 'stock' if 'stock' in data_path else 'crypto'
    df = load_training_data(pq_prefix)
    if df.empty:
        # Direct CSV fallback if data_utils didn't find anything
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Dataset: {len(df)} rows")

    # Detect multi-horizon columns
    valid_target_cols = {f'Target_Return_{fb}' for fb in FORWARD_BARS}
    csv_target_cols = [c for c in df.columns if c.startswith('Target_Return_') and c in valid_target_cols]
    stale = [c for c in df.columns if c.startswith('Target_Return_') and c not in valid_target_cols and c != 'Target_Return']
    if stale:
        df = df.drop(columns=stale)
        print(f"Dropped stale target columns: {stale}")
    has_multi_horizon = len(csv_target_cols) > 0
    if has_multi_horizon:
        print(f"Multi-horizon targets: {sorted(csv_target_cols)}")
    else:
        print("Legacy single-horizon dataset (Target_Return only, treated as forward_bars=4)")

    if has_multi_horizon:
        _missing = [fb for fb in FORWARD_BARS
                    if f'Target_Return_{fb}' not in df.columns]
        if _missing:
            print(f"[WARN] Target_Return columns missing for fb={_missing}: "
                  f"trials at those horizons will train AND holdout-gate on "
                  f"the legacy shortest-horizon 'Target_Return' substitute "
                  f"(silently mis-specified objective) — re-harvest before "
                  f"trusting this search (adaptive forward_bars likely "
                  f"expanded after the last harvest)")

    exclude_cols = ['Ticker', 'Date', 'Datetime', 'NextClose']
    exclude_cols += [c for c in df.columns if c.startswith('Target_Return')]
    exclude_cols += [c for c in df.columns if c.startswith('TB_')]
    # Eff_Spread_Pct is a per-bar COST annotation (wave 6), not a predictive
    # feature, and is not computed on the live path — including it would both
    # leak cost into the inputs and break train/live feature parity.
    exclude_cols += ['Eff_Spread_Pct']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Filter features by preset
    from indicator_config import load_indicator_config, get_preset_features
    preset_name = preset_override or load_indicator_config()["preset"]
    preset_features = get_preset_features(preset_name)
    if preset_features is not None:
        feature_cols = [c for c in feature_cols if c in preset_features]
    print(f"Preset: {preset_name} ({len(feature_cols)} features)")

    # R2C-05 (FR-02): trailing training-window mask, per ticker — each name
    # keeps only its last window_days of bars, cutoff anchored to its own
    # final bar (objective_utils.window_cutoff). None = full history,
    # byte-identical. Applied BEFORE the --max-rows cap so the cap's
    # rows/ticker arithmetic sees the windowed panel. Only the FR-02
    # window_ab.py runner passes this today.
    if window_days is not None:
        n_before = len(df)
        kept = []
        for ticker in df['Ticker'].unique():
            tdf = df[df['Ticker'] == ticker]
            t = tdf.index.view('int64') // 10**9
            cut = window_cutoff(t, window_days)
            kept.append(tdf if cut is None else tdf[t >= cut])
        if kept:
            df = pd.concat(kept).sort_index()
        print(f"[WINDOW] window_days={window_days}: {n_before} -> {len(df)} "
              f"rows (per-ticker trailing mask)")

    # Apply --max-rows cap: keep most recent rows per ticker
    tickers = df['Ticker'].unique()
    n_tickers = len(tickers)
    if len(df) > max_rows and n_tickers > 0:
        rows_per_ticker = max_rows // n_tickers
        print(f"Capping to {max_rows} rows ({rows_per_ticker}/ticker)...")
        df = df.sort_index()
        capped = []
        for ticker in tickers:
            tdf = df[df['Ticker'] == ticker]
            capped.append(tdf.tail(rows_per_ticker))
        df = pd.concat(capped).sort_index()
        print(f"After capping: {len(df)} rows")

    # Per-ticker data arrays (scaler fit deferred to per-fold)
    all_features_list = []
    all_returns_dict = {fb: [] for fb in FORWARD_BARS}
    all_tb_dict = {fb: [] for fb in FORWARD_BARS}
    # Per-row label SPAN (bars held) — kept so the holdout gate and the
    # training loss can weight by average uniqueness (sample_weights.py).
    # Previously TB_Bars_* were dropped at load; that discarded the only
    # signal that says how non-IID the overlapping labels are.
    all_tb_bars_dict = {fb: [] for fb in FORWARD_BARS}
    has_tb = any(f'TB_Ret_{fb}' in df.columns for fb in FORWARD_BARS)
    has_tb_bars = any(f'TB_Bars_{fb}' in df.columns for fb in FORWARD_BARS)
    all_times_list = []
    all_label_times_list = []
    ticker_boundaries = {}

    max_fb = max(FORWARD_BARS)
    offset = 0
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        features = tdf[feature_cols].values.astype(np.float32)
        all_features_list.append(features)

        # Row timestamps + the timestamp of each row's LABEL bar (the bar
        # max_fb steps ahead). Label times let folds purge any train row
        # whose forward-return window crosses the fold boundary — the core
        # of purged walk-forward CV.
        times = tdf.index.view('int64') // 10**9  # epoch seconds
        n = len(times)
        label_idx = np.minimum(np.arange(n) + max_fb, n - 1)
        all_times_list.append(times.astype(np.int64))
        all_label_times_list.append(times[label_idx].astype(np.int64))

        for fb in FORWARD_BARS:
            col = f'Target_Return_{fb}'
            if col in tdf.columns:
                all_returns_dict[fb].append(tdf[col].values.astype(np.float32))
            else:
                all_returns_dict[fb].append(tdf['Target_Return'].values.astype(np.float32))
            tb_col = f'TB_Ret_{fb}'
            if has_tb and tb_col in tdf.columns:
                all_tb_dict[fb].append(tdf[tb_col].values.astype(np.float32))
            bars_col = f'TB_Bars_{fb}'
            if has_tb_bars and bars_col in tdf.columns:
                all_tb_bars_dict[fb].append(
                    tdf[bars_col].values.astype(np.float32))

        ticker_boundaries[ticker] = (offset, offset + len(features))
        offset += len(features)

    all_features = np.vstack(all_features_list)
    # FIX-R1 parity guard, right before any scaler sees the matrix (the
    # per-fold RobustScaler fits): (inf - median) / iqr = inf would poison
    # every fold. In place (all_features is a fresh vstack we own), so no
    # extra full-matrix copy on the Jetson. No-op on finite data. Runs
    # after the float32 cast, so a float64 value that overflowed to inf
    # is caught too.
    all_features, _n_bad, _n_inf, _bad_idx = _sanitize_nonfinite_features(
        all_features, copy=False)
    if _n_bad:
        print(f"[SANITIZE] {_n_bad} non-finite feature value(s) "
              f"({_n_inf} inf) in {[feature_cols[j] for j in _bad_idx][:6]} "
              f"-> neutral {NONFINITE_FILL}")
    all_times = np.concatenate(all_times_list)
    all_label_times = np.concatenate(all_label_times_list)
    all_returns_by_fb = {}
    all_tb_bars_by_fb = {}
    for fb in FORWARD_BARS:
        if all_returns_dict[fb]:
            all_returns_by_fb[fb] = np.concatenate(all_returns_dict[fb])
        # Triple-barrier (exit-stack-matched) targets, when the harvest
        # produced them: stored under the 'tb' key namespace so the
        # search can choose target_kind per trial
        if has_tb and all_tb_dict[fb] and len(all_tb_dict[fb]) == len(all_returns_dict[fb]):
            all_returns_by_fb[('tb', fb)] = np.concatenate(all_tb_dict[fb])
        # Per-row label spans (bars held), aligned to the same contiguous
        # ticker-concatenated index as all_returns_by_fb[fb].
        if has_tb_bars and all_tb_bars_dict[fb] and \
                len(all_tb_bars_dict[fb]) == len(all_returns_dict[fb]):
            all_tb_bars_by_fb[fb] = np.concatenate(all_tb_bars_dict[fb])
    if has_tb:
        print(f"Triple-barrier targets available: "
              f"{sorted(k[1] for k in all_returns_by_fb if isinstance(k, tuple))}")
    if all_tb_bars_by_fb:
        print(f"Label-span (TB_Bars) horizons for uniqueness weighting: "
              f"{sorted(all_tb_bars_by_fb)}")

    del all_features_list, all_returns_dict, all_tb_dict, all_tb_bars_dict
    del all_times_list, all_label_times_list, df
    gc.collect()

    print(f"Contiguous arrays: {all_features.shape}, {all_features.nbytes / 1e6:.1f} MB")
    input_dim = all_features.shape[1]

    return (all_features, all_returns_by_fb, all_times, all_label_times,
            tickers, ticker_boundaries,
            feature_cols, input_dim, preset_name, has_multi_horizon,
            all_tb_bars_by_fb)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

# Final 12% of the pooled ROW-timestamp distribution (a row quantile — equal to
# calendar time only on a balanced panel), never shown to Optuna; a fixed trailing
# span replaces it under FIXED_HOLDOUT_DAYS (R2C-05 / FR-01).
HOLDOUT_FRACTION = 0.12


def _valid_indices(tickers, ticker_boundaries, seq_len):
    """Row indices with at least seq_len bars of same-ticker history."""
    all_valid = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        if end - start > seq_len:
            all_valid.append(np.arange(start + seq_len, end))
    return np.concatenate(all_valid) if all_valid else np.array([], dtype=np.int64)


def _fixed_holdout_days():
    """R2C-05 (FR-01): fixed trailing holdout span in days, or None = legacy
    proportional holdout.

    TRADER_FIXED_HOLDOUT_DAYS env (numeric) wins over
    strategy_config.FIXED_HOLDOUT_DAYS; both default None (byte-identical).
    Flipping changes fold layout AND what the gate scores — rides Chain-2's
    single gotcha-#2 study reset, never a reset of its own.
    """
    v = os.environ.get('TRADER_FIXED_HOLDOUT_DAYS')
    if v not in (None, ''):
        try:
            return float(v)
        except ValueError:
            print(f"[HOLDOUT] ignoring non-numeric "
                  f"TRADER_FIXED_HOLDOUT_DAYS={v!r}")
    try:
        from strategy_config import FIXED_HOLDOUT_DAYS
        return (float(FIXED_HOLDOUT_DAYS)
                if FIXED_HOLDOUT_DAYS is not None else None)
    except Exception:
        return None


def get_holdout_boundary(all_times) -> int:
    """Timestamp separating the search region from the untouched holdout.

    R2C-05 (FR-01): the ONE choke point every consumer inherits — folds,
    the final-refit purge, the blend M4 guard, the OOF pack and
    evaluate_on_holdout. Default (FIXED_HOLDOUT_DAYS None) delegates to the
    legacy 0.12-quantile rule, byte-identical; a fixed span pins successive
    retrains / FR-02 window arms to comparable-width holdouts.
    """
    return holdout_boundary(all_times, fixed_days=_fixed_holdout_days(),
                            holdout_fraction=HOLDOUT_FRACTION)


def get_holdout_indices(all_times, tickers, ticker_boundaries, seq_len):
    """Rows in the final holdout slice (scored ONCE, for the winner only)."""
    valid = _valid_indices(tickers, ticker_boundaries, seq_len)
    boundary = get_holdout_boundary(all_times)
    return valid[all_times[valid] > boundary]


def get_walk_forward_folds(all_times, all_label_times, tickers,
                           ticker_boundaries, seq_len, n_folds=NUM_FOLDS,
                           purge_val_labels=False):
    """Expanding-window walk-forward folds split by CALENDAR TIME.

    The old implementation sliced the index array POSITIONALLY — but rows
    are stored as contiguous per-ticker blocks, so "train on the first 60%"
    meant "train on the first tickers' ENTIRE 2021-2026 history and
    validate on other tickers over the same calendar window". With 6
    cryptos correlated 0.7-0.9 to BTC that is near-direct leakage, and
    train even contained bars dated AFTER val.

    Now:
      - fold boundaries are timestamp quantiles over the search region
        (the final HOLDOUT_FRACTION of time is excluded entirely);
      - PURGE: a train row is kept only if its label window (the bar
        max(FORWARD_BARS) steps ahead) completes before the boundary;
      - EMBARGO: validation starts EMBARGO_MULTIPLIER * seq_len hours
        after the boundary (calendar seconds — legacy); under
        TRAINING_REPAIRS_V1 (R2C-04 / L6) the embargo counts DISTINCT
        BARS on the pooled grid instead, so a seq_len=40 stock embargo
        is 40 RTH bars, not the ~11 the 40-calendar-hour rule yields.

    Returns list of (train_indices, val_indices).
    """
    valid = _valid_indices(tickers, ticker_boundaries, seq_len)
    if len(valid) == 0:
        return []

    t = all_times[valid]
    search_mask = t <= get_holdout_boundary(all_times)
    search_times = t[search_mask]
    if len(search_times) < 1000:
        return []

    embargo_seconds = seq_len * EMBARGO_MULTIPLIER * 3600
    _repairs = _training_repairs()
    _l6_gap_h = None  # SIG-R2-3: fold-0 embargo actually applied (print only)
    folds = []
    for fold_idx in range(n_folds):
        train_end_pct = 0.55 + fold_idx * (0.45 / n_folds)
        val_end_pct = train_end_pct + (0.45 / n_folds)

        t_train_end = int(np.quantile(search_times, min(train_end_pct, 1.0)))
        t_val_end = int(np.quantile(search_times, min(val_end_pct, 1.0)))

        # Purge: the LABEL must complete before the boundary, not just the bar
        train_mask = search_mask & (all_label_times[valid] <= t_train_end)
        if _repairs:
            # TRAINING_REPAIRS_V1 (L6): embargo denominated in BARS of the
            # actual pooled bar grid. The legacy calendar-seconds rule
            # shrinks a seq_len=40 stock embargo to ~11 RTH bars (40
            # calendar hours); counting distinct bar timestamps keeps the
            # serial-correlation buffer at seq_len bars on both books
            # (identical to legacy on crypto's continuous hourly grid).
            _val_start = embargo_end_time(search_times, t_train_end,
                                          seq_len * EMBARGO_MULTIPLIER)
            if _l6_gap_h is None:
                _l6_gap_h = (_val_start - t_train_end) / 3600.0
            val_mask = (search_mask
                        & (t >= _val_start)
                        & (t < t_val_end))
        else:
            val_mask = (search_mask
                        & (t >= t_train_end + embargo_seconds)
                        & (t < t_val_end))
        if purge_val_labels:
            # OBJECTIVE_V3: val rows whose label windows cross into the
            # holdout previously leaked holdout returns into checkpoint /
            # threshold selection (confirmed T1 sub-defect).
            val_mask = val_mask & (all_label_times[valid]
                                   <= get_holdout_boundary(all_times))

        train_indices = valid[train_mask]
        val_indices = valid[val_mask]
        if len(train_indices) < 500 or len(val_indices) < 200:
            continue
        folds.append((train_indices, val_indices))

    if _repairs:
        # SIG-R2-3: instrumentation only — prove L6 ran (values untouched)
        print(f"[REPAIRS] L6 embargo in bars: {seq_len * EMBARGO_MULTIPLIER} "
              f"distinct bars (fold-0 val starts +{_l6_gap_h}h after train "
              f"end; legacy calendar rule +{embargo_seconds / 3600:.0f}h); "
              f"folds={len(folds)}")
    return folds


# ---------------------------------------------------------------------------
# Sharpe ratio computation
# ---------------------------------------------------------------------------

# Real Alpaca round-trip costs (fees + typical spread), percent of notional.
# The old constant (5 bps) was ~10x below crypto reality (25 bps taker per
# side + spread), which selected models whose edge cannot survive live fees.
TXN_COST_PCT = {'crypto': 0.60, 'stock': 0.11}
BARS_PER_YEAR = {'crypto': 8760, 'stock': 1638}


def _objective_long_only():
    """Read strategy_config.OBJECTIVE_LONG_ONLY (default False = legacy)."""
    try:
        from strategy_config import OBJECTIVE_LONG_ONLY
        return bool(OBJECTIVE_LONG_ONLY)
    except Exception:
        return False


def _hypersearch_v3():
    """Read strategy_config.HYPERSEARCH_V3 (default False = legacy)."""
    try:
        from strategy_config import HYPERSEARCH_V3
        return bool(HYPERSEARCH_V3)
    except Exception:
        return False


def _objective_v3():
    """Read strategy_config.OBJECTIVE_V3 (default False = legacy)."""
    try:
        from strategy_config import OBJECTIVE_V3
        return bool(OBJECTIVE_V3)
    except Exception:
        return False


def _lgb_refit_full():
    """Read strategy_config.LGB_REFIT_FULL (default False = legacy)."""
    try:
        from strategy_config import LGB_REFIT_FULL
        return bool(LGB_REFIT_FULL)
    except Exception:
        return False


def _training_repairs():
    """Read strategy_config.TRAINING_REPAIRS_V1 (default False = legacy).

    R2C-04 bundle of score-changing training-loop repairs (L1 val-loss
    criterion, L2 regime look-ahead, L5 OOM-probe side effects, L6
    embargo-in-bars). Rides Chain-2's single gotcha-#2 study reset.
    """
    try:
        from strategy_config import TRAINING_REPAIRS_V1
        return bool(TRAINING_REPAIRS_V1)
    except Exception:
        return False


def _holdout_span_by_target():
    """SIG-R1-A3: HOLDOUT_SPAN_BY_TARGET (default False = legacy).

    TRADER_HOLDOUT_SPAN_BY_TARGET env ('1'/'true'/'yes'/'on') wins over
    strategy_config.HOLDOUT_SPAN_BY_TARGET (read with getattr — absent =
    False). ON: a raw-target (target_kind != 'tb') holdout trade's
    calendar window for the n_eff deflation is its fb-bar label window
    instead of the triple-barrier TB_Bars span. Gate-facing (tightens).
    """
    v = os.environ.get('TRADER_HOLDOUT_SPAN_BY_TARGET')
    if v not in (None, ''):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        import strategy_config as _sc
        return bool(getattr(_sc, 'HOLDOUT_SPAN_BY_TARGET', False))
    except Exception:
        return False


def _failed_trial_prune():
    """SIG-R2-1: FAILED_TRIAL_PRUNE (default False = legacy 0.0 sentinel).

    TRADER_FAILED_TRIAL_PRUNE env ('1'/'true'/'yes'/'on') wins over
    strategy_config.FAILED_TRIAL_PRUNE (read with getattr — absent =
    False). ON: a trial that produced no honest score (OOM/RuntimeError,
    no walk-forward folds, zero completed folds, a fold that timed out
    before its first checkpoint) raises optuna.TrialPruned instead of
    returning the 0.0 sentinel — state PRUNED, so it can never be
    study.best_trial, never counts as a COMPLETE (tested) configuration in
    the deflation pool / cum_trials, and TPE files it in the 'above'
    (bad) set. Legacy 0.0 outranked every negative trial (the
    OBJECTIVE_LONG_ONLY regime). Model-facing (changes which trial wins).
    """
    v = os.environ.get('TRADER_FAILED_TRIAL_PRUNE')
    if v not in (None, ''):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        import strategy_config as _sc
        return bool(getattr(_sc, 'FAILED_TRIAL_PRUNE', False))
    except Exception:
        return False


def _prune_failed_trial(trial, reason):
    """SIG-R2-1 ON path: record why, then raise optuna.TrialPruned."""
    print(f"  [FAILED-TRIAL] Trial {trial.number}: {reason} — pruned "
          f"(FAILED_TRIAL_PRUNE: not scored, not in the deflation pool)")
    try:
        trial.set_user_attr('failed_trial', str(reason))
    except Exception:
        pass
    raise optuna.TrialPruned(str(reason))


def _trainer_seed():
    """R2C-04 (L3): base training seed, or None = legacy unseeded.

    TRADER_TRAINER_SEED env (int) wins over strategy_config.TRAINER_SEED;
    both default to None. Sub-seeds derive via objective_utils.derive_seed
    from (study_name, trial.number, fold) so folds / trials / sampler /
    refit never share an RNG stream.
    """
    v = os.environ.get('TRADER_TRAINER_SEED')
    if v not in (None, ''):
        try:
            return int(v)
        except ValueError:
            print(f"[SEED] ignoring non-integer TRADER_TRAINER_SEED={v!r}")
    try:
        from strategy_config import TRAINER_SEED
        return int(TRAINER_SEED) if TRAINER_SEED is not None else None
    except Exception:
        return None


def _objective_session_mask():
    """SIG-R2-MASK: OBJECTIVE_SESSION_MASK (default False = legacy).

    TRADER_OBJECTIVE_SESSION_MASK env ('1'/'true'/'yes'/'on') wins over
    strategy_config.OBJECTIVE_SESSION_MASK (read with getattr — absent =
    False). ON: stock-book LONG entries are scored only on rows the live
    book could enter (_session_entry_ok). Model-facing (fewer eligible
    entries -> different trial scores / n_trades / DSR pool) — flip with
    a gotcha-#2 study reset.
    """
    v = os.environ.get('TRADER_OBJECTIVE_SESSION_MASK')
    if v not in (None, ''):
        return v.strip().lower() in ('1', 'true', 'yes', 'on')
    try:
        import strategy_config as _sc
        return bool(getattr(_sc, 'OBJECTIVE_SESSION_MASK', False))
    except Exception:
        return False


def _session_entry_ok(times, asset_type):
    """SIG-R2-MASK: per-row bool 'the live book may ENTER here', or None.

    None (flag OFF, or any non-stock book — crypto_loop.check_market_hours
    is always True and crypto has no entry window) = callers leave
    long_veto exactly as today. Stock + ON: bar open-time inside
    strategy_config.STOCK_ENTRY_WINDOWS_ET (stock_loop._in_entry_window's
    rule) on a weekday; with ENTRY_WINDOWS_ENABLED False the live gate is
    the market clock alone, mirrored as RTH 09:30-16:00 ET
    (stock_loop.check_market_hours' wall-clock fallback). Scorers pass
    long_veto = ~entry_ok (OR'd with any q10 veto).
    """
    if asset_type != 'stock' or not _objective_session_mask():
        return None
    from strategy_config import STOCK_ENTRY_WINDOWS_ET, ENTRY_WINDOWS_ENABLED
    from objective_utils import session_entry_mask
    windows = (STOCK_ENTRY_WINDOWS_ET if ENTRY_WINDOWS_ENABLED
               else [('09:30', '16:00')])
    return session_entry_mask(times, windows)


def simulate_trades(predictions, actual_returns, threshold, forward_bars,
                    txn_cost_pct, return_entries=False, long_only=None,
                    block_ids=None, long_veto=None):
    """Non-overlapping hold simulation. Returns per-trade net returns.

    The model predicts the fb-bar forward return, so once a position is
    entered it is HELD for fb bars and intervening signals are skipped.
    The old simulator counted every signal bar as an independent trade
    earning the full overlapping fb-bar return — inflating scores by
    ~sqrt(fb) and mechanically favoring the longest horizon.

    long_only (None -> strategy_config.OBJECTIVE_LONG_ONLY): when True the
        short leg is NOT scored. The live book is long-only, so a model whose
        Sharpe is carried by bear-side accuracy can otherwise clear the trial
        score AND the holdout DSR while the deployable long side has ~zero
        expectancy (2026-07 review). Default False preserves the historical
        objective — flipping the flag changes trial scores, which makes old
        Optuna scores incomparable (CLAUDE.md gotcha #2 applies).

    return_entries: also return an int array of the row index (into the
        input arrays) each trade was entered at — lets the holdout gate map
        a trade back to its label span for the effective-n / uniqueness
        deflation (sample_weights.py).

    block_ids / long_veto (both None = legacy, byte-identical): per-row
        ticker-block ids (a hold never spans a block boundary, OBJECTIVE_V3)
        and a boolean long-entry veto mask (q10 tail veto mirror,
        HYPERSEARCH_V3 blend certificate). Delegated to the pure
        objective_utils.simulate_trades_core (Mac-testable).
    """
    if long_only is None:
        long_only = _objective_long_only()
    ret, entries = simulate_trades_core(predictions, actual_returns, threshold,
                                        forward_bars, txn_cost_pct,
                                        long_only=long_only,
                                        block_ids=block_ids,
                                        long_veto=long_veto)
    if return_entries:
        return ret, entries
    return ret


def compute_sharpe(predictions, actual_returns, threshold, forward_bars=24,
                   asset_type='crypto', block_ids=None, long_veto=None):
    """Annualized Sharpe of the non-overlapping hold policy, net of costs.

    Annualization uses the asset class's actual bar count (8760 hourly
    crypto bars/yr vs 1638 stock RTH bars/yr — the old code applied the
    stock constant to 24/7 crypto).
    """
    trade_returns = simulate_trades(predictions, actual_returns, threshold,
                                    forward_bars, TXN_COST_PCT.get(asset_type, 0.6),
                                    block_ids=block_ids, long_veto=long_veto)
    if len(trade_returns) < 10:
        return 0.0
    std = trade_returns.std()
    if std < 1e-8:
        return 0.0
    # BARS_PER_YEAR_MEASURED (default OFF -> this module's dict, the
    # pre-flag expression) — see bars_calendar for the measured stock value.
    try:
        from bars_calendar import bars_per_year as _bars_per_year
        bars_per_year = _bars_per_year(asset_type, BARS_PER_YEAR, 8760)
    except ImportError:
        bars_per_year = BARS_PER_YEAR.get(asset_type, 8760)
    # Each trade occupies forward_bars bars; cap at full investment
    slots_per_year = bars_per_year / forward_bars
    occupancy = min(len(trade_returns) * forward_bars / max(len(predictions), 1), 1.0)
    trades_per_year = occupancy * slots_per_year
    return float((trade_returns.mean() / std) * np.sqrt(max(trades_per_year, 1.0)))


# SIG-R3-DECOMP: per-fold trade decomposition recorded as trial user_attrs
# (one list entry per fold, aligned with fold_sharpes). Instrumentation
# only — the score and every pre-existing user_attr are untouched.
DECOMP_KEYS = ('gross_ret_mean', 'gross_ret_std', 'net_ret_mean', 'cost_drag',
               'n_trades', 'hit_rate', 'mean_hold_bars',
               'threshold_pass_rate', 'n_rows')


def fold_trade_decomposition(predictions, actual_returns, threshold,
                             forward_bars=24, asset_type='crypto',
                             block_ids=None, long_veto=None):
    """Gross/net/cost split of the SAME walk compute_sharpe scores.

    One extra simulate_trades call (same inputs + return_entries). The
    net array is exactly the one compute_sharpe sees. Gross is read, not
    re-simulated: the walk's entry test never reads txn_cost_pct
    (objective_utils.simulate_trades_core), so trade i's gross return is
    +r[e_i] for a long entry (p > threshold and not vetoed) and -r[e_i]
    for a short; net_i = gross_i - cost up to float rounding.
    mean_hold_bars: bars each entry blocks in the walk = min(fb, rows
    left, next ticker block) — the walk holds fb bars for BOTH target
    kinds (a TB label's own exit bar is not consulted).
    Never raises: instrumentation must not change a trial's fate (the
    objective maps RuntimeError to 0.0 and the regime block's except would
    skip the penalty) — any error yields all-None fields + 'error'.
    """
    try:
        return _fold_trade_decomposition(predictions, actual_returns,
                                         threshold, forward_bars, asset_type,
                                         block_ids, long_veto)
    except Exception as e:  # pragma: no cover - defensive
        out = {k: None for k in DECOMP_KEYS}
        out['error'] = f'{type(e).__name__}: {e}'[:200]
        return out


def _fold_trade_decomposition(predictions, actual_returns, threshold,
                              forward_bars, asset_type, block_ids, long_veto):
    cost = TXN_COST_PCT.get(asset_type, 0.6)
    p = np.asarray(predictions)
    r = np.asarray(actual_returns)
    net, entries = simulate_trades(p, r, threshold, forward_bars, cost,
                                   return_entries=True, block_ids=block_ids,
                                   long_veto=long_veto)
    net = np.asarray(net, dtype=np.float64)
    e = np.asarray(entries, dtype=np.int64)
    n = int(len(e))
    out = {'n_trades': n, 'n_rows': int(len(p)),
           'threshold_pass_rate': (float((p > threshold).mean())
                                   if len(p) else None)}
    if n == 0:
        out.update({k: None for k in ('gross_ret_mean', 'gross_ret_std',
                                      'net_ret_mean', 'cost_drag',
                                      'hit_rate', 'mean_hold_bars')})
        return out
    long_ok = p[e] > threshold
    if long_veto is not None:
        long_ok &= ~np.asarray(long_veto, dtype=bool)[e]
    gross = np.where(long_ok, r[e], -r[e]).astype(np.float64)
    end = np.minimum(e + int(forward_bars), len(p))
    if block_ids is not None:
        b = np.asarray(block_ids)
        change = np.flatnonzero(np.diff(b) != 0) + 1
        end = np.minimum(end, np.append(change, len(p))[
            np.searchsorted(change, e, side='right')])
    out.update({
        'gross_ret_mean': float(gross.mean()),
        'gross_ret_std': float(gross.std()),
        'net_ret_mean': float(net.mean()),
        'cost_drag': float(gross.mean() - net.mean()),
        'hit_rate': float((gross > 0).mean()),
        'mean_hold_bars': float((end - e).mean()),
    })
    return out


def compute_regime_sharpes(predictions, actual_returns, threshold,
                           forward_bars=24, asset_type='crypto',
                           block_ids=None, long_veto=None,
                           decomp_out=None):
    """Compute Sharpe in bull/bear/sideways regimes separately.

    long_veto (None = legacy): per-row long-entry veto, subset with each
    regime mask like block_ids (SIG-R2-MASK session mask).

    Regime labels approximate the trailing 50-bar cumulative return from
    the (overlapping) fb-bar forward returns by scaling the trailing mean:
    mean(fb-bar returns over 50 bars) * (50 / fb).

    Returns dict: {'bull': sharpe, 'bear': sharpe, 'sideways': sharpe, 'min': min}
    """
    if len(actual_returns) < 60:
        return {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0, 'min': 0.0}

    if _training_repairs():
        # TRAINING_REPAIRS_V1 (L2): the mask at t uses only returns
        # completed by t (the legacy series embedded returns through t+fb
        # — look-ahead in a selection-shaping penalty) and NaN warmup rows
        # join NO regime instead of polluting them with under-scaled
        # partial sums. Known residual (recorded, out of this repair's
        # scope): boolean-masked subsets concatenate non-adjacent rows, so
        # the fb-bar hold walk still treats them as contiguous.
        rolling_ret = lagged_regime_series(actual_returns, forward_bars,
                                           window=50)
        known = np.isfinite(rolling_ret)
        regimes = {
            'bull': known & (rolling_ret > 2.0),
            'bear': known & (rolling_ret < -2.0),
        }
        regimes['sideways'] = known & ~regimes['bull'] & ~regimes['bear']
        # SIG-R2-3: instrumentation only — prove L2 ran (values untouched)
        print(f"[REPAIRS] L2 regime masks from lagged completed-by-t returns: "
              f"bull={int(regimes['bull'].sum())} "
              f"bear={int(regimes['bear'].sum())} "
              f"sideways={int(regimes['sideways'].sum())} "
              f"warmup_excluded={int((~known).sum())}")
    else:
        finite = np.where(np.isfinite(actual_returns), actual_returns, 0.0)
        window = 50
        kernel = np.ones(window) / window
        trailing_mean = np.convolve(finite, kernel, mode='full')[:len(finite)]
        rolling_ret = trailing_mean * (window / max(forward_bars, 1))

        regimes = {
            'bull': rolling_ret > 2.0,
            'bear': rolling_ret < -2.0,
        }
        regimes['sideways'] = ~regimes['bull'] & ~regimes['bear']

    result = {}
    for name, mask in regimes.items():
        if mask.sum() < 10:
            result[name] = 0.0
            continue
        # Per-row block ids survive boolean subsetting; the core recomputes
        # change-points on the subset.
        result[name] = compute_sharpe(predictions[mask], actual_returns[mask],
                                      threshold, forward_bars, asset_type,
                                      block_ids=(block_ids[mask]
                                                 if block_ids is not None
                                                 else None),
                                      long_veto=(long_veto[mask]
                                                 if long_veto is not None
                                                 else None))
        if decomp_out is not None:  # SIG-R3-DECOMP (instrumentation)
            decomp_out[name] = fold_trade_decomposition(
                predictions[mask], actual_returns[mask], threshold,
                forward_bars, asset_type,
                block_ids=(block_ids[mask] if block_ids is not None
                           else None),
                long_veto=(long_veto[mask] if long_veto is not None
                           else None))

    result['min'] = min(result.values())
    return result


# ---------------------------------------------------------------------------
# Sequence cache
# ---------------------------------------------------------------------------

class ScaledCache:
    """Cache ONE fold's scaled 2-D feature matrix + scaler at a time.

    The old SeqCache materialized full (N, seq_len, F) sequence arrays —
    ~1.0GB of host numpy for the crypto fold-3 train split, UNPROTECTED by
    the 40% CUDA allocator cap. During Saturday retrains that risked the
    kernel OOM-killer SIGKILLing training (uncatchable). We now keep only
    the scaled matrix (~25-45MB) and gather each batch's windows on the
    fly (~8MB per batch) — same per-batch copy cost as before.
    """
    def __init__(self, all_features):
        self._all_features = all_features
        self._key = None
        self._all_scaled = None
        self._scaler = None

    def get(self, fold_idx, train_indices):
        """Return (all_scaled, scaler), fitting the scaler on train rows only."""
        n = len(train_indices)
        key = (fold_idx, n,
               int(train_indices[0]) if n else -1,
               int(train_indices[-1]) if n else -1)
        if key == self._key and self._all_scaled is not None:
            return self._all_scaled, self._scaler

        self._all_scaled = None
        self._scaler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        scaler = RobustScaler()
        scaler.fit(self._all_features[train_indices])
        all_scaled = scaler.transform(self._all_features).astype(np.float32)

        print(f"  [CACHE] fold={fold_idx}: scaler fit on {n} train rows "
              f"({all_scaled.nbytes / 1e6:.0f} MB scaled matrix, "
              f"{time.time() - t0:.1f}s)")

        self._key = key
        self._all_scaled = all_scaled
        self._scaler = scaler
        return all_scaled, scaler


def gather_windows(all_scaled, indices, offsets):
    """Build (len(indices), seq_len, F) windows for one batch on the fly."""
    return all_scaled[indices[:, None] + offsets[None, :]]


def average_states(states: list[dict]) -> dict:
    """Uniform weight soup over checkpoint state_dicts (same architecture,
    same run). Non-float buffers (counters) take the first checkpoint's
    value; float tensors are element-wise averaged."""
    avg = {}
    for k in states[0]:
        v0 = states[0][k]
        if torch.is_floating_point(v0):
            avg[k] = torch.stack([s[k].float() for s in states]).mean(0).to(v0.dtype)
        else:
            avg[k] = v0.clone()
    return avg


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(all_features, all_returns_by_fb, all_times, all_label_times,
                     tickers, ticker_boundaries,
                     input_dim, _state_cache, asset_type='crypto',
                     has_multi_horizon=True, adaptive_space=None,
                     study_name=''):

    MAX_TRIAL_SECONDS = 900

    seq_cache = ScaledCache(all_features)

    # Use adaptive search space if provided, otherwise use defaults
    _space = adaptive_space or {}

    # SIG-R2-MASK (OBJECTIVE_SESSION_MASK, default OFF -> None, legacy):
    # built once per data load; every trial's fold / pruning / regime
    # scorer vetoes LONG entries on rows the live stock book cannot enter.
    _entry_ok = _session_entry_ok(all_times, asset_type)
    if _entry_ok is not None:
        print(f"[SESSION] OBJECTIVE_SESSION_MASK ON: {int(_entry_ok.sum())}/"
              f"{len(_entry_ok)} rows ({_entry_ok.mean():.1%}) entry-"
              f"eligible (stock entry windows, bar open-time ET)")

    def objective(trial):
        trial_start = time.time()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Hyperparameters (from adaptive search space) ---
        if has_multi_horizon:
            fb_choices = _space.get('forward_bars', FORWARD_BARS)
            forward_bars = trial.suggest_categorical('forward_bars', fb_choices)
        else:
            forward_bars = 4

        sl_range = _space.get('seq_len', [8, 40])
        seq_len = trial.suggest_int('seq_len', sl_range[0], sl_range[-1], step=2)
        hd_range_vals = _space.get('hidden_dim', [64, 384])
        hidden_dim = trial.suggest_int('hidden_dim', hd_range_vals[0], hd_range_vals[-1], step=32)
        nl_range = _space.get('num_layers', [1, 2])
        num_layers = trial.suggest_int('num_layers', nl_range[0], nl_range[-1])
        n_heads = trial.suggest_categorical('n_heads',
            _space.get('n_heads', [2, 4]))
        dr_range = _space.get('dropout', [0.10, 0.40])
        dropout = trial.suggest_float('dropout', dr_range[0], dr_range[1], step=0.05)
        lr_range = _space.get('learning_rate', [5e-4, 3e-3])
        learning_rate = trial.suggest_float('learning_rate', lr_range[0], lr_range[1], log=True)
        batch_size = trial.suggest_categorical('batch_size',
            _space.get('batch_size', [512, 1024, 2048]))
        wd_range = _space.get('weight_decay', [1e-5, 5e-4])
        weight_decay = trial.suggest_float('weight_decay', wd_range[0], wd_range[1], log=True)
        hd_range = _space.get('huber_delta', [0.5, 2.0])
        huber_delta = trial.suggest_float('huber_delta', hd_range[0], hd_range[1], step=0.1)
        tt_range = _space.get('trade_threshold', [0.05, 1.0])
        if _objective_v3():
            # D05-threshold: floor-anchored to the book's deployment edge;
            # the adaptive state's own range/edge-expansion is overridden
            # while the flag is ON (runtime-only — DEFAULT_SEARCH_SPACE
            # stays untouched).
            tt_range = v3_trade_threshold_range(asset_type)
        trade_threshold = trial.suggest_float('trade_threshold', tt_range[0], tt_range[1], step=0.01)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau'])

        # Target kind: raw fb-bar forward return vs triple-barrier
        # (exit-stack-matched) return. When TB labels exist, the search
        # itself decides which target produces the better POLICY Sharpe —
        # both flow through the same cost-aware simulator and gates.
        if ('tb', forward_bars) in all_returns_by_fb:
            target_kind = trial.suggest_categorical('target_kind', ['raw', 'tb'])
        else:
            target_kind = 'raw'

        cfg = {
            'seq_len': seq_len, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'n_heads': n_heads,
            'dropout': dropout, 'learning_rate': learning_rate,
            'batch_size': batch_size, 'weight_decay': weight_decay,
            'huber_delta': huber_delta, 'trade_threshold': trade_threshold,
            'scheduler': scheduler, 'forward_bars': forward_bars,
            'target_kind': target_kind,
        }
        trial.set_user_attr('cfg', cfg)

        # Select returns for this horizon + target kind
        key = ('tb', forward_bars) if target_kind == 'tb' else forward_bars
        if key in all_returns_by_fb:
            trial_returns = all_returns_by_fb[key]
        elif forward_bars in all_returns_by_fb:
            trial_returns = all_returns_by_fb[forward_bars]
        else:
            trial_returns = next(v for k, v in all_returns_by_fb.items()
                                 if not isinstance(k, tuple))

        try:
            return _train_walk_forward(
                trial, trial_start, cfg, trial_returns, input_dim, seq_cache,
                _state_cache,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # CUDA context recovers after NvMap/NVML errors — don't kill the
            # process. Clean up, return 0 (bad score), Optuna moves on.
            print(f"  [OOM] Trial {trial.number}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            torch.cuda.empty_cache()
            if _failed_trial_prune():
                _prune_failed_trial(
                    trial, f"{type(e).__name__}: {str(e)[:120]}")
            return 0.0

    def _train_walk_forward(trial, trial_start, cfg, trial_returns, input_dim,
                            seq_cache, _state_cache):
        seq_len = cfg['seq_len']
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        n_heads = cfg['n_heads']
        dropout = cfg['dropout']
        learning_rate = cfg['learning_rate']
        batch_size = cfg['batch_size']
        weight_decay = cfg['weight_decay']
        huber_delta = cfg['huber_delta']
        trade_threshold = cfg['trade_threshold']
        scheduler_type = cfg['scheduler']

        folds = get_walk_forward_folds(all_times, all_label_times, tickers,
                                       ticker_boundaries, seq_len,
                                       purge_val_labels=_objective_v3())
        if not folds:
            if _failed_trial_prune():
                _prune_failed_trial(trial, 'no walk-forward folds')
            return 0.0

        _repairs = _training_repairs()  # R2C-04 TRAINING_REPAIRS_V1 (L1/L5)
        _repairs_logged = set()  # SIG-R2-3: one [REPAIRS] line per trial
        base_seed = _trainer_seed()     # R2C-04 TRAINER_SEED (L3)

        offsets = np.arange(-seq_len, 0)
        fold_sharpes = []
        # per-fold epochs TRAINED at the best-val-loss checkpoint (a 1-based
        # COUNT: loop index + 1 — SIG-R1-A4; final_refit trains exactly this
        # many epochs, like fixed_boost_rounds' LightGBM best_iteration)
        fold_best_epochs = []
        fold_decomp = {k: [] for k in DECOMP_KEYS}  # SIG-R3-DECOMP
        oof_fold_rows, oof_fold_preds, oof_fold_ids = [], [], []
        best_fold_state = None
        best_fold_scaler = None
        best_fold_val = None  # (val_indices, scaler) of the winning fold
        best_fold_sharpe = -999

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            # Filter out indices where the target return is NaN
            # (happens at the end of each ticker series for larger forward_bars)
            train_indices = train_indices[~np.isnan(trial_returns[train_indices])]
            val_indices = val_indices[~np.isnan(trial_returns[val_indices])]
            if len(train_indices) < 500 or len(val_indices) < 100:
                continue
            # OBJECTIVE_V3: per-row ticker-block ids so a simulated hold
            # never spans a ticker boundary (None = legacy scoring).
            vb = (ticker_block_ids(val_indices, ticker_boundaries)
                  if _objective_v3() else None)
            sv = (~_entry_ok[val_indices] if _entry_ok is not None
                  else None)  # SIG-R2-MASK long veto (None = legacy)

            all_scaled, scaler = seq_cache.get(fold_idx, train_indices)
            n_train = len(train_indices)
            n_val = len(val_indices)

            y_train = trial_returns[train_indices]
            y_val = trial_returns[val_indices]

            # R2C-04 (L3): per-fold derived seed; None = legacy unseeded.
            # Seeding sits INSIDE the retry loop so an OOM retry replays
            # the same init/dropout stream instead of continuing it.
            fold_seed = None
            fold_rng = None
            if base_seed is not None:
                fold_seed = derive_seed(base_seed, study_name, trial.number,
                                        fold_idx)
                fold_rng = np.random.default_rng(fold_seed)

            # OOM retry: if batch doesn't fit, halve until it does (min 128)
            eff_batch_size = batch_size
            oom_retries = 0
            while True:
                try:
                    if fold_seed is not None:
                        torch.manual_seed(fold_seed)
                    model = RegressionLSTM(input_dim, hidden_dim, num_layers,
                                           dropout, n_heads).to(device)
                    criterion = nn.HuberLoss(delta=huber_delta, reduction='none')
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                                           weight_decay=weight_decay)

                    if scheduler_type == 'cosine':
                        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
                    elif scheduler_type == 'plateau':
                        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
                    else:
                        sched = None

                    # FP16 mixed precision
                    use_amp = device.type == 'cuda'
                    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

                    if _repairs:
                        # TRAINING_REPAIRS_V1 (L5): snapshot pristine init
                        # so the memory probe leaves no trace (legacy: the
                        # probe's unclipped optimizer step trained the
                        # model before epoch 0).
                        _init_snap = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}

                    # Run one batch to test if this config fits in memory
                    model.train()
                    test_idx = train_indices[:min(eff_batch_size, n_train)]
                    xb = torch.from_numpy(gather_windows(all_scaled, test_idx, offsets)).to(device)
                    yb = torch.from_numpy(y_train[:len(test_idx)]).to(device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        pred = model(xb)
                        raw_loss = criterion(pred, yb)
                        weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                        loss = (raw_loss * weights).mean()
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    del xb, yb, pred, raw_loss, weights, loss
                    if _repairs:
                        # L5: restore init weights + fresh optimizer /
                        # scheduler / grad-scaler — epoch 0 starts clean.
                        model.load_state_dict(_init_snap)
                        optimizer = optim.Adam(model.parameters(),
                                               lr=learning_rate,
                                               weight_decay=weight_decay)
                        if scheduler_type == 'cosine':
                            sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer, T_0=20, T_mult=2)
                        elif scheduler_type == 'plateau':
                            sched = optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer, patience=6, factor=0.5)
                        grad_scaler = torch.amp.GradScaler('cuda',
                                                           enabled=use_amp)
                        del _init_snap
                    break  # fits in memory
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # CUDA context recovers after NvMap/INTERNAL ASSERT errors
                    # on Jetson — halve batch and retry instead of giving up
                    try:
                        del model, criterion, optimizer
                    except NameError:
                        pass
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    oom_retries += 1
                    eff_batch_size //= 2
                    if eff_batch_size < 128:
                        raise  # give up, let outer handler catch it
                    print(f"  [OOM-RETRY] fold {fold_idx}: batch {eff_batch_size*2}→{eff_batch_size}"
                          f" ({str(e)[:60]})")

            if oom_retries > 0:
                print(f"  [OOM-RETRY] fold {fold_idx}: training with batch_size={eff_batch_size} "
                      f"(was {batch_size})")
            if _repairs and 'L5' not in _repairs_logged:
                _repairs_logged.add('L5')  # SIG-R2-3: print only
                print(f"[REPAIRS] L5 trial {trial.number} fold {fold_idx}: "
                      f"memory probe undone (init weights restored; fresh "
                      f"optimizer/scheduler/grad-scaler)")

            best_val_loss = float('inf')
            best_epoch = -1
            best_state = None
            top_states: list[tuple[float, dict]] = []  # K-best checkpoint soup
            counter = 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                # R2C-04 (L3): fold_rng None = legacy ambient-RNG shuffle
                perm = (fold_rng.permutation(n_train)
                        if fold_rng is not None
                        else np.random.permutation(n_train))
                for i in range(0, n_train, eff_batch_size):
                    bi = perm[i:i + eff_batch_size]
                    # Gather this batch's windows on the fly (~8MB) instead
                    # of holding the whole fold as sequences (~1GB)
                    xb = torch.from_numpy(
                        gather_windows(all_scaled, train_indices[bi], offsets)).to(device)
                    yb = torch.from_numpy(y_train[bi]).to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        pred = model(xb)
                        raw_loss = criterion(pred, yb)
                        # Weight by |actual_return| + 1.0 (emphasize big moves)
                        weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                        loss = (raw_loss * weights).mean()

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                if scheduler_type == 'cosine' and sched:
                    sched.step()

                # Validation
                model.eval()
                val_loss_sum = 0.0
                val_preds = []
                with torch.inference_mode():
                    for i in range(0, n_val, eff_batch_size):
                        vidx = val_indices[i:i + eff_batch_size]
                        xvb = torch.from_numpy(
                            gather_windows(all_scaled, vidx, offsets)).to(device)
                        yvb = torch.from_numpy(y_val[i:i + eff_batch_size]).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model(xvb)
                        if _repairs:
                            # TRAINING_REPAIRS_V1 (L1): score checkpoints
                            # with the TRIAL'S criterion (huber_delta +
                            # |return|+1 weights). Legacy used default
                            # huber (delta=1.0, unweighted): soup
                            # admission, early stopping and the refit
                            # epoch budget selected on a mismatched loss
                            # and huber_delta barely influenced selection.
                            v_raw = criterion(vo, yvb)
                            v_w = torch.clamp(torch.abs(yvb) + 1.0,
                                              max=50.0)
                            val_loss_sum += (v_raw * v_w).sum().item()
                        else:
                            val_loss_sum += nn.functional.huber_loss(vo, yvb).item() * xvb.size(0)
                        val_preds.append(vo.cpu().numpy())

                val_loss = val_loss_sum / n_val
                val_preds_np = np.concatenate(val_preds)
                if _repairs and 'L1' not in _repairs_logged:
                    _repairs_logged.add('L1')  # SIG-R2-3: print only
                    print(f"[REPAIRS] L1 trial {trial.number} fold {fold_idx}: "
                          f"val loss on the trial criterion (huber_delta="
                          f"{huber_delta}, |y|+1 weights cap 50) "
                          f"epoch0 val_loss={val_loss:.6g}")

                if scheduler_type == 'plateau' and sched:
                    sched.step(val_loss)

                # Compute Sharpe for pruning feedback. Step must be
                # MONOTONIC within a trial (fold-major), otherwise Optuna's
                # last_step freezes at fold-0 values and pruning goes inert
                # for later folds.
                epoch_sharpe = compute_sharpe(val_preds_np, y_val, trade_threshold,
                                              forward_bars=cfg['forward_bars'],
                                              asset_type=asset_type,
                                              block_ids=vb, long_veto=sv)
                trial.report(epoch_sharpe, fold_idx * MAX_EPOCHS + epoch)

                if epoch >= PRUNE_WARMUP_EPOCHS and trial.should_prune():
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

                # Timeout
                if time.time() - trial_start > MAX_TRIAL_SECONDS:
                    print(f"  [TIMEOUT] Trial {trial.number} at epoch {epoch} fold {fold_idx}")
                    break

                # K-best checkpoint pool for the weight soup. clone() is
                # load-bearing: on CPU training, .cpu() returns a VIEW of
                # the live weights and later epochs would mutate the
                # "snapshot" in place.
                if len(top_states) < SOUP_K or val_loss < top_states[-1][0]:
                    snap = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
                    top_states.append((val_loss, snap))
                    top_states.sort(key=lambda t: t[0])
                    del top_states[SOUP_K:]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    counter = 0
                else:
                    counter += 1
                    if counter >= EARLY_STOP_PATIENCE:
                        break

            # Checkpoint soup (SWA-style): uniform weight average of the
            # K best-val-loss epochs. Within one run the checkpoints share
            # a loss basin, and the average sits in a flatter region than
            # any single epoch (Izmailov et al. 2018; Wortsman et al.
            # 2022) — fold Sharpe, threshold selection, the holdout gate
            # and the saved artifact all flow from this averaged state.
            if top_states:
                best_state = average_states([s for _, s in top_states])

            # Compute fold Sharpe with best (souped) model
            if best_state is not None:
                model.load_state_dict(best_state)
                model.eval()
                final_preds = []
                with torch.inference_mode():
                    for i in range(0, n_val, eff_batch_size):
                        vidx = val_indices[i:i + eff_batch_size]
                        xvb = torch.from_numpy(
                            gather_windows(all_scaled, vidx, offsets)).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model(xvb)
                        final_preds.append(vo.cpu().numpy())
                final_preds_np = np.concatenate(final_preds)
                fold_sharpe = compute_sharpe(final_preds_np, y_val, trade_threshold,
                                             forward_bars=cfg['forward_bars'],
                                             asset_type=asset_type,
                                             block_ids=vb, long_veto=sv)
                _fd = fold_trade_decomposition(
                    final_preds_np, y_val, trade_threshold,
                    forward_bars=cfg['forward_bars'], asset_type=asset_type,
                    block_ids=vb, long_veto=sv)
                for _k in DECOMP_KEYS:
                    fold_decomp[_k].append(_fd[_k])
                # D12/B04.1: keep this fold's honest val predictions (souped
                # model, its own purged val slice) so the winner's OOF preds
                # can be persisted at save time.
                oof_fold_rows.append(val_indices.copy())
                oof_fold_preds.append(final_preds_np.astype(np.float32))
                oof_fold_ids.append(fold_idx)
                fold_best_epochs.append(best_epoch + 1)  # -1 (none) -> 0
            else:
                if _failed_trial_prune():
                    # Only reachable when the loop broke before epoch 0's
                    # snapshot — the MAX_TRIAL_SECONDS timeout (a prune
                    # raises; early stopping needs >= 1 epoch).
                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _prune_failed_trial(
                        trial, f'fold {fold_idx} timed out before its '
                               f'first checkpoint')
                fold_sharpe = 0.0
                fold_best_epochs.append(best_epoch + 1)
                for _k in DECOMP_KEYS:  # SIG-R3-DECOMP: keep lists aligned
                    fold_decomp[_k].append(
                        {'n_trades': 0, 'n_rows': int(n_val)}.get(_k))

            fold_sharpes.append(fold_sharpe)

            # Track best fold for saving
            if fold_sharpe > best_fold_sharpe:
                best_fold_sharpe = fold_sharpe
                best_fold_state = best_state
                best_fold_scaler = scaler
                best_fold_val = (val_indices, y_val)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not fold_sharpes and _failed_trial_prune():
            _prune_failed_trial(trial, 'zero completed folds (every fold '
                                       'below the row minimum)')
        avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
        std_sharpe = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0.0
        # Risk-adjusted score: penalize inconsistency across folds.
        # A model with [2.8, 2.9, 2.7] (score=2.80) beats [4.0, 4.0, -1.0] (score=2.83)
        score = avg_sharpe - 0.5 * std_sharpe

        # Regime-aware penalty: penalize models with negative Sharpe in any
        # regime — evaluated on the WINNING fold's own validation slice
        # with that fold's scaler (the old code reused loop-leaked state
        # from whatever fold happened to run last).
        if best_fold_state is not None and best_fold_val is not None \
                and len(best_fold_val[0]) > 60:
            model_tmp = None
            try:
                rg_val_indices, rg_y_val = best_fold_val
                rg_scaled = best_fold_scaler.transform(all_features).astype(np.float32)
                model_tmp = RegressionLSTM(input_dim, hidden_dim, num_layers,
                                            dropout, n_heads).to(device)
                model_tmp.load_state_dict(best_fold_state)
                model_tmp.eval()
                all_preds = []
                use_amp = device.type == 'cuda'
                with torch.inference_mode():
                    for i in range(0, len(rg_val_indices), 1024):
                        vidx = rg_val_indices[i:i + 1024]
                        xvb = torch.from_numpy(
                            gather_windows(rg_scaled, vidx, offsets)).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model_tmp(xvb)
                        all_preds.append(vo.cpu().numpy())
                all_preds_np = np.concatenate(all_preds)
                _rg_decomp = {}  # SIG-R3-DECOMP
                regime_sharpes = compute_regime_sharpes(
                    all_preds_np, rg_y_val, trade_threshold,
                    forward_bars=cfg['forward_bars'], asset_type=asset_type,
                    block_ids=(ticker_block_ids(rg_val_indices,
                                                ticker_boundaries)
                               if _objective_v3() else None),
                    long_veto=(~_entry_ok[rg_val_indices]
                               if _entry_ok is not None else None),
                    decomp_out=_rg_decomp)
                trial.set_user_attr('regime_sharpes', regime_sharpes)
                try:
                    trial.set_user_attr('regime_trade_decomp', _rg_decomp)
                except Exception:
                    pass
                # Penalize if any regime has negative Sharpe
                if regime_sharpes['min'] < -0.5:
                    if _repairs:
                        # SIG-R1-A2 (TRAINING_REPAIRS_V1): sign-correct
                        # 30% penalty — identical to *0.7 for score > 0,
                        # but a negative score gets WORSE (legacy *0.7
                        # moved it toward 0, i.e. rewarded the bad regime).
                        score -= 0.3 * abs(score)
                    else:
                        score *= 0.7  # 30% penalty
                del rg_scaled
            except Exception as e:
                print(f"  [REGIME] Penalty eval failed: {e}")
            finally:
                del model_tmp
                gc.collect()

        trial.set_user_attr('fold_sharpes', fold_sharpes)
        trial.set_user_attr('avg_sharpe', avg_sharpe)
        trial.set_user_attr('std_sharpe', std_sharpe)
        for _k in DECOMP_KEYS:  # SIG-R3-DECOMP (lists aligned w/ fold_sharpes)
            trial.set_user_attr(_k, fold_decomp[_k])

        if best_fold_state is not None and score > 0:
            # Only keep this trial's state; clear old entries to avoid memory leak
            _state_cache.clear()
            _state_cache[trial.number] = {
                'state': best_fold_state,
                'scaler': best_fold_scaler,
                # D12: fold-val OOF preds for the save-time npz (~3 fold
                # arrays for the single current-best trial; the clear()
                # above bounds the memory).
                'oof_rows': oof_fold_rows,
                'oof_preds': oof_fold_preds,
                'oof_fold_ids': oof_fold_ids,
                # T1: per-fold best epochs — the final refit's fixed budget
                'fold_best_epochs': fold_best_epochs,
            }

        return score

    return objective


# ---------------------------------------------------------------------------
# LightGBM ensemble training
# ---------------------------------------------------------------------------

LGB_MAX_ROWS = 120_000  # absolute row ceiling (never raised by the byte cap)
# X_train byte budget for the flattened window matrix. The old fixed 120k-row
# cap was written against 24x23 float32 cols (~250MB); at the 72-feature
# stationary preset with seq_len 40 the same rows are ~1.4GB PLUS LightGBM's
# Dataset construction copy — the largest host allocation in the pipeline,
# uncapped by the CUDA fraction and colliding with the systemd MemoryMax=6G
# ceiling. Rows are now capped to whichever is smaller: 120k or what fits in
# this budget at the winning config's actual (seq_len x n_features).
LGB_X_BYTE_BUDGET = 600_000_000


def train_lgb_ensemble(prefix, scaler, cfg, all_features, all_returns_by_fb,
                       all_times, all_label_times, tickers, ticker_boundaries,
                       all_tb_bars_by_fb=None, save=True):
    """Train the LightGBM leg of the ensemble on the winning config.

    Tree ensembles are the stronger learner at this data size (Grinsztajn
    et al. 2022; every top-50 M5 entry used LightGBM) — model_lgb.py was
    fully built but never trained, so predict_now's ensemble path always
    fell back to LSTM-only. Trains on the same scaled features, same
    horizon, same time split (holdout excluded); predict_now's
    flatten_sequence ordering == windows.reshape(-1).

    save=True (legacy, byte-identical): write the boosters to disk and
    return the mean booster. save=False (HYPERSEARCH_V3 pre-gate path):
    write NOTHING and return (booster, q10, q10_floor, n_q10_val) — the
    caller routes the writes through save_model_atomically's
    extra_artifacts AFTER the holdout gate passes. Total failure returns
    None regardless of save.

    LGB_REFIT_FULL (R2C-03, default OFF = byte-identical): the fold
    training above becomes round-count discovery only; both legs are
    RETRAINED on all purged pre-holdout rows at the fixed round count
    (no early stopping) and the q10 veto floor is recomputed from the
    refit q10 on the original fold-val rows (in-sample caveat recorded
    in the q10 meta via the booster's _r2c03_meta_extra attribute, which
    travels through the unchanged save=False tuple).
    """
    try:
        from model_lgb import train_lgb, save_lgb_model
    except ImportError:
        print("[LGB] lightgbm not installed — skipping ensemble leg")
        return None
    try:
        seq_len = cfg['seq_len']
        fb = cfg.get('forward_bars', 24)
        key = ('tb', fb) if cfg.get('target_kind') == 'tb' else fb
        returns = all_returns_by_fb.get(key)
        if returns is None:
            returns = all_returns_by_fb.get(fb)
        if returns is None:
            returns = next(v for k, v in all_returns_by_fb.items()
                           if not isinstance(k, tuple))

        _purge = _objective_v3()
        if _hypersearch_v3() and not _purge:
            # M4 guard (R2C-02b): this fold's val slice calibrates the q10
            # veto floor and (upstream) anchors the blend fit while the
            # certificate scores the holdout — HYPERSEARCH_V3 without
            # OBJECTIVE_V3 must still purge val rows whose label windows
            # cross the holdout boundary (code-couples the runbook's
            # "flip both flags together" pairing instead of trusting it).
            _purge = True
            print("[LGB] M4 guard: HYPERSEARCH_V3 without OBJECTIVE_V3 — "
                  "forcing purge_val_labels=True for the LGB fold build")
        folds = get_walk_forward_folds(all_times, all_label_times, tickers,
                                       ticker_boundaries, seq_len,
                                       purge_val_labels=_purge)
        if not folds:
            return None
        train_idx, val_idx = folds[-1]  # largest train window, latest val
        train_idx = train_idx[~np.isnan(returns[train_idx])]
        val_idx = val_idx[~np.isnan(returns[val_idx])]
        row_bytes = int(seq_len) * int(all_features.shape[1]) * 4  # float32
        max_rows = int(min(LGB_MAX_ROWS,
                           max(20_000, LGB_X_BYTE_BUDGET // max(row_bytes, 1))))
        if len(train_idx) > max_rows:  # most recent rows
            print(f"[LGB] row cap {max_rows} "
                  f"({row_bytes} B/row at seq {seq_len} x "
                  f"{all_features.shape[1]} feats; budget "
                  f"{LGB_X_BYTE_BUDGET // 1_000_000} MB)")
            train_idx = train_idx[np.argsort(all_times[train_idx])][-max_rows:]
        if len(val_idx) > 30_000:
            val_idx = val_idx[np.argsort(all_times[val_idx])][-30_000:]

        all_scaled = scaler.transform(all_features).astype(np.float32)
        offsets = np.arange(-seq_len, 0)
        X_train = gather_windows(all_scaled, train_idx, offsets).reshape(len(train_idx), -1)
        X_val = gather_windows(all_scaled, val_idx, offsets).reshape(len(val_idx), -1)
        y_train, y_val = returns[train_idx], returns[val_idx]

        # Average-uniqueness training weights (wave-8 #1): the DSR gate already
        # deflates by effective-n, but the loss still over-counts overlapping
        # hourly labels ~k times. Weights are aligned to the FINAL train_idx/
        # val_idx (after the NaN filter + most-recent truncation above). OFF by
        # default; the SAME mean-1 vector feeds the mean and q10 legs so the
        # quantile leg stays calibrated. NEVER blend uniqueness x |return| here.
        w_train = w_val = None
        try:
            from strategy_config import UNIQUENESS_WEIGHTS_ENABLED
        except Exception:
            UNIQUENESS_WEIGHTS_ENABLED = False
        if UNIQUENESS_WEIGHTS_ENABLED and all_tb_bars_by_fb:
            tb_spans = all_tb_bars_by_fb.get(fb)
            if tb_spans is not None:
                from sample_weights import fold_train_weights
                w_train = fold_train_weights(tb_spans, train_idx, ticker_boundaries)
                w_val = fold_train_weights(tb_spans, val_idx, ticker_boundaries)
                print(f"[LGB] uniqueness weighting ON "
                      f"(train mean-1 over {int((w_train > 0).sum())} of "
                      f"{len(w_train)} rows)")

        print(f"[LGB] Training on {X_train.shape[0]} rows x {X_train.shape[1]} "
              f"flattened features (fb={fb})")
        booster = train_lgb(X_train, y_train, X_val, y_val,
                            sample_weight=w_train, sample_weight_val=w_val)
        # R2C-03 (M3/M4-floor): under default-OFF LGB_REFIT_FULL the fold
        # training above only supplies the fixed round count — the mean
        # save is deferred until after the full refit below.
        _refit_full = _lgb_refit_full()
        _mean_saved = False
        if save and not _refit_full:
            save_lgb_model(booster, prefix=prefix.rstrip('_'))
            _mean_saved = True

        # Left-tail (q10) quantile model for the entry tail veto: a
        # bullish MEAN prediction can coexist with a fat left tail; the
        # 10th-percentile regression flags those states. The veto floor
        # is self-calibrating — the 15th percentile of q10 over the
        # validation slice (worst ~15% of tail states get vetoed).
        q10 = None
        q10_floor_val = None
        n_q10_val = None
        try:
            q10 = train_lgb(X_train, y_train, X_val, y_val,
                            params={'objective': 'quantile', 'alpha': 0.10,
                                    'metric': 'quantile'},
                            sample_weight=w_train, sample_weight_val=w_val)

            # --- R2C-03 LGB_REFIT_FULL (default OFF -> block skipped) ---
            # M3: D22's recency fix is asymmetric — the LSTM final-refits
            # on ALL purged pre-holdout data while both LGB legs stop at
            # folds[-1] train (ending at the 0.85 search-region quantile),
            # shipping blind to the newest ~15% every retrain. Collective-
            # early-stopping analog: the fold trainings above supply the
            # round counts; both legs RETRAIN on final_refit's purge
            # (label window completes on/before the holdout boundary,
            # NaN-filtered) under the SAME LGB_MAX_ROWS/byte-budget
            # most-recent-first cap, at those FIXED round counts with no
            # early stopping (num_iterations in params overrides
            # train_lgb's num_boost_round; no val set -> no early-stopping
            # callback). Fail-soft: any refit failure falls back loudly
            # to the fold boosters (the legacy artifact).
            if _refit_full:
                try:
                    _r_mean = fixed_boost_rounds(
                        getattr(booster, 'best_iteration', None),
                        booster.current_iteration())
                    _r_q10 = fixed_boost_rounds(
                        getattr(q10, 'best_iteration', None),
                        q10.current_iteration())
                    _rf_idx = lgb_refit_indices(
                        _valid_indices(tickers, ticker_boundaries,
                                       seq_len),
                        all_label_times, all_times, returns,
                        get_holdout_boundary(all_times), max_rows)
                    if _r_mean and _r_q10 and len(_rf_idx) >= 500:
                        w_refit = None
                        if w_train is not None:
                            from sample_weights import fold_train_weights
                            w_refit = fold_train_weights(
                                all_tb_bars_by_fb.get(fb), _rf_idx,
                                ticker_boundaries)
                        del X_train
                        X_train = None  # keeps the shared cleanup valid
                        gc.collect()
                        X_refit = gather_windows(
                            all_scaled, _rf_idx,
                            offsets).reshape(len(_rf_idx), -1)
                        y_refit = returns[_rf_idx]
                        print(f"[LGB-REFIT] full refit on {len(_rf_idx)} "
                              f"purged pre-holdout rows at fixed rounds "
                              f"mean={_r_mean} q10={_r_q10}")
                        _new_mean = train_lgb(
                            X_refit, y_refit,
                            params={'num_iterations': _r_mean},
                            sample_weight=w_refit)
                        _new_q10 = train_lgb(
                            X_refit, y_refit,
                            params={'objective': 'quantile',
                                    'alpha': 0.10, 'metric': 'quantile',
                                    'num_iterations': _r_q10},
                            sample_weight=w_refit)
                        del X_refit
                        gc.collect()
                        # Both-or-nothing rebind: a q10-refit failure
                        # raises BEFORE this line, so the except path
                        # below can never ship a refit mean beside a
                        # fold q10 — the fold pair stays intact unless
                        # BOTH refits succeeded.
                        booster, q10 = _new_mean, _new_q10
                        # M4-floor: the floor below is recomputed from
                        # the REFIT q10 on the original fold-val rows —
                        # rows INSIDE the refit train window, so the
                        # floor is in-sample; the caveat rides the q10
                        # meta json via _r2c03_meta_extra (both save
                        # paths). NOTE: q10 holdout coverage has NO
                        # tool yet — scripts/reliability_report.py is the
                        # META-calibration Brier/ECE report — so the
                        # 10% +/- 3pp check is an open Jetson item.
                        q10._r2c03_meta_extra = {
                            'refit_full': True,
                            'floor_source': 'refit_q10_on_fold_val',
                            'floor_in_sample': True,
                            'refit_rows': int(len(_rf_idx)),
                            'fixed_rounds_mean': int(_r_mean),
                            'fixed_rounds_q10': int(_r_q10)}
                    else:
                        print(f"[LGB-REFIT] skipped (rounds "
                              f"mean={_r_mean} q10={_r_q10}, "
                              f"{len(_rf_idx)} usable rows) — shipping "
                              f"the fold boosters")
                except Exception as e:
                    print(f"[LGB-REFIT] full refit failed (non-fatal) — "
                          f"shipping the fold boosters: {e}")
            if save and _refit_full:
                save_lgb_model(booster, prefix=prefix.rstrip('_'))
                _mean_saved = True

            q10_val = q10.predict(X_val)
            floor = float(np.percentile(q10_val, 15))
            q10_floor_val = floor
            n_q10_val = int(len(q10_val))
            if save:
                q10.save_model(f'{prefix}lgb_q10.txt')
                _meta = {'alpha': 0.10, 'floor': round(floor, 6),
                         'val_rows': int(len(q10_val))}
                _meta.update(getattr(q10, '_r2c03_meta_extra', None) or {})
                with open(f'{prefix}lgb_q10_meta.json', 'w') as f:
                    json.dump(_meta, f)
                print(f"[LGB-Q10] tail model saved (veto floor {floor:+.4f}%)")
            else:
                print(f"[LGB-Q10] tail model trained, save deferred to the "
                      f"gated atomic save (veto floor {floor:+.4f}%)")
        except Exception as e:
            print(f"[LGB-Q10] quantile training failed (non-fatal): {e}")
            if save and _refit_full and not _mean_saved:
                # the mean save was deferred for the refit — never lose it
                print("[LGB-REFIT] saving the mean booster (deferred "
                      "save; q10 path failed)")
                save_lgb_model(booster, prefix=prefix.rstrip('_'))

        del X_train, X_val, all_scaled
        gc.collect()
        if save:
            return booster
        return (booster, q10, q10_floor_val, n_q10_val)
    except Exception as e:
        print(f"[LGB] Ensemble training failed (non-fatal): {e}")
        return None


def final_refit(cfg, returns, all_features, all_times, all_label_times,
                tickers, ticker_boundaries, input_dim, fold_best_epochs,
                fold_sharpes, seed=None):
    """HYPERSEARCH_V3 (D22 / B12.1): ONE final refit of the winning config
    on ALL pre-holdout data.

    The legacy artifact is the best FOLD's checkpoint — trained on a
    truncated window and selected by fold-max (winner's curse). The refit
    trains on the full purged pre-holdout region with a FIXED epoch budget
    (median of the winning trial's per-fold best epochs — "collective
    early stopping", no validation pass, no early stopping) and ships an
    SWA tail soup (uniform average of the LAST SOUP_K epoch checkpoints).
    Returns (state, scaler, info) or None — the caller falls back loudly
    to the fold-max checkpoint; a refit failure must NEVER kill the run.

    seed (R2C-04 / L3, None = legacy unseeded): derived TRAINER_SEED
    sub-seed — seeds torch init/dropout and the epoch batch permutations
    so two refits at the same seed byte-compare (the FR-02 / FR-13
    determinism prerequisite; verified as a Jetson step).
    """
    try:
        epochs = refit_epoch_budget(fold_best_epochs, MAX_EPOCHS)
        if epochs is None:
            print('[REFIT] no per-fold epoch record — skipping refit')
            return None
        # Regime tripwire FIRST (warn, never block)
        tripwire = bool(fold_sharpes and fold_sharpes[-1] < 0
                        and np.mean(fold_sharpes) > 0)
        if tripwire:
            print(f"[REFIT] TRIPWIRE: newest fold Sharpe "
                  f"{fold_sharpes[-1]:.2f} < 0 while trial mean "
                  f"{np.mean(fold_sharpes):.2f} > 0 — refit-on-all may "
                  f"bake in a stale-regime model; flagging for owner "
                  f"review (proceeding)")

        seq_len = cfg['seq_len']
        valid = _valid_indices(tickers, ticker_boundaries, seq_len)
        boundary = get_holdout_boundary(all_times)
        # PURGE: label windows must complete before the holdout starts
        train_idx = valid[all_label_times[valid] <= boundary]
        train_idx = train_idx[~np.isnan(returns[train_idx])]
        if len(train_idx) < 500:
            print(f'[REFIT] only {len(train_idx)} usable rows — skipping')
            return None

        scaler = RobustScaler().fit(all_features[train_idx])
        all_scaled = scaler.transform(all_features).astype(np.float32)
        y_train = returns[train_idx]
        n_train = len(train_idx)
        offsets = np.arange(-seq_len, 0)

        use_amp = device.type == 'cuda'
        eff_batch_size = cfg['batch_size']
        _repairs = _training_repairs()  # R2C-04 TRAINING_REPAIRS_V1 (L5)
        # R2C-04 (L3): None = legacy unseeded; seeding sits INSIDE the
        # retry loop so an OOM retry replays the same init stream.
        refit_rng = (np.random.default_rng(int(seed))
                     if seed is not None else None)
        while True:
            try:
                if seed is not None:
                    torch.manual_seed(int(seed))
                model = RegressionLSTM(input_dim, cfg['hidden_dim'],
                                       cfg['num_layers'], cfg['dropout'],
                                       cfg['n_heads']).to(device)
                criterion = nn.HuberLoss(delta=cfg['huber_delta'],
                                         reduction='none')
                optimizer = optim.Adam(model.parameters(),
                                       lr=cfg['learning_rate'],
                                       weight_decay=cfg['weight_decay'])
                if cfg.get('scheduler') == 'cosine':
                    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=20, T_mult=2)
                else:
                    # 'plateau' steps on val loss — no validation pass
                    # exists here, so the refit runs at constant LR.
                    sched = None
                grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

                if _repairs:
                    # TRAINING_REPAIRS_V1 (L5): snapshot pristine init so
                    # the memory probe leaves no trace (legacy: the probe's
                    # unclipped optimizer step trained the model before
                    # epoch 0).
                    _init_snap = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}

                # One probe batch to test if this config fits in memory
                # (same pattern as the fold loop's first-batch probe)
                model.train()
                test_idx = train_idx[:min(eff_batch_size, n_train)]
                xb = torch.from_numpy(
                    gather_windows(all_scaled, test_idx, offsets)).to(device)
                yb = torch.from_numpy(y_train[:len(test_idx)]).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(xb)
                    raw_loss = criterion(pred, yb)
                    weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                    loss = (raw_loss * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                del xb, yb, pred, raw_loss, weights, loss
                if _repairs:
                    # L5: restore init weights + fresh optimizer /
                    # scheduler / grad-scaler — epoch 0 starts clean.
                    model.load_state_dict(_init_snap)
                    optimizer = optim.Adam(model.parameters(),
                                           lr=cfg['learning_rate'],
                                           weight_decay=cfg['weight_decay'])
                    if cfg.get('scheduler') == 'cosine':
                        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer, T_0=20, T_mult=2)
                    grad_scaler = torch.amp.GradScaler('cuda',
                                                       enabled=use_amp)
                    del _init_snap
                break  # fits in memory
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                try:
                    del model, criterion, optimizer
                except NameError:
                    pass
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                eff_batch_size //= 2
                if eff_batch_size < 128:
                    raise  # let the outer handler report and fall back
                print(f"  [REFIT] OOM-RETRY: batch "
                      f"{eff_batch_size * 2}→{eff_batch_size} "
                      f"({str(e)[:60]})")
        if _repairs:  # SIG-R2-3: print only
            print("[REPAIRS] L5 final_refit: memory probe undone (init "
                  "weights restored; fresh optimizer/scheduler/grad-scaler)")

        snaps: list[dict] = []
        for epoch in range(epochs):
            model.train()
            # R2C-04 (L3): refit_rng None = legacy ambient-RNG shuffle
            perm = (refit_rng.permutation(n_train)
                    if refit_rng is not None
                    else np.random.permutation(n_train))
            for i in range(0, n_train, eff_batch_size):
                bi = perm[i:i + eff_batch_size]
                xb = torch.from_numpy(
                    gather_windows(all_scaled, train_idx[bi],
                                   offsets)).to(device)
                yb = torch.from_numpy(y_train[bi]).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(xb)
                    raw_loss = criterion(pred, yb)
                    weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                    loss = (raw_loss * weights).mean()
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            if cfg.get('scheduler') == 'cosine' and sched:
                sched.step()
            # SWA tail soup: keep only the LAST SOUP_K epoch snapshots.
            # clone() is load-bearing: on CPU training, .cpu() returns a
            # VIEW of the live weights and later epochs would mutate the
            # "snapshot" in place.
            snaps.append({k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()})
            del snaps[:-SOUP_K]
        state = average_states(snaps)

        del model, all_scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[REFIT] final refit: {n_train} rows, {epochs} fixed epochs "
              f"(SWA tail soup K={SOUP_K})")
        return state, scaler, {'epochs': int(epochs),
                               'n_rows': int(len(train_idx)),
                               'tripwire': bool(tripwire)}
    except Exception as e:
        print(f"[REFIT] final refit failed ({e}) — caller falls back to the "
              f"fold-max checkpoint")
        return None


# ---------------------------------------------------------------------------
# Holdout evaluation + atomic save
# ---------------------------------------------------------------------------

def evaluate_on_holdout(state, scaler, cfg, all_features, all_returns_by_fb,
                        all_times, tickers, ticker_boundaries,
                        input_dim, asset_type, n_trials,
                        all_tb_bars_by_fb=None, lgb_booster=None,
                        q10_booster=None, q10_floor=None, lstm_weight=None):
    """Score the winning config ONCE on the untouched final time slice.

    Returns None, or a dict with keys: sharpe, dsr, dsr_min, n_trades,
    n_eff, n_eff_v2, status, min_trl, n_trials_pool, u_bar_mean,
    trade_returns, n_rows, hit_rate, pred_deciles — plus certified,
    lstm_weight, q10_vetoed and cs_rank_ic when the blend path ran.

    lgb_booster/q10_booster/q10_floor/lstm_weight (all None = legacy raw-
    LSTM gate, byte-identical): under HYPERSEARCH_V3 the certificate is
    issued against the DEPLOYED predictor — w*LSTM + (1-w)*LGB with the
    q10 tail veto applied to long entries (exact predict_now/backtest
    semantics); the report then carries 'certified'/'lstm_weight'/
    'q10_vetoed' and pred_deciles/hit_rate/trade_returns become
    blend-based (the live PSI drift monitor compares live BLENDED preds —
    the certificate now matches).
    """
    from validation import dsr_from_trade_returns, DSR_MIN
    try:
        seq_len = cfg['seq_len']
        fb = cfg.get('forward_bars', 24)
        threshold = cfg['trade_threshold']
        holdout_idx = get_holdout_indices(all_times, tickers, ticker_boundaries, seq_len)
        # Evaluate the holdout on the SAME target kind the trial trained on
        key = ('tb', fb) if cfg.get('target_kind') == 'tb' else fb
        returns = all_returns_by_fb.get(key)
        if returns is None:
            returns = all_returns_by_fb.get(fb)
        if returns is None:
            returns = next(v for k, v in all_returns_by_fb.items()
                           if not isinstance(k, tuple))
        holdout_idx = holdout_idx[~np.isnan(returns[holdout_idx])]
        if len(holdout_idx) < 200:
            print(f"  [HOLDOUT] only {len(holdout_idx)} usable rows — gate fails closed")
            return {'sharpe': 0.0, 'dsr': 0.0, 'dsr_min': DSR_MIN,
                    'n_trades': 0, 'n_rows': int(len(holdout_idx))}

        scaled = scaler.transform(all_features).astype(np.float32)
        offsets = np.arange(-seq_len, 0)
        mdl = RegressionLSTM(input_dim, cfg['hidden_dim'], cfg['num_layers'],
                             cfg['dropout'], cfg['n_heads']).to(device)
        mdl.load_state_dict(state)
        mdl.eval()
        preds = []
        use_amp = device.type == 'cuda'
        with torch.inference_mode():
            for i in range(0, len(holdout_idx), 1024):
                vidx = holdout_idx[i:i + 1024]
                xvb = torch.from_numpy(gather_windows(scaled, vidx, offsets)).to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    vo = mdl(xvb)
                preds.append(vo.cpu().numpy())
        preds = np.concatenate(preds)
        y = returns[holdout_idx]

        # D25 (HYPERSEARCH_V3): certify the DEPLOYED blend. Fail-safe: any
        # leg-scoring failure degrades loudly to the raw-LSTM certificate.
        blended = False
        long_veto = None
        n_vetoed = 0
        if lgb_booster is not None and lstm_weight is not None:
            try:
                n_h = len(holdout_idx)
                lgb_preds = np.empty(n_h, dtype=np.float64)
                do_q10 = (q10_booster is not None and q10_floor is not None
                          and np.isfinite(q10_floor))
                q10_preds = np.empty(n_h, dtype=np.float64) if do_q10 else None
                for i in range(0, n_h, 1024):
                    vidx = holdout_idx[i:i + 1024]
                    # predict_now's flatten_sequence contract:
                    # == windows.reshape(-1) (pinned in train_lgb_ensemble)
                    X = gather_windows(scaled, vidx,
                                       offsets).reshape(len(vidx), -1)
                    lgb_preds[i:i + len(vidx)] = lgb_booster.predict(X)
                    if do_q10:
                        q10_preds[i:i + len(vidx)] = q10_booster.predict(X)
                # Exact ensemble_predict arithmetic
                blend_preds = (float(lstm_weight) * preds
                               + (1.0 - float(lstm_weight)) * lgb_preds)
                if do_q10:
                    long_veto = q10_preds < q10_floor
                    n_vetoed = int(long_veto.sum())
                # R2C-07 (FR-05): keep the raw-LSTM leg for the
                # cross-sectional rank-IC certificate lines below.
                lstm_preds = preds.copy()
                preds = blend_preds
                blended = True
            except Exception as e:
                print(f"  [HOLDOUT] blend leg scoring failed ({e}) — gating "
                      f"raw LSTM (certificate degraded)")
                long_veto = None
                n_vetoed = 0
                blended = False

        # SIG-R2-MASK (default OFF -> None): certify only the entries the
        # live stock book could take; OR'd with the q10 veto (n_vetoed
        # stays the q10-only count).
        _sess_ok = _session_entry_ok(all_times[holdout_idx], asset_type)
        if _sess_ok is not None:
            long_veto = (~_sess_ok if long_veto is None
                         else (long_veto | ~_sess_ok))
        hb = (ticker_block_ids(holdout_idx, ticker_boundaries)
              if _objective_v3() else None)
        sharpe = compute_sharpe(preds, y, threshold, forward_bars=fb,
                                asset_type=asset_type, block_ids=hb,
                                long_veto=long_veto)
        trade_returns, entry_pos = simulate_trades(
            preds, y, threshold, fb, TXN_COST_PCT.get(asset_type, 0.6),
            return_entries=True, block_ids=hb, long_veto=long_veto)

        # R2C-05 / FR-01 instrumentation (ships direct): quantile vs fixed
        # boundary side-by-side with holdout row/trade counts, so the
        # FIXED_HOLDOUT_DAYS flip can be judged from a legacy run per book
        # (06 plan §4.2: require >= 10 calendar-effective trades at 60d
        # before any window experiment). Row counts are pre-NaN-filter
        # valid rows past each boundary; scored rows/trades are the
        # ACTIVE boundary's.
        try:
            _fd = _fixed_holdout_days()
            _ref_days = 60.0 if _fd is None else float(_fd)
            _b_q = holdout_boundary(all_times,
                                    holdout_fraction=HOLDOUT_FRACTION)
            _b_f = holdout_boundary(all_times, fixed_days=_ref_days)
            _valid_t = all_times[_valid_indices(tickers, ticker_boundaries,
                                                seq_len)]
            print(f"  [HOLDOUT] FR-01 boundary quantile={_b_q} "
                  f"({int((_valid_t > _b_q).sum())} rows) vs "
                  f"fixed-{_ref_days:g}d={_b_f} "
                  f"({int((_valid_t > _b_f).sum())} rows); active="
                  f"{'fixed' if _fd is not None else 'quantile'}, "
                  f"scored rows={len(holdout_idx)}, "
                  f"trades={len(trade_returns)}")
        except Exception as _fe:
            print(f"  [HOLDOUT] FR-01 boundary instrumentation failed "
                  f"({_fe})")

        # Effective-n deflation: the DSR null assumes n INDEPENDENT trade-SR
        # draws. With overlapping forward-window labels the effective count
        # is sum(average-uniqueness) < n, so the no-skill expected-max bar is
        # actually higher. Measure n_eff from the realized trades' label
        # spans (TB_Bars) instead of assuming IID. Falls back to IID (n_eff
        # = n) when spans were not harvested — the gate is never loosened.
        # PROMOTION_GATE_V2 (2026-08 Q1): OFF = legacy uniqueness->clustered
        # numerics byte-identical (plus side-by-side instrumentation); ON =
        # calendar-concurrency average uniqueness as the ONE non-IID
        # correction, failing closed below the 10-effective-trade floor.
        try:
            from strategy_config import (PROMOTION_GATE_V2 as _gate_v2,
                                         KISH_NEFF_ENABLED as _kish,
                                         KISH_RHO_FLOOR as _rho_floor)
        except ImportError:
            _gate_v2, _kish, _rho_floor = False, False, {}
        n_eff = None
        u_bar_traded = None
        entry_t = exit_t = None
        n_eff_v2 = None
        if all_tb_bars_by_fb and fb in all_tb_bars_by_fb and len(entry_pos):
            tb = all_tb_bars_by_fb[fb]
            if (cfg.get('target_kind') != 'tb'
                    and _holdout_span_by_target()):
                # SIG-R1-A3: a raw-target trade books the fb-bar forward
                # return, so its label window is exactly fb bars; TB_Bars
                # is the triple-barrier exit span of a DIFFERENT label
                # (crypto store mean 0.35-0.78 x fb) and understates the
                # overlap the deflation measures.
                tb = np.full(len(tb), float(fb), dtype=np.float64)
            global_rows = holdout_idx[entry_pos]
            # Hoisted [entry, exit] calendar-time reconstruction — shared by
            # the legacy clustering, the v2 estimator, and the side-by-side
            # instrumentation (identical math to the pre-hoist version).
            try:
                if isinstance(ticker_boundaries, dict):
                    _blocks = sorted(ticker_boundaries.values())
                else:
                    _blocks = sorted(ticker_boundaries)
                _starts = np.asarray([b[0] for b in _blocks])
                _ends = np.asarray([b[1] for b in _blocks])
                entry_t = all_times[global_rows]
                exit_rows = np.empty(len(global_rows), dtype=np.int64)
                for _k, _row in enumerate(global_rows):
                    _bi = int(np.searchsorted(_starts, _row,
                                              side='right') - 1)
                    _block_last = int(_ends[_bi]) - 1
                    _span = tb[_row]
                    _span = int(_span) if np.isfinite(_span) else 0
                    exit_rows[_k] = min(_row + max(_span, 0), _block_last)
                exit_t = all_times[exit_rows]
            except Exception as te:
                print(f"  [HOLDOUT] entry/exit time reconstruction failed "
                      f"({te})")
                entry_t = exit_t = None

            if _gate_v2:
                # Exactly ONE non-IID correction: calendar-concurrency
                # average uniqueness across ALL names (supersedes both the
                # per-ticker uniqueness and the cluster count; never stacked
                # with Lo-2002 — CLAUDE.md gotcha #4).
                try:
                    from sample_weights import calendar_effective_n
                    if entry_t is None:
                        raise RuntimeError(
                            'entry/exit time reconstruction failed')
                    _rho = None
                    if _kish and asset_type in _rho_floor:
                        _rho = _rho_floor[asset_type]
                    # SIG-R1-A1: numeric times are HOURS by
                    # calendar_effective_n's contract; entry_t/exit_t are
                    # epoch SECONDS (per-second bins: wrong n_eff and a
                    # ~0.9 GB transient on a 220-day crypto holdout).
                    _cal = calendar_effective_n(entry_t / 3600.0,
                                                exit_t / 3600.0,
                                                rho_bar=_rho)
                    n_eff = float(_cal['n_eff'])
                    n_eff_v2 = float(_cal['n_eff'])
                    print(f"  [HOLDOUT] v2 calendar n_eff="
                          f"{_cal['n_eff']:.1f} (max_concurrency="
                          f"{_cal['max_concurrency']}, rho_bar={_rho})")
                except Exception as ve:
                    print(f"  [HOLDOUT] v2 calendar n_eff unavailable "
                          f"({ve}) — falling back to IID null")
                    n_eff = None
            else:
                try:
                    from sample_weights import average_uniqueness, effective_n
                    # Concurrency is computed among the TRADED labels ONLY — a
                    # trade's independence is measured against the other trades
                    # the gate counts, not against every (non-traded) panel bar.
                    # Mask: only traded rows carry their span; all else is NaN
                    # (NaN rows neither hold nor count concurrency). Because
                    # simulate_trades skips fb bars after each entry and a label
                    # span is <= fb, holdout trades are near-non-overlapping, so
                    # n_eff ~ n_trades here by construction — the correction bites
                    # only if a model's realized entries crowd inside their holds.
                    masked = np.full(len(tb), np.nan, dtype=np.float64)
                    masked[global_rows] = tb[global_rows]
                    u_all = average_uniqueness(masked, ticker_boundaries)
                    u_bar_traded = u_all[global_rows]
                    n_eff = effective_n(u_bar_traded)
                    if n_eff == 0.0:
                        print("  [HOLDOUT] WARNING: effective_n returned the "
                              "0.0 unmeasurable sentinel UNGUARDED — the DSR "
                              "clamp will fabricate a 10-observation floor "
                              "(known defect D02; PROMOTION_GATE_V2 fails "
                              "this closed)")

                    # Cross-sectional clustering (2026-07 review): the per-ticker
                    # uniqueness above counts same-hour trades on N correlated
                    # names as N independent draws — on the 6-coin crypto panel
                    # (pairwise rho 0.7-0.9) that overstates the DSR's sqrt(n)
                    # breadth 2-4x, raising a zero-edge model's false-pass rate
                    # from ~0.2% to 5-9% per attempt. Collapse trades whose
                    # [entry, exit] CALENDAR windows overlap across names (rho=1
                    # worst case) and take the harsher of the two counts.
                    try:
                        from sample_weights import clustered_effective_n
                        if entry_t is None:
                            raise RuntimeError(
                                'entry/exit time reconstruction failed')
                        n_x = clustered_effective_n(entry_t, exit_t)
                        if 0 < n_x < (n_eff if n_eff else len(global_rows)):
                            print(f"  [HOLDOUT] cross-sectional clustering: "
                                  f"{len(global_rows)} trades -> {n_x} disjoint "
                                  f"calendar clusters (n_eff {n_eff:.1f} -> {n_x})")
                            n_eff = float(n_x)
                    except Exception as ce:
                        print(f"  [HOLDOUT] cross-sectional n_eff unavailable "
                              f"({ce}) — keeping per-ticker n_eff")
                except Exception as ue:
                    print(f"  [HOLDOUT] uniqueness n_eff unavailable ({ue}) — "
                          f"falling back to IID null")
                    n_eff = None
                # Known-defect crowding warning (mirror of backtest.py's) +
                # legacy-vs-v2 side-by-side line. Instrumentation only.
                if (n_eff is not None and len(trade_returns)
                        and n_eff < len(trade_returns) / 5):
                    print(f"  [HOLDOUT] WARNING: cross-sectional clustering "
                          f"collapsed {len(trade_returns)} trades to "
                          f"{int(n_eff)} clusters "
                          f"({n_eff / len(trade_returns):.1%}) — the DSR null "
                          f"is being set by trade crowding, not by the model")
                try:
                    from sample_weights import calendar_effective_n
                    if entry_t is not None:
                        # SIG-R1-A1: seconds -> hours (see the v2 branch)
                        _cal = calendar_effective_n(entry_t / 3600.0,
                                                    exit_t / 3600.0)
                        n_eff_v2 = float(_cal['n_eff'])
                        print(f"  [HOLDOUT] n_eff legacy={n_eff} "
                              f"(uniqueness->clustered) vs v2 calendar="
                              f"{_cal['n_eff']:.1f} (max_concurrency="
                              f"{_cal['max_concurrency']})")
                except Exception:
                    pass

        if _gate_v2:
            # n_eff_source labels provenance honestly: None when the
            # calendar estimator was unavailable and the IID null applies.
            dsr = dsr_from_trade_returns(trade_returns, n_trials=n_trials,
                                         n_eff=n_eff,
                                         n_eff_source=('calendar_uniqueness'
                                                       if n_eff is not None
                                                       else None),
                                         fail_closed_floor=True)
        else:
            dsr = dsr_from_trade_returns(trade_returns, n_trials=n_trials,
                                         n_eff=n_eff)
        del mdl, scaled
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        u_bar_mean = (round(float(np.nanmean(u_bar_traded)), 4)
                      if u_bar_traded is not None and len(u_bar_traded)
                      else None)
        # MinTRL on a failed gate (instrumentation, both modes): how many
        # more EFFECTIVE trades this SR needs to clear the deflation bar.
        _min_trl = dsr.get('min_trl')
        if (_min_trl is not None and math.isfinite(_min_trl)
                and (sharpe <= 0 or dsr['dsr'] < DSR_MIN)):
            print(f"  [HOLDOUT] MinTRL: need "
                  f"~{max(0.0, _min_trl - dsr['n_eff']):.0f} more effective "
                  f"trades at this SR (min_trl={_min_trl:.0f}, "
                  f"n_eff={dsr['n_eff']})")
        report = {'sharpe': round(float(sharpe), 4),
                  'dsr': round(float(dsr['dsr']), 4),
                  'dsr_min': DSR_MIN,
                  'n_trades': int(dsr['n']),
                  'n_eff': dsr.get('n_eff'),
                  'n_eff_v2': n_eff_v2,
                  'status': dsr.get('status'),
                  'min_trl': _min_trl,
                  'n_trials_pool': dsr['n_trials'],
                  'u_bar_mean': u_bar_mean,
                  # Persist the realized holdout trade returns so a champion's
                  # DSR is re-checkable later without a full re-inference pass
                  # (closes the gap where save_artifacts kept only the summary).
                  'trade_returns': [round(float(x), 6) for x in trade_returns],
                  'n_rows': int(len(holdout_idx)),
                  # Net-of-cost hit rate — the CUSUM live monitor's baseline
                  'hit_rate': (round(float(np.mean(trade_returns > 0)), 4)
                               if len(trade_returns) else None),
                  # Reference distribution for the live PSI drift monitor:
                  # deciles of the holdout predictions (monitor_drift.py)
                  'pred_deciles': [round(float(x), 6) for x in
                                   np.percentile(preds, np.arange(0, 101, 10))]}
        if _sess_ok is not None:  # SIG-R2-MASK: ON-only keys
            report['session_mask'] = True
            report['session_rows_ok'] = int(_sess_ok.sum())
        # ONLY when blended (flag-OFF report stays key-for-key identical)
        if blended:
            report['certified'] = 'blend'
            report['lstm_weight'] = round(float(lstm_weight), 4)
            report['q10_vetoed'] = n_vetoed
            # R2C-07 (FR-05): cross-sectional rank-IC certificate lines
            # for the LSTM leg, the LGB leg, and the deployed blend —
            # the go/no-go evidence for FR-06 (strongly-positive LGB
            # rank IC => deprioritize the LambdaRankIC port; near-zero /
            # negative => the in-house justification to build it).
            # Fail-soft: an IC failure never degrades the certificate.
            try:
                _gids = all_times[holdout_idx]

                def _ic_round(r):
                    return {
                        'mean': (round(r['mean'], 4)
                                 if r['mean'] is not None else None),
                        'se': (round(r['se'], 4)
                               if r['se'] is not None else None),
                        'n_groups': r['n_groups'],
                        'n_skipped': r['n_skipped'],
                        'splits': [{'mean': (round(s['mean'], 4)
                                             if s['mean'] is not None
                                             else None),
                                    'n_groups': s['n_groups']}
                                   for s in r['splits']],
                    }

                _cs = {'lstm': _ic_round(cs_rank_ic(lstm_preds, y, _gids)),
                       'lgb': _ic_round(cs_rank_ic(lgb_preds, y, _gids)),
                       'blend': _ic_round(cs_rank_ic(preds, y, _gids))}
                for _leg in ('lstm', 'lgb', 'blend'):
                    _r = _cs[_leg]
                    _sub = '/'.join(
                        ('na' if s['mean'] is None else f"{s['mean']:.3f}")
                        for s in _r['splits'])
                    print(f"  [HOLDOUT] FR-05 cs-rank-IC {_leg}: "
                          f"mean={_r['mean']} se={_r['se']} "
                          f"n_groups={_r['n_groups']} "
                          f"(skipped {_r['n_skipped']} thin) "
                          f"splits=[{_sub}]")
                if asset_type == 'crypto':
                    print("  [HOLDOUT] FR-05 caveat: crypto cross-sections "
                          "are ~6 names — per-timestamp rank ICs carry "
                          "wide CIs; read the pooled mean/SE and the "
                          "sub-period signs, never single groups")
                report['cs_rank_ic'] = _cs
            except Exception as _ce:
                print(f"  [HOLDOUT] FR-05 cs-rank-IC failed (non-fatal): "
                      f"{_ce}")
        return report
    except Exception as e:
        print(f"  [HOLDOUT] evaluation failed: {e} — gate fails closed")
        return None


def _record_holdout_gate(adaptive_state, asset_type):
    """SIG-R3-HGATES: +1 cum_holdout_gates per winner actually SCORED on
    the holdout (the documented intent: research/campaign_2026-08/
    02_research.md B03.2 "+1 per winner actually scored on the holdout" —
    evaluations, pass OR fail; a None report scored nothing and is not
    counted by the caller).

    Written twice, deliberately: into the in-memory state (update_after_
    search / the initial-params save re-save THIS object later) AND
    straight to disk (the losing-run branch never saves this object —
    record_trials reloads from disk). Both start from the same persisted
    value, so every later save agrees on base+1. Measurement-only,
    fail-soft: never blocks a save or changes the gate.
    """
    try:
        adaptive_state['cum_holdout_gates'] = int(
            adaptive_state.get('cum_holdout_gates', 0) or 0) + 1
        from adaptive_config import save_adaptive_state
        disk = load_adaptive_state(asset_type)
        disk['cum_holdout_gates'] = int(
            disk.get('cum_holdout_gates', 0) or 0) + 1
        save_adaptive_state(disk)
        print(f"[GATE] cum_holdout_gates={adaptive_state['cum_holdout_gates']}")
    except Exception as e:
        print(f"[GATE] cum_holdout_gates not recorded (non-fatal): {e}")


def save_model_atomically(prefix, state, best_cfg, input_dim, config, scaler,
                          feature_cols, score=0.0, oof_pack=None,
                          extra_artifacts=None):
    """Write all four artifacts via tmp+rename, then a manifest LAST.

    The four files were previously written non-atomically while bots
    hot-reload on the .pth mtime alone — a reload mid-save could pair new
    weights with an old scaler. Bots now key reloads on the manifest,
    which only appears after every artifact is fully on disk.

    extra_artifacts (T1/HYPERSEARCH_V3): optional {path: writer(path)}
    written tmp+rename alongside the core four — BEFORE the OOF npz and
    the manifest, so under the flag ALL artifacts (LGB boosters included)
    are on disk before the manifest appears (manifest-LAST invariant
    preserved and strengthened). The .prev backup loop already names the
    lgb files, so the OLD boosters are backed up before new bytes land.
    None (default) = legacy flow, byte-identical.
    """
    mdl = RegressionLSTM(input_dim, best_cfg['hidden_dim'],
                         best_cfg['num_layers'], best_cfg['dropout'],
                         best_cfg['n_heads'])
    mdl.load_state_dict(state)

    artifacts = {
        f'{prefix}model_v2.pth': lambda p: torch.save(mdl.state_dict(), p),
        f'{prefix}config_v2.pkl': lambda p: joblib.dump(config, p),
        f'{prefix}scaler_v2.pkl': lambda p: joblib.dump(scaler, p),
        f'{prefix}feature_cols_v2.pkl': lambda p: joblib.dump(feature_cols, p),
    }
    # Keep the outgoing model as .prev so the policy-backtest gate
    # (backtest.py --gate) can roll back a promotion that fails on policy
    # P&L even though it cleared the fit-level holdout.
    import shutil
    for path in list(artifacts) + [f'{prefix}model_v2.manifest.json',
                                   f'{prefix}lgb_model.txt',
                                   f'{prefix}lgb_q10.txt',
                                   f'{prefix}lgb_q10_meta.json',
                                   f'{prefix}oof_preds.npz']:
        if os.path.exists(path):
            try:
                shutil.copy2(path, f'{path}.prev')
            except OSError:
                pass
    # Single timestamp so the OOF npz and the manifest carry the SAME
    # fingerprint (behavior-identical hoist of the manifest's saved_at).
    saved_at = datetime.now().isoformat()
    for path, writer in {**artifacts, **(extra_artifacts or {})}.items():
        tmp = f'{path}.tmp.{os.getpid()}'
        writer(tmp)
        os.replace(tmp, path)

    # D12/B04.1: persist the winner's purged walk-forward val predictions
    # BEFORE the manifest (manifest-LAST invariant: the npz is on disk before
    # the manifest that fingerprints it appears). Fail-soft — an OOF persist
    # failure must NEVER block a model save.
    if oof_pack is not None:
        try:
            from meta_label import write_oof_npz
            write_oof_npz(f'{prefix}oof_preds.npz', oof_pack, saved_at,
                          round(float(score), 4))
            print(f"Saved: {prefix}oof_preds.npz ({len(oof_pack['ts_ns'])} OOF rows)")
        except Exception as e:
            print(f"[OOF] npz persist failed (non-fatal — meta will fall back "
                  f"loudly to in-sample): {e}")
    else:
        print("[OOF] no OOF pack for this save — any existing "
              f"{prefix}oof_preds.npz is now fingerprint-stale (meta falls back loudly)")

    manifest = {
        'saved_at': saved_at,
        'score': round(float(score), 4),
        'files': list(artifacts.keys()),
        'config': {k: v for k, v in config.items() if k != 'holdout'},
        'holdout': config.get('holdout'),
    }
    mpath = f'{prefix}model_v2.manifest.json'
    tmp = f'{mpath}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, mpath)

    for path in {**artifacts, **(extra_artifacts or {})}:
        print(f"Saved: {path}")
    print(f"Manifest: {mpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# SIG-R2-3: every training-path flag, printed on ONE grep-able line at
# trainer start (`grep '^\[FLAGS\]' <log>`). Raw strategy_config values via
# getattr (absent = None); env overrides listed separately; *.eff = the
# resolved value the trainer actually uses where an env override exists.
BANNER_FLAGS = ('OBJECTIVE_LONG_ONLY', 'HYPERSEARCH_V3', 'OBJECTIVE_V3',
                'TRAINING_REPAIRS_V1', 'BLEND_FIT_ON_REFIT',
                'BLEND_THRESHOLD_RESELECT', 'PROMOTION_GATE_V2',
                'KISH_NEFF_ENABLED', 'LGB_REFIT_FULL',
                'HOLDOUT_SPAN_BY_TARGET', 'BARS_PER_YEAR_MEASURED',
                'FAILED_TRIAL_PRUNE', 'OBJECTIVE_SESSION_MASK',
                'WICK_PRINT_FILTER')
BANNER_ENV = ('TRADER_HOLDOUT_SPAN_BY_TARGET', 'TRADER_FAILED_TRIAL_PRUNE',
              'TRADER_TRAINER_SEED', 'TRADER_FIXED_HOLDOUT_DAYS',
              'TRADER_OBJECTIVE_SESSION_MASK',
              'TRADER_WICK_PRINT_FILTER')


def flags_banner(args, mode, num_trials, seed_base):
    """The one-line [FLAGS] banner (instrumentation only; never raises)."""
    def _safe(fn):
        try:
            return fn()
        except Exception as e:
            return f'?({type(e).__name__})'
    try:
        import strategy_config as _sc
    except Exception:
        _sc = None
    parts = [f'{n}={getattr(_sc, n, None)}' for n in BANNER_FLAGS]
    parts.append(f'HOLDOUT_SPAN_BY_TARGET.eff={_safe(_holdout_span_by_target)}')
    _ftp = globals().get('_failed_trial_prune')  # SIG-R2-1 resolver, if landed
    parts.append(f'FAILED_TRIAL_PRUNE.eff={_safe(_ftp) if _ftp else None}')
    parts += [f'env.{n}={os.environ.get(n) or None}' for n in BANNER_ENV]

    def _preset():
        from indicator_config import load_indicator_config
        return args.preset or load_indicator_config()["preset"]

    def _data_path():
        from data_utils import get_data_path
        return get_data_path('stock' if 'stock' in args.data else 'crypto')
    parts += [f'preset={_safe(_preset)}', f'preset_arg={args.preset}',
              f'seed_base={seed_base}',
              f'fixed_holdout_days={_safe(_fixed_holdout_days)}',
              f'prefix={args.prefix or None}', f'mode={mode}',
              f'trials={num_trials}', f'trials_arg={args.trials}',
              f'data={_safe(_data_path)}', f'data_arg={args.data}']
    return '[FLAGS] ' + ' '.join(str(x).replace(' ', '_') for x in parts)


def main():
    args = parse_args()
    prefix = f'{args.prefix}_' if args.prefix else ''
    asset_type = args.prefix or 'crypto'

    # Load adaptive state
    adaptive_state = load_adaptive_state(asset_type)
    adaptive_space = get_search_space_for_trial(adaptive_state)

    # Determine mode and trial count
    mode = args.mode or adaptive_state.get('mode', 'refine')
    is_initial = mode == 'initial'
    adaptive_trials = get_trial_count(mode, is_initial=is_initial)
    # --trials overrides adaptive count only if explicitly set (not default)
    num_trials = args.trials if args.trials != NUM_TRIALS else adaptive_trials

    # Update FORWARD_BARS global from adaptive state (used by load_data)
    global FORWARD_BARS
    fb_from_state = adaptive_space.get('forward_bars')
    if fb_from_state:
        FORWARD_BARS = sorted(fb_from_state)

    print(f"[ADAPTIVE] {asset_type}: mode={mode}, trials={num_trials}, "
          f"forward_bars={FORWARD_BARS}")
    if adaptive_state.get('best_score', 0) > 0:
        print(f"[ADAPTIVE] Prior best score={adaptive_state['best_score']:.3f}, "
              f"cycles_without_improvement={adaptive_state.get('cycles_without_improvement', 0)}")
    edges = []
    if adaptive_state.get('best_params'):
        from adaptive_config import detect_edges
        edges = detect_edges(adaptive_state['best_params'], adaptive_space)
        if edges:
            print(f"[ADAPTIVE] Edges detected: {edges}")

    db_path = f'{prefix}v2_study.db'
    study_name = f'{prefix}v2_search'
    # R2C-04 (L3): base training seed — None = legacy unseeded everywhere
    _seed_base = _trainer_seed()
    if _seed_base is not None:
        print(f"[SEED] TRAINER_SEED={_seed_base} — torch init/dropout, "
              f"batch shuffles, TPESampler and the final refit are "
              f"derived-seeded (objective_utils.derive_seed)")
    # SIG-R2-3: provable flag state, before the study is created/deleted
    print(flags_banner(args, mode, num_trials, _seed_base), flush=True)

    if args.fresh and os.path.exists(db_path):
        # Instrumentation (B03.2): persist the deletion event — the trials
        # erased here still count as selection pressure in cum_trials.
        from adaptive_config import record_db_deletion
        record_db_deletion(asset_type, db_path, reason='--fresh')
        os.remove(db_path)
        print(f"Deleted existing study DB: {db_path}")

    storage = f'sqlite:///{db_path}'

    (all_features, all_returns_by_fb, all_times, all_label_times,
     tickers, ticker_boundaries,
     feature_cols, input_dim, preset_name,
     has_multi_horizon, all_tb_bars_by_fb) = load_data(
         args.data, preset_override=args.preset, max_rows=args.max_rows)

    best_state_holder = {'state': None, 'scaler': None, 'score': 0.0,
                         'cfg': None, 'fold_sharpes': [],
                         'oof_rows': None, 'oof_preds': None,
                         'oof_fold_ids': None, 'fold_best_epochs': None}
    _state_cache = {}

    phase_id = f'{asset_type}_search'
    phase_label = f'Training {asset_type.title()} v2 Model'
    _pipeline_status = {
        'phase': phase_id,
        'phase_label': phase_label,
        'phase_idx': 0,
        'total_phases': 1,
        'trial_current': 0,
        'trial_total': num_trials,
        'best_score': 0.0,
        'elapsed_sec': 0,
        'model_version': 2,
        'adaptive_mode': mode,
    }
    _write_pipeline_status(_pipeline_status)

    results_log = []
    t0 = time.time()
    trials_since_improvement = 0

    def trial_callback(study, trial):
        nonlocal trials_since_improvement

        elapsed = time.time() - t0
        n = trial.number + 1
        score = trial.value if trial.value is not None else 0.0
        cfg = trial.user_attrs.get('cfg', {})
        fold_sharpes = trial.user_attrs.get('fold_sharpes', [])

        tag = ""
        trials_since_improvement += 1
        if trial.state == optuna.trial.TrialState.PRUNED:
            tag = " [PRUNED]"
        elif score > best_state_holder['score']:
            cached = _state_cache.get(trial.number)
            if cached is not None:
                best_state_holder['state'] = cached['state']
                best_state_holder['scaler'] = cached['scaler']
                best_state_holder['score'] = score
                best_state_holder['cfg'] = cfg
                # Winner's own fold scores — the noisy ratchet's sigma source
                best_state_holder['fold_sharpes'] = fold_sharpes
                # D12: the winner's fold-val OOF predictions travel with it
                best_state_holder['oof_rows'] = cached.get('oof_rows')
                best_state_holder['oof_preds'] = cached.get('oof_preds')
                best_state_holder['oof_fold_ids'] = cached.get('oof_fold_ids')
                # T1: the winner's per-fold best epochs feed final_refit
                best_state_holder['fold_best_epochs'] = \
                    cached.get('fold_best_epochs')
                trials_since_improvement = 0
                tag = " ** BEST **"

        fb = cfg.get('forward_bars', 24)
        th = cfg.get('trade_threshold', '')
        avg_s = trial.user_attrs.get('avg_sharpe', score)
        std_s = trial.user_attrs.get('std_sharpe', 0)
        sharpes_str = '/'.join(f'{s:.2f}' for s in fold_sharpes) if fold_sharpes else ''
        print(f"[{n:3d}] score={score:.3f} (mean={avg_s:.2f} std={std_s:.2f}) folds=[{sharpes_str}] "
              f"| fb={fb} s={cfg.get('seq_len','')} h={cfg.get('hidden_dim','')} "
              f"l={cfg.get('num_layers','')} nh={cfg.get('n_heads','')} "
              f"d={cfg.get('dropout', 0):.2f} "
              f"lr={cfg.get('learning_rate', 0):.4f} "
              f"th={th if th == '' else f'{th:.2f}'} "
              f"hd={cfg.get('huber_delta', '')}"
              f"{tag}")

        results_log.append({
            'i': n, 'cfg': cfg, 'score': score,
            'avg_sharpe': avg_s, 'std_sharpe': std_s,
            'fold_sharpes': fold_sharpes,
            'state': str(trial.state), 'time': elapsed,
        })

        # Update pipeline status for GUI every trial
        _pipeline_status['trial_current'] = n
        _pipeline_status['best_score'] = best_state_holder['score']
        _pipeline_status['elapsed_sec'] = int(elapsed)
        _write_pipeline_status(_pipeline_status)

        if n % 10 == 0:
            with open(f'hypersearch_{prefix}v2_log.json', 'w') as f:
                json.dump(results_log, f, indent=2, default=str)
            print(f"  --- {elapsed/60:.1f}min elapsed, best score={best_state_holder['score']:.3f}, "
                  f"total trials in study={len(study.trials)}, "
                  f"{trials_since_improvement} since last improvement ---")

    # --- Create / resume study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=PRUNE_STARTUP_TRIALS,
                            n_warmup_steps=PRUNE_WARMUP_EPOCHS),
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=PRUNE_STARTUP_TRIALS,
            # R2C-04 (L3): seed=None is the constructor default (legacy)
            seed=(derive_seed(_seed_base, study_name, 'sampler')
                  if _seed_base is not None else None)),
    )

    prior_trials = len(study.trials)
    # COMPLETE-with-value count BEFORE this run: lets the post-search code
    # attribute exactly the trials THIS run added to the selection pool.
    prior_completed = len([t for t in study.trials
                           if t.state == optuna.trial.TrialState.COMPLETE
                           and t.value is not None])
    print(f"\n{'='*70}")
    print(f"OPTUNA V2 REGRESSION SEARCH: {num_trials} new trials (TPE + pruning)")
    print(f"Adaptive mode: {mode}")
    print(f"Optimizing: risk-adjusted Sharpe (mean - 0.5*std, {NUM_FOLDS}-fold walk-forward CV)")
    if has_multi_horizon:
        print(f"Multi-horizon: forward_bars in {FORWARD_BARS}")
    print(f"Mixed precision: {'FP16' if device.type == 'cuda' else 'disabled (CPU)'}")
    print(f"Max rows: {args.max_rows:,}")
    print(f"Resuming from {prior_trials} prior trials in {db_path}")
    print(f"{'='*70}\n")

    # Seed best from prior trials
    if prior_trials > 0:
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if (t.value or 0) > best_state_holder['score']:
                best_state_holder['score'] = t.value
                best_state_holder['cfg'] = t.user_attrs.get('cfg', {})
        if best_state_holder['score'] > 0:
            print(f"Prior best score={best_state_holder['score']:.3f} — new trials must beat this")

    objective_fn = create_objective(all_features, all_returns_by_fb,
                                    all_times, all_label_times,
                                    tickers, ticker_boundaries, input_dim,
                                    _state_cache, asset_type=asset_type,
                                    has_multi_horizon=has_multi_horizon,
                                    adaptive_space=adaptive_space,
                                    study_name=study_name)
    study.optimize(objective_fn, n_trials=num_trials, callbacks=[trial_callback],
                   catch=(Exception,))

    # --- Results ---
    total_time = time.time() - t0
    total_trials = len(study.trials)
    print(f"\n{'='*70}")
    print(f"DONE: {num_trials} new trials in {total_time/60:.1f}min ({total_trials} total in study)")
    print(f"{'='*70}")

    # Check if new best exceeds the existing model's score (protect against regression)
    existing_score = adaptive_state.get('best_score', 0.0)
    new_score = best_state_holder['score']
    holdout_report = None
    model_saved = False

    # --- Selection-pressure pool (B03.2) ----------------------------------
    # Deflate against the CUMULATIVE trials ever run against this holdout
    # family, not just this study's visible count (Harvey & Liu 2015: the
    # pool is judgment-inclusive; DB deletions do not erase the pressure).
    try:
        from strategy_config import PROMOTION_GATE_V2 as _GATE_V2
    except ImportError:
        _GATE_V2 = False
    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE
                        and t.value is not None]
    n_new_completed = max(len(completed_trials) - prior_completed, 0)
    cum_prev = int(adaptive_state.get('cum_trials', 0))
    cum_now = cum_prev + n_new_completed
    print(f"[GATE] deflating against {cum_now} cumulative trials "
          f"({len(completed_trials)} this study, this run "
          f"+{n_new_completed})")
    if _GATE_V2:
        _hist = list(adaptive_state.get('trial_history') or [])
        if _hist:
            from adaptive_config import overlap_weighted_trials
            _hist = _hist + [{'date': datetime.now().isoformat(),
                              'n': n_new_completed}]
            _pool = overlap_weighted_trials(_hist)
        else:
            _pool = float(cum_now)
        n_trials_pool = max(int(round(_pool)), len(completed_trials), 2)
    else:
        n_trials_pool = max(len(completed_trials), 2)  # legacy, byte-identical

    # --- Thresholdout-shaped noisy ratchet (B03.2) ------------------------
    # Seeded from study name + date so the draw is reproducible/loggable.
    from adaptive_config import noisy_ratchet
    _seed = int.from_bytes(hashlib.sha256(
        f"{study_name}|{datetime.now().date().isoformat()}".encode()
    ).digest()[:8], 'big')
    rat = noisy_ratchet(new_score, existing_score,
                        best_state_holder.get('fold_sharpes') or [],
                        seed=_seed)
    if _GATE_V2:
        accept_new = rat['accept']
        print(f"[RATCHET] v2 {'ACCEPT' if rat['accept'] else 'REJECT'}: "
              f"new={new_score:.3f} vs stored+2sigma+eta="
              f"{rat['threshold']:.3f} (sigma={rat['sigma']:.3f}, "
              f"eta={rat['noise']:.4f}, seed={_seed})")
    else:
        accept_new = new_score > existing_score  # legacy strict comparison
        print(f"[RATCHET] v2 would "
              f"{'ACCEPT' if rat['accept'] else 'REJECT'}: "
              f"new={new_score:.3f} vs stored+2sigma+eta="
              f"{rat['threshold']:.3f} (sigma={rat['sigma']:.3f})")

    if best_state_holder['state'] is not None and accept_new:
        best_cfg = best_state_holder['cfg']
        best_scaler = best_state_holder['scaler']

        # --- HYPERSEARCH_V3 (T1, D22/D23/D25): final refit + pre-gate LGB
        # legs + blend-weight fit. Flag OFF: every local below is inert
        # (ship_state/ship_scaler == the legacy fold-max checkpoint pair,
        # all new evaluate_on_holdout kwargs None).
        _v3 = _hypersearch_v3()
        ship_state, ship_scaler = best_state_holder['state'], best_scaler
        refit_info = None
        lgb_pack = None
        lstm_weight = None
        blend_diag = None
        if _v3:
            # (a) the winning config's returns array — SAME key logic as
            # evaluate_on_holdout
            _fb = best_cfg.get('forward_bars', 24)
            _rkey = (('tb', _fb) if best_cfg.get('target_kind') == 'tb'
                     else _fb)
            returns = all_returns_by_fb.get(_rkey)
            if returns is None:
                returns = all_returns_by_fb.get(_fb)
            if returns is None:
                returns = next(v for k, v in all_returns_by_fb.items()
                               if not isinstance(k, tuple))
            # (b) ONE final refit on all purged pre-holdout data (D22)
            _refit = final_refit(
                best_cfg, returns, all_features, all_times, all_label_times,
                tickers, ticker_boundaries, input_dim,
                best_state_holder.get('fold_best_epochs') or [],
                best_state_holder.get('fold_sharpes') or [],
                seed=(derive_seed(_seed_base, study_name, 'refit')
                      if _seed_base is not None else None))
            if _refit:
                ship_state, ship_scaler, refit_info = _refit
            else:
                print('[REFIT] final refit unavailable — falling back to '
                      'fold-max checkpoint (legacy artifact)')
            # (c) LGB legs BEFORE the gate, on the SHIPPING scaler
            # (predict_now feeds both legs one scaler — train/serve
            # parity); nothing touches disk until the gate passes.
            lgb_pack = train_lgb_ensemble(prefix, ship_scaler, best_cfg,
                                          all_features, all_returns_by_fb,
                                          all_times, all_label_times,
                                          tickers, ticker_boundaries,
                                          all_tb_bars_by_fb=all_tb_bars_by_fb,
                                          save=False)
            # (d) Blend fit (D23) on the winning trial's LAST fold's val
            # slice — the only slice out-of-sample for BOTH legs (the LGB
            # booster trains on folds[-1] train; earlier folds' val rows
            # sit inside it). Deliberate B12 simplification: a full 3-fold
            # LGB OOF pass would cost 3 extra LGB trainings; both legs
            # here carry symmetric val-selection optimism — the
            # OOF-symmetry requirement's actual point.
            try:
                if (lgb_pack and lgb_pack[0] is not None
                        and best_state_holder.get('oof_rows')
                        and best_state_holder.get('oof_preds')):
                    from blend_fit import (fit_blend_weight,
                                           fit_blend_weight_v2,
                                           smooth_across_retrains)
                    _booster = lgb_pack[0]
                    rows = np.asarray(best_state_holder['oof_rows'][-1])
                    lstm_oof = np.asarray(
                        best_state_holder['oof_preds'][-1])
                    # M4 guard (R2C-02b): under HYPERSEARCH_V3 without
                    # OBJECTIVE_V3 the trial folds were built with
                    # purge_val_labels=False, so folds[-1] val rows can
                    # carry label windows crossing INTO the holdout the
                    # certificate then scores — purge them so the
                    # deployed w is never fitted on holdout returns.
                    if not _objective_v3():
                        _hb = get_holdout_boundary(all_times)
                        _keep = all_label_times[rows] <= _hb
                        if not np.all(_keep):
                            print(f"[BLEND] M4 guard: purged "
                                  f"{int((~_keep).sum())}/{len(rows)} "
                                  f"blend-fit rows whose label windows "
                                  f"cross the holdout boundary")
                            rows = rows[_keep]
                            lstm_oof = lstm_oof[_keep]
                    y_fit = returns[rows]
                    _sl = best_cfg['seq_len']
                    _off = np.arange(-_sl, 0)
                    # ship_scaler transform computed once here and freed
                    _scaled_fit = ship_scaler.transform(
                        all_features).astype(np.float32)
                    lgb_oof = np.empty(len(rows), dtype=np.float64)
                    for i in range(0, len(rows), 1024):
                        _ri = rows[i:i + 1024]
                        _X = gather_windows(_scaled_fit, _ri,
                                            _off).reshape(len(_ri), -1)
                        lgb_oof[i:i + len(_ri)] = _booster.predict(_X)
                    # M2 (R2C-02c): the stale fit consumes the fold-souped
                    # trial checkpoint's val preds under the FOLD scaler,
                    # while the shipped LSTM is ship_state under
                    # ship_scaler (lgb_oof above is already ship_scaler-
                    # based — mixed inputs). Re-predict folds[-1] val rows
                    # with the SHIPPING state+scaler so both legs' fit
                    # inputs match the deployed predictor; stale-w vs
                    # refit-w are logged side by side, and the deployment
                    # source flips only under default-OFF
                    # BLEND_FIT_ON_REFIT (Jetson A/B first — 06 plan
                    # §4.11). Reuses _scaled_fit (one transform).
                    lstm_oof_refit = None
                    try:
                        _rmdl = RegressionLSTM(
                            input_dim, best_cfg['hidden_dim'],
                            best_cfg['num_layers'], best_cfg['dropout'],
                            best_cfg['n_heads']).to(device)
                        _rmdl.load_state_dict(ship_state)
                        _rmdl.eval()
                        _rp = []
                        _r_amp = device.type == 'cuda'
                        with torch.inference_mode():
                            for i in range(0, len(rows), 1024):
                                _ri = rows[i:i + 1024]
                                _xb = torch.from_numpy(gather_windows(
                                    _scaled_fit, _ri, _off)).to(device)
                                with torch.amp.autocast('cuda',
                                                        enabled=_r_amp):
                                    _vo = _rmdl(_xb)
                                _rp.append(_vo.cpu().numpy())
                        lstm_oof_refit = np.concatenate(_rp)
                        del _rmdl, _rp
                    except Exception as _re:
                        print(f"[BLEND] refit-state val inference failed "
                              f"({_re}) — stale-w only")
                    del _scaled_fit
                    gc.collect()
                    # B12 BINDING: shrink_to=0.5, shrink_lambda=0.5 defaults
                    fit_stale = fit_blend_weight_v2(
                        lstm_oof, lgb_oof, y_fit,
                        forward_bars=best_cfg.get('forward_bars', 24))
                    fit_refit = None
                    if lstm_oof_refit is not None:
                        fit_refit = fit_blend_weight_v2(
                            lstm_oof_refit, lgb_oof, y_fit,
                            forward_bars=best_cfg.get('forward_bars', 24))
                        print(f"[BLEND] M2 side-by-side: stale-w="
                              f"{fit_stale['w']:.4f} "
                              f"(raw={fit_stale['w_raw']}) vs refit-w="
                              f"{fit_refit['w']:.4f} "
                              f"(raw={fit_refit['w_raw']})")
                    try:
                        from strategy_config import (BLEND_FIT_ON_REFIT
                                                     as _fit_on_refit)
                    except Exception:
                        _fit_on_refit = False
                    if _fit_on_refit and fit_refit is not None:
                        fit = fit_refit
                        print("[BLEND] BLEND_FIT_ON_REFIT ON — deploying "
                              "the refit-based weight")
                    else:
                        fit = fit_stale
                    # Sharpe-grid DIAGNOSTIC (logged only, never deployed)
                    # SIG-R2-MASK: ON -> scored on entry-eligible rows only
                    # (blend_fit._policy_sharpe is a per-row take-set, so
                    # row-subsetting == a long veto); None = legacy.
                    _sess_rows = _session_entry_ok(all_times[rows],
                                                   asset_type)
                    _gsel = (_sess_rows if _sess_rows is not None
                             else slice(None))
                    w_grid = fit_blend_weight(
                        lstm_oof[_gsel], lgb_oof[_gsel], y_fit[_gsel],
                        objective='sharpe',
                        threshold=best_cfg['trade_threshold'],
                        shrink_lambda=0.0)
                    # Champion slot's persisted weight = cross-retrain memory
                    try:
                        w_prev = joblib.load(
                            f'{prefix}config_v2.pkl').get('lstm_weight')
                    except Exception:
                        w_prev = None
                    lstm_weight = smooth_across_retrains(fit['w'], w_prev)
                    blend_diag = {
                        'w_raw': (round(float(fit['w_raw']), 4)
                                  if fit['w_raw'] is not None else None),
                        'se': (round(float(fit['se']), 4)
                               if fit['se'] is not None else None),
                        'significant': bool(fit['significant']),
                        'n_fit': int(fit['n']),
                        'w_fit': round(float(fit['w']), 4),
                        'w_prev': (round(float(w_prev), 4)
                                   if w_prev is not None else None),
                        'w_sharpe_grid': round(float(w_grid), 4),
                        # R2C-02c instrumentation: both candidate fits +
                        # which one was deployed (Jetson A/B readout).
                        'w_stale': round(float(fit_stale['w']), 4),
                        'w_refit': (round(float(fit_refit['w']), 4)
                                    if fit_refit is not None else None),
                        'fit_source': ('refit'
                                       if (_fit_on_refit
                                           and fit_refit is not None)
                                       else 'stale'),
                    }
                    print(f"[BLEND] w_raw={blend_diag['w_raw']} "
                          f"se={blend_diag['se']} "
                          f"significant={blend_diag['significant']} -> "
                          f"w_fit={blend_diag['w_fit']} smoothed "
                          f"w={round(float(lstm_weight), 4)} "
                          f"(prev={blend_diag['w_prev']}, "
                          f"grid diag={blend_diag['w_sharpe_grid']})")
                    # Forecast-encompassing flag (B12)
                    if (fit['w_raw'] is not None
                            and fit['w_raw'] < 0.15):
                        print(f"[BLEND] OWNER FLAG: unshrunk NNLS "
                              f"w={fit['w_raw']:.3f} < 0.15 — LSTM leg "
                              f"near-encompassed; if this repeats on BOTH "
                              f"books, dropping torch from the live loops "
                              f"is the prize (do not auto-drop)")
                    # H2 (R2C-02d): the searched trade_threshold was
                    # scored on raw-LSTM fold preds but serves against
                    # the variance-compressed blend. Re-score it on the
                    # BLENDED folds[-1] val preds over the v3 grid; the
                    # reselected value DEPLOYS (mutating best_cfg before
                    # certification, so cert == deploy) only under
                    # default-OFF BLEND_THRESHOLD_RESELECT — otherwise
                    # this is the side-by-side log the Jetson A/B reads.
                    try:
                        from blend_fit import reselect_trade_threshold
                        _lstm_leg = (lstm_oof_refit
                                     if lstm_oof_refit is not None
                                     else lstm_oof)
                        _blend_val = (float(lstm_weight) * _lstm_leg
                                      + (1.0 - float(lstm_weight))
                                      * lgb_oof)
                        _tt_bids = (ticker_block_ids(rows,
                                                     ticker_boundaries)
                                    if _objective_v3() else None)
                        _old_thr = float(best_cfg['trade_threshold'])
                        _tt_veto = (~_sess_rows if _sess_rows is not None
                                    else None)  # SIG-R2-MASK

                        def _tt_score(p, yy, th, _b=_tt_bids, _v=_tt_veto):
                            return compute_sharpe(
                                p, yy, th, forward_bars=_fb,
                                asset_type=asset_type, block_ids=_b,
                                long_veto=_v)

                        _new_thr, _new_sc = reselect_trade_threshold(
                            _blend_val, y_fit,
                            v3_trade_threshold_range(asset_type),
                            _old_thr, _tt_score)
                        _cost = TXN_COST_PCT.get(asset_type, 0.6)
                        _n_old = len(simulate_trades(
                            _blend_val, y_fit, _old_thr, _fb, _cost,
                            block_ids=_tt_bids, long_veto=_tt_veto))
                        _n_new = len(simulate_trades(
                            _blend_val, y_fit, _new_thr, _fb, _cost,
                            block_ids=_tt_bids, long_veto=_tt_veto))
                        try:
                            from strategy_config import (
                                BLEND_THRESHOLD_RESELECT as _tt_deploy)
                        except Exception:
                            _tt_deploy = False
                        print(f"[BLEND] threshold reselect (H2): "
                              f"searched={_old_thr} "
                              f"(n_trades={_n_old} on blended val) -> "
                              f"blend-optimal={_new_thr} "
                              f"(n_trades={_n_new}, "
                              f"sharpe={_new_sc:.2f}) "
                              f"[{'DEPLOYED' if _tt_deploy else 'logged only'}]")
                        blend_diag['threshold_searched'] = _old_thr
                        blend_diag['threshold_blend'] = _new_thr
                        blend_diag['n_trades_searched'] = _n_old
                        blend_diag['n_trades_blend'] = _n_new
                        blend_diag['threshold_deployed'] = (
                            'blend' if _tt_deploy else 'searched')
                        if _tt_deploy:
                            best_cfg['trade_threshold'] = float(_new_thr)
                    except Exception as _te:
                        print(f"[BLEND] threshold reselect failed "
                              f"(non-fatal): {_te}")
                else:
                    print('[BLEND] LGB leg or OOF arrays unavailable — no '
                          'fitted blend weight')
            except Exception as e:
                lstm_weight = None
                print(f"[BLEND] blend fit failed ({e}) — no fitted "
                      f"blend weight")
            # H1 (R2C-02a): cert == deploy. Whenever the boosters WILL
            # ship, predict_now/backtest blend at cfg.get('lstm_weight',
            # 0.6) even if the fit above failed — so the certificate and
            # the shipped config must resolve a None weight to the SAME
            # default (blend_fit.DEFAULT_LSTM_WEIGHT, the certificate-
            # visible home of the pending B12.2 0.6->0.5 owner question).
            # Raw-LSTM certificates remain only when NO boosters ship.
            try:
                from blend_fit import (effective_lstm_weight,
                                       DEFAULT_LSTM_WEIGHT)
                _ship = bool(lgb_pack and lgb_pack[0] is not None)
                if _ship and lstm_weight is None:
                    print(f"[BLEND] H1 parity: no fitted weight but "
                          f"boosters ship — certifying AND deploying the "
                          f"default w={DEFAULT_LSTM_WEIGHT} "
                          f"(cert==deploy)")
                lstm_weight = effective_lstm_weight(lstm_weight, _ship)
            except Exception as e:
                print(f"[BLEND] H1 parity resolution failed ({e})")

        # Winner's-curse instrumentation (B12, direct-ship)
        _fs = best_state_holder.get('fold_sharpes') or []
        if _fs:
            print(f"[CURSE] deployable-edge point estimate = avg fold "
                  f"Sharpe {np.mean(_fs):.2f} (best-fold max "
                  f"{np.max(_fs):.2f} inflates by ~0.85*std="
                  f"{0.85 * np.std(_fs):.2f} over correlated folds)")

        # R2C-07 (FR-05): gate-time per-fold LSTM cross-sectional rank IC
        # from the winner's OOF fold-val predictions (print-only, ships
        # direct — no report keys, no flag; works in legacy mode too).
        try:
            _oof_rows_ic = best_state_holder.get('oof_rows') or []
            _oof_preds_ic = best_state_holder.get('oof_preds') or []
            if _oof_rows_ic and _oof_preds_ic:
                _fb_ic = best_cfg.get('forward_bars', 24)
                _rk_ic = (('tb', _fb_ic)
                          if best_cfg.get('target_kind') == 'tb' else _fb_ic)
                _ret_ic = all_returns_by_fb.get(_rk_ic)
                if _ret_ic is None:
                    _ret_ic = all_returns_by_fb.get(_fb_ic)
                if _ret_ic is None:
                    _ret_ic = next(v for k, v in all_returns_by_fb.items()
                                   if not isinstance(k, tuple))
                _fold_ids_ic = (best_state_holder.get('oof_fold_ids')
                                or list(range(len(_oof_rows_ic))))
                for _fi, _rws, _fps in zip(_fold_ids_ic, _oof_rows_ic,
                                           _oof_preds_ic):
                    _rws = np.asarray(_rws)
                    _ric = cs_rank_ic(np.asarray(_fps), _ret_ic[_rws],
                                      all_times[_rws])
                    _m = ('n/a' if _ric['mean'] is None
                          else f"{_ric['mean']:.4f}")
                    _s = ('n/a' if _ric['se'] is None
                          else f"{_ric['se']:.4f}")
                    print(f"[FR-05] fold {_fi} LSTM cs-rank-IC mean={_m} "
                          f"se={_s} n_groups={_ric['n_groups']} "
                          f"(skipped {_ric['n_skipped']} thin groups)")
                if asset_type == 'crypto':
                    print("[FR-05] caveat: crypto cross-sections are ~6 "
                          "names — per-fold rank ICs carry wide CIs; read "
                          "the mean/SE, not single timestamps")
        except Exception as _ie:
            print(f"[FR-05] per-fold cs-rank-IC failed (non-fatal): {_ie}")

        # --- FINAL HOLDOUT GATE -------------------------------------------
        # The winner was selected, early-stopped, AND scored on the same
        # validation slices across hundreds of trials. Before deployment it
        # must clear a time slice Optuna NEVER saw, deflated for the size
        # of the selection pool (Bailey & Lopez de Prado DSR).
        holdout_report = evaluate_on_holdout(
            ship_state, ship_scaler, best_cfg,
            all_features, all_returns_by_fb, all_times,
            tickers, ticker_boundaries, input_dim, asset_type,
            n_trials=n_trials_pool,
            all_tb_bars_by_fb=all_tb_bars_by_fb,
            lgb_booster=(lgb_pack[0] if _v3 and lgb_pack else None),
            q10_booster=(lgb_pack[1] if _v3 and lgb_pack else None),
            q10_floor=(lgb_pack[2] if _v3 and lgb_pack else None),
            lstm_weight=lstm_weight,
        )
        if holdout_report is not None:
            _record_holdout_gate(adaptive_state, asset_type)
        gate_ok = (holdout_report is not None
                   and holdout_report['sharpe'] > 0
                   and holdout_report['dsr'] >= holdout_report['dsr_min'])

        if not gate_ok:
            _hr = ({k: v for k, v in holdout_report.items()
                    if k != 'trade_returns'} if holdout_report else None)
            print(f"\nModel NOT saved: failed holdout gate "
                  f"({_hr})")
            print("A higher in-search score that cannot clear unseen data "
                  "is selection bias, not skill.")
        else:
            print(f"\nNew best model (score={new_score:.3f} > existing {existing_score:.3f}, "
                  f"holdout sharpe={holdout_report['sharpe']:.2f}, "
                  f"DSR={holdout_report['dsr']:.2f}):")
            for k, v in best_cfg.items():
                print(f"  {k}: {v}")

            config = {
                'model_version': 2,
                'input_dim': input_dim,
                'hidden_dim': best_cfg['hidden_dim'],
                'num_layers': best_cfg['num_layers'],
                'n_heads': best_cfg['n_heads'],
                'dropout': best_cfg['dropout'],
                'seq_len': best_cfg['seq_len'],
                'trade_threshold': best_cfg['trade_threshold'],
                'forward_bars': best_cfg.get('forward_bars', 24),
                'target_kind': best_cfg.get('target_kind', 'raw'),
                'huber_delta': best_cfg['huber_delta'],
                'prefix': args.prefix,
                'indicator_preset': preset_name,
                'holdout': holdout_report,
            }
            # T1: the fitted blend weight ships in the config — the key
            # predict_now.py / backtest.py already read (default 0.6).
            # R2C-02a: after the H1 parity resolution above, lstm_weight
            # is non-None whenever boosters ship — the config carries the
            # SAME effective weight the certificate scored (cert==deploy,
            # including the blend-fit failure path's default).
            if _v3 and lstm_weight is not None:
                config['lstm_weight'] = round(float(lstm_weight), 4)
                if blend_diag is not None:
                    config['blend_diag'] = blend_diag
            if refit_info:
                config['refit'] = refit_info
            # Shadow mode: with a champion already deployed, the gated
            # new model enters the CHALLENGER slot. It earns promotion
            # only by beating the champion on LIVE predictions (DM-HLN
            # test, shadow.py) — static gates can't prove that.
            save_prefix = prefix
            if args.shadow:
                try:
                    from shadow import challenger_prefix, champion_exists
                    if champion_exists(args.prefix):
                        cfg_pfx = challenger_prefix(args.prefix)
                        save_prefix = f'{cfg_pfx}_'
                        config['prefix'] = cfg_pfx
                        print(f"[SHADOW] champion present — saving as "
                              f"challenger ('{cfg_pfx}'); promotion via "
                              f"live DM test")
                except Exception as e:
                    print(f"[SHADOW] slot check failed ({e}) — saving as champion")

            # D12/B04.1: assemble the winner's OOF-prediction pack (fail-soft
            # — a pack-build failure never blocks the save; the npz simply
            # goes stale/absent and meta falls back loudly to in-sample).
            oof_pack = None
            try:
                if best_state_holder.get('oof_rows'):
                    from meta_label import oof_pack_from_folds
                    oof_pack = oof_pack_from_folds(
                        best_state_holder['oof_rows'],
                        best_state_holder['oof_preds'],
                        best_state_holder.get('oof_fold_ids'),
                        all_times, tickers, ticker_boundaries,
                        get_holdout_boundary(all_times))
            except Exception as e:
                print(f"[OOF] pack build failed (non-fatal): {e}")
                oof_pack = None

            # T1: under HYPERSEARCH_V3 the pre-gate LGB boosters ride the
            # atomic save (old boosters .prev-backed-up, ALL artifacts on
            # disk before the manifest). lgb_pack None -> nothing extra is
            # written and live falls back LSTM-only, exactly as a failed
            # legacy LGB training does today.
            extra_artifacts = None
            if _v3 and lgb_pack and lgb_pack[0] is not None:
                _booster, _q10b, _q10f, _n_q10 = lgb_pack
                extra_artifacts = {
                    f'{save_prefix}lgb_model.txt':
                        (lambda p, b=_booster: b.save_model(p)),
                }
                if _q10b is not None and _q10f is not None:
                    extra_artifacts[f'{save_prefix}lgb_q10.txt'] = \
                        (lambda p, b=_q10b: b.save_model(p))

                    def _write_q10_meta(p, fl=_q10f, nv=_n_q10,
                                        extra=getattr(_q10b,
                                                      '_r2c03_meta_extra',
                                                      None)):
                        # R2C-03: under LGB_REFIT_FULL the booster carries
                        # the in-sample-floor caveat; flag OFF -> extra is
                        # None and the json is byte-identical to legacy.
                        _m = {'alpha': 0.10,
                              'floor': round(fl, 6),
                              'val_rows': int(nv or 0)}
                        _m.update(extra or {})
                        with open(p, 'w') as f:
                            json.dump(_m, f)
                    extra_artifacts[f'{save_prefix}lgb_q10_meta.json'] = \
                        _write_q10_meta

            save_model_atomically(save_prefix, ship_state, best_cfg,
                                  input_dim, config, ship_scaler, feature_cols,
                                  score=new_score, oof_pack=oof_pack,
                                  extra_artifacts=extra_artifacts)
            model_saved = True

            # Train the LightGBM ensemble leg on the winning config
            # (legacy post-save path; under _v3 the legs were trained
            # pre-gate and saved atomically above)
            if not _v3:
                train_lgb_ensemble(save_prefix, best_scaler, best_cfg,
                                   all_features, all_returns_by_fb,
                                   all_times, all_label_times,
                                   tickers, ticker_boundaries,
                                   all_tb_bars_by_fb=all_tb_bars_by_fb)

            # R2C-07 (FR-08): retrain-gain ledger — score the incumbent
            # stack (champion artifacts on a --shadow challenger save,
            # .prev on a same-slot save) against the fresh stack on the
            # trailing week of purged bars, appending one paired row to
            # the adaptive-state ledger. Runs AFTER the boosters land in
            # both modes. Fail-soft, measurement-only — never blocks a
            # save; >=12 weekly rows feed the B03.3 IM block-t (owner
            # cadence decision).
            try:
                from retrain_ledger import record_retrain_gain
                # state=adaptive_state: update_after_search re-saves this
                # very object further down — appending into a freshly
                # loaded copy would be clobbered by that later save.
                record_retrain_gain(
                    prefix, save_prefix, asset_type,
                    all_features, feature_cols, all_returns_by_fb,
                    all_times, tickers, ticker_boundaries,
                    state=adaptive_state)
            except Exception as _le:
                print(f"[LEDGER] retrain-gain recording failed "
                      f"(non-fatal): {_le}")
    elif best_state_holder['state'] is not None:
        print(f"\nModel NOT saved: new best {new_score:.3f} <= existing {existing_score:.3f}")
        print("Existing model preserved (higher score).")
    else:
        print(f"\nNo new best found (prior best score={best_state_holder['score']:.3f})")

    # Param importance
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nParameter importance:")
            for param, imp in importance.items():
                print(f"  {param}: {imp:.3f}")
        except Exception:
            pass

    # Probability of backtest overfitting (coarse CSCV over fold scores)
    if len(completed) >= 10:
        try:
            from validation import pbo_from_fold_scores
            rows = [t.user_attrs.get('fold_sharpes') for t in completed]
            pbo = pbo_from_fold_scores([r for r in rows if r])
            if pbo is not None:
                print(f"\nPBO (coarse CSCV from fold Sharpes): {pbo:.2f}"
                      f"{'  WARNING: >0.5 — the IS winner is a below-median OOS performer more often than not (Bailey et al.)' if pbo > 0.5 else ''}")
        except Exception as e:
            print(f"  PBO computation failed: {e}")

    # Monte Carlo robustness test on completed trials
    if len(completed) >= 20 and best_state_holder['score'] > 0:
        try:
            scores = [t.value for t in completed if t.value and t.value > 0]
            if len(scores) >= 10:
                n_sims = 5000
                mc_scores = []
                for _ in range(n_sims):
                    sample = np.random.choice(scores, size=len(scores), replace=True)
                    mc_scores.append(np.mean(sample) - 0.5 * np.std(sample))
                mc_scores.sort()
                p5 = mc_scores[int(n_sims * 0.05)]
                p50 = mc_scores[int(n_sims * 0.50)]
                p95 = mc_scores[int(n_sims * 0.95)]
                print(f"\nMonte Carlo robustness ({n_sims} simulations):")
                print(f"  5th percentile: {p5:.3f}")
                print(f"  Median: {p50:.3f}")
                print(f"  95th percentile: {p95:.3f}")
                if p5 <= 0:
                    print("  WARNING: 5th percentile <= 0 — model may be overfit")
        except Exception as e:
            print(f"  Monte Carlo failed: {e}")

    # Deliberately AFTER the study reads above (importance/PBO/Monte-Carlo):
    # update_after_search may DELETE {prefix}v2_study.db on categorical
    # expansion, and a fresh pooled sqlite connection would otherwise
    # recreate an empty DB and silently blank those diagnostics.
    # Update adaptive state with results. When the holdout gate REJECTED the
    # winner, do NOT ratchet best_score — otherwise future (honest) models
    # would have to beat a score that belongs to an unsaved, overfit config.
    final_score = best_state_holder['score']
    if not model_saved:
        final_score = min(final_score, existing_score)
    final_params = best_state_holder.get('cfg', {})
    if final_score > 0 and final_params:
        adaptive_state = update_after_search(adaptive_state, final_score, final_params,
            study_db_path=db_path,
            new_trials_completed=n_new_completed,
            store_score=(rat['store_value']
                         if (_GATE_V2 and model_saved) else None))
        print(f"\n[ADAPTIVE] Updated state: mode={adaptive_state['mode']}, "
              f"cycles_without_improvement={adaptive_state['cycles_without_improvement']}")
        if adaptive_state.get('expansion_history'):
            latest = adaptive_state['expansion_history'][-1]
            if latest.get('expansions'):
                print(f"[ADAPTIVE] Expansions: {latest['expansions']}")
    elif final_params:
        # No improvement but still save params if this is initial run
        if not adaptive_state.get('best_params'):
            adaptive_state['best_params'] = final_params
            adaptive_state['best_score'] = final_score
            from adaptive_config import save_adaptive_state
            save_adaptive_state(adaptive_state)
        # Losing run: update_after_search is skipped but the trials still
        # accrued selection pressure against the holdout — persist them
        # (AFTER the save above: record_trials reloads from disk and saves).
        if n_new_completed > 0:
            from adaptive_config import record_trials
            record_trials(asset_type, n_new_completed,
                          event='search_no_update')
    elif n_new_completed > 0:
        # No params at all (every trial failed/pruned to nothing) — the
        # completed trials still count as selection pressure.
        from adaptive_config import record_trials
        record_trials(asset_type, n_new_completed, event='search_no_update')

    # Final pipeline status
    _pipeline_status['phase'] = 'complete'
    _pipeline_status['trial_current'] = num_trials
    _pipeline_status['best_score'] = best_state_holder['score']
    _pipeline_status['elapsed_sec'] = int(total_time)
    score_key = f'{asset_type}_final_score'
    _pipeline_status[score_key] = round(best_state_holder['score'], 4)
    _write_pipeline_status(_pipeline_status)

    # Save full log
    with open(f'hypersearch_{prefix}v2_log.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nTrials: {total_trials} total, {pruned} pruned")
    print(f"Log: hypersearch_{prefix}v2_log.json")


if __name__ == '__main__':
    args = parse_args()
    if args.no_status:
        _SKIP_STATUS = True
    lock_label = f"hypersearch_v2_{args.prefix or 'crypto'}"
    with acquire_for_training(lock_label):
        main()
