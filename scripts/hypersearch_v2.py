"""
Regression-based Optuna hyperparameter search with walk-forward cross-validation.

Replaces the dual bear/bull classification approach with a single regression model
that predicts continuous returns, optimized via Sharpe ratio.

Key improvements over hypersearch_dual.py:
  - Single model (no separate bear/bull)
  - Huber loss with return-weighted emphasis
  - Walk-forward CV with expanding windows (3 folds)
  - Sharpe ratio as optimization objective
  - FP16 mixed precision for ~2x speedup on Jetson tensor cores
  - Stationary features only (no raw price/volume drift)

Usage:
    python scripts/hypersearch_v2.py --trials 200 --data training_data.csv --preset stationary
    python scripts/hypersearch_v2.py --trials 50 --prefix stock --data stock_training_data.csv
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import json
import os
import time
from datetime import datetime

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
PRUNE_WARMUP_EPOCHS = 12
PRUNE_STARTUP_TRIALS = 60
NUM_FOLDS = 3
EMBARGO_MULTIPLIER = 1  # embargo = seq_len * this

FORWARD_BARS = [12, 18, 24, 32, 48]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Cap CUDA allocator to prevent fatal kernel-level OOM on Jetson (unified memory).
# Without this, OOM triggers NvMapMemAllocInternalTagged errors that corrupt CUDA
# context and kill the process. With the cap, PyTorch raises catchable OutOfMemoryError.
if device.type == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.40)  # ~3GB of 7.6GB for CUDA
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

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
    parser.add_argument('--preset', type=str, default='stationary',
                        help='Indicator preset (default: stationary)')
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

def load_data(data_path='training_data.csv', preset_override=None, max_rows=500_000):
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

    exclude_cols = ['Ticker', 'Date', 'Datetime', 'NextClose']
    exclude_cols += [c for c in df.columns if c.startswith('Target_Return')]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Filter features by preset
    from indicator_config import load_indicator_config, get_preset_features
    preset_name = preset_override or load_indicator_config()["preset"]
    preset_features = get_preset_features(preset_name)
    if preset_features is not None:
        feature_cols = [c for c in feature_cols if c in preset_features]
    print(f"Preset: {preset_name} ({len(feature_cols)} features)")

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
    ticker_boundaries = {}

    offset = 0
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        features = tdf[feature_cols].values.astype(np.float32)
        all_features_list.append(features)

        for fb in FORWARD_BARS:
            col = f'Target_Return_{fb}'
            if col in tdf.columns:
                all_returns_dict[fb].append(tdf[col].values.astype(np.float32))
            else:
                all_returns_dict[fb].append(tdf['Target_Return'].values.astype(np.float32))

        ticker_boundaries[ticker] = (offset, offset + len(features))
        offset += len(features)

    all_features = np.vstack(all_features_list)
    all_returns_by_fb = {}
    for fb in FORWARD_BARS:
        if all_returns_dict[fb]:
            all_returns_by_fb[fb] = np.concatenate(all_returns_dict[fb])

    del all_features_list, all_returns_dict, df
    gc.collect()

    print(f"Contiguous arrays: {all_features.shape}, {all_features.nbytes / 1e6:.1f} MB")
    input_dim = all_features.shape[1]

    return (all_features, all_returns_by_fb, tickers, ticker_boundaries,
            feature_cols, input_dim, preset_name, has_multi_horizon)


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def get_walk_forward_folds(tickers, ticker_boundaries, seq_len, n_folds=NUM_FOLDS):
    """Generate expanding-window walk-forward folds.

    Returns list of (train_indices, val_indices) tuples.
    Each fold: train on first X%, validate on next chunk, with embargo gap.
    """
    embargo = seq_len * EMBARGO_MULTIPLIER
    folds = []

    # Compute total valid indices per ticker (need seq_len history)
    all_valid = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        valid = list(range(start + seq_len, end))
        all_valid.extend(valid)

    total = len(all_valid)
    all_valid = np.array(all_valid)

    # Expanding window: train on first 60%, 73%, 86%; val on next ~14% each
    for fold_idx in range(n_folds):
        train_end_pct = 0.60 + fold_idx * (0.40 / n_folds)
        val_end_pct = train_end_pct + (0.40 / n_folds)

        train_end = int(total * train_end_pct)
        val_start = min(train_end + embargo, total)
        val_end = min(int(total * val_end_pct), total)

        if val_start >= val_end:
            continue

        train_indices = all_valid[:train_end]
        val_indices = all_valid[val_start:val_end]
        folds.append((train_indices, val_indices))

    return folds


# ---------------------------------------------------------------------------
# Sharpe ratio computation
# ---------------------------------------------------------------------------

TRANSACTION_COST_BPS = 5  # 5 basis points round-trip cost per trade


def compute_sharpe(predictions, actual_returns, threshold, txn_cost_bps=TRANSACTION_COST_BPS):
    """Simulate trades and compute annualized Sharpe ratio.

    Buy when pred > threshold, sell when pred < -threshold, else flat.
    Sharpe computed only on bars where a trade occurs, annualized by
    the actual trade frequency (trades_per_bar * bars_per_year).
    Transaction costs (5 bps round-trip) are subtracted per trade.
    """
    signals = np.where(predictions > threshold, 1,
              np.where(predictions < -threshold, -1, 0))
    traded = signals != 0
    n_trades = traded.sum()
    if n_trades < 10:
        return 0.0
    trade_returns = signals[traded] * actual_returns[traded]
    # Subtract transaction costs (convert bps to percentage)
    trade_returns = trade_returns - (txn_cost_bps / 100.0)
    if trade_returns.std() < 1e-8:
        return 0.0
    # Annualize: trade_freq * bars/year.  1638 hourly stock bars/year.
    trade_freq = n_trades / len(signals)
    trades_per_year = trade_freq * 1638
    return float((trade_returns.mean() / trade_returns.std()) * np.sqrt(trades_per_year))


def compute_regime_sharpes(predictions, actual_returns, threshold):
    """Compute Sharpe in bull/bear/sideways regimes separately.

    Labels each bar based on a rolling 50-bar return:
      > +2% → bull, < -2% → bear, else → sideways.

    Returns dict: {'bull': sharpe, 'bear': sharpe, 'sideways': sharpe, 'min': min_sharpe}
    """
    if len(actual_returns) < 50:
        return {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0, 'min': 0.0}

    # Rolling 50-bar return for regime labeling
    cumret = np.cumsum(actual_returns)
    rolling_ret = np.zeros(len(actual_returns))
    rolling_ret[50:] = cumret[50:] - cumret[:-50]

    regimes = {}
    regimes['bull'] = rolling_ret > 2.0
    regimes['bear'] = rolling_ret < -2.0
    regimes['sideways'] = ~regimes['bull'] & ~regimes['bear']

    result = {}
    for name, mask in regimes.items():
        if mask.sum() < 10:
            result[name] = 0.0
            continue
        result[name] = compute_sharpe(predictions[mask], actual_returns[mask], threshold)

    result['min'] = min(result.values())
    return result


# ---------------------------------------------------------------------------
# Sequence cache (reused from hypersearch_dual pattern)
# ---------------------------------------------------------------------------

class SeqCache:
    """Cache ONE fold's numpy sequence arrays at a time.

    On Jetson Orin Nano (8GB unified CPU/GPU memory), keeping multiple folds
    cached simultaneously (~280 MB each) exhausts RAM and causes CUDA failures.
    Only one fold is cached; rebuilds take ~0.5s which is acceptable.
    """
    def __init__(self, all_features, tickers, ticker_boundaries):
        self._all_features = all_features
        self._tickers = tickers
        self._boundaries = ticker_boundaries
        self._key = None
        self._X_train = None
        self._X_val = None
        self._scaler = None

    def get(self, seq_len, fold_idx, train_indices, val_indices):
        """Return (X_train, X_val, scaler) numpy arrays, fitting scaler on train only."""
        key = (seq_len, fold_idx, len(train_indices), len(val_indices))
        if key == self._key and self._X_train is not None:
            return self._X_train, self._X_val, self._scaler

        # Free previous cache before allocating new one
        self._X_train = None
        self._X_val = None
        self._scaler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()

        # Fit scaler on train fold only (no leakage)
        scaler = RobustScaler()
        scaler.fit(self._all_features[train_indices])

        # Scale ALL features using train-only scaler, build sequences
        all_scaled = scaler.transform(self._all_features).astype(np.float32)

        offsets = np.arange(-seq_len, 0)
        X_train = np.ascontiguousarray(
            all_scaled[train_indices[:, None] + offsets[None, :]])
        X_val = np.ascontiguousarray(
            all_scaled[val_indices[:, None] + offsets[None, :]])

        del all_scaled  # free ~18 MB immediately
        gc.collect()

        elapsed = time.time() - t0
        print(f"  [CACHE] fold={fold_idx} seq_len={seq_len}: "
              f"{len(train_indices)} train + {len(val_indices)} val "
              f"({X_train.nbytes / 1e6:.0f}+{X_val.nbytes / 1e6:.0f} MB, "
              f"{elapsed:.1f}s)")

        self._key = key
        self._X_train = X_train
        self._X_val = X_val
        self._scaler = scaler
        return X_train, X_val, scaler


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(all_features, all_returns_by_fb, tickers, ticker_boundaries,
                     input_dim, _state_cache, has_multi_horizon=True,
                     adaptive_space=None):

    MAX_TRIAL_SECONDS = 900

    seq_cache = SeqCache(all_features, tickers, ticker_boundaries)

    # Use adaptive search space if provided, otherwise use defaults
    _space = adaptive_space or {}

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
        trade_threshold = trial.suggest_float('trade_threshold', tt_range[0], tt_range[1], step=0.01)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau'])

        cfg = {
            'seq_len': seq_len, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'n_heads': n_heads,
            'dropout': dropout, 'learning_rate': learning_rate,
            'batch_size': batch_size, 'weight_decay': weight_decay,
            'huber_delta': huber_delta, 'trade_threshold': trade_threshold,
            'scheduler': scheduler, 'forward_bars': forward_bars,
        }
        trial.set_user_attr('cfg', cfg)

        # Select returns for this horizon
        if forward_bars in all_returns_by_fb:
            trial_returns = all_returns_by_fb[forward_bars]
        else:
            trial_returns = list(all_returns_by_fb.values())[0]

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

        folds = get_walk_forward_folds(tickers, ticker_boundaries, seq_len)
        if not folds:
            return 0.0

        fold_sharpes = []
        best_fold_state = None
        best_fold_scaler = None
        best_fold_sharpe = -999

        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            # Filter out indices where the target return is NaN
            # (happens at the end of each ticker series for larger forward_bars)
            train_mask = ~np.isnan(trial_returns[train_indices])
            val_mask = ~np.isnan(trial_returns[val_indices])
            train_indices = train_indices[train_mask]
            val_indices = val_indices[val_mask]

            X_train, X_val, scaler = seq_cache.get(seq_len, fold_idx, train_indices, val_indices)
            n_train = len(train_indices)
            n_val = len(val_indices)

            y_train = trial_returns[train_indices]
            y_val = trial_returns[val_indices]

            # OOM retry: if batch doesn't fit, halve until it does (min 128)
            eff_batch_size = batch_size
            oom_retries = 0
            while True:
                try:
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

                    # Run one batch to test if this config fits in memory
                    model.train()
                    test_bi = np.arange(min(eff_batch_size, n_train))
                    xb = torch.from_numpy(X_train[test_bi]).to(device)
                    yb = torch.from_numpy(y_train[test_bi]).to(device)
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
                    break  # fits in memory
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if 'INTERNAL ASSERT' in str(e):
                        raise  # fatal, can't recover
                    del model, criterion, optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    oom_retries += 1
                    eff_batch_size //= 2
                    if eff_batch_size < 128:
                        raise  # give up, let outer handler catch it
                    print(f"  [OOM-RETRY] fold {fold_idx}: batch {eff_batch_size*2}→{eff_batch_size}")

            if oom_retries > 0:
                print(f"  [OOM-RETRY] fold {fold_idx}: training with batch_size={eff_batch_size} "
                      f"(was {batch_size})")

            best_val_loss = float('inf')
            best_state = None
            counter = 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                perm = np.random.permutation(n_train)
                for i in range(0, n_train, eff_batch_size):
                    bi = perm[i:i + eff_batch_size]
                    xb = torch.from_numpy(X_train[bi]).to(device)
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
                        xvb = torch.from_numpy(X_val[i:i + eff_batch_size]).to(device)
                        yvb = torch.from_numpy(y_val[i:i + eff_batch_size]).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model(xvb)
                        val_loss_sum += nn.functional.huber_loss(vo, yvb).item() * xvb.size(0)
                        val_preds.append(vo.cpu().numpy())

                val_loss = val_loss_sum / n_val
                val_preds_np = np.concatenate(val_preds)

                if scheduler_type == 'plateau' and sched:
                    sched.step(val_loss)

                # Compute Sharpe for pruning feedback
                epoch_sharpe = compute_sharpe(val_preds_np, y_val, trade_threshold)
                trial.report(epoch_sharpe, epoch * NUM_FOLDS + fold_idx)

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
                    if counter >= EARLY_STOP_PATIENCE:
                        break

            # Compute fold Sharpe with best model
            if best_state is not None:
                model.load_state_dict(best_state)
                model.eval()
                final_preds = []
                with torch.inference_mode():
                    for i in range(0, n_val, eff_batch_size):
                        xvb = torch.from_numpy(X_val[i:i + eff_batch_size]).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model(xvb)
                        final_preds.append(vo.cpu().numpy())
                final_preds_np = np.concatenate(final_preds)
                fold_sharpe = compute_sharpe(final_preds_np, y_val, trade_threshold)
            else:
                fold_sharpe = 0.0

            fold_sharpes.append(fold_sharpe)

            # Track best fold for saving
            if fold_sharpe > best_fold_sharpe:
                best_fold_sharpe = fold_sharpe
                best_fold_state = best_state
                best_fold_scaler = scaler

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0
        std_sharpe = np.std(fold_sharpes) if len(fold_sharpes) > 1 else 0.0
        # Risk-adjusted score: penalize inconsistency across folds.
        # A model with [2.8, 2.9, 2.7] (score=2.80) beats [4.0, 4.0, -1.0] (score=2.83)
        score = avg_sharpe - 0.5 * std_sharpe

        # Regime-aware penalty: penalize models with negative Sharpe in any regime
        if best_fold_state is not None and len(val_indices) > 50:
            model_tmp = None
            try:
                model_tmp = RegressionLSTM(input_dim, hidden_dim, num_layers,
                                            dropout, n_heads).to(device)
                model_tmp.load_state_dict(best_fold_state)
                model_tmp.eval()
                all_preds = []
                with torch.inference_mode():
                    for i in range(0, n_val, eff_batch_size):
                        xvb = torch.from_numpy(X_val[i:i + eff_batch_size]).to(device)
                        with torch.amp.autocast('cuda', enabled=use_amp):
                            vo = model_tmp(xvb)
                        all_preds.append(vo.cpu().numpy())
                all_preds_np = np.concatenate(all_preds)
                regime_sharpes = compute_regime_sharpes(all_preds_np, y_val, trade_threshold)
                trial.set_user_attr('regime_sharpes', regime_sharpes)
                # Penalize if any regime has negative Sharpe
                if regime_sharpes['min'] < -0.5:
                    score *= 0.7  # 30% penalty
            except Exception as e:
                print(f"  [REGIME] Penalty eval failed: {e}")
            finally:
                del model_tmp
                gc.collect()

        trial.set_user_attr('fold_sharpes', fold_sharpes)
        trial.set_user_attr('avg_sharpe', avg_sharpe)
        trial.set_user_attr('std_sharpe', std_sharpe)

        if best_fold_state is not None and score > 0:
            # Only keep this trial's state; clear old entries to avoid memory leak
            _state_cache.clear()
            _state_cache[trial.number] = {
                'state': best_fold_state,
                'scaler': best_fold_scaler,
            }

        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    if args.fresh and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing study DB: {db_path}")

    storage = f'sqlite:///{db_path}'

    (all_features, all_returns_by_fb, tickers, ticker_boundaries,
     feature_cols, input_dim, preset_name,
     has_multi_horizon) = load_data(args.data, preset_override=args.preset,
                                     max_rows=args.max_rows)

    best_state_holder = {'state': None, 'scaler': None, 'score': 0.0, 'cfg': None}
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
                trials_since_improvement = 0
                tag = " ** BEST **"

        fb = cfg.get('forward_bars', 12)
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
        sampler=optuna.samplers.TPESampler(n_startup_trials=PRUNE_STARTUP_TRIALS),
    )

    prior_trials = len(study.trials)
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
                                    tickers, ticker_boundaries, input_dim,
                                    _state_cache, has_multi_horizon=has_multi_horizon,
                                    adaptive_space=adaptive_space)
    study.optimize(objective_fn, n_trials=num_trials, callbacks=[trial_callback],
                   catch=(Exception,))

    # --- Results ---
    total_time = time.time() - t0
    total_trials = len(study.trials)
    print(f"\n{'='*70}")
    print(f"DONE: {num_trials} new trials in {total_time/60:.1f}min ({total_trials} total in study)")
    print(f"{'='*70}")

    if best_state_holder['state'] is not None:
        best_cfg = best_state_holder['cfg']
        best_scaler = best_state_holder['scaler']

        print(f"\nBest model (score={best_state_holder['score']:.3f}, mean-0.5*std):")
        for k, v in best_cfg.items():
            print(f"  {k}: {v}")

        mdl = RegressionLSTM(input_dim, best_cfg['hidden_dim'],
                              best_cfg['num_layers'], best_cfg['dropout'],
                              best_cfg['n_heads'])
        mdl.load_state_dict(best_state_holder['state'])
        torch.save(mdl.state_dict(), f'{prefix}model_v2.pth')

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
            'huber_delta': best_cfg['huber_delta'],
            'prefix': args.prefix,
            'indicator_preset': preset_name,
        }
        joblib.dump(config, f'{prefix}config_v2.pkl')
        joblib.dump(best_scaler, f'{prefix}scaler_v2.pkl')
        joblib.dump(feature_cols, f'{prefix}feature_cols_v2.pkl')

        print(f"\nModel saved: {prefix}model_v2.pth")
        print(f"Config saved: {prefix}config_v2.pkl")
        print(f"Scaler saved: {prefix}scaler_v2.pkl")
        print(f"Features saved: {prefix}feature_cols_v2.pkl")
    else:
        print(f"\nNo new best found (prior best score={best_state_holder['score']:.3f})")

    # Update adaptive state with results
    final_score = best_state_holder['score']
    final_params = best_state_holder.get('cfg', {})
    if final_score > 0 and final_params:
        adaptive_state = update_after_search(adaptive_state, final_score, final_params,
                                                   study_db_path=db_path)
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
