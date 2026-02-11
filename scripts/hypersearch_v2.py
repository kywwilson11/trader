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

NUM_TRIALS = 200
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 10
PRUNE_WARMUP_EPOCHS = 12
PRUNE_STARTUP_TRIALS = 50
NUM_FOLDS = 3
EMBARGO_MULTIPLIER = 1  # embargo = seq_len * this

FORWARD_BARS = [4, 8, 12, 18, 24]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path='training_data.csv', preset_override=None, max_rows=500_000):
    print("Loading data...")
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
        capped = []
        for ticker in tickers:
            tdf = df[df['Ticker'] == ticker].sort_index()
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

def compute_sharpe(predictions, actual_returns, threshold):
    """Simulate trades and compute annualized Sharpe ratio.

    Buy when pred > threshold, sell when pred < -threshold, else flat.
    Annualized for hourly bars: sqrt(252 * 6.5) ≈ sqrt(1638).
    """
    signals = np.where(predictions > threshold, 1,
              np.where(predictions < -threshold, -1, 0))
    trade_returns = signals * actual_returns
    if trade_returns.std() == 0:
        return 0.0
    return float((trade_returns.mean() / trade_returns.std()) * np.sqrt(1638))


# ---------------------------------------------------------------------------
# Sequence cache (reused from hypersearch_dual pattern)
# ---------------------------------------------------------------------------

class SeqCache:
    """Cache numpy sequence arrays by (seq_len, fold_idx, scaler_key).

    Builds sequences once per unique configuration, reuses across trials.
    """
    def __init__(self, all_features, tickers, ticker_boundaries):
        self._all_features = all_features
        self._tickers = tickers
        self._boundaries = ticker_boundaries
        self._cache = {}

    def get(self, seq_len, fold_idx, train_indices, val_indices):
        """Return (X_train, X_val) numpy arrays, fitting scaler on train only."""
        key = (seq_len, fold_idx)
        if key in self._cache:
            return self._cache[key]

        # Free old caches if memory is tight
        if len(self._cache) > 6:
            self._cache.clear()
            gc.collect()
            torch.cuda.empty_cache()

        t0 = time.time()

        # Fit scaler on train fold only (no leakage)
        scaler = RobustScaler()
        # Gather raw features for train indices
        train_features = self._all_features[train_indices]
        scaler.fit(train_features)

        # Scale ALL features using train-only scaler
        all_scaled = scaler.transform(self._all_features).astype(np.float32)

        offsets = np.arange(-seq_len, 0)
        X_train = np.ascontiguousarray(
            all_scaled[train_indices[:, None] + offsets[None, :]])
        X_val = np.ascontiguousarray(
            all_scaled[val_indices[:, None] + offsets[None, :]])

        elapsed = time.time() - t0
        print(f"  [CACHE] fold={fold_idx} seq_len={seq_len}: "
              f"{len(train_indices)} train + {len(val_indices)} val "
              f"({X_train.nbytes / 1e6:.0f}+{X_val.nbytes / 1e6:.0f} MB, "
              f"{elapsed:.1f}s)")

        result = (X_train, X_val, scaler)
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(all_features, all_returns_by_fb, tickers, ticker_boundaries,
                     input_dim, _state_cache, has_multi_horizon=True):

    MAX_TRIAL_SECONDS = 900

    seq_cache = SeqCache(all_features, tickers, ticker_boundaries)

    def objective(trial):
        trial_start = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        # --- Hyperparameters ---
        if has_multi_horizon:
            forward_bars = trial.suggest_categorical('forward_bars', FORWARD_BARS)
        else:
            forward_bars = 4

        seq_len = trial.suggest_categorical('seq_len', [12, 18, 24, 32])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        n_heads = trial.suggest_categorical('n_heads', [2, 4])
        dropout = trial.suggest_float('dropout', 0.10, 0.40, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 5e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True)
        huber_delta = trial.suggest_float('huber_delta', 0.5, 2.0, step=0.1)
        trade_threshold = trial.suggest_float('trade_threshold', 0.05, 1.0, step=0.01)
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
        except RuntimeError as e:
            err_str = str(e)
            print(f"  [ERROR] Trial {trial.number}: {err_str}")
            gc.collect()
            torch.cuda.empty_cache()
            if 'INTERNAL ASSERT FAILED' in err_str:
                raise RuntimeError(f"CUDA allocator corrupted: {err_str}") from e
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
            X_train, X_val, scaler = seq_cache.get(seq_len, fold_idx, train_indices, val_indices)
            n_train = len(train_indices)
            n_val = len(val_indices)

            y_train = trial_returns[train_indices]
            y_val = trial_returns[val_indices]

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

            best_val_loss = float('inf')
            best_state = None
            counter = 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                perm = np.random.permutation(n_train)
                for i in range(0, n_train, batch_size):
                    bi = perm[i:i + batch_size]
                    xb = torch.from_numpy(X_train[bi]).to(device)
                    yb = torch.from_numpy(y_train[bi]).to(device)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        pred = model(xb)
                        raw_loss = criterion(pred, yb)
                        # Weight by |actual_return| + 1.0 (emphasize big moves)
                        weights = torch.abs(yb) + 1.0
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
                    for i in range(0, n_val, batch_size):
                        xvb = torch.from_numpy(X_val[i:i + batch_size]).to(device)
                        yvb = torch.from_numpy(y_val[i:i + batch_size]).to(device)
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
                    torch.cuda.empty_cache()
                    raise optuna.TrialPruned()

                # Timeout
                if time.time() - trial_start > MAX_TRIAL_SECONDS:
                    print(f"  [TIMEOUT] Trial {trial.number} at epoch {epoch} fold {fold_idx}")
                    break

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
                    for i in range(0, n_val, batch_size):
                        xvb = torch.from_numpy(X_val[i:i + batch_size]).to(device)
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
            torch.cuda.empty_cache()

        avg_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0

        trial.set_user_attr('fold_sharpes', fold_sharpes)
        trial.set_user_attr('avg_sharpe', avg_sharpe)

        if best_fold_state is not None and avg_sharpe > 0:
            _state_cache[trial.number] = {
                'state': best_fold_state,
                'scaler': best_fold_scaler,
            }

        return avg_sharpe

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    num_trials = args.trials
    prefix = f'{args.prefix}_' if args.prefix else ''

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

        fb = cfg.get('forward_bars', 4)
        th = cfg.get('trade_threshold', '')
        sharpes_str = '/'.join(f'{s:.2f}' for s in fold_sharpes) if fold_sharpes else ''
        print(f"[{n:3d}] sharpe={score:.3f} folds=[{sharpes_str}] "
              f"| fb={fb} s={cfg.get('seq_len','')} h={cfg.get('hidden_dim','')} "
              f"l={cfg.get('num_layers','')} nh={cfg.get('n_heads','')} "
              f"d={cfg.get('dropout', ''):.2f} "
              f"lr={cfg.get('learning_rate', ''):.4f} "
              f"th={th if th == '' else f'{th:.2f}'} "
              f"hd={cfg.get('huber_delta', '')}"
              f"{tag}")

        results_log.append({
            'i': n, 'cfg': cfg, 'avg_sharpe': score,
            'fold_sharpes': fold_sharpes,
            'state': str(trial.state), 'time': elapsed,
        })

        if n % 10 == 0:
            with open(f'hypersearch_{prefix}v2_log.json', 'w') as f:
                json.dump(results_log, f, indent=2, default=str)
            print(f"  --- {elapsed/60:.1f}min elapsed, best sharpe={best_state_holder['score']:.3f}, "
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
    print(f"Optimizing: Sharpe ratio (walk-forward {NUM_FOLDS}-fold CV)")
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
            print(f"Prior best sharpe={best_state_holder['score']:.3f} — new trials must beat this")

    objective_fn = create_objective(all_features, all_returns_by_fb,
                                    tickers, ticker_boundaries, input_dim,
                                    _state_cache, has_multi_horizon=has_multi_horizon)
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

        print(f"\nBest model (sharpe={best_state_holder['score']:.3f}):")
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
            'forward_bars': best_cfg.get('forward_bars', 4),
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
        print(f"\nNo new best found (prior best sharpe={best_state_holder['score']:.3f})")

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

    # Save full log
    with open(f'hypersearch_{prefix}v2_log.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nTrials: {total_trials} total, {pruned} pruned")
    print(f"Log: hypersearch_{prefix}v2_log.json")


if __name__ == '__main__':
    main()
