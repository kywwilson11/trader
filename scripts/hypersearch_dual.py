"""
Dual-mode Optuna hyperparameter search — specialized bear or bull models.

Usage:
    python hypersearch_dual.py --target bear   # Optimize bear class accuracy
    python hypersearch_dual.py --target bull   # Optimize bull class accuracy

Each mode runs 250 trials and saves to {target}_model.pth / {target}_config.pkl.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CryptoLSTM
from sklearn.preprocessing import RobustScaler
import joblib
import gc
import json
import os
import time
import optuna
from optuna.pruners import MedianPruner

NUM_TRIALS = 200
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 12
PRUNE_WARMUP_EPOCHS = 15       # don't prune until model has had time to learn
PRUNE_STARTUP_TRIALS = 50      # match TPE's random exploration phase
TRAIN_RATIO = 0.8
NUM_CLASSES = 3

# Multi-horizon forward return targets (must match harvest scripts)
FORWARD_BARS = [4, 8, 12, 16, 24, 32]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")



def parse_args():
    parser = argparse.ArgumentParser(description='Dual-mode hyperparameter search')
    parser.add_argument('--target', required=True, choices=['bear', 'bull'],
                        help='Which class to optimize: bear or bull')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS,
                        help=f'Number of trials (default: {NUM_TRIALS})')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing study DB and start fresh')
    parser.add_argument('--data', type=str, default='training_data.csv',
                        help='Path to training CSV (default: training_data.csv)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for output files (e.g. "stock" -> stock_bear_model.pth)')
    parser.add_argument('--fixed-threshold', type=float, default=None,
                        help='Use a fixed bull_threshold instead of searching (for shared threshold between bear/bull)')
    parser.add_argument('--preset', type=str, default=None,
                        help='Indicator preset: minimal, standard, full')
    parser.add_argument('--max-rows', type=int, default=500_000,
                        help='Max total rows to load (default: 500000). Keeps most recent rows per ticker to prevent OOM.')
    return parser.parse_args()


def load_data(data_path='training_data.csv', preset_override=None, max_rows=500_000):
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Dataset: {len(df)} rows")

    # Detect multi-horizon columns — only keep columns matching FORWARD_BARS
    valid_target_cols = {f'Target_Return_{fb}' for fb in FORWARD_BARS}
    csv_target_cols = [c for c in df.columns if c.startswith('Target_Return_') and c in valid_target_cols]
    # Drop stale horizon columns not in FORWARD_BARS (e.g. Target_Return_1, _2)
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
    # Exclude ALL target return columns from features
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

    scaler_X = RobustScaler()
    scaler_X.fit(df[feature_cols].values)

    all_scaled_list = []
    # Dict of forward_bars -> returns array, for multi-horizon support
    all_returns_dict = {fb: [] for fb in FORWARD_BARS}
    # Also keep legacy Target_Return
    all_returns_legacy = []
    ticker_boundaries = {}

    offset = 0
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        scaled = scaler_X.transform(tdf[feature_cols].values).astype(np.float32)
        all_scaled_list.append(scaled)

        # Collect returns for each horizon
        for fb in FORWARD_BARS:
            col = f'Target_Return_{fb}'
            if col in tdf.columns:
                all_returns_dict[fb].append(tdf[col].values.astype(np.float32))
            else:
                # Legacy dataset: only Target_Return exists, map to fb=4
                all_returns_dict[fb].append(tdf['Target_Return'].values.astype(np.float32))

        all_returns_legacy.append(tdf['Target_Return'].values.astype(np.float32))
        ticker_boundaries[ticker] = (offset, offset + len(scaled))
        offset += len(scaled)

    all_scaled = np.vstack(all_scaled_list)
    # Concatenate per-horizon returns
    all_returns_by_fb = {}
    for fb in FORWARD_BARS:
        if all_returns_dict[fb]:
            all_returns_by_fb[fb] = np.concatenate(all_returns_dict[fb])
    all_returns_legacy = np.concatenate(all_returns_legacy)

    del all_scaled_list, all_returns_dict, df
    gc.collect()

    print(f"Contiguous arrays: {all_scaled.shape}, {all_scaled.nbytes / 1e6:.1f} MB")
    input_dim = all_scaled.shape[1]

    return (all_scaled, all_returns_legacy, all_returns_by_fb, tickers,
            ticker_boundaries, scaler_X, feature_cols, input_dim, preset_name,
            has_multi_horizon)


def get_split_indices(tickers, ticker_boundaries, seq_len):
    """Get train/val indices — depends only on seq_len, not threshold."""
    train_indices = []
    val_indices = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        ticker_valid = list(range(start + seq_len, end))
        split = int(len(ticker_valid) * TRAIN_RATIO)
        train_indices.extend(ticker_valid[:split])
        # Embargo gap: skip seq_len bars to prevent sequence overlap leakage
        val_start = min(split + seq_len, len(ticker_valid))
        val_indices.extend(ticker_valid[val_start:])
    return np.array(train_indices), np.array(val_indices)


def classify_returns(all_returns, bull_thresh):
    """Assign bear/neutral/bull classes based on threshold."""
    classes = np.ones(len(all_returns), dtype=np.int64)
    classes[all_returns > bull_thresh] = 2
    classes[all_returns < -bull_thresh] = 0
    return classes


class SeqCache:
    """Cache numpy sequence arrays by seq_len.

    Train/val indices depend only on seq_len (not threshold or forward_bars),
    so X_train and X_val numpy arrays can be reused across trials with the
    same seq_len. Only labels change per trial (cheap to rebuild).

    On Jetson unified memory, keeping data in numpy and transferring per-batch
    via torch.from_numpy().to(device) is faster than GPU-resident tensors
    (avoids GPU scatter-gather kernel overhead on small GPU).
    """
    def __init__(self, all_scaled, tickers, ticker_boundaries):
        self._all_scaled = all_scaled
        self._tickers = tickers
        self._boundaries = ticker_boundaries
        self._cached_seq_len = None
        self._X_train = None
        self._X_val = None
        self._train_arr = None
        self._val_arr = None

    def get(self, seq_len):
        """Return (X_train_np, X_val_np, train_arr, val_arr) — numpy arrays."""
        if seq_len == self._cached_seq_len:
            return self._X_train, self._X_val, self._train_arr, self._val_arr

        # Free old cache
        self._X_train = None
        self._X_val = None
        gc.collect()

        t0 = time.time()
        train_arr, val_arr = get_split_indices(
            self._tickers, self._boundaries, seq_len)

        offsets = np.arange(-seq_len, 0)

        # Vectorized numpy gather — stays on CPU, transferred per-batch
        self._X_train = np.ascontiguousarray(
            self._all_scaled[train_arr[:, None] + offsets[None, :]])
        self._X_val = np.ascontiguousarray(
            self._all_scaled[val_arr[:, None] + offsets[None, :]])

        self._train_arr = train_arr
        self._val_arr = val_arr
        self._cached_seq_len = seq_len

        elapsed = time.time() - t0
        print(f"  [CACHE] Built seq_len={seq_len}: "
              f"{len(train_arr)} train + {len(val_arr)} val "
              f"({self._X_train.nbytes / 1e6:.0f}+{self._X_val.nbytes / 1e6:.0f} MB, "
              f"{elapsed:.1f}s)")

        return self._X_train, self._X_val, self._train_arr, self._val_arr


def create_objective(target, all_scaled, all_returns, all_returns_by_fb,
                     tickers, ticker_boundaries, input_dim, _state_cache,
                     fixed_threshold=None, has_multi_horizon=True):
    # target class index: bear=0, bull=2
    target_class = 0 if target == 'bear' else 2

    MAX_TRIAL_SECONDS = 900  # kill any trial running longer than 15 min

    # Persistent cache: numpy sequence arrays reused across trials with same seq_len
    seq_cache = SeqCache(all_scaled, tickers, ticker_boundaries)

    def objective(trial):
        trial_start = time.time()

        # Forward bars horizon (searchable if multi-horizon data available)
        if has_multi_horizon:
            forward_bars = trial.suggest_categorical('forward_bars', FORWARD_BARS)
        else:
            forward_bars = 4  # legacy single-horizon

        # Select returns for this horizon
        if forward_bars in all_returns_by_fb:
            trial_returns = all_returns_by_fb[forward_bars]
        else:
            trial_returns = all_returns  # fallback to legacy

        seq_len = trial.suggest_categorical('seq_len', [12, 18, 24])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.05, 0.45, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])

        # Adaptive threshold ranges based on forward_bars horizon
        if fixed_threshold is not None:
            bull_threshold = fixed_threshold
        elif forward_bars <= 4:
            bull_threshold = trial.suggest_float('bull_threshold', 0.15, 0.50, step=0.01)
        elif forward_bars <= 12:
            bull_threshold = trial.suggest_float('bull_threshold', 0.30, 1.00, step=0.01)
        else:  # 16, 24, 32
            bull_threshold = trial.suggest_float('bull_threshold', 0.50, 2.00, step=0.05)

        weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none'])

        cfg = {
            'seq_len': seq_len, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'dropout': dropout,
            'learning_rate': learning_rate, 'batch_size': batch_size,
            'bull_threshold': bull_threshold, 'weight_decay': weight_decay,
            'scheduler': scheduler, 'forward_bars': forward_bars,
        }

        # Store config early so callback can log params even on rejected/failed trials
        trial.set_user_attr('cfg', cfg)

        try:
            return _train_and_evaluate(
                trial, trial_start, cfg, target_class,
                trial_returns, input_dim, _state_cache, seq_cache,
            )
        except RuntimeError as e:
            # CUDA OOM or other GPU errors — clean up and return 0
            print(f"  [ERROR] Trial {trial.number}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return 0.0

    def _train_and_evaluate(trial, trial_start, cfg, target_class,
                            trial_returns, input_dim, _state_cache, seq_cache):
        seq_len = cfg['seq_len']
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        dropout = cfg['dropout']
        learning_rate = cfg['learning_rate']
        batch_size = cfg['batch_size']
        bull_threshold = cfg['bull_threshold']
        weight_decay = cfg['weight_decay']
        scheduler = cfg['scheduler']

        # Get cached numpy arrays (instant if seq_len unchanged, ~1s on cache miss)
        X_train_np, X_val_np, train_arr, val_arr = seq_cache.get(seq_len)
        n_train = len(train_arr)
        n_val = len(val_arr)

        # Compute labels per trial (cheap — only depends on threshold + forward_bars)
        classes = classify_returns(trial_returns, bull_threshold)
        train_labels = classes[train_arr]
        unique = np.unique(train_labels)
        if len(unique) < 3:
            return 0.0

        counts = np.bincount(train_labels, minlength=3).astype(np.float64)
        total = counts.sum()
        weights = total / (3.0 * counts)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        val_labels = classes[val_arr]
        y_val_t = torch.tensor(val_labels, dtype=torch.long, device=device)

        model = CryptoLSTM(input_dim, hidden_dim, num_layers,
                            dropout, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_t)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)

        if scheduler == 'cosine':
            sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
        elif scheduler == 'plateau':
            sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        else:
            sched = None

        best_val_acc = 0.0
        best_state = None
        best_preds = None
        best_labels = None
        counter = 0
        best_epoch_score = 0.0  # track best intermediate score for timeout recovery
        timed_out = False

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = np.random.permutation(n_train)
            for i in range(0, n_train, batch_size):
                bi = perm[i:i + batch_size]
                xb = torch.from_numpy(X_train_np[bi]).to(device)
                yb = torch.tensor(train_labels[bi], dtype=torch.long, device=device)
                out = model(xb)
                loss = criterion(out, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler == 'cosine' and sched:
                sched.step()

            model.eval()
            with torch.inference_mode():
                all_preds_list = []
                val_loss_sum = 0.0
                for i in range(0, n_val, batch_size):
                    xvb = torch.from_numpy(X_val_np[i:i + batch_size]).to(device)
                    yvb = torch.tensor(val_labels[i:i + batch_size], dtype=torch.long, device=device)
                    vo = model(xvb)
                    val_loss_sum += criterion(vo, yvb).item() * xvb.size(0)
                    all_preds_list.append(vo.argmax(1))
                ep_preds_t = torch.cat(all_preds_list)

            val_acc = (ep_preds_t == y_val_t).float().mean().item()
            val_loss = val_loss_sum / n_val

            if scheduler == 'plateau' and sched:
                sched.step(val_loss)

            # Compute epoch composite_score for pruner (same metric as final objective)
            ep_preds = ep_preds_t.cpu().numpy()
            ep_tp = int(((ep_preds == target_class) & (val_labels == target_class)).sum())
            ep_fp = int(((ep_preds == target_class) & (val_labels != target_class)).sum())
            ep_fn = int(((ep_preds != target_class) & (val_labels == target_class)).sum())
            ep_prec = ep_tp / (ep_tp + ep_fp) if (ep_tp + ep_fp) > 0 else 0.0
            ep_rec = ep_tp / (ep_tp + ep_fn) if (ep_tp + ep_fn) > 0 else 0.0
            ep_f1 = 2 * ep_prec * ep_rec / (ep_prec + ep_rec) if (ep_prec + ep_rec) > 0 else 0.0
            ep_bal = sum((ep_preds[val_labels == c] == c).mean() if (val_labels == c).sum() > 0 else 0.0
                         for c in range(NUM_CLASSES)) / NUM_CLASSES
            ep_cat = (int(((val_labels == 0) & (ep_preds == 2)).sum()) +
                      int(((val_labels == 2) & (ep_preds == 0)).sum())) / n_val
            epoch_score = max(ep_f1 * 0.5 + ep_bal * 0.2 - ep_cat * 0.3, 0.0)
            if epoch_score > best_epoch_score:
                best_epoch_score = epoch_score

            trial.report(epoch_score, epoch)
            if epoch >= PRUNE_WARMUP_EPOCHS and trial.should_prune():
                del model, weights_t
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            # Hard time limit per trial
            if time.time() - trial_start > MAX_TRIAL_SECONDS:
                print(f"  [TIMEOUT] Trial {trial.number} exceeded {MAX_TRIAL_SECONDS}s at epoch {epoch}, "
                      f"best_epoch_score={best_epoch_score:.3f}")
                timed_out = True
                break

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_preds = ep_preds
                best_labels = val_labels
                counter = 0
            else:
                counter += 1
                if counter >= EARLY_STOP_PATIENCE:
                    break

        # Per-class accuracy + confusion matrix analysis (using saved preds — no re-inference)
        per_class = {}
        composite_score = 0.0
        n_samples = 0
        target_f1 = 0.0
        catastrophic_rate = 1.0
        if best_preds is None and timed_out and best_epoch_score > 0:
            # Timeout before any val_acc improvement — use last epoch's preds
            print(f"  [TIMEOUT-RECOVER] No best_preds saved, using last epoch preds (score={best_epoch_score:.3f})")
            best_preds = ep_preds
            best_labels = val_labels
        if best_preds is not None:
            ap, al = best_preds, best_labels
            n_samples = len(al)
            for c, n in [(0, 'bear'), (1, 'neutral'), (2, 'bull')]:
                m = al == c
                per_class[n] = float((ap[m] == c).mean()) if m.sum() > 0 else 0.0

        # Reject non-discriminative models: every class must exceed a floor.
        MIN_CLASS_ACC = 0.10
        pc_vals = list(per_class.values())
        if pc_vals and min(pc_vals) < MIN_CLASS_ACC:
            del model, weights_t
            gc.collect()
            torch.cuda.empty_cache()
            # On timeout, return best intermediate score instead of 0
            if timed_out and best_epoch_score > 0:
                print(f"  [TIMEOUT-RECOVER] Using best_epoch_score={best_epoch_score:.3f}")
                return best_epoch_score
            return 0.0

        # --- Confusion-matrix scoring for real-world trading ---
        if n_samples > 0:
            tp = int(((ap == target_class) & (al == target_class)).sum())
            fp = int(((ap == target_class) & (al != target_class)).sum())
            fn = int(((ap != target_class) & (al == target_class)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            target_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            balanced_acc = sum(pc_vals) / len(pc_vals) if pc_vals else 0.0

            bear_as_bull = int(((al == 0) & (ap == 2)).sum())
            bull_as_bear = int(((al == 2) & (ap == 0)).sum())
            catastrophic_rate = (bear_as_bull + bull_as_bear) / n_samples

            composite_score = target_f1 * 0.5 + balanced_acc * 0.2 - catastrophic_rate * 0.3
            composite_score = max(composite_score, 0.0)

        trial.set_user_attr('per_class', per_class)
        trial.set_user_attr('composite_score', composite_score)
        trial.set_user_attr('target_f1', target_f1)
        trial.set_user_attr('catastrophic_rate', catastrophic_rate)
        trial.set_user_attr('val_acc', best_val_acc)
        trial.set_user_attr('cfg', cfg)

        _state_cache[trial.number] = best_state

        del model, weights_t
        gc.collect()
        torch.cuda.empty_cache()

        return composite_score

    return objective


def main():
    args = parse_args()
    target = args.target
    num_trials = args.trials
    prefix = f'{args.prefix}_' if args.prefix else ''

    # Persistent SQLite storage — Bayesian memory survives across invocations
    db_path = f'{prefix}{target}_study.db'
    study_name = f'{prefix}{target}_search'

    if args.fresh and os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing study DB: {db_path}")

    storage = f'sqlite:///{db_path}'

    (all_scaled, all_returns, all_returns_by_fb, tickers, ticker_boundaries,
     scaler_X, feature_cols, input_dim, preset_name,
     has_multi_horizon) = load_data(args.data, preset_override=args.preset,
                                     max_rows=args.max_rows)

    # Track best model weights in memory (can't store in SQLite efficiently)
    best_state_holder = {'state': None, 'score': 0.0, 'cfg': None, 'val_acc': 0.0, 'per_class': {}}

    # Shared cache: trial.number -> model state_dict (objective writes, callback reads)
    _state_cache = {}

    # Callback state
    results_log = []
    t0 = time.time()
    trials_since_improvement = 0

    def trial_callback(study, trial):
        nonlocal trials_since_improvement

        elapsed = time.time() - t0
        n = trial.number + 1
        score = trial.value if trial.value is not None else 0.0
        pc = trial.user_attrs.get('per_class', {})
        cfg = trial.user_attrs.get('cfg', {})
        val_acc = trial.user_attrs.get('val_acc', 0.0)

        tag = ""
        trials_since_improvement += 1
        if trial.state == optuna.trial.TrialState.PRUNED:
            tag = " [PRUNED]"
        elif score > best_state_holder['score'] and val_acc > 0.34:
            # Grab model weights from shared cache (keyed by trial number)
            state = _state_cache.get(trial.number)
            if state is not None:
                best_state_holder['state'] = state
                best_state_holder['score'] = score
                best_state_holder['cfg'] = cfg
                best_state_holder['val_acc'] = val_acc
                best_state_holder['per_class'] = pc
                trials_since_improvement = 0
                tag = " ** BEST **"

        d = cfg.get('dropout', '')
        lr = cfg.get('learning_rate', '')
        th = cfg.get('bull_threshold', '')
        fb = cfg.get('forward_bars', 4)
        f1 = trial.user_attrs.get('target_f1', 0.0)
        cat = trial.user_attrs.get('catastrophic_rate', 0.0)
        print(f"[{n:3d}] score={score:.3f} F1={f1:.2f} cat={cat:.2f} "
              f"B:{pc.get('bear',0):.0%} N:{pc.get('neutral',0):.0%} U:{pc.get('bull',0):.0%} "
              f"| fb={fb} s={cfg.get('seq_len','')} h={cfg.get('hidden_dim','')} "
              f"l={cfg.get('num_layers','')} d={d if d == '' else f'{d:.2f}'} "
              f"lr={lr if lr == '' else f'{lr:.4f}'} th={th if th == '' else f'{th:.2f}'}"
              f"{tag}")

        results_log.append({
            'i': n, 'cfg': cfg, 'val_acc': val_acc,
            'composite_score': score, 'target_f1': f1,
            'catastrophic_rate': cat, 'per_class': pc,
            'state': str(trial.state),
            'time': elapsed,
        })

        if n % 10 == 0:
            with open(f'hypersearch_{prefix}{target}_log.json', 'w') as f:
                json.dump(results_log, f, indent=2, default=str)
            print(f"  --- {elapsed/60:.1f}min elapsed, best score={best_state_holder['score']:.3f}, "
                  f"total trials in study={len(study.trials)}, "
                  f"{trials_since_improvement} since last improvement ---")

    # --- MAIN SEARCH ---
    # Load or create persistent study
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
    print(f"OPTUNA {target.upper()} MODEL SEARCH: {num_trials} new trials (TPE + pruning)")
    print(f"Optimizing: F1 * 0.5 + balanced_acc * 0.2 - catastrophic * 0.3 (target={target})")
    if has_multi_horizon:
        print(f"Multi-horizon: forward_bars in {FORWARD_BARS}")
    print(f"Max rows: {args.max_rows:,}")
    print(f"Resuming from {prior_trials} prior trials in {db_path}")
    if args.fixed_threshold is not None:
        print(f"Using FIXED threshold: {args.fixed_threshold:.2f} (shared from bear model)")
    print(f"{'='*70}\n")

    # Seed best_state_holder from study's historical best (score only — no weights)
    if prior_trials > 0:
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.user_attrs.get('val_acc', 0) <= 0.34:
                continue
            # Skip non-discriminative trials (any class below floor)
            pc = t.user_attrs.get('per_class', {})
            pc_vals = list(pc.values())
            if pc_vals and min(pc_vals) < 0.10:
                continue
            if (t.value or 0) > best_state_holder['score']:
                best_state_holder['score'] = t.value
                best_state_holder['cfg'] = t.user_attrs.get('cfg', {})
                best_state_holder['val_acc'] = t.user_attrs.get('val_acc', 0)
                best_state_holder['per_class'] = t.user_attrs.get('per_class', {})
        if best_state_holder['score'] > 0:
            pc = best_state_holder['per_class']
            print(f"Prior best score={best_state_holder['score']:.3f} "
                  f"B:{pc.get('bear',0):.0%} N:{pc.get('neutral',0):.0%} U:{pc.get('bull',0):.0%} "
                  f"— new trials must beat this")

    objective_fn = create_objective(target, all_scaled, all_returns, all_returns_by_fb,
                                    tickers, ticker_boundaries, input_dim, _state_cache,
                                    fixed_threshold=args.fixed_threshold,
                                    has_multi_horizon=has_multi_horizon)
    study.optimize(objective_fn, n_trials=num_trials, callbacks=[trial_callback],
                   catch=(Exception,))

    # --- RESULTS ---
    total_time = time.time() - t0
    total_trials = len(study.trials)
    print(f"\n{'='*70}")
    print(f"DONE: {num_trials} new {target} trials in {total_time/60:.1f}min ({total_trials} total in study)")
    print(f"{'='*70}")

    # Save model if we found a new best in THIS run (we have weights in memory)
    if best_state_holder['state'] is not None:
        best_cfg = best_state_holder['cfg']
        best_state = best_state_holder['state']
        pc = best_state_holder['per_class']

        print(f"\nBest {target} model (score={best_state_holder['score']:.3f}, "
              f"acc={best_state_holder['val_acc']:.3f}):")
        for k, v in best_cfg.items():
            print(f"  {k}: {v}")
        print(f"  {pc}")

        mdl = CryptoLSTM(input_dim, best_cfg['hidden_dim'],
                          best_cfg['num_layers'], best_cfg['dropout'], NUM_CLASSES)
        mdl.load_state_dict(best_state)
        torch.save(mdl.state_dict(), f'{prefix}{target}_model.pth')

        config = {
            'input_dim': input_dim,
            'hidden_dim': best_cfg['hidden_dim'],
            'num_layers': best_cfg['num_layers'],
            'dropout': best_cfg['dropout'],
            'seq_len': best_cfg['seq_len'],
            'num_classes': NUM_CLASSES,
            'mode': 'classification',
            'bull_threshold': best_cfg['bull_threshold'],
            'bear_threshold': -best_cfg['bull_threshold'],
            'forward_bars': best_cfg.get('forward_bars', 4),
            'target': target,
            'prefix': args.prefix,
            'shared_threshold': args.fixed_threshold is not None,
            'indicator_preset': preset_name,
        }
        joblib.dump(config, f'{prefix}{target}_config.pkl')
        joblib.dump(scaler_X, f'{prefix}scaler_X.pkl')
        joblib.dump(feature_cols, f'{prefix}feature_cols.pkl')
        joblib.dump(None, f'{prefix}scaler_y.pkl')
        print(f"\n{target.capitalize()} model saved to {prefix}{target}_model.pth / {prefix}{target}_config.pkl")
    else:
        print(f"\nNo new best found in this run (prior best {target}={best_state_holder['score']:.3f})")
        print(f"Existing {prefix}{target}_model.pth (if any) unchanged.")

    # Param importance (across ALL trials in study)
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
    with open(f'hypersearch_{prefix}{target}_log.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nTrials: {total_trials} total (study), {pruned} pruned")
    print(f"Log: hypersearch_{prefix}{target}_log.json")


if __name__ == '__main__':
    main()
