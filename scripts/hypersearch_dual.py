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

NUM_TRIALS = 300
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 15
PRUNE_WARMUP_EPOCHS = 20       # don't prune until model has had time to learn
PRUNE_STARTUP_TRIALS = 50      # match TPE's random exploration phase
TRAIN_RATIO = 0.8
NUM_CLASSES = 3

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
    return parser.parse_args()


def load_data(data_path='training_data.csv', preset_override=None):
    print("Loading data...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Dataset: {len(df)} rows")

    exclude_cols = ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # Filter features by preset
    from indicator_config import load_indicator_config, get_preset_features
    preset_name = preset_override or load_indicator_config()["preset"]
    preset_features = get_preset_features(preset_name)
    if preset_features is not None:
        feature_cols = [c for c in feature_cols if c in preset_features]
    print(f"Preset: {preset_name} ({len(feature_cols)} features)")

    scaler_X = RobustScaler()
    scaler_X.fit(df[feature_cols].values)

    tickers = df['Ticker'].unique()
    all_scaled_list = []
    all_returns_list = []
    ticker_boundaries = {}

    offset = 0
    for ticker in tickers:
        tdf = df[df['Ticker'] == ticker].sort_index()
        scaled = scaler_X.transform(tdf[feature_cols].values).astype(np.float32)
        returns = tdf['Target_Return'].values.astype(np.float32)
        all_scaled_list.append(scaled)
        all_returns_list.append(returns)
        ticker_boundaries[ticker] = (offset, offset + len(scaled))
        offset += len(scaled)

    all_scaled = np.vstack(all_scaled_list)
    all_returns = np.concatenate(all_returns_list)
    del all_scaled_list, all_returns_list, df
    gc.collect()

    print(f"Contiguous arrays: {all_scaled.shape}, {all_scaled.nbytes / 1e6:.1f} MB")
    input_dim = all_scaled.shape[1]

    return all_scaled, all_returns, tickers, ticker_boundaries, scaler_X, feature_cols, input_dim, preset_name


def get_indices_and_classes(all_returns, tickers, ticker_boundaries, bull_thresh, seq_len):
    bear_thresh = -bull_thresh
    classes = np.ones(len(all_returns), dtype=np.int64)
    classes[all_returns > bull_thresh] = 2
    classes[all_returns < bear_thresh] = 0

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

    return train_indices, val_indices, classes


def create_objective(target, all_scaled, all_returns, tickers, ticker_boundaries, input_dim, _state_cache, fixed_threshold=None):
    # target class index: bear=0, bull=2
    target_class = 0 if target == 'bear' else 2

    MAX_TRIAL_SECONDS = 600  # kill any trial running longer than 10 min

    def objective(trial):
        trial_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()

        seq_len = trial.suggest_categorical('seq_len', [12, 18, 24])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.05, 0.45, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        if fixed_threshold is not None:
            bull_threshold = fixed_threshold
        else:
            bull_threshold = trial.suggest_float('bull_threshold', 0.20, 0.50, step=0.01)
        weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none'])

        cfg = {
            'seq_len': seq_len, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'dropout': dropout,
            'learning_rate': learning_rate, 'batch_size': batch_size,
            'bull_threshold': bull_threshold, 'weight_decay': weight_decay,
            'scheduler': scheduler,
        }

        # Store config early so callback can log params even on rejected/failed trials
        trial.set_user_attr('cfg', cfg)

        try:
            return _train_and_evaluate(
                trial, trial_start, cfg, target_class,
                all_scaled, all_returns, tickers, ticker_boundaries,
                input_dim, _state_cache,
            )
        except RuntimeError as e:
            # CUDA OOM or other GPU errors — clean up and return 0
            print(f"  [ERROR] Trial {trial.number}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            return 0.0

    def _train_and_evaluate(trial, trial_start, cfg, target_class,
                            all_scaled, all_returns, tickers, ticker_boundaries,
                            input_dim, _state_cache):
        seq_len = cfg['seq_len']
        hidden_dim = cfg['hidden_dim']
        num_layers = cfg['num_layers']
        dropout = cfg['dropout']
        learning_rate = cfg['learning_rate']
        batch_size = cfg['batch_size']
        bull_threshold = cfg['bull_threshold']
        weight_decay = cfg['weight_decay']
        scheduler = cfg['scheduler']

        train_idx, val_idx, classes = get_indices_and_classes(
            all_returns, tickers, ticker_boundaries, bull_threshold, seq_len)

        train_classes = classes[train_idx]
        unique = np.unique(train_classes)
        if len(unique) < 3:
            return 0.0

        counts = np.bincount(train_classes, minlength=3).astype(np.float64)
        total = counts.sum()
        weights = total / (3.0 * counts)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

        # Pre-allocate full sequence tensors on GPU (avoids per-batch numpy→torch→GPU)
        X_train = torch.stack([torch.from_numpy(all_scaled[i - seq_len:i]) for i in train_idx]).to(device)
        y_train = torch.tensor(classes[train_idx], dtype=torch.long, device=device)
        X_val = torch.stack([torch.from_numpy(all_scaled[i - seq_len:i]) for i in val_idx]).to(device)
        y_val = torch.tensor(classes[val_idx], dtype=torch.long, device=device)
        n_train = X_train.size(0)
        n_val = X_val.size(0)

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

        for epoch in range(MAX_EPOCHS):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                out = model(X_train[idx])
                loss = criterion(out, y_train[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler == 'cosine' and sched:
                sched.step()

            model.eval()
            with torch.inference_mode():
                # Single forward pass over entire val set (batched to avoid OOM on large sets)
                all_preds_list = []
                val_loss_sum = 0.0
                for i in range(0, n_val, batch_size):
                    X_vb = X_val[i:i + batch_size]
                    y_vb = y_val[i:i + batch_size]
                    vo = model(X_vb)
                    val_loss_sum += criterion(vo, y_vb).item() * X_vb.size(0)
                    all_preds_list.append(vo.argmax(1))
                ep_preds_t = torch.cat(all_preds_list)

            val_acc = (ep_preds_t == y_val).float().mean().item()
            val_loss = val_loss_sum / n_val

            if scheduler == 'plateau' and sched:
                sched.step(val_loss)

            # Compute epoch composite_score for pruner (same metric as final objective)
            ep_preds = ep_preds_t.cpu().numpy()
            ep_labels = y_val.cpu().numpy()
            ep_tp = int(((ep_preds == target_class) & (ep_labels == target_class)).sum())
            ep_fp = int(((ep_preds == target_class) & (ep_labels != target_class)).sum())
            ep_fn = int(((ep_preds != target_class) & (ep_labels == target_class)).sum())
            ep_prec = ep_tp / (ep_tp + ep_fp) if (ep_tp + ep_fp) > 0 else 0.0
            ep_rec = ep_tp / (ep_tp + ep_fn) if (ep_tp + ep_fn) > 0 else 0.0
            ep_f1 = 2 * ep_prec * ep_rec / (ep_prec + ep_rec) if (ep_prec + ep_rec) > 0 else 0.0
            ep_bal = sum((ep_preds[ep_labels == c] == c).mean() if (ep_labels == c).sum() > 0 else 0.0
                         for c in range(NUM_CLASSES)) / NUM_CLASSES
            ep_cat = (int(((ep_labels == 0) & (ep_preds == 2)).sum()) +
                      int(((ep_labels == 2) & (ep_preds == 0)).sum())) / n_val
            epoch_score = max(ep_f1 * 0.5 + ep_bal * 0.2 - ep_cat * 0.3, 0.0)

            trial.report(epoch_score, epoch)
            if epoch >= PRUNE_WARMUP_EPOCHS and trial.should_prune():
                del model, X_train, y_train, X_val, y_val, weights_t
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            # Hard time limit per trial
            if time.time() - trial_start > MAX_TRIAL_SECONDS:
                print(f"  [TIMEOUT] Trial {trial.number} exceeded {MAX_TRIAL_SECONDS}s at epoch {epoch}, stopping")
                break

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_preds = ep_preds
                best_labels = ep_labels
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
            del model, X_train, y_train, X_val, y_val, weights_t
            gc.collect()
            torch.cuda.empty_cache()
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

        del model, X_train, y_train, X_val, y_val, weights_t
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

    all_scaled, all_returns, tickers, ticker_boundaries, scaler_X, feature_cols, input_dim, preset_name = load_data(args.data, preset_override=args.preset)

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
        f1 = trial.user_attrs.get('target_f1', 0.0)
        cat = trial.user_attrs.get('catastrophic_rate', 0.0)
        print(f"[{n:3d}] score={score:.3f} F1={f1:.2f} cat={cat:.2f} "
              f"B:{pc.get('bear',0):.0%} N:{pc.get('neutral',0):.0%} U:{pc.get('bull',0):.0%} "
              f"| s={cfg.get('seq_len','')} h={cfg.get('hidden_dim','')} "
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

    objective_fn = create_objective(target, all_scaled, all_returns, tickers, ticker_boundaries, input_dim, _state_cache, fixed_threshold=args.fixed_threshold)
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
