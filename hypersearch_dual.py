"""
Dual-mode Optuna hyperparameter search — specialized bear or bull models.

Usage:
    python hypersearch_dual.py --target bear   # Optimize bear class accuracy
    python hypersearch_dual.py --target bull   # Optimize bull class accuracy

Each mode runs 250 trials and saves to {target}_model.pth / {target}_config.pkl.
"""
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CryptoLSTM
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import gc
import json
import time
import optuna
from optuna.pruners import MedianPruner

NUM_TRIALS = 250
MAX_EPOCHS = 80
EARLY_STOP_PATIENCE = 15
TRAIN_RATIO = 0.8
NUM_CLASSES = 3
PRUNE_WARMUP_EPOCHS = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SequenceDataset(Dataset):
    """On-the-fly sequence generator — no pre-allocation of massive arrays."""
    def __init__(self, indices, all_scaled, all_classes, seq_len):
        self.indices = indices
        self.all_scaled = all_scaled
        self.all_classes = all_classes
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        end = self.indices[idx]
        start = end - self.seq_len
        x = torch.from_numpy(self.all_scaled[start:end])
        y = self.all_classes[end]
        return x, y


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-mode hyperparameter search')
    parser.add_argument('--target', required=True, choices=['bear', 'bull'],
                        help='Which class to optimize: bear or bull')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS,
                        help=f'Number of trials (default: {NUM_TRIALS})')
    return parser.parse_args()


def load_data():
    print("Loading data...")
    df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
    print(f"Dataset: {len(df)} rows")

    exclude_cols = ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"Features: {len(feature_cols)}")

    scaler_X = MinMaxScaler()
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

    return all_scaled, all_returns, tickers, ticker_boundaries, scaler_X, feature_cols, input_dim


def get_indices_and_classes(all_returns, tickers, ticker_boundaries, bull_thresh, seq_len):
    bear_thresh = -bull_thresh
    classes = np.ones(len(all_returns), dtype=np.int64)
    classes[all_returns > bull_thresh] = 2
    classes[all_returns < bear_thresh] = 0

    valid_indices = []
    for ticker in tickers:
        start, end = ticker_boundaries[ticker]
        for i in range(start + seq_len, end):
            valid_indices.append(i)

    return valid_indices, classes


def create_objective(target, all_scaled, all_returns, tickers, ticker_boundaries, input_dim):
    # target class index: bear=0, bull=2
    target_class = 0 if target == 'bear' else 2

    def objective(trial):
        torch.cuda.empty_cache()
        gc.collect()

        seq_len = trial.suggest_categorical('seq_len', [12, 18, 24])
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128])
        num_layers = trial.suggest_int('num_layers', 1, 2)
        dropout = trial.suggest_float('dropout', 0.05, 0.45, step=0.05)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [128, 256])
        bull_threshold = trial.suggest_float('bull_threshold', 0.05, 0.35, step=0.01)
        weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
        scheduler = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none'])

        cfg = {
            'seq_len': seq_len, 'hidden_dim': hidden_dim,
            'num_layers': num_layers, 'dropout': dropout,
            'learning_rate': learning_rate, 'batch_size': batch_size,
            'bull_threshold': bull_threshold, 'weight_decay': weight_decay,
            'scheduler': scheduler,
        }

        valid_indices, classes = get_indices_and_classes(
            all_returns, tickers, ticker_boundaries, bull_threshold, seq_len)

        split = int(len(valid_indices) * TRAIN_RATIO)
        train_idx = valid_indices[:split]
        val_idx = valid_indices[split:]

        train_classes = classes[train_idx]
        unique = np.unique(train_classes)
        if len(unique) < 3:
            return 0.0

        counts = np.bincount(train_classes, minlength=3).astype(np.float64)
        total = counts.sum()
        weights = total / (3.0 * counts)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

        train_ds = SequenceDataset(train_idx, all_scaled, classes, seq_len)
        val_ds = SequenceDataset(val_idx, all_scaled, classes, seq_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

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
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                loss = criterion(out, y_b)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler == 'cosine' and sched:
                sched.step()

            model.eval()
            vc, vt, vl = 0, 0, 0.0
            with torch.inference_mode():
                for X_vb, y_vb in val_loader:
                    X_vb, y_vb = X_vb.to(device), y_vb.to(device)
                    vo = model(X_vb)
                    vl += criterion(vo, y_vb).item() * X_vb.size(0)
                    _, p = torch.max(vo, 1)
                    vc += (p == y_vb).sum().item()
                    vt += y_vb.size(0)

            val_acc = vc / vt
            val_loss = vl / vt

            if scheduler == 'plateau' and sched:
                sched.step(val_loss)

            trial.report(val_acc, epoch)
            if epoch >= PRUNE_WARMUP_EPOCHS and trial.should_prune():
                del model, train_ds, val_ds, train_loader, val_loader, weights_t
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                counter = 0
            else:
                counter += 1
                if counter >= EARLY_STOP_PATIENCE:
                    break

        # Per-class accuracy
        per_class = {}
        if best_state:
            model.load_state_dict(best_state)
            model.eval()
            ap, al = [], []
            with torch.inference_mode():
                for X_vb, y_vb in val_loader:
                    X_vb = X_vb.to(device)
                    _, p = torch.max(model(X_vb), 1)
                    ap.extend(p.cpu().numpy())
                    al.extend(y_vb.numpy())
            ap, al = np.array(ap), np.array(al)
            for c, n in [(0, 'bear'), (1, 'neutral'), (2, 'bull')]:
                m = al == c
                per_class[n] = float((ap[m] == c).mean()) if m.sum() > 0 else 0.0

        # Score = target class accuracy
        target_name = 'bear' if target_class == 0 else 'bull'
        target_score = per_class.get(target_name, 0.0)
        trading_score = (per_class.get('bear', 0) + per_class.get('bull', 0)) / 2

        trial.set_user_attr('per_class', per_class)
        trial.set_user_attr('target_score', target_score)
        trial.set_user_attr('trading_score', trading_score)
        trial.set_user_attr('val_acc', best_val_acc)
        trial.set_user_attr('best_state', best_state)
        trial.set_user_attr('cfg', cfg)

        del model, train_ds, val_ds, train_loader, val_loader, weights_t
        gc.collect()
        torch.cuda.empty_cache()

        return target_score

    return objective


def main():
    args = parse_args()
    target = args.target
    num_trials = args.trials

    all_scaled, all_returns, tickers, ticker_boundaries, scaler_X, feature_cols, input_dim = load_data()

    # Callback state
    results_log = []
    t0 = time.time()
    best_score_so_far = 0.0
    trials_since_improvement = 0

    def trial_callback(study, trial):
        nonlocal best_score_so_far, trials_since_improvement

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
        elif score > best_score_so_far and val_acc > 0.34:
            best_score_so_far = score
            trials_since_improvement = 0
            tag = " ** BEST **"

        print(f"[{n:3d}] acc={val_acc:.3f} {target}={score:.3f} "
              f"B:{pc.get('bear',0):.0%} N:{pc.get('neutral',0):.0%} U:{pc.get('bull',0):.0%} "
              f"| s={cfg.get('seq_len','')} h={cfg.get('hidden_dim','')} "
              f"l={cfg.get('num_layers','')} d={cfg.get('dropout',''):.2f} "
              f"lr={cfg.get('learning_rate',''):.4f} th={cfg.get('bull_threshold',''):.2f}"
              f"{tag}")

        results_log.append({
            'i': n, 'cfg': cfg, 'val_acc': val_acc,
            'target_score': score, 'per_class': pc,
            'state': str(trial.state),
            'time': elapsed,
        })

        if n % 10 == 0:
            with open(f'hypersearch_{target}_log.json', 'w') as f:
                json.dump(results_log, f, indent=2, default=str)
            print(f"  --- {elapsed/60:.1f}min elapsed, best {target}={best_score_so_far:.3f}, "
                  f"{trials_since_improvement} trials since last improvement ---")

    # --- MAIN SEARCH ---
    print(f"\n{'='*70}")
    print(f"OPTUNA {target.upper()} MODEL SEARCH: {num_trials} trials (TPE + pruning)")
    print(f"Optimizing: {target} class accuracy")
    print(f"{'='*70}\n")

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=PRUNE_WARMUP_EPOCHS),
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )

    objective_fn = create_objective(target, all_scaled, all_returns, tickers, ticker_boundaries, input_dim)
    study.optimize(objective_fn, n_trials=num_trials, callbacks=[trial_callback])

    # --- RESULTS ---
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE: {num_trials} {target} trials in {total_time/60:.1f}min")
    print(f"{'='*70}")

    # Find best completed trial
    best_trial = None
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.user_attrs.get('val_acc', 0) <= 0.34:
            continue
        if best_trial is None or (t.value or 0) > (best_trial.value or 0):
            best_trial = t

    if best_trial is not None:
        best_cfg = best_trial.user_attrs['cfg']
        best_state = best_trial.user_attrs['best_state']
        pc = best_trial.user_attrs['per_class']

        print(f"\nBest {target} model ({target}={best_trial.value:.3f}, acc={best_trial.user_attrs['val_acc']:.3f}):")
        for k, v in best_cfg.items():
            print(f"  {k}: {v}")
        print(f"  {pc}")

        mdl = CryptoLSTM(input_dim, best_cfg['hidden_dim'],
                          best_cfg['num_layers'], best_cfg['dropout'], NUM_CLASSES)
        mdl.load_state_dict(best_state)
        torch.save(mdl.state_dict(), f'{target}_model.pth')

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
        }
        joblib.dump(config, f'{target}_config.pkl')
        joblib.dump(scaler_X, 'scaler_X.pkl')
        joblib.dump(feature_cols, 'feature_cols.pkl')
        joblib.dump(None, 'scaler_y.pkl')
        print(f"\n{target.capitalize()} model saved to {target}_model.pth / {target}_config.pkl")

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
    else:
        print(f"\nNo valid {target} model found!")

    # Save full log
    with open(f'hypersearch_{target}_log.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nTrials: {len(study.trials)} total, {pruned} pruned")
    print(f"Log: hypersearch_{target}_log.json")


if __name__ == '__main__':
    main()
