"""Retrain the best model from an Optuna v2 study on the final (largest) fold.

Usage:
    python scripts/retrain_best.py --data stock_training_data.csv --prefix stock --preset stationary --max-rows 200000
    python scripts/retrain_best.py --data training_data.csv --prefix crypto --preset stationary --max-rows 350000
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import optuna

from model_v2 import RegressionLSTM
from gpu_lock import acquire_for_training
from scripts.hypersearch_v2 import (
    load_data, get_walk_forward_folds, compute_sharpe,
    FORWARD_BARS, NUM_FOLDS, MAX_EPOCHS,
)
from sklearn.preprocessing import RobustScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='Retrain best model from v2 study')
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--prefix', type=str, default='', help='File prefix (e.g. "stock")')
    parser.add_argument('--preset', type=str, default='stationary', help='Indicator preset')
    parser.add_argument('--max-rows', type=int, default=500_000, help='Max rows to load')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Max epochs for final training (default: 80, more than search)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stop patience (default: 15, more patient than search)')
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = f'{args.prefix}_' if args.prefix else ''
    db_path = f'{prefix}v2_study.db'
    study_name = f'{prefix}v2_search'

    # Load best trial from study
    print(f"Loading study from {db_path}...")
    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{db_path}')
    best = study.best_trial
    cfg = best.params
    print(f"Best trial {best.number}: score={best.value:.3f}")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Load data
    (all_features, all_returns_by_fb, tickers, ticker_boundaries,
     feature_cols, input_dim, preset_name,
     has_multi_horizon) = load_data(args.data, preset_override=args.preset,
                                     max_rows=args.max_rows)

    forward_bars = cfg['forward_bars']
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

    trial_returns = all_returns_by_fb[forward_bars]

    # Use the final (largest) walk-forward fold for training
    folds = get_walk_forward_folds(tickers, ticker_boundaries, seq_len)
    train_indices, val_indices = folds[-1]  # last fold = most training data

    # Filter NaN targets
    train_mask = ~np.isnan(trial_returns[train_indices])
    val_mask = ~np.isnan(trial_returns[val_indices])
    train_indices = train_indices[train_mask]
    val_indices = val_indices[val_mask]

    n_train = len(train_indices)
    n_val = len(val_indices)
    print(f"\nFinal fold: {n_train} train + {n_val} val samples")

    # Fit scaler on train only
    scaler = RobustScaler()
    scaler.fit(all_features[train_indices])
    all_scaled = scaler.transform(all_features).astype(np.float32)

    # Build sequences
    offsets = np.arange(-seq_len, 0)
    X_train = np.ascontiguousarray(all_scaled[train_indices[:, None] + offsets[None, :]])
    X_val = np.ascontiguousarray(all_scaled[val_indices[:, None] + offsets[None, :]])
    del all_scaled
    gc.collect()

    print(f"X_train: {X_train.shape} ({X_train.nbytes / 1e6:.0f} MB)")
    print(f"X_val: {X_val.shape} ({X_val.nbytes / 1e6:.0f} MB)")

    y_train = trial_returns[train_indices]
    y_val = trial_returns[val_indices]

    # Build model
    model = RegressionLSTM(input_dim, hidden_dim, num_layers, dropout, n_heads).to(device)
    criterion = nn.HuberLoss(delta=huber_delta, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == 'cosine':
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    elif scheduler_type == 'plateau':
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
    else:
        sched = None

    use_amp = device.type == 'cuda'
    grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    best_state = None
    counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"Device: {device}, AMP: {use_amp}")
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_train)
        train_loss_sum = 0.0
        for i in range(0, n_train, batch_size):
            bi = perm[i:i + batch_size]
            xb = torch.from_numpy(X_train[bi]).to(device)
            yb = torch.from_numpy(y_train[bi]).to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(xb)
                raw_loss = criterion(pred, yb)
                weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                loss = (raw_loss * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / n_train

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
        sharpe = compute_sharpe(val_preds_np, y_val, trade_threshold)

        if scheduler_type == 'plateau' and sched:
            sched.step(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_sharpe = sharpe
            counter = 0
            improved = " *"
        else:
            counter += 1

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:3d}/{args.epochs}: "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"sharpe={sharpe:.2f} lr={lr:.6f}{improved}")

        if counter >= args.patience:
            print(f"  Early stop at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed/60:.1f} min")

    # Final evaluation with best model
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
        final_sharpe = compute_sharpe(final_preds_np, y_val, trade_threshold)

        # Trade statistics
        signals = np.where(final_preds_np > trade_threshold, 1,
                  np.where(final_preds_np < -trade_threshold, -1, 0))
        n_buys = (signals == 1).sum()
        n_sells = (signals == -1).sum()
        n_flat = (signals == 0).sum()
        trade_returns = signals[signals != 0] * y_val[signals != 0]
        win_rate = (trade_returns > 0).mean() * 100 if len(trade_returns) > 0 else 0

        print(f"\nFinal model Sharpe: {final_sharpe:.3f}")
        print(f"Trade stats: {n_buys} buys, {n_sells} sells, {n_flat} flat")
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Mean trade return: {trade_returns.mean():.4f}%" if len(trade_returns) > 0 else "")

        # Save model
        mdl = RegressionLSTM(input_dim, hidden_dim, num_layers, dropout, n_heads)
        mdl.load_state_dict(best_state)
        torch.save(mdl.state_dict(), f'{prefix}model_v2.pth')

        config = {
            'model_version': 2,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'n_heads': n_heads,
            'dropout': dropout,
            'seq_len': seq_len,
            'trade_threshold': trade_threshold,
            'forward_bars': forward_bars,
            'huber_delta': huber_delta,
            'prefix': args.prefix,
            'indicator_preset': preset_name,
            'retrain_sharpe': final_sharpe,
        }
        joblib.dump(config, f'{prefix}config_v2.pkl')
        joblib.dump(scaler, f'{prefix}scaler_v2.pkl')
        joblib.dump(feature_cols, f'{prefix}feature_cols_v2.pkl')

        print(f"\nSaved: {prefix}model_v2.pth")
        print(f"Saved: {prefix}config_v2.pkl")
        print(f"Saved: {prefix}scaler_v2.pkl")
        print(f"Saved: {prefix}feature_cols_v2.pkl")
    else:
        print("ERROR: No best state saved during training!")


if __name__ == '__main__':
    args = parse_args()
    lock_label = f"retrain_{args.prefix or 'crypto'}"
    with acquire_for_training(lock_label):
        main()
