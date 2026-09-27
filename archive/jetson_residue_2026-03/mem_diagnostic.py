"""Test OOM resilience with memory fraction cap on Jetson.

Tests the exact same code path as hypersearch_v2 with the new
set_per_process_memory_fraction(0.40) cap and OOM retry logic.
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import RobustScaler

from model_v2 import RegressionLSTM

# Apply the same CUDA cap as hypersearch_v2
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.40)
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    free, total = torch.cuda.mem_get_info()
    cap = int(0.40 * total / 1e6)
    print(f"CUDA: {free/1e6:.0f}MB free / {total/1e6:.0f}MB total, cap={cap}MB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_sys_avail():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


def test_config(all_features, all_returns, tickers, ticker_boundaries,
                seq_len, hidden_dim, num_layers, n_heads, batch_size, input_dim):
    """Test one config with OOM retry logic (mirrors hypersearch_v2 flow)."""
    print(f"\n{'='*75}")
    print(f"s={seq_len}  h={hidden_dim}  l={num_layers}  nh={n_heads}  bs={batch_size}"
          f"  | sys_avail={get_sys_avail():.0f}MB")

    gc.collect()
    torch.cuda.empty_cache()

    from scripts.hypersearch_v2 import get_walk_forward_folds
    folds = get_walk_forward_folds(tickers, ticker_boundaries, seq_len)
    fold_idx = 0
    train_indices, val_indices = folds[fold_idx]

    train_mask = ~np.isnan(all_returns[train_indices])
    val_mask = ~np.isnan(all_returns[val_indices])
    train_indices = train_indices[train_mask]
    val_indices = val_indices[val_mask]

    # Build cache
    scaler = RobustScaler()
    scaler.fit(all_features[train_indices])
    all_scaled = scaler.transform(all_features).astype(np.float32)
    offsets = np.arange(-seq_len, 0)
    X_train = np.ascontiguousarray(all_scaled[train_indices[:, None] + offsets[None, :]])
    X_val = np.ascontiguousarray(all_scaled[val_indices[:, None] + offsets[None, :]])
    del all_scaled
    gc.collect()

    y_train = all_returns[train_indices]
    y_val = all_returns[val_indices]
    n_train = len(train_indices)

    print(f"  cache: train={X_train.nbytes/1e6:.0f}MB  val={X_val.nbytes/1e6:.0f}MB"
          f"  | sys_avail={get_sys_avail():.0f}MB")

    # OOM retry loop — same as hypersearch_v2
    eff_batch_size = batch_size
    oom_retries = 0
    while True:
        try:
            model = RegressionLSTM(input_dim, hidden_dim, num_layers, 0.2, n_heads).to(device)
            criterion = nn.HuberLoss(delta=1.5, reduction='none')
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            grad_scaler = torch.amp.GradScaler('cuda', enabled=True)

            # Test batch
            model.train()
            test_bi = np.arange(min(eff_batch_size, n_train))
            xb = torch.from_numpy(X_train[test_bi]).to(device)
            yb = torch.from_numpy(y_train[test_bi]).to(device)
            with torch.amp.autocast('cuda', enabled=True):
                pred = model(xb)
                raw_loss = criterion(pred, yb)
                weights = torch.clamp(torch.abs(yb) + 1.0, max=50.0)
                loss = (raw_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            del xb, yb, pred, raw_loss, weights, loss
            break  # fits!
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err = str(e)[:80]
            try:
                del model, criterion, optimizer
            except NameError:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            oom_retries += 1
            eff_batch_size //= 2
            if eff_batch_size < 128:
                print(f"  FAIL: can't fit even bs=128 ({err})")
                del X_train, X_val, y_train, y_val
                gc.collect()
                return False
            print(f"  [OOM-RETRY] bs {eff_batch_size*2}→{eff_batch_size} ({err})")

    # Run 3 real epochs to verify stability
    for epoch in range(3):
        model.train()
        perm = np.random.permutation(n_train)
        for i in range(0, n_train, eff_batch_size):
            bi = perm[i:i + eff_batch_size]
            xb = torch.from_numpy(X_train[bi]).to(device)
            yb = torch.from_numpy(y_train[bi]).to(device)
            with torch.amp.autocast('cuda', enabled=True):
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

    avail = get_sys_avail()
    print(f"  OK: 3 epochs with bs={eff_batch_size}"
          f" (retries={oom_retries}) | sys_avail={avail:.0f}MB")

    # Cleanup
    del model, optimizer, grad_scaler, criterion
    del X_train, X_val, y_train, y_val
    gc.collect()
    torch.cuda.empty_cache()
    return True


def main():
    from scripts.hypersearch_v2 import load_data

    (all_features, all_returns_by_fb, tickers, ticker_boundaries,
     feature_cols, input_dim, preset_name, has_multi_horizon) = load_data(
        data_path='stock_training_data.csv',
        preset_override='stationary',
        max_rows=200_000,
    )
    all_returns = all_returns_by_fb.get(96, list(all_returns_by_fb.values())[0])
    print(f"\nData loaded: {all_features.shape}, sys_avail={get_sys_avail():.0f}MB")

    configs = [
        # (seq_len, hidden_dim, num_layers, n_heads, batch_size)
        (8,  64,  1, 2, 2048),
        (12, 96,  2, 2, 2048),
        (18, 128, 2, 4, 2048),
        (18, 192, 2, 2, 2048),
        (24, 128, 2, 4, 2048),
        (24, 192, 3, 4, 2048),
        (32, 64,  2, 2, 2048),
        (32, 128, 2, 4, 2048),
        (32, 192, 2, 2, 2048),
        (32, 256, 3, 8, 2048),
    ]

    results = []
    for cfg in configs:
        try:
            ok = test_config(all_features, all_returns, tickers, ticker_boundaries,
                           *cfg, input_dim)
            results.append((cfg, 'OK' if ok else 'FAIL'))
        except Exception as e:
            print(f"  UNEXPECTED: {e}")
            results.append((cfg, f'UNEXPECTED: {e}'))
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    for cfg, status in results:
        s, h, l, nh, bs = cfg
        print(f"  s={s:2d} h={h:3d} l={l} nh={nh} bs={bs}  →  {status}")


if __name__ == '__main__':
    main()
