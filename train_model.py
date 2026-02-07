import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import CryptoLSTM
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import gc

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 40
LEARNING_RATE = 0.001
SEQ_LEN = 24  # 24 hours of history per sample
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
TRAIN_RATIO = 0.8  # chronological split

# Classification thresholds (in % return)
BULL_THRESHOLD = 0.15   # > +0.15% = bullish
BEAR_THRESHOLD = -0.15  # < -0.15% = bearish
# Between = neutral
NUM_CLASSES = 3  # 0=bearish, 1=neutral, 2=bullish

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
print("Loading data...")
df = pd.read_csv('training_data.csv', index_col=0, parse_dates=True)
print(f"Dataset: {len(df)} rows")

# Determine feature columns (exclude non-feature columns)
exclude_cols = ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']
feature_cols = [c for c in df.columns if c not in exclude_cols]
feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
target_col = 'Target_Return'

print(f"Features ({len(feature_cols)}): {feature_cols}")
joblib.dump(feature_cols, 'feature_cols.pkl')

# Create classification target
# 0 = bearish (return < -0.15%), 1 = neutral, 2 = bullish (return > +0.15%)
target_return = df[target_col].values
target_class = np.ones(len(target_return), dtype=np.int64)  # default: neutral
target_class[target_return > BULL_THRESHOLD] = 2  # bullish
target_class[target_return < BEAR_THRESHOLD] = 0  # bearish

class_counts = np.bincount(target_class, minlength=3)
print(f"Class distribution: bearish={class_counts[0]} ({class_counts[0]/len(target_class)*100:.1f}%), "
      f"neutral={class_counts[1]} ({class_counts[1]/len(target_class)*100:.1f}%), "
      f"bullish={class_counts[2]} ({class_counts[2]/len(target_class)*100:.1f}%)")

df['Target_Class'] = target_class

# --- Process each ticker separately to build sequences ---
tickers = df['Ticker'].unique()
print(f"Tickers: {list(tickers)}")

scaler_X = MinMaxScaler()
all_X = df[feature_cols].values
scaler_X.fit(all_X)
joblib.dump(scaler_X, 'scaler_X.pkl')

# Save a dummy scaler_y for backward compat (predict_now checks for it)
# In classification mode, we don't scale targets
joblib.dump(None, 'scaler_y.pkl')

all_sequences_X = []
all_sequences_y = []

for ticker in tickers:
    ticker_df = df[df['Ticker'] == ticker].sort_index()
    X_ticker = scaler_X.transform(ticker_df[feature_cols].values)
    y_ticker = ticker_df['Target_Class'].values

    for i in range(SEQ_LEN, len(X_ticker)):
        all_sequences_X.append(X_ticker[i - SEQ_LEN:i])
        all_sequences_y.append(y_ticker[i])

all_sequences_X = np.array(all_sequences_X, dtype=np.float32)
all_sequences_y = np.array(all_sequences_y, dtype=np.int64)

print(f"Total sequences: {len(all_sequences_X)}, shape: {all_sequences_X.shape}")

del df, all_X
gc.collect()

# --- Chronological time-series split ---
split_idx = int(len(all_sequences_X) * TRAIN_RATIO)
X_train, y_train = all_sequences_X[:split_idx], all_sequences_y[:split_idx]
X_val, y_val = all_sequences_X[split_idx:], all_sequences_y[split_idx:]
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Compute class weights for imbalanced classes (from training set only)
class_weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights}")

del all_sequences_X, all_sequences_y
gc.collect()

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
del X_train, y_train, X_val, y_val
gc.collect()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


input_dim = train_dataset[0][0].shape[1]
model = CryptoLSTM(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT, NUM_CLASSES).to(device)
print(f"\nModel Architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# --- 3. TRAINING LOOP ---
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

best_val_acc = 0.0
best_val_loss = float('inf')
counter = 0

print(f"\n--- STARTING TRAINING (batch_size={BATCH_SIZE}, {len(train_loader)} batches/epoch) ---")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss /= len(train_dataset)
    train_acc = correct / total
    scheduler.step()

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels = []
    with torch.inference_mode():
        for X_vb, y_vb in val_loader:
            X_vb, y_vb = X_vb.to(device), y_vb.to(device)
            val_out = model(X_vb)
            val_loss_sum += criterion(val_out, y_vb).item() * X_vb.size(0)
            _, pred = torch.max(val_out, 1)
            val_correct += (pred == y_vb).sum().item()
            val_total += y_vb.size(0)
            val_preds.extend(pred.cpu().numpy())
            val_labels.extend(y_vb.cpu().numpy())

    val_loss = val_loss_sum / val_total
    val_acc = val_correct / val_total

    if (epoch + 1) % 25 == 0:
        lr = optimizer.param_groups[0]['lr']
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels)
        # Per-class accuracy
        per_class = []
        for cls, name in [(0, 'bear'), (1, 'neut'), (2, 'bull')]:
            mask = val_labels_arr == cls
            if mask.sum() > 0:
                cls_acc = (val_preds_arr[mask] == cls).mean() * 100
                per_class.append(f"{name}:{cls_acc:.0f}%")
            else:
                per_class.append(f"{name}:N/A")
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}/{val_loss:.4f}, '
              f'Acc: {train_acc:.3f}/{val_acc:.3f}, LR: {lr:.6f}, [{", ".join(per_class)}]')

    # Early Stopping (on val accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'stock_predictor.pth')
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save model config
config = {
    'input_dim': input_dim,
    'hidden_dim': HIDDEN_DIM,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'seq_len': SEQ_LEN,
    'num_classes': NUM_CLASSES,
    'mode': 'classification',
    'bull_threshold': BULL_THRESHOLD,
    'bear_threshold': BEAR_THRESHOLD,
}
joblib.dump(config, 'model_config.pkl')

print(f"\nTraining Complete.")
print(f"Best Val Accuracy: {best_val_acc:.4f} (Loss: {best_val_loss:.4f})")
print("Saved: stock_predictor.pth, scaler_X.pkl, feature_cols.pkl, model_config.pkl")
