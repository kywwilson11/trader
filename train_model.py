import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- CONFIGURATION ---
BATCH_SIZE = 512
EPOCHS = 1000
PATIENCE = 20
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
print("Loading data...")
df = pd.read_csv('training_data.csv')
print(f"Dataset: {len(df)} rows")

feature_cols = [c for c in df.columns if c not in ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']]
target_col = 'Target_Return'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# Normalize Data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Split into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Use DataLoaders for mini-batch training (fits in Orin Nano's VRAM)
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# --- 2. MODEL DEFINITION ---
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Initialize Model
input_dim = X_train.shape[1]
model = StockPredictor(input_dim).to(device)
print(f"Model Architecture: {model}")

# --- 3. TRAINING LOOP ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')
counter = 0

print(f"\n--- STARTING TRAINING (batch_size={BATCH_SIZE}, {len(train_loader)} batches/epoch) ---")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)

    epoch_loss /= len(train_dataset)

    # Validation
    model.eval()
    with torch.inference_mode():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss.item():.6f}')

    # Early Stopping
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'stock_predictor.pth')
        counter = 0
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\nTraining Complete.")
print("Model saved as 'stock_predictor.pth'")
print("Scalers saved as 'scaler_X.pkl' and 'scaler_y.pkl'")
