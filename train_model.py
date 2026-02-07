import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # To save the scaler for later use

# --- CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
print("Loading data...")
df = pd.read_csv('training_data.csv')

# Drop non-numeric columns (like 'Ticker' or Date strings)
# We only want the calculated indicators and price data
# Note: We assume the last column is 'Target_Return' from the previous step
feature_cols = [c for c in df.columns if c not in ['Target_Return', 'Ticker', 'Date', 'Datetime', 'NextClose']]
target_col = 'Target_Return'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# Normalize Data (Crucial for Neural Nets)
# We scale everything to be between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save the scalers! We need them later to translate "Real World" -> "Robot World"
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Split into Training (80%) and Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors and move to GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# --- 2. MODEL DEFINITION ---
class StockPredictor(nn.Module):
    def __init__(self, input_dim):
        super(StockPredictor, self).__init__()
        
        # A simple 3-layer network
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Input Layer
            nn.ReLU(),                  # Activation Function
            nn.Dropout(0.2),            # Prevent overfitting
            
            nn.Linear(128, 64),         # Hidden Layer
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)            # Output Layer (Single Value: Predicted Return)
        )
        
    def forward(self, x):
        return self.net(x)

# Initialize Model
input_dim = X_train.shape[1]
model = StockPredictor(input_dim).to(device)
print(f"Model Architecture: {model}")

# --- 3. TRAINING LOOP ---
criterion = nn.MSELoss()  # Mean Squared Error (Standard for Regression)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
patience = 20 # Early stopping if no improvement
best_loss = float('inf')
counter = 0

print("\n--- STARTING TRAINING ON JETSON GPU ---")
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
        
    # Early Stopping Check
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'stock_predictor.pth') # Save the best model
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\nTraining Complete.")
print("Model saved as 'stock_predictor.pth'")
print("Scalers saved as 'scaler_X.pkl' and 'scaler_y.pkl'")
