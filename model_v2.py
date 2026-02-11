"""RegressionLSTM — LSTM + Multi-Head Attention for continuous return prediction.

Architecture: stacked LSTM -> MultiheadAttention (with residual + LayerNorm) -> FC head -> 1 output.
Predicts continuous return percentages instead of discrete bear/neutral/bull classes.

Designed for walk-forward training with Huber loss and Sharpe ratio optimization.
Supports JIT tracing for faster inference on the Jetson.
"""

import torch
import torch.nn as nn


class RegressionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 dropout=0.3, n_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # self-attention
        combined = self.norm(attn_out + lstm_out)  # residual + LayerNorm
        # Pool over sequence: mean of all timesteps
        pooled = combined.mean(dim=1)  # (batch, hidden_dim)
        return self.fc(pooled).squeeze(-1)  # (batch,)
