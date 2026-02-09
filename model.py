"""CryptoLSTM model — LSTM-based classifier with attention for market regime prediction.

Architecture: stacked LSTM -> attention pooling -> FC head (hidden -> 128 -> 64 -> num_classes).
Three output classes: bearish / neutral / bullish, used by the trading loops to decide
buy/sell signals via softmax probabilities.

The attention layer learns which timesteps in the sequence matter most, allowing the model
to focus on e.g. a significant RSI divergence 3 hours ago rather than hoping that information
survives through many LSTM timesteps.

Designed for sequence lengths of 24-72 bars of technical indicator features.
Supports JIT tracing for ~30% faster inference on the Jetson.
"""

import torch
import torch.nn as nn


class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, num_classes=3):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Attention: learn a score per timestep, then weighted-sum the LSTM outputs
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        return self.fc(context)
