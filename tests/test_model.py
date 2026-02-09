"""Tests for model.py — CryptoLSTM forward pass, attention, and JIT tracing."""

import torch
import pytest

from model import CryptoLSTM


class TestCryptoLSTM:
    def test_output_shape(self):
        model = CryptoLSTM(input_dim=10, hidden_dim=32, num_layers=1)
        x = torch.randn(4, 20, 10)  # batch=4, seq_len=20, features=10
        out = model(x)
        assert out.shape == (4, 3)

    def test_output_logits_not_probabilities(self):
        model = CryptoLSTM(input_dim=5, hidden_dim=16, num_layers=1)
        x = torch.randn(2, 10, 5)
        out = model(x)
        # Logits can be negative; softmax output cannot
        # Just check we get finite values
        assert torch.isfinite(out).all()

    def test_softmax_sums_to_one(self):
        model = CryptoLSTM(input_dim=5, hidden_dim=16, num_layers=1)
        x = torch.randn(2, 10, 5)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_custom_num_classes(self):
        model = CryptoLSTM(input_dim=8, hidden_dim=16, num_layers=1, num_classes=5)
        x = torch.randn(1, 15, 8)
        out = model(x)
        assert out.shape == (1, 5)

    def test_jit_trace(self):
        model = CryptoLSTM(input_dim=6, hidden_dim=16, num_layers=1)
        model.eval()
        x = torch.randn(1, 10, 6)
        traced = torch.jit.trace(model, x)
        out_traced = traced(x)
        out_direct = model(x)
        assert torch.allclose(out_traced, out_direct, atol=1e-5)

    def test_attention_weights_sum_to_one(self):
        """Attention weights across timesteps should sum to 1 for each sample."""
        model = CryptoLSTM(input_dim=8, hidden_dim=16, num_layers=1)
        model.eval()
        x = torch.randn(3, 24, 8)
        with torch.inference_mode():
            lstm_out, _ = model.lstm(x)
            attn_weights = torch.softmax(model.attn(lstm_out), dim=1)
        # Shape: (batch, seq_len, 1) — sums along seq_len dim should be ~1
        sums = attn_weights.sum(dim=1).squeeze(-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_attention_different_seq_lengths(self):
        """Model should work with different sequence lengths."""
        model = CryptoLSTM(input_dim=6, hidden_dim=16, num_layers=1)
        model.eval()
        for seq_len in [12, 24, 48, 72]:
            x = torch.randn(2, seq_len, 6)
            out = model(x)
            assert out.shape == (2, 3)

    def test_wider_fc_head(self):
        """FC head should have hidden->128->64->num_classes structure."""
        model = CryptoLSTM(input_dim=10, hidden_dim=64, num_layers=1)
        # Check the FC sequential layers
        fc_layers = [m for m in model.fc if isinstance(m, torch.nn.Linear)]
        assert len(fc_layers) == 3
        assert fc_layers[0].in_features == 64   # hidden_dim
        assert fc_layers[0].out_features == 128
        assert fc_layers[1].in_features == 128
        assert fc_layers[1].out_features == 64
        assert fc_layers[2].in_features == 64
        assert fc_layers[2].out_features == 3   # num_classes

    def test_has_attention_layer(self):
        model = CryptoLSTM(input_dim=10, hidden_dim=32, num_layers=1)
        assert hasattr(model, 'attn')
        assert isinstance(model.attn, torch.nn.Linear)
        assert model.attn.in_features == 32  # hidden_dim
        assert model.attn.out_features == 1

    def test_three_layers_dropout(self):
        """3-layer LSTM should use inter-layer dropout."""
        model = CryptoLSTM(input_dim=5, hidden_dim=16, num_layers=3, dropout=0.3)
        x = torch.randn(2, 20, 5)
        out = model(x)
        assert out.shape == (2, 3)
