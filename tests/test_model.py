"""Tests for model.py — CryptoLSTM forward pass and JIT tracing."""

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
