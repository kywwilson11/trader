"""Tests for model_v2.py — RegressionLSTM architecture."""

import pytest
import torch

from model_v2 import RegressionLSTM


class TestOutputShape:
    def test_basic_shape(self):
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=1, n_heads=2)
        x = torch.randn(8, 16, 20)
        out = model(x)
        assert out.shape == (8,)

    def test_various_input_dims(self):
        for input_dim in [10, 25, 40]:
            model = RegressionLSTM(input_dim=input_dim, hidden_dim=64, num_layers=1, n_heads=2)
            x = torch.randn(4, 12, input_dim)
            out = model(x)
            assert out.shape == (4,)

    def test_single_sample(self):
        model = RegressionLSTM(input_dim=15, hidden_dim=32, num_layers=1, n_heads=2)
        x = torch.randn(1, 24, 15)
        out = model(x)
        assert out.shape == (1,)

    def test_larger_batch(self):
        model = RegressionLSTM(input_dim=20, hidden_dim=128, num_layers=2, n_heads=4)
        x = torch.randn(32, 18, 20)
        out = model(x)
        assert out.shape == (32,)


class TestJITTrace:
    def test_trace_compatible(self):
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=1, n_heads=2)
        model.eval()
        dummy = torch.randn(1, 16, 20)
        # check_trace=False needed: MHA internal graph varies across invocations
        traced = torch.jit.trace(model, dummy, check_trace=False)
        out = traced(dummy)
        assert out.shape == (1,)

    def test_traced_matches_eager(self):
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=1, n_heads=2)
        model.eval()
        x = torch.randn(4, 16, 20)
        eager_out = model(x)
        traced = torch.jit.trace(model, torch.randn(1, 16, 20), check_trace=False)
        traced_out = traced(x)
        assert torch.allclose(eager_out, traced_out, atol=1e-5)


class TestGradientFlow:
    def test_backward_produces_grads(self):
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=2, n_heads=2)
        x = torch.randn(8, 16, 20)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestMultiHeadAttention:
    @pytest.mark.parametrize("n_heads", [1, 2, 4])
    def test_different_head_counts(self, n_heads):
        # hidden_dim must be divisible by n_heads
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=1, n_heads=n_heads)
        x = torch.randn(4, 16, 20)
        out = model(x)
        assert out.shape == (4,)

    def test_output_is_continuous(self):
        """Output should be unbounded continuous values, not class logits."""
        model = RegressionLSTM(input_dim=20, hidden_dim=64, num_layers=1, n_heads=2)
        x = torch.randn(100, 16, 20)
        out = model(x)
        # Should have a range of values, not just a few discrete ones
        assert out.unique().numel() == 100
