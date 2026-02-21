"""Tests for predict_now.py — path generation and model loading."""

import os
import tempfile

import joblib
import numpy as np
import pytest
import torch

from predict_now import _prefixed_paths, load_model, load_models
from model_v2 import RegressionLSTM


class TestPrefixedPaths:
    def test_no_prefix(self):
        paths = _prefixed_paths("")
        assert paths["model"] == "model_v2.pth"
        assert paths["config"] == "config_v2.pkl"
        assert paths["scaler"] == "scaler_v2.pkl"
        assert paths["features"] == "feature_cols_v2.pkl"

    def test_stock_prefix(self):
        paths = _prefixed_paths("stock")
        assert paths["model"] == "stock_model_v2.pth"
        assert paths["config"] == "stock_config_v2.pkl"
        assert paths["scaler"] == "stock_scaler_v2.pkl"
        assert paths["features"] == "stock_feature_cols_v2.pkl"

    def test_returns_all_required_keys(self):
        paths = _prefixed_paths("test")
        required = {"model", "config", "scaler", "features"}
        assert required == set(paths.keys())


class TestModelLoading:
    """Test model load/save round-trip."""

    @pytest.fixture
    def model_files(self, tmp_path):
        """Create a temporary model on disk."""
        input_dim = 15
        hidden_dim = 32
        model = RegressionLSTM(input_dim, hidden_dim, num_layers=1, dropout=0.1, n_heads=2)

        config = {
            'model_version': 2,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': 1,
            'n_heads': 2,
            'dropout': 0.1,
            'seq_len': 12,
            'trade_threshold': 0.25,
            'forward_bars': 8,
        }

        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(np.random.randn(100, input_dim))

        feature_cols = [f'feat_{i}' for i in range(input_dim)]

        torch.save(model.state_dict(), tmp_path / 'model_v2.pth')
        joblib.dump(config, tmp_path / 'config_v2.pkl')
        joblib.dump(scaler, tmp_path / 'scaler_v2.pkl')
        joblib.dump(feature_cols, tmp_path / 'feature_cols_v2.pkl')

        return tmp_path, config

    def test_load_models_returns_model(self, model_files, monkeypatch):
        tmp_path, config = model_files
        monkeypatch.chdir(tmp_path)

        model, cfg, scaler_X, feature_cols = load_models(inference_device='cpu')
        assert model is not None
        assert cfg['trade_threshold'] == 0.25
        assert len(feature_cols) == config['input_dim']

    def test_load_models_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            load_models(inference_device='cpu')

    def test_inference_shape(self, model_files, monkeypatch):
        tmp_path, config = model_files
        monkeypatch.chdir(tmp_path)

        model, cfg, scaler_X, feature_cols = load_models(inference_device='cpu')
        x = torch.randn(1, cfg['seq_len'], cfg['input_dim'])
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1,)

    def test_load_model_returns_five_tuple(self, model_files, monkeypatch):
        tmp_path, config = model_files
        monkeypatch.chdir(tmp_path)

        result = load_model(inference_device='cpu')
        assert len(result) == 5
        model, scaler_X, cfg, seq_len, feature_cols = result
        assert seq_len == config['seq_len']

    def test_with_prefix(self, tmp_path, monkeypatch):
        """Test loading with a prefix (e.g. 'stock')."""
        input_dim = 10
        model = RegressionLSTM(input_dim, 32, num_layers=1, dropout=0.1, n_heads=2)
        config = {
            'input_dim': input_dim,
            'hidden_dim': 32,
            'num_layers': 1,
            'n_heads': 2,
            'dropout': 0.1,
            'seq_len': 8,
            'trade_threshold': 0.30,
            'forward_bars': 4,
        }
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(np.random.randn(50, input_dim))
        feature_cols = [f'f_{i}' for i in range(input_dim)]

        torch.save(model.state_dict(), tmp_path / 'stock_model_v2.pth')
        joblib.dump(config, tmp_path / 'stock_config_v2.pkl')
        joblib.dump(scaler, tmp_path / 'stock_scaler_v2.pkl')
        joblib.dump(feature_cols, tmp_path / 'stock_feature_cols_v2.pkl')

        monkeypatch.chdir(tmp_path)
        model, cfg, scaler_X, fc = load_models(inference_device='cpu', prefix='stock')
        assert model is not None
        assert cfg['trade_threshold'] == 0.30
        assert len(fc) == input_dim
