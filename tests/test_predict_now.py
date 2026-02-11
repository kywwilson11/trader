"""Tests for predict_now.py — path generation and v2 model support."""

import os
import tempfile

import joblib
import numpy as np
import pytest
import torch

from predict_now import _prefixed_paths, load_model_v2, load_v2_models
from model_v2 import RegressionLSTM


class TestPrefixedPaths:
    def test_no_prefix(self):
        paths = _prefixed_paths("")
        assert paths["bear_model"] == "bear_model.pth"
        assert paths["bear_config"] == "bear_config.pkl"
        assert paths["bull_model"] == "bull_model.pth"
        assert paths["scaler_X"] == "scaler_X.pkl"
        assert paths["feature_cols"] == "feature_cols.pkl"

    def test_stock_prefix(self):
        paths = _prefixed_paths("stock")
        assert paths["bear_model"] == "stock_bear_model.pth"
        assert paths["bear_config"] == "stock_bear_config.pkl"
        assert paths["bull_model"] == "stock_bull_model.pth"
        assert paths["scaler_X"] == "stock_scaler_X.pkl"

    def test_default_model_path_unchanged(self):
        paths = _prefixed_paths("stock")
        assert paths["default_model"] == "stock_predictor.pth"

    def test_returns_all_required_keys(self):
        paths = _prefixed_paths("test")
        required = {"bear_model", "bear_config", "bull_model", "bull_config",
                     "scaler_X", "feature_cols", "default_model", "default_config",
                     "v2_model", "v2_config", "v2_scaler", "v2_features"}
        assert required.issubset(paths.keys())

    def test_v2_paths_no_prefix(self):
        paths = _prefixed_paths("")
        assert paths["v2_model"] == "model_v2.pth"
        assert paths["v2_config"] == "config_v2.pkl"
        assert paths["v2_scaler"] == "scaler_v2.pkl"
        assert paths["v2_features"] == "feature_cols_v2.pkl"

    def test_v2_paths_with_prefix(self):
        paths = _prefixed_paths("stock")
        assert paths["v2_model"] == "stock_model_v2.pth"
        assert paths["v2_config"] == "stock_config_v2.pkl"


class TestV2ModelLoading:
    """Test v2 model load/save round-trip."""

    @pytest.fixture
    def v2_model_files(self, tmp_path):
        """Create a temporary v2 model on disk."""
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

    def test_load_v2_models_returns_model(self, v2_model_files, monkeypatch):
        tmp_path, config = v2_model_files
        monkeypatch.chdir(tmp_path)

        model, cfg, scaler_X, feature_cols = load_v2_models(inference_device='cpu')
        assert model is not None
        assert cfg['model_version'] == 2
        assert cfg['trade_threshold'] == 0.25
        assert len(feature_cols) == config['input_dim']

    def test_load_v2_not_found(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        model, cfg, scaler_X, feature_cols = load_v2_models(inference_device='cpu')
        assert model is None
        assert cfg is None

    def test_v2_inference_shape(self, v2_model_files, monkeypatch):
        tmp_path, config = v2_model_files
        monkeypatch.chdir(tmp_path)

        model, cfg, scaler_X, feature_cols = load_v2_models(inference_device='cpu')
        x = torch.randn(1, cfg['seq_len'], cfg['input_dim'])
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1,)


class TestModelVersionDetection:
    def test_v1_config_has_no_model_version(self):
        """v1 configs don't have model_version key."""
        v1_config = {'input_dim': 20, 'hidden_dim': 64, 'seq_len': 16,
                     'bull_threshold': 0.30}
        assert v1_config.get('model_version', 1) == 1

    def test_v2_config_has_model_version_2(self):
        v2_config = {'model_version': 2, 'input_dim': 20,
                     'trade_threshold': 0.25}
        assert v2_config.get('model_version', 1) == 2
