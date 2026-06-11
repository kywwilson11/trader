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


class TestMissingFeatureSafetyNet:
    """Models trained on columns live can't produce must degrade
    (neutral-fill few / fail closed many), never KeyError-brick."""

    def _run(self, monkeypatch, feature_cols, df_cols):
        import numpy as np
        import pandas as pd
        import torch
        import predict_now

        idx = pd.date_range('2026-06-01', periods=40, freq='h', tz='UTC')
        bars = pd.DataFrame({c: np.full(40, 100.0)
                             for c in ('Open', 'High', 'Low', 'Close',
                                       'Volume')}, index=idx)
        feat_df = bars.copy()
        for c in df_cols:
            feat_df[c] = 1.0

        monkeypatch.setattr(predict_now, 'fetch_bars_yfinance',
                            lambda s: bars)
        monkeypatch.setattr(predict_now, 'compute_features',
                            lambda df, btc_close=None: feat_df)
        predict_now._warned_missing.clear()

        class Scaler:
            def transform(self, x):
                return np.asarray(x, dtype=np.float32)

        class Model:
            def __call__(self, t):
                return torch.tensor([0.42])

        config = {'seq_len': 8, 'trade_threshold': 0.2, 'prefix': 'zz_nope'}
        return predict_now.get_live_prediction(
            'BTC-USD', Model(), Scaler(), config, feature_cols,
            asset_type='crypto')

    def test_small_missing_set_neutral_fills(self, monkeypatch):
        # Fake_A/Fake_B have no injection branch (stays offline) and are
        # absent from the live frame -> neutral-filled, prediction runs
        cols = ['Close', 'Volume', 'RSI', 'Fake_A', 'Fake_B']
        pred = self._run(monkeypatch, cols, df_cols=['RSI'])
        assert pred == pytest.approx(0.42)

    def test_large_missing_set_fails_closed(self, monkeypatch):
        cols = ['Close', 'RSI', 'Fake_A', 'Fake_B', 'Fake_C',
                'Fake_D', 'Fake_E', 'Fake_F']
        # 6 of 8 missing (> 25%) -> no prediction
        pred = self._run(monkeypatch, cols, df_cols=['RSI'])
        assert pred is None

    def test_nothing_missing_unchanged(self, monkeypatch):
        cols = ['Close', 'Volume', 'RSI']
        pred = self._run(monkeypatch, cols, df_cols=['RSI'])
        assert pred == pytest.approx(0.42)
