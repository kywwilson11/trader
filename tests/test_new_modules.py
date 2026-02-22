"""Tests for new modules: volatility, macro_indicators, portfolio, regime_detector,
model_lgb, types_mod, log_config, and indicator enhancements.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


# --- types_mod ---

class TestTypes:
    def test_position_dataclass(self):
        from types_mod import Position
        pos = Position(qty=1.5, entry_price=100.0, high_water_mark=105.0)
        assert pos.qty == 1.5
        assert pos.stop_order_id is None
        assert pos.garch_sigma is None

    def test_position_to_dict(self):
        from types_mod import Position
        pos = Position(qty=1.0, entry_price=50.0, high_water_mark=52.0,
                       trailing_activated=True)
        d = pos.to_dict()
        assert d['qty'] == 1.0
        assert d['trailing_activated'] is True

    def test_macro_regime(self):
        from types_mod import MacroRegime
        r = MacroRegime(stress_level=0.5, vix=20.0, cape=30.0,
                        regime_label='caution', sizing_mult=0.8)
        assert r.is_defensive
        assert not r.should_halt_stocks

    def test_macro_regime_halt(self):
        from types_mod import MacroRegime
        r = MacroRegime(stress_level=2.0, vix=40.0, cape=None,
                        regime_label='crisis', sizing_mult=0.3)
        assert r.should_halt_stocks

    def test_quote_dataclass(self):
        from types_mod import Quote
        q = Quote(bid=99.0, ask=101.0, spread=2.0, midpoint=100.0, spread_pct=2.0)
        d = q.to_dict()
        assert d['midpoint'] == 100.0


# --- log_config ---

class TestLogConfig:
    def test_get_logger(self):
        from log_config import get_logger
        logger = get_logger('test_module')
        assert logger.name == 'test_module'

    def test_logger_levels(self):
        from log_config import get_logger
        import logging
        logger = get_logger('test_levels')
        assert logger.getEffectiveLevel() <= logging.DEBUG


# --- volatility ---

class TestVolatility:
    def test_fit_garch(self):
        from volatility import fit_garch
        np.random.seed(42)
        returns = np.random.standard_t(5, size=500) * 1.5
        result = fit_garch(returns)
        assert result is not None

    def test_fit_garch_insufficient_data(self):
        from volatility import fit_garch
        returns = np.random.randn(50)
        result = fit_garch(returns)
        assert result is None

    def test_forecast_volatility(self):
        from volatility import fit_garch, forecast_volatility
        np.random.seed(42)
        returns = np.random.standard_t(5, size=500) * 1.5
        result = fit_garch(returns)
        sigma = forecast_volatility(result)
        assert sigma is not None
        assert 0 < sigma < 1.0

    def test_forecast_volatility_none(self):
        from volatility import forecast_volatility
        assert forecast_volatility(None) is None

    def test_get_garch_stop(self):
        from volatility import get_garch_stop
        stop = get_garch_stop(100.0, 0.05, multiplier=2.0)
        assert stop < 100.0
        assert stop > 80.0  # not more than 20% away

    def test_get_garch_stop_floor(self):
        from volatility import get_garch_stop
        # Very low sigma should be floored
        stop = get_garch_stop(100.0, 0.001, multiplier=2.0, floor_pct=0.03)
        assert stop == pytest.approx(97.0, abs=0.01)

    def test_compute_vol_adjusted_size(self):
        from volatility import compute_vol_adjusted_size
        # Low vol → bigger position
        assert compute_vol_adjusted_size(1000, 0.01, target_vol=0.02) == 2000
        # High vol → smaller position
        assert compute_vol_adjusted_size(1000, 0.04, target_vol=0.02) == 500

    def test_vol_adjusted_size_clamp(self):
        from volatility import compute_vol_adjusted_size
        # Very high vol → clamped at 0.5x
        assert compute_vol_adjusted_size(1000, 0.10, target_vol=0.02) == 500
        # Very low vol → clamped at 2.0x
        assert compute_vol_adjusted_size(1000, 0.005, target_vol=0.02) == 2000


# --- portfolio ---

class TestPortfolio:
    def test_correlation_matrix(self):
        from portfolio import compute_correlation_matrix
        np.random.seed(42)
        base = np.random.randn(100)
        returns_dict = {
            'A': base,
            'B': base + np.random.randn(100) * 0.5,
            'C': np.random.randn(100),
        }
        corr = compute_correlation_matrix(returns_dict, window=50)
        # A and B should be highly correlated
        assert abs(corr[('A', 'B')]) > 0.5
        # A and C should be less correlated
        assert abs(corr.get(('A', 'C'), 0)) < abs(corr[('A', 'B')])

    def test_check_portfolio_correlation_allowed(self):
        from portfolio import check_portfolio_correlation
        corr = {('A', 'C'): 0.3, ('B', 'C'): 0.2}
        allowed, avg = check_portfolio_correlation(['A', 'B'], 'C', corr)
        assert allowed
        assert avg < 0.7

    def test_check_portfolio_correlation_rejected(self):
        from portfolio import check_portfolio_correlation
        corr = {('A', 'C'): 0.9, ('B', 'C'): 0.85}
        allowed, avg = check_portfolio_correlation(['A', 'B'], 'C', corr)
        assert not allowed

    def test_empty_portfolio(self):
        from portfolio import check_portfolio_correlation
        allowed, avg = check_portfolio_correlation([], 'A', {})
        assert allowed
        assert avg == 0.0

    def test_correlation_sizing_factor(self):
        from portfolio import get_correlation_sizing_factor
        corr = {('A', 'B'): 0.5}
        factor = get_correlation_sizing_factor('B', ['A'], corr)
        assert 0.5 <= factor <= 1.0


# --- regime_detector ---

class TestRegimeDetector:
    def test_fit_hmm(self):
        from regime_detector import fit_hmm
        np.random.seed(42)
        returns = np.concatenate([
            np.random.randn(100) + 1,   # bull
            np.random.randn(100) - 1,   # bear
            np.random.randn(100) * 0.5, # neutral
        ])
        model, labels = fit_hmm(returns, n_states=3)
        assert model is not None
        assert len(labels) == 3

    def test_fit_hmm_insufficient(self):
        from regime_detector import fit_hmm
        model, labels = fit_hmm(np.random.randn(50))
        assert model is None

    def test_get_current_regime(self):
        from regime_detector import fit_hmm, get_current_regime
        np.random.seed(42)
        returns = np.concatenate([
            np.random.randn(100) + 1,
            np.random.randn(100) - 1,
            np.random.randn(100) * 0.5,
        ])
        model, labels = fit_hmm(returns)
        regime = get_current_regime(model, labels, returns[-50:])
        assert regime['label'] in ('bull', 'bear', 'neutral', 'high_vol', 'unknown')
        assert 0 < regime['sizing_mult'] <= 1.5

    def test_default_regime(self):
        from regime_detector import _default_regime
        r = _default_regime()
        assert r['sizing_mult'] == 1.0


# --- model_lgb ---

class TestModelLgb:
    def test_flatten_sequence(self):
        from model_lgb import flatten_sequence
        seq = np.random.randn(10, 3)
        flat, names = flatten_sequence(seq, ['A', 'B', 'C'])
        assert flat.shape == (30,)
        assert len(names) == 30
        assert names[-1] == 'C'
        assert names[-4] == 'C_lag1'

    def test_ensemble_predict(self):
        from model_lgb import ensemble_predict
        combined = ensemble_predict(0.5, 0.3, lstm_weight=0.6)
        assert combined == pytest.approx(0.42)

    def test_ensemble_predict_no_lgb(self):
        from model_lgb import ensemble_predict
        combined = ensemble_predict(0.5, None)
        assert combined == 0.5


# --- indicators (Hurst) ---

class TestHurstExponent:
    def test_hurst_trending(self):
        from indicators import compute_hurst
        # Create a strongly trending series
        trend = pd.Series(np.arange(200, dtype=float))
        h = compute_hurst(trend, window=50)
        # Trending → H > 0.5
        assert h.iloc[-1] > 0.5

    def test_hurst_random_walk(self):
        from indicators import compute_hurst
        np.random.seed(42)
        rw = pd.Series(np.random.randn(200).cumsum())
        h = compute_hurst(rw, window=50)
        # Random walk → H ≈ 0.5
        assert 0.3 < h.iloc[-1] < 0.8

    def test_hurst_length(self):
        from indicators import compute_hurst
        prices = pd.Series(np.random.randn(200).cumsum())
        h = compute_hurst(prices, window=50)
        assert len(h) == len(prices)
        # First 49 should be NaN
        assert np.isnan(h.iloc[0])
        assert not np.isnan(h.iloc[-1])


# --- trading_utils (Kelly) ---

class TestKelly:
    def test_compute_kelly_no_history(self):
        from trading_utils import compute_kelly_fraction
        # No trade memory file → None
        result = compute_kelly_fraction(min_trades=50)
        # Could be None or a value depending on whether trade_memory.json exists
        # Just check it doesn't crash
        assert result is None or 0 < result <= 0.25

    def test_kelly_position_size_default(self):
        from trading_utils import kelly_position_size
        # Should return at least the base
        size = kelly_position_size(1000, 100000, min_trades=999999)
        assert size == 1000  # no history → use base

    def test_shared_constants(self):
        from trading_utils import LLM_VETO_THRESHOLD, THERMAL_THROTTLE_TEMP, ORDER_TIMEOUT
        assert LLM_VETO_THRESHOLD == 0.15
        assert THERMAL_THROTTLE_TEMP == 75
        assert ORDER_TIMEOUT == 30


# --- hypersearch_v2 (compute_sharpe with txn costs) ---

class TestSharpeWithTxnCosts:
    def test_txn_costs_reduce_sharpe(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
        from hypersearch_v2 import compute_sharpe
        np.random.seed(42)
        preds = np.random.randn(500) * 0.5
        rets = np.random.randn(500) * 1.0

        sharpe_no_cost = compute_sharpe(preds, rets, 0.3, txn_cost_bps=0)
        sharpe_with_cost = compute_sharpe(preds, rets, 0.3, txn_cost_bps=5)
        # Transaction costs should reduce Sharpe
        assert sharpe_with_cost < sharpe_no_cost

    def test_regime_sharpes(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
        from hypersearch_v2 import compute_regime_sharpes
        np.random.seed(42)
        preds = np.random.randn(500) * 0.5
        rets = np.random.randn(500) * 1.0
        result = compute_regime_sharpes(preds, rets, 0.3)
        assert 'bull' in result
        assert 'bear' in result
        assert 'sideways' in result
        assert 'min' in result


# --- order_utils (circuit breaker error handling) ---

class TestCircuitBreakerErrorHandling:
    def test_circuit_breaker_api_error(self):
        """Circuit breaker should return (False, 0.0) on API error."""
        from order_utils import check_circuit_breaker

        class FakeAPI:
            def get_account(self):
                raise Exception("Network error")

        tripped, dd = check_circuit_breaker(FakeAPI())
        assert not tripped
        assert dd == 0.0
