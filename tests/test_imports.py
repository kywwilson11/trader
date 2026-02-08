"""Smoke test: every module except gui.py imports without error."""

import importlib

import pytest

# All .py modules in the project root (excluding gui.py which needs PySide6)
MODULES = [
    "connection_test",
    "crypto_loop",
    "evolve",
    "fundamentals",
    "harvest_crypto_data",
    "harvest_stock_data",
    "hw_monitor",
    "hypersearch_dual",
    "indicator_config",
    "indicators",
    "llm_analyst",
    "llm_client",
    "llm_config",
    "market_data",
    "model",
    "order_utils",
    "predict_now",
    "run_pipeline",
    "sentiment",
    "sentiment_history",
    "stock_config",
    "stock_loop",
    "trade_journal",
    "trading_utils",
    "watchdog",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    """Module imports without raising."""
    importlib.import_module(module_name)
