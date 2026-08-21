"""Smoke test: every module except gui.py imports without error."""

import importlib

import pytest

# All .py modules in the project root (excluding gui.py which needs PySide6)
MODULES = [
    "connection_test",
    "crypto_loop",
    "fundamentals",
    "harvest_crypto_data",
    "harvest_stock_data",
    "hw_monitor",
    "indicator_config",
    "indicators",
    "llm_analyst",
    "llm_client",
    "llm_config",
    "market_data",
    "order_utils",
    "predict_now",
    "run_pipeline",
    "sentiment",
    "sentiment_history",
    "stock_config",
    "stock_loop",
    "trade_journal",
    "trading_utils",
]

# Heavy dependencies absent on the dev Mac (CLAUDE.md two-machine table). A
# ModuleNotFoundError rooted at one of these is an environment gap -> SKIP.
# ANY other import error (repo-internal ModuleNotFoundError, ImportError,
# SyntaxError, ...) is real breakage and must FAIL. On Jetson/CI (full deps)
# nothing skips, so this cannot mask breakage where the deps exist.
HEAVY_DEPS = {
    "torch", "torchvision", "lightgbm", "optuna", "joblib", "numba",
    "sklearn", "dotenv", "alpaca", "alpaca_trade_api", "finnhub",
    "PySide6", "pyqtgraph",
}


def import_or_skip_missing_heavy(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        root = (e.name or "").split(".")[0]
        if root in HEAVY_DEPS:
            pytest.skip(f"missing heavy dependency: {e.name} (dev-Mac env gap)")
        raise


@pytest.mark.parametrize("module_name", MODULES)
def test_import(module_name):
    """Module imports without raising."""
    import_or_skip_missing_heavy(module_name)
