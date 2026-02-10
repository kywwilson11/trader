"""Shared fixtures for the trader test suite."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root and scripts/ are on sys.path so imports work
_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / 'scripts'))


@pytest.fixture
def sample_ohlcv_df():
    """120-row OHLCV DataFrame with realistic random walk prices."""
    np.random.seed(42)
    n = 120
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 50000, size=n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


@pytest.fixture
def tmp_json_file(tmp_path):
    """Factory fixture: write a dict to a temp JSON file and return the path."""
    def _make(data, filename="test.json"):
        p = tmp_path / filename
        p.write_text(json.dumps(data))
        return p
    return _make
