"""Tests for market_data.py — yfinance column flattening."""

import pandas as pd
import pytest

from market_data import flatten_yfinance_columns


class TestFlattenYfinanceColumns:
    def test_flattens_multiindex(self):
        arrays = [["Close", "Open", "Volume"], ["BTC-USD", "BTC-USD", "BTC-USD"]]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2, 3]], columns=index)

        result = flatten_yfinance_columns(df)
        assert list(result.columns) == ["Close", "Open", "Volume"]

    def test_noop_on_flat_columns(self):
        df = pd.DataFrame({"Close": [1], "Open": [2], "Volume": [3]})
        result = flatten_yfinance_columns(df)
        assert list(result.columns) == ["Close", "Open", "Volume"]
