"""Wave-7 Finding 10 (feature half): crypto squeeze interaction columns."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from squeeze_features import squeeze_interaction, SQUEEZE_FUNDING_Z


class TestSqueezeInteraction:
    def test_only_high_oi_amplifies(self):
        # negative OI_Z is clipped to 0 -> no amplification regardless of funding
        out = squeeze_interaction([3.0, 3.0], [-2.0, 2.0])
        assert out['Funding_x_OI'][0] == 0.0          # low OI -> 0
        assert out['Funding_x_OI'][1] == pytest.approx(6.0)  # 3 * 2

    def test_squeeze_setup_needs_crowded_shorts_and_oi(self):
        # crowded shorts = very negative funding (< -2) AND high OI
        out = squeeze_interaction([-3.0, -1.0, -3.0], [2.0, 2.0, -1.0])
        assert out['Squeeze_Setup'][0] == pytest.approx(2.0)  # -3 funding, +2 OI
        assert out['Squeeze_Setup'][1] == 0.0   # funding -1 not crowded enough
        assert out['Squeeze_Setup'][2] == 0.0   # OI negative -> no fuel

    def test_threshold_boundary(self):
        # exactly -2 is NOT < -2 -> no setup
        out = squeeze_interaction([SQUEEZE_FUNDING_Z], [3.0])
        assert out['Squeeze_Setup'][0] == 0.0

    def test_nan_neutral_filled(self):
        out = squeeze_interaction([np.nan, 2.0], [3.0, np.nan])
        assert np.isfinite(out['Funding_x_OI']).all()
        assert out['Funding_x_OI'][0] == 0.0  # nan funding -> 0
        assert out['Funding_x_OI'][1] == 0.0  # nan OI -> 0

    def test_sign_preserved(self):
        # positive funding with high OI stays positive; negative stays negative
        out = squeeze_interaction([2.0, -2.5], [3.0, 3.0])
        assert out['Funding_x_OI'][0] > 0 and out['Funding_x_OI'][1] < 0
