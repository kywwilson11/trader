"""Wave-7 Finding 1: calibrated entry-tactic table (execution_policy).

Pins the decision: crypto always posts; mega always crosses (never 'wide');
tight stock spreads cross; genuinely-wide spec spreads post; the middle band
ladders; and edge headroom gates whether a non-spec wide name risks a passive
non-fill."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution_policy import choose_entry_tactic


class TestCrypto:
    def test_crypto_always_posts(self):
        for sp in (0.01, 0.1, 0.5):
            out = choose_entry_tactic('crypto', sp)
            assert out['tactic'] == 'post'


class TestMega:
    def test_mega_always_crosses_even_when_quote_wide(self):
        # a transient wide quote on a mega name must NOT be treated as 'wide'
        out = choose_entry_tactic('stock', 0.40, name_class='mega')
        assert out['tactic'] == 'cross'

    def test_mega_tight_crosses(self):
        assert choose_entry_tactic('stock', 0.02, name_class='mega')['tactic'] == 'cross'


class TestStockBands:
    def test_tight_spread_crosses(self):
        out = choose_entry_tactic('stock', 0.03, name_class='mid')
        assert out['tactic'] == 'cross' and out['post_offset_pct'] == 0.0

    def test_spec_wide_posts(self):
        out = choose_entry_tactic('stock', 0.30, name_class='spec')
        assert out['tactic'] == 'post' and out['post_offset_pct'] > 0

    def test_mid_band_ladders(self):
        # between taker floor (0.05) and wide (0.15) -> ladder
        out = choose_entry_tactic('stock', 0.10, name_class='mid')
        assert out['tactic'] == 'ladder'

    def test_wide_nonspec_needs_edge_headroom_to_post(self):
        # wide mid-class name, thin edge -> NOT post (would risk a non-fill on
        # thin edge); falls to ladder
        thin = choose_entry_tactic('stock', 0.20, pred_return=0.10,
                                   edge_floor=0.10, name_class='mid')
        assert thin['tactic'] == 'ladder'
        # same name, fat edge (>=1.5x floor) -> post
        fat = choose_entry_tactic('stock', 0.20, pred_return=0.30,
                                  edge_floor=0.10, name_class='mid')
        assert fat['tactic'] == 'post'

    def test_unknown_class_defaults_mid(self):
        out = choose_entry_tactic('stock', 0.10, name_class='bogus')
        assert out['tactic'] == 'ladder'

    def test_post_offset_is_inside_half_spread(self):
        sp = 0.30
        out = choose_entry_tactic('stock', sp, name_class='spec')
        # offset = sp * 0.5 * EXEC_POST_INSIDE_FRAC(0.40) = 0.06
        assert out['post_offset_pct'] == pytest.approx(0.06, abs=1e-9)
        assert out['post_offset_pct'] < sp  # never crosses the far touch
