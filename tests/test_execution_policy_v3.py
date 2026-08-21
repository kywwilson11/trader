"""Panel adjudication (2026-07, batch A) — execution_policy hardening pins.

Characterization + invariant tests for choose_entry_tactic: band-boundary
inclusivity, branch precedence, EXEC_* config invariants, the degraded-input
reason vocabulary, the name_class echo/coercion marker, the non-finite
headroom guard, the edge_floor contract fork (documented, unresolved — owner
decision), the journal-vocabulary collision tripwire, and the
strategy_config DECLARED-AHEAD dormancy marker. Pure imports only — runs on
the dev Mac.
"""

import logging
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

import fees
from execution_policy import VALID_CLASSES, choose_entry_tactic
from stock_config import SECTOR_BUCKETS
from strategy_config import (EXEC_EDGE_HEADROOM_MULT, EXEC_POST_INSIDE_FRAC,
                             EXEC_TAKER_FLOOR_PCT, EXEC_WIDE_SPREAD_PCT)


class TestConfigInvariants:
    """A config typo here silently rewrites entry tactics — pin the ranges."""

    def test_threshold_ordering(self):
        # floor >= wide would delete the ladder band and shadow the post row
        assert 0.0 < EXEC_TAKER_FLOOR_PCT < EXEC_WIDE_SPREAD_PCT

    def test_post_frac_stays_inside_half_spread(self):
        # >= 1.0 lets a 'post' reach/cross the mid; >= 2.0 crosses the far
        # touch — a taker fill journaled as passive.
        assert 0.0 < EXEC_POST_INSIDE_FRAC < 1.0

    def test_headroom_mult_at_least_one(self):
        # < 1.0 would risk a passive non-fill on an edge thinner than its floor
        assert EXEC_EDGE_HEADROOM_MULT >= 1.0


class TestBoundaries:
    def test_taker_floor_inclusive(self):
        assert choose_entry_tactic('stock', EXEC_TAKER_FLOOR_PCT,
                                   name_class='mid')['tactic'] == 'cross'
        just_above = math.nextafter(EXEC_TAKER_FLOOR_PCT, math.inf)
        assert choose_entry_tactic('stock', just_above,
                                   name_class='mid')['tactic'] == 'ladder'

    def test_wide_threshold_inclusive_for_spec(self):
        assert choose_entry_tactic('stock', EXEC_WIDE_SPREAD_PCT,
                                   name_class='spec')['tactic'] == 'post'
        just_below = math.nextafter(EXEC_WIDE_SPREAD_PCT, 0.0)
        assert choose_entry_tactic('stock', just_below,
                                   name_class='spec')['tactic'] == 'ladder'

    def test_wide_threshold_mid_without_headroom_ladders(self):
        assert choose_entry_tactic('stock', EXEC_WIDE_SPREAD_PCT,
                                   name_class='mid')['tactic'] == 'ladder'

    def test_headroom_boundary_inclusive(self):
        floor = 0.10
        exactly = EXEC_EDGE_HEADROOM_MULT * floor
        out = choose_entry_tactic('stock', 0.20, pred_return=exactly,
                                  edge_floor=floor, name_class='mid')
        assert out['tactic'] == 'post'
        below = math.nextafter(exactly, 0.0)
        out = choose_entry_tactic('stock', 0.20, pred_return=below,
                                  edge_floor=floor, name_class='mid')
        assert out['tactic'] == 'ladder'


class TestPrecedence:
    def test_crypto_beats_name_class(self):
        # BTC/ETH will rank 'mega' under any sane Eff_Spread_Pct seed; the
        # crypto row must stay ABOVE the class table or every crypto entry
        # silently flips maker -> taker (25 vs 15 bps/side).
        for sp in (0.02, 0.40):
            assert choose_entry_tactic('crypto', sp,
                                       name_class='mega')['tactic'] == 'post'

    def test_mega_beats_bands(self):
        out = choose_entry_tactic('stock', 0.40, name_class='mega')
        assert out['tactic'] == 'cross'
        assert out['reason'] == 'mega_tight_book_cross'


class TestReasonVocabulary:
    """`reason` is the module's only audit surface — pin the disambiguated
    codes. Tactic and post_offset_pct for every input here are unchanged
    from the pre-adjudication module; only the labels were split."""

    def test_missing_spread_is_labelled_unavailable(self):
        for bad in (None, float('nan'), float('inf'), float('-inf'),
                    'garbage', [], {}, object()):
            out = choose_entry_tactic('stock', bad, name_class='mid')
            assert out['tactic'] == 'cross' and out['post_offset_pct'] == 0.0
            assert out['reason'] == 'spread_unavailable_cross', bad

    def test_crossed_quote_is_labelled_crossed(self):
        out = choose_entry_tactic('stock', -0.5, name_class='spec')
        assert out['tactic'] == 'cross' and out['post_offset_pct'] == 0.0
        assert out['reason'] == 'crossed_quote_cross'

    def test_genuine_tight_spread_keeps_original_reason(self):
        for sp in (0.0, 0.03, EXEC_TAKER_FLOOR_PCT):
            assert (choose_entry_tactic('stock', sp,
                                        name_class='mid')['reason']
                    == 'spread_below_taker_floor')

    def test_wide_no_headroom_is_not_mid_band(self):
        # A wide spread that failed the spec/headroom test is a DIFFERENT
        # regime from a genuine 0.05-0.15 middle band: it points at
        # EXEC_EDGE_HEADROOM_MULT, not EXEC_WIDE_SPREAD_PCT.
        out = choose_entry_tactic('stock', 0.20, pred_return=0.10,
                                  edge_floor=0.10, name_class='mid')
        assert out['tactic'] == 'ladder'
        assert out['reason'] == 'wide_no_headroom_ladder'

    def test_genuine_mid_band(self):
        out = choose_entry_tactic('stock', 0.10, name_class='mid')
        assert out['tactic'] == 'ladder'
        assert out['reason'] == 'mid_band_ladder'


class TestNameClassEcho:
    def test_resolved_class_and_coercion_flag(self):
        out = choose_entry_tactic('stock', 0.10, name_class='bogus')
        assert out['name_class'] == 'mid'
        assert out['name_class_coerced'] is True
        assert out['tactic'] == 'ladder'  # behavior unchanged, only visible now
        for good in VALID_CLASSES:
            out = choose_entry_tactic('stock', 0.10, name_class=good)
            assert out['name_class'] == good
            assert out['name_class_coerced'] is False

    def test_sector_buckets_vocabulary_is_disjoint(self):
        # borrow_proxy.class_lookup_from_config -> SECTOR_BUCKETS emits
        # 'megacap_tech'/'spec_growth'/... — NOT this module's vocabulary.
        # Wiring it in silently coerces every symbol to 'mid'.
        assert not (set(SECTOR_BUCKETS.values()) & set(VALID_CLASSES))
        out = choose_entry_tactic('stock', 0.40, name_class='spec_growth')
        assert out['name_class'] == 'mid'
        assert out['name_class_coerced'] is True
        assert out['tactic'] == 'ladder'  # NOT the spec/post row

    def test_case_variants_are_not_normalized(self):
        # 'MEGA'/' mega' coerce to 'mid' — normalizing them would change the
        # chosen tactic and is an owner decision; pin today's semantics.
        for variant in ('MEGA', ' mega', None):
            out = choose_entry_tactic('stock', 0.40, name_class=variant)
            assert out['tactic'] == 'ladder', variant
            assert out['name_class_coerced'] is True


class TestHeadroomGuards:
    def test_non_finite_pred_return_fails_closed_to_ladder(self):
        # pred=+inf used to fabricate infinite headroom and buy a passive
        # rest on the widest quotes; non-finite now takes the same path as
        # None (mirror of the shipped non-finite spread guard).
        for bad in (float('inf'), float('-inf'), float('nan')):
            out = choose_entry_tactic('stock', 0.20, pred_return=bad,
                                      edge_floor=0.10, name_class='mid')
            assert out['tactic'] == 'ladder', bad

    def test_non_finite_edge_floor_fails_closed_to_ladder(self):
        for bad in (float('inf'), float('-inf'), float('nan')):
            out = choose_entry_tactic('stock', 0.20, pred_return=0.50,
                                      edge_floor=bad, name_class='mid')
            assert out['tactic'] == 'ladder', bad

    def test_non_numeric_pred_or_floor_fails_closed_not_raises(self):
        # Same unambiguous-bug class as the spread type guard: garbage edge
        # inputs (str/dict/list/object) must fail the headroom test, not
        # raise TypeError out of a pure lookup and kill the caller's whole
        # entry cycle (pre-hardening they raised).
        for bad in ('garbage', [], {}, object()):
            out = choose_entry_tactic('stock', 0.20, pred_return=bad,
                                      edge_floor=0.10, name_class='mid')
            assert out['tactic'] == 'ladder', bad
            out = choose_entry_tactic('stock', 0.20, pred_return=0.50,
                                      edge_floor=bad, name_class='mid')
            assert out['tactic'] == 'ladder', bad

    def test_missing_or_nonpositive_floor_disables_post(self):
        # edge_floor <= 0 / None == 'cost floor unavailable' -> fail closed
        # (documented intent; NOT 'infinite headroom').
        for floor in (None, 0.0, -1.0):
            out = choose_entry_tactic('stock', 0.20, pred_return=5.0,
                                      edge_floor=floor, name_class='mid')
            assert out['tactic'] == 'ladder', floor
        out = choose_entry_tactic('stock', 0.20, pred_return=None,
                                  edge_floor=0.10, name_class='mid')
        assert out['tactic'] == 'ladder'

    def test_finite_headroom_still_posts(self):
        out = choose_entry_tactic('stock', 0.20, pred_return=0.30,
                                  edge_floor=0.10, name_class='mid')
        assert out['tactic'] == 'post'


class TestEdgeFloorContractFork:
    """edge_floor is UNPINNED between two fees.py quantities differing by
    exactly MIN_EDGE_MULTIPLE=2.0. These tests DOCUMENT the fork so a wiring
    cannot pretend it does not exist. Which reading is correct is an OPEN
    OWNER DECISION — do not 'fix' these tests by picking one."""

    def test_two_readings_flip_the_tactic(self):
        sp = 0.20
        # thinnest edge the live should_trade gate admits at this spread
        pred = fees.required_edge_pct('stock', sp) + 1e-9
        raw = choose_entry_tactic(
            'stock', sp, pred_return=pred,
            edge_floor=fees.round_trip_cost_pct('stock', sp),
            name_class='mid')
        req = choose_entry_tactic(
            'stock', sp, pred_return=pred,
            edge_floor=fees.required_edge_pct('stock', sp),
            name_class='mid')
        assert raw['tactic'] == 'post'
        assert req['tactic'] == 'ladder'

    def test_raw_cost_reading_makes_headroom_vacuous(self):
        # Any entry admitted by should_trade satisfies pred > 2.0*raw_cost;
        # 2.0 > EXEC_EDGE_HEADROOM_MULT, so under the raw-cost reading the
        # headroom test can never reject a live-admitted trade.
        assert EXEC_EDGE_HEADROOM_MULT < fees.MIN_EDGE_MULTIPLE


class TestOffsetArithmetic:
    def test_offset_identity_and_half_spread_invariant(self):
        for sp in (0.06, 0.10, 0.16, 0.30, 1.0):
            out = choose_entry_tactic('stock', sp, name_class='spec')
            assert out['post_offset_pct'] == pytest.approx(
                round(sp * 0.5 * EXEC_POST_INSIDE_FRAC, 4), abs=1e-12)
            # true design invariant: the rest price never reaches the MID
            # (tighter than test_execution_policy.py's `< sp` far-touch pin)
            assert out['post_offset_pct'] < sp * 0.5

    def test_post_and_ladder_share_the_offset_formula(self):
        post = choose_entry_tactic('stock', 0.20, name_class='spec')
        ladder = choose_entry_tactic('stock', 0.20, pred_return=0.10,
                                     edge_floor=0.10, name_class='mid')
        assert post['tactic'] == 'post' and ladder['tactic'] == 'ladder'
        assert post['post_offset_pct'] == ladder['post_offset_pct']

    def test_offset_is_unbounded_by_design_today(self):
        # No upper sanity bound on live_spread_pct (owner decision open): a
        # junk-but-finite quote produces an absurd offset. Pin it so any
        # future ceiling is a conscious change, not a silent one.
        out = choose_entry_tactic('stock', 500.0, name_class='spec')
        assert out['tactic'] == 'post'
        assert out['post_offset_pct'] == 100.0


class TestAssetTypeWarning:
    def test_unknown_label_warns_and_routing_is_unchanged(self, caplog):
        with caplog.at_level(logging.WARNING, logger='execution_policy'):
            out = choose_entry_tactic('CRYPTO', 0.02)
        assert out['tactic'] == 'cross'  # stock table, routing unchanged
        assert out['post_offset_pct'] == 0.0
        assert 'unknown asset_type' in caplog.text

    def test_known_labels_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger='execution_policy'):
            choose_entry_tactic('crypto', 0.02)
            choose_entry_tactic('stock', 0.02)
        assert 'unknown asset_type' not in caplog.text


class TestJournalVocabularyCollision:
    def test_tactic_vocabulary_never_matches_maker_prefix(self):
        # fees.realized_crypto_maker_share prefix-matches 'maker' on the
        # journal's entry_tactic field (fees.py). If cross/post/ladder are
        # ever written there, the realized maker share silently reads 0.0
        # and the live crypto cost gate reverts to full-taker pricing. See
        # the CALLER CONTRACT note in execution_policy's module docstring.
        assert not any(t.startswith('maker')
                       for t in ('cross', 'post', 'ladder'))


class TestDocPins:
    def test_execution_policy_contract_docs_present(self):
        src = (REPO / 'execution_policy.py').read_text()
        assert '20%-of-half-spread' in src       # pre-existing b02 pin, keep
        assert '10%-of-half-spread' not in src   # pre-existing b02 pin, keep
        assert 'MAKER_ENTRIES_ENABLED' in src    # HOW-not-WHETHER contract
        assert 'entry_tactic' in src             # journal-vocabulary contract
        assert 'place_maker_buy' in src          # crypto 'post' semantics
        assert '_round_price_band' in src        # tick-rounding obligation
        assert 'SECTOR_BUCKETS' in src           # name_class trap named

    def test_strategy_config_dormancy_marker(self):
        cfg = (REPO / 'strategy_config.py').read_text()
        assert 'the backtester reads too' not in cfg
        assert 'DECLARED-AHEAD: execution_policy.choose_entry_tactic' in cfg

    def test_table_is_still_dormant(self):
        # The day a wiring lands this fails: update the strategy_config
        # DECLARED-AHEAD comment (and these pins) consciously in that change.
        for mod in ('base_loop.py', 'crypto_loop.py', 'stock_loop.py',
                    'order_utils.py', 'backtest.py'):
            assert 'choose_entry_tactic' not in (REPO / mod).read_text(), mod
