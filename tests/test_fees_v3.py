"""fees.py v3 contract tests — panel-adjudicated definition pins + guards.

Mac-green by construction: needs only fees, trade_journal, liquidity
(numpy) and stdlib — no torch/lightgbm/optuna/joblib/numba/sklearn/dotenv.
"""

import inspect
import json
import logging
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fees
import liquidity

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def journals(tmp_path, monkeypatch):
    """Point trade_journal.JOURNAL_DIR at a tmp dir; reset the cache."""
    import trade_journal
    monkeypatch.setattr(trade_journal, 'JOURNAL_DIR', tmp_path)
    fees._maker_share_cache = None
    yield tmp_path
    fees._maker_share_cache = None


def _write_rows(path_dir, rows, day_offset=0):
    day = (datetime.now().date() - timedelta(days=day_offset)).isoformat()
    with open(path_dir / f'{day}.jsonl', 'a') as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r)) + '\n')


def _maker_rows(n):
    return [{'symbol': 'BTC/USD', 'action': 'buy',
             'entry_tactic': 'maker'}] * n


class TestValuePins:
    def test_flat_values(self):
        assert fees.round_trip_cost_pct('crypto', 0.10) == pytest.approx(0.60)
        assert fees.required_edge_pct('crypto', 0.10) == pytest.approx(1.20)
        assert fees.round_trip_cost_pct('stock', 0.05) == pytest.approx(0.113)
        assert fees.required_edge_pct('stock', 0.05) == pytest.approx(0.226)
        assert fees.round_trip_cost_pct('stock', 0.0) == pytest.approx(0.063)

    def test_required_is_cost_times_multiple_all_flags(self):
        for a in ('crypto', 'stock'):
            for mk in (False, True):
                for s in (0.0, 0.05, 0.10, 0.5):
                    for m in (1.0, 2.0, 3.5):
                        assert fees.required_edge_pct(
                            a, s, maker=mk, min_edge=m) == pytest.approx(
                            fees.round_trip_cost_pct(a, s, maker=mk) * m)


class TestLinearityContract:
    """liquidity.per_bar_round_trip_cost vectorizes on fee_const + spread —
    a CORRECTNESS precondition of every per-bar backtest/meta cost."""

    def test_additive_in_spread_all_branches(self):
        for a in ('crypto', 'stock'):
            for mk in (False, True):
                base = fees.round_trip_cost_pct(a, 0.0, maker=mk)
                for s in (0.0, 1e-6, 0.05, 0.10, 1.5):
                    assert fees.round_trip_cost_pct(
                        a, s, maker=mk) == pytest.approx(base + s)

    def test_vector_path_matches_scalar_in_domain(self):
        for a in ('crypto', 'stock'):
            for s in (0.0, 0.05, 0.10, 1.5):
                got = liquidity.per_bar_round_trip_cost(a, np.array([s]))[0]
                assert got == pytest.approx(fees.round_trip_cost_pct(a, s))


class TestMakerLiveSemantics:
    def test_exit_leg_always_taker_and_maker_delta_fee_only(self):
        delta = (fees.CRYPTO_TAKER_BPS - fees.CRYPTO_MAKER_BPS) / 100.0
        for s in (0.0, 0.02, 0.10, 0.5):
            assert (fees.round_trip_cost_pct('crypto', s)
                    - fees.round_trip_cost_pct('crypto', s, maker=True)
                    ) == pytest.approx(delta)
        assert fees.round_trip_cost_pct(
            'crypto', 0.0, maker=True) == pytest.approx(
            (fees.CRYPTO_MAKER_BPS + fees.CRYPTO_TAKER_BPS) / 100.0)

    def test_maker_is_noop_for_stock(self):
        assert fees.round_trip_cost_pct('stock', 0.05, maker=True) == \
            fees.round_trip_cost_pct('stock', 0.05)

    def test_maker_true_is_the_documented_offline_subtaker_exception(self):
        # maker=True bypasses the static-taker rule even with live=False —
        # documented exception (see round_trip_cost_pct docstring), not a bug.
        assert fees.round_trip_cost_pct(
            'crypto', 0.10, maker=True, live=False) == pytest.approx(0.50)

    def test_live_is_crypto_only_and_maker_overrides_live(self, journals):
        _write_rows(journals, _maker_rows(40))  # 100% maker realized
        assert fees.round_trip_cost_pct('stock', 0.05, live=True) == \
            fees.round_trip_cost_pct('stock', 0.05)
        assert fees.round_trip_cost_pct(
            'crypto', 0.10, maker=True, live=True) == \
            fees.round_trip_cost_pct('crypto', 0.10, maker=True)


class TestDegenerateSpreadInputs:
    def test_nan_propagates_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            assert math.isnan(fees.round_trip_cost_pct('stock', float('nan')))
            assert math.isnan(fees.required_edge_pct('crypto', float('nan')))
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('spread_pct' in m for m in msgs)
        # backtest.py:315 admits on `p < edge_floor`; a NaN floor fails OPEN
        # there (0.9 < nan is False) — characterization, not endorsement.
        assert (0.9 < fees.required_edge_pct('crypto', float('nan'))) is False

    def test_negative_clamps_to_zero_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            got = fees.round_trip_cost_pct('crypto', -0.2)
        # crossed quote prices as a FREE book — open owner decision
        assert got == pytest.approx(fees.round_trip_cost_pct('crypto', 0.0))
        assert got < fees.round_trip_cost_pct(
            'crypto', fees.FLAT_SPREAD_PCT['crypto'])
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('spread_pct' in m for m in msgs)

    def test_inf_propagates_inf_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            got = fees.round_trip_cost_pct('crypto', float('inf'))
            floor = fees.required_edge_pct('crypto', float('inf'))
        assert math.isinf(got) and got > 0
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('spread_pct' in m for m in msgs)
        # An infinite floor REJECTS every prediction (p < inf is True in
        # backtest's admit check) — the OPPOSITE failure mode from NaN's
        # fail-open; both degenerate states must warn, not just NaN.
        assert (0.9 < floor) is True

    def test_zero_and_positive_stay_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger='fees'):
            fees.round_trip_cost_pct('stock', 0.0)
            fees.round_trip_cost_pct('crypto', 0.10)
        assert not [r for r in caplog.records if r.name == 'fees']

    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            fees.round_trip_cost_pct('stock', None)

    def test_min_edge_zero_disables_the_gate(self):
        # documented consequence: floor 0.0 admits every non-zero prediction
        assert fees.required_edge_pct('crypto', 0.10, min_edge=0.0) == 0.0


class TestCallTimeConstantResolution:
    def test_min_edge_default_tracks_module_constant(self, monkeypatch):
        # pre-fix this returned 2.0x — the promotion gate (backtest.py:277
        # omits min_edge) would not track a retune the live gate tracks.
        monkeypatch.setattr(fees, 'MIN_EDGE_MULTIPLE', 5.0)
        want = fees.round_trip_cost_pct('stock', 0.05) * 5.0
        assert fees.required_edge_pct('stock', 0.05) == pytest.approx(want)
        assert fees.required_edge_pct(
            'stock', 0.05, min_edge=None) == pytest.approx(want)

    def test_min_edge_signature_default_is_none(self):
        p = inspect.signature(fees.required_edge_pct).parameters['min_edge']
        assert p.default is None

    def test_maker_share_constants_resolve_at_call_time(self, journals,
                                                        monkeypatch):
        monkeypatch.setattr(fees, 'MAKER_SHARE_MIN_ENTRIES', 5)
        _write_rows(journals, _maker_rows(10))
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)

    def test_default_cache_key_is_resolved_constants(self, journals):
        _write_rows(journals, _maker_rows(40))
        fees.realized_crypto_maker_share()
        assert list(fees._maker_share_cache.keys()) == [
            (fees.MAKER_SHARE_WINDOW_DAYS, fees.MAKER_SHARE_MIN_ENTRIES)]


class TestScanRowGuards:
    def test_malformed_rows_skipped_not_fatal(self, journals, caplog):
        # completes fixes_applied[124]: the sibling guards for the row
        # object and entry_tactic that #124 shipped for `symbol`.
        _write_rows(journals, _maker_rows(30))
        _write_rows(journals, [
            'null',                                   # skipped by prefilter
            '"entry_tactic"',                         # parses to str, not dict
            '["entry_tactic"]',                       # parses to list
            {'symbol': 'BTC/USD', 'action': 'buy', 'entry_tactic': True},
            {'symbol': 'BTC/USD', 'action': 'buy', 'entry_tactic': 1},
        ])
        with caplog.at_level(logging.WARNING, logger='fees'):
            assert fees.realized_crypto_maker_share() == pytest.approx(1.0)
        assert not [r for r in caplog.records if r.name == 'fees']

    def test_prefilter_is_superset_not_semantic(self, journals):
        # key text inside a VALUE: parsed then rejected by the predicate;
        # rows lacking the key entirely: skipped cheaply — neither counted.
        _write_rows(journals, _maker_rows(30))
        _write_rows(journals, [
            {'symbol': 'BTC/USD', 'action': 'skip',
             'llm_reasoning': 'the "entry_tactic" was maker'},
            {'symbol': 'ETH/USD', 'action': 'buy'},
        ])
        assert fees.realized_crypto_maker_share() == pytest.approx(1.0)


class TestRecomputeLogging:
    def test_thin_sample_logs_inactive_with_n(self, journals, caplog):
        _write_rows(journals, _maker_rows(10))
        with caplog.at_level(logging.INFO, logger='fees'):
            assert fees.realized_crypto_maker_share() is None
        msgs = [r.getMessage() for r in caplog.records if r.name == 'fees']
        assert any('maker-share' in m and '10 crypto entries' in m
                   and 'thin sample' in m for m in msgs)

    def test_active_share_logged_once_per_recompute(self, journals, caplog):
        _write_rows(journals, _maker_rows(40))
        with caplog.at_level(logging.INFO, logger='fees'):
            fees.realized_crypto_maker_share()
            fees.realized_crypto_maker_share()  # TTL cache hit — no re-log
        msgs = [r.getMessage() for r in caplog.records
                if r.name == 'fees' and 'maker-share:' in r.getMessage()]
        assert len(msgs) == 1
        assert 'share=1.000' in msgs[0]


class TestBlendClamp:
    def test_out_of_range_share_clamped_to_fee_schedule(self, monkeypatch):
        monkeypatch.setattr(fees, 'realized_crypto_maker_share',
                            lambda *a, **k: 1.5)
        assert fees.crypto_entry_fee_bps(live=True) == fees.CRYPTO_MAKER_BPS
        monkeypatch.setattr(fees, 'realized_crypto_maker_share',
                            lambda *a, **k: -0.5)
        assert fees.crypto_entry_fee_bps(live=True) == fees.CRYPTO_TAKER_BPS


class TestCrossSourceGuards:
    def test_live_true_only_in_order_utils(self):
        # The module's hardest invariant: every offline context prices full
        # taker. A single live=True keyword in any of these files would
        # loosen training/promotion/meta with a green suite.
        #
        # Adapted from the literal spec pattern (`live\s*=\s*True`): that
        # naive form also matches liquidity.py's docstring prose ("fees.py
        # forbids live=True outside live paths" — prose, not a call), which
        # would fail this test against liquidity.py even though no code
        # there ever passes live=True. Requiring the match be followed by
        # the closing `)` or a `,` (i.e. it reads as an actual keyword
        # argument in a call, as at order_utils.py's `live=True)`) keeps the
        # invariant honest without editing liquidity.py (out of ownership
        # scope here).
        pat = re.compile(r'live\s*=\s*True\s*[,)]')
        assert pat.search((REPO / 'order_utils.py').read_text())
        for fname in ('backtest.py', 'meta_label.py', 'decision_report.py',
                      'llm_analyst.py', 'liquidity.py', 'short_cost.py',
                      'scripts/hypersearch_v2.py'):
            assert not pat.search((REPO / fname).read_text()), (
                f'{fname} passes live=True — offline contexts must stay on '
                'the conservative static taker model (fees.crypto_entry_fee_bps)')

    def test_hypersearch_txn_cost_tracks_fees(self):
        # The training objective's own un-imported cost copy — the only one
        # with no cross-check before this test. Drift here selects models
        # on economics no gate will honor.
        src = (REPO / 'scripts' / 'hypersearch_v2.py').read_text()
        m = re.search(r"TXN_COST_PCT\s*=\s*\{'crypto':\s*([\d.]+),"
                      r"\s*'stock':\s*([\d.]+)\}", src)
        assert m, ('TXN_COST_PCT literal not found — if hypersearch_v2 now '
                   'imports fees, update this test')
        crypto, stock = float(m.group(1)), float(m.group(2))
        # crypto matches exactly today (0.60); stock 0.11 vs 0.113 is a
        # known accepted rounding — a fee-schedule change is not.
        assert abs(crypto - fees.round_trip_cost_pct(
            'crypto', fees.FLAT_SPREAD_PCT['crypto'])) <= 0.005
        assert abs(stock - fees.round_trip_cost_pct(
            'stock', fees.FLAT_SPREAD_PCT['stock'])) <= 0.005

    def test_llm_analyst_spread_literals_track_flat_spread(self):
        src = (REPO / 'llm_analyst.py').read_text()
        m = re.search(r"spread_pct=([\d.]+)\s+if\s+asset_type\s*==\s*"
                      r"'crypto'\s+else\s+([\d.]+)", src)
        assert m, ('spread literal not found — if llm_analyst now imports '
                   'FLAT_SPREAD_PCT, update this test')
        assert float(m.group(1)) == pytest.approx(
            fees.FLAT_SPREAD_PCT['crypto'])
        assert float(m.group(2)) == pytest.approx(
            fees.FLAT_SPREAD_PCT['stock'])


class TestDocstringTruth:
    def test_required_edge_names_admission_floor(self):
        d = fees.required_edge_pct.__doc__
        assert 'ADMISSION' in d and 'break-even' in d

    def test_round_trip_names_midpoint_and_linearity(self):
        d = fees.round_trip_cost_pct.__doc__
        assert 'MIDPOINT' in d and 'linear' in d.lower()

    def test_maker_share_names_count_share(self):
        d = fees.realized_crypto_maker_share.__doc__
        assert 'ORDER-COUNT' in d
