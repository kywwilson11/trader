"""Tests for wave-5 Tier1-1 conviction instrumentation (measurement-only).

Verifies the journaling helpers fire with the right fields, NEVER affect
control flow, and that decision_report consumes the new rows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import base_loop


class _Stub(base_loop.BaseTradingLoop):
    """Minimal concrete loop for exercising the journaling helpers in
    isolation (no API, no real gates)."""
    MODEL_PREFIX = ''

    def __init__(self):
        # bypass the heavy BaseTradingLoop.__init__
        self.trade_threshold = 0.2
        self._buys_allowed = True
        self._last_meta_p = {}

    def get_asset_type(self):
        return 'stock'

    # ABC stubs — unused by the journaling helpers under test
    def check_market_hours(self): return True
    def flatten_before_close(self): pass
    def get_benchmark_close(self): return None
    def get_headlines(self, s): return []
    def get_quote(self, s): return None
    def get_symbol_universe(self): return []
    def place_buy_order(self, *a, **k): return None
    def place_sell_order(self, *a, **k): return None
    def write_prediction_cache(self, *a, **k): pass


@pytest.fixture
def stub(monkeypatch):
    loop = _Stub()
    captured = []
    monkeypatch.setattr(base_loop, 'log_decision', captured.append)
    return loop, captured


class TestConvictionFields:
    def test_tier_thresholds(self, stub):
        loop, _ = stub
        # strong pred (>=1.5x thr) AND strong meta (>=0.6) -> A
        assert loop._conviction_tier(0.35, 0.7) == 'A'
        # only one strong -> B
        assert loop._conviction_tier(0.35, 0.4) == 'B'
        assert loop._conviction_tier(0.21, 0.7) == 'B'
        # neither -> C
        assert loop._conviction_tier(0.21, 0.4) == 'C'
        # None meta, strong pred -> B
        assert loop._conviction_tier(0.35, None) == 'B'

    def test_conv_fields_compose(self, stub):
        loop, _ = stub
        loop._last_meta_p['NVDA'] = 0.72
        snap = {'Q10': -0.5, 'Q10_Floor': -1.2}
        f = loop._conv_fields('NVDA', 0.4, snap, rank=2)
        assert f['entry_rank'] == 2
        assert f['pred_thresh_ratio'] == 2.0      # 0.4 / 0.2
        assert f['q10'] == -0.5 and f['q10_floor'] == -1.2
        assert f['meta_p'] == 0.72
        assert f['conviction_tier'] == 'A'

    def test_conv_fields_minimal_without_meta(self, stub):
        loop, _ = stub
        f = loop._conv_fields('BTC/USD', 0.3, {}, rank=None)
        assert 'entry_rank' not in f
        assert 'meta_p' not in f and 'conviction_tier' not in f
        assert f['pred_thresh_ratio'] == 1.5


class TestJournalHelpers:
    def test_journal_skip_emits_priced_row(self, stub):
        loop, cap = stub
        loop._last_meta_p['AMD'] = 0.55
        loop._journal_skip('AMD', 'correlation', rank=3, pred=0.4,
                           snapshot={'Q10': -0.2, 'Q10_Floor': -0.9})
        assert len(cap) == 1
        r = cap[0]
        assert r['action'] == 'skip' and r['skip_reason'] == 'correlation'
        assert r['symbol'] == 'AMD' and r['entry_rank'] == 3
        assert r['pred_return'] == 0.4 and r['meta_p'] == 0.55

    def test_journal_window_summary(self, stub):
        loop, cap = stub
        from collections import Counter
        loop._journal_entry_window(5, ['NVDA', 'AMD'],
                                   Counter({'below_threshold': 2,
                                            'cooldown': 1}))
        assert len(cap) == 1
        r = cap[0]
        assert r['action'] == 'entry_window'
        assert r['n_candidates'] == 5 and r['admitted_k'] == 2
        assert r['admitted'] == ['NVDA', 'AMD']
        assert r['veto_counts']['below_threshold'] == 2

    def test_window_skipped_when_no_candidates(self, stub):
        loop, cap = stub
        loop._journal_entry_window(0, [], {})
        assert cap == []  # nothing evaluated -> no row

    def test_disabled_by_flag(self, stub, monkeypatch):
        loop, cap = stub
        import strategy_config
        monkeypatch.setattr(strategy_config, 'CONVICTION_JOURNAL_ENABLED',
                            False, raising=False)
        loop._journal_skip('NVDA', 'correlation', rank=1, pred=0.4)
        loop._journal_entry_window(3, ['NVDA'], {'x': 1})
        assert cap == []

    def test_journal_never_raises(self, stub, monkeypatch):
        loop, _ = stub

        def boom(_):
            raise RuntimeError('disk full')

        monkeypatch.setattr(base_loop, 'log_decision', boom)
        # Must swallow — instrumentation can never break the trading loop
        loop._journal_skip('NVDA', 'correlation', rank=1, pred=0.4)
        loop._journal_entry_window(3, ['NVDA'], {'x': 1})


class TestMetaGateStash:
    def test_meta_gate_stashes_probability(self, stub, monkeypatch):
        loop, cap = stub
        import meta_label
        monkeypatch.setattr(meta_label, 'meta_probability_live',
                            lambda *a, **k: 0.66)
        monkeypatch.setattr(meta_label, 'meta_size_mult', lambda p: 1.1)
        ok, mult = loop._meta_gate('NVDA', 0.4, {'NVDA': {}})
        assert ok and loop._last_meta_p['NVDA'] == 0.66

    def test_meta_veto_row_carries_rank(self, stub, monkeypatch):
        loop, cap = stub
        import meta_label
        monkeypatch.setattr(meta_label, 'meta_probability_live',
                            lambda *a, **k: 0.10)  # below META_VETO_PROB
        ok, _ = loop._meta_gate('NVDA', 0.4, {'NVDA': {}}, rank=4)
        assert not ok
        veto = [r for r in cap if r.get('skip_reason') == 'meta_veto'][0]
        assert veto['entry_rank'] == 4 and veto['meta_prob'] == 0.1


class TestDecisionReportConsumers:
    def test_admitted_k_distribution(self):
        import decision_report as dr
        rows = [
            {'action': 'entry_window', 'asset_type': 'stock',
             'admitted_k': 2, 'n_candidates': 7,
             'veto_counts': {'below_threshold': 3, 'cooldown': 2}},
            {'action': 'entry_window', 'asset_type': 'stock',
             'admitted_k': 0, 'n_candidates': 7,
             'veto_counts': {'meta_veto': 5}},
            {'action': 'entry_window', 'asset_type': 'stock',
             'admitted_k': 6, 'n_candidates': 7, 'veto_counts': {}},
            {'action': 'buy', 'symbol': 'NVDA'},  # ignored
        ]
        out = dr.admitted_k_distribution(rows)
        assert out['stock']['windows'] == 3
        assert out['stock']['mean_admitted_k'] == pytest.approx(2.67, abs=0.01)
        assert out['stock']['pct_windows_k_ge_6'] == pytest.approx(1 / 3, abs=1e-3)
        assert out['stock']['pct_windows_zero'] == pytest.approx(1 / 3, abs=1e-3)
        assert out['stock']['total_vetoes_by_reason']['below_threshold'] == 3
        assert out['stock']['admitted_k_hist']['0'] == 1

    def test_gate_reasons_include_new_priced_gates(self):
        import decision_report as dr
        for g in ('below_threshold', 'correlation', 'bucket_cap',
                  'winners_curse', 'trend_filter', 'sizing_zero'):
            assert g in dr.GATE_REASONS
