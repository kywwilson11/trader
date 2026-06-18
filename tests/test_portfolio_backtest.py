"""Wave-5 Tier1-2: cross-sectional A/B policy engine (portfolio_backtest).

Verifies the admission policies, the equal-weight period return, turnover-cost
accounting (a held name isn't re-charged; a churned name is), cash periods, and
the A/B comparator deltas — the instrument that scores the conviction flagship."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio_backtest as pb


def _period(*triples):
    # triples of (symbol, signal, fwd_return) [+ optional meta_p]
    out = []
    for t in triples:
        d = {'symbol': t[0], 'signal': t[1], 'fwd_return': t[2]}
        if len(t) > 3:
            d['meta_p'] = t[3]
        out.append(d)
    return out


class TestPolicies:
    def test_top_k_admits_highest_signal(self):
        cands = pb._sorted_desc(_period(('A', 0.1, 1), ('B', 0.9, 1), ('C', 0.5, 1)))
        admitted = pb.top_k(2)(cands)
        assert [c['symbol'] for c in admitted] == ['B', 'C']

    def test_conviction_gate_floors_dynamic_k(self):
        cands = pb._sorted_desc(_period(('A', 0.9, 1, 0.7), ('B', 0.8, 1, 0.4),
                                        ('C', 0.7, 1, 0.65)))
        # meta_floor 0.6 drops B (0.4) -> only A and C admitted (k floats to 2)
        admitted = pb.conviction_gated(3, meta_floor=0.6)(cands)
        assert [c['symbol'] for c in admitted] == ['A', 'C']

    def test_conviction_gate_can_go_to_zero(self):
        cands = pb._sorted_desc(_period(('A', 0.9, 1, 0.3), ('B', 0.8, 1, 0.2)))
        assert pb.conviction_gated(3, meta_floor=0.6)(cands) == []


class TestRunPolicy:
    def test_equal_weight_period_return(self):
        panel = [_period(('A', 0.9, 2.0), ('B', 0.5, 0.0))]
        r = pb.run_policy(panel, pb.top_k(2))
        assert r['gross_total'] == pytest.approx(1.0)  # mean(2,0)
        assert r['mean_admitted_k'] == 2.0

    def test_held_name_not_recharged_but_churn_is(self):
        # policy A: holds {A} both periods -> 1 entry, no churn cost period 2.
        hold = [_period(('A', 0.9, 1.0)), _period(('A', 0.9, 1.0))]
        rh = pb.run_policy(hold, pb.top_k(1), cost_pct=0.5)
        # only one entry total (period 1); period 2 reuses A
        assert rh['entries'] == 1
        # churn: top-1 flips A->B->A; pays an entry every period
        churn = [_period(('A', 0.9, 1.0), ('B', 0.1, 1.0)),
                 _period(('B', 0.9, 1.0), ('A', 0.1, 1.0))]
        rc = pb.run_policy(churn, pb.top_k(1), cost_pct=0.5)
        assert rc['entries'] == 2 and rc['exits'] >= 1
        assert rc['net_total'] < rh['net_total']  # churn pays more cost

    def test_cash_period_earns_zero(self):
        panel = [_period(('A', 0.1, 5.0))]  # gated out -> all cash
        r = pb.run_policy(panel, pb.conviction_gated(3, signal_floor=0.5))
        assert r['mean_admitted_k'] == 0.0
        assert r['pct_periods_cash'] == 1.0
        assert r['net_total'] == 0.0

    def test_perfect_signal_positive_sharpe(self):
        # signal ranks returns correctly every period -> top-1 captures the best
        panel = [_period(('A', 0.9, 1.0), ('B', 0.1, -1.0)),
                 _period(('C', 0.9, 1.0), ('D', 0.1, -1.0)),
                 _period(('E', 0.9, 1.0), ('F', 0.1, -1.0))]
        r = pb.run_policy(panel, pb.top_k(1))
        assert r['net_total'] == pytest.approx(3.0)
        assert r['hit_rate'] == 1.0


class TestCompare:
    def test_ab_deltas(self):
        panel = [_period(('A', 0.9, 2.0, 0.7), ('B', 0.8, -1.0, 0.3)),
                 _period(('C', 0.9, 1.5, 0.65), ('D', 0.8, -1.0, 0.2))]
        out = pb.compare(panel, {
            'fixed_top2': pb.top_k(2),
            'conviction': pb.conviction_gated(2, meta_floor=0.6),
        }, baseline='fixed_top2')
        # conviction drops the negative-return low-meta names -> higher net
        assert out['baseline'] == 'fixed_top2'
        assert out['deltas']['conviction']['net_delta'] > 0
        assert out['deltas']['conviction']['k_delta'] < 0  # fewer trades
        assert out['deltas']['fixed_top2']['sharpe_delta'] == 0.0  # vs itself

    def test_default_baseline_is_first(self):
        panel = [_period(('A', 0.9, 1.0))]
        out = pb.compare(panel, {'x': pb.top_k(1), 'y': pb.top_k(1)})
        assert out['baseline'] == 'x'
