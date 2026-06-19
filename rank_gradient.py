"""Rank-gradient Stage-0 gate for the conviction flagship + edge-Kelly (wave-9 #4/#5).

Concentrating into the top-K and sizing by edge is only +EV if this model's
rank-1-3 forecasts realize MATERIALLY more net return than rank-6-7 on this liquid
universe (Avramov: the gradient may simply be absent here). This computes the
per-rank-bucket net and the go/no-go verdict, and must PASS on BOTH the offline
holdout panel AND the live journals before either lever ships.

`decision_report` already buckets the LIVE side (rank_1_3/4_5/6_7); this adds the
generic verdict and the HOLDOUT-panel computation (from a portfolio_backtest
panel). Pure numpy — Mac-testable.
"""
import numpy as np

DEFAULT_BUCKETS = (('rank_1_3', 1, 3), ('rank_4_5', 4, 5), ('rank_6_7', 6, 7))


def rank_gradient_from_panel(panel, buckets=DEFAULT_BUCKETS, cost_pct=0.0):
    """Per-rank-bucket mean net forward return from an A/B panel (holdout side).

    Each period: sort candidates by signal desc, rank 1..k, drop each into its
    rank bucket (net of a flat per-trade cost if given). Returns
    {bucket: {n, mean_net_pct, hit_rate}} — the same shape decision_report emits.
    """
    collected = {label: [] for label, _, _ in buckets}
    for period in panel:
        ranked = sorted(period, key=lambda c: c.get('signal', 0.0), reverse=True)
        for i, c in enumerate(ranked, start=1):
            net = float(c['fwd_return']) - cost_pct
            for label, lo, hi in buckets:
                if lo <= i <= hi:
                    collected[label].append(net)
                    break
    out = {}
    for label, vals in collected.items():
        if vals:
            arr = np.asarray(vals, float)
            out[label] = {'n': len(arr), 'mean_net_pct': round(float(arr.mean()), 4),
                          'hit_rate': round(float((arr > 0).mean()), 3)}
    return out


def rank_gradient_verdict(buckets, ratio_threshold=0.5):
    """Go/no-go: does a real rank gradient exist? Consumes the bucket dict from
    EITHER decision_report or rank_gradient_from_panel.

    PASS iff rank-6-7 carries materially less edge than rank-1-3: rank_6_7 mean
    net <= 0, OR rank_6_7 mean < ratio_threshold * rank_1_3 mean. If absent, the
    concentration / edge-Kelly levers are regime-mining — ship NEITHER.
    """
    r13 = (buckets.get('rank_1_3') or {}).get('mean_net_pct')
    r67 = (buckets.get('rank_6_7') or {}).get('mean_net_pct')
    if r13 is None or r67 is None:
        return {'gradient_exists': None,
                'verdict': 'insufficient rank coverage — need rank_1_3 and rank_6_7 buckets',
                'rank_1_3': r13, 'rank_6_7': r67}
    exists = (r67 <= 0.0) or (r13 > 0 and r67 < ratio_threshold * r13)
    ratio = round(r67 / r13, 3) if r13 not in (0, None) else None
    return {
        'gradient_exists': bool(exists),
        'rank_1_3': r13, 'rank_6_7': r67, 'ratio_6_7_over_1_3': ratio,
        'verdict': ('rank gradient CONFIRMED — conviction/edge-sizing levers are justified'
                    if exists else
                    'NO rank gradient on this universe — concentration is regime-mining, ship NEITHER'),
    }
