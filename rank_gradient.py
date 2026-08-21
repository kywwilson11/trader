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

# decision_report.py re-declares these boundaries inline (plus an extra
# ('rank_8_plus', 8, 9999)) — the go/no-go compares THIS holdout side against
# that live side, so the two definitions must stay in sync.
DEFAULT_BUCKETS = (('rank_1_3', 1, 3), ('rank_4_5', 4, 5), ('rank_6_7', 6, 7))

# Strict-verdict floor: below this per-bucket n a bucket mean is noise.
# Consumed by rank_gradient_verdict(min_bucket_n=...) and the report
# script's --strict; default calls stay point-estimate (pinned by tests).
MIN_BUCKET_N = 30
_Z90 = 1.6449   # two-sided 90% normal quantile — matches decision_report's ci90 level


def rank_gradient_from_panel(panel, buckets=DEFAULT_BUCKETS, cost_pct=0.0,
                             fwd_bars=1):
    """Per-rank-bucket mean net forward return from an A/B panel (holdout side).

    Each period: sort candidates by signal desc, rank 1..k, drop each into its
    rank bucket (net of a flat per-trade cost if given). Returns
    {bucket: {n, mean_net_pct, hit_rate}} — the same shape decision_report emits.

    fwd_bars declares the panel's forward-return horizon in bars: on a panel
    sampled every bar, adjacent samples share fwd_bars-1 bars of the SAME
    move, so the ci90 standard error uses n_eff = n / fwd_bars (cf.
    portfolio_backtest.compare_deflated). Each bucket now also carries ci90
    [lo, hi] (overlap-adjusted normal 90% CI on the mean) and n_eff — the
    fields rank_gradient_verdict's strict mode consumes.
    """
    collected = {label: [] for label, _, _ in buckets}
    for period in panel:
        # c['signal'] (not .get) so a malformed panel fails loudly, exactly
        # like the c['fwd_return'] below — a silently mid-ranked candidate
        # would feed garbage bucket means into the ship gate.
        ranked = sorted(period, key=lambda c: c['signal'], reverse=True)
        for i, c in enumerate(ranked, start=1):
            net = float(c['fwd_return']) - cost_pct
            for label, lo, hi in buckets:
                if lo <= i <= hi:
                    collected[label].append(net)
                    break
    fb = max(int(fwd_bars), 1)
    out = {}
    for label, vals in collected.items():
        if vals:
            arr = np.asarray(vals, float)
            n = len(arr)
            n_eff = max(n / fb, 1.0)
            m = float(arr.mean())
            stat = {'n': n, 'mean_net_pct': round(m, 4),
                    'hit_rate': round(float((arr > 0).mean()), 3),
                    'n_eff': round(n_eff, 1)}
            if n >= 2:
                half = _Z90 * float(arr.std(ddof=1)) / float(np.sqrt(n_eff))
                stat['ci90'] = [round(m - half, 4), round(m + half, 4)]
            else:
                stat['ci90'] = [round(m, 4), round(m, 4)]   # cf. decision_report n==1
            out[label] = stat
    return out


def rank_gradient_verdict(buckets, ratio_threshold=0.5, min_bucket_n=0,
                          require_ci=False):
    """Go/no-go: does a real rank gradient exist? Consumes the bucket dict from
    EITHER decision_report or rank_gradient_from_panel — including the full
    decision_report.json object (rank buckets nested under 'conviction'), so
    `rank_gradient_report.py --buckets decision_report.json` works as documented.

    PASS iff rank-1-3 actually beats rank-6-7 AND rank-6-7 carries materially
    less edge: rank_1_3 > rank_6_7, and (rank_6_7 mean net <= 0 OR rank_6_7
    mean < ratio_threshold * rank_1_3 mean). Without the direction guard an
    inverted or flat-negative panel (both buckets losing, top bucket worst)
    would read as CONFIRMED. If absent, the concentration / edge-Kelly levers
    are regime-mining — ship NEITHER. ratio_6_7_over_1_3 is only reported when
    rank_1_3 > 0 (negative/negative reads as a healthy gradient when it is
    actually inverted).

    Defaults (min_bucket_n=0, require_ci=False) reproduce the historical
    point-estimate verdict. For any real ship/no-ship decision pass
    min_bucket_n=MIN_BUCKET_N and require_ci=True (the report script's
    --strict): CONFIRMED then additionally requires n >= min_bucket_n in
    BOTH buckets AND the rank_1_3 ci90 lower bound to exceed the rank_6_7
    mean. Both producers emit n; ci90 comes from decision_report's bootstrap
    (live side) or rank_gradient_from_panel's fwd_bars-widened normal CI
    (holdout side). A point-estimate mean over a handful of trades must not
    green-light concentration/edge-Kelly.
    """
    if 'rank_1_3' not in buckets and isinstance(buckets.get('conviction'), dict):
        buckets = buckets['conviction']   # full decision_report.json wrapper
    r13 = (buckets.get('rank_1_3') or {}).get('mean_net_pct')
    r67 = (buckets.get('rank_6_7') or {}).get('mean_net_pct')
    if r13 is None or r67 is None:
        return {'gradient_exists': None,
                'verdict': 'insufficient rank coverage — need rank_1_3 and rank_6_7 buckets',
                'rank_1_3': r13, 'rank_6_7': r67}
    exists = (r13 > r67) and ((r67 <= 0.0)
                              or (r13 > 0 and r67 < ratio_threshold * r13))
    ratio = round(r67 / r13, 3) if r13 > 0 else None
    n13 = (buckets.get('rank_1_3') or {}).get('n')
    n67 = (buckets.get('rank_6_7') or {}).get('n')
    ci13 = (buckets.get('rank_1_3') or {}).get('ci90')
    base = {'rank_1_3': r13, 'rank_6_7': r67, 'ratio_6_7_over_1_3': ratio,
            'n_1_3': n13, 'n_6_7': n67, 'ci90_1_3': ci13,
            'min_bucket_n': int(min_bucket_n), 'require_ci': bool(require_ci)}
    if exists and min_bucket_n:
        bad = {lbl: nn for lbl, nn in (('rank_1_3', n13), ('rank_6_7', n67))
               if not (isinstance(nn, (int, float)) and nn >= min_bucket_n)}
        if bad:
            return dict(base, gradient_exists=None, verdict=(
                f'INSUFFICIENT EVIDENCE — bucket n below '
                f'min_bucket_n={min_bucket_n} ({bad}); a point-estimate '
                f'gradient on this few trades cannot CONFIRM the '
                f'conviction/edge-Kelly levers'))
    if exists and require_ci:
        lo = (ci13[0] if isinstance(ci13, (list, tuple)) and len(ci13) == 2
              and ci13[0] is not None else None)
        if lo is None:
            return dict(base, gradient_exists=None, verdict=(
                'INSUFFICIENT EVIDENCE — rank_1_3 carries no ci90; '
                'regenerate the buckets with a producer that emits it '
                '(decision_report.py or rank_gradient_from_panel)'))
        if not (lo > r67):
            return dict(base, gradient_exists=False, verdict=(
                f'rank gradient NOT ESTABLISHED — rank_1_3 ci90 {ci13} '
                f'includes the rank_6_7 mean {r67}: the apparent gradient '
                f'is within noise — ship NEITHER'))
    return dict(base, gradient_exists=bool(exists), verdict=(
        'rank gradient CONFIRMED — conviction/edge-sizing levers are justified'
        if exists else
        'NO rank gradient on this universe — concentration is regime-mining, '
        'ship NEITHER'))
