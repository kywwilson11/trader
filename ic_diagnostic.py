"""Per-name information-coefficient diagnostic — the universe-promotion gate (wave-9 #3).

Promoting the diversified candidate pool into the LIVE tradable universe is only
+EV for names the model actually predicts at the hourly horizon. The √breadth
math assumes constant IC, which is almost certainly FALSE on megacap defensives
(Avramov-Cheng-Metzker: ML profitability is weakest on liquid large-caps). So the
gate is empirical: promote a name ONLY IF its out-of-sample rank-IC is positive
AND consistent across sub-periods — and KILL the rest (keep them training-only).

rank-IC = Spearman(prediction, forward return). Pure numpy/scipy — Mac-testable
on synthetic names with and without edge; on the Jetson it runs over a
backtest.py prediction frame for the full ticker set.
"""
import numpy as np


def rank_ic(pred, fwd_return):
    """Spearman rank correlation of predictions with realized forward returns,
    or None when there is too little/degenerate data to judge."""
    from scipy.stats import spearmanr
    pred = np.asarray(pred, float)
    fwd = np.asarray(fwd_return, float)
    m = np.isfinite(pred) & np.isfinite(fwd)
    if int(m.sum()) < 5 or np.std(pred[m]) < 1e-12 or np.std(fwd[m]) < 1e-12:
        return None
    rho = spearmanr(pred[m], fwd[m]).correlation
    return float(rho) if np.isfinite(rho) else None


def ic_by_name(rows, name_key='symbol', pred_key='pred', fwd_key='fwd_return',
               n_subperiods=4):
    """Per-name overall rank-IC + IC in each of n_subperiods (time order) +
    positive-consistency (fraction of sub-periods with IC > 0).

    rows: an iterable of dicts, assumed time-ordered PER NAME (the caller sorts).
    Returns {name: {ic, n, subperiod_ics, positive_consistency}}.
    """
    by_name = {}
    for r in rows:
        by_name.setdefault(r[name_key], []).append(r)
    out = {}
    for name, rs in by_name.items():
        pred = [r.get(pred_key) for r in rs]
        fwd = [r.get(fwd_key) for r in rs]
        ic = rank_ic(pred, fwd)
        subs = []
        n = len(rs)
        if n >= n_subperiods * 5:
            edges = np.linspace(0, n, n_subperiods + 1).astype(int)
            for i in range(n_subperiods):
                s = rank_ic(pred[edges[i]:edges[i + 1]], fwd[edges[i]:edges[i + 1]])
                if s is not None:
                    subs.append(s)
        if subs:
            consistency = float(np.mean([s > 0 for s in subs]))
        else:
            consistency = 1.0 if (ic is not None and ic > 0) else 0.0
        out[name] = {'ic': round(ic, 4) if ic is not None else None, 'n': n,
                     'subperiod_ics': [round(s, 4) for s in subs],
                     'positive_consistency': round(consistency, 3)}
    return out


def promote_set(ic_table, min_ic=0.0, min_consistency=0.6, min_t=2.0):
    """Names safe to promote into the LIVE universe.

    Three hurdles, all required: overall IC above min_ic, positive in
    >= min_consistency of sub-periods, AND statistically SIGNIFICANT
    (|IC|*sqrt(n-1) >= min_t). The significance hurdle is the one that matters —
    a small spurious IC over a short history (e.g. 0.07 at n=400, ~1.3 sigma) is
    indistinguishable from zero and must NOT be promoted on a √breadth argument.
    Everything that fails stays training-only. Returns a sorted list.
    """
    out = []
    for name, m in ic_table.items():
        if m['ic'] is None or m['ic'] <= min_ic or m['positive_consistency'] < min_consistency:
            continue
        t = abs(m['ic']) * np.sqrt(max(m['n'] - 1, 1))
        if t >= min_t:
            out.append(name)
    return sorted(out)
