"""FR-07-A horizon-transfer curves — rho(r^delta, r^Delta) vs the IID null.

Measurement-only (R2C-06b, ships direct, no flags). The Label Horizon
Paradox claim (Song-Liu-Chen, arXiv 2602.03395) is that the optimal
supervision horizon delta* can differ from the trading horizon Delta. The
CHEAP stage-A evidence is mechanical: the correlation between the
forward return over delta bars and the forward return over Delta bars,
both anchored at the same bar. Under IID returns, for delta <= Delta the
delta-window is a prefix of the Delta-window and

    rho_null(delta, Delta) = sqrt(delta / Delta).

Excess rho over that null (persistence) or a deficit (reversal inside the
longer window) is off-diagonal structure worth the FR-07-B probe; a curve
sitting on the null kills the topic cheaply.

Overlap honesty: anchors are strided by the LONGER horizon (Delta) so
sampled label windows never overlap; significance/SEs come from a
weekly-block bootstrap (hourly forward returns share news within a week —
IID SEs would be optimistic). Pure numpy — Mac-testable on synthetic
AR(1)/IID series (tests/test_r2c_measurement_kernels.py); real curves come
from scripts/horizon_transfer_report.py on the harvest Target_Return_*
columns (Jetson).
"""
import numpy as np

WEEK_NS = 7 * 24 * 3600 * 1_000_000_000
MIN_PAIRS = 8  # below this a correlation is noise, not a measurement


def forward_returns(closes, horizon):
    """PERCENT forward return over `horizon` bars anchored at each bar:
    out[i] = (c[i+h]-c[i])/c[i]*100, NaN in the last h slots and wherever
    either close is non-finite/zero. Matches the harvest Target_Return_h
    convention (harvest_*_data.py)."""
    c = np.asarray(closes, dtype=np.float64)
    h = int(horizon)
    n = len(c)
    out = np.full(n, np.nan)
    if h <= 0 or n <= h:
        return out
    c0, c1 = c[:-h], c[h:]
    ok = np.isfinite(c0) & np.isfinite(c1) & (c0 != 0.0)
    out[:-h][ok] = (c1[ok] - c0[ok]) / c0[ok] * 100.0
    return out


def iid_null(delta, big_delta):
    """sqrt(delta/Delta) — corr of a prefix-window sum with the full-window
    sum under IID returns."""
    return float(np.sqrt(float(delta) / float(big_delta)))


def strided_pair(fwd_d, fwd_D, times_ns, stride):
    """Non-overlapping anchor sample for one name: keep every `stride`-th
    bar (greedy from the first jointly-finite anchor) where BOTH forward
    returns are finite. Returns (x, y, t) — delta-returns, Delta-returns,
    anchor times (int64 ns)."""
    x = np.asarray(fwd_d, dtype=np.float64)
    y = np.asarray(fwd_D, dtype=np.float64)
    t = np.asarray(times_ns, dtype=np.int64)
    s = max(1, int(stride))
    finite = np.isfinite(x) & np.isfinite(y)
    keep, last = [], None
    for i in np.flatnonzero(finite):
        if last is None or i - last >= s:
            keep.append(i)
            last = i
    keep = np.asarray(keep, dtype=np.int64)
    return x[keep], y[keep], t[keep]


def _pearson(x, y):
    if len(x) < MIN_PAIRS or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def weekly_block_ids(times_ns):
    """Calendar-week block labels (epoch-ns // one week) — the resampling
    unit for the block bootstrap."""
    return np.asarray(times_ns, dtype=np.int64) // WEEK_NS


def block_bootstrap_se(stat_fn, block_ids, n_boot=200, seed=0):
    """SE of stat_fn under a block bootstrap: resample whole blocks with
    replacement (len(blocks) draws per rep), call stat_fn(row_indices) on
    the concatenated rows, SE = std (ddof=1) over reps that returned a
    finite stat. Returns (se, n_effective_reps); (None, 0) when fewer
    than 2 blocks or <2 usable reps."""
    ids = np.asarray(block_ids)
    uniq = np.unique(ids)
    if len(uniq) < 2:
        return None, 0
    members = {b: np.flatnonzero(ids == b) for b in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[b] for b in draw])
        s = stat_fn(idx)
        if s is not None and np.isfinite(s):
            stats.append(float(s))
    if len(stats) < 2:
        return None, len(stats)
    return float(np.std(stats, ddof=1)), len(stats)


def transfer_stat(x, y, t, n_boot=200, seed=0):
    """{rho, se, n, n_weeks} for one anchor sample (se None when the
    bootstrap cannot run)."""
    rho = _pearson(x, y)
    if rho is None:
        return {'rho': None, 'se': None, 'n': int(len(x)), 'n_weeks': 0}
    blocks = weekly_block_ids(t)
    se, _ = block_bootstrap_se(
        lambda idx: _pearson(x[idx], y[idx]), blocks,
        n_boot=n_boot, seed=seed)
    return {'rho': rho, 'se': se, 'n': int(len(x)),
            'n_weeks': int(len(np.unique(blocks)))}


def transfer_matrix(per_name, horizons=None, n_boot=200, seed=0,
                    stride=None):
    """Per-name + pooled transfer curves.

    per_name: {name: (fwd_by_h, times_ns)} where fwd_by_h is
        {horizon_int: forward-return array} (e.g. the harvest
        Target_Return_* columns) and times_ns the bar clock (int64 ns,
        sorted, same length).
    horizons: horizon ints to pair (default: sorted keys common to all
        names). Every ordered pair delta < Delta is scored.
    stride: anchor stride in bars (default: Delta of the pair — the
        non-overlap guarantee; pass a larger value for extra spacing).

    Returns a list of pair dicts:
        {delta, Delta, null, per_name: {name: stat}, pooled: stat}
    where stat = transfer_stat's dict. Pooled concatenates every name's
    strided anchors (raw, not per-name standardized — documented choice:
    the null sqrt(delta/Delta) is scale-free and per-name vol differences
    only widen the bootstrap SE)."""
    names = sorted(per_name)
    if horizons is None:
        hsets = [set(per_name[n][0]) for n in names]
        horizons = sorted(set.intersection(*hsets)) if hsets else []
    out = []
    for a, d in enumerate(horizons):
        for D in horizons[a + 1:]:
            s = int(stride) if stride is not None else int(D)
            per, xs, ys, ts = {}, [], [], []
            for name in names:
                fwd_by_h, tns = per_name[name]
                x, y, t = strided_pair(fwd_by_h[d], fwd_by_h[D], tns, s)
                per[name] = transfer_stat(x, y, t, n_boot=n_boot,
                                          seed=seed)
                xs.append(x)
                ys.append(y)
                ts.append(t)
            xp = np.concatenate(xs) if xs else np.empty(0)
            yp = np.concatenate(ys) if ys else np.empty(0)
            tp = (np.concatenate(ts) if ts
                  else np.empty(0, dtype=np.int64))
            out.append({'delta': int(d), 'Delta': int(D),
                        'null': iid_null(d, D), 'per_name': per,
                        'pooled': transfer_stat(xp, yp, tp,
                                                n_boot=n_boot,
                                                seed=seed)})
    return out
