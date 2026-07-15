"""Measure whether the LLM gate adds signal BEYOND the ML model.

The system spends API budget and applies 0.65-1.5x sizing (plus forced
veto-sells) based on LLM conviction scores `s`. The question that justifies the
spend is NOT "does s correlate with returns?" — the prompt SHOWS the LLM the ML
`pred`, so a high raw correlation can be the LLM merely ECHOING the model
(redundant), and a low one can hide real orthogonal alpha. The question is
whether s adds value INCREMENTAL to pred.

This evaluator answers that, three ways (wave-8 #3):

  1. PRIMARY — partial Spearman rho(s, realized | pred): rank-correlation of s
     with realized return after partialling OUT the ML pred. The gap between the
     RAW and PARTIAL rho is the echo diagnostic.
  2. The 2x2 {LLM bull/bear} x {ML bull/bear} agree/disagree grid: the
     off-diagonal (disagreement) cells are the only place s can add orthogonal
     value; reported with per-cell n.
  3. SECONDARY/verdict-driver — an encompassing regression
        realized = a + b1*pred + b2*z_s
     with Newey-West HAC standard errors at lag = forward_bars-1 (overlapping
     h-step returns) and a small-sample Student-t p-value. b2 significantly > 0
     is the honest "the LLM earns its place" signal.

The keep/kill verdict gates on the SIGNIFICANCE of b2 (with a sample-size floor),
not a bright-line rho — at typical journal volume a rho of 0.05 is noise.

Usage:
    python llm_eval.py --days 14
    python llm_eval.py --days 30 --asset crypto

The statistics live in compute_incremental_report(), which is pure
(numpy + scipy) and unit-tested on synthetic data without Alpaca.
"""
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

JOURNAL_DIR = BASE_DIR / "journals"
# Single-sourced from trading_utils so this evaluator can never drift from
# the threshold the live loop actually vetoes at (they were two independent
# 0.15 literals before — 2026-07 review).
try:
    from trading_utils import LLM_VETO_THRESHOLD as VETO_THRESHOLD
except Exception:  # standalone use without the trading stack
    VETO_THRESHOLD = 0.15
# Below this many realized samples a partial-rho / coefficient is indistinguishable
# from zero (null std ~0.15-0.19 at n=30-50); abstain rather than flip.
MIN_POWER_N = 60


def _load_entries(days: int) -> list[dict]:
    entries = []
    today = datetime.now().date()
    for d in range(days + 1):
        day = today - timedelta(days=d)
        path = JOURNAL_DIR / f"{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("action") == "llm_analysis":
                    entries.append(e)
    return entries


def _load_advisor_entries(days: int) -> list[dict]:
    """Mirror of _load_entries filtering the advisor-v2 shadow rows
    (action=='llm_advisor_v2') journaled by llm_analyst.py when
    advisor_v2_enabled. Does not touch the 'llm_analysis' rows _load_entries
    reads — the existing run_eval verdict path is unaffected."""
    entries = []
    today = datetime.now().date()
    for d in range(days + 1):
        day = today - timedelta(days=d)
        path = JOURNAL_DIR / f"{day.isoformat()}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("action") == "llm_advisor_v2":
                    entries.append(e)
    return entries


def _bars_lookup(api, symbol: str, asset_type: str, start, end):
    """Hourly closes for [start, end] as a sorted (timestamps, closes)."""
    try:
        if asset_type == 'crypto':
            sym = symbol.replace('-', '/') if '-' in symbol else symbol
            bars = api.get_crypto_bars(sym, '1Hour', start=start.isoformat(),
                                       end=end.isoformat())
        else:
            bars = api.get_bars(symbol, '1Hour', start=start.isoformat(),
                                end=end.isoformat(), adjustment='all')
        ts, closes = [], []
        for b in bars:
            t = b.t
            if hasattr(t, 'to_pydatetime'):
                t = t.to_pydatetime()
            ts.append(t.timestamp())
            closes.append(float(b.c))
        order = np.argsort(ts)
        return np.array(ts)[order], np.array(closes)[order]
    except Exception as e:
        print(f"  [BARS] {symbol}: {e}")
        return np.array([]), np.array([])


def _realized_forward_return(ts_arr, closes, t0: float, horizon_hours: int):
    """Return % from the first bar at/after t0 to ~horizon_hours later."""
    if len(ts_arr) == 0:
        return None
    i0 = int(np.searchsorted(ts_arr, t0))
    if i0 >= len(ts_arr):
        return None
    t_target = ts_arr[i0] + horizon_hours * 3600
    i1 = int(np.searchsorted(ts_arr, t_target))
    if i1 >= len(ts_arr):
        return None  # horizon not yet realized
    if closes[i0] <= 0:
        return None
    return (closes[i1] - closes[i0]) / closes[i0] * 100.0


def realize_scored_rows(rows: list[dict], api=None) -> list[tuple]:
    """Realize forward returns for a list of scored rows.

    rows: each a dict with keys {symbol, asset_type, t0, horizon, s, pred}
    (extra keys are ignored). Returns one (s, realized, pred, t0) tuple per
    input row, in the SAME order (realized is None when the horizon hasn't
    elapsed yet or bars are unavailable — callers filter, same convention as
    compute_incremental_report's own None-filter).

    Groups rows by (symbol, asset_type) so each group needs only ONE
    _bars_lookup call spanning its full [min t0, max t0+horizon] range —
    shares the exact realization machinery run_eval() uses (same
    _bars_lookup / _realized_forward_return) so the offline prompt_ab.py
    harness can never drift from the live scorecard's notion of "realized
    return".

    api=None -> trading_utils.get_api() (Jetson/Alpaca-gated; not needed by
    callers that pass a fake/mock api, e.g. tests).
    """
    if api is None:
        from trading_utils import get_api
        api = get_api()

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['symbol'], r.get('asset_type', 'crypto'))].append((i, r))

    out: list[tuple | None] = [None] * len(rows)
    for (sym, asset), items in groups.items():
        t0s = [r['t0'] for _, r in items]
        max_h = max(int(r.get('horizon', 24) or 24) for _, r in items)
        start = datetime.fromtimestamp(min(t0s)) - timedelta(hours=2)
        end = datetime.fromtimestamp(max(t0s)) + timedelta(hours=max_h + 6)
        ts_arr, closes = _bars_lookup(api, sym, asset, start, end)
        for i, r in items:
            horizon = int(r.get('horizon', 24) or 24)
            realized = _realized_forward_return(ts_arr, closes, r['t0'], horizon)
            out[i] = (r.get('s'), realized, r.get('pred'), r['t0'])
    return out


# --------------------------------------------------------------------------- #
# Pure statistics (no I/O, no Alpaca) — unit-tested in tests/test_llm_eval.py
# --------------------------------------------------------------------------- #

def _avg_rank(x):
    """Average ranks with proper tie handling (scipy.stats.rankdata)."""
    from scipy.stats import rankdata
    return rankdata(np.asarray(x, dtype=float), method='average')


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ols_beta_resid(X, y):
    """OLS via lstsq. Returns (beta, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def _newey_west_se(X, resid, lag):
    """HAC (Newey-West, Bartlett kernel) standard errors for OLS beta.

    Var(beta) = (X'X)^-1 S (X'X)^-1 with the lag-weighted score sandwich
    S = sum u_t u_t' + sum_l w_l (G_l + G_l'), u_t = x_t e_t, w_l = 1 - l/(L+1).
    Rows are assumed time-ordered (the caller sorts by entry time); this corrects
    the serial overlap of h-step returns. Contemporaneous cross-sectional
    dependence (same bar, different symbols) is NOT corrected — a known limit.
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    u = X * resid[:, None]                      # n x k score contributions
    S = u.T @ u                                 # lag 0
    lag = int(max(0, min(lag, n - 1)))
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        G = u[l:].T @ u[:-l]                    # sum_t u_t u_{t-l}'
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    var = np.clip(np.diag(cov), 0.0, None)
    return np.sqrt(var)


def partial_spearman(s, realized, pred):
    """Spearman rho(s, realized) after partialling out pred (rank-residualized).

    Returns (partial_rho, pred_degenerate). If pred has no rank variation we
    cannot control for it, so partial == raw and the flag is True.
    """
    rs, rr, rp = _avg_rank(s), _avg_rank(realized), _avg_rank(pred)
    if rp.std() < 1e-12:
        return _pearson(rs, rr), True
    A = np.column_stack([np.ones_like(rp), rp])
    _, es = _ols_beta_resid(A, rs)
    _, er = _ols_beta_resid(A, rr)
    return _pearson(es, er), False


def two_by_two_grid(s, realized, pred, s_thresh=0.5):
    """{LLM bull/bear} x {ML bull/bear} cells with n and mean realized return.

    Off-diagonal (disagreement) cells are the only place the LLM can add value
    orthogonal to the model; they are also the thinnest, so n is always reported.
    """
    s = np.asarray(s, float); realized = np.asarray(realized, float)
    pred = np.asarray(pred, float)
    llm_bull = s >= s_thresh
    ml_bull = pred > 0.0
    grid = {}
    for name, mask in (
        ('agree_bull', llm_bull & ml_bull),
        ('agree_bear', ~llm_bull & ~ml_bull),
        ('llm_bull_ml_bear', llm_bull & ~ml_bull),
        ('llm_bear_ml_bull', ~llm_bull & ml_bull),
    ):
        n = int(mask.sum())
        grid[name] = {
            'n': n,
            'avg_fwd_ret_pct': round(float(realized[mask].mean()), 3) if n else None,
        }
    return grid


def compute_incremental_report(samples, forward_bars: int = 24,
                               min_n: int = MIN_POWER_N, extra=None) -> dict:
    """Does s add signal incremental to pred? Pure; samples = (s, realized, pred[, t0]).

    samples may be 3-tuples (s, realized, pred) or 4-tuples with a trailing
    timestamp t0; when t0 is present rows are sorted by it so the Newey-West HAC
    is computed in time order.

    extra: optional (len(samples),) or (len(samples), k) array-like of
    additional covariates (e.g. journaled fng_value) — z-scored per column
    (missing/non-finite values mean-imputed before z-scoring, i.e. a neutral
    0 contribution) and appended to the encompassing regression's X. Reused
    unchanged for the advisor-v2 echo-gap question (p_up in the s slot).
    extra=None (the default) is numerically IDENTICAL to before this kwarg
    existed — no new columns, no behavior change.
    """
    idx_rows = [(i, r) for i, r in enumerate(samples)
               if r[0] is not None and r[1] is not None and r[2] is not None
               and np.isfinite(r[0]) and np.isfinite(r[1]) and np.isfinite(r[2])]
    n = len(idx_rows)
    rep = {'n': n, 'min_n': min_n}
    if n < 5:
        rep['verdict'] = 'insufficient_power'
        rep['insufficient_power'] = True
        return rep

    has_ts = len(idx_rows[0][1]) >= 4
    if has_ts:
        idx_rows = sorted(idx_rows,
                          key=lambda ir: (ir[1][3] if ir[1][3] is not None else 0.0))
    orig_idx = [ir[0] for ir in idx_rows]
    rows = [ir[1] for ir in idx_rows]
    s = np.array([r[0] for r in rows], float)
    realized = np.array([r[1] for r in rows], float)
    pred = np.array([r[2] for r in rows], float)

    extra_arr = None
    if extra is not None:
        try:
            extra_full = np.asarray(extra, dtype=float)
            if extra_full.ndim == 1:
                extra_full = extra_full.reshape(-1, 1)
            extra_arr = extra_full[orig_idx]
        except Exception:
            extra_arr = None

    raw = _pearson(_avg_rank(s), _avg_rank(realized))
    partial, pred_degenerate = partial_spearman(s, realized, pred)
    rep['raw_spearman_s_vs_return'] = round(raw, 4) if raw is not None else None
    rep['partial_spearman_s_given_pred'] = round(partial, 4) if partial is not None else None
    if raw is not None and partial is not None:
        rep['echo_gap'] = round(raw - partial, 4)  # raw >> partial => s echoes pred
    rep['pred_degenerate'] = pred_degenerate
    rep['grid'] = two_by_two_grid(s, realized, pred)

    # Encompassing regression realized = a + b1*pred + b2*z_s [+ b_extra*z_extra]
    # (HAC SEs).
    enc = None
    if not pred_degenerate and s.std() > 1e-12 and n >= 6:
        z_s = (s - s.mean()) / s.std()
        cols = [np.ones(n), pred, z_s]
        extra_cols_used = []
        if extra_arr is not None and extra_arr.shape[0] == n:
            for k in range(extra_arr.shape[1]):
                col = extra_arr[:, k]
                finite = np.isfinite(col)
                if finite.sum() < 2:
                    continue
                mean = col[finite].mean()
                col_filled = np.where(finite, col, mean)
                std = col_filled.std()
                if std <= 1e-12:
                    continue
                cols.append((col_filled - mean) / std)
                extra_cols_used.append(k)
        X = np.column_stack(cols)
        beta, resid = _ols_beta_resid(X, realized)
        se = _newey_west_se(X, resid, lag=max(0, forward_bars - 1))
        b2, se2 = float(beta[2]), float(se[2])
        dof = max(1, n - X.shape[1])
        if se2 > 1e-12:
            from scipy.stats import t as _t
            tstat = b2 / se2
            pval = float(2.0 * _t.sf(abs(tstat), dof))
        else:
            tstat, pval = None, None
        enc = {'b2_s': round(b2, 5), 'se_hac': round(se2, 5),
               't': round(tstat, 3) if tstat is not None else None,
               'p_value': round(pval, 4) if pval is not None else None,
               'b1_pred': round(float(beta[1]), 5), 'dof': dof}
        if extra_cols_used:
            b_extra = []
            for j, k in enumerate(extra_cols_used):
                col_idx = 3 + j
                b_extra.append({'col': k, 'beta': round(float(beta[col_idx]), 5),
                               'se_hac': round(float(se[col_idx]), 5)})
            enc['b_extra'] = b_extra
    rep['encompassing'] = enc

    # Verdict — gate on b2 significance + a sample floor, never a bright-line rho.
    if n < min_n:
        rep['verdict'] = (f'insufficient_power (n={n} < {min_n}); collect more '
                          'journals before trusting the incremental estimate')
        rep['insufficient_power'] = True
    elif pred_degenerate:
        rep['verdict'] = 'pred_degenerate — ML pred has no variation; cannot assess increment'
    elif enc and enc['p_value'] is not None and enc['p_value'] < 0.05 and enc['b2_s'] > 0:
        rep['verdict'] = ('LLM adds signal INCREMENTAL to the ML pred '
                          f'(b2={enc["b2_s"]}%/SD, p={enc["p_value"]}) — keep it')
    elif raw is not None and partial is not None and abs(raw) > 0.05 and abs(partial) < 0.5 * abs(raw):
        rep['verdict'] = ('LLM largely ECHOES the ML pred (raw rho '
                          f'{raw:.3f} -> partial {partial:.3f}) — little independent value')
    else:
        rep['verdict'] = ('no measurable incremental value beyond the ML pred at '
                          'this sample — candidate to disable and save the spend')
    return rep


def compute_calibration_report(p_up, realized, conviction=None, abstain=None,
                               n_bins: int = 5, min_n: int = MIN_POWER_N) -> dict:
    """Is the advisor's stated p_up actually calibrated? Pure numpy/scipy —
    Mac-testable on synthetic data, no Alpaca/journals.

    Reliability bins (fixed edges [0,.2,.4,.6,.8,1.001]) of stated p_up vs
    empirical up-frequency, Brier score + skill vs the base rate, a
    calibration slope/intercept (outcome ~ a + b*p_up; overconfidence shows
    as slope < 1 — no frontier model hits nominal coverage in the
    literature, so this is expected, not a bug), plus two OPTIONAL blocks:

      conviction: per-level {n, avg_fwd_ret_pct, hit_rate} + Spearman(
        conviction, realized) — monotonicity check.
      abstain: the abstention-artifact check — abstain is only meaningful
        evidence if active-row IC exceeds abstain-row IC AND abstain
        hit-rate is ~coin-flip (an abstain that is itself predictive would
        mean the model is hiding a real view behind the abstain flag).

    verdict is a CALIBRATION description (well_calibrated / overconfident /
    underconfident / mixed) — never a keep/kill trade verdict; that
    authority stays with compute_incremental_report's b2 gate.
    """
    p_up = np.asarray(p_up, dtype=float)
    realized = np.asarray(realized, dtype=float)
    mask = np.isfinite(p_up) & np.isfinite(realized)
    p = p_up[mask]
    r = realized[mask]
    n = int(mask.sum())
    rep = {'n': n, 'min_n': min_n}
    if n < min_n:
        rep['verdict'] = (f'insufficient_power (n={n} < {min_n}); collect more '
                          'journals before trusting calibration')
        rep['insufficient_power'] = True
        return rep

    outcome = (r > 0).astype(float)

    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]
    bins = []
    gaps = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bmask = (p >= lo) & (p < hi)
        bn = int(bmask.sum())
        if bn == 0:
            bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': 0, 'mean_p': None,
                        'emp_freq': None, 'gap': None})
            continue
        mean_p = float(p[bmask].mean())
        emp_freq = float(outcome[bmask].mean())
        gap = emp_freq - mean_p
        gaps.append(gap)
        bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': bn,
                    'mean_p': round(mean_p, 4), 'emp_freq': round(emp_freq, 4),
                    'gap': round(gap, 4)})
    rep['bins'] = bins

    base_rate = float(outcome.mean())
    brier = float(np.mean((p - outcome) ** 2))
    brier_base = float(np.mean((base_rate - outcome) ** 2))
    rep['base_rate'] = round(base_rate, 4)
    rep['brier'] = round(brier, 5)
    rep['brier_base'] = round(brier_base, 5)
    rep['brier_skill'] = (round(1.0 - brier / brier_base, 4)
                          if brier_base >= 1e-12 else None)

    slope = intercept = None
    if p.std() > 1e-12 and n >= 3:
        X = np.column_stack([np.ones(n), p])
        beta, _resid = _ols_beta_resid(X, outcome)
        intercept, slope = float(beta[0]), float(beta[1])
    rep['calibration_intercept'] = round(intercept, 4) if intercept is not None else None
    rep['calibration_slope'] = round(slope, 4) if slope is not None else None

    mean_abs_gap = float(np.mean(np.abs(gaps))) if gaps else None
    rep['mean_abs_gap'] = round(mean_abs_gap, 4) if mean_abs_gap is not None else None

    if mean_abs_gap is None or slope is None:
        rep['verdict'] = 'insufficient_power'
    elif mean_abs_gap < 0.06 and 0.85 <= slope <= 1.15:
        rep['verdict'] = 'well_calibrated'
    elif slope < 0.85:
        rep['verdict'] = 'overconfident'
    elif slope > 1.15:
        rep['verdict'] = 'underconfident'
    else:
        rep['verdict'] = 'mixed'

    if conviction is not None:
        conv = np.asarray(conviction, dtype=float)
        conv = conv[mask] if conv.shape[0] == mask.shape[0] else conv
        levels = {}
        for lvl in range(1, 6):
            lvl_mask = conv == lvl
            ln = int(lvl_mask.sum())
            if ln == 0:
                continue
            levels[str(lvl)] = {
                'n': ln,
                'avg_fwd_ret_pct': round(float(r[lvl_mask].mean()), 3),
                'hit_rate': round(float((r[lvl_mask] > 0).mean()), 3),
            }
        finite = np.isfinite(conv)
        spearman = (_pearson(_avg_rank(conv[finite]), _avg_rank(r[finite]))
                   if finite.sum() >= 3 else None)
        rep['conviction'] = {
            'levels': levels,
            'spearman_conviction_vs_realized':
                round(spearman, 4) if spearman is not None else None,
        }

    if abstain is not None:
        ab = np.asarray(abstain, dtype=bool)
        ab = ab[mask] if ab.shape[0] == mask.shape[0] else ab
        n_abstain = int(ab.sum())
        n_active = int((~ab).sum())
        block = {'n_abstain': n_abstain,
                'abstain_rate': round(n_abstain / n, 4) if n else None}
        block['hit_rate_abstain'] = (round(float((r[ab] > 0).mean()), 4)
                                     if n_abstain else None)
        block['mean_abs_ret_abstain'] = (round(float(np.abs(r[ab]).mean()), 4)
                                         if n_abstain else None)
        block['hit_rate_active'] = (round(float((r[~ab] > 0).mean()), 4)
                                    if n_active else None)
        block['mean_abs_ret_active'] = (round(float(np.abs(r[~ab]).mean()), 4)
                                        if n_active else None)
        spearman_active = (_pearson(_avg_rank(p[~ab]), _avg_rank(r[~ab]))
                           if n_active >= 3 else None)
        block['spearman_active_only'] = (round(spearman_active, 4)
                                         if spearman_active is not None else None)
        rep['abstain'] = block

    return rep


def run_eval(days: int = 14, asset_filter: str | None = None) -> dict:
    from trading_utils import get_api

    entries = _load_entries(days)
    if asset_filter:
        entries = [e for e in entries if e.get('asset_type') == asset_filter]
    if not entries:
        print("No llm_analysis journal entries found — the bots journal one "
              "per LLM cycle; run again after some trading.")
        return {}

    api = get_api()

    # Group needed bar ranges by (symbol, asset_type)
    needed = defaultdict(list)  # (symbol, asset) -> [(t0, horizon, s, pred)]
    forward_bars = 24
    for e in entries:
        try:
            t0 = datetime.fromisoformat(e['ts']).timestamp()
        except (KeyError, ValueError):
            continue
        horizon = int(e.get('forward_bars', 24) or 24)
        forward_bars = horizon
        for sym, v in (e.get('scores') or {}).items():
            s = v.get('s')
            if s is None:
                continue
            needed[(sym, e.get('asset_type', 'crypto'))].append(
                (t0, horizon, float(s), v.get('pred')))

    samples = []  # (s, realized, pred, t0)
    for (sym, asset), rows in needed.items():
        t0s = [r[0] for r in rows]
        max_h = max(r[1] for r in rows)
        start = datetime.fromtimestamp(min(t0s)) - timedelta(hours=2)
        end = datetime.fromtimestamp(max(t0s)) + timedelta(hours=max_h + 6)
        ts_arr, closes = _bars_lookup(api, sym, asset, start, end)
        for t0, horizon, s, pred in rows:
            realized = _realized_forward_return(ts_arr, closes, t0, horizon)
            if realized is not None:
                samples.append((s, realized, pred, t0))

    if not samples:
        print("No realized samples yet (horizons may not have elapsed).")
        return {}

    s_vals = np.array([x[0] for x in samples])
    realized = np.array([x[1] for x in samples])

    buckets = [(0.0, 0.15, 'VETO  '), (0.15, 0.35, 'bear  '),
               (0.35, 0.50, 'lean- '), (0.50, 0.65, 'lean+ '),
               (0.65, 0.85, 'bull  '), (0.85, 1.01, 'strong')]
    print(f"\n=== LLM GATE EVALUATION ({len(samples)} scored samples, "
          f"last {days}d) ===")
    print(f"{'bucket':<8}{'n':>5}{'avg fwd ret %':>15}{'hit rate':>10}")
    report = {'n': len(samples), 'buckets': {}}
    for lo, hi, label in buckets:
        mask = (s_vals >= lo) & (s_vals < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        avg = float(realized[mask].mean())
        hit = float((realized[mask] > 0).mean())
        report['buckets'][label.strip()] = {'n': n, 'avg_fwd_ret_pct': round(avg, 3),
                                            'hit_rate': round(hit, 3)}
        print(f"{label:<8}{n:>5}{avg:>15.3f}{hit:>10.2f}")

    # Incremental-over-pred: the question that justifies the LLM spend.
    inc = compute_incremental_report(samples, forward_bars=forward_bars)
    report['incremental'] = inc
    raw = inc.get('raw_spearman_s_vs_return')
    par = inc.get('partial_spearman_s_given_pred')
    print(f"\nRaw Spearman (s vs realized):        "
          f"{raw:+.3f}" if raw is not None else "\nRaw Spearman: n/a")
    print(f"Partial Spearman (s | ML pred):      "
          f"{par:+.3f}   <- the honest signal" if par is not None else "Partial Spearman: n/a")
    enc = inc.get('encompassing')
    if enc and enc.get('p_value') is not None:
        print(f"Encompassing b2 (return %/SD of s):  {enc['b2_s']:+.4f} "
              f"(HAC p={enc['p_value']:.3f}, n={inc['n']})")
    g = inc.get('grid', {})
    if g:
        print("LLM-vs-ML disagreement cells (where s can add orthogonal value):")
        for cell in ('llm_bull_ml_bear', 'llm_bear_ml_bull'):
            c = g.get(cell, {})
            print(f"  {cell:<18} n={c.get('n', 0):>4}  avg fwd ret "
                  f"{c.get('avg_fwd_ret_pct')}")

    veto_mask = s_vals < VETO_THRESHOLD
    if veto_mask.any():
        saved = -float(realized[veto_mask].sum())
        report['veto_counterfactual_pct'] = round(saved, 2)
        print(f"Veto counterfactual: blocking s<{VETO_THRESHOLD} avoided "
              f"{saved:+.2f}% cumulative forward return "
              f"({'saved money' if saved > 0 else 'COST money'})")

    report['verdict'] = inc.get('verdict')
    print(f"\nVerdict: {inc.get('verdict')}")

    out = BASE_DIR / 'llm_eval_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")
    return report


def _advisor_event_flag_stats(flag_lists, realized_arr):
    """Per-flag {n, avg_fwd_ret_pct, hit_rate}, cells with n<20 omitted —
    thin conditional cells are worse than no cell (wave-6 lesson)."""
    agg = defaultdict(list)
    for i, flags in enumerate(flag_lists):
        for f in (flags or []):
            agg[f].append(i)
    out = {}
    for f, idxs in agg.items():
        if len(idxs) < 20:
            continue
        rets = realized_arr[idxs]
        out[f] = {'n': len(idxs), 'avg_fwd_ret_pct': round(float(rets.mean()), 3),
                  'hit_rate': round(float((rets > 0).mean()), 3)}
    return out


def advisor_report(days: int = 14, asset_filter: str | None = None) -> dict:
    """Advisor-v2 verdict: echo-gap of p_up vs the ML pred (incremental),
    calibration (reliability/Brier/slope + conviction/abstain checks),
    per-prompt-version drift (prompt-instability detector), and
    per-event-flag conditional stats. Jetson-gated (Alpaca bars) — reuses
    _bars_lookup/_realized_forward_return via realize_scored_rows exactly
    like run_eval, so this can never silently drift from the live
    scorecard's notion of "realized return". Writes llm_advisor_report.json.
    """
    entries = _load_advisor_entries(days)
    if asset_filter:
        entries = [e for e in entries if e.get('asset_type') == asset_filter]
    if not entries:
        print("No llm_advisor_v2 journal entries found — enable "
              "advisor_v2_enabled and run again after some trading.")
        return {}

    rows = []       # for realize_scored_rows: {symbol, asset_type, t0, horizon, s, pred}
    meta = []        # parallel: (p_up, conviction, abstain, fng_value, prompt_version,
                     #            computed_events, llm_event_flags)
    forward_bars = 24
    for e in entries:
        try:
            t0 = datetime.fromisoformat(e['ts']).timestamp()
        except (KeyError, ValueError):
            continue
        horizon = int(e.get('forward_bars', 24) or 24)
        forward_bars = horizon
        for sym, v in (e.get('scores') or {}).items():
            s = v.get('s')
            if s is None:
                continue
            rows.append({'symbol': sym, 'asset_type': e.get('asset_type', 'crypto'),
                        't0': t0, 'horizon': horizon, 's': float(s),
                        'pred': v.get('pred')})
            meta.append((v.get('p_up'), v.get('conviction'), v.get('abstain'),
                        e.get('fng_value'), e.get('prompt_version'),
                        v.get('computed_events') or [], v.get('event_flags') or []))

    if not rows:
        print("No advisor rows had a scored symbol — nothing to realize.")
        return {}

    realized_tuples = realize_scored_rows(rows)  # (s, realized, pred, t0) per row, aligned

    samples, s_list, p_up_list, conv_list, abstain_list, fng_list = (
        [], [], [], [], [], [])
    prompt_versions, computed_events_list, llm_flags_list = [], [], []
    for rt, (p_up, conviction, abstain, fng_value, prompt_version,
            computed_events, llm_flags) in zip(realized_tuples, meta):
        if rt is None or rt[1] is None:
            continue
        s, realized, pred, t0 = rt
        signal = p_up if p_up is not None else s
        samples.append((signal, realized, pred, t0))
        s_list.append(s)
        p_up_list.append(p_up if p_up is not None else np.nan)
        conv_list.append(conviction if conviction is not None else np.nan)
        abstain_list.append(bool(abstain) if abstain is not None else False)
        fng_list.append(fng_value if fng_value is not None else np.nan)
        prompt_versions.append(prompt_version)
        computed_events_list.append(computed_events)
        llm_flags_list.append(llm_flags)

    if not samples:
        print("No realized advisor samples yet (horizons may not have elapsed).")
        return {}

    n = len(samples)
    fng_arr = np.array(fng_list, dtype=float)
    fng_present_frac = float(np.isfinite(fng_arr).mean()) if n else 0.0
    extra = fng_arr if fng_present_frac >= 0.8 else None

    # Echo-gap of p_up (falls back to s when p_up wasn't emitted) vs the ML
    # pred — reuses the same encompassing-regression machinery as run_eval.
    incremental = compute_incremental_report(samples, forward_bars=forward_bars,
                                             extra=extra)

    realized_arr = np.array([x[1] for x in samples], dtype=float)
    p_up_arr = np.array(p_up_list, dtype=float)
    conv_arr = np.array(conv_list, dtype=float)
    abstain_arr = np.array(abstain_list, dtype=bool)
    calibration = compute_calibration_report(p_up_arr, realized_arr,
                                             conviction=conv_arr,
                                             abstain=abstain_arr)

    # samples[i][0] is the SIGNAL (p_up when present, else s) — the raw
    # conviction score for the prompt-version/veto diagnostics below is the
    # separately-tracked s_list.
    s_arr = np.array(s_list, dtype=float)

    by_version = {}
    pv_arr = np.array(prompt_versions, dtype=object)
    for version in sorted({v for v in prompt_versions if v}):
        vmask = (pv_arr == version)
        vn = int(vmask.sum())
        if vn == 0:
            continue
        p_up_v = p_up_arr[vmask]
        finite_p = p_up_v[np.isfinite(p_up_v)]
        by_version[version] = {
            'n': vn,
            'mean_s': round(float(s_arr[vmask].mean()), 4),
            'std_s': round(float(s_arr[vmask].std()), 4),
            'veto_rate': round(float((s_arr[vmask] < VETO_THRESHOLD).mean()), 4),
            'abstain_rate': round(float(abstain_arr[vmask].mean()), 4),
            'mean_p_up': round(float(finite_p.mean()), 4) if len(finite_p) else None,
        }

    event_conditional = {
        'computed_events': _advisor_event_flag_stats(computed_events_list, realized_arr),
        'llm_event_flags': _advisor_event_flag_stats(llm_flags_list, realized_arr),
    }

    veto_mask = s_arr < VETO_THRESHOLD
    veto_counterfactual = (round(-float(realized_arr[veto_mask].sum()), 2)
                           if veto_mask.any() else None)

    finite_p_mask = np.isfinite(p_up_arr)
    s_vs_p_up = (_pearson(_avg_rank(s_arr[finite_p_mask]),
                          _avg_rank(p_up_arr[finite_p_mask]))
                if finite_p_mask.sum() >= 5 else None)

    report = {
        'n': n,
        'incremental': incremental,
        'calibration': calibration,
        'by_prompt_version': by_version,
        'event_conditional': event_conditional,
        'veto_counterfactual_pct': veto_counterfactual,
        's_vs_p_up_spearman': round(s_vs_p_up, 4) if s_vs_p_up is not None else None,
    }

    print(f"\n=== LLM ADVISOR v2 EVALUATION ({n} scored samples, last {days}d) ===")
    print(f"Incremental verdict (p_up vs pred): {incremental.get('verdict')}")
    print(f"Calibration verdict: {calibration.get('verdict')}")
    if s_vs_p_up is not None:
        note = ' <- p_up looks redundant with s' if abs(s_vs_p_up) > 0.9 else ''
        print(f"s vs p_up Spearman: {s_vs_p_up:+.3f}{note}")
    for version, stats in by_version.items():
        print(f"  [{version}] n={stats['n']} mean_s={stats['mean_s']:.3f} "
              f"veto_rate={stats['veto_rate']:.3f} "
              f"abstain_rate={stats['abstain_rate']:.3f}")
    if veto_counterfactual is not None:
        print(f"Veto counterfactual: {veto_counterfactual:+.2f}% cumulative "
              f"forward return "
              f"({'saved money' if veto_counterfactual > 0 else 'COST money'})")

    out = BASE_DIR / 'llm_advisor_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report: {out}")
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate LLM gate vs realized returns')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--asset', choices=['crypto', 'stock'], default=None)
    ap.add_argument('--advisor', action='store_true',
                    help='Run the advisor-v2 calibration/incremental report '
                         '(llm_advisor_v2 shadow rows) instead of the base '
                         'llm_analysis eval')
    args = ap.parse_args()
    if args.advisor:
        advisor_report(args.days, args.asset)
    else:
        run_eval(args.days, args.asset)
