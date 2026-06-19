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
                               min_n: int = MIN_POWER_N) -> dict:
    """Does s add signal incremental to pred? Pure; samples = (s, realized, pred[, t0]).

    samples may be 3-tuples (s, realized, pred) or 4-tuples with a trailing
    timestamp t0; when t0 is present rows are sorted by it so the Newey-West HAC
    is computed in time order.
    """
    rows = [r for r in samples
            if r[0] is not None and r[1] is not None and r[2] is not None
            and np.isfinite(r[0]) and np.isfinite(r[1]) and np.isfinite(r[2])]
    n = len(rows)
    rep = {'n': n, 'min_n': min_n}
    if n < 5:
        rep['verdict'] = 'insufficient_power'
        rep['insufficient_power'] = True
        return rep

    has_ts = len(rows[0]) >= 4
    if has_ts:
        rows = sorted(rows, key=lambda r: (r[3] if r[3] is not None else 0.0))
    s = np.array([r[0] for r in rows], float)
    realized = np.array([r[1] for r in rows], float)
    pred = np.array([r[2] for r in rows], float)

    raw = _pearson(_avg_rank(s), _avg_rank(realized))
    partial, pred_degenerate = partial_spearman(s, realized, pred)
    rep['raw_spearman_s_vs_return'] = round(raw, 4) if raw is not None else None
    rep['partial_spearman_s_given_pred'] = round(partial, 4) if partial is not None else None
    if raw is not None and partial is not None:
        rep['echo_gap'] = round(raw - partial, 4)  # raw >> partial => s echoes pred
    rep['pred_degenerate'] = pred_degenerate
    rep['grid'] = two_by_two_grid(s, realized, pred)

    # Encompassing regression realized = a + b1*pred + b2*z_s (HAC SEs).
    enc = None
    if not pred_degenerate and s.std() > 1e-12 and n >= 6:
        z_s = (s - s.mean()) / s.std()
        X = np.column_stack([np.ones(n), pred, z_s])
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Evaluate LLM gate vs realized returns')
    ap.add_argument('--days', type=int, default=14)
    ap.add_argument('--asset', choices=['crypto', 'stock'], default=None)
    args = ap.parse_args()
    run_eval(args.days, args.asset)
