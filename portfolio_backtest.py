"""Cross-sectional A/B policy engine (wave-5 Tier1-2).

The conviction flagship — "make fewer, higher-conviction trades" via a
conviction-gated dynamic top-K admission — cannot be promoted on priors. The
shadow.py Diebold-Mariano gate tests FORECAST errors, which is the wrong
instrument for an ADMISSION-POLICY change (a verified category error in the
wave-5 kill-list). This module is the right instrument: a cross-sectional
portfolio backtest that replays two admission policies over the SAME ranked
panel and scores the difference net of turnover cost.

A "panel" is a time-ordered list of periods; each period is a list of candidate
dicts {symbol, signal, fwd_return, ...optional conviction fields}. A "policy"
is a pure callable(candidates_sorted_by_signal_desc) -> admitted subset. The
engine forms an equal-weight portfolio of the admitted names each period,
charges a round-trip cost on every newly-entered name (amortized once over its
hold), and reports return / Sharpe / turnover / admitted-k.

Everything is pure numpy — no model, no I/O, no look-ahead (a period only ever
sees its own candidates). Unit-tested on synthetic panels.
"""

import numpy as np

DEFAULT_PERIODS_PER_YEAR = 1638.0   # stock RTH hourly bars/yr (compute_sharpe)


# ---------------------------------------------------------------------------
# Admission policies (pure callables over one period's candidates)
# ---------------------------------------------------------------------------

def top_k(k):
    """Fixed top-K by signal — the incumbent admission policy."""
    def policy(cands):
        return cands[:max(int(k), 0)]
    return policy


def conviction_gated(k_max, signal_floor=None, meta_floor=None,
                     ratio_floor=None):
    """Dynamic K in [0, k_max]: admit a top-K name ONLY if it clears the
    conviction floors. K floats down when few names are convincing (the
    "fewer, higher-conviction" thesis) — the policy under test.

    signal_floor: minimum signal; meta_floor: minimum meta_p; ratio_floor:
    minimum pred/threshold ratio. Any floor left None is not applied.
    """
    def policy(cands):
        out = []
        for c in cands[:max(int(k_max), 0)]:
            if signal_floor is not None and c.get('signal', 0) < signal_floor:
                continue
            if meta_floor is not None and c.get('meta_p', 1.0) < meta_floor:
                continue
            if ratio_floor is not None and c.get('pred_thresh_ratio', 1.0) < ratio_floor:
                continue
            out.append(c)
        return out
    return policy


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _sorted_desc(period):
    return sorted(period, key=lambda c: c.get('signal', 0.0), reverse=True)


def run_policy(panel, policy, cost_pct=0.0,
               periods_per_year=DEFAULT_PERIODS_PER_YEAR):
    """Replay one admission policy over the panel. Returns a metrics dict.

    Per period: admit = policy(sorted candidates); gross = equal-weight mean of
    admitted fwd_return; cost = cost_pct * (fraction of the book newly entered)
    — a round-trip charged once at entry, so a name held across periods is not
    re-charged. net = gross - cost. Cash (empty admit) earns 0.
    """
    prev = set()
    nets, grosses, ks, entries_frac = [], [], [], []
    n_entries_total = n_exits_total = 0
    for period in panel:
        admitted = policy(_sorted_desc(period))
        syms = [c['symbol'] for c in admitted]
        k = len(syms)
        ks.append(k)
        cur = set(syms)
        new = cur - prev
        gone = prev - cur
        n_entries_total += len(new)
        n_exits_total += len(gone)
        if k > 0:
            gross = float(np.mean([c['fwd_return'] for c in admitted]))
            ef = len(new) / k                       # fraction of book entered
        else:
            gross, ef = 0.0, 0.0                    # all cash
        entries_frac.append(ef)
        grosses.append(gross)
        nets.append(gross - cost_pct * ef)
        prev = cur

    nets = np.asarray(nets, dtype=float)
    n = len(nets)
    sharpe = 0.0
    if n > 1 and nets.std() > 1e-12:
        sharpe = float(nets.mean() / nets.std() * np.sqrt(periods_per_year))
    return {
        'n_periods': n,
        'mean_admitted_k': round(float(np.mean(ks)), 3) if n else 0.0,
        'pct_periods_cash': round(float(np.mean(np.asarray(ks) == 0)), 4) if n else 0.0,
        'gross_total': round(float(np.sum(grosses)), 4),
        'net_total': round(float(nets.sum()), 4),
        'sharpe': round(sharpe, 4),
        'avg_entry_fraction': round(float(np.mean(entries_frac)), 4) if n else 0.0,
        'entries': n_entries_total,
        'exits': n_exits_total,
        'hit_rate': round(float(np.mean(nets > 0)), 4) if n else 0.0,
    }


def compare(panel, policies, cost_pct=0.0,
            periods_per_year=DEFAULT_PERIODS_PER_YEAR, baseline=None):
    """Run several named policies over the same panel and report deltas.

    policies: {name: policy_callable}. baseline: the name to difference others
    against (default: the first key). Returns {'results': {name: metrics},
    'deltas': {name: {sharpe_delta, net_delta, k_delta}}, 'baseline': name}.
    """
    names = list(policies)
    base = baseline if baseline in policies else names[0]
    results = {name: run_policy(panel, policies[name], cost_pct, periods_per_year)
               for name in names}
    b = results[base]
    deltas = {}
    for name in names:
        r = results[name]
        deltas[name] = {
            'sharpe_delta': round(r['sharpe'] - b['sharpe'], 4),
            'net_delta': round(r['net_total'] - b['net_total'], 4),
            'k_delta': round(r['mean_admitted_k'] - b['mean_admitted_k'], 3),
        }
    return {'results': results, 'deltas': deltas, 'baseline': base}
