"""Cross-book (account-level) stop-risk cap + a two-book equity simulator.

Both trading loops size positions off the SAME account.equity, and the
equicorrelation ENB cap (MAX_BOOK_RISK_PCT) is enforced PER BOOK. So the
stock book can run to 2.5% single-factor stop-risk while the crypto book runs
to another 2.5% — ~5% combined — even though the books are NOT independent:
the stock book holds crypto-proxies (COIN/MSTR/MARA) while the crypto book
holds spot BTC/ETH, i.e. the SAME factor. Naive per-book diversification
overstates the protection.

This module nets the two books with one extra cross-book correlation term and
caps the COMBINED single-factor stop-risk (ACCOUNT_RISK_CAP ~ 3%). It is a
CONSTRAINT, not a forecast — low overfit risk — but it only ever shrinks or
blocks a marginal entry; it never enlarges one (fail-open: any missing input
falls back to today's per-book behavior).

Model. Each book's diversified stop-risk under equicorrelation is
    R_b = sqrt((1-rho_b)*sum(r_i^2) + rho_b*(sum r_i)^2)
(portfolio.diversified_book_risk). Treating each book as a super-position with
risk R_b and a cross-book correlation rho_x, the account stop-risk is the
two-asset portfolio formula
    R_acct = sqrt(R_s^2 + R_c^2 + 2*rho_x*R_s*R_c).
R_acct lies between |R_s - R_c| (rho_x=-1) and R_s + R_c (rho_x=+1), and
equals the per-book sum exactly when the books move in lockstep — which is the
regime the crypto-proxy overlap creates in a risk-off shock.

The GATE-2 promotion check (does the cap actually reduce max-drawdown without
degrading DSR?) needs a concurrent-equity backtest the repo lacks; the
two-book simulator below supplies it. Everything here is pure-numpy and
PIT-clean (uses trailing realized correlations only).
"""

import fcntl
import json
import os
import time

import numpy as np

from log_config import get_logger
from portfolio import diversified_book_risk

logger = get_logger(__name__)

# Combined single-factor stop-risk ceiling across BOTH books (fraction of
# equity). Below the per-book sum (2*0.025) on purpose — that headroom was the
# double-count. Tunable; lives here so strategy_config can re-export it.
ACCOUNT_RISK_CAP = 0.03


def account_stop_risk(stock_risks, crypto_risks, rho_stock, rho_crypto,
                      rho_cross):
    """Combined single-factor stop-risk of both books (fraction of equity).

    stock_risks / crypto_risks: per-position stop-risks (fractions of equity)
        in each book. rho_* clipped to [0,1] (a book's avg |corr|); rho_cross
        clipped to [-1,1] (signed — books can hedge).
    """
    r_s = diversified_book_risk(stock_risks, rho_stock)
    r_c = diversified_book_risk(crypto_risks, rho_crypto)
    rx = min(max(float(rho_cross), -1.0), 1.0)
    return float(np.sqrt(max(r_s ** 2 + r_c ** 2 + 2.0 * rx * r_s * r_c, 0.0)))


def account_risk_budget(candidate_book, stock_risks, crypto_risks,
                        rho_stock, rho_crypto, rho_cross,
                        cap=ACCOUNT_RISK_CAP, max_risk=None):
    """Max stop-risk (fraction of equity) a NEW position may add to
    `candidate_book` ('stock'|'crypto') before the ACCOUNT cap binds.

    account_stop_risk is strictly increasing in the candidate's risk for
    rho_cross >= 0 but U-shaped for rho_cross < 0 (added hedge risk first NETS
    the account down before re-raising it). Bisection stays valid either way:
    when acct(0) < cap < acct(hi) the U-shape has exactly one rising
    cap-crossing, and the return-hi branch is safe because a U-shape's max sits
    at an endpoint. The acct(0) >= cap early-exit deliberately returns 0 even
    when a hedging candidate would REDUCE account risk below the cap — it only
    ever blocks, never enlarges; do not hand out hedge budgets without the
    promotion path. Returns 0.0 when the cap is already exhausted. `max_risk`
    bounds the search (default: the per-trade risk budget headroom, capped at
    the account cap itself).
    """
    hi = float(max_risk) if max_risk is not None else float(cap)
    if hi <= 0:
        return 0.0
    base = (list(stock_risks) if candidate_book == 'stock'
            else list(crypto_risks))
    other = (crypto_risks if candidate_book == 'stock' else stock_risks)
    rho_b = rho_stock if candidate_book == 'stock' else rho_crypto
    rho_o = rho_crypto if candidate_book == 'stock' else rho_stock

    def acct_with(rc):
        b = base + [rc]
        if candidate_book == 'stock':
            return account_stop_risk(b, other, rho_b, rho_o, rho_cross)
        return account_stop_risk(other, b, rho_o, rho_b, rho_cross)

    if acct_with(0.0) >= cap:
        return 0.0
    if acct_with(hi) <= cap:
        return hi
    lo, hi2 = 0.0, hi
    for _ in range(40):  # ~1e-12 precision; trivial CPU per candidate
        mid = 0.5 * (lo + hi2)
        if acct_with(mid) <= cap:
            lo = mid
        else:
            hi2 = mid
    return lo


def allocate_book_caps(vol_stock, vol_crypto, total_cap=ACCOUNT_RISK_CAP,
                       clamp=(0.25, 0.75)):
    """Split the account cap into per-book sub-caps by INVERSE realized vol.

    The lower-vol book gets the larger share; each share is clamped to
    [clamp_lo, clamp_hi] of total_cap so neither book is ever starved or
    handed the whole budget. Returns (cap_stock, cap_crypto). Fail-open to an
    even split when a vol is missing/zero.
    """
    vs = float(vol_stock) if vol_stock and vol_stock > 0 else None
    vc = float(vol_crypto) if vol_crypto and vol_crypto > 0 else None
    if vs is None or vc is None:
        w_s = 0.5
    else:
        inv_s, inv_c = 1.0 / vs, 1.0 / vc
        w_s = inv_s / (inv_s + inv_c)
    lo, hi = clamp
    w_s = min(max(w_s, lo), hi)
    return total_cap * w_s, total_cap * (1.0 - w_s)


def scale_for_account_cap(candidate_risk, candidate_book, stock_risks,
                          crypto_risks, rho_stock, rho_crypto, rho_cross,
                          cap=ACCOUNT_RISK_CAP):
    """Multiplier in [0,1] to bring a candidate within the account cap.

    Returns (scale, budget): `scale` = min(1, budget/candidate_risk) and the
    raw `budget`. scale<1 means the account cap binds and the position should
    be shrunk; scale==0 means block. Fail-open (scale=1) on a non-finite or
    non-positive candidate_risk.
    """
    if not np.isfinite(candidate_risk) or candidate_risk <= 0:
        return 1.0, float('inf')
    budget = account_risk_budget(candidate_book, stock_risks, crypto_risks,
                                 rho_stock, rho_crypto, rho_cross, cap=cap,
                                 max_risk=candidate_risk)
    return min(1.0, budget / candidate_risk), budget


# ---------------------------------------------------------------------------
# Two-book concurrent-equity simulator (the missing GATE-2 backtest)
# ---------------------------------------------------------------------------

def simulate_two_books(stock_trades, crypto_trades, periods=None):
    """Replay both books on ONE shared equity timeline.

    Each trade is a dict with at least:
        exit_period: int index into a common period grid (e.g. day number);
        net_pct:     realized net return of the trade, in PERCENT;
        weight:      fraction of equity allocated to the trade (default 1.0).
    Realized P&L lands on the bucket of its exit_period (mark-on-close), so two
    books drawing down in the same period compound into the SAME equity curve —
    which is exactly the cross-book concentration the per-book caps miss.

    Returns a dict with the combined and per-book equity curves (cumulative
    return %), max drawdown %, and 'combined_sharpe' — mean/std * sqrt(n)
    over the period grid, i.e. a sqrt(N)-scaled t-statistic rather than a
    per-period Sharpe: comparable only across SAME-length windows (fine for
    the capped-vs-uncapped A/B this exists for, where it is monotone-
    equivalent). `periods` overrides the grid length (default =
    max exit_period + 1); trades whose exit_period falls outside the grid are
    excluded from the P&L and counted in 'n_dropped_trades'.
    """
    def _bucket(trades, n):
        pnl = np.zeros(n, dtype=float)
        for t in trades:
            p = int(t['exit_period'])
            if 0 <= p < n:
                pnl[p] += float(t['net_pct']) * float(t.get('weight', 1.0))
        return pnl

    all_trades = list(stock_trades) + list(crypto_trades)
    n = int(periods) if periods is not None else (
        max(int(t['exit_period']) for t in all_trades) + 1 if all_trades else 0)
    if not all_trades or n <= 0:
        return {'n_periods': 0, 'combined_max_drawdown_pct': 0.0,
                'combined_sharpe': 0.0, 'combined_total_pct': 0.0,
                'n_dropped_trades': len(all_trades)}
    n_dropped = sum(1 for t in all_trades
                    if not (0 <= int(t['exit_period']) < n))

    s_pnl = _bucket(stock_trades, n)
    c_pnl = _bucket(crypto_trades, n)
    combined = s_pnl + c_pnl

    def _curve_stats(pnl):
        equity = np.cumsum(pnl)                       # cumulative return %
        running_max = np.maximum.accumulate(
            np.concatenate([[0.0], equity]))[1:]
        max_dd = float(np.max(running_max - equity)) if equity.size else 0.0
        sharpe = 0.0
        if pnl.std() > 1e-9:
            sharpe = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))
        return equity, max_dd, sharpe

    s_eq, s_dd, _ = _curve_stats(s_pnl)
    c_eq, c_dd, _ = _curve_stats(c_pnl)
    comb_eq, comb_dd, comb_sharpe = _curve_stats(combined)

    return {
        'n_periods': int(n),
        'n_dropped_trades': int(n_dropped),
        'combined_total_pct': round(float(comb_eq[-1]), 4),
        'combined_max_drawdown_pct': round(comb_dd, 4),
        'combined_sharpe': round(comb_sharpe, 4),
        'stock_max_drawdown_pct': round(s_dd, 4),
        'crypto_max_drawdown_pct': round(c_dd, 4),
        # Diversification check: if the books were independent the combined
        # drawdown would be < the sum of per-book drawdowns. A combined DD
        # approaching the sum is the concentration the account cap targets.
        'drawdown_concentration': round(
            comb_dd / max(s_dd + c_dd, 1e-9), 4),
        'realized_cross_corr': (round(float(np.corrcoef(s_pnl, c_pnl)[0, 1]), 4)
                                if s_pnl.std() > 1e-9 and c_pnl.std() > 1e-9
                                else None),
    }


# ---------------------------------------------------------------------------
# GATE-1: cross-book stop-risk registry + measurement (wave-8 #7)
# ---------------------------------------------------------------------------
# The live clamp needs both books' current risk, but the loops run as separate
# processes. Each writes its own diversified stop-risk to a shared registry and
# reads the other's; the report below is JOURNALED ONLY (no trading decision),
# so we can measure whether the per-book ENB caps actually let the two books
# stack toward the account cap before wiring the clamp (the model-facing step).

ACCOUNT_RISK_REGISTRY = 'account_risk_registry.json'


def read_registry(path=ACCOUNT_RISK_REGISTRY):
    """Load the shared per-book stop-risk registry; {} if absent/corrupt."""
    try:
        with open(path) as f:
            reg = json.load(f)
    except (ValueError, OSError):
        # ValueError covers json.JSONDecodeError AND the UnicodeDecodeError a
        # binary-garbage file raises (power loss mid-write); OSError covers
        # FileNotFoundError et al.
        return {}
    # Valid-but-non-dict JSON ('[]', 'null') would crash write_book_risk's
    # upsert — and, being parseable, would never self-heal. Treat as corrupt.
    return reg if isinstance(reg, dict) else {}


_write_warned = False  # once-per-process: registry writes are per-cycle noise


def write_book_risk(book, book_risk, rho_book, path=ACCOUNT_RISK_REGISTRY, now=None):
    """Atomically upsert one book's diversified stop-risk into the registry.

    Both books write this file (two processes, or two threads under
    --combined-bots), so the read-modify-write is serialized with an flock on
    a sidecar lockfile ('{path}.lock' — never replaced, so the lock target is
    stable across os.replace) and staged through a per-writer tmp name, else
    interleavings could drop the other book's entry, rename each other's tmp
    away, or expose torn JSON to a concurrent reader.

    Returns the updated registry. Fail-open: a write error is swallowed
    (the measurement must never break the trading loop) but logged once per
    process so a stale GATE-1 journal is diagnosable.
    """
    global _write_warned
    now = time.time() if now is None else float(now)
    entry = {'risk': float(book_risk), 'rho': float(rho_book), 'ts': now}
    reg = read_registry(path)
    reg[str(book)] = entry
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(f'{path}.lock', 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            reg = read_registry(path)  # re-read under the lock: pick up the
            reg[str(book)] = entry     # other book's concurrent write
            with open(tmp, 'w') as f:
                json.dump(reg, f)
            os.replace(tmp, path)
    except OSError as e:
        if not _write_warned:
            _write_warned = True
            logger.warning("[ACCT-RISK] registry write failed (%s) — GATE-1 "
                           "measurement goes stale until writes recover", e)
        try:
            os.unlink(tmp)             # don't leave partial tmp on disk-full
        except OSError:
            pass
    return reg


def account_risk_gate1_report(registry, rho_cross=1.0, cap=ACCOUNT_RISK_CAP,
                              stale_after_s=900.0, now=None):
    """Combined cross-book stop-risk from the registry — measurement only.

    Each book stored its already-diversified risk R_b; combine via the two-
    super-position formula at rho_cross. Entries older than stale_after_s are
    dropped (book treated as down). Returns the combined risk, the naive per-book
    sum, the cap, and the over-cap flag — the evidence that justifies (or kills)
    the live clamp. Takes NO trading action.
    """
    now = time.time() if now is None else float(now)
    fresh = {}
    for b, e in (registry or {}).items():
        if not isinstance(e, dict):
            continue
        try:
            # Coerce defensively: a hand-edited/corrupt registry can hold
            # string or None values; treat the entry as missing (-> stale).
            risk = float(e.get('risk'))
            ts = float(e.get('ts', 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(risk):
            continue
        if (now - ts) <= stale_after_s:
            fresh[b] = risk
    r_s = fresh.get('stock', 0.0)
    r_c = fresh.get('crypto', 0.0)
    account = account_stop_risk([r_s], [r_c], 0.0, 0.0, rho_cross)
    book_sum = r_s + r_c
    return {
        'account_stop_risk': round(account, 5),
        'book_sum': round(book_sum, 5),
        'stock_risk': round(r_s, 5),
        'crypto_risk': round(r_c, 5),
        'cap': float(cap),
        'over_cap': bool(account > cap),
        'headroom': round(cap - account, 5),
        'concentration': round(account / book_sum, 4) if book_sum > 1e-9 else None,
        'stale_books': sorted(set((registry or {})) - set(fresh)),
        'rho_cross': float(min(max(rho_cross, -1.0), 1.0)),
    }


def record_book_risk_and_report(book, book_risks, rho_book, rho_cross=1.0,
                                cap=ACCOUNT_RISK_CAP, path=ACCOUNT_RISK_REGISTRY,
                                now=None):
    """Write this book's diversified stop-risk to the registry, return the
    combined GATE-1 report. One call for the per-cycle measurement hook."""
    r_b = diversified_book_risk(list(book_risks), rho_book) if len(book_risks) else 0.0
    reg = write_book_risk(book, r_b, rho_book, path=path, now=now)
    return account_risk_gate1_report(reg, rho_cross=rho_cross, cap=cap, now=now)
