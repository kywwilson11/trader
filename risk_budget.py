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
import threading
import time

import numpy as np

from log_config import get_logger
from portfolio import diversified_book_risk

logger = get_logger(__name__)

# Combined single-factor stop-risk ceiling across BOTH books (fraction of
# equity). Below the per-book sum (2*0.025) on purpose — that headroom was the
# double-count. Tunable; defined ONLY here — strategy_config does NOT re-export it.
ACCOUNT_RISK_CAP = 0.03

# Books the GATE-1 report accounts for. Any other registry key is surfaced in
# the report's 'unknown_books' (fresh) or 'stale_books' (stale) — never
# silently aggregated.
EXPECTED_BOOKS = ('stock', 'crypto')

# Freshness window for registry entries (seconds). The writer cadence is
# base_loop's every-10th-cycle hook (~5-10 min incl. thermal throttle), so any
# consumer staleness window must stay comfortably above ~2 cadences. gui.py's
# risk gauge currently hardcodes its own 600s — one file, keep both in mind.
REGISTRY_STALE_AFTER_S = 900.0

# Tolerance for entries stamped slightly in the FUTURE (two processes + NTP
# steps on an RTC-less Jetson). Beyond it an entry is treated as unusable.
CLOCK_SKEW_TOL_S = 60.0


def _clip_rho_cross(rho_cross):
    """Clip rho_cross to [-1, 1]. Python min/max do NOT clip NaN, so a
    non-finite value maps to +1.0 — the lockstep worst case (conservative:
    the constraint may only shrink or block, never enlarge)."""
    v = float(rho_cross)
    if not np.isfinite(v):
        return 1.0
    return min(max(v, -1.0), 1.0)


def account_stop_risk(stock_risks, crypto_risks, rho_stock, rho_crypto,
                      rho_cross):
    """Combined single-factor stop-risk of both books (fraction of equity).

    stock_risks / crypto_risks: per-position stop-risks (fractions of equity)
        in each book. rho_* clipped to [0,1] (a book's avg |corr|); rho_cross
        clipped to [-1,1] (signed — books can hedge).
    """
    r_s = diversified_book_risk(stock_risks, rho_stock)
    r_c = diversified_book_risk(crypto_risks, rho_crypto)
    rx = _clip_rho_cross(rho_cross)
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
    bounds the search (default: `cap`). Raises ValueError for a candidate_book
    other than 'stock'/'crypto' (a sizing input must fail loudly, not silently
    pick a book).
    """
    if candidate_book not in ('stock', 'crypto'):
        raise ValueError(
            f"candidate_book must be 'stock' or 'crypto', got {candidate_book!r}")
    hi = float(max_risk) if max_risk is not None else float(cap)
    if not np.isfinite(hi) or hi <= 0:
        return 0.0
    base = (list(stock_risks) if candidate_book == 'stock'
            else list(crypto_risks))
    other = (crypto_risks if candidate_book == 'stock' else stock_risks)
    rho_b = rho_stock if candidate_book == 'stock' else rho_crypto
    rho_o = rho_crypto if candidate_book == 'stock' else rho_stock
    is_stock = candidate_book == 'stock'
    # Hoisted loop invariants: the OTHER book's risk and the rho_cross clip do
    # not change across the bisection. Operand order in acct_with matches the
    # original account_stop_risk call per branch, so results are bit-identical.
    r_other = diversified_book_risk(other, rho_o)
    rx = _clip_rho_cross(rho_cross)

    def acct_with(rc):
        r_b = diversified_book_risk(base + [rc], rho_b)
        if is_stock:
            return float(np.sqrt(max(
                r_b ** 2 + r_other ** 2 + 2.0 * rx * r_b * r_other, 0.0)))
        return float(np.sqrt(max(
            r_other ** 2 + r_b ** 2 + 2.0 * rx * r_other * r_b, 0.0)))

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

    The lower-vol book gets the larger share. Only the STOCK share is clamped
    to [clamp_lo, clamp_hi]; the crypto share is its complement (1 - w_s), so
    the 'neither book starved' guarantee holds iff clamp_lo + clamp_hi == 1
    (true for the default). An asymmetric clamp can push the crypto share
    outside the band. Returns (cap_stock, cap_crypto). Fail-open to an even
    split when a vol is missing/zero.
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

# Upper bound on the simulator period grid: _bucket allocates np.zeros(n) and
# the Jetson shares 8 GB with the live bots — a timestamp passed as
# exit_period (1.7e9) would try to allocate ~13.6 GB. Fail loudly instead.
MAX_SIM_PERIODS = 100_000


def simulate_two_books(stock_trades, crypto_trades, periods=None):
    """Replay both books on ONE shared equity timeline.

    Each trade is a dict with at least:
        exit_period: int index into a common period grid (e.g. day number);
        net_pct:     realized net return of the trade, in PERCENT;
        weight:      fraction of equity allocated to the trade (default 1.0;
                     an explicit None also means 1.0).
    Realized P&L lands on the bucket of its exit_period (mark-on-close) and is
    SUMMED into one shared per-period P&L series. The equity curve is the
    ADDITIVE cumulative sum of net_pct in percentage points — NOT a compounded
    curve, and there is no equity feedback between periods; drawdowns are in
    additive-% space, realized-exit-only (open mark-to-market is invisible),
    and are therefore a LOWER bound on the true concurrent-equity drawdown.
    Comparable across A/B variants of the same window; not an absolute
    drawdown estimate.

    Inputs are materialized once, so iterators/generators are accepted.
    `periods` overrides the grid length (default = max exit_period + 1);
    trades whose exit_period falls outside the grid are excluded from the P&L
    and counted in 'n_dropped_trades'. A grid longer than MAX_SIM_PERIODS
    raises ValueError.

    Returns a dict of scalars (no curves): n_periods, n_dropped_trades,
    combined_total_pct, combined_max_drawdown_pct, combined_sharpe,
    stock_total_pct, crypto_total_pct, stock_max_drawdown_pct,
    crypto_max_drawdown_pct, drawdown_concentration (None when neither book
    drew down), realized_cross_corr, realized_cross_corr_active and
    n_overlap_periods. 'combined_sharpe' is mean/std * sqrt(n) over the period
    grid, i.e. a sqrt(N)-scaled t-statistic rather than a per-period Sharpe:
    comparable only across SAME-length windows (fine for the capped-vs-
    uncapped A/B this exists for, where it is monotone-equivalent).
    'realized_cross_corr' correlates the FULL zero-filled grids, so sparse or
    non-overlapping activity dilutes/biases it (it measures co-activity as
    much as co-movement); 'realized_cross_corr_active' correlates only periods
    where BOTH books had non-zero P&L and is None below 10 such periods
    (n_overlap_periods reports the count). With no trades or a non-positive
    grid, a REDUCED 5-key dict is returned (n_periods, n_dropped_trades,
    combined_total_pct, combined_max_drawdown_pct, combined_sharpe) and
    n_periods is 0 regardless of the `periods` argument — callers must .get()
    the per-book/diagnostic keys.
    """
    def _bucket(trades, n):
        pnl = np.zeros(n, dtype=float)
        for t in trades:
            p = int(t['exit_period'])
            if 0 <= p < n:
                w = t.get('weight')
                pnl[p] += float(t['net_pct']) * (1.0 if w is None else float(w))
        return pnl

    stock_trades = list(stock_trades)
    crypto_trades = list(crypto_trades)
    all_trades = stock_trades + crypto_trades
    n = int(periods) if periods is not None else (
        max(int(t['exit_period']) for t in all_trades) + 1 if all_trades else 0)
    if not all_trades or n <= 0:
        return {'n_periods': 0, 'combined_max_drawdown_pct': 0.0,
                'combined_sharpe': 0.0, 'combined_total_pct': 0.0,
                'n_dropped_trades': len(all_trades)}
    if n > MAX_SIM_PERIODS:
        raise ValueError(
            f'simulate_two_books: period grid n={n} exceeds MAX_SIM_PERIODS='
            f'{MAX_SIM_PERIODS} — exit_period values look like timestamps, '
            f'not period indices')
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

    active = (s_pnl != 0.0) & (c_pnl != 0.0)
    n_overlap = int(active.sum())
    if (n_overlap >= 10 and s_pnl[active].std() > 1e-9
            and c_pnl[active].std() > 1e-9):
        cross_active = round(
            float(np.corrcoef(s_pnl[active], c_pnl[active])[0, 1]), 4)
    else:
        cross_active = None

    return {
        'n_periods': int(n),
        'n_dropped_trades': int(n_dropped),
        'combined_total_pct': round(float(comb_eq[-1]), 4),
        'combined_max_drawdown_pct': round(comb_dd, 4),
        'combined_sharpe': round(comb_sharpe, 4),
        'stock_max_drawdown_pct': round(s_dd, 4),
        'crypto_max_drawdown_pct': round(c_dd, 4),
        'stock_total_pct': round(float(s_eq[-1]), 4),
        'crypto_total_pct': round(float(c_eq[-1]), 4),
        # Diversification check: if the books were independent the combined
        # drawdown would be < the sum of per-book drawdowns. A combined DD
        # approaching the sum is the concentration the account cap targets.
        'drawdown_concentration': (round(comb_dd / (s_dd + c_dd), 4)
                                   if (s_dd + c_dd) > 1e-9 else None),
        'realized_cross_corr': (round(float(np.corrcoef(s_pnl, c_pnl)[0, 1]), 4)
                                if s_pnl.std() > 1e-9 and c_pnl.std() > 1e-9
                                else None),
        'realized_cross_corr_active': cross_active,
        'n_overlap_periods': n_overlap,
    }


# ---------------------------------------------------------------------------
# GATE-1: cross-book stop-risk registry + measurement (wave-8 #7)
# ---------------------------------------------------------------------------
# The live clamp needs both books' current risk, but the loops run as separate
# processes. Each writes its own diversified stop-risk to a shared registry and
# reads the other's; the report below is JOURNALED ONLY (no trading decision),
# so we can measure whether the per-book ENB caps actually let the two books
# stack toward the account cap before wiring the clamp (the model-facing step).

# Anchored to this module's directory (repo convention — gpu_lock, trade
# journal, notify all do the same), so the two loop processes and gui.py's
# BASE_DIR-anchored reader agree on ONE file regardless of launch CWD.
ACCOUNT_RISK_REGISTRY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'account_risk_registry.json')


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


def _write_book_risk(book, book_risk, rho_book, path=ACCOUNT_RISK_REGISTRY,
                     now=None):
    """Internal: upsert one book's entry; returns (registry, write_ok).

    write_ok is False when nothing reached disk (lock starvation, OSError,
    non-finite risk) — the returned registry still carries the entry in
    memory on the OSError path so the caller's report stays usable.
    """
    global _write_warned
    now = time.time() if now is None else float(now)
    risk = float(book_risk)
    rho = float(rho_book)
    if not np.isfinite(risk):
        # Refuse to persist: json.dump would emit a non-RFC-8259 NaN/Infinity
        # token into a three-consumer file. The previous entry ages into
        # stale_books — the honest 'unknown' signal.
        logger.warning("[ACCT-RISK] non-finite risk for %s not written "
                       "(risk=%r) — entry will go stale", book, book_risk)
        return read_registry(path), False
    if not np.isfinite(rho):
        rho = 0.0
    entry = {'risk': risk, 'rho': rho, 'ts': now}
    # Per-writer tmp: pid alone collides between the two --combined-bots
    # THREADS, and the error-path unlink could delete the sibling's in-flight
    # staging file.
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        with open(f'{path}.lock', 'w') as lock_f:
            # Bounded non-blocking acquire (~1s worst case): an unbounded
            # LOCK_EX inside the trading cycle could park the loop behind a
            # SIGSTOPped/stalled peer — a block is not an exception, so no
            # try/except upstream could rescue it. Exhaustion raises
            # BlockingIOError (an OSError) into the fail-open branch below.
            for attempt in range(40):
                try:
                    fcntl.flock(lock_f.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if attempt == 39:
                        raise
                    time.sleep(0.025)
            reg = read_registry(path)  # read under the lock: pick up the
            reg[str(book)] = entry     # other book's concurrent write
            with open(tmp, 'w') as f:
                json.dump(reg, f)
                f.flush()
                os.fsync(f.fileno())   # os.replace must commit real data
            os.replace(tmp, path)
        _write_warned = False          # recovered: a NEW outage warns again
        return reg, True
    except OSError as e:
        if not _write_warned:
            _write_warned = True
            logger.warning("[ACCT-RISK] registry write failed (%s) — GATE-1 "
                           "measurement goes stale until writes recover", e)
        try:
            os.unlink(tmp)             # don't leave partial tmp on disk-full
        except OSError:
            pass
        reg = read_registry(path)      # fail-open: return a usable registry
        reg[str(book)] = entry
        return reg, False


def write_book_risk(book, book_risk, rho_book, path=ACCOUNT_RISK_REGISTRY,
                    now=None):
    """Atomically upsert one book's diversified stop-risk into the registry.

    ON-DISK CONTRACT (three consumers — change nothing lightly):
    {book: {'risk': float, 'rho': float, 'ts': float}} in ONE JSON file.
    - key: book name from BaseTradingLoop.get_asset_type() — 'stock' or
      'crypto' ONLY; the top level admits book names only, so any metadata
      key would surface as a phantom entry in stale_books (schema changes
      need a migration, not an added key).
    - risk: this book's diversified single-factor stop-risk as a FRACTION of
      account equity (compare per-book to MAX_BOOK_RISK_PCT; summed toward
      ACCOUNT_RISK_CAP across books). Always finite as written (non-finite is
      refused here); >= 0 by the writer's contract (diversified_book_risk
      output — the gate-1 reader drops any negative).
    - rho: the intra-book avg |corr| used to diversify it (0.0 if unknown).
    - ts: time.time() wall clock at write. Written roughly every 10th loop
      cycle (~5-10 min incl. thermal throttle).
    Consumers: read_registry/account_risk_gate1_report here, AND gui.py's
    _refresh_risk_gauge (own inline json.load, own 600s staleness rule) —
    change either side and check both.

    Both books write this file (two processes, or two threads under
    --combined-bots), so the read-modify-write is serialized with an flock on
    a sidecar lockfile ('{path}.lock' — never replaced, so the lock target is
    stable across os.replace) and staged through a per-writer tmp name.

    Returns the updated registry. Fail-open: a write error is swallowed
    (the measurement must never break the trading loop) but logged once per
    outage so a stale GATE-1 journal is diagnosable.
    """
    return _write_book_risk(book, book_risk, rho_book, path=path, now=now)[0]


def account_risk_gate1_report(registry, rho_cross=1.0, cap=ACCOUNT_RISK_CAP,
                              stale_after_s=REGISTRY_STALE_AFTER_S, now=None):
    """Combined cross-book stop-risk from the registry — measurement only.

    Each book stored its already-diversified risk R_b; combine via the two-
    super-position formula at rho_cross. Entries older than stale_after_s, or
    stamped more than CLOCK_SKEW_TOL_S in the FUTURE (RTC-less Jetson + NTP
    steps), or negative/non-finite, are dropped (book treated as down).
    Takes NO trading action.

    DEGENERACY WARNING: at rho_cross == 1.0 (the production CROSS_BOOK_RHO
    placeholder) account_stop_risk == book_sum and concentration == 1.0
    IDENTICALLY — those fields are then a worst-case bound restating the
    assumption, not a measurement. 'account_stop_risk_indep' (the rho_cross=0
    combination) is journaled alongside so every row brackets the truth in
    [indep, book_sum].

    Diagnostics: 'stale_books' = registry keys dropped as unusable;
    'missing_books' = EXPECTED_BOOKS absent from the registry entirely (a
    0.0 stock_risk/crypto_risk means 'flat OR unreported' — check both
    lists); 'unknown_books' = FRESH keys outside EXPECTED_BOOKS whose risk is
    NOT aggregated; 'skewed_books' = future-dated beyond tolerance (also in
    stale_books); 'book_ages_s' = now - ts per parseable finite-ts entry
    (dropped entries included);
    'stock_rho'/'crypto_rho' = the fresh entries' stored intra-book rho
    (None when the book is missing/stale).
    """
    now = time.time() if now is None else float(now)
    registry = registry if isinstance(registry, dict) else {}
    fresh, rhos, ages, skewed = {}, {}, {}, []
    for b, e in registry.items():
        if not isinstance(e, dict):
            continue
        try:
            # Coerce defensively: a hand-edited/corrupt registry can hold
            # string or None values; treat the entry as missing (-> stale).
            risk = float(e.get('risk'))
            ts = float(e.get('ts', 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ts):
            # NaN/inf ts (json.load accepts the bare NaN token) would put a
            # NaN age into book_ages_s — a non-RFC-8259 token in the JSON
            # journal. Treat as unparseable (-> stale).
            continue
        age = now - ts
        ages[b] = round(age, 1)  # recorded for DROPPED entries too — the age
        #                          of a bad entry is diagnostic signal
        if not np.isfinite(risk) or risk < 0:
            # Negative would silently split paths: diversified_book_risk
            # drops it from account_stop_risk while book_sum keeps it.
            continue
        if age < -CLOCK_SKEW_TOL_S:
            skewed.append(b)
            continue
        if age <= stale_after_s:
            fresh[b] = risk
            try:
                rho_e = float(e.get('rho'))
                rhos[b] = round(rho_e, 4) if np.isfinite(rho_e) else None
            except (TypeError, ValueError):
                rhos[b] = None
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
        'stale_books': sorted(set(registry) - set(fresh)),
        'rho_cross': _clip_rho_cross(rho_cross),
        'account_stop_risk_indep': round(
            account_stop_risk([r_s], [r_c], 0.0, 0.0, 0.0), 5),
        'missing_books': sorted(set(EXPECTED_BOOKS) - set(registry)),
        'unknown_books': sorted(set(fresh) - set(EXPECTED_BOOKS)),
        'skewed_books': sorted(skewed),
        'book_ages_s': ages,
        'stock_rho': rhos.get('stock'),
        'crypto_rho': rhos.get('crypto'),
    }


def record_book_risk_and_report(book, book_risks, rho_book, rho_cross=1.0,
                                cap=ACCOUNT_RISK_CAP, path=ACCOUNT_RISK_REGISTRY,
                                stale_after_s=REGISTRY_STALE_AFTER_S, now=None):
    """Write this book's diversified stop-risk to the registry, return the
    combined GATE-1 report. One call for the per-cycle measurement hook.

    book_risks may be any iterable of non-negative stop-risk fractions, OR
    None meaning 'state unknown' (e.g. equity unavailable): then NOTHING is
    written — the previous entry ages honestly into stale_books instead of
    being overwritten with a confident 0.0 — and the report is built from the
    registry as-is. A written risk of 0.0 therefore always means 'measured
    flat'. Adds provenance keys to the report: 'self_risk'/'self_rho' (this
    book's own just-computed values, known even when the write fails),
    'n_positions', and 'registry_write_ok' (None when no write attempted).
    """
    if book_risks is None:
        rep = account_risk_gate1_report(read_registry(path),
                                        rho_cross=rho_cross, cap=cap,
                                        stale_after_s=stale_after_s, now=now)
        rep['self_risk'] = None
        rep['self_rho'] = None
        rep['n_positions'] = None
        rep['registry_write_ok'] = None
        return rep
    risks = list(book_risks)
    r_b = diversified_book_risk(risks, rho_book)
    reg, ok = _write_book_risk(book, r_b, rho_book, path=path, now=now)
    rep = account_risk_gate1_report(reg, rho_cross=rho_cross, cap=cap,
                                    stale_after_s=stale_after_s, now=now)
    rep['self_risk'] = round(float(r_b), 5) if np.isfinite(r_b) else None
    rep['self_rho'] = (round(float(rho_book), 4)
                       if np.isfinite(float(rho_book)) else None)
    rep['n_positions'] = len(risks)
    rep['registry_write_ok'] = bool(ok)
    return rep
