"""Anytime-valid LLM-spend ledger (INTEL W10, 2026-09) — MEASUREMENT-ONLY.

Asks: does the *deployed* LLM size tilt (m = 0.5 + s, base_loop.py:3315)
pay for itself, net of fees and LLM spend? It answers with three one-sided
betting e-processes on a daily mark-to-market statistic d_t, following
Scout C's Design A (scratchpad SCOUT_C.md). Every tunable number
(d_t construction, clip bound B, bet, alpha/threshold, first look,
horizon, fallback, legacy-journal exclusion) is read from the
pre-registration sheet
research/campaign_2026-09_jetson/llm_eprocess_params.json.
In-code defaults (`DEFAULT_SHEET`) are used only by --selftest when that
sheet is missing; the committed sheet is byte-for-byte the same values
(pinned by a test).

NOTHING in the repo reads this module's verdict. It places no orders,
makes zero LLM calls, never imports llm_client, and gates no trade. It
writes only its own JSON.

Modes
  --selftest            synthetic daily d series (null boundaries, +-1
                        bp/day) -> size / power at the sheet's parameters.
                        The only mode that runs while the sheet is unsigned.
  --journals DIR --days N
                        live ledger. REFUSES (exit 3) unless the sheet has
                        non-null signed_by / signed_at and its
                        registration_sha equals registration_sha(sheet)
                        (sha256 of the canonical JSON minus that key). This
                        module never signs.

Statistic (Scout C, "d_t"). Lot i = one journaled buy row
(base_loop.py:3497-3509 / stock_loop.py:1364-1372: llm_multiplier,
final_notional, fill_price, ts); it closes at the next sell row for the
symbol (base_loop.py:1995-2002, fill_price). q_i = N_i / P_i^fill, tilt
qty tau_i = q_i * (1 - m_bar / m_i), where m_bar is the burn-in mean
multiplier (primary) or 1.0 (secondary, report-only).
  G_t = sum_i tau_i (P_i^end,t - P_i^start,t)
        - sum_{entries on t} phi_e |tau_i| P^fill - sum_{exits on t} phi_x |tau_i| P^exit
  d_t = 1e4 (G_t - c_t) / E_{t-1}                      [bp of prior equity]
start = max(entry, day start), end = min(exit, day end), marks = last
close <= day end; phi_e = phi_x = round_trip_cost_pct / 2 (fees.py:187).
c^lo = sum of journaled llm_analysis cost_usd (base_loop.py:1836-1839);
c^hi = the $1.00 daily cap (llm_client.py:261). KEEP uses c^hi, the KILL
tests use c^lo (conservative each way).

Clipping. B = max(delta_mult*delta, mad_mult*mad_scale*MAD(d^lo burn-in)),
frozen after burn-in; x_t = (clip(d_t, -B, B) + B) / (2B) in [0, 1]. A
decision counts only while the clip rate in the direction favouring it is
<= max_favouring_clip_rate (lower clips for KEEP, upper for KILL).

E-processes (Waudby-Smith & Ramdas 2024, JRSSB 86(1):1-27, aGRAPA bet;
Ville's inequality per Ramdas et al. 2023, Stat. Sci. 38(4)). For a null
point m0 = (a + B) / (2B):
  up   (H0: mean <= m0): K_t = prod_{s<=t} (1 + lam_s (x_s - m0)),
       lam_s = clip((mu_{s-1} - m0) / (var_{s-1} + (mu_{s-1} - m0)^2), 0, c/m0)
  down (H0: mean >= m0): mirror with (m0 - x_s) and cap c/(1 - m0)
  mu_t = (mu_prior + sum_{i<=t} x_i) / (t + 1)
  var_t = (var_prior + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)
Under the weak (time-averaged conditional-mean) null each K is a
nonnegative supermartingale, so P(sup_t K_t >= 1/alpha) <= alpha for any
dependence. Tests: KEEP = up on x^hi at a=0; KILL_HARM = down on x^lo at
a=0; KILL_FUTILITY = down on x^lo at a=delta. Display-only: the two-sided
hedged-capital CS (WSR 2024 Thm 3, predictable plug-in bet).

Schedule. Burn-in days freeze m_bar and B (untested). From CS day
first_look.cs_day with >= first_look.min_lots lots, stop the first day an
e-value reaches the threshold (priority HARM > FUTILITY > KEEP). At the
horizon: one-sided t_{n-1} fallback on clipped d at fallback.alpha
(Ibragimov-Mueller K-block t if |rho_1| > fallback.rho1_abs_max), else the
owner's pre-signed inconclusive_default (null -> CONTINUE).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PARAMS_PATH = (BASE_DIR / 'research' / 'campaign_2026-09_jetson'
                       / 'llm_eprocess_params.json')
DEFAULT_LIVE_OUT = BASE_DIR / 'logs' / 'llm_eprocess_report.json'

VERDICTS = ('KEEP', 'KILL_HARM', 'KILL_FUTILITY', 'CONTINUE', 'NOT_SIGNED')
EXIT_OK, EXIT_USAGE, EXIT_NOT_SIGNED, EXIT_DATA = 0, 2, 3, 4

# ---------------------------------------------------------------------------
# Pre-registration sheet (Scout C table rows 1-17 + two text-level keys).
# The committed JSON must equal this dict (tests pin it); the live path reads
# ONLY the JSON file.
# ---------------------------------------------------------------------------
DEFAULT_SHEET = {
    "schema": "llm_eprocess_params/1",
    "registration_id": None,
    "signed_by": None,
    "signed_at": None,
    "registration_sha": None,
    "start": {
        "value": {"rule": "first_buy_after", "not_before_utc": None,
                  "git_sha": None},
        "reason": "Row 1. Start at the first post-retrain buy under the current "
                  "gate code (s<0.15 veto only, base_loop.py:3307-3315); the "
                  "owner records not_before_utc and the git sha at activation."},
    "legacy_exclusion": {
        "value": {"exclude_on_or_before_utc": "2026-05-07",
                  "removed_gate": "llm_below_buy_min (s<0.60)",
                  "observed_window_utc": ["2026-04-06", "2026-05-07"]},
        "reason": "Brief constraint 5. The Apr-May journals ran a since-removed "
                  "s<0.60 gate (first llm_below_buy_min skip in journals/"
                  "2026-04-06.jsonl, still 1213 in 2026-05-07.jsonl, the last "
                  "journal on disk): not the same policy, so every row dated on "
                  "or before the boundary is dropped regardless of start."},
    "unit_day": {
        "value": {"unit": "bp_of_prior_equity", "day": "UTC",
                  "books": "combined"},
        "reason": "Row 2. Spend is shared across both books (llm_client.py "
                  "daily cap); E_{t-1} = prior daily equity."},
    "statistic": {
        "value": "daily_mtm_tilt_pnl_net_fees_cost",
        "reason": "Row 3. Day-disjoint mark-to-market keeps the lag at 1 day, so "
                  "the martingale-difference structure holds."},
    "m_bar": {
        "value": {"primary": "burn_in_mean", "secondary": 1.0},
        "reason": "Row 4. Budget-neutral primary values the reallocation, not "
                  "leverage; 1.0 mirrors llm_eval's neutral baseline (report only)."},
    "fees": {
        "value": {"source": "fees.round_trip_cost_pct", "entry_share": 0.5,
                  "exit_share": 0.5, "spread_fallback": "fees.FLAT_SPREAD_PCT"},
        "reason": "Row 5. One cost model (fees.py:187), split half/half."},
    "cost": {
        "value": {"c_hi_usd_per_day": 1.0,
                  "c_lo": "sum_journaled_llm_analysis_cost_usd"},
        "reason": "Row 6. llm_cost.json keeps today only; KEEP is charged the "
                  "$1.00 cap, KILL the journaled analyst-role spend."},
    "burn_in": {
        "value": {"days": 10},
        "reason": "Row 7. Freezes m_bar and B without look-ahead; untested."},
    "clip": {
        "value": {"delta_mult": 2.5, "mad_mult": 5.0, "mad_scale": 1.4826,
                  "max_favouring_clip_rate": 0.05},
        "reason": "Row 8. Bounded-mean betting needs a range; the 2.5*delta "
                  "floor keeps every null point strictly inside (0,1)."},
    "delta_bp_per_day": {
        "value": 1.0,
        "reason": "Row 9. Futility margin ~ $10/day at $100k, 10x the spend cap."},
    "bet": {
        "value": {"kind": "aGRAPA", "c": 0.5, "mu_prior": 0.5,
                  "var_prior": 0.25},
        "reason": "Row 10. WSR 2024 aGRAPA; simple and fixed."},
    "alpha": {
        "value": {"sequential_per_direction": 0.025, "threshold": 40.0,
                  "fallback": 0.025},
        "reason": "Row 11. Union bound 0.025 + 0.025 = 0.05 per direction; "
                  "threshold = 1/alpha_seq (Ville)."},
    "first_look": {
        "value": {"cs_day": 10, "min_lots": 30},
        "reason": "Row 12. Robustness only; validity holds at any time."},
    "horizon": {
        "value": {"cs_day": 180, "interim_status_cs_day": 90},
        "reason": "Row 13. Scout C sim: 90 days gives 43% KEEP power at +1 "
                  "bp/day, 180 days 91%."},
    "fallback": {
        "value": {"test": "one_sided_t", "alpha": 0.025, "im_blocks": 8,
                  "rho1_abs_max": 0.2},
        "reason": "Row 14. Fixed-N read at the horizon (Koning-van Meer "
                  "embedding logic); IM K-block t when lag-1 autocorr is material."},
    "veto": {
        "value": "KILL ends the spend only if decision_report's llm_veto CI "
                 "does not exclude > 0; otherwise 'kill the tilt, keep the veto'",
        "reason": "Row 15. The veto is part of what the spend buys; not "
                  "evaluated by this module (owner reads decision_report)."},
    "llm_eval_b2": {
        "value": "diagnostic_non_voting",
        "reason": "Row 16. One alpha per decision."},
    "inconclusive_default": {
        "value": None, "allowed": ["KEEP", "KILL_FUTILITY"],
        "reason": "Row 17. Owner fixes it ex ante; null -> CONTINUE at horizon."},
    "cs_display": {
        "value": {"alpha": 0.05, "theta": 0.5, "c": 0.5, "grid": 1001},
        "reason": "Design A text: two-sided hedged-capital CS, display only."},
    "capital_basis": {
        "value": "paper_equity",
        "reason": "Owner ask 4 default: normalise to paper equity."},
}


def registration_sha(sheet: dict) -> str:
    """sha256 of the canonical JSON of the sheet WITHOUT its registration_sha
    key (sort_keys, compact separators, ASCII). Written into the sheet at
    signing; recomputed on every live run."""
    body = {k: v for k, v in sheet.items() if k != 'registration_sha'}
    blob = json.dumps(body, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode('ascii')).hexdigest()


def signature_status(sheet: dict) -> tuple[bool, str]:
    """(ok, reason). ok iff signed_by / signed_at / registration_sha are all
    non-null and registration_sha matches the recomputed hash."""
    for key in ('signed_by', 'signed_at', 'registration_sha'):
        if not sheet.get(key):
            return False, f"sheet not signed: '{key}' is null"
    want = registration_sha(sheet)
    if sheet['registration_sha'] != want:
        return False, ("sheet tampered: registration_sha "
                       f"{sheet['registration_sha'][:12]}... != recomputed "
                       f"{want[:12]}... (any edit after signing is a new "
                       "registration and restarts the clock)")
    return True, 'signed'


def load_sheet(path) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def config_from_sheet(sheet: dict) -> dict:
    """Flatten the sheet's `value` fields into the typed config the ledger
    uses. Raises ValueError on a missing key or an inconsistent threshold."""
    try:
        v = {k: sheet[k]['value'] for k in (
            'burn_in', 'clip', 'delta_bp_per_day', 'bet', 'alpha',
            'first_look', 'horizon', 'fallback', 'cost', 'fees', 'm_bar',
            'inconclusive_default', 'cs_display', 'start', 'legacy_exclusion')}
    except (KeyError, TypeError) as e:
        raise ValueError(f"params sheet missing key/value: {e}") from None
    a = v['alpha']
    thr, a_seq = float(a['threshold']), float(a['sequential_per_direction'])
    if not (0 < a_seq < 1) or abs(thr - 1.0 / a_seq) > 1e-9:
        raise ValueError(f"alpha.threshold {thr} != 1/sequential_per_direction")
    inc = v['inconclusive_default']
    if inc is not None and inc not in sheet['inconclusive_default'].get('allowed', []):
        raise ValueError(f"inconclusive_default {inc!r} not in allowed")
    return {
        'burn_days': int(v['burn_in']['days']),
        'delta': float(v['delta_bp_per_day']),
        'delta_mult': float(v['clip']['delta_mult']),
        'mad_mult': float(v['clip']['mad_mult']),
        'mad_scale': float(v['clip']['mad_scale']),
        'max_clip': float(v['clip']['max_favouring_clip_rate']),
        'bet_c': float(v['bet']['c']),
        'mu_prior': float(v['bet']['mu_prior']),
        'var_prior': float(v['bet']['var_prior']),
        'threshold': thr,
        'first_cs_day': int(v['first_look']['cs_day']),
        'min_lots': int(v['first_look']['min_lots']),
        'horizon': int(v['horizon']['cs_day']),
        'interim_day': int(v['horizon']['interim_status_cs_day']),
        'fb_alpha': float(v['fallback']['alpha']),
        'fb_blocks': int(v['fallback']['im_blocks']),
        'fb_rho_max': float(v['fallback']['rho1_abs_max']),
        'c_hi_usd': float(v['cost']['c_hi_usd_per_day']),
        'fee_entry_share': float(v['fees']['entry_share']),
        'fee_exit_share': float(v['fees']['exit_share']),
        'm_bar_secondary': float(v['m_bar']['secondary']),
        'inconclusive_default': inc,
        'cs_alpha': float(v['cs_display']['alpha']),
        'cs_theta': float(v['cs_display']['theta']),
        'cs_c': float(v['cs_display']['c']),
        'cs_grid': int(v['cs_display']['grid']),
        'not_before_utc': v['start'].get('not_before_utc'),
        'exclude_through_utc': v['legacy_exclusion']['exclude_on_or_before_utc'],
    }


# ---------------------------------------------------------------------------
# Estimators (pure numpy)
# ---------------------------------------------------------------------------
def clip_bound(d_burn, delta, delta_mult, mad_mult, mad_scale):
    """B = max(delta_mult*delta, mad_mult*mad_scale*MAD(d_burn)) along the
    last axis (Scout C row 8)."""
    d = np.asarray(d_burn, dtype=float)
    med = np.median(d, axis=-1, keepdims=True)
    mad = np.median(np.abs(d - med), axis=-1)
    return np.maximum(delta_mult * delta, mad_mult * mad_scale * mad)


def to_unit(d, B):
    """x = (clip(d,-B,B)+B)/(2B); returns (x, lower_clip, upper_clip)."""
    d = np.asarray(d, dtype=float)
    Bc = np.asarray(B, dtype=float)[..., None] if np.ndim(B) else float(B)
    lo, hi = d <= -Bc, d >= Bc
    return (np.clip(d, -Bc, Bc) + Bc) / (2 * Bc), lo, hi


def agrapa_eprocess(x, m0, up: bool, c=0.5, mu_prior=0.5, var_prior=0.25):
    """One-sided aGRAPA betting e-process (WSR 2024 §5; Scout C Design A).
    x: (T,) or (R, T) in [0,1]; m0: scalar or (R,). up=True tests
    H0: mean <= m0 with K_t = prod(1 + lam_s (x_s - m0)), lam capped at c/m0;
    up=False tests H0: mean >= m0 with the mirror (m0 - x_s), cap c/(1-m0).
    lam_s uses mu/var estimated from x_1..x_{s-1} (predictable). Returns the
    running K_t with x's shape."""
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    X = np.atleast_2d(x)
    R, T = X.shape
    m = np.broadcast_to(np.asarray(m0, dtype=float), (R,)).copy()
    cap = c / m if up else c / (1.0 - m)
    mu = np.full(R, mu_prior)
    var = np.full(R, var_prior)
    s_x = np.zeros(R)
    s_v = np.zeros(R)
    logk = np.zeros(R)
    out = np.empty((R, T))
    for t in range(T):
        g = (mu - m) if up else (m - mu)
        lam = np.clip(g / (var + g * g), 0.0, cap)
        inc = (X[:, t] - m) if up else (m - X[:, t])
        logk += np.log1p(lam * inc)
        out[:, t] = np.exp(logk)
        n = t + 1
        s_x += X[:, t]
        mu = (mu_prior + s_x) / (n + 1)
        s_v += (X[:, t] - mu) ** 2
        var = (var_prior + s_v) / (n + 1)
    return out[0] if squeeze else out


def hedged_cs(x, alpha=0.05, theta=0.5, c=0.5, grid=1001,
              mu_prior=0.5, var_prior=0.25):
    """Running two-sided hedged-capital CS for mean(x), x in [0,1] (WSR 2024
    Thm 3, predictable plug-in lam_t = sqrt(2 log(2/alpha) /
    (var_{t-1} t log(1+t))), truncated at c/m and c/(1-m)). Returns (L, U)
    arrays, running-intersected. Display only."""
    x = np.asarray(x, dtype=float)
    ms = np.linspace(0.0, 1.0, grid + 2)[1:-1]
    lp = np.zeros_like(ms)
    lm = np.zeros_like(ms)
    L, U = np.empty(len(x)), np.empty(len(x))
    lo_run, hi_run = 0.0, 1.0
    var, s_x, s_v = var_prior, 0.0, 0.0
    thr = math.log(1.0 / alpha)
    for t, xt in enumerate(x):
        n = t + 1
        lam = math.sqrt(2 * math.log(2 / alpha) / (var * n * math.log(1 + n)))
        lp += np.log1p(np.minimum(lam, c / ms) * (xt - ms))
        lm += np.log1p(-np.minimum(lam, c / (1 - ms)) * (xt - ms))
        logk = np.logaddexp(math.log(theta) + lp, math.log(1 - theta) + lm)
        keep = ms[logk < thr]
        if keep.size:
            lo_run, hi_run = max(lo_run, keep.min()), min(hi_run, keep.max())
        L[t], U[t] = lo_run, hi_run
        s_x += xt
        mu = (mu_prior + s_x) / (n + 1)
        s_v += (xt - mu) ** 2
        var = (var_prior + s_v) / (n + 1)
    return L, U


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T_df > t), pure python (no scipy)."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def fallback_test(dc, cfg, target=0.0, direction='greater'):
    """Horizon fallback on clipped daily d (Scout C row 14): one-sided
    t_{n-1}; if |rho_1| > rho1_abs_max, an Ibragimov-Mueller t on K
    contiguous day-block means (K-1 dof). Returns (p, method, rho1)."""
    y = np.asarray(dc, dtype=float) - target
    n = len(y)
    if n < 3:
        return 1.0, 'n<3', float('nan')
    yc = y - y.mean()
    den = float((yc * yc).sum())
    rho1 = float((yc[1:] * yc[:-1]).sum() / den) if den > 0 else 0.0
    if abs(rho1) > cfg['fb_rho_max']:
        blocks = [b.mean() for b in np.array_split(y, cfg['fb_blocks']) if len(b)]
        y, method = np.asarray(blocks), f"im_k{len(blocks)}"
    else:
        method = 't'
    m = len(y)
    sd = float(y.std(ddof=1)) if m > 1 else 0.0
    if sd <= 0:
        return 1.0, method, rho1
    tstat = float(y.mean() / (sd / math.sqrt(m)))
    p = t_sf(tstat, m - 1) if direction == 'greater' else t_sf(-tstat, m - 1)
    return p, method, rho1


def run_ledger_batch(d_lo, d_hi, lots_cum, cfg):
    """The decision engine, vectorized over R paths (rows).

    d_lo / d_hi: (R, T) daily bp series charged c^lo / c^hi; the first
    cfg['burn_days'] columns are burn-in. lots_cum: (R, T) cumulative lot
    count. Returns a dict of arrays: B, E_keep/E_harm/E_fut (R, T_cs),
    lower/upper clip rates (running, x^hi lower for KEEP, x^lo upper for
    KILL), verdict (R,) strings, basis, decision_cs_day (0 = none)."""
    d_lo = np.atleast_2d(np.asarray(d_lo, dtype=float))
    d_hi = np.atleast_2d(np.asarray(d_hi, dtype=float))
    lots_cum = np.atleast_2d(np.asarray(lots_cum, dtype=float))
    R, T = d_lo.shape
    nb = cfg['burn_days']
    B = clip_bound(d_lo[:, :nb], cfg['delta'], cfg['delta_mult'],
                   cfg['mad_mult'], cfg['mad_scale'])
    x_lo, _, up_lo = to_unit(d_lo[:, nb:], B)
    x_hi, low_hi, _ = to_unit(d_hi[:, nb:], B)
    Tcs = T - nb
    kw = dict(c=cfg['bet_c'], mu_prior=cfg['mu_prior'], var_prior=cfg['var_prior'])
    m_zero = 0.5 * np.ones(R)
    m_fut = (cfg['delta'] + B) / (2 * B)
    E_keep = agrapa_eprocess(x_hi, m_zero, True, **kw)
    E_harm = agrapa_eprocess(x_lo, m_zero, False, **kw)
    E_fut = agrapa_eprocess(x_lo, m_fut, False, **kw)
    k = np.arange(1, Tcs + 1)
    rate_low = np.cumsum(low_hi, axis=1) / k
    rate_up = np.cumsum(up_lo, axis=1) / k
    elig = (k >= cfg['first_cs_day'])[None, :] & (lots_cum[:, nb:] >= cfg['min_lots'])
    elig &= (k <= cfg['horizon'])[None, :]
    thr, mc = cfg['threshold'], cfg['max_clip']
    fire = {
        'KILL_HARM': elig & (E_harm >= thr) & (rate_up <= mc),
        'KILL_FUTILITY': elig & (E_fut >= thr) & (rate_up <= mc),
        'KEEP': elig & (E_keep >= thr) & (rate_low <= mc),
    }
    first = {}
    for name, f in fire.items():
        any_ = f.any(axis=1)
        first[name] = np.where(any_, f.argmax(axis=1) + 1, 0)
    verdict = np.full(R, 'CONTINUE', dtype=object)
    basis = np.full(R, 'running', dtype=object)
    day = np.zeros(R, dtype=int)
    for r in range(R):
        cands = [(first[nm][r], i, nm) for i, nm in
                 enumerate(('KILL_HARM', 'KILL_FUTILITY', 'KEEP')) if first[nm][r] > 0]
        if cands:
            dd, _, nm = min(cands)
            verdict[r], basis[r], day[r] = nm, 'sequential', dd
    # Horizon fallback for the undecided paths that reached the horizon.
    H = cfg['horizon']
    if Tcs >= H:
        for r in np.where(day == 0)[0]:
            dlo_c = np.clip(d_lo[r, nb:nb + H], -B[r], B[r])
            dhi_c = np.clip(d_hi[r, nb:nb + H], -B[r], B[r])
            a = cfg['fb_alpha']
            tests = (
                ('KILL_HARM', fallback_test(dlo_c, cfg, 0.0, 'less')),
                ('KILL_FUTILITY', fallback_test(dlo_c, cfg, cfg['delta'], 'less')),
                ('KEEP', fallback_test(dhi_c, cfg, 0.0, 'greater')),
            )
            hit = next(((nm, res) for nm, res in tests if res[0] < a), None)
            day[r] = H
            if hit:
                verdict[r], basis[r] = hit[0], f"fallback_{hit[1][1]}"
            elif cfg['inconclusive_default']:
                verdict[r], basis[r] = cfg['inconclusive_default'], 'inconclusive_default'
            else:
                verdict[r], basis[r] = 'CONTINUE', 'horizon_inconclusive_no_default'
    return {'B': B, 'E_keep': E_keep, 'E_harm': E_harm, 'E_fut': E_fut,
            'rate_low_keep': rate_low, 'rate_up_kill': rate_up,
            'first': first, 'verdict': verdict, 'basis': basis,
            'decision_cs_day': day, 'x_lo': x_lo, 'x_hi': x_hi}


# ---------------------------------------------------------------------------
# Journals -> lots -> d_t (pure; the fetchers are injected)
# ---------------------------------------------------------------------------
def _parse_ts(ts):
    t = _dt.datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
    if t.tzinfo is None:
        t = t.replace(tzinfo=_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc)


def _open_day_file(path: Path):
    """Plain .jsonl, else its .jsonl.gz sibling (mirrors
    trade_journal.open_journal, trade_journal.py:126-138, without importing
    trade_journal — that import touches logs/trader.log and llm_config)."""
    if path.exists():
        return open(path, encoding='utf-8', errors='replace')
    gz = Path(f'{path}.gz')
    if gz.exists():
        return gzip.open(gz, 'rt', encoding='utf-8', errors='replace')
    return None


def load_journal_rows(journal_dir, first_day, last_day, exclude_through=None,
                      not_before=None, actions=('buy', 'sell', 'llm_analysis')):
    """Rows with action in `actions` whose UTC ts date lies in
    [first_day, last_day]. Day files are named by the box's LOCAL date
    (trade_journal.py:98), so one extra file each side is scanned. Rows dated
    on or before `exclude_through` (legacy_exclusion) or before `not_before`
    are dropped. Returns (rows sorted by ts, {file: sha256})."""
    jd = Path(journal_dir)
    rows, hashes = [], {}
    excl = _dt.date.fromisoformat(exclude_through) if exclude_through else None
    nb = _parse_ts(not_before) if not_before else None
    day = first_day - _dt.timedelta(days=1)
    while day <= last_day + _dt.timedelta(days=1):
        p = jd / f"{day.isoformat()}.jsonl"
        f = _open_day_file(p)
        day += _dt.timedelta(days=1)
        if f is None:
            continue
        h = hashlib.sha256()
        with f:
            for line in f:
                h.update(line.encode('utf-8', 'replace'))
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get('action') not in actions or 'ts' not in e:
                    continue
                try:
                    t = _parse_ts(e['ts'])
                except ValueError:
                    continue
                if not (first_day <= t.date() <= last_day):
                    continue
                if excl is not None and t.date() <= excl:
                    continue
                if nb is not None and t < nb:
                    continue
                e['_t'] = t
                rows.append(e)
        hashes[p.name] = h.hexdigest()
    rows.sort(key=lambda e: e['_t'])
    return rows, hashes


def _asset_of(symbol: str) -> str:
    return 'crypto' if '/' in str(symbol) else 'stock'   # journals: 'BTC/USD'


def build_lots(rows, fee_rt_pct):
    """Pair buy rows with the next sell of the same symbol (a sell closes
    every open lot of that symbol — exits are position-level). fee_rt_pct
    (asset, buy_row) -> round-trip cost in PERCENT. Returns lot dicts."""
    open_by_sym, lots = {}, []
    for e in rows:
        sym, act = e.get('symbol'), e.get('action')
        if act == 'buy':
            try:
                n = float(e['final_notional'])
                px = float(e['fill_price'])
                m = float(e.get('llm_multiplier') if e.get('llm_multiplier')
                          is not None else 1.0)
            except (KeyError, TypeError, ValueError):
                continue
            if not (n > 0 and px > 0 and m > 0):
                continue
            asset = _asset_of(sym)
            lot = {'symbol': sym, 'asset': asset, 'entry_t': e['_t'],
                   'entry_px': px, 'qty': n / px, 'mult': m,
                   'fee_rt': float(fee_rt_pct(asset, e)) / 100.0,
                   'exit_t': None, 'exit_px': None}
            lots.append(lot)
            open_by_sym.setdefault(sym, []).append(lot)
        elif act == 'sell' and open_by_sym.get(sym):
            try:
                px = float(e['fill_price'])
            except (KeyError, TypeError, ValueError):
                continue
            for lot in open_by_sym.pop(sym):
                lot['exit_t'], lot['exit_px'] = e['_t'], px
    return lots


def daily_costs(rows):
    """c^lo per UTC day: sum of llm_analysis cost_usd."""
    out = {}
    for e in rows:
        if e.get('action') == 'llm_analysis' and e.get('cost_usd') is not None:
            try:
                out[e['_t'].date()] = out.get(e['_t'].date(), 0.0) + float(e['cost_usd'])
            except (TypeError, ValueError):
                pass
    return out


def _mark(marks_sym, day):
    """Last mark dated <= day (stocks carry Friday's RTH close over weekends)."""
    best = None
    for dd in marks_sym:
        if dd <= day and (best is None or dd > best):
            best = dd
    if best is None:
        raise KeyError(day)
    return marks_sym[best]


def daily_tilt_series(lots, days, marks, equity_prev, cost_lo, c_hi_usd,
                      m_bar, entry_share=0.5, exit_share=0.5):
    """d^lo_t, d^hi_t (bp of E_{t-1}) and G_t for each UTC day in `days`
    (Scout C formula; module docstring). marks: {symbol: {date: close at
    day end}}; equity_prev: {date: E_{t-1}}; cost_lo: {date: usd}."""
    G = np.zeros(len(days))
    lots_cum = np.zeros(len(days))
    for j, day in enumerate(days):
        d0 = _dt.datetime.combine(day, _dt.time(), tzinfo=_dt.timezone.utc)
        d1 = d0 + _dt.timedelta(days=1)
        g = 0.0
        for lot in lots:
            if lot['entry_t'] >= d1:
                continue
            lots_cum[j] += 1
            if lot['exit_t'] is not None and lot['exit_t'] < d0:
                continue
            tau = lot['qty'] * (1.0 - m_bar / lot['mult'])
            ms = marks[lot['symbol']]
            p0 = lot['entry_px'] if lot['entry_t'] >= d0 else \
                _mark(ms, day - _dt.timedelta(days=1))
            exited = lot['exit_t'] is not None and lot['exit_t'] < d1
            p1 = lot['exit_px'] if exited else _mark(ms, day)
            g += tau * (p1 - p0)
            if lot['entry_t'] >= d0:
                g -= entry_share * lot['fee_rt'] * abs(tau) * lot['entry_px']
            if exited:
                g -= exit_share * lot['fee_rt'] * abs(tau) * lot['exit_px']
        G[j] = g
    E = np.array([float(equity_prev[d]) for d in days])
    clo = np.array([float(cost_lo.get(d, 0.0)) for d in days])
    return {'G': G, 'd_lo': 1e4 * (G - clo) / E,
            'd_hi': 1e4 * (G - c_hi_usd) / E, 'lots_cum': lots_cum}


def burn_in_m_bar(lots, first_day, burn_days):
    last = first_day + _dt.timedelta(days=burn_days - 1)
    ms = [lot['mult'] for lot in lots if first_day <= lot['entry_t'].date() <= last]
    return float(np.mean(ms)) if ms else None


# ---------------------------------------------------------------------------
# Selftest (synthetic; Scout C's sizing model, vectorized)
# ---------------------------------------------------------------------------
SELFTEST_MODEL = {'equity': 1e5, 'lots_per_day': 5.0, 'notional': 3000.0,
                  'tilt_frac': 0.13, 'vol': 0.035, 'rho': 0.5, 't_df': 3}


def synth_noise(rng, R, T, model=SELFTEST_MODEL):
    """(R, T) tilt-P&L noise in bp: Poisson open lots/day, tilt notional
    N(0, tilt_frac*notional), t3 returns (unit-variance scaled) at `vol` with
    a sqrt(rho) common factor per day. Also returns cumulative lot counts."""
    k = rng.poisson(model['lots_per_day'], size=(R, T))
    nl = int(k.sum())
    day_of = np.repeat(np.arange(R * T), k.ravel())
    w = rng.normal(0.0, model['tilt_frac'] * model['notional'], nl)
    df = model['t_df']
    scale = math.sqrt(df / (df - 2.0))
    common = rng.standard_t(df, size=R * T)
    z = (math.sqrt(model['rho']) * common[day_of]
         + math.sqrt(1 - model['rho']) * rng.standard_t(df, size=nl))
    pnl = np.bincount(day_of, weights=w * z * model['vol'] / scale,
                      minlength=R * T)
    return (pnl.reshape(R, T) / model['equity'] * 1e4), np.cumsum(k, axis=1)


def selftest(cfg, reps=1000, seed=20260927, mus=None, horizon=None,
             model=SELFTEST_MODEL):
    """Per-drift fire rates of each sequential test and the verdict mix at
    the sheet's parameters. mus: [(label, mu)] (or bare floats); mu is the
    drift of d^lo in bp/day (c^lo = 0); d^hi = d^lo - 1e4*c_hi_usd/equity.
    Default scenarios: the three null boundaries (KEEP at mu=c_hi, HARM at
    0, FUTILITY at delta) and +-1 bp/day. Returns results keyed by label."""
    cfg = dict(cfg)
    if horizon is not None:
        cfg['horizon'] = int(horizon)
    c_hi_bp = 1e4 * cfg['c_hi_usd'] / model['equity']
    if mus is None:
        mus = [('keep_null', c_hi_bp), ('harm_null', 0.0),
               ('futility_null', cfg['delta']), ('plus1', 1.0), ('minus1', -1.0)]
    mus = [m if isinstance(m, tuple) else (f"mu={m:g}", m) for m in mus]
    rng = np.random.default_rng(seed)
    T = cfg['burn_days'] + cfg['horizon']
    out = {}
    for label, mu in mus:
        noise, lots = synth_noise(rng, reps, T, model)
        d_lo = mu + noise
        res = run_ledger_batch(d_lo, d_lo - c_hi_bp, lots, cfg)
        fr = {nm: float(np.mean(res['first'][nm] > 0)) for nm in res['first']}
        fr_i = {nm: float(np.mean((res['first'][nm] > 0) &
                                  (res['first'][nm] <= cfg['interim_day'])))
                for nm in res['first']}
        vals, cnt = np.unique(res['verdict'].astype(str), return_counts=True)
        med = {nm: (int(np.median(res['first'][nm][res['first'][nm] > 0]))
                    if (res['first'][nm] > 0).any() else None)
               for nm in res['first']}
        out[label] = {
            'mu_bp': float(mu), 'fire_by_horizon': fr, 'fire_by_interim': fr_i,
            'median_fire_cs_day': med,
            'verdicts': {str(v): float(c) / reps for v, c in zip(vals, cnt)},
            'clip_rate_low': float(np.mean(res['rate_low_keep'][:, -1])),
            'clip_rate_up': float(np.mean(res['rate_up_kill'][:, -1])),
            'B_median_bp': float(np.median(res['B'])),
            'sd_d_median_bp': float(np.median(d_lo.std(axis=1))),
        }
    return {'reps': reps, 'seed': seed, 'horizon': cfg['horizon'],
            'c_hi_bp': c_hi_bp, 'model': dict(model), 'by_mu': out}


def _print_selftest(res, cfg):
    R = res['reps']
    se = math.sqrt(0.025 * 0.975 / R)
    print(f"llm_eprocess --selftest  R={R} seed={res['seed']} horizon="
          f"{res['horizon']} (interim {cfg['interim_day']}) threshold="
          f"{cfg['threshold']:g} delta={cfg['delta']} bp/day c_hi="
          f"{res['c_hi_bp']:.3f} bp/day  MC s.e. at 2.5%: {100*se:.2f}pp")
    print("scenario        mu   KEEP%  HARM%  FUTIL%  (by interim K/H/F)  "
          "median day K/H/F  clip lo/up%  verdicts")
    for label, r in res['by_mu'].items():
        f, fi, md = r['fire_by_horizon'], r['fire_by_interim'], r['median_fire_cs_day']
        vd = ' '.join(f"{k}={100*v:.0f}%" for k, v in sorted(r['verdicts'].items()))
        print(f"{label:13s} {r['mu_bp']:5.2f} {100*f['KEEP']:6.1f} {100*f['KILL_HARM']:6.1f} "
              f"{100*f['KILL_FUTILITY']:6.1f}   ({100*fi['KEEP']:.1f}/"
              f"{100*fi['KILL_HARM']:.1f}/{100*fi['KILL_FUTILITY']:.1f})   "
              f"{md['KEEP']}/{md['KILL_HARM']}/{md['KILL_FUTILITY']}   "
              f"{100*r['clip_rate_low']:.1f}/{100*r['clip_rate_up']:.1f}   {vd}")
    print("size rows: KEEP at mu=c_hi, HARM at mu=0, FUTILITY at mu=delta must "
          f"be <= {100*0.025:.1f}% (+MC s.e.)")


# ---------------------------------------------------------------------------
# Live mode (refuses unless signed)
# ---------------------------------------------------------------------------
def _live_fee_fn(asset, row):
    import fees
    sp = row.get('spread_pct')
    if sp is None:
        sp = fees.FLAT_SPREAD_PCT.get(asset, 0.10)
    return fees.round_trip_cost_pct(asset, spread_pct=float(sp),
                                    maker=bool(row.get('maker')))


def _live_marks(lots, first_day, last_day):
    """{symbol: {date: last hourly close <= 24:00 UTC}} from read-only Alpaca
    bars via llm_eval._bars_lookup (stocks: RTH bars only)."""
    import pandas as pd
    from llm_eval import _bars_lookup
    from trading_utils import get_api
    api = get_api()
    t0 = _dt.datetime.combine(first_day - _dt.timedelta(days=5), _dt.time(),
                              tzinfo=_dt.timezone.utc)
    t1 = _dt.datetime.combine(last_day + _dt.timedelta(days=1), _dt.time(),
                              tzinfo=_dt.timezone.utc)
    out = {}
    for sym, asset in sorted({(l['symbol'], l['asset']) for l in lots}):
        ts, cl = _bars_lookup(api, sym, asset, t0, t1)
        if not len(ts):
            continue
        s = pd.Series(cl, index=pd.to_datetime(ts, unit='s', utc=True))
        if asset == 'stock':
            ny = s.index.tz_convert('America/New_York')
            mins = ny.hour * 60 + ny.minute
            s = s[(mins >= 9 * 60 + 30) & (mins < 16 * 60)]
        s = s.groupby(s.index.date).last()
        out[sym] = {d: float(v) for d, v in s.items()}
    return out


def _live_equity_prev(days):
    """{day: equity at the latest daily stamp strictly before day} from
    beta_ledger.load_equity_alpaca (beta_ledger.py:692)."""
    from beta_ledger import load_equity_alpaca
    eq = load_equity_alpaca(len(days) + 10)
    out = {}
    for d in days:
        prior = eq[[ix.date() < d for ix in eq.index]]
        if len(prior):
            out[d] = float(prior.iloc[-1])
    return out


def ledger_report(cfg, days, series, m_bar):
    """Assemble the per-day JSON body from a single-path ledger run."""
    res = run_ledger_batch(series['d_lo'][None, :], series['d_hi'][None, :],
                           series['lots_cum'][None, :], cfg)
    nb = cfg['burn_days']
    B = float(res['B'][0])
    L, U = hedged_cs(res['x_lo'][0], cfg['cs_alpha'], cfg['cs_theta'],
                     cfg['cs_c'], cfg['cs_grid'], cfg['mu_prior'], cfg['var_prior'])
    rows = []
    for j, day in enumerate(days):
        r = {'date': day.isoformat(), 'cs_day': max(0, j + 1 - nb),
             'd_lo_bp': round(float(series['d_lo'][j]), 6),
             'd_hi_bp': round(float(series['d_hi'][j]), 6),
             'lots_cum': int(series['lots_cum'][j])}
        if j >= nb:
            k = j - nb
            r.update({'E_keep': float(res['E_keep'][0, k]),
                      'E_harm': float(res['E_harm'][0, k]),
                      'E_futility': float(res['E_fut'][0, k]),
                      'cs_lo_bp': round(float(L[k]) * 2 * B - B, 6),
                      'cs_hi_bp': round(float(U[k]) * 2 * B - B, 6)})
        rows.append(r)
    dday = int(res['decision_cs_day'][0])
    return {'verdict': str(res['verdict'][0]), 'verdict_basis': str(res['basis'][0]),
            'decision_cs_day': dday or None,
            'decision_date': rows[nb + dday - 1]['date'] if dday else None,
            'interim_status_reached': len(days) - nb >= cfg['interim_day'],
            'B_bp': B, 'm_bar': m_bar, 'days': rows}


def run_live(args, sheet, cfg) -> int:
    today = _dt.datetime.now(_dt.timezone.utc).date()
    last_day = today - _dt.timedelta(days=1)            # complete days only
    first_day = last_day - _dt.timedelta(days=args.days - 1)
    rows, hashes = load_journal_rows(args.journals, first_day, last_day,
                                     cfg['exclude_through_utc'], cfg['not_before_utc'])
    buys = [e for e in rows if e.get('action') == 'buy']
    if not buys:
        print("llm_eprocess: no eligible buy rows after the legacy exclusion "
              "/ start rule — nothing to measure", file=sys.stderr)
        return EXIT_DATA
    first_day = buys[0]['_t'].date()                     # Row 1: first buy
    days = [first_day + _dt.timedelta(days=i)
            for i in range((last_day - first_day).days + 1)]
    lots = build_lots(rows, _live_fee_fn)
    m_bar = burn_in_m_bar(lots, first_day, cfg['burn_days'])
    if m_bar is None or len(days) <= cfg['burn_days']:
        print("llm_eprocess: still in burn-in — no test yet", file=sys.stderr)
        return EXIT_DATA
    try:
        marks = _live_marks(lots, first_day, last_day)
        eq_prev = _live_equity_prev(days)
        costs = daily_costs(rows)
        kw = dict(entry_share=cfg['fee_entry_share'], exit_share=cfg['fee_exit_share'])
        prim = daily_tilt_series(lots, days, marks, eq_prev, costs,
                                 cfg['c_hi_usd'], m_bar, **kw)
        sec = daily_tilt_series(lots, days, marks, eq_prev, costs,
                                cfg['c_hi_usd'], cfg['m_bar_secondary'], **kw)
    except (KeyError, ImportError, OSError, ValueError) as e:
        print(f"llm_eprocess: data error: {e!r}", file=sys.stderr)
        return EXIT_DATA
    body = ledger_report(cfg, days, prim, m_bar)
    body['secondary_m_bar_1'] = {'mean_d_lo_bp': float(np.mean(sec['d_lo'])),
                                 'd_lo_bp': [round(float(v), 6) for v in sec['d_lo']]}
    report = {'module': 'llm_eprocess', 'measurement_only': True,
              'read_by': 'nothing in the repo',
              'generated_utc': _dt.datetime.now(_dt.timezone.utc).isoformat(),
              'params_path': str(args.params),
              'registration_id': sheet.get('registration_id'),
              'registration_sha': sheet.get('registration_sha'),
              'input_sha256': hashes,
              'veto_clause': sheet['veto']['value'] + ' (not evaluated here)',
              **body}
    _write_json(args.json or DEFAULT_LIVE_OUT, report)
    print(f"llm_eprocess: verdict={report['verdict']} ({report['verdict_basis']}) "
          f"-> {args.json or DEFAULT_LIVE_OUT}")
    return EXIT_OK


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, default=str)
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--selftest', action='store_true',
                    help='synthetic size/power at the sheet parameters')
    ap.add_argument('--journals', help='journal directory (live mode)')
    ap.add_argument('--days', type=int, help='live window in UTC days')
    ap.add_argument('--params', default=str(DEFAULT_PARAMS_PATH))
    ap.add_argument('--json', help='output JSON path (live default '
                                   'logs/llm_eprocess_report.json; selftest: none)')
    ap.add_argument('--reps', type=int, default=1000, help='selftest paths per drift')
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--print-registration-sha', action='store_true',
                    help='print sha256 of the sheet (for the owner signing it); '
                         'writes nothing')
    args = ap.parse_args(argv)

    if args.print_registration_sha:
        print(registration_sha(load_sheet(args.params)))
        return EXIT_OK
    if args.selftest:
        try:
            sheet, src = load_sheet(args.params), args.params
        except (OSError, json.JSONDecodeError):
            sheet, src = copy.deepcopy(DEFAULT_SHEET), 'in-code DEFAULT_SHEET'
        cfg = config_from_sheet(sheet)
        t0 = time.time()
        res = selftest(cfg, reps=args.reps, seed=args.seed)
        res['params_source'] = str(src)
        res['signed'] = signature_status(sheet)[0]
        _print_selftest(res, cfg)
        print(f"({time.time() - t0:.1f}s)")
        if args.json:
            _write_json(args.json, res)
        return EXIT_OK
    if not args.journals or not args.days or args.days < 1:
        ap.print_usage(sys.stderr)
        print("need --selftest, or --journals DIR --days N", file=sys.stderr)
        return EXIT_USAGE
    try:
        sheet = load_sheet(args.params)
    except (OSError, json.JSONDecodeError) as e:
        print(f"llm_eprocess: REFUSED (NOT_SIGNED): cannot read params {e}",
              file=sys.stderr)
        return EXIT_NOT_SIGNED
    ok, why = signature_status(sheet)
    if not ok:
        print(f"llm_eprocess: REFUSED (NOT_SIGNED): {why}. The live ledger runs "
              f"only on an owner-signed pre-registration sheet ({args.params}).",
              file=sys.stderr)
        if args.json:
            _write_json(args.json, {'module': 'llm_eprocess', 'verdict': 'NOT_SIGNED',
                                    'reason': why, 'measurement_only': True})
        return EXIT_NOT_SIGNED
    try:
        cfg = config_from_sheet(sheet)
    except ValueError as e:
        print(f"llm_eprocess: REFUSED (NOT_SIGNED): invalid sheet: {e}", file=sys.stderr)
        return EXIT_NOT_SIGNED
    return run_live(args, sheet, cfg)


if __name__ == '__main__':
    sys.exit(main())
