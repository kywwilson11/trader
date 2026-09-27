#!/usr/bin/env python3
"""HAR size audit for llm_eval's keep/kill b2 test (measurement-only).

Question: at the live panel geometry, how often does each b2 p-value reject a
TRUE null (b2 = 0) at nominal 5 %? llm_eval's verdict carrier is
Driscoll-Kraay with Bartlett lag forward_bars-1 judged against t_{G-1}; the
INTEL 2026-09 Scout B simulation measured its size at 12.4-13.8 % (K=6,
fb=24, n_eff 20/30/60, AR(1) 0.98). This script re-measures that with the
estimators IMPORTED from llm_eval (no copies) — it calls
llm_eval.compute_incremental_report on each synthetic panel, i.e. the exact
production path, and reads four p-values from its 'encompassing' block:

  dk_prod   enc['p_value']         DK, lag fb-1, t_{G-1}   (the verdict carrier)
  dk_fixedb enc['b2_dk_fixedb_p']  DK, M=max(1.3*sqrt(T), 2*fb), KV-2005 fixed-b
  ewc       enc['b2_ewc_p']        EWC, nu=floor(0.4*T^(2/3)), t_nu (LLSW 2018)
  im        enc['b2_im_p']         Ibragimov-Mueller K=8 blocks, t_{m-1}

Synthetic null (stated): T = n_eff*fb hourly t0 clusters, K names per
cluster. Hourly returns r = sqrt(.5)*common + sqrt(.5)*idio (iid N(0,1));
realized = the overlapping fb-bar forward sum of r (an MA(fb-1) overlap, the
live geometry). pred ~ AR(1) with phi (default 0.9) per name, unit variance;
s = AR(1)(phi) + 0.3*pred — both independent of realized, so the TRUE b2 = 0
(and b1 = 0). None p-values (e.g. fixed-b with b > 1 at tiny T) count as
non-rejections and are tallied separately.

PRE-REGISTERED RULE: an estimator is fit to carry the keep/kill verdict iff
its measured size at n_eff=20 is within [3%, 7%] at nominal 5% (MC s.e. ~1%).
(Measurement only — which estimator carries the verdict is owner ask #1.)

Usage:
    python scripts/har_size_audit.py                       # K=6 fb=24 n_eff 20,30,60 reps 500
    python scripts/har_size_audit.py --reps 200 --json out.json
    python scripts/har_size_audit.py --check-kv            # re-check the KV-2005 polynomials
Exit 0 on success (2 = bad args, from argparse).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_eval as L  # noqa: E402  (reused estimators; no copies)

ESTIMATORS = (('dk_prod', 'p_value'), ('dk_fixedb', 'b2_dk_fixedb_p'),
              ('ewc', 'b2_ewc_p'), ('im', 'b2_im_p'))
FIT_BAND = (0.03, 0.07)
RULE = ("an estimator is fit to carry the keep/kill verdict iff its measured "
        "size at n_eff=20 is within [3%, 7%] at nominal 5% (MC s.e. ~1%)")


def _ar1(T, K, phi, rng):
    x = np.empty((T, K))
    e = rng.standard_normal((T, K))
    x[0] = e[0]
    c = np.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + c * e[t]
    return x


def synth_samples(T, K, fb, phi, rng):
    """One null panel as llm_eval samples: (s, realized, pred, t0) tuples."""
    tot = T + fb
    f = rng.standard_normal(tot)[:, None]
    r = np.sqrt(0.5) * f + np.sqrt(0.5) * rng.standard_normal((tot, K))
    c = np.cumsum(np.vstack([np.zeros((1, K)), r]), axis=0)
    y = c[fb:fb + T] - c[:T]                     # overlapping fb-bar fwd return
    pred = _ar1(T, K, phi, rng)
    s = _ar1(T, K, phi, rng) + 0.3 * pred
    t0 = 1.7e9 + 3600.0 * np.repeat(np.arange(T), K)
    return list(zip(s.ravel().tolist(), y.ravel().tolist(),
                    pred.ravel().tolist(), t0.tolist()))


def run_cell(n_eff, K, fb, phi, reps, rng):
    T = int(n_eff * fb)
    rej = {k: 0 for k, _ in ESTIMATORS}
    none = {k: 0 for k, _ in ESTIMATORS}
    t_start = time.time()
    for _ in range(reps):
        rep = L.compute_incremental_report(synth_samples(T, K, fb, phi, rng),
                                           forward_bars=fb)
        enc = rep.get('encompassing') or {}
        for k, key in ESTIMATORS:
            p = enc.get(key)
            if p is None:
                none[k] += 1
            elif p < 0.05:
                rej[k] += 1
    size = {k: round(v / reps, 4) for k, v in rej.items()}
    return {'n_eff': n_eff, 'T_clusters': T, 'K': K, 'fb': fb, 'phi': phi,
            'rows': T * K, 'reps': reps, 'size': size, 'n_none': none,
            'mc_se_at_5pct': round(float(np.sqrt(0.05 * 0.95 / reps)), 4),
            'seconds': round(time.time() - t_start, 1)}


def check_kv(reps=40000, T=500, seed=1):
    """Simulate the fixed-b location-model t (Bartlett, M=bT, iid N(0,1)) and
    compare its |t| quantiles with llm_eval._KV2005_BARTLETT_CV."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in (0.1, 0.2, 0.4, 0.7, 1.0):
        M = int(round(b * T))
        w = 1.0 - np.arange(M) / M
        w[1:] *= 2.0
        ts = []
        for _ in range(max(1, reps // 5000)):
            x = rng.standard_normal((5000, T))
            m = x.mean(1)
            e = x - m[:, None]
            F = np.fft.rfft(e, n=2 * T, axis=1)
            ac = np.fft.irfft(F * np.conj(F), n=2 * T, axis=1)[:, :M] / T
            ts.append(np.sqrt(T) * m / np.sqrt(np.maximum(ac @ w, 1e-300)))
        t = np.abs(np.concatenate(ts))
        for q, (a0, a1, a2, a3) in L._KV2005_BARTLETT_CV:
            rows.append({'b': b, 'q': q,
                         'sim': round(float(np.quantile(t, 2 * q - 1)), 3),
                         'poly': round(a0 + a1 * b + a2 * b * b + a3 * b ** 3, 3)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--K', type=int, default=6, help='names per t0 cluster')
    ap.add_argument('--fb', type=int, default=24, help='forward bars (overlap)')
    ap.add_argument('--n-eff', default='20,30,60',
                    help='comma list of n_eff = T/fb cells')
    ap.add_argument('--reps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--phi', type=float, default=0.9,
                    help='AR(1) coefficient of pred and s')
    ap.add_argument('--json', default=None, metavar='PATH')
    ap.add_argument('--check-kv', action='store_true',
                    help='also re-check the KV-2005 fixed-b polynomials by '
                         'simulation (~1 min)')
    a = ap.parse_args(argv)
    try:
        n_effs = [int(x) for x in str(a.n_eff).split(',') if x.strip()]
    except ValueError:
        ap.error('--n-eff must be a comma list of integers')
    if a.K < 1 or a.fb < 1 or a.reps < 1 or not n_effs or min(n_effs) < 1:
        ap.error('--K, --fb, --reps and every --n-eff must be >= 1')
    rng = np.random.default_rng(a.seed)
    cells = [run_cell(ne, a.K, a.fb, a.phi, a.reps, rng) for ne in n_effs]

    print(f"HAR size audit — null b2=0, nominal 5%, K={a.K} fb={a.fb} "
          f"phi={a.phi} reps={a.reps} seed={a.seed}")
    hdr = f"{'n_eff':>5} {'T':>5} " + ' '.join(f"{k:>10}" for k, _ in ESTIMATORS)
    print(hdr)
    for c in cells:
        print(f"{c['n_eff']:>5} {c['T_clusters']:>5} " +
              ' '.join(f"{c['size'][k]:>10.3f}" for k, _ in ESTIMATORS) +
              f"   (MC s.e. {c['mc_se_at_5pct']:.3f}, {c['seconds']}s)")
    fit = None
    c20 = next((c for c in cells if c['n_eff'] == 20), None)
    if c20 is not None:
        fit = {k: FIT_BAND[0] <= c20['size'][k] <= FIT_BAND[1]
               for k, _ in ESTIMATORS}
        print('fit to carry verdict (size@n_eff=20 in [3%,7%]): ' +
              ', '.join(f"{k}={'YES' if v else 'no'}" for k, v in fit.items()))
    print(f"rule: {RULE}")
    out = {'params': {'K': a.K, 'fb': a.fb, 'n_eff': n_effs, 'reps': a.reps,
                      'seed': a.seed, 'phi': a.phi},
           'rule': RULE, 'cells': cells, 'fit_at_n_eff_20': fit}
    if a.check_kv:
        out['kv_check'] = check_kv()
        for r in out['kv_check']:
            print(f"KV b={r['b']} q={r['q']}: sim {r['sim']} poly {r['poly']}")
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"JSON: {a.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
