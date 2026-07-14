"""Crypto squeeze INTERACTION features (wave-7, Finding 10 — the cheap half).

Funding_Z, OI_Z, taker imbalance, top-trader L/S are fed to the model as
INDEPENDENT raw z-scores, but the squeeze mechanism is an INTERACTION: extreme
funding is only dangerous when OPEN INTEREST is high (lots of leverage to
unwind), and a crowded-short setup only resolves violently when that leverage
is actually there. Leaving a tree to rediscover this product from a starved
panel is wasteful — give it the interaction explicitly.

Two point-in-time, neutral-filled columns (0.0 for any missing input so the
harvest dropna never silently drops bars):

  Funding_x_OI   = Funding_Z * clip(OI_Z, 0, +inf)
                   funding stress amplified by (only) high open interest
  Squeeze_Setup  = 1{Funding_Z < -2} * clip(OI_Z, 0, +inf)
                   very negative funding (crowded SHORTS paying longs) with the
                   open interest to fuel a short squeeze (price-up risk)

This is the FEATURE half INTENDED to ship into harvest_crypto_data — that
wiring is NOT yet done (the columns have no consumers outside this module; see
the wave-8 activation backlog: orphaned/zero consumers) — once wired it lets
the model learn the interaction. The directional squeeze-timing EVENT-STUDY
(gated by net expectancy on a purged-CV holdout, per the wave-5 discipline) is
a separate research script that needs the re-fetched funding/OI archives.
PREFER the model learning from this column over any hardcoded squeeze
threshold.
"""

import numpy as np

SQUEEZE_FUNDING_Z = -2.0   # "crowded shorts" trigger on the funding z-score


def squeeze_interaction(funding_z, oi_z):
    """Return {'Funding_x_OI', 'Squeeze_Setup'} as numpy arrays.

    Inputs are per-bar Funding_Z / OI_Z (arrays or Series). NaNs are treated as
    neutral (0.0) so the outputs are always finite and never drop a bar.
    """
    fz = np.nan_to_num(np.asarray(funding_z, dtype=float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    oz = np.nan_to_num(np.asarray(oi_z, dtype=float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    oi_pos = np.clip(oz, 0.0, None)            # only HIGH open interest amplifies
    funding_x_oi = fz * oi_pos
    squeeze_setup = (fz < SQUEEZE_FUNDING_Z).astype(float) * oi_pos
    return {'Funding_x_OI': funding_x_oi, 'Squeeze_Setup': squeeze_setup}
