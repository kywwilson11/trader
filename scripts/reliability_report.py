"""Calibration before/after gate (wave-9 #1 activation harness).

On the Jetson, run the meta model BOTH ways on the holdout — legacy same-slice
isotonic vs the new purged out-of-fold calibrator — dump {p_legacy, p_purged, y}
arrays to JSON, then:

    python scripts/reliability_report.py --in calib_holdout.json

PURPOSE: decide whether to flip META_CALIBRATION_MODE='purged_oof'. Flip ONLY if
the purged calibrator is at least as well-calibrated (lower-or-equal Brier AND
ECE) on the HOLDOUT — never on a same-slice score, never on an exact tie (equal
metrics carry zero evidence), and re-certify in shadow before enabling any
p-consuming sizing lever (edge-Kelly, conviction tiers).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from calibration import brier, compare_calibrations, expected_calibration_error  # noqa: E402

REQUIRED_KEYS = ('p_legacy', 'p_purged', 'y')


def _label(before, after):
    """Tri-state per-metric label in the same criterion family as the verdict.

    compare_calibrations green-lights on lower-or-EQUAL, so a two-state
    'better'/'no gain' label contradicts the verdict on exact ties (both
    metrics read 'no gain' while the verdict says safe to flip)."""
    if before is None or after is None:
        return 'n/a'
    if after < before:
        return 'better'
    return 'tie (no worse)' if after == before else 'worse'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', required=True,
                    help='JSON file with p_legacy, p_purged, y arrays')
    ap.add_argument('--bins', type=int, default=10)
    args = ap.parse_args()
    if args.bins < 2:
        ap.error('--bins must be >= 2')

    d = json.loads(Path(args.inp).read_text())
    if not isinstance(d, dict) or any(k not in d for k in REQUIRED_KEYS):
        have = sorted(d) if isinstance(d, dict) else type(d).__name__
        print(f"ERROR: {args.inp} must be a JSON object with "
              f"{list(REQUIRED_KEYS)} arrays (have: {have})", file=sys.stderr)
        return 2
    p_legacy, p_purged, y = d['p_legacy'], d['p_purged'], d['y']
    if not (len(p_legacy) == len(p_purged) == len(y)):
        print(f"ERROR: array length mismatch in {args.inp}: "
              f"p_legacy={len(p_legacy)}  p_purged={len(p_purged)}  y={len(y)}",
              file=sys.stderr)
        return 2

    identical = bool(np.allclose(np.asarray(p_legacy, float),
                                 np.asarray(p_purged, float), equal_nan=True))
    if identical:
        print("WARNING: p_legacy and p_purged are (near-)identical — the dump "
              "did not actually run the calibrator both ways; this holdout "
              "carries zero comparative evidence.", file=sys.stderr)

    rep = compare_calibrations(p_legacy, p_purged, y, n_bins=args.bins)
    # Unrounded metrics via the SAME functions the gate uses — rep's values are
    # display-rounded, and the tie/label decisions must match the <= criterion
    # compare_calibrations actually gated on.
    y_arr = np.asarray(y, float)
    bl, bp = brier(p_legacy, y_arr), brier(p_purged, y_arr)
    el = expected_calibration_error(p_legacy, y_arr, args.bins)
    ep = expected_calibration_error(p_purged, y_arr, args.bins)
    tied = identical or (bp == bl and el is not None and ep == el)

    verdict = rep['verdict']
    if tied:
        verdict = ('tied — no evidence on this holdout; do NOT flip '
                   'META_CALIBRATION_MODE (collect a discriminating dump)')

    print(f"\n=== Meta-label calibration: legacy vs purged-OOF (n={rep['n']}) ===")
    print(f"  Brier  legacy {rep['brier_legacy']}  ->  purged {rep['brier_purged']}  "
          f"({_label(bl, bp)})")
    print(f"  ECE    legacy {rep['ece_legacy']}  ->  purged {rep['ece_purged']}  "
          f"({_label(el, ep)})")
    print("  reliability (predicted -> observed), purged:")
    for b in rep['reliability_purged']:
        if b['n']:
            print(f"    bin {b['bin']}: pred {b['pred_mean']:>6}  obs {b['obs_freq']:>6}  (n={b['n']})")
    print(f"\n  VERDICT: {verdict}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
