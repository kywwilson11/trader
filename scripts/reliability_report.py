"""Calibration before/after gate (wave-9 #1 activation harness).

On the Jetson, run the meta model BOTH ways on the holdout — legacy same-slice
isotonic vs the new purged out-of-fold calibrator — dump {p_legacy, p_purged, y}
arrays to JSON, then:

    python scripts/reliability_report.py --in calib_holdout.json

PURPOSE: decide whether to flip META_CALIBRATION_MODE='purged_oof'. Flip ONLY if
the purged calibrator is better (lower Brier AND ECE) on the HOLDOUT — never on a
same-slice score, and re-certify in shadow before enabling any p-consuming sizing
lever (edge-Kelly, conviction tiers).
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from calibration import compare_calibrations  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', required=True,
                    help='JSON file with p_legacy, p_purged, y arrays')
    ap.add_argument('--bins', type=int, default=10)
    args = ap.parse_args()

    d = json.loads(Path(args.inp).read_text())
    rep = compare_calibrations(d['p_legacy'], d['p_purged'], d['y'], n_bins=args.bins)

    print(f"\n=== Meta-label calibration: legacy vs purged-OOF (n={rep['n']}) ===")
    print(f"  Brier  legacy {rep['brier_legacy']}  ->  purged {rep['brier_purged']}  "
          f"({'better' if rep['brier_improved'] else 'no gain'})")
    print(f"  ECE    legacy {rep['ece_legacy']}  ->  purged {rep['ece_purged']}  "
          f"({'better' if rep['ece_improved'] else 'no gain'})")
    print("  reliability (predicted -> observed), purged:")
    for b in rep['reliability_purged']:
        if b['n']:
            print(f"    bin {b['bin']}: pred {b['pred_mean']:>6}  obs {b['obs_freq']:>6}  (n={b['n']})")
    print(f"\n  VERDICT: {rep['verdict']}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
