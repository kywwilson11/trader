"""Offline CSCV Probability-of-Backtest-Overfitting audit (wave-8 #2).

The promotion gate today is a single ~12% holdout + Deflated Sharpe. Per Bailey,
Borwein, Lopez de Prado & Zhu (2017), a single holdout "cannot assess
representativeness" once hundreds of configs are searched. validation.pbo_cscv()
implements the honest combinatorially-symmetric estimator but is unwired; this
harness runs it offline over a hypersearch run's per-trial out-of-sample returns.

HONEST SCOPE: this is FORWARD-LOOKING. No saved study currently records the
per-trial OOS return stream (the champion's trade_returns is persisted, but not
every trial's). To use this you first add, inside the Optuna objective on the
Jetson:

    blocks = validation.build_oos_blocks(trial_trade_returns, 8)
    trial.set_user_attr('oos_block_perf',
                        None if blocks is None else blocks.tolist())

(build_oos_blocks returns None for trials with fewer than 8 finite trade
returns — routine for degenerate configs in a TPE search; JSON round-trips it
as null and pbo_from_oos_blocks filters None rows, so no audit-side change.)

then dump those vectors to JSON and run:

    python scripts/cscv_audit.py --blocks trial_blocks.json
    python scripts/cscv_audit.py --returns trial_returns.json --n-blocks 8

Input JSON is a list (one entry per trial) of either pre-built block vectors
(--blocks) or raw per-trade return arrays (--returns). PBO > 0.5 (Bailey et al.
reject threshold; AFML uses 0.5 as the practical floor) means in-sample selection
did not carry out of sample — the search is overfit.
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import validation  # noqa: E402


def audit(block_rows, n_groups: int = 8) -> dict | None:
    return validation.pbo_from_oos_blocks(block_rows, n_groups=n_groups)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--blocks', help='JSON list of per-trial block vectors')
    src.add_argument('--returns', help='JSON list of per-trial raw return arrays')
    ap.add_argument('--n-blocks', type=int, default=8,
                    help='blocks per trial when binning raw returns (default 8)')
    ap.add_argument('--n-groups', type=int, default=8,
                    help='CSCV symmetric groups, even (default 8)')
    ap.add_argument('--pbo-max', type=float, default=0.5,
                    help='reject threshold for the printed verdict (default 0.5)')
    args = ap.parse_args()

    if args.n_groups < 2 or args.n_groups % 2 != 0:
        ap.error('--n-groups must be an even integer >= 2 (pbo_cscv splits '
                 'the groups into two symmetric halves)')
    if args.returns is not None and args.n_blocks % args.n_groups != 0:
        ap.error('--n-blocks must be divisible by --n-groups, else pbo_cscv '
                 'would silently drop the trailing blocks of every trial')

    if args.blocks is not None:
        block_rows = json.loads(Path(args.blocks).read_text())
    else:
        raw = json.loads(Path(args.returns).read_text())
        block_rows = [validation.build_oos_blocks(r, args.n_blocks) for r in raw]

    res = audit(block_rows, n_groups=args.n_groups)
    if res is None:
        print("PBO: n/a — too few valid trials / blocks to judge (gate stays on "
              "DSR). Need >=2 trials each with >= n_groups finite, non-constant "
              "block contributions, all with the modal block width, and that "
              "width divisible by n_groups (rows of any other length are "
              "dropped).")
        return 0
    print("\n=== CSCV Probability of Backtest Overfitting ===")
    print(f"  PBO           : {res['pbo']:.3f}  (fraction of {res['n_splits']} "
          f"symmetric splits where the IS-best config was at or below the OOS "
          f"median)")
    print(f"  median logit  : {res['median_logit']:+.3f}  (>0 = selection persists OOS)")
    print(f"  mean OOS rank : {res['mean_oos_rank']:.3f}")
    verdict = ("OVERFIT — in-sample selection does not generalize; tighten the "
               "search / parsimony" if res['pbo'] > args.pbo_max
               else "acceptable — selection carries out of sample")
    print(f"  verdict       : {verdict}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
