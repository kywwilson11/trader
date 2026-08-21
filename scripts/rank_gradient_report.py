"""Rank-gradient Stage-0 gate (wave-9 #4/#5 activation harness).

HOLDOUT side: dump a predictions frame (rows {ts, symbol, signal, fwd_return},
one per symbol-bar over the universe) to JSON/CSV, then:
    python scripts/rank_gradient_report.py --preds preds.json
The dump is produced by backtest.py (B02, default ON): {prefix}stage0_preds.json
— per-(symbol, bar) rows with both 'pred' and 'signal' keys, PERCENT units,
non-overlapping stride = the label horizon (scripts/ic_by_name.py reads the
same file; backtest prints both consumers' invocations). UNITS: signal and fwd_return are
PERCENT, matching harvest Target_Return_* and decision_report's mean_net_pct;
--cost-pct is subtracted raw from fwd_return, so a fractional-return dump with
a percent cost would silently corrupt the bucket means.

LIVE side: feed decision_report's rank buckets directly:
    python scripts/rank_gradient_report.py --buckets decision_report.json

PASS on BOTH the holdout panel AND >=20-30d of live journals before shipping the
conviction flagship (CONCENTRATION_ENABLED) or edge-Kelly (EDGE_KELLY_ENABLED).
Exit status: 0 only when the verdict CONFIRMS the gradient, 1 on a ran-but-no-go
verdict, and 2 when the --buckets input cannot be gated at all (a STALE
decision_report placeholder, or a payload that is not a JSON object) — so
scripted use can tell "gate ran and said no" apart from both "gate confirmed"
and "input unusable / not trustworthy".

--strict enforces the anti-noise verdict (per-bucket n >= rank_gradient.MIN_BUCKET_N
and the rank_1_3 ci90 excluding the rank_6_7 mean) — use it for any real
ship/no-ship decision. --fwd-bars declares the dump's forward-return horizon
so the holdout ci90 is overlap-widened (n_eff = n/fwd_bars); it does not
apply to --buckets (live trade buckets are non-overlapping trades).
--extra-cols 'meta_p,pred_thresh_ratio' carries the conviction fields into
the panel so the SAME dump also feeds portfolio_backtest's
conviction_gated(strict=True) A/B, whose strict floors FAIL on absent
fields.
"""
import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rank_gradient import (MIN_BUCKET_N, rank_gradient_from_panel,  # noqa: E402
                           rank_gradient_verdict)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--preds', help='predictions frame JSON/CSV (ts, symbol, signal, fwd_return)')
    src.add_argument('--buckets', help='decision_report JSON with rank_1_3/rank_6_7')
    ap.add_argument('--signal-lag', type=int, default=0)
    ap.add_argument('--cost-pct', type=float, default=0.0,
                    help='flat per-trade cost in the SAME units as fwd_return '
                         '(repo convention: percent, cf. fees.round_trip_cost_pct)')
    ap.add_argument('--fwd-bars', type=int, default=1,
                    help='forward-return horizon of the dump in bars; >1 widens '
                         'the holdout ci90 for overlapping samples '
                         '(n_eff = n/fwd_bars). --preds only.')
    ap.add_argument('--extra-cols', default='',
                    help="comma-separated extra columns to carry into the panel "
                         "(e.g. 'meta_p,pred_thresh_ratio' when the dump also "
                         "feeds the conviction A/B — portfolio_backtest's "
                         "strict floors FAIL on absent fields). --preds only.")
    ap.add_argument('--strict', action='store_true',
                    help=f'anti-noise verdict: require bucket n >= {MIN_BUCKET_N} '
                         'and the rank_1_3 ci90 to exclude the rank_6_7 mean '
                         'before CONFIRMED (use for any ship/no-ship decision)')
    args = ap.parse_args()

    if args.preds:
        import pandas as pd
        from portfolio_backtest import panel_from_frame
        p = Path(args.preds)
        df = (pd.read_csv(p) if p.suffix.lower() == '.csv'
              else pd.DataFrame(json.loads(p.read_text())))
        df = df.set_index(pd.to_datetime(df['ts']))
        # The documented frame names the ticker column 'symbol' (shared with
        # ic_by_name); tolerate a repo-internal 'Ticker'-style dump too.
        tcol = 'symbol' if 'symbol' in df.columns else 'Ticker'
        extra = [c.strip() for c in args.extra_cols.split(',') if c.strip()]
        panel = panel_from_frame(df, 'signal', 'fwd_return', ticker_col=tcol,
                                 signal_lag=args.signal_lag,
                                 extra_cols=extra or None)
        buckets = rank_gradient_from_panel(panel, cost_pct=args.cost_pct,
                                           fwd_bars=args.fwd_bars)
    else:
        buckets = json.loads(Path(args.buckets).read_text())
        if not isinstance(buckets, dict):
            # A list/scalar payload would AttributeError on .get() below; name
            # the shape mismatch instead of dumping a traceback.
            print(f"ERROR: {args.buckets} must be a JSON object (a decision_report "
                  f"or a bare rank-bucket dict); got {type(buckets).__name__}",
                  file=sys.stderr)
            return 2
        # decision_report.py writes a placeholder marked 'stale': True whenever it
        # could not price counterfactuals (no reachable API, or an empty journal),
        # and its docstring names THIS script as the consumer meant to "refuse to
        # trust it". Do so explicitly: a stale report has no rank buckets, so the
        # generic path would misreport it as merely 'insufficient rank coverage'
        # and exit 1 — indistinguishable from a real ran-but-no-gradient verdict.
        if buckets.get('stale'):
            print(f"REFUSED: {args.buckets} is a STALE decision_report "
                  f"(stale=true, api_available={buckets.get('api_available')!r}) — "
                  f"it carries no live rank buckets and cannot gate the "
                  f"conviction / edge-Kelly levers. Re-run decision_report.py on "
                  f"the Jetson with a reachable API and >=20-30d of journals, then "
                  f"gate on the fresh report.", file=sys.stderr)
            return 2
        # Same unwrap as rank_gradient_verdict: a full decision_report.json
        # nests the rank buckets under 'conviction' — without this, the
        # per-bucket evidence lines below silently print nothing.
        if 'rank_1_3' not in buckets and isinstance(buckets.get('conviction'), dict):
            buckets = buckets['conviction']

    v = (rank_gradient_verdict(buckets, min_bucket_n=MIN_BUCKET_N,
                               require_ci=True)
         if args.strict else rank_gradient_verdict(buckets))
    print("\n=== Rank-gradient Stage-0 ===")
    for b in ('rank_1_3', 'rank_4_5', 'rank_6_7'):
        if b in buckets:
            line = (f"  {b}: mean_net {buckets[b]['mean_net_pct']}  "
                    f"(n={buckets[b].get('n')})")
            if buckets[b].get('ci90') is not None:
                line += f" ci90={buckets[b]['ci90']}"
            print(line)
    print(f"  ratio 6-7 / 1-3: {v.get('ratio_6_7_over_1_3')}")
    print(f"\n  VERDICT: {v['verdict']}\n")
    return 0 if v.get('gradient_exists') else 1


if __name__ == '__main__':
    raise SystemExit(main())
