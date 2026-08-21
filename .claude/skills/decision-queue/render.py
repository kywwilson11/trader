#!/usr/bin/env python3
"""Render the 2026-07 module-review owner decision queue (stdlib only — Mac-safe).

Reads research/module_review_2026-07.json -> decision_queue_p0_p2_not_autofixed
(items: severity/module/where/desc/fix_sketch). These were deliberately NOT auto-fixed
(model-facing / behavior-changing) — they are OWNER decisions, not a to-do list.
"""
import argparse
import json
import sys
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "research" / "module_review_2026-07.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="path to the review ledger JSON")
    ap.add_argument("--severity", choices=["P0", "P1", "P2"], default=None)
    ap.add_argument("--module", default=None, help="substring filter on module name")
    ap.add_argument("--full", action="store_true", help="also show where + fix_sketch")
    args = ap.parse_args()

    try:
        data = json.loads(Path(args.ledger).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"cannot read ledger {args.ledger}: {exc}")

    queue = data.get("decision_queue_p0_p2_not_autofixed", [])
    items = [x for x in queue
             if (not args.severity or x.get("severity") == args.severity)
             and (not args.module or args.module.lower() in x.get("module", "").lower())]
    order = {"P0": 0, "P1": 1, "P2": 2}
    items.sort(key=lambda x: (order.get(x.get("severity", "P2"), 9), x.get("module", "")))

    counts = {}
    for x in queue:
        counts[x.get("severity", "?")] = counts.get(x.get("severity", "?"), 0) + 1
    print(f"module-review decision queue (generated {data.get('generated', '?')}) — "
          f"{len(queue)} owner-decision items: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"showing {len(items)} after filters. Deliberately NOT auto-fixed "
          f"(model-facing / behavior-changing) — the owner decides.\n")

    last = None
    for x in items:
        sev = x.get("severity", "?")
        if sev != last:
            print(f"== {sev} ==")
            last = sev
        print(f"[{sev}] {x.get('module', '?')}: {(x.get('desc') or '').strip()}")
        if args.full:
            if x.get("where"):
                print(f"      where: {x['where']}")
            if x.get("fix_sketch"):
                print(f"      fix:   {x['fix_sketch']}")
    if not items:
        print("(no items match the filters)")


if __name__ == "__main__":
    main()
