"""Offline prompt A/B harness for the LLM conviction gate.

The adjudication instrument for EVERY future gate-behavior change to the
LLM analyst (system-prompt swap, pred-blind scoring, rich-context adoption,
model-panel dispersion — see CLAUDE.md task PART B). This script NEVER
touches llm_analysis.json or any live gate state: every analyze_trades()
call it makes passes persist=False, and it never runs in the hot trading
loop.

Usage:
    python scripts/prompt_ab.py run   --days 14 [--asset stock|crypto]
                                      [--system-b FILE] [--hide-pred-b]
                                      [--rich-context-b] [--model MODEL]
                                      [--max-cycles N] [--sleep-sec S]
                                      [--out FILE] [--dry-run]
    python scripts/prompt_ab.py score --in FILE [--min-n 60]

`run` replays cycles captured by llm_analyst._journal_replay (written
whenever replay_capture_enabled, default ON — see journals/llm_replay/)
through TWO prompt variants at the SAME pinned model, and appends
per-symbol scored rows to --out (resumable — skips (t0, symbol) pairs
already present). `dry-run` builds both prompts and prints size estimates
+ one sample pair with ZERO API calls (Mac-testable without provider keys).

`score` realizes forward returns for those rows (Alpaca-gated — Jetson/CI)
and prints the incremental-over-pred verdict for each variant (reusing
llm_eval.compute_incremental_report — the exact statistics the live
scorecard uses) plus the paired A/B comparison and the ADOPT/KEEP/
insufficient_power decision.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

# Single-sourced from trading_utils, same fail-soft pattern as llm_eval.py —
# this harness's veto-rate stat can never drift from the live gate threshold.
try:
    from trading_utils import LLM_VETO_THRESHOLD as VETO_THRESHOLD
except Exception:  # standalone use without the trading stack
    VETO_THRESHOLD = 0.15

MIN_POWER_N_DEFAULT = 60  # mirrors llm_eval.MIN_POWER_N

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIR = REPO_ROOT / "journals" / "llm_replay"
DEFAULT_OUT = REPO_ROOT / "llm_prompt_ab_scores.jsonl"
DEFAULT_REPORT = REPO_ROOT / "llm_prompt_ab_report.json"


# --------------------------------------------------------------------------- #
# Replay loading (pure — no API calls, no Alpaca)
# --------------------------------------------------------------------------- #

def load_replay_cycles(days: int, asset_filter: str | None = None,
                       replay_dir: Path | None = None) -> list[dict]:
    """Load + dedup replay-journal cycles from the last `days` days.

    Dedup key: (ts, asset_type) — a cycle can appear at most once even if
    written by more than one process (e.g. a GUI refresh and a live bot
    landing in the same jsonl). Returns newest-first.

    replay_dir=None resolves REPLAY_DIR at CALL time (not def time) so
    tests/drivers can redirect the module constant.
    """
    if replay_dir is None:
        replay_dir = REPLAY_DIR
    if not replay_dir.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen: dict[tuple, dict] = {}
    for path in sorted(replay_dir.glob("*.jsonl")):
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if asset_filter and rec.get("asset_type") != asset_filter:
                        continue
                    try:
                        ts = datetime.fromisoformat(rec["ts"])
                    except (KeyError, ValueError):
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                    key = (rec["ts"], rec.get("asset_type"))
                    seen[key] = rec
        except OSError:
            continue
    cycles = list(seen.values())
    cycles.sort(key=lambda r: r["ts"], reverse=True)
    return cycles


def load_existing_pairs(out_path: Path) -> set:
    """(t0, symbol) pairs already scored in --out — the resumability set."""
    pairs = set()
    if not out_path.exists():
        return pairs
    try:
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t0, sym = row.get("t0"), row.get("symbol")
                if t0 is not None and sym is not None:
                    pairs.add((t0, sym))
    except OSError:
        pass
    return pairs


# --------------------------------------------------------------------------- #
# Variant-B construction
# --------------------------------------------------------------------------- #

def build_variant_b_candidates(candidates: list[dict], rich_context_b: bool,
                               warn_once: list) -> list[dict]:
    """Copy candidates for variant B.

    --rich-context-b: the 'profile' field is populated only for candidates
    that were ALREADY captured with rich_context_enabled=True live (that
    flag defaults OFF in Part A) — this does not retroactively synthesize
    evidence from data the capture didn't record (raw snapshots aren't
    journaled in Part A; see PART B #3). If no candidate in this cycle
    carries a profile, warn once and variant B == variant A on this axis.
    """
    out = [dict(c) for c in candidates]
    if rich_context_b:
        has_profile = any(c.get("profile") for c in out)
        if not has_profile and not warn_once:
            print("[prompt_ab] --rich-context-b: no captured profile/snapshot "
                 "in this cycle (rich_context_enabled was off at capture "
                 "time) — variant B has nothing to inject this cycle; "
                 "enable rich_context_enabled live for a while first, then "
                 "re-run to get cycles with captured evidence.")
            warn_once.append(True)
    return out


def variant_b_description(system_b: str | None, hide_pred_b: bool,
                          rich_context_b: bool) -> str:
    bits = []
    if system_b:
        bits.append(f"system_b={system_b}")
    if hide_pred_b:
        bits.append("hide_pred")
    if rich_context_b:
        bits.append("rich_context")
    return ",".join(bits) if bits else "none"


# --------------------------------------------------------------------------- #
# run subcommand
# --------------------------------------------------------------------------- #

def _dry_run_preview(cycles: list[dict], hide_pred_b: bool, rich_context_b: bool,
                     system_b_text: str | None):
    from llm_analyst import _build_prompt, _SYSTEM_PROMPT

    if not cycles:
        print("[prompt_ab] --dry-run: no replay cycles found under "
             f"{REPLAY_DIR} — nothing to preview. Write a hand-crafted "
             "journals/llm_replay/<date>.jsonl line to test this path.")
        return

    rec = cycles[0]
    candidates = rec.get("candidates") or []
    candidates_b = build_variant_b_candidates(candidates, rich_context_b, [])
    model_config = {"forward_bars": rec.get("forward_bars", 24)}
    positions = rec.get("positions") or []
    position_details = rec.get("position_details") or {}
    equity = rec.get("equity", 0)
    fng = rec.get("fng")
    asset_type = rec.get("asset_type", "crypto")

    prompt_a = _build_prompt(candidates, asset_type, equity, positions, fng,
                             model_config, position_details=position_details)
    prompt_b = _build_prompt(candidates_b, asset_type, equity, positions, fng,
                             model_config, position_details=position_details,
                             include_pred=not hide_pred_b)
    system_a_text = _SYSTEM_PROMPT
    system_b = system_b_text or _SYSTEM_PROMPT

    def _stats(label, system_text, prompt_text):
        full = system_text + "\n" + prompt_text
        print(f"--- {label} --- chars={len(full)}  ~tokens(est)={len(full) // 4}")

    print(f"[prompt_ab] previewing cycle ts={rec.get('ts')} "
         f"asset={asset_type} n_candidates={len(candidates)}")
    _stats("Variant A", system_a_text, prompt_a)
    _stats("Variant B", system_b, prompt_b)
    print("\n=== Variant A sample prompt (first 2000 chars) ===")
    print(prompt_a[:2000])
    print("\n=== Variant B sample prompt (first 2000 chars) ===")
    print(prompt_b[:2000])
    print("\n[prompt_ab] --dry-run: zero API calls made.")


def cmd_run(args):
    from llm_analyst import analyze_trades
    from llm_client import get_recommended_model

    system_b_text = None
    if args.system_b:
        system_b_text = Path(args.system_b).read_text()

    cycles = load_replay_cycles(args.days, asset_filter=args.asset)
    if args.max_cycles:
        cycles = cycles[:args.max_cycles]
    print(f"[prompt_ab] loaded {len(cycles)} replay cycle(s) "
         f"(days={args.days}, asset={args.asset or 'all'})")

    if args.dry_run:
        _dry_run_preview(cycles, args.hide_pred_b, args.rich_context_b,
                         system_b_text)
        return

    model = args.model or get_recommended_model('analyst')
    variant_b_desc = variant_b_description(args.system_b, args.hide_pred_b,
                                           args.rich_context_b)
    print(f"[prompt_ab] pinned model: {model}  variant_b: {variant_b_desc}")

    out_path = Path(args.out)
    existing = load_existing_pairs(out_path)
    warn_once: list = []
    written = 0

    with open(out_path, "a") as out_f:
        for rec in cycles:
            try:
                ts = datetime.fromisoformat(rec["ts"])
                t0 = ts.timestamp()
            except (KeyError, ValueError):
                continue
            candidates = rec.get("candidates") or []
            if not candidates:
                continue
            asset_type = rec.get("asset_type", "crypto")
            pending = [c for c in candidates
                      if (t0, c.get("symbol")) not in existing]
            if not pending:
                continue  # every symbol in this cycle already scored

            model_config = {"forward_bars": rec.get("forward_bars", 24)}
            positions = rec.get("positions") or []
            position_details = rec.get("position_details") or {}
            equity = rec.get("equity", 0)
            fng = rec.get("fng")
            candidates_b = build_variant_b_candidates(
                candidates, args.rich_context_b, warn_once)

            # Fail-open: a failed cycle (provider outage, bad system-b file,
            # etc.) is logged and skipped — never fatal to the whole run.
            try:
                result_a = analyze_trades(
                    candidates, asset_type, equity=equity,
                    positions=positions, position_details=position_details,
                    fng_value=fng, model_config=model_config,
                    persist=False, model_override=model)
                time.sleep(args.sleep_sec)
                result_b = analyze_trades(
                    candidates_b, asset_type, equity=equity,
                    positions=positions, position_details=position_details,
                    fng_value=fng, model_config=model_config,
                    system_prompt=system_b_text,
                    include_pred=not args.hide_pred_b,
                    persist=False, model_override=model)
            except Exception as e:
                print(f"[prompt_ab] cycle ts={rec.get('ts')} failed "
                     f"(skipped, fail-open): {e}")
                continue
            time.sleep(args.sleep_sec)

            # analyze_trades fails OPEN by returning {} (not raising) — if
            # BOTH variants came back empty, don't write all-None rows:
            # that would mark the (t0, symbol) pairs as scored and make a
            # transient provider outage permanent. Skip -> retryable.
            if not result_a and not result_b:
                print(f"[prompt_ab] cycle ts={rec.get('ts')} returned no "
                     "scores for either variant (provider fail-open) — "
                     "skipped, retryable on next run")
                continue

            for c in candidates:
                sym = c.get("symbol")
                if (t0, sym) in existing:
                    continue
                row = {
                    "symbol": sym,
                    "asset_type": asset_type,
                    "t0": t0,
                    "horizon": rec.get("forward_bars", 24),
                    "pred": c.get("pred_return"),
                    "s_a": (result_a.get(sym) or {}).get("s"),
                    "s_b": (result_b.get(sym) or {}).get("s"),
                    "variant_b_desc": variant_b_desc,
                    "model": model,
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                existing.add((t0, sym))
                written += 1

    print(f"[prompt_ab] wrote {written} new scored row(s) -> {out_path}")


# --------------------------------------------------------------------------- #
# score subcommand
# --------------------------------------------------------------------------- #

def _load_score_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pair_variant_samples(rows: list[dict], realized_tuples: list[tuple],
                         veto_threshold: float = VETO_THRESHOLD) -> dict:
    """Pure: build per-variant (s, realized, pred, t0) sample lists PLUS
    paired A/B stats from scored rows + their realized-return tuples (same
    order/length as `rows` — see llm_eval.realize_scored_rows).

    Returns {'samples_a': [...], 'samples_b': [...], 'paired': {...}}.
    """
    samples_a, samples_b = [], []
    deltas = []
    veto_a = veto_b = flips = n_total = 0
    s_a_list, s_b_list, pred_list = [], [], []

    for row, tup in zip(rows, realized_tuples):
        _s, realized, pred, t0 = tup
        s_a, s_b = row.get("s_a"), row.get("s_b")
        if s_a is not None:
            samples_a.append((s_a, realized, pred, t0))
        if s_b is not None:
            samples_b.append((s_b, realized, pred, t0))
        if s_a is not None and s_b is not None:
            deltas.append(abs(s_b - s_a))
            if (s_a < 0.5) != (s_b < 0.5):
                flips += 1
            s_a_list.append(s_a)
            s_b_list.append(s_b)
            pred_list.append(pred)
        if s_a is not None and s_a < veto_threshold:
            veto_a += 1
        if s_b is not None and s_b < veto_threshold:
            veto_b += 1
        n_total += 1

    paired = {
        "n_paired": len(deltas),
        "mean_abs_delta_s": round(float(np.mean(deltas)), 4) if deltas else None,
        "veto_rate_a": round(veto_a / n_total, 4) if n_total else None,
        "veto_rate_b": round(veto_b / n_total, 4) if n_total else None,
        "n_flips_across_0.5": flips,
    }
    valid = [(a, b, p) for a, b, p in zip(s_a_list, s_b_list, pred_list)
            if p is not None]
    if len(valid) >= 3:
        from scipy.stats import spearmanr
        va = [x[0] for x in valid]
        vb = [x[1] for x in valid]
        vp = [x[2] for x in valid]
        rho_a, _ = spearmanr(va, vp)
        rho_b, _ = spearmanr(vb, vp)
        paired["spearman_s_a_vs_pred"] = (round(float(rho_a), 4)
                                          if rho_a == rho_a else None)
        paired["spearman_s_b_vs_pred"] = (round(float(rho_b), 4)
                                          if rho_b == rho_b else None)
    return {"samples_a": samples_a, "samples_b": samples_b, "paired": paired}


def decide_adopt(report_a: dict, report_b: dict,
                 min_n: int = MIN_POWER_N_DEFAULT) -> str:
    """ADOPT/KEEP/insufficient_power per the printed decision rule:

    ADOPT B only if: n >= MIN_POWER_N, AND b2_B > 0 with p < 0.05, AND
    (b2_B > b2_A OR A's b2 not significant), AND echo_gap_B <= echo_gap_A.
    Otherwise: KEEP A / collect more cycles. Below the power floor:
    insufficient_power — refuse a verdict (same abstain discipline as
    llm_eval.compute_incremental_report).
    """
    n = min(report_a.get('n', 0), report_b.get('n', 0))
    if (report_a.get('insufficient_power') or report_b.get('insufficient_power')
            or n < min_n):
        return (f"insufficient_power (n={n} < {min_n}) — collect more "
               "cycles before trusting a verdict")

    enc_a = report_a.get('encompassing') or {}
    enc_b = report_b.get('encompassing') or {}
    b2_a, p_a = enc_a.get('b2_s'), enc_a.get('p_value')
    b2_b, p_b = enc_b.get('b2_s'), enc_b.get('p_value')
    echo_a = report_a.get('echo_gap')
    echo_b = report_b.get('echo_gap')

    b_significant = (b2_b is not None and p_b is not None
                     and b2_b > 0 and p_b < 0.05)
    if not b_significant:
        return f"KEEP A — variant B's b2 not significant (b2_B={b2_b}, p={p_b})"

    a_significant = (b2_a is not None and p_a is not None
                     and b2_a > 0 and p_a < 0.05)
    if a_significant and not (b2_b is not None and b2_a is not None
                             and b2_b > b2_a):
        return (f"KEEP A — A's b2 already significant and B does not "
               f"exceed it (b2_A={b2_a}, b2_B={b2_b})")

    if echo_a is not None and echo_b is not None and echo_b > echo_a:
        return (f"KEEP A — variant B's echo_gap worsened "
               f"(echo_A={echo_a}, echo_B={echo_b})")

    return (f"ADOPT B — n={n}, b2_B={b2_b} (p={p_b}), b2_A={b2_a}, "
           f"echo_A={echo_a}, echo_B={echo_b}")


def cmd_score(args):
    from llm_eval import realize_scored_rows, compute_incremental_report

    in_path = Path(args.infile)
    rows = _load_score_rows(in_path)
    if not rows:
        print(f"[prompt_ab] no rows in {in_path}")
        return

    # One realization pass shared by both variants — realized return does
    # not depend on `s`, so this avoids fetching bars twice.
    base_rows = [{"symbol": r["symbol"], "asset_type": r.get("asset_type", "crypto"),
                 "t0": r["t0"], "horizon": r.get("horizon", 24), "s": None,
                 "pred": r.get("pred")} for r in rows]
    realized_tuples = realize_scored_rows(base_rows)

    horizons = [r.get("horizon", 24) for r in rows]
    forward_bars = max(horizons) if horizons else 24

    built = pair_variant_samples(rows, realized_tuples)
    report_a = compute_incremental_report(built["samples_a"],
                                          forward_bars=forward_bars,
                                          min_n=args.min_n)
    report_b = compute_incremental_report(built["samples_b"],
                                          forward_bars=forward_bars,
                                          min_n=args.min_n)

    print(f"\n=== Variant A ({len(built['samples_a'])} samples) ===")
    print(json.dumps(report_a, indent=2))
    print(f"\n=== Variant B ({len(built['samples_b'])} samples) ===")
    print(json.dumps(report_b, indent=2))
    print("\n=== Paired A/B comparison ===")
    print(json.dumps(built["paired"], indent=2))

    decision = decide_adopt(report_a, report_b, args.min_n)
    print("\nDecision rule: ADOPT B only if n>=MIN_POWER_N AND b2_B "
         "significant (p<0.05) AND (b2_B>b2_A OR A not significant) AND "
         "echo_gap_B<=echo_gap_A. Otherwise KEEP A / collect more cycles.")
    print(f"Verdict: {decision}")

    out_report = {
        "variant_a": report_a, "variant_b": report_b,
        "paired": built["paired"], "decision": decision,
    }
    with open(DEFAULT_REPORT, "w") as f:
        json.dump(out_report, f, indent=2)
    print(f"Report written: {DEFAULT_REPORT}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Offline prompt A/B harness for the LLM conviction gate",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="replay captured cycles through "
                                       "variant A vs B (persist=False)")
    run_p.add_argument("--days", type=int, default=14)
    run_p.add_argument("--asset", choices=["stock", "crypto"], default=None)
    run_p.add_argument("--system-b", dest="system_b", default=None,
                       help="path to a system-prompt text file for variant B")
    run_p.add_argument("--hide-pred-b", action="store_true",
                       help="variant B withholds the ML pred line "
                            "(pred-blind scoring experiment)")
    run_p.add_argument("--rich-context-b", action="store_true",
                       help="variant B uses captured rich-context profiles "
                            "when available (see build_variant_b_candidates)")
    run_p.add_argument("--model", default=None,
                       help="pin the exact model for BOTH variants "
                            "(default: get_recommended_model('analyst'))")
    run_p.add_argument("--max-cycles", type=int, default=None)
    run_p.add_argument("--sleep-sec", type=float, default=2.0)
    run_p.add_argument("--out", default=str(DEFAULT_OUT))
    run_p.add_argument("--dry-run", action="store_true",
                       help="build both prompts, print size estimates + a "
                            "sample pair, make ZERO API calls")

    score_p = sub.add_parser("score", help="realize forward returns + "
                                           "print the ADOPT/KEEP verdict")
    score_p.add_argument("--in", dest="infile", required=True)
    score_p.add_argument("--min-n", type=int, default=MIN_POWER_N_DEFAULT)

    return ap


def main():
    args = build_arg_parser().parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "score":
        cmd_score(args)


if __name__ == "__main__":
    main()
