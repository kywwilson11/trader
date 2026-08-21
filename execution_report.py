"""Implementation-shortfall report — what execution actually costs.

Slippage concentrates in exactly the trades that matter (stop-outs in fast
markets), so backtest assumptions must be checked against REALIZED fills.
The loops journal decision_price (quote midpoint when the decision fired)
and fill_price on every confirmed entry and exit; this tool aggregates
them.

Sign convention: positive slippage_bps always means "worse than the
decision price" (buys filled higher, sells filled lower).

Usage:
    python execution_report.py --days 14
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import fees
from trade_journal import JOURNAL_DIR, open_journal

BASE_DIR = Path(__file__).resolve().parent


def _load(days: int):
    """Return (rows, n_skipped). A partial trailing line from a concurrent
    append is expected; counting skips makes systematic corruption visible."""
    rows = []
    n_skipped = 0
    today = datetime.now().date()
    for d in range(days + 1):
        path = JOURNAL_DIR / f"{(today - timedelta(days=d)).isoformat()}.jsonl"
        try:
            f = open_journal(path)   # transparent .jsonl.gz fallback
        except (FileNotFoundError, OSError):
            continue
        try:
            with f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        n_skipped += 1
                        continue
                    if (e.get('slippage_bps') is not None
                            or e.get('action') in ('buy', 'llm_analysis',
                                                   'llm_backoff')):
                        rows.append(e)
        except (OSError, EOFError):
            continue
    return rows, n_skipped


def _write_json(report: dict) -> None:
    out = BASE_DIR / 'execution_report.json'
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {out}")


def run_report(days: int = 14) -> dict:
    rows, n_skipped = _load(days)
    if n_skipped:
        print(f"note: {n_skipped} unparseable journal line(s) skipped")

    # Stamped so a stale execution_report.json on disk is distinguishable
    # from a fresh run; written even on an empty window for the same reason.
    report = {
        'generated_at': datetime.now().isoformat(),
        'window_days': days,
    }
    if not rows:
        print("No fills with slippage data yet — the loops journal "
              "decision_price/fill_price on every confirmed fill.")
        _write_json(report)
        return report

    def crypto(sym):
        return '/' in sym or (sym.endswith('USD') and len(sym) > 5)

    # _load also collects buy/llm_analysis/llm_backoff rows without
    # slippage (c26 T7); the shortfall stats keep reading fills only.
    fills = [e for e in rows if e.get('slippage_bps') is not None]

    if fills:
        groups = defaultdict(list)
        for e in fills:
            sym = e.get('symbol', '?')
            asset = 'crypto' if crypto(sym) else 'stock'
            action = e.get('action', '?')
            reason = e.get('exit_reason') or 'entry'
            groups[(asset, action, reason)].append(float(e['slippage_bps']))

        print(f"\n=== IMPLEMENTATION SHORTFALL (last {days}d, {len(fills)} fills) ===")
        print(f"{'asset':<8}{'action':<7}{'reason':<14}{'n':>5}"
              f"{'mean bps':>10}{'median':>9}{'p90':>8}{'worst':>9}")
        for (asset, action, reason), vals in sorted(groups.items()):
            a = np.array(vals)
            entry = {
                'n': len(a),
                'mean_bps': round(float(a.mean()), 2),
                'median_bps': round(float(np.median(a)), 2),
                'p90_bps': round(float(np.percentile(a, 90)), 2),
                'worst_bps': round(float(a.max()), 2),
            }
            report[f"{asset}/{action}/{reason}"] = entry
            print(f"{asset:<8}{action:<7}{reason:<14}{entry['n']:>5}"
                  f"{entry['mean_bps']:>10.1f}{entry['median_bps']:>9.1f}"
                  f"{entry['p90_bps']:>8.1f}{entry['worst_bps']:>9.1f}")

        all_bps = np.array([float(e['slippage_bps']) for e in fills])
        overall = round(float(all_bps.mean()), 2)
        report['overall_mean_bps'] = overall
        print(f"\nOverall mean shortfall: {overall:+.1f} bps per fill")

    # Maker share of crypto entries (the maker ladder journals
    # entry_tactic per buy). Realized fee/RT = 2*taker - (taker-maker)*share
    # bps (entry taker-(taker-maker)*share, exit always taker) vs the 2*taker
    # taker-taker baseline, with taker/maker = fees.CRYPTO_TAKER_BPS /
    # CRYPTO_MAKER_BPS (today 25/15, i.e. RT = 50 - 10*maker_share bps) —
    # feeds the fees.py review. Same '/'-only symbol predicate as
    # fees.realized_crypto_maker_share, so the share printed here is the one
    # the live fee gate actually computes.
    tactics = []
    for e in rows:
        sym = e.get('symbol', '')
        if (e.get('action') == 'buy' and isinstance(sym, str)
                and '/' in sym and e.get('entry_tactic')):
            tactics.append(e['entry_tactic'])
    if tactics:
        maker_n = sum(1 for t in tactics if t.startswith('maker'))
        share = maker_n / len(tactics)
        report['crypto_maker_share'] = round(share, 3)
        taker = fees.CRYPTO_TAKER_BPS       # taker leg (also the exit leg)
        entry_fee_bps = taker - (taker - fees.CRYPTO_MAKER_BPS) * share
        caveat = ('' if len(tactics) >= fees.MAKER_SHARE_MIN_ENTRIES else
                  f' (below live-gate min n={fees.MAKER_SHARE_MIN_ENTRIES})')
        # Entry-leg fee: maker (fees.CRYPTO_MAKER_BPS) vs taker
        # (fees.CRYPTO_TAKER_BPS); exit is always taker.
        print(f"Crypto maker share (entries): {share:.0%} of {len(tactics)} "
              f"— entry fee ≈ {entry_fee_bps:.1f} bps vs {taker:.0f} "
              f"taker{caveat}")

    # --- c26 T7 additive blocks (new journal-key adoption; state-map C.5) ---
    # (a) Entry slippage decomposed by quote age at decision time
    # (quote_age_s journaled by base_loop, read by nothing until now).
    buckets = {'lt2s': (0, 2), '2_10s': (2, 10), 'gte10s': (10, float('inf'))}
    by_age = {}
    for bucket_label, (lo, hi) in buckets.items():
        vals = [float(e['slippage_bps']) for e in fills
                if e.get('action') == 'buy'
                and isinstance(e.get('quote_age_s'), (int, float))
                and lo <= float(e['quote_age_s']) < hi]
        if vals:
            by_age[bucket_label] = {'n': len(vals),
                                    'mean_bps': round(float(np.mean(vals)), 2)}
    if by_age:
        report['entry_slippage_by_quote_age'] = by_age
        print("Entry slippage by quote age: "
              + ", ".join(f"{k} n={v['n']} mean={v['mean_bps']:+.1f}bps"
                          for k, v in by_age.items()))

    # (b) Crypto maker NOTIONAL share — count-based share above overweights
    # small fills; per-rung maker_notional (T6, if/when journaled) overrides
    # the entry_tactic heuristic. Defensive: no hard dependency on T6.
    m_not = t_not = 0.0
    for e in rows:
        sym = e.get('symbol', '')
        if (e.get('action') == 'buy' and isinstance(sym, str) and '/' in sym
                and isinstance(e.get('final_notional'), (int, float))):
            fn = float(e['final_notional'])
            t_not += fn
            mn = e.get('maker_notional')   # T6 per-rung split, if/when journaled
            if isinstance(mn, (int, float)):
                m_not += min(float(mn), fn)
            elif str(e.get('entry_tactic', '')).startswith('maker'):
                m_not += fn
    if t_not > 0:
        report['crypto_maker_notional_share'] = round(m_not / t_not, 3)
        print(f"Crypto maker NOTIONAL share: "
              f"{report['crypto_maker_notional_share']:.1%} "
              f"of ${t_not:,.0f} entered")

    # (c) LLM analysis-call economics (model/dedup_hit/latency_ms/cost_usd
    # journaled per call; llm_backoff rows count skipped-call cycles).
    llm_rows = [e for e in rows if e.get('action') == 'llm_analysis']
    if llm_rows:
        hits = sum(1 for e in llm_rows if e.get('dedup_hit'))
        lat = [float(e['latency_ms']) for e in llm_rows
               if isinstance(e.get('latency_ms'), (int, float))]
        cost = [float(e['cost_usd']) for e in llm_rows
                if isinstance(e.get('cost_usd'), (int, float))]
        report['llm_analysis'] = {
            'n_calls': len(llm_rows),
            'dedup_hits': hits,
            'dedup_hit_rate': round(hits / len(llm_rows), 3),
            'mean_latency_ms': (round(float(np.mean(lat)), 1) if lat else None),
            'total_cost_usd': (round(float(np.sum(cost)), 4) if cost else None),
            'n_backoffs': sum(1 for e in rows
                              if e.get('action') == 'llm_backoff'),
        }
        la = report['llm_analysis']
        print(f"LLM analysis: {la['n_calls']} calls, dedup "
              f"{la['dedup_hit_rate']:.0%}, mean latency "
              f"{la['mean_latency_ms']} ms, cost ${la['total_cost_usd']}, "
              f"{la['n_backoffs']} backoff(s)")

    print("Compare against the backtest's assumptions (fees.py spread "
          "haircuts: crypto 10 bps, stock 5 bps round trip). If realized "
          "shortfall is persistently higher, the backtest is optimistic — "
          "raise SPREAD_PCT in backtest.py and the entry edge floor.")

    _write_json(report)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Realized execution-cost report')
    ap.add_argument('--days', type=int, default=14)
    args = ap.parse_args()
    run_report(args.days)
