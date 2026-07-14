"""Structured trade journal — append-only JSONL logging of every decision.

One file per day in journals/ directory: journals/2026-02-08.jsonl
Each line is a self-contained JSON object. log_decision() guarantees only the
`ts` key (offset-aware ISO-8601); every other key is set by the PRODUCER
(base_loop.py / stock_loop.py). The schema below is the producer contract the
Stage-0 consumers read — keep it in sync when adding a row type or key.

Row schema (tagged union on `action`; `ts` on every row):
  "buy"           symbol, pred_return, sentiment_gate, sentiment_reasons,
                  llm_multiplier, llm_score, llm_reasoning, final_notional,
                  decision_price, fill_price, slippage_bps, entry_tactic, maker,
                  skip_reason(=None); optional conviction fields + nested `sizing`.
  "sell"          symbol, exit_reason, pnl_pct, decision_price, fill_price,
                  slippage_bps, estimated.
  "skip"          symbol, skip_reason (sentiment_block | llm_veto | meta_veto |
                  cost/qty_zero/…); optional pred_return, meta_prob, entry_rank,
                  + conviction fields (spread_pct, _fetch_failed, …).
  "llm_analysis"  asset_type, forward_bars, scores={sym:{s,pred}}.
  "entry_window"  asset_type, n_candidates, admitted_k, admitted, veto_counts,
                  buys_allowed.
  "account_risk"  book, plus the record_book_risk_and_report payload.

Consumers (read-only; break on a rename):
  decision_report.py  action, skip_reason, symbol, spread_pct, exit_reason,
                      pnl_pct, pred_return, meta_p/meta_prob, entry_rank,
                      asset_type, admitted_k, veto_counts, _fetch_failed.
  llm_eval.py         action=="llm_analysis": asset_type, forward_bars,
                      scores{s,pred}, ts.
  fees.py             action=="buy": symbol, entry_tactic (maker-share feedback).
  execution_report.py action, symbol, entry_tactic, exit_reason, slippage_bps.
"""

import datetime
import json
from pathlib import Path

from llm_config import load_llm_config
from log_config import get_logger

logger = get_logger(__name__)

JOURNAL_DIR = Path(__file__).resolve().parent / "journals"

_disabled_warned = False


def log_decision(entry: dict):
    """Append one decision record to today's journal file.

    Never raises: call sites are live trading loops — buy sites journal
    AFTER the order filled but BEFORE cooldown/trade-count stamping, so an
    exception here would skip that bookkeeping and abort the cycle.

    The `journal_enabled` config switch (GUI "Trade Journal" checkbox)
    silences EVERY row type, not just trade rows: account_risk rows, the
    llm_analysis rows llm_eval scores, the conviction/Stage-0 skip rows,
    and the buy rows fees.py's live maker-share feedback reads (which then
    drifts to full-taker pricing). Disable with care.
    """
    global _disabled_warned
    try:
        config = load_llm_config()
        if not config.get("journal_enabled", True):
            if not _disabled_warned:
                _disabled_warned = True
                logger.warning("[JOURNAL] journaling disabled — dropping ALL "
                               "rows (trade/skip/account_risk/llm_analysis; "
                               "Stage-0 and maker-share inputs)")
            return

        JOURNAL_DIR.mkdir(exist_ok=True)

        # One clock read for both the ts field and the filename (a row
        # stamped 23:59:59.9 must not land in the next day's file), and
        # offset-aware so the two Stage-0 consumers agree on the epoch:
        # decision_report's pd.Timestamp tz-localizes naive ts as UTC while
        # llm_eval's fromisoformat().timestamp() reads it as local time —
        # with an explicit offset both are exact regardless of box timezone.
        now = datetime.datetime.now().astimezone()
        record = {**entry, "ts": now.isoformat()}
        filepath = JOURNAL_DIR / f"{now.date().isoformat()}.jsonl"

        line = json.dumps(record, default=str) + "\n"
        with open(filepath, "a") as f:
            f.write(line)
            f.flush()
    except Exception as e:
        logger.warning("[JOURNAL] Error writing: %s", e)


def iter_journal_rows(days: int = 30):
    """Yield parsed rows from the last days+1 daily journal files
    (today inclusive), oldest file first, rows in append order.

    Canonical shared reader: skips blank lines and corrupt rows per line
    (a torn trailing line from a concurrent append is expected). Consumers
    apply their own row filters.
    """
    today = datetime.date.today()
    for offset in range(days, -1, -1):
        day = today - datetime.timedelta(days=offset)
        filepath = JOURNAL_DIR / f"{day.isoformat()}.jsonl"
        if not filepath.exists():
            continue
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("[JOURNAL] Error reading %s: %s", filepath, e)


def get_journal_summary(date: str = None) -> dict:
    """Read a day's journal and return summary stats.

    Args:
        date: ISO date string (e.g. '2026-02-08'). Defaults to today.

    Returns dict with:
        total, buys, sells, skips, llm_blocks, avg_multiplier,
        skipped_lines, entries

    Note: `total` counts every row in the file, including non-decision
    rows (llm_analysis / entry_window / account_risk), not just the
    buy/sell/skip decisions broken out below it.
    """
    if date is None:
        date = datetime.date.today().isoformat()

    filepath = JOURNAL_DIR / f"{date}.jsonl"
    if not filepath.exists():
        return {"total": 0, "buys": 0, "sells": 0, "skips": 0,
                "llm_blocks": 0, "avg_multiplier": 1.0,
                "skipped_lines": 0, "entries": []}

    entries = []
    skipped_lines = 0
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # Torn line from a concurrent append — skip the row,
                    # keep every row after it readable.
                    skipped_lines += 1
    except Exception as e:
        logger.warning("[JOURNAL] Error reading %s: %s", filepath, e)

    buys = sum(1 for e in entries if e.get("action") == "buy")
    sells = sum(1 for e in entries if e.get("action") == "sell")
    skips = sum(1 for e in entries if e.get("action") == "skip")
    # Writers emit 'llm_veto' — the old 'llm_block' key was never written,
    # so this metric was permanently zero
    llm_blocks = sum(1 for e in entries
                     if e.get("skip_reason") in ("llm_veto", "llm_block"))

    multipliers = [e["llm_multiplier"] for e in entries if "llm_multiplier" in e and e["llm_multiplier"] is not None]
    avg_mult = sum(multipliers) / len(multipliers) if multipliers else 1.0

    return {
        "total": len(entries),
        "buys": buys,
        "sells": sells,
        "skips": skips,
        "llm_blocks": llm_blocks,
        "avg_multiplier": round(avg_mult, 2),
        "skipped_lines": skipped_lines,
        "entries": entries,
    }
