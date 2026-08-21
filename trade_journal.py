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

Retention (c26 T7): TRADER_JOURNAL_ROTATE_DAYS=N (default 0 = OFF) gzips day
files older than N days to journals/YYYY-MM-DD.jsonl.gz via rotate_old_journals.
iter_journal_rows / get_journal_summary read rotated .gz files transparently
through open_journal.
"""

import datetime
import gzip
import json
import os
import threading
from pathlib import Path

from llm_config import load_llm_config
from log_config import get_logger

logger = get_logger(__name__)

JOURNAL_DIR = Path(__file__).resolve().parent / "journals"

# --- Journal retention (c26 T7 / B19 Jetson small-disk protection) ---
# Rotation RENAMES day files to .jsonl.gz => changes data-store contents in
# normal operation => default OFF per campaign classification. Several
# OUT-OF-SCOPE readers (fees.py, llm_eval.py, decision_report.py,
# chart_core.py, gui.py staleness glob, llm_analyst.py, scripts/prompt_ab.py,
# scripts/sizing_cofire_report.py) open the plain .jsonl paths directly and
# would go blind past the rotation horizon — activate only after they gain
# the same .gz fallback (see campaign handoff), via TRADER_JOURNAL_ROTATE_DAYS=N
# (N>=7, recommended 30+) on the Jetson.
try:
    JOURNAL_ROTATE_DAYS = int(os.environ.get('TRADER_JOURNAL_ROTATE_DAYS', '0') or '0')
except ValueError:
    JOURNAL_ROTATE_DAYS = 0
_ROTATE_MAX_PER_CALL = 30          # bound worst-case inline latency
_rotate_lock = threading.Lock()    # combined-mode: two loop threads share the process
_rotate_done_date = None

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

        # One clock read for the ts field, the filename, AND the rotation
        # date (a row stamped 23:59:59.9 must not land in the next day's
        # file, and the rotation guard must agree with it), and
        # offset-aware so the two Stage-0 consumers agree on the epoch:
        # decision_report's pd.Timestamp tz-localizes naive ts as UTC while
        # llm_eval's fromisoformat().timestamp() reads it as local time —
        # with an explicit offset both are exact regardless of box timezone.
        now = datetime.datetime.now().astimezone()

        # Once-daily retention pass (no-op unless TRADER_JOURNAL_ROTATE_DAYS>0)
        global _rotate_done_date
        _today = now.date()
        if JOURNAL_ROTATE_DAYS > 0 and _rotate_done_date != _today:
            _rotate_done_date = _today
            rotate_old_journals()

        record = {**entry, "ts": now.isoformat()}
        filepath = JOURNAL_DIR / f"{now.date().isoformat()}.jsonl"

        line = json.dumps(record, default=str) + "\n"
        with open(filepath, "a") as f:
            f.write(line)
            f.flush()
    except Exception as e:
        logger.warning("[JOURNAL] Error writing: %s", e)


def open_journal(path):
    """Text-mode handle for a day file: the plain .jsonl if present, else its
    .jsonl.gz sibling (transparent gzip reading for every consumer). Raises
    FileNotFoundError when neither exists. When BOTH exist (rotation crash
    window between os.replace and unlink) the plain file wins — it is the
    original; the next rotation pass re-verifies and finishes the unlink."""
    path = Path(path)
    if path.exists():
        return open(path)
    gz = Path(f'{path}.gz')
    if gz.exists():
        return gzip.open(gz, 'rt')
    raise FileNotFoundError(path)


def rotate_old_journals(now=None) -> int:
    """Gzip day files older than JOURNAL_ROTATE_DAYS (journals/D.jsonl ->
    journals/D.jsonl.gz). Returns the number of files rotated this call
    (capped at _ROTATE_MAX_PER_CALL — a backlog drains across calls).

    No-op (0, before ANY filesystem access) when TRADER_JOURNAL_ROTATE_DAYS
    is unset/0. Crash-safe per file: compress to a .gz.tmp, fsync, verify the
    round-trip against the original bytes, os.replace to .gz, only then
    unlink the plain file — the original is never removed until a verified
    copy exists. Never raises; never touches today's file.
    """
    if JOURNAL_ROTATE_DAYS <= 0:
        return 0
    if not _rotate_lock.acquire(blocking=False):
        return 0
    count = 0
    try:
        today = now or datetime.date.today()
        cutoff = today - datetime.timedelta(days=JOURNAL_ROTATE_DAYS)
        now_ts = datetime.datetime.now().timestamp()
        # Stale tmp files from a crashed prior pass: best-effort cleanup.
        try:
            for tmp in JOURNAL_DIR.glob('*.gz.tmp'):
                try:
                    if now_ts - tmp.stat().st_mtime > 86400:
                        tmp.unlink()
                except OSError:
                    pass
        except OSError:
            pass
        for p in sorted(JOURNAL_DIR.glob('*.jsonl')):
            if count >= _ROTATE_MAX_PER_CALL:
                break
            try:
                file_date = datetime.date.fromisoformat(p.stem)
            except ValueError:
                continue  # not a date-named journal file
            if file_date >= cutoff:
                continue  # inside the retention window (never today's file)
            try:
                raw = p.read_bytes()
                gz = Path(f'{p}.gz')
                tmp = Path(f'{p}.gz.tmp')
                if gz.exists():
                    # Crash leftover between os.replace and unlink: verify,
                    # then just finish the unlink; a mismatch is rebuilt.
                    try:
                        with gzip.open(gz, 'rb') as gf:
                            existing = gf.read()
                    except (OSError, EOFError):
                        existing = None
                    if existing == raw:
                        p.unlink()
                        count += 1
                        continue
                    gz.unlink()
                with gzip.open(tmp, 'wb') as gf:
                    gf.write(raw)
                    gf.flush()
                    os.fsync(gf.fileno())
                with gzip.open(tmp, 'rb') as gf:
                    ok = gf.read() == raw
                if not ok:
                    tmp.unlink()   # original untouched — never lose rows
                    continue
                os.replace(tmp, gz)
                p.unlink()
                count += 1
            except OSError as e:
                logger.warning("[JOURNAL] rotation failed for %s: %s", p, e)
                continue
    except Exception as e:
        logger.warning("[JOURNAL] rotation error: %s", e)
    finally:
        _rotate_lock.release()
    return count


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
        try:
            f = open_journal(filepath)
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.warning("[JOURNAL] Error reading %s: %s", filepath, e)
            continue
        try:
            with f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except (OSError, EOFError) as e:
            # A corrupt/truncated .gz surfaces here from the gzip stream.
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

    entries = []
    skipped_lines = 0
    try:
        try:
            f = open_journal(filepath)
        except FileNotFoundError:
            return {"total": 0, "buys": 0, "sells": 0, "skips": 0,
                    "llm_blocks": 0, "avg_multiplier": 1.0,
                    "skipped_lines": 0, "entries": []}
        with f:
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
