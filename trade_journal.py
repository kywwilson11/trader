"""Structured trade journal — append-only JSONL logging of every decision.

One file per day in journals/ directory: journals/2026-02-08.jsonl
Each line is a self-contained JSON object with all inputs and reasoning.
"""

import json
import datetime
from pathlib import Path

from llm_config import load_llm_config

JOURNAL_DIR = Path(__file__).resolve().parent / "journals"


def log_decision(entry: dict):
    """Append one decision record to today's journal file."""
    config = load_llm_config()
    if not config.get("journal_enabled", True):
        return

    JOURNAL_DIR.mkdir(exist_ok=True)

    entry["ts"] = datetime.datetime.now().isoformat()
    today = datetime.date.today().isoformat()
    filepath = JOURNAL_DIR / f"{today}.jsonl"

    try:
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"[JOURNAL] Error writing: {e}")


def get_journal_summary(date: str = None) -> dict:
    """Read a day's journal and return summary stats.

    Args:
        date: ISO date string (e.g. '2026-02-08'). Defaults to today.

    Returns dict with:
        total, buys, sells, skips, llm_blocks, avg_multiplier, entries
    """
    if date is None:
        date = datetime.date.today().isoformat()

    filepath = JOURNAL_DIR / f"{date}.jsonl"
    if not filepath.exists():
        return {"total": 0, "buys": 0, "sells": 0, "skips": 0,
                "llm_blocks": 0, "avg_multiplier": 1.0, "entries": []}

    entries = []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        print(f"[JOURNAL] Error reading {filepath}: {e}")

    buys = sum(1 for e in entries if e.get("action") == "buy")
    sells = sum(1 for e in entries if e.get("action") == "sell")
    skips = sum(1 for e in entries if e.get("action") == "skip")
    llm_blocks = sum(1 for e in entries if e.get("skip_reason") == "llm_block")

    multipliers = [e["llm_multiplier"] for e in entries if "llm_multiplier" in e and e["llm_multiplier"] is not None]
    avg_mult = sum(multipliers) / len(multipliers) if multipliers else 1.0

    return {
        "total": len(entries),
        "buys": buys,
        "sells": sells,
        "skips": skips,
        "llm_blocks": llm_blocks,
        "avg_multiplier": round(avg_mult, 2),
        "entries": entries,
    }
