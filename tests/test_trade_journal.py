"""Tests for trade_journal.py — JSONL logging and summaries."""

import json
import datetime
import pytest
from unittest.mock import patch
from pathlib import Path

from trade_journal import log_decision, get_journal_summary


@pytest.fixture
def journal_dir(tmp_path):
    """Patch JOURNAL_DIR to a temp directory."""
    jdir = tmp_path / "journals"
    jdir.mkdir()
    with patch("trade_journal.JOURNAL_DIR", jdir):
        yield jdir


class TestLogDecision:
    def test_creates_file_and_appends(self, journal_dir):
        with patch("trade_journal.load_llm_config", return_value={"journal_enabled": True}):
            log_decision({"action": "buy", "symbol": "TSLA"})

        today = datetime.date.today().isoformat()
        filepath = journal_dir / f"{today}.jsonl"
        assert filepath.exists()
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "buy"
        assert entry["symbol"] == "TSLA"
        assert "ts" in entry


class TestGetJournalSummary:
    def test_empty_when_no_file(self, journal_dir):
        summary = get_journal_summary("2020-01-01")
        assert summary["total"] == 0
        assert summary["entries"] == []

    def test_counts_actions(self, journal_dir):
        today = datetime.date.today().isoformat()
        filepath = journal_dir / f"{today}.jsonl"
        entries = [
            {"action": "buy", "llm_multiplier": 1.2},
            {"action": "sell", "llm_multiplier": 0.8},
            {"action": "skip", "skip_reason": "llm_block", "llm_multiplier": 0.0},
        ]
        filepath.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        summary = get_journal_summary(today)
        assert summary["total"] == 3
        assert summary["buys"] == 1
        assert summary["sells"] == 1
        assert summary["skips"] == 1
        assert summary["llm_blocks"] == 1
        assert 0.0 < summary["avg_multiplier"] < 1.5
