"""Tests for run_pipeline.py's command-acknowledgement instrumentation.

2026-07 GUI review §7 (IPC row): the GUI wrote pipeline_command.json and
went straight to an optimistic "Starting..." label — a silent rejection
(e.g. "training in progress") was invisible until the next Models-tab
poll. run_pipeline.py now echoes every consumed command's outcome to
command_result.json via _write_command_result(); this file pins that
schema and the "never breaks dispatch" contract.

run_pipeline.py imports no heavy deps at module level (stdlib +
adaptive_config, both pure) and is fully importable on the dev Mac, but
pytest.importorskip guards the import anyway per repo convention, so this
module SKIPS (not errors) if that ever stops being true.
"""

import json
import os

import pytest
from unittest.mock import MagicMock

rp = pytest.importorskip("run_pipeline")


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Redirect every file this module writes into tmp_path, and reset the
    mutable dispatch globals, so tests never touch the real repo directory
    (a concurrent session may have a live pipeline_status.json) or leak
    state across tests."""
    monkeypatch.setattr(rp, "STATUS_FILE", str(tmp_path / "pipeline_status.json"))
    monkeypatch.setattr(rp, "COMMAND_RESULT_FILE", str(tmp_path / "command_result.json"))
    monkeypatch.setattr(rp, "_manually_stopped", set())
    monkeypatch.setattr(rp, "_COMBINED_BOTS", False)
    monkeypatch.setattr(rp, "_last_status_write", 0)
    yield


def _read_ack(tmp_path):
    with open(tmp_path / "command_result.json") as f:
        return json.load(f)


def _status(phase="trading"):
    return {"phase": phase, "phase_label": "", "phase_idx": 0,
            "started_at": "", "bots_running": False,
            "crypto_bot_running": False, "stock_bot_running": False}


class TestWriteCommandResultSchema:
    """_write_command_result is the pure, separable ack-schema writer."""

    def test_accepted_schema_exact_keys(self, tmp_path):
        before = rp.time.time()
        rp._write_command_result("start_bot", True, False, "accepted")
        after = rp.time.time()
        data = _read_ack(tmp_path)
        assert set(data.keys()) == {"command", "crypto", "stock", "result", "reason", "ts"}
        assert data["command"] == "start_bot"
        assert data["crypto"] is True
        assert data["stock"] is False
        assert data["result"] == "accepted"
        assert data["reason"] == ""
        assert before <= data["ts"] <= after

    def test_rejected_carries_reason(self, tmp_path):
        rp._write_command_result("start_bot", False, True, "rejected",
                                 "training in progress")
        data = _read_ack(tmp_path)
        assert data["result"] == "rejected"
        assert data["reason"] == "training in progress"
        assert data["crypto"] is False
        assert data["stock"] is True

    def test_crypto_stock_coerced_to_real_bool(self, tmp_path):
        # cmd.get(...) can hand back None/0/1 (JSON has no concept baked
        # in) — the frozen schema promises JSON true/false, not 0/1/null.
        rp._write_command_result("stop_bot", None, 1, "accepted")
        data = _read_ack(tmp_path)
        assert data["crypto"] is False
        assert data["stock"] is True

    def test_atomic_write_leaves_no_tmp_file(self, tmp_path):
        rp._write_command_result("stop_bot", True, True, "accepted")
        leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
        assert leftovers == []

    def test_overwrites_previous_result(self, tmp_path):
        rp._write_command_result("start_bot", True, False, "accepted")
        rp._write_command_result("stop_bot", False, True, "rejected", "x")
        data = _read_ack(tmp_path)
        assert data["command"] == "stop_bot"
        assert data["result"] == "rejected"


class TestWriteCommandResultNeverRaises:
    """Contract: an ack-write failure can NEVER break command dispatch."""

    def test_survives_unwritable_target_directory(self, tmp_path, monkeypatch):
        # open(tmp, 'w') raises FileNotFoundError when the parent dir is
        # missing — must be swallowed, not propagated.
        bad_path = tmp_path / "nonexistent_subdir" / "command_result.json"
        monkeypatch.setattr(rp, "COMMAND_RESULT_FILE", str(bad_path))
        rp._write_command_result("start_bot", True, False, "accepted")
        assert not bad_path.exists()

    def test_survives_target_being_a_directory(self, tmp_path, monkeypatch):
        # os.replace(tmp, COMMAND_RESULT_FILE) raises IsADirectoryError-ish
        # when the destination is an existing directory.
        as_dir = tmp_path / "command_result.json"
        as_dir.mkdir()
        monkeypatch.setattr(rp, "COMMAND_RESULT_FILE", str(as_dir))
        rp._write_command_result("start_bot", True, False, "accepted")


class TestHandleCommandAck:
    """Dispatch-level: each _handle_command outcome writes the matching
    ack. Bot-launching helpers are mocked out where a real call would
    shell out to the Jetson-only python path — these tests assert the
    mock was (not) called, which also catches a regression that removes
    an early-rejection return."""

    def test_start_bot_rejected_during_training(self, tmp_path, monkeypatch):
        launch = MagicMock()
        monkeypatch.setattr(rp, "_launch_bots", launch)
        monkeypatch.setattr(rp, "_start_single_bot", launch)
        status = _status(phase="crypto_search")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "start_bot", "crypto": True, "stock": False},
                               [], log_fh, status)
        launch.assert_not_called()
        data = _read_ack(tmp_path)
        assert data["command"] == "start_bot"
        assert data["result"] == "rejected"
        assert data["reason"] == "training in progress"
        assert data["crypto"] is True
        assert data["stock"] is False

    def test_start_bot_accepted_when_idle(self, tmp_path, monkeypatch):
        start_mock = MagicMock()
        monkeypatch.setattr(rp, "_start_single_bot", start_mock)
        status = _status(phase="trading")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "start_bot", "crypto": True, "stock": False},
                               [], log_fh, status)
        start_mock.assert_called_once()
        data = _read_ack(tmp_path)
        assert data["result"] == "accepted"
        assert data["reason"] == ""

    def test_start_bot_rejected_when_combined_already_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rp, "_COMBINED_BOTS", True)
        fake_proc = MagicMock(poll=MagicMock(return_value=None))  # still alive
        bots = [("Bots", fake_proc, MagicMock())]
        status = _status(phase="trading")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "start_bot", "crypto": True, "stock": True},
                               bots, log_fh, status)
        data = _read_ack(tmp_path)
        assert data["result"] == "rejected"
        assert "already running" in data["reason"]

    def test_stop_bot_accepted_with_no_bots_running(self, tmp_path):
        # bots=[] means _stop_single_bot is a pure no-op loop — no
        # subprocess touched, safe to exercise the real function.
        status = _status(phase="trading")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "stop_bot", "crypto": True, "stock": False},
                               [], log_fh, status)
        data = _read_ack(tmp_path)
        assert data["command"] == "stop_bot"
        assert data["result"] == "accepted"

    def test_suspend_and_start_bot_accepted(self, tmp_path):
        status = _status(phase="crypto_search")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "suspend_and_start_bot",
                                "crypto": True, "stock": False},
                               [], log_fh, status)
        data = _read_ack(tmp_path)
        assert data["result"] == "accepted"
        assert status["_pending_bot_start"] == {"crypto": True, "stock": False}

    def test_unknown_command_rejected(self, tmp_path):
        status = _status(phase="trading")
        with open(tmp_path / "log.txt", "a") as log_fh:
            rp._handle_command({"command": "flarp", "crypto": False, "stock": False},
                               [], log_fh, status)
        data = _read_ack(tmp_path)
        assert data["result"] == "rejected"
        assert "flarp" in data["reason"]
