"""Review batch b09: short_flow diagnostics/robustness + squeeze_features docstring.

Covers: load_archive no longer swallows a corrupt archive silently (warns via
the module logger); sync separates the request budget (attempts, 404s included
— unchanged) from files actually downloaded (fetched) vs parsed (ingested), so
a pure-holiday run no longer logs as "synced N day-files"; urlopen is closed
deterministically via context manager; urllib.error is imported explicitly;
squeeze_features' docstring no longer claims its columns already ship into the
harvest (they have zero consumers — wiring is wave-8 backlog).

Runs on the dev Mac: short_flow's heavy imports are lazy, network/parquet are
monkeypatched.
"""
import logging
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import short_flow as sf
import squeeze_features

SRC = (REPO / 'short_flow.py').read_text()

HEADER = 'Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market'


# ---------------------------------------------------------------- docstring

def test_squeeze_docstring_reflects_unwired_reality():
    doc = squeeze_features.__doc__
    # The old present-tense claim ("the FEATURE half that ships into
    # harvest_crypto_data") was false — the columns have no consumers
    # outside the module and its test.
    assert 'that ships into harvest_crypto_data' not in doc
    assert 'INTENDED to ship' in doc
    assert 'NOT yet done' in doc


# ------------------------------------------------------------ source guards

def test_urllib_error_imported_explicitly():
    # 'except urllib.error.HTTPError' must not rely on urllib.request's
    # transitive submodule binding.
    assert re.search(r'^import urllib\.error$', SRC, re.M)


def test_urlopen_used_as_context_manager():
    assert 'with urllib.request.urlopen' in SRC
    assert '.read().decode()\n            fetched' not in SRC


# ------------------------------------------------------------- load_archive

def test_load_archive_corrupt_logs_warning_and_returns_empty(
        tmp_path, monkeypatch, caplog):
    bad = tmp_path / 'short_flow.parquet'
    bad.write_bytes(b'not a parquet file')
    monkeypatch.setattr(sf, 'ARCHIVE_FILE', bad)
    with caplog.at_level(logging.WARNING, logger='short_flow'):
        out = sf.load_archive()
    # On the Mac read_parquet raises ImportError (no engine); on the Jetson
    # it raises on the garbage bytes — both must hit the warning path.
    assert 'archive read failed' in caplog.text
    assert out.empty
    assert list(out.columns) == ['date', 'symbol', 'short_vol', 'total_vol']


def test_load_archive_missing_file_stays_quiet(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sf, 'ARCHIVE_FILE', tmp_path / 'nope.parquet')
    with caplog.at_level(logging.WARNING, logger='short_flow'):
        out = sf.load_archive()
    assert 'archive read failed' not in caplog.text   # normal first-run path
    assert out.empty


# --------------------------------------------------------------------- sync

class _FakeResp:
    def __init__(self, payload: str):
        self._payload = payload
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def read(self):
        return self._payload.encode()


def _wire_sync(monkeypatch, tmp_path, payloads, calls, responses):
    """Patch symbols/archive/parquet-IO/urlopen. `payloads` is consumed in
    order: str -> HTTP 200 body, int -> HTTPError code; exhausted -> 404."""
    import pandas as pd

    monkeypatch.setattr(sf, '_panel_symbols', lambda: {'AAPL'})
    monkeypatch.setattr(sf, 'ARCHIVE_FILE', tmp_path / 'short_flow.parquet')
    monkeypatch.setattr(
        pd.DataFrame, 'to_parquet',
        lambda self, path, *a, **k: Path(path).write_bytes(b'PARQ'))

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        item = payloads.pop(0) if payloads else 404
        if isinstance(item, int):
            raise urllib.error.HTTPError(req.full_url, item, 'err', None, None)
        resp = _FakeResp(item)
        responses.append(resp)
        return resp

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)


def test_sync_summary_separates_fetched_from_ingested(
        tmp_path, monkeypatch, capsys):
    # 2 downloads (one parses to AAPL rows, one has no panel symbols) then
    # 404s: budget of 5 -> 5 attempts, fetched=2, ingested=1.
    payloads = [
        HEADER + '\n20260601|AAPL|100|0|400|B\n20260601|AAPL|50|0|100|Q\n',
        HEADER + '\n20260602|MSFT|10|0|20|B\n',
    ]
    calls, responses = [], []
    _wire_sync(monkeypatch, tmp_path, payloads, calls, responses)

    assert sf.sync(days_back=30, max_files=5) is True
    out = capsys.readouterr().out
    # Two venue prints groupby-aggregate to one archive row.
    assert 'fetched 2 day-files, ingested 1 (1 rows)' in out
    assert 'synced' not in out
    assert len(calls) == 5                       # 404s still consume budget
    assert responses and all(r.exited for r in responses)   # urlopen closed
    assert (tmp_path / 'short_flow.parquet').exists()


def test_sync_pure_404_run_reports_up_to_date_not_synced(
        tmp_path, monkeypatch, capsys):
    # The old code logged a pure-holiday/404 run as "synced N day-files".
    calls, responses = [], []
    _wire_sync(monkeypatch, tmp_path, [], calls, responses)   # 404 forever

    assert sf.sync(days_back=30, max_files=5) is False
    out = capsys.readouterr().out
    assert 'up to date (0 rows)' in out
    assert 'fetched' not in out
    assert 'synced' not in out


def test_sync_budget_still_counts_404_attempts(tmp_path, monkeypatch, capsys):
    # Intentionally preserved behavior: max_files caps HTTP requests spent,
    # including 404s/errors — not successful downloads.
    calls, responses = [], []
    _wire_sync(monkeypatch, tmp_path, [], calls, responses)

    sf.sync(days_back=600, max_files=7)
    capsys.readouterr()
    assert len(calls) == 7


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
