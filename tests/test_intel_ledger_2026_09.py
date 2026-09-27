"""INTEL W9 (2026-09-27): llm_cost_history.jsonl rollover append.

At the daily cost rollover (llm_client._rollover_cost_locked) the shared
ledger llm_cost.json is overwritten with the new day's $0; the previous
day's spend used to survive only as a stdout print. The rollover now
appends ONE JSON line describing the rolled-over day to
llm_cost_history.jsonl next to _COST_FILE. Measurement-only: fail-soft,
under the existing _cost_file_lock, and the ledger JSON itself is
byte-pinned unchanged.

Every test runs against a tmp_path ledger (autouse fixture below, the
test_g4b_fixes_2026_09.py idiom) — the repo-root llm_cost.json family is
never touched.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

llm_client = pytest.importorskip("llm_client")

_KEYS = {"date", "cost", "src", "mem_date", "mem_cost", "reset_at", "pid"}


@pytest.fixture(autouse=True)
def _sandboxed_cost_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_client, '_COST_FILE',
                        str(tmp_path / 'llm_cost.json'))
    monkeypatch.setattr(llm_client, '_cost_reset_date', '')
    monkeypatch.setattr(llm_client, '_daily_cost', 0.0)
    return tmp_path


def _today():
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _hist(tmp_path):
    return tmp_path / 'llm_cost_history.jsonl'


def _lines(tmp_path):
    p = _hist(tmp_path)
    if not p.exists():
        return []
    return p.read_bytes().decode('utf-8').splitlines()


def _stale_ledger(tmp_path, monkeypatch, file_cost=0.9, mem_date='2000-01-01',
                  mem_cost=0.9, file_date='2000-01-01'):
    Path(llm_client._COST_FILE).write_text(
        json.dumps({'date': file_date, 'cost': file_cost}))
    monkeypatch.setattr(llm_client, '_cost_reset_date', mem_date)
    monkeypatch.setattr(llm_client, '_daily_cost', mem_cost)


# --- path -------------------------------------------------------------------

def test_history_path_derives_from_cost_file(tmp_path):
    assert llm_client._cost_history_path() == str(_hist(tmp_path))


# --- rollover writes exactly one well-formed line ---------------------------

def test_rollover_writes_exactly_one_well_formed_line(tmp_path, monkeypatch,
                                                      capsys):
    _stale_ledger(tmp_path, monkeypatch)
    llm_client._maybe_reset_quota()
    assert 'Daily cost reset (yesterday: $0.9000)' in capsys.readouterr().out
    raw = _hist(tmp_path).read_bytes()
    assert raw.endswith(b'\n') and raw.count(b'\n') == 1
    assert len(raw) <= llm_client._COST_HISTORY_MAX_LINE
    rec = json.loads(raw)
    assert set(rec) == _KEYS
    assert rec['date'] == '2000-01-01' and rec['cost'] == 0.9
    assert rec['src'] == 'file'
    assert rec['mem_date'] == '2000-01-01' and rec['mem_cost'] == 0.9
    assert rec['pid'] == os.getpid()
    ts = datetime.fromisoformat(rec['reset_at'])
    assert ts.utcoffset().total_seconds() == 0


def test_fresh_process_rollover_records_the_files_day(tmp_path, monkeypatch):
    # A process started after midnight: in-memory ledger empty, file still on
    # yesterday. Pre-change this day's spend vanished without even a print.
    _stale_ledger(tmp_path, monkeypatch, file_cost=0.37, mem_date='',
                  mem_cost=0.0)
    llm_client._maybe_reset_quota()
    [line] = _lines(tmp_path)
    rec = json.loads(line)
    assert (rec['date'], rec['cost'], rec['src']) == ('2000-01-01', 0.37,
                                                      'file')
    assert rec['mem_date'] is None and rec['mem_cost'] == 0.0


def test_file_total_wins_over_stale_memory(tmp_path, monkeypatch):
    # Another process spent after this one's last read: the shared file holds
    # the cross-process total; the in-memory view is kept alongside.
    _stale_ledger(tmp_path, monkeypatch, file_cost=0.5, mem_cost=0.3)
    llm_client._maybe_reset_quota()
    rec = json.loads(_lines(tmp_path)[0])
    assert rec['cost'] == 0.5 and rec['mem_cost'] == 0.3


def test_mem_fallback_when_ledger_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, '_cost_reset_date', '2000-01-01')
    monkeypatch.setattr(llm_client, '_daily_cost', 0.2)
    llm_client._maybe_reset_quota()
    rec = json.loads(_lines(tmp_path)[0])
    assert (rec['date'], rec['cost'], rec['src']) == ('2000-01-01', 0.2, 'mem')


def test_record_cost_across_midnight_appends_once(tmp_path, monkeypatch):
    _stale_ledger(tmp_path, monkeypatch)
    llm_client._record_cost('claude-haiku-4-5', 0, 0,
                            {'promptTokenCount': 100,
                             'candidatesTokenCount': 20,
                             'thoughtsTokenCount': 0})
    rec = json.loads(_lines(tmp_path)[0])
    assert len(_lines(tmp_path)) == 1 and rec['cost'] == 0.9
    data = json.loads(Path(llm_client._COST_FILE).read_text())
    assert data['date'] == _today()
    assert data['cost'] == pytest.approx(0.0002, abs=1e-12)


# --- non-rollover writes nothing --------------------------------------------

def test_same_day_writes_nothing(tmp_path):
    Path(llm_client._COST_FILE).write_text(
        json.dumps({'date': _today(), 'cost': 0.4}))
    llm_client._maybe_reset_quota()
    llm_client._record_cost('claude-haiku-4-5', 0, 0,
                            {'promptTokenCount': 10,
                             'candidatesTokenCount': 1,
                             'thoughtsTokenCount': 0})
    assert llm_client.get_daily_cost()[0] > 0.4
    assert not _hist(tmp_path).exists()


def test_no_prior_ledger_day_writes_nothing(tmp_path):
    # Fresh process, no ledger file anywhere: nothing was rolled over.
    llm_client._maybe_reset_quota()
    assert Path(llm_client._COST_FILE).exists()  # rollover itself still ran
    assert not _hist(tmp_path).exists()


def test_second_process_after_rollover_does_not_duplicate(tmp_path,
                                                          monkeypatch):
    _stale_ledger(tmp_path, monkeypatch)
    llm_client._maybe_reset_quota()
    # A second process still holding yesterday in memory: under the flock its
    # _load_shared_cost now sees today's file, so it never reaches the append.
    monkeypatch.setattr(llm_client, '_cost_reset_date', '2000-01-01')
    monkeypatch.setattr(llm_client, '_daily_cost', 0.9)
    llm_client._maybe_reset_quota()
    assert len(_lines(tmp_path)) == 1


# --- lock discipline ---------------------------------------------------------

def test_append_happens_under_the_cost_file_lock(tmp_path, monkeypatch):
    state = {'held': False, 'appends': []}

    class RecLock:
        def __enter__(self):
            state['held'] = True
            return self

        def __exit__(self, *exc):
            state['held'] = False
            return False

    real_append = llm_client._append_cost_history

    def spy(*a, **k):
        state['appends'].append(state['held'])
        return real_append(*a, **k)

    monkeypatch.setattr(llm_client, '_cost_file_lock', RecLock)
    monkeypatch.setattr(llm_client, '_append_cost_history', spy)
    _stale_ledger(tmp_path, monkeypatch)
    llm_client._maybe_reset_quota()
    assert state['appends'] == [True]
    assert len(_lines(tmp_path)) == 1


# --- fail-soft: the ledger behaves identically when the append fails --------

def _run_scenario(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm_client, '_COST_FILE',
                        str(tmp_path / 'llm_cost.json'))
    _stale_ledger(tmp_path, monkeypatch)
    capsys.readouterr()
    llm_client._maybe_reset_quota()
    ok = llm_client._cost_ok()
    spent = llm_client.get_daily_cost()
    out = capsys.readouterr().out
    return (Path(llm_client._COST_FILE).read_bytes(),
            llm_client._cost_reset_date, llm_client._daily_cost,
            ok, spent), out


@pytest.mark.parametrize('breakage', ['dir_at_path', 'missing_dir'])
def test_write_failure_leaves_ledger_identical(tmp_path, monkeypatch, capsys,
                                               breakage):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    good, out_good = _run_scenario(a, monkeypatch, capsys)
    assert len(_lines(a)) == 1
    if breakage == 'dir_at_path':
        _hist(b).mkdir()  # the history path cannot be opened for append
    else:
        monkeypatch.setattr(llm_client, '_cost_history_path',
                            lambda: str(b / 'no_such_dir' / 'h.jsonl'))
    bad, out_bad = _run_scenario(b, monkeypatch, capsys)
    assert bad == good
    assert out_bad.count('[LLM-COST] cost history append failed') == 1
    assert [ln for ln in out_bad.splitlines()
            if 'cost history append failed' not in ln] == \
        out_good.splitlines()


@pytest.mark.parametrize('bad_file', [
    {'date': 'x' * 300, 'cost': 0.1},          # line would exceed the cap
    {'date': '2000-01-01', 'cost': float('nan')},  # non-finite cost
])
def test_unwritable_record_is_skipped_softly(tmp_path, monkeypatch, capsys,
                                             bad_file):
    Path(llm_client._COST_FILE).write_text(json.dumps(bad_file))
    llm_client._maybe_reset_quota()
    assert 'cost history append failed' in capsys.readouterr().out
    assert not _hist(tmp_path).exists()
    assert json.loads(Path(llm_client._COST_FILE).read_text()) == {
        'date': _today(), 'cost': 0.0}


def test_ledger_write_failure_still_raises_into_the_old_handler(
        tmp_path, monkeypatch, capsys):
    # Unchanged pre-existing contract (test_llm_fixes_2026_09): a failing
    # ledger write is reported by _maybe_reset_quota; no history is written
    # for a rollover whose ledger write raised.
    def boom():
        raise OSError('disk full')
    monkeypatch.setattr(llm_client, '_save_shared_cost', boom)
    _stale_ledger(tmp_path, monkeypatch)
    assert llm_client._cost_ok() is True
    assert 'daily rollover failed' in capsys.readouterr().out
    assert not _hist(tmp_path).exists()


# --- concurrency: O_APPEND single-write lines never interleave --------------

def test_concurrent_appends_yield_n_parseable_lines(tmp_path):
    n = 64
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        llm_client._append_cost_history(
            ('2000-01-%02d' % (i % 28 + 1), i / 1000.0), f'm{i:03d}', 0.0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = _lines(tmp_path)
    assert len(lines) == n
    recs = [json.loads(line) for line in lines]
    assert all(set(r) == _KEYS for r in recs)
    assert sorted(r['mem_date'] for r in recs) == [f'm{i:03d}'
                                                   for i in range(n)]


# --- byte-pin: the ledger JSON itself is unchanged ---------------------------

def test_ledger_json_bytes_unchanged(tmp_path, monkeypatch):
    _stale_ledger(tmp_path, monkeypatch)
    llm_client._maybe_reset_quota()
    today = _today()
    assert Path(llm_client._COST_FILE).read_bytes() == (
        '{"date": "%s", "cost": 0.0}' % today).encode()
    monkeypatch.setattr(llm_client, '_daily_cost', 0.1234567)
    llm_client._save_shared_cost()
    assert Path(llm_client._COST_FILE).read_bytes() == (
        '{"date": "%s", "cost": 0.123457}' % today).encode()
    assert sorted(os.listdir(tmp_path)) == ['llm_cost.json',
                                            'llm_cost.json.lock',
                                            'llm_cost_history.jsonl']
