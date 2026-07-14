"""Review batch b05 — data_utils, data_sources, sentiment_history fixes.

Covers:
  - data_utils: stale-parquet removal on failed parquet save, save success
    flag, CSV-freshness preference on read, atomic migration, deterministic
    tmp names + 0644 modes, stock gap-check calendar awareness, dead-code
    removal (get_latest_timestamp).
  - data_sources: urlopen context-manager hygiene, docstring contract.
  - sentiment_history: unimportable on the dev Mac (dotenv), so source-guard
    tests (pattern from test_prediction_cache.py) plus a real-sqlite3 test of
    the INSERT OR IGNORE rowcount mechanism the article counter now relies on.
"""
import os
import sqlite3
import stat
import time
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# data_utils
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_dir(tmp_path):
    """Redirect data_utils to a temp directory."""
    with mock.patch('data_utils._BASE_DIR', tmp_path):
        yield tmp_path


def _df(n=10):
    idx = pd.date_range('2024-01-01', periods=n, freq='h', tz='UTC')
    return pd.DataFrame({'Close': range(n), 'Ticker': 'BTC-USD'}, index=idx)


def _boom(self, *args, **kwargs):
    raise RuntimeError('simulated write failure')


def _fake_parquet_writer(self, path, **kwargs):
    Path(path).write_bytes(b'PQFAKE')


def test_save_removes_stale_parquet_when_parquet_write_fails(base_dir, capsys):
    """Failed parquet + successful CSV must not leave a stale parquet that
    shadows the fresh CSV in every existence-based loader."""
    from data_utils import save_training_data, get_data_path

    stale = base_dir / 'training_data.parquet'
    stale.write_bytes(b'old frozen parquet')

    with mock.patch.object(pd.DataFrame, 'to_parquet', new=_boom):
        ok = save_training_data(_df(), 'crypto')

    assert ok is True                       # CSV persisted the frame
    assert not stale.exists()               # stale parquet removed
    assert (base_dir / 'training_data.csv').exists()
    assert str(get_data_path('crypto')).endswith('.csv')
    assert 'removed stale' in capsys.readouterr().out


def test_save_returns_false_when_both_writes_fail(base_dir, capsys):
    """Both writes failing must be detectable by callers and loud."""
    from data_utils import save_training_data

    old_csv = base_dir / 'training_data.csv'
    old_csv.write_text('old,data\n1,2\n')

    with mock.patch.object(pd.DataFrame, 'to_parquet', new=_boom), \
         mock.patch.object(pd.DataFrame, 'to_csv', new=_boom):
        ok = save_training_data(_df(), 'crypto')

    assert ok is False
    assert old_csv.read_text() == 'old,data\n1,2\n'   # old copy never destroyed
    assert list(base_dir.glob('*.tmp')) == []          # no tmp litter
    assert 'STALE' in capsys.readouterr().out


def test_save_tmp_is_deterministic_and_mode_644(base_dir):
    """Crash orphans get overwritten (not accumulated) and files are 0644."""
    from data_utils import save_training_data

    orphan = base_dir / 'training_data.csv.tmp'
    orphan.write_text('orphan from a previous OOM-kill')

    with mock.patch.object(pd.DataFrame, 'to_parquet', new=_boom):
        assert save_training_data(_df(), 'crypto') is True

    assert not orphan.exists()              # consumed by os.replace, not piled up
    assert list(base_dir.glob('*.tmp')) == []
    csv_path = base_dir / 'training_data.csv'
    assert stat.S_IMODE(os.stat(csv_path).st_mode) == 0o644


def test_get_data_path_prefers_fresh_csv_over_stale_parquet(base_dir):
    from data_utils import get_data_path

    pq = base_dir / 'training_data.parquet'
    csv = base_dir / 'training_data.csv'
    pq.write_bytes(b'junk')
    csv.write_text('a,b\n1,2\n')

    now = time.time()
    os.utime(pq, (now, now))
    os.utime(csv, (now, now))
    assert str(get_data_path('crypto')).endswith('.parquet')  # same age -> parquet

    os.utime(pq, (now - 7200, now - 7200))                    # csv 2h newer
    assert str(get_data_path('crypto')).endswith('.csv')


def test_load_training_data_prefers_fresh_csv(base_dir, capsys):
    """A CSV >1h newer than the parquet is served instead of the frozen parquet."""
    from data_utils import load_training_data

    pq = base_dir / 'training_data.parquet'
    pq.write_bytes(b'not a real parquet')
    _df(5).to_csv(base_dir / 'training_data.csv')

    now = time.time()
    os.utime(pq, (now - 7200, now - 7200))

    df = load_training_data('crypto')
    assert len(df) == 5
    assert 'treating parquet as stale' in capsys.readouterr().out


def test_migrate_csv_to_parquet_is_atomic(base_dir):
    """Migration writes via tmp + os.replace; a failed write leaves nothing."""
    from data_utils import migrate_csv_to_parquet

    _df(5).to_csv(base_dir / 'training_data.csv')
    pq = base_dir / 'training_data.parquet'

    with mock.patch.object(pd.DataFrame, 'to_parquet', new=_boom):
        assert migrate_csv_to_parquet('crypto') is False
    assert not pq.exists()                  # no corrupt partial file
    assert list(base_dir.glob('*.tmp')) == []

    with mock.patch.object(pd.DataFrame, 'to_parquet', new=_fake_parquet_writer):
        assert migrate_csv_to_parquet('crypto') is True
    assert pq.read_bytes() == b'PQFAKE'
    assert list(base_dir.glob('*.tmp')) == []


def _stock_df(days, hours=range(14, 20)):
    """Weekday-only hourly bars (UTC), like an RTH stock dataset."""
    idx = [pd.Timestamp(f'{d} {h:02d}:00', tz='UTC') for d in days for h in hours]
    return pd.DataFrame({'Close': range(len(idx)), 'Ticker': 'AAPL'},
                        index=pd.DatetimeIndex(idx))


def test_validate_stock_weekends_and_overnights_not_flagged(base_dir):
    from data_utils import validate_training_data

    # Two full trading weeks: overnight (19h) and weekend (67h) gaps only.
    days = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
            '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12']
    report = validate_training_data(_stock_df(days), 'stock')
    assert report['gaps'] == []


def test_validate_stock_multiday_outage_flagged(base_dir):
    from data_utils import validate_training_data

    # Wed 2024-01-10 and Thu 2024-01-11 missing entirely -> 2 full weekdays.
    days = ['2024-01-08', '2024-01-09', '2024-01-12']
    report = validate_training_data(_stock_df(days), 'stock')
    assert len(report['gaps']) == 1
    assert report['gaps'][0]['duration_h'] == pytest.approx(67.0)


def test_validate_stock_single_holiday_not_flagged(base_dir):
    from data_utils import validate_training_data

    # MLK Monday 2024-01-15 closed: Fri -> Tue spans only 1 full weekday.
    days = ['2024-01-11', '2024-01-12', '2024-01-16', '2024-01-17']
    report = validate_training_data(_stock_df(days), 'stock')
    assert report['gaps'] == []


def test_validate_crypto_gap_rule_unchanged(base_dir):
    from data_utils import validate_training_data

    idx = list(pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC'))
    idx.append(pd.Timestamp('2024-01-01 15:00', tz='UTC'))  # 6h gap
    df = pd.DataFrame({'Close': range(len(idx)), 'Ticker': 'BTC-USD'},
                      index=pd.DatetimeIndex(idx))
    report = validate_training_data(df, 'crypto')
    assert len(report['gaps']) == 1


def test_get_latest_timestamp_deleted():
    """Dead helper (zero callers; harvest scripts carry their own copies)."""
    import data_utils
    assert not hasattr(data_utils, 'get_latest_timestamp')


# ---------------------------------------------------------------------------
# data_sources
# ---------------------------------------------------------------------------

def test_cc_urlopen_closed_via_context_manager():
    """The response socket is released deterministically, not left to GC."""
    import json as _json
    from data_sources import fetch_cryptocompare_hourly

    base_ts = int(pd.Timestamp('2024-01-01', tz='UTC').timestamp())
    bars = [{'time': base_ts + i * 3600, 'open': 1, 'high': 1, 'low': 1,
             'close': 1.5, 'volumefrom': 10} for i in range(3)]
    payload = {'Response': 'Success', 'Data': {'Data': bars}}
    exits = []

    class Resp:
        def read(self):
            return _json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            exits.append(True)

    with mock.patch('data_sources.urllib.request.urlopen', return_value=Resp()):
        df = fetch_cryptocompare_hourly('BTC-USD', '2024-01-01', '2024-01-01 03:00')

    assert exits, "urlopen response was not used as a context manager"
    assert df is not None and len(df) == 3


def test_fetch_with_fallback_docstring_matches_stock_gate():
    from data_sources import fetch_with_fallback
    doc = fetch_with_fallback.__doc__
    assert 'Merges all sources' not in doc     # old wrong contract
    assert 'never merged' in doc               # stocks: exclusive fallback


# ---------------------------------------------------------------------------
# sentiment_history (source guards — module needs dotenv, unavailable here)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def sh_src():
    return (REPO / 'sentiment_history.py').read_text()


def test_sh_source_compiles(sh_src):
    compile(sh_src, 'sentiment_history.py', 'exec')


def test_sh_dead_code_removed(sh_src):
    assert 'import math' not in sh_src
    assert 'cached_symbols' not in sh_src
    assert 'except sqlite3.IntegrityError' not in sh_src  # unreachable w/ OR IGNORE
    assert '.rowcount' in sh_src                          # real-insert counting


def test_sh_docstrings_updated(sh_src):
    assert '7-day windows' not in sh_src
    assert '30-day windows' in sh_src
    assert 'GUI display' not in sh_src


def test_sh_poll_handles_permanent_failures(sh_src):
    # 4xx (minus transient 408/429) clears the dead job instead of 'pending'.
    assert 'except urllib.error.HTTPError' in sh_src
    assert '(408, 429)' in sh_src
    # Staleness guard actually reads the persisted submitted_at.
    assert '_BATCH_MAX_AGE_H' in sh_src
    assert "state['submitted_at']" in sh_src


def test_sh_failed_batch_gets_cooldown(sh_src):
    # submit-failure AND poll-failure paths both arm the 1h sync fallback.
    assert sh_src.count('_batch_unavailable_until = time.time() + 3600') == 2


def test_sh_ingest_uses_metadata_key(sh_src):
    assert "int(key.split('-')[1])" in sh_src
    assert "(item.get('metadata') or {}).get('key')" in sh_src


def test_sh_live_fng_failure_logged_once(sh_src):
    assert '_live_fng_warned' in sh_src
    assert 'live FnG failed' in sh_src


def test_insert_or_ignore_rowcount_mechanism():
    """The article counter now uses cursor.rowcount — verify sqlite3 reports
    1 for a real insert and 0 for an OR-IGNOREd duplicate."""
    db = sqlite3.connect(':memory:')
    db.execute("""CREATE TABLE articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL, date TEXT NOT NULL, headline TEXT NOT NULL,
        keyword_score REAL NOT NULL,
        UNIQUE(symbol, date, headline))""")
    ins = "INSERT OR IGNORE INTO articles (symbol, date, headline, keyword_score) VALUES (?,?,?,?)"
    row = ('AAPL', '2026-01-01', 'Apple beats earnings', 0.3)
    assert db.execute(ins, row).rowcount == 1   # real insert
    assert db.execute(ins, row).rowcount == 0   # duplicate ignored, not raised
    assert db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 1
