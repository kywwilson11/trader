"""Tests for the 'data' review group — data_sources, data_utils, trade_journal,
trade_memory. Focused on GAPS not already covered by test_data_utils.py /
test_data_sources.py / test_trade_journal.py. Biggest gap: trade_memory had
zero dedicated coverage, including its locking contract.

All tests here are Mac-runnable: CSV-only / mock-only, no to_parquet/read_parquet
(pyarrow is absent on this dev Mac).
"""

import datetime
import json
import os
import threading
import time
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# trade_memory.py
# ---------------------------------------------------------------------------

def test_record_and_load_roundtrip(tmp_path):
    from trade_memory import record_trade, load_all
    mem_file = tmp_path / "trade_memory.json"
    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        record_trade('BTC/USD', 'sell', 100, 110, 10.0,
                      llm_score=0.8, exit_reason='take_profit')
        data = load_all()

    assert 'BTC/USD' in data
    records = data['BTC/USD']
    assert len(records) == 1
    rec = records[0]
    assert set(rec.keys()) == {
        'ts', 'action', 'entry', 'exit', 'pnl_pct', 'llm_score',
        'reasoning', 'news', 'exit_reason', 'estimated',
    }
    assert rec['entry'] == 100
    assert rec['exit'] == 110
    assert rec['pnl_pct'] == 10.0


def test_rolling_window_trim(tmp_path):
    from trade_memory import record_trade, load_all
    mem_file = tmp_path / "trade_memory.json"
    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        for i in range(60):
            record_trade('BTC/USD', 'sell', 100, 110, float(i))
        data = load_all()

    records = data['BTC/USD']
    assert len(records) == 50
    assert min(r['pnl_pct'] for r in records) == 10.0


def test_estimated_flag_stored(tmp_path):
    from trade_memory import record_trade, load_all
    mem_file = tmp_path / "trade_memory.json"
    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        record_trade('ETH/USD', 'sell', 100, 105, 5.0, estimated=True)
        data = load_all()

    assert data['ETH/USD'][0]['estimated'] is True


def test_get_lesson_summary_format(tmp_path):
    from trade_memory import record_trade, get_lesson_summary
    mem_file = tmp_path / "trade_memory.json"
    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        record_trade('BTC/USD', 'sell', 100, 110, 10.0,
                      llm_score=0.8, exit_reason='take_profit')
        summary = get_lesson_summary('BTC/USD')
        empty = get_lesson_summary('NONE')

    assert summary == ('Last 1 trades: 1W/0L, avg PnL +10.00%, '
                        'most common exit: take_profit')
    assert empty == ''


def test_corrupt_file_quarantined(tmp_path):
    from trade_memory import _load
    mem_file = tmp_path / "trade_memory.json"
    mem_file.write_text('[1,2,3]')

    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        result = _load()

    assert result == {}
    corrupt_file = tmp_path / "trade_memory.json.corrupt"
    assert corrupt_file.exists()
    assert corrupt_file.read_text() == '[1,2,3]'


def test_concurrency_no_lost_updates(tmp_path):
    """LOCKING CONTRACT: 4 threads, distinct symbols, no lost updates.

    Distinct symbols are the correct discriminator: without _write_lock the
    whole-dict load-modify-save clobbers entire symbols; a same-symbol count
    would stay pinned at the 50 cap and hide the race.

    Note: the cross-process flock path (_cross_process_lock, fcntl on a
    sidecar file) is not unit-tested here — that requires actual separate
    processes and is exercised on the Jetson / manually.
    """
    from trade_memory import record_trade, load_all
    mem_file = tmp_path / "trade_memory.json"

    def _worker(sym):
        for i in range(10):
            record_trade(sym, 'sell', 100, 110, float(i))

    with mock.patch("trade_memory._MEMORY_FILE", mem_file):
        threads = [threading.Thread(target=_worker, args=(sym,))
                   for sym in ['AAA', 'BBB', 'CCC', 'DDD']]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = load_all()

    assert set(data.keys()) == {'AAA', 'BBB', 'CCC', 'DDD'}
    total = sum(len(v) for v in data.values())
    assert total == 40


# ---------------------------------------------------------------------------
# data_utils.py
# ---------------------------------------------------------------------------

def test_stock_gap_overnight_not_flagged():
    # _stock_gap_spans_trading_days returns np.bool_ (np.busday_count >= 2),
    # so we assert truthiness rather than `is False`/`is True` identity.
    from data_utils import _stock_gap_spans_trading_days
    gap_start = pd.Timestamp('2024-01-02 16:00', tz='UTC')  # Tue
    gap_end = pd.Timestamp('2024-01-03 09:30', tz='UTC')    # Wed
    dur = gap_end - gap_start
    assert not _stock_gap_spans_trading_days(gap_end, dur)


def test_stock_gap_weekend_not_flagged():
    from data_utils import _stock_gap_spans_trading_days
    gap_start = pd.Timestamp('2024-01-05 16:00', tz='UTC')  # Fri
    gap_end = pd.Timestamp('2024-01-08 09:30', tz='UTC')    # Mon
    dur = gap_end - gap_start
    assert not _stock_gap_spans_trading_days(gap_end, dur)


def test_stock_gap_single_holiday_not_flagged():
    from data_utils import _stock_gap_spans_trading_days
    gap_start = pd.Timestamp('2024-01-03 16:00', tz='UTC')  # Wed
    gap_end = pd.Timestamp('2024-01-05 09:30', tz='UTC')    # Fri (Thu holiday)
    dur = gap_end - gap_start
    assert not _stock_gap_spans_trading_days(gap_end, dur)


def test_stock_gap_two_weekdays_flagged():
    from data_utils import _stock_gap_spans_trading_days

    gap_start = pd.Timestamp('2024-01-02 16:00', tz='UTC')  # Tue
    gap_end = pd.Timestamp('2024-01-05 09:30', tz='UTC')    # Fri
    dur = gap_end - gap_start
    assert _stock_gap_spans_trading_days(gap_end, dur)

    gap_start2 = pd.Timestamp('2024-01-05 16:00', tz='UTC')  # Fri
    gap_end2 = pd.Timestamp('2024-01-10 09:30', tz='UTC')    # next Wed
    dur2 = gap_end2 - gap_start2
    assert _stock_gap_spans_trading_days(gap_end2, dur2)


def test_stale_parquet_prefers_csv(tmp_path):
    from data_utils import _csv_is_fresher, get_data_path, load_training_data

    with mock.patch('data_utils._BASE_DIR', tmp_path):
        parquet_path = tmp_path / 'training_data.parquet'
        csv_path = tmp_path / 'training_data.csv'

        parquet_path.write_bytes(b'GARBAGE NOT A REAL PARQUET FILE')
        df = pd.DataFrame({'Close': [1.0, 2.0]})
        df.to_csv(csv_path)

        old_time = time.time() - 10000
        os.utime(parquet_path, (old_time, old_time))

        assert _csv_is_fresher(parquet_path, csv_path) is True

        path = get_data_path('crypto')
        assert str(path).endswith('.csv')

        loaded = load_training_data('crypto')
        assert len(loaded) == 2


def test_save_returns_true_csv_only(tmp_path):
    from data_utils import save_training_data, load_training_data

    with mock.patch('data_utils._BASE_DIR', tmp_path):
        df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]})
        result = save_training_data(df, 'crypto')

        # Portable: on this Mac (no pyarrow) the parquet write fails
        # gracefully and only CSV persists; on the Jetson both formats write.
        assert result is True
        assert (tmp_path / 'training_data.csv').exists()

        loaded = load_training_data('crypto')
        assert len(loaded) == len(df)


def test_validate_detects_inf():
    from data_utils import validate_training_data
    df = pd.DataFrame({
        'Close': [1.0, np.inf, 3.0],
        'Ticker': ['BTC-USD'] * 3,
    }, index=pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'))

    report = validate_training_data(df, 'crypto')
    inf_cols = [c['column'] for c in report['inf_columns']]
    assert 'Close' in inf_cols


# ---------------------------------------------------------------------------
# trade_journal.py
# ---------------------------------------------------------------------------

def test_journal_disabled_drops_rows(tmp_path):
    from trade_journal import log_decision
    jdir = tmp_path / "journals"
    jdir.mkdir()

    with mock.patch("trade_journal.JOURNAL_DIR", jdir), \
         mock.patch("trade_journal.load_llm_config",
                     return_value={"journal_enabled": False}):
        log_decision({"action": "buy"})

    today = datetime.date.today().isoformat()
    filepath = jdir / f"{today}.jsonl"
    assert not filepath.exists()


def test_iter_journal_rows_order_and_corrupt_skip(tmp_path):
    from trade_journal import iter_journal_rows
    jdir = tmp_path / "journals"
    jdir.mkdir()

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    (jdir / f"{yesterday.isoformat()}.jsonl").write_text(
        json.dumps({"n": 1}) + "\n")
    (jdir / f"{today.isoformat()}.jsonl").write_text(
        json.dumps({"n": 2}) + "\n{bad json\n" + json.dumps({"n": 3}) + "\n")

    with mock.patch("trade_journal.JOURNAL_DIR", jdir):
        rows = list(iter_journal_rows(days=2))

    assert [r['n'] for r in rows] == [1, 2, 3]


def test_summary_counts_skipped_torn_line(tmp_path):
    from trade_journal import get_journal_summary
    jdir = tmp_path / "journals"
    jdir.mkdir()

    today = datetime.date.today().isoformat()
    filepath = jdir / f"{today}.jsonl"
    filepath.write_text(json.dumps({"action": "sell"}) + "\n{torn line not json\n")

    with mock.patch("trade_journal.JOURNAL_DIR", jdir):
        summary = get_journal_summary(today)

    assert summary['skipped_lines'] == 1
    assert summary['sells'] == 1
