"""Review batch b20 — log_config, stock_config, scripts/wave6_stage0.

Covers:
  - log_config: setup failure does not latch the _configured flag (fail loud,
    retry next call), concurrent first calls add exactly one handler pair,
    utf-8 file handler, numba/charset_normalizer suppression.
  - stock_config: load falls back (with a WARNING) on torn JSON and on
    non-string entries (AttributeError path) instead of crashing; save is
    atomic (tmp + os.replace, target intact if the replace never happens);
    CRYPTO_POOL comment no longer claims harvest wiring that does not exist.
  - wave6_stage0: horizons discovered from the data's TB_Bars_* columns (not
    the static list), wrong --fb names the available set, load/empty messages
    no longer blame the parquet, zero-measurement runs say NOTHING MEASURED
    and exit 1.

All tests are Mac-safe: pure stdlib + numpy/pandas, no heavy deps.
"""

import json
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stock_config
import wave6_stage0 as w6

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# log_config
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_log_config(monkeypatch, tmp_path):
    """log_config forced unconfigured, pointed at tmp_path; root restored."""
    import log_config
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    before_level = root.level
    monkeypatch.setattr(log_config, '_configured', False)
    monkeypatch.setattr(log_config, '_LOG_DIR', tmp_path / 'logs')
    monkeypatch.setattr(log_config, '_LOG_FILE', tmp_path / 'logs' / 'trader.log')
    yield log_config
    for h in list(root.handlers):
        if h not in before_handlers:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
    root.handlers[:] = before_handlers
    root.setLevel(before_level)


class TestLogConfig:
    def test_setup_failure_does_not_latch(self, fresh_log_config, monkeypatch):
        lc = fresh_log_config
        root = logging.getLogger()
        n_before = len(root.handlers)

        class Boom(OSError):
            pass

        def exploding_handler(*a, **k):
            raise Boom("disk full")

        monkeypatch.setattr(lc, 'RotatingFileHandler', exploding_handler)
        with pytest.raises(Boom):
            lc.get_logger('b20_fail')
        # Flag NOT latched and no half-added handlers on root.
        assert lc._configured is False
        assert len(root.handlers) == n_before

        # Transient failure clears -> the very next call configures fully.
        monkeypatch.setattr(lc, 'RotatingFileHandler', RotatingFileHandler)
        logger = lc.get_logger('b20_fail')
        assert logger.name == 'b20_fail'
        assert lc._configured is True
        assert len(root.handlers) == n_before + 2  # console + file

    def test_concurrent_first_calls_configure_once(self, fresh_log_config):
        lc = fresh_log_config
        root = logging.getLogger()
        n_before = len(root.handlers)
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker():
            barrier.wait()
            try:
                lc.get_logger('b20_thread')
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # Exactly ONE console+file pair despite 8 simultaneous first calls.
        assert len(root.handlers) == n_before + 2

    def test_file_handler_utf8_and_third_party_suppression(self, fresh_log_config):
        lc = fresh_log_config
        root = logging.getLogger()
        before_ids = {id(h) for h in root.handlers}
        logger = lc.get_logger('b20_enc')
        added = [h for h in root.handlers if id(h) not in before_ids]
        file_handlers = [h for h in added if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        fh = file_handlers[0]
        assert fh.encoding == 'utf-8'
        assert fh.maxBytes == lc._MAX_BYTES
        assert fh.backupCount == lc._BACKUP_COUNT
        # numba (byteflow/SSA DEBUG floods) + charset_normalizer now suppressed.
        for name in ('numba', 'charset_normalizer', 'urllib3', 'yfinance'):
            assert logging.getLogger(name).level == logging.WARNING
        # Repo log lines contain em-dashes; they must round-trip to the file.
        logger.debug("wave-9 gate — em-dash probe")
        fh.flush()
        text = lc._LOG_FILE.read_text(encoding='utf-8')
        assert "— em-dash probe" in text


# ---------------------------------------------------------------------------
# stock_config
# ---------------------------------------------------------------------------

class TestStockConfigLoad:
    def test_nonstring_entries_fall_back_with_warning(self, monkeypatch, tmp_path, caplog):
        p = tmp_path / 'stock_universe.json'
        p.write_text('["TSLA", 42]')  # element-type corruption -> AttributeError
        monkeypatch.setattr(stock_config, '_FILE', p)
        with caplog.at_level(logging.WARNING, logger='stock_config'):
            out = stock_config.load_stock_universe()
        assert out == list(stock_config._DEFAULTS)  # falls back, no crash
        warned = [r for r in caplog.records
                  if r.name == 'stock_config' and 'falling back' in r.getMessage()]
        assert len(warned) == 1

    def test_torn_json_falls_back_with_warning(self, monkeypatch, tmp_path, caplog):
        p = tmp_path / 'stock_universe.json'
        p.write_text('["TSLA", "NV')  # torn mid-write
        monkeypatch.setattr(stock_config, '_FILE', p)
        with caplog.at_level(logging.WARNING, logger='stock_config'):
            out = stock_config.load_stock_universe()
        assert out == list(stock_config._DEFAULTS)
        assert any('falling back' in r.getMessage() for r in caplog.records)

    def test_happy_path_unchanged_and_silent(self, monkeypatch, tmp_path, caplog):
        p = tmp_path / 'stock_universe.json'
        p.write_text('["msft", " tsla ", "btc/usd"]')
        monkeypatch.setattr(stock_config, '_FILE', p)
        with caplog.at_level(logging.WARNING, logger='stock_config'):
            out = stock_config.load_stock_universe()
        assert out == ['MSFT', 'TSLA', 'BTC/USD']
        assert not caplog.records


class TestStockConfigSave:
    def test_save_roundtrip_no_tmp_leftover(self, monkeypatch, tmp_path):
        p = tmp_path / 'stock_universe.json'
        p.write_text('["OLD"]')
        monkeypatch.setattr(stock_config, '_FILE', p)
        stock_config.save_stock_universe(['b', 'a', 'eth/usd', 'a'])
        assert json.loads(p.read_text()) == ['A', 'B', 'ETH/USD']
        assert not p.with_name(p.name + '.tmp').exists()

    def test_interrupted_save_leaves_target_intact(self, monkeypatch, tmp_path):
        """The dump goes to a sibling tmp; if the final rename never happens
        (crash / power loss), the target file is untouched — no torn JSON."""
        p = tmp_path / 'stock_universe.json'
        p.write_text('["OLD"]')
        monkeypatch.setattr(stock_config, '_FILE', p)

        def no_replace(src, dst):
            raise OSError("simulated crash before rename")

        monkeypatch.setattr(stock_config.os, 'replace', no_replace)
        with pytest.raises(OSError):
            stock_config.save_stock_universe(['NEW'])
        assert json.loads(p.read_text()) == ['OLD']  # target never torn


class TestCryptoPoolComment:
    def test_false_harvest_wiring_claim_removed(self):
        src = (REPO / 'stock_config.py').read_text(encoding='utf-8')
        # The old comment claimed the harvest reads CRYPTO_POOL — it does not.
        assert 'so the harvest can include them' not in src
        # The corrected comment must point at the list that actually needs
        # editing on the Jetson.
        assert 'CRYPTO_TICKERS' in src
        # And the underlying fact still holds: the harvest does NOT import it.
        harvest_src = (REPO / 'scripts' / 'harvest_crypto_data.py').read_text(
            encoding='utf-8')
        assert 'CRYPTO_POOL' not in harvest_src

    def test_crypto_pool_value_unchanged(self):
        assert stock_config.CRYPTO_POOL == stock_config.CRYPTO_SYMBOLS + [
            'AVAX/USD', 'BCH/USD', 'DOT/USD', 'LTC/USD']


# ---------------------------------------------------------------------------
# wave6_stage0
# ---------------------------------------------------------------------------

def _fake_panel():
    """Two tickers, spans harvested at fb=24 and fb=64 (64 is OUTSIDE the
    static FORWARD_BARS list — the adaptive-state drift case)."""
    n = 40
    return pd.DataFrame({
        'Ticker': ['AAA'] * 20 + ['BBB'] * 20,
        'Close': np.linspace(1.0, 2.0, n),  # feature col, must be ignored
        'TB_Bars_24': np.full(n, 3.0),
        'TB_Bars_64': np.full(n, 5.0),
    })


def _patch_loader(monkeypatch, fn):
    import data_utils
    monkeypatch.setattr(data_utils, 'load_training_data', fn)


class TestWave6Horizons:
    def test_harvested_horizons_parser(self):
        cols = ['Close', 'TB_Bars_24', 'TB_Bars_8', 'TB_Bars_x',
                'XTB_Bars_9', 'TB_Bars_12_extra']
        assert w6._harvested_horizons(cols) == [8, 24]

    def test_discovers_horizons_from_data_not_static_list(self, monkeypatch):
        _patch_loader(monkeypatch, lambda prefix, columns=None: _fake_panel())
        out = w6.measure_book('crypto')  # horizons=None -> discover
        assert out is not None
        assert sorted(out['horizons']) == [24, 64]  # 64 not in FORWARD_BARS
        assert out['rows'] == 40 and out['tickers'] == 2
        rec = out['horizons'][24]
        assert rec['n_labels'] == 40
        assert 0 < rec['n_eff'] <= 40
        assert rec['hold_bars_median'] == 3.0

    def test_unharvested_fb_names_available_set(self, monkeypatch, capsys):
        _patch_loader(monkeypatch, lambda prefix, columns=None: _fake_panel())
        out = w6.measure_book('crypto', horizons=[20])
        assert out is None
        text = capsys.readouterr().out
        assert 'horizon 20 not in this dataset' in text
        assert '[24, 64]' in text
        assert 're-run the harvest' not in text  # no longer blames the harvest

    def test_requested_fb_still_measured_when_present(self, monkeypatch):
        _patch_loader(monkeypatch, lambda prefix, columns=None: _fake_panel())
        out = w6.measure_book('crypto', horizons=[64])
        assert out is not None and list(out['horizons']) == [64]


class TestWave6Messages:
    def test_load_failure_message_no_longer_blames_parquet(self, monkeypatch, capsys):
        def raiser(prefix, columns=None):
            raise RuntimeError('disk gone')
        _patch_loader(monkeypatch, raiser)
        assert w6.measure_book('stock') is None
        text = capsys.readouterr().out
        assert 'could not load training data' in text
        assert 'parquet' not in text.lower()

    def test_empty_data_message_no_longer_blames_parquet(self, monkeypatch, capsys):
        _patch_loader(monkeypatch, lambda prefix, columns=None: pd.DataFrame())
        assert w6.measure_book('stock') is None
        text = capsys.readouterr().out
        assert 'no training data' in text
        assert 'parquet' not in text.lower()

    def test_docstring_claims_corrected(self):
        src = (REPO / 'scripts' / 'wave6_stage0.py').read_text(encoding='utf-8')
        assert 'reads ONLY' not in src            # false single-source claim
        assert 'exactly the n_eff' not in src     # false gate-parity claim
        assert 'load_training_data' in src.split('"""')[1]  # docstring names it


class TestWave6NothingMeasured:
    def test_verdict_flags_zero_measurement(self, capsys):
        ok = w6.verdict([None, {'book': 'stock', 'rows': 0, 'tickers': 0,
                                'horizons': {}}])
        text = capsys.readouterr().out
        assert ok is False
        assert 'NOTHING MEASURED' in text
        # The closing exhortation must not refer to numbers that don't exist.
        assert 'Use the realized per-book' not in text

    def test_verdict_true_when_measured(self, capsys):
        r = {'book': 'crypto', 'rows': 10, 'tickers': 1,
             'horizons': {24: {'u_bar_mean': 0.2}}}
        assert w6.verdict([r, None]) is True
        text = capsys.readouterr().out
        assert 'NON-IID' in text
        assert 'Use the realized per-book' in text

    def test_main_exits_1_and_writes_empty_json(self, monkeypatch, tmp_path, capsys):
        _patch_loader(monkeypatch, lambda prefix, columns=None: pd.DataFrame())
        jpath = tmp_path / 'out.json'
        monkeypatch.setattr(sys, 'argv', ['wave6_stage0.py', '--json', str(jpath)])
        with pytest.raises(SystemExit) as excinfo:
            w6.main()
        assert excinfo.value.code == 1
        assert json.loads(jpath.read_text()) == []  # wrapper-readable "nothing"
        assert 'NOTHING MEASURED' in capsys.readouterr().out

    def test_main_exits_0_when_measured(self, monkeypatch, capsys):
        _patch_loader(monkeypatch, lambda prefix, columns=None: _fake_panel())
        monkeypatch.setattr(sys, 'argv', ['wave6_stage0.py', '--book', 'crypto'])
        w6.main()  # no SystemExit -> exit code 0
        assert 'NOTHING MEASURED' not in capsys.readouterr().out
