"""Review batch b08: funding.py / funding_archive.py / oi_archive.py.

Covers the applied fixes: silent-failure logging (archive loads, history
persistence, live OKX fetches), the urlopen context manager, z in the
crowded-tilt log, funding_archive sync robustness (parse guard, junk-cell
coerce, listing-month floor, distinct failure summary), oi_archive sync
robustness (parse guard, consecutive-failure abort), corrupt-archive
preservation, the oi_history.json write lock, negative-result caching on
live fetches, OKX-map single-sourcing, and dead-code removal — plus the
core funding_archive kernel coverage the review flagged as missing
(_months, _parse_zip, sd==0 z-mask, ffill PIT alignment, warmup cutoff).

Everything runs on the dev Mac: numpy/pandas/pytest only, network is
monkeypatched, and no test needs a parquet engine (pyarrow is absent
here, so parquet writes are stubbed where a sync flush is asserted).
"""

import datetime as dt
import inspect
import io
import json
import logging
import sys
import threading
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import funding
import funding_archive
import oi_archive

REPO = Path(__file__).resolve().parent.parent


# --- fixtures -----------------------------------------------------------

@pytest.fixture
def funding_env(tmp_path, monkeypatch):
    """Isolate funding.py module state and its history file."""
    monkeypatch.setattr(funding, '_HISTORY_FILE', str(tmp_path / 'fh.json'))
    monkeypatch.setattr(funding, '_cache', {})
    monkeypatch.setattr(funding, '_history', None)
    monkeypatch.setattr(funding, '_save_warned', False)
    return tmp_path


@pytest.fixture
def farc_env(tmp_path, monkeypatch):
    monkeypatch.setattr(funding_archive, 'ARCHIVE_FILE',
                        tmp_path / 'funding.parquet')
    return tmp_path


@pytest.fixture
def oi_env(tmp_path, monkeypatch):
    monkeypatch.setattr(oi_archive, 'ARCHIVE_FILE', tmp_path / 'oi.parquet')
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE',
                        tmp_path / 'oi_history.json')
    monkeypatch.setattr(oi_archive, '_live_cache', {})
    monkeypatch.setattr(oi_archive, '_ls_cache', {})
    monkeypatch.setattr(oi_archive, '_taker_cache', {})
    monkeypatch.setattr(oi_archive, '_fail_cache', {})
    monkeypatch.setattr(oi_archive, '_hist_read_warned', False)
    monkeypatch.setattr(oi_archive, '_hist_write_warned', False)
    return tmp_path


# --- helpers ------------------------------------------------------------

class _Resp:
    """Fake urlopen response that records context-manager exit."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self.exited = False

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False


def _funding_zip(month='2026-05', n=90, rate=0.0001, junk_time_rows=0):
    """Synthetic Binance monthly fundingRate zip (headered CSV)."""
    t0 = pd.Timestamp(f'{month}-01T00:00:00Z')
    lines = ['symbol,fundingTime,fundingRate']
    for i in range(n):
        ts_ms = int((t0 + pd.Timedelta(hours=8 * i)).value // 10 ** 6)
        lines.append(f'BTCUSDT,{ts_ms},{rate}')
    for _ in range(junk_time_rows):
        lines.append(f'BTCUSDT,junk,{rate}')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f'BTCUSDT-fundingRate-{month}.csv', '\n'.join(lines))
    return buf.getvalue()


def _month_ago(k: int) -> str:
    today = dt.date.today()
    y, m = today.year, today.month - k
    while m < 1:
        m += 12
        y -= 1
    return f'{y:04d}-{m:02d}'


# --- funding.py ---------------------------------------------------------

def test_get_funding_rate_closes_response(funding_env, monkeypatch):
    """urlopen is used as a context manager (socket freed deterministically)."""
    resp = _Resp(json.dumps({'data': [{'fundingRate': '0.0002'}]}).encode())
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=10: resp)
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(0.0002)
    assert resp.exited


def test_save_history_warns_once_on_oserror(funding_env, tmp_path,
                                            monkeypatch, caplog):
    monkeypatch.setattr(funding, '_HISTORY_FILE',
                        str(tmp_path / 'no_such_dir' / 'fh.json'))
    monkeypatch.setattr(funding, '_history', {'BTC/USD': [0.0001]})
    with caplog.at_level(logging.DEBUG, logger='funding'):
        funding._save_history()
        funding._save_history()  # second failure must not warn again
    warns = [r for r in caplog.records
             if 'history persist failed' in r.getMessage()]
    assert len(warns) == 1
    assert warns[0].levelno == logging.WARNING


def test_live_features_log_archive_failure_and_fall_back(funding_env,
                                                         monkeypatch, caplog):
    def _boom(sym):
        raise RuntimeError('corrupt parquet')

    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: 0.0001)
    monkeypatch.setattr(funding_archive, 'get_funding_series', _boom)
    monkeypatch.setattr(funding, '_history', {})
    with caplog.at_level(logging.DEBUG, logger='funding'):
        out = funding.live_funding_features('BTC/USD')
    # Fail-open (degraded fallback) is unchanged, but no longer silent
    assert out == pytest.approx({'Funding_Rate_Ann': 0.0001 * 3 * 365,
                                 'Funding_Z': 0.0, 'Funding_Chg_24h': 0.0})
    assert 'archive unavailable' in caplog.text
    assert 'corrupt parquet' in caplog.text


def test_crowded_log_shows_z_when_z_triggered(funding_env, monkeypatch,
                                              caplog):
    """z-only trigger (annualized ~11%/yr, well under 30%) must log its z."""
    mu, sd = 0.0001, 1e-6
    hist = [mu - sd, mu + sd] * 15          # 30 samples, pstdev == sd
    rate = mu + 2.5 * sd                    # z = 2.5, ann ~= 11.2%/yr
    monkeypatch.setattr(funding, '_history', {'BTC/USD': hist})
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: rate)
    with caplog.at_level(logging.INFO, logger='funding'):
        assert funding.funding_tilt('BTC/USD') == 0.6
    assert 'crowded longs' in caplog.text
    assert 'z=2.5' in caplog.text


def test_crowded_log_shows_na_without_history(funding_env, monkeypatch,
                                              caplog):
    monkeypatch.setattr(funding, '_history', {})
    monkeypatch.setattr(funding, 'get_funding_rate', lambda s: 0.0004)
    with caplog.at_level(logging.INFO, logger='funding'):
        assert funding.funding_tilt('BTC/USD') == 0.6  # ann ~43.8%/yr
    assert 'z=n/a' in caplog.text


# --- funding_archive.py: _parse_zip -------------------------------------

def test_parse_zip_happy_path():
    out = funding_archive._parse_zip(_funding_zip(n=60, rate=0.0003))
    assert out is not None and len(out) == 60
    assert out['ts'].dt.tz is not None
    assert out['rate'].iloc[0] == pytest.approx(0.0003)


def test_parse_zip_drops_junk_time_rows():
    """A junk cell in the time column is coerced+dropped, not a ValueError."""
    out = funding_archive._parse_zip(_funding_zip(n=50, junk_time_rows=2))
    assert out is not None and len(out) == 50


def test_parse_zip_unidentifiable_columns_warns(capsys):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('x.csv', 'foo,bar\n1,2\n3,4')
    assert funding_archive._parse_zip(buf.getvalue()) is None
    assert 'could not identify' in capsys.readouterr().out


# --- funding_archive.py: load/sync robustness ----------------------------

def test_funding_load_archive_corrupt_prints(farc_env, capsys):
    funding_archive.ARCHIVE_FILE.write_bytes(b'not parquet')
    arc = funding_archive.load_archive()
    assert arc.empty and list(arc.columns) == ['symbol', 'ts', 'rate']
    assert 'corrupt archive' in capsys.readouterr().out


def test_sync_survives_bad_zip_and_reports_failures(farc_env, capsys,
                                                    monkeypatch):
    """Non-zip 200 body must not abort the sync or read as 'up to date'."""
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=30: io.BytesIO(b'not a zip'))
    ok = funding_archive.sync(symbols=['BTC/USD'], start=_month_ago(2))
    out = capsys.readouterr().out
    assert 'parse failed' in out
    assert 'up to date' not in out
    assert 'month-fetches failed' in out
    assert ok is False
    assert not funding_archive.ARCHIVE_FILE.exists()


def test_sync_flushes_good_months_despite_bad_ones(farc_env, capsys,
                                                   monkeypatch):
    """The month AFTER a corrupt one still lands (pre-fix: total abort)."""
    m_bad, m_good = _month_ago(2), _month_ago(1)
    good_zip = _funding_zip(month=m_good, n=90)

    def fake(req, timeout=30):
        if m_bad in req.full_url:
            return io.BytesIO(b'garbage')
        return io.BytesIO(good_zip)

    written = {}

    def fake_to_parquet(self, path, *a, **k):  # no pyarrow on the dev Mac
        written['df'] = self.copy()
        Path(path).write_bytes(b'parquet-stub')

    monkeypatch.setattr(urllib.request, 'urlopen', fake)
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fake_to_parquet)
    ok = funding_archive.sync(symbols=['BTC/USD'], start=m_bad)
    out = capsys.readouterr().out
    assert ok is True
    assert 'parse failed' in out and 'synced' in out and 'failed' in out
    assert len(written['df']) == 90
    assert funding_archive.ARCHIVE_FILE.exists()


def test_sync_404s_still_report_up_to_date(farc_env, capsys, monkeypatch):
    def _404(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 404, 'nf', None, None)

    monkeypatch.setattr(urllib.request, 'urlopen', _404)
    ok = funding_archive.sync(symbols=['BTC/USD'], start=_month_ago(2))
    out = capsys.readouterr().out
    assert 'up to date' in out and 'failed' not in out
    assert ok is False


def test_sync_network_errors_not_reported_up_to_date(farc_env, capsys,
                                                     monkeypatch):
    def _down(req, timeout=30):
        raise urllib.error.URLError('unreachable')

    monkeypatch.setattr(urllib.request, 'urlopen', _down)
    funding_archive.sync(symbols=['BTC/USD'], start=_month_ago(2))
    out = capsys.readouterr().out
    assert 'up to date' not in out
    assert 'month-fetches failed' in out


# --- funding_archive.py: listing floor -----------------------------------

def test_listing_floor_skips_prelisting_months(farc_env, monkeypatch):
    requested = []

    def _404(req, timeout=30):
        requested.append(req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, 'nf', None, None)

    monkeypatch.setattr(urllib.request, 'urlopen', _404)
    funding_archive.sync(symbols=['SOL/USD'], start='2020-01')
    months = sorted(u.rsplit('-fundingRate-', 1)[1][:7] for u in requested)
    assert months[0] == '2020-09'           # floor applied: no 2020-01..08
    assert len(months) == len(funding_archive._months('2020-09'))

    requested.clear()
    funding_archive.sync(symbols=['BTC/USD'], start='2020-01')
    months = sorted(u.rsplit('-fundingRate-', 1)[1][:7] for u in requested)
    assert months[0] == '2020-01'           # later explicit start respected


def test_listing_floor_covers_universe():
    assert set(funding_archive.LISTING_MONTH) == set(
        funding_archive.BINANCE_SYMBOLS)


# --- funding_archive.py: core kernel (coverage the review found missing) --

def test_months_boundaries():
    months = funding_archive._months('2020-01')
    assert months[0] == '2020-01'
    today = dt.date.today()
    this_month = f'{today.year:04d}-{today.month:02d}'
    assert this_month not in months          # only COMPLETE months

    def nxt(m):
        y, mm = map(int, m.split('-'))
        mm += 1
        if mm > 12:
            y, mm = y + 1, 1
        return f'{y:04d}-{mm:02d}'

    assert all(nxt(a) == b for a, b in zip(months, months[1:]))  # no gaps
    assert nxt(months[-1]) == this_month
    assert funding_archive._months(this_month) == []


def test_funding_features_pit_no_lookahead(monkeypatch):
    """A bar strictly before a funding print must never see that print."""
    idx8h = pd.date_range('2026-01-01', periods=100, freq='8h', tz='UTC')
    s = pd.Series(0.0001, index=idx8h)
    spike_t = idx8h[-1]
    s.iloc[-1] = 0.001
    monkeypatch.setattr(funding_archive, 'get_funding_series', lambda sym: s)
    bars = pd.date_range(spike_t - pd.Timedelta(hours=3), periods=6,
                         freq='h', tz='UTC')
    out = funding_archive.funding_features_for_index('BTC/USD', bars)
    ann = out['Funding_Rate_Ann']
    pre = np.asarray(bars < spike_t)
    assert np.allclose(ann[pre], 0.0001 * 3 * 365)
    assert np.allclose(ann[~pre], 0.001 * 3 * 365)


def test_funding_features_flat_z_zero_and_warmup_none(monkeypatch):
    idx8h = pd.date_range('2026-01-01', periods=120, freq='8h', tz='UTC')
    flat = pd.Series(0.0001, index=idx8h)
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda sym: flat)
    bars = pd.date_range('2026-01-20', periods=24, freq='h', tz='UTC')
    out = funding_archive.funding_features_for_index('BTC/USD', bars)
    # sd==0 stretches are z=0, never NaN (NaN would drop bars at dropna)
    assert np.nanmax(np.abs(out['Funding_Z'])) == pytest.approx(0.0)

    short = flat.iloc[:39]                   # < 40 prints -> None
    monkeypatch.setattr(funding_archive, 'get_funding_series',
                        lambda sym: short)
    assert funding_archive.funding_features_for_index('BTC/USD', bars) is None


# --- oi_archive.py: archive side -----------------------------------------

def test_okx_map_single_sourced_from_funding():
    assert oi_archive.OKX_INSTRUMENTS is funding.OKX_INSTRUMENTS
    assert 'from funding import OKX_INSTRUMENTS' in (
        REPO / 'oi_archive.py').read_text()


def test_oi_load_archive_preserves_corrupt_file(oi_env, capsys):
    oi_archive.ARCHIVE_FILE.write_bytes(b'junk')
    arc = oi_archive.load_archive()
    assert arc.empty
    assert 'corrupt archive' in capsys.readouterr().out
    assert not oi_archive.ARCHIVE_FILE.exists()       # moved aside, not left
    corrupt = Path(str(oi_archive.ARCHIVE_FILE) + '.corrupt')
    assert corrupt.exists() and corrupt.read_bytes() == b'junk'


def test_oi_sync_survives_bad_zip(oi_env, capsys, monkeypatch):
    monkeypatch.setattr(urllib.request, 'urlopen',
                        lambda req, timeout=30: io.BytesIO(b'garbage'))
    start = (dt.datetime.now(dt.timezone.utc).date()
             - dt.timedelta(days=3)).isoformat()
    ok = oi_archive.sync(symbols=['BTC/USD'], start=start, max_files=10)
    out = capsys.readouterr().out
    assert 'parse failed' in out
    assert 'aborting' not in out             # 3 failures < breaker threshold
    assert ok is False


def test_oi_sync_aborts_after_consecutive_failures(oi_env, capsys,
                                                   monkeypatch):
    calls = []

    def _down(req, timeout=30):
        calls.append(req.full_url)
        raise urllib.error.URLError('blackhole')

    monkeypatch.setattr(urllib.request, 'urlopen', _down)
    ok = oi_archive.sync(symbols=['BTC/USD', 'ETH/USD'],
                         start='2026-01-01', max_files=500)
    out = capsys.readouterr().out
    assert 'aborting sync after' in out
    assert len(calls) == oi_archive._MAX_CONSEC_FAILURES  # 10, not 500
    assert ok is False


def test_oi_sync_404s_do_not_trip_breaker(oi_env, capsys, monkeypatch):
    calls = []

    def _404(req, timeout=30):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, 'nf', None, None)

    monkeypatch.setattr(urllib.request, 'urlopen', _404)
    oi_archive.sync(symbols=['BTC/USD'], start='2026-01-01', max_files=15)
    out = capsys.readouterr().out
    assert 'aborting' not in out
    assert len(calls) == 15                  # ran to the cap: 404s are benign


# --- oi_archive.py: live side ---------------------------------------------

def test_concurrent_live_oi_writes_keep_all_symbols(oi_env, monkeypatch):
    """5-worker concurrent RMW of oi_history.json loses no symbols."""
    assert isinstance(oi_archive._live_lock, type(threading.Lock()))
    assert 'with _live_lock' in inspect.getsource(oi_archive.live_oi_features)
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    symbols = [f'SYM{i}/USD' for i in range(8)]
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(oi_archive.live_oi_features, symbols))
    assert all(r == {'OI_Chg_24h': 0.0, 'OI_Z': 0.0} for r in results)
    hist = json.loads((oi_env / 'oi_history.json').read_text())
    assert set(hist) == set(symbols)         # last-writer-wins would drop some


def test_live_history_corrupt_warns_missing_silent(oi_env, caplog):
    with caplog.at_level(logging.DEBUG, logger='oi_archive'):
        assert oi_archive._load_live_history() == {}   # cold start: no noise
    assert 'unreadable' not in caplog.text
    (oi_env / 'oi_history.json').write_text('{corrupt')
    with caplog.at_level(logging.DEBUG, logger='oi_archive'):
        assert oi_archive._load_live_history() == {}
    assert 'unreadable' in caplog.text


def test_live_history_persist_failure_warns_once(oi_env, monkeypatch, caplog):
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE',
                        oi_env / 'no_dir' / 'oi_history.json')
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    with caplog.at_level(logging.DEBUG, logger='oi_archive'):
        out = oi_archive.live_oi_features('BTC/USD')
        oi_archive.live_oi_features('ETH/USD')
    assert out == {'OI_Chg_24h': 0.0, 'OI_Z': 0.0}     # features still served
    assert caplog.text.count('persist failed') == 1


def test_okx_oi_failure_logged_and_negative_cached(oi_env, monkeypatch,
                                                   caplog):
    calls = []

    def _down(req, timeout=10):
        calls.append(1)
        raise urllib.error.URLError('okx down')

    monkeypatch.setattr(urllib.request, 'urlopen', _down)
    with caplog.at_level(logging.DEBUG, logger='oi_archive'):
        assert oi_archive._fetch_okx_oi('BTC/USD') is None
        assert oi_archive._fetch_okx_oi('BTC/USD') is None  # suppressed
    assert len(calls) == 1
    assert 'OKX OI fetch failed' in caplog.text
    # Marker expires -> retried (an outage never becomes permanent)
    oi_archive._fail_cache[('oi', 'BTC/USD')] -= oi_archive._NEG_TTL + 1
    assert oi_archive._fetch_okx_oi('BTC/USD') is None
    assert len(calls) == 2


def test_ls_and_taker_failures_logged_and_negative_cached(oi_env,
                                                          monkeypatch,
                                                          caplog):
    calls = []

    def _down(req, timeout=10):
        calls.append(1)
        raise urllib.error.URLError('okx down')

    monkeypatch.setattr(urllib.request, 'urlopen', _down)
    with caplog.at_level(logging.DEBUG, logger='oi_archive'):
        assert oi_archive.live_ls_features('BTC/USD') is None
        assert oi_archive.live_ls_features('BTC/USD') is None
        assert oi_archive.live_taker_features('BTC/USD') is None
        assert oi_archive.live_taker_features('BTC/USD') is None
    assert len(calls) == 2                   # one per endpoint, retries held
    assert 'long/short fetch failed' in caplog.text
    assert 'taker-volume fetch failed' in caplog.text


# --- dead code / import hygiene -------------------------------------------

def test_dead_code_removed_and_imports_explicit():
    fa_src = (REPO / 'funding_archive.py').read_text()
    oi_src = (REPO / 'oi_archive.py').read_text()
    f_src = (REPO / 'funding.py').read_text()
    # funding_archive: vestigial sys.path mutation gone (with its import)
    assert 'sys.path.insert' not in fa_src
    assert '\nimport sys\n' not in fa_src
    # duplicate in-function pandas re-imports gone
    assert inspect.getsource(funding_archive._parse_zip).count(
        'import pandas') == 1
    assert inspect.getsource(oi_archive._parse_zip).count('import pandas') == 1
    # urllib.error used -> imported explicitly, not via urllib.request's
    # internal side effect
    assert 'import urllib.error' in fa_src
    assert 'import urllib.error' in oi_src
    # funding.py: response handle is context-managed
    assert 'with urllib.request.urlopen' in f_src
