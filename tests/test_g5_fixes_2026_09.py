"""G5 (2026-09) risk / non-model-gate fixes — regression tests.

Covers the G5 hunt findings (Jetson campaign 2026-09-26/27):
  G5-1  volatility._merge_complete_day_rrvs skips the truncated HEAD day
  G5-2  macro_indicators.fetch_vix never accepts/caches a NaN/<=0 close
  G5-3  events_calendar._last_attempt starts at -inf (monotonic-at-boot trap)
  G5-4  short_flow / funding_archive archive + per-symbol memo (bit-identical)
  G5-5  funding.get_funding_rate negative cache (one attempt per _NEG_TTL)
  G5-6  the five JSON loaders survive a non-UTF-8 byte (UnicodeDecodeError)
  G5-7  volatility.forecast_volatility: non-finite -> None (never a 1.5x boost)
  G5-8  volatility.get_garch_stop removed (archived in 08_removed_code.md)
  G5-9  oi_archive.live_oi_features keeps oi_history.json in memory

Mac-safe: numpy/pandas only. Parquet reads are stubbed (pd.read_parquet is
monkeypatched), yfinance/urlopen/finnhub are stubbed — no network, no
pyarrow. The GARCH test importorskips `arch`.
"""

import importlib.util
import json
import math
import os
import socket
import sys
import time
import types
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edgar_events  # noqa: E402
import events_calendar  # noqa: E402
import funding  # noqa: E402
import funding_archive  # noqa: E402
import macro_indicators  # noqa: E402
import oi_archive  # noqa: E402
import short_flow  # noqa: E402
import stock_config  # noqa: E402

BAD_BYTES = b'{"by_symbol": {"AAPL": [\xff\xfe]}}'   # invalid UTF-8


# ---------------------------------------------------------------------------
# G5-2  fetch_vix: NaN yfinance close falls through to FRED, never cached
# ---------------------------------------------------------------------------

@pytest.fixture
def vix_env(monkeypatch):
    macro_indicators._cache.clear()
    fred_calls = []
    state = {'closes': [28.1, 29.4, 30.2, 31.0, float('nan')],
             'fred_body': b"observation_date,VIXCLS\n2026-09-24,30.0\n"}

    class _Ticker:
        def __init__(self, sym):
            pass

        def history(self, period='5d'):
            return pd.DataFrame({'Close': state['closes']})

    fake_yf = types.ModuleType('yfinance')
    fake_yf.Ticker = _Ticker
    monkeypatch.setitem(sys.modules, 'yfinance', fake_yf)

    class _Resp:
        def read(self):
            return state['fred_body']

    def fake_urlopen(req, timeout=None):
        fred_calls.append(getattr(req, 'full_url', req))
        if state['fred_body'] is None:
            raise OSError('FRED down (stub)')
        return _Resp()

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    yield state, fred_calls
    macro_indicators._cache.clear()


def test_fetch_vix_nan_close_falls_through_to_fred(vix_env):
    state, fred_calls = vix_env
    v = macro_indicators.fetch_vix()
    assert v == 30.0 and fred_calls, 'NaN close must fall through to FRED'
    assert macro_indicators._get_cached('vix', 3600) == 30.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0.0, -5.0])
def test_fetch_vix_never_caches_nonfinite(vix_env, bad):
    state, fred_calls = vix_env
    state['closes'] = [20.0, bad]
    state['fred_body'] = None                       # FRED also down
    assert macro_indicators.fetch_vix() is None     # blind -> documented None
    assert 'vix' not in macro_indicators._cache


def test_fetch_vix_fred_nonfinite_row_skipped(vix_env):
    state, _ = vix_env
    state['closes'] = [float('nan')]
    # newest row non-finite -> the next-older finite row is served
    state['fred_body'] = b"observation_date,VIXCLS\n2026-09-23,27.5\n2026-09-24,nan\n"
    assert macro_indicators.fetch_vix() == 27.5


def test_fetch_vix_happy_path_unchanged(vix_env):
    state, fred_calls = vix_env
    state['closes'] = [17.0]
    assert macro_indicators.fetch_vix() == 17.0
    assert not fred_calls


# ---------------------------------------------------------------------------
# G5-3  events_calendar: first refresh right after boot is NOT throttled
# ---------------------------------------------------------------------------

def _fresh_events_module():
    """A brand-new events_calendar instance (module-initial globals)."""
    spec = importlib.util.spec_from_file_location(
        '_ec_fresh_g5', ROOT / 'events_calendar.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize('uptime', [5.0, 120.0, 1799.0])
def test_events_first_refresh_after_boot_fetches(tmp_path, uptime):
    import datetime
    ec = _fresh_events_module()
    assert ec._last_attempt == -float('inf')
    ec._CACHE_FILE = str(tmp_path / 'earnings_calendar.json')
    calls = []
    fresh = {'fetched_at': datetime.datetime.now().isoformat(),
             'by_symbol': {'NVDA': [{'date': datetime.date.today().isoformat(),
                                     'hour': 'amc'}]}}
    ec._fetch_calendar = lambda: (calls.append(1), fresh)[1]
    ec.time = types.SimpleNamespace(monotonic=lambda: uptime)
    ec._mem = {'fetched_at': (datetime.datetime.now()
                              - datetime.timedelta(days=3)).isoformat(),
               'by_symbol': {'AAPL': [{'date': '2026-01-01', 'hour': 'bmo'}]}}
    assert ec.refresh_if_stale() is True
    assert calls == [1], 'first attempt after boot must not be throttled'
    assert ec.earnings_within_days('NVDA')


def test_events_failed_attempt_still_throttled(tmp_path):
    ec = _fresh_events_module()
    ec._CACHE_FILE = str(tmp_path / 'earnings_calendar.json')
    calls = []
    ec._fetch_calendar = lambda: (calls.append(1), None)[1]
    clock = {'t': 100.0}
    ec.time = types.SimpleNamespace(monotonic=lambda: clock['t'])
    ec._mem = {}
    ec.refresh_if_stale()
    clock['t'] = 100.0 + 1799
    ec.refresh_if_stale()
    assert calls == [1]                     # throttle intact after a failure
    clock['t'] = 100.0 + 1801
    ec.refresh_if_stale()
    assert calls == [1, 1]


# ---------------------------------------------------------------------------
# G5-4  short_flow / funding_archive memo (parquet decode stubbed -> Mac-safe)
# ---------------------------------------------------------------------------

def _sf_frame():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range('2025-01-01', periods=160)
    rows = []
    for sym in ('NVDA', 'AAPL', 'MSFT'):
        tv = rng.uniform(1e6, 2e6, len(dates))
        sv = tv * rng.uniform(0.3, 0.6, len(dates))
        rows.append(pd.DataFrame({'date': dates, 'symbol': sym,
                                  'short_vol': sv, 'total_vol': tv}))
    rows.append(pd.DataFrame({'date': dates[:10], 'symbol': 'THIN',
                              'short_vol': 1.0, 'total_vol': 2.0}))
    return pd.concat(rows, ignore_index=True)


def _fa_frame():
    rng = np.random.default_rng(4)
    ts = pd.date_range('2025-01-01', periods=300, freq='8h', tz='UTC')
    rows = [pd.DataFrame({'symbol': s, 'ts': ts,
                          'rate': rng.normal(1e-4, 5e-5, len(ts))})
            for s in ('BTC/USD', 'ETH/USD')]
    dup = rows[0].iloc[[5]].assign(rate=9e-4)      # duplicate ts -> keep='last'
    return pd.concat(rows + [dup], ignore_index=True)


@pytest.fixture(params=['short_flow', 'funding_archive'])
def memo_env(request, tmp_path, monkeypatch):
    mod = short_flow if request.param == 'short_flow' else funding_archive
    frames = {'v': _sf_frame() if mod is short_flow else _fa_frame()}
    reads = []
    path = tmp_path / f'{request.param}.parquet'
    path.write_bytes(b'stub-v1')                     # real file -> real stamp
    monkeypatch.setattr(mod, 'ARCHIVE_FILE', path)
    monkeypatch.setattr(mod, '_memo', {'key': None, 'df': None, 'by_sym': {}})

    def fake_read(p, *a, **k):
        reads.append(str(p))
        return frames['v'].copy()

    monkeypatch.setattr(pd, 'read_parquet', fake_read)
    return mod, frames, reads, path


def _series_fn(mod):
    return short_flow.svr_series if mod is short_flow else funding_archive.get_funding_series


def _syms(mod):
    return (['NVDA', 'aapl', 'MSFT', 'THIN', 'ZZZ'] if mod is short_flow
            else ['BTC/USD', 'ETH/USD', 'SOL/USD'])


def _uncached(mod, frame, sym):
    return (short_flow._svr_series_from(frame, sym) if mod is short_flow
            else funding_archive._funding_series_from(frame, sym))


def _assert_same(a, b):
    if a is None or b is None:
        assert a is None and b is None
        return
    for x, y in zip(a if isinstance(a, tuple) else (a,),
                    b if isinstance(b, tuple) else (b,)):
        pd.testing.assert_series_equal(x, y, check_exact=True)


def test_memo_reads_archive_once_per_stamp(memo_env):
    mod, frames, reads, path = memo_env
    fn = _series_fn(mod)
    for _ in range(3):
        for s in _syms(mod):
            fn(s)
    assert len(reads) == 1, 'one parquet decode per archive stamp'
    assert mod.load_archive() is mod.load_archive()


def test_memo_outputs_bit_identical_to_uncached(memo_env):
    mod, frames, reads, path = memo_env
    fn = _series_fn(mod)
    for s in _syms(mod):
        first = fn(s)
        second = fn(s)                     # memo hit
        _assert_same(first, _uncached(mod, frames['v'], s))
        _assert_same(second, first)
    # the frame served is the parsed frame itself
    pd.testing.assert_frame_equal(mod.load_archive(), frames['v'])


def test_memo_invalidates_on_rewrite(memo_env):
    mod, frames, reads, path = memo_env
    fn = _series_fn(mod)
    sym = _syms(mod)[0]
    before = fn(sym)
    frames['v'] = frames['v'].assign(
        **({'short_vol': frames['v']['short_vol'] * 0.5} if mod is short_flow
           else {'rate': frames['v']['rate'] * 2.0}))
    tmp = str(path) + '.tmp'
    Path(tmp).write_bytes(b'stub-v2-longer')          # sync(): tmp + os.replace
    os.replace(tmp, path)
    after = fn(sym)
    assert len(reads) == 2
    _assert_same(after, _uncached(mod, frames['v'], sym))
    with pytest.raises(AssertionError):
        _assert_same(after, before)


def test_memo_repoint_archive_file_invalidates(memo_env, tmp_path, monkeypatch):
    mod, frames, reads, path = memo_env
    fn = _series_fn(mod)
    fn(_syms(mod)[0])
    other = tmp_path / 'other.parquet'
    other.write_bytes(b'stub-v1')
    monkeypatch.setattr(mod, 'ARCHIVE_FILE', other)
    fn(_syms(mod)[0])
    assert len(reads) == 2


def test_memo_missing_file_is_empty_and_uncached(memo_env):
    mod, frames, reads, path = memo_env
    path.unlink()
    assert mod.load_archive().empty
    assert _series_fn(mod)(_syms(mod)[0]) is None
    assert reads == []


def test_memo_failed_read_not_memoized(memo_env, monkeypatch):
    mod, frames, reads, path = memo_env
    n = {'c': 0}

    def boom(p, *a, **k):
        n['c'] += 1
        raise OSError('corrupt (stub)')

    monkeypatch.setattr(pd, 'read_parquet', boom)
    assert mod.load_archive().empty
    assert mod.load_archive().empty
    assert n['c'] == 2                       # retries every call, as before
    assert mod._memo['key'] is None


def test_live_svr_features_bit_identical(memo_env):
    mod, frames, reads, path = memo_env
    if mod is not short_flow:
        pytest.skip('short_flow only')
    for s in ('NVDA', 'AAPL', 'MSFT'):
        svr, z = short_flow._svr_series_from(frames['v'], s)
        expect = {'SVR_21': float(svr.iloc[-1]), 'SVR_Z': float(z.iloc[-1])}
        assert short_flow.live_svr_features(s) == expect
        assert short_flow.live_svr_features(s) == expect


def test_funding_features_for_index_bit_identical(memo_env):
    mod, frames, reads, path = memo_env
    if mod is not funding_archive:
        pytest.skip('funding_archive only')
    idx = pd.date_range('2025-02-01', periods=400, freq='h', tz='UTC')
    a = funding_archive.funding_features_for_index('BTC/USD', idx)
    b = funding_archive.funding_features_for_index('BTC/USD', idx)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert len(reads) == 1


def test_memo_on_real_archives_matches_head_math():
    """Real Jetson archives (skipped where absent/unreadable): memoized ==
    a fresh uncached recompute, bit-identical, for every symbol."""
    pytest.importorskip('pyarrow')
    for mod, syms in ((short_flow, None), (funding_archive,
                      ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD',
                       'DOGE/USD', 'LINK/USD'])):
        if not Path(mod.ARCHIVE_FILE).exists():
            continue
        frame = pd.read_parquet(mod.ARCHIVE_FILE)
        if syms is None:
            syms = sorted(frame['symbol'].unique())[:46]
        fn = _series_fn(mod)
        for s in syms:
            _assert_same(fn(s), _uncached(mod, frame, s))
            _assert_same(fn(s), _uncached(mod, frame, s))


# ---------------------------------------------------------------------------
# G5-5  funding.get_funding_rate negative cache
# ---------------------------------------------------------------------------

@pytest.fixture
def funding_env(tmp_path, monkeypatch):
    monkeypatch.setattr(funding, '_HISTORY_FILE', str(tmp_path / 'fh.json'))
    monkeypatch.setattr(funding, '_cache', {})
    monkeypatch.setattr(funding, '_history', None)
    clock = {'t': 10_000.0}
    monkeypatch.setattr(funding.time, 'monotonic', lambda: clock['t'])
    return clock


class _RateResp:
    def __init__(self, rate):
        self._b = json.dumps({'data': [{'fundingRate': str(rate)}]}).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_neg_ttl_matches_oi_archive():
    assert funding._NEG_TTL == oi_archive._NEG_TTL == 300


def test_funding_outage_one_attempt_per_ttl(funding_env, monkeypatch):
    clock = funding_env
    calls = []

    def hang(req, timeout=None):
        calls.append(timeout)
        raise socket.timeout('timed out')

    monkeypatch.setattr(funding.urllib.request, 'urlopen', hang)
    syms = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD', 'LINK/USD']
    for cycle in range(9):                   # 9 x 30 s cycles < 300 s
        clock['t'] = 10_000.0 + 30 * cycle
        for s in syms:
            assert funding.get_funding_rate(s) is None
            assert funding.live_funding_features(s) is None
            assert funding.funding_tilt(s) == 1.0
    assert len(calls) == len(syms)           # one attempt per symbol per TTL
    clock['t'] = 10_000.0 + funding._NEG_TTL + 1
    funding.get_funding_rate('BTC/USD')
    assert len(calls) == len(syms) + 1       # retried after the TTL


def test_funding_recovers_after_ttl_and_success_path_unchanged(funding_env,
                                                               monkeypatch):
    clock = funding_env
    state = {'down': True}

    def urlopen(req, timeout=None):
        if state['down']:
            raise OSError('down')
        return _RateResp(0.0002)

    monkeypatch.setattr(funding.urllib.request, 'urlopen', urlopen)
    assert funding.get_funding_rate('BTC/USD') is None
    state['down'] = False
    clock['t'] += funding._NEG_TTL - 1
    assert funding.get_funding_rate('BTC/USD') is None     # still suppressed
    clock['t'] += 2
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(0.0002)
    # positive entry honours the normal 15-min TTL
    state['down'] = True
    clock['t'] += funding._CACHE_TTL - 1
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(0.0002)


def test_funding_negative_entry_lives_in_cache_dict(funding_env, monkeypatch):
    """Resetting funding._cache (what the existing fixtures do) also resets
    the negative entries — no cross-test leak."""
    monkeypatch.setattr(funding.urllib.request, 'urlopen',
                        lambda req, timeout=None: (_ for _ in ()).throw(OSError('x')))
    funding.get_funding_rate('BTC/USD')
    assert funding._cache['BTC/USD'][1] is None
    funding._cache.clear()
    monkeypatch.setattr(funding.urllib.request, 'urlopen',
                        lambda req, timeout=None: _RateResp(0.0003))
    assert funding.get_funding_rate('BTC/USD') == pytest.approx(0.0003)


# ---------------------------------------------------------------------------
# G5-6  the five JSON loaders survive a non-UTF-8 byte
# ---------------------------------------------------------------------------

@pytest.fixture
def bad_json(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_bytes(BAD_BYTES)
    with pytest.raises(UnicodeDecodeError):
        with open(p) as f:
            json.load(f)
    return p


def test_unicode_decode_error_is_value_error_not_json_error():
    assert issubclass(UnicodeDecodeError, ValueError)
    assert not issubclass(UnicodeDecodeError, json.JSONDecodeError)


def test_events_calendar_non_utf8(bad_json, monkeypatch):
    monkeypatch.setattr(events_calendar, '_CACHE_FILE', str(bad_json))
    monkeypatch.setattr(events_calendar, '_mem', None)
    monkeypatch.setattr(events_calendar, '_fetch_calendar', lambda: None)
    monkeypatch.setattr(events_calendar, '_last_attempt', -float('inf'))
    assert events_calendar.calendar_available() is False   # sleeve fails closed
    assert events_calendar.earnings_within_days('AAPL') is False
    events_calendar.blocks_overnight_hold('AAPL')           # must not raise


def test_edgar_events_non_utf8(bad_json, monkeypatch):
    monkeypatch.setattr(edgar_events, '_CACHE_FILE', bad_json)
    monkeypatch.setattr(edgar_events, '_cache_memo', None)
    assert edgar_events._load_cache() == {}


def test_funding_history_non_utf8(bad_json, monkeypatch):
    monkeypatch.setattr(funding, '_HISTORY_FILE', str(bad_json))
    monkeypatch.setattr(funding, '_history', None)
    assert funding._load_history() == {}


def test_oi_live_history_non_utf8(bad_json, monkeypatch, caplog):
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE', bad_json)
    monkeypatch.setattr(oi_archive, '_hist_memo', None)
    monkeypatch.setattr(oi_archive, '_hist_read_warned', False)
    assert oi_archive._load_live_history() == {}
    # self-heal: the next successful persist rewrites the file
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 5000.0)
    assert oi_archive.live_oi_features('BTC/USD') == {'OI_Chg_24h': 0.0,
                                                      'OI_Z': 0.0}
    assert list(json.loads(bad_json.read_text())) == ['BTC/USD']


def test_stock_config_non_utf8_falls_back(bad_json, monkeypatch):
    monkeypatch.setattr(stock_config, '_FILE', bad_json)
    assert stock_config.load_stock_universe() == list(stock_config._DEFAULTS)


# ---------------------------------------------------------------------------
# G5-9  oi_archive.live_oi_features: oi_history.json kept in memory
# ---------------------------------------------------------------------------

@pytest.fixture
def oi_live_env(tmp_path, monkeypatch):
    path = tmp_path / 'oi_history.json'
    monkeypatch.setattr(oi_archive, '_LIVE_HISTORY_FILE', path)
    monkeypatch.setattr(oi_archive, '_hist_memo', None)
    loads = []
    real_load = json.load

    def counting_load(f, *a, **k):
        loads.append(getattr(f, 'name', None))
        return real_load(f, *a, **k)

    monkeypatch.setattr(oi_archive.json, 'load', counting_load)
    return path, loads


def _seed_history(path, now, n=200):
    rng = np.random.default_rng(9)
    hist = {s: [[now - 3600 * (n - i), float(1e6 + rng.normal(0, 1e4))]
                for i in range(n)]
            for s in ('BTC/USD', 'ETH/USD')}
    path.write_text(json.dumps(hist))
    return hist


def _reference_features(hist_all, symbol, oi, now):
    """The pre-memo computation, re-derived from a fresh json read."""
    import statistics
    usable = [(ts, v) for ts, v in hist_all.get(symbol, [])
              if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
    chg = 0.0
    cands = [(abs(ts - (now - 86400)), v) for ts, v in usable
             if abs(ts - (now - 86400)) <= 3 * 3600]
    if cands:
        _, ref = min(cands)
        chg = (oi - ref) / ref * 100
    z = 0.0
    vals = [v for _, v in usable]
    if len(vals) >= 168:
        sd = statistics.pstdev(vals)
        if sd > 1e-12:
            z = (oi - statistics.fmean(vals)) / sd
    return {'OI_Chg_24h': chg, 'OI_Z': z}


def test_oi_history_parsed_once_across_cycles(oi_live_env, monkeypatch):
    path, loads = oi_live_env
    now = time.time()
    _seed_history(path, now)
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 1.01e6)
    t = {'now': now}
    monkeypatch.setattr(oi_archive.time, 'time', lambda: t['now'])
    for cycle in range(20):                  # 30 s cycles, no hourly append
        t['now'] = now + 30 * cycle
        for s in ('BTC/USD', 'ETH/USD'):
            ref = _reference_features(json.loads(path.read_text()), s,
                                      1.01e6, t['now'])
            assert oi_archive.live_oi_features(s) == ref
    assert len(loads) == 1


def test_oi_history_memo_equals_disk_after_append(oi_live_env, monkeypatch):
    path, loads = oi_live_env
    now = time.time()
    _seed_history(path, now)
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 1.02e6)
    t = {'now': now + 4000}                  # > _LIVE_THIN_SEC -> appends
    monkeypatch.setattr(oi_archive.time, 'time', lambda: t['now'])
    oi_archive.live_oi_features('BTC/USD')
    on_disk = json.loads(path.read_text())
    assert oi_archive._hist_memo[1] == on_disk
    assert oi_archive._load_live_history() == on_disk
    n = len(loads)
    oi_archive.live_oi_features('ETH/USD')   # append -> rewrite -> memo refresh
    assert len(loads) == n                   # served from the memo, not re-read
    assert oi_archive._load_live_history() == json.loads(path.read_text())


def test_oi_history_foreign_rewrite_is_picked_up(oi_live_env, monkeypatch):
    path, loads = oi_live_env
    now = time.time()
    _seed_history(path, now)
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 1.0e6)
    monkeypatch.setattr(oi_archive.time, 'time', lambda: now)
    oi_archive.live_oi_features('BTC/USD')
    path.write_text(json.dumps({'BTC/USD': []}))      # another writer
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10**9))
    oi_archive.live_oi_features('BTC/USD')
    assert list(json.loads(path.read_text())['BTC/USD'][-1]) == [now, 1.0e6]


def test_oi_history_failed_persist_leaves_no_phantom_sample(oi_live_env,
                                                            monkeypatch):
    path, loads = oi_live_env
    now = time.time()
    _seed_history(path, now)
    disk_before = path.read_text()
    monkeypatch.setattr(oi_archive, '_fetch_okx_oi', lambda s: 1.0e6)
    monkeypatch.setattr(oi_archive.time, 'time', lambda: now + 4000)
    real_replace = os.replace

    def fail_replace(a, b):
        raise OSError('disk full (stub)')

    monkeypatch.setattr(oi_archive.os, 'replace', fail_replace)
    oi_archive.live_oi_features('BTC/USD')
    assert oi_archive._hist_memo is None
    monkeypatch.setattr(oi_archive.os, 'replace', real_replace)
    assert path.read_text() == disk_before
    assert oi_archive._load_live_history() == json.loads(disk_before)


# ---------------------------------------------------------------------------
# volatility (G5-1, G5-7, G5-8, _rv_save per-writer tmp)
# ---------------------------------------------------------------------------

import collections  # noqa: E402
import threading  # noqa: E402

import volatility  # noqa: E402

_PER_BAR = np.log(101 / 100) ** 2 / (4 * np.log(2))   # Parkinson, H/L = 1.01


def _flat_range_bars(start='2026-06-01', days=40):
    idx = pd.date_range(start, periods=24 * days, freq='h', tz='UTC')
    return pd.DataFrame({'High': 101.0, 'Low': 100.0}, index=idx)


@pytest.mark.parametrize('min_bars', [None, 20])
def test_rrv_rolling_frame_stores_true_complete_day(min_bars):
    """Live cadence: fetch_bars_alpaca(limit=250) == .tail(250), one new
    hourly bar at a time. Every settled day must store its TRUE 24-bar RRV
    (pre-fix: 0.042 of it for min_bars=None, 0.833 for the HAR store)."""
    bars = _flat_range_bars()
    hist = {}
    for t in range(250, len(bars) + 1):
        volatility._merge_complete_day_rrvs(hist, bars.iloc[:t].tail(250),
                                            365, min_bars=min_bars)
    settled = sorted(hist)[:-11]                 # rolled fully out of the frame
    assert len(settled) >= 25
    ratios = np.array([hist[d] / (24 * _PER_BAR) for d in settled])
    np.testing.assert_allclose(ratios, 1.0, rtol=1e-12)


def test_rrv_truncated_head_day_skipped_midnight_head_kept():
    bars = _flat_range_bars(days=5)
    # frame starting 19:00 -> its first calendar day is a 5-bar partial
    h = {}
    volatility._merge_complete_day_rrvs(h, bars.iloc[19:], 365)
    assert '2026-06-01' not in h
    assert sorted(h) == ['2026-06-02', '2026-06-03', '2026-06-04']
    # frame starting exactly at 00:00 -> the head day is complete and kept
    h = {}
    volatility._merge_complete_day_rrvs(h, bars, 365)
    assert sorted(h) == ['2026-06-01', '2026-06-02', '2026-06-03', '2026-06-04']
    assert h['2026-06-01'] == pytest.approx(24 * _PER_BAR, rel=1e-12)


def test_rrv_state_stationary_market_not_pinned_in_crisis(tmp_path, monkeypatch):
    """110 days of stationary iid hourly ranges on the live-shaped rolling
    frame: the B06 state must be mostly non-crisis (pre-fix: 240/240 'crisis'
    once the 90-day history filled, get_crypto_rv_mult -> 0.3)."""
    monkeypatch.setattr(volatility, '_CRYPTO_RV_FILE', str(tmp_path / 'rv.json'))
    volatility._reset_crypto_rv_state()
    rng = np.random.default_rng(7)
    n = 24 * 110
    idx = pd.date_range('2026-05-01', periods=n, freq='h', tz='UTC')
    lr = np.exp(rng.normal(np.log(0.006), 0.35, n))
    bars = pd.DataFrame({'High': 100 * np.exp(lr), 'Low': 100.0}, index=idx)
    states = collections.Counter()
    pct = []
    try:
        for t in range(250, n + 1):
            volatility.update_crypto_rv_state('BTC/USD', bars.iloc[:t].tail(250))
            if t > n - 24 * 10:
                states[volatility._crypto_rv['state']] += 1
                pct.append(volatility._crypto_rv['pctile'])
        mult = volatility.get_crypto_rv_mult()
    finally:
        volatility._reset_crypto_rv_state()
    assert states['crisis'] < 0.25 * sum(states.values()), states
    assert states['normal'] > 0.3 * sum(states.values()), states
    assert 25 < np.nanmedian(pct) < 75
    assert mult[0] != 0.3 or mult[1] != 'crisis'


class _NaNVarResult:
    """Minimal arch-result stand-in whose forecast variance is NaN."""

    def __init__(self, var):
        self._var = var

    def forecast(self, horizon=1):
        return types.SimpleNamespace(
            variance=pd.DataFrame([[self._var]]))


@pytest.mark.parametrize('var', [float('nan'), float('inf'), 0.0, -1.0])
def test_forecast_volatility_nonfinite_is_none(var):
    assert volatility.forecast_volatility(_NaNVarResult(var)) is None


def test_forecast_volatility_finite_unchanged():
    assert volatility.forecast_volatility(_NaNVarResult(4.0)) == pytest.approx(0.02)


@pytest.mark.parametrize('sigma', [float('nan'), float('inf'), -float('inf'), 0.0])
def test_vol_adjusted_size_nonfinite_sigma_is_neutral(sigma):
    assert volatility.compute_vol_adjusted_size(1000.0, sigma, 'stock') == 1000.0
    assert volatility.compute_vol_adjusted_size(1000.0, sigma, 'crypto') == 1000.0


def test_flat_series_garch_never_boosts():
    """Real arch on a zero-variance series (frozen/halted feed): pre-fix the
    NaN sigma became the max 1.5x boost."""
    pytest.importorskip('arch')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = volatility.fit_garch(np.zeros(200))
        sigma = volatility.forecast_volatility(res)
    assert sigma is None or (np.isfinite(sigma) and sigma > 0)
    if sigma is not None:
        assert volatility.compute_vol_adjusted_size(1.0, sigma, 'stock') <= 1.5


def test_get_garch_stop_removed_and_archived():
    assert not hasattr(volatility, 'get_garch_stop')
    arch_md = (ROOT / 'research' / 'campaign_2026-08' / '08_removed_code.md').read_text()
    assert 'def get_garch_stop(entry_price: float, sigma: float' in arch_md
    assert 'stop_dist = max(floor_pct, min(ceil_pct, sigma * multiplier))' in arch_md
    assert 'def test_get_garch_stop(self):' in arch_md
    assert 'def test_get_garch_stop_floor(self):' in arch_md
    # no production module references the name any more
    for p in list(ROOT.glob('*.py')) + list((ROOT / 'scripts').glob('*.py')):
        assert 'get_garch_stop(' not in p.read_text(errors='replace'), p


@pytest.fixture
def rv_file(tmp_path, monkeypatch):
    path = str(tmp_path / 'crypto_rv_history.json')
    monkeypatch.setattr(volatility, '_CRYPTO_RV_FILE', path)
    monkeypatch.setitem(volatility._crypto_rv, 'history',
                        {f'2026-01-{d:02d}': 1e-4 * d for d in range(1, 29)})
    monkeypatch.setitem(volatility._crypto_rv, 'state', 'high')
    monkeypatch.setitem(volatility._crypto_rv, 'exit_count', 1)
    monkeypatch.setitem(volatility._crypto_rv, 'last_bar_ts', '2026-01-28T23:00:00+00:00')
    return tmp_path, path


def _rv_payload():
    return {'rrv': volatility._crypto_rv['history'],
            'state': volatility._crypto_rv['state'],
            'exit_count': volatility._crypto_rv['exit_count'],
            'last_bar_ts': volatility._crypto_rv['last_bar_ts']}


def test_rv_save_roundtrip_no_tmp_left(rv_file):
    tmp_dir, path = rv_file
    volatility._rv_save()
    with open(path) as f:
        assert json.load(f) == _rv_payload()
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []


def test_rv_save_tmp_is_per_writer(rv_file, monkeypatch):
    tmp_dir, path = rv_file
    seen = []
    real = os.replace

    def spy(src, dst):
        seen.append(os.path.basename(src))
        return real(src, dst)

    monkeypatch.setattr(volatility.os, 'replace', spy)
    volatility._rv_save()
    assert seen == [f'crypto_rv_history.json.{os.getpid()}.'
                    f'{threading.get_ident()}.tmp']


def test_rv_save_failure_removes_tmp(rv_file, monkeypatch):
    tmp_dir, path = rv_file

    def boom(src, dst):
        raise OSError('disk full')

    monkeypatch.setattr(volatility.os, 'replace', boom)
    volatility._rv_save()                     # never raises
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []
    assert not os.path.exists(path)


def test_rv_save_interleaved_writers_no_tear(rv_file, monkeypatch):
    """Writer A pauses mid-dump while writer B saves; with the old shared
    fixed '.tmp' B truncated A's file -> torn JSON in the live path."""
    tmp_dir, path = rv_file
    real_dump = json.dump
    a_half = threading.Event()
    b_done = threading.Event()

    def shim(obj, f, *args, **kw):
        name = threading.current_thread().name
        if name == 'writer-A':
            text = json.dumps(obj)
            f.write(text[:len(text) // 2])
            f.flush()
            a_half.set()
            assert b_done.wait(10)
            f.write(text[len(text) // 2:])
        elif name == 'writer-B':
            real_dump({'rrv': {}, 'state': 'B'}, f)
        else:
            real_dump(obj, f, *args, **kw)

    monkeypatch.setattr(json, 'dump', shim)

    def run_b():
        assert a_half.wait(10)
        volatility._rv_save()
        b_done.set()

    ta = threading.Thread(target=volatility._rv_save, name='writer-A')
    tb = threading.Thread(target=run_b, name='writer-B')
    ta.start(); tb.start()
    ta.join(15); tb.join(15)
    assert not ta.is_alive() and not tb.is_alive()
    with open(path) as f:
        assert json.load(f) == _rv_payload()      # A replaced last, intact
    assert sorted(p.name for p in tmp_dir.glob('*.tmp')) == []
