"""scripts/crypto_quote_staleness_census.py — replay path, None-rate /
threshold arithmetic, reason attribution, and parity with the REAL
order_utils.get_crypto_quote verdict (stub API objects; no network, no SDK).
Mac-safe: numpy/pandas + order_utils (stdlib + log_config) only."""

import datetime
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))

import order_utils  # noqa: E402
import crypto_quote_staleness_census as cq  # noqa: E402

UTC = datetime.timezone.utc


def _now():
    return datetime.datetime.now(UTC)


class _Api:
    """Stub SDK: returns {sym: quote} (or raises), like get_latest_crypto_quotes."""

    def __init__(self, quote=None, exc=None, drop=False):
        self.quote, self.exc, self.drop = quote, exc, drop

    def get_latest_crypto_quotes(self, symbols):
        if self.exc is not None:
            raise self.exc
        return {} if self.drop else {symbols[0]: self.quote}


def _q(bp=100.0, ap=100.1, age=None, t='auto'):
    if t == 'auto':
        t = None if age is None else _now() - datetime.timedelta(seconds=age)
    return SimpleNamespace(bp=bp, ap=ap, t=t)


# ------------------------------------------------ live-rule constant pinned

@pytest.mark.parametrize('age,is_none', [(5, False), (170, False),
                                         (190, True), (212, True),
                                         (900, True)])
def test_live_constant_matches_real_get_crypto_quote(age, is_none):
    live = order_utils.get_crypto_quote(_Api(_q(age=age)), 'DOGE/USD')
    assert (live is None) is is_none
    assert (age > cq.LIVE_MAX_AGE_S) is is_none


# ------------------------------------- reason attribution == live verdict

_NY_TS = pd.Timestamp(_now() - datetime.timedelta(seconds=212)).tz_convert(
    'America/New_York')

SCENARIOS = [
    ('fresh', _Api(_q(age=10)), 'ok'),
    ('stale_212s', _Api(_q(age=212)), 'stale'),
    ('stale_pandas_ny_tz', _Api(_q(t=_NY_TS)), 'stale'),
    ('stale_naive_utc', _Api(_q(t=(_now() - datetime.timedelta(seconds=400))
                                .replace(tzinfo=None))), 'stale'),
    ('no_timestamp', _Api(_q(t=None)), 'stale'),  # fail-closed (ENGINE r3 W10)
    ('unparseable_ts', _Api(_q(t='not-a-time')), 'stale'),  # fail-closed (ENGINE r3 W10)
    ('nan_bid', _Api(_q(bp=float('nan'), age=5)), 'degenerate'),
    ('zero_bid', _Api(_q(bp=0.0, age=5)), 'degenerate'),
    ('crossed_is_not_none', _Api(_q(bp=100.2, ap=100.0, age=5)), 'ok'),
    ('none_price', _Api(_q(bp=None, age=5)), 'error'),
    ('symbol_absent', _Api(drop=True), 'missing'),
    ('value_none', _Api(quote=None), 'missing'),
    ('sdk_raises', _Api(exc=RuntimeError('HTTP 429')), 'error'),
]


@pytest.mark.parametrize('name,api,reason', SCENARIOS,
                         ids=[s[0] for s in SCENARIOS])
def test_sample_once_reason_and_live_parity(name, api, reason):
    rec = cq.sample_once(api, 'DOGE/USD', 0, order_utils.get_crypto_quote)
    assert rec['reason'] == reason
    # The census's live column is the real function's verdict ...
    direct = order_utils.get_crypto_quote(api, 'DOGE/USD')
    assert rec['live_none'] is (direct is None)
    # ... and the classifier attributes every None (and only Nones).
    assert rec['live_none'] is (reason != 'ok')
    json.dumps(rec)   # record must be JSON-serialisable for --json/--replay


def test_crossed_flag_and_negative_spread():
    rec = cq.sample_once(_Api(_q(bp=100.2, ap=100.0, age=5)), 'X/USD', 0,
                         order_utils.get_crypto_quote)
    assert rec['crossed'] is True and rec['spread_bps'] < 0


def test_to_utc_parses_iso_nanoseconds_and_offsets():
    a = cq._to_utc('2026-09-27T04:42:00.123456789Z')
    b = cq._to_utc('2026-09-26T23:42:00.123456-05:00')
    assert a == b == datetime.datetime(2026, 9, 27, 4, 42, 0, 123456, tzinfo=UTC)
    assert cq._to_utc(None) is None and cq._to_utc('garbage') is None


# --------------------------------------------------- pure table arithmetic

def _rec(sym, poll, reason, age=None, spread=10.0, crossed=False):
    return {'symbol': sym, 'poll': poll, 'reason': reason, 'age_s': age,
            'spread_bps': spread, 'crossed': crossed,
            'live_none': reason != 'ok'}


def _fixture_records():
    # DOGE: 10 polls; ages 10,20,...; polls 4-6 stale at 200/250/650 s;
    # poll 8 an error (no age). XRP: all fresh; one crossed.
    doge_ages = [10, 20, 30, 40, 200, 250, 650, 60, None, 90]
    recs = []
    for i, a in enumerate(doge_ages):
        if a is None:
            recs.append(_rec('DOGE/USD', i, 'error', None, None))
        else:
            recs.append(_rec('DOGE/USD', i,
                             'stale' if a > 180 else 'ok', a, 20.0 + i))
    for i in range(10):
        recs.append(_rec('XRP/USD', i, 'ok', 5.0 + i, 8.0,
                         crossed=(i == 3)))
    return recs


def test_none_under_is_strict_and_non_age_reasons_always_none():
    assert cq.none_under(_rec('A', 0, 'ok', 180.0), 180.0) is False
    assert cq.none_under(_rec('A', 0, 'stale', 180.5), 180.0) is True
    assert cq.none_under(_rec('A', 0, 'stale', 250.0), 300.0) is False
    for r in ('error', 'missing', 'degenerate'):
        assert cq.none_under(_rec('A', 0, r, None), 1e9) is True
    assert cq.none_under(_rec('A', 0, 'ok', None), 0.0) is False


def test_build_table_rates_streak_and_distribution():
    t = cq.build_table(_fixture_records(), (180, 300, 600, 900))
    d = t['DOGE/USD']
    assert d['n'] == 10
    assert d['reasons'] == {'ok': 6, 'error': 1, 'missing': 0,
                            'degenerate': 0, 'stale': 3}
    assert d['none_rate_live'] == pytest.approx(0.4)
    # 180: 3 stale + 1 error; 300: 650 + error; 600: same; 900: error only
    assert d['none_rate_at'] == pytest.approx(
        {'180': 0.4, '300': 0.2, '600': 0.2, '900': 0.1})
    assert d['max_none_streak'] == 3
    assert d['age_max_s'] == 650
    assert d['age_p50_s'] == pytest.approx(60.0)   # median of 9 ages
    assert d['live_disagreements'] == 0
    x = t['XRP/USD']
    assert x['none_rate_live'] == 0.0 and x['crossed'] == 1
    assert x['spread_p50_bps'] == pytest.approx(8.0)
    assert all(v == 0.0 for v in x['none_rate_at'].values())


def test_live_column_at_live_threshold_equals_live_rate():
    t = cq.build_table(_fixture_records(), (cq.LIVE_MAX_AGE_S,))
    for row in t.values():
        assert row['none_rate_at']['180'] == row['none_rate_live']


def test_disagreement_is_counted():
    recs = _fixture_records()
    recs[0]['live_none'] = True     # classifier says ok, live said None
    assert cq.build_table(recs)['DOGE/USD']['live_disagreements'] == 1


# ------------------------------------------------------------ replay path

def test_replay_recomputes_table_without_network(tmp_path, monkeypatch,
                                                 capsys):
    def _boom(*a, **k):
        raise AssertionError('replay must not sample')
    monkeypatch.setattr(cq, 'sample', _boom)
    saved = tmp_path / 'census.json'
    saved.write_text(json.dumps({
        'meta': {'symbols': ['DOGE/USD', 'XRP/USD'], 'interval_s': 30.0},
        'records': _fixture_records()}))
    out = tmp_path / 'replayed.json'
    rc = cq.main(['--replay', str(saved), '--thresholds', '180,300,600,900',
                  '--json', str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert 'DOGE/USD' in text and '40.0%' in text
    got = json.loads(out.read_text())
    assert got['table']['DOGE/USD']['none_rate_at']['300'] == pytest.approx(0.2)
    assert got['thresholds_s'] == [180.0, 300.0, 600.0, 900.0]
    assert len(got['records']) == 20


def test_live_records_round_trip_through_replay(tmp_path, capsys):
    """Records produced by sample_once (stubbed SDK) replay to the same
    table as the in-memory build."""
    recs = []
    for poll, age in enumerate([10, 212, 225, 30]):
        recs.append(cq.sample_once(_Api(_q(age=age)), 'DOGE/USD', poll,
                                   order_utils.get_crypto_quote))
    f = tmp_path / 's.json'
    f.write_text(json.dumps({'meta': {'symbols': ['DOGE/USD'],
                                      'interval_s': 30.0}, 'records': recs}))
    cq.main(['--replay', str(f)])
    mem = cq.build_table(recs)['DOGE/USD']
    assert mem['none_rate_live'] == pytest.approx(0.5)
    assert mem['max_none_streak'] == 2
    assert mem['none_rate_at']['300'] == 0.0
    assert not math.isnan(mem['age_p90_s'])
    assert 'DOGE/USD' in capsys.readouterr().out
