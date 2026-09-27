"""ENGINE r4 — quote-timestamp parsing must not emit pandas'
"Discarding nonzero nanoseconds in conversion" UserWarning (order_utils
get_quote, ~:190). Alpaca crypto quotes carry nanosecond timestamps, so the
default `Timestamp.to_pydatetime()` printed one warning per symbol per 30 s
cycle in the live bots (CEO observation 2026-09-27 03:09). The fix passes
`warn=False`, which truncates to microseconds exactly as before — the age
and the staleness verdict are byte-identical. Mac-safe (pandas only)."""
import datetime as _dt
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

import order_utils


class _QApi:
    def __init__(self, q):
        self.q = q

    def get_latest_crypto_quotes(self, symbols):
        return {symbols[0]: self.q}

    def get_latest_quote(self, symbol):
        return self.q


def _ns_quote(age_s, ns=789):
    t = pd.Timestamp.now(tz='UTC') - pd.Timedelta(seconds=age_s)
    t = t.floor('us') + pd.Timedelta(nanoseconds=ns)
    assert t.nanosecond == ns
    return SimpleNamespace(bp=100.0, ap=100.1, t=t)


@pytest.mark.parametrize('age_s', [5, 60, 179])
def test_fresh_nanosecond_timestamp_emits_no_warning_and_is_accepted(age_s):
    q = _ns_quote(age_s)
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning -> test failure
        out = order_utils.get_crypto_quote(_QApi(q), 'BTC/USD')
    assert out is not None and out['bid'] == 100.0 and out['ask'] == 100.1


def test_stale_nanosecond_timestamp_still_rejected_without_warning():
    q = _ns_quote(400)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert order_utils.get_crypto_quote(_QApi(q), 'BTC/USD') is None


def test_verdict_identical_to_microsecond_timestamp_at_the_boundary():
    """Truncating <1 µs cannot move a quote across the 180 s line: a
    ns-carrying timestamp and its µs-truncated twin get the same verdict on
    both sides of the boundary."""
    for age_s, expect_none in ((179.0, False), (181.0, True)):
        q_ns = _ns_quote(age_s)
        q_us = SimpleNamespace(bp=100.0, ap=100.1, t=q_ns.t.floor('us'))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            a = order_utils.get_crypto_quote(_QApi(q_ns), 'BTC/USD')
            b = order_utils.get_crypto_quote(_QApi(q_us), 'BTC/USD')
        assert (a is None) == (b is None) == expect_none


def test_plain_datetime_path_unchanged():
    t = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=10)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = order_utils.get_crypto_quote(_QApi(SimpleNamespace(bp=1.0, ap=1.01, t=t)), 'X/USD')
    assert out is not None


def test_source_pin_warn_false():
    import inspect
    src = inspect.getsource(order_utils)   # the parse lives in the shared get_quote body
    assert 'to_pydatetime(warn=False)' in src
