"""2026-09 Jetson campaign — execution fixes (hunt G2-1, G2-3, G1 F3).

Mac-safe: stub broker objects only, no network, no orders, no LLM calls.

  G2-1  order_utils: after a timeout cancel that is NOT confirmed (cancel
        raised / still pending_cancel / fetch shows it live) the lifecycle
        must not send the market fallback and the maker ladder must abort
        as 'maker_unknown' — never a second live order for the same intent.
        Happy path (cancel settles the order) keeps the fallback / ladder.
  G2-3  trading_utils.cooldown_ok measures true elapsed seconds across a
        DST transition (naive local wall-clock stamps, .fold honoured).
  F3    llm_analyst._save_analysis: per-writer tmp + in-process lock — two
        loop threads saving simultaneously never tear the file, never fail
        the replace, and never wipe each other's section.
"""

import datetime as _dt
import json
import os
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import order_utils  # noqa: E402

# trading_utils does `from dotenv import load_dotenv` at import; stub it only
# where it is absent (dev Mac) and drop the stub again right after.
_stubbed_dotenv = False
try:
    import dotenv  # noqa: F401
except ImportError:
    _dv = types.ModuleType('dotenv')
    _dv.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _dv
    _stubbed_dotenv = True
try:
    import trading_utils  # noqa: E402
finally:
    if _stubbed_dotenv:
        sys.modules.pop('dotenv', None)

import llm_analyst  # noqa: E402


# ---------------------------------------------------------------------------
# G2-1 — fake broker
# ---------------------------------------------------------------------------

_LIVE = ('new', 'accepted', 'partially_filled', 'pending_cancel')


class _Broker:
    """Limit orders never fill passively. cancel_mode:
      'ok'             — cancel settles the order ('canceled'), like Alpaca;
      'raise'          — cancel raises (transient 5xx / timeout);
      'pending_cancel' — cancel accepted but order still pending_cancel;
      'raise_blind'    — cancel raises AND every later get_order raises.
    Market orders fill immediately."""

    def __init__(self, cancel_mode, partial=0.0):
        self.cancel_mode = cancel_mode
        self.partial = partial
        self.orders = {}
        self.submitted = []
        self.cancels = []
        self._blind = False

    def submit_order(self, **kw):
        oid = f'o{len(self.submitted) + 1}'
        o = SimpleNamespace(id=oid, symbol=kw['symbol'], qty=kw['qty'],
                            side=kw['side'], type=kw['type'], status='new',
                            filled_qty=0.0, filled_avg_price=None)
        if kw['type'] == 'market':
            o.status, o.filled_qty, o.filled_avg_price = (
                'filled', kw['qty'], 100.0)
        elif self.partial:
            o.status, o.filled_qty, o.filled_avg_price = (
                'partially_filled', self.partial, 100.0)
        self.orders[oid] = o
        self.submitted.append(kw)
        return o

    def get_order(self, oid):
        if self._blind:
            raise RuntimeError('503 Service Unavailable')
        return self.orders[oid]

    def cancel_order(self, oid):
        self.cancels.append(oid)
        if self.cancel_mode in ('raise', 'raise_blind'):
            if self.cancel_mode == 'raise_blind':
                self._blind = True
            raise RuntimeError('503 Service Unavailable')
        o = self.orders[oid]
        if o.status == 'filled':
            return
        o.status = ('canceled' if self.cancel_mode == 'ok'
                    else 'pending_cancel')

    def live_orders(self):
        return [o for o in self.orders.values() if o.status in _LIVE]

    def market_submits(self):
        return [s for s in self.submitted if s['type'] == 'market']


_Q = {'bid': 100.0, 'ask': 100.1, 'midpoint': 100.05, 'spread': 0.1,
      'spread_pct': 0.1}


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(order_utils.time, 'sleep', lambda s: None)
    tactics = []
    monkeypatch.setattr(order_utils, '_journal_entry_fills',
                        lambda sym, tactic, m, t: tactics.append(tactic))
    return tactics


def test_settled_statuses_constant():
    assert set(order_utils._SETTLED_STATUSES) == {
        'filled', 'canceled', 'expired', 'rejected'}
    for live in _LIVE:
        assert live not in order_utils._SETTLED_STATUSES


class TestLifecycleNoSecondOrder:

    @pytest.mark.parametrize('mode', ['raise', 'pending_cancel'])
    def test_unconfirmed_cancel_skips_market_fallback(self, fast, mode):
        b = _Broker(mode)
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        res = order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=True)
        assert b.market_submits() == []
        assert len(b.submitted) == 1
        assert len(b.live_orders()) == 1
        # freshest fetched state returned; caller judges by filled_qty
        assert res is b.orders[o.id]
        assert res.status not in order_utils._SETTLED_STATUSES

    def test_unconfirmed_cancel_keeps_partial_fill_evidence(self, fast):
        b = _Broker('pending_cancel', partial=0.4)
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        res = order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=True)
        assert b.market_submits() == []
        assert order_utils._filled_qty(res) == pytest.approx(0.4)

    def test_cancel_failed_and_state_unknown_returns_none(self, fast):
        b = _Broker('raise_blind')
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        res = order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=True)
        assert res is None
        assert b.market_submits() == []
        assert len(b.submitted) == 1

    def test_happy_path_cancel_settles_then_market_fallback(self, fast):
        b = _Broker('ok')
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        res = order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=True)
        mkt = b.market_submits()
        assert len(mkt) == 1 and mkt[0]['qty'] == pytest.approx(1.0)
        assert b.orders[o.id].status == 'canceled'
        assert b.live_orders() == []
        assert res.status == 'filled'

    def test_happy_path_partial_chases_only_remainder(self, fast):
        b = _Broker('ok', partial=0.4)
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=True)
        mkt = b.market_submits()
        assert len(mkt) == 1 and mkt[0]['qty'] == pytest.approx(0.6)

    def test_no_fallback_requested_is_unchanged(self, fast):
        # fallback_to_market=False never submitted anything before; the
        # new guard only applies when a fallback would be sent.
        b = _Broker('pending_cancel')
        o = b.submit_order(symbol='BTC/USD', qty=1.0, side='buy',
                           type='limit')
        res = order_utils.manage_order_lifecycle(
            b, o.id, timeout=4, poll_interval=2, fallback_to_market=False)
        assert res is b.orders[o.id]
        assert len(b.submitted) == 1


class TestMakerLadderNoStacking:

    @pytest.mark.parametrize('mode', ['raise', 'pending_cancel'])
    def test_working_rung_aborts_as_maker_unknown(self, fast, mode):
        b = _Broker(mode)
        res, tactic = order_utils.place_maker_buy(
            b, 'BTC/USD', 1000.0, lambda: dict(_Q),
            stage_timeout=4, max_reprices=1)
        assert tactic == 'maker_unknown'
        assert fast == ['maker_unknown']
        assert len(b.submitted) == 1            # no 2nd rung, no taker
        assert b.market_submits() == []
        assert len(b.live_orders()) == 1        # exactly one working order

    def test_working_partial_rung_returned_as_best_evidence(self, fast):
        b = _Broker('pending_cancel', partial=3.0)
        res, tactic = order_utils.place_maker_buy(
            b, 'BTC/USD', 1000.0, lambda: dict(_Q),
            stage_timeout=4, max_reprices=1)
        assert tactic == 'maker_unknown'
        assert order_utils._filled_qty(res) == pytest.approx(3.0)
        assert len(b.submitted) == 1

    def test_happy_path_ladder_reprices_then_falls_back(self, fast):
        # Cancel settles each rung -> ladder proceeds rung 2, then the
        # taker fallback (unchanged behaviour).
        b = _Broker('ok')
        res, tactic = order_utils.place_maker_buy(
            b, 'BTC/USD', 1000.0, lambda: dict(_Q),
            stage_timeout=4, max_reprices=1)
        limits = [s for s in b.submitted if s['type'] == 'limit']
        assert len(limits) >= 3                 # 2 rungs + fallback limit
        assert tactic != 'maker_unknown'
        assert len(b.market_submits()) == 1
        assert b.live_orders() == []


# ---------------------------------------------------------------------------
# G2-3 — cooldown across DST
# ---------------------------------------------------------------------------
#
# Naive datetime.timestamp() uses the process's local zone, so the DST cases
# run in a child interpreter with TZ=America/Chicago (the Jetson's zone).
# A child process is used rather than time.tzset(): some builds (e.g. the
# Jetson conda py3.10) do not expose tzset.

_DST_CHILD = r'''
import datetime as _dt, json, sys, time, types
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
if time.strftime('%Z', time.localtime(1790000000)) not in ('CDT', 'CST'):
    print(json.dumps({'skip': 'America/Chicago zone data unavailable'}))
    raise SystemExit(0)
try:
    import dotenv  # noqa: F401
except ImportError:
    _dv = types.ModuleType('dotenv'); _dv.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = _dv
import trading_utils as tu
real = _dt.datetime

def at(now, last, minutes):
    class _FakeDT(real):
        @classmethod
        def now(cls, tz=None):
            return now
    tu.datetime = SimpleNamespace(datetime=_FakeDT)
    try:
        return tu.cooldown_ok({'X': last}, 'X', cooldown_minutes=minutes)
    finally:
        tu.datetime = _dt

out = {}
# spring-forward: 01:50 CST -> 03:05 CDT = 15 real minutes (naive says 75)
sl, sn = real(2027, 3, 14, 1, 50), real(2027, 3, 14, 3, 5)
out['spring_elapsed'] = sn.timestamp() - sl.timestamp()
out['spring'] = [at(sn, sl, 15), at(sn, sl, 30), at(sn, sl, 60)]
# fall-back: first 01:50 CDT -> second 01:30 CST = 40 real min (naive -20)
fl, fn = real(2026, 11, 1, 1, 50, fold=0), real(2026, 11, 1, 1, 30, fold=1)
out['fall_elapsed'] = fn.timestamp() - fl.timestamp()
out['fall'] = [at(fn, fl, 30), at(fn, fl, 60)]
# restore path: base_loop rebuilds stamps via fromtimestamp(), which sets
# .fold in the repeated hour, so .timestamp() is exact there too.
ts = real(2026, 11, 1, 1, 10, fold=1).timestamp()
rl = real.fromtimestamp(ts)
out['restore_fold'] = rl.fold
out['restore'] = [at(real.fromtimestamp(ts + 31 * 60), rl, 30),
                  at(real.fromtimestamp(ts + 29 * 60), rl, 30)]
print(json.dumps(out))
'''


@pytest.fixture(scope='module')
def dst_results():
    import subprocess
    env = dict(os.environ, TZ='America/Chicago', CUDA_VISIBLE_DEVICES='')
    proc = subprocess.run([sys.executable, '-c', _DST_CHILD, str(REPO)],
                          env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if 'skip' in out:
        pytest.skip(out['skip'])
    return out


class TestCooldownDST:

    def test_spring_forward_true_elapsed(self, dst_results):
        assert dst_results['spring_elapsed'] == 15 * 60
        # cooldown_ok(15m)=True, (30m)=False, (60m)=False — the old naive
        # subtraction said 75 min and returned True for all three.
        assert dst_results['spring'] == [True, False, False]

    def test_fall_back_true_elapsed(self, dst_results):
        assert dst_results['fall_elapsed'] == 40 * 60
        # old naive subtraction said -20 min -> (30m) False
        assert dst_results['fall'] == [True, False]

    def test_restore_path_fromtimestamp_sets_fold(self, dst_results):
        assert dst_results['restore_fold'] == 1
        assert dst_results['restore'] == [True, False]

    def test_no_duration_change_off_dst(self):
        now = _dt.datetime.now()
        assert trading_utils.cooldown_ok(
            {'X': now - _dt.timedelta(minutes=29)}, 'X', 30) is False
        assert trading_utils.cooldown_ok(
            {'X': now - _dt.timedelta(minutes=31)}, 'X', 30) is True

    def test_aware_stamps_supported(self):
        now = _dt.datetime.now(_dt.timezone.utc)
        assert trading_utils.cooldown_ok(
            {'X': now - _dt.timedelta(minutes=31)}, 'X', 30) is True
        assert trading_utils.cooldown_ok(
            {'X': now - _dt.timedelta(minutes=5)}, 'X', 30) is False


# ---------------------------------------------------------------------------
# F3 — llm_analysis.json concurrent writers
# ---------------------------------------------------------------------------

def _payload(book, n=40):
    return {f'{book}{i}': {'s': 0.5, 'r': 'x' * 300, 'bull': 'b' * 150,
                           'bear': 'c' * 150} for i in range(n)}


@pytest.fixture
def analysis_file(tmp_path, monkeypatch):
    target = tmp_path / 'llm_analysis.json'
    monkeypatch.setattr(llm_analyst, '_ANALYSIS_FILE', target)
    errs = []
    real_print = print

    def _capture(*a, **k):
        s = ' '.join(map(str, a))
        if 'Error saving analysis' in s:
            errs.append(s)
        else:
            real_print(*a, **k)
    monkeypatch.setattr(llm_analyst, 'print', _capture, raising=False)
    return target, errs


def test_save_analysis_two_thread_stress(analysis_file):
    target, errs = analysis_file
    N = 300
    bad_reads = []
    missing_book = []
    barrier = threading.Barrier(2)
    checkpoint = threading.Barrier(3)

    def worker(book):
        p = _payload(book)
        for _ in range(N):
            barrier.wait()
            llm_analyst._save_analysis(p, book, 'm')
            checkpoint.wait()

    def reader(stop):
        while not stop.is_set():
            try:
                raw = target.read_text()
            except FileNotFoundError:
                continue
            try:
                json.loads(raw)
            except json.JSONDecodeError as e:
                bad_reads.append(str(e)[:60])

    stop = threading.Event()
    r = threading.Thread(target=reader, args=(stop,))
    r.start()
    ts = [threading.Thread(target=worker, args=(b,))
          for b in ('crypto', 'stock')]
    for t in ts:
        t.start()
    try:
        for _ in range(N):
            checkpoint.wait()          # both saves of this round done
            try:
                d = json.loads(target.read_text())
            except json.JSONDecodeError as e:
                bad_reads.append('installed: ' + str(e)[:60])
                continue
            if set(d) != {'crypto', 'stock'}:
                missing_book.append(sorted(d))
    finally:
        for t in ts:
            t.join()
        stop.set()
        r.join()

    assert errs == []                  # 0 replace failures
    assert bad_reads == []             # 0 torn files (seen or installed)
    assert missing_book == []          # no section ever wiped
    final = json.loads(target.read_text())
    assert set(final) == {'crypto', 'stock'}
    assert len(final['crypto']) == 40 and len(final['stock']) == 40
    assert list(target.parent.glob('*.tmp')) == []   # no tmp left behind


def test_save_analysis_preserves_other_section_and_newest_entry(
        analysis_file):
    # Section semantics unchanged: a save touches only its own asset_type
    # section and overwrites the symbol's record with a fresh timestamp.
    target, errs = analysis_file
    llm_analyst._save_analysis({'BTC/USD': {'s': 0.2}}, 'crypto', 'm1')
    llm_analyst._save_analysis({'AAPL': {'s': 0.3}}, 'stock', 'm1')
    llm_analyst._save_analysis({'BTC/USD': {'s': 0.6}}, 'crypto', 'm2')
    d = json.loads(target.read_text())
    assert d['stock']['AAPL']['s'] == 0.3
    assert d['crypto']['BTC/USD']['s'] == 0.6
    assert d['crypto']['BTC/USD']['model'] == 'm2'
    assert errs == []


def test_save_analysis_tmp_name_is_per_writer(analysis_file, monkeypatch):
    target, errs = analysis_file
    seen = []
    real_replace = os.replace

    def _spy(src, dst):
        seen.append(Path(src).name)
        return real_replace(src, dst)
    monkeypatch.setattr(llm_analyst.os, 'replace', _spy)
    llm_analyst._save_analysis({'ETH/USD': {'s': 0.1}}, 'crypto', 'm')
    assert seen == [f'llm_analysis.json.{os.getpid()}.'
                    f'{threading.get_ident()}.tmp']


def test_save_analysis_failed_replace_cleans_tmp(analysis_file, monkeypatch):
    target, errs = analysis_file

    def _boom(src, dst):
        raise OSError('disk full')
    monkeypatch.setattr(llm_analyst.os, 'replace', _boom)
    llm_analyst._save_analysis({'ETH/USD': {'s': 0.1}}, 'crypto', 'm')
    assert len(errs) == 1
    assert list(target.parent.glob('*.tmp')) == []
    assert not target.exists()
