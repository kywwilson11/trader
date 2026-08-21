"""Synthetic-data unit tests for journal_stats.py — pure stdlib, runs on the
dev Mac (no numpy/pandas/torch/etc. anywhere in this module or its target).

Synthetic journal rows below use the REAL buy/sell schema, cited exactly as
journal_stats.py's own module docstring cites it:
  buy  -> base_loop.py:1950-1961 (crypto/shared) & stock_loop.py:961-973
          (stock): symbol, action, final_notional, decision_price,
          fill_price, ts.
  sell -> base_loop.py:1080-1086 `_record_confirmed_exit` (both books, every
          exit_reason) & stock_loop.py:643-649 `_journal_external_close`:
          symbol, action, exit_reason, pnl_pct, decision_price, fill_price,
          slippage_bps, estimated, ts.
Only the keys journal_stats.py actually reads are included in the fixtures
below (symbol/action/final_notional/ts for buys; symbol/action/pnl_pct/
exit_reason/ts for sells) — the rest of the real schema (llm_*, sentiment_*,
sizing, conviction fields, slippage_bps, estimated, decision_price,
fill_price) is deliberately omitted since journal_stats.py ignores it via
.get(), exactly like decision_report.py/chart_core.py's readers do.
"""

import datetime
import json

import pytest

import journal_stats as js


def _write_day(tmp_path, day, rows, corrupt=True):
    """Write one journals/YYYY-MM-DD.jsonl file. By default also appends a
    corrupt line + a blank line, mirroring
    tests/test_chart_core.py::TestLoadTradeMarkers._write_day (the sibling
    reader's own test helper) so both readers are exercised the same way."""
    fp = tmp_path / f"{day.isoformat()}.jsonl"
    with open(fp, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
        if corrupt:
            f.write('{not valid json\n')
            f.write('\n')
    return fp


def _buy(symbol, ts, notional):
    return {'symbol': symbol, 'action': 'buy', 'final_notional': notional,
            'ts': ts.isoformat()}


def _sell(symbol, ts, pnl_pct, exit_reason='signal_sell'):
    return {'symbol': symbol, 'action': 'sell', 'exit_reason': exit_reason,
            'pnl_pct': pnl_pct, 'ts': ts.isoformat()}


# ---------------------------------------------------------------------------
# Hand-computed exactness (win_rate / expectancy / profit_factor / etc.)
# ---------------------------------------------------------------------------
class TestHandComputedStats:
    def _build(self, tmp_path):
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        t0, t0x = now, now + datetime.timedelta(hours=2)
        t1, t1x = now + datetime.timedelta(hours=3), now + datetime.timedelta(hours=6)
        t2, t2x = now + datetime.timedelta(hours=7), now + datetime.timedelta(hours=8)
        t3, t3x = now + datetime.timedelta(hours=9), now + datetime.timedelta(hours=14)
        rows = [
            _buy('BTC/USD', t0, 1000.0),
            _sell('BTC/USD', t0x, 5.0, 'signal_sell'),     # win
            _buy('BTC/USD', t1, 1000.0),
            _sell('BTC/USD', t1x, -2.0, 'hard_stop'),      # loss
            _buy('ETH/USD', t2, 500.0),
            _sell('ETH/USD', t2x, 10.0, 'take_profit'),    # win
            _buy('AAPL', t3, 2000.0),
            _sell('AAPL', t3x, -4.0, 'signal_sell'),       # loss
        ]
        _write_day(tmp_path, now.date(), rows)
        return now

    def test_overall_exactness(self, tmp_path):
        now = self._build(tmp_path)
        t2x = now + datetime.timedelta(hours=8)    # ETH/USD sell — the best trade
        t3x = now + datetime.timedelta(hours=14)   # AAPL sell — the worst trade
        trades = js.load_trades(tmp_path)
        assert len(trades) == 4
        o = js.compute_stats(trades)['overall']
        assert o['n_trades'] == 4
        assert o['win_rate'] == pytest.approx(0.5)
        assert o['expectancy_pct'] == pytest.approx(2.25)          # (5-2+10-4)/4
        assert o['profit_factor'] == pytest.approx(2.5)            # 15/6
        assert o['avg_win_pct'] == pytest.approx(7.5)               # (5+10)/2
        assert o['avg_loss_pct'] == pytest.approx(-3.0)             # (-2-4)/2
        assert o['median_holding_hours'] == pytest.approx(2.5)      # sorted[2,3,1,5]->[1,2,3,5]
        assert o['best_trade'] == {'symbol': 'ETH/USD', 'pnl_pct': pytest.approx(10.0),
                                    'ts': pytest.approx(t2x.timestamp())}
        assert o['worst_trade'] == {'symbol': 'AAPL', 'pnl_pct': pytest.approx(-4.0),
                                     'ts': pytest.approx(t3x.timestamp())}
        assert o['by_exit_reason'] == {'signal_sell': 2, 'hard_stop': 1, 'take_profit': 1}

    def test_pnl_dollars_from_notional_pairing(self, tmp_path):
        self._build(tmp_path)
        trades = js.load_trades(tmp_path)
        by_symbol = {}
        for t in trades:
            by_symbol.setdefault(t['symbol'], []).append(t)
        btc = sorted(by_symbol['BTC/USD'], key=lambda t: t['exit_ts'])
        assert btc[0]['pnl_dollars'] == pytest.approx(50.0)    # 1000 * 5%
        assert btc[1]['pnl_dollars'] == pytest.approx(-20.0)   # 1000 * -2%
        assert by_symbol['ETH/USD'][0]['pnl_dollars'] == pytest.approx(50.0)   # 500 * 10%
        assert by_symbol['AAPL'][0]['pnl_dollars'] == pytest.approx(-80.0)     # 2000 * -4%
        for t in trades:
            assert t['entry_ts'] is not None
            assert t['holding_hours'] is not None and t['holding_hours'] > 0

    def test_by_book_grouping(self, tmp_path):
        self._build(tmp_path)
        stats = js.compute_stats(js.load_trades(tmp_path))
        crypto, stock = stats['by_book']['crypto'], stats['by_book']['stock']
        assert crypto['n_trades'] == 3   # 2x BTC/USD + 1x ETH/USD
        assert crypto['win_rate'] == pytest.approx(2 / 3)
        assert crypto['profit_factor'] == pytest.approx(7.5)   # 15/2
        assert stock['n_trades'] == 1    # AAPL
        assert stock['win_rate'] == pytest.approx(0.0)
        assert stock['profit_factor'] == pytest.approx(0.0)    # 0/4, has a loss -> not None
        assert stock['avg_win_pct'] is None

    def test_by_symbol_grouping(self, tmp_path):
        self._build(tmp_path)
        by_symbol = js.compute_stats(js.load_trades(tmp_path))['by_symbol']
        assert set(by_symbol) == {'BTC/USD', 'ETH/USD', 'AAPL'}
        assert by_symbol['BTC/USD']['n_trades'] == 2
        assert by_symbol['ETH/USD']['n_trades'] == 1
        assert by_symbol['AAPL']['n_trades'] == 1


# ---------------------------------------------------------------------------
# profit_factor edge cases
# ---------------------------------------------------------------------------
class TestProfitFactorEdgeCases:
    def test_no_losses_profit_factor_none(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), 2.0),
            _buy('BTC/USD', now + datetime.timedelta(hours=2), 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=3), 5.0),
        ], corrupt=False)
        stats = js.compute_stats(js.load_trades(tmp_path))
        assert stats['overall']['n_trades'] == 2
        assert stats['overall']['profit_factor'] is None
        assert stats['overall']['avg_loss_pct'] is None
        assert stats['overall']['win_rate'] == pytest.approx(1.0)

    def test_all_losses_profit_factor_zero_not_none(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), -2.0),
        ], corrupt=False)
        stats = js.compute_stats(js.load_trades(tmp_path))
        assert stats['overall']['profit_factor'] == pytest.approx(0.0)
        assert stats['overall']['avg_win_pct'] is None


# ---------------------------------------------------------------------------
# Corrupt-line tolerance
# ---------------------------------------------------------------------------
class TestCorruptLineTolerance:
    def test_corrupt_lines_counted_not_raised(self, tmp_path):
        now = datetime.datetime.now().astimezone()
        fp = tmp_path / f"{now.date().isoformat()}.jsonl"
        good = [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), 2.0),
        ]
        with open(fp, 'w') as f:
            for r in good:
                f.write(json.dumps(r) + '\n')
            f.write('{not valid json at all\n')                          # JSON parse error
            f.write('[1, 2, 3]\n')                                       # valid JSON, not an object
            f.write(json.dumps({'action': 'buy', 'final_notional': 1.0,
                                 'ts': now.isoformat()}) + '\n')          # missing symbol
            f.write(json.dumps({'symbol': 'ETH/USD', 'action': 'sell',
                                 'exit_reason': 'signal_sell',
                                 'ts': now.isoformat()}) + '\n')          # missing pnl_pct
            f.write(json.dumps({'symbol': 'ETH/USD', 'action': 'buy',
                                 'final_notional': 1.0}) + '\n')         # missing ts
            f.write('\n')                                                # blank line, NOT corrupt

        stats = {}
        trades = js.load_trades(tmp_path, stats=stats)
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'BTC/USD'
        assert stats['corrupt_lines'] == 5
        assert stats['files_read'] == 1

    def test_skip_and_other_actions_ignored_not_corrupt(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone()
        fp = tmp_path / f"{today.isoformat()}.jsonl"
        with open(fp, 'w') as f:
            f.write(json.dumps({'symbol': 'BTC/USD', 'action': 'skip',
                                 'skip_reason': 'llm_veto', 'ts': now.isoformat()}) + '\n')
            f.write(json.dumps({'action': 'llm_analysis', 'asset_type': 'crypto',
                                 'ts': now.isoformat()}) + '\n')
            f.write(json.dumps({'action': 'account_risk', 'book': 'crypto',
                                 'ts': now.isoformat()}) + '\n')
        stats = {}
        trades = js.load_trades(tmp_path, stats=stats)
        assert trades == []
        assert stats['corrupt_lines'] == 0

    def test_non_date_named_file_ignored(self, tmp_path):
        (tmp_path / "notes.jsonl").write_text(
            json.dumps(_buy('BTC/USD', datetime.datetime.now().astimezone(), 1.0)) + '\n')
        trades = js.load_trades(tmp_path)
        assert trades == []


# ---------------------------------------------------------------------------
# since_ts / until_ts filtering
# ---------------------------------------------------------------------------
class TestSinceUntilFilter:
    def _two_days(self, tmp_path):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        now = datetime.datetime.now().astimezone().replace(
            hour=12, minute=0, second=0, microsecond=0)
        yest = now - datetime.timedelta(days=1)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), 3.0),
        ], corrupt=False)
        _write_day(tmp_path, yesterday, [
            _buy('ETH/USD', yest, 500.0),
            _sell('ETH/USD', yest + datetime.timedelta(hours=1), -1.0),
        ], corrupt=False)
        return today, now

    def test_since_ts_excludes_earlier_trade(self, tmp_path):
        today, now = self._two_days(tmp_path)
        midnight_today = datetime.datetime.combine(
            today, datetime.time(0, 0), tzinfo=now.tzinfo).timestamp()
        trades = js.load_trades(tmp_path, since_ts=midnight_today)
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'BTC/USD'

    def test_until_ts_excludes_later_trade(self, tmp_path):
        today, now = self._two_days(tmp_path)
        midnight_today = datetime.datetime.combine(
            today, datetime.time(0, 0), tzinfo=now.tzinfo).timestamp()
        trades = js.load_trades(tmp_path, until_ts=midnight_today)
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'ETH/USD'

    def test_window_splitting_buy_from_sell_leaves_entry_ts_none(self, tmp_path):
        """Documented tradeoff (see journal_stats.py module docstring):
        since_ts filters rows, not whole positions, so a window opening
        strictly between a buy and its sell still yields the trade (the
        sell is self-contained) but with entry_ts/holding_hours/
        pnl_dollars unresolved."""
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=12, minute=0, second=0, microsecond=0)
        buy_ts = now
        sell_ts = now + datetime.timedelta(hours=2)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', buy_ts, 1000.0),
            _sell('BTC/USD', sell_ts, 3.0),
        ], corrupt=False)
        cutoff = (buy_ts + datetime.timedelta(hours=1)).timestamp()
        trades = js.load_trades(tmp_path, since_ts=cutoff)
        assert len(trades) == 1
        t = trades[0]
        assert t['entry_ts'] is None
        assert t['holding_hours'] is None
        assert t['pnl_dollars'] is None
        assert t['pnl_pct'] == pytest.approx(3.0)
        # still usable for win/expectancy stats despite the missing pairing
        stats = js.compute_stats(trades)
        assert stats['overall']['n_trades'] == 1
        assert stats['overall']['win_rate'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Buy/sell pairing semantics
# ---------------------------------------------------------------------------
class TestPairing:
    def test_scale_in_overwrites_pairing_slot_lifo(self, tmp_path):
        """A second buy before the sell (position add) overwrites the
        pairing slot with the NEWER buy's ts/notional — mirrors
        base_loop.py:1937-1944 overwriting pos.entry_price on every add
        rather than blending a weighted-average cost basis."""
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        t0 = now
        t1 = now + datetime.timedelta(hours=1)
        t2 = now + datetime.timedelta(hours=4)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', t0, 1000.0),
            _buy('BTC/USD', t1, 2000.0),   # scale-in add
            _sell('BTC/USD', t2, 4.0),
        ], corrupt=False)
        trades = js.load_trades(tmp_path)
        assert len(trades) == 1
        t = trades[0]
        assert t['entry_ts'] == pytest.approx(t1.timestamp())
        assert t['holding_hours'] == pytest.approx((t2 - t1).total_seconds() / 3600.0)
        assert t['pnl_dollars'] == pytest.approx(2000.0 * 4.0 / 100.0)

    def test_sell_without_preceding_buy_still_counts_with_none_entry(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        _write_day(tmp_path, today, [_sell('BTC/USD', now, 3.0)], corrupt=False)
        trades = js.load_trades(tmp_path)
        assert len(trades) == 1
        assert trades[0]['entry_ts'] is None
        assert trades[0]['holding_hours'] is None
        assert trades[0]['pnl_dollars'] is None
        assert trades[0]['pnl_pct'] == pytest.approx(3.0)

    def test_symbol_format_derives_book(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), 1.0),
            _buy('AAPL', now, 1000.0),
            _sell('AAPL', now + datetime.timedelta(hours=1), 1.0),
        ], corrupt=False)
        trades = {t['symbol']: t for t in js.load_trades(tmp_path)}
        assert trades['BTC/USD']['book'] == 'crypto'
        assert trades['AAPL']['book'] == 'stock'


# ---------------------------------------------------------------------------
# daily_realized
# ---------------------------------------------------------------------------
class TestDailyRealized:
    def test_daily_sum_and_none_when_no_dollars(self, tmp_path):
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        now = datetime.datetime.now().astimezone().replace(
            hour=12, minute=0, second=0, microsecond=0)
        yest = now - datetime.timedelta(days=1)
        _write_day(tmp_path, yesterday, [
            _buy('BTC/USD', yest, 1000.0),
            _sell('BTC/USD', yest + datetime.timedelta(hours=1), 2.0),
            _buy('ETH/USD', yest, 500.0),
            _sell('ETH/USD', yest + datetime.timedelta(hours=2), -4.0),
        ], corrupt=False)
        # today: a sell with no matching buy -> pnl_dollars unresolved
        _write_day(tmp_path, today, [_sell('AAPL', now, 1.0)], corrupt=False)

        trades = js.load_trades(tmp_path)
        daily = dict(js.compute_stats(trades)['daily_realized'])
        assert daily[yesterday.isoformat()] == pytest.approx(1000 * 2 / 100 + 500 * -4 / 100)
        assert daily[today.isoformat()] is None


# ---------------------------------------------------------------------------
# Empty / missing-directory safety
# ---------------------------------------------------------------------------
class TestEmptyAndMissingSafety:
    def test_missing_directory_returns_empty_without_raising(self, tmp_path):
        trades = js.load_trades(tmp_path / 'does_not_exist')
        assert trades == []
        stats = js.compute_stats(trades)
        assert stats['overall']['n_trades'] == 0
        assert stats['overall']['win_rate'] is None
        assert stats['overall']['profit_factor'] is None
        assert stats['overall']['best_trade'] is None
        assert stats['by_book'] == {}
        assert stats['by_symbol'] == {}
        assert stats['daily_realized'] == []

    def test_empty_directory_no_files(self, tmp_path):
        assert js.load_trades(tmp_path) == []

    def test_format_summary_empty(self):
        text = js.format_summary(js.compute_stats([]))
        assert 'No closed trades' in text

    def test_format_summary_nonempty_smoke(self, tmp_path):
        today = datetime.date.today()
        now = datetime.datetime.now().astimezone().replace(
            hour=1, minute=0, second=0, microsecond=0)
        _write_day(tmp_path, today, [
            _buy('BTC/USD', now, 1000.0),
            _sell('BTC/USD', now + datetime.timedelta(hours=1), 5.0),
        ], corrupt=False)
        stats = js.compute_stats(js.load_trades(tmp_path))
        text = js.format_summary(stats)
        assert 'Overall: 1 trades' in text
        assert 'crypto' in text
        assert 'BTC/USD' in text          # best/worst trade line
        assert 'PF n/a' in text           # single win, no losses -> profit_factor None
