"""c26 packet T7 — ops/measurement remainder tests.

Covers: journal gzip retention (TRADER_JOURNAL_ROTATE_DAYS, default OFF)
and transparent .gz readers; journal_stats date fast-path + new buy-derived
trade keys + EOD digest builder; execution_report additive blocks
(quote-age slippage, maker notional share, LLM economics); shadow promotion
ledger (append-only, decision logic untouched); run_bots ops thread
(kill switch defer/crash isolation, daily guards); options_overlay verdict
runner; run_pipeline EOD digest hook.

All Mac-green (stdlib + numpy only). Tests monkeypatch module CONSTANTS,
never env vars.
"""
import datetime
import gzip
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trade_journal as tj
import journal_stats as js
import execution_report as er
import options_overlay as oo
import run_bots as rb
import run_pipeline as rp
import shadow as sh


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ts(day, hour=12, minute=0):
    """Offset-aware ISO ts inside calendar day `day` (matches the writer's
    one-clock-read filename/ts invariant)."""
    return datetime.datetime.combine(
        day, datetime.time(hour, minute)).astimezone().isoformat()


def _write_journal(dirpath, day, rows):
    p = Path(dirpath) / f'{day.isoformat()}.jsonl'
    with open(p, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    return p


def _gzip_file(path, remove_plain=True):
    path = Path(path)
    gz = Path(f'{path}.gz')
    gz.write_bytes(gzip.compress(path.read_bytes()))
    if remove_plain:
        path.unlink()
    return gz


TODAY = datetime.date.today()


# ---------------------------------------------------------------------------
# 1) Journal rotation (trade_journal.rotate_old_journals)
# ---------------------------------------------------------------------------

class TestJournalRotation:
    @pytest.fixture(autouse=True)
    def _sandbox(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tj, 'JOURNAL_DIR', tmp_path)
        monkeypatch.setattr(tj, '_rotate_done_date', None)
        self.dir = tmp_path

    def test_flag_off_is_noop(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 0)
        old = _write_journal(self.dir, TODAY - datetime.timedelta(days=100),
                             [{'action': 'skip'}])
        assert tj.rotate_old_journals() == 0
        assert old.exists()
        assert list(self.dir.glob('*.gz')) == []

    def test_rotates_old_keeps_recent(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 7)
        old = _write_journal(self.dir, TODAY - datetime.timedelta(days=10),
                             [{'action': 'buy', 'symbol': 'BTC/USD'}])
        raw = old.read_bytes()
        recent = _write_journal(self.dir, TODAY - datetime.timedelta(days=3),
                                [{'action': 'skip'}])
        today_f = _write_journal(self.dir, TODAY, [{'action': 'skip'}])
        assert tj.rotate_old_journals() == 1
        assert not old.exists()
        gz = Path(f'{old}.gz')
        assert gz.exists()
        assert gzip.decompress(gz.read_bytes()) == raw
        assert recent.exists() and today_f.exists()
        assert not Path(f'{recent}.gz').exists()
        assert not Path(f'{today_f}.gz').exists()

    def test_crash_leftover_both_files(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 7)
        # (a) matching gz: finish the unlink, keep the gz
        p1 = _write_journal(self.dir, TODAY - datetime.timedelta(days=20),
                            [{'action': 'sell', 'symbol': 'BTC/USD'}])
        raw1 = p1.read_bytes()
        Path(f'{p1}.gz').write_bytes(gzip.compress(raw1))
        # (b) MISmatching gz: rebuilt from the plain file (rows never lost)
        p2 = _write_journal(self.dir, TODAY - datetime.timedelta(days=21),
                            [{'action': 'buy', 'symbol': 'ETH/USD'}])
        raw2 = p2.read_bytes()
        Path(f'{p2}.gz').write_bytes(gzip.compress(b'stale other content'))
        assert tj.rotate_old_journals() == 2
        assert not p1.exists() and not p2.exists()
        assert gzip.decompress(Path(f'{p1}.gz').read_bytes()) == raw1
        assert gzip.decompress(Path(f'{p2}.gz').read_bytes()) == raw2

    def test_non_date_ignored_and_stale_tmp_cleaned(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 7)
        notes = self.dir / 'notes.jsonl'
        notes.write_text('{"x": 1}\n')
        stale = self.dir / '2020-01-01.jsonl.gz.tmp'
        stale.write_bytes(b'partial')
        two_days_ago = datetime.datetime.now().timestamp() - 2 * 86400
        os.utime(stale, (two_days_ago, two_days_ago))
        assert tj.rotate_old_journals() == 0
        assert notes.exists()          # never rotated
        assert not notes.with_suffix('.jsonl.gz').exists()
        assert not stale.exists()      # cleaned

    def test_per_call_cap(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 7)
        monkeypatch.setattr(tj, '_ROTATE_MAX_PER_CALL', 3)
        for i in range(8):
            _write_journal(self.dir, TODAY - datetime.timedelta(days=30 + i),
                           [{'action': 'skip'}])
        assert tj.rotate_old_journals() == 3
        assert len(list(self.dir.glob('*.jsonl.gz'))) == 3
        assert len(list(self.dir.glob('*.jsonl'))) == 5

    def test_log_decision_triggers_once_per_day(self, monkeypatch):
        monkeypatch.setattr(tj, 'JOURNAL_ROTATE_DAYS', 7)
        monkeypatch.setattr(tj, 'load_llm_config',
                            lambda: {'journal_enabled': True})
        old1 = _write_journal(self.dir, TODAY - datetime.timedelta(days=15),
                              [{'action': 'skip'}])
        tj.log_decision({'action': 'skip', 'symbol': 'X'})
        assert not old1.exists()
        assert Path(f'{old1}.gz').exists()
        # second call same day: rotation guard short-circuits
        old2 = _write_journal(self.dir, TODAY - datetime.timedelta(days=16),
                              [{'action': 'skip'}])
        tj.log_decision({'action': 'skip', 'symbol': 'Y'})
        assert old2.exists()
        assert not Path(f'{old2}.gz').exists()


# ---------------------------------------------------------------------------
# 2) Transparent gz readers
# ---------------------------------------------------------------------------

class TestGzTransparentReaders:
    def test_iter_journal_rows_reads_gz(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tj, 'JOURNAL_DIR', tmp_path)
        rows = [{'action': 'buy', 'symbol': 'BTC/USD', 'ts': _ts(TODAY)},
                {'action': 'sell', 'symbol': 'BTC/USD', 'ts': _ts(TODAY, 13)}]
        p = _write_journal(tmp_path, TODAY, rows)
        plain = list(tj.iter_journal_rows(0))
        _gzip_file(p)
        gzed = list(tj.iter_journal_rows(0))
        assert plain == gzed == rows

    def test_get_journal_summary_gz_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tj, 'JOURNAL_DIR', tmp_path)
        rows = [{'action': 'buy'}, {'action': 'sell'},
                {'action': 'skip', 'skip_reason': 'llm_veto'}]
        p = _write_journal(tmp_path, TODAY, rows)
        before = tj.get_journal_summary(TODAY.isoformat())
        _gzip_file(p)
        after = tj.get_journal_summary(TODAY.isoformat())
        assert before == after
        assert after['buys'] == 1 and after['sells'] == 1
        assert after['skips'] == 1 and after['llm_blocks'] == 1

    def test_open_journal_precedence_and_missing(self, tmp_path):
        p = tmp_path / '2026-01-01.jsonl'
        p.write_text('plain\n')
        Path(f'{p}.gz').write_bytes(gzip.compress(b'gz-version\n'))
        with tj.open_journal(p) as f:
            assert f.read() == 'plain\n'   # plain wins when both exist
        p.unlink()
        with tj.open_journal(p) as f:
            assert f.read() == 'gz-version\n'
        Path(f'{p}.gz').unlink()
        with pytest.raises(FileNotFoundError):
            tj.open_journal(p)

    def test_execution_report_load_reads_gz(self, monkeypatch, tmp_path):
        # er binds JOURNAL_DIR at import — patch the er attribute.
        monkeypatch.setattr(er, 'JOURNAL_DIR', tmp_path)
        p = _write_journal(tmp_path, TODAY, [
            {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 3.0}])
        _gzip_file(p)
        rows, n_skipped = er._load(0)
        assert n_skipped == 0
        assert len(rows) == 1 and rows[0]['slippage_bps'] == 3.0

    def test_load_trades_gz_only_and_plain_preference(self, tmp_path):
        rows = [{'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 100,
                 'ts': _ts(TODAY, 10)},
                {'action': 'sell', 'symbol': 'BTC/USD', 'pnl_pct': 2.0,
                 'exit_reason': 'tp', 'ts': _ts(TODAY, 11)}]
        p = _write_journal(tmp_path, TODAY, rows)
        _gzip_file(p, remove_plain=True)
        trades = js.load_trades(tmp_path)
        assert len(trades) == 1
        assert trades[0]['pnl_dollars'] == pytest.approx(2.0)
        # both exist -> plain wins (write DIFFERENT plain content)
        _write_journal(tmp_path, TODAY, rows + [
            {'action': 'sell', 'symbol': 'ETH/USD', 'pnl_pct': -1.0,
             'exit_reason': 'stop', 'ts': _ts(TODAY, 12)}])
        assert len(js.load_trades(tmp_path)) == 2


# ---------------------------------------------------------------------------
# 3) load_trades date fast-path
# ---------------------------------------------------------------------------

def _three_day_journal(tmp_path):
    for offset in (0, 5, 20):
        d = TODAY - datetime.timedelta(days=offset)
        _write_journal(tmp_path, d, [
            {'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 100,
             'ts': _ts(d, 10)},
            {'action': 'sell', 'symbol': 'BTC/USD', 'pnl_pct': 1.0,
             'exit_reason': 'tp', 'ts': _ts(d, 11)}])


class TestLoadTradesFastPath:
    def test_bounded_matches_manual_filter(self, tmp_path):
        _three_day_journal(tmp_path)
        since = datetime.datetime.combine(
            TODAY - datetime.timedelta(days=6),
            datetime.time.min).astimezone().timestamp()
        unbounded = js.load_trades(tmp_path)
        bounded = js.load_trades(tmp_path, since_ts=since)
        assert bounded == [t for t in unbounded if t['exit_ts'] >= since]
        assert len(bounded) == 2

    def test_files_outside_window_not_opened(self, tmp_path):
        _three_day_journal(tmp_path)
        since = datetime.datetime.combine(
            TODAY - datetime.timedelta(days=6),
            datetime.time.min).astimezone().timestamp()
        stats_all, stats_bounded = {}, {}
        js.load_trades(tmp_path, stats=stats_all)
        js.load_trades(tmp_path, since_ts=since, stats=stats_bounded)
        assert stats_all['files_read'] == 3      # no-bounds: opens all
        assert stats_bounded['files_read'] == 2  # day-20 file never opened


# ---------------------------------------------------------------------------
# 4) new buy-derived trade keys
# ---------------------------------------------------------------------------

class TestTradeDictNewKeys:
    def test_keys_carried_defaulted_and_unpaired(self, tmp_path):
        rows = [
            {'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 100,
             'entry_tactic': 'maker_ladder', 'maker': True, 'avg_corr': 0.42,
             'sizing': {'stack': 'v2', 'x': 1}, 'ts': _ts(TODAY, 9)},
            {'action': 'sell', 'symbol': 'BTC/USD', 'pnl_pct': 1.0,
             'exit_reason': 'tp', 'ts': _ts(TODAY, 10)},
            {'action': 'buy', 'symbol': 'ETH/USD', 'final_notional': 50,
             'ts': _ts(TODAY, 9, 30)},
            {'action': 'sell', 'symbol': 'ETH/USD', 'pnl_pct': -1.0,
             'exit_reason': 'stop', 'ts': _ts(TODAY, 11)},
            {'action': 'sell', 'symbol': 'NVDA', 'pnl_pct': 0.5,
             'exit_reason': 'signal_sell', 'ts': _ts(TODAY, 12)},
        ]
        _write_journal(tmp_path, TODAY, rows)
        trades = {t['symbol']: t for t in js.load_trades(tmp_path)}
        rich = trades['BTC/USD']
        assert rich['entry_tactic'] == 'maker_ladder'
        assert rich['maker'] is True
        assert rich['avg_corr'] == pytest.approx(0.42)
        assert rich['sizing_stack'] == 'v2'
        for t in (trades['ETH/USD'], trades['NVDA']):
            assert t['entry_tactic'] is None and t['maker'] is None
            assert t['avg_corr'] is None and t['sizing_stack'] is None


# ---------------------------------------------------------------------------
# 5) EOD digest builder
# ---------------------------------------------------------------------------

class TestEodDigest:
    def test_full_day_digest(self, tmp_path):
        rows = [
            {'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 1000,
             'ts': _ts(TODAY, 9)},
            {'action': 'sell', 'symbol': 'BTC/USD', 'pnl_pct': 2.0,
             'exit_reason': 'tp', 'ts': _ts(TODAY, 10)},
            {'action': 'buy', 'symbol': 'ETH/USD', 'final_notional': 500,
             'ts': _ts(TODAY, 10, 30)},   # still open — not a closed trade
            {'action': 'sell', 'symbol': 'NVDA', 'pnl_pct': -1.0,
             'exit_reason': 'stop', 'ts': _ts(TODAY, 11)},   # unpaired
            {'action': 'skip', 'symbol': 'SOL/USD', 'skip_reason': 'llm_veto',
             'ts': _ts(TODAY, 9, 5)},
            {'action': 'skip', 'symbol': 'DOGE/USD', 'skip_reason': 'llm_veto',
             'ts': _ts(TODAY, 9, 6)},
            {'action': 'skip', 'symbol': 'AMD', 'skip_reason': 'meta_veto',
             'ts': _ts(TODAY, 9, 7)},
        ]
        _write_journal(tmp_path, TODAY, rows)
        positions = [{'symbol': 'BTC/USD', 'unrealized_pl': 5.0},
                     {'symbol': 'NVDA', 'unrealized_pl': -2.5}]
        digest = js.build_eod_digest(tmp_path, positions=positions)
        assert digest.startswith(f'EOD digest {TODAY}')
        assert '2 buys, 2 sells, 3 skips' in digest
        assert 'realized $+20.00' in digest            # crypto: resolvable
        assert '~$' in digest                          # stock: partial
        assert 'BTC/USD +2.00%' in digest              # top win
        assert 'NVDA -1.00%' in digest                 # worst loss
        assert 'Unrealized: crypto $+5.00, stock $-2.50' in digest
        assert 'llm_veto=2' in digest and 'meta_veto=1' in digest

    def test_missing_dir_never_raises(self, tmp_path):
        digest = js.build_eod_digest(tmp_path / 'nope')
        assert isinstance(digest, str)
        assert digest.startswith('EOD digest')
        assert 'build failed' not in digest

    def test_positions_none_and_truncation(self, tmp_path):
        # Huge skip-reason strings make the top-4 line alone exceed the cap,
        # proving truncation actually fires (not just that output is small).
        rows = [{'action': 'skip', 'symbol': f'S{i}',
                 'skip_reason': f'reason{i}_' + 'x' * 1500,
                 'ts': _ts(TODAY, 9)} for i in range(6)]
        _write_journal(tmp_path, TODAY, rows)
        digest = js.build_eod_digest(tmp_path, positions=None)
        assert 'Unrealized: n/a' in digest
        assert len(digest) == 3500   # cap applied to an oversized digest


# ---------------------------------------------------------------------------
# 6) execution_report new blocks
# ---------------------------------------------------------------------------

class TestExecutionReportNewBlocks:
    @pytest.fixture(autouse=True)
    def _sandbox(self, monkeypatch, tmp_path):
        monkeypatch.setattr(er, 'JOURNAL_DIR', tmp_path)
        monkeypatch.setattr(er, 'BASE_DIR', tmp_path)
        self.dir = tmp_path

    def test_quote_age_buckets(self):
        rows = [
            {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 2.0,
             'quote_age_s': 1.0},
            {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 4.0,
             'quote_age_s': 0.5},
            {'action': 'buy', 'symbol': 'ETH/USD', 'slippage_bps': 6.0,
             'quote_age_s': 5},
            {'action': 'buy', 'symbol': 'NVDA', 'slippage_bps': 10.0,
             'quote_age_s': 60.0},
            {'action': 'buy', 'symbol': 'AMD', 'slippage_bps': 99.0},  # no age
        ]
        _write_journal(self.dir, TODAY, rows)
        report = er.run_report(0)
        by_age = report['entry_slippage_by_quote_age']
        assert by_age['lt2s'] == {'n': 2, 'mean_bps': 3.0}
        assert by_age['2_10s'] == {'n': 1, 'mean_bps': 6.0}
        assert by_age['gte10s'] == {'n': 1, 'mean_bps': 10.0}

    def test_crypto_maker_notional_share(self):
        rows = [
            {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 1.0,
             'final_notional': 100, 'entry_tactic': 'maker_fill'},
            {'action': 'buy', 'symbol': 'ETH/USD', 'slippage_bps': 1.0,
             'final_notional': 100, 'entry_tactic': 'taker'},
            # explicit per-rung maker_notional (T6 shape) overrides tactic
            {'action': 'buy', 'symbol': 'SOL/USD', 'slippage_bps': 1.0,
             'final_notional': 100, 'entry_tactic': 'taker',
             'maker_notional': 40},
            # stock buy: excluded from the crypto notional share
            {'action': 'buy', 'symbol': 'NVDA', 'slippage_bps': 1.0,
             'final_notional': 500, 'entry_tactic': 'maker_fill'},
        ]
        _write_journal(self.dir, TODAY, rows)
        report = er.run_report(0)
        assert report['crypto_maker_notional_share'] == round(140 / 300, 3)

    def test_llm_analysis_block(self):
        rows = [
            {'action': 'llm_analysis', 'dedup_hit': True, 'latency_ms': 100,
             'cost_usd': 0.01},
            {'action': 'llm_analysis', 'dedup_hit': False, 'latency_ms': 200,
             'cost_usd': 0.02},
            {'action': 'llm_analysis'},   # keyless row: block stays well-formed
            {'action': 'llm_backoff'},
            {'action': 'buy', 'symbol': 'BTC/USD', 'slippage_bps': 1.0},
        ]
        _write_journal(self.dir, TODAY, rows)
        report = er.run_report(0)
        la = report['llm_analysis']
        assert la['n_calls'] == 3
        assert la['dedup_hits'] == 1
        assert la['dedup_hit_rate'] == round(1 / 3, 3)
        assert la['mean_latency_ms'] == 150.0
        assert la['total_cost_usd'] == round(0.03, 4)
        assert la['n_backoffs'] == 1

    def test_llm_block_keyless_only_gives_none_costs(self):
        _write_journal(self.dir, TODAY, [{'action': 'llm_analysis'},
                                         {'action': 'llm_analysis'}])
        report = er.run_report(0)
        la = report['llm_analysis']
        assert la['n_calls'] == 2 and la['dedup_hits'] == 0
        assert la['mean_latency_ms'] is None
        assert la['total_cost_usd'] is None

    def test_legacy_keys_unchanged_for_pure_fills(self, capsys):
        rows = []
        for i in range(40):
            rows.append({'action': 'buy', 'symbol': 'BTC/USD',
                         'slippage_bps': 1.0,
                         'entry_tactic': 'maker' if i < 20 else 'taker'})
        _write_journal(self.dir, TODAY, rows)
        report = er.run_report(0)
        assert report['crypto/buy/entry']['n'] == 40
        assert report['crypto/buy/entry']['mean_bps'] == 1.0
        assert report['overall_mean_bps'] == 1.0
        assert report['crypto_maker_share'] == 0.5
        out = capsys.readouterr().out
        assert 'IMPLEMENTATION SHORTFALL' in out
        assert 'Crypto maker share (entries): 50% of 40' in out

    def test_llm_only_window_no_crash(self):
        _write_journal(self.dir, TODAY, [{'action': 'llm_analysis'},
                                         {'action': 'llm_backoff'}])
        report = er.run_report(0)
        assert 'overall_mean_bps' not in report
        assert not any(k.count('/') == 2 for k in report)   # no shortfall rows
        assert report['llm_analysis']['n_calls'] == 1
        assert (self.dir / 'execution_report.json').exists()


# ---------------------------------------------------------------------------
# 7) shadow promotion ledger
# ---------------------------------------------------------------------------

def _discard_report():
    return {'age_days': 30.0, 'n': 250, 'p': 0.5, 'mean_d': -0.01,
            'dm': 0.1, 'hit_champ': 0.5, 'hit_chall': 0.45}


def _wire_discard(monkeypatch, tmp_path, report=None):
    monkeypatch.setattr(sh, 'BASE_DIR', tmp_path)
    monkeypatch.setattr(sh, 'DM_V2_ENABLED', False)
    (tmp_path / 'stock_challenger_model_v2.manifest.json').write_text(
        json.dumps({'saved_at': '2026-08-01', 'score': 0.5}))
    (tmp_path / 'stock_model_v2.manifest.json').write_text(
        json.dumps({'saved_at': '2026-07-01', 'score': 0.4,
                    'promoted_from_shadow': True}))
    (tmp_path / 'stock_challenger_policy_gate.json').write_text(
        json.dumps({'passed': True, 'sharpe': 1.2, 'dsr': 0.7,
                    'n_trades': 50, 'challenger_manifest_mtime': 123}))
    rep = report or _discard_report()
    monkeypatch.setattr(sh, 'evaluate_shadow', lambda prefix, api=None: rep)
    monkeypatch.setattr(sh, '_discard_challenger', lambda prefix: None)
    monkeypatch.setattr(sh, '_notify', lambda msg: None)
    monkeypatch.setattr(sh, '_write_shadow_status', lambda *a, **k: None)


class TestPromotionLedger:
    def test_append_writes_one_json_line_and_appends(self, monkeypatch,
                                                     tmp_path):
        monkeypatch.setattr(sh, 'BASE_DIR', tmp_path)
        fps = {'champion': {'mtime': 1, 'score': 0.4},
               'challenger': {'mtime': 2, 'score': 0.5}}
        sh._append_promotion_ledger(
            'stock', 'stock', 'discarded', _discard_report(),
            gate_hold=None, fingerprints=fps, policy_gate={'passed': True})
        ledger = tmp_path / 'stock_promotion_ledger.jsonl'
        lines = ledger.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row['prefix'] == 'stock' and row['decision'] == 'discarded'
        assert row['p'] == 0.5 and row['mean_d'] == -0.01
        assert row['dm_v2_enabled'] in (True, False)
        assert row['gate_hold'] is None
        assert row['champion_manifest'] == {'mtime': 1, 'score': 0.4}
        assert row['challenger_manifest'] == {'mtime': 2, 'score': 0.5}
        assert row['policy_gate'] == {'passed': True}
        assert 'ts' in row
        sh._append_promotion_ledger(
            'stock', 'stock', 'held', _discard_report(),
            gate_hold='why', fingerprints=None, policy_gate=None)
        assert len(ledger.read_text().splitlines()) == 2   # append, not truncate

    def test_manifest_fingerprint(self, tmp_path):
        man = tmp_path / 'm.json'
        man.write_text(json.dumps({'saved_at': '2026-08-01', 'score': 1.5}))
        fp = sh._manifest_fingerprint(man)
        assert isinstance(fp['mtime'], int)
        assert fp['saved_at'] == '2026-08-01' and fp['score'] == 1.5
        assert sh._manifest_fingerprint(tmp_path / 'missing.json') is None

    def test_discard_path_writes_ledger_with_pre_discard_fingerprint(
            self, monkeypatch, tmp_path):
        _wire_discard(monkeypatch, tmp_path)
        report = sh.evaluate_and_maybe_promote('stock', 'stock', api=object())
        assert report['decision'] == 'discarded'
        ledger = tmp_path / 'stock_promotion_ledger.jsonl'
        lines = ledger.read_text().splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row['decision'] == 'discarded'
        # captured BEFORE _discard_challenger would delete the manifest
        assert row['challenger_manifest'] is not None
        assert row['challenger_manifest']['saved_at'] == '2026-08-01'
        assert row['champion_manifest']['promoted_from_shadow'] is True
        assert row['policy_gate']['passed'] is True
        assert row['policy_gate']['challenger_manifest_mtime'] == 123

    def test_continue_cycle_writes_nothing(self, monkeypatch, tmp_path):
        _wire_discard(monkeypatch, tmp_path,
                      report={'age_days': 5.0, 'n': 50, 'p': 0.5,
                              'mean_d': 0.0, 'dm': 0.0,
                              'hit_champ': 0.5, 'hit_chall': 0.5})
        report = sh.evaluate_and_maybe_promote('stock', 'stock', api=object())
        assert report['decision'] == 'continue'
        assert not (tmp_path / 'stock_promotion_ledger.jsonl').exists()

    def test_ledger_append_failure_never_raises(self, monkeypatch, tmp_path):
        _wire_discard(monkeypatch, tmp_path)
        monkeypatch.setattr(
            sh, 'promotion_ledger_file',
            lambda prefix: tmp_path / 'no_such_dir' / 'ledger.jsonl')
        report = sh.evaluate_and_maybe_promote('stock', 'stock', api=object())
        assert report['decision'] == 'discarded'   # flow completed anyway


# ---------------------------------------------------------------------------
# 8) run_bots ops thread
# ---------------------------------------------------------------------------

def _notify_stub(commands, calls):
    stub = types.ModuleType('notify')
    stub.poll_telegram_commands = (
        commands if callable(commands) else (lambda: list(commands)))
    stub.set_halt = lambda reason: calls.setdefault('halt', []).append(reason)
    stub.clear_halt = lambda: calls.setdefault('clear', []).append(True)
    stub.halt_active = lambda: False
    stub.request_flatten = (
        lambda reason: calls.setdefault('flatten', []).append(reason))
    stub.notify = (
        lambda msg, **kw: calls.setdefault('notify', []).append((msg, kw)))
    return stub


class TestRunBotsOps:
    def test_pipeline_alive(self, monkeypatch, tmp_path):
        status = tmp_path / 'pipeline_status.json'
        monkeypatch.setattr(rb, '_PIPELINE_STATUS', str(status))
        assert rb._pipeline_alive() is False        # missing
        status.write_text('{}')
        assert rb._pipeline_alive() is True         # fresh
        old = datetime.datetime.now().timestamp() - 300
        os.utime(status, (old, old))
        assert rb._pipeline_alive() is False        # stale

    def test_handle_commands_halt_and_flatten(self, monkeypatch):
        calls = {}
        monkeypatch.setitem(sys.modules, 'notify',
                            _notify_stub(['/halt'], calls))
        rb._ops_handle_commands([])
        assert calls['halt'] == ['telegram /halt']
        assert len(calls['notify']) == 1
        calls2 = {}
        monkeypatch.setitem(sys.modules, 'notify',
                            _notify_stub(['/flatten'], calls2))
        rb._ops_handle_commands([])
        assert calls2['flatten'] == ['telegram /flatten']

    def test_ops_cycle_crash_isolated_and_defers(self, monkeypatch):
        def boom():
            raise RuntimeError('telegram down')
        calls = {}
        monkeypatch.setitem(sys.modules, 'notify', _notify_stub(boom, calls))
        monkeypatch.setattr(rb, '_pipeline_alive', lambda: False)
        monkeypatch.setattr(rb, '_ops_daily_drift', lambda threads: None)
        monkeypatch.setattr(rb, '_ops_daily_rotation', lambda: None)
        rb._ops_cycle([])   # must not raise
        # pipeline alive -> poll never touched
        polled = []
        calls3 = {}
        monkeypatch.setitem(
            sys.modules, 'notify',
            _notify_stub(lambda: polled.append(1) or [], calls3))
        monkeypatch.setattr(rb, '_pipeline_alive', lambda: True)
        rb._ops_cycle([])
        assert polled == []

    def test_daily_guards_once_per_date(self, monkeypatch):
        drift_calls = []
        md_stub = types.ModuleType('monitor_drift')
        md_stub.run_check = (
            lambda prefix, label: drift_calls.append((prefix, label)))
        monkeypatch.setitem(sys.modules, 'monitor_drift', md_stub)
        monkeypatch.setattr(rb, '_ops_drift_date', None)
        rb._ops_daily_drift([])
        rb._ops_daily_drift([])
        assert drift_calls == [('', 'crypto'), ('stock', 'stock')]  # once

        rot_calls = []
        monkeypatch.setattr(tj, 'rotate_old_journals',
                            lambda now=None: rot_calls.append(1) or 0)
        monkeypatch.setattr(rb, '_ops_rotate_date', None)
        rb._ops_daily_rotation()
        rb._ops_daily_rotation()
        assert rot_calls == [1]                                     # once


# ---------------------------------------------------------------------------
# 9) options verdict runner
# ---------------------------------------------------------------------------

class TestOptionsVerdictRunner:
    def test_realized_vol_annual(self):
        import numpy as np
        assert oo.realized_vol_annual(np.full(100, 50.0)) == pytest.approx(0.0)
        rng = np.random.default_rng(0)
        sigma_ann = 0.30
        rets = rng.normal(0.0, sigma_ann / np.sqrt(252), 500)
        closes = 100.0 * np.exp(np.cumsum(rets))
        rv = oo.realized_vol_annual(closes)
        assert rv == pytest.approx(sigma_ann, abs=0.04)
        assert oo.realized_vol_annual(closes[:15]) is None   # < 20 returns

    def test_run_verdict_synthetic(self):
        import numpy as np
        rng = np.random.default_rng(1)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 200)))
        opens = closes * (1.0 + rng.normal(0.0, 0.005, 200))
        verdict = oo.run_verdict({'NVDA': (opens, closes)})
        v = verdict['per_name']['NVDA']
        assert v['verdict'] in ('GO', 'NO_GO', 'NO_VIABLE_STRIKES',
                                'INSUFFICIENT_DATA')
        assert v.get('rv_annual') is not None
        assert verdict['pre_registered_expectation'] == 'NO_GO'
        assert verdict['params']['dte'] == 1
        assert verdict['params']['min_multiple'] == oo.MIN_EDGE_MULTIPLE
        counts = (verdict['n_go'], verdict['n_no_go'],
                  verdict['n_insufficient'])
        assert sum(counts) == 1
        expected = ('GO' if counts[0] else
                    ('NO_GO' if counts[1] else 'INSUFFICIENT_DATA'))
        assert verdict['overall_verdict'] == expected
        assert isinstance(json.loads(json.dumps(verdict)), dict)  # JSON-safe

    def test_run_verdict_insufficient(self):
        import numpy as np
        closes = np.linspace(100, 101, 10)
        opens = closes.copy()
        verdict = oo.run_verdict({'ZZZZ': (opens, closes)})
        assert verdict['per_name']['ZZZZ']['verdict'] == 'INSUFFICIENT_DATA'
        assert verdict['n_insufficient'] == 1
        assert verdict['overall_verdict'] == 'INSUFFICIENT_DATA'


# ---------------------------------------------------------------------------
# 10) run_pipeline EOD digest hook
# ---------------------------------------------------------------------------

class TestPipelineDigestHook:
    @pytest.fixture(autouse=True)
    def _sandbox(self, monkeypatch):
        monkeypatch.setattr(rp, 'EOD_DIGEST_ENABLED', True)
        monkeypatch.setattr(rp, 'EOD_DIGEST_HOUR', 0)
        monkeypatch.setattr(rp, '_last_digest_date', None)
        # positions path: trading_utils import fails -> positions=None
        monkeypatch.setitem(sys.modules, 'trading_utils', None)
        self.notified = []
        stub = types.ModuleType('notify')
        stub.notify = (
            lambda msg, **kw: self.notified.append((msg, kw)))
        monkeypatch.setitem(sys.modules, 'notify', stub)

    def test_sends_once_per_day(self, monkeypatch):
        monkeypatch.setattr(js, 'build_eod_digest',
                            lambda jd, positions=None: 'DIGEST-TEXT')
        log = io.StringIO()
        rp._maybe_send_eod_digest(log)
        assert len(self.notified) == 1
        msg, kw = self.notified[0]
        assert msg == 'DIGEST-TEXT'
        assert kw['level'] == 'info'
        today = datetime.date.today().isoformat()
        assert kw['dedupe_key'] == f'eod-digest-{today}'
        assert '[DIGEST] sent' in log.getvalue()
        rp._maybe_send_eod_digest(log)          # same day: guard holds
        assert len(self.notified) == 1

    def test_disabled_and_failing_build_swallowed(self, monkeypatch):
        monkeypatch.setattr(rp, 'EOD_DIGEST_ENABLED', False)
        rp._maybe_send_eod_digest(io.StringIO())
        assert self.notified == []

        monkeypatch.setattr(rp, 'EOD_DIGEST_ENABLED', True)
        monkeypatch.setattr(rp, '_last_digest_date', None)

        def boom(jd, positions=None):
            raise RuntimeError('journal on fire')
        monkeypatch.setattr(js, 'build_eod_digest', boom)
        log = io.StringIO()
        rp._maybe_send_eod_digest(log)          # must not raise
        assert self.notified == []
        assert '[DIGEST] failed' in log.getvalue()
