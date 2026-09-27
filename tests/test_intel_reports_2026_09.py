"""INTEL W7 (2026-09-27): `--out DIR` for decision_report / execution_report,
and scripts/lexicon_fix_impact.py (G4-05 + B13 impact instrument).

Mac-safe: stdlib sqlite3 + the pure sentiment module; journals and the API are
faked, the sentiment DB is a synthetic temp file. No network.
"""
import datetime as dt
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

import decision_report
import execution_report

REPO = Path(__file__).resolve().parent.parent


# ===========================================================================
# 1. decision_report --out
# ===========================================================================

def _dr_journal(tmp_path, monkeypatch, rows):
    jd = tmp_path / 'journals'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(decision_report, 'JOURNAL_DIR', jd)
    monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)


def _dr_stub_api(monkeypatch, ok=True):
    monkeypatch.setitem(sys.modules, 'dotenv',
                        types.SimpleNamespace(load_dotenv=lambda *a, **k: None))

    def get_api():
        if not ok:
            raise RuntimeError('no keys')
        return object()
    monkeypatch.setitem(sys.modules, 'trading_utils',
                        types.SimpleNamespace(get_api=get_api))


def _dr_stub_sections(monkeypatch):
    for name in ('gate_attribution', 'signal_exit_audit',
                 'conviction_calibration'):
        monkeypatch.setattr(decision_report, name, lambda *a, **k: {})


_SKIP_ROW = {'action': 'skip', 'symbol': 'AAA', 'skip_reason': 'x',
             'ts': dt.datetime.now(dt.timezone.utc).isoformat()}


def test_decision_report_default_path_is_repo_root_literal(tmp_path, monkeypatch):
    monkeypatch.setattr(decision_report, 'BASE_DIR', tmp_path)
    assert decision_report._report_path() == tmp_path / 'decision_report.json'
    assert decision_report._report_path(None) == tmp_path / 'decision_report.json'
    # the real default is the path gui.py reads (gui.py:4330)
    monkeypatch.undo()
    assert decision_report._report_path() == REPO / 'decision_report.json'


def test_decision_report_default_normal_write_unchanged(tmp_path, monkeypatch):
    _dr_journal(tmp_path, monkeypatch, [_SKIP_ROW])
    _dr_stub_api(monkeypatch)
    _dr_stub_sections(monkeypatch)
    rep = decision_report.run_report(days=14)
    root = tmp_path / 'decision_report.json'
    assert json.loads(root.read_text()) == rep
    assert not rep.get('stale', False) or 'stale_reason' in rep


def test_decision_report_out_redirects_normal_write(tmp_path, monkeypatch):
    _dr_journal(tmp_path, monkeypatch, [_SKIP_ROW])
    _dr_stub_api(monkeypatch)
    _dr_stub_sections(monkeypatch)
    out = tmp_path / 'ev' / 'nested'
    rep = decision_report.run_report(days=14, out_dir=out)
    assert json.loads((out / 'decision_report.json').read_text()) == rep
    assert 'api_available' not in rep or rep['api_available'] is not False
    assert not (tmp_path / 'decision_report.json').exists()


@pytest.mark.parametrize('journal,api_ok,expect_api', [
    ([], True, None),              # no journal rows  -> stale (api None)
    ([_SKIP_ROW], False, False),   # get_api() raises -> stale (api False)
])
def test_decision_report_out_redirects_stale_write(tmp_path, monkeypatch,
                                                   journal, api_ok, expect_api):
    _dr_journal(tmp_path, monkeypatch, journal)
    _dr_stub_api(monkeypatch, ok=api_ok)
    out = tmp_path / 'ev'
    rep = decision_report.run_report(days=3, out_dir=out)
    assert rep['stale'] is True and rep['api_available'] is expect_api
    assert json.loads((out / 'decision_report.json').read_text()) == rep
    assert not (tmp_path / 'decision_report.json').exists()


def test_decision_report_default_stale_write_unchanged(tmp_path, monkeypatch):
    _dr_journal(tmp_path, monkeypatch, [])
    rep = decision_report.run_report(days=3)
    assert json.loads((tmp_path / 'decision_report.json').read_text()) == rep
    assert decision_report._write_stale_report(1, None)['stale'] is True


# ===========================================================================
# 2. execution_report --out
# ===========================================================================

def _er_journal(tmp_path, monkeypatch, rows):
    jd = tmp_path / 'journals'
    jd.mkdir()
    with open(jd / f'{dt.date.today().isoformat()}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    monkeypatch.setattr(execution_report, 'JOURNAL_DIR', jd)
    monkeypatch.setattr(execution_report, 'BASE_DIR', tmp_path)


_BUY = {'action': 'buy', 'symbol': 'BTC/USD', 'final_notional': 1000.0,
        'entry_tactic': 'taker'}


@pytest.mark.parametrize('rows', [[], [_BUY]], ids=['empty', 'nonempty'])
def test_execution_report_default_path_unchanged(tmp_path, monkeypatch, rows):
    _er_journal(tmp_path, monkeypatch, rows)
    rep = execution_report.run_report(days=1)
    assert json.loads((tmp_path / 'execution_report.json').read_text()) == rep


@pytest.mark.parametrize('rows', [[], [_BUY]], ids=['empty', 'nonempty'])
def test_execution_report_out_redirects(tmp_path, monkeypatch, rows):
    _er_journal(tmp_path, monkeypatch, rows)
    out = tmp_path / 'ev' / 'x'
    rep = execution_report.run_report(days=1, out_dir=out)
    assert json.loads((out / 'execution_report.json').read_text()) == rep
    assert not (tmp_path / 'execution_report.json').exists()


@pytest.mark.parametrize('script', ['decision_report.py', 'execution_report.py'])
def test_cli_accepts_out(script):
    r = subprocess.run([sys.executable, str(REPO / script), '--help'],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert '--out DIR' in r.stdout


# ===========================================================================
# 3. scripts/lexicon_fix_impact.py
# ===========================================================================

def _load_lfi():
    spec = importlib.util.spec_from_file_location(
        'lexicon_fix_impact', str(REPO / 'scripts' / 'lexicon_fix_impact.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lfi = _load_lfi()

CURLY = 'Company doesn’t beat expectations'          # G4-05
PHRASE = 'Fed signals rate cut in September'               # B13
NEG_PHRASE = ('Bitcoin recovery is not happening anytime soon '
              'says veteran trader')                       # inner negator kept
PLAIN = 'Bitcoin surges to all time high'


def _mk_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("""CREATE TABLE articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        date TEXT NOT NULL, headline TEXT NOT NULL, summary TEXT DEFAULT '',
        url TEXT DEFAULT '', keyword_score REAL NOT NULL, llm_score REAL,
        fetched_at TEXT NOT NULL, llm_scored_at TEXT)""")
    for sym, date, head, summ, llm in rows:
        import sentiment_history as sh
        con.execute("INSERT INTO articles (symbol, date, headline, summary, "
                    "keyword_score, llm_score, fetched_at) VALUES (?,?,?,?,?,?,?)",
                    (sym, date, head, summ, sh._keyword_score(head, summ), llm,
                     '2026-09-26T00:00:00'))
    con.commit()
    con.close()


_ROWS = [
    ('AAA', '2026-09-20', CURLY, '', None),
    ('AAA', '2026-09-20', PLAIN, '', None),
    ('BBB', '2026-09-21', PHRASE, '', None),
    ('CCC', '2026-09-22', NEG_PHRASE, '', None),
    ('DDD', '2026-09-22', PHRASE, '', 0.4),            # mixed cell
    ('DDD', '2026-09-22', PLAIN, '', None),
    ('EEE', '2024-01-02', PHRASE, '', None),            # outside --days 365
]


def test_scorer_values_match_w5_examples():
    assert lfi.score_a(PHRASE) == pytest.approx(0.0, abs=1e-12)
    assert lfi.score_c(PHRASE) == pytest.approx(0.475, abs=5e-4)
    assert lfi.score_a(CURLY) == pytest.approx(0.888, abs=5e-4)
    assert lfi.score_b(CURLY) < -0.05          # negator now seen -> sign flip
    assert lfi.score_c(CURLY) == pytest.approx(-0.581, abs=5e-4)
    assert lfi.score_c(NEG_PHRASE) == pytest.approx(lfi.score_a(NEG_PHRASE))
    assert lfi.score_b(PHRASE) == lfi.score_a(PHRASE)   # no curly -> identical


def test_unmasked_copy_equals_live_scorer_on_runner_cases():
    import sentiment
    spec = importlib.util.spec_from_file_location(
        '_tsh', str(REPO / 'tests' / 'test_sentiment_headlines.py'))
    tsh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsh)
    texts = [h for h, _e, _c in tsh.ALL_TESTS] + [CURLY, PHRASE, NEG_PHRASE]
    for t in texts:
        assert lfi.score_mask2(t, apos=False, mask=False) == sentiment._score_text(t)
        assert lfi.score_mask2(t, apos=True, mask=False) == lfi.score_b(t)


def test_keyword_and_aggregate_mirror_sentiment_history(tmp_path):
    import sentiment_history as sh
    for h, s in [(CURLY, PHRASE), (PLAIN, ''), ('', PHRASE), ('x', '')]:
        assert lfi.keyword_score(h, s, lfi.score_a) == sh._keyword_score(h, s)
    con = sqlite3.connect(':memory:')
    con.execute("CREATE TABLE articles (symbol TEXT, date TEXT, "
                "keyword_score REAL, llm_score REAL)")
    con.execute("CREATE TABLE daily_sentiment (symbol TEXT, date TEXT, score REAL, "
                "article_count INTEGER, llm_count INTEGER, score_type TEXT, "
                "PRIMARY KEY (symbol, date))")
    cases = {'K': [(0.3, None), (-0.1, None)], 'L': [(0.3, 0.5), (0.1, -0.2)],
             'M': [(0.3, 0.5), (0.1, None), (-0.4, None)]}
    for sym, arts in cases.items():
        con.executemany("INSERT INTO articles VALUES (?,?,?,?)",
                        [(sym, '2026-01-01', k, l) for k, l in arts])
        sh._aggregate_daily(con, sym, '2026-01-01')
        want = con.execute("SELECT score FROM daily_sentiment WHERE symbol=?",
                           (sym,)).fetchone()[0]
        got = lfi.aggregate_cell([k for k, _ in arts], [l for _, l in arts])
        assert got == pytest.approx(want, abs=1e-15)


def test_synthetic_db_run(tmp_path, capsys):
    db = tmp_path / 'cache.db'
    _mk_db(db, _ROWS)
    digest = hashlib.sha256(db.read_bytes()).hexdigest()
    js = tmp_path / 'o' / 'r.json'
    assert lfi.main(['--db', str(db), '--json', str(js), '--skip-runner']) == 0
    res = json.loads(js.read_text())
    assert hashlib.sha256(db.read_bytes()).hexdigest() == digest   # read-only
    assert res['n_articles'] == 7
    assert res['stored_keyword_score_ne_current'] == 0
    rb, rc = res['rows']['b_apostrophe'], res['rows']['c_apos_mask2']
    assert rb['n_changed'] == 1 and rb['sign_flips'] == 1          # CURLY only
    # c: CURLY + three PHRASE rows (0 -> +0.475 is a change, not a flip)
    assert rc['n_changed'] == 4 and rc['sign_flips'] == 1
    d = res['daily']
    # window drops EEE (2024) -> cells AAA/BBB/CCC/DDD
    assert d['b_apostrophe']['n_cells'] == 4
    assert d['c_apos_mask2']['n_cells'] == 4
    # BBB baseline cell is 0.0 (the B13 cancel) -> not a non-zero ticker-day
    assert d['c_apos_mask2']['n_nonzero'] == 3
    assert d['b_apostrophe']['nonzero_n_gt_0.05'] == 1             # AAA
    assert d['c_apos_mask2']['nonzero_n_gt_0.05'] == 2             # AAA + DDD
    assert d['c_apos_mask2']['n_gt_0.05'] == 3                     # + BBB
    assert res['verdict'] == 'BUNDLE'
    assert 'VERDICT: BUNDLE' in capsys.readouterr().out


def test_verdict_branches():
    def dd(fb, fc):
        return {'b_apostrophe': {'nonzero_frac_gt_0.05': fb},
                'c_apos_mask2': {'nonzero_frac_gt_0.05': fc}}
    assert lfi.verdict(dd(0.0, 0.0)) == ('correctness-only', [])
    assert lfi.verdict(dd(0.0099, 0.0099))[0] == 'correctness-only'
    assert lfi.verdict(dd(0.01, 0.0)) == ('BUNDLE', ['b_apostrophe'])
    assert lfi.verdict(dd(0.0, 0.0272)) == ('BUNDLE', ['c_apos_mask2'])


def test_no_change_db_is_correctness_only(tmp_path):
    db = tmp_path / 'c.db'
    _mk_db(db, [('AAA', '2026-09-20', PLAIN, '', None)])
    js = tmp_path / 'r.json'
    assert lfi.main(['--db', str(db), '--json', str(js), '--skip-runner']) == 0
    assert json.loads(js.read_text())['verdict'] == 'correctness-only'


def test_missing_db_exits_2(tmp_path, capsys):
    missing = tmp_path / 'nope.db'
    assert lfi.main(['--db', str(missing), '--skip-runner']) == 2
    assert not missing.exists()                 # never created by the ro open
    assert 'cannot read' in capsys.readouterr().err


def test_db_without_articles_table_exits_2(tmp_path):
    db = tmp_path / 'empty.db'
    sqlite3.connect(str(db)).close()
    assert lfi.main(['--db', str(db), '--skip-runner']) == 2


def test_runner_scores_counts():
    r = lfi.runner_scores()
    assert r['validate_text']['pass'] == r['validate_text']['total']
    for k in lfi.VARIANTS:
        assert r[k]['total'] == r['a_current']['total']
        assert r[k]['runner_equiv_total'] == (r[k]['total']
                                              + r['validate_text']['total'] + 4)
