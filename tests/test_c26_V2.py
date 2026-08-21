"""Packet V2 tests — learned sentiment lexicon (learned_lexicon.py +
scripts/train_lexicon.py).

Pure numpy/pandas/scipy/sqlite3 + stdlib; everything under tmp_path; no test
touches the real sentiment_cache.db, journals/, or repo-root artifact paths.
"""

import collections
import datetime
import importlib.util
import json
import py_compile
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import learned_lexicon as ll  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic corpus with a planted term (and a placebo)
# ---------------------------------------------------------------------------

# >= 40 filler words, none in STOPWORDS, none containing a corpus symbol.
FILLER = [
    'quantum', 'widget', 'launch', 'pivot', 'margin', 'growth', 'signal',
    'robust', 'sector', 'device', 'energy', 'metal', 'mining', 'miner',
    'tablet', 'driver', 'cloud', 'model', 'chips', 'sales', 'deal',
    'merger', 'upgrade', 'target', 'profit', 'revenue', 'guidance',
    'outlook', 'demand', 'supply', 'factory', 'pilot', 'orbit', 'beacon',
    'matrix', 'vector', 'kernel', 'packet', 'circuit', 'sensor', 'router',
    'modem', 'panel', 'engine',
]


def make_synth(seed=0):
    """2 symbols x ~220 business days, 1-2 articles per symbol-day.

    Plants 'zephyrium' in ~25%% of symbol-days: each planted (sym, D) adds
    +0.008 to that symbol's daily return on D+1..D+3. Placebo 'quorvex' in
    an independent ~25%% with no return effect. prices tidy
    [symbol, date, open, close] with open = prior close, so the open-entry
    PIT mode is exercised (entry = open[D+1] = close[D]).
    """
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range('2024-01-02', periods=220).date)
    syms = ['AAA', 'BBB']
    art_rows, planted = [], {}
    for sym in syms:
        for d in days:
            pl = bool(rng.random() < 0.25)
            pb = bool(rng.random() < 0.25)
            planted[(sym, d)] = pl
            n_art = 1 + int(rng.random() < 0.5)
            for a in range(n_art):
                k = int(rng.integers(6, 11))
                words = [str(w) for w in rng.choice(FILLER, size=k)]
                if a == 0 and pl:
                    words[2] = 'zephyrium'
                if a == 0 and pb:
                    words[4] = 'quorvex'
                llm = None
                if rng.random() < 0.5:
                    llm = float(np.clip((0.5 if pl else 0.0)
                                        + rng.normal(0.0, 0.1), -1.0, 1.0))
                art_rows.append({
                    'symbol': sym, 'date': d.isoformat(),
                    'headline': ' '.join(words), 'summary': '',
                    'keyword_score': float(rng.normal(0.0, 0.05)),
                    'llm_score': llm,
                })
    price_rows = []
    for sym in syms:
        rets = rng.normal(0.0, 0.006, size=len(days))
        for di, d in enumerate(days):
            if planted[(sym, d)]:
                for j in range(di + 1, min(di + 4, len(days))):
                    rets[j] += 0.008
        close = 100.0 * np.cumprod(1.0 + rets)
        for di, d in enumerate(days):
            price_rows.append({
                'symbol': sym, 'date': d,
                'open': float(close[di - 1]) if di > 0 else 100.0,
                'close': float(close[di]),
            })
    return (pd.DataFrame(art_rows), pd.DataFrame(price_rows), planted)


@pytest.fixture(scope='module')
def synth():
    return make_synth(0)


@pytest.fixture(scope='module')
def trained(synth):
    arts, prices, _ = synth
    return ll.train_lexicon(arts, prices, horizons=(1, 3, 5),
                            fit_horizon=3, bootstrap_b=100, seed=0)


# ---------------------------------------------------------------------------
# T1 tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_rules(self):
        text = "AAA Signals <b>Rate</b> Cut in 2024; 7 analysts, Reuters says"
        out = ll.tokenize(text, blacklist=frozenset({'aaa'}))
        assert out[:4] == ['signals', 'rate', 'cut', 'analysts']
        assert 'rate_cut' in out                      # surviving-seq bigram
        assert 'signals_rate' in out
        assert not any('2024' in t or t == '7' for t in out)  # leading letter
        assert 'aaa' not in out and 'in' not in out and 'says' not in out
        assert 'b' not in out                          # HTML tag stripped

    def test_deterministic(self):
        t = 'Merger deal boosts chipmaker outlook sharply'
        assert ll.tokenize(t) == ll.tokenize(t)

    def test_empty(self):
        assert ll.tokenize('') == []
        assert ll.tokenize(None) == []


# ---------------------------------------------------------------------------
# T2 build_vocab
# ---------------------------------------------------------------------------

class TestVocab:
    def test_min_df_floor_and_postings(self):
        docs = pd.DataFrame({'tokens': [
            collections.Counter({'aa': 2, 'bb': 1}),
            collections.Counter({'aa': 1}),
            collections.Counter({'aa': 1, 'cc': 1}),
            collections.Counter({'bb': 1}),
        ]})
        terms, postings = ll.build_vocab(docs, min_df=3)
        assert terms == ['aa']                     # df('bb')=2 < 3 dropped
        assert list(postings['aa']) == [0, 1, 2]
        terms2, _ = ll.build_vocab(docs, min_df=2)
        assert 'bb' in terms2 and 'cc' not in terms2


# ---------------------------------------------------------------------------
# T3 load_articles + schema fidelity (exact _SCHEMA from sentiment_history)
# ---------------------------------------------------------------------------

# Pasted verbatim from sentiment_history.py (lines 35-73). Do NOT import
# sentiment_history here (it runs load_dotenv at import); pasting keeps the
# test hermetic and pins schema drift via the source-text assertion below.
SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT DEFAULT '',
    url TEXT DEFAULT '',
    keyword_score REAL NOT NULL,
    llm_score REAL,
    fetched_at TEXT NOT NULL,
    llm_scored_at TEXT,
    UNIQUE(symbol, date, headline)
);

CREATE TABLE IF NOT EXISTS daily_sentiment (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    score REAL NOT NULL,
    article_count INTEGER NOT NULL,
    llm_count INTEGER DEFAULT 0,
    score_type TEXT DEFAULT 'keyword',
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS fng_daily (
    date TEXT PRIMARY KEY,
    value INTEGER NOT NULL,
    score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_unscored ON articles(llm_score) WHERE llm_score IS NULL;
CREATE INDEX IF NOT EXISTS idx_symbol_date ON articles(symbol, date);
"""

_INS = ("INSERT OR IGNORE INTO articles (symbol, date, headline, summary, "
        "url, keyword_score, llm_score, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")


class TestLoadArticles:
    def test_schema_source_still_pinned(self):
        src = (REPO / 'sentiment_history.py').read_text()
        assert 'UNIQUE(symbol, date, headline)' in src

    def test_roundtrip_dedupe_null_llm(self, tmp_path):
        db = tmp_path / 'cache.db'
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA)
        row = ('NVDA', '2024-01-02', 'Chip demand surges', 's', 'u',
               0.4, None, '2024-01-02T00:00:00')
        conn.execute(_INS, row)
        conn.execute(_INS, row)   # duplicate (symbol,date,headline)
        conn.execute(_INS, ('NVDA', '2024-01-02', 'Second headline', 's',
                            'u', 0.1, 0.5, '2024-01-02T00:00:00'))
        conn.execute(_INS, ('NVDA', '2024-01-03', 'Solo headline', 's',
                            'u', 0.2, None, '2024-01-03T00:00:00'))
        conn.commit()
        n = conn.execute('SELECT COUNT(*) FROM articles').fetchone()[0]
        conn.close()
        assert n == 3                            # dedupe held
        arts = ll.load_articles(str(db))
        assert len(arts) == 3
        assert list(arts.columns) == ll._ARTICLE_COLS
        first = arts[arts['headline'] == 'Chip demand surges'].iloc[0]
        assert pd.isna(first['llm_score'])       # NULL round-trips to NaN
        docs = ll.build_docs(arts)
        d2 = docs[docs['date'] == datetime.date(2024, 1, 2)].iloc[0]
        assert d2['n_articles'] == 2 and d2['llm_n'] == 1
        assert d2['llm_doc'] == pytest.approx(0.5)
        d3 = docs[docs['date'] == datetime.date(2024, 1, 3)].iloc[0]
        assert np.isnan(d3['llm_doc'])           # all-NULL llm -> NaN
        # date filter
        late = ll.load_articles(str(db), start=datetime.date(2024, 1, 3))
        assert set(late['date']) == {'2024-01-03'}

    def test_missing_db_fails_open(self, tmp_path):
        arts = ll.load_articles(str(tmp_path / 'nope.db'))
        assert len(arts) == 0
        assert list(arts.columns) == ll._ARTICLE_COLS
        # mode=ro pinned: a plain connect would CREATE the file; read-only
        # must not (the trainer can never write the live cache).
        assert not (tmp_path / 'nope.db').exists()


# ---------------------------------------------------------------------------
# T4 strict PIT of attach_forward_returns
# ---------------------------------------------------------------------------

class TestPIT:
    def _prices(self):
        dates = [datetime.date(2024, 1, d) for d in (1, 2, 3, 4, 5, 8, 9, 10)]
        closes = [100.0, 101.0, 999.0, 102.0, 104.0, 106.0, 107.0, 108.0]
        return pd.DataFrame({'symbol': 'XYZ', 'date': dates,
                             'close': closes}), dates, closes

    def _docs(self, pub_dates):
        return pd.DataFrame({
            'symbol': ['XYZ'] * len(pub_dates), 'date': pub_dates,
            'tokens': [collections.Counter()] * len(pub_dates),
        })

    def test_entry_strictly_after_pub(self):
        prices, dates, closes = self._prices()
        docs = ll.attach_forward_returns(
            self._docs([datetime.date(2024, 1, 3)]), prices, (1,))
        assert docs.at[0, 'entry_date'] == datetime.date(2024, 1, 4)
        assert docs.at[0, 'entry_date'] > docs.at[0, 'date']
        # close-mode: entry close[01-04], exit close[01-05]; the huge 999
        # print ON the publication day never enters the label.
        assert docs.at[0, 'fwd_ret_1'] == pytest.approx(104.0 / 102.0 - 1.0)

    def test_weekend_pub_rolls_to_next_trading_day(self):
        prices, _, closes = self._prices()
        docs = ll.attach_forward_returns(
            self._docs([datetime.date(2024, 1, 6)]), prices, (1,))  # Saturday
        assert docs.at[0, 'entry_date'] == datetime.date(2024, 1, 8)
        assert docs.at[0, 'fwd_ret_1'] == pytest.approx(107.0 / 106.0 - 1.0)

    def test_out_of_range_horizon_nan(self):
        prices, _, _ = self._prices()
        docs = ll.attach_forward_returns(
            self._docs([datetime.date(2024, 1, 9)]), prices, (3,))
        assert np.isnan(docs.at[0, 'fwd_ret_3'])

    def test_open_entry_mode(self):
        prices, dates, closes = self._prices()
        prices = prices.copy()
        prices['open'] = [100.0] + closes[:-1]   # open = prior close
        docs = ll.attach_forward_returns(
            self._docs([datetime.date(2024, 1, 3)]), prices, (2,))
        # entry = open[01-04] = close[01-03] = 999; exit = close[e+h-1]
        assert docs.at[0, 'fwd_ret_2'] == pytest.approx(104.0 / 999.0 - 1.0)


# ---------------------------------------------------------------------------
# T5 uniqueness weights
# ---------------------------------------------------------------------------

class TestUniqueness:
    def test_overlap_disjoint_cross_symbol(self):
        docs = pd.DataFrame({
            'symbol': ['X', 'X', 'X', 'Y', 'X'],
            'date': [datetime.date(2024, 1, 1)] * 5,
            'entry_idx': [0, 0, 10, 0, 20],
            'exit_idx_3': [2, 2, 12, 2, 22],
            'fwd_ret_3': [0.01, 0.02, 0.01, 0.03, np.nan],
        })
        w = ll.uniqueness_weights(docs, 3)
        assert w[0] == pytest.approx(0.5) and w[1] == pytest.approx(0.5)
        assert w[0] < 1.0                        # overlapping -> < 1, equal
        assert w[2] == pytest.approx(1.0)        # disjoint -> 1
        assert w[3] == pytest.approx(1.0)        # other symbol: no interaction
        assert np.isnan(w[4])                    # NaN label -> NaN weight


# ---------------------------------------------------------------------------
# T6 purged_folds geometry (purge + embargo, expanding)
# ---------------------------------------------------------------------------

class TestPurgedFolds:
    @pytest.mark.parametrize('h', [3, 5])
    def test_purge_and_default_embargo(self, synth, h):
        arts, prices, _ = synth
        docs = ll.build_docs(arts)
        docs = ll.attach_forward_returns(docs, prices, (1, 3, 5))
        folds = ll.purged_folds(docs, h, n_folds=4)   # default embargo = h
        assert len(folds) >= 2
        pub = np.array([d for d in docs['date']])
        for tr, va in folds:
            vs = min(pub[va])
            assert all(pub[va] >= vs)
            for i in tr:
                assert docs.at[i, f'exit_date_{h}'] < vs        # purge
                assert (vs - pub[i]).days >= h                  # embargo = h
        # expanding: later folds' val starts strictly later
        starts = [min(pub[va]) for _, va in folds]
        assert starts == sorted(starts) and len(set(starts)) == len(starts)

    def test_embargo_binds_beyond_purge(self, synth):
        # With h=1 the purge alone only needs exit_date < val_start (~1
        # trading day); embargo_days=10 must independently push every train
        # pub_date at least 10 calendar days before val_start — pins the
        # embargo as a distinct constraint, not a purge side effect.
        arts, prices, _ = synth
        docs = ll.build_docs(arts)
        docs = ll.attach_forward_returns(docs, prices, (1,))
        folds = ll.purged_folds(docs, 1, n_folds=4, embargo_days=10)
        assert len(folds) >= 2
        pub = np.array([d for d in docs['date']])
        for tr, va in folds:
            vs = min(pub[va])
            for i in tr:
                assert docs.at[i, 'exit_date_1'] < vs
                assert (vs - pub[i]).days >= 10


# ---------------------------------------------------------------------------
# T7 SESTM screen
# ---------------------------------------------------------------------------

class TestScreen:
    def test_planted_passes_placebo_fails(self, synth):
        arts, prices, _ = synth
        docs = ll.build_docs(arts)
        docs = ll.attach_forward_returns(docs, prices, (3,))
        y_all = docs['fwd_ret_3'].to_numpy(dtype=float)
        sub = docs[np.isfinite(y_all)].reset_index(drop=True)
        y = sub['fwd_ret_3'].to_numpy(dtype=float)
        w = np.ones(len(sub))
        _, postings = ll.build_vocab(sub, min_df=10)
        screened, tstats = ll.screen_terms(postings, y, w, screen_t=2.0)
        assert 'zephyrium' in screened
        assert abs(tstats['zephyrium']) >= 2.0
        assert 'quorvex' not in screened

    def test_sees_only_given_rows(self, synth):
        arts, prices, _ = synth
        docs = ll.build_docs(arts)
        docs = ll.attach_forward_returns(docs, prices, (3,))
        y_all = docs['fwd_ret_3'].to_numpy(dtype=float)
        fin = docs[np.isfinite(y_all)].reset_index(drop=True)
        sl = fin.iloc[:150].reset_index(drop=True)
        y = sl['fwd_ret_3'].to_numpy(dtype=float)
        w = np.ones(len(sl))
        _, p1 = ll.build_vocab(sl, min_df=5)
        s1, t1 = ll.screen_terms(p1, y, w)
        # identical result on an independent copy of the same slice
        sl2 = sl.copy(deep=True)
        _, p2 = ll.build_vocab(sl2, min_df=5)
        s2, t2 = ll.screen_terms(p2, sl2['fwd_ret_3'].to_numpy(dtype=float),
                                 np.ones(len(sl2)))
        assert s1 == s2 and t1 == t2
        # leak guard: postings pointing beyond y raise
        with pytest.raises(ValueError):
            ll.screen_terms({'x': np.array([len(y)])}, y, w)


# ---------------------------------------------------------------------------
# T8 ridge closed form
# ---------------------------------------------------------------------------

class TestRidge:
    def test_lam0_matches_ols(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(12, 3))
        y = rng.normal(size=12)
        w = np.ones(12)
        beta, b0 = ll.fit_ridge(X, y, w, 0.0)
        coef = np.linalg.lstsq(np.column_stack([np.ones(12), X]), y,
                               rcond=None)[0]
        assert np.allclose(b0, coef[0], atol=1e-8)
        assert np.allclose(beta, coef[1:], atol=1e-8)

    def test_huge_lam_shrinks(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 4))
        y = rng.normal(size=30)
        beta, _ = ll.fit_ridge(X, y, np.ones(30), 1e6)
        assert np.linalg.norm(beta) < 1e-3

    def test_weight_duplication_equivalence(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(12, 3))
        y = rng.normal(size=12)
        X2 = np.vstack([X, X[0]])
        y2 = np.append(y, y[0])
        w3 = np.ones(12)
        w3[0] = 2.0
        b_dup, i_dup = ll.fit_ridge(X2, y2, np.ones(13), 0.5)
        b_w, i_w = ll.fit_ridge(X, y, w3, 0.5)
        assert np.allclose(b_dup, b_w, atol=1e-10)
        assert np.allclose(i_dup, i_w, atol=1e-10)


# ---------------------------------------------------------------------------
# T9 end-to-end planted-signal recovery
# ---------------------------------------------------------------------------

ALLOWED_VERDICTS = {'insufficient_data', 'no_incremental_ic',
                    'candidate_signal_review_required'}


class TestEndToEnd:
    def test_recovery(self, trained):
        lex, rep = trained
        assert 'zephyrium' in lex['terms']
        zw = lex['terms']['zephyrium']
        assert zw > 0
        qw = abs(lex['terms'].get('quorvex', 0.0))
        assert qw < zw / 3
        assert rep['verdict'] in ALLOWED_VERDICTS
        h3 = rep['horizons']['3']
        assert h3['ic_learned'] is not None and h3['ic_learned'] > 0
        assert h3['ic_learned'] > (h3['ic_kw'] or 0.0)
        meta = lex['meta']
        for k in ('fit_window', 'lambda', 'screen_t', 'n_docs',
                  'fit_horizon', 'embargo_days', 'seed', 'generated_at',
                  'consumers', 'n_terms_screened', 'horizons_evaluated'):
            assert k in meta
        assert meta['fit_horizon'] == 3
        assert meta['embargo_days'] == 5          # default = max horizon
        assert meta['n_terms_screened'] == len(lex['terms'])
        assert rep['top_terms'] and any(
            t['term'] == 'zephyrium' for t in rep['top_terms'])


# ---------------------------------------------------------------------------
# T10 determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical(self, synth, trained):
        arts, prices, _ = synth
        lex1, rep1 = trained
        lex2, rep2 = ll.train_lexicon(arts, prices, horizons=(1, 3, 5),
                                      fit_horizon=3, bootstrap_b=100, seed=0)

        def strip(lex):
            m = dict(lex['meta'])
            m.pop('generated_at')   # wall-clock stamp — the only nondet field
            return json.dumps({'terms': lex['terms'], 'meta': m},
                              sort_keys=True)
        assert strip(lex1) == strip(lex2)
        assert (json.dumps(rep1['horizons'], sort_keys=True)
                == json.dumps(rep2['horizons'], sort_keys=True))

    def test_other_seed_no_crash(self, synth):
        arts, prices, _ = synth
        lex, rep = ll.train_lexicon(arts, prices, horizons=(3,),
                                    fit_horizon=3, bootstrap_b=50, seed=7)
        assert rep['verdict'] in ALLOWED_VERDICTS


# ---------------------------------------------------------------------------
# T11 fail-open degenerate inputs
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_empty_articles(self, synth):
        _, prices, _ = synth
        empty = pd.DataFrame(columns=ll._ARTICLE_COLS)
        lex, rep = ll.train_lexicon(empty, prices)
        assert lex['terms'] == {}
        assert lex['meta']['n_terms_screened'] == 0
        assert rep['verdict'] == 'insufficient_data'

    def test_zero_screened_terms(self, synth):
        arts, prices, _ = synth
        lex, rep = ll.train_lexicon(arts, prices, horizons=(3,),
                                    fit_horizon=3, screen_t=1e9,
                                    bootstrap_b=10, seed=0)
        assert lex['terms'] == {}
        assert rep['verdict'] == 'insufficient_data'

    def test_missing_price_symbol_dropped_and_counted(self, synth):
        arts, prices, _ = synth
        only_a = prices[prices['symbol'] == 'AAA'].reset_index(drop=True)
        lex, rep = ll.train_lexicon(arts, only_a, horizons=(3,),
                                    fit_horizon=3, bootstrap_b=10, seed=0)
        # every BBB symbol-day doc (220 of them) has no prices -> dropped
        assert rep['dropped_rows']['3'] >= 220
        assert rep['verdict'] in ALLOWED_VERDICTS


# ---------------------------------------------------------------------------
# T12 offline novelty mirror
# ---------------------------------------------------------------------------

class TestOfflineNovelty:
    def test_semantics(self):
        H = 'quantum widget launches amazing product line today'
        arts = pd.DataFrame({
            'symbol': ['X', 'X', 'X', 'X', 'Y', 'X', 'X'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-02',
                     '2024-01-02', '2024-01-02', '2024-01-11', '2024-01-05'],
            'headline': [
                H,                                          # 0 first -> 1.0
                H,                                          # 1 same-day dup
                H,                                          # 2 next-day dup
                'completely unrelated words regarding merger news',  # 3
                H,                                          # 4 other symbol
                H,                                          # 5 >7d after 01-02
                '',                                         # 6 empty headline
            ],
        })
        nov = ll.offline_novelty(arts)
        assert nov[0] == pytest.approx(1.0)          # first ever
        assert nov[1] == pytest.approx(1.0)          # same-day: strictly-
        #                                              earlier-date rule
        assert nov[2] == pytest.approx(0.0)          # exact reprint next day
        assert nov[3] > 0.9                          # unrelated -> fresh
        assert nov[4] == pytest.approx(1.0)          # per-symbol history
        assert nov[5] == pytest.approx(1.0)          # outside 7-day window
        assert nov[6] == pytest.approx(1.0)          # empty -> neutral 1.0
        assert ((nov >= 0.0) & (nov <= 1.0)).all()

    def test_within_window_reprint_caught(self):
        H = 'quantum widget launches amazing product line today'
        arts = pd.DataFrame({
            'symbol': ['X', 'X'],
            'date': ['2024-01-05', '2024-01-11'],     # 6 days apart (<= 7)
            'headline': [H, H],
        })
        nov = ll.offline_novelty(arts)
        assert nov[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T13 llm_analysis journal reader
# ---------------------------------------------------------------------------

class TestJournalScores:
    def test_reader(self, tmp_path):
        jd = tmp_path / 'journals'
        jd.mkdir()
        today = datetime.date.today()
        d0 = today - datetime.timedelta(days=1)
        # Row schema copied from base_loop.py _run_llm_analysis (~1411).
        rows0 = [
            {'action': 'llm_analysis', 'asset_type': 'stock',
             'forward_bars': 24,
             'scores': {'NVDA': {'s': 0.7, 'pred': 0.01},
                        'AMD': {'s': None, 'pred': 0.02}},   # null s (D33)
             'ts': f'{d0.isoformat()}T14:00:00+00:00'},
            {'action': 'order_submitted', 'symbol': 'NVDA',
             'ts': f'{d0.isoformat()}T15:00:00+00:00'},       # foreign row
        ]
        rows1 = [
            {'action': 'llm_analysis', 'asset_type': 'stock',
             'forward_bars': 24,
             'scores': {'NVDA': {'s': 0.6, 'pred': 0.0}},
             'ts': f'{today.isoformat()}T10:00:00+00:00'},
            {'action': 'llm_analysis', 'asset_type': 'stock',
             'forward_bars': 24,
             'scores': {'NVDA': {'s': 0.8, 'pred': 0.0}},
             'ts': f'{today.isoformat()}T16:00:00+00:00'},
        ]
        with open(jd / f'{d0.isoformat()}.jsonl', 'w') as f:
            for r in rows0:
                f.write(json.dumps(r) + '\n')
            f.write('{"action": "llm_analysis", "scores"\n')  # torn line
        with open(jd / f'{today.isoformat()}.jsonl', 'w') as f:
            for r in rows1:
                f.write(json.dumps(r) + '\n')
        res = ll.load_llm_journal_scores(str(jd), days=5)
        assert set(res['symbol']) == {'NVDA'}       # AMD's s was null
        by = {(r.symbol, r.date): (r.s_mean, r.n)
              for r in res.itertuples(index=False)}
        assert by[('NVDA', d0)] == (pytest.approx(0.7), 1)
        assert by[('NVDA', today)] == (pytest.approx(0.7), 2)

    def test_missing_dir_and_days0(self, tmp_path):
        assert len(ll.load_llm_journal_scores(str(tmp_path / 'nope'), 5)) == 0
        assert len(ll.load_llm_journal_scores(str(tmp_path), 0)) == 0


# ---------------------------------------------------------------------------
# T14 atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_write(self, tmp_path):
        p = tmp_path / 'artifact.json'
        obj = {'b': 1, 'a': [1, 2], 'd': datetime.date(2024, 1, 2)}
        ll.write_json_atomic(str(p), obj)
        loaded = json.loads(p.read_text())
        assert loaded['b'] == 1 and loaded['a'] == [1, 2]
        assert loaded['d'] == '2024-01-02'          # default=str
        assert not (tmp_path / 'artifact.json.tmp').exists()


# ---------------------------------------------------------------------------
# T15 CLI hygiene (stdlib-only import on the Mac)
# ---------------------------------------------------------------------------

class TestCLI:
    def test_compiles_and_imports(self):
        cli = REPO / 'scripts' / 'train_lexicon.py'
        py_compile.compile(str(cli), doraise=True)
        spec = importlib.util.spec_from_file_location('c26v2_cli', cli)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                # stdlib-only top level
        assert callable(mod.main)
