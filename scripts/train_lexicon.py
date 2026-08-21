"""Offline trainer CLI for the learned sentiment lexicon — DARK ARTIFACT.

Runs on the Jetson (needs the gitignored, Jetson-local sentiment_cache.db
and stock training parquet). Writes learned_lexicon.json and
lexicon_eval_report.json — local research outputs that NOTHING consumes;
promotion to a live feature is a later owner harvest+retrain decision
(CLAUDE.md gotcha #2). Nothing imports this script; its exit code is
informational only. Top-level imports are stdlib-only so py_compile and
import stay Mac-safe; pandas + learned_lexicon load lazily inside main().
"""

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_REPO = Path(__file__).resolve().parent.parent


def _parse_date(s):
    return datetime.date.fromisoformat(s) if s else None


def _fmt_ic(v):
    return 'nan' if v is None else f'{v:+.4f}'


def main(argv=None):
    p = argparse.ArgumentParser(description=(
        'OFFLINE supervised lexicon induction (Jegadeesh-Wu word power / '
        'SESTM). Dark artifact: writes learned_lexicon.json + '
        'lexicon_eval_report.json, which nothing consumes. Needs '
        'Jetson-local data (sentiment_cache.db, stock training parquet).'))
    p.add_argument('--db', default=str(_REPO / 'sentiment_cache.db'))
    p.add_argument('--data', default=str(_REPO / 'stock_training_data.parquet'))
    p.add_argument('--horizons', default='1,3,5')
    p.add_argument('--fit-horizon', type=int, default=3)
    p.add_argument('--start', default=None, help='ISO date (optional)')
    p.add_argument('--end', default=None, help='ISO date (optional)')
    p.add_argument('--min-df', type=int, default=10)
    p.add_argument('--screen-t', type=float, default=2.0)
    p.add_argument('--embargo-days', type=int, default=None,
                   help='default: max horizon')
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--lambda-grid', default='0.01,0.1,1,10,100')
    p.add_argument('--journal-days', type=int, default=0,
                   help='0 = skip the llm_analysis journal join')
    p.add_argument('--no-novelty', action='store_true',
                   help='skip offline novelty weighting')
    p.add_argument('--recompute-kw', action='store_true',
                   help='recompute the static baseline via '
                        'sentiment_history._keyword_score (read-only) '
                        'instead of the stored keyword_score column')
    p.add_argument('--out', default=str(_REPO / 'learned_lexicon.json'))
    p.add_argument('--report', default=str(_REPO / 'lexicon_eval_report.json'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    try:
        import pandas as pd

        import learned_lexicon as ll

        horizons = tuple(int(h) for h in args.horizons.split(','))
        grid = tuple(float(x) for x in args.lambda_grid.split(','))

        articles = ll.load_articles(args.db, start=_parse_date(args.start),
                                    end=_parse_date(args.end))
        if len(articles) == 0:
            print('[TRAIN-LEXICON] sentiment_cache.db has no articles in '
                  'range — run sentiment_history --fetch-stocks on the '
                  'Jetson first')
            sys.exit(1)

        if args.recompute_kw:
            # Read-only use of the exact function that produced the stored
            # training feature (sentiment_history._keyword_score).
            from sentiment_history import _keyword_score
            articles = articles.copy()
            articles['keyword_score'] = [
                float(_keyword_score(h or '', s or ''))
                for h, s in zip(articles['headline'], articles['summary'])]

        # Hourly OHLCV -> daily tidy [symbol, date, open, close].
        df = pd.read_parquet(args.data)
        for c in ('Ticker', 'Close'):
            if c not in df.columns:
                raise ValueError(f'{args.data} missing column {c!r}')
        idx = df.index
        if getattr(idx, 'tz', None) is not None:
            dates = idx.tz_convert('UTC').date
        else:
            print('[TRAIN-LEXICON] warning: naive datetime index — '
                  'assuming it is already UTC')
            dates = idx.date
        df = df.assign(_date=dates)
        df = df[df['Ticker'].isin(set(articles['symbol'].unique()))]
        g = df.groupby(['Ticker', '_date'], sort=True)
        prices = g['Close'].last().rename('close').reset_index()
        if 'Open' in df.columns:
            prices['open'] = g['Open'].first().to_numpy()
        prices = prices.rename(columns={'Ticker': 'symbol', '_date': 'date'})

        novelty = (None if args.no_novelty
                   else ll.offline_novelty(articles))
        journal_scores = (
            ll.load_llm_journal_scores(str(_REPO / 'journals'),
                                       args.journal_days)
            if args.journal_days > 0 else None)

        lexicon, report = ll.train_lexicon(
            articles, prices, horizons=horizons,
            fit_horizon=args.fit_horizon, min_df=args.min_df,
            screen_t=args.screen_t, embargo_days=args.embargo_days,
            n_folds=args.folds, lambda_grid=grid, novelty=novelty,
            journal_scores=journal_scores, seed=args.seed)

        out_path = str(Path(args.out))
        rep_path = str(Path(args.report))
        ll.write_json_atomic(out_path, lexicon)
        ll.write_json_atomic(rep_path, report)

        meta = lexicon['meta']
        print(f"[TRAIN-LEXICON] n_articles={len(articles)} "
              f"n_docs={meta['n_docs']} "
              f"n_terms_screened={meta['n_terms_screened']} "
              f"lambda={meta['lambda']}")
        for h, v in sorted(report.get('horizons', {}).items(),
                           key=lambda kv: int(kv[0])):
            lo, hi = v.get('ic_ci', [None, None])
            ci = (f"[{_fmt_ic(lo)},{_fmt_ic(hi)}]"
                  if lo is not None else '[--]')
            print(f"  h={h}d learned={_fmt_ic(v.get('ic_learned'))} "
                  f"CI={ci} n={v.get('n_oos')} | "
                  f"kw={_fmt_ic(v.get('ic_kw'))} (n={v.get('n_kw')}) | "
                  f"llm={_fmt_ic(v.get('ic_llm'))} (n={v.get('n_llm')}) | "
                  f"journal={_fmt_ic(v.get('ic_journal'))} "
                  f"(n={v.get('n_journal')})")
        print(f"[TRAIN-LEXICON] verdict: {report.get('verdict')}")
        print(f"[TRAIN-LEXICON] wrote {out_path} and {rep_path} "
              f"(local research outputs — gitignore them; consumed by "
              f"NOTHING until an owner harvest+retrain decision)")
    except SystemExit:
        raise
    except Exception as e:
        print(f'[TRAIN-LEXICON] failed: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
