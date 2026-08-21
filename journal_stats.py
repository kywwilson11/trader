"""Pure analytics over the trade journal (journals/*.jsonl).

Measurement-only: no writes, no trading-path imports, stdlib only
(json/pathlib/datetime/math/collections — no numpy, no pandas). Consumed by
the GUI's future journal-analytics view (research/gui_review_2026-07.md §4
"journal analytics" missing item, §11 Phase 2.5) — the same role
chart_core.py plays for chart math: gui.py owns Qt rendering, this module
just hands back plain dicts/lists it can render directly.

Row schema this module reads (see trade_journal.py's own module docstring
for the full producer contract; only the keys below are ever touched here):

  "buy"  (base_loop.py:1950-1961 — crypto AND the shared/base path;
          stock_loop.py:961-973 — stock's own bracket-order path; both
          write the identical key set): symbol, action="buy",
          final_notional, decision_price, fill_price, ts. `final_notional`
          is the dollar SIZE of the fill (Alpaca notional, not qty) — the
          only place a dollar amount is journaled for an entry.
  "sell" (base_loop.py:1080-1086 `_record_confirmed_exit` — used by BOTH
          books for every signal/stop/TP/EOD-flatten/circuit-breaker exit,
          e.g. exit_reason values 'signal_sell' base_loop.py:1123,
          'server_stop' base_loop.py:708, 'circuit_breaker' base_loop.py:477,
          'eod_flatten' stock_loop.py:387; PLUS stock_loop.py:643-649
          `_journal_external_close` for broker-side closes the bot didn't
          initiate — identical key set in every case): symbol,
          action="sell", exit_reason, pnl_pct, ts. pnl_pct is REALIZED and
          self-contained (written as
          `((fill_price - pos.entry_price) / pos.entry_price) * 100` at
          exit time) — a sell row alone is enough to score a trade's
          outcome. No dollar amount is ever written on a sell row, so
          pnl_dollars can only come from pairing with a buy row's
          `final_notional`.

Every row also carries `ts` (offset-aware ISO-8601, stamped once by
trade_journal.py:81 for every row type) and lives in a date-named file
`journals/YYYY-MM-DD.jsonl` (trade_journal.py:82).

Pairing strategy (buys -> round-trip trades): LIFO, one open slot per
symbol. Crypto and stock symbols never collide — crypto is always
'BASE/QUOTE' (e.g. 'BTC/USD'), stocks never contain '/' — the same
convention used codebase-wide to tell the books apart (order_utils.py:30,
trading_utils.py:202, monitor_drift.py:280, gui.py:2560), so one dict keyed
by bare symbol is enough to keep both books' open entries apart without
ambiguity. A second buy for the same symbol before its sell (a scale-in
"add") OVERWRITES the pairing slot with the newer buy's ts/notional rather
than blending — this mirrors what the live bots themselves do:
`self.positions[symbol] = Position(entry_price=fill_price, ...)` on every
buy (base_loop.py:1937-1944) REPLACES entry_price with the latest fill
instead of computing a weighted-average cost basis, so the sell row's own
`pnl_pct` is already anchored to the MOST RECENT buy — LIFO pairing
reproduces the same basis the writer used, it doesn't invent a new one.

A sell with no open buy in view (position opened before the loaded window,
or before file retention/rotation) still becomes a trade — pnl_pct and
exit_reason are self-contained on the sell row — but entry_ts/
holding_hours/pnl_dollars come back None. `since_ts`/`until_ts` filter
individual ROWS (not whole files) before pairing, so a window that starts
between a buy and its sell will show that trade with entry_ts=None even
though the buy technically exists on disk just outside the window. That's
a documented tradeoff, not a bug: the alternative (always reading one
extra file before since_ts "just in case" a position spans the boundary)
adds asymmetric complexity for a rare edge case and isn't done here.

entry_ts/exit_ts on trade dicts, and the `ts` inside best/worst-trade
blocks, are epoch-second floats
(`datetime.datetime.fromisoformat(row['ts']).timestamp()`) — the same
numeric convention chart_core.py's arrays use, so the GUI can plot them
without re-parsing ISO strings. `daily_realized`'s date keys are plain ISO
date strings (a calendar-day bucket label, not an instant).
"""

import datetime
import gzip
import json
import math
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) Loading
# ---------------------------------------------------------------------------

def _open_journal_file(path):
    """Plain .jsonl preferred; .jsonl.gz fallback (post-rotation). Mirrors
    trade_journal.open_journal without importing the trading path."""
    path = Path(path)
    if path.exists():
        return open(path)
    gz = Path(f'{path}.gz')
    if gz.exists():
        return gzip.open(gz, 'rt')
    raise FileNotFoundError(path)

def _derive_book(symbol: str) -> str:
    """'/' marks a crypto pair (e.g. 'BTC/USD'); stocks never contain one —
    the same convention used across the codebase (order_utils.py:30,
    trading_utils.py:202, monitor_drift.py:280, gui.py:2560)."""
    return 'crypto' if '/' in symbol else 'stock'


def load_trades(journal_dir, since_ts: float = None, until_ts: float = None,
                 stats: dict = None) -> list:
    """Scan journals/YYYY-MM-DD.jsonl in `journal_dir`, pair buy/sell rows
    per symbol, and return closed round-trip trades.

    Args:
        journal_dir: directory holding the date-named *.jsonl files (str or
            Path).
        since_ts / until_ts: optional POSIX-epoch-second bounds (same units
            as chart_core.py's `load_trade_markers`); rows whose `ts` falls
            outside [since_ts, until_ts] are excluded before pairing.
        stats: optional dict; if given, populated with
            {'files_read', 'rows_seen', 'corrupt_lines', 'trades'} counts.
            An out-parameter rather than a second return value so the
            return type stays exactly list[dict] for every caller, while
            tests/GUI can still see how many lines were unusable.

    Returns:
        list[dict], ordered by exit_ts ascending (rows are globally sorted
        by ts before pairing, and trades are appended in that same order).
        Each dict: symbol, book, entry_ts, exit_ts, holding_hours, pnl_pct,
        pnl_dollars, exit_reason, entry_tactic, maker, avg_corr,
        sizing_stack — see the module docstring for exact semantics.
        entry_tactic/maker/avg_corr/sizing_stack come from the paired buy
        row (sizing_stack = the buy's nested sizing['stack']); all four are
        None when the buy is outside the window or never carried them.

    Never raises: a missing directory, unreadable file, malformed JSON
    line, non-object JSON line, or a buy/sell row missing a required field
    is skipped and counted rather than propagated — mirroring
    chart_core.py's `load_trade_markers` and trade_journal.py's own
    `iter_journal_rows`/`get_journal_summary` corrupt-line tolerance.
    """
    jdir = Path(journal_dir)
    corrupt_lines = 0
    files_read = 0
    rows = []

    if jdir.exists():
        names = {}
        for path in jdir.glob('*.jsonl'):
            names[path.stem] = path            # plain wins over .gz
        for path in jdir.glob('*.jsonl.gz'):
            stem = path.name[:-len('.jsonl.gz')]
            names.setdefault(stem, path)
        for stem in sorted(names):
            try:
                file_date = datetime.date.fromisoformat(stem)
            except ValueError:
                continue  # not a date-named journal file — ignore
            # Date fast-path: the writer stamps ts and the filename from ONE
            # clock read (trade_journal.py log_decision), so every row in
            # file D has local date D — files wholly outside
            # [since_ts, until_ts] can be skipped without opening
            # (byte-identical results; their rows would have been filtered
            # anyway).
            if since_ts is not None and file_date < datetime.date.fromtimestamp(since_ts):
                continue
            if until_ts is not None and file_date > datetime.date.fromtimestamp(until_ts):
                continue
            try:
                f = _open_journal_file(jdir / f'{stem}.jsonl')
            except (FileNotFoundError, OSError):
                continue
            try:
                with f:
                    lines = f.readlines()
            except (OSError, EOFError):
                continue
            files_read += 1
            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    corrupt_lines += 1
                    continue
                if not isinstance(row, dict):
                    corrupt_lines += 1
                    continue
                if row.get('action') not in ('buy', 'sell'):
                    continue
                symbol = row.get('symbol')
                if not isinstance(symbol, str) or not symbol:
                    corrupt_lines += 1
                    continue
                try:
                    ts_epoch = datetime.datetime.fromisoformat(row.get('ts')).timestamp()
                except (TypeError, ValueError):
                    corrupt_lines += 1
                    continue
                if since_ts is not None and ts_epoch < since_ts:
                    continue
                if until_ts is not None and ts_epoch > until_ts:
                    continue
                rows.append((ts_epoch, row))

    rows.sort(key=lambda item: item[0])

    open_buy = {}
    trades = []
    for ts_epoch, row in rows:
        symbol = row['symbol']
        if row['action'] == 'buy':
            notional = row.get('final_notional')
            try:
                notional = float(notional) if notional is not None else None
            except (TypeError, ValueError):
                notional = None
            if notional is not None and not math.isfinite(notional):
                notional = None
            avg_corr = row.get('avg_corr')
            try:
                avg_corr = float(avg_corr) if avg_corr is not None else None
            except (TypeError, ValueError):
                avg_corr = None
            sizing = row.get('sizing')
            open_buy[symbol] = {
                'ts': ts_epoch, 'notional': notional,
                'entry_tactic': row.get('entry_tactic'),
                'maker': row.get('maker'),
                'avg_corr': avg_corr,
                'sizing_stack': (sizing.get('stack') if isinstance(sizing, dict) else None),
            }
            continue

        # sell
        pnl_raw = row.get('pnl_pct')
        try:
            pnl_pct = float(pnl_raw) if pnl_raw is not None else None
        except (TypeError, ValueError):
            pnl_pct = None
        if pnl_pct is None or not math.isfinite(pnl_pct):
            corrupt_lines += 1
            continue

        entry = open_buy.pop(symbol, None)
        entry_ts = None
        holding_hours = None
        pnl_dollars = None
        entry_tactic = maker = avg_corr = sizing_stack = None
        if entry is not None:
            entry_ts = entry['ts']
            notional = entry['notional']
            holding_hours = (ts_epoch - entry_ts) / 3600.0
            if notional is not None:
                pnl_dollars = notional * pnl_pct / 100.0
            entry_tactic = entry['entry_tactic']
            maker = entry['maker']
            avg_corr = entry['avg_corr']
            sizing_stack = entry['sizing_stack']

        trades.append({
            'symbol': symbol,
            'book': _derive_book(symbol),
            'entry_ts': entry_ts,
            'exit_ts': ts_epoch,
            'holding_hours': holding_hours,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'exit_reason': row.get('exit_reason'),
            'entry_tactic': entry_tactic,
            'maker': maker,
            'avg_corr': avg_corr,
            'sizing_stack': sizing_stack,
        })

    if stats is not None:
        stats['files_read'] = files_read
        stats['rows_seen'] = len(rows)
        stats['corrupt_lines'] = corrupt_lines
        stats['trades'] = len(trades)
    return trades


# ---------------------------------------------------------------------------
# 2) Stats
# ---------------------------------------------------------------------------

def _median(values: list) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _empty_block() -> dict:
    return {
        'n_trades': 0,
        'win_rate': None,
        'expectancy_pct': None,
        'profit_factor': None,
        'avg_win_pct': None,
        'avg_loss_pct': None,
        'median_holding_hours': None,
        'best_trade': None,
        'worst_trade': None,
        'by_exit_reason': {},
    }


def _trade_block(trades: list) -> dict:
    """Stats for one group of trade dicts (overall / one book / one
    symbol). All divisions guarded; an empty group returns the zeroed
    shape from _empty_block() so a GUI never has to special-case it."""
    n = len(trades)
    if n == 0:
        return _empty_block()

    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)  # positive magnitude

    holds = [t['holding_hours'] for t in trades if t['holding_hours'] is not None]

    best = max(trades, key=lambda t: t['pnl_pct'])
    worst = min(trades, key=lambda t: t['pnl_pct'])

    reasons = Counter(t['exit_reason'] if t['exit_reason'] else 'unknown' for t in trades)

    return {
        'n_trades': n,
        'win_rate': len(wins) / n,
        'expectancy_pct': sum(pnls) / n,
        'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else None,
        'avg_win_pct': (gross_win / len(wins)) if wins else None,
        'avg_loss_pct': (sum(losses) / len(losses)) if losses else None,
        'median_holding_hours': _median(holds) if holds else None,
        'best_trade': {'symbol': best['symbol'], 'pnl_pct': best['pnl_pct'],
                        'ts': best['exit_ts']},
        'worst_trade': {'symbol': worst['symbol'], 'pnl_pct': worst['pnl_pct'],
                         'ts': worst['exit_ts']},
        'by_exit_reason': dict(reasons),
    }


def compute_stats(trades: list) -> dict:
    """Overall + per-book + per-symbol blocks, plus a daily realized-P&L
    split. Never raises; `trades=[]` returns the zeroed/None shape every
    block uses so a GUI can render it directly with no special-casing.

    Returns:
        {
          'overall': {block},
          'by_book': {'crypto': {block}, 'stock': {block}, ...},
          'by_symbol': {symbol: {block}, ...},
          'daily_realized': [(date_str, dollars_or_None), ...] ascending,
        }
        where {block} = n_trades, win_rate, expectancy_pct, profit_factor,
        avg_win_pct, avg_loss_pct, median_holding_hours, best_trade,
        worst_trade, by_exit_reason (see _trade_block/_empty_block).

    win_rate / expectancy_pct / profit_factor / avg_win_pct / avg_loss_pct
    are all computed on pnl_pct (percentage points), NOT pnl_dollars —
    pnl_dollars is frequently None (only known when a trade's buy row was
    inside the loaded window AND carried `final_notional`), so a
    dollar-based profit factor would silently drop most trades. Profit
    factor is therefore "gross % of winning trades / gross % of |losing
    trades|", not the classic dollar-gross ratio — documented here because
    it's the one place this module's numbers deviate from the textbook
    definition. `daily_realized` is the one dollar-based view, kept
    separate for exactly that reason: a day's sum is None only when NOT
    ONE trade closed that day has a resolvable dollar amount; otherwise
    it's the sum of whichever trades that day do (unresolvable trades
    within an otherwise-resolvable day are silently excluded from the sum,
    not treated as zero).
    """
    overall = _trade_block(trades)

    by_book_trades = {}
    by_symbol_trades = {}
    for t in trades:
        by_book_trades.setdefault(t['book'], []).append(t)
        by_symbol_trades.setdefault(t['symbol'], []).append(t)

    by_book = {b: _trade_block(ts) for b, ts in by_book_trades.items()}
    by_symbol = {s: _trade_block(ts) for s, ts in by_symbol_trades.items()}

    day_dollar_sums = {}
    day_has_trade = set()
    for t in trades:
        if t['exit_ts'] is None:
            continue
        d = datetime.date.fromtimestamp(t['exit_ts']).isoformat()
        day_has_trade.add(d)
        if t['pnl_dollars'] is not None:
            day_dollar_sums[d] = day_dollar_sums.get(d, 0.0) + t['pnl_dollars']
    daily_realized = [(d, day_dollar_sums.get(d)) for d in sorted(day_has_trade)]

    return {
        'overall': overall,
        'by_book': by_book,
        'by_symbol': by_symbol,
        'daily_realized': daily_realized,
    }


# ---------------------------------------------------------------------------
# 3) Display
# ---------------------------------------------------------------------------

def _pct1(x):
    return 'n/a' if x is None else f'{x * 100:.1f}%'


def _spct(x):
    return 'n/a' if x is None else f'{x:+.2f}%'


def _ratio(x):
    return 'n/a' if x is None else f'{x:.2f}'


def _hours(x):
    return 'n/a' if x is None else f'{x:.1f}h'


def format_summary(stats: dict) -> str:
    """Compact multi-line text rendering of `compute_stats()`'s output — a
    v1 monospace-label view for the GUI's journal-analytics panel."""
    overall = stats.get('overall') or _empty_block()
    n = overall.get('n_trades', 0)

    lines = ['Journal Analytics', '=================']
    if not n:
        lines.append('No closed trades in range.')
        return '\n'.join(lines)

    lines.append(
        f"Overall: {n} trades | win rate {_pct1(overall['win_rate'])} | "
        f"expectancy {_spct(overall['expectancy_pct'])} | "
        f"profit factor {_ratio(overall['profit_factor'])}")
    lines.append(
        f"  avg win {_spct(overall['avg_win_pct'])} | "
        f"avg loss {_spct(overall['avg_loss_pct'])} | "
        f"median hold {_hours(overall['median_holding_hours'])}")
    best, worst = overall.get('best_trade'), overall.get('worst_trade')
    if best and worst:
        lines.append(
            f"  best: {best['symbol']} {_spct(best['pnl_pct'])}   "
            f"worst: {worst['symbol']} {_spct(worst['pnl_pct'])}")

    by_book = stats.get('by_book') or {}
    if by_book:
        lines.append('')
        lines.append('By book:')
        for book in sorted(by_book):
            b = by_book[book]
            lines.append(
                f"  {book:<7} {b['n_trades']:>4} trades | win {_pct1(b['win_rate'])} | "
                f"expectancy {_spct(b['expectancy_pct'])} | PF {_ratio(b['profit_factor'])}")

    by_symbol = stats.get('by_symbol') or {}
    if by_symbol:
        lines.append('')
        lines.append(f'Symbols traded: {len(by_symbol)}')

    reasons = overall.get('by_exit_reason') or {}
    if reasons:
        parts = ', '.join(f'{k}={v}' for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
        lines.append('')
        lines.append(f'Exit reasons: {parts}')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# 4) EOD digest
# ---------------------------------------------------------------------------

def _is_crypto_symbol(sym: str) -> bool:
    """Same book predicate as execution_report.crypto(): '/' pairs are
    crypto, and so are bare 'XXXUSD' Alpaca crypto symbols."""
    return '/' in sym or (sym.endswith('USD') and len(sym) > 5)


def build_eod_digest(journal_dir, positions=None, now=None) -> str:
    """One compact end-of-day summary string for notify (c26 T7 / B19).

    positions: optional list of plain {'symbol', 'unrealized_pl'} dicts
    (fetched best-effort by the caller); None renders unrealized as 'n/a'.
    Never raises; output truncated to 3500 chars (Telegram's 4096 limit).
    """
    try:
        now = now or datetime.datetime.now().astimezone()
        today = now.date()

        # Closed round-trips: load the last 7 days so multi-day holds that
        # closed today still pair with their buy rows.
        since = datetime.datetime.combine(
            today - datetime.timedelta(days=7),
            datetime.time.min).astimezone().timestamp()
        trades = load_trades(journal_dir, since_ts=since)
        closed_today = [t for t in trades
                        if datetime.date.fromtimestamp(t['exit_ts']) == today]

        # Raw gate counts from today's file only.
        buys = sells = skips = llm_calls = llm_backoffs = 0
        skip_reasons = Counter()
        try:
            f = _open_journal_file(
                Path(journal_dir) / f'{today.isoformat()}.jsonl')
        except (FileNotFoundError, OSError):
            f = None
        if f is not None:
            try:
                with f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(row, dict):
                            continue
                        action = row.get('action')
                        if action == 'buy':
                            buys += 1
                        elif action == 'sell':
                            sells += 1
                        elif action == 'skip':
                            skips += 1
                            skip_reasons[row.get('skip_reason') or 'unknown'] += 1
                        elif action == 'llm_analysis':
                            llm_calls += 1
                        elif action == 'llm_backoff':
                            llm_backoffs += 1
            except (OSError, EOFError):
                pass

        lines = [f'EOD digest {today}']
        lines.append(f'Rows today: {buys} buys, {sells} sells, {skips} skips, '
                     f'{llm_calls} LLM calls')

        by_book = {}
        for t in closed_today:
            by_book.setdefault(t['book'], []).append(t)
        for book in sorted(by_book):
            bt = by_book[book]
            wins = sum(1 for t in bt if t['pnl_pct'] > 0)
            losses = sum(1 for t in bt if t['pnl_pct'] < 0)
            expectancy = sum(t['pnl_pct'] for t in bt) / len(bt)
            dollars = [t['pnl_dollars'] for t in bt
                       if t['pnl_dollars'] is not None]
            # '~' marks partial resolution: some closed trades had no
            # resolvable dollar amount (buy outside window / no notional).
            approx = '~' if any(t['pnl_dollars'] is None for t in bt) else ''
            realized = sum(dollars) if dollars else 0.0
            lines.append(f'{book}: {len(bt)} closed ({wins}W/{losses}L), '
                         f'expectancy {expectancy:+.2f}%, '
                         f'realized {approx}${realized:+,.2f}')
        if not closed_today:
            lines.append('No trades closed today.')

        if positions is None:
            lines.append('Unrealized: n/a')
        else:
            crypto_u = stock_u = 0.0
            for p in positions:
                try:
                    pl = float(p.get('unrealized_pl'))
                except (TypeError, ValueError):
                    continue
                if _is_crypto_symbol(str(p.get('symbol', ''))):
                    crypto_u += pl
                else:
                    stock_u += pl
            lines.append(f'Unrealized: crypto ${crypto_u:+,.2f}, '
                         f'stock ${stock_u:+,.2f} ({len(positions)} open)')

        if closed_today:
            ranked = sorted(closed_today, key=lambda t: t['pnl_pct'],
                            reverse=True)
            tops = ', '.join(f"{t['symbol']} {t['pnl_pct']:+.2f}%"
                             for t in ranked[:2])
            bottoms = ', '.join(f"{t['symbol']} {t['pnl_pct']:+.2f}%"
                                for t in ranked[-2:])
            lines.append(f'Top: {tops}')
            lines.append(f'Bottom: {bottoms}')

        if skip_reasons:
            top4 = ', '.join(f'{k}={v}'
                             for k, v in skip_reasons.most_common(4))
            lines.append(f'Skip reasons: {top4}')
        if llm_backoffs:
            lines.append(f'LLM backoffs: {llm_backoffs}')

        return '\n'.join(lines)[:3500]
    except Exception as e:
        return (f'EOD digest {datetime.date.today()}: '
                f'(build failed: {type(e).__name__}: {e})')
