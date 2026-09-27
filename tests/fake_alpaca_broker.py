"""Deterministic in-memory fake of the legacy ``alpaca_trade_api.REST``
surface the live loops use (base_loop / crypto_loop / stock_loop /
order_utils) — and a recorded-day replay harness built on it.

Stdlib only — no alpaca / torch / numpy imports — so it is Mac-safe. NOT a
pytest module (no ``test_`` prefix); import it from tests.

Surface (census of every ``api.``/``self.api.`` attribute used by
base_loop.py, crypto_loop.py, stock_loop.py and order_utils.py, 2026-09-27):
get_account, list_positions, get_position, list_orders, get_order,
submit_order, cancel_order, cancel_all_orders, get_latest_crypto_quotes,
get_latest_quote, get_clock. (No consumer calls replace_order,
close_position, get_calendar or get_latest_trade; market_data's get_bars /
get_crypto_bars are stubbed at their module seam by the tests instead.)

PRICES — two tape forms, both advanced by ``tick()``:

* legacy mid tape ``{'BTC/USD': [p0, p1, ...]}``: bid/ask = mid -/+
  ``spread_bps``/2, last = mid. Symbols without a tape use the snapshot
  position's ``current_price``.
* ``Tape`` rows (recorded or synthetic), JSON schema
  ``fake_alpaca_broker.tape/v1``::

      {"schema": "fake_alpaca_broker.tape/v1",
       "recorded_at": "<iso utc>", "source": "...", "interval_s": 30.0,
       "symbols": {"BTC/USD": [[ts_iso, bid, ask, last, quote_age_s], ...]}}

  ``quote_age_s`` (5th element) is optional/None; with
  ``replay_quote_age=True`` a quote's ``t`` is ``clock() - age`` so the
  loop's 180 s staleness rule sees the recorded age. Row k is the price at
  step k (clamped to the last row).

Quotes carry ``t = clock()`` (minus the recorded age, above).

TRIGGER price for stop / trailing orders: the BID on a legacy mid tape
(unchanged round-1 rule), ``last`` on a ``Tape``.

SYMBOLS: orders keep the spelling they were submitted with ('BTC/USD');
positions are keyed slashless ('BTCUSD') like Alpaca; lookups accept both.
Crypto = contains '/' or ends in 'USD' with len > 5; everything else is a
US equity (asset_class 'us_equity').

SESSIONS (stocks only): ``sessions=[(open_utc, close_utc), ...]``; None =
always open (legacy). While closed, stock orders are accepted but do not
fill (Alpaca queues them); a tick that crosses a session close EXPIRES
every open stock 'day' order (status 'expired'). ``get_clock()`` reports
is_open / next_open / next_close from the sessions.

FILL / REJECT RULES (every one recorded in ``events``):

* BUY: rejected when qty*ref_price exceeds available funds, ref_price =
  limit_price for limit orders, ask otherwise. Crypto: ``insufficient
  balance for USD (requested: X, available: Y)`` against
  ``non_marginable_buying_power`` (= cash - open buy notional). Stock:
  ``insufficient buying power`` against buying_power (cash + the snapshot's
  buying_power-cash offset - open buy notional). Wording modelled on
  Alpaca's; exact live text unverified.
* SELL: rejected when qty exceeds position qty minus qty reserved by open
  sell orders (an OCO bracket group reserves ONCE). Crypto: ``insufficient
  balance for BTC (requested: X, available: Y)``; stock: ``insufficient qty
  available for order (requested: X, available: Y)``.
* limit_price / stop_price <= 0 rejected (``stop_price must be > 0``).
* market: fills at ask (buy) / bid (sell) — stocks only while open.
* limit: fills at the touch when marketable (buy limit >= ask -> ask; sell
  limit <= bid -> bid), on submit or any later tick; else rests 'new'.
* stop_limit (sell): TRIGGERS when the trigger price <= stop_price; once
  triggered it is a sell limit: fills at the bid when bid >= limit_price,
  otherwise keeps resting (a gap through the limit leaves it unfilled —
  the bounded-slippage design). A limit can never fill BELOW its limit.
* stop (sell, stock): triggers like stop_limit, then fills at the bid.
* trailing_stop (sell, ``trail_percent``): high-water mark of the trigger
  price since submit; triggers at hwm*(1-pct/100), fills at the bid.
* bracket (``order_class='bracket'``, stop_loss={'stop_price'[, 'limit_price']},
  take_profit={'limit_price'}): Alpaca's buy-bracket price rules are
  enforced — take_profit.limit_price >= base + 0.01 and stop_loss.stop_price
  <= base - 0.01 (base = parent limit, else ask), else rejected. Two child
  legs (TP limit + SL stop/stop_limit) are created 'held', activate ('new')
  when the parent fills, reserve the qty once, and are One-Cancels-Other: a
  leg fill cancels its sibling; cancelling one leg cancels the group;
  cancelling the unfilled parent cancels the legs. The parent's order
  object carries ``legs`` (list of leg objects). (Alpaca's exact held/new
  leg statuses after the parent fill are unverified.)
* cancel_order: immediate (status 'canceled'); a terminal order raises
  ``order is not cancelable``.
* ``report_zero_basis = True`` keeps a position's avg_entry_price at 0 after
  buy fills (the paper quirk) while the order's filled_avg_price is real.
* Fills move cash and the position; a buy into an existing position
  re-averages avg_entry_price as (cost_basis + fill notional) / new qty —
  with the paper zero-basis quirk (cost_basis=0) that yields a TINY average.
  This blending rule is an assumption (unverified for Alpaca).
* equity = cash + sum(qty * mid); last_equity stays the snapshot value
  (the circuit-breaker baseline only rolls at the broker's daily reset).

CONTROLLABLE FAULTS (ENGINE r7 W20; every one default-OFF, so the round-1..6
consumers see byte-identical behaviour; a "window" is broker steps
[at_step, at_step + cycles) and at_step defaults to the CURRENT step — one
harness cycle = one tick = one step):

* ``hide_position(symbol, cycles=1, at_step=None)`` (H1, the S7 "position
  vanishes" paper quirk): inside the window list_positions omits the
  position and get_position raises ``position does not exist``; a NEW sell
  for it is rejected as if nothing were held (crypto wording ``insufficient
  balance for BTC (requested: X, available: 0)``). Orders are untouched:
  resting orders stay open and keep matching against the tape, and the
  position's qty and value stay in equity (equity intact). What live
  Alpaca answers to a sell during S7 is unverified.
* ``equity_equals_cash(cycles=1, at_step=None)`` (H2, the 2026-08-13 print
  equity == cash): inside the window get_account reports equity ==
  portfolio_value == cash (long_market_value 0) and EVERY position is
  hidden as above.
* ``fill_avg_none(position=True, order=False)`` (H3/H5): position=True
  mimics a NULL basis (not "0"): new buy fills keep basis 0 (implies
  ``report_zero_basis``) and ``_pos_obj`` reports avg_entry_price and
  cost_basis as None for every zero-basis position; order=True makes
  filled BUY orders report filled_avg_price None. A snapshot position whose
  avg_entry_price is None is reported null too (until a real-basis fill).
* ``reserve_qty_on_resting_stops`` (H4, default True = the round-1 rule
  above): open sells reserve qty, so qty_available = qty - reserved and a
  second full-qty stop_limit is rejected with the ``insufficient balance``
  wording. False turns the reservation off (a counterfactual control).
* ``ns_timestamps=True`` (ctor kw / attribute, W16 F2a): quote ``t`` is a
  tz-aware pandas Timestamp with 123 ns added — the live SDK's type, which
  drives order_utils.get_quote through ``to_pydatetime(warn=False)``.
  pandas is imported lazily, only on this path.
* ``canceled_at`` (W16 F2b): every order carries ``canceled_at`` (None
  until cancelled; the clock's iso time when cancel_order / an OCO or
  parent cascade / cancel_all_orders cancels it). Events are unchanged.
* ``apply_live_modes(monkeypatch, workdir, no_model=, halt=)`` (W16 F2c):
  the live 03:09 startup configuration — no_model chdirs into an empty dir
  so the REAL predict_now.load_models raises FileNotFoundError (its paths
  are cwd-relative) and _load_models fails closed; the halt flag
  (notify._HALT_FLAG) is redirected into ``workdir`` and, with halt=True,
  created. Callers must not stub notify.halt_active.

``calls`` logs every public API method invocation (name, symbol/order id,
step) — the round-trip counter used by the O2 unprotected-window numbers.

CLOCK: ``FakeClock`` (aware UTC). ``install_clock(monkeypatch, clock,
modules)`` points each module's ``datetime`` / ``time`` globals at shims
whose now()/today()/time()/monotonic() read the clock (sleep = no-op) —
production time calls are untouched; only the module globals are swapped.
"""

from __future__ import annotations

import copy
import datetime as _dt
import itertools
import json
import time as _time_mod
import types
from pathlib import Path
from types import SimpleNamespace


class FakeAPIError(Exception):
    """Stands in for alpaca_trade_api.rest.APIError (str() == message)."""


class FakeClock:
    """Controllable UTC clock. Starts at real now (or ``start``, made
    aware-UTC) so the loop's wall-clock quote-staleness check
    (order_utils.get_quote, 180 s) passes; advance() moves it
    deterministically."""

    def __init__(self, start: _dt.datetime | None = None):
        if start is not None and start.tzinfo is None:
            start = start.replace(tzinfo=_dt.timezone.utc)
        self._now = start or _dt.datetime.now(_dt.timezone.utc)

    def __call__(self) -> _dt.datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += _dt.timedelta(seconds=seconds)

    def set(self, when: _dt.datetime) -> None:
        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        self._now = when


def make_datetime_module(clock):
    """A stand-in for the ``datetime`` module whose datetime.now() /
    date.today() read ``clock`` (aware UTC). now() with no tz returns naive
    LOCAL time exactly like the real one; returned values are plain
    datetime/date instances."""
    real = _dt

    class _ClockDateTime(real.datetime):
        @classmethod
        def now(cls, tz=None):
            t = clock()
            if tz is None:
                return real.datetime.fromtimestamp(t.timestamp())
            return t.astimezone(tz)

        @classmethod
        def utcnow(cls):
            return clock().astimezone(real.timezone.utc).replace(tzinfo=None)

        @classmethod
        def today(cls):
            return cls.now()

    class _ClockDate(real.date):
        @classmethod
        def today(cls):
            return real.datetime.fromtimestamp(clock().timestamp()).date()

    mod = types.ModuleType('datetime')
    mod.__dict__.update({k: v for k, v in real.__dict__.items()
                         if not k.startswith('__')})
    mod.datetime = _ClockDateTime
    mod.date = _ClockDate
    mod._fake_clock_shim = True
    return mod


def make_time_module(clock):
    """A stand-in for the ``time`` module: time()/monotonic() read the
    clock, sleep() is a no-op (the tape advances only via tick())."""
    mod = types.ModuleType('time')
    mod.__dict__.update({k: v for k, v in _time_mod.__dict__.items()
                         if not k.startswith('__')})
    mod.sleep = lambda s=0: None
    mod.time = lambda: clock().timestamp()
    mod.monotonic = lambda: clock().timestamp()
    mod._fake_clock_shim = True
    return mod


def install_clock(monkeypatch, clock, modules):
    """Point every given module's ``datetime`` / ``time`` global (when it
    is the real stdlib module) at clock-driven shims. ``monkeypatch`` is
    pytest's fixture (duck-typed: needs setattr). A shim left by an
    earlier install in the same test is replaced too (re-install = new
    clock)."""
    dmod, tmod = make_datetime_module(clock), make_time_module(clock)
    for m in modules:
        cur = getattr(m, 'datetime', None)
        if cur is _dt or getattr(cur, '_fake_clock_shim', False):
            monkeypatch.setattr(m, 'datetime', dmod)
        cur = getattr(m, 'time', None)
        if cur is _time_mod or getattr(cur, '_fake_clock_shim', False):
            monkeypatch.setattr(m, 'time', tmod)
    return dmod, tmod


def _norm(symbol: str) -> str:
    return symbol.replace('/', '')


def _is_crypto(symbol: str) -> bool:
    return '/' in symbol or (symbol.endswith('USD') and len(symbol) > 5)


def _f(x) -> float:
    return float(x) if x is not None else 0.0


_OPEN = ('new', 'accepted', 'partially_filled', 'pending_new', 'held')
_TERMINAL = ('filled', 'canceled', 'expired', 'rejected')


class Tape:
    """Per-symbol rows (ts_iso, bid, ask, last[, quote_age_s]); row k is
    the price at broker step k (clamped to the last row)."""

    def __init__(self, rows: dict, meta: dict | None = None):
        self.rows = {}
        for s, rs in rows.items():
            out = []
            for r in rs:
                r = list(r) + [None] * (5 - len(r))
                out.append((r[0], float(r[1]), float(r[2]), float(r[3]),
                            None if r[4] is None else float(r[4])))
            self.rows[_norm(s)] = out
        self.meta = dict(meta or {})
        self._spelling = {_norm(s): s for s in rows}

    @classmethod
    def from_dict(cls, d: dict) -> 'Tape':
        if d.get('schema') != 'fake_alpaca_broker.tape/v1':
            raise ValueError(f"unknown tape schema {d.get('schema')!r}")
        return cls(d['symbols'], meta={k: v for k, v in d.items()
                                       if k != 'symbols'})

    @classmethod
    def from_json(cls, path) -> 'Tape':
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def symbols(self) -> list[str]:
        return [self._spelling[k] for k in self.rows]

    def __len__(self) -> int:
        return max((len(r) for r in self.rows.values()), default=0)

    def __contains__(self, symbol) -> bool:
        return _norm(symbol) in self.rows

    def row(self, symbol: str, step: int):
        rs = self.rows[_norm(symbol)]
        return rs[min(step, len(rs) - 1)]

    def start_time(self) -> _dt.datetime | None:
        firsts = [rs[0][0] for rs in self.rows.values() if rs and rs[0][0]]
        if not firsts:
            return None
        return min(_dt.datetime.fromisoformat(t) for t in firsts)


def synthetic_rth_tape(symbols_open: dict, day: _dt.date, step_s: int = 300,
                       spread_bps: float = 4.0, gap_at=(10, 0),
                       gap_pct: float = -0.04, rally_from=(14, 0),
                       rally_pct: float = 0.03, pre_s: int = 900,
                       post_s: int = 900):
    """Deterministic RTH day for US equities (ET 09:30-16:00): flat open,
    a ``gap_pct`` step at ``gap_at`` ET, flat, then a linear ``rally_pct``
    rally from ``rally_from`` into the close. Rows every ``step_s`` from
    open-``pre_s`` to close+``post_s``. Returns (Tape, sessions) with the
    session as aware-UTC datetimes."""
    import zoneinfo
    et = zoneinfo.ZoneInfo('America/New_York')
    o = _dt.datetime(day.year, day.month, day.day, 9, 30, tzinfo=et)
    c = _dt.datetime(day.year, day.month, day.day, 16, 0, tzinfo=et)
    g = o.replace(hour=gap_at[0], minute=gap_at[1])
    r0 = o.replace(hour=rally_from[0], minute=rally_from[1])
    t, end = o - _dt.timedelta(seconds=pre_s), c + _dt.timedelta(seconds=post_s)
    rows = {s: [] for s in symbols_open}
    while t <= end:
        mult = 1.0
        if t >= g:
            mult *= 1.0 + gap_pct
        if t >= r0:
            frac = min(1.0, (t - r0).total_seconds() / (c - r0).total_seconds())
            mult *= 1.0 + rally_pct * frac
        for s, p0 in symbols_open.items():
            mid = round(p0 * mult, 4)
            h = mid * spread_bps / 2e4
            rows[s].append([t.astimezone(_dt.timezone.utc).isoformat(),
                            round(mid - h, 4), round(mid + h, 4), mid, 0.0])
        t += _dt.timedelta(seconds=step_s)
    sessions = [(o.astimezone(_dt.timezone.utc), c.astimezone(_dt.timezone.utc))]
    return Tape(rows, meta={'source': 'synthetic_rth_tape',
                            'interval_s': float(step_s)}), sessions


class FakeAlpacaBroker:
    def __init__(self, snapshot: dict, tape=None, spread_bps: float = 10.0,
                 clock=None, sessions=None, replay_quote_age: bool = False,
                 ns_timestamps: bool = False):
        acct = snapshot['account']
        self._acct = dict(acct)
        self.cash = _f(acct['cash'])
        self.last_equity = _f(acct['last_equity'])
        self._bp_offset = _f(acct.get('buying_power')) - self.cash
        self.positions: dict[str, dict] = {}
        # Keys whose basis is reported NULL (fill_avg_none / snapshot None);
        # kept out of the position dicts so they stay byte-identical.
        self._null_basis: set[str] = set()
        for p in snapshot.get('positions', []):
            if p.get('avg_entry_price') is None:
                self._null_basis.add(_norm(p['symbol']))
            self.positions[_norm(p['symbol'])] = {
                'symbol': p['symbol'], 'qty': _f(p['qty']),
                'avg_entry_price': _f(p['avg_entry_price']),
                'cost_basis': _f(p['cost_basis']),
                'asset_class': p.get('asset_class', 'crypto'),
                'snapshot_price': _f(p['current_price']),
            }
        if isinstance(tape, Tape):
            self.rtape, self.tape = tape, {}
        else:
            self.rtape = None
            self.tape = {_norm(k): list(v) for k, v in (tape or {}).items()}
        self.step = 0
        self.spread_bps = float(spread_bps)
        self.clock = clock or FakeClock()
        self.sessions = sessions
        self.replay_quote_age = replay_quote_age
        self.orders: dict[str, dict] = {}
        self._ids = itertools.count(1)
        self.events: list[dict] = []
        self.calls: list[tuple] = []
        # True = mimic the paper zero-basis quirk on NEW fills too: the
        # position keeps avg_entry_price=0/cost_basis=0 while the ORDER
        # carries a valid filled_avg_price.
        self.report_zero_basis = False
        # ENGINE r7 fault switches (module docstring, CONTROLLABLE FAULTS).
        self.report_null_basis = False
        self.report_null_fill_avg = False
        self.reserve_qty_on_resting_stops = True
        self.ns_timestamps = bool(ns_timestamps)
        self._hide: dict[str, tuple[int, int]] = {}
        self._cash_equity: tuple[int, int] | None = None
        for o in snapshot.get('open_orders', []) or []:
            oid = str(o.get('id') or f'snap-{next(self._ids)}')
            self.orders[oid] = dict(o, id=oid)

    def _call(self, name, arg=None):
        self.calls.append((name, arg, self.step))

    # ------------------------------------------------------- faults (ENGINE r7)
    def _window(self, cycles, at_step):
        start = self.step if at_step is None else int(at_step)
        return (start, start + int(cycles))

    def _in(self, w) -> bool:
        return w is not None and w[0] <= self.step < w[1]

    def hide_position(self, symbol, cycles: int = 1, at_step=None):
        """H1: hide ``symbol``'s position for ``cycles`` steps from
        ``at_step`` (default: now). Orders untouched (docstring)."""
        self._hide[_norm(symbol)] = self._window(cycles, at_step)

    def equity_equals_cash(self, cycles: int = 1, at_step=None):
        """H2: equity == cash and every position hidden for the window."""
        self._cash_equity = self._window(cycles, at_step)

    def fill_avg_none(self, position: bool = True, order: bool = False):
        """H3/H5: null basis on positions and/or null filled_avg_price on
        filled buy orders (docstring)."""
        if position:
            self.report_zero_basis = True
            self.report_null_basis = True
        if order:
            self.report_null_fill_avg = True

    def is_hidden(self, symbol) -> bool:
        return (self._in(self._hide.get(_norm(symbol)))
                or self._in(self._cash_equity))

    # ------------------------------------------------------------------ prices
    def _row(self, symbol):
        if self.rtape is not None and symbol in self.rtape:
            return self.rtape.row(symbol, self.step)
        return None

    def mid(self, symbol: str) -> float:
        r = self._row(symbol)
        if r is not None:
            return (r[1] + r[2]) / 2.0
        key = _norm(symbol)
        if key in self.tape and self.tape[key]:
            t = self.tape[key]
            return float(t[min(self.step, len(t) - 1)])
        return self.positions[key]['snapshot_price']

    def bid_ask(self, symbol: str) -> tuple[float, float]:
        r = self._row(symbol)
        if r is not None:
            return r[1], r[2]
        m = self.mid(symbol)
        h = m * self.spread_bps / 2e4
        return m - h, m + h

    def last(self, symbol: str) -> float:
        r = self._row(symbol)
        return r[3] if r is not None else self.mid(symbol)

    def _trigger_px(self, symbol: str) -> float:
        """Stop/trailing trigger: ``last`` on a Tape, the bid on a legacy
        mid tape (round-1 rule, kept byte-for-byte for its consumers)."""
        if self._row(symbol) is not None:
            return self.last(symbol)
        return self.bid_ask(symbol)[0]

    def session_open(self, when=None) -> bool:
        if self.sessions is None:
            return True
        when = when or self.clock()
        return any(o <= when < c for o, c in self.sessions)

    def _can_fill(self, symbol) -> bool:
        return _is_crypto(symbol) or self.session_open()

    def tick(self, n: int = 1, seconds: float = 30.0) -> None:
        """Advance the tape n steps (and the clock), expire stock day
        orders at a session close, then match resting orders."""
        for _ in range(n):
            before = self.clock()
            self.step += 1
            if hasattr(self.clock, 'advance'):
                self.clock.advance(seconds)
            if self.sessions is not None:
                after = self.clock()
                if any(before < c <= after for _, c in self.sessions):
                    self._expire_day_orders()
            self._match_resting()

    def _expire_day_orders(self):
        for o in self.orders.values():
            if (o['status'] in _OPEN and not _is_crypto(o['symbol'])
                    and o.get('time_in_force') == 'day'):
                o['status'] = 'expired'
                self._log('expire', o)

    # ------------------------------------------------------------------ account
    def _equity(self) -> float:
        return self.cash + sum(p['qty'] * self.mid(s)
                               for s, p in self.positions.items())

    def _open_buy_notional(self) -> float:
        tot = 0.0
        for o in self.orders.values():
            if o['status'] in _OPEN and o['side'] == 'buy':
                ref = o.get('limit_price') or self.bid_ask(o['symbol'])[1]
                tot += (o['qty'] - o['filled_qty']) * float(ref)
        return tot

    def _reserved_qty(self, symbol: str) -> float:
        if not self.reserve_qty_on_resting_stops:
            return 0.0
        tot, groups = 0.0, set()
        for o in self.orders.values():
            if not (o['status'] in _OPEN and o['side'] == 'sell'
                    and _norm(o['symbol']) == _norm(symbol)):
                continue
            g = o.get('oco_group')
            if g is not None:
                if g in groups:
                    continue
                groups.add(g)
            tot += o['qty'] - o['filled_qty']
        return tot

    def get_account(self):
        self._call('get_account')
        eq = self._equity()
        if self._in(self._cash_equity):
            eq = self.cash                     # H2: the equity == cash print
        nmbp = max(0.0, self.cash - self._open_buy_notional())
        a = dict(self._acct)
        a.update(cash=f'{self.cash:.2f}', equity=f'{eq:.2f}',
                 portfolio_value=f'{eq:.2f}',
                 last_equity=repr(self.last_equity),
                 non_marginable_buying_power=f'{nmbp:.2f}',
                 buying_power=f'{max(0.0, nmbp + self._bp_offset):.2f}',
                 long_market_value=f'{eq - self.cash:.2f}')
        return SimpleNamespace(**a)

    def get_clock(self):
        self._call('get_clock')
        now = self.clock()
        sess = sorted(self.sessions or [])
        is_open = self.session_open(now)
        nxt_open = next((o for o, _ in sess if o > now), None)
        nxt_close = next((c for _, c in sess if c > now), None)
        return SimpleNamespace(timestamp=now, is_open=is_open,
                               next_open=nxt_open, next_close=nxt_close)

    # ---------------------------------------------------------------- positions
    def _pos_obj(self, key: str):
        p = self.positions[key]
        px = self.mid(key)
        null = key in self._null_basis or (
            self.report_null_basis and p['avg_entry_price'] == 0)
        return SimpleNamespace(
            symbol=key, qty=repr(p['qty']),
            qty_available=repr(p['qty'] - self._reserved_qty(key)),
            avg_entry_price=None if null else repr(p['avg_entry_price']),
            cost_basis=None if null else repr(p['cost_basis']), side='long',
            asset_class=p['asset_class'], current_price=repr(px),
            market_value=repr(p['qty'] * px))

    def list_positions(self):
        self._call('list_positions')
        return [self._pos_obj(k) for k in sorted(self.positions)
                if self.positions[k]['qty'] > 0 and not self.is_hidden(k)]

    def get_position(self, symbol):
        self._call('get_position', symbol)
        key = _norm(symbol)
        if (key not in self.positions or self.positions[key]['qty'] <= 0
                or self.is_hidden(key)):
            raise FakeAPIError('position does not exist')
        return self._pos_obj(key)

    # ------------------------------------------------------------------- orders
    def _order_obj(self, o: dict):
        d = copy.deepcopy(o)
        for k in ('qty', 'filled_qty'):
            d[k] = repr(d[k])
        if d.get('filled_avg_price') is not None:
            d['filled_avg_price'] = repr(d['filled_avg_price'])
        if self.report_null_fill_avg and d.get('side') == 'buy':
            d['filled_avg_price'] = None       # H3: null order fill price
        d['order_type'] = d['type']
        if o.get('legs'):
            d['legs'] = [self._order_obj(self.orders[i]) for i in o['legs']]
        return SimpleNamespace(**d)

    def _log(self, kind, o=None, **extra):
        ev = {'kind': kind, 'step': self.step, 't': self.clock().isoformat()}
        if o is not None:
            ev.update({k: o.get(k) for k in ('id', 'symbol', 'side', 'type',
                                              'qty', 'limit_price',
                                              'stop_price', 'client_order_id')})
            for k in ('order_class', 'parent_id', 'trail_percent',
                      'time_in_force'):
                if o.get(k) is not None and (k != 'time_in_force'
                                             or not _is_crypto(o['symbol'])):
                    ev[k] = o[k]
        ev.update(extra)
        self.events.append(ev)

    def _reject(self, o, msg):
        self._log('reject', o, reason=msg)
        raise FakeAPIError(msg)

    def _new_order(self, symbol, qty, side, type, time_in_force, limit_price,
                   stop_price, client_order_id, notional=None, **extra):
        o = {'id': f'ord-{next(self._ids)}', 'client_order_id': client_order_id,
             'symbol': symbol, 'side': side, 'type': type,
             'time_in_force': time_in_force,
             'qty': float(qty) if qty is not None else None,
             'notional': notional,
             'limit_price': float(limit_price) if limit_price is not None else None,
             'stop_price': float(stop_price) if stop_price is not None else None,
             'status': 'new', 'filled_qty': 0.0, 'filled_avg_price': None,
             'triggered': False, 'submitted_at': self.clock().isoformat(),
             'filled_at': None, 'canceled_at': None}
        o.update(extra)
        return o

    def submit_order(self, symbol, qty=None, side=None, type=None,
                     time_in_force=None, limit_price=None, stop_price=None,
                     client_order_id=None, notional=None, order_class=None,
                     stop_loss=None, take_profit=None, trail_percent=None,
                     **kw):
        self._call('submit_order', symbol)
        o = self._new_order(symbol, qty, side, type, time_in_force,
                            limit_price, stop_price, client_order_id,
                            notional=notional)
        if order_class:
            o['order_class'] = order_class
        if trail_percent is not None:
            o['trail_percent'] = float(trail_percent)
        for fld in ('limit_price', 'stop_price'):
            if o[fld] is not None and not o[fld] > 0:
                self._reject(o, f'{fld} must be > 0')
        if type == 'trailing_stop' and not (o.get('trail_percent') or 0) > 0:
            self._reject(o, 'trail_percent must be > 0')
        bid, ask = self.bid_ask(symbol)
        if o['qty'] is None:
            if notional is None:
                raise FakeAPIError('qty or notional is required')
            o['qty'] = float(notional) / ask
        crypto = _is_crypto(symbol)
        if order_class == 'bracket':
            base = o['limit_price'] if type == 'limit' else ask
            sl = float((stop_loss or {}).get('stop_price') or 0)
            tp = float((take_profit or {}).get('limit_price') or 0)
            if side != 'buy' or not stop_loss or not take_profit:
                self._reject(o, 'bracket orders require side=buy with '
                                'take_profit and stop_loss')
            if tp < base + 0.01 - 1e-9:
                self._reject(o, f'take_profit.limit_price must be >= '
                                f'base_price + 0.01 (tp {tp}, base {base})')
            if sl > base - 0.01 + 1e-9 or sl <= 0:
                self._reject(o, f'stop_loss.stop_price must be <= '
                                f'base_price - 0.01 (stop {sl}, base {base})')
        if side == 'buy':
            ref = o['limit_price'] if type == 'limit' else ask
            need = o['qty'] * ref
            if crypto:
                avail = max(0.0, self.cash - self._open_buy_notional())
                msg = (f'insufficient balance for USD (requested: {need:.2f},'
                       f' available: {avail:.2f})')
            else:
                avail = max(0.0, self.cash + self._bp_offset
                            - self._open_buy_notional())
                msg = 'insufficient buying power'
            if need > avail + 1e-9:
                self._reject(o, msg)
        else:
            key = _norm(symbol)
            held = self.positions.get(key, {}).get('qty', 0.0)
            if self.is_hidden(key):
                held = 0.0                     # H1/H2: broker sees nothing
            avail = held - self._reserved_qty(symbol)
            if o['qty'] > avail + 1e-12:
                if crypto:
                    asset = key[:-3] if key.endswith('USD') else key
                    msg = (f'insufficient balance for {asset} (requested: '
                           f'{o["qty"]}, available: {max(avail, 0.0):g})')
                else:
                    msg = (f'insufficient qty available for order (requested:'
                           f' {o["qty"]:g}, available: {max(avail, 0.0):g})')
                self._reject(o, msg)
        if type == 'trailing_stop':
            o['hwm'] = self._trigger_px(symbol)
        self.orders[o['id']] = o
        self._log('submit', o, **({} if crypto else
                                  {'mkt_open': self.session_open()}))
        if order_class == 'bracket':
            tif = time_in_force
            legs = []
            tpo = self._new_order(symbol, o['qty'], 'sell', 'limit', tif,
                                  take_profit['limit_price'], None,
                                  f'{client_order_id}-tp' if client_order_id
                                  else None, parent_id=o['id'],
                                  oco_group=o['id'], leg='take_profit')
            slt = 'stop_limit' if stop_loss.get('limit_price') else 'stop'
            slo = self._new_order(symbol, o['qty'], 'sell', slt, tif,
                                  stop_loss.get('limit_price'),
                                  stop_loss['stop_price'],
                                  f'{client_order_id}-sl' if client_order_id
                                  else None, parent_id=o['id'],
                                  oco_group=o['id'], leg='stop_loss')
            for leg in (tpo, slo):
                leg['status'] = 'held'
                self.orders[leg['id']] = leg
                legs.append(leg['id'])
                self._log('leg', leg)
            o['legs'] = legs
        self._try_fill(o)
        return self._order_obj(o)

    def _fill(self, o, px):
        q = o['qty'] - o['filled_qty']
        key = _norm(o['symbol'])
        if o['side'] == 'buy':
            self.cash -= q * px
            p = self.positions.setdefault(key, {
                'symbol': key, 'qty': 0.0, 'avg_entry_price': 0.0,
                'cost_basis': 0.0,
                'asset_class': 'crypto' if _is_crypto(key) else 'us_equity',
                'snapshot_price': px})
            p['qty'] += q
            if not self.report_zero_basis:
                self._null_basis.discard(key)
                p['cost_basis'] += q * px
                p['avg_entry_price'] = p['cost_basis'] / p['qty']
        else:
            self.cash += q * px
            p = self.positions[key]
            if p['qty'] > 0:
                p['cost_basis'] *= max(0.0, (p['qty'] - q) / p['qty'])
            p['qty'] -= q
            if p['qty'] <= 1e-12:
                del self.positions[key]
        o['filled_qty'] = o['qty']
        o['filled_avg_price'] = px
        o['status'] = 'filled'
        o['filled_at'] = self.clock().isoformat()
        self._log('fill', o, fill_price=px)
        for lid in o.get('legs') or []:          # parent filled: arm legs
            leg = self.orders[lid]
            if leg['status'] == 'held':
                leg['status'] = 'new'
        g = o.get('oco_group')
        if g is not None and o.get('parent_id'):  # leg filled: OCO cancel
            for sib in self.orders.values():
                if (sib.get('oco_group') == g and sib is not o
                        and sib['status'] in _OPEN):
                    sib['status'] = 'canceled'
                    sib['canceled_at'] = self.clock().isoformat()
                    self._log('cancel', sib, reason='oco')

    def _try_fill(self, o):
        if o['status'] not in _OPEN or o['status'] == 'held':
            return
        if not self._can_fill(o['symbol']):
            return
        bid, ask = self.bid_ask(o['symbol'])
        t = o['type']
        if t == 'market':
            self._fill(o, ask if o['side'] == 'buy' else bid)
        elif t == 'limit':
            if o['side'] == 'buy' and o['limit_price'] >= ask:
                self._fill(o, ask)
            elif o['side'] == 'sell' and o['limit_price'] <= bid:
                self._fill(o, bid)
        elif t in ('stop_limit', 'stop') and o['side'] == 'sell':
            trig = self._trigger_px(o['symbol'])
            if not o['triggered'] and trig <= o['stop_price']:
                o['triggered'] = True
                self._log('trigger', o, bid=bid, trigger_px=trig)
            if o['triggered'] and (t == 'stop' or bid >= o['limit_price']):
                self._fill(o, bid)
        elif t == 'trailing_stop' and o['side'] == 'sell':
            trig = self._trigger_px(o['symbol'])
            o['hwm'] = max(o.get('hwm') or trig, trig)
            if trig <= o['hwm'] * (1 - o['trail_percent'] / 100.0):
                o['triggered'] = True
                self._log('trigger', o, bid=bid, trigger_px=trig)
                self._fill(o, bid)

    def _match_resting(self):
        for o in list(self.orders.values()):
            self._try_fill(o)

    def get_order(self, order_id, **kw):
        self._call('get_order', order_id)
        if order_id not in self.orders:
            raise FakeAPIError('order not found')
        return self._order_obj(self.orders[order_id])

    def list_orders(self, status='open', limit=None, symbols=None, **kw):
        """status 'open' keeps insertion order (round-1 behaviour);
        'closed'/'all' are newest-first like Alpaca's default direction."""
        self._call('list_orders', status)
        out = []
        want = {_norm(s) for s in symbols} if symbols else None
        for o in self.orders.values():
            if status == 'open' and o['status'] not in _OPEN:
                continue
            if status == 'closed' and o['status'] not in _TERMINAL:
                continue
            if want is not None and _norm(o['symbol']) not in want:
                continue
            out.append(self._order_obj(o))
        if status in ('closed', 'all'):
            out.reverse()
        return out[:limit] if limit else out

    def _cancel(self, o, reason=None):
        o['status'] = 'canceled'
        o['canceled_at'] = self.clock().isoformat()
        self._log('cancel', o, **({'reason': reason} if reason else {}))

    def cancel_order(self, order_id):
        self._call('cancel_order', order_id)
        o = self.orders.get(order_id)
        if o is None:
            raise FakeAPIError('order not found')
        if o['status'] not in _OPEN:
            raise FakeAPIError('order is not cancelable')
        self._cancel(o)
        g = o.get('oco_group')
        for other in self.orders.values():
            if other is o or other['status'] not in _OPEN:
                continue
            if (g is not None and other.get('oco_group') == g) or \
                    other.get('parent_id') == order_id:
                self._cancel(other, reason='oco' if g is not None else 'parent')

    def cancel_all_orders(self):
        self._call('cancel_all_orders')
        self._log('cancel_all')
        for o in self.orders.values():
            if o['status'] in _OPEN:
                o['status'] = 'canceled'
                o['canceled_at'] = self.clock().isoformat()
                self._log('cancel', o)

    # ------------------------------------------------------------------- quotes
    def _quote(self, symbol):
        bid, ask = self.bid_ask(symbol)
        t = self.clock()
        r = self._row(symbol)
        if self.replay_quote_age and r is not None and r[4]:
            t = t - _dt.timedelta(seconds=r[4])
        if self.ns_timestamps:
            import pandas as pd                # lazy: the ns (live SDK) path
            t = pd.Timestamp(t) + pd.Timedelta(123, 'ns')
        return SimpleNamespace(bp=bid, ap=ask, bs=1.0, as_=1.0, t=t)

    def get_latest_crypto_quotes(self, symbols, **kw):
        self._call('get_latest_crypto_quotes', tuple(symbols))
        return {s: self._quote(s) for s in symbols}

    def get_latest_quote(self, symbol, **kw):
        self._call('get_latest_quote', symbol)
        return self._quote(symbol)

    # ------------------------------------------------------------ test helpers
    def events_of(self, kind, symbol=None):
        return [e for e in self.events if e['kind'] == kind
                and (symbol is None or _norm(e.get('symbol') or '') == _norm(symbol))]

    def open_orders(self, symbol=None):
        return [o for o in self.orders.values() if o['status'] in _OPEN
                and (symbol is None or _norm(o['symbol']) == _norm(symbol))]

    def normalized_events(self):
        """Events with the random uuid tail of client_order_id reduced to
        its tag (make_client_order_id = '<tag>-<uuid4 hex>') — the
        determinism comparison key."""
        out = []
        for e in self.events:
            e = dict(e)
            if e.get('client_order_id'):
                e['client_order_id'] = e['client_order_id'].split('-')[0]
            out.append(e)
        return out


def apply_live_modes(monkeypatch, workdir, no_model: bool = False,
                     halt: bool = False) -> dict:
    """Reproduce the live 2026-09-27 03:09 startup configuration (W16 F2c)
    around the REAL loop: ``no_model`` chdirs into an empty directory so
    predict_now.load_models' cwd-relative artifact paths are absent and
    BaseTradingLoop._load_models takes its fail-closed FileNotFoundError
    branch; the halt flag is ALWAYS redirected to ``workdir`` (so the repo's
    own trading_halt.flag can never leak into a test) and created when
    ``halt``. The caller must leave notify.halt_active unstubbed and use the
    real _load_models / _get_predictions. Returns the paths used."""
    import notify                                  # lazy: repo module
    workdir = Path(workdir)
    out = {}
    if no_model:
        d = workdir / 'no_model_cwd'
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(d)
        out['model_dir'] = d
    flag = workdir / 'trading_halt.flag'
    monkeypatch.setattr(notify, '_HALT_FLAG', str(flag))
    if halt:
        flag.write_text('halt (fake_alpaca_broker live mode)\n')
    out['halt_flag'] = flag
    return out


# ---------------------------------------------------------------------------
# Read-only tape recorder (ENGINE r2 W6). Duck-typed on the legacy SDK's
# get_latest_crypto_quotes / get_latest_crypto_trades; stdlib-only; GETs only.
# ---------------------------------------------------------------------------

TAPE_SCHEMA = 'fake_alpaca_broker.tape/v1'


def record_crypto_tape(api, symbols, n_samples=20, interval_s=30.0,
                       sleep=None, now=None):
    """Sample latest quote + trade for ``symbols`` ``n_samples`` times,
    ``interval_s`` apart, into the tape JSON dict (schema: module
    docstring). Read-only market-data GETs; a failed sample for a symbol
    is skipped (row count may differ per symbol)."""
    sleep = sleep or _time_mod.sleep
    now = now or (lambda: _dt.datetime.now(_dt.timezone.utc))
    rows = {s: [] for s in symbols}
    for i in range(n_samples):
        if i:
            sleep(interval_s)
        ts = now()
        try:
            quotes = api.get_latest_crypto_quotes(list(symbols))
        except Exception:
            quotes = {}
        try:
            trades = api.get_latest_crypto_trades(list(symbols))
        except Exception:
            trades = {}
        for s in symbols:
            q = quotes.get(s) if hasattr(quotes, 'get') else None
            if q is None:
                continue
            try:
                bid, ask = float(q.bp), float(q.ap)
            except (TypeError, ValueError, AttributeError):
                continue
            tr = trades.get(s) if hasattr(trades, 'get') else None
            try:
                last = float(tr.p)
            except (TypeError, ValueError, AttributeError):
                last = (bid + ask) / 2.0
            age = None
            qt = getattr(q, 't', None)
            try:
                if hasattr(qt, 'to_pydatetime'):
                    qt = qt.to_pydatetime()
                if qt is not None:
                    if qt.tzinfo is None:
                        qt = qt.replace(tzinfo=_dt.timezone.utc)
                    age = round((ts - qt).total_seconds(), 1)
            except Exception:
                age = None
            rows[s].append([ts.isoformat(timespec='seconds'), bid, ask, last,
                            age])
    return {'schema': TAPE_SCHEMA,
            'recorded_at': now().isoformat(timespec='seconds'),
            'source': 'alpaca latest crypto quote+trade (read-only)',
            'interval_s': float(interval_s), 'symbols': rows}
