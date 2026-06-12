"""alpaca-py adapter exposing the legacy alpaca-trade-api REST surface.

alpaca-trade-api is unmaintained (last release Jan 2024, maintenance ended
2022) and its dependency pins (websockets<11, urllib3<2) conflict with this
repo's own requirements — pip silently downgrades today and will eventually
fail outright. This adapter lets the entire codebase keep its existing
`api.submit_order(...)` / `api.get_bars(...)` call sites while running on
the maintained alpaca-py SDK underneath.

Selection logic lives in trading_utils.get_api():
  - TRADER_USE_ALPACA_PY=1  -> force this adapter
  - alpaca-trade-api import fails (dependency rot finally lands) -> adapter
  - otherwise -> legacy SDK (battle-tested in this repo)

Every returned object is a thin shim exposing the legacy attribute names
(bar.o/.h/.l/.c/.v/.t, quote.bp/.ap, order.filled_avg_price, ...).
"""

from types import SimpleNamespace


def _shim_order(o):
    status = getattr(o, 'status', None)
    return SimpleNamespace(
        id=str(o.id),
        client_order_id=getattr(o, 'client_order_id', None),
        symbol=o.symbol,
        qty=float(o.qty) if o.qty is not None else None,
        side=getattr(getattr(o, 'side', None), 'value', str(getattr(o, 'side', ''))),
        type=getattr(getattr(o, 'order_type', None), 'value',
                     str(getattr(o, 'order_type', ''))),
        status=getattr(status, 'value', str(status) if status else None),
        filled_qty=float(o.filled_qty) if getattr(o, 'filled_qty', None) is not None else 0.0,
        filled_avg_price=(float(o.filled_avg_price)
                          if getattr(o, 'filled_avg_price', None) is not None else None),
        legs=[_shim_order(l) for l in (getattr(o, 'legs', None) or [])],
    )


def _shim_position(p):
    return SimpleNamespace(
        symbol=p.symbol,
        qty=float(p.qty),
        avg_entry_price=float(p.avg_entry_price),
        current_price=(float(p.current_price)
                       if getattr(p, 'current_price', None) is not None else None),
        market_value=(float(p.market_value)
                      if getattr(p, 'market_value', None) is not None else 0.0),
        unrealized_pl=(float(p.unrealized_pl)
                       if getattr(p, 'unrealized_pl', None) is not None else 0.0),
    )


def _shim_bar(b):
    return SimpleNamespace(o=b.open, h=b.high, l=b.low, c=b.close,
                           v=b.volume, t=b.timestamp)


class CompatREST:
    """Legacy alpaca-trade-api REST interface over alpaca-py clients."""

    def __init__(self, key, secret, base_url=None, api_version='v2'):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import (StockHistoricalDataClient,
                                            CryptoHistoricalDataClient)
        paper = bool(base_url) and 'paper' in str(base_url)
        self._trading = TradingClient(key, secret, paper=paper)
        self._stock_data = StockHistoricalDataClient(key, secret)
        self._crypto_data = CryptoHistoricalDataClient(key, secret)

    # --- Orders ---

    def submit_order(self, symbol, qty=None, side='buy', type='market',
                     time_in_force='gtc', limit_price=None, stop_price=None,
                     trail_percent=None, notional=None, order_class=None,
                     take_profit=None, stop_loss=None, client_order_id=None,
                     **_ignored):
        from alpaca.trading.requests import (
            MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
            TrailingStopOrderRequest, TakeProfitRequest, StopLossRequest)
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        kwargs = dict(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce(time_in_force),
            client_order_id=client_order_id,
        )
        if qty is not None:
            kwargs['qty'] = qty
        if notional is not None:
            kwargs['notional'] = notional
        if order_class == 'bracket':
            kwargs['order_class'] = OrderClass.BRACKET
            if take_profit:
                kwargs['take_profit'] = TakeProfitRequest(
                    limit_price=take_profit['limit_price'])
            if stop_loss:
                sl = {'stop_price': stop_loss['stop_price']}
                if 'limit_price' in stop_loss:
                    sl['limit_price'] = stop_loss['limit_price']
                kwargs['stop_loss'] = StopLossRequest(**sl)

        if type == 'market':
            req = MarketOrderRequest(**kwargs)
        elif type == 'limit':
            req = LimitOrderRequest(limit_price=limit_price, **kwargs)
        elif type == 'stop':
            req = StopOrderRequest(stop_price=stop_price, **kwargs)
        elif type == 'trailing_stop':
            req = TrailingStopOrderRequest(trail_percent=trail_percent, **kwargs)
        else:
            raise ValueError(f"unsupported order type: {type}")
        return _shim_order(self._trading.submit_order(req))

    def get_order(self, order_id):
        return _shim_order(self._trading.get_order_by_id(order_id))

    @staticmethod
    def _parse_dt(v):
        from datetime import datetime
        if v is None or isinstance(v, datetime):
            return v
        return datetime.fromisoformat(str(v).replace('Z', '+00:00'))

    def list_orders(self, status='open', symbols=None, limit=None,
                    after=None, until=None, direction=None, **_ignored):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        # after/until/direction were silently swallowed by **_ignored —
        # the GUI's clean-slate cutoff (list_orders(after=...)) returned
        # the full order history under the adapter
        req = GetOrdersRequest(
            status=QueryOrderStatus(status) if status else None,
            symbols=list(symbols) if symbols else None,
            limit=limit,
            after=self._parse_dt(after),
            until=self._parse_dt(until),
            direction=direction,
        )
        return [_shim_order(o) for o in self._trading.get_orders(req)]

    def cancel_order(self, order_id):
        self._trading.cancel_order_by_id(order_id)

    def cancel_all_orders(self):
        return self._trading.cancel_orders()

    # --- Positions / account ---

    def list_positions(self):
        return [_shim_position(p) for p in self._trading.get_all_positions()]

    def get_position(self, symbol):
        return _shim_position(self._trading.get_open_position(symbol))

    def close_all_positions(self, cancel_orders=True):
        return self._trading.close_all_positions(cancel_orders=cancel_orders)

    def get_account(self):
        a = self._trading.get_account()
        return SimpleNamespace(
            equity=float(a.equity),
            last_equity=float(a.last_equity),
            buying_power=float(a.buying_power),
            status=getattr(getattr(a, 'status', None), 'value',
                           str(getattr(a, 'status', ''))),
            trading_blocked=bool(getattr(a, 'trading_blocked', False)),
        )

    def get_clock(self):
        c = self._trading.get_clock()
        return SimpleNamespace(is_open=c.is_open, next_open=c.next_open,
                               next_close=c.next_close, timestamp=c.timestamp)

    def get_calendar(self, start=None, end=None):
        from alpaca.trading.requests import GetCalendarRequest
        req = GetCalendarRequest(start=start, end=end)
        return self._trading.get_calendar(req)

    def get_portfolio_history(self, period='1M', timeframe='1D',
                              extended_hours=None, **_ignored):
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe,
                                         extended_hours=extended_hours)
        h = self._trading.get_portfolio_history(req)
        return SimpleNamespace(
            equity=[float(e) for e in (h.equity or []) if e is not None],
            timestamp=list(h.timestamp or []),
        )

    # --- Market data ---

    @staticmethod
    def _timeframe(tf: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        mapping = {
            '1Min': TimeFrame.Minute,
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day,
        }
        if tf in mapping:
            return mapping[tf]
        if tf.endswith('Min'):
            return TimeFrame(int(tf[:-3]), TimeFrameUnit.Minute)
        if tf.endswith('Hour'):
            return TimeFrame(int(tf[:-4]), TimeFrameUnit.Hour)
        return TimeFrame.Hour

    def get_bars(self, symbol, timeframe, start=None, end=None, limit=None,
                 adjustment='raw', **_ignored):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import Adjustment
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(timeframe),
            start=start, end=end, limit=limit,
            adjustment=Adjustment(adjustment) if adjustment else None,
        )
        bars = self._stock_data.get_stock_bars(req)
        return [_shim_bar(b) for b in bars.data.get(symbol, [])]

    def get_crypto_bars(self, symbol, timeframe, start=None, end=None,
                        limit=None, **_ignored):
        from alpaca.data.requests import CryptoBarsRequest
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(timeframe),
            start=start, end=end, limit=limit,
        )
        bars = self._crypto_data.get_crypto_bars(req)
        return [_shim_bar(b) for b in bars.data.get(symbol, [])]

    def get_latest_quote(self, symbol):
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        q = self._stock_data.get_stock_latest_quote(req)[symbol]
        return SimpleNamespace(bp=q.bid_price, ap=q.ask_price)

    def get_latest_crypto_quotes(self, symbols):
        from alpaca.data.requests import CryptoLatestQuoteRequest
        req = CryptoLatestQuoteRequest(symbol_or_symbols=list(symbols))
        quotes = self._crypto_data.get_crypto_latest_quote(req)
        return {sym: SimpleNamespace(bp=q.bid_price, ap=q.ask_price)
                for sym, q in quotes.items()}
