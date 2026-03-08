"""Structured data types for the trading system.

Replaces raw dicts with typed dataclasses for positions, quotes, LLM results,
and macro regime data. Provides both type safety and self-documenting code.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Position:
    qty: float
    entry_price: float
    high_water_mark: float
    stop_order_id: str | None = None
    trailing_activated: bool = False
    entry_atr: float | None = None
    take_profit_price: float | None = None
    garch_sigma: float | None = None

    def to_dict(self) -> dict:
        return {
            'qty': self.qty,
            'entry_price': self.entry_price,
            'high_water_mark': self.high_water_mark,
            'stop_order_id': self.stop_order_id,
            'trailing_activated': self.trailing_activated,
            'entry_atr': self.entry_atr,
            'take_profit_price': self.take_profit_price,
            'garch_sigma': self.garch_sigma,
        }


@dataclass(slots=True)
class Quote:
    bid: float
    ask: float
    spread: float
    midpoint: float
    spread_pct: float

    def to_dict(self) -> dict:
        return {
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'midpoint': self.midpoint,
            'spread_pct': self.spread_pct,
        }


@dataclass(slots=True)
class LLMResult:
    s: float
    m: float
    r: str
    bull: str = ""
    bear: str = ""


@dataclass(slots=True)
class MacroRegime:
    stress_level: float | None
    vix: float | None
    cape: float | None
    regime_label: str
    sizing_mult: float = 1.0
    stop_mult: float = 1.0
    stablecoin_alert: bool = False

    @property
    def is_defensive(self) -> bool:
        return self.sizing_mult <= 0.8

    @property
    def should_halt_stocks(self) -> bool:
        return self.vix is not None and self.vix > 35

    @property
    def should_block_risky_entries(self) -> bool:
        """Block new entries in high-beta / speculative names when VIX > 25."""
        return self.vix is not None and self.vix > 25
