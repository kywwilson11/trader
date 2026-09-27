"""Synthetic-data unit tests for tax_lots.py — pure stdlib (datetime +
collections), runs on the dev Mac. No Alpaca/Qt/pandas anywhere."""

import datetime as dt
import types

import pytest

from tax_lots import _field, _is_long_term, _mintax_sort_key, estimate_taxes

BASE = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)


def _iso(t):
    """Format a datetime as an Alpaca-style ISO8601 string with trailing Z."""
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_order(symbol, side, qty, price, filled_at, status="filled"):
    """Plain dict order shaped like gui.py's DataFetcher.fetch_orders() output."""
    return {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "type": "market",
        "status": status,
        "submitted_at": filled_at,
        "filled_at": filled_at,
        "filled_avg_price": price,
        "notional": None,
        "filled_qty": qty,
    }


# 1) Basic single-lot matching
class TestSingleLot:
    def test_single_buy_sell_gain(self):
        orders = [
            make_order("AAPL", "buy", 10, 100.0, _iso(BASE)),
            make_order("AAPL", "sell", 10, 150.0, _iso(BASE + dt.timedelta(days=10))),
        ]
        result = estimate_taxes(orders)
        assert result["realized_gain"] == pytest.approx(500.0)
        assert result["short_term_gain"] == pytest.approx(500.0)
        assert result["long_term_gain"] == pytest.approx(0.0)
        assert result["num_trades"] == 1
        assert result["basis_complete"] is True
        assert result["unmatched_sell_qty"] == pytest.approx(0.0)

    def test_single_buy_sell_loss_generates_no_tax_rebate(self):
        orders = [
            make_order("AAPL", "buy", 10, 100.0, _iso(BASE)),
            make_order("AAPL", "sell", 10, 80.0, _iso(BASE + dt.timedelta(days=10))),
        ]
        result = estimate_taxes(orders)
        assert result["realized_gain"] == pytest.approx(-200.0)
        assert result["estimated_tax"] == pytest.approx(0.0)
        assert result["net_after_tax"] == pytest.approx(-200.0)


# 2) MinTax multi-lot priority ordering
class TestMinTaxPriority:
    def test_loss_before_long_term_before_short_term(self):
        """One sell spans three lots in three different tiers; the leftover
        (unsold) quantity must come from the LOWEST-priority tier (ST),
        proving loss > long-term-gain > short-term-gain priority."""
        sell_time = BASE + dt.timedelta(days=400)
        orders = [
            make_order("XYZ", "buy", 5, 50.0, _iso(BASE)),                                  # LT gain lot
            make_order("XYZ", "buy", 5, 95.0, _iso(sell_time - dt.timedelta(days=10))),     # ST loss lot
            make_order("XYZ", "buy", 5, 60.0, _iso(sell_time - dt.timedelta(days=5))),      # ST gain lot
            make_order("XYZ", "sell", 6, 80.0, _iso(sell_time)),
        ]
        result = estimate_taxes(orders)
        # loss lot (5 units) fully consumed first, then exactly 1 more unit
        # from the long-term lot — the short-term gain lot must be untouched.
        assert result["num_trades"] == 2
        assert result["short_term_gain"] == pytest.approx((80 - 95) * 5)
        assert result["long_term_gain"] == pytest.approx((80 - 50) * 1)
        assert result["realized_gain"] == pytest.approx((80 - 95) * 5 + (80 - 50) * 1)

    def test_highest_cost_basis_within_tier_first(self):
        """Two same-tier (short-term gain) lots at different cost bases;
        MinTax must prefer the higher-cost lot first (smaller recognized
        gain) — a naive lowest-cost-first or FIFO order would give 110, not 70."""
        sell_time = BASE + dt.timedelta(days=10)
        orders = [
            make_order("QQQ", "buy", 5, 60.0, _iso(BASE)),
            make_order("QQQ", "buy", 5, 70.0, _iso(BASE + dt.timedelta(days=1))),
            make_order("QQQ", "sell", 6, 80.0, _iso(sell_time)),
        ]
        result = estimate_taxes(orders)
        assert result["realized_gain"] == pytest.approx(70.0)
        assert result["short_term_gain"] == pytest.approx(70.0)
        assert result["long_term_gain"] == pytest.approx(0.0)
        assert result["num_trades"] == 2


# 3) _mintax_sort_key helper directly
class TestMintaxSortKeyHelper:
    def test_loss_sorts_before_gains(self):
        loss_lot = {"price": 100.0, "time": _iso(BASE), "qty": 1}
        gain_lot = {"price": 50.0, "time": _iso(BASE), "qty": 1}
        sell_time = _iso(BASE + dt.timedelta(days=1))
        assert _mintax_sort_key(loss_lot, 80.0, sell_time) < _mintax_sort_key(gain_lot, 80.0, sell_time)

    def test_long_term_gain_sorts_before_short_term_gain(self):
        lt_lot = {"price": 50.0, "time": _iso(BASE), "qty": 1}
        st_lot = {"price": 50.0, "time": _iso(BASE + dt.timedelta(days=390)), "qty": 1}
        sell_time = _iso(BASE + dt.timedelta(days=400))
        assert _mintax_sort_key(lt_lot, 80.0, sell_time) < _mintax_sort_key(st_lot, 80.0, sell_time)

    def test_higher_cost_basis_sorts_first_within_tier(self):
        cheap = {"price": 60.0, "time": _iso(BASE), "qty": 1}
        pricier = {"price": 70.0, "time": _iso(BASE), "qty": 1}
        sell_time = _iso(BASE + dt.timedelta(days=1))
        assert _mintax_sort_key(pricier, 80.0, sell_time) < _mintax_sort_key(cheap, 80.0, sell_time)


# 4) Long-term boundary (the sanctioned >365 fix)
class TestLongTermBoundary:
    def test_exactly_365_days_is_short_term(self):
        sell_time = BASE + dt.timedelta(days=365)
        orders = [
            make_order("AAA", "buy", 1, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 1, 200.0, _iso(sell_time)),
        ]
        result = estimate_taxes(orders)
        assert result["short_term_gain"] == pytest.approx(100.0)
        assert result["long_term_gain"] == pytest.approx(0.0)

    def test_366_days_is_long_term(self):
        sell_time = BASE + dt.timedelta(days=366)
        orders = [
            make_order("AAA", "buy", 1, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 1, 200.0, _iso(sell_time)),
        ]
        result = estimate_taxes(orders)
        assert result["long_term_gain"] == pytest.approx(100.0)
        assert result["short_term_gain"] == pytest.approx(0.0)

    def test_is_long_term_helper_boundary_directly(self):
        assert _is_long_term(BASE, BASE + dt.timedelta(days=365)) is False
        assert _is_long_term(BASE, BASE + dt.timedelta(days=366)) is True

    def test_is_long_term_missing_timestamps_are_short_term(self):
        assert _is_long_term(None, BASE) is False
        assert _is_long_term(BASE, None) is False


# 5) Unmatched sells (counted instead of silently dropped)
class TestUnmatchedSells:
    def test_sell_with_no_prior_buy_is_fully_unmatched(self):
        orders = [make_order("ZZZ", "sell", 10, 50.0, _iso(BASE))]
        result = estimate_taxes(orders)
        assert result["unmatched_sell_qty"] == pytest.approx(10.0)
        assert result["basis_complete"] is False
        assert result["realized_gain"] == pytest.approx(0.0)
        assert result["num_trades"] == 0

    def test_partially_matched_sell_counts_the_remainder(self):
        orders = [
            make_order("QQQ", "buy", 5, 40.0, _iso(BASE)),
            make_order("QQQ", "sell", 10, 60.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        result = estimate_taxes(orders)
        assert result["unmatched_sell_qty"] == pytest.approx(5.0)
        assert result["basis_complete"] is False
        assert result["realized_gain"] == pytest.approx((60 - 40) * 5)

    def test_fully_matched_sells_leave_basis_complete_true(self):
        orders = [
            make_order("AAA", "buy", 1, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 1, 150.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        result = estimate_taxes(orders)
        assert result["unmatched_sell_qty"] == pytest.approx(0.0)
        assert result["basis_complete"] is True


# 6) window_truncated flag
class TestWindowTruncated:
    def test_window_truncated_forces_incomplete_even_if_all_matched(self):
        orders = [
            make_order("AAA", "buy", 1, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 1, 150.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        truncated = estimate_taxes(orders, window_truncated=True)
        complete = estimate_taxes(orders, window_truncated=False)
        assert truncated["basis_complete"] is False
        assert complete["basis_complete"] is True
        # it's purely a completeness flag — must not change the tax arithmetic
        assert truncated["realized_gain"] == pytest.approx(complete["realized_gain"])


# 7) Rate parametrization
class TestRateParametrization:
    def test_default_rates_match_documented_constants(self):
        orders = [
            make_order("AAA", "buy", 10, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 10, 200.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        result = estimate_taxes(orders)
        assert result["estimated_tax"] == pytest.approx(1000.0 * (0.37 + 0.05))

    def test_custom_short_and_state_rates_change_tax(self):
        orders = [
            make_order("AAA", "buy", 10, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 10, 200.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        cheap = estimate_taxes(orders, fed_short=0.10, state_rate=0.0)
        assert cheap["estimated_tax"] == pytest.approx(1000.0 * 0.10)
        assert cheap["net_after_tax"] == pytest.approx(1000.0 - 100.0)

    def test_custom_long_term_rate_changes_tax(self):
        orders = [
            make_order("AAA", "buy", 10, 100.0, _iso(BASE)),
            make_order("AAA", "sell", 10, 200.0, _iso(BASE + dt.timedelta(days=400))),
        ]
        result = estimate_taxes(orders, fed_long=0.0, state_rate=0.0)
        assert result["estimated_tax"] == pytest.approx(0.0)
        assert result["net_after_tax"] == pytest.approx(1000.0)


# 8) End-to-end with attribute-style objects (as opposed to plain dicts)
class TestEndToEndAttributeStyleOrders:
    def test_simplenamespace_orders_like_a_raw_sdk_object(self):
        """Mirrors the shapes gui.py's DataFetcher builds today (dicts), but
        via attribute access, proving estimate_taxes is genuinely duck-typed
        and not hardwired to dict subscripting."""
        buy = types.SimpleNamespace(
            symbol="BTCUSD", side="buy", qty="0.5", type="market",
            status="filled", submitted_at=_iso(BASE), filled_at=_iso(BASE),
            filled_avg_price="30000.0", notional=None, filled_qty="0.5",
        )
        sell = types.SimpleNamespace(
            symbol="BTCUSD", side="sell", qty="0.5", type="market",
            status="filled",
            submitted_at=_iso(BASE + dt.timedelta(days=31)),
            filled_at=_iso(BASE + dt.timedelta(days=31)),
            filled_avg_price="35000.0", notional=None, filled_qty="0.5",
        )
        open_order = types.SimpleNamespace(
            symbol="ETHUSD", side="buy", qty="1", type="market",
            status="new", submitted_at=_iso(BASE), filled_at=None,
            filled_avg_price=None, notional=None, filled_qty=None,
        )
        result = estimate_taxes(
            [buy, sell, open_order], crypto_symbols=frozenset({"BTCUSD", "ETHUSD"}),
        )
        assert result["realized_gain"] == pytest.approx((35000.0 - 30000.0) * 0.5)
        assert result["short_term_gain"] == pytest.approx((35000.0 - 30000.0) * 0.5)
        assert result["basis_complete"] is True
        assert result["unmatched_sell_qty"] == pytest.approx(0.0)
        assert result["num_trades"] == 1


# 9) Filtering / ordering behavior carried over from the original
class TestFilteringAndOrdering:
    def test_non_filled_orders_are_ignored(self):
        orders = [
            make_order("AAA", "buy", 10, 100.0, _iso(BASE), status="new"),
            make_order("AAA", "sell", 10, 200.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        result = estimate_taxes(orders)
        # the buy never filled, so the sell has no lot to match
        assert result["unmatched_sell_qty"] == pytest.approx(10.0)
        assert result["realized_gain"] == pytest.approx(0.0)

    def test_zero_qty_orders_are_skipped(self):
        orders = [
            make_order("AAA", "buy", 0, 100.0, _iso(BASE)),
            make_order("AAA", "buy", 10, 90.0, _iso(BASE)),
            make_order("AAA", "sell", 10, 150.0, _iso(BASE + dt.timedelta(days=1))),
        ]
        result = estimate_taxes(orders)
        assert result["realized_gain"] == pytest.approx((150 - 90) * 10)

    def test_orders_out_of_chronological_order_in_the_input_still_match(self):
        orders = [
            make_order("AAA", "sell", 10, 200.0, _iso(BASE + dt.timedelta(days=1))),
            make_order("AAA", "buy", 10, 100.0, _iso(BASE)),
        ]
        result = estimate_taxes(orders)
        assert result["realized_gain"] == pytest.approx(1000.0)
        assert result["basis_complete"] is True


# 10) _field duck-typed accessor
class TestFieldHelper:
    def test_field_dict_access(self):
        assert _field({"a": 1}, "a") == 1
        assert _field({"a": 1}, "b", "default") == "default"

    def test_field_attribute_access(self):
        obj = types.SimpleNamespace(a=1)
        assert _field(obj, "a") == 1
        assert _field(obj, "b", "default") == "default"


# ---------------------------------------------------------------------------
# 2026-09 G8 hunt (Jetson): fills on non-"filled" statuses, leap-year
# long-term boundary, float dust. Real-account shapes: Alpaca string qtys,
# a GTC limit that partially filled and was then canceled.
# ---------------------------------------------------------------------------
def _g8_order(sym, side, qty, px, when, status="filled", filled_qty=None):
    return {"id": f"{sym}{side}{when}", "symbol": sym, "side": side,
            "qty": str(qty), "type": "limit", "status": status,
            "submitted_at": when, "filled_at": when,
            "filled_avg_price": None if px is None else str(px),
            "notional": None,
            "filled_qty": str(qty if filled_qty is None else filled_qty)}


class TestG8FillsOnNonFilledStatus:
    def test_canceled_with_partial_fill_becomes_a_lot(self):
        q = 14.723202954
        r = estimate_taxes([
            _g8_order("SOL/USD", "buy", 20, 86.5548, "2026-04-26T15:21:11Z",
                      status="canceled", filled_qty=q),
            _g8_order("SOL/USD", "sell", q, 90.0, "2026-04-27T15:00:00Z"),
        ])
        assert r["realized_gain"] == pytest.approx(q * (90.0 - 86.5548))
        assert r["unmatched_sell_qty"] == pytest.approx(0.0)
        assert r["basis_complete"] is True
        assert r["num_trades"] == 1

    def test_canceled_partial_sell_is_matched_for_its_filled_qty_only(self):
        r = estimate_taxes([
            _g8_order("AAA", "buy", 10, 100.0, _iso(BASE)),
            _g8_order("AAA", "sell", 10, 120.0, _iso(BASE + dt.timedelta(days=2)),
                      status="expired", filled_qty=4),
        ])
        assert r["realized_gain"] == pytest.approx(4 * 20.0)
        assert r["unmatched_sell_qty"] == pytest.approx(0.0)

    @pytest.mark.parametrize("status", ["canceled", "expired",
                                        "partially_filled", "done_for_day"])
    def test_fill_bearing_statuses_count(self, status):
        r = estimate_taxes([
            _g8_order("AAA", "buy", 5, 100.0, _iso(BASE), status=status,
                      filled_qty=5),
            _g8_order("AAA", "sell", 5, 110.0, _iso(BASE + dt.timedelta(days=1))),
        ])
        assert r["realized_gain"] == pytest.approx(50.0)
        assert r["basis_complete"] is True

    @pytest.mark.parametrize("status", ["canceled", "expired"])
    def test_zero_or_missing_filled_qty_on_canceled_is_ignored(self, status):
        for fq in (0, "0", None):
            o = _g8_order("AAA", "buy", 5, 100.0, _iso(BASE), status=status)
            o["filled_qty"] = fq
            r = estimate_taxes([o, _g8_order(
                "AAA", "sell", 5, 110.0, _iso(BASE + dt.timedelta(days=1)))])
            assert r["unmatched_sell_qty"] == pytest.approx(5.0)
            assert r["realized_gain"] == pytest.approx(0.0)

    def test_new_and_replaced_still_ignored_even_with_filled_qty(self):
        for status in ("new", "accepted", "replaced", "rejected"):
            r = estimate_taxes([
                _g8_order("AAA", "buy", 5, 100.0, _iso(BASE), status=status),
                _g8_order("AAA", "sell", 5, 110.0,
                          _iso(BASE + dt.timedelta(days=1))),
            ])
            assert r["unmatched_sell_qty"] == pytest.approx(5.0), status


class TestG8LeapYearLongTerm:
    @staticmethod
    def _d(y, m, d):
        return dt.datetime(y, m, d, 15, tzinfo=dt.timezone.utc)

    def test_366_days_across_leap_day_is_short_term(self):
        b, s = self._d(2024, 1, 15), self._d(2025, 1, 15)
        assert (s - b).days == 366
        assert _is_long_term(b, s) is False
        r = estimate_taxes([
            _g8_order("AAA", "buy", 1, 100, "2024-01-15T15:00:00Z"),
            _g8_order("AAA", "sell", 1, 200, "2025-01-15T15:00:00Z"),
        ])
        assert r["short_term_gain"] == pytest.approx(100.0)
        assert r["long_term_gain"] == pytest.approx(0.0)
        assert r["estimated_tax"] == pytest.approx(100 * (0.37 + 0.05))

    def test_anniversary_plus_one_day_is_long_term(self):
        assert _is_long_term(self._d(2024, 1, 15), self._d(2025, 1, 16)) is True
        # 366 days spanning 29 Feb 2024 is the exact anniversary -> short
        assert _is_long_term(self._d(2023, 3, 1), self._d(2024, 3, 1)) is False
        assert _is_long_term(self._d(2023, 3, 1), self._d(2024, 3, 2)) is True
        r = estimate_taxes([
            _g8_order("AAA", "buy", 1, 100, "2024-01-15T15:00:00Z"),
            _g8_order("AAA", "sell", 1, 200, "2025-01-16T15:00:00Z"),
        ])
        assert r["long_term_gain"] == pytest.approx(100.0)
        assert r["estimated_tax"] == pytest.approx(100 * (0.20 + 0.05))

    def test_feb_29_purchase(self):
        assert _is_long_term(self._d(2024, 2, 29), self._d(2025, 2, 28)) is False
        assert _is_long_term(self._d(2024, 2, 29), self._d(2025, 3, 1)) is True

    def test_dates_compared_in_utc_and_naive_tolerated(self):
        b = dt.datetime(2023, 1, 1, 23, 30, tzinfo=dt.timezone.utc)
        # 2024-01-02 00:10 +02:00 == 2024-01-01 22:10 UTC -> the anniversary
        s = dt.datetime(2024, 1, 2, 0, 10,
                        tzinfo=dt.timezone(dt.timedelta(hours=2)))
        assert _is_long_term(b, s) is False
        assert _is_long_term(dt.datetime(2023, 1, 1), self._d(2024, 1, 2)) is True


class TestG8FloatDust:
    def test_float_dust_does_not_flip_basis_complete(self):
        r = estimate_taxes([
            _g8_order("BTC/USD", "buy", 0.3, 100, "2026-01-01T00:00:00Z"),
            _g8_order("BTC/USD", "sell", 0.1, 110, "2026-01-02T00:00:00Z"),
            _g8_order("BTC/USD", "sell", 0.2, 110, "2026-01-03T00:00:00Z"),
        ])
        assert r["unmatched_sell_qty"] == 0.0
        assert r["basis_complete"] is True
        assert r["realized_gain"] == pytest.approx(3.0)
        assert r["num_trades"] == 2

    def test_dust_lot_is_popped_not_matched_again(self):
        r = estimate_taxes([
            _g8_order("X", "buy", 0.1, 100, "2026-01-01T00:00:00Z"),
            _g8_order("X", "buy", 0.2, 100, "2026-01-01T00:00:01Z"),
            _g8_order("X", "sell", 0.3, 110, "2026-01-02T00:00:00Z"),
            _g8_order("X", "sell", 1.0, 110, "2026-01-03T00:00:00Z"),
        ])
        assert r["unmatched_sell_qty"] == pytest.approx(1.0)
        assert r["num_trades"] == 2

    def test_real_shortfall_above_tolerance_still_counts(self):
        r = estimate_taxes([
            _g8_order("X", "buy", 1.0, 100, "2026-01-01T00:00:00Z"),
            _g8_order("X", "sell", 1.000001, 110, "2026-01-02T00:00:00Z"),
        ])
        assert r["unmatched_sell_qty"] == pytest.approx(1e-6)
        assert r["basis_complete"] is False
