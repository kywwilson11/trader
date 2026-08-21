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
