"""Bracket legs must be verified at the broker, not assumed.

MOTIVATING INCIDENT (2026-09-03, SOLT). `submit_bracket_order` placed three
orders and returned their ids without confirming any of them reached IB:

    tp1_trade  = self._ib.placeOrder(contract, tp1_order)
    stop_trade = self._ib.placeOrder(contract, stop_order)
    # ... cache, return ids          <- no wait, no check, no retry

IB silently dropped the TP1 child (11924). It never produced a single
orderStatus event and never appeared in openTrades. The daemon wrote
tp1_order_id=11924 into open_positions and believed the position was protected
on the upside for the rest of the session. SOLT ran through its 59.99 target to
60.93 with nothing resting to sell it.

The leg that survived was the STOP -- the order carrying transmit=True. The
parent filled in the same instant the group transmitted (PreSubmitted -> Filled,
no intervening Submitted), and the still-unregistered TP1 child was orphaned.
1 of 20 filled brackets in the retained logs; rare, silent, unbounded.

The signature already promised the cure and never used it: `timeout: float = 5.0`.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from orb_live.data.ib_client import IBClient

SYMBOL = "SOLT"


# -- helpers ------------------------------------------------------------------

def _trade(order_id: int, status: str, symbol: str = SYMBOL):
    """A minimal ib_async Trade double whose status can be mutated in place."""
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id, ocaGroup=None, parentId=0,
                              transmit=True, lmtPrice=0.0, auxPrice=0.0,
                              action="SELL", totalQuantity=135.0,
                              orderType="LMT", tif="DAY", orderRef=""),
        orderStatus=SimpleNamespace(status=status, filled=0.0, avgFillPrice=0.0,
                                    remaining=0.0),
        contract=SimpleNamespace(symbol=symbol),
    )


def _client(symbol: str = SYMBOL, min_tick: float = 0.01) -> IBClient:
    with patch("orb_live.data.ib_client.IB") as MockIB:
        MockIB.return_value = MagicMock()
        c = IBClient(paper=True)
    c._asset_cache[symbol] = {"symbol": symbol, "tradable": True, "shortable": True,
                              "status": "active", "primary_exchange": "ARCA",
                              "conId": 1}
    c._contract_cache[symbol] = SimpleNamespace(symbol=symbol, conId=1)
    c._min_tick_cache[symbol] = min_tick
    c._ib.isConnected.return_value = True
    return c


def _bracket(c: IBClient, **kw):
    return c.submit_bracket_order(
        SYMBOL, "buy", 135, entry_price=58.04,
        tp1_limit_price=59.99, stop_price=57.06, **kw)


# -- Part 1: the client verifies both children reached the broker -------------

class TestBracketLegVerification:

    def test_happy_path_reports_verified_and_does_not_replace(self):
        """Both children live -> exactly three placeOrder calls, tp1_verified."""
        c = _client()
        c._ib.placeOrder.side_effect = [_trade(11923, "PreSubmitted"),
                                        _trade(11924, "PreSubmitted"),
                                        _trade(11925, "PreSubmitted")]

        r = _bracket(c, timeout=0.05)

        assert c._ib.placeOrder.call_count == 3, "no replacement should be needed"
        assert r["tp1_order_id"] == "11924"
        assert r["stop_order_id"] == "11925"
        assert r["tp1_verified"] is True
        assert r["tp1_replaced"] is False

    def test_dropped_tp1_is_replaced(self):
        """THE SOLT CASE. TP1 never leaves PendingSubmit -> re-placed standalone."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"),                   # parent fills instantly
            _trade(11924, "PendingSubmit"),            # IB never acknowledges it
            _trade(11925, "PreSubmitted"),             # stop survives
            _trade(11930, "Submitted"),                # the replacement
        ]

        r = _bracket(c, timeout=0.05)

        assert c._ib.placeOrder.call_count == 4, "TP1 must be re-placed"
        assert r["tp1_replaced"] is True
        assert r["tp1_verified"] is True
        assert r["tp1_order_id"] == "11930", "caller must get the id that is live"

    def test_replacement_joins_the_stop_oca_group(self):
        """Otherwise a stop fill would leave the replacement TP resting alone."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PendingSubmit"),
            _trade(11925, "PreSubmitted"), _trade(11930, "Submitted"),
        ]

        r = _bracket(c, timeout=0.05)

        placed = [call.args[1] for call in c._ib.placeOrder.call_args_list]
        stop_order, repl_order = placed[2], placed[3]
        assert stop_order.ocaGroup
        assert repl_order.ocaGroup == stop_order.ocaGroup
        assert repl_order.ocaType == 1
        assert r["oca_group"] == stop_order.ocaGroup

    def test_replacement_carries_no_parent_id_and_transmits(self):
        """The parent already filled; a parentId would re-hold the order."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PendingSubmit"),
            _trade(11925, "PreSubmitted"), _trade(11930, "Submitted"),
        ]

        _bracket(c, timeout=0.05)

        repl = c._ib.placeOrder.call_args_list[3].args[1]
        assert not getattr(repl, "parentId", 0)
        assert repl.transmit is True

    def test_replacement_preserves_price_side_and_quantity(self):
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PendingSubmit"),
            _trade(11925, "PreSubmitted"), _trade(11930, "Submitted"),
        ]

        _bracket(c, timeout=0.05)

        repl = c._ib.placeOrder.call_args_list[3].args[1]
        assert repl.action == "SELL", "exit side of a long"
        assert float(repl.totalQuantity) == pytest.approx(135)
        assert float(repl.lmtPrice) == pytest.approx(59.99)

    def test_unverifiable_tp1_is_reported_not_swallowed(self):
        """Replacement also dropped -> caller must be able to see it failed."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PendingSubmit"),
            _trade(11925, "PreSubmitted"), _trade(11930, "PendingSubmit"),
        ]
        c._log = MagicMock()

        r = _bracket(c, timeout=0.05)

        assert r["tp1_verified"] is False
        assert r["tp1_replaced"] is True
        assert c._log.critical.called, "an unprotected position must be loud"

    def test_dropped_stop_is_replaced_too(self):
        """Symmetric: the stop is the leg that survived on SOLT, not a given."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PreSubmitted"),
            _trade(11925, "PendingSubmit"), _trade(11931, "Submitted"),
        ]

        r = _bracket(c, timeout=0.05)

        assert r["stop_replaced"] is True
        assert r["stop_verified"] is True
        assert r["stop_order_id"] == "11931"

    def test_verification_polls_within_timeout_and_yields(self):
        """Must pump the event loop rather than spin, and must be bounded."""
        c = _client()
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"), _trade(11924, "PendingSubmit"),
            _trade(11925, "PreSubmitted"), _trade(11930, "Submitted"),
        ]

        _bracket(c, timeout=0.05)

        assert c._ib.sleep.called, "verification must yield to the ib_async loop"

    def test_leg_that_arrives_late_is_not_replaced(self):
        """A child that reaches PreSubmitted during the poll window is fine."""
        c = _client()
        late = _trade(11924, "PendingSubmit")

        def _promote(_secs):
            late.orderStatus.status = "PreSubmitted"

        c._ib.sleep.side_effect = _promote
        c._ib.placeOrder.side_effect = [
            _trade(11923, "PreSubmitted"), late, _trade(11925, "PreSubmitted")]

        r = _bracket(c, timeout=1.0)

        assert c._ib.placeOrder.call_count == 3, "no replacement for a slow leg"
        assert r["tp1_verified"] is True
        assert r["tp1_replaced"] is False


# -- Part 2: the manager re-arms protection it finds missing at runtime -------

class TestProtectiveOrderReconciliation:

    def _open(self, mock_broker, tmp_store):
        from orb_live.tests.test_position_manager import _build_mgr, _make_entry
        mgr = _build_mgr(mock_broker, tmp_store)
        pos = mgr.open_position(_make_entry(), SYMBOL, +1, date(2026, 1, 5))
        assert pos is not None
        return mgr

    def test_missing_tp1_is_rearmed(self, mock_broker, tmp_store):
        """The SOLT failure mode, caught after the fact by the periodic sweep."""
        mgr = self._open(mock_broker, tmp_store)
        db = tmp_store.get_open_position(SYMBOL)
        mock_broker._orders.pop(db["tp1_order_id"], None)     # IB dropped it

        result = mgr.verify_protective_orders()

        assert result[SYMBOL] == "tp1_rearmed"
        new_id = tmp_store.get_open_position(SYMBOL)["tp1_order_id"]
        assert new_id != db["tp1_order_id"]
        assert mock_broker.get_order(new_id)["status"] != "not_found"

    def test_live_orders_are_left_alone(self, mock_broker, tmp_store):
        mgr = self._open(mock_broker, tmp_store)
        before = tmp_store.get_open_position(SYMBOL)["tp1_order_id"]

        result = mgr.verify_protective_orders()

        assert result[SYMBOL] == "ok"
        assert tmp_store.get_open_position(SYMBOL)["tp1_order_id"] == before

    def test_flat_position_is_not_rearmed(self, mock_broker, tmp_store):
        """Never place a protective order against a position that is gone --
        that is how a flat book becomes a naked short."""
        mgr = self._open(mock_broker, tmp_store)
        db = tmp_store.get_open_position(SYMBOL)
        tmp_store.update_open_position(SYMBOL, remaining=0)
        mock_broker._orders.pop(db["tp1_order_id"], None)

        result = mgr.verify_protective_orders()

        assert result[SYMBOL] == "skipped_flat"


# -- Part 3: the wait for the close is chunked so sweeps can run --------------

class TestPeriodicProtectiveSweep:

    def _runner(self):
        from orb_live.runner.session_runner import SessionRunner
        r = SessionRunner.__new__(SessionRunner)
        r._log = None
        r._mgr = MagicMock()
        r._slept = []
        r._sleep = r._slept.append
        return r

    def test_wait_is_chunked_and_sweeps_between_chunks(self):
        r = self._runner()

        r._wait_with_protective_sweeps(750.0)

        assert sum(r._slept) == pytest.approx(750.0), "must still wait the full time"
        assert r._slept == [300.0, 300.0, 150.0]
        assert r._mgr.verify_protective_orders.call_count == 2

    def test_short_wait_sweeps_nothing(self):
        r = self._runner()

        r._wait_with_protective_sweeps(10.0)

        assert r._slept == [10.0]
        r._mgr.verify_protective_orders.assert_not_called()

    def test_sweep_failure_does_not_abort_the_wait(self):
        """A broken sweep must never leave the runner from reaching its flatten."""
        r = self._runner()
        r._mgr.verify_protective_orders.side_effect = RuntimeError("ib down")

        r._wait_with_protective_sweeps(750.0)

        assert sum(r._slept) == pytest.approx(750.0)


# -- Part 4: verification must never break entry (regression, 2026-09-04) -----

class TestVerificationNeverBreaksEntry:
    """`ib.sleep()` drives the loop via run_until_complete. Breakout entries are
    placed from inside a bar callback, i.e. already inside that loop, so the
    call raises "This event loop is already running". The first version of the
    V44 fix let it propagate: on 2026-09-04 all five entries filled at the
    broker and every one was recorded as `entry_rejected_or_unfilled`, leaving
    three real positions the daemon did not know it held."""

    def _legs(self, c):
        c._ib.placeOrder.side_effect = [
            _trade(11923, "Filled"),            # parent fills instantly
            _trade(11924, "PendingSubmit"),     # would normally trigger a wait
            _trade(11925, "PreSubmitted"),
            _trade(11930, "Submitted"),
        ]

    def test_running_loop_does_not_raise(self):
        c = _client()
        self._legs(c)
        c._ib.sleep.side_effect = RuntimeError("This event loop is already running")

        r = _bracket(c, timeout=0.05)          # must not raise

        assert r["entry_order_id"] == "11923"
        assert r["tp1_order_id"] == "11924"

    def test_running_loop_does_not_replace_an_unverifiable_leg(self):
        """Never re-place a leg you were unable to check — that is how one
        bracket becomes two."""
        c = _client()
        self._legs(c)
        c._ib.sleep.side_effect = RuntimeError("This event loop is already running")

        r = _bracket(c, timeout=0.05)

        assert c._ib.placeOrder.call_count == 3, "no duplicate protective orders"
        assert r["tp1_replaced"] is False
        assert r["stop_replaced"] is False

    def test_unexpected_verification_error_still_returns_the_entry(self):
        """Any failure in the safety net leaves the entry standing."""
        c = _client()
        self._legs(c)
        c._log = MagicMock()
        c._ib.sleep.side_effect = ValueError("something else entirely")

        r = _bracket(c, timeout=0.05)

        assert r["entry_order_id"] == "11923"
        assert c._log.critical.called
