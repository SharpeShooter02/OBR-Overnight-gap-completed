"""IB Gateway restarts daily. The session must notice before it trades.

The daemon sleeps between sessions with a plain time.sleep (main.py
_run_daemon), which neither pumps the asyncio loop nor checks the socket. So a
Gateway restart at 23:45 leaves a dead connection that nothing observes until
08:30 -- and a dead IB socket does NOT raise. It degrades quietly:

    get_account()   accountValues() returns []  -> equity 0.0 (logged, not raised)
    check_margin()  is_connected() False        -> full-notional fallback, 100%
    get_positions() portfolio() returns []      -> looks flat

A session started that way runs to completion, has every entry rejected by the
risk gate on zero equity, and books nothing. No crash, no ConnectionError, and
the daemon's `except ConnectionError` recovery -- which exists and works -- is
never reached. A silent no-trade day is the worst failure mode available here,
because it looks exactly like a day with no qualifying gaps.

run_session therefore reconnects first, and raises ConnectionError if it
cannot, so the daemon's existing handler retries the same day rather than
proceeding on a dead socket.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _runner(broker):
    """SessionRunner with every collaborator stubbed except the broker."""
    from orb_live.runner.session_runner import SessionRunner

    mgr = MagicMock()
    return SessionRunner(
        config=SimpleNamespace(eod_flatten_lead_secs=30, symbols=[]),
        broker=broker,
        state_store=MagicMock(),
        bar_cache=MagicMock(),
        bar_router=MagicMock(),
        pre_market_job=MagicMock(),
        strategy_engine=MagicMock(),
        position_manager=mgr,
        risk_gate=MagicMock(),
        underlying_store=SimpleNamespace(),
        clock=MagicMock(),
        _sleep=lambda _: None,
    ), mgr


class TestSessionRefusesToStartDisconnected:
    def test_reconnects_when_the_socket_is_down(self):
        broker = MagicMock()
        broker.verify_connection.side_effect = [False, True]
        broker.reconnect.return_value = True

        runner, mgr = _runner(broker)
        runner._ensure_connected()

        broker.reconnect.assert_called_once()

    def test_raises_connection_error_when_reconnect_fails(self):
        """ConnectionError specifically -- that is what the daemon catches and
        retries the same day. Any other exception kills the daemon loop."""
        broker = MagicMock()
        broker.verify_connection.return_value = False
        broker.reconnect.return_value = False

        runner, _ = _runner(broker)
        with pytest.raises(ConnectionError):
            runner._ensure_connected()

    def test_no_reconnect_when_already_connected(self):
        broker = MagicMock()
        broker.verify_connection.return_value = True
        runner, _ = _runner(broker)
        runner._ensure_connected()
        broker.reconnect.assert_not_called()

    def test_run_session_checks_before_touching_the_broker(self):
        """The guard must run BEFORE startup_reconcile, which asks the broker
        for positions and would read [] off a dead socket as 'flat'."""
        broker = MagicMock()
        broker.verify_connection.return_value = False
        broker.reconnect.return_value = False

        runner, mgr = _runner(broker)
        with pytest.raises(ConnectionError):
            runner.run_session(date(2026, 8, 26))
        mgr.startup_reconcile.assert_not_called()

    def test_broker_without_is_connected_is_tolerated(self):
        """dry_run and test brokers do not implement it; absence must not
        become a hard failure."""
        broker = SimpleNamespace()
        runner, _ = _runner(broker)
        runner._ensure_connected()      # must not raise


class TestStaleFlagIsNotTrusted:
    """2026-08-28: the guard above passed and the session still died.

    IB Gateway restarted at 02:50 ET. At 08:30:02.01 `_ensure_connected` asked
    `is_connected()`, got True, and returned. 70ms later the first real request
    raised WinError 10054. Every subsequent IB call logged 'IBClient is not
    connected', equity came back 0.0, cap_factor 0.0, and Phase 1 found zero
    candidates -- the exact silent no-trade day this module was written to
    prevent.

    `isConnected()` reads a flag ib_async clears only when its reader coroutine
    processes the peer's EOF. The daemon's plain time.sleep never pumps the
    loop, so the flag stayed True across a two-day-old dead socket. A liveness
    check that cannot observe death is not a liveness check.
    """

    def test_prefers_verify_connection_over_the_cached_flag(self):
        broker = MagicMock()
        broker.is_connected.return_value = True      # the lie
        broker.verify_connection.return_value = False
        broker.reconnect.return_value = True

        runner, _ = _runner(broker)
        runner._ensure_connected()

        broker.reconnect.assert_called_once()

    def test_falls_back_to_is_connected_when_verify_is_absent(self):
        broker = SimpleNamespace(
            is_connected=lambda: True, reconnect=MagicMock(return_value=True))
        runner, _ = _runner(broker)
        runner._ensure_connected()          # must not raise
        broker.reconnect.assert_not_called()

    def test_verify_raising_is_treated_as_down(self):
        broker = MagicMock()
        broker.verify_connection.side_effect = OSError("10054")
        broker.reconnect.return_value = False

        runner, _ = _runner(broker)
        with pytest.raises(ConnectionError):
            runner._ensure_connected()
