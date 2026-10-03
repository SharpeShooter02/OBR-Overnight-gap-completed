"""The recovery path must not be able to starve itself. Measured 2026-09-17.

From 00:23 to 05:20 ET the Gateway accepted connections, logged the client on,
and then answered nothing: positions, open orders and account updates all timed
out, with 1100s in between. The idle heartbeat reconnected every 5 minutes and
failed every time, for five hours, and maybe_restart_gateway never fired.

Why: connect() set _upstream_lost_at = None, so every reconnect attempt reset
the loss clock. The 1100 landed a few seconds after each connect, so the
measured loss was ~290s when the next heartbeat arrived -- permanently just
under the 300s restart threshold. The escape hatch was unreachable by
construction.

Two rules follow:
  * only proof of recovery clears the loss clock -- a 1101/1102, or account
    data actually arriving. Opening a socket proves nothing.
  * 1100 is not the only way a Gateway dies. One that logs you on and then
    times out every request is equally dead, so repeated failed reconnects
    must escalate to a restart on their own.
"""
from __future__ import annotations

import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


def _client(mock_ib=None, **kwargs):
    from orb_live.data.ib_client import IBClient
    mock_ib = mock_ib or MagicMock()
    with patch("orb_live.data.ib_client.IB", return_value=mock_ib):
        return IBClient(paper=True, host="127.0.0.1", port=4002, client_id=3, **kwargs)


class _FakeIBC:
    def __init__(self):
        self.received: list[bytes] = []
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(4)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            with conn:
                self.received.append(conn.recv(1024))
                conn.sendall(b"OK\n")

    def close(self):
        self._srv.close()


@pytest.fixture
def ibc():
    srv = _FakeIBC()
    yield srv
    srv.close()


# ── only proof of recovery clears the loss clock ──────────────────────────────

class TestLossClockSurvivesReconnects:
    def test_connect_does_not_clear_an_outstanding_loss(self, monkeypatch):
        """The 2026-09-17 bug in one assertion."""
        ib = MagicMock()
        ib.accountValues.return_value = []          # upstream still dead
        c = _client(ib)
        monkeypatch.setattr(c, "_validate_subscription", lambda: None)
        c._on_ib_error(-1, 1100, "lost", None)
        lost_at = c._upstream_lost_at

        c.connect()

        assert c._upstream_lost_at == lost_at

    def test_account_data_arriving_clears_the_loss(self, monkeypatch):
        """Real account values are proof the upstream came back."""
        ib = MagicMock()
        ib.accountValues.return_value = [object()]
        c = _client(ib)
        monkeypatch.setattr(c, "_validate_subscription", lambda: None)
        c._on_ib_error(-1, 1100, "lost", None)

        c.connect()

        assert c.upstream_lost_for() is None

    def test_1102_clears_the_loss(self):
        c = _client()
        c._on_ib_error(-1, 1100, "lost", None)
        c._on_ib_error(-1, 1102, "restored", None)
        assert c.upstream_lost_for() is None

    def test_the_clock_measures_from_the_first_loss_across_reconnects(self, monkeypatch):
        ib = MagicMock()
        ib.accountValues.return_value = []
        c = _client(ib)
        monkeypatch.setattr(c, "_validate_subscription", lambda: None)
        c._upstream_lost_at = time.time() - 1000

        c.connect()

        assert c.upstream_lost_for() >= 1000


# ── repeated failures escalate on their own ───────────────────────────────────

class TestFailedReconnectsEscalate:
    def _wire(self, c, monkeypatch, connect_ok=True):
        monkeypatch.setattr(c, "disconnect", lambda: None)
        monkeypatch.setattr("orb_live.data.ib_client.time.sleep", lambda s: None)
        if connect_ok:
            monkeypatch.setattr(c, "connect", lambda: None)
        else:
            def _boom():
                raise ConnectionError("refused")
            monkeypatch.setattr(c, "connect", _boom)

    def test_exhausted_reconnects_are_counted(self, monkeypatch):
        ib = MagicMock()
        ib.isConnected.return_value = False
        c = _client(ib)
        self._wire(c, monkeypatch, connect_ok=False)

        assert c.reconnect(max_attempts=1) is False
        assert c._failed_reconnects == 1
        assert c.reconnect(max_attempts=1) is False
        assert c._failed_reconnects == 2

    def test_a_successful_reconnect_resets_the_counter(self, monkeypatch):
        ib = MagicMock()
        ib.isConnected.return_value = True
        c = _client(ib)
        self._wire(c, monkeypatch)
        c._failed_reconnects = 5

        assert c.reconnect(max_attempts=1) is True
        assert c._failed_reconnects == 0

    def test_restart_fires_on_repeated_failures_with_no_1100(self, ibc, monkeypatch):
        """A Gateway that logs you on and then times out every request never
        sends 1100 -- the failure count is the only signal there is."""
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_FAILURES", "2")
        c = _client()
        assert c.upstream_lost_for() is None

        c._failed_reconnects = 2
        assert c.maybe_restart_gateway() is True
        time.sleep(0.2)
        assert ibc.received and ibc.received[0].strip() == b"RESTART"

    def test_one_failure_is_not_enough(self, ibc, monkeypatch):
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_FAILURES", "2")
        c = _client()
        c._failed_reconnects = 1
        assert c.maybe_restart_gateway() is False
        assert not ibc.received

    def test_the_five_hour_loop_now_escalates(self, ibc, monkeypatch):
        """End to end on the measured shape: each connect succeeds at the socket
        level, a 1100 lands straight after, every attempt fails. The restart
        must arrive within a handful of cycles rather than never."""
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_SECS", "300")
        monkeypatch.setenv("IBC_RESTART_AFTER_FAILURES", "2")
        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.accountValues.return_value = []
        c = _client(ib)
        monkeypatch.setattr(c, "disconnect", lambda: None)
        monkeypatch.setattr(c, "_validate_subscription", lambda: None)
        monkeypatch.setattr("orb_live.data.ib_client.time.sleep", lambda s: None)

        def connect():
            c._ib.connect("127.0.0.1", 4002, clientId=1, timeout=10)
            c._on_ib_error(-1, 1100, "lost", None)   # Gateway replays it at once
        monkeypatch.setattr(c, "connect", connect)

        for _ in range(3):
            c.reconnect(max_attempts=1)

        time.sleep(0.2)
        assert ibc.received, "no RESTART was ever sent -- the 09-17 five-hour loop"
