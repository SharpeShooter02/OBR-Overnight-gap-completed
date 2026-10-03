"""IB Gateway alive locally, IBKR upstream gone (Error 1100). Measured 2026-09-15.

The Gateway accepted the socket at 08:30 ET but had lost IBKR (1100 x48, no
1102). The session-start probe passed, then the first qualifyContracts in the
underlying refresh blocked for five hours: ib_async's RequestTimeout defaults
to 0, which means wait forever. When the Gateway was finally restarted at
13:31, the session resumed on a dead socket, got equity 0.0, planned zero
candidates and recorded a 0/0 equity row. No alert fired at any point.
09-09 was the same failure.

Four defences:
  1. every blocking IB request times out, and a timeout never poisons a cache;
  2. 1100/1102 are tracked, so a socket that answers locally but has no
     upstream is not considered connected;
  3. no equity -> no session (ConnectionError, so the daemon retries), and
     never write a 0 equity row;
  4. the daemon probes the broker while idle and, after a sustained upstream
     loss, asks IBC to restart the Gateway.
"""
from __future__ import annotations

import socket
import threading
import time
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


def _client(mock_ib=None, **kwargs):
    from orb_live.data.ib_client import IBClient
    mock_ib = mock_ib or MagicMock()
    with patch("orb_live.data.ib_client.IB", return_value=mock_ib):
        return IBClient(paper=True, host="127.0.0.1", port=4002, client_id=3, **kwargs)


# ── 1. blocking requests time out ─────────────────────────────────────────────

class TestRequestTimeout:
    def test_blocking_requests_have_a_finite_timeout(self, monkeypatch):
        monkeypatch.delenv("IB_REQUEST_TIMEOUT", raising=False)
        ib = MagicMock()
        _client(ib)
        assert 0 < ib.RequestTimeout <= 60

    def test_timeout_is_configurable(self, monkeypatch):
        monkeypatch.setenv("IB_REQUEST_TIMEOUT", "7")
        ib = MagicMock()
        _client(ib)
        assert ib.RequestTimeout == 7.0

    def test_a_timed_out_contract_lookup_is_not_cached(self):
        """A transient timeout must not mark the symbol untradable for the
        lifetime of the client."""
        ib = MagicMock()
        ib.qualifyContracts.side_effect = TimeoutError()
        c = _client(ib)

        assert c.get_asset("AMD")["tradable"] is False
        assert "AMD" not in c._asset_cache
        c.get_asset("AMD")
        assert ib.qualifyContracts.call_count == 2


# ── 2. upstream loss is tracked ───────────────────────────────────────────────

class TestUpstreamState:
    def test_1100_marks_upstream_lost_and_1102_restores(self):
        c = _client()
        assert c.upstream_lost_for() is None
        c._on_ib_error(-1, 1100, "Connectivity between IBKR and TWS has been lost.", None)
        assert c.upstream_lost_for() is not None
        c._on_ib_error(-1, 1102, "Connectivity restored - data maintained.", None)
        assert c.upstream_lost_for() is None

    def test_1101_also_restores(self):
        c = _client()
        c._on_ib_error(-1, 1100, "lost", None)
        c._on_ib_error(-1, 1101, "restored - data lost", None)
        assert c.upstream_lost_for() is None

    def test_repeated_1100_keeps_the_first_timestamp(self):
        c = _client()
        c._on_ib_error(-1, 1100, "lost", None)
        first = c._upstream_lost_at
        c._on_ib_error(-1, 1100, "lost", None)
        assert c._upstream_lost_at == first

    def test_verify_connection_false_while_upstream_lost(self):
        ib = MagicMock()
        ib.isConnected.return_value = True
        ib.reqCurrentTime.return_value = object()
        c = _client(ib)
        assert c.verify_connection() is True
        c._on_ib_error(-1, 1100, "lost", None)
        assert c.verify_connection() is False

    def test_reconnect_is_not_success_while_upstream_still_lost(self, monkeypatch):
        """The Gateway replays 1100 on connect when it has no upstream."""
        ib = MagicMock()
        ib.isConnected.return_value = True
        c = _client(ib)
        monkeypatch.setattr(c, "disconnect", lambda: None)
        monkeypatch.setattr(c, "maybe_restart_gateway", lambda: False)

        def connect():
            c._upstream_lost_at = time.time()
        monkeypatch.setattr(c, "connect", connect)
        monkeypatch.setattr("orb_live.data.ib_client.time.sleep", lambda s: None)

        assert c.reconnect(max_attempts=2) is False


# ── IBC command server ────────────────────────────────────────────────────────

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
                data = conn.recv(1024)
                self.received.append(data)
                conn.sendall(b"OK\n")

    def close(self):
        self._srv.close()


@pytest.fixture
def ibc():
    srv = _FakeIBC()
    yield srv
    srv.close()


class TestGatewayRestart:
    def _lost(self, c, secs):
        c._upstream_lost_at = time.time() - secs

    def test_sends_restart_after_sustained_upstream_loss(self, ibc, monkeypatch):
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_SECS", "300")
        c = _client()
        self._lost(c, 400)

        assert c.maybe_restart_gateway() is True
        time.sleep(0.2)
        assert ibc.received and ibc.received[0].strip() == b"RESTART"

    def test_no_restart_for_a_brief_blip(self, ibc, monkeypatch):
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_SECS", "300")
        c = _client()
        self._lost(c, 30)
        assert c.maybe_restart_gateway() is False
        assert not ibc.received

    def test_no_restart_when_upstream_is_fine(self, ibc, monkeypatch):
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        c = _client()
        assert c.maybe_restart_gateway() is False

    def test_disabled_without_a_command_port(self, monkeypatch):
        monkeypatch.delenv("IBC_COMMAND_PORT", raising=False)
        c = _client()
        self._lost(c, 10_000)
        assert c.maybe_restart_gateway() is False

    def test_does_not_restart_again_inside_the_cooldown(self, ibc, monkeypatch):
        monkeypatch.setenv("IBC_COMMAND_PORT", str(ibc.port))
        monkeypatch.setenv("IBC_RESTART_AFTER_SECS", "300")
        c = _client()
        self._lost(c, 400)
        assert c.maybe_restart_gateway() is True
        self._lost(c, 400)
        assert c.maybe_restart_gateway() is False

    def test_unreachable_command_server_returns_false(self, monkeypatch):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        monkeypatch.setenv("IBC_COMMAND_PORT", str(port))
        c = _client()
        self._lost(c, 10_000)
        assert c.maybe_restart_gateway() is False

    def test_reconnect_waits_for_the_gateway_after_a_restart(self, monkeypatch):
        monkeypatch.setenv("IBC_RESTART_WAIT_SECS", "90")
        ib = MagicMock()
        ib.isConnected.return_value = True
        c = _client(ib)
        order: list = []
        monkeypatch.setattr(c, "disconnect", lambda: None)
        monkeypatch.setattr(c, "maybe_restart_gateway", lambda: order.append("restart") or True)
        monkeypatch.setattr(c, "connect", lambda: order.append("connect"))
        monkeypatch.setattr("orb_live.data.ib_client.time.sleep",
                            lambda s: order.append(("sleep", s)))

        assert c.reconnect(max_attempts=1) is True
        assert order[0] == "restart"
        assert ("sleep", 90.0) in order
        assert order.index(("sleep", 90.0)) < order.index("connect")


# ── 3. no equity, no session ──────────────────────────────────────────────────

def _runner(broker=None, store=None, gate=None):
    from orb_live.runner.session_runner import SessionRunner
    clock = MagicMock()
    clock.is_half_day.return_value = False
    return SessionRunner(
        config=SimpleNamespace(eod_flatten_lead_secs=30, symbols=[],
                               prior_session_filters={}),
        broker=broker or MagicMock(),
        state_store=store or MagicMock(),
        bar_cache=MagicMock(),
        bar_router=MagicMock(),
        pre_market_job=MagicMock(),
        strategy_engine=MagicMock(),
        position_manager=MagicMock(),
        risk_gate=gate or MagicMock(),
        underlying_store=MagicMock(),
        clock=clock,
        _sleep=lambda _: None,
    )


class TestEquityUnavailable:
    def test_pre_market_raises_connection_error_without_equity(self):
        gate, store = MagicMock(), MagicMock()
        r = _runner(gate=gate, store=store)
        r._fetch_start_equity = lambda: None

        with pytest.raises(ConnectionError):
            r._run_pre_market(date(2026, 9, 15))
        gate.session_start.assert_not_called()
        store.upsert_day_state.assert_not_called()

    def test_pre_market_proceeds_with_equity(self):
        gate = MagicMock()
        r = _runner(gate=gate)
        r._fetch_start_equity = lambda: 31_476.30
        try:
            r._run_pre_market(date(2026, 9, 15))
        except ConnectionError:
            pytest.fail("valid equity must not abort the session")
        except Exception:
            pass    # later pre-market steps are stubbed; only the gate matters
        gate.session_start.assert_called_once()

    def test_eod_never_records_a_zero_equity_row(self):
        broker, store = MagicMock(), MagicMock()
        broker.get_account.return_value = {"equity": 0.0}
        r = _runner(broker=broker, store=store)
        r._session_start_equity = None
        with patch("orb_live.ops.reports.generate_daily_report"):
            r._run_eod(date(2026, 9, 15))
        store.record_equity.assert_not_called()

    def test_eod_records_when_equity_is_valid(self):
        broker, store = MagicMock(), MagicMock()
        broker.get_account.return_value = {"equity": 31_500.0}
        r = _runner(broker=broker, store=store)
        r._session_start_equity = 31_476.30
        with patch("orb_live.ops.reports.generate_daily_report"):
            r._run_eod(date(2026, 9, 15))
        store.record_equity.assert_called_once()


# ── 4. idle heartbeat ─────────────────────────────────────────────────────────

class TestBrokerHeartbeat:
    def test_healthy_broker_is_left_alone(self):
        broker = MagicMock()
        broker.verify_connection.return_value = True
        r = _runner(broker=broker)
        assert r.broker_heartbeat() is True
        broker.reconnect.assert_not_called()

    def test_dead_broker_is_reconnected(self):
        broker = MagicMock()
        broker.verify_connection.return_value = False
        broker.reconnect.return_value = True
        r = _runner(broker=broker)
        assert r.broker_heartbeat() is True
        broker.reconnect.assert_called_once()

    def test_probe_exception_counts_as_down(self):
        broker = MagicMock()
        broker.verify_connection.side_effect = OSError("10054")
        broker.reconnect.return_value = False
        r = _runner(broker=broker)
        assert r.broker_heartbeat() is False
        broker.reconnect.assert_called_once()


class _FakeClock:
    def __init__(self, fixed_now):
        self._fixed_now = fixed_now

    def now_et(self):
        return self._fixed_now

    def next_premarket_start(self):
        from orb_live.core.clock import MarketClock
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: self._fixed_now
        return mc.next_premarket_start()


class TestDaemonHeartbeat:
    def _run(self, runner, shutdown):
        from orb_live.runner.main import _run_daemon
        clock = _FakeClock(datetime(2026, 9, 16, 3, 0, tzinfo=ET))   # 5.5h to 08:30
        _run_daemon(runner, clock, _sleep=lambda _s: None,
                    _shutdown=shutdown, _sleep_interval=60.0)

    def test_idle_daemon_probes_the_broker_periodically(self):
        from orb_live.runner.main import BROKER_HEARTBEAT_SECS
        shutdown, beats = [False], [0]

        class R:
            def broker_heartbeat(self):
                beats[0] += 1
                return True

            def run_session(self, d):
                shutdown[0] = True

        self._run(R(), shutdown)
        expected = 5.5 * 3600 / BROKER_HEARTBEAT_SECS
        assert expected - 2 <= beats[0] <= expected + 2

    def test_heartbeat_failure_does_not_kill_the_daemon(self):
        shutdown, ran = [False], []

        class R:
            def broker_heartbeat(self):
                raise RuntimeError("boom")

            def run_session(self, d):
                ran.append(d)
                shutdown[0] = True

        self._run(R(), shutdown)
        assert ran == [date(2026, 9, 16)]

    def test_runner_without_heartbeat_is_tolerated(self):
        shutdown, ran = [False], []

        class R:
            def run_session(self, d):
                ran.append(d)
                shutdown[0] = True

        self._run(R(), shutdown)
        assert ran == [date(2026, 9, 16)]
