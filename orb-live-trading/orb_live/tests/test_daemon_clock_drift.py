"""The daemon must wake from the clock, not from a countdown. Measured 2026-09-17.

The pre-market sleep decremented a local counter by the nominal chunk size
(`remaining -= chunk`) instead of re-reading the clock. Anything that blocked
inside the loop -- and the broker heartbeat blocks for a minute or more per
failing reconnect -- drifted the counter away from wall time. On 09-17 the
daemon was still counting down to an 08:30 pre-market at 10:25 ET: it never
logged daemon_session_starting, and the session was lost with no error anywhere.

Three rules:
  * remaining time is always (next_premarket - clock.now_et()), recomputed;
  * a heartbeat near the open is skipped rather than allowed to delay it;
  * the broker client gets the logger, so its reconnect and gateway-restart
    events are not silently dropped (every ib_reconnect_* and ibc_* line was
    invisible on 09-17 because IBClient was built without one).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")


class _MovingClock:
    """Clock the injected sleep advances, so tests model real elapsed time."""

    def __init__(self, start: datetime):
        self.now = start

    def advance(self, secs: float) -> None:
        self.now += timedelta(seconds=secs)

    def now_et(self) -> datetime:
        return self.now

    def next_premarket_start(self) -> datetime:
        from orb_live.core.clock import MarketClock
        mc = MarketClock(broker_client=None)
        mc.now_et = lambda: self.now
        return mc.next_premarket_start()


def _run(runner, clock, shutdown, sleep_fn=None, interval=60.0):
    from orb_live.runner.main import _run_daemon
    _run_daemon(runner, clock, _sleep=sleep_fn or clock.advance,
                _shutdown=shutdown, _sleep_interval=interval)


class TestSleepFollowsTheClock:
    def test_blocking_work_cannot_delay_the_session(self):
        """The 09-17 failure: each heartbeat burns wall time the countdown
        never sees. The session must still start on schedule."""
        clock = _MovingClock(datetime(2026, 9, 17, 3, 0, tzinfo=ET))  # 5.5h early
        shutdown, started = [False], []

        class R:
            def broker_heartbeat(self):
                clock.advance(150)      # a failing reconnect storm, 2.5 min
                return False

            def run_session(self, d):
                started.append(clock.now_et())
                shutdown[0] = True

        _run(R(), clock, shutdown)

        assert started, "daemon never started the session"
        premarket = datetime(2026, 9, 17, 8, 30, tzinfo=ET)
        late_by = (started[0] - premarket).total_seconds()
        assert 0 <= late_by <= 180, f"session started {late_by:.0f}s off schedule"

    def test_a_clock_jump_forward_is_noticed(self):
        """Laptop suspend, VM pause, NTP correction: the wait ends at once."""
        clock = _MovingClock(datetime(2026, 9, 17, 3, 0, tzinfo=ET))
        shutdown, started = [False], []

        def sleep_fn(secs):
            clock.advance(secs)
            if clock.now < datetime(2026, 9, 17, 8, 0, tzinfo=ET):
                clock.advance(3600)     # an hour vanishes each hop

        class R:
            def run_session(self, d):
                started.append(clock.now_et())
                shutdown[0] = True

        _run(R(), clock, shutdown, sleep_fn=sleep_fn)

        assert started and started[0].date() == date(2026, 9, 17)

    def test_no_early_start(self):
        """Following the clock must not make it fire before pre-market."""
        clock = _MovingClock(datetime(2026, 9, 17, 3, 0, tzinfo=ET))
        shutdown, started = [False], []

        class R:
            def run_session(self, d):
                started.append(clock.now_et())
                shutdown[0] = True

        _run(R(), clock, shutdown)

        assert started[0] >= datetime(2026, 9, 17, 8, 30, tzinfo=ET)


class TestHeartbeatIsTimeBoxed:
    def test_no_heartbeat_immediately_before_the_open(self):
        from orb_live.runner.main import HEARTBEAT_QUIET_SECS
        clock = _MovingClock(datetime(2026, 9, 17, 8, 30, tzinfo=ET)
                             - timedelta(seconds=HEARTBEAT_QUIET_SECS - 5))
        shutdown, beats = [False], []

        class R:
            def broker_heartbeat(self):
                beats.append(clock.now_et())
                return True

            def run_session(self, d):
                shutdown[0] = True

        _run(R(), clock, shutdown, interval=10.0)

        assert not beats, "a heartbeat ran inside the pre-open quiet window"

    def test_heartbeat_still_runs_when_there_is_time(self):
        clock = _MovingClock(datetime(2026, 9, 17, 3, 0, tzinfo=ET))
        shutdown, beats = [False], []

        class R:
            def broker_heartbeat(self):
                beats.append(clock.now_et())
                return True

            def run_session(self, d):
                shutdown[0] = True

        _run(R(), clock, shutdown)

        assert len(beats) > 10


class TestBrokerGetsTheLogger:
    def test_build_client_from_env_accepts_and_stores_a_logger(self):
        from unittest.mock import MagicMock, patch
        from orb_live.data.ib_client import build_client_from_env
        log = MagicMock()
        with patch("orb_live.data.ib_client.IB", return_value=MagicMock()):
            client = build_client_from_env(paper=True, logger=log)
        assert client._log is log

    def test_the_daemon_passes_its_logger_to_the_broker(self):
        """Every ib_reconnect_* and ibc_* event was dropped on 09-17 because
        the daemon built IBClient without a logger. Assert on the client the
        daemon's own construction path hands back, not on the source text --
        the first version of this test passed while the call site still built
        the broker loggerless.
        """
        from unittest.mock import MagicMock, patch
        from orb_live.data.ib_client import IBClient
        from orb_live.runner.main import build_broker_from_env
        log = MagicMock()
        with patch("orb_live.data.ib_client.IB", return_value=MagicMock()), \
             patch.object(IBClient, "connect", lambda self: None):
            client = build_broker_from_env(paper=True, logger=log)
        assert client._log is log

    def test_build_components_wires_the_logger_through(self):
        """The call site itself, since that is what was actually broken."""
        import inspect
        from orb_live.runner import main
        src = inspect.getsource(main._build_components)
        assert "build_broker_from_env(paper=paper, logger=" in src, (
            "_build_components must pass its logger to the broker")
