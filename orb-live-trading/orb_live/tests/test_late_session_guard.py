"""A session must refuse to start late. Measured 2026-09-17.

The daemon's countdown had drifted (see test_daemon_clock_drift), so it woke
for its "08:30" pre-market at 11:01 ET. Nothing in the session objected. It
scanned gaps off correct historical data, read the real 09:30-10:00 ORB out of
history, and then treated the 11:01 tape as a breakout of it:

    JNUG  ORB 171.47-176.07   "breakout" 176.77 at 11:01:16 -> filled 176.88
    LABU  ORB 263.72-271.50   "breakout" 279.11 at 11:01:16 -> filled 274.23

LABU entered 7.61 above the ORB high, ~90 minutes after the window closed.
Neither trade is the strategy; both are noise booked as live results, and the
pair lost ~41 dollars before being flattened by hand.

The rule: a session for TODAY may only begin before the open. After 09:30 the
ORB window can no longer be observed as it forms, so the day is skipped, not
traded. Two carve-outs, both deliberate:

  * recover() bypasses it -- resuming a crashed mid-day session is exactly the
    case where starting at 13:00 is correct;
  * an explicit session_date that is not today is a replay/backfill and is
    never time-checked.

The daemon must treat the refusal as "skip this day", not as an error to retry:
a retry loop would re-raise every 60 seconds until 16:00.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")
SESSION = date(2026, 9, 17)


def _runner(now: datetime):
    from orb_live.runner.session_runner import SessionRunner

    clock = MagicMock()
    clock.now_et.return_value = now
    broker = MagicMock()
    broker.verify_connection.return_value = True
    mgr = MagicMock()
    runner = SessionRunner(
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
        clock=clock,
        logger=MagicMock(),
        _sleep=lambda _: None,
    )
    phases = []
    for name in ("_run_pre_market", "_run_open_eval", "_run_orb_window",
                 "_run_post_orb", "_run_eod"):
        setattr(runner, name, lambda d, n=name: phases.append(n))
    return runner, mgr, phases


def _at(h: int, m: int, s: int = 0) -> datetime:
    return datetime(2026, 9, 17, h, m, s, tzinfo=ET)


class TestTheSessionRefusesToStartLate:
    def test_the_09_17_start_is_refused(self):
        """11:01 ET, the measured case."""
        from orb_live.runner.session_runner import LateSessionStart
        runner, _, phases = _runner(_at(11, 1))

        with pytest.raises(LateSessionStart):
            runner.run_session(SESSION)

        assert phases == [], "a late session ran its phases anyway"

    def test_a_timely_start_is_untouched(self):
        runner, _, phases = _runner(_at(8, 30))
        runner.run_session(SESSION)
        assert phases == ["_run_pre_market", "_run_open_eval",
                          "_run_orb_window", "_run_post_orb", "_run_eod"]

    def test_the_cutoff_is_the_open(self):
        from orb_live.runner.session_runner import LateSessionStart
        runner, _, phases = _runner(_at(9, 29, 59))
        runner.run_session(SESSION)
        assert phases, "09:29:59 must still be allowed to start"

        runner, _, _ = _runner(_at(9, 30, 0))
        with pytest.raises(LateSessionStart):
            runner.run_session(SESSION)

    def test_nothing_is_reconciled_or_subscribed_on_a_refusal(self):
        """The guard runs before the broker is touched: startup_reconcile
        purges DB rows and asks for positions, and new_session resets engine
        state. A day we are not trading must leave all of it alone."""
        from orb_live.runner.session_runner import LateSessionStart
        runner, mgr, _ = _runner(_at(11, 1))
        engine = runner._engine

        with pytest.raises(LateSessionStart):
            runner.run_session(SESSION)

        mgr.startup_reconcile.assert_not_called()
        engine.new_session.assert_not_called()
        runner._broker.verify_connection.assert_not_called()

    def test_the_refusal_is_logged_loudly(self):
        from orb_live.runner.session_runner import LateSessionStart
        runner, _, _ = _runner(_at(11, 1))
        with pytest.raises(LateSessionStart):
            runner.run_session(SESSION)
        events = [c.args[0] for c in runner._log.critical.call_args_list]
        assert "session_start_too_late" in events


class TestTheCarveOuts:
    def test_recover_may_start_at_any_hour(self):
        """A crashed session resumed at 13:00 is the whole point of --recover."""
        runner, mgr, phases = _runner(_at(13, 0))
        runner.recover(SESSION)
        assert phases, "recover() must not be blocked by the late guard"
        mgr.reconcile_from_broker.assert_called_once()

    def test_an_explicit_other_date_is_not_time_checked(self):
        """Replay/backfill of a past date: 'late' is meaningless there."""
        runner, _, phases = _runner(_at(11, 1))
        runner.run_session(date(2026, 9, 10))
        assert phases

    def test_a_future_date_is_not_time_checked(self):
        runner, _, phases = _runner(_at(11, 1))
        runner.run_session(date(2026, 9, 18))
        assert phases


class TestTheDaemonSkipsTheDay:
    def _clock(self, now: datetime, premarket: datetime):
        c = MagicMock()
        c.now_et.return_value = now
        c.next_premarket_start.return_value = premarket
        return c

    def test_a_late_refusal_does_not_retry_all_day(self):
        """ConnectionError means 'retry this day'. LateSessionStart means the
        opposite -- the day is gone. Retrying would re-raise every minute
        until 16:00."""
        from orb_live.runner.main import _run_daemon
        from orb_live.runner.session_runner import LateSessionStart

        clock = self._clock(_at(11, 1), _at(8, 30))
        shutdown = [False]
        calls = []

        class R:
            def run_session(self, d):
                calls.append(d)
                raise LateSessionStart("too late")

        def _sleep(_secs):
            if len(calls) >= 1:
                shutdown[0] = True

        _run_daemon(R(), clock, _sleep=_sleep, _shutdown=shutdown,
                    _sleep_interval=60.0)

        assert calls == [SESSION], f"run_session called {len(calls)}x, not once"

    def test_the_daemon_survives_the_refusal(self):
        """It must keep running and pick up the next day, not die."""
        from orb_live.runner.main import _run_daemon
        from orb_live.runner.session_runner import LateSessionStart

        now = [_at(11, 1)]
        premarket = [_at(8, 30)]
        clock = MagicMock()
        clock.now_et.side_effect = lambda: now[0]
        clock.next_premarket_start.side_effect = lambda: premarket[0]
        shutdown = [False]
        ran = []

        class R:
            def run_session(self, d):
                if d == SESSION:
                    raise LateSessionStart("too late")
                ran.append(d)
                shutdown[0] = True

        def _sleep(_secs):
            # the clock rolls past the close; tomorrow's pre-market is next
            now[0] = datetime(2026, 9, 18, 8, 29, tzinfo=ET)
            premarket[0] = datetime(2026, 9, 18, 8, 30, tzinfo=ET)

        _run_daemon(R(), clock, _sleep=_sleep, _shutdown=shutdown,
                    _sleep_interval=60.0)

        assert ran == [date(2026, 9, 18)]
