"""
runner/session_runner.py — Top-level session orchestrator.

Production session timeline (all times ET):
  ~08:30  pre-market:   Phase 1 gap scan + PS filter
   09:30  ORB window:   subscribe_bars begins; bars accumulate in bar_cache
   10:00  ORB closes:   Phase 2 (RTG + routing + preflight)
  10:00+  trade_active: per-bar dispatch loop (LOAD-BEARING ORDER below)
  16:00   eod_exit:     wait for eod_exit_hour, then flatten_all("eod_sweep")
  16:30   report:       generate_daily_report

LOAD-BEARING BAR DISPATCH ORDER (do not reorder):
  1. bar_cache.add_bar(symbol, bar)
  2. position_manager.on_bar(symbol, bar, ts) ← exit management
  3. strategy_engine.on_bar(symbol, bar, ts)  ← entry detection (ORB_COMPLETE only)
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import date, datetime, time as dtime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class LateSessionStart(RuntimeError):
    """Raised when a session for today is asked to start after the open.

    Distinct from ConnectionError on purpose: the daemon retries a
    ConnectionError on the same day, and must NOT retry this one.
    """


#: Latest wall-clock time a session for today may begin. Past the open the
#: 9:30-10:00 range can no longer be observed as it forms, so a session
#: started after it reads the ORB out of history and treats the current tape
#: as a breakout of it -- on 2026-09-17 that entered LABU 7.61 above the ORB
#: high, 90 minutes after the window closed.
LATE_START_CUTOFF = dtime(9, 30)


class SessionRunner:
    """
    Wires all sub-components together for one trading session.

    All components are injected at construction so the class is fully
    testable without broker credentials.  In production, build the components
    via `runner.main` and pass them in.
    """

    def __init__(
        self,
        config,                # LiveConfig
        broker,                # BrokerClient | DryRunBroker | MockBroker
        state_store,           # StateStore
        bar_cache,             # BarCache
        bar_router,            # BarRouter
        pre_market_job,        # PreMarketJob
        strategy_engine,       # StrategyEngine
        position_manager,      # LivePositionManager
        risk_gate,             # RiskGate
        underlying_store,      # UnderlyingDataStore
        clock,                 # MarketClock
        logger=None,
        _sleep=None,           # injectable for tests; defaults to time.sleep
    ):
        self._cfg         = config
        self._broker      = broker
        self._store       = state_store
        self._cache       = bar_cache
        self._router      = bar_router
        self._pre_market  = pre_market_job
        self._engine      = strategy_engine
        self._mgr         = position_manager
        self._gate        = risk_gate
        self._ul          = underlying_store
        self._clock       = clock
        self._log         = logger
        # _raw_sleep is the real (loop-pumping) sleep; _sleep wraps it so a
        # mid-session socket drop reconnects and resumes in place instead of
        # crashing the session (and losing position/subscription state).
        self._raw_sleep   = _sleep or getattr(broker, "sleep", time.sleep)
        self._sleep       = self._resilient_sleep

        self._session_date: Optional[date] = None
        self._phase: str = "idle"
        self._phase1_results = []
        self._watched: list[str] = []   # symbols currently subscribed to bars
        self._session_start_equity: Optional[float] = None  # captured at pre-market

        # Register bar dispatch with router. The 1-min listener drives cache,
        # exits (and is the entry fallback); the 5-sec entry
        # listener drives low-latency breakout detection only.
        self._router.register_listener(self._on_bar_dispatch)
        self._router.register_entry_listener(self._on_entry_bar_dispatch)

    # ── Connection resilience ─────────────────────────────────────────────────

    def _resilient_sleep(self, secs: float) -> None:
        """Sleep that survives an IB socket drop. If the broker disconnects
        mid-wait, reconnect + re-subscribe, then finish the remaining time — so
        the session (and its open positions) resume in place. Re-raises only if
        reconnection is impossible, letting the daemon handle a real outage."""
        end = time.monotonic() + secs
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            try:
                self._raw_sleep(remaining)
                return
            except ConnectionError as exc:
                if self._log:
                    self._log.critical("session_sleep_disconnect", error=str(exc))
                if not self._reconnect_and_resubscribe():
                    raise   # genuine outage — surface to the daemon backstop

    def _reconnect_and_resubscribe(self) -> bool:
        """Reconnect the broker and re-establish bar subscriptions (which are
        dropped by IB on disconnect). Returns False if the broker can't
        reconnect."""
        reconnect = getattr(self._broker, "reconnect", None)
        if reconnect is None or not reconnect():
            return False
        if self._watched:
            try:
                self._router.unsubscribe()
                self._router.subscribe(self._watched)
                if self._log:
                    self._log.warning("bars_resubscribed_after_reconnect",
                                      n=len(self._watched))
            except Exception as exc:
                if self._log:
                    self._log.error("resubscribe_after_reconnect_failed", error=str(exc))
        return True

    def reconnect_broker(self, max_attempts: int = 5) -> bool:
        """Reconnect the broker after a disconnect that escaped a session.
        Used by the daemon before resuming the same day."""
        reconnect = getattr(self._broker, "reconnect", None)
        if reconnect is None:
            return False
        try:
            return bool(reconnect(max_attempts=max_attempts))
        except Exception as exc:
            if self._log:
                self._log.error("reconnect_broker_failed", error=str(exc))
            return False

    def broker_heartbeat(self) -> bool:
        """Probe the broker while the daemon is idle; reconnect if it is dead.

        Called from the daemon's between-session sleep so a Gateway restart or
        an IBKR upstream loss is found and repaired overnight, not at 08:30.
        Returns True if the broker is usable afterwards.
        """
        probe = getattr(self._broker, "verify_connection", None)
        if probe is None:
            return True
        try:
            if probe():
                return True
        except Exception:
            pass
        if self._log:
            self._log.critical("daemon_idle_broker_down", action="reconnecting")
        ok = self.reconnect_broker()
        if self._log:
            if ok:
                self._log.warning("daemon_idle_broker_reconnected")
            else:
                self._log.critical("daemon_idle_reconnect_failed")
        return ok

    def _ensure_connected(self) -> None:
        """Reconnect before a session starts, or refuse to start.

        IB Gateway restarts daily. The daemon sleeps through it with a plain
        time.sleep (main._run_daemon), which neither pumps the asyncio loop nor
        checks the socket, so nothing observes the drop until 08:30.

        A dead IB socket does not raise -- it degrades quietly. get_account()
        returns equity 0.0, check_margin() falls back to full notional, and
        get_positions() returns [] so the book looks flat. The session would
        then run to completion with every entry rejected by the risk gate on
        zero equity: no crash, no ConnectionError, and the daemon's recovery
        path never reached. That is indistinguishable from a day with no
        qualifying gaps, which makes it the worst failure available here.

        Raises ConnectionError specifically, because that is what the daemon
        catches to retry the same session date. Any other exception type
        escapes the loop and kills the process.
        """
        # verify_connection round-trips; is_connected only reads a cached flag
        # that survives the peer vanishing (2026-08-28 -- see its docstring).
        is_conn = getattr(self._broker, "verify_connection", None)             or getattr(self._broker, "is_connected", None)
        if is_conn is None:
            return              # dry-run / test brokers do not implement it
        try:
            if is_conn():
                return
        except Exception:
            pass                # treat an unanswerable socket as down
        if self._log:
            self._log.critical("session_start_disconnected", action="reconnecting")
        if not self.reconnect_broker():
            raise ConnectionError(
                "broker disconnected at session start and reconnect failed")
        if self._log:
            self._log.warning("session_start_reconnected")

    # ── Main entry point ───────────────────────────────────────────────────────

    def _assert_not_late(self, session_date: date) -> None:
        """Refuse a session for today that is already past the open.

        Only today is time-checked: an explicit session_date for another day
        is a replay or a backfill, where 'late' has no meaning.
        """
        now = self._clock.now_et()
        if now.date() != session_date:
            return
        cutoff = datetime.combine(session_date, LATE_START_CUTOFF).replace(tzinfo=ET)
        if now < cutoff:
            return
        late_by = (now - cutoff).total_seconds()
        if self._log:
            self._log.critical(
                "session_start_too_late",
                date=str(session_date),
                now_et=now.isoformat(),
                cutoff_et=cutoff.isoformat(),
                late_by_seconds=round(late_by, 1),
            )
        raise LateSessionStart(
            f"session {session_date} asked to start at {now.time()} "
            f"({late_by / 60:.0f} min past {LATE_START_CUTOFF}); day skipped"
        )

    def run_session(self, session_date: date, allow_late: bool = False,
                    adopt_live_positions: bool = False) -> None:
        """Run a complete session.  Installs SIGTERM/SIGINT handlers.

        `adopt_live_positions` is set by recover(): keep and manage positions the
        broker still holds instead of flattening them at startup.
        """
        if not allow_late:
            self._assert_not_late(session_date)

        self._session_date = session_date
        self._session_start_equity = None

        # Before anything asks the broker a question. startup_reconcile reads
        # positions, and a dead socket answers [] rather than raising.
        self._ensure_connected()

        self._engine.new_session(session_date)

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

        # Purge stale DB rows before any trading begins so that a crashed prior
        # session's 'entering' or ghost 'open' rows never block re-entry.
        self._mgr.startup_reconcile(adopt=adopt_live_positions)

        try:
            self._phase = "pre_market"
            self._run_pre_market(session_date)
            self._phase = "open_eval"
            self._run_open_eval(session_date)
            self._phase = "orb_window"
            self._run_orb_window(session_date)
            self._phase = "trade_active"
            self._run_post_orb(session_date)
            self._phase = "eod_exit"
            self._run_eod(session_date)
        except SystemExit:
            raise
        except Exception as exc:
            if self._log:
                self._log.critical("session_crash", exc=str(exc), date=str(session_date))
            self._shutdown("crash")
            raise
        finally:
            self._phase = "idle"
            self._router.unsubscribe()

    def health_snapshot(self) -> dict:
        """Live session state for HealthServer's /status endpoint.

        Without this wired in, /status fell back to its defaults and reported
        `session_state: "idle"` with `todays_unrealized_pnl: 0.0` for the whole
        of a session that was holding positions (2026-09-18), which makes the
        documented pre_market → orb_window → trade_active → eod_exit monitoring
        flow useless and hides the kill-switch. Must never raise: the health
        server is what an operator checks when things are already wrong.
        """
        unrealized = 0.0
        try:
            unrealized = sum(
                float(p.get("unrealized_pl", 0.0) or 0.0)
                for p in self._broker.get_positions()
            )
        except Exception:
            pass

        try:
            killed = bool(self._gate.is_session_killed())
        except Exception:
            killed = False

        return {
            "session_date":          self._session_date or datetime.now(ET).date(),
            "session_state":         self._phase,
            "active_candidates":     len(self._watched),
            "todays_unrealized_pnl": round(unrealized, 2),
            "session_kill_active":   killed,
        }

    def recover(self, session_date: date) -> None:
        """
        Resume a crashed session.  Reconciles open positions from broker before
        re-entering the session loop at the current phase.
        """
        if self._log:
            self._log.info("session_recovery_start", date=str(session_date))
        self._mgr.reconcile_from_broker()
        # Resuming a crashed session mid-day is precisely the case the
        # late-start guard must not block. adopt_live_positions keeps the
        # positions reconcile_from_broker just confirmed: without it
        # startup_reconcile flattens them and pre-market re-enters on a stale
        # breakout, turning one position into two round trips (2026-09-18).
        self.run_session(session_date, allow_late=True, adopt_live_positions=True)

    # ── Phase runners ──────────────────────────────────────────────────────────

    def _fetch_start_equity(self, retries: int = 5, wait: float = 2.0):
        """Return a positive account equity, or None if it never becomes valid.

        get_account() returns 0.0 when the account download hasn't arrived. A 0
        must never flow into sizing/risk/session_pnl, so retry (pumping the IB
        loop via the raw sleep) before giving up.
        """
        equity = float(self._broker.get_account().get("equity", 0.0))
        attempt = 0
        while equity <= 0.0 and attempt < retries:
            self._raw_sleep(wait)
            equity = float(self._broker.get_account().get("equity", 0.0))
            attempt += 1
        return equity if equity > 0.0 else None

    def _run_pre_market(self, session_date: date) -> None:
        # Refresh underlying data and abort loudly if any are still stale.
        underlyings = {
            spec[0]
            for spec in self._cfg.prior_session_filters.values()
            if spec is not None
        }
        try:
            self._ul.refresh_and_assert_fresh(session_date, underlyings)
        except RuntimeError as exc:
            if self._log:
                self._log.critical("underlying_refresh_failed", error=str(exc))
            raise

        # A 0/negative equity means the account download wasn't ready and
        # get_account fell back — never trust it for sizing, the risk gate, or
        # the session_pnl baseline. If it stays invalid the broker is not
        # usable: raise ConnectionError so the daemon reconnects and retries.
        # 2026-09-15 ran a whole session on equity 0.0 and planned nothing.
        equity_val = self._fetch_start_equity()
        if equity_val is None:
            if self._log:
                self._log.critical("session_start_equity_unavailable", date=str(session_date))
            raise ConnectionError("account equity unavailable at session start")
        equity = equity_val
        self._session_start_equity = equity_val
        self._gate.session_start(equity, session_date)
        self._store.upsert_day_state(session_date, phase="pre_market")

        is_half  = self._clock.is_half_day()
        close_t  = self._clock.effective_close()
        if self._log:
            self._log.info(
                "pre_market_start", equity=equity, date=str(session_date),
                is_half_day=is_half,
                todays_close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
            )
        if is_half and self._log:
            self._log.warning(
                "half_day_session",
                close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
                date=str(session_date),
            )

        # Wait until 09:31 — the 9:30 bar must be closed before Phase 1 can
        # read its close price as the gap reference.
        open_eval_dt = self._clock.next_open_eval_start()
        secs = (open_eval_dt - self._clock.now_et()).total_seconds()
        if secs > 0:
            if self._log:
                self._log.info("waiting_for_open_eval", seconds=round(secs, 1))
            self._sleep(secs)

    def _run_open_eval(self, session_date: date) -> None:
        """Run Phase 1 at 09:31 ET using the 9:30 bar close as the gap reference."""
        if self._log:
            self._log.info("open_eval_start", date=str(session_date))

        self._phase1_results = self._pre_market.run_phase1(session_date)
        p1_symbols = [r.symbol for r in self._phase1_results]

        self._store.upsert_day_state(
            session_date, phase="open_eval", n_prequalified=len(p1_symbols)
        )
        if self._log:
            self._log.info("phase1_complete", n=len(p1_symbols), symbols=p1_symbols)

    def _run_orb_window(self, session_date: date) -> None:
        """Sleep until the ORB window closes at 10:00 ET."""
        self._store.upsert_day_state(session_date, phase="orb_window")
        orb_end = self._clock.orb_end_et(self._cfg.orb_minutes)
        secs    = (orb_end - self._clock.now_et()).total_seconds()
        if secs > 0:
            if self._log:
                self._log.info("orb_window_wait", seconds=round(secs, 1))
            self._sleep(secs)

    def _run_post_orb(self, session_date: date) -> None:
        """Run Phase 2, then wait until the EOD exit."""
        self._store.upsert_day_state(session_date, phase="trade_active")

        if not self._phase1_results:
            if self._log:
                self._log.info("no_phase1_candidates", date=str(session_date))
            self._wait_until_eod(session_date)
            return

        p2_results = self._pre_market.run_phase2(session_date, self._phase1_results)

        # Arm the margin gate from the measured profile table before any entry
        # can be placed. This does NOT depend on _prewarm_candidate_margin,
        # which is unreachable: the loop below reads bar_cache, and the market
        # data subscription that fills it is deliberately deferred until after
        # the loop (IB line cap). Without this the budget stays None and the
        # leveraged-ETF margin check never runs at all.
        self._seed_margin_rates()


        for p2 in p2_results:
            symbol = p2.symbol

            if p2.orb is None:
                self._engine.on_orb_complete(symbol, p2, None)
                continue

            # Measure IB's TRUE initial-margin rate for this candidate, on the
            # main thread (check_margin blocks — unsafe in a bar callback).
            # MUST stay above the bar-cache read: the market-data subscription
            # that fills that cache is deliberately deferred until after this
            # loop (subscribing all ~59 symbols at open exceeds IB's line cap),
            # so the cache is always empty here and the `raw.empty` branch below
            # used to skip this entirely. House requirements move, so the CSV
            # table seeded above is a fallback, not the source of truth.
            self._prewarm_candidate_margin(p2)

            # ORB bars from cache, for the engine's opening-range computation.
            raw = self._cache.get_bars(symbol)
            if raw.empty:
                self._engine.on_orb_complete(symbol, p2, None)
                continue

            raw = raw.copy()
            raw["timestamp"] = _to_et(raw["timestamp"])
            raw = raw.set_index("timestamp")

            # Filter to ORB window only for seeding
            orb_end_dt   = self._clock.orb_end_et(self._cfg.orb_minutes)
            orb_start_dt = orb_end_dt - timedelta(minutes=self._cfg.orb_minutes)
            mask = (raw.index >= _aware(orb_start_dt)) & (raw.index < _aware(orb_end_dt))
            orb_df = raw.loc[mask]

            self._engine.on_orb_complete(symbol, p2, orb_df)

            if self._log:
                self._log.info(
                    "phase2_ready", symbol=symbol,
                    is_candidate=p2.is_candidate,
                    rtg_excluded=p2.rtg_excluded,
                )

        # Subscribe only the day's candidates after ORB close (~10:00 ET).
        # Subscribing all ~59 symbols at open exceeds IB's market-data line cap,
        # causing IB to silently deliver no bars at all (silent failure — no error).
        watched = [r.symbol for r in p2_results if r.is_candidate]
        self._watched = watched   # remembered so we can re-subscribe after a reconnect
        if watched:
            self._router.subscribe(watched)
            if self._log:
                self._log.info("bar_subscription_started", n=len(watched), symbols=watched)
            # Watchdog: 5-min grace period, then CRITICAL + REST fallback if any
            # candidate still has zero bars (reqRealTimeBars silently rejected by IB).
            self._sleep(300)
            zero_bar_syms = [s for s in watched if self._router.bars_received(s) == 0]
            if zero_bar_syms:
                for sym in zero_bar_syms:
                    if self._log:
                        self._log.critical(
                            "zero_bars_watchdog", symbol=sym,
                            msg="No bars 5 min post-ORB; re-issuing reqRealTimeBars",
                        )
                # Re-issue subscriptions on the main thread (IB is not thread-safe;
                # the original subscribe call must already be on the main thread, and
                # re-subscribing here recovers any silently-dropped registrations).
                self._router.unsubscribe()
                self._router.subscribe(watched)

        self._wait_until_eod(session_date)

    def _seed_margin_rates(self) -> None:
        """Push measured per-symbol, per-side margin rates into the manager.

        Rates come from master_universe.csv, populated by
        orb_live/scripts/measure_universe_margin.py against IB whatIf. Best
        effort: a failure here leaves the manager on its conservative 1.0
        default, which over-reserves rather than over-trades.
        """
        try:
            instruments = getattr(self._cfg, "instruments", None) or {}
            rates = {}
            for sym, inst in instruments.items():
                for direction in (1, -1):
                    rate = inst.margin_rate(direction)
                    if rate:
                        rates[(sym, direction)] = float(rate)
            if rates:
                self._mgr.seed_margin_rates(rates)
            elif self._log:
                self._log.warning("margin_rates_missing_from_profile")
        except Exception as exc:
            if self._log:
                self._log.warning("seed_margin_rates_failed", exc=str(exc))

    def _prewarm_candidate_margin(self, p2) -> None:
        """Cache IB's true initial-margin rate for a tradable candidate using a
        representative order sized like the planned entry. Best-effort — any
        failure leaves the symbol at the conservative default rate."""
        if not getattr(p2, "is_candidate", False):
            return
        if p2.orb is None or getattr(p2, "size_mult", 0.0) == 0.0:
            return
        try:
            price = float(p2.orb["high"] if p2.gap_direction == 1 else p2.orb["low"])
            # Match live sizing: base_notional_pct of equity, falling back to
            # the fixed notional. The rate is scale-invariant, but a probe the
            # size of the real order also surfaces size-dependent rejections.
            base = 0.0
            pct = float(getattr(self._cfg, "base_notional_pct", 0.0) or 0.0)
            if pct > 0:
                try:
                    base = pct * float(self._broker.get_account().get("equity", 0.0))
                except Exception:
                    base = 0.0
            if base <= 0:
                base = float(getattr(self._cfg, "v1_base_notional", 0.0) or 0.0)
            mult  = float(getattr(p2, "size_mult", 1.0) or 1.0)
            if price <= 0 or base <= 0:
                return
            qty  = max(1, int((base * mult) / price))
            side = "buy" if p2.gap_direction == 1 else "sell"
            self._mgr.prewarm_margin(p2.symbol, side, qty, price)
        except Exception as exc:
            if self._log:
                self._log.warning("prewarm_margin_error", symbol=p2.symbol, exc=str(exc))

    def _wait_until_eod(self, session_date: date) -> None:
        # Respect half-day schedules (e.g. July 3 closes at 13:00 ET not 16:00).
        # Wake `eod_flatten_lead_secs` BEFORE the close so the EOD flatten runs
        # while regular-hours liquidity is still available (see _run_eod).
        close_t = self._clock.effective_close()
        now     = self._clock.now_et()
        eod     = now.replace(hour=close_t.hour, minute=close_t.minute,
                               second=0, microsecond=0)
        lead    = int(getattr(self._cfg, "eod_flatten_lead_secs", 120))
        secs    = (eod - now).total_seconds() - lead

        if self._clock.is_half_day():
            if self._log:
                self._log.warning(
                    "half_day_detected",
                    close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
                    date=str(session_date),
                )

        if secs > 0:
            if self._log:
                self._log.info("waiting_for_eod", seconds=round(secs, 1),
                               close_et=f"{close_t.hour:02d}:{close_t.minute:02d}",
                               flatten_lead_secs=lead)
            self._wait_with_protective_sweeps(secs)

    #: How often, while waiting for the close, to re-check that every open
    #: position still has a live take-profit at the broker. IB can drop a
    #: bracket child silently, and the position then runs unprotected.
    PROTECTIVE_SWEEP_SECS: float = 300.0

    def _wait_with_protective_sweeps(self, secs: float) -> None:
        """Wait `secs`, re-verifying protective orders between chunks.

        The wait itself is unchanged in total duration — chunking exists only
        so the sweep gets to run. A failing sweep must never stop the runner
        from reaching its EOD flatten, so it is caught and logged.
        """
        sweep = getattr(self._mgr, "verify_protective_orders", None)
        remaining = float(secs)
        while remaining > 0:
            chunk = min(self.PROTECTIVE_SWEEP_SECS, remaining)
            self._sleep(chunk)
            remaining -= chunk
            if remaining <= 0 or sweep is None:
                continue
            try:
                sweep()
            except Exception as exc:
                if self._log:
                    self._log.critical("protective_sweep_failed", exc=str(exc))

    def _run_eod(self, session_date: date) -> None:
        """Flatten all positions ~`eod_flatten_lead_secs` before the close, while
        regular-hours liquidity is still available. Market exits sent after 16:00
        ET are rejected/unfilled by IB on leveraged ETFs, so we must be flat
        BEFORE the close rather than after it."""
        self._mgr.flatten_all("eod_sweep")
        self._store.upsert_day_state(session_date, phase="closed")

        # The flatten books each exit before its fill exists. Re-price every
        # closed trade from the broker's executions (actual fills + commission)
        # before the daily report sums them. Bookkeeping only: never let it
        # stand between the session and its close-out.
        try:
            self._mgr.reconcile_closed_trades(session_date)
        except Exception as exc:
            if self._log:
                self._log.error("closed_trade_reconcile_crashed", exc=str(exc))

        # Real session P&L = end equity (post-flatten) − start equity (pre-market).
        # IB's equity (NetLiquidation) already includes open-position marks, so the
        # delta reflects realized session P&L. Fall back to a flat session if the
        # start baseline is missing (e.g. EOD reached without a pre-market phase).
        end_equity   = float(self._broker.get_account().get("equity", 0.0))
        start_equity = (self._session_start_equity
                        if self._session_start_equity is not None else end_equity)
        session_pnl  = end_equity - start_equity
        if end_equity <= 0.0:
            # A 0 row reads as a total wipeout in every equity-curve statistic.
            if self._log:
                self._log.critical("session_end_equity_unavailable", date=str(session_date))
        else:
            self._store.record_equity(session_date, start_equity, end_equity, session_pnl)
            if self._log:
                self._log.info("session_equity_recorded",
                               start_equity=round(start_equity, 2),
                               end_equity=round(end_equity, 2),
                               session_pnl=round(session_pnl, 2))

        from orb_live.ops.reports import generate_daily_report
        generate_daily_report(self._store, session_date, logger=self._log)

        if self._log:
            self._log.info("session_complete", date=str(session_date))

    # ── Bar dispatch (LOAD-BEARING ORDER) ─────────────────────────────────────

    def _is_post_orb(self, ts) -> bool:
        """True once ts is at/after the ORB close (market open + orb_minutes)."""
        bar_time = ts.time() if hasattr(ts, "time") else dtime(0, 0)
        orb_end_minutes = (
            self._cfg.strategy_config.market_open_hour * 60
            + self._cfg.strategy_config.market_open_minute
            + self._cfg.orb_minutes
        )
        orb_end = dtime(orb_end_minutes // 60, orb_end_minutes % 60)
        return bar_time >= orb_end

    def _on_entry_bar_dispatch(self, symbol: str, bar: dict) -> None:
        """
        5-sec entry path: forward the raw bar to the entry state machine ONLY.

        Deliberately does NOT touch cache/position_manager — those
        stay on the 1-min pipeline (_on_bar_dispatch). The first path to see the
        breakout flips the symbol to IN_POSITION; the other then no-ops via the
        engine's state guard, so running both is idempotent.
        """
        ts = bar.get("timestamp")
        if ts is None or not self._is_post_orb(ts):
            return
        self._engine.on_bar(symbol, bar, ts)

    def _on_bar_dispatch(self, symbol: str, bar: dict) -> None:
        """
        Called by BarRouter for every incoming bar (live or replayed).

        ORDER IS LOAD-BEARING — do not reorder steps 2/3/4:
          1. bar_cache.add_bar       — persist to cache
          3. position_manager.on_bar — exit checks for open positions
          4. strategy_engine.on_bar  — entry detection (ORB_COMPLETE symbols only)
        """
        # 1. Cache
        self._cache.add_bar(symbol, bar)

        # Only process post-ORB bars through the engine
        ts = bar.get("timestamp")
        if ts is None:
            return

        if not self._is_post_orb(ts):
            return  # ORB window bar — cached but not dispatched to engine

        # 2. Position management (exits)
        self._mgr.on_bar(symbol, bar, ts)

        # 3. Entry detection
        if self._log:
            self._log.debug(
                "engine_on_bar", symbol=symbol,
                bar_time=str(ts.time() if hasattr(ts, "time") else ts),
                close=bar.get("close"),
            )
        self._engine.on_bar(symbol, bar, ts)

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def _handle_signal(self, signum, frame) -> None:
        reason = "sigterm" if signum == signal.SIGTERM else "sigint"
        self._shutdown(reason)
        sys.exit(0)

    def _shutdown(self, reason: str) -> None:
        if self._log:
            self._log.warning("shutdown", reason=reason)
        try:
            self._mgr.flatten_all(reason)
        except Exception as exc:
            if self._log:
                self._log.error("shutdown_flatten_failed", exc=str(exc))
        self._router.unsubscribe()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_et(series):
    """Coerce a timestamp Series to ET-aware datetimes."""
    s = series.copy()
    if hasattr(s, "dt"):
        if s.dt.tz is None:
            s = s.dt.tz_localize(ET)
        else:
            s = s.dt.tz_convert(ET)
    return s


def _aware(dt: datetime) -> datetime:
    """Return dt as an ET-aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)
