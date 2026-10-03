"""Regression tests for the 2026-09-18 restart incident.

A session crashed mid-day holding two positions. Restarting with --recover
flattened both and then replayed pre-market, which re-entered the same two
symbols on breakouts that had fired an hour earlier — one position per symbol
became two round trips, and the P&L of the flattened pair never reached
closed_trades so the daily report under-reported the session by $211.89.

Covered here:
  1. --recover adopts live positions instead of flattening them.
  2. A reconcile-flatten (the non-adopt path) books a closed_trades row.
  3. A leg that could not be polled is not reported as unprotected.
  4. A pending entry fill is not reported as a rejection.
  5. /status reflects the live session rather than defaulting to idle.
"""

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from orb_live.tests.test_position_manager import _build_mgr, SYMBOL, TRADE_DATE


def _save_open_row(store, symbol=SYMBOL, qty=100, direction=1):
    store.save_open_position(
        symbol=symbol, trade_date=TRADE_DATE,
        direction=direction, status="open",
        entry_price=100.0, actual_entry_price=100.0,
        qty=qty, entry_shares=qty, remaining=qty,
        orb_range=1.0, stop_price=99.0, current_stop=99.0,
        tp1_price=101.0, tp2_price=102.0,
        tp1_shares=qty, tp2_shares=0, tp3_shares=0,
        tp1_hit=False, tp2_hit=False, tp3_hit=False,
        use_trail_atp1=False, trail_atp1_dist=0.0,
        max_fav=0.0, post_tp2_mfe=0.0, decision_reason="",
        stop_order_id="900", tp1_order_id="901",
    )


# ── 1. --recover adopts rather than flattens ─────────────────────────────────

def test_recover_adopts_live_position_instead_of_flattening(mock_broker, tmp_store):
    """startup_reconcile(adopt=True) must keep a broker-backed position.

    This is the core 2026-09-18 bug: the flatten destroyed a correctly tracked
    live position, and the pre-market replay that followed re-entered it.
    """
    _save_open_row(tmp_store)
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order") as mock_flat:
        mgr.startup_reconcile(adopt=True)

    mock_flat.assert_not_called(), "recover must not flatten a live position"
    row = tmp_store.get_open_position(SYMBOL)
    assert row is not None and row["status"] == "open", \
        "the DB row must survive adoption"
    assert SYMBOL in mgr._positions, \
        "the position must be loaded into memory so exits/EOD manage it"
    assert mgr._positions[SYMBOL].remaining == 100


def test_adopted_position_keeps_its_stop_and_targets(mock_broker, tmp_store):
    """Adoption must rebuild the full Position, not a stub — the EOD sweep and
    exit logic read current_stop/tp1_price off it."""
    _save_open_row(tmp_store)
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.startup_reconcile(adopt=True)

    pos = mgr._positions[SYMBOL]
    assert pos.current_stop == 99.0
    assert pos.tp1_price == 101.0
    assert pos.direction == 1
    assert pos.session_date == TRADE_DATE
    assert pos.stop_order_id == "900"
    assert pos.tp1_order_id == "901"


def test_adopt_syncs_remaining_to_broker_truth(mock_broker, tmp_store):
    """If a leg filled while the runner was dead, adopt the broker's share
    count, not the stale DB one."""
    _save_open_row(tmp_store, qty=100)
    mock_broker.set_broker_position(SYMBOL, 60)   # 40 filled while down
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.startup_reconcile(adopt=True)

    assert mgr._positions[SYMBOL].remaining == 60
    assert tmp_store.get_open_position(SYMBOL)["remaining"] == 60


def test_default_start_still_flattens(mock_broker, tmp_store):
    """A normal (non-recover) start keeps the clean-slate guarantee: an
    unmanaged broker position with no in-memory owner is the greater risk."""
    _save_open_row(tmp_store)
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-1"}) as mock_flat:
        mgr.startup_reconcile()

    mock_flat.assert_called_once()
    assert tmp_store.get_open_position(SYMBOL) is None


# ── 2. Reconcile-flatten books the trade ─────────────────────────────────────

def test_reconcile_flatten_writes_a_closed_trade(mock_broker, tmp_store):
    """The flatten moves real shares, so it must leave a closed_trades row.

    Without this the daily report silently omits the P&L (2026-09-18 lost
    $211.89 of realized gain from its own summary).
    """
    _save_open_row(tmp_store, qty=100, direction=1)
    mock_broker.set_broker_position(SYMBOL, 100)
    mock_broker.set_quote(bid=104.90, ask=105.10)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-2"}):
        mgr.startup_reconcile()

    trades = tmp_store.get_closed_trades(TRADE_DATE)
    assert len(trades) == 1, "reconcile-flatten must book exactly one trade"
    t = trades[0]
    assert t["symbol"] == SYMBOL
    assert t["exit_reason"] == "RECONCILE_FLATTEN"
    assert t["qty"] == 100
    # Long 100 @ 100.00 flattened at the 105.00 mid → ~+$500.
    assert t["dollar_pnl"] == pytest.approx(500.0, abs=1.0)


def test_reconcile_flatten_books_short_side_correctly(mock_broker, tmp_store):
    """A short that is flattened below its entry is a gain, not a loss."""
    _save_open_row(tmp_store, qty=100, direction=-1)
    mock_broker.set_broker_position(SYMBOL, -100)
    mock_broker.set_quote(bid=94.90, ask=95.10)
    mgr = _build_mgr(mock_broker, tmp_store)

    with patch.object(mock_broker, "submit_market_order",
                      return_value={"id": "flat-3"}):
        mgr.startup_reconcile()

    t = tmp_store.get_closed_trades(TRADE_DATE)[0]
    assert t["dollar_pnl"] == pytest.approx(500.0, abs=1.0)


def test_adopted_position_books_no_trade(mock_broker, tmp_store):
    """Adoption moves no shares, so it must not fabricate a closed trade."""
    _save_open_row(tmp_store)
    mock_broker.set_broker_position(SYMBOL, 100)
    mgr = _build_mgr(mock_broker, tmp_store)

    mgr.startup_reconcile(adopt=True)

    assert tmp_store.get_closed_trades(TRADE_DATE) == []


# ── 3. Bracket-leg verification must not cry wolf ────────────────────────────

class _FakeTrade:
    def __init__(self, status):
        self.orderStatus = SimpleNamespace(status=status)
        self.order = SimpleNamespace(orderId=1)


def _client():
    from orb_live.data.ib_client import IBClient
    return IBClient.__new__(IBClient)


def test_unpollable_leg_is_reported_as_deferred_not_unverified():
    """Entries are placed from inside the IB event loop, where ib.sleep()
    raises. That is 'cannot check here', not 'unprotected' — reporting it as
    CRITICAL fired on 4/4 entries on 2026-09-18 while all legs were resting."""
    client = _client()
    client._ib = SimpleNamespace(
        sleep=lambda _: (_ for _ in ()).throw(RuntimeError("loop already running"))
    )
    live, checkable = client._wait_leg_live(_FakeTrade("PendingSubmit"), timeout=1.0)

    assert live is True,      "must not let verification break or replace the entry"
    assert checkable is False, "must not claim the leg was confirmed"


def test_pollable_leg_that_never_goes_live_is_still_flagged():
    """The real failure — IB silently dropping a leg we *can* poll — must keep
    reporting itself."""
    client = _client()
    client._ib = SimpleNamespace(sleep=lambda _: None)
    live, checkable = client._wait_leg_live(_FakeTrade("PendingSubmit"), timeout=0.3)

    assert live is False
    assert checkable is True


def test_live_leg_reports_verified():
    client = _client()
    client._ib = SimpleNamespace(sleep=lambda _: None)
    live, checkable = client._wait_leg_live(_FakeTrade("Submitted"), timeout=1.0)

    assert live is True and checkable is True


# ── 4. Pending entries are not rejections ────────────────────────────────────

def _armed_engine(tmp_store, mock_broker, pending: bool):
    """A real StrategyEngine armed on TQQQ, with a manager whose open_position
    returns None — the shape a live IB entry always has, because the fill lands
    seconds later on the async path."""
    from orb_live.runner.strategy_engine import StrategyEngine, SymbolState
    from orb_live.signals.pre_market import Phase2Result
    from orb_live.config.live_config import load_live_config

    calls = []
    mgr = SimpleNamespace(
        open_position=lambda **kw: calls.append(kw) or None,
        has_pending_entry=lambda sym: pending,
    )
    cfg    = load_live_config()
    engine = StrategyEngine(mgr, cfg, tmp_store, mock_broker)
    engine.new_session(TRADE_DATE)

    engine._states[SYMBOL] = SymbolState.ORB_COMPLETE
    engine._p2[SYMBOL] = Phase2Result(
        symbol=SYMBOL, gap_abs=0.02, gap_direction=1, prior_close=100.0,
        first_open=100.0,
        orb={"high": 101.0, "low": 99.0, "midpoint": 100.0,
             "size_pct": 0.02, "n_bars": 6, "dollar_per_min": 1e6},
        tp1_mult=2.0, tp2_mult=0.0, rtg_val=None, rtg_pct=None,
        rtg_excluded=False, routing_action="normal", size_mult=1.0,
        preflight=None, is_candidate=True,
    )
    return engine, calls


def test_pending_entry_is_not_reported_as_rejected(tmp_store, mock_broker):
    """open_position() returns None for a working-but-unfilled IB order as well
    as for a rejection. Treating both as rejection left the engine's state
    machine wrong about every live position it opened (2026-09-18, 4/4)."""
    from datetime import datetime
    from orb_live.runner.strategy_engine import SymbolState

    engine, calls = _armed_engine(tmp_store, mock_broker, pending=True)
    ts  = datetime(2026, 1, 5, 10, 5, tzinfo=None)
    bar = {"timestamp": ts, "open": 100.5, "high": 101.5,
           "low": 100.4, "close": 101.4, "volume": 50_000}

    engine.on_bar(SYMBOL, bar, ts)

    assert calls, "the breakout must actually have attempted an entry"
    assert engine.get_state(SYMBOL) == SymbolState.IN_POSITION, \
        "a working entry order is not a rejection"


def test_genuine_rejection_still_marks_the_symbol_skipped(tmp_store, mock_broker):
    """The real rejection path must keep working — otherwise a symbol whose
    entry IB refused would be left looking live."""
    from datetime import datetime
    from orb_live.runner.strategy_engine import SymbolState

    engine, calls = _armed_engine(tmp_store, mock_broker, pending=False)
    ts  = datetime(2026, 1, 5, 10, 5, tzinfo=None)
    bar = {"timestamp": ts, "open": 100.5, "high": 101.5,
           "low": 100.4, "close": 101.4, "volume": 50_000}

    engine.on_bar(SYMBOL, bar, ts)

    assert calls
    assert engine.get_state(SYMBOL) == SymbolState.EXITED_OR_SKIPPED


def test_has_pending_entry_tracks_the_manager(mock_broker, tmp_store):
    mgr = _build_mgr(mock_broker, tmp_store)
    assert mgr.has_pending_entry(SYMBOL) is False
    mgr._pending_entries[SYMBOL] = {"order_id": "1"}
    assert mgr.has_pending_entry(SYMBOL) is True


# ── 5. /status reports the live session ──────────────────────────────────────

def _runner_with_phase(phase, watched, positions, killed=False):
    from orb_live.runner.session_runner import SessionRunner
    r = SessionRunner.__new__(SessionRunner)
    r._session_date = TRADE_DATE
    r._phase        = phase
    r._watched      = watched
    r._broker       = SimpleNamespace(get_positions=lambda: positions)
    r._gate         = SimpleNamespace(is_session_killed=lambda: killed)
    return r


def test_health_snapshot_reports_phase_and_unrealized():
    """/status reported session_state 'idle' and unrealized 0.0 for an entire
    session holding two positions, because main.py never passed a state fn."""
    r = _runner_with_phase(
        "trade_active", ["SBIT", "XRPT"],
        [{"unrealized_pl": 74.07}, {"unrealized_pl": 90.05}],
    )
    snap = r.health_snapshot()

    assert snap["session_state"] == "trade_active"
    assert snap["active_candidates"] == 2
    assert snap["todays_unrealized_pnl"] == pytest.approx(164.12)
    assert snap["session_kill_active"] is False


def test_health_snapshot_surfaces_the_kill_switch():
    r = _runner_with_phase("trade_active", [], [], killed=True)
    assert r.health_snapshot()["session_kill_active"] is True


def test_health_snapshot_survives_a_broken_broker():
    """The health endpoint is what an operator reads when things are already
    wrong; it must degrade rather than raise."""
    from orb_live.runner.session_runner import SessionRunner

    def _boom():
        raise ConnectionError("gateway down")

    r = SessionRunner.__new__(SessionRunner)
    r._session_date = TRADE_DATE
    r._phase        = "trade_active"
    r._watched      = ["SBIT"]
    r._broker       = SimpleNamespace(get_positions=_boom)
    r._gate         = SimpleNamespace(is_session_killed=_boom)

    snap = r.health_snapshot()
    assert snap["session_state"] == "trade_active"
    assert snap["todays_unrealized_pnl"] == 0.0
    assert snap["session_kill_active"] is False


def test_main_wires_the_health_snapshot():
    """Regression for the actual defect: HealthServer was constructed without
    session_state_fn, so every field it provides silently used its default."""
    import inspect
    from orb_live.runner import main as main_mod

    src = inspect.getsource(main_mod._build_components)
    assert "session_state_fn=runner.health_snapshot" in src, \
        "main.py must pass the runner's snapshot into HealthServer"
