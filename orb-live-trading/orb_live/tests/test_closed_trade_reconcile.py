"""EOD closed-trade reconciliation against the broker's executions.

On 2026-09-21 the daily report said +$134.55 while the account made +$101.98.
The flatten booked six exits from a quote snapshot before their fills existed
(+$17.56 too high), and no path recorded IB's ~$1/order commission ($15.01).
reconcile_closed_trades re-prices every row from the actual fills.
"""

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from orb_live.execution.position_manager import _round_trips
from orb_live.tests.test_position_manager import _build_mgr

DAY = date(2026, 9, 21)
_T0 = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)


def _fills(*rows):
    """(symbol, side, qty, price, commission) → time-ordered fill dicts."""
    return [
        {"symbol": s, "side": side, "qty": float(q), "price": p,
         "commission": c, "commission_reported": True,
         "order_id": str(i), "exec_id": f"e{i}", "time": _T0 + timedelta(seconds=i)}
        for i, (s, side, q, p, c) in enumerate(rows)
    ]


# The 18 real executions from 2026-09-21, with IB's reported commissions.
SEPT_21_FILLS = _fills(
    ("SOLT", "buy",   83, 76.38, 1.000249),
    ("AMDL", "buy",   20, 78.75, 1.00006),
    ("TSLL", "buy",  154, 10.35, 1.000462),
    ("XRPT", "buy",  164, 38.75, 1.000492),
    ("SBIT", "sell", 101, 28.89, 1.080107),
    ("SBIT", "sell", 100, 28.90, 0.084334),
    ("SBIT", "sell",  19, 28.87, 0.110062),
    ("KORU", "buy",   72, 22.03, 1.000216),
    ("ETHU", "buy",  200, 31.96, 1.0006),
    ("SOLT", "sell",  83, 74.13, 1.143181),
    ("TSLL", "sell", 100, 10.19, 1.040791),
    ("TSLL", "sell",  54, 10.19, 0.022027),
    ("ETHU", "sell", 100, 32.37, 1.086482),
    ("ETHU", "sell", 100, 32.37, 0.086482),
    ("XRPT", "sell", 164, 39.00, 1.16423),
    ("SBIT", "buy",  220, 28.29, 1.10066),
    ("KORU", "sell",  72, 22.56, 1.047717),
    ("AMDL", "sell",  20, 80.48, 1.037118),
)

# What closed_trades held after the session: quote-priced EOD exits, no commission.
SEPT_21_BOOKED = [
    ("SBIT", -1, 28.892818, 28.28, 220, 134.82, "EOD"),
    ("ETHU",  1, 31.96,     32.35, 200,  78.00, "EOD"),
    ("XRPT",  1, 38.75,     39.07, 164,  52.48, "EOD"),
    ("KORU",  1, 22.03,     22.61,  72,  41.76, "EOD"),
    ("AMDL",  1, 78.75,     80.54,  20,  35.80, "EOD"),
    ("TSLL",  1, 10.35,     10.21, 154, -21.56, "EOD"),
    ("SOLT",  1, 76.38,     74.13,  83, -186.75, "STOP"),
]


def _seed(store, booked):
    for sym, d, entry, exit_, qty, pnl, reason in booked:
        store.save_closed_trade(
            trade_date=DAY, symbol=sym, direction=d,
            entry_price=entry, exit_price=exit_, realized_exit_price=exit_,
            qty=float(qty), dollar_pnl=pnl, pnl_pct=0.0,
            exit_reason=reason, opened_at=_T0,
        )


def _mgr_with_fills(mock_broker, tmp_store, fills):
    mock_broker.get_executions = lambda trade_date: fills
    return _build_mgr(mock_broker, tmp_store)


# ── Round-trip pairing ──────────────────────────────────────────────────────

def test_long_round_trip_is_priced_from_fills():
    (t,) = _round_trips(_fills(("X", "buy", 100, 10.0, 1.0),
                               ("X", "sell", 100, 11.0, 1.0)))
    assert t["direction"] == 1
    assert t["gross_pnl"] == pytest.approx(100.0)
    assert t["commission"] == pytest.approx(2.0)
    assert t["net_pnl"] == pytest.approx(98.0)


def test_short_with_split_entry_fills_uses_weighted_average():
    """SBIT's short filled in three pieces at three prices."""
    fills = [f for f in SEPT_21_FILLS if f["symbol"] == "SBIT"]
    (t,) = _round_trips(fills)
    assert t["direction"] == -1
    assert t["entry_avg"] == pytest.approx(28.892818, abs=1e-6)
    assert t["exit_avg"] == pytest.approx(28.29)
    assert t["net_pnl"] == pytest.approx(130.244837, abs=1e-4)  # IB's realizedPNL


def test_reentry_after_flatten_is_a_second_trip():
    """The 2026-09-18 restart: flatten, then re-enter the same symbol."""
    trips = _round_trips(_fills(
        ("X", "sell", 188, 33.30, 1.0), ("X", "buy", 188, 33.01, 1.0),
        ("X", "sell", 190, 33.00, 1.0), ("X", "buy", 190, 32.63, 1.0),
    ))
    assert len(trips) == 2
    assert trips[0]["gross_pnl"] == pytest.approx((33.30 - 33.01) * 188)
    assert trips[1]["gross_pnl"] == pytest.approx((33.00 - 32.63) * 190)


def test_still_open_trip_is_not_reported():
    assert _round_trips(_fills(("X", "buy", 100, 10.0, 1.0),
                               ("X", "sell", 40, 11.0, 1.0))) == []


def test_missing_commission_report_is_flagged():
    fills = _fills(("X", "buy", 100, 10.0, 0.0), ("X", "sell", 100, 11.0, 1.0))
    fills[0]["commission_reported"] = False
    (t,) = _round_trips(fills)
    assert t["commission_complete"] is False


# ── Reconciliation against the store ───────────────────────────────────────

def test_sept_21_report_matches_the_account(mock_broker, tmp_store):
    """The actual incident: after reconciling, the trades sum to the $101.98
    the account made, not the $134.55 the report claimed."""
    _seed(tmp_store, SEPT_21_BOOKED)
    mgr = _mgr_with_fills(mock_broker, tmp_store, SEPT_21_FILLS)

    results = mgr.reconcile_closed_trades(DAY)

    assert set(results.values()) == {"ok"}
    rows = {r["symbol"]: r for r in tmp_store.get_closed_trades(DAY)}
    assert sum(r["dollar_pnl"] for r in rows.values()) == pytest.approx(101.98, abs=0.01)
    assert sum(r["commission"] for r in rows.values()) == pytest.approx(15.01, abs=0.01)
    assert rows["XRPT"]["realized_exit_price"] == pytest.approx(39.00)
    assert rows["XRPT"]["dollar_pnl"] == pytest.approx(38.835278, abs=1e-4)


def test_reference_exit_price_is_preserved(mock_broker, tmp_store):
    """exit_price is the target/reference field; only the realized price moves,
    so slippage stays measurable."""
    _seed(tmp_store, SEPT_21_BOOKED)
    mgr = _mgr_with_fills(mock_broker, tmp_store, SEPT_21_FILLS)
    mgr.reconcile_closed_trades(DAY)

    xrpt = next(r for r in tmp_store.get_closed_trades(DAY) if r["symbol"] == "XRPT")
    assert xrpt["exit_price"] == pytest.approx(39.07)
    assert xrpt["realized_exit_price"] == pytest.approx(39.00)


def test_row_count_mismatch_leaves_the_symbol_untouched(mock_broker, tmp_store):
    """Two rows but one trip: pairing is ambiguous, so do not guess."""
    _seed(tmp_store, [("X", 1, 10.0, 11.0, 100, 100.0, "EOD"),
                      ("X", 1, 10.0, 11.0, 100, 100.0, "EOD")])
    mgr = _mgr_with_fills(mock_broker, tmp_store, _fills(
        ("X", "buy", 100, 10.0, 1.0), ("X", "sell", 100, 10.5, 1.0)))

    assert mgr.reconcile_closed_trades(DAY) == {"X": "mismatch"}
    assert all(r["dollar_pnl"] == 100.0 for r in tmp_store.get_closed_trades(DAY))


def test_direction_mismatch_leaves_the_symbol_untouched(mock_broker, tmp_store):
    _seed(tmp_store, [("X", 1, 10.0, 11.0, 100, 100.0, "EOD")])
    mgr = _mgr_with_fills(mock_broker, tmp_store, _fills(
        ("X", "sell", 100, 10.0, 1.0), ("X", "buy", 100, 9.0, 1.0)))

    assert mgr.reconcile_closed_trades(DAY) == {"X": "mismatch"}
    assert tmp_store.get_closed_trades(DAY)[0]["dollar_pnl"] == 100.0


def test_symbol_without_fills_is_left_alone(mock_broker, tmp_store):
    _seed(tmp_store, [("X", 1, 10.0, 11.0, 100, 100.0, "EOD")])
    mgr = _mgr_with_fills(mock_broker, tmp_store, [])

    assert mgr.reconcile_closed_trades(DAY) == {"X": "no_fills"}
    assert tmp_store.get_closed_trades(DAY)[0]["dollar_pnl"] == 100.0


def test_broker_without_executions_is_a_no_op(mock_broker, tmp_store):
    _seed(tmp_store, SEPT_21_BOOKED)
    mgr = _build_mgr(mock_broker, tmp_store)
    mgr._broker = SimpleNamespace()          # no get_executions at all
    assert mgr.reconcile_closed_trades(DAY) == {}
    assert sum(r["dollar_pnl"] for r in tmp_store.get_closed_trades(DAY)) \
        == pytest.approx(134.55)


def test_fills_without_a_side_are_never_used_to_price(mock_broker, tmp_store):
    """The conftest MockBroker's executions carry no side. Pairing them would
    invent a direction, so the rows must stay as booked."""
    _seed(tmp_store, [("X", 1, 10.0, 11.0, 100, 100.0, "EOD")])
    mgr = _mgr_with_fills(mock_broker, tmp_store, [
        {"symbol": "X", "qty": 100.0, "price": 10.0, "commission": 0.0},
        {"symbol": "X", "qty": 100.0, "price": 10.5, "commission": 0.0},
    ])

    assert mgr.reconcile_closed_trades(DAY) == {"X": "no_fills"}
    assert tmp_store.get_closed_trades(DAY)[0]["dollar_pnl"] == 100.0


def test_executions_failure_does_not_raise(mock_broker, tmp_store):
    _seed(tmp_store, SEPT_21_BOOKED)
    def _boom(trade_date):
        raise ConnectionError("gateway down")
    mock_broker.get_executions = _boom
    mgr = _build_mgr(mock_broker, tmp_store)

    assert mgr.reconcile_closed_trades(DAY) == {}
    assert sum(r["dollar_pnl"] for r in tmp_store.get_closed_trades(DAY)) \
        == pytest.approx(134.55)


# ── IBClient.get_executions ─────────────────────────────────────────────────

def _ib_fill(symbol, side, shares, price, commission, when):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol),
        execution=SimpleNamespace(side=side, shares=shares, price=price,
                                  orderId=1, execId="x", time=when),
        commissionReport=SimpleNamespace(commission=commission),
    )


def test_get_executions_filters_by_et_date_and_scrubs_the_sentinel():
    from orb_live.data.ib_client import IBClient

    client = IBClient.__new__(IBClient)
    today     = datetime(2026, 9, 21, 19, 59, tzinfo=timezone.utc)
    yesterday = datetime(2026, 9, 20, 19, 59, tzinfo=timezone.utc)
    client._ib = SimpleNamespace(
        reqExecutions=lambda f: None,
        sleep=lambda s: None,
        fills=lambda: [
            _ib_fill("A", "BOT", 10, 5.0, 1.0, today),
            _ib_fill("B", "SLD", 10, 5.0, 1.7976931348623157e308, today),
            _ib_fill("C", "BOT", 10, 5.0, 1.0, yesterday),
        ],
    )
    client.is_connected = lambda: True

    out = client.get_executions(DAY)

    assert [f["symbol"] for f in out] == ["A", "B"]
    assert out[0]["side"] == "buy" and out[0]["commission"] == 1.0
    assert out[1]["side"] == "sell"
    assert out[1]["commission"] == 0.0
    assert out[1]["commission_reported"] is False
