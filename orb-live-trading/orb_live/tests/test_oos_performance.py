"""The out-of-sample performance record must not flatter itself.

This tool feeds a public page, so every defect in it is a published wrong
number. The live data has four traps in it already, and each one produces a
plausible-looking result rather than an error:

1. The paper account was refunded mid-record -- equity jumps 10,632 -> 30,001
   between 2026-08-18 and 2026-08-19. Any return computed from equity
   differences reads that as a +182% day.
2. Two sessions ran against a dead socket and recorded start_equity = 0.
   Counting those as flat days dilutes every average.
3. `closed_trades.commission` is never populated, so trade-level P&L is GROSS.
   The broker's own account delta is lower -- about $20/day recently. Summing
   trades overstates the record.
4. The configuration changed twice inside the sample. Pooling epochs reports a
   number that belongs to no strategy that was ever run.

Each test below pins one of those.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from orb_live.ops import oos_performance as oos


# ── fixtures ──────────────────────────────────────────────────────────────────

def _db(tmp_path: Path, equity, trades=()) -> Path:
    p = tmp_path / "live.db"
    c = sqlite3.connect(p)
    c.execute("""create table equity_curve(trade_date text, start_equity real,
                 end_equity real, session_pnl real, recorded_at text)""")
    c.execute("""create table closed_trades(id integer primary key, trade_date text,
                 symbol text, direction int, entry_price real, exit_price real,
                 qty real, pnl_pct real, dollar_pnl real, exit_reason text,
                 opened_at text, closed_at text, realized_exit_price real,
                 commission real)""")
    for d, s, pnl in equity:
        c.execute("insert into equity_curve values(?,?,?,?,?)",
                  (d, s, s + pnl, pnl, ""))
    for i, (d, sym, pnl, why) in enumerate(trades):
        c.execute("""insert into closed_trades(id,trade_date,symbol,direction,
                     entry_price,exit_price,qty,pnl_pct,dollar_pnl,exit_reason,
                     commission) values(?,?,?,1,10,11,1,0.1,?,?,0)""",
                  (i, d, sym, pnl, why))
    c.commit()
    c.close()
    return p


def _sessions(tmp_path: Path, mapping: dict[str, dict]) -> Path:
    d = tmp_path / "sessions"
    d.mkdir()
    for date, profile in mapping.items():
        (d / f"{date}.json").write_text(
            json.dumps({"trade_date": date, "profile": profile}), encoding="utf-8")
    return d


# ── the capital injection ─────────────────────────────────────────────────────

def test_refund_is_not_a_return(tmp_path):
    """Equity 10,632 -> 30,001 overnight is a deposit, not a +182% day."""
    db = _db(tmp_path, [("2026-08-18", 10_696.23, -63.59),
                        ("2026-08-19", 30_001.04, 386.81)])
    rows = oos.daily_rows(db, {})
    r = {x["date"]: x for x in rows}
    assert r["2026-08-19"]["ret"] == pytest.approx(386.81 / 30_001.04)
    assert r["2026-08-19"]["ret"] < 0.05


def test_return_is_pnl_over_that_days_starting_equity(tmp_path):
    db = _db(tmp_path, [("2026-09-01", 31_565.27, 37.92)])
    assert oos.daily_rows(db, {})[0]["ret"] == pytest.approx(37.92 / 31_565.27)


def test_cumulative_chains_returns_rather_than_differencing_equity(tmp_path):
    """Chaining is what makes the record survive a deposit."""
    db = _db(tmp_path, [("2026-08-18", 10_000.0, 100.0),
                        ("2026-08-19", 30_000.0, 300.0)])
    s = oos.summarize(oos.daily_rows(db, {}))
    assert s["cum_return"] == pytest.approx(1.01 * 1.01 - 1)


# ── broken sessions ───────────────────────────────────────────────────────────

def test_zero_equity_sessions_are_excluded_not_counted_flat(tmp_path):
    db = _db(tmp_path, [("2026-08-11", 0.0, 0.0),
                        ("2026-08-13", 10_706.20, 11.85)])
    rows = oos.daily_rows(db, {})
    assert [r["date"] for r in rows] == ["2026-08-13"]


def test_excluded_sessions_are_reported_not_silently_dropped(tmp_path):
    """A dropped day must stay visible or the record looks cleaner than it is."""
    db = _db(tmp_path, [("2026-08-11", 0.0, 0.0), ("2026-08-13", 10_706.20, 11.85)])
    ex = oos.excluded(db, {})
    assert any(e["date"] == "2026-08-11" and e["reason"] == "zero_equity"
               for e in ex)


def test_untracked_equity_day_with_trades_is_excluded(tmp_path):
    """July: equity_curve wrote 0.00 while trades were closing. Not a flat day."""
    db = _db(tmp_path,
             [("2026-07-20", 10_000.0, 0.0)],
             [("2026-07-20", "SOXL", 80.24, "EOD")])
    rows = oos.daily_rows(db, {})
    assert rows == []
    assert oos.excluded(db, {})[0]["reason"] == "untracked_equity"


def test_genuine_flat_day_is_kept(tmp_path):
    """No trades and no P&L is a real observation of zero and must count."""
    db = _db(tmp_path, [("2026-08-27", 31_180.30, 0.0)])
    rows = oos.daily_rows(db, {})
    assert len(rows) == 1
    assert rows[0]["ret"] == 0.0


# ── P&L source ────────────────────────────────────────────────────────────────

def test_pnl_comes_from_the_broker_not_from_summing_trades(tmp_path):
    """closed_trades is gross -- commission is never populated. The account
    delta is the only cost-inclusive number available."""
    db = _db(tmp_path,
             [("2026-08-20", 30_387.86, 1_522.75)],
             [("2026-08-20", "XRPT", 759.92, "TP1"),
              ("2026-08-20", "ETHD", 595.32, "TP1"),
              ("2026-08-20", "SBIT", 188.37, "EOD")])
    rows = oos.daily_rows(db, {})
    assert rows[0]["pnl"] == pytest.approx(1_522.75)
    assert rows[0]["pnl"] < sum([759.92, 595.32, 188.37])


# ── config epochs ─────────────────────────────────────────────────────────────

def test_epoch_is_derived_from_the_profile_not_a_hardcoded_date(tmp_path):
    sess = _sessions(tmp_path, {
        "2026-08-28": {"stop": 0.75, "w": 1},
        "2026-08-31": {"stop": 1.00, "w": 2},
    })
    ep = oos.profile_epochs(sess)
    assert ep["2026-08-28"] != ep["2026-08-31"]


def test_identical_profiles_share_an_epoch(tmp_path):
    sess = _sessions(tmp_path, {"2026-08-24": {"a": 1}, "2026-08-25": {"a": 1}})
    ep = oos.profile_epochs(sess)
    assert ep["2026-08-24"] == ep["2026-08-25"]


def test_days_without_a_session_file_get_the_unknown_epoch(tmp_path):
    db = _db(tmp_path, [("2026-08-06", 10_777.65, -52.63)])
    rows = oos.daily_rows(db, {})
    assert rows[0]["epoch"] == oos.EPOCH_UNKNOWN


def test_summary_is_segmented_by_epoch(tmp_path):
    db = _db(tmp_path, [("2026-08-28", 31_182.14, 375.78),
                        ("2026-09-01", 31_565.27, 37.92)])
    ep = {"2026-08-28": "aaaa", "2026-09-01": "bbbb"}
    out = oos.build_report(db, ep)
    assert set(out["epochs"]) == {"aaaa", "bbbb"}
    assert out["epochs"]["bbbb"]["n_days"] == 1


# ── honesty about n ───────────────────────────────────────────────────────────

def test_no_annualised_sharpe_on_a_tiny_sample(tmp_path):
    """An annualised figure on 2 days is not a statistic, it is decoration."""
    db = _db(tmp_path, [("2026-08-31", 31_559.75, 0.0),
                        ("2026-09-01", 31_565.27, 37.92)])
    s = oos.summarize(oos.daily_rows(db, {}))
    assert s["sharpe"] is None
    assert s["n_days"] == 2


def test_sharpe_appears_once_the_sample_justifies_it(tmp_path):
    eq = [(f"2026-0{6 + i // 28}-{i % 28 + 1:02d}", 30_000.0, 10.0 * (-1) ** i)
          for i in range(oos.MIN_DAYS_FOR_SHARPE + 5)]
    db = _db(tmp_path, eq)
    s = oos.summarize(oos.daily_rows(db, {}))
    assert s["sharpe"] is not None


def test_report_states_how_far_from_decisive(tmp_path):
    db = _db(tmp_path, [("2026-09-01", 31_565.27, 37.92)])
    out = oos.build_report(db, {})
    assert out["power"]["days_required"] == oos.DAYS_TO_DECISIVE
    assert out["power"]["days_remaining"] == oos.DAYS_TO_DECISIVE - 1


# ── two ways the first version misreported ────────────────────────────────────

def test_flat_no_trade_days_do_not_count_as_losing_days(tmp_path):
    """A day the strategy found no setup is not a day it lost.

    The first version reported 16.1% winning days across a record whose
    sessions were mostly flat, which reads as a catastrophe and is simply the
    wrong denominator.
    """
    db = _db(tmp_path,
             [("2026-08-25", 30_000.0, 0.0),
              ("2026-08-26", 30_000.0, 0.0),
              ("2026-08-27", 30_000.0, 0.0),
              ("2026-08-28", 30_000.0, 300.0)],
             [("2026-08-28", "SOLT", 378.0, "TP1")])
    s = oos.summarize(oos.daily_rows(db, {}))
    assert s["n_flat_days"] == 3
    assert s["n_active_days"] == 1
    assert s["win_days_active"] == pytest.approx(1.0)
    assert s["win_days"] == pytest.approx(0.25)


def test_pooled_sharpe_is_withheld_when_the_config_changed(tmp_path):
    """A Sharpe pooled across configurations belongs to no strategy that ran."""
    eq = [(f"2026-08-{i + 1:02d}", 30_000.0, 10.0 * (-1) ** i)
          for i in range(oos.MIN_DAYS_FOR_SHARPE + 4)]
    db = _db(tmp_path, eq)
    ep = {d: ("aaaa" if i < 10 else "bbbb") for i, (d, _, _) in enumerate(eq)}
    out = oos.build_report(db, ep)
    assert out["pooled"]["sharpe"] is None
    assert out["pooled"]["sharpe_withheld"] == "multiple_epochs"


def test_single_epoch_pooled_sharpe_is_allowed(tmp_path):
    eq = [(f"2026-08-{i + 1:02d}", 30_000.0, 10.0 * (-1) ** i)
          for i in range(oos.MIN_DAYS_FOR_SHARPE + 4)]
    db = _db(tmp_path, eq)
    ep = {d: "aaaa" for d, _, _ in eq}
    out = oos.build_report(db, ep)
    assert out["pooled"]["sharpe"] is not None
