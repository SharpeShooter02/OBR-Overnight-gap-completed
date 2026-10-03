"""Out-of-sample live performance record, built from the broker's own numbers.

This feeds a public page, so its job is to be hard to flatter. The live data
carries four traps, and every one of them produces a plausible number rather
than an error:

1. THE ACCOUNT WAS REFUNDED mid-record. Equity goes 10,632 -> 30,001 between
   2026-08-18 and 2026-08-19. Any return computed by differencing equity reads
   that as a +182% day. Returns here are always `session_pnl / start_equity`
   for that same day, and the cumulative figure CHAINS those daily returns.
   Deposits then cost nothing, because each day is measured against the capital
   it actually started with.

2. TWO SESSIONS RAN AGAINST A DEAD SOCKET and recorded start_equity = 0
   (2026-08-11, 2026-08-12). Counting them as flat days would dilute every
   average with observations that never happened. They are excluded -- and
   listed, because a record that quietly drops its bad days looks better than
   it is.

3. `closed_trades.commission` WAS NEVER POPULATED before the EOD execution
   reconcile (reconcile_closed_trades, added 2026-09-21), so trade-level P&L
   on earlier rows is gross — and EOD exits were priced from a quote rather
   than the fill. The broker's account delta is lower by roughly $20/day on
   those days. This module therefore takes P&L from `equity_curve`, never from
   summing trades. Trades are used only for counts, exit reasons and
   per-symbol detail.

   The corollary is that July is unusable: `equity_curve` wrote 0.00 while
   trades were closing, so those days are excluded as `untracked_equity` rather
   than reported as flat.

4. THE CONFIGURATION CHANGED TWICE inside the sample. Pooling the epochs
   reports a number belonging to no strategy that was ever run. The epoch key
   is a hash of the profile block recorded in each session file, so it is
   derived from what actually ran rather than from a date someone remembered
   to update.

WHAT IT DELIBERATELY WILL NOT DO. No annualised Sharpe below MIN_DAYS_FOR_SHARPE
sessions. An annualised figure on a handful of days is decoration, and the
temptation to quote it is exactly why this is a constant rather than a judgment
call at the call site.

    python -m orb_live.ops.oos_performance
    python -m orb_live.ops.oos_performance --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DB = Path("orb_live/state/live.db")
SESSIONS = Path("data/sessions")
REPORT = Path("data/reports/oos_performance.json")

#: Mirrored into the analysis repo so the report renderer can template it.
MIRROR = Path(
    str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'BacktestingGaps/analysis/out/live_performance.json'))

EPOCH_UNKNOWN = "pre-record"

#: Below this, an annualised Sharpe is not a statistic. 21 sessions is one
#: trading month and still far too few -- it is a floor on absurdity, not a
#: threshold for significance.
MIN_DAYS_FOR_SHARPE = 21

#: Trading days for the live record to discriminate the strategy's edge from
#: zero at the effect size the backtest implies (register V38).
DAYS_TO_DECISIVE = 154

TRADING_DAYS_PER_YEAR = 252
RF_ANNUAL = 0.043


# ── inputs ────────────────────────────────────────────────────────────────────

def profile_epochs(sessions_dir: Path = SESSIONS) -> dict[str, str]:
    """{trade_date: epoch} from the profile block each session recorded.

    Hashing what ran beats hardcoding a date: a config change nobody remembered
    to write down still splits the record correctly.
    """
    out: dict[str, str] = {}
    for p in sorted(Path(sessions_dir).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        prof = d.get("profile")
        date = d.get("trade_date") or p.stem
        if prof is None:
            continue
        out[date] = hashlib.sha1(
            json.dumps(prof, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return out


def _rows(db: Path):
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    try:
        eq = list(c.execute("select * from equity_curve order by trade_date"))
        tr = list(c.execute("select * from closed_trades order by trade_date, id"))
    finally:
        c.close()
    return eq, tr


def _trades_by_date(trades) -> dict[str, list]:
    out: dict[str, list] = {}
    for t in trades:
        out.setdefault(t["trade_date"], []).append(t)
    return out


def _classify(row, day_trades) -> str | None:
    """Reason this session cannot be used, or None if it counts."""
    start = row["start_equity"] or 0.0
    if start <= 0:
        return "zero_equity"
    # equity_curve wrote 0.00 through July while trades were closing; a day with
    # fills but no account movement is a tracking failure, not a flat session.
    if (row["session_pnl"] or 0.0) == 0.0 and day_trades:
        return "untracked_equity"
    return None


# ── series ────────────────────────────────────────────────────────────────────

def daily_rows(db: Path = DB, epochs: dict[str, str] | None = None) -> list[dict]:
    """Usable sessions, one row each, as returns on that day's own capital."""
    epochs = epochs or {}
    eq, tr = _rows(db)
    byday = _trades_by_date(tr)
    out = []
    for r in eq:
        d = r["trade_date"]
        if _classify(r, byday.get(d, [])) is not None:
            continue
        start, pnl = float(r["start_equity"]), float(r["session_pnl"] or 0.0)
        ts = byday.get(d, [])
        out.append({
            "date": d,
            "epoch": epochs.get(d, EPOCH_UNKNOWN),
            "start_equity": start,
            "pnl": pnl,
            "ret": pnl / start,
            "n_trades": len(ts),
            "exits": {k: sum(1 for t in ts if (t["exit_reason"] or "") == k)
                      for k in ("TP1", "STOP", "EOD")},
        })
    return out


def excluded(db: Path = DB, epochs: dict[str, str] | None = None) -> list[dict]:
    """Sessions left out, with the reason. Published alongside the record."""
    epochs = epochs or {}
    eq, tr = _rows(db)
    byday = _trades_by_date(tr)
    out = []
    for r in eq:
        d = r["trade_date"]
        why = _classify(r, byday.get(d, []))
        if why:
            out.append({"date": d, "reason": why,
                        "epoch": epochs.get(d, EPOCH_UNKNOWN),
                        "n_trades": len(byday.get(d, []))})
    return out


def _max_drawdown(rets: list[float]) -> float:
    eq, peak, mdd = 1.0, 1.0, 0.0
    for r in rets:
        eq *= (1 + r)
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return mdd


def summarize(rows: list[dict]) -> dict:
    """Aggregate a set of sessions. Sharpe only once n justifies one."""
    n = len(rows)
    if n == 0:
        return {"n_days": 0, "n_active_days": 0, "n_flat_days": 0,
                "n_trades": 0, "pnl": 0.0, "cum_return": 0.0,
                "mean_daily": None, "win_days": None, "win_days_active": None,
                "sharpe": None, "sharpe_withheld": "no_data",
                "max_dd": 0.0, "best": None, "worst": None, "exits": {}}

    rets = [r["ret"] for r in rows]
    cum = 1.0
    for r in rets:
        cum *= (1 + r)

    sharpe, withheld = None, None
    if n < MIN_DAYS_FOR_SHARPE:
        withheld = "n_too_small"
    elif len({r["epoch"] for r in rows}) > 1:
        # A Sharpe pooled across configurations describes no strategy that was
        # ever run. The live record spans three profiles and one of them is a
        # two-day window that returned +6.36%; pooling lets that dominate a
        # figure presented as the strategy's.
        withheld = "multiple_epochs"
    else:
        sd = statistics.stdev(rets)
        if sd > 0:
            rf = RF_ANNUAL / TRADING_DAYS_PER_YEAR
            sharpe = ((statistics.fmean(rets) - rf) / sd
                      * (TRADING_DAYS_PER_YEAR ** 0.5))
        else:
            withheld = "zero_variance"

    # A day the strategy found no setup is an observation of zero, not a loss.
    # Reporting wins over ALL sessions put the live win rate at 16.1% on a
    # record that was mostly flat -- the wrong denominator, in the unflattering
    # direction. Both are reported; the active one is the meaningful one.
    active = [r for r in rows if r["n_trades"] > 0]
    n_act = len(active)

    exits: dict[str, int] = {}
    for r in rows:
        for k, v in r["exits"].items():
            exits[k] = exits.get(k, 0) + v

    return {
        "n_days": n,
        "n_active_days": n_act,
        "n_flat_days": n - n_act,
        "n_trades": sum(r["n_trades"] for r in rows),
        "pnl": sum(r["pnl"] for r in rows),
        "cum_return": cum - 1,
        "mean_daily": statistics.fmean(rets),
        "win_days": sum(1 for r in rets if r > 0) / n,
        "win_days_active": (sum(1 for r in active if r["ret"] > 0) / n_act
                            if n_act else None),
        "sharpe": sharpe,
        "sharpe_withheld": withheld,
        "max_dd": _max_drawdown(rets),
        "best": max(rows, key=lambda r: r["ret"])["date"],
        "worst": min(rows, key=lambda r: r["ret"])["date"],
        "exits": exits,
    }


def build_report(db: Path = DB, epochs: dict[str, str] | None = None) -> dict:
    epochs = epochs or {}
    rows = daily_rows(db, epochs)
    ex = excluded(db, epochs)

    by_epoch: dict[str, list] = {}
    for r in rows:
        by_epoch.setdefault(r["epoch"], []).append(r)

    current = rows[-1]["epoch"] if rows else None
    n_cur = len(by_epoch.get(current, []))

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "broker account delta (equity_curve), cost-inclusive",
        "pooled": summarize(rows),
        "current_epoch": current,
        "epochs": {k: {**summarize(v),
                       "first": v[0]["date"], "last": v[-1]["date"]}
                   for k, v in by_epoch.items()},
        "power": {
            "days_required": DAYS_TO_DECISIVE,
            "days_observed": n_cur,
            "days_remaining": max(DAYS_TO_DECISIVE - n_cur, 0),
            "note": ("Trading days needed for the live record to distinguish "
                     "this strategy's edge from zero at the backtest's effect "
                     "size. Until then the record is descriptive, not evidence."),
        },
        "excluded": ex,
        "days": rows,
    }


# ── cli ───────────────────────────────────────────────────────────────────────

def _pct(v, dp=2):
    return "--" if v is None else f"{v * 100:.{dp}f}%"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Live out-of-sample record")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--sessions", default=str(SESSIONS))
    ap.add_argument("--json", action="store_true", help="print the JSON")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()

    epochs = profile_epochs(Path(a.sessions))
    rep = build_report(Path(a.db), epochs)

    if a.json:
        print(json.dumps(rep, indent=2))
        return

    p = rep["pooled"]
    print("=" * 78)
    print("LIVE OUT-OF-SAMPLE RECORD -- paper account, broker P&L")
    print("=" * 78)
    print(f"  sessions counted   {p['n_days']}")
    print(f"  trades             {p['n_trades']}")
    print(f"  net P&L            {p['pnl']:+,.2f}")
    print(f"  cumulative return  {_pct(p['cum_return'])}")
    print(f"  mean day           {_pct(p['mean_daily'], 3)}")
    print(f"  active sessions    {p['n_active_days']}  "
          f"({p['n_flat_days']} flat, no setup found)")
    print(f"  winning days       {_pct(p['win_days_active'], 1)} of active  "
          f"({_pct(p['win_days'], 1)} of all sessions)")
    print(f"  max drawdown       {_pct(p['max_dd'])}")
    _why = {"n_too_small": f"withheld -- n < {MIN_DAYS_FOR_SHARPE} sessions",
            "multiple_epochs": "withheld -- record spans several configurations",
            "zero_variance": "withheld -- no variance",
            "no_data": "withheld -- no data"}
    sh = (_why.get(p["sharpe_withheld"], "withheld") if p["sharpe"] is None
          else f"{p['sharpe']:.2f}")
    print(f"  Sharpe             {sh}")

    print("\n  BY CONFIGURATION EPOCH")
    print(f"  {'epoch':<12}{'from':<12}{'to':<12}{'days':>6}{'trades':>8}"
          f"{'P&L':>11}{'return':>10}")
    for k, v in rep["epochs"].items():
        cur = " *" if k == rep["current_epoch"] else ""
        print(f"  {k:<12}{v['first']:<12}{v['last']:<12}{v['n_days']:>6}"
              f"{v['n_trades']:>8}{v['pnl']:>+11,.2f}{_pct(v['cum_return']):>10}{cur}")
    print("  * = configuration currently running")

    if rep["excluded"]:
        print(f"\n  EXCLUDED SESSIONS ({len(rep['excluded'])})")
        for e in rep["excluded"]:
            print(f"    {e['date']}  {e['reason']:<18} trades={e['n_trades']}")

    pw = rep["power"]
    print(f"\n  {pw['days_observed']} of {pw['days_required']} sessions on the "
          f"current configuration; {pw['days_remaining']} to go before the live "
          f"record can discriminate.")

    if not a.no_write:
        for dest in (REPORT, MIRROR):
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(rep, indent=2), encoding="utf-8")
                print(f"\nwrote {dest}")
            except OSError as exc:
                print(f"\ncould not write {dest}: {exc}")


if __name__ == "__main__":
    main()
