"""Step 4 (V48): the frozen configuration, run fresh from the engine, no caches.

  prior close     official IBKR auction close (V42)
  PS filter       k = 0.5 on point-in-time expanding sigma; crypto on UTC daily bars
  universe        full, no universe-level ADV gate
  execution cost  per-trade participation slippage + commission
  liquidity       cost/reward <= 0.15 at 10:00
  sizing          LOCKED_WEIGHTS by class x SPY-gap regime, time-ordered margin
                  allocator, 10-unit budget, per-trade need <= 5 units

Slippage is re-charged per trade at the weighted order size: the engine charges
the calibration notional (live median order); a weight-w trade pays
model(w x notional) - model(notional) more on entry, and on the stop when stopped.

    python analysis/freeze_run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'orb-live-trading'))

OUT = Path("analysis/out")
INTRADAY = Path("orb_event_study/cache/intraday")
HOLDOUT_START = pd.Timestamp("2026-05-21")
MAX_NEED_FRAC = 0.5
REGIMES = ("quiet", "active", "flood")
START = 10_000.0


def main() -> None:
    import analysis.authoritative_run as A
    import scripts.etf_basis as EB
    from analysis.liquidity_rule import week_parity
    from analysis.refit_sleeve_weights_spy import spy_regimes
    from scripts import as_traded as AT
    from scripts.optimize_v1_3class import run_v1_at_k, stats, trading_days
    from scripts.skip_cheap_by_class import add_force_includes_to_master, allotment_group
    from scripts.etf_basis import load_margin_rates, rate_for
    from scripts.run_allocation_policies import run_sequential, EQUITY_UNITS, BASE_NOTIONAL, _lev, classify

    AT.USE_AS_TRADED = True
    AT._cache.clear()
    EB.SAMPLE_END = None
    master = add_force_includes_to_master(pd.read_csv("master_universe.csv"))
    rates = load_margin_rates(master)
    locked = A.locked_args()
    W = A.LOCKED_WEIGHTS
    sm = locked["slippage_model"]
    now = pd.Timestamp.now(tz="America/New_York")
    today = now.tz_localize(None).normalize()

    print(f"FROZEN CONFIG  k={A.LOCKED_K_SIGMA} sigma={A.LOCKED_SIGMA_MODE}  c/r<={locked['max_cost_to_reward']}  "
          f"weights {W}", flush=True)
    print("running engine ...", flush=True)
    t = run_v1_at_k(A.LOCKED_K_SIGMA, master, entry_slippage_bps=0.0, stop_slippage_bps=0.0,
                    eod_slippage_bps=0.0, **locked)
    t["date"] = pd.to_datetime(t["date"]).dt.tz_localize(None).dt.normalize()
    t = t[t["date"] < today]
    t.to_parquet(OUT / "freeze_trades.parquet")
    print("building candidates ...", flush=True)
    detail = A.locked_candidates(master)

    # ---- opening-range $/min, to re-price slippage at the weighted size -------
    mc: dict = {}

    def dpm(sym, d):
        key = (sym, d.strftime("%Y-%m"))
        if key not in mc:
            p = INTRADAY / sym / f"{key[1]}.parquet"
            if not p.exists():
                mc[key] = pd.Series(dtype=float)
            else:
                x = pd.read_parquet(p)
                ts = pd.to_datetime(x["timestamp"])
                ts = ts.dt.tz_localize("America/New_York") if ts.dt.tz is None else ts.dt.tz_convert("America/New_York")
                m = ts.dt.hour * 60 + ts.dt.minute
                sel = (m >= 570) & (m < 600)
                mc[key] = (x.loc[sel, "close"] * x.loc[sel, "volume"]).groupby(
                    ts[sel].dt.tz_localize(None).dt.normalize()).median()
        return float(mc[key].get(d, np.nan))

    def slip(notional, v):
        pmax = float(sm["part_max"])
        part = pmax if not (v and v > 0 and math.isfinite(v)) else min(notional / v, pmax)
        r = math.sqrt(part)
        return (max(0.0, sm["entry"]["a"] + sm["entry"]["b"] * r),
                max(0.0, sm["stop"]["a"] + sm["stop"]["b"] * r))

    t["dpm"] = [dpm(s, d) for s, d in zip(t["symbol"], t["date"])]
    f = t.set_index(["date", "symbol"])
    f = f[~f.index.duplicated(keep="first")]
    lo, hi = t["date"].min(), t["date"].max()
    reg = spy_regimes()
    n0 = sm["default_notional"]

    days = {}
    for d, cands in detail.items():
        dd = pd.Timestamp(d)
        dd = dd.tz_localize(None).normalize() if dd.tz else dd.normalize()
        if dd < lo or dd > hi or not cands:
            continue
        rg = reg.get(dd)
        if not isinstance(rg, str):
            continue
        rows = []
        for c in cands:
            sym = c["symbol"]
            lev = abs(float(c.get("lev") or 0)) or _lev(sym)
            cl = classify(sym)
            w = W[cl][REGIMES.index(rg)]
            k = (dd, sym)
            fired = k in f.index
            pnl = np.nan
            if fired:
                row = f.loc[k]
                e0, s0 = slip(n0, row["dpm"])
                e1, s1 = slip(n0 * max(w, 1e-9), row["dpm"])
                extra = (e1 - e0) + ((s1 - s0) if str(row["exit_reason"]).upper().startswith("STOP") else 0.0)
                pnl = row["pnl_pct"] - extra / 1e4
            rows.append({"symbol": sym, "cls": cl, "w": w, "r": rate_for(sym, c["etf_dir"], rates, lev),
                         "gap": abs(c.get("etf_gap") or 0.0) / (lev * 0.02),
                         "entry_time": f.loc[k, "entry_time"] if fired else pd.NaT,
                         "pnl_pct": pnl, "ul": c["underlying"], "group": allotment_group(c["underlying"])})
        tab = pd.DataFrame(rows)
        tab["w"] = tab["w"] / tab.groupby("group")["ul"].transform("nunique")
        tab = tab[tab["w"] > 0]
        tab["fired"] = tab["entry_time"].notna()
        if not tab.empty:
            days[dd] = (tab, rg)

    sized = run_sequential(days, EQUITY_UNITS, "time", partial=True, max_need=MAX_NEED_FRAC * EQUITY_UNITS)
    sized = sized.assign(cls=sized["symbol"].map(classify), year=pd.to_datetime(sized["date"]).dt.year)
    sized.to_parquet(OUT / "freeze_sized.parquet")
    sessions = trading_days(lo, hi)
    missing = sorted(set(pd.to_datetime(sized["date"]).dt.normalize()) - set(sessions))
    if missing:
        raise SystemExit(
            f"CALENDAR GAP: {len(missing)} trade session(s) are missing from the SPY "
            f"calendar ({', '.join(str(d.date()) for d in missing[:6])}) and would be "
            "silently dropped from every span. Run analysis/extend_daily_calendar.py.")
    ds = sized.groupby("date")["sized_pnl"].sum().reindex(sessions, fill_value=0.0)

    def sh0(x):
        r = x.to_numpy(float); sd = r.std(ddof=1)
        return float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")

    def compounded(x):
        eq = START * np.cumprod(1 + x.to_numpy(float) / START)
        yrs = len(x) / 252
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        return eq[-1], (eq[-1] / START) ** (1 / yrs) - 1 if yrs > 0 else np.nan, dd

    ins = sessions[sessions < HOLDOUT_START]
    mid = ins[len(ins) // 2]
    par = week_parity(ins)
    spans = {"full": sessions, "in-sample": ins, "holdout": sessions[sessions >= HOLDOUT_START],
             "IS H1": ins[ins < mid], "IS H2": ins[ins >= mid], "IS odd wks": ins[par], "IS even wks": ins[~par]}
    report = {}
    print("\n" + "=" * 118)
    print(f"FROZEN RUN  {lo.date()} .. {hi.date()}   $10k base (1 unit = ${BASE_NOTIONAL:,.0f})")
    print("=" * 118)
    print(f"  {'span':<12}{'sessions':>9}{'trades':>7}{'P&L':>9}{'CAGR':>8}{'Sharpe':>8}{'Sh0':>7}{'maxDD':>9}"
          f"{'Calmar':>8}{'win':>7}   compounded: end equity / CAGR / maxDD")
    for n, ss in spans.items():
        x = ds.reindex(ss)
        s = stats(x)
        tr = sized[sized["date"].isin(ss)]
        eq, cg, cdd = compounded(x)
        report[n] = dict({k: float(v) for k, v in s.items() if isinstance(v, (int, float))},
                         sharpe0=sh0(x), trades=len(tr), comp_end=float(eq), comp_cagr=float(cg), comp_dd=float(cdd))
        print(f"  {n:<12}{len(ss):>9}{len(tr):>7}{s['net_pnl']:>9,.0f}{s['ann_ret']:>8.1%}{s['sharpe']:>8.2f}"
              f"{sh0(x):>7.2f}{s['max_dd']:>9.2%}{s['calmar']:>8.2f}{(tr['sized_pnl'] > 0).mean():>7.1%}"
              f"   ${eq:>9,.0f} / {cg:>6.1%} / {cdd:>7.2%}")

    print("\n  by class (full)")
    for c, g in sized.groupby("cls"):
        x = g.groupby("date")["sized_pnl"].sum().reindex(sessions, fill_value=0.0)
        print(f"    {c}: {len(g):>5} trades  P&L {g['sized_pnl'].sum():>8,.0f}  win {(g['sized_pnl'] > 0).mean():.1%}  "
              f"Sh0 {sh0(x):.2f}  holdout {g.loc[g['date'] >= HOLDOUT_START, 'sized_pnl'].sum():>6,.0f}")
    print("\n  by year")
    for y, g in sized.groupby("year"):
        ss = sessions[sessions.year == y]
        x = ds.reindex(ss)
        s = stats(x)
        print(f"    {y}: {len(g):>4} trades  P&L {g['sized_pnl'].sum():>8,.0f}  Sh0 {sh0(x):>5.2f}  "
              f"maxDD {s['max_dd']:>7.2%}  C1/C2/C3 " + " / ".join(
                  f"{g.loc[g['cls'] == c, 'sized_pnl'].sum():,.0f}" for c in ("C1", "C2", "C3")))
    print(f"\n  partial fills: {(sized['frac'] < 0.999).mean():.1%} of trades")
    print(f"  at $100k equity ($10k units) fixed-size P&L scales x10: full ${report['full']['net_pnl'] * 10:,.0f}; "
          f"slippage is charged at live's median order x weight, i.e. already live-sized.")
    (OUT / "freeze_run.json").write_text(json.dumps(
        {"config": {"k": A.LOCKED_K_SIGMA, "sigma": A.LOCKED_SIGMA_MODE,
                    "c2r": locked["max_cost_to_reward"], "weights": W,
                    "run_at": now.isoformat(), "span": [str(lo.date()), str(hi.date())]},
         "spans": report}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'freeze_run.json'}")


if __name__ == "__main__":
    main()
