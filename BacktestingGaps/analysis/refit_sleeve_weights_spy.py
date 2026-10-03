"""Sleeve weights on an EXOGENOUS regime: |SPY overnight gap|. V43 follow-on.

The n_uls regime is endogenous to the universe, and it broke the moment the
universe changed. Screening 29 underlyings down to 18 dropped mean n_uls from
2.58 to 2.18 and collapsed flood days from 88 to 35, while ACTIVE_MIN=5 and
FLOOD_MIN=8 stayed at values calibrated for the larger universe. The optimizer
then fitted `independent/active` on 95 days and `crypto/flood` on 35, and
returned non-monotone weights (market 4.0/0.5/2.0) that no mechanism explains.
That is register V16 -- regime as an artifact of universe construction --
biting in a new way, and it is the same shape as V34.

|SPY overnight gap| does not move when the universe changes. It is also the
variable the mechanism story actually refers to, and it separates the sleeves
far better than n_uls did:

    C3/market edge by SPY-gap quintile:  -0.006  +0.332  +0.026  +0.401  +1.620
    C1/crypto edge by the same:          +0.096  +0.654  +0.691  -0.776  -0.968

Broad names earn when the market gaps; crypto is harmed by it. n_uls showed
crypto best at 5-7 underlyings and bad at 8+, with nothing to explain it.

THRESHOLDS ARE ROUND AND FIXED, not quantiles of this sample -- a sample
quantile is a look-ahead boundary, and a searched one is V1 all over again:

    quiet   |SPY gap| <  0.4%     57.6% of sessions
    active  0.4% - 1.0%           30.1%
    flood   |SPY gap| >  1.0%     12.3%

Against n_uls's 86/8/5 split, every cell is now populated well enough to mean
something.

IN-SAMPLE, no holdout (V3). Still the ceiling of what the change is worth.

    python analysis/refit_sleeve_weights_spy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'orb-live-trading'))

OUT = Path("analysis/out")
SLEEVES = ("market", "independent", "crypto")
REGIMES = ("quiet", "active", "flood")
#: Widened past 5.0 -- independent/flood pinned to the old ceiling, which is
#: clipping, not optimising (the V31 failure refit_weights.py exists to catch).
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
QUIET_MAX, FLOOD_MIN = 0.4, 1.0     # percent, |SPY overnight gap|
#: Ceiling on one position's margin, as a fraction of the 10-unit budget.
#: Without it the optimizer does not use a large weight to size a sleeve up --
#: it uses it to saturate the day's budget and starve whatever fires later,
#: because `w` cancels out of run_sequential's partial-fill branch. Measured on
#: the gap-selector refit: sweeping crypto/active 0 -> 8 gained 2,109 of P&L, of
#: which only 904 was crypto's own book and 1,205 came from other sleeves being
#: crowded down. Crypto's own P&L PEAKED at w=3.0 and fell after. Sharpe over
#: w=2.5..8.0 spanned 0.012 while max drawdown went 12.4% -> 16.2%.
#: ROUND AND FIXED at half the budget, not searched.
MAX_NEED_FRAC = 0.5


def spy_regimes() -> pd.Series:
    d = pd.read_parquet("orb_event_study/cache/daily/SPY.parquet")
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.set_index("date").sort_index()
    g = (d["open"] / d["close"].shift(1) - 1).abs() * 100
    return pd.Series(np.where(g < QUIET_MAX, "quiet",
                              np.where(g > FLOOD_MIN, "flood", "active")),
                     index=g.index).where(g.notna())


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["sharpe", "calmar"],
                    default="sharpe",
                    help="what coordinate descent maximises. Sharpe is blind "
                         "to drawdown, and the ceiling-pinned cells bought "
                         "0.01 of Sharpe for several points of DD.")
    ap.add_argument("--no-cap", action="store_true",
                    help="disable the per-trade margin ceiling, restoring the "
                         "crowd-out lever. For comparison only.")
    ap.add_argument("--select", choices=["price", "advol", "lev", "gap", "gapn"], default="price",
                    help="sibling selector. 'price' is the locked v1 rule on "
                         "as-traded prices; 'advol' ranks by point-in-time "
                         "dollar volume and is basis-invariant.")
    args = ap.parse_args()

    import analysis.authoritative_run as A
    import orb_live.strategy.v1_strategy as V1
    from scripts import as_traded as AT
    from scripts import run_allocation_policies as RAP
    from scripts.optimize_v1_3class import run_v1_at_k, stats, daily_series
    from scripts.skip_cheap_by_class import add_force_includes_to_master, allotment_group
    from scripts.etf_basis import compute_candidates_etf_basis, load_margin_rates, rate_for
    from scripts.run_allocation_policies import run_sequential, EQUITY_UNITS, _lev
    from orb_event_study.config import INSTRUMENTS

    AT.USE_AS_TRADED = True
    AT._cache.clear()

    screen = pd.DataFrame(json.loads((OUT / "underlying_screen.json").read_text()))
    keep = screen[screen["verdict"] == "KEEP"]
    sleeve_of_ul = {}
    for _, r in keep.iterrows():
        s = r["sleeve"]
        sleeve_of_ul[r["ul"]] = ("crypto" if s.startswith("crypto")
                                 else "market" if s == "market" else "independent")
    keep_uls = set(sleeve_of_ul)

    def sleeve(sym: str) -> str:
        return sleeve_of_ul.get(INSTRUMENTS.get(sym, {}).get("underlying"), "market")

    obj = args.objective
    cap = None if args.no_cap else MAX_NEED_FRAC * EQUITY_UNITS
    reg = spy_regimes()
    print(f"sibling selector: {args.select}")
    print(f"per-trade margin ceiling: "
          + ("NONE -- crowd-out lever live" if cap is None
             else f"{cap:.1f} of {EQUITY_UNITS:.0f} budget units"))
    print(f"regime thresholds: quiet <{QUIET_MAX}%, flood >{FLOOD_MIN}%")
    print("  session shares: " + "  ".join(
        f"{k}={v/len(reg.dropna())*100:.0f}%"
        for k, v in reg.dropna().value_counts().items()))

    master = add_force_includes_to_master(pd.read_csv("master_universe.csv"))
    rates = load_margin_rates(master)
    fr = json.loads((OUT / "live_friction.json").read_text(encoding="utf-8"))
    sc = A._scenarios(fr)["mean"]

    print("building candidates ...", flush=True)
    detail = compute_candidates_etf_basis(
        master, A.K_SIGMA, prune_all=True, top_n=1, return_detail=True,
        select_by=args.select, max_rate=A.MAX_MARGIN_RATE)
    detail = {d: [c for c in cs if c["underlying"] in keep_uls]
              for d, cs in detail.items()}
    detail = {d: cs for d, cs in detail.items() if cs}

    base = dict(A.BASE_ARGS)
    base["extra_drops"] = set(base.get("extra_drops") or set()) | {
        s for s, i in INSTRUMENTS.items() if i["underlying"] not in keep_uls}
    print("running backtest ...", flush=True)
    t = A.truncate_trades(run_v1_at_k(
        A.K_SIGMA, master, entry_slippage_bps=sc["entry"],
        stop_slippage_bps=sc["stop"], eod_slippage_bps=sc["eod"], **base))
    t = t[t["symbol"].map(lambda s: INSTRUMENTS.get(s, {}).get("underlying"))
          .isin(keep_uls)]
    print(f"  {len(t):,} trades")

    # unit-weight day tables, regime from SPY rather than n_uls
    tr = t.copy()
    tr["date"] = pd.to_datetime(tr["date"]).dt.tz_localize(None).dt.normalize()
    fired = tr.set_index(["date", "symbol"])[["entry_time", "pnl_pct"]]
    fired = fired[~fired.index.duplicated(keep="first")]
    lo, hi = tr["date"].min(), tr["date"].max()

    days, seen = {}, {"quiet": 0, "active": 0, "flood": 0}
    for d, cands in detail.items():
        dd = pd.Timestamp(d)
        dd = dd.tz_localize(None).normalize() if dd.tz else dd.normalize()
        if dd < lo or dd > hi:
            continue
        rg = reg.get(dd)
        if not isinstance(rg, str):
            continue
        seen[rg] += 1
        rows = []
        for c in cands:
            sym = c["symbol"]
            lev = abs(float(c.get("lev") or 0)) or _lev(sym)
            key = (dd, sym)
            rows.append({
                "symbol": sym, "cls": sleeve(sym), "w": 1.0,
                "r": rate_for(sym, c["etf_dir"], rates, lev),
                "gap": abs(c.get("etf_gap") or 0.0) / (lev * 0.02),
                "entry_time": fired["entry_time"].get(key, pd.NaT),
                "pnl_pct": fired["pnl_pct"].get(key, np.nan),
                "group": allotment_group(c["underlying"]),
            })
        if not rows:
            continue
        tab = pd.DataFrame(rows)
        tab["w"] = tab["w"] / tab.groupby("group")["group"].transform("size")
        tab["fired"] = tab["entry_time"].notna()
        days[dd] = (tab, rg)
    print(f"  {len(days):,} day tables   " + "  ".join(f"{k}={v}" for k, v in seen.items()))

    def evaluate(w):
        scaled = {}
        for d, (tab, rg) in days.items():
            mult = tab["cls"].map(lambda c: w[(c, rg)])
            t2 = tab.assign(w=tab["w"] * mult)
            t2 = t2[t2["w"] > 0]
            if not t2.empty:
                scaled[d] = (t2, rg)
        if not scaled:
            return None
        sized = run_sequential(scaled, EQUITY_UNITS, "time", partial=True,
                               max_need=cap)
        return None if sized.empty else stats(daily_series(sized))

    def fmt(w):
        return "  ".join(f"{s[:4]} " + "/".join(f"{w[(s, r)]:.1f}" for r in REGIMES)
                         for s in SLEEVES)

    flat = {(s, r): 1.0 for s in SLEEVES for r in REGIMES}
    base_s = evaluate(flat)
    print(f"\nflat 1.0: sharpe {base_s['sharpe']:.3f}"
          f"  calmar {base_s['calmar']:.3f}"
          f"  P&L {base_s['net_pnl']:,.0f}")

    w, best = dict(flat), base_s[obj]
    print("\n" + "=" * 92)
    print(f"COORDINATE DESCENT on {obj} -- exogenous SPY-gap regime")
    print("=" * 92)
    for p in range(1, 7):
        moved = False
        for s in SLEEVES:
            for rg in REGIMES:
                cur = w[(s, rg)]
                for g in GRID:
                    if g == cur:
                        continue
                    trial = dict(w)
                    trial[(s, rg)] = g
                    st = evaluate(trial)
                    if st and st[obj] > best + 1e-6:
                        best, w, moved, cur = st[obj], trial, True, g
                        print(f"  pass {p}: {s}/{rg} -> {g}   {obj} {best:.4f}",
                              flush=True)
        if not moved:
            print(f"  converged after {p} pass(es)")
            break

    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    print(f"  {'matrix':<12}{'Sharpe':>9}{'P&L':>10}{'MaxDD':>9}{'Calmar':>8}   weights")
    for lab, ww in (("flat 1.0", flat), ("refit", w)):
        s_ = evaluate(ww)
        print(f"  {lab:<12}{s_['sharpe']:>9.3f}{s_['net_pnl']:>10,.0f}"
              f"{s_['max_dd']:>8.2%}{s_['calmar']:>8.2f}   {fmt(ww)}")

    hi_g = max(GRID)
    ceil = [f"{s}/{r}" for s in SLEEVES for r in REGIMES if w[(s, r)] == hi_g]
    print("")
    print(f"  at CEILING ({hi_g}) -- CLIPPED, not optimised: "
          + (", ".join(ceil) if ceil else "none"))
    print("\n  monotonicity per sleeve (quiet -> active -> flood):")
    for s in SLEEVES:
        v = [w[(s, r)] for r in REGIMES]
        shape = ("rising" if v[0] <= v[1] <= v[2] else
                 "falling" if v[0] >= v[1] >= v[2] else "ZIGZAG -- suspect")
        print(f"    {s:<14}{v}   {shape}")

    p_ = OUT / (f"refit_sleeve_weights_spy_{args.select}"
                + ("" if obj == "sharpe" else f"_{obj}")
                + ("_nocap" if args.no_cap else "") + ".json")
    _ = p_
    p_.write_text(json.dumps({
        "select_by": args.select,
        "max_need": cap,
        "objective": obj,
        "thresholds": {"quiet_max": QUIET_MAX, "flood_min": FLOOD_MIN},
        "regime_days": seen,
        "refit": {f"{s}_{r}": w[(s, r)] for s in SLEEVES for r in REGIMES},
        "flat_stats": {k: float(v) for k, v in base_s.items()},
        "refit_stats": {k: float(v) for k, v in evaluate(w).items()},
    }, indent=2), encoding="utf-8")
    AT.USE_AS_TRADED = False
    print(f"\nwrote {p_}")


if __name__ == "__main__":
    main()
