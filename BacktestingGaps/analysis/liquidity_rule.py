"""Opening-liquidity rule: skip trades the account cannot afford to execute.

The participation slippage model (analysis/calibrate_slippage_model.py) prices
each trade by order size against opening-range dollar volume. Charged across the
full universe it took C2 from +1,162 to +70 and C3 down 587, while crypto moved
-80 -- the thin names brought back when the whole-history ADV screen was removed
are expensive to trade at live size. The same variable can decide which trades
to take at all, causally, per trade, and scaled to account size -- replacing the
universe-level ADV cut (V46) that misjudged names like KORU.

Two rules, both computable at 10:00 when the opening range closes:

  participation cap   skip if notional / (median opening-range $ per minute) > P
  cost-to-reward cap  skip if (modelled entry slip + modelled stop slip)
                      / (TP1 distance = tp1_multiple x ORB range) > R
                      -- scale-free, and ties cost to what the trade can earn

THRESHOLDS ARE ROUND AND DECLARED BEFORE SCORING. A threshold picked by its P&L
is V1/V47 again. Each rung is scored on five subsets (full, H1, H2, odd ISO
weeks, even ISO weeks) against no rule on the same sessions; a rule that is real
improves most of them, and neighbouring rungs should agree (a plateau, not a
spike).

Book: flat, full universe, no ADV gate, crypto PS as shipped, V42 official prior
close, participation friction charged at the live median order size.
The filter is applied to trades after the backtest; at flat weights the margin
budget never binds (margin_utilisation.py), so dropping a trade does not change
which other trades fill.

    python analysis/liquidity_rule.py
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
INTRADAY = Path("orb_event_study/cache/intraday")
MAX_NEED_FRAC = 0.5

RULES = [
    ("no rule", None, None),
    ("participation <= 1.00", 1.00, None),
    ("participation <= 0.50", 0.50, None),
    ("participation <= 0.25", 0.25, None),
    ("participation <= 0.10", 0.10, None),
    ("cost/reward <= 0.20", None, 0.20),
    ("cost/reward <= 0.10", None, 0.10),
    ("cost/reward <= 0.05", None, 0.05),
]


def week_parity(idx: pd.DatetimeIndex) -> np.ndarray:
    wk = idx.isocalendar()
    return ((np.asarray(wk["year"], dtype=int) * 53 + np.asarray(wk["week"], dtype=int)) % 2) == 0


def main() -> None:
    import orb_backtester as OB
    import scripts.etf_basis as EB
    import analysis.authoritative_run as A
    import _production_run as P
    from scripts import as_traded as AT
    from scripts.optimize_v1_3class import run_v1_at_k, stats, daily_series, trading_days
    from scripts.skip_cheap_by_class import add_force_includes_to_master, allotment_group
    from scripts.etf_basis import compute_candidates_etf_basis, load_margin_rates, rate_for
    from scripts.run_allocation_policies import run_sequential, EQUITY_UNITS, _lev, classify

    AT.USE_AS_TRADED = True
    AT._cache.clear()
    cut = EB.SAMPLE_END
    EB.SAMPLE_END = None
    cap = MAX_NEED_FRAC * EQUITY_UNITS

    sm = A.load_slippage_model()
    notional = sm["default_notional"]
    master = add_force_includes_to_master(pd.read_csv("master_universe.csv"))
    rates = load_margin_rates(master)
    fr = json.loads((OUT / "live_friction.json").read_text(encoding="utf-8"))
    sc = A._scenarios(fr)["mean"]
    base = dict(A.BASE_ARGS, adv_mode="none", slippage_model=sm)
    print(f"friction: participation model at ${notional:,.0f} per order; "
          f"TP1 multiple {base['tp1_target_multiple']}")

    print("building candidates ...", flush=True)
    detail = compute_candidates_etf_basis(master, A.K_SIGMA, prune_all=True, top_n=1,
                                          return_detail=True, select_by="price",
                                          max_rate=A.MAX_MARGIN_RATE)
    print("running backtest ...", flush=True)
    t = run_v1_at_k(A.K_SIGMA, master, entry_slippage_bps=0.0, stop_slippage_bps=0.0,
                    eod_slippage_bps=sc["eod"], **base)
    t["date"] = pd.to_datetime(t["date"]).dt.tz_localize(None).dt.normalize()
    now_et = pd.Timestamp.now(tz="America/New_York")
    if now_et.hour * 60 + now_et.minute < 16 * 60 + 15:
        t = t[t["date"] < now_et.tz_localize(None).normalize()]

    # ---- per-trade liquidity features, known at 10:00 --------------------------
    month_cache: dict = {}

    def orb_dpm(sym: str, d: pd.Timestamp) -> float:
        key = (sym, d.strftime("%Y-%m"))
        if key not in month_cache:
            p = INTRADAY / sym / f"{key[1]}.parquet"
            if not p.exists():
                month_cache[key] = pd.Series(dtype=float)
            else:
                x = pd.read_parquet(p)
                ts = pd.to_datetime(x["timestamp"])
                ts = ts.dt.tz_localize("America/New_York") if ts.dt.tz is None else ts.dt.tz_convert("America/New_York")
                mins = ts.dt.hour * 60 + ts.dt.minute
                w = x[(mins >= 570) & (mins < 600)].assign(
                    day=ts[(mins >= 570) & (mins < 600)].dt.tz_localize(None).dt.normalize())
                month_cache[key] = (w["close"] * w["volume"]).groupby(w["day"]).median()
        return float(month_cache[key].get(d, np.nan))

    t["dpm"] = [orb_dpm(s, d) for s, d in zip(t["symbol"], t["date"])]
    pmax = float(sm["part_max"])
    t["part"] = np.where(t["dpm"] > 0, notional / t["dpm"], np.inf)
    costs = [OB.model_slippage_bps(type("C", (), {"slippage_model": sm, "slippage_notional": notional})(),
                                   {"dollar_per_min": v}) for v in t["dpm"]]
    t["entry_model_bps"] = [c[0] for c in costs]
    t["stop_model_bps"] = [c[1] for c in costs]
    t["cs_bps"] = t["symbol"].map(lambda s: P._CS_BPS_RT.get(s, P._DEFAULT_CS_BPS_RT))
    boundary = np.where(t["direction"] > 0, t["orb_high"], t["orb_low"])
    t["tp_dist_bps"] = base["tp1_target_multiple"] * (t["orb_high"] - t["orb_low"]) / boundary * 1e4
    # no spread term: whole-sample constant (look-ahead) already inside measured slippage
    t["cost_reward"] = (t["entry_model_bps"] + t["stop_model_bps"]) / t["tp_dist_bps"]
    t["cls"] = t["symbol"].map(classify)
    print(f"  {len(t):,} trades; participation median {np.nanmedian(t['part'][np.isfinite(t['part'])]):.3f}; "
          f"cost/reward median {t['cost_reward'].median():.3f}; no opening volume on "
          f"{int((~np.isfinite(t['part'])).sum())}")

    # ---- flat book with a trade filter ---------------------------------------
    def book(keep_mask: pd.Series, lo, hi):
        tr = t[keep_mask & (t["date"] >= lo) & (t["date"] <= hi)]
        f = tr.set_index(["date", "symbol"])[["entry_time", "pnl_pct"]]
        f = f[~f.index.duplicated(keep="first")]
        days = {}
        for d, cands in detail.items():
            dd = pd.Timestamp(d)
            dd = dd.tz_localize(None).normalize() if dd.tz else dd.normalize()
            if dd < lo or dd > hi or not cands:
                continue
            rows = []
            for c in cands:
                sym = c["symbol"]
                lev = abs(float(c.get("lev") or 0)) or _lev(sym)
                rows.append({"symbol": sym, "cls": classify(sym), "w": 1.0,
                             "r": rate_for(sym, c["etf_dir"], rates, lev),
                             "gap": abs(c.get("etf_gap") or 0.0) / (lev * 0.02),
                             "entry_time": f["entry_time"].get((dd, sym), pd.NaT),
                             "pnl_pct": f["pnl_pct"].get((dd, sym), np.nan),
                             "ul": c["underlying"], "group": allotment_group(c["underlying"])})
            tab = pd.DataFrame(rows)
            tab["w"] = tab["w"] / tab.groupby("group")["ul"].transform("nunique")
            tab["fired"] = tab["entry_time"].notna()
            days[dd] = (tab, "flat")
        sized = run_sequential(days, EQUITY_UNITS, "time", partial=True, max_need=cap)
        return sized

    lo_all, hi_all = t["date"].min(), t["date"].max()
    sessions = trading_days(lo_all, hi_all)
    mid = sessions[len(sessions) // 2]
    par = week_parity(sessions)
    subsets = {"full": set(sessions), "H1": set(sessions[sessions < mid]),
               "H2": set(sessions[sessions >= mid]),
               "odd wks": set(sessions[par]), "even wks": set(sessions[~par])}

    results = {}
    for name, pcap, rcap in RULES:
        keep = pd.Series(True, index=t.index)
        if pcap is not None:
            keep &= np.isfinite(t["part"]) & (t["part"] <= pcap)
        if rcap is not None:
            keep &= t["cost_reward"] <= rcap
        sized = book(keep, lo_all, hi_all)
        ds = daily_series(sized).reindex(sessions, fill_value=0.0)
        sized = sized.assign(cls=sized["symbol"].map(classify))
        sub = {}
        for k, ss in subsets.items():
            x = ds[ds.index.isin(ss)]
            sub[k] = stats(x)
        results[name] = {"sub": sub, "kept": int(keep.sum()),
                         "by_cls": sized.groupby("cls")["sized_pnl"].agg(["size", "sum"])}
        print(f"  scored: {name}", flush=True)

    print("\n" + "=" * 110)
    print("OPENING-LIQUIDITY RULES -- flat full universe, participation friction, $10k base")
    print("=" * 110)
    print(f"  {'rule':<24}{'trades kept':>12}{'P&L':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>9}"
          f"{'Calmar':>8}   C1 / C2 / C3 P&L")
    for name, *_ in RULES:
        r = results[name]; s = r["sub"]["full"]; bc = r["by_cls"]["sum"]
        print(f"  {name:<24}{r['kept']:>12}{s['net_pnl']:>9,.0f}{s['ann_ret']:>7.1%}"
              f"{s['sharpe']:>8.2f}{s['max_dd']:>8.2%}{s['calmar']:>8.2f}   "
              f"{bc.get('C1', 0):>6,.0f} /{bc.get('C2', 0):>6,.0f} /{bc.get('C3', 0):>6,.0f}")

    for metric in ("sharpe", "calmar"):
        print(f"\n  {metric.upper()} delta vs no rule, by subset")
        print(f"  {'rule':<24}" + "".join(f"{k:>11}" for k in subsets) + "   improved")
        b0 = results["no rule"]["sub"]
        for name, *_ in RULES[1:]:
            s = results[name]["sub"]
            cells = [s[k][metric] - b0[k][metric] for k in subsets]
            wins = sum(c > 0 for c in cells)
            print(f"  {name:<24}" + "".join(f"{c:>+11.3f}" for c in cells) + f"   {wins}/5")

    print("\n  what each rule removes (all trades, not just the flat book's picks):")
    for name, pcap, rcap in RULES[1:]:
        drop = pd.Series(False, index=t.index)
        if pcap is not None:
            drop |= ~(np.isfinite(t["part"]) & (t["part"] <= pcap))
        if rcap is not None:
            drop |= ~(t["cost_reward"] <= rcap)
        g = t[drop]
        print(f"  {name:<24} drops {len(g):>5}  mean {g['pnl_pct'].mean() * 100:+.3f}%  "
              f"by class {g['cls'].value_counts().to_dict()}")

    EB.SAMPLE_END = cut
    AT.USE_AS_TRADED = False
    print("\nThresholds were declared before scoring. Prefer a rung whose neighbours")
    print("agree over the single best row.")


if __name__ == "__main__":
    main()
