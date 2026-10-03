"""THE run. Every number published about this strategy comes from here.

Motivation: the headline figures have been quoted from five different scripts
reading three different cached parquets, and at least four published numbers
were stale copies of a value that had since moved. This script rebuilds the
whole pipeline from source in one process, prints the configuration it used
next to the LIVE profile's value for the same field, and writes a JSON sidecar
that report prose is templated from. A number that is not in that JSON does not
go in the report.

It also charges the execution friction the backtest structurally cannot see.
The zero-friction arm is the honest description of the MODEL; it is not a
forecast, because live fills at the ORB boundary do not exist. The friction
arms are charged from fills measured on the paper account -- see
analysis/measure_live_friction.py, which writes analysis/out/live_friction.json.

GATE. The zero-friction arm must reproduce scratchpad/tpsweep/tp2.00.parquet
exactly, trade for trade. That parquet was built by the pre-friction code, so a
byte of drift means the friction patch changed behaviour at 0 bps and every
comparison below is meaningless. The run aborts rather than report.

    python analysis/authoritative_run.py            # full run, ~20 min cold
    python analysis/authoritative_run.py --fast     # reuse cached trade sets
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'orb-live-trading'))

from scripts.etf_basis import (
    compute_candidates_etf_basis, load_margin_rates, truncate_trades, SAMPLE_END)
from scripts.optimize_v1_3class import run_v1_at_k, stats, daily_series
from scripts.run_final_config_causal import classify
from scripts.skip_cheap_by_class import add_force_includes_to_master
from scripts.run_allocation_policies import (
    build_day_table, run_sequential, EQUITY_UNITS, BASE_NOTIONAL)

from orb_live.config.live_config import load_live_config
from orb_live.strategy.v1_strategy import K_SIGMA, MAX_MARGIN_RATE, WEIGHTS, CAP_UNITS
from orb_live.signals.strategy_signals import STOP_ORB_DISTANCE

OUT = Path("analysis/out")
TRADES = OUT / "trades"
#: The zero-friction reference the gate compares against. The ORIGINAL
#: reference (scratchpad/tpsweep/tp2.00.parquet) is RETIRED: it was built with
#: the static ADV gate against a cache ending 2026-05-20, and neither condition
#: can be reproduced now -- the cache was extended for the holdout and the gate
#: is point-in-time. It is kept on disk as the provenance of every figure
#: published before 2026-08-27.
GATE_REF = Path("analysis/out/gate_reference_pit.parquet")
GATE_REF_RETIRED = Path("scratchpad/tpsweep/tp2.00.parquet")

#: run_v1_at_k arguments that define the locked v1 universe and rules. Held in
#: one dict so no scenario can quietly differ from another in anything but
#: friction -- the failure mode that produced the 1.0-vs-2.0 TP1 split.
BASE_ARGS = dict(
    prune_all=True,
    entry_at_boundary=True,
    extra_drops={"BTFX", "UBR"},
    top_n_per_ul=99,
    tp1_target_multiple=2.0,
    #: Point-in-time liquidity gate. The original "static" gate read the last
    #: 60 days in the cache and applied one verdict to six years of backtest --
    #: look-ahead, and irreproducible: extending the cache for the holdout moved
    #: the window and silently deleted 69 trades (register V34). Switching this
    #: retires the old gate reference; see GATE_REF below.
    adv_mode="pit",
    #: Read from the live constant, never written as a literal here. The stop
    #: moved 0.75 -> 1.00 on 2026-08-30 (register V30); the backtest default is
    #: still the old closed form, so a literal in this dict would be one edit
    #: away from silently backtesting a different strategy than live runs.
    stop_orb_distance=STOP_ORB_DISTANCE,
)


#: Opening-liquidity rule threshold, locked 2026-09-14 (register V48). Centre of
#: the 0.10-0.20 plateau: both neighbours improved Sharpe and Calmar in all five
#: subsets, and 0.05 is where the rule starts discarding profitable trades.
MAX_COST_TO_REWARD = 0.15

#: PS filter, locked 2026-09-15 (register V48). k=0.5 tops a 0.25-1.0 plateau
#: (k_sigma_compare.py); sigma is point-in-time expanding, not sigma_master.csv's
#: full history (pit_sigma_test.py: Sharpe 1.43 vs 1.44). Live still reads
#: v1_strategy.K_SIGMA (1.0) and the static CSV -- not yet deployed.
LOCKED_K_SIGMA = 0.5
LOCKED_SIGMA_MODE = "expanding"

#: Sizing, locked 2026-09-15 (register V48 section 9). Class weight by SPY
#: open-gap regime (|SPY open / prior close - 1|: quiet < 0.4%, flood > 1.0%,
#: analysis.refit_sleeve_weights_spy.spy_regimes), (quiet, active, flood).
#: Imposed round numbers; chosen from declared rungs by full-span Calmar.
LOCKED_WEIGHTS = {"C1": (2.0, 2.0, 0.0), "C2": (0.5, 0.5, 0.5), "C3": (1.0, 1.0, 1.5)}


def locked_candidates(master, **kw):
    """compute_candidates_etf_basis detail for the locked configuration."""
    return compute_candidates_etf_basis(master, LOCKED_K_SIGMA, prune_all=True, top_n=1,
                                        return_detail=True, select_by="price",
                                        max_rate=MAX_MARGIN_RATE,
                                        sigma_mode=LOCKED_SIGMA_MODE, **kw)


def locked_args() -> dict:
    """run_v1_at_k arguments for the locked 2026-09-14 configuration (V48).

    BASE_ARGS plus: full universe with no universe-level dollar-ADV gate,
    per-trade participation slippage (commission-only apply_trade_costs), and the
    cost/reward opening-liquidity rule. Pass entry_slippage_bps=stop_slippage_bps=0
    -- the model supplies both. The prior close is the official auction close
    (V42) and the crypto PS filter is as shipped (V45); both are engine defaults.
    """
    return dict(BASE_ARGS, adv_mode="none", slippage_model=load_slippage_model(),
                max_cost_to_reward=MAX_COST_TO_REWARD, sigma_mode=LOCKED_SIGMA_MODE)


def _by_symbol(rate_by_class: dict) -> dict:
    """Expand {class: bps} to {symbol: bps} over the whole instrument set.

    The backtester keys friction by symbol because that is what it has at trade
    construction. Classes come from the LIVE taxonomy, so a symbol reclassified
    in the live profile changes the charge here too, rather than drifting.
    """
    from orb_event_study.config import INSTRUMENTS
    return {sym: float(rate_by_class.get(classify(sym), 0.0))
            for sym in INSTRUMENTS}


def load_slippage_model(path: Path = Path("analysis/out/slippage_model.json")) -> dict:
    """Calibrated participation slippage model -- pass as run_v1_at_k(slippage_model=...).

    Replaces the per-class constant entry/stop rates with a per-trade rate
    driven by order size against opening-range dollar volume. Built by
    analysis/calibrate_slippage_model.py from live fills.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _scenarios(fr: dict) -> dict:
    """Friction arms, in bps, built from the measured JSON -- never typed.

    PER CLASS, not pooled. The live fill sample is 50% C1 crypto while the
    backtest book is 23% C1 and 57% C3, so a single blended rate charges the
    crypto spread (29.5 bps) to SOXL (10.9 bps measured). The pooled_mean arm
    is kept so that composition effect stays visible rather than asserted.

    EOD IS NOT CHARGED, and that is a deliberate reversal. An earlier version
    charged it at the STOP path's rate on the grounds that both are market
    orders. That was a category error. A stop is adverse BY SELECTION -- it
    triggers exactly when price is running against the position, which is why
    91% of live stop fills are adverse. The EOD exit has no target and no
    selection: it fires at 15:58 whatever price is doing, and the gap to live's
    ~15:59:05 fill is a timing difference, not a missed level.

    analysis/eod_exit_drift.py tests that on 2,025 EOD exits: the 15:58->15:59
    move in the position's direction is -1.13 bps mean (t = -0.71, 95% CI
    [-4.29, +2.02]), adverse only 41% of the time. Indistinguishable from zero,
    so it is variance rather than expected loss. The genuinely adverse part of
    an EOD exit is the half-spread, and apply_trade_costs already charges a
    round-trip Corwin-Schultz spread -- charging EOD on top double-counts it.

    eod_stress keeps the superseded assumption visible so the reversal is
    auditable rather than merely claimed.
    """
    pc = fr["per_class"]
    e_mean = {c: pc["entry"][c]["mean"] for c in ("C1", "C2", "C3")}
    e_med = {c: pc["entry"][c]["median"] for c in ("C1", "C2", "C3")}
    s_mean = {c: pc["stop"][c]["mean"] for c in ("C1", "C2", "C3")}
    s_med = {c: pc["stop"][c]["median"] for c in ("C1", "C2", "C3")}
    e5, st = fr["entry_5sec"], fr["exit_stop"]
    return {
        "zero": dict(entry=0.0, stop=0.0, eod=0.0, rates=None,
                     label="zero friction -- the model as published"),
        "entry_only": dict(entry=_by_symbol(e_mean), stop=0.0, eod=0.0,
                           rates={"entry": e_mean},
                           label="entry fill alone, per-class means"),
        "median": dict(entry=_by_symbol(e_med), stop=_by_symbol(s_med), eod=0.0,
                       rates={"entry": e_med, "stop": s_med},
                       label="per-class medians -- understates, skewed right"),
        "mean": dict(entry=_by_symbol(e_mean), stop=_by_symbol(s_mean), eod=0.0,
                     rates={"entry": e_mean, "stop": s_mean},
                     label="per-class means -- THE central estimate"),
        "pooled_mean": dict(entry=e5["mean"], stop=st["mean"], eod=0.0,
                            rates={"entry": {"all": e5["mean"]},
                                   "stop": {"all": st["mean"]}},
                            label="one blended rate -- shows the composition bias"),
        "p90": dict(entry=e5["p90"], stop=st["p90"], eod=0.0,
                    rates={"entry": {"all": e5["p90"]},
                           "stop": {"all": st["p90"]}},
                    label="pooled p90 on every trade -- stress bound, NOT a forecast"),
        "eod_stress": dict(entry=_by_symbol(e_mean), stop=_by_symbol(s_mean),
                           eod=st["mean"],
                           rates={"entry": e_mean, "stop": s_mean,
                                  "eod": {"all": st["mean"]}},
                           label="mean + EOD charged -- superseded, kept for audit"),
    }


def _fmt_rates(sc: dict) -> str:
    r = sc.get("rates")
    if not r:
        return "no friction"
    return "  ".join(
        f"{leg} " + "/".join(f"{k}:{v:.1f}" for k, v in vals.items())
        for leg, vals in r.items())


def trades_for(master, key: str, sc: dict, fast: bool) -> pd.DataFrame:
    """A backtest per friction arm, cached.

    The cache key hashes BASE_ARGS as well as the friction spec. It did not
    until 2026-08-30, and that was a live hazard of exactly the V34 shape: the
    stop moved 0.75 -> 1.00 and every cached arm on disk was still the 0.75
    trade set, so a `--fast` run would have reproduced the old numbers under
    the new label without one line of output looking wrong. A cache key that
    omits a parameter the cached artifact depends on is a silent-wrong-answer
    generator, not an optimisation.
    """
    TRADES.mkdir(parents=True, exist_ok=True)
    spec = json.dumps({"friction": {k: sc[k] for k in ("entry", "stop", "eod")},
                       "base": BASE_ARGS},
                      sort_keys=True, default=str)
    p = TRADES / f"{key}_{hashlib.sha1(spec.encode()).hexdigest()[:10]}.parquet"
    if fast and p.exists():
        return pd.read_parquet(p)
    print(f"  [{key}] {_fmt_rates(sc)} ...", flush=True)
    t = run_v1_at_k(K_SIGMA, master,
                    entry_slippage_bps=sc["entry"],
                    stop_slippage_bps=sc["stop"],
                    eod_slippage_bps=sc["eod"],
                    **BASE_ARGS)
    t = truncate_trades(t)
    t.to_parquet(p)
    print(f"    {len(t):,} trades", flush=True)
    return t


def gate(t0: pd.DataFrame, allow_change: bool = False) -> str:
    """Zero friction must reproduce the stored reference, trade for trade.

    Self-establishing: with no reference on disk it writes one and says so,
    rather than passing vacuously. A gate that reports success when it checked
    nothing is worse than no gate -- that is precisely what L7 was.
    """
    cols = ["date", "symbol", "entry_price", "pnl_pct", "exit_reason"]
    a = t0[cols].sort_values(["date", "symbol"]).reset_index(drop=True)
    if not GATE_REF.exists():
        GATE_REF.parent.mkdir(parents=True, exist_ok=True)
        a.to_parquet(GATE_REF, index=False)
        return (f"REFERENCE ESTABLISHED -- {len(a):,} trades written to "
                f"{GATE_REF.name}. Nothing was verified on this run.")
    b = pd.read_parquet(GATE_REF)[cols]
    b = b.sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(a) != len(b):
        msg = (f"GATE FAILED: {len(a):,} trades vs reference {len(b):,}. "
               "Something upstream changed the universe or the trade set.")
        if not allow_change:
            raise SystemExit(msg)
        # --allow-trade-set-change: the operator is asserting the change is
        # intended (V43 rebased the intraday cache, so the trade set MUST
        # move). The reference on disk is now stale and has to be re-cut
        # deliberately; this run reports against nothing.
        return msg + "  [ALLOWED -- reference is stale, re-cut it]"
    dp = (a["pnl_pct"] - b["pnl_pct"]).abs().max()
    de = (a["entry_price"] - b["entry_price"]).abs().max()
    if dp > 1e-12 or de > 1e-9 or not (a["symbol"] == b["symbol"]).all():
        raise SystemExit(
            f"GATE FAILED: zero-friction output moved "
            f"(max pnl_pct drift {dp:.3e}, max entry drift {de:.3e})")
    return f"PASS -- {len(a):,} trades bit-identical to {GATE_REF.name}"


def evaluate(detail, rates, trades):
    days = build_day_table(detail, trades, rates)
    sized = run_sequential(days, EQUITY_UNITS, "time", partial=True)
    ds = daily_series(sized)
    s = stats(ds)
    sized = sized.copy()
    sized["cls"] = sized["symbol"].map(classify)
    return sized, ds, s


def summarise(sized, ds, s, raw) -> dict:
    yr = {}
    for y, g in ds.groupby(ds.index.year):
        eq = g.cumsum()
        yr[int(y)] = {"pnl": float(g.sum()),
                      "worst_dd": float((eq - eq.cummax()).min()),
                      "days": int(len(g))}
    by_cls = {c: {"trades": int((sized["cls"] == c).sum()),
                  "pnl": float(sized.loc[sized["cls"] == c, "sized_pnl"].sum())}
              for c in ["C1", "C2", "C3"]}
    d = ds[ds != 0]
    return {
        "trades": int(len(sized)),
        "trading_days": int(sized["date"].nunique()),
        "net_pnl": float(s["net_pnl"]),
        "sharpe": float(s["sharpe"]),
        "max_dd": float(s["max_dd"]),
        "calmar": float(s["calmar"]),
        "ann_ret": float(s["ann_ret"]),
        "daily_vol": float(ds.std()),
        "win_days": float((d > 0).mean()) if len(d) else float("nan"),
        "partials": int((sized["frac"] < 0.999).sum()),
        "eod_exit_rate": float(
            raw["exit_reason"].astype(str).str.startswith("EOD").mean()),
        "tp1_hit_rate": float(raw["tp1_hit"].mean()),
        "cost_bps_rt_mean": float(raw["cost_bps_rt"].mean()),
        "by_class": by_cls,
        "by_year": yr,
    }


def main() -> None:
    fast = "--fast" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)

    fr_path = OUT / "live_friction.json"
    if not fr_path.exists():
        raise SystemExit("run analysis/measure_live_friction.py first")
    fr = json.loads(fr_path.read_text(encoding="utf-8"))
    scen = _scenarios(fr)

    live = load_live_config()
    ls = live.strategy_config

    print("=" * 92)
    print("CONFIGURATION -- backtest value, and the live field it must equal")
    print("=" * 92)
    conf = [
        ("sample start", "2020-01-01", "-", "backtest only"),
        ("sample end", str(SAMPLE_END.date()), "-", "live trades past it (D1)"),
        ("k_sigma (PS filter)", K_SIGMA, live.ps_filter_k, "both read v1_strategy"),
        ("cap_units", CAP_UNITS, live.cap_units, "both read v1_strategy"),
        ("max margin rate", MAX_MARGIN_RATE, "-", "sibling cap"),
        ("ORB window", "09:30-10:00", "09:30-10:00", ""),
        ("entry_at_boundary", True, ls.entry_at_boundary,
         "fill on the boundary touch, not the bar close"),
        ("EOD exit", "15:58", f"{ls.eod_exit_hour}:{ls.eod_exit_minute:02d}", ""),
        ("TP1 multiple", BASE_ARGS["tp1_target_multiple"],
         ls.tp1_target_multiple, "x ORB range"),
        ("TP1 by class", "{}", str(ls.tp1_target_multiple_by_class or {}),
         "split reverted 2026-08-25"),
        ("exit ratios", "1.00/0.00/0.00", f"{ls.exit_ratio_tp1:.2f}/"
         f"{ls.exit_ratio_tp2:.2f}/{ls.exit_ratio_tp3:.2f}", "TP1 takes all"),
        # Both sides now read STOP_ORB_DISTANCE, so this row can no longer
        # disagree with itself -- but it still prints, because a reader of the
        # report needs the number and the previous hard-coded pair of strings
        # would have matched each other while both were wrong.
        ("stop", f"{BASE_ARGS['stop_orb_distance']:.2f} x ORB range",
         f"{STOP_ORB_DISTANCE:.2f} x ORB range", "ORB-anchored, V30"),
        ("budget", f"{EQUITY_UNITS:.0f} units (1x equity)", "-", ""),
        ("base notional", f"{BASE_NOTIONAL:,.0f}", "-", ""),
    ]
    mism = 0
    for name, bt, lv, note in conf:
        bad = str(lv) != "-" and str(bt) != str(lv)
        mism += bad
        print(f"{'>>> ' if bad else '    '}{name:<22}{str(bt):<22}"
              f"{str(lv):<22}{note}")
    print("\nweights  " + "   ".join(
        f"{c} {WEIGHTS[(c,'quiet')]:.1f}/{WEIGHTS[(c,'active')]:.1f}"
        f"/{WEIGHTS[(c,'flood')]:.1f}" for c in ["C1", "C2", "C3"]))
    if mism:
        raise SystemExit(f"{mism} configuration mismatch(es) -- fix before publishing")

    print("")
    print("=" * 92)
    print("MEASURED EXECUTION FRICTION -- from paper fills, not assumed")
    print("=" * 92)
    e5, o1, st = fr["entry_5sec"], fr["entry_1min"], fr["exit_stop"]
    print(f"  entry, 5s trigger (live today)  n={e5['n']:<4} "
          f"mean {e5['mean']:.1f}  median {e5['median']:.1f}  p90 {e5['p90']:.1f} bps")
    print(f"  entry, 1m trigger (retired)     n={o1['n']:<4} "
          f"mean {o1['mean']:.1f}  median {o1['median']:.1f}  p90 {o1['p90']:.1f} bps")
    print(f"  stop exit (market order)        n={st['n']:<4} "
          f"mean {st['mean']:.1f}  median {st['median']:.1f}  p90 {st['p90']:.1f} bps")
    print(f"  EOD exit (market order)         n={fr['exit_eod_n']:<4} "
          "not estimable; charged at the stop rate")
    if fr.get("outages_excluded"):
        print(f"  excluded outage sessions: {', '.join(fr['outages_excluded'])}")

    print("")
    print("building candidate table ...", flush=True)
    master = add_force_includes_to_master(pd.read_csv("master_universe.csv"))
    rates = load_margin_rates(master)
    detail = compute_candidates_etf_basis(
        master, K_SIGMA, prune_all=True, top_n=1, return_detail=True,
        select_by="price", max_rate=MAX_MARGIN_RATE)

    print("running scenarios ...", flush=True)
    res, raws = {}, {}
    for key, sc in scen.items():
        t = trades_for(master, key, sc, fast)
        raws[key] = t
        res[key] = evaluate(detail, rates, t)

    print("")
    print("=" * 92)
    print("GATE")
    print("=" * 92)
    g = gate(raws["zero"], allow_change="--allow-trade-set-change" in sys.argv)
    print(f"  zero-friction reproduction: {g}")

    print("")
    print("=" * 92)
    print("HEADLINE")
    print("=" * 92)
    print(f"{'scenario':<12}{'trades':>8}{'P&L':>10}{'vs zero':>10}"
          f"{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>8}   friction (bps)")
    base = res["zero"][2]["net_pnl"]
    summ = {}
    for key, sc in scen.items():
        sized, ds, s = res[key]
        summ[key] = summarise(sized, ds, s, raws[key])
        summ[key]["friction_bps"] = sc.get("rates")
        summ[key]["label"] = sc["label"]
        delta = "" if key == "zero" else f"{s['net_pnl'] - base:+,.0f}"
        print(f"{key:<12}{len(sized):>8,}{s['net_pnl']:>10,.0f}{delta:>10}"
              f"{s['sharpe']:>9.3f}{s['max_dd']*100:>8.2f}%{s['calmar']:>8.2f}"
              f"   {_fmt_rates(sc)}")

    print("")
    for key in scen:
        print(f"  {key:<8} {scen[key]['label']}")

    print("")
    print("=" * 92)
    print("COSTS ALREADY CHARGED IN EVERY ARM ABOVE (including zero friction)")
    print("=" * 92)
    r = raws["zero"]
    pv = r["entry_price"] * r["shares"]
    print(f"  IBKR commission     {(r['cost_ibkr_rt']/pv*1e4).mean():>6.1f} bps "
          "round-trip mean   ($0.005/sh, min $1, max 1%/leg)")
    print(f"  spread + impact     {(r['cost_spread_rt']/pv*1e4).mean():>6.1f} bps "
          "round-trip mean   (Corwin-Schultz + Amihud)")
    print(f"  TOTAL modelled cost {r['cost_bps_rt'].mean():>6.1f} bps "
          "round-trip mean")
    print("")
    print("  Borrow and margin interest are ZERO, not omitted. IBKR accrues both")
    print("  on end-of-day settled balances; every position is flat by 15:58, so")
    print("  neither accrues. The real short-side risk is a failed locate, which")
    print("  costs a trade rather than basis points and is unmodelled either way.")

    print("")
    print("=" * 92)
    print("PER-YEAR NET P&L")
    print("=" * 92)
    yrs = sorted(summ["zero"]["by_year"])
    print(f"{'':<10}" + "".join(f"{y:>10}" for y in yrs))
    for key in scen:
        print(f"{key:<10}" + "".join(
            f"{summ[key]['by_year'][y]['pnl']:>10,.0f}" for y in yrs))
    print("")
    print(f"{'worst DD':<10}" + "".join(f"{y:>10}" for y in yrs))
    for key in scen:
        print(f"{key:<10}" + "".join(
            f"{summ[key]['by_year'][y]['worst_dd']:>10,.0f}" for y in yrs))

    print("")
    print("=" * 92)
    print("PER-CLASS NET P&L")
    print("=" * 92)
    print(f"{'':<10}" + "".join(f"{c:>12}" for c in ["C1", "C2", "C3"]))
    for key in scen:
        print(f"{key:<10}" + "".join(
            f"{summ[key]['by_class'][c]['pnl']:>12,.0f}" for c in ["C1", "C2", "C3"]))

    z = summ["zero"]
    print("")
    print(f"  EOD exit rate {z['eod_exit_rate']:.1%} | TP1 hit rate "
          f"{z['tp1_hit_rate']:.1%} | winning days {z['win_days']:.1%} | "
          f"partial fills {z['partials']:,}")

    payload = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "sample_start": "2020-01-01",
        "sample_end": str(SAMPLE_END.date()),
        "gate": g,
        "config": {n: {"backtest": str(b), "live": str(l)} for n, b, l, _ in conf},
        "weights": {f"{c}_{rg}": WEIGHTS[(c, rg)]
                    for c in ["C1", "C2", "C3"]
                    for rg in ["quiet", "active", "flood"]},
        "friction_measured": fr,
        "eod_drift_test": json.loads(Path("analysis/out/eod_drift.json").read_text())
        if Path("analysis/out/eod_drift.json").exists() else None,
        "scenarios": summ,
    }
    p = OUT / "authoritative.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {p}")
    print("Report prose templates from this file. A number not in it is not published.")


if __name__ == "__main__":
    main()
