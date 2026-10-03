"""Three ways to spend a limited margin budget, compared head to head.

The margin study measured what the constraint COSTS. It never asked how to
spend the budget well. Three policies, one harness, same trades underneath:

  scale-to-fit    Shrink every position uniformly until the day's candidate set
                  fits. What the backtest does today. Nobody is turned away, but
                  margin is committed at 09:31 to candidates that may never fire
                  — and only ~40% of them do.

  first-come      Full size, take breakouts in the order they happen, stop when
                  the budget is exhausted. What LIVE does today. Wastes nothing
                  on non-firers, but orders by breakout TIME, which has no
                  relationship to edge.

  gap-ranked      Full size, priority by gap size in multiples of each fund's
                  own threshold — the one signal that survived every cut
                  (monotone across leverage, side and regime). Holds back
                  capacity for higher-priority candidates that have not fired
                  yet, governed by `reserve`.

`reserve` is the whole reserve-vs-commit tension in one number. When a
breakout arrives, capacity still owed to un-fired higher-priority candidates is
    reserve x sum(their margin)
and the arriving trade is taken only if it fits around that.

    reserve = 0.0   ignore them -> degenerates to first-come with a gap tiebreak
    reserve = 1.0   hold the full amount -> maximal protection, maximal waste
                    on candidates that never fire
    reserve = P(fire)  hold what they are expected to consume. This is the
                    principled setting and needs no fitting: it is measured,
                    not chosen. ~0.43 in-sample.

CAUSALITY: priority, weights, regime and cap_factor are all fixed from the
09:31 candidate set. At each breakout the policy knows only which candidates
have fired SO FAR — never which will fire later. A candidate that never fires
is indistinguishable at decision time from one that fires at 15:45, which is
exactly the problem live faces.
"""
from __future__ import annotations

import sys
import io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # idempotent; wrapping twice closes the buffer
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.etf_basis import compute_candidates_etf_basis, truncate_trades
from scripts.optimize_v1_3class import run_v1_at_k, stats, daily_series
from scripts.skip_cheap_by_class import add_force_includes_to_master
from scripts.run_final_config_causal import classify, assign_regime
from scripts.etf_basis import load_margin_rates, rate_for
from scripts.skip_cheap_by_class import allotment_group

from orb_live.strategy.v1_strategy import K_SIGMA, CAP_UNITS, WEIGHTS


_SPY_REGIME_CACHE = {}


def _spy_regimes():
    """|SPY overnight gap| regime per session; {} if SPY daily is unavailable."""
    if "r" not in _SPY_REGIME_CACHE:
        try:
            from analysis.refit_sleeve_weights_spy import spy_regimes
            _SPY_REGIME_CACHE["r"] = spy_regimes()
        except Exception:
            _SPY_REGIME_CACHE["r"] = {}
    return _SPY_REGIME_CACHE["r"]

TRADES_CACHE = Path("scratchpad/_priority_trades.parquet")

#: One unit of weight = 10% of equity, so equity is 10 units and CAP_UNITS=20
#: is the 200% gross cap.
EQUITY_UNITS = 10.0
BASE_NOTIONAL = 1_000.0     # what pnl_pct is quoted against

BUDGET_PCT = 1.0            # margin budget as a fraction of equity


def load_trades(master) -> pd.DataFrame:
    if TRADES_CACHE.exists():
        print(f"Reusing {TRADES_CACHE}", flush=True)
        return truncate_trades(pd.read_parquet(TRADES_CACHE))
    print("Running backtest for fired trades...", flush=True)
    t = run_v1_at_k(
        K_SIGMA, master, prune_all=True,
        entry_at_boundary=True,
        extra_drops={"BTFX", "UBR"},
        top_n_per_ul=1,
    )
    TRADES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(TRADES_CACHE)
    return truncate_trades(t)


def build_day_table(detail: dict, trades: pd.DataFrame,
                    margin_rates: dict | None = None) -> dict:
    """Per-day candidate records, annotated with what actually happened.

    Returns {date: DataFrame} with one row per candidate: weight, margin rate,
    gap priority, and (entry_time, pnl_pct) where the candidate fired.
    """
    tr = trades.copy()
    tr["date"] = pd.to_datetime(tr["date"]).dt.tz_localize(None).dt.normalize()
    fired = tr.set_index(["date", "symbol"])[["entry_time", "pnl_pct"]]
    fired = fired[~fired.index.duplicated(keep="first")]

    lo, hi = tr["date"].min(), tr["date"].max()
    margin_rates = margin_rates or {}

    days = {}
    for d, cands in detail.items():
        if d < lo or d > hi:
            continue          # outside the traded span — cannot fire by construction
        # WEIGHTS is keyed on the SPY-gap regime since 2026-09-15 (register
        # V47/V48 §9). Bucketing by candidate count here would look up
        # SPY-gap-fitted weights with a different variable -- and the count
        # regime is endogenous to the universe (V16). Days with no SPY gap
        # (cache gap) fall back to the count regime rather than vanishing.
        regime = _spy_regimes().get(pd.Timestamp(d).tz_localize(None).normalize()
                                  if pd.Timestamp(d).tz else pd.Timestamp(d).normalize())
        if not isinstance(regime, str):
            n_uls = len({c["underlying"] for c in cands})
            n_crypto = len({c["underlying"] for c in cands
                            if classify(c["symbol"]) == "C1"})
            regime = assign_regime(n_uls, n_crypto)

        rows = []
        for c in cands:
            sym = c["symbol"]
            cls = classify(sym)
            w = WEIGHTS.get((cls, regime), 0.0)
            if w <= 0:
                continue      # zero-weight buckets are not traded at all
            lev = abs(float(c.get("lev") or 0)) or _lev(sym)
            key = (d, sym)
            et = fired["entry_time"].get(key, pd.NaT)
            rows.append({
                "symbol": sym, "cls": cls, "w": w,
                "r": rate_for(sym, c["etf_dir"], margin_rates, lev),
                "gap": abs(c.get("etf_gap") or 0.0) / (lev * 0.02),
                "entry_time": et,
                "pnl_pct": fired["pnl_pct"].get(key, np.nan),
            })
        if not rows:
            continue
        t = pd.DataFrame(rows)
        # Shared allotment: underlyings that are one exposure divide a single
        # position's worth between them, rather than taking one each. Mirrors
        # SHARED_ALLOTMENT in the live profile -- if these two sides disagree,
        # live and backtest size gold differently and nothing catches it.
        t["group"] = [allotment_group(c["underlying"]) for c in cands
                      if WEIGHTS.get((classify(c["symbol"]), regime), 0.0) > 0]
        share = t.groupby("group")["group"].transform("size")
        t["w"] = t["w"] / share
        t["fired"] = t["entry_time"].notna()
        days[d] = (t, regime)
    return days


def _lev(sym: str) -> float:
    from orb_event_study.config import INSTRUMENTS
    return float(INSTRUMENTS.get(sym, {}).get("leverage", 1) or 1)


def _cap_factor(t: pd.DataFrame, budget: float | None) -> float:
    """The gross-exposure cap, and optionally the margin cap alongside it."""
    exp_mult = t["w"].sum()
    cf = min(1.0, CAP_UNITS / exp_mult) if exp_mult > 0 else 0.0
    if budget is not None:
        exp_margin = (t["w"] * t["r"]).sum()
        if exp_margin > 0:
            cf = min(cf, budget / exp_margin)
    return cf


def run_scale_to_fit(days: dict, budget: float) -> pd.DataFrame:
    """Uniform shrink so the whole candidate set fits. Every firer trades."""
    out = []
    for d, (t, _regime) in days.items():
        cf = _cap_factor(t, budget)
        f = t[t["fired"]]
        for _, r in f.iterrows():
            out.append({"date": d, "symbol": r["symbol"],
                        "sized_pnl": r["pnl_pct"] * BASE_NOTIONAL * r["w"] * cf})
    return pd.DataFrame(out)


def run_sequential(days: dict, budget: float, order: str,
                   reserve: float = 0.0, partial: bool = False,
                   min_frac: float = 0.10,
                   max_need: float | None = None) -> pd.DataFrame:
    """Full size, taken in `order`, until the budget is gone.

    order    — "time" (first-come, what live does) or "gap" (priority).
    reserve  — fraction of un-fired higher-priority margin held back. Only
               meaningful for order="gap"; first-come has no notion of
               priority, so nothing is ever held back for anyone.
    partial  — when a trade does not fit, take it at whatever size the
               remaining budget allows instead of skipping it. Leaving budget
               unused is strictly worse than a smaller position, so this is
               what the account should do; the all-or-nothing default is only
               there to reproduce live's current behaviour.
    min_frac — floor below which a partial is skipped rather than booked.
               Live trades whole shares, so a 2% slice of a position is a
               rounding artifact, not a trade.
    max_need — ceiling on one position's margin, in budget units. The weight
               is scaled down to meet it, so P&L and margin stay consistent;
               a weight past this point does nothing at all. Without it a
               weight large enough to saturate the budget stops sizing its own
               sleeve (w cancels out of the partial-fill branch) and becomes a
               device for starving whatever fires later, which is a lever the
               matrix is not meant to have and a live account cannot pull.
    """
    out = []
    for d, (t, _regime) in days.items():
        t = t.copy()
        if max_need is not None:
            # Clip BEFORE the gross cap reads sum(w). Capping after leaves the
            # weight inert in its own position but still inflating exp_mult,
            # which shrinks cf for every other trade that day -- the crowd-out
            # lever with the sizing removed, i.e. exactly backwards.
            t["w"] = np.minimum(
                t["w"], max_need / t["r"].where(t["r"] > 0, np.inf))
        cf = _cap_factor(t, None)          # gross cap only; margin handled here
        t["need"] = t["w"] * t["r"] * cf
        # Priority is fixed at 09:31, before anything fires.
        t = t.sort_values("gap", ascending=False).reset_index(drop=True)
        t["rank"] = np.arange(len(t))

        f = t[t["fired"]].sort_values("entry_time")
        if order == "gap":
            # Still processed in time order — the day happens in time order —
            # but priority decides who gets held back for whom.
            pass

        committed = 0.0
        fired_so_far: set = set()
        for _, r in f.iterrows():
            if order == "gap" and reserve > 0:
                # Higher priority, not yet seen to fire. Indistinguishable at
                # this moment from one that fires later and one that never does.
                pending = t[(t["rank"] < r["rank"])
                            & (~t["symbol"].isin(fired_so_far))]
                held = reserve * pending["need"].sum()
            else:
                held = 0.0
            room = budget - held - committed
            if r["need"] <= room:
                frac = 1.0
            elif partial and r["need"] > 0 and room / r["need"] >= min_frac:
                frac = room / r["need"]
            else:
                frac = 0.0
            if frac > 0:
                committed += r["need"] * frac
                out.append({"date": d, "symbol": r["symbol"], "frac": frac,
                            "sized_pnl": r["pnl_pct"] * BASE_NOTIONAL
                                         * r["w"] * cf * frac})
            fired_so_far.add(r["symbol"])
    return pd.DataFrame(out)


def report(name: str, sized: pd.DataFrame, base: float | None = None) -> float:
    if sized.empty:
        print(f"{name:<26} no trades")
        return 0.0
    s = stats(daily_series(sized))
    delta = "" if base is None else f"{(s['net_pnl']/base - 1)*100:>+8.1f}%"
    print(f"{name:<26}{len(sized):>7}{s['sharpe']:>9.3f}{s['max_dd']*100:>8.2f}%"
          f"{s['calmar']:>8.2f}{s['net_pnl']:>11,.0f}{delta}")
    return s["net_pnl"]


def main() -> None:
    master = add_force_includes_to_master(pd.read_csv(Path("master_universe.csv")))
    trades = load_trades(master)
    print(f"  {len(trades)} fired trades", flush=True)

    print("Building candidates on the ETF basis...", flush=True)
    detail = compute_candidates_etf_basis(
        master, K_SIGMA, prune_all=True, top_n=1, return_detail=True)
    days = build_day_table(detail, trades, load_margin_rates(master))

    n_cand = sum(len(t) for t, _ in days.values())
    n_fire = sum(int(t["fired"].sum()) for t, _ in days.values())
    p_fire = n_fire / n_cand if n_cand else 0.0
    print(f"  {len(days)} days, {n_cand} candidates, {n_fire} fired "
          f"(P(fire) = {p_fire:.1%})")

    budget = EQUITY_UNITS * BUDGET_PCT

    print(f"\n{'='*74}")
    print(f"ALLOCATION POLICIES — margin budget {BUDGET_PCT:.0%} of equity")
    print(f"{'='*74}")
    print(f"{'policy':<26}{'trades':>7}{'Sharpe':>9}{'MaxDD':>9}"
          f"{'Calmar':>8}{'Net P&L':>11}{'vs base':>9}")

    # Unconstrained reference: no margin limit at all.
    unc = run_sequential(days, budget=1e9, order="time")
    base = report("unconstrained", unc)

    report("scale-to-fit", run_scale_to_fit(days, budget), base)
    report("first-come (live today)", run_sequential(days, budget, "time"), base)
    for res in (0.0, p_fire, 1.0):
        label = f"gap-ranked, reserve={res:.2f}"
        if abs(res - p_fire) < 1e-9:
            label += " *"
        report(label, run_sequential(days, budget, "gap", reserve=res), base)

    print("\n  * reserve = measured P(fire); not a fitted parameter.")


if __name__ == "__main__":
    main()
