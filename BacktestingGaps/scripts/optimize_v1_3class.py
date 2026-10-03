"""
scripts/optimize_v1_3class.py
==============================
Re-optimize v1 with the 3-class macro-independence taxonomy and broad-gap
regime definition.

Phase 1 — sigma (PS k) sweep × caps
Phase 2 — coordinate descent on 9-bucket 3-class × 3-regime sizing matrix,
          at the best k for each cap
"""
import sys, io, contextlib
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from orb_backtester import StrategyConfig, run_backtest
from orb_event_study.config import INSTRUMENTS
import _production_run as P
from scripts.skip_cheap_by_class import (
    filter_master, gate_by_sigma_coverage, run_hybrid, apply_pit_dollar_adv_filter,
    skip_cheap_then_top2_when_3plus, add_force_includes_to_master,
    apply_static_dollar_adv_filter, load_daily_close,
    apply_3class_regime_sizing, compute_broad_gap_days,
    CLASS_1_SYMS, CLASS_2_SYMS,
    EXPLICIT_DROPS, FORCE_INCLUDE, DIRECTION_FILTERS,
)
from scripts.sigma_master import load_sigma_master

ROOT       = Path(".")
MASTER_CSV = ROOT / "master_universe.csv"
START      = 10_000.0

#: Sharpe annualises on TRADING days, the standard convention. Until 2026-08-31
#: this project padded P&L onto a calendar range -- weekends as structural
#: zeros -- and used sqrt(365). That is algebraically almost the same thing,
#: because padding shrinks the mean by p and the sd by sqrt(p) with p = 252/365,
#: so sqrt(365)*sqrt(p) = sqrt(252); measured at this strategy's moments the two
#: differ by 0.54%, the padded form reading LOWER. Nothing published under it was
#: inflated. It is changed because a reader reconciling a quoted Sharpe against
#: sqrt(252) would find a discrepancy and could not tell it was deliberate.
TRADING_DAYS_PER_YEAR = 252
#: Per-PERIOD risk-free. Must be divided by the same factor Sharpe multiplies by
#: or it is silently mis-scaled by 365/252.
RF_DAILY   = 0.043 / TRADING_DAYS_PER_YEAR

#: Liquid, continuously listed, and already in the cache -- its bar dates ARE the
#: NYSE sessions, so the trading calendar is read from data rather than kept as a
#: hard-coded holiday list that would silently rot. (orb_live.core.calendar only
#: carries 2025-2027 holidays; this sample starts in 2020.)
_CALENDAR_SYMBOL = "SPY"
_CALENDAR_CACHE: pd.DatetimeIndex | None = None


def trading_days(lo, hi) -> pd.DatetimeIndex:
    """NYSE sessions in [lo, hi], inclusive."""
    global _CALENDAR_CACHE
    if _CALENDAR_CACHE is None:
        d = pd.read_parquet(
            Path("orb_event_study/cache/daily") / f"{_CALENDAR_SYMBOL}.parquet",
            columns=["date"])
        _CALENDAR_CACHE = pd.DatetimeIndex(
            pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize().unique()
        ).sort_values()
    lo, hi = pd.Timestamp(lo).normalize(), pd.Timestamp(hi).normalize()
    return _CALENDAR_CACHE[(_CALENDAR_CACHE >= lo) & (_CALENDAR_CACHE <= hi)]


def stats(daily: pd.Series) -> dict:
    if daily.empty: return dict(sharpe=float("nan"), max_dd=float("nan"),
                                 calmar=float("nan"), net_pnl=0.0)
    r = daily / START
    sh = ((r.mean() - RF_DAILY) / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
          if r.std() > 0 else float("nan"))
    eq = START * (1 + r).cumprod()
    pk = eq.cummax()
    dd = ((eq - pk) / pk).min()
    yrs = (daily.index[-1] - daily.index[0]).days / 365
    ann = (eq.iloc[-1] / START) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else float("nan")
    cal = ann / abs(dd) if dd != 0 else float("nan")
    return dict(sharpe=sh, max_dd=dd, calmar=cal, ann_ret=ann, net_pnl=daily.sum())


def daily_series(sized: pd.DataFrame) -> pd.Series:
    """Daily P&L on the NYSE session calendar.

    A trading day with no position IS an observation of zero return and stays.
    A weekend is not an observation at all and is dropped -- see
    TRADING_DAYS_PER_YEAR. Drawdown, annual return and Calmar are unaffected
    either way: a zero contributes 1.0 to the cumulative product and moves no
    date. Only Sharpe changes.
    """
    d = sized.groupby("date")["sized_pnl"].sum()
    idx = trading_days(sized["date"].min(), sized["date"].max())
    if len(idx) == 0:
        return d
    return d.reindex(idx, fill_value=0.0)


# ── Run a single backtest at given k, return annotated combined trades ────────
def run_v1_at_k(k: float, master: pd.DataFrame,
                  prune: bool = True,
                  extra_drops: set | None = None,
                  prune_all: bool = False,
                  tp1_target_multiple: float = 2.0,
                  entry_at_boundary: bool = False,
                  top_n_per_ul: int = 2,
                  # 15:58 — the time live actually flattens (LiveConfig
                  # eod_flatten_lead_secs=120 against a 16:00 close). This used
                  # to be 16:00, which no delivered RTH bar reaches, so the exit
                  # silently fell through to the end-of-loop fallback and booked
                  # the 15:59 close: every backtest figure carried four minutes
                  # of drift live never saw, worth 5.5% of total P&L.
                  eod_exit_hour: int = 15,
                  eod_exit_minute: int = 58,
                  stop_from_entry: bool = False,
                  stop_orb_distance: float | None = None,
                  stop_entry_distance: float = 0.75,
                  same_bar_fill_risk: bool = False,
                  # Execution friction, bps. 0.0 reproduces every published
                  # figure exactly; measured live values are written to
                  # analysis/out/live_friction.json by measure_live_friction.py.
                  entry_slippage_bps: float = 0.0,
                  stop_slippage_bps: float = 0.0,
                  eod_slippage_bps: float = 0.0,
                  #: Participation slippage model (analysis/out/slippage_model.json).
                  #: When given, it replaces entry_ and stop_slippage_bps per trade.
                  slippage_model: dict | None = None,
                  slippage_notional: float | None = None,
                  #: Opening-liquidity rule; requires slippage_model. See
                  #: StrategyConfig.max_cost_to_reward. None = off.
                  max_cost_to_reward: float | None = None,
                  #: "static" is the ORIGINAL gate: one $ADV from the most
                  #: recent 60 days in the cache, applied to the whole
                  #: 2020-2026 backtest. That is look-ahead and it is not
                  #: reproducible -- see register V34. "pit" decides each trade
                  #: on the trailing $ADV as of that date, lagged one session.
                  #: Default stays "static" so legacy callers do not change
                  #: silently; the authoritative pipeline passes "pit".
                  adv_mode: str = "static",
                  #: UL-equivalent gap threshold. The live rule is
                  #: |ETF gap| >= leverage * gap_threshold, so a 3x fund needs
                  #: 6% to clear a 2% UL move. Exposed for the stability sweep
                  #: (register V41); the default reproduces every published run.
                  #: NOTE this constant ALSO gates the candidate pool in
                  #: scripts/etf_basis.GAP_THRESHOLD. Moving one without the
                  #: other measures a strategy that does not exist -- see
                  #: analysis/param_stability.py, which moves both together.
                  gap_threshold: float = 0.02,
                  #: Opening-range length in minutes. StrategyConfig defaults to
                  #: 30 and run_v1_at_k never overrode it, so this was
                  #: unreachable from the sweep before V41.
                  orb_minutes: int = 30,
                  #: Whether binary_event symbols bypass the $5M dollar-ADV
                  #: floor. They always have, but on the Alpha Vantage cache
                  #: most of them had too few bars to generate trades, so the
                  #: exemption was never cashed. On the IBKR rebuild it lets 269
                  #: sub-floor trades through -- EURL at $0.56M ADV, MEXX at
                  #: $0.46M. Default True preserves published behaviour; the
                  #: counterfactual passes False (register V43).
                  adv_exempt_binary_event: bool = True,
                  #: "static" = sigma_master.csv; "expanding" = point-in-time (V48)
                  sigma_mode: str = "static") -> pd.DataFrame:
    """If prune=False, the skip-cheap-then-top-2 dedup is skipped (trade every
    qualifying ETF on every UL).  extra_drops removes specific symbols from the
    universe entirely (used for single-stock paring).
    If prune_all=True, broad/BE are ALSO pruned (default behavior exempts them)."""
    broad_etfs, _, _ = filter_master(master, ["1_BROAD"])
    be_mask  = master["binary_event"].astype(str).str.upper().eq("YES")
    be_etfs, _, _ = filter_master(master[be_mask], list(master["class"].unique()))
    pc_etfs, _, _ = filter_master(master, ["2_SECTOR_pair", "4_CRYPTO", "force_include"])
    extra_force = [s for s in FORCE_INCLUDE if s in master["etf"].values and s not in pc_etfs]
    pc_etfs = sorted(set(pc_etfs) | set(extra_force))
    filter_etfs = [s for s in pc_etfs if s not in set(be_etfs)]
    unfiltered_etfs = sorted(set(broad_etfs) | set(be_etfs))
    unfiltered_etfs, _ = gate_by_sigma_coverage(unfiltered_etfs, master)
    filter_etfs, _     = gate_by_sigma_coverage(filter_etfs, master)
    if extra_drops:
        unfiltered_etfs = [s for s in unfiltered_etfs if s not in extra_drops]
        filter_etfs     = [s for s in filter_etfs     if s not in extra_drops]

    all_etfs = sorted(set(unfiltered_etfs) | set(filter_etfs))
    universe = {s: INSTRUMENTS[s] for s in all_etfs if s in INSTRUMENTS}
    sigmas = load_sigma_master()
    ps_filters = {}
    if k:
        for s, info in universe.items():
            ul = info["underlying"]
            sig = sigmas.get(ul)
            if sig is None: continue
            thr = sig * k
            if sigma_mode == "expanding":
                from scripts.pit_sigma import Threshold
                thr = Threshold(ul, k)
            ps_filters[s] = (ul, thr, True) if info.get("inverse") else (ul, thr)
    cfg = StrategyConfig(
        symbols=list(universe.keys()),
        instrument_gap_filters={s: round(i["leverage"] * gap_threshold, 4)
                                for s, i in universe.items()},
        prior_session_filters=ps_filters,
        day_of_week_exclusions={},
        direction_filters={s: d for s, d in DIRECTION_FILTERS.items() if s in universe},
        use_rtg_scaling=False, rtg_gap_exclusion=False,
        rtg_gap_exclusion_threshold=0.0, rtg_gap_exclusion_symbols=tuple(),
        min_increment_pct=0.0, min_profit_pct=0.0,
        start_date="2020-01-01", initial_equity=100_000.0, daily_risk_pct=0.20,
        eod_exit_hour=eod_exit_hour, eod_exit_minute=eod_exit_minute,
        exit_ratio_tp1=1.00, exit_ratio_tp2=0.00, exit_ratio_tp3=0.00,
        tp1_target_multiple=tp1_target_multiple,
        orb_minutes=orb_minutes,
        entry_at_boundary=entry_at_boundary,
        stop_from_entry=stop_from_entry,
        stop_orb_distance=stop_orb_distance,
        stop_entry_distance=stop_entry_distance,
        same_bar_fill_risk=same_bar_fill_risk,
        entry_slippage_bps=entry_slippage_bps,
        stop_slippage_bps=stop_slippage_bps,
        eod_slippage_bps=eod_slippage_bps,
        slippage_model=slippage_model,
        slippage_notional=slippage_notional,
        max_cost_to_reward=max_cost_to_reward,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        raw = run_backtest(cfg)
    if raw.empty: return raw
    # With the participation slippage model on, spread crossing and market
    # impact are already in the modelled fills; apply_trade_costs' Corwin-Schultz
    # and Amihud terms would charge them again, from whole-sample constants.
    net = P.apply_trade_costs(raw, spread_and_impact=slippage_model is None)
    net["date"] = pd.to_datetime(net["date"])

    if adv_mode != "none":
        _adv_filter = (apply_pit_dollar_adv_filter if adv_mode == "pit"
                       else apply_static_dollar_adv_filter)
        net, kept_syms, _, _ = _adv_filter(
            net, universe=all_etfs, min_dollar_adv=5_000_000.0,
            exempt=(set(be_etfs) | FORCE_INCLUDE) if adv_exempt_binary_event
                    else set(FORCE_INCLUDE))

    unfilt_set = set(unfiltered_etfs)
    filt_set   = set(filter_etfs) - unfilt_set
    unfilt_trades = net[net["symbol"].isin(unfilt_set)].copy()
    filt_trades   = net[net["symbol"].isin(filt_set)].copy()

    if prune_all and prune:
        prices = {s: load_daily_close(s) for s in sorted(net["symbol"].unique())}
        prices = {s: ser for s, ser in prices.items() if not ser.empty}
        combined = skip_cheap_then_top2_when_3plus(net, prices, top_n=top_n_per_ul) if not net.empty else net.copy()
    else:
        if prune:
            prices = {s: load_daily_close(s) for s in sorted(filt_trades["symbol"].unique())}
            prices = {s: ser for s, ser in prices.items() if not ser.empty}
            filt_pruned = skip_cheap_then_top2_when_3plus(filt_trades, prices, top_n=top_n_per_ul) if not filt_trades.empty else filt_trades
        else:
            filt_pruned = filt_trades
        combined = pd.concat([unfilt_trades, filt_pruned], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    return combined


# ── Sized with cap ────────────────────────────────────────────────────────────
def size_3class_with_cap(trades: pd.DataFrame, master: pd.DataFrame,
                          weights: dict, cap_units: float | None) -> pd.DataFrame:
    sized = apply_3class_regime_sizing(trades, master, weights=weights)
    if cap_units is not None and cap_units > 0 and not sized.empty:
        day_mult = sized.groupby("date_norm")["mult"].sum().rename("day_mult_sum")
        sized = sized.merge(day_mult, on="date_norm")
        sized["cap_factor"] = (cap_units / sized["day_mult_sum"]).clip(upper=1.0)
        sized["sized_pnl"] = sized["dollar_pnl"] * sized["mult"] * sized["cap_factor"]
    return sized


DEFAULT_WEIGHTS_3x3 = {
    ("C1","quiet"):  3.0, ("C1","active"):3.0, ("C1","flood"): 3.0, ("C1","cluster"):1.5,
    ("C2","quiet"):  2.0, ("C2","active"):2.0, ("C2","flood"): 2.0, ("C2","cluster"):1.0,
    ("C3","quiet"):  1.0, ("C3","active"):1.5, ("C3","flood"): 2.0, ("C3","cluster"):0.75,
    "purity_skip": 0.0,
}
CAPS = {"100%": 10.0, "200%": 20.0, "400%": 40.0, "uncapped": None}
K_VALUES = [0.75, 1.00, 1.25, 1.50]


# 12-bucket grid. Active/flood ranges allowed to LIFT vs quiet (data shows
# per-trade edge scales monotonically with n_uls).
WEIGHT_GRID = {
    ("C1","quiet"):   [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    ("C1","active"):  [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    ("C1","flood"):   [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    ("C1","cluster"): [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
    ("C2","quiet"):   [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
    ("C2","active"):  [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ("C2","flood"):   [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    ("C2","cluster"): [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    ("C3","quiet"):   [0.0, 0.25, 0.5, 0.75, 1.0],
    ("C3","active"):  [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5],
    ("C3","flood"):   [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    ("C3","cluster"): [0.0, 0.25, 0.5, 0.75, 1.0],
}


def phase1_sigma_sweep(master):
    print("\n" + "=" * 100)
    print("PHASE 1 — SIGMA (PS k) SWEEP × CAPS  (default 3-class weights)")
    print("=" * 100)
    results = []
    trades_by_k = {}
    for k in K_VALUES:
        print(f"\n[k={k:.2f}] running backtest...")
        trades = run_v1_at_k(k, master)
        trades_by_k[k] = trades
        print(f"  {len(trades)} post-rule trades")
        for cap_name, cap in CAPS.items():
            sized = size_3class_with_cap(trades, master, DEFAULT_WEIGHTS_3x3, cap)
            s = stats(daily_series(sized))
            s["k"] = k; s["cap"] = cap_name
            s["trades"] = (sized["mult"] > 0).sum()
            results.append(s)

    df = pd.DataFrame(results)
    print(f"\n{'k':>5} {'cap':>10} {'trades':>7} {'sharpe':>7} {'maxdd':>8} {'calmar':>7} {'net_pnl':>11}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['k']:>5.2f} {r['cap']:>10} {r['trades']:>7} {r['sharpe']:>7.3f} "
              f"{r['max_dd']*100:>7.2f}% {r['calmar']:>7.2f} ${r['net_pnl']:>10,.0f}")
    print(f"\nBest k per cap (by Sharpe):")
    for cap_name in CAPS:
        sub = df[df["cap"] == cap_name].sort_values("sharpe", ascending=False).iloc[0]
        print(f"  {cap_name:<10} → k={sub['k']:.2f}  Sharpe={sub['sharpe']:.3f}  Calmar={sub['calmar']:.2f}  P&L=${sub['net_pnl']:,.0f}")
    return df, trades_by_k


def phase2_weight_descent(master, trades, cap_name, cap, label,
                            optimize_for="sharpe", max_passes=6):
    w = dict(DEFAULT_WEIGHTS_3x3)
    s = stats(daily_series(size_3class_with_cap(trades, master, w, cap)))
    best_metric = s[optimize_for]
    changed = True; iteration = 0
    while changed and iteration < max_passes:
        changed = False; iteration += 1
        for bucket, grid in WEIGHT_GRID.items():
            cur = w[bucket]; best_val = cur; local_best = best_metric
            for v in grid:
                if v == cur: continue
                test = dict(w); test[bucket] = v
                stt = stats(daily_series(size_3class_with_cap(trades, master, test, cap)))
                if stt[optimize_for] > local_best:
                    local_best = stt[optimize_for]; best_val = v
            if best_val != cur:
                w[bucket] = best_val; best_metric = local_best; changed = True
    final = stats(daily_series(size_3class_with_cap(trades, master, w, cap)))
    print(f"\n[{label}] cap={cap_name}  converged after {iteration} passes")
    nice = {f"{k[0]}_{k[1]}": v for k, v in w.items() if isinstance(k, tuple)}
    nice["purity_skip"] = w["purity_skip"]
    print(f"  weights: {nice}")
    print(f"  Sharpe={final['sharpe']:.3f}  MaxDD={final['max_dd']*100:.2f}%  "
          f"Calmar={final['calmar']:.2f}  P&L=${final['net_pnl']:,.0f}")
    return w, final


if __name__ == "__main__":
    master = pd.read_csv(MASTER_CSV)
    master = add_force_includes_to_master(master)

    df_p1, trades_by_k = phase1_sigma_sweep(master)

    print("\n" + "=" * 100)
    print("PHASE 2 — 3×3 WEIGHT COORDINATE DESCENT (at best k per cap)")
    print("=" * 100)
    best_per_cap = {}
    for cap_name, cap in CAPS.items():
        sub = df_p1[df_p1["cap"] == cap_name].sort_values("sharpe", ascending=False).iloc[0]
        k_best = sub["k"]
        trades = trades_by_k[k_best]
        w, s = phase2_weight_descent(master, trades, cap_name, cap, f"k={k_best:.2f}")
        best_per_cap[cap_name] = dict(k=k_best, weights=w, stats=s)

    print("\n" + "=" * 100)
    print("FINAL OPTIMIZED CONFIGS PER CAP (3-class regime)")
    print("=" * 100)
    for cap_name, info in best_per_cap.items():
        s = info["stats"]
        print(f"\n--- Cap = {cap_name} ---")
        print(f"  PS k = {info['k']:.2f}")
        nice = {f"{k[0]}_{k[1]}": v for k, v in info["weights"].items() if isinstance(k, tuple)}
        nice["purity_skip"] = info["weights"]["purity_skip"]
        for label, val in nice.items():
            print(f"    {label:<14} = {val}")
        print(f"  Sharpe={s['sharpe']:.3f}  MaxDD={s['max_dd']*100:.2f}%  "
              f"Calmar={s['calmar']:.2f}  P&L=${s['net_pnl']:,.0f}")
