"""
Final config — CAUSAL VERSION.

Fixes look-ahead bias in regime detection and cap-factor scaling.

Previously: n_uls and day_mult_sum were computed from POST-FIRING trades (only
trades whose ORB breakout succeeded at 9:31). This is a forward-looking signal:
at 9:30 we don't know which breakouts will fire.

Now:
  1. Compute causal candidate set per session at 9:30 (gap threshold + PS filter
     + direction filter + skip-cheap-top-2)
  2. Use that for n_uls (regime tier) and for expected_day_mult (cap factor)
  3. Apply the regime-determined multiplier and cap_factor to the trades that
     actually fired

This is causally honest — every input is known at 9:30 ET.
"""
import sys, io, contextlib
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

from scripts.optimize_v1_3class import run_v1_at_k, stats, daily_series
from scripts.skip_cheap_by_class import (
    add_force_includes_to_master, filter_master, gate_by_sigma_coverage,
    overnight_gap_from_daily, FORCE_INCLUDE, DIRECTION_FILTERS,
    CLASS_1_SYMS, CLASS_2_SYMS,
)
from scripts.sigma_master import load_sigma_master
from orb_event_study.config import INSTRUMENTS

_DAILY_CACHE = Path("orb_event_study/cache/daily")
WEIGHTS = {
    ("C1","quiet"):   2.5, ("C1","active"): 4.0, ("C1","flood"): 1.0, ("C1","cluster"): 3.0,
    ("C2","quiet"):   0.5, ("C2","active"): 1.0, ("C2","flood"): 2.5, ("C2","cluster"): 1.0,
    ("C3","quiet"):   0.0, ("C3","active"): 2.0, ("C3","flood"): 2.5, ("C3","cluster"): 1.0,
}
CAP_UNITS = 20.0   # 200% cap
K = 1.25
ACTIVE_MIN = 5
FLOOD_MIN = 8
CLUSTER_THR = 3


def classify(sym):
    if sym in CLASS_1_SYMS: return "C1"
    if sym in CLASS_2_SYMS: return "C2"
    return "C3"


_CRYPTO_HOURLY_CACHE = Path("orb_event_study/cache/crypto_hourly_yf")
_CRYPTO_UNDERLYINGS  = frozenset({"BTC", "ETH", "SOL", "XRP"})


def _load_crypto_close_at_1600et(ul: str) -> pd.Series:
    """For crypto ULs, derive per-date 4pm-ET closes from hourly bars.
    Matches the ETF calendar so PS-filter prior-return math is consistent
    with the overnight gap computation.
    """
    p = _CRYPTO_HOURLY_CACHE / f"{ul}_1h.parquet"
    if not p.exists(): return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns: return pd.Series(dtype=float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    et = df.index.tz_convert("America/New_York")
    df = df.copy()
    df["et_date"] = et.normalize()
    df["et_hour"] = et.hour
    bars_1600 = df[df["et_hour"] == 16][["et_date", "open"]]
    bars_1600 = bars_1600.drop_duplicates(subset="et_date", keep="last")
    bars_1600 = bars_1600.set_index("et_date")["open"]
    bars_1600.index = pd.to_datetime(bars_1600.index).tz_localize(None).normalize()
    bars_1600.name = "close"
    return bars_1600.sort_index()


def load_ul_close(ul):
    if ul in _CRYPTO_UNDERLYINGS:
        return _load_crypto_close_at_1600et(ul)
    p = _DAILY_CACHE / f"{ul}.parquet"
    if not p.exists(): return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()["close"]


def load_sym_close(sym):
    """Daily close from intraday cache (most reliable for ETF pricing)."""
    sym_dir = Path("orb_event_study/cache/intraday") / sym
    if not sym_dir.exists(): return pd.Series(dtype=float)
    frames = []
    for pq in sorted(sym_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
            if not df.empty: frames.append(df)
        except Exception: continue
    if not frames: return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)
    out = df.groupby("date")["close"].last()
    # The cache is forward-adjusted, so ranking siblings on it is decided by
    # splits AFTER the trade date -- see scripts/as_traded.py (register V43).
    from scripts import as_traded as _AT
    if _AT.USE_AS_TRADED:
        out = _AT.to_as_traded(sym, out)
    return out


def build_universe(master):
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
    all_etfs = sorted(set(unfiltered_etfs) | set(filter_etfs))
    return all_etfs


from scripts.margin_model import expected_margin, margin_units


def compute_causal_candidates(master, k, prune_all=False, top_n: int = 2,
                              return_detail: bool = False):
    """Return dict: date (Timestamp) -> list of candidate symbols at 9:30.
    Causal: uses only data known by market open.
    If prune_all=True, broad/BE are also pruned (no exception).
    top_n: max ETFs kept per (date, underlying); defaults to 2 (locked v1)."""
    sigmas = load_sigma_master()
    all_etfs = build_universe(master)
    uls = sorted({INSTRUMENTS[s]["underlying"] for s in all_etfs if s in INSTRUMENTS})

    # Per UL: signed overnight gap and which dates pass gap-threshold + PS
    ul_qualifying = {}
    for ul in uls:
        gap = overnight_gap_from_daily(ul)
        if gap.empty: continue
        gap.index = pd.to_datetime(gap.index).normalize()
        # Instrument-level threshold = leverage * 2%, but UL-level is always 2%
        # since |ETF gap| = |UL gap| × leverage and threshold = leverage × 2%
        qual = gap[gap.abs() >= 0.02]
        if k > 0 and ul in sigmas:
            thr = sigmas[ul] * k
            ul_close = load_ul_close(ul)
            if not ul_close.empty:
                ul_close.index = pd.to_datetime(ul_close.index).normalize()
                keep = []
                for d in qual.index:
                    prior = ul_close[ul_close.index < d].tail(2)
                    if len(prior) < 2:
                        keep.append(d); continue
                    c1 = prior.iloc[-1]; c2 = prior.iloc[-2]
                    if c2 <= 0:
                        keep.append(d); continue
                    prior_ret = (c1 - c2) / c2
                    gap_sign = 1 if qual.loc[d] > 0 else -1
                    dir_adj = prior_ret * gap_sign
                    if dir_adj <= thr:
                        keep.append(d)
                qual = qual[qual.index.isin(keep)]
        ul_qualifying[ul] = qual

    # Per instrument: direction filter
    cand_by_date = defaultdict(list)
    for sym in all_etfs:
        if sym not in INSTRUMENTS: continue
        info = INSTRUMENTS[sym]
        ul = info["underlying"]; is_inverse = info.get("inverse", False)
        qual = ul_qualifying.get(ul)
        if qual is None or qual.empty: continue
        # ETF's gap direction = ul_gap sign × (-1 if inverse else +1)
        for d, ul_gap in qual.items():
            ul_sign = 1 if ul_gap > 0 else -1
            etf_dir = -ul_sign if is_inverse else ul_sign
            if sym in DIRECTION_FILTERS and DIRECTION_FILTERS[sym] != etf_dir:
                continue
            cand_by_date[d].append({"symbol": sym, "underlying": ul, "etf_dir": etf_dir})

    # Skip-cheap-top-2 per (date, UL) using prior-day closes
    # Pre-load price series for filter_etfs only — broad/BE skip prune entirely
    broad_etfs, _, _ = filter_master(master, ["1_BROAD"])
    be_mask = master["binary_event"].astype(str).str.upper().eq("YES")
    be_etfs, _, _ = filter_master(master[be_mask], list(master["class"].unique()))
    no_prune = set() if prune_all else (set(broad_etfs) | set(be_etfs))

    sym_close = {}   # cached
    def get_price(sym, date):
        if sym not in sym_close:
            sym_close[sym] = load_sym_close(sym)
        ser = sym_close[sym]
        if ser.empty: return 0.0
        hist = ser[ser.index < date]
        return float(hist.iloc[-1]) if not hist.empty else 0.0

    final, detail = {}, {}
    for d, cands in cand_by_date.items():
        keep = []
        # Group by UL
        by_ul = defaultdict(list)
        for c in cands:
            by_ul[c["underlying"]].append(c)
        for ul, grp in by_ul.items():
            # If any candidate is in no_prune set, skip pruning for that UL
            if any(c["symbol"] in no_prune for c in grp):
                keep.extend(c["symbol"] for c in grp)
                continue
            if len(grp) == 1:
                keep.append(grp[0]["symbol"]); continue
            for c in grp:
                c["price"] = get_price(c["symbol"], d)
            grp.sort(key=lambda x: -x["price"])
            keep.extend(c["symbol"] for c in grp[:max(1, top_n)])
        final[d] = keep
        detail[d] = [{"symbol": c["symbol"], "underlying": c["underlying"],
                      "etf_dir": c["etf_dir"]}
                     for c in cands if c["symbol"] in set(keep)]
    if return_detail:
        return detail
    return final


def assign_regime(n_uls, n_crypto_uls):
    if n_uls >= FLOOD_MIN: return "flood"
    if n_uls >= ACTIVE_MIN: return "active"
    return "quiet"


def causal_size(trades, candidates, master, weights, cap,
                margin_budget_units=None, candidate_detail=None):
    """Size trades using causal regime + causal cap factor.

    margin_budget_units — if set, positions are additionally scaled so the
        candidate set's initial-margin requirement fits the budget (in the
        same units as `cap`; 10.0 = 100% of equity). Requires
        candidate_detail for per-candidate trade direction. Leave None to
        reproduce pre-margin-model behaviour.
    """
    if trades.empty: return trades
    crypto_uls = set(master[master["class"] == "4_CRYPTO"]["underlying"].unique())
    lev_by_sym = dict(zip(master["etf"], master["leverage"]))

    # Per-date causal stats
    date_stats = {}
    for d, cand_syms in candidates.items():
        cand_uls = {INSTRUMENTS[s]["underlying"] for s in cand_syms if s in INSTRUMENTS}
        cand_crypto_uls = cand_uls & crypto_uls
        n_uls = len(cand_uls)
        n_crypto_uls = len(cand_crypto_uls)
        regime = assign_regime(n_uls, n_crypto_uls)
        # Expected day mult = sum of weights for candidates given regime
        exp_mult = 0.0
        for s in cand_syms:
            cls = classify(s)
            exp_mult += weights.get((cls, regime), 1.0)
        cap_factor = min(1.0, cap / exp_mult) if exp_mult > 0 else 0.0

        # Broker margin constraint. FINRA 4210 scales the requirement by fund
        # leverage, so a 3x short costs 90% of notional; on busy days that
        # binds well before the 200% gross cap does.
        exp_margin = 0.0
        if margin_budget_units is not None and candidate_detail is not None:
            exp_margin = expected_margin(
                candidate_detail.get(d, []), weights, classify, regime, lev_by_sym)
            if exp_margin > 0:
                cap_factor = min(cap_factor, margin_budget_units / exp_margin)

        date_stats[d] = {"n_uls": n_uls, "n_crypto_uls": n_crypto_uls,
                          "regime": regime, "exp_mult": exp_mult,
                          "exp_margin": exp_margin,
                          "cap_factor": cap_factor, "total_n": len(cand_syms)}

    t = trades.copy()
    t["date_norm"] = pd.to_datetime(t["date"]).dt.tz_localize(None).dt.normalize()
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    t["v1_class"] = t["symbol"].map(classify)
    t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
    t["regime"]      = t["date_norm"].map(lambda d: date_stats.get(d, {}).get("regime", "quiet"))
    t["cap_factor"]  = t["date_norm"].map(lambda d: date_stats.get(d, {}).get("cap_factor", 1.0))
    t["n_uls_causal"]= t["date_norm"].map(lambda d: date_stats.get(d, {}).get("n_uls", 0))
    t["total_n_cand"]= t["date_norm"].map(lambda d: date_stats.get(d, {}).get("total_n", 0))

    def mult(row):
        return weights.get((row["v1_class"], row["regime"]), 0.0)
    t["mult"] = t.apply(mult, axis=1)
    t["sized_pnl"] = t["dollar_pnl"] * t["mult"] * t["cap_factor"]
    return t, date_stats


def main():
    master = pd.read_csv(Path("master_universe.csv"))
    master = add_force_includes_to_master(master)

    print("Building causal candidate set per session (this takes a bit)...", flush=True)
    candidates = compute_causal_candidates(master, K)
    print(f"  {len(candidates)} sessions with at least one candidate", flush=True)

    print(f"\nRunning backtest at k={K}...", flush=True)
    trades = run_v1_at_k(K, master)
    print(f"  {len(trades)} actual fired trades", flush=True)

    sized, date_stats = causal_size(trades, candidates, master, WEIGHTS, CAP_UNITS)
    s = stats(daily_series(sized))

    print(f"\n{'='*70}")
    print(f"FINAL CONFIG (CAUSAL) — 200% cap, k=1.25")
    print(f"{'='*70}")
    print(f"Sharpe = {s['sharpe']:.3f}")
    print(f"MaxDD  = {s['max_dd']*100:.2f}%")
    print(f"Calmar = {s['calmar']:.2f}")
    print(f"Net P&L = ${s['net_pnl']:,.0f}")
    print(f"Ann ret = {s.get('ann_ret', 0)*100:.2f}%")

    # Compare to look-ahead version: regime distribution
    print(f"\nCausal regime distribution:")
    reg_counts = pd.Series([v["regime"] for v in date_stats.values()]).value_counts()
    print(reg_counts.to_string())

    print(f"\nCausal n_uls distribution (sessions w/ candidates):")
    n_uls_counts = pd.Series([v["n_uls"] for v in date_stats.values()]).value_counts().sort_index()
    print(n_uls_counts.to_string())

    print(f"\nMean cap_factor: {pd.Series([v['cap_factor'] for v in date_stats.values()]).mean():.3f}")
    print(f"Days where cap binds (cap_factor<1.0): "
          f"{sum(1 for v in date_stats.values() if v['cap_factor'] < 1.0)}/{len(date_stats)}")

    # Bucket attribution
    print(f"\nBucket attribution:")
    attr = sized.groupby(["v1_class","regime"]).agg(
        n=("symbol","count"),
        raw=("dollar_pnl","sum"),
        sized=("sized_pnl","sum"),
        mean=("dollar_pnl","mean"),
        wr=("pnl_pct", lambda x: (x>0).mean()*100),
    ).reset_index().sort_values(["v1_class","regime"])
    for _, r in attr.iterrows():
        print(f"  {r['v1_class']} {r['regime']:<8} N={int(r['n']):>4}  "
              f"raw=${r['raw']:>+7,.0f}  sized=${r['sized']:>+7,.0f}  "
              f"$/tr=${r['mean']:>+6.2f}  WR={r['wr']:.1f}%")

    # Year by year
    print(f"\nYear-by-year:")
    sized["year"] = pd.to_datetime(sized["date"]).dt.year
    yr = sized.groupby("year").agg(
        n=("symbol","count"),
        pnl=("sized_pnl","sum"),
        wr=("pnl_pct", lambda x: (x>0).mean()*100),
    )
    for y, row in yr.iterrows():
        print(f"  {y}: {int(row['n']):>4} trades  ${row['pnl']:>+8,.0f}  WR {row['wr']:.1f}%")


if __name__ == "__main__":
    main()
