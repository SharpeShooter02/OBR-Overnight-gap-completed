"""Recover the price the tape actually printed. Register V43.

The rebuilt intraday cache is FORWARD-ADJUSTED: IBKR expresses history on
today's share basis, so a fund that reverse-split 1:10 since 2020 has its 2020
prices multiplied by 10. Every ratio the strategy computes -- gap, opening
range, pnl_pct -- is unaffected, because the factor cancels.

Two decisions read the LEVEL, and for those it does not cancel:

    select_by="price"                  keep the most expensive sibling per UL
    skip_cheap_then_top2_when_3plus    drop the cheaper of two siblings

Both then rank siblings whose adjustment factors differ, which means the
ranking is decided by how much each fund split AFTER the trade date. Measured
over 28,023 sibling-days, the winner differs from the as-traded winner on
54.2% of them -- 88.9% for IWM. That is look-ahead of the V34 shape.

Live has never had this problem: it ranks on `prior_close` straight from the
broker, which is the as-traded print. So this is a backtest/live divergence,
not a design choice, and the fix restores the locked v1 rule rather than
changing it.

    as_traded(t) = cached(t) * cum_factor(t)
    cum_factor(t) = product of every split ratio dated AFTER t

INVARIANT UNDER FUTURE SPLITS, which is the property the raw cached price
lacks: when a new split lands, IBKR restates history downward by the ratio and
the split table gains the same ratio. The product does not move.

WHAT THIS TRUSTS. The split table (Yahoo, cached by intraday_basis_reconcile).
A missing or wrong split silently mis-ranks that symbol, so `validate()` checks
the reconstruction lands in a plausible traded range and that any jump in the
series coincides with a known split date.

    python scripts/as_traded.py            # validate every active symbol
    python scripts/as_traded.py --symbols SOXS,TQQQ
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("orb_event_study/cache/intraday")
SPLITS_JSON = Path("analysis/out/splits_history.json")

PLAUSIBLE_LO, PLAUSIBLE_HI = 0.5, 5_000.0

#: Global basis switch consulted by the two price-ranking loaders
#: (run_final_config_causal.load_sym_close and
#: skip_cheap_by_class.load_daily_close). False reproduces every published
#: figure; True ranks siblings on the price the tape printed, which is what
#: live does. A module global rather than a parameter because both loaders are
#: called several frames below the caller that wants to choose.
USE_AS_TRADED = False

_splits: dict | None = None
_cache: dict[str, pd.Series] = {}


def _split_table() -> dict:
    global _splits
    if _splits is None:
        _splits = json.loads(SPLITS_JSON.read_text(encoding="utf-8")) \
            if SPLITS_JSON.exists() else {}
    return _splits


def split_dates(sym: str) -> dict[pd.Timestamp, float]:
    return {pd.Timestamp(k): float(v)
            for k, v in _split_table().get(sym, {}).items()
            if not k.startswith("__") and float(v) > 0}


def cum_factor(sym: str, index: pd.DatetimeIndex) -> pd.Series:
    """Product of every split ratio dated strictly after each timestamp."""
    idx = pd.DatetimeIndex(index)
    naive = idx.tz_localize(None) if idx.tz is not None else idx
    out = np.ones(len(naive))
    for d, r in split_dates(sym).items():
        out[naive < d] *= r
    return pd.Series(out, index=index)


def to_as_traded(sym: str, adjusted: pd.Series) -> pd.Series:
    """Undo IBKR's forward split adjustment on a price series."""
    if adjusted.empty:
        return adjusted
    return adjusted * cum_factor(sym, adjusted.index).values


def daily_close(sym: str, as_traded: bool = True) -> pd.Series:
    """Session closes from the intraday cache, on the requested basis."""
    key = f"{sym}:{as_traded}"
    if key in _cache:
        return _cache[key]
    d = CACHE / sym
    if not d.exists():
        _cache[key] = pd.Series(dtype=float)
        return _cache[key]
    frames = []
    for pq in sorted(d.glob("*.parquet")):
        try:
            t = pd.read_parquet(pq)
        except Exception:
            continue
        if t.empty or "timestamp" not in t.columns:
            continue
        frames.append(t[["timestamp", "close"]])
    if not frames:
        _cache[key] = pd.Series(dtype=float)
        return _cache[key]
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[ts.notna()].copy()
    df["date"] = ts[ts.notna()].dt.normalize().dt.tz_localize(None)
    s = df.groupby("date")["close"].last().sort_index()
    if as_traded:
        s = to_as_traded(sym, s)
    _cache[key] = s
    return s


def validate(sym: str) -> dict | None:
    """Is the reconstruction believable? Range, and jumps only at splits."""
    adj = daily_close(sym, as_traded=False)
    raw = daily_close(sym, as_traded=True)
    if len(raw) < 60:
        return None
    sd = sorted(split_dates(sym))
    r = (raw / raw.shift(1)).dropna()
    # a jump the split table does not explain
    unexplained, worst, worst_d = 0, 1.0, None
    for d, v in r.items():
        j = max(v, 1 / v) if v > 0 else np.inf
        if j <= 1.6:
            continue
        near = any(abs((d - s).days) <= 3 for s in sd)
        if not near:
            unexplained += 1
            if j > worst:
                worst, worst_d = j, d
    lo, hi = float(raw.min()), float(raw.max())
    return {"symbol": sym, "sessions": int(len(raw)),
            "as_traded_min": lo, "as_traded_max": hi,
            "adjusted_first": float(adj.iloc[0]), "as_traded_first": float(raw.iloc[0]),
            "n_splits": len(sd),
            "unexplained_jumps": unexplained,
            "worst_unexplained": float(worst) if worst_d is not None else None,
            "worst_date": str(worst_d.date()) if worst_d is not None else None,
            "in_range": bool(PLAUSIBLE_LO <= lo and hi <= PLAUSIBLE_HI)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, ".")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()

    if a.symbols:
        syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    else:
        from scripts.skip_cheap_by_class import filter_master
        m = pd.read_csv("master_universe.csv")
        syms, _, _ = filter_master(m, sorted(m["class"].unique()))
        syms = sorted(syms)

    rows = [r for s in syms if (r := validate(s)) is not None]
    df = pd.DataFrame(rows)

    print("=" * 96)
    print("AS-TRADED RECONSTRUCTION -- validation")
    print("=" * 96)
    print(f"  symbols                     {len(df)}")
    print(f"  outside plausible range     {int((~df['in_range']).sum())}")
    print(f"  with unexplained jumps      {int((df['unexplained_jumps'] > 0).sum())}")

    bad = df[(~df["in_range"]) | (df["unexplained_jumps"] > 0)]
    if len(bad):
        print(f"\n  {'symbol':<8}{'sessions':>9}{'min':>11}{'max':>12}"
              f"{'splits':>8}{'unexpl':>8}{'worst':>9}  date")
        for _, r in bad.sort_values("unexplained_jumps", ascending=False).iterrows():
            print(f"  {r['symbol']:<8}{int(r['sessions']):>9}{r['as_traded_min']:>11.2f}"
                  f"{r['as_traded_max']:>12.2f}{int(r['n_splits']):>8}"
                  f"{int(r['unexplained_jumps']):>8}"
                  f"{(r['worst_unexplained'] or 0):>9.1f}  {r['worst_date'] or ''}")

    print(f"\n  {'symbol':<8}{'adjusted first':>16}{'as-traded first':>17}{'ratio':>10}")
    for _, r in df.reindex((df["adjusted_first"] / df["as_traded_first"])
                           .sort_values().index).head(8).iterrows():
        print(f"  {r['symbol']:<8}{r['adjusted_first']:>16,.2f}{r['as_traded_first']:>17,.2f}"
              f"{r['adjusted_first'] / r['as_traded_first']:>10.4g}")

    out = Path("analysis/out/as_traded_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
