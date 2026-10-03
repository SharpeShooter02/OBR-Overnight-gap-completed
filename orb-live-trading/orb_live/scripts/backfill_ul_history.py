"""One-time: extend the live underlying daily store back to full history.

Point-in-time σ (V48 §8) is std(|daily return|) over EVERY close before the
session; the live store starts 2021-06 (2024-05 for some), so σ from it would
not match the backtest's. This prepends rows older than each live file's first
date from the backtest's daily cache — the same bars the backtest σ uses —
leaving every existing live row untouched.

Reports the close mismatch on the overlap first; refuses a symbol whose
overlapping closes differ by more than --max-dev (price-basis mismatch).

    python -m orb_live.scripts.backfill_ul_history --dry-run
    python -m orb_live.scripts.backfill_ul_history
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "underlyings"
BACKTEST_DIR = Path(str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'BacktestingGaps/orb_event_study/cache/daily'))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=BACKTEST_DIR)
    ap.add_argument("--max-dev", type=float, default=0.02)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for p in sorted(LIVE_DIR.glob("*.parquet")):
        ul = p.stem
        src = a.source / p.name
        if not src.exists():
            print(f"{ul:6} no backtest history — skipped")
            continue
        live = pd.read_parquet(p)
        live["date"] = pd.to_datetime(live["date"]).dt.normalize()
        bt = pd.read_parquet(src)
        bt["date"] = pd.to_datetime(bt["date"]).dt.tz_localize(None).dt.normalize() \
            if getattr(pd.to_datetime(bt["date"]).dt, "tz", None) else pd.to_datetime(bt["date"]).dt.normalize()
        ov = live.merge(bt[["date", "close"]], on="date", suffixes=("", "_bt"))
        dev = (ov["close"] / ov["close_bt"] - 1).abs()
        med, p95 = (float(dev.median()), float(dev.quantile(0.95))) if len(ov) else (float("nan"),) * 2
        older = bt[bt["date"] < live["date"].min()]
        cols = [c for c in live.columns if c in older.columns]
        verdict = "ok"
        if not len(ov) or med > a.max_dev:
            verdict = "REFUSED (basis mismatch)"
        print(f"{ul:6} live {len(live):5} from {live['date'].min().date()}  +{len(older):5} older rows  "
              f"overlap {len(ov):5} |close dev| median {med:.4%} p95 {p95:.4%}  {verdict}")
        if a.dry_run or verdict != "ok" or older.empty:
            continue
        # Dividend-adjusted vs unadjusted sources differ by a near-constant
        # ratio; rescale the prepended prices so the join adds no fake return.
        ratio = float((ov["close"] / ov["close_bt"]).median())
        older = older[cols].copy()
        for c in ("open", "high", "low", "close"):
            if c in older.columns:
                older[c] = older[c] * ratio
        out = pd.concat([older, live], ignore_index=True).drop_duplicates("date").sort_values("date")
        out.to_parquet(p, index=False)


if __name__ == "__main__":
    main()
