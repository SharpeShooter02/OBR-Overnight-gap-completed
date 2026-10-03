"""Rebuild the public chart from aggregated backtest P&L, never live records."""
from pathlib import Path
import csv
from datetime import datetime
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "docs/charts"


def main():
    metadata = json.loads((CHARTS / "pnl_provenance.json").read_text())
    with (CHARTS / "pnl_daily.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    dates = [datetime.fromisoformat(row["date"]) for row in rows]
    pnl = [float(row["backtest_pnl_usd"]) for row in rows]
    split = datetime.fromisoformat(metadata["holdout_start"])
    assert dates == sorted(set(dates))
    assert all(math.isfinite(x) for x in pnl)
    assert math.isclose(sum(pnl), metadata["pnl_usd"]["full"], abs_tol=1e-6)
    totals = []
    running = 0.0
    for value in pnl:
        running += value
        totals.append(running)
    before = [i for i, d in enumerate(dates) if d < split]
    after = [i for i, d in enumerate(dates) if d >= split]
    assert before and after
    assert math.isclose(sum(pnl[i] for i in before), metadata["pnl_usd"]["in_sample"], abs_tol=1e-6)
    assert math.isclose(sum(pnl[i] for i in after), metadata["pnl_usd"]["out_of_sample"], abs_tol=1e-6)

    blue, orange = "#2463a6", "#d16b18"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.hashsalt": "orb-v49-public-pnl"})
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.6), gridspec_kw={"height_ratios": [1.45, 1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("OBR Overnight gap completed", x=0.08, y=0.97, ha="left", fontsize=21, fontweight="bold", color="#172a40")
    fig.text(0.08, 0.921, "V49 archived backtest | Fixed $1,000 notional per unit | Modelled costs included", color="#45566b", fontsize=11)
    ax, zoom = axes
    for a in axes:
        a.set_facecolor("white")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.20)
        a.axhline(0, color="#8e9bab", lw=0.8)
        a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
        a.set_ylabel("Cumulative backtest P&L (USD)")
        a.set_axisbelow(True)
    ax.axvspan(dates[0], split, color=blue, alpha=0.08)
    ax.axvspan(split, dates[-1], color=orange, alpha=0.13)
    ax.plot([dates[i] for i in before], [totals[i] for i in before], color=blue, lw=2.0, label="In sample")
    bridge = [before[-1]] + after
    ax.plot([dates[i] for i in bridge], [totals[i] for i in bridge], color=orange, lw=2.0, label="Out of sample (date split)")
    ax.axvline(split, color=orange, ls="--", lw=1.1)
    ax.text(0.02, 0.95, "IN SAMPLE\n2020-01-02 to 2026-05-20", transform=ax.transAxes, va="top", color=blue, fontweight="bold")
    ax.annotate("OUT OF SAMPLE\nStarts 2026-05-21", xy=(split, totals[before[-1]]),
                xytext=(-100, -66), textcoords="offset points", color=orange, fontsize=10,
                arrowprops={"arrowstyle": "->", "color": orange})
    ax.text(0.98, 0.43, f"Total P&L\n${totals[-1]:,.0f}", transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    oos_dates = [dates[before[-1]]] + [dates[i] for i in after]
    oos_totals = [0.0]
    running = 0.0
    for i in after:
        running += pnl[i]
        oos_totals.append(running)
    zoom.axvspan(split, dates[-1], color=orange, alpha=0.08)
    zoom.plot(oos_dates, oos_totals, color=orange, lw=2.0)
    zoom.set_title("Out-of-sample detail: cumulative P&L reset to $0 at the split", loc="left", fontsize=12, pad=12)
    zoom.text(0.02, 0.93, f"{split:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}\n{metadata['trades']['out_of_sample']} trades | P&L ${running:,.0f}",
              transform=zoom.transAxes, va="top", color=orange, fontweight="bold")
    zoom.xaxis.set_major_locator(mdates.MonthLocator())
    zoom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    zoom.set_xlim(oos_dates[0], dates[-1])
    fig.text(0.08, 0.078, "Provisional historical simulation, not actual account P&L. Source run: 2026-09-29; data through 2026-09-28.", fontsize=9, color="#536276")
    fig.text(0.08, 0.049, "Stale underlying caches affect recent results. The configuration changed after the split; this is not an untouched holdout.", fontsize=9, color="#8b491b")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.865, bottom=0.15, hspace=0.42)
    fig.savefig(CHARTS / "pnl_in_sample_out_of_sample.svg", metadata={"Date": None, "Creator": "ORB research release"})
    # Local raster preview only; the public export selects the scanned SVG.
    fig.savefig(CHARTS / "pnl_preview.png", dpi=150, metadata={"Software": "ORB research release"})
    plt.close(fig)
    print("Saved labeled in-sample / out-of-sample P&L chart; aggregate totals verified.")


if __name__ == "__main__":
    main()
