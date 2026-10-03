"""
_production_run.py
==================
Final production config. All decisions locked in.

When imported: exposes universe/sigma/filter definitions only (safe for live system).
When run directly: also executes full backtest, computes stats, saves equity chart.
"""
import sys, io, contextlib, json
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, ".")
from orb_backtester import StrategyConfig, run_backtest
from orb_event_study.config import INSTRUMENTS

# ── Active trading universe ───────────────────────────────────────────────────
# Explicit whitelist — grouped by sector. Everything in INSTRUMENTS not listed
# here is implicitly cut. Class A = independent catalyst (2× quiet, excl filter,
# no purity skip). Class B = market-driven (purity-skipped when total_n ≤ 3).
# Direction filters and special rules noted inline.

ACTIVE = [
    # ── Broad Market — Class B ────────────────────────────────────────────────
    "TQQQ",   # QQQ  3x bull
    "SQQQ",   # QQQ  3x bear
    "UPRO",   # SPY  3x bull
    "SPXS",   # SPY  3x bear
    "UDOW",   # DIA  3x bull
    "SDOW",   # DIA  3x bear
    "FNGD",   # FANG 3x bear  (FNGU cut: duplicate QQQ exposure)

    # ── Technology Sector — Class B ───────────────────────────────────────────
    "TECL",   # XLK  3x bull
    "TECS",   # XLK  3x bear
    "WEBL",   # XLC  3x bull  (communication services)
    "WEBS",   # XLC  3x bear

    # ── Semiconductors — Class B ──────────────────────────────────────────────
    "SOXL",   # SOXX 3x bull
    "SOXS",   # SOXX 3x bear

    # ── Financials — Class B ──────────────────────────────────────────────────
    "FAZ",    # XLF  3x bear  (FAS cut: 47.1% WR weak sister)

    # ── Healthcare — Class B ──────────────────────────────────────────────────
    "CURE",   # XLV  3x bull

    # ── Volatility — Class B ──────────────────────────────────────────────────
    "UVIX",   # VIX  2x bull

    # ── Real Estate — Class B ─────────────────────────────────────────────────
    "URE",    # IYR  2x bull

    # ── Homebuilders — Class B ────────────────────────────────────────────────
    "NAIL",   # ITB  3x bull

    # ── Energy E&P — Class B ──────────────────────────────────────────────────
    "GUSH",   # XOP  2x bull

    # ── Gold Miners — Class B ─────────────────────────────────────────────────
    "NUGT",   # GDX  2x bull
    "DUST",   # GDX  2x bear  (JDST cut: DUST/JDST redundant, DUST stronger per-trade)
    "JNUG",   # GDXJ 2x bull

    # ── EM / International — Class A ──────────────────────────────────────────
    "EDC",    # EEM  3x bull  [direction: up gaps only]  [flood loser: sized 0×]
    "EDZ",    # EEM  3x bear  [direction: down gaps only]
    "KORU",   # EWY  3x bull  (South Korea)
    "MEXX",   # EWW  3x bull  (Mexico)
    "CHAU",   # ASHR 2x bull  (China A shares)  [direction: up gaps only]
    "INDL",   # INDY 3x bull  (India)

    # ── Biotech — Class A ─────────────────────────────────────────────────────
    "LABU",   # IBB  3x bull  [direction: up gaps only]
    "LABD",   # IBB  3x bear  [direction: down gaps only / short trade]
    "MSOX",   # MSOS 2x bull  (cannabis)

    # ── Regional Banking — Class A ────────────────────────────────────────────
    "DPST",   # KRE  3x bull

    # ── Nat Gas — Class A (drift instrument) ──────────────────────────────────
    "KOLD",   # UNG  2x bear  [gap excl exemption: big gaps allowed through]

    # ── Single Stock — Class A ────────────────────────────────────────────────
    "NVDU",   # NVDA 2x bull
    "NVDG", "NVDL", "NVDW", "NVDX",   # NVDA 2x bull (alternate providers — EXPANSION)
    "TSLL",   # TSLA 2x bull
    "TSL", "TSLG", "TSLI", "TSLR", "TSLT", "TSLW",   # TSLA 2x bull alt providers — EXPANSION
    "AMDL",   # AMD  2x bull
    "AMDG", "AMUU",   # AMD 2x bull alt providers — EXPANSION

    # ── Crypto (BTC) — Class A ────────────────────────────────────────────────
    "BITU",   # BTC  2x bull
    "BTCL",   # BTC  2x bull (Rex — alternate provider)
    "BITX", "BTFX",   # BTC 2x bull alt providers — EXPANSION (BITX previously rejected as redundant)
    "BTCZ",   # BTC  2x bear  [direction: down gaps only / short trade]
    "SBIT",   # BTC  2x bear  [direction: down gaps only / short trade]

    # ── Crypto (ETH) — Class A ────────────────────────────────────────────────
    "ETHU",   # ETH  2x bull  [gap filter: 2% flat]  [Mon excluded: 65h weekend gap]
    "ETHD",   # ETH  2x bear  [gap filter: 2% flat]  [Mon excluded]
    "ETU",    # ETH  2x bull (ProShares — alternate provider)
    "ETH", "ETHT",   # ETH 2x bull alt providers — EXPANSION (ETHT previously rejected as correlation cut)

    # ── Crypto (SOL / XRP) — Class A ──────────────────────────────────────────
    "SOLT",   # SOL  2x bull  (thin: ~2024+)
    "XRPT",   # XRP  2x bull  (thin: ~2025+)
    "UXRP",   # XRP  2x bull  (thin: ~2025+)

    # ── EXPANSION: re-litigate previously cut names + new same-bucket additions ─
    "SPXL",   # SPY  3x bull (paired with UPRO, SPXS)
    "FAS",    # XLF  3x bull (previously rejected, 47.1% WR weak sister)
    "FNGU",   # QQQ  3x bull (previously rejected: FNGD wins both directions)
    "BULZ",   # QQQ  3x bull (FANG bull alt — MicroSectors)
    "GDXU",   # GDX  2x bull (paired with NUGT/DUST)
    "JDST",   # GDXJ 2x bear (previously rejected as loser)
    "BOIL",   # UNG  2x bull (previously rejected: thin margins, paired with KOLD)
]

# Key cuts: BITX (low-quality exclusive days vs BITU), BOIL (thin margins),
# FNGU (QQQ duplication), FAS/TNA/SRTY/ETHT/DRN (correlation cuts),
# all 2x broad-market (no edge over 3x), rates (unvalidated),
# crypto 1x/spot (no gap depletion mechanism).

UNIVERSE = {s: INSTRUMENTS[s] for s in ACTIVE if s in INSTRUMENTS}
ETH_SYMS = {sym for sym, info in UNIVERSE.items() if info["underlying"] == "ETH"}
GAP_FILTER = {
    sym: 0.02 if sym in ETH_SYMS else round(info["leverage"] * 0.02, 4)
    for sym, info in UNIVERSE.items()
}
DOW_EXCL = {sym: {0} for sym in ("ETHU", "ETHD") if sym in UNIVERSE}
SYMS = list(UNIVERSE.keys())
SIGMA = {
    "BTC":0.0198,"ETH":0.0300,"SOL":0.0457,"XRP":0.0369,
    "QQQ":0.0126,"SPY":0.0087,"IWM":0.0105,"DIA":0.0088,
    "XLF":0.0139,"XLE":0.0130,"IBB":0.0112,"IHE":0.0082,
    "GDX":0.0181,"GDXJ":0.0195,"SOXX":0.0145,
    "EEM":0.0128,"VWO":0.0124,"FXI":0.0153,"EWY":0.0150,
    "XLK":0.0119,"XLC":0.0101,"XLV":0.0081,
    "KRE":0.0157,"XRT":0.0120,"ITB":0.0161,"INDY":0.0101,"KWEB":0.0171,
    "GLD":0.0081,"SLV":0.0154,"USO":0.0165,"UNG":0.0208,"XOP":0.0176,
    "MSTR":0.0381,"COIN":0.0365,"TSM":0.0199,
    "NVDA":0.0278,"TSLA":0.0260,"SMCI":0.0309,"AMD":0.0268,
    "MSOS":0.0348,
    "TLT":0.0060,"IEF":0.0028,"VIX":0.0603,
    "ASHR":0.0131,"EWW":0.0134,"ITA":0.0105,"IYR":0.0130,"XLU":0.0088,
}
# ── Transaction cost model ────────────────────────────────────────────────────
# Commission: IBKR Fixed plan — $0.005/share, min $1/order, max 1% of trade value.
# Spread + slippage: estimated one-way bps by instrument class.
# Total cost is applied round-trip (entry leg + exit leg).
IBKR_PER_SHARE: float = 0.005   # $/share per leg
IBKR_MIN:       float = 1.00    # $ minimum per order (per leg)
IBKR_MAX_PCT:   float = 0.01    # 1% of trade value maximum per leg

# Mechanical per-instrument round-trip spread and Amihud illiquidity.
# Generated by scripts/compute_market_impact.py (Corwin-Schultz 2012 +
# Amihud 2002) from intraday minute bars.  Re-run if universe changes.
_MKTIMPACT_PATH = Path(__file__).parent / "market_impact.json"
_MKTIMPACT: dict[str, dict] = {}
if _MKTIMPACT_PATH.exists():
    with open(_MKTIMPACT_PATH) as _f:
        _MKTIMPACT = json.load(_f)

_CS_BPS_RT: dict[str, float] = {
    sym: d["cs_spread_bps"]
    for sym, d in _MKTIMPACT.items()
    if d.get("cs_spread_bps") is not None
}
_ILLIQ: dict[str, float] = {
    sym: d["amihud_illiq"]
    for sym, d in _MKTIMPACT.items()
    if d.get("amihud_illiq") is not None
}
_DEFAULT_CS_BPS_RT: float = 10.0   # round-trip fallback for missing instruments


def apply_trade_costs(trades: pd.DataFrame, scale: float = 1.0,
                      spread_and_impact: bool = True) -> pd.DataFrame:
    """
    Deduct round-trip transaction costs from pnl_pct.
    Combines IBKR Fixed commission ($0.005/sh, min $1, max 1% per leg)
    with per-instrument spread/slippage estimates.

    scale: cost multiplier for sensitivity sweep (0 / 0.5 / 1.0 / 2.0 / 3.0).
    pnl_pct is reduced by total_cost / position_value (same denominator as pnl_pct).
    Returns a copy; dollar_pnl must be recomputed after calling this.
    """
    t = trades.copy()
    pos_val = t["entry_price"] * t["shares"]   # same denominator as pnl_pct

    # IBKR commission: max(min_fee, min(per_share_cost, max_pct_fee)) per leg × 2 legs
    per_share_cost = t["shares"] * IBKR_PER_SHARE
    max_fee        = pos_val * IBKR_MAX_PCT
    commission_leg = per_share_cost.clip(lower=IBKR_MIN).clip(upper=max_fee)
    ibkr_rt        = 2.0 * commission_leg * scale

    # Corwin-Schultz round-trip spread + Amihud market impact (both round-trip)
    cs_bps_rt     = t["symbol"].map(_CS_BPS_RT).fillna(_DEFAULT_CS_BPS_RT)
    illiq         = t["symbol"].map(_ILLIQ).fillna(0.0)
    impact_bps_rt = 2.0 * illiq * pos_val * 10_000
    spread_rt     = (cs_bps_rt + impact_bps_rt) / 10_000 * pos_val * scale
    if not spread_and_impact:
        # commission only -- a per-trade slippage model is charging the rest
        spread_rt = spread_rt * 0.0

    total_cost          = ibkr_rt + spread_rt
    t["cost_bps_rt"]    = total_cost / pos_val * 10_000   # effective round-trip bps
    t["cost_ibkr_rt"]   = ibkr_rt
    t["cost_spread_rt"] = spread_rt
    t["pnl_pct"]        = t["pnl_pct"] - total_cost / pos_val
    return t


def ps_filter(sym, info, k=1.25):
    ul, sig = info["underlying"], SIGMA.get(info["underlying"])
    if sig is None: return None
    return (ul, sig*k, True) if info["inverse"] else (ul, sig*k)
PS_FILTERS = {s: ps_filter(s, i) for s, i in UNIVERSE.items() if ps_filter(s, i)}

# Class A — independent catalyst; gap exclusion applied to Class A only
_CLASS_A_UL   = {"BTC","ETH","SOL","XRP","NVDA","TSLA","SMCI","AMD","UNG"}
_CLASS_A_SYMS = {"KORU","CHAU","MEXX","INDL","LABU","LABD","DPST","KOLD","MSOX"}
_EXCL_EXCEPTIONS = {"KOLD"}   # drift instrument — let big gaps through
CLASS_A_EXCL_SYMS = tuple(
    s for s, i in UNIVERSE.items()
    if s not in _EXCL_EXCEPTIONS
    and (s in _CLASS_A_SYMS or i["underlying"] in _CLASS_A_UL)
)

# ── Two-class instrument taxonomy ─────────────────────────────────────────────
# Class A vs Class B classification is about GAP-CONTINUATION BEHAVIOR, not
# correlation profile.
#   Class A = catalyst-driven (earnings, FOMC for crypto, news on single stocks,
#     country policy for intl ETFs). These benefit from RTG filtering (excluding
#     wide-ORB-small-gap whipsaw days) AND the 2× quiet-day sizing bump.
#   Class B = macro-driven (broad market, sector, commodity, gold miners). These
#     continue directionally even on wide-ORB-small-gap days, so no RTG filter.
# Gold miners DIVERSIFY the portfolio (low correlation ~0.07 with Class A) but
# behave like Class B for trade-quality purposes — confirmed by 2026-06 test
# extending RTG to NUGT/DUST/JNUG: zero risk-adjusted improvement, -$967 P&L.
# ──────────────────────────────────────────────────────────────────────────────
CRYPTO_UL          = {"BTC","ETH","SOL","XRP"}
SINGLESTK_UL       = {"NVDA","TSLA","SMCI","AMD"}
NATGAS_UL          = {"UNG"}
INTERNATIONAL_SYMS = {"KORU","CHAU","MEXX","INDL"}
BIOTECH_SYMS       = {"LABU","LABD","MSOX"}
BANKING_SYMS       = {"DPST"}
DRIFT_SYMS         = {"KOLD"}
CLASS_A_UL         = CRYPTO_UL | SINGLESTK_UL | NATGAS_UL
CLASS_A_SYMS       = INTERNATIONAL_SYMS | BIOTECH_SYMS | BANKING_SYMS | DRIFT_SYMS
FLOOD_LOSERS       = {"EDC"}

def _is_class_a(sym, ul):
    return sym in CLASS_A_SYMS or ul in CLASS_A_UL

# ── Production sizing parameters ──────────────────────────────────────────────
BASELINE_PCT = 20.0
CAP_UNITS    = float("inf")

def apply_sizing(df):
    df = df.copy()
    def mult(row):
        if row["purity_skip"]:   return 0.0
        if row["equity_flood"] and row["is_flood_loser"]:
            return 0.0
        if row["equity_flood"]:
            # Flood days: Class A 1.5x / Class B 0.75x
            return 1.5 if row["is_class_a"] else 0.75
        if row["crypto_cluster"]:
            # Crypto cluster days: 1x for everyone — caps correlated crypto
            # exposure on days when ≥3 crypto underlyings fire together.
            return 1.0
        if row["quiet"]:
            # Quiet days: Class A 2.5x (catalyst-day premium) / Class B 1x
            return 2.5 if row["is_class_a"] else 1.0
        return 1.0
    df["mult"] = df.apply(mult, axis=1)
    day_mult_sum = df.groupby("date")["mult"].sum().rename("day_mult_sum")
    df = df.merge(day_mult_sum, on="date")
    df["cap_factor"]  = (CAP_UNITS / df["day_mult_sum"]).clip(upper=1.0)
    df["capped_mult"] = df["mult"] * df["cap_factor"]
    df["sized_pnl"]   = df["dollar_pnl"] * df["capped_mult"] * (BASELINE_PCT / 10.0)
    return df

# ── Walk-forward PS filter schedule (5-year rolling sigma) ───────────────────
def _build_ps_filters_from_sigmas(sigmas: dict, k: float) -> dict:
    """Build a prior_session_filters dict from a {underlying: sigma} map."""
    result = {}
    for sym, info in UNIVERSE.items():
        ul = info["underlying"]
        sig = sigmas.get(ul)
        if sig is None:
            continue
        result[sym] = (ul, sig * k, True) if info["inverse"] else (ul, sig * k)
    return result

def build_rolling_ps_schedule(years, lookback_years=5.0, k=1.25):
    from orb_live.ops.calibration import compute_rolling_sigma_schedule
    underlyings = list({i["underlying"] for i in UNIVERSE.values()})
    return compute_rolling_sigma_schedule(
        underlyings=underlyings,
        years=years,
        lookback_years=lookback_years,
        k=k,
        build_ps_fn=_build_ps_filters_from_sigmas,
    )

# ── StrategyConfig for this production universe ───────────────────────────────
cfg = StrategyConfig(
    symbols=SYMS, instrument_gap_filters=GAP_FILTER,
    prior_session_filters=PS_FILTERS, day_of_week_exclusions=DOW_EXCL,
    direction_filters={"SBIT":-1,"BTCZ":-1,"LABD":-1,"EDZ":-1,"LABU":+1,"CHAU":+1,"EDC":+1,"BITU":+1},
    use_rtg_scaling=False,
    rtg_gap_exclusion=True, rtg_gap_exclusion_threshold=0.08,
    rtg_gap_exclusion_symbols=CLASS_A_EXCL_SYMS,
    min_increment_pct=0.0, min_profit_pct=0.0,
    start_date="2020-01-01", initial_equity=100_000.0, daily_risk_pct=0.20,
    eod_exit_hour=16, eod_exit_minute=0,
    # Exit weights — validated 2026-05-26 production grid search (45+29 combos)
    exit_ratio_tp1=1.00, exit_ratio_tp2=0.00, exit_ratio_tp3=0.00,
)

# ── Backtest execution — only when run as script ──────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    print("Running backtest...")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        trades_raw = run_backtest(cfg)

    # ── Shared annotation (cost-independent) ────────────────────────────────
    _sym_to_ul = {s: i["underlying"] for s, i in UNIVERSE.items()}

    def _annotate(raw: pd.DataFrame) -> pd.DataFrame:
        t = raw.copy()
        t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
        t["date"]       = pd.to_datetime(t["date"])
        t["underlying"] = t["symbol"].map(_sym_to_ul)
        t["year"]       = t["date"].dt.year
        t["is_crypto"]      = t["underlying"].isin(CRYPTO_UL)
        t["is_class_a"]     = t.apply(lambda r: _is_class_a(r["symbol"], r["underlying"]), axis=1)
        t["is_flood_loser"] = t["symbol"].isin(FLOOD_LOSERS)
        return t

    # Day-level statistics don't depend on P&L — compute once from raw counts.
    _base = _annotate(trades_raw)
    day_stats = _base.groupby("date").agg(
        total_n      = ("symbol", "count"),
        n_crypto_uls = ("underlying", lambda x: x[_base.loc[x.index, "is_crypto"]].nunique()),
    ).reset_index()
    day_stats["equity_flood"]   = day_stats["total_n"] >= 10
    day_stats["crypto_cluster"] = day_stats["n_crypto_uls"] >= 3
    day_stats["quiet"]          = ~day_stats["equity_flood"] & ~day_stats["crypto_cluster"]
    PURITY_THRESH = 3

    def _full_pipeline(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Annotate → merge day stats → size → return (trades, baseline_daily, sized_daily)."""
        t = _annotate(raw)
        t = t.merge(day_stats[["date","total_n","equity_flood","crypto_cluster","quiet"]], on="date")
        t["purity_skip"] = ~t["is_class_a"] & (t["total_n"] <= PURITY_THRESH)
        t = apply_sizing(t)
        b_daily = t.groupby("date")["dollar_pnl"].sum() * (BASELINE_PCT / 10.0)
        s_daily = t.groupby("date")["sized_pnl"].sum()
        return t, b_daily, s_daily

    trades_gross, baseline_gross, sized_gross = _full_pipeline(trades_raw)
    trades_net,   baseline_net,   sized_net   = _full_pipeline(apply_trade_costs(trades_raw))

    # ── Metrics ─────────────────────────────────────────────────────────────
    rf_daily = 0.043 / 365
    START    = 10_000.0
    cal      = pd.date_range(trades_gross["date"].min(), trades_gross["date"].max(), freq="D")

    def compound_equity(daily: pd.Series) -> pd.Series:
        filled = daily.reindex(cal, fill_value=0.0)
        return START * (1 + filled / START).cumprod()

    def stats(daily: pd.Series) -> tuple[float, float, float]:
        filled  = daily.reindex(cal, fill_value=0.0)
        r       = filled / START          # return on total reserved capital
        sh      = (r.mean()-rf_daily)/r.std()*np.sqrt(365) if r.std()>0 else np.nan
        eq      = compound_equity(daily)
        peak    = eq.cummax()
        maxdd   = ((eq - peak) / peak).min()
        years   = (cal[-1] - cal[0]).days / 365
        ann_ret = (eq.iloc[-1] / START) ** (1/years) - 1
        calmar  = ann_ret / abs(maxdd) if maxdd != 0 else np.nan
        return sh, maxdd, calmar

    sh_bg, dd_bg, cal_bg = stats(baseline_gross)
    sh_sg, dd_sg, cal_sg = stats(sized_gross)
    sh_bn, dd_bn, cal_bn = stats(baseline_net)
    sh_sn, dd_sn, cal_sn = stats(sized_net)

    # ── Main report ──────────────────────────────────────────────────────────
    n_purity = trades_gross["purity_skip"].sum()
    n_flood  = (~trades_gross["purity_skip"] & trades_gross["mult"].eq(0)).sum()
    avg_cost = trades_net["cost_bps_rt"].mean()
    tot_cost = (trades_net["cost_bps_rt"] / 10_000 * 1_000).sum() * (BASELINE_PCT / 10.0)

    print(f"\n{'='*68}")
    print("PRODUCTION CONFIG — RESULTS  (gross vs net of transaction costs)")
    print(f"{'='*68}")
    print(f"  Trades:      {len(trades_gross):>5}  ({n_purity} purity-skipped  {n_flood} flood/other-skipped)")
    print(f"  Date range:  {trades_gross['date'].min().date()} → {trades_gross['date'].max().date()}")
    print(f"  Avg cost:    {avg_cost:.1f} bps/trade round-trip  |  Total cost drag: ${tot_cost:,.0f}")
    print(f"\n  {'Metric':<16}  {'Flat/Gross':>10}  {'Flat/Net':>10}  {'Prod/Gross':>10}  {'Prod/Net':>10}")
    print(f"  {'─'*62}")
    print(f"  {'Sharpe':<16}  {sh_bg:>10.3f}  {sh_bn:>10.3f}  {sh_sg:>10.3f}  {sh_sn:>10.3f}")
    print(f"  {'Max DD':<16}  {dd_bg:>10.1%}  {dd_bn:>10.1%}  {dd_sg:>10.1%}  {dd_sn:>10.1%}")
    print(f"  {'Calmar':<16}  {cal_bg:>10.3f}  {cal_bn:>10.3f}  {cal_sg:>10.3f}  {cal_sn:>10.3f}")
    print(f"  {'Total P&L':<16}  ${baseline_gross.sum():>9,.0f}  ${baseline_net.sum():>9,.0f}"
          f"  ${sized_gross.sum():>9,.0f}  ${sized_net.sum():>9,.0f}")

    print(f"\n  Year-by-year  (Production sizing, net of costs):")
    print(f"  {'Year':>5}  {'Gross Sh':>9}  {'Net Sh':>8}  {'Cost $':>8}  {'Net P&L':>10}  {'WR':>6}")
    for yr in sorted(trades_gross["year"].unique()):
        tg = trades_gross[trades_gross["year"]==yr]
        tn = trades_net[trades_net["year"]==yr]
        ds_g = tg.groupby("date")["sized_pnl"].sum()
        ds_n = tn.groupby("date")["sized_pnl"].sum()
        yr_cal = pd.date_range(ds_g.index.min(), ds_g.index.max(), freq="D")
        def _sh_yr(d):
            r = d.reindex(yr_cal, fill_value=0.0) / START
            return (r.mean()-rf_daily)/r.std()*np.sqrt(365) if r.std()>0 else np.nan
        shg = _sh_yr(ds_g); shn = _sh_yr(ds_n)
        cost_yr = (tn["cost_bps_rt"] / 10_000 * 1_000).sum() * (BASELINE_PCT / 10.0)
        wr = (tn[tn["mult"]>0]["sized_pnl"] > 0).mean()
        flag = " ▲" if shn > 1.0 else (" ▼" if shn < 0.0 else "")
        print(f"  {yr:>5}  {shg:>9.3f}  {shn:>8.3f}  ${cost_yr:>7,.0f}  "
              f"${ds_n.sum():>9,.0f}  {wr:>6.1%}{flag}")

    # ── Cost sensitivity sweep ───────────────────────────────────────────────
    print(f"\n  Cost sensitivity (Production sizing):")
    print(f"  {'Scale':>6}  {'Avg bps':>8}  {'Sharpe':>8}  {'Max DD':>8}  {'Calmar':>8}  {'Total P&L':>12}")
    print(f"  {'─'*58}")
    for scale, label in [(0.0, "0× gross"), (0.5, "0.5×"), (1.0, "1× baseline"),
                         (2.0, "2×"), (3.0, "3×")]:
        t_sw = apply_trade_costs(trades_raw, scale=scale)
        _, _, s_sw = _full_pipeline(t_sw)
        sh_sw, dd_sw, cal_sw = stats(s_sw)
        avg_bps = trades_net["cost_bps_rt"].mean() * scale
        print(f"  {label:>6}  {avg_bps:>8.1f}  {sh_sw:>8.3f}  {dd_sw:>8.1%}  {cal_sw:>8.3f}  ${s_sw.sum():>11,.0f}")

    # ── Equity chart ─────────────────────────────────────────────────────────
    eq_gross = compound_equity(sized_gross)
    eq_net   = compound_equity(sized_net)
    dd_gross_pct = (eq_gross - eq_gross.cummax()) / eq_gross.cummax() * 100
    dd_net_pct   = (eq_net   - eq_net.cummax())   / eq_net.cummax()   * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    flood_dates   = day_stats[day_stats["equity_flood"]]["date"]
    cluster_dates = day_stats[day_stats["crypto_cluster"] & ~day_stats["equity_flood"]]["date"]
    for d in flood_dates:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="#ff6b35", lw=0)
    for d in cluster_dates:
        ax1.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="#7b68ee", lw=0)

    ax1.plot(eq_gross.index, eq_gross.values, color="#555555", lw=1.2, alpha=0.7,
             label=f"Gross (no costs)  Sh={sh_sg:.2f}  MaxDD={dd_gross_pct.min():.1f}%")
    ax1.plot(eq_net.index,   eq_net.values,   color="#00d4aa", lw=1.8,
             label=f"Net (costs incl)  Sh={sh_sn:.2f}  MaxDD={dd_net_pct.min():.1f}%")
    ax1.axhline(START, color="#444444", lw=0.8, ls="--")
    ax1.set_ylabel("Portfolio Value ($)", color="#aaaaaa", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", framealpha=0.2, fontsize=9,
               labelcolor="#cccccc", facecolor="#1a1a1a",
               handles=[
                   plt.Line2D([],[],color="#555555",lw=1.2,
                              label=f"Gross (no costs)  Sh={sh_sg:.2f}  MaxDD={dd_gross_pct.min():.1f}%"),
                   plt.Line2D([],[],color="#00d4aa",lw=1.8,
                              label=f"Net (costs incl)  Sh={sh_sn:.2f}  MaxDD={dd_net_pct.min():.1f}%"),
                   Patch(color="#ff6b35", alpha=0.4, label="Equity flood days"),
                   Patch(color="#7b68ee", alpha=0.4, label="Crypto cluster days"),
               ])
    ax1.set_title("ORB Strategy — Production Config  ($10k account, 20% baseline/trade, costs included)",
                  color="#eeeeee", fontsize=12, pad=10)
    ax1.grid(axis="y", color="#222222", lw=0.5)
    for yr in range(2021, 2027):
        ax1.axvline(pd.Timestamp(f"{yr}-01-01"), color="#333333", lw=0.8, ls=":")

    ax2.fill_between(dd_gross_pct.index, dd_gross_pct.values, 0,
                     color="#555555", alpha=0.4, label="Gross DD")
    ax2.fill_between(dd_net_pct.index,   dd_net_pct.values,   0,
                     color="#00d4aa", alpha=0.35, label="Net DD")
    ax2.set_ylabel("Drawdown %", color="#aaaaaa", fontsize=9)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(loc="lower left", framealpha=0.2, fontsize=8,
               labelcolor="#cccccc", facecolor="#1a1a1a")
    ax2.grid(axis="y", color="#222222", lw=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.tick_params(axis="x", colors="#aaaaaa")

    plt.tight_layout(h_pad=0.5)
    out_path = "production_equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nChart saved → {out_path}")
