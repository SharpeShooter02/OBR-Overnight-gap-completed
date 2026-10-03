"""
Run the skip-cheap-with-singletons algorithm on subsets of master_universe.csv.

Strategy config (consistent baseline, no hand-tuning):
  - Gap threshold = leverage × 2% per instrument
  - NO direction filters
  - NO prior-session filters
  - NO day-of-week exclusions
  - NO day-level gates (flood/cluster/purity skip OFF)
  - Flat sizing: each trade earns pnl_pct × $1000 (no bucket boost)

Skip-cheap-with-singletons rule:
  - Per (date, UL) group: if 2+ instruments gapped, drop the cheapest by
    trailing-252d median close
  - If only 1 instrument gapped on a UL, keep it

Usage: edit CLASSES_TO_INCLUDE at bottom for the run you want.
"""
import sys, io, contextlib
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from pathlib import Path

from orb_backtester import StrategyConfig, run_backtest
from orb_event_study.config import INSTRUMENTS
import _production_run as P
from scripts.sigma_master import verify_or_load as sigma_verify_or_load
from scripts.sigma_master import load_sigma_master


PS_K = 1.25   # set to 0 (or None) to disable PS filters

# RTG gap exclusion (matches production-accepted verdict): drop trades where
# overnight Range-to-Gap < 8%, applied to Class A symbols only. KOLD exempted
# because it's a drift play that benefits from wide-gap days.
RTG_GAP_EXCLUSION           = False   # tested -$2k P&L; left off
RTG_GAP_EXCLUSION_THRESHOLD = 0.08
RTG_DRIFT_EXEMPT            = {"KOLD"}

# Direction filters (+1 = trade only gap-up days, -1 = gap-down days).
# LABU/LABD: only trade biotech ETFs after bullish IBB overnight catalysts
# (LABU gaps up = IBB up; LABD gaps down = IBB up). The reverse direction has
# essentially no edge (break-even, see per-direction breakdown 2026-06-08).
DIRECTION_FILTERS = {
    "LABU": +1,
    "LABD": -1,
}

# Explicit drops — overrides master_universe.csv inclusion.
# Reason for each:
#   CWEB — China internet bull, -$486 P&L; YINN/YANG provide better China exposure
#   CHAU — China A-shares bull, -$117 P&L; YINN/YANG also cover this UL space
EXPLICIT_DROPS = {
    "CWEB", "CHAU",                   # China internet, China A-shares — China exposure covered by YINN/YANG
    "BNKU", "DFEN", "DPST",           # bull-only macro (KBE/ITA/KRE)
    # C3 single-sector bulls dropped 2026-06-09 — pairs better capture broad-market moves
    "NAIL", "RETL", "DRN", "MIDU",
    "NRGU",                           # XOP exposure replaced by GUSH/DRIP pair
    # ETH is a 1x spot Ethereum trust carried as leverage=2, and the only symbol
    # that is also an underlying key (etf_daily["ETH"] ~24 vs ul_daily["ETH"]
    # ~2400). Dropped 2026-09-04; see orb_live/strategy/v1_strategy.py.
    "ETH",
}

# 3-class taxonomy — IMPORTED from the live strategy profile, not duplicated.
#
# There used to be a second copy here. It drifted: live dropped UVIX, 12
# never-trading single-stock clones and the five gold miners, and this copy kept
# all 18, so classify() disagreed across the two repos and the weight applied to
# a gold trade depended on which side you asked. The profile lives in
# orb-live-trading; the backtest reads it.
#
# C1 (CRYPTO)   — crypto only; cluster reduction (1.5x) applies appropriately
# C2 (PARTIAL)  — international + single-stocks + XOP
# C3 (MACRO+)   — broad market, large sector pairs, biotech, natural gas, and
#                 the gold miners, which contradict C2's flood notch
from orb_live.strategy.v1_strategy import (  # noqa: E402
    CLASS_1_SYMS, CLASS_2_SYMS, SHARED_ALLOTMENT, allotment_group,
)

# C3 default = broad market (SPY/QQQ/IWM/DIA pairs), large sector pairs
# (SOXX/XLK/XLF/XLC), biotech (LABU/LABD/MSOX), and natural gas (KOLD/BOIL)
# Class 3 = everything else (broad market + macro sector pairs + LABU/LABD/IBB
# + tech sector + financials + utilities + REITs + retail + housing)

# Force-include symbols that aren't in master_universe.csv (KOLD/BOIL) or get
# filtered out by class/$ADV but the user wants tested anyway.  Each becomes a
# synthetic row in master with class="force_include" if not already present.
_SINGLE_STOCK_PICKS = {"AMDL", "TSLL", "NVDU"}    # hand-picked single ETF per UL
_UNG_DRIFT          = {"KOLD", "BOIL"}
_XRP_CRYPTO         = {"UXRP", "XRPT"}
_XOP_PAIR           = {"GUSH"}                    # + "DRIP" once cached
FORCE_INCLUDE       = _SINGLE_STOCK_PICKS | _UNG_DRIFT | _XRP_CRYPTO | _XOP_PAIR

# Symbols that get treated as Class A regardless of BE/crypto status (drift &
# single-stock plays that get the catalyst-day premium in production).
FORCE_CLASS_A       = _SINGLE_STOCK_PICKS | _UNG_DRIFT


START_EQUITY = 10_000.0
CACHE_DIR    = Path("orb_event_study/cache/intraday")
MASTER_CSV   = Path("master_universe.csv")


def load_master():
    return pd.read_csv(MASTER_CSV)


def add_force_includes_to_master(master: pd.DataFrame) -> pd.DataFrame:
    """Append synthetic rows for FORCE_INCLUDE symbols not already in master.
    UL and leverage come from INSTRUMENTS dict; class='force_include'."""
    have = set(master["etf"].astype(str).values)
    extras = []
    for sym in FORCE_INCLUDE:
        if sym in have: continue
        info = INSTRUMENTS.get(sym)
        if not info: continue
        extras.append({
            "etf": sym,
            "underlying": info["underlying"],
            "class": "force_include",
            "leverage": info.get("leverage", 2),
            "direction": "bear" if info.get("inverse") else "bull",
            "binary_event": None,
        })
    if extras:
        master = pd.concat([master, pd.DataFrame(extras)], ignore_index=True)
    # For symbols already in master that should also be force-included (UXRP, XRPT,
    # AMDL, TSLL, NVDU), no row append needed — they're picked up via the class
    # filter expansion in main().
    return master


def load_daily_close(symbol: str) -> pd.Series:
    sym_dir = CACHE_DIR / symbol
    if not sym_dir.exists():
        return pd.Series(dtype=float)
    frames = []
    for pq in sorted(sym_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
            if not df.empty: frames.append(df)
        except Exception: continue
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"]      = df["timestamp"].dt.normalize()
    out = df.groupby("date")["close"].last()
    # Same forward-adjustment problem as load_sym_close: this series ranks
    # siblings in skip_cheap_then_top2_when_3plus (register V43).
    from scripts import as_traded as _AT
    if _AT.USE_AS_TRADED:
        out = _AT.to_as_traded(symbol, out)
    return out


def load_daily_volume(symbol: str, window: int = 60) -> pd.Series:
    """Return trailing-`window`-d rolling avg DOLLAR volume (sum of close*volume per bar)."""
    sym_dir = CACHE_DIR / symbol
    if not sym_dir.exists(): return pd.Series(dtype=float)
    frames = []
    for pq in sorted(sym_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
            if not df.empty: frames.append(df)
        except Exception: continue
    if not frames: return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], utc=True)
    df["date"]       = df["timestamp"].dt.normalize()
    df["dollar_vol"] = df["close"] * df["volume"]
    daily = df.groupby("date")["dollar_vol"].sum().sort_index()
    return daily.rolling(window, min_periods=20).mean()


def get_recent_dollar_adv(symbol: str, window: int = 60) -> float:
    """Most-recent valid trailing-`window`-d $ADV. Static universe gate, avoids split-corruption in early years."""
    v = load_daily_volume(symbol, window)
    v = v.dropna()
    if v.empty: return float("nan")
    return float(v.iloc[-1])


def apply_static_dollar_adv_filter(trades: pd.DataFrame, universe: list[str],
                                    min_dollar_adv: float = 5_000_000.0,
                                    window: int = 60,
                                    exempt: set | None = None):
    """Static gate: compute most-recent 60d $ADV per symbol; drop the symbol entirely if below threshold.
    Symbols in `exempt` bypass the gate (kept regardless of $ADV)."""
    exempt = set(exempt or set())
    adv = {s: get_recent_dollar_adv(s, window) for s in universe}
    kept_syms = [s for s, v in adv.items()
                 if s in exempt or (pd.notna(v) and v >= min_dollar_adv)]
    drop_syms = [(s, v) for s, v in adv.items()
                 if s not in exempt and not (pd.notna(v) and v >= min_dollar_adv)]
    kept_trades = trades[trades["symbol"].isin(set(kept_syms))].reset_index(drop=True) if not trades.empty else trades
    return kept_trades, kept_syms, drop_syms, adv


def build_pit_dollar_adv(universe, window: int = 60) -> dict:
    """{symbol: trailing-`window`d $ADV series, lagged one session}.

    The rolling mean at date D includes D's own volume, which is not knowable
    at 09:31 when the universe is decided. Shifting by one session makes the
    gate strictly causal: a trade on D sees volume through D-1 only.
    """
    out = {}
    for s in universe:
        v = load_daily_volume(s, window).dropna()
        if not v.empty:
            v = v.shift(1).dropna()
            if not v.empty:
                out[s] = v
    return out


def apply_pit_dollar_adv_filter(trades: pd.DataFrame, universe: list[str],
                                min_dollar_adv: float = 5_000_000.0,
                                window: int = 60,
                                exempt: set | None = None):
    """Point-in-time liquidity gate -- keep a TRADE if the symbol cleared the
    threshold AS OF THAT DATE.

    Replaces apply_static_dollar_adv_filter, which computed one $ADV from the
    most recent 60 days in the cache and applied that single verdict to the
    whole 2020-2026 backtest. That was look-ahead (EDZ's 2020 trades were
    decided by its 2026 volume) and it was not reproducible (extending the
    cache for the holdout moved the window and silently deleted 69 trades --
    register V34).

    Returns the same 4-tuple shape as the static filter. `kept_syms` is now
    "symbols with at least one qualifying day", which is only meaningful for
    reporting; the real filtering is per trade.
    """
    exempt = set(exempt or set())
    adv = build_pit_dollar_adv(universe, window)
    if trades.empty:
        return trades, sorted(exempt), [], adv

    t = trades.copy()
    d = pd.to_datetime(t["date"])
    d = d.dt.tz_localize("UTC") if d.dt.tz is None else d.dt.tz_convert("UTC")
    t["_d"] = d.dt.normalize()

    keep = np.zeros(len(t), dtype=bool)
    for sym, g in t.groupby("symbol"):
        if sym in exempt:
            keep[g.index] = True
            continue
        ser = adv.get(sym)
        if ser is None or ser.empty:
            continue                      # no volume history -> not tradeable
        idx = ser.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        ser = pd.Series(ser.values, index=idx).sort_index()
        # asof: most recent observation at or before the trade date
        vals = ser.reindex(ser.index.union(g["_d"])).ffill().reindex(g["_d"]).values
        keep[g.index] = pd.notna(vals) & (vals >= min_dollar_adv)

    kept_trades = t[keep].drop(columns="_d").reset_index(drop=True)
    kept_syms = sorted(set(kept_trades["symbol"]))
    dropped = sorted(set(t["symbol"]) - set(kept_syms))
    drop_syms = [(s, float("nan")) for s in dropped]
    return kept_trades, kept_syms, drop_syms, adv


def filter_master(master: pd.DataFrame, classes: list[str]) -> tuple[list[str], dict]:
    """Pick rows by class.  Returns (etf_list, ul_to_etfs_map).  Skips
    instruments missing from either INSTRUMENTS or the intraday cache."""
    sub = master[master["class"].isin(classes)]
    etfs = []
    skipped = []
    ul_map = {}
    for _, row in sub.iterrows():
        sym = row["etf"]
        if pd.isna(row.get("underlying")) or str(row["underlying"]).strip() == "":
            skipped.append((sym, "no underlying specified"))
            continue
        if sym not in INSTRUMENTS:
            skipped.append((sym, "not in INSTRUMENTS"))
            continue
        cache = CACHE_DIR / sym
        if not cache.exists() or len(list(cache.glob("*.parquet"))) == 0:
            skipped.append((sym, "no intraday cache"))
            continue
        if sym in EXPLICIT_DROPS:
            skipped.append((sym, "EXPLICIT_DROPS"))
            continue
        etfs.append(sym)
        ul_map.setdefault(row["underlying"], []).append(sym)
    return sorted(set(etfs)), ul_map, skipped


def gate_by_sigma_coverage(etfs: list[str], master: pd.DataFrame) -> tuple[list[str], list[tuple]]:
    """Auto-compute sigmas for any new underlyings in master_universe.csv.
    Drop any ETF whose underlying has no resolved sigma.
    """
    sigmas, kept, dropped = sigma_verify_or_load(etfs, master, auto_compute=True, verbose=True)
    if dropped:
        print(f"  sigma gate: {len(etfs)} -> {len(kept)} instruments ({len(dropped)} dropped)")
        for s, why in dropped:
            print(f"    {s:6s}  {why}")
    return kept, dropped


def build_cfg(active: list[str], use_ps_filter: bool = False, k: float = PS_K,
              master: pd.DataFrame | None = None) -> StrategyConfig:
    """Clean baseline + optional PS filter and optional RTG gap exclusion."""
    universe = {s: INSTRUMENTS[s] for s in active if s in INSTRUMENTS}
    gap_filter = {s: round(i["leverage"] * 0.02, 4) for s, i in universe.items()}

    ps_filters = {}
    if use_ps_filter and k:
        sigmas = load_sigma_master()
        for s, info in universe.items():
            ul = info["underlying"]
            sig = sigmas.get(ul)
            if sig is None: continue
            ps_filters[s] = (ul, sig * k, True) if info.get("inverse") else (ul, sig * k)

    # RTG gap exclusion symbols = v1 Class A symbols minus drift exceptions
    rtg_syms = tuple()
    if RTG_GAP_EXCLUSION and master is not None:
        be_set     = set(master[master["binary_event"].astype(str).str.upper().eq("YES")]["etf"])
        crypto_set = set(master[master["class"] == "4_CRYPTO"]["etf"])
        class_a    = (be_set | crypto_set | FORCE_CLASS_A) & set(universe.keys())
        rtg_syms   = tuple(sorted(class_a - RTG_DRIFT_EXEMPT))

    return StrategyConfig(
        symbols=list(universe.keys()),
        instrument_gap_filters=gap_filter,
        prior_session_filters=ps_filters,
        day_of_week_exclusions={},
        direction_filters={s: d for s, d in DIRECTION_FILTERS.items() if s in universe},
        use_rtg_scaling=False,
        rtg_gap_exclusion=bool(rtg_syms) and RTG_GAP_EXCLUSION,
        rtg_gap_exclusion_threshold=RTG_GAP_EXCLUSION_THRESHOLD,
        rtg_gap_exclusion_symbols=rtg_syms,
        min_increment_pct=0.0, min_profit_pct=0.0,
        start_date="2020-01-01", initial_equity=100_000.0, daily_risk_pct=0.20,
        eod_exit_hour=16, eod_exit_minute=0,
        exit_ratio_tp1=1.00, exit_ratio_tp2=0.00, exit_ratio_tp3=0.00,
    )


def skip_cheap_with_singletons(trades: pd.DataFrame, prices: dict) -> pd.DataFrame:
    t = trades.copy()
    t["date"]       = pd.to_datetime(t["date"])
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    def asof(row):
        ser = prices.get(row["symbol"])
        if ser is None: return None
        d = row["date"] if row["date"].tz is not None else row["date"].tz_localize("UTC")
        idx = ser.index.searchsorted(d, side="right") - 1
        if idx < 0: return None
        v = ser.iloc[idx]
        return float(v) if pd.notna(v) and v > 0 else None
    t["asof_price"] = t.apply(asof, axis=1)
    group_sizes = t.groupby(["date","underlying"]).size().rename("group_n").reset_index()
    t = t.merge(group_sizes, on=["date","underlying"])
    t = t.sort_values(["date","underlying","asof_price"], ascending=[True,True,True], na_position="last")
    t["is_cheapest"] = ~t.duplicated(subset=["date","underlying"], keep="first")
    keep = (t["group_n"] == 1) | (~t["is_cheapest"])
    return t[keep].reset_index(drop=True)


def skip_cheap_top2_plus_singlestock_consensus(trades: pd.DataFrame, prices: dict, master: pd.DataFrame) -> pd.DataFrame:
    """Per (date, underlying):
       - SS multi-ETF UL: trade only if N >= min(pack_size_available, 3); keep top-1 by price
       - Else:
         - N==1: keep
         - N==2: drop cheapest (skip-cheap)
         - N>=3: keep top 2 by price

       "pack_size_available" = number of distinct ETFs we hold for that UL across
       the whole trades dataframe (proxy for how many sister ETFs exist).
    """
    ss_in_master = master[master["class"] == "5_SINGLE_STOCK"]
    ss_pack_size = ss_in_master.groupby("underlying")["etf"].nunique().to_dict()

    t = trades.copy()
    t["date"]       = pd.to_datetime(t["date"])
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    def asof(row):
        ser = prices.get(row["symbol"])
        if ser is None: return None
        d = row["date"] if row["date"].tz is not None else row["date"].tz_localize("UTC")
        idx = ser.index.searchsorted(d, side="right") - 1
        if idx < 0: return None
        v = ser.iloc[idx]
        return float(v) if pd.notna(v) and v > 0 else None
    t["asof_price"] = t.apply(asof, axis=1)

    # Available pack size per UL = unique symbols we actually have data for
    available_pack = t[t["underlying"].isin(ss_pack_size.keys())].groupby("underlying")["symbol"].nunique().to_dict()

    t["price_rank"] = t.groupby(["date","underlying"])["asof_price"].rank(
        method="first", ascending=False, na_option="bottom")
    group_sizes = t.groupby(["date","underlying"]).size().rename("group_n").reset_index()
    t = t.merge(group_sizes, on=["date","underlying"])
    t["is_ss_ul"]    = t["underlying"].isin(ss_pack_size.keys())
    t["pack_size"]   = t["underlying"].map(available_pack).fillna(0).astype(int)
    t["consensus_n"] = t["pack_size"].apply(lambda p: min(p, 3) if p >= 2 else 99)  # 99 = never triggers

    n = t["group_n"]
    keep_ss     = t["is_ss_ul"] & (n >= t["consensus_n"]) & (t["price_rank"] == 1)
    keep_normal = (~t["is_ss_ul"]) & (
        (n == 1) |
        ((n == 2) & (t["price_rank"] == 1)) |
        ((n >= 3) & (t["price_rank"] <= 2))
    )
    return t[keep_ss | keep_normal].reset_index(drop=True)


def skip_cheap_then_top2_when_3plus(trades: pd.DataFrame, prices: dict,
                                    top_n: int = 2) -> pd.DataFrame:
    """Per (date, underlying):
       - N == 1: keep
       - N == 2: drop cheapest (classic skip-cheap) if top_n>=2, else keep top-1
       - N >= 3: keep ONLY the `top_n` highest-priced (default 2)
    top_n=1 collapses to strict "one ticker per underlying per day."
    """
    t = trades.copy()
    t["date"]       = pd.to_datetime(t["date"])
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    def asof(row):
        ser = prices.get(row["symbol"])
        if ser is None: return None
        d = row["date"] if row["date"].tz is not None else row["date"].tz_localize("UTC")
        idx = ser.index.searchsorted(d, side="right") - 1
        if idx < 0: return None
        v = ser.iloc[idx]
        return float(v) if pd.notna(v) and v > 0 else None
    t["asof_price"] = t.apply(asof, axis=1)
    group_sizes = t.groupby(["date","underlying"]).size().rename("group_n").reset_index()
    t = t.merge(group_sizes, on=["date","underlying"])
    # rank 1 = most expensive within (date, UL)
    t["price_rank"] = t.groupby(["date","underlying"])["asof_price"].rank(method="first", ascending=False, na_option="bottom")
    n = t["group_n"]
    n_keep = max(1, top_n)
    keep = (
        (n == 1) |                              # singleton — keep
        ((n >= 2) & (t["price_rank"] <= n_keep))  # keep top-N by price
    )
    return t[keep].reset_index(drop=True)


def stats(daily_series: pd.Series, label: str = "") -> dict:
    if daily_series.empty:
        return dict(label=label, trades=0, sharpe=float("nan"),
                    max_dd=float("nan"), calmar=float("nan"), net_pnl=0.0)
    rf = 0.043 / 365
    r  = daily_series / START_EQUITY
    sh = (r.mean() - rf) / r.std() * np.sqrt(365) if r.std() > 0 else float("nan")
    eq = START_EQUITY * (1 + r).cumprod()
    pk = eq.cummax()
    dd = ((eq - pk) / pk).min()
    yrs = (daily_series.index[-1] - daily_series.index[0]).days / 365
    ann = (eq.iloc[-1] / START_EQUITY) ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else float("nan")
    calmar = ann / abs(dd) if dd != 0 else float("nan")
    return dict(label=label, sharpe=sh, max_dd=dd, calmar=calmar,
                net_pnl=daily_series.sum())


def run_hybrid(unfiltered_etfs: list[str], filtered_etfs: list[str], label: str,
               min_volume: float = 0.0, use_ps_filter: bool = False,
               adv_exempt: set | None = None, master: pd.DataFrame | None = None):
    """Run backtest on the union, but apply skip-cheap ONLY to filtered_etfs.
    If min_volume > 0, drop any trade whose trailing-60d ADV is below min_volume.
    If use_ps_filter, apply k*sigma prior-session filter (k = PS_K)."""
    all_etfs = sorted(set(unfiltered_etfs) | set(filtered_etfs))
    cfg = build_cfg(all_etfs, use_ps_filter=use_ps_filter, master=master)
    if use_ps_filter:
        print(f"  PS filters active: k={PS_K}, {len(cfg.prior_session_filters)} symbols gated")
    if cfg.rtg_gap_exclusion:
        print(f"  RTG gap exclusion active: <{cfg.rtg_gap_exclusion_threshold:.0%}, "
              f"{len(cfg.rtg_gap_exclusion_symbols)} Class A symbols")
    with contextlib.redirect_stdout(io.StringIO()):
        raw = run_backtest(cfg)
    if raw.empty:
        print(f"  {label}: no trades"); return {}, raw

    net = P.apply_trade_costs(raw)
    net["date"] = pd.to_datetime(net["date"])

    # Optional volume filter applied BEFORE everything else
    n_before_vol = len(net)
    if min_volume > 0:
        print(f"  static universe gate: recent-60d $ADV >= ${min_volume:,.0f}")
        net, kept_syms, drop_syms, adv_map = apply_static_dollar_adv_filter(
            net, universe=all_etfs, min_dollar_adv=min_volume, window=60, exempt=adv_exempt)
        if adv_exempt:
            print(f"  $ADV exempt (kept regardless): {sorted(adv_exempt)}")
        print(f"  vol gate: {n_before_vol} -> {len(net)} trades ({n_before_vol-len(net)} dropped)")
        print(f"  universe: {len(all_etfs)} -> {len(kept_syms)} instruments")
        if drop_syms:
            drop_syms = sorted(drop_syms, key=lambda x: (float("inf") if pd.isna(x[1]) else -x[1]))
            print(f"  dropped instruments ($ADV recent):")
            for sym, v in drop_syms:
                vshow = f"${v:>12,.0f}" if pd.notna(v) else "      no data"
                print(f"    {sym:6s}  {vshow}")

    if net.empty:
        print("  no trades survive volume filter"); return {}, net

    # Baseline (no prune on anything, post-vol-filter)
    cal = pd.date_range(net["date"].min(), net["date"].max(), freq="D")
    pre_daily = (net["pnl_pct"] * 1_000.0).groupby(net["date"]).sum().reindex(cal, fill_value=0.0)
    s_pre = stats(pre_daily, f"{label} — baseline (no prune)"); s_pre["trades"] = len(net)

    # Split: unfiltered side (keep all) + filtered side (apply skip-cheap)
    unfiltered_set = set(unfiltered_etfs)
    filtered_set   = set(filtered_etfs)
    unfilt_trades = net[net["symbol"].isin(unfiltered_set)].copy()
    filt_trades   = net[net["symbol"].isin(filtered_set)].copy()

    print("  loading price data for skip-cheap...")
    prices = {s: load_daily_close(s) for s in sorted(filt_trades["symbol"].unique())}
    prices = {s: ser for s, ser in prices.items() if not ser.empty}
    # Need master_universe for single-stock UL identification — re-load here
    filt_pruned = skip_cheap_then_top2_when_3plus(filt_trades, prices) if not filt_trades.empty else filt_trades

    if not filt_trades.empty:
        diag = filt_trades.copy()
        diag["date"] = pd.to_datetime(diag["date"])
        diag["underlying"] = diag["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
        gs = diag.groupby(["date","underlying"]).size()
        n1 = (gs == 1).sum(); n2 = (gs == 2).sum(); n3 = (gs == 3).sum(); n4p = (gs >= 4).sum()
        drop_n2 = (gs[gs == 2] - 1).sum()
        drop_n3 = (gs[gs == 3] - 2).sum()
        drop_n4p = (gs[gs >= 4] - 2).sum()
        print(f"  rule: skip-cheap @ N=2, top2 @ N>=3")
        print(f"    group sizes: N=1:{n1}  N=2:{n2}  N=3:{n3}  N>=4:{n4p}")
        print(f"    dropped: N=2:{drop_n2}  N=3:{drop_n3}  N>=4:{drop_n4p}  | total:{drop_n2+drop_n3+drop_n4p}")

    combined = pd.concat([unfilt_trades, filt_pruned], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    post_daily = (combined["pnl_pct"] * 1_000.0).groupby(combined["date"]).sum().reindex(cal, fill_value=0.0)
    s_post = stats(post_daily, f"{label} — hybrid (skip-cheap @ N=2, top2 @ N>=3)")
    s_post["trades"] = len(combined)

    # Per-segment diagnostics
    print(f"  unfiltered (broad+BE) trades kept: {len(unfilt_trades):>5}  P&L ${unfilt_trades['pnl_pct'].sum()*1000:>9,.0f}")
    print(f"  filtered (pairs+crypto) raw→keep:  {len(filt_trades):>5} → {len(filt_pruned):>5}  P&L ${filt_pruned['pnl_pct'].sum()*1000:>9,.0f}")

    return s_pre, s_post, net, combined


def run_universe(active: list[str], label: str) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    cfg = build_cfg(active)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = run_backtest(cfg)
    if raw.empty:
        print(f"  {label}: no trades generated")
        return {}, {}, raw, raw

    net = P.apply_trade_costs(raw)

    # Pre-skip stats (baseline)
    net["date"] = pd.to_datetime(net["date"])
    cal = pd.date_range(net["date"].min(), net["date"].max(), freq="D")
    pre_daily = (net["pnl_pct"] * 1_000.0).groupby(net["date"]).sum().reindex(cal, fill_value=0.0)
    s_pre = stats(pre_daily, f"{label} — baseline (no prune)")
    s_pre["trades"] = len(net)

    # Apply skip-cheap with singletons
    print("  loading price data for skip-cheap...")
    prices = {s: load_daily_close(s) for s in sorted(active)}
    prices = {s: ser for s, ser in prices.items() if not ser.empty}
    pruned = skip_cheap_with_singletons(net, prices)
    post_daily = (pruned["pnl_pct"] * 1_000.0).groupby(pd.to_datetime(pruned["date"])).sum().reindex(cal, fill_value=0.0)
    s_post = stats(post_daily, f"{label} — skip-cheap")
    s_post["trades"] = len(pruned)

    return s_pre, s_post, net, pruned


_DAILY_CACHE          = Path("orb_event_study/cache/daily")
_CRYPTO_HOURLY_CACHE  = Path("orb_event_study/cache/crypto_hourly_yf")
_CRYPTO_UNDERLYINGS   = frozenset({"BTC", "ETH", "SOL", "XRP"})


def overnight_gap_from_hourly(underlying: str) -> pd.Series:
    """Overnight gap for CRYPTO underlyings, computed from hourly bars using
    the actual ET session-boundary window (4:00 PM ET → 9:30 AM ET).

    yfinance daily bars for BTC-USD/ETH-USD/etc. are UTC-day-bounded, which
    makes their 'open' and prior 'close' one minute apart in wall time —
    producing near-zero 'overnight' gaps that never trigger the 2% threshold.
    Sampling hourly bars at ET session boundaries reproduces the ETF's real
    overnight move that a live system observes at 9:30 AM ET.

    Returns Series indexed by tz-naive normalized date, values = gap fraction.
    """
    p = _CRYPTO_HOURLY_CACHE / f"{underlying}_1h.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "timestamp" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    et = df.index.tz_convert("America/New_York")
    df = df.copy()
    df["et"] = et
    df["et_date"] = et.normalize()
    df["et_hour"] = et.hour

    # Reference prices: 16:00 ET yesterday (prior close proxy) and 09:00 ET
    # today (today open proxy — the 09:00 ET bar covers 09:00-10:00 ET, so its
    # open is the price at exactly 09:00 ET, ~30 min before 9:30 open).
    # We use OPEN of each bar since that's the tick at bar-start-timestamp.
    close_bars = df[df["et_hour"] == 16][["et_date", "open"]].copy()
    close_bars = close_bars.rename(columns={"open": "close_at_1600ET"})
    close_bars = close_bars.drop_duplicates(subset="et_date", keep="last")
    close_bars = close_bars.set_index("et_date")

    open_bars = df[df["et_hour"] == 9][["et_date", "open"]].copy()
    open_bars = open_bars.rename(columns={"open": "open_at_0900ET"})
    open_bars = open_bars.drop_duplicates(subset="et_date", keep="first")
    open_bars = open_bars.set_index("et_date")

    joined = open_bars.join(close_bars.shift(1), how="inner")
    joined = joined.dropna()
    gap = (joined["open_at_0900ET"] / joined["close_at_1600ET"] - 1)
    gap.index = pd.to_datetime(gap.index).tz_localize(None).normalize()
    return gap.dropna()


def overnight_gap_from_daily(underlying: str) -> pd.Series:
    """Return overnight gap series for an underlying (signed pct change).
    overnight_gap = today_open / prior_day_close - 1.  Causal at 9:30 ET.

    Uses the daily cache (which has open/close per date) — not the intraday
    cache. The daily cache is the right source for UNDERLYINGS (SPY, QQQ, etc.)
    which we don't trade but need for regime detection.

    For CRYPTO underlyings (BTC, ETH, SOL, XRP), dispatches to
    overnight_gap_from_hourly because daily bars from yfinance are UTC-bounded
    and produce meaningless overnight gaps for 24/7 markets.
    """
    if underlying in _CRYPTO_UNDERLYINGS:
        return overnight_gap_from_hourly(underlying)

    p = _DAILY_CACHE / f"{underlying}.parquet"
    if not p.exists(): return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "open" not in df.columns or "close" not in df.columns: return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return (df["open"] / df["close"].shift(1) - 1).dropna()


def compute_broad_gap_days(min_count: int = 2, threshold: float = 0.01) -> set:
    """Set of dates where >= min_count of SPY/QQQ/IWM/DIA had an OVERNIGHT
    (prior_close → today_open) absolute gap >= threshold.  Causal."""
    per_ul = {}
    for ul in ["SPY", "QQQ", "IWM", "DIA"]:
        gaps = overnight_gap_from_daily(ul)
        if gaps.empty: continue
        idx = gaps[gaps.abs() >= threshold].index
        per_ul[ul] = set(idx.normalize())
    if not per_ul: return set()
    all_dates = set().union(*per_ul.values())
    counts = {d: sum(1 for ul in per_ul if d in per_ul[ul]) for d in all_dates}
    return {d for d, c in counts.items() if c >= min_count}


def compute_broad_gap_days_spy_iwm(threshold: float = 0.01) -> set:
    """Causal regime trigger: BOTH SPY AND IWM gap >= threshold overnight.
    This is the v1 broad-event regime — captures large-cap + small-cap moving
    together (i.e., a real macro day, not sector rotation).  ~470 days
    (29% of active trading days) at threshold=0.01.
    """
    spy = overnight_gap_from_daily("SPY")
    iwm = overnight_gap_from_daily("IWM")
    if spy.empty or iwm.empty: return set()
    spy_days = set(spy[spy.abs() >= threshold].index.normalize())
    iwm_days = set(iwm[iwm.abs() >= threshold].index.normalize())
    return spy_days & iwm_days


# Catalyst taxonomy 2026-06-09: single-stock derivatives + international/EM +
# crypto + biotech (incl MSOS).  Everything else (broad ETFs, gold, UNG, VIX,
# tech/financial sectors, XOP) becomes Class B.
CATALYST_SYMS = {
    # Crypto
    "BITX","BITU","SBIT","BTCZ","BTCL","BTFX",
    "ETHU","ETHT","ETHD","ETH","ETU",
    "SOLT","UXRP","XRPT",
    # Biotech (incl MSOS per user spec)
    "LABU","LABD","MSOX",
    # Natural gas drift (UNG) per user spec
    "KOLD","BOIL",
    # Single-stock derivatives
    "AMDL","AMDG","AMUU",
    "TSLL","TSL","TSLG","TSLI","TSLR","TSLT","TSLW",
    "NVDU","NVDG","NVDL","NVDW","NVDX",
    # International (developed + emerging country-specific)
    "YINN","YANG","KORU","BRZU","INDL","MEXX",
    # Emerging markets broad
    "EDC","EDZ",
}


def apply_2class_regime_sizing(trades: pd.DataFrame, master: pd.DataFrame,
                                 weights: dict | None = None,
                                 cluster_threshold: int = 3,
                                 active_min: int = 4,
                                 flood_min: int = 6,
                                 catalyst_set: set | None = None) -> pd.DataFrame:
    """2-class × 4-regime sizing (8 buckets).

    Classes:
      CH = catalyst (single-stock derivatives, international/EM, crypto, biotech)
           — defaults to CATALYST_SYMS unless catalyst_set passed
      CM = macro (everything else — broad/gold/UNG/VIX/sectors)

    Regimes: same as apply_3class_regime_sizing.
    """
    if trades.empty: return trades
    defaults = {
        ("CH","quiet"):3.0, ("CH","active"):3.0, ("CH","flood"):3.0, ("CH","cluster"):1.5,
        ("CM","quiet"):1.0, ("CM","active"):1.5, ("CM","flood"):2.0, ("CM","cluster"):0.75,
        "purity_skip": 0.0,
    }
    if weights is None: weights = defaults
    else: weights = {**defaults, **weights}

    crypto_uls = set(master[master["class"] == "4_CRYPTO"]["underlying"].unique())
    catalyst = catalyst_set if catalyst_set is not None else CATALYST_SYMS

    t = trades.copy()
    t["date"]       = pd.to_datetime(t["date"])
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
    t["v1_class"]   = t["symbol"].map(lambda s: "CH" if s in catalyst else "CM")
    t["is_crypto"]  = t["underlying"].isin(crypto_uls)
    t["date_norm"]  = t["date"].dt.tz_localize(None).dt.normalize() if t["date"].dt.tz is not None else t["date"].dt.normalize()

    ds = t.groupby("date_norm").agg(
        total_n=("symbol","count"),
        n_uls=("underlying","nunique"),
        n_crypto_uls=("underlying", lambda x: x[t.loc[x.index,"is_crypto"]].nunique()),
    ).reset_index()

    def assign_regime(row):
        if row["n_uls"] >= flood_min:                      return "flood"
        if row["n_uls"] >= active_min:                     return "active"
        if row["n_crypto_uls"] >= cluster_threshold:       return "cluster"
        return "quiet"
    ds["regime"] = ds.apply(assign_regime, axis=1)

    t = t.merge(ds[["date_norm","total_n","n_uls","regime"]], on="date_norm", suffixes=("","_day"))
    t["purity_skip"] = (t["v1_class"] == "CM") & (t["regime"] == "quiet") & (t["total_n"] <= 3)

    def mult(row):
        if row["purity_skip"]: return weights.get("purity_skip", 0.0)
        return weights[(row["v1_class"], row["regime"])]
    t["mult"]      = t.apply(mult, axis=1)
    t["sized_pnl"] = t["dollar_pnl"] * t["mult"]
    return t


def apply_3class_regime_sizing(trades: pd.DataFrame, master: pd.DataFrame,
                                 weights: dict | None = None,
                                 cluster_threshold: int = 3,
                                 active_min: int = 5,
                                 flood_min: int = 8) -> pd.DataFrame:
    """3-class × 4-regime sizing.

    Classes:
      C1 = CLASS_1_SYMS (catalyst — crypto/UNG/MSOX/AMD)
      C2 = CLASS_2_SYMS (partial independence — gold/intl/singles/VIX)
      C3 = everything else (broad / sector pairs / LABU/LABD/etc.)

    Regimes — based on causal n_uls (distinct underlyings with qualifying
    overnight gap, known at 9:30 ET):
      flood    = n_uls >= flood_min (default 6)
      active   = active_min <= n_uls < flood_min (default 4-5)
      cluster  = n_uls < active_min AND n_crypto_uls >= cluster_threshold (default 3)
      quiet    = everything else (n_uls 0-3, not crypto-cluster)

    Priority: flood > active > cluster > quiet.

    purity_skip = C3 singletons on quiet days with total_n <= 3
    """
    if trades.empty: return trades
    defaults = {
        ("C1", "quiet"):   3.0, ("C1", "active"): 3.0, ("C1", "flood"): 3.0, ("C1", "cluster"): 1.5,
        ("C2", "quiet"):   2.0, ("C2", "active"): 2.0, ("C2", "flood"): 2.0, ("C2", "cluster"): 1.0,
        ("C3", "quiet"):   1.0, ("C3", "active"): 1.5, ("C3", "flood"): 2.0, ("C3", "cluster"): 0.75,
        "purity_skip": 0.0,
    }
    if weights is None: weights = defaults
    else: weights = {**defaults, **weights}

    crypto_uls = set(master[master["class"] == "4_CRYPTO"]["underlying"].unique())

    t = trades.copy()
    t["date"]       = pd.to_datetime(t["date"])
    t["underlying"] = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
    def classify(sym):
        if sym in CLASS_1_SYMS: return "C1"
        if sym in CLASS_2_SYMS: return "C2"
        return "C3"
    t["v1_class"] = t["symbol"].map(classify)
    t["is_crypto"] = t["underlying"].isin(crypto_uls)
    t["date_norm"] = t["date"].dt.tz_localize(None).dt.normalize() if t["date"].dt.tz is not None else t["date"].dt.normalize()

    ds = t.groupby("date_norm").agg(
        total_n=("symbol","count"),
        n_uls=("underlying","nunique"),
        n_crypto_uls=("underlying", lambda x: x[t.loc[x.index,"is_crypto"]].nunique()),
    ).reset_index()

    def assign_regime(row):
        if row["n_uls"] >= flood_min:                      return "flood"
        if row["n_uls"] >= active_min:                     return "active"
        if row["n_crypto_uls"] >= cluster_threshold:       return "cluster"
        return "quiet"
    ds["regime"] = ds.apply(assign_regime, axis=1)

    t = t.merge(ds[["date_norm","total_n","n_uls","regime"]], on="date_norm", suffixes=("","_day"))

    # purity_skip: C3 on quiet day with very few total trades
    t["purity_skip"] = (t["v1_class"] == "C3") & (t["regime"] == "quiet") & (t["total_n"] <= 3)

    def mult(row):
        if row["purity_skip"]: return weights.get("purity_skip", 0.0)
        return weights[(row["v1_class"], row["regime"])]
    t["mult"]      = t.apply(mult, axis=1)
    t["sized_pnl"] = t["dollar_pnl"] * t["mult"]
    return t


def apply_class_ab_sizing(trades: pd.DataFrame, master: pd.DataFrame,
                           flood_threshold: int = 10,
                           cluster_threshold: int = 3,
                           purity_threshold: int = 3,
                           flood_losers: set | None = None) -> pd.DataFrame:
    """Production-style Class A / B sizing with user-redefined Class A:
       Class A = binary_event=YES OR class == 4_CRYPTO
       Multipliers:
         purity_skip                : 0.0
         equity_flood + flood_loser : 0.0
         equity_flood               : 1.5 (A) / 0.75 (B)
         crypto_cluster             : 1.0
         quiet                      : 2.5 (A) / 1.0 (B)
    """
    if trades.empty: return trades
    flood_losers = set(flood_losers or set())

    be_map     = {row["etf"]: (str(row.get("binary_event","")).strip().upper() == "YES")
                  for _, row in master.iterrows()}
    cls_map    = dict(zip(master["etf"], master["class"]))
    crypto_uls = set(master[master["class"] == "4_CRYPTO"]["underlying"].unique())

    t = trades.copy()
    t["date"]        = pd.to_datetime(t["date"])
    t["underlying"]  = t["symbol"].map(lambda s: INSTRUMENTS[s]["underlying"])
    t["dollar_pnl"]  = t["pnl_pct"] * 1_000.0
    t["is_class_a"]  = t.apply(lambda r: (r["symbol"] in FORCE_CLASS_A)
                                          or be_map.get(r["symbol"], False)
                                          or cls_map.get(r["symbol"], "") == "4_CRYPTO", axis=1)
    t["is_crypto"]   = t["underlying"].isin(crypto_uls)
    t["is_flood_loser"] = t["symbol"].isin(flood_losers)

    day_stats = t.groupby("date").agg(
        total_n      = ("symbol", "count"),
        n_crypto_uls = ("underlying", lambda x: x[t.loc[x.index, "is_crypto"]].nunique()),
    ).reset_index()
    day_stats["equity_flood"]   = day_stats["total_n"] >= flood_threshold
    day_stats["crypto_cluster"] = day_stats["n_crypto_uls"] >= cluster_threshold
    day_stats["quiet"]          = ~day_stats["equity_flood"] & ~day_stats["crypto_cluster"]
    t = t.merge(day_stats[["date","total_n","equity_flood","crypto_cluster","quiet"]], on="date")
    t["purity_skip"] = ~t["is_class_a"] & (t["total_n"] <= purity_threshold)

    def mult(row):
        if row["purity_skip"]: return 0.0
        if row["equity_flood"] and row["is_flood_loser"]: return 0.0
        if row["equity_flood"]: return 1.5 if row["is_class_a"] else 0.75
        if row["crypto_cluster"]: return 1.0
        if row["quiet"]: return 2.5 if row["is_class_a"] else 1.0
        return 1.0
    t["mult"] = t.apply(mult, axis=1)
    t["sized_pnl"] = t["dollar_pnl"] * t["mult"]
    return t


def report_sizing_attribution(sized: pd.DataFrame, label: str):
    if sized.empty: return
    sized = sized.copy()
    def bucket(r):
        if r["purity_skip"]: return "purity_skip"
        if r["equity_flood"] and r["is_flood_loser"]: return "flood_loser"
        cls = "A" if r["is_class_a"] else "B"
        if r["equity_flood"]:   return f"flood_{cls}"
        if r["crypto_cluster"]: return "crypto_cluster"
        if r["quiet"]:          return f"quiet_{cls}"
        return "other"
    sized["bucket"] = sized.apply(bucket, axis=1)
    print(f"\n  {label} — bucket attribution:")
    print(f"  {'bucket':<15} {'N':>5} {'sum_raw':>10} {'sum_sized':>10} {'mean_raw':>9} {'WR':>5}")
    for b, g in sized.groupby("bucket"):
        wr = (g["sized_pnl"] > 0).mean() * 100
        print(f"  {b:<15} {len(g):>5} ${g['dollar_pnl'].sum():>9,.0f} ${g['sized_pnl'].sum():>9,.0f} "
              f"${g['dollar_pnl'].mean():>8,.2f} {wr:>4.1f}%")


def per_instrument_breakdown(trades: pd.DataFrame, label: str, master: pd.DataFrame):
    if trades.empty: return
    t = trades.copy()
    t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
    etf_to_ul    = dict(zip(master["etf"], master["underlying"]))
    etf_to_class = dict(zip(master["etf"], master["class"]))
    etf_to_be    = {row["etf"]: (str(row.get("binary_event","")).strip().upper() == "YES")
                    for _, row in master.iterrows()}

    rows = []
    for sym, g in t.groupby("symbol"):
        n  = len(g)
        wr = (g["dollar_pnl"] > 0).mean() * 100
        pnl = g["dollar_pnl"].sum()
        per = pnl / n if n else 0.0
        ann_sharpe = (g["dollar_pnl"].mean() / g["dollar_pnl"].std()) * np.sqrt(252) if g["dollar_pnl"].std() > 0 else float("nan")
        rows.append((sym, etf_to_class.get(sym, "?"), etf_to_ul.get(sym, "?"),
                     "BE" if etf_to_be.get(sym) else "  ",
                     n, wr, pnl, per, ann_sharpe))

    df = pd.DataFrame(rows, columns=["sym","class","ul","be","n","wr","pnl","per","sh"])
    df = df.sort_values("pnl", ascending=False).reset_index(drop=True)

    print(f"\n  {label} — per-instrument:")
    print(f"  {'Sym':<6} {'Class':<20} {'UL':<10} {'BE':<3} {'N':>5} {'WR':>5} {'Total P&L':>10} {'$/Tr':>7} {'TrdSh':>6}")
    for _, r in df.iterrows():
        print(f"  {r['sym']:<6} {r['class'][:20]:<20} {str(r['ul'])[:10]:<10} {r['be']:<3} "
              f"{r['n']:>5} {r['wr']:>4.1f}% ${r['pnl']:>9,.0f} ${r['per']:>6,.2f} {r['sh']:>6.2f}")

    # BE subset summary
    be_df = df[df["be"] == "BE"]
    if not be_df.empty:
        print(f"\n  --- BE instruments summary ---")
        print(f"  Total BE instruments: {len(be_df)}, trades: {be_df['n'].sum()}, P&L: ${be_df['pnl'].sum():,.0f}")
        print(f"  Positive P&L: {(be_df['pnl'] > 0).sum()}/{len(be_df)}")
        print(f"  Worst contributors:")
        for _, r in be_df.sort_values("pnl").head(5).iterrows():
            print(f"    {r['sym']:<6} {r['n']:>4} trades  ${r['pnl']:>8,.0f}  ${r['per']:>6,.2f}/tr  WR {r['wr']:>4.1f}%")
        print(f"  Best contributors:")
        for _, r in be_df.sort_values("pnl", ascending=False).head(5).iterrows():
            print(f"    {r['sym']:<6} {r['n']:>4} trades  ${r['pnl']:>8,.0f}  ${r['per']:>6,.2f}/tr  WR {r['wr']:>4.1f}%")


def yearly_breakdown(trades: pd.DataFrame, label: str):
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["year"] = t["date"].dt.year
    t["dollar_pnl"] = t["pnl_pct"] * 1_000.0
    print(f"\n  {label} — year-by-year:")
    print(f"  {'Year':>6} {'Trades':>7} {'Net P&L':>10} {'WR':>6}")
    for y, g in t.groupby("year"):
        cal = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        daily = g.groupby("date")["dollar_pnl"].sum().reindex(cal, fill_value=0.0)
        wr = (g["dollar_pnl"] > 0).mean() * 100
        print(f"  {y:>6} {len(g):>7} ${daily.sum():>9,.0f} {wr:>5.1f}%")


def report(label: str, etfs: list[str], ul_map: dict, skipped: list):
    n_ul = len(ul_map)
    print(f"\n{label}")
    print(f"  Underlyings: {n_ul}  ({sorted(ul_map.keys())})")
    print(f"  Instruments: {len(etfs)}")
    if skipped:
        print(f"  Skipped {len(skipped)}: {[s[0] for s in skipped]}")


if __name__ == "__main__":
    master = load_master()
    master = add_force_includes_to_master(master)
    print(f"Master universe (after force-includes): {len(master)} rows across "
          f"{master['class'].nunique()} classes\n")

    # Hybrid v2: broad raw + binary-event raw + (pairs+crypto non-binary) skip-cheap
    broad_etfs,  broad_ul,  _ = filter_master(master, ["1_BROAD"])

    # Binary-event YES across ALL classes (treat as event-driven, no skip-cheap)
    be_mask = master["binary_event"].astype(str).str.upper().eq("YES")
    be_syms = master.loc[be_mask, "etf"].tolist()
    be_etfs, be_ul, be_skip = filter_master(master[be_mask], list(master["class"].unique()))

    # Skip-cheap bucket: pairs + crypto + force-includes MINUS any binary-event YES
    pc_etfs, pc_ul, _ = filter_master(master, ["2_SECTOR_pair", "4_CRYPTO", "force_include"])
    # Pull XRPT/UXRP/AMDL/TSLL/NVDU into pc_etfs even if their class isn't "force_include"
    extra_force = [s for s in FORCE_INCLUDE if s in master["etf"].values and s not in pc_etfs]
    pc_etfs = sorted(set(pc_etfs) | set(extra_force))
    filter_etfs = [s for s in pc_etfs if s not in set(be_etfs)]

    # Unfiltered bucket = broad + binary-event
    unfiltered_etfs = sorted(set(broad_etfs) | set(be_etfs))

    # Sigma coverage gate (auto-computes for any new underlyings in master_universe.csv)
    print("\n[sigma] verifying coverage for unfiltered bucket...")
    unfiltered_etfs, _ = gate_by_sigma_coverage(unfiltered_etfs, master)
    print("[sigma] verifying coverage for skip-cheap bucket...")
    filter_etfs, _ = gate_by_sigma_coverage(filter_etfs, master)

    print("\nHYBRID v2: broad raw + binary-event raw + (pairs+crypto non-BE) skip-cheap")
    print(f"  Broad (raw):              {len(broad_etfs)} instruments")
    print(f"  Binary-event (raw):       {len(be_etfs)} instruments  {be_etfs}")
    if be_skip:
        print(f"    BE skipped (no data):   {[s[0] for s in be_skip]}")
    print(f"  Pairs+crypto non-BE (sc): {len(filter_etfs)} instruments")
    print(f"  Total universe:           {len(unfiltered_etfs) + len(filter_etfs)} instruments")

    s_pre, s_post, net, combined = run_hybrid(unfiltered_etfs, filter_etfs,
                                               "Hybrid v2 + 3-pick SS + drops + PS k=1.25 + LABU/LABD dir",
                                               min_volume=5_000_000.0,
                                               use_ps_filter=True,
                                               adv_exempt=set(be_etfs) | FORCE_INCLUDE,
                                               master=master)
    all_results = [s_pre, s_post] if s_pre else []
    if not combined.empty:
        per_instrument_breakdown(combined, "Hybrid v2 (BE re-add)", master)
        yearly_breakdown(combined, "Hybrid v2")

        # 3-class × 3-regime sizing (replaces production Class A/B with macro-indep classes)
        sized = apply_3class_regime_sizing(combined, master)
        cal = pd.date_range(sized["date"].min(), sized["date"].max(), freq="D")
        sized_daily = sized.groupby("date")["sized_pnl"].sum().reindex(cal, fill_value=0.0)
        s_sized = stats(sized_daily, "Hybrid v2 + 3-class regime sizing")
        s_sized["trades"] = len(sized[sized["mult"] > 0])
        all_results.append(s_sized)

        # Bucket attribution
        sized["bucket"] = sized.apply(lambda r: "purity_skip" if r["purity_skip"]
                                                    else f"{r['v1_class']}_{r['regime']}", axis=1)
        print(f"\n  3-class regime sizing — bucket attribution:")
        print(f"  {'bucket':<14} {'N':>5} {'raw $':>10} {'sized $':>10} {'mean raw':>9} {'WR':>5}")
        for b, g in sized.groupby("bucket"):
            v = g["dollar_pnl"]; sv = g["sized_pnl"]
            print(f"  {b:<14} {len(g):>5} ${v.sum():>9,.0f} ${sv.sum():>9,.0f} ${v.mean():>8.2f} {(sv>0).mean()*100:>4.1f}%")
        yearly_breakdown(sized.assign(pnl_pct=sized["sized_pnl"]/1000.0), "Hybrid v2 + 3-class sized")

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Variant':<55} {'Trades':>7} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'Calmar':>7} {'Net P&L':>11}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['label']:<55} {r['trades']:>7} {r['sharpe']:>7.3f} "
              f"{r['max_dd']*100:>6.2f}% {r['calmar']:>7.2f} ${r['net_pnl']:>10,.0f}")
