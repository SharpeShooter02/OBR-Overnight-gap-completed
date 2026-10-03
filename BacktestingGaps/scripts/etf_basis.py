"""Candidate selection on the ETF gap basis — what live actually does.

The existing `compute_causal_candidates` measures the *underlying's* real
overnight gap and tests it at a flat 2%. Its comment explains why that was
believed equivalent to the engine's per-instrument `leverage x 2%` test:

    |ETF gap| == |UL gap| x leverage

That identity is false. Tracking error, decay, a different crypto
reconstruction, and any stale or misaligned bar on either side all break it,
and the wedge is not small — it accounted for 780 of 2,386 trades.

More to the point, live cannot use the underlying's gap even if it wanted to.
At 09:31 there is no settled daily bar for GDX; there is a print for GDXU. So
live measures the ETF's own gap against `inst.leverage * GAP_THRESHOLD` and
*reconstructs* a UL-equivalent as `etf_gap_signed / leverage`
(`orb_live/signals/pre_market.py:162,195`). This module does the same, so the
backtest selects from the population live can actually trade.

The prior-session filter still reads the underlying's real closes, because
that is what live does too — daily UL data is settled by the next morning.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from scripts.skip_cheap_by_class import filter_master, DIRECTION_FILTERS
from scripts.sigma_master import load_sigma_master
from scripts.run_final_config_causal import build_universe, load_sym_close, load_ul_close
from orb_event_study.config import INSTRUMENTS

CACHE_DIR = Path("orb_event_study/cache/intraday")

#: The reference print. Live uses the close of the 09:30 bar rather than the
#: open field, because the raw open can be a single odd-lot print.
REF_HOUR, REF_MINUTE = 9, 30

#: Maximum calendar days between a session and the prior session used for its
#: End of the trustworthy sample. After 2026-05-20 the intraday cache stops
#: covering the universe: 176 symbols in 2026-05, 32 in 2026-06, 20 in 2026-08,
#: with a 12-trading-day blackout (2026-05-28..06-15) in between. What survives
#: is not a thinner draw from the same population -- pull_ibkr_last_2mo.py
#: backfilled only (symbol, date) pairs that could plausibly fire, so coverage
#: is conditioned on the signal. The window is thin (90 candidates over 33 days
#: from 18 symbols), crypto-weighted (C1 goes from 12.8% of candidates to 35.6%
#: as C3's stale underlyings drop out) and biased toward firing (P(fire) .567
#: vs .432 pooled, elevated within every class).
#:
#: Truncating costs 3 months of a 6.5-year sample and buys back a population
#: that means the same thing end to end. See OPEN_ISSUES D1.
SAMPLE_END = pd.Timestamp("2026-05-20")

#: gap. The intraday cache has holes, and shift(1) happily pairs a session with
#: whichever one precedes it in the file — across a hole that fabricates a gap
#: out of a week of ordinary drift (GDXU 2026-08-07 reads as +21.3% this way).
#: Live never sees this: it reads a real prior daily close from the broker.
#: 5 covers weekends and holiday closures; anything wider is a cache hole.
MAX_PRIOR_GAP_DAYS = 5

#: Sessions whose gap exceeds this are treated as data errors, not gaps. The
#: intraday cache stores raw prices, so a split leaves the prior close on one
#: side of the ratio and the 09:30 print on the other: SOXS 2020-08-28 reads as
#: +13,813%. SQQQ, SOXS, BOIL, DUST and FNGD reverse-split repeatedly.
#: 31 in-sample sessions of 72,542 (0.04%) are dropped by this, all in those
#: names. It is a guard, not a fix — the fix is split-adjusting the intraday
#: ETF cache the way apply_split_correction.py did the daily UL cache.
#: Left deliberately loose: a 3x fund on a violent underlying day can legitimately
#: gap 30%, so cutting nearer the real distribution would discard live trades.
MAX_PLAUSIBLE_GAP = 0.50

GAP_THRESHOLD = 0.02

#: Fallback when master_universe.csv carries no measured rate for a symbol
#: (BOIL, GUSH and KOLD are FORCE_INCLUDE entries with no CSV row).
from scripts.margin_model import margin_rate as _finra_rate


def load_margin_rates(master) -> dict:
    """{(symbol, side): initial-margin rate} from the measured CSV columns.

    side is +1 long / -1 short. Rates differ by side and the difference is not
    a constant: buying SQQQ costs 0.79 and selling it 0.95, while TSLL is 1.00
    long and 0.72 short. Measured by orb_live/scripts/measure_universe_margin.py
    against IB whatIf; see that script for why the FINRA model is not enough.
    """
    out = {}
    if "margin_init_long" not in master.columns:
        return out
    for _, row in master.iterrows():
        sym = str(row["etf"]).strip()
        for side, col in ((1, "margin_init_long"), (-1, "margin_init_short")):
            v = row.get(col)
            if pd.notna(v) and float(v) > 0:
                out[(sym, side)] = float(v)
    return out


def rate_for(sym: str, etf_dir: int, rates: dict, lev: float) -> float:
    """Measured rate if we have one, else the FINRA approximation."""
    r = rates.get((sym, etf_dir))
    return r if r is not None else _finra_rate(lev, etf_dir)


def load_etf_gaps(symbol: str) -> pd.DataFrame:
    """Per-session ETF gap, measured the way the live scanner measures it.

    Returns columns [date, ref_price, prior_close, prior_date, gap_signed],
    one row per session that has a 09:30 bar and a prior session close within
    MAX_PRIOR_GAP_DAYS. Empty frame if the symbol has no intraday cache.
    """
    sym_dir = CACHE_DIR / symbol
    if not sym_dir.exists():
        return pd.DataFrame(columns=["date", "ref_price", "prior_close", "gap_signed"])

    frames = []
    for pq in sorted(sym_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(pq)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "ref_price", "prior_close", "gap_signed"])

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # US sessions sit inside one UTC calendar day, so normalising in UTC gives
    # the session date — the convention the rest of this repo's loaders use.
    df["date"] = df["timestamp"].dt.normalize().dt.tz_localize(None)

    et = df["timestamp"].dt.tz_convert("America/New_York")
    ref = df[(et.dt.hour == REF_HOUR) & (et.dt.minute == REF_MINUTE)]
    ref = ref.drop_duplicates(subset="date", keep="first").set_index("date")["close"]

    # Prior session's last print, from the same source as the reference, so a
    # gap can never be manufactured by mixing two vendors' price scales.
    session_close = df.groupby("date")["close"].last().sort_index()
    # Official auction close where it exists, matching live (register V42).
    from orb_backtester import official_prior_close
    session_close = official_prior_close(symbol, session_close)
    prior = pd.DataFrame({
        "prior_close": session_close.shift(1),
        "prior_date": session_close.index.to_series().shift(1),
    })

    out = pd.DataFrame({"ref_price": ref}).join(prior, how="inner")
    out = out[(out["ref_price"] > 0) & (out["prior_close"] > 0)]
    # Reject pairs straddling a cache hole — see MAX_PRIOR_GAP_DAYS.
    span = (out.index - pd.to_datetime(out["prior_date"])).dt.days
    out = out[span <= MAX_PRIOR_GAP_DAYS]
    out["gap_signed"] = (out["ref_price"] - out["prior_close"]) / out["prior_close"]
    # Split artifacts — see MAX_PLAUSIBLE_GAP.
    out = out[out["gap_signed"].abs() <= MAX_PLAUSIBLE_GAP]
    # Sample bound - see SAMPLE_END. Applied here so every harness that
    # builds candidates through this function truncates identically.
    if SAMPLE_END is not None:
        out = out[out.index <= SAMPLE_END]
    return out.reset_index()


def _ps_passes(ul: str, date, gap_sign: int, threshold: float,
               ul_close_cache: dict) -> bool:
    """Direction-adjusted prior-session filter on the underlying's real closes.

    Missing data allows the trade, matching both the engine and live.
    """
    from scripts.pit_sigma import prior_session_leg
    if ul not in ul_close_cache:
        # Same daily bars as the engine and live (crypto: yfinance UTC daily,
        # register V45). load_ul_close's 16:00-ET crypto closes disagreed with
        # the engine on 10% of crypto gap days, and the book took the AND of both.
        # V49: the frame keeps open, since the prior session is open-to-close.
        from orb_backtester import load_daily
        d = load_daily(ul)
        if d is None or d.empty:
            d = pd.DataFrame(columns=["date", "open", "close"])
        else:
            d = d.copy()
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        ul_close_cache[ul] = d
    df = ul_close_cache[ul]
    if df.empty:
        return True
    leg = prior_session_leg(ul, df[df["date"] < date].tail(2))
    if leg is None:
        return True
    end, start = leg
    return ((end - start) / start) * gap_sign <= threshold


# Optional (ul, date) -> sigma override for point-in-time tests; None = sigma_master.csv.
SIGMA_ASOF = None

_ADV_CACHE: dict = {}


def _adv_asof(sym: str, d) -> float:
    """Trailing dollar ADV for `sym` as of `d`, lagged one session.

    Same series the liquidity gate uses, so a sibling ranked here on liquidity
    is ranked on the number that decides whether it is tradeable at all.
    """
    import pandas as _pd
    if sym not in _ADV_CACHE:
        from scripts.skip_cheap_by_class import build_pit_dollar_adv
        _ADV_CACHE.update(build_pit_dollar_adv([sym], 60))
        _ADV_CACHE.setdefault(sym, _pd.Series(dtype=float))
    ser = _ADV_CACHE.get(sym)
    if ser is None or len(ser) == 0:
        return 0.0
    idx = ser.index
    dd = _pd.Timestamp(d)
    if idx.tz is not None:
        dd = dd.tz_localize("UTC") if dd.tz is None else dd.tz_convert("UTC")
        idx = idx.tz_convert("UTC")
    elif dd.tz is not None:
        dd = dd.tz_localize(None)
    prior = ser[_pd.DatetimeIndex(idx) <= dd]
    return float(prior.iloc[-1]) if len(prior) else 0.0


def compute_candidates_etf_basis(master, k, prune_all: bool = False,
                                 top_n: int = 2, return_detail: bool = False,
                                 gap_tables: dict | None = None,
                                 select_by: str = "price",
                                 max_rate: float | None = None,
                                 sigma_mode: str = "static"):
    """Drop-in replacement for compute_causal_candidates, on the ETF basis.

    Same return shape: {date -> [symbol]}, or the per-candidate detail dict
    when return_detail=True.

    gap_tables — optional {symbol: DataFrame} from load_etf_gaps, to avoid
        re-reading the intraday cache across repeated runs.
    select_by — which sibling to keep when several ETFs track one underlying:

        "price"   the locked v1 rule: the most expensive by prior close.
        "advol"   the most liquid, by point-in-time trailing dollar volume.
                  Basis-invariant, so it cannot be decided by future splits
                  the way "price" on an adjusted cache is (register V43).
        "gap"     the sibling that gapped furthest, |etf_gap|, measured from
                  each ETF's own prices rather than inferred from the UL.
        "gapn"    |etf_gap| / leverage -- furthest relative to its own leverage.
        "lev"     the largest ETF gap. Siblings share one UL gap and
                  etf_gap = ul_gap x leverage, so this is exactly "the most
                  leveraged sibling" -- ranking on gap alone always ties.
        "margin"  the cheapest way to express the same directional view,
                  by measured initial-margin rate, ties broken by price.
                  MEASURED BADLY: -40% net P&L. Long is cheaper than short for
                  52 of 55 symbols (median spread 0.157), so this degenerates
                  into "never short", and shorts carry ~1.5x the edge per
                  margin dollar. It buys cheap margin with expensive edge.

    max_rate  — with select_by="price", drop siblings whose rate exceeds this
                before applying the price rule, falling back to the full group
                only if none qualify. This is the narrow version of the margin
                idea and the one that survives: keep skip-cheap everywhere, and
                override it only where the price pick cannot actually be
                filled. ETH gapping down picks a short ETHU at 4.09x, which
                needs $39k of margin against $32k of available funds — not an
                expensive trade, an impossible one. ETU, BTCL and BTCZ are the
                same. None can ever fill at weight 3.0, so the backtest has
                been booking P&L on trades live could not place.

    Why "margin" is not merely a tweak: a long and a short expressing the SAME
    underlying move cost different margin. GDX gapping down can be short NUGT
    (0.62) or long DUST (0.53). It also disarms the worst case in the universe
    — ETH gapping down is short ETHU at 4.09x or long ETHD at 1.02x, and only
    the latter can actually be filled at full size.
    """
    sigmas = load_sigma_master()
    all_etfs = build_universe(master)
    lev_by_sym = {s: float(INSTRUMENTS[s].get("leverage", 1) or 1)
                  for s in all_etfs if s in INSTRUMENTS}

    if gap_tables is None:
        gap_tables = {s: load_etf_gaps(s) for s in all_etfs if s in INSTRUMENTS}

    margin_rates = load_margin_rates(master)

    # ── Step 1: per-ETF gap qualification, then reconstruct the UL gap ────────
    # Only ETFs clearing their OWN leverage x 2% test contribute, exactly as in
    # live, where nothing else reaches overnight_gaps.
    passed: dict[tuple, float] = {}          # (date, symbol) -> reconstructed ul_gap
    ul_gap: dict[tuple, float] = {}          # (date, ul)     -> most extreme
    for sym, tbl in gap_tables.items():
        if tbl.empty:
            continue
        info = INSTRUMENTS[sym]
        ul = info["underlying"]
        lev = lev_by_sym.get(sym, 1.0) or 1.0
        is_inverse = info.get("inverse", False)

        qual = tbl[tbl["gap_signed"].abs() >= lev * GAP_THRESHOLD]
        for d, g in zip(qual["date"], qual["gap_signed"]):
            recon = (-g if is_inverse else g) / lev
            passed[(d, sym)] = recon
            prior = ul_gap.get((d, ul))
            # Most extreme wins when siblings disagree — live's rule.
            if prior is None or abs(recon) > abs(prior):
                ul_gap[(d, ul)] = recon

    # ── Step 2: PS filter per (date, UL), on the underlying's real closes ─────
    ul_close_cache: dict = {}
    ul_ok: dict[tuple, float] = {}
    for (d, ul), gap in ul_gap.items():
        if abs(gap) < GAP_THRESHOLD:      # cannot fire, but mirrors live's re-check
            continue
        if k > 0 and ul in sigmas:
            if SIGMA_ASOF is not None:
                sig = SIGMA_ASOF(ul, d)
            elif sigma_mode == "expanding":
                from scripts.pit_sigma import sigma_asof
                sig = sigma_asof(ul, d)
            else:
                sig = sigmas[ul]
            if sig is not None and not _ps_passes(ul, d, 1 if gap > 0 else -1, sig * k, ul_close_cache):
                continue
        ul_ok[(d, ul)] = gap

    # ── Step 3: direction filter, per surviving ETF ───────────────────────────
    cand_by_date = defaultdict(list)
    for (d, sym), _recon in passed.items():
        info = INSTRUMENTS[sym]
        ul = info["underlying"]
        gap = ul_ok.get((d, ul))
        if gap is None:
            continue
        ul_sign = 1 if gap > 0 else -1
        etf_dir = -ul_sign if info.get("inverse", False) else ul_sign
        if sym in DIRECTION_FILTERS and DIRECTION_FILTERS[sym] != etf_dir:
            continue
        cand_by_date[d].append({"symbol": sym, "underlying": ul, "etf_dir": etf_dir})

    # ── Step 4: skip-cheap pruning per (date, UL) ─────────────────────────────
    broad_etfs, _, _ = filter_master(master, ["1_BROAD"])
    be_mask = master["binary_event"].astype(str).str.upper().eq("YES")
    be_etfs, _, _ = filter_master(master[be_mask], list(master["class"].unique()))
    no_prune = set() if prune_all else (set(broad_etfs) | set(be_etfs))

    sym_close: dict = {}

    def get_price(sym, date):
        if sym not in sym_close:
            sym_close[sym] = load_sym_close(sym)
        ser = sym_close[sym]
        if ser.empty:
            return 0.0
        hist = ser[ser.index < date]
        return float(hist.iloc[-1]) if not hist.empty else 0.0

    final, detail = {}, {}
    for d, cands in cand_by_date.items():
        keep = []
        by_ul = defaultdict(list)
        for c in cands:
            by_ul[c["underlying"]].append(c)
        for ul, grp in by_ul.items():
            if any(c["symbol"] in no_prune for c in grp):
                keep.extend(c["symbol"] for c in grp)
                continue
            if len(grp) == 1:
                keep.append(grp[0]["symbol"])
                continue
            for c in grp:
                c["price"] = get_price(c["symbol"], d)
                c["rate"] = rate_for(c["symbol"], c["etf_dir"], margin_rates,
                                     lev_by_sym.get(c["symbol"], 1.0))
            if select_by == "margin":
                # Cheapest margin first; price only breaks ties.
                grp.sort(key=lambda x: (round(x["rate"], 4), -x["price"]))
            elif select_by in ("gap", "gapn"):
                if max_rate is not None:
                    fillable = [c for c in grp if c["rate"] <= max_rate]
                    if fillable:
                        grp = fillable
                # "gap"  raw |etf_gap|: the sibling that moved furthest in
                #        absolute terms. Dominated by leverage, but NOT identical
                #        to it -- each ETF's gap is measured from its own prices,
                #        so tracking error and differing prior closes separate
                #        siblings. Measured spread in leverage-normalised gap
                #        within a group: median 1.026, p90 1.112, p99 1.537.
                # "gapn" the same divided by leverage: which sibling moved most
                #        RELATIVE to what its leverage implies. Picks a different
                #        winner from "gap" in 3.5% of groups.
                # `etf_gap` is not on the candidate yet -- it is attached
                # further down when `detail` is built. Reading c["etf_gap"]
                # here silently yields 0.0 for every sibling, the price
                # tie-break then decides, and the selector becomes a no-op:
                # measured, it picked identically to "price" on 2,715 of 2,715
                # UL-days. `passed` holds the reconstructed UL gap per
                # (date, symbol), which is the honest per-sibling number.
                for c in grp:
                    lv = abs(float(lev_by_sym.get(c["symbol"], 1.0))) or 1.0
                    ulg = abs(float(passed.get((d, c["symbol"]), 0.0)))
                    c["_gap"] = ulg * lv if select_by == "gap" else ulg
                grp.sort(key=lambda x: (-x["_gap"], -x["price"]))
            elif select_by in ("advol", "lev"):
                if max_rate is not None:
                    fillable = [c for c in grp if c["rate"] <= max_rate]
                    if fillable:
                        grp = fillable
                if select_by == "advol":
                    # Most liquid sibling. Basis-invariant: price x volume
                    # cancels any split adjustment, so unlike "price" this
                    # cannot be decided by splits after the trade date (V43).
                    for c in grp:
                        c["advol"] = _adv_asof(c["symbol"], d)
                    grp.sort(key=lambda x: (-(x["advol"] or 0.0), -x["price"]))
                else:
                    # Largest ETF gap. Every sibling shares one UL gap, and
                    # etf_gap = ul_gap x leverage, so this IS "most leveraged"
                    # -- ranking on the gap itself would be a tie every time.
                    for c in grp:
                        c["lev"] = abs(float(lev_by_sym.get(c["symbol"], 1.0)))
                    grp.sort(key=lambda x: (-x["lev"], -x["price"]))
            else:
                if max_rate is not None:
                    fillable = [c for c in grp if c["rate"] <= max_rate]
                    # Only narrow the field when something survives; if every
                    # sibling is unfillable, keep the price pick rather than
                    # inventing a preference between two impossible trades.
                    if fillable:
                        grp = fillable
                grp.sort(key=lambda x: -x["price"])
            keep.extend(c["symbol"] for c in grp[:max(1, top_n)])
        kept = set(keep)
        final[d] = keep
        detail[d] = [{"symbol": c["symbol"], "underlying": c["underlying"],
                      "etf_dir": c["etf_dir"],
                      "ul_gap": ul_ok.get((d, c["underlying"])),
                      "rate": c.get("rate"),
                      "price": c.get("price"),
                      "etf_gap": passed.get((d, c["symbol"]), 0.0) * lev_by_sym.get(c["symbol"], 1.0)}
                     for c in cands if c["symbol"] in kept]
    if return_detail:
        return detail
    return final


def truncate_trades(trades: pd.DataFrame,
                    end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Trim a trade frame to the trustworthy sample (see SAMPLE_END).

    Candidates are bounded inside load_etf_gaps, but cached trade parquets were
    produced before the bound existed, so the trade side has to be trimmed too
    or the two disagree about which days exist.
    """
    end = SAMPLE_END if end is None else end
    if end is None or trades.empty:
        return trades
    d = pd.to_datetime(trades["date"]).dt.tz_localize(None)
    return trades[d <= end].copy()
