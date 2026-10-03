"""
scripts/sigma_master.py
=======================
Maintains `sigma_master.csv` — full-history sigma per underlying ticker.

Public API
----------
load_sigma_master() -> dict[underlying -> sigma]
ensure_sigmas(underlyings, force=False) -> (sigmas, unresolved)
filter_universe_by_sigma_coverage(etfs, master_universe_df, sigmas) -> (kept, dropped)
verify_or_raise(etfs, master_universe_df) -> sigmas
build_full() -> bulk recompute for everything in master_universe.csv

CLI
---
    python scripts/sigma_master.py            # build / refresh master for all unknowns
    python scripts/sigma_master.py --rebuild  # recompute everything from scratch
"""
import sys, argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.compute_sigma import compute_sigma, _load_from_cache

ROOT          = Path(__file__).parent.parent
MASTER_CSV    = ROOT / "master_universe.csv"
SIGMA_CSV     = ROOT / "sigma_master.csv"
UNRESOLVED_CSV = ROOT / "sigma_unresolved.csv"

SIGMA_COLS = ["underlying", "sigma", "n_obs", "source", "computed_at"]
UNRES_COLS = ["underlying", "reason", "checked_at"]


def load_sigma_master() -> dict[str, float]:
    if not SIGMA_CSV.exists():
        return {}
    df = pd.read_csv(SIGMA_CSV)
    return dict(zip(df["underlying"], df["sigma"]))


def _save_master(df: pd.DataFrame):
    df = df.drop_duplicates(subset=["underlying"], keep="last")
    df = df.sort_values("underlying").reset_index(drop=True)
    df.to_csv(SIGMA_CSV, index=False)


def _save_unresolved(df: pd.DataFrame):
    df = df.drop_duplicates(subset=["underlying"], keep="last")
    df = df.sort_values("underlying").reset_index(drop=True)
    df.to_csv(UNRESOLVED_CSV, index=False)


def ensure_sigmas(underlyings: list[str], force: bool = False, verbose: bool = True) -> tuple[dict, list]:
    """For each underlying, compute sigma if missing. Returns ({ul: sigma}, [unresolved_uls])."""
    existing = pd.read_csv(SIGMA_CSV) if SIGMA_CSV.exists() else pd.DataFrame(columns=SIGMA_COLS)
    unresolved_df = pd.read_csv(UNRESOLVED_CSV) if UNRESOLVED_CSV.exists() else pd.DataFrame(columns=UNRES_COLS)
    have = set(existing["underlying"]) if not existing.empty else set()
    known_unresolved = set(unresolved_df["underlying"]) if not unresolved_df.empty else set()

    todo = []
    for ul in dict.fromkeys(underlyings):
        if force or ul not in have:
            if not force and ul in known_unresolved:
                continue
            todo.append(ul)

    if verbose and todo:
        print(f"[sigma] computing sigmas for {len(todo)} underlying(s)...")

    new_rows = []
    new_unres = []
    for ul in todo:
        try:
            sig, n = compute_sigma(ul, years=None, prefer_cache=True)
            src = "cache" if _load_from_cache(ul) is not None else "yfinance"
            new_rows.append({"underlying": ul, "sigma": sig, "n_obs": n,
                             "source": src, "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if verbose: print(f"  [+] {ul:<45s}  sigma={sig:.4f}  n={n}  src={src}")
        except Exception as e:
            new_unres.append({"underlying": ul, "reason": str(e),
                              "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if verbose: print(f"  [-] {ul:<45s}  UNRESOLVED ({e})")

    if new_rows:
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        _save_master(existing)
    if new_unres:
        unresolved_df = pd.concat([unresolved_df, pd.DataFrame(new_unres)], ignore_index=True)
        _save_unresolved(unresolved_df)

    sigmas = dict(zip(existing["underlying"], existing["sigma"])) if not existing.empty else {}
    unresolved = sorted(set(unresolved_df["underlying"]) if not unresolved_df.empty else set())
    return sigmas, [ul for ul in underlyings if ul in set(unresolved)]


def filter_universe_by_sigma_coverage(etfs: list[str], master_df: pd.DataFrame,
                                       sigmas: dict | None = None) -> tuple[list, list]:
    """Return (kept_etfs, dropped_with_reason). ETF is dropped if its underlying has no sigma."""
    if sigmas is None: sigmas = load_sigma_master()
    have = set(sigmas.keys())
    etf_to_ul = dict(zip(master_df["etf"], master_df["underlying"]))
    kept, dropped = [], []
    for s in etfs:
        ul = etf_to_ul.get(s)
        if ul is None:
            dropped.append((s, "not in master_universe.csv")); continue
        if ul not in have:
            dropped.append((s, f"no sigma for underlying '{ul}'")); continue
        kept.append(s)
    return kept, dropped


def verify_or_load(etfs: list[str], master_df: pd.DataFrame, auto_compute: bool = True,
                    verbose: bool = True) -> tuple[dict, list, list]:
    """Ensure every ETF's underlying has a sigma. Auto-computes missing if auto_compute=True.

    Returns (sigmas_dict, kept_etfs, dropped_etfs_with_reason).
    """
    etf_to_ul = dict(zip(master_df["etf"], master_df["underlying"]))
    needed_uls = sorted(set(etf_to_ul[s] for s in etfs if s in etf_to_ul))

    if auto_compute:
        sigmas, _ = ensure_sigmas(needed_uls, verbose=verbose)
    else:
        sigmas = load_sigma_master()

    kept, dropped = filter_universe_by_sigma_coverage(etfs, master_df, sigmas)
    return sigmas, kept, dropped


def build_full(force: bool = False, verbose: bool = True):
    """Bulk recompute sigmas for every underlying referenced in master_universe.csv."""
    master = pd.read_csv(MASTER_CSV)
    uls = sorted(set(master["underlying"].dropna().astype(str)))
    if verbose: print(f"[sigma] {len(uls)} unique underlyings in master_universe.csv")
    sigmas, _ = ensure_sigmas(uls, force=force, verbose=verbose)

    # Summary
    have = sorted(set(sigmas))
    miss = sorted(set(uls) - set(have))
    if verbose:
        print()
        print(f"[sigma] resolved   : {len(have)}/{len(uls)}")
        print(f"[sigma] unresolved : {len(miss)}/{len(uls)}")
        if miss:
            print(f"  these underlyings have no daily series (need a proxy ticker):")
            for ul in miss: print(f"    - {ul}")

        # Per-ETF coverage
        etf_to_ul = dict(zip(master["etf"], master["underlying"]))
        excluded_etfs = [(s, etf_to_ul[s]) for s in master["etf"]
                          if etf_to_ul[s] not in set(have)]
        print(f"\n[sigma] ETFs excluded by missing sigma: {len(excluded_etfs)}")
        for s, ul in excluded_etfs[:25]:
            print(f"    {s:<8s}  underlying = {ul}")
        if len(excluded_etfs) > 25:
            print(f"    ... and {len(excluded_etfs)-25} more")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Force recompute everything")
    args = ap.parse_args()
    build_full(force=args.rebuild)
