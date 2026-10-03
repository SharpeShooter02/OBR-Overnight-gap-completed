"""
scripts/compute_sigma.py
========================
Compute sigma (std of daily absolute returns) for any underlying ticker.
Uses yfinance full history by default. Can also load from the project's
daily parquet cache.

Usage:
    from scripts.compute_sigma import compute_sigma, compute_sigmas_batch

    sig = compute_sigma("TSLA")               # full yfinance history
    sig = compute_sigma("TSLA", years=3)      # last 3 years
    sigs = compute_sigmas_batch(["TSLA","IWM","BTC"])

Run directly to print sigmas for a list of tickers:
    python scripts/compute_sigma.py TSLA IWM BTC SPY
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TICKER_ALIASES = {
    "VIX":    "^VIX",
    "NYFANG": "^NYFANG",
    "TLT":    "TLT",        # treasury daily in cache
}

_DAILY_CACHE = Path(__file__).parent.parent / "orb_event_study" / "cache" / "daily"


def _load_from_cache(underlying: str) -> pd.Series | None:
    """Load close prices from local daily parquet cache."""
    path = _DAILY_CACHE / f"{underlying}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "close" not in df.columns or len(df) < 30:
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def _load_from_yfinance(underlying: str, years: float | None) -> pd.Series | None:
    """Download close prices from yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    ticker = TICKER_ALIASES.get(underlying, underlying)
    try:
        if years is None:
            df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        else:
            end   = pd.Timestamp.today()
            start = end - pd.DateOffset(years=years)
            df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
    except Exception:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 30:
        return None
    return df["Close"].dropna()


def compute_sigma(
    underlying: str,
    years: float | None = None,
    prefer_cache: bool = True,
) -> tuple[float, int]:
    """
    Compute sigma = std(daily absolute returns) for an underlying.

    Returns (sigma, n_observations).
    Raises ValueError if fewer than 30 observations are available.

    Parameters
    ----------
    underlying : str
        Ticker symbol (e.g. "TSLA", "IWM", "BTC", "VIX").
    years : float | None
        Lookback window in years. None = full available history.
    prefer_cache : bool
        Try local daily parquet cache first (faster). Falls back to yfinance.
    """
    closes = None

    if prefer_cache:
        closes = _load_from_cache(underlying)

    if closes is None:
        closes = _load_from_yfinance(underlying, years)

    if closes is None or len(closes) < 30:
        raise ValueError(f"Insufficient data for '{underlying}'")

    if years is not None:
        cutoff = closes.index.max() - pd.DateOffset(years=years)
        closes = closes[closes.index >= cutoff]

    returns = closes.pct_change().dropna().abs()
    if len(returns) < 30:
        raise ValueError(f"Fewer than 30 return observations for '{underlying}'")

    return float(returns.std()), len(returns)


def compute_sigmas_batch(
    underlyings: list[str],
    years: float | None = None,
    k: float = 1.0,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Compute sigma for multiple underlyings. Returns {underlying: sigma * k}.

    Parameters
    ----------
    underlyings : list of tickers
    years : lookback in years (None = full history)
    k : multiplier applied to sigma (1.0 = raw sigma threshold)
    verbose : print results as computed
    """
    results = {}
    if verbose:
        print(f"  {'Underlying':<12}  {'Sigma':>7}  {'k*Sigma':>8}  {'N':>6}  Source")
        print("  " + "-" * 50)

    for ul in underlyings:
        try:
            sig, n = compute_sigma(ul, years=years)
            threshold = sig * k
            results[ul] = threshold
            if verbose:
                # Determine source
                src = "cache" if _load_from_cache(ul) is not None else "yfinance"
                print(f"  {ul:<12}  {sig:>7.4f}  {threshold:>8.4f}  {n:>6,}  {src}")
        except ValueError as e:
            if verbose:
                print(f"  {ul:<12}  {'ERROR':>7}  -- {e}")

    return results


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else [
        "SPY", "QQQ", "IWM", "XLF", "XLE", "SOXX",
        "TSLA", "MSTR", "BTC", "ETH", "VIX", "TLT",
    ]
    print(f"\nSigma (std of daily absolute returns) — full history, k=1.0\n")
    compute_sigmas_batch(tickers)
