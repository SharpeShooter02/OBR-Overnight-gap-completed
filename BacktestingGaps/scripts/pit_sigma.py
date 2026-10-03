"""Point-in-time PS-filter sigma (register V48).

std(|daily return|) over every daily close strictly before the session -- the
sigma_master.csv definition without the future. analysis/pit_sigma_test.py:
equity sigma drifts <5%/yr; crypto's early sigma ran 10-40% above full-history.

V49: the prior session's move is open-to-close for equities, so sigma is
std(|close / open - 1|) on the same basis. 24h underlyings (crypto, UTC daily
bars, V45) stay close-to-close. The definition is mirrored in
orb-live-trading v1_strategy.prior_session_leg / point_in_time_sigmas.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

MIN_OBS = 250
PS_24H_ULS = frozenset({"BTC", "ETH", "SOL", "XRP", "LINK", "ADA", "XLM"})   # = live CRYPTO_UNDERLYINGS
_cache: dict[str, pd.Series] = {}


def prior_session_leg(ul: str, prior_rows: pd.DataFrame) -> Optional[tuple[float, float]]:
    """(end, start) prices of the prior session's move from the last two daily bars
    (ascending): (close_t-1, open_t-1), or (close_t-1, close_t-2) for 24h underlyings.
    None (missing bars or open) means the filter allows the trade."""
    if prior_rows is None or len(prior_rows) < 2:
        return None
    last, prev = prior_rows.iloc[-1], prior_rows.iloc[-2]
    start = prev["close"] if ul in PS_24H_ULS else last.get("open")
    end = last["close"]
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not (start > 0) or end != end:
        return None
    return end, start


def _series(ul: str) -> pd.Series:
    if ul not in _cache:
        from orb_backtester import load_daily
        d = load_daily(ul)
        if d is None or len(d) < MIN_OBS + 2:
            _cache[ul] = pd.Series(dtype=float)
        else:
            d = d.set_index("date")
            if ul in PS_24H_ULS:
                r = d["close"].pct_change().abs()
            else:
                r = (d["close"] / d["open"].where(d["open"] > 0) - 1).abs()
            _cache[ul] = r.expanding(MIN_OBS).std().shift(1).dropna()
    return _cache[ul]


def sigma_asof(ul: str, date) -> float | None:
    s = _series(ul)
    if s.empty:
        return None
    i = s.index.searchsorted(pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tz else pd.Timestamp(date), side="right") - 1
    if i < 0:
        return None
    v = float(s.iloc[i])
    return v if np.isfinite(v) else None


class Threshold:
    """k x point-in-time sigma, evaluated per session by the engine's PS filter."""

    def __init__(self, ul: str, k: float):
        self.ul, self.k = ul, k

    def __call__(self, date):
        s = sigma_asof(self.ul, date)
        return None if s is None else s * self.k

    def __repr__(self):
        return f"PIT({self.ul}, k={self.k})"
