"""Broker margin requirements for leveraged ETFs.

Why this exists: the strategy's exposure cap (CAP_UNITS = 200% of equity) is
not the only constraint on how much the account can hold. FINRA Rule 4210
requires margin on a leveraged ETF to be the ordinary requirement multiplied
by the fund's leverage factor, so a 3x short costs 90% margin per dollar of
notional against an ordinary equity's 30%. On busy days that binds long
before 200% gross does.

Measured on 2020-2026 with the v1 config: margin exceeds equity on 3.9% of
trading days, but those days carry 14.6% of trades and 28.9% of P&L, because
the days that saturate margin are exactly the high-candidate days where the
strategy earns. Ignoring it makes the backtest plan positions live cannot
take.

The practical consequence is that this turns sizing into a knapsack problem:
edge per margin dollar is a different ranking from edge per notional dollar,
so a 2x long (50%) and a 3x short (90%) are not interchangeable even at
equal expected return.

RATES ARE STATUTORY MINIMA. IBKR house requirements can exceed them.
orb_live.execution.position_manager.prewarm_margin already measures the true
per-symbol rate via broker.check_margin(); prefer those when available.
"""

from __future__ import annotations

#: FINRA 4210 base maintenance requirements, before the leverage multiplier.
SHORT_BASE = 0.30
LONG_BASE = 0.25

#: Margin budget as a fraction of equity. 1.0 = the account may commit all of
#: its equity to initial margin. Lower it to leave headroom for adverse moves
#: intraday, which is what actually triggers a margin call.
DEFAULT_MARGIN_BUDGET_PCT = 1.0


def margin_rate(leverage: float, direction: int) -> float:
    """Initial-margin requirement as a fraction of notional.

    leverage  — fund leverage factor, sign-insensitive (3 for a 3x fund).
    direction — +1 long, -1 short. Shorts carry the higher base rate.
    """
    base = SHORT_BASE if direction < 0 else LONG_BASE
    return base * abs(float(leverage or 1.0))


def margin_units(
    equity_units: float,
    budget_pct: float = DEFAULT_MARGIN_BUDGET_PCT,
) -> float:
    """Margin budget expressed in the same units as CAP_UNITS.

    The backtest sizes a base position at $1,000 against $10,000 of equity,
    so one unit is 10% of equity and equity itself is 10.0 units. CAP_UNITS
    = 20.0 is therefore 200% of equity, and a 100% margin budget is 10.0.
    """
    return equity_units * budget_pct


def expected_margin(
    candidates: list[dict],
    weights: dict,
    classify_fn,
    regime: str,
    leverage_by_symbol: dict[str, float],
) -> float:
    """Margin the day's candidate set would consume at unit cap_factor.

    Mirrors the exp_mult calculation it sits beside: causal, computed from
    the candidate set known at 9:30, never from what actually fired.
    """
    total = 0.0
    for c in candidates:
        sym = c["symbol"]
        w = weights.get((classify_fn(sym), regime), 1.0)
        if w <= 0:
            continue
        total += w * margin_rate(leverage_by_symbol.get(sym, 1.0), c["etf_dir"])
    return total
