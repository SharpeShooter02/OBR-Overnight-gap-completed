"""The stop sits one full ORB range from the entry boundary.

CHANGED 2026-08-30, from 0.75 to 1.00 (register V30).

The old geometry was written as `(midpoint + low) / 2`, which is algebraically
`orb_high - 0.75 x orb_range` but does not look like it -- and that opacity is
part of why the parameter went unexamined. It is now expressed as an explicit
ORB-range fraction so the number is visible and testable.

At 1.00 the stop lands exactly on the OPPOSITE ORB boundary: a long entered on
a break of the high stops at the low. That is a meaningful property, not a
coincidence of the fit -- it means the trade is wrong only once the whole
opening range has been retraced.

WHY 1.00. Under the adopted weight matrix, in-sample: P&L 22,687 -> 24,790,
Sharpe 1.77 -> 1.85, win days 50% -> 53%, but MaxDD -9.84% -> -11.32%. Three
metrics better, drawdown worse. Out-of-sample over 47 sessions it led on all
four (P&L +367 vs +9, MaxDD -7.47% vs -8.02%).

WHAT THIS IS NOT. The out-of-sample gap is NOT significant -- V35 measured the
same comparison at paired p=0.605, worse on 31 of 55 differing days, so the
aggregate sign comes from a few large days. This is a decision taken on an
in-sample trade-off with a non-contradicting holdout, not a validated result.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from orb_live.signals.strategy_signals import STOP_ORB_DISTANCE, compute_entry


def _cfg():
    """The real StrategyConfig, not a stub -- compute_entry reads a dozen
    fields and a hand-rolled namespace only discovers that one AttributeError
    at a time."""
    from dataclasses import replace
    from orb_live.config.strategy_config import StrategyConfig
    return replace(StrategyConfig(), entry_at_boundary=True,
                   tp1_target_multiple=2.0, tp2_target_multiple=3.0)


ORB = {"high": 110.0, "low": 100.0, "midpoint": 105.0}


def _levels(direction):
    return compute_entry(
        orb=ORB,
        bar=pd.Series({"open": 110.0, "high": 111.5, "low": 109.8,
                       "close": 110.2}, name=datetime(2022, 1, 7, 10, 5)),
        gap_direction=direction,
        config=_cfg(), current_equity=100_000.0, v1_base_notional=1000.0)


class TestStopDistance:
    def test_constant_is_one_orb_range(self):
        assert STOP_ORB_DISTANCE == 1.00

    def test_long_stop_is_the_opposite_boundary(self):
        """A long off the high is wrong only once the whole range is retraced."""
        assert _levels(1)["stop_price"] == pytest.approx(ORB["low"])

    def test_short_stop_is_the_opposite_boundary(self):
        assert _levels(-1)["stop_price"] == pytest.approx(ORB["high"])

    @pytest.mark.parametrize("direction", [1, -1])
    def test_stop_is_the_wrong_side_of_entry(self, direction):
        r = _levels(direction)
        if direction == 1:
            assert r["stop_price"] < r["entry_price"]
        else:
            assert r["stop_price"] > r["entry_price"]

    def test_distance_is_the_constant_times_the_range(self):
        r = _levels(1)
        rng = ORB["high"] - ORB["low"]
        assert abs(r["entry_price"] - r["stop_price"]) == pytest.approx(
            STOP_ORB_DISTANCE * rng)

    def test_the_old_geometry_is_no_longer_used(self):
        """(midpoint + low) / 2 was 0.75. If a refactor reintroduces it this
        fails rather than silently reverting the decision."""
        assert _levels(1)["stop_price"] != pytest.approx(
            (ORB["midpoint"] + ORB["low"]) / 2.0)
