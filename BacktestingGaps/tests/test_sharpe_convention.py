"""Sharpe annualises on trading days, the standard convention.

CHANGED 2026-08-31 (register V5). Previously `daily_series` reindexed P&L onto
a CALENDAR range -- weekends and holidays inserted as structural zeros -- and
`stats` annualised with sqrt(365).

That was self-consistent and very nearly correct: padding with zeros and scaling
by sqrt(365) is algebraically almost identical to not padding and scaling by
sqrt(252), because padding shrinks the mean by p and the sd by sqrt(p) where
p = 252/365. Measured at this strategy's moments (mu~35, sd~177) the two
conventions differ by 0.54%, with the padded version reading LOWER. So nothing
published under the old convention was inflated.

It is still the wrong convention to publish under. A weekend is not an
observation of zero return, it is an absence of observation, and a reader
reconciling a quoted Sharpe against sqrt(252) would find a discrepancy and have
no way to know it was deliberate.

WHAT MUST NOT CHANGE. Dropping weekend zeros cannot alter max drawdown, annual
return, or Calmar: zeros contribute a factor of 1.0 to the cumulative product,
and the span in calendar days between first and last observation is unchanged.
If any of those move, the reindex has dropped a real trading day. That is what
most of the tests below actually guard.

A trading day on which the strategy took no position IS a real observation of
zero return and stays in the series.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, ".")

from scripts.optimize_v1_3class import (
    START, RF_DAILY, TRADING_DAYS_PER_YEAR, daily_series, stats, trading_days)


def _sized(dates, pnl):
    return pd.DataFrame({"date": pd.to_datetime(dates), "sized_pnl": pnl})


class TestTradingCalendar:
    def test_excludes_weekends(self):
        idx = trading_days(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
        assert not any(d.weekday() >= 5 for d in idx)

    def test_excludes_new_years_day(self):
        idx = trading_days(pd.Timestamp("2025-12-28"), pd.Timestamp("2026-01-05"))
        assert pd.Timestamp("2026-01-01") not in set(idx)

    def test_lands_near_252_per_year(self):
        idx = trading_days(pd.Timestamp("2020-01-01"), pd.Timestamp("2026-05-20"))
        yrs = (idx[-1] - idx[0]).days / 365.25
        assert 249.0 < len(idx) / yrs < 253.0


class TestDailySeries:
    def test_no_weekends_in_the_series(self):
        s = daily_series(_sized(["2026-01-05", "2026-01-09"], [100.0, -50.0]))
        assert not any(d.weekday() >= 5 for d in s.index)

    def test_flat_trading_days_are_kept_as_zero(self):
        """A day the strategy chose not to trade really did return 0."""
        s = daily_series(_sized(["2026-01-05", "2026-01-09"], [100.0, -50.0]))
        assert len(s) == 5                      # Mon..Fri inclusive
        assert s.loc["2026-01-06"] == 0.0

    def test_totals_survive_the_reindex(self):
        s = daily_series(_sized(["2026-01-05", "2026-01-09"], [100.0, -50.0]))
        assert s.sum() == pytest.approx(50.0)


class TestAnnualisation:
    def test_constant_is_252(self):
        assert TRADING_DAYS_PER_YEAR == 252

    def test_risk_free_matches_the_period(self):
        """RF must be per-period, or it is silently mis-scaled by 365/252."""
        assert RF_DAILY == pytest.approx(0.043 / 252)

    def test_sharpe_uses_the_trading_day_factor(self):
        rng = np.random.default_rng(0)
        idx = trading_days(pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01"))
        x = pd.Series(rng.normal(35.0, 177.0, len(idx)), index=idx)
        r = x / START
        expect = (r.mean() - RF_DAILY) / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        assert stats(x)["sharpe"] == pytest.approx(expect)

    def test_shift_from_the_old_convention_is_small_and_upward(self):
        """The old padded/sqrt(365) form read ~0.5% LOW at our moments. The new
        number must therefore be slightly higher, not lower -- a large move in
        either direction means something other than the convention changed."""
        rng = np.random.default_rng(1)
        idx = trading_days(pd.Timestamp("2020-01-01"), pd.Timestamp("2026-05-20"))
        x = pd.Series(rng.normal(35.0, 177.0, len(idx)), index=idx)
        new = stats(x)["sharpe"]

        cal = pd.date_range(idx[0], idx[-1], freq="D")
        padded = x.reindex(cal, fill_value=0.0) / START
        old = ((padded.mean() - 0.043 / 365) / padded.std()) * np.sqrt(365)

        assert new > old
        assert (new / old - 1) < 0.02


class TestUnaffectedStatistics:
    """Drawdown, annual return and Calmar must be identical either way -- a
    zero contributes 1.0 to the cumulative product and moves no date."""

    def _series(self):
        rng = np.random.default_rng(7)
        idx = trading_days(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-01-01"))
        return pd.Series(rng.normal(30.0, 170.0, len(idx)), index=idx)

    @pytest.mark.parametrize("key", ["max_dd", "ann_ret", "calmar", "net_pnl"])
    def test_padding_does_not_move_it(self, key):
        x = self._series()
        cal = pd.date_range(x.index[0], x.index[-1], freq="D")
        assert stats(x)[key] == pytest.approx(stats(x.reindex(cal, fill_value=0.0))[key])
