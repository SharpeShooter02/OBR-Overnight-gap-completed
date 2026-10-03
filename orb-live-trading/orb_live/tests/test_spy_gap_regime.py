"""The SPY gap sets the regime, and the regime decides whether C1 trades at all.

WEIGHTS gives C1 2.0 in quiet and active but 0.0 in flood, so a SPY gap that
cannot be measured is not a small sizing detail -- it is the difference between
full crypto size and none. _spy_gap previously returned None from five separate
paths while logging only one of them (the exception), so an empty 09:30 bar
frame was indistinguishable in the log from a healthy 0.00% gap.

What this pins down:
  * every unmeasurable path names its reason at critical level;
  * the SPY open comes from the same ref-price path every other symbol uses,
    instead of a bespoke 09:30 fetch;
  * a single empty bar frame is retried before the day is given up on;
  * an unmeasured regime falls back to flood (C1 off), not active (C1 full).
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

TDATE = date(2026, 9, 16)


def _daily(closes, end=date(2026, 9, 15)):
    idx = pd.date_range(end=pd.Timestamp(end), periods=len(closes), freq="D")
    return pd.DataFrame({"date": idx, "close": list(closes)})


def _job(client=None, logger=None):
    from orb_live.config.live_config import load_live_config
    from orb_live.signals.pre_market import PreMarketJob
    return PreMarketJob(load_live_config(), MagicMock(), client or MagicMock(),
                        MagicMock(), logger=logger or MagicMock())


def _reasons(log):
    """Reasons logged by spy_gap_unavailable, critical level only."""
    return [c.kwargs.get("reason") for c in log.critical.call_args_list
            if c.args and c.args[0] == "spy_gap_unavailable"]


# ── a. every None path names itself ───────────────────────────────────────────

class TestUnmeasurableIsLoud:
    def test_no_prior_close(self):
        log = MagicMock()
        job = _job(logger=log)
        # daily frame exists but holds nothing before trade_date
        ul = {"SPY": _daily([500.0], end=TDATE)}
        assert job._spy_gap(TDATE, ul, {"SPY_OPEN": 505.0}) is None
        assert _reasons(log) == ["no_prior_close"]

    def test_no_open_bar(self):
        log = MagicMock()
        client = MagicMock()
        client.get_intraday_bars.return_value = pd.DataFrame()
        job = _job(client=client, logger=log)
        assert job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])}, {}) is None
        assert _reasons(log) == ["no_open_bar"]

    def test_bad_prior_close(self):
        log = MagicMock()
        job = _job(logger=log)
        assert job._spy_gap(TDATE, {"SPY": _daily([495.0, 0.0])}, {"SPY_OPEN": 505.0}) is None
        assert _reasons(log) == ["bad_prior_close"]

    def test_no_daily_data(self):
        log = MagicMock()
        client = MagicMock()
        client.get_daily_bars.return_value = pd.DataFrame()
        job = _job(client=client, logger=log)
        assert job._spy_gap(TDATE, {}, {"SPY_OPEN": 505.0}) is None
        assert _reasons(log) == ["no_daily_data"]

    def test_exception_path_also_names_itself(self):
        log = MagicMock()
        client = MagicMock()
        client.get_daily_bars.side_effect = RuntimeError("IB down")
        job = _job(client=client, logger=log)
        assert job._spy_gap(TDATE, {}, {"SPY_OPEN": 505.0}) is None
        assert _reasons(log) == ["exception"]

    def test_a_measured_gap_logs_nothing(self):
        log = MagicMock()
        job = _job(logger=log)
        gap = job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])}, {"SPY_OPEN": 505.0})
        assert gap == pytest.approx(0.01)
        assert _reasons(log) == []


# ── b. the SPY open uses the shared ref-price path ────────────────────────────

class TestOpenComesFromRefPrices:
    def test_spy_open_key_is_used_directly(self):
        client = MagicMock()
        job = _job(client=client)
        assert job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])},
                            {"SPY_OPEN": 510.0}) == pytest.approx(0.02)
        client.get_intraday_bars.assert_not_called()

    def test_plain_spy_ref_price_is_used(self):
        """run_phase1 fetches reference prices per symbol; SPY must be able to
        ride along in that same dict rather than forcing a separate fetch."""
        client = MagicMock()
        job = _job(client=client)
        assert job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])},
                            {"SPY": 510.0}) == pytest.approx(0.02)
        client.get_intraday_bars.assert_not_called()

    def test_run_phase1_prefetches_the_spy_reference_price(self):
        """SPY is an underlying, not a tradeable symbol, so it is not in
        cfg.symbols -- run_phase1 must still fetch its open the shared way."""
        from orb_live.signals.pre_market import PreMarketJob
        asked: list[str] = []

        cfg = SimpleNamespace(symbols=["TQQQ"], instruments={}, sigmas={},
                              ps_filter_k=1.0, prior_session_filters={})
        job = PreMarketJob(cfg, MagicMock(), MagicMock(), MagicMock(), logger=MagicMock())
        job._get_ref_price = lambda sym, refs, d: asked.append(sym) or 100.0

        job._prefetch_ref_prices(TDATE, {"TQQQ": 100.0})
        assert "SPY" in asked


# ── c. one empty frame is not the end of the day ──────────────────────────────

class TestRetry:
    def test_empty_first_frame_is_retried(self):
        client = MagicMock()
        client.get_intraday_bars.side_effect = [
            pd.DataFrame(),
            pd.DataFrame({"close": [505.0], "open": [505.0]}),
        ]
        job = _job(client=client)
        gap = job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])}, {})
        assert gap == pytest.approx(0.01)
        assert client.get_intraday_bars.call_count == 2

    def test_gives_up_after_the_retries(self):
        log = MagicMock()
        client = MagicMock()
        client.get_intraday_bars.return_value = pd.DataFrame()
        job = _job(client=client, logger=log)
        assert job._spy_gap(TDATE, {"SPY": _daily([495.0, 500.0])}, {}) is None
        assert client.get_intraday_bars.call_count >= 2
        assert _reasons(log) == ["no_open_bar"]


# ── d. an unmeasured regime is the cautious one ───────────────────────────────

class TestFallbackRegime:
    def test_unmeasured_regime_is_flood(self):
        from orb_live.strategy.v1_strategy import assign_regime
        assert assign_regime(None) == "flood"
        assert assign_regime(float("nan")) == "flood"

    def test_measured_gaps_are_unaffected(self):
        from orb_live.strategy.v1_strategy import assign_regime
        assert assign_regime(0.001) == "quiet"
        assert assign_regime(-0.006) == "active"
        assert assign_regime(0.015) == "flood"

    def test_c1_is_flat_when_the_regime_could_not_be_measured(self):
        """The point of the fallback: no SPY gap must not mean full crypto size."""
        from orb_live.strategy.v1_strategy import WEIGHTS, assign_regime
        assert WEIGHTS[("C1", assign_regime(None))] == 0.0
