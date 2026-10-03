"""V48 parity: point-in-time σ, SPY regime, opening-liquidity rule.

Each test pins live to the backtest definition in BacktestingGaps
(scripts/pit_sigma.py, orb_backtester.passes_cost_to_reward /
model_slippage_bps, analysis/refit_sleeve_weights_spy.spy_regimes).
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from orb_live.signals.cost_reward import (
    MAX_COST_TO_REWARD, check_cost_to_reward, load_model, model_slippage_bps,
)
from orb_live.signals.gap_scan import scan_gaps
from orb_live.signals.strategy_signals import compute_opening_range
from orb_live.strategy import v1_strategy as v1


def _daily(closes, end="2026-09-15", opens=None):
    idx = pd.bdate_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=len(closes))
    if opens is None:                       # no overnight gap: open == previous close
        opens = [closes[0]] + list(closes[:-1])
    return pd.DataFrame({"date": idx, "open": opens, "close": closes})


class TestPointInTimeSigma:
    def test_matches_backtest_definition(self):
        """V49: equities' sigma is std(|close / open - 1|), the prior session's own move."""
        rng = np.random.default_rng(1)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        opens = closes * (1 + rng.normal(0, 0.004, 400))
        df = _daily(closes, opens=opens)
        got = v1.point_in_time_sigmas({"QQQ": df}, date(2026, 9, 15))["QQQ"]
        expected = pd.Series(np.abs(closes / opens - 1)).std()
        assert got == pytest.approx(expected, rel=1e-12)

    def test_24h_underlyings_stay_close_to_close(self):
        rng = np.random.default_rng(2)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 400))
        opens = closes * (1 + rng.normal(0, 0.01, 400))
        df = _daily(closes, opens=opens)
        got = v1.point_in_time_sigmas({"BTC": df}, date(2026, 9, 15))["BTC"]
        assert got == pytest.approx(pd.Series(closes).pct_change().abs().dropna().std(), rel=1e-12)

    def test_24h_set_matches_the_data_layer(self):
        from orb_live.data.underlying_data import CRYPTO_UNDERLYINGS
        assert v1.PS_24H_ULS == CRYPTO_UNDERLYINGS

    def test_excludes_the_trade_date_and_later(self):
        closes = list(np.linspace(100, 140, 300))
        df = _daily(closes + [10.0], end="2026-09-16")   # last row dated 2026-09-15
        a = v1.point_in_time_sigmas({"X": df}, date(2026, 9, 15))["X"]
        b = v1.point_in_time_sigmas({"X": _daily(closes)}, date(2026, 9, 15))["X"]
        assert a == pytest.approx(b)

    def test_short_history_keeps_fallback(self):
        df = _daily(list(np.linspace(100, 110, 50)))
        out = v1.point_in_time_sigmas({"X": df}, date(2026, 9, 15), fallback={"X": 0.02})
        assert out["X"] == 0.02


class TestGapScanUsesSessionSigma:
    def _run(self, sigmas, k):
        inst = {"TQQQ": v1.Instrument("TQQQ", "QQQ", 3, False)}
        ul = _daily([100.0] * 10 + [100.0, 101.0])       # prior session +1.0%
        etf = _daily([50.0] * 5)
        cfg = type("C", (), {"direction_filters": {},
                             "prior_session_filters": {"TQQQ": ("QQQ", 0.05)}})()
        return scan_gaps(date(2026, 9, 15), symbols=["TQQQ"], instruments=inst,
                         ref_prices={"TQQQ": 54.0}, etf_daily={"TQQQ": etf},
                         ul_daily={"QQQ": ul}, config=cfg, sigmas=sigmas, k=k).scans[0]

    def test_static_threshold_without_sigmas(self):
        s = self._run(None, None)
        assert s.ps_passed and s.threshold_pct == 0.05

    def test_session_sigma_times_k_overrides(self):
        s = self._run({"QQQ": 0.012}, 0.5)          # 0.6% < 1.0% prior move
        assert s.threshold_pct == pytest.approx(0.006)
        assert not s.ps_passed and s.filter_reason == "ps_filter"

    def _run_weekend(self, ul_name):
        """Gap down into a session whose prior session (Monday) was flat open-to-close
        after a -2.9% weekend gap: close-to-close reads -2.9%, open-to-close reads 0."""
        inst = {"TQQQ": v1.Instrument("TQQQ", ul_name, 3, False)}
        n = 12
        opens, closes = [100.0] * n, [100.0] * n
        opens[-2], closes[-2] = 103.0, 103.0        # Friday
        opens[-1], closes[-1] = 100.0, 100.0        # Monday: gapped to 100, flat all day
        ul = _daily(closes, opens=opens)
        etf = _daily([50.0] * 5)
        cfg = type("C", (), {"direction_filters": {},
                             "prior_session_filters": {"TQQQ": (ul_name, 0.05)}})()
        return scan_gaps(date(2026, 9, 15), symbols=["TQQQ"], instruments=inst,
                         ref_prices={"TQQQ": 46.0}, etf_daily={"TQQQ": etf},   # -8% gap
                         ul_daily={ul_name: ul}, config=cfg, sigmas={ul_name: 0.012}, k=0.5).scans[0]

    def test_weekend_gap_is_not_part_of_the_prior_session(self):
        s = self._run_weekend("QQQ")
        assert s.ps_passed and s.qualifies
        assert s.ul_move_pct == pytest.approx(0.0)

    def test_24h_underlying_still_sees_the_whole_day(self):
        s = self._run_weekend("BTC")
        assert not s.ps_passed and s.filter_reason == "ps_filter"
        assert s.ul_move_pct == pytest.approx(0.029126, abs=1e-6)


class TestK:
    def test_k_is_half(self):
        assert v1.K_SIGMA == 0.5


class TestCostReward:
    def test_model_formula(self):
        m = load_model()
        e, s, p = model_slippage_bps(100_000.0, m["default_notional"], m)
        root = math.sqrt(m["default_notional"] / 100_000.0)
        assert e == pytest.approx(m["entry"]["a"] + m["entry"]["b"] * root)
        assert s == pytest.approx(m["stop"]["a"] + m["stop"]["b"] * root)

    def test_missing_volume_charges_the_cap(self):
        m = load_model()
        _, _, p = model_slippage_bps(float("nan"), 1_000.0, m)
        assert p == m["part_max"]

    def test_threshold_and_reward(self):
        orb = {"high": 101.0, "low": 99.0, "dollar_per_min": 1e9}
        d = check_cost_to_reward(orb, +1, 2.0)
        assert d.reward_bps == pytest.approx(2.0 * 2.0 / 101.0 * 1e4)
        assert d.passed and MAX_COST_TO_REWARD == 0.15

    def test_thin_narrow_setup_is_skipped(self):
        orb = {"high": 100.2, "low": 100.0, "dollar_per_min": 500.0}
        assert not check_cost_to_reward(orb, -1, 2.0).passed


class TestOrbDollarVolume:
    def test_median_close_times_volume(self):
        idx = pd.date_range("2026-09-15 09:30", periods=30, freq="1min")
        bars = pd.DataFrame({"close": np.linspace(10, 11, 30), "high": np.linspace(10.1, 11.1, 30),
                             "low": np.linspace(9.9, 10.9, 30), "volume": np.arange(30) * 100.0},
                            index=idx)
        cfg = type("C", (), {"market_open_hour": 9, "market_open_minute": 30, "orb_minutes": 30,
                             "min_orb_bars": 25, "min_orb_bars_sparse": 10,
                             "sparse_data_symbols": ()})()
        orb = compute_opening_range(bars, cfg)
        assert orb["dollar_per_min"] == pytest.approx(float((bars["close"] * bars["volume"]).median()))
