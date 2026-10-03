"""Offline source checks; account replays and licensed data checks are separate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "orb-live-trading"), str(ROOT / "BacktestingGaps")]

import orb_live
import pytest

assert Path(orb_live.__file__).resolve().is_relative_to(ROOT), "Wrong installed source"

import analysis.freeze_run
import analysis.authoritative_run as backtest
from orb_live.config.live_config import load_live_config
from orb_live.signals.strategy_signals import STOP_ORB_DISTANCE

assert Path(analysis.freeze_run.__file__).resolve().is_relative_to(ROOT)
assert len(load_live_config(use_rolling=False).symbols) == 57
assert backtest.LOCKED_K_SIGMA == 0.5
assert STOP_ORB_DISTANCE == 1.0

raise SystemExit(pytest.main([
    str(ROOT / "orb-live-trading/orb_live/tests"),
    str(ROOT / "BacktestingGaps/tests/test_sharpe_convention.py") + "::TestAnnualisation::test_constant_is_252",
    str(ROOT / "BacktestingGaps/tests/test_sharpe_convention.py") + "::TestAnnualisation::test_risk_free_matches_the_period",
    "--rootdir=" + str(ROOT), "--confcutdir=" + str(ROOT),
    "-c", str(ROOT / "orb-live-trading/pyproject.toml"),
    "-o", "addopts=", "-p", "no:cacheprovider", "-q", "--tb=short",
    "--basetemp=" + str(ROOT / ".pytest-tmp"),
    "-k", "not TestRecordedSessionsReplay",
]))
