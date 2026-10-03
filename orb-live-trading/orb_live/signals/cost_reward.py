"""signals/cost_reward.py — opening-liquidity rule, decided at the ORB close.

Skip a setup when modelled execution cost is too large a share of what the
trade can earn (BacktestingGaps register V48 §3-4; orb_backtester
passes_cost_to_reward / model_slippage_bps):

    V       = median(close x volume) over the ORB minutes
    part    = min(notional / V, part_max)           (part_max if V missing)
    entry   = max(0, a_e + b_e sqrt(part))           bps
    stop    = max(0, a_s + b_s sqrt(part))           bps
    reward  = tp1_multiple x (ORB high - ORB low) / boundary x 1e4
    skip if (entry + stop) / reward > MAX_COST_TO_REWARD

Coefficients: orb_live/strategy/data/slippage_model.json, a copy of
BacktestingGaps/analysis/out/slippage_model.json. The backtest charges the
calibration notional (live's median order), not the weighted order size, so
that is the default here too.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

MAX_COST_TO_REWARD = 0.15
_MODEL_PATH = Path(__file__).resolve().parents[1] / "strategy" / "data" / "slippage_model.json"


@lru_cache(maxsize=4)
def load_model(path: Path = _MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class CostRewardDecision:
    passed: bool
    ratio: float
    entry_bps: float
    stop_bps: float
    reward_bps: float
    participation: float
    dollar_per_min: float
    notional: float


def model_slippage_bps(dollar_per_min: float, notional: float, model: dict) -> tuple[float, float, float]:
    """(entry_bps, stop_bps, participation)."""
    pmax = float(model["part_max"])
    v = dollar_per_min
    part = pmax if not (v and v > 0 and math.isfinite(v)) else min(notional / v, pmax)
    root = math.sqrt(part)
    entry = max(0.0, model["entry"]["a"] + model["entry"]["b"] * root)
    stop = max(0.0, model["stop"]["a"] + model["stop"]["b"] * root)
    return entry, stop, part


def check_cost_to_reward(
    orb: dict,
    gap_direction: int,
    tp1_multiple: float,
    notional: Optional[float] = None,
    max_ratio: float = MAX_COST_TO_REWARD,
    model: Optional[dict] = None,
) -> CostRewardDecision:
    m = model or load_model()
    n = float(notional if notional is not None else m["default_notional"])
    dpm = float(orb.get("dollar_per_min", float("nan")))
    entry, stop, part = model_slippage_bps(dpm, n, m)
    bound = orb["high"] if gap_direction == 1 else orb["low"]
    reward = tp1_multiple * (orb["high"] - orb["low"]) / bound * 1e4 if bound > 0 else 0.0
    ratio = (entry + stop) / reward if reward > 0 else float("inf")
    return CostRewardDecision(
        passed=reward > 0 and ratio <= max_ratio, ratio=ratio, entry_bps=entry,
        stop_bps=stop, reward_bps=reward, participation=part,
        dollar_per_min=dpm, notional=n,
    )
