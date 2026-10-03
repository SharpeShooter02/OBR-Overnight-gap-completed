"""The adopted weight matrix, and why each row is the shape it is.

ADOPTED 2026-09-15 (BacktestingGaps register V47, V48 §9), replacing the fitted
C1 2.5/2.5/1.0 - C2 0.0/1.0/0.0 - C3 3.5/3.5/3.5 on the candidate-count regime.

V47: every fitted sleeve x regime matrix beat flat in-sample and lost to flat
out of sample in 4/4 split-half directions -- fitting noise. The replacement is
imposed round numbers on a regime set by |SPY overnight gap| (quiet < 0.4%,
flood > 1.0%), scored against flat on five subsets, winner among declared
finalists by full-span Calmar.

WHAT EACH ROW MEANS (per-trade edge on the locked base, flat sizing)

  C1  2.0 / 2.0 / 0.0   crypto earns on quiet (+0.92%) and active (+1.25%) days,
                        loses on flood days (-0.68%): sized up, then off.
  C2  0.5 / 0.5 / 0.5   weakest class; only flood clears zero (+0.74%). Halved.
  C3  1.0 / 1.0 / 1.5   the book; flood is its strongest cell (+1.51%).

HONESTY BOUND: in-sample, 17 sizing rungs scored on one sample; the crypto
evidence is 2024+ only; the holdout is flat under every configuration.
"""
from __future__ import annotations

import pytest

from orb_live.strategy.v1_strategy import WEIGHTS, assign_regime

REGIMES_ALL = ("quiet", "active", "flood")

ADOPTED = {
    ("C1", "quiet"): 2.0, ("C1", "active"): 2.0, ("C1", "flood"): 0.0,
    ("C2", "quiet"): 0.5, ("C2", "active"): 0.5, ("C2", "flood"): 0.5,
    ("C3", "quiet"): 1.0, ("C3", "active"): 1.0, ("C3", "flood"): 1.5,
}


class TestAdoptedMatrix:
    @pytest.mark.parametrize("key,value", sorted(ADOPTED.items()))
    def test_bucket(self, key, value):
        assert WEIGHTS[key] == value

    def test_no_extra_buckets(self):
        assert set(WEIGHTS) == set(ADOPTED)


class TestShape:
    """Each of these is a claim the analysis actually supports. If a future
    change breaks one, that is a finding to argue with, not a test to silence."""

    def test_c1_off_on_flood(self):
        assert WEIGHTS[("C1", "flood")] == 0.0
        assert WEIGHTS[("C1", "quiet")] > 1.0

    def test_c2_is_the_smallest_class(self):
        for r in REGIMES_ALL:
            assert WEIGHTS[("C2", r)] <= WEIGHTS[("C3", r)]

    def test_c3_leans_into_flood(self):
        assert WEIGHTS[("C3", "flood")] > WEIGHTS[("C3", "quiet")]

    def test_weights_are_round(self):
        """Imposed, not fitted: every weight is a multiple of 0.5."""
        assert all((w * 2) == int(w * 2) for w in WEIGHTS.values())


class TestSpyRegime:
    @pytest.mark.parametrize("gap,regime", [
        (0.0, "quiet"), (0.0039, "quiet"), (-0.0039, "quiet"),
        (0.004, "active"), (0.010, "active"), (-0.0075, "active"),
        (0.0101, "flood"), (-0.02, "flood"),
    ])
    def test_cuts(self, gap, regime):
        assert assign_regime(gap) == regime

    def test_missing_gap_falls_back(self):
        """Unmeasurable -> flood, the cautious reading: C1 is 0.0 there, so a
        failed SPY fetch cannot buy full crypto size (2026-09-16)."""
        assert assign_regime(None) == "flood"
        assert assign_regime(float("nan")) == "flood"
