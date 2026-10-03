"""
tests/test_universe_parity.py — Verify live universe matches v1 build_universe().

These are canary tests: if someone edits the v1 strategy CSVs or live_config
without keeping them in sync, this suite catches the drift.
"""

import pytest


def test_live_symbols_match_v1_universe(live_cfg):
    from orb_live.strategy.v1_strategy import load_master, load_sigmas, build_universe
    from pathlib import Path

    data_dir  = Path(__file__).parent.parent / "strategy" / "data"
    instr     = load_master(data_dir / "master_universe.csv")
    sigmas    = load_sigmas(data_dir / "sigma_master.csv")
    expected  = set(build_universe(instr, sigmas, data_dir / "master_universe.csv"))
    actual    = set(live_cfg.symbols)

    assert actual == expected, (
        f"Symbol drift.\n"
        f"  In v1 but not live: {expected - actual}\n"
        f"  In live but not v1: {actual - expected}"
    )


def test_symbol_count(live_cfg):
    n = len(live_cfg.symbols)
    assert 50 <= n <= 75, f"Expected ~60 symbols, got {n}"


def test_all_live_symbols_in_instruments(live_cfg):
    missing = [s for s in live_cfg.symbols if s not in live_cfg.instruments]
    assert not missing, f"Live symbols missing from instruments: {missing}"


def test_direction_filters_subset_of_universe(live_cfg):
    unknown = [s for s in live_cfg.direction_filters if s not in live_cfg.symbols]
    assert not unknown, f"direction_filters reference symbols not in live universe: {unknown}"


def test_no_day_of_week_exclusions(live_cfg):
    """v1 removes DOW exclusions; strategy_config.day_of_week_exclusions must be empty."""
    assert live_cfg.strategy_config.day_of_week_exclusions == {}, (
        "v1 config must have no DOW exclusions"
    )


def test_no_instrument_gap_filters(live_cfg):
    """v1 uses per-ETF leverage × GAP_THRESHOLD; instrument_gap_filters must be empty."""
    assert live_cfg.strategy_config.instrument_gap_filters == {}, (
        "v1 config must have no instrument_gap_filters"
    )


class TestSkipCheapParity:
    """Parity test for the skip-cheap pruning rule.

    Current rule: always keep exactly 1 per UL group (the most expensive by
    prior close). Capped at 1 ticker per underlying regardless of N.
    """

    def _run(self, syms_with_prices: list[tuple[str, float]]) -> list[str]:
        from orb_live.strategy.v1_strategy import (
            compute_candidates, Instrument,
        )
        ul = "TESTUL"
        instruments = {
            sym: Instrument(sym, ul, 2, False)
            for sym, _ in syms_with_prices
        }
        universe = [s for s, _ in syms_with_prices]
        prior_etf_close = {s: p for s, p in syms_with_prices}
        overnight_gaps = {ul: 0.05}  # 5% gap, above 2% threshold
        prior_two_closes = {ul: (100.0, 100.0)}  # flat — PS filter always passes

        cands = compute_candidates(
            universe, instruments, {"TESTUL": 0.02},
            overnight_gaps, prior_two_closes, prior_etf_close,
        )
        return [c.symbol for c in cands]

    def test_n1_keeps_1(self):
        kept = self._run([("A", 100.0)])
        assert kept == ["A"]

    def test_n2_keeps_1_most_expensive(self):
        # N==2: classic skip-cheap — keep only the pricier one
        kept = self._run([("CHEAP", 50.0), ("PRICEY", 150.0)])
        assert len(kept) == 1
        assert kept[0] == "PRICEY"

    def test_n3_keeps_1_most_expensive(self):
        kept = self._run([("A", 30.0), ("B", 100.0), ("C", 200.0)])
        assert kept == ["C"]

    def test_n4_keeps_1_most_expensive(self):
        kept = self._run([("A", 10.0), ("B", 50.0), ("C", 100.0), ("D", 200.0)])
        assert kept == ["D"]


# ── ETH: a 1x spot trust in a leveraged-ETF strategy ──────────────────────────

def test_eth_is_dropped(live_cfg):
    """ETH is a 1x spot Ethereum trust, not a leveraged ETF, and master_universe
    had it as leverage=2. Measured against its siblings over nine sessions the
    ratio is exactly 2.0 (2026-09-04: ETH -2.50% vs ETHT -5.05%, ETHU -5.22%,
    ETU -4.98%), i.e. ETH moves 1x while every sibling moves 2x.

    Two independent problems, one instrument:
      * mislabelled leverage doubles its gap threshold to 4% when a 1x product
        should face 2%, so it is silently under-qualified; and if it were ever
        selected, sizing and the implied UL gap would both be off by 2x.
      * `ETH` is the ONLY symbol in the universe that is also an underlying key,
        so etf_daily["ETH"] (~24) and ul_daily["ETH"] (~2400) collide -- a 100x
        error for any lookup that resolves against the wrong one.

    Dropped rather than relabelled: the strategy's premise is leverage
    amplifying an underlying's gap, and ETH is covered by ETHT/ETHU/ETU/ETHD.
    """
    from orb_live.strategy.v1_strategy import EXPLICIT_DROPS

    assert "ETH" in EXPLICIT_DROPS
    assert "ETH" not in live_cfg.symbols
    # `instruments` is the full master catalog, not the tradeable universe, so
    # the ETH row legitimately survives there -- it just must never be traded.


def test_no_symbol_is_also_an_underlying(live_cfg):
    """The namespace collision that made ETH dangerous must not come back."""
    collisions = {
        s for s in live_cfg.symbols
        if s in {i.underlying for i in live_cfg.instruments.values()}
    }
    assert not collisions, (
        f"symbol(s) also used as an underlying key: {sorted(collisions)} -- "
        f"etf_daily[sym] and ul_daily[sym] resolve to different price scales"
    )
