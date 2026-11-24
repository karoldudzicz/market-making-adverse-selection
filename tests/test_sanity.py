"""Statistical sanity checks of the model on fixed seeds."""

from dataclasses import replace

import numpy as np
import pytest

from mmsim.config import (
    AvellanedaStoikovConfig,
    InventorySkewConfig,
    MakerConfig,
    NaiveConfig,
    SimulationConfig,
    StrategyConfig,
    TakerConfig,
)
from mmsim.engine import simulate
from mmsim.metrics import pnl_breakdown
from mmsim.strategies import build_strategy
from quoters import NoQuotes

SEEDS = range(12)
BASE = SimulationConfig(n_steps=2_000)
DEFAULT_STRATEGIES = [NaiveConfig(), InventorySkewConfig(), AvellanedaStoikovConfig()]


def mean_pnl(config: SimulationConfig, strategy: StrategyConfig) -> float:
    runs = [simulate(config, build_strategy(strategy), s) for s in SEEDS]
    return float(np.mean([pnl_breakdown(r).total for r in runs]))


def with_informed_share(share: float) -> SimulationConfig:
    return replace(BASE, takers=TakerConfig(informed_share=share))


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_market_maker_profits_without_informed_flow(strategy: StrategyConfig) -> None:
    assert mean_pnl(with_informed_share(0.0), strategy) > 0


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES)
def test_market_maker_loses_against_purely_informed_flow(strategy: StrategyConfig) -> None:
    assert mean_pnl(with_informed_share(1.0), strategy) < 0


def test_latency_does_not_reduce_adverse_selection_loss() -> None:
    def mean_informed_edge(latency: int) -> float:
        config = replace(
            BASE, takers=TakerConfig(informed_share=0.3), maker=MakerConfig(latency=latency)
        )
        runs = [simulate(config, build_strategy(NaiveConfig()), s) for s in SEEDS]
        return float(np.mean([pnl_breakdown(r).informed_edge for r in runs]))

    edges = [mean_informed_edge(latency) for latency in (0, 2, 10)]

    assert edges[0] >= edges[1] >= edges[2]


def test_market_mid_tracks_true_value_without_drift() -> None:
    config = SimulationConfig(n_steps=8_000)
    errors = np.array(
        [np.abs(r.market_mid - r.values) for r in (simulate(config, NoQuotes(), s) for s in SEEDS)]
    )

    early = errors[:, 1:1_001].mean()
    late = errors[:, -1_000:].mean()

    assert early < 3.0
    assert late < 1.5 * early
