"""Statistical sanity checks of the model on fixed seeds."""

from dataclasses import replace

import numpy as np

from mmsim.config import MakerConfig, SimulationConfig, TakerConfig
from mmsim.engine import simulate
from mmsim.metrics import pnl_breakdown
from mmsim.strategies import Strategy
from quoters import NoQuotes, SymmetricQuoter

SEEDS = range(12)
BASE = SimulationConfig(n_steps=2_000)


def mean_pnl(config: SimulationConfig, strategy: Strategy) -> float:
    return float(np.mean([pnl_breakdown(simulate(config, strategy, s)).total for s in SEEDS]))


def with_informed_share(share: float) -> SimulationConfig:
    return replace(BASE, takers=TakerConfig(informed_share=share))


def test_market_maker_profits_without_informed_flow() -> None:
    assert mean_pnl(with_informed_share(0.0), SymmetricQuoter(1)) > 0


def test_market_maker_loses_against_purely_informed_flow() -> None:
    assert mean_pnl(with_informed_share(1.0), SymmetricQuoter(1)) < 0


def test_latency_does_not_reduce_adverse_selection_loss() -> None:
    def mean_informed_edge(latency: int) -> float:
        config = replace(
            BASE, takers=TakerConfig(informed_share=0.3), maker=MakerConfig(latency=latency)
        )
        runs = [simulate(config, SymmetricQuoter(1), s) for s in SEEDS]
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
