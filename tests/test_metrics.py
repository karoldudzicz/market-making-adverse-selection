from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mmsim.config import MakerConfig, SimulationConfig, TakerConfig
from mmsim.engine import Fill, SimulationResult, simulate
from mmsim.market import TakerType
from mmsim.metrics import compute_metrics, markouts, pnl_breakdown
from mmsim.orderbook import Side
from quoters import SymmetricQuoter


def handmade_result() -> SimulationResult:
    """Buy 1 @ 100 from a noise taker in step 1, sell 1 @ 102 to an informed taker in step 2."""
    nan = float("nan")
    return SimulationResult(
        config=SimulationConfig(n_steps=3),
        seed=0,
        values=np.array([100.0, 101.0, 103.0, 102.0]),
        jumps=np.zeros(4),
        reference=np.zeros(4),
        market_mid=np.array([100.0, 100.5, 103.0, 101.0]),
        background_mid=np.zeros(4),
        bid=np.full(4, nan),
        ask=np.full(4, nan),
        position=np.array([0, 1, 0, 0]),
        cash=np.array([0, -100, 2, 2]),
        fills=(
            Fill(1, Side.BUY, 100, 1, TakerType.NOISE),
            Fill(2, Side.SELL, 102, 1, TakerType.INFORMED),
        ),
        trades=(),
    )


def test_pnl_breakdown_of_handmade_run() -> None:
    breakdown = pnl_breakdown(handmade_result())

    assert breakdown.total == 2.0
    assert breakdown.noise_edge == 1.0
    assert breakdown.informed_edge == -1.0
    assert breakdown.inventory == 2.0


def test_markouts_of_handmade_run() -> None:
    table = markouts(handmade_result(), horizons=(1, 2, 5))

    assert table[TakerType.NOISE] == {1: 3.0, 2: 2.0, 5: pytest.approx(float("nan"), nan_ok=True)}
    assert table[TakerType.INFORMED][1] == 0.0
    assert np.isnan(table[TakerType.INFORMED][2])


def test_run_metrics_of_handmade_run() -> None:
    metrics = compute_metrics(handmade_result())

    assert (metrics.n_fills, metrics.n_noise_fills, metrics.n_informed_fills) == (2, 1, 1)
    assert metrics.mean_abs_position == pytest.approx(1 / 3)
    assert metrics.max_abs_position == 1
    assert metrics.mean_abs_mid_error == pytest.approx(0.5)
    assert metrics.max_abs_mid_error == 1.0


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    informed_share=st.floats(min_value=0.0, max_value=1.0),
    latency=st.integers(min_value=0, max_value=10),
    half_spread=st.sampled_from([0.0, 0.5, 1.0, 2.0, 4.0]),
)
def test_pnl_decomposition_is_an_exact_identity(
    seed: int, informed_share: float, latency: int, half_spread: float
) -> None:
    config = SimulationConfig(
        n_steps=400,
        takers=TakerConfig(informed_share=informed_share),
        maker=MakerConfig(latency=latency),
    )

    breakdown = pnl_breakdown(simulate(config, SymmetricQuoter(half_spread), seed))

    parts = breakdown.noise_edge + breakdown.informed_edge + breakdown.inventory
    assert breakdown.total == pytest.approx(parts, abs=1e-6)


def test_fills_split_by_counterparty_type() -> None:
    config = replace(SimulationConfig(n_steps=2_000), takers=TakerConfig(informed_share=0.5))

    metrics = compute_metrics(simulate(config, SymmetricQuoter(1), seed=0))

    assert metrics.n_noise_fills > 0
    assert metrics.n_informed_fills > 0
    assert metrics.n_fills == metrics.n_noise_fills + metrics.n_informed_fills
