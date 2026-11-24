import math

import numpy as np
import pytest

from mmsim.config import NaiveConfig
from mmsim.experiments.breakeven import Cell, breakeven_half_spread
from mmsim.experiments.calibrate import JUMP_BEFORE, fit_intensity, recovery_steps
from mmsim.experiments.common import Estimate, mean_ci
from mmsim.experiments.tune import GridPoint, Objective, select


def test_mean_ci_uses_normal_approximation() -> None:
    sample = np.array([1.0, 2.0, 3.0, 4.0])

    estimate = mean_ci(sample)

    assert estimate.mean == 2.5
    assert estimate.half_width == pytest.approx(1.959964 * sample.std(ddof=1) / 2)
    assert (estimate.low, estimate.high) == pytest.approx(
        (2.5 - estimate.half_width, 2.5 + estimate.half_width)
    )


def test_fit_intensity_recovers_exponential_decay() -> None:
    distance = np.arange(0, 6.5, 0.5)
    quoted = np.full(distance.size, 1_000_000, dtype=np.int64)
    fills = np.rint(quoted * 0.2 * np.exp(-0.7 * distance)).astype(np.int64)

    fit = fit_intensity(distance, quoted, fills)

    assert fit.amplitude == pytest.approx(0.2, rel=1e-3)
    assert fit.k == pytest.approx(0.7, rel=1e-3)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)


def test_fit_intensity_ignores_buckets_with_few_fills() -> None:
    distance = np.array([0.0, 1.0, 2.0, 3.0])
    quoted = np.array([10_000, 10_000, 10_000, 10], dtype=np.int64)
    fills = np.array([1_000, 368, 135, 9], dtype=np.int64)

    fit = fit_intensity(distance, quoted, fills)

    assert fit.k == pytest.approx(1.0, rel=0.01)


def test_recovery_steps_counts_steps_after_the_jump() -> None:
    response = np.concatenate([np.ones(JUMP_BEFORE), [10.0, 8.0, 4.0, 1.2, 1.0]])

    assert recovery_steps(response, baseline=1.0) == 3
    assert recovery_steps(np.full(JUMP_BEFORE + 5, 10.0), baseline=1.0) is None


def point(pnl: list[float]) -> GridPoint:
    return GridPoint(NaiveConfig(), np.array(pnl), fills=0.0, mean_abs_position=0.0)


def test_mean_objective_prefers_lower_risk_among_statistically_tied_points() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, 200)
    risky_best = point(list(100 + 50 * noise))
    safe_tied = point(list(99 + 5 * noise))
    clearly_worse = point(list(50 + 1 * noise))

    assert select([risky_best, safe_tied, clearly_worse], Objective.MEAN) is safe_tied


def test_sharpe_objective_prefers_higher_mean_among_tied_ratios() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1.0, 200)
    noise = (noise - noise.mean()) / noise.std(ddof=1)
    small = point(list(10 + 1 * noise))
    large = point(list(99 + 10 * noise))
    risky = point(list(200 + 100 * noise))

    assert math.isclose(small.sharpe, 10.0)
    assert select([small, large, risky], Objective.SHARPE) is large


def test_breakeven_half_spread_is_first_non_negative_grid_point() -> None:
    def cell(h: float, pnl: float) -> Cell:
        return Cell(0.5, NaiveConfig(h), Estimate(pnl, 1.0), Estimate(0.0, 0.0), fills=1.0)

    assert breakeven_half_spread([cell(0.0, -5), cell(0.5, -1), cell(1.0, 3), cell(1.5, -2)]) == 1.0
    assert breakeven_half_spread([cell(0.0, -5), cell(0.5, -1)]) is None
