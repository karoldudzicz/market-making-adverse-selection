"""Shared experiment infrastructure: seed sets, parallel runs, statistics and CSV output."""

import csv
import math
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from mmsim.config import SimulationConfig, StrategyConfig
from mmsim.engine import simulate
from mmsim.market import FloatArray, TakerType
from mmsim.metrics import RunMetrics, compute_metrics
from mmsim.strategies import build_strategy

TUNING_SEEDS = range(0, 200)
REPORT_SEEDS = range(1000, 1500)
RESULTS_DIR = Path("results")
DEFAULT_MODEL = SimulationConfig()
Z_95 = 1.959963984540054
COUNTERPARTIES = {TakerType.NOISE: "noise takers", TakerType.INFORMED: "informed takers"}


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One simulation: market and maker configuration, strategy and seed."""

    config: SimulationConfig
    strategy: StrategyConfig
    seed: int


@dataclass(frozen=True, slots=True)
class Estimate:
    """Sample mean with the half-width of its 95% confidence interval."""

    mean: float
    half_width: float

    @property
    def low(self) -> float:
        """Lower end of the confidence interval."""
        return self.mean - self.half_width

    @property
    def high(self) -> float:
        """Upper end of the confidence interval."""
        return self.mean + self.half_width


def run_metrics(spec: RunSpec) -> RunMetrics:
    """Simulate one run and return its summary metrics."""
    return compute_metrics(simulate(spec.config, build_strategy(spec.strategy), spec.seed))


def run_groups(
    groups: Sequence[tuple[SimulationConfig, StrategyConfig]], seeds: range
) -> list[list[RunMetrics]]:
    """Run every (model, strategy) pair on every seed in one parallel batch.

    Returns the metrics of each pair, in seed order, in the order of ``groups``.
    """
    specs = [RunSpec(config, strategy, seed) for config, strategy in groups for seed in seeds]
    metrics = parallel_map(run_metrics, specs)
    size = len(seeds)
    return [metrics[index * size : (index + 1) * size] for index in range(len(groups))]


def column(runs: Sequence[RunMetrics], metric: Callable[[RunMetrics], float]) -> FloatArray:
    """Extract one metric of every run as an array."""
    return np.array([metric(run) for run in runs], dtype=np.float64)


PNL_COMPONENTS: dict[str, Callable[[RunMetrics], float]] = {
    "pnl": lambda run: run.pnl.total,
    "noise_edge": lambda run: run.pnl.noise_edge,
    "informed_edge": lambda run: run.pnl.informed_edge,
    "inventory": lambda run: run.pnl.inventory,
}
"""Total PnL and the three parts of its exact decomposition."""


def parallel_map[T, R](fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
    """Apply ``fn`` to every item in worker processes, preserving order.

    ``fn`` must be a module-level function so that it can be pickled.
    """
    workers = os.cpu_count() or 1
    chunksize = max(1, len(items) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))


def mean_ci(values: ArrayLike) -> Estimate:
    """Mean and normal-approximation 95% confidence interval of a sample.

    For paired comparisons pass the per-seed differences.
    """
    sample = np.asarray(values, dtype=np.float64)
    half_width = Z_95 * float(sample.std(ddof=1)) / math.sqrt(sample.size)
    return Estimate(float(sample.mean()), half_width)


def ratio_ci(numerators: ArrayLike, denominators: ArrayLike) -> Estimate:
    """Pooled ratio sum(x) / sum(y) over runs with a delta-method 95% confidence interval.

    Used for per-fill quantities such as edge per fill, where runs differ in fill counts.
    """
    x = np.asarray(numerators, dtype=np.float64)
    y = np.asarray(denominators, dtype=np.float64)
    ratio = float(x.sum() / y.sum())
    residual = x - ratio * y
    half_width = Z_95 * float(residual.std(ddof=1)) * math.sqrt(x.size) / float(y.sum())
    return Estimate(ratio, half_width)


def with_informed_share(config: SimulationConfig, informed_share: float) -> SimulationConfig:
    """Return ``config`` with a different flow toxicity pi."""
    return replace(config, takers=replace(config.takers, informed_share=informed_share))


def with_latency(config: SimulationConfig, latency: int) -> SimulationConfig:
    """Return ``config`` with a different market maker latency L."""
    return replace(config, maker=replace(config.maker, latency=latency))


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write rows to a CSV file, formatting floats to six significant digits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [f"{value:.6g}" if isinstance(value, float) else value for value in row]
            )
