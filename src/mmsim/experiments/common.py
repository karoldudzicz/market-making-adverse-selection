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
from mmsim.metrics import RunMetrics, compute_metrics
from mmsim.strategies import build_strategy

TUNING_SEEDS = range(0, 200)
REPORT_SEEDS = range(1000, 1500)
RESULTS_DIR = Path("results")
DEFAULT_MODEL = SimulationConfig()
Z_95 = 1.959963984540054


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
