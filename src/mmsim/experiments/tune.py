"""Strategy parameter tuning on the tuning seeds at the default model (pi = 0.2, L = 0).

Run with ``python -m mmsim.experiments.tune``. Two selection objectives are reported, both
using a one-standard-error rule so that grid noise does not drive the choice:

* ``MEAN`` (risk-neutral): among points whose mean PnL is within one standard error of the
  best (on paired per-seed differences), take the one with the lowest PnL standard deviation.
* ``SHARPE`` (risk-adjusted): among points whose Sharpe ratio across seeds (mean / std of run
  PnL) is within one standard error of the best, take the one with the highest mean PnL.
"""

import csv
import math
from collections.abc import Sequence
from dataclasses import astuple, dataclass, fields
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mmsim.config import (
    AvellanedaStoikovConfig,
    InventorySkewConfig,
    NaiveConfig,
    SimulationConfig,
    StrategyConfig,
)
from mmsim.experiments.common import (
    DEFAULT_MODEL,
    RESULTS_DIR,
    TUNING_SEEDS,
    RunSpec,
    mean_ci,
    parallel_map,
    run_metrics,
    write_csv,
)

NAIVE_HALF_SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
SKEW_HALF_SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
SKEWS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
GAMMAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
HORIZONS = (10.0, 30.0, 100.0, 300.0, 1000.0)


class Objective(Enum):
    """Tuning objective."""

    MEAN = "mean"
    SHARPE = "sharpe"


@dataclass(frozen=True, slots=True)
class GridPoint:
    """Per-seed results of one strategy configuration.

    Attributes
    ----------
    strategy : StrategyConfig
        Evaluated configuration.
    pnl : ndarray
        PnL per seed.
    fills : float
        Mean number of fills per run.
    mean_abs_position : float
        Mean over runs of the average |q|.
    """

    strategy: StrategyConfig
    pnl: NDArray[np.float64]
    fills: float
    mean_abs_position: float

    @property
    def sharpe(self) -> float:
        """Mean over standard deviation of run PnL; zero for a point that never trades."""
        std = float(self.pnl.std(ddof=1))
        return float(self.pnl.mean()) / std if std > 0 else 0.0

    @property
    def sharpe_se(self) -> float:
        """Standard error of the Sharpe ratio, sqrt((1 + S^2 / 2) / n) (Lo, 2002)."""
        return math.sqrt((1 + self.sharpe**2 / 2) / self.pnl.size)


@dataclass(frozen=True, slots=True)
class TunedStrategies:
    """Selected configuration of each strategy family."""

    naive: NaiveConfig
    skew: InventorySkewConfig
    avellaneda_stoikov: AvellanedaStoikovConfig


def evaluate(
    strategies: Sequence[StrategyConfig], config: SimulationConfig, seeds: range
) -> list[GridPoint]:
    """Run every strategy on every seed and collect per-seed PnL."""
    specs = [RunSpec(config, strategy, seed) for strategy in strategies for seed in seeds]
    metrics = parallel_map(run_metrics, specs)
    points = []
    for index, strategy in enumerate(strategies):
        runs = metrics[index * len(seeds) : (index + 1) * len(seeds)]
        points.append(
            GridPoint(
                strategy=strategy,
                pnl=np.array([run.pnl.total for run in runs]),
                fills=float(np.mean([run.n_fills for run in runs])),
                mean_abs_position=float(np.mean([run.mean_abs_position for run in runs])),
            )
        )
    return points


def tied_with_best(points: Sequence[GridPoint], objective: Objective) -> list[GridPoint]:
    """Return the points within one standard error of the best point under ``objective``."""
    if objective is Objective.MEAN:
        best = max(points, key=lambda point: point.pnl.mean())
        return [point for point in points if _mean_within_one_se(best, point)]
    leader = max(points, key=lambda point: point.sharpe)
    return [point for point in points if point.sharpe >= leader.sharpe - leader.sharpe_se]


def select(points: Sequence[GridPoint], objective: Objective) -> GridPoint:
    """Select one grid point by the one-standard-error rule of ``objective``."""
    tied = tied_with_best(points, objective)
    if objective is Objective.MEAN:
        return min(tied, key=lambda point: point.pnl.std())
    return max(tied, key=lambda point: point.pnl.mean())


def tune(
    k: float,
    seeds: range = TUNING_SEEDS,
    config: SimulationConfig = DEFAULT_MODEL,
    out_dir: Path = RESULTS_DIR,
) -> dict[Objective, TunedStrategies]:
    """Tune all strategy families, write the grids to ``out_dir`` and return the selections.

    Parameters
    ----------
    k : float
        Fill intensity decay from E0, used by Avellaneda-Stoikov.
    seeds : range
        Seeds to tune on.
    config : SimulationConfig
        Market model to tune in.
    out_dir : Path
        Output directory for the grid CSVs.

    Returns
    -------
    dict[Objective, TunedStrategies]
        Selected configuration of each family under each objective.
    """
    naive = evaluate([NaiveConfig(h) for h in NAIVE_HALF_SPREADS], config, seeds)
    skew = evaluate(
        [InventorySkewConfig(h, a) for h in SKEW_HALF_SPREADS for a in SKEWS], config, seeds
    )
    rolling = evaluate(
        [AvellanedaStoikovConfig(g, tau, k) for g in GAMMAS for tau in HORIZONS], config, seeds
    )
    finite = evaluate([AvellanedaStoikovConfig(g, None, k) for g in GAMMAS], config, seeds)

    selections = {
        objective: TunedStrategies(
            naive=_as(NaiveConfig, select(naive, objective).strategy),
            skew=_as(InventorySkewConfig, select(skew, objective).strategy),
            avellaneda_stoikov=_as(AvellanedaStoikovConfig, select(rolling, objective).strategy),
        )
        for objective in Objective
    }
    _write_grid(out_dir / "tuning_naive.csv", naive, naive)
    _write_grid(out_dir / "tuning_skew.csv", skew, skew)
    _write_grid(out_dir / "tuning_avellaneda_stoikov.csv", rolling + finite, rolling)
    return selections


def load_selection(objective: Objective, out_dir: Path = RESULTS_DIR) -> TunedStrategies:
    """Read the configurations selected under ``objective`` from the tuning CSVs."""

    def selected_row(name: str) -> dict[str, str]:
        with (out_dir / name).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        (row,) = (r for r in rows if r[f"selected_{objective.value}"] == "True")
        return row

    naive = selected_row("tuning_naive.csv")
    skew = selected_row("tuning_skew.csv")
    stoikov = selected_row("tuning_avellaneda_stoikov.csv")
    return TunedStrategies(
        naive=NaiveConfig(float(naive["half_spread"])),
        skew=InventorySkewConfig(float(skew["half_spread"]), float(skew["skew"])),
        avellaneda_stoikov=AvellanedaStoikovConfig(
            float(stoikov["gamma"]),
            float(stoikov["horizon"]) if stoikov["horizon"] else None,
            float(stoikov["k"]),
        ),
    )


def _mean_within_one_se(best: GridPoint, point: GridPoint) -> bool:
    difference = best.pnl - point.pnl
    standard_error = float(difference.std(ddof=1)) / math.sqrt(difference.size)
    return float(difference.mean()) <= standard_error


def _as[C](kind: type[C], strategy: StrategyConfig) -> C:
    if not isinstance(strategy, kind):
        raise TypeError(f"expected {kind.__name__}, got {type(strategy).__name__}")
    return strategy


def _write_grid(path: Path, points: Sequence[GridPoint], candidates: Sequence[GridPoint]) -> None:
    """Write a tuning grid; only ``candidates`` take part in the selection."""
    flags = {}
    for objective in Objective:
        tied = tied_with_best(candidates, objective)
        flags[objective] = (tied, select(candidates, objective))
    header = [
        *(field.name for field in fields(points[0].strategy)),
        "mean_pnl",
        "pnl_ci95",
        "std_pnl",
        "sharpe",
        "sharpe_se",
        "fills_per_run",
        "mean_abs_position",
    ]
    for objective in Objective:
        header += [f"tied_{objective.value}", f"selected_{objective.value}"]
    rows = []
    for point in points:
        pnl = mean_ci(point.pnl)
        row = [
            *astuple(point.strategy),
            pnl.mean,
            pnl.half_width,
            float(point.pnl.std(ddof=1)),
            point.sharpe,
            point.sharpe_se,
            point.fills,
            point.mean_abs_position,
        ]
        for objective in Objective:
            tied, selected = flags[objective]
            row += [any(point is p for p in tied), point is selected]
        rows.append(row)
    write_csv(path, header, rows)


def main() -> None:
    """Tune on the tuning seeds with k from ``results/e0_intensity_fit.csv``."""
    with (RESULTS_DIR / "e0_intensity_fit.csv").open(encoding="utf-8") as handle:
        k = float(handle.readlines()[1].split(",")[1])
    for objective, selected in tune(k).items():
        print(objective.value, selected)


if __name__ == "__main__":
    main()
