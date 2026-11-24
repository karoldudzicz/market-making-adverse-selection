"""E4: break-even half-spread and PnL as functions of flow toxicity pi.

``python -m mmsim.experiments.breakeven --preview`` prints a coarse preview on the tuning seeds.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from mmsim.config import InventorySkewConfig, NaiveConfig, SimulationConfig, StrategyConfig
from mmsim.experiments.common import (
    DEFAULT_MODEL,
    TUNING_SEEDS,
    Estimate,
    RunSpec,
    mean_ci,
    parallel_map,
    run_metrics,
    with_informed_share,
)
from mmsim.experiments.tune import Objective, TunedStrategies, load_selection

HALF_SPREADS = tuple(float(h) for h in np.arange(0.0, 6.01, 0.5))
PREVIEW_INFORMED_SHARES = (0.0, 0.2, 0.4, 0.6)


@dataclass(frozen=True, slots=True)
class Cell:
    """PnL of one strategy configuration at one toxicity level.

    Attributes
    ----------
    informed_share : float
        Flow toxicity pi.
    strategy : StrategyConfig
        Evaluated configuration.
    pnl : Estimate
        Mean PnL per run with its 95% confidence interval.
    informed_edge : Estimate
        Mean edge against informed takers per run.
    fills : float
        Mean number of fills per run.
    """

    informed_share: float
    strategy: StrategyConfig
    pnl: Estimate
    informed_edge: Estimate
    fills: float


def evaluate_cells(
    grid: Sequence[tuple[float, StrategyConfig]],
    seeds: range,
    config: SimulationConfig = DEFAULT_MODEL,
) -> list[Cell]:
    """Evaluate (pi, strategy) pairs on the same seeds."""
    specs = [
        RunSpec(with_informed_share(config, share), strategy, seed)
        for share, strategy in grid
        for seed in seeds
    ]
    metrics = parallel_map(run_metrics, specs)
    cells = []
    for index, (share, strategy) in enumerate(grid):
        runs = metrics[index * len(seeds) : (index + 1) * len(seeds)]
        cells.append(
            Cell(
                informed_share=share,
                strategy=strategy,
                pnl=mean_ci([run.pnl.total for run in runs]),
                informed_edge=mean_ci([run.pnl.informed_edge for run in runs]),
                fills=float(np.mean([run.n_fills for run in runs])),
            )
        )
    return cells


def half_spread_family(tuned: TunedStrategies) -> dict[str, list[StrategyConfig]]:
    """Naive and inventory-skew configurations over the half-spread grid."""
    return {
        "naive": [NaiveConfig(h) for h in HALF_SPREADS],
        "skew": [replace(tuned.skew, half_spread=h) for h in HALF_SPREADS],
    }


def breakeven_half_spread(cells: Sequence[Cell]) -> float | None:
    """Smallest half-spread on the grid whose mean PnL is non-negative."""
    for cell in cells:
        if cell.pnl.mean >= 0:
            return _half_spread(cell.strategy)
    return None


def _half_spread(strategy: StrategyConfig) -> float:
    if not isinstance(strategy, NaiveConfig | InventorySkewConfig):
        raise TypeError(f"{type(strategy).__name__} has no half-spread")
    return strategy.half_spread


def preview(seeds: range = TUNING_SEEDS) -> None:
    """Print a coarse E4 preview for the selections of both tuning objectives."""
    selections = {objective: load_selection(objective) for objective in Objective}
    families = half_spread_family(selections[Objective.SHARPE])
    stoikov = {f"A-S ({o.value})": s.avellaneda_stoikov for o, s in selections.items()}
    grid: list[tuple[float, StrategyConfig]] = [
        (share, strategy)
        for share in PREVIEW_INFORMED_SHARES
        for strategy in [*families["naive"], *families["skew"], *stoikov.values()]
    ]
    cells = evaluate_cells(grid, seeds)
    for name in ("naive", "skew"):
        print(f"\n{name}: mean PnL per run [fills per run] by half-spread h")
        print("   pi " + "".join(f"{h:>13.1f}" for h in HALF_SPREADS) + "   break-even h")
        for share in PREVIEW_INFORMED_SHARES:
            row = [c for c in cells if c.informed_share == share and c.strategy in families[name]]
            values = "".join(f"{c.pnl.mean:7.0f} [{c.fills:4.0f}]" for c in row)
            print(f"{share:5.1f} {values}   {breakeven_half_spread(row)}")
    print("\nAvellaneda-Stoikov: mean PnL ± CI, informed edge, fills per run")
    for label, strategy in stoikov.items():
        print(f"{label}: {strategy}")
        for share in PREVIEW_INFORMED_SHARES:
            (cell,) = [c for c in cells if c.informed_share == share and c.strategy == strategy]
            print(
                f"  pi={share:.1f}: {cell.pnl.mean:7.0f} ± {cell.pnl.half_width:4.0f}, "
                f"informed edge {cell.informed_edge.mean:7.0f}, fills {cell.fills:5.0f}"
            )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="coarse preview on tuning seeds")
    args = parser.parse_args()
    if args.preview:
        preview()


if __name__ == "__main__":
    main()
