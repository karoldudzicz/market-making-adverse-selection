"""Reproduce every table and figure in ``results/``.

Run from the repository root with ``python -m mmsim.experiments.run_all``. Tuning is rerun
from scratch and must select the parameters set in ``mmsim.config``; otherwise the run stops
before reporting results for stale parameters.
"""

import math
import time
from collections.abc import Callable
from dataclasses import replace

from mmsim.config import REPORTED_STRATEGIES, AvellanedaStoikovConfig, StrategyConfig
from mmsim.experiments.breakeven import run_e4
from mmsim.experiments.calibrate import IntensityFit, run_e0
from mmsim.experiments.compare import plot_example_run, run_e2_e3_e6
from mmsim.experiments.latency import run_e5
from mmsim.experiments.tune import Objective, TunedStrategies, tune


def check_tuned_parameters(fit: IntensityFit, selections: dict[Objective, TunedStrategies]) -> None:
    """Raise if the freshly calibrated and tuned parameters differ from ``mmsim.config``."""
    configured_k = AvellanedaStoikovConfig().k
    if not math.isclose(fit.k, configured_k, rel_tol=1e-5):
        raise RuntimeError(f"E0 gives k = {fit.k:.6g}, config.py has k = {configured_k}")
    sharpe, mean = selections[Objective.SHARPE], selections[Objective.MEAN]
    tuned: dict[str, StrategyConfig] = {
        "Naive": sharpe.naive,
        "Skew": sharpe.skew,
        "A-S (Sharpe)": replace(sharpe.avellaneda_stoikov, k=configured_k),
        "A-S (mean)": replace(mean.avellaneda_stoikov, k=configured_k),
    }
    stale = {
        label: config for label, config in tuned.items() if REPORTED_STRATEGIES[label] != config
    }
    if stale:
        raise RuntimeError(f"tuning selected parameters that differ from config.py: {stale}")


def main() -> None:
    """Run E0, tuning, E2-E6 and E1, printing the duration of each stage."""
    durations: dict[str, float] = {}

    def timed[T](stage: str, action: Callable[[], T]) -> T:
        start = time.perf_counter()
        outcome = action()
        durations[stage] = time.perf_counter() - start
        print(f"{stage}: {durations[stage]:.0f} s", flush=True)
        return outcome

    fit = timed("E0 diagnostics and k calibration", run_e0)
    selections = timed("parameter tuning", lambda: tune(fit.k))
    check_tuned_parameters(fit, selections)
    timed("E2, E3, E6 strategy comparison", run_e2_e3_e6)
    timed("E4 toxicity sweep and jump sensitivity", run_e4)
    timed("E5 latency", run_e5)
    timed("E1 example run", plot_example_run)
    print(f"total: {sum(durations.values()) / 60:.1f} min")


if __name__ == "__main__":
    main()
