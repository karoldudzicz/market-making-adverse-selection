"""E4: how the profitable half-spread and the PnL depend on flow toxicity pi.

Naive and inventory-skew quotes are swept over half-spreads h at every pi; the strategies in
``REPORTED_STRATEGIES`` are evaluated at every pi with their tuned parameters. Run with
``python -m mmsim.experiments.breakeven``.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from mmsim.config import (
    REPORTED_STRATEGIES,
    InventorySkewConfig,
    NaiveConfig,
    SimulationConfig,
    StrategyConfig,
)
from mmsim.experiments import plotting
from mmsim.experiments.common import (
    DEFAULT_MODEL,
    PNL_COMPONENTS,
    RESULTS_DIR,
    Estimate,
    column,
    mean_ci,
    run_groups,
    with_informed_share,
    write_csv,
)
from mmsim.metrics import RunMetrics

INFORMED_SHARES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
HALF_SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
E4_SEEDS = range(1000, 1200)
SENSITIVITY_SEEDS = range(1000, 1100)
JUMP_SENSITIVITY_MODEL = replace(DEFAULT_MODEL, value=replace(DEFAULT_MODEL.value, jump_prob=0.01))
DECOMPOSED_STRATEGIES = ("Naive", "A-S (Sharpe)")
SWEPT_FAMILIES = ("Naive", "Skew")


@dataclass(frozen=True, slots=True)
class ToxicitySweep:
    """Per-seed results of E4 for one market model.

    Attributes
    ----------
    half_spread_sweeps : dict
        For each swept family, the runs at every (pi, h).
    tuned : dict
        For each reported strategy, the runs at every pi.
    """

    half_spread_sweeps: dict[str, dict[tuple[float, float], list[RunMetrics]]]
    tuned: dict[str, dict[float, list[RunMetrics]]]

    def mean_pnl_by_half_spread(self, family: str, share: float) -> list[float]:
        """Mean PnL over the half-spread grid of a family at one pi."""
        sweep = self.half_spread_sweeps[family]
        return [float(column(sweep[share, h], PNL_COMPONENTS["pnl"]).mean()) for h in HALF_SPREADS]


def breakeven_half_spread(mean_pnl: Sequence[float]) -> float | None:
    """Smallest half-spread on the grid whose mean PnL is non-negative."""
    for half_spread, pnl in zip(HALF_SPREADS, mean_pnl, strict=True):
        if pnl >= 0:
            return half_spread
    return None


def optimal_half_spread(mean_pnl: Sequence[float]) -> float:
    """Half-spread on the grid with the highest mean PnL."""
    return HALF_SPREADS[int(np.argmax(mean_pnl))]


def swept_strategy(family: str, half_spread: float) -> StrategyConfig:
    """Return a swept family at a given half-spread, other parameters as tuned."""
    tuned = REPORTED_STRATEGIES[family]
    if isinstance(tuned, NaiveConfig):
        return replace(tuned, half_spread=half_spread)
    if isinstance(tuned, InventorySkewConfig):
        return replace(tuned, half_spread=half_spread)
    raise ValueError(f"{family} has no half-spread to sweep")


def run_sweep(model: SimulationConfig, seeds: range) -> ToxicitySweep:
    """Evaluate the half-spread sweeps and the reported strategies at every pi."""
    swept = [
        (family, share, h)
        for family in SWEPT_FAMILIES
        for share in INFORMED_SHARES
        for h in HALF_SPREADS
    ]
    tuned = [(label, share) for label in REPORTED_STRATEGIES for share in INFORMED_SHARES]
    groups = [
        (with_informed_share(model, share), swept_strategy(family, h)) for family, share, h in swept
    ] + [(with_informed_share(model, share), REPORTED_STRATEGIES[label]) for label, share in tuned]
    runs = run_groups(groups, seeds)
    sweeps: dict[str, dict[tuple[float, float], list[RunMetrics]]] = {f: {} for f in SWEPT_FAMILIES}
    for (family, share, h), group in zip(swept, runs[: len(swept)], strict=True):
        sweeps[family][share, h] = group
    by_strategy: dict[str, dict[float, list[RunMetrics]]] = {
        label: {} for label in REPORTED_STRATEGIES
    }
    for (label, share), group in zip(tuned, runs[len(swept) :], strict=True):
        by_strategy[label][share] = group
    return ToxicitySweep(sweeps, by_strategy)


def write_tables(sweep: ToxicitySweep, out_dir: Path, suffix: str = "") -> None:
    """Write the E4 grid, break-even, tuned-strategy and toxicity-cost tables."""
    grid_rows, breakeven_rows = [], []
    for family in SWEPT_FAMILIES:
        for share in INFORMED_SHARES:
            for h in HALF_SPREADS:
                runs = sweep.half_spread_sweeps[family][share, h]
                grid_rows.append([family, share, h, *_component_estimates(runs), _mean_fills(runs)])
            mean_pnl = sweep.mean_pnl_by_half_spread(family, share)
            best = optimal_half_spread(mean_pnl)
            best_pnl = mean_ci(
                column(sweep.half_spread_sweeps[family][share, best], PNL_COMPONENTS["pnl"])
            )
            breakeven = breakeven_half_spread(mean_pnl)
            breakeven_rows.append(
                [
                    family,
                    share,
                    "" if breakeven is None else breakeven,
                    best,
                    best_pnl.mean,
                    best_pnl.half_width,
                ]
            )
    component_header = [f"{name}{part}" for name in PNL_COMPONENTS for part in ("", "_ci95")]
    write_csv(
        out_dir / f"e4_half_spread_grid{suffix}.csv",
        ["family", "informed_share", "half_spread", *component_header, "fills_per_run"],
        grid_rows,
    )
    write_csv(
        out_dir / f"e4_breakeven{suffix}.csv",
        [
            "family",
            "informed_share",
            "breakeven_half_spread",
            "optimal_half_spread",
            "max_mean_pnl",
            "max_mean_pnl_ci95",
        ],
        breakeven_rows,
    )
    tuned_rows, cost_rows = [], []
    for label, by_share in sweep.tuned.items():
        for share in INFORMED_SHARES:
            runs = by_share[share]
            after_jumps = float(column(runs, lambda run: run.jumps.informed_edge).mean())
            informed_fills = float(column(runs, lambda run: run.n_informed_fills).mean())
            tuned_rows.append(
                [
                    label,
                    share,
                    *_component_estimates(runs),
                    _mean_fills(runs),
                    informed_fills,
                    after_jumps,
                ]
            )
        for low, high in itertools.pairwise(INFORMED_SHARES):
            changes = []
            for metric in PNL_COMPONENTS.values():
                change = mean_ci(column(by_share[high], metric) - column(by_share[low], metric))
                changes += [change.mean, change.half_width]
            cost_rows.append([label, low, high, *changes])
    write_csv(
        out_dir / f"e4_tuned_strategies{suffix}.csv",
        [
            "strategy",
            "informed_share",
            *component_header,
            "fills_per_run",
            "informed_fills_per_run",
            "informed_edge_after_jumps",
        ],
        tuned_rows,
    )
    write_csv(
        out_dir / f"e4_toxicity_cost{suffix}.csv",
        [
            "strategy",
            "informed_share_from",
            "informed_share_to",
            *(f"change_{c}" for c in component_header),
        ],
        cost_rows,
    )


def plot_half_spread_curves(sweep: ToxicitySweep, out_dir: Path) -> None:
    """Plot mean PnL against the half-spread for every pi, one panel per swept family."""
    figure = plotting.new_figure(11, 4.2)
    panels = figure.subplots(1, len(SWEPT_FAMILIES), sharey=True)
    colors = plotting.sequential(len(INFORMED_SHARES))
    for axes, family in zip(panels, SWEPT_FAMILIES, strict=True):
        for share, color in zip(INFORMED_SHARES, colors, strict=True):
            mean_pnl = sweep.mean_pnl_by_half_spread(family, share)
            axes.plot(
                HALF_SPREADS,
                mean_pnl,
                color=color,
                marker="o",
                markersize=3,
                label=f"π = {share:.1f}",
            )
            best = int(np.argmax(mean_pnl))
            axes.plot(
                HALF_SPREADS[best],
                mean_pnl[best],
                "o",
                color=color,
                markersize=8,
                markeredgecolor=plotting.SURFACE,
            )
        axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
        axes.set_title(f"{family}: mean PnL per run vs half-spread h (large dot = optimum)")
        axes.set_xlabel("half-spread h (ticks)")
    panels[0].set_ylabel("mean PnL per run (ticks)")
    panels[-1].legend(loc="upper right", title="toxicity", title_fontsize=8)
    plotting.save(figure, out_dir / "e4_pnl_vs_half_spread.png")


def plot_breakeven(base: ToxicitySweep, jumps: ToxicitySweep, out_dir: Path) -> None:
    """Plot break-even and optimal half-spreads and the tuned strategies' PnL against pi."""
    figure = plotting.new_figure(14, 4.2)
    breakeven_axes, optimal_axes, pnl_axes = figure.subplots(1, 3)
    for family, color in zip(SWEPT_FAMILIES, plotting.SERIES, strict=False):
        for sweep, style, model in ((base, "-", "base"), (jumps, (0, (4, 2)), "jumps p = 0.01")):
            curves = [sweep.mean_pnl_by_half_spread(family, share) for share in INFORMED_SHARES]
            breakeven = [breakeven_half_spread(curve) for curve in curves]
            breakeven_axes.plot(
                INFORMED_SHARES,
                [np.nan if b is None else b for b in breakeven],
                color=color,
                linestyle=style,
                marker="o",
                markersize=4,
                label=f"{family}, {model}",
            )
            optimal_axes.plot(
                INFORMED_SHARES,
                [optimal_half_spread(curve) for curve in curves],
                color=color,
                linestyle=style,
                marker="o",
                markersize=4,
                label=f"{family}, {model}",
            )
    breakeven_axes.set_title("Break-even half-spread (smallest h with mean PnL ≥ 0)")
    optimal_axes.set_title("Optimal half-spread (highest mean PnL)")
    for axes in (breakeven_axes, optimal_axes):
        axes.set_xlabel("toxicity π")
        axes.set_ylabel("half-spread h (ticks)")
        axes.set_ylim(-0.1, max(HALF_SPREADS) + 0.1)
        axes.legend(loc="upper left")
    for label, color in zip(REPORTED_STRATEGIES, plotting.SERIES, strict=False):
        estimates = [
            mean_ci(column(base.tuned[label][share], PNL_COMPONENTS["pnl"]))
            for share in INFORMED_SHARES
        ]
        _line_with_band(pnl_axes, estimates, color, label)
    pnl_axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
    pnl_axes.set_title("Tuned strategies: mean PnL per run (95% CI)")
    pnl_axes.set_xlabel("toxicity π")
    pnl_axes.set_ylabel("mean PnL per run (ticks)")
    pnl_axes.legend(loc="upper right")
    plotting.save(figure, out_dir / "e4_breakeven.png")


def plot_decomposition(sweep: ToxicitySweep, out_dir: Path) -> None:
    """Plot the PnL decomposition against pi for the decomposed strategies."""
    figure = plotting.new_figure(11, 4.2)
    panels = figure.subplots(1, len(DECOMPOSED_STRATEGIES), sharey=True)
    parts = {
        "noise_edge": ("edge from noise takers", plotting.SERIES[0]),
        "informed_edge": ("edge from informed takers", plotting.SERIES[1]),
        "inventory": ("inventory PnL", plotting.SERIES[2]),
        "pnl": ("total PnL", plotting.INK),
    }
    for axes, label in zip(panels, DECOMPOSED_STRATEGIES, strict=True):
        for component, (name, color) in parts.items():
            metric = PNL_COMPONENTS[component]
            estimates = [
                mean_ci(column(sweep.tuned[label][share], metric)) for share in INFORMED_SHARES
            ]
            _line_with_band(axes, estimates, color, name)
        axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
        axes.set_title(f"{label}: PnL decomposition by toxicity")
        axes.set_xlabel("toxicity π")
    panels[0].set_ylabel("mean per run (ticks, 95% CI)")
    panels[-1].legend(loc="upper right")
    plotting.save(figure, out_dir / "e4_decomposition.png")


def run_e4(out_dir: Path = RESULTS_DIR) -> tuple[ToxicitySweep, ToxicitySweep]:
    """Run E4 and its jump-frequency sensitivity variant, writing tables and figures."""
    base = run_sweep(DEFAULT_MODEL, E4_SEEDS)
    write_tables(base, out_dir)
    jumps = run_sweep(JUMP_SENSITIVITY_MODEL, SENSITIVITY_SEEDS)
    write_tables(jumps, out_dir, suffix="_jumps")
    plot_half_spread_curves(base, out_dir)
    plot_breakeven(base, jumps, out_dir)
    plot_decomposition(base, out_dir)
    return base, jumps


def _component_estimates(runs: Sequence[RunMetrics]) -> list[float]:
    values = []
    for metric in PNL_COMPONENTS.values():
        estimate = mean_ci(column(runs, metric))
        values += [estimate.mean, estimate.half_width]
    return values


def _mean_fills(runs: Sequence[RunMetrics]) -> float:
    return float(column(runs, lambda run: run.n_fills).mean())


def _line_with_band(axes: Axes, estimates: Sequence[Estimate], color: str, label: str) -> None:
    means = np.array([estimate.mean for estimate in estimates])
    widths = np.array([estimate.half_width for estimate in estimates])
    axes.fill_between(
        INFORMED_SHARES, means - widths, means + widths, color=color, alpha=0.15, linewidth=0
    )
    axes.plot(INFORMED_SHARES, means, color=color, marker="o", markersize=4, label=label)


def main() -> None:
    """Run E4 on the report seeds."""
    run_e4()


if __name__ == "__main__":
    main()
