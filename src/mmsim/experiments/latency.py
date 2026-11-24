"""E5: cost of quote latency L for each reported strategy.

Run with ``python -m mmsim.experiments.latency``. Changes are measured against L = 0 on the
same seeds (paired differences).
"""

from pathlib import Path

from mmsim.config import REPORTED_STRATEGIES, SimulationConfig
from mmsim.experiments import plotting
from mmsim.experiments.common import (
    DEFAULT_MODEL,
    PNL_COMPONENTS,
    REPORT_SEEDS,
    RESULTS_DIR,
    column,
    mean_ci,
    run_groups,
    with_latency,
    write_csv,
)
from mmsim.metrics import RunMetrics

LATENCIES = (0, 1, 2, 5, 10)
PLOTTED_CHANGES = {
    "pnl": "Change in mean PnL vs L = 0",
    "informed_edge": "Change in edge from informed takers vs L = 0",
    "noise_edge": "Change in edge from noise takers vs L = 0",
}

type LatencyResults = dict[str, dict[int, list[RunMetrics]]]


def run_latency(
    model: SimulationConfig = DEFAULT_MODEL, seeds: range = REPORT_SEEDS
) -> LatencyResults:
    """Run every reported strategy at every latency on the same seeds."""
    cells = [(label, latency) for label in REPORTED_STRATEGIES for latency in LATENCIES]
    groups = [
        (with_latency(model, latency), REPORTED_STRATEGIES[label]) for label, latency in cells
    ]
    results: LatencyResults = {label: {} for label in REPORTED_STRATEGIES}
    for (label, latency), runs in zip(cells, run_groups(groups, seeds), strict=True):
        results[label][latency] = runs
    return results


def write_table(results: LatencyResults, out_dir: Path) -> None:
    """Write levels and paired changes against L = 0 of the PnL decomposition."""
    header = ["strategy", "latency"]
    for name in PNL_COMPONENTS:
        header += [name, f"{name}_ci95", f"change_{name}", f"change_{name}_ci95"]
    header += ["fills_per_run", "informed_fills_per_run", "informed_edge_after_jumps"]
    rows = []
    for label, by_latency in results.items():
        for latency, runs in by_latency.items():
            row: list[object] = [label, latency]
            for metric in PNL_COMPONENTS.values():
                level = mean_ci(column(runs, metric))
                change = mean_ci(column(runs, metric) - column(by_latency[0], metric))
                row += [level.mean, level.half_width, change.mean, change.half_width]
            row += [
                float(column(runs, lambda run: run.n_fills).mean()),
                float(column(runs, lambda run: run.n_informed_fills).mean()),
                float(column(runs, lambda run: run.jumps.informed_edge).mean()),
            ]
            rows.append(row)
    write_csv(out_dir / "e5_latency.csv", header, rows)


def plot_latency(results: LatencyResults, out_dir: Path) -> None:
    """Plot paired changes of PnL and its edge components against latency."""
    figure = plotting.new_figure(14, 4.2)
    panels = figure.subplots(1, len(PLOTTED_CHANGES), sharey=True)
    for axes, (component, title) in zip(panels, PLOTTED_CHANGES.items(), strict=True):
        metric = PNL_COMPONENTS[component]
        for (label, by_latency), color in zip(results.items(), plotting.SERIES, strict=False):
            baseline = column(by_latency[0], metric)
            changes = [mean_ci(column(by_latency[lat], metric) - baseline) for lat in LATENCIES]
            axes.errorbar(
                LATENCIES,
                [change.mean for change in changes],
                yerr=[change.half_width for change in changes],
                color=color,
                marker="o",
                markersize=4,
                capsize=2,
                label=plotting.strategy_label(label),
            )
        axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
        axes.set_xticks(LATENCIES)
        axes.set_title(title)
        axes.set_xlabel("latency L (steps)")
    panels[0].set_ylabel("paired change per run (ticks, 95% CI)")
    handles, names = panels[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="outside lower center", ncols=2)
    plotting.save(figure, out_dir / "e5_latency.png")


def run_e5(out_dir: Path = RESULTS_DIR) -> LatencyResults:
    """Run E5 and write its table and figure."""
    results = run_latency()
    write_table(results, out_dir)
    plot_latency(results, out_dir)
    return results


def main() -> None:
    """Run E5 on the report seeds."""
    run_e5()


if __name__ == "__main__":
    main()
