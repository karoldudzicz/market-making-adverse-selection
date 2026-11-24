"""E1, E2, E3 and E6: comparison of the reported strategies at the default model.

Run with ``python -m mmsim.experiments.compare``; all runs use the report seeds.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from mmsim.config import REPORTED_STRATEGIES, SimulationConfig
from mmsim.engine import SimulationResult, simulate
from mmsim.experiments import plotting
from mmsim.experiments.common import (
    DEFAULT_MODEL,
    PNL_COMPONENTS,
    REPORT_SEEDS,
    RESULTS_DIR,
    column,
    mean_ci,
    run_groups,
    write_csv,
)
from mmsim.market import TakerType
from mmsim.metrics import MARKOUT_HORIZONS, RunMetrics
from mmsim.orderbook import Side
from mmsim.strategies import build_strategy

type StrategyRuns = dict[str, list[RunMetrics]]

BASELINE = "Naive"
DECOMPOSITION: dict[str, tuple[str, Callable[[RunMetrics], float]]] = {
    "noise_edge": ("edge from noise takers", PNL_COMPONENTS["noise_edge"]),
    "informed_edge": ("edge from informed takers", PNL_COMPONENTS["informed_edge"]),
    "inventory": ("inventory PnL", PNL_COMPONENTS["inventory"]),
}
JUMP_PARTS: dict[str, Callable[[RunMetrics], float]] = {
    "informed_edge_after_jumps": lambda run: run.jumps.informed_edge,
    "inventory_from_jumps": lambda run: run.jumps.inventory,
}


def run_comparison(
    model: SimulationConfig = DEFAULT_MODEL, seeds: range = REPORT_SEEDS
) -> StrategyRuns:
    """Run every reported strategy on the same seeds."""
    groups = [(model, strategy) for strategy in REPORTED_STRATEGIES.values()]
    return dict(zip(REPORTED_STRATEGIES, run_groups(groups, seeds), strict=True))


def write_summary(results: StrategyRuns, out_dir: Path) -> None:
    """Write E2: PnL distribution statistics and paired differences against the baseline."""
    header = [
        "strategy",
        "mean_pnl",
        "pnl_ci95",
        "std_pnl",
        "sharpe",
        "median_pnl",
        "p05_pnl",
        "p95_pnl",
        "share_of_losing_runs",
        "fills_per_run",
        "informed_fills_per_run",
        "mean_abs_position",
        "mean_max_abs_position",
    ]
    rows = []
    for label, runs in results.items():
        pnl = column(runs, PNL_COMPONENTS["pnl"])
        estimate = mean_ci(pnl)
        rows.append(
            [
                label,
                estimate.mean,
                estimate.half_width,
                float(pnl.std(ddof=1)),
                estimate.mean / float(pnl.std(ddof=1)),
                float(np.median(pnl)),
                float(np.percentile(pnl, 5)),
                float(np.percentile(pnl, 95)),
                float(np.mean(pnl < 0)),
                float(column(runs, lambda run: run.n_fills).mean()),
                float(column(runs, lambda run: run.n_informed_fills).mean()),
                float(column(runs, lambda run: run.mean_abs_position).mean()),
                float(column(runs, lambda run: run.max_abs_position).mean()),
            ]
        )
    write_csv(out_dir / "e2_strategy_summary.csv", header, rows)

    baseline = column(results[BASELINE], PNL_COMPONENTS["pnl"])
    difference_rows = []
    for label, runs in results.items():
        if label == BASELINE:
            continue
        difference = column(runs, PNL_COMPONENTS["pnl"]) - baseline
        estimate = mean_ci(difference)
        difference_rows.append(
            [label, BASELINE, estimate.mean, estimate.half_width, float(np.mean(difference > 0))]
        )
    write_csv(
        out_dir / "e2_paired_differences.csv",
        ["strategy", "baseline", "mean_pnl_difference", "ci95", "share_of_seeds_better"],
        difference_rows,
    )


def plot_distribution(results: StrategyRuns, out_dir: Path) -> None:
    """Plot E2: distribution of run PnL per strategy."""
    figure = plotting.new_figure(8, 4.5)
    axes = figure.subplots()
    labels = list(results)
    data = [column(results[label], PNL_COMPONENTS["pnl"]) for label in labels]
    boxes = axes.boxplot(
        data,
        tick_labels=labels,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": plotting.INK, "linewidth": 1.5},
        meanprops={
            "marker": "D",
            "markerfacecolor": plotting.INK,
            "markeredgecolor": plotting.SURFACE,
        },
        whiskerprops={"color": plotting.INK_SECONDARY},
        capprops={"color": plotting.INK_SECONDARY},
    )
    for patch, color in zip(boxes["boxes"], plotting.SERIES, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for position, pnl in enumerate(data, start=1):
        estimate = mean_ci(pnl)
        axes.annotate(
            f"mean {estimate.mean:.0f} ± {estimate.half_width:.0f}\nstd {pnl.std(ddof=1):.0f}",
            (position, float(np.percentile(pnl, 95))),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=plotting.INK_SECONDARY,
        )
    axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
    axes.set_title(
        f"PnL per run, {len(data[0])} seeds, π = {DEFAULT_MODEL.takers.informed_share}, L = 0 "
        "(box: quartiles, whiskers: 5-95%, diamond: mean)"
    )
    axes.set_ylabel("PnL per run (ticks)")
    axes.set_ylim(top=max(float(np.percentile(pnl, 95)) for pnl in data) * 1.12)
    plotting.save(figure, out_dir / "e2_pnl_distribution.png")


def write_decomposition(results: StrategyRuns, out_dir: Path) -> None:
    """Write E3: mean PnL components per strategy, including the parts tied to jumps."""
    parts = {**PNL_COMPONENTS, **JUMP_PARTS}
    rows = []
    for label, runs in results.items():
        for name, metric in parts.items():
            estimate = mean_ci(column(runs, metric))
            rows.append([label, name, estimate.mean, estimate.half_width])
    write_csv(out_dir / "e3_pnl_decomposition.csv", ["strategy", "component", "mean", "ci95"], rows)


def plot_decomposition(results: StrategyRuns, out_dir: Path) -> None:
    """Plot E3: PnL components per strategy as grouped bars with the total marked."""
    figure = plotting.new_figure(9, 4.8)
    axes = figure.subplots()
    labels = list(results)
    positions = np.arange(len(labels))
    width = 0.26
    for offset, ((name, metric), color) in enumerate(
        zip(DECOMPOSITION.values(), plotting.SERIES, strict=False)
    ):
        estimates = [mean_ci(column(results[label], metric)) for label in labels]
        x = positions + (offset - 1) * width
        means = [estimate.mean for estimate in estimates]
        axes.bar(x, means, width * 0.92, color=color, label=name)
        axes.errorbar(
            x,
            means,
            yerr=[estimate.half_width for estimate in estimates],
            fmt="none",
            ecolor=plotting.INK_SECONDARY,
            capsize=2,
            linewidth=1,
        )
        for xi, estimate in zip(x, estimates, strict=True):
            mean = estimate.mean
            axes.annotate(
                f"{mean:.0f}",
                (xi, mean + (estimate.half_width if mean >= 0 else -estimate.half_width)),
                xytext=(0, 3 if mean >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=plotting.INK_SECONDARY,
            )
    totals = [mean_ci(column(results[label], PNL_COMPONENTS["pnl"])) for label in labels]
    axes.plot(
        positions,
        [total.mean for total in totals],
        "D",
        color=plotting.INK,
        markeredgecolor=plotting.SURFACE,
        markersize=7,
        label="total PnL",
    )
    axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
    axes.set_xticks(positions, labels)
    axes.set_title(
        f"Where the PnL comes from: mean per run over {len(results[labels[0]])} seeds "
        f"(π = {DEFAULT_MODEL.takers.informed_share}, L = 0, 95% CI)"
    )
    axes.set_ylabel("ticks per run")
    figure.legend(loc="outside lower center", ncols=4)
    plotting.save(figure, out_dir / "e3_pnl_decomposition.png")


def markout_curves(results: StrategyRuns) -> dict[str, dict[TakerType, list[float]]]:
    """Mean markout per lot at horizon 0 and ``MARKOUT_HORIZONS`` by strategy and counterparty.

    Horizon 0 is the edge at the fill. Per-run means are pooled with weights equal to the run's
    number of fills against that counterparty type.
    """
    edges = {
        TakerType.NOISE: PNL_COMPONENTS["noise_edge"],
        TakerType.INFORMED: PNL_COMPONENTS["informed_edge"],
    }
    counts: dict[TakerType, Callable[[RunMetrics], float]] = {
        TakerType.NOISE: lambda run: run.n_noise_fills,
        TakerType.INFORMED: lambda run: run.n_informed_fills,
    }
    curves: dict[str, dict[TakerType, list[float]]] = {}
    for label, runs in results.items():
        curves[label] = {}
        for taker in TakerType:
            weights = column(runs, counts[taker])
            curve = [float(column(runs, edges[taker]).sum() / weights.sum())]
            for horizon in MARKOUT_HORIZONS:
                values = np.array([run.markouts[taker][horizon] for run in runs])
                usable = ~np.isnan(values)
                curve.append(float(np.average(values[usable], weights=weights[usable])))
            curves[label][taker] = curve
    return curves


def write_markouts(results: StrategyRuns, out_dir: Path) -> None:
    """Write E6: markout curves by strategy and counterparty type."""
    rows = [
        [label, taker.value, horizon, value]
        for label, by_taker in markout_curves(results).items()
        for taker, curve in by_taker.items()
        for horizon, value in zip((0, *MARKOUT_HORIZONS), curve, strict=True)
    ]
    write_csv(
        out_dir / "e6_markouts.csv",
        ["strategy", "counterparty", "horizon", "markout_per_lot"],
        rows,
    )


def plot_markouts(results: StrategyRuns, out_dir: Path) -> None:
    """Plot E6: markout per lot against horizon, one panel per counterparty type."""
    curves = markout_curves(results)
    horizons = (0, *MARKOUT_HORIZONS)
    figure = plotting.new_figure(11, 4.2)
    panels = figure.subplots(1, len(TakerType))
    for axes, taker in zip(panels, TakerType, strict=True):
        for (label, by_taker), color in zip(curves.items(), plotting.SERIES, strict=False):
            axes.plot(horizons, by_taker[taker], color=color, marker="o", markersize=4, label=label)
        axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
        axes.set_xscale("symlog", linthresh=1)
        axes.set_xlim(0, 120)
        axes.set_xticks(horizons, [str(h) for h in horizons])
        axes.set_title(f"Markout s·(V(t+h) − P) per lot, fills against {taker.value} takers")
        axes.set_xlabel("horizon h (steps after the fill)")
        axes.set_ylabel("ticks per lot")
    panels[0].legend(loc="lower right")
    plotting.save(figure, out_dir / "e6_markouts.png")


def plot_example_run(out_dir: Path = RESULTS_DIR, seed: int = REPORT_SEEDS.start) -> None:
    """Plot E1: one A-S (Sharpe) run around a jump, with positions and PnL against Naive."""
    focus, contrast = "A-S (Sharpe)", "Naive"
    colors = dict(zip(REPORTED_STRATEGIES, plotting.SERIES, strict=False))
    runs = {
        label: simulate(DEFAULT_MODEL, build_strategy(REPORTED_STRATEGIES[label]), seed)
        for label in (contrast, focus)
    }
    result = runs[focus]
    n_steps = DEFAULT_MODEL.n_steps
    candidates = [int(j) for j in np.flatnonzero(result.jumps) if 300 <= j <= n_steps - 300]
    jump = candidates[0] if candidates else n_steps // 2
    window = (jump - 60, jump + 120)

    figure = plotting.new_figure(11, 8.5)
    prices, positions, pnl = figure.subplots(3, 1, height_ratios=(1.4, 1, 1))
    _plot_quotes_window(prices, result, window, colors[focus])
    prices.set_title(f"{focus} around a price jump (seed {seed}, step {jump})")
    steps = np.arange(n_steps + 1)
    for label, run in runs.items():
        positions.plot(steps, run.position, color=colors[label], linewidth=1.0, label=label)
        pnl.plot(steps, run.cash + run.position * run.values, color=colors[label], label=label)
    for axes in (positions, pnl):
        axes.axvspan(*window, color=plotting.GRID, alpha=0.6, linewidth=0)
        axes.axhline(0, color=plotting.AXIS, linewidth=1.0)
        axes.set_xlim(0, n_steps)
        axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    positions.set_title("Position q (shaded: window above)")
    positions.set_ylabel("lots")
    pnl.set_title("Cumulative PnL marked to V")
    pnl.set_ylabel("ticks")
    pnl.set_xlabel("step")
    plotting.save(figure, out_dir / "e1_example_run.png")


def _plot_quotes_window(
    axes: Axes, result: SimulationResult, window: tuple[int, int], quote_color: str
) -> None:
    start, end = window
    steps = np.arange(start, end + 1)
    span = slice(start, end + 1)
    axes.plot(steps, result.values[span], color=plotting.INK, label="true value V")
    axes.step(
        steps, result.market_mid[span], where="post", color=plotting.MUTED, label="market mid"
    )
    axes.step(
        steps, result.bid[span], where="post", color=quote_color, linewidth=1.2, label="MM bid, ask"
    )
    axes.step(steps, result.ask[span], where="post", color=quote_color, linewidth=1.2)
    taker_colors = {TakerType.NOISE: plotting.INK_SECONDARY, TakerType.INFORMED: plotting.SERIES[1]}
    for taker, color in taker_colors.items():
        for side, marker, verb in ((Side.BUY, "^", "buys"), (Side.SELL, "v", "sells")):
            fills = [
                fill
                for fill in result.fills
                if fill.counterparty is taker and fill.side is side and start <= fill.step <= end
            ]
            if not fills:
                continue
            axes.plot(
                [fill.step for fill in fills],
                [fill.price for fill in fills],
                marker,
                color=color,
                markersize=7,
                linestyle="none",
                label=f"MM {verb} from {taker.value} taker",
            )
    axes.set_ylabel("price (ticks)")
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))


def run_e2_e3_e6(out_dir: Path = RESULTS_DIR) -> StrategyRuns:
    """Run the strategy comparison and write E2, E3 and E6."""
    results = run_comparison()
    write_summary(results, out_dir)
    plot_distribution(results, out_dir)
    write_decomposition(results, out_dir)
    plot_decomposition(results, out_dir)
    write_markouts(results, out_dir)
    plot_markouts(results, out_dir)
    return results


def main() -> None:
    """Run E1, E2, E3 and E6 on the report seeds."""
    run_e2_e3_e6()
    plot_example_run()


if __name__ == "__main__":
    main()
