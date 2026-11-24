"""E0: model diagnostics and calibration of the fill intensity decay k.

Run with ``python -m mmsim.experiments.calibrate``; uses the tuning seeds only.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mmsim.config import NaiveConfig, SimulationConfig
from mmsim.engine import simulate
from mmsim.experiments import plotting
from mmsim.experiments.common import (
    RESULTS_DIR,
    TUNING_SEEDS,
    RunSpec,
    mean_ci,
    parallel_map,
    run_metrics,
    write_csv,
)
from mmsim.market import INFORMED_TAKER_ID
from mmsim.metrics import RunMetrics
from mmsim.orderbook import Side
from mmsim.strategies import Quote, QuoteState, build_strategy

CALIBRATION_HALF_SPREADS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
REFERENCE_HALF_SPREAD = 1.0
MIN_BUCKET_FILLS = 20
DISTANCE_BUCKETS = 41
JUMP_BEFORE = 10
JUMP_AFTER = 100
RECOVERY_TOLERANCE = 1.25
ROLLING_WINDOW = 100


@dataclass(frozen=True, slots=True)
class IntensityFit:
    """Weighted least-squares fit of the fill intensity lambda(delta) = A * exp(-k * delta).

    Attributes
    ----------
    amplitude : float
        A, fills per side per step at zero distance from the mid.
    k : float
        Decay rate per tick of distance.
    r_squared : float
        Weighted R^2 of the fit on the log scale.
    """

    amplitude: float
    k: float
    r_squared: float


@dataclass(frozen=True, slots=True)
class _MarketRun:
    mid_error: NDArray[np.float64]
    jump_steps: NDArray[np.intp]
    n_trades: int
    n_informed_trades: int


class _NoQuotes:
    def quote(self, state: QuoteState) -> Quote:
        return None, None


def cancel_rule_variants(config: SimulationConfig) -> dict[str, SimulationConfig]:
    """Return the model with the through-reference cancel rule switched on and off."""
    return {
        label: replace(
            config, background=replace(config.background, cancel_through_reference=enabled)
        )
        for label, enabled in (("on", True), ("off", False))
    }


def fit_intensity(
    distance: NDArray[np.float64], quoted: NDArray[np.int64], fills: NDArray[np.int64]
) -> IntensityFit:
    """Fit log lambda = log A - k * delta over distance buckets.

    The variance of a log rate estimated from n fills is about 1/n, so buckets are weighted by
    their fill count; buckets with fewer than ``MIN_BUCKET_FILLS`` fills are ignored.
    """
    usable = fills >= MIN_BUCKET_FILLS
    x = distance[usable]
    y = np.log(fills[usable] / quoted[usable])
    weights = fills[usable].astype(np.float64)
    slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(weights))
    residual = y - (intercept + slope * x)
    spread = y - np.average(y, weights=weights)
    r_squared = 1.0 - np.sum(weights * residual**2) / np.sum(weights * spread**2)
    return IntensityFit(float(np.exp(intercept)), float(-slope), float(r_squared))


def jump_response(runs: list[_MarketRun]) -> tuple[NDArray[np.float64], int]:
    """Mean |mid - V| from ``JUMP_BEFORE`` steps before to ``JUMP_AFTER`` steps after a jump.

    Only isolated jumps, with no other jump within ``JUMP_AFTER`` steps, are used. Returns the
    mean response and the number of jumps behind it.
    """
    windows = []
    for run in runs:
        for jump in run.jump_steps:
            fits = JUMP_BEFORE < jump < run.mid_error.size - JUMP_AFTER
            isolated = np.sum(np.abs(run.jump_steps - jump) <= JUMP_AFTER) == 1
            if fits and isolated:
                windows.append(run.mid_error[jump - JUMP_BEFORE : jump + JUMP_AFTER + 1])
    return np.mean(windows, axis=0), len(windows)


def recovery_steps(response: NDArray[np.float64], baseline: float) -> int | None:
    """First step after a jump at which the mean error is within 25% of its unconditional mean."""
    recovered = np.flatnonzero(response[JUMP_BEFORE + 1 :] <= RECOVERY_TOLERANCE * baseline)
    return int(recovered[0]) + 1 if recovered.size else None


def run_e0(seeds: range = TUNING_SEEDS, out_dir: Path = RESULTS_DIR) -> IntensityFit:
    """Run E0, write its tables and figures to ``out_dir`` and return the fitted intensity."""
    variants = cancel_rule_variants(SimulationConfig())
    market_runs = {
        label: parallel_map(_market_run, [(config, seed) for seed in seeds])
        for label, config in variants.items()
    }
    reference = NaiveConfig(half_spread=REFERENCE_HALF_SPREAD)
    maker_runs = {
        label: parallel_map(run_metrics, [RunSpec(config, reference, seed) for seed in seeds])
        for label, config in variants.items()
    }
    _write_diagnostics(market_runs, maker_runs, out_dir)
    _plot_mid_error(market_runs, out_dir)

    calibration = [
        RunSpec(variants["on"], NaiveConfig(half_spread=h), seed)
        for h in CALIBRATION_HALF_SPREADS
        for seed in seeds
    ]
    counts = np.sum(parallel_map(_fill_counts, calibration), axis=0)
    distance = np.arange(DISTANCE_BUCKETS) / 2
    fit = fit_intensity(distance, counts[0], counts[1])
    _write_intensity(distance, counts, fit, out_dir)
    _plot_intensity(distance, counts, fit, out_dir)
    return fit


def _market_run(spec: tuple[SimulationConfig, int]) -> _MarketRun:
    config, seed = spec
    result = simulate(config, _NoQuotes(), seed)
    informed = sum(INFORMED_TAKER_ID in (t.buyer_id, t.seller_id) for t in result.trades)
    return _MarketRun(
        mid_error=np.abs(result.market_mid - result.values),
        jump_steps=np.flatnonzero(result.jumps),
        n_trades=len(result.trades),
        n_informed_trades=informed,
    )


def _fill_counts(spec: RunSpec) -> NDArray[np.int64]:
    """Count quoted steps (row 0) and fills (row 1) per side by distance from the mid.

    The distance of a quote active in step t is measured from the background mid at the end of
    step t - 1, the mid it was computed from at zero latency, in half-tick buckets.
    """
    result = simulate(spec.config, build_strategy(spec.strategy), spec.seed)
    filled = {side: np.zeros(spec.config.n_steps + 1, dtype=np.int64) for side in Side}
    for fill in result.fills:
        filled[fill.side][fill.step] += fill.qty
    mid_before = result.background_mid[:-1]
    distances = {Side.BUY: mid_before - result.bid[1:], Side.SELL: result.ask[1:] - mid_before}
    counts = np.zeros((2, DISTANCE_BUCKETS), dtype=np.int64)
    for side, distance in distances.items():
        quoted = ~np.isnan(distance)
        buckets = np.rint(2 * distance[quoted]).astype(np.intp)
        valid = (buckets >= 0) & (buckets < DISTANCE_BUCKETS)
        np.add.at(counts[0], buckets[valid], 1)
        np.add.at(counts[1], buckets[valid], filled[side][1:][quoted][valid])
    return counts


def _write_diagnostics(
    market_runs: dict[str, list[_MarketRun]],
    maker_runs: dict[str, list[RunMetrics]],
    out_dir: Path,
) -> None:
    header = [
        "cancel_through_reference",
        "mean_abs_mid_error",
        "p95_abs_mid_error",
        "max_abs_mid_error",
        "share_of_steps_error_above_5",
        "jump_recovery_steps",
        "isolated_jumps",
        "market_trades_per_run",
        "informed_trades_per_run",
        "mm_fills_per_run",
        "mm_informed_fills_per_run",
        "mm_mean_pnl",
        "mm_pnl_ci95",
    ]
    rows = []
    for label, runs in market_runs.items():
        errors = np.concatenate([run.mid_error[1:] for run in runs])
        response, n_jumps = jump_response(runs)
        recovery = recovery_steps(response, float(errors.mean()))
        metrics = maker_runs[label]
        pnl = mean_ci([m.pnl.total for m in metrics])
        rows.append(
            [
                label,
                float(errors.mean()),
                float(np.percentile(errors, 95)),
                float(errors.max()),
                float(np.mean(errors > 5)),
                recovery if recovery is not None else f">{JUMP_AFTER}",
                n_jumps,
                float(np.mean([run.n_trades for run in runs])),
                float(np.mean([run.n_informed_trades for run in runs])),
                float(np.mean([m.n_fills for m in metrics])),
                float(np.mean([m.n_informed_fills for m in metrics])),
                pnl.mean,
                pnl.half_width,
            ]
        )
    write_csv(out_dir / "e0_model_diagnostics.csv", header, rows)


def _write_intensity(
    distance: NDArray[np.float64], counts: NDArray[np.int64], fit: IntensityFit, out_dir: Path
) -> None:
    quoted, fills = counts
    observed = quoted > 0
    rows = [
        [
            float(d),
            int(q),
            int(f),
            float(f / q),
            fit.amplitude * float(np.exp(-fit.k * d)),
            bool(f >= MIN_BUCKET_FILLS),
        ]
        for d, q, f in zip(distance[observed], quoted[observed], fills[observed], strict=True)
    ]
    header = ["distance_ticks", "quoted_side_steps", "fills", "fill_rate", "fitted_rate", "in_fit"]
    write_csv(out_dir / "e0_fill_intensity.csv", header, rows)
    write_csv(
        out_dir / "e0_intensity_fit.csv",
        ["amplitude", "k", "r_squared"],
        [[fit.amplitude, fit.k, fit.r_squared]],
    )


def _plot_mid_error(market_runs: dict[str, list[_MarketRun]], out_dir: Path) -> None:
    figure = plotting.new_figure(11, 4)
    over_time, around_jumps = figure.subplots(1, 2)
    labels = {"on": "through-reference cancel on (default)", "off": "off (base specification)"}
    kernel = np.ones(ROLLING_WINDOW) / ROLLING_WINDOW
    for (label, runs), color in zip(market_runs.items(), plotting.SERIES, strict=False):
        errors = np.array([run.mid_error[1:] for run in runs])
        rolling = np.convolve(errors.mean(axis=0), kernel, mode="valid")
        steps = np.arange(ROLLING_WINDOW, ROLLING_WINDOW + rolling.size)
        over_time.plot(steps, rolling, color=color, label=f"{labels[label]}: {errors.mean():.2f}")
        response, n_jumps = jump_response(runs)
        recovery = recovery_steps(response, float(errors.mean()))
        lags = np.arange(-JUMP_BEFORE, JUMP_AFTER + 1)
        around_jumps.plot(lags, response, color=color, label=f"{labels[label]}: {recovery} steps")
        around_jumps.axhline(errors.mean(), color=color, linewidth=0.8, linestyle=(0, (4, 3)))
    over_time.set_ylim(0, None)
    over_time.set_title(f"Mean |mid − V| over {len(errors)} seeds, {ROLLING_WINDOW}-step rolling")
    over_time.set_xlabel("step")
    over_time.set_ylabel("ticks")
    over_time.legend(loc="lower right", title="overall mean", title_fontsize=8)
    around_jumps.axvline(0, color=plotting.MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    around_jumps.set_ylim(0, None)
    around_jumps.set_title(f"Mean |mid − V| around {n_jumps} isolated jumps (no market maker)")
    around_jumps.set_xlabel("steps since jump")
    around_jumps.set_ylabel("ticks")
    around_jumps.legend(loc="upper right", title="back to normal after", title_fontsize=8)
    plotting.save(figure, out_dir / "e0_mid_error.png")


def _plot_intensity(
    distance: NDArray[np.float64], counts: NDArray[np.int64], fit: IntensityFit, out_dir: Path
) -> None:
    quoted, fills = counts
    used = fills >= MIN_BUCKET_FILLS
    rate = fills[used] / quoted[used]
    figure = plotting.new_figure(6, 4)
    axes = figure.subplots()
    axes.semilogy(distance[used], rate, "o", color=plotting.SERIES[0], label="observed")
    grid = np.linspace(0, distance[used].max(), 200)
    fit_label = (
        f"fit A·exp(−kδ): A = {fit.amplitude:.3f}, k = {fit.k:.3f}, R² = {fit.r_squared:.2f}"
    )
    axes.semilogy(
        grid, fit.amplitude * np.exp(-fit.k * grid), color=plotting.SERIES[1], label=fit_label
    )
    axes.set_title("Fill rate per side per step vs distance from background mid")
    axes.set_xlabel("quote distance from background mid δ (ticks)")
    axes.set_ylabel("fills per side per step")
    axes.legend(loc="lower left")
    plotting.save(figure, out_dir / "e0_fill_intensity.png")


def main() -> None:
    """Run E0 on the tuning seeds and print the calibrated k."""
    fit = run_e0()
    print(f"A = {fit.amplitude:.4f}, k = {fit.k:.4f}, R^2 = {fit.r_squared:.3f}")


if __name__ == "__main__":
    main()
