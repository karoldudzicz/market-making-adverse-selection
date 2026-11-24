"""PnL decomposition, markouts and model diagnostics of a simulation run.

All PnL figures are in ticks times lots, marked to the true value V_t.
"""

from dataclasses import dataclass

import numpy as np

from mmsim.engine import Fill, SimulationResult
from mmsim.market import TakerType

MARKOUT_HORIZONS = (1, 5, 20, 100)
JUMP_WINDOW = 20


@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    """Exact decomposition of the market maker's PnL.

    ``total == noise_edge + informed_edge + inventory`` holds up to floating point error.

    Attributes
    ----------
    total : float
        cash_T + q_T * V_T.
    noise_edge : float
        Sum of s * qty * (V_t - P) over fills against noise takers (spread capture).
    informed_edge : float
        Same sum over fills against informed takers (adverse selection).
    inventory : float
        Sum over steps of q_{t-1} * (V_t - V_{t-1}).
    """

    total: float
    noise_edge: float
    informed_edge: float
    inventory: float


@dataclass(frozen=True, slots=True)
class JumpAttribution:
    """Parts of the PnL associated with jumps of the true value.

    Attributes
    ----------
    informed_edge : float
        Edge against informed takers on fills in the step of a jump or the
        ``JUMP_WINDOW - 1`` steps after it.
    inventory : float
        Jump part of the inventory PnL, the sum over steps of q_{t-1} * J_t.
    """

    informed_edge: float
    inventory: float


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Summary statistics of one run.

    Attributes
    ----------
    pnl : PnLBreakdown
        PnL and its decomposition.
    jumps : JumpAttribution
        Parts of the PnL associated with jumps.
    n_fills, n_noise_fills, n_informed_fills : int
        Number of market maker fills, in total and by counterparty type.
    mean_abs_position : float
        Average |q_t| over the steps of the run.
    max_abs_position : int
        Largest |q_t| over the run.
    mean_abs_mid_error, max_abs_mid_error : float
        Average and largest |mid_t - V_t| of the market mid.
    markouts : dict
        Mean markout per lot by counterparty type and horizon (NaN without fills).
    """

    pnl: PnLBreakdown
    jumps: JumpAttribution
    n_fills: int
    n_noise_fills: int
    n_informed_fills: int
    mean_abs_position: float
    max_abs_position: int
    mean_abs_mid_error: float
    max_abs_mid_error: float
    markouts: dict[TakerType, dict[int, float]]


def pnl_breakdown(result: SimulationResult) -> PnLBreakdown:
    """Decompose the PnL of a run into spread capture, adverse selection and inventory.

    The edge of a fill in step t uses V_t after that step's value move, which happens before
    any trading in the step; with this convention the decomposition is an exact identity.
    """
    values = result.values
    edges = dict.fromkeys(TakerType, 0.0)
    for fill in result.fills:
        edges[fill.counterparty] += _signed_qty(fill) * (values[fill.step] - fill.price)
    inventory = float(np.dot(result.position[:-1], np.diff(values)))
    total = float(result.cash[-1] + result.position[-1] * values[-1])
    return PnLBreakdown(total, edges[TakerType.NOISE], edges[TakerType.INFORMED], inventory)


def jump_attribution(result: SimulationResult, window: int = JUMP_WINDOW) -> JumpAttribution:
    """Attribute informed-flow losses and inventory PnL to jumps of the true value.

    A fill counts as following a jump if it happens in the step of a jump or in the
    ``window - 1`` steps after it.
    """
    after_jump = np.zeros(len(result.values) + window, dtype=bool)
    for step in np.flatnonzero(result.jumps):
        after_jump[step : step + window] = True
    informed_edge = sum(
        _signed_qty(fill) * (result.values[fill.step] - fill.price)
        for fill in result.fills
        if fill.counterparty is TakerType.INFORMED and after_jump[fill.step]
    )
    inventory = float(np.dot(result.position[:-1], result.jumps[1:]))
    return JumpAttribution(float(informed_edge), inventory)


def markouts(
    result: SimulationResult, horizons: tuple[int, ...] = MARKOUT_HORIZONS
) -> dict[TakerType, dict[int, float]]:
    """Mean per-lot markout s * (V_{t+h} - P) by counterparty type and horizon.

    Fills with t + h beyond the end of the run are excluded for that horizon.
    """
    values = result.values
    last_step = len(values) - 1
    table: dict[TakerType, dict[int, float]] = {}
    for taker in TakerType:
        fills = [fill for fill in result.fills if fill.counterparty is taker]
        table[taker] = {}
        for horizon in horizons:
            usable = [fill for fill in fills if fill.step + horizon <= last_step]
            lots = sum(fill.qty for fill in usable)
            total = sum(_signed_qty(f) * (values[f.step + horizon] - f.price) for f in usable)
            table[taker][horizon] = total / lots if lots else float("nan")
    return table


def compute_metrics(result: SimulationResult) -> RunMetrics:
    """Compute all summary statistics of a run."""
    abs_position = np.abs(result.position[1:])
    mid_error = np.abs(result.market_mid[1:] - result.values[1:])
    return RunMetrics(
        pnl=pnl_breakdown(result),
        jumps=jump_attribution(result),
        n_fills=len(result.fills),
        n_noise_fills=sum(f.counterparty is TakerType.NOISE for f in result.fills),
        n_informed_fills=sum(f.counterparty is TakerType.INFORMED for f in result.fills),
        mean_abs_position=float(abs_position.mean()),
        max_abs_position=int(abs_position.max()),
        mean_abs_mid_error=float(mid_error.mean()),
        max_abs_mid_error=float(mid_error.max()),
        markouts=markouts(result),
    )


def _signed_qty(fill: Fill) -> int:
    return int(fill.side) * fill.qty
