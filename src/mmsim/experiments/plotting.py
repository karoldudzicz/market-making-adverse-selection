"""Shared figure style; figures are drawn on the Agg canvas without pyplot."""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.style
from matplotlib.figure import Figure

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")

_STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}


def new_figure(width: float, height: float) -> Figure:
    """Create a figure with the project style and constrained layout."""
    matplotlib.use("Agg")
    matplotlib.style.use(_STYLE)
    return Figure(figsize=(width, height), layout="constrained")


def save(figure: Figure, path: Path) -> None:
    """Save a figure as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
