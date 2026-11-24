"""Quoting strategies and their common interface."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class QuoteState:
    """Information available to a strategy when it computes quotes.

    Attributes
    ----------
    mid : float
        Background mid: mid of the book excluding the market maker's own orders.
    position : int
        Current market maker position in lots.
    step : int
        Current simulation step (1-based).
    steps_remaining : int
        Steps left until the end of the run, T - t.
    sigma : float
        Per-step volatility of the true value, known to the market maker.
    """

    mid: float
    position: int
    step: int
    steps_remaining: int
    sigma: float


Quote = tuple[int | None, int | None]
"""Bid and ask prices in ticks; ``None`` leaves that side unquoted."""


class Strategy(Protocol):
    """Common interface of all quoting strategies."""

    def quote(self, state: QuoteState) -> Quote:
        """Return the (bid, ask) prices in ticks for the given state."""
        ...
