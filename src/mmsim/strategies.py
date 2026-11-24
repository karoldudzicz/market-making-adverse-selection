"""Quoting strategies and their common interface."""

import math
from dataclasses import dataclass
from typing import Protocol, assert_never

from mmsim.config import (
    AvellanedaStoikovConfig,
    InventorySkewConfig,
    NaiveConfig,
    StrategyConfig,
)

_ROUNDING_TOLERANCE = 1e-9


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


type Quote = tuple[int | None, int | None]
"""Bid and ask prices in ticks; ``None`` leaves that side unquoted."""


class Strategy(Protocol):
    """Common interface of all quoting strategies."""

    def quote(self, state: QuoteState) -> Quote:
        """Return the (bid, ask) prices in ticks for the given state."""
        ...


class Naive:
    """Symmetric quotes around the background mid: bid = mid - h, ask = mid + h.

    Parameters
    ----------
    config : NaiveConfig
        Strategy parameters.
    """

    def __init__(self, config: NaiveConfig) -> None:
        self._half_spread = config.half_spread

    def quote(self, state: QuoteState) -> Quote:
        """Return symmetric quotes around the mid."""
        return to_ticks(state.mid - self._half_spread, state.mid + self._half_spread)


class InventorySkew:
    """Symmetric quotes around a center shifted against the position: mid - alpha * q.

    Parameters
    ----------
    config : InventorySkewConfig
        Strategy parameters.
    """

    def __init__(self, config: InventorySkewConfig) -> None:
        self._half_spread = config.half_spread
        self._skew = config.skew

    def quote(self, state: QuoteState) -> Quote:
        """Return quotes around the position-skewed center."""
        center = state.mid - self._skew * state.position
        return to_ticks(center - self._half_spread, center + self._half_spread)


class AvellanedaStoikov:
    """Avellaneda-Stoikov (2008) quotes around the reservation price.

    Reservation price r = s - q * gamma * sigma^2 * tau and total spread
    delta = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k); quotes are r -/+ delta / 2.
    tau is the constant rolling horizon from the config, or T - t when it is ``None``.

    Parameters
    ----------
    config : AvellanedaStoikovConfig
        Strategy parameters.
    """

    def __init__(self, config: AvellanedaStoikovConfig) -> None:
        self._gamma = config.gamma
        self._horizon = config.horizon
        self._intensity_spread = (2.0 / config.gamma) * math.log1p(config.gamma / config.k)

    def reservation_price(self, state: QuoteState) -> float:
        """Return the reservation price r = s - q * gamma * sigma^2 * tau."""
        return state.mid - state.position * self._risk_term(state)

    def spread(self, state: QuoteState) -> float:
        """Return the optimal total spread delta."""
        return self._risk_term(state) + self._intensity_spread

    def quote(self, state: QuoteState) -> Quote:
        """Return quotes symmetric around the reservation price."""
        reservation = self.reservation_price(state)
        half_spread = self.spread(state) / 2
        return to_ticks(reservation - half_spread, reservation + half_spread)

    def _risk_term(self, state: QuoteState) -> float:
        tau = state.steps_remaining if self._horizon is None else self._horizon
        return self._gamma * state.sigma**2 * tau


def to_ticks(bid: float, ask: float) -> tuple[int, int]:
    """Round quotes outward to whole ticks (bid down, ask up), keeping bid < ask.

    Parameters
    ----------
    bid, ask : float
        Quote prices in ticks.

    Returns
    -------
    tuple[int, int]
        Bid and ask in whole ticks.
    """
    # The tolerance stops float noise (e.g. 99.99999999999999) from widening a quote by a tick.
    bid_ticks = math.floor(bid + _ROUNDING_TOLERANCE)
    ask_ticks = math.ceil(ask - _ROUNDING_TOLERANCE)
    return bid_ticks, max(ask_ticks, bid_ticks + 1)


def build_strategy(config: StrategyConfig) -> Strategy:
    """Create the strategy described by a strategy config."""
    match config:
        case NaiveConfig():
            return Naive(config)
        case InventorySkewConfig():
            return InventorySkew(config)
        case AvellanedaStoikovConfig():
            return AvellanedaStoikov(config)
        case _:
            assert_never(config)
