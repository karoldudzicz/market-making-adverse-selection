"""Hidden true value and the non-strategic market participants.

All randomness of the market is drawn up front, so the value path, background order flow and
taker arrivals are identical across market maker strategies run on the same seed.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from mmsim.config import BackgroundConfig, TakerConfig, ValueConfig
from mmsim.orderbook import Order, OrderBook, Side, Trade

MARKET_MAKER_ID = 0
INFORMED_TAKER_ID = 1
NOISE_TAKER_ID = 2
FIRST_BACKGROUND_ID = 3

FloatArray = NDArray[np.float64]


class TakerType(Enum):
    """Type of the taker on the other side of a market maker fill."""

    NOISE = "noise"
    INFORMED = "informed"


@dataclass(frozen=True, slots=True)
class ValuePath:
    """Realised true value path.

    Attributes
    ----------
    values : ndarray
        ``values[0]`` is V_0 and ``values[t]`` is V_t after the move in step t.
    jumps : ndarray
        Jump component of the move in step t (zero at index 0 and in steps without a jump).
    """

    values: FloatArray
    jumps: FloatArray


def simulate_value_path(config: ValueConfig, n_steps: int, rng: np.random.Generator) -> ValuePath:
    """Draw a true value path of ``n_steps`` moves.

    Parameters
    ----------
    config : ValueConfig
        Value process parameters.
    n_steps : int
        Number of steps.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    ValuePath
        Arrays of length ``n_steps + 1``.
    """
    diffusion = rng.normal(0.0, config.sigma, n_steps)
    has_jump = rng.random(n_steps) < config.jump_prob
    jump_sign = np.where(rng.random(n_steps) < 0.5, -1.0, 1.0)
    jumps = np.concatenate(([0.0], np.where(has_jump, jump_sign * config.jump_size, 0.0)))
    increments = np.concatenate(([0.0], diffusion)) + jumps
    return ValuePath(values=config.initial + np.cumsum(increments), jumps=jumps)


class BackgroundLiquidity:
    """Liquidity providers quoting around a delayed, noisy view of the true value.

    In each step, each provider independently posts one post-only limit order per side with
    probability ``post_prob``, at a uniform random depth from the rounded reference
    R_t = V_{t-d} + eps_t. Orders are cancelled when they move more than ``max_distance``
    from R_t, or at the end of a geometric lifetime drawn at posting, which is equivalent to
    a per-step cancel probability ``cancel_prob`` but keeps the draws independent of the
    state of the book.

    Parameters
    ----------
    config : BackgroundConfig
        Background liquidity parameters.
    values : ndarray
        True value path, ``values[t]`` = V_t.
    rng : numpy.random.Generator
        Source of randomness.
    """

    def __init__(self, config: BackgroundConfig, values: FloatArray, rng: np.random.Generator):
        n_slots = len(values)
        delayed = values[np.maximum(np.arange(n_slots) - config.delay, 0)]
        self.reference: FloatArray = delayed + rng.normal(0.0, config.noise_sd, n_slots)
        shape = (n_slots, config.n_providers, 2)
        posts = rng.random(shape) < config.post_prob
        depths = rng.integers(config.min_depth, config.max_depth + 1, shape)
        lifetimes = rng.geometric(config.cancel_prob, shape)
        self._schedule: list[list[tuple[int, Side, int, int]]] = [[] for _ in range(n_slots)]
        steps, providers, side_index = np.nonzero(posts)
        for step, provider, side, depth, lifetime in zip(
            steps.tolist(),
            providers.tolist(),
            side_index.tolist(),
            depths[posts].tolist(),
            lifetimes[posts].tolist(),
            strict=True,
        ):
            self._schedule[step].append(
                (
                    FIRST_BACKGROUND_ID + provider,
                    Side.BUY if side == 0 else Side.SELL,
                    depth,
                    lifetime,
                )
            )
        self._max_distance = config.max_distance
        self._order_qty = config.order_qty
        self._live: list[tuple[Order, int]] = []

    def act(self, book: OrderBook, step: int) -> None:
        """Cancel expired and distant orders, then post this step's new orders."""
        reference = float(self.reference[step])
        self._cancel(book, step, reference)
        center = round(reference)
        for owner_id, side, depth, lifetime in self._schedule[step]:
            price = center - depth if side is Side.BUY else center + depth
            order = book.submit_limit(owner_id, side, price, self._order_qty)
            self._live.append((order, step + lifetime))

    def _cancel(self, book: OrderBook, step: int, reference: float) -> None:
        kept: list[tuple[Order, int]] = []
        for order, expiry in self._live:
            if order.qty == 0:
                continue
            if step >= expiry or abs(order.price - reference) > self._max_distance:
                book.cancel(order.order_id)
            else:
                kept.append((order, expiry))
        self._live = kept


class Takers:
    """Market-order flow from informed and noise takers.

    Each step a taker arrives with probability ``arrival_prob`` and is informed with
    probability ``informed_share``. The type draw compares a fixed uniform variate with
    ``informed_share``, so on a given seed raising pi turns a subset of noise takers into
    informed ones and leaves everything else unchanged.

    Parameters
    ----------
    config : TakerConfig
        Taker parameters.
    n_steps : int
        Number of simulation steps.
    rng : numpy.random.Generator
        Source of randomness.
    """

    def __init__(self, config: TakerConfig, n_steps: int, rng: np.random.Generator):
        n_slots = n_steps + 1
        self._arrives: list[bool] = (rng.random(n_slots) < config.arrival_prob).tolist()
        self._informed: list[bool] = (rng.random(n_slots) < config.informed_share).tolist()
        self._noise_buys: list[bool] = (rng.random(n_slots) < 0.5).tolist()
        self._threshold = config.informed_threshold
        self._qty = config.order_qty

    def act(self, book: OrderBook, step: int, value: float) -> list[Trade]:
        """Send this step's market order, if any, and return the resulting trades."""
        if not self._arrives[step]:
            return []
        if self._informed[step]:
            return self._trade_informed(book, step, value)
        side = Side.BUY if self._noise_buys[step] else Side.SELL
        return book.submit_market(NOISE_TAKER_ID, side, self._qty, step)

    def _trade_informed(self, book: OrderBook, step: int, value: float) -> list[Trade]:
        best_ask = book.best_ask()
        if best_ask is not None and best_ask < value - self._threshold:
            return book.submit_market(INFORMED_TAKER_ID, Side.BUY, self._qty, step)
        best_bid = book.best_bid()
        if best_bid is not None and best_bid > value + self._threshold:
            return book.submit_market(INFORMED_TAKER_ID, Side.SELL, self._qty, step)
        return []


def taker_type(owner_id: int) -> TakerType:
    """Map a taker owner id to its type."""
    if owner_id == INFORMED_TAKER_ID:
        return TakerType.INFORMED
    if owner_id == NOISE_TAKER_ID:
        return TakerType.NOISE
    raise ValueError(f"owner {owner_id} is not a taker")
