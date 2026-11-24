"""Simulation loop: a market maker with quote latency trading against takers."""

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mmsim.config import MakerConfig, SimulationConfig
from mmsim.market import (
    MARKET_MAKER_ID,
    BackgroundLiquidity,
    FloatArray,
    Takers,
    TakerType,
    simulate_value_path,
    taker_type,
)
from mmsim.orderbook import Order, OrderBook, Side, Trade
from mmsim.strategies import Quote, QuoteState, Strategy

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Fill:
    """Market maker side of a trade.

    Attributes
    ----------
    step : int
        Step of the fill.
    side : Side
        Market maker side; its value is the sign s of the position change.
    price : int
        Fill price in ticks.
    qty : int
        Filled quantity in lots.
    counterparty : TakerType
        Type of the taker that hit the quote.
    """

    step: int
    side: Side
    price: int
    qty: int
    counterparty: TakerType


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Recorded path of one run.

    Per-step arrays have length ``n_steps + 1``: index 0 is the initial state and index t
    the state at the end of step t. ``bid`` and ``ask`` hold the market maker quotes active
    during step t (NaN when a side is not quoted).
    """

    config: SimulationConfig
    seed: int
    values: FloatArray
    jumps: FloatArray
    reference: FloatArray
    market_mid: FloatArray
    background_mid: FloatArray
    bid: FloatArray
    ask: FloatArray
    position: IntArray
    cash: IntArray
    fills: tuple[Fill, ...]
    trades: tuple[Trade, ...]


class MarketMaker:
    """Order management of the market maker: latency, position limit and accounting.

    Quotes computed in step t are activated in step t + L. Activation replaces the resting
    order on a side only when the requested price changes, so an unchanged quote keeps its
    queue priority. A side is left unquoted when a fill there could breach the position
    limit.

    Parameters
    ----------
    config : MakerConfig
        Execution constraints.
    """

    def __init__(self, config: MakerConfig) -> None:
        self._config = config
        self._pending: deque[Quote] = deque()
        self._resting: dict[Side, tuple[Order, int]] = {}
        self.position = 0
        self.cash = 0

    def submit(self, book: OrderBook, quote: Quote) -> None:
        """Schedule a new quote and activate the one that is due, if any."""
        self._pending.append(quote)
        if len(self._pending) > self._config.latency:
            bid, ask = self._pending.popleft()
            qty, limit = self._config.quote_qty, self._config.position_limit
            self._requote(book, Side.BUY, bid if self.position + qty <= limit else None)
            self._requote(book, Side.SELL, ask if self.position - qty >= -limit else None)

    def resting_price(self, side: Side) -> float:
        """Return the price of the live quote on a side, or NaN if there is none."""
        resting = self._resting.get(side)
        if resting is None or resting[0].qty == 0:
            return float("nan")
        return float(resting[0].price)

    def record_fill(self, trade: Trade) -> Fill | None:
        """Update position and cash for a trade; return the fill if the maker took part."""
        if trade.buyer_id == MARKET_MAKER_ID:
            side, taker_id = Side.BUY, trade.seller_id
        elif trade.seller_id == MARKET_MAKER_ID:
            side, taker_id = Side.SELL, trade.buyer_id
        else:
            return None
        self.position += side * trade.qty
        self.cash -= side * trade.qty * trade.price
        return Fill(trade.step, side, trade.price, trade.qty, taker_type(taker_id))

    def _requote(self, book: OrderBook, side: Side, price: int | None) -> None:
        resting = self._resting.pop(side, None)
        if resting is not None:
            order, requested_price = resting
            if order.qty > 0 and requested_price == price:
                self._resting[side] = resting
                return
            book.cancel(order.order_id)
        if price is not None:
            order = book.submit_limit(MARKET_MAKER_ID, side, price, self._config.quote_qty)
            self._resting[side] = (order, price)


def quote_state(
    book: OrderBook, position: int, step: int, n_steps: int, sigma: float
) -> QuoteState:
    """Build the strategy input; the mid excludes the market maker's own orders."""
    return QuoteState(
        mid=book.mid(exclude_owner=MARKET_MAKER_ID),
        position=position,
        step=step,
        steps_remaining=n_steps - step,
        sigma=sigma,
    )


def simulate(config: SimulationConfig, strategy: Strategy, seed: int) -> SimulationResult:
    """Run one simulation.

    Each step: (1) the true value moves; (2) the strategy quotes from the book left by the
    previous step and the quote due this step is activated; (3) background liquidity acts;
    (4) a taker may arrive; (5) the state is recorded.

    Parameters
    ----------
    config : SimulationConfig
        Simulation parameters.
    strategy : Strategy
        Quoting strategy of the market maker.
    seed : int
        Seed of the market randomness.

    Returns
    -------
    SimulationResult
        Recorded path of the run.
    """
    value_seed, background_seed, taker_seed = np.random.SeedSequence(seed).spawn(3)
    n_steps = config.n_steps
    path = simulate_value_path(config.value, n_steps, np.random.default_rng(value_seed))
    background = BackgroundLiquidity(
        config.background, path.values, np.random.default_rng(background_seed)
    )
    takers = Takers(config.takers, n_steps, np.random.default_rng(taker_seed))
    book = OrderBook(initial_mid=config.value.initial)
    maker = MarketMaker(config.maker)
    values: list[float] = path.values.tolist()

    initial = config.value.initial
    market_mid, background_mid = [initial], [initial]
    bids, asks = [float("nan")], [float("nan")]
    positions, cash = [0], [0]
    fills: list[Fill] = []
    trades: list[Trade] = []

    for step in range(1, n_steps + 1):
        value = values[step]
        state = quote_state(book, maker.position, step, n_steps, config.value.sigma)
        maker.submit(book, strategy.quote(state))
        bids.append(maker.resting_price(Side.BUY))
        asks.append(maker.resting_price(Side.SELL))
        background.act(book, step)
        for trade in takers.act(book, step, value):
            trades.append(trade)
            fill = maker.record_fill(trade)
            if fill is not None:
                fills.append(fill)
        market_mid.append(book.mid())
        background_mid.append(book.mid(exclude_owner=MARKET_MAKER_ID))
        positions.append(maker.position)
        cash.append(maker.cash)

    return SimulationResult(
        config=config,
        seed=seed,
        values=path.values,
        jumps=path.jumps,
        reference=background.reference,
        market_mid=np.array(market_mid),
        background_mid=np.array(background_mid),
        bid=np.array(bids),
        ask=np.array(asks),
        position=np.array(positions, dtype=np.int64),
        cash=np.array(cash, dtype=np.int64),
        fills=tuple(fills),
        trades=tuple(trades),
    )
