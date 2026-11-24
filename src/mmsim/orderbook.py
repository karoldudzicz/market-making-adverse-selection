"""Price-time priority limit order book with integer tick prices."""

import bisect
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum


class Side(IntEnum):
    """Order side; the value is the sign of the owner's position change."""

    BUY = 1
    SELL = -1


@dataclass(slots=True, eq=False)
class Order:
    """Resting limit order.

    The book updates ``qty`` in place; it drops to zero once the order is fully filled or
    cancelled, after which the order is no longer in the book.

    Attributes
    ----------
    order_id : int
        Unique identifier assigned by the book.
    owner_id : int
        Participant that submitted the order.
    side : Side
        Order side.
    price : int
        Resting price in ticks, after the post-only adjustment.
    qty : int
        Remaining quantity in lots.
    """

    order_id: int
    owner_id: int
    side: Side
    price: int
    qty: int


@dataclass(frozen=True, slots=True)
class Trade:
    """Execution of an incoming market order against a resting limit order.

    Attributes
    ----------
    step : int
        Simulation step of the execution.
    price : int
        Execution price in ticks (the resting order's price).
    qty : int
        Executed quantity in lots.
    buyer_id : int
        Owner of the buying side.
    seller_id : int
        Owner of the selling side.
    aggressor : Side
        Side of the market order that initiated the trade.
    """

    step: int
    price: int
    qty: int
    buyer_id: int
    seller_id: int
    aggressor: Side


@dataclass(slots=True, eq=False)
class _Level:
    orders: deque[Order] = field(default_factory=deque)
    qty: int = 0


class _BookSide:
    """Price levels of one side, with prices kept sorted ascending for best-price lookup."""

    __slots__ = ("_best_index", "_levels", "_prices")

    def __init__(self, side: Side) -> None:
        self._levels: dict[int, _Level] = {}
        self._prices: list[int] = []
        self._best_index = -1 if side is Side.BUY else 0

    def best_price(self) -> int | None:
        return self._prices[self._best_index] if self._prices else None

    def best_price_excluding(self, owner_id: int) -> int | None:
        for price in self.prices_from_best():
            if any(order.owner_id != owner_id for order in self._levels[price].orders):
                return price
        return None

    def prices_from_best(self) -> list[int]:
        return self._prices[::-1] if self._best_index == -1 else list(self._prices)

    def level_qty(self, price: int) -> int:
        return self._levels[price].qty

    def level_orders(self, price: int) -> list[Order]:
        return list(self._levels[price].orders)

    def add(self, order: Order) -> None:
        level = self._levels.get(order.price)
        if level is None:
            level = _Level()
            self._levels[order.price] = level
            bisect.insort(self._prices, order.price)
        level.orders.append(order)
        level.qty += order.qty

    def remove(self, order: Order) -> None:
        level = self._levels[order.price]
        level.orders.remove(order)
        level.qty -= order.qty
        if not level.orders:
            self._drop_level(order.price)

    def match(self, qty: int) -> list[tuple[Order, int]]:
        """Fill up to ``qty`` lots against resting orders; return (order, filled qty) pairs."""
        fills: list[tuple[Order, int]] = []
        while qty > 0 and self._prices:
            price = self._prices[self._best_index]
            level = self._levels[price]
            order = level.orders[0]
            fill = min(qty, order.qty)
            order.qty -= fill
            level.qty -= fill
            qty -= fill
            fills.append((order, fill))
            if order.qty == 0:
                level.orders.popleft()
                if not level.orders:
                    self._drop_level(price)
        return fills

    def _drop_level(self, price: int) -> None:
        del self._levels[price]
        del self._prices[bisect.bisect_left(self._prices, price)]


class OrderBook:
    """Limit order book with price-time priority and integer tick prices.

    Limit orders are post-only and market orders are immediate-or-cancel: a market order
    sweeps the opposite side from the best price, and any quantity left once that side is
    empty is discarded.

    Parameters
    ----------
    initial_mid : float
        Mid price reported before both sides of the book have been observed.
    """

    def __init__(self, initial_mid: float) -> None:
        self._bids = _BookSide(Side.BUY)
        self._asks = _BookSide(Side.SELL)
        self._orders: dict[int, Order] = {}
        self._next_order_id = 0
        self._initial_mid = initial_mid
        self._last_mid: dict[int | None, float] = {}

    def submit_limit(self, owner_id: int, side: Side, price: int, qty: int) -> Order:
        """Place a post-only limit order.

        A price that would cross the book is moved to one tick behind the best opposite
        price, so the order always rests and never trades on arrival.

        Parameters
        ----------
        owner_id : int
            Submitting participant.
        side : Side
            Order side.
        price : int
            Requested limit price in ticks.
        qty : int
            Quantity in lots, positive.

        Returns
        -------
        Order
            The resting order; its ``price`` reflects any post-only adjustment.
        """
        if not isinstance(price, int):
            raise TypeError(f"price must be an int number of ticks, got {type(price).__name__}")
        _check_qty(qty)
        order = Order(self._next_order_id, owner_id, side, self._post_only_price(side, price), qty)
        self._next_order_id += 1
        self._orders[order.order_id] = order
        self._side(side).add(order)
        return order

    def submit_market(self, owner_id: int, side: Side, qty: int, step: int) -> list[Trade]:
        """Execute a market order against the opposite side of the book.

        Parameters
        ----------
        owner_id : int
            Submitting participant.
        side : Side
            Side of the market order.
        qty : int
            Quantity in lots, positive.
        step : int
            Simulation step recorded on the resulting trades.

        Returns
        -------
        list[Trade]
            One trade per resting order hit, in execution order; empty if the opposite
            side is empty.
        """
        _check_qty(qty)
        opposite = self._asks if side is Side.BUY else self._bids
        trades: list[Trade] = []
        for maker, filled in opposite.match(qty):
            if maker.qty == 0:
                del self._orders[maker.order_id]
            if side is Side.BUY:
                buyer_id, seller_id = owner_id, maker.owner_id
            else:
                buyer_id, seller_id = maker.owner_id, owner_id
            trades.append(Trade(step, maker.price, filled, buyer_id, seller_id, side))
        return trades

    def cancel(self, order_id: int) -> int:
        """Cancel a resting order.

        Parameters
        ----------
        order_id : int
            Identifier returned by :meth:`submit_limit`.

        Returns
        -------
        int
            Cancelled quantity; zero if the order is no longer resting.
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return 0
        self._side(order.side).remove(order)
        cancelled = order.qty
        order.qty = 0
        return cancelled

    def best_bid(self, exclude_owner: int | None = None) -> int | None:
        """Return the best bid price, optionally ignoring the orders of one owner."""
        if exclude_owner is None:
            return self._bids.best_price()
        return self._bids.best_price_excluding(exclude_owner)

    def best_ask(self, exclude_owner: int | None = None) -> int | None:
        """Return the best ask price, optionally ignoring the orders of one owner."""
        if exclude_owner is None:
            return self._asks.best_price()
        return self._asks.best_price_excluding(exclude_owner)

    def mid(self, exclude_owner: int | None = None) -> float:
        """Return the mid price, optionally ignoring the orders of one owner.

        When a side is empty, returns the last two-sided mid computed for the same
        ``exclude_owner``, or the initial mid if there is none.
        """
        bid = self.best_bid(exclude_owner)
        ask = self.best_ask(exclude_owner)
        if bid is None or ask is None:
            return self._last_mid.get(exclude_owner, self._initial_mid)
        mid = (bid + ask) / 2
        self._last_mid[exclude_owner] = mid
        return mid

    def depth(self, side: Side) -> list[tuple[int, int]]:
        """Return aggregated (price, quantity) levels of one side, best price first."""
        book_side = self._side(side)
        return [(price, book_side.level_qty(price)) for price in book_side.prices_from_best()]

    def orders(self, side: Side) -> list[Order]:
        """Return the resting orders of one side in priority order."""
        book_side = self._side(side)
        return [
            order
            for price in book_side.prices_from_best()
            for order in book_side.level_orders(price)
        ]

    def _side(self, side: Side) -> _BookSide:
        return self._bids if side is Side.BUY else self._asks

    def _post_only_price(self, side: Side, price: int) -> int:
        # Sliding instead of rejecting keeps takers as the only aggressors, so every
        # maker fill has an unambiguous taker counterparty.
        if side is Side.BUY:
            best_ask = self._asks.best_price()
            if best_ask is not None and price >= best_ask:
                return best_ask - 1
        else:
            best_bid = self._bids.best_price()
            if best_bid is not None and price <= best_bid:
                return best_bid + 1
        return price


def _check_qty(qty: int) -> None:
    if qty <= 0:
        raise ValueError(f"quantity must be positive, got {qty}")
