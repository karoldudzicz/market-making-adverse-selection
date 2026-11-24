from collections import defaultdict

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from mmsim.orderbook import OrderBook, Side

owners = st.integers(min_value=0, max_value=3)
sides = st.sampled_from(Side)
prices = st.integers(min_value=95, max_value=105)
quantities = st.integers(min_value=1, max_value=5)


class OrderBookMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.book = OrderBook(initial_mid=100.0)
        self.order_ids: list[int] = []
        self.limit_submitted = 0
        self.market_submitted = 0
        self.market_unfilled = 0
        self.filled = 0
        self.cancelled = 0
        self.step = 0

    @rule(owner=owners, side=sides, price=prices, qty=quantities)
    def submit_limit(self, owner: int, side: Side, price: int, qty: int) -> None:
        order = self.book.submit_limit(owner, side, price, qty)
        self.order_ids.append(order.order_id)
        self.limit_submitted += qty

    @rule(owner=owners, side=sides, qty=st.integers(min_value=1, max_value=12))
    def submit_market(self, owner: int, side: Side, qty: int) -> None:
        trades = self.book.submit_market(owner, side, qty, self.step)
        self.step += 1
        filled = sum(t.qty for t in trades)
        trade_prices = [t.price for t in trades]
        assert trade_prices == sorted(trade_prices, reverse=side is Side.SELL)
        assert all(t.aggressor is side for t in trades)
        self.market_submitted += qty
        self.market_unfilled += qty - filled
        self.filled += filled

    @precondition(lambda self: bool(self.order_ids))
    @rule(data=st.data())
    def cancel(self, data: st.DataObject) -> None:
        order_id = data.draw(st.sampled_from(self.order_ids))
        self.cancelled += self.book.cancel(order_id)

    @invariant()
    def book_is_not_crossed(self) -> None:
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if bid is not None and ask is not None:
            assert bid < ask

    @invariant()
    def level_quantities_match_resting_orders(self) -> None:
        for side in Side:
            from_orders: defaultdict[int, int] = defaultdict(int)
            for order in self.book.orders(side):
                assert order.qty > 0
                assert order.side is side
                from_orders[order.price] += order.qty
            depth = self.book.depth(side)
            assert dict(depth) == from_orders
            level_prices = [price for price, _ in depth]
            assert level_prices == sorted(from_orders, reverse=side is Side.BUY)

    @invariant()
    def levels_are_fifo(self) -> None:
        for side in Side:
            ids_by_price: defaultdict[int, list[int]] = defaultdict(list)
            for order in self.book.orders(side):
                ids_by_price[order.price].append(order.order_id)
            assert all(ids == sorted(ids) for ids in ids_by_price.values())

    @invariant()
    def quantity_is_conserved(self) -> None:
        resting = sum(qty for side in Side for _, qty in self.book.depth(side))
        assert self.limit_submitted == self.filled + resting + self.cancelled
        assert self.market_submitted == self.filled + self.market_unfilled

    @invariant()
    def best_prices_excluding_owner_match_brute_force(self) -> None:
        for owner in range(4):
            bids = [o.price for o in self.book.orders(Side.BUY) if o.owner_id != owner]
            asks = [o.price for o in self.book.orders(Side.SELL) if o.owner_id != owner]
            assert self.book.best_bid(exclude_owner=owner) == max(bids, default=None)
            assert self.book.best_ask(exclude_owner=owner) == min(asks, default=None)


OrderBookMachine.TestCase.settings = settings(
    max_examples=200, stateful_step_count=50, deadline=None
)
TestOrderBook = OrderBookMachine.TestCase
