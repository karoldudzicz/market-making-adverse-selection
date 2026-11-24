import pytest

from mmsim.orderbook import OrderBook, Side, Trade

MM = 0
BG = 1
TAKER = 9


@pytest.fixture
def book() -> OrderBook:
    return OrderBook(initial_mid=100.0)


def test_empty_book_has_no_prices_and_reports_initial_mid(book: OrderBook) -> None:
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert book.mid() == 100.0
    assert book.depth(Side.BUY) == []
    assert book.depth(Side.SELL) == []


def test_market_order_on_empty_side_does_not_trade(book: OrderBook) -> None:
    book.submit_limit(BG, Side.BUY, 99, 1)

    assert book.submit_market(TAKER, Side.BUY, 3, step=0) == []
    assert book.depth(Side.BUY) == [(99, 1)]


def test_better_price_has_priority(book: OrderBook) -> None:
    book.submit_limit(1, Side.SELL, 102, 1)
    book.submit_limit(2, Side.SELL, 101, 1)

    (trade,) = book.submit_market(TAKER, Side.BUY, 1, step=0)

    assert (trade.price, trade.seller_id) == (101, 2)


def test_fifo_within_price_level(book: OrderBook) -> None:
    book.submit_limit(1, Side.BUY, 99, 1)
    book.submit_limit(2, Side.BUY, 99, 1)
    book.submit_limit(3, Side.BUY, 99, 1)

    trades = book.submit_market(TAKER, Side.SELL, 2, step=0)

    assert [t.buyer_id for t in trades] == [1, 2]
    assert [o.owner_id for o in book.orders(Side.BUY)] == [3]


def test_partial_fill_keeps_remainder_at_front_of_queue(book: OrderBook) -> None:
    first = book.submit_limit(1, Side.SELL, 101, 5)
    book.submit_limit(2, Side.SELL, 101, 1)

    (trade,) = book.submit_market(TAKER, Side.BUY, 2, step=0)

    assert (trade.qty, trade.seller_id) == (2, 1)
    assert first.qty == 3
    assert book.depth(Side.SELL) == [(101, 4)]
    assert [o.owner_id for o in book.orders(Side.SELL)] == [1, 2]


def test_market_order_sweeps_multiple_levels(book: OrderBook) -> None:
    book.submit_limit(1, Side.SELL, 101, 1)
    book.submit_limit(2, Side.SELL, 102, 2)
    book.submit_limit(3, Side.SELL, 103, 5)

    trades = book.submit_market(TAKER, Side.BUY, 4, step=0)

    assert [(t.price, t.qty) for t in trades] == [(101, 1), (102, 2), (103, 1)]
    assert book.depth(Side.SELL) == [(103, 4)]


def test_market_order_larger_than_book_discards_remainder(book: OrderBook) -> None:
    book.submit_limit(1, Side.BUY, 99, 2)
    book.submit_limit(2, Side.BUY, 98, 1)

    trades = book.submit_market(TAKER, Side.SELL, 10, step=0)

    assert sum(t.qty for t in trades) == 3
    assert book.depth(Side.BUY) == []


def test_trade_records_step_counterparties_and_aggressor(book: OrderBook) -> None:
    book.submit_limit(MM, Side.BUY, 99, 1)
    book.submit_limit(MM, Side.SELL, 101, 1)

    sell_trades = book.submit_market(TAKER, Side.SELL, 1, step=7)
    buy_trades = book.submit_market(TAKER, Side.BUY, 1, step=8)

    assert sell_trades == [Trade(7, 99, 1, buyer_id=MM, seller_id=TAKER, aggressor=Side.SELL)]
    assert buy_trades == [Trade(8, 101, 1, buyer_id=TAKER, seller_id=MM, aggressor=Side.BUY)]


def test_cancel_removes_order(book: OrderBook) -> None:
    order = book.submit_limit(1, Side.BUY, 99, 4)

    assert book.cancel(order.order_id) == 4
    assert order.qty == 0
    assert book.depth(Side.BUY) == []
    assert book.cancel(order.order_id) == 0


def test_cancel_partially_filled_order_returns_remaining_qty(book: OrderBook) -> None:
    order = book.submit_limit(1, Side.SELL, 101, 4)
    book.submit_market(TAKER, Side.BUY, 1, step=0)

    assert book.cancel(order.order_id) == 3


def test_cancel_of_filled_order_is_a_no_op(book: OrderBook) -> None:
    order = book.submit_limit(1, Side.SELL, 101, 1)
    book.submit_market(TAKER, Side.BUY, 1, step=0)

    assert order.qty == 0
    assert book.cancel(order.order_id) == 0


def test_cancel_keeps_other_orders_at_same_price(book: OrderBook) -> None:
    first = book.submit_limit(1, Side.BUY, 99, 1)
    book.submit_limit(2, Side.BUY, 99, 2)

    book.cancel(first.order_id)

    assert book.depth(Side.BUY) == [(99, 2)]


def test_post_only_buy_crossing_the_ask_slides_one_tick_below_it(book: OrderBook) -> None:
    book.submit_limit(BG, Side.SELL, 101, 1)

    order = book.submit_limit(MM, Side.BUY, 103, 1)

    assert order.price == 100
    assert book.best_ask() == 101


def test_post_only_sell_crossing_the_bid_slides_one_tick_above_it(book: OrderBook) -> None:
    book.submit_limit(BG, Side.BUY, 99, 1)

    order = book.submit_limit(MM, Side.SELL, 99, 1)

    assert order.price == 100
    assert book.best_bid() == 99


def test_post_only_keeps_non_crossing_price(book: OrderBook) -> None:
    book.submit_limit(BG, Side.SELL, 105, 1)

    assert book.submit_limit(MM, Side.BUY, 104, 1).price == 104


def test_best_prices_can_exclude_an_owner(book: OrderBook) -> None:
    book.submit_limit(MM, Side.BUY, 100, 1)
    book.submit_limit(BG, Side.BUY, 98, 1)
    book.submit_limit(MM, Side.SELL, 101, 1)
    book.submit_limit(BG, Side.SELL, 103, 1)

    assert (book.best_bid(), book.best_ask()) == (100, 101)
    assert (book.best_bid(exclude_owner=MM), book.best_ask(exclude_owner=MM)) == (98, 103)
    assert book.mid(exclude_owner=MM) == 100.5


def test_best_price_excluding_owner_uses_other_orders_on_shared_level(book: OrderBook) -> None:
    book.submit_limit(MM, Side.BUY, 100, 1)
    book.submit_limit(BG, Side.BUY, 100, 1)

    assert book.best_bid(exclude_owner=MM) == 100


def test_best_price_excluding_sole_owner_is_none(book: OrderBook) -> None:
    book.submit_limit(MM, Side.SELL, 101, 1)

    assert book.best_ask(exclude_owner=MM) is None


def test_mid_falls_back_to_last_two_sided_mid(book: OrderBook) -> None:
    book.submit_limit(BG, Side.BUY, 99, 1)
    book.submit_limit(BG, Side.SELL, 102, 1)
    assert book.mid() == 100.5

    book.submit_market(TAKER, Side.BUY, 1, step=0)

    assert book.best_ask() is None
    assert book.mid() == 100.5


@pytest.mark.parametrize("qty", [0, -1])
def test_rejects_non_positive_quantity(book: OrderBook, qty: int) -> None:
    with pytest.raises(ValueError, match="quantity"):
        book.submit_limit(BG, Side.BUY, 99, qty)
    with pytest.raises(ValueError, match="quantity"):
        book.submit_market(TAKER, Side.BUY, qty, step=0)


def test_rejects_non_integer_price(book: OrderBook) -> None:
    with pytest.raises(TypeError, match="price"):
        book.submit_limit(BG, Side.BUY, 99.5, 1)  # type: ignore[arg-type]
