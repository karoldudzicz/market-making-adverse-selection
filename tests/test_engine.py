from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from mmsim.config import MakerConfig, NaiveConfig, SimulationConfig, TakerConfig
from mmsim.engine import MarketMaker, SimulationResult, quote_state, simulate
from mmsim.market import FIRST_BACKGROUND_ID, INFORMED_TAKER_ID, MARKET_MAKER_ID, NOISE_TAKER_ID
from mmsim.orderbook import OrderBook, Side
from mmsim.strategies import Naive
from quoters import AggressiveBid, NoQuotes, StepEncodingBid

SHORT = SimulationConfig(n_steps=1_000)
NAIVE = Naive(NaiveConfig(half_spread=1.0))


def with_latency(config: SimulationConfig, latency: int) -> SimulationConfig:
    return replace(config, maker=replace(config.maker, latency=latency))


def assert_same_run(a: SimulationResult, b: SimulationResult) -> None:
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.market_mid, b.market_mid)
    np.testing.assert_array_equal(a.position, b.position)
    np.testing.assert_array_equal(a.cash, b.cash)
    assert a.fills == b.fills
    assert a.trades == b.trades


def test_same_seed_reproduces_run() -> None:
    assert_same_run(simulate(SHORT, NAIVE, seed=11), simulate(SHORT, NAIVE, seed=11))


def test_different_seeds_give_different_runs() -> None:
    a = simulate(SHORT, NAIVE, seed=11)
    b = simulate(SHORT, NAIVE, seed=12)

    assert not np.array_equal(a.values, b.values)
    assert a.trades != b.trades


def test_market_randomness_does_not_depend_on_strategy() -> None:
    a = simulate(SHORT, NAIVE, seed=5)
    b = simulate(SHORT, NoQuotes(), seed=5)

    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.reference, b.reference)


@pytest.mark.parametrize("latency", [0, 3])
def test_accounting_is_consistent_across_participants(latency: int) -> None:
    result = simulate(with_latency(SHORT, latency), NAIVE, seed=2)

    positions: defaultdict[int, int] = defaultdict(int)
    cash: defaultdict[int, int] = defaultdict(int)
    for trade in result.trades:
        positions[trade.buyer_id] += trade.qty
        positions[trade.seller_id] -= trade.qty
        cash[trade.buyer_id] -= trade.qty * trade.price
        cash[trade.seller_id] += trade.qty * trade.price
        takers = {INFORMED_TAKER_ID, NOISE_TAKER_ID}
        assert (trade.buyer_id if trade.aggressor is Side.BUY else trade.seller_id) in takers

    assert sum(positions.values()) == 0
    assert sum(cash.values()) == 0
    assert positions[MARKET_MAKER_ID] == result.position[-1]
    assert cash[MARKET_MAKER_ID] == result.cash[-1]
    assert len(result.fills) == sum(
        MARKET_MAKER_ID in (t.buyer_id, t.seller_id) for t in result.trades
    )


def test_position_path_matches_fills() -> None:
    result = simulate(SHORT, NAIVE, seed=4)

    changes = np.zeros(SHORT.n_steps + 1, dtype=np.int64)
    for fill in result.fills:
        changes[fill.step] += fill.side * fill.qty

    np.testing.assert_array_equal(np.cumsum(changes), result.position)


@pytest.mark.parametrize("latency", [0, 1, 4])
def test_quotes_activate_exactly_latency_steps_after_computation(latency: int) -> None:
    config = with_latency(SimulationConfig(n_steps=30), latency)

    result = simulate(config, StepEncodingBid(), seed=0)

    assert np.isnan(result.bid[1 : latency + 1]).all()
    steps = np.arange(latency + 1, 31)
    np.testing.assert_array_equal(result.bid[steps], StepEncodingBid.BASE + steps - latency)
    assert np.isnan(result.ask).all()


@pytest.mark.parametrize("latency", [0, 3])
def test_position_limit_is_never_breached(latency: int) -> None:
    config = replace(
        SHORT,
        takers=TakerConfig(arrival_prob=0.5, informed_share=0.0),
        maker=MakerConfig(latency=latency, position_limit=5),
    )

    result = simulate(config, AggressiveBid(), seed=0)

    assert result.position.max() == 5
    assert result.position.min() >= 0


def test_unchanged_quote_keeps_its_order_and_queue_priority() -> None:
    book = OrderBook(initial_mid=100.0)
    maker = MarketMaker(MakerConfig())

    maker.submit(book, (99, 101))
    first_orders = book.orders(Side.BUY) + book.orders(Side.SELL)
    maker.submit(book, (99, 102))

    assert book.orders(Side.BUY) == first_orders[:1]
    assert book.orders(Side.SELL) != first_orders[1:]
    assert book.best_ask() == 102


def test_quote_state_mid_ignores_market_maker_orders() -> None:
    book = OrderBook(initial_mid=100.0)
    book.submit_limit(FIRST_BACKGROUND_ID, Side.BUY, 96, 1)
    book.submit_limit(FIRST_BACKGROUND_ID, Side.SELL, 104, 1)
    before = quote_state(book, position=0, step=1, n_steps=10, sigma=0.5)

    book.submit_limit(MARKET_MAKER_ID, Side.BUY, 101, 1)
    book.submit_limit(MARKET_MAKER_ID, Side.SELL, 103, 1)
    after = quote_state(book, position=0, step=1, n_steps=10, sigma=0.5)

    assert before.mid == after.mid == 100.0
    assert after.steps_remaining == 9


def test_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency"):
        with_latency(SHORT, -1)
