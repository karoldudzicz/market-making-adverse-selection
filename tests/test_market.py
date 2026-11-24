import numpy as np
import pytest

from mmsim.config import BackgroundConfig, TakerConfig, ValueConfig
from mmsim.market import (
    FIRST_BACKGROUND_ID,
    INFORMED_TAKER_ID,
    NOISE_TAKER_ID,
    BackgroundLiquidity,
    Takers,
    TakerType,
    simulate_value_path,
    taker_type,
)
from mmsim.orderbook import OrderBook, Side


def test_value_path_starts_at_initial_value_and_jumps_are_fixed_size() -> None:
    config = ValueConfig(initial=500.0, jump_prob=0.05, jump_size=7.0)

    path = simulate_value_path(config, 1_000, np.random.default_rng(0))

    assert path.values.shape == path.jumps.shape == (1_001,)
    assert path.values[0] == 500.0
    assert path.jumps[0] == 0.0
    assert set(np.unique(path.jumps)) <= {-7.0, 0.0, 7.0}


def test_value_path_increment_statistics_match_config() -> None:
    config = ValueConfig(sigma=0.5, jump_prob=0.01, jump_size=10.0)

    path = simulate_value_path(config, 200_000, np.random.default_rng(1))
    diffusion = np.diff(path.values) - path.jumps[1:]

    assert diffusion.std() == pytest.approx(0.5, rel=0.01)
    assert np.mean(path.jumps != 0) == pytest.approx(0.01, rel=0.1)


def test_reference_lags_true_value_by_delay() -> None:
    values = np.arange(20, dtype=np.float64) + 100.0
    config = BackgroundConfig(delay=3, noise_sd=0.0)

    background = BackgroundLiquidity(config, values, np.random.default_rng(0))

    assert background.reference[10] == values[7]
    assert background.reference[2] == values[0]


def test_background_posts_at_configured_depths_around_reference() -> None:
    values = np.full(5, 10_000.0)
    config = BackgroundConfig(n_providers=3, post_prob=1.0, noise_sd=0.0, min_depth=2, max_depth=4)
    background = BackgroundLiquidity(config, values, np.random.default_rng(0))
    book = OrderBook(initial_mid=10_000.0)

    background.act(book, step=1)

    bids, asks = book.orders(Side.BUY), book.orders(Side.SELL)
    assert len(bids) == len(asks) == 3
    assert all(9_996 <= o.price <= 9_998 for o in bids)
    assert all(10_002 <= o.price <= 10_004 for o in asks)
    assert {o.owner_id for o in bids} == set(range(FIRST_BACKGROUND_ID, FIRST_BACKGROUND_ID + 3))


def test_background_cancels_orders_far_from_reference() -> None:
    values = np.array([10_000.0] * 5 + [10_100.0] * 5)
    config = BackgroundConfig(
        post_prob=1.0, noise_sd=0.0, delay=0, cancel_prob=1e-9, cancel_through_reference=False
    )
    background = BackgroundLiquidity(config, values, np.random.default_rng(0))
    book = OrderBook(initial_mid=10_000.0)

    for step in range(1, 7):
        background.act(book, step)

    prices = [o.price for side in Side for o in book.orders(side)]
    assert prices
    assert all(abs(price - 10_100) <= 15 for price in prices)


@pytest.mark.parametrize("enabled", [True, False])
def test_orders_left_on_wrong_side_by_a_jump_are_cancelled_when_reference_passes_them(
    enabled: bool,
) -> None:
    values = np.array([10_000.0] * 10 + [9_990.0] * 10)
    config = BackgroundConfig(
        post_prob=1.0,
        noise_sd=0.0,
        delay=3,
        cancel_prob=1e-9,
        max_distance=100.0,
        cancel_through_reference=enabled,
    )
    background = BackgroundLiquidity(config, values, np.random.default_rng(0))
    book = OrderBook(initial_mid=10_000.0)
    passing_step = 13
    assert background.reference[passing_step - 1] == 10_000.0
    assert background.reference[passing_step] == 9_990.0

    for step in range(1, passing_step):
        background.act(book, step)
    old_bids, old_asks = book.orders(Side.BUY), book.orders(Side.SELL)
    background.act(book, passing_step)

    assert old_bids
    assert all(o.price >= 9_990 for o in old_bids)
    assert all((o.qty == 0) is enabled for o in old_bids)
    assert all(o.qty > 0 for o in old_asks)


def test_background_orders_expire_after_their_lifetime() -> None:
    values = np.full(10, 10_000.0)
    config = BackgroundConfig(n_providers=2, post_prob=1.0, noise_sd=0.0, cancel_prob=1.0)
    background = BackgroundLiquidity(config, values, np.random.default_rng(0))
    book = OrderBook(initial_mid=10_000.0)

    for step in range(1, 6):
        background.act(book, step)

    assert len(book.orders(Side.BUY)) == len(book.orders(Side.SELL)) == 2


def _book_with_quotes(bid: int, ask: int) -> OrderBook:
    book = OrderBook(initial_mid=(bid + ask) / 2)
    book.submit_limit(FIRST_BACKGROUND_ID, Side.BUY, bid, 1)
    book.submit_limit(FIRST_BACKGROUND_ID, Side.SELL, ask, 1)
    return book


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (102.0, Side.BUY),
        (101.0, None),
        (97.0, None),
        (93.5, Side.SELL),
    ],
)
def test_informed_taker_trades_only_beyond_threshold(value: float, expected: Side | None) -> None:
    takers = Takers(
        TakerConfig(arrival_prob=1.0, informed_share=1.0, informed_threshold=1.0),
        n_steps=1,
        rng=np.random.default_rng(0),
    )
    book = _book_with_quotes(bid=95, ask=100)

    trades = takers.act(book, step=1, value=value)

    if expected is None:
        assert trades == []
    else:
        (trade,) = trades
        assert trade.aggressor is expected
        assert INFORMED_TAKER_ID in (trade.buyer_id, trade.seller_id)


def test_noise_taker_trades_on_every_arrival() -> None:
    takers = Takers(TakerConfig(arrival_prob=1.0, informed_share=0.0), 50, np.random.default_rng(0))

    sides = []
    for step in range(1, 51):
        (trade,) = takers.act(_book_with_quotes(95, 100), step, value=97.5)
        assert NOISE_TAKER_ID in (trade.buyer_id, trade.seller_id)
        sides.append(trade.aggressor)

    assert set(sides) == {Side.BUY, Side.SELL}


def test_raising_informed_share_only_converts_noise_takers() -> None:
    def informed_and_arrival_steps(share: float) -> tuple[set[int], set[int]]:
        takers = Takers(
            TakerConfig(arrival_prob=0.5, informed_share=share), 500, np.random.default_rng(3)
        )
        informed, arrived = set(), set()
        for step in range(1, 501):
            trades = takers.act(_book_with_quotes(95, 100), step, value=1_000.0)
            if trades:
                arrived.add(step)
            if any(INFORMED_TAKER_ID in (t.buyer_id, t.seller_id) for t in trades):
                informed.add(step)
        return informed, arrived

    low_informed, low_arrived = informed_and_arrival_steps(0.2)
    high_informed, high_arrived = informed_and_arrival_steps(0.5)

    assert low_arrived == high_arrived
    assert low_informed < high_informed


def test_taker_type_maps_taker_ids_and_rejects_others() -> None:
    assert taker_type(INFORMED_TAKER_ID) is TakerType.INFORMED
    assert taker_type(NOISE_TAKER_ID) is TakerType.NOISE
    with pytest.raises(ValueError, match="not a taker"):
        taker_type(FIRST_BACKGROUND_ID)
