import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mmsim.config import (
    AvellanedaStoikovConfig,
    InventorySkewConfig,
    MakerConfig,
    NaiveConfig,
    SimulationConfig,
    StrategyConfig,
    TakerConfig,
)
from mmsim.engine import simulate
from mmsim.market import MARKET_MAKER_ID
from mmsim.orderbook import Side
from mmsim.strategies import (
    AvellanedaStoikov,
    InventorySkew,
    Naive,
    Quote,
    QuoteState,
    Strategy,
    build_strategy,
    to_ticks,
)

strategy_configs = st.one_of(
    st.builds(NaiveConfig, half_spread=st.floats(0.0, 10.0)),
    st.builds(InventorySkewConfig, half_spread=st.floats(0.0, 10.0), skew=st.floats(0.0, 2.0)),
    st.builds(
        AvellanedaStoikovConfig,
        gamma=st.floats(1e-3, 1.0),
        horizon=st.one_of(st.none(), st.floats(1.0, 1_000.0)),
        k=st.floats(0.1, 10.0),
    ),
)
states = st.builds(
    QuoteState,
    mid=st.integers(19_000, 21_000).map(lambda half_ticks: half_ticks / 2),
    position=st.integers(-20, 20),
    step=st.integers(1, 5_000),
    steps_remaining=st.integers(0, 5_000),
    sigma=st.floats(0.1, 2.0),
)


def state(mid: float = 10_000.0, position: int = 0, steps_remaining: int = 1_000) -> QuoteState:
    return QuoteState(mid, position, step=1, steps_remaining=steps_remaining, sigma=0.5)


@given(config=strategy_configs, state=states)
def test_quotes_are_whole_ticks_with_bid_below_ask(
    config: StrategyConfig, state: QuoteState
) -> None:
    bid, ask = build_strategy(config).quote(state)

    assert isinstance(bid, int)
    assert isinstance(ask, int)
    assert bid < ask


@given(config=strategy_configs, state=states, other_position=st.integers(-20, 20))
def test_quotes_do_not_rise_with_position(
    config: StrategyConfig, state: QuoteState, other_position: int
) -> None:
    strategy = build_strategy(config)
    low, high = sorted((state.position, other_position))

    low_bid, low_ask = strategy.quote(replace(state, position=low))
    high_bid, high_ask = strategy.quote(replace(state, position=high))

    assert high_bid <= low_bid
    assert high_ask <= low_ask


@pytest.mark.parametrize(
    "config", [InventorySkewConfig(half_spread=1.0, skew=0.2), AvellanedaStoikovConfig()]
)
def test_long_position_moves_both_quotes_down(config: StrategyConfig) -> None:
    strategy = build_strategy(config)

    flat_bid, flat_ask = strategy.quote(state(position=0))
    long_bid, long_ask = strategy.quote(state(position=10))

    assert long_bid < flat_bid
    assert long_ask < flat_ask


def test_naive_quotes_ignore_position() -> None:
    strategy = Naive(NaiveConfig(half_spread=2.0))

    assert strategy.quote(state(position=15)) == strategy.quote(state(position=0)) == (9998, 10002)


def test_inventory_skew_shifts_center_by_alpha_times_position() -> None:
    strategy = InventorySkew(InventorySkewConfig(half_spread=1.0, skew=0.5))

    assert strategy.quote(state(position=4)) == (9997, 9999)
    assert strategy.quote(state(position=-4)) == (10001, 10003)


@given(
    gamma=st.floats(1e-3, 1.0),
    factor=st.floats(1.01, 10.0),
    position=st.integers(-20, 20).filter(bool),
)
def test_larger_gamma_gives_larger_reservation_adjustment(
    gamma: float, factor: float, position: int
) -> None:
    s = state(position=position)
    low = AvellanedaStoikov(AvellanedaStoikovConfig(gamma=gamma))
    high = AvellanedaStoikov(AvellanedaStoikovConfig(gamma=gamma * factor))

    low_shift = low.reservation_price(s) - s.mid
    high_shift = high.reservation_price(s) - s.mid

    assert abs(high_shift) > abs(low_shift)
    assert math.copysign(1, high_shift) == -math.copysign(1, position)


def test_avellaneda_stoikov_matches_closed_form() -> None:
    gamma, k, tau, sigma, q, s = 0.1, 1.5, 50.0, 0.5, 3, 10_000.0
    strategy = AvellanedaStoikov(AvellanedaStoikovConfig(gamma=gamma, horizon=tau, k=k))
    quote_state = QuoteState(s, q, step=1, steps_remaining=4_999, sigma=sigma)

    reservation = s - q * gamma * sigma**2 * tau
    spread = gamma * sigma**2 * tau + (2 / gamma) * math.log(1 + gamma / k)

    assert strategy.reservation_price(quote_state) == pytest.approx(reservation)
    assert strategy.spread(quote_state) == pytest.approx(spread)
    assert strategy.quote(quote_state) == (
        math.floor(reservation - spread / 2),
        math.ceil(reservation + spread / 2),
    )


def test_avellaneda_stoikov_finite_horizon_uses_time_to_end() -> None:
    gamma, k = 0.05, 2.0
    strategy = AvellanedaStoikov(AvellanedaStoikovConfig(gamma=gamma, horizon=None, k=k))

    at_end = state(position=5, steps_remaining=0)
    earlier = state(position=5, steps_remaining=100)

    assert strategy.reservation_price(at_end) == at_end.mid
    assert strategy.spread(at_end) == pytest.approx((2 / gamma) * math.log(1 + gamma / k))
    assert strategy.reservation_price(earlier) == pytest.approx(10_000 - 5 * gamma * 0.25 * 100)


def test_avellaneda_stoikov_rejects_non_positive_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        AvellanedaStoikovConfig(gamma=0.0)
    with pytest.raises(ValueError, match="positive"):
        AvellanedaStoikovConfig(k=-1.0)


@pytest.mark.parametrize(
    ("bid", "ask", "expected"),
    [
        (99.5, 101.5, (99, 102)),
        (100.0, 100.0, (100, 101)),
        (99.99999999999999, 101.00000000000001, (100, 101)),
    ],
)
def test_to_ticks_rounds_outward_and_keeps_bid_below_ask(
    bid: float, ask: float, expected: tuple[int, int]
) -> None:
    assert to_ticks(bid, ask) == expected


@pytest.mark.parametrize(
    ("config", "kind"),
    [
        (NaiveConfig(), Naive),
        (InventorySkewConfig(), InventorySkew),
        (AvellanedaStoikovConfig(), AvellanedaStoikov),
    ],
)
def test_build_strategy_dispatches_on_config_type(config: StrategyConfig, kind: type) -> None:
    assert isinstance(build_strategy(config), kind)


INVENTORY_BUILDERS = [
    NaiveConfig(half_spread=0.0),
    InventorySkewConfig(half_spread=0.0, skew=0.01),
    AvellanedaStoikovConfig(gamma=1e-4, horizon=100.0, k=4.0),
]
BOOK_CROSSERS = [
    NaiveConfig(half_spread=0.0),
    InventorySkewConfig(half_spread=0.5, skew=2.0),
    AvellanedaStoikovConfig(gamma=0.05, horizon=100.0, k=4.0),
]


class RecordingStrategy:
    def __init__(self, inner: Strategy) -> None:
        self.inner = inner
        self.quotes: dict[int, Quote] = {}

    def quote(self, state: QuoteState) -> Quote:
        self.quotes[state.step] = self.inner.quote(state)
        return self.quotes[state.step]


@pytest.mark.parametrize("config", INVENTORY_BUILDERS)
@pytest.mark.parametrize("latency", [0, 5])
def test_position_limit_is_reached_but_never_breached(config: StrategyConfig, latency: int) -> None:
    sim = SimulationConfig(n_steps=3_000, maker=MakerConfig(latency=latency, position_limit=20))

    result = simulate(sim, build_strategy(config), seed=0)

    assert np.abs(result.position).max() == 20


@pytest.mark.parametrize("config", BOOK_CROSSERS)
def test_quotes_never_cross_the_book(config: StrategyConfig) -> None:
    sim = SimulationConfig(n_steps=3_000, takers=TakerConfig(informed_share=0.3))
    strategy = RecordingStrategy(build_strategy(config))

    result = simulate(sim, strategy, seed=1)

    requested = np.array([strategy.quotes[t] for t in range(1, sim.n_steps + 1)], dtype=float)
    resting = np.column_stack((result.bid[1:], result.ask[1:]))
    quoted = ~np.isnan(requested) & ~np.isnan(resting)
    assert (requested[quoted] != resting[quoted]).any(), "post-only adjustment never exercised"
    both = ~np.isnan(resting).any(axis=1)
    assert (resting[both, 0] < resting[both, 1]).all()
    assert all(
        (t.buyer_id if t.aggressor is Side.BUY else t.seller_id) != MARKET_MAKER_ID
        for t in result.trades
    )
