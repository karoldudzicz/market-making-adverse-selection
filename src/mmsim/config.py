"""Simulation parameters.

Prices and distances are in ticks, quantities in lots, time in simulation steps.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueConfig:
    """Hidden true value: Gaussian random walk with rare symmetric jumps.

    Attributes
    ----------
    initial : float
        Starting value V_0.
    sigma : float
        Standard deviation of the diffusive increment per step.
    jump_prob : float
        Probability of a jump in a step.
    jump_size : float
        Absolute jump size; the sign is equally likely to be positive or negative.
    """

    initial: float = 10_000.0
    sigma: float = 0.5
    jump_prob: float = 0.002
    jump_size: float = 10.0


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    """Background liquidity quoting around a delayed, noisy reference R_t = V_{t-d} + eps_t.

    Attributes
    ----------
    n_providers : int
        Number of background liquidity providers.
    post_prob : float
        Probability that a provider posts one order on a given side in a step.
    min_depth, max_depth : int
        Range of the uniformly drawn distance of a new order from the rounded reference.
    cancel_prob : float
        Per-step probability that a resting order is cancelled.
    max_distance : float
        Orders further than this from the current reference are always cancelled.
    cancel_through_reference : bool
        Cancel orders at or through the current reference (bid >= R_t or ask <= R_t), which
        a provider would value at a loss by its own reference.
    delay : int
        Information delay d of the reference behind the true value.
    noise_sd : float
        Standard deviation of the reference noise eps_t.
    order_qty : int
        Size of each background order in lots.
    """

    n_providers: int = 4
    post_prob: float = 0.25
    min_depth: int = 2
    max_depth: int = 10
    cancel_prob: float = 0.05
    max_distance: float = 15.0
    cancel_through_reference: bool = True
    delay: int = 5
    noise_sd: float = 1.0
    order_qty: int = 1


@dataclass(frozen=True, slots=True)
class TakerConfig:
    """Market-order flow from informed and noise takers.

    Attributes
    ----------
    arrival_prob : float
        Probability lambda that a taker arrives in a step.
    informed_share : float
        Probability pi that an arriving taker is informed (flow toxicity).
    informed_threshold : float
        Informed takers buy if best ask < V_t - threshold and sell if best bid > V_t + threshold.
    order_qty : int
        Size of each market order in lots.
    """

    arrival_prob: float = 0.3
    informed_share: float = 0.2
    informed_threshold: float = 1.0
    order_qty: int = 1


@dataclass(frozen=True, slots=True)
class MakerConfig:
    """Market maker execution constraints.

    Attributes
    ----------
    latency : int
        Steps L between computing quotes and their activation in the book.
    quote_qty : int
        Size of each quote in lots.
    position_limit : int
        Maximum absolute position; quotes that could breach it are not placed.
    """

    latency: int = 0
    quote_qty: int = 1
    position_limit: int = 20


@dataclass(frozen=True, slots=True)
class NaiveConfig:
    """Symmetric quotes around the background mid: bid = mid - h, ask = mid + h.

    Default values are provisional until parameter tuning.

    Attributes
    ----------
    half_spread : float
        Half-spread h in ticks.
    """

    half_spread: float = 1.0


@dataclass(frozen=True, slots=True)
class InventorySkewConfig:
    """Symmetric quotes shifted against the position: mid - h - alpha*q and mid + h - alpha*q.

    Default values are provisional until parameter tuning.

    Attributes
    ----------
    half_spread : float
        Half-spread h in ticks.
    skew : float
        Skew alpha in ticks per lot of position.
    """

    half_spread: float = 1.0
    skew: float = 0.1


@dataclass(frozen=True, slots=True)
class AvellanedaStoikovConfig:
    """Avellaneda-Stoikov (2008) quotes.

    Default values are provisional until parameter tuning and calibration of ``k``.

    Attributes
    ----------
    gamma : float
        Risk aversion, positive.
    horizon : float or None
        Rolling horizon tau in steps; ``None`` uses the time to the end of the run, T - t,
        as in the original model.
    k : float
        Decay rate of the fill intensity A * exp(-k * delta), per tick; positive.
    """

    gamma: float = 0.01
    horizon: float | None = 100.0
    k: float = 1.0

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.k <= 0:
            raise ValueError(f"gamma and k must be positive, got {self.gamma} and {self.k}")


type StrategyConfig = NaiveConfig | InventorySkewConfig | AvellanedaStoikovConfig


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Full configuration of one simulation run."""

    n_steps: int = 5_000
    value: ValueConfig = ValueConfig()
    background: BackgroundConfig = BackgroundConfig()
    takers: TakerConfig = TakerConfig()
    maker: MakerConfig = MakerConfig()

    def __post_init__(self) -> None:
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.maker.latency < 0:
            raise ValueError(f"latency must be non-negative, got {self.maker.latency}")
