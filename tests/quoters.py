"""Minimal quoting rules used as test doubles for the engine."""

import math

from mmsim.strategies import Quote, QuoteState


class SymmetricQuoter:
    def __init__(self, half_spread: float) -> None:
        self.half_spread = half_spread

    def quote(self, state: QuoteState) -> Quote:
        return math.floor(state.mid - self.half_spread), math.ceil(state.mid + self.half_spread)


class NoQuotes:
    def quote(self, state: QuoteState) -> Quote:
        return None, None


class StepEncodingBid:
    """Bids far below the market at a price that encodes the step the quote was computed in."""

    BASE = 9_000

    def quote(self, state: QuoteState) -> Quote:
        return self.BASE + state.step, None


class AggressiveBid:
    """Always bids at the top of the book; post-only keeps it one tick below the ask."""

    def quote(self, state: QuoteState) -> Quote:
        return math.ceil(state.mid) + 100, None
