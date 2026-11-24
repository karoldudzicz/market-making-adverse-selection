"""Minimal quoting rules used as test doubles for the engine."""

import math

from mmsim.strategies import Quote, QuoteState


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
