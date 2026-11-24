# Market Making Under Adverse Selection

How should a market maker quote to earn the spread without losing to better-informed traders, and how much do toxic flow and latency cost?

Work in progress.

## Design decisions

- All limit orders (market maker and background liquidity) are post-only: a price that would cross the book slides to one tick behind the best opposite price. Takers are therefore the only aggressors.
- Market orders are immediate-or-cancel: quantity left after the opposite side is exhausted is discarded.
- When one side of the book is empty, the mid falls back to the last two-sided mid seen for the same view (full book, or book excluding one owner), and to the initial value before that.
- All market randomness (value path, background order flow, taker arrivals) is drawn up front from separate seeded streams. Background order lifetimes are drawn as geometric at posting, which is equivalent to a 5% per-step cancel probability but does not depend on the state of the book. On a given seed every strategy therefore faces the same market.
- The taker type is a fixed uniform draw per step compared with π, so raising π on a seed turns a subset of noise takers into informed ones and leaves the rest of the flow unchanged.
- Background liquidity also cancels its orders at or through its own reference (bid >= R_t or ask <= R_t). This rule is not in the base specification, which cancels only orders more than 15 ticks from R_t. After a 10-tick jump, the orders left on the wrong side of the new value are within 15 ticks, so without the rule they rest until their lifetime ends and the market mid lags V for tens of steps instead of about d. With the rule, d is the actual information delay of the background. Informed takers still pick off the stale background during those d steps. The rule is the flag `BackgroundConfig.cancel_through_reference` (on by default); E0 reports both variants.
- Background liquidity: 4 providers, each posting a 1-lot order per side with probability 0.25 per step at a uniform depth of 2-10 ticks from the rounded reference. Default π = 0.2. The book starts empty and fills within the first steps; there is no warm-up period.
- The market maker keeps a resting quote whose price is unchanged instead of cancelling and re-posting it, so it does not lose queue priority. The position limit is checked when a quote is activated, against the position at that moment.
- Markouts at horizon h exclude fills with t + h beyond the end of the run.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m mypy src
```
