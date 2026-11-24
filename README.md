# Market Making Under Adverse Selection

How should a market maker quote to earn the spread without losing to better-informed traders, and how much do toxic flow and latency cost?

Work in progress.

## Design decisions

- All limit orders (market maker and background liquidity) are post-only: a price that would cross the book slides to one tick behind the best opposite price. Takers are therefore the only aggressors.
- Market orders are immediate-or-cancel: quantity left after the opposite side is exhausted is discarded.
- When one side of the book is empty, the mid falls back to the last two-sided mid seen for the same view (full book, or book excluding one owner), and to the initial value before that.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m mypy src
```
