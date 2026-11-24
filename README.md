# Market Making Under Adverse Selection

How should a market maker quote to earn the spread without losing to better-informed traders, and how much do toxic flow and latency cost?

Work in progress.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m mypy src
```
