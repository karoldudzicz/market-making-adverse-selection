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
- Strategy quotes are rounded outward to whole ticks (bid down, ask up), with a 1e-9 tolerance so float noise does not widen a quote by a tick. If rounding leaves bid >= ask, the ask is set to bid + 1.
- Avellaneda-Stoikov uses a constant rolling horizon τ, tuned together with γ, instead of τ = T - t. With T = 5,000 steps and σ = 0.5, σ²(T - t) is about 1,250 at the start of a run, so the original quotes would be extremely wide and skewed early and almost unskewed at the end, and behaviour would depend mainly on the time within the run. The original form is available as `AvellanedaStoikovConfig(horizon=None)`. With a rolling horizon, γ and τ enter the reservation price and the risk part of the spread only through γ·τ, and γ alone has a small effect on the intensity part, so grid points with the same γ·τ give near-identical results.
- k is calibrated from naive runs with h from 0 to 6 ticks on the tuning seeds. The fill rate per quoted side per step is measured in half-tick buckets of distance from the background mid the quote was computed from. log(rate) = log A - kδ is fitted by least squares weighted by the fill count of each bucket (the variance of a log rate is about 1/fills), using buckets with at least 20 fills.
- Confidence intervals are normal approximations (mean ± 1.96·s/√n). Strategy comparisons use per-seed paired differences.
- Strategy parameters are tuned on seeds 0-199 at π = 0.2 and L = 0. The objective is the Sharpe ratio across seeds (mean / standard deviation of run PnL), with a one-standard-error rule: among grid points within one standard error of the best Sharpe ratio (SE ≈ √((1 + S²/2)/n)), the one with the highest mean PnL is selected. Because V is a martingale, inventory has zero expected PnL, so a mean-PnL objective drives α and γ·τ towards zero and skew and A-S collapse into naive quoting. A-S tuned on mean PnL (γ = 0.01, τ = 10, one-standard-error rule picking the lowest PnL standard deviation) is reported alongside as the risk-neutral contrast. The full grids under both objectives are in `results/tuning_*.csv`. `run_all` reruns the tuning and stops if it selects parameters other than those in `mmsim/config.py`.
- E2, E3, E5 and E6 use report seeds 1000-1499. E4 (7 values of π × 9 half-spreads × 2 swept families, plus the 4 tuned strategies) uses seeds 1000-1199 to keep `run_all` near 15 minutes. Its jump-frequency sensitivity variant (jump probability 0.01 instead of 0.002, strategy parameters unchanged and not retuned) uses seeds 1000-1099.
- The break-even half-spread is the smallest h on the grid (step 0.5 ticks) with mean PnL ≥ 0; 0 means the tightest quotes are already profitable. The optimal half-spread is the grid point with the highest mean PnL.
- The part of the adverse-selection loss attributed to jumps is the edge on informed fills in the step of a jump or the 19 steps after it. The 20-step window covers the background delay d = 5 and the 9-step recovery of the mid measured in E0.
- Markout curves pool the per-run means weighted by each run's number of fills against that counterparty type.

## How to run

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m mypy src
uv run python -m mmsim.experiments.run_all   # all results in results/, about 15 min on 12 threads
```
