# Phase 5 report: execution strategy comparison

Phase 3 verdict: **FAIL** (no validated edge). The portfolio figures below are descriptive, not evidence of skill: they show execution cost drag and selection behaviour on a signal with no validated edge.

Signal: `ridge_h5_baseline` predictions, 50180 rows, replayed 2016-01-05 to 2025-12-23.
Long-only top 5 of 20, equal weight, fixed $25,000,000 base capital (no compounding), rebalanced every 5 trading days (`[phase3] target_horizon_days`, not duplicated). Orders decided at close t, executed during t+1; two benchmarks are reported: arrival (the open of t+1) and interval VWAP (the volume-weighted price over t+1's synthetic slices).

Intraday data is synthetic: each daily bar is expanded into 13 slices (deterministic open, extreme, extreme, close price path; U-shaped volume). Timing differences between strategies therefore reflect that assumed path, not real intraday behaviour; the robustness grid below shows how much.

Costs (assumed, uncalibrated): fee 1.0 bps, half-spread 1.0 bps, impact 10.0 bps x sqrt(participation), participation cap 10% per slice, latency 1 slice(s).

## Base cell: realized execution cost vs arrival (bps, notional-weighted; positive is worse)

| strategy | orders | fill rate | drift | spread | impact | fee | total |
|---|---|---|---|---|---|---|---|
| immediate | 3458 | 1.0000 | -0.57 | 1.00 | 2.14 | 1.00 | **3.57** |
| twap | 3458 | 1.0000 | 1.83 | 1.00 | 0.84 | 1.00 | **4.67** |
| vwap | 3458 | 1.0000 | 1.91 | 1.00 | 0.80 | 1.00 | **4.71** |

## Base cell: realized execution cost vs interval_vwap (bps, notional-weighted; positive is worse)

| strategy | orders | fill rate | drift | spread | impact | fee | total |
|---|---|---|---|---|---|---|---|
| immediate | 3458 | 1.0000 | -2.23 | 1.00 | 2.14 | 1.00 | **1.91** |
| twap | 3458 | 1.0000 | 0.16 | 1.00 | 0.84 | 1.00 | **3.00** |
| vwap | 3458 | 1.0000 | 0.25 | 1.00 | 0.80 | 1.00 | **3.05** |

Lowest total cost vs arrival (base cell): **immediate**.

## Robustness grid

Every strategy re-run across intraday variant (base, swap_extremes, flat_volume, plus noisy_volume's 5 seeded paths reported as mean and range), cost multiplier (0.5x, 1x, 2x), and capital ($25M, $250M, $2.5B). Run via the Python reference fill (Phase 4 proves this matches the C++ simulator bit for bit), not the base cell's subprocess path. Cells are split into ones where every strategy filled 100% ('feasible') and ones where at least one did not; a flipped ranking that only happens in infeasible cells is a cap-binding artifact, not a real execution-strategy finding.

### vs arrival

Feasible cells (100% fill): ranking **not established** across 12 cells.
Win counts by cell: immediate: 6, twap: 4, vwap: 2. Mean gap to the per-cell winner: immediate: 0.37 bps, twap: 0.71 bps, vwap: 0.72 bps.
The 12 feasible cells are 4 independent intraday paths x 3 cost multipliers. Win counts by path (collapsing the multipliers first): immediate: 3, twap: 1.
The mean gaps (0.37 to 0.72 bps) are smaller than the 3.40 bps cost change between the 0.5x and 2x cost multipliers, so execution-strategy differences here are below the cost model's own uncertainty.

Infeasible cells (partial fill): ranking **not established** across 24 cells.
Win counts by cell: immediate: 9, twap: 5, vwap: 10. Mean gap to the per-cell winner: immediate: 0.50 bps, twap: 0.46 bps, vwap: 0.43 bps.

### vs interval_vwap

Feasible cells (100% fill): ranking **not established** across 12 cells.
Win counts by cell: immediate: 6, twap: 4, vwap: 2. Mean gap to the per-cell winner: immediate: 0.37 bps, twap: 0.71 bps, vwap: 0.72 bps.
The 12 feasible cells are 4 independent intraday paths x 3 cost multipliers. Win counts by path (collapsing the multipliers first): immediate: 3, twap: 1.
The mean gaps (0.37 to 0.72 bps) are smaller than the 3.40 bps cost change between the 0.5x and 2x cost multipliers, so execution-strategy differences here are below the cost model's own uncertainty.

Infeasible cells (partial fill): ranking **not established** across 24 cells.
Win counts by cell: immediate: 9, twap: 9, vwap: 6. Mean gap to the per-cell winner: immediate: 0.51 bps, twap: 0.45 bps, vwap: 0.44 bps.

### noisy_volume (5 seeds): mean and range, total bps

| strategy | benchmark | mean | min | max |
|---|---|---|---|---|
| immediate | arrival | 5.19 | 1.88 | 9.73 |
| immediate | interval_vwap | 4.20 | 0.30 | 9.77 |
| twap | arrival | 5.68 | 1.95 | 10.24 |
| twap | interval_vwap | 4.66 | 2.03 | 8.67 |
| vwap | arrival | 5.58 | 1.89 | 10.07 |
| vwap | interval_vwap | 4.59 | 2.05 | 8.58 |

## Capital and participation

Requested participation (order quantity over the slice-day's total volume), by capital level, pooled across every grid cell and strategy.

| capital | median | 95th pct | max |
|---|---|---|---|
| $25,000,000 | 0.1284% | 1.5381% | 4.7146% |
| $250,000,000 | 1.8107% | 13.4859% | 47.1458% |
| $2,500,000,000 | 19.0430% | 125.7017% | 407.9687% |

Fill rate by capital level (mean and worst cell, across every variant/multiplier/strategy/benchmark at that capital). Below 100% is the participation cap binding, not a bug: the cell is reporting a partial fill explicitly, not silently dropping it.

| capital | mean fill rate | worst cell fill rate |
|---|---|---|
| $25,000,000 | 100.00% | 100.00% |
| $250,000,000 | 86.76% | 85.32% |
| $2,500,000,000 | 30.64% | 29.49% |

At $25M across 20 large caps, order sizes are a small share of daily volume; the sweep shows where, if anywhere, strategies diverge as capital scales up, and where the participation cap starts binding hard enough to leave orders partially unfilled.

## Comparators

| run | return | Sharpe | max drawdown |
|---|---|---|---|
| immediate | 232.94% | 0.94 | -24.73% |
| twap | 229.30% | 0.92 | -26.09% |
| vwap | 229.24% | 0.92 | -26.09% |
| frictionless (zero costs, immediate) | 249.13% | 1.00 | -23.67% |
| equal-weight all 20, no trading | 1461.22% | 1.15 | -38.89% |

Equal-weight buy-and-hold is flattered by survivorship bias (all 20 names survived to 2026). Random top-5 null, 200 seeded draws, same rebalance schedule and cost model: Sharpe mean 0.81, range 0.39 to 1.21. The **immediate** strategy's Sharpe (0.94) sits at the **84th percentile** of this null, i.e. that much of the null falls below it. This is below the 95th-percentile threshold Phase 7's edge rule requires, so this result alone would not support an edge claim even if the forward-IC and gate conditions were otherwise met.

These numbers are not evidence of profitability. Survivorship bias applies; the cost model is assumed and uncalibrated; the window is short.
