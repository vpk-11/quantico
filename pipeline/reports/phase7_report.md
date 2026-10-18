# Phase 6/7 report: paper trading loop and performance tracking

Phase 3 verdict: **FAIL** (no validated edge). Replay and forward are reported separately and never pooled; no edge claim is made unless the rule below passes.

Last updated: 2026-10-16 (from the data, not the clock). Paper money only. Signal: baseline ridge refit at each rebalance. Execution: C++ simulator, `immediate`. Long-only top 5 of 20, fixed $25,000,000, rebalance every 5 trading days. Intraday slices are synthetic (see phase5_report.md).

Limits: survivorship bias (all 20 names survived to 2026), price basis is `close` (no dividends), cost model is assumed and uncalibrated, the window is short, and dates up to `[phase6] replay_end_date` are a pseudo-live replay, not true paper trading.

## Replay

2026-01-02 to 2026-10-12 (195 trading days, 39 rebalances, 39 with a matured 5-day outcome).

| metric | value |
|---|---|
| NAV start / end | 25,000,000 / 30,129,988 |
| return | 20.52% |
| Sharpe (annualized, daily) | 1.25 |
| max drawdown | -9.92% |
| forward rank IC | mean 0.0249, Newey-West t 0.61 |

### Attribution (USD)

| component | value |
|---|---|
| market (equal-weight 20) | 3,775,251 |
| selection | 1,887,172 |
| costs | -532,435 |
| total NAV change | 5,129,988 |

### Sector contribution (frictionless P&L, USD)

| sector | pnl |
|---|---|
| industrials | 2,215,302 |
| materials | 1,268,089 |
| consumer_discretionary | 961,214 |
| real_estate | 641,303 |
| communication_services | 615,095 |
| information_technology | 558,910 |
| health_care | 41,726 |
| consumer_staples | 41,610 |
| financials | -86,470 |
| utilities | -110,456 |
| energy | -483,901 |

### Signal decay

Rolling 12-rebalance IC: 0.0726, Newey-West t 1.13.
Trend (IC vs rebalance index): slope 0.000457 (rising).
Phase 3 out-of-sample yearly IC band: -0.0507 to 0.0401. Replay mean IC is inside that band.

### Comparators

Equal-weight buy-and-hold Sharpe over the same dates: 3.91 (flattered by survivorship bias: all 20 names survived to 2026).
Random top-5 null, 200 seeded draws, same rebalance schedule and cost model: Sharpe mean 1.07, range -1.11 to 3.41. Strategy Sharpe sits at the 57th percentile.

## Forward

2026-10-13 to 2026-10-16 (4 trading days, 1 rebalances, 0 with a matured 5-day outcome).

| metric | value |
|---|---|
| NAV start / end | 30,129,988 / 29,520,931 |
| return | -2.02% |
| Sharpe (annualized, daily) | -14.22 |
| max drawdown | -1.10% |
| forward rank IC | mean nan |

Insufficient sample: 1 forward rebalances, need 52. No claim on edge or decay.

## Phase 3 out-of-sample yearly IC (for comparison)

| year | mean IC | days |
|---|---|---|
| 2016 | 0.0334 | 252 |
| 2017 | 0.0289 | 251 |
| 2018 | 0.0191 | 251 |
| 2019 | -0.0041 | 252 |
| 2020 | -0.0119 | 253 |
| 2021 | -0.0507 | 252 |
| 2022 | -0.0273 | 251 |
| 2023 | 0.0276 | 250 |
| 2024 | 0.0080 | 252 |
| 2025 | 0.0401 | 245 |

Yearly IC flips sign (positive 2016 to 2018 and 2023 to 2025, negative 2019 to 2022), which is what an edge that never cleared its significance gates looks like.
