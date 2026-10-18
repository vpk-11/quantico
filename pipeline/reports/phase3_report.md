# Phase 3 report: baseline ridge, expanding walk-forward

Model: `ridge_h5_baseline`. Trial count k = 1.

## Verdict: FAIL

Label: no validated edge

## Config snapshot

```
target_horizon_days = 5
initial_train_years = 8
test_step_years = 1
ridge_alpha = 1.0
log_liquidity = True
min_tickers_per_day = 10
significance_alpha = 0.05
newey_west_lags = 5
min_positive_years = 7
n_permutations = 200
bootstrap_resamples = 2000
bootstrap_block_days = 20
random_seed = 42
```

## Folds

First usable year: 2008 (expected 2008).

| test year | train first | train last | train rows | purged rows | test rows |
|---|---|---|---|---|---|
| 2016 | 2008-01-03 | 2015-12-23 | 40180 | 100 | 5040 |
| 2017 | 2008-01-03 | 2016-12-22 | 45220 | 100 | 5020 |
| 2018 | 2008-01-03 | 2017-12-21 | 50240 | 100 | 5020 |
| 2019 | 2008-01-03 | 2018-12-21 | 55260 | 100 | 5040 |
| 2020 | 2008-01-03 | 2019-12-23 | 60300 | 100 | 5060 |
| 2021 | 2008-01-03 | 2020-12-23 | 65360 | 100 | 5040 |
| 2022 | 2008-01-03 | 2021-12-23 | 70400 | 100 | 5020 |
| 2023 | 2008-01-03 | 2022-12-22 | 75420 | 100 | 5000 |
| 2024 | 2008-01-03 | 2023-12-21 | 80420 | 100 | 5040 |
| 2025 | 2008-01-03 | 2024-12-23 | 85460 | 100 | 4900 |

## Gates

| gate | name | value | threshold | result |
|---|---|---|---|---|
| G1 | IC significance | mean IC 0.00619, NW t 0.526 | mean IC > 0 and t >= 1.960 | FAIL |
| G2 | Beats naive | R2 vs naive 0.001191 | > 0 | PASS |
| G3 | Consistency | 6 of 10 years positive | >= 7 | FAIL |
| G4 | Permutation test | p 0.09950 (200 shuffles) | < 0.05000 | FAIL |
| G5 | Block bootstrap | lower bound -0.02063 | > 0 at 0.9500 two-sided | FAIL |

## Integrity

| test | name | result | detail |
|---|---|---|---|
| I1 | Split integrity | ok | 10 folds, test years 2016..2025, no overlap |
| I2 | Target correctness | ok | exact targets on known series, last 5 rows per ticker NULL |
| I3 | Scaler isolation | ok | scaler and training transform unchanged when a fold's test features are shifted |
| I4 | Shuffle control | ok | |mean| 0.000104 < 3*std/sqrt(n) 0.000933 |
| I5 | Planted-signal recovery | ok | mean IC 0.4932 (need > 0.3), gates failed: none |
| I6 | Determinism | ok | two runs: identical predictions and metrics |

## Extras (no gate)

### IC by year

| year | mean IC |
|---|---|
| 2016 | 0.03345 |
| 2017 | 0.02892 |
| 2018 | 0.01908 |
| 2019 | -0.00413 |
| 2020 | -0.01192 |
| 2021 | -0.05067 |
| 2022 | -0.02734 |
| 2023 | 0.02762 |
| 2024 | 0.00798 |
| 2025 | 0.04014 |

IC in 2020: -0.01192. IC in 2022: -0.02734.

Hit rate (sign of prediction equals sign of target): 0.5415

Quintile spread (top fifth minus bottom fifth, daily mean 5-day return): 0.001075, NW t 0.833

Single-feature benchmark (mom_21 as prediction): mean IC -0.00451, NW t -0.397

Daily IC distribution over 2509 days: std 0.32798

| quantile | IC |
|---|---|
| 0.0 | -0.83910 |
| 0.05 | -0.53083 |
| 0.25 | -0.23609 |
| 0.5 | 0.00752 |
| 0.75 | 0.23759 |
| 0.95 | 0.55789 |
| 1.0 | 0.87970 |

### Ridge coefficients per fold (standardized features)

| test year | mom_21 | mom_63 | mom_126 | mom_252 | mean_rev_z | volatility | liquidity |
|---|---|---|---|---|---|---|---|
| 2016 | +0.000150 | -0.000779 | +0.000510 | -0.000527 | -0.001127 | +0.000605 | -0.000644 |
| 2017 | +0.001095 | -0.001227 | +0.000701 | -0.000302 | -0.001479 | +0.001199 | -0.000775 |
| 2018 | +0.000984 | -0.001117 | +0.000579 | +0.000062 | -0.001463 | +0.001083 | -0.000675 |
| 2019 | +0.001041 | -0.000975 | +0.000542 | +0.000055 | -0.001604 | +0.000978 | -0.000742 |
| 2020 | +0.000748 | -0.000914 | +0.000279 | -0.000012 | -0.001357 | +0.000888 | -0.000400 |
| 2021 | +0.000411 | -0.000964 | +0.000420 | +0.000258 | -0.000975 | +0.001893 | -0.000293 |
| 2022 | +0.000178 | -0.000709 | +0.000548 | +0.000081 | -0.000887 | +0.001870 | -0.000056 |
| 2023 | +0.000108 | -0.000913 | +0.000595 | +0.000236 | -0.000763 | +0.001829 | -0.000611 |
| 2024 | +0.000031 | -0.000854 | +0.000790 | -0.000039 | -0.000656 | +0.001889 | -0.000219 |
| 2025 | +0.000043 | -0.000756 | +0.000419 | +0.000307 | -0.000627 | +0.001874 | +0.000023 |
