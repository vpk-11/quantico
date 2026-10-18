# Phase 3.5 report: escalation comparisons

Phase 3 verdict: FAIL. Path: exploratory (3.5a random forest `rf_h5`, 3.5c 10-day horizon `ridge_h10`). 3.5b recency weighting is skipped on this path.

Final trial count k = 3. Every gate below is restated at this k, including the baseline's. All variant results are labelled exploratory.

## Adoption outcome

Phase 4 input: `ridge_h5_baseline` (no validated edge).

| model | gates passed | integrity ok | paired test | qualifies |
|---|---|---|---|---|
| rf_h5 | not all | yes | FAIL | no |
| ridge_h10 | not all | yes | n/a (different target) | no |

## Comparison at k = 3

| model | label | mean IC | NW t | R2 vs naive | G1 | G2 | G3 | G4 | G5 | paired mean diff | paired NW t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ridge_h5_baseline | baseline (no validated edge) | 0.00619 | 0.526 | 0.001191 | FAIL | PASS | FAIL | FAIL | FAIL | n/a | n/a |
| rf_h5 | exploratory | -0.00315 | -0.317 | -0.000960 | FAIL | FAIL | FAIL | FAIL | FAIL | -0.00934 | -0.887 |
| ridge_h10 | exploratory | 0.01403 | 0.952 | 0.001686 | FAIL | PASS | FAIL | PASS | FAIL | n/a | n/a |

`ridge_h10` predicts a 10-day return, so its IC is not directly comparable to the 5-day models. It is judged against the gates only, with no paired test.

## Gate values and thresholds

### ridge_h5_baseline

| gate | name | value | threshold | result |
|---|---|---|---|---|
| G1 | IC significance | mean IC 0.00619, NW t 0.526 | mean IC > 0 and t >= 2.394 | FAIL |
| G2 | Beats naive | R2 vs naive 0.001191 | > 0 | PASS |
| G3 | Consistency | 6 of 10 years positive | >= 7 | FAIL |
| G4 | Permutation test | p 0.09950 (200 shuffles) | < 0.01667 | FAIL |
| G5 | Block bootstrap | lower bound -0.02696 | > 0 at 0.9833 two-sided | FAIL |

### rf_h5

| gate | name | value | threshold | result |
|---|---|---|---|---|
| G1 | IC significance | mean IC -0.00315, NW t -0.317 | mean IC > 0 and t >= 2.394 | FAIL |
| G2 | Beats naive | R2 vs naive -0.000960 | > 0 | FAIL |
| G3 | Consistency | 5 of 10 years positive | >= 7 | FAIL |
| G4 | Permutation test | p 0.74129 (200 shuffles) | < 0.01667 | FAIL |
| G5 | Block bootstrap | lower bound -0.03016 | > 0 at 0.9833 two-sided | FAIL |

Paired test vs baseline: mean IC difference -0.00934, NW t -0.887, needs >= 2.394: FAIL

### ridge_h10

| gate | name | value | threshold | result |
|---|---|---|---|---|
| G1 | IC significance | mean IC 0.01403, NW t 0.952 | mean IC > 0 and t >= 2.394 | FAIL |
| G2 | Beats naive | R2 vs naive 0.001686 | > 0 | PASS |
| G3 | Consistency | 6 of 10 years positive | >= 7 | FAIL |
| G4 | Permutation test | p 0.00498 (200 shuffles) | < 0.01667 | PASS |
| G5 | Block bootstrap | lower bound -0.02571 | > 0 at 0.9833 two-sided | FAIL |

## Integrity (I1, I4, I5, I6 per new model)

| model | test | name | result | detail |
|---|---|---|---|---|
| rf_h5 | I1 | Split integrity | ok | 10 folds, test years 2016..2025, no overlap |
| rf_h5 | I4 | Shuffle control | ok | |mean| 0.000091 < 3*std/sqrt(n) 0.000994 |
| rf_h5 | I5 | Planted-signal recovery | ok | mean IC 0.5024 (need > 0.3), gates failed: none |
| rf_h5 | I6 | Determinism | ok | two runs: identical predictions and metrics |
| ridge_h10 | I1 | Split integrity | ok | 10 folds, test years 2016..2025, no overlap |
| ridge_h10 | I4 | Shuffle control | ok | |mean| 0.000652 < 3*std/sqrt(n) 0.001006 |
| ridge_h10 | I5 | Planted-signal recovery | ok | mean IC 0.5087 (need > 0.3), gates failed: none |
| ridge_h10 | I6 | Determinism | ok | two runs: identical predictions and metrics |

## Random forest feature importances per fold

Impurity-based importances are biased toward high-variance features, so read them as a rough guide only.

| test year | mom_21 | mom_63 | mom_126 | mom_252 | mean_rev_z | volatility | liquidity |
|---|---|---|---|---|---|---|---|
| 2016 | 0.1101 | 0.1344 | 0.2843 | 0.1884 | 0.0466 | 0.1758 | 0.0604 |
| 2017 | 0.0984 | 0.1929 | 0.2483 | 0.2018 | 0.0410 | 0.1680 | 0.0497 |
| 2018 | 0.1128 | 0.1879 | 0.2279 | 0.2181 | 0.0454 | 0.1600 | 0.0478 |
| 2019 | 0.1149 | 0.1938 | 0.2334 | 0.1949 | 0.0452 | 0.1668 | 0.0511 |
| 2020 | 0.1159 | 0.2048 | 0.2070 | 0.2076 | 0.0469 | 0.1672 | 0.0506 |
| 2021 | 0.2228 | 0.1954 | 0.1701 | 0.1487 | 0.0495 | 0.1690 | 0.0445 |
| 2022 | 0.2164 | 0.2126 | 0.1692 | 0.1485 | 0.0448 | 0.1540 | 0.0545 |
| 2023 | 0.2120 | 0.2238 | 0.1447 | 0.1557 | 0.0511 | 0.1586 | 0.0541 |
| 2024 | 0.1959 | 0.2244 | 0.1688 | 0.1509 | 0.0482 | 0.1527 | 0.0591 |
| 2025 | 0.1910 | 0.2082 | 0.1638 | 0.1678 | 0.0459 | 0.1423 | 0.0809 |

## Config snapshot

```
rf_n_estimators = 300
rf_max_depth = 6
rf_min_samples_leaf = 200
rf_max_features = 0.5
recency_half_lives_years = [2, 4, 8]
alt_horizon_days = 10
alt_horizon_newey_west_lags = 10
```
