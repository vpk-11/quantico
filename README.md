# Quantico
<!-- version: v0.5.0 -->
![Version](https://img.shields.io/badge/version-v0.5.0-blue)

A quantitative research pipeline that predicts 5-trading-day forward returns for liquid US
equities, validates those predictions without look-ahead bias, and tests them against a modeled
cost model using a custom C++ execution simulator.

This is a systematic/quantitative research project, not high-frequency trading: daily cadence,
no microsecond-scale infrastructure, no live brokerage integration, no real money at any stage.
The subject of this project is the pipeline itself, a correctly engineered and correctly gated
system, not the profitability of the trading signal.

## Pipeline

A 7-stage pipeline plus a conditional Phase 3.5:

1. **Research store**: pulls and stores ~19 years of daily OHLCV (2007 to 2025) for 20 large-cap
   US equities, spread across 11 sectors, spanning the 2008 and 2020 crises.
2. **Feature engineering**: momentum, mean reversion, volatility, and liquidity features, with
   no look-ahead anywhere.
3. **Baseline model**: a pooled ridge regression, scored out-of-sample under an expanding
   walk-forward with five pre-registered statistical gates.
3.5. **Escalation** (conditional on the Phase 3 verdict): random forest and a 10-day horizon,
   scored as controlled comparisons against the baseline, never blended in.
4. **C++ execution simulator**: a standalone program, run as a subprocess, that applies modeled
   fees, spread, impact, and latency to trade decisions instead of assuming free, instant fills.
5. **Execution strategy comparison**: the same signal replayed through three strategies
   (immediate, TWAP, VWAP) and a robustness grid (intraday-path variants, cost multipliers,
   capital levels).
6. **Paper trading loop**: runs the full pipeline forward daily on simulated money, tagging
   every row `replay` or `forward` so a short or lucky replay window is never read as live
   performance.
7. **Performance tracking**: NAV attribution, sector contribution, signal decay, and a
   pre-registered edge-claim rule, evaluated separately per mode and never pooled.

## Phase 3 verdict and limits

**FAIL ("no validated edge")** at k=1: mean daily IC 0.0062, Newey-West t 0.53 (needs 1.96), 6 of
10 years positive (needs 7), permutation p 0.0995 (needs below 0.05). All six integrity tests
pass, so this is a trustworthy "no signal found", not a broken harness. Baseline predictions flow
forward under that label anyway: this project's subject is the pipeline, not the signal, so a
FAIL does not block later phases. Escalation candidates (random forest, 10-day horizon) were
tried and neither qualified under the pre-registered adoption rule either.

The paper loop's replay window (2026-01-02 to 2026-10-12) shows NAV $25.0M to
$30,129,988 (return 20.52%), Sharpe 1.25, max drawdown 9.92%. These figures are descriptive, not
evidence of skill: they measure execution cost drag and selection behaviour on a signal with no
validated edge, over a short window, and no edge claim is made anywhere in this repo's generated
reports unless a pre-registered rule (at least 52 forward rebalances, IC significance, and a
random-null Sharpe percentile, all three) passes. Forward rows accumulate from 2026-10-13;
current status is in `pipeline/reports/phase7_report.md`.

Limits: survivorship bias (all 20 names survived to 2026, none delisted or merged away), price
basis is `close` (split-adjusted but not dividend-adjusted), the execution cost model is assumed
and uncalibrated (fee, spread, and impact parameters are not fit to any real execution data),
intraday data is synthetic (no real intraday history exists for the backtest window, so each
daily bar is expanded into a deterministic 13-slice path), and dates up to 2026-10-12 are a
pseudo-live replay, not true forward paper trading.

## Run

```
conda activate quantico
cd pipeline
pip install -r requirements.txt

python src/research_store.py build && python src/features.py build   # Phases 1 to 2
python src/baseline.py run                                           # Phase 3
bash scripts/run_phase35.sh                                           # Phase 3.5 (hours)
make -C ../simulator && python src/backtest.py run                   # Phases 4 to 5
python src/paper.py run                                              # Phases 6 to 7
```

Full commands, undo notes, and generated reports are in `pipeline/README.md`.

## Tests

```
python -m pytest pipeline/tests
```

91 tests across 8 files, no framework required (each file also runs standalone with
`python tests/test_<name>.py`): `test_research_store.py` (9), `test_features.py` (10),
`test_phase3.py` (17), `test_phase35.py` (9), `test_phase4.py` (16), `test_phase5.py` (7),
`test_phase6.py` (13), `test_phase7.py` (10).

## Changelog
- **v0.5.0** (2026-10-18): minor bump
- **v0.4.1** (2026-10-18): patch bump
- **v0.4.0** (2026-10-18): minor bump
- **v0.3.0** (2026-10-18): minor bump
- **v0.2.0** (2026-09-28): minor bump
