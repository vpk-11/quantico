# pipeline/

Python side of Quantico: the research-to-paper-trading pipeline. `simulator/` (C++)
is the separate execution-simulator component.

Folder name is a working placeholder, same as the project name.

## Layout

```
pipeline/
  config.toml        date range, basket, cleaning + feature windows (data, not code)
  requirements.txt
  src/
    research_store.py  Phase 1: pull / clean / store / query daily OHLCV
    features.py        Phase 2: momentum, mean reversion, volatility, liquidity
    walkforward.py     Phase 3: target, preprocessor, folds (purge gap), runner
    models.py          Phase 3: model interface + registry (ridge)
    evaluate.py        Phase 3: IC, Newey-West, gates G1 to G5, extras
    controls.py        Phase 3: shuffle and planted-signal controls
    integrity.py       Phase 3: integrity tests I1 to I6
    trials.py          Phase 3: trial registry + predictions storage
    report.py          Phase 3: report renderer
    baseline.py        Phase 3: CLI run / verify / report
    escalation.py      Phase 3.5: random forest and 10-day horizon comparisons, adoption rule
    execution.py       Phase 4: synthetic intraday slices, Python reference fills, simulator subprocess wrapper
    backtest.py        Phase 5: immediate / TWAP / VWAP comparison on one held-constant signal
    paper.py           Phase 6/7: forward paper loop and performance tracking
  scripts/
    run_phase35.sh     Phase 3.5 run + verify (hours, random forest nulls)
    daily.sh           daily paper step (cron-ready)
  tests/
    test_research_store.py   offline: clean() logic + sqlite round-trip
    test_features.py          offline: feature formulas + no-lookahead + round-trip
    test_phase3.py            offline: Phase 3 integrity tests on synthetic data
    test_phase35.py           offline: weights, horizon-10, random forest integrity, adoption rule
    test_phase4.py            C++ vs Python reference bit-exact parity (needs `make -C simulator`)
    test_phase6.py            offline: fill accounting, target selection, no-lookahead in predict
  reports/
    phase3_report.md, phase3_5_report.md, phase5_report.md, phase7_report.md   generated (tracked)
  data/
    research_store.db  SQLite store (git-ignored, holds daily_bars + features + trials + predictions)
```

Later phases (model + walk-forward, paper loop, tracking) add modules under
`src/` and tests under `tests/`.

## Setup

```
conda activate quantico            # env: python 3.12
pip install -r requirements.txt
```

## Run

All commands run from `pipeline/`.

```
python src/research_store.py build                     # Phase 1: pull + clean + store the basket
python src/research_store.py verify                     # Phase 1: coverage + spot-check report
python src/research_store.py query AAPL 2020-01-01 2020-03-31

python src/features.py build                           # Phase 2: compute + store all features
python src/features.py verify                           # Phase 2: coverage + no-lookahead + spot checks
python src/features.py show AAPL 2020-03-20 2020-03-25

python src/baseline.py run                             # Phase 3: walk-forward, gates, store predictions, write report
python src/baseline.py verify                          # Phase 3: re-derive and check stored output
python src/baseline.py report                          # Phase 3: regenerate reports/phase3_report.md

bash scripts/run_phase35.sh                            # Phase 3.5: random forest + 10-day horizon (hours)

make -C ../simulator                                   # Phase 4: build the C++ simulator
python src/backtest.py run                             # Phase 5: execution comparison -> reports/phase5_report.md
python src/backtest.py bench                           # Phase 4: C++ vs Python reference timing
python src/paper.py run                                # Phase 6/7: pull bars, process new days, write reports/phase7_report.md
bash scripts/daily.sh                                  # the same, as a cron job
```

`features.py build` needs `daily_bars` to exist already (run `research_store.py
build` first).

## Phase 1: research store

Pulls, cleans, and stores 19 years of daily OHLCV (2007-01-01 to 2025-12-31)
for a 20-ticker basket of large-cap US equities.

One row per ticker per trading day in `daily_bars`: `open, high, low, close`
are as-traded prices (back-adjusted for splits by yfinance), `adj_close` also
folds in dividends (`yfinance` called with `auto_adjust=False`). Nothing is
adjusted in this codebase; downstream phases pick which price to use.

Cleaning drops unusable rows (NaN, duplicate date, non-positive price, negative
volume) and flags but keeps merely-suspicious rows (OHLC internal
inconsistency). Gaps longer than `max_gap_days` between consecutive trading
rows are reported, never filled.

## Phase 2: features

One row per ticker per trading day in `features`, keyed `(ticker, date)`,
joined to `daily_bars` on the same key:

| column | definition |
|---|---|
| `mom_21`, `mom_63`, `mom_126`, `mom_252` | trailing simple return, `close_t / close_{t-w} - 1`, over each window in `config.toml` `momentum_windows_days` |
| `volatility` | rolling std of daily returns over `mean_reversion_window_days` |
| `mean_rev_z` | `(close_t - rolling_mean) / rolling_std`, same window; undefined (NULL) if the window std is 0 |
| `liquidity` | rolling mean of `close * volume` (dollar volume), same window |

No-lookahead: every rolling window ends at day t (`center=False`) and every
lagged term reads a strictly past row. `test_features.py` and `features.py
verify` both check this by truncating future rows and asserting past values do
not move.

Warm-up: the first rows of each ticker cannot fill a full window and are stored
as NULL (not 0, not backfilled). First live date per feature, identical across
all 20 tickers (shared NYSE calendar from 2007-01-03):

| feature | first live date |
|---|---|
| `mean_rev_z`, `liquidity` | 2007-01-31 |
| `volatility` | 2007-02-01 |
| `mom_21` | 2007-02-02 |
| `mom_63` | 2007-04-04 |
| `mom_126` | 2007-07-05 |
| `mom_252` | 2008-01-03 |

## Config

`config.toml` holds the date range, the basket, `max_gap_days`, and the Phase 2
feature windows. No values are hardcoded in `src/`.

## Phase 3: baseline model + walk-forward

Pooled ridge regression on the 7 features predicting the 5-trading-day forward
return, scored out-of-sample over 10 expanding walk-forward folds (test years
2016 to 2025) against five pre-registered gates in `config.toml [phase3]`.
Verdict at k=1: FAIL, "no validated edge" (mean IC 0.0062, Newey-West t 0.53).
All six integrity tests pass. Predictions are stored in `predictions` and the
full result is in `reports/phase3_report.md`.

## Undo

Delete `data/research_store.db` and rebuild:
`python src/research_store.py build` then `python src/features.py build`.
`research_store` writes are `INSERT OR REPLACE`; `features build` drops and
recreates the `features` table each run, so a change to
`momentum_windows_days` never leaves stale columns.

## Tests

```
python tests/test_research_store.py     # or: python -m pytest tests
python tests/test_features.py
python tests/test_phase3.py
```

No framework required; each file has a plain `assert`-based runner. `pytest`
also collects them (`src/` is put on the path by a shim at the top of each
test file).
