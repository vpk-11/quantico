# pipeline/

Python side of Quantico: the research-to-paper-trading pipeline. `simulator/` (C++)
is the separate execution-simulator component.

Folder name is a working placeholder, same as the project name.

## Phase 1: research store

Pulls, cleans, and stores 19 years of daily OHLCV (2007-01-01 to 2025-12-31)
for a 20-ticker basket of large-cap US equities into a local SQLite file.
Raw-data storage only. No features, no model.

### Setup

```
conda activate quantico            # env: python 3.12, created for this project
pip install -r requirements.txt
```

### Run

```
python research_store.py build                         # pull + clean + store the basket
python research_store.py verify                         # coverage + spot-check report
python research_store.py query AAPL 2020-01-01 2020-03-31
```

The store lands at `data/research_store.db` (git-ignored).

### What it stores

One row per ticker per trading day in `daily_bars`:
`open, high, low, close` are raw as-traded prices; `adj_close` is Yahoo's
split/dividend-adjusted close (`yfinance` called with `auto_adjust=False`).
Downstream phases pick which to use; nothing is adjusted here.

Cleaning drops rows that are unusable (NaN, duplicate date, non-positive
price, negative volume) and flags but keeps rows that are merely suspicious
(OHLC internal inconsistency). Gaps longer than `max_gap_days` between
consecutive trading rows are reported, never filled.

### Config

`config.toml` holds the date range, the basket, and `max_gap_days`. No values
are hardcoded in `research_store.py`.

### Undo

Delete `data/research_store.db`. Re-running `build` rebuilds it from scratch
(writes are `INSERT OR REPLACE`, so a partial re-run is safe too).

### Tests

```
python test_research_store.py       # offline: clean() logic + sqlite round-trip
```
