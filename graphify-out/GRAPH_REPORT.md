# Graph Report - quantico  (2026-09-03)

## Corpus Check
- 5 files · ~4,498 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 80 nodes · 139 edges · 13 communities (10 shown, 3 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7965ba18`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Cleaning|Data Cleaning]]
- [[_COMMUNITY_Database Query|Database Query]]
- [[_COMMUNITY_Test Suite|Test Suite]]
- [[_COMMUNITY_Test Framework|Test Framework]]
- [[_COMMUNITY_Data Fetch|Data Fetch]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]

## God Nodes (most connected - your core abstractions)
1. `_bars()` - 10 edges
2. `pipeline/` - 9 edges
3. `build()` - 9 edges
4. `_frame()` - 9 edges
5. `build()` - 8 edges
6. `verify()` - 7 edges
7. `Phase 1: research store` - 6 edges
8. `feature_columns()` - 6 edges
9. `compute_features()` - 6 edges
10. `load_bars()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `build()` --calls--> `Path`  [INFERRED]
  pipeline/src/features.py →   _Bridges community 3 → community 6_
- `verify()` --calls--> `Path`  [INFERRED]
  pipeline/src/features.py →   _Bridges community 4 → community 6_
- `test_sqlite_roundtrip_preserves_nulls_and_order()` --calls--> `Path`  [INFERRED]
  pipeline/tests/test_features.py →   _Bridges community 6 → community 1_
- `test_query_respects_date_bounds()` --calls--> `Path`  [INFERRED]
  pipeline/tests/test_research_store.py →   _Bridges community 6 → community 2_
- `init_features_table()` --calls--> `feature_columns()`  [EXTRACTED]
  pipeline/src/features.py → pipeline/src/features.py  _Bridges community 4 → community 3_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **research store phases** — pipeline_research_store, pipeline_verify, pipeline_build, pipeline_query [EXTRACTED 0.75]

## Communities (13 total, 3 thin omitted)

### Community 1 - "Data Cleaning"
Cohesion: 0.27
Nodes (11): _bars(), Offline checks for features.compute_features and the sqlite round-trip.  No netw, test_liquidity_is_trailing_mean_dollar_volume(), test_momentum_is_trailing_return(), test_no_lookahead_truncation_invariance(), test_rolling_windows_not_centered(), test_sqlite_roundtrip_preserves_nulls_and_order(), test_volatility_matches_manual_std() (+3 more)

### Community 2 - "Database Query"
Cohesion: 0.29
Nodes (10): _frame(), Offline checks for research_store.clean() and the sqlite round-trip.  No network, test_detects_gap_but_does_not_fill(), test_drops_duplicate_dates_keep_last(), test_drops_nan_rows(), test_drops_nonpositive_prices(), test_flags_ohlc_inconsistency_without_dropping(), test_query_respects_date_bounds() (+2 more)

### Community 3 - "Test Suite"
Cohesion: 0.32
Nodes (8): Connection, build(), init_features_table(), load_bars(), (Re)create `features` from scratch. Dropped first so a change to the     configu, Write every feature row for one ticker. NaN -> SQL NULL., All stored bars for one ticker, ascending, date as the index., store_features()

### Community 4 - "Test Framework"
Cohesion: 0.43
Nodes (7): compute_features(), feature_columns(), main(), Feature engineering: momentum, mean reversion, volatility, liquidity.  Phase 2 s, Ordered feature column names: one momentum column per window, then the     three, Pure function. `bars`: DatetimeIndex ascending, columns `close`, `volume`., verify()

### Community 5 - "Data Fetch"
Cohesion: 0.31
Nodes (9): Config, Layout, Phase 1: research store, Phase 2: features, pipeline/, Run, Setup, Tests (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.47
Nodes (6): Path, load_config(), main(), query(), Return rows for ticker in [start, end] inclusive, ordered by date ascending., verify()

### Community 7 - "Community 7"
Cohesion: 0.40
Nodes (5): DataFrame, first_live_dates(), _independent_recompute(), First date each feature column has a non-null value (its warm-up ends)., Recompute (momentum, volatility) for one date WITHOUT compute_features,     usin

### Community 8 - "Community 8"
Cohesion: 0.50
Nodes (4): _plus_one_day(), pull(), Research store: pull, clean, and store daily OHLCV for the Quantico basket.  Pha, Fetch daily OHLCV for one ticker. Raises if the response is empty.      `end` is

### Community 9 - "Community 9"
Cohesion: 0.60
Nodes (4): build(), clean(), CleanReport, Drop unusable rows, flag suspicious ones, detect gaps. Never fills.

### Community 10 - "Community 10"
Cohesion: 0.50
Nodes (4): Connection, init_db(), Idempotent write: re-running build replaces existing rows, never dupes., store()

## Knowledge Gaps
- **4 isolated node(s):** `Layout`, `Phase 2: features`, `yfinance==1.7.0`, `pandas==3.0.5`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `test_sqlite_roundtrip_preserves_nulls_and_order()` connect `Data Cleaning` to `Community 6`?**
  _High betweenness centrality (0.214) - this node is a cross-community bridge._
- **Why does `build()` connect `Test Suite` to `Test Framework`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `verify()` connect `Test Framework` to `Test Suite`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.128) - this node is a cross-community bridge._
- **What connects `Layout`, `Phase 2: features`, `Feature engineering: momentum, mean reversion, volatility, liquidity.  Phase 2 s` to the rest of the system?**
  _20 weakly-connected nodes found - possible documentation gaps or missing edges._