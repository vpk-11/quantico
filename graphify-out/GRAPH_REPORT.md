# Graph Report - .  (2026-08-30)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 41 nodes · 68 edges · 6 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4dbfafc2`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Pipeline|Data Pipeline]]
- [[_COMMUNITY_Data Cleaning|Data Cleaning]]
- [[_COMMUNITY_Database Query|Database Query]]
- [[_COMMUNITY_Test Suite|Test Suite]]
- [[_COMMUNITY_Test Framework|Test Framework]]
- [[_COMMUNITY_Data Fetch|Data Fetch]]

## God Nodes (most connected - your core abstractions)
1. `build()` - 9 edges
2. `_frame()` - 9 edges
3. `query()` - 5 edges
4. `verify()` - 5 edges
5. `main()` - 5 edges
6. `CleanReport` - 4 edges
7. `basket()` - 4 edges
8. `pull()` - 4 edges
9. `clean()` - 4 edges
10. `init_db()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_query_respects_date_bounds()` --calls--> `Path`  [INFERRED]
  pipeline/test_research_store.py →   _Bridges community 2 → community 4_
- `verify()` --calls--> `basket()`  [EXTRACTED]
  pipeline/research_store.py → pipeline/research_store.py  _Bridges community 1 → community 2_
- `build()` --calls--> `pull()`  [EXTRACTED]
  pipeline/research_store.py → pipeline/research_store.py  _Bridges community 5 → community 1_
- `test_detects_gap_but_does_not_fill()` --calls--> `_frame()`  [EXTRACTED]
  pipeline/test_research_store.py → pipeline/test_research_store.py  _Bridges community 4 → community 3_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **research store phases** — pipeline_research_store, pipeline_verify, pipeline_build, pipeline_query [EXTRACTED 0.75]

## Communities (6 total, 0 thin omitted)

### Community 0 - "Data Pipeline"
Cohesion: 0.22
Nodes (7): data/research_store.db, pandas==3.0.5, build, query, Research store: pull, clean, and store daily OHLCV for the ARL basket.  Phase 1, verify, yfinance==1.7.0

### Community 1 - "Data Cleaning"
Cohesion: 0.28
Nodes (8): basket(), build(), clean(), CleanReport, Flatten config basket into ordered (ticker, sector) pairs., Drop unusable rows, flag suspicious ones, detect gaps. Never fills., Idempotent write: re-running build replaces existing rows, never dupes., store()

### Community 2 - "Database Query"
Cohesion: 0.36
Nodes (8): Connection, Path, init_db(), load_config(), main(), query(), Return rows for ticker in [start, end] inclusive, ordered by date ascending., verify()

### Community 3 - "Test Suite"
Cohesion: 0.33
Nodes (4): Offline checks for research_store.clean() and the sqlite round-trip.  No network, test_detects_gap_but_does_not_fill(), test_drops_duplicate_dates_keep_last(), test_drops_nan_rows()

### Community 4 - "Test Framework"
Cohesion: 0.33
Nodes (6): _frame(), test_drops_nonpositive_prices(), test_flags_ohlc_inconsistency_without_dropping(), test_query_respects_date_bounds(), test_sorts_ascending(), test_sqlite_roundtrip_is_ordered_and_idempotent()

### Community 5 - "Data Fetch"
Cohesion: 0.67
Nodes (3): _plus_one_day(), pull(), Fetch daily OHLCV for one ticker. Raises if the response is empty.      `end` is

## Knowledge Gaps
- **6 isolated node(s):** `verify`, `build`, `query`, `data/research_store.db`, `yfinance==1.7.0` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `test_sqlite_roundtrip_is_ordered_and_idempotent()` connect `Test Framework` to `Database Query`, `Test Suite`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `test_query_respects_date_bounds()` connect `Test Framework` to `Database Query`, `Test Suite`?**
  _High betweenness centrality (0.186) - this node is a cross-community bridge._
- **Why does `init_db()` connect `Database Query` to `Data Pipeline`, `Data Cleaning`?**
  _High betweenness centrality (0.177) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Path` (e.g. with `test_query_respects_date_bounds()` and `test_sqlite_roundtrip_is_ordered_and_idempotent()`) actually correct?**
  _`Path` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Research store: pull, clean, and store daily OHLCV for the ARL basket.  Phase 1`, `Flatten config basket into ordered (ticker, sector) pairs.`, `Fetch daily OHLCV for one ticker. Raises if the response is empty.      `end` is` to the rest of the system?**
  _13 weakly-connected nodes found - possible documentation gaps or missing edges._