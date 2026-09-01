# Graph Report - quantico  (2026-09-01)

## Corpus Check
- 3 files · ~2,001 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 46 nodes · 72 edges · 8 communities
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 2 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ec10b434`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Pipeline|Data Pipeline]]
- [[_COMMUNITY_Data Cleaning|Data Cleaning]]
- [[_COMMUNITY_Database Query|Database Query]]
- [[_COMMUNITY_Test Suite|Test Suite]]
- [[_COMMUNITY_Test Framework|Test Framework]]
- [[_COMMUNITY_Data Fetch|Data Fetch]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]

## God Nodes (most connected - your core abstractions)
1. `build()` - 9 edges
2. `_frame()` - 9 edges
3. `Phase 1: research store` - 7 edges
4. `query()` - 5 edges
5. `verify()` - 5 edges
6. `main()` - 5 edges
7. `CleanReport` - 4 edges
8. `basket()` - 4 edges
9. `pull()` - 4 edges
10. `clean()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `test_query_respects_date_bounds()` --calls--> `Path`  [INFERRED]
  pipeline/test_research_store.py →   _Bridges community 6 → community 4_
- `load_config()` --references--> `Path`  [EXTRACTED]
  pipeline/research_store.py →   _Bridges community 2 → community 6_
- `verify()` --calls--> `Path`  [EXTRACTED]
  pipeline/research_store.py →   _Bridges community 6 → community 7_
- `build()` --calls--> `basket()`  [EXTRACTED]
  pipeline/research_store.py → pipeline/research_store.py  _Bridges community 7 → community 1_
- `build()` --calls--> `init_db()`  [EXTRACTED]
  pipeline/research_store.py → pipeline/research_store.py  _Bridges community 6 → community 1_

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **research store phases** — pipeline_research_store, pipeline_verify, pipeline_build, pipeline_query [EXTRACTED 0.75]

## Communities (8 total, 0 thin omitted)

### Community 0 - "Data Pipeline"
Cohesion: 0.40
Nodes (3): pandas==3.0.5, Research store: pull, clean, and store daily OHLCV for the Quantico basket.  Pha, yfinance==1.7.0

### Community 1 - "Data Cleaning"
Cohesion: 0.32
Nodes (7): build(), clean(), CleanReport, _plus_one_day(), pull(), Fetch daily OHLCV for one ticker. Raises if the response is empty.      `end` is, Drop unusable rows, flag suspicious ones, detect gaps. Never fills.

### Community 2 - "Database Query"
Cohesion: 0.50
Nodes (4): load_config(), main(), query(), Return rows for ticker in [start, end] inclusive, ordered by date ascending.

### Community 3 - "Test Suite"
Cohesion: 0.33
Nodes (4): Offline checks for research_store.clean() and the sqlite round-trip.  No network, test_detects_gap_but_does_not_fill(), test_drops_duplicate_dates_keep_last(), test_drops_nan_rows()

### Community 4 - "Test Framework"
Cohesion: 0.33
Nodes (6): _frame(), test_drops_nonpositive_prices(), test_flags_ohlc_inconsistency_without_dropping(), test_query_respects_date_bounds(), test_sorts_ascending(), test_sqlite_roundtrip_is_ordered_and_idempotent()

### Community 5 - "Data Fetch"
Cohesion: 0.22
Nodes (8): Config, Phase 1: research store, pipeline/, Run, Setup, Tests, Undo, What it stores

### Community 6 - "Community 6"
Cohesion: 0.40
Nodes (5): Connection, Path, init_db(), Idempotent write: re-running build replaces existing rows, never dupes., store()

### Community 7 - "Community 7"
Cohesion: 0.67
Nodes (3): basket(), Flatten config basket into ordered (ticker, sector) pairs., verify()

## Knowledge Gaps
- **8 isolated node(s):** `Setup`, `Run`, `What it stores`, `Config`, `Undo` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `test_sqlite_roundtrip_is_ordered_and_idempotent()` connect `Test Framework` to `Test Suite`, `Community 6`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `test_query_respects_date_bounds()` connect `Test Framework` to `Test Suite`, `Community 6`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `init_db()` connect `Community 6` to `Data Pipeline`, `Data Cleaning`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Path` (e.g. with `test_query_respects_date_bounds()` and `test_sqlite_roundtrip_is_ordered_and_idempotent()`) actually correct?**
  _`Path` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Setup`, `Run`, `What it stores` to the rest of the system?**
  _15 weakly-connected nodes found - possible documentation gaps or missing edges._