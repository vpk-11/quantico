"""Feature engineering: momentum, mean reversion, volatility, liquidity.

Phase 2 scope only. Reads `daily_bars` from the Phase 1 research store and
writes one `features` row per (ticker, trading day) into the same SQLite file.
No model, no feature selection, no walk-forward validation.

No-lookahead rule (the load-bearing property of this phase): every value in a
`features` row for day t is computed only from `daily_bars` rows for day t and
earlier. Every rolling window ends at t (`center=False`); every lagged term
reads a strictly past row. `verify` and `test_features.py` both check this
explicitly by truncating future rows and asserting past feature values do not
move.

Price basis: `daily_bars.close` is already split-adjusted in this store (raw
close and volume are back-adjusted for splits by yfinance 1.7.0 even with
auto_adjust=False), so trailing returns and z-scores cross a split cleanly with
no fake jump. `adj_close` additionally folds in dividends and is left for a
later phase if a total-return framing is wanted; Phase 2 uses `close`.

Warm-up: the first rows of each ticker cannot fill a full window. Those cells
are stored as NULL (not 0, not backfilled). Each feature's first live date is
reported by `build` and pinned in .claude/context/architecture.md.

Usage (run from pipeline/):
    python src/features.py build                   # compute + store for the whole basket
    python src/features.py verify                  # coverage + no-lookahead + spot checks
    python src/features.py show AAPL 2020-01-01 2020-03-31
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import research_store as rs

_ROOT = Path(__file__).resolve().parents[1]  # pipeline/ (holds config.toml + data/)

_FIXED_COLS = ["mean_rev_z", "volatility", "liquidity"]


def feature_columns(momentum_windows: list[int]) -> list[str]:
    """Ordered feature column names: one momentum column per window, then the
    three single-window features. Column set is derived from config, so adding
    or changing a momentum window needs no code edit here."""
    return [f"mom_{w}" for w in momentum_windows] + _FIXED_COLS


def compute_features(
    bars: pd.DataFrame, momentum_windows: list[int], mr_window: int
) -> pd.DataFrame:
    """Pure function. `bars`: DatetimeIndex ascending, columns `close`, `volume`.

    Returns a DataFrame on the same index with the columns from
    `feature_columns(momentum_windows)`, NaN in each feature's warm-up span.
    Nothing here reads a row later than the row being computed.
    """
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)

    # Momentum: trailing simple return, close_t / close_{t-w} - 1. shift(w) reads
    # the row w trading days back, which is <= t.
    for w in momentum_windows:
        out[f"mom_{w}"] = close / close.shift(w) - 1.0

    daily_ret = close.pct_change()

    # Volatility: rolling std of daily returns. Window ends at t.
    out["volatility"] = daily_ret.rolling(mr_window, center=False).std()

    # Mean reversion: z-score of close against its own trailing mean/std, same
    # window. A zero-std window (flat price) leaves the z-score undefined -> NaN,
    # not +/-inf.
    roll = close.rolling(mr_window, center=False)
    roll_std = roll.std()
    out["mean_rev_z"] = (close - roll.mean()) / roll_std.replace(0.0, np.nan)

    # Liquidity: trailing average dollar volume (close * volume), same window.
    out["liquidity"] = (close * bars["volume"]).rolling(mr_window, center=False).mean()

    return out[feature_columns(momentum_windows)]


def load_bars(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """All stored bars for one ticker, ascending, date as the index."""
    return pd.read_sql_query(
        "SELECT date, close, volume FROM daily_bars WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,),
        parse_dates=["date"],
        index_col="date",
    )


def init_features_table(conn: sqlite3.Connection, momentum_windows: list[int]) -> None:
    """(Re)create `features` from scratch. Dropped first so a change to the
    configured momentum windows never leaves stale columns behind; a full
    recompute of the basket is well under a second."""
    cols = ",\n    ".join(f"{c} REAL" for c in feature_columns(momentum_windows))
    conn.executescript(
        "DROP TABLE IF EXISTS features;\n"
        "CREATE TABLE features (\n"
        "    ticker TEXT NOT NULL,\n"
        "    date   TEXT NOT NULL,   -- ISO 'YYYY-MM-DD'\n"
        f"    {cols},\n"
        "    PRIMARY KEY (ticker, date)\n"
        ");"
    )
    conn.commit()


def store_features(conn: sqlite3.Connection, ticker: str, feats: pd.DataFrame) -> int:
    """Write every feature row for one ticker. NaN -> SQL NULL."""
    cols = list(feats.columns)
    rows = [
        (
            ticker,
            idx.strftime("%Y-%m-%d"),
            *(None if pd.isna(v) else float(v) for v in values),
        )
        for idx, values in zip(feats.index, feats.to_numpy())
    ]
    placeholders = ", ".join(["?"] * (2 + len(cols)))
    conn.executemany(
        f"INSERT OR REPLACE INTO features (ticker, date, {', '.join(cols)}) "
        f"VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)


def first_live_dates(feats: pd.DataFrame) -> dict[str, str | None]:
    """First date each feature column has a non-null value (its warm-up ends)."""
    out: dict[str, str | None] = {}
    for col in feats.columns:
        valid = feats[col].first_valid_index()
        out[col] = valid.strftime("%Y-%m-%d") if valid is not None else None
    return out


# --- entrypoints -----------------------------------------------------------


def build(cfg: dict) -> None:
    mom_windows = cfg["features"]["momentum_windows_days"]
    mr_window = int(cfg["features"]["mean_reversion_window_days"])
    db_path = _ROOT / cfg["data"]["db_path"]
    if not Path(db_path).exists():
        raise RuntimeError(f"no research store at {db_path}; run research_store.py build first")

    conn = sqlite3.connect(db_path)
    last_feats = None
    try:
        init_features_table(conn, mom_windows)
        for ticker, sector in rs.basket(cfg):
            bars = load_bars(conn, ticker)
            last_feats = compute_features(bars, mom_windows, mr_window)
            written = store_features(conn, ticker, last_feats)
            print(f"[{sector}] {ticker:<6} {written:>5} feature rows  {bars.index[0].date()}..{bars.index[-1].date()}")
    finally:
        conn.close()

    # Warm-up is calendar-identical across tickers (all share the 4780-day NYSE
    # calendar from 2007-01-03), so one ticker's first-live dates stand for all.
    print("\n--- first live date per feature (warm-up end) ---")
    for col, d in first_live_dates(last_feats).items():
        print(f"  {col:<12} {d}")
    print(f"\nDB: {db_path}")


def _independent_recompute(
    bars: pd.DataFrame, mom_window: int, mr_window: int, date: str
) -> tuple[float, float]:
    """Recompute (momentum, volatility) for one date WITHOUT compute_features,
    using plain slicing, as an independent formula check for `verify`."""
    close = bars.loc[:date, "close"].to_numpy(dtype=float)
    ret = np.diff(close) / close[:-1]
    momentum = close[-1] / close[-1 - mom_window] - 1.0
    volatility = ret[-mr_window:].std(ddof=1)
    return momentum, volatility


def verify(cfg: dict) -> bool:
    mom_windows = cfg["features"]["momentum_windows_days"]
    mr_window = int(cfg["features"]["mean_reversion_window_days"])
    cols = feature_columns(mom_windows)
    db_path = _ROOT / cfg["data"]["db_path"]
    if not Path(db_path).exists():
        print(f"no DB at {db_path}; run `build` first")
        return False

    conn = sqlite3.connect(db_path)
    ok = True
    try:
        # 1. coverage: one feature row per bar, per ticker.
        print("--- coverage (feature rows vs bars) ---")
        for ticker, _ in rs.basket(cfg):
            nb = conn.execute("SELECT COUNT(*) FROM daily_bars WHERE ticker = ?", (ticker,)).fetchone()[0]
            nf = conn.execute("SELECT COUNT(*) FROM features WHERE ticker = ?", (ticker,)).fetchone()[0]
            aligned = conn.execute(
                "SELECT COUNT(*) FROM features f JOIN daily_bars b USING (ticker, date) WHERE f.ticker = ?",
                (ticker,),
            ).fetchone()[0]
            flag = "" if nb == nf == aligned else "  <-- CHECK"
            if flag:
                ok = False
            print(f"  {ticker:<6} bars={nb} features={nf} joined={aligned}{flag}")

        # 2. NULLs only in the warm-up prefix: for each feature column, every
        #    NULL must sit strictly before that column's first non-NULL date.
        print("\n--- NULLs confined to warm-up ---")
        for col in cols:
            bad = conn.execute(
                f"SELECT COUNT(*) FROM features WHERE {col} IS NULL AND date >= "
                f"(SELECT MIN(date) FROM features WHERE {col} IS NOT NULL AND ticker = features.ticker)"
            ).fetchone()[0]
            first = conn.execute(f"SELECT MIN(date) FROM features WHERE {col} IS NOT NULL").fetchone()[0]
            flag = "" if bad == 0 else f"  <-- {bad} late NULLs"
            if bad:
                ok = False
            print(f"  {col:<12} first_live={first}  late_nulls={bad}{flag}")

        # 3. no-lookahead: recompute AAPL features from bars truncated at a fixed
        #    date; every stored value at that date must be unchanged.
        probe_date = "2015-06-15"
        bars = load_bars(conn, "AAPL")
        truncated = bars.loc[:probe_date]
        recomputed = compute_features(truncated, mom_windows, mr_window).loc[probe_date]
        stored = conn.execute(
            f"SELECT {', '.join(cols)} FROM features WHERE ticker = 'AAPL' AND date = ?",
            (probe_date,),
        ).fetchone()
        print(f"\n--- no-lookahead (AAPL {probe_date}, recompute from truncated history) ---")
        drift = False
        for col, s, r in zip(cols, stored, recomputed.to_numpy()):
            match = (s is None and pd.isna(r)) or (s is not None and abs(s - r) <= 1e-9 * max(1.0, abs(s)))
            if not match:
                drift = True
            print(f"  {col:<12} stored={s}  truncated_recompute={r}  {'ok' if match else 'DRIFT'}")
        if drift:
            ok = False

        # 4. independent formula spot check (no compute_features involved).
        print("\n--- spot checks vs plain-slicing recompute ---")
        long_mom = mom_windows[-1]
        for ticker, date in [("MSFT", "2018-03-01"), ("XOM", "2012-09-20"), ("JPM", "2021-11-10")]:
            expected = _independent_recompute(load_bars(conn, ticker), long_mom, mr_window, date)
            stored_row = conn.execute(
                f"SELECT mom_{long_mom}, volatility FROM features WHERE ticker = ? AND date = ?",
                (ticker, date),
            ).fetchone()
            for name, ev, gv in zip((f"mom_{long_mom}", "volatility"), expected, stored_row):
                verdict = "ok" if abs(ev - gv) <= 1e-9 * max(1.0, abs(ev)) else "MISMATCH"
                if verdict != "ok":
                    ok = False
                print(f"  {ticker:<6} {date}  {name:<10} stored={gv:.8f}  independent={ev:.8f}  {verdict}")

        # 5. join returns aligned, ordered rows.
        print("\n--- join check (AAPL 2020-02-03..2020-02-14) ---")
        rows = conn.execute(
            "SELECT b.date, b.close, f.mom_21 FROM daily_bars b JOIN features f USING (ticker, date) "
            "WHERE b.ticker = 'AAPL' AND b.date BETWEEN ? AND ? ORDER BY b.date ASC",
            ("2020-02-03", "2020-02-14"),
        ).fetchall()
        dates = [r[0] for r in rows]
        ordered = dates == sorted(dates)
        print(f"  {len(rows)} rows, ascending={ordered}, first={dates[0] if dates else None}")
        if not (rows and ordered):
            ok = False
    finally:
        conn.close()
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico feature engineering (Phase 2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="compute + store features for the whole basket")
    sub.add_parser("verify", help="coverage + no-lookahead + spot-check report")
    s = sub.add_parser("show", help="print stored feature rows for a ticker + date range")
    s.add_argument("ticker")
    s.add_argument("start")
    s.add_argument("end")
    args = parser.parse_args(argv)

    cfg = rs.load_config()
    if args.cmd == "build":
        build(cfg)
        return 0
    if args.cmd == "verify":
        return 0 if verify(cfg) else 1
    if args.cmd == "show":
        cols = feature_columns(cfg["features"]["momentum_windows_days"])
        conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
        try:
            rows = conn.execute(
                f"SELECT date, {', '.join(cols)} FROM features "
                "WHERE ticker = ? AND date BETWEEN ? AND ? ORDER BY date ASC",
                (args.ticker.upper(), args.start, args.end),
            ).fetchall()
        finally:
            conn.close()
        print(f"date  {'  '.join(cols)}")
        for r in rows[:20]:
            print(r)
        if len(rows) > 20:
            print("...")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
