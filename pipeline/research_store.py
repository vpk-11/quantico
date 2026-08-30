"""Research store: pull, clean, and store daily OHLCV for the ARL basket.

Phase 1 scope only. Raw-data storage. No features, no model.

Pipeline stages, each a plain function with a defined input/output:
    load_config  -> dict
    basket       -> [(ticker, sector)]
    pull         -> DataFrame (raw OHLCV + adj_close, Date index)
    clean        -> (DataFrame, report)   # never fabricates or forward-fills
    init_db      -> sqlite3.Connection
    store        -> rows written
    query        -> [dict] ordered by date

Adjusted-close handling: yfinance is called with auto_adjust=False, so the
Open/High/Low/Close columns are the raw (unadjusted) prices as traded, and
'Adj Close' is Yahoo's split- and dividend-adjusted close. Both are stored.
Downstream phases choose which to use; nothing is adjusted here.

Usage:
    python research_store.py build                  # pull + clean + store everything
    python research_store.py query AAPL 2020-01-01 2020-03-31
    python research_store.py verify                 # coverage + spot-check report
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).resolve().parent

# yfinance (auto_adjust=False, multi_level_index=False) column -> our column
_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
_PRICE_COLS = ("open", "high", "low", "close", "adj_close")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    ticker     TEXT    NOT NULL,
    date       TEXT    NOT NULL,   -- ISO 'YYYY-MM-DD'
    open       REAL    NOT NULL,   -- raw (unadjusted) prices as traded
    high       REAL    NOT NULL,
    low        REAL    NOT NULL,
    close      REAL    NOT NULL,
    adj_close  REAL    NOT NULL,   -- Yahoo split/dividend-adjusted close
    volume     INTEGER NOT NULL,
    PRIMARY KEY (ticker, date)
);
"""


@dataclass
class CleanReport:
    ticker: str
    rows_in: int = 0
    rows_out: int = 0
    dropped_nan: int = 0
    dropped_dupe_dates: int = 0
    dropped_nonpositive_price: int = 0
    dropped_negative_volume: int = 0
    ohlc_inconsistencies: int = 0            # low>high or close outside [low,high]; flagged, not dropped
    first_date: str | None = None
    last_date: str | None = None
    gaps: list[tuple[str, str, int]] = field(default_factory=list)  # (prev, next, calendar_days)

    def summary(self) -> str:
        drops = (
            self.dropped_nan
            + self.dropped_dupe_dates
            + self.dropped_nonpositive_price
            + self.dropped_negative_volume
        )
        parts = [
            f"{self.ticker:<6} {self.rows_out:>5} rows  {self.first_date}..{self.last_date}",
            f"dropped={drops}",
        ]
        if self.ohlc_inconsistencies:
            parts.append(f"ohlc_flags={self.ohlc_inconsistencies}")
        if self.gaps:
            worst = max(g[2] for g in self.gaps)
            parts.append(f"gaps={len(self.gaps)} (worst {worst}d)")
        return "  ".join(parts)


def load_config(path: str | Path = HERE / "config.toml") -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def basket(cfg: dict) -> list[tuple[str, str]]:
    """Flatten config basket into ordered (ticker, sector) pairs."""
    out: list[tuple[str, str]] = []
    for sector, tickers in cfg["data"]["basket"].items():
        for t in tickers:
            out.append((t, sector))
    return out


def pull(ticker: str, start: str, end: str):
    """Fetch daily OHLCV for one ticker. Raises if the response is empty.

    `end` is treated as inclusive here by adding a day, since yfinance's `end`
    is exclusive.
    """
    end_exclusive = _plus_one_day(end)
    df = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        auto_adjust=False,
        actions=False,
        progress=False,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: yfinance returned no rows for {start}..{end}")
    df = df.rename(columns=_COLUMN_MAP)
    missing = [c for c in _COLUMN_MAP.values() if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing expected columns {missing}; got {list(df.columns)}")
    df.index.name = "date"
    return df[list(_COLUMN_MAP.values())]


def clean(df, ticker: str, max_gap_days: int) -> tuple["object", CleanReport]:
    """Drop unusable rows, flag suspicious ones, detect gaps. Never fills."""
    rep = CleanReport(ticker=ticker, rows_in=len(df))
    df = df.sort_index()

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    rep.dropped_dupe_dates = before - len(df)

    before = len(df)
    df = df.dropna(subset=list(_COLUMN_MAP.values()))
    rep.dropped_nan = before - len(df)

    before = len(df)
    positive = (df[list(_PRICE_COLS)] > 0).all(axis=1)
    df = df[positive]
    rep.dropped_nonpositive_price = before - len(df)

    before = len(df)
    df = df[df["volume"] >= 0]
    rep.dropped_negative_volume = before - len(df)

    # OHLC internal consistency: flag only. yfinance has known minor quirks;
    # dropping real trading days over a rounding artifact loses more than it saves.
    lo, hi = df["low"], df["high"]
    body_lo = df[["open", "close"]].min(axis=1)
    body_hi = df[["open", "close"]].max(axis=1)
    bad = (lo > hi) | (body_lo < lo - 1e-6) | (body_hi > hi + 1e-6)
    rep.ohlc_inconsistencies = int(bad.sum())

    rep.rows_out = len(df)
    if len(df):
        rep.first_date = df.index[0].strftime("%Y-%m-%d")
        rep.last_date = df.index[-1].strftime("%Y-%m-%d")
        day_gaps = df.index.to_series().diff().dt.days
        for pos, gap in enumerate(day_gaps):
            if pos == 0 or gap != gap:  # skip first (NaT)
                continue
            if gap > max_gap_days:
                prev = df.index[pos - 1].strftime("%Y-%m-%d")
                cur = df.index[pos].strftime("%Y-%m-%d")
                rep.gaps.append((prev, cur, int(gap)))
    return df, rep


def init_db(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def store(conn: sqlite3.Connection, ticker: str, df) -> int:
    """Idempotent write: re-running build replaces existing rows, never dupes."""
    rows = [
        (
            ticker,
            idx.strftime("%Y-%m-%d"),
            float(r.open),
            float(r.high),
            float(r.low),
            float(r.close),
            float(r.adj_close),
            int(round(r.volume)),
        )
        for idx, r in df.iterrows()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO daily_bars "
        "(ticker, date, open, high, low, close, adj_close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def query(conn: sqlite3.Connection, ticker: str, start: str, end: str) -> list[dict]:
    """Return rows for ticker in [start, end] inclusive, ordered by date ascending."""
    cur = conn.execute(
        "SELECT ticker, date, open, high, low, close, adj_close, volume "
        "FROM daily_bars WHERE ticker = ? AND date >= ? AND date <= ? "
        "ORDER BY date ASC",
        (ticker, start, end),
    )
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _plus_one_day(iso_date: str) -> str:
    from datetime import date, timedelta

    y, m, d = (int(x) for x in iso_date.split("-"))
    return (date(y, m, d) + timedelta(days=1)).isoformat()


# --- entrypoints -------------------------------------------------------------

def build(cfg: dict) -> list[CleanReport]:
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    max_gap = int(cfg["data"].get("max_gap_days", 5))
    db_path = HERE / cfg["data"]["db_path"]

    conn = init_db(db_path)
    reports: list[CleanReport] = []
    try:
        for ticker, sector in basket(cfg):
            raw = pull(ticker, start, end)
            cleaned, rep = clean(raw, ticker, max_gap)
            written = store(conn, ticker, cleaned)
            reports.append(rep)
            print(f"[{sector}] {rep.summary()}  (stored {written})")
    finally:
        conn.close()

    print("\n--- gap detail (unexplained runs > "
          f"{max_gap} calendar days between trading rows) ---")
    any_gap = False
    for rep in reports:
        for prev, cur, days in rep.gaps:
            any_gap = True
            print(f"  {rep.ticker:<6} {prev} -> {cur}  ({days}d)")
    if not any_gap:
        print("  none")
    print(f"\nDB: {db_path}")
    return reports


# (ticker, date, field, expected) reference points for the manual spot-check.
# Raw (unadjusted) as-traded closes. AAPL and MSFT are independently-published
# historical prices; JPM 2008-09-15 (Lehman Monday) is cross-checked against the
# full week's shape: ~41 the prior Friday, gap down to 37.00 on 2x volume, then
# the 2008-09-19 short-sale-ban spike to ~47. verify prints stored vs expected.
_SPOT_CHECKS = [
    ("AAPL", "2020-01-02", "close", 75.09),
    ("MSFT", "2010-01-04", "close", 30.95),
    ("JPM", "2008-09-15", "close", 37.00),
]


def verify(cfg: dict) -> bool:
    db_path = HERE / cfg["data"]["db_path"]
    if not Path(db_path).exists():
        print(f"no DB at {db_path}; run `build` first")
        return False
    conn = sqlite3.connect(db_path)
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    ok = True
    try:
        print("--- coverage ---")
        for ticker, sector in basket(cfg):
            row = conn.execute(
                "SELECT COUNT(*), MIN(date), MAX(date) FROM daily_bars WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            n, lo, hi = row
            flag = "" if n and lo <= "2007-12-31" and hi >= "2025-12-01" else "  <-- CHECK"
            print(f"  {ticker:<6} {n:>5} rows  {lo}..{hi}{flag}")
            if flag:
                ok = False

        print("\n--- spot checks (stored vs public reference, raw close) ---")
        for ticker, d, field_name, expected in _SPOT_CHECKS:
            got = conn.execute(
                f"SELECT {field_name} FROM daily_bars WHERE ticker = ? AND date = ?",
                (ticker, d),
            ).fetchone()
            if got is None:
                print(f"  {ticker:<6} {d}  MISSING")
                ok = False
                continue
            diff = abs(got[0] - expected)
            verdict = "ok" if diff <= max(0.5, expected * 0.02) else "MISMATCH"
            if verdict != "ok":
                ok = False
            print(f"  {ticker:<6} {d}  stored={got[0]:.2f}  expected~{expected:.2f}  diff={diff:.2f}  {verdict}")

        print("\n--- ordered-query check (AAPL 2020-01-01..2020-01-15) ---")
        rows = query(conn, "AAPL", "2020-01-01", "2020-01-15")
        dates = [r["date"] for r in rows]
        ordered = dates == sorted(dates)
        print(f"  {len(rows)} rows, ascending={ordered}, first={dates[0] if dates else None}")
        if not (rows and ordered):
            ok = False
    finally:
        conn.close()
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARL research store (Phase 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="pull + clean + store the whole basket")
    q = sub.add_parser("query", help="print stored rows for a ticker + date range")
    q.add_argument("ticker")
    q.add_argument("start")
    q.add_argument("end")
    sub.add_parser("verify", help="coverage + spot-check report")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.cmd == "build":
        build(cfg)
        return 0
    if args.cmd == "verify":
        return 0 if verify(cfg) else 1
    if args.cmd == "query":
        conn = sqlite3.connect(HERE / cfg["data"]["db_path"])
        try:
            rows = query(conn, args.ticker.upper(), args.start, args.end)
        finally:
            conn.close()
        print(f"{len(rows)} rows")
        for r in rows[:10]:
            print(r)
        if len(rows) > 10:
            print("...")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
