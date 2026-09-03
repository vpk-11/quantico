"""Offline checks for research_store.clean() and the sqlite round-trip.

No network. Run: python tests/test_research_store.py  (or: pytest)
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import research_store as rs  # noqa: E402


def _frame(dates, **cols):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    base = dict(open=1.0, high=1.0, low=1.0, close=1.0, adj_close=1.0, volume=100)
    data = {k: cols.get(k, [base[k]] * len(idx)) for k in base}
    return pd.DataFrame(data, index=idx)


def test_drops_nan_rows():
    df = _frame(["2020-01-02", "2020-01-03"], close=[10.0, np.nan])
    out, rep = rs.clean(df, "T", max_gap_days=5)
    assert rep.dropped_nan == 1
    assert len(out) == 1


def test_drops_duplicate_dates_keep_last():
    df = _frame(["2020-01-02", "2020-01-02"], close=[10.0, 11.0])
    out, rep = rs.clean(df, "T", max_gap_days=5)
    assert rep.dropped_dupe_dates == 1
    assert out["close"].iloc[0] == 11.0


def test_drops_nonpositive_prices():
    df = _frame(["2020-01-02", "2020-01-03", "2020-01-06"], close=[10.0, 0.0, -5.0])
    out, rep = rs.clean(df, "T", max_gap_days=5)
    assert rep.dropped_nonpositive_price == 2
    assert len(out) == 1


def test_flags_ohlc_inconsistency_without_dropping():
    # low above high on row 2 -> flagged, kept
    df = _frame(
        ["2020-01-02", "2020-01-03"],
        low=[1.0, 9.0], high=[2.0, 3.0],
        open=[1.5, 2.5], close=[1.8, 2.8],
    )
    out, rep = rs.clean(df, "T", max_gap_days=5)
    assert rep.ohlc_inconsistencies == 1
    assert len(out) == 2


def test_detects_gap_but_does_not_fill():
    df = _frame(["2020-01-02", "2020-01-20"], close=[10.0, 11.0])
    out, rep = rs.clean(df, "T", max_gap_days=5)
    assert len(out) == 2                       # nothing fabricated
    assert rep.gaps == [("2020-01-02", "2020-01-20", 18)]


def test_sorts_ascending():
    df = _frame(["2020-01-06", "2020-01-02", "2020-01-03"])
    out, _ = rs.clean(df, "T", max_gap_days=5)
    assert list(out.index) == sorted(out.index)


def test_sqlite_roundtrip_is_ordered_and_idempotent():
    df = _frame(["2020-01-03", "2020-01-02", "2020-01-06"], close=[3.0, 2.0, 6.0])
    clean_df, _ = rs.clean(df, "AAPL", max_gap_days=5)
    with tempfile.TemporaryDirectory() as d:
        conn = rs.init_db(Path(d) / "t.db")
        rs.store(conn, "AAPL", clean_df)
        rs.store(conn, "AAPL", clean_df)  # second write must not duplicate
        rows = rs.query(conn, "AAPL", "2020-01-01", "2020-01-31")
        conn.close()
    assert len(rows) == 3
    assert [r["date"] for r in rows] == ["2020-01-02", "2020-01-03", "2020-01-06"]
    assert rows[0]["close"] == 2.0


def test_query_respects_date_bounds():
    df = _frame(["2020-01-02", "2020-02-02", "2020-03-02"])
    clean_df, _ = rs.clean(df, "X", max_gap_days=1000)
    with tempfile.TemporaryDirectory() as d:
        conn = rs.init_db(Path(d) / "t.db")
        rs.store(conn, "X", clean_df)
        rows = rs.query(conn, "X", "2020-01-15", "2020-02-15")
        conn.close()
    assert [r["date"] for r in rows] == ["2020-02-02"]


def test_basket_flattens_to_20():
    cfg = rs.load_config()
    pairs = rs.basket(cfg)
    tickers = [t for t, _ in pairs]
    assert len(tickers) == 20
    assert len(set(tickers)) == 20
    assert tickers[0] == "AAPL"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
