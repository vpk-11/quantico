"""Phase 6/7 checks on synthetic data: fill accounting, target selection, no-lookahead in
`paper.predict`, replay/forward mode tagging, bar-completeness gating (clock and missing-ticker
halt), and idempotent/catch-up processing.

No network, no research store. Run: python tests/test_phase6.py  (or: python -m pytest tests)
"""

import datetime as dt
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest as bt  # noqa: E402
import paper  # noqa: E402

TICKERS = ["A", "B", "C", "D"]
CFG = {
    "features": {"momentum_windows_days": [21, 63, 126, 252], "mean_reversion_window_days": 20},
    "phase3": {"target_horizon_days": 5, "ridge_alpha": 1.0, "log_liquidity": True},
    "phase4": {"slices": 13, "fee_bps": 1.0, "half_spread_bps": 1.0, "impact_coeff_bps": 10.0,
               "participation_cap": 0.10, "latency_slices": 1, "cost_multiplier": 1.0,
               "capital": 1_000_000, "top_n": 2},
    "phase6": {"paper_execution_strategy": "immediate", "start_date": "2020-01-01",
               "replay_end_date": "2024-06-15"},
    "data": {"basket": {"sector": TICKERS}},
}
P = {"capital": 1_000_000, "top_n": 2}


def _db(n_days=700, seed=3):
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(paper._SCHEMA)
    conn.execute("CREATE TABLE daily_bars (ticker, date, open, high, low, close, adj_close, volume)")
    days = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d")
    for t in ["A", "B", "C", "D"]:
        c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        for i, d in enumerate(days):
            conn.execute("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?)", (t, d, c[i], c[i], c[i], c[i], c[i], int(1e6 + 1e5 * rng.random())))
    return conn, list(days)


def test_positions_and_cash_from_fills():
    conn, _ = _db(5)
    conn.executemany("INSERT INTO paper_fills VALUES (?,?,?,?,?,?,?,?,?)", [
        ("d1", "A", 1, 100, 100, 5000.0, 1.0, 50.0, "replay"),
        ("d2", "A", -1, 40, 40, 2200.0, 0.5, 55.0, "replay"),
        ("d2", "B", 1, 10, 0, 0.0, 0.0, 10.0, "replay"),
    ])
    pos, cash = paper.positions(conn, 10_000.0)
    assert pos == {"A": 60}  # B never filled: no position
    assert cash == 10_000.0 - 5000.0 - 1.0 + 2200.0 - 0.5


def test_top_n_targets_long_only_equal_weight():
    pred = pd.Series({"A": 0.1, "B": 0.3, "C": -0.2, "D": 0.2})
    close = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0})
    assert bt.top_n_targets(pred, close, P) == {"B": 25000, "D": 12500}


def test_predict_ignores_future_rows():
    conn, days = _db()
    cut = days[600]
    full = paper.predict(conn, CFG, cut)
    conn.execute("DELETE FROM daily_bars WHERE date > ?", (days[620],))  # same data up to cut + a later tail
    again = paper.predict(conn, CFG, cut)
    conn.execute("DELETE FROM daily_bars WHERE date > ?", (cut,))  # truncated at cut
    trunc = paper.predict(conn, CFG, cut)
    assert len(full) == 4
    assert np.array_equal(full.sort_index().to_numpy(), again.sort_index().to_numpy())
    assert np.array_equal(full.sort_index().to_numpy(), trunc.sort_index().to_numpy())


def _live(conn, days, tickers=TICKERS, skip=()):
    """Inserts a live_bars row for each (ticker, date) pair not in `skip` (a set of (ticker, date))."""
    rng = np.random.default_rng(9)
    rows = []
    for d in days:
        for t in tickers:
            if (t, d) in skip:
                continue
            px = 100.0 + rng.normal()
            rows.append((t, d, px, px * 1.01, px * 0.99, px * 1.005, px * 1.005, 1_000_000))
    conn.executemany("INSERT INTO live_bars VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def test_mode_for_replay_then_forward_boundary():
    cfg = {"phase6": {"replay_end_date": "2026-10-12"}}
    assert paper.mode_for("2026-10-12", cfg) == "replay"
    assert paper.mode_for("2026-10-11", cfg) == "replay"
    assert paper.mode_for("2026-10-13", cfg) == "forward"


def test_eligible_dates_gated_by_et_clock():
    days = ["2026-10-14", "2026-10-15", "2026-10-16"]
    before_close = dt.datetime(2026, 10, 16, 16, 59, tzinfo=paper.ET)
    assert paper.eligible_dates(days, before_close) == ["2026-10-14", "2026-10-15"]
    at_close = dt.datetime(2026, 10, 16, 17, 0, tzinfo=paper.ET)
    assert paper.eligible_dates(days, at_close) == days


def _live_days_after(days, n=10):
    """`n` trading days right after the synthetic daily_bars history, as live_bars dates."""
    return pd.bdate_range(start=pd.Timestamp(days[-1]) + pd.Timedelta(days=1), periods=n).strftime("%Y-%m-%d").tolist()


def test_check_complete_halts_on_missing_ticker_then_resumes():
    conn, days = _db(700)
    d = _live_days_after(days, 1)[0]
    _live(conn, [d], skip={("D", d)})
    try:
        paper.check_complete(conn, CFG, d)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    _live(conn, [d], tickers=["D"])  # the missing ticker's bar arrives
    paper.check_complete(conn, CFG, d)  # no longer raises


def test_process_pending_idempotent_and_catchup_in_order():
    conn, days = _db(700)
    live_days = _live_days_after(days, 10)
    _live(conn, live_days)
    total1, new1 = paper._process_pending(conn, CFG, live_days)
    assert new1 == 10
    stored_first_pass = pd.read_sql_query("SELECT * FROM paper_nav ORDER BY date", conn)

    total2, new2 = paper._process_pending(conn, CFG, live_days)
    assert new2 == 0  # idempotent: rerunning processes 0 new days

    conn.execute("DELETE FROM paper_nav WHERE date >= ?", (live_days[-3],))  # simulate a gap
    conn.execute("DELETE FROM paper_fills WHERE date >= ?", (live_days[-3],))
    conn.execute("DELETE FROM paper_predictions WHERE date >= ?", (live_days[-3],))
    conn.execute("DELETE FROM paper_targets WHERE decided_on >= ?", (live_days[-3],))
    conn.commit()
    total3, new3 = paper._process_pending(conn, CFG, live_days)
    assert new3 == 3  # catch-up restores exactly the missing days, in order
    stored_second_pass = pd.read_sql_query("SELECT * FROM paper_nav ORDER BY date", conn)
    pd.testing.assert_frame_equal(stored_first_pass, stored_second_pass)


def test_halted_day_blocks_later_days_no_skipping():
    conn, days = _db(700)
    live_days = _live_days_after(days, 10)
    _live(conn, live_days, skip={("D", live_days[2])})  # day 2 is incomplete
    try:
        paper._process_pending(conn, CFG, live_days)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    processed = {r[0] for r in conn.execute("SELECT date FROM paper_nav")}
    assert processed == set(live_days[:2])  # days 0 and 1 processed; day 2 and later are not


def test_nav_matches_positions_and_cash():
    conn, days = _db(700)
    live_days = _live_days_after(days, 10)
    _live(conn, live_days)
    paper._process_pending(conn, CFG, live_days)
    pos, cash = paper.positions(conn, CFG["phase4"]["capital"])
    closes = pd.read_sql_query("SELECT ticker, date, close FROM live_bars WHERE date = ?", conn, params=(live_days[-1],))
    px = closes.set_index("ticker")["close"]
    expected_nav = cash + sum(q * px[t] for t, q in pos.items())
    stored_nav = conn.execute("SELECT nav FROM paper_nav WHERE date = ?", (live_days[-1],)).fetchone()[0]
    assert abs(expected_nav - stored_nav) < 1e-6


def test_mode_recorded_correctly_in_paper_nav():
    conn, days = _db(700)
    live_days = _live_days_after(days, 6)
    cfg = {**CFG, "phase6": {**CFG["phase6"], "replay_end_date": live_days[2]}}
    _live(conn, live_days)
    paper._process_pending(conn, cfg, live_days)
    modes = dict(conn.execute("SELECT date, mode FROM paper_nav").fetchall())
    for d in live_days[:3]:
        assert modes[d] == "replay"
    for d in live_days[3:]:
        assert modes[d] == "forward"


def test_zero_bars_day_is_skipped_not_halted():
    conn, days = _db(700)
    live_days = _live_days_after(days, 5)
    missing_day = live_days[2]
    present_days = [d for d in live_days if d != missing_day]
    _live(conn, present_days)  # missing_day gets no live_bars rows for any ticker at all
    total, new = paper._process_pending(conn, CFG, live_days)  # no RuntimeError
    assert new == 4  # every day except the all-missing one
    processed = {r[0] for r in conn.execute("SELECT date FROM paper_nav")}
    assert missing_day not in processed
    assert processed == set(present_days)  # later days were NOT blocked by the all-missing one


def test_check_complete_return_value():
    conn, days = _db(700)
    d = _live_days_after(days, 1)[0]
    assert paper.check_complete(conn, CFG, d) is False  # no rows at all: skip
    _live(conn, [d])
    assert paper.check_complete(conn, CFG, d) is True  # every ticker present: proceed


def test_reset_refuses_with_forward_rows_unless_include_forward():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "scratch.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(paper._SCHEMA)
        conn.execute("CREATE TABLE daily_bars (ticker, date, open, high, low, close, adj_close, volume)")
        rng = np.random.default_rng(3)
        days = pd.bdate_range("2020-01-01", periods=700).strftime("%Y-%m-%d")
        for t in TICKERS:
            c = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 700)))
            for i, d in enumerate(days):
                conn.execute("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?,?)",
                             (t, d, c[i], c[i], c[i], c[i], c[i], int(1e6 + 1e5 * rng.random())))
        conn.commit()
        live_days = _live_days_after(list(days), 6)
        cfg = {**CFG, "data": {**CFG["data"], "db_path": str(db_path)},
               "phase6": {**CFG["phase6"], "replay_end_date": live_days[2]}}
        _live(conn, live_days)
        paper._process_pending(conn, cfg, live_days)
        conn.close()

        try:
            paper.reset(cfg)
            assert False, "expected RuntimeError: forward rows present"
        except RuntimeError:
            pass
        assert paper.reset(cfg, include_forward=True) == 0
        conn2 = sqlite3.connect(db_path)
        tables = [r[0] for r in conn2.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        conn2.close()
        assert "paper_nav" not in tables


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
