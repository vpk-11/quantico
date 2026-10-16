"""Phase 6/7 checks on synthetic data: fill accounting, target selection, and no-lookahead in `paper.predict`.

No network, no research store. Run: python tests/test_phase6.py  (or: python -m pytest tests)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest as bt  # noqa: E402
import paper  # noqa: E402

CFG = {
    "features": {"momentum_windows_days": [21, 63, 126, 252], "mean_reversion_window_days": 20},
    "phase3": {"target_horizon_days": 5, "ridge_alpha": 1.0, "log_liquidity": True},
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
    conn.executemany("INSERT INTO paper_fills VALUES (?,?,?,?,?,?,?,?)", [
        ("d1", "A", 1, 100, 100, 5000.0, 1.0, 50.0),
        ("d2", "A", -1, 40, 40, 2200.0, 0.5, 55.0),
        ("d2", "B", 1, 10, 0, 0.0, 0.0, 10.0),
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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
