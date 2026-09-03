"""Offline checks for features.compute_features and the sqlite round-trip.

No network, no research store needed. Run: python tests/test_features.py  (or: pytest)
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as ft  # noqa: E402

MOM = [3, 5]
MR = 4


def _bars(closes, volumes=None):
    idx = pd.bdate_range("2020-01-01", periods=len(closes), name="date")
    vol = volumes if volumes is not None else [1_000] * len(closes)
    return pd.DataFrame({"close": [float(c) for c in closes], "volume": vol}, index=idx)


def test_momentum_is_trailing_return():
    closes = [10, 11, 12, 13, 14, 15, 16, 17]
    out = ft.compute_features(_bars(closes), MOM, MR)
    # mom_3 at position 5 uses close[5] / close[2] - 1 = 15/12 - 1
    assert abs(out["mom_3"].iloc[5] - (15 / 12 - 1)) < 1e-12
    # warm-up: first 3 rows of mom_3 are NaN, position 3 is the first live value
    assert out["mom_3"].iloc[:3].isna().all()
    assert not np.isnan(out["mom_3"].iloc[3])
    assert abs(out["mom_3"].iloc[3] - (13 / 10 - 1)) < 1e-12


def test_no_lookahead_truncation_invariance():
    # The explicit phase-doc test: past feature values must not move when future
    # rows are removed from the input.
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 1, 60))
    vols = rng.integers(1_000, 5_000, 60)
    full = ft.compute_features(_bars(closes, vols), MOM, MR)
    cut = 40
    truncated = ft.compute_features(_bars(closes[:cut], vols[:cut]), MOM, MR)
    common = full.iloc[:cut]
    pd.testing.assert_frame_equal(common, truncated, check_exact=False, atol=1e-12, rtol=0)


def test_rolling_windows_not_centered():
    # A spike at t+1 must not touch any feature at t.
    closes = [100.0] * 30
    a = ft.compute_features(_bars(closes), MOM, MR)
    closes_spiked = closes.copy()
    closes_spiked[20] = 250.0
    b = ft.compute_features(_bars(closes_spiked), MOM, MR)
    pd.testing.assert_series_equal(a.iloc[:20].stack(), b.iloc[:20].stack())


def test_warmup_nulls_then_dense():
    closes = list(range(1, 41))
    out = ft.compute_features(_bars(closes), MOM, MR)
    for col in out.columns:
        s = out[col]
        fv = s.first_valid_index()
        assert fv is not None
        # everything from the first live index onward is non-null
        assert s.loc[fv:].notna().all(), col
        # everything before it is null
        assert s.loc[:fv].iloc[:-1].isna().all(), col


def test_volatility_matches_manual_std():
    rng = np.random.default_rng(1)
    closes = 50 + np.cumsum(rng.normal(0, 0.5, 30))
    out = ft.compute_features(_bars(closes), MOM, MR)
    ret = pd.Series(closes).pct_change()
    manual = ret.iloc[-MR:].std(ddof=1)
    assert abs(out["volatility"].iloc[-1] - manual) < 1e-12


def test_zscore_matches_manual():
    rng = np.random.default_rng(2)
    closes = 20 + np.cumsum(rng.normal(0, 0.3, 25))
    out = ft.compute_features(_bars(closes), MOM, MR)
    w = pd.Series(closes).iloc[-MR:]
    manual = (closes[-1] - w.mean()) / w.std(ddof=1)
    assert abs(out["mean_rev_z"].iloc[-1] - manual) < 1e-12


def test_zero_std_window_gives_nan_not_inf():
    closes = [100.0] * 20
    out = ft.compute_features(_bars(closes), MOM, MR)
    assert out["mean_rev_z"].replace([np.inf, -np.inf], np.nan).isna().all()


def test_liquidity_is_trailing_mean_dollar_volume():
    closes = [10.0] * 10
    vols = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    out = ft.compute_features(_bars(closes, vols), MOM, MR)
    # position 5: mean of close*vol over positions 2..5 = 10 * mean(300,400,500,600)
    assert abs(out["liquidity"].iloc[5] - 10 * np.mean([300, 400, 500, 600])) < 1e-9


def test_feature_columns_track_config_windows():
    assert ft.feature_columns([21, 63, 126, 252]) == [
        "mom_21", "mom_63", "mom_126", "mom_252", "mean_rev_z", "volatility", "liquidity",
    ]
    assert ft.feature_columns([10]) == ["mom_10", "mean_rev_z", "volatility", "liquidity"]


def test_sqlite_roundtrip_preserves_nulls_and_order():
    closes = list(range(1, 31))
    feats = ft.compute_features(_bars(closes), MOM, MR)
    with tempfile.TemporaryDirectory() as d:
        conn = sqlite3.connect(Path(d) / "t.db")
        ft.init_features_table(conn, MOM)
        ft.store_features(conn, "AAA", feats)
        ft.store_features(conn, "AAA", feats)  # idempotent
        cols = ft.feature_columns(MOM)
        rows = conn.execute(
            f"SELECT date, {', '.join(cols)} FROM features WHERE ticker = 'AAA' ORDER BY date ASC"
        ).fetchall()
        conn.close()
    assert len(rows) == len(closes)
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    # row tuple is (date, *cols), so column c sits at 1 + cols.index(c)
    mom5_idx = 1 + cols.index("mom_5")
    assert all(r[mom5_idx] is None for r in rows[:5])
    assert rows[5][mom5_idx] is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
