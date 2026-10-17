"""Offline checks for the Phase 3 harness on synthetic data (integrity I1 to I6 plus units).

No network, no research store. Run: python tests/test_phase3.py  (or: python -m pytest tests)
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import integrity as ig  # noqa: E402
import models  # noqa: E402
import trials  # noqa: E402
from controls import Shuffler, permutation_null  # noqa: E402
from walkforward import Scaler, prepare_frame  # noqa: E402

COLS = ft.feature_columns([21, 63, 126, 252])
CFG = {
    "target_horizon_days": 5, "initial_train_years": 8, "test_step_years": 1, "ridge_alpha": 1.0,
    "log_liquidity": True, "min_tickers_per_day": 10, "significance_alpha": 0.05, "newey_west_lags": 5,
    "min_positive_years": 7, "n_permutations": 40, "bootstrap_resamples": 300,
    "bootstrap_block_days": 20, "random_seed": 42,
}
WARMUP = 270  # rows per ticker with NULL features, so the first usable year is 2008


def _model():
    return models.make("ridge", CFG)


def synth(n_tickers=12, seed=0):
    """Random-walk prices, iid features independent of future returns (no signal)."""
    rng = np.random.default_rng(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2007-01-01", "2025-12-31")]
    frames = []
    for i in range(n_tickers):
        n = len(dates)
        f = pd.DataFrame(rng.normal(size=(n, len(COLS))), columns=COLS)
        f["liquidity"] = np.exp(rng.normal(20, 1, n))
        f.iloc[:WARMUP] = np.nan
        f.insert(0, "close", 100 * np.exp(np.cumsum(0.01 * rng.normal(size=n))))
        f.insert(0, "date", dates)
        f.insert(0, "ticker", f"T{i:02d}")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


_PREP = None


def prep():
    global _PREP
    if _PREP is None:
        _PREP = prepare_frame(synth(), COLS, CFG)
    return _PREP


def test_i2_target_correctness():
    c = ig.i2_target()
    assert c.ok, c.detail


def test_i1_split_integrity_and_fold_boundaries():
    p = prep()
    c = ig.i1_split(p, list(range(2016, 2026)))
    assert c.ok, c.detail
    assert [f.test_year for f in p.folds] == list(range(2016, 2026))
    # purge gap is live: the last rows before a test year are dropped from training
    year = p.frame["date"].str[:4].astype(int).to_numpy()
    f0 = p.folds[0]
    assert len(f0.train) < int((year < 2016).sum())
    assert p.frame["date"].iloc[f0.train].min()[:4] == "2008"


def test_i1_detects_leaky_fold():
    p = prep()
    f = p.folds[0]
    f.train = np.concatenate([f.train, f.test[:5]])  # leak test rows into training
    try:
        assert not ig.i1_split(p).ok
    finally:
        f.train = f.train[:-5]


def test_i3_scaler_isolation():
    c = ig.i3_scaler(prep(), _model)
    assert c.ok, c.detail


def test_i6_determinism():
    c = ig.i6_determinism(prep(), _model, CFG)
    assert c.ok, c.detail


def test_i4_shuffle_null_is_centered():
    null = permutation_null(prep(), _model, CFG)
    c = ig.i4_shuffle(null)
    assert c.ok, c.detail
    assert len(null) == CFG["n_permutations"]


def test_i5_planted_signal_is_recovered():
    c = ig.i5_planted(prep(), _model, CFG, k=1)
    assert c.ok, c.detail


def test_noise_does_not_pass_gates():
    p = prep()
    from walkforward import stack, walk_forward
    res = walk_forward(p.X, p.y, p.folds, _model)
    pred, naive = stack(res)
    sc = ev.score(pred, p.y[p.test_idx], naive, p.panel, CFG)
    gs = ev.gates(sc, permutation_null(p, _model, CFG), 1, CFG)
    assert not all(g.passed for g in gs)
    assert abs(sc.mean_ic) < 0.02


def test_shuffler_permutes_within_date_only():
    p = prep()
    dates = p.frame["date"].to_numpy()
    y2 = Shuffler(dates).shuffle(p.y, np.random.default_rng(1))
    assert not np.array_equal(y2, p.y)
    for d in ["2012-03-14", "2020-07-01"]:
        m = dates == d
        assert np.array_equal(np.sort(y2[m]), np.sort(p.y[m]))
    assert np.array_equal(np.sort(y2), np.sort(p.y))


def test_log_liquidity_only_at_model_input():
    p = prep()
    j = COLS.index("liquidity")
    assert np.allclose(p.X[:, j], np.log(p.frame["liquidity"].to_numpy()))
    assert (p.frame["liquidity"] > 1e6).all()  # frame itself untouched


def test_scaler_uses_fit_rows_only_and_rejects_constant():
    X = np.array([[1.0, 5.0], [3.0, 5.5], [5.0, 6.0]])
    s = Scaler().fit(X)
    assert np.allclose(s.transform(X).mean(axis=0), 0) and np.allclose(s.transform(X).std(axis=0), 1)
    try:
        Scaler().fit(np.ones((4, 2)))
        raise AssertionError("constant feature should raise")
    except ValueError:
        pass


def test_daily_ic_matches_scipy_spearman():
    rng = np.random.default_rng(3)
    nd, nt = 30, 15
    dates = np.repeat([f"2020-01-{d + 1:02d}" for d in range(nd)], nt)
    ticks = np.tile([f"T{i:02d}" for i in range(nt)], nd)
    pred, tgt = rng.normal(size=nd * nt), rng.normal(size=nd * nt)
    tgt += 0.5 * pred
    panel = ev.make_panel(dates, ticks)
    ic = ev.daily_ic(pred, tgt, panel, 10)
    for d in range(nd):
        sl = slice(d * nt, (d + 1) * nt)
        assert abs(ic[d] - spearmanr(pred[sl], tgt[sl])[0]) < 1e-12
    # a date with too few tickers gives NaN
    keep = ~((dates == "2020-01-01") & (np.arange(len(dates)) % nt >= 5))
    p2 = ev.make_panel(dates[keep], ticks[keep])
    assert np.isnan(ev.daily_ic(pred[keep], tgt[keep], p2, 10)[0])


def test_newey_west_matches_worked_example():
    x = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 3.0, 7.0])
    n, L = len(x), 2
    d = x - x.mean()
    g = [d[l:] @ d[: n - l] / n for l in range(L + 1)]
    lrv = g[0] + 2 * (1 - 1 / 3) * g[1] + 2 * (1 - 2 / 3) * g[2]
    assert abs(ev.newey_west_t(x, L) - x.mean() / np.sqrt(lrv / n)) < 1e-12
    # zero lags reduces to the plain t-stat with ddof=0 variance
    assert abs(ev.newey_west_t(x, 0) - x.mean() / (x.std() / np.sqrt(n))) < 1e-12


def test_z_crit_matches_phase_doc_table():
    got = [round(ev.z_crit(0.05 / k), 2) for k in (1, 2, 3, 4, 6)]
    assert got == [1.96, 2.24, 2.39, 2.50, 2.64]


def test_r2_vs_naive():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ev.r2_vs_naive(y, y, np.full(4, y.mean())) == 1.0
    assert ev.r2_vs_naive(y, np.full(4, y.mean()), np.full(4, y.mean())) == 0.0


def test_bootstrap_lower_bound_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    assert ev.bootstrap_lower(0.1 + 0.05 * rng.normal(size=500), 20, 500, 0.05, 1) > 0
    assert ev.bootstrap_lower(0.05 * rng.normal(size=500), 20, 500, 0.05, 1) < 0


def test_trial_registry_counts_distinct_noncontrol_fingerprints():
    conn = sqlite3.connect(":memory:")
    trials.init(conn)
    fp = trials.fingerprint("ridge", {"alpha": 1.0}, 5, {"initial": 8})
    trials.register(conn, fp, "ridge_h5_baseline", CFG)
    trials.register(conn, fp, "ridge_h5_baseline", CFG)  # rerun does not inflate k
    assert trials.count_k(conn) == 1
    trials.register(conn, trials.fingerprint("ridge", {"alpha": 2.0}, 5, {"initial": 8}), "other", CFG)
    assert trials.count_k(conn) == 2
    trials.register(conn, trials.fingerprint("rf", {}, 5, {}), "ctl", CFG, is_control=True)
    assert trials.count_k(conn) == 2
    df = pd.DataFrame({"ticker": ["A", "B"], "date": ["2020-01-02"] * 2, "fold_year": [2020] * 2,
                       "prediction": [0.1, 0.2], "target": [0.0, 0.3]})
    trials.store_predictions(conn, "m", df)
    assert trials.store_predictions(conn, "m", df) == 2  # replace, not append
    assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
