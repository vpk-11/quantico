"""Offline checks for the Phase 3.5 additions on synthetic data.

Weight function, horizon-10 target and purge gap, random forest integrity (I1, I4, I5, I6),
paired test, and the adoption rule. No network, no research store.
Run: python tests/test_phase35.py  (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import escalation as es  # noqa: E402
import integrity as ig  # noqa: E402
import models  # noqa: E402
from controls import permutation_null  # noqa: E402
from test_phase3 import CFG, COLS, synth  # noqa: E402
from walkforward import prepare_frame, recency_weights, stack, walk_forward  # noqa: E402

RF_CFG = {**CFG, "rf_n_estimators": 10, "rf_max_depth": 3, "rf_min_samples_leaf": 500,
          "rf_max_features": 0.5, "n_permutations": 25}
CFG10 = {**CFG, "target_horizon_days": 10, "newey_west_lags": 10}


def _rf():
    return models.make("rf", RF_CFG)


_P5 = None
_P10 = None


def prep5():
    global _P5
    if _P5 is None:
        _P5 = prepare_frame(synth(), COLS, CFG)
    return _P5


def prep10():
    global _P10
    if _P10 is None:
        _P10 = prepare_frame(synth(), COLS, CFG10)
    return _P10


def test_weight_age_zero_is_one_and_half_life_is_half():
    dates = np.array(["2012-01-01", "2016-01-01"])  # exactly 1461 days = 4 * 365.25
    w = recency_weights(dates, 4)
    assert w[1] == 1.0 and abs(w[0] - 0.5) < 1e-12


def test_weights_decrease_monotonically_with_age():
    dates = np.array(["2008-01-02", "2010-06-01", "2012-01-03", "2015-12-31"])
    for hl in (2, 4, 8):
        w = recency_weights(dates, hl)
        assert (np.diff(w) > 0).all() and w.max() == 1.0


def test_weights_depend_only_on_training_dates():
    p = prep5()
    seen = []

    def spy(train_dates):
        seen.append((train_dates.max(), train_dates.copy()))
        return recency_weights(train_dates, 4)

    dates = p.frame["date"].to_numpy()
    walk_forward(p.X, p.y, p.folds, lambda: models.make("ridge", CFG), dates=dates, weight_fn=spy)
    for f, (last, d) in zip(p.folds, seen):
        assert last < f.test_start  # no test-period date reaches the weight function
        assert np.array_equal(recency_weights(d, 4), recency_weights(d[::-1], 4)[::-1])
    base = stack(walk_forward(p.X, p.y, p.folds, lambda: models.make("ridge", CFG)))[0]
    weighted = stack(walk_forward(p.X, p.y, p.folds, lambda: models.make("ridge", CFG), dates=dates,
                                  weight_fn=lambda d: recency_weights(d, 2)))[0]
    assert not np.array_equal(base, weighted)  # sample_weight reaches the model


def test_horizon_10_target_and_purge_gap():
    assert ig.i2_target(10).ok
    p = prep10()
    assert ig.i1_split(p, list(range(2016, 2026))).ok
    n_tickers = p.frame["ticker"].nunique()
    year = p.frame["date"].str[:4].astype(int).to_numpy()
    for f in p.folds:
        assert int((year < f.test_year).sum()) - len(f.train) == 10 * n_tickers
    assert (prep5().frame["target_end"] != p.frame["target_end"].reindex(prep5().frame.index)).any()


def test_random_forest_is_deterministic_and_exposes_importances():
    p = prep5()
    f = p.folds[0]
    a = _rf().fit(p.X[f.train], p.y[f.train])
    b = _rf().fit(p.X[f.train], p.y[f.train])
    assert np.array_equal(a.predict(p.X[f.test]), b.predict(p.X[f.test]))
    assert abs(a.feature_importances_.sum() - 1.0) < 1e-9


def test_random_forest_integrity_i1_i4_i5_i6():
    p = prep5()
    null = permutation_null(p, _rf, RF_CFG)
    assert ig.i1_split(p, list(range(2016, 2026))).ok
    c4 = ig.i4_shuffle(null)
    assert c4.ok, c4.detail
    c5 = ig.i5_planted(p, _rf, RF_CFG, k=1)
    assert c5.ok, c5.detail
    c6 = ig.i6_determinism(p, _rf, RF_CFG)
    assert c6.ok, c6.detail


def test_paired_test():
    rng = np.random.default_rng(0)
    base = 0.01 + 0.2 * rng.normal(size=600)
    better = base + 0.08 + 0.02 * rng.normal(size=600)
    cfg = {"significance_alpha": 0.05, "newey_west_lags": 5}
    assert es.paired_test(better, base, 3, cfg)["passed"]
    assert not es.paired_test(base, base, 3, cfg)["passed"]  # zero difference
    assert not es.paired_test(base, better, 1, cfg)["passed"]  # worse than baseline
    assert es.paired_test(better, base, 3, cfg)["z_crit"] > es.paired_test(better, base, 1, cfg)["z_crit"]


def _row(mid, ic, gates=True, integ=True, same=True, paired=True):
    return {"model_id": mid, "mean_ic": ic, "gates_ok": gates, "integrity_ok": integ,
            "same_target": same, "paired_ok": paired}


def test_adoption_rule():
    label = "no validated edge"
    assert es.adopt([_row("a", 0.05, gates=False), _row("b", 0.04, integ=False)], label) == ("ridge_h5_baseline", label)
    assert es.adopt([_row("a", 0.05, paired=False)], label) == ("ridge_h5_baseline", label)  # same target needs the paired test
    assert es.adopt([_row("h10", 0.05, same=False, paired=False)], label) == ("h10", "exploratory-validated")
    assert es.adopt([_row("a", 0.03), _row("b", 0.06), _row("c", 0.05, gates=False)], label) == ("b", "exploratory-validated")


def test_fail_path_candidates_have_distinct_fingerprints():
    cfg35 = {"rf_n_estimators": 300, "rf_max_depth": 6, "rf_min_samples_leaf": 200, "rf_max_features": 0.5,
             "alt_horizon_days": 10, "alt_horizon_newey_west_lags": 10}
    c = es.candidates(CFG, cfg35)
    fps = {es.fingerprint(x, CFG) for x in c.values()}
    from baseline import fingerprint as base_fp
    assert len(fps) == 2 and base_fp(CFG) not in fps
    assert c["ridge_h10"].horizon == 10 and c["ridge_h10"].lags == 10 and not c["ridge_h10"].same_target


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
