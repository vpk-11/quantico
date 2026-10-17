"""Integrity tests I1 to I6. A failure here is a harness bug, never a result.

Each check takes plain inputs and returns a `Check`, so the same functions run
on synthetic data in `tests/` and on the real store in `baseline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from controls import permutation_null, plant_signal
from evaluate import gates, null_centered, score
from walkforward import Prepared, build_target, stack, walk_forward


@dataclass
class Check:
    id: str
    name: str
    ok: bool
    detail: str


def _fit(prep: Prepared, X, make_model):
    results = walk_forward(X, prep.y, prep.folds, make_model)
    pred, naive = stack(results)
    return results, pred, naive


def i1_split(prep: Prepared, expected_test_years: list[int] | None = None) -> Check:
    date, tend = prep.frame["date"].to_numpy(), prep.frame["target_end"].to_numpy()
    for f in prep.folds:
        if len(f.train) == 0:
            return Check("I1", "Split integrity", False, f"fold {f.test_year} has no training rows")
        if not date[f.train].max() < date[f.test].min():
            return Check("I1", "Split integrity", False, f"fold {f.test_year}: train date reaches test period")
        if not tend[f.train].max() < date[f.test].min():
            return Check("I1", "Split integrity", False, f"fold {f.test_year}: train target window reaches test period")
    years = [f.test_year for f in prep.folds]
    if expected_test_years is not None and years != expected_test_years:
        return Check("I1", "Split integrity", False, f"test years {years} != expected {expected_test_years}")
    return Check("I1", "Split integrity", True, f"{len(years)} folds, test years {years[0]}..{years[-1]}, no overlap")


def i2_target(h: int = 5) -> Check:
    """Synthetic closes with known returns reproduce exact targets; tail rows are NULL."""
    rets = {"AAA": [0.01, -0.02, 0.03, 0.00, 0.05, -0.01, 0.02, 0.04, -0.03, 0.01, 0.02],
            "BBB": [-0.01, 0.02, 0.01, 0.03, -0.02, 0.04, 0.00, -0.05, 0.02, 0.01, 0.03]}
    rows = []
    for t, r in rets.items():
        close = 100.0 * np.cumprod([1.0, *[1 + x for x in r]])
        rows += [(t, f"2020-01-{d + 1:02d}", c) for d, c in enumerate(close)]
    out = build_target(pd.DataFrame(rows, columns=["ticker", "date", "close"]), h)
    for _, g in out.groupby("ticker"):
        c, d, tg, te = g["close"].to_numpy(), g["date"].to_numpy(), g["target"].to_numpy(), g["target_end"].to_numpy()
        n = len(g)
        if not np.allclose(tg[: n - h], c[h:] / c[: n - h] - 1.0, rtol=0, atol=1e-12):
            return Check("I2", "Target correctness", False, "target != close(t+h)/close(t) - 1")
        if not (np.isnan(tg[n - h:]).all() and pd.isna(te[n - h:]).all() and (te[: n - h] == d[h:]).all()):
            return Check("I2", "Target correctness", False, "tail rows not NULL or target_end misaligned")
    return Check("I2", "Target correctness", True, f"exact targets on known series, last {h} rows per ticker NULL")


def i3_scaler(prep: Prepared, make_model) -> Check:
    """Changing a fold's test features must not move its scaler statistics or training transform."""
    for f in prep.folds:
        X2 = prep.X.copy()
        X2[f.test] += 1000.0
        (a,) = walk_forward(prep.X, prep.y, [f], make_model)
        (b,) = walk_forward(X2, prep.y, [f], make_model)
        same = (
            np.array_equal(a.scaler.mean_, b.scaler.mean_)
            and np.array_equal(a.scaler.scale_, b.scaler.scale_)
            and np.array_equal(a.scaler.transform(prep.X[f.train]), b.scaler.transform(X2[f.train]))
        )
        if not same:
            return Check("I3", "Scaler isolation", False, f"fold {f.test_year}: scaler depends on test rows")
    return Check("I3", "Scaler isolation", True, "scaler and training transform unchanged when a fold's test features are shifted")


def i4_shuffle(null_means: np.ndarray) -> Check:
    ok, detail = null_centered(null_means)
    return Check("I4", "Shuffle control", ok, detail)


def i5_planted(prep: Prepared, make_model, cfg: dict, k: int) -> Check:
    rng = np.random.default_rng([cfg["random_seed"], 10**6])
    Xp = plant_signal(prep.X, prep.y, rng)
    _, pred, naive = _fit(prep, Xp, make_model)
    sc = score(pred, prep.y[prep.test_idx], naive, prep.panel, cfg)
    gs = gates(sc, permutation_null(prep, make_model, cfg, X=Xp), k, cfg)
    failed = [g.id for g in gs if not g.passed]
    ok = sc.mean_ic > 0.3 and not failed
    return Check("I5", "Planted-signal recovery", ok,
                 f"mean IC {sc.mean_ic:.4f} (need > 0.3), gates failed: {failed or 'none'}")


def i6_determinism(prep: Prepared, make_model, cfg: dict) -> Check:
    runs = []
    for _ in range(2):
        _, pred, naive = _fit(prep, prep.X, make_model)
        runs.append((pred, score(pred, prep.y[prep.test_idx], naive, prep.panel, cfg)))
    (p1, s1), (p2, s2) = runs
    ok = np.array_equal(p1, p2) and (s1.mean_ic, s1.nw_t, s1.r2) == (s2.mean_ic, s2.nw_t, s2.r2)
    return Check("I6", "Determinism", ok, "two runs: identical predictions and metrics" if ok else "runs differ")
