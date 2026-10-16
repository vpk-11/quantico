"""Model interface + registry for the walk-forward harness.

Every model exposes `fit(X, y, sample_weight=None)` and `predict(X)`. Later
phases register new models here without touching the harness.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor as _SkForest
from sklearn.linear_model import Ridge as _SkRidge


class Ridge:
    """Pooled ridge regression, fixed penalty from config, intercept fitted."""

    def __init__(self, alpha: float):
        self._m = _SkRidge(alpha=alpha)

    def fit(self, X, y, sample_weight=None):
        self._m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._m.predict(X)

    @property
    def coef_(self):
        return self._m.coef_

    @property
    def intercept_(self) -> float:
        return float(self._m.intercept_)


class RandomForest:
    """Fixed-parameter random forest (Phase 3.5a). Parameters come from config, never tuned."""

    def __init__(self, n_estimators: int, max_depth: int, min_samples_leaf: int, max_features: float, seed: int):
        self._m = _SkForest(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=seed, n_jobs=6,
        )

    def fit(self, X, y, sample_weight=None):
        self._m.fit(X, y, sample_weight=sample_weight)
        # Trees are seeded up front, so a parallel fit is deterministic. Parallel predict sums
        # tree outputs in thread order, which changes float rounding, so predict single-threaded.
        self._m.n_jobs = 1
        return self

    def predict(self, X):
        return self._m.predict(X)

    @property
    def feature_importances_(self):
        return self._m.feature_importances_


# name -> factory(config dict) -> model
REGISTRY = {
    "ridge": lambda cfg: Ridge(cfg["ridge_alpha"]),
    "rf": lambda cfg: RandomForest(
        cfg["rf_n_estimators"], cfg["rf_max_depth"], cfg["rf_min_samples_leaf"],
        cfg["rf_max_features"], cfg["random_seed"],
    ),
}


def make(name: str, cfg: dict):
    return REGISTRY[name](cfg)
