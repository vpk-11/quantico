"""Controls: shuffled-target null (G4/I4) and planted-signal recovery (I5).

Control runs never write to `predictions` and never register a trial.
"""

from __future__ import annotations

import numpy as np

from evaluate import mean_ic
from walkforward import Prepared, stack, walk_forward


class Shuffler:
    """Permutes a target across tickers within each date."""

    def __init__(self, dates: np.ndarray):
        _, self.date_code = np.unique(dates, return_inverse=True)
        self.base = np.argsort(self.date_code, kind="stable")

    def shuffle(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Sorting by (date, random key) permutes rows inside each date block;
        # `base` lists the same blocks in the same order, so values stay in-date.
        order = np.lexsort((rng.random(len(y)), self.date_code))
        out = np.empty_like(y)
        out[self.base] = y[order]
        return out


def permutation_null(prep: Prepared, make_model, cfg: dict, X: np.ndarray | None = None) -> np.ndarray:
    """Mean IC of `n_permutations` full walk-forwards on date-shuffled targets.

    Targets are shuffled for both training and scoring, per the G4 definition.
    """
    X = prep.X if X is None else X
    shuffler = Shuffler(prep.frame["date"].to_numpy())
    out = np.empty(cfg["n_permutations"])
    for i in range(len(out)):
        y = shuffler.shuffle(prep.y, np.random.default_rng([cfg["random_seed"], i]))
        pred, _ = stack(walk_forward(X, y, prep.folds, make_model))
        out[i] = mean_ic(pred, y[prep.test_idx], prep.panel, cfg["min_tickers_per_day"])
    return out


def plant_signal(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Append a synthetic feature: realized target plus noise with the target's std."""
    return np.column_stack([X, y + rng.normal(0.0, y.std(), len(y))])
