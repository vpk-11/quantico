"""Walk-forward harness: target builder, preprocessor, fold generator, runner.

Phase 3 scope. Everything here is a function over arrays/frames except
`load_frame`. Time-respecting discipline lives in two places:
  * `make_folds`: expanding training window plus a purge gap, so no training
    row's target window reaches into the test period.
  * `Scaler`: fitted on a fold's training rows only, then applied to the test rows.

Price basis for the target is `daily_bars.close`, the same column the Phase 2
features use (split-adjusted in this store).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluate import Panel, make_panel

LIQUIDITY = "liquidity"


def load_frame(conn: sqlite3.Connection, feature_cols: list[str]) -> pd.DataFrame:
    cols = ", ".join(f"f.{c}" for c in feature_cols)
    return pd.read_sql_query(
        f"SELECT b.ticker, b.date, b.close, {cols} FROM daily_bars b "
        "JOIN features f USING (ticker, date) ORDER BY b.ticker, b.date",
        conn,
    )


def build_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add `target` = close(t+h)/close(t) - 1 and `target_end` = date(t+h) per ticker.

    `df` must be sorted by (ticker, date). The last `horizon` rows per ticker
    get NaN for both columns.
    """
    g = df.groupby("ticker", sort=False)
    out = df.copy()
    out["target"] = g["close"].shift(-horizon) / df["close"] - 1.0
    out["target_end"] = g["date"].shift(-horizon)
    return out


@dataclass
class Fold:
    test_year: int
    train: np.ndarray  # row indices into the usable frame
    test: np.ndarray
    test_start: str


def make_folds(usable: pd.DataFrame, initial_train_years: int, test_step_years: int) -> list[Fold]:
    """Expanding-window folds over calendar years of the usable rows.

    A training row is kept only if its target window ends strictly before the
    first test date (purge gap). Fold boundaries come from the data: the first
    test year is `first usable year + initial_train_years`.
    """
    year = usable["date"].str[:4].astype(int).to_numpy()
    date = usable["date"].to_numpy()
    tend = usable["target_end"].to_numpy()
    folds = []
    for ty in range(year.min() + initial_train_years, year.max() + 1, test_step_years):
        test = np.flatnonzero((year >= ty) & (year < ty + test_step_years))
        start = str(date[test].min())
        train = np.flatnonzero((year < ty) & (tend < start))
        folds.append(Fold(ty, train, test, start))
    return folds


class Scaler:
    """Per-feature standardization. Statistics come from `fit` input only."""

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean_, self.scale_ = X.mean(axis=0), X.std(axis=0)
        if (self.scale_ == 0).any():
            raise ValueError("zero-variance feature in training rows")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


@dataclass
class Prepared:
    frame: pd.DataFrame  # usable rows only, reset index
    X: np.ndarray  # model-input features (log liquidity applied), unstandardized
    y: np.ndarray
    feature_cols: list[str]
    folds: list[Fold]
    test_idx: np.ndarray  # all test rows, concatenated in fold order
    panel: Panel  # grid over test_idx


def prepare_frame(df: pd.DataFrame, feature_cols: list[str], cfg: dict) -> Prepared:
    """Preprocessor: target, drop NULL rows, log liquidity, folds, test panel."""
    frame = (
        build_target(df, cfg["target_horizon_days"])
        .dropna(subset=[*feature_cols, "target"])
        .reset_index(drop=True)
    )
    X = frame[feature_cols].to_numpy(float, copy=True)
    if cfg["log_liquidity"]:
        j = feature_cols.index(LIQUIDITY)
        if (X[:, j] <= 0).any():
            raise ValueError("non-positive liquidity cannot be logged")
        X[:, j] = np.log(X[:, j])
    folds = make_folds(frame, cfg["initial_train_years"], cfg["test_step_years"])
    test_idx = np.concatenate([f.test for f in folds])
    panel = make_panel(frame["date"].to_numpy()[test_idx], frame["ticker"].to_numpy()[test_idx])
    return Prepared(frame, X, frame["target"].to_numpy(float), feature_cols, folds, test_idx, panel)


@dataclass
class FoldResult:
    test_year: int
    pred: np.ndarray
    naive: float  # training-mean target, the G2 baseline predictor
    scaler: Scaler
    model: object


def recency_weights(dates: np.ndarray, half_life_years: float) -> np.ndarray:
    """Sample weights `0.5 ** (age_years / half_life)`, age measured from the last date given.

    Pass a fold's training dates only, so the weights depend on training-period dates alone.
    """
    d = pd.to_datetime(dates)
    age_years = np.asarray((d.max() - d).days) / 365.25
    return 0.5 ** (age_years / half_life_years)


def walk_forward(
    X: np.ndarray, y: np.ndarray, folds: list[Fold], make_model, dates: np.ndarray | None = None, weight_fn=None
) -> list[FoldResult]:
    """`weight_fn(training dates) -> sample weights` is optional; `dates` aligns with the rows of X."""
    out = []
    for f in folds:
        sc = Scaler().fit(X[f.train])
        model = make_model()
        w = None if weight_fn is None else weight_fn(dates[f.train])
        model.fit(sc.transform(X[f.train]), y[f.train], sample_weight=w)
        out.append(FoldResult(f.test_year, model.predict(sc.transform(X[f.test])), float(y[f.train].mean()), sc, model))
    return out


def stack(results: list[FoldResult]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate fold predictions and per-row naive values, in fold order."""
    pred = np.concatenate([r.pred for r in results])
    naive = np.concatenate([np.full(len(r.pred), r.naive) for r in results])
    return pred, naive
