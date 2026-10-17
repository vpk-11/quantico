"""Trial registry and prediction storage (Phase 3).

k = number of distinct non-control fingerprints. Every scored run registers its
fingerprint (no-op if already present), so reruns never inflate k.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

_DDL = """
CREATE TABLE IF NOT EXISTS trials (
    fingerprint TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,
    is_control INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    model_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    fold_year INTEGER NOT NULL,
    prediction REAL NOT NULL,
    target REAL NOT NULL,
    PRIMARY KEY (model_id, ticker, date)
);
"""


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(_DDL)


def fingerprint(model_name: str, params: dict, horizon: int, fold_cfg: dict) -> str:
    blob = json.dumps(
        {"model": model_name, "params": params, "horizon": horizon, "folds": fold_cfg}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def register(conn: sqlite3.Connection, fp: str, model_id: str, snapshot: dict, is_control: bool = False) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO trials VALUES (?, ?, ?, ?, ?)",
        (fp, model_id, datetime.now(timezone.utc).isoformat(), json.dumps(snapshot, sort_keys=True), int(is_control)),
    )
    conn.commit()


def count_k(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(DISTINCT fingerprint) FROM trials WHERE is_control = 0").fetchone()[0]


def store_predictions(conn: sqlite3.Connection, model_id: str, df: pd.DataFrame) -> int:
    """Replace all rows for `model_id`. `df` columns: ticker, date, fold_year, prediction, target."""
    conn.execute("DELETE FROM predictions WHERE model_id = ?", (model_id,))
    conn.executemany(
        "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        [(model_id, r.ticker, r.date, int(r.fold_year), float(r.prediction), float(r.target))
         for r in df.itertuples(index=False)],
    )
    conn.commit()
    return len(df)
