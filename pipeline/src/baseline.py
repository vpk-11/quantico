"""Phase 3 baseline: ridge regression under expanding walk-forward validation.

Scores out-of-sample predictions against five gates fixed in config.toml
before the first run, checks six integrity tests, and records a verdict by
rule: PASS only if G1 to G5 hold and every integrity test passes; INVALID if
any integrity test fails. A FAIL does not block Phase 4: baseline predictions
flow forward labelled "no validated edge".

Usage (run from pipeline/):
    python src/baseline.py run       # walk-forward + controls + gates, store predictions + trial, write report
    python src/baseline.py verify    # re-derive everything, check stored output, print VERIFY: PASS
    python src/baseline.py report    # regenerate reports/phase3_report.md from config alone
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import features as ft
import integrity as ig
import models
import report
import research_store as rs
import trials
from controls import permutation_null
from evaluate import extras, gates, score
from walkforward import Prepared, load_frame, prepare_frame, stack, walk_forward

_ROOT = Path(__file__).resolve().parents[1]  # pipeline/
REPORT_PATH = _ROOT / "reports" / "phase3_report.md"
MODEL_NAME = "ridge"
BASELINE_K = 1  # the Phase 3 report is the k=1 record; later trials are scored in phase3_5_report.md
EXPECTED_TEST_YEARS = list(range(2016, 2026))  # phase doc: train 2008-2015 / test 2016 .. test 2025
SPOT_CHECKS = [("AAPL", "2016-06-15"), ("JPM", "2020-03-16"), ("XOM", "2025-06-02")]


def model_id(cfg: dict) -> str:
    return f"{MODEL_NAME}_h{cfg['target_horizon_days']}_baseline"


def feature_cols(full_cfg: dict) -> list[str]:
    return ft.feature_columns(full_cfg["features"]["momentum_windows_days"])


def prepare(conn: sqlite3.Connection, full_cfg: dict) -> Prepared:
    cols = feature_cols(full_cfg)
    return prepare_frame(load_frame(conn, cols), cols, full_cfg["phase3"])


def fingerprint(cfg: dict) -> str:
    return trials.fingerprint(
        MODEL_NAME,
        {"ridge_alpha": cfg["ridge_alpha"], "log_liquidity": cfg["log_liquidity"]},
        cfg["target_horizon_days"],
        {"initial_train_years": cfg["initial_train_years"], "test_step_years": cfg["test_step_years"]},
    )


def compute_all(prep: Prepared, cfg: dict, k: int, expected_years: list[int] | None) -> dict:
    """Run the baseline, the G4 shuffles, all integrity tests, and derive the verdict."""
    make_model = lambda: models.make(MODEL_NAME, cfg)  # noqa: E731
    y_test = prep.y[prep.test_idx]
    results = walk_forward(prep.X, prep.y, prep.folds, make_model)
    pred, naive = stack(results)
    sc = score(pred, y_test, naive, prep.panel, cfg)
    null = permutation_null(prep, make_model, cfg)
    gs = gates(sc, null, k, cfg)

    checks = [
        ig.i1_split(prep, expected_years),
        ig.i2_target(),
        ig.i3_scaler(prep, make_model),
        ig.i4_shuffle(null),
        ig.i5_planted(prep, make_model, cfg, k),
        ig.i6_determinism(prep, make_model, cfg),
    ]
    if not all(c.ok for c in checks):
        verdict, label = "INVALID", "integrity failure, no verdict recorded"
    elif all(g.passed for g in gs):
        verdict, label = "PASS", "validated edge"
    else:
        verdict, label = "FAIL", "no validated edge"

    frame = prep.frame
    year = frame["date"].str[:4].astype(int).to_numpy()
    fold_rows = []
    for f in prep.folds:
        fold_rows.append({
            "test_year": f.test_year,
            "train_first": frame["date"].iloc[f.train].min(),
            "train_last": frame["date"].iloc[f.train].max(),
            "n_train": len(f.train),
            "n_purged": int((year < f.test_year).sum()) - len(f.train),
            "n_test": len(f.test),
        })
    ex = extras(
        pred, y_test, frame["mom_21"].to_numpy()[prep.test_idx],
        {r.test_year: np.asarray(r.model.coef_) for r in results}, sc, prep.panel, cfg,
    )
    return {
        "cfg": cfg, "k": k, "model_id": model_id(cfg), "feature_cols": prep.feature_cols,
        "first_year": int(year.min()), "folds": fold_rows, "gates": gs, "checks": checks,
        "extras": ex, "verdict": verdict, "label": label,
        "results": results, "pred": pred, "score": sc,
    }


def predictions_frame(prep: Prepared, res: dict) -> pd.DataFrame:
    idx = prep.test_idx
    return pd.DataFrame({
        "ticker": prep.frame["ticker"].to_numpy()[idx],
        "date": prep.frame["date"].to_numpy()[idx],
        "fold_year": np.concatenate([np.full(len(f.test), f.test_year) for f in prep.folds]),
        "prediction": res["pred"],
        "target": prep.y[idx],
    })


def print_summary(res: dict) -> None:
    print(f"\nmodel {res['model_id']}  k={res['k']}  first usable year {res['first_year']}  folds {len(res['folds'])}")
    for f in res["folds"]:
        print(f"  test {f['test_year']}  train {f['train_first']}..{f['train_last']}  "
              f"train={f['n_train']} purged={f['n_purged']} test={f['n_test']}")
    print("\n--- gates (value vs threshold) ---")
    for g in res["gates"]:
        print(f"  {g.id} {g.name:<18} {g.value:<46} threshold {g.threshold:<34} {'PASS' if g.passed else 'FAIL'}")
    print("\n--- integrity ---")
    for c in res["checks"]:
        print(f"  {c.id} {c.name:<24} {'ok' if c.ok else 'FAILED':<7} {c.detail}")
    print(f"\nVERDICT: {res['verdict']} ({res['label']})")


def _connect(cfg: dict) -> sqlite3.Connection:
    path = _ROOT / cfg["data"]["db_path"]
    if not path.exists():
        raise RuntimeError(f"no research store at {path}; run research_store.py and features.py build first")
    return sqlite3.connect(path)


def write_report(res: dict) -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report.render(res))
    print(f"report: {REPORT_PATH}")


def run(full_cfg: dict) -> int:
    cfg = full_cfg["phase3"]
    conn = _connect(full_cfg)
    try:
        trials.init(conn)
        trials.register(conn, fingerprint(cfg), model_id(cfg), cfg)
        k = BASELINE_K
        prep = prepare(conn, full_cfg)
        res = compute_all(prep, cfg, k, EXPECTED_TEST_YEARS)
        n = trials.store_predictions(conn, model_id(cfg), predictions_frame(prep, res))
        print(f"predictions stored: {n} rows for {res['model_id']}")
    finally:
        conn.close()
    write_report(res)
    print_summary(res)
    return 0 if res["verdict"] != "INVALID" else 1


def regenerate(full_cfg: dict) -> int:
    cfg = full_cfg["phase3"]
    conn = _connect(full_cfg)
    try:
        k = BASELINE_K
        prep = prepare(conn, full_cfg)
    finally:
        conn.close()
    res = compute_all(prep, cfg, k, EXPECTED_TEST_YEARS)
    write_report(res)
    print_summary(res)
    return 0 if res["verdict"] != "INVALID" else 1


def _expected_prediction_rows(conn: sqlite3.Connection, cols: list[str], horizon: int, first_test_year: int) -> int:
    """Independent SQL count: rows in the test years with a forward price and no NULL feature."""
    sel = ", ".join(f"f.{c} AS {c}" for c in cols)
    notnull = " AND ".join(f"{c} IS NOT NULL" for c in cols)
    return conn.execute(
        "SELECT COUNT(*) FROM (SELECT b.ticker AS ticker, b.date AS date, "
        f"LEAD(b.close, {horizon}) OVER (PARTITION BY b.ticker ORDER BY b.date) AS f_close, {sel} "
        "FROM daily_bars b JOIN features f USING (ticker, date)) "
        f"WHERE date >= '{first_test_year}-01-01' AND f_close IS NOT NULL AND {notnull}"
    ).fetchone()[0]


def verify(full_cfg: dict) -> bool:
    cfg = full_cfg["phase3"]
    conn = _connect(full_cfg)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  {'ok  ' if passed else 'FAIL'} {label}  {detail}")

    try:
        k = BASELINE_K
        prep = prepare(conn, full_cfg)
        res1 = compute_all(prep, cfg, k, EXPECTED_TEST_YEARS)
        res2 = compute_all(prep, cfg, k, EXPECTED_TEST_YEARS)

        print("--- integrity I1 to I6 ---")
        for c in res1["checks"]:
            check(f"{c.id} {c.name}", c.ok, c.detail)

        print("\n--- folds ---")
        years = [f["test_year"] for f in res1["folds"]]
        check("10 folds, test years 2016..2025", years == EXPECTED_TEST_YEARS, f"got {years}")
        check("first usable year is 2008", res1["first_year"] == 2008, f"got {res1['first_year']}")

        print("\n--- predictions table ---")
        stored = pd.read_sql_query(
            "SELECT ticker, date, fold_year, prediction, target FROM predictions WHERE model_id = ? ORDER BY ticker, date",
            conn, params=(model_id(cfg),),
        )
        want = _expected_prediction_rows(conn, prep.feature_cols, cfg["target_horizon_days"], EXPECTED_TEST_YEARS[0])
        check("row count equals usable-target rows in test years", len(stored) == want, f"stored {len(stored)}, independent SQL {want}")
        fresh = predictions_frame(prep, res1).sort_values(["ticker", "date"]).reset_index(drop=True)
        check("stored predictions equal recomputed predictions",
              all(np.array_equal(stored[c].to_numpy(), fresh[c].to_numpy()) for c in stored.columns))
        registered = conn.execute("SELECT COUNT(*) FROM trials WHERE model_id = ?", (model_id(cfg),)).fetchone()[0]
        check("baseline is a registered trial", registered == 1, f"rows={registered}")

        print("\n--- gates at k=1 and verdict rule ---")
        for g in res1["gates"]:
            print(f"       {g.id} {g.value}  threshold {g.threshold}  {'PASS' if g.passed else 'FAIL'}")
        rule = "INVALID" if not all(c.ok for c in res1["checks"]) else ("PASS" if all(g.passed for g in res1["gates"]) else "FAIL")
        check("verdict follows from the rule", res1["verdict"] == rule, f"verdict {res1['verdict']}")

        print("\n--- reruns and report ---")
        text1, text2 = report.render(res1), report.render(res2)
        check("two reruns identical (report text and predictions)",
              text1 == text2 and np.array_equal(res1["pred"], res2["pred"]))
        check("report file regenerates byte-identical",
              REPORT_PATH.exists() and REPORT_PATH.read_text() == text1)

        print("\n--- spot checks: prediction from scaler parameters and coefficients ---")
        cols = prep.feature_cols
        by_year = {f.test_year: r for f, r in zip(prep.folds, res1["results"])}
        for ticker, date in SPOT_CHECKS:
            row = conn.execute(
                f"SELECT date, {', '.join(cols)} FROM features WHERE ticker = ? AND date >= ? ORDER BY date LIMIT 1",
                (ticker, date),
            ).fetchone()
            x = np.array(row[1:], dtype=float)
            if cfg["log_liquidity"]:
                x[cols.index("liquidity")] = np.log(x[cols.index("liquidity")])
            r = by_year[int(row[0][:4])]
            hand = r.model.intercept_ + float(((x - r.scaler.mean_) / r.scaler.scale_) @ r.model.coef_)
            got = conn.execute(
                "SELECT prediction FROM predictions WHERE model_id = ? AND ticker = ? AND date = ?",
                (model_id(cfg), ticker, row[0]),
            ).fetchone()[0]
            check(f"{ticker} {row[0]}", abs(hand - got) < 1e-9, f"hand {hand:.12f} stored {got:.12f}")
    finally:
        conn.close()

    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico baseline model + walk-forward (Phase 3)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="walk-forward, controls, gates; store predictions + trial; write report")
    sub.add_parser("verify", help="re-derive and check stored output")
    sub.add_parser("report", help="regenerate the report from config alone")
    args = parser.parse_args(argv)

    full_cfg = rs.load_config()
    if args.cmd == "run":
        return run(full_cfg)
    if args.cmd == "verify":
        return 0 if verify(full_cfg) else 1
    return regenerate(full_cfg)


if __name__ == "__main__":
    sys.exit(main())
