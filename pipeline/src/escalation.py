"""Phase 3.5 escalation comparisons against the Phase 3 baseline.

The path is set by the Phase 3 verdict, not by judgment. Built here: the FAIL
path (3.5a random forest `rf_h5`, 3.5c ridge at a 10-day horizon `ridge_h10`),
both exploratory. The PASS path (3.5b recency weighting) is not run; its weight
function `walkforward.recency_weights` exists and is unit tested.

Every candidate runs on the identical folds, preprocessing, gates and integrity
tests as the baseline. Each scored variant is a trial, so all gates are
restated at the final k, including the baseline's. A variant replaces the
baseline as the Phase 4 input only by the adoption rule in `adopt`.

Usage (run from pipeline/):
    python src/escalation.py run       # run candidates, store predictions + trials, write report
    python src/escalation.py verify    # re-derive and check, print VERIFY: PASS
    python src/escalation.py report    # regenerate reports/phase3_5_report.md from config alone
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import baseline as bl
import integrity as ig
import models
import report
import research_store as rs
import trials
from controls import permutation_null
from evaluate import gates, newey_west_t, score, z_crit
from walkforward import Prepared, load_frame, prepare_frame, stack, walk_forward

_ROOT = Path(__file__).resolve().parents[1]  # pipeline/
REPORT_PATH = _ROOT / "reports" / "phase3_5_report.md"
PHASE3_REPORT = _ROOT / "reports" / "phase3_report.md"
FAIL_PATH = ["rf_h5", "ridge_h10"]


@dataclass
class Candidate:
    model_id: str
    model_name: str
    horizon: int
    lags: int
    same_target: bool  # same target as the baseline, so a paired test applies
    params: dict  # trial fingerprint parameters


def phase3_verdict() -> str:
    m = re.search(r"^## Verdict: (\w+)", PHASE3_REPORT.read_text(), re.M)
    return m.group(1)


def candidate_cfg(base: Candidate | None, cfg3: dict, cfg35: dict) -> dict:
    """Config dict one candidate runs under: Phase 3 keys, 3.5 keys, then its horizon overrides."""
    cfg = {**cfg3, **cfg35}
    if base is not None:
        cfg["target_horizon_days"], cfg["newey_west_lags"] = base.horizon, base.lags
    return cfg


def candidates(cfg3: dict, cfg35: dict) -> dict[str, Candidate]:
    rf_keys = ["rf_n_estimators", "rf_max_depth", "rf_min_samples_leaf", "rf_max_features"]
    return {
        "rf_h5": Candidate(
            "rf_h5", "rf", cfg3["target_horizon_days"], cfg3["newey_west_lags"], True,
            {**{k: cfg35[k] for k in rf_keys}, "random_seed": cfg3["random_seed"], "log_liquidity": cfg3["log_liquidity"]},
        ),
        "ridge_h10": Candidate(
            "ridge_h10", "ridge", cfg35["alt_horizon_days"], cfg35["alt_horizon_newey_west_lags"], False,
            {"ridge_alpha": cfg3["ridge_alpha"], "log_liquidity": cfg3["log_liquidity"]},
        ),
    }


def fingerprint(c: Candidate, cfg3: dict) -> str:
    return trials.fingerprint(
        c.model_name, c.params, c.horizon,
        {"initial_train_years": cfg3["initial_train_years"], "test_step_years": cfg3["test_step_years"]},
    )


def paired_test(ic_variant: np.ndarray, ic_base: np.ndarray, k: int, cfg: dict) -> dict:
    """Daily IC difference (variant minus baseline) with the same Newey-West correction."""
    d = ic_variant - ic_base
    d = d[~np.isnan(d)]
    z = z_crit(cfg["significance_alpha"] / k)
    t = newey_west_t(d, cfg["newey_west_lags"])
    return {"mean_diff": float(d.mean()), "t": t, "z_crit": z, "passed": bool(d.mean() > 0 and t >= z)}


def adopt(rows: list[dict], baseline_label: str) -> tuple[str, str]:
    """Adoption rule. Each row: model_id, mean_ic, gates_ok, integrity_ok, same_target, paired_ok.

    A variant qualifies only if it passes all five gates at the final k and every integrity
    test, and, when it shares the baseline's target, also wins the paired test. Highest mean IC
    among qualifiers wins; otherwise the baseline stays, labelled per its own verdict.
    """
    ok = [r for r in rows if r["gates_ok"] and r["integrity_ok"] and (r["paired_ok"] or not r["same_target"])]
    if not ok:
        return "ridge_h5_baseline", baseline_label
    return max(ok, key=lambda r: r["mean_ic"])["model_id"], "exploratory-validated"


def _timed(label: str, t0: float) -> float:
    now = time.time()
    print(f"  {label}: {now - t0:.0f}s", flush=True)
    return now


def evaluate(prep: Prepared, cfg: dict, make_model, k: int, expected_years, with_checks: bool) -> dict:
    t = time.time()
    results = walk_forward(prep.X, prep.y, prep.folds, make_model)
    pred, naive = stack(results)
    sc = score(pred, prep.y[prep.test_idx], naive, prep.panel, cfg)
    null = permutation_null(prep, make_model, cfg)
    t = _timed("walk-forward + shuffled null", t)
    gs = gates(sc, null, k, cfg)
    checks = []
    if with_checks:
        checks = [
            ig.i1_split(prep, expected_years),
            ig.i4_shuffle(null),
            ig.i5_planted(prep, make_model, cfg, k),
            ig.i6_determinism(prep, make_model, cfg),
        ]
        _timed("integrity (I1, I4, I5, I6)", t)
    return {"results": results, "pred": pred, "score": sc, "gates": gs, "checks": checks}


def compute_all(conn: sqlite3.Connection, full_cfg: dict, k: int) -> dict:
    cfg3, cfg35 = full_cfg["phase3"], full_cfg["phase3_5"]
    verdict = phase3_verdict()
    if verdict != "FAIL":
        raise NotImplementedError(f"Phase 3 verdict {verdict}: PASS path (recency weighting) is not built")
    cols = bl.feature_cols(full_cfg)
    df = load_frame(conn, cols)
    prep5 = prepare_frame(df, cols, cfg3)
    cands = candidates(cfg3, cfg35)

    print("baseline at final k", flush=True)
    base = evaluate(prep5, cfg3, lambda: models.make("ridge", cfg3), k, bl.EXPECTED_TEST_YEARS, False)
    rows = []
    out = {"phase3_verdict": verdict, "k": k, "cfg35": cfg35, "baseline": base, "variants": {}, "feature_cols": cols}
    for mid in FAIL_PATH:
        c = cands[mid]
        cfg_c = candidate_cfg(c, cfg3, cfg35)
        prep = prep5 if c.horizon == cfg3["target_horizon_days"] else prepare_frame(df, cols, cfg_c)
        print(f"{mid} (h={c.horizon})", flush=True)
        v = evaluate(prep, cfg_c, lambda c=c, cfg_c=cfg_c: models.make(c.model_name, cfg_c), k, bl.EXPECTED_TEST_YEARS, True)
        v["prep"], v["cand"] = prep, c
        v["paired"] = paired_test(v["score"].ic, base["score"].ic, k, cfg3) if c.same_target else None
        if hasattr(v["results"][0].model, "feature_importances_"):
            v["importances"] = {r.test_year: np.asarray(r.model.feature_importances_) for r in v["results"]}
        out["variants"][mid] = v
        rows.append({
            "model_id": mid, "mean_ic": v["score"].mean_ic, "gates_ok": all(g.passed for g in v["gates"]),
            "integrity_ok": all(x.ok for x in v["checks"]), "same_target": c.same_target,
            "paired_ok": bool(v["paired"] and v["paired"]["passed"]),
        })
    out["rows"] = rows
    out["baseline_label"] = "no validated edge"
    out["adopted"], out["adopted_label"] = adopt(rows, out["baseline_label"])
    out["prep5"] = prep5
    return out


def print_summary(res: dict) -> None:
    print(f"\nPhase 3 verdict {res['phase3_verdict']}: exploratory path {FAIL_PATH}. Final k={res['k']}")
    for name, g in [("ridge_h5_baseline", res["baseline"]["gates"])] + [(m, v["gates"]) for m, v in res["variants"].items()]:
        sc = (res["baseline"] if name == "ridge_h5_baseline" else res["variants"][name])["score"]
        print(f"\n{name}: mean IC {sc.mean_ic:.5f}  NW t {sc.nw_t:.3f}")
        for x in g:
            print(f"  {x.id} {x.name:<18} {x.value:<46} threshold {x.threshold:<34} {'PASS' if x.passed else 'FAIL'}")
    for m, v in res["variants"].items():
        print(f"\n{m} integrity")
        for c in v["checks"]:
            print(f"  {c.id} {c.name:<24} {'ok' if c.ok else 'FAILED':<7} {c.detail}")
        if v["paired"]:
            p = v["paired"]
            print(f"  paired vs baseline: mean diff {p['mean_diff']:.5f}  NW t {p['t']:.3f}  need >= {p['z_crit']:.3f}  {'PASS' if p['passed'] else 'FAIL'}")
    print(f"\nADOPTION: {res['adopted']} ({res['adopted_label']}) feeds Phase 4")


def _connect(cfg: dict) -> sqlite3.Connection:
    return bl._connect(cfg)


def write_report(res: dict) -> None:
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(report.render_35(res))
    print(f"report: {REPORT_PATH}")


def run(full_cfg: dict) -> int:
    cfg3, cfg35 = full_cfg["phase3"], full_cfg["phase3_5"]
    conn = _connect(full_cfg)
    try:
        trials.init(conn)
        cands = candidates(cfg3, cfg35)
        for mid in FAIL_PATH:
            trials.register(conn, fingerprint(cands[mid], cfg3), mid, {**cfg3, **cfg35})
        k = trials.count_k(conn)
        res = compute_all(conn, full_cfg, k)
        for mid, v in res["variants"].items():
            n = trials.store_predictions(conn, mid, bl.predictions_frame(v["prep"], {"pred": v["pred"]}))
            print(f"predictions stored: {n} rows for {mid}")
    finally:
        conn.close()
    write_report(res)
    print_summary(res)
    return 0 if all(c.ok for v in res["variants"].values() for c in v["checks"]) else 1


def regenerate(full_cfg: dict) -> int:
    conn = _connect(full_cfg)
    try:
        res = compute_all(conn, full_cfg, trials.count_k(conn))
    finally:
        conn.close()
    write_report(res)
    print_summary(res)
    return 0


def verify(full_cfg: dict) -> bool:
    cfg3 = full_cfg["phase3"]
    conn = _connect(full_cfg)
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  {'ok  ' if passed else 'FAIL'} {label}  {detail}")

    try:
        k = trials.count_k(conn)
        res = compute_all(conn, full_cfg, k)
        print(f"\nPhase 3 verdict {res['phase3_verdict']}: exploratory path {FAIL_PATH}")

        print("\n--- integrity I1, I4, I5, I6 per candidate ---")
        for m, v in res["variants"].items():
            for c in v["checks"]:
                check(f"{m} {c.id} {c.name}", c.ok, c.detail)

        print("\n--- horizon-10 purge gap ---")
        h10 = res["variants"]["ridge_h10"]
        per_fold = {f.test_year: len(f.train) for f in h10["prep"].folds}
        n_tickers = h10["prep"].frame["ticker"].nunique()
        year = h10["prep"].frame["date"].str[:4].astype(int).to_numpy()
        purged = {f.test_year: int((year < f.test_year).sum()) - per_fold[f.test_year] for f in h10["prep"].folds}
        check("purged rows per fold equal 10 per ticker", all(p == 10 * n_tickers for p in purged.values()),
              f"{sorted(set(purged.values()))} expected {10 * n_tickers}")

        print("\n--- trials and predictions ---")
        check("k equals scored variants plus baseline", k == len(FAIL_PATH) + 1, f"k={k}")
        for m, v in res["variants"].items():
            stored = conn.execute("SELECT COUNT(*) FROM predictions WHERE model_id = ?", (m,)).fetchone()[0]
            want = bl._expected_prediction_rows(conn, res["feature_cols"], v["cand"].horizon, bl.EXPECTED_TEST_YEARS[0])
            check(f"{m} predictions row count equals independent SQL count", stored == want, f"stored {stored}, SQL {want}")
            got = conn.execute(
                "SELECT prediction FROM predictions WHERE model_id = ? ORDER BY ticker, date", (m,)
            ).fetchall()
            fresh = bl.predictions_frame(v["prep"], {"pred": v["pred"]}).sort_values(["ticker", "date"])["prediction"].to_numpy()
            check(f"{m} stored predictions equal recomputed", np.array_equal(np.array([g[0] for g in got]), fresh))
        base_rows = bl.predictions_frame(res["prep5"], {"pred": res["baseline"]["pred"]}).sort_values(["ticker", "date"])
        stored_b = conn.execute(
            "SELECT prediction FROM predictions WHERE model_id = 'ridge_h5_baseline' ORDER BY ticker, date"
        ).fetchall()
        check("baseline predictions equal recomputed baseline",
              np.array_equal(np.array([g[0] for g in stored_b]), base_rows["prediction"].to_numpy()))

        print("\n--- adoption rule and report ---")
        print(f"       adoption outcome: {res['adopted']} ({res['adopted_label']})")
        again, _ = adopt(res["rows"], res["baseline_label"])
        check("adoption outcome re-derived from the rule", again == res["adopted"])
        check("report file regenerates byte-identical",
              REPORT_PATH.exists() and REPORT_PATH.read_text() == report.render_35(res))
    finally:
        conn.close()
    print(f"\nVERIFY: {'PASS' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico escalation comparisons (Phase 3.5)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="run candidates, store predictions + trials, write report")
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
