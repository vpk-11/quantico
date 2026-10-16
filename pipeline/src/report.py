"""Reporter: renders the Phase 3 report from a `compute_all` result dict.

Pure function of its input, no timestamps, so the same config and data always
produce byte-identical text.
"""

from __future__ import annotations


def _fmt_ic_table(year_ic: dict[int, float]) -> list[str]:
    lines = ["| year | mean IC |", "|---|---|"]
    lines += [f"| {y} | {v:.5f} |" for y, v in sorted(year_ic.items())]
    return lines


def render(res: dict) -> str:
    cfg, ex = res["cfg"], res["extras"]
    L = ["# Phase 3 report: baseline ridge, expanding walk-forward", ""]
    L += [f"Model: `{res['model_id']}`. Trial count k = {res['k']}.", ""]
    L += [f"## Verdict: {res['verdict']}", "", f"Label: {res['label']}", ""]

    L += ["## Config snapshot", "", "```"]
    L += [f"{key} = {val}" for key, val in cfg.items()]
    L += ["```", ""]

    L += ["## Folds", "",
          f"First usable year: {res['first_year']} (expected 2008).", "",
          "| test year | train first | train last | train rows | purged rows | test rows |",
          "|---|---|---|---|---|---|"]
    L += [f"| {f['test_year']} | {f['train_first']} | {f['train_last']} | {f['n_train']} | {f['n_purged']} | {f['n_test']} |"
          for f in res["folds"]]
    L.append("")

    L += ["## Gates", "", "| gate | name | value | threshold | result |", "|---|---|---|---|---|"]
    L += [f"| {g.id} | {g.name} | {g.value} | {g.threshold} | {'PASS' if g.passed else 'FAIL'} |" for g in res["gates"]]
    L.append("")

    L += ["## Integrity", "", "| test | name | result | detail |", "|---|---|---|---|"]
    L += [f"| {c.id} | {c.name} | {'ok' if c.ok else 'FAILED'} | {c.detail} |" for c in res["checks"]]
    L.append("")

    L += ["## Extras (no gate)", "", "### IC by year", ""] + _fmt_ic_table(ex["year_ic"])
    L += ["", f"IC in 2020: {ex['year_ic'].get(2020, float('nan')):.5f}. IC in 2022: {ex['year_ic'].get(2022, float('nan')):.5f}.", ""]
    L += [f"Hit rate (sign of prediction equals sign of target): {ex['hit_rate']:.4f}", "",
          f"Quintile spread (top fifth minus bottom fifth, daily mean 5-day return): {ex['quintile_mean']:.6f}, NW t {ex['quintile_t']:.3f}", "",
          f"Single-feature benchmark (mom_21 as prediction): mean IC {ex['bench_mean_ic']:.5f}, NW t {ex['bench_t']:.3f}", "",
          f"Daily IC distribution over {ex['n_days']} days: std {ex['ic_std']:.5f}", "",
          "| quantile | IC |", "|---|---|"]
    L += [f"| {q} | {v:.5f} |" for q, v in ex["ic_quantiles"].items()]
    L += ["", "### Ridge coefficients per fold (standardized features)", ""]
    cols = res["feature_cols"]
    L += ["| test year | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {y} | " + " | ".join(f"{c:+.6f}" for c in coef) + " |" for y, coef in ex["coefs"].items()]
    L.append("")
    return "\n".join(L)


def render_35(res: dict) -> str:
    """Phase 3.5 comparison report from `escalation.compute_all` output."""
    k = res["k"]
    models_ = [("ridge_h5_baseline", "baseline (" + res["baseline_label"] + ")", res["baseline"], None)]
    for mid, v in res["variants"].items():
        models_.append((mid, "exploratory", v, v["paired"]))

    L = ["# Phase 3.5 report: escalation comparisons", ""]
    L += [f"Phase 3 verdict: {res['phase3_verdict']}. Path: exploratory (3.5a random forest `rf_h5`, 3.5c 10-day horizon `ridge_h10`). "
          "3.5b recency weighting is skipped on this path.", "",
          f"Final trial count k = {k}. Every gate below is restated at this k, including the baseline's. "
          "All variant results are labelled exploratory.", ""]

    L += ["## Adoption outcome", "",
          f"Phase 4 input: `{res['adopted']}` ({res['adopted_label']}).", ""]
    L += ["| model | gates passed | integrity ok | paired test | qualifies |", "|---|---|---|---|---|"]
    for r in res["rows"]:
        paired = "n/a (different target)" if not r["same_target"] else ("PASS" if r["paired_ok"] else "FAIL")
        q = r["gates_ok"] and r["integrity_ok"] and (r["paired_ok"] or not r["same_target"])
        L.append(f"| {r['model_id']} | {'all' if r['gates_ok'] else 'not all'} | {'yes' if r['integrity_ok'] else 'no'} | {paired} | {'yes' if q else 'no'} |")
    L.append("")

    L += [f"## Comparison at k = {k}", "",
          "| model | label | mean IC | NW t | R2 vs naive | G1 | G2 | G3 | G4 | G5 | paired mean diff | paired NW t |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mid, label, m, paired in models_:
        sc = m["score"]
        gs = " | ".join("PASS" if g.passed else "FAIL" for g in m["gates"])
        pd_ = f"{paired['mean_diff']:.5f} | {paired['t']:.3f}" if paired else "n/a | n/a"
        L.append(f"| {mid} | {label} | {sc.mean_ic:.5f} | {sc.nw_t:.3f} | {sc.r2:.6f} | {gs} | {pd_} |")
    L += ["", "`ridge_h10` predicts a 10-day return, so its IC is not directly comparable to the 5-day models. "
          "It is judged against the gates only, with no paired test.", ""]

    L += ["## Gate values and thresholds", ""]
    for mid, _, m, paired in models_:
        L += [f"### {mid}", "", "| gate | name | value | threshold | result |", "|---|---|---|---|---|"]
        L += [f"| {g.id} | {g.name} | {g.value} | {g.threshold} | {'PASS' if g.passed else 'FAIL'} |" for g in m["gates"]]
        if paired:
            L.append(f"\nPaired test vs baseline: mean IC difference {paired['mean_diff']:.5f}, NW t {paired['t']:.3f}, "
                     f"needs >= {paired['z_crit']:.3f}: {'PASS' if paired['passed'] else 'FAIL'}")
        L.append("")

    L += ["## Integrity (I1, I4, I5, I6 per new model)", "", "| model | test | name | result | detail |", "|---|---|---|---|---|"]
    for mid, v in res["variants"].items():
        L += [f"| {mid} | {c.id} | {c.name} | {'ok' if c.ok else 'FAILED'} | {c.detail} |" for c in v["checks"]]
    L.append("")

    cols = res["feature_cols"]
    L += ["## Random forest feature importances per fold", "",
          "Impurity-based importances are biased toward high-variance features, so read them as a rough guide only.", "",
          "| test year | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    L += [f"| {y} | " + " | ".join(f"{x:.4f}" for x in imp) + " |"
          for y, imp in res["variants"]["rf_h5"]["importances"].items()]
    L += ["", "## Config snapshot", "", "```"]
    L += [f"{key} = {val}" for key, val in res["cfg35"].items()]
    L += ["```", ""]
    return "\n".join(L)
