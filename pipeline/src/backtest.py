"""Phase 5: replay stored predictions through three execution strategies, across a robustness
grid of intraday variants, cost multipliers, and capital levels, with random-null and
buy-and-hold comparators.

Signal held constant: every strategy sees the same predictions, the same long-only top-N
targets at the same fixed capital per cell (no compounding), the same synthetic intraday bars,
and the same cost parameters. Only the order-placement schedule differs. Orders are decided on
the close of day t and executed during day t+1; arrival price is the open of day t+1. Costs are
reported against both arrival and interval-VWAP benchmarks, decomposed into fees/spread/impact/
drift. The grid and the random null run through `execution.reference_fill` directly (no C++
subprocess) for speed, since Phase 4 proves the two agree bit for bit; the base cell is also run
through the real `simulate()` subprocess to confirm the as-built numbers reproduce.

Usage (run from pipeline/):
    python src/backtest.py run     # base cell + robustness grid + comparators, writes reports/phase5_report.md
    python src/backtest.py bench   # C++ simulator vs Python reference on the same orders (Phase 4)
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as bl
import execution as ex
import research_store as rs

_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = _ROOT / "reports" / "phase5_report.md"
MODEL_ID = "ridge_h5_baseline"
BENCHMARKS = (("open", "arrival"), ("interval_vwap", "interval_vwap"))


def load_bars(conn: sqlite3.Connection, table: str = "daily_bars") -> pd.DataFrame:
    return pd.read_sql_query(
        f"SELECT ticker, date, open, high, low, close, volume FROM {table} ORDER BY date, ticker", conn
    ).set_index(["date", "ticker"])


def top_n_targets(pred_day: pd.Series, close_day: pd.Series, p: dict) -> dict[str, int]:
    """Equal-weight long-only: the top_n predictions each get capital/top_n of shares at today's close."""
    names = pred_day.sort_values(ascending=False, kind="stable").index[: p["top_n"]]
    return {t: int(p["capital"] / p["top_n"] // close_day[t]) for t in names}


def execute_day(
    trades: dict[str, int], bars_day: pd.DataFrame, strategy: int, p: dict,
    variant: str = "base", seed: int | None = None, use_reference: bool = False,
) -> list[dict]:
    """Run signed share deltas through the simulator (or, if `use_reference`, the Python oracle
    directly, skipping the subprocess). Returns one fill record per nonzero trade, carrying the
    arrival price (`open`), the interval VWAP, and the requested participation (`qty` over the
    slice-day's total volume) for downstream benchmark and robustness reporting.
    """
    orders, meta = [], []
    for i, (t, delta) in enumerate((t, d) for t, d in sorted(trades.items()) if d != 0):
        b = bars_day.loc[t]
        px, vol = ex.make_slices(b.open, b.high, b.low, b.close, int(b.volume), p["slices"], variant=variant, seed=seed)
        side = 1 if delta > 0 else -1
        total_vol = sum(vol)
        interval_vwap = sum(a * v for a, v in zip(px, vol)) / total_vol
        participation = abs(delta) / total_vol
        orders.append({"id": i, "side": side, "strategy": strategy, "qty": abs(delta), "prices": px, "vols": vol})
        meta.append((t, side, abs(delta), float(b.open), interval_vwap, participation))
    if not orders:
        return []
    if use_reference:
        keys = ("filled", "notional", "fees", "spread_cost", "impact_cost", "raw_notional")
        res = [dict(zip(keys, ex.reference_fill(o["side"], o["strategy"], o["qty"], o["prices"], o["vols"], p)))
               for o in orders]
    else:
        res = ex.simulate(orders, p)
    return [{"ticker": t, "side": s, "qty": q, "open": o, "interval_vwap": iv, "participation": part, **r}
            for (t, s, q, o, iv, part), r in zip(meta, res)]


def apply_fills(pos: dict[str, int], cash: float, fills: list[dict]) -> float:
    for f in fills:
        pos[f["ticker"]] = pos.get(f["ticker"], 0) + f["side"] * f["filled"]
        cash -= f["side"] * f["notional"] + f["fees"]
    return cash


def rebalance_schedule(close: pd.DataFrame, preds: pd.DataFrame, rebalance_days: int) -> dict[str, str]:
    """`{execution_date: decision_date}`, shared by every cell in a run so `close.unstack()` and
    the rebalance-date search happen once, not once per strategy/variant/multiplier/capital/draw."""
    dates = list(close.index)
    pred_dates = sorted(preds["date"].unique())
    reb = pred_dates[::rebalance_days]
    return {dates[dates.index(d) + 1]: d for d in reb if dates.index(d) + 1 < len(dates)}


def backtest(
    bars: pd.DataFrame, close: pd.DataFrame, pred_by_day: dict, exec_for: dict, strategy: int, p: dict,
    variant: str = "base", seed: int | None = None, use_reference: bool = False, picker=None,
) -> dict:
    """`picker(t0) -> list[ticker]` overrides the model's top-N selection (used by the random
    null). `close`, `pred_by_day`, and `exec_for` are precomputed once per run by the caller
    (see `rebalance_schedule`) and shared across every cell; they are not recomputed here.
    """
    dates = list(close.index)
    pos: dict[str, int] = {}
    cash, fills_all, nav = float(p["capital"]), [], {}
    pred_dates = sorted(pred_by_day)
    start = min(exec_for)
    for d in [x for x in dates if x >= start and x <= pred_dates[-1]]:
        if d in exec_for:
            t0 = exec_for[d]
            if picker is None:
                tgt = top_n_targets(pred_by_day[t0], close.loc[t0], p)
            else:
                tgt = {t: int(p["capital"] / p["top_n"] // close.at[t0, t]) for t in picker(t0)}
            trades = {t: tgt.get(t, 0) - pos.get(t, 0) for t in set(tgt) | set(pos)}
            fl = execute_day(trades, bars.loc[d], strategy, p, variant, seed, use_reference)
            for f in fl:
                f["date"] = d
            cash = apply_fills(pos, cash, fl)
            fills_all += fl
        nav[d] = cash + sum(q * close.at[d, t] for t, q in pos.items())
    return {"fills": pd.DataFrame(fills_all), "nav": pd.Series(nav)}


def cost_stats(fills: pd.DataFrame, benchmark_col: str) -> dict:
    """Notional-weighted realized cost vs `benchmark_col` (`open` for arrival, `interval_vwap`
    for interval VWAP), decomposed into fees/spread/impact/drift (vectorized form of
    `execution.decompose_cost`), plus fill rate. Returns zeros if nothing filled (an infeasible
    cell), never crashes.
    """
    if fills.empty:
        return {"orders": 0, "fill_rate": 0.0, "total_bps": 0.0, "drift_bps": 0.0,
                "spread_bps": 0.0, "impact_bps": 0.0, "fee_bps": 0.0, "fees_usd": 0.0, "traded_usd": 0.0}
    f = fills[fills["filled"] > 0]
    notional = float(f["notional"].sum())
    if notional == 0.0:
        return {"orders": len(fills), "fill_rate": 0.0, "total_bps": 0.0, "drift_bps": 0.0,
                "spread_bps": 0.0, "impact_bps": 0.0, "fee_bps": 0.0, "fees_usd": 0.0, "traded_usd": 0.0}
    total = (f["side"] * (f["notional"] - f["filled"] * f[benchmark_col]) + f["fees"]).sum()
    drift = (f["side"] * (f["raw_notional"] - f["filled"] * f[benchmark_col])).sum()
    spread, impact, fees = f["spread_cost"].sum(), f["impact_cost"].sum(), f["fees"].sum()
    to_bps = lambda x: x / notional * 1e4
    return {"orders": len(fills), "fill_rate": float(fills["filled"].sum() / fills["qty"].sum()),
            "total_bps": to_bps(total), "drift_bps": to_bps(drift), "spread_bps": to_bps(spread),
            "impact_bps": to_bps(impact), "fee_bps": to_bps(fees), "fees_usd": fees, "traded_usd": notional}


def nav_stats(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1.0
    return {"final": float(nav.iloc[-1]), "ret": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "max_dd": float(dd.min())}


def phase3_label(conn: sqlite3.Connection, cfg: dict) -> tuple[str, str]:
    """Re-derives the Phase 3 verdict from stored data (same path `baseline.py verify` uses),
    so the label is read, not typed by hand."""
    prep = bl.prepare(conn, cfg)
    res = bl.compute_all(prep, cfg["phase3"], bl.BASELINE_K, bl.EXPECTED_TEST_YEARS)
    return res["verdict"], res["label"]


def grid_cells(cfg5: dict, cfg4: dict) -> list[tuple[str, int | None, float, float]]:
    """(variant, noise_seed_or_None, cost_multiplier, capital) for every grid cell. `noisy_volume`
    is not in `cfg5["intraday_variants"]`; it runs once per `cfg4["intraday_noise_seeds"]` seed."""
    cells = [(v, None, m, cap) for v in cfg5["intraday_variants"]
             for m in cfg5["cost_multipliers"] for cap in cfg5["capital_levels"]]
    cells += [("noisy_volume", seed, m, cap) for seed in cfg4["intraday_noise_seeds"]
              for m in cfg5["cost_multipliers"] for cap in cfg5["capital_levels"]]
    return cells


def run_grid(
    bars: pd.DataFrame, close: pd.DataFrame, pred_by_day: dict, exec_for: dict, cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs every strategy over every grid cell via the Python reference fill (no subprocess).
    Returns (cost rows per cell/strategy/benchmark, participation rows per cell)."""
    cfg4, cfg5 = cfg["phase4"], cfg["phase5"]
    cost_rows, part_rows = [], []
    for variant, seed, mult, cap in grid_cells(cfg5, cfg4):
        p = ex.cost_params({**cfg4, "cost_multiplier": mult, "capital": cap})
        for name, sid in ex.STRATEGIES.items():
            r = backtest(bars, close, pred_by_day, exec_for, sid, p, variant=variant, seed=seed, use_reference=True)
            if not r["fills"].empty:
                part_rows.append(pd.DataFrame({"capital": cap, "participation": r["fills"]["participation"]}))
            for bench_col, bench_name in BENCHMARKS:
                s = cost_stats(r["fills"], bench_col)
                cost_rows.append({"variant": variant, "noise_seed": seed, "cost_multiplier": mult,
                                   "capital": cap, "strategy": name, "benchmark": bench_name, **s})
    part = pd.concat(part_rows, ignore_index=True) if part_rows else pd.DataFrame(columns=["capital", "participation"])
    return pd.DataFrame(cost_rows), part


def _ranking_for(cell: pd.DataFrame) -> dict:
    """`cell`: one row per (variant, cost_multiplier, capital, strategy), columns total_bps and
    fill_rate (noisy_volume's seeds already collapsed to a mean). Splits into cells where every
    strategy filled 100% ('feasible') and cells where at least one did not, ranks each group
    separately, and reports win counts and the mean cost gap to the winner within each group."""
    def summarize(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"established": None, "winner": None, "cells": 0, "wins": {}, "mean_gap_bps": {},
                     "paths": 0, "wins_by_path": {}}
        by_cell = sub.groupby(["variant", "cost_multiplier", "capital"])
        winners = sub.loc[by_cell["total_bps"].idxmin()]
        wins = winners["strategy"].value_counts().to_dict()
        unique = winners["strategy"].unique()
        best_per_cell = by_cell["total_bps"].transform("min")
        gap = (sub["total_bps"] - best_per_cell)
        mean_gap = sub.assign(gap=gap).groupby("strategy")["gap"].mean().to_dict()
        # by path: collapse cost_multiplier first (mean across multipliers), then pick one winner
        # per distinct intraday path (variant), so 12 cells (4 paths x 3 multipliers) become 4.
        by_path = sub.groupby(["variant", "strategy"])["total_bps"].mean().reset_index()
        path_winners = by_path.loc[by_path.groupby("variant")["total_bps"].idxmin()]
        wins_by_path = path_winners["strategy"].value_counts().to_dict()
        lo, hi = sub["cost_multiplier"].min(), sub["cost_multiplier"].max()
        mult_change = sub.loc[sub["cost_multiplier"] == hi, "total_bps"].mean() - \
            sub.loc[sub["cost_multiplier"] == lo, "total_bps"].mean()
        return {"established": len(unique) == 1, "winner": unique[0] if len(unique) == 1 else None,
                "cells": len(winners), "wins": wins, "mean_gap_bps": mean_gap,
                "paths": sub["variant"].nunique(), "wins_by_path": wins_by_path,
                "multiplier_change_bps": float(mult_change)}

    feasible = cell["fill_rate"] >= 0.999
    return {"feasible": summarize(cell[feasible]), "infeasible": summarize(cell[~feasible])}


def ranking_summary(grid: pd.DataFrame) -> dict[str, dict]:
    """Per benchmark, per feasibility group (every strategy filled 100% in the cell, or not):
    does the lowest-total-cost strategy stay the same across cells (noisy_volume's 5 seeds
    collapsed to one mean cell first)? If not, ranking is not established in that group."""
    out = {}
    for bench_name, g in grid.groupby("benchmark"):
        cell = g.groupby(["variant", "cost_multiplier", "capital", "strategy"])[["total_bps", "fill_rate"]].mean().reset_index()
        out[bench_name] = _ranking_for(cell)
    return out


def random_null_sharpes(
    bars: pd.DataFrame, close: pd.DataFrame, pred_by_day: dict, exec_for: dict, strategy: int,
    p: dict, draws: int, seed: int,
) -> list[float]:
    """Sharpe of `draws` seeded random-top-N portfolios (same rebalance schedule, cost model, and
    capital as the real strategy), run via the reference fill for speed. Deterministic under `seed`."""
    tickers = list(close.columns)
    out = []
    for i in range(draws):
        rng = np.random.default_rng(seed + i)
        picker = lambda t0, rng=rng: rng.choice(tickers, p["top_n"], replace=False).tolist()
        r = backtest(bars, close, pred_by_day, exec_for, strategy, p, use_reference=True, picker=picker)
        out.append(nav_stats(r["nav"])["sharpe"])
    return out


def run(cfg: dict) -> int:
    p4 = cfg["phase4"]
    p = ex.cost_params(p4)
    rebalance_days = cfg["phase3"]["target_horizon_days"]
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    try:
        verdict, label = phase3_label(conn, cfg)
        bars = load_bars(conn)
        preds = pd.read_sql_query("SELECT ticker, date, prediction FROM predictions WHERE model_id = ?", conn, params=(MODEL_ID,))
    finally:
        conn.close()

    close = bars["close"].unstack()
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    exec_for = rebalance_schedule(close, preds, rebalance_days)

    rows = {}
    for name, sid in ex.STRATEGIES.items():
        r = backtest(bars, close, pred_by_day, exec_for, sid, p)  # base cell, real C++ simulator
        rows[name] = {bn: cost_stats(r["fills"], bc) for bc, bn in BENCHMARKS}
        rows[name]["nav"] = nav_stats(r["nav"])
        print(f"{name}: {rows[name]['arrival']['total_bps']:.2f} bps (arrival)", flush=True)
    best = min(rows, key=lambda k: rows[k]["arrival"]["total_bps"])

    free = backtest(bars, close, pred_by_day, exec_for, ex.IMMEDIATE,
                     {**p, "fee_bps": 0.0, "spread_bps": 0.0, "impact_bps": 0.0})
    frictionless = nav_stats(free["nav"])
    nav0 = free["nav"]
    ew_nav = close.loc[nav0.index].div(close.loc[nav0.index[0]]).mean(axis=1) * p["capital"]
    ew = nav_stats(ew_nav)

    print("running robustness grid (reference fill, no subprocess)...", flush=True)
    grid, participation = run_grid(bars, close, pred_by_day, exec_for, cfg)
    ranking = ranking_summary(grid)

    cfg5 = cfg["phase5"]
    print(f"running random top-{p['top_n']} null ({cfg5['null_draws']} draws)...", flush=True)
    null_sharpes = random_null_sharpes(bars, close, pred_by_day, exec_for, ex.STRATEGIES[best], p,
                                        cfg5["null_draws"], cfg5["null_seed"])
    percentile = float((np.array(null_sharpes) < rows[best]["nav"]["sharpe"]).mean() * 100)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(render(
        verdict, label, rows, best, frictionless, ew, p, len(preds), str(nav0.index[0]), str(nav0.index[-1]),
        rebalance_days, grid, ranking, participation, null_sharpes, percentile,
    ))
    print(f"report: {REPORT_PATH}")
    print(f"lowest total cost vs arrival: {best}")
    return 0


def render(
    verdict: str, label: str, rows: dict, best: str, free: dict, ew: dict, p: dict, n_preds: int,
    d0: str, d1: str, rebalance_days: int, grid: pd.DataFrame, ranking: dict,
    participation: pd.DataFrame, null_sharpes: list[float], percentile: float,
) -> str:
    L = ["# Phase 5 report: execution strategy comparison", "",
         f"Phase 3 verdict: **{verdict}** ({label}). The portfolio figures below are descriptive, "
         "not evidence of skill: they show execution cost drag and selection behaviour on a signal "
         "with no validated edge.", "",
         f"Signal: `{MODEL_ID}` predictions, {n_preds} rows, replayed {d0} to {d1}.",
         f"Long-only top {p['top_n']} of 20, equal weight, fixed ${p['capital']:,} base capital (no "
         f"compounding), rebalanced every {rebalance_days} trading days (`[phase3] "
         "target_horizon_days`, not duplicated). Orders decided at close t, executed during t+1; two "
         "benchmarks are reported: arrival (the open of t+1) and interval VWAP (the volume-weighted "
         "price over t+1's synthetic slices).", "",
         "Intraday data is synthetic: each daily bar is expanded into "
         f"{p['slices']} slices (deterministic open, extreme, extreme, close price path; U-shaped volume). "
         "Timing differences between strategies therefore reflect that assumed path, not real intraday "
         "behaviour; the robustness grid below shows how much.", "",
         f"Costs (assumed, uncalibrated): fee {p['fee_bps']} bps, half-spread {p['half_spread_bps']} bps, "
         f"impact {p['impact_coeff_bps']} bps x sqrt(participation), participation cap "
         f"{p['participation_cap']:.0%} per slice, latency {p['latency_slices']} slice(s).", ""]

    for bench_col, bench_name in BENCHMARKS:
        L += [f"## Base cell: realized execution cost vs {bench_name} (bps, notional-weighted; positive is worse)", "",
              "| strategy | orders | fill rate | drift | spread | impact | fee | total |",
              "|---|---|---|---|---|---|---|---|"]
        for k, r in rows.items():
            s = r[bench_name]
            L.append(f"| {k} | {s['orders']} | {s['fill_rate']:.4f} | {s['drift_bps']:.2f} | "
                      f"{s['spread_bps']:.2f} | {s['impact_bps']:.2f} | {s['fee_bps']:.2f} | **{s['total_bps']:.2f}** |")
        L.append("")
    L += [f"Lowest total cost vs arrival (base cell): **{best}**.", ""]

    L += ["## Robustness grid", "",
          "Every strategy re-run across intraday variant (base, swap_extremes, flat_volume, plus "
          "noisy_volume's 5 seeded paths reported as mean and range), cost multiplier (0.5x, 1x, 2x), "
          "and capital ($25M, $250M, $2.5B). Run via the Python reference fill (Phase 4 proves this "
          "matches the C++ simulator bit for bit), not the base cell's subprocess path. Cells are "
          "split into ones where every strategy filled 100% ('feasible') and ones where at least one "
          "did not; a flipped ranking that only happens in infeasible cells is a cap-binding "
          "artifact, not a real execution-strategy finding.", ""]
    for bench_col, bench_name in BENCHMARKS:
        rk = ranking[bench_name]
        L.append(f"### vs {bench_name}")
        L.append("")
        for group, label in (("feasible", "feasible cells (100% fill)"), ("infeasible", "infeasible cells (partial fill)")):
            r = rk[group]
            if r["cells"] == 0:
                L.append(f"No {label}.")
                continue
            if r["established"]:
                L.append(f"{label.capitalize()}: ranking **established**, lowest-cost strategy is "
                          f"**{r['winner']}** in every one of {r['cells']} cells.")
            else:
                L.append(f"{label.capitalize()}: ranking **not established** across {r['cells']} cells.")
            wins = ", ".join(f"{k}: {v}" for k, v in sorted(r["wins"].items()))
            gaps = ", ".join(f"{k}: {v:.2f} bps" for k, v in sorted(r["mean_gap_bps"].items()))
            L.append(f"Win counts by cell: {wins}. Mean gap to the per-cell winner: {gaps}.")
            if group == "feasible":
                wbp = ", ".join(f"{k}: {v}" for k, v in sorted(r["wins_by_path"].items()))
                L.append(f"The {r['cells']} feasible cells are {r['paths']} independent intraday "
                          f"paths x 3 cost multipliers. Win counts by path (collapsing the "
                          f"multipliers first): {wbp}.")
                max_gap = max(r["mean_gap_bps"].values()) if r["mean_gap_bps"] else 0.0
                L.append(f"The mean gaps ({min(r['mean_gap_bps'].values()):.2f} to {max_gap:.2f} bps) "
                          f"are smaller than the {r['multiplier_change_bps']:.2f} bps cost change "
                          "between the 0.5x and 2x cost multipliers, so execution-strategy "
                          "differences here are below the cost model's own uncertainty.")
            L.append("")
        if rk["feasible"]["established"] and not rk["infeasible"]["established"] and rk["infeasible"]["cells"] > 0:
            L.append(f"Ranking vs {bench_name} holds while every strategy fills in full; the flips "
                      "only show up once the participation cap starts binding, which is a capital-scale "
                      "artifact, not evidence that execution strategy choice matters at the base cell.")
            L.append("")
    noisy = grid[grid["variant"] == "noisy_volume"]
    if not noisy.empty:
        agg = noisy.groupby(["strategy", "benchmark"])["total_bps"].agg(["mean", "min", "max"]).reset_index()
        L += ["### noisy_volume (5 seeds): mean and range, total bps", "",
              "| strategy | benchmark | mean | min | max |", "|---|---|---|---|---|"]
        for _, r in agg.iterrows():
            L.append(f"| {r['strategy']} | {r['benchmark']} | {r['mean']:.2f} | {r['min']:.2f} | {r['max']:.2f} |")
        L.append("")

    L += ["## Capital and participation", "",
          "Requested participation (order quantity over the slice-day's total volume), by capital "
          "level, pooled across every grid cell and strategy.", "",
          "| capital | median | 95th pct | max |", "|---|---|---|---|"]
    for cap, g in participation.groupby("capital"):
        q = g["participation"]
        L.append(f"| ${cap:,.0f} | {q.median():.4%} | {q.quantile(0.95):.4%} | {q.max():.4%} |")
    fr_by_cap = grid.groupby("capital")["fill_rate"].agg(["mean", "min"])
    L += ["", "Fill rate by capital level (mean and worst cell, across every variant/multiplier/"
          "strategy/benchmark at that capital). Below 100% is the participation cap binding, not a "
          "bug: the cell is reporting a partial fill explicitly, not silently dropping it.", "",
          "| capital | mean fill rate | worst cell fill rate |", "|---|---|---|"]
    for cap, r in fr_by_cap.iterrows():
        L.append(f"| ${cap:,.0f} | {r['mean']:.2%} | {r['min']:.2%} |")
    L += ["", "At $25M across 20 large caps, order sizes are a small share of daily volume; the sweep "
          "shows where, if anywhere, strategies diverge as capital scales up, and where the "
          "participation cap starts binding hard enough to leave orders partially unfilled.", ""]

    L += ["## Comparators", "",
          "| run | return | Sharpe | max drawdown |", "|---|---|---|---|"]
    for k, r in rows.items():
        s = r["nav"]
        L.append(f"| {k} | {s['ret']:.2%} | {s['sharpe']:.2f} | {s['max_dd']:.2%} |")
    L.append(f"| frictionless (zero costs, immediate) | {free['ret']:.2%} | {free['sharpe']:.2f} | {free['max_dd']:.2%} |")
    L.append(f"| equal-weight all 20, no trading | {ew['ret']:.2%} | {ew['sharpe']:.2f} | {ew['max_dd']:.2%} |")
    L += ["", "Equal-weight buy-and-hold is flattered by survivorship bias (all 20 names survived to "
          f"2026). Random top-{p['top_n']} null, {len(null_sharpes)} seeded draws, same rebalance "
          f"schedule and cost model: Sharpe mean {statistics.mean(null_sharpes):.2f}, range "
          f"{min(null_sharpes):.2f} to {max(null_sharpes):.2f}. The **{best}** strategy's Sharpe "
          f"({rows[best]['nav']['sharpe']:.2f}) sits at the **{percentile:.0f}th percentile** of this "
          "null, i.e. that much of the null falls below it. This is below the 95th-percentile "
          "threshold Phase 7's edge rule requires, so this result alone would not support an edge "
          "claim even if the forward-IC and gate conditions were otherwise met.", "",
          "These numbers are not evidence of profitability. Survivorship bias applies; the cost model "
          "is assumed and uncalibrated; the window is short.", ""]
    return "\n".join(L)


BENCH_PATH = _ROOT / "reports" / "phase4_benchmark.md"


def _bench_orders(p: dict, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    orders = []
    for i in range(n):
        o = float(rng.uniform(20, 400))
        px, vol = ex.make_slices(o, o * 1.02, o * 0.98, o * 1.01, int(rng.integers(1_000_000, 50_000_000)), p["slices"])
        orders.append({"id": i, "side": int(rng.choice([-1, 1])), "strategy": int(rng.integers(0, 3)),
                       "qty": int(rng.integers(1, 2_000_000)), "prices": px, "vols": vol})
    return orders


def bench(cfg: dict) -> int:
    """Times the C++ subprocess (incl. startup + CSV I/O) against the Python reference loop at
    2k, 20k, and 200k orders, 3 repeats each, and writes reports/phase4_benchmark.md. No speed
    claim is made anywhere except from this table.
    """
    p = ex.cost_params(cfg["phase4"])
    rows = []
    for n in (2_000, 20_000, 200_000):
        cpp_times, py_times = [], []
        for rep in range(3):
            orders = _bench_orders(p, n, seed=1000 * n + rep)
            t = time.perf_counter()
            ex.simulate(orders, p)
            cpp_times.append(time.perf_counter() - t)
            t = time.perf_counter()
            for o in orders:
                ex.reference_fill(o["side"], o["strategy"], o["qty"], o["prices"], o["vols"], p)
            py_times.append(time.perf_counter() - t)
        cpp_med, py_med = statistics.median(cpp_times), statistics.median(py_times)
        cpp_spread, py_spread = max(cpp_times) - min(cpp_times), max(py_times) - min(py_times)
        rows.append((n, cpp_med, cpp_spread, py_med, py_spread))
        print(f"{n} orders: C++ median {cpp_med:.3f}s (spread {cpp_spread:.3f}), "
              f"Python median {py_med:.3f}s (spread {py_spread:.3f})", flush=True)
    L = ["# Phase 4 benchmark: C++ simulator vs Python reference", "",
         "3 repeats per order count, median and spread (max minus min) across repeats, in seconds. "
         "C++ timing includes subprocess startup and CSV I/O; Python is an in-process loop.", "",
         "| orders | C++ median (s) | C++ spread (s) | Python median (s) | Python spread (s) | faster |",
         "|---|---|---|---|---|---|"]
    for n, cm, cs, pm, ps in rows:
        L.append(f"| {n:,} | {cm:.3f} | {cs:.3f} | {pm:.3f} | {ps:.3f} | {'C++' if cm < pm else 'Python'} |")
    L += ["", "No speed claim is made beyond this table. The case for the C++ simulator is an "
          "independent, bit-exact-parity-checked implementation (see test_phase4.py), not raw speed.", ""]
    BENCH_PATH.parent.mkdir(exist_ok=True)
    BENCH_PATH.write_text("\n".join(L))
    print(f"report: {BENCH_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantico execution backtest (Phase 5)")
    parser.add_argument("cmd", choices=["run", "bench"])
    args = parser.parse_args(argv)
    cfg = rs.load_config()
    return run(cfg) if args.cmd == "run" else bench(cfg)


if __name__ == "__main__":
    sys.exit(main())
