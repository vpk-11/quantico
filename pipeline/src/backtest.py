"""Phase 5: replay stored predictions through the C++ simulator under three execution strategies.

Signal held constant: every strategy sees the same predictions, the same long-only top-N
targets at the same fixed capital (no compounding), the same synthetic intraday bars, and
the same cost parameters. Only the order-placement schedule differs. Orders are decided on
the close of day t and executed during day t+1; arrival price is the open of day t+1.

Usage (run from pipeline/):
    python src/backtest.py run     # three strategies + frictionless + equal-weight benchmark, writes reports/phase5_report.md
    python src/backtest.py bench   # C++ simulator vs Python reference on the same orders
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

import execution as ex
import research_store as rs

_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = _ROOT / "reports" / "phase5_report.md"
MODEL_ID = "ridge_h5_baseline"


def load_bars(conn: sqlite3.Connection, table: str = "daily_bars") -> pd.DataFrame:
    return pd.read_sql_query(
        f"SELECT ticker, date, open, high, low, close, volume FROM {table} ORDER BY date, ticker", conn
    ).set_index(["date", "ticker"])


def top_n_targets(pred_day: pd.Series, close_day: pd.Series, p: dict) -> dict[str, int]:
    """Equal-weight long-only: the top_n predictions each get capital/top_n of shares at today's close."""
    names = pred_day.sort_values(ascending=False, kind="stable").index[: p["top_n"]]
    return {t: int(p["capital"] / p["top_n"] // close_day[t]) for t in names}


def execute_day(trades: dict[str, int], bars_day: pd.DataFrame, strategy: int, p: dict) -> list[dict]:
    """Run signed share deltas through the simulator. Returns one fill record per nonzero trade."""
    orders, meta = [], []
    for i, (t, delta) in enumerate((t, d) for t, d in sorted(trades.items()) if d != 0):
        b = bars_day.loc[t]
        px, vol = ex.make_slices(b.open, b.high, b.low, b.close, int(b.volume), p["slices"])
        side = 1 if delta > 0 else -1
        orders.append({"id": i, "side": side, "strategy": strategy, "qty": abs(delta), "prices": px, "vols": vol})
        meta.append((t, side, abs(delta), float(b.open)))
    if not orders:
        return []
    res = ex.simulate(orders, p)
    return [{"ticker": t, "side": s, "qty": q, "open": o, **r} for (t, s, q, o), r in zip(meta, res)]


def apply_fills(pos: dict[str, int], cash: float, fills: list[dict]) -> float:
    for f in fills:
        pos[f["ticker"]] = pos.get(f["ticker"], 0) + f["side"] * f["filled"]
        cash -= f["side"] * f["notional"] + f["fees"]
    return cash


def backtest(bars: pd.DataFrame, preds: pd.DataFrame, strategy: int, p: dict) -> dict:
    close = bars["close"].unstack()
    dates = list(close.index)
    pred_dates = sorted(preds["date"].unique())
    reb = pred_dates[:: p["rebalance_days"]]
    pos: dict[str, int] = {}
    cash, fills_all, nav = float(p["capital"]), [], {}
    exec_for = {dates[dates.index(d) + 1]: d for d in reb if dates.index(d) + 1 < len(dates)}
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    start = min(exec_for)
    for d in [x for x in dates if x >= start and x <= pred_dates[-1]]:
        if d in exec_for:
            t0 = exec_for[d]
            tgt = top_n_targets(pred_by_day[t0], close.loc[t0], p)
            trades = {t: tgt.get(t, 0) - pos.get(t, 0) for t in set(tgt) | set(pos)}
            fl = execute_day(trades, bars.loc[d], strategy, p)
            for f in fl:
                f["date"] = d
            cash = apply_fills(pos, cash, fl)
            fills_all += fl
        nav[d] = cash + sum(q * close.at[d, t] for t, q in pos.items())
    return {"fills": pd.DataFrame(fills_all), "nav": pd.Series(nav)}


def cost_stats(fills: pd.DataFrame) -> dict:
    """Notional-weighted realized cost versus arrival (open), in bps, plus fill rate."""
    f = fills[fills["filled"] > 0]
    avg = f["notional"] / f["filled"]
    slip = (f["side"] * (avg / f["open"] - 1.0) * 1e4 * f["notional"]).sum() / f["notional"].sum()
    fee = f["fees"].sum() / f["notional"].sum() * 1e4
    return {"orders": len(fills), "fill_rate": fills["filled"].sum() / fills["qty"].sum(),
            "shortfall_bps": slip, "fee_bps": fee, "total_bps": slip + fee,
            "fees_usd": f["fees"].sum(), "traded_usd": f["notional"].sum()}


def nav_stats(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1.0
    return {"final": float(nav.iloc[-1]), "ret": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "max_dd": float(dd.min())}


def run(cfg: dict) -> int:
    p = ex.cost_params(cfg["phase4"])
    conn = sqlite3.connect(_ROOT / cfg["data"]["db_path"])
    try:
        bars = load_bars(conn)
        preds = pd.read_sql_query("SELECT ticker, date, prediction FROM predictions WHERE model_id = ?", conn, params=(MODEL_ID,))
    finally:
        conn.close()
    rows = {}
    for name, sid in ex.STRATEGIES.items():
        r = backtest(bars, preds, sid, p)
        rows[name] = {**cost_stats(r["fills"]), **nav_stats(r["nav"])}
        print(f"{name}: {rows[name]['total_bps']:.2f} bps", flush=True)
    free = backtest(bars, preds, ex.IMMEDIATE, {**p, "fee_bps": 0.0, "spread_bps": 0.0, "impact_bps": 0.0})
    frictionless = nav_stats(free["nav"])
    close = bars["close"].unstack()
    nav0 = free["nav"]
    ew = close.loc[nav0.index].div(close.loc[nav0.index[0]]).mean(axis=1) * p["capital"]
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(render(rows, frictionless, nav_stats(ew), p, len(preds), str(nav0.index[0]), str(nav0.index[-1])))
    print(f"report: {REPORT_PATH}")
    best = min(rows, key=lambda k: rows[k]["total_bps"])
    print(f"lowest total cost: {best}")
    return 0


def render(rows: dict, free: dict, ew: dict, p: dict, n_preds: int, d0: str, d1: str) -> str:
    L = ["# Phase 5 report: execution strategy comparison", "",
         f"Signal: `{MODEL_ID}` predictions (label: no validated edge), {n_preds} rows, replayed {d0} to {d1}.",
         f"Long-only top {p['top_n']} of 20, equal weight, fixed ${p['capital']:,} (no compounding), rebalanced every "
         f"{p['rebalance_days']} trading days. Orders decided at close t, executed during t+1; arrival price is the open of t+1.", "",
         "Intraday data is synthetic: each daily bar is expanded into "
         f"{p['slices']} slices (deterministic open, extreme, extreme, close price path; U-shaped volume). "
         "Timing differences between strategies therefore reflect that assumed path, not real intraday behaviour.", "",
         f"Costs: fee {p['fee_bps']} bps, half-spread {p['spread_bps']} bps, impact {p['impact_bps']} bps x sqrt(participation), "
         f"max participation {p['max_participation']:.0%} per slice, latency {p['latency_slices']} slice(s).", "",
         "## Realized execution cost (notional-weighted, bps versus arrival; positive is worse)", "",
         "| strategy | orders | fill rate | shortfall bps | fee bps | total bps | fees USD |", "|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['orders']} | {r['fill_rate']:.4f} | {r['shortfall_bps']:.2f} | {r['fee_bps']:.2f} | "
                 f"**{r['total_bps']:.2f}** | {r['fees_usd']:,.0f} |")
    best = min(rows, key=lambda k: rows[k]["total_bps"])
    L += ["", f"Lowest total cost: **{best}**. This selects the Phase 6 strategy by rule. Order sizes are small against "
          "slice volume, so impact is minor and spreading an order over the day mostly adds drift away from the open "
          "along the assumed price path. A real intraday tape could rank them differently.", "",
          "## Portfolio context (same signal, fixed capital)", "",
          "| run | return | Sharpe | max drawdown |", "|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['ret']:.2%} | {r['sharpe']:.2f} | {r['max_dd']:.2%} |")
    L.append(f"| frictionless (zero costs, immediate) | {free['ret']:.2%} | {free['sharpe']:.2f} | {free['max_dd']:.2%} |")
    L.append(f"| equal-weight all 20, no trading | {ew['ret']:.2%} | {ew['sharpe']:.2f} | {ew['max_dd']:.2%} |")
    L += ["", "The signal has no validated edge (Phase 3). These numbers show execution cost drag on a held-constant "
          "signal. They are not evidence of profitability. Survivorship bias applies (all 20 names survived to 2026).", ""]
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
