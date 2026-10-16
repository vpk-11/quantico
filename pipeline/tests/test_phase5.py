"""Phase 5 checks: the cost decomposition stays consistent between the vectorized `cost_stats`
and the per-order `execution.decompose_cost`, feasible orders fill 100% while infeasible ones
report a partial fill explicitly (never a crash), every strategy in a cell decides identical
orders, execution never happens before the decision date, and the random null is deterministic.

Offline, synthetic data only. No network, no research store.
Run: python tests/test_phase5.py  (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import backtest as bt  # noqa: E402
import execution as ex  # noqa: E402

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
P = ex.cost_params({
    "slices": 13, "fee_bps": 1.0, "half_spread_bps": 1.0, "impact_coeff_bps": 10.0,
    "participation_cap": 0.10, "latency_slices": 1, "cost_multiplier": 1.0,
    "capital": 1_000_000, "top_n": 3,
})


def _synthetic(n_days: int = 20, volume: int = 2_000_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(5)
    dates = [f"2024-01-{d:02d}" for d in range(1, n_days + 1)]
    rows, preds = [], []
    for d_i, d in enumerate(dates):
        for t in TICKERS:
            px = 100.0 + d_i * 0.1
            rows.append({"ticker": t, "date": d, "open": px, "high": px * 1.02, "low": px * 0.98,
                         "close": px * 1.01, "volume": volume})
            preds.append({"ticker": t, "date": d, "prediction": float(rng.normal())})
    return pd.DataFrame(rows).set_index(["date", "ticker"]), pd.DataFrame(preds)


def test_cost_stats_matches_decompose_cost_sum():
    bars, preds = _synthetic()
    close = bars["close"].unstack()
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    exec_for = bt.rebalance_schedule(close, preds, rebalance_days=5)
    r = bt.backtest(bars, close, pred_by_day, exec_for, ex.VWAP, P, use_reference=True)
    for bench_col in ("open", "interval_vwap"):
        s = bt.cost_stats(r["fills"], bench_col)
        f = r["fills"][r["fills"]["filled"] > 0]
        parts = [ex.decompose_cost(row.side, row.filled, row.notional, row.fees, row.spread_cost,
                                    row.impact_cost, row.raw_notional, row[bench_col]) for _, row in f.iterrows()]
        notional = f["notional"].sum()
        assert abs(s["total_bps"] - sum(p["total"] for p in parts) / notional * 1e4) < 1e-6
        assert abs(s["drift_bps"] - sum(p["drift"] for p in parts) / notional * 1e4) < 1e-6


def test_fill_rate_full_when_feasible_partial_when_not_no_crash():
    bars, preds = _synthetic(volume=50_000_000)  # ample volume: feasible
    close = bars["close"].unstack()
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    exec_for = bt.rebalance_schedule(close, preds, rebalance_days=5)
    r = bt.backtest(bars, close, pred_by_day, exec_for, ex.IMMEDIATE, P, use_reference=True)
    s = bt.cost_stats(r["fills"], "open")
    assert s["fill_rate"] == 1.0

    huge_p = {**P, "capital": 10_000_000_000}  # capital far beyond liquidity: infeasible
    bars2, preds2 = _synthetic(volume=1_000)
    close2 = bars2["close"].unstack()
    pred_by_day2 = {d: g.set_index("ticker")["prediction"] for d, g in preds2.groupby("date")}
    exec_for2 = bt.rebalance_schedule(close2, preds2, rebalance_days=5)
    r2 = bt.backtest(bars2, close2, pred_by_day2, exec_for2, ex.IMMEDIATE, huge_p, use_reference=True)
    s2 = bt.cost_stats(r2["fills"], "open")
    assert 0.0 <= s2["fill_rate"] < 1.0  # partial, reported explicitly, no crash


def test_empty_fills_no_crash():
    s = bt.cost_stats(pd.DataFrame(), "open")
    assert s["orders"] == 0 and s["total_bps"] == 0.0


def test_same_orders_decided_across_strategies_in_a_cell():
    bars, preds = _synthetic()
    close = bars["close"].unstack()
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    exec_for = bt.rebalance_schedule(close, preds, rebalance_days=5)
    keys = []
    for sid in ex.STRATEGIES.values():
        r = bt.backtest(bars, close, pred_by_day, exec_for, sid, P, use_reference=True)
        f = r["fills"].sort_values(["date", "ticker"]).reset_index(drop=True)
        keys.append(tuple(zip(f["date"], f["ticker"], f["qty"], f["side"])))
    assert keys[0] == keys[1] == keys[2]


def test_execution_never_before_decision():
    bars, preds = _synthetic()
    close = bars["close"].unstack()
    exec_for = bt.rebalance_schedule(close, preds, rebalance_days=5)
    dates = list(close.index)
    for d_exec, d_decide in exec_for.items():
        assert dates.index(d_exec) == dates.index(d_decide) + 1  # the very next trading day, never same or earlier


def test_random_null_deterministic_under_seed():
    bars, preds = _synthetic()
    close = bars["close"].unstack()
    pred_by_day = {d: g.set_index("ticker")["prediction"] for d, g in preds.groupby("date")}
    exec_for = bt.rebalance_schedule(close, preds, rebalance_days=5)
    a = bt.random_null_sharpes(bars, close, pred_by_day, exec_for, ex.IMMEDIATE, P, draws=5, seed=11)
    b = bt.random_null_sharpes(bars, close, pred_by_day, exec_for, ex.IMMEDIATE, P, draws=5, seed=11)
    assert a == b


def test_grid_cells_cover_variants_and_noisy_seeds():
    cfg5 = {"intraday_variants": ["base", "swap_extremes", "flat_volume"],
            "cost_multipliers": [0.5, 1.0, 2.0], "capital_levels": [1, 2, 3]}
    cfg4 = {"intraday_noise_seeds": [1, 2, 3, 4, 5]}
    cells = bt.grid_cells(cfg5, cfg4)
    assert len(cells) == 3 * 3 * 3 + 5 * 3 * 3
    assert sum(1 for v, *_ in cells if v == "noisy_volume") == 5 * 3 * 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
