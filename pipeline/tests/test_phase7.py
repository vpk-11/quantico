"""Phase 7 checks: the edge-claim rule (pure function, every condition tested individually),
attribution components summing to the total NAV change, replay and forward never sharing a row
in the IC or cost computations, and the two "not enough forward data" render paths.

Offline, synthetic data only. No network, no research store.
Run: python tests/test_phase7.py  (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paper  # noqa: E402

CFG7 = {"min_forward_rebalances": 50, "null_percentile_threshold": 95}
CFG_MIN = {"phase3": {"min_tickers_per_day": 2, "target_horizon_days": 5}, "phase7": CFG7}
P_MIN = {"capital": 1_000_000.0, "top_n": 2}


def test_edge_rule_passes_only_with_all_three_conditions():
    low_null = [0.5] * 50  # strategy sharpe (1.5) clears every draw: 100th percentile
    ok, label = paper.edge_rule(60, 0.02, 3.0, 2.0, 1.5, low_null, CFG7)
    assert ok and label == "edge supported"


def test_edge_rule_fails_on_insufficient_sample_alone():
    ok, label = paper.edge_rule(10, 0.02, 3.0, 2.0, 1.5, [0.5] * 50, CFG7)
    assert not ok and "insufficient sample" in label


def test_edge_rule_fails_on_ic_not_significant_alone():
    ok, label = paper.edge_rule(60, 0.02, 1.0, 2.0, 1.5, [0.5] * 50, CFG7)  # t below z
    assert not ok and label == "no edge established"


def test_edge_rule_fails_on_ic_sign_alone():
    ok, label = paper.edge_rule(60, -0.01, 3.0, 2.0, 1.5, [0.5] * 50, CFG7)  # mean IC <= 0
    assert not ok and label == "no edge established"


def test_edge_rule_fails_on_null_percentile_alone():
    high_null = [2.0] * 50  # strategy sharpe (1.5) clears none: 0th percentile
    ok, label = paper.edge_rule(60, 0.02, 3.0, 2.0, 1.5, high_null, CFG7)
    assert not ok and label == "no edge established"


def test_attribution_components_sum_to_total():
    fills = pd.DataFrame([
        {"ticker": "A", "side": 1, "qty": 100, "filled": 100, "notional": 10_100.0, "fees": 1.0, "open": 100.0},
        {"ticker": "B", "side": -1, "qty": 50, "filled": 50, "notional": 4_900.0, "fees": 0.5, "open": 100.0},
    ])
    ew_nav = pd.Series({"d0": 1_000_000.0, "d1": 1_010_000.0})
    att = paper.attribution(fills, ew_nav, nav0=1_000_000.0, nav1=1_005_000.0, prior_date="d0", last_date="d1")
    assert abs(att["total"] - (att["market"] + att["selection"] + att["costs"])) < 1e-6
    assert att["total"] == 5_000.0


def test_attribution_cost_isolated_by_mode():
    """The cost figure for one mode must not pick up a fill tagged with the other mode."""
    fills_all = pd.DataFrame([
        {"ticker": "A", "side": 1, "qty": 10, "filled": 10, "notional": 1_010.0, "fees": 1.0, "open": 100.0,
         "date": "d1", "mode": "replay"},
        {"ticker": "A", "side": 1, "qty": 10, "filled": 10, "notional": 1_050.0, "fees": 5.0, "open": 100.0,
         "date": "d2", "mode": "forward"},
    ])
    ew_nav = pd.Series({"d0": 1_000.0, "d1": 1_000.0})
    replay_fills = fills_all[fills_all["mode"] == "replay"]
    att = paper.attribution(replay_fills, ew_nav, nav0=1_000.0, nav1=989.0, prior_date="d0", last_date="d1")
    assert att["costs"] == -11.0  # only the replay fill's cost: side*(1010-1000)+1 = 11, not the forward fill's 5 fee


def test_mode_filtering_never_mixes_ic_between_replay_and_forward():
    preds_all = pd.DataFrame([
        {"date": "2026-01-02", "ticker": "A", "prediction": 1.0, "mode": "replay"},
        {"date": "2026-01-02", "ticker": "B", "prediction": -1.0, "mode": "replay"},
        {"date": "2026-10-13", "ticker": "A", "prediction": -1.0, "mode": "forward"},
        {"date": "2026-10-13", "ticker": "B", "prediction": 1.0, "mode": "forward"},
    ])
    realized = pd.DataFrame([
        {"date": "2026-01-02", "ticker": "A", "target": 0.05},
        {"date": "2026-01-02", "ticker": "B", "target": -0.05},
        {"date": "2026-10-13", "ticker": "A", "target": 0.05},
        {"date": "2026-10-13", "ticker": "B", "target": -0.05},
    ])
    replay_ic = paper._ic_series(preds_all[preds_all["mode"] == "replay"][["date", "ticker", "prediction"]], realized, min_tickers=2)
    forward_ic = paper._ic_series(preds_all[preds_all["mode"] == "forward"][["date", "ticker", "prediction"]], realized, min_tickers=2)
    assert replay_ic.iloc[0] > 0   # predicted A high / B low, realized A up / B down: positive rank IC
    assert forward_ic.iloc[0] < 0  # predictions inverted vs realized ranks: negative rank IC


def test_render_mode_forward_no_data_yet():
    nav_all = pd.DataFrame({"date": ["2026-01-02"], "nav": [1_000_000.0], "mode": ["replay"]})
    fills_all = pd.DataFrame(columns=["ticker", "side", "qty", "filled", "notional", "fees", "open", "date", "mode"])
    preds_all = pd.DataFrame(columns=["date", "ticker", "prediction", "mode"])
    out = paper.render_mode("forward", nav_all, fills_all, preds_all, pd.DataFrame(columns=["ticker", "date", "target"]),
                             None, None, pd.Series(dtype=float), pd.DataFrame(columns=["mean", "count"]),
                             CFG_MIN, P_MIN, 1, 2.0)
    assert out == ["## Forward", "", "No forward data yet.", ""]


def test_render_mode_forward_insufficient_sample():
    nav_all = pd.DataFrame({
        "date": ["2026-01-02", "2026-10-13", "2026-10-14"],
        "nav": [1_000_000.0, 1_050_000.0, 1_051_000.0],
        "mode": ["replay", "forward", "forward"],
    })
    preds_all = pd.DataFrame({
        "date": ["2026-10-13", "2026-10-14"], "ticker": ["A", "A"],
        "prediction": [0.1, 0.2], "mode": ["forward", "forward"],
    })
    fills_all = pd.DataFrame(columns=["ticker", "side", "qty", "filled", "notional", "fees", "open", "date", "mode"])
    realized = pd.DataFrame(columns=["ticker", "date", "target"])
    out = paper.render_mode("forward", nav_all, fills_all, preds_all, realized,
                             None, None, pd.Series(dtype=float), pd.DataFrame(columns=["mean", "count"]),
                             CFG_MIN, P_MIN, 1, 2.0)
    text = "\n".join(out)
    assert "Insufficient sample" in text
    assert "2 forward rebalances" in text
    assert "### Attribution" not in text  # no claim drawn past the sample-size line


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
