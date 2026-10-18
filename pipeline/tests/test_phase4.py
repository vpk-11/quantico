"""Phase 4 checks: slice synthesis (all variants), reference fill invariants, cost decomposition,
the latency scheduling-bug regression, edge cases, and C++ vs Python bit-exact parity.

Needs the simulator built (`make -C simulator`). No network, no research store.
Run: python tests/test_phase4.py  (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import execution as ex  # noqa: E402

P4 = {
    "slices": 13, "fee_bps": 1.0, "half_spread_bps": 1.0, "impact_coeff_bps": 10.0,
    "participation_cap": 0.10, "latency_slices": 1, "cost_multiplier": 1.0,
}


def _p(cost_multiplier: float = 1.0) -> dict:
    return ex.cost_params({**P4, "cost_multiplier": cost_multiplier})


def test_slices_respect_bar_every_variant():
    for variant in ex.VARIANTS:
        for o, h, l, c in [(100, 110, 95, 105), (100, 110, 95, 98)]:
            seed = 1 if variant == "noisy_volume" else None
            px, vol = ex.make_slices(o, h, l, c, 1_000_000, 13, variant=variant, seed=seed)
            assert px[0] == o and px[-1] == c
            assert min(px) == l and max(px) == h
            assert sum(vol) == 1_000_000, (variant, sum(vol))


def test_noisy_volume_seeds_are_distinct_and_deterministic():
    paths = [ex.make_slices(100, 110, 95, 105, 1_000_000, 13, variant="noisy_volume", seed=s)[1] for s in (1, 2, 3)]
    assert paths[0] != paths[1] and paths[1] != paths[2]
    again = ex.make_slices(100, 110, 95, 105, 1_000_000, 13, variant="noisy_volume", seed=1)[1]
    assert again == paths[0]


def test_unknown_variant_rejected():
    try:
        ex.make_slices(100, 110, 95, 105, 1_000_000, 13, variant="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_reference_invariants():
    p = _p()
    px, vol = ex.make_slices(100, 101, 99, 100.5, 5_000_000, 13)
    for strat in (ex.IMMEDIATE, ex.TWAP, ex.VWAP):
        f, n, fees, sc, ic, rn = ex.reference_fill(1, strat, 10_000, px, vol, p)
        assert 0 < f <= 10_000 and fees > 0 and sc > 0 and ic > 0
        assert n >= f * min(px)  # buys never fill below the lowest slice price
        assert rn < n  # raw (no-slip) notional is less than the slip-adjusted buy notional
        fs, ns, _, _, _, rns = ex.reference_fill(-1, strat, 10_000, px, vol, p)
        assert fs == f and ns < f * max(px)  # sells fill below the slice price
        assert rns > ns  # raw notional is more than the slip-adjusted sell notional
    zero = {**p, "fee_bps": 0.0, "spread_bps": 0.0, "impact_bps": 0.0}
    f, n, fees, sc, ic, rn = ex.reference_fill(1, ex.IMMEDIATE, 10, px, vol, zero)
    assert f == 10 and fees == 0.0 and sc == 0.0 and ic == 0.0 and n == rn == 10 * px[1]  # latency 1, no slip


def test_cost_multiplier_scales_spread_and_impact_not_fee():
    px, vol = ex.make_slices(100, 101, 99, 100.5, 5_000_000, 13)
    base, double = _p(1.0), _p(2.0)
    fb, nb, feesb, scb, icb, _ = ex.reference_fill(1, ex.VWAP, 10_000, px, vol, base)
    fd, nd, feesd, scd, icd, _ = ex.reference_fill(1, ex.VWAP, 10_000, px, vol, double)
    assert fb == fd  # same fills, only price impact changes
    assert abs(scd - 2 * scb) < 1e-6 and abs(icd - 2 * icb) < 1e-6
    assert abs(feesb / nb - feesd / nd) < 1e-9  # fee rate untouched by the multiplier (dollar fees move with notional)


def test_cost_components_sum_to_total():
    """`drift` is computed from `raw_notional` (independent sum of filled_i * slice_price_i,
    no slip), not as `total` minus the other parts, so this checks a real identity: a bug in
    spread_cost or impact_cost would break it (see the next test)."""
    rng = np.random.default_rng(3)
    p = _p()
    for _ in range(1_000):
        o = float(rng.uniform(20, 400))
        h, l = o * rng.uniform(1.0, 1.05), o * rng.uniform(0.95, 1.0)
        c = float(rng.uniform(l, h))
        px, vol = ex.make_slices(o, h, l, c, int(rng.integers(100_000, 50_000_000)), 13)
        side, strat, qty = int(rng.choice([-1, 1])), int(rng.integers(0, 3)), int(rng.integers(0, 2_000_000))
        filled, notional, fees, sc, ic, rn = ex.reference_fill(side, strat, qty, px, vol, p)
        for benchmark in (px[0], sum(a * b for a, b in zip(px, vol)) / sum(vol)):
            d = ex.decompose_cost(side, filled, notional, fees, sc, ic, rn, benchmark)
            parts = d["drift"] + d["spread"] + d["impact"] + d["fees"]
            assert abs(d["total"] - parts) <= 1e-9 * max(1.0, abs(d["total"]))


def test_cost_decomposition_catches_a_broken_component():
    """Sanity check that the identity above is a real test, not a tautology: corrupting
    spread_cost (independent of drift, which comes from raw_notional) must break the sum."""
    p = _p()
    px, vol = ex.make_slices(100, 101, 99, 100.5, 5_000_000, 13)
    filled, notional, fees, sc, ic, rn = ex.reference_fill(1, ex.VWAP, 10_000, px, vol, p)
    d = ex.decompose_cost(1, filled, notional, fees, sc * 1.5, ic, rn, px[0])  # inject a bug
    parts = d["drift"] + d["spread"] + d["impact"] + d["fees"]
    assert abs(d["total"] - parts) > 1.0  # the broken component is not silently absorbed


def test_participation_cap_and_carry():
    p = _p()
    px, vol = ex.make_slices(100, 100, 100, 100, 1000, 13)  # tiny volume: cap binds
    f, *_ = ex.reference_fill(1, ex.IMMEDIATE, 10_000_000, px, vol, p)
    assert f == sum(int(0.10 * v) for v in vol[1:])  # slice 0 unused by latency, rest capped
    assert 0 < f < 10_000_000  # partial fill, no crash


def test_latency_scheduling_regression_full_fill_when_feasible():
    """Guards the bug where late-day child orders were dropped once latency shifted them past
    the close: with a feasible order size, every strategy must fill 100% of the request."""
    p = _p()
    px, vol = ex.make_slices(100, 101, 99, 100.5, 20_000_000, 13)
    feasible_qty = int(0.5 * min(vol[1:]))  # well inside every slice's 10% cap after latency
    for strat in (ex.IMMEDIATE, ex.TWAP, ex.VWAP):
        f, *_ = ex.reference_fill(1, strat, feasible_qty, px, vol, p)
        assert f == feasible_qty, (strat, f, feasible_qty)


def test_edge_cases_no_crash():
    p = _p()
    px, vol = ex.make_slices(100, 101, 99, 100.5, 5_000_000, 13)
    for strat in (ex.IMMEDIATE, ex.TWAP, ex.VWAP):
        f, n, fees, sc, ic, rn = ex.reference_fill(1, strat, 0, px, vol, p)  # zero quantity
        assert f == 0 and n == 0.0 and fees == 0.0 and sc == 0.0 and ic == 0.0 and rn == 0.0
    vol_gap = list(vol)
    vol_gap[5] = 0  # missing-volume slice
    f, *_ = ex.reference_fill(1, ex.VWAP, 10_000, px, vol_gap, p)
    assert f >= 0  # no division by zero, no crash
    f, *_ = ex.reference_fill(1, ex.IMMEDIATE, 10 ** 12, px, vol, p)  # exceeds cap every slice
    assert 0 < f < 10 ** 12  # partial fill, not a crash


def test_cpp_matches_python_bit_exact():
    rng = np.random.default_rng(7)
    for variant in ex.VARIANTS:
        for cost_multiplier in (0.5, 1.0, 2.0):
            p = _p(cost_multiplier)
            orders, expect = [], []
            for i in range(300):
                o = float(rng.uniform(20, 400))
                h, l = o * rng.uniform(1.0, 1.05), o * rng.uniform(0.95, 1.0)
                c = float(rng.uniform(l, h))
                seed = int(rng.integers(0, 1_000_000)) if variant == "noisy_volume" else None
                px, vol = ex.make_slices(o, h, l, c, int(rng.integers(0, 50_000_000)), 13, variant=variant, seed=seed)
                side, strat, qty = int(rng.choice([-1, 1])), int(rng.integers(0, 3)), int(rng.integers(0, 2_000_000))
                orders.append({"id": i, "side": side, "strategy": strat, "qty": qty, "prices": px, "vols": vol})
                expect.append(ex.reference_fill(side, strat, qty, px, vol, p))
            got = ex.simulate(orders, p)
            assert len(got) == 300
            for g, e in zip(got, expect):
                assert (g["filled"], g["notional"], g["fees"], g["spread_cost"], g["impact_cost"], g["raw_notional"]) == e, (variant, cost_multiplier, g, e)


def test_simulate_determinism():
    rng = np.random.default_rng(11)
    p = _p()
    orders = []
    for i in range(50):
        o = float(rng.uniform(20, 400))
        px, vol = ex.make_slices(o, o * 1.02, o * 0.98, o * 1.01, int(rng.integers(1_000_000, 50_000_000)), 13)
        orders.append({"id": i, "side": int(rng.choice([-1, 1])), "strategy": int(rng.integers(0, 3)),
                       "qty": int(rng.integers(1, 2_000_000)), "prices": px, "vols": vol})
    a, b = ex.simulate(orders, p), ex.simulate(orders, p)
    assert a == b


def _order(**kw) -> dict:
    px, vol = ex.make_slices(100, 102, 98, 101, 1_000_000, 13)
    return {"id": 0, "side": 1, "strategy": 1, "qty": 5_000, "prices": px, "vols": vol, **kw}


def test_cpp_rejects_bad_input():
    import pytest
    bad = [
        {"side": 0}, {"strategy": 3}, {"strategy": -1}, {"qty": -1}, {"qty": 2**53 + 1},
        {"prices": [float("nan")] * 13}, {"prices": [0.0] * 13}, {"prices": [float("inf")] * 13},
        {"vols": [-1] * 13}, {"vols": [2**53 + 1] * 13},
    ]
    for kw in bad:
        with pytest.raises(RuntimeError, match="simulator exited 2"):
            ex.simulate([_order(**kw)], _p())
    assert ex.simulate([], _p()) == []


def test_cpp_rejects_truncated_stream_and_bad_args():
    import subprocess
    args = [str(ex.SIM_BIN), "1.0", "1.0", "10.0", "0.1", "1", "13"]
    short = subprocess.run(args, input=np.array([1, 1, 5], dtype="<i8").tobytes() + b"\x00" * 30, capture_output=True)
    assert short.returncode == 2 and b"truncated" in short.stderr
    for i, v in [(1, "x"), (1, "-1"), (3, "nan"), (4, "0"), (4, "1.5"), (5, "-1"), (6, "0")]:
        a = list(args)
        a[i] = v
        r = subprocess.run(a, input=b"", capture_output=True)
        assert r.returncode == 2, (i, v)


def test_cpp_vwap_large_qty_matches_python_past_int64_product():
    # qty * cum reaches 2^103, far past int64: C++ must use 128-bit math to agree with Python ints.
    p = _p()
    px = [100.0 + i for i in range(13)]
    vol = [2**49] * 13
    o = _order(strategy=2, qty=2**53, prices=px, vols=vol)
    got = ex.simulate([o], p)[0]
    e = ex.reference_fill(o["side"], o["strategy"], o["qty"], px, vol, p)
    assert (got["filled"], got["notional"], got["fees"], got["spread_cost"], got["impact_cost"], got["raw_notional"]) == e


def test_cpp_fill_invariants_random():
    rng = np.random.default_rng(23)
    p = _p()
    orders = []
    for i in range(2000):
        o = float(rng.uniform(20, 400))
        px, vol = ex.make_slices(o, o * 1.02, o * 0.98, o * 1.01, int(rng.choice([0, 1, 1_000, 40_000_000])), 13)
        orders.append({"id": i, "side": int(rng.choice([-1, 1])), "strategy": int(rng.integers(0, 3)),
                       "qty": int(rng.choice([0, 1, 13, 2_000_000])), "prices": px, "vols": vol})
    for o, r in zip(orders, ex.simulate(orders, p)):
        assert r["id"] == o["id"]
        assert 0 <= r["filled"] <= o["qty"]
        assert (r["notional"] > 0) == (r["filled"] > 0)
        assert r["fees"] >= 0 and r["spread_cost"] >= 0 and r["impact_cost"] >= 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
