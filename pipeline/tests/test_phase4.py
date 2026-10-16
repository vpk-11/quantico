"""Phase 4 checks: slice synthesis, reference fill invariants, and C++ vs Python bit-exact parity.

Needs the simulator built (`make -C simulator`). No network, no research store.
Run: python tests/test_phase4.py  (or: python -m pytest tests)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import execution as ex  # noqa: E402

P = {"slices": 13, "fee_bps": 1.0, "spread_bps": 1.0, "impact_bps": 10.0, "max_participation": 0.10, "latency_slices": 1}


def test_slices_respect_bar():
    for o, h, l, c in [(100, 110, 95, 105), (100, 110, 95, 98)]:
        px, vol = ex.make_slices(o, h, l, c, 1_000_000, 13)
        assert px[0] == o and px[-1] == c
        assert min(px) == l and max(px) == h
        assert sum(vol) <= 1_000_000 and sum(vol) > 999_900
        assert vol[0] > vol[6] < vol[12]  # U shape


def test_reference_invariants():
    px, vol = ex.make_slices(100, 101, 99, 100.5, 5_000_000, 13)
    for strat in (ex.IMMEDIATE, ex.TWAP, ex.VWAP):
        f, n, fees = ex.reference_fill(1, strat, 10_000, px, vol, P)
        assert 0 < f <= 10_000 and fees > 0
        assert n >= f * min(px)  # buys never fill below the lowest slice price
        fs, ns, _ = ex.reference_fill(-1, strat, 10_000, px, vol, P)
        assert fs == f and ns < f * max(px)  # sells fill below the slice price
    zero = {**P, "fee_bps": 0.0, "spread_bps": 0.0, "impact_bps": 0.0}
    f, n, fees = ex.reference_fill(1, ex.IMMEDIATE, 10, px, vol, zero)
    assert f == 10 and fees == 0.0 and n == 10 * px[1]  # latency 1: fills in slice 1


def test_participation_cap_and_carry():
    px, vol = ex.make_slices(100, 100, 100, 100, 1000, 13)  # tiny volume: cap binds
    f, _, _ = ex.reference_fill(1, ex.IMMEDIATE, 10_000_000, px, vol, P)
    assert f == sum(int(0.10 * v) for v in vol[1:])  # slice 0 unused by latency, rest capped


def test_cpp_matches_python_bit_exact():
    rng = np.random.default_rng(7)
    orders, expect = [], []
    for i in range(300):
        o = float(rng.uniform(20, 400))
        h, l = o * rng.uniform(1.0, 1.05), o * rng.uniform(0.95, 1.0)
        c = float(rng.uniform(l, h))
        px, vol = ex.make_slices(o, h, l, c, int(rng.integers(0, 50_000_000)), 13)
        side, strat, qty = int(rng.choice([-1, 1])), int(rng.integers(0, 3)), int(rng.integers(0, 2_000_000))
        orders.append({"id": i, "side": side, "strategy": strat, "qty": qty, "prices": px, "vols": vol})
        expect.append(ex.reference_fill(side, strat, qty, px, vol, P))
    got = ex.simulate(orders, P)
    assert len(got) == 300
    for g, e in zip(got, expect):
        assert (g["filled"], g["notional"], g["fees"]) == e, (g, e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
