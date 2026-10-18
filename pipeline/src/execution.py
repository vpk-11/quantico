"""Phase 4/5: intraday slice synthesis, Python reference fill model, simulator subprocess wrapper.

Daily bars carry no intraday data, so each (ticker, day) is expanded into `slices` equal
slices: a deterministic open -> extreme -> extreme -> close price path and a U-shaped volume
profile summing to the day's volume. The slices are synthetic and the results say so.

`reference_fill` is the oracle for simulator/src/sim.cpp: same inputs, same operation order,
so the two agree bit for bit. `simulate` runs the C++ binary as a subprocess.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

SIM_BIN = Path(__file__).resolve().parents[2] / "simulator" / "build" / "sim"
IMMEDIATE, TWAP, VWAP = 0, 1, 2
STRATEGIES = {"immediate": IMMEDIATE, "twap": TWAP, "vwap": VWAP}


VARIANTS = ("base", "swap_extremes", "flat_volume", "noisy_volume")


def make_slices(
    o: float, h: float, l: float, c: float, volume: int, n: int,
    variant: str = "base", seed: int | None = None,
) -> tuple[list[float], list[int]]:
    """Synthetic intraday path: open, first extreme, second extreme, close at slices 0, n//3, 2n//3, n-1.

    `variant` selects a robustness alternative to the base path (see `VARIANTS`); `swap_extremes`
    reverses which extreme the path visits first, `flat_volume` and `noisy_volume` (needs `seed`)
    change the volume profile only. Every variant still reproduces the daily bar exactly: first
    price is `o`, last is `c`, min/max are `l`/`h`, and slice volumes sum to exactly `volume`.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown intraday variant {variant}")
    knots = [0, n // 3, 2 * n // 3, n - 1]
    up = c >= o
    if variant == "swap_extremes":
        up = not up
    levels = [o, l, h, c] if up else [o, h, l, c]
    prices = np.interp(np.arange(n), knots, levels).tolist()
    if variant == "flat_volume":
        w = np.ones(n)
    else:
        w = np.array([1.0 + 2.0 * ((i - (n - 1) / 2) / ((n - 1) / 2)) ** 2 for i in range(n)])
        if variant == "noisy_volume":
            w = w * np.random.default_rng(seed).uniform(0.5, 1.5, size=n)
    vols = np.floor(volume * w / w.sum()).astype(np.int64)
    vols[-1] += volume - int(vols.sum())  # exact sum to volume; remainder absorbed by the last slice
    return prices, vols.tolist()


def cost_params(p4: dict) -> dict:
    """Translates `config.toml [phase4]`'s named cost keys into the internal fee/spread/impact/
    participation dict `reference_fill` and `simulate` use, applying `cost_multiplier` to spread
    and impact only (never to fees). Pass a modified `cost_multiplier` in `p4` to get a different
    sensitivity cell; all other keys pass through unchanged.
    """
    m = p4["cost_multiplier"]
    out = dict(p4)
    out["spread_bps"] = p4["half_spread_bps"] * m
    out["impact_bps"] = p4["impact_coeff_bps"] * m
    out["max_participation"] = p4["participation_cap"]
    return out


def _schedule(strategy: int, qty: int, vol: list[int]) -> list[int]:
    n = len(vol)
    s = [0] * n
    if strategy == IMMEDIATE:
        s[0] = qty
    elif strategy == TWAP:
        s = [qty // n + (1 if i < qty % n else 0) for i in range(n)]
    elif strategy == VWAP:
        total = sum(vol)
        if total == 0:
            return s
        cum = prev = 0
        for i in range(n):
            cum += vol[i]
            cur = qty * cum // total
            s[i], prev = cur - prev, cur
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return s


def reference_fill(
    side: int, strategy: int, qty: int, prices: list[float], vol: list[int], p: dict,
) -> tuple[int, float, float, float, float, float]:
    """Returns (filled, notional, fees, spread_cost, impact_cost, raw_notional), all in dollars
    except filled (shares). `p`: fee_bps, spread_bps, impact_bps, max_participation, latency_slices.

    `spread_cost` and `impact_cost` are each fill's slice price times its component rate, summed.
    `raw_notional` is the sum of fill times slice price with no slip applied (the price the trader
    would have paid with zero spread and impact); `notional` is the slip-adjusted price actually
    paid. These three accumulators are each independent of the others, so `decompose_cost` can
    check their relationship as a real identity, not a tautology.
    """
    # Child i executes in slice i + latency: schedule over the slices that can still fill.
    sched = _schedule(strategy, qty, vol[min(p["latency_slices"], len(vol) - 1):])
    carry = filled = 0
    notional = fees = spread_cost = impact_cost = raw_notional = 0.0
    for j in range(len(vol)):
        k = j - p["latency_slices"]
        want = (sched[k] if 0 <= k < len(sched) else 0) + carry
        cap = math.floor(p["max_participation"] * float(vol[j]))
        fill = min(want, cap)
        carry = want - fill
        if fill <= 0:
            continue
        part = float(fill) / float(vol[j])
        spread_component = p["spread_bps"]
        impact_component = p["impact_bps"] * math.sqrt(part)
        slip = spread_component + impact_component
        px = prices[j] * (1.0 + side * slip / 10000.0)
        notl = float(fill) * px
        notional += notl
        fees += notl * p["fee_bps"] / 10000.0
        spread_cost += float(fill) * prices[j] * spread_component / 10000.0
        impact_cost += float(fill) * prices[j] * impact_component / 10000.0
        raw_notional += float(fill) * prices[j]
        filled += fill
    return filled, notional, fees, spread_cost, impact_cost, raw_notional


def decompose_cost(
    side: int, filled: int, notional: float, fees: float, spread_cost: float, impact_cost: float,
    raw_notional: float, benchmark_price: float,
) -> dict:
    """Signed cost in dollars vs `benchmark_price` (e.g. arrival or interval VWAP), split into
    fees, spread, impact, and drift. Each part comes from its own accumulator, not from
    subtracting the others off a total: `total` uses `notional` (slip-adjusted), `drift` uses
    `raw_notional` (no slip) independently. The four parts sum to `total` as an algebraic
    identity (`notional = raw_notional + side * (spread_cost + impact_cost)`), checked, not assumed.
    """
    if filled == 0:
        return {"total": 0.0, "drift": 0.0, "spread": 0.0, "impact": 0.0, "fees": 0.0}
    total = side * (notional - filled * benchmark_price) + fees
    drift = side * (raw_notional - filled * benchmark_price)
    return {"total": total, "drift": drift, "spread": spread_cost, "impact": impact_cost, "fees": fees}


_RESULT = np.dtype([("filled", "<i8"), ("notional", "<f8"), ("fees", "<f8"),
                    ("spread_cost", "<f8"), ("impact_cost", "<f8"), ("raw_notional", "<f8")])


def simulate(orders: list[dict], p: dict) -> list[dict]:
    """Run the C++ simulator. Each order: id, side, strategy, qty, prices, vols. Returns results in order.

    Orders go to the subprocess's stdin and results come back on stdout as raw binary records
    (layout in simulator/src/sim.cpp); doubles round-trip exactly.
    """
    if not orders:
        return []
    n = p["slices"]
    arr = np.empty(len(orders), dtype=[("side", "<i8"), ("strategy", "<i8"), ("qty", "<i8"),
                                       ("prices", "<f8", n), ("vols", "<i8", n)])
    for key in ("side", "strategy", "qty", "prices", "vols"):
        arr[key] = [o[key] for o in orders]
    args = [str(SIM_BIN), repr(p["fee_bps"]), repr(p["spread_bps"]), repr(p["impact_bps"]),
            repr(p["max_participation"]), str(p["latency_slices"]), str(n)]
    proc = subprocess.run(args, input=arr.tobytes(), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"simulator exited {proc.returncode}: {proc.stderr.decode().strip()}")
    out = proc.stdout
    assert len(out) == len(orders) * _RESULT.itemsize, "simulator returned a short or long result stream"
    rec = np.frombuffer(out, dtype=_RESULT)
    return [{"id": o["id"], **dict(zip(rec.dtype.names, r))} for o, r in zip(orders, rec.tolist())]
