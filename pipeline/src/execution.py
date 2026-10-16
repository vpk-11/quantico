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
import tempfile
from pathlib import Path

import numpy as np

SIM_BIN = Path(__file__).resolve().parents[2] / "simulator" / "build" / "sim"
IMMEDIATE, TWAP, VWAP = 0, 1, 2
STRATEGIES = {"immediate": IMMEDIATE, "twap": TWAP, "vwap": VWAP}


def make_slices(o: float, h: float, l: float, c: float, volume: int, n: int) -> tuple[list[float], list[int]]:
    """Synthetic intraday path: open, first extreme, second extreme, close at slices 0, n//3, 2n//3, n-1."""
    knots = [0, n // 3, 2 * n // 3, n - 1]
    levels = [o, l, h, c] if c >= o else [o, h, l, c]
    prices = np.interp(np.arange(n), knots, levels).tolist()
    w = np.array([1.0 + 2.0 * ((i - (n - 1) / 2) / ((n - 1) / 2)) ** 2 for i in range(n)])
    vols = np.floor(volume * w / w.sum()).astype(np.int64).tolist()
    return prices, vols


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


def reference_fill(side: int, strategy: int, qty: int, prices: list[float], vol: list[int], p: dict) -> tuple[int, float, float]:
    """Returns (filled, notional, fees). `p`: fee_bps, spread_bps, impact_bps, max_participation, latency_slices."""
    # Child i executes in slice i + latency: schedule over the slices that can still fill.
    sched = _schedule(strategy, qty, vol[min(p["latency_slices"], len(vol) - 1):])
    carry = filled = 0
    notional = fees = 0.0
    for j in range(len(vol)):
        k = j - p["latency_slices"]
        want = (sched[k] if 0 <= k < len(sched) else 0) + carry
        cap = math.floor(p["max_participation"] * float(vol[j]))
        fill = min(want, cap)
        carry = want - fill
        if fill <= 0:
            continue
        part = float(fill) / float(vol[j])
        slip = p["spread_bps"] + p["impact_bps"] * math.sqrt(part)
        px = prices[j] * (1.0 + side * slip / 10000.0)
        notl = float(fill) * px
        notional += notl
        fees += notl * p["fee_bps"] / 10000.0
        filled += fill
    return filled, notional, fees


def simulate(orders: list[dict], p: dict) -> list[dict]:
    """Run the C++ simulator. Each order: id, side, strategy, qty, prices, vols. Returns results in order."""
    n = p["slices"]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        for o in orders:
            assert len(o["prices"]) == n and len(o["vols"]) == n
            f.write(",".join([str(o["id"]), str(o["side"]), str(o["strategy"]), str(o["qty"]),
                              *(repr(x) for x in o["prices"]), *(str(v) for v in o["vols"])]) + "\n")
    try:
        args = [str(SIM_BIN), f.name, repr(p["fee_bps"]), repr(p["spread_bps"]), repr(p["impact_bps"]),
                repr(p["max_participation"]), str(p["latency_slices"]), str(n)]
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    finally:
        Path(f.name).unlink()
    res = []
    for line in out.splitlines():
        oid, filled, notional, fees = line.split(",")
        res.append({"id": oid, "filled": int(filled), "notional": float(notional), "fees": float(fees)})
    return res
