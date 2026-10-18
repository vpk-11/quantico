// Execution simulator. Usage: sim <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>
//
// Reads orders from stdin and writes results to stdout, both raw native-endian binary (no text
// parsing, which dominated runtime when this was CSV).
// Order record: i64 side (1 buy, -1 sell), i64 strategy (0 immediate, 1 TWAP, 2 VWAP), i64 qty,
//   then `slices` f64 prices, then `slices` i64 volumes.
// Result record: i64 filled, f64 notional, f64 fees, f64 spread_cost, f64 impact_cost, f64 raw_notional.
// Results are in input order; doubles round-trip exactly, so output is bit-exact with Python.
//
// Child orders are scheduled per slice, shifted by `latency` slices, capped at max_part of the
// slice volume (the remainder carries to the next slice), and priced at the slice price moved
// against the trader by spread + impact * sqrt(participation). Anything not filled by the last
// slice is dropped. Operation order mirrors pipeline/src/execution.py:reference_fill exactly.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using i64 = int64_t;

struct Result {
    i64 filled;
    double notional, fees, spread_cost, impact_cost, raw_notional;
};

static void die(const std::string& msg) {
    std::fprintf(stderr, "sim: %s\n", msg.c_str());
    std::exit(2);
}

static void read_exact(void* dst, size_t bytes, bool at_boundary) {
    const size_t got = std::fread(dst, 1, bytes, stdin);
    if (got == 0 && at_boundary && std::feof(stdin)) std::exit(0);  // clean end of input at a record boundary
    if (got != bytes) die("truncated order record");
}

// Writes the schedule for `vol[0..n)` into `s` (n entries).
static void schedule(int strategy, i64 qty, const i64* vol, int n, i64* s) {
    std::fill(s, s + n, 0);
    if (strategy == 0) {
        s[0] = qty;
    } else if (strategy == 1) {
        for (int i = 0; i < n; ++i) s[i] = qty / n + (i < qty % n ? 1 : 0);
    } else if (strategy == 2) {
        i64 total = 0;
        for (int i = 0; i < n; ++i) total += vol[i];
        if (total == 0) return;
        i64 cum = 0, prev = 0;
        for (int i = 0; i < n; ++i) {
            cum += vol[i];
            const i64 cur = qty * cum / total;
            s[i] = cur - prev;
            prev = cur;
        }
    } else {
        die("unknown strategy " + std::to_string(strategy));
    }
}

int main(int argc, char** argv) {
    if (argc != 7) die("usage: sim <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>");
    const double fee = std::atof(argv[1]), spread = std::atof(argv[2]), impact = std::atof(argv[3]);
    const double max_part = std::atof(argv[4]);
    const int latency = std::atoi(argv[5]), n = std::atoi(argv[6]);
    if (n <= 0 || latency < 0) die("bad slices or latency");

    static char in_buf[1 << 20], out_buf[1 << 20];
    std::setvbuf(stdin, in_buf, _IOFBF, sizeof in_buf);
    std::setvbuf(stdout, out_buf, _IOFBF, sizeof out_buf);

    // Child i executes in slice i + latency, so the schedule spans the n - latency slices
    // that can still fill, and VWAP weights by the volume of those execution slices.
    const int skip = std::min(latency, n - 1), m = n - skip;
    std::vector<double> price(n);
    std::vector<i64> vol(n), sched(m);
    for (;;) {
        i64 head[3];
        read_exact(head, sizeof head, true);
        const i64 side = head[0], strategy = head[1], qty = head[2];
        if ((side != 1 && side != -1) || qty < 0) die("bad side or qty");
        read_exact(price.data(), n * sizeof(double), false);
        read_exact(vol.data(), n * sizeof(i64), false);

        schedule(static_cast<int>(strategy), qty, vol.data() + skip, m, sched.data());
        i64 carry = 0, filled = 0;
        double notional = 0.0, fees = 0.0, spread_cost = 0.0, impact_cost = 0.0, raw_notional = 0.0;
        for (int j = 0; j < n; ++j) {
            const i64 want = (j >= latency && j - latency < m ? sched[j - latency] : 0) + carry;
            const i64 cap = static_cast<i64>(std::floor(max_part * static_cast<double>(vol[j])));
            const i64 fill = std::min(want, cap);
            carry = want - fill;
            if (fill <= 0) continue;
            const double part = static_cast<double>(fill) / static_cast<double>(vol[j]);
            const double spread_component = spread;
            const double impact_component = impact * std::sqrt(part);
            const double slip = spread_component + impact_component;
            const double px = price[j] * (1.0 + side * slip / 10000.0);
            const double notl = static_cast<double>(fill) * px;
            notional += notl;
            fees += notl * fee / 10000.0;
            spread_cost += static_cast<double>(fill) * price[j] * spread_component / 10000.0;
            impact_cost += static_cast<double>(fill) * price[j] * impact_component / 10000.0;
            raw_notional += static_cast<double>(fill) * price[j];
            filled += fill;
        }
        const Result r{filled, notional, fees, spread_cost, impact_cost, raw_notional};
        std::fwrite(&r, sizeof r, 1, stdout);
    }
}
