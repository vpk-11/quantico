// Execution simulator. Usage: sim <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>
//
// Reads orders from stdin and writes results to stdout, both raw native-endian binary (no text
// parsing, which dominated runtime when this was CSV).
// Order record: i64 side (1 buy, -1 sell), i64 strategy (0 immediate, 1 TWAP, 2 VWAP), i64 qty,
//   then `slices` f64 prices, then `slices` i64 volumes.
// Result record: i64 filled, f64 notional, f64 fees, f64 spread_cost, f64 impact_cost, f64 raw_notional.
// Results are in input order; doubles round-trip exactly, so output is bit-exact with Python.
// Input contract, enforced here because this is the process boundary: side is +-1; strategy is 0..2;
// qty and volumes are in [0, 2^53] (exact as doubles); prices are finite and positive. Any violation,
// a truncated record, or a bad argument exits with status 2 and a message on stderr.
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

// Returns false on clean end of input at a record boundary; dies on a partial read.
static bool read_exact(void* dst, size_t bytes, bool at_boundary) {
    const size_t got = std::fread(dst, 1, bytes, stdin);
    if (got == 0 && at_boundary && std::feof(stdin)) return false;
    if (got != bytes) die("truncated order record");
    return true;
}

static double parse_double(const char* name, const char* text) {
    char* end = nullptr;
    const double v = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(v)) die(std::string("bad ") + name + ": " + text);
    return v;
}

static int parse_int(const char* name, const char* text) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > 1000000) die(std::string("bad ") + name + ": " + text);
    return static_cast<int>(v);
}

// Writes the schedule for `vol[0..n)` into `s` (n entries).
static void schedule(int strategy, i64 qty, const i64* vol, int n, i64* s) {
    std::fill(s, s + n, 0);
    if (strategy == 0) {
        s[0] = qty;
    } else if (strategy == 1) {
        for (int i = 0; i < n; ++i) s[i] = qty / n + (i < qty % n ? 1 : 0);
    } else {  // 2, VWAP
        i64 total = 0;
        for (int i = 0; i < n; ++i) total += vol[i];
        if (total == 0) return;
        i64 cum = 0, prev = 0;
        for (int i = 0; i < n; ++i) {
            cum += vol[i];
            // 128-bit product: qty * cum can pass 2^63, where Python's ints would not overflow.
            const i64 cur = static_cast<i64>(static_cast<__int128>(qty) * cum / total);
            s[i] = cur - prev;
            prev = cur;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 7) die("usage: sim <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>");
    const double fee = parse_double("fee_bps", argv[1]), spread = parse_double("spread_bps", argv[2]);
    const double impact = parse_double("impact_bps", argv[3]), max_part = parse_double("max_part", argv[4]);
    const int latency = parse_int("latency", argv[5]), n = parse_int("slices", argv[6]);
    if (fee < 0 || spread < 0 || impact < 0 || max_part <= 0 || max_part > 1 || n == 0) die("argument out of range");

    static char in_buf[1 << 20], out_buf[1 << 20];
    std::setvbuf(stdin, in_buf, _IOFBF, sizeof in_buf);
    std::setvbuf(stdout, out_buf, _IOFBF, sizeof out_buf);

    // Child i executes in slice i + latency, so the schedule spans the n - latency slices
    // that can still fill, and VWAP weights by the volume of those execution slices.
    const int skip = std::min(latency, n - 1), m = n - skip;
    std::vector<double> price(n);
    std::vector<i64> vol(n), sched(m);
    constexpr i64 kMaxExact = i64{1} << 53;
    i64 head[3];
    while (read_exact(head, sizeof head, true)) {
        const i64 side = head[0], strategy = head[1], qty = head[2];
        if (side != 1 && side != -1) die("bad side");
        if (strategy < 0 || strategy > 2) die("unknown strategy " + std::to_string(strategy));
        if (qty < 0 || qty > kMaxExact) die("qty out of range");
        read_exact(price.data(), n * sizeof(double), false);
        read_exact(vol.data(), n * sizeof(i64), false);
        for (int i = 0; i < n; ++i) {
            if (!(price[i] > 0.0) || !std::isfinite(price[i])) die("price must be finite and positive");
            if (vol[i] < 0 || vol[i] > kMaxExact) die("volume out of range");
        }

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
    if (std::fflush(stdout) != 0 || std::ferror(stdout)) die("write failed");
    return 0;
}
