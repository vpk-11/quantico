// Execution simulator. Usage: sim <orders.csv> <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>
//
// One order per input line: id,side,strategy,qty,p_0..p_{N-1},v_0..v_{N-1}
//   side: 1 buy, -1 sell. strategy: 0 immediate, 1 TWAP, 2 VWAP. prices double, volumes integer.
// One result per line: id,filled,notional,fees  (%.17g, so Python can compare bit for bit).
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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using i64 = int64_t;

static void die(const std::string& msg) {
    std::fprintf(stderr, "sim: %s\n", msg.c_str());
    std::exit(2);
}

static std::vector<i64> schedule(int strategy, i64 qty, const std::vector<i64>& vol) {
    const int n = static_cast<int>(vol.size());
    std::vector<i64> s(n, 0);
    if (strategy == 0) {
        s[0] = qty;
    } else if (strategy == 1) {
        for (int i = 0; i < n; ++i) s[i] = qty / n + (i < qty % n ? 1 : 0);
    } else if (strategy == 2) {
        i64 total = 0;
        for (i64 v : vol) total += v;
        if (total == 0) return s;
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
    return s;
}

int main(int argc, char** argv) {
    if (argc != 8) die("usage: sim <orders.csv> <fee_bps> <spread_bps> <impact_bps> <max_part> <latency> <slices>");
    const double fee = std::atof(argv[2]), spread = std::atof(argv[3]), impact = std::atof(argv[4]);
    const double max_part = std::atof(argv[5]);
    const int latency = std::atoi(argv[6]), n = std::atoi(argv[7]);
    if (n <= 0 || latency < 0) die("bad slices or latency");

    std::ifstream in(argv[1]);
    if (!in) die("cannot open orders file");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f;
        std::stringstream ss(line);
        for (std::string t; std::getline(ss, t, ',');) f.push_back(t);
        if (f.size() != static_cast<size_t>(4 + 2 * n)) die("bad field count: " + line);

        const int side = std::atoi(f[1].c_str()), strategy = std::atoi(f[2].c_str());
        const i64 qty = std::atoll(f[3].c_str());
        if ((side != 1 && side != -1) || qty < 0) die("bad side or qty: " + line);
        std::vector<double> price(n);
        std::vector<i64> vol(n);
        for (int i = 0; i < n; ++i) {
            price[i] = std::strtod(f[4 + i].c_str(), nullptr);
            vol[i] = std::atoll(f[4 + n + i].c_str());
        }

        // Child i executes in slice i + latency, so the schedule spans the n - latency slices
        // that can still fill, and VWAP weights by the volume of those execution slices.
        const std::vector<i64> sched =
            schedule(strategy, qty, std::vector<i64>(vol.begin() + std::min(latency, n - 1), vol.end()));
        i64 carry = 0, filled = 0;
        double notional = 0.0, fees = 0.0;
        for (int j = 0; j < n; ++j) {
            const i64 want = (j >= latency && j - latency < static_cast<int>(sched.size()) ? sched[j - latency] : 0) + carry;
            const i64 cap = static_cast<i64>(std::floor(max_part * static_cast<double>(vol[j])));
            const i64 fill = std::min(want, cap);
            carry = want - fill;
            if (fill <= 0) continue;
            const double part = static_cast<double>(fill) / static_cast<double>(vol[j]);
            const double slip = spread + impact * std::sqrt(part);
            const double px = price[j] * (1.0 + side * slip / 10000.0);
            const double notl = static_cast<double>(fill) * px;
            notional += notl;
            fees += notl * fee / 10000.0;
            filled += fill;
        }
        std::printf("%s,%lld,%.17g,%.17g\n", f[0].c_str(), static_cast<long long>(filled), notional, fees);
    }
    return 0;
}
