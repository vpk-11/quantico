# Phase 4 benchmark: C++ simulator vs Python reference

3 repeats per order count, median and spread (max minus min) across repeats, in seconds. C++ timing includes Python-side packing, subprocess startup and binary pipe I/O; Python is an in-process loop on the same orders.

| orders | C++ median (s) | C++ spread (s) | Python median (s) | Python spread (s) | faster |
|---|---|---|---|---|---|
| 2,000 | 0.007 | 0.001 | 0.008 | 0.000 | C++ |
| 20,000 | 0.043 | 0.002 | 0.081 | 0.002 | C++ |
| 200,000 | 0.404 | 0.001 | 0.819 | 0.016 | C++ |

The speed claim is limited to this table: end to end, on synthetic orders, with Python-side packing included. The C++ simulator is also an independent, bit-exact-parity-checked implementation (see test_phase4.py).
