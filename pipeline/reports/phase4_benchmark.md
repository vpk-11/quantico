# Phase 4 benchmark: C++ simulator vs Python reference

3 repeats per order count, median and spread (max minus min) across repeats, in seconds. C++ timing includes Python-side packing, subprocess startup and binary pipe I/O; Python is an in-process loop on the same orders.

| orders | C++ median (s) | C++ spread (s) | Python median (s) | Python spread (s) | faster |
|---|---|---|---|---|---|
| 2,000 | 0.008 | 0.057 | 0.008 | 0.002 | C++ |
| 20,000 | 0.044 | 0.001 | 0.082 | 0.001 | C++ |
| 200,000 | 0.419 | 0.015 | 0.833 | 0.026 | C++ |

The speed claim is limited to this table: end to end, on synthetic orders, with Python-side packing included. The C++ simulator is also an independent, bit-exact-parity-checked implementation (see test_phase4.py).
