CXX ?= c++
# -ffp-contract=off: no fused multiply-add, so results match the Python reference bit for bit.
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -ffp-contract=off

build/sim: src/sim.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf build

# Debug build with ASan and UBSan at the same path; run `pytest tests/test_phase4.py`, then `make -B` to restore.
sanitize:
	@mkdir -p build
	$(CXX) -std=c++17 -O1 -g -Wall -Wextra -ffp-contract=off -fsanitize=address,undefined -fno-sanitize-recover=undefined -o build/sim src/sim.cpp
