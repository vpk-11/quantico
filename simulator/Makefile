CXX ?= c++
# -ffp-contract=off: no fused multiply-add, so results match the Python reference bit for bit.
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -ffp-contract=off

build/sim: src/sim.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf build
