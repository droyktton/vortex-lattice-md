NVCC     ?= nvcc
CXXSTD   ?= c++14
OPT      ?= -O2

# Random123 ships boxmuller.hpp separately from the Debian librandom123-dev
# package on this machine, so fall back to the sibling checkout if the system
# headers don't have it. Override with: make RANDOM123_DIR=/some/path/include
RANDOM123_DIR ?= $(if $(wildcard /usr/include/Random123/boxmuller.hpp),/usr/include,/home/koltona/Codigos/random123/include)

NVCCFLAGS = $(OPT) -std=$(CXXSTD) -I$(RANDOM123_DIR)

SRC           = main.cu
TARGET_CELL   = vortex_sim
TARGET_DIRECT = vortex_sim_on2

.PHONY: all cell direct run run_direct clean clean-data

all: cell direct

cell: $(TARGET_CELL)

direct: $(TARGET_DIRECT)

$(TARGET_CELL): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DUSE_CELL_LIST=1 $< -o $@

$(TARGET_DIRECT): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DUSE_CELL_LIST=0 $< -o $@

run: cell
	./$(TARGET_CELL)

run_direct: direct
	./$(TARGET_DIRECT)

clean:
	rm -f $(TARGET_CELL) $(TARGET_DIRECT)

# Removes simulation output, not the binaries. Separate from `clean` since
# these are results you may still want.
clean-data:
	rm -f configIni.dat config_step_*.dat simulation.log
