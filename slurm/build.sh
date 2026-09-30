#!/bin/bash
# Build vortex_sim (and vortex_sim_on2) with this cluster's toolchain.
# Safe to re-run any time; does a clean rebuild.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/slurm/env.sh"

if [ ! -f "$RANDOM123_DIR/Random123/boxmuller.hpp" ]; then
    echo "Random123 headers not found at $RANDOM123_DIR -- cloning..."
    git clone --depth 1 https://github.com/DEShawResearch/random123.git \
        "$(dirname "$RANDOM123_DIR")"
fi

cd "$REPO_ROOT"
make clean
make RANDOM123_DIR="$RANDOM123_DIR" NVCCFLAGS="$VORTEX_NVCCFLAGS"
echo "Built $REPO_ROOT/vortex_sim and $REPO_ROOT/vortex_sim_on2"
