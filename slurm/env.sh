#!/bin/bash
# Cluster-specific build/run environment for vortex-lattice-md.
# Source this (don't execute it) before building or running vortex_sim
# on this cluster: `source slurm/env.sh`.
#
# Why all of this is needed (discovered the hard way -- see slurm/README.md):
#
# - The login/compute node shells have a conda base environment active that
#   exports NVCC_PREPEND_FLAGS pointing nvcc at conda's own host compiler,
#   which pulls in a bundled Thrust/CUB (C++17-only) ahead of CUDA's own.
#   That breaks the build outright, so those env vars must be unset.
# - `module load` does not persist across separate shells, so it must be
#   re-done in whatever shell (or batch job) actually builds/runs.
# - The spack gcc modules here are missing libmpfr.so.4 unless the mpfr
#   module is also loaded explicitly (not auto-pulled as a dependency).
# - CUDA 11.1's cudafe++ front end mangles `__int128`/`__attribute__` when
#   it reparses gcc's fixincludes copy of bits/sched.h (needed on this
#   glibc/Rocky9 combo), producing bogus syntax errors in
#   /usr/include/linux/types.h. Predefining the real file's include guard
#   (_BITS_SCHED_H) makes gcc skip that file's body entirely, side-stepping
#   the bug (the project doesn't need anything from it).
# - The `gpu` partition is heterogeneous (RTX 3080 / 3080 Ti / A10, all
#   sm_86, plus a newer RTX 5070 Ti on a later architecture). Building both
#   sm_86 SASS and sm_86 PTX lets the driver JIT the PTX on whatever isn't
#   natively sm_86, so one binary runs on every node in the partition.

# The analysis scripts need a python3 with numpy/scipy/matplotlib/pandas,
# e.g. a miniconda base env. Put its python3 first on PATH regardless of how
# this job's environment was inherited (e.g. `sbatch --export=NONE` from a
# script/cron shell). Point CONDA_BASE elsewhere if yours isn't in
# ~/miniconda3; if that directory doesn't exist, whatever python3 is already
# on PATH is used. This has to come BEFORE the module loads below: conda's
# bin/ also ships its own (much newer) nvcc, and the module's CUDA 11.1
# nvcc must be the one found first.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
if [ -d "$CONDA_BASE/bin" ]; then
    export PATH="$CONDA_BASE/bin:$PATH"
fi

module purge
module load cuda/11.1.0-zj6lnzj gcc/8.2.0-cytb66h mpfr/3.1.6-ubdztn6

unset NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS CUDA_HOME CUDA_PATH

# Random123 headers: default to a checkout next to this repo
# (<parent>/random123/include); slurm/build.sh clones it there if missing.
_VORTEX_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RANDOM123_DIR="${RANDOM123_DIR:-$(dirname "$_VORTEX_REPO_ROOT")/random123/include}"
unset _VORTEX_REPO_ROOT
export VORTEX_NVCCFLAGS="-O2 -std=c++14 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -D_BITS_SCHED_H=1 -I${RANDOM123_DIR}"
