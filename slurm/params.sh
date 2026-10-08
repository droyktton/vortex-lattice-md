#!/bin/bash
# Physical + run parameters shared by sweep_array.sbatch and
# hysteresis_sweep.sbatch. Source this (don't execute it); it fills the
# SIM_ARGS array with the matching tsweep.py flags.
#
# Every parameter can be overridden with an environment variable of the same
# name, e.g.
#   sbatch --export=ALL,K=1.5,DT=0.005,OUT_DIR=$PWD/runs/k1.5 slurm/hysteresis_sweep.sbatch
# or by putting VAR=value lines in a file and pointing PARAMS_FILE at it:
#   sbatch --export=ALL,PARAMS_FILE=$PWD/my_params.sh slurm/hysteresis_sweep.sbatch
# Variables set in the environment take precedence over PARAMS_FILE.
# Defaults are the values the example_output/ runs used.

if [ -n "${PARAMS_FILE:-}" ]; then
    if [ ! -f "$PARAMS_FILE" ]; then
        echo "Error: PARAMS_FILE '$PARAMS_FILE' not found" >&2
        exit 1
    fi
    # A full copy of this script as PARAMS_FILE would source itself forever
    # and hang the job without ever starting vortex_sim.
    if grep -q 'SIM_ARGS' "$PARAMS_FILE"; then
        echo "Error: PARAMS_FILE '$PARAMS_FILE' looks like a copy of slurm/params.sh;" \
             "it should only hold VAR=value lines (e.g. K=2.5)" >&2
        exit 1
    fi
    _VORTEX_ENV_OVERRIDES="$(declare -p NX NY NZ A0 K DT CUTOFF SKIN SEED \
        PRINT_INTERVAL MSD_WINDOW MSD_ORIGIN_SPACING 2>/dev/null || true)"
    source "$PARAMS_FILE"
    eval "$_VORTEX_ENV_OVERRIDES"
    unset _VORTEX_ENV_OVERRIDES
fi

SIM_ARGS=(
    --nx "${NX:-30}" --ny "${NY:-30}" --nz "${NZ:-4}"   # lattice size (vortices per layer, layers)
    --a0 "${A0:-1.0}"                                   # lattice constant
    --k "${K:-0.5}"                                     # inter-layer spring constant
    --dt "${DT:-0.01}"                                  # timestep
    --cutoff "${CUTOFF:-3.0}"                           # interaction cutoff radius
    --skin "${SKIN:-0.5}"                               # Verlet skin width
    --seed "${SEED:-1234567}"                           # RNG seed
    --print-interval "${PRINT_INTERVAL:-15}"            # steps between snapshots
    --msd-window "${MSD_WINDOW:-100}"                   # msd.py window, in snapshots
    --msd-origin-spacing "${MSD_ORIGIN_SPACING:-10}"    # msd.py origin spacing, in snapshots
)

echo "Sim parameters: ${SIM_ARGS[*]}"
