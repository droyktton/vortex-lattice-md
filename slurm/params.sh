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
        PRINT_INTERVAL MSD_WINDOW MSD_ORIGIN_SPACING \
        PIN_NP PIN_AP PIN_RP DISORDER_SEED CORRELATED_DISORDER \
        PLANE_N PLANE_AP PLANE_RP SHIFTED_FORCE \
        FDC FAC FREQ DRIVE_ANGLE VELOCITY_INTERVAL SWEEP_PARAM SWEEP_T 2>/dev/null || true)"
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

# Quenched disorder: only passed on when requested (PIN_NP or PIN_AP set and
# nonzero), so sweeps without disorder run exactly the same command as before.
if [ "${PIN_NP:-0}" != "0" ] || [ "${PIN_AP:-0}" != "0" ]; then
    SIM_ARGS+=(
        --np "${PIN_NP:-0}"                             # pinning centers per unit area, per layer
        --Ap "${PIN_AP:-0}"                             # pin strength
        --rp "${PIN_RP:-0.1}"                           # pin range
        --disorder-seed "${DISORDER_SEED:-1}"           # disorder realization
    )
    if [ "${CORRELATED_DISORDER:-0}" != "0" ]; then
        SIM_ARGS+=(--correlated-disorder)               # same pins in every layer (columnar)
    fi
fi

# Planar disorder (planes x = x_i, parallel to y and z): likewise only passed
# on when PLANE_N or PLANE_AP is set and nonzero. Additive to the pins above.
if [ "${PLANE_N:-0}" != "0" ] || [ "${PLANE_AP:-0}" != "0" ]; then
    SIM_ARGS+=(
        --n-planes "${PLANE_N:-0}"                      # number of planes, x_i uniform in [0, Lx)
        --Ap-plane "${PLANE_AP:-0}"                     # plane strength
        --rp-plane "${PLANE_RP:-0.1}"                   # plane range
        --disorder-seed "${DISORDER_SEED:-1}"           # disorder realization
    )
fi

# Shifted-force cutoff (force continuous at the cutoff): only when asked for.
if [ "${SHIFTED_FORCE:-0}" != "0" ]; then
    SIM_ARGS+=(--shifted-force)
fi

# External drive: likewise only passed on when FDC or FAC is set and nonzero,
# or when the sweep itself runs over the drive (SWEEP_PARAM below).
if [ "${FDC:-0}" != "0" ] || [ "${FAC:-0}" != "0" ] || [ "${SWEEP_PARAM:-T}" != "T" ]; then
    SIM_ARGS+=(
        --Fdc "${FDC:-0}"                               # DC drive force
        --Fac "${FAC:-0}"                               # AC drive amplitude
        --freq "${FREQ:-0}"                             # AC frequency
        --drive-angle "${DRIVE_ANGLE:-0}"               # drive direction, degrees from x
    )
    if [ -n "${VELOCITY_INTERVAL:-}" ]; then
        SIM_ARGS+=(--velocity-interval "$VELOCITY_INTERVAL")  # steps between velocity.dat rows
    fi
fi

# What the first column of the temperatures file sets: T (default), or Fdc,
# Fac or freq at the fixed temperature SWEEP_T. collect.sbatch needs these
# too (to find the run folders), so they also live in their own array.
SWEEP_ARGS=()
if [ -n "${SWEEP_PARAM:-}" ] && [ "$SWEEP_PARAM" != "T" ]; then
    SWEEP_ARGS=(--sweep-param "$SWEEP_PARAM" --T "${SWEEP_T:-0.01}")
fi
SIM_ARGS+=("${SWEEP_ARGS[@]}")

echo "Sim parameters: ${SIM_ARGS[*]}"
