#!/bin/bash
# Full hysteresis sweep over a temperatures file (default
# slurm/temperatures.tsv, same format as for sweep_array.sbatch): ramps up
# through every row in file order, then back down through the same rows minus
# the last (peak) one, each T restarting from the previous T's final
# configuration (tsweep.py --hysteresis). Unlike sweep_array.sbatch this
# runs everything sequentially on ONE GPU -- hysteresis needs each leg to
# start from the previous leg's end state, so it can't be split across
# nodes like the one-way sweep is.
#
# Submit from the repo root, after vortex_sim has been built:
#   sbatch slurm/hysteresis_sweep.sbatch
#   sbatch --export=ALL,TEMPS_FILE=$PWD/my_temps.tsv,OUT_DIR=$PWD/runs/hyst2 slurm/hysteresis_sweep.sbatch
#SBATCH --job-name=vortex_hysteresis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/hysteresis_%j.out
set -euo pipefail

# SLURM copies this script into a spool dir before running it, so
# BASH_SOURCE can't be used to find the repo -- use where sbatch was
# invoked from instead (submit from the repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

TEMPS_FILE="${TEMPS_FILE:-$REPO_ROOT/slurm/temperatures.tsv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/hysteresis}"

mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$TEMPS_FILE")

T_LIST=(); STEPS_LIST=(); STEPMIN_LIST=(); STRIDE_LIST=()
for line in "${LINES[@]}"; do
    read -r T STEPS STEP_MIN STRIDE <<< "$line"
    T_LIST+=("$T"); STEPS_LIST+=("$STEPS"); STEPMIN_LIST+=("$STEP_MIN"); STRIDE_LIST+=("$STRIDE")
done

# Down leg: same rows, reversed, minus the last (peak) entry -- the peak was
# already measured once on the way up, so this avoids an immediate
# restart-into-itself at the same T.
N=${#LINES[@]}
for ((i = N - 2; i >= 0; i--)); do
    read -r T STEPS STEP_MIN STRIDE <<< "${LINES[$i]}"
    T_LIST+=("$T"); STEPS_LIST+=("$STEPS"); STEPMIN_LIST+=("$STEP_MIN"); STRIDE_LIST+=("$STRIDE")
done

echo "Hysteresis sweep: ${#T_LIST[@]} legs (up: $N, down: $((N - 1)))"
echo "T sequence: ${T_LIST[*]}"

python3 "$REPO_ROOT/tsweep.py" "${T_LIST[@]}" \
    --out-dir "$OUT_DIR" \
    --vortex-sim "$REPO_ROOT/vortex_sim" \
    --hysteresis \
    --steps "${STEPS_LIST[@]}" --step-min "${STEPMIN_LIST[@]}" --stride "${STRIDE_LIST[@]}" \
    --nx "${NX:-30}" --ny "${NY:-30}" --nz "${NZ:-4}" \
    --print-interval "${PRINT_INTERVAL:-15}"

echo "Hysteresis sweep complete. Results in $OUT_DIR/summary.dat, $OUT_DIR/summary_*.png"
