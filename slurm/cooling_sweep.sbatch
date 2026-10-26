#!/bin/bash
# Cooling-only sweep (the down half of hysteresis_sweep.sbatch) over a
# temperatures file (default slurm/temperatures.tsv, same format as for
# sweep_array.sbatch). Rows are run from the HIGHEST T to the lowest,
# whatever order they appear in the file, each T restarting from the
# previous T's final configuration (tsweep.py --hysteresis). The first
# (hottest) T starts from the pristine ordered lattice, so give it enough
# steps to melt fully before the cooling starts. Runs sequentially on ONE
# GPU, like hysteresis_sweep.sbatch.
#
# Submit from the repo root, after vortex_sim has been built:
#   sbatch slurm/cooling_sweep.sbatch
#   sbatch --export=ALL,TEMPS_FILE=$PWD/my_temps.tsv,OUT_DIR=$PWD/runs/cool2 slurm/cooling_sweep.sbatch
# Physical/run parameters (K, DT, NX, SEED, ...): see slurm/params.sh.
#SBATCH --job-name=vortex_cooling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/cooling_%j.out
set -euo pipefail

# SLURM copies this script into a spool dir before running it, so
# BASH_SOURCE can't be used to find the repo -- use where sbatch was
# invoked from instead (submit from the repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"
source "$REPO_ROOT/slurm/params.sh"

TEMPS_FILE="${TEMPS_FILE:-$REPO_ROOT/slurm/temperatures.tsv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/cooling}"

# Hottest first: sort the rows by T, descending.
mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$TEMPS_FILE" | sort -g -r -k1,1)

T_LIST=(); STEPS_LIST=(); STEPMIN_LIST=(); STRIDE_LIST=()
for line in "${LINES[@]}"; do
    read -r T STEPS STEP_MIN STRIDE <<< "$line"
    T_LIST+=("$T"); STEPS_LIST+=("$STEPS"); STEPMIN_LIST+=("$STEP_MIN"); STRIDE_LIST+=("$STRIDE")
done

echo "Cooling sweep: ${#T_LIST[@]} legs"
echo "T sequence: ${T_LIST[*]}"

python3 "$REPO_ROOT/tsweep.py" "${T_LIST[@]}" \
    --out-dir "$OUT_DIR" \
    --vortex-sim "$REPO_ROOT/vortex_sim" \
    --hysteresis \
    --steps "${STEPS_LIST[@]}" --step-min "${STEPMIN_LIST[@]}" --stride "${STRIDE_LIST[@]}" \
    "${SIM_ARGS[@]}"

echo "Cooling sweep complete. Results in $OUT_DIR/summary.dat, $OUT_DIR/summary_*.png"
