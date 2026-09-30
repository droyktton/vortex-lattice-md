#!/bin/bash
# One array task = one temperature. Submitted by submit_sweep.sh, which sets
# --array to match the number of (non-comment) lines in the temperatures file.
#SBATCH --job-name=vortex_tsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/sweep_%A_%a.out
set -euo pipefail

# SLURM copies this script into a spool dir before running it, so
# BASH_SOURCE can't be used to find the repo -- use where sbatch was
# invoked from instead (submit_sweep.sh always submits from repo root).
REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

TEMPS_FILE="${TEMPS_FILE:-$REPO_ROOT/slurm/temperatures.tsv}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs}"

mapfile -t LINES < <(grep -vE '^[[:space:]]*(#|$)' "$TEMPS_FILE")
LINE="${LINES[$SLURM_ARRAY_TASK_ID]}"
read -r T STEPS STEP_MIN STRIDE <<< "$LINE"

echo "[task $SLURM_ARRAY_TASK_ID] T=$T steps=$STEPS step_min=$STEP_MIN stride=$STRIDE" \
     "(node $SLURMD_NODENAME, GPU $CUDA_VISIBLE_DEVICES)"

python3 "$REPO_ROOT/tsweep.py" "$T" \
    --out-dir "$OUT_DIR" \
    --vortex-sim "$REPO_ROOT/vortex_sim" \
    --steps "$STEPS" --step-min "$STEP_MIN" --stride "$STRIDE" \
    --nx "${NX:-30}" --ny "${NY:-30}" --nz "${NZ:-4}" \
    --print-interval "${PRINT_INTERVAL:-15}"
