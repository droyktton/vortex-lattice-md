#!/bin/bash
# Runs after every sweep_array task finishes (submitted with a dependency by
# submit_sweep.sh). Each array task already ran the full analysis pipeline
# for its own T, so this just re-reads the resulting msd_diffusion.dat/
# equil_disclinations.dat/equil_sq.dat/equil_roughness.dat from every run
# folder (--skip-sim --skip-analysis) to build the combined summary.dat and
# summary_*.png -- tsweep.py only aggregates across the temperatures given
# to one invocation, so this is the step that stitches the per-T array
# tasks back into a single sweep. Deliberately NOT re-running analyze.py/
# disclinations.py/roughness.py here: that's the expensive part of the
# pipeline (see README.md), and doing it a second time for every
# temperature is what made an earlier version of this job time out at 1h
# after redundantly recomputing analysis the array job had already done.
#SBATCH --job-name=vortex_tsweep_collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/collect_%j.out
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
T_LIST=(); STEPS_LIST=(); STEPMIN_LIST=(); STRIDE_LIST=()
for line in "${LINES[@]}"; do
    read -r T STEPS STEP_MIN STRIDE <<< "$line"
    T_LIST+=("$T"); STEPS_LIST+=("$STEPS"); STEPMIN_LIST+=("$STEP_MIN"); STRIDE_LIST+=("$STRIDE")
done

python3 "$REPO_ROOT/tsweep.py" "${T_LIST[@]}" \
    --out-dir "$OUT_DIR" --skip-sim --skip-analysis \
    --steps "${STEPS_LIST[@]}" --step-min "${STEPMIN_LIST[@]}" --stride "${STRIDE_LIST[@]}"

echo "Combined summary written to $OUT_DIR/summary.dat"
