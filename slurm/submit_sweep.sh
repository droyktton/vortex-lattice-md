#!/bin/bash
# Submit the full temperature sweep: one GPU array task per temperature in
# the temperatures file, then a CPU job (dependent on the whole array
# succeeding) that collects everything into runs/summary.dat + plots.
#
# Usage:
#   slurm/submit_sweep.sh                          # uses slurm/temperatures.tsv
#   slurm/submit_sweep.sh path/to/other_temps.tsv
#   MAX_CONCURRENT=4 slurm/submit_sweep.sh          # throttle simultaneous array tasks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEMPS_FILE="${1:-slurm/temperatures.tsv}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"

if [[ "$TEMPS_FILE" = /* ]]; then
    TEMPS_ABS="$TEMPS_FILE"
else
    TEMPS_ABS="$REPO_ROOT/$TEMPS_FILE"
fi

mkdir -p slurm/logs

N=$(grep -vE '^[[:space:]]*(#|$)' "$TEMPS_ABS" | wc -l)
if [ "$N" -eq 0 ]; then
    echo "No temperatures found in $TEMPS_FILE" >&2
    exit 1
fi
LAST=$((N - 1))

if [ ! -x ./vortex_sim ]; then
    echo "vortex_sim not built yet -- building it now."
    slurm/build.sh
fi

ARRAY_JOB=$(sbatch --parsable \
    --array=0-${LAST}%${MAX_CONCURRENT} \
    --export=ALL,TEMPS_FILE="$TEMPS_ABS" \
    slurm/sweep_array.sbatch)
echo "Submitted sweep array: job $ARRAY_JOB ($N temperatures, up to $MAX_CONCURRENT at once)"

COLLECT_JOB=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB} \
    --export=ALL,TEMPS_FILE="$TEMPS_ABS" \
    slurm/collect.sbatch)
echo "Submitted collect job $COLLECT_JOB (runs once the whole sweep succeeds)"
echo
echo "Track with:  squeue -u $USER"
echo "Logs in:     slurm/logs/sweep_${ARRAY_JOB}_*.out, slurm/logs/collect_${COLLECT_JOB}.out"
echo "Result in:   ${OUT_DIR:-$REPO_ROOT/runs}/summary.dat, summary_*.png"
