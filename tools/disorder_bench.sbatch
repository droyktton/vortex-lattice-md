#!/bin/bash
# Cost of the quenched disorder (see "Quenched disorder > Performance" in
# README.md): ms/step of vortex_sim (cell-list build) over a grid of pin
# densities NP_LIST and ranges RP_LIST, each against a run without disorder
# at the same dt. dt is the suggested one for each rp (capped at DT_MAX), so
# the summary also gives the cost per unit physical time. Everything runs on
# one GPU, so the numbers are comparable. tools/disorder_bench_summary.py
# prints the tables at the end.
#
# Submit from the repo root, after slurm/build.sh:
#   sbatch tools/disorder_bench.sbatch
#   sbatch --export=ALL,SIZES="256 16",AP=0.5 tools/disorder_bench.sbatch
#SBATCH --job-name=disorder_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/disorder_bench_%j.out
set -uo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

O="${OUT_DIR:-$REPO_ROOT/runs/disorder_bench}"
SIZES="${SIZES:-64 16;128 16}"     # "nx nz" pairs (ny = nx), separated by ';'
NP_LIST="${NP_LIST:-0.25 1 4}"
RP_LIST="${RP_LIST:-0.05 0.1 0.2 0.4}"
AP="${AP:-0.3}"
T="${T:-0.01}"
DT_MAX="${DT_MAX:-0.005}"
STEPS="${STEPS:-3000}"
REPS="${REPS:-2}"
COL_RP="${COL_RP:-0.2}"      # columnar disorder is measured at np = 1 and this rp
rm -rf "$O"; mkdir -p "$O"

echo "node $SLURMD_NODENAME"; nvidia-smi -L

# Suggested dt, as vortex_sim computes it: min(0.1 rp / Fmax, rp^2 / 2T), capped.
dt_for() {
    python3 -c "rp, Ap, T = $1, $AP, $T
dt = 0.1 * rp / (0.8577639 * abs(Ap) / rp)
if T > 0: dt = min(dt, rp * rp / (2 * T))
print(f'{min($DT_MAX, dt):.6g}')"
}

one() {  # nx nz np rp correlated dt rep
    local d="$O/n$1_z$2_np$3_rp$4_c$5_dt$6_r$7"
    local extra=()
    [ "$3" != 0 ] && extra=(--np "$3" --Ap "$AP" --rp "$4")
    [ "$5" = 1 ] && extra+=(--correlated-disorder)
    mkdir -p "$d"
    ( cd "$d" && "$REPO_ROOT/vortex_sim" --nx "$1" --ny "$1" --nz "$2" --T "$T" --dt "$6" \
        --steps "$STEPS" --print-interval 100000000 --seed 7 "${extra[@]}" > out.txt 2> err.txt
      rm -f config_step_*.dat configIni.dat )
}

IFS=';' read -ra SIZE_ARR <<< "$SIZES"
for size in "${SIZE_ARR[@]}"; do
    read -r nx nz <<< "$size"
    for rep in $(seq 1 "$REPS"); do
        for rp in $RP_LIST; do
            dt=$(dt_for "$rp")
            one "$nx" "$nz" 0 0 0 "$dt" "$rep"
            for np in $NP_LIST; do one "$nx" "$nz" "$np" "$rp" 0 "$dt" "$rep"; done
        done
        # columnar at one point, to compare with uncorrelated
        one "$nx" "$nz" 1 "$COL_RP" 1 "$(dt_for "$COL_RP")" "$rep"
    done
    echo "[${nx}x${nx}x${nz}] done"
done

python3 "$REPO_ROOT/tools/disorder_bench_summary.py" "$O" --dt-ref "$DT_MAX"
