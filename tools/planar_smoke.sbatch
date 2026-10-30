#!/bin/bash
# Smoke test of the planar disorder (see "Planar disorder" in README.md).
# Runs a set of short simulations on one GPU and then tools/planar_check.py
# on them. Checks: without planes the output is bit-identical to a reference
# build of the code before planes existed (REF_REV, built here from git),
# with and without point pins; with planes runs are reproducible, planes.dat
# matches the log, another --disorder-seed gives other planes, vortices are
# attracted to the planes, cell-list and direct agree, the dt warning fires,
# and restarts stay close to a continuous run.
#
# Submit from the repo root, after slurm/build.sh:
#   sbatch tools/planar_smoke.sbatch
#   sbatch --export=ALL,REF_REV=<commit> tools/planar_smoke.sbatch
#SBATCH --job-name=planar_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/planar_smoke_%j.out
set -uo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

# Last commit before planar disorder was added.
REF_REV="${REF_REV:-e1fd008}"
O="${OUT_DIR:-$REPO_ROOT/runs/planar_smoke}"
rm -rf "$O"; mkdir -p "$O/ref_bin"

echo "node $SLURMD_NODENAME"; nvidia-smi -L
echo "reference build: $REF_REV"
git show "$REF_REV:main.cu" > "$O/ref_bin/main.cu" || exit 1
for mode in 1 0; do
    nvcc $VORTEX_NVCCFLAGS -I"$RANDOM123_DIR" -DUSE_CELL_LIST=$mode \
        "$O/ref_bin/main.cu" -o "$O/ref_bin/vortex_sim_$mode" || exit 1
done

COMMON=(--nx 20 --ny 20 --nz 4 --T 0.01 --dt 0.005 --print-interval 500 --seed 42)
PIN=(--np 1 --Ap 0.3 --rp 0.2)
PL=(--n-planes 4 --Ap-plane 0.3 --rp-plane 0.3)
run() {  # name binary args...
    local name=$1 bin=$2; shift 2
    mkdir -p "$O/$name"
    ( cd "$O/$name" && "$bin" "$@" > stdout.txt 2> stderr.txt; echo "exit=$?" >> stderr.txt )
    echo "[$name] done"
}

run ref_nodis      "$O/ref_bin/vortex_sim_1"   "${COMMON[@]}" --steps 2000
run br_nodis       "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" --steps 2000
run ref_nodis_on2  "$O/ref_bin/vortex_sim_0"   "${COMMON[@]}" --steps 2000
run br_nodis_on2   "$REPO_ROOT/vortex_sim_on2" "${COMMON[@]}" --steps 2000
run ref_pin        "$O/ref_bin/vortex_sim_1"   "${COMMON[@]}" "${PIN[@]}" --steps 2000
run br_pin         "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" "${PIN[@]}" --steps 2000
run br_pl          "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" "${PL[@]}" --steps 2000
run br_pl_rep      "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" "${PL[@]}" --steps 2000
run br_pl_on2      "$REPO_ROOT/vortex_sim_on2" "${COMMON[@]}" "${PL[@]}" --steps 2000
run br_pl_ds2      "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" "${PL[@]}" --disorder-seed 2 --steps 2000
run br_pl_pin      "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" "${PL[@]}" "${PIN[@]}" --steps 2000
run br_warn        "$REPO_ROOT/vortex_sim"     "${COMMON[@]}" --n-planes 4 --Ap-plane 0.3 --rp-plane 0.05 --steps 200

# Restart with planes: 1000 steps, then resume to 2000.
run rs_pl "$REPO_ROOT/vortex_sim" "${COMMON[@]}" "${PL[@]}" --steps 1000
( cd "$O/rs_pl" && "$REPO_ROOT/vortex_sim" "${COMMON[@]}" "${PL[@]}" --steps 2000 \
    --restart config_step_999.dat > stdout2.txt 2> stderr2.txt; echo "exit=$?" >> stderr2.txt )
echo "[rs_pl resume] done"

python3 "$REPO_ROOT/tools/planar_check.py" "$O"
