#!/bin/bash
# Smoke test of the external drive (see "External drive" in README.md). Runs
# short simulations on one GPU and then tools/drive_check.py on them. Checks:
# a clean lattice at T = 0 moves rigidly at exactly v = F (DC, any angle,
# and AC), at T > 0 its mean velocity is F and its drift-subtracted
# diffusion matches the undriven one, strong pinning holds the lattice at
# a weak drive and lets it slide at a strong one, a restart continues
# velocity.dat, the dt warnings fire, and tsweep.py --sweep-param Fdc
# --hysteresis runs end to end.
#
# Submit from the repo root, after slurm/build.sh:
#   sbatch tools/drive_smoke.sbatch
#SBATCH --job-name=drive_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/drive_smoke_%j.out
set -uo pipefail

REPO_ROOT="$SLURM_SUBMIT_DIR"
source "$REPO_ROOT/slurm/env.sh"
cd "$REPO_ROOT"

O="${OUT_DIR:-$REPO_ROOT/runs/drive_smoke}"
rm -rf "$O"; mkdir -p "$O"
echo "node $SLURMD_NODENAME"; nvidia-smi -L

SIM="$REPO_ROOT/vortex_sim"
COMMON=(--nx 20 --ny 20 --nz 4 --dt 0.01 --print-interval 20 --seed 42)
PIN=(--np 1 --Ap 0.3 --rp 0.2)
run() {  # name binary args...
    local name=$1 bin=$2; shift 2
    mkdir -p "$O/$name"
    ( cd "$O/$name" && "$bin" "$@" > stdout.txt 2> stderr.txt; echo "exit=$?" >> stderr.txt )
    echo "[$name] done"
}

# Rigid motion of a clean lattice at T = 0
run dc_T0        "$SIM"                       "${COMMON[@]}" --T 0 --steps 1000 --Fdc 0.5 --drive-angle 30
run dc_T0_on2    "$REPO_ROOT/vortex_sim_on2"  "${COMMON[@]}" --T 0 --steps 1000 --Fdc 0.5 --drive-angle 30
run ac_T0        "$SIM"                       "${COMMON[@]}" --T 0 --steps 2000 --Fac 0.5 --freq 0.2 --velocity-interval 5
# Thermal: driven vs undriven diffusion of a clean lattice (liquid, T = 0.03)
run nodrive_T    "$SIM"                       "${COMMON[@]}" --T 0.03 --steps 6000
run dc_T         "$SIM"                       "${COMMON[@]}" --T 0.03 --steps 6000 --Fdc 0.3
# Strong pinning: weak drive stays pinned, strong drive slides
run pin_weak     "$SIM"  "${COMMON[@]}" "${PIN[@]}" --dt 0.005 --T 0.002 --steps 6000 --Fdc 0.02
run pin_strong   "$SIM"  "${COMMON[@]}" "${PIN[@]}" --dt 0.005 --T 0.002 --steps 6000 --Fdc 1.5
# Restart: 1000 steps, then resume to 2000 in the same folder
run rs           "$SIM"  "${COMMON[@]}" --T 0.01 --steps 1000 --Fdc 0.3
( cd "$O/rs" && "$SIM" "${COMMON[@]}" --T 0.01 --steps 2000 --Fdc 0.3 \
    --restart config_step_999.dat > stdout2.txt 2> stderr2.txt; echo "exit=$?" >> stderr2.txt )
echo "[rs resume] done"
# Warnings
run warn_dt      "$SIM"  "${COMMON[@]}" --T 0.01 --steps 20 --Fdc 20
run warn_ac      "$SIM"  "${COMMON[@]}" --T 0.01 --steps 20 --Fac 0.1 --freq 5
run err_freq     "$SIM"  "${COMMON[@]}" --T 0.01 --steps 20 --Fac 0.1

# Analysis on the thermal runs
for r in nodrive_T dc_T; do
    ( cd "$O/$r" && python3 "$REPO_ROOT/msd.py" --window 100 --origin-spacing 10 --t0-min 100 > msd.txt 2>&1 )
done
for r in dc_T0 ac_T0 dc_T pin_weak pin_strong; do
    ( cd "$O/$r" && python3 "$REPO_ROOT/velocity.py" > velocity.txt 2>&1 )
done

# Force sweep with tsweep.py: ramp Fdc up and down at fixed T, pinned
python3 "$REPO_ROOT/tsweep.py" 0.05 1.0 0.05 --sweep-param Fdc --T 0.002 --hysteresis \
    --out-dir "$O/fsweep" --vortex-sim "$SIM" --nx 20 --ny 20 --nz 4 --dt 0.005 \
    --np 1 --Ap 0.3 --rp 0.2 --steps 2000 --stride 10 --print-interval 20 \
    > "$O/fsweep.txt" 2>&1
echo "fsweep exit=$?" >> "$O/fsweep.txt"
echo "[fsweep] done"

python3 "$REPO_ROOT/tools/drive_check.py" "$O"
