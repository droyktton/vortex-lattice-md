#!/usr/bin/env python3
"""Checks on the runs made by tools/drive_smoke.sbatch.

    python3 tools/drive_check.py runs/drive_smoke

Prints PASS/FAIL per check and exits nonzero if any check fails.
"""
import glob
import os
import re
import sys

import numpy as np

O = sys.argv[1]
results = []


def report(name, ok, info=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")


def text(run, f):
    p = f"{O}/{run}/{f}"
    return open(p).read() if os.path.exists(p) else ""


def velocity(run):
    names = open(f"{O}/{run}/velocity.dat").readline().split()
    d = np.loadtxt(f"{O}/{run}/velocity.dat", skiprows=1, ndmin=2)
    return {n: d[:, i] for i, n in enumerate(names)}


def config(run, step):
    """(x, y) of every vortex, sorted by global id."""
    rows = [l.split() for l in open(f"{O}/{run}/config_step_{step}.dat").read().split("\n")[3:] if l.strip()]
    a = np.array(rows, dtype=float)
    a = a[np.argsort(a[:, 5])]
    return a[:, 0], a[:, 1]


def box(run):
    t = text(run, "simulation.log")
    return (float(re.search(r"Box Size Lx\s*:\s*([\d.]+)", t).group(1)),
            float(re.search(r"Box Size Ly\s*:\s*([\d.]+)", t).group(1)))


def last_step(run):
    return max(int(re.search(r"config_step_(\d+)", f).group(1)) for f in glob.glob(f"{O}/{run}/config_step_*.dat"))


def mean_file(run, f):
    return np.loadtxt(f"{O}/{run}/{f}", skiprows=1, ndmin=2)[0]


for run in sorted(d for d in os.listdir(O) if os.path.isfile(f"{O}/{d}/stderr.txt")):
    if run == "err_freq":
        continue
    report(f"{run}: exit 0", "exit=0" in text(run, "stderr.txt"))

# --- T = 0, clean lattice: rigid motion at exactly v = F ---
a = np.radians(30)
Fx, Fy = 0.5 * np.cos(a), 0.5 * np.sin(a)
for run in ["dc_T0", "dc_T0_on2"]:
    v = velocity(run)
    err = max(np.abs(v["vx"] - Fx).max(), np.abs(v["vy"] - Fy).max(),
              *(np.abs(v[f"vx_z{z}"] - Fx).max() for z in range(4)))
    report(f"{run}: every layer moves at v = F", err < 1e-4, f"max |v - F| = {err:.1e}")
    report(f"{run}: drive column is F", np.allclose(v["Fx"], Fx) and np.allclose(v["Fy"], Fy))
    # Rigid translation of the whole lattice by F t (mod the box)
    Lx, Ly = box(run)
    n = last_step(run)
    x0, y0 = config(run, 0)
    x1, y1 = config(run, n)
    t = n * 0.01   # config_step_k is written after step k's update: n steps apart
    dx = (x1 - x0 - Fx * t + Lx / 2) % Lx - Lx / 2
    dy = (y1 - y0 - Fy * t + Ly / 2) % Ly - Ly / 2
    # The center of mass moves by exactly F t; single vortices scatter around it
    # by float32 rounding of x += F dt (positions ~ L, so ~1e-6 per step).
    cm = max(abs(dx.mean()), abs(dy.mean()))
    err = max(np.abs(dx).max(), np.abs(dy).max())
    report(f"{run}: lattice translated rigidly by F t", cm < 1e-4 and err < 5e-3,
           f"center of mass off by {cm:.1e}, single vortices by up to {err:.1e} (float32)")
v1, v2 = velocity("dc_T0"), velocity("dc_T0_on2")
report("dc_T0: cell-list and direct give the same velocities",
       np.abs(v1["vx"] - v2["vx"]).max() < 1e-5 and np.abs(v1["vy"] - v2["vy"]).max() < 1e-5)

v = velocity("ac_T0")
F = 0.5 * np.sin(2 * np.pi * 0.2 * v["t"])
report("ac_T0: drive column is Fac sin(2 pi f t)", np.abs(v["Fx"] - F).max() < 1e-5)
report("ac_T0: v(t) follows F(t)", np.abs(v["vx"] - F).max() < 1e-4 and np.abs(v["vy"]).max() < 1e-4,
       f"max |v - F| = {np.abs(v['vx'] - F).max():.1e}")
m = mean_file("ac_T0", "velocity_mean.dat")
report("ac_T0: mean velocity ~ 0 over whole periods", abs(m[0]) < 0.01, f"V_par = {m[0]:.2e}")

# --- T > 0, clean lattice: mean velocity F, diffusion unchanged by the drive ---
m = mean_file("dc_T", "velocity_mean.dat")
report("dc_T: V_par = F, V_perp = 0", abs(m[0] - 0.3) < 1e-3 and abs(m[1]) < 1e-3,
       f"V_par = {m[0]:.5f}, V_perp = {m[1]:.1e}")
D0 = mean_file("nodrive_T", "msd_diffusion.dat")
D1 = mean_file("dc_T", "msd_diffusion.dat")
report("dc_T: drift subtracted, D within 30% of the undriven run", abs(D1[0] / D0[0] - 1) < 0.3,
       f"D undriven {D0[0]:.3e}, driven {D1[0]:.3e} (D_par {D1[4]:.3e}, D_perp {D1[5]:.3e})")
report("nodrive_T: msd.dat keeps two columns", text("nodrive_T", "msd.dat").split("\n")[0].split() == ["t", "MSD"])
report("dc_T: msd.dat has parallel/transverse columns",
       text("dc_T", "msd.dat").split("\n")[0].split() == ["t", "MSD", "MSD_par", "MSD_perp"])
report("nodrive_T: no velocity.dat, no [Driving Force]",
       not os.path.exists(f"{O}/nodrive_T/velocity.dat") and "[Driving Force]" not in text("nodrive_T", "simulation.log"))
report("dc_T: [Driving Force] in simulation.log", "[Driving Force]" in text("dc_T", "simulation.log"))

# --- pinning: weak drive pinned, strong drive slides ---
w = mean_file("pin_weak", "velocity_mean.dat")
s = mean_file("pin_strong", "velocity_mean.dat")
report("pin_weak: pinned (V < 10% of F)", w[0] < 0.1 * 0.02, f"V_par = {w[0]:.2e} for F = 0.02")
report("pin_strong: sliding (V > 50% of F)", s[0] > 0.5 * 1.5, f"V_par = {s[0]:.3f} for F = 1.5")

# --- restart appends to velocity.dat ---
report("rs: resume exit 0", "exit=0" in text("rs", "stderr2.txt"))
v = velocity("rs")
steps = v["step"].astype(int)
expected = sorted(set(list(range(0, 1000, 20)) + [999] + list(range(1000, 2000, 20)) + [1999]))
report("rs: velocity.dat continued across the restart", list(steps) == expected and
       text("rs", "velocity.dat").count("step") == 1, f"{len(steps)} rows")

# --- warnings and errors ---
report("warn_dt: drift-per-step warning", "the drive moves vortices" in text("warn_dt", "stderr.txt"))
report("warn_ac: AC period warning", "AC period" in text("warn_ac", "stderr.txt"))
report("err_freq: --Fac without --freq is an error", "needs a positive --freq" in text("err_freq", "stderr.txt")
       and "exit=0" not in text("err_freq", "stderr.txt"))

# --- tsweep.py --sweep-param Fdc --hysteresis ---
fs = open(f"{O}/fsweep.txt").read()
report("fsweep: tsweep.py exit 0", "fsweep exit=0" in fs)
if os.path.exists(f"{O}/fsweep/summary.dat"):
    hdr = open(f"{O}/fsweep/summary.dat").readline().split()
    d = np.loadtxt(f"{O}/fsweep/summary.dat", skiprows=1, ndmin=2)
    report("fsweep: summary.dat has Fdc and V columns",
           hdr == ["Fdc", "D", "defect_fraction", "Sq_max", "mean_u2", "V_par", "V_perp"])
    report("fsweep: folders named by Fdc",
           all(os.path.isdir(f"{O}/fsweep/{i:03d}_Fdc_{v}") for i, v in enumerate([0.05, 1.0, 0.05])))
    report("fsweep: V grows with F", d[1, 5] > d[0, 5] and d[1, 5] > d[2, 5],
           f"V_par = {np.round(d[:, 5], 4)} for Fdc = {d[:, 0]}")
    report("fsweep: V vs F plot", os.path.exists(f"{O}/fsweep/summary_V_vs_Fdc.png"))
else:
    report("fsweep: summary.dat written", False)

print(f"\n{sum(results)}/{len(results)} PASS")
sys.exit(0 if all(results) else 1)
