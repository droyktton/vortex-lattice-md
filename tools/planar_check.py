#!/usr/bin/env python3
"""Checks on the runs made by tools/planar_smoke.sbatch.

    python3 tools/planar_check.py runs/planar_smoke

Prints PASS/FAIL per check and exits nonzero if any check fails. As in
tools/disorder_check.py, cell-list vs direct and restart vs uninterrupted
run are compared statistically, not bit for bit.
"""
import os
import re
import sys

import numpy as np

O = sys.argv[1]
results = []


def report(name, ok, info=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")


def blocks(fn, skip):
    """Blank-line-separated numeric blocks (one per layer) after `skip` header lines."""
    out, cur = [], []
    for line in open(fn).read().split("\n")[skip:]:
        if line.strip():
            cur.append([float(v) for v in line.split()])
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def config(run, step):
    """Snapshot rows sorted by global vortex id (column 5)."""
    c = np.vstack(blocks(f"{O}/{run}/config_step_{step}.dat", 3))
    return c[np.argsort(c[:, 5])]


def snapshots(run):
    return sorted(f for f in os.listdir(f"{O}/{run}") if f.startswith("config_step_"))


def same_files(a, b, names):
    return all(open(f"{O}/{a}/{n}", "rb").read() == open(f"{O}/{b}/{n}", "rb").read() for n in names)


def exited_ok(run, f="stderr.txt"):
    return "exit=0" in open(f"{O}/{run}/{f}").read()


def displacement(a, b):
    """In-plane distance between the same vortices of two snapshots."""
    return np.hypot(*(a[:, :2] - b[:, :2]).T)


def planes(run):
    return np.loadtxt(f"{O}/{run}/planes.dat", skiprows=1, ndmin=1)


def log_value(run, label):
    t = open(f"{O}/{run}/simulation.log").read()
    m = re.search(re.escape(label) + r"\s*:\s*([-\d.eE+]+)", t)
    return float(m.group(1)) if m else None


FINAL = 1999

# --- every run finished ---
for run in sorted(d for d in os.listdir(O) if os.path.isfile(f"{O}/{d}/stderr.txt")):
    report(f"{run}: exit 0", exited_ok(run))

# --- without planes: bit-identical to the reference build ---
for ref, new in [("ref_nodis", "br_nodis"), ("ref_nodis_on2", "br_nodis_on2"), ("ref_pin", "br_pin")]:
    s = snapshots(ref)
    report(f"{new} == {ref}: configs bit-identical",
           s == snapshots(new) and same_files(ref, new, s + ["configIni.dat"]), f"{len(s)} snapshots")
    report(f"{new} == {ref}: simulation.log identical", same_files(ref, new, ["simulation.log"]))
    report(f"{new}: no planes.dat", not os.path.exists(f"{O}/{new}/planes.dat"))

# --- with planes: reproducibility, and cell-list vs direct ---
s = snapshots("br_pl")
report("br_pl == br_pl_rep: bit-identical", s == snapshots("br_pl_rep") and same_files("br_pl", "br_pl_rep", s))
report("planes.dat identical in cell-list and direct builds", same_files("br_pl", "br_pl_on2", ["planes.dat"]))
d = displacement(config("br_pl", FINAL), config("br_pl_on2", FINAL))
report("cell-list vs direct with planes: agree statistically",
       np.median(d) < 1e-3 and (d > 1e-3).mean() < 0.1,
       f"median {np.median(d):.1e}, max {d.max():.1e}, diverged (>1e-3) {(d > 1e-3).mean():.1%}")

# --- planes.dat and log section ---
Lx = log_value("br_pl", "Box Size Lx")
for run in ["br_pl", "br_pl_ds2", "br_pl_pin"]:
    xp = planes(run)
    n = log_value(run, "Planes (N)")
    report(f"{run}: [Planar Disorder] in simulation.log", n is not None)
    report(f"{run}: planes.dat count matches the log", n == len(xp), f"{len(xp)} planes")
    report(f"{run}: planes sorted and inside the box",
           (np.diff(xp) >= 0).all() and (xp >= 0).all() and (xp < Lx).all(), f"x = {np.round(xp, 3)}")
report("--disorder-seed 2 gives other planes", not np.array_equal(planes("br_pl"), planes("br_pl_ds2")))
report("br_pl_pin: point pins on too", os.path.exists(f"{O}/br_pl_pin/pins.dat"))
report("br_pl_pin: same planes as br_pl", np.array_equal(planes("br_pl"), planes("br_pl_pin")))


# --- physics sanity: vortices are attracted to the planes ---
def fraction_near(c, xp, Lx, rp=0.3):
    dx = c[:, None, 0] - xp[None, :]
    dx -= Lx * np.round(dx / Lx)
    return (np.abs(dx).min(axis=1) < rp / 2).mean()


xp = planes("br_pl")
f_pl = fraction_near(config("br_pl", FINAL), xp, Lx)
f_free = fraction_near(config("br_nodis", FINAL), xp, Lx)
report("vortices attracted to the planes", f_pl > 1.5 * f_free,
       f"fraction within rp/2 of a plane: {f_pl:.3f} with planes, {f_free:.3f} without")

# --- dt warning ---
report("br_warn: dt warning printed", "too large to resolve the planar" in open(f"{O}/br_warn/stderr.txt").read())
report("br_pl: no dt warning", "too large" not in open(f"{O}/br_pl/stderr.txt").read())

# --- restart vs uninterrupted run (snapshots have 6 decimals) ---
report("rs_pl: resume exit 0", exited_ok("rs_pl", "stderr2.txt"))
report("rs_pl: no parameter mismatch warning", "mismatch" not in open(f"{O}/rs_pl/stderr2.txt").read())
d = displacement(config("rs_pl", FINAL), config("br_pl", FINAL))
report("rs_pl: restart vs uninterrupted run agree", np.median(d) < 1e-3 and d.max() < 0.1,
       f"median {np.median(d):.1e}, max {d.max():.1e}")

print(f"\n{sum(results)}/{len(results)} PASS")
sys.exit(0 if all(results) else 1)
