#!/usr/bin/env python3
"""Checks on the runs made by tools/disorder_smoke.sbatch.

    python3 tools/disorder_check.py runs/disorder_smoke

Prints PASS/FAIL per check and exits nonzero if any check fails.

With disorder, the cell-list and direct builds (and a restart vs an
uninterrupted run) only agree statistically: a vortex near the saddle
between two pins falls one way or the other depending on float rounding.
Those comparisons therefore check the median displacement and the fraction
of diverged vortices, not bit identity.
"""
import filecmp
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
    return sorted(os.path.basename(f) for f in glob.glob(f"{O}/{run}/config_step_*.dat"))


def same_files(a, b, names):
    return all(filecmp.cmp(f"{O}/{a}/{n}", f"{O}/{b}/{n}", shallow=False) for n in names)


def exited_ok(run, f="stderr.txt"):
    return "exit=0" in open(f"{O}/{run}/{f}").read()


def displacement(a, b):
    """In-plane distance between the same vortices of two snapshots."""
    return np.hypot(*(a[:, :2] - b[:, :2]).T)


def disorder_log(run):
    t = open(f"{O}/{run}/simulation.log").read()
    m = re.search(r"Pins per Layer\s*:\s*([\d ]+)\s+\(Poisson, mean ([\d.]+)\)", t)
    Lx, Ly = (float(re.search(rf"Box Size L{x}\s*:\s*([\d.]+)", t).group(1)) for x in "xy")
    return "[Quenched Disorder]" in t, [int(v) for v in m.group(1).split()], float(m.group(2)), Lx, Ly


FINAL = 1999

# --- every run finished ---
for run in sorted(d for d in os.listdir(O) if os.path.isfile(f"{O}/{d}/stderr.txt")):
    report(f"{run}: exit 0", exited_ok(run))

# --- without disorder: bit-identical to the reference build ---
for ref, new in [("ref_nodis", "br_nodis"), ("ref_nodis_on2", "br_nodis_on2")]:
    s = snapshots(ref)
    report(f"{new} == {ref}: configs bit-identical",
           s == snapshots(new) and same_files(ref, new, s + ["configIni.dat"]), f"{len(s)} snapshots")
    report(f"{new} == {ref}: simulation.log identical", same_files(ref, new, ["simulation.log"]))
    report(f"{new}: no pins.dat", not os.path.exists(f"{O}/{new}/pins.dat"))

# --- with disorder: reproducibility, and cell-list vs direct ---
s = snapshots("br_pin")
report("br_pin == br_pin_rep: bit-identical", s == snapshots("br_pin_rep") and same_files("br_pin", "br_pin_rep", s))
report("pins.dat identical in cell-list and direct builds", same_files("br_pin", "br_pin_on2", ["pins.dat"]))
d = displacement(config("br_pin", FINAL), config("br_pin_on2", FINAL))
report("cell-list vs direct with disorder: agree statistically",
       np.median(d) < 1e-3 and (d > 1e-3).mean() < 0.1,
       f"median {np.median(d):.1e}, max {d.max():.1e}, diverged (>1e-3) {(d > 1e-3).mean():.1%}")

# --- pins.dat, log section, Poisson counts, columnar ---
for run in ["br_pin", "br_col", "br_pin_ds2"]:
    has_section, counts, mean, Lx, Ly = disorder_log(run)
    P = blocks(f"{O}/{run}/pins.dat", 1)
    report(f"{run}: [Quenched Disorder] in simulation.log", has_section)
    report(f"{run}: pins.dat counts match the log", [len(p) for p in P] == counts,
           f"{counts}, Poisson mean {mean:.1f}")
    z = (np.array(counts) - mean) / np.sqrt(mean)
    report(f"{run}: counts compatible with Poisson", np.all(np.abs(z) < 4), f"z = {np.round(z, 2)}")
    allp = np.vstack(P)
    report(f"{run}: pins inside the box",
           (allp[:, 0] >= 0).all() and (allp[:, 0] < Lx).all() and (allp[:, 1] >= 0).all() and (allp[:, 1] < Ly).all())
    identical = all(np.array_equal(P[0][:, :2], p[:, :2]) for p in P[1:])
    if run == "br_col":
        report("br_col: every layer identical (columnar)", identical)
    else:
        report(f"{run}: layers differ (uncorrelated)", not identical)
P1, P2 = (np.vstack(blocks(f"{O}/{r}/pins.dat", 1)) for r in ["br_pin", "br_pin_ds2"])
report("--disorder-seed 2 gives another realization", P1.shape != P2.shape or not np.array_equal(P1, P2))


# --- physics sanity: vortices are attracted to the pins ---
def fraction_near(c, pins, Lx, Ly, rp=0.2):
    near = 0
    for z, p in enumerate(pins):
        v = c[np.isclose(c[:, 2], z)]
        dx = v[:, None, 0] - p[None, :, 0]
        dy = v[:, None, 1] - p[None, :, 1]
        dx -= Lx * np.round(dx / Lx)
        dy -= Ly * np.round(dy / Ly)
        near += (np.hypot(dx, dy).min(axis=1) < rp).sum()
    return near / len(c)


*_, Lx, Ly = disorder_log("br_pin")
pins = blocks(f"{O}/br_pin/pins.dat", 1)
f_pin = fraction_near(config("br_pin", FINAL), pins, Lx, Ly)
f_free = fraction_near(config("br_nodis", FINAL), pins, Lx, Ly)
report("vortices attracted to the pins", f_pin > 2 * f_free,
       f"fraction within rp of a pin: {f_pin:.3f} with pinning, {f_free:.3f} without")

# --- dt warning ---
w = open(f"{O}/br_warn/stderr.txt").read()
report("br_warn: dt warning printed", "too large to resolve" in w)
report("br_pin: no dt warning", "too large" not in open(f"{O}/br_pin/stderr.txt").read())

# --- restart vs uninterrupted run (snapshots have 6 decimals) ---
for c, ref in [("pin", "br_pin"), ("nodis", "br_nodis")]:
    report(f"rs_{c}: resume exit 0", exited_ok(f"rs_{c}", "stderr2.txt"))
    d = displacement(config(f"rs_{c}", FINAL), config(ref, FINAL))
    report(f"rs_{c}: restart vs uninterrupted run agree", np.median(d) < 1e-3 and d.max() < 0.1,
           f"median {np.median(d):.1e}, max {d.max():.1e}")

print(f"\n{sum(results)}/{len(results)} PASS")
sys.exit(0 if all(results) else 1)
