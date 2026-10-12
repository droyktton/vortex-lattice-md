#!/usr/bin/env python3
"""Overlay the summary.dat of several temperature sweeps (one panel per observable).

    python3 tools/compare_summaries.py --out compare.png \
        clean:slurm/example_output/sweep/summary.dat point:runs/point/summary.dat

summary.dat is what tsweep.py / slurm/collect.sbatch write (T or the swept
parameter, D, defect_fraction, Sq_max, mean_u2, and V_par V_perp for driven
sweeps, which add two panels). D and the roughness are on a log scale
(nonpositive values left out). In an Fdc sweep the V_par panel also shows
V = F as a dotted line.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+", help="label:path/to/summary.dat")
p.add_argument("--out", default="summary_compare.png")
p.add_argument("--title", default="")
a = p.parse_args()

panels = [("D", "diffusion constant D", True), ("defect_fraction", "disclination fraction", False),
          ("Sq_max", "S(q) Bragg peak", False), ("mean_u2", "line roughness ⟨u²⟩", True)]
data = []
for spec in a.files:
    label, path = spec.split(":", 1)
    names = open(path).readline().split()
    d = np.loadtxt(path, skiprows=1, ndmin=2)
    data.append((label, names, {n: d[:, i] for i, n in enumerate(names)}))
# Driven sweeps: also the mean velocity, parallel and transverse to the drive
if any("V_par" in col for _, _, col in data):
    panels = [("V_par", "V parallel to the drive", False), ("V_perp", "V transverse to the drive", False)] + panels
ncols = len(panels) // 2
fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 7.5), squeeze=False)
for label, names, col in data:
    x = col[names[0]]
    for ax, (key, ylabel, logy) in zip(axes.flat, panels):
        if key not in col:
            continue
        y = col[key]
        keep = y > 0 if logy else np.isfinite(y)   # D can come out <= 0 for a solid
        ax.plot(x[keep], y[keep], "o-", ms=4, label=label)
        ax.set_xlabel(names[0]); ax.set_ylabel(ylabel)
        if logy: ax.set_yscale("log")
        if key == "V_par" and names[0] == "Fdc":
            lim = [0, x.max()]
            ax.plot(lim, lim, ":", color="0.5", lw=1)   # V = F: no pinning
axes[0, 0].legend(fontsize=9)
for ax in axes.flat: ax.grid(alpha=0.3)
if a.title: fig.suptitle(a.title)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("wrote", a.out)
