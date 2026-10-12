#!/usr/bin/env python3
"""Overlay the summary.dat of several temperature sweeps (one panel per observable).

    python3 tools/compare_summaries.py --out compare.png \
        clean:slurm/example_output/sweep/summary.dat point:runs/point/summary.dat

summary.dat is what tsweep.py / slurm/collect.sbatch write (T, D,
defect_fraction, Sq_max, mean_u2). D and the roughness are on a log scale.
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
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for spec in a.files:
    label, path = spec.split(":", 1)
    names = open(path).readline().split()
    d = np.loadtxt(path, skiprows=1)
    col = {n: d[:, i] for i, n in enumerate(names)}
    for ax, (key, ylabel, logy) in zip(axes.flat, panels):
        ax.plot(col["T"], col[key], "o-", ms=4, label=label)
        ax.set_xlabel("T"); ax.set_ylabel(ylabel)
        if logy: ax.set_yscale("log")
axes[0, 0].legend(fontsize=9)
for ax in axes.flat: ax.grid(alpha=0.3)
if a.title: fig.suptitle(a.title)
fig.tight_layout()
fig.savefig(a.out, dpi=130)
print("wrote", a.out)
