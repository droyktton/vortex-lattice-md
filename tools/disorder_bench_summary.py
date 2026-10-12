#!/usr/bin/env python3
"""Tables from the runs made by tools/disorder_bench.sbatch.

    python3 tools/disorder_bench_summary.py runs/disorder_bench [--dt-ref 0.005]

Per system size and pin range: ms/step without disorder (same dt), and for
every pin density the ms/step, the slowdown per step, the cost per unit
physical time relative to a run without disorder at --dt-ref, and the
number of Verlet-list rebuilds. Also writes results.tsv in the run folder.
"""
import argparse
import glob
import os
import re

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--dt-ref", type=float, default=0.005,
               help="dt of the run without disorder used for the cost per unit time")
args = p.parse_args()

rows = []
for f in glob.glob(f"{args.dir}/*/out.txt"):
    m = re.search(r"n(\d+)_z(\d+)_np([\d.]+)_rp([\d.]+)_c(\d)_dt([\d.e-]+)_r(\d+)", f)
    t = open(f).read()
    ms = re.search(r"Average Time/Step:\s*([\d.]+)", t)
    rb = re.search(r"Rebuilds:\s*(\d+)", t)
    if not (m and ms and rb):
        print(f"skipping {os.path.dirname(f)} (incomplete run)")
        continue
    nx, nz, np_, rp, c, dt, rep = m.groups()
    rows.append(dict(nx=int(nx), nz=int(nz), np=float(np_), rp=float(rp), correlated=int(c),
                     dt=float(dt), rep=int(rep), ms=float(ms.group(1)), rebuilds=int(rb.group(1))))
d = pd.DataFrame(rows).sort_values(["nx", "nz", "rp", "np", "correlated", "rep"])
d.to_csv(f"{args.dir}/results.tsv", sep="\t", index=False)

g = (d.groupby(["nx", "nz", "np", "rp", "correlated", "dt"])
       .agg(ms=("ms", "mean"), sd=("ms", "std"), rebuilds=("rebuilds", "mean")).reset_index().fillna({"sd": 0.0}))
for (nx, nz), s in g.groupby(["nx", "nz"]):
    print(f"\n=== {nx}x{nx}x{nz} ({nx * nx * nz} vortices) ===")
    ref = s[(s.np == 0) & (s.dt == args.dt_ref)]
    ref_rate = ref.ms.iloc[0] / args.dt_ref if len(ref) else None
    print(f"{'rp':>5} {'np':>5} {'col':>3} {'pins/cell':>9} {'dt':>9} {'ms/step':>14} "
          f"{'x/step':>7} {'x/time':>7} {'rebuilds':>8}")
    for rp in sorted(s[s.rp > 0].rp.unique()):
        dt = s[s.rp == rp].dt.iloc[0]
        base = s[(s.np == 0) & (s.dt == dt)].iloc[0]
        print(f"{rp:5g} {'-':>5} {'-':>3} {'-':>9} {dt:9.5f} {base.ms:8.3f}±{base.sd:.3f} "
              f"{'1':>7} {base.ms / dt / ref_rate if ref_rate else float('nan'):7.2f} {base.rebuilds:8.0f}")
        for _, r in s[s.rp == rp].iterrows():
            print(f"{'':5} {r.np:5g} {int(r.correlated):3d} {r.np * (3 * rp) ** 2:9.2f} {'':9} "
                  f"{r.ms:8.3f}±{r.sd:.3f} {r.ms / base.ms:7.2f} "
                  f"{r.ms / dt / ref_rate if ref_rate else float('nan'):7.2f} {r.rebuilds:8.0f}")
