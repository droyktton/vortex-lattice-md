# vortex-lattice-md

CUDA + Thrust molecular dynamics of a periodic 3D vortex lattice: stacked
triangular layers of vortices, interacting in-plane through a screened
(Bessel K1) pair potential and coupled between layers by harmonic springs,
integrated with overdamped Langevin dynamics.

## Physical model

- **In-plane interaction**: modified Bessel K1 pair force, soft-cored at
  `R_CORE`, periodic in x and y (minimum image convention).
- **Inter-layer coupling**: harmonic springs connecting each vortex line to
  its counterpart in the neighboring layers, periodic in z.
- **Integrator**: overdamped Langevin (Brownian) dynamics. Each particle
  draws independent noise from a counter-based Philox4x32 RNG keyed on
  `(global vortex id, step, seed)`, so trajectories are reproducible
  regardless of how particles get reordered in memory.

## Force evaluation

Two interchangeable backends, selected at compile time via `USE_CELL_LIST`:

- **Direct O(N²)** (`vortex_sim_on2`) — every pair, every step.
- **O(N) cell list with a Verlet skin** (`vortex_sim`) — cells are sized to
  `cutoff + skin`; the particle→cell sort is only rebuilt once some particle
  has drifted more than `skin/2` since the last build. Since a pair's
  separation can change by at most `skin` between rebuilds, the stale cell
  assignment is guaranteed to stay a superset of the true neighbor list in
  between — typically cutting the rebuild frequency by an order of magnitude.

## Build

Requires `nvcc` and the [Random123](https://github.com/DEShawResearch/random123)
headers (`Random123/philox.h`, `Random123/boxmuller.hpp`).

```sh
make            # builds both vortex_sim and vortex_sim_on2
make cell       # only the cell-list version
make direct     # only the direct O(N^2) version
make run        # build + run the cell-list version
```

If your system's Random123 install doesn't include `boxmuller.hpp`, point at
one that does:

```sh
make RANDOM123_DIR=/path/to/random123/include
```

## Run

```sh
./vortex_sim                                   # defaults
./vortex_sim --nx 40 --ny 40 --T 0.05 --steps 2000
./vortex_sim --help                            # list all options
```

Every physical and run parameter has a default and can be overridden on the
command line:

| Flag               | Meaning                        | Default   |
|---------------------|---------------------------------|-----------|
| `--nx`, `--ny`      | in-plane lattice size           | 20, 20    |
| `--nz`              | number of layers                | 4         |
| `--a0`              | lattice constant                | 1.0       |
| `--k`               | inter-layer spring constant     | 0.5       |
| `--T`               | temperature                     | 0.01      |
| `--dt`              | timestep                        | 0.01      |
| `--steps`           | number of integration steps     | 500       |
| `--cutoff`          | interaction cutoff radius       | 3.0       |
| `--skin`            | Verlet skin width                | 0.5       |
| `--seed`            | RNG seed                        | 1234567   |
| `--print-interval`  | steps between snapshot writes   | 100       |

Writes `simulation.log` (run parameters, mesh, Verlet-skin settings) and
periodic snapshots `config_step_<N>.dat`: position, force, and identity of
every vortex, saved every `--print-interval` steps plus always the last one
(`config_step_<steps-1>.dat`).

## Which analysis do I run?

Every measurement is a separate Python script over the `.dat` files
`vortex_sim` already writes — nothing to enable in the simulation itself,
except that a couple of them need a *trajectory* (several evenly spaced
snapshots) rather than just one:

| I want to see...                            | Script         | Needs                          |
|-----------------------------------------------|----------------|---------------------------------|
| The lattice / flux lines                      | `vizconfig.py` | one snapshot                    |
| Positional order: g(r), S(k), S(q)            | `analyze.py`   | one snapshot (or many, see below) |
| Topological defects: disclinations            | `disclinations.py` | one snapshot                |
| Diffusion: MSD vs t                           | `msd.py`       | a trajectory (small `--print-interval`) |
| Compare a quantity across runs (e.g. vs T)    | `compare.py`   | one `.dat` per run              |

For a trajectory, run with a small `--print-interval` so there are enough
`config_step_*.dat` files to work with, e.g. `./vortex_sim --steps 500
--print-interval 10`. Details and full flag lists for each script are below.

## Visualize

```sh
python3 vizconfig.py config_step_499.dat          # saves PNGs
python3 vizconfig.py config_step_499.dat --show    # also opens interactive windows
```

Requires `pandas` and `matplotlib`. Produces:

- `<name>_layers.png` — every layer's in-plane positions overlaid.
- `<name>_3d.png` — 3D flux lines: each vortex's positions across the
  z-stack are linked into one continuous curve by its persistent identity
  (`global_id % Nxy`, constant across layers), unwrapped across the periodic
  x/y boundary, and colored by how much it wanders transversely.

`verlattice.gnu` is a lighter gnuplot alternative for a quick look (no
periodic unwrapping, so lines anchored near a box edge can show a spurious
diagonal jump).

## Analyze (g(r), S(k), S(q))

```sh
python3 analyze.py config_step_499.dat
python3 analyze.py config_step_499.dat --dr 0.02 --kmax 15 --show

# average over time too, once the trajectory has equilibrated:
python3 analyze.py "config_step_*.dat" --step-min 1500 --step-max 3000
python3 analyze.py "config_step_*.dat" --step-min 1500 --step-max 3000 --stride 10
```

Computes, per layer, the 2D radial distribution function g(r) and the structure factor
S(k) = |Σⱼ exp(-i k·rⱼ)|²/N, then averages both over the z-stack (and, if the
first argument is a glob pattern like `"config_step_*.dat"` instead of one
file, over every matching snapshot too — filtered to `--step-min`/
`--step-max` so you can pick out just the equilibrated part of a run, and
thinned by `--stride` since S(k) is the expensive part: it's O(grid_size × N)
per (config, layer), so averaging over many configurations at full
resolution can take minutes). S(k) is
evaluated only at the box's own reciprocal lattice, kx = 2πm/Lx, ky = 2πn/Ly
for integer m, n — the only k-points where summing over one periodic cell
exactly reproduces the coherent scattering of the infinite PBC-tiled system
(every periodic image contributes an identical phase there). Off that
lattice, the same sum is indistinguishable from the Fourier transform of a
single finite, non-periodic rectangular cluster, which produces spurious
sinc-like streaking along the kx=0/ky=0 axes — a finite-window artifact of
the box shape that persists regardless of the simulation's own periodic
boundaries, since it comes from *where* S(k) is sampled, not from the
dynamics. Produces:

- `<name>_gr.png` — g(r); a crystalline lattice shows persistent oscillations
  around 1, not the decay of a liquid.
- `<name>_sk.png` — S(kx, ky) on a log color scale (k=0 masked, since it's a
  trivial peak equal to N); a triangular lattice shows hexagonal rings of
  Bragg-like peaks. Speckle in the background reflects the small number of
  z-layers being averaged (no time averaging is done).
- `<name>_sq.png` / `<name>_sq.dat` — S(q), the azimuthal average of
  S(kx, ky) over rings of fixed q = √(kx²+ky²) (k=0 excluded), the k-space
  analog of g(r). A liquid shows one broad principal peak decaying to
  S(q)→1; a crystal shows sharp, much taller peaks that don't decay.

Because the grid is tied to the box, `--kres` doesn't exist — resolution is
whatever `2π/Lx`, `2π/Ly` give you; `--kmax` (default: ~4 reciprocal shells
based on `a0`, read from `simulation.log`) only controls how many shells to
compute. `--dq` (S(q) bin width) defaults to the finer of `2π/Lx`, `2π/Ly`.

## Analyze (disclinations)

```sh
python3 disclinations.py config_step_499.dat
python3 disclinations.py config_step_499.dat --show
```

Requires `scipy` (`scipy.spatial.Delaunay`). For each layer, triangulates
the vortex positions under periodic boundary conditions (the standard
ghost-image trick: tile into the 8 neighboring periodic copies, triangulate
the padded point set, fold the edges back) and marks every vortex whose
coordination number isn't 6 — a disclination, with topological charge
6 − coordination. Produces:

- `<name>_disclinations.png` — one panel per layer: the bond network in
  gray, 5-fold sites (charge +1) in blue, 7-fold (charge −1) in red, and
  anything more exotic in orange.
- `<name>_disclinations.dat` — per layer: N, defect count, defect fraction,
  and the 5-fold/7-fold/other breakdown.

A near-perfect triangular lattice is close to the worst case for Delaunay:
every hexagonal ring of 6 neighbors sits almost exactly on a circle around
the central site, so the triangulation there is nearly degenerate, and
floating point can pick the "wrong" diagonal and wire up a spurious bond to
a *second*-shell neighbor (distance a0√3) instead of a real first-shell one
(distance a0) — inflating the defect count with numerical artifacts, not
real physics. `--bond-cutoff` (default `1.35*a0`, between the two shells)
drops Delaunay edges longer than that; pass `--bond-cutoff 0` to disable it
and see the raw (noisier) triangulation. Verified against the temperature
sweep from `compare.py`: 0% defects at T=0.005 (a clean hexagonal mesh with
no colored sites at all), climbing smoothly through ~5-8% at T=0.01 to
~44-46% at T=0.03 — consistent with the same T=0.01–0.015 melting range
g(r)/S(q)/MSD already pointed to.

## Analyze (mean squared displacement)

```sh
./vortex_sim --steps 500 --print-interval 10
python3 msd.py                                                 # every snapshot as a time origin, full window
python3 msd.py --window 20 --origin-spacing 5                   # fixed 20-snapshot window, origins every 5 snapshots
python3 msd.py --window 20 --origin-spacing 5 --t0-min 10       # ...and skip the first 10 snapshots (equilibration)
```

Computes MSD(Δt) = ⟨[r(t₀+Δt) − r(t₀)]²⟩, averaged over every vortex and over
equally spaced reference times t₀ (starting no earlier than `--t0-min`
snapshots in, to leave out the initial equilibration transient) within a
fixed window (the standard multiple-time-origins trick for better statistics
from one trajectory).
Positions are unwrapped across time (minimum image between consecutive
snapshots) so a vortex crossing the periodic boundary doesn't register as a
huge jump — this assumes true displacement between consecutive *saved*
snapshots stays under half the box, so don't set `--print-interval` too
coarse relative to how fast the vortices actually move. Any trailing
snapshot that breaks uniform step spacing (e.g. the always-saved final step)
is dropped automatically. Produces:

- `msd.dat` — two columns: t, MSD (averaged over windows).
- `msd.png` — that average, vs t.
- `msd_windows.png` — every window's own MSD(Δt) curve, colored by its start
  time t₀, plus the average in black. Use this to check convergence to
  steady state: if the earliest (darkest) curves sit apart from the rest,
  the trajectory hadn't equilibrated yet at those t₀ — raise `--t0-min` to
  exclude them.

## Compare across runs

`msd.dat` and `<name>_sq.dat` are both plain two-column (x, y) text files, so
overlaying them from different runs — e.g. a temperature sweep — is just:

```sh
python3 compare.py --out msd_vs_T.png --xlabel "t" --ylabel "MSD" --title "MSD vs T" \
    T=0.01:runs/T_0.01/msd.dat T=0.02:runs/T_0.02/msd.dat T=0.03:runs/T_0.03/msd.dat

python3 compare.py --out sq_vs_T.png --xlabel q --ylabel "S(q)" --logy --hline 1.0 \
    T=0.01:runs/T_0.01/config_step_2999_sq.dat T=0.02:runs/T_0.02/config_step_2999_sq.dat
```

Each positional argument is `label:path`. Useful for exactly the kind of
question this project keeps coming back to: at what temperature does the
lattice melt? (MSD turns from saturating to linear, and S(q)'s higher-order
peaks disappear, at the same T.)

## End-to-end example

```sh
make
./vortex_sim --nx 30 --ny 30 --nz 4 --steps 1000 --T 0.02 --print-interval 200
python3 vizconfig.py config_step_999.dat --show
```

3600 flux lines (30×30×4), run at a higher temperature than the defaults —
`config_step_999_3d.png` shows visibly more transverse wander than a
`T=0.01` run.

## Files

- `main.cu` — the simulation
- `Makefile` — build targets
- `vizconfig.py` — visualization
- `analyze.py` — g(r), S(k), and S(q), z-averaged (and optionally time-averaged)
- `disclinations.py` — per-layer Delaunay triangulation and disclinations
- `msd.py` — mean squared displacement vs time
- `compare.py` — overlay a quantity (MSD, S(q), ...) across several runs
- `verlattice.gnu` — gnuplot alternative
