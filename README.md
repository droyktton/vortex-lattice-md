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
- `verlattice.gnu` — gnuplot alternative
