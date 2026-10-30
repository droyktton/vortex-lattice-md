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
- **Quenched disorder (optional, off by default)**: random Gaussian pinning
  wells, independent in every layer (uncorrelated point disorder) or
  identical in every layer (columnar). See [Quenched disorder](#quenched-disorder).
- **External drive (optional, off by default)**: a uniform DC and/or AC
  force on every vortex, for driven (moving) vortex lattices, with
  noise-free lattice and per-layer velocities. See [External drive](#external-drive).

## Force evaluation

Two interchangeable backends, selected at compile time via `USE_CELL_LIST`:

- **Direct O(N²)** (`vortex_sim_on2`) — every pair, every step.
- **O(N) cell list with a Verlet skin** (`vortex_sim`) — cells are sized to
  `cutoff + skin`; the particle→cell sort is only rebuilt once some particle
  has drifted more than `skin/2` since the last build. Since a pair's
  separation can change by at most `skin` between rebuilds, the stale cell
  assignment is guaranteed to stay a superset of the true neighbor list in
  between — typically cutting the rebuild frequency by an order of magnitude.

### Cutoff: plain or shifted-force

By default the pair force `K1(r)` is cut at `--cutoff`, so it jumps from
`K1(cutoff)` to 0 there (0.040 at the default 3.0). With `a0 = 1` the
default cutoff falls exactly on the third neighbor shell of the triangular
lattice (`r = 3 a0`, 6 neighbors). In a solid, those neighbors then keep
crossing the cutoff thermally, and each crossing kicks the force by
`K1(cutoff)`. That noise does not vanish as T → 0: in a 30×30×4 crystal at
T = 0.002 it is more than half the variance of the vortex-vortex force, and
it makes the force distribution flatter than a Gaussian (see
[forces.py](#analyze-interaction-force-distributions)).

There are two ways around it:

- a cutoff between shells (shells at `1, √3, 2, √7, 3, √12, √13, ...` × a0),
  or a longer one where `K1` is small (`K1(4.5)` = 0.006);
- `--shifted-force`: `F(r) = K1(r) − K1(cutoff)` for `r < cutoff`. The force
  then goes continuously to 0 at the cutoff, whatever a0 is. The log gets a
  `Shifted Force` line, and `forces.py` reads it.

Without `--shifted-force` the force is the plain cut, bit-identical to
earlier versions.

## Build

Requires `nvcc` and the [Random123](https://github.com/DEShawResearch/random123)
headers (`Random123/philox.h`, `Random123/boxmuller.hpp`).

```sh
make            # builds both vortex_sim and vortex_sim_on2
make cell       # only the cell-list version
make direct     # only the direct O(N^2) version
make run        # build + run the cell-list version
```

If your system's Random123 install doesn't include `boxmuller.hpp`, the
Makefile falls back to a checkout next to this repo
(`<parent of repo>/random123/include`, e.g. `git clone
https://github.com/DEShawResearch/random123.git ../random123`), or point it at
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
| `--shifted-force`   | force goes continuously to 0 at the cutoff (see below) | off |
| `--skin`            | Verlet skin width                | 0.5       |
| `--seed`            | RNG seed                        | 1234567   |
| `--print-interval`  | steps between snapshot writes   | 100       |
| `--restart`, `--start-step` | resume from a checkpoint (see below) | off |
| `--np`, `--Ap`, `--rp`, `--disorder-seed`, `--correlated-disorder` | quenched disorder (see below) | off |
| `--n-planes`, `--Ap-plane`, `--rp-plane` | planar disorder (see below) | off |
| `--Fdc`, `--Fac`, `--freq`, `--drive-angle`, `--velocity-interval` | external drive (see below) | off |

Writes `simulation.log` (run parameters, mesh, Verlet-skin settings) and
periodic snapshots `config_step_<N>.dat`: position, force, and identity of
every vortex, saved every `--print-interval` steps plus always the last one
(`config_step_<steps-1>.dat`). With quenched disorder it also writes
`pins.dat`, with planar disorder `planes.dat`, and with an external drive
`velocity.dat`.

### Quenched disorder

Optional random pinning: Gaussian wells `U(r) = -Ap exp(-r²/rp²)` that act
on every vortex within `3 rp` of a pin (the force is cut there). It is off
by default. Without the disorder flags the simulation is exactly the one
without them: same trajectories, same `config_step_*.dat`, same
`simulation.log`, and no `pins.dat`.

```sh
# uncorrelated point disorder: independent pins in every layer
./vortex_sim --nx 30 --ny 30 --T 0.01 --dt 0.005 --np 1 --Ap 0.3 --rp 0.2
# columnar disorder: the same pins in every layer
./vortex_sim --nx 30 --ny 30 --T 0.01 --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 --correlated-disorder
# another realization of the same disorder
./vortex_sim --nx 30 --ny 30 --T 0.01 --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 --disorder-seed 2
```

| Flag                    | Meaning                                              | Default |
|-------------------------|------------------------------------------------------|---------|
| `--np`                  | pinning centers per unit area, per layer             | 0       |
| `--Ap`                  | pin strength, `U = -Ap exp(-r²/rp²)` (attractive for `Ap > 0`, repulsive for `Ap < 0`) | 0 |
| `--rp`                  | pin range (force cut at 3 rp)                        | 0.1     |
| `--disorder-seed`       | seed of the disorder realization                     | 1       |
| `--correlated-disorder` | same pins in every layer instead of independent ones | off     |

Disorder is enabled only when both `--np > 0` and `--Ap != 0`. If only one
of them is set, `vortex_sim` warns and runs without disorder.

#### How the pins are generated

The pins are a Poisson point process. Each layer is cut into an implicit
grid of pinning cells at least `3 rp` wide. The number of pins in each cell
(Poisson, mean `np · cell area`) and their positions are drawn from Philox
keyed on `(cell, layer, disorder seed)`; with `--correlated-disorder` the
layer is left out of the key, so every layer gets the same pins. As a
result:

- **No memory cost.** Nothing is stored; the force kernel regenerates the
  pins of the 3×3 pinning cells around each vortex on the fly.
- **Restarts see the same pins**, given the same disorder flags.
- **The disorder is independent of `--seed`** (the thermal noise). Keep
  `--disorder-seed` fixed across a temperature sweep and vary it to average
  over disorder realizations.
- **The number of pins per layer is random**: Poisson with mean
  `np · Lx · Ly`, where `Lx = nx a0` and `Ly = ny a0 √3/2`.

Two hard limits make `vortex_sim` stop with an error:

- the box must be at least 3 pinning cells (`9 rp`) wide in x and y;
- at most 16 pins per pinning cell on average, i.e. `np · (3 rp)² ≤ 16`.

#### Output

- **`pins.dat`**: every pin, as `x y z` columns, one blank-line-separated
  block per layer (same layout as the config files). To overlay the pins of
  layer 0 on a snapshot in gnuplot:
  `plot 'config_step_999.dat' skip 3 i 0 u 1:2, 'pins.dat' skip 1 i 0 u 1:2 pt 7 ps 0.5`
  (the `skip` drops the header lines).
- **`[Quenched Disorder]` section of `simulation.log`**: `np`, `Ap`, `rp`,
  the force cutoff, the disorder seed, correlated or not, the pinning grid,
  the actual number of pins in every layer (with the Poisson mean), and the
  suggested maximum `dt`.
- **stdout**: one line saying which kind of disorder is on.

#### Choosing the parameters

**Density.** The vortex density is `2/(√3 a0²) ≈ 1.15/a0²`, so `np ≈ 1.15`
gives about one pin per vortex. `np` around 1 is a natural starting point;
use `np ≪ 1` for dilute pins and `np ≫ 1` for dense, collectively acting
pins.

**Range.** Keep `rp` between `0.1` and `0.3 a0`: short-ranged compared with
the vortex spacing, which is the point-disorder regime, without paying for
a tiny `dt` (see below). Go below `0.1` only if the physics needs it; each
halving of `rp` costs 4× the steps.

**Strength.** `Ap` sets the depth of the wells and has to be compared with
the elastic energy scale and with `T`. Scan it: no default fits every
regime. For reference, with `T = 0.01`, `np = 1`, `rp = 0.2` and `Ap = 0.3`,
53% of the vortices sit within `rp` of a pin after 2000 steps, against 11%
by chance without disorder. That is already strong pinning.

**Timestep.** This is the one that matters most. A well's force peaks at
`≈ 0.86 Ap/rp`, so resolving it needs
`dt ≲ 0.12 rp²/Ap` (drift of at most `0.1 rp` per step) and
`dt ≲ rp²/(2T)` (thermal step of at most `rp`). `vortex_sim` prints a
warning with the suggested `dt` when it is violated, and logs it in every
run with disorder, but never changes `dt` itself. Check stderr, or the
`Suggested Max dt` line of `simulation.log`, before a long run. For
example, with `Ap = 0.3` and `T = 0.01`:

| `rp` | suggested `dt` ≲ |
|------|------------------|
| 0.4  | 0.062            |
| 0.3  | 0.035            |
| 0.2  | 0.016            |
| 0.1  | 0.004            |
| 0.05 | 0.001            |

Doubling `Ap` halves the allowed `dt`. The default `--dt 0.01` is too large
for the default `--rp 0.1` with `Ap ≳ 0.12`.

**Averaging.** To average over disorder, run the same parameters (the
whole temperature sweep) with several `--disorder-seed` values and average
the results. `--seed` only changes the thermal noise.

#### Sweeps (`tsweep.py` and SLURM)

`tsweep.py` takes the same flags (`--np`, `--Ap`, `--rp`,
`--disorder-seed`, `--correlated-disorder`) and passes them to every
temperature, so the whole sweep sees the same realization. On the cluster,
set them in your `params.sh` (see `slurm/README.md`):

```sh
PIN_NP=1                 # pins per unit area, per layer
PIN_AP=0.3               # pin strength
PIN_RP=0.2               # pin range
DISORDER_SEED=1          # realization
CORRELATED_DISORDER=0    # 1 = columnar
DT=0.005                 # check it against the suggested dt above
```

In both cases the disorder flags are only passed to `vortex_sim` when
`np` or `Ap` is nonzero, so sweeps without disorder run the same command
line as before. Hysteresis sweeps (`--hysteresis`, `slurm/hysteresis_sweep.sbatch`)
restart each temperature from the previous one; the pins are regenerated
identically, so the whole cycle sees one realization.

#### Performance

The pinning force adds a 3×3 sweep of pinning cells per vortex per step,
and its cost grows with the mean number of pins per pinning cell,
`np · (3 rp)²`. Measured with `tools/disorder_bench.sbatch` on an RTX 3080
(cell-list build, `Ap = 0.3`, `T = 0.01`, 3000 steps), relative to the
same run without disorder and with the same `dt`:

| `rp` \ `np` | 0.25 | 1 | 4 |
|---|---|---|---|
| 0.05 | ×1.02 | ×1.03 | ×1.04 |
| 0.1  | ×1.03 | ×1.03 | ×1.07 |
| 0.2  | ×1.04 | ×1.06 | ×1.14 |
| 0.4  | ×1.06 | ×1.12 | ×1.27 |

That is for 128×128×16 (262k vortices, 1.56 ms/step without disorder);
64×64×16 gives the same numbers within a few percent (up to ×1.31 for
`np = 4, rp = 0.4`). Columnar and uncorrelated disorder cost the same, and
the Verlet-list rebuild rate barely changes.

What does cost is the smaller `dt`. Per unit of physical time, relative to
a run without disorder at `dt = 0.005` (`np = 1`, `Ap = 0.3`):

| `rp` | `dt` used | cost per unit time |
|------|-----------|--------------------|
| 0.4  | 0.005     | ×1.12              |
| 0.2  | 0.005     | ×1.06              |
| 0.1  | 0.0039    | ×1.33              |
| 0.05 | 0.00097   | ×5.1               |

#### Reproducibility

- **Without disorder** the code is bit-identical to the version before
  disorder was added, in both the cell-list and direct builds.
- **With disorder**, a run is bit-reproducible given the same flags and
  build. The cell-list and direct builds see exactly the same pins and give
  the same forces up to float rounding, but their trajectories drift apart
  after a few thousand steps: a vortex sitting near the saddle between two
  pins falls one way or the other depending on the last bit. Most vortices
  agree to ~1e-5; a few percent can end up `~0.05 a0` apart. The two builds
  agree statistically, not trajectory by trajectory.
- **Restarts** read positions from text snapshots with 6 decimals, so a
  restarted run is not bit-identical to an uninterrupted one. Pinning
  amplifies that rounding in the same way: typically ~1e-5, up to ~1e-2 for
  the odd vortex.

#### Example: melting with disorder

`slurm/example_output/disorder/` has temperature sweeps of the 30x30x4
lattice with point disorder of increasing strength and with columnar
disorder, compared with the clean lattice. Weak pins shift the melting
transition down and broaden it; strong ones leave an amorphous pinned state
at every T. See "Example output" in `slurm/README.md`. To overlay sweeps of
your own, use `tools/compare_summaries.py`.

#### Checking and benchmarking (`tools/`)

Both scripts run on one GPU of the cluster; submit them from the repo root
after `slurm/build.sh`. Output goes to `runs/` (ignored by git) and the log
to `slurm/logs/`.

```sh
sbatch tools/disorder_smoke.sbatch     # ~1 min: correctness checks, PASS/FAIL table
sbatch tools/disorder_bench.sbatch     # ~6 min: the performance tables above
```

- **`tools/disorder_smoke.sbatch`** builds a reference `vortex_sim` from
  the last commit without disorder (`REF_REV`, default `acf9aba`) and runs
  a set of short simulations. `tools/disorder_check.py` then checks them:
  - without disorder, the configs and `simulation.log` are bit-identical to
    the reference, in both builds;
  - with disorder, runs are reproducible and `pins.dat` matches the log;
  - the pin counts are Poisson and the pins lie inside the box;
  - columnar layers are identical and another `--disorder-seed` gives
    another realization;
  - vortices are attracted to the pins;
  - the `dt` warning fires when it should;
  - cell-list vs direct and restart vs uninterrupted runs agree
    statistically.

  It exits nonzero if any check fails. Rerun it after touching the force
  kernels or the disorder code.
- **`tools/disorder_bench.sbatch`** measures ms/step over a grid of `np`
  and `rp`, each against a run without disorder at the same `dt`. The
  grid can be changed with environment variables (`SIZES`, `NP_LIST`,
  `RP_LIST`, `AP`, `T`, `DT_MAX`, `STEPS`, `REPS`, `COL_RP`; see the
  script's header), e.g.
  `sbatch --export=ALL,SIZES="256 16",AP=0.5 tools/disorder_bench.sbatch`.
  `tools/disorder_bench_summary.py` prints the tables and writes
  `results.tsv`. The numbers depend on the GPU, so compare runs on the
  same GPU model; the job log shows which GPU it ran on.

#### Planar disorder

Optional planes `x = x_i`, parallel to y and z (like twin boundaries), each
a Gaussian trough `U(x) = -Ap exp(-(x - x_i)²/rp²)` cut at `3 rp`. The
force is along x only and the same in every layer. There are exactly
`--n-planes` planes, with positions uniform and independent in `[0, Lx)`.
They add to the point or columnar pins, so both kinds can be on at once.

```sh
# 5 planes, no point disorder
./vortex_sim --nx 30 --ny 30 --T 0.01 --dt 0.005 --n-planes 5 --Ap-plane 0.3 --rp-plane 0.2
# planes on top of weak point disorder
./vortex_sim --nx 30 --ny 30 --T 0.01 --dt 0.005 --np 1 --Ap 0.03 --rp 0.2 \
             --n-planes 5 --Ap-plane 0.3 --rp-plane 0.2
```

| Flag           | Meaning                                              | Default |
|----------------|------------------------------------------------------|---------|
| `--n-planes`   | number of planes                                     | 0       |
| `--Ap-plane`   | plane strength, `U = -Ap exp(-d²/rp²)`               | 0       |
| `--rp-plane`   | plane range (force cut at 3 rp)                      | 0.1     |

The planes are on only when both `--n-planes > 0` and `--Ap-plane != 0`.
Without them the simulation is the same as before (no `planes.dat`, no
new log section). The positions come from Philox keyed on the plane index
and `--disorder-seed`, with a different key from the pins'. Restarts see
the same planes, and changing `--disorder-seed` changes both pins and
planes. The positions are computed once and kept in a small device array,
and the force loops over all planes, so the cost grows with `--n-planes`.
A few tens of planes are cheap next to the pair forces. The box must be at
least `9 rp` wide in x. The dt warning uses the same criterion as for the
pins.

Output: `planes.dat` (one sorted `x_i` per line, after an `x` header), a
`[Planar Disorder]` section in `simulation.log`, and one line on stdout. To
draw the planes over a snapshot in gnuplot, plot them as vertical lines:
`plot 'config_step_999.dat' skip 3 i 0 u 1:2, 'planes.dat' skip 1 u 1:(0):(0):(100) w vectors nohead`.

With a drive along y (`--drive-angle 90`) the vortices slide along the
planes; along x they have to cross them.

In SLURM sweeps the planes are set with `PLANE_N`, `PLANE_AP` and
`PLANE_RP` (see `slurm/README.md`), e.g. in a `PARAMS_FILE`:

```sh
PLANE_N=5
PLANE_AP=0.3
PLANE_RP=0.2
```

**Checking** (`tools/planar_smoke.sbatch`, ~2 min, PASS/FAIL table):

```sh
sbatch tools/planar_smoke.sbatch
```

It builds a reference `vortex_sim` from the commit before the planes
(`REF_REV`, default `e1fd008`), runs short simulations, and
`tools/planar_check.py` checks that:

- without planes, the output is bit-identical to the reference, in both
  builds and with point pins;
- with planes, runs are reproducible, `planes.dat` matches the log, the
  planes are sorted and inside the box, and another `--disorder-seed`
  gives other planes;
- planes and point pins combine, with the planes unchanged;
- the cell-list and direct builds agree statistically;
- vortices gather near the planes (about 4x more within `rp/2` than
  without planes);
- the dt warning fires for a small `--rp-plane`;
- a restart stays close to an uninterrupted run, without parameter
  mismatch warnings.

### External drive

A uniform force on every vortex, for driven (moving) vortex lattices: DC,
AC, or both, along a fixed direction `e = (cos θ, sin θ)`:

```
F(t) = [Fdc + Fac sin(2π f t)] e,     t = step · dt
```

```sh
# DC drive along x, with point disorder: depinning / moving lattice
./vortex_sim --nx 30 --ny 30 --T 0.002 --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 --Fdc 0.8
# AC drive along y
./vortex_sim --nx 30 --ny 30 --T 0.01 --Fac 0.5 --freq 0.05 --drive-angle 90
# DC + AC along the same direction (e.g. to look for Shapiro steps)
./vortex_sim --nx 30 --ny 30 --T 0.002 --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 \
    --Fdc 0.8 --Fac 0.3 --freq 0.1 --velocity-interval 5
```

| Flag                  | Meaning                                         | Default |
|-----------------------|-------------------------------------------------|---------|
| `--Fdc`               | DC force                                        | 0       |
| `--Fac`               | AC amplitude                                    | 0       |
| `--freq`              | AC frequency (1/time units; required with `--Fac`) | 0    |
| `--drive-angle`       | drive direction θ, degrees from x               | 0       |
| `--velocity-interval` | steps between `velocity.dat` rows               | `--print-interval` |

The drive is on when `--Fdc` or `--Fac` is nonzero. Without them the
simulation is exactly the undriven one: same trajectories, same
`simulation.log`, and no `velocity.dat`. The drive can be combined with
quenched disorder. Restarts pick the AC phase up correctly, since `t` is
counted from step 0.

#### Velocity (`velocity.dat`)

The dynamics is overdamped with unit mobility, so the mean velocity of a
vortex (averaged over the noise) is the deterministic force on it. The
velocity of layer `z` is therefore `<f>_z + F(t)`, where `<f>_z` is the mean
interaction + pinning force on that layer. That is exact and noise-free,
with no need to difference positions. In-plane pair forces cancel within a
layer, so `<f>_z` is the mean pinning force plus the net spring pull from
the neighboring layers; summed over all layers the springs cancel too, so
the lattice velocity is `F(t) + <f_pin>`. Without pinning a lattice slides
at exactly `v = F`.

`velocity.dat` has one row every `--velocity-interval` steps (and the last
step), with columns
`step t Fx Fy vx vy vx_z0 vy_z0 vx_z1 vy_z1 ...`: the drive, the lattice
velocity, and the velocity of every layer. A spread in the per-layer
velocities means the layers slide past each other (decoupling). A restart
in the same folder appends to the file. The `fx fy` columns of the
`config_step_*.dat` snapshots still hold the interaction + pinning force
only, without the drive.

```sh
python3 velocity.py                     # in a run folder
python3 velocity.py --step-min 3000     # average only past the transient
```

`velocity.py` plots v(t) (lattice and layers, parallel and transverse to
the drive) to `velocity.png` and writes `velocity_mean.dat`, which holds:

- `V_par` and `V_perp`: the time-averaged velocity parallel and transverse
  to the drive;
- the mean drive force;
- `layer_spread`: max − min of the per-layer `V_par`.

The default `--step-min` is half of the run.

#### Diffusion of a moving lattice

For a driven run `msd.py` subtracts the center-of-mass displacement before
computing the MSD, so the drift doesn't swamp it (`--keep-drift` turns that
off). It also splits the MSD into the components parallel and transverse to
the drive:

- `msd.dat` gets `MSD_par MSD_perp` columns;
- `msd_diffusion.dat` gets `D_par D_perp` columns, from `MSD_i ≈ 2 D_i t`.

`D` is still from the total MSD `≈ 4 D t`. Undriven runs write exactly the
same files as before. A clean lattice is Galilean invariant, so its
drift-subtracted `D` is the same with and without the drive (checked in
`tools/drive_smoke.sbatch`).

`msd.py` unwraps trajectories assuming every vortex moves less than half the
box between snapshots. `vortex_sim` warns when the drift per snapshot,
`(|Fdc| + |Fac|) dt · print-interval`, exceeds a quarter of the box.

#### Choosing the parameters

- **Timestep.** The drive moves each vortex `(|Fdc| + |Fac|) dt` per step.
  Keep that under 0.1 of the smallest length in the problem: `a0`, or
  `rp` with disorder. `vortex_sim` warns otherwise. With pinning, this adds
  to the `dt` limit of the wells (see "Quenched disorder").
- **AC drive.** Resolve the period with at least ~20 timesteps, and sample
  it with `--velocity-interval` at least ~8 times per period. `vortex_sim`
  warns otherwise. Average velocities over a whole number of periods; long
  windows make that error negligible.
- **Depinning.** A single isolated pin holds a vortex up to
  `Fmax ≈ 0.86 Ap/rp` (1.29 for `Ap = 0.3, rp = 0.2`). The lattice depins
  collectively well below that for weak pinning. Scan `Fdc` (below) to find
  it.

#### Force sweeps (V–F curves)

`tsweep.py --sweep-param Fdc` sweeps the DC force at a fixed temperature
`--T` instead of sweeping T. `Fac` and `freq` can be swept the same way. Run
folders are named after the parameter (`runs/Fdc_0.5`), and
`summary.dat`'s first column is the swept parameter. When the runs are
driven, `summary.dat` gets two more columns, `V_par V_perp`, and there is
one more plot, `summary_V_vs_Fdc.png`: the V–F curve. Each run also goes
through the usual analysis (defects, S(q), roughness, drift-subtracted D),
so the dynamical phases of the moving lattice (pinned, plastic flow,
moving smectic, moving glass) can be told apart from the same table.

```sh
# V-F curve at T = 0.002 with point disorder
python3 tsweep.py 0.1 0.2 0.4 0.6 0.8 1.0 1.5 --sweep-param Fdc --T 0.002 \
    --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 --steps 6000
# ramp up and back down, each force starting from the previous one's
# configuration: hysteresis in F
python3 tsweep.py 0.1 0.4 0.8 1.2 0.8 0.4 0.1 --sweep-param Fdc --T 0.002 --hysteresis \
    --dt 0.005 --np 1 --Ap 0.3 --rp 0.2 --steps 6000
```

The other drive flags (`--Fdc`, `--Fac`, `--freq`, `--drive-angle`,
`--velocity-interval`) fix what isn't swept. On the cluster, set
`SWEEP_PARAM=Fdc` and `SWEEP_T=...` (plus `FDC`, `FAC`, `FREQ`,
`DRIVE_ANGLE`) in your `params.sh`. The "temperatures" file's first
column then holds the forces; see `slurm/README.md`.

#### Performance

The drive is cheap. The uniform drift makes the Verlet list rebuild more
often, about every `skin / (2 F dt)` steps, and sampling the velocity costs
one reduction over the vortices per `velocity.dat` row. Measured on an RTX
3080 (128×128×16 = 262k vortices, `T = 0.01`, `dt = 0.005`, 3000 steps,
no disorder; run-to-run noise about ±3%):

| `Fdc` | ms/step, `--velocity-interval 15` | ms/step, `--velocity-interval 1` | rebuilds / 3000 steps |
|-------|------|------|-----|
| 0 (no drive) | 1.60 | — | 99 |
| 0.1   | 1.57 | 1.68 | 102 |
| 0.3   | 1.68 | 1.70 | 123 |
| 1     | 1.66 | 1.73 | 191 |
| 3     | 1.72 | 1.82 | 347 |

So the drive costs at most ~10% at the default `--velocity-interval`, and
writing a row every step adds ~5% on top.

#### Example: dynamic reordering

`slurm/example_output/drive/` has a V–F curve of the 30x30x4 lattice with
point disorder (`Ap = 0.1`, `np = 1`, `rp = 0.2`, T = 0.002). The lattice
goes from pinned and amorphous through plastic flow to a moving crystal
that recrystallizes above `Fdc` ≈ 0.6: defects 42% → 0, Bragg peak
3.6 → 25, `V` → `F`. See "Example output" in `slurm/README.md`.

#### Checking (`tools/drive_smoke.sbatch`)

```sh
sbatch tools/drive_smoke.sbatch     # ~3 min, PASS/FAIL table
```

Runs short simulations and `tools/drive_check.py` on them. It checks that:

- a clean lattice at T = 0 moves rigidly at `v = F`, for a DC drive at
  30° in both builds and for an AC drive;
- at T > 0 the mean velocity is `F` and the drift-subtracted `D` matches
  the undriven one;
- strong pinning holds the lattice under a weak drive and lets it slide
  under a strong one;
- a restart continues `velocity.dat`;
- the warnings fire when they should;
- `tsweep.py --sweep-param Fdc --hysteresis` runs end to end.

Without a drive, `tools/disorder_smoke.sbatch` with `REF_REV=<commit before
the drive>` checks bit-identity with the previous code.

### Restart / checkpoint

```sh
./vortex_sim --nx 30 --ny 30 --T 0.02 --steps 3000 --seed 42            # hits a wall-clock limit at step 1800
./vortex_sim --nx 30 --ny 30 --T 0.02 --steps 3000 --seed 42 \
    --restart config_step_1800.dat                                      # resumes at 1801, runs to 3000
```

`--restart FILE` resumes from a previously written `config_step_<N>.dat` (or
`configIni.dat`, to restart from the pristine initial lattice) instead of
generating a fresh triangular lattice. `--steps` keeps meaning the final
target step count, not "how many more steps" — the run continues from
wherever `--restart` left off up to `--steps`.

The step to resume at defaults to `N+1` (parsed from the filename), or `0`
for `configIni.dat`; override with `--start-step` if you've renamed the
file. Two things make this possible without a dedicated checkpoint format:
the position/layer/identity columns already in `config_step_<N>.dat` are
everything physically needed, and the RNG (Philox4x32, keyed on
`(vortex id, step, seed)`) is stateless — resuming just means continuing to
count `step` from the right place, not restoring any generator state.

**You must pass the same `--nx`/`--ny`/`--nz`/`--a0`/`--k`/`--T`/`--dt`/
`--cutoff`/`--skin`/`--seed` (and the disorder and drive flags, if any) as the original run** — none of these are
recoverable from the checkpoint file's physical state alone, and getting one
wrong (especially `--seed`) silently continues under different physics or
replays already-used noise rather than failing loudly. As a safety net, if a
`simulation.log` from the original run sits next to the restart file, its
parameters are cross-checked against the current invocation and any mismatch
prints a warning (not a hard error, since deliberately changing `--T` mid-run
for an annealing schedule is a legitimate use of `--restart` too). A restart
file with the wrong number of rows (from mismatched `--nx`/`--ny`/`--nz`) is
a hard error, since positions can't be meaningfully loaded at all.

`configIni.dat` and the initial cell-list build are skipped on restart (they
represent/apply to the pristine lattice, not the resume point); everything
else — output file naming, `--print-interval`, the Verlet-skin cell list —
behaves exactly as in a fresh run.

## Which analysis do I run?

Every measurement is a separate Python script over the `.dat` files
`vortex_sim` already writes — nothing to enable in the simulation itself,
except that a couple of them need a *trajectory* (several evenly spaced
snapshots) rather than just one:

| I want to see...                            | Script         | Needs                          |
|-----------------------------------------------|----------------|---------------------------------|
| The lattice / flux lines                      | `vizconfig.py` | one snapshot                    |
| Positional order: g(r), S(k), S(q)            | `analyze.py`   | one snapshot (or many, see below) |
| Topological defects: disclinations            | `disclinations.py` | one snapshot (or many)     |
| Flux-line roughness (wandering vs own mean)   | `roughness.py` | one snapshot (or many)          |
| Diffusion: MSD vs t, and D                    | `msd.py`       | a trajectory (small `--print-interval`) |
| Vortex-vortex force distributions P(fx), P(fy) | `forces.py`   | one snapshot (or many)          |
| Compare a quantity across runs (e.g. vs T)    | `compare.py`   | one `.dat` per run              |
| All of the above across a temperature sweep   | `tsweep.py`    | just `vortex_sim`; it drives the rest |

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

# aggregate defect statistics over an equilibrated trajectory too (same
# --step-min/--step-max/--stride convention as analyze.py):
python3 disclinations.py "config_step_*.dat" --step-min 1500 --step-max 3000 --stride 10
```

Requires `scipy` (`scipy.spatial.Delaunay`). For each layer, triangulates
the vortex positions under periodic boundary conditions (the standard
ghost-image trick: tile into the 8 neighboring periodic copies, triangulate
the padded point set, fold the edges back) and marks every vortex whose
coordination number isn't 6 — a disclination, with topological charge
6 − coordination. Produces:

- `<name>_disclinations.png` — one panel per layer: the bond network in
  gray, 5-fold sites (charge +1) in blue, 7-fold (charge −1) in red, and
  anything more exotic in orange. **Single-snapshot mode only** — a bond
  network doesn't have a meaningful "average" across time, so this plot is
  skipped when the first argument is a glob pattern matching more than one
  file (a console note says so).
- `<name>_disclinations.dat` — per layer: N, defect count, defect fraction,
  and the 5-fold/7-fold/other breakdown, pooled across every matched
  configuration in multi-config mode. The total defect count/fraction
  across all layers (and configs) is also printed to the console.
- `<name>_coord_hist.png` — coordination-number histogram, grouped bars per
  layer, pooled the same way. A perfect crystal is a single spike at 6; a
  liquid spreads out around it, roughly symmetric between 5- and 7-fold.

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

## Analyze (vortex line roughness)

```sh
python3 roughness.py config_step_499.dat
python3 roughness.py "config_step_*.dat" --step-min 1500 --step-max 3000 --stride 10
```

How much each flux line wanders transversely around its own mean position —
a roughness/Lindemann-type measure of a single vortex line, as opposed to
disclinations (positional order of the lattice) or MSD (diffusion in time).
For each vortex line (persistent in-plane `label`, z-unwrapped the same way
as the 3D flux-line plot so a line anchored near a box edge doesn't bias its
own mean), computes

&nbsp;&nbsp;⟨u²⟩ = (1/Nz) Σ_z |r(z) − r_cm|² ,  r_cm = (1/Nz) Σ_z r(z)

then averages ⟨u²⟩ over every line (and, in multi-config mode, over every
matched configuration too — same `--step-min`/`--step-max`/`--stride`
convention as `analyze.py`/`disclinations.py`). Produces:

- `<name>_roughness.dat` — one row: the ensemble-averaged ⟨u²⟩ and the
  number of (line, config) samples it was averaged over.
- `<name>_roughness_hist.png` — histogram of ⟨u²⟩ across individual lines,
  with the ensemble mean marked. A crystal's lines vibrate a little around
  straight; a liquid's wander much further — the two reference runs used to
  validate this script give ⟨u²⟩ ≈ 0.0023 (T=0.005, solid) vs ≈ 0.016-0.018
  (T=0.03, liquid), roughly an order of magnitude apart, and it's now part
  of `tsweep.py`'s summary alongside D, defect fraction, and S(q) peak.

## Analyze (mean squared displacement)

```sh
./vortex_sim --steps 500 --print-interval 10
python3 msd.py                                                 # every snapshot as a time origin, full window
python3 msd.py --window 20 --origin-spacing 5                   # fixed 20-snapshot window, origins every 5 snapshots
python3 msd.py --window 20 --origin-spacing 5 --t0-min 10       # ...and skip the first 10 snapshots (equilibration)
python3 msd.py --fit-tmin 5                                     # fit D to the t >= 5 portion explicitly
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
is dropped automatically. Also fits a diffusion constant D from the late-time
(already-diffusive) slope: MSD ≈ 4Dt in 2D, so D = slope/4; `--fit-tmin`
picks where the fit starts (default: half of the largest available Δt) —
check `msd_windows.png` first, the fit is meaningless if `t_min` still falls
in the short-time non-diffusive rise. Pass `--fit-tmin -1` to skip it.
Produces:

- `msd.dat` — two columns: t, MSD (averaged over windows).
- `msd.png` — that average, vs t, with the D fit overlaid as a dashed line.
- `msd_windows.png` — every window's own MSD(Δt) curve, colored by its start
  time t₀, plus the average in black. Use this to check convergence to
  steady state: if the earliest (darkest) curves sit apart from the rest,
  the trajectory hadn't equilibrated yet at those t₀ — raise `--t0-min` to
  exclude them.
- `msd_diffusion.dat` — one row: D, the fit's `t_min`, slope, intercept.
  D≈0 (down to the fit's own noise floor) means a solid; a clearly nonzero,
  stable D means a liquid.

For a driven run (its `simulation.log` has a `[Driving Force]` section),
`msd.py` first subtracts the center-of-mass displacement (`--keep-drift`
to skip that), and adds the components parallel and transverse to the
drive: `msd.dat` gets `MSD_par MSD_perp` columns, `msd_diffusion.dat` gets
`D_par D_perp` (from `MSD_i ≈ 2 D_i t`), and `msd.png` shows all three. See
[External drive](#external-drive).

## Analyze (interaction force distributions)

`forces.py` gives the distributions of the in-plane vortex-vortex force on a
vortex, `P(fx)`, `P(fy)` and `P(|f|)`, accumulated over all vortices, layers
and the matched snapshots:

```sh
python3 forces.py config_step_999.dat
python3 forces.py "config_step_*.dat" --step-min 1500            # past the transient
python3 forces.py "config_step_*.dat" --step-min 1500 --bins 201 --fmax 2
```

Only the repulsion between vortices of the same layer is counted:
`f_i = Σ_j K1(r_ij)/r_ij (r_i − r_j)` over the `j` within the interaction
cutoff, with the same core cutoff and minimum image as `main.cu`. The
springs between layers, pinning, planes and drive are left out. The
`fx fy` columns of the snapshots can't be used for this, for two reasons:
they hold the total deterministic force, and they were evaluated at the
positions *before* the last Langevin step. So the script recomputes the pair
forces from the positions. The box size and cutoff come from
`simulation.log` (`--cutoff` overrides the latter).

| Flag                        | Meaning                                          | Default |
|-----------------------------|--------------------------------------------------|---------|
| `--step-min`, `--step-max`, `--stride` | which `config_step_*.dat` to use, as in `analyze.py` | all |
| `--bins`                    | histogram bins                                   | 101     |
| `--fmax`                    | histogram range `[-fmax, fmax]` (`[0, √2 fmax]` for `|f|`) | largest component |
| `--cutoff`                  | interaction cutoff                               | from the log |
| `--shifted-force`           | subtract `K1(cutoff)`, as `vortex_sim --shifted-force` | from the log |
| `--out-prefix`, `--show`    | as in `analyze.py`                               |         |

Output, with the prefix as in `analyze.py` (`config_step_999` or
`avg_step<A>-<B>`):

- `<prefix>_forces.dat`: columns `f P(fx) P(fy)`; the header has the mean,
  standard deviation and excess kurtosis of each component.
- `<prefix>_forces_abs.dat`: columns `|f| P(|f|)`.
- `<prefix>_forces.png`: `P(fx)` and `P(fy)` on a log scale, each with the
  Gaussian of the same variance.
- stdout: the same moments, and the largest net pair force in a layer.
  That force is zero by Newton's third law, so a nonzero value flags a
  problem.

Notes:

- With the plain cutoff, a neighbor crossing `r = cutoff` changes the
  force by `K1(cutoff)` (0.040 for the default 3.0). See
  [Cutoff: plain or shifted-force](#cutoff-plain-or-shifted-force): for force
  statistics in a solid, use `--shifted-force` or a cutoff between neighbor
  shells. In a *perfect*
  lattice with `a0 = 1` there are neighbors at exactly `r = 3`, and the
  6-decimal rounding of the files puts some inside the cut and their
  mirror images outside. A perfect `configIni.dat` therefore shows
  `std(fy) ≈ 0.04` instead of 0. Thermal snapshots are not affected.
- Checked against the simulation (`--k 0`, no disorder, `--print-interval
  1`): the `fx fy` of snapshot `s` match the forces recomputed at the
  positions of snapshot `s − 1`, with a median difference of 2e-6 (the
  file rounding). In 2 of 120 000 forces, the rounding moved a pair across
  the cut.

## Compare across runs

`msd.dat` and `<name>_sq.dat` are both plain two-column (x, y) text files, so
overlaying them from different runs — e.g. a temperature sweep — is just:

```sh
python3 compare.py --out msd_vs_T.png --xlabel "t" --ylabel "MSD" --title "MSD vs T" \
    T=0.01:runs/T_0.01/msd.dat T=0.02:runs/T_0.02/msd.dat T=0.03:runs/T_0.03/msd.dat

python3 compare.py --out sq_vs_T.png --xlabel q --ylabel "S(q)" --logy --hline 1.0 \
    T=0.01:runs/T_0.01/equil_sq.dat T=0.02:runs/T_0.02/equil_sq.dat
```

Each positional argument is `label:path`. Useful for exactly the kind of
question this project keeps coming back to: at what temperature does the
lattice melt? (MSD turns from saturating to linear, and S(q)'s higher-order
peaks disappear, at the same T.) `tsweep.py` below runs the sweep and does
this comparison step for you.

## Temperature sweeps

```sh
python3 tsweep.py 0.005 0.01 0.015 0.02 0.03
```

Runs the whole pipeline — `vortex_sim`, then `analyze.py`, `msd.py`,
`disclinations.py`, `roughness.py` on the equilibrated part of each run —
once per temperature, and collects the melting-relevant scalars (D,
disclination fraction, S(q) peak height, line roughness) into one table
plotted vs T. Concretely, for each T this:

1. Runs `./vortex_sim --T <T> ...` in its own `runs/T_<T>/` folder.
2. Runs `analyze.py "config_step_*.dat" --step-min ... --stride ... --out-prefix equil`
   (g(r)/S(k)/S(q), time-averaged over the equilibrated part) in that folder.
3. Runs `msd.py --window ... --origin-spacing ... --t0-min ...` (MSD + the D fit).
4. Runs `disclinations.py "config_step_*.dat" --step-min ... --stride ... --out-prefix equil`
   (pooled defect statistics over the same equilibrated part).
5. Runs `roughness.py "config_step_*.dat" --step-min ... --stride ... --out-prefix equil`
   (pooled line roughness over the same equilibrated part).
6. Reads `D` from `msd_diffusion.dat`, the overall defect fraction from
   `equil_disclinations.dat`, `max(S(q))` from `equil_sq.dat`, and the
   ensemble ⟨u²⟩ from `equil_roughness.dat`.

Produces, in `--out-dir` (default `runs/`):

- `summary.dat` — one row per T: `T D defect_fraction Sq_max mean_u2`.
- `summary_D_vs_T.png`, `summary_defects_vs_T.png`, `summary_Sqmax_vs_T.png`,
  `summary_roughness_vs_T.png`.

**Equilibration time is very T-dependent** — it grows sharply near the
melting transition (critical slowing down) and can be much shorter far from
it — so a single `--steps`/`--step-min` for every temperature is rarely
right for a serious sweep. Both accept either one value (applied to every T)
or one value per T, positionally matched to the `temperatures` list:

```sh
python3 tsweep.py 0.005 0.01 0.012 0.015 0.02 \
    --steps      2000 3000 6000 6000 3000 \
    --step-min   500  1000 3000 3000 1000
```

Give temperatures close to a suspected transition more `--steps` (to reach
equilibrium at all) and a later `--step-min` (to actually discard the longer
transient), and cheaper/shorter runs to temperatures you're confident are
far from it. `--step-min` defaults to half of that T's own `--steps` if not
given. `--stride` (also broadcastable per-T) controls the same
analyze.py/disclinations.py subsampling cost tradeoff described above.

Other flags: `--nx`/`--ny`/`--nz`/`--print-interval` (shared by every T),
`--np`/`--Ap`/`--rp`/`--disorder-seed`/`--correlated-disorder` (quenched
disorder, same realization at every T; only passed to `vortex_sim` when
`--np` or `--Ap` is nonzero),
`--Fdc`/`--Fac`/`--freq`/`--drive-angle`/`--velocity-interval` (external
drive; only passed when `--Fdc` or `--Fac` is nonzero; driven runs also go
through `velocity.py`, and `summary.dat` gets `V_par V_perp` columns),
`--sweep-param Fdc|Fac|freq` with `--T` (sweep the drive at fixed T instead;
see [Force sweeps](#force-sweeps-vf-curves)),
`--msd-window`/`--msd-origin-spacing`, and `--skip-sim` to rerun just the
analysis steps against `config_step_*.dat` files a previous sweep already
produced (e.g. while retuning `--stride` or fit parameters, without repaying
the simulation cost).

**Cost**: `analyze.py`'s S(k) is O(grid_size × N) per (config, layer) — the
dominant cost of the whole sweep. A run of the size used to validate all of
this (30×30×4, `--stride 10`, ~10 equilibrated configs per T) took a few
minutes per T; budget accordingly for more temperatures, finer `--stride`,
or larger `--kmax`.

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
  (also optionally time-averaged)
- `roughness.py` — flux-line roughness ⟨u²⟩ (also optionally time-averaged)
- `msd.py` — mean squared displacement vs time, and the diffusion constant D
- `forces.py` — distributions of the vortex-vortex force on a vortex, P(fx), P(fy), P(|f|)
- `compare.py` — overlay a quantity (MSD, S(q), ...) across several runs
- `velocity.py` — velocity of a driven lattice vs time, and its average
- `tsweep.py` — run the full pipeline across a temperature sweep (or a
  sweep of the drive force or frequency)
- `tools/` — correctness checks and benchmark of the quenched disorder
  (see [Quenched disorder](#quenched-disorder)) and of the external drive
  (see [External drive](#external-drive)), and
  `compare_summaries.py` to overlay the `summary.dat` of several sweeps
- `slurm/` — SLURM scripts to run `tsweep.py` sweeps (one-way and hysteresis)
  on the GPU cluster; see `slurm/README.md`
- `verlattice.gnu` — gnuplot alternative
