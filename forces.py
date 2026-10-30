import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.special import k1

from vizconfig import load_config, find_box_size, find_log_value, gather_configs, config_label, default_out_prefix

# Same core cutoff as R_CORE in main.cu.
R_CORE = 1.0e-3


def pair_forces_layer(x, y, Lx, Ly, cutoff, shifted=False):
    """In-plane vortex-vortex force on every vortex of one layer, as in the
    force kernels of main.cu: f_i = sum_j K1(r_ij)/r_ij (r_i - r_j) over the
    j within `cutoff` (minimum image), with r clamped to R_CORE; with
    `shifted`, K1(cutoff) is subtracted (--shifted-force). Springs, pinning
    and the drive are not included."""
    pos = np.column_stack([np.mod(x, Lx), np.mod(y, Ly)])
    # boxsize can round a coordinate up to exactly L; cKDTree wants [0, L).
    pos[pos[:, 0] >= Lx, 0] = 0.0
    pos[pos[:, 1] >= Ly, 1] = 0.0
    tree = cKDTree(pos, boxsize=[Lx, Ly])
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    i, j = pairs[:, 0], pairs[:, 1]

    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)
    r = np.maximum(np.hypot(dx, dy), R_CORE)
    f = (k1(r) - (k1(cutoff) if shifted else 0.0)) / r

    fx = np.zeros(len(x))
    fy = np.zeros(len(x))
    np.add.at(fx, i, f * dx)
    np.add.at(fy, i, f * dy)
    np.add.at(fx, j, -f * dx)
    np.add.at(fy, j, -f * dy)
    return fx, fy


def moments(f):
    m = f.mean()
    s = f.std()
    kurt = ((f - m) ** 4).mean() / s ** 4 - 3.0 if s > 0 else np.nan
    return m, s, kurt


def plot_distributions(centers, P, sigmas, label, out_path):
    """P(fx), P(fy) on a log scale, each with the Gaussian of the same
    variance for reference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name, p, s in zip(axes, ['f_x', 'f_y'], P, sigmas):
        ax.semilogy(centers, p, 'o', ms=3, label=f'${name}$')
        if s > 0:
            g = np.exp(-centers ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
            ax.semilogy(centers, g, '-', color='gray', lw=1, label=f'Gaussian, $\\sigma$ = {s:.3g}')
        if (p > 0).any():
            ax.set_ylim(0.3 * p[p > 0].min(), 3 * p.max())
        ax.set_xlabel(f'${name}$ (vortex-vortex)')
        ax.set_ylabel(f'$P({name})$')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'Interaction force distributions: {label}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Distributions of the in-plane vortex-vortex (repulsive) force on a '
                    'vortex, P(fx), P(fy) and P(|f|), recomputed from the positions in '
                    'config files with the same K1 force and cutoff as main.cu (springs, '
                    'pinning and drive excluded). Accumulated over all vortices, layers '
                    'and matched files.')
    parser.add_argument('pattern', nargs='?', default='configIni.dat',
                        help='a config file, or a glob pattern matching several '
                             '(e.g. "config_step_*.dat") to also accumulate over time')
    parser.add_argument('--step-min', type=int, default=None,
                        help='only use config_step_<N>.dat with N >= this (default: no limit); '
                             'set it past the initial transient')
    parser.add_argument('--step-max', type=int, default=None,
                        help='only use config_step_<N>.dat with N <= this (default: no limit)')
    parser.add_argument('--stride', type=int, default=1,
                        help='use every this-th matched file (default 1: all of them)')
    parser.add_argument('--bins', type=int, default=101,
                        help='number of histogram bins (default 101)')
    parser.add_argument('--fmax', type=float, default=None,
                        help='histogram range [-fmax, fmax] for fx, fy and [0, fmax] for |f| '
                             '(default: the largest |component| found)')
    parser.add_argument('--cutoff', type=float, default=None,
                        help='interaction cutoff (default: Interaction Cutoff from simulation.log)')
    parser.add_argument('--shifted-force', action='store_true',
                        help='subtract K1(cutoff), as vortex_sim --shifted-force (default: '
                             'from simulation.log)')
    parser.add_argument('--out-prefix', default=None,
                        help='output file prefix (default: derived from the matched file(s))')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
    args = parser.parse_args()

    files = gather_configs(args.pattern, args.step_min, args.step_max, args.stride)
    if not files:
        rng = ''
        if args.step_min is not None or args.step_max is not None:
            rng = f" with step in [{args.step_min}, {args.step_max}]"
        print(f"Error: no files matched '{args.pattern}'{rng}.")
        sys.exit(1)

    if len(files) == 1:
        print(f"Using {files[0]}")
    else:
        print(f"Accumulating over {len(files)} configurations: "
              f"{os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
    label = config_label(files)

    Lx, Ly = find_box_size(files[0])
    if Lx is None:
        print("Error: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "the forces need the box size for periodic wrapping.")
        sys.exit(1)
    cutoff = args.cutoff if args.cutoff is not None else find_log_value(files[0], 'Interaction Cutoff')
    if cutoff is None:
        print("Error: could not read Interaction Cutoff from simulation.log; pass --cutoff.")
        sys.exit(1)
    shifted = args.shifted_force or bool(find_log_value(files[0], 'Shifted Force'))
    if cutoff >= 0.5 * min(Lx, Ly):
        print(f"Error: cutoff {cutoff} is not below half the box ({0.5 * min(Lx, Ly)}); "
              "the minimum image would miss pairs.")
        sys.exit(1)

    FX, FY = [], []
    worst_net = 0.0
    for fpath in files:
        try:
            df = load_config(fpath)
        except FileNotFoundError:
            print(f"Error: File '{fpath}' not found.")
            sys.exit(1)
        for z in sorted(df['z'].unique()):
            g = df[df['z'] == z]
            fx, fy = pair_forces_layer(g['x'].to_numpy(), g['y'].to_numpy(), Lx, Ly, cutoff, shifted)
            # Pair forces cancel within a layer (Newton's third law).
            worst_net = max(worst_net, abs(fx.sum()), abs(fy.sum()))
            FX.append(fx)
            FY.append(fy)
    fx = np.concatenate(FX)
    fy = np.concatenate(FY)
    fabs = np.hypot(fx, fy)
    kind = 'shifted-force cutoff' if shifted else 'cutoff'
    print(f"{len(fx)} vortex forces ({kind} {cutoff}); largest net force per layer: {worst_net:.2e}")

    fmax = args.fmax if args.fmax is not None else max(np.abs(fx).max(), np.abs(fy).max())
    if fmax <= 0:
        fmax = 1.0   # e.g. a perfect lattice, every force zero
    edges = np.linspace(-fmax, fmax, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Px, _ = np.histogram(fx, bins=edges, density=True)
    Py, _ = np.histogram(fy, bins=edges, density=True)
    edges_abs = np.linspace(0.0, fmax * np.sqrt(2), args.bins + 1)
    centers_abs = 0.5 * (edges_abs[:-1] + edges_abs[1:])
    Pabs, _ = np.histogram(fabs, bins=edges_abs, density=True)
    outside = ((np.abs(fx) > fmax) | (np.abs(fy) > fmax)).mean()
    if outside > 0:
        print(f"Note: {outside:.2%} of the forces fall outside [-fmax, fmax] and are not binned.")

    mx, sx, kx = moments(fx)
    my, sy, ky = moments(fy)
    print(f"fx: mean {mx:.4g}, std {sx:.4g}, excess kurtosis {kx:.3g}")
    print(f"fy: mean {my:.4g}, std {sy:.4g}, excess kurtosis {ky:.3g}")
    print(f"|f|: mean {fabs.mean():.4g}, rms {np.sqrt((fabs ** 2).mean()):.4g}")

    prefix = default_out_prefix(files, args.out_prefix)
    header = (f"{label}; {len(fx)} vortex forces, {kind} {cutoff}\n"
              f"fx: mean {mx:.6g} std {sx:.6g} excess_kurtosis {kx:.6g}\n"
              f"fy: mean {my:.6g} std {sy:.6g} excess_kurtosis {ky:.6g}\n")
    np.savetxt(f"{prefix}_forces.dat", np.column_stack([centers, Px, Py]),
               header=header + "f\tP(fx)\tP(fy)", delimiter='\t', fmt='%.6e')
    print(f"Saved {prefix}_forces.dat")
    np.savetxt(f"{prefix}_forces_abs.dat", np.column_stack([centers_abs, Pabs]),
               header=header + "|f|\tP(|f|)", delimiter='\t', fmt='%.6e')
    print(f"Saved {prefix}_forces_abs.dat")

    if not args.show:
        matplotlib.use('Agg')
    plot_distributions(centers, [Px, Py], [sx, sy], label, f"{prefix}_forces.png")
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
