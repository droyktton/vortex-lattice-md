import sys
import os
import glob
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from vizconfig import load_config, find_box_size, find_log_value, parse_step


def gather_configs(pattern, step_min, step_max, stride=1):
    """Resolve `pattern` to a sorted list of config files.

    A literal filename with no glob metacharacter (e.g. the default
    'configIni.dat') is used as-is, whatever it's named.

    A wildcard pattern (e.g. 'config_step_*.dat') is expected to match
    trajectory snapshots, so results are restricted to files that parse as
    config_step_<N>.dat and fall in [step_min, step_max], then subsampled
    every `stride`-th one. This also matters because the pattern can
    accidentally pick up this script's OWN outputs sitting in the same
    directory (config_step_<N>_sq.dat, _gr.png, ...) -- those don't match
    the strict step pattern and are silently dropped rather than smuggled in
    unfiltered. S(k) is the expensive part here, O(grid_size x N) per
    (config, layer), so a long time range can get costly to average in full;
    that's what --stride is for."""
    if not any(c in pattern for c in '*?['):
        return [pattern]

    dated = sorted((parse_step(f), f) for f in glob.glob(pattern) if parse_step(f) is not None)
    in_range = [f for s, f in dated
                if (step_min is None or s >= step_min) and (step_max is None or s <= step_max)]
    return in_range[::stride]


def compute_gr_layer(x, y, Lx, Ly, dr, r_max):
    """2D radial distribution function for one layer, periodic minimum image."""
    N = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dx -= Lx * np.round(dx / Lx)
    dy -= Ly * np.round(dy / Ly)
    r = np.hypot(dx, dy)

    iu = np.triu_indices(N, k=1)
    bins = np.arange(0.0, r_max + dr, dr)
    counts, edges = np.histogram(r[iu], bins=bins)
    counts = counts * 2  # unordered pairs -> both (i,j) and (j,i) neighbor contributions

    rho = N / (Lx * Ly)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    shell_area = 2 * np.pi * r_mid * dr
    g = counts / (N * rho * shell_area)
    return r_mid, g


def compute_sk_layer(x, y, Lx, Ly, k_max):
    """Structure factor S(kx, ky) = |sum_j exp(-i k.r_j)|^2 / N, evaluated
    only at the box's OWN reciprocal lattice: kx_m = 2*pi*m/Lx, ky_n =
    2*pi*n/Ly for integer m, n. These are the only k-points at which summing
    over the N particles in one periodic cell exactly reproduces the
    coherent scattering of the infinite PBC-tiled system -- every periodic
    image contributes the identical phase, exp(-i k_mn . (Lx*p, Ly*q)) = 1
    for any integers p, q. At any other k, the same sum is indistinguishable
    from the Fourier transform of a single finite, non-periodic rectangular
    cluster, which produces spurious sinc-like streaking along the axes
    (visible as a "cross" through the origin) -- a finite-window artifact of
    the box shape, not real structure, and it does not go away just because
    the simulation itself used periodic boundaries."""
    N = len(x)
    m_max = max(1, int(np.floor(k_max * Lx / (2 * np.pi))))
    n_max = max(1, int(np.floor(k_max * Ly / (2 * np.pi))))
    kx_axis = (2 * np.pi / Lx) * np.arange(-m_max, m_max + 1)
    ky_axis = (2 * np.pi / Ly) * np.arange(-n_max, n_max + 1)
    KX, KY = np.meshgrid(kx_axis, ky_axis)
    phase = KX.ravel()[:, None] * x[None, :] + KY.ravel()[:, None] * y[None, :]
    rho_k = np.exp(-1j * phase).sum(axis=1)
    S = (np.abs(rho_k) ** 2) / N
    return KX, KY, S.reshape(KX.shape)


def radial_average_sk(KX, KY, S, dq, q_max):
    """Azimuthal average of S(kx, ky) over rings of fixed q = sqrt(kx^2+ky^2),
    the same way g(r) is already a radial average in real space. Excludes the
    trivial k=0 point (S(0) = N)."""
    q = np.hypot(KX, KY).ravel()
    s = S.ravel()
    mask = q > 1e-9
    q, s = q[mask], s[mask]

    bins = np.arange(0.0, q_max + dq, dq)
    sum_s, edges = np.histogram(q, bins=bins, weights=s)
    counts, _ = np.histogram(q, bins=bins)
    q_mid = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid='ignore'):
        s_radial = sum_s / counts
    return q_mid, s_radial


def plot_gr(r, g, label, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(r, g)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title(f'Radial distribution function (z-averaged) — {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_sk(KX, KY, S, label, out_path):
    plt.figure(figsize=(7, 6))
    # k=0 always equals N (trivial peak); mask it so it doesn't wash out
    # the Bragg peaks on a log color scale.
    S_masked = S.copy()
    center = S_masked.shape[0] // 2, S_masked.shape[1] // 2
    S_masked[center] = np.nan
    vmin = max(np.nanmin(S_masked[S_masked > 0]), 1e-3)
    plt.pcolormesh(KX, KY, S_masked, shading='auto', cmap='inferno',
                    norm=LogNorm(vmin=vmin, vmax=np.nanmax(S_masked)))
    plt.colorbar(label='S(k)')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.gca().set_aspect('equal')
    plt.title(f'Structure factor (z-averaged) — {label}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_sq(q, s, label, out_path, title_suffix=''):
    plt.figure(figsize=(7, 5))
    plt.plot(q, s)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.yscale('log')
    plt.xlabel('q')
    plt.ylabel('S(q)')
    plt.title(f'Structure factor, radially averaged{title_suffix} — {label}')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute the radial distribution function g(r), structure factor '
                     'S(k), and its radial average S(q) from a vortex lattice '
                     'configuration file, averaged over z and (optionally) over time.')
    parser.add_argument('pattern', nargs='?', default='configIni.dat',
                        help='a config file, or a glob pattern matching several '
                             '(e.g. "config_step_*.dat") to also average over time, '
                             'in addition to z')
    parser.add_argument('--step-min', type=int, default=None,
                        help='only average config_step_<N>.dat with N >= this (default: no limit)')
    parser.add_argument('--step-max', type=int, default=None,
                        help='only average config_step_<N>.dat with N <= this (default: no limit); '
                             'use --step-min/--step-max together to average only equilibrated '
                             'configurations, once the initial transient has passed')
    parser.add_argument('--stride', type=int, default=1,
                        help='use every this-th matched file (default 1: all of them); '
                             'S(k) is the expensive part here, so a long --step-min/--step-max '
                             'range may be worth subsampling')
    parser.add_argument('--dr', type=float, default=0.05, help='g(r) bin width (default 0.05)')
    parser.add_argument('--rmax', type=float, default=None,
                        help='g(r) max radius (default: min(Lx,Ly)/2, the minimum-image limit)')
    parser.add_argument('--kmax', type=float, default=None,
                        help='S(k) max |kx|,|ky| (default: ~4 Brillouin zones based on a0); '
                             'the grid itself is fixed by the box size (2*pi/Lx, 2*pi/Ly), not adjustable')
    parser.add_argument('--dq', type=float, default=None,
                        help='S(q) radial bin width (default: finer of 2*pi/Lx, 2*pi/Ly)')
    parser.add_argument('--out-prefix', default=None,
                        help='output file prefix (default: derived from the matched file(s))')
    parser.add_argument('--show', action='store_true', help='also open interactive windows')
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
        label = os.path.basename(files[0])
    else:
        print(f"Averaging over {len(files)} configurations: "
              f"{os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
        s0, s1 = parse_step(files[0]), parse_step(files[-1])
        label = f"{len(files)} configs, steps {s0}-{s1}" if s0 is not None else f"{len(files)} configs"

    Lx, Ly = find_box_size(files[0])
    if Lx is None:
        print("Error: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "g(r) and S(k) need the box size for periodic wrapping and normalization.")
        sys.exit(1)

    r_max = args.rmax if args.rmax is not None else 0.5 * min(Lx, Ly)

    a0 = find_log_value(files[0], 'Lattice Constant (a0)')
    if args.kmax is not None:
        k_max = args.kmax
    elif a0:
        k_max = 4 * (2 * np.pi / a0)
    else:
        print("Warning: could not read a0 from simulation.log; defaulting --kmax to 10.0.")
        k_max = 10.0

    gr_list, sk_list = [], []
    r_mid = None
    KX = KY = None
    for fpath in files:
        try:
            df = load_config(fpath)
        except FileNotFoundError:
            print(f"Error: File '{fpath}' not found.")
            sys.exit(1)
        for z in sorted(df['z'].unique()):
            g = df[df['z'] == z]
            x, y = g['x'].to_numpy(), g['y'].to_numpy()
            r_mid, gr = compute_gr_layer(x, y, Lx, Ly, args.dr, r_max)
            gr_list.append(gr)
            kx_grid, ky_grid, sk = compute_sk_layer(x, y, Lx, Ly, k_max)
            sk_list.append(sk)
            KX, KY = kx_grid, ky_grid

    print(f"Averaged over {len(gr_list)} (config, layer) samples")
    gr_avg = np.mean(gr_list, axis=0)
    sk_avg = np.mean(sk_list, axis=0)

    dq = args.dq if args.dq is not None else min(2 * np.pi / Lx, 2 * np.pi / Ly)
    q_mid, sq_avg = radial_average_sk(KX, KY, sk_avg, dq, k_max)

    if args.out_prefix:
        base = args.out_prefix
    elif len(files) == 1:
        base, _ = os.path.splitext(files[0])
    else:
        d = os.path.dirname(files[0]) or '.'
        s0, s1 = parse_step(files[0]), parse_step(files[-1])
        base = os.path.join(d, f"avg_step{s0}-{s1}" if s0 is not None else "avg")

    plot_gr(r_mid, gr_avg, label, base + '_gr.png')
    plot_sk(KX, KY, sk_avg, label, base + '_sk.png')
    plot_sq(q_mid, sq_avg, label, base + '_sq.png')
    np.savetxt(base + '_sq.dat', np.column_stack([q_mid, sq_avg]), header='q  S(q)', comments='')
    print(f"Saved {base}_sq.dat")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
