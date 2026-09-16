import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from vizconfig import load_config, find_box_size, find_log_value


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


def compute_sk_layer(x, y, k_max, n_k):
    """Structure factor S(kx, ky) = |sum_j exp(-i k.r_j)|^2 / N on a regular
    k-grid, for one layer."""
    N = len(x)
    k_axis = np.linspace(-k_max, k_max, n_k)
    KX, KY = np.meshgrid(k_axis, k_axis)
    phase = KX.ravel()[:, None] * x[None, :] + KY.ravel()[:, None] * y[None, :]
    rho_k = np.exp(-1j * phase).sum(axis=1)
    S = (np.abs(rho_k) ** 2) / N
    return KX, KY, S.reshape(KX.shape)


def plot_gr(r, g, filename, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(r, g)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title(f'Radial distribution function (z-averaged) — {os.path.basename(filename)}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_sk(KX, KY, S, filename, out_path):
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
    plt.title(f'Structure factor (z-averaged) — {os.path.basename(filename)}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute the z-averaged radial distribution function g(r) '
                     'and structure factor S(k) from a vortex lattice configuration file.')
    parser.add_argument('filename', nargs='?', default='configIni.dat',
                        help='config_*.dat file printed by the simulation')
    parser.add_argument('--dr', type=float, default=0.05, help='g(r) bin width (default 0.05)')
    parser.add_argument('--rmax', type=float, default=None,
                        help='g(r) max radius (default: min(Lx,Ly)/2, the minimum-image limit)')
    parser.add_argument('--kmax', type=float, default=None,
                        help='S(k) max |kx|,|ky| (default: ~4 Brillouin zones based on a0)')
    parser.add_argument('--kres', type=int, default=201, help='S(k) grid points per axis (default 201)')
    parser.add_argument('--show', action='store_true', help='also open interactive windows')
    args = parser.parse_args()

    print(f"Loading configuration from: {args.filename}")
    try:
        df = load_config(args.filename)
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)

    Lx, Ly = find_box_size(args.filename)
    if Lx is None:
        print("Error: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "g(r) and S(k) need the box size for periodic wrapping and normalization.")
        sys.exit(1)

    r_max = args.rmax if args.rmax is not None else 0.5 * min(Lx, Ly)

    a0 = find_log_value(args.filename, 'Lattice Constant (a0)')
    if args.kmax is not None:
        k_max = args.kmax
    elif a0:
        k_max = 4 * (2 * np.pi / a0)
    else:
        print("Warning: could not read a0 from simulation.log; defaulting --kmax to 10.0.")
        k_max = 10.0

    layers = sorted(df['z'].unique())
    gr_list, sk_list = [], []
    r_mid = None
    KX = KY = None
    for z in layers:
        g = df[df['z'] == z]
        x, y = g['x'].to_numpy(), g['y'].to_numpy()
        r_mid, gr = compute_gr_layer(x, y, Lx, Ly, args.dr, r_max)
        gr_list.append(gr)
        kx_grid, ky_grid, sk = compute_sk_layer(x, y, k_max, args.kres)
        sk_list.append(sk)
        KX, KY = kx_grid, ky_grid

    gr_avg = np.mean(gr_list, axis=0)
    sk_avg = np.mean(sk_list, axis=0)

    base, _ = os.path.splitext(args.filename)
    plot_gr(r_mid, gr_avg, args.filename, base + '_gr.png')
    plot_sk(KX, KY, sk_avg, args.filename, base + '_sk.png')

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
