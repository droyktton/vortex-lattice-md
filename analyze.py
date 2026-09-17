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


def radial_average_sk(KX, KY, S, dq, q_max, exclude_cross=False):
    """Azimuthal average of S(kx, ky) over rings of fixed q = sqrt(kx^2+ky^2),
    the same way g(r) is already a radial average in real space. Excludes the
    trivial k=0 point (S(0) = N).

    exclude_cross=True additionally drops the whole kx=0 and ky=0 lines, not
    just the origin. Fourier-transforming any finite rectangular window (our
    periodic box) produces spurious sinc-like intensity along the axes
    aligned with its edges -- a finite-size artifact of the box shape, not
    real structure -- and it otherwise leaks into every ring that crosses
    those two lines."""
    q = np.hypot(KX, KY).ravel()
    s = S.ravel()
    mask = q > 1e-9
    if exclude_cross:
        dk = abs(KX[0, 1] - KX[0, 0])
        mask &= (np.abs(KX).ravel() > dk / 2) & (np.abs(KY).ravel() > dk / 2)
    q, s = q[mask], s[mask]

    bins = np.arange(0.0, q_max + dq, dq)
    sum_s, edges = np.histogram(q, bins=bins, weights=s)
    counts, _ = np.histogram(q, bins=bins)
    q_mid = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid='ignore'):
        s_radial = sum_s / counts
    return q_mid, s_radial


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


def plot_sq(q, s, filename, out_path, title_suffix=''):
    plt.figure(figsize=(7, 5))
    plt.plot(q, s)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    plt.yscale('log')
    plt.xlabel('q')
    plt.ylabel('S(q)')
    plt.title(f'Structure factor, radially averaged{title_suffix} — {os.path.basename(filename)}')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute the z-averaged radial distribution function g(r), '
                     'structure factor S(k), and its radial average S(q) '
                     'from a vortex lattice configuration file.')
    parser.add_argument('filename', nargs='?', default='configIni.dat',
                        help='config_*.dat file printed by the simulation')
    parser.add_argument('--dr', type=float, default=0.05, help='g(r) bin width (default 0.05)')
    parser.add_argument('--rmax', type=float, default=None,
                        help='g(r) max radius (default: min(Lx,Ly)/2, the minimum-image limit)')
    parser.add_argument('--kmax', type=float, default=None,
                        help='S(k) max |kx|,|ky| (default: ~4 Brillouin zones based on a0)')
    parser.add_argument('--kres', type=int, default=201, help='S(k) grid points per axis (default 201)')
    parser.add_argument('--dq', type=float, default=None,
                        help='S(q) radial bin width (default: matches the k-grid spacing, 2*kmax/(kres-1))')
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

    dq = args.dq if args.dq is not None else 2 * k_max / (args.kres - 1)
    q_mid, sq_avg = radial_average_sk(KX, KY, sk_avg, dq, k_max)
    q_mid_nc, sq_avg_nc = radial_average_sk(KX, KY, sk_avg, dq, k_max, exclude_cross=True)

    base, _ = os.path.splitext(args.filename)
    plot_gr(r_mid, gr_avg, args.filename, base + '_gr.png')
    plot_sk(KX, KY, sk_avg, args.filename, base + '_sk.png')
    plot_sq(q_mid, sq_avg, args.filename, base + '_sq.png')
    np.savetxt(base + '_sq.dat', np.column_stack([q_mid, sq_avg]), header='q  S(q)', comments='')
    print(f"Saved {base}_sq.dat")
    plot_sq(q_mid_nc, sq_avg_nc, args.filename, base + '_sq_nocross.png',
            title_suffix=' (kx=0, ky=0 excluded)')
    np.savetxt(base + '_sq_nocross.dat', np.column_stack([q_mid_nc, sq_avg_nc]), header='q  S(q)', comments='')
    print(f"Saved {base}_sq_nocross.dat")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
