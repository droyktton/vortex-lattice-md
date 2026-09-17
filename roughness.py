import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from vizconfig import load_config, find_box_size, gather_configs, config_label, default_out_prefix, unwrap_along_z


def line_roughness(df, Lx, Ly):
    """For each vortex line (persistent in-plane `label`, constant across
    layers), the mean squared transverse displacement of its Nz pancake
    positions from their own center of mass:

        <u^2> = (1/Nz) sum_z |r(z) - r_cm|^2,   r_cm = (1/Nz) sum_z r(z)

    r(z) is unwrapped across z first (the same minimum-image walk used for
    the 3D flux-line plot), since folding each layer independently into
    [0, Lx) x [0, Ly) would otherwise make r_cm meaningless for a line
    anchored near a box edge. Returns one value per line (length Nxy)."""
    values = []
    for _, g in df.groupby('label'):
        g = g.sort_values('z')
        xs, ys = g['x'].to_numpy(), g['y'].to_numpy()
        ux, uy = unwrap_along_z(xs, ys, Lx, Ly)
        xcm, ycm = ux.mean(), uy.mean()
        values.append(np.mean((ux - xcm) ** 2 + (uy - ycm) ** 2))
    return np.array(values)


def plot_hist(u2, label, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(u2, bins=40)
    mean_u2 = float(u2.mean())
    plt.axvline(mean_u2, color='tab:red', linestyle='--', label=f'mean = {mean_u2:.4g}')
    plt.xlabel(r'$\langle u^2 \rangle$ per vortex line')
    plt.ylabel('count')
    plt.title(f'Vortex line roughness — {label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Vortex line roughness: mean squared transverse displacement of each '
                     "line's pancakes from their own (z-unwrapped) center of mass, "
                     'averaged over all lines (and optionally over time).')
    parser.add_argument('pattern', nargs='?', default='configIni.dat',
                        help='a config file, or a glob pattern matching several '
                             '(e.g. "config_step_*.dat") to also average over time, '
                             'in addition to over lines')
    parser.add_argument('--step-min', type=int, default=None,
                        help='only use config_step_<N>.dat with N >= this (default: no limit)')
    parser.add_argument('--step-max', type=int, default=None,
                        help='only use config_step_<N>.dat with N <= this (default: no limit)')
    parser.add_argument('--stride', type=int, default=1,
                        help='use every this-th matched file (default 1: all of them)')
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
        print(f"Averaging over {len(files)} configurations: "
              f"{os.path.basename(files[0])} .. {os.path.basename(files[-1])}")
    label = config_label(files)

    Lx, Ly = find_box_size(files[0])
    if Lx is None:
        print("Error: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "unwrapping across z needs the box size.")
        sys.exit(1)

    all_u2 = []
    for fpath in files:
        try:
            df = load_config(fpath)
        except FileNotFoundError:
            print(f"Error: File '{fpath}' not found.")
            sys.exit(1)
        all_u2.append(line_roughness(df, Lx, Ly))

    u2 = np.concatenate(all_u2)
    mean_u2 = float(u2.mean())
    print(f"<u^2> = {mean_u2:.6g}  (sqrt = {mean_u2 ** 0.5:.6g}), "
          f"averaged over {len(u2)} (line, config) samples")

    base = default_out_prefix(files, args.out_prefix)
    np.savetxt(base + '_roughness.dat', [[mean_u2, len(u2)]],
               header='mean_u2  n_samples', comments='')
    print(f"Saved {base}_roughness.dat")
    plot_hist(u2, label, base + '_roughness_hist.png')

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
