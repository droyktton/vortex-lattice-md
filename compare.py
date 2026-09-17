import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description='Overlay several two-column .dat files (as written by msd.py '
                     'or analyze.py --  msd.dat, <name>_sq.dat) on one plot, to '
                     'compare across runs -- different temperatures, sizes, etc.')
    parser.add_argument('files', nargs='+',
                        help='label:path pairs, e.g. T=0.02:T_0.02/msd.dat T=0.03:T_0.03/msd.dat')
    parser.add_argument('--out', default='compare.png', help='output image path (default compare.png)')
    parser.add_argument('--xlabel', default='x')
    parser.add_argument('--ylabel', default='y')
    parser.add_argument('--title', default='')
    parser.add_argument('--logy', action='store_true', help='log scale on y (e.g. for S(q))')
    parser.add_argument('--hline', type=float, default=None,
                        help='draw a horizontal reference line at this y (e.g. 1.0 for S(q))')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
    args = parser.parse_args()

    cmap = matplotlib.colormaps['coolwarm']
    n = len(args.files)

    plt.figure(figsize=(7, 5))
    for i, spec in enumerate(args.files):
        if ':' not in spec:
            print(f"Error: '{spec}' is not label:path")
            sys.exit(1)
        label, path = spec.split(':', 1)
        try:
            data = np.loadtxt(path, skiprows=1)
        except OSError:
            print(f"Error: could not read '{path}'")
            sys.exit(1)
        color = cmap(i / max(n - 1, 1))
        plt.plot(data[:, 0], data[:, 1], label=label, color=color)

    if args.hline is not None:
        plt.axhline(args.hline, color='gray', linestyle='--', linewidth=1)
    if args.logy:
        plt.yscale('log')
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.grid(True, which='both' if args.logy else 'major')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
