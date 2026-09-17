import sys
import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay

from vizconfig import load_config, find_box_size, find_log_value


def periodic_neighbors(x, y, Lx, Ly, bond_cutoff=None):
    """Delaunay-triangulation neighbor lists under periodic boundary
    conditions, via the standard ghost-image trick: tile the layer's points
    into all 8 neighboring periodic images plus the original, run an
    ordinary (non-periodic) Delaunay on the padded set, then fold every edge
    back to the original point indices. Valid as long as the box is large
    compared to the nearest-neighbor spacing, which holds here.

    A near-perfect triangular lattice is close to the worst case for this:
    every hexagonal ring of 6 neighbors sits almost exactly on a circle
    around the central point, so the local Delaunay triangulation is nearly
    degenerate there (any of the 3 ways to slice the hexagon into triangles
    is an almost-equally-valid triangulation). Floating point can pick the
    "wrong" diagonal, wiring up a spurious bond to a *second*-shell neighbor
    (distance a0*sqrt(3)) instead of the real first-shell one (distance a0)
    -- inflating the apparent defect count with numerical artifacts rather
    than real disclinations. bond_cutoff drops any Delaunay edge longer than
    that, which -- sitting between the two shells -- keeps genuine bonds and
    discards the degenerate ones."""
    N = len(x)
    shifts = [(sx, sy) for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
    px = np.concatenate([x + sx * Lx for sx, sy in shifts])
    py = np.concatenate([y + sy * Ly for sx, sy in shifts])
    orig = np.tile(np.arange(N), len(shifts))

    tri = Delaunay(np.column_stack([px, py]))
    neighbors = [set() for _ in range(N)]
    for simplex in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                ia, ib = simplex[a], simplex[b]
                if bond_cutoff is not None:
                    d = np.hypot(px[ia] - px[ib], py[ia] - py[ib])
                    if d > bond_cutoff:
                        continue
                i, j = orig[ia], orig[ib]
                if i != j:
                    neighbors[i].add(int(j))
                    neighbors[j].add(int(i))
    return neighbors


def bond_segments(x, y, neighbors, Lx, Ly):
    """Delaunay bonds as (possibly periodicity-unwrapped) line segments, one
    per unique neighbor pair, for plotting."""
    segments = []
    for i, ns in enumerate(neighbors):
        for j in ns:
            if j <= i:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            segments.append([(x[i], y[i]), (x[i] + dx, y[i] + dy)])
    return segments


# Disclination charge = 6 - coordination. Colors follow the usual convention
# in 2D defect plots: 5-fold (charge +1) blue, 7-fold (charge -1) red,
# anything more exotic (4-, 8-fold, ...) orange, 6-fold (no defect) gray.
CHARGE_COLORS = {1: 'tab:blue', -1: 'tab:red'}
DEFAULT_COLOR = 'tab:orange'
BULK_COLOR = '0.65'


def plot_layer(ax, x, y, Lx, Ly, neighbors):
    coord = np.array([len(n) for n in neighbors])
    charge = 6 - coord

    segments = bond_segments(x, y, neighbors, Lx, Ly)
    ax.add_collection(LineCollection(segments, colors='0.8', linewidths=0.5, zorder=1))

    colors = np.full(len(x), BULK_COLOR, dtype=object)
    for q, c in CHARGE_COLORS.items():
        colors[charge == q] = c
    colors[(charge != 0) & ~np.isin(charge, list(CHARGE_COLORS))] = DEFAULT_COLOR
    sizes = np.where(charge == 0, 5, 22)

    ax.scatter(x, y, c=colors, s=sizes, zorder=2, edgecolors='none')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')

    n_defects = int(np.sum(coord != 6))
    n5 = int(np.sum(charge == 1))
    n7 = int(np.sum(charge == -1))
    n_other = n_defects - n5 - n7
    return n_defects, n5, n7, n_other, coord


def plot_coord_histogram(coord_by_layer, out_path):
    """Coordination number histogram, one group of bars per layer so they're
    directly comparable. A perfect triangular lattice is a single spike at 6;
    a liquid spreads out around it, roughly symmetric between 5- and 7-fold."""
    layers = list(coord_by_layer.keys())
    all_coord = np.concatenate(list(coord_by_layer.values()))
    c_min, c_max = int(all_coord.min()), int(all_coord.max())
    edges = np.arange(c_min, c_max + 2)
    centers = edges[:-1]
    width = 0.8 / len(layers)

    plt.figure(figsize=(7, 5))
    for i, z in enumerate(layers):
        counts, _ = np.histogram(coord_by_layer[z], bins=edges)
        offset = (i - (len(layers) - 1) / 2) * width
        plt.bar(centers + offset, counts, width=width, label=f'z={z}')
    plt.axvline(6, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('coordination number')
    plt.ylabel('count')
    plt.xticks(centers)
    plt.title('Coordination number histogram')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Per-layer periodic Delaunay triangulation of a vortex lattice '
                     'configuration, marking disclinations (coordination number != 6).')
    parser.add_argument('filename', nargs='?', default='configIni.dat',
                        help='config_*.dat file printed by the simulation')
    parser.add_argument('--bond-cutoff', type=float, default=None,
                        help='max Delaunay bond length (default: 1.35*a0, between the 1st '
                             '(a0) and 2nd (a0*sqrt(3)) neighbor shells) -- drops the '
                             'spurious long edges a near-degenerate triangular lattice can '
                             'produce; pass 0 to disable filtering entirely')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
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
              "the periodic triangulation needs the box size.")
        sys.exit(1)

    if args.bond_cutoff is not None:
        bond_cutoff = args.bond_cutoff if args.bond_cutoff > 0 else None
    else:
        a0 = find_log_value(args.filename, 'Lattice Constant (a0)')
        if a0:
            bond_cutoff = 1.35 * a0
        else:
            print("Warning: could not read a0 from simulation.log; not filtering Delaunay "
                  "bonds by length -- defect counts may include numerical artifacts from "
                  "near-degenerate triangulation (see --bond-cutoff).")
            bond_cutoff = None

    layers = sorted(df['z'].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(4.5 * len(layers), 4.5), squeeze=False)
    axes = axes[0]

    rows = []
    coord_by_layer = {}
    for ax, z in zip(axes, layers):
        g = df[df['z'] == z]
        x, y = g['x'].to_numpy(), g['y'].to_numpy()
        neighbors = periodic_neighbors(x, y, Lx, Ly, bond_cutoff)
        n_defects, n5, n7, n_other, coord = plot_layer(ax, x, y, Lx, Ly, neighbors)
        frac = n_defects / len(x)
        rows.append((z, len(x), n_defects, frac, n5, n7, n_other))
        coord_by_layer[z] = coord
        ax.set_title(f'z={z} — {n_defects}/{len(x)} defects ({100*frac:.1f}%)', fontsize=9)
        print(f"  z={z}: N={len(x)}  defects={n_defects} ({100*frac:.2f}%)  "
              f"5-fold={n5}  7-fold={n7}  other={n_other}")

    total_N = sum(r[1] for r in rows)
    total_defects = sum(r[2] for r in rows)
    print(f"Total: {total_defects}/{total_N} defects ({100*total_defects/total_N:.2f}%) "
          f"across {len(layers)} layers")

    fig.suptitle(f'Delaunay triangulation & disclinations — {os.path.basename(args.filename)}')
    fig.tight_layout()

    base, _ = os.path.splitext(args.filename)
    out_path = base + '_disclinations.png'
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    dat_path = base + '_disclinations.dat'
    np.savetxt(dat_path, np.array(rows),
               header='z  N  n_defects  defect_fraction  n_5fold  n_7fold  n_other', comments='')
    print(f"Saved {dat_path}")

    plot_coord_histogram(coord_by_layer, base + '_coord_hist.png')

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
