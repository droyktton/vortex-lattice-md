import sys
import os
import re
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the 3d projection)


def load_config(filename):
    # 3 header lines: "Iter = ...", "Temperatura = ...", column names.
    df = pd.read_csv(
        filename,
        skiprows=3,
        sep=r'\s+',
        names=['x', 'y', 'z', 'fx', 'fy', 'vortex', 'label', 'cell'],
    )
    df = df.dropna(subset=['x', 'y'])
    df['z'] = df['z'].astype(int)
    df['label'] = df['label'].astype(int)
    return df


def find_log_value(filename, label):
    """Read a `label : value` line from the simulation.log next to this
    config file (shared by all the analysis/visualization scripts)."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(filename)), 'simulation.log')
    if not os.path.exists(log_path):
        return None
    text = open(log_path).read()
    m = re.search(re.escape(label) + r'\s*:\s*([0-9.eE+-]+)', text)
    return float(m.group(1)) if m else None


def find_box_size(filename):
    """Read Lx, Ly, so inter-layer segments can be unwrapped across the
    periodic x/y boundary."""
    Lx = find_log_value(filename, 'Box Size Lx')
    Ly = find_log_value(filename, 'Box Size Ly')
    if Lx is None or Ly is None:
        return None, None
    return Lx, Ly


def parse_step(filename):
    """Extract N from a config_step_<N>.dat filename, or None (e.g. for
    configIni.dat) if it doesn't match that pattern."""
    m = re.search(r'config_step_(\d+)\.dat$', os.path.basename(filename))
    return int(m.group(1)) if m else None


def gather_configs(pattern, step_min, step_max, stride=1):
    """Resolve `pattern` to a sorted list of config files. Shared by
    analyze.py and disclinations.py, whose first argument accepts either a
    single config file or a glob pattern to average/aggregate over time.

    A literal filename with no glob metacharacter (e.g. the default
    'configIni.dat') is used as-is, whatever it's named.

    A wildcard pattern (e.g. 'config_step_*.dat') is expected to match
    trajectory snapshots, so results are restricted to files that parse as
    config_step_<N>.dat and fall in [step_min, step_max], then subsampled
    every `stride`-th one. This also matters because the pattern can
    accidentally pick up a script's OWN outputs sitting in the same
    directory (config_step_<N>_sq.dat, _gr.png, ...) -- those don't match
    the strict step pattern and are silently dropped rather than smuggled in
    unfiltered."""
    if not any(c in pattern for c in '*?['):
        return [pattern]

    dated = sorted((parse_step(f), f) for f in glob.glob(pattern) if parse_step(f) is not None)
    in_range = [f for s, f in dated
                if (step_min is None or s >= step_min) and (step_max is None or s <= step_max)]
    return in_range[::stride]


def config_label(files):
    """Short label for plot titles: one filename, or 'N configs, steps A-B'."""
    if len(files) == 1:
        return os.path.basename(files[0])
    s0, s1 = parse_step(files[0]), parse_step(files[-1])
    return f"{len(files)} configs, steps {s0}-{s1}" if s0 is not None else f"{len(files)} configs"


def default_out_prefix(files, out_prefix=None):
    """Where multi-config scripts (analyze.py, disclinations.py) write their
    output, absent an explicit --out-prefix."""
    if out_prefix:
        return out_prefix
    if len(files) == 1:
        base, _ = os.path.splitext(files[0])
        return base
    d = os.path.dirname(files[0]) or '.'
    s0, s1 = parse_step(files[0]), parse_step(files[-1])
    return os.path.join(d, f"avg_step{s0}-{s1}" if s0 is not None else "avg")


def unwrap_along_z(xs, ys, Lx, Ly):
    """A flux line's in-plane position is folded into [0, L) independently at
    each layer, so consecutive layers can appear to jump across the whole box.
    Undo that with a minimum-image walk so the line reads as one continuous
    curve, the same way min_dist() in main.cu treats any other separation."""
    ux, uy = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        if Lx:
            dx -= Lx * round(dx / Lx)
        if Ly:
            dy -= Ly * round(dy / Ly)
        ux.append(ux[-1] + dx)
        uy.append(uy[-1] + dy)
    return np.array(ux), np.array(uy)


def plot_layers_2d(df, filename, out_path):
    plt.figure(figsize=(7, 6))
    for layer in sorted(df['z'].unique()):
        layer_df = df[df['z'] == layer]
        plt.scatter(layer_df['x'], layer_df['y'], label=f'Layer z={layer}', alpha=0.7, s=25)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Vortex positions per layer — {os.path.basename(filename)}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_flux_lines_3d(df, filename, out_path, Lx, Ly):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # `label` = global_id % Nxy is the vortex line's identity, constant across
    # layers (see InitTriangularLattice in main.cu), so grouping by it links
    # each line's positions across the z-stack into one continuous flux line.
    records = []
    for label, g in df.groupby('label'):
        g = g.sort_values('z')
        xs, ys, zs = g['x'].to_numpy(), g['y'].to_numpy(), g['z'].to_numpy()
        ux, uy = unwrap_along_z(xs, ys, Lx, Ly)
        bend = float(np.hypot(ux - ux.mean(), uy - uy.mean()).max())
        records.append((ux, uy, zs, bend))

    bends = np.array([r[3] for r in records])
    vmax = max(float(bends.max()), 1e-6)
    cmap = matplotlib.colormaps['plasma']

    for ux, uy, zs, bend in records:
        color = cmap(bend / vmax)
        ax.plot(ux, uy, zs, color=color, linewidth=1.0, alpha=0.85)
        ax.scatter(ux, uy, zs, color=color, s=8)

    mappable = matplotlib.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(bends)
    fig.colorbar(mappable, ax=ax, shrink=0.6, label='Flux line transverse wander')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer (z)')
    ax.set_title(f'Vortex flux lines — {os.path.basename(filename)}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize a vortex lattice configuration file.')
    parser.add_argument('filename', nargs='?', default='configIni.dat',
                        help='config_*.dat file printed by the simulation')
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
        print("Warning: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "flux lines will not be unwrapped across the periodic x/y boundary.")

    base, _ = os.path.splitext(args.filename)
    plot_layers_2d(df, args.filename, base + '_layers.png')
    plot_flux_lines_3d(df, args.filename, base + '_3d.png', Lx, Ly)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
