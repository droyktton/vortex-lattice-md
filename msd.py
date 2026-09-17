import sys
import os
import glob
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from vizconfig import load_config, find_box_size, find_log_value, parse_step


def gather_snapshots(pattern):
    """Find config_step_<N>.dat files matching `pattern`, sorted by step."""
    snaps = []
    for f in glob.glob(pattern):
        step = parse_step(f)
        if step is not None:
            snaps.append((step, f))
    snaps.sort(key=lambda t: t[0])
    return snaps


def keep_uniform_prefix(snaps):
    """Drop any trailing snapshots that break uniform step spacing (e.g. the
    simulation always force-saves the final step, which usually doesn't land
    on the --print-interval grid). Returns (kept_snaps, spacing_in_steps)."""
    if len(snaps) < 2:
        return snaps, None
    steps = [s for s, _ in snaps]
    spacing = int(np.median(np.diff(steps)))
    kept = [snaps[0]]
    expected = steps[0] + spacing
    for s, f in snaps[1:]:
        if s == expected:
            kept.append((s, f))
            expected += spacing
    dropped = len(snaps) - len(kept)
    if dropped:
        print(f"Note: dropped {dropped} snapshot(s) that break the uniform "
              f"{spacing}-step spacing (e.g. the forced final-step save).")
    return kept, spacing


def load_trajectory(snaps):
    """Positions aligned by the persistent global vortex id ('vortex' column,
    constant over time for a given particle), shape (Nt, Nmax)."""
    xs = ys = None
    for i, (_, f) in enumerate(snaps):
        df = load_config(f).sort_values('vortex')
        x, y = df['x'].to_numpy(), df['y'].to_numpy()
        if xs is None:
            xs = np.empty((len(snaps), len(x)))
            ys = np.empty((len(snaps), len(x)))
        xs[i], ys[i] = x, y
    return xs, ys


def unwrap_trajectory(xs, ys, Lx, Ly):
    """Undo periodic folding between consecutive snapshots (minimum image),
    so displacements aren't corrupted by a particle crossing the box edge.
    Assumes true displacement between consecutive SAVED snapshots is less
    than half the box -- true as long as --print-interval isn't too coarse
    relative to how fast the vortices actually move."""
    ux, uy = np.empty_like(xs), np.empty_like(ys)
    ux[0], uy[0] = xs[0], ys[0]
    for t in range(1, xs.shape[0]):
        dx, dy = xs[t] - xs[t - 1], ys[t] - ys[t - 1]
        if Lx:
            dx -= Lx * np.round(dx / Lx)
        if Ly:
            dy -= Ly * np.round(dy / Ly)
        ux[t] = ux[t - 1] + dx
        uy[t] = uy[t - 1] + dy
    return ux, uy


def compute_msd(ux, uy, origin_spacing, window, t0_min=0):
    """<[r(t0+lag) - r(t0)]^2>, averaged over particles and over equally
    spaced reference times t0 (starting no earlier than snapshot `t0_min`,
    to skip the initial equilibration transient) within a fixed window of
    `window` snapshots."""
    Nt = ux.shape[0]
    if window is None or window >= Nt - t0_min:
        window = Nt - 1 - t0_min

    origins = list(range(t0_min, Nt - window, origin_spacing))
    if not origins:
        origins = [t0_min]
        window = Nt - 1 - t0_min

    lags = np.arange(0, window + 1)
    msd_per_origin = np.empty((len(origins), len(lags)))
    for i, t0 in enumerate(origins):
        dx = ux[t0:t0 + window + 1] - ux[t0]
        dy = uy[t0:t0 + window + 1] - uy[t0]
        msd_per_origin[i] = (dx ** 2 + dy ** 2).mean(axis=1)
    msd = msd_per_origin.mean(axis=0)
    return lags, msd, origins, msd_per_origin


def plot_msd(t, msd, out_path, time_label):
    plt.figure(figsize=(7, 5))
    plt.plot(t, msd, marker='o', markersize=3)
    plt.xlabel(time_label)
    plt.ylabel(r'MSD $= \langle [r(t_0+\Delta t) - r(t_0)]^2 \rangle$')
    plt.title('Vortex mean squared displacement')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_msd_windows(t, msd_per_origin, t0_physical, msd_avg, out_path, time_label):
    """One curve per time origin, colored by t0, plus the average -- lets you
    check by eye whether the per-window MSD has settled down (curves from
    different t0 overlap) or is still drifting (steady state not reached)."""
    plt.figure(figsize=(7, 5))
    cmap = matplotlib.colormaps['viridis']
    vmax = max(float(np.max(t0_physical)), 1e-12)
    for row, t0 in zip(msd_per_origin, t0_physical):
        plt.plot(t, row, color=cmap(t0 / vmax), alpha=0.6, linewidth=1)
    plt.plot(t, msd_avg, color='black', linewidth=2, label='average over windows')

    mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    mappable.set_array(t0_physical)
    plt.colorbar(mappable, ax=plt.gca(), label=r'window start $t_0$')

    plt.xlabel(time_label)
    plt.ylabel(r'MSD per window $= [r(t_0+\Delta t) - r(t_0)]^2$')
    plt.title('Per-window MSD (check convergence to steady state)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Mean squared displacement of the vortices vs time, averaged '
                     'over particles and over equally spaced reference times t0 '
                     'within a fixed time window.')
    parser.add_argument('pattern', nargs='?', default='config_step_*.dat',
                        help='glob pattern for the trajectory snapshots (default: config_step_*.dat)')
    parser.add_argument('--origin-spacing', type=int, default=1,
                        help='spacing between reference times t0, in snapshots (default 1: every snapshot is an origin)')
    parser.add_argument('--window', type=int, default=None,
                        help='max lag per origin, in snapshots (default: as large as the data allows)')
    parser.add_argument('--t0-min', type=int, default=0,
                        help='skip this many snapshots before the first allowed reference time t0 '
                             '(default 0); use it to discard the initial equilibration transient')
    parser.add_argument('--out-prefix', default=None,
                        help='output file prefix (default: msd, next to the snapshots)')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
    args = parser.parse_args()

    snaps = gather_snapshots(args.pattern)
    if len(snaps) < 2:
        print(f"Error: found {len(snaps)} snapshot(s) matching '{args.pattern}'; need at least 2.")
        sys.exit(1)

    snaps, spacing = keep_uniform_prefix(snaps)
    print(f"Using {len(snaps)} snapshots, steps {snaps[0][0]}..{snaps[-1][0]} "
          f"(spacing {spacing} step{'s' if spacing != 1 else ''})")

    Lx, Ly = find_box_size(snaps[0][1])
    dt = find_log_value(snaps[0][1], 'Time Step (dt)')
    if Lx is None:
        print("Warning: could not read Box Size Lx/Ly from a sibling simulation.log; "
              "displacements will not be unwrapped across the periodic boundary.")

    if args.t0_min >= len(snaps) - 1:
        print(f"Error: --t0-min {args.t0_min} leaves fewer than 2 snapshots "
              f"({len(snaps)} available); reduce it.")
        sys.exit(1)

    xs, ys = load_trajectory(snaps)
    ux, uy = unwrap_trajectory(xs, ys, Lx, Ly)

    lags, msd, origins, msd_per_origin = compute_msd(ux, uy, args.origin_spacing, args.window, args.t0_min)
    print(f"{len(origins)} time origin(s), window {lags[-1]} snapshots")

    lag_steps = lags * spacing
    t0_steps = np.array(origins) * spacing
    if dt:
        t_axis, time_label = lag_steps * dt, r'$\Delta t$ (simulation time)'
        t0_physical = t0_steps * dt
    else:
        t_axis, time_label = lag_steps, r'$\Delta t$ (steps)'
        t0_physical = t0_steps

    out_prefix = args.out_prefix or os.path.join(os.path.dirname(snaps[0][1]) or '.', 'msd')
    np.savetxt(out_prefix + '.dat', np.column_stack([t_axis, msd]),
               header='t  MSD', comments='')
    print(f"Saved {out_prefix}.dat")
    plot_msd(t_axis, msd, out_prefix + '.png', time_label)
    plot_msd_windows(t_axis, msd_per_origin, t0_physical, msd, out_prefix + '_windows.png', time_label)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
