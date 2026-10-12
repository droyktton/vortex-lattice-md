import os
import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from vizconfig import find_log_value


def load_velocity(path):
    """velocity.dat as a dict of columns (step, t, Fx, Fy, vx, vy, vx_z0, vy_z0, ...)."""
    names = open(path).readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return {n: data[:, i] for i, n in enumerate(names)}


def layer_count(cols):
    return sum(1 for n in cols if n.startswith('vx_z'))


def main():
    parser = argparse.ArgumentParser(
        description='Velocity of a driven vortex lattice, from the velocity.dat that vortex_sim '
                    'writes when --Fdc/--Fac is set: v(t) of the lattice and of every layer, and '
                    'its time average parallel and transverse to the drive.')
    parser.add_argument('path', nargs='?', default='velocity.dat',
                        help='velocity.dat to read (default: velocity.dat)')
    parser.add_argument('--step-min', type=int, default=None,
                        help='only average rows at or past this step (default: half of the run)')
    parser.add_argument('--out-prefix', default=None,
                        help='output file prefix (default: velocity, next to velocity.dat)')
    parser.add_argument('--show', action='store_true', help='also open an interactive window')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: '{args.path}' not found (was the run driven? velocity.dat is only "
              f"written with --Fdc or --Fac).")
        sys.exit(1)
    c = load_velocity(args.path)
    angle = find_log_value(args.path, 'Drive Angle (deg)') or 0.0
    a = np.radians(angle)
    ex, ey = np.cos(a), np.sin(a)
    nz = layer_count(c)

    step_min = args.step_min if args.step_min is not None else int(c['step'][-1]) // 2
    mask = c['step'] >= step_min
    if mask.sum() == 0:
        print(f"Error: no rows at or past step {step_min}.")
        sys.exit(1)

    def par(vx, vy):
        return vx * ex + vy * ey

    def perp(vx, vy):
        return -vx * ey + vy * ex

    v_par, v_perp = par(c['vx'], c['vy']), perp(c['vx'], c['vy'])
    F_par = par(c['Fx'], c['Fy'])
    V_par, V_perp = v_par[mask].mean(), v_perp[mask].mean()
    F_mean = F_par[mask].mean()
    layers_par = [par(c[f'vx_z{z}'], c[f'vy_z{z}'])[mask].mean() for z in range(nz)]
    layers_perp = [perp(c[f'vx_z{z}'], c[f'vy_z{z}'])[mask].mean() for z in range(nz)]
    # Spread of the layer velocities: nonzero when layers slide past each other
    spread = float(np.max(layers_par) - np.min(layers_par)) if nz > 1 else 0.0

    print(f"{mask.sum()} rows at step >= {step_min} (drive angle {angle:g} deg)")
    print(f"  <F_par> = {F_mean:.6g}   V_par = {V_par:.6g}   V_perp = {V_perp:.6g}")
    print(f"  per layer V_par: {np.round(layers_par, 6)}  (spread {spread:.3g})")

    out_prefix = args.out_prefix or os.path.join(os.path.dirname(args.path) or '.', 'velocity')
    np.savetxt(out_prefix + '_mean.dat', [[V_par, V_perp, F_mean, spread, step_min]],
               header='V_par  V_perp  F_par_mean  layer_spread  step_min', comments='')
    print(f"Saved {out_prefix}_mean.dat")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    t = c['t']
    axes[0].plot(t, F_par, color='0.6', lw=1, label='drive F(t)')
    axes[0].plot(t, v_par, lw=1, label='lattice')
    if nz > 1:
        for z in range(nz):
            axes[0].plot(t, par(c[f'vx_z{z}'], c[f'vy_z{z}']), lw=0.5, alpha=0.6, label=f'layer {z}')
    axes[0].axvline(t[mask][0], color='k', ls=':', lw=1)
    axes[0].set_ylabel('velocity parallel to the drive')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title(f'V_par = {V_par:.4g}, V_perp = {V_perp:.3g} (average after the dotted line)')
    axes[1].plot(t, v_perp, lw=1)
    axes[1].axvline(t[mask][0], color='k', ls=':', lw=1)
    axes[1].set_ylabel('velocity transverse to the drive')
    axes[1].set_xlabel('t')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_prefix + '.png', dpi=150)
    print(f"Saved {out_prefix}.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
