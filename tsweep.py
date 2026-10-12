import os
import re
import sys
import glob
import argparse
import subprocess

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def broadcast(values, n, name):
    """A --flag that takes one value (applied to every T) or one per T."""
    if len(values) == 1:
        return list(values) * n
    if len(values) != n:
        print(f"Error: --{name} must be given once (applied to every T) or once per T "
              f"({n} values); got {len(values)}.")
        sys.exit(1)
    return list(values)


def run(cmd, cwd):
    print(f"$ {' '.join(cmd)}   (in {cwd})")
    subprocess.run(cmd, cwd=cwd, check=True)


def find_latest_config(run_dir):
    """The config_step_<N>.dat with the highest N in run_dir, or None if there
    isn't one -- i.e. the configuration a run left off at, for chaining into
    the next T of a hysteresis sweep."""
    best_step, best_path = -1, None
    for path in glob.glob(os.path.join(run_dir, 'config_step_*.dat')):
        m = re.match(r'config_step_(\d+)\.dat$', os.path.basename(path))
        if m and int(m.group(1)) > best_step:
            best_step, best_path = int(m.group(1)), path
    return best_path


def load_scalar(path, col=0):
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=1)
    return float(data[col]) if data.ndim == 1 else float(data[0, col])


def load_defect_fraction(path):
    """Total defect fraction across all layers, from a disclinations.dat
    (columns: z, N, n_defects, defect_fraction, n_5fold, n_7fold, n_other)."""
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    return float(data[:, 2].sum() / data[:, 1].sum())


def load_sq_max(path):
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=1)
    return float(np.nanmax(data[:, 1]))


def load_mean_u2(path):
    """mean_u2 from a roughness.dat (columns: mean_u2, n_samples)."""
    return load_scalar(path, col=0)


def load_velocity_mean(path):
    """(V_par, V_perp) from a velocity_mean.dat written by velocity.py, or None."""
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    return float(data[0, 0]), float(data[0, 1])


# Parameters a sweep can run over, besides the temperature: name -> vortex_sim flag
SWEEPABLE = {'T': '--T', 'Fdc': '--Fdc', 'Fac': '--Fac', 'freq': '--freq'}


def plot_summary(Ts, values, ylabel, title, out_path, logy=False, xlabel='T'):
    # A log axis can't show D <= 0 (a solid's fit noise), so leave those out
    pairs = [(T, v) for T, v in zip(Ts, values) if v is not None and (v > 0 or not logy)]
    if not pairs:
        print(f"Skipping {out_path}: no data.")
        return
    xs, ys = zip(*pairs)
    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, ys, marker='o')
    if logy:
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both' if logy else 'major')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run vortex_sim plus the full analysis pipeline (analyze.py, msd.py, '
                     'disclinations.py, roughness.py, and velocity.py for driven runs) across '
                     'a temperature sweep -- or, with --sweep-param, a sweep of the drive '
                     'force or frequency at fixed T -- and collect a scalar summary (D, '
                     'defect fraction, S(q) peak, line roughness, and the mean velocity when '
                     'driven) vs the swept parameter.')
    parser.add_argument('temperatures', type=float, nargs='+', metavar='values',
                        help='values of the swept parameter (T unless --sweep-param says '
                             'otherwise), e.g. 0.005 0.01 0.015 0.02 0.03')
    parser.add_argument('--sweep-param', choices=list(SWEEPABLE), default='T',
                        help='parameter the positional values set (default T). With Fdc, Fac '
                             'or freq, the temperature is fixed by --T and the run folders are '
                             'named after the parameter (e.g. runs/Fdc_0.5)')
    parser.add_argument('--T', type=float, default=0.01,
                        help='fixed temperature when --sweep-param is not T (default 0.01)')
    parser.add_argument('--out-dir', default='runs',
                        help='parent directory for per-T run folders (default: runs)')
    parser.add_argument('--vortex-sim', default='./vortex_sim', help='path to the vortex_sim binary')
    parser.add_argument('--nx', type=int, default=30)
    parser.add_argument('--ny', type=int, default=30)
    parser.add_argument('--nz', type=int, default=4)
    parser.add_argument('--a0', type=float, default=1.0, help='lattice constant (default 1.0)')
    parser.add_argument('--k', type=float, default=0.5,
                        help='inter-layer spring constant (default 0.5)')
    parser.add_argument('--dt', type=float, default=0.01, help='timestep (default 0.01)')
    parser.add_argument('--cutoff', type=float, default=3.0,
                        help='interaction cutoff radius (default 3.0)')
    parser.add_argument('--skin', type=float, default=0.5, help='Verlet skin width (default 0.5)')
    parser.add_argument('--seed', type=int, default=1234567, help='RNG seed (default 1234567)')
    parser.add_argument('--print-interval', type=int, default=15)
    parser.add_argument('--np', type=float, default=0.0,
                        help='pinning centers per unit area, per layer (default 0 = no disorder)')
    parser.add_argument('--Ap', type=float, default=0.0,
                        help='pin strength, U = -Ap exp(-r^2/rp^2) (default 0 = no disorder)')
    parser.add_argument('--rp', type=float, default=0.1, help='pin range (default 0.1)')
    parser.add_argument('--disorder-seed', type=int, default=1,
                        help='seed of the disorder realization, kept fixed across every T (default 1)')
    parser.add_argument('--correlated-disorder', action='store_true',
                        help='same pins in every layer (columnar) instead of independent per layer')
    parser.add_argument('--Fdc', type=float, default=0.0,
                        help='DC drive force (default 0 = no drive)')
    parser.add_argument('--Fac', type=float, default=0.0,
                        help='AC drive amplitude, F(t) = [Fdc + Fac sin(2 pi freq t)] (default 0)')
    parser.add_argument('--freq', type=float, default=0.0, help='AC drive frequency (default 0)')
    parser.add_argument('--drive-angle', type=float, default=0.0,
                        help='drive direction in degrees from x (default 0)')
    parser.add_argument('--velocity-interval', type=int, default=None,
                        help='steps between velocity.dat rows (default: --print-interval)')
    parser.add_argument('--steps', type=int, nargs='+', default=[3000],
                        help='steps for this run; one value for every T, or one per T '
                             '(default 3000). Equilibration time is very T-dependent -- '
                             'especially near the melting transition, where it can be much '
                             'longer -- so give temperatures close to a suspected transition '
                             'more steps than ones far from it')
    parser.add_argument('--step-min', type=int, nargs='+', default=None,
                        help='equilibration cutoff in steps: only average snapshots at or '
                             'past this. One value for every T, or one per T (default: half '
                             'of that T\'s own --steps). Override per-T when some temperatures '
                             'need longer to equilibrate than others')
    parser.add_argument('--stride', type=int, nargs='+', default=[10],
                        help='config subsampling stride for analyze.py/disclinations.py '
                             '(S(k) is the expensive part of the pipeline); one value for '
                             'every T, or one per T (default 10)')
    parser.add_argument('--msd-window', type=int, default=100,
                        help='msd.py --window, in snapshots (default 100)')
    parser.add_argument('--msd-origin-spacing', type=int, default=10,
                        help='msd.py --origin-spacing, in snapshots (default 10)')
    parser.add_argument('--hysteresis', action='store_true',
                        help='start each T (or swept value, after the first) from the PREVIOUS '
                             'one\'s final configuration -- via vortex_sim\'s --restart/--start-step 0 -- '
                             'instead of always starting from the pristine ordered lattice. '
                             'Give a --temperatures list that ramps up and back down (e.g. '
                             '0.01 0.02 0.03 0.04 0.03 0.02 0.01) to trace out a heating/'
                             'cooling hysteresis loop; the first T is still a fresh ordered '
                             'start. Run folders are named by sweep position (e.g. '
                             'runs/000_T_0.01, runs/001_T_0.02, ...) so a T value revisited '
                             'on the way back down gets its own folder instead of colliding '
                             'with the one from the way up. Implies --skip-sim is NOT used '
                             'together with a from-scratch first T -- combine with --skip-sim '
                             'only to redo analysis on an already-run hysteresis sweep')
    parser.add_argument('--skip-sim', action='store_true',
                        help='skip step 1 (running vortex_sim) and just redo the analysis on '
                             'whatever config_step_*.dat already sits in each run folder -- '
                             'useful for retuning analysis parameters without repaying the '
                             'simulation cost')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='also skip steps 2-5 (analyze.py/msd.py/disclinations.py/'
                             'roughness.py) and just re-read each run folder\'s already-computed '
                             'msd_diffusion.dat/equil_disclinations.dat/equil_sq.dat/'
                             'equil_roughness.dat to rebuild summary.dat and the summary plots -- '
                             'for stitching per-T runs that already have their own full analysis '
                             '(e.g. separate cluster jobs) into one combined sweep, without '
                             'repaying the analysis cost a second time')
    args = parser.parse_args()

    n = len(args.temperatures)
    steps_list = broadcast(args.steps, n, 'steps')
    stride_list = broadcast(args.stride, n, 'stride')
    step_min_list = (broadcast(args.step_min, n, 'step-min') if args.step_min is not None
                      else [s // 2 for s in steps_list])

    vortex_sim = os.path.abspath(args.vortex_sim)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)

    name = args.sweep_param
    summary = []
    prev_run_dir = None
    for i, (value, steps, step_min, stride) in enumerate(
            zip(args.temperatures, steps_list, step_min_list, stride_list)):
        # Everything vortex_sim gets that the sweep can change
        point = {'T': args.T, 'Fdc': args.Fdc, 'Fac': args.Fac, 'freq': args.freq}
        point[name] = value
        T = point['T']
        if args.hysteresis:
            run_dir = os.path.join(args.out_dir, f"{i:03d}_{name}_{value}")
        else:
            run_dir = os.path.join(args.out_dir, f"{name}_{value}")
        os.makedirs(run_dir, exist_ok=True)

        restart_file = None
        if args.hysteresis and i > 0 and not args.skip_sim:
            restart_file = find_latest_config(prev_run_dir)
            if restart_file is None:
                print(f"Error: --hysteresis needs a config_step_*.dat in the previous run "
                      f"folder '{prev_run_dir}' to restart {name}={value} from, but none was found "
                      f"(did that run finish, and was it run without --skip-sim?).")
                sys.exit(1)

        tag = f", restarting from {restart_file}" if restart_file else ""
        fixed = f", T={T}" if name != 'T' else ""
        print(f"\n=== {name}={value}{fixed}  (steps={steps}, step_min={step_min}, stride={stride}){tag} ===")

        if not args.skip_sim:
            cmd = [vortex_sim, '--nx', str(args.nx), '--ny', str(args.ny), '--nz', str(args.nz),
                   '--a0', str(args.a0), '--k', str(args.k), '--dt', str(args.dt),
                   '--cutoff', str(args.cutoff), '--skin', str(args.skin), '--seed', str(args.seed),
                   '--T', str(T), '--steps', str(steps),
                   '--print-interval', str(args.print_interval)]
            # Disorder flags only when asked for, so runs without disorder keep
            # the exact command line older vortex_sim builds accept.
            if args.np != 0 or args.Ap != 0:
                cmd += ['--np', str(args.np), '--Ap', str(args.Ap), '--rp', str(args.rp),
                        '--disorder-seed', str(args.disorder_seed)]
                if args.correlated_disorder:
                    cmd += ['--correlated-disorder']
            # Same for the drive
            if point['Fdc'] != 0 or point['Fac'] != 0:
                cmd += ['--Fdc', str(point['Fdc']), '--Fac', str(point['Fac']),
                        '--freq', str(point['freq']), '--drive-angle', str(args.drive_angle)]
                if args.velocity_interval is not None:
                    cmd += ['--velocity-interval', str(args.velocity_interval)]
            if restart_file is not None:
                cmd += ['--restart', os.path.abspath(restart_file), '--start-step', '0']
            run(cmd, cwd=run_dir)

        if not args.skip_analysis:
            run(['python3', os.path.join(scripts_dir, 'analyze.py'), 'config_step_*.dat',
                 '--step-min', str(step_min), '--stride', str(stride), '--out-prefix', 'equil'],
                cwd=run_dir)

            t0_min_snapshots = step_min // args.print_interval
            run(['python3', os.path.join(scripts_dir, 'msd.py'),
                 '--window', str(args.msd_window), '--origin-spacing', str(args.msd_origin_spacing),
                 '--t0-min', str(t0_min_snapshots)],
                cwd=run_dir)

            run(['python3', os.path.join(scripts_dir, 'disclinations.py'), 'config_step_*.dat',
                 '--step-min', str(step_min), '--stride', str(stride), '--out-prefix', 'equil'],
                cwd=run_dir)

            run(['python3', os.path.join(scripts_dir, 'roughness.py'), 'config_step_*.dat',
                 '--step-min', str(step_min), '--stride', str(stride), '--out-prefix', 'equil'],
                cwd=run_dir)

            if os.path.exists(os.path.join(run_dir, 'velocity.dat')):
                run(['python3', os.path.join(scripts_dir, 'velocity.py'),
                     '--step-min', str(step_min)], cwd=run_dir)

        D = load_scalar(os.path.join(run_dir, 'msd_diffusion.dat'))
        defect_frac = load_defect_fraction(os.path.join(run_dir, 'equil_disclinations.dat'))
        sq_max = load_sq_max(os.path.join(run_dir, 'equil_sq.dat'))
        mean_u2 = load_mean_u2(os.path.join(run_dir, 'equil_roughness.dat'))
        vel = load_velocity_mean(os.path.join(run_dir, 'velocity_mean.dat'))
        summary.append((value, D, defect_frac, sq_max, mean_u2, vel))
        vel_msg = f"  V_par={vel[0]}  V_perp={vel[1]}" if vel else ""
        print(f"--- {name}={value}: D={D}  defect_fraction={defect_frac}  S(q)_max={sq_max}  "
              f"mean_u2={mean_u2}{vel_msg}")

        prev_run_dir = run_dir

    # First column is the swept parameter; V_par/V_perp only when some run was
    # driven, so undriven temperature sweeps write the same summary.dat as before.
    nan = lambda v: v if v is not None else np.nan
    driven = any(r[5] is not None for r in summary)
    rows = []
    for x, D, f, s, u, vel in summary:
        row = [x, nan(D), nan(f), nan(s), nan(u)]
        if driven:
            row += list(vel) if vel is not None else [np.nan, np.nan]
        rows.append(row)
    header = f'{name}  D  defect_fraction  Sq_max  mean_u2' + ('  V_par  V_perp' if driven else '')
    summary_path = os.path.join(args.out_dir, 'summary.dat')
    np.savetxt(summary_path, rows, header=header, comments='')
    print(f"\nSaved {summary_path}")

    xs = [r[0] for r in summary]
    out = lambda q: os.path.join(args.out_dir, f'summary_{q}_vs_{name}.png')
    plot_summary(xs, [r[1] for r in summary], 'D', f'Diffusion constant vs {name}',
                 out('D'), logy=True, xlabel=name)
    plot_summary(xs, [r[2] for r in summary], 'defect fraction', f'Disclination fraction vs {name}',
                 out('defects'), xlabel=name)
    plot_summary(xs, [r[3] for r in summary], 'max S(q)', f'S(q) peak height vs {name}',
                 out('Sqmax'), logy=True, xlabel=name)
    plot_summary(xs, [r[4] for r in summary], r'$\langle u^2 \rangle$', f'Vortex line roughness vs {name}',
                 out('roughness'), logy=True, xlabel=name)
    if driven:
        plot_summary(xs, [r[5][0] if r[5] else None for r in summary], 'V parallel to the drive',
                     f'Mean velocity vs {name}', out('V'), xlabel=name)


if __name__ == "__main__":
    main()
