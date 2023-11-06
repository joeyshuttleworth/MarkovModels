#!/usr/bin/env python3

import argparse
import gc
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
from matplotlib.gridspec import GridSpec

from quality_control.leak_fit import fit_leak_lr
from markovmodels.utilities import setup_output_directory, calculate_reversal_potential
from markovmodels.voltage_protocols import get_ramp_protocol_from_csv
from markovmodels.fitting import infer_reversal_potential

# Agg was causing some memory leak issues
# matplotlib.use('Agg')
# rc('text', usetex=True)

Erev = calculate_reversal_potential(T=298.15, K_in=120, K_out=5)

pool_kws = {'maxtasksperchild': 1}

subtracted_trace_dirname = "subtracted_traces"


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--selection_file', type=str)
    parser.add_argument('--ignore_QC7', action='store_true')
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--ramp_start', type=float, default=300)
    parser.add_argument('--ramp_end', type=float, default=900)
    parser.add_argument('--figsize', type=int, nargs=2, default=[8, 12])
    parser.add_argument('--output_all', action='store_true')
    parser.add_argument('--no_plot', action='store_true')
    parser.add_argument('--Erev', type=float)

    global args
    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    if len(args.protocols) == 0:
        args.protocols = get_protocol_list(args.data_directory)

    global output
    output = setup_output_directory(args.output, f"subtract_leak_{experiment_name}")

    global leak_subtraction_plots_dir
    leak_subtraction_plots_dir = os.path.join(output, 'subtraction_plots')

    if not os.path.exists(leak_subtraction_plots_dir):
        os.makedirs(leak_subtraction_plots_dir)

    reversal_plot_dir = os.path.join(output, 'reversal_plots')
    if not os.path.exists(reversal_plot_dir):
        os.makedirs(reversal_plot_dir)

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    if args.selection_file is not None:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
        if not args.output_all:
            args.wells = [well for well in args.wells if well in selected_wells]
    else:
        selected_wells = args.wells

    print(args.wells, args.protocols)

    if not os.path.exists(os.path.join(output, subtracted_trace_dirname)):
        os.makedirs(os.path.join(output, subtracted_trace_dirname))

    tasks = [(well, protocol, args, output) for well in args.wells for protocol in args.protocols]

    pool_size = min(len(tasks), args.cpus)

    global scatter_plots_dir
    scatter_plots_dir = os.path.join(output, 'scatter_plots')
    if not os.path.exists(scatter_plots_dir):
        os.makedirs(scatter_plots_dir)

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(subtract_leak, tasks)

    df = pd.concat(res, ignore_index=True)
    qc_vals_df = pd.read_csv(os.path.join(args.data_directory, f"{experiment_name}_qc_estimates.csv"))
    qc_vals_df.to_csv(os.path.join(output, 'qc_vals.csv'))

    if not args.ignore_QC7:
        df['passed QC7'] = False
        with multiprocessing.Pool(pool_size, **pool_kws) as pool:
            QC7_res = pool.map(QC7, df.well.unique())

            for i, well in enumerate(df.well.unique()):
                passed_QC7 = QC7_res[i]
                df.loc[df.well == well, 'passed QC7'] = passed_QC7

    if args.selection_file:
        df['selected'] = [well in selected_wells for well in df['well']]
    else:
        df['selected'] = True

    E_Kr_spread = compute_E_Kr_spread(df)
    df['E_Kr_spread'] = [E_Kr_spread[well] if well in E_Kr_spread else np.nan for well in df.well]

    df['QC E_Kr_spread'] = np.abs(df.E_Kr_spread.values) <= 5
    df.to_csv(os.path.join(output, "subtraction_qc.csv"))

    # Find wells passing all traces
    passed_lst = []
    for well in np.unique(df['well'].values):
        failed = False
        for _, row in df[df.well == well].iterrows():
            if row['passed QC6'] is False:
                failed = True
                break
            elif not np.isfinite(row['fitted_E_rev']):
                failed = True
                break
            elif not row['QC E_Kr_spread']:
                failed = True
                break
            elif not row['passed QC.Erev']:
                failed = True
                break
            elif not args.ignore_QC7:
                if not row['passed QC7']:
                    failed = True
                    break
            elif args.selection_file:
                if well not in selected_wells:
                    failed = True
                    break
        if not failed:
            passed_lst.append(well)

    print(f"Wells with all successful traces: {passed_lst}")

    with open(os.path.join(output, "passed_wells.txt"), 'w') as fout:
        for well in passed_lst:
            fout.write(well)
            fout.write("\n")

    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        pool.map(overlay_first_last_staircases, df.well.unique())


def compute_E_Kr_spread(df):
    df.fitted_E_rev = df.fitted_E_rev.values.astype(np.float64)

    failed_to_infer = [well for well in df.well.unique() if not
                       np.all(np.isfinite(df[df.well == well]['fitted_E_rev'].values))]

    df = df[~df.well.isin(failed_to_infer)]
    pivot_df = df.pivot_table(index='well', columns='protocol', values='fitted_E_rev')
    pivot_df['E_Kr min'] = pivot_df.values.min(axis=1)
    pivot_df['E_Kr max'] = pivot_df.values.max(axis=1)
    pivot_df['E_Kr range'] = pivot_df['E_Kr max'] - pivot_df['E_Kr min']

    return dict(zip(pivot_df.index, pivot_df['E_Kr range'].values.flatten()))


def get_wells_list(input_dir):
    regex = re.compile(f"{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after")
    wells = []

    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(2)[1]
        if well not in wells:
            wells.append(well)
    return list(np.unique(wells))


def get_protocol_list(input_dir):
    regex = re.compile(f"{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after")
    protocols = []
    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(3)[0]
        if protocols not in protocols:
            protocols.append(well)
    return list(np.unique(protocols))


def setup_subtraction_grid(fig, nsweeps):
    # Use 5 x 2 grid
    gs = GridSpec(6, nsweeps, figure=fig)

    # plot protocol at the top
    protocol_axs = [fig.add_subplot(gs[0, i]) for i in range(nsweeps)]

    # Plot before drug traces
    before_axs = [fig.add_subplot(gs[1, i]) for i in range(nsweeps)]

    # Plot after traces
    after_axs = [fig.add_subplot(gs[2, i]) for i in range(nsweeps)]

    # Leak corrected traces
    corrected_axs = [fig.add_subplot(gs[3, i]) for i in range(nsweeps)]

    # Subtracted traces on one axis
    subtracted_ax = fig.add_subplot(gs[4, :])

    long_protocol_ax = fig.add_subplot(gs[5, :])

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def overlay_first_last_staircases(well):

    first_filename = f"{experiment_name}-staircaseramp1-{well}-sweep1.csv"
    final_filename = f"{experiment_name}-staircaseramp2-{well}-sweep2.csv"

    times_filename = f"{experiment_name}-staircaseramp1-times.csv"

    subtracted_traces_dir = os.path.join(output, 'subtracted_traces')

    try:
        times = pd.read_csv(os.path.join(subtracted_traces_dir, times_filename))['time'].values.flatten()
    except FileNotFoundError as exc:
        print(str(exc))
        return

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax1, ax2 = fig.subplots(2, 1, height_ratios=[3, 1])

    try:
        before_trace = pd.read_csv(os.path.join(subtracted_traces_dir, first_filename))['current'].values.flatten()
    except FileNotFoundError as exc:
        before_trace = np.full(times.shape, np.nan)
        print(str(exc))

    try:
        after_trace = pd.read_csv(os.path.join(subtracted_traces_dir, final_filename))['current'].values.flatten()
    except FileNotFoundError as exc:
        after_trace = np.full(times.shape, np.nan)
        print(str(exc))

    ax1.plot(times, before_trace, label='staircaseramp1 sweep-1')
    ax1.plot(times, after_trace, label='staircaseramp2 sweep-2')

    prot_func, _, _, = get_ramp_protocol_from_csv('staircaseramp1')
    voltages = np.array([prot_func(t) for t in times])
    ax2.plot(times, voltages)

    sub_dir = os.path.join(output, 'first_last_staircase_compare')

    try:
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

    except FileExistsError as exc:
        # Directory already exists or there's a file here
        print(str(exc))

    fig.savefig(os.path.join(sub_dir, f"{well}_overlaid.png"))
    plt.close(fig)


def subtract_leak(well, protocol, args, output_dir=None):

    if output_dir is None:
        output_dir = output
    leak_subtraction_plots_dir = os.path.join(output_dir, 'subtraction_plots')

    if not os.path.exists(leak_subtraction_plots_dir):
        os.makedirs(leak_subtraction_plots_dir)

    if not args.no_plot:
        fig = plt.figure(figsize=args.figsize, clear=True, constrained_layout=True)
        subtract_scatter_fig = plt.figure(figsize=args.figsize)
        axs = subtract_scatter_fig.subplots(2, 2)
        [[scatter_ax_before, window_ax_before], [scatter_ax_after, window_ax_after]] = axs

    nsweeps = 1
    sweep2_fname = f"{args.experiment_name}-{protocol}-{well}-before-sweep2.csv"
    if os.path.exists(os.path.join(args.data_directory, sweep2_fname)):
        nsweeps = 2

    if not args.no_plot:
        protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, \
            long_protocol_ax = setup_subtraction_grid(fig, nsweeps)

    protocol_func, _, _ = get_ramp_protocol_from_csv(protocol)
    observation_times = pd.read_csv(os.path.join(
        args.data_directory, f"{args.experiment_name}-{protocol}-times.csv")).to_numpy()[:, -1].flatten()*1e3
    protocol_voltages = np.array([protocol_func(t) for t in observation_times])
    dt = observation_times[1] - observation_times[0]

    print(f"dt={dt}")

    reversal_plot_dir = os.path.join(output_dir, 'reversal_plots')

    full_subtracted_trace_dir = os.path.join(output_dir, subtracted_trace_dirname)
    if not os.path.exists(full_subtracted_trace_dir):
        os.makedirs(full_subtracted_trace_dir)

    scatter_plots_dir = os.path.join(output_dir, 'scatter_plots')
    if not os.path.exists(scatter_plots_dir):
        os.makedirs(scatter_plots_dir)

    df = []
    for sweep in range(1, nsweeps + 1):
        before_filename = f"{args.experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
        after_filename = f"{args.experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"

        indices_to_plot = [i for i, t in enumerate(observation_times) if t
                           <= args.ramp_end * 2]

        tracename = 'subtracted'

        try:
            before_trace_df = pd.read_csv(os.path.join(args.data_directory, before_filename))
            before_trace = before_trace_df[before_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)
        except FileNotFoundError as exc:
            before_trace = None
            print(str(exc))

        try:
            after_trace_df = pd.read_csv(os.path.join(args.data_directory, after_filename))
            after_trace = after_trace_df[after_trace_df.columns[-1]].to_numpy().flatten().astype(np.float64)
        except FileNotFoundError as exc:
            after_trace = None
            print(str(exc))

        if before_trace is not None and np.all(np.isfinite(before_trace)):
            g_leak_before, E_leak_before, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, before_trace, dt=dt,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end
            )

            n = len(x)
            # msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()

            infer_reversal_potential(protocol, before_trace,
                                     observation_times, plot=True,
                                     output_path=os.path.join(reversal_plot_dir,
                                                              f"{well}_{protocol}_sweep{sweep}_before"),
                                     known_Erev=args.Erev
                                     )

            if not args.no_plot:
                scatter_ax_before.scatter(x, y, marker='s', color='grey', s=2)
                ypred = (x - E_leak_before) * g_leak_before
                scatter_ax_before.plot(x, ypred, color='red')

                window_ax_before.plot(observation_times[indices_to_plot],
                                      before_trace[indices_to_plot], alpha=.5, color='grey')

                window_ax_before.plot(observation_times[indices_to_plot],
                                      (protocol_voltages[indices_to_plot] - E_leak_before) * g_leak_before)

                window_ax_before.axvspan(args.ramp_start, args.ramp_end,
                                         color='grey', alpha=.5)
        else:
            g_leak_before = np.nan
            E_leak_before = np.nan

        if after_trace is not None and np.all(np.isfinite(after_trace)):
            g_leak_after, E_leak_after, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, after_trace, dt=dt,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end
            )
            n = len(x)
            # msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()

            infer_reversal_potential(protocol, before_trace,
                                     observation_times, plot=True,
                                     output_path=os.path.join(reversal_plot_dir,
                                                              f"{well}_{protocol}_sweep{sweep}_after"))
            if not args.no_plot:
                scatter_ax_after.scatter(x, y, color='grey', s=2, marker='s')
                ypred = (x - E_leak_after) * g_leak_after
                scatter_ax_after.plot(x, ypred, color='red')
                window_ax_after.plot(observation_times[indices_to_plot],
                                     after_trace[indices_to_plot], alpha=.5, color='grey')

                window_ax_after.plot(observation_times[indices_to_plot],
                                     (protocol_voltages[indices_to_plot] - E_leak_after) * g_leak_after)

                window_ax_after.axvspan(args.ramp_start, args.ramp_end,
                                        color='grey', alpha=.25)

        else:
            g_leak_after = np.nan
            E_leak_after = np.nan

        if not args.no_plot:
            window_ax_before.set_xlabel('time / ms')
            window_ax_after.set_xlabel('time / ms')

            window_ax_before.set_ylabel(r'I_Kr / pA')
            window_ax_after.set_ylabel(r'I_Kr / pA')

            scatter_ax_before.set_xlabel(r'$V$ / mV')
            scatter_ax_after.set_xlabel(r'$V$ / mV')

            scatter_ax_after.set_xlim([-125, -75])

            subtract_scatter_fig.savefig(os.path.join(scatter_plots_dir,
                                                      f"{well}_{protocol}_sweep{sweep}_subtraction_scatter"))

            window_ax_after.cla()
            window_ax_before.cla()
            scatter_ax_after.cla()
            scatter_ax_before.cla()

        if before_trace is not None:
            before_corrected = before_trace - (g_leak_before * (protocol_voltages - E_leak_before))
            infer_reversal_potential(protocol, before_corrected,
                                     observation_times,
                                     output_path=os.path.join(reversal_plot_dir,
                                                              f"{protocol}_{well}_before_drug_leak_corrected"),
                                     plot=not args.no_plot)

        if after_trace is not None:
            after_corrected = after_trace - (g_leak_after * (protocol_voltages - E_leak_after))
            infer_reversal_potential(protocol, after_corrected,
                                     observation_times,
                                     output_path=os.path.join(reversal_plot_dir,
                                                              f"{protocol}_{well}_after_drug_leak_corrected"),
                                     plot=not args.no_plot)

        if before_trace is not None and after_trace is not None:
            subtracted_trace = before_corrected - after_corrected
        else:
            subtracted_trace = np.array([np.nan])

        if np.all(np.isfinite(subtracted_trace)):
            fitted_E_rev = infer_reversal_potential(protocol,
                                                    subtracted_trace,
                                                    observation_times,
                                                    known_Erev=args.Erev,
                                                    output_path=os.path.join(reversal_plot_dir,
                                                                             f"{protocol}_{well}_subtracted"),
                                                    plot=not args.no_plot)

        else:
            fitted_E_rev = np.nan

        passed1 = False

        if before_trace is not None and after_trace is not None:
            subtracted_trace_df = pd.DataFrame(np.column_stack(
                (observation_times, subtracted_trace)), columns=('time', 'current'))

            fname = f"{args.experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
            subtracted_trace_df.to_csv(os.path.join(full_subtracted_trace_dir, fname))

            subtracted_trace_df['time'].to_csv(os.path.join(
                full_subtracted_trace_dir, f"{args.experiment_name}-{protocol}-times.csv"))

            # Check that the current isn't negative on the first step after the leak ramp
            first_step = [(i, v) for i, v in enumerate(protocol_voltages) if v > 30]
            lst = []
            for i, (j, voltage) in enumerate(first_step):
                if j - i > first_step[0][0]:
                    # Moved past the first step
                    break
                lst.append(j)
                # Ignore first few timesteps
            first_step_indices = lst[10:-10]

            ax_col = sweep - 1

            if not args.no_plot:
                protocol_axs[ax_col].plot(observation_times, protocol_voltages)
                protocol_axs[ax_col].set_title("Voltage protocol")
                protocol_axs[ax_col].set_ylabel("Voltage /mV")
                protocol_axs[ax_col].set_xlabel("time /ms")

                before_axs[ax_col].plot(observation_times, before_trace)
                before_axs[ax_col].plot(observation_times, g_leak_before * (protocol_voltages - E_leak_before), label=f"leak current E={E_leak_before:.2f}mV, g={g_leak_before:.2f}")
                before_axs[ax_col].set_title(f"Before drug raw trace sweep{sweep}")
                before_axs[ax_col].legend(loc=1)
                before_axs[ax_col].set_ylabel('current / pA')

                after_axs[ax_col].plot(observation_times, after_trace)
                after_axs[ax_col].plot(observation_times, g_leak_after * (protocol_voltages - E_leak_after), label=f"leak current E={E_leak_after:.2f}mV, g={g_leak_after:.2f}")
                after_axs[ax_col].set_title(f"After drug raw trace sweep{sweep}")
                after_axs[ax_col].legend(loc=1)
                after_axs[ax_col].set_ylabel('current / pA')

                corrected_axs[ax_col].plot(observation_times, before_corrected, label='pre-drug')
                corrected_axs[ax_col].plot(observation_times, after_corrected, label='post-drug')

                corrected_axs[ax_col].legend(loc=1)
                corrected_axs[ax_col].set_title(f"leak corrected traces sweep{sweep}")

                subtracted_ax.plot(observation_times, subtracted_trace, label=f"sweep{sweep}")

            tracename, trace = ('subtracted', subtracted_trace)
            estimated_noise = trace[0:200].std()
            trace = trace[first_step_indices]
            n = len(trace)
            if trace.mean() > -2*estimated_noise:
                print(f"{protocol} {well} {tracename} \tpassed QC6")
                passed1 = True
            else:
                print(f"{protocol}, {well}, {tracename} \tfailed QC6")
                passed1 = False

        # Can we infer reversal potential from subtracted trace
        Erev = infer_reversal_potential(protocol, subtracted_trace,
                                        observation_times,
                                        plot=False)

        if Erev > -50 or Erev < -120:
            print(f"{protocol}, {well} \tpassed QC.Erev")
            passed_Erev = False
        else:
            print(f"{protocol}, {well} \tfailed QC.Erev")
            passed_Erev = True

        if after_trace is not None:
            R_leftover = np.sqrt(np.sum(after_corrected**2)/(np.sum(before_corrected**2)))
        else:
            R_leftover = np.nan

        df.append((protocol, well, sweep, tracename, fitted_E_rev,
                   passed1, passed_Erev, R_leftover,
                   g_leak_before, g_leak_after, E_leak_before,
                   E_leak_after))

    if not args.no_plot:
        subtracted_ax.set_xlabel('time / ms')
        subtracted_ax.set_ylabel('current / pA')
        subtracted_ax.set_title('subtracted traces')
        subtracted_ax.legend(loc=1)

        long_protocol_ax.plot(observation_times, protocol_voltages)
        long_protocol_ax.set_title("Voltage protocol")
        long_protocol_ax.set_ylabel("Voltage /mV")
        long_protocol_ax.set_xlabel("time /ms")

        fig.savefig(os.path.join(leak_subtraction_plots_dir, f"{well}_{protocol}_traces_from_leak_subtraction"))
        plt.close(fig)
        plt.close(subtract_scatter_fig)

    df = pd.DataFrame(df, columns=('protocol', 'well', 'sweep', 'before/after',
                                   'fitted_E_rev', 'passed QC6',
                                   'passed QC.Erev', 'R_leftover', 'pre-drug'
                                   ' leak conductance', 'post-drug leak'
                                   ' conductance', 'pre-drug leak reversal',
                                   'post-drug leak reversal'))

    plt.close('all')
    gc.collect()

    return df


def QC7(well):
    try:
        first_staircase_df = pd.read_csv(os.path.join(output, 'subtracted_traces',
                                                      f"{args.experiment_name}-staircaseramp1-{well}-sweep1.csv"))
        last_staircase_df = pd.read_csv(os.path.join(output, 'subtracted_traces',
                                                     f"{args.experiment_name}-staircaseramp2-{well}-sweep2.csv"))

    except FileNotFoundError as e:
        print(str(e))
        return False

    recording1 = first_staircase_df['current'].values
    recording2 = last_staircase_df['current'].values

    noise_1 = np.std(first_staircase_df['current'].values[:200])
    noise_2 = np.std(last_staircase_df['current'].values[:200])
    rmsd0_1 = np.sqrt(np.mean((recording1) ** 2))
    rmsd0_2 = np.sqrt(np.mean((recording2) ** 2))

    rmsdc = max(np.mean([rmsd0_1, rmsd0_2]) * 0.2,
                np.mean([noise_1, noise_2]) * 6)

    rmsd = np.sqrt(np.mean((recording1 - recording2) ** 2))

    return rmsd < rmsdc and np.isfinite(rmsd) and np.isfinite(rmsdc)


if __name__ == "__main__":
    main()
