#!/usr/bin/env python3

from MarkovModels import common
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
from MarkovModels.MarkovModel import MarkovModel
from MarkovModels.BeattieModel import BeattieModel
from quality_control.leak_fit import fit_leak_lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import regex as re
import itertools
import uuid
from matplotlib import rc
import multiprocessing

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

from matplotlib.gridspec import GridSpec

Erev = common.calculate_reversal_potential(T=298.15, K_in=120, K_out=5)

pool_kws = {'maxtasksperchild': 1}

subtracted_trace_dirname = "subtracted_traces"


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--percentage_to_remove', default=0, type=float)
    parser.add_argument('--selection_file', default=None, type=str)
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--ramp_start', type=float, default=300)
    parser.add_argument('--ramp_end', type=float, default=900)
    parser.add_argument('--figsize', type=int, nargs=2, default=[8, 12])
    parser.add_argument('--output_all', action='store_true')
    parser.add_argument('--no_plot', action='store_true')

    global args
    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    if len(args.protocols) == 0:
        args.protocols = get_protocol_list(args.data_directory)

    global output
    output = common.setup_output_directory(args.output, f"subtract_leak_{experiment_name}")

    if not os.path.exists(output):
        os.makedirs(output)

    global reversal_plot_dir
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

    print(args.wells, args.protocols)

    if not os.path.exists(os.path.join(output, subtracted_trace_dirname)):
        os.makedirs(os.path.join(output, subtracted_trace_dirname))

    tasks = [(well, protocol) for well in args.wells for protocol in args.protocols]

    pool_size = min(len(tasks), args.cpus)
    with multiprocessing.Pool(pool_size, **pool_kws) as pool:
        res = pool.starmap(subtract_leak, tasks)

    df = pd.concat(res, ignore_index=True)

    df['passed QC7'] = False

    for well in df.well.unique():
        passed_QC7 = QC7(well)
        df.loc[df.well == well, 'passed QC7'] = passed_QC7

    if args.selection_file:
        df['selected'] = [well in selected_wells for well in df['well']]

    E_Kr_spread = compute_E_Kr_spread(df)
    df['E_Kr_spread'] = [E_Kr_spread[well] if well in E_Kr_spread else np.nan for well in df.well]
    df.to_csv(os.path.join(output, "subtraction_qc.csv"))

    df['QC E_Kr_spread'] = np.abs(df.E_Kr_spread.values) <= 10

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
            elif not row['passed QC7']:
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


def subtract_leak(well, protocol):
    if not args.no_plot:
        fig = plt.figure(figsize=args.figsize, clear=True, constrained_layout=True)
        reversal_fig = plt.figure(figsize=args.figsize, constrained_layout=True)
        reversal_ax = reversal_fig.subplots()

    nsweeps = 1
    sweep2_fname = f"{experiment_name}-{protocol}-{well}-before-sweep2.csv"
    if os.path.exists(os.path.join(args.data_directory, sweep2_fname)):
        nsweeps = 2

    if not args.no_plot:
        protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, \
            long_protocol_ax = setup_subtraction_grid(fig, nsweeps)

    protocol_func, _ = common.get_protocol_from_csv(protocol)
    observation_times = pd.read_csv(os.path.join(
        args.data_directory, f"{experiment_name}-{protocol}-times.csv")).values.flatten() * 1e3
    protocol_voltages = np.array([protocol_func(t) for t in observation_times])
    dt = observation_times[1] - observation_times[0]

    df = []
    for sweep in range(1, nsweeps + 1):
        before_filename = f"{experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
        after_filename = f"{experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"

        tracename = 'subtracted'

        try:
            before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
        except FileNotFoundError as exc:
            before_trace = None
            print(str(exc))

        try:
            after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()
        except FileNotFoundError as exc:
            after_trace = None
            print(str(exc))

        if before_trace is not None:
            g_leak_before, E_leak_before, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, before_trace, dt=dt,
                percentage_to_remove=args.percentage_to_remove,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end
            )
            n = len(x)
            msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()
            sd = np.sqrt(msres * (1 / n + (40 - x.mean())**2 / ((x**2).sum())))
            before_sd = sd

        else:
            g_leak_before = np.nan
            E_leak_before = np.nan

        if after_trace is not None:
            g_leak_after, E_leak_after, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, after_trace, dt=dt,
                percentage_to_remove=args.percentage_to_remove,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end
            )
            n = len(x)
            msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()
            sd = np.sqrt(msres * (1 / n + (40 - x.mean())**2 / ((x**2).sum())))
        else:
            g_leak_after = np.nan
            E_leak_after = np.nan

        if before_trace is not None:
            before_corrected = before_trace - (g_leak_before * (protocol_voltages - E_leak_before))
        if after_trace is not None:
            after_corrected = after_trace - (g_leak_after * (protocol_voltages - E_leak_after))

        if before_trace is not None and after_trace is not None:
            subtracted_trace = before_corrected - after_corrected
        else:
            subtracted_trace = np.array([np.nan])

        if args.no_plot:
            reversal_ax = None

        if np.all(np.isfinite(subtracted_trace)):
            fitted_E_rev = common.infer_reversal_potential(protocol,
                                                           subtracted_trace,
                                                           observation_times,
                                                           output_path=os.path.join(reversal_plot_dir,
                                                                                    f"reversal_potential_{protocol}_{well}"),
                                                           ax=reversal_ax,
                                                           plot=not args.no_plot)

        else:
            fitted_E_rev = np.nan

        if not args.no_plot:
            reversal_ax.cla()
            plt.close(reversal_fig)

        passed1 = False

        if before_trace is not None and after_trace is not None:
            subtracted_trace_df = pd.DataFrame(np.column_stack(
                (observation_times, subtracted_trace)), columns=('time', 'current'))

            fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
            subtracted_trace_df.to_csv(os.path.join(output, subtracted_trace_dirname, fname))

            subtracted_trace_df['time'].to_csv(os.path.join(
                output, subtracted_trace_dirname, f"{experiment_name}-{protocol}-times.csv"))

            # Check that the current isn't negative on the first step after the leak ramp
            first_step = [(i, v) for i, v in enumerate(protocol_voltages) if v > 30]
            lst = []
            for i, (j, voltage) in enumerate(first_step):
                if j - i > first_step[0][0]:
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
                before_axs[ax_col].set_ylabel('current / nA')

                after_axs[ax_col].plot(observation_times, after_trace)
                after_axs[ax_col].plot(observation_times, g_leak_after * (protocol_voltages - E_leak_after), label=f"leak current E={E_leak_after:.2f}mV, g={g_leak_after:.2f}")
                after_axs[ax_col].set_title(f"After drug raw trace sweep{sweep}")
                after_axs[ax_col].legend(loc=1)
                after_axs[ax_col].set_ylabel('current / nA')

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
        try:
            output_path = os.path.join(args.output_path,
                                       f"{protocol}_{well}_{tracename}_infer_reversal_potential.png")
            Erev = common.infer_reversal_potential(protocol, subtracted_trace,
                                                   observation_times, plot=True,
                                                   output_path=output_path)
        except Exception:
            Erev = -np.inf

        if Erev > -50 or Erev < -100:
            print(f"{protocol}, {well} \tpassed QC.Erev")
            passed_Erev = True
        else:
            print(f"{protocol}, {well} \tfailed QC.Erev")
            passed_Erev = False

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

        fig.savefig(os.path.join(output, f"{well}_{protocol}_traces_from_leak_subtraction"))
        plt.close(fig)

    df = pd.DataFrame(df, columns=('protocol', 'well', 'sweep', 'before/after',
                                   'fitted_E_rev', 'passed QC6',
                                   'passed QC.Erev', 'R_leftover', 'pre-drug'
                                   ' leak conductance', 'post-drug leak'
                                   ' conductance', 'pre-drug leak reversal',
                                   'post-drug leak reversal'))

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

    rmsdc = max(np.mean([rmsd0_1, rmsd0_2]) * 2,
                np.mean([noise_1, noise_2]) * 6)

    rmsd = np.sqrt(np.mean((recording1 - recording2) ** 2))

    return rmsd > rmsdc or not (np.isfinite(rmsd) and np.isfinite(rmsdc))


if __name__ == "__main__":
    main()
