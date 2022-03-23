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

from matplotlib.gridspec import GridSpec

Erev = common.calculate_reversal_potential(T=298.15, K_in=120, K_out=5)

# Order in which the experiments were performed.
protocol_chrono_order = ['staircaseramp1',
                         'sis',
                         'rtovmaxdiff',
                         'rvotmaxdiff',
                         'spacefill10',
                         'spacefill19',
                         'spacefill26',
                         'longap',
                         'hhbrute3gstep',
                         'hhsobol3step',
                         'wangbrute3gstep',
                         'wangsobol3step',
                         'staircaseramp2']


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


def setup_subtraction_grid(fig):
    # Use 5 x 2 grid
    gs = GridSpec(6, 2, figure=fig)

    # plot protocol at the top
    protocol_axs = [fig.add_subplot(gs[0, i]) for i in range(2)]

    # Plot before drug traces
    before_axs = [fig.add_subplot(gs[1, i]) for i in range(2)]

    # Plot after traces
    after_axs = [fig.add_subplot(gs[2, i]) for i in range(2)]

    # Leak corrected traces
    corrected_axs = [fig.add_subplot(gs[3, i]) for i in range(2)]

    # Subtracted traces on one axis
    subtracted_ax = fig.add_subplot(gs[4, :])

    long_protocol_ax = fig.add_subplot(gs[5, :])

    return protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--percentage_to_remove', default=0, type=float)
    parser.add_argument('-e', '--extra_points', nargs=2, default=(0, 0), type=int)
    parser.add_argument('--selection_file', default=None, type=str)
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--ramp_start', type=float, default=300)
    parser.add_argument('--ramp_end', type=float, default=900)

    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    if len(args.protocols) == 0:
        args.protocols = get_protocol_list(args.data_directory)

    output = common.setup_output_directory(args.output, f"subtract_leak_{args.extra_points[0]}_{args.extra_points[1]}")

    if not os.path.exists(output):
        os.makedirs(output)

    reversal_plot_dir = os.path.join(output, 'reversal_plots')

    if not os.path.exists(reversal_plot_dir):
        os.makedirs(reversal_plot_dir)

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    if args.selection_file is not None:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
        args.wells = [well for well in args.wells if well in selected_wells]

    print(args.wells, args.protocols)

    df = []

    fig = plt.figure(figsize=(20, 16), clear=True)

    subtracted_trace_dirname = "subtracted_traces"

    if not os.path.exists(os.path.join(output, subtracted_trace_dirname)):
        os.makedirs(os.path.join(output, subtracted_trace_dirname))

    reversal_fig = plt.figure(figsize=(18, 16))
    reversal_ax  = reversal_fig.subplots()

    for well in args.wells:
        for protocol in args.protocols:

            fig.clf()
            protocol_axs, before_axs, after_axs, corrected_axs, subtracted_ax, long_protocol_ax = setup_subtraction_grid(fig)

            protocol_func, t_start, t_end, _ = common.get_protocol_from_csv(protocol)
            observation_times = pd.read_csv(os.path.join(
                args.data_directory, f"{experiment_name}-{protocol}-times.csv")).values.flatten() * 1e3
            protocol_voltages = np.array([protocol_func(t) for t in observation_times])
            dt = observation_times[1] - observation_times[0]

            extra_points = [int(val / dt) for val in args.extra_points]

            # Find first few steps where voltage is big
            if args.extra_points[1] > 0:
                extra_steps = np.array(list(itertools.islice(filter(lambda x:
                                                                    x[1] > 20, enumerate(protocol_voltages)),
                                                             extra_points[1])))[extra_points[0]:, 0].astype(int)
            else:
                extra_steps = []

            for sweep in (1, 2):
                before_filename = f"{experiment_name}-{protocol}-{well}-before-sweep{sweep}.csv"
                after_filename = f"{experiment_name}-{protocol}-{well}-after-sweep{sweep}.csv"

                try:
                    before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
                    after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()
                except FileNotFoundError as exc:
                    print(str(exc))
                    continue

                g_leak_before, E_leak_before, _, _, _, x, y = fit_leak_lr(
                    protocol_voltages, before_trace, dt=dt,
                    percentage_to_remove=args.percentage_to_remove,
                    extra_points=extra_steps,
                    ramp_start=args.ramp_start,
                    ramp_end=args.ramp_end
                )

                n = len(x)

                msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()
                sd = np.sqrt(msres * (1 / n + (x - x.mean())**2 / ((x**2).sum())))
                before_sd = sd[-1]

                g_leak_after, E_leak_after, _, _, _, x, y = fit_leak_lr(
                    protocol_voltages, after_trace, dt=dt,
                    percentage_to_remove=args.percentage_to_remove,
                    extra_points=extra_steps,
                    ramp_start=args.ramp_start,
                    ramp_end=args.ramp_end
                )

                n = len(x)

                msres = (((x - E_leak_before) * g_leak_before - y)**2 / (n - 2)).sum()
                sd = np.sqrt(msres * (1 / n + (x - x.mean())**2 / ((x**2).sum())))
                after_sd = sd[-1]

                before_corrected = before_trace - (g_leak_before * (protocol_voltages - E_leak_before))
                after_corrected = after_trace - (g_leak_after * (protocol_voltages - E_leak_after))

                subtracted_trace = before_corrected - after_corrected

                reversal_ax.cla()
                fitted_E_rev = common.infer_reversal_potential(protocol,
                                                               subtracted_trace,
                                                               observation_times,
                                                               output_path=os.path.join(reversal_plot_dir,
                                                                                        f"reversal_potential_{protocol}_{well}"),
                                                               ax=reversal_ax,
                                                               plot=True)

                subtracted_trace_df = pd.DataFrame(np.column_stack(
                    (observation_times, subtracted_trace)), columns=('time', 'current'))

                fname = f"{experiment_name}-{protocol}-{well}.csv"

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

                for tracename, trace in (('subtracted', subtracted_trace),):
                    estimated_noise = trace[0:200].std()
                    trace = trace[first_step_indices]
                    n = len(trace)
                    if trace.mean() > estimated_noise:
                        print(f"{protocol} {well} {tracename} \tpassed QC6a")
                        passed1 = True
                    else:
                        print(f"{protocol}, {well}, {tracename} \tfailed QC6a")
                        passed1 = False

                    if trace.mean() > 0:
                        print(f"{protocol} {well} {tracename} \tpassed QC6b")
                        passed2 = True
                    else:
                        print(f"{protocol}, {well}, {tracename} \tfailed QC6b")
                        passed2 = False

                    if trace.mean() > -(before_sd + after_sd) * 2:
                        print(f"{protocol}, {well}, {tracename} \tpassed QC6c")
                        passed3 = True
                    else:
                        print(f"{protocol}, {well}, {tracename} \tfailed QC6c")
                        passed3 = False

                    df.append((protocol, well, sweep, tracename, fitted_E_rev, passed1, passed2, passed3))
            subtracted_ax.set_xlabel('time / ms')
            subtracted_ax.set_ylabel('current / pA')
            subtracted_ax.set_title('subtracted traces')
            subtracted_ax.legend(loc=1)

            long_protocol_ax.plot(observation_times, protocol_voltages)
            long_protocol_ax.set_title("Voltage protocol")
            long_protocol_ax.set_ylabel("Voltage /mV")
            long_protocol_ax.set_xlabel("time /ms")

            fig.tight_layout()
            fig.savefig(os.path.join(output, f"{well}_{protocol}_traces_from_leak_subtraction"))

    df = pd.DataFrame(df, columns=('protocol', 'well', 'sweep', 'before/after', 'fitted_E_rev',
                                   'passed QC6a', 'passed QC6b', 'passed QC6c'))
    df.to_csv(os.path.join(output, "subtraction_qc.csv"))
    print(df)

    # Find wells passing all traces
    passed_lst = []
    for well in np.unique(df['well'].values):
        failed = False
        for _, row in df.iterrows():
            if row['well'] != well:
                continue
            elif row['passed QC6c'] is False:
                failed = True
                break
            elif not np.isfinite(row['fitted_E_rev']):
                failed = True
                break
        if not failed:
            passed_lst.append(well)

    print(f"Wells with all successful traces: {passed_lst}")

    with open(os.path.join(output, "passed_wells.txt"), 'w') as fout:
        for well in passed_lst:
            fout.writeline(well)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.subplots()
    erev_dir = os.path.join(output, "erev_plots")

    if not os.path.exists(erev_dir):
        os.makedirs(erev_dir)

    df['xlocs'] = df['protocol'].map(lambda x: protocol_chrono_order.index(x))
    # Plot reversal potentials over time for each well
    for well in np.unique(df['well'].values):
        df_to_plot = df[df.well == well]
        df_to_plot = df_to_plot.set_index('xlocs').sort_index()
        print(df_to_plot)
        df_to_plot.plot('protocol', 'fitted_E_rev', ax=ax)
        labels = df_to_plot['protocol'].values
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel('Nernst potential (fitted) / mV')
        ax.legend()
        ax.axhline(Erev, ls="--", label='Nernst potential (calculated from concentrations) / mV')
        fig.savefig(os.path.join(erev_dir, f"{well}_erevs"))
        ax.cla()

if __name__ == "__main__":
    main()
