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


def get_wells_list(input_dir):
    regex = re.compile("^newtonrun4-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-raw_after.csv$")
    wells = []
    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(2)[1]
        if well not in wells:
            wells.append(well)
    return wells


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--wells', '-w', action='append', default=None)
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--percentage_to_remove', default=0, type=float)
    parser.add_argument('-e', '--extra_points', nargs=2, default=(0, 0), type=int)

    args = parser.parse_args()

    if len(args.protocols) == 0:
        default_protocol_list = ["sis", "longap", "rtovmaxdiff", "rvotmaxdiff", "spacefill10",
                                 "spacefill19", "spacefill26", "hhsobol3step", "hhbrute3gstep", "wangsobol3step", "wangbrute3gstep"]
        args.protocols = default_protocol_list

    output_dir = os.path.join(args.output, f"leak_subtraction_{args.extra_points}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    print(args.wells, args.protocols)

    df = []

    fig = plt.figure(figsize=(20, 16), clear=True)
    axs = fig.subplots(6)

    subtracted_trace_dirname = "subtracted_traces"

    if not os.path.exists(os.path.join(output_dir, subtracted_trace_dirname)):
        os.makedirs(os.path.join(output_dir, subtracted_trace_dirname))

    for well in args.wells:
        for protocol in args.protocols:
            protocol_func, t_start, t_end, t_step = common.get_protocol_from_csv(protocol)
            observation_times = pd.read_csv(os.path.join(
                args.data_directory, f"newtonrun4-{protocol}-times.csv")).values.flatten() * 1e3
            dt = (observation_times[1] - observation_times[0])
            print(f"using {dt*args.extra_points[1]}ms of extra data")
            extra_points = [int(val / dt) for val in args.extra_points]
            protocol_voltages = np.array([protocol_func(t) for t in observation_times])

            # Find first few steps where voltage is big
            if args.extra_points[1] > 0:
                extra_steps = np.array(list(itertools.islice(filter(lambda x: x[1] > 20, enumerate(protocol_voltages)), extra_points[1])))[
                    extra_points[0]:, 0].astype(int)
            else:
                extra_steps = []

            before_filename = f"newtonrun4-{protocol}-{well}-raw_before.csv"
            after_filename = f"newtonrun4-{protocol}-{well}-raw_after.csv"

            before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
            after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()

            g_leak_before, E_leak_before, _, _, _, _, _ = fit_leak_lr(
                protocol_voltages, before_trace, dt=5e-4, percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)
            g_leak_after, E_leak_after, _, _, _, _, _ = fit_leak_lr(
                protocol_voltages, after_trace, dt=5e-4, percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)

            before_subtracted = before_trace - (g_leak_before * (protocol_voltages - E_leak_before))
            after_subtracted = after_trace - (g_leak_after * (protocol_voltages - E_leak_after))

            subtracted_trace = before_subtracted - after_subtracted
            subtracted_trace_df = pd.DataFrame(np.column_stack(
                (observation_times, subtracted_trace)), columns=('time', 'current'))

            fname = f"newtonrun4-{protocol}-{well}.csv"

            subtracted_trace_df.to_csv(os.path.join(output_dir, subtracted_trace_dirname, fname))
            subtracted_trace_df['time'].to_csv(os.path.join(
                output_dir, subtracted_trace_dirname, f"newtonrun4-{protocol}-times.csv"))

            # Check that the current isn't negative on the first step after the leak ramp
            first_step = [(i, v) for i, v in enumerate(protocol_voltages) if v > 30]
            lst = []
            for i, (j, voltage) in enumerate(first_step):
                if j - i > first_step[0][0]:
                    break
                lst.append(j)
            # Ignore first few timesteps
            first_step_indices = lst[10:-10]

            xpred = np.linspace(-120, 40, 1000)

            g_leak, E_leak, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, before_trace, dt=dt, percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)
            n = len(x)
            before_leak_current = (protocol_voltages - E_leak) * g_leak
            predictions = (xpred - E_leak) * g_leak
            msres = (((x - E_leak) * g_leak - y)**2 / (n - 2)).sum()
            sd = np.sqrt(msres * (1 / n + (x - x.mean())**2 / ((x**2).sum())))
            before_mean = predictions[-1]
            before_sd = sd[-1]

            g_leak, E_leak, _, _, _, x, y = fit_leak_lr(
                protocol_voltages, after_trace, dt=dt, percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)
            after_leak_current = (protocol_voltages - E_leak) * g_leak
            before_leak_current = (protocol_voltages - E_leak) * g_leak
            predictions = (xpred - E_leak) * g_leak
            msres = (((x - E_leak) * g_leak - y)**2 / (n - 2)).sum()
            sd = np.sqrt(msres * (1 / n + (x - x.mean())**2 / ((x**2).sum())))
            after_mean = predictions[-1]
            after_sd = sd[-1]

            window = list(range(int(1 / dt)))
            axs[0].plot(observation_times, before_trace)
            axs[0].set_title("Before drug raw trace")
            axs[1].plot(observation_times, after_trace)
            axs[1].set_title("After drug raw trace")
            axs[2].plot(observation_times, before_subtracted)
            axs[2].set_title("Before drug leak_corrected trace")
            axs[3].plot(observation_times, after_subtracted)
            axs[2].set_title("After drug leak_corrected trace")
            axs[4].plot(observation_times, subtracted_trace)
            axs[4].set_title("Subtracted trace")
            axs[5].plot(observation_times, protocol_voltages)
            axs[5].set_title("Voltage protocol")

            for i in range(5):
                axs[i].set_ylabel(f"Current /nA")

            axs[5].set_ylabel(f"Voltage /mV")
            axs[5].set_xlabel("time /ms")

            fig.savefig(os.path.join(output_dir, f"{well}_{protocol}_traces_from_leak_subtraction"))
            for ax in axs:
                ax.cla()

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

                df.append((protocol, well, tracename, passed1, passed2, passed3))

    df = pd.DataFrame(df, columns=('protocol', 'well', 'before/after', 'passed QC6a', 'passed QC6b', 'passed QC6c'))
    df.to_csv(os.path.join(output_dir, "subtraction_qc"))
    print(df)

    # Find wells passing all traces
    passed_lst = []
    for well in np.unique(df['well'].values):
        failed = False
        for _, row in df.iterrows():
            if row['well'] == well and row['passed QC6c'] == False:
                failed = True
                break
        if failed == False:
            passed_lst.append(well)

    print(f"Wells with all successful traces: {passed_lst}")


if __name__ == "__main__":
    main()
