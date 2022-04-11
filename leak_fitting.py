#!/usr/bin/env python3

from MarkovModels import common
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
from MarkovModels import MarkovModel
from MarkovModels import BeattieModel
from quality_control.leak_fit import fit_leak_lr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import argparse
import regex as re
import itertools


def get_wells_list(input_dir):
    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after-sweep1.csv$")
    wells = []
    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(2)[1]
        if well not in wells:
            wells.append(well)
    return wells


def get_protocol_list(input_dir):
    regex = re.compile(f"^{experiment_name}-([a-z|A-Z|0-9]*)-([A-Z][0-9][0-9])-after-sweep1.csv$")
    protocols = []
    for f in filter(regex.match, os.listdir(input_dir)):
        well = re.search(regex, f).groups(3)[0]
        if protocols not in protocols:
            protocols.append(well)
    return protocols


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--percentage_to_remove', default=0, type=float)
    parser.add_argument('-e', '--extra_points', nargs=2, default=(0, 0), type=int)
    parser.add_argument('--plot', '-p', action="store_true")
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--ramp_start', type=float)
    parser.add_argument('--ramp_end', type=float)

    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    if len(args.protocols) == 0:
        default_protocol_list = ["sis", "longap", "rtovmaxdiff", "rvotmaxdiff", "spacefill10",
                                 "spacefill19", "spacefill26", "hhsobol3step", "hhbrute3gstep", "wangsobol3step", "wangbrute3gstep"]

    output_dir = common.setup_output_directory(args.output, f"leak_fitting_{args.percentage_to_remove:.0f}_{args.extra_points}")

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    if not args.protocols:
        args.protocols = get_protocol_list(args.data_directory)

    fit_fig = plt.figure(figsize=(18, 15), clear=True)
    fit_axs = fit_fig.subplots(4)

    scatter_fig = plt.figure()
    scatter_ax = scatter_fig.subplots()

    for well in np.unique(args.wells):
        df = []
        fits_dir = os.path.join(output_dir, f"{well}_fits")

        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)

        for protocol in np.unique(args.protocols):
            protocol_func, t_start, t_end, t_step = common.get_protocol(protocol)
            observation_times = pd.read_csv(os.path.join(
                args.data_directory, f"{experiment_name}-{protocol}-times.csv")).values.flatten() * 1e3
            dt = (observation_times[1] - observation_times[0])
            print(protocol, dt)
            protocol_voltages = np.array([protocol_func(t) for t in observation_times])

            # Find first few steps where voltage is big
            if args.extra_points[1] > 0:
                extra_steps = np.array(list(itertools.islice(filter(lambda x: x[1] > 20, enumerate(
                    protocol_voltages)), args.extra_points[1])))[args.extra_points[0]:, 0].astype(int)
            else:
                extra_steps = []

            before_filename = f"{experiment_name}-{protocol}-{well}-before-sweep1.csv"
            after_filename = f"{experiment_name}-{protocol}-{well}-after-sweep1.csv"

            try:
                before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
                after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()
            except FileNotFoundError as exc:
                print(str(exc))
                continue

            df.append([f"{protocol}", "before", well] +
                      list(fit_leak_lr(protocol_voltages, before_trace,
                                       dt=dt,
                                       percentage_to_remove=args.percentage_to_remove,
                                       extra_points=extra_steps,
                                       ramp_start=args.ramp_start,
                                       ramp_end=args.ramp_end))[0:5])
            df.append([f"{protocol}", "after", well] +
                      list(fit_leak_lr(protocol_voltages, after_trace,
                                       dt=dt,
                                       percentage_to_remove=args.percentage_to_remove,
                                       extra_points=extra_steps,
                                       ramp_start=args.ramp_start,
                                       ramp_end=args.ramp_end))[0:5])

            g_leak, E_leak, _, s_alpha, s_beta, x, y = fit_leak_lr(
                protocol_voltages, before_trace, dt=dt,
                percentage_to_remove=args.percentage_to_remove,
                extra_points=extra_steps,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end)

            before_leak_current = (protocol_voltages - E_leak) * g_leak
            fit_axs[2].scatter(x, y, marker='s', color='grey', s=2)
            n = len(x)
            xpred = np.linspace(-120, 40, 1000)
            predictions = (xpred - E_leak) * g_leak

            fit_axs[2].plot(xpred, predictions, color='red')

            msres = (((x - E_leak) * g_leak - y)**2 / (n - 2)).sum()
            confidence_region = np.sqrt(msres * (1 / n + (xpred - x.mean())**2 / ((x**2).sum())))

            fit_axs[2].fill_between(xpred, 1.96 * confidence_region + predictions,
                                    -1.96 * confidence_region + predictions,
                                    color='blue',
                                    alpha=0.5)
            before_mean = predictions[-1]
            before_sd = confidence_region[-1]
            before_subtraced = before_trace - (protocol_voltages - E_leak) * g_leak
            print(
                f"{well}, {protocol}: prediction of pre-drug leak current at 40mV: mean = {predictions[-1]}, sd={confidence_region[-1]}")

            fit_axs[2].set_xlabel("voltage mV")
            fit_axs[2].set_ylabel("before current")

            g_leak, E_leak, _, s_alpha, s_beta, x, y = fit_leak_lr(
                protocol_voltages, after_trace, dt=dt,
                percentage_to_remove=args.percentage_to_remove,
                extra_points=extra_steps,
                ramp_start=args.ramp_start,
                ramp_end=args.ramp_end
            )

            predictions = (xpred - E_leak) * g_leak
            msres = (((x - E_leak) * g_leak - y)**2 / (n - 2)).sum()
            confidence_region = np.sqrt(msres * (1 / n + (xpred - x.mean())**2 / ((x**2).sum())))

            after_mean = predictions[-1]
            after_sd = confidence_region[-1]
            after_subtraced = after_trace - (protocol_voltages - E_leak) * g_leak
            confidence_region = np.sqrt(msres * (1 / n + (xpred - x.mean())**2 / ((x**2).sum())))

            # fit_axs[3].fill_between(xpred, 1.96 * confidence_region + predictions,
            #                         -1.96 * confidence_region + predictions,
            #                         color='blue',
            #                         alpha=0.5)

            print(
                f"{well}, {protocol}: prediction of post-drug leak current at 40mV: mean = {predictions[-1]}, sd={confidence_region[-1]}")
            fit_axs[3].scatter(x, y, marker='s', color='grey', s=2)
            fit_axs[3].plot(x, (x - E_leak) * g_leak, color='red')
            fit_axs[3].set_xlabel("voltage mV")
            fit_axs[3].set_ylabel("after current")
            after_leak_current = (protocol_voltages - E_leak) * g_leak

            # 1 second window
            window = list(range(int(1e3 / dt)))
            fit_axs[0].plot(observation_times[window], before_leak_current[window])
            fit_axs[0].plot(observation_times[window], before_trace[window], alpha=.5)
            fit_axs[1].plot(observation_times[window], after_leak_current[window])
            fit_axs[1].plot(observation_times[window], after_trace[window], alpha=.5)

            fit_axs[0].fill_between(np.linspace(args.ramp_start, args.ramp_end,
                                                250),
                                    np.max(before_trace[window]),
                                    np.min(before_trace[window]),
                                    alpha=0.1, color='grey')

            fit_axs[1].fill_between(np.linspace(args.ramp_start, args.ramp_end,
                                                250),
                                    np.max(after_trace[window]),
                                    np.min(after_trace[window]),
                                    alpha=0.1, color='grey')

            fit_axs[0].set_ylabel('before E4031 leak current')
            fit_axs[1].set_ylabel('after E4031 leak current')
            fit_axs[0].set_title(f"{well} {protocol}")

            if args.plot:
                plt.show()
            else:
                fit_fig.savefig(os.path.join(fits_dir, f"{well}_{protocol}_fit.png"))

            for ax in fit_axs:
                ax.cla()
            subtracted_trace = before_trace - after_trace
            print(
                f"{well}, {protocol}: prediction of subtracted trace at 1s: mean = {subtracted_trace[window[-1]]}, sd={before_sd + after_sd}")

        df = pd.DataFrame(df, columns=('trace', "E4031", 'well', 'g_leak', 'E_leak', 'r', 's_alpha', 's_beta'))
        color = df.E4031.map({'before': 'b', 'after': 'r'})
        df.set_index(['trace', 'E4031'], inplace=True)
        df.plot.scatter('g_leak', 'E_leak', ax=scatter_ax, title=well, color=color, marker='x')

        for i, row in df.iterrows():
            text_color = 'red' if i[1] == "after" else 'blue'
            scatter_ax.annotate(i[0], (row['g_leak'], row['E_leak']), color=text_color)

        red_x = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                              markersize=10, label='Red squares')

        blue_x = mlines.Line2D([], [], color='blue', marker='x', linestyle='None',
                               markersize=10, label='Red squares')
        scatter_ax.legend([red_x, blue_x], ['before E-4031', 'after E-4031'])

        if args.plot:
            plt.show()
        else:
            scatter_fig.savefig(os.path.join(output_dir, f"{well}.png"))
        scatter_ax.cla()

        fig2 = plt.figure(figsize=(20, 18), clear=True)
        ax2 = fig2.subplots()
        df['r'].plot.bar(ax=ax2, color=color)
        if args.plot:
            plt.show()
        else:
            fig2.savefig(os.path.join(output_dir, f"{well}_r_barplot.png"))
        ax2.cla()
        plt.close(fig2)


if __name__ == "__main__":
    main()
