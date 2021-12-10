#!/usr/bin/env python3

import common
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
from MarkovModel import MarkovModel
from BeattieModel import BeattieModel
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
    description=""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_directory', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--wells', '-w', action='append', default=None)
    parser.add_argument('--output', '-o', default='output')
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--percentage_to_remove', default=0, type=float)
    # parser.add_argument('--extra_points', default=0, type=int)
    parser.add_argument('--plot', '-p', action="store_true")

    args = parser.parse_args()
    args.extra_points = 0

    if len(args.protocols)==0:
        default_protocol_list = ["sis", "longap", "rtovmaxdiff", "rvotmaxdiff", "spacefill10", "spacefill19", "spacefill26", "hhsobol3step", "hhbrute3gstep", "wangsobol3step", "wangbrute3gstep"]
        args.protocols = default_protocol_list

    output_dir = os.path.join(args.output, f"leak_fitting_{args.percentage_to_remove:.0f}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    print(args.wells, args.protocols)

    fit_fig = plt.figure(figsize = (18, 15), clear=True)
    fit_axs = fit_fig.subplots(4)

    scatter_fig = plt.figure()
    scatter_ax = scatter_fig.subplots()

    for well in args.wells:
        df = []
        fits_dir = os.path.join(output_dir, f"{well}_fits")

        if not os.path.exists(fits_dir):
            os.makedirs(fits_dir)

        for protocol in args.protocols:
            protocol_func, t_start, t_end, t_step = common.get_protocol(protocol)
            observation_times = pd.read_csv(os.path.join(args.data_directory, f"newtonrun4-{protocol}-times.csv")).values.flatten()*1e3
            dt = (observation_times[1] - observation_times[0])*1e-3
            protocol_voltages = np.array([protocol_func(t) for t in observation_times])

            # Find first few steps where voltage is big
            if args.extra_points > 0:
                extra_steps = np.array(list(itertools.islice(filter(lambda x : x[1] > 20, enumerate(protocol_voltages)), args.extra_points)))[:,0].astype(int)
            else:
                extra_steps = []

            before_filename = f"newtonrun4-{protocol}-{well}-raw_before.csv"
            after_filename = f"newtonrun4-{protocol}-{well}-raw_after.csv"

            before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
            after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()
            df.append([f"{protocol}", "before", well] + list(fit_leak_lr(protocol_voltages, before_trace, dt=5e-4,  percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps))[0:5])
            df.append([f"{protocol}", "after", well] + list(fit_leak_lr(protocol_voltages, after_trace, dt=5e-4, percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps))[0:5])

            g_leak, E_leak, _, _, _, x, y = fit_leak_lr(protocol_voltages, before_trace, dt=dt,  percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)
            before_leak_current = (protocol_voltages - E_leak) * g_leak
            fit_axs[2].scatter(x, y, marker='s', color = 'grey', s=2)
            fit_axs[2].plot(x, (x - E_leak) * g_leak, color='red')
            fit_axs[2].set_xlabel("voltage mV")
            fit_axs[2].set_ylabel("before current")
            g_leak, E_leak, _, x, y, _, _ = fit_leak_lr(protocol_voltages, after_trace, dt=dt,  percentage_to_remove=args.percentage_to_remove, extra_points=extra_steps)
            fit_axs[3].scatter(x, y, marker='s', color= 'grey', s=2)
            fit_axs[3].plot(x, (x - E_leak) * g_leak, color='red')
            fit_axs[3].set_xlabel("voltage mV")
            fit_axs[3].set_ylabel("after current")
            after_leak_current = (protocol_voltages - E_leak) * g_leak
            window = list(range(int(1/dt)))
            fit_axs[0].plot(observation_times[window], before_leak_current[window])
            fit_axs[0].plot(observation_times[window], before_trace[window], alpha=.5)
            fit_axs[1].plot(observation_times[window], after_leak_current[window])
            fit_axs[1].plot(observation_times[window], after_trace[window], alpha=.5)
            fit_axs[0].set_ylabel('before E4031 leak current')
            fit_axs[1].set_ylabel('after E4031 leak current')
            fit_axs[0].set_title(f"{well} {protocol}")

            if args.plot:
                plt.show()
            else:
                plt.savefig(os.path.join(fits_dir, f"{well}_{protocol}_fit.png"))
            for ax in fit_axs:
                ax.cla()

        df = pd.DataFrame(df, columns=('trace', "E4031", 'well', 'g_leak', 'E_leak', 'r', 's_alpha', 's_beta'))
        color = df.E4031.map({'before':'b', 'after':'r'})
        df.set_index(['trace', 'E4031'], inplace=True)
        df.plot.scatter('g_leak', 'E_leak', ax=scatter_ax, title=well, color=color, marker='x')
        plt.close(scatter_fig)

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

        fig2 = plt.figure(clear=True)
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
