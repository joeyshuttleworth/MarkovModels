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
    default_protocol_list = ["sis", "longap", "rtovmaxdiff", "rvotmaxdiff", "spacefill10", "spacefill19", "spacefill26", "hhsobol3step", "hhbrute3gstep", "wangsobol3step", "wangbrute3gstep"]
    parser.add_argument('--protocols', action='append', default=default_protocol_list)
    parser.add_argument('--percentage_to_remove', default=100, type=float)

    args = parser.parse_args()

    output_dir = os.path.join(args.output, f"leak_fitting_{args.percentage_to_remove}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.wells is None:
        args.wells = get_wells_list(args.data_directory)

    for well in args.wells:
        df = []
        for protocol in args.protocols:
            protocol_func, t_start, t_end, t_step = common.get_protocol(protocol)
            observation_times = pd.read_csv(os.path.join(args.data_directory, f"newtonrun4-{protocol}-times.csv")).values.flatten()*1e3
            protocol_voltages = np.array([protocol_func(t) for t in observation_times])

            before_filename = f"newtonrun4-{protocol}-{well}-raw_before.csv"
            after_filename = f"newtonrun4-{protocol}-{well}-raw_after.csv"

            before_trace = pd.read_csv(os.path.join(args.data_directory, before_filename)).values.flatten()
            after_trace = pd.read_csv(os.path.join(args.data_directory, after_filename)).values.flatten()
            df.append([f"{protocol}", "before", well] + list(fit_leak_lr(protocol_voltages, before_trace, dt=5e-4,  percentage_to_remove=args.percentage_to_remove)))
            df.append([f"{protocol}", "after", well] + list(fit_leak_lr(protocol_voltages, after_trace, dt=5e-4, percentage_to_remove=args.percentage_to_remove)))

        df = pd.DataFrame(df, columns=('trace', "E4031", 'well', 'g_leak', 'E_leak', 'r^2'))
        color = df.E4031.map({'before':'b', 'after':'r'})
        df.set_index(['trace', 'E4031'], inplace=True)
        fig, ax = plt.subplots()
        df.plot.scatter('g_leak', 'E_leak', ax=ax, title=well, color=color, marker='x')
        for i, row in df.iterrows():
            text_color = 'red' if i[1] == "after" else 'blue'
            ax.annotate(i[0], (row['g_leak'], row['E_leak']), color=text_color)

        red_x = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                                   markersize=10, label='Red squares')

        blue_x = mlines.Line2D([], [], color='blue', marker='x', linestyle='None',
                                   markersize=10, label='Red squares')
        ax.legend([red_x, blue_x], ['before E-4031', 'after E-4031'])
        plt.savefig(os.path.join(output_dir, f"{well}.pdf"))
        plt.clf()
        fig = plt.figure(figsize=(18,15))
        ax = fig.subplots()
        df['r^2'].plot.bar(ax=ax, color=color)
        plt.savefig(os.path.join(output_dir, f"{well}_r_barplot.png"))
        plt.clf()




if __name__ == "__main__":
    main()
