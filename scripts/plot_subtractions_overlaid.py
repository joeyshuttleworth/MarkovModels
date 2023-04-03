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
import scipy
import numba
import uuid
from matplotlib import rc
import multiprocessing
from numba import njit

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

Erev = common.calculate_reversal_potential(T=298.15, K_in=120, K_out=5)

pool_kws = {'maxtasksperchild': 1}

subtracted_trace_dirname = "subtracted_traces"


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


def main():
    description = ""
    parser = argparse.ArgumentParser(description)

    parser.add_argument('data_dir', type=str, help="path to the directory containing the raw data")
    parser.add_argument('--chrono_file', type=str, help="path to file listing the protocols in order")
    parser.add_argument('--cpus', '-c', default=1, type=int)
    parser.add_argument('--wells', '-w', nargs='+', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--protocols', type=str, default=[], nargs='+')
    parser.add_argument('--selection_file', default=None, type=str)
    parser.add_argument('--experiment_name', default='newtonrun4')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 16])

    global args
    args = parser.parse_args()

    global experiment_name
    experiment_name = args.experiment_name

    global output_dir
    output_dir = common.setup_output_directory(args.output, 'plot_subtractions_overlaid')

    leak_parameters_df = pd.read_csv(os.path.join(args.data_dir, 'subtraction_qc.csv'))

    passed_wells = []
    for well in leak_parameters_df.well.unique():
        if False in leak_parameters_df[leak_parameters_df.well == well]['passed QC6a']\
           .values:
            continue
        else:
            passed_wells.append(well)

    leak_parameters_df['passed QC6a'] = leak_parameters_df['passed QC6a'].astype(bool)

    global wells
    wells = leak_parameters_df.well.unique()
    global protocols
    protocols = leak_parameters_df.protocol.unique()
    print(protocols)

    if args.chrono_file:
        with open(args.chrono_file) as fin:
            lines = fin.read().splitlines()
            protocol_order = [line.split(' ')[0] for line in lines]

        leak_parameters_df['protocol'] = pd.Categorical(leak_parameters_df['protocol'],
                                                        categories=protocol_order,
                                                        ordered=True)
        leak_parameters_df.sort_values('protocol', inplace=True)
        do_chronological_plots(leak_parameters_df)

    plot_reversal_spread(leak_parameters_df)
    do_scatter_matrix(leak_parameters_df)
    make_reversal_histogram(leak_parameters_df)
    overlay_reversal_plots(leak_parameters_df)
    do_combined_plots(leak_parameters_df)


def do_chronological_plots(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    sub_dir = os.path.join(output_dir, 'chrono_plots')

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    print(leak_parameters_df.columns)

    vars = ['post-drug leak conductance', 'pre-drug leak conductance', 'leak reversal',
            'leak reversal', 'post-drug leak reversal', 'R_leftover', 'leak reversal', 'fitted_E_rev']
    for var in vars:
        sns.scatterplot(data=leak_parameters_df, x='protocol', y=var, hue='passed QC6a', ax=ax)
        sns.lineplot(data=leak_parameters_df, x='protocol', y=var, hue='passed QC6a', ax=ax, style='well', legend=False)
        fig.savefig(os.path.join(sub_dir, var.replace(' ', '_')))
        ax.cla()


def do_combined_plots(leak_parameters_df):

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    palette = sns.color_palette('husl', len(leak_parameters_df.well.unique()))

    for protocol in leak_parameters_df.protocol.unique():

        times_fname = f"{experiment_name}-{protocol}-times.csv"
        times = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', times_fname))
        times = times['time'].values.flatten().astype(np.float64)

        reference_current = None

        for i, well in enumerate(leak_parameters_df.well.unique()):
            fname = f"{experiment_name}-{protocol}-{well}.csv"
            data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))
            current = data['current'].values.flatten().astype(np.float64)

            if reference_current is None:
                reference_current = current

            scaled_current = scale_to_reference(current, reference_current)
            col = palette[i]
            ax.plot(times, scaled_current, color=col, alpha=.5, label=well)

        fig_fname = f"{protocol}_overlaid_subtracted_traces"
        fig.suptitle(f"{protocol}: all wells")
        ax.set_xlabel(r'time / ms')
        fig.savefig(os.path.join(output_dir, fig_fname))
        ax.cla()

    palette = sns.color_palette('husl', len(leak_parameters_df.protocol.unique()))

    fig2 = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs2 = fig2.subplots(1, 2, sharey=True)

    for well in leak_parameters_df.well.unique():
        for i, protocol in enumerate(leak_parameters_df.protocol.unique()):
            times_fname = f"{experiment_name}-{protocol}-times.csv"
            times_df = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', times_fname))
            times = times_df['time'].values.flatten().astype(np.float64)

            fname = f"{experiment_name}-{protocol}-{well}.csv"
            data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))
            current = data['current'].values.flatten().astype(np.float64)

            indices_pre_ramp = times < 3000
            # if reference_current is None:
            #     reference_current = current
            # scaled_current = scale_to_reference(current, reference_current)

            col = palette[i]
            axs2[0].plot(times[indices_pre_ramp], current[indices_pre_ramp], color=col, alpha=.5,
                         label=protocol)

            indices_post_ramp = times > (times[-1] - 2000)
            post_times = times[indices_post_ramp].copy()
            post_times = post_times - post_times[0] + 5000
            axs2[1].plot(post_times, current[indices_post_ramp], color=col, alpha=.5,
                         label=protocol)

        axs2[0].legend()
        axs2[0].set_title('before protocol')
        axs2[0].set_xlabel(r'time / ms')
        axs2[1].set_title('after protocol')
        axs2[1].set_xlabel(r'time / ms')

        fig2_fname = f"{well}_overlaid_subtracted_traces_pre"
        fig2.suptitle(f"Leak ramp comparison: {well}")
        fig2.savefig(os.path.join(output_dir, fig2_fname))
        axs2[0].cla()
        axs2[1].cla()


def do_scatter_matrix(df):
    df = df.drop(['passed QC6b', 'passed QC6c', 'fitted_E_rev',
                  'sweep', 'passed QC.Erev'],
                 axis='columns')
    grid = sns.pairplot(data=df, hue='passed QC6a')
    grid.savefig(os.path.join(output_dir, 'scatter_matrix'))


def plot_reversal_spread(df):
    passed_wells = [well for well in df.well.unique() if np.all(df[df.well == well]['passed QC6a'].values)]

    df.fitted_E_rev = df.fitted_E_rev.values.astype(np.float64)

    failed_to_infer = [well for well in df.well.unique() if not np.all(np.isfinite(df[df.well==well]['fitted_E_rev'].values))]

    df = df[~df.well.isin(failed_to_infer)]

    df = df.pivot_table(index='well', columns='protocol', values='fitted_E_rev')

    df['E_Kr min'] = df.values.min(axis=1)
    df['E_Kr max'] = df.values.max(axis=1)
    df['E_Kr range'] = df['E_Kr max'] - df['E_Kr min']

    df['passed'] = [well in passed_wells and well not in failed_to_infer for well in df.index]

    print(df.columns)
    print(df)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    sns.histplot(data=df, x='E_Kr range', hue='passed', ax=ax)

    ax.set_xlabel(r'spread in inferred $E_Kr$ / mV')

    fig.savefig(os.path.join(output_dir, 'spread_of_fitted_E_Kr'))


def make_reversal_histogram(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()
    sns.histplot(leak_parameters_df, x='fitted_E_rev', hue='well', ax=ax)
    fig.savefig(os.path.join(output_dir, 'reversal_potential_histogram'))


def overlay_reversal_plots(leak_parameters_df):
    ''' Method copied from common.infer_reversal_potential'''
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    palette = sns.color_palette('husl', len(leak_parameters_df.protocol.unique()))

    sub_dir = os.path.join(output_dir, 'overlaid_reversal_plots')

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    for well in wells:
        # Setup figure
        for i, protocol in enumerate(protocols):
            protocol_func, _, protocol_desc = common.get_ramp_protocol_from_csv(protocol)
            fname = f"{experiment_name}-{protocol}-{well}.csv"
            data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))
            times = data['time'].values.astype(np.float64)

            # First, find the reversal ramp. Search backwards along the protocol until we find a >= 40mV step
            step = next(filter(lambda x: x[2] >= -74, reversed(protocol_desc)))
            step = step[0:2]

            # Next extract steps
            istart = np.argmax(times >= step[0])
            iend = np.argmax(times > step[1])

            if istart == 0 or iend == 0 or istart == iend:
                raise Exception("Couldn't identify reversal ramp")

            # Plot voltage vs current
            current = data['current'].values.astype(np.float64)

            voltages = np.array([protocol_func(t) for t in times])

            col = palette[i]

            ax.scatter(voltages[istart:iend], current[istart:iend], label=protocol,
                       color=col, s=1.2)

            fitted_poly = np.poly1d(np.polyfit(voltages[istart:iend], current[istart:iend], 4))
            ax.plot(voltages[istart:iend], fitted_poly(voltages[istart:iend]), color=col)

        ax.legend()
        # Save figure
        fig.savefig(os.path.join(sub_dir, f"overlaid_reversal_ramps_{well}"))

        # Clear figure
        ax.cla()

    return


def scale_to_reference(trace, reference):

    @njit
    def error2(p):
        return np.sum((p*trace - reference)**2)

    res = scipy.optimize.minimize_scalar(error2, method='brent')

    return trace * res.x


if __name__ == "__main__":
    main()
