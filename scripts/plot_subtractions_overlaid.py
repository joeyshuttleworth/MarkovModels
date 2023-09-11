#!/usr/bin/env python3

from markovmodels import common
import logging
import os
import numpy as np
import pandas as pd
import seaborn as sns
from markovmodels.MarkovModel import MarkovModel
from markovmodels.BeattieModel import BeattieModel
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

import string
import matplotlib

from numba import njit

# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

Erev = common.calculate_reversal_potential(T=298.15, K_in=130, K_out=4)

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
    parser.add_argument('-r', '--reversal', type=float, default=np.nan)
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

    if 'selected' not in leak_parameters_df.columns and args.selection_file:
        with open(args.selection_file) as fin:
            selected_wells = fin.read().splitlines()
        leak_parameters_df['selected'] = [well in selected_wells for well in leak_parameters_df.well]
    elif args.selection_file:
        selected_wells = [well for well in leak_parameters_df.well.unique() if
                          np.all(leak_parameters_df[leak_parameters_df.well ==
                                                    well]['selected'].values)]
    else:
        raise Exception('no selection file provided and no selected column in dataframe')

    with open(os.path.join(args.data_dir, 'passed_wells.txt')) as fin:
        global passed_wells
        passed_wells = fin.read().splitlines()

    leak_parameters_df['passed QC'] = [
        (well in passed_wells) and (well in selected_wells)
        for well in leak_parameters_df.well
    ]

    # Compute new variables
    leak_parameters_df = compute_leak_magnitude(leak_parameters_df)

    global wells
    wells = leak_parameters_df.well.unique()
    global protocols
    protocols = leak_parameters_df.protocol.unique()

    if args.chrono_file:
        with open(args.chrono_file) as fin:
            lines = fin.read().splitlines()
            protocol_order = [line.split(' ')[0] for line in lines]

        leak_parameters_df['protocol'] = pd.Categorical(leak_parameters_df['protocol'],
                                                        categories=protocol_order,
                                                        ordered=True)
        leak_parameters_df.sort_values('protocol', inplace=True)
        do_chronological_plots(leak_parameters_df)

    if 'passed QC' not in leak_parameters_df.columns and\
       'passed QC6a' in leak_parameters_df.columns:
        leak_parameters_df['passed QC'] = leak_parameters_df['passed QC6a']

    if 'selected' in leak_parameters_df.columns:
        leak_parameters_df['passed QC'] = leak_parameters_df['passed QC'] \
            & leak_parameters_df['selected']

    plot_reversal_spread(leak_parameters_df)
    if np.isfinite(args.reversal):
        plot_spatial_Erev(leak_parameters_df)
    do_scatter_matrix(leak_parameters_df)
    plot_histograms(leak_parameters_df)
    overlay_reversal_plots(leak_parameters_df)
    do_combined_plots(leak_parameters_df)


def compute_leak_magnitude(df, lims=[-120, 60]):
    def compute_magnitude(g, E, lims=lims):
        # RMSE
        lims = np.array(lims)
        evals = lims**3 / 3 - E * lims**2 + E**2 * lims
        return np.abs(g) * np.sqrt(evals[1] - evals[0])

    before_lst = []
    after_lst = []
    for i, row in df.iterrows():
        g_before = row['pre-drug leak conductance']
        E_before = row['pre-drug leak reversal']
        leak_magnitude_before = compute_magnitude(g_before, E_before)
        before_lst.append(leak_magnitude_before)

        g_after = row['post-drug leak conductance']
        E_after = row['post-drug leak reversal']
        leak_magnitude_after = compute_magnitude(g_after, E_after)
        after_lst.append(leak_magnitude_after)

    df['pre-drug leak magnitude'] = before_lst
    df['post-drug leak magnitude'] = after_lst

    return df


def do_chronological_plots(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    sub_dir = os.path.join(output_dir, 'chrono_plots')

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    vars = ['post-drug leak conductance', 'pre-drug leak conductance',
            'post-drug leak reversal', 'R_leftover', 'pre-drug leak reversal',
            'post-drug leak reversal', 'fitted_E_rev', 'pre-drug leak magnitude',
            'post-drug leak magnitude']

    df = leak_parameters_df[leak_parameters_df['selected']]

    for var in vars:
        sns.scatterplot(data=df, x='protocol', y=var, hue='passed QC', ax=ax,
                        hue_order=[False, True])
        sns.lineplot(data=leak_parameters_df, x='protocol', y=var, hue='passed QC', ax=ax, style='well', legend=False)

        if var == 'fitted_E_rev' and np.isfinite(args.reversal):
            ax.axhline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')
        fig.savefig(os.path.join(sub_dir, var.replace(' ', '_')))

        ax.cla()

    plt.close(fig)


def do_combined_plots(leak_parameters_df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    wells = [well for well in leak_parameters_df.well.unique() if well in passed_wells]

    print(f"passed wells are {passed_wells}")

    protocol_overlaid_dir = os.path.join(output_dir, 'overlaid_by_protocol')
    if not os.path.exists(protocol_overlaid_dir):
        os.makedirs(protocol_overlaid_dir)

    palette = sns.color_palette('husl', len(leak_parameters_df.groupby(['well', 'sweep'])))
    for protocol in leak_parameters_df.protocol.unique():
        times_fname = f"{experiment_name}-{protocol}-times.csv"
        try:
            times = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', times_fname))
        except FileNotFoundError:
            continue

        times = times['time'].values.flatten().astype(np.float64)

        reference_current = None

        i = 0
        for sweep in leak_parameters_df.sweep.unique():
            for well in wells:
                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))

                except FileNotFoundError:
                    continue

                current = data['current'].values.flatten().astype(np.float64)

                if reference_current is None:
                    reference_current = current

                scaled_current = scale_to_reference(current, reference_current)
                col = palette[i]
                i += 1
                ax.plot(times, scaled_current, color=col, alpha=.5, label=well)

        fig_fname = f"{protocol}_overlaid_subtracted_traces_scaled"
        fig.suptitle(f"{protocol}: all wells")
        ax.set_xlabel(r'time / ms')
        ax.set_ylabel('current scaled to reference trace')
        ax.legend()
        fig.savefig(os.path.join(protocol_overlaid_dir, fig_fname))
        ax.cla()

    plt.close(fig)

    palette = sns.color_palette('husl',
                                len(leak_parameters_df.groupby(['protocol', 'sweep'])))

    fig2 = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs2 = fig2.subplots(1, 2, sharey=True)

    wells_overlaid_dir = os.path.join(output_dir, 'overlaid_by_well')
    if not os.path.exists(wells_overlaid_dir):
        os.makedirs(wells_overlaid_dir)

    print('overlaying traces by well')

    for well in passed_wells:
        i = 0
        for sweep in leak_parameters_df.sweep.unique():
            for protocol in leak_parameters_df.protocol.unique():
                times_fname = f"{experiment_name}-{protocol}-times.csv"
                times_df = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', times_fname))
                times = times_df['time'].values.flatten().astype(np.float64)

                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))
                except FileNotFoundError:
                    continue

                current = data['current'].values.flatten().astype(np.float64)

                indices_pre_ramp = times < 3000

                col = palette[i]
                i += 1

                label = f"{protocol}_sweep{sweep}"

                axs2[0].plot(times[indices_pre_ramp], current[indices_pre_ramp], color=col, alpha=.5,
                             label=label)

                indices_post_ramp = times > (times[-1] - 2000)
                post_times = times[indices_post_ramp].copy()
                post_times = post_times - post_times[0] + 5000
                axs2[1].plot(post_times, current[indices_post_ramp], color=col, alpha=.5,
                             label=label)

        axs2[0].legend()
        axs2[0].set_title('before drug')
        axs2[0].set_xlabel(r'time / ms')
        axs2[1].set_title('after drug')
        axs2[1].set_xlabel(r'time / ms')

        axs2[0].set_ylabel('current / pA')
        axs2[1].set_ylabel('current / pA')

        fig2_fname = f"{well}_overlaid_subtracted_traces"
        fig2.suptitle(f"Leak ramp comparison: {well}")

        fig2.savefig(os.path.join(wells_overlaid_dir, fig2_fname))
        axs2[0].cla()
        axs2[1].cla()

    plt.close(fig2)


def do_scatter_matrix(df):
    df = df.drop([df.columns[0], 'sweep', 'passed QC.Erev',
                  'selected', 'pre-drug leak reversal', 'post-drug leak reversal'],
                 axis='columns')

    grid = sns.pairplot(data=df, hue='passed QC', diag_kind='hist',
                        plot_kws={'alpha': 0.4, 'edgecolor': None},
                        hue_order=[False, True])
    grid.savefig(os.path.join(output_dir, 'scatter_matrix_by_QC'))

    if args.reversal:
        true_reversal = args.reversal
    else:
        true_reversal = df['fitted_E_rev'].values.mean()

    df['hue'] = df['fitted_E_rev'] > true_reversal
    grid = sns.pairplot(data=df, hue='hue', diag_kind='hist',
                        plot_kws={'alpha': 0.4, 'edgecolor': None},
                        hue_order=[False, True])
    grid.savefig(os.path.join(output_dir, 'scatter_matrix_by_reversal'))


def plot_reversal_spread(df):
    df.fitted_E_rev = df.fitted_E_rev.values.astype(np.float64)

    failed_to_infer = [well for well in df.well.unique() if not
                       np.all(np.isfinite(df[df.well == well]['fitted_E_rev'].values))]

    df = df[~df.well.isin(failed_to_infer)]
    df['passed QC'] = [well in passed_wells for well in df.well]

    def spread_func(x):
        return x.max() - x.min()

    group_df = df[['fitted_E_rev', 'well', 'passed QC']].groupby('well').agg(
        {
            'well': 'first',
            'fitted_E_rev': spread_func,
            'passed QC': 'min'
        })
    group_df['E_Kr range'] = group_df['fitted_E_rev']

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    print(group_df)

    sns.histplot(data=group_df, x='E_Kr range', hue='passed QC', ax=ax,
                 stat='probability')

    ax.set_xlabel(r'spread in inferred E_Kr / mV')

    fig.savefig(os.path.join(output_dir, 'spread_of_fitted_E_Kr'))
    df.to_csv(os.path.join(output_dir, 'spread_of_fitted_E_Kr.csv'))


def plot_spatial_Erev(df):
    fig = plt.figure(figsize=args.figsize)
    ax = fig.subplots()

    def func(protocol, sweep):
        zs = []
        found_value = False
        for row in range(18):
            for column in range(24):
                well = f"{string.ascii_uppercase[row]}{column+1:02d}"
                sub_df = df[(df.protocol == protocol) & (df.sweep == sweep)
                            & (df.well == well)]

                if len(sub_df.index) > 1:
                    Exception("Multiple rows values for same (protocol, sweep, well)"
                              "\n ({protocol}, {sweep}, {well})")
                elif len(sub_df.index) == 0:
                    EKr = np.nan
                else:
                    EKr = sub_df['fitted_E_rev'].values.astype(np.float64)[0]

                found_value = True

                zs.append(EKr)

        zs = np.array(zs).reshape((18, 24))

        if found_value:
            return

        im = ax.pcolormesh(zs, cmap=matplotlib.cm.gray, edgecolors='white',
                           linewidths=1, antialiased=True)
        try:
            fig.colorbar(im)
        except Exception as exc:
            print(str(exc))

        fig.savefig(os.path.join(output_dir, f"{protocol}_sweep{sweep}_E_Kr_map"))

        ax.cla()

        zs = np.array(zs) > args.reversal
        im = ax.pcolormesh(zs, cmap='binary', edgecolors='white',
                           linewidths=1, antialiased=True)

        try:
            fig.colorbar(im)
        except Exception as exc:
            print(str(exc))

    for protocol in df.protocol.unique():
        for sweep in df.sweep.unique():
            func(protocol, sweep)

    print('saving spatial map')
    fig.savefig(os.path.join(output_dir, f"{protocol}_sweep{sweep}_E_Kr_map_binary"))
    plt.close(fig)


def plot_histograms(df):
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()
    sns.histplot(df,
                 x='fitted_E_rev', hue='passed QC', ax=ax,
                 # stat='probability',
                 # common_norm=False
                 )
    if np.isfinite(args.reversal):
        ax.axvline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')
    fig.savefig(os.path.join(output_dir, 'reversal_potential_histogram'))
    ax.cla()

    averaged_fitted_EKr = df.groupby(['well'])['fitted_E_rev'].mean().copy().to_frame()
    averaged_fitted_EKr['passed QC'] = [np.all(df[df.well == well]['passed QC']) for well in averaged_fitted_EKr.index]

    sns.histplot(averaged_fitted_EKr,
                 x='fitted_E_rev', hue='passed QC', ax=ax,
                 # stat='probability',
                 # common_norm=False
                 )
    fig.savefig(os.path.join(output_dir, 'averaged_reversal_potential_histogram'))

    if np.isfinite(args.reversal):
        ax.axvline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')

    fig.savefig(os.path.join(output_dir, 'reversal_potential_histogram'))

    ax.cla()
    sns.histplot(df,
                 x='pre-drug leak magnitude', hue='passed QC', ax=ax,
                 stat='probability', common_norm=False)

    fig.savefig(os.path.join(output_dir, 'pre_drug_leak_magnitude'))
    ax.cla()

    sns.histplot(df,
                 x='post-drug leak magnitude', hue='passed QC', ax=ax,
                 stat='probability', common_norm=False)
    fig.savefig(os.path.join(output_dir, 'post_drug_leak_magnitude'))

    ax.cla()
    sns.histplot(df,
                 x='R_leftover', hue='passed QC', ax=ax,
                 stat='probability', common_norm=False)

    fig.savefig(os.path.join(output_dir, 'R_leftover'))

    plt.close(fig)


def overlay_reversal_plots(leak_parameters_df):
    ''' Method copied from common.infer_reversal_potential'''
    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    ax = fig.subplots()

    palette = sns.color_palette('husl', len(leak_parameters_df.groupby(['protocol', 'sweep'])))

    sub_dir = os.path.join(output_dir, 'overlaid_reversal_plots')

    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

    for well in wells:
        # Setup figure
        if False in leak_parameters_df[leak_parameters_df.well == well]['passed QC'].values:
            continue
        i = 0
        for protocol in protocols:
            if protocol == np.nan:
                continue
            for sweep in [1, 2]:
                protocol_func, _, protocol_desc = common.get_ramp_protocol_from_csv(protocol)
                fname = f"{experiment_name}-{protocol}-{well}-sweep{sweep}.csv"
                try:
                    data = pd.read_csv(os.path.join(args.data_dir, 'subtracted_traces', fname))
                except FileNotFoundError:
                    continue

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
                i += 1

        if np.isfinite(args.reversal):
            ax.axvline(args.reversal, linestyle='--', color='grey', label='Calculated Nernst potential')

        ax.legend()
        # Save figure
        fig.savefig(os.path.join(sub_dir, f"overlaid_reversal_ramps_{well}"))

        # Clear figure
        ax.cla()

    plt.close(fig)
    return


def scale_to_reference(trace, reference):

    @njit
    def error2(p):
        return np.sum((p*trace - reference)**2)

    res = scipy.optimize.minimize_scalar(error2, method='brent')
    return trace * res.x


if __name__ == "__main__":
    main()
